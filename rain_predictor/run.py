"""
Rain Predictor Addon for Home Assistant
========================================
Runs ONNX LSTM model in real-time on temperature/humidity sensor changes.

Architecture:
  1. On startup: backfill 32-slot buffer from HA history API
  2. Subscribe to state_changed events via WebSocket
  3. On each sensor update: recompute window, run ONNX inference
  4. Write binary_sensor and probability sensor via REST API

Versions: onnxruntime==1.22.0, numpy==2.2.3, aiohttp==3.11.18
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiohttp
import numpy as np
import onnxruntime as ort

# ─── Constants ───────────────────────────────────────────────────────────────

WINDOW_SIZE = 32  # must match training config
INTERVAL_SECONDS = 15 * 60  # 15 minutes

MODEL_PATH = Path(__file__).parent / "rain_model.onnx"
SCALER_PATH = Path(__file__).parent / "scaler_params.json"
OPTIONS_PATH = Path("/data/options.json")

# HA Supervisor API (available inside addons)
SUPERVISOR_URL = "http://supervisor"
SUPERVISOR_TOKEN = os.environ.get("SUPERVISOR_TOKEN", "")

# Output entity IDs
BINARY_SENSOR_ID = "binary_sensor.neural_network_rain_prediction"
PROBABILITY_SENSOR_ID = "sensor.neural_network_rain_probability"

log = logging.getLogger("rain_predictor")


# ─── Config ──────────────────────────────────────────────────────────────────


def load_config():
    """Load addon options from /data/options.json."""
    if OPTIONS_PATH.exists():
        with open(OPTIONS_PATH) as f:
            return json.load(f)
    # Fallback for local testing
    return {
        "temperature_entity": "sensor.outside_t_and_h_temperature",
        "humidity_entity": "sensor.outside_t_and_h_humidity",
        "threshold": 0.89,
        "debounce_seconds": 10,
        "log_level": "info",
    }


# ─── Model ───────────────────────────────────────────────────────────────────


class RainPredictor:
    """ONNX-based rain predictor with StandardScaler normalization."""

    def __init__(self):
        # Load scaler params
        with open(SCALER_PATH) as f:
            params = json.load(f)
        self.mean = np.array(params["mean"], dtype=np.float32)
        self.scale = np.array(params["scale"], dtype=np.float32)
        self.threshold = params.get("threshold", 0.89)

        # Load ONNX model
        self.session = ort.InferenceSession(
            str(MODEL_PATH),
            providers=["CPUExecutionProvider"],
        )
        log.info("Model loaded: %s", MODEL_PATH.name)
        log.info("Scaler: mean=%s, scale=%s", self.mean.tolist(), self.scale.tolist())

    def predict(self, window: np.ndarray, threshold: float | None = None) -> dict:
        """
        Run inference on a (WINDOW_SIZE, 2) array of [temperature, humidity].

        Returns dict with 'is_raining', 'probability', 'threshold'.
        """
        if threshold is None:
            threshold = self.threshold

        # Normalize
        normalized = (window - self.mean) / self.scale
        input_tensor = normalized.reshape(1, WINDOW_SIZE, 2).astype(np.float32)

        # Run ONNX
        logit = self.session.run(None, {"input": input_tensor})[0].item()
        probability = 1.0 / (1.0 + np.exp(-logit))  # sigmoid

        return {
            "is_raining": bool(probability >= threshold),
            "probability": round(float(probability), 4),
            "threshold": threshold,
        }


# ─── History Buffer ──────────────────────────────────────────────────────────


class SensorBuffer:
    """
    Maintains timestamped sensor readings and builds the 32-slot window.

    On each prediction request, computes 32 target timestamps going back
    from the current time at 15-minute intervals, and finds the closest
    reading for each slot.
    """

    def __init__(self):
        # List of (timestamp, temperature, humidity) sorted by time
        self.readings: list[tuple[float, float, float]] = []

    def add_reading(self, timestamp: float, temperature: float, humidity: float):
        """Add a single reading. Keeps readings sorted by timestamp."""
        self.readings.append((timestamp, temperature, humidity))
        # Keep only last 24 hours to avoid unbounded growth
        cutoff = time.time() - 24 * 3600
        self.readings = [(t, temp, hum) for t, temp, hum in self.readings if t > cutoff]

    def build_window(self, now: float | None = None) -> np.ndarray | None:
        """
        Build a (WINDOW_SIZE, 2) array for the model.

        Computes target timestamps: now, now-15m, now-30m, ..., now-7h45m
        For each target, finds the closest reading within ±10 minutes.
        Returns None if we don't have enough data.
        """
        if not self.readings:
            return None

        if now is None:
            now = time.time()

        # Target timestamps: current, -15m, -30m, ..., -(WINDOW_SIZE-1)*15m
        targets = [now - i * INTERVAL_SECONDS for i in range(WINDOW_SIZE)]
        targets.reverse()  # oldest first

        window = np.zeros((WINDOW_SIZE, 2), dtype=np.float32)
        max_gap = INTERVAL_SECONDS * 0.75  # ±11.25 minutes tolerance

        for i, target in enumerate(targets):
            # Find closest reading to this target
            best = None
            best_dist = float("inf")
            for ts, temp, hum in self.readings:
                dist = abs(ts - target)
                if dist < best_dist:
                    best_dist = dist
                    best = (temp, hum)

            if best is None or best_dist > max_gap:
                # For the most recent slot, allow wider tolerance (use latest reading)
                if i == WINDOW_SIZE - 1 and self.readings:
                    latest = self.readings[-1]
                    window[i] = [latest[1], latest[2]]
                else:
                    log.debug("Gap at slot %d (target=%s, best_dist=%.0fs)",
                              i, datetime.fromtimestamp(target).strftime("%H:%M"), best_dist)
                    # Use nearest available or interpolate from neighbors
                    if best is not None:
                        window[i] = [best[0], best[1]]
                    elif i > 0:
                        window[i] = window[i - 1]  # carry forward
                    else:
                        return None  # can't fill first slot
            else:
                window[i] = [best[0], best[1]]

        return window

    @property
    def count(self):
        return len(self.readings)


# ─── HA API Helpers ──────────────────────────────────────────────────────────


def api_headers():
    return {
        "Authorization": f"Bearer {SUPERVISOR_TOKEN}",
        "Content-Type": "application/json",
    }


async def get_sensor_state(session: aiohttp.ClientSession, entity_id: str) -> float | None:
    """Get current numeric state of a sensor entity."""
    url = f"{SUPERVISOR_URL}/core/api/states/{entity_id}"
    try:
        async with session.get(url, headers=api_headers()) as resp:
            if resp.status == 200:
                data = await resp.json()
                state = data.get("state")
                if state and state not in ("unknown", "unavailable"):
                    return float(state)
    except Exception as e:
        log.warning("Failed to get state for %s: %s", entity_id, e)
    return None


async def get_sensor_history(
    session: aiohttp.ClientSession, entity_id: str, hours: int = 9
) -> list[tuple[float, float]]:
    """
    Get historical readings for a sensor from HA's recorder.

    Returns list of (unix_timestamp, value) sorted by time.
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)
    start_str = start.strftime("%Y-%m-%dT%H:%M:%S+00:00")

    url = (
        f"{SUPERVISOR_URL}/core/api/history/period/{start_str}"
        f"?filter_entity_id={entity_id}&minimal_response&no_attributes"
    )

    try:
        async with session.get(url, headers=api_headers()) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data and len(data) > 0:
                    readings = []
                    for entry in data[0]:
                        state = entry.get("s", entry.get("state"))
                        last_changed = entry.get("lu", entry.get("last_changed"))
                        if state and state not in ("unknown", "unavailable"):
                            try:
                                val = float(state)
                                if isinstance(last_changed, (int, float)):
                                    ts = last_changed
                                else:
                                    ts = datetime.fromisoformat(
                                        str(last_changed)
                                    ).timestamp()
                                readings.append((ts, val))
                            except (ValueError, TypeError):
                                continue
                    return sorted(readings, key=lambda x: x[0])
    except Exception as e:
        log.warning("Failed to get history for %s: %s", entity_id, e)
    return []


async def set_entity_state(
    session: aiohttp.ClientSession, entity_id: str, state: str, attributes: dict
):
    """Set (or create) an entity state via the HA REST API."""
    url = f"{SUPERVISOR_URL}/core/api/states/{entity_id}"
    payload = {"state": state, "attributes": attributes}
    try:
        async with session.post(url, headers=api_headers(), json=payload) as resp:
            if resp.status not in (200, 201):
                text = await resp.text()
                log.warning("Failed to set %s: %d %s", entity_id, resp.status, text)
    except Exception as e:
        log.warning("Failed to set %s: %s", entity_id, e)


# ─── Main Loop ───────────────────────────────────────────────────────────────


async def backfill_buffer(
    session: aiohttp.ClientSession,
    buffer: SensorBuffer,
    temp_entity: str,
    humid_entity: str,
):
    """Populate buffer from HA history on startup."""
    log.info("Backfilling sensor history...")

    temp_history = await get_sensor_history(session, temp_entity)
    humid_history = await get_sensor_history(session, humid_entity)

    log.info("  Temperature: %d readings", len(temp_history))
    log.info("  Humidity: %d readings", len(humid_history))

    if not temp_history or not humid_history:
        log.warning("No history available — predictions will start once enough data accumulates")
        return

    # Merge temperature and humidity by finding closest timestamps
    for t_ts, t_val in temp_history:
        # Find closest humidity reading
        best_h = None
        best_dist = float("inf")
        for h_ts, h_val in humid_history:
            dist = abs(h_ts - t_ts)
            if dist < best_dist:
                best_dist = dist
                best_h = h_val

        if best_h is not None and best_dist < 300:  # within 5 minutes
            buffer.add_reading(t_ts, t_val, best_h)

    log.info("  Buffer populated with %d readings", buffer.count)


async def run_prediction(
    http_session: aiohttp.ClientSession,
    predictor: RainPredictor,
    buffer: SensorBuffer,
    threshold: float,
):
    """Run inference and update HA entities."""
    window = buffer.build_window()
    if window is None:
        log.debug("Not enough data for prediction yet")
        return

    result = predictor.predict(window, threshold)

    log.info(
        "Prediction: probability=%.4f, is_raining=%s (threshold=%.2f)",
        result["probability"],
        result["is_raining"],
        result["threshold"],
    )

    # Update binary sensor
    await set_entity_state(
        http_session,
        BINARY_SENSOR_ID,
        "on" if result["is_raining"] else "off",
        {
            "friendly_name": "Neural Network Rain Prediction",
            "device_class": "moisture",
            "icon": "mdi:weather-rainy" if result["is_raining"] else "mdi:weather-sunny",
            "probability": result["probability"],
            "threshold": result["threshold"],
            "buffer_size": buffer.count,
        },
    )

    # Update probability sensor
    await set_entity_state(
        http_session,
        PROBABILITY_SENSOR_ID,
        str(result["probability"]),
        {
            "friendly_name": "Neural Network Rain Probability",
            "unit_of_measurement": "",
            "icon": "mdi:percent-circle-outline",
            "threshold": result["threshold"],
            "is_raining": result["is_raining"],
            "buffer_size": buffer.count,
        },
    )


async def websocket_listener(
    config: dict,
    predictor: RainPredictor,
    buffer: SensorBuffer,
    http_session: aiohttp.ClientSession,
):
    """
    Connect to HA WebSocket API, subscribe to state changes,
    and trigger predictions on sensor updates.
    """
    temp_entity = config["temperature_entity"]
    humid_entity = config["humidity_entity"]
    threshold = config["threshold"]
    debounce = config["debounce_seconds"]
    watched = {temp_entity, humid_entity}

    ws_url = f"{SUPERVISOR_URL}/core/websocket"
    last_prediction_time = 0.0

    # Track latest known values
    latest_temp = None
    latest_humid = None

    while True:
        try:
            async with http_session.ws_connect(ws_url) as ws:
                # Step 1: Receive auth_required
                msg = await ws.receive_json()
                log.debug("WS: %s", msg.get("type"))

                # Step 2: Authenticate
                await ws.send_json({
                    "type": "auth",
                    "access_token": SUPERVISOR_TOKEN,
                })
                msg = await ws.receive_json()
                if msg.get("type") != "auth_ok":
                    log.error("WS auth failed: %s", msg)
                    await asyncio.sleep(5)
                    continue
                log.info("WebSocket connected and authenticated")

                # Step 3: Subscribe to state_changed events
                await ws.send_json({
                    "id": 1,
                    "type": "subscribe_events",
                    "event_type": "state_changed",
                })
                msg = await ws.receive_json()
                log.debug("Subscribed: %s", msg)

                # Step 4: Listen for events
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        if data.get("type") != "event":
                            continue

                        event = data.get("event", {})
                        event_data = event.get("data", {})
                        entity_id = event_data.get("entity_id")

                        if entity_id not in watched:
                            continue

                        new_state = event_data.get("new_state", {})
                        state_val = new_state.get("state")
                        if state_val in (None, "unknown", "unavailable"):
                            continue

                        try:
                            val = float(state_val)
                        except (ValueError, TypeError):
                            continue

                        # Update latest values
                        if entity_id == temp_entity:
                            latest_temp = val
                        elif entity_id == humid_entity:
                            latest_humid = val

                        # Add reading if we have both
                        if latest_temp is not None and latest_humid is not None:
                            now = time.time()
                            buffer.add_reading(now, latest_temp, latest_humid)

                            # Debounce
                            if now - last_prediction_time >= debounce:
                                await run_prediction(
                                    http_session, predictor, buffer, threshold
                                )
                                last_prediction_time = now

                    elif msg.type in (
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.ERROR,
                    ):
                        log.warning("WebSocket closed/error, reconnecting...")
                        break

        except Exception as e:
            log.error("WebSocket error: %s", e)

        log.info("Reconnecting in 5 seconds...")
        await asyncio.sleep(5)


async def main():
    config = load_config()

    # Configure logging
    log_level = getattr(logging, config.get("log_level", "info").upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    log.info("=" * 50)
    log.info("Rain Predictor Addon starting")
    log.info("=" * 50)
    log.info("Temperature entity: %s", config["temperature_entity"])
    log.info("Humidity entity: %s", config["humidity_entity"])
    log.info("Threshold: %.2f", config["threshold"])
    log.info("Debounce: %ds", config["debounce_seconds"])

    # Load model
    predictor = RainPredictor()

    # Initialize buffer
    buffer = SensorBuffer()

    # HTTP session for API calls
    async with aiohttp.ClientSession() as http_session:
        # Backfill from history
        await backfill_buffer(
            http_session,
            buffer,
            config["temperature_entity"],
            config["humidity_entity"],
        )

        # Run initial prediction if we have data
        if buffer.count > 0:
            await run_prediction(
                http_session, predictor, buffer, config["threshold"]
            )

        # Listen for updates
        await websocket_listener(config, predictor, buffer, http_session)


if __name__ == "__main__":
    asyncio.run(main())
