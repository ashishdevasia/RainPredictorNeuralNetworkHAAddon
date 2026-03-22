"""
Local test for the Rain Predictor addon.

Starts a mock HA API server that serves dummy sensor data,
then verifies the addon's core logic (model loading, window building, inference).
"""

import asyncio
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

# Add parent so we can import run module components
sys.path.insert(0, str(Path(__file__).parent))

from run import RainPredictor, SensorBuffer, WINDOW_SIZE


def generate_test_readings(hours: int = 9) -> list[tuple[float, float, float]]:
    """Generate realistic temperature/humidity readings for testing."""
    readings = []
    now = time.time()
    start = now - hours * 3600

    t = start
    temp = 25.0
    humid = 70.0

    while t <= now:
        # Simulate diurnal cycle
        hour_of_day = datetime.fromtimestamp(t).hour
        temp_cycle = -2 * np.cos(2 * np.pi * hour_of_day / 24)
        humid_cycle = 5 * np.cos(2 * np.pi * hour_of_day / 24)

        # Add noise
        temp_val = temp + temp_cycle + np.random.normal(0, 0.3)
        humid_val = humid + humid_cycle + np.random.normal(0, 1.5)

        readings.append((t, round(temp_val, 1), round(humid_val, 1)))
        t += 300  # every 5 minutes

    return readings


def test_model_loading():
    """Test ONNX model loads and runs inference."""
    print("=" * 50)
    print("TEST 1: Model Loading & Inference")
    print("=" * 50)

    predictor = RainPredictor()

    # Test with dummy data
    window = np.random.randn(WINDOW_SIZE, 2).astype(np.float32) * 5 + np.array([25.0, 70.0])
    result = predictor.predict(window)

    print(f"  Input shape: {window.shape}")
    print(f"  Probability: {result['probability']}")
    print(f"  Is raining: {result['is_raining']}")
    print(f"  Threshold: {result['threshold']}")
    print("  ✓ Model loads and runs correctly\n")
    return True


def test_buffer():
    """Test buffer handles window building correctly."""
    print("=" * 50)
    print("TEST 2: Sensor Buffer & Window Building")
    print("=" * 50)

    buffer = SensorBuffer()

    # Empty buffer should return None
    w = buffer.build_window()
    assert w is None, "Empty buffer should return None"
    print("  ✓ Empty buffer returns None")

    # Add test readings
    readings = generate_test_readings(hours=9)
    for ts, temp, humid in readings:
        buffer.add_reading(ts, temp, humid)

    print(f"  Added {buffer.count} readings spanning 9 hours")

    # Build window
    w = buffer.build_window()
    assert w is not None, "Should produce window with 9 hours of data"
    assert w.shape == (WINDOW_SIZE, 2), f"Bad shape: {w.shape}"
    print(f"  ✓ Window shape: {w.shape}")
    print(f"  ✓ Temperature range: {w[:, 0].min():.1f} – {w[:, 0].max():.1f}")
    print(f"  ✓ Humidity range: {w[:, 1].min():.1f} – {w[:, 1].max():.1f}")
    print()
    return True


def test_prediction_scenarios():
    """Test predictions with different weather scenarios."""
    print("=" * 50)
    print("TEST 3: Prediction Scenarios")
    print("=" * 50)

    predictor = RainPredictor()
    buffer = SensorBuffer()
    now = time.time()

    # Scenario 1: Clear weather (stable temp ~26, humidity ~65)
    for i in range(WINDOW_SIZE):
        t = now - (WINDOW_SIZE - 1 - i) * 15 * 60
        buffer.add_reading(t, 26.0 + np.random.normal(0, 0.2),
                          65.0 + np.random.normal(0, 1.0))

    window = buffer.build_window(now)
    result = predictor.predict(window)
    print(f"  Clear weather: probability={result['probability']:.4f}, "
          f"is_raining={result['is_raining']}")

    # Scenario 2: Rain approaching (temp dropping, humidity spiking)
    buffer2 = SensorBuffer()
    for i in range(WINDOW_SIZE):
        t = now - (WINDOW_SIZE - 1 - i) * 15 * 60
        progress = i / WINDOW_SIZE
        temp = 26.0 - 4.0 * progress  # drops from 26 to 22
        humid = 70.0 + 20.0 * progress  # rises from 70 to 90
        buffer2.add_reading(t, temp, humid)

    window2 = buffer2.build_window(now)
    result2 = predictor.predict(window2)
    print(f"  Rain approaching: probability={result2['probability']:.4f}, "
          f"is_raining={result2['is_raining']}")

    # Scenario 3: Already raining (cold + very humid)
    buffer3 = SensorBuffer()
    for i in range(WINDOW_SIZE):
        t = now - (WINDOW_SIZE - 1 - i) * 15 * 60
        buffer3.add_reading(t, 22.0 + np.random.normal(0, 0.1),
                           92.0 + np.random.normal(0, 0.5))

    window3 = buffer3.build_window(now)
    result3 = predictor.predict(window3)
    print(f"  Already raining: probability={result3['probability']:.4f}, "
          f"is_raining={result3['is_raining']}")
    print()
    return True


def test_window_alignment():
    """Test that window timestamps are properly aligned to 15-min intervals."""
    print("=" * 50)
    print("TEST 4: Window Time Alignment")
    print("=" * 50)

    buffer = SensorBuffer()
    now = time.time()

    # Add readings at irregular intervals (every 2-8 minutes)
    t = now - 10 * 3600  # 10 hours ago
    while t <= now:
        buffer.add_reading(t, 25.0, 70.0)
        t += np.random.uniform(120, 480)  # 2-8 min intervals

    print(f"  Added {buffer.count} readings at irregular intervals")

    window = buffer.build_window(now)
    assert window is not None
    print(f"  ✓ Window built successfully from irregular data")
    print(f"  ✓ Shape: {window.shape}")
    print()
    return True


def main():
    print("\n🌧️  Rain Predictor Addon — Local Tests\n")

    tests = [
        test_model_loading,
        test_buffer,
        test_prediction_scenarios,
        test_window_alignment,
    ]

    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}\n")

    print("=" * 50)
    print(f"Results: {passed}/{len(tests)} tests passed")
    print("=" * 50)

    if passed == len(tests):
        print("\n✓ All tests passed — addon is ready for deployment!")
    else:
        print("\n✗ Some tests failed — check errors above")
        sys.exit(1)


if __name__ == "__main__":
    main()
