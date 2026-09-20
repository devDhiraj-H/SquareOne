"""
experiments/simulation/mock_uav_telemetry.py

Standalone Mock UAV Telemetry Generator:
- Streams genuine, physically consistent hovering/cruising MAVLink packets to UDP port 14552.
- Allows testing the live IDS detector and attack injectors immediately even if ArduPilot SITL is not running.
"""

import sys
import time
import math
import argparse

def stream_mock_telemetry(target_port=14552):
    try:
        from pymavlink import mavutil
    except ImportError:
        print("[!] pymavlink not installed. Please run: .venv/bin/pip install pymavlink")
        sys.exit(1)

    print("=" * 70)
    print("STANDALONE MOCK UAV FLIGHT STREAMER")
    print("=" * 70)
    print(f"[*] Streaming authentic nominal quadcopter flight to udp:127.0.0.1:{target_port}")
    print("[*] Press Ctrl+C to stop.")
    print("=" * 70)

    conn = mavutil.mavlink_connection(f"udpout:127.0.0.1:{target_port}", source_system=1, source_component=1)

    t0 = time.time()
    step = 0
    base_alt = 15.0 # meters

    while True:
        now = time.time()
        t = now - t0
        
        # Physically authentic hover with subtle aerodynamic breeze
        alt = base_alt + 0.3 * math.sin(t * 0.4)
        vx = 0.5 * math.cos(t * 0.3)
        vy = 0.3 * math.sin(t * 0.3)
        vz = 0.3 * 0.4 * math.cos(t * 0.4) # exact derivative dh/dt
        
        pitch = math.degrees(math.asin(vx * 0.1 / 9.81))
        roll = math.degrees(-math.asin(vy * 0.1 / 9.81))
        yaw = (t * 2.0) % 360
        
        # 1. Heartbeat
        if step % 10 == 0:
            conn.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_QUADROTOR,
                mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA,
                mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED,
                0, mavutil.mavlink.MAV_STATE_ACTIVE
            )

        # time_boot_ms: MAVLink requires uint32 (0 to 4294967295) representing milliseconds since boot
        time_boot_ms = int(t * 1000) & 0xFFFFFFFF

        # 2. Attitude
        conn.mav.attitude_send(
            time_boot_ms,
            math.radians(roll),
            math.radians(pitch),
            math.radians(yaw),
            0.01, 0.01, 0.02
        )

        # 3. Global Position Int
        conn.mav.global_position_int_send(
            time_boot_ms,
            int(37.7749 * 1e7),
            int(-122.4194 * 1e7),
            int(alt * 1000),
            int(alt * 1000),
            int(vx * 100),
            int(vy * 100),
            int(vz * 100),
            int(yaw * 100)
        )
        
        step += 1
        time.sleep(0.1) # 10 Hz telemetry


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Mock UAV Telemetry Streamer")
    parser.add_argument("--port", type=int, default=14552, help="Target UDP port")
    args = parser.parse_args()

    try:
        stream_mock_telemetry(target_port=args.port)
    except KeyboardInterrupt:
        print("\n[*] Mock telemetry stopped.")
