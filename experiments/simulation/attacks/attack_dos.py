"""
experiments/simulation/attacks/attack_dos.py

Simulates Denial-of-Service (DoS) Wi-Fi / MAVLink Packet Flood:
- Floods the MAVLink telemetry channel with high-rate burst packets (100-1000 pkts/sec).
- Squeezes inter-arrival time to ~0.001s and spikes packet rate.
- Triggers Layer 2 Temporal Classifier RED ALERT (DoS Classification).
"""

import os
import time
import json
import socket
import argparse

MARKER_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".active_attack.json"))

def set_attack_marker(attack_name, active=True):
    try:
        with open(MARKER_FILE, "w") as f:
            json.dump({"attack": attack_name if active else "Benign", "active": active, "time": time.time()}, f)
    except Exception:
        pass

def launch_dos_attack(target_ip="127.0.0.1", target_port=14552, duration=15, rate=500):
    print("=" * 70)
    print("ATTACK INJECTION: DENIAL-OF-SERVICE (DoS) PACKET FLOOD")
    print("=" * 70)
    print(f"[*] Target Destination: udp://{target_ip}:{target_port}")
    print(f"[*] Target Flood Rate  : ~{rate} packets/second")
    print(f"[*] Attack Duration    : {duration} seconds")
    print("[*] Expected Response  : Layer 2 Multimodal RED ALERT (DoS Classification)")
    print("=" * 70)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # Pre-crafted MAVLink-like payload buffer
    payload = b"\xfd\x09\x00\x00\x00\x01\x01\x00\x00\x00" + (b"\xaa" * 48)
    
    set_attack_marker("DoS", True)
    try:
        start_time = time.time()
        packet_count = 0
        delay = 1.0 / rate if rate > 0 else 0.002
        
        while time.time() - start_time < duration:
            sock.sendto(payload, (target_ip, target_port))
            packet_count += 1
            
            if packet_count % 100 == 0:
                elapsed = time.time() - start_time
                actual_rate = packet_count / elapsed if elapsed > 0 else 0
                print(f"[DoS FLOODING] t={elapsed:4.1f}s | Sent {packet_count} packets | Current Rate: {actual_rate:5.1f} pkts/s")
                
            time.sleep(delay)
    finally:
        sock.close()
        set_attack_marker("Benign", False)
        print(f"\n[+] DoS Flood Complete. Sent {packet_count} total packets.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DoS Packet Flood Injector")
    parser.add_argument("--port", type=int, default=14552, help="Target MAVLink detector port")
    parser.add_argument("--rate", type=int, default=400, help="Packet flood rate (packets/sec)")
    parser.add_argument("--duration", type=int, default=15, help="Duration in seconds")
    args = parser.parse_args()

    launch_dos_attack(target_port=args.port, duration=args.duration, rate=args.rate)
