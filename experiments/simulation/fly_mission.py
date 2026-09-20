"""
experiments/simulation/fly_mission.py

Autonomous Flight Mission Script for ArduPilot SITL:
- Arms the quadcopter in GUIDED mode and takes off to 15m.
- Flies a continuous 4-waypoint patrol box (30m x 30m) at 5 m/s.
- Displays live mission flight telemetry (Speed, Heading, Distance to Waypoint).
- Allows testing FDI, DoS, Replay, and Evil Twin attacks mid-flight!

Bug Fixes vs previous version:
  1. Port auto-detection: tries 14553 first, falls back to 14550 with clear message.
  2. Robust arming: waits for motors_armed() confirmation before issuing takeoff.
  3. Stream requests: explicitly requests LOCAL_POSITION_NED @ 10 Hz and
     GLOBAL_POSITION_INT @ 4 Hz — no more 20s fallback timer at every waypoint.
"""

import sys
import time
import math
import argparse


# ─── Helpers ──────────────────────────────────────────────────────────────────

def request_message_interval(master, mavutil, message_id, frequency_hz):
    """Ask ArduPilot to stream a specific message at the given rate."""
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,
        message_id,
        int(1e6 / frequency_hz),  # interval in microseconds
        0, 0, 0, 0, 0
    )


def wait_until_armed(master, timeout=30):
    """
    Block until the autopilot confirms motors are armed.
    Returns True on success, False on timeout.
    """
    deadline = time.time() + timeout
    print(f"[*] Waiting for arm confirmation (timeout {timeout}s)...")
    while time.time() < deadline:
        msg = master.recv_match(type='HEARTBEAT', blocking=True, timeout=1.0)
        if msg:
            armed = (msg.base_mode & 0x80) != 0  # MAV_MODE_FLAG_SAFETY_ARMED
            if armed:
                return True
        time.sleep(0.1)
    return False


def get_current_altitude(master, samples=30, timeout_per=0.3):
    """
    Read the current AGL altitude from GLOBAL_POSITION_INT.relative_alt only.

    IMPORTANT: VFR_HUD.alt is MSL (barometric) altitude in ArduPilot, NOT AGL.
    Using it gives values like 584m when the SITL origin is at 584m ASL,
    making the script falsely believe the drone is already airborne.
    """
    for _ in range(samples):
        msg = master.recv_match(
            type='GLOBAL_POSITION_INT',
            blocking=True, timeout=timeout_per
        )
        if msg:
            return msg.relative_alt / 1000.0   # mm → m, always AGL
    return 0.0


def connect_mavlink(ports, mavutil):
    """
    Try each port in order and return the first successful connection.
    Prints a clear message about which port was used.
    """
    for port in ports:
        conn_str = f"udpin:0.0.0.0:{port}"
        print(f"[*] Trying MAVLink connection on {conn_str} ...")
        try:
            master = mavutil.mavlink_connection(conn_str)
            master.wait_heartbeat(timeout=4)
            print(f"[+] Connected on port {port}!  "
                  f"sysid={master.target_system}  compid={master.target_component}")
            return master, port
        except Exception:
            print(f"[!] No heartbeat on port {port}, trying next...")
    return None, None


# ─── Main mission ─────────────────────────────────────────────────────────────

def run_mission(connection_str=None, speed=5.0, alt=15.0, laps=0):
    try:
        from pymavlink import mavutil
    except ImportError:
        print("[!] pymavlink not installed. Run: .venv/bin/pip install pymavlink")
        sys.exit(1)

    print("=" * 75)
    print("        AUTONOMOUS QUADCOPTER PATROL MISSION CONTROLLER")
    print("=" * 75)
    print(f"[*] Mission Parameters: Altitude={alt}m | Target Speed={speed} m/s")
    print("[*] Patrol Route: 30m x 30m Survey Box (4 Waypoints Continuous Loop)")
    print("=" * 75)

    # ── 1. Connect ────────────────────────────────────────────────────────────
    if connection_str:
        # Explicit connection string provided (e.g. --connect udpin:0.0.0.0:14550)
        print(f"[*] Connecting to MAVLink endpoint: {connection_str}")
        master = mavutil.mavlink_connection(connection_str)
        print("[*] Waiting for UAV heartbeat...")
        master.wait_heartbeat()
        print(f"[+] Connected!  sysid={master.target_system}  compid={master.target_component}")
    else:
        # Auto-detect: 14553 is the dedicated fly_mission port; fall back to 14550
        # if SITL wasn't started with --out=udp:127.0.0.1:14553.
        master, used_port = connect_mavlink([14553, 14550], mavutil)
        if master is None:
            print()
            print("[!] Could not connect on ports 14553 or 14550.")
            print("[!] Make sure SITL is running with at least one of:")
            print("      --out=udp:127.0.0.1:14553")
            print("      --out=udp:127.0.0.1:14550")
            print("[!] Or pass an explicit endpoint: --connect udpin:0.0.0.0:<port>")
            sys.exit(1)
        if used_port == 14550:
            print("[!] Note: using fallback port 14550 (QGroundControl may compete).")
            print("[!] Add '--out=udp:127.0.0.1:14553' to your SITL command to avoid this.")

    # ── 2. Request telemetry streams ─────────────────────────────────────────
    # Without explicit requests ArduPilot may not stream LOCAL_POSITION_NED,
    # causing the waypoint tracker to rely on a 20-second fallback timer.
    print("[*] Requesting telemetry streams...")
    msg_ids = {
        'LOCAL_POSITION_NED':    (32,  10),   # 10 Hz  — waypoint tracking
        'GLOBAL_POSITION_INT':   (33,   4),   # 4 Hz   — altitude checks
        'VFR_HUD':               (74,   4),   # 4 Hz   — speed/heading HUD
        'ATTITUDE':              (30,  10),   # 10 Hz  — orientation
    }
    for name, (msg_id, hz) in msg_ids.items():
        request_message_interval(master, mavutil, msg_id, hz)
        time.sleep(0.05)
    time.sleep(0.5)   # Let ArduPilot apply the stream rates

    # ── 3. Set patrol speed via DO_CHANGE_SPEED ───────────────────────────────
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
        0,
        1,      # speed type: 1 = groundspeed
        speed,  # m/s
        -1,     # throttle (-1 = no change)
        0, 0, 0, 0
    )

    # ── 4. Arm & Takeoff (if on the ground) ──────────────────────────────────
    current_alt = get_current_altitude(master)
    print(f"[*] Current altitude: {current_alt:.1f} m")

    if current_alt >= 3.0:
        print(f"[+] UAV already airborne ({current_alt:.1f}m). Skipping takeoff.")
        master.set_mode('GUIDED')
        time.sleep(0.5)

    else:
        print("[*] UAV on ground — setting GUIDED mode and arming...")
        master.set_mode('GUIDED')
        time.sleep(1.0)

        # Send arm command
        master.arducopter_arm()

        # ── Wait for arm confirmation (pre-arm checks may take a few seconds) ──
        armed = wait_until_armed(master, timeout=30)
        if not armed:
            print()
            print("[!] Motors did not arm within 30 seconds.")
            print("[!] Check the MAVProxy/SITL console for pre-arm failure messages.")
            print("[!] Common causes: EKF not ready, GPS not locked, bad AHRS.")
            sys.exit(1)
        print("[+] Motors armed!")

        # ── Issue takeoff command ─────────────────────────────────────────────
        master.mav.command_long_send(
            master.target_system, master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,
            0, 0, 0, 0,   # params 1-4 unused for copter
            0, 0,          # lat/lon (0 = current position)
            alt            # target altitude (m)
        )
        print(f"[*] Takeoff command sent. Climbing to {alt}m...")

        # ── Wait for climb ────────────────────────────────────────────────────
        # Give ArduPilot ~1s to start the takeoff before polling altitude
        time.sleep(1.0)
        climb_deadline = time.time() + 60   # 60s hard timeout
        while current_alt < alt * 0.90:
            if time.time() > climb_deadline:
                print("\n[!] Climb timed out (60s). Check SITL / flight controller.")
                sys.exit(1)
            msg = master.recv_match(
                type='GLOBAL_POSITION_INT',   # relative_alt = true AGL
                blocking=True, timeout=1.0
            )
            if msg:
                current_alt = msg.relative_alt / 1000.0
            print(f"  [Climbing] {current_alt:5.1f} m / {alt:.1f} m", end='\r')
            time.sleep(0.1)

    print(f"\n[+] Patrol altitude confirmed: {current_alt:.1f}m. Starting waypoint circuit.")

    # ── 5. Waypoint patrol ────────────────────────────────────────────────────
    waypoints = [
        {"name": "WP 1 (North 30m)",          "n":  30.0, "e":   0.0, "d": -alt},
        {"name": "WP 2 (North 30m, East 30m)", "n":  30.0, "e":  30.0, "d": -alt},
        {"name": "WP 3 (East 30m)",            "n":   0.0, "e":  30.0, "d": -alt},
        {"name": "WP 4 (Home / Origin)",       "n":   0.0, "e":   0.0, "d": -alt},
    ]

    # Position-only type mask (ignore velocity & accel setpoints)
    # Bit field: 0b0000_1111_1111_1000 = 0x0DF8
    type_mask = 0x0DF8

    circuit = 1
    wp_idx  = 0
    LOCAL_POS_TIMEOUT = 30.0   # per-waypoint timeout (s) — raised from 20s

    def send_position_target(target_wp):
        tb = int(time.time() * 1000) & 0xFFFFFFFF
        master.mav.set_position_target_local_ned_send(
            tb,
            master.target_system, master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            type_mask,
            target_wp['n'], target_wp['e'], target_wp['d'],
            0, 0, 0,
            0, 0, 0,
            0, 0
        )

    try:
        while True:
            target_wp   = waypoints[wp_idx]
            print(f"\n[MISSION LAP {circuit}] Routing to {target_wp['name']}...")

            send_position_target(target_wp)

            target_reached = False
            start_wp_time  = time.time()
            last_resend    = start_wp_time

            while not target_reached:
                now = time.time()

                # Re-send setpoint every 3s to keep GUIDED active
                if now - last_resend > 3.0:
                    send_position_target(target_wp)
                    last_resend = now

                msg = master.recv_match(
                    type=['LOCAL_POSITION_NED', 'VFR_HUD'],
                    blocking=True, timeout=0.3
                )

                if msg and msg.get_type() == 'LOCAL_POSITION_NED':
                    cur_n = msg.x
                    cur_e = msg.y
                    cur_d = msg.z
                    dist  = math.sqrt(
                        (target_wp['n'] - cur_n) ** 2 +
                        (target_wp['e'] - cur_e) ** 2
                    )
                    spd = math.sqrt(msg.vx ** 2 + msg.vy ** 2)
                    print(
                        f"  [FLYING] {target_wp['name'][:18]:<18} | "
                        f"Dist: {dist:5.1f}m | Speed: {spd:4.1f} m/s | "
                        f"Alt: {-cur_d:4.1f}m",
                        end='\r'
                    )
                    if dist < 2.5:
                        print(f"\n[+] Waypoint reached: {target_wp['name']}!")
                        target_reached = True
                        time.sleep(1.0)

                elif now - start_wp_time > LOCAL_POS_TIMEOUT:
                    # Fallback: LOCAL_POSITION_NED stream not arriving
                    print(
                        f"\n[!] Waypoint timeout ({LOCAL_POS_TIMEOUT:.0f}s). "
                        "LOCAL_POSITION_NED not streaming — check stream requests."
                    )
                    target_reached = True

            wp_idx += 1
            if wp_idx >= len(waypoints):
                wp_idx = 0
                print(f"\n[+] Patrol circuit {circuit} complete!")
                if laps > 0 and circuit >= laps:
                    print(f"\n[+] All {laps} circuit(s) complete. Commanding RTL...")
                    master.set_mode('RTL')
                    print("[+] UAV returning to launch and landing.")
                    break
                circuit += 1
                print(f"[*] Starting patrol circuit {circuit}...")

    except KeyboardInterrupt:
        print("\n\n[*] Mission stopped by user (Ctrl+C). Commanding RTL...")
        master.set_mode('RTL')
        print("[+] Mode switched to RTL. UAV returning to launch and landing.")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Autonomous UAV Flight Mission",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect port (tries 14553 then 14550):
  .venv/bin/python experiments/simulation/fly_mission.py

  # Explicit port (if SITL only outputs to 14550):
  .venv/bin/python experiments/simulation/fly_mission.py --connect udpin:0.0.0.0:14550

  # 3 laps at 8 m/s, 20m altitude:
  .venv/bin/python experiments/simulation/fly_mission.py --laps 3 --speed 8 --alt 20
        """
    )
    parser.add_argument("--connect", type=str,  default=None,
                        help="Explicit MAVLink connection string (e.g. udpin:0.0.0.0:14550). "
                             "Omit for auto-detect (14553→14550).")
    parser.add_argument("--speed",   type=float, default=5.0,
                        help="Patrol groundspeed in m/s (default: 5.0)")
    parser.add_argument("--alt",     type=float, default=15.0,
                        help="Patrol altitude in metres AGL (default: 15.0)")
    parser.add_argument("--laps",    type=int,   default=0,
                        help="Number of 30×30m patrol laps (0 = fly until Ctrl+C)")
    args = parser.parse_args()

    run_mission(
        connection_str=args.connect,
        speed=args.speed,
        alt=args.alt,
        laps=args.laps
    )
