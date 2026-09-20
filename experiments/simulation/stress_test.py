"""
experiments/simulation/stress_test.py

Automated Stress-Test Orchestrator for the UAV Dual-Layer IDS.

Runs a gauntlet of attack scenarios designed to expose edge cases that
the standard fixed-duration tests miss:

  Scenario 1  — Short-burst DoS         (3s, standard rate)
  Scenario 2  — Full-duration DoS        (15s, standard rate)
  Scenario 3  — High-rate DoS            (15s, 1500 pkts/s — 2x flood)
  Scenario 4  — Short-burst FDI          (3s)
  Scenario 5  — Full-duration FDI        (20s)
  Scenario 6  — Short-burst Replay       (sniff 3s, replay 5s)
  Scenario 7  — Full-duration Replay     (sniff 5s, replay 20s)
  Scenario 8  — Short-burst Evil Twin    (5s)
  Scenario 9  — Full-duration Evil Twin  (20s)
  Scenario 10 — Multi-vector: DoS + FDI simultaneously (10s)
  Scenario 11 — Multi-vector: Replay + Evil Twin simultaneously

Prerequisites (run in separate terminals before this script):
  Terminal 1: SITL running with --out=udp:127.0.0.1:14550 14552 14553
  Terminal 2: fly_mission.py  (drone airborne and patrolling)
  Terminal 3: live_ids_detector.py  (logging to simulation_telemetry_dataset.csv)

Usage:
  .venv/bin/python experiments/simulation/stress_test.py

  # Skip evaluation at the end:
  .venv/bin/python experiments/simulation/stress_test.py --no-eval

  # Dry run — print plan without launching attacks:
  .venv/bin/python experiments/simulation/stress_test.py --dry-run

  # Run only specific scenarios (e.g. multi-vector ones):
  .venv/bin/python experiments/simulation/stress_test.py --scenarios 10,11
"""

import os
import sys
import time
import argparse
import subprocess
from datetime import datetime

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ATTACKS_DIR = os.path.join(SCRIPT_DIR, "attacks")
PYTHON      = os.path.join(SCRIPT_DIR, "..", "..", ".venv", "bin", "python")
EVALUATE_PY = os.path.join(SCRIPT_DIR, "evaluate_simulation_run.py")
LOG_PATH    = os.path.join(SCRIPT_DIR, "stress_test_results.log")

# ANSI colours
R  = "\033[91m"
G  = "\033[92m"
Y  = "\033[93m"
C  = "\033[96m"
M  = "\033[95m"
W  = "\033[97m"
B  = "\033[1m"
RS = "\033[0m"

# ─────────────────────────────────────────────────────────────────────────────
# Scenario definitions
# procs: list of argv lists — run concurrently when >1 (multi-vector)
# rest:  seconds of clean benign flight AFTER this scenario completes
# ─────────────────────────────────────────────────────────────────────────────
SCENARIOS = [
    # ── DoS variants ─────────────────────────────────────────────────────────
    {
        "name":     "Short-Burst DoS (3s)",
        "category": "DoS",
        "procs":    [["attack_dos.py", "--duration", "3"]],
        "rest":     10,
    },
    {
        "name":     "Standard DoS (15s)",
        "category": "DoS",
        "procs":    [["attack_dos.py", "--duration", "15"]],
        "rest":     10,
    },
    {
        "name":     "High-Rate DoS (15s @ 1500 pkts/s)",
        "category": "DoS",
        "procs":    [["attack_dos.py", "--duration", "15", "--rate", "1500"]],
        "rest":     10,
    },
    # ── FDI variants ─────────────────────────────────────────────────────────
    {
        "name":     "Short-Burst FDI (3s)",
        "category": "FDI",
        "procs":    [["attack_fdi.py", "--duration", "3"]],
        "rest":     10,
    },
    {
        "name":     "Full-Duration FDI (20s)",
        "category": "FDI",
        "procs":    [["attack_fdi.py", "--duration", "20"]],
        "rest":     10,
    },
    # ── Replay variants ───────────────────────────────────────────────────────
    {
        "name":     "Short-Burst Replay (sniff 3s, replay 5s)",
        "category": "Replay",
        "procs":    [["attack_replay.py",
                      "--sniff-duration", "3",
                      "--replay-duration", "5"]],
        "rest":     10,
    },
    {
        "name":     "Full-Duration Replay (sniff 5s, replay 20s)",
        "category": "Replay",
        "procs":    [["attack_replay.py",
                      "--sniff-duration", "5",
                      "--replay-duration", "20"]],
        "rest":     10,
    },
    # ── Evil Twin variants ────────────────────────────────────────────────────
    {
        "name":     "Short-Burst Evil Twin (5s)",
        "category": "Evil_Twin",
        "procs":    [["attack_evil_twin.py", "--duration", "5"]],
        "rest":     10,
    },
    {
        "name":     "Full-Duration Evil Twin (20s)",
        "category": "Evil_Twin",
        "procs":    [["attack_evil_twin.py", "--duration", "20"]],
        "rest":     10,
    },
    # ── Multi-vector (simultaneous) ───────────────────────────────────────────
    {
        "name":     "Multi-Vector: DoS + FDI simultaneously (10s)",
        "category": "Multi-Vector",
        "procs":    [
            ["attack_dos.py", "--duration", "10"],
            ["attack_fdi.py", "--duration", "10"],
        ],
        "rest":     15,
    },
    {
        "name":     "Multi-Vector: Replay + Evil Twin simultaneously",
        "category": "Multi-Vector",
        "procs":    [
            ["attack_replay.py",
             "--sniff-duration", "4",
             "--replay-duration", "15"],
            ["attack_evil_twin.py", "--duration", "19"],
        ],
        "rest":     15,
    },
]

# ─────────────────────────────────────────────────────────────────────────────

def log(msg, lf=None):
    print(msg)
    if lf:
        # Strip ANSI codes for the log file
        import re
        clean = re.sub(r'\033\[[0-9;]*m', '', msg)
        lf.write(clean + "\n")
        lf.flush()


def countdown(n, label):
    for i in range(n, 0, -1):
        print(f"  {Y}[COUNTDOWN]{RS} '{label}' starts in {i}s ...", end='\r')
        time.sleep(1)
    print(" " * 70, end='\r')


def run_scenario(scenario, dry_run=False, lf=None):
    name      = scenario["name"]
    procs_def = scenario["procs"]
    rest      = scenario["rest"]

    log(f"\n{B}{C}{'─'*70}{RS}", lf)
    log(f"{B}{C}  ▶  {name}{RS}", lf)
    log(f"{C}{'─'*70}{RS}", lf)

    if dry_run:
        for p in procs_def:
            log(f"  {Y}[DRY-RUN]{RS}  would run: {' '.join(p)}", lf)
        log(f"  {Y}[DRY-RUN]{RS}  rest: {rest}s\n", lf)
        return True

    # Launch all attack processes concurrently
    processes = []
    for argv in procs_def:
        script = os.path.join(ATTACKS_DIR, argv[0])
        cmd    = [PYTHON, script] + argv[1:]
        log(f"  {G}[LAUNCH]{RS}  {' '.join(argv)}", lf)
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes.append((argv[0], proc))
        except Exception as e:
            log(f"  {R}[ERROR]{RS}  Failed to start {argv[0]}: {e}", lf)
            return False

    # Wait for all to finish (120s hard cap each)
    all_ok = True
    for script_name, proc in processes:
        try:
            stdout, _ = proc.communicate(timeout=120)
            rc = proc.returncode
            if rc == 0:
                status = f"{G}OK (rc=0){RS}"
            else:
                status = f"{R}FAILED (rc={rc}){RS}"
                all_ok = False
            log(f"  {status}  [{script_name}]", lf)
            # Show last 3 output lines for quick sanity check
            if stdout:
                for line in stdout.strip().split("\n")[-3:]:
                    log(f"    {W}{line}{RS}", lf)
        except subprocess.TimeoutExpired:
            proc.kill()
            log(f"  {R}[TIMEOUT]{RS}  {script_name} killed after 120s", lf)
            all_ok = False

    log(f"\n  {Y}[REST]{RS}  Benign flight for {rest}s ...", lf)
    time.sleep(rest)
    return all_ok


def print_plan(scenarios):
    cat_col = {
        "DoS":          R,
        "FDI":          Y,
        "Replay":       C,
        "Evil_Twin":    M,
        "Multi-Vector": G,
    }
    total_est = 0
    for s in scenarios:
        for argv in s["procs"]:
            if "--duration" in argv:
                total_est += int(argv[argv.index("--duration") + 1])
            elif "--replay-duration" in argv:
                total_est += (int(argv[argv.index("--replay-duration") + 1]) +
                              int(argv[argv.index("--sniff-duration") + 1]))
        total_est += s["rest"]

    print(f"\n{B}{'='*70}")
    print(f"  UAV IDS STRESS TEST — {len(scenarios)} scenario(s)")
    print(f"{'='*70}{RS}")
    for i, s in enumerate(scenarios, 1):
        cc = cat_col.get(s["category"], W)
        print(f"  {B}{i:>2}.{RS} {cc}{s['category']:<12}{RS}  {s['name']}")
    print(f"\n  {B}Estimated total time:{RS} ~{total_est // 60}m {total_est % 60}s + evaluation")
    print()


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="UAV IDS Stress-Test Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dry-run",   action="store_true",
                        help="Print scenario plan without launching attacks")
    parser.add_argument("--no-eval",   action="store_true",
                        help="Skip evaluate_simulation_run.py at the end")
    parser.add_argument("--warmup",    type=int, default=10,
                        help="Seconds of clean benign flight before first attack (default 10)")
    parser.add_argument("--scenarios", type=str, default=None,
                        help="Comma-separated scenario numbers to run, e.g. '1,3,10'. "
                             "Default: run all.")
    args = parser.parse_args()

    # Resolve scenario list
    if args.scenarios:
        idxs      = [int(x.strip()) - 1 for x in args.scenarios.split(",")]
        scenarios = [SCENARIOS[i] for i in idxs if 0 <= i < len(SCENARIOS)]
        if not scenarios:
            print(f"{R}[!] No valid scenario numbers found in: {args.scenarios}{RS}")
            sys.exit(1)
    else:
        scenarios = SCENARIOS

    print_plan(scenarios)

    if args.dry_run:
        print(f"{Y}[DRY-RUN MODE]{RS} No attacks will be launched.\n")

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{B}Start time:{RS}  {ts}")
    print(f"{B}Log file:  {RS}  {LOG_PATH}\n")

    if not args.dry_run:
        # Quick sanity-check: attack scripts exist?
        missing = [s for s in [
            "attack_dos.py", "attack_fdi.py",
            "attack_replay.py", "attack_evil_twin.py"
        ] if not os.path.exists(os.path.join(ATTACKS_DIR, s))]
        if missing:
            print(f"{R}[!] Missing attack scripts in {ATTACKS_DIR}: {missing}{RS}")
            sys.exit(1)

        print(f"{Y}Before continuing, confirm in your other terminals:{RS}")
        print(f"  1. ArduPilot SITL running with ports 14550, 14552, 14553")
        print(f"  2. fly_mission.py — drone airborne and patrolling")
        print(f"  3. live_ids_detector.py — actively logging telemetry")
        print()
        print(f"{Y}Press Enter to start the stress test, or Ctrl+C to abort...{RS}")
        try:
            input()
        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(0)

    with open(LOG_PATH, "a") as lf:
        log(f"\n{'='*70}", lf)
        log(f"STRESS TEST  {ts}  |  {len(scenarios)} scenario(s)  |  warmup {args.warmup}s", lf)
        log(f"{'='*70}", lf)

        if not args.dry_run and args.warmup > 0:
            log(f"\n{G}[WARMUP]{RS}  {args.warmup}s of benign flight before first attack ...", lf)
            time.sleep(args.warmup)

        passed = failed = 0
        for i, scenario in enumerate(scenarios, 1):
            log(f"\n{B}[{i}/{len(scenarios)}]{RS}", lf)
            if not args.dry_run:
                countdown(3, scenario["name"])
            ok = run_scenario(scenario, dry_run=args.dry_run, lf=lf)
            if ok:
                passed += 1
            else:
                failed += 1

        log(f"\n{B}{'='*70}", lf)
        log(f"  STRESS TEST COMPLETE", lf)
        log(f"{'='*70}{RS}", lf)
        log(f"  Scenarios: {len(scenarios)}  |  {G}Passed: {passed}{RS}  |  "
            f"{(R + f'Failed: {failed}' + RS) if failed else G + 'Failed: 0' + RS}", lf)
        log(f"  Log: {LOG_PATH}", lf)

    if not args.no_eval and not args.dry_run:
        print(f"\n{C}[*] Running evaluator on updated telemetry dataset...{RS}\n")
        try:
            subprocess.run([PYTHON, EVALUATE_PY], check=True)
        except subprocess.CalledProcessError as e:
            print(f"{R}[!] Evaluator failed: {e}{RS}")
    elif args.dry_run:
        print(f"\n{Y}[DRY-RUN]{RS}  Evaluation skipped.")

    print(f"\n{G}[+] Done. Full log at: {LOG_PATH}{RS}\n")


if __name__ == "__main__":
    main()
