#!/usr/bin/env python3
"""
Dual-camera FPS benchmark for carm_deploy.

What it does:
1) Sweep candidate FPS values for both cameras.
2) Launch dual_camera.launch with start_record:=false for each candidate.
3) Measure:
   - wrist color topic FPS
   - third_person color topic FPS
   - approximate-sync paired FPS (with configurable sync_slop)
4) Print and save a report with recommended capture FPS and record_freq.

Usage example:
  python3 scripts/benchmark_dual_camera_fps.py \
    --wrist-serial 218622279840 \
    --third-serial 037522250003 \
    --fps-list 15,30,45,60 \
    --sync-slop 0.05 \
    --record-freq 120 \
    --duration 20
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

try:
    import rospy
    from sensor_msgs.msg import Image
    import message_filters
except Exception as exc:  # pragma: no cover
    print(f"[ERROR] Missing ROS Python deps: {exc}")
    print("Please run in ROS Python env (e.g. source devel/setup.bash).")
    sys.exit(1)


@dataclass
class CaseResult:
    target_fps: int
    wrist_fps: float
    third_fps: float
    sync_fps: float
    sync_ratio: float
    pass_stable: bool
    note: str


class Counter:
    def __init__(self, wrist_topic: str, third_topic: str, sync_slop: float):
        self.wrist_count = 0
        self.third_count = 0
        self.sync_count = 0
        self.active = False

        self._wrist_sub = rospy.Subscriber(wrist_topic, Image, self._on_wrist, queue_size=50)
        self._third_sub = rospy.Subscriber(third_topic, Image, self._on_third, queue_size=50)

        self._wrist_mf = message_filters.Subscriber(wrist_topic, Image)
        self._third_mf = message_filters.Subscriber(third_topic, Image)
        self._ats = message_filters.ApproximateTimeSynchronizer(
            [self._wrist_mf, self._third_mf], queue_size=80, slop=sync_slop
        )
        self._ats.registerCallback(self._on_sync)

    def _on_wrist(self, _msg: Image) -> None:
        if self.active:
            self.wrist_count += 1

    def _on_third(self, _msg: Image) -> None:
        if self.active:
            self.third_count += 1

    def _on_sync(self, _msg1: Image, _msg2: Image) -> None:
        if self.active:
            self.sync_count += 1

    def start(self) -> None:
        self.wrist_count = 0
        self.third_count = 0
        self.sync_count = 0
        self.active = True

    def stop(self) -> None:
        self.active = False

    def close(self) -> None:
        self._wrist_sub.unregister()
        self._third_sub.unregister()
        self._wrist_mf.sub.unregister()
        self._third_mf.sub.unregister()


def wait_for_topic(topic: str, timeout_s: float) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            rospy.wait_for_message(topic, Image, timeout=1.0)
            return True
        except Exception:
            pass
    return False


def launch_case(
    wrist_serial: str,
    third_serial: str,
    fps: int,
    sync_slop: float,
    record_freq: int,
    roslaunch_file: str,
) -> subprocess.Popen:
    cmd = [
        "roslaunch",
        "carm_deploy",
        roslaunch_file,
        f"wrist_serial:={wrist_serial}",
        f"third_serial:={third_serial}",
        "start_record:=false",
        "enable_depth:=false",
        "align_depth:=false",
        f"wrist_color_fps:={fps}",
        f"third_color_fps:={fps}",
        f"record_freq:={record_freq}",
        f"sync_slop:={sync_slop}",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def stop_launch(proc: subprocess.Popen, grace_s: float = 8.0) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    t0 = time.time()
    while time.time() - t0 < grace_s:
        if proc.poll() is not None:
            return
        time.sleep(0.1)
    proc.kill()


def score_case(target_fps: int, wrist_fps: float, third_fps: float, sync_fps: float) -> tuple[bool, float, str]:
    min_single = min(wrist_fps, third_fps)
    single_ok = min_single >= 0.90 * target_fps
    sync_ok = sync_fps >= 0.85 * target_fps
    ratio = 0.0 if min_single <= 1e-6 else sync_fps / min_single
    ratio_ok = ratio >= 0.90
    passed = single_ok and sync_ok and ratio_ok

    note_parts = []
    if not single_ok:
        note_parts.append("single_fps_low")
    if not sync_ok:
        note_parts.append("sync_fps_low")
    if not ratio_ok:
        note_parts.append("sync_ratio_low")
    note = "ok" if passed else ",".join(note_parts)
    return passed, ratio, note


def recommend(results: List[CaseResult]) -> dict:
    stable = [r for r in results if r.pass_stable]
    if stable:
        best = max(stable, key=lambda r: r.target_fps)
        rec_capture = int(max(5, math.floor(best.sync_fps * 0.85)))
        return {
            "mode": "stable_pass",
            "recommended_camera_fps": best.target_fps,
            "recommended_record_freq": rec_capture,
            "recommended_sync_slop": 0.05 if best.sync_ratio < 0.95 else 0.03,
            "reason": f"highest stable target={best.target_fps}, sync_fps={best.sync_fps:.2f}",
        }

    fallback = max(results, key=lambda r: r.sync_fps)
    rec_capture = int(max(5, math.floor(fallback.sync_fps * 0.8)))
    return {
        "mode": "fallback_best_sync",
        "recommended_camera_fps": fallback.target_fps,
        "recommended_record_freq": rec_capture,
        "recommended_sync_slop": 0.08,
        "reason": f"no stable pass, best sync_fps at target={fallback.target_fps}",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark dual-camera FPS and suggest stable params")
    parser.add_argument("--wrist-serial", default="218622279840")
    parser.add_argument("--third-serial", default="037522250003")
    parser.add_argument("--fps-list", default="15,30,45,60", help="comma-separated list")
    parser.add_argument("--sync-slop", type=float, default=0.05)
    parser.add_argument("--record-freq", type=int, default=120, help="high value to avoid recorder bottleneck")
    parser.add_argument("--duration", type=float, default=20.0, help="measure window in seconds")
    parser.add_argument("--settle", type=float, default=6.0, help="warmup seconds before counting")
    parser.add_argument("--topic-timeout", type=float, default=25.0)
    parser.add_argument("--report-dir", default="/home/lizh/maniskill_benchmark/recorded_data/benchmark_reports")
    parser.add_argument("--roslaunch-file", default="dual_camera.launch")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fps_list = [int(x.strip()) for x in args.fps_list.split(",") if x.strip()]
    if not fps_list:
        print("[ERROR] Empty fps-list")
        return 1

    if not rospy.core.is_initialized():
        rospy.init_node("dual_camera_fps_benchmark", anonymous=True, disable_signals=True)

    wrist_topic = "/wrist/color/image_raw"
    third_topic = "/third_person/color/image_raw"

    results: List[CaseResult] = []

    print("\n===== Dual Camera FPS Benchmark =====")
    print(f"wrist_serial={args.wrist_serial}")
    print(f"third_serial={args.third_serial}")
    print(f"fps_list={fps_list}")
    print(f"sync_slop={args.sync_slop}")
    print(f"record_freq(stress)={args.record_freq}")
    print(f"duration={args.duration}s settle={args.settle}s\n")

    for fps in fps_list:
        print(f"[CASE] target_fps={fps}")
        proc = launch_case(
            wrist_serial=args.wrist_serial,
            third_serial=args.third_serial,
            fps=fps,
            sync_slop=args.sync_slop,
            record_freq=args.record_freq,
            roslaunch_file=args.roslaunch_file,
        )

        try:
            ok_wrist = wait_for_topic(wrist_topic, args.topic_timeout)
            ok_third = wait_for_topic(third_topic, args.topic_timeout)
            if not (ok_wrist and ok_third):
                note = "topic_timeout"
                print(f"  [WARN] topic wait failed: wrist={ok_wrist}, third={ok_third}")
                results.append(
                    CaseResult(
                        target_fps=fps,
                        wrist_fps=0.0,
                        third_fps=0.0,
                        sync_fps=0.0,
                        sync_ratio=0.0,
                        pass_stable=False,
                        note=note,
                    )
                )
                continue

            counter = Counter(wrist_topic, third_topic, args.sync_slop)
            time.sleep(args.settle)
            counter.start()
            time.sleep(args.duration)
            counter.stop()

            wrist_fps = counter.wrist_count / args.duration
            third_fps = counter.third_count / args.duration
            sync_fps = counter.sync_count / args.duration
            passed, ratio, note = score_case(fps, wrist_fps, third_fps, sync_fps)
            counter.close()

            print(
                "  "
                f"wrist={wrist_fps:.2f} third={third_fps:.2f} "
                f"sync={sync_fps:.2f} ratio={ratio:.3f} pass={passed} ({note})"
            )

            results.append(
                CaseResult(
                    target_fps=fps,
                    wrist_fps=wrist_fps,
                    third_fps=third_fps,
                    sync_fps=sync_fps,
                    sync_ratio=ratio,
                    pass_stable=passed,
                    note=note,
                )
            )
        finally:
            stop_launch(proc)
            time.sleep(2.0)

    rec = recommend(results)

    print("\n===== Summary =====")
    for r in results:
        print(
            f"target={r.target_fps:>3} | wrist={r.wrist_fps:>6.2f} | "
            f"third={r.third_fps:>6.2f} | sync={r.sync_fps:>6.2f} | "
            f"ratio={r.sync_ratio:>5.3f} | pass={str(r.pass_stable):>5} | {r.note}"
        )

    print("\n===== Recommendation =====")
    print(json.dumps(rec, indent=2, ensure_ascii=False))

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"fps_benchmark_{ts}.json"
    payload = {
        "meta": {
            "wrist_serial": args.wrist_serial,
            "third_serial": args.third_serial,
            "fps_list": fps_list,
            "sync_slop": args.sync_slop,
            "record_freq": args.record_freq,
            "duration": args.duration,
            "settle": args.settle,
        },
        "results": [asdict(r) for r in results],
        "recommendation": rec,
    }
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[INFO] report saved: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
