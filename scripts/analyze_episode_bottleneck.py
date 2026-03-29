#!/usr/bin/env python3
import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_episode_id(episode_path: Path) -> int:
    m = re.search(r"episode_(\d+)_", episode_path.name)
    if not m:
        raise ValueError(f"无法从文件名解析 episode id: {episode_path.name}")
    return int(m.group(1))


def load_episode_steps(timeline_path: Path, episode_id: int) -> List[Dict]:
    rows = []
    with timeline_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("event") == "record_step" and int(item.get("episode", -1)) == episode_id:
                rows.append(item)
    rows.sort(key=lambda x: int(x.get("step", 0)))
    return rows


def choose_timeline(dir_path: Path, episode_id: int, explicit: Optional[Path]) -> Tuple[Path, List[Dict]]:
    if explicit:
        steps = load_episode_steps(explicit, episode_id)
        return explicit, steps

    best_file = None
    best_steps: List[Dict] = []
    for f in sorted(dir_path.glob("timeline_record_*.jsonl")):
        steps = load_episode_steps(f, episode_id)
        if len(steps) > len(best_steps):
            best_steps = steps
            best_file = f

    if best_file is None:
        raise FileNotFoundError(f"目录下未找到 timeline_record_*.jsonl: {dir_path}")
    return best_file, best_steps


def percentile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    arr = sorted(values)
    k = (len(arr) - 1) * q
    f = int(k)
    c = min(f + 1, len(arr) - 1)
    if f == c:
        return arr[f]
    return arr[f] * (c - k) + arr[c] * (k - f)


def describe(name: str, values: List[float]) -> str:
    if not values:
        return f"{name}: no data"
    return (
        f"{name}: mean={statistics.mean(values)*1000:.2f}ms, "
        f"median={statistics.median(values)*1000:.2f}ms, "
        f"p95={percentile(values, 0.95)*1000:.2f}ms, "
        f"max={max(values)*1000:.2f}ms"
    )


def main():
    parser = argparse.ArgumentParser(description="分析单条 episode 的采集吞吐与瓶颈")
    parser.add_argument("episode_path", type=str, help="episode_XXXX_xxx.hdf5 路径")
    parser.add_argument("--timeline", type=str, default="", help="可选：指定 timeline_record_xxx.jsonl")
    args = parser.parse_args()

    episode_path = Path(args.episode_path).expanduser().resolve()
    if not episode_path.exists():
        raise FileNotFoundError(f"文件不存在: {episode_path}")

    episode_id = parse_episode_id(episode_path)
    timeline_arg = Path(args.timeline).expanduser().resolve() if args.timeline else None
    timeline_path, steps = choose_timeline(episode_path.parent, episode_id, timeline_arg)

    if not steps:
        print(f"[WARN] 在 {timeline_path} 中未找到 episode={episode_id} 的 record_step")
        return

    n = len(steps)
    t_sys = [float(x["t_sys"]) for x in steps if x.get("t_sys") is not None]
    obs_stamp = [float(x["obs_stamp_ros"]) for x in steps if x.get("obs_stamp_ros") is not None]

    cam_sync = [float(x["delta_obs"]) for x in steps if x.get("delta_obs") is not None]
    action_query = [
        float(x["t_action_query_sys"]) - float(x["t_obs_ready_sys"])
        for x in steps
        if x.get("t_action_query_sys") is not None and x.get("t_obs_ready_sys") is not None
    ]
    post_query = [
        float(x["t_sys"]) - float(x["t_action_query_sys"])
        for x in steps
        if x.get("t_sys") is not None and x.get("t_action_query_sys") is not None
    ]

    hz_sys = float("nan")
    hz_obs = float("nan")
    if len(t_sys) > 1 and t_sys[-1] > t_sys[0]:
        hz_sys = (len(t_sys) - 1) / (t_sys[-1] - t_sys[0])
    if len(obs_stamp) > 1 and obs_stamp[-1] > obs_stamp[0]:
        hz_obs = (len(obs_stamp) - 1) / (obs_stamp[-1] - obs_stamp[0])

    same_stamp = 0
    for i in range(1, len(obs_stamp)):
        if abs(obs_stamp[i] - obs_stamp[i - 1]) < 1e-9:
            same_stamp += 1
    same_ratio = same_stamp / max(1, len(obs_stamp) - 1)

    mean_cam = statistics.mean(cam_sync) if cam_sync else 0.0
    mean_action = statistics.mean(action_query) if action_query else 0.0
    mean_post = statistics.mean(post_query) if post_query else 0.0

    contributions = {
        "相机同步+图像到达延迟": mean_cam,
        "动作查询开销": mean_action,
        "其余开销(循环调度/日志/Python开销)": mean_post,
    }
    dominant = max(contributions, key=contributions.get)

    print("=" * 80)
    print(f"Episode 文件: {episode_path}")
    print(f"Episode ID: {episode_id}")
    print(f"使用 timeline: {timeline_path}")
    print(f"匹配 record_step 数: {n}")
    print("-" * 80)
    print(f"实际吞吐(基于 t_sys): {hz_sys:.3f} Hz")
    print(f"观测到达频率(基于 obs_stamp_ros): {hz_obs:.3f} Hz")
    print(f"重复 obs_stamp 比例: {same_stamp}/{max(1, len(obs_stamp)-1)} = {same_ratio*100:.2f}%")
    print("-" * 80)
    print(describe("相机同步+图像到达(delta_obs)", cam_sync))
    print(describe("动作查询(t_action_query_sys - t_obs_ready_sys)", action_query))
    print(describe("其余开销(t_sys - t_action_query_sys)", post_query))
    print("-" * 80)
    total = mean_cam + mean_action + mean_post
    if total > 0:
        for k, v in contributions.items():
            print(f"{k} 贡献: {v/total*100:.2f}%")
    print(f"主要瓶颈: {dominant}")

    print("-" * 80)
    print("解释:")
    print("1) 该日志按步记录在内存中，磁盘写入主要发生在 episode 结束保存阶段，通常不是逐步吞吐瓶颈。")
    print("2) 若 delta_obs 明显高于其它项，瓶颈通常在相机流/同步/图像处理链路。")
    print("3) 若动作查询项高，通常是 backend HTTP 查询慢。")
    print("=" * 80)


if __name__ == "__main__":
    main()
