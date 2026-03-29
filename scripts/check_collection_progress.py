#!/usr/bin/env python3
import argparse
from pathlib import Path


def count_episodes(path: Path) -> int:
    return len(list(path.glob("episode_*.hdf5")))


def main():
    parser = argparse.ArgumentParser(description="统计多条件采集进度")
    parser.add_argument("--base-dir", type=str, default="~/recorded_data/diverse", help="条件数据根目录")
    parser.add_argument("--target", type=int, default=50, help="每个条件目标条数")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).expanduser()
    if not base_dir.exists():
        print(f"[ERROR] 目录不存在: {base_dir}")
        return

    cond_dirs = sorted([p for p in base_dir.iterdir() if p.is_dir()])
    if not cond_dirs:
        print(f"[WARN] 未发现条件文件夹: {base_dir}")
        return

    print(f"[INFO] 根目录: {base_dir}")
    print(f"[INFO] 目标条数: {args.target}\n")

    total = 0
    done = 0
    for cond in cond_dirs:
        c = count_episodes(cond)
        total += c
        left = max(args.target - c, 0)
        ok = "✅" if c >= args.target else "⏳"
        if c >= args.target:
            done += 1
        print(f"{ok} {cond.name:35s}  {c:3d}/{args.target:3d}  剩余 {left:3d}")

    print("\n" + "-" * 70)
    print(f"条件完成: {done}/{len(cond_dirs)}")
    print(f"总 episode: {total}")


if __name__ == "__main__":
    main()
