#!/usr/bin/env bash
set -euo pipefail

# 按“条件名”创建数据文件夹并启动双相机采集
# 示例:
#   bash scripts/collect_condition_data.sh \
#     --condition "light_low_obj_left" \
#     --third-serial "<D455_SERIAL>" \
#     --base-dir "$HOME/recorded_data/diverse" \
#     --target 50

BASE_DIR="$HOME/recorded_data/diverse"
CONDITION=""
DEFAULT_THIRD_SERIAL="037522250003"
DEFAULT_WRIST_SERIAL="218622279840"
THIRD_SERIAL="$DEFAULT_THIRD_SERIAL"
WRIST_SERIAL="$DEFAULT_WRIST_SERIAL"
TARGET=50
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --condition)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --condition 缺少参数值" >&2
        exit 1
      fi
      CONDITION="$2"; shift 2 ;;
    --third-serial)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --third-serial 缺少参数值" >&2
        exit 1
      fi
      THIRD_SERIAL="$2"; shift 2 ;;
    --wrist-serial)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --wrist-serial 缺少参数值" >&2
        exit 1
      fi
      WRIST_SERIAL="$2"; shift 2 ;;
    --base-dir)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --base-dir 缺少参数值" >&2
        exit 1
      fi
      BASE_DIR="$2"; shift 2 ;;
    --target)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --target 缺少参数值" >&2
        exit 1
      fi
      TARGET="$2"; shift 2 ;;
    --extra-args)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --extra-args 缺少参数值" >&2
        exit 1
      fi
      EXTRA_ARGS="$2"; shift 2 ;;
    -h|--help)
      sed -n '1,80p' "$0"; exit 0 ;;
    *)
      echo "[ERROR] Unknown arg: $1" >&2
      exit 1 ;;
  esac
done

if [[ -z "$CONDITION" ]]; then
  echo "[ERROR] --condition 不能为空，例如: light_low_obj_left" >&2
  exit 1
fi

# 即使用户传入空字符串，也回退到脚本内置默认序列号。
THIRD_SERIAL="${THIRD_SERIAL:-$DEFAULT_THIRD_SERIAL}"
WRIST_SERIAL="${WRIST_SERIAL:-$DEFAULT_WRIST_SERIAL}"

COND_SAFE="$(echo "$CONDITION" | tr ' ' '_' | tr -cd '[:alnum:]_\-')"
if [[ -z "$COND_SAFE" ]]; then
  echo "[ERROR] 条件名清洗后为空，请使用字母/数字/下划线/横线" >&2
  exit 1
fi

OUT_DIR="$BASE_DIR/$COND_SAFE"
mkdir -p "$OUT_DIR"

COUNT=$(find "$OUT_DIR" -maxdepth 1 -type f -name 'episode_*.hdf5' | wc -l | awk '{print $1}')
LEFT=$(( TARGET - COUNT ))
if (( LEFT < 0 )); then LEFT=0; fi

cat <<MSG
[INFO] 条件名:       $COND_SAFE
[INFO] 输出目录:     $OUT_DIR
[INFO] 已有条数:     $COUNT
[INFO] 目标条数:     $TARGET
[INFO] 还需采集:     $LEFT
[INFO] 采集控制:     键盘 s 开始/停止, y 保存, n 丢弃, q 退出
MSG

if (( LEFT == 0 )); then
  echo "[INFO] 该条件已达到目标条数，可直接退出。"
fi

cd "$(dirname "$0")/.."

CMD=(
  roslaunch carm_deploy dual_camera.launch
  output_dir:="$OUT_DIR"
  wrist_serial:="$WRIST_SERIAL"
  third_serial:="$THIRD_SERIAL"
)

if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=( $EXTRA_ARGS )
  CMD+=("${EXTRA_ARR[@]}")
fi

echo "[INFO] 执行命令: ${CMD[*]}"
exec "${CMD[@]}"
