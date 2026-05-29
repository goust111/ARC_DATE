# Combo variant of b5ca7ac4 (index 14)
# combo_spec: D14/0+D17/0
# Name: 取消分组统一堆叠+从内向外生长

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import crop_bbox, get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# concretization_and_trace:
# - 物体仍然是原题那批 5x5 带框方块：框色固定为 teal=8 或 red=2，内部是 3x3 纯色，
#   背景是 22x22 网格里的众数颜色。
# - 融合前，方块先按框色拆成两组，再分别往左右两侧堆叠；并且原题默认从边缘开始，
#   行重叠的后续方块才退到第二列。
# - 融合后，这两层语义一起被改掉：
#   1) D14/0: 不再分青/红两组，所有方块都进入同一个左侧堆叠序列。
#   2) D17/0: 这个单一堆叠不是从边缘起步，而是先占中间偏左的 col 5，行重叠时再向外长到 col 0。
# - 一个具体追踪例子：
#   输入里有三个方块，顶行分别是 r=2(red), r=5(teal), r=12(red)。其中 r=2 与 r=5 的 5 行跨度重叠，
#   r=12 与前两者都不重叠。
#   原题会把 teal 放左边、red 放右边，且默认先占边缘列。
#   本融合版则完全忽略框色，只按顶行从上到下处理：
#     r=2 -> col 5
#     r=5 -> 因与先前 inner-slot 方块重叠，被推到 col 0
#     r=12 -> 与先前 inner-slot 不重叠，仍落在 col 5
#   所以输出会在左半边形成一个“从中间起步，再向左外扩”的统一栈。
#
# complexity_calibration:
# - 若仍允许总数达到 9，则单一左侧双列带状区无法合法容纳：22 行上每列最多放 4 个互不重叠的 5x5 方块，
#   两列合计上限是 8。因此 generator 显式把总数限制在 6-8，保留原题密度但避免物理不可实现。
# - 同时强制 outward 方块至少 2 个，且两列里都出现 red 与 teal，避免样例退化成“只有一两个特殊例外”
#   或“颜色看起来仍在分组”的弱信号。
#
# implicit_assumption_audit:
# - 原 solver 把“框色决定堆到哪一边”当作核心前提；融合后该前提失效，main() 不能再按 frame 拆组。
# - 原 solver 的处理顺序依赖输入列（teal 升序、red 降序）；融合后改成对所有方块统一按顶行升序，
#   因此 generator 必须避免相同顶行造成的不可见 tie。下面会保证所有方块的 r0 两两不同。
# - D17/0 把默认落点从边缘改到中间，因此重叠记录也必须改成“记录已占据 inner-left 的方块”，
#   而不是记录外侧列。
# - main() 仍然只支持两层深度：col 5 的默认层 + col 0 的 outward 层。generator 用“每个 default
#   最多派生一个 outward，且 outward 之间互不重叠”来维持这一假设，不会制造第三层需求。
#
# placement_legality:
# - 输出只使用 col 5 与 col 0 两个 5 宽槽位，互不重叠且都在 22x22 边界内。
# - default_rows 彼此不重叠；outward_rows 彼此也不重叠。每个 outward row 都比它依赖的 default row
#   更靠下 1-4 格，因此既保证了行重叠，又保证了 top-to-bottom 顺序下确实会被视为“后来的重叠者”。
# - 输入阶段再做一次矩形级别的重叠检查，并把所有输入列都放在输出带之外，确保输入散开、输出集中。
#
# visual_observability:
# - 人类从样例里可以直接看到：red 方框不再去右侧，而是和 teal 一起堆到左边；这已经让 D14/0 的
#   “取消分组”十分显眼。
# - 还可以看到左侧多数方块先停在 col 5，只有那些顶行稍低且压到前面方块行带的对象才被顶到 col 0；
#   这给 D17/0 的“从内向外生长”提供了稳定证据。
# - 排序完全依赖可见的几何属性：方块的顶行位置。没有使用颜色编号大小、扫描顺序等人类不可见因素。
#
# visual_cleanliness:
# - 输出始终只占左侧 10 列，结构清晰；同一列内不会出现两个方块抢同一片行带，也不会发生覆盖。
# - 输入仍保持散布状态，不使用额外标记或颜色变化，因此视觉对象类型没有膨胀。
#
# color_and_rule_inferability:
# - 颜色不发生映射，red/teal 仅保留对象身份。generator 强制两种框色都同时出现在 col 5 与 col 0，
#   这样观察者能推断“颜色不再控制目的地”。
# - generator 还强制至少存在一个 cross-color 支撑对：某个 outward 方块与其触发它外扩的 earlier default
#   颜色不同，从而进一步说明重叠规则对所有颜色统一生效。
#
# generator_non_degeneracy_and_solver_checks:
# - 除了 main(input)==output 之外，generator 还保证至少有一个支撑对在输入中满足
#   input_col(default) > input_col(outward)。这意味着若错误地按输入列排序，至少这一对会得到相反的
#   先后关系，从而与本变体的 row-order 规则分离。
# - main() 新增的隐式假设包括：方块顶行唯一、所有 outward 都依赖某个更早且行重叠的 inner-slot 方块、
#   输出深度最多 2、输入中不存在重叠方块。后面的构造和最终验证都逐项保证这些条件。


GRID_H, GRID_W = 22, 22
SQ = 5
AZURE, RED = 8, 2
LEFT_OUTER_COL = 0
LEFT_INNER_COL = SQ

N_AZURE_RANGE = (3, 4)
N_RED_RANGE = (3, 5)
MAX_TOTAL_SQUARES = 8
MIN_OUTWARD = 2
ROW_LAYOUT_RETRIES = 300
FRAME_ASSIGN_RETRIES = 200
INPUT_COL_RETRIES = 300
GEN_RETRIES = 800
INPUT_COL_CHOICES = [c for c in range(GRID_W - SQ + 1) if c not in {LEFT_OUTER_COL, LEFT_INNER_COL}]


def _make_patch(frame_color: int, inner_color: int) -> np.ndarray:
    p = np.full((SQ, SQ), frame_color, dtype=int)
    p[1:-1, 1:-1] = inner_color
    return p


def _rows_overlap(r_a: int, r_b: int) -> bool:
    return abs(r_a - r_b) < SQ


def _rects_overlap(r1: int, c1: int, r2: int, c2: int) -> bool:
    return abs(r1 - r2) < SQ and abs(c1 - c2) < SQ


def _scan_squares(grid: np.ndarray) -> list[dict[str, int | np.ndarray]]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    squares: list[dict[str, int | np.ndarray]] = []
    for r0 in range(H - SQ + 1):
        for c0 in range(W - SQ + 1):
            patch = crop_bbox(g, (int(r0), int(c0), int(r0 + SQ - 1), int(c0 + SQ - 1)))
            border = np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])
            if int(border.min()) != int(border.max()):
                continue
            frame_col = int(border[0])
            if frame_col not in {AZURE, RED}:
                continue
            inner = patch[1:-1, 1:-1]
            if not bool((inner != frame_col).any()):
                continue
            squares.append({"frame": frame_col, "r0": int(r0), "c0": int(c0), "patch": patch})
    return squares


def main(grid: np.ndarray) -> np.ndarray:
    """
    Gather every framed 5x5 square into one left-side stack that starts near the middle and grows outward.

    Input structure:
    - The background is the most common color in a 22x22 grid.
    - Several teal-framed (8) and red-framed (2) 5x5 squares with colored 3x3 interiors are scattered
      across the board without overlap.

    Rule:
    - Ignore frame color when choosing the destination: all detected squares belong to one common stack.
    - Keep each square on its original row and process the whole set from top to bottom using original
      top row ascending; original column is only a tie-breaker.
    - A square whose 5-row span does not overlap any earlier square already placed in the inner-left slot
      lands at col 5.
    - A later square whose row span overlaps an earlier inner-left square is pushed outward to col 0.

    Key insight: Color no longer chooses a side. The output is controlled only by visible row order plus
    overlap against the previously claimed inner-left rows, so the unified stack begins at col 5 and expands leftward.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    out = np.full_like(g, bg)

    def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
        return not (int(a1) < int(b0) or int(b1) < int(a0))

    squares = sorted(_scan_squares(g), key=lambda s: (int(s["r0"]), int(s["c0"])))
    placed_at_inner: list[tuple[int, int]] = []

    for square in squares:
        r0 = int(square["r0"])
        r1 = int(r0 + SQ - 1)
        has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_inner)
        dest_c = LEFT_OUTER_COL if has_overlap else LEFT_INNER_COL
        paste(out, square["patch"], r0, dest_c, transparent=bg)
        if dest_c == LEFT_INNER_COL:
            placed_at_inner.append((r0, r1))

    return out


def _choose_counts() -> tuple[int, int]:
    for _ in range(200):
        n_azure = random.randint(*N_AZURE_RANGE)
        n_red = random.randint(*N_RED_RANGE)
        if n_azure + n_red <= MAX_TOTAL_SQUARES:
            return n_azure, n_red
    raise RuntimeError("Unable to sample realizable frame counts")


def _assign_output_rows(
    n_total: int,
) -> tuple[list[int], list[tuple[int, int]]] | None:
    min_default = max((n_total + 1) // 2, 2)
    max_default = min(4, n_total - MIN_OUTWARD)
    if min_default > max_default:
        return None

    all_rows = list(range(GRID_H - SQ + 1))
    for _ in range(ROW_LAYOUT_RETRIES):
        n_default = random.randint(min_default, max_default)
        n_outward = n_total - n_default

        shuffled_rows = all_rows[:]
        random.shuffle(shuffled_rows)
        default_rows: list[int] = []
        for r in shuffled_rows:
            if all(not _rows_overlap(r, prev) for prev in default_rows):
                default_rows.append(r)
                if len(default_rows) == n_default:
                    break
        if len(default_rows) < n_default:
            continue
        default_rows.sort()

        eligible_supports = [idx for idx, r in enumerate(default_rows) if r < GRID_H - SQ]
        if len(eligible_supports) < n_outward:
            continue
        support_indices = random.sample(eligible_supports, n_outward)
        random.shuffle(support_indices)

        outward_specs: list[tuple[int, int]] = []

        def backtrack(k: int) -> bool:
            if k == len(support_indices):
                return True
            support_idx = support_indices[k]
            support_row = default_rows[support_idx]
            candidates = list(range(support_row + 1, min(support_row + SQ - 1, GRID_H - SQ) + 1))
            random.shuffle(candidates)
            for cand in candidates:
                if cand in default_rows:
                    continue
                if any(_rows_overlap(cand, other_row) for _, other_row in outward_specs):
                    continue
                outward_specs.append((support_idx, cand))
                if backtrack(k + 1):
                    return True
                outward_specs.pop()
            return False

        if backtrack(0):
            return default_rows, outward_specs

    return None


def _assign_frames_to_slots(
    n_azure: int,
    n_red: int,
    default_rows: list[int],
    outward_specs: list[tuple[int, int]],
) -> tuple[list[int], list[int]] | None:
    n_default = len(default_rows)
    low = max(1, n_default - (n_red - 1))
    high = min(n_azure - 1, n_default - 1)
    if low > high:
        return None

    for _ in range(FRAME_ASSIGN_RETRIES):
        n_default_azure = random.randint(low, high)
        default_frames = [AZURE] * n_default_azure + [RED] * (n_default - n_default_azure)
        random.shuffle(default_frames)

        outward_frames = [AZURE] * (n_azure - n_default_azure) + [RED] * (n_red - (n_default - n_default_azure))
        random.shuffle(outward_frames)

        if len(set(default_frames)) < 2 or len(set(outward_frames)) < 2:
            continue
        if not any(default_frames[support_idx] != outward_frames[i] for i, (support_idx, _) in enumerate(outward_specs)):
            continue
        return default_frames, outward_frames

    return None


def _build_square_records(
    bg: int,
    default_rows: list[int],
    outward_specs: list[tuple[int, int]],
    default_frames: list[int],
    outward_frames: list[int],
) -> tuple[list[dict[str, int | np.ndarray]], list[dict[str, int | np.ndarray]], np.ndarray]:
    avail_azure = [c for c in range(10) if c not in {bg, AZURE}]
    avail_red = [c for c in range(10) if c not in {bg, RED}]

    n_azure = default_frames.count(AZURE) + outward_frames.count(AZURE)
    n_red = default_frames.count(RED) + outward_frames.count(RED)
    azure_colors = random.sample(avail_azure, n_azure)
    red_colors = random.sample(avail_red, n_red)
    azure_idx = 0
    red_idx = 0

    output = np.full((GRID_H, GRID_W), bg, dtype=int)
    default_records: list[dict[str, int | np.ndarray]] = []
    outward_records: list[dict[str, int | np.ndarray]] = []

    for idx, r0 in enumerate(default_rows):
        frame = default_frames[idx]
        if frame == AZURE:
            inner = azure_colors[azure_idx]
            azure_idx += 1
        else:
            inner = red_colors[red_idx]
            red_idx += 1
        patch = _make_patch(frame, inner)
        paste(output, patch, r0, LEFT_INNER_COL)
        default_records.append(
            {
                "name": idx,
                "r0": int(r0),
                "frame": int(frame),
                "out_col": LEFT_INNER_COL,
                "patch": patch,
                "kind": 1,
            }
        )

    for idx, (support_idx, r0) in enumerate(outward_specs):
        frame = outward_frames[idx]
        if frame == AZURE:
            inner = azure_colors[azure_idx]
            azure_idx += 1
        else:
            inner = red_colors[red_idx]
            red_idx += 1
        patch = _make_patch(frame, inner)
        paste(output, patch, r0, LEFT_OUTER_COL)
        outward_records.append(
            {
                "name": idx,
                "r0": int(r0),
                "frame": int(frame),
                "out_col": LEFT_OUTER_COL,
                "patch": patch,
                "kind": 0,
                "support": int(support_idx),
            }
        )

    return default_records, outward_records, output


def _assign_input_positions(
    default_records: list[dict[str, int | np.ndarray]],
    outward_records: list[dict[str, int | np.ndarray]],
) -> list[tuple[int, int, np.ndarray]] | None:
    records = default_records + outward_records
    overlap_degree = {
        (int(rec["kind"]), int(rec["name"])): sum(
            1 for other in records
            if other is not rec and _rows_overlap(int(rec["r0"]), int(other["r0"]))
        )
        for rec in records
    }
    ordered_records = sorted(
        records,
        key=lambda rec: (-overlap_degree[(int(rec["kind"]), int(rec["name"]))], int(rec["r0"]), int(rec["kind"])),
    )

    for _ in range(INPUT_COL_RETRIES):
        assigned: dict[tuple[int, int], int] = {}

        def backtrack(k: int) -> bool:
            if k == len(ordered_records):
                return any(
                    assigned[(1, int(default_records[int(outward["support"])]["name"]))] > assigned[(0, int(outward["name"]))]
                    for outward in outward_records
                )

            rec = ordered_records[k]
            key = (int(rec["kind"]), int(rec["name"]))
            candidates = INPUT_COL_CHOICES[:]
            random.shuffle(candidates)
            for cand in candidates:
                if any(
                    _rects_overlap(int(rec["r0"]), cand, int(other["r0"]), assigned[other_key])
                    for other_key, other_col in assigned.items()
                    for other in records
                    if (int(other["kind"]), int(other["name"])) == other_key
                ):
                    continue
                assigned[key] = cand
                if backtrack(k + 1):
                    return True
                del assigned[key]
            return False

        if not backtrack(0):
            continue

        positions: list[tuple[int, int, np.ndarray]] = []
        for rec in records:
            key = (int(rec["kind"]), int(rec["name"]))
            positions.append((int(rec["r0"]), assigned[key], rec["patch"]))
        return positions

    return None


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([c for c in range(10) if c not in {AZURE, RED}])
    n_azure, n_red = _choose_counts()
    n_total = n_azure + n_red

    layout = _assign_output_rows(n_total)
    if layout is None:
        return None
    default_rows, outward_specs = layout

    frame_assign = _assign_frames_to_slots(n_azure, n_red, default_rows, outward_specs)
    if frame_assign is None:
        return None
    default_frames, outward_frames = frame_assign

    default_records, outward_records, output = _build_square_records(
        bg, default_rows, outward_specs, default_frames, outward_frames
    )

    input_positions = _assign_input_positions(default_records, outward_records)
    if input_positions is None:
        return None

    inp = np.full((GRID_H, GRID_W), bg, dtype=int)
    for r0, c0, patch in input_positions:
        paste(inp, patch, r0, c0)

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
