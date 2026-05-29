# Variant of b5ca7ac4
# Dimension: D13 — 符号指令——堆叠方向的编码 (symbol_instructions)
# Change: 无显式符号指令，堆叠方向（青色向左、红色向右）和处理顺序完全靠样例隐式归纳
#   → 在左右边缘加入与组颜色相同的单像素标记；标记在哪一侧，该组就朝哪一侧堆叠。
#
# Design analysis
# ===============
# variant_spec:
#   preserved:
#     - 22x22 画布、5x5 带框方块、框色固定为 8/2、内部仍是 3x3 单色填充
#     - 方块保持原始行位置
#     - 两层堆叠逻辑不变：先占边缘列，若与已占边缘列的同组方块发生纵向重叠，则退到内侧一列
#   changed:
#     - 输入额外包含 2 个控制像素：一个 8 色像素、一个 2 色像素
#     - 控制像素只会出现在左边界或右边界
#     - 某个颜色组的控制像素在左边界 => 该组输出堆到左侧；在右边界 => 堆到右侧
#     - “近目标边先处理”的顺序也随目标边镜像：
#         left  => 按输入列从小到大
#         right => 按输入列从大到小
#     - 控制像素在输出中保留原位
#
# a) concretization_and_trace:
#   - 输入里仍是散落的 5x5 方块，但左右边界各会出现一个孤立单像素标记。
#   - 例如：若 8 色标记在右边，2 色标记在左边，那么青框组就不再去左边，而是按“离右边近的先处理”
#     堆到右侧；红框组则镜像去左侧。原题固定为 “8->左, 2->右”，现在这一步由边缘像素显式给出。
#   - 小例子：两个青框方块列坐标分别是 4 和 13，且它们纵向重叠。
#       原规则：青框固定向左，按列升序处理，col=4 的先占 col 0，col=13 的退到 col 5。
#       新规则（若青色控制像素在右边）：按列降序处理，col=13 的先占 col W-5，col=4 的退到 col W-10。
#     差异就在“目标边”和“谁先占边缘列”都由控制像素决定。
#
# b) complexity_calibration:
#   - 控制信息只有左右两个档位，且每个样例里两个颜色组都会重复使用该规则，复杂度足够但不过载。
#   - 为避免退化成原题，生成器会随机产生 “保持原侧” 与 “左右互换” 两类样例。
#   - 控制像素颜色直接等于对应组的框色，避免再引入额外颜色映射。
#
# c) implicit_assumption_audit_of_original_main:
#   - 原代码把颜色 8 固定送到左边、颜色 2 固定送到右边；这里必须改成先读边缘标记再决定侧向。
#   - 原代码把排序方向写死为 azure 升序 / red 降序；这里必须改成“向左则升序、向右则降序”。
#   - 原代码默认边界上的 2/8 像素只可能来自输出目标，不需要识别输入指令；这里必须单独识别控制像素。
#   - 原代码输出只有方块；这里若不把控制像素回写到输出，会丢失输入中显式存在的符号指令。
#
# d) placement_legality:
#   - 输入方块禁止触碰左右边界，这样左右边界上的 2/8 像素就能稳定充当控制像素。
#   - 输出仍只使用两层：边缘列与内侧列，保证不越界。
#   - 控制像素的行号专门避开该侧边缘列方块的 5 行跨度，避免输出时被边缘方块吞没。
#
# e) visual_observability:
#   - 人类能直接看到：哪种框色在左/右边界有一个同色单像素；输出中该组就整体堆向那一侧。
#   - 当生成器采样到“左右互换”时，同样的两种框色会反向落位，D13 变化在 2-3 组样例内可见。
#   - 排序只依赖空间位置（输入列离哪侧近），不依赖颜色 id、编号大小等不可见因素。
#
# f) visual_cleanliness:
#   - 控制像素始终是孤立单点，不与输入方块相连。
#   - 方块本体仍是原题的规整 5x5 框，输出只在左右两条带状区域组织，不会引入新的杂乱区域。
#
# g) color_inference:
#   - 没有新增颜色变换；控制像素直接复用组颜色，因此解题者只需学“边上的同色点 = 该组去这一边”。
#
# new_main_assumptions_checked_against_generator:
#   - main() 假设每种框色在左右边界恰有一个不属于任何 5x5 方块的控制像素；generator 明确构造这一点。
#   - main() 假设同组方块的最终深度最多 2 层；generator 仍按原题的 edge/inward 结构构造。
#   - main() 假设控制像素所在侧能决定唯一排序方向；generator 只使用 left/right 两种侧向且两组互为相反侧。

from __future__ import annotations

import random

import numpy as np

from arc_utils import crop_bbox, get_bg, paste


GRID_H, GRID_W = 22, 22
SQ = 5
AZURE, RED = 8, 2
FRAME_COLORS = (AZURE, RED)
N_AZURE_RANGE = (3, 4)
N_RED_RANGE = (3, 5)
MAX_INWARD = 2
INPUT_COL_RETRIES = 50
GEN_RETRIES = 500


def _rows_overlap(r_a: int, r_b: int) -> bool:
    """Check whether two SQ-tall row spans overlap."""
    return abs(int(r_a) - int(r_b)) < SQ


def _rects_overlap(r1: int, c1: int, r2: int, c2: int) -> bool:
    """Check whether two SQxSQ rectangles overlap."""
    return abs(int(r1) - int(r2)) < SQ and abs(int(c1) - int(c2)) < SQ


def _any_pairwise_overlap(positions: list[tuple[int, int, object]]) -> bool:
    """Return True if any two (r, c, ...) entries overlap as SQxSQ rects."""
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if _rects_overlap(
                int(positions[i][0]),
                int(positions[i][1]),
                int(positions[j][0]),
                int(positions[j][1]),
            ):
                return True
    return False


def _make_patch(frame_color: int, inner_color: int) -> np.ndarray:
    """Create a SQxSQ bordered patch with uniform interior."""
    p = np.full((SQ, SQ), int(frame_color), dtype=int)
    p[1:-1, 1:-1] = int(inner_color)
    return p


def _target_cols(side: str, width: int) -> tuple[int, int]:
    """Return (edge_col, inward_col) for a leftward or rightward stack."""
    if side == "left":
        return 0, SQ
    return int(width - SQ), int(width - 2 * SQ)


def _sort_descending(side: str) -> bool:
    """A right-targeting group processes squares nearest the right edge first."""
    return side == "right"


def _detect_squares(g: np.ndarray) -> list[dict[str, object]]:
    """Find all framed 5x5 squares, independent of control pixels."""
    H, W = g.shape
    squares: list[dict[str, object]] = []
    seen: set[tuple[int, int]] = set()
    for r0 in range(H - SQ + 1):
        for c0 in range(W - SQ + 1):
            patch = crop_bbox(g, (int(r0), int(c0), int(r0 + SQ - 1), int(c0 + SQ - 1)))
            border = np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])
            if int(border.min()) != int(border.max()):
                continue
            frame_col = int(border[0])
            if frame_col not in FRAME_COLORS:
                continue
            inner = patch[1:-1, 1:-1]
            if not bool((inner != int(frame_col)).any()):
                continue
            key = (int(r0), int(c0))
            if key in seen:
                continue
            seen.add(key)
            squares.append({"frame": int(frame_col), "r0": int(r0), "c0": int(c0), "patch": patch})
    return squares


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use edge control pixels to decide which side each frame-color group stacks toward.

    Input structure:
    - bg is the most common color.
    - Red (2) and teal (8) 5x5 framed squares with colored interiors are scattered in the grid.
    - A lone red border pixel and a lone teal border pixel sit on the left or right edge as instructions.

    Rule:
    - A group's edge marker chooses its destination side: left-edge marker -> stack at col 0, right-edge marker -> stack at col W-5.
    - The same marker also chooses processing order: leftward groups use input columns ascending, rightward groups descending.
    - Each square keeps its original row; if its 5-row span overlaps an already placed edge square from the same group, it shifts one square-width inward instead.
    - The control pixels themselves stay where they are.

    Key insight: The border pixels are explicit symbols for direction, so the original stacking logic is applied per color after first decoding those side instructions.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    squares = _detect_squares(g)
    square_mask = np.zeros((H, W), dtype=bool)
    for s in squares:
        r0 = int(s["r0"])
        c0 = int(s["c0"])
        square_mask[int(r0) : int(r0) + SQ, int(c0) : int(c0) + SQ] = True

    control_points: list[tuple[int, int, int]] = []
    dirs: dict[int, str] = {}
    for frame_col in FRAME_COLORS:
        pts = [
            (int(r), int(c))
            for r, c in np.argwhere(g == int(frame_col))
            if (int(c) == 0 or int(c) == int(W - 1)) and not bool(square_mask[int(r), int(c)])
        ]
        if len(pts) != 1:
            raise ValueError(f"Expected exactly one control pixel for frame color {frame_col}, found {len(pts)}")
        r, c = pts[0]
        control_points.append((int(r), int(c), int(frame_col)))
        dirs[int(frame_col)] = "left" if int(c) == 0 else "right"

    out = np.full_like(g, bg)

    def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
        return not (int(a1) < int(b0) or int(b1) < int(a0))

    for frame_col in FRAME_COLORS:
        side = dirs[int(frame_col)]
        edge_col, inward_col = _target_cols(side, W)
        group = sorted(
            [s for s in squares if int(s["frame"]) == int(frame_col)],
            key=lambda s: int(s["c0"]),
            reverse=_sort_descending(side),
        )
        placed_at_edge: list[tuple[int, int]] = []
        for s in group:
            r0 = int(s["r0"])
            r1 = int(r0 + SQ - 1)
            has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_edge)
            dest_c = int(inward_col) if has_overlap else int(edge_col)
            paste(out, s["patch"], r0, dest_c, transparent=bg)
            if dest_c == int(edge_col):
                placed_at_edge.append((r0, r1))

    for r, c, color in control_points:
        out[int(r), int(c)] = int(color)

    return out


def _assign_output_rows(n: int) -> tuple[list[int], list[int]] | None:
    """Pick edge and inward row positions for n squares on one side."""
    max_inward = min(MAX_INWARD, n - 1)
    n_inward = random.randint(1, max_inward)
    n_edge = n - n_inward

    all_rows = list(range(GRID_H - SQ + 1))
    random.shuffle(all_rows)
    edge_rows: list[int] = []
    for r in all_rows:
        if all(not _rows_overlap(r, er) for er in edge_rows):
            edge_rows.append(int(r))
            if len(edge_rows) == n_edge:
                break
    if len(edge_rows) < n_edge:
        return None

    random.shuffle(all_rows)
    inward_rows: list[int] = []
    for r in all_rows:
        if not any(_rows_overlap(r, er) for er in edge_rows):
            continue
        if any(_rows_overlap(r, ir) for ir in inward_rows):
            continue
        inward_rows.append(int(r))
        if len(inward_rows) == n_inward:
            break
    if len(inward_rows) < n_inward:
        return None
    return edge_rows, inward_rows


def _place_group_squares(
    output: np.ndarray,
    edge_rows: list[int],
    inward_rows: list[int],
    colors: list[int],
    frame_color: int,
    side: str,
) -> list[tuple[int, int, bool, np.ndarray]]:
    """Render a group's edge and inward squares to the chosen side."""
    edge_col, inward_col = _target_cols(side, output.shape[1])
    placed: list[tuple[int, int, bool, np.ndarray]] = []
    for i, r in enumerate(edge_rows):
        p = _make_patch(frame_color, colors[i])
        paste(output, p, int(r), int(edge_col))
        placed.append((int(r), int(edge_col), True, p))
    for i, r in enumerate(inward_rows):
        p = _make_patch(frame_color, colors[len(edge_rows) + i])
        paste(output, p, int(r), int(inward_col))
        placed.append((int(r), int(inward_col), False, p))
    return placed


def _derive_input_positions(
    sq_list: list[tuple[int, int, bool, np.ndarray]],
    side: str,
) -> list[tuple[int, int, np.ndarray]] | None:
    """Assign random input columns that reproduce the solver's directed stacking."""
    edges = [(r, oc, p) for (r, oc, is_edge, p) in sq_list if is_edge]
    inwards = [(r, oc, p) for (r, oc, is_edge, p) in sq_list if not is_edge]
    inward_deps = [
        [i for i, (er, _, _) in enumerate(edges) if _rows_overlap(ir, er)]
        for (ir, _, _) in inwards
    ]

    n = len(sq_list)
    col_range = list(range(1, GRID_W - SQ))
    if len(col_range) < n:
        return None

    descending = _sort_descending(side)
    for _ in range(INPUT_COL_RETRIES):
        edge_indices = list(range(len(edges)))
        random.shuffle(edge_indices)
        order: list[tuple[str, int]] = [("edge", i) for i in edge_indices]

        inward_order = list(range(len(inwards)))
        random.shuffle(inward_order)
        for iw_idx in inward_order:
            latest_pos = max(
                (
                    pos
                    for dep in inward_deps[iw_idx]
                    for pos, (typ, idx) in enumerate(order)
                    if typ == "edge" and idx == dep
                ),
                default=-1,
            )
            insert_pos = random.randint(latest_pos + 1, len(order))
            order.insert(insert_pos, ("inward", iw_idx))

        ordered_squares = [
            (edges[idx] if typ == "edge" else inwards[idx])
            for typ, idx in order
        ]

        chosen_cols = sorted(random.sample(col_range, n), reverse=descending)
        result: list[tuple[int, int, np.ndarray]] = []
        ok = True
        for i, (r, oc, p) in enumerate(ordered_squares):
            ic = int(chosen_cols[i])
            if ic == int(oc):
                ok = False
                break
            result.append((int(r), int(ic), p))
        if not ok:
            continue

        if not _any_pairwise_overlap(result):
            return result

    return None


def _choose_marker_row(edge_rows: list[int]) -> int | None:
    """Pick a border row not covered by any edge square on that side."""
    occupied = set()
    for r0 in edge_rows:
        occupied.update(range(int(r0), int(r0) + SQ))
    candidates = [r for r in range(GRID_H) if r not in occupied]
    if not candidates:
        return None
    return int(random.choice(candidates))


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid."""
    bg = random.choice([c for c in range(10) if c not in FRAME_COLORS])
    n_azure = random.randint(*N_AZURE_RANGE)
    n_red = random.randint(*N_RED_RANGE)

    swap_sides = random.choice([False, True])
    sides = {
        AZURE: ("right" if swap_sides else "left"),
        RED: ("left" if swap_sides else "right"),
    }

    azure_assign = _assign_output_rows(n_azure)
    if azure_assign is None:
        return None
    red_assign = _assign_output_rows(n_red)
    if red_assign is None:
        return None
    azure_edge_rows, azure_inward_rows = azure_assign
    red_edge_rows, red_inward_rows = red_assign

    avail_azure = [c for c in range(10) if c not in {bg, AZURE}]
    avail_red = [c for c in range(10) if c not in {bg, RED}]
    if len(avail_azure) < n_azure or len(avail_red) < n_red:
        return None
    azure_colors = random.sample(avail_azure, n_azure)
    red_colors = random.sample(avail_red, n_red)

    output = np.full((GRID_H, GRID_W), int(bg), dtype=int)
    azure_out = _place_group_squares(
        output,
        azure_edge_rows,
        azure_inward_rows,
        azure_colors,
        AZURE,
        sides[AZURE],
    )
    red_out = _place_group_squares(
        output,
        red_edge_rows,
        red_inward_rows,
        red_colors,
        RED,
        sides[RED],
    )

    marker_rows = {
        "left": _choose_marker_row(azure_edge_rows if sides[AZURE] == "left" else red_edge_rows),
        "right": _choose_marker_row(azure_edge_rows if sides[AZURE] == "right" else red_edge_rows),
    }
    if marker_rows["left"] is None or marker_rows["right"] is None:
        return None

    control_points = [
        (
            int(marker_rows[sides[AZURE]]),
            0 if sides[AZURE] == "left" else GRID_W - 1,
            AZURE,
        ),
        (
            int(marker_rows[sides[RED]]),
            0 if sides[RED] == "left" else GRID_W - 1,
            RED,
        ),
    ]
    for r, c, color in control_points:
        output[int(r), int(c)] = int(color)

    azure_input = _derive_input_positions(azure_out, sides[AZURE])
    if azure_input is None:
        return None
    red_input = _derive_input_positions(red_out, sides[RED])
    if red_input is None:
        return None

    all_positions = azure_input + red_input
    if _any_pairwise_overlap(all_positions):
        return None

    inp = np.full((GRID_H, GRID_W), int(bg), dtype=int)
    for r, c, p in all_positions:
        paste(inp, p, int(r), int(c))
    for r, c, color in control_points:
        inp[int(r), int(c)] = int(color)

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
