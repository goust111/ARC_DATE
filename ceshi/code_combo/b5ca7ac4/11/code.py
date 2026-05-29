# Combo variant of b5ca7ac4 (index 11)
# combo_spec: D5/0+D13/0
# Name: 边缘标记指示+反转堆叠方向
#
# Design analysis
# ===============
# combo_variant_spec:
#   preserved:
#     - 22x22 画布、5x5 带框方块、框色固定为 8/2、内部仍是 3x3 单色填充
#     - 方块保持原始行位置
#     - 每组仍只有两层堆叠：边缘列 + 向内一格
#     - 是否退到内层仍只由“与已占边缘列的同组方块是否纵向重叠”决定
#   changed:
#     - 加入两个显式控制像素：一个 8、一个 2，且都位于左右边界之一
#     - 控制像素所在边就是该组的目标边，解题时必须先读指令再排序/堆叠
#     - 同时套用 D5 反转方向：本融合版固定构造 8 在右边、2 在左边
#     - 因此排序方向也随目标边镜像：
#         left  => 输入列升序
#         right => 输入列降序
#     - 控制像素在输出中保留原位
#
# a) concretization_and_trace:
#   - 原题：8 框固定去左，按列升序；2 框固定去右，按列降序。
#   - 融合后：输入边缘上会多出一个孤立 8 像素和一个孤立 2 像素。
#     这里生成器固定让 8 像素出现在右边界、2 像素出现在左边界，
#     所以青框组改为“向右堆叠 + 右侧优先”，红框组改为“向左堆叠 + 左侧优先”。
#   - 具体例子：若两个青框方块列坐标是 3 和 14，且它们纵向重叠。
#       原题会让 col=3 的方块先占左边 col 0，另一个退到 col 5。
#       本变体会先读到 8 色控制像素在右边，于是 col=14 的方块先占右边 col W-5，
#       col=3 的方块退到 col W-10。
#     差异点不是单独的“方向翻转”或“出现标记”，而是标记明确声明了这次翻转后的目标边。
#
# b) complexity_calibration:
#   - 方向只有左右两档，没有额外颜色映射，复杂度足够低。
#   - 两个颜色组都会重复使用同一套“看边缘像素 -> 定方向 -> 定排序 -> 堆叠”的链条，
#     不会只在单个对象上偶发触发。
#   - 为避免退化成原题，生成器固定采样反转后的映射：8->右、2->左，
#     因而与原始 `8->左, 2->右` 在大多数样本上会产生不同输出。
#
# c) implicit_assumption_audit_of_original_main:
#   - 原代码把 8/2 的目标边写死；这里必须改成先找不属于 5x5 方块的边界单像素。
#   - 原代码把排序方向写死为 8 升序、2 降序；这里必须改成“朝左则升序，朝右则降序”。
#   - 原代码默认边界上的 2/8 像素只可能是方块一部分；这里边界单像素可能是控制指令，
#     所以要先检测方块，再用方块遮罩排除边框像素。
#   - 原代码输出不含额外符号；这里若不把控制像素回写，会丢掉输入中的显式规则提示。
#
# d) placement_legality:
#   - 输入方块列坐标只从 1..W-SQ-1 采样，避免接触左右边界，保证控制像素始终孤立可辨。
#   - 输出仍只用两层列位：左侧 0/5，右侧 W-5/W-10，不会越界。
#   - 控制像素的行号专门避开该侧 edge 层方块覆盖的 5 行跨度，避免输出时被方块覆盖。
#   - 生成器继续做整网格矩形重叠检查，防止输入方块互相遮挡。
#
# e) visual_observability:
#   - 人类能直接看到：右边界上的 8 像素对应青框整体堆到右带，左边界上的 2 像素对应红框堆到左带。
#   - 输出中的边缘/内层分配仍由“谁离目标边更近且先被处理”决定，能在 2-3 个样例中看到列顺序反转。
#   - 排序完全由空间上的接近目标边决定，不依赖颜色编号大小等不可见因素。
#
# f) visual_cleanliness:
#   - 控制信号始终是两个孤立单点，不与任意 5x5 方块接触。
#   - 方块仍集中在左右两条带状区域内移动；没有新增杂色区域或高层数堆叠。
#   - 两个颜色组落在相反边，视觉分区清晰。
#
# g) color_inference:
#   - 没有新增颜色替换规则；控制像素直接复用对应框色。
#   - 解题者只需归纳“边上的同色点告诉你这组往哪边堆”，然后从样例观察到此题实际是 8->右、2->左。
#
# new_main_assumptions_checked_against_generator:
#   - main() 假设每种框色恰有一个不属于任何 5x5 方块的边界控制像素；generator 明确构造这一点。
#   - main() 假设所有目标堆叠都只需两层；generator 仍按 edge/inward 二层输出先构造。
#   - main() 假设控制像素所在边与该组输出目标边一致；generator 固定放成 8 在右、2 在左。
#   - main() 假设输入方块不贴左右边界，否则控制像素和方块边框会混淆；generator 用输入列范围保证这一点。

from __future__ import annotations

import random

import numpy as np

from arc_utils import crop_bbox, get_bg, paste


GRID_H, GRID_W = 22, 22
SQ = 5
AZURE, RED = 8, 2
FRAME_COLORS = (AZURE, RED)
TARGET_SIDE = {
    AZURE: "right",
    RED: "left",
}
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
    """A right-targeting group is processed nearest the right edge first."""
    return side == "right"


def _detect_squares(g: np.ndarray) -> list[dict[str, object]]:
    """Find all framed 5x5 squares while ignoring control pixels."""
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


def _find_control_points(g: np.ndarray, squares: list[dict[str, object]]) -> tuple[list[tuple[int, int, int]], dict[int, str]]:
    """Locate one border control pixel per frame color, excluding square pixels."""
    H, W = g.shape
    square_mask = np.zeros((H, W), dtype=bool)
    for s in squares:
        r0 = int(s["r0"])
        c0 = int(s["c0"])
        square_mask[int(r0) : int(r0) + SQ, int(c0) : int(c0) + SQ] = True

    control_points: list[tuple[int, int, int]] = []
    directions: dict[int, str] = {}
    for frame_col in FRAME_COLORS:
        pts = [
            (int(r), int(c))
            for r, c in np.argwhere(g == int(frame_col))
            if (int(c) == 0 or int(c) == int(W - 1)) and not bool(square_mask[int(r), int(c)])
        ]
        if len(pts) != 1:
            raise ValueError(f"Expected exactly one control pixel for frame color {frame_col}, found {len(pts)}")
        r, c = pts[0]
        side = "left" if int(c) == 0 else "right"
        control_points.append((int(r), int(c), int(frame_col)))
        directions[int(frame_col)] = side
    return control_points, directions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read edge markers to decide the reversed stacking side for each framed-square group.

    Input structure:
    - bg is the most common color.
    - Teal (8) and red (2) 5x5 framed squares with colored 3x3 interiors are scattered in the grid.
    - One lone teal pixel and one lone red pixel sit on the left or right border as direction markers.

    Rule:
    - A border marker tells its own frame-color group which side to stack toward: left-edge marker means stack at cols 0/5, right-edge marker means stack at cols W-5/W-10.
    - A left-targeting group is processed by input column ascending; a right-targeting group is processed by input column descending.
    - Each square keeps its original row; if its 5-row span overlaps an already placed edge-layer square from the same group, it moves one square-width inward instead of taking the edge column.
    - The markers themselves remain at their original border cells.

    Key insight: The old color-to-side default is no longer valid; first decode the edge markers, then apply the same overlap-based two-layer stacking logic on the indicated side.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    squares = _detect_squares(g)
    control_points, directions = _find_control_points(g, squares)

    out = np.full((H, W), int(bg), dtype=int)

    def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
        return not (int(a1) < int(b0) or int(b1) < int(a0))

    for frame_col in FRAME_COLORS:
        side = directions[int(frame_col)]
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
            paste(out, s["patch"], int(r0), int(dest_c), transparent=bg)
            if dest_c == int(edge_col):
                placed_at_edge.append((int(r0), int(r1)))

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
    """Render edge and inward squares for one color group on its target side."""
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
    """Assign random input columns that reproduce the side-dependent processing order."""
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
    """Pick a border row not covered by any edge-layer square on that side."""
    occupied = set()
    for r0 in edge_rows:
        occupied.update(range(int(r0), int(r0) + SQ))
    candidates = [r for r in range(GRID_H) if r not in occupied]
    if not candidates:
        return None
    return int(random.choice(candidates))


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid for the combo rule."""
    bg = random.choice([c for c in range(10) if c not in FRAME_COLORS])
    n_azure = random.randint(*N_AZURE_RANGE)
    n_red = random.randint(*N_RED_RANGE)

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
        TARGET_SIDE[AZURE],
    )
    red_out = _place_group_squares(
        output,
        red_edge_rows,
        red_inward_rows,
        red_colors,
        RED,
        TARGET_SIDE[RED],
    )

    azure_marker_row = _choose_marker_row(azure_edge_rows)
    red_marker_row = _choose_marker_row(red_edge_rows)
    if azure_marker_row is None or red_marker_row is None:
        return None

    control_points = [
        (int(azure_marker_row), GRID_W - 1, AZURE),
        (int(red_marker_row), 0, RED),
    ]
    for r, c, color in control_points:
        output[int(r), int(c)] = int(color)

    azure_input = _derive_input_positions(azure_out, TARGET_SIDE[AZURE])
    if azure_input is None:
        return None
    red_input = _derive_input_positions(red_out, TARGET_SIDE[RED])
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
