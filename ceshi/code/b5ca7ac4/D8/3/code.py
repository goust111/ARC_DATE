# Variant of b5ca7ac4
# Dimension: D8 — 变换——框架在移动过程中的变化 (transform)
# Change: 框架在从输入位置移动到输出位置时保持完全不变：边框颜色、内部颜色、尺寸均不改变... → 堆叠后内部颜色转移到边框

from __future__ import annotations

import random

import numpy as np

from arc_utils import crop_bbox, get_bg, paste


# Design analysis
# a) 具体化与行为追踪
#    - 输入仍然是若干个 5x5 框架：外圈 16 格是固定框色，青色框=8，红色框=2；中心 3x3 是单一纯色填充。
#    - 位置规则完全保留原题：青框按输入列从小到大处理，堆到左边；红框按输入列从大到小处理，堆到右边；
#      若与已经占据边缘列的同组框在 5 行跨度上重叠，则退到第二列。
#    - D8 变体只作用在“贴到输出”这一刻：原来直接复制 5x5 patch；现在改成交换框色与内部色。
#      例如输入里某个青框是“8 边框 + 3 内部”，原逻辑输出仍是“8 边框 + 3 内部”，
#      变体输出则是“3 边框 + 8 内部”。几何形状、目标列、原始行号都不变。
# b) 复杂度校准
#    - 不引入额外形状、额外排序键或多段式颜色表，只做局部颜色交换，避免偏离 D8 核心。
#    - 为了让交换足够显眼，generator 禁止内部色取 2/8/bg：这样输出边框不会继续长成原题那种红/青分组样子，
#      而是明确显示“边框吃掉了原内部色，内部变成了原框色”。
# c) 原 main() 中与变体相关的旧前提
#    - 原代码默认检测到的 patch 可以原样 paste；这在变体中失效，必须先构造交换后的 patch。
#    - 原代码用 transparent=bg paste。原题无害，因为 patch 中不会出现 bg；变体虽然生成器仍禁止内部=bg，
#      但 solver 改成不透明 paste，更稳妥，不把“内部绝不为 bg”当成隐式覆盖前提。
#    - 颜色交换要求内部存在单一可转移颜色，因此 solver 检测时显式要求 3x3 内部为纯色。
# d) 放置合法性
#    - 仍沿用原题 output-first：先决定输出堆叠的行/列，再反推输入列顺序。
#    - 输出中的框依旧都是完整 5x5，左右两侧最多两列，不会彼此重叠；输入位置仍做同样的矩形冲突检查。
# e) 视觉可观察性
#    - 解题者能看到：所有对象移动方式与原题一致，但每个对象在到达目标堆叠位置后，边框和内部对调。
#    - 因为每个对象都保留了原行号和完整轮廓，输入/输出之间能一一对应，不需要借助不可见排序。
# f) 可视化整洁度
#    - 仍只在左右两侧各使用两列堆叠，避免多层覆盖；每个框内部仍为纯色，输出不杂乱。
# g) 颜色变化可推断性
#    - 颜色变化是局部且一致的：每个框都执行同一个“边框色 ← 原内部色；内部色 ← 原边框色”。
#    - 不是依赖具体颜色 id 的查表，而是依赖可见结构位置（边框 vs 内部），因此 2-3 个样例就足以归纳。
# 9) 新隐式假设审查
#    - 扫描 5x5 窗口时假设真对象的内部是纯色且非框色；generator 保证这一点。
#    - 排序和重叠判定仍只看输入边框色与原始列坐标；generator 的反推输入列逻辑继续保证该顺序可复现。
#    - 交换后的输出边框色不再固定为 2/8；因此 solver 绝不能从输出边框反推组别，只能从输入框色分类。


GRID_H, GRID_W = 22, 22
SQ = 5
AZURE, RED = 8, 2
N_AZURE_RANGE = (3, 4)
N_RED_RANGE = (3, 5)
MAX_INWARD = 2
INPUT_COL_RETRIES = 50
GEN_RETRIES = 500


def _make_patch(frame_color: int, inner_color: int) -> np.ndarray:
    """Create a SQxSQ bordered patch with a uniform 3x3 interior."""
    p = np.full((SQ, SQ), int(frame_color), dtype=int)
    p[1:-1, 1:-1] = int(inner_color)
    return p


def _swap_patch_colors(patch: np.ndarray) -> np.ndarray:
    """Swap the uniform border color with the uniform interior color."""
    border = np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])
    if int(border.min()) != int(border.max()):
        raise ValueError("patch border is not uniform")
    frame_color = int(border[0])

    inner = patch[1:-1, 1:-1]
    if int(inner.min()) != int(inner.max()):
        raise ValueError("patch interior is not uniform")
    inner_color = int(inner[0, 0])
    if inner_color == frame_color:
        raise ValueError("patch interior matches frame color")

    return _make_patch(inner_color, frame_color)


def _rows_overlap(r_a: int, r_b: int) -> bool:
    """Check whether two SQ-tall row spans overlap."""
    return abs(int(r_a) - int(r_b)) < SQ


def _rects_overlap(r1: int, c1: int, r2: int, c2: int) -> bool:
    """Check whether two SQxSQ rectangles overlap."""
    return abs(int(r1) - int(r2)) < SQ and abs(int(c1) - int(c2)) < SQ


def _any_pairwise_overlap(positions: list[tuple[int, int, np.ndarray]]) -> bool:
    """Return True if any two (r, c, patch) entries overlap as SQxSQ rects."""
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if _rects_overlap(positions[i][0], positions[i][1], positions[j][0], positions[j][1]):
                return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move 5x5 framed squares to side stacks based on their INPUT frame color, then swap frame and fill colors.

    Input structure:
    - The background is the most common color.
    - Scattered 5x5 squares have a uniform teal (8) or red (2) border and a uniform 3x3 interior color.

    Rule:
    - Teal-bordered squares move toward the left stack; red-bordered squares move toward the right stack.
    - Each square keeps its original top row. Teal squares are processed by increasing input column; red squares by decreasing input column.
    - If a square's 5-row span overlaps an already placed edge square of the same side, it goes one square-width inward.
    - After choosing that destination, swap the square's colors: new border = old interior, new interior = old border.

    Key insight:
    - Placement still depends only on the input border color and the original horizontal order; the visible color change happens after movement, so the output border colors no longer identify the groups.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    out = np.full_like(g, bg)

    def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
        return not (int(a1) < int(b0) or int(b1) < int(a0))

    squares = []
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
            if int(inner.min()) != int(inner.max()):
                continue
            inner_col = int(inner[0, 0])
            if inner_col == frame_col:
                continue

            squares.append(
                {
                    "frame": frame_col,
                    "r0": int(r0),
                    "c0": int(c0),
                    "patch_out": _make_patch(inner_col, frame_col),
                }
            )

    azure_squares = sorted((s for s in squares if int(s["frame"]) == AZURE), key=lambda s: int(s["c0"]))
    placed_at_edge: list[tuple[int, int]] = []
    for s in azure_squares:
        r0 = int(s["r0"])
        r1 = int(r0 + SQ - 1)
        has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_edge)
        dest_c = SQ if has_overlap else 0
        paste(out, s["patch_out"], r0, dest_c)
        if dest_c == 0:
            placed_at_edge.append((r0, r1))

    red_squares = sorted((s for s in squares if int(s["frame"]) == RED), key=lambda s: -int(s["c0"]))
    placed_at_edge = []
    for s in red_squares:
        r0 = int(s["r0"])
        r1 = int(r0 + SQ - 1)
        has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_edge)
        dest_c = int(W - 2 * SQ) if has_overlap else int(W - SQ)
        paste(out, s["patch_out"], r0, dest_c)
        if dest_c == int(W - SQ):
            placed_at_edge.append((r0, r1))

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
            edge_rows.append(r)
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
        inward_rows.append(r)
        if len(inward_rows) == n_inward:
            break
    if len(inward_rows) < n_inward:
        return None
    return edge_rows, inward_rows


def _place_side_squares(
    output: np.ndarray,
    edge_rows: list[int],
    inward_rows: list[int],
    colors: list[int],
    frame_color: int,
    edge_col: int,
    inward_col: int,
) -> list[tuple[int, int, bool, np.ndarray]]:
    """Render swapped-color output squares and return input-patch metadata."""
    placed: list[tuple[int, int, bool, np.ndarray]] = []
    idx = 0
    for r in edge_rows:
        input_patch = _make_patch(frame_color, colors[idx])
        paste(output, _swap_patch_colors(input_patch), r, edge_col)
        placed.append((r, edge_col, True, input_patch))
        idx += 1
    for r in inward_rows:
        input_patch = _make_patch(frame_color, colors[idx])
        paste(output, _swap_patch_colors(input_patch), r, inward_col)
        placed.append((r, inward_col, False, input_patch))
        idx += 1
    return placed


def _derive_input_positions(
    sq_list: list[tuple[int, int, bool, np.ndarray]],
    ascending: bool,
) -> list[tuple[int, int, np.ndarray]] | None:
    """Assign input columns that recreate the requested edge/inward output layout."""
    edges = [(r, oc, p) for (r, oc, is_edge, p) in sq_list if is_edge]
    inwards = [(r, oc, p) for (r, oc, is_edge, p) in sq_list if not is_edge]

    inward_deps = [[i for i, (er, _, _) in enumerate(edges) if _rows_overlap(ir, er)] for (ir, _, _) in inwards]

    n = len(sq_list)
    col_range = list(range(0, GRID_W - SQ + 1))
    if len(col_range) < n:
        return None

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
                    for d in inward_deps[iw_idx]
                    for pos, (typ, idx) in enumerate(order)
                    if typ == "edge" and idx == d
                ),
                default=-1,
            )
            insert_pos = random.randint(latest_pos + 1, len(order))
            order.insert(insert_pos, ("inward", iw_idx))

        ordered_squares = [edges[idx] if typ == "edge" else inwards[idx] for typ, idx in order]

        chosen_cols = sorted(random.sample(col_range, n))
        if not ascending:
            chosen_cols = chosen_cols[::-1]

        result = []
        ok = True
        for i, (r, oc, p) in enumerate(ordered_squares):
            ic = chosen_cols[i]
            if ic == oc:
                ok = False
                break
            result.append((r, ic, p))
        if not ok:
            continue

        if not _any_pairwise_overlap(result):
            return result

    return None


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid."""
    bg = random.choice([c for c in range(10) if c not in {AZURE, RED}])
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

    # Exclude both original frame colors so the swap is visually obvious in the output.
    avail_azure = [c for c in range(10) if c not in {bg, AZURE, RED}]
    avail_red = [c for c in range(10) if c not in {bg, AZURE, RED}]
    if len(avail_azure) < n_azure or len(avail_red) < n_red:
        return None
    azure_colors = random.sample(avail_azure, n_azure)
    red_colors = random.sample(avail_red, n_red)

    output = np.full((GRID_H, GRID_W), bg, dtype=int)
    azure_out = _place_side_squares(output, azure_edge_rows, azure_inward_rows, azure_colors, AZURE, 0, SQ)
    red_out = _place_side_squares(
        output,
        red_edge_rows,
        red_inward_rows,
        red_colors,
        RED,
        GRID_W - SQ,
        GRID_W - 2 * SQ,
    )

    azure_input = _derive_input_positions(azure_out, ascending=True)
    if azure_input is None:
        return None
    red_input = _derive_input_positions(red_out, ascending=False)
    if red_input is None:
        return None

    all_positions = azure_input + red_input
    if _any_pairwise_overlap(all_positions):
        return None

    inp = np.full((GRID_H, GRID_W), bg, dtype=int)
    for r, c, p in all_positions:
        paste(inp, p, r, c)
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
