# Variant of b5ca7ac4
# Dimension: D1 — 分组标准——边框颜色二分法 (match_cond)
# Change: 通过边框颜色将框架分为两组：边框为青色(8)的归为左堆叠组，边框为红色(2)的归为右堆叠组... → 按面积或像素数分组
#
# Design analysis
# a) 具体化与行为追踪
# - 输入仍是“带单色边框、单色内部”的正方形框架，但不再全是 5x5；改为混合 7x7 大框架与 5x5 小框架。
# - 边框颜色仍只用红色(2)与青色(8)，但颜色不再决定去向；两种尺寸里都会出现两种边框色。
# - 输出规则保持原任务的空间语法：左组按原列从小到大处理并贴左边，右组按原列从大到小处理并贴右边；
#   只有“谁属于左组 / 右组”从看颜色改成看尺寸。
# - 具体追踪例子：若一个 7x7 红框和一个 7x7 青框都存在，它们都会去左侧；若一个 5x5 青框和一个
#   5x5 红框都存在，它们都会去右侧。原题会把红框送右侧、青框送左侧；本变体会把大框架送左、小框架送右。
#
# b) 复杂度校准
# - 只保留两档尺寸：7x7 与 5x5。这样“按面积分两组”在视觉上清楚，也避免尺寸自由度过多导致难以归纳。
# - 颜色仍限制在 {2, 8}，并且每个尺寸组内都强制同时出现两种颜色，确保解题者不能退回到颜色分组。
#
# c) 隐式假设审查（对应原 main 的关键依赖）
# - 原代码固定扫描 5x5 窗口；这里必须改成扫描 7x7 和 5x5 两档，否则会漏掉大框架。
# - 原代码用 frame color 直接划分组；这里必须改成用检测到的 size 划分组，否则 D1 没有真正变化。
# - 原代码 inward 偏移量固定为 5；这里左组要按 7 列 inward，右组仍按 5 列 inward，因为两组尺寸不同。
# - 原代码的 row-overlap 判断默认双方高度都等于 5；这里要显式使用各自尺寸计算区间重叠。
# - 原代码“只记录贴边对象，再决定后续是否 inward”的逻辑仍成立，但前提是每组最多两层；生成器继续保证。
#
# d) 放置合法性
# - 由于左侧最多占两层 7x7（宽 14），右侧最多占两层 5x5（宽 10），22 列已不够，因此画布改为 26x26。
# - 生成器继续采用 output-first：先选输出行与边/内层，再反推输入列，并用 main(input)==output 验证。
# - 输入与输出都保证对象不重叠、不越界；输入散布时还保证同一网格中的大框架与小框架互不覆盖。
#
# e) 视觉可观察性
# - 每个样例中，大组与小组都混入红框和青框，因此“颜色→去向”会被样例直接反证。
# - 仍保留原题的“靠边堆叠 + 行重叠触发 inward”外观，让唯一显著变化集中在“分组依据从颜色变为尺寸”。
#
# f) 可视化整洁度
# - 对象仍是规则方框，内部纯色，边框厚度 1，避免复杂内部纹理干扰对尺寸的判断。
# - 只允许两层堆叠，左区与右区在 26 列画布上留有空隙，输出不会挤成一团。
#
# g) 颜色变化可推断性
# - 本变体不做颜色映射，只把颜色从“决定分组的线索”降级为“无关外观属性”。
# - 颜色种类仍然很少，避免引入额外的不可见规则。
#
# h) main() 的新假设与 generator 的对应保证
# - main() 假设合法对象只会是 7x7 或 5x5 的单色边框正方形；generator 只生成这两种。
# - main() 假设每个对象内部是单色且不等于边框色；generator 显式这样构造，避免扫描误检。
# - main() 假设每个尺寸组最多只有 edge + inward 两层；generator 通过输出行分配保证这一点。

from __future__ import annotations

import random

import numpy as np

from arc_utils import crop_bbox, get_bg, paste


LARGE = 7
SMALL = 5
FRAME_COLORS = (2, 8)


def _make_patch(size: int, frame_color: int, inner_color: int) -> np.ndarray:
    """Create a size x size bordered square with a uniform interior."""
    patch = np.full((size, size), int(frame_color), dtype=int)
    patch[1:-1, 1:-1] = int(inner_color)
    return patch


def _rows_overlap(r_a: int, size_a: int, r_b: int, size_b: int) -> bool:
    """Return True if the two vertical spans intersect."""
    a1 = int(r_a + size_a - 1)
    b1 = int(r_b + size_b - 1)
    return not (a1 < int(r_b) or b1 < int(r_a))


def _rects_overlap(a: tuple, b: tuple) -> bool:
    """Return True if two square placements overlap."""
    ar, ac, asz = map(int, a[:3])
    br, bc, bsz = map(int, b[:3])
    return not (
        ar + asz - 1 < br
        or br + bsz - 1 < ar
        or ac + asz - 1 < bc
        or bc + bsz - 1 < ac
    )


def _any_pairwise_overlap(positions: list[tuple]) -> bool:
    """Check whether any two (row, col, size, ...) placements overlap."""
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if _rects_overlap(positions[i], positions[j]):
                return True
    return False


def _detect_frames(grid: np.ndarray) -> list[dict]:
    """Scan for 7x7 and 5x5 bordered squares with uniform interiors."""
    g = np.array(grid, dtype=int)
    H, W = g.shape
    found: list[dict] = []

    for size in (LARGE, SMALL):
        for r0 in range(H - size + 1):
            for c0 in range(W - size + 1):
                patch = crop_bbox(g, (int(r0), int(c0), int(r0 + size - 1), int(c0 + size - 1)))
                border = np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])
                if int(border.min()) != int(border.max()):
                    continue
                frame_col = int(border[0])
                if frame_col not in FRAME_COLORS:
                    continue

                inner = patch[1:-1, 1:-1]
                inner_col = int(inner[0, 0])
                if inner_col == frame_col:
                    continue
                if int(inner.min()) != int(inner.max()):
                    continue

                found.append(
                    {
                        "size": int(size),
                        "frame": int(frame_col),
                        "r0": int(r0),
                        "c0": int(c0),
                        "patch": patch,
                    }
                )

    return found


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move mixed-size bordered squares to their designated grid edges by measured size.

    Input structure:
    - bg = most common color.
    - Scattered bordered squares come in two sizes: 7x7 and 5x5.
    - Frame colors are still red (2) or teal (8), but both colors may appear in either size.

    Rule:
    - Large 7x7 squares → left edge (col 0); small 5x5 squares → right edge (col W−5).
    - Each square keeps its original row.
    - Process large squares left-to-right, small squares right-to-left.
    - If a square's rows overlap an already-placed edge square from the same size group, place it one own-width inward instead.

    Key insight: The deciding feature is geometric size, not border color; color is now a distractor that appears in both destination groups.
    """
    g = np.array(grid, dtype=int)
    _H, W = g.shape
    bg = int(get_bg(g))
    out = np.full_like(g, bg)

    squares = _detect_frames(g)

    large_squares = sorted((s for s in squares if int(s["size"]) == LARGE), key=lambda s: int(s["c0"]))
    placed_large_edges: list[int] = []
    for sq in large_squares:
        r0 = int(sq["r0"])
        has_overlap = any(_rows_overlap(r0, LARGE, pr0, LARGE) for pr0 in placed_large_edges)
        dest_c = int(LARGE) if has_overlap else 0
        paste(out, sq["patch"], r0, dest_c, transparent=bg)
        if dest_c == 0:
            placed_large_edges.append(r0)

    small_squares = sorted((s for s in squares if int(s["size"]) == SMALL), key=lambda s: -int(s["c0"]))
    placed_small_edges: list[int] = []
    for sq in small_squares:
        r0 = int(sq["r0"])
        has_overlap = any(_rows_overlap(r0, SMALL, pr0, SMALL) for pr0 in placed_small_edges)
        dest_c = int(W - 2 * SMALL) if has_overlap else int(W - SMALL)
        paste(out, sq["patch"], r0, dest_c, transparent=bg)
        if dest_c == int(W - SMALL):
            placed_small_edges.append(r0)

    return out


# --------------- generator (D1 variant) ---------------

GRID_H, GRID_W = 26, 26
N_LARGE_RANGE = (3, 4)
N_SMALL_RANGE = (3, 4)
MAX_INWARD = 2
INPUT_COL_RETRIES = 80
GEN_RETRIES = 800


def _assign_output_rows(n: int, size: int) -> tuple[list[int], list[int]] | None:
    """Choose edge rows and inward rows for one size group."""
    max_inward = min(MAX_INWARD, n - 1)
    n_inward = random.randint(1, max_inward)
    n_edge = n - n_inward

    all_rows = list(range(GRID_H - size + 1))
    random.shuffle(all_rows)
    edge_rows: list[int] = []
    for r in all_rows:
        if all(not _rows_overlap(r, size, er, size) for er in edge_rows):
            edge_rows.append(r)
            if len(edge_rows) == n_edge:
                break
    if len(edge_rows) < n_edge:
        return None

    random.shuffle(all_rows)
    inward_rows: list[int] = []
    for r in all_rows:
        if not any(_rows_overlap(r, size, er, size) for er in edge_rows):
            continue
        if any(_rows_overlap(r, size, ir, size) for ir in inward_rows):
            continue
        inward_rows.append(r)
        if len(inward_rows) == n_inward:
            break
    if len(inward_rows) < n_inward:
        return None

    return edge_rows, inward_rows


def _assign_frame_colors(n: int) -> list[int]:
    """Use both frame colors inside each size group to break the old color rule."""
    colors = [FRAME_COLORS[0], FRAME_COLORS[1]]
    colors.extend(random.choice(FRAME_COLORS) for _ in range(n - 2))
    random.shuffle(colors)
    return colors


def _assign_inner_colors(frame_colors: list[int], bg: int) -> list[int]:
    """Pick mostly distinct interior colors compatible with each frame color."""
    used: set[int] = set()
    colors: list[int] = []
    for frame in frame_colors:
        choices = [c for c in range(10) if c not in {bg, int(frame)} and c not in used]
        if not choices:
            choices = [c for c in range(10) if c not in {bg, int(frame)}]
        chosen = int(random.choice(choices))
        colors.append(chosen)
        used.add(chosen)
    return colors


def _place_side_squares(
    output: np.ndarray,
    edge_rows: list[int],
    inward_rows: list[int],
    size: int,
    frame_colors: list[int],
    inner_colors: list[int],
    edge_col: int,
    inward_col: int,
) -> list[tuple[int, int, int, bool, np.ndarray]]:
    """Render one size group into the output and return metadata."""
    placed: list[tuple[int, int, int, bool, np.ndarray]] = []
    idx = 0
    for r in edge_rows:
        patch = _make_patch(size, frame_colors[idx], inner_colors[idx])
        paste(output, patch, r, edge_col)
        placed.append((int(r), int(edge_col), int(size), True, patch))
        idx += 1
    for r in inward_rows:
        patch = _make_patch(size, frame_colors[idx], inner_colors[idx])
        paste(output, patch, r, inward_col)
        placed.append((int(r), int(inward_col), int(size), False, patch))
        idx += 1
    return placed


def _derive_input_positions(
    sq_list: list[tuple[int, int, int, bool, np.ndarray]],
    ascending: bool,
    size: int,
) -> list[tuple[int, int, int, np.ndarray]] | None:
    """Assign input columns that reproduce the intended edge/inward output."""
    edges = [(r, oc, sz, patch) for (r, oc, sz, is_edge, patch) in sq_list if is_edge]
    inwards = [(r, oc, sz, patch) for (r, oc, sz, is_edge, patch) in sq_list if not is_edge]

    inward_deps = [
        [i for i, (er, _, _, _) in enumerate(edges) if _rows_overlap(ir, size, er, size)]
        for (ir, _, _, _) in inwards
    ]

    n = len(sq_list)
    col_range = list(range(0, GRID_W - size + 1))
    if len(col_range) < n:
        return None

    for _ in range(INPUT_COL_RETRIES):
        edge_indices = list(range(len(edges)))
        random.shuffle(edge_indices)
        order: list[tuple[str, int]] = [("edge", i) for i in edge_indices]

        inward_indices = list(range(len(inwards)))
        random.shuffle(inward_indices)
        for iw_idx in inward_indices:
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

        ordered = [edges[idx] if typ == "edge" else inwards[idx] for typ, idx in order]

        chosen_cols = sorted(random.sample(col_range, n))
        if not ascending:
            chosen_cols = chosen_cols[::-1]

        result: list[tuple[int, int, int, np.ndarray]] = []
        ok = True
        for i, (r, oc, sz, patch) in enumerate(ordered):
            input_col = int(chosen_cols[i])
            if input_col == int(oc):
                ok = False
                break
            result.append((int(r), input_col, int(sz), patch))
        if not ok:
            continue

        if not _any_pairwise_overlap(result):
            return result

    return None


def _gen_attempt() -> np.ndarray | None:
    """Single output-first generation attempt."""
    bg = int(random.choice([c for c in range(10) if c not in FRAME_COLORS]))
    n_large = int(random.randint(*N_LARGE_RANGE))
    n_small = int(random.randint(*N_SMALL_RANGE))

    large_assign = _assign_output_rows(n_large, LARGE)
    if large_assign is None:
        return None
    small_assign = _assign_output_rows(n_small, SMALL)
    if small_assign is None:
        return None
    large_edge_rows, large_inward_rows = large_assign
    small_edge_rows, small_inward_rows = small_assign

    large_frames = _assign_frame_colors(n_large)
    small_frames = _assign_frame_colors(n_small)
    large_inners = _assign_inner_colors(large_frames, bg)
    small_inners = _assign_inner_colors(small_frames, bg)

    output = np.full((GRID_H, GRID_W), bg, dtype=int)
    large_out = _place_side_squares(
        output,
        large_edge_rows,
        large_inward_rows,
        LARGE,
        large_frames,
        large_inners,
        edge_col=0,
        inward_col=LARGE,
    )
    small_out = _place_side_squares(
        output,
        small_edge_rows,
        small_inward_rows,
        SMALL,
        small_frames,
        small_inners,
        edge_col=GRID_W - SMALL,
        inward_col=GRID_W - 2 * SMALL,
    )

    large_input = _derive_input_positions(large_out, ascending=True, size=LARGE)
    if large_input is None:
        return None
    small_input = _derive_input_positions(small_out, ascending=False, size=SMALL)
    if small_input is None:
        return None

    all_positions = large_input + small_input
    if _any_pairwise_overlap(all_positions):
        return None

    inp = np.full((GRID_H, GRID_W), bg, dtype=int)
    for r, c, _size, patch in all_positions:
        paste(inp, patch, r, c)

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
