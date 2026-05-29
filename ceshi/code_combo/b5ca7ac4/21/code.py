# Combo variant of b5ca7ac4 (index 21)
# combo_spec: D1/1+D3/0
# Name: 面积分组+可变尺寸框架
#
# Design analysis
# a) 具体化与行为追踪
# - 输入对象仍然是“单色边框 + 单色内部”的正方形框架，但现在混合两种可见尺寸：7x7 和 5x5。
# - 融合后的分组不再看边框颜色，而是直接看框架面积/尺寸：7x7 大框架整组去左边，5x5 小框架整组去右边。
# - 具体例子：一个 7x7 红框、一个 7x7 青框、一个 5x5 红框、一个 5x5 青框同时出现时，
#   原题会按颜色把青框送左、红框送右；本融合版会把两个 7x7 一起送左，把两个 5x5 一起送右。
# - 堆叠机制仍保留原题语法：左组按输入列从小到大处理，右组按输入列从大到小处理；若与已贴边对象
#   发生纵向重叠，则只退一层到第二列带。因为左组全是 7x7，右组全是 5x5，所以 inward 偏移分别是 7 和 5。
#
# b) 复杂度校准
# - 只使用 7x7 与 5x5 两档尺寸，让“按面积分组”在 2-4 个样例里能被肉眼稳定观察到。
# - 每个尺寸组都强制同时含有红框(2)和青框(8)，从而直接反证“颜色决定去向”的旧规则。
# - 每侧都至少出现一个 inward 对象，保证“行重叠触发第二层”的结构在样例中持续可见，而不是偶然消失。
#
# c) 隐式假设审查
# - 原 solver 固定扫描 5x5；融合后必须同时扫描 7x7 和 5x5，否则会漏检大框架。
# - 原 solver 依赖边框颜色分组；这里必须改成依赖 size，否则 D1/1 没真正落地。
# - 原 solver 默认所有对象高度相同；这里的重叠判断必须显式吃入 size。
# - 原 solver 的 inward 列是写死的 5 与 W-10；这里虽然最终仍是固定两档，但其来源变成“组尺寸”，
#   因为左组对象宽 7、右组对象宽 5，而不是旧规则里的统一 5。
# - 检测阶段新增一个生成器配套假设：内部颜色不使用 2/8。这样 7x7 框内部的 5x5 区块不会被误识别成独立小框。
#
# d) 放置合法性
# - 若左侧允许两层 7x7，右侧允许两层 5x5，则 22 列会过窄；因此画布改为 26x26。
# - 输出布局采用 output-first：先决定每组哪些框贴边、哪些框 inward，再反推输入列顺序。
# - 输入与输出都要求对象不重叠、不越界；输入阶段还显式检查跨组矩形碰撞。
#
# e) 视觉可观察性
# - 解题者能直接看到：有红框被送到左边，也有青框被送到右边，因此颜色不再是可靠线索。
# - 同时还能直接看到：所有大框架都去左，所有小框架都去右，分组依据是可见的几何尺寸。
# - 排序只依赖原始列位置和可见的行重叠，没有使用颜色编号大小、生成顺序等不可见因素。
#
# f) 可视化整洁度
# - 对象仍是规则方框，内部纯色，边框厚 1，视觉上容易计数尺寸。
# - 26 列画布下，左组最大占用 0..13，右组最大占用 16..25，中间保留空带，输出不会挤成一团。
# - 同侧最多两层，避免多层级遮挡造成样例过乱。
#
# g) 变化的可推断性
# - 本融合版不做颜色映射，只改变“什么属性决定去向”。
# - 颜色种类仍然很少，且边框颜色只承担干扰项作用；真正稳定的结构线索是框架尺寸和边缘堆叠位置。
#
# h) main() 的新假设与 generator 的对应保证
# - main() 假设合法对象只会是 7x7 或 5x5 的单色边框方框；generator 只生成这两种。
# - main() 假设内部为单色，且不等于 bg、也不等于 2/8；generator 显式满足，避免尺寸嵌套误检。
# - main() 假设每个尺寸组最多两层(edge + inward)；generator 的输出行分配专门保证这一点。

from __future__ import annotations

import random

import numpy as np

from arc_utils import crop_bbox, get_bg, paste


LARGE = 7
SMALL = 5
FRAME_COLORS = (2, 8)
SIZES = (LARGE, SMALL)

GRID_H = 26
GRID_W = 26
N_LARGE_RANGE = (3, 4)
N_SMALL_RANGE = (3, 4)
MAX_INWARD = 2
INPUT_COL_RETRIES = 180
GEN_RETRIES = 1000


def _make_patch(size: int, frame_color: int, inner_color: int) -> np.ndarray:
    """Create a size x size square frame with a uniform interior."""
    patch = np.full((size, size), int(frame_color), dtype=int)
    patch[1:-1, 1:-1] = int(inner_color)
    return patch


def _rows_overlap(r_a: int, size_a: int, r_b: int, size_b: int) -> bool:
    """Return True when the two vertical spans intersect."""
    a1 = int(r_a + size_a - 1)
    b1 = int(r_b + size_b - 1)
    return not (a1 < int(r_b) or b1 < int(r_a))


def _rects_overlap(a: tuple[int, int, int], b: tuple[int, int, int]) -> bool:
    """Return True if two square placements overlap."""
    ar, ac, asz = map(int, a)
    br, bc, bsz = map(int, b)
    return not (
        ar + asz - 1 < br
        or br + bsz - 1 < ar
        or ac + asz - 1 < bc
        or bc + bsz - 1 < ac
    )


def _any_pairwise_overlap(entries: list[tuple[int, int, int]]) -> bool:
    """Check whether any two (row, col, size) placements overlap."""
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            if _rects_overlap(entries[i], entries[j]):
                return True
    return False


def _frame_border(patch: np.ndarray) -> np.ndarray:
    """Return the 1-pixel border of a square patch as a 1D array."""
    return np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])


def _detect_frames(grid: np.ndarray, bg: int) -> list[dict]:
    """Scan for 7x7 and 5x5 square frames with valid uniform interiors."""
    g = np.array(grid, dtype=int)
    H, W = g.shape
    found: list[dict] = []
    seen: set[tuple[int, int, int]] = set()

    for size in SIZES:
        for r0 in range(H - size + 1):
            for c0 in range(W - size + 1):
                patch = crop_bbox(g, (int(r0), int(c0), int(r0 + size - 1), int(c0 + size - 1)))
                border = _frame_border(patch)
                if int(border.min()) != int(border.max()):
                    continue
                frame_col = int(border[0])
                if frame_col not in FRAME_COLORS:
                    continue

                inner = patch[1:-1, 1:-1]
                if inner.size == 0:
                    continue
                inner_vals = np.unique(inner)
                if len(inner_vals) != 1:
                    continue
                inner_col = int(inner_vals[0])
                if inner_col in {bg, frame_col, *FRAME_COLORS}:
                    continue

                key = (int(r0), int(c0), int(size))
                if key in seen:
                    continue
                seen.add(key)
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
    Move mixed-size square frames to opposite edges according to their area class.

    Input structure:
    - The background is the most common color.
    - Scattered objects are square frames with a 1-cell border and a uniform interior.
    - Two visible sizes appear: 7x7 and 5x5; border colors are still red (2) and teal (8), and either color may occur in either size.

    Rule:
    - Large 7x7 frames form the left group and move to the left side; small 5x5 frames form the right group and move to the right side.
    - Each frame keeps its original top row.
    - Left-group frames are processed by increasing input column; right-group frames are processed by decreasing input column.
    - A frame with no row-overlap against an already edge-placed frame in its own size group sits on the edge; otherwise it shifts inward by one group width (7 for large, 5 for small).

    Key insight: The destination side is determined by visible frame size, while the old border colors survive only as distractors and no longer predict left versus right.
    """
    g = np.array(grid, dtype=int)
    _H, W = g.shape
    bg = int(get_bg(g))
    out = np.full_like(g, bg)

    frames = _detect_frames(g, bg)

    large_frames = sorted((f for f in frames if int(f["size"]) == LARGE), key=lambda f: int(f["c0"]))
    large_edge_rows: list[int] = []
    for frame in large_frames:
        r0 = int(frame["r0"])
        has_overlap = any(_rows_overlap(r0, LARGE, er, LARGE) for er in large_edge_rows)
        dest_c = int(LARGE) if has_overlap else 0
        paste(out, frame["patch"], r0, dest_c, transparent=bg)
        if dest_c == 0:
            large_edge_rows.append(r0)

    small_frames = sorted((f for f in frames if int(f["size"]) == SMALL), key=lambda f: -int(f["c0"]))
    small_edge_rows: list[int] = []
    for frame in small_frames:
        r0 = int(frame["r0"])
        has_overlap = any(_rows_overlap(r0, SMALL, er, SMALL) for er in small_edge_rows)
        dest_c = int(W - 2 * SMALL) if has_overlap else int(W - SMALL)
        paste(out, frame["patch"], r0, dest_c, transparent=bg)
        if dest_c == int(W - SMALL):
            small_edge_rows.append(r0)

    return out


def _assign_output_rows(n: int, size: int) -> tuple[list[int], list[int]] | None:
    """Pick non-overlapping edge rows and overlapping inward rows for one size group."""
    max_inward = min(MAX_INWARD, n - 1)
    n_inward = random.randint(1, max_inward)
    n_edge = n - n_inward

    candidates = list(range(GRID_H - size + 1))
    random.shuffle(candidates)
    edge_rows: list[int] = []
    for r0 in candidates:
        if all(not _rows_overlap(r0, size, er, size) for er in edge_rows):
            edge_rows.append(int(r0))
            if len(edge_rows) == n_edge:
                break
    if len(edge_rows) < n_edge:
        return None

    random.shuffle(candidates)
    inward_rows: list[int] = []
    for r0 in candidates:
        if not any(_rows_overlap(r0, size, er, size) for er in edge_rows):
            continue
        if any(_rows_overlap(r0, size, ir, size) for ir in inward_rows):
            continue
        inward_rows.append(int(r0))
        if len(inward_rows) == n_inward:
            break
    if len(inward_rows) < n_inward:
        return None

    return edge_rows, inward_rows


def _assign_frame_colors(n: int) -> list[int]:
    """Ensure both border colors appear in each size group."""
    colors = [FRAME_COLORS[0], FRAME_COLORS[1]]
    colors.extend(random.choice(FRAME_COLORS) for _ in range(n - 2))
    random.shuffle(colors)
    return [int(c) for c in colors]


def _assign_inner_colors(n: int, bg: int) -> list[int]:
    """Pick interior colors that cannot imitate a frame border."""
    pool = [c for c in range(10) if c not in {bg, *FRAME_COLORS}]
    if len(pool) < n:
        raise ValueError("Not enough inner colors for this configuration")
    return [int(c) for c in random.sample(pool, n)]


def _place_group_output(
    output: np.ndarray,
    size: int,
    edge_rows: list[int],
    inward_rows: list[int],
    frame_colors: list[int],
    inner_colors: list[int],
    edge_col: int,
    inward_col: int,
) -> list[dict]:
    """Paint one size group into the output and return placement metadata."""
    entries: list[dict] = []
    idx = 0
    for r0 in edge_rows:
        patch = _make_patch(size, frame_colors[idx], inner_colors[idx])
        paste(output, patch, int(r0), int(edge_col))
        entries.append(
            {
                "size": int(size),
                "frame": int(frame_colors[idx]),
                "r0": int(r0),
                "c0": int(edge_col),
                "is_edge": True,
                "patch": patch,
            }
        )
        idx += 1
    for r0 in inward_rows:
        patch = _make_patch(size, frame_colors[idx], inner_colors[idx])
        paste(output, patch, int(r0), int(inward_col))
        entries.append(
            {
                "size": int(size),
                "frame": int(frame_colors[idx]),
                "r0": int(r0),
                "c0": int(inward_col),
                "is_edge": False,
                "patch": patch,
            }
        )
        idx += 1
    return entries


def _build_processing_order(entries: list[dict]) -> list[dict]:
    """Create an input processing order where each inward item comes after an overlapping edge item."""
    edges = [e for e in entries if bool(e["is_edge"])]
    inwards = [e for e in entries if not bool(e["is_edge"])]
    size = int(entries[0]["size"])

    edge_indices = list(range(len(edges)))
    random.shuffle(edge_indices)
    order: list[tuple[str, int]] = [("edge", idx) for idx in edge_indices]

    inward_indices = list(range(len(inwards)))
    random.shuffle(inward_indices)
    for inward_idx in inward_indices:
        r0 = int(inwards[inward_idx]["r0"])
        deps = [
            idx
            for idx, edge in enumerate(edges)
            if _rows_overlap(r0, size, int(edge["r0"]), size)
        ]
        latest_pos = max(
            (
                pos
                for dep in deps
                for pos, (kind, idx) in enumerate(order)
                if kind == "edge" and idx == dep
            ),
            default=-1,
        )
        insert_pos = random.randint(latest_pos + 1, len(order))
        order.insert(insert_pos, ("inward", inward_idx))

    return [edges[idx] if kind == "edge" else inwards[idx] for kind, idx in order]


def _derive_input_positions(
    entries: list[dict],
    ascending: bool,
    forbidden: list[tuple[int, int, int]] | None = None,
) -> list[tuple[int, int, int, np.ndarray]] | None:
    """Assign input columns that reproduce the intended solver order and avoid overlap."""
    ordered = _build_processing_order(entries)
    size = int(entries[0]["size"])

    for _ in range(INPUT_COL_RETRIES):
        placed: list[tuple[int, int, int, np.ndarray]] = []
        ok = True
        for idx, entry in enumerate(ordered):
            r0 = int(entry["r0"])
            out_c = int(entry["c0"])

            if ascending:
                lower = 0 if idx == 0 else int(placed[-1][1]) + 1
                upper = GRID_W - size
                for prev_r0, prev_c0, prev_size, _prev_patch in placed:
                    if _rows_overlap(r0, size, int(prev_r0), int(prev_size)):
                        lower = max(lower, int(prev_c0) + size)
            else:
                lower = 0
                upper = GRID_W - size if idx == 0 else int(placed[-1][1]) - 1
                for prev_r0, prev_c0, prev_size, _prev_patch in placed:
                    if _rows_overlap(r0, size, int(prev_r0), int(prev_size)):
                        upper = min(upper, int(prev_c0) - size)

            if lower > upper:
                ok = False
                break

            candidates = [c for c in range(lower, upper + 1) if c != out_c]
            if forbidden is not None:
                candidates = [
                    c
                    for c in candidates
                    if all(
                        not _rects_overlap((r0, int(c), size), (fr, fc, fs))
                        for fr, fc, fs in forbidden
                    )
                ]
            if not candidates:
                ok = False
                break

            c0 = int(random.choice(candidates))
            placed.append((r0, c0, size, entry["patch"]))

        if ok and not _any_pairwise_overlap([(r, c, s) for r, c, s, _p in placed]):
            return placed

    return None


def _gen_attempt() -> np.ndarray | None:
    """Build one valid input grid by generating the desired output first."""
    bg = int(random.choice([c for c in range(10) if c not in FRAME_COLORS]))
    n_large = int(random.randint(*N_LARGE_RANGE))
    n_small = int(random.randint(*N_SMALL_RANGE))

    large_rows = _assign_output_rows(n_large, LARGE)
    if large_rows is None:
        return None
    small_rows = _assign_output_rows(n_small, SMALL)
    if small_rows is None:
        return None
    large_edge_rows, large_inward_rows = large_rows
    small_edge_rows, small_inward_rows = small_rows

    output = np.full((GRID_H, GRID_W), bg, dtype=int)

    large_frames = _assign_frame_colors(n_large)
    small_frames = _assign_frame_colors(n_small)
    large_inners = _assign_inner_colors(n_large, bg)
    small_inners = _assign_inner_colors(n_small, bg)

    large_output = _place_group_output(
        output,
        LARGE,
        large_edge_rows,
        large_inward_rows,
        large_frames,
        large_inners,
        edge_col=0,
        inward_col=LARGE,
    )
    small_output = _place_group_output(
        output,
        SMALL,
        small_edge_rows,
        small_inward_rows,
        small_frames,
        small_inners,
        edge_col=GRID_W - SMALL,
        inward_col=GRID_W - 2 * SMALL,
    )

    large_input = _derive_input_positions(large_output, ascending=True)
    if large_input is None:
        return None
    large_rects = [(int(r), int(c), int(s)) for r, c, s, _patch in large_input]

    small_input = _derive_input_positions(small_output, ascending=False, forbidden=large_rects)
    if small_input is None:
        return None
    all_rects = large_rects + [(int(r), int(c), int(s)) for r, c, s, _patch in small_input]
    if _any_pairwise_overlap(all_rects):
        return None

    inp = np.full((GRID_H, GRID_W), bg, dtype=int)
    for r0, c0, _size, patch in large_input + small_input:
        paste(inp, patch, int(r0), int(c0))

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
