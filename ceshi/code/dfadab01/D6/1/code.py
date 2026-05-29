# Variant of dfadab01
# Dimension: D6 — 网格边缘裁剪规则 (grid_shape)
# Change: 字形放置时如果超出网格边界，超出部分被裁剪（只显示网格内的部分）... → 环绕放置（toroidal）

from __future__ import annotations

import random

import numpy as np


# Design analysis
# ---------------
# spec:
#   baseline_preserved:
#     - 标记颜色集合、4x4 固定字形字典、图例结构、标签位于图例右下角 (+4,+4) 的语义保持不变。
#     - 输入仍以单像素 marker 为主，输出仍是“marker -> 固定 4x4 图案”的替换任务。
#   source_task_calibration:
#     - 参考 88bcf3b4 后，本变体只引入“越界后从对边继续”的局部环绕机制，不额外加入路径搜索、方向推断等复杂度。
#     - 因此 solver 的核心变化应集中在字形绘制坐标映射，而不是识别逻辑的大改写。
#   concrete_variant:
#     - 当 marker 的 4x4 字形越过底边或右边时，越界像素通过模运算落到顶部或左侧。
#     - 例如 10x10 网格中，marker 2 放在 (8, 7)：
#       - 原题 clipped: 只会显示行 8..9、列 7..9 内的字形残片。
#       - 本变体 toroidal: 行 10..11 回到 0..1，列 10 回到 0，所以同一个 4x4 字形会拆成上下/左右多块。
#   flow_impact:
#     - 影响环节是“把 marker 展开为 4x4 字形”的坐标写入阶段。
#     - 图例标签判定仍基于一个完整、非环绕的 4x4 样例；生成器会把图例固定在网格内部，从而把 D6 改动限定在真实 marker 的输出上。
#   complexity_calibration:
#     - 若仍允许 10x10 上出现 3 个以上 toroidal 行/列槽位，则 4x4 边界框会在环面上相互重叠，破坏原题“字形互不覆盖”的清晰性。
#     - 因此生成器把每个维度的槽位数限制为 floor(size/4)：10x10 只能 2x2，20x20 可 2x2 到 5x5。
#     - 同一输入中强制至少一个 wrapped marker，且至少一个同色 interior marker，保证人眼能对照看出“同一字形在边缘会绕回去”。
#   implicit_assumptions_review:
#     - _has_source_glyph 原本依赖“图例样例完整在界内”；该前提在变体中对图例仍成立，因为生成器显式保证图例 4x4+label 不触边。
#     - 原 _paint_clipped 依赖“越界像素直接丢弃”；该前提失效，必须改为对 h/w 取模。
#     - 主循环按 marker 逐个上色仍安全，前提是不同 marker 的 toroidal 4x4 边界框不重叠；生成器会构造性保证这一点。
#   placement_legality:
#     - 行/列槽位按环面上的近均匀间隔构造，任意两槽位循环距离都 >= 4，因此对应 4x4 边界框互不相交。
#     - 图例占用一个普通 5x5 内部窗口（4x4 glyph + label），不与任何真实 marker 位置冲突。
#   observability_and_cleanliness:
#     - 不使用颜色 id 做任何隐藏排序或变换；颜色映射仍由固定字形字典决定。
#     - wrapped 样例会在顶/底或左/右边界出现明显“分裂”的同色图案，且同图中保留未环绕的同型 marker 作为对照。
#   solver_assumptions_after_change:
#     - 输入至少包含一个真实 marker。
#     - 图例若出现，必定是完整的 4x4 样例加右下标签，不使用环绕图例。
#     - 真实 marker 的 toroidal 字形边界框互不重叠，因此输出写入次序不会改变结果。


GLYPHS: dict[int, tuple[int, np.ndarray]] = {
    2: (
        4,
        np.array(
            [
                [1, 1, 1, 1],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [1, 1, 1, 1],
            ],
            dtype=bool,
        ),
    ),
    3: (
        1,
        np.array(
            [
                [0, 1, 1, 0],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [0, 1, 1, 0],
            ],
            dtype=bool,
        ),
    ),
    5: (
        6,
        np.array(
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ],
            dtype=bool,
        ),
    ),
    8: (
        7,
        np.array(
            [
                [1, 0, 0, 1],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [1, 0, 0, 1],
            ],
            dtype=bool,
        ),
    ),
}


def _has_source_glyph(g: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> bool:
    h, w = g.shape
    if r0 < 0 or c0 < 0 or r0 + 4 > h or c0 + 4 > w:
        return False
    patch = g[r0 : r0 + 4, c0 : c0 + 4]
    return bool(np.all(patch[mask] == color) and np.all(patch[~mask] != color))


def _paint_clipped(out: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    h, w = out.shape
    for dr in range(4):
        rr = r0 + dr
        if rr < 0 or rr >= h:
            continue
        for dc in range(4):
            cc = c0 + dc
            if 0 <= cc < w and bool(mask[dr, dc]):
                out[rr, cc] = color


def _paint_toroidal(out: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    h, w = out.shape
    for dr in range(4):
        for dc in range(4):
            if bool(mask[dr, dc]):
                out[(r0 + dr) % h, (c0 + dc) % w] = color


def _marker_wraps(r0: int, c0: int, mask: np.ndarray, h: int, w: int) -> bool:
    for dr in range(4):
        for dc in range(4):
            if bool(mask[dr, dc]) and (r0 + dr >= h or c0 + dc >= w):
                return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each colored control pixel into its corresponding fixed 4x4 glyph on a torus.

    Input structure:
    - Black background with single-cell markers in colors 2, 3, 5, and/or 8.
    - Some inputs also contain one internal legend: a 4x4 sample glyph in its output color, plus a same-type marker one row and one column beyond the sample's lower-right corner.

    Rule:
    - Replace every real marker with its learned 4x4 glyph: 2->yellow frame, 3->blue ring, 5->magenta diagonal blocks, 8->orange X.
    - Anchor the glyph's top-left corner at the marker cell; if any glyph pixels run past the bottom or right edge, continue them from the opposite side of the grid.
    - Remove legend glyphs from the output and ignore the legend label marker instead of expanding it.

    Key insight:
    - The marker colors still select from a fixed glyph alphabet; the only new behavior is that glyph coordinates live on a wraparound surface rather than a clipped rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(np.unique(g)).issubset({0, *GLYPHS.keys(), 1, 4, 6, 7})

    out = np.zeros_like(g)
    seen_marker = False
    for marker_color, (glyph_color, mask) in GLYPHS.items():
        coords = np.argwhere(g == marker_color)
        if coords.size:
            seen_marker = True
        for r, c in coords:
            if _has_source_glyph(g, int(r) - 4, int(c) - 4, glyph_color, mask):
                continue
            _paint_toroidal(out, int(r), int(c), glyph_color, mask)

    assert seen_marker
    return out


def _cyclic_positions(size: int, count: int) -> list[int]:
    assert 2 <= count <= size // 4
    base = size // count
    slack = size % count
    gap_ids = list(range(count))
    random.shuffle(gap_ids)
    gaps = [base] * count
    for idx in gap_ids[:slack]:
        gaps[idx] += 1

    for _ in range(200):
        offset = random.randrange(size)
        positions = [offset]
        cur = offset
        for gap in gaps[:-1]:
            cur = (cur + gap) % size
            positions.append(cur)
        positions = sorted(positions)
        has_interior = any(p <= size - 4 for p in positions)
        has_edge = any(p >= size - 3 for p in positions)
        if has_interior and has_edge:
            return positions
    raise AssertionError("failed to build positions with both interior and edge slots")


def _expected_output(markers: dict[tuple[int, int], int], size: int, toroidal: bool) -> np.ndarray:
    out = np.zeros((size, size), dtype=int)
    for (r, c), marker_color in markers.items():
        glyph_color, mask = GLYPHS[marker_color]
        if toroidal:
            _paint_toroidal(out, r, c, glyph_color, mask)
        else:
            _paint_clipped(out, r, c, glyph_color, mask)
    return out


def _try_place_legend(inp: np.ndarray, used_colors: list[int]) -> bool:
    size = inp.shape[0]
    candidates = list(used_colors)
    random.shuffle(candidates)
    for marker_color in candidates:
        glyph_color, mask = GLYPHS[marker_color]
        for _ in range(80):
            r0 = random.randrange(size - 4)
            c0 = random.randrange(size - 4)
            bbox_clear = True
            for rr in range(r0, r0 + 5):
                for cc in range(c0, c0 + 5):
                    if int(inp[rr, cc]) != 0:
                        bbox_clear = False
                        break
                if not bbox_clear:
                    break
            if not bbox_clear:
                continue
            for dr in range(4):
                for dc in range(4):
                    if bool(mask[dr, dc]):
                        inp[r0 + dr, c0 + dc] = glyph_color
            inp[r0 + 4, c0 + 4] = marker_color
            return True
    return False


def generate_input() -> np.ndarray:
    for _ in range(400):
        size = random.choice([10, 20])
        max_slots = size // 4
        row_count = random.randint(2, max_slots)
        col_count = random.randint(2, max_slots)
        rows = _cyclic_positions(size, row_count)
        cols = _cyclic_positions(size, col_count)

        cells = [(r, c) for r in rows for c in cols]
        wrap_cells = [(r, c) for r, c in cells if r >= size - 3 or c >= size - 3]
        interior_cells = [(r, c) for r, c in cells if r <= size - 4 and c <= size - 4]
        if not wrap_cells or not interior_cells:
            continue

        fill_min = (6 * len(cells) + 9) // 10
        fill_count = random.randint(max(3, fill_min), len(cells))
        max_used = min(4, fill_count - 1)
        used_count = random.randint(1, max_used)
        used_colors = random.sample(list(GLYPHS.keys()), used_count)
        focus_color = random.choice(used_colors)

        wrapped_focus = random.choice(wrap_cells)
        interior_focus = random.choice(interior_cells)
        selected = {wrapped_focus, interior_focus}
        remaining = [cell for cell in cells if cell not in selected]
        random.shuffle(remaining)
        for cell in remaining[: fill_count - len(selected)]:
            selected.add(cell)

        markers: dict[tuple[int, int], int] = {
            wrapped_focus: focus_color,
            interior_focus: focus_color,
        }
        unassigned = [cell for cell in selected if cell not in markers]
        random.shuffle(unassigned)
        for marker_color in used_colors:
            if marker_color == focus_color:
                continue
            if not unassigned:
                break
            markers[unassigned.pop()] = marker_color
        for cell in unassigned:
            markers[cell] = random.choice(used_colors)

        wrapped_real = sum(
            _marker_wraps(r, c, GLYPHS[marker_color][1], size, size)
            for (r, c), marker_color in markers.items()
        )
        if wrapped_real == 0:
            continue

        toroidal_out = _expected_output(markers, size, toroidal=True)
        clipped_out = _expected_output(markers, size, toroidal=False)
        if np.array_equal(toroidal_out, clipped_out):
            continue

        inp = np.zeros((size, size), dtype=int)
        for (r, c), marker_color in markers.items():
            inp[r, c] = marker_color

        if random.random() < 0.6:
            _try_place_legend(inp, used_colors)

        solved = main(inp)
        assert np.array_equal(solved, toroidal_out)
        return inp

    raise AssertionError("generate_input failed after retries")
