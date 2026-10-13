from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


BG = 0
_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_SHAPE_POOL: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
    ((0, 1), (1, 1), (2, 0), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0)),
    ((0, 1), (0, 2), (1, 0), (1, 1), (2, 0), (2, 1)),
)
_BRANCH_LABELS = (1, 2, 3)
_GEN_RETRIES = 400


def _normalize(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0 = int(rows.min())
    c0 = int(cols.min())
    return tuple(sorted((int(r) - r0, int(c) - c0) for r, c in coords))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return (r0 + r1) // 2, (c0 + c1) // 2


def _find_components(grid: np.ndarray, bg: int = BG) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in _NEIGHBORS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.asarray(cells, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                }
            )
    return comps


def _palette_sequence(grid: np.ndarray) -> list[int]:
    row = [int(v) for v in np.asarray(grid[-1], dtype=int).tolist()]
    palette = [v for v in row if v != BG]
    assert palette, "palette row must contain at least one color"
    assert len(palette) == len(set(palette)), "palette colors must be unique"
    return palette


def _polyline_cells(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    sr, sc = start
    er, ec = end
    cells: list[tuple[int, int]] = []
    dc = 0 if sc == ec else (1 if ec > sc else -1)
    c = sc
    while True:
        cells.append((sr, c))
        if c == ec:
            break
        c += dc
    dr = 0 if sr == er else (1 if er > sr else -1)
    r = sr + dr
    while dr != 0 and True:
        cells.append((r, ec))
        if r == er:
            break
        r += dr
    return cells


def _stride_for_gap(gap: int) -> int:
    if gap == 1:
        return 1
    if gap == 2:
        return 2
    return 3


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each pair of equal-shape objects using a palette-index-dependent polyline style.

    Input structure:
    - The background is black, and the bottom row contains a contiguous palette sequence of distinct colors.
    - Above the palette are several small 4-connected monochrome objects whose colors are chosen from that palette.
    - Objects are grouped by translation-normalized shape, and each shape appears exactly twice in different colors.

    Rule:
    - Read the non-black colors on the bottom row from left to right as palette indices.
    - Pair the upper objects by exact translation-normalized shape, ignoring color.
    - For each pair, compare the two colors' palette indices: use the earlier palette color as the line color and let the index gap decide the style.
    - Draw a horizontal-then-vertical polyline between the two bbox centers: solid for gap 1, alternating dashed for gap 2, and one-pixel-every-three sparse dots for larger gaps.
    - Keep the original objects and the palette row unchanged, and only paint line pixels onto black cells.

    Key insight:
    - Shape identity determines the pairing, while the bottom palette row acts as a lookup table whose left-to-right spacing controls both the line color and its stroke pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 3
    palette = _palette_sequence(g)
    color_to_index = {color: idx for idx, color in enumerate(palette)}

    out = g.copy()
    comps = _find_components(g[:-1, :], bg=BG)
    grouped: dict[tuple[tuple[int, int], ...], list[dict[str, object]]] = defaultdict(list)
    for comp in comps:
        key = _normalize(np.asarray(comp["coords"], dtype=int))
        grouped[key].append(comp)

    pairs: list[tuple[tuple[int, int], tuple[int, int], int, int]] = []
    for group in grouped.values():
        assert len(group) == 2, "every shape must appear exactly twice"
        first, second = group
        color_a = int(first["color"])
        color_b = int(second["color"])
        assert color_a != color_b
        assert color_a in color_to_index and color_b in color_to_index
        idx_a = color_to_index[color_a]
        idx_b = color_to_index[color_b]
        if idx_a <= idx_b:
            start_comp = first
            end_comp = second
            gap = idx_b - idx_a
            line_color = color_a
        else:
            start_comp = second
            end_comp = first
            gap = idx_a - idx_b
            line_color = color_b
        assert gap >= 1
        start = _bbox_center(start_comp["bbox"])  # type: ignore[arg-type]
        end = _bbox_center(end_comp["bbox"])  # type: ignore[arg-type]
        pairs.append((start, end, gap, line_color))

    pairs.sort(key=lambda item: (item[0][0], item[0][1], item[1][0], item[1][1]))
    for start, end, gap, line_color in pairs:
        stride = _stride_for_gap(gap)
        for idx, (r, c) in enumerate(_polyline_cells(start, end)):
            if idx % stride != 0:
                continue
            if int(out[r, c]) == BG:
                out[r, c] = line_color

    return out


def _shift_cells(shape: tuple[tuple[int, int], ...], top: int, left: int) -> list[tuple[int, int]]:
    return [(top + r, left + c) for r, c in shape]


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _pair_type(pair: tuple[int, int]) -> int:
    gap = pair[1] - pair[0]
    if gap == 1:
        return 1
    if gap == 2:
        return 2
    return 3


def _choose_pairs(palette_len: int, pair_count: int, rng: random.Random) -> list[tuple[int, int]]:
    by_type: dict[int, list[tuple[int, int]]] = {1: [], 2: [], 3: []}
    for i in range(palette_len):
        for j in range(i + 1, palette_len):
            by_type[_pair_type((i, j))].append((i, j))

    base_choices: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []
    best_score = -1
    for p1 in by_type[1]:
        for p2 in by_type[2]:
            for p3 in by_type[3]:
                pairs = (p1, p2, p3)
                used = {idx for pair in pairs for idx in pair}
                score = len(used)
                if score > best_score:
                    best_score = score
                    base_choices = [pairs]
                elif score == best_score:
                    base_choices.append(pairs)
    chosen = list(rng.choice(base_choices))
    if pair_count > 3:
        extra_type = rng.choice(list(_BRANCH_LABELS))
        extra_pool = [pair for pair in by_type[extra_type] if pair not in chosen]
        if not extra_pool:
            extra_pool = by_type[extra_type]
        chosen.append(rng.choice(extra_pool))
    rng.shuffle(chosen)
    return chosen


def _draw_object(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _visible_line_count(
    start: tuple[int, int],
    end: tuple[int, int],
    stride: int,
    blocked: set[tuple[int, int]],
) -> int:
    count = 0
    for idx, cell in enumerate(_polyline_cells(start, end)):
        if idx % stride == 0 and cell not in blocked:
            count += 1
    return count


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.choice((3, 4))
    palette_len = rng.randint(7, 9)
    top_margin = rng.randint(0, 2)
    bottom_gap = rng.randint(1, 2)
    height = top_margin + pair_count * 4 + (pair_count - 1) + bottom_gap + 1
    width = rng.randint(max(18, palette_len + 7), 28)
    if height > 30 or width > 30:
        return None

    grid = np.zeros((height, width), dtype=int)
    palette_colors = rng.sample(list(range(1, 10)), palette_len)
    palette_start = rng.randint(0, width - palette_len)
    grid[-1, palette_start : palette_start + palette_len] = palette_colors

    pairs = _choose_pairs(palette_len, pair_count, rng)
    shapes = rng.sample(list(_SHAPE_POOL), pair_count)
    specs = list(zip(pairs, shapes))
    rng.shuffle(specs)

    occupied = np.zeros((height - 1, width), dtype=bool)
    band_tops = [top_margin + i * 5 for i in range(pair_count)]

    for band_top, (pair, shape) in zip(band_tops, specs):
        shape_h, shape_w = _shape_dims(shape)
        band_height = 4
        if shape_h > band_height:
            return None

        start_on_left = bool(rng.randint(0, 1))
        lower_idx, upper_idx = pair
        left_color = palette_colors[lower_idx] if start_on_left else palette_colors[upper_idx]
        right_color = palette_colors[upper_idx] if start_on_left else palette_colors[lower_idx]

        left_col_max = width // 2 - shape_w - 3
        right_col_min = width // 2 + 2
        right_col_max = width - shape_w - 2
        if left_col_max < 1 or right_col_min > right_col_max:
            return None

        left_col = rng.randint(1, left_col_max)
        right_col = rng.randint(right_col_min, right_col_max)
        top_a = band_top + rng.randint(0, band_height - shape_h)
        top_b = band_top + rng.randint(0, band_height - shape_h)

        left_cells = _shift_cells(shape, top_a, left_col)
        right_cells = _shift_cells(shape, top_b, right_col)
        if any(occupied[r, c] for r, c in left_cells + right_cells):
            return None

        start_cells = left_cells if start_on_left else right_cells
        end_cells = right_cells if start_on_left else left_cells
        start_bbox = _bbox(np.asarray(start_cells, dtype=int))
        end_bbox = _bbox(np.asarray(end_cells, dtype=int))
        start_center = _bbox_center(start_bbox)
        end_center = _bbox_center(end_bbox)
        blocked = set(start_cells) | set(end_cells)
        stride = _stride_for_gap(abs(upper_idx - lower_idx))
        if _visible_line_count(start_center, end_center, stride, blocked) < 2:
            return None

        _draw_object(grid, left_cells, left_color)
        _draw_object(grid, right_cells, right_color)
        for r, c in left_cells + right_cells:
            occupied[r, c] = True

    comps = _find_components(grid[:-1, :], bg=BG)
    if len(comps) != pair_count * 2:
        return None
    grouped: dict[tuple[tuple[int, int], ...], int] = defaultdict(int)
    for comp in comps:
        grouped[_normalize(np.asarray(comp["coords"], dtype=int))] += 1
    if sorted(grouped.values()) != [2] * pair_count:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
