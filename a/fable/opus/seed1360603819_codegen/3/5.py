from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
STAMP_HOLES = {
    1: ((1, 1),),
    2: ((1, 0), (1, 2)),
    3: ((0, 0), (1, 1), (2, 2)),
    4: ((0, 0), (0, 2), (2, 0), (2, 2)),
}


def _unique_most_common(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    max_count = max(counts.values())
    winners = [color for color, count in counts.items() if count == max_count]
    assert len(winners) == 1, "expected a unique most common color"
    return int(winners[0])


def _boundary_bg(grid: np.ndarray) -> int:
    boundary = np.concatenate((grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]))
    return _unique_most_common(boundary)


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _components(mask: np.ndarray) -> list[dict[str, object]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append({"coords": arr, "bbox": _bbox(arr), "area": int(len(coords))})
    return comps


def _row_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return max(int(a[0]), int(b[0])) <= min(int(a[2]), int(b[2]))


def _map_row(row: int, src_h: int, dst_h: int) -> int:
    top = int(row)
    bottom = int(src_h - 1 - row)
    assert top != bottom, "marker row must be closer to one horizontal edge"
    mapped = top if top < bottom else int(dst_h - 1 - bottom)
    assert 0 <= mapped < dst_h, "mapped row must fit in the target rectangle"
    return int(mapped)


def _map_col_run(c0: int, c1: int, src_w: int, dst_w: int) -> tuple[int, int]:
    mid2 = int(src_w - 1)
    if c0 == c1:
        left = int(c0)
        right = int(src_w - 1 - c0)
        assert left != right, "marker column must be closer to one vertical edge"
        mapped = left if left < right else int(dst_w - 1 - right)
        assert 0 <= mapped < dst_w, "mapped column must fit in the target rectangle"
        return int(mapped), int(mapped)

    if 2 * int(c1) <= mid2:
        start, end = int(c0), int(c1)
    elif 2 * int(c0) >= mid2:
        start = int(dst_w - 1 - (src_w - 1 - c0))
        end = int(dst_w - 1 - (src_w - 1 - c1))
    else:
        start = int(c0)
        end = int(dst_w - 1 - (src_w - 1 - c1))

    assert 0 <= start <= end < dst_w, "mapped run must fit in the target rectangle"
    return int(start), int(end)


def _marker_runs(row: np.ndarray, base_color: int) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    c = 0
    while c < row.size:
        color = int(row[c])
        if color == int(base_color):
            c += 1
            continue
        start = c
        while c + 1 < row.size and int(row[c + 1]) == color:
            c += 1
        runs.append((int(start), int(c), color))
        c += 1
    return runs


def _adapt_colored_marks(src: np.ndarray, base_color: int, dst_shape: tuple[int, int]) -> np.ndarray:
    src_h, src_w = map(int, src.shape)
    dst_h, dst_w = map(int, dst_shape)
    out = np.full((dst_h, dst_w), -1, dtype=int)
    found = False
    for row in range(src_h):
        runs = _marker_runs(src[row], base_color)
        if not runs:
            continue
        found = True
        dst_row = _map_row(row, src_h, dst_h)
        for c0, c1, color in runs:
            dst_c0, dst_c1 = _map_col_run(c0, c1, src_w, dst_w)
            existing = out[dst_row, dst_c0 : dst_c1 + 1]
            assert np.all((existing == -1) | (existing == color)), "conflicting mapped marker colors"
            out[dst_row, dst_c0 : dst_c1 + 1] = int(color)
    assert found, "expected marker pixels in source rectangle"
    return out


def _parse_scene(grid: np.ndarray, bg: int) -> tuple[dict[int, int], list[dict[str, object]]]:
    legend: dict[int, int] = {}
    rectangles: list[dict[str, object]] = []
    for comp in _components(grid != int(bg)):
        bbox = comp["bbox"]
        sub = _crop_bbox(grid, bbox)
        colors = set(map(int, np.unique(sub)))
        if BLACK in colors:
            colors.discard(int(bg))
            assert len(colors - {BLACK}) == 1, "legend entries must use one non-black color"
            color = int(next(iter(colors - {BLACK})))
            assert np.all((sub == BLACK) | (sub == color)), "legend entry must only contain color and black"
            assert sub.shape[0] == sub.shape[1] and sub.shape[0] >= 3, "legend entry must be a square"
            holes = int(np.sum(sub == BLACK))
            assert 1 <= holes <= 4, "legend hole count must match a known stamp"
            assert color not in legend, "legend colors must be unique"
            legend[color] = holes
            continue

        assert np.all(sub != int(bg)), "main rectangles must be dense"
        base_color = _unique_most_common(sub)
        marker_colors = set(map(int, sub.reshape(-1))) - {int(base_color)}
        rectangles.append(
            {
                "bbox": bbox,
                "sub": sub,
                "base_color": int(base_color),
                "marker_colors": marker_colors,
            }
        )

    assert legend, "expected a legend"
    assert rectangles, "expected at least one main rectangle"
    return legend, rectangles


def _apply_stamp(out: np.ndarray, center_r: int, center_c: int, color: int, holes: int) -> None:
    h, w = out.shape
    assert 1 <= center_r < h - 1 and 1 <= center_c < w - 1, "stamp center must fit a 3x3 block"
    patch = out[center_r - 1 : center_r + 2, center_c - 1 : center_c + 2]
    patch[:, :] = int(color)
    for dr, dc in STAMP_HOLES[int(holes)]:
        patch[int(dr), int(dc)] = BLACK


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the unique plain rectangle, infer the shared border-relative colored marker pattern
    from row-aligned marked rectangles, then replace each transferred marker pixel with a
    3x3 stamp whose black-hole count is given by the corner legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _boundary_bg(g)
    legend, rectangles = _parse_scene(g, bg)

    plain = [obj for obj in rectangles if not obj["marker_colors"]]
    assert len(plain) == 1, "expected exactly one plain rectangle"
    target = plain[0]
    tr0, tc0, tr1, tc1 = target["bbox"]
    target_shape = (int(tr1 - tr0 + 1), int(tc1 - tc0 + 1))

    sources = [
        obj
        for obj in rectangles
        if obj["marker_colors"] and _row_overlap(obj["bbox"], target["bbox"])
    ]
    assert sources, "expected at least one row-aligned marked rectangle"

    patterns = [_adapt_colored_marks(obj["sub"], obj["base_color"], target_shape) for obj in sources]
    pattern = patterns[0]
    for other in patterns[1:]:
        assert np.array_equal(other, pattern), "aligned sources must encode the same colored pattern"

    out = np.full(target_shape, int(target["base_color"]), dtype=int)
    for r, c in np.argwhere(pattern != -1):
        color = int(pattern[r, c])
        assert color in legend, "every marker color must appear in the legend"
        _apply_stamp(out, int(r), int(c), color, int(legend[color]))
    return out


GEN_BG = 5
GEN_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 400
PLAIN_H_RANGE = (7, 12)
PLAIN_W_RANGE = (7, 12)
RECT_H_RANGE = (7, 16)
RECT_W_RANGE = (7, 16)
OVERLAP_SOURCE_COUNT_RANGE = (2, 3)
EXTRA_RECT_COUNT_RANGE = (0, 2)
MARK_COLOR_COUNT_OPTIONS = (2, 3, 4)
MARK_COUNT_RANGE = (2, 6)
BORDER_GAP = 1
ITEM_GAP = 1
BAND_GAP = 2
LEGEND_GAP = 2
LEGEND_ENTRY_GAP = 1
GRID_H_MAX = 40
GRID_W_MAX = 40


def _safe_centers(height: int, width: int) -> list[tuple[int, int]]:
    return [
        (int(r), int(c))
        for r in range(1, int(height) - 1)
        for c in range(1, int(width) - 1)
        if int(r) != int(height - 1 - r) and int(c) != int(width - 1 - c)
    ]


def _far_enough(chosen: list[tuple[int, int]], cand: tuple[int, int]) -> bool:
    return all(max(abs(cand[0] - r), abs(cand[1] - c)) >= 3 for r, c in chosen)


def _choose_centers(height: int, width: int, count: int) -> list[tuple[int, int]] | None:
    candidates = _safe_centers(height, width)
    if len(candidates) < count:
        return None
    for _ in range(50):
        random.shuffle(candidates)
        chosen: list[tuple[int, int]] = []
        for cand in candidates:
            if _far_enough(chosen, cand):
                chosen.append(cand)
                if len(chosen) == count:
                    return chosen
    return None


def _pick_dim(near_vals: list[int], lo: int, hi: int) -> int:
    min_dim = max(int(lo), max(2 * int(v) + 2 for v in near_vals))
    if min_dim > int(hi):
        return -1
    return int(random.randint(min_dim, int(hi)))


def _target_to_source_coord(target_idx: int, target_len: int, source_len: int) -> int:
    near_low = int(target_idx)
    near_high = int(target_len - 1 - target_idx)
    assert near_low != near_high, "target marker must be closer to one edge"
    return int(near_low if near_low < near_high else source_len - 1 - near_high)


def _pattern_grid(shape: tuple[int, int], centers: list[tuple[int, int]], colors: list[int]) -> np.ndarray:
    out = np.full(shape, -1, dtype=int)
    for (r, c), color in zip(centers, colors, strict=True):
        out[int(r), int(c)] = int(color)
    return out


def _make_marked_rect(
    height: int,
    width: int,
    base_color: int,
    target_shape: tuple[int, int],
    centers: list[tuple[int, int]],
    colors: list[int],
) -> np.ndarray:
    out = np.full((int(height), int(width)), int(base_color), dtype=int)
    target_h, target_w = map(int, target_shape)
    for (tr, tc), color in zip(centers, colors, strict=True):
        sr = _target_to_source_coord(int(tr), target_h, int(height))
        sc = _target_to_source_coord(int(tc), target_w, int(width))
        out[int(sr), int(sc)] = int(color)
    return out


def _legend_entry(color: int, holes: int) -> np.ndarray:
    entry = np.full((3, 3), int(color), dtype=int)
    for r, c in STAMP_HOLES[int(holes)]:
        entry[int(r), int(c)] = BLACK
    return entry


def _render_legend(
    grid: np.ndarray,
    marker_colors: list[int],
    hole_map: dict[int, int],
    top_side: bool,
    left_side: bool,
    vertical: bool,
) -> None:
    entries = [_legend_entry(color, hole_map[color]) for color in marker_colors]
    if vertical:
        legend_h = len(entries) * 3 + (len(entries) - 1) * LEGEND_ENTRY_GAP
        legend_w = 3
    else:
        legend_h = 3
        legend_w = len(entries) * 3 + (len(entries) - 1) * LEGEND_ENTRY_GAP

    r0 = BORDER_GAP if top_side else grid.shape[0] - BORDER_GAP - legend_h
    c0 = BORDER_GAP if left_side else grid.shape[1] - BORDER_GAP - legend_w
    cursor_r = int(r0)
    cursor_c = int(c0)
    for entry in entries:
        eh, ew = entry.shape
        grid[cursor_r : cursor_r + eh, cursor_c : cursor_c + ew] = entry
        if vertical:
            cursor_r += eh + LEGEND_ENTRY_GAP
        else:
            cursor_c += ew + LEGEND_ENTRY_GAP


def _make_overlap_items(rects: list[np.ndarray], plain_index: int) -> list[tuple[int, np.ndarray]] | None:
    band_h = max(int(rect.shape[0]) for rect in rects)
    items = [(0, rect) for rect in rects]
    order = list(range(len(items)))
    random.shuffle(order)
    items = [items[idx] for idx in order]
    plain_pos = order.index(int(plain_index))
    plain_rect = items[plain_pos][1]
    plain_h = int(plain_rect.shape[0])

    offset_p = random.randint(0, max(0, band_h - plain_h))
    items[plain_pos] = (offset_p, plain_rect)
    plain_r0 = int(offset_p)
    plain_r1 = int(offset_p + plain_h - 1)

    for idx, (_off, rect) in enumerate(items):
        if idx == plain_pos:
            continue
        rect_h = int(rect.shape[0])
        lo = max(0, plain_r0 - rect_h + 1)
        hi = min(band_h - rect_h, plain_r1)
        if lo > hi:
            return None
        items[idx] = (random.randint(lo, hi), rect)
    return items


def _render_band(grid: np.ndarray, top: int, items: list[tuple[int, np.ndarray]]) -> tuple[int, int, int]:
    band_h = max(int(off + rect.shape[0]) for off, rect in items)
    band_w = sum(int(rect.shape[1]) for _off, rect in items) + ITEM_GAP * (len(items) - 1)
    max_left = grid.shape[1] - BORDER_GAP - band_w
    min_left = BORDER_GAP
    assert min_left <= max_left, "band must fit horizontally"
    left = random.randint(min_left, max_left)
    cur_c = int(left)
    for off, rect in items:
        h, w = rect.shape
        r0 = int(top + off)
        grid[r0 : r0 + h, cur_c : cur_c + w] = rect
        cur_c += int(w) + ITEM_GAP
    return int(band_h), int(left), int(band_w)


def _build_expected(
    plain_shape: tuple[int, int],
    plain_base: int,
    centers: list[tuple[int, int]],
    colors: list[int],
    hole_map: dict[int, int],
) -> np.ndarray:
    out = np.full(plain_shape, int(plain_base), dtype=int)
    for (r, c), color in zip(centers, colors, strict=True):
        _apply_stamp(out, int(r), int(c), int(color), int(hole_map[int(color)]))
    return out


def _sample_pattern() -> tuple[tuple[int, int], list[tuple[int, int]], list[int], dict[int, int]] | None:
    plain_h = random.randint(*PLAIN_H_RANGE)
    plain_w = random.randint(*PLAIN_W_RANGE)
    n_colors = random.choice(MARK_COLOR_COUNT_OPTIONS)
    marker_colors = random.sample(GEN_COLORS, n_colors)
    hole_counts = random.sample([1, 2, 3, 4], n_colors)
    hole_map = {int(color): int(holes) for color, holes in zip(marker_colors, hole_counts, strict=True)}

    max_marks = min(MARK_COUNT_RANGE[1], len(_safe_centers(plain_h, plain_w)))
    if max_marks < n_colors or max_marks < MARK_COUNT_RANGE[0]:
        return None
    n_marks = random.randint(max(n_colors, MARK_COUNT_RANGE[0]), max_marks)
    centers = _choose_centers(plain_h, plain_w, n_marks)
    if centers is None:
        return None

    colors = marker_colors.copy()
    while len(colors) < n_marks:
        colors.append(random.choice(marker_colors))
    random.shuffle(colors)
    return (int(plain_h), int(plain_w)), centers, colors, hole_map


def _gen_attempt() -> np.ndarray | None:
    pattern_info = _sample_pattern()
    if pattern_info is None:
        return None
    plain_shape, centers, colors, hole_map = pattern_info
    plain_h, plain_w = plain_shape
    marker_colors = list(hole_map.keys())
    near_rows = [min(int(r), plain_h - 1 - int(r)) for r, _c in centers]
    near_cols = [min(int(c), plain_w - 1 - int(c)) for _r, c in centers]

    base_palette = [color for color in GEN_COLORS if color not in marker_colors]
    plain_base = int(random.choice(base_palette))
    overlap_count = random.randint(*OVERLAP_SOURCE_COUNT_RANGE)
    extra_count = random.randint(*EXTRA_RECT_COUNT_RANGE)

    rectangles: list[np.ndarray] = [np.full((plain_h, plain_w), plain_base, dtype=int)]
    for _ in range(overlap_count + extra_count):
        rect_h = _pick_dim(near_rows, RECT_H_RANGE[0], RECT_H_RANGE[1])
        rect_w = _pick_dim(near_cols, RECT_W_RANGE[0], RECT_W_RANGE[1])
        if rect_h < 0 or rect_w < 0:
            return None
        base_color = int(random.choice(base_palette))
        rectangles.append(
            _make_marked_rect(rect_h, rect_w, base_color, plain_shape, centers, colors)
        )

    expected_pattern = _pattern_grid(plain_shape, centers, colors)
    for rect in rectangles[1:]:
        if not np.array_equal(_adapt_colored_marks(rect, _unique_most_common(rect), plain_shape), expected_pattern):
            return None

    overlap_rects = rectangles[1 : 1 + overlap_count]
    extra_rects = rectangles[1 + overlap_count :]
    overlap_items = _make_overlap_items([rectangles[0], *overlap_rects], plain_index=0)
    if overlap_items is None:
        return None

    overlap_band_h = max(int(off + rect.shape[0]) for off, rect in overlap_items)
    band_items: list[list[tuple[int, np.ndarray]]] = [overlap_items]
    band_heights = [overlap_band_h]
    band_widths = [
        sum(int(rect.shape[1]) for _off, rect in overlap_items) + ITEM_GAP * (len(overlap_items) - 1)
    ]
    for rect in extra_rects:
        band_items.append([(0, rect)])
        band_heights.append(int(rect.shape[0]))
        band_widths.append(int(rect.shape[1]))

    legend_top = random.choice([True, False])
    legend_left = random.choice([True, False])
    legend_vertical = random.choice([True, False])
    if legend_vertical:
        legend_h = len(marker_colors) * 3 + (len(marker_colors) - 1) * LEGEND_ENTRY_GAP
        legend_w = 3
    else:
        legend_h = 3
        legend_w = len(marker_colors) * 3 + (len(marker_colors) - 1) * LEGEND_ENTRY_GAP

    content_h = sum(band_heights) + BAND_GAP * (len(band_heights) - 1)
    content_w = max(band_widths)
    grid_h = 2 * BORDER_GAP + legend_h + LEGEND_GAP + content_h + random.randint(0, 4)
    grid_w = 2 * BORDER_GAP + max(content_w, legend_w) + random.randint(0, 5)
    if grid_h > GRID_H_MAX or grid_w > GRID_W_MAX:
        return None

    grid = np.full((grid_h, grid_w), GEN_BG, dtype=int)
    if legend_top:
        legend_r0 = BORDER_GAP
        content_r0 = BORDER_GAP + legend_h + LEGEND_GAP
    else:
        content_r0 = BORDER_GAP
        legend_r0 = BORDER_GAP + content_h + LEGEND_GAP

    cur_r = int(content_r0)
    for items, band_h in zip(band_items, band_heights, strict=True):
        _render_band(grid, cur_r, items)
        cur_r += int(band_h) + BAND_GAP

    _render_legend(grid, marker_colors, hole_map, legend_top, legend_left, legend_vertical)

    expected = _build_expected(plain_shape, plain_base, centers, colors, hole_map)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
