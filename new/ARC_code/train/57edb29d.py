from collections import Counter

import numpy as np

from arc_utils import crop_bbox, find_components
import random


def _unique_most_common(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    max_count = max(counts.values())
    winners = [color for color, count in counts.items() if count == max_count]
    assert len(winners) == 1, "expected a unique dominant color"
    return int(winners[0])


def _boundary_bg(g: np.ndarray) -> int:
    boundary = np.concatenate((g[0, :], g[-1, :], g[:, 0], g[:, -1]))
    return _unique_most_common(boundary)


def _runs(cols: np.ndarray) -> list[tuple[int, int]]:
    assert cols.size > 0, "expected a nonempty marker row"
    spans = []
    start = end = int(cols[0])
    for col in map(int, cols[1:]):
        if col == end + 1:
            end = col
        else:
            spans.append((start, end))
            start = end = col
    spans.append((start, end))
    return spans


def _map_row(row: int, src_h: int, dst_h: int) -> int:
    top = int(row)
    bottom = int(src_h - 1 - row)
    assert top != bottom, "marker row must be closer to one horizontal edge"
    mapped = top if top < bottom else int(dst_h - 1 - bottom)
    assert 0 <= mapped < dst_h, "mapped marker row must fit in target rectangle"
    return int(mapped)


def _map_col_run(c0: int, c1: int, src_w: int, dst_w: int) -> tuple[int, int]:
    mid2 = int(src_w - 1)
    if c0 == c1:
        left = int(c0)
        right = int(src_w - 1 - c0)
        assert left != right, "single marker pixel must be closer to one vertical edge"
        mapped = left if left < right else int(dst_w - 1 - right)
        assert 0 <= mapped < dst_w, "mapped marker column must fit in target rectangle"
        return int(mapped), int(mapped)

    if 2 * int(c1) <= mid2:
        start, end = int(c0), int(c1)
    elif 2 * int(c0) >= mid2:
        start = int(dst_w - 1 - (src_w - 1 - c0))
        end = int(dst_w - 1 - (src_w - 1 - c1))
    else:
        start = int(c0)
        end = int(dst_w - 1 - (src_w - 1 - c1))

    assert 0 <= start <= end < dst_w, "mapped marker run must fit in target rectangle"
    return int(start), int(end)


def _adapt_marker_mask(src: np.ndarray, marker_color: int, dst_shape: tuple[int, int]) -> np.ndarray:
    src_h, src_w = map(int, src.shape)
    dst_h, dst_w = map(int, dst_shape)
    src_mask = src == int(marker_color)
    assert bool(src_mask.any()), "source rectangle must contain marker pixels"

    out = np.zeros((dst_h, dst_w), dtype=bool)
    for row in map(int, np.flatnonzero(src_mask.any(axis=1))):
        dst_row = _map_row(row, src_h, dst_h)
        for c0, c1 in _runs(np.flatnonzero(src_mask[row, :])):
            dst_c0, dst_c1 = _map_col_run(c0, c1, src_w, dst_w)
            out[dst_row, dst_c0 : dst_c1 + 1] = True
    return out


def _row_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return max(int(a[0]), int(b[0])) <= min(int(a[2]), int(b[2]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the single plain rectangle and redraw it with the aligned rectangles' border-relative marker pattern.

    Input structure:
    - The boundary color is the background/separator.
    - Foreground objects are dense rectangular regions; exactly one rectangle has only its base color.
    - Other rectangles contain their base color plus marker pixels.

    Rule:
    - Crop the only plain rectangle; this crop is the output canvas.
    - Use marked rectangles whose row span overlaps the plain rectangle's row span.
    - Transfer their marker pixels by preserving each marker run's distance from the nearest rectangle edges.

    Key insight:
    - The marked rectangles are examples of an edge-relative pattern, so the missing plain rectangle receives the same pattern resized to its own dimensions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _boundary_bg(g)

    objects = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=False):
        bbox = tuple(map(int, comp["bbox"]))
        sub = crop_bbox(g, bbox)
        assert np.all(sub != bg), "foreground rectangles must be dense"

        base_color = _unique_most_common(sub)
        marker_colors = set(map(int, sub.reshape(-1))) - {int(base_color)}
        objects.append(
            {
                "bbox": bbox,
                "sub": sub,
                "base_color": int(base_color),
                "marker_colors": marker_colors,
            }
        )

    plain = [obj for obj in objects if not obj["marker_colors"]]
    assert len(plain) == 1, "expected exactly one plain rectangle"
    target = plain[0]
    tr0, tc0, tr1, tc1 = target["bbox"]
    target_shape = (int(tr1 - tr0 + 1), int(tc1 - tc0 + 1))

    sources = [
        obj
        for obj in objects
        if obj["marker_colors"] and _row_overlap(obj["bbox"], target["bbox"])
    ]
    assert sources, "expected at least one marked rectangle aligned with the plain rectangle"

    marker_colors = set()
    for source in sources:
        marker_colors.update(source["marker_colors"])
    assert len(marker_colors) == 1, "aligned marked rectangles must share one marker color"
    marker_color = int(next(iter(marker_colors)))

    masks = []
    for source in sources:
        assert source["marker_colors"] == {marker_color}, "source must contain only the shared marker color"
        masks.append(_adapt_marker_mask(source["sub"], marker_color, target_shape))

    pattern = masks[0]
    for mask in masks[1:]:
        assert np.array_equal(mask, pattern), "aligned marked rectangles must encode the same pattern"

    out = np.full(target_shape, int(target["base_color"]), dtype=int)
    out[pattern] = marker_color
    return out

# --------------- generator (att1, Claude) ---------------

# Canvas and separator colors.
BG = 4
NON_BG = [c for c in range(10) if c != BG]
GRID_H_RANGE = (20, 28)
GRID_W_RANGE = (20, 28)
BORDER_GAP = 1
BAND_GAP = 1
# Rectangle counts and dimensions.
RECT_COUNT_RANGE = (3, 6)
OVERLAP_MARKED_COUNT = 2
RECT_H_RANGE = (5, 20)
RECT_W_RANGE = (5, 12)
PLAIN_EXTRA_ROW_ROOM = 2
# Marker pattern sampling.
MARKER_PAIR_COUNT_RANGE = (1, 3)
MARKER_EDGE_PAIRS = [(1, 1), (1, 2), (2, 1), (2, 2)]
# Retry budget.
GENERATION_ATTEMPTS = 300


def _stamp_pattern(rect: np.ndarray, marker: int, pairs: list[tuple[int, int]]):
    h, w = rect.shape
    for nr, nc in pairs:
        for r in (nr, h - 1 - nr):
            for c in (nc, w - 1 - nc):
                rect[r, c] = marker


def _pick_dim(near_vals: list[int], lo: int, hi: int) -> int:
    min_dim = max(lo, max(2 * nv + 2 for nv in near_vals))
    cands = [d for d in range(min_dim, hi + 1) if all(d >= 2 * nv + 2 and d != 2 * nv + 1 for nv in near_vals)]
    return random.choice(cands) if cands else -1


def _choose_marker_pattern():
    n_pairs = random.randint(*MARKER_PAIR_COUNT_RANGE)
    pairs = sorted(random.sample(MARKER_EDGE_PAIRS, min(n_pairs, len(MARKER_EDGE_PAIRS))))
    near_rows = [p[0] for p in pairs]
    near_cols = [p[1] for p in pairs]
    return pairs, near_rows, near_cols


def _choose_palette(n_rects: int):
    needed = n_rects + 1  # base colors + marker
    if needed > len(NON_BG):
        return None
    chosen = random.sample(NON_BG, needed)
    plain_base, marker_color = chosen[0], chosen[-1]
    marked_bases = chosen[1:-1]
    return plain_base, marker_color, marked_bases


def _make_marked_rect(h: int, w: int, base: int, marker: int, pairs: list[tuple[int, int]]) -> np.ndarray:
    rect = np.full((h, w), base, dtype=int)
    _stamp_pattern(rect, marker, pairs)
    return rect


def _choose_plain_shape(near_rows: list[int], near_cols: list[int], avail_h: int, avail_w: int):
    hp = _pick_dim(near_rows, RECT_H_RANGE[0], min(RECT_H_RANGE[1], avail_h - PLAIN_EXTRA_ROW_ROOM))
    wp = _pick_dim(near_cols, RECT_W_RANGE[0], min(RECT_W_RANGE[1], avail_w))
    if hp < 0 or wp < 0:
        return None
    return hp, wp


def _has_overlap_width_budget(avail_w: int, plain_w: int) -> bool:
    remaining_w = avail_w - plain_w - OVERLAP_MARKED_COUNT * BAND_GAP
    max_overlap_w = remaining_w // OVERLAP_MARKED_COUNT
    if max_overlap_w >= RECT_W_RANGE[0]:
        return True
    return remaining_w >= RECT_W_RANGE[0]


def _build_overlap_rects(avail_h, avail_w, plain_w, near_rows, near_cols, marked_bases, marker_color, pairs):
    if not _has_overlap_width_budget(avail_w, plain_w):
        return None

    min_w, max_w = RECT_W_RANGE
    min_h, max_h = RECT_H_RANGE
    overlap_rects = []
    remaining_w = avail_w - plain_w - OVERLAP_MARKED_COUNT * BAND_GAP
    for i in range(OVERLAP_MARKED_COUNT):
        if i == 0:
            max_w_i = min(remaining_w - min_w, max_w)
        else:
            max_w_i = min(remaining_w, max_w)
        wm = _pick_dim(near_cols, min_w, max(min_w, max_w_i))
        if wm < 0:
            return None
        remaining_w -= wm
        if remaining_w < 0:
            return None
        hm = _pick_dim(near_rows, min_h, min(max_h, avail_h))
        if hm < 0:
            return None
        overlap_rects.append(_make_marked_rect(hm, wm, marked_bases[i], marker_color, pairs))
    return overlap_rects


def _build_non_overlap_rects(n_marked, overlap_band_h, avail_h, avail_w, near_rows, near_cols, marked_bases, marker_color, pairs):
    min_h, max_h = RECT_H_RANGE
    min_w, max_w = RECT_W_RANGE
    non_overlap_rects = []
    for i in range(OVERLAP_MARKED_COUNT, n_marked):
        remaining_h_for_bands = avail_h - overlap_band_h - BAND_GAP
        if remaining_h_for_bands < min_h:
            return None
        hm = _pick_dim(near_rows, min_h, min(max_h, remaining_h_for_bands))
        wm = _pick_dim(near_cols, min_w, min(max_w, avail_w))
        if hm < 0 or wm < 0:
            return None
        non_overlap_rects.append(_make_marked_rect(hm, wm, marked_bases[i], marker_color, pairs))
    return non_overlap_rects


def _pack_non_overlap_bands(rects: list[np.ndarray], avail_w: int) -> list[list[np.ndarray]]:
    bands = []
    remaining = list(rects)
    random.shuffle(remaining)
    while remaining:
        band = [remaining.pop(0)]
        band_w = band[0].shape[1]
        i = 0
        while i < len(remaining):
            if band_w + BAND_GAP + remaining[i].shape[1] <= avail_w:
                band_w += BAND_GAP + remaining[i].shape[1]
                band.append(remaining.pop(i))
            else:
                i += 1
        bands.append(band)
    return bands


def _choose_layout_shape(target_h, target_w, plain_w, overlap_rects, overlap_band_h, non_overlap_bands):
    non_overlap_heights = [max(r.shape[0] for r in band) for band in non_overlap_bands]
    total_content_h = overlap_band_h + sum(non_overlap_heights) + len(non_overlap_bands) * BAND_GAP
    total_h = total_content_h + 2 * BORDER_GAP
    if total_h > target_h:
        return None

    actual_h = max(target_h, total_h)
    if actual_h > GRID_H_RANGE[1]:
        return None

    overlap_band_w = plain_w + sum(rect.shape[1] for rect in overlap_rects) + OVERLAP_MARKED_COUNT * BAND_GAP
    max_band_w = overlap_band_w
    for band in non_overlap_bands:
        band_w = sum(r.shape[1] for r in band) + max(0, len(band) - 1) * BAND_GAP
        max_band_w = max(max_band_w, band_w)
    actual_w = max(target_w, max_band_w + 2 * BORDER_GAP)
    if actual_w > GRID_W_RANGE[1]:
        return None
    return actual_h, actual_w


def _make_overlap_items(plain_h, plain_w, plain_base, overlap_rects, overlap_band_h):
    """Return band items whose marked rectangles overlap the plain rectangle's rows."""
    overlap_items = [(0, np.full((plain_h, plain_w), plain_base, dtype=int))]
    for rect in overlap_rects:
        overlap_items.append((0, rect))
    random.shuffle(overlap_items)

    plain_idx = None
    for idx, (_, rect) in enumerate(overlap_items):
        if rect.shape == (plain_h, plain_w) and np.all(rect == plain_base):
            plain_idx = idx
            break
    if plain_idx is None:
        return None

    offset_p = random.randint(0, max(0, overlap_band_h - plain_h))
    overlap_items[plain_idx] = (offset_p, overlap_items[plain_idx][1])
    p_r0_local, p_r1_local = offset_p, offset_p + plain_h - 1

    for idx in range(len(overlap_items)):
        if idx == plain_idx:
            continue
        rect = overlap_items[idx][1]
        rh = rect.shape[0]
        lo_off = max(0, p_r0_local - rh + 1)
        hi_off = min(overlap_band_h - rh, p_r1_local)
        if lo_off > hi_off:
            return None
        off = random.randint(lo_off, hi_off)
        overlap_items[idx] = (off, rect)
    return overlap_items


def _make_non_overlap_items(non_overlap_bands: list[list[np.ndarray]]) -> list[list[tuple[int, np.ndarray]]]:
    bands = []
    for band_rects in non_overlap_bands:
        band_h = max(r.shape[0] for r in band_rects)
        items = [(random.randint(0, max(0, band_h - r.shape[0])), r) for r in band_rects]
        bands.append(items)
    return bands


def _render_bands(actual_h: int, actual_w: int, all_bands: list[list[tuple[int, np.ndarray]]]):
    """Render band items with one-cell separators and a one-cell border."""
    random.shuffle(all_bands)

    band_heights = []
    for items in all_bands:
        bh = max(off + r.shape[0] for off, r in items)
        band_heights.append(bh)

    grid = np.full((actual_h, actual_w), BG, dtype=int)
    cur_row = BORDER_GAP
    for bi, items in enumerate(all_bands):
        band_content_w = sum(r.shape[1] for _, r in items) + max(0, len(items) - 1) * BAND_GAP
        remaining_w = actual_w - 2 * BORDER_GAP - band_content_w
        left_pad = random.randint(0, max(0, remaining_w))
        cur_col = BORDER_GAP + left_pad

        for off, rect in items:
            rh, rw = rect.shape
            r0 = cur_row + off
            c0 = cur_col
            if r0 + rh > actual_h - BORDER_GAP or c0 + rw > actual_w - BORDER_GAP:
                return None
            grid[r0:r0+rh, c0:c0+rw] = rect
            cur_col += rw + BAND_GAP

        cur_row += band_heights[bi] + BAND_GAP
    return grid


def _accept_if_solver_matches(grid: np.ndarray, output: np.ndarray) -> np.ndarray | None:
    try:
        result = main(grid)
        if np.array_equal(result, output):
            return grid
    except Exception:
        pass
    return None


def _gen_attempt() -> np.ndarray | None:
    pairs, near_rows, near_cols = _choose_marker_pattern()

    n_rects = random.randint(*RECT_COUNT_RANGE)
    n_marked = n_rects - 1
    palette = _choose_palette(n_rects)
    if palette is None:
        return None
    plain_base, marker_color, marked_bases = palette

    target_h = random.randint(*GRID_H_RANGE)
    target_w = random.randint(*GRID_W_RANGE)
    avail_h = target_h - 2 * BORDER_GAP
    avail_w = target_w - 2 * BORDER_GAP

    plain_shape = _choose_plain_shape(near_rows, near_cols, avail_h, avail_w)
    if plain_shape is None:
        return None
    plain_h, plain_w = plain_shape
    output = _make_marked_rect(plain_h, plain_w, plain_base, marker_color, pairs)

    overlap_rects = _build_overlap_rects(
        avail_h, avail_w, plain_w, near_rows, near_cols, marked_bases, marker_color, pairs
    )
    if overlap_rects is None:
        return None
    overlap_band_h = max(plain_h, *(rect.shape[0] for rect in overlap_rects))

    non_overlap_rects = _build_non_overlap_rects(
        n_marked, overlap_band_h, avail_h, avail_w, near_rows, near_cols, marked_bases, marker_color, pairs
    )
    if non_overlap_rects is None:
        return None
    non_overlap_bands = _pack_non_overlap_bands(non_overlap_rects, avail_w)

    layout_shape = _choose_layout_shape(
        target_h, target_w, plain_w, overlap_rects, overlap_band_h, non_overlap_bands
    )
    if layout_shape is None:
        return None
    actual_h, actual_w = layout_shape

    overlap_items = _make_overlap_items(plain_h, plain_w, plain_base, overlap_rects, overlap_band_h)
    if overlap_items is None:
        return None
    all_bands = [overlap_items] + _make_non_overlap_items(non_overlap_bands)

    grid = _render_bands(actual_h, actual_w, all_bands)
    if grid is None:
        return None
    return _accept_if_solver_matches(grid, output)


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GENERATION_ATTEMPTS} attempts")
