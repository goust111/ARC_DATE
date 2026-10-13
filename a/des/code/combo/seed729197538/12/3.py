from __future__ import annotations

from collections import Counter

import numpy as np


BG = 0
WHITE = 8
GRAY = 5
MARKER = 6
LABEL_COLORS = (1, 2, 3, 4, 7, 9)
FLAG_COLORS = (1, 2, 3, 4, 7, 9)
MAX_ATTEMPTS = 400

PARTITIONS = {
    1: ((1, 1),),
    2: ((1, 2), (2, 1)),
    3: ((1, 3), (3, 1)),
    4: ((1, 4), (2, 2), (4, 1)),
    5: ((1, 5), (5, 1)),
}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _fill_bbox_mask(mask: np.ndarray, bbox: tuple[int, int, int, int], value: bool = True) -> None:
    r0, c0, r1, c1 = bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = value


def _find_legend_frame(g: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    h, w = g.shape
    best_bbox = None
    best_key = None
    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        for comp in _components(g == color):
            r0, c0, r1, c1 = _bbox(comp)
            bh = r1 - r0 + 1
            bw = c1 - c0 + 1
            if bh < 3 or bw < 3:
                continue
            if len(comp) != 2 * (bh + bw) - 4:
                continue
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            border = np.zeros((bh, bw), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not bool(np.all(sub[border] == color)):
                continue
            if bool(np.any(sub[~border] == color)):
                continue
            corner_dist = min(
                r0 + c0,
                r0 + (w - 1 - c1),
                (h - 1 - r1) + c0,
                (h - 1 - r1) + (w - 1 - c1),
            )
            key = (-int(corner_dist), int(bh * bw))
            if best_key is None or key > best_key:
                best_key = key
                best_bbox = (int(r0), int(c0), int(r1), int(c1))
    assert best_bbox is not None, "expected one gray framed legend"
    return best_bbox


def _read_legend(g: np.ndarray, bg: int, bbox: tuple[int, int, int, int]) -> dict[int, int]:
    r0, c0, r1, c1 = bbox
    inner = g[r0 + 1 : r1, c0 + 1 : c1]
    row_has_content = np.any(inner != bg, axis=1)
    mapping: dict[int, int] = {}
    rr = 0
    while rr < inner.shape[0]:
        if not bool(row_has_content[rr]):
            rr += 1
            continue
        start = rr
        while rr + 1 < inner.shape[0] and bool(row_has_content[rr + 1]):
            rr += 1
        band = inner[start : rr + 1]
        comps = _components(band != bg)
        assert len(comps) >= 2, "each legend row needs markers plus a label"
        rightmost = max(range(len(comps)), key=lambda i: _bbox(comps[i])[3])
        label_comp = comps[rightmost]
        label_colors = {int(band[r, c]) for r, c in label_comp}
        assert len(label_colors) == 1, "legend label must be monochrome"
        mapping[len(comps) - 1] = int(next(iter(label_colors)))
        rr += 1
    assert mapping, "legend must define at least one mapping row"
    return mapping


def _count_flag_regions(g: np.ndarray, comp: list[tuple[int, int]]) -> int:
    r0, c0, r1, c1 = _bbox(comp)
    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    comp_mask = np.zeros(sub.shape, dtype=bool)
    for r, c in comp:
        comp_mask[r - r0, c - c0] = True
    region_mask = comp_mask & (sub != WHITE)
    return len(_components(region_mask))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each flag by counting how many subregions its white dividers create,
    then looking up the replacement color in the corner legend.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
    legend_bbox = _find_legend_frame(g, bg)
    mapping = _read_legend(g, bg, legend_bbox)

    legend_mask = np.zeros(g.shape, dtype=bool)
    _fill_bbox_mask(legend_mask, legend_bbox, True)

    out = g.copy()
    flag_mask = (g != bg) & (~legend_mask)
    for comp in _components(flag_mask):
        regions = _count_flag_regions(g, comp)
        assert regions in mapping, "flag region count missing from legend"
        fill_color = int(mapping[regions])
        for r, c in comp:
            out[r, c] = fill_color
    return out


def _sample_segment_sizes(parts: int, rng: np.random.Generator) -> list[int]:
    if parts >= 4:
        lo, hi = 2, 4
    elif parts == 3:
        lo, hi = 2, 5
    elif parts == 2:
        lo, hi = 2, 6
    else:
        lo, hi = 4, 8
    return [int(v) for v in rng.integers(lo, hi, size=parts)]


def _build_flag_patch(region_count: int, color: int, rng: np.random.Generator) -> np.ndarray:
    row_regions, col_regions = PARTITIONS[int(region_count)][int(rng.integers(0, len(PARTITIONS[int(region_count)])))]
    row_sizes = _sample_segment_sizes(int(row_regions), rng)
    col_sizes = _sample_segment_sizes(int(col_regions), rng)

    h = sum(row_sizes) + int(row_regions) - 1
    w = sum(col_sizes) + int(col_regions) - 1
    patch = np.full((h, w), int(color), dtype=int)

    pos = 0
    for size in row_sizes[:-1]:
        pos += int(size)
        patch[pos, :] = WHITE
        pos += 1

    pos = 0
    for size in col_sizes[:-1]:
        pos += int(size)
        patch[:, pos] = WHITE
        pos += 1

    return patch


def _build_legend_patch(max_regions: int, mapping: dict[int, int]) -> np.ndarray:
    inner_h = 2 * int(max_regions) - 1
    inner_w = 2 * int(max_regions) + 2
    patch = np.full((inner_h + 2, inner_w + 2), BG, dtype=int)
    patch[0, :] = GRAY
    patch[-1, :] = GRAY
    patch[:, 0] = GRAY
    patch[:, -1] = GRAY

    for count in range(1, int(max_regions) + 1):
        row = 1 + 2 * (count - 1)
        for i in range(count):
            patch[row, 1 + 2 * i] = MARKER
        patch[row, inner_w] = int(mapping[count])

    return patch


def _corner_position(
    grid_shape: tuple[int, int],
    patch_shape: tuple[int, int],
    corner: int,
    margin_r: int,
    margin_c: int,
) -> tuple[int, int]:
    gh, gw = grid_shape
    ph, pw = patch_shape
    if corner == 0:
        return int(margin_r), int(margin_c)
    if corner == 1:
        return int(margin_r), int(gw - pw - margin_c)
    if corner == 2:
        return int(gh - ph - margin_r), int(margin_c)
    return int(gh - ph - margin_r), int(gw - pw - margin_c)


def _can_place(occupied: np.ndarray, top: int, left: int, shape: tuple[int, int], pad: int) -> bool:
    h, w = occupied.shape
    ph, pw = shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + ph + pad)
    c1 = min(w, left + pw + pad)
    return not bool(occupied[r0:r1, c0:c1].any())


def _mark_occupied(occupied: np.ndarray, top: int, left: int, shape: tuple[int, int], pad: int) -> None:
    h, w = occupied.shape
    ph, pw = shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + ph + pad)
    c1 = min(w, left + pw + pad)
    occupied[r0:r1, c0:c1] = True


def _paste(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    ph, pw = patch.shape
    dst[top : top + ph, left : left + pw] = patch


def _generate_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    max_regions = int(rng.integers(3, 6))
    label_palette = rng.choice(np.array(LABEL_COLORS, dtype=int), size=max_regions, replace=False)
    mapping = {count: int(color) for count, color in enumerate(label_palette, start=1)}
    legend = _build_legend_patch(max_regions, mapping)

    n_flags = int(rng.integers(2, 5))
    counts = []
    distinct_needed = min(max_regions, max(2, n_flags - 1))
    distinct = rng.choice(np.arange(1, max_regions + 1), size=distinct_needed, replace=False)
    counts.extend(int(v) for v in distinct)
    while len(counts) < n_flags:
        counts.append(int(rng.integers(1, max_regions + 1)))
    rng.shuffle(counts)
    if all(count == 1 for count in counts):
        counts[0] = int(rng.integers(2, max_regions + 1))

    flag_specs = []
    total_area = legend.shape[0] * legend.shape[1]
    for count in counts:
        color = int(rng.choice(np.array(FLAG_COLORS, dtype=int)))
        patch = _build_flag_patch(int(count), color, rng)
        flag_specs.append((int(count), patch))
        total_area += int(patch.shape[0] * patch.shape[1])

    max_flag_h = max(int(patch.shape[0]) for _, patch in flag_specs)
    max_flag_w = max(int(patch.shape[1]) for _, patch in flag_specs)
    min_h = max(22, int(legend.shape[0]) + 4, max_flag_h + 4)
    min_w = max(22, int(legend.shape[1]) + 4, max_flag_w + 4)
    density_hint = int(np.ceil(np.sqrt(total_area * 2.4)))
    min_h = max(min_h, density_hint)
    min_w = max(min_w, density_hint)
    if min_h > 30 or min_w > 30:
        return None

    grid_h = int(rng.integers(min_h, 31))
    grid_w = int(rng.integers(min_w, 31))
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    expected = grid.copy()
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    corner = int(rng.integers(0, 4))
    max_margin_r = min(1, grid_h - int(legend.shape[0]))
    max_margin_c = min(1, grid_w - int(legend.shape[1]))
    margin_r = int(rng.integers(0, max_margin_r + 1))
    margin_c = int(rng.integers(0, max_margin_c + 1))
    legend_r0, legend_c0 = _corner_position(grid.shape, legend.shape, corner, margin_r, margin_c)
    _paste(grid, legend, legend_r0, legend_c0)
    _paste(expected, legend, legend_r0, legend_c0)
    _mark_occupied(occupied, legend_r0, legend_c0, legend.shape, 1)

    placements = []
    for count, patch in sorted(flag_specs, key=lambda item: item[1].shape[0] * item[1].shape[1], reverse=True):
        placed = False
        for _ in range(300):
            top = int(rng.integers(0, grid_h - int(patch.shape[0]) + 1))
            left = int(rng.integers(0, grid_w - int(patch.shape[1]) + 1))
            if not _can_place(occupied, top, left, patch.shape, 1):
                continue
            _paste(grid, patch, top, left)
            expected[top : top + patch.shape[0], left : left + patch.shape[1]] = int(mapping[count])
            _mark_occupied(occupied, top, left, patch.shape, 1)
            placements.append((count, top, left, patch.shape))
            placed = True
            break
        if not placed:
            return None

    if not placements:
        return None
    if not any(count > 1 for count, *_rest in placements):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
