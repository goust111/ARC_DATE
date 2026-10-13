from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
GRAY = 5
KEY_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

LEGEND_SHAPES = (
    np.array([[1, 1, 1, 1]], dtype=int),
    np.array([[1], [1], [1], [1]], dtype=int),
    np.array([[1, 1], [1, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=int),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=int),
    np.array([[0, 0, 1], [1, 1, 1]], dtype=int),
    np.array([[1, 1], [1, 0], [1, 0]], dtype=int),
    np.array([[1, 1], [0, 1], [0, 1]], dtype=int),
    np.array([[0, 1, 1], [1, 1, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 0], [1, 1], [0, 1]], dtype=int),
)

DISTRACTOR_SHAPES = (
    np.array([[1, 1, 1], [1, 0, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1], [0, 1, 0]], dtype=int),
)

GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([[r0, c0]], dtype=int)
    out[rel[:, 0], rel[:, 1]] = True
    return out


def _scale_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(mask.astype(np.uint8), np.ones((factor, factor), dtype=np.uint8)).astype(bool)


def _match_scale(legend_mask: np.ndarray, obj_mask: np.ndarray) -> int | None:
    lh, lw = legend_mask.shape
    oh, ow = obj_mask.shape
    if oh % lh != 0 or ow % lw != 0:
        return None
    k_row = oh // lh
    k_col = ow // lw
    if k_row != k_col or k_row < 1:
        return None
    scaled = _scale_mask(legend_mask, k_row)
    if np.array_equal(scaled, obj_mask):
        return int(k_row)
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group grey field objects by scale-invariant legend shape and output one colored bar per legend.

    Input structure:
    - A black-background grid has a left legend column made of distinct four-cell shapes, each drawn in its own key color.
    - One full-height grey separator column splits the legend area from the field area.
    - The field contains several grey 4-connected objects: some are integer-scaled copies of legend shapes, others are distractors.

    Rule:
    - Read the colored legend shapes from top to bottom and normalize each by cropping to its bbox.
    - For every grey field object, crop its bbox and check whether it equals an integer k-times scale-up of exactly one legend shape without rotation or reflection.
    - For each legend, sum the matched scale factors k and output a row of that legend color whose length equals the sum; pad the remaining cells with black to the maximum row length.

    Key insight:
    - Pixel count alone cannot identify the class because every legend has the same four-cell area; the solver must remove scaling and compare normalized shape geometry, while ignoring grey distractors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    h, w = g.shape
    bg = _get_bg(g)

    sep_cols = [c for c in range(w) if len(set(map(int, g[:, c]))) == 1 and int(g[0, c]) != bg]
    assert len(sep_cols) == 1, "expected one full-height separator column"
    sep_c = int(sep_cols[0])
    object_color = int(g[0, sep_c])

    legend_area = g[:, :sep_c]
    field_area = g[:, sep_c + 1 :]
    assert legend_area.size > 0 and field_area.size > 0

    legend_entries = []
    for color in sorted(set(map(int, legend_area.ravel())) - {bg, object_color}):
        comps = _component_coords(legend_area == color)
        assert len(comps) == 1, "each legend color should define one shape"
        coords = comps[0]
        assert coords.shape[0] == 4, "legend shapes must have four cells"
        bbox = _bbox_from_coords(coords)
        legend_entries.append(
            {
                "color": int(color),
                "top": int(bbox[0]),
                "mask": _coords_to_mask(coords),
            }
        )

    legend_entries.sort(key=lambda item: item["top"])
    assert len(legend_entries) >= 2

    totals = [0 for _ in legend_entries]
    for coords in _component_coords(field_area == object_color):
        obj_mask = _coords_to_mask(coords)
        hits = []
        for idx, entry in enumerate(legend_entries):
            scale = _match_scale(entry["mask"], obj_mask)
            if scale is not None:
                hits.append((idx, scale))
        assert len(hits) <= 1, "field object should match at most one legend"
        if hits:
            idx, scale = hits[0]
            totals[idx] += int(scale)

    width = max(totals)
    out = np.zeros((len(legend_entries), width), dtype=int)
    for row, (entry, total) in enumerate(zip(legend_entries, totals)):
        out[row, :total] = int(entry["color"])
    return out


def _sample_match_plan(n_legends: int) -> list[list[int]] | None:
    lure_idx, weighted_idx = random.sample(range(n_legends), 2)
    scales_by_legend = [[] for _ in range(n_legends)]

    lure_count = random.randint(1, 3)
    scales_by_legend[lure_idx] = [1] * lure_count

    weighted_count = random.randint(1, 3)
    weighted = [random.choice((1, 2, 2, 3)) for _ in range(weighted_count)]
    if all(scale == 1 for scale in weighted):
        weighted[random.randrange(weighted_count)] = random.choice((2, 3))
    scales_by_legend[weighted_idx] = weighted

    for idx in range(n_legends):
        if idx in (lure_idx, weighted_idx):
            continue
        count = random.choice((0, 0, 1, 1, 2))
        if count:
            scales_by_legend[idx] = [random.choice((1, 1, 2, 3)) for _ in range(count)]

    totals = [sum(scales) for scales in scales_by_legend]
    object_count = sum(len(scales) for scales in scales_by_legend)
    if not (3 <= object_count <= 6):
        return None
    if max(totals) > 10 or max(totals) == 0:
        return None
    if sum(total > 0 for total in totals) < 2:
        return None
    if not any(scale == 1 for scales in scales_by_legend for scale in scales):
        return None
    if not any(scale > 1 for scales in scales_by_legend for scale in scales):
        return None
    return scales_by_legend


def _legend_layout(shapes: list[np.ndarray]) -> tuple[int, list[int], int] | None:
    top_margin = random.randint(1, 2)
    bottom_margin = random.randint(1, 2)
    gaps = [random.randint(1, 3) for _ in range(len(shapes) - 1)]
    rows = []
    cur = top_margin
    for idx, shape in enumerate(shapes):
        rows.append(cur)
        cur += int(shape.shape[0])
        if idx < len(gaps):
            cur += gaps[idx]
    height_needed = cur + bottom_margin
    if height_needed > 28:
        return None
    sep_c = max(int(shape.shape[1]) for shape in shapes) + random.randint(1, 2)
    return height_needed, rows, sep_c


def _place_patches(
    h: int,
    w: int,
    left_bound: int,
    patches: list[np.ndarray],
) -> list[tuple[int, int]] | None:
    occupied = np.zeros((h, w), dtype=bool)
    placements: list[tuple[int, int]] = []

    indexed = sorted(enumerate(patches), key=lambda item: int(item[1].shape[0] * item[1].shape[1]), reverse=True)
    chosen: dict[int, tuple[int, int]] = {}
    for idx, patch in indexed:
        ph, pw = patch.shape
        candidates = []
        for r0 in range(1, h - ph):
            for c0 in range(left_bound, w - pw):
                rr0 = max(0, r0 - 1)
                rr1 = min(h, r0 + ph + 1)
                cc0 = max(0, c0 - 1)
                cc1 = min(w, c0 + pw + 1)
                if occupied[rr0:rr1, cc0:cc1].any():
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None
        pos = random.choice(candidates)
        chosen[idx] = pos
        r0, c0 = pos
        occupied[max(0, r0 - 1) : min(h, r0 + ph + 1), max(0, c0 - 1) : min(w, c0 + pw + 1)] = True

    for idx in range(len(patches)):
        placements.append(chosen[idx])
    return placements


def _render_expected(colors: list[int], scales_by_legend: list[list[int]]) -> np.ndarray:
    totals = [sum(scales) for scales in scales_by_legend]
    width = max(totals)
    out = np.zeros((len(colors), width), dtype=int)
    for row, (color, total) in enumerate(zip(colors, totals)):
        out[row, :total] = int(color)
    return out


def _render_input(
    shapes: list[np.ndarray],
    colors: list[int],
    scales_by_legend: list[list[int]],
) -> np.ndarray | None:
    layout = _legend_layout(shapes)
    if layout is None:
        return None
    legend_h_needed, legend_rows, sep_c = layout

    field_patches: list[np.ndarray] = []
    for shape, scales in zip(shapes, scales_by_legend):
        shape_mask = shape.astype(bool)
        for scale in scales:
            field_patches.append(np.where(_scale_mask(shape_mask, scale), GRAY, BG))

    distractor_count = random.randint(1, 2)
    for _ in range(distractor_count):
        distractor = random.choice(DISTRACTOR_SHAPES)
        field_patches.append(np.where(distractor.astype(bool), GRAY, BG))

    max_patch_h = max(int(patch.shape[0]) for patch in field_patches)
    max_patch_w = max(int(patch.shape[1]) for patch in field_patches)
    min_h = max(legend_h_needed, max_patch_h + 2, 14)
    min_w = max(sep_c + 4 + max_patch_w, 18)
    if min_h > 30 or min_w > 30:
        return None

    h = random.randint(min_h, min(30, max(min_h, 24)))
    w = random.randint(min_w, 30)
    field_left = sep_c + 2
    placements = _place_patches(h, w, field_left, field_patches)
    if placements is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    for row, shape, color in zip(legend_rows, shapes, colors):
        sh, sw = shape.shape
        grid[row : row + sh, :sw] = np.where(shape, int(color), BG)
    grid[:, sep_c] = GRAY

    for patch, (r0, c0) in zip(field_patches, placements):
        ph, pw = patch.shape
        mask = patch == GRAY
        grid[r0 : r0 + ph, c0 : c0 + pw][mask] = GRAY

    return grid


def _gen_attempt() -> np.ndarray | None:
    n_legends = random.randint(2, 4)
    shapes = [shape.copy() for shape in random.sample(LEGEND_SHAPES, n_legends)]
    colors = random.sample(KEY_COLORS, n_legends)

    scales_by_legend = _sample_match_plan(n_legends)
    if scales_by_legend is None:
        return None

    inp = _render_input(shapes, colors, scales_by_legend)
    if inp is None:
        return None

    expected = _render_expected(colors, scales_by_legend)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid legend-weighing input")
