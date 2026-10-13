import random
from collections import deque

import numpy as np


BG = 0
GRAY = 5
MAX_GENERATION_RETRIES = 300
GRID_MIN = 18
GRID_MAX = 30
LEGEND_GAP = 1
PLACEMENT_GAP = 1


BASE_HOLES = [
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1], [1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords):
    pts = np.asarray(coords, dtype=int)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _crop(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    assert len(pts) > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _shape_variants(mask: np.ndarray):
    arr = mask.astype(np.uint8)
    seen = set()
    variants = []
    for k in range(4):
        rot = np.rot90(arr, k)
        for oriented in (rot, np.fliplr(rot)):
            cropped = _crop(oriented.astype(bool))
            key = (cropped.shape, tuple(int(x) for x in cropped.ravel()))
            if key not in seen:
                seen.add(key)
                variants.append(cropped)
    return variants


def _shape_key(mask: np.ndarray):
    keys = []
    for variant in _shape_variants(mask):
        keys.append((variant.shape, tuple(int(x) for x in variant.ravel())))
    return min(keys)


def _random_orientation(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    return rng.choice(_shape_variants(mask))


def _extract_single_hole(gray_coords) -> np.ndarray | None:
    r0, c0, r1, c1 = _bbox(gray_coords)
    solid = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in gray_coords:
        solid[r - r0, c - c0] = True

    if not np.all(solid[0]) or not np.all(solid[-1]) or not np.all(solid[:, 0]) or not np.all(solid[:, -1]):
        return None

    holes = []
    for comp in _components(~solid):
        if any(rr == 0 or cc == 0 or rr == solid.shape[0] - 1 or cc == solid.shape[1] - 1 for rr, cc in comp):
            continue
        holes.append(comp)
    if len(holes) != 1:
        return None

    hole = np.zeros_like(solid)
    for rr, cc in holes[0]:
        hole[rr, cc] = True
    return _crop(hole)


def _legend_mapping(grid: np.ndarray):
    h, w = grid.shape
    mapping = {}
    for comp in _components(grid != BG):
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        gray_coords = [(r, c) for r, c in comp if int(grid[r, c]) == GRAY]
        color_cells = [(r, c, int(grid[r, c])) for r, c in comp if int(grid[r, c]) not in (BG, GRAY)]
        if not gray_coords or len(color_cells) != 1:
            continue
        cr, cc, color = color_cells[0]
        if not any(abs(cr - r) + abs(cc - c) == 1 for r, c in gray_coords):
            continue
        hole = _extract_single_hole(gray_coords)
        if hole is None:
            continue
        key = _shape_key(hole)
        assert key not in mapping, "legend shapes must be unique"
        mapping[key] = color
    assert mapping, "expected at least one legend mapping"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each rectangle's internal hole shape to an edge legend and recolor that rectangle's border.

    Input structure:
    - The background is black.
    - Away from the boundary are solid gray rectangles, each with exactly one enclosed black hole shape cut out of its interior.
    - Along one grid edge are small gray legend plaques; each plaque contains one enclosed black hole shape and touches one adjacent colored pixel.

    Rule:
    - Read every edge legend as a shape-to-color mapping by extracting the black hole inside its gray plaque.
    - For each non-edge gray rectangle, extract its enclosed black hole and find the congruent legend shape, allowing rotations and reflections.
    - Recolor only the rectangle's outer gray border to the mapped legend color; keep the rectangle interior, the black hole, and the legend area unchanged.

    Key insight:
    - The color is not tied to the rectangle itself; it is determined by matching the rectangle's interior hole geometry to the hole geometry shown in the boundary legend.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    mapping = _legend_mapping(g)

    h, w = g.shape
    targets = 0
    for gray_comp in _components(g == GRAY):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in gray_comp):
            continue
        hole = _extract_single_hole(gray_comp)
        if hole is None:
            continue
        key = _shape_key(hole)
        assert key in mapping, "every target hole must appear in the legend"
        color = mapping[key]
        r0, c0, r1, c1 = _bbox(gray_comp)
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
        targets += 1

    assert targets > 0, "expected at least one target rectangle"
    return out


def _build_legend_patch(shape: np.ndarray, color: int, side: str, rng: random.Random) -> np.ndarray:
    hole = _random_orientation(shape, rng)
    hh, hw = hole.shape
    anchor = rng.randrange(hw) if side in ("top", "bottom") else rng.randrange(hh)
    if side == "top":
        patch = np.full((hh + 3, hw + 2), BG, dtype=int)
        patch[: hh + 2, :] = GRAY
        patch[1 : 1 + hh, 1 : 1 + hw][hole] = BG
        patch[hh + 2, 1 + anchor] = color
    elif side == "bottom":
        patch = np.full((hh + 3, hw + 2), BG, dtype=int)
        patch[1:, :] = GRAY
        patch[0, 1 + anchor] = color
        patch[1 : 1 + hh, 1 : 1 + hw][hole] = BG
    elif side == "left":
        patch = np.full((hh + 2, hw + 3), BG, dtype=int)
        patch[:, : hw + 2] = GRAY
        patch[1 : 1 + hh, 1 : 1 + hw][hole] = BG
        patch[1 + anchor, hw + 2] = color
    elif side == "right":
        patch = np.full((hh + 2, hw + 3), BG, dtype=int)
        patch[:, 1:] = GRAY
        patch[1 + anchor, 0] = color
        patch[1 : 1 + hh, 1 : 1 + hw][hole] = BG
    else:
        raise ValueError(side)
    return patch


def _build_target_patch(shape: np.ndarray, rng: random.Random) -> np.ndarray:
    hole = _random_orientation(shape, rng)
    hh, hw = hole.shape
    outer_h = hh + rng.randint(4, 7)
    outer_w = hw + rng.randint(4, 7)
    patch = np.full((outer_h, outer_w), GRAY, dtype=int)
    hr = rng.randint(1, outer_h - hh - 1)
    hc = rng.randint(1, outer_w - hw - 1)
    patch[hr : hr + hh, hc : hc + hw][hole] = BG
    return patch


def _random_partition(total: int, parts: int, rng: random.Random):
    cuts = sorted(rng.sample(range(total + parts - 1), parts - 1)) if parts > 1 else []
    out = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _layout_legends(grid: np.ndarray, occ: np.ndarray, patches, side: str, rng: random.Random):
    h, w = grid.shape
    if side in ("top", "bottom"):
        widths = [p.shape[1] for p in patches]
        span = sum(widths) + LEGEND_GAP * (len(widths) - 1)
        slack = w - span
        extras = _random_partition(slack, len(widths) + 1, rng)
        cursor = extras[0]
        for i, patch in enumerate(patches):
            ph, pw = patch.shape
            r0 = 0 if side == "top" else h - ph
            c0 = cursor
            grid[r0 : r0 + ph, c0 : c0 + pw] = patch
            occ[max(0, r0 - PLACEMENT_GAP) : min(h, r0 + ph + PLACEMENT_GAP),
                max(0, c0 - PLACEMENT_GAP) : min(w, c0 + pw + PLACEMENT_GAP)] = True
            cursor += pw + LEGEND_GAP
            if i + 1 < len(extras):
                cursor += extras[i + 1]
    else:
        heights = [p.shape[0] for p in patches]
        span = sum(heights) + LEGEND_GAP * (len(heights) - 1)
        slack = h - span
        extras = _random_partition(slack, len(heights) + 1, rng)
        cursor = extras[0]
        for i, patch in enumerate(patches):
            ph, pw = patch.shape
            r0 = cursor
            c0 = 0 if side == "left" else w - pw
            grid[r0 : r0 + ph, c0 : c0 + pw] = patch
            occ[max(0, r0 - PLACEMENT_GAP) : min(h, r0 + ph + PLACEMENT_GAP),
                max(0, c0 - PLACEMENT_GAP) : min(w, c0 + pw + PLACEMENT_GAP)] = True
            cursor += ph + LEGEND_GAP
            if i + 1 < len(extras):
                cursor += extras[i + 1]


def _candidate_positions(occ: np.ndarray, patch: np.ndarray):
    h, w = occ.shape
    ph, pw = patch.shape
    positions = []
    for r0 in range(1, h - ph):
        for c0 in range(1, w - pw):
            if occ[r0 - PLACEMENT_GAP : r0 + ph + PLACEMENT_GAP, c0 - PLACEMENT_GAP : c0 + pw + PLACEMENT_GAP].any():
                continue
            positions.append((r0, c0))
    return positions


def _place_targets(grid: np.ndarray, occ: np.ndarray, patches, colors, rng: random.Random):
    order = sorted(range(len(patches)), key=lambda i: patches[i].shape[0] * patches[i].shape[1], reverse=True)
    placements = []
    for idx in order:
        patch = patches[idx]
        positions = _candidate_positions(occ, patch)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        ph, pw = patch.shape
        grid[r0 : r0 + ph, c0 : c0 + pw] = patch
        occ[r0 - PLACEMENT_GAP : r0 + ph + PLACEMENT_GAP, c0 - PLACEMENT_GAP : c0 + pw + PLACEMENT_GAP] = True
        placements.append((r0, c0, ph, pw, colors[idx]))
    return placements


def _expected_output(inp: np.ndarray, placements):
    out = inp.copy()
    for r0, c0, ph, pw, color in placements:
        out[r0, c0 : c0 + pw] = color
        out[r0 + ph - 1, c0 : c0 + pw] = color
        out[r0 : r0 + ph, c0] = color
        out[r0 : r0 + ph, c0 + pw - 1] = color
    return out


def _gen_attempt():
    rng = random.Random()
    side = rng.choice(["top", "bottom", "left", "right"])
    shape_count = rng.randint(2, 4)
    chosen_shapes = rng.sample(BASE_HOLES, shape_count)
    legend_colors = rng.sample([c for c in range(1, 10) if c != GRAY], shape_count)

    target_shape_indices = list(range(shape_count))
    duplicate_count = rng.randint(0, 2)
    for _ in range(duplicate_count):
        target_shape_indices.append(rng.randrange(shape_count))
    rng.shuffle(target_shape_indices)

    legend_patches = [_build_legend_patch(shape, color, side, rng) for shape, color in zip(chosen_shapes, legend_colors)]
    target_patches = [_build_target_patch(chosen_shapes[i], rng) for i in target_shape_indices]
    target_colors = [legend_colors[i] for i in target_shape_indices]

    legend_span = (
        sum(p.shape[1] for p in legend_patches) + LEGEND_GAP * (len(legend_patches) - 1)
        if side in ("top", "bottom")
        else sum(p.shape[0] for p in legend_patches) + LEGEND_GAP * (len(legend_patches) - 1)
    )
    legend_strip = max((p.shape[0] if side in ("top", "bottom") else p.shape[1]) for p in legend_patches)
    max_target_h = max(p.shape[0] for p in target_patches)
    max_target_w = max(p.shape[1] for p in target_patches)
    total_target_area = sum(p.shape[0] * p.shape[1] for p in target_patches)

    h_min = max(GRID_MIN, legend_strip + 6, max_target_h + 3)
    w_min = max(GRID_MIN, max_target_w + 3)
    if side in ("top", "bottom"):
        w_min = max(w_min, legend_span)
        h_min = max(h_min, min(30, 8 + int(np.ceil(total_target_area / max(8, w_min - 4)))))
    else:
        h_min = max(h_min, legend_span)
        w_min = max(w_min, min(30, 8 + int(np.ceil(total_target_area / max(8, h_min - 4)))))

    if h_min > GRID_MAX or w_min > GRID_MAX:
        return None

    h = rng.randint(h_min, GRID_MAX)
    w = rng.randint(w_min, GRID_MAX)
    grid = np.full((h, w), BG, dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    _layout_legends(grid, occ, legend_patches, side, rng)
    placements = _place_targets(grid, occ, target_patches, target_colors, rng)
    if placements is None:
        return None

    expected = _expected_output(grid, placements)
    try:
        predicted = main(grid)
    except Exception:
        return None
    if not np.array_equal(predicted, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_RETRIES} attempts")
