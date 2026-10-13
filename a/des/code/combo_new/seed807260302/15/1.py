from __future__ import annotations

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
BG = 0
FRAME_COLOR = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_from_coords(coords: list[tuple[int, int]], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _find_monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            bbox = _bbox_from_coords(coords)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox,
                    "area": len(coords),
                }
            )
    return comps


def _find_square_frame(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for comp in _find_monochrome_components(grid, bg):
        r0, c0, r1, c1 = comp["bbox"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h != w or h < 4:
            continue
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if int(comp["area"]) == int(border.sum()) and np.array_equal(sub == int(comp["color"]), border):
            candidates.append((int(comp["area"]), int(comp["color"]), (r0, c0, r1, c1)))

    assert candidates, "expected a hollow square frame"
    largest_area = max(area for area, _color, _bbox in candidates)
    largest = [item for item in candidates if item[0] == largest_area]
    assert len(largest) == 1, "frame must be structurally unique"
    _, color, bbox = largest[0]
    return color, bbox


def _quadrant_index(interior_shape: tuple[int, int], seed_rc: tuple[int, int]) -> int:
    h, w = interior_shape
    assert h == w and h % 2 == 0 and w % 2 == 0
    r, c = seed_rc
    half = h // 2
    if r < half and c < half:
        return 0
    if r < half and c >= half:
        return 1
    if r >= half and c >= half:
        return 2
    return 3


def _quadrant_to_ccw_turns(quadrant: int) -> int:
    return {0: 0, 1: 3, 2: 2, 3: 1}[int(quadrant)]


def _centered_footprint(interior_shape: tuple[int, int], mask: np.ndarray) -> tuple[np.ndarray, int, int]:
    ih, iw = interior_shape
    mh, mw = mask.shape
    assert mh <= ih and mw <= iw
    assert (ih - mh) % 2 == 0 and (iw - mw) % 2 == 0, "center alignment requires matching parity"
    top = (ih - mh) // 2
    left = (iw - mw) // 2
    footprint = np.zeros((ih, iw), dtype=bool)
    footprint[top : top + mh, left : left + mw][mask] = True
    return footprint, top, left


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the large frame with the color-matched outer template, rotated by the seed pixel's quadrant.

    Input structure:
    - The background is black.
    - One large grey hollow square frame appears in the grid, with an even side length and a black interior.
    - Exactly one non-black, non-grey seed pixel lies inside the frame interior.
    - Outside the frame are several small monochrome connected template objects, each in a distinct non-grey color.

    Rule:
    - Find the unique seed pixel inside the frame and determine which interior quadrant it occupies.
    - Use the seed color to select the outer template of the same color.
    - Rotate that template according to the seed quadrant: top-left -> 0 degrees, top-right -> 90 degrees clockwise, bottom-right -> 180 degrees, bottom-left -> 270 degrees clockwise.
    - Center the rotated template's bounding box inside the frame interior and paint it there, while keeping the original seed pixel.

    Key insight:
    - The seed color decides which template to use, while the seed position decides how that template must be rotated before being inserted.
    """
    g = np.asarray(grid, dtype=int)
    bg = BG
    frame_color, (r0, c0, r1, c1) = _find_square_frame(g, bg)

    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    seed_coords = np.argwhere((interior != bg) & (interior != frame_color))
    assert len(seed_coords) == 1, "frame interior must contain exactly one seed pixel"
    seed_r, seed_c = map(int, seed_coords[0])
    seed_color = int(interior[seed_r, seed_c])
    quadrant = _quadrant_index(interior.shape, (seed_r, seed_c))

    template_candidates = []
    for comp in _find_monochrome_components(g, bg):
        if int(comp["color"]) != seed_color:
            continue
        outside = all(not (r0 <= rr <= r1 and c0 <= cc <= c1) for rr, cc in comp["coords"])
        if outside:
            template_candidates.append(comp)
    assert len(template_candidates) == 1, "seed color must match exactly one outer template"

    template = template_candidates[0]
    template_mask = _mask_from_coords(template["coords"], template["bbox"])
    rotated = np.rot90(template_mask, _quadrant_to_ccw_turns(quadrant))
    _, top, left = _centered_footprint(interior.shape, rotated)

    out = g.copy()
    target = out[r0 + 1 : r1, c0 + 1 : c1]
    rh, rw = rotated.shape
    target[top : top + rh, left : left + rw][rotated] = seed_color
    return out


# --------------- generator ---------------

_RNG = np.random.default_rng()

GRID_H_RANGE = (18, 31)
GRID_W_RANGE = (18, 31)
FRAME_SIDE_MIN = 8
FRAME_SIDE_MAX = 16
TEMPLATE_COUNT_RANGE = (3, 7)
GEN_RETRIES = 400
SHAPE_RETRIES = 100


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _random_path(rng: np.random.Generator, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    er, ec = end
    steps: list[tuple[int, int]] = []
    steps.extend([(1 if er > r else -1, 0)] * abs(er - r))
    steps.extend([(0, 1 if ec > c else -1)] * abs(ec - c))
    if steps:
        order = rng.permutation(len(steps))
        steps = [steps[int(i)] for i in order]
    out = [(r, c)]
    for dr, dc in steps:
        r += dr
        c += dc
        out.append((r, c))
    return out


def _canonical_shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    best: tuple[tuple[int, int], bytes] | None = None
    for k in range(4):
        rot = np.rot90(mask, k)
        for var in (rot, np.fliplr(rot)):
            key = (tuple(map(int, var.shape)), var.tobytes())
            if best is None or key < best:
                best = key
    assert best is not None
    return best


def _make_template_mask(
    rng: np.random.Generator,
    bbox_h: int,
    bbox_w: int,
) -> np.ndarray | None:
    for _ in range(SHAPE_RETRIES):
        anchors = [
            (0, int(rng.integers(0, bbox_w))),
            (bbox_h - 1, int(rng.integers(0, bbox_w))),
            (int(rng.integers(0, bbox_h)), 0),
            (int(rng.integers(0, bbox_h)), bbox_w - 1),
        ]
        order = list(map(int, rng.permutation(len(anchors))))
        cells = {anchors[order[0]]}
        for idx in order[1:]:
            anchor = anchors[idx]
            target = min(cells, key=lambda p: abs(p[0] - anchor[0]) + abs(p[1] - anchor[1]) + rng.random() * 0.01)
            cells.update(_random_path(rng, anchor, target))

        base_area = len(cells)
        box_area = bbox_h * bbox_w
        lo = max(base_area, max(bbox_h, bbox_w), 3)
        hi = box_area if box_area == lo else max(lo, box_area - int(rng.random() < 0.8))
        if lo > hi:
            continue
        target_area = int(rng.integers(lo, hi + 1))

        while len(cells) < target_area:
            frontier = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < bbox_h and 0 <= nc < bbox_w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            options = list(frontier)
            weights = np.array(
                [1 + sum((r + dr, c + dc) in cells for dr, dc in DIR4) for r, c in options],
                dtype=float,
            )
            pick = int(rng.choice(len(options), p=weights / weights.sum()))
            cells.add(options[pick])

        mask = np.zeros((bbox_h, bbox_w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        if not (mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue

        border = np.zeros_like(mask)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if np.array_equal(mask, border):
            continue
        return mask
    return None


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    grid[r0 : r0 + h, c0 : c0 + w][mask] = int(color)


def _place_patch(
    grid: np.ndarray,
    forbidden: np.ndarray,
    mask: np.ndarray,
    color: int,
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    h, w = mask.shape
    gh, gw = grid.shape
    rows = list(range(gh - h + 1))
    cols = list(range(gw - w + 1))
    positions = [(r, c) for r in rows for c in cols]
    if not positions:
        return None
    order = list(map(int, rng.permutation(len(positions))))
    for idx in order:
        r0, c0 = positions[idx]
        if bool((forbidden[r0 : r0 + h, c0 : c0 + w] & mask).any()):
            continue
        _paint_mask(grid, mask, r0, c0, color)
        placed = np.zeros_like(forbidden)
        placed[r0 : r0 + h, c0 : c0 + w][mask] = True
        forbidden |= _dilate8(placed)
        return r0, c0
    return None


def _quadrant_cells(side: int, quadrant: int) -> np.ndarray:
    half = side // 2
    mask = np.zeros((side, side), dtype=bool)
    if quadrant == 0:
        mask[:half, :half] = True
    elif quadrant == 1:
        mask[:half, half:] = True
    elif quadrant == 2:
        mask[half:, half:] = True
    else:
        mask[half:, :half] = True
    return mask


def _build_expected(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    template_mask: np.ndarray,
    seed_color: int,
    quadrant: int,
) -> np.ndarray:
    r0, c0, r1, c1 = frame_bbox
    out = grid.copy()
    interior = out[r0 + 1 : r1, c0 + 1 : c1]
    rotated = np.rot90(template_mask, _quadrant_to_ccw_turns(quadrant))
    _, top, left = _centered_footprint(interior.shape, rotated)
    rh, rw = rotated.shape
    interior[top : top + rh, left : left + rw][rotated] = int(seed_color)
    return out


def _generate_attempt() -> np.ndarray | None:
    rng = _RNG
    grid_h = int(rng.integers(*GRID_H_RANGE))
    grid_w = int(rng.integers(*GRID_W_RANGE))

    feasible_sides = [
        side
        for side in range(FRAME_SIDE_MIN, min(FRAME_SIDE_MAX, grid_h - 2, grid_w - 2) + 1, 2)
    ]
    if not feasible_sides:
        return None
    frame_side = int(feasible_sides[int(rng.integers(0, len(feasible_sides)))])
    inner_side = frame_side - 2

    dims = []
    max_dim = min(inner_side, 8)
    for h in range(2, max_dim + 1, 2):
        for w in range(2, max_dim + 1, 2):
            if h * w <= (inner_side * inner_side) // 2:
                dims.append((h, w))
    if not dims:
        return None

    template_count = int(rng.integers(TEMPLATE_COUNT_RANGE[0], TEMPLATE_COUNT_RANGE[1]))
    if template_count > len(PALETTE):
        return None

    colors = list(PALETTE)
    rng.shuffle(colors)
    template_colors = colors[:template_count]

    templates: list[dict] = []
    seen_shapes: set[tuple[tuple[int, int], bytes]] = set()
    for color in template_colors:
        for _ in range(SHAPE_RETRIES):
            bbox_h, bbox_w = dims[int(rng.integers(0, len(dims)))]
            mask = _make_template_mask(rng, bbox_h, bbox_w)
            if mask is None:
                continue
            key = _canonical_shape_key(mask)
            if key in seen_shapes:
                continue
            seen_shapes.add(key)
            templates.append({"color": int(color), "mask": mask})
            break
        else:
            return None

    frame_r0 = int(rng.integers(1, grid_h - frame_side + 1))
    frame_c0 = int(rng.integers(1, grid_w - frame_side + 1))
    frame_bbox = (frame_r0, frame_c0, frame_r0 + frame_side - 1, frame_c0 + frame_side - 1)

    grid = np.zeros((grid_h, grid_w), dtype=int)
    grid[frame_r0, frame_c0 : frame_c0 + frame_side] = FRAME_COLOR
    grid[frame_r0 + frame_side - 1, frame_c0 : frame_c0 + frame_side] = FRAME_COLOR
    grid[frame_r0 : frame_r0 + frame_side, frame_c0] = FRAME_COLOR
    grid[frame_r0 : frame_r0 + frame_side, frame_c0 + frame_side - 1] = FRAME_COLOR

    forbidden = np.zeros_like(grid, dtype=bool)
    r_lo = max(0, frame_r0 - 1)
    r_hi = min(grid_h, frame_r0 + frame_side + 1)
    c_lo = max(0, frame_c0 - 1)
    c_hi = min(grid_w, frame_c0 + frame_side + 1)
    forbidden[r_lo:r_hi, c_lo:c_hi] = True

    order = sorted(range(len(templates)), key=lambda i: templates[i]["mask"].shape[0] * templates[i]["mask"].shape[1], reverse=True)
    placements: dict[int, tuple[int, int]] = {}
    for idx in order:
        tpl = templates[idx]
        placed = _place_patch(grid, forbidden, tpl["mask"], int(tpl["color"]), rng)
        if placed is None:
            return None
        placements[idx] = placed

    chosen_idx = int(rng.integers(0, len(templates)))
    chosen = templates[chosen_idx]
    quadrant = int(rng.integers(0, 4))
    seed_color = int(chosen["color"])
    rotated = np.rot90(chosen["mask"], _quadrant_to_ccw_turns(quadrant))
    footprint, _, _ = _centered_footprint((inner_side, inner_side), rotated)
    qmask = _quadrant_cells(inner_side, quadrant)
    on_cells = np.argwhere(footprint & qmask)
    off_cells = np.argwhere((~footprint) & qmask)
    prefer_on = bool(rng.integers(0, 2))
    if prefer_on and len(on_cells) > 0:
        pool = on_cells
    elif not prefer_on and len(off_cells) > 0:
        pool = off_cells
    else:
        pool = on_cells if len(on_cells) > 0 else off_cells
    if len(pool) == 0:
        return None
    seed_r, seed_c = map(int, pool[int(rng.integers(0, len(pool)))])
    grid[frame_r0 + 1 + seed_r, frame_c0 + 1 + seed_c] = seed_color

    expected = _build_expected(grid, frame_bbox, chosen["mask"], seed_color, quadrant)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
