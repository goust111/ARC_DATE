from __future__ import annotations

import numpy as np


BG = 0
FRAME = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    h, w = mask.shape
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        stack = [(sr, sc)]
        seen[sr, sc] = True
        comp: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]] | np.ndarray) -> tuple[int, int, int, int]:
    arr = np.array(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_frame(g: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for color in np.unique(g):
        color = int(color)
        if color == BG:
            continue
        for comp in _components(g == color):
            r0, c0, r1, c1 = _bbox(comp)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 3 or w < 3:
                continue
            border = np.zeros((h, w), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if len(comp) != int(border.sum()):
                continue
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            if np.array_equal(sub == color, border):
                candidates.append((len(comp), color, (r0, c0, r1, c1)))

    assert candidates, "expected one hollow rectangular frame"
    best_area = max(area for area, _color, _box in candidates)
    best = [item for item in candidates if item[0] == best_area]
    assert len(best) == 1, "frame must be the unique largest hollow rectangle"
    _area, color, box = best[0]
    return int(color), box


def _external_template_mask(g: np.ndarray, color: int, frame_box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = frame_box
    outside = np.ones(g.shape, dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    mask = (g == color) & outside
    comps = _components(mask)
    assert len(comps) == 1, "the selected template color must appear once outside the frame"
    tr0, tc0, tr1, tc1 = _bbox(comps[0])
    patch = mask[tr0 : tr1 + 1, tc0 : tc1 + 1]
    assert patch.shape[0] == patch.shape[1], "candidate template bbox must be square"
    return patch


def _rotation_from_seeds(seed_coords: np.ndarray, shape: tuple[int, int]) -> int:
    h, w = shape
    center = np.array([(h - 1) / 2.0, (w - 1) / 2.0])
    centroid = seed_coords.astype(float).mean(axis=0)
    dr, dc = centroid - center
    assert abs(float(dr)) > 1e-9 and abs(float(dc)) > 1e-9, "seed centroid must lie in a quadrant"
    if dr < 0 and dc < 0:
        return 0
    if dr < 0 and dc > 0:
        return -1
    if dr > 0 and dc > 0:
        return 2
    return 1


def _scaled_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mh, mw = mask.shape
    assert h % mh == 0 and w % mw == 0, "frame interior must be an integer scale of the template"
    sh, sw = h // mh, w // mw
    assert sh == sw, "template must use a uniform integer scale"
    return np.kron(mask, np.ones((sh, sw), dtype=bool))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the frame with the outside template selected by seed color and oriented by seed quadrant.

    Input structure:
    - A black background contains one hollow grey rectangular frame.
    - The frame interior is black except for several same-colored seed pixels.
    - Outside the frame are multiple monochrome connected candidate templates with distinct colors and square bounding boxes.

    Rule:
    - Find the seed color inside the frame, then crop the outside template drawn in that same color.
    - Compare the seed-pixel centroid with the frame center: upper-left means no rotation, upper-right means clockwise 90 degrees, lower-right means 180 degrees, and lower-left means clockwise 270 degrees.
    - Rotate the selected template, scale it by one integer factor so it exactly covers the frame interior, clear the old interior, and paint the scaled mask with the seed color.
    - Leave the frame and all outside templates unchanged.

    Key insight:
    The seed pixels encode both choices: their color selects the template, while their spatial quadrant selects the rotation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    frame_color, (r0, c0, r1, c1) = _find_frame(g)
    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    seed_colors = set(map(int, np.unique(interior))) - {BG, frame_color}
    assert len(seed_colors) == 1, "frame interior must contain one seed color"
    color = seed_colors.pop()

    seed_coords = np.argwhere(interior == color)
    assert len(seed_coords) > 0, "expected seed pixels"
    template = _external_template_mask(g, color, (r0, c0, r1, c1))
    assert int(template.sum()) == int(len(seed_coords)), "seed count must match template area"

    rot_k = _rotation_from_seeds(seed_coords, interior.shape)
    rotated = np.rot90(template, rot_k)
    scaled = _scaled_mask(rotated, interior.shape)

    out = g.copy()
    target = out[r0 + 1 : r1, c0 + 1 : c1]
    target[:, :] = BG
    target[scaled] = color
    return out


# --------------- generator ---------------

GEN_ATTEMPTS = 500
TEMPLATE_COUNTS = [3, 4, 5]
TEMPLATE_SIZES = [3, 4, 5]
SCALES = [2, 3, 4]
BAND_SIDES = ["top", "right", "bottom", "left"]
MIN_GAP = 1
_RNG = np.random.default_rng()


def _rng_int(lo: int, hi: int) -> int:
    return int(_RNG.integers(lo, hi + 1))


def _is_full_bbox(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _is_connected(mask: np.ndarray) -> bool:
    comps = _components(mask)
    return len(comps) == 1 and len(comps[0]) == int(mask.sum())


def _grow_from(mask: np.ndarray, target_area: int) -> np.ndarray:
    out = mask.copy()
    size = out.shape[0]
    while int(out.sum()) < target_area:
        frontier: set[tuple[int, int]] = set()
        for r, c in np.argwhere(out):
            r, c = int(r), int(c)
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size and not out[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            break
        choices = sorted(frontier)
        nr, nc = choices[_rng_int(0, len(choices) - 1)]
        out[nr, nc] = True
    return out


def _fallback_mask(size: int, area: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    r = _rng_int(0, size - 1)
    c = _rng_int(0, size - 1)
    mask[:, c] = True
    mask[r, :] = True
    return _grow_from(mask, area)


def _random_template_mask(size: int) -> np.ndarray:
    min_area = 2 * size - 1
    max_area = min(size * size, 2 * size + 5)
    area = _rng_int(min_area, max_area)

    for _ in range(80):
        mask = np.zeros((size, size), dtype=bool)
        mask[_rng_int(0, size - 1), _rng_int(0, size - 1)] = True
        mask = _grow_from(mask, area)
        if _is_full_bbox(mask) and _is_connected(mask):
            return mask

    mask = _fallback_mask(size, area)
    assert _is_full_bbox(mask) and _is_connected(mask)
    return mask


def _linear_starts(lengths: list[int], span: int) -> list[int] | None:
    total = sum(lengths) + MIN_GAP * (len(lengths) - 1)
    if total > span:
        return None
    slack = span - total
    extras = _RNG.multinomial(slack, np.ones(len(lengths) + 1) / (len(lengths) + 1))
    starts: list[int] = []
    pos = int(extras[0])
    for i, length in enumerate(lengths):
        starts.append(pos)
        pos += int(length)
        if i < len(lengths) - 1:
            pos += MIN_GAP + int(extras[i + 1])
    return starts


def _paint_mask(grid: np.ndarray, mask: np.ndarray, color: int, r0: int, c0: int) -> None:
    h, w = mask.shape
    patch = grid[r0 : r0 + h, c0 : c0 + w]
    assert not np.any(patch[mask] != BG)
    patch[mask] = color


def _draw_frame(grid: np.ndarray, r0: int, c0: int, size: int) -> None:
    grid[r0, c0 : c0 + size] = FRAME
    grid[r0 + size - 1, c0 : c0 + size] = FRAME
    grid[r0 : r0 + size, c0] = FRAME
    grid[r0 : r0 + size, c0 + size - 1] = FRAME


def _frame_origin(side: str, h: int, w: int, frame_size: int, max_template: int) -> tuple[int, int] | None:
    if side == "top":
        min_r = max_template + 2
        max_r = h - frame_size
        if min_r > max_r:
            return None
        return _rng_int(min_r, max_r), _rng_int(0, w - frame_size)
    if side == "bottom":
        max_r = h - frame_size - max_template - 2
        if max_r < 0:
            return None
        return _rng_int(0, max_r), _rng_int(0, w - frame_size)
    if side == "left":
        min_c = max_template + 2
        max_c = w - frame_size
        if min_c > max_c:
            return None
        return _rng_int(0, h - frame_size), _rng_int(min_c, max_c)
    max_c = w - frame_size - max_template - 2
    if max_c < 0:
        return None
    return _rng_int(0, h - frame_size), _rng_int(0, max_c)


def _place_templates(
    grid: np.ndarray,
    templates: list[dict],
    side: str,
    frame_box: tuple[int, int, int, int],
) -> bool:
    h, w = grid.shape
    r0, c0, r1, c1 = frame_box
    sizes = [int(t["mask"].shape[0]) for t in templates]
    if side in {"top", "bottom"}:
        starts = _linear_starts(sizes, w)
        if starts is None:
            return False
        if side == "top":
            band_top = 0
            band_h = r0 - 1
        else:
            band_top = r1 + 2
            band_h = h - band_top
        if band_h < max(sizes):
            return False
        for t, c_start, size in zip(templates, starts, sizes):
            rr = band_top + _rng_int(0, band_h - size)
            _paint_mask(grid, t["mask"], int(t["color"]), rr, int(c_start))
        return True

    starts = _linear_starts(sizes, h)
    if starts is None:
        return False
    if side == "left":
        band_left = 0
        band_w = c0 - 1
    else:
        band_left = c1 + 2
        band_w = w - band_left
    if band_w < max(sizes):
        return False
    for t, r_start, size in zip(templates, starts, sizes):
        cc = band_left + _rng_int(0, band_w - size)
        _paint_mask(grid, t["mask"], int(t["color"]), int(r_start), cc)
    return True


def _seed_cells(inner_size: int, count: int, quadrant: int) -> list[tuple[int, int]] | None:
    center = (inner_size - 1) / 2.0
    top_rows = [r for r in range(inner_size) if r < center]
    bottom_rows = [r for r in range(inner_size) if r > center]
    left_cols = [c for c in range(inner_size) if c < center]
    right_cols = [c for c in range(inner_size) if c > center]
    if quadrant == 0:
        rows, cols = top_rows, left_cols
    elif quadrant == 1:
        rows, cols = top_rows, right_cols
    elif quadrant == 2:
        rows, cols = bottom_rows, right_cols
    else:
        rows, cols = bottom_rows, left_cols
    cells = [(r, c) for r in rows for c in cols]
    if len(cells) < count:
        return None
    pick = _RNG.choice(len(cells), size=count, replace=False)
    return [cells[int(i)] for i in pick]


def _expected_output(
    grid: np.ndarray,
    target_mask: np.ndarray,
    color: int,
    frame_box: tuple[int, int, int, int],
    scale: int,
    quadrant: int,
) -> np.ndarray:
    r0, c0, r1, c1 = frame_box
    rot_k = [0, -1, 2, 1][quadrant]
    rotated = np.rot90(target_mask, rot_k)
    scaled = np.kron(rotated, np.ones((scale, scale), dtype=bool))
    out = grid.copy()
    interior = out[r0 + 1 : r1, c0 + 1 : c1]
    interior[:, :] = BG
    interior[scaled] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    count = TEMPLATE_COUNTS[_rng_int(0, len(TEMPLATE_COUNTS) - 1)]
    colors = list(_RNG.choice(PALETTE, size=count, replace=False))
    target_idx = _rng_int(0, count - 1)
    target_size = TEMPLATE_SIZES[_rng_int(0, len(TEMPLATE_SIZES) - 1)]
    scale = SCALES[_rng_int(0, len(SCALES) - 1)]
    inner_size = target_size * scale
    frame_size = inner_size + 2
    quadrant = _rng_int(0, 3)

    templates: list[dict] = []
    used_shapes: set[tuple[int, bytes]] = set()
    for i in range(count):
        size = target_size if i == target_idx else TEMPLATE_SIZES[_rng_int(0, len(TEMPLATE_SIZES) - 1)]
        for _ in range(50):
            mask = _random_template_mask(size)
            key = (size, mask.tobytes())
            if key not in used_shapes:
                used_shapes.add(key)
                break
        else:
            return None
        templates.append({"mask": mask, "color": int(colors[i]), "target": i == target_idx})

    _RNG.shuffle(templates)
    target = next(t for t in templates if bool(t["target"]))
    target_area = int(target["mask"].sum())
    seeds = _seed_cells(inner_size, target_area, quadrant)
    if seeds is None:
        return None

    side = BAND_SIDES[_rng_int(0, len(BAND_SIDES) - 1)]
    sizes = [int(t["mask"].shape[0]) for t in templates]
    max_template = max(sizes)
    linear_span = sum(sizes) + MIN_GAP * (len(sizes) - 1)
    if side in {"top", "bottom"}:
        min_h = frame_size + max_template + 2
        min_w = max(frame_size, linear_span)
    else:
        min_h = max(frame_size, linear_span)
        min_w = frame_size + max_template + 2
    if min_h > 30 or min_w > 30:
        return None
    h = _rng_int(max(16, min_h), 30)
    w = _rng_int(max(16, min_w), 30)

    origin = _frame_origin(side, h, w, frame_size, max_template)
    if origin is None:
        return None
    fr, fc = origin
    frame_box = (fr, fc, fr + frame_size - 1, fc + frame_size - 1)

    grid = np.full((h, w), BG, dtype=int)
    _draw_frame(grid, fr, fc, frame_size)
    if not _place_templates(grid, templates, side, frame_box):
        return None

    for rr, cc in seeds:
        grid[fr + 1 + rr, fc + 1 + cc] = int(target["color"])

    expected = _expected_output(grid, target["mask"], int(target["color"]), frame_box, scale, quadrant)
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
    raise RuntimeError("failed to generate a valid seed385645705/2/1 input")
