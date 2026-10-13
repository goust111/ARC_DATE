import random
from collections import deque

import numpy as np


BLACK = 0
PALETTE_KEYS = (1, 2, 3)
PALETTE_VALUES = (4, 5, 7)
PALETTE_MAP = dict(zip(PALETTE_KEYS, PALETTE_VALUES))
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
MAX_GENERATION_ATTEMPTS = 400

HOLLOW_SHAPES = (
    np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1, 1], [1, 0, 0, 1], [1, 1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
)

SOLID_SHAPES = (
    np.array([[1, 1, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1], [1], [1]], dtype=bool),
    np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
)

BLOCKER_SHAPES = (
    np.array([[1, 1, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool),
)


def _bbox(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _component_coords(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(coords)
    return components


def _read_palette(grid):
    assert grid.ndim == 2 and grid.shape[0] >= 2
    cols = np.flatnonzero(np.any(grid[:2] != BLACK, axis=0))
    assert cols.size > 0, "expected a top-edge palette"
    mapping = {}
    for c in cols:
        key = int(grid[0, c])
        value = int(grid[1, c])
        if key != BLACK and value != BLACK:
            mapping[key] = value
    assert mapping, "palette must define at least one mapping"
    return mapping, int(cols[0]), int(cols[-1])


def _enclosed_holes(body, coords):
    r0, c0, r1, c1 = _bbox(coords)
    window = body[r0 : r1 + 1, c0 : c1 + 1]
    bg_mask = window == BLACK
    holes = []
    for region in _component_coords(bg_mask):
        if any(
            rr == 0
            or cc == 0
            or rr == bg_mask.shape[0] - 1
            or cc == bg_mask.shape[1] - 1
            for rr, cc in region
        ):
            continue
        holes.append([(r0 + rr, c0 + cc) for rr, cc in region])
    return holes


def _analyze_objects(body):
    objects = []
    for coords in _component_coords(body != BLACK):
        color = int(body[coords[0][0], coords[0][1]])
        holes = _enclosed_holes(body, coords)
        objects.append(
            {
                "coords": coords,
                "coord_set": set(coords),
                "color": color,
                "bbox": _bbox(coords),
                "holes": holes,
            }
        )
    return objects


def _hole_columns(holes):
    start_by_col = {}
    for hole in holes:
        for r, c in hole:
            start_by_col[c] = max(start_by_col.get(c, -1), r)
    return start_by_col


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow object via the palette, then cast its fill color downward until another object blocks it.

    Input structure:
    - A black grid contains a 2x3 palette legend on the top edge and 4-5 detached monochrome objects below it.
    - Object colors come from the palette's first row; some objects are hollow frames with enclosed black holes, others are solid distractors.
    - Hollow objects do not share the same ray columns, and some have another object somewhere below them.

    Rule:
    - Read the top-edge palette as border-color -> fill-color.
    - Erase the palette area to black, then find every non-black object below it.
    - For each object with one or more enclosed black regions, fill those holes with the palette color mapped from that object's border color.
    - For each occupied hole column, continue downward beneath the same object and paint black cells with that fill color until a different object is reached or the bottom edge is reached.
    - Leave solid objects unchanged apart from acting as blockers.

    Key insight:
    - Only hollow objects create new colored source pixels, and those filled pixels define the columns of the downward rays while other objects merely stop the rays.
    """
    g = np.asarray(grid, dtype=int)
    mapping, c0, c1 = _read_palette(g)

    body = g.copy()
    body[:2, c0 : c1 + 1] = BLACK
    out = body.copy()
    h, _ = body.shape

    for obj in _analyze_objects(body):
        if not obj["holes"]:
            continue
        fill_color = int(mapping[obj["color"]])
        for hole in obj["holes"]:
            for r, c in hole:
                out[r, c] = fill_color

        for c, start_r in _hole_columns(obj["holes"]).items():
            rr = start_r + 1
            while rr < h and (rr, c) in obj["coord_set"]:
                rr += 1
            while rr < h and body[rr, c] == BLACK:
                out[rr, c] = fill_color
                rr += 1

    return out


def _mask_variants(mask):
    variants = []
    seen = set()
    for k in range(4):
        rotated = np.rot90(mask, k)
        for variant in (rotated, np.fliplr(rotated)):
            key = variant.tobytes() + bytes(variant.shape)
            if key not in seen:
                seen.add(key)
                variants.append(variant.copy())
    return variants


def _absolute_coords(mask, top, left):
    rows, cols = np.where(mask)
    return [(top + int(r), left + int(c)) for r, c in zip(rows, cols)]


def _can_place(blocked, coords):
    h, w = blocked.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if blocked[r, c]:
            return False
    return True


def _mark_blocked(blocked, coords):
    h, w = blocked.shape
    for r, c in coords:
        blocked[r, c] = True
        for dr, dc in NEIGHBORS8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                blocked[nr, nc] = True


def _paint_object(grid, blocked, mask, color, top, left):
    coords = _absolute_coords(mask, top, left)
    if not _can_place(blocked, coords):
        return None
    for r, c in coords:
        grid[r, c] = color
    _mark_blocked(blocked, coords)
    return {
        "mask": mask,
        "color": color,
        "top": top,
        "left": left,
        "coords": coords,
        "bbox": _bbox(coords),
        "holes": _enclosed_holes(grid, coords),
    }


def _sample_hollows(n_hollows):
    masks = []
    for _ in range(n_hollows):
        base = random.choice(HOLLOW_SHAPES)
        masks.append(random.choice(_mask_variants(base)))
    random.shuffle(masks)
    return masks


def _distributed_lefts(width, masks):
    widths = [mask.shape[1] for mask in masks]
    required = sum(widths) + max(0, len(widths) - 1)
    if required > width:
        return None
    slack = width - required
    extras = [0] * (len(widths) + 1)
    for _ in range(slack):
        extras[random.randrange(len(extras))] += 1
    lefts = []
    cursor = extras[0]
    for idx, mask_w in enumerate(widths):
        lefts.append(cursor)
        cursor += mask_w + 1 + extras[idx + 1]
    return lefts


def _blocked_by_open_columns(coords, open_cols, min_row):
    if not open_cols:
        return False
    return any(r > min_row and c in open_cols for r, c in coords)


def _place_targeted_blocker(grid, blocked, hollow, open_cols, open_min_row):
    h, w = grid.shape
    ray_cols = sorted(_hole_columns(hollow["holes"]))
    top_min = hollow["bbox"][2] + 2
    candidates = []
    for base in BLOCKER_SHAPES:
        for mask in _mask_variants(base):
            mh, mw = mask.shape
            for top in range(top_min, h - mh + 1):
                left_lo = max(0, max(ray_cols) - mw + 1)
                left_hi = min(min(ray_cols), w - mw)
                for left in range(left_lo, left_hi + 1):
                    coords = _absolute_coords(mask, top, left)
                    occ_cols = {c for _, c in coords}
                    if not set(ray_cols).issubset(occ_cols):
                        continue
                    if _blocked_by_open_columns(coords, open_cols, open_min_row):
                        continue
                    if _can_place(blocked, coords):
                        candidates.append((mask, top, left))
    if not candidates:
        return None
    mask, top, left = random.choice(candidates)
    return _paint_object(grid, blocked, mask, random.choice(PALETTE_KEYS), top, left)


def _place_random_solid(grid, blocked, open_cols, open_min_row):
    h, w = grid.shape
    for _ in range(200):
        mask = random.choice(_mask_variants(random.choice(SOLID_SHAPES)))
        mh, mw = mask.shape
        top = random.randint(3, h - mh)
        left = random.randint(0, w - mw)
        coords = _absolute_coords(mask, top, left)
        if _blocked_by_open_columns(coords, open_cols, open_min_row):
            continue
        if _can_place(blocked, coords):
            return _paint_object(grid, blocked, mask, random.choice(PALETTE_KEYS), top, left)
    return None


def _ray_hits_blocker(body, obj):
    h, _ = body.shape
    statuses = []
    for c, start_r in _hole_columns(obj["holes"]).items():
        rr = start_r + 1
        while rr < h and (rr, c) in obj["coord_set"]:
            rr += 1
        while rr < h and body[rr, c] == BLACK:
            rr += 1
        statuses.append(rr < h)
    return all(statuses), not any(statuses)


def _validate_generated(grid):
    mapping, c0, c1 = _read_palette(grid)
    assert mapping == PALETTE_MAP

    body = grid.copy()
    body[:2, c0 : c1 + 1] = BLACK
    objects = _analyze_objects(body)
    assert 4 <= len(objects) <= 5
    hollows = [obj for obj in objects if obj["holes"]]
    solids = [obj for obj in objects if not obj["holes"]]
    assert 2 <= len(hollows) <= 3
    assert len(solids) >= 1

    seen_cols = set()
    for obj in hollows:
        cols = set(_hole_columns(obj["holes"]))
        assert cols.isdisjoint(seen_cols)
        seen_cols |= cols

    branches = [_ray_hits_blocker(body, obj) for obj in hollows]
    assert any(blocked for blocked, _ in branches)
    assert any(open_path for _, open_path in branches)

    out = main(grid)
    assert out.shape == grid.shape
    assert np.any(out != grid)
    assert np.all(out[:2, c0 : c1 + 1] == BLACK)


def _gen_attempt():
    height = random.randint(14, 18)
    width = random.randint(17, 22)
    grid = np.zeros((height, width), dtype=int)

    palette_left = random.randint(0, width - 3)
    grid[0, palette_left : palette_left + 3] = PALETTE_KEYS
    grid[1, palette_left : palette_left + 3] = PALETTE_VALUES

    blocked = np.zeros((height, width), dtype=bool)
    blocked[:3, :] = True

    n_objects = random.randint(4, 5)
    n_hollows = 3 if n_objects == 5 and random.random() < 0.6 else 2
    if n_objects == 4 and random.random() < 0.35:
        n_hollows = 3
    n_solids = n_objects - n_hollows

    hollow_masks = _sample_hollows(n_hollows)
    lefts = _distributed_lefts(width, hollow_masks)
    if lefts is None:
        return None

    hollow_colors = random.sample(PALETTE_KEYS, k=n_hollows)
    hollows = []
    blocker_idx = random.randrange(n_hollows)
    open_idx = (blocker_idx + random.randint(1, n_hollows - 1)) % n_hollows

    for idx, (mask, left, color) in enumerate(zip(hollow_masks, lefts, hollow_colors)):
        mh, _ = mask.shape
        max_top = height - mh - 2
        if idx in (blocker_idx, open_idx):
            max_top = min(max_top, height - mh - 7)
        if max_top < 3:
            return None
        top = random.randint(3, max_top)
        obj = _paint_object(grid, blocked, mask, color, top, left)
        if obj is None:
            return None
        hollows.append(obj)

    open_cols = set(_hole_columns(hollows[open_idx]["holes"]))
    open_min_row = hollows[open_idx]["bbox"][2]

    blocker = _place_targeted_blocker(
        grid,
        blocked,
        hollows[blocker_idx],
        open_cols,
        open_min_row,
    )
    if blocker is None:
        return None

    for _ in range(n_solids - 1):
        solid = _place_random_solid(grid, blocked, open_cols, open_min_row)
        if solid is None:
            return None

    _validate_generated(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
