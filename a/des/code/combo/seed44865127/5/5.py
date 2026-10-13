from __future__ import annotations

import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRECTIONS = ("up", "down", "left", "right")
OPPOSITE_STEP = {
    "up": (1, 0),
    "down": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}

GEN_RETRIES = 300
GRID_MIN = 12
GRID_MAX = 18
OBJECT_MIN = 2
OBJECT_MAX = 5
MASK_MIN_DIM = 3
MASK_MAX_DIM = 5
MASK_MIN_AREA = 4
MASK_MAX_AREA = 8


def _round_half_up(value: float) -> int:
    return int(np.floor(float(value) + 0.5))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == int(bg):
                continue
            color = int(g[r, c])
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            pts = np.asarray(coords, dtype=int)
            r0 = int(pts[:, 0].min())
            r1 = int(pts[:, 0].max())
            c0 = int(pts[:, 1].min())
            c1 = int(pts[:, 1].max())
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                }
            )
    return comps


def _direction_from_coords(coords: list[tuple[int, int]]) -> str:
    pts = np.asarray(coords, dtype=int)
    rows = pts[:, 0].astype(float)
    cols = pts[:, 1].astype(float)
    r0 = int(rows.min())
    r1 = int(rows.max())
    c0 = int(cols.min())
    c1 = int(cols.max())
    center_r = (float(r0) + float(r1)) / 2.0
    center_c = (float(c0) + float(c1)) / 2.0

    vis_r = _round_half_up(float(rows.mean()))
    vis_c = _round_half_up(float(cols.mean()))
    off_r = float(vis_r) - center_r
    off_c = float(vis_c) - center_c

    if abs(off_r) > abs(off_c) and abs(off_r) > 0:
        return "down" if off_r > 0 else "up"
    if abs(off_c) > abs(off_r) and abs(off_c) > 0:
        return "right" if off_c > 0 else "left"

    # Valid inputs are generated to avoid ties; this fallback only resolves edge cases.
    exact_r = float(rows.mean()) - center_r
    exact_c = float(cols.mean()) - center_c
    if abs(exact_r) > abs(exact_c) and abs(exact_r) > 1e-9:
        return "down" if exact_r > 0 else "up"
    if abs(exact_c) > abs(exact_r) and abs(exact_c) > 1e-9:
        return "right" if exact_c > 0 else "left"
    raise AssertionError("object direction is ambiguous")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each foreground object one cell opposite to its centroid bias.

    The bias direction is computed by comparing the rounded row/column means
    of the object's pixels with the center of its tight bounding box.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = map(int, g.shape)
    bg = _get_bg(g)
    comps = _find_components(g, bg)

    out = np.full((h, w), int(bg), dtype=int)
    occupied: set[tuple[int, int]] = set()

    for comp in comps:
        direction = _direction_from_coords(comp["coords"])
        dr, dc = OPPOSITE_STEP[direction]
        translated = []
        for r, c in comp["coords"]:
            nr = int(r + dr)
            nc = int(c + dc)
            assert 0 <= nr < h and 0 <= nc < w
            translated.append((nr, nc))

        for cell in translated:
            assert cell not in occupied
            occupied.add(cell)
            out[cell] = int(comp["color"])

    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0 = int(pts[:, 0].min())
    r1 = int(pts[:, 0].max())
    c0 = int(pts[:, 1].min())
    c1 = int(pts[:, 1].max())
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _mask_direction(mask: np.ndarray) -> str | None:
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    try:
        return _direction_from_coords(coords)
    except AssertionError:
        return None


def _mass_skew_ok(mask: np.ndarray, direction: str) -> bool:
    pts = np.argwhere(mask)
    rows = pts[:, 0].astype(float)
    cols = pts[:, 1].astype(float)
    center_r = (mask.shape[0] - 1) / 2.0
    center_c = (mask.shape[1] - 1) / 2.0

    if direction == "up":
        return int(np.sum(rows < center_r)) > int(np.sum(rows > center_r))
    if direction == "down":
        return int(np.sum(rows > center_r)) > int(np.sum(rows < center_r))
    if direction == "left":
        return int(np.sum(cols < center_c)) > int(np.sum(cols > center_c))
    return int(np.sum(cols > center_c)) > int(np.sum(cols < center_c))


def _exact_bias_ok(mask: np.ndarray, direction: str) -> bool:
    pts = np.argwhere(mask)
    mean_r = float(pts[:, 0].mean())
    mean_c = float(pts[:, 1].mean())
    center_r = (mask.shape[0] - 1) / 2.0
    center_c = (mask.shape[1] - 1) / 2.0
    diff_r = mean_r - center_r
    diff_c = mean_c - center_c

    if direction == "up":
        return diff_r < -0.12
    if direction == "down":
        return diff_r > 0.12
    if direction == "left":
        return diff_c < -0.12
    return diff_c > 0.12


def _random_mask_attempt(direction: str) -> np.ndarray | None:
    height = random.randint(MASK_MIN_DIM, MASK_MAX_DIM)
    width = random.randint(MASK_MIN_DIM, MASK_MAX_DIM)
    max_area = min(MASK_MAX_AREA, height * width - 1)
    if max_area < MASK_MIN_AREA:
        return None
    area = random.randint(MASK_MIN_AREA, max_area)

    if direction == "up":
        seed = (random.randrange(min(2, height)), random.randrange(width))
    elif direction == "down":
        seed = (height - 1 - random.randrange(min(2, height)), random.randrange(width))
    elif direction == "left":
        seed = (random.randrange(height), random.randrange(min(2, width)))
    else:
        seed = (random.randrange(height), width - 1 - random.randrange(min(2, width)))

    cells = {seed}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for dr, dc in DIR4:
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return None

        frontier = list(frontier)
        weights = []
        for r, c in frontier:
            if direction == "up":
                score = height - r
            elif direction == "down":
                score = r + 1
            elif direction == "left":
                score = width - c
            else:
                score = c + 1
            weights.append(float(score * score))
        cells.add(random.choices(frontier, weights=weights, k=1)[0])

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[int(r), int(c)] = True
    mask = _crop_mask(mask)

    if min(mask.shape) < 2:
        return None
    if int(mask.sum()) < MASK_MIN_AREA:
        return None
    if bool(mask.all()):
        return None
    if _mask_direction(mask) != direction:
        return None
    if not _mass_skew_ok(mask, direction):
        return None
    if not _exact_bias_ok(mask, direction):
        return None
    return mask


def _sample_mask(direction: str) -> np.ndarray:
    for _ in range(200):
        mask = _random_mask_attempt(direction)
        if mask is not None:
            return mask
    raise RuntimeError(f"failed to build a mask biased {direction}")


def _expand_8(cells: set[tuple[int, int]], height: int, width: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < height and 0 <= nc < width:
                    out.add((nr, nc))
    return out


def _place_objects(
    height: int,
    width: int,
    specs: list[dict],
) -> list[dict] | None:
    reserved: set[tuple[int, int]] = set()
    placed: list[dict] = []

    for spec in sorted(specs, key=lambda item: -int(item["mask"].sum())):
        mask = np.asarray(spec["mask"], dtype=bool)
        move_r, move_c = OPPOSITE_STEP[str(spec["direction"])]
        rel = [(int(r), int(c)) for r, c in np.argwhere(mask)]
        obj_h, obj_w = map(int, mask.shape)

        top_lo = max(0, -int(move_r))
        top_hi = min(height - obj_h, height - obj_h - int(move_r))
        left_lo = max(0, -int(move_c))
        left_hi = min(width - obj_w, width - obj_w - int(move_c))
        if top_lo > top_hi or left_lo > left_hi:
            return None

        candidates = [(int(top), int(left)) for top in range(top_lo, top_hi + 1) for left in range(left_lo, left_hi + 1)]
        random.shuffle(candidates)

        chosen = None
        for top, left in candidates:
            src = {(int(top + r), int(left + c)) for r, c in rel}
            dst = {(int(r + move_r), int(c + move_c)) for r, c in src}
            union = src | dst
            if _expand_8(union, height, width) & reserved:
                continue
            chosen = {
                "color": int(spec["color"]),
                "direction": str(spec["direction"]),
                "source": src,
            }
            reserved |= _expand_8(union, height, width)
            break

        if chosen is None:
            return None
        placed.append(chosen)

    return placed


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    bg = random.randrange(10)
    num_objects = random.randint(OBJECT_MIN, OBJECT_MAX)
    colors = random.sample([color for color in range(10) if color != bg], num_objects)
    directions = [random.choice(DIRECTIONS) for _ in range(num_objects)]

    specs = []
    for color, direction in zip(colors, directions):
        specs.append(
            {
                "color": int(color),
                "direction": str(direction),
                "mask": _sample_mask(str(direction)),
            }
        )

    placements = _place_objects(height, width, specs)
    if placements is None:
        return None

    grid = np.full((height, width), int(bg), dtype=int)
    intended = {}
    for item in placements:
        intended[int(item["color"])] = str(item["direction"])
        for r, c in item["source"]:
            grid[int(r), int(c)] = int(item["color"])

    if _get_bg(grid) != int(bg):
        return None

    comps = _find_components(grid, bg)
    if len(comps) != num_objects:
        return None
    for comp in comps:
        try:
            actual_direction = _direction_from_coords(comp["coords"])
        except AssertionError:
            return None
        if actual_direction != intended[int(comp["color"])]:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if len(_find_components(out, bg)) != num_objects:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
