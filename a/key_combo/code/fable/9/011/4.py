from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
BODY_PALETTE = [3, 4, 5, 6, 7, 8, 9]

OUTPUT_SIDE_RANGE = (4, 7)
OBJECT_COLLISION_SIZE = (2, 3)
MAX_GENERATION_ATTEMPTS = 400
GRID_SIDE_MIN = 20
GRID_SIDE_MAX = 30
MIN_BOX_H = 3
MAX_BOX_H = 6
MIN_BOX_W = 3
MAX_BOX_W = 7
MAX_BOX_SPARE = 6
PLACEMENT_GAP = 1


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mask_connected(mask: np.ndarray) -> bool:
    comps = _component_coords(mask)
    return len(comps) == 1


def _spans_box(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _extract_objects(grid: np.ndarray) -> list[dict]:
    comps = _component_coords(grid != BG)
    objects: list[dict] = []
    for coords in comps:
        colors = set(map(int, np.unique(grid[coords[:, 0], coords[:, 1]])))
        assert RED in colors and BLUE in colors, "each object must contain both marker colors"
        body_colors = sorted(colors - {RED, BLUE})
        assert len(body_colors) == 1, "each object must have exactly one body color"
        body = body_colors[0]
        red_count = int(np.count_nonzero(grid[coords[:, 0], coords[:, 1]] == RED))
        blue_count = int(np.count_nonzero(grid[coords[:, 0], coords[:, 1]] == BLUE))
        assert red_count > 0 and blue_count > 0, "marker counts must form 1-based coordinates"
        objects.append(
            {
                "coords": coords,
                "bbox": _bbox(coords),
                "body": int(body),
                "red": red_count,
                "blue": blue_count,
            }
        )
    assert objects, "expected at least one foreground object"
    return objects


def _expected_output_from_specs(specs: list[dict], side: int) -> np.ndarray:
    out = np.full((side, side), BG, dtype=int)
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for spec in specs:
        buckets[(int(spec["red"]), int(spec["blue"]))].append(int(spec["body"]))
    for (red_count, blue_count), bodies in buckets.items():
        if len(bodies) == 1:
            out[red_count - 1, blue_count - 1] = bodies[0]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count red and blue markers inside each object, use the counts as output coordinates, and erase coordinate collisions.

    Input structure:
    - A black background surrounds several separated 4-connected foreground objects.
    - Each object contains exactly one body color plus red and blue single-pixel markers scattered inside it.
    - Object shapes, sizes, and positions vary independently of the output.

    Rule:
    - For each whole foreground object, count its red markers and blue markers.
    - Create a square black output whose side length is the maximum marker count seen in either color over all objects.
    - Paint the object's body color at output row `red_count` and column `blue_count`, using 1-based coordinates.
    - If two or more objects map to the same output cell, that cell is reset to black.

    Key insight:
    - The geometry of each object is irrelevant; only the two marker counts matter, and shared coordinates annihilate instead of stacking.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    objects = _extract_objects(g)
    side = max(max(obj["red"], obj["blue"]) for obj in objects)
    assert side > 0
    return _expected_output_from_specs(objects, side)


def _pick_distinct(values: list[int], k: int) -> list[int]:
    return random.sample(values, k)


def _sample_specs() -> tuple[list[dict], int]:
    side = random.randint(*OUTPUT_SIDE_RANGE)
    pool = list(range(1, side))
    branch = random.choice(["none", "none", "hide_row", "hide_col", "hide_both"])
    collision_size = random.randint(*OBJECT_COLLISION_SIZE)

    if branch == "none":
        c1, c2 = _pick_distinct(pool, 2)
        r1, r2 = _pick_distinct(pool, 2)
        base = [(side, c1), (side, c2), (r1, side), (r2, side)]
        collision = random.choice([(r, c) for r in pool for c in pool if (r, c) not in base])
        coords = base + [collision] * collision_size
    elif branch == "hide_row":
        row_dup, c1, c2 = _pick_distinct(pool, 3)
        r1, r2 = _pick_distinct([v for v in pool if v != row_dup], 2)
        base = [(row_dup, c1), (row_dup, c2), (r1, side), (r2, side)]
        collision = random.choice([(side, c) for c in pool if (side, c) not in base])
        coords = base + [collision] * collision_size
    elif branch == "hide_col":
        col_dup, r1, r2 = _pick_distinct(pool, 3)
        c1, c2 = _pick_distinct([v for v in pool if v != col_dup], 2)
        base = [(r1, col_dup), (r2, col_dup), (side, c1), (side, c2)]
        collision = random.choice([(r, side) for r in pool if (r, side) not in base])
        coords = base + [collision] * collision_size
    else:
        row_dup, c1, c2 = _pick_distinct(pool, 3)
        col_dup, r1, r2 = _pick_distinct(pool, 3)
        coords = [(row_dup, c1), (row_dup, c2), (r1, col_dup), (r2, col_dup)] + [(side, side)] * collision_size

    used = set(coords)
    extra_candidates = [(r, c) for r in range(1, side + 1) for c in range(1, side + 1) if (r, c) not in used]
    if collision_size == 2 and extra_candidates and random.random() < 0.65:
        coords.append(random.choice(extra_candidates))

    assert 5 <= len(coords) <= 7
    colors = random.sample(BODY_PALETTE, len(coords))
    specs = [{"red": r, "blue": c, "body": body} for (r, c), body in zip(coords, colors)]
    return specs, side


def _choose_box_and_area(red_count: int, blue_count: int) -> tuple[int, int, int]:
    marker_total = red_count + blue_count
    body_cells = random.randint(max(4, max(red_count, blue_count)), max(6, max(red_count, blue_count) + 3))
    target_area = marker_total + body_cells

    options = [
        (h, w)
        for h in range(MIN_BOX_H, MAX_BOX_H + 1)
        for w in range(MIN_BOX_W, MAX_BOX_W + 1)
        if h * w >= target_area and h * w - target_area <= MAX_BOX_SPARE
    ]
    assert options, "expected a feasible bbox for the sampled marker counts"
    return random.choice(options) + (target_area,)


def _random_object_mask(h: int, w: int, area: int) -> np.ndarray | None:
    full_area = h * w
    remove_count = full_area - area
    for _ in range(120):
        mask = np.ones((h, w), dtype=bool)
        cells = [(r, c) for r in range(h) for c in range(w)]
        random.shuffle(cells)
        ok = True
        for _step in range(remove_count):
            removed = False
            random.shuffle(cells)
            for r, c in cells:
                if not mask[r, c]:
                    continue
                trial = mask.copy()
                trial[r, c] = False
                if _spans_box(trial) and _mask_connected(trial):
                    mask = trial
                    removed = True
                    break
            if not removed:
                ok = False
                break
        if ok and _mask_connected(mask) and _spans_box(mask) and int(mask.sum()) == area:
            return mask
    return None


def _render_object(spec: dict) -> np.ndarray | None:
    h, w, area = _choose_box_and_area(int(spec["red"]), int(spec["blue"]))
    mask = _random_object_mask(h, w, area)
    if mask is None:
        return None

    patch = np.full((h, w), BG, dtype=int)
    coords = [tuple(cell) for cell in np.argwhere(mask)]
    random.shuffle(coords)
    marker_total = int(spec["red"]) + int(spec["blue"])
    marker_cells = coords[:marker_total]
    body_cells = coords[marker_total:]
    if len(body_cells) < 4:
        return None

    patch[mask] = int(spec["body"])
    for r, c in marker_cells[: int(spec["red"])]:
        patch[r, c] = RED
    for r, c in marker_cells[int(spec["red"]) :]:
        patch[r, c] = BLUE
    return patch


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + PLACEMENT_GAP < br0
        or br1 + PLACEMENT_GAP < ar0
        or ac1 + PLACEMENT_GAP < bc0
        or bc1 + PLACEMENT_GAP < ac0
    )


def _place_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    total_box_area = sum(int(p.shape[0] * p.shape[1]) for p in patches)
    min_side = max(GRID_SIDE_MIN, int(np.ceil(np.sqrt(total_box_area * 2.6))))
    if min_side > GRID_SIDE_MAX:
        return None

    rows = random.randint(min_side, GRID_SIDE_MAX)
    cols = random.randint(min_side, GRID_SIDE_MAX)
    grid = np.full((rows, cols), BG, dtype=int)
    placed: list[tuple[int, int, int, int]] = []

    order = sorted(range(len(patches)), key=lambda idx: int(patches[idx].shape[0] * patches[idx].shape[1]), reverse=True)
    for idx in order:
        patch = patches[idx]
        h, w = patch.shape
        candidates = [(r, c) for r in range(rows - h + 1) for c in range(cols - w + 1)]
        random.shuffle(candidates)
        chosen = None
        for r0, c0 in candidates:
            box = (r0, c0, r0 + h - 1, c0 + w - 1)
            if any(_boxes_too_close(box, other) for other in placed):
                continue
            chosen = box
            break
        if chosen is None:
            return None
        r0, c0, r1, c1 = chosen
        grid[r0 : r1 + 1, c0 : c1 + 1][patch != BG] = patch[patch != BG]
        placed.append(chosen)
    return grid


def _validate_specs(specs: list[dict], side: int, expected: np.ndarray) -> bool:
    coords = [(int(spec["red"]), int(spec["blue"])) for spec in specs]
    counts = Counter(coords)
    visible = [coord for coord in coords if counts[coord] == 1]
    collisions = [coord for coord, count in counts.items() if count >= 2]

    if not collisions or len(visible) < 2:
        return False
    if max(r for r, _ in coords) != side or max(c for _, c in coords) != side:
        return False

    by_row: dict[int, set[int]] = defaultdict(set)
    by_col: dict[int, set[int]] = defaultdict(set)
    for r, c in coords:
        by_row[r].add(c)
        by_col[c].add(r)
    if not any(len(cols) >= 2 for cols in by_row.values()):
        return False
    if not any(len(rows) >= 2 for rows in by_col.values()):
        return False
    if np.count_nonzero(expected) < 2:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    specs, side = _sample_specs()
    expected = _expected_output_from_specs(specs, side)
    if not _validate_specs(specs, side, expected):
        return None

    patches = []
    for spec in specs:
        patch = _render_object(spec)
        if patch is None:
            return None
        patches.append(patch)

    grid = _place_patches(patches)
    if grid is None:
        return None

    try:
        parsed = _extract_objects(grid)
    except AssertionError:
        return None

    if len(parsed) != len(specs):
        return None

    parsed_by_body = {obj["body"]: (obj["red"], obj["blue"]) for obj in parsed}
    spec_by_body = {int(spec["body"]): (int(spec["red"]), int(spec["blue"])) for spec in specs}
    if parsed_by_body != spec_by_body:
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    visible_rows = [r for r, c in spec_by_body.values() if expected[r - 1, c - 1] != BG]
    visible_cols = [c for r, c in spec_by_body.values() if expected[r - 1, c - 1] != BG]
    if not visible_rows or not visible_cols:
        return None
    if max(actual.shape) > 30:
        return None
    if max(grid.shape) > 30:
        return None
    if max(visible_rows) == side and max(visible_cols) == side and np.count_nonzero(actual[-1, :]) == 0 and np.count_nonzero(actual[:, -1]) == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
