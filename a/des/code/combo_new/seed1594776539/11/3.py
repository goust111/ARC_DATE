import random
from itertools import combinations

import numpy as np


BG = 0
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# The 3-cell arrow hole is a 2x2 L shape. We use a fixed orientation mapping:
# up    = top-right tip + bottom row pair
# right = bottom-right tip + left column pair
# down  = bottom-left tip + top row pair
# left  = top-left tip + right column pair
ARROW_SHAPES = {
    "up": {(0, 1), (1, 0), (1, 1)},
    "right": {(0, 0), (1, 0), (1, 1)},
    "down": {(0, 0), (0, 1), (1, 0)},
    "left": {(0, 0), (0, 1), (1, 1)},
}
DIR_TO_DELTA = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GRID_SIDE_RANGE = (20, 30)
OBJECT_COUNT_RANGE = (2, 4)
PATCH_SIDE_RANGE = (4, 7)
EXTRA_HOLE_OPTIONS = (0, 1, 2, 3)
PLACEMENT_PAD = 1
GEN_ATTEMPTS = 500
PATCH_ATTEMPTS = 120


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(points: list[tuple[int, int]]) -> set[tuple[int, int]]:
    r0, c0, _, _ = _bbox(points)
    return {(r - r0, c - c0) for r, c in points}


def _arrow_direction(points: list[tuple[int, int]]) -> str | None:
    if len(points) != 3:
        return None
    shape = _normalize(points)
    for name, pattern in ARROW_SHAPES.items():
        if shape == pattern:
            return name
    return None


def _inside_holes(patch: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = patch.shape
    holes = []
    for comp in _components(patch == BG):
        if all(0 < r < h - 1 and 0 < c < w - 1 for r, c in comp):
            holes.append(comp)
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate each colored rectangle according to its internal arrow hole and extra hole count.

    Input structure:
    - The background is Black.
    - Several non-overlapping monochrome rectangular regions appear on the canvas, each in a different non-black color.
    - Every rectangle contains black interior holes: exactly one 3-cell arrow-shaped hole and zero to three extra 1x1 black holes.

    Rule:
    - For each colored rectangle, inspect the black hole components inside its bounding box.
    - Identify the unique 3-cell arrow hole to determine one of four directions: up, down, left, or right.
    - Count the remaining single-cell holes; if there are N of them, translate the whole rectangle patch, including its holes, by N cells in the arrow direction.
    - Clear the rectangle's original location to Black before placing the translated patch. If N = 0, the rectangle stays in place.

    Key insight:
    One kind of hole encodes direction while the other holes encode distance, so the same black-hole motif simultaneously specifies how and how far each region should move.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    moves: list[tuple[int, int, np.ndarray, int, int]] = []

    for color in np.unique(g):
        color = int(color)
        if color == BG:
            continue
        for comp in _components(g == color):
            r0, c0, r1, c1 = _bbox(comp)
            patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
            if not np.isin(patch, [BG, color]).all():
                continue

            holes = _inside_holes(patch)
            if not holes:
                continue
            if any(len(hole) not in (1, 3) for hole in holes):
                continue

            arrow_candidates = [(idx, _arrow_direction(hole)) for idx, hole in enumerate(holes)]
            arrow_candidates = [(idx, direction) for idx, direction in arrow_candidates if direction is not None]
            if len(arrow_candidates) != 1:
                continue

            arrow_idx, direction = arrow_candidates[0]
            extra_count = sum(1 for idx, hole in enumerate(holes) if idx != arrow_idx and len(hole) == 1)
            dy, dx = DIR_TO_DELTA[direction]
            moves.append((r0, c0, patch, dy * extra_count, dx * extra_count))

    for r0, c0, patch, _dy, _dx in moves:
        h, w = patch.shape
        out[r0 : r0 + h, c0 : c0 + w] = BG

    for r0, c0, patch, dy, dx in moves:
        h, w = patch.shape
        nr, nc = r0 + dy, c0 + dx
        out[nr : nr + h, nc : nc + w] = patch

    return out


def _valid_extra_holes(
    height: int,
    width: int,
    arrow_cells: set[tuple[int, int]],
    extra_count: int,
) -> list[tuple[tuple[int, int], ...]]:
    if extra_count == 0:
        return [tuple()]

    interior = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
    blocked = set(arrow_cells)
    for r, c in tuple(arrow_cells):
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 < nr < height - 1 and 0 < nc < width - 1:
                blocked.add((nr, nc))
    candidates = [cell for cell in interior if cell not in blocked]

    valid: list[tuple[tuple[int, int], ...]] = []
    for combo in combinations(candidates, extra_count):
        ok = True
        for i, (r0, c0) in enumerate(combo):
            for r1, c1 in combo[i + 1 :]:
                if abs(r0 - r1) + abs(c0 - c1) == 1:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            valid.append(tuple(combo))
    return valid


def _make_patch(color: int, direction: str, extra_count: int, rng: random.Random) -> np.ndarray | None:
    for _ in range(PATCH_ATTEMPTS):
        height = rng.randint(*PATCH_SIDE_RANGE)
        width = rng.randint(*PATCH_SIDE_RANGE)
        if height < 4 or width < 4:
            continue

        anchors = [
            (r, c)
            for r in range(1, height - 2)
            for c in range(1, width - 2)
        ]
        rng.shuffle(anchors)
        for ar, ac in anchors:
            arrow_cells = {(ar + dr, ac + dc) for dr, dc in ARROW_SHAPES[direction]}
            valid_extras = _valid_extra_holes(height, width, arrow_cells, extra_count)
            if not valid_extras:
                continue

            patch = np.full((height, width), color, dtype=int)
            for r, c in arrow_cells:
                patch[r, c] = BG
            for r, c in rng.choice(valid_extras):
                patch[r, c] = BG
            return patch
    return None


def _rect_is_free(
    reserved: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    pad: int,
) -> bool:
    h, w = reserved.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    return not reserved[r0:r1, c0:c1].any()


def _mark_rect(
    reserved: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    pad: int,
) -> None:
    h, w = reserved.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    reserved[r0:r1, c0:c1] = True


def _place_objects(
    objects: list[dict[str, object]],
    grid_h: int,
    grid_w: int,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    reserved = np.zeros((grid_h, grid_w), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(objects)
    order = sorted(
        range(len(objects)),
        key=lambda idx: int(np.prod(np.array(objects[idx]["patch"]).shape)),
        reverse=True,
    )

    for idx in order:
        patch = np.array(objects[idx]["patch"], dtype=int)
        dy, dx = objects[idx]["delta"]
        height, width = patch.shape

        row_lo = max(0, -dy)
        row_hi = min(grid_h - height, grid_h - height - dy)
        col_lo = max(0, -dx)
        col_hi = min(grid_w - width, grid_w - width - dx)
        if row_lo > row_hi or col_lo > col_hi:
            return None

        candidates = [(r, c) for r in range(row_lo, row_hi + 1) for c in range(col_lo, col_hi + 1)]
        rng.shuffle(candidates)
        chosen = None
        for top, left in candidates:
            if not _rect_is_free(reserved, top, left, height, width, PLACEMENT_PAD):
                continue
            if not _rect_is_free(reserved, top + dy, left + dx, height, width, PLACEMENT_PAD):
                continue
            chosen = (top, left)
            break
        if chosen is None:
            return None

        positions[idx] = chosen
        _mark_rect(reserved, chosen[0], chosen[1], height, width, PLACEMENT_PAD)
        _mark_rect(reserved, chosen[0] + dy, chosen[1] + dx, height, width, PLACEMENT_PAD)

    return [(int(r), int(c)) for r, c in positions if r is not None and c is not None]


def _build_grids(
    objects: list[dict[str, object]],
    positions: list[tuple[int, int]],
    grid_h: int,
    grid_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((grid_h, grid_w), BG, dtype=int)
    out = np.full((grid_h, grid_w), BG, dtype=int)
    for obj, (top, left) in zip(objects, positions):
        patch = np.array(obj["patch"], dtype=int)
        dy, dx = obj["delta"]
        height, width = patch.shape
        inp[top : top + height, left : left + width] = patch
        out[top + dy : top + dy + height, left + dx : left + dx + width] = patch
    return inp, out


def _generate_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    object_count = rng.randint(*OBJECT_COUNT_RANGE)
    colors = rng.sample(COLOR_POOL, object_count)
    directions = [rng.choice(list(ARROW_SHAPES)) for _ in range(object_count)]
    extra_counts = [rng.choice(EXTRA_HOLE_OPTIONS) for _ in range(object_count)]
    if all(count == 0 for count in extra_counts):
        extra_counts[rng.randrange(object_count)] = rng.choice((1, 2, 3))

    objects: list[dict[str, object]] = []
    for color, direction, extra_count in zip(colors, directions, extra_counts):
        patch = _make_patch(color, direction, extra_count, rng)
        if patch is None:
            return None
        dy, dx = DIR_TO_DELTA[direction]
        objects.append(
            {
                "patch": patch,
                "delta": (dy * extra_count, dx * extra_count),
                "direction": direction,
                "extra_count": extra_count,
            }
        )

    grid_h = rng.randint(*GRID_SIDE_RANGE)
    grid_w = rng.randint(*GRID_SIDE_RANGE)
    positions = _place_objects(objects, grid_h, grid_w, rng)
    if positions is None or len(positions) != len(objects):
        return None

    inp, out = _build_grids(objects, positions, grid_h, grid_w)
    if np.array_equal(inp, out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        result = _generate_case(rng)
        if result is None:
            continue
        inp, expected = result
        if np.array_equal(main(inp), expected):
            return inp
    raise RuntimeError("generate_input: failed to produce a valid sample")
