import random
from collections import deque

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))

GRID_SIDE_RANGE = (16, 30)
OBJECT_COUNT_RANGE = (1, 4)
HOLE_SIZE_RANGE = (1, 4)
OPPOSITE_GAP_RANGE = (2, 4)
ORTHOGONAL_GAP_RANGE = (2, 4)
SCENE_ATTEMPTS = 300
OBJECT_ATTEMPTS = 80

DIRECTIONS = ("up", "down", "left", "right")


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components_of_color(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(coords)
    return components


def _find_objects(grid: np.ndarray) -> list[dict[str, int]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict[str, int]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            patch_h, patch_w = patch.shape

            if np.any(patch[[0, -1], :] == BG) or np.any(patch[:, [0, -1]] == BG):
                raise ValueError("object border must be solid")

            hole_mask = patch == BG
            hole_comps = _components_of_color(hole_mask)
            if len(hole_comps) != 1:
                raise ValueError("expected exactly one rectangular hole")
            hole = hole_comps[0]
            hr_values = [rr for rr, _ in hole]
            hc_values = [cc for _, cc in hole]
            hr0, hr1 = min(hr_values), max(hr_values)
            hc0, hc1 = min(hc_values), max(hc_values)

            hole_h = hr1 - hr0 + 1
            hole_w = hc1 - hc0 + 1
            if len(hole) != hole_h * hole_w:
                raise ValueError("hole must be rectangular")
            if not (0 < hr0 <= hr1 < patch_h - 1 and 0 < hc0 <= hc1 < patch_w - 1):
                raise ValueError("hole must stay strictly inside the object")

            rebuilt = np.full((patch_h, patch_w), color, dtype=int)
            rebuilt[hr0 : hr1 + 1, hc0 : hc1 + 1] = BG
            if not np.array_equal(rebuilt, patch):
                raise ValueError("object must be a rectangle with one rectangular hole")

            top_gap = hr0
            bottom_gap = patch_h - 1 - hr1
            left_gap = hc0
            right_gap = patch_w - 1 - hc1
            gap_map = {
                "up": top_gap,
                "down": bottom_gap,
                "left": left_gap,
                "right": right_gap,
            }
            min_gap = min(gap_map.values())
            directions = [name for name, gap in gap_map.items() if gap == min_gap]
            if len(directions) != 1:
                raise ValueError("hole direction must be unique")
            direction = directions[0]

            objects.append(
                {
                    "color": color,
                    "r0": r0,
                    "c0": c0,
                    "r1": r1,
                    "c1": c1,
                    "hr0": hr0,
                    "hc0": hc0,
                    "hr1": hr1,
                    "hc1": hc1,
                    "hole_h": hole_h,
                    "hole_w": hole_w,
                    "direction": direction,
                }
            )

    objects.sort(key=lambda obj: (obj["r0"], obj["c0"], obj["color"]))
    return objects


def _projection_cells(obj: dict[str, int], grid_shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = grid_shape
    color_r0 = obj["r0"]
    color_c0 = obj["c0"]
    color_r1 = obj["r1"]
    color_c1 = obj["c1"]
    hr0 = color_r0 + obj["hr0"]
    hr1 = color_r0 + obj["hr1"]
    hc0 = color_c0 + obj["hc0"]
    hc1 = color_c0 + obj["hc1"]
    hole_h = obj["hole_h"]
    hole_w = obj["hole_w"]
    direction = obj["direction"]

    cells: list[tuple[int, int]] = []
    if direction == "up":
        row_start = max(0, color_r0 - hole_h)
        for rr in range(row_start, color_r0):
            for cc in range(hc0, hc1 + 1):
                cells.append((rr, cc))
    elif direction == "down":
        row_stop = min(h, color_r1 + 1 + hole_h)
        for rr in range(color_r1 + 1, row_stop):
            for cc in range(hc0, hc1 + 1):
                cells.append((rr, cc))
    elif direction == "left":
        col_start = max(0, color_c0 - hole_w)
        for rr in range(hr0, hr1 + 1):
            for cc in range(col_start, color_c0):
                cells.append((rr, cc))
    else:
        col_stop = min(w, color_c1 + 1 + hole_w)
        for rr in range(hr0, hr1 + 1):
            for cc in range(color_c1 + 1, col_stop):
                cells.append((rr, cc))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each hollow rectangle outward using its off-center hole as a same-shape stamp.

    Input structure:
    - The background is black.
    - Each foreground object is a monochrome solid rectangle with exactly one rectangular black hole inside it.
    - The hole never touches the outer boundary, but one side of the hole sits closest to one outer side of the object.
    - Different objects are spatially separate and may have different colors, sizes, hole sizes, and hole offsets.

    Rule:
    - For each object, find which outer side is closest to the hole; that side determines the projection direction.
    - Copy the hole's rectangular footprint straight outward from the corresponding outer boundary of the object.
    - The projection keeps the object's color, and its extent along the projection direction equals the hole size along that same direction.
    - Clip projections at the grid edge and never overwrite already non-black pixels.
    - Keep the original object and its hole unchanged.

    Key insight:
    - The hole encodes two linked pieces of information at once: its eccentric position picks the direction, and its own dimensions set the projected rectangle's footprint.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for obj in _find_objects(g):
        for rr, cc in _projection_cells(obj, g.shape):
            if int(out[rr, cc]) == BG:
                out[rr, cc] = obj["color"]
    return out


def _sample_object_spec(color: int, direction: str) -> dict[str, int]:
    hole_h = random.randint(*HOLE_SIZE_RANGE)
    hole_w = random.randint(*HOLE_SIZE_RANGE)

    if direction == "up":
        top_gap = 1
        bottom_gap = random.randint(*OPPOSITE_GAP_RANGE)
        left_gap = random.randint(*ORTHOGONAL_GAP_RANGE)
        right_gap = random.randint(*ORTHOGONAL_GAP_RANGE)
    elif direction == "down":
        top_gap = random.randint(*OPPOSITE_GAP_RANGE)
        bottom_gap = 1
        left_gap = random.randint(*ORTHOGONAL_GAP_RANGE)
        right_gap = random.randint(*ORTHOGONAL_GAP_RANGE)
    elif direction == "left":
        top_gap = random.randint(*ORTHOGONAL_GAP_RANGE)
        bottom_gap = random.randint(*ORTHOGONAL_GAP_RANGE)
        left_gap = 1
        right_gap = random.randint(*OPPOSITE_GAP_RANGE)
    else:
        top_gap = random.randint(*ORTHOGONAL_GAP_RANGE)
        bottom_gap = random.randint(*ORTHOGONAL_GAP_RANGE)
        left_gap = random.randint(*OPPOSITE_GAP_RANGE)
        right_gap = 1

    outer_h = top_gap + hole_h + bottom_gap
    outer_w = left_gap + hole_w + right_gap
    return {
        "color": color,
        "direction": direction,
        "outer_h": outer_h,
        "outer_w": outer_w,
        "hole_h": hole_h,
        "hole_w": hole_w,
        "hole_r0": top_gap,
        "hole_c0": left_gap,
        "hole_r1": top_gap + hole_h - 1,
        "hole_c1": left_gap + hole_w - 1,
    }


def _projection_cells_from_spec(
    spec: dict[str, int], top: int, left: int, grid_shape: tuple[int, int]
) -> tuple[list[tuple[int, int]], int]:
    h, w = grid_shape
    bottom = top + spec["outer_h"] - 1
    right = left + spec["outer_w"] - 1
    hr0 = top + spec["hole_r0"]
    hr1 = top + spec["hole_r1"]
    hc0 = left + spec["hole_c0"]
    hc1 = left + spec["hole_c1"]

    cells: list[tuple[int, int]] = []
    if spec["direction"] == "up":
        row_start = max(0, top - spec["hole_h"])
        for rr in range(row_start, top):
            for cc in range(hc0, hc1 + 1):
                cells.append((rr, cc))
        return cells, top - row_start
    if spec["direction"] == "down":
        row_stop = min(h, bottom + 1 + spec["hole_h"])
        for rr in range(bottom + 1, row_stop):
            for cc in range(hc0, hc1 + 1):
                cells.append((rr, cc))
        return cells, row_stop - (bottom + 1)
    if spec["direction"] == "left":
        col_start = max(0, left - spec["hole_w"])
        for rr in range(hr0, hr1 + 1):
            for cc in range(col_start, left):
                cells.append((rr, cc))
        return cells, left - col_start

    col_stop = min(w, right + 1 + spec["hole_w"])
    for rr in range(hr0, hr1 + 1):
        for cc in range(right + 1, col_stop):
            cells.append((rr, cc))
    return cells, col_stop - (right + 1)


def _candidate_positions(
    spec: dict[str, int],
    grid_shape: tuple[int, int],
    input_blocked: np.ndarray,
    output_blocked: np.ndarray,
    require_clipped: bool,
) -> list[tuple[int, int, list[tuple[int, int]]]]:
    h, w = grid_shape
    outer_h = spec["outer_h"]
    outer_w = spec["outer_w"]
    full_len = spec["hole_h"] if spec["direction"] in ("up", "down") else spec["hole_w"]
    candidates: list[tuple[int, int, list[tuple[int, int]]]] = []

    for top in range(h - outer_h + 1):
        for left in range(w - outer_w + 1):
            if input_blocked[top : top + outer_h, left : left + outer_w].any():
                continue
            if output_blocked[top : top + outer_h, left : left + outer_w].any():
                continue

            cells, actual_len = _projection_cells_from_spec(spec, top, left, grid_shape)
            if actual_len == 0:
                continue
            clipped = actual_len < full_len
            if clipped != require_clipped:
                continue
            if any(output_blocked[rr, cc] for rr, cc in cells):
                continue

            candidates.append((top, left, cells))
    return candidates


def _paint_object(grid: np.ndarray, spec: dict[str, int], top: int, left: int) -> None:
    bottom = top + spec["outer_h"]
    right = left + spec["outer_w"]
    grid[top:bottom, left:right] = spec["color"]
    grid[
        top + spec["hole_r0"] : top + spec["hole_r1"] + 1,
        left + spec["hole_c0"] : left + spec["hole_c1"] + 1,
    ] = BG


def _reserve_input_bbox(mask: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(mask.shape[0], top + height + 1)
    c1 = min(mask.shape[1], left + width + 1)
    mask[r0:r1, c0:c1] = True


def _place_one_object(
    grid: np.ndarray,
    input_blocked: np.ndarray,
    output_blocked: np.ndarray,
    color: int,
    direction: str,
    require_clipped: bool,
) -> bool:
    for _ in range(OBJECT_ATTEMPTS):
        spec = _sample_object_spec(color, direction)
        candidates = _candidate_positions(
            spec, grid.shape, input_blocked, output_blocked, require_clipped
        )
        if not candidates:
            continue
        top, left, projection = random.choice(candidates)
        _paint_object(grid, spec, top, left)
        _reserve_input_bbox(input_blocked, top, left, spec["outer_h"], spec["outer_w"])
        output_blocked[top : top + spec["outer_h"], left : left + spec["outer_w"]] = True
        for rr, cc in projection:
            output_blocked[rr, cc] = True
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    n_objects = random.randint(*OBJECT_COUNT_RANGE)
    colors = random.sample(COLORS, n_objects)
    directions = random.sample(DIRECTIONS, n_objects)

    require_any_clipped = n_objects > 0 and random.random() < 0.5
    clipped_count = random.randint(1, n_objects) if require_any_clipped else 0
    clipped_indices = set(random.sample(range(n_objects), clipped_count))

    grid = np.zeros((h, w), dtype=int)
    input_blocked = np.zeros((h, w), dtype=bool)
    output_blocked = np.zeros((h, w), dtype=bool)

    order = list(range(n_objects))
    random.shuffle(order)
    for idx in order:
        if not _place_one_object(
            grid,
            input_blocked,
            output_blocked,
            colors[idx],
            directions[idx],
            idx in clipped_indices,
        ):
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(SCENE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
