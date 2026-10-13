import random
from collections import deque

import numpy as np


DIR_VECTORS: dict[str, tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
DELTA_TO_DIRECTION = {delta: name for name, delta in DIR_VECTORS.items()}
ORTHOGONAL_STEPS = tuple(DIR_VECTORS.values())


def _find_components(grid: np.ndarray, bg: int = 0) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            seen[r, c] = True
            queue = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in ORTHOGONAL_STEPS:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))

            components.append({"color": color, "coords": coords, "area": len(coords)})

    return components


def _direction_from_singleton_extreme(coords: list[tuple[int, int]]) -> str | None:
    cells = set(coords)
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    min_r = min(rows)
    max_r = max(rows)
    min_c = min(cols)
    max_c = max(cols)

    candidates: list[str] = []

    top = [(r, c) for r, c in coords if r == min_r]
    if len(top) == 1:
        r, c = top[0]
        if (r + 1, c) in cells:
            candidates.append("up")

    bottom = [(r, c) for r, c in coords if r == max_r]
    if len(bottom) == 1:
        r, c = bottom[0]
        if (r - 1, c) in cells:
            candidates.append("down")

    left = [(r, c) for r, c in coords if c == min_c]
    if len(left) == 1:
        r, c = left[0]
        if (r, c + 1) in cells:
            candidates.append("left")

    right = [(r, c) for r, c in coords if c == max_c]
    if len(right) == 1:
        r, c = right[0]
        if (r, c - 1) in cells:
            candidates.append("right")

    unique = sorted(set(candidates))
    if len(unique) == 1:
        return unique[0]
    return None


def _object_direction(coords: list[tuple[int, int]]) -> str:
    cells = set(coords)
    leaves: list[tuple[int, int]] = []
    leaf_neighbor: dict[tuple[int, int], tuple[int, int]] = {}

    for r, c in coords:
        neighbors = [
            (r + dr, c + dc)
            for dr, dc in ORTHOGONAL_STEPS
            if (r + dr, c + dc) in cells
        ]
        if len(neighbors) == 1:
            leaves.append((r, c))
            leaf_neighbor[(r, c)] = neighbors[0]

    if len(leaves) == 1:
        leaf = leaves[0]
        nbr = leaf_neighbor[leaf]
        delta = (int(leaf[0] - nbr[0]), int(leaf[1] - nbr[1]))
        if delta in DELTA_TO_DIRECTION:
            return DELTA_TO_DIRECTION[delta]

    fallback = _direction_from_singleton_extreme(coords)
    if fallback is not None:
        return fallback
    raise AssertionError("object must contain one identifiable protruding pixel")


def _translated_cells(
    coords: list[tuple[int, int]],
    dr: int,
    dc: int,
) -> list[tuple[int, int]]:
    return [(int(r + dr), int(c + dc)) for r, c in coords]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each monochrome object repeatedly in the direction indicated by its single protruding pixel.

    Input structure:
    - The background is black.
    - Each non-black color forms one small 4-connected object.
    - Every object is a compact block with exactly one one-cell protrusion attached orthogonally to one edge.
    - Objects are separated from one another and remain fully inside the grid.

    Rule:
    - For each object, locate the unique protruding leaf pixel; its outward side gives the stamping direction.
    - Count that object's total pixels and call the count N.
    - Keep the original object, then translate the entire object by N cells repeatedly along the protrusion direction.
    - Draw each translated copy in the same color until the next translated copy would leave the grid.

    Key insight:
    - Each object encodes two controls at once: shape gives the direction, and area gives the repeat step.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    for component in _find_components(g, bg=0):
        color = int(component["color"])
        coords = list(component["coords"])
        direction = _object_direction(coords)
        step = int(component["area"])
        dr, dc = DIR_VECTORS[direction]

        shift = step
        while True:
            translated = _translated_cells(coords, dr * shift, dc * shift)
            if any(not (0 <= r < h and 0 <= c < w) for r, c in translated):
                break
            for r, c in translated:
                out[r, c] = color
            shift += step

    return out


GRID_HEIGHT_RANGE = (18, 30)
GRID_WIDTH_RANGE = (18, 30)
OBJECT_COUNT_CHOICES = (3, 4)
CORE_SIDE_CHOICES = (2, 2, 3, 3, 4)
EXTRA_COPY_RANGE = (1, 3)
GEN_ATTEMPTS = 500
PLACEMENT_PADDING = 1


def _make_local_object(
    core_h: int,
    core_w: int,
    direction: str,
    offset: int,
) -> tuple[list[tuple[int, int]], int, int]:
    coords = {(r, c) for r in range(core_h) for c in range(core_w)}

    if direction == "up":
        coords = {(r + 1, c) for r, c in coords}
        coords.add((0, offset))
        bbox_h, bbox_w = core_h + 1, core_w
    elif direction == "down":
        coords.add((core_h, offset))
        bbox_h, bbox_w = core_h + 1, core_w
    elif direction == "left":
        coords = {(r, c + 1) for r, c in coords}
        coords.add((offset, 0))
        bbox_h, bbox_w = core_h, core_w + 1
    else:
        assert direction == "right"
        coords.add((offset, core_w))
        bbox_h, bbox_w = core_h, core_w + 1

    return sorted(coords), bbox_h, bbox_w


def _trajectory_cells(
    local_coords: list[tuple[int, int]],
    top: int,
    left: int,
    direction: str,
    step: int,
    h: int,
    w: int,
) -> tuple[set[tuple[int, int]], int]:
    dr, dc = DIR_VECTORS[direction]
    cells: set[tuple[int, int]] = set()
    extra_copies = 0
    shift = 0

    while True:
        translated = {
            (int(top + r + dr * shift), int(left + c + dc * shift))
            for r, c in local_coords
        }
        if any(not (0 <= r < h and 0 <= c < w) for r, c in translated):
            break
        cells |= translated
        if shift > 0:
            extra_copies += 1
        shift += step

    return cells, extra_copies


def _padded(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-PLACEMENT_PADDING, PLACEMENT_PADDING + 1):
            for dc in range(-PLACEMENT_PADDING, PLACEMENT_PADDING + 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _sample_object_specs(count: int, h: int, w: int) -> list[dict[str, object]] | None:
    colors = random.sample(range(1, 10), count)
    directions = random.sample(list(DIR_VECTORS.keys()), 4)[:count]
    random.shuffle(directions)

    specs: list[dict[str, object]] = []
    for color, direction in zip(colors, directions):
        for _ in range(50):
            core_h = random.choice(CORE_SIDE_CHOICES)
            core_w = random.choice(CORE_SIDE_CHOICES)
            axis_len = core_w if direction in ("up", "down") else core_h
            offset = random.randrange(axis_len)
            local_coords, bbox_h, bbox_w = _make_local_object(core_h, core_w, direction, offset)
            step = len(local_coords)
            axis_budget = h - bbox_h if direction in ("up", "down") else w - bbox_w
            max_copies = axis_budget // step
            if max_copies < EXTRA_COPY_RANGE[0]:
                continue
            extra_copies = random.randint(EXTRA_COPY_RANGE[0], min(EXTRA_COPY_RANGE[1], max_copies))
            specs.append(
                {
                    "color": color,
                    "direction": direction,
                    "coords": local_coords,
                    "bbox_h": bbox_h,
                    "bbox_w": bbox_w,
                    "step": step,
                    "extra_copies": extra_copies,
                }
            )
            break
        else:
            return None

    random.shuffle(specs)
    specs.sort(key=lambda spec: (int(spec["step"]) * (1 + int(spec["extra_copies"]))), reverse=True)
    return specs


def _place_specs(specs: list[dict[str, object]], h: int, w: int) -> np.ndarray | None:
    grid = np.zeros((h, w), dtype=int)
    forbidden: set[tuple[int, int]] = set()
    used_directions: set[str] = set()
    used_copy_counts: set[int] = set()

    for spec in specs:
        local_coords = list(spec["coords"])
        bbox_h = int(spec["bbox_h"])
        bbox_w = int(spec["bbox_w"])
        step = int(spec["step"])
        direction = str(spec["direction"])
        target_copies = int(spec["extra_copies"])
        color = int(spec["color"])

        candidates: list[tuple[int, int, set[tuple[int, int]]]] = []
        for top in range(h - bbox_h + 1):
            for left in range(w - bbox_w + 1):
                cells, extra_copies = _trajectory_cells(local_coords, top, left, direction, step, h, w)
                if extra_copies != target_copies:
                    continue
                if cells & forbidden:
                    continue
                candidates.append((top, left, cells))

        if not candidates:
            return None

        top, left, cells = random.choice(candidates)
        for r, c in _translated_cells(local_coords, top, left):
            grid[r, c] = color
        forbidden |= _padded(cells, h, w)
        used_directions.add(direction)
        used_copy_counts.add(target_copies)

    if len(used_directions) < min(3, len(specs)):
        return None
    if len(used_copy_counts) < 2:
        return None
    return grid


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    object_count = random.choice(OBJECT_COUNT_CHOICES)
    specs = _sample_object_specs(object_count, h, w)
    if specs is None:
        return None

    grid = _place_specs(specs, h, w)
    if grid is None:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape:
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    if len(_find_components(grid, bg=0)) != object_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
