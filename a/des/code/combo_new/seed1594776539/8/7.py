import numpy as np
from collections import deque


BG = 0
GREY = 5
ORANGE = 7
MAROON = 9
OBJECT_COLORS = np.array([1, 2, 3, 4, 6, 8], dtype=int)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GENERATION_ATTEMPTS = 400


def _cc4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _normalize(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    cells = sorted(set(coords))
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _transform_shape(
    coords: tuple[tuple[int, int], ...],
    rotations: int,
    flip: bool,
) -> tuple[tuple[int, int], ...]:
    cells = list(coords)
    for _ in range(rotations % 4):
        cells = [(c, -r) for r, c in cells]
    if flip:
        cells = [(r, -c) for r, c in cells]
    return _normalize(cells)


def _shape_dims(coords: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in coords) + 1, max(c for _, c in coords) + 1


def _sample_polyomino(rng: np.random.Generator) -> tuple[tuple[int, int], ...]:
    target = int(rng.integers(4, 8))
    cells: set[tuple[int, int]] = {(0, 0)}
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for dr, dc in DIR4:
                cand = (r + dr, c + dc)
                if cand in cells:
                    continue
                trial = cells | {cand}
                rs = [rr for rr, _ in trial]
                cs = [cc for _, cc in trial]
                if max(rs) - min(rs) <= 4 and max(cs) - min(cs) <= 4:
                    frontier.append(cand)
        if not frontier:
            break
        choice = frontier[int(rng.integers(len(frontier)))]
        cells.add(choice)
    return _normalize(cells)


def _sample_shape(rng: np.random.Generator) -> tuple[tuple[int, int], ...]:
    family = rng.choice(
        ["rectangle", "bar", "ell", "tee", "stair", "plus", "poly"],
        p=[0.20, 0.12, 0.18, 0.14, 0.14, 0.10, 0.12],
    )
    if family == "rectangle":
        h = int(rng.integers(1, 4))
        w = int(rng.integers(2, 5))
        if h == 1 and w == 1:
            w = 2
        base = tuple((r, c) for r in range(h) for c in range(w))
    elif family == "bar":
        length = int(rng.integers(2, 6))
        base = tuple((0, c) for c in range(length))
    elif family == "ell":
        h = int(rng.integers(2, 5))
        w = int(rng.integers(2, 5))
        base = tuple(sorted({(r, 0) for r in range(h)} | {(h - 1, c) for c in range(w)}))
    elif family == "tee":
        cap_w = int(rng.choice([3, 5]))
        stem_h = int(rng.integers(2, 5))
        center = cap_w // 2
        base = tuple(sorted({(0, c) for c in range(cap_w)} | {(r, center) for r in range(stem_h)}))
    elif family == "stair":
        steps = int(rng.integers(2, 5))
        base = tuple(sorted((r, r + dc) for r in range(steps) for dc in (0, 1)))
    elif family == "plus":
        arm = int(rng.integers(1, 3))
        center = arm
        cells = {(center, center)}
        for d in range(1, arm + 1):
            cells.update(
                {
                    (center - d, center),
                    (center + d, center),
                    (center, center - d),
                    (center, center + d),
                }
            )
        base = tuple(sorted(cells))
    else:
        base = _sample_polyomino(rng)
    return _transform_shape(base, int(rng.integers(4)), bool(rng.integers(2)))


def _mark_buffer(mask: np.ndarray, cells: list[tuple[int, int]], pad: int) -> None:
    h, w = mask.shape
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    mask[rr, cc] = True


def _project_cells(
    cells: list[tuple[int, int]],
    gap: int,
    direction: int,
) -> list[tuple[int, int]]:
    return [(r + gap, c + direction * gap) for r, c in cells]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw maroon ground shadows for floating objects using the light corner and each object's height above the ground.

    Input structure:
    - The background is black.
    - The entire bottom row is a grey ground line.
    - One orange light-source pixel sits in the upper-left or upper-right corner.
    - Above the ground are several separated monochromatic 4-connected objects with varying gaps above the ground.

    Rule:
    - Read the light position to determine shadow direction: upper-left light means shadows shift right, upper-right light means shadows shift left.
    - For each floating object, measure the number of empty rows between its lowest pixel and the grey ground line.
    - Copy that object's exact shape downward until its lowest row sits directly above the ground, and shift it horizontally by the same number of cells as the measured gap.
    - Paint only the resulting shadow cells in maroon, and never overwrite any pre-existing non-black pixel.

    Key insight:
    - The vertical gap to the ground is reused as the shadow's horizontal offset, while the light corner chooses whether that offset points left or right.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    if g.ndim != 2 or g.size == 0:
        return out

    h, w = g.shape
    lights = np.argwhere(g == ORANGE)
    if lights.shape[0] != 1:
        return out
    _, light_c = map(int, lights[0])
    direction = 1 if light_c < w // 2 else -1

    mask = (g != BG) & (g != GREY) & (g != ORANGE)
    for comp in _cc4(mask):
        bottom = max(r for r, _ in comp)
        gap = h - bottom - 2
        if gap < 1:
            continue
        for rr, cc in _project_cells(comp, gap, direction):
            if 0 <= rr < h and 0 <= cc < w and g[rr, cc] == BG:
                out[rr, cc] = MAROON

    return out


def _placeable_columns(width: int, shape_w: int, gap: int, direction: int) -> list[int]:
    if direction > 0:
        start, stop = 0, width - shape_w - gap
    else:
        start, stop = gap, width - shape_w
    if stop < start:
        return []
    return list(range(start, stop + 1))


def _build_object_cells(
    shape: tuple[tuple[int, int], ...],
    top: int,
    left: int,
) -> list[tuple[int, int]]:
    return [(top + r, left + c) for r, c in shape]


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(12, 25))
    width = int(rng.integers(12, 31))
    direction = 1 if bool(rng.integers(2)) else -1
    light = (0, 0) if direction > 0 else (0, width - 1)

    max_objects = min(5, len(OBJECT_COLORS), max(2, (width - 2) // 4))
    if max_objects < 2:
        return None
    n_objects = int(rng.integers(2, max_objects + 1))
    colors = [int(c) for c in rng.choice(OBJECT_COLORS, size=n_objects, replace=False)]

    specs = []
    for color in colors:
        for _ in range(80):
            shape = _sample_shape(rng)
            shape_h, shape_w = _shape_dims(shape)
            max_gap = min(8, height - shape_h - 2)
            if max_gap < 1:
                continue
            gap = int(rng.integers(1, max_gap + 1))
            if not _placeable_columns(width, shape_w, gap, direction):
                continue
            specs.append(
                {
                    "color": color,
                    "shape": shape,
                    "shape_h": shape_h,
                    "shape_w": shape_w,
                    "gap": gap,
                }
            )
            break
        else:
            return None

    if len({spec["gap"] for spec in specs}) < 2:
        return None

    specs.sort(key=lambda spec: (-spec["gap"], -(spec["shape_w"] + spec["shape_h"])))

    grid = np.full((height, width), BG, dtype=int)
    grid[-1, :] = GREY
    grid[light] = ORANGE

    actual = np.zeros((height, width), dtype=bool)
    actual[-1, :] = True
    actual[light] = True
    object_buffer = np.zeros((height, width), dtype=bool)
    object_buffer[-1, :] = True
    object_buffer[light] = True
    shadow_buffer = np.zeros((height, width), dtype=bool)
    shadow_cells_all: set[tuple[int, int]] = set()

    for spec in specs:
        gap = int(spec["gap"])
        shape = spec["shape"]
        top = height - gap - int(spec["shape_h"]) - 1
        if top < 1:
            return None

        candidates = _placeable_columns(width, int(spec["shape_w"]), gap, direction)
        rng.shuffle(candidates)
        placed = False

        for left in candidates:
            obj_cells = _build_object_cells(shape, top, left)
            proj_cells = _project_cells(obj_cells, gap, direction)
            if set(obj_cells) & set(proj_cells):
                continue
            if any(actual[r, c] or object_buffer[r, c] or shadow_buffer[r, c] for r, c in obj_cells):
                continue
            if any(actual[r, c] or shadow_buffer[r, c] for r, c in proj_cells):
                continue

            for r, c in obj_cells:
                grid[r, c] = int(spec["color"])
                actual[r, c] = True
            _mark_buffer(object_buffer, obj_cells, pad=1)
            _mark_buffer(shadow_buffer, proj_cells, pad=1)
            shadow_cells_all.update(proj_cells)
            placed = True
            break

        if not placed:
            return None

    expected = grid.copy()
    for r, c in shadow_cells_all:
        expected[r, c] = MAROON

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
