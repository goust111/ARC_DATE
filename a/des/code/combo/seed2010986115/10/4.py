from __future__ import annotations

import random

import numpy as np


Coord = tuple[int, int]
Template = dict[str, object]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_4(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(sorted(coords), dtype=int)
            components.append({"color": color, "coords": arr})

    return components


def _normalize_shape(coords: list[Coord] | tuple[Coord, ...]) -> tuple[Coord, ...]:
    min_r = min(r for r, _c in coords)
    min_c = min(c for _r, c in coords)
    return tuple(sorted((r - min_r, c - min_c) for r, c in coords))


def _rotate_shape(coords: tuple[Coord, ...]) -> tuple[Coord, ...]:
    return _normalize_shape([(c, -r) for r, c in coords])


def _farthest_index(coords: np.ndarray) -> int:
    center = coords.mean(axis=0)
    deltas = coords.astype(float) - center
    dist2 = np.sum(deltas * deltas, axis=1)
    max_dist2 = float(np.max(dist2))
    farthest = [idx for idx, value in enumerate(dist2) if abs(float(value) - max_dist2) < 1e-9]
    return max(farthest, key=lambda idx: (int(coords[idx, 0]), int(coords[idx, 1])))


def _growth_step(coords: np.ndarray) -> Coord:
    center = coords.mean(axis=0)
    anchor = coords[_farthest_index(coords)].astype(float)
    dr = float(anchor[0] - center[0])
    dc = float(anchor[1] - center[1])

    # Exact diagonal ties are rare here; prefer the row axis to keep the rule stable.
    if abs(dr) >= abs(dc):
        return (1, 0) if dr >= 0 else (-1, 0)
    return (0, 1) if dc >= 0 else (0, -1)


def _span_along(coords: np.ndarray, step: Coord) -> int:
    if step[0] != 0:
        return int(coords[:, 0].max() - coords[:, 0].min() + 1)
    return int(coords[:, 1].max() - coords[:, 1].min() + 1)


def _shifted(coords: np.ndarray, dr: int, dc: int) -> np.ndarray:
    shifted = coords.copy()
    shifted[:, 0] += int(dr)
    shifted[:, 1] += int(dc)
    return shifted


def _fits(shape: tuple[int, int], coords: np.ndarray) -> bool:
    h, w = shape
    return bool(
        np.all((0 <= coords[:, 0]) & (coords[:, 0] < h))
        and np.all((0 <= coords[:, 1]) & (coords[:, 1] < w))
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each colored seed cluster by stamping translated self-copies along its inferred direction.

    Input structure:
    - Black is the background.
    - Each foreground object is a small 4-connected monochromatic seed cluster.
    - Cluster sizes are between 2 and 5, and shapes are simple lines or L-shapes.

    Rule:
    - For each seed, compute its geometric centroid and find the farthest seed pixel.
    - Convert the centroid-to-pixel vector to the nearest cardinal direction.
    - Let the number of growth layers equal the seed size.
    - Let the translation span equal the seed's extent along the growth direction.
    - Stamp one unchanged copy of the seed at offsets 1*span, 2*span, ... until the next copy would exit the grid.
    - Keep the original seed unchanged.

    Key insight:
    - The seed geometry determines direction, while the seed area determines how many copies may appear.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    out = g.copy()

    for component in _components_4(g, bg):
        color = int(component["color"])
        coords = np.asarray(component["coords"], dtype=int)
        size = int(coords.shape[0])
        assert 2 <= size <= 5, "seed size must be between 2 and 5"

        step = _growth_step(coords)
        span = _span_along(coords, step)

        for layer in range(1, size + 1):
            delta_r = int(step[0] * span * layer)
            delta_c = int(step[1] * span * layer)
            shifted = _shifted(coords, delta_r, delta_c)
            if not _fits(g.shape, shifted):
                break
            for r, c in shifted:
                cell = int(out[r, c])
                assert cell in (bg, color), "growth copies must not overlap other clusters"
                out[r, c] = color

    return out


# --------------- generator ---------------

BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]

GRID_HEIGHT_RANGE = (12, 20)
GRID_WIDTH_RANGE = (12, 20)
OBJECT_COUNT_SMALL = (2, 3)
OBJECT_COUNT_LARGE = (3, 4)
LARGE_GRID_AREA = 260

MAX_SCENE_ATTEMPTS = 400
MAX_OBJECT_ATTEMPTS = 120
MAX_PLACEMENT_ATTEMPTS = 120

LINE_FAMILY = "line"
CORNER_FAMILY = "corner"
FAMILIES = [LINE_FAMILY, CORNER_FAMILY]

NEIGHBOR_8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 0), (0, 1),
    (1, -1), (1, 0), (1, 1),
]


def _make_template(coords: tuple[Coord, ...], family: str) -> Template:
    arr = np.array(coords, dtype=int)
    step = _growth_step(arr)
    span = _span_along(arr, step)
    return {
        "coords": arr,
        "family": family,
        "size": int(arr.shape[0]),
        "step": step,
        "span": span,
    }


def _template_library() -> tuple[list[Template], list[Template]]:
    line_shapes: set[tuple[Coord, ...]] = set()
    for length in range(2, 6):
        base = tuple((0, c) for c in range(length))
        shape = _normalize_shape(base)
        for _ in range(4):
            line_shapes.add(shape)
            shape = _rotate_shape(shape)

    corner_shapes: set[tuple[Coord, ...]] = set()
    bases = [
        ((0, 0), (1, 0), (1, 1)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
        ((0, 0), (1, 0), (2, 0), (3, 0), (3, 1)),
    ]
    for base in bases:
        shape = _normalize_shape(base)
        for _ in range(4):
            corner_shapes.add(shape)
            shape = _rotate_shape(shape)

    line_templates = [_make_template(shape, LINE_FAMILY) for shape in sorted(line_shapes)]
    corner_templates = [_make_template(shape, CORNER_FAMILY) for shape in sorted(corner_shapes)]
    return line_templates, corner_templates


LINE_TEMPLATES, CORNER_TEMPLATES = _template_library()


def _trajectory_cells(template: Template, top: int, left: int, visible_layers: int) -> set[Coord]:
    coords = np.asarray(template["coords"], dtype=int)
    span = int(template["span"])
    step_r, step_c = template["step"]
    cells: set[Coord] = set()

    for layer in range(visible_layers + 1):
        delta_r = int(top + step_r * span * layer)
        delta_c = int(left + step_c * span * layer)
        shifted = _shifted(coords, delta_r, delta_c)
        for r, c in shifted:
            cells.add((int(r), int(c)))

    return cells


def _expand_cells(cells: set[Coord]) -> set[Coord]:
    expanded: set[Coord] = set()
    for r, c in cells:
        for dr, dc in NEIGHBOR_8:
            expanded.add((r + dr, c + dc))
    return expanded


def _placement_ranges(template: Template, visible_layers: int, height: int, width: int) -> tuple[int, int, int, int] | None:
    coords = np.asarray(template["coords"], dtype=int)
    size = int(template["size"])
    span = int(template["span"])
    step_r, step_c = template["step"]
    max_r = int(coords[:, 0].max())
    max_c = int(coords[:, 1].max())

    row_lo, row_hi = 0, height - 1 - max_r
    col_lo, col_hi = 0, width - 1 - max_c

    if step_r == 1:
        row_hi = min(row_hi, height - 1 - max_r - visible_layers * span)
        if visible_layers < size:
            row_lo = max(row_lo, height - max_r - (visible_layers + 1) * span)
    elif step_r == -1:
        row_lo = max(row_lo, visible_layers * span)
        if visible_layers < size:
            row_hi = min(row_hi, (visible_layers + 1) * span - 1)

    if step_c == 1:
        col_hi = min(col_hi, width - 1 - max_c - visible_layers * span)
        if visible_layers < size:
            col_lo = max(col_lo, width - max_c - (visible_layers + 1) * span)
    elif step_c == -1:
        col_lo = max(col_lo, visible_layers * span)
        if visible_layers < size:
            col_hi = min(col_hi, (visible_layers + 1) * span - 1)

    if row_lo > row_hi or col_lo > col_hi:
        return None
    return row_lo, row_hi, col_lo, col_hi


def _random_template(family: str) -> Template:
    if family == LINE_FAMILY:
        return random.choice(LINE_TEMPLATES)
    return random.choice(CORNER_TEMPLATES)


def _sample_family() -> str:
    return random.choices(FAMILIES, weights=[2, 3], k=1)[0]


def _place_object(
    grid: np.ndarray,
    blocked: set[Coord],
    color: int,
    mode: str,
) -> tuple[bool, set[Coord], dict[str, object] | None]:
    height, width = grid.shape

    for _ in range(MAX_OBJECT_ATTEMPTS):
        template = _random_template(_sample_family())
        size = int(template["size"])
        visible_layers = size if mode == "full" else random.randint(1, size - 1)
        ranges = _placement_ranges(template, visible_layers, height, width)
        if ranges is None:
            continue

        row_lo, row_hi, col_lo, col_hi = ranges
        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            top = random.randint(row_lo, row_hi)
            left = random.randint(col_lo, col_hi)
            path_cells = _trajectory_cells(template, top, left, visible_layers)
            if any(cell in blocked for cell in path_cells):
                continue

            seed_coords = _shifted(np.asarray(template["coords"], dtype=int), top, left)
            for r, c in seed_coords:
                grid[r, c] = color

            return True, _expand_cells(path_cells), {
                "template": template,
                "top": top,
                "left": left,
                "visible_layers": visible_layers,
                "color": color,
            }

    return False, set(), None


def _scene_object_count(height: int, width: int) -> int:
    if height * width >= LARGE_GRID_AREA:
        return random.randint(*OBJECT_COUNT_LARGE)
    return random.randint(*OBJECT_COUNT_SMALL)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    object_count = _scene_object_count(height, width)
    colors = random.sample(PALETTE, object_count)

    grid = np.full((height, width), BG, dtype=int)
    blocked: set[Coord] = set()
    records: list[dict[str, object]] = []

    modes = ["full", "truncated"]
    while len(modes) < object_count:
        modes.append(random.choice(["full", "truncated"]))
    random.shuffle(modes)

    for color, mode in zip(colors, modes):
        ok, new_blocked, record = _place_object(grid, blocked, color, mode)
        if not ok or record is None:
            return None
        blocked |= new_blocked
        records.append(record)

    out = main(grid)
    if not np.any(out != grid):
        return None

    has_full = any(int(record["visible_layers"]) == int(record["template"]["size"]) for record in records)
    has_truncated = any(int(record["visible_layers"]) < int(record["template"]["size"]) for record in records)
    if not (has_full and has_truncated):
        return None

    for component in _components_4(grid, BG):
        coords = np.asarray(component["coords"], dtype=int)
        size = int(coords.shape[0])
        step = _growth_step(coords)
        span = _span_along(coords, step)
        seen_new = 0
        for layer in range(1, size + 1):
            shifted = _shifted(coords, step[0] * span * layer, step[1] * span * layer)
            if not _fits(grid.shape, shifted):
                break
            seen_new += 1
            if not np.all(out[shifted[:, 0], shifted[:, 1]] == int(component["color"])):
                return None
        if seen_new == 0:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_SCENE_ATTEMPTS):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid growth-cluster scene")
