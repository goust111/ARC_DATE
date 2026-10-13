from __future__ import annotations

import random

import numpy as np


BLACK = 0
GRAY = 5
DIRS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
DIAG_DIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
FG_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GRID_HEIGHT_RANGE = (10, 18)
GRID_WIDTH_RANGE = (10, 18)
OBJECT_COUNT_RANGE = (2, 5)
OBJECT_AREA_RANGE = (3, 8)
SHAPE_CANVAS_HEIGHTS = range(2, 6)
SHAPE_CANVAS_WIDTHS = range(2, 7)
MAX_GENERATION_ATTEMPTS = 400
MAX_OBJECT_ATTEMPTS = 200
MAX_SHAPE_ATTEMPTS = 200


def _checkerboard(shape: tuple[int, int], even_color: int, odd_color: int) -> np.ndarray:
    rr, cc = np.indices(shape)
    return np.where((rr + cc) % 2 == 0, even_color, odd_color).astype(int)


def _dominant(values: np.ndarray) -> int:
    colors, counts = np.unique(values.astype(int), return_counts=True)
    return int(colors[np.argmax(counts)])


def _background_colors(grid: np.ndarray) -> tuple[int, int]:
    rr, cc = np.indices(grid.shape)
    even_mask = (rr + cc) % 2 == 0
    odd_mask = ~even_mask
    even_color = _dominant(grid[even_mask])
    odd_color = _dominant(grid[odd_mask])
    assert even_color != odd_color, "checkerboard phases must use two different colors"
    return even_color, odd_color


def _foreground_mask(grid: np.ndarray, even_bg: int, odd_bg: int) -> np.ndarray:
    rr, cc = np.indices(grid.shape)
    expected = np.where((rr + cc) % 2 == 0, even_bg, odd_bg)
    return grid != expected


def _extract_objects(grid: np.ndarray) -> list[dict[str, object]]:
    # Same-phase checkerboard objects cannot be raw 4-connected, so we recover
    # them as monochrome 8-connected diagonal clusters.
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    even_bg, odd_bg = _background_colors(g)
    fg = _foreground_mask(g, even_bg, odd_bg)
    visited = np.zeros((h, w), dtype=bool)
    objects: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or not fg[r, c]:
                continue
            color = int(g[r, c])
            phase = (r + c) % 2
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIRS8:
                    nr = cr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or not fg[nr, nc]:
                        continue
                    if int(g[nr, nc]) != color or (nr + nc) % 2 != phase:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            assert arr.size > 0
            assert len({(int(rr) + int(cc)) % 2 for rr, cc in coords}) == 1
            objects.append(
                {
                    "color": color,
                    "phase": int(phase),
                    "coords": arr,
                }
            )

    return objects


def _flip_coords(coords: np.ndarray, horizontal: bool) -> np.ndarray:
    out = coords.copy()
    r0 = int(coords[:, 0].min())
    r1 = int(coords[:, 0].max())
    c0 = int(coords[:, 1].min())
    c1 = int(coords[:, 1].max())
    if horizontal:
        out[:, 1] = c0 + c1 - out[:, 1]
    else:
        out[:, 0] = r0 + r1 - out[:, 0]
    return out


def _sorted_coords(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(r), int(c)) for r, c in coords))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flip each foreground object inside its own bounding box according to the
    checkerboard phase it occupies.

    Objects on the top-left checkerboard phase are mirrored horizontally.
    Objects on the opposite phase are mirrored vertically.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    even_bg, odd_bg = _background_colors(g)
    out = _checkerboard(g.shape, even_bg, odd_bg)

    objects = _extract_objects(g)
    for obj in objects:
        coords = obj["coords"]
        phase = int(obj["phase"])
        color = int(obj["color"])
        transformed = _flip_coords(coords, horizontal=(phase == 0))

        for r, c in transformed:
            rr = int(r)
            cc = int(c)
            expected_bg = even_bg if (rr + cc) % 2 == 0 else odd_bg
            assert int(out[rr, cc]) == expected_bg, "transformed objects cannot overlap"
            out[rr, cc] = color

    return out


def _normalize(coords: set[tuple[int, int]]) -> np.ndarray:
    r0 = min(r for r, _c in coords)
    c0 = min(c for _r, c in coords)
    arr = np.array(sorted((r - r0, c - c0) for r, c in coords), dtype=int)
    return arr


def _shape_frontier(
    cells: set[tuple[int, int]],
    available: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    frontier: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr, dc in DIAG_DIRS:
            nxt = (r + dr, c + dc)
            if nxt in available and nxt not in cells:
                frontier.add(nxt)
    return list(frontier)


def _generate_shape(horizontal: bool) -> np.ndarray | None:
    for _ in range(MAX_SHAPE_ATTEMPTS):
        canvas_h = random.choice(tuple(SHAPE_CANVAS_HEIGHTS))
        canvas_w = random.choice(tuple(SHAPE_CANVAS_WIDTHS))
        local_phase = random.randint(0, 1)
        available = {
            (r, c)
            for r in range(canvas_h)
            for c in range(canvas_w)
            if (r + c) % 2 == local_phase
        }
        if len(available) < OBJECT_AREA_RANGE[0]:
            continue

        max_area = min(OBJECT_AREA_RANGE[1], len(available))
        area = random.randint(OBJECT_AREA_RANGE[0], max_area)
        seed = random.choice(tuple(available))
        cells = {seed}

        while len(cells) < area:
            frontier = _shape_frontier(cells, available)
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != area:
            continue

        shape = _normalize(cells)
        if len({(int(r) + int(c)) % 2 for r, c in shape}) != 1:
            continue

        height = int(shape[:, 0].max()) + 1
        width = int(shape[:, 1].max()) + 1
        flipped = _flip_coords(shape, horizontal=horizontal)
        if horizontal:
            if width < 2:
                continue
        else:
            if height < 2:
                continue
        if _sorted_coords(shape) == _sorted_coords(flipped):
            continue
        return shape

    return None


def _shape_phase(shape: np.ndarray) -> int:
    phase_values = {(int(r) + int(c)) % 2 for r, c in shape}
    assert len(phase_values) == 1
    return phase_values.pop()


def _bbox_with_margin(
    top: int,
    left: int,
    height: int,
    width: int,
    grid_shape: tuple[int, int],
    margin: int = 1,
) -> tuple[slice, slice]:
    h, w = grid_shape
    return (
        slice(max(0, top - margin), min(h, top + height + margin)),
        slice(max(0, left - margin), min(w, left + width + margin)),
    )


def _valid_positions(
    reserved: np.ndarray,
    grid_shape: tuple[int, int],
    shape: np.ndarray,
    target_phase: int,
) -> list[tuple[int, int]]:
    h, w = grid_shape
    shape_h = int(shape[:, 0].max()) + 1
    shape_w = int(shape[:, 1].max()) + 1
    local_phase = _shape_phase(shape)
    top_left_phase = (target_phase - local_phase) % 2
    positions: list[tuple[int, int]] = []

    for top in range(h - shape_h + 1):
        for left in range(w - shape_w + 1):
            if (top + left) % 2 != top_left_phase:
                continue
            rs, cs = _bbox_with_margin(top, left, shape_h, shape_w, grid_shape)
            if reserved[rs, cs].any():
                continue
            positions.append((top, left))

    return positions


def _place_shape(grid: np.ndarray, reserved: np.ndarray, shape: np.ndarray, color: int, phase: int) -> dict[str, object] | None:
    positions = _valid_positions(reserved, grid.shape, shape, phase)
    if not positions:
        return None

    top, left = random.choice(positions)
    coords = shape + np.array([top, left], dtype=int)
    grid[coords[:, 0], coords[:, 1]] = int(color)
    shape_h = int(shape[:, 0].max()) + 1
    shape_w = int(shape[:, 1].max()) + 1
    rs, cs = _bbox_with_margin(top, left, shape_h, shape_w, grid.shape)
    reserved[rs, cs] = True

    return {
        "color": int(color),
        "phase": int(phase),
        "coords": coords,
    }


def _render_expected(shape: tuple[int, int], objects: list[dict[str, object]]) -> np.ndarray:
    out = _checkerboard(shape, BLACK, GRAY)
    for obj in objects:
        coords = np.asarray(obj["coords"], dtype=int)
        transformed = _flip_coords(coords, horizontal=(int(obj["phase"]) == 0))
        out[transformed[:, 0], transformed[:, 1]] = int(obj["color"])
    return out


def _phase_schedule(count: int) -> list[int]:
    half = count // 2
    phases = [0] * half + [1] * half
    if count % 2 == 1:
        phases.append(random.randint(0, 1))
    random.shuffle(phases)
    if 0 not in phases:
        phases[0] = 0
    if 1 not in phases:
        phases[-1] = 1
    random.shuffle(phases)
    return phases


def _generate_candidate() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    grid = _checkerboard((height, width), BLACK, GRAY)
    reserved = np.zeros((height, width), dtype=bool)

    obj_count = random.randint(*OBJECT_COUNT_RANGE)
    phases = _phase_schedule(obj_count)
    colors = random.sample(FG_COLORS, obj_count)
    specs: list[tuple[int, int, np.ndarray]] = []

    for phase in phases:
        horizontal = phase == 0
        shape = _generate_shape(horizontal=horizontal)
        if shape is None:
            return None
        specs.append((phase, colors[len(specs)], shape))

    # Place larger bounding boxes first to reduce retries.
    specs.sort(
        key=lambda item: (
            -int(item[2][:, 0].max()) - int(item[2][:, 1].max()),
            -len(item[2]),
            int(item[0]),
        )
    )

    placed: list[dict[str, object]] = []
    for phase, color, shape in specs:
        obj = _place_shape(grid, reserved, shape, color, phase)
        if obj is None:
            return None
        placed.append(obj)

    expected = _render_expected(grid.shape, placed)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _generate_candidate()
        if candidate is None:
            continue
        grid, expected = candidate

        objects = _extract_objects(grid)
        if len(objects) < 2:
            continue
        phases = {int(obj["phase"]) for obj in objects}
        if phases != {0, 1}:
            continue
        if any(len(obj["coords"]) < OBJECT_AREA_RANGE[0] or len(obj["coords"]) > OBJECT_AREA_RANGE[1] for obj in objects):
            continue
        if np.array_equal(grid, expected):
            continue
        if not np.array_equal(main(grid), expected):
            continue
        return grid

    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
