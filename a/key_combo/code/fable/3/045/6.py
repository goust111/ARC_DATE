from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]

_BG = 0
_PALETTE = tuple(range(1, 10))
_PITCH = 6
_CELL_COUNT = 4
_GEN_RETRIES = 600

_DIRS: tuple[Coord, ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)
_DIAGONALS: tuple[Coord, ...] = ((-1, -1), (-1, 1), (1, -1), (1, 1))
_CARDINALS: tuple[Coord, ...] = ((-1, 0), (0, -1), (0, 1), (1, 0))

_SHAPES: dict[str, tuple[Coord, ...]] = {
    "dot": ((0, 0),),
    "hline3": ((0, 0), (0, 1), (0, 2)),
    "vline3": ((0, 0), (1, 0), (2, 0)),
}
_AREA_SHAPES = {
    1: ("dot",),
    3: ("hline3", "vline3"),
}


def _get_bg(g: np.ndarray) -> int:
    return int(Counter(map(int, g.ravel())).most_common(1)[0][0])


def _colors_present(g: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(g)))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _sign(value: int) -> int:
    return int(value > 0) - int(value < 0)


def _shape_size(shape: str) -> tuple[int, int]:
    offsets = _SHAPES[shape]
    rows = [r for r, _c in offsets]
    cols = [c for _r, c in offsets]
    return max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _paint_shape(g: np.ndarray, top_left: Coord, shape: str, color: int) -> None:
    r0, c0 = top_left
    for dr, dc in _SHAPES[shape]:
        g[r0 + dr, c0 + dc] = int(color)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    for r_raw, c_raw in np.argwhere(mask):
        r = int(r_raw)
        c = int(c_raw)
        if seen[r, c]:
            continue
        seen[r, c] = True
        q: deque[Coord] = deque([(r, c)])
        comp: list[Coord] = []
        while q:
            pr, pc = q.popleft()
            comp.append((pr, pc))
            for nr, nc in _neighbors4(pr, pc):
                if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.array(comp, dtype=int))
    return comps


def _regular_lattice(g: np.ndarray, bg: int) -> tuple[int, list[int], list[int], int, int]:
    candidates: list[tuple[int, list[int], list[int], int, int]] = []
    for color in _colors_present(g, exclude={bg}):
        coords = np.argwhere(g == int(color))
        rows = sorted({int(r) for r, _c in coords})
        cols = sorted({int(c) for _r, c in coords})
        if len(rows) < 3 or len(cols) < 3:
            continue
        if int(coords.shape[0]) != len(rows) * len(cols):
            continue
        if any(int(g[r, c]) != int(color) for r in rows for c in cols):
            continue

        row_diffs = np.diff(rows)
        col_diffs = np.diff(cols)
        if len(set(map(int, row_diffs))) != 1 or len(set(map(int, col_diffs))) != 1:
            continue
        row_pitch = int(row_diffs[0])
        col_pitch = int(col_diffs[0])
        if row_pitch >= 4 and col_pitch >= 4:
            candidates.append((int(color), rows, cols, row_pitch, col_pitch))

    assert len(candidates) == 1, "expected exactly one complete regular marker lattice"
    return candidates[0]


def _cell_for_bbox(
    box: tuple[int, int, int, int],
    marker_rows: list[int],
    marker_cols: list[int],
) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    row_cells = [i for i in range(len(marker_rows) - 1) if marker_rows[i] < r0 and r1 < marker_rows[i + 1]]
    col_cells = [j for j in range(len(marker_cols) - 1) if marker_cols[j] < c0 and c1 < marker_cols[j + 1]]
    assert len(row_cells) == 1 and len(col_cells) == 1, "sprite bbox must lie inside one lattice cell"
    return int(row_cells[0]), int(col_cells[0])


def _centered_top_left(axis_markers: list[int], cell_index: int, span: int) -> int:
    center2 = int(axis_markers[cell_index] + axis_markers[cell_index + 1])
    numerator = center2 - (int(span) - 1)
    assert numerator % 2 == 0, "sprite bbox parity must match the lattice cell center"
    return numerator // 2


def _build_sprite_plans(g: np.ndarray) -> tuple[int, int, list[dict]]:
    bg = _get_bg(g)
    marker_color, marker_rows, marker_cols, _row_pitch, _col_pitch = _regular_lattice(g, bg)
    sprite_colors = _colors_present(g, exclude={bg, marker_color})
    assert sprite_colors, "expected at least one sprite color"

    plans: list[dict] = []
    source_cells: set[tuple[int, int]] = set()
    target_cells: set[tuple[int, int]] = set()
    cell_rows = len(marker_rows) - 1
    cell_cols = len(marker_cols) - 1

    for color in sprite_colors:
        for coords in _components(g == int(color)):
            area = int(coords.shape[0])
            assert area > 0, "sprite must have positive area"
            r0, c0, r1, c1 = _bbox(coords)
            source_cell = _cell_for_bbox((r0, c0, r1, c1), marker_rows, marker_cols)
            assert source_cell not in source_cells, "expected at most one sprite in each lattice cell"
            source_cells.add(source_cell)

            si, sj = source_cell
            cell_center2_r = int(marker_rows[si] + marker_rows[si + 1])
            cell_center2_c = int(marker_cols[sj] + marker_cols[sj + 1])
            sprite_center2_r = int(r0 + r1)
            sprite_center2_c = int(c0 + c1)
            direction = (
                _sign(sprite_center2_r - cell_center2_r),
                _sign(sprite_center2_c - cell_center2_c),
            )

            if direction == (0, 0):
                target_cell = source_cell
            else:
                target_cell = (si + direction[0] * area, sj + direction[1] * area)
            ti, tj = target_cell
            assert 0 <= ti < cell_rows and 0 <= tj < cell_cols, "sprite velocity must land inside the lattice"
            assert target_cell not in target_cells, "target lattice cells must be unique"
            target_cells.add(target_cell)

            box_h = r1 - r0 + 1
            box_w = c1 - c0 + 1
            target_r0 = _centered_top_left(marker_rows, ti, box_h)
            target_c0 = _centered_top_left(marker_cols, tj, box_w)
            target_box = (target_r0, target_c0, target_r0 + box_h - 1, target_c0 + box_w - 1)
            assert _cell_for_bbox(target_box, marker_rows, marker_cols) == target_cell

            rel = coords - np.array([[r0, c0]], dtype=int)
            target_coords = rel + np.array([[target_r0, target_c0]], dtype=int)
            plans.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "target_coords": target_coords,
                    "area": area,
                    "direction": direction,
                    "source_cell": source_cell,
                    "target_cell": target_cell,
                }
            )

    assert plans, "expected at least one sprite"
    return bg, marker_color, plans


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each sprite through the marker lattice using its in-cell offset as direction and its area as distance.

    Input structure:
    - The background is the dominant color.
    - One foreground color forms a complete regular Cartesian marker lattice whose adjacent marker rows and columns define equal lattice cells.
    - Every other foreground component is a sprite fully contained in one open lattice cell; its bbox center is either exactly at that cell center or offset horizontally, vertically, or diagonally from it.

    Rule:
    - Keep the marker lattice fixed and clear every original sprite pixel.
    - For each off-center sprite, read the sign of its bbox-center offset from the containing cell center as an eight-way velocity direction.
    - Move the sprite by a number of lattice cells equal to its pixel area in that direction, and redraw the same relative shape centered in the target cell.
    - A sprite whose bbox center already equals its cell center is redrawn centered in the same cell.

    Key insight:
    - The apparent misalignment is an instruction: the in-cell offset gives direction, while sprite area gives the travel distance along the lattice.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert 1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30, "expected an ARC-sized grid"

    bg, _marker_color, plans = _build_sprite_plans(g)

    out = g.copy()
    for plan in plans:
        coords = plan["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    for plan in plans:
        color = int(plan["color"])
        target_coords = plan["target_coords"]
        assert np.all(out[target_coords[:, 0], target_coords[:, 1]] == bg), "target sprites must not overlap markers or other sprites"
        out[target_coords[:, 0], target_coords[:, 1]] = color

    return out


# --------------- generator ---------------


def _choose_canvas() -> tuple[tuple[int, int], list[int], list[int]]:
    first_r = random.randint(0, 2)
    first_c = random.randint(0, 2)
    marker_rows = [first_r + _PITCH * i for i in range(_CELL_COUNT + 1)]
    marker_cols = [first_c + _PITCH * i for i in range(_CELL_COUNT + 1)]
    max_tail_r = 30 - (marker_rows[-1] + 1)
    max_tail_c = 30 - (marker_cols[-1] + 1)
    if max_tail_r < 0 or max_tail_c < 0:
        raise RuntimeError("invalid generator lattice geometry")
    h = marker_rows[-1] + 1 + random.randint(0, max_tail_r)
    w = marker_cols[-1] + 1 + random.randint(0, max_tail_c)
    return (h, w), marker_rows, marker_cols


def _target_cell(source: Coord, direction: Coord, area: int) -> Coord:
    if direction == (0, 0):
        return source
    return source[0] + direction[0] * int(area), source[1] + direction[1] * int(area)


def _cell_options(area: int, direction: Coord, reserved: set[Coord]) -> list[Coord]:
    options: list[Coord] = []
    for r in range(_CELL_COUNT):
        for c in range(_CELL_COUNT):
            source = (r, c)
            target = _target_cell(source, direction, area)
            if not (0 <= target[0] < _CELL_COUNT and 0 <= target[1] < _CELL_COUNT):
                continue
            needed = {source, target}
            if needed & reserved:
                continue
            options.append(source)
    random.shuffle(options)
    return options


def _add_spec(specs: list[dict], reserved: set[Coord], area: int, direction: Coord, shape: str | None = None) -> bool:
    if shape is None:
        shape = random.choice(_AREA_SHAPES[int(area)])
    for source in _cell_options(area, direction, reserved):
        target = _target_cell(source, direction, area)
        specs.append(
            {
                "area": int(area),
                "direction": direction,
                "source_cell": source,
                "target_cell": target,
                "shape": shape,
            }
        )
        reserved.add(source)
        reserved.add(target)
        return True
    return False


def _cell_center(marker_rows: list[int], marker_cols: list[int], cell: Coord) -> Coord:
    r, c = cell
    return (marker_rows[r] + marker_rows[r + 1]) // 2, (marker_cols[c] + marker_cols[c + 1]) // 2


def _centered_shape_top_left(marker_rows: list[int], marker_cols: list[int], cell: Coord, shape: str) -> Coord:
    h, w = _shape_size(shape)
    return _centered_top_left(marker_rows, cell[0], h), _centered_top_left(marker_cols, cell[1], w)


def _source_top_left(marker_rows: list[int], marker_cols: list[int], spec: dict) -> Coord:
    shape = str(spec["shape"])
    direction = spec["direction"]
    top_r, top_c = _centered_shape_top_left(marker_rows, marker_cols, spec["source_cell"], shape)
    if direction == (0, 0):
        return top_r, top_c

    shape_h, shape_w = _shape_size(shape)
    max_dr = max(0, (_PITCH - shape_h) // 2)
    max_dc = max(0, (_PITCH - shape_w) // 2)
    dr, dc = direction
    off_r = 0 if dr == 0 else dr * random.randint(1, max_dr)
    off_c = 0 if dc == 0 else dc * random.randint(1, max_dc)
    return top_r + off_r, top_c + off_c


def _render_from_specs(
    shape: tuple[int, int],
    marker_rows: list[int],
    marker_cols: list[int],
    marker_color: int,
    specs: list[dict],
    *,
    use_source_offsets: bool,
    target_override: dict[int, Coord] | None = None,
) -> np.ndarray:
    grid = np.full(shape, _BG, dtype=int)
    grid[np.ix_(marker_rows, marker_cols)] = int(marker_color)
    for index, spec in enumerate(specs):
        shape_name = str(spec["shape"])
        if target_override is None:
            cell = spec["source_cell"] if use_source_offsets else spec["target_cell"]
        else:
            cell = target_override[index]

        if use_source_offsets and target_override is None:
            top_left = spec["source_top_left"]
        else:
            top_left = _centered_shape_top_left(marker_rows, marker_cols, cell, shape_name)
        _paint_shape(grid, top_left, shape_name, int(spec["color"]))
    return grid


def _make_required_specs() -> list[dict] | None:
    specs: list[dict] = []
    reserved: set[Coord] = set()

    required = [
        (3, random.choice(_DIRS), None),
        (1, random.choice(_DIAGONALS), "dot"),
        (1, random.choice(_CARDINALS), "dot"),
        (random.choice((1, 3)), (0, 0), None),
    ]
    for area, direction, shape in required:
        if not _add_spec(specs, reserved, area, direction, shape):
            return None

    target_count = random.randint(5, 7)
    while len(specs) < target_count:
        added = False
        for _ in range(40):
            area = 1 if random.random() < 0.62 else 3
            direction = (0, 0) if random.random() < 0.20 else random.choice(_DIRS)
            if _add_spec(specs, reserved, area, direction):
                added = True
                break
        if not added:
            return None

    random.shuffle(specs)
    return specs


def _assign_colors(specs: list[dict]) -> int:
    marker_color = random.choice(_PALETTE)
    sprite_palette = [color for color in _PALETTE if color != marker_color]
    colors = random.sample(sprite_palette, len(specs))
    for spec, color in zip(specs, colors):
        spec["color"] = int(color)
    return int(marker_color)


def _wrong_step_one_targets(specs: list[dict]) -> dict[int, Coord]:
    targets: dict[int, Coord] = {}
    for index, spec in enumerate(specs):
        source = spec["source_cell"]
        direction = spec["direction"]
        if direction == (0, 0):
            targets[index] = source
        else:
            targets[index] = (source[0] + direction[0], source[1] + direction[1])
    return targets


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, specs: list[dict], marker_rows: list[int], marker_cols: list[int], marker_color: int) -> bool:
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if np.array_equal(inp, expected):
        return False

    moving = [spec for spec in specs if spec["direction"] != (0, 0)]
    centered = [spec for spec in specs if spec["direction"] == (0, 0)]
    if not moving or not centered:
        return False
    if {1, 3} - {int(spec["area"]) for spec in moving}:
        return False
    if not any(abs(spec["direction"][0]) + abs(spec["direction"][1]) == 2 for spec in moving):
        return False
    if not any(abs(spec["direction"][0]) + abs(spec["direction"][1]) == 1 for spec in moving):
        return False

    dot_dirs = {spec["direction"] for spec in moving if spec["shape"] == "dot"}
    if len(dot_dirs) < 2:
        return False
    if not any(
        int(spec["area"]) == 3
        and spec["direction"] != (0, 0)
        and spec["target_cell"] != _target_cell(spec["source_cell"], spec["direction"], 1)
        for spec in specs
    ):
        return False

    same_cell_targets = {i: spec["source_cell"] for i, spec in enumerate(specs)}
    center_only = _render_from_specs(inp.shape, marker_rows, marker_cols, marker_color, specs, use_source_offsets=False, target_override=same_cell_targets)
    if np.array_equal(center_only, expected):
        return False

    step_one = _render_from_specs(inp.shape, marker_rows, marker_cols, marker_color, specs, use_source_offsets=False, target_override=_wrong_step_one_targets(specs))
    if np.array_equal(step_one, expected):
        return False

    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    canvas_shape, marker_rows, marker_cols = _choose_canvas()
    specs = _make_required_specs()
    if specs is None:
        return None

    marker_color = _assign_colors(specs)
    for spec in specs:
        spec["source_top_left"] = _source_top_left(marker_rows, marker_cols, spec)

    inp = _render_from_specs(
        canvas_shape,
        marker_rows,
        marker_cols,
        marker_color,
        specs,
        use_source_offsets=True,
    )
    expected = _render_from_specs(
        canvas_shape,
        marker_rows,
        marker_cols,
        marker_color,
        specs,
        use_source_offsets=False,
    )

    if not _semantic_checks(inp, expected, specs, marker_rows, marker_cols, marker_color):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lattice-offset velocity input")
