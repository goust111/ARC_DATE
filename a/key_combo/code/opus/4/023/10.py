from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

FRAME_COLORS = (BLUE, RED, GREEN, YELLOW)
MOVE_STEP = {
    BLUE: (-1, 0),
    RED: (1, 0),
    GREEN: (0, -1),
    YELLOW: (0, 1),
}

_RNG = np.random.default_rng()
_GEN_CALL_COUNT = 0
_GEN_RETRIES = 500
_SCENARIO_CYCLE = ("both", "contact_only", "sticky_only", "plain")


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _validate_frame(grid: np.ndarray) -> None:
    h, w = grid.shape
    assert h >= 5 and w >= 5, "grid too small for the bordered layout"
    assert int(grid[0, 0]) == BLACK and int(grid[0, w - 1]) == BLACK
    assert int(grid[h - 1, 0]) == BLACK and int(grid[h - 1, w - 1]) == BLACK
    assert np.all(grid[0, 1 : w - 1] == BLUE), "top frame must be blue"
    assert np.all(grid[h - 1, 1 : w - 1] == RED), "bottom frame must be red"
    assert np.all(grid[1 : h - 1, 0] == GREEN), "left frame must be green"
    assert np.all(grid[1 : h - 1, w - 1] == YELLOW), "right frame must be yellow"


def _extract_objects(grid: np.ndarray) -> list[dict]:
    interior = np.asarray(grid[1:-1, 1:-1], dtype=int)
    objects: list[dict] = []
    for color in FRAME_COLORS:
        for coords in _components_mask(interior == color):
            abs_coords = coords + np.array([1, 1], dtype=int)
            objects.append(
                {
                    "color": int(color),
                    "coords": abs_coords,
                    "area": int(abs_coords.shape[0]),
                    "bbox": _bbox(abs_coords),
                }
            )
    return objects


def _is_blocked(coords: np.ndarray, color: int, height: int, width: int) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    if color == BLUE:
        return r0 == 1
    if color == RED:
        return r1 == height - 2
    if color == GREEN:
        return c0 == 1
    if color == YELLOW:
        return c1 == width - 2
    raise AssertionError("unexpected moving color")


def _shift(coords: np.ndarray, dr: int, dc: int) -> np.ndarray:
    return coords + np.array([dr, dc], dtype=int)


def _touches_set(coords: np.ndarray, other: set[tuple[int, int]]) -> bool:
    for r_raw, c_raw in coords:
        r = int(r_raw)
        c = int(c_raw)
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in other:
                return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split interior objects by extreme area, then move only the middle-area objects one step toward the frame side matching their color.

    Input structure:
    - The outer ring is a one-pixel frame whose corners are black, with blue on the top edge, red on the bottom edge, green on the left edge, and yellow on the right edge.
    - Inside the frame, black is background and 4 to 6 single-color 4-connected objects use only the frame-side colors.
    - All object areas are distinct, so the smallest and largest objects are unique.

    Rule:
    - Measure each interior object's area.
    - Erase the unique smallest object to black.
    - Recolor the unique largest object to orange and leave it fixed.
    - Move every remaining object one cell toward the frame side whose color matches the object color; if such a move would overlap the frame, that object stays put instead.
    - After moving, recolor any moved object that is 4-neighbor adjacent to the fixed orange object to orange as well.
    - Keep the frame unchanged.

    Key insight:
    - Area decides each object's role, while color independently decides the direction for the movable roles; the orange contact rule appears only after these two criteria interact.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    _validate_frame(g)

    objects = _extract_objects(g)
    assert 4 <= len(objects) <= 6, "expected 4 to 6 interior objects"
    areas = [obj["area"] for obj in objects]
    assert len(set(areas)) == len(areas), "all object areas must be distinct"

    ordered = sorted(objects, key=lambda obj: obj["area"])
    smallest = ordered[0]
    largest = ordered[-1]
    assert smallest["color"] != largest["color"], "largest and smallest colors must differ"

    movers = ordered[1:-1]
    mover_colors = [obj["color"] for obj in movers]
    assert len(mover_colors) == len(set(mover_colors)), "movable colors must be unique"

    out = g.copy()
    out[smallest["coords"][:, 0], smallest["coords"][:, 1]] = BLACK
    out[largest["coords"][:, 0], largest["coords"][:, 1]] = ORANGE

    for obj in movers:
        out[obj["coords"][:, 0], obj["coords"][:, 1]] = BLACK

    frozen = {(int(r), int(c)) for r, c in largest["coords"]}
    painted: list[tuple[np.ndarray, int]] = []
    for obj in movers:
        dr, dc = MOVE_STEP[obj["color"]]
        moved = obj["coords"] if _is_blocked(obj["coords"], obj["color"], *g.shape) else _shift(obj["coords"], dr, dc)
        assert np.all((1 <= moved[:, 0]) & (moved[:, 0] < g.shape[0] - 1))
        assert np.all((1 <= moved[:, 1]) & (moved[:, 1] < g.shape[1] - 1))
        painted.append((moved, obj["color"]))

    occupied = set(frozen)
    for moved, _color in painted:
        cell_set = {(int(r), int(c)) for r, c in moved}
        assert occupied.isdisjoint(cell_set), "moved objects must not overlap the frozen object or each other"
        occupied |= cell_set

    for moved, color in painted:
        paint_color = ORANGE if _touches_set(moved, frozen) else color
        out[moved[:, 0], moved[:, 1]] = int(paint_color)

    return out


def _mask_from_rows(rows: tuple[str, ...]) -> np.ndarray:
    width = max(len(row) for row in rows)
    return np.array(
        [[row[c] == "1" if c < len(row) else False for c in range(width)] for row in rows],
        dtype=bool,
    )


def _normalize_coords(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    norm = coords.copy()
    norm[:, 0] -= norm[:, 0].min()
    norm[:, 1] -= norm[:, 1].min()
    return tuple(sorted((int(r), int(c)) for r, c in norm))


def _shape_variants(rows: tuple[str, ...]) -> list[np.ndarray]:
    mask = _mask_from_rows(rows)
    variants: dict[tuple[tuple[int, int], ...], np.ndarray] = {}
    for flip in (False, True):
        cur = np.fliplr(mask) if flip else mask
        for k in range(4):
            rot = np.rot90(cur, k)
            coords = np.argwhere(rot)
            key = _normalize_coords(coords)
            variants.setdefault(key, np.asarray(key, dtype=int))
    return list(variants.values())


_BASE_SHAPES = [
    ("1",),
    ("11",),
    ("111",),
    ("10", "11"),
    ("1111",),
    ("11", "11"),
    ("111", "010"),
    ("11111",),
    ("010", "111", "010"),
    ("110", "110", "100"),
    ("111", "111"),
    ("1111", "1000", "1000"),
    ("1111", "111"),
    ("1111", "1111"),
    ("111", "101", "111"),
]

_AREA_SHAPES: dict[int, list[np.ndarray]] = {}
for rows in _BASE_SHAPES:
    for variant in _shape_variants(rows):
        area = int(variant.shape[0])
        _AREA_SHAPES.setdefault(area, [])
        if not any(np.array_equal(variant, old) for old in _AREA_SHAPES[area]):
            _AREA_SHAPES[area].append(variant)


def _make_frame(height: int, width: int) -> np.ndarray:
    g = np.zeros((height, width), dtype=int)
    g[0, 1 : width - 1] = BLUE
    g[height - 1, 1 : width - 1] = RED
    g[1 : height - 1, 0] = GREEN
    g[1 : height - 1, width - 1] = YELLOW
    return g


def _coords_to_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in coords}


def _shift_set(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + int(dr), c + int(dc)) for r, c in cells}


def _cells_from_top_left(local: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    return {(int(top + r), int(left + c)) for r, c in local}


def _bbox_size(local: np.ndarray) -> tuple[int, int]:
    r1 = int(local[:, 0].max())
    c1 = int(local[:, 1].max())
    return r1 + 1, c1 + 1


def _inside_interior(cells: set[tuple[int, int]], height: int, width: int) -> bool:
    return all(1 <= r < height - 1 and 1 <= c < width - 1 for r, c in cells)


def _touches8(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r1, c1 in a:
        for r2, c2 in b:
            if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                return True
    return False


def _touches4(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r1, c1 in a:
        for r2, c2 in b:
            if abs(r1 - r2) + abs(c1 - c2) == 1:
                return True
    return False


def _blocked_by_frame_cells(cells: set[tuple[int, int]], color: int, height: int, width: int) -> bool:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    if color == BLUE:
        return min(rows) == 1
    if color == RED:
        return max(rows) == height - 2
    if color == GREEN:
        return min(cols) == 1
    if color == YELLOW:
        return max(cols) == width - 2
    raise AssertionError("unexpected color")


def _scenario_flags(name: str) -> tuple[bool, bool]:
    return name in {"both", "contact_only"}, name in {"both", "sticky_only"}


def _positions_for_shape(local: np.ndarray, height: int, width: int) -> list[tuple[int, int]]:
    shape_h, shape_w = _bbox_size(local)
    return [(r, c) for r in range(1, height - shape_h) for c in range(1, width - shape_w)]


def _pick_one(candidates: list):
    idx = int(_RNG.integers(0, len(candidates)))
    return candidates[idx]


def _place_max_object(spec: dict, height: int, width: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    local = spec["local"]
    shape_h, shape_w = _bbox_size(local)
    central = [
        (r, c)
        for r in range(2, max(3, height - shape_h - 1))
        for c in range(2, max(3, width - shape_w - 1))
    ]
    candidates = central if central else _positions_for_shape(local, height, width)
    _RNG.shuffle(candidates)
    for top, left in candidates:
        cells = _cells_from_top_left(local, top, left)
        if _inside_interior(cells, height, width):
            return cells, cells
    return None


def _place_contact_mover(
    spec: dict,
    height: int,
    width: int,
    input_groups: list[set[tuple[int, int]]],
    final_groups: list[tuple[str, set[tuple[int, int]]]],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    dr, dc = MOVE_STEP[spec["color"]]
    frozen = dict(final_groups)["max"]
    candidates = []
    for top, left in _positions_for_shape(spec["local"], height, width):
        final_cells = _cells_from_top_left(spec["local"], top, left)
        input_cells = _shift_set(final_cells, -dr, -dc)
        if not _inside_interior(final_cells, height, width) or not _inside_interior(input_cells, height, width):
            continue
        if _blocked_by_frame_cells(input_cells, spec["color"], height, width):
            continue
        if not _touches4(final_cells, frozen):
            continue
        if _touches4(input_cells, frozen) or _touches8(input_cells, frozen):
            continue
        ok = True
        for other in input_groups:
            if _touches8(input_cells, other):
                ok = False
                break
        if not ok:
            continue
        for name, other in final_groups:
            if name == "max":
                if final_cells & other or (_touches8(final_cells, other) and not _touches4(final_cells, other)):
                    ok = False
                    break
            elif _touches8(final_cells, other):
                ok = False
                break
        if ok:
            candidates.append((input_cells, final_cells))
    if not candidates:
        return None
    return _pick_one(candidates)


def _place_sticky_mover(
    spec: dict,
    height: int,
    width: int,
    input_groups: list[set[tuple[int, int]]],
    final_groups: list[tuple[str, set[tuple[int, int]]]],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    candidates = []
    for top, left in _positions_for_shape(spec["local"], height, width):
        cells = _cells_from_top_left(spec["local"], top, left)
        if not _blocked_by_frame_cells(cells, spec["color"], height, width):
            continue
        ok = True
        for other in input_groups:
            if _touches8(cells, other):
                ok = False
                break
        if not ok:
            continue
        for _, other in final_groups:
            if _touches8(cells, other):
                ok = False
                break
        if ok:
            candidates.append((cells, cells))
    if not candidates:
        return None
    return _pick_one(candidates)


def _place_plain_mover(
    spec: dict,
    height: int,
    width: int,
    input_groups: list[set[tuple[int, int]]],
    final_groups: list[tuple[str, set[tuple[int, int]]]],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    dr, dc = MOVE_STEP[spec["color"]]
    candidates = []
    for top, left in _positions_for_shape(spec["local"], height, width):
        final_cells = _cells_from_top_left(spec["local"], top, left)
        input_cells = _shift_set(final_cells, -dr, -dc)
        if not _inside_interior(final_cells, height, width) or not _inside_interior(input_cells, height, width):
            continue
        if _blocked_by_frame_cells(input_cells, spec["color"], height, width):
            continue
        ok = True
        for other in input_groups:
            if _touches8(input_cells, other):
                ok = False
                break
        if not ok:
            continue
        for _, other in final_groups:
            if _touches8(final_cells, other):
                ok = False
                break
        if ok:
            candidates.append((input_cells, final_cells))
    if not candidates:
        return None
    return _pick_one(candidates)


def _place_min_object(
    spec: dict,
    height: int,
    width: int,
    input_groups: list[set[tuple[int, int]]],
    final_groups: list[tuple[str, set[tuple[int, int]]]],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    candidates = []
    for top, left in _positions_for_shape(spec["local"], height, width):
        cells = _cells_from_top_left(spec["local"], top, left)
        ok = True
        for other in input_groups:
            if _touches8(cells, other):
                ok = False
                break
        if not ok:
            continue
        for _, other in final_groups:
            if _touches8(cells, other):
                ok = False
                break
        if ok:
            candidates.append((cells, set()))
    if not candidates:
        return None
    return _pick_one(candidates)


def _render_scene(height: int, width: int, placed_specs: list[dict], *, use_final: bool) -> np.ndarray:
    grid = _make_frame(height, width)
    for spec in placed_specs:
        cells = spec["final_cells"] if use_final else spec["input_cells"]
        color = spec["final_color"] if use_final else spec["color"]
        for r, c in cells:
            grid[r, c] = int(color)
    return grid


def _expected_from_plan(height: int, width: int, placed_specs: list[dict]) -> np.ndarray:
    frozen = next(spec["final_cells"] for spec in placed_specs if spec["role"] == "max")
    grid = _make_frame(height, width)
    for spec in placed_specs:
        if spec["role"] == "min":
            continue
        final_color = spec["color"]
        if spec["role"] == "max":
            final_color = ORANGE
        elif spec["role"] == "move" and _touches4(spec["final_cells"], frozen):
            final_color = ORANGE
        spec["final_color"] = final_color
        for r, c in spec["final_cells"]:
            grid[r, c] = int(final_color)
    return grid


def _actual_branch_profile(placed_specs: list[dict], height: int, width: int) -> tuple[int, int]:
    frozen = next(spec["final_cells"] for spec in placed_specs if spec["role"] == "max")
    sticky = 0
    contact = 0
    for spec in placed_specs:
        if spec["role"] != "move":
            continue
        if spec["input_cells"] == spec["final_cells"]:
            sticky += 1
        if _touches4(spec["final_cells"], frozen):
            contact += 1
        if spec["input_cells"] != spec["final_cells"]:
            assert not _blocked_by_frame_cells(spec["input_cells"], spec["color"], height, width)
    return contact, sticky


def _simple_area_only_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    objects = sorted(_extract_objects(g), key=lambda obj: obj["area"])
    out = g.copy()
    out[objects[0]["coords"][:, 0], objects[0]["coords"][:, 1]] = BLACK
    out[objects[-1]["coords"][:, 0], objects[-1]["coords"][:, 1]] = ORANGE
    return out


def _validate_generated_scene(
    inp: np.ndarray,
    expected: np.ndarray,
    planned: list[dict],
    scenario: str,
) -> bool:
    try:
        _validate_frame(inp)
        _validate_frame(expected)
        objects = _extract_objects(inp)
    except Exception:
        return False

    if len(objects) != len(planned):
        return False
    if len(set(obj["area"] for obj in objects)) != len(objects):
        return False

    ordered = sorted(objects, key=lambda obj: obj["area"])
    movers = ordered[1:-1]
    if ordered[0]["color"] == ordered[-1]["color"]:
        return False
    if len({obj["color"] for obj in movers}) != len(movers):
        return False

    contact_need, sticky_need = _scenario_flags(scenario)
    actual_contact, actual_sticky = _actual_branch_profile(planned, *inp.shape)
    if contact_need and actual_contact == 0:
        return False
    if not contact_need and actual_contact != 0:
        return False
    if sticky_need and actual_sticky == 0:
        return False
    if not sticky_need and actual_sticky != 0:
        return False

    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(expected, inp):
        return False
    if np.array_equal(expected, _simple_area_only_output(inp)):
        return False
    return True


def _choose_areas(n_objects: int) -> list[int] | None:
    areas_pool = np.array(sorted(_AREA_SHAPES), dtype=int)
    for _ in range(80):
        choice = sorted(int(v) for v in _RNG.choice(areas_pool, size=n_objects, replace=False))
        if choice[-1] - choice[0] >= 3:
            return choice
    return None


def _choose_colors(mover_count: int) -> tuple[list[int], int, int]:
    mover_colors = [int(v) for v in _RNG.choice(np.array(FRAME_COLORS, dtype=int), size=mover_count, replace=False)]
    max_color = int(_pick_one(list(FRAME_COLORS)))
    min_choices = [c for c in FRAME_COLORS if c != max_color]
    min_color = int(_pick_one(min_choices))
    return mover_colors, max_color, min_color


def _build_specs(n_objects: int, scenario: str) -> list[dict] | None:
    areas = _choose_areas(n_objects)
    if areas is None:
        return None

    mover_count = n_objects - 2
    mover_colors, max_color, min_color = _choose_colors(mover_count)
    specs = [
        {
            "name": "min",
            "role": "min",
            "area": areas[0],
            "color": min_color,
            "local": _pick_one(_AREA_SHAPES[areas[0]]).copy(),
        }
    ]
    for idx, (area, color) in enumerate(zip(areas[1:-1], mover_colors), start=1):
        specs.append(
            {
                "name": f"move_{idx}",
                "role": "move",
                "area": area,
                "color": int(color),
                "local": _pick_one(_AREA_SHAPES[area]).copy(),
                "force_contact": False,
                "force_sticky": False,
            }
        )
    specs.append(
        {
            "name": "max",
            "role": "max",
            "area": areas[-1],
            "color": max_color,
            "local": _pick_one(_AREA_SHAPES[areas[-1]]).copy(),
        }
    )

    movers = [spec for spec in specs if spec["role"] == "move"]
    contact_need, sticky_need = _scenario_flags(scenario)
    _RNG.shuffle(movers)
    if contact_need:
        movers[0]["force_contact"] = True
    if sticky_need:
        sticky_idx = 1 if contact_need and len(movers) > 1 else 0
        movers[sticky_idx]["force_sticky"] = True
    return specs


def _attempt_scene(scenario: str) -> np.ndarray | None:
    height = int(_RNG.integers(11, 16))
    width = int(_RNG.integers(11, 16))
    n_objects = int(_RNG.integers(4, 7))
    specs = _build_specs(n_objects, scenario)
    if specs is None:
        return None

    max_spec = next(spec for spec in specs if spec["role"] == "max")
    placed_max = _place_max_object(max_spec, height, width)
    if placed_max is None:
        return None
    max_spec["input_cells"], max_spec["final_cells"] = placed_max

    input_groups = [max_spec["input_cells"]]
    final_groups = [("max", max_spec["final_cells"])]

    movers = [spec for spec in specs if spec["role"] == "move"]
    _RNG.shuffle(movers)
    for spec in movers:
        if spec["force_contact"]:
            placed = _place_contact_mover(spec, height, width, input_groups, final_groups)
        elif spec["force_sticky"]:
            placed = _place_sticky_mover(spec, height, width, input_groups, final_groups)
        else:
            placed = _place_plain_mover(spec, height, width, input_groups, final_groups)
        if placed is None:
            return None
        spec["input_cells"], spec["final_cells"] = placed
        input_groups.append(spec["input_cells"])
        final_groups.append((spec["name"], spec["final_cells"]))

    min_spec = next(spec for spec in specs if spec["role"] == "min")
    placed_min = _place_min_object(min_spec, height, width, input_groups, final_groups)
    if placed_min is None:
        return None
    min_spec["input_cells"], min_spec["final_cells"] = placed_min

    placed_specs = sorted(specs, key=lambda spec: (spec["area"], spec["name"]))
    for spec in placed_specs:
        spec["coords_area"] = len(spec["input_cells"])

    inp = _render_scene(height, width, placed_specs, use_final=False)
    expected = _expected_from_plan(height, width, placed_specs)
    if not _validate_generated_scene(inp, expected, placed_specs, scenario):
        return None
    return inp


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    scenario = _SCENARIO_CYCLE[_GEN_CALL_COUNT % len(_SCENARIO_CYCLE)]
    _GEN_CALL_COUNT += 1

    for _ in range(_GEN_RETRIES):
        inp = _attempt_scene(scenario)
        if inp is not None:
            return inp

    raise RuntimeError(f"failed to generate a valid sample for scenario {scenario}")
