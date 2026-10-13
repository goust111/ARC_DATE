from __future__ import annotations

import random

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
MAROON = 9
SPECIAL_COLORS = {BLACK, YELLOW, GREY, MAROON}
TOWER_COLORS = (1, 2, 3, 6, 7, 8)

UP, RIGHT, DOWN, LEFT = range(4)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

GRID_MIN = 18
GRID_MAX = 26
GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1)]


def _diag_positions(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)]


def _solid_square(r0: int, c0: int, size: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r0 + size) for c in range(c0, c0 + size)}


def _same_color_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in SPECIAL_COLORS or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            cells = [(r, c)]
            while stack:
                cr, cc = stack.pop()
                for nr, nc in _neighbors4(cr, cc):
                    if not _in_bounds(grid.shape, nr, nc):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
                    cells.append((nr, nc))
            comps.append({"color": color, "cells": cells})
    return comps


def _parse_tower_component(comp: dict) -> dict:
    color = int(comp["color"])
    cells = {(int(r), int(c)) for r, c in comp["cells"]}
    assert len(cells) == 10, "tower component must have 10 cells"

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    hh = r1 - r0 + 1
    ww = c1 - c0 + 1

    body: set[tuple[int, int]]
    antenna: tuple[int, int]
    direction: int

    if hh == 4 and ww == 3:
        center_c = c0 + 1
        top_antenna = _solid_square(r0 + 1, c0, 3) | {(r0, center_c)}
        bottom_antenna = _solid_square(r0, c0, 3) | {(r1, center_c)}
        if cells == top_antenna:
            body = _solid_square(r0 + 1, c0, 3)
            antenna = (r0, center_c)
            direction = UP
        elif cells == bottom_antenna:
            body = _solid_square(r0, c0, 3)
            antenna = (r1, center_c)
            direction = DOWN
        else:
            raise AssertionError("invalid vertical tower shape")
    elif hh == 3 and ww == 4:
        center_r = r0 + 1
        left_antenna = _solid_square(r0, c0 + 1, 3) | {(center_r, c0)}
        right_antenna = _solid_square(r0, c0, 3) | {(center_r, c1)}
        if cells == left_antenna:
            body = _solid_square(r0, c0 + 1, 3)
            antenna = (center_r, c0)
            direction = LEFT
        elif cells == right_antenna:
            body = _solid_square(r0, c0, 3)
            antenna = (center_r, c1)
            direction = RIGHT
        else:
            raise AssertionError("invalid horizontal tower shape")
    else:
        raise AssertionError("tower bbox must be 4x3 or 3x4")

    return {
        "color": color,
        "body": body,
        "cells": set(cells),
        "antenna": antenna,
        "dir": direction,
    }


def _extract_scene(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], dict]]:
    towers = [_parse_tower_component(comp) for comp in _same_color_components(grid)]
    assert towers, "expected at least one signal tower"
    towers.sort(key=lambda t: (t["antenna"][0], t["antenna"][1], t["color"]))

    occupied = set()
    for tower in towers:
        assert occupied.isdisjoint(tower["cells"]), "towers must not overlap"
        occupied |= tower["cells"]

    maroon_cells = {tuple(map(int, rc)) for rc in np.argwhere(grid == MAROON)}
    mirrors: dict[tuple[int, int], dict] = {}
    for r, c in np.argwhere(grid == GREY):
        r = int(r)
        c = int(c)
        found: list[tuple[tuple[int, int], str]] = []
        candidates = [
            ((r - 1, c - 1), "\\"),
            ((r + 1, c + 1), "\\"),
            ((r - 1, c + 1), "/"),
            ((r + 1, c - 1), "/"),
        ]
        for pos, kind in candidates:
            if pos in maroon_cells:
                found.append((pos, kind))
        assert len(found) == 1, "each mirror must have exactly one diagonal maroon marker"
        marker, kind = found[0]
        assert (r, c) not in occupied, "mirror overlaps a tower"
        assert marker not in occupied, "mirror marker overlaps a tower"
        assert (r, c) not in mirrors, "duplicate mirror cell"
        mirrors[(r, c)] = {"kind": kind, "marker": marker}

    assert len(maroon_cells) == len(mirrors), "every maroon pixel must belong to one mirror"
    return towers, mirrors


def _turn_dir(direction: int, mirror_kind: str) -> int:
    if mirror_kind == "\\":
        return (direction + 1) % 4
    return (direction - 1) % 4


def _paint_rays(
    grid: np.ndarray,
    towers: list[dict],
    mirrors: dict[tuple[int, int], dict],
) -> tuple[np.ndarray, list[dict]]:
    out = np.array(grid, dtype=int, copy=True)
    hit_colors: dict[tuple[int, int], set[int]] = {}
    tower_cells = {cell for tower in towers for cell in tower["cells"]}
    records: list[dict] = []

    for tower in towers:
        r, c = tower["antenna"]
        direction = int(tower["dir"])
        visited_states: set[tuple[int, int, int]] = set()
        path: list[tuple[int, int]] = []
        reflections = 0
        stop = "edge"
        stop_cell: tuple[int, int] | None = None

        while True:
            state = (r, c, direction)
            if state in visited_states:
                stop = "loop"
                break
            visited_states.add(state)

            dr, dc = DIRS[direction]
            nr, nc = r + dr, c + dc
            if not _in_bounds(grid.shape, nr, nc):
                stop = "edge"
                break

            if (nr, nc) in mirrors:
                direction = _turn_dir(direction, mirrors[(nr, nc)]["kind"])
                reflections += 1
                r, c = nr, nc
                continue

            cell_value = int(grid[nr, nc])
            if cell_value != BLACK:
                stop = "tower" if (nr, nc) in tower_cells else "marker"
                stop_cell = (nr, nc)
                break

            path.append((nr, nc))
            hit_colors.setdefault((nr, nc), set()).add(int(tower["color"]))
            r, c = nr, nc

        records.append(
            {
                "color": int(tower["color"]),
                "path": path,
                "reflections": reflections,
                "stop": stop,
                "stop_cell": stop_cell,
            }
        )

    for cell, colors in hit_colors.items():
        out[cell] = YELLOW if len(colors) >= 2 else next(iter(colors))
    return out, records


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit colored rays from 3x3 towers, reflect them on marked grey mirrors, and
    mark multi-color intersections in yellow.
    """
    g = np.asarray(grid, dtype=int)
    towers, mirrors = _extract_scene(g)
    out, _ = _paint_rays(g, towers, mirrors)
    return out


def _interior_rows(h: int) -> list[int]:
    return list(range(5, h - 5))


def _interior_cols(w: int) -> list[int]:
    return list(range(5, w - 5))


def _make_side_tower(side: str, center: int, color: int, shape: tuple[int, int]) -> dict:
    h, w = shape
    if side == "left":
        body = {(r, c) for r in range(center - 1, center + 2) for c in range(0, 3)}
        antenna = (center, 3)
        direction = RIGHT
    elif side == "right":
        body = {(r, c) for r in range(center - 1, center + 2) for c in range(w - 3, w)}
        antenna = (center, w - 4)
        direction = LEFT
    elif side == "top":
        body = {(r, c) for r in range(0, 3) for c in range(center - 1, center + 2)}
        antenna = (3, center)
        direction = DOWN
    elif side == "bottom":
        body = {(r, c) for r in range(h - 3, h) for c in range(center - 1, center + 2)}
        antenna = (h - 4, center)
        direction = UP
    else:
        raise ValueError(side)

    cells = set(body)
    cells.add(antenna)
    assert all(_in_bounds(shape, r, c) for r, c in cells)
    return {
        "side": side,
        "color": int(color),
        "body": body,
        "cells": cells,
        "antenna": antenna,
        "dir": direction,
    }


def _occupied_cells(towers: list[dict], mirrors: dict[tuple[int, int], dict]) -> set[tuple[int, int]]:
    occupied = {cell for tower in towers for cell in tower["cells"]}
    for cell, mirror in mirrors.items():
        occupied.add(cell)
        occupied.add(tuple(mirror["marker"]))
    return occupied


def _make_mirror(
    shape: tuple[int, int],
    cell: tuple[int, int],
    kind: str,
    reserved: set[tuple[int, int]],
    other_mirror_cells: set[tuple[int, int]],
) -> dict | None:
    if cell in reserved or not _in_bounds(shape, *cell):
        return None
    r, c = cell
    all_diags = [pos for pos in _diag_positions(r, c) if _in_bounds(shape, *pos)]
    if any(pos in reserved for pos in all_diags):
        return None
    if kind == "\\":
        options = [(r - 1, c - 1), (r + 1, c + 1)]
    else:
        options = [(r - 1, c + 1), (r + 1, c - 1)]
    options = [pos for pos in options if _in_bounds(shape, *pos) and pos not in reserved]
    if not options:
        return None
    valid = []
    for marker in options:
        if any(marker in _diag_positions(mr, mc) for mr, mc in other_mirror_cells):
            continue
        valid.append(marker)
    if not valid:
        return None
    return {"kind": kind, "marker": random.choice(valid)}


def _scene_to_grid(shape: tuple[int, int], towers: list[dict], mirrors: dict[tuple[int, int], dict]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for tower in towers:
        for r, c in tower["cells"]:
            assert grid[r, c] == BLACK
            grid[r, c] = int(tower["color"])
    for (r, c), mirror in mirrors.items():
        assert grid[r, c] == BLACK
        grid[r, c] = GREY
        mr, mc = mirror["marker"]
        assert grid[mr, mc] == BLACK
        grid[mr, mc] = MAROON
    return grid


def _template_h_duel_mirror(shape: tuple[int, int], colors: list[int]):
    h, w = shape
    duel_row = random.choice(_interior_rows(h))
    top_col = random.choice(_interior_cols(w))
    mirror_rows = [r for r in _interior_rows(h) if abs(r - duel_row) >= 4]
    if not mirror_rows:
        return None
    mirror_row = random.choice(mirror_rows)

    towers = [
        _make_side_tower("left", duel_row, colors[0], shape),
        _make_side_tower("right", duel_row, colors[1], shape),
        _make_side_tower("top", top_col, colors[2], shape),
    ]
    mirrors: dict[tuple[int, int], dict] = {}
    mirror = _make_mirror(
        shape,
        (mirror_row, top_col),
        random.choice(["\\", "/"]),
        _occupied_cells(towers, mirrors),
        set(mirrors),
    )
    if mirror is None:
        return None
    mirrors[(mirror_row, top_col)] = mirror
    return towers, mirrors, {"min_reflections": 1, "min_tower": 2, "min_edge": 1, "min_yellow": 1}


def _template_v_duel_mirror(shape: tuple[int, int], colors: list[int]):
    h, w = shape
    duel_col = random.choice(_interior_cols(w))
    left_row = random.choice(_interior_rows(h))
    mirror_cols = [c for c in _interior_cols(w) if abs(c - duel_col) >= 4]
    if not mirror_cols:
        return None
    mirror_col = random.choice(mirror_cols)

    towers = [
        _make_side_tower("top", duel_col, colors[0], shape),
        _make_side_tower("bottom", duel_col, colors[1], shape),
        _make_side_tower("left", left_row, colors[2], shape),
    ]
    mirrors: dict[tuple[int, int], dict] = {}
    mirror = _make_mirror(
        shape,
        (left_row, mirror_col),
        random.choice(["\\", "/"]),
        _occupied_cells(towers, mirrors),
        set(mirrors),
    )
    if mirror is None:
        return None
    mirrors[(left_row, mirror_col)] = mirror
    return towers, mirrors, {"min_reflections": 1, "min_tower": 2, "min_edge": 1, "min_yellow": 1}


def _template_mirror_body(shape: tuple[int, int], colors: list[int]):
    h, w = shape
    target_side = random.choice(["left", "right"])
    if target_side == "right":
        top_choices = [c for c in _interior_cols(w) if c >= 8]
    else:
        top_choices = [c for c in _interior_cols(w) if c <= w - 9]
    if not top_choices:
        return None

    top_col = random.choice(top_choices)
    hit_row = random.choice(_interior_rows(h))
    if target_side == "right":
        bottom_choices = [c for c in _interior_cols(w) if c <= top_col - 3]
        mirror_kind = "/"
    else:
        bottom_choices = [c for c in _interior_cols(w) if c >= top_col + 3]
        mirror_kind = "\\"
    if not bottom_choices:
        return None
    bottom_col = random.choice(bottom_choices)

    towers = [
        _make_side_tower("top", top_col, colors[0], shape),
        _make_side_tower(target_side, hit_row, colors[1], shape),
        _make_side_tower("bottom", bottom_col, colors[2], shape),
    ]
    mirrors: dict[tuple[int, int], dict] = {}
    mirror = _make_mirror(
        shape,
        (hit_row, top_col),
        mirror_kind,
        _occupied_cells(towers, mirrors),
        set(mirrors),
    )
    if mirror is None:
        return None
    mirrors[(hit_row, top_col)] = mirror
    return towers, mirrors, {"min_reflections": 1, "min_tower": 1, "min_edge": 1, "min_yellow": 0}


def _template_four_cross(shape: tuple[int, int], colors: list[int]):
    cross_row = random.choice(_interior_rows(shape[0]))
    cross_col = random.choice(_interior_cols(shape[1]))
    towers = [
        _make_side_tower("left", cross_row, colors[0], shape),
        _make_side_tower("right", cross_row, colors[1], shape),
        _make_side_tower("top", cross_col, colors[2], shape),
        _make_side_tower("bottom", cross_col, colors[3], shape),
    ]
    mirrors: dict[tuple[int, int], dict] = {}
    return towers, mirrors, {"min_reflections": 0, "min_tower": 4, "min_edge": 0, "min_yellow": 1}


def _template_mirror_cross_edge(shape: tuple[int, int], colors: list[int]):
    h, w = shape
    top_choices = [c for c in _interior_cols(w) if c <= w - 9]
    left_choices = [r for r in _interior_rows(h) if r <= h - 9]
    if not top_choices or not left_choices:
        return None

    top_col = random.choice(top_choices)
    left_row = random.choice(left_choices)
    bend_row_choices = [r for r in _interior_rows(h) if r >= left_row + 3]
    bend_col_choices = [c for c in _interior_cols(w) if c >= top_col + 3]
    if not bend_row_choices or not bend_col_choices:
        return None
    bend_row = random.choice(bend_row_choices)
    bend_col = random.choice(bend_col_choices)

    towers = [
        _make_side_tower("top", top_col, colors[0], shape),
        _make_side_tower("left", left_row, colors[1], shape),
    ]
    mirrors: dict[tuple[int, int], dict] = {}
    reserved = _occupied_cells(towers, mirrors)
    first = _make_mirror(shape, (bend_row, top_col), "/", reserved, set(mirrors))
    if first is None:
        return None
    mirrors[(bend_row, top_col)] = first
    reserved = _occupied_cells(towers, mirrors)
    second = _make_mirror(shape, (left_row, bend_col), "\\", reserved, set(mirrors))
    if second is None:
        return None
    mirrors[(left_row, bend_col)] = second
    return towers, mirrors, {"min_reflections": 2, "min_tower": 0, "min_edge": 2, "min_yellow": 1}


TEMPLATES = (
    ("h_duel_mirror", 3, _template_h_duel_mirror),
    ("v_duel_mirror", 3, _template_v_duel_mirror),
    ("mirror_body", 3, _template_mirror_body),
    ("four_cross", 4, _template_four_cross),
    ("mirror_cross_edge", 2, _template_mirror_cross_edge),
)


def _count_yellow(grid: np.ndarray) -> int:
    return int(np.count_nonzero(grid == YELLOW))


def _add_spare_mirrors(
    shape: tuple[int, int],
    towers: list[dict],
    mirrors: dict[tuple[int, int], dict],
    forbidden: set[tuple[int, int]],
) -> None:
    extra = random.randint(0, 2)
    if extra == 0:
        return

    occupied = _occupied_cells(towers, mirrors) | set(forbidden)
    candidates = [
        (r, c)
        for r in range(2, shape[0] - 2)
        for c in range(2, shape[1] - 2)
        if (r, c) not in occupied
    ]
    random.shuffle(candidates)

    for cell in candidates:
        if extra == 0:
            break
        kind = random.choice(["\\", "/"])
        mirror = _make_mirror(shape, cell, kind, occupied, set(mirrors))
        if mirror is None:
            continue
        mirrors[cell] = mirror
        occupied = _occupied_cells(towers, mirrors) | set(forbidden)
        extra -= 1


def _generate_attempt() -> np.ndarray | None:
    shape = (random.randint(GRID_MIN, GRID_MAX), random.randint(GRID_MIN, GRID_MAX))
    name, color_count, template = random.choice(TEMPLATES)
    colors = random.sample(list(TOWER_COLORS), color_count)
    built = template(shape, colors)
    if built is None:
        return None
    towers, mirrors, constraints = built

    base = _scene_to_grid(shape, towers, mirrors)
    expected, records = _paint_rays(base, towers, mirrors)
    if any(record["stop"] == "loop" for record in records):
        return None

    path_cells = {cell for record in records for cell in record["path"]}
    if name != "four_cross":
        _add_spare_mirrors(shape, towers, mirrors, path_cells)
        base = _scene_to_grid(shape, towers, mirrors)
        expected, records = _paint_rays(base, towers, mirrors)
        if any(record["stop"] == "loop" for record in records):
            return None

    tower_stops = sum(record["stop"] == "tower" for record in records)
    edge_stops = sum(record["stop"] == "edge" for record in records)
    reflection_count = sum(record["reflections"] for record in records)
    if reflection_count < constraints["min_reflections"]:
        return None
    if tower_stops < constraints["min_tower"]:
        return None
    if edge_stops < constraints["min_edge"]:
        return None
    if _count_yellow(expected) < constraints["min_yellow"]:
        return None

    actual = main(base)
    if not np.array_equal(actual, expected):
        return None
    return base


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
