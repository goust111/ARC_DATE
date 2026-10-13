from __future__ import annotations

import random

import numpy as np


UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)
PERPENDICULAR = {
    UP: (LEFT, RIGHT),
    DOWN: (LEFT, RIGHT),
    LEFT: (UP, DOWN),
    RIGHT: (UP, DOWN),
}

GRID_H_RANGE = (14, 22)
GRID_W_RANGE = (14, 22)
TOWER_COUNT_CHOICES = (1, 2, 3)
TOWER_COUNT_WEIGHTS = (2, 3, 2)
BEND_COUNT_CHOICES = (0, 1, 2)
BEND_COUNT_WEIGHTS = (2, 3, 2)
MIN_SEG_LEN = 2
MAX_SEG_LEN = 7
TOWER_PLACE_ATTEMPTS = 120
PATH_SEARCH_ATTEMPTS = 160
GEN_ATTEMPTS = 400


def _add(p: tuple[int, int], d: tuple[int, int], k: int = 1) -> tuple[int, int]:
    return int(p[0] + d[0] * k), int(p[1] + d[1] * k)


def _in_bounds(shape: tuple[int, int], p: tuple[int, int]) -> bool:
    return 0 <= int(p[0]) < int(shape[0]) and 0 <= int(p[1]) < int(shape[1])


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == int(bg):
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            out.append(
                {
                    "color": color,
                    "coords": coords,
                    "coord_set": set(coords),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )
    return out


def _component_mask(comp: dict[str, object]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in comp["coords"]:
        mask[int(r) - r0, int(c) - c0] = True
    return mask


def _detect_tower(comp: dict[str, object]) -> dict[str, object] | None:
    if int(comp["area"]) != 11:
        return None
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    mask = _component_mask(comp)

    patterns = [
        {
            "direction": UP,
            "mask": np.array(
                [
                    [0, 1, 0],
                    [0, 1, 0],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                ],
                dtype=bool,
            ),
            "tip": (r0, c0 + 1),
        },
        {
            "direction": DOWN,
            "mask": np.array(
                [
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                    [0, 1, 0],
                    [0, 1, 0],
                ],
                dtype=bool,
            ),
            "tip": (r0 + 4, c0 + 1),
        },
        {
            "direction": LEFT,
            "mask": np.array(
                [
                    [0, 0, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1],
                ],
                dtype=bool,
            ),
            "tip": (r0 + 1, c0),
        },
        {
            "direction": RIGHT,
            "mask": np.array(
                [
                    [1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 0, 0],
                ],
                dtype=bool,
            ),
            "tip": (r0 + 1, c0 + 4),
        },
    ]

    for spec in patterns:
        pmask = spec["mask"]
        if (h, w) != pmask.shape:
            continue
        if np.array_equal(mask, pmask):
            return {
                "kind": "tower",
                "color": int(comp["color"]),
                "cells": set(comp["coord_set"]),
                "tip": tuple(map(int, spec["tip"])),
                "direction": tuple(map(int, spec["direction"])),
            }
    return None


def _detect_prism(comp: dict[str, object]) -> dict[str, object] | None:
    if int(comp["area"]) != 3:
        return None
    r0, c0, r1, c1 = map(int, comp["bbox"])
    if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
        return None
    mask = _component_mask(comp)
    if int(mask.sum()) != 3:
        return None
    missing = np.argwhere(~mask)
    if missing.shape != (1, 2):
        return None
    mr, mc = map(int, missing[0])
    return {
        "kind": "prism",
        "color": int(comp["color"]),
        "cells": set(comp["coord_set"]),
        "missing_row": "up" if mr == 0 else "down",
        "missing_col": "left" if mc == 0 else "right",
    }


def _turn(direction: tuple[int, int], prism: dict[str, object]) -> tuple[int, int]:
    if int(direction[0]) == 0:
        return UP if prism["missing_row"] == "up" else DOWN
    return LEFT if prism["missing_col"] == "left" else RIGHT


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast a ray from each 3x3 tower along its 2-cell antenna, bend it at L-shaped prisms,
    and color each straight run with the tower color or the most recently hit prism color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _bg_color(g)

    towers: list[dict[str, object]] = []
    prisms: list[dict[str, object]] = []
    for comp in _find_components(g, bg):
        tower = _detect_tower(comp)
        if tower is not None:
            towers.append(tower)
            continue
        prism = _detect_prism(comp)
        if prism is not None:
            prisms.append(prism)
            continue
        raise AssertionError("found non-background component that is neither a tower nor a prism")

    assert towers, "expected at least one tower"

    tower_cells = {cell for tower in towers for cell in tower["cells"]}
    prism_by_cell: dict[tuple[int, int], dict[str, object]] = {}
    for prism in prisms:
        for cell in prism["cells"]:
            prism_by_cell[cell] = prism

    out = g.copy()
    h, w = g.shape
    ordered_towers = sorted(
        towers,
        key=lambda tower: (
            int(tower["tip"][0]),
            int(tower["tip"][1]),
            int(tower["direction"][0]),
            int(tower["direction"][1]),
            int(tower["color"]),
        ),
    )

    for tower in ordered_towers:
        pos = tuple(map(int, tower["tip"]))
        direction = tuple(map(int, tower["direction"]))
        paint_color = int(tower["color"])
        seen_states: set[tuple[tuple[int, int], tuple[int, int]]] = set()

        while True:
            state = (pos, direction)
            if state in seen_states:
                break
            seen_states.add(state)

            nxt = _add(pos, direction)
            if not _in_bounds((h, w), nxt):
                break
            if nxt in tower_cells:
                break

            prism = prism_by_cell.get(nxt)
            if prism is not None:
                direction = _turn(direction, prism)
                paint_color = int(prism["color"])
                continue

            assert int(g[nxt]) == int(bg), "rays may only travel through background before painting"
            out[nxt] = int(paint_color)
            pos = nxt

    return out


def _tower_geometry(square_top: tuple[int, int], direction: tuple[int, int]) -> dict[str, object] | None:
    r0, c0 = map(int, square_top)
    square = {(r0 + dr, c0 + dc) for dr in range(3) for dc in range(3)}
    if direction == UP:
        antenna = {(r0 - 1, c0 + 1), (r0 - 2, c0 + 1)}
        tip = (r0 - 2, c0 + 1)
    elif direction == DOWN:
        antenna = {(r0 + 3, c0 + 1), (r0 + 4, c0 + 1)}
        tip = (r0 + 4, c0 + 1)
    elif direction == LEFT:
        antenna = {(r0 + 1, c0 - 1), (r0 + 1, c0 - 2)}
        tip = (r0 + 1, c0 - 2)
    else:
        assert direction == RIGHT
        antenna = {(r0 + 1, c0 + 3), (r0 + 1, c0 + 4)}
        tip = (r0 + 1, c0 + 4)
    cells = square | antenna
    return {"cells": cells, "tip": tip, "direction": direction}


def _prism_candidates(
    pos: tuple[int, int],
    direction: tuple[int, int],
    turn_dir: tuple[int, int],
    shape: tuple[int, int],
) -> list[dict[str, object]]:
    contact = _add(pos, direction)
    out: list[dict[str, object]] = []

    if direction in (LEFT, RIGHT):
        allowed_missing = ((0, 0), (0, 1)) if turn_dir == UP else ((1, 0), (1, 1))
    else:
        allowed_missing = ((0, 0), (1, 0)) if turn_dir == LEFT else ((0, 1), (1, 1))

    for br in (contact[0] - 1, contact[0]):
        for bc in (contact[1] - 1, contact[1]):
            corners = [
                (int(br), int(bc)),
                (int(br), int(bc + 1)),
                (int(br + 1), int(bc)),
                (int(br + 1), int(bc + 1)),
            ]
            if not all(_in_bounds(shape, cell) for cell in corners):
                continue
            for mr, mc in allowed_missing:
                missing = (int(br + mr), int(bc + mc))
                cells = {cell for cell in corners if cell != missing}
                if contact not in cells:
                    continue
                if pos in cells or _add(pos, turn_dir) in cells:
                    continue
                out.append(
                    {
                        "cells": cells,
                        "missing_row": "up" if mr == 0 else "down",
                        "missing_col": "left" if mc == 0 else "right",
                    }
                )
    random.shuffle(out)
    return out


def _choose_tower_placement(
    shape: tuple[int, int],
    blocked_input: set[tuple[int, int]],
    blocked_drawn: set[tuple[int, int]],
) -> dict[str, object] | None:
    h, w = map(int, shape)
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for direction in DIRS:
        for r0 in range(h):
            for c0 in range(w):
                geom = _tower_geometry((r0, c0), direction)
                cells = geom["cells"]
                if not all(_in_bounds(shape, cell) for cell in cells):
                    continue
                if cells & blocked_input or cells & blocked_drawn:
                    continue
                if not _in_bounds(shape, _add(tuple(map(int, geom["tip"])), direction)):
                    continue
                candidates.append(((r0, c0), direction))
    if not candidates:
        return None
    square_top, direction = random.choice(candidates)
    geom = _tower_geometry(square_top, direction)
    assert geom is not None
    return {
        "square_top": square_top,
        "cells": set(geom["cells"]),
        "tip": tuple(map(int, geom["tip"])),
        "direction": tuple(map(int, direction)),
    }


def _extend_to_edge(
    pos: tuple[int, int],
    direction: tuple[int, int],
    shape: tuple[int, int],
    blocked_input: set[tuple[int, int]],
    blocked_drawn: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    cur = pos
    cells: list[tuple[int, int]] = []
    while True:
        nxt = _add(cur, direction)
        if not _in_bounds(shape, nxt):
            break
        if nxt in blocked_input or nxt in blocked_drawn:
            return None
        cells.append(nxt)
        cur = nxt
    if len(cells) < MIN_SEG_LEN:
        return None
    return cells


def _extend_to_tower(
    pos: tuple[int, int],
    direction: tuple[int, int],
    shape: tuple[int, int],
    blocked_input: set[tuple[int, int]],
    blocked_drawn: set[tuple[int, int]],
    tower_targets: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    cur = pos
    cells: list[tuple[int, int]] = []
    while True:
        nxt = _add(cur, direction)
        if not _in_bounds(shape, nxt):
            return None
        if nxt in blocked_drawn:
            return None
        if nxt in blocked_input:
            if nxt in tower_targets and len(cells) >= MIN_SEG_LEN:
                return cells
            return None
        cells.append(nxt)
        cur = nxt


def _search_path(
    pos: tuple[int, int],
    direction: tuple[int, int],
    bends_left: int,
    end_mode: str,
    shape: tuple[int, int],
    blocked_input: set[tuple[int, int]],
    blocked_drawn: set[tuple[int, int]],
    tower_targets: set[tuple[int, int]],
) -> dict[str, object] | None:
    if bends_left == 0:
        if end_mode == "edge":
            final_cells = _extend_to_edge(pos, direction, shape, blocked_input, blocked_drawn)
        else:
            final_cells = _extend_to_tower(pos, direction, shape, blocked_input, blocked_drawn, tower_targets)
        if final_cells is None:
            return None
        return {"path_cells": final_cells, "prisms": []}

    run: list[tuple[int, int]] = []
    cur = pos
    endpoints: list[tuple[int, list[tuple[int, int]]]] = []
    while len(run) < MAX_SEG_LEN:
        nxt = _add(cur, direction)
        if not _in_bounds(shape, nxt):
            break
        if nxt in blocked_input or nxt in blocked_drawn:
            break
        run.append(nxt)
        cur = nxt
        if len(run) >= MIN_SEG_LEN and _in_bounds(shape, _add(cur, direction)):
            endpoints.append((len(run), run.copy()))
    random.shuffle(endpoints)

    for _length, segment_cells in endpoints:
        pivot = segment_cells[-1]
        for turn_dir in random.sample(list(PERPENDICULAR[direction]), k=2):
            for prism in _prism_candidates(pivot, direction, turn_dir, shape):
                prism_cells = set(prism["cells"])
                next_step = _add(pivot, turn_dir)
                if not _in_bounds(shape, next_step):
                    continue
                if prism_cells & blocked_input or prism_cells & blocked_drawn:
                    continue
                if next_step in blocked_input or next_step in blocked_drawn or next_step in prism_cells:
                    continue
                new_input = blocked_input | prism_cells
                new_drawn = blocked_drawn | set(segment_cells)
                sub = _search_path(
                    pivot,
                    turn_dir,
                    bends_left - 1,
                    end_mode,
                    shape,
                    new_input,
                    new_drawn,
                    tower_targets,
                )
                if sub is None:
                    continue
                return {
                    "path_cells": segment_cells + list(sub["path_cells"]),
                    "prisms": [prism] + list(sub["prisms"]),
                }
    return None


def _plan_tower(
    shape: tuple[int, int],
    blocked_input: set[tuple[int, int]],
    blocked_drawn: set[tuple[int, int]],
    tower_targets: set[tuple[int, int]],
    end_mode: str,
    bends: int,
) -> dict[str, object] | None:
    for _ in range(TOWER_PLACE_ATTEMPTS):
        tower = _choose_tower_placement(shape, blocked_input, blocked_drawn)
        if tower is None:
            return None
        plan = _search_path(
            tuple(map(int, tower["tip"])),
            tuple(map(int, tower["direction"])),
            int(bends),
            end_mode,
            shape,
            blocked_input | set(tower["cells"]),
            blocked_drawn,
            tower_targets,
        )
        if plan is None:
            continue
        tower["path_cells"] = list(plan["path_cells"])
        tower["prisms"] = list(plan["prisms"])
        tower["end_mode"] = end_mode
        return tower
    return None


def _render_input(
    shape: tuple[int, int],
    towers: list[dict[str, object]],
    prisms: list[dict[str, object]],
) -> np.ndarray:
    object_count = len(towers) + len(prisms)
    assert object_count <= 9, "need distinct colors for all objects plus background"
    bg = random.randrange(10)
    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)
    tower_colors = palette[: len(towers)]
    prism_colors = palette[len(towers) : len(towers) + len(prisms)]

    grid = np.full(shape, bg, dtype=int)
    for tower, color in zip(towers, tower_colors):
        tower["color"] = int(color)
        for r, c in tower["cells"]:
            grid[int(r), int(c)] = int(color)
    for prism, color in zip(prisms, prism_colors):
        prism["color"] = int(color)
        for r, c in prism["cells"]:
            grid[int(r), int(c)] = int(color)
    return grid


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    shape = (h, w)

    tower_count = random.choices(TOWER_COUNT_CHOICES, weights=TOWER_COUNT_WEIGHTS, k=1)[0]
    bend_targets = [random.choices(BEND_COUNT_CHOICES, weights=BEND_COUNT_WEIGHTS, k=1)[0] for _ in range(tower_count)]
    if max(bend_targets) == 0 and random.random() < 0.7:
        bend_targets[random.randrange(tower_count)] = random.choice((1, 2))

    want_tower_stop = tower_count >= 2 and random.random() < 0.5

    towers: list[dict[str, object]] = []
    prisms: list[dict[str, object]] = []
    blocked_input: set[tuple[int, int]] = set()
    blocked_drawn: set[tuple[int, int]] = set()

    for i in range(tower_count):
        end_mode = "tower" if want_tower_stop and i == tower_count - 1 else "edge"
        tower_targets = {cell for tower in towers for cell in tower["cells"]}
        bends = bend_targets[i]
        tower = _plan_tower(shape, blocked_input, blocked_drawn, tower_targets, end_mode, bends)
        if tower is None:
            return None
        towers.append(tower)
        blocked_input |= set(tower["cells"])
        blocked_drawn |= set(tower["path_cells"])
        for prism in tower["prisms"]:
            prisms.append(prism)
            blocked_input |= set(prism["cells"])

    if len(prisms) > 6:
        return None

    grid = _render_input(shape, towers, prisms)
    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
