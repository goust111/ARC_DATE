from __future__ import annotations

import random
from typing import Iterable

import numpy as np


MIRROR_BASE = 5
RECEIVER_COLOR = 8

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)
TURN_OPTIONS = {
    UP: (LEFT, RIGHT),
    DOWN: (LEFT, RIGHT),
    LEFT: (UP, DOWN),
    RIGHT: (UP, DOWN),
}
REFLECT = {
    ("\\", RIGHT): DOWN,
    ("\\", DOWN): RIGHT,
    ("\\", LEFT): UP,
    ("\\", UP): LEFT,
    ("/", RIGHT): UP,
    ("/", UP): RIGHT,
    ("/", LEFT): DOWN,
    ("/", DOWN): LEFT,
}

PALETTE = [1, 2, 3, 4, 6, 7, 9]
GENERATE_ATTEMPTS = 400
PATH_ATTEMPTS = 120


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _is_edge(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _neighbors8(r: int, c: int) -> Iterable[tuple[int, int]]:
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _component8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors8(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _edge_cells(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    cells: list[tuple[int, int]] = []
    for c in range(w):
        cells.append((0, int(c)))
        cells.append((h - 1, int(c)))
    for r in range(1, h - 1):
        cells.append((int(r), 0))
        cells.append((int(r), w - 1))
    return cells


def _start_direction(shape: tuple[int, int], pos: tuple[int, int]) -> tuple[int, int]:
    r, c = pos
    h, w = shape
    if r == 0 and 0 < c < w - 1:
        return DOWN
    if r == h - 1 and 0 < c < w - 1:
        return UP
    if c == 0 and 0 < r < h - 1:
        return RIGHT
    if c == w - 1 and 0 < r < h - 1:
        return LEFT
    raise AssertionError("edge object must not be in a corner")


def _ray_to_edge(start: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    r, c = start
    dr, dc = direction
    cells: list[tuple[int, int]] = []
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            break
        cells.append((int(r), int(c)))
        if _is_edge(shape, int(r), int(c)):
            break
    return cells


def _cells_to_target(start: tuple[int, int], target: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = start
    r1, c1 = target
    if r0 == r1:
        step = 1 if c1 > c0 else -1
        return [(int(r0), int(c)) for c in range(c0 + step, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 > r0 else -1
        return [(int(r), int(c0)) for r in range(r0 + step, r1 + step, step)]
    raise AssertionError("segment must be axis-aligned")


def _orientation_from_cells(a: tuple[int, int], b: tuple[int, int]) -> str:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    assert abs(dr) == 1 and abs(dc) == 1, "mirror must occupy diagonal neighbors"
    return "\\" if dr == dc else "/"


def _mirror_other(
    hit: tuple[int, int], incoming: tuple[int, int], outgoing: tuple[int, int]
) -> tuple[str, tuple[int, int]]:
    table = {
        (RIGHT, DOWN): ("\\", (1, 1)),
        (DOWN, RIGHT): ("\\", (1, 1)),
        (LEFT, UP): ("\\", (-1, -1)),
        (UP, LEFT): ("\\", (-1, -1)),
        (RIGHT, UP): ("/", (1, -1)),
        (DOWN, LEFT): ("/", (1, -1)),
        (LEFT, DOWN): ("/", (-1, 1)),
        (UP, RIGHT): ("/", (-1, 1)),
    }
    orientation, (dr, dc) = table[(incoming, outgoing)]
    return orientation, (hit[0] + dr, hit[1] + dc)


def _detect_mirrors(grid: np.ndarray, bg: int) -> dict[tuple[int, int], dict]:
    h, w = grid.shape
    mask = np.zeros((h, w), dtype=bool)
    mask[1 : h - 1, 1 : w - 1] = grid[1 : h - 1, 1 : w - 1] != bg
    mirrors: dict[tuple[int, int], dict] = {}
    for comp in _component8(mask):
        assert len(comp) == 2, "every interior foreground object must be a 2-pixel mirror"
        (r0, c0), (r1, c1) = comp
        orientation = _orientation_from_cells((r0, c0), (r1, c1))
        colors = [int(grid[r0, c0]), int(grid[r1, c1])]
        assert colors.count(MIRROR_BASE) == 1, "mirror must contain exactly one visible base pixel"
        tint_cell = comp[0] if int(grid[r0, c0]) != MIRROR_BASE else comp[1]
        tint = int(grid[tint_cell])
        assert tint not in (bg, MIRROR_BASE, RECEIVER_COLOR), "mirror tint must be a visible non-role color"
        info = {
            "cells": {comp[0], comp[1]},
            "orientation": orientation,
            "tint": tint,
        }
        mirrors[comp[0]] = info
        mirrors[comp[1]] = info
    return mirrors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace colored rays through diagonal mirrors.

    Roles:
    - background = most common color
    - mirror base = fixed grey stand-in color 5
    - receiver = fixed special edge color 8
    - emitters = any other edge singletons

    Mirrors are 2-pixel diagonal interior objects. One of their two cells keeps
    the grey base and the other exposes the mirror's recolor dot.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"
    h, w = g.shape
    assert h >= 6 and w >= 6, "grid is too small"

    bg = _most_common_color(g)
    mirror_map = _detect_mirrors(g, bg)

    emitters: list[tuple[tuple[int, int], int, tuple[int, int]]] = []
    receivers: set[tuple[int, int]] = set()
    for r, c in _edge_cells(g.shape):
        color = int(g[r, c])
        if color == bg:
            continue
        if color == RECEIVER_COLOR:
            receivers.add((int(r), int(c)))
            continue
        assert color != MIRROR_BASE, "mirror base color may not appear on the edge"
        emitters.append(((int(r), int(c)), color, _start_direction(g.shape, (int(r), int(c)))))

    assert emitters, "expected at least one emitter"
    assert receivers, "expected at least one receiver"

    out = g.copy()
    ordered_emitters = sorted(emitters, key=lambda item: (item[0][0], item[0][1], item[1]))

    for start, ray_color, direction in ordered_emitters:
        pos = start
        current_color = int(ray_color)
        current_dir = direction
        seen_states: set[tuple[tuple[int, int], tuple[int, int], int]] = set()

        while True:
            state = (pos, current_dir, current_color)
            assert state not in seen_states, "ray loop detected"
            seen_states.add(state)

            r, c = pos
            dr, dc = current_dir
            nr, nc = r + dr, c + dc
            if not _in_bounds(g.shape, nr, nc):
                break

            nxt = (int(nr), int(nc))
            mirror = mirror_map.get(nxt)
            if mirror is not None:
                pos = nxt
                current_dir = REFLECT[(mirror["orientation"], current_dir)]
                current_color = int(mirror["tint"])
                continue

            cell = int(g[nr, nc])
            if nxt in receivers:
                out[nr, nc] = current_color
                break

            assert cell == bg, "rays may only traverse background before reaching the edge"
            out[nr, nc] = current_color
            pos = nxt
            if _is_edge(g.shape, nr, nc):
                break

    return out


def _segment_conflicts(
    cells: list[tuple[int, int]],
    occupied_drawn: set[tuple[int, int]],
    occupied_static: set[tuple[int, int]],
) -> bool:
    return any(cell in occupied_drawn or cell in occupied_static for cell in cells)


def _mirror_touches_existing(
    pair: set[tuple[int, int]], existing_mirror_cells: set[tuple[int, int]]
) -> bool:
    for r, c in pair:
        for nr, nc in _neighbors8(r, c):
            if (nr, nc) in existing_mirror_cells and (nr, nc) not in pair:
                return True
    return False


def _extend_path(
    shape: tuple[int, int],
    current: tuple[int, int],
    direction: tuple[int, int],
    turns_left: int,
    want_receiver: bool,
    occupied_drawn: set[tuple[int, int]],
    occupied_static: set[tuple[int, int]],
    existing_mirror_cells: set[tuple[int, int]],
    used_edge_cells: set[tuple[int, int]],
    local_drawn: set[tuple[int, int]],
    local_static: set[tuple[int, int]],
    mirrors: list[dict],
) -> dict | None:
    ray = _ray_to_edge(current, direction, shape)
    if not ray:
        return None

    if turns_left == 0:
        endpoint = ray[-1]
        if endpoint in used_edge_cells or endpoint in occupied_static or endpoint in local_static:
            return None
        segment = ray[:-1] if want_receiver else ray
        if _segment_conflicts(segment, occupied_drawn | local_drawn, occupied_static | local_static):
            return None
        static_out = set(local_static)
        drawn_out = set(local_drawn)
        if want_receiver:
            static_out.add(endpoint)
            drawn_out.update(ray[:-1])
        else:
            drawn_out.update(ray)
        return {
            "drawn": drawn_out,
            "static": static_out,
            "mirrors": list(mirrors),
            "end": endpoint,
            "receiver": bool(want_receiver),
        }

    hit_candidates = ray[:-1]
    if len(hit_candidates) < 2:
        return None
    indices = list(range(1, len(hit_candidates)))
    random.shuffle(indices)

    for idx in indices:
        hit = hit_candidates[idx]
        segment = hit_candidates[:idx]
        if _segment_conflicts(segment, occupied_drawn | local_drawn, occupied_static | local_static):
            continue

        out_dirs = list(TURN_OPTIONS[direction])
        random.shuffle(out_dirs)
        for out_dir in out_dirs:
            orientation, other = _mirror_other(hit, direction, out_dir)
            if not _in_bounds(shape, other[0], other[1]) or _is_edge(shape, other[0], other[1]):
                continue

            pair = {hit, other}
            if pair & (occupied_drawn | local_drawn):
                continue
            if pair & (occupied_static | local_static):
                continue
            if _mirror_touches_existing(pair, existing_mirror_cells | {cell for m in mirrors for cell in m["cells"]}):
                continue

            tint_cell = random.choice([hit, other])
            next_mirrors = list(mirrors)
            next_mirrors.append(
                {
                    "cells": pair,
                    "hit": hit,
                    "other": other,
                    "orientation": orientation,
                    "tint_cell": tint_cell,
                }
            )
            result = _extend_path(
                shape,
                hit,
                out_dir,
                turns_left - 1,
                want_receiver,
                occupied_drawn,
                occupied_static,
                existing_mirror_cells,
                used_edge_cells,
                local_drawn | set(segment),
                local_static | pair,
                next_mirrors,
            )
            if result is not None:
                return result
    return None


def _choose_start(
    shape: tuple[int, int], used_edge_cells: set[tuple[int, int]]
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = shape
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for c in range(1, w - 1):
        candidates.append(((0, c), DOWN))
        candidates.append(((h - 1, c), UP))
    for r in range(1, h - 1):
        candidates.append(((r, 0), RIGHT))
        candidates.append(((r, w - 1), LEFT))
    random.shuffle(candidates)
    for pos, direction in candidates:
        if pos not in used_edge_cells:
            return pos, direction
    return None


def _plan_path(
    shape: tuple[int, int],
    turns: int,
    want_receiver: bool,
    occupied_drawn: set[tuple[int, int]],
    occupied_static: set[tuple[int, int]],
    existing_mirror_cells: set[tuple[int, int]],
    used_edge_cells: set[tuple[int, int]],
) -> dict | None:
    for _ in range(PATH_ATTEMPTS):
        start_choice = _choose_start(shape, used_edge_cells)
        if start_choice is None:
            return None
        start, direction = start_choice
        result = _extend_path(
            shape,
            start,
            direction,
            turns,
            want_receiver,
            occupied_drawn,
            occupied_static,
            existing_mirror_cells,
            used_edge_cells,
            local_drawn=set(),
            local_static={start},
            mirrors=[],
        )
        if result is None:
            continue
        result["start"] = start
        result["start_dir"] = direction
        return result
    return None


def _assign_colors(paths: list[dict]) -> None:
    remaining_emitters = PALETTE.copy()
    random.shuffle(remaining_emitters)
    for path in paths:
        if not remaining_emitters:
            remaining_emitters = PALETTE.copy()
            random.shuffle(remaining_emitters)
        emitter_color = remaining_emitters.pop()
        path["emitter_color"] = int(emitter_color)
        current = int(emitter_color)
        for mirror in path["mirrors"]:
            choices = [c for c in PALETTE if c != current]
            mirror["tint"] = int(random.choice(choices))
            current = int(mirror["tint"])


def _render_input(shape: tuple[int, int], paths: list[dict], extra_receivers: list[tuple[int, int]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for path in paths:
        sr, sc = path["start"]
        grid[sr, sc] = int(path["emitter_color"])
        for mirror in path["mirrors"]:
            a, b = tuple(mirror["cells"])
            tint_cell = mirror["tint_cell"]
            base_cell = b if tint_cell == a else a
            grid[base_cell] = MIRROR_BASE
            grid[tint_cell] = int(mirror["tint"])
        if path["receiver"]:
            er, ec = path["end"]
            grid[er, ec] = RECEIVER_COLOR
    for r, c in extra_receivers:
        grid[r, c] = RECEIVER_COLOR
    return grid


def _render_expected(inp: np.ndarray, paths: list[dict]) -> np.ndarray:
    out = np.array(inp, copy=True)
    for path in paths:
        color = int(path["emitter_color"])
        pos = path["start"]
        for mirror in path["mirrors"]:
            for r, c in _cells_to_target(pos, mirror["hit"])[:-1]:
                out[r, c] = color
            pos = mirror["hit"]
            color = int(mirror["tint"])
        trail = _cells_to_target(pos, path["end"])
        if path["receiver"]:
            for r, c in trail[:-1]:
                out[r, c] = color
            er, ec = path["end"]
            out[er, ec] = color
        else:
            for r, c in trail:
                out[r, c] = color
    return out


def _free_edge_cells(shape: tuple[int, int], used: set[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = shape
    cells = []
    for c in range(1, w - 1):
        if (0, c) not in used:
            cells.append((0, int(c)))
        if (h - 1, c) not in used:
            cells.append((h - 1, int(c)))
    for r in range(1, h - 1):
        if (r, 0) not in used:
            cells.append((int(r), 0))
        if (r, w - 1) not in used:
            cells.append((int(r), w - 1))
    return cells


def _build_scene() -> np.ndarray | None:
    h = random.randint(12, 22)
    w = random.randint(12, 22)
    shape = (h, w)

    occupied_drawn: set[tuple[int, int]] = set()
    occupied_static: set[tuple[int, int]] = set()
    mirror_cells: set[tuple[int, int]] = set()
    used_edge_cells: set[tuple[int, int]] = set()
    paths: list[dict] = []

    specs: list[tuple[int, bool]] = []
    primary_turns = random.randint(1, 3)
    specs.append((primary_turns, True))
    extra_count = random.choice([0, 1, 1, 2])
    for _ in range(extra_count):
        specs.append((random.randint(0, 3), bool(random.getrandbits(1))))
    if len(specs) >= 2 and all(want for _turns, want in specs):
        idx = random.randrange(1, len(specs))
        specs[idx] = (specs[idx][0], False)
    if len(specs) > 2:
        tail = specs[1:]
        random.shuffle(tail)
        specs[1:] = tail

    for turns, want_receiver in specs:
        plan = _plan_path(
            shape,
            turns,
            want_receiver,
            occupied_drawn,
            occupied_static,
            mirror_cells,
            used_edge_cells,
        )
        if plan is None:
            return None
        paths.append(plan)
        occupied_drawn.update(plan["drawn"])
        occupied_static.update(plan["static"])
        used_edge_cells.add(plan["start"])
        used_edge_cells.add(plan["end"])
        for mirror in plan["mirrors"]:
            mirror_cells.update(mirror["cells"])

    _assign_colors(paths)

    free_edges = _free_edge_cells(shape, used_edge_cells)
    random.shuffle(free_edges)
    extra_receiver_count = random.randint(0, min(2, len(free_edges)))
    extra_receivers = free_edges[:extra_receiver_count]
    inp = _render_input(shape, paths, extra_receivers)

    if _most_common_color(inp) != 0:
        return None

    if int(np.count_nonzero(inp == RECEIVER_COLOR)) == 0:
        return None

    expected = _render_expected(inp, paths)
    actual = main(inp)
    if not np.array_equal(expected, actual):
        return None

    if np.array_equal(inp, actual):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        grid = _build_scene()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-ray puzzle")
