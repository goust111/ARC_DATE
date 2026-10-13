from __future__ import annotations

from typing import Iterable

import numpy as np


BG = 0
MIRROR = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
RING_OFFSETS = [
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
]

UP = (-1, 0)
RIGHT = (0, 1)
DOWN = (1, 0)
LEFT = (0, -1)
DIRS = [UP, RIGHT, DOWN, LEFT]
ROT_CW = {UP: RIGHT, RIGHT: DOWN, DOWN: LEFT, LEFT: UP}

GRID_MIN = 24
GRID_MAX = 34
SEED_COUNT_CHOICES = [2, 3]
MIRROR_COUNT_CHOICES = [1, 2, 3, 4]
MAX_GENERATION_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = int(cr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(sorted(comp))
    return comps


def _line_orientation(comp: Iterable[tuple[int, int]]) -> str | None:
    pts = list(comp)
    rows = {r for r, _ in pts}
    cols = {c for _, c in pts}
    if len(pts) < 2 or len(pts) > 4:
        return None
    if len(rows) == 1:
        ordered = sorted(c for _, c in pts)
        if ordered == list(range(ordered[0], ordered[0] + len(ordered))):
            return "h"
    if len(cols) == 1:
        ordered = sorted(r for r, _ in pts)
        if ordered == list(range(ordered[0], ordered[0] + len(ordered))):
            return "v"
    return None


def _detect_mirror_color(grid: np.ndarray, bg: int) -> tuple[int, dict[tuple[int, int], str]]:
    candidates: list[tuple[int, int, dict[tuple[int, int], str]]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        comps = _components_of_color(grid, color)
        if not comps:
            continue
        cell_to_axis: dict[tuple[int, int], str] = {}
        ok = True
        for comp in comps:
            axis = _line_orientation(comp)
            if axis is None:
                ok = False
                break
            for cell in comp:
                cell_to_axis[cell] = axis
        if ok:
            candidates.append((sum(len(comp) for comp in comps), int(color), cell_to_axis))
    assert len(candidates) == 1, "expected exactly one structurally valid mirror color"
    _size, mirror_color, mapping = candidates[0]
    return int(mirror_color), mapping


def _find_seeds(grid: np.ndarray, bg: int, mirror_color: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    seeds: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            center = int(grid[r, c])
            if center in (bg, mirror_color):
                continue
            ok = True
            for dr, dc in RING_OFFSETS:
                val = int(grid[r + dr, c + dc])
                if val in (bg, mirror_color):
                    ok = False
                    break
            if ok:
                seeds.append((int(r), int(c)))
    return sorted(seeds)


def _read_ring_sequence(grid: np.ndarray, seed: tuple[int, int]) -> list[int]:
    r, c = seed
    return [int(grid[r + dr, c + dc]) for dr, dc in RING_OFFSETS]


def _find_initial_direction(
    grid: np.ndarray,
    seed: tuple[int, int],
    bg: int,
    mirror_color: int,
) -> tuple[int, int]:
    h, w = grid.shape
    sr, sc = seed
    candidates: list[tuple[int, tuple[int, int]]] = []
    for dr, dc in DIRS:
        step = 1
        valid = True
        while True:
            rr, cc = int(sr + dr * step), int(sc + dc * step)
            if not (0 <= rr < h and 0 <= cc < w):
                break
            val = int(grid[rr, cc])
            if val == mirror_color:
                if step >= 3:
                    candidates.append((step, (int(dr), int(dc))))
                break
            if step == 1:
                if val in (bg, mirror_color):
                    valid = False
                    break
            else:
                if val != bg:
                    valid = False
                    break
            step += 1
        if not valid:
            continue
    assert candidates, "seed must align with a mirror"
    candidates.sort(key=lambda item: item[0])
    best_dist = candidates[0][0]
    best_dirs = [d for dist, d in candidates if dist == best_dist]
    assert len(best_dirs) == 1, "nearest mirror direction must be unique"
    return best_dirs[0]


def _trace_seed_path(
    grid: np.ndarray,
    out: np.ndarray,
    seed: tuple[int, int],
    ring_seq: list[int],
    direction: tuple[int, int],
    bg: int,
    mirror_color: int,
    mirror_axes: dict[tuple[int, int], str],
) -> None:
    h, w = grid.shape
    r, c = seed
    dr, dc = direction
    seq_idx = 0
    seen_states: set[tuple[int, int, int, int]] = set()

    for _ in range(h * w * 8):
        state = (int(r), int(c), int(dr), int(dc))
        assert state not in seen_states, "ray loop detected"
        seen_states.add(state)

        nr, nc = int(r + dr), int(c + dc)
        if not (0 <= nr < h and 0 <= nc < w):
            return

        val = int(grid[nr, nc])
        if val == mirror_color:
            axis = mirror_axes[(int(nr), int(nc))]
            if axis == "h":
                assert dr == 0, "generated samples hit horizontal mirrors horizontally"
            else:
                assert dc == 0, "generated samples hit vertical mirrors vertically"
            r, c = int(nr), int(nc)
            dr, dc = ROT_CW[(int(dr), int(dc))]
            continue

        if val == bg:
            out[nr, nc] = int(ring_seq[seq_idx % len(ring_seq)])
            seq_idx += 1

        r, c = int(nr), int(nc)

    raise AssertionError("ray did not terminate before step budget")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace each seed's ray through grey mirror segments and color only background cells.

    The seed's 8-neighborhood, read from the cell above and then clockwise, is the
    cyclic color sequence written along the ray. In the generated task family, each
    mirror bounce turns the ray 90 degrees clockwise.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    mirror_color, mirror_axes = _detect_mirror_color(g, bg)
    seeds = _find_seeds(g, bg, mirror_color)
    assert 2 <= len(seeds) <= 3, "expected 2-3 seeds"

    out = g.copy()
    for seed in seeds:
        ring_seq = _read_ring_sequence(g, seed)
        direction = _find_initial_direction(g, seed, bg, mirror_color)
        _trace_seed_path(g, out, seed, ring_seq, direction, bg, mirror_color, mirror_axes)
    return out


def _axis_of(direction: tuple[int, int]) -> str:
    return "h" if direction[0] == 0 else "v"


def _add(p: tuple[int, int], q: tuple[int, int]) -> tuple[int, int]:
    return int(p[0] + q[0]), int(p[1] + q[1])


def _scale(direction: tuple[int, int], length: int) -> tuple[int, int]:
    return int(direction[0] * length), int(direction[1] * length)


def _cells_between(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    dr = int(np.sign(b[0] - a[0]))
    dc = int(np.sign(b[1] - a[1]))
    steps = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
    return [(int(a[0] + dr * i), int(a[1] + dc * i)) for i in range(1, steps + 1)]


def _sample_ring_sequence(rng: np.random.Generator, forbidden: set[int]) -> tuple[int, list[int]]:
    colors = [c for c in COLOR_POOL if c not in forbidden]
    ring_size = int(rng.integers(2, 4))
    ring_palette = list(rng.choice(colors, size=ring_size, replace=False))
    rng.shuffle(ring_palette)
    motif = ring_palette[:ring_size]
    motif_len = len(motif)
    ring_seq = [int(motif[i % motif_len]) for i in range(8)]
    seed_choices = [c for c in colors if c not in set(ring_seq)]
    seed_color = int(rng.choice(seed_choices))
    return seed_color, ring_seq


def _generate_pattern(rng: np.random.Generator) -> dict | None:
    seed = (0, 0)
    initial_dir = DIRS[int(rng.integers(0, 4))]
    mirror_count = int(rng.choice(MIRROR_COUNT_CHOICES))
    directions = [initial_dir]
    for _ in range(mirror_count):
        directions.append(ROT_CW[directions[-1]])

    lengths: list[int] = []
    prev_by_axis: dict[str, int] = {}
    axis_occurrences: dict[str, int] = {"h": 0, "v": 0}
    for idx, direction in enumerate(directions):
        axis = _axis_of(direction)
        occ = axis_occurrences[axis]
        if occ == 0:
            base_min = 3 if idx == 0 else 2
            base_max = 7 if idx == mirror_count else 6
            length = int(rng.integers(base_min, base_max + 1))
        else:
            growth = int(rng.integers(3, 6))
            length = int(prev_by_axis[axis] + growth)
        prev_by_axis[axis] = length
        axis_occurrences[axis] += 1
        lengths.append(length)

    contacts: list[tuple[int, int]] = []
    pos = seed
    for idx in range(mirror_count):
        pos = _add(pos, _scale(directions[idx], lengths[idx]))
        contacts.append(pos)
    tail_end = _add(pos, _scale(directions[mirror_count], lengths[mirror_count]))

    ring_cells = {(dr, dc) for dr, dc in RING_OFFSETS}
    mirror_cells: list[set[tuple[int, int]]] = []
    occupied = set(ring_cells)
    occupied.add(seed)
    for idx, contact in enumerate(contacts):
        length = int(rng.integers(2, 5))
        dr, dc = directions[idx]
        cells = {
            (int(contact[0] + dr * step), int(contact[1] + dc * step))
            for step in range(length)
        }
        if occupied & cells:
            return None
        mirror_cells.append(cells)
        occupied |= cells

    path_cells_ordered: list[tuple[int, int]] = []
    pos = seed
    for idx in range(mirror_count):
        nxt = contacts[idx]
        leg_cells = _cells_between(pos, nxt)
        for cell in leg_cells[:-1]:
            if cell in occupied and cell not in ring_cells:
                return None
            path_cells_ordered.append(cell)
        path_cells_ordered.append(nxt)
        pos = nxt
    tail_cells = _cells_between(pos, tail_end)
    for cell in tail_cells:
        if cell in occupied:
            return None
        path_cells_ordered.append(cell)

    if len(set(path_cells_ordered)) != len(path_cells_ordered):
        return None

    black_cells = [
        cell for cell in path_cells_ordered if cell not in occupied and cell not in ring_cells and cell != seed
    ]
    if len(black_cells) < 8:
        return None

    used_colors = {MIRROR}
    seed_color, ring_seq = _sample_ring_sequence(rng, used_colors)
    input_cells = {seed: seed_color}
    for (dr, dc), color in zip(RING_OFFSETS, ring_seq):
        input_cells[(int(seed[0] + dr), int(seed[1] + dc))] = int(color)
    for cells in mirror_cells:
        for cell in cells:
            input_cells[cell] = MIRROR

    all_coords = set(input_cells) | set(path_cells_ordered)
    rows = [r for r, _ in all_coords]
    cols = [c for _, c in all_coords]
    exit_dir = directions[mirror_count]

    return {
        "input_cells": input_cells,
        "path_cells": set(path_cells_ordered),
        "all_coords": all_coords,
        "min_r": min(rows),
        "max_r": max(rows),
        "min_c": min(cols),
        "max_c": max(cols),
        "tail_end": tail_end,
        "exit_dir": exit_dir,
    }


def _translate_pattern(
    pattern: dict,
    h: int,
    w: int,
    rng: np.random.Generator,
    blocked_input: set[tuple[int, int]],
    blocked_path: set[tuple[int, int]],
) -> dict | None:
    min_r = int(pattern["min_r"])
    max_r = int(pattern["max_r"])
    min_c = int(pattern["min_c"])
    max_c = int(pattern["max_c"])
    tail_r, tail_c = pattern["tail_end"]
    exit_dir = pattern["exit_dir"]

    r_low = -min_r
    r_high = h - 1 - max_r
    c_low = -min_c
    c_high = w - 1 - max_c
    if r_low > r_high or c_low > c_high:
        return None

    r_candidates: list[int]
    c_candidates: list[int]
    if exit_dir == UP:
        fixed_r = -tail_r
        if not (r_low <= fixed_r <= r_high):
            return None
        r_candidates = [int(fixed_r)]
        c_candidates = list(range(int(c_low), int(c_high) + 1))
    elif exit_dir == DOWN:
        fixed_r = h - 1 - tail_r
        if not (r_low <= fixed_r <= r_high):
            return None
        r_candidates = [int(fixed_r)]
        c_candidates = list(range(int(c_low), int(c_high) + 1))
    elif exit_dir == LEFT:
        fixed_c = -tail_c
        if not (c_low <= fixed_c <= c_high):
            return None
        r_candidates = list(range(int(r_low), int(r_high) + 1))
        c_candidates = [int(fixed_c)]
    else:
        fixed_c = w - 1 - tail_c
        if not (c_low <= fixed_c <= c_high):
            return None
        r_candidates = list(range(int(r_low), int(r_high) + 1))
        c_candidates = [int(fixed_c)]

    offsets = [(rr, cc) for rr in r_candidates for cc in c_candidates]
    rng.shuffle(offsets)

    for roff, coff in offsets:
        translated_input = {(r + roff, c + coff): color for (r, c), color in pattern["input_cells"].items()}
        translated_path = {(r + roff, c + coff) for (r, c) in pattern["path_cells"]}
        translated_all = {(r + roff, c + coff) for (r, c) in pattern["all_coords"]}

        if translated_input.keys() & blocked_input:
            continue
        if set(translated_input) & blocked_path:
            continue
        if translated_path & blocked_path:
            continue
        if translated_path & blocked_input:
            continue
        if any(not (0 <= r < h and 0 <= c < w) for r, c in translated_all):
            continue

        halo_conflict = False
        for r, c in translated_input:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if (r + dr, c + dc) in blocked_input:
                        halo_conflict = True
                        break
                if halo_conflict:
                    break
            if halo_conflict:
                break
        if halo_conflict:
            continue

        return {
            "input_cells": translated_input,
            "path_cells": translated_path,
        }
    return None


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    w = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    grid = np.full((h, w), BG, dtype=int)

    pattern_count = int(rng.choice(SEED_COUNT_CHOICES))
    blocked_input: set[tuple[int, int]] = set()
    blocked_path: set[tuple[int, int]] = set()

    for _ in range(pattern_count):
        placed = None
        for _ in range(100):
            pattern = _generate_pattern(rng)
            if pattern is None:
                continue
            placed = _translate_pattern(pattern, h, w, rng, blocked_input, blocked_path)
            if placed is not None:
                break
        if placed is None:
            return None
        for (r, c), color in placed["input_cells"].items():
            grid[r, c] = int(color)
        blocked_input |= set(placed["input_cells"])
        blocked_path |= set(placed["path_cells"])

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is None:
            continue
        return grid
    raise RuntimeError("failed to generate a valid mirror-ray sample")
