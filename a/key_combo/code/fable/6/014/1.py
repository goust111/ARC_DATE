from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np


_HELPER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_HELPER_DIR) not in sys.path:
    sys.path.append(str(_HELPER_DIR))

from arc_utils import Color, find_components, get_bg, in_bounds


GRID_MIN = 12
GRID_MAX = 24
GEN_TRIES = 500
LENGTH_POOL = (2, 3, 4, 5, 6, 7)
BAR_COLORS = [int(Color.RED), int(Color.GREEN), int(Color.YELLOW), int(Color.GREY), int(Color.PINK), int(Color.ORANGE), int(Color.TEAL), int(Color.MAROON)]


def _coord_tuples(coords: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in coords]


def _is_diagonal_domino(coords: list[tuple[int, int]]) -> bool:
    if len(coords) != 2:
        return False
    (r0, c0), (r1, c1) = sorted(coords)
    return abs(r0 - r1) == 1 and abs(c0 - c1) == 1


def _find_seed(grid: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int]]]:
    comps = find_components(
        grid,
        bg=bg,
        connectivity=8,
        monochromatic=True,
        colors=[int(Color.BLUE)],
        return_format="dict",
    )
    seeds = []
    for comp in comps:
        coords = _coord_tuples(comp["coords"])
        if _is_diagonal_domino(coords):
            seeds.append((int(comp["color"]), coords))
    assert len(seeds) == 1, "expected exactly one 2-cell diagonal blue seed"
    return seeds[0]


def _initial_state(shape: tuple[int, int], seed: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    if len({r - c for r, c in seed}) == 1:
        endpoints = ((min(seed), (-1, -1)), (max(seed), (1, 1)))
    else:
        top = min(seed)
        bottom = max(seed)
        endpoints = ((top, (-1, 1)), (bottom, (1, -1)))

    candidates = []
    for (r, c), (dr, dc) in endpoints:
        if in_bounds(shape, int(r + dr), int(c + dc)):
            candidates.append((int(r), int(c), int(dr), int(dc)))

    assert len(candidates) == 1, "seed must have exactly one in-bounds outward extension"
    return candidates[0]


def _parse_bars(grid: np.ndarray, bg: int, seed: list[tuple[int, int]]) -> tuple[list[dict], dict[tuple[int, int], dict]]:
    work = np.asarray(grid, dtype=int).copy()
    for r, c in seed:
        work[int(r), int(c)] = int(bg)

    comps = find_components(work, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    bars: list[dict] = []
    for comp in comps:
        coords = _coord_tuples(comp["coords"])
        if len(coords) < 2:
            continue

        rows = sorted({r for r, _ in coords})
        cols = sorted({c for _, c in coords})
        if len(rows) == 1:
            cols_sorted = sorted(c for _, c in coords)
            assert cols_sorted == list(range(cols_sorted[0], cols_sorted[-1] + 1)), "horizontal bar must be solid"
            orientation = "h"
        elif len(cols) == 1:
            rows_sorted = sorted(r for r, _ in coords)
            assert rows_sorted == list(range(rows_sorted[0], rows_sorted[-1] + 1)), "vertical bar must be solid"
            orientation = "v"
        else:
            raise AssertionError("foreground bars must be horizontal or vertical")

        bars.append(
            {
                "color": int(comp["color"]),
                "coords": coords,
                "orientation": orientation,
                "length": int(len(coords)),
            }
        )

    length_freq = Counter(int(bar["length"]) for bar in bars)
    lookup: dict[tuple[int, int], dict] = {}
    for bar in bars:
        bar["reflective"] = int(length_freq[int(bar["length"])]) >= 2
        for cell in bar["coords"]:
            lookup[cell] = bar
    return bars, lookup


def _trace_ray(grid: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    ray_color, seed = _find_seed(g, bg)
    _bars, lookup = _parse_bars(g, bg, seed)
    r, c, dr, dc = _initial_state(g.shape, seed)

    out = g.copy()
    seen: set[tuple[int, int, int, int]] = set()
    reflective_hits = 0
    transparent_hits = 0
    while True:
        state = (int(r), int(c), int(dr), int(dc))
        assert state not in seen, "ray path should not loop"
        seen.add(state)

        nr, nc = int(r + dr), int(c + dc)
        if not in_bounds(g.shape, nr, nc):
            break

        bar = lookup.get((nr, nc))
        if bar is not None and bool(bar["reflective"]):
            reflective_hits += 1
            if bar["orientation"] == "h":
                dr = -int(dr)
            else:
                dc = -int(dc)
            continue

        if bar is not None:
            transparent_hits += 1

        out[nr, nc] = int(ray_color)
        r, c = nr, nc

    return out, {"reflective_hits": int(reflective_hits), "transparent_hits": int(transparent_hits)}


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Extend the blue diagonal ray. Bars whose length appears at least twice anywhere
    act as mirrors; unique-length bars are transparent and get overwritten.
    """
    out, _stats = _trace_ray(np.asarray(input_grid, dtype=int))
    return out


def _reflect(orientation: str, dr: int, dc: int) -> tuple[int, int]:
    return (-int(dr), int(dc)) if orientation == "h" else (int(dr), -int(dc))


def _steps_to_exit(shape: tuple[int, int], r: int, c: int, dr: int, dc: int) -> int:
    rr, cc = int(r), int(c)
    steps = 0
    while in_bounds(shape, int(rr + dr), int(cc + dc)):
        rr += int(dr)
        cc += int(dc)
        steps += 1
    return int(steps)


def _random_start(shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = map(int, shape)
    for _ in range(100):
        edge = random.choice(("top", "bottom", "left", "right"))
        if edge == "top":
            r0, c0, dr, dc = 0, random.randrange(w), 1, random.choice((-1, 1))
        elif edge == "bottom":
            r0, c0, dr, dc = h - 1, random.randrange(w), -1, random.choice((-1, 1))
        elif edge == "left":
            r0, c0, dr, dc = random.randrange(h), 0, random.choice((-1, 1)), 1
        else:
            r0, c0, dr, dc = random.randrange(h), w - 1, random.choice((-1, 1)), -1

        if in_bounds(shape, int(r0 + dr), int(c0 + dc)):
            return int(r0), int(c0), int(dr), int(dc)
    raise RuntimeError("failed to sample a valid seed start")


def _segment_cells(anchor: tuple[int, int], orientation: str, length: int, offset: int) -> list[tuple[int, int]]:
    r, c = map(int, anchor)
    cells = []
    if orientation == "h":
        start_c = int(c - offset)
        for cc in range(start_c, start_c + int(length)):
            cells.append((int(r), int(cc)))
    else:
        start_r = int(r - offset)
        for rr in range(start_r, start_r + int(length)):
            cells.append((int(rr), int(c)))
    return cells


def _anchored_options(
    shape: tuple[int, int],
    anchor: tuple[int, int],
    orientation: str,
    length: int,
) -> list[list[tuple[int, int]]]:
    opts = []
    for offset in range(int(length)):
        cells = _segment_cells(anchor, orientation, length, offset)
        if all(in_bounds(shape, r, c) for r, c in cells):
            opts.append(cells)
    random.shuffle(opts)
    return opts


def _free_segment_options(shape: tuple[int, int], orientation: str, length: int) -> list[list[tuple[int, int]]]:
    h, w = map(int, shape)
    opts = []
    if orientation == "h":
        for r in range(h):
            for c0 in range(w - int(length) + 1):
                opts.append([(int(r), int(c0 + k)) for k in range(int(length))])
    else:
        for r0 in range(h - int(length) + 1):
            for c in range(w):
                opts.append([(int(r0 + k), int(c)) for k in range(int(length))])
    random.shuffle(opts)
    return opts


def _cells_fit(
    cells: list[tuple[int, int]],
    blocked: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    allowed_overlap: set[tuple[int, int]] | None = None,
) -> bool:
    allowed = set() if allowed_overlap is None else set(allowed_overlap)
    for cell in cells:
        if cell in occupied:
            return False
        if cell in blocked and cell not in allowed:
            return False
    return True


def _plan_ray(shape: tuple[int, int]) -> dict | None:
    start_r, start_c, dr, dc = _random_start(shape)
    seed_cells = [(int(start_r), int(start_c)), (int(start_r + dr), int(start_c + dc))]

    current_r = int(seed_cells[-1][0])
    current_c = int(seed_cells[-1][1])
    occupied = set(seed_cells)
    path_cells: list[tuple[int, int]] = []
    mirrors: list[dict] = []

    mirror_target = random.randint(1, 2)
    for mirror_index in range(mirror_target):
        max_steps = _steps_to_exit(shape, current_r, current_c, dr, dc)
        if max_steps < 2:
            return None

        seg_lengths = list(range(1, int(max_steps)))
        random.shuffle(seg_lengths)
        choice = None
        for seg_len in seg_lengths:
            segment = []
            for step in range(1, int(seg_len) + 1):
                segment.append((int(current_r + dr * step), int(current_c + dc * step)))
            mirror_cell = (int(current_r + dr * (seg_len + 1)), int(current_c + dc * (seg_len + 1)))
            if any(cell in occupied for cell in segment) or mirror_cell in occupied:
                continue

            orient_choices = ["h", "v"]
            random.shuffle(orient_choices)
            for orientation in orient_choices:
                next_dr, next_dc = _reflect(orientation, dr, dc)
                steps_after = _steps_to_exit(shape, segment[-1][0], segment[-1][1], next_dr, next_dc)
                min_needed = 2 if mirror_index + 1 < mirror_target else 1
                if steps_after < min_needed:
                    continue
                choice = (segment, mirror_cell, orientation, next_dr, next_dc)
                break
            if choice is not None:
                break

        if choice is None:
            return None

        segment, mirror_cell, orientation, dr, dc = choice
        path_cells.extend(segment)
        occupied.update(segment)
        occupied.add(mirror_cell)
        current_r, current_c = map(int, segment[-1])
        mirrors.append({"hit": mirror_cell, "orientation": orientation})

    tail_len = _steps_to_exit(shape, current_r, current_c, dr, dc)
    if tail_len < 1:
        return None
    tail = []
    for step in range(1, int(tail_len) + 1):
        tail.append((int(current_r + dr * step), int(current_c + dc * step)))
    if any(cell in occupied for cell in tail):
        return None
    path_cells.extend(tail)

    if len(path_cells) < 4:
        return None
    return {"seed": seed_cells, "path": path_cells, "mirrors": mirrors}


def _paint_bar(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[int(r), int(c)] = int(color)


def generate_input() -> np.ndarray:
    for _ in range(GEN_TRIES):
        h = random.randint(GRID_MIN, GRID_MAX)
        w = random.randint(GRID_MIN, GRID_MAX)
        shape = (int(h), int(w))

        plan = _plan_ray(shape)
        if plan is None:
            continue

        seed_cells = list(plan["seed"])
        path_cells = list(plan["path"])
        mirror_events = list(plan["mirrors"])
        transparent_target = random.randint(1, 2)
        if len(path_cells) < transparent_target:
            continue

        all_reserved = set(seed_cells) | set(path_cells) | {tuple(event["hit"]) for event in mirror_events}
        grid = np.full(shape, int(Color.BLACK), dtype=int)
        for r, c in seed_cells:
            grid[int(r), int(c)] = int(Color.BLUE)

        occupied_bars: set[tuple[int, int]] = set()
        used_lengths: set[int] = set()
        bars_expected = 0
        colors = random.sample(BAR_COLORS, 2 * len(mirror_events) + transparent_target)
        color_idx = 0

        mirror_ok = True
        for event in mirror_events:
            hit = tuple(map(int, event["hit"]))
            orientation = str(event["orientation"])
            placed = False
            lengths = [length for length in LENGTH_POOL if length <= max(shape)]
            random.shuffle(lengths)
            for length in lengths:
                if int(length) in used_lengths:
                    continue
                anchor_opts = _anchored_options(shape, hit, orientation, int(length))
                anchor_opts = [
                    cells
                    for cells in anchor_opts
                    if _cells_fit(cells, all_reserved, occupied_bars, {hit})
                ]
                if not anchor_opts:
                    continue

                partner_orients = ["h", "v"]
                random.shuffle(partner_orients)
                for anchor_cells in anchor_opts:
                    for partner_orientation in partner_orients:
                        partner_opts = _free_segment_options(shape, partner_orientation, int(length))
                        for partner_cells in partner_opts:
                            if not _cells_fit(
                                partner_cells,
                                all_reserved,
                                occupied_bars | set(anchor_cells),
                            ):
                                continue
                            _paint_bar(grid, anchor_cells, colors[color_idx])
                            _paint_bar(grid, partner_cells, colors[color_idx + 1])
                            occupied_bars.update(anchor_cells)
                            occupied_bars.update(partner_cells)
                            used_lengths.add(int(length))
                            color_idx += 2
                            bars_expected += 2
                            placed = True
                            break
                        if placed:
                            break
                    if placed:
                        break
                if placed:
                    break

            if not placed:
                mirror_ok = False
                break

        if not mirror_ok:
            continue

        available_path = path_cells[:]
        random.shuffle(available_path)
        transparent_done = 0
        while available_path and transparent_done < transparent_target:
            anchor = tuple(map(int, available_path.pop()))
            placed = False
            lengths = [length for length in LENGTH_POOL if length <= max(shape)]
            random.shuffle(lengths)
            for length in lengths:
                if int(length) in used_lengths:
                    continue
                orientations = ["h", "v"]
                random.shuffle(orientations)
                for orientation in orientations:
                    anchor_opts = _anchored_options(shape, anchor, orientation, int(length))
                    anchor_opts = [
                        cells
                        for cells in anchor_opts
                        if _cells_fit(cells, all_reserved, occupied_bars, {anchor})
                    ]
                    if not anchor_opts:
                        continue
                    cells = random.choice(anchor_opts)
                    _paint_bar(grid, cells, colors[color_idx])
                    occupied_bars.update(cells)
                    used_lengths.add(int(length))
                    color_idx += 1
                    bars_expected += 1
                    transparent_done += 1
                    placed = True
                    break
                if placed:
                    break

        if transparent_done != transparent_target:
            continue

        out, stats = _trace_ray(grid)
        if int(stats["reflective_hits"]) != len(mirror_events):
            continue
        if int(stats["transparent_hits"]) != transparent_target:
            continue
        if not isinstance(out, np.ndarray) or out.ndim != 2:
            continue
        if out.shape[0] > 30 or out.shape[1] > 30:
            continue

        bg = int(get_bg(grid))
        bars, _lookup = _parse_bars(grid, bg, seed_cells)
        if len(bars) != bars_expected:
            continue

        freq = Counter(int(bar["length"]) for bar in bars)
        if not any(int(count) >= 2 for count in freq.values()):
            continue
        if not any(int(count) == 1 for count in freq.values()):
            continue
        return grid

    raise RuntimeError("failed to generate a valid ARC instance")
