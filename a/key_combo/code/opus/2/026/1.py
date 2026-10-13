from __future__ import annotations

import itertools
import random
from collections import defaultdict

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
FOREGROUND = (RED, GREEN)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _find_pivot(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid == GREY)
    assert coords.shape[0] == 1, "expected exactly one grey pivot"
    r, c = coords[0]
    return int(r), int(c)


def _base_canvas(grid: np.ndarray) -> np.ndarray:
    base = np.asarray(grid, dtype=int).copy()
    movable = (base == RED) | (base == GREEN)
    base[movable] = BLACK
    return base


def _foreground_pixels(grid: np.ndarray) -> list[tuple[int, int, int]]:
    coords = np.argwhere((grid == RED) | (grid == GREEN))
    return [(int(r), int(c), int(grid[r, c])) for r, c in coords]


def _orbit_from_offset(pr: int, pc: int, dr: int, dc: int) -> list[tuple[int, int]]:
    return [
        (int(pr + dr), int(pc + dc)),
        (int(pr + dc), int(pc - dr)),
        (int(pr - dr), int(pc - dc)),
        (int(pr - dc), int(pc + dr)),
    ]


def _rotated_positions(pr: int, pc: int, r: int, c: int) -> list[tuple[int, int]]:
    dr = int(r - pr)
    dc = int(c - pc)
    return [
        (int(pr + dc), int(pc - dr)),
        (int(pr - dr), int(pc - dc)),
        (int(pr - dc), int(pc + dr)),
    ]


def _settle_pixels(
    grid: np.ndarray, *, descending: bool = True
) -> list[tuple[int, int, int]]:
    g = np.asarray(grid, dtype=int)
    h, _w = g.shape
    pixels = _foreground_pixels(g)
    if descending:
        pixels.sort(key=lambda item: (-int(item[0]), int(item[1])))
    else:
        pixels.sort(key=lambda item: (int(item[0]), int(item[1])))

    occupied = g == BLUE
    settled: list[tuple[int, int, int]] = []
    for r, c, color in pixels:
        rr = int(r)
        while rr + 1 < h and not bool(occupied[rr + 1, c]):
            rr += 1
        settled.append((int(rr), int(c), int(color)))
        occupied[rr, c] = True
    return settled


def _render_completed(
    grid: np.ndarray,
    settled: list[tuple[int, int, int]],
    pivot: tuple[int, int],
) -> np.ndarray:
    out = _base_canvas(grid)
    pr, pc = pivot
    for r, c, color in settled:
        out[r, c] = int(color)

    for r, c, color in settled:
        for rr, cc in _rotated_positions(pr, pc, r, c):
            if _in_bounds(out.shape, rr, cc) and int(out[rr, cc]) == BLACK:
                out[rr, cc] = int(color)
    return out


def _paint_settled_only(
    grid: np.ndarray, settled: list[tuple[int, int, int]]
) -> np.ndarray:
    out = _base_canvas(grid)
    for r, c, color in settled:
        out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop the scattered colored pixels onto shelves, then complete the settled figure by 90-degree rotation around the pivot.

    Input structure:
    - A black background grid contains one grey pivot pixel, several immobile blue horizontal shelf segments, and scattered red/green single pixels.
    - The red/green pixels are movable foreground; the shelves remain fixed and the pivot only marks the rotation center.
    - Foreground pixels start in the upper half while the pivot sits lower in the canvas.

    Rule:
    - Process the red/green pixels in descending initial row order, breaking ties left-to-right.
    - Each pixel falls straight down in its own column until the next row would be a blue shelf, a previously settled pixel, or the bottom edge.
    - After all pixels settle, keep the shelves, pivot, and settled originals, then for every settled pixel fill its three 90-degree rotational mates around the grey pivot whenever the target cell is in bounds and currently black.

    Key insight:
    - The important structure appears only after gravity reorders the colored pixels; the rotational gaps are defined by the settled layout, not by the initial scatter.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    pivot = _find_pivot(g)
    settled = _settle_pixels(g, descending=True)
    return _render_completed(g, settled, pivot)


_SIMPLE_PATTERNS_2 = [(2, 2), (2, 3), (3, 2), (3, 3), (1, 3), (3, 1)]
_SIMPLE_PATTERNS_3 = [
    pattern
    for pattern in itertools.product((1, 2, 3), repeat=3)
    if 4 <= sum(pattern) <= 8
]


def _choose_pivot(h: int, w: int) -> tuple[int, int] | None:
    row_lo = max(4, h // 2)
    row_hi = h - 4
    col_lo = 3
    col_hi = w - 4
    if row_lo > row_hi or col_lo > col_hi:
        return None

    for _ in range(60):
        pr = random.randint(row_lo, row_hi)
        pc = random.randint(col_lo, col_hi)
        if (pr, pc) == (h // 2, w // 2):
            continue
        radius = min(pr, h - 1 - pr, pc, w - 1 - pc, 5)
        if radius >= 3:
            return int(pr), int(pc)
    return None


def _all_orbit_candidates(
    h: int, w: int, pivot: tuple[int, int], *, allow_axis: bool
) -> list[dict]:
    pr, pc = pivot
    radius = min(pr, h - 1 - pr, pc, w - 1 - pc, 5)
    seen: set[tuple[tuple[int, int], ...]] = set()
    candidates: list[dict] = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr == 0 and dc == 0:
                continue
            if (dr == 0 or dc == 0) and not allow_axis:
                continue
            if dr != 0 and dc != 0 and abs(dr) == abs(dc):
                continue

            cells = _orbit_from_offset(pr, pc, dr, dc)
            if len(set(cells)) != 4:
                continue
            if not all(_in_bounds((h, w), r, c) for r, c in cells):
                continue
            key = tuple(sorted(cells))
            if key in seen:
                continue
            seen.add(key)
            candidates.append({"offset": (int(dr), int(dc)), "cells": cells})
    random.shuffle(candidates)
    return candidates


def _subset_choices(
    cells: list[tuple[int, int]],
    size: int,
    used_cols: set[int],
    h: int,
    pivot_col: int,
) -> list[tuple[int, ...]]:
    choices: list[tuple[int, ...]] = []
    for combo in itertools.combinations(range(4), size):
        cols = [int(cells[idx][1]) for idx in combo]
        rows = [int(cells[idx][0]) for idx in combo]
        if len(set(cols)) != len(cols):
            continue
        if any(col in used_cols for col in cols):
            continue
        if any(col == pivot_col for col in cols):
            continue
        if any(row < 1 or row >= h - 1 for row in rows):
            continue
        choices.append(tuple(int(idx) for idx in combo))
    return choices


def _selected_cells_from_orbits(orbits: list[dict]) -> list[tuple[int, int, int]]:
    selected: list[tuple[int, int, int]] = []
    for orbit in orbits:
        color = int(orbit["color"])
        for idx in orbit["selected"]:
            r, c = orbit["cells"][int(idx)]
            selected.append((int(r), int(c), color))
    return selected


def _full_orbit_cells(orbits: list[dict]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for orbit in orbits:
        cells.update((int(r), int(c)) for r, c in orbit["cells"])
    return cells


def _build_simple_orbits(h: int, w: int, pivot: tuple[int, int]) -> list[dict] | None:
    candidates = _all_orbit_candidates(h, w, pivot, allow_axis=False)
    orbit_count = random.choice((2, 2, 3))
    patterns = _SIMPLE_PATTERNS_2 if orbit_count == 2 else _SIMPLE_PATTERNS_3
    pivot_col = int(pivot[1])

    if len(candidates) < orbit_count:
        return None

    for _ in range(100):
        size_pattern = list(random.choice(patterns))
        random.shuffle(size_pattern)
        used_full: set[tuple[int, int]] = set()
        used_cols: set[int] = set()
        orbits: list[dict] = []

        shuffled = candidates[:]
        random.shuffle(shuffled)
        for size in size_pattern:
            picked = None
            for cand in shuffled:
                full = {tuple(map(int, rc)) for rc in cand["cells"]}
                if full & used_full:
                    continue
                options = _subset_choices(cand["cells"], size, used_cols, h, pivot_col)
                if not options:
                    continue
                picked = {
                    "cells": list(cand["cells"]),
                    "color": int(random.choice(FOREGROUND)),
                    "selected": random.choice(options),
                }
                break
            if picked is None:
                break
            orbits.append(picked)
            used_full.update((int(r), int(c)) for r, c in picked["cells"])
            used_cols.update(int(picked["cells"][idx][1]) for idx in picked["selected"])

        if len(orbits) != orbit_count:
            continue

        selected = _selected_cells_from_orbits(orbits)
        support_rows = {int(r + 1) for r, _c, _color in selected if int(r) < h - 1}
        if not (4 <= len(selected) <= 8):
            continue
        if len(support_rows) < 2:
            continue
        return orbits
    return None


def _build_collision_orbits(
    h: int, w: int, pivot: tuple[int, int]
) -> list[dict] | None:
    pr, pc = pivot
    radius = min(pr, h - 1 - pr, pc, w - 1 - pc, 5)
    if radius < 3:
        return None

    bs = list(range(2, radius + 1))
    random.shuffle(bs)
    for b in bs:
        orbit_a = _orbit_from_offset(pr, pc, 1, b)
        orbit_b = _orbit_from_offset(pr, pc, 0, b)
        if len(set(orbit_a)) != 4 or len(set(orbit_b)) != 4:
            continue
        if set(orbit_a) & set(orbit_b):
            continue
        colors = random.sample(list(FOREGROUND), 2)
        orbits = [
            {"cells": orbit_a, "color": int(colors[0]), "selected": (0, 2)},
            {"cells": orbit_b, "color": int(colors[1]), "selected": (0, 2)},
        ]

        if random.random() < 0.6:
            used_full = _full_orbit_cells(orbits)
            used_cols = {int(c) for _r, c, _color in _selected_cells_from_orbits(orbits)}
            pivot_col = int(pivot[1])
            extras = _all_orbit_candidates(h, w, pivot, allow_axis=False)
            random.shuffle(extras)
            for cand in extras:
                full = {tuple(map(int, rc)) for rc in cand["cells"]}
                if full & used_full:
                    continue
                size = random.choice((1, 2, 2, 3))
                options = _subset_choices(cand["cells"], size, used_cols, h, pivot_col)
                if not options:
                    continue
                orbits.append(
                    {
                        "cells": list(cand["cells"]),
                        "color": int(random.choice(FOREGROUND)),
                        "selected": random.choice(options),
                    }
                )
                break

        selected = _selected_cells_from_orbits(orbits)
        support_rows = {int(r + 1) for r, _c, _color in selected if int(r) < h - 1}
        if not (4 <= len(selected) <= 8):
            continue
        if len(support_rows) < 2:
            continue
        return orbits
    return None


def _cover_required_columns(
    required: set[int],
    candidates: list[tuple[int, int, tuple[int, ...]]],
    chosen: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    if not required:
        return []

    first = min(required)
    options = [cand for cand in candidates if first in cand[2]]
    random.shuffle(options)
    for a, b, covered in options:
        if any(not (b < oa - 1 or ob < a - 1) for oa, ob in chosen):
            continue
        rest = _cover_required_columns(
            required - set(covered), candidates, chosen + [(int(a), int(b))]
        )
        if rest is not None:
            return [(int(a), int(b))] + rest
    return None


def _build_shelf_segments(
    h: int,
    w: int,
    pivot: tuple[int, int],
    selected: list[tuple[int, int, int]],
    protected: set[tuple[int, int]],
) -> list[tuple[int, int, int]] | None:
    by_col: dict[int, list[int]] = defaultdict(list)
    for r, c, _color in selected:
        by_col[int(c)].append(int(r))

    required_by_row: dict[int, set[int]] = defaultdict(set)
    for c, rows in by_col.items():
        rows = sorted(rows)
        if any(rows[idx + 1] != rows[idx] + 1 for idx in range(len(rows) - 1)):
            return None
        bottom = rows[-1]
        if bottom < h - 1:
            required_by_row[int(bottom + 1)].add(int(c))

    if len(required_by_row) < 2:
        return None

    placed_cells: set[tuple[int, int]] = set()
    segments: list[tuple[int, int, int]] = []
    for row in sorted(required_by_row):
        required = sorted(required_by_row[row])
        danger_cols = {int(c) for r, c, _color in selected if int(r) >= int(row)}
        forbidden_cols = danger_cols - set(required)
        candidates: list[tuple[int, int, tuple[int, ...]]] = []

        for length in (2, 3, 4):
            for start in range(0, w - length + 1):
                end = int(start + length - 1)
                covered = tuple(c for c in required if start <= c <= end)
                if not covered:
                    continue

                ok = True
                for col in range(start, end + 1):
                    if col in forbidden_cols:
                        ok = False
                        break
                    if (int(row), int(col)) in protected:
                        ok = False
                        break
                    if (int(row - 1), int(col)) in placed_cells or (int(row + 1), int(col)) in placed_cells:
                        ok = False
                        break
                if ok:
                    candidates.append((int(start), int(end), covered))

        if not candidates:
            return None

        cover = _cover_required_columns(set(required), candidates, [])
        if cover is None:
            return None

        for start, end in cover:
            segments.append((int(row), int(start), int(end)))
            for col in range(start, end + 1):
                placed_cells.add((int(row), int(col)))

    return segments


def _build_input_from_orbits(
    h: int, w: int, pivot: tuple[int, int], orbits: list[dict]
) -> tuple[np.ndarray, list[tuple[int, int, int]], list[tuple[int, int, int]]] | None:
    selected = _selected_cells_from_orbits(orbits)
    protected = _full_orbit_cells(orbits)
    protected.add((int(pivot[0]), int(pivot[1])))

    segments = _build_shelf_segments(h, w, pivot, selected, protected)
    if segments is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    grid[pivot] = GREY
    for r, c0, c1 in segments:
        grid[r, c0 : c1 + 1] = BLUE

    by_col: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r, c, color in selected:
        by_col[int(c)].append((int(r), int(color)))

    top_limit = h // 2 - 1
    if top_limit < 0:
        return None

    for c, group in by_col.items():
        group.sort(key=lambda item: int(item[0]))
        max_start = min(top_limit, int(group[0][0]) - 1)
        if max_start < 0:
            return None
        available = [r for r in range(0, max_start + 1) if int(grid[r, c]) == BLACK]
        if len(available) < len(group):
            return None

        start_rows = sorted(random.sample(available, len(group)))
        final_desc = sorted(group, key=lambda item: int(item[0]), reverse=True)
        start_desc = sorted(start_rows, reverse=True)
        for start_r, (_final_r, color) in zip(start_desc, final_desc):
            grid[int(start_r), int(c)] = int(color)

    if int(np.count_nonzero((grid == RED) | (grid == GREEN))) != len(selected):
        return None
    return grid, selected, segments


def _blue_segments_valid(grid: np.ndarray) -> bool:
    blue = grid == BLUE
    if not bool(blue.any()):
        return False
    if bool(np.any(blue[:-1, :] & blue[1:, :])):
        return False

    h, w = grid.shape
    for r in range(h):
        c = 0
        while c < w:
            if not bool(blue[r, c]):
                c += 1
                continue
            c0 = c
            while c + 1 < w and bool(blue[r, c + 1]):
                c += 1
            if not 2 <= c - c0 + 1 <= 4:
                return False
            c += 1
    return True


def _norm_cells(cells: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    return sorted((int(r), int(c), int(color)) for r, c, color in cells)


def _validate_generated(grid: np.ndarray, meta: dict) -> bool:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    if g.ndim != 2 or max(h, w) > 30:
        return False
    if np.argwhere(g == GREY).shape[0] != 1:
        return False
    if not _blue_segments_valid(g):
        return False

    initial = _foreground_pixels(g)
    if not (4 <= len(initial) <= 8):
        return False
    if any(int(r) >= h // 2 for r, _c, _color in initial):
        return False

    pivot = _find_pivot(g)
    settled = _settle_pixels(g, descending=True)
    target_settled = meta["selected_cells"]
    if _norm_cells(settled) != _norm_cells(target_settled):
        return False

    out = main(g)
    expected = _render_completed(g, target_settled, pivot)
    if not np.array_equal(out, expected):
        return False

    settled_only = _paint_settled_only(g, target_settled)
    if np.array_equal(out, settled_only):
        return False

    skip_gravity = _render_completed(g, initial, pivot)
    if np.array_equal(out, skip_gravity):
        return False

    moved = any((r, c, color) not in set(_norm_cells(target_settled)) for r, c, color in initial)
    if not moved:
        return False

    support_rows = {int(r + 1) for r, _c, _color in target_settled if int(r) < h - 1}
    if len(support_rows) < 2:
        return False

    if meta["mode"] == "collision":
        wrong = _settle_pixels(g, descending=False)
        if _norm_cells(wrong) == _norm_cells(target_settled):
            return False
        wrong_out = _render_completed(g, wrong, pivot)
        if np.array_equal(out, wrong_out):
            return False

    colored_out = int(np.count_nonzero((out == RED) | (out == GREEN)))
    if colored_out <= len(target_settled):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(13, 20)
    w = random.randint(13, 20)
    pivot = _choose_pivot(h, w)
    if pivot is None:
        return None

    mode = random.choices(("collision", "simple"), weights=(3, 2), k=1)[0]
    if mode == "collision":
        orbits = _build_collision_orbits(h, w, pivot)
        if orbits is None:
            orbits = _build_simple_orbits(h, w, pivot)
            mode = "simple"
    else:
        orbits = _build_simple_orbits(h, w, pivot)
        if orbits is None:
            orbits = _build_collision_orbits(h, w, pivot)
            mode = "collision"

    if orbits is None:
        return None

    built = _build_input_from_orbits(h, w, pivot, orbits)
    if built is None:
        return None
    grid, selected, segments = built

    meta = {
        "mode": mode,
        "pivot": pivot,
        "orbits": orbits,
        "selected_cells": selected,
        "segments": segments,
    }
    if _validate_generated(grid, meta):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gravity-rotation ARC input")
