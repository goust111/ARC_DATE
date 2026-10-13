from __future__ import annotations

import random
from collections import Counter

import numpy as np


BACKGROUND = 0
TRAIL = 9
INPUT_COLORS = (1, 2, 3, 4)
GEN_ATTEMPTS = 500
SPECIAL_TRIES = 120
RANDOM_PLACE_TRIES = 240


BASE_SHAPES: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 1), (2, 2)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (2, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (1, 1), (2, 1), (2, 2), (3, 2)),
    ((0, 0), (0, 1), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2)),
)

DIAGONAL_SHAPES: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0), (1, 1)),
    ((0, 0), (1, 1), (2, 2)),
    ((0, 0), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (2, 1)),
    ((0, 0), (1, 1), (2, 1), (2, 2)),
)


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _component_lists(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 8:
        steps = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        steps = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in steps:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _count_components(grid: np.ndarray, color: int, connectivity: int) -> int:
    return len(_component_lists(grid == color, connectivity))


def _shift_map(grid: np.ndarray, connectivity: int = 8) -> dict[int, int]:
    colors = [int(c) for c in np.unique(grid) if int(c) not in (BACKGROUND, TRAIL)]
    return {color: _count_components(grid, color, connectivity) for color in colors}


def _render_with_shifts(
    grid: np.ndarray,
    shifts: dict[int, int],
    *,
    larger_shift_on_top: bool = True,
    leave_trail: bool = True,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = np.full_like(g, BACKGROUND)

    ordered = sorted(shifts.items(), key=lambda kv: kv[1], reverse=not larger_shift_on_top)
    for color, shift in ordered:
        rr, cc = np.where(g == color)
        for r, c in zip(rr.tolist(), cc.tolist()):
            nr = r + shift
            if 0 <= nr < h:
                out[nr, c] = color

    if leave_trail:
        rr, cc = np.where((g != BACKGROUND) & (g != TRAIL))
        for r, c in zip(rr.tolist(), cc.tolist()):
            if out[r, c] == BACKGROUND:
                out[r, c] = TRAIL
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Shift each color downward by its 8-connected object count and mark vacated cells with a maroon trail.

    Input structure:
    - The grid is black background with 2-4 foreground colors chosen from blue, red, green, and yellow.
    - Each foreground color appears as one or more small monochromatic 8-connected objects.
    - Different objects are separated by at least one background cell, and no trail color is present in the input.

    Rule:
    - Count the number of 8-connected components of each foreground color; that count is the color's downward shift amount.
    - Move every pixel of that color straight downward by exactly that many rows, clipping pixels that leave the bottom edge.
    - After all shifts, fill each original foreground cell with maroon if it is now empty.
    - If shifted pixels of two colors land on the same cell, the color with the larger shift amount remains visible there.

    Key insight:
    - The puzzle does not draw the counts directly: each color's component count is encoded as the distance that all pixels of that color fall.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert TRAIL not in set(map(int, g.ravel())), "input must not already contain the trail color"

    shifts = _shift_map(g, connectivity=8)
    assert shifts, "expected at least one foreground color"
    assert len(set(shifts.values())) == len(shifts), "foreground colors must have distinct component counts"
    return _render_with_shifts(g, shifts, larger_shift_on_top=True, leave_trail=True)


def _transform_coords(
    coords: tuple[tuple[int, int], ...], rng: random.Random, *, force_anchor: bool = True
) -> list[tuple[int, int]]:
    pts = [(int(r), int(c)) for r, c in coords]
    if rng.random() < 0.5:
        pts = [(r, -c) for r, c in pts]
    for _ in range(rng.randrange(4)):
        pts = [(-c, r) for r, c in pts]
    anchor_index = rng.randrange(len(pts)) if force_anchor else 0
    ar, ac = pts[anchor_index]
    pts = sorted({(r - ar, c - ac) for r, c in pts})
    return pts


def _sample_shape(rng: random.Random, *, force_diagonal: bool = False) -> list[tuple[int, int]]:
    library = DIAGONAL_SHAPES if force_diagonal else BASE_SHAPES
    return _transform_coords(library[rng.randrange(len(library))], rng)


def _abs_coords(anchor: tuple[int, int], rel_coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    ar, ac = anchor
    return [(ar + dr, ac + dc) for dr, dc in rel_coords]


def _can_place(grid: np.ndarray, blocked: np.ndarray, coords: list[tuple[int, int]]) -> bool:
    h, w = grid.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if grid[r, c] != BACKGROUND or blocked[r, c]:
            return False
    return True


def _blocked_after(blocked: np.ndarray, coords: list[tuple[int, int]]) -> np.ndarray:
    out = blocked.copy()
    h, w = out.shape
    for r, c in coords:
        for nr, nc in _neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = True
        out[r, c] = True
    return out


def _commit_object(
    grid: np.ndarray,
    blocked: np.ndarray,
    color: int,
    coords: list[tuple[int, int]],
    placed_objects: list[dict],
) -> np.ndarray:
    for r, c in coords:
        grid[r, c] = color
    placed_objects.append({"color": color, "coords": tuple(coords)})
    return _blocked_after(blocked, coords)


def _random_anchor(rng: random.Random, h: int, w: int) -> tuple[int, int]:
    return rng.randrange(h), rng.randrange(w)


def _try_place_anchor_object(
    grid: np.ndarray,
    blocked: np.ndarray,
    color: int,
    anchor: tuple[int, int],
    rng: random.Random,
    placed_objects: list[dict],
    *,
    force_diagonal: bool = False,
    must_truncate_by: int | None = None,
) -> np.ndarray | None:
    h, _w = grid.shape
    for _ in range(SPECIAL_TRIES):
        rel = _sample_shape(rng, force_diagonal=force_diagonal)
        coords = _abs_coords(anchor, rel)
        if not _can_place(grid, blocked, coords):
            continue
        if must_truncate_by is not None and not any(r + must_truncate_by >= h for r, _c in coords):
            continue
        return _commit_object(grid, blocked, color, coords, placed_objects)
    return None


def _try_place_overlap_pair(
    grid: np.ndarray,
    blocked: np.ndarray,
    low_color: int,
    low_shift: int,
    high_color: int,
    high_shift: int,
    rng: random.Random,
    placed_objects: list[dict],
    *,
    force_diagonal_low: bool = False,
) -> np.ndarray | None:
    diff = high_shift - low_shift
    h, w = grid.shape
    if diff < 2:
        return None
    for _ in range(SPECIAL_TRIES):
        c = rng.randrange(w)
        low_r = rng.randrange(diff, h)
        high_r = low_r - diff

        rel_low = _sample_shape(rng, force_diagonal=force_diagonal_low)
        coords_low = _abs_coords((low_r, c), rel_low)
        if not _can_place(grid, blocked, coords_low):
            continue
        blocked_after_low = _blocked_after(blocked, coords_low)

        rel_high = _sample_shape(rng, force_diagonal=False)
        coords_high = _abs_coords((high_r, c), rel_high)
        if not _can_place(grid, blocked_after_low, coords_high):
            continue

        blocked_after_low = _commit_object(grid, blocked, low_color, coords_low, placed_objects)
        return _commit_object(grid, blocked_after_low, high_color, coords_high, placed_objects)
    return None


def _try_place_reoccupy_pair(
    grid: np.ndarray,
    blocked: np.ndarray,
    mover_color: int,
    mover_shift: int,
    target_color: int,
    rng: random.Random,
    placed_objects: list[dict],
    *,
    force_diagonal_mover: bool = False,
) -> np.ndarray | None:
    h, w = grid.shape
    if mover_shift < 2:
        return None
    for _ in range(SPECIAL_TRIES):
        c = rng.randrange(w)
        mover_r = rng.randrange(0, h - mover_shift)
        target_r = mover_r + mover_shift

        rel_mover = _sample_shape(rng, force_diagonal=force_diagonal_mover)
        coords_mover = _abs_coords((mover_r, c), rel_mover)
        if not _can_place(grid, blocked, coords_mover):
            continue
        blocked_after_mover = _blocked_after(blocked, coords_mover)

        rel_target = _sample_shape(rng, force_diagonal=False)
        coords_target = _abs_coords((target_r, c), rel_target)
        if not _can_place(grid, blocked_after_mover, coords_target):
            continue

        blocked_after_mover = _commit_object(grid, blocked, mover_color, coords_mover, placed_objects)
        return _commit_object(grid, blocked_after_mover, target_color, coords_target, placed_objects)
    return None


def _try_place_random_object(
    grid: np.ndarray,
    blocked: np.ndarray,
    color: int,
    rng: random.Random,
    placed_objects: list[dict],
    *,
    force_diagonal: bool = False,
) -> np.ndarray | None:
    h, w = grid.shape
    for _ in range(RANDOM_PLACE_TRIES):
        anchor = _random_anchor(rng, h, w)
        rel = _sample_shape(rng, force_diagonal=force_diagonal)
        coords = _abs_coords(anchor, rel)
        if _can_place(grid, blocked, coords):
            return _commit_object(grid, blocked, color, coords, placed_objects)
    return None


def _choose_shift_assignment(
    rng: random.Random,
    n_colors: int,
    *,
    want_overlap: bool,
    want_reoccupy: bool,
) -> dict[int, int]:
    while True:
        colors = rng.sample(list(INPUT_COLORS), n_colors)
        shifts = rng.sample([1, 2, 3, 4], n_colors)
        assignment = {color: shift for color, shift in zip(colors, shifts)}
        values = sorted(assignment.values())
        if want_overlap and max(values) - min(values) < 2:
            continue
        if want_reoccupy and max(values) < 2:
            continue
        return assignment


def _analyze(grid: np.ndarray) -> dict:
    h, _w = grid.shape
    shifts8 = _shift_map(grid, connectivity=8)
    shifts4 = _shift_map(grid, connectivity=4)
    target_sources: dict[tuple[int, int], list[tuple[int, int, tuple[int, int]]]] = {}
    truncated = 0
    reoccupy = False
    for color, shift in shifts8.items():
        rr, cc = np.where(grid == color)
        for r, c in zip(rr.tolist(), cc.tolist()):
            nr = r + shift
            if nr >= h:
                truncated += 1
                continue
            key = (nr, c)
            target_sources.setdefault(key, []).append((color, shift, (r, c)))
            if grid[nr, c] != BACKGROUND:
                reoccupy = True

    overlap_cells = {cell: srcs for cell, srcs in target_sources.items() if len(srcs) > 1}
    totals = {color: int(np.count_nonzero(grid == color)) for color in shifts8}
    bottoms = {
        color: int(np.max(np.argwhere(grid == color)[:, 0]))
        for color in shifts8
    }
    actual = _render_with_shifts(grid, shifts8, larger_shift_on_top=True, leave_trail=True)
    reverse_priority = _render_with_shifts(grid, shifts8, larger_shift_on_top=False, leave_trail=True)
    no_trail = _render_with_shifts(grid, shifts8, larger_shift_on_top=True, leave_trail=False)
    pixel_shift = _render_with_shifts(grid, totals, larger_shift_on_top=True, leave_trail=True)
    four_conn = _render_with_shifts(grid, shifts4, larger_shift_on_top=True, leave_trail=True)
    max_comp_size = {}
    for color in shifts8:
        comps = _component_lists(grid == color, 8)
        max_comp_size[color] = max(len(comp) for comp in comps)
    max_size_shift = _render_with_shifts(grid, max_comp_size, larger_shift_on_top=True, leave_trail=True)

    anti_size_pair = False
    anti_bottom_pair = False
    colors = list(shifts8)
    for i, c1 in enumerate(colors):
        for c2 in colors[i + 1 :]:
            s1, s2 = shifts8[c1], shifts8[c2]
            if (s1 - s2) * (totals[c1] - totals[c2]) < 0:
                anti_size_pair = True
            if (s1 - s2) * (bottoms[c1] - bottoms[c2]) <= 0:
                anti_bottom_pair = True

    return {
        "shifts8": shifts8,
        "shifts4": shifts4,
        "actual": actual,
        "reverse_priority": reverse_priority,
        "no_trail": no_trail,
        "pixel_shift": pixel_shift,
        "four_conn": four_conn,
        "max_size_shift": max_size_shift,
        "overlap_cells": overlap_cells,
        "truncated": truncated,
        "reoccupy": reoccupy,
        "trail_count": int(np.count_nonzero(actual == TRAIL)),
        "anti_size_pair": anti_size_pair,
        "anti_bottom_pair": anti_bottom_pair,
    }


def _valid_global_separation(grid: np.ndarray) -> bool:
    fg = [(int(r), int(c)) for r, c in np.argwhere(grid != BACKGROUND)]
    fg_set = set(fg)
    for r, c in fg:
        color = int(grid[r, c])
        for nr, nc in _neighbors8(r, c):
            if (nr, nc) in fg_set and int(grid[nr, nc]) != color:
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    want_overlap = rng.random() < 0.75
    want_reoccupy = rng.random() < 0.7
    want_trunc = rng.random() < 0.65
    want_diagonal = True

    n_colors = rng.choices([2, 3, 4], weights=[2, 5, 3])[0]
    if want_overlap and want_reoccupy and n_colors == 2:
        n_colors = 3

    h = rng.randint(13, 18)
    w = rng.randint(13, 18)
    shifts = _choose_shift_assignment(
        rng,
        n_colors,
        want_overlap=want_overlap,
        want_reoccupy=want_reoccupy,
    )

    grid = np.full((h, w), BACKGROUND, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    placed_objects: list[dict] = []
    remaining = dict(shifts)

    diagonal_used = False

    if want_overlap:
        overlap_pairs = [
            (c1, s1, c2, s2)
            for c1, s1 in shifts.items()
            for c2, s2 in shifts.items()
            if c1 != c2 and s2 - s1 >= 2 and remaining[c1] > 0 and remaining[c2] > 0
        ]
        rng.shuffle(overlap_pairs)
        if overlap_pairs:
            low_color, low_shift, high_color, high_shift = overlap_pairs[0]
            new_blocked = _try_place_overlap_pair(
                grid,
                blocked,
                low_color,
                low_shift,
                high_color,
                high_shift,
                rng,
                placed_objects,
                force_diagonal_low=want_diagonal,
            )
            if new_blocked is not None:
                blocked = new_blocked
                remaining[low_color] -= 1
                remaining[high_color] -= 1
                diagonal_used = want_diagonal
            else:
                want_overlap = False
        else:
            want_overlap = False

    if want_reoccupy:
        reoccupy_pairs = [
            (mover, shifts[mover], target)
            for mover in shifts
            for target in shifts
            if mover != target
            and shifts[mover] >= 2
            and remaining[mover] > 0
            and remaining[target] > 0
        ]
        rng.shuffle(reoccupy_pairs)
        placed = False
        for mover_color, mover_shift, target_color in reoccupy_pairs:
            new_blocked = _try_place_reoccupy_pair(
                grid,
                blocked,
                mover_color,
                mover_shift,
                target_color,
                rng,
                placed_objects,
                force_diagonal_mover=want_diagonal and not diagonal_used,
            )
            if new_blocked is None:
                continue
            blocked = new_blocked
            remaining[mover_color] -= 1
            remaining[target_color] -= 1
            diagonal_used = diagonal_used or (want_diagonal and not diagonal_used)
            placed = True
            break
        if not placed:
            want_reoccupy = False

    if want_trunc:
        trunc_candidates = [color for color, count in remaining.items() if count > 0]
        rng.shuffle(trunc_candidates)
        placed = False
        for color in trunc_candidates:
            shift = shifts[color]
            anchor_row = rng.randint(max(0, h - shift - 2), h - 1)
            anchor_col = rng.randrange(w)
            new_blocked = _try_place_anchor_object(
                grid,
                blocked,
                color,
                (anchor_row, anchor_col),
                rng,
                placed_objects,
                force_diagonal=want_diagonal and not diagonal_used,
                must_truncate_by=shift,
            )
            if new_blocked is None:
                continue
            blocked = new_blocked
            remaining[color] -= 1
            diagonal_used = diagonal_used or (want_diagonal and not diagonal_used)
            placed = True
            break
        if not placed:
            want_trunc = False

    if want_diagonal and not diagonal_used:
        diag_candidates = [color for color, count in remaining.items() if count > 0]
        rng.shuffle(diag_candidates)
        placed = False
        for color in diag_candidates:
            new_blocked = _try_place_random_object(
                grid,
                blocked,
                color,
                rng,
                placed_objects,
                force_diagonal=True,
            )
            if new_blocked is None:
                continue
            blocked = new_blocked
            remaining[color] -= 1
            diagonal_used = True
            placed = True
            break
        if not placed:
            return None

    colors_order = list(shifts)
    rng.shuffle(colors_order)
    for color in colors_order:
        while remaining[color] > 0:
            new_blocked = _try_place_random_object(
                grid,
                blocked,
                color,
                rng,
                placed_objects,
                force_diagonal=False,
            )
            if new_blocked is None:
                return None
            blocked = new_blocked
            remaining[color] -= 1

    if not _valid_global_separation(grid):
        return None

    analysis = _analyze(grid)
    if not np.array_equal(main(grid), analysis["actual"]):
        return None
    if any(analysis["shifts8"].get(color, 0) != shift for color, shift in shifts.items()):
        return None
    if len(set(analysis["shifts8"].values())) != len(analysis["shifts8"]):
        return None
    if not any(np.any(grid == color) for color in shifts):
        return None
    if analysis["trail_count"] == 0:
        return None
    if np.array_equal(grid, analysis["actual"]):
        return None
    if not np.any((analysis["actual"] != BACKGROUND) & (analysis["actual"] != TRAIL)):
        return None
    if not any(
        any(r + shift < h for r, _c in np.argwhere(grid == color))
        for color, shift in shifts.items()
    ):
        return None
    if np.array_equal(analysis["actual"], analysis["no_trail"]):
        return None
    if np.array_equal(analysis["actual"], analysis["pixel_shift"]):
        return None
    if np.array_equal(analysis["actual"], analysis["max_size_shift"]):
        return None
    if np.array_equal(analysis["actual"], analysis["four_conn"]):
        return None
    if want_overlap:
        if not analysis["overlap_cells"]:
            return None
        if np.array_equal(analysis["actual"], analysis["reverse_priority"]):
            return None
    if want_reoccupy and not analysis["reoccupy"]:
        return None
    if want_trunc and analysis["truncated"] == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
