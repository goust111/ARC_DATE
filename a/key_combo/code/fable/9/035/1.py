from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2

_DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _blue_seed_coords(grid: np.ndarray) -> set[tuple[int, int]]:
    coords = {tuple(map(int, rc)) for rc in np.argwhere(grid == BLUE)}
    assert 2 <= len(coords) <= 3, "expected a 2-3 cell blue diagonal seed"

    same_diff = len({r - c for r, c in coords}) == 1
    same_sum = len({r + c for r, c in coords}) == 1
    assert same_diff ^ same_sum, "seed must lie on exactly one diagonal"

    rows = sorted(r for r, _c in coords)
    cols = sorted(c for _r, c in coords)
    assert rows == list(range(rows[0], rows[-1] + 1))
    assert cols == list(range(cols[0], cols[-1] + 1))
    return coords


def _initial_state(shape: tuple[int, int], seed: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    if len({r - c for r, c in seed}) == 1:
        first = min(seed)
        last = max(seed)
        endpoints = ((first, (-1, -1)), (last, (1, 1)))
    else:
        top = min(seed)
        bottom = max(seed)
        endpoints = ((top, (-1, 1)), (bottom, (1, -1)))

    candidates: list[tuple[int, int, int, int]] = []
    for (r, c), (dr, dc) in endpoints:
        if _in_bounds(shape, r + dr, c + dc):
            candidates.append((r, c, dr, dc))
    assert len(candidates) == 1, "seed must have exactly one in-bounds extension"
    return candidates[0]


def _red_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros(grid.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    h, w = grid.shape
    for r0, c0 in map(tuple, np.argwhere(grid == RED)):
        r0 = int(r0)
        c0 = int(c0)
        if seen[r0, c0]:
            continue
        seen[r0, c0] = True
        comp: list[tuple[int, int]] = []
        q = deque([(r0, c0)])
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in _DIR4:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < h
                    and 0 <= nc < w
                    and not seen[nr, nc]
                    and int(grid[nr, nc]) == RED
                ):
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _mirror_orientations(grid: np.ndarray) -> dict[tuple[int, int], str]:
    orientations: dict[tuple[int, int], str] = {}
    for comp in _red_components(grid):
        rows = sorted({r for r, _c in comp})
        cols = sorted({c for _r, c in comp})
        if len(rows) == 1:
            assert cols == list(range(cols[0], cols[-1] + 1))
            orient = "horizontal"
        elif len(cols) == 1:
            assert rows == list(range(rows[0], rows[-1] + 1))
            orient = "vertical"
        else:
            raise AssertionError("each red component must be a straight mirror segment")
        for cell in comp:
            orientations[cell] = orient
    return orientations


def _simulate(
    grid: np.ndarray,
    *,
    consume: bool,
) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    seed = _blue_seed_coords(g)
    r, c, dr, dc = _initial_state(g.shape, seed)
    mirrors = _mirror_orientations(g)
    live = set(mirrors)
    original_mirrors = set(mirrors)

    out = g.copy()
    reflections: list[tuple[int, int]] = []
    traversed_broken: list[tuple[int, int]] = []
    painted_path: list[tuple[int, int]] = []
    seen_states: set[tuple[int, int, int, int, frozenset[tuple[int, int]]]] = set()

    while True:
        state = (r, c, dr, dc, frozenset(live) if consume else frozenset())
        assert state not in seen_states, "ray path must terminate without looping"
        seen_states.add(state)

        nr, nc = r + dr, c + dc
        if not _in_bounds(g.shape, nr, nc):
            break

        if (nr, nc) in live:
            orient = mirrors[(nr, nc)]
            reflections.append((nr, nc))
            if consume:
                live.remove((nr, nc))
                out[nr, nc] = BLACK
            if orient == "horizontal":
                dr = -dr
            else:
                dc = -dc
            continue

        if (nr, nc) in original_mirrors:
            traversed_broken.append((nr, nc))
        out[nr, nc] = BLUE
        painted_path.append((nr, nc))
        r, c = nr, nc

    return {
        "output": out,
        "reflections": reflections,
        "traversed_broken": traversed_broken,
        "painted_path": painted_path,
        "mirror_cells": original_mirrors,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the diagonal blue ray while consuming each struck red mirror pixel.

    Input structure:
    - A black background contains exactly one 2-3 cell blue diagonal seed.
    - Red foreground objects are straight horizontal or vertical mirror segments.
    - The seed has exactly one endpoint whose outward diagonal continuation stays inside the grid.

    Rule:
    - Start from that in-bounds seed endpoint and extend the ray one diagonal cell at a time.
    - If the next cell is an intact red mirror pixel, do not paint it; erase that pixel to black and
      reflect by flipping the row-direction on a horizontal mirror or the column-direction on a
      vertical mirror.
    - If the ray later reaches a previously shattered mirror position, it paints that cell blue and
      continues straight through. Stop when the ray would leave the grid.

    Key insight:
    - Mirror cells are consumable state, so the same geometric location can first reflect the ray and
      later become an ordinary pass-through cell; the path must be simulated in time order.
    """
    return _simulate(grid, consume=True)["output"]  # type: ignore[index]


def _available_span(
    shape: tuple[int, int],
    cell: tuple[int, int],
    orientation: str,
    blocked: set[tuple[int, int]],
) -> tuple[int, int]:
    r, c = cell
    if orientation == "horizontal":
        left = c
        while left - 1 >= 0 and (r, left - 1) not in blocked:
            left -= 1
        right = c
        w = shape[1]
        while right + 1 < w and (r, right + 1) not in blocked:
            right += 1
        return left, right
    top = r
    while top - 1 >= 0 and (top - 1, c) not in blocked:
        top -= 1
    bottom = r
    h = shape[0]
    while bottom + 1 < h and (bottom + 1, c) not in blocked:
        bottom += 1
    return top, bottom


def _choose_mirror_segment(
    shape: tuple[int, int],
    cell: tuple[int, int],
    orientation: str,
    blocked: set[tuple[int, int]],
    occupied_red: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    lo, hi = _available_span(shape, cell, orientation, blocked)
    pivot = cell[1] if orientation == "horizontal" else cell[0]
    min_len = 2
    max_len = min(7, hi - lo + 1)
    if max_len < min_len:
        return None

    choices: list[set[tuple[int, int]]] = []
    for length in range(min_len, max_len + 1):
        start_min = max(lo, pivot - length + 1)
        start_max = min(pivot, hi - length + 1)
        for start in range(start_min, start_max + 1):
            end = start + length - 1
            if orientation == "horizontal":
                segment = {(cell[0], cc) for cc in range(start, end + 1)}
            else:
                segment = {(rr, cell[1]) for rr in range(start, end + 1)}
            bad_contact = False
            for r, c in segment:
                for dr, dc in _DIR4:
                    nbr = (r + dr, c + dc)
                    if nbr in occupied_red and nbr not in segment:
                        bad_contact = True
                        break
                if bad_contact:
                    break
            if bad_contact:
                continue
            choices.append(segment)
    if not choices:
        return None
    return random.choice(choices)


def _random_unused_segment(
    grid: np.ndarray,
    forbidden: set[tuple[int, int]],
    occupied_red: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    h, w = grid.shape
    axes = []
    for r in range(h):
        for c0 in range(w - 1):
            for c1 in range(c0 + 1, min(w, c0 + 8)):
                cells = {(r, c) for c in range(c0, c1 + 1)}
                if cells.isdisjoint(forbidden):
                    ok = True
                    for rr, cc in cells:
                        for dr, dc in _DIR4:
                            nbr = (rr + dr, cc + dc)
                            if nbr in occupied_red and nbr not in cells:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        continue
                    axes.append(cells)
    for c in range(w):
        for r0 in range(h - 1):
            for r1 in range(r0 + 1, min(h, r0 + 8)):
                cells = {(r, c) for r in range(r0, r1 + 1)}
                if cells.isdisjoint(forbidden):
                    ok = True
                    for rr, cc in cells:
                        for dr, dc in _DIR4:
                            nbr = (rr + dr, cc + dc)
                            if nbr in occupied_red and nbr not in cells:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        continue
                    axes.append(cells)
    if not axes:
        return None
    return random.choice(axes)


def _apply_random_symmetry(grid: np.ndarray) -> np.ndarray:
    k = random.randrange(4)
    out = np.rot90(grid, k)
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _canonical_case() -> np.ndarray | None:
    seed_len = random.randint(2, 3)
    first_leg = random.randint(2, 6)
    revisit_gap = random.randint(2, 5)
    start_row_min = max(1, revisit_gap + 3 - seed_len - first_leg)
    start_row = random.randint(start_row_min, start_row_min + 4)
    right_pad = random.randint(1, 4)
    bottom_pad = random.randint(1, 4)

    seed = [(start_row + i, i) for i in range(seed_len)]
    p0 = seed[-1]
    p1 = (p0[0] + first_leg, p0[1] + first_leg)
    m1 = (p1[0] + 1, p1[1] + 1)
    p2 = (p1[0] - revisit_gap, p1[1] + revisit_gap)
    m2 = (p2[0] - 1, p2[1] + 1)
    p3 = (p2[0] + 1, p2[1] + 1)
    m3 = (p3[0] + 1, p3[1] + 1)

    seg0 = [(p0[0] + t, p0[1] + t) for t in range(1, first_leg + 1)]
    seg1 = [(p1[0] - t, p1[1] + t) for t in range(1, revisit_gap + 1)]
    seg2 = [p3]
    post_len = p3[1]
    seg3 = [(p3[0] + t, p3[1] - t) for t in range(1, post_len + 1)]

    all_path = set(seed) | set(seg0) | set(seg1) | set(seg2) | set(seg3)
    all_hits = {m1, m2, m3}
    max_r = max(r for r, _c in all_path | all_hits)
    max_c = max(c for _r, c in all_path | all_hits)
    h = max_r + 1 + bottom_pad
    w = max_c + 1 + right_pad
    if max(h, w) > 30:
        return None

    grid = np.zeros((h, w), dtype=int)
    for r, c in seed:
        grid[r, c] = BLUE

    blocked = set(all_path)
    mirror_segments: list[set[tuple[int, int]]] = []
    reserved = set()
    for cell, orient in ((m1, "horizontal"), (m2, "horizontal"), (m3, "vertical")):
        other_hits = all_hits - {cell}
        segment = _choose_mirror_segment(
            (h, w),
            cell,
            orient,
            blocked | reserved | other_hits,
            reserved,
        )
        if segment is None:
            return None
        mirror_segments.append(segment)
        reserved |= segment

    forbidden = blocked | reserved
    for _ in range(random.randint(0, 2)):
        extra = _random_unused_segment(grid, forbidden, reserved)
        if extra is None:
            continue
        mirror_segments.append(extra)
        reserved |= extra
        forbidden |= extra

    for segment in mirror_segments:
        for r, c in segment:
            grid[r, c] = RED

    try:
        result = _simulate(grid, consume=True)
        static = _simulate(grid, consume=False)
    except AssertionError:
        return None
    out = result["output"]
    reflections = result["reflections"]
    traversed_broken = result["traversed_broken"]

    if len(reflections) < 3:
        return None
    if m1 not in traversed_broken:
        return None
    if np.array_equal(out, static["output"]):
        return None
    if not np.any((grid == RED) & (out == RED)):
        return None
    if not np.any(out == BLUE) or np.array_equal(grid, out):
        return None

    return _apply_random_symmetry(grid)


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _canonical_case()
        if grid is None:
            continue

        dynamic = _simulate(grid, consume=True)
        static = _simulate(grid, consume=False)
        if np.array_equal(dynamic["output"], static["output"]):
            continue
        if len(dynamic["traversed_broken"]) == 0:
            continue
        return grid
    raise RuntimeError("failed to generate a valid consumed-mirror example")
