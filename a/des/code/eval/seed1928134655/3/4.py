from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np


BLACK = 0
GREY = 5
VOTE_COLORS = (1, 2, 3, 4, 7)

# --------------- solver ---------------


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r, c in np.argwhere(mask):
        rr, cc = int(r), int(c)
        if seen[rr, cc]:
            continue
        seen[rr, cc] = True
        stack = [(rr, cc)]
        comp: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            comp.append((int(cr), int(cc)))
            for nr, nc in _neighbors4(int(cr), int(cc)):
                if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                    seen[nr, nc] = True
                    stack.append((int(nr), int(nc)))
        comps.append(comp)
    return comps


def _perimeter_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(int(c0), int(c1) + 1):
        coords.append((int(r0), int(c)))
        coords.append((int(r1), int(c)))
    for r in range(int(r0) + 1, int(r1)):
        coords.append((int(r), int(c0)))
        coords.append((int(r), int(c1)))
    return coords


def _gap_side(gap: tuple[int, int], r0: int, c0: int, r1: int, c1: int) -> str:
    gr, gc = gap
    if gr == int(r0) and int(c0) < gc < int(c1):
        return "top"
    if gr == int(r1) and int(c0) < gc < int(c1):
        return "bottom"
    if gc == int(c0) and int(r0) < gr < int(r1):
        return "left"
    if gc == int(c1) and int(r0) < gr < int(r1):
        return "right"
    raise ValueError("gap must lie on a non-corner border cell")


def _find_rectangles(grid: np.ndarray) -> list[dict]:
    grey_mask = grid == int(GREY)
    rects: list[dict] = []
    for comp in _components(grey_mask):
        rs = [r for r, _c in comp]
        cs = [c for _r, c in comp]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        if r1 - r0 < 4 or c1 - c0 < 4:
            continue

        perimeter = set(_perimeter_coords(r0, c0, r1, c1))
        coords = set(comp)
        if not coords.issubset(perimeter):
            continue

        missing = list(perimeter - coords)
        if len(missing) != 1:
            continue
        gap = missing[0]
        if int(grid[gap]) != int(BLACK):
            continue

        side = _gap_side(gap, r0, c0, r1, c1)
        rects.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "gap": (int(gap[0]), int(gap[1])),
                "side": side,
            }
        )

    rects.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return rects


def _pick_fill_color(values: Iterable[int], global_counts: Counter[int]) -> int:
    seq = [int(v) for v in values]
    local = Counter(seq)
    max_count = max(local.values())
    tied = [color for color, count in local.items() if count == max_count]
    if len(tied) == 1:
        return int(tied[0])

    rarest = min(global_counts[color] for color in tied)
    rare_candidates = {color for color in tied if global_counts[color] == rarest}
    if len(rare_candidates) == 1:
        return int(next(iter(rare_candidates)))

    for value in seq:
        if value in rare_candidates:
            return int(value)
    raise AssertionError("expected at least one tied color to remain")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each gapped grey rectangle with a voted stripe pattern.

    - The majority interior vote color sets the stripe color.
    - If local vote counts tie, choose the tied color with the fewest global
      vote pixels in the whole grid.
    - A top/bottom gap means horizontal stripes; a left/right gap means
      vertical stripes.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    global_counts = Counter(int(v) for v in g.ravel() if int(v) not in (BLACK, GREY))
    rects = _find_rectangles(g)

    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        votes = [int(v) for v in interior.ravel() if int(v) not in (BLACK, GREY)]
        assert votes, "each rectangle must contain vote pixels"

        fill_color = _pick_fill_color(votes, global_counts)
        patch = np.zeros_like(interior)
        if rect["side"] in ("top", "bottom"):
            patch[::2, :] = int(fill_color)
        else:
            patch[:, ::2] = int(fill_color)
        out[r0 + 1 : r1, c0 + 1 : c1] = patch

    return out


# --------------- generator ---------------

GRID_H_RANGE = (18, 31)
GRID_W_RANGE = (18, 31)
N_RECTS_RANGE = (2, 6)
INNER_SIDE_RANGE = (3, 7)
PLACEMENT_PAD = 2
MAX_ATTEMPTS = 400

MAJORITY_PATTERNS = (
    (2, 1),
    (2, 1, 1),
    (3, 1),
    (3, 1, 1),
    (3, 2),
)
TIE_PATTERNS = (
    (2, 2),
    (2, 2, 1),
)


def _sample_sides(rng: np.random.Generator, n_rects: int) -> list[str]:
    all_sides = np.array(["top", "bottom", "left", "right"], dtype=object)
    if n_rects >= 4:
        sides = list(rng.permutation(all_sides))
        while len(sides) < n_rects:
            sides.append(str(rng.choice(all_sides)))
    else:
        sides = [
            str(rng.choice(np.array(["top", "bottom"], dtype=object))),
            str(rng.choice(np.array(["left", "right"], dtype=object))),
        ]
        while len(sides) < n_rects:
            sides.append(str(rng.choice(all_sides)))
        sides = [sides[i] for i in rng.permutation(len(sides))]
    return [str(side) for side in sides]


def _sample_branches(rng: np.random.Generator, n_rects: int) -> list[str]:
    branches = ["majority", "tie"]
    while len(branches) < n_rects:
        branches.append(str(rng.choice(np.array(["majority", "tie"], dtype=object))))
    return [branches[i] for i in rng.permutation(len(branches))]


def _place_rectangles(
    rng: np.random.Generator,
    shapes: list[tuple[int, int]],
    grid_h: int,
    grid_w: int,
) -> list[tuple[int, int]] | None:
    order = sorted(
        range(len(shapes)),
        key=lambda idx: shapes[idx][0] * shapes[idx][1],
        reverse=True,
    )
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(shapes)

    for idx in order:
        outer_h, outer_w = shapes[idx]
        candidates: list[tuple[int, int]] = []
        for r0 in range(1, grid_h - outer_h):
            for c0 in range(1, grid_w - outer_w):
                rr0 = max(0, r0 - PLACEMENT_PAD)
                cc0 = max(0, c0 - PLACEMENT_PAD)
                rr1 = min(grid_h, r0 + outer_h + PLACEMENT_PAD)
                cc1 = min(grid_w, c0 + outer_w + PLACEMENT_PAD)
                if bool(occupied[rr0:rr1, cc0:cc1].any()):
                    continue
                candidates.append((int(r0), int(c0)))
        if not candidates:
            return None

        pick = candidates[int(rng.integers(0, len(candidates)))]
        positions[idx] = pick
        r0, c0 = pick
        rr0 = max(0, r0 - PLACEMENT_PAD)
        cc0 = max(0, c0 - PLACEMENT_PAD)
        rr1 = min(grid_h, r0 + outer_h + PLACEMENT_PAD)
        cc1 = min(grid_w, c0 + outer_w + PLACEMENT_PAD)
        occupied[rr0:rr1, cc0:cc1] = True

    return [(int(r0), int(c0)) for r0, c0 in positions if (r0, c0) is not None]


def _sample_votes(
    rng: np.random.Generator,
    branch: str,
) -> tuple[list[int], tuple[int, ...]]:
    if branch == "majority":
        counts = tuple(int(v) for v in MAJORITY_PATTERNS[int(rng.integers(0, len(MAJORITY_PATTERNS)))])
    else:
        counts = tuple(int(v) for v in TIE_PATTERNS[int(rng.integers(0, len(TIE_PATTERNS)))])

    colors = [int(v) for v in rng.choice(np.array(VOTE_COLORS), size=len(counts), replace=False)]
    votes: list[int] = []
    for color, count in zip(colors, counts):
        votes.extend([int(color)] * int(count))
    rng.shuffle(votes)
    return votes, counts


def _draw_frame(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, gap: tuple[int, int]) -> None:
    for r, c in _perimeter_coords(r0, c0, r1, c1):
        if (int(r), int(c)) != (int(gap[0]), int(gap[1])):
            grid[int(r), int(c)] = int(GREY)


def _independent_expected_output(grid: np.ndarray, specs: list[dict], global_counts: Counter[int]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        fill_color = _pick_fill_color(spec["votes"], global_counts)
        patch = np.zeros((r1 - r0 - 1, c1 - c0 - 1), dtype=int)
        if spec["side"] in ("top", "bottom"):
            patch[::2, :] = int(fill_color)
        else:
            patch[:, ::2] = int(fill_color)
        out[r0 + 1 : r1, c0 + 1 : c1] = patch
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n_rects = int(rng.integers(*N_RECTS_RANGE))
    grid_h = int(rng.integers(max(GRID_H_RANGE[0], 14 + 2 * n_rects), GRID_H_RANGE[1]))
    grid_w = int(rng.integers(max(GRID_W_RANGE[0], 14 + 2 * n_rects), GRID_W_RANGE[1]))

    inner_shapes = [
        (
            int(rng.integers(*INNER_SIDE_RANGE)),
            int(rng.integers(*INNER_SIDE_RANGE)),
        )
        for _ in range(n_rects)
    ]
    outer_shapes = [(h + 2, w + 2) for h, w in inner_shapes]
    placements = _place_rectangles(rng, outer_shapes, grid_h, grid_w)
    if placements is None:
        return None

    sides = _sample_sides(rng, n_rects)
    branches = _sample_branches(rng, n_rects)

    specs: list[dict] = []
    global_counts: Counter[int] = Counter()
    for idx in range(n_rects):
        inner_h, inner_w = inner_shapes[idx]
        r0, c0 = placements[idx]
        r1 = r0 + inner_h + 1
        c1 = c0 + inner_w + 1

        if sides[idx] == "top":
            gap = (int(r0), int(c0 + rng.integers(1, inner_w + 1)))
        elif sides[idx] == "bottom":
            gap = (int(r1), int(c0 + rng.integers(1, inner_w + 1)))
        elif sides[idx] == "left":
            gap = (int(r0 + rng.integers(1, inner_h + 1)), int(c0))
        else:
            gap = (int(r0 + rng.integers(1, inner_h + 1)), int(c1))

        votes, _counts = _sample_votes(rng, branches[idx])
        cells = [
            (int(r0 + 1 + rr), int(c0 + 1 + cc))
            for rr in range(inner_h)
            for cc in range(inner_w)
        ]
        chosen_idx = rng.choice(len(cells), size=len(votes), replace=False)
        vote_cells = [cells[int(i)] for i in chosen_idx]

        for color in votes:
            global_counts[int(color)] += 1

        specs.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "side": sides[idx],
                "gap": gap,
                "votes": [int(v) for v in votes],
                "vote_cells": vote_cells,
                "branch": branches[idx],
            }
        )

    for spec in specs:
        if spec["branch"] != "tie":
            continue
        local = Counter(spec["votes"])
        max_count = max(local.values())
        tied = [color for color, count in local.items() if count == max_count]
        rarest = min(global_counts[color] for color in tied)
        if sum(global_counts[color] == rarest for color in tied) != 1:
            return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        _draw_frame(grid, r0, c0, r1, c1, spec["gap"])
        for (vr, vc), color in zip(spec["vote_cells"], spec["votes"]):
            grid[int(vr), int(vc)] = int(color)

    expected = _independent_expected_output(grid, specs, global_counts)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid input")
