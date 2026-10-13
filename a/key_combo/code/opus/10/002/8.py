from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import random

import numpy as np


BLACK = 0
SQUARE_COLORS = (1, 2, 3)
MARKER_COLORS = (4, 6, 7, 8, 9)
DIAGONAL_RAYS = (
    (0, 0, -1, -1),
    (0, 1, -1, 1),
    (1, 0, 1, -1),
    (1, 1, 1, 1),
)
GEN_ATTEMPTS = 400


@dataclass(frozen=True)
class Square:
    color: int
    top: int
    left: int

    @property
    def bottom(self) -> int:
        return self.top + 1

    @property
    def right(self) -> int:
        return self.left + 1

    @property
    def cells(self) -> set[tuple[int, int]]:
        return {
            (self.top, self.left),
            (self.top, self.right),
            (self.bottom, self.left),
            (self.bottom, self.right),
        }


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _find_squares_and_markers(grid: np.ndarray) -> tuple[list[Square], dict[tuple[int, int], int]]:
    g = np.asarray(grid, dtype=int)
    squares: list[Square] = []
    markers: dict[tuple[int, int], int] = {}

    for color in sorted(int(v) for v in np.unique(g) if int(v) != BLACK):
        for comp in _components(g == color):
            if len(comp) == 1:
                markers[comp[0]] = color
                continue

            r0, c0, r1, c1 = _bbox(comp)
            if len(comp) == 4 and (r1 - r0 + 1, c1 - c0 + 1) == (2, 2):
                if set(comp) == {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
                    squares.append(Square(color=color, top=r0, left=c0))
                    continue
            raise AssertionError("unexpected foreground object")

    squares.sort(key=lambda sq: (sq.top, sq.left, sq.color))
    assert 2 <= len(squares) <= 3
    assert len({sq.color for sq in squares}) == len(squares)
    return squares, markers


def _square_mask(shape: tuple[int, int], squares: list[Square]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for sq in squares:
        mask[sq.top : sq.top + 2, sq.left : sq.left + 2] = True
    return mask


def _emit_rays(grid: np.ndarray, squares: list[Square]) -> tuple[np.ndarray, np.ndarray]:
    out = np.asarray(grid, dtype=int).copy()
    ray_mask = np.zeros_like(out, dtype=bool)
    for sq in squares:
        for dr0, dc0, dr, dc in DIAGONAL_RAYS:
            r = sq.top + dr0 + dr
            c = sq.left + dc0 + dc
            while in_bounds(out.shape, r, c):
                if out[r, c] == BLACK:
                    out[r, c] = sq.color
                    ray_mask[r, c] = True
                r += dr
                c += dc
    return out, ray_mask


def _region_trace(
    base_grid: np.ndarray,
    barrier_mask: np.ndarray,
    marker_map: dict[tuple[int, int], int],
    *,
    tie_mode: str = "lex",
) -> tuple[np.ndarray, list[dict]]:
    out = np.asarray(base_grid, dtype=int).copy()
    h, w = out.shape
    passable = ~barrier_mask
    marker_positions = set(marker_map)
    seen = np.zeros((h, w), dtype=bool)
    regions: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not passable[r, c] or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            black_cells: list[tuple[int, int]] = []
            markers: list[tuple[int, int]] = []

            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                if (rr, cc) in marker_positions:
                    markers.append((rr, cc))
                else:
                    black_cells.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(out.shape, nr, nc) and passable[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            boundary_cells = [
                cell
                for cell in cells
                if any(
                    not in_bounds(out.shape, nr, nc) or barrier_mask[nr, nc]
                    for nr, nc in neighbors4(cell[0], cell[1])
                )
            ]
            counts = Counter(marker_map[cell] for cell in markers)
            winner = None
            tied_colors: tuple[int, ...] = ()
            is_tie = False
            if counts:
                max_count = max(counts.values())
                tied_colors = tuple(sorted(color for color, cnt in counts.items() if cnt == max_count))
                is_tie = len(tied_colors) > 1
                if tie_mode == "lex":
                    for cell in sorted(markers):
                        if marker_map[cell] in tied_colors:
                            winner = marker_map[cell]
                            break
                elif tie_mode == "min_color":
                    winner = min(tied_colors)
                else:
                    raise ValueError(f"unknown tie mode: {tie_mode}")
                for rr, cc in black_cells:
                    out[rr, cc] = winner

            regions.append(
                {
                    "cells": cells,
                    "black_cells": black_cells,
                    "markers": markers,
                    "boundary_cells": boundary_cells,
                    "counts": counts,
                    "winner": winner,
                    "tied_colors": tied_colors,
                    "is_tie": is_tie,
                }
            )

    return out, regions


def _solve_trace(grid: np.ndarray, *, tie_mode: str = "lex") -> dict:
    g = np.asarray(grid, dtype=int)
    squares, markers = _find_squares_and_markers(g)
    ray_grid, ray_mask = _emit_rays(g, squares)
    square_mask = _square_mask(g.shape, squares)
    barrier = square_mask | ray_mask
    out, regions = _region_trace(ray_grid, barrier, markers, tie_mode=tie_mode)
    return {
        "output": out,
        "squares": squares,
        "markers": markers,
        "ray_grid": ray_grid,
        "ray_mask": ray_mask,
        "square_mask": square_mask,
        "barrier": barrier,
        "regions": regions,
    }


def _global_fill(ray_grid: np.ndarray, barrier_mask: np.ndarray, marker_map: dict[tuple[int, int], int]) -> np.ndarray:
    out = ray_grid.copy()
    if not marker_map:
        return out
    counts = Counter(marker_map.values())
    max_count = max(counts.values())
    tied = {color for color, cnt in counts.items() if cnt == max_count}
    winner = next(marker_map[cell] for cell in sorted(marker_map) if marker_map[cell] in tied)
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            if not barrier_mask[r, c] and (r, c) not in marker_map:
                out[r, c] = winner
    return out


def _solve_without_rays(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    squares, markers = _find_squares_and_markers(g)
    square_mask = _square_mask(g.shape, squares)
    out, _ = _region_trace(g, square_mask, markers, tie_mode="lex")
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Partition the canvas with four-corner diagonal rays and fill each open region by marker vote.

    Input structure:
    - The grid has a black background, 2-3 distinct solid 2x2 squares, and several single-pixel colored markers.
    - Square colors and marker colors are disjoint.
    - Markers lie in the open space around the squares.

    Rule:
    - From each square corner, extend one diagonal ray outward to the grid edge in the matching diagonal direction.
    - Rays use the square color and only paint black cells; they pass through existing non-black cells without stopping.
    - Treat all square cells and painted ray cells as barriers, then 4-connected flood-fill the remaining black cells together with the marker cells into open regions.
    - In each region, count the marker colors inside that region and fill the region's black cells with the majority marker color; if the top count ties, choose the tied color whose marker appears first in row-major order within that region; regions with no markers stay black.

    Key insight:
    - The squares determine a diagonal barrier network, while the markers determine only the color of each barrier-defined region, so both the geometry and the local marker counts are necessary.
    """
    return _solve_trace(np.asarray(grid, dtype=int))["output"]


def _ray_cells_for_square(square: Square, shape: tuple[int, int]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for dr0, dc0, dr, dc in DIAGONAL_RAYS:
        r = square.top + dr0 + dr
        c = square.left + dc0 + dc
        while in_bounds(shape, r, c):
            cells.add((r, c))
            r += dr
            c += dc
    return cells


def _square_gap(a: Square, b: Square) -> tuple[int, int]:
    row_gap = max(0, max(a.top - (b.bottom + 1), b.top - (a.bottom + 1)))
    col_gap = max(0, max(a.left - (b.right + 1), b.left - (a.right + 1)))
    return row_gap, col_gap


def _squares_compatible(existing: list[Square], candidate: Square, shape: tuple[int, int]) -> bool:
    if candidate.top < 3 or candidate.left < 3:
        return False
    if candidate.bottom > shape[0] - 4 or candidate.right > shape[1] - 4:
        return False

    candidate_ray_cells = _ray_cells_for_square(candidate, shape)
    candidate_cells = candidate.cells
    for sq in existing:
        row_gap, col_gap = _square_gap(sq, candidate)
        if row_gap + col_gap < 4 or max(row_gap, col_gap) < 2:
            return False
        if sq.cells & candidate_cells:
            return False
        if sq.cells & candidate_ray_cells:
            return False
        if candidate_cells & _ray_cells_for_square(sq, shape):
            return False
    return True


def _sample_squares(h: int, w: int) -> list[Square] | None:
    count = random.choices([2, 3], weights=[3, 5])[0]
    colors = random.sample(list(SQUARE_COLORS), count)
    positions = [(r, c) for r in range(3, h - 4) for c in range(3, w - 4)]
    random.shuffle(positions)
    squares: list[Square] = []

    for color in colors:
        placed = False
        random.shuffle(positions)
        for r, c in positions:
            candidate = Square(color=color, top=r, left=c)
            if _squares_compatible(squares, candidate, (h, w)):
                squares.append(candidate)
                placed = True
                break
        if not placed:
            return None
    return squares


def _render_squares(shape: tuple[int, int], squares: list[Square]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for sq in squares:
        grid[sq.top : sq.top + 2, sq.left : sq.left + 2] = sq.color
    return grid


def _pick_marker_cells(region: dict, k: int) -> list[tuple[int, int]] | None:
    boundary = sorted(set(region["boundary_cells"]))
    all_cells = sorted(set(region["black_cells"]))
    if len(all_cells) < k:
        return None

    def far_enough(cell: tuple[int, int], chosen: list[tuple[int, int]]) -> bool:
        return all(abs(cell[0] - r) + abs(cell[1] - c) >= 2 for r, c in chosen)

    for _ in range(80):
        first = boundary[:]
        second = [cell for cell in all_cells if cell not in boundary]
        random.shuffle(first)
        random.shuffle(second)
        chosen: list[tuple[int, int]] = []
        for cell in first + second:
            if far_enough(cell, chosen):
                chosen.append(cell)
                if len(chosen) == k:
                    return sorted(chosen)
    return None


def _place_region_markers(grid: np.ndarray, region: dict, color_order: list[int]) -> bool:
    cells = _pick_marker_cells(region, len(color_order))
    if cells is None:
        return False
    for cell, color in zip(cells, color_order):
        grid[cell] = color
    return True


def _region_color_order(plan: dict) -> list[int]:
    order: list[int] = []
    earliest = plan["earliest"]
    counts = plan["counts"].copy()
    assert counts[earliest] > 0
    order.append(earliest)
    counts[earliest] -= 1
    remainder: list[int] = []
    for color, count in counts.items():
        remainder.extend([color] * count)
    random.shuffle(remainder)
    order.extend(remainder)
    return order


def _plan_templates() -> tuple[list[dict], bool, bool]:
    use_tie = random.random() < 0.6
    keep_empty_region = random.random() < 0.45

    if use_tie:
        a, b = random.sample(list(MARKER_COLORS), 2)
        tie_winner = max(a, b) if random.random() < 0.7 else min(a, b)
        return (
            [
                {"counts": Counter({a: 2, b: 1}), "winner": a, "earliest": b},
                {"counts": Counter({a: 1, b: 2}), "winner": b, "earliest": a},
                {"counts": Counter({a: 1, b: 1}), "winner": tie_winner, "earliest": tie_winner},
            ],
            use_tie,
            keep_empty_region,
        )

    if random.random() < 0.5:
        a, b = random.sample(list(MARKER_COLORS), 2)
        return (
            [
                {"counts": Counter({a: 2, b: 1}), "winner": a, "earliest": b},
                {"counts": Counter({a: 1, b: 2}), "winner": b, "earliest": a},
                {"counts": Counter({a: 2}), "winner": a, "earliest": a},
            ],
            use_tie,
            keep_empty_region,
        )

    a, b, c = random.sample(list(MARKER_COLORS), 3)
    return (
        [
            {"counts": Counter({a: 2, b: 1}), "winner": a, "earliest": b},
            {"counts": Counter({b: 2, c: 1}), "winner": b, "earliest": c},
            {"counts": Counter({c: 2, a: 1}), "winner": c, "earliest": a},
        ],
        use_tie,
        keep_empty_region,
    )


def _apply_marker_plan(base_grid: np.ndarray, layout_regions: list[dict]) -> tuple[np.ndarray, dict] | None:
    plans, use_tie, keep_empty_region = _plan_templates()
    need_sizes = sorted((sum(plan["counts"].values()) for plan in plans), reverse=True)
    region_indices = list(range(len(layout_regions)))
    random.shuffle(region_indices)

    for _ in range(60):
        chosen = random.sample(region_indices, len(plans))
        chosen.sort(key=lambda idx: len(layout_regions[idx]["black_cells"]), reverse=True)
        if any(len(layout_regions[idx]["black_cells"]) < need for idx, need in zip(chosen, need_sizes)):
            continue

        empty_idx = None
        if keep_empty_region:
            empties = [
                idx
                for idx in region_indices
                if idx not in chosen and len(layout_regions[idx]["black_cells"]) >= 3
            ]
            if not empties:
                continue
            empty_idx = random.choice(empties)

        grid = base_grid.copy()
        ok = True
        plan_order = sorted(plans, key=lambda plan: sum(plan["counts"].values()), reverse=True)
        for plan, region_idx in zip(plan_order, chosen):
            color_order = _region_color_order(plan)
            if not _place_region_markers(grid, layout_regions[region_idx], color_order):
                ok = False
                break
        if not ok:
            continue

        info = {
            "expect_tie": use_tie,
            "keep_empty_region": keep_empty_region,
            "empty_region": empty_idx,
        }
        return grid, info
    return None


def _validate_instance(grid: np.ndarray, expect_tie: bool, keep_empty_region: bool) -> bool:
    try:
        trace = _solve_trace(grid, tie_mode="lex")
    except AssertionError:
        return False

    squares = trace["squares"]
    markers = trace["markers"]
    if not (2 <= len(squares) <= 3):
        return False
    if len(markers) < 6 or len(markers) > 10:
        return False

    marker_counts = Counter(markers.values())
    if len(marker_counts) < 2:
        return False
    if any(count < 2 for count in marker_counts.values()):
        return False
    if set(marker_counts) & {sq.color for sq in squares}:
        return False

    occupied = [region for region in trace["regions"] if region["markers"]]
    if len(occupied) < 3:
        return False
    winners = {region["winner"] for region in occupied if region["winner"] is not None}
    if len(winners) < 2:
        return False

    count_needed = any(
        (not region["is_tie"]) and region["markers"] and trace["markers"][sorted(region["markers"])[0]] != region["winner"]
        for region in occupied
    )
    if not count_needed:
        return False

    if expect_tie:
        tie_regions = [region for region in occupied if region["is_tie"]]
        if not tie_regions:
            return False
        if not any(region["winner"] != min(region["tied_colors"]) for region in tie_regions):
            return False
        wrong_tie = _solve_trace(grid, tie_mode="min_color")["output"]
        if np.array_equal(wrong_tie, trace["output"]):
            return False

    if keep_empty_region:
        if not any((not region["markers"]) and len(region["black_cells"]) >= 2 for region in trace["regions"]):
            return False

    if np.array_equal(trace["output"], grid):
        return False
    if np.array_equal(_solve_without_rays(grid), trace["output"]):
        return False
    if np.array_equal(_global_fill(trace["ray_grid"], trace["barrier"], markers), trace["output"]):
        return False

    changed = int(np.count_nonzero(trace["output"] != trace["ray_grid"]))
    if changed < 6:
        return False
    if np.count_nonzero(trace["output"] == BLACK) < 2:
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(15, 20)
    w = random.randint(15, 20)
    squares = _sample_squares(h, w)
    if squares is None:
        return None

    base_grid = _render_squares((h, w), squares)
    base_trace = _solve_trace(base_grid)
    layout_regions = base_trace["regions"]
    if len(layout_regions) < 4:
        return None
    if sum(len(region["black_cells"]) >= 3 for region in layout_regions) < 4:
        return None

    planned = _apply_marker_plan(base_grid, layout_regions)
    if planned is None:
        return None
    grid, info = planned

    if not _validate_instance(grid, info["expect_tie"], info["keep_empty_region"]):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
