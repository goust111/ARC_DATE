from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
GREY = 5
MARKER_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_HEIGHT_RANGE = (14, 30)
GRID_WIDTH_RANGE = (14, 30)
REGION_COUNT_RANGE = (1, 4)
REGION_OUTER_SIZE_MAX = 13
PLACEMENT_PADDING = 2
REGION_PLACEMENT_TRIES = 100
CELL_SAMPLING_TRIES = 40
GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(height):
        for c in range(width):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and bool(mask[nr, nc])
                        and not bool(seen[nr, nc])
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)

    return comps


def _paint_rect_border(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _rect_border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for comp in _components(grid == GREY):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must enclose a nonempty interior"
        assert set(comp) == _rect_border_cells(r0, c0, r1, c1), "grey component must be a 1-cell frame"
        frames.append((r0, c0, r1, c1))
    assert frames, "expected at least one grey rectangular frame"
    frames.sort()
    return frames


def _majority_marker(interior: np.ndarray) -> tuple[int, int]:
    markers = [int(v) for v in interior.reshape(-1) if int(v) not in (BLACK, GREY)]
    counts = Counter(markers)
    assert 2 <= len(counts) <= 4, "each region must contain 2-4 marker colors"
    top_count = max(counts.values())
    winners = [color for color, count in counts.items() if count == top_count]
    assert len(winners) == 1, "expected a unique majority marker color"
    return winners[0], top_count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Encode each region's marker-frequency majority into inward wall thickness.

    Input structure:
    - Background is black.
    - One or more grey 1-pixel-thick rectangular frames enclose regions.
    - Inside each region are scattered single-pixel colored markers.

    Rule:
    - Count marker pixels inside each frame by color.
    - Find the unique most frequent color C with count N.
    - Clear all markers to black.
    - Keep the original grey outer wall, then expand the wall inward to total
      thickness N by drawing the added inner layers in color C.
    - Any space still deeper than the thickened wall remains black.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = _find_frames(g)
    covered = np.zeros_like(g, dtype=bool)
    for r0, c0, r1, c1 in frames:
        covered[r0 + 1 : r1, c0 + 1 : c1] = True

    marker_mask = (g != BLACK) & (g != GREY)
    assert np.all(covered[marker_mask]), "all markers must lie inside a frame interior"

    out = g.copy()
    for r0, c0, r1, c1 in frames:
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        draw_color, thickness = _majority_marker(interior)

        out[r0 + 1 : r1, c0 + 1 : c1] = BLACK
        for inset in range(1, thickness):
            rr0, cc0 = r0 + inset, c0 + inset
            rr1, cc1 = r1 - inset, c1 - inset
            if rr0 > rr1 or cc0 > cc1:
                break
            _paint_rect_border(out, rr0, cc0, rr1, cc1, draw_color)

    return out


def _sample_marker_counts() -> tuple[dict[int, int], int, int]:
    color_count = random.randint(2, 4)
    colors = random.sample(MARKER_COLORS, color_count)
    majority_color = random.choice(colors)
    majority_count = random.choice([2, 3])

    counts: dict[int, int] = {}
    for color in colors:
        if color == majority_color:
            counts[color] = majority_count
        else:
            counts[color] = random.randint(1, majority_count - 1)

    total_markers = sum(counts.values())
    return counts, majority_color, total_markers


def _choose_sparse_cells(
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    needed: int,
) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    for _ in range(CELL_SAMPLING_TRIES):
        random.shuffle(cells)
        chosen: list[tuple[int, int]] = []
        for cell in cells:
            if all(max(abs(cell[0] - rr), abs(cell[1] - cc)) > 1 for rr, cc in chosen):
                chosen.append(cell)
                if len(chosen) == needed:
                    return chosen
    return None


def _can_place(used: np.ndarray, r0: int, c0: int, height: int, width: int) -> bool:
    h, w = used.shape
    rr0 = max(0, r0 - PLACEMENT_PADDING)
    cc0 = max(0, c0 - PLACEMENT_PADDING)
    rr1 = min(h, r0 + height + PLACEMENT_PADDING)
    cc1 = min(w, c0 + width + PLACEMENT_PADDING)
    return not bool(np.any(used[rr0:rr1, cc0:cc1]))


def _mark_used(used: np.ndarray, r0: int, c0: int, height: int, width: int) -> None:
    h, w = used.shape
    rr0 = max(0, r0 - PLACEMENT_PADDING)
    cc0 = max(0, c0 - PLACEMENT_PADDING)
    rr1 = min(h, r0 + height + PLACEMENT_PADDING)
    cc1 = min(w, c0 + width + PLACEMENT_PADDING)
    used[rr0:rr1, cc0:cc1] = True


def _apply_region_rule(
    grid: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    color: int,
    thickness: int,
) -> None:
    grid[r0 + 1 : r1, c0 + 1 : c1] = BLACK
    for inset in range(1, thickness):
        _paint_rect_border(grid, r0 + inset, c0 + inset, r1 - inset, c1 - inset, color)


def _gen_attempt() -> np.ndarray | None:
    region_count = random.randint(*REGION_COUNT_RANGE)
    min_side = max(GRID_HEIGHT_RANGE[0], 10 + 4 * region_count)
    height = random.randint(min_side, GRID_HEIGHT_RANGE[1])
    width = random.randint(min_side, GRID_WIDTH_RANGE[1])

    inp = np.full((height, width), BLACK, dtype=int)
    expected = inp.copy()
    used = np.zeros((height, width), dtype=bool)

    for _ in range(region_count):
        counts, majority_color, total_markers = _sample_marker_counts()
        thickness = counts[majority_color]
        min_outer = 2 * thickness + 1

        placed = False
        for _ in range(REGION_PLACEMENT_TRIES):
            outer_h = random.randint(min_outer, REGION_OUTER_SIZE_MAX)
            outer_w = random.randint(min_outer, REGION_OUTER_SIZE_MAX)
            if outer_h >= height or outer_w >= width:
                continue

            r0 = random.randint(0, height - outer_h)
            c0 = random.randint(0, width - outer_w)
            if not _can_place(used, r0, c0, outer_h, outer_w):
                continue

            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1
            marker_cells = _choose_sparse_cells(r0, c0, r1, c1, total_markers)
            if marker_cells is None:
                continue

            _paint_rect_border(inp, r0, c0, r1, c1, GREY)
            _paint_rect_border(expected, r0, c0, r1, c1, GREY)

            marker_values: list[int] = []
            for color, count in counts.items():
                marker_values.extend([color] * count)
            random.shuffle(marker_values)
            for (rr, cc), color in zip(marker_cells, marker_values):
                inp[rr, cc] = color

            _apply_region_rule(expected, r0, c0, r1, c1, majority_color, thickness)
            _mark_used(used, r0, c0, outer_h, outer_w)
            placed = True
            break

        if not placed:
            return None

    if np.array_equal(main(inp), expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
