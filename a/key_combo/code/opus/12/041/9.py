from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

MARKER_COLORS = (BLUE, RED, GREEN, YELLOW)
GEN_ATTEMPTS = 500


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw Manhattan routes between same-color edge markers, then fill enclosed regions by boundary segment count.

    Input structure:
    - The grid is a black background with 2-4 marker pairs on the outer border.
    - Each marker pair uses one visible color and appears exactly twice.
    - The two markers of a pair lie on different grid edges and define one horizontal-first Manhattan route.

    Rule:
    - For each same-color pair, choose the lexicographically earlier marker `(smaller row, then smaller col)` as the route start.
    - Paint that pair's route grey: move horizontally from the start to the target column, then vertically to the target row; both markers also become grey.
    - After all routes are drawn, find every maximal 4-connected black region that does not touch the grid border.
    - Count how many independent grey boundary segments touch each enclosed region, where each maximal straight horizontal or vertical boundary run counts as one segment; fill the region with red for 2, green for 3, yellow for 4, and blue for 5 or more.

    Key insight:
    - The edge markers do not directly encode the final fill colors: they create a grey line network first, and the topology of the enclosed black regions determines the output through boundary-segment counting.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30

    pairs = _extract_marker_pairs(g)
    routed = _draw_routes(g.shape, pairs)
    line_mask = routed == GREY

    out = routed.copy()
    region_infos = _enclosed_black_regions(routed)
    assert region_infos, "expected at least one enclosed region"
    for info in region_infos:
        region_mask = info["mask"]
        segments = _count_boundary_segments(line_mask, region_mask)
        assert segments >= 2
        out[region_mask] = _fill_color_for_segment_count(segments)
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_candidate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")


def _extract_marker_pairs(grid: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    h, w = grid.shape
    colors = [int(color) for color in np.unique(grid) if int(color) != BLACK]
    assert 2 <= len(colors) <= 4

    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for color in colors:
        coords = np.argwhere(grid == color)
        assert coords.shape == (2, 2), f"color {color} must appear exactly twice"
        p1 = (int(coords[0, 0]), int(coords[0, 1]))
        p2 = (int(coords[1, 0]), int(coords[1, 1]))
        assert _is_border_cell(p1, h, w) and _is_border_cell(p2, h, w)
        assert not _same_edge(p1, p2, h, w), "paired markers must lie on different edges"
        pairs.append((p1, p2))
    return pairs


def _is_border_cell(pos: tuple[int, int], h: int, w: int) -> bool:
    r, c = pos
    return r == 0 or r == h - 1 or c == 0 or c == w - 1


def _same_edge(a: tuple[int, int], b: tuple[int, int], h: int, w: int) -> bool:
    ar, ac = a
    br, bc = b
    return (
        (ar == 0 and br == 0)
        or (ar == h - 1 and br == h - 1)
        or (ac == 0 and bc == 0)
        or (ac == w - 1 and bc == w - 1)
    )


def _draw_routes(shape: tuple[int, int], pairs: Iterable[tuple[tuple[int, int], tuple[int, int]]]) -> np.ndarray:
    routed = np.zeros(shape, dtype=int)
    for p1, p2 in pairs:
        _paint_route(routed, p1, p2)
    return routed


def _paint_route(grid: np.ndarray, p1: tuple[int, int], p2: tuple[int, int]) -> None:
    (r1, c1), (r2, c2) = sorted((p1, p2))
    c0, c1b = sorted((c1, c2))
    r0, r1b = sorted((r1, r2))
    grid[r1, c0 : c1b + 1] = GREY
    grid[r0 : r1b + 1, c2] = GREY


def _enclosed_black_regions(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    regions: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != BLACK or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            touches_border = False
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                if cr == 0 or cr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == BLACK and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if touches_border:
                continue
            mask = np.zeros((h, w), dtype=bool)
            for rr, cc in coords:
                mask[rr, cc] = True
            regions.append({"mask": mask, "area": len(coords)})
    return regions


def _count_runs(values: np.ndarray) -> int:
    count = 0
    in_run = False
    for value in values:
        if bool(value):
            if not in_run:
                count += 1
                in_run = True
        else:
            in_run = False
    return count


def _count_boundary_segments(line_mask: np.ndarray, region_mask: np.ndarray) -> int:
    h, w = region_mask.shape
    total = 0

    for c in range(w):
        if c > 0:
            total += _count_runs(region_mask[:, c] & line_mask[:, c - 1])
        if c + 1 < w:
            total += _count_runs(region_mask[:, c] & line_mask[:, c + 1])

    for r in range(h):
        if r > 0:
            total += _count_runs(region_mask[r, :] & line_mask[r - 1, :])
        if r + 1 < h:
            total += _count_runs(region_mask[r, :] & line_mask[r + 1, :])

    return total


def _fill_color_for_segment_count(count: int) -> int:
    if count == 2:
        return RED
    if count == 3:
        return GREEN
    if count == 4:
        return YELLOW
    return BLUE


def _generate_candidate(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(14, 18)
    w = rng.randint(14, 18)

    c_tb_max = min(4, w - 8)
    if c_tb_max < 2:
        return None
    c_tb = rng.randint(2, c_tb_max)
    c_rb = rng.randint(c_tb + 3, w - 5)
    c_lb = rng.randint(c_rb + 2, w - 3)

    r_lb = rng.randint(2, 4)
    r_rb_hi = min(r_lb + 5, h - 7)
    if r_rb_hi < r_lb + 2:
        return None
    r_rb = rng.randint(r_lb + 2, r_rb_hi)
    r_h = rng.randint(r_rb + 2, h - 4)

    lr_from_left = rng.random() < 0.5
    if lr_from_left:
        other_hi = min(h - 2, r_h + 3)
        if other_hi < r_h + 1:
            return None
        r_other = rng.randint(r_h + 1, other_hi)
        lr_pair = ((r_h, 0), (r_other, w - 1))
    else:
        other_lo = max(1, r_h - 3)
        if other_lo > r_h - 1:
            return None
        r_other = rng.randint(other_lo, r_h - 1)
        lr_pair = ((r_other, 0), (r_h, w - 1))

    pairs = [
        ((0, c_tb), (h - 1, c_tb)),
        lr_pair,
        ((r_lb, 0), (h - 1, c_lb)),
        ((r_rb, w - 1), (h - 1, c_rb)),
    ]

    if rng.random() < 0.5:
        pairs = [(_mirror_h(p1, w), _mirror_h(p2, w)) for p1, p2 in pairs]

    colors = list(MARKER_COLORS)
    rng.shuffle(colors)
    rng.shuffle(pairs)

    grid = np.zeros((h, w), dtype=int)
    for color, (p1, p2) in zip(colors, pairs):
        if p1 == p2 or grid[p1] != BLACK or grid[p2] != BLACK:
            return None
        grid[p1] = color
        grid[p2] = color

    if not _candidate_is_semantically_valid(grid):
        return None
    return grid


def _mirror_h(pos: tuple[int, int], width: int) -> tuple[int, int]:
    r, c = pos
    return r, width - 1 - c


def _candidate_is_semantically_valid(grid: np.ndarray) -> bool:
    try:
        pairs = _extract_marker_pairs(grid)
        routed = _draw_routes(grid.shape, pairs)
        infos = _enclosed_black_regions(routed)
        output = main(grid)
    except AssertionError:
        return False

    if len(infos) < 2:
        return False

    segment_counts = []
    areas = []
    line_mask = routed == GREY
    for info in infos:
        region_mask = info["mask"]
        count = _count_boundary_segments(line_mask, region_mask)
        segment_counts.append(count)
        areas.append(int(info["area"]))

    if len(set(segment_counts)) < 2:
        return False
    if min(segment_counts) < 4:
        return False
    if not any(count == 4 for count in segment_counts):
        return False
    if not any(count >= 5 for count in segment_counts):
        return False
    if min(areas) < 4:
        return False
    if np.count_nonzero(routed == GREY) > int(0.45 * grid.size):
        return False
    if np.array_equal(output, routed):
        return False

    return True
