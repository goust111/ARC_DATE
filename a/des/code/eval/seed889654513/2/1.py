from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Band:
    color: int
    r0: int
    c0: int
    r1: int
    c1: int

    @property
    def height(self) -> int:
        return self.r1 - self.r0 + 1

    @property
    def width(self) -> int:
        return self.c1 - self.c0 + 1


LEGEND_SIZE = 4
LEGEND_INNER = 2
LEGEND_BORDER_COLOR = 5
MAX_GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _find_legend(grid: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], int, int, int]:
    h, w = grid.shape
    candidates: list[tuple[tuple[int, int, int, int], int, int, int]] = []
    for r0 in range(h - LEGEND_SIZE + 1):
        for c0 in range(w - LEGEND_SIZE + 1):
            if r0 not in (0, h - LEGEND_SIZE) and c0 not in (0, w - LEGEND_SIZE):
                continue
            sub = grid[r0 : r0 + LEGEND_SIZE, c0 : c0 + LEGEND_SIZE]
            frame = int(sub[0, 0])
            if frame == bg:
                continue
            if not np.all(sub[0, :] == frame):
                continue
            if not np.all(sub[-1, :] == frame):
                continue
            if not np.all(sub[:, 0] == frame):
                continue
            if not np.all(sub[:, -1] == frame):
                continue
            inner = sub[1:3, 1:3]
            top_color = int(inner[0, 0])
            under_color = int(inner[0, 1])
            if top_color == under_color:
                continue
            if int(inner[1, 0]) != under_color or int(inner[1, 1]) != top_color:
                continue
            candidates.append(((r0, c0, r0 + 3, c0 + 3), frame, top_color, under_color))
    assert len(candidates) == 1, "expected exactly one edge legend"
    return candidates[0]


def _collect_bands(
    grid: np.ndarray, bg: int, excluded: set[int]
) -> tuple[list[Band], list[Band]]:
    horizontals: list[Band] = []
    verticals: list[Band] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg or color in excluded:
            continue
        coords = np.argwhere(grid == color)
        assert coords.size > 0
        r0, c0, r1, c1 = _bbox(coords)
        band = Band(color=color, r0=r0, c0=c0, r1=r1, c1=c1)
        if band.height <= 2 and band.width >= 3:
            horizontals.append(band)
        elif band.width <= 2 and band.height >= 3:
            verticals.append(band)
        else:
            raise AssertionError(f"unrecognized band shape for color {color}")
    assert horizontals and verticals, "expected both horizontal and vertical bands"
    return sorted(horizontals, key=lambda b: (b.r0, b.c0, b.color)), sorted(
        verticals, key=lambda b: (b.c0, b.r0, b.color)
    )


def _band_intersections(horizontals: list[Band], verticals: list[Band]) -> list[tuple[int, int, int, int, int, int]]:
    intersections: list[tuple[int, int, int, int, int, int]] = []
    for h_band in horizontals:
        for v_band in verticals:
            r0 = max(h_band.r0, v_band.r0)
            r1 = min(h_band.r1, v_band.r1)
            c0 = max(h_band.c0, v_band.c0)
            c1 = min(h_band.c1, v_band.c1)
            if r0 <= r1 and c0 <= c1:
                intersections.append((r0, c0, r1, c1, h_band.color, v_band.color))
    intersections.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    assert len(intersections) >= 2, "need at least two intersections for odd/even alternation"
    return intersections


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resolve alternating weaving crossings and erase the edge legend.

    The input contains horizontal and vertical color bands. Their overlap rectangles
    are marked by the two legend colors from the framed 2x2 checker on the grid edge.
    Crossings are ordered in row-major order. Odd crossings show the horizontal band;
    even crossings show the vertical band.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    legend_bbox, frame_color, horizontal_top_color, vertical_top_color = _find_legend(g, bg)

    excluded = {frame_color, horizontal_top_color, vertical_top_color}
    horizontals, verticals = _collect_bands(g, bg, excluded)
    intersections = _band_intersections(horizontals, verticals)

    out = g.copy()
    lr0, lc0, lr1, lc1 = legend_bbox
    out[lr0 : lr1 + 1, lc0 : lc1 + 1] = bg

    for index, (r0, c0, r1, c1, h_color, v_color) in enumerate(intersections, start=1):
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        allowed = {horizontal_top_color, vertical_top_color, h_color, v_color}
        assert set(np.unique(patch)).issubset(allowed), "unexpected crossing colors"
        out[r0 : r1 + 1, c0 : c1 + 1] = h_color if index % 2 == 1 else v_color

    return out


def _pick_legend_position(
    rng: random.Random,
    height: int,
    width: int,
    weave_top: int,
    weave_bottom: int,
    weave_left: int,
    weave_right: int,
) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    for r0 in range(height - LEGEND_SIZE + 1):
        for c0 in range(width - LEGEND_SIZE + 1):
            if r0 not in (0, height - LEGEND_SIZE) and c0 not in (0, width - LEGEND_SIZE):
                continue
            r1 = r0 + LEGEND_SIZE - 1
            c1 = c0 + LEGEND_SIZE - 1
            if r1 < weave_top - 1 or r0 > weave_bottom + 1 or c1 < weave_left - 1 or c0 > weave_right + 1:
                candidates.append((r0, c0))
    if not candidates:
        raise ValueError("no room for edge legend")
    return rng.choice(candidates)


def _sample_tracks(
    rng: random.Random,
    count: int,
    total: int,
    gap_min: int,
    gap_max: int,
) -> list[tuple[int, int]]:
    widths = [rng.choice((1, 2)) for _ in range(count)]
    min_needed = sum(widths) + gap_min * (count - 1)
    if min_needed > total:
        raise ValueError("axis span too small")
    extra = total - min_needed
    gaps = [gap_min] * (count - 1)
    while extra > 0:
        idx = rng.randrange(count - 1) if count > 1 else 0
        if count == 1:
            break
        if gaps[idx] < gap_max:
            gaps[idx] += 1
            extra -= 1
        elif all(g == gap_max for g in gaps):
            break
    start = 0
    spans: list[tuple[int, int]] = []
    for i, width in enumerate(widths):
        spans.append((start, start + width - 1))
        start += width
        if i < count - 1:
            start += gaps[i]
    return spans


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    band_count_options = [(2, 2), (2, 3), (3, 2), (3, 3)]
    h_count, v_count = rng.choice(band_count_options)

    height = rng.randint(14, 22)
    width = rng.randint(14, 22)
    bg_choices = [color for color in range(10) if color != LEGEND_BORDER_COLOR]
    bg = rng.choice(bg_choices)

    available = [color for color in range(10) if color not in (bg, LEGEND_BORDER_COLOR)]
    if len(available) < h_count + v_count + 2:
        return None
    rng.shuffle(available)
    horizontal_marker = available.pop()
    vertical_marker = available.pop()
    horizontal_colors = [available.pop() for _ in range(h_count)]
    vertical_colors = [available.pop() for _ in range(v_count)]

    row_total = rng.randint(max(4, 2 * h_count), min(8, height - 8))
    col_total = rng.randint(max(4, 2 * v_count), min(8, width - 8))
    try:
        row_tracks = _sample_tracks(rng, h_count, row_total, gap_min=1, gap_max=3)
        col_tracks = _sample_tracks(rng, v_count, col_total, gap_min=1, gap_max=3)
    except ValueError:
        return None

    top_tail = rng.randint(1, 2)
    bottom_tail = rng.randint(1, 2)
    left_tail = rng.randint(1, 2)
    right_tail = rng.randint(1, 2)
    total_weave_height = row_total + top_tail + bottom_tail
    total_weave_width = col_total + left_tail + right_tail

    weave_top = rng.randint(2, height - total_weave_height - 2)
    weave_left = rng.randint(2, width - total_weave_width - 2)
    weave_bottom = weave_top + total_weave_height - 1
    weave_right = weave_left + total_weave_width - 1
    cross_top = weave_top + top_tail
    cross_left = weave_left + left_tail

    try:
        legend_r0, legend_c0 = _pick_legend_position(
            rng, height, width, weave_top, weave_bottom, weave_left, weave_right
        )
    except ValueError:
        return None

    grid = np.full((height, width), bg, dtype=int)

    horizontal_bands: list[Band] = []
    for idx, (r0_local, r1_local) in enumerate(row_tracks):
        band = Band(
            color=horizontal_colors[idx],
            r0=cross_top + r0_local,
            c0=weave_left,
            r1=cross_top + r1_local,
            c1=weave_right,
        )
        grid[band.r0 : band.r1 + 1, band.c0 : band.c1 + 1] = band.color
        horizontal_bands.append(band)

    vertical_bands: list[Band] = []
    for idx, (c0_local, c1_local) in enumerate(col_tracks):
        band = Band(
            color=vertical_colors[idx],
            r0=weave_top,
            c0=cross_left + c0_local,
            r1=weave_bottom,
            c1=cross_left + c1_local,
        )
        grid[band.r0 : band.r1 + 1, band.c0 : band.c1 + 1] = band.color
        vertical_bands.append(band)

    intersections = _band_intersections(horizontal_bands, vertical_bands)
    for index, (r0, c0, r1, c1, _h_color, _v_color) in enumerate(intersections, start=1):
        grid[r0 : r1 + 1, c0 : c1 + 1] = horizontal_marker if index % 2 == 1 else vertical_marker

    legend = np.full((LEGEND_SIZE, LEGEND_SIZE), LEGEND_BORDER_COLOR, dtype=int)
    legend[1:3, 1:3] = np.array(
        [
            [horizontal_marker, vertical_marker],
            [vertical_marker, horizontal_marker],
        ],
        dtype=int,
    )
    grid[legend_r0 : legend_r0 + LEGEND_SIZE, legend_c0 : legend_c0 + LEGEND_SIZE] = legend

    values, counts = np.unique(grid, return_counts=True)
    count_map = {int(v): int(c) for v, c in zip(values, counts)}
    bg_count = count_map.get(bg, 0)
    if any(color != bg and count >= bg_count for color, count in count_map.items()):
        return None

    solved = main(grid)
    if np.array_equal(grid, solved):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid weaving sample")
