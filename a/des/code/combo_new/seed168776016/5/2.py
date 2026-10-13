from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
CARDINALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BRANCHES = ("hh", "vv", "hv")
PAIR_COUNT_CHOICES = (3, 4)
ODD_LENGTHS = (3, 5, 7)
DELTA_CHOICES = (-5, -4, -3, 3, 4, 5)
PATCH_ATTEMPTS = 200
GEN_ATTEMPTS = 300
MAX_PATCH_DIM = 12


Coord = tuple[int, int]


@dataclass(frozen=True)
class Segment:
    color: int
    orientation: str
    mid: Coord
    cells: tuple[Coord, ...]


@dataclass(frozen=True)
class Patch:
    color: int
    height: int
    width: int
    input_cells: tuple[Coord, ...]


def _axis_line(a: Coord, b: Coord) -> list[Coord]:
    if a[0] == b[0]:
        step = 1 if b[1] >= a[1] else -1
        return [(a[0], c) for c in range(a[1], b[1] + step, step)]
    if a[1] == b[1]:
        step = 1 if b[0] >= a[0] else -1
        return [(r, a[1]) for r in range(a[0], b[0] + step, step)]
    raise AssertionError("bridge segments must be axis-aligned")


def _merge_paths(*parts: list[Coord]) -> tuple[Coord, ...]:
    merged: list[Coord] = []
    seen: set[Coord] = set()
    for part in parts:
        for cell in part:
            if cell not in seen:
                seen.add(cell)
                merged.append(cell)
    return tuple(merged)


def _bridge_cells(seg_a: Segment, seg_b: Segment) -> tuple[Coord, ...]:
    if seg_a.orientation == "h" and seg_b.orientation == "h":
        first, second = sorted((seg_a, seg_b), key=lambda seg: seg.mid)
        bend = (second.mid[0], first.mid[1])
        return _merge_paths(_axis_line(first.mid, bend), _axis_line(bend, second.mid))

    if seg_a.orientation == "v" and seg_b.orientation == "v":
        first, second = sorted((seg_a, seg_b), key=lambda seg: seg.mid)
        bend = (first.mid[0], second.mid[1])
        return _merge_paths(_axis_line(first.mid, bend), _axis_line(bend, second.mid))

    hseg = seg_a if seg_a.orientation == "h" else seg_b
    vseg = seg_a if seg_a.orientation == "v" else seg_b
    bend = (vseg.mid[0], hseg.mid[1])
    return _merge_paths(_axis_line(hseg.mid, bend), _axis_line(vseg.mid, bend))


def _segment_from_coords(color: int, coords: np.ndarray) -> Segment:
    assert coords.ndim == 2 and coords.shape[1] == 2
    rows = sorted(set(int(r) for r in coords[:, 0]))
    cols = sorted(set(int(c) for c in coords[:, 1]))
    if len(rows) == 1:
        row = rows[0]
        c0, c1 = cols[0], cols[-1]
        expected = [(row, c) for c in range(c0, c1 + 1)]
        assert len(expected) == coords.shape[0]
        assert set(expected) == {tuple(map(int, cell)) for cell in coords}
        mid = (row, (c0 + c1) // 2)
        return Segment(color=color, orientation="h", mid=mid, cells=tuple(expected))

    if len(cols) == 1:
        col = cols[0]
        r0, r1 = rows[0], rows[-1]
        expected = [(r, col) for r in range(r0, r1 + 1)]
        assert len(expected) == coords.shape[0]
        assert set(expected) == {tuple(map(int, cell)) for cell in coords}
        mid = ((r0 + r1) // 2, col)
        return Segment(color=color, orientation="v", mid=mid, cells=tuple(expected))

    raise AssertionError("each foreground component must be a straight segment")


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    mask = grid == color
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in CARDINALS:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-colored pair of straight segments with an orientation-dependent bridge.

    Input structure:
    - The background is Black.
    - Every non-background component is a one-pixel-wide straight horizontal or vertical segment.
    - Each non-background color appears on exactly two separated segments, and the segments do not overlap.

    Rule:
    - Pair the two segments of each color.
    - Use the segments' midpoint cells to draw the bridge: horizontal-horizontal pairs use a vertical-then-horizontal L, vertical-vertical pairs use a horizontal-then-vertical L, and mixed-orientation pairs draw a vertical leg from the horizontal segment and a horizontal leg from the vertical segment until the two legs meet.
    - For same-orientation pairs, if both midpoint row and midpoint column differ, use the earlier midpoint in row-major order as the L's starting midpoint.
    - Paint every bridge with the pair's own color and keep the original segments in place; any covered pixel takes the bridge color.

    Key insight:
    Color matching alone is not enough: the bridge geometry depends on whether the matched segments are both horizontal, both vertical, or mixed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    colors = [int(color) for color in np.unique(g) if int(color) != BG]
    assert colors

    pairs: list[tuple[Coord, int, tuple[Segment, Segment]]] = []
    for color in colors:
        components = _components_of_color(g, color)
        assert len(components) == 2, "each foreground color must form exactly two segments"
        seg_a = _segment_from_coords(color, components[0])
        seg_b = _segment_from_coords(color, components[1])
        anchor = min(seg_a.mid, seg_b.mid)
        pairs.append((anchor, color, (seg_a, seg_b)))

    for _anchor, color, (seg_a, seg_b) in sorted(pairs, key=lambda item: item[0]):
        for r, c in _bridge_cells(seg_a, seg_b):
            out[r, c] = color

    return out


def _make_segment(color: int, orientation: str, mid: Coord, length: int) -> Segment:
    half = length // 2
    if orientation == "h":
        cells = tuple((mid[0], c) for c in range(mid[1] - half, mid[1] + half + 1))
    else:
        cells = tuple((r, mid[1]) for r in range(mid[0] - half, mid[0] + half + 1))
    return Segment(color=color, orientation=orientation, mid=mid, cells=cells)


def _touch_or_overlap(cells_a: tuple[Coord, ...], cells_b: tuple[Coord, ...]) -> bool:
    set_b = set(cells_b)
    for r, c in cells_a:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (r + dr, c + dc) in set_b:
                    return True
    return False


def _shift_cells(cells: tuple[Coord, ...], dr: int, dc: int) -> tuple[Coord, ...]:
    return tuple((r + dr, c + dc) for r, c in cells)


def _make_patch(color: int, branch: str) -> Patch:
    for _ in range(PATCH_ATTEMPTS):
        length_a = random.choice(ODD_LENGTHS)
        length_b = random.choice(ODD_LENGTHS)
        delta_r = random.choice(DELTA_CHOICES)
        delta_c = random.choice(DELTA_CHOICES)

        if branch == "hh":
            seg_a = _make_segment(color, "h", (0, 0), length_a)
            seg_b = _make_segment(color, "h", (delta_r, delta_c), length_b)
        elif branch == "vv":
            seg_a = _make_segment(color, "v", (0, 0), length_a)
            seg_b = _make_segment(color, "v", (delta_r, delta_c), length_b)
        else:
            if random.random() < 0.5:
                seg_a = _make_segment(color, "h", (0, 0), length_a)
                seg_b = _make_segment(color, "v", (delta_r, delta_c), length_b)
            else:
                seg_a = _make_segment(color, "v", (0, 0), length_a)
                seg_b = _make_segment(color, "h", (delta_r, delta_c), length_b)

        if _touch_or_overlap(seg_a.cells, seg_b.cells):
            continue

        bridge = _bridge_cells(seg_a, seg_b)
        footprint = set(seg_a.cells) | set(seg_b.cells) | set(bridge)
        min_r = min(r for r, _c in footprint)
        max_r = max(r for r, _c in footprint)
        min_c = min(c for _r, c in footprint)
        max_c = max(c for _r, c in footprint)

        height = max_r - min_r + 3
        width = max_c - min_c + 3
        if height > MAX_PATCH_DIM or width > MAX_PATCH_DIM:
            continue

        dr = 1 - min_r
        dc = 1 - min_c
        input_cells = _shift_cells(seg_a.cells + seg_b.cells, dr, dc)
        return Patch(color=color, height=height, width=width, input_cells=input_cells)

    raise RuntimeError(f"failed to build patch for branch {branch}")


def _row_patterns(count: int) -> list[tuple[int, ...]]:
    if count == 3:
        return [(3,), (1, 2), (2, 1)]
    if count == 4:
        return [(2, 2), (1, 3), (3, 1), (4,)]
    raise AssertionError("pair count must be 3 or 4")


def _pack_patches(patches: list[Patch]) -> np.ndarray | None:
    shuffled = patches[:]
    random.shuffle(shuffled)
    patterns = _row_patterns(len(shuffled))
    random.shuffle(patterns)

    for pattern in patterns:
        rows: list[list[Patch]] = []
        start = 0
        for size in pattern:
            rows.append(shuffled[start : start + size])
            start += size

        gap_rows = [random.randint(1, 3) for _ in range(max(0, len(rows) - 1))]
        gap_cols: list[list[int]] = []
        row_widths: list[int] = []
        row_heights: list[int] = []
        for row in rows:
            row_gaps = [random.randint(1, 3) for _ in range(max(0, len(row) - 1))]
            gap_cols.append(row_gaps)
            row_width = sum(patch.width for patch in row) + sum(row_gaps)
            row_height = max(patch.height for patch in row)
            row_widths.append(row_width)
            row_heights.append(row_height)

        tight_h = sum(row_heights) + sum(gap_rows)
        tight_w = max(row_widths)
        if tight_h > 30 or tight_w > 30:
            continue

        extra_h = random.randint(0, min(4, 30 - tight_h))
        extra_w = random.randint(0, min(4, 30 - tight_w))
        top_pad = random.randint(0, extra_h)
        left_pad = random.randint(0, extra_w)
        h = tight_h + extra_h
        w = tight_w + extra_w

        grid = np.full((h, w), BG, dtype=int)
        row_top = top_pad
        for row_index, row in enumerate(rows):
            slack = w - row_widths[row_index]
            row_left = random.randint(0, slack)
            col_left = row_left
            for patch_index, patch in enumerate(row):
                for r, c in patch.input_cells:
                    grid[row_top + r, col_left + c] = patch.color
                if patch_index < len(row) - 1:
                    col_left += patch.width + gap_cols[row_index][patch_index]
                else:
                    col_left += patch.width
            row_top += row_heights[row_index]
            if row_index < len(rows) - 1:
                row_top += gap_rows[row_index]

        assert max(grid.shape) <= 30
        return grid

    return None


def _validate_segments_not_adjacent(grid: np.ndarray) -> None:
    foreground = np.argwhere(grid != BG)
    for idx, (r0, c0) in enumerate(foreground):
        for r1, c1 in foreground[idx + 1 :]:
            if abs(int(r0) - int(r1)) <= 1 and abs(int(c0) - int(c1)) <= 1:
                if int(grid[r0, c0]) != int(grid[r1, c1]) or (abs(int(r0) - int(r1)) + abs(int(c0) - int(c1))) == 1:
                    same_component = (
                        int(grid[r0, c0]) == int(grid[r1, c1])
                        and (int(r0) == int(r1) or int(c0) == int(c1))
                    )
                    if not same_component:
                        raise AssertionError("input segments must stay separated by at least one background cell")


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice(PAIR_COUNT_CHOICES)
    branches = list(BRANCHES)
    if pair_count == 4:
        branches.append(random.choice(BRANCHES))
    random.shuffle(branches)

    colors = random.sample(PALETTE, pair_count)
    patches: list[Patch] = []
    for color, branch in zip(colors, branches):
        try:
            patches.append(_make_patch(color, branch))
        except RuntimeError:
            return None

    grid = _pack_patches(patches)
    if grid is None:
        return None

    try:
        _validate_segments_not_adjacent(grid)
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
