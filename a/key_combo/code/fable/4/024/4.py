from __future__ import annotations

import random
from collections import Counter

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = sorted(int(color) for color in np.unique(grid))
    if exclude is None:
        return colors
    return [color for color in colors if color not in exclude]


def _border_coords(r0: int, c0: int, r1: int, c1: int) -> list[Coord]:
    coords: list[Coord] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c1))
    for c in range(c1, c0 - 1, -1):
        coords.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        coords.append((r, c0))
    return coords


def _find_frame(grid: np.ndarray, bg: int) -> tuple[int, BBox, Coord, str]:
    g = np.asarray(grid, dtype=int)
    fg_coords = np.argwhere(g != bg)
    assert fg_coords.size > 0, "expected foreground"

    candidates: list[tuple[int, BBox, Coord, str]] = []
    for color in _colors_present(g, exclude={bg}):
        coords = np.argwhere(g == color)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0 = int(r0)
        c0 = int(c0)
        r1 = int(r1)
        c1 = int(c1)
        if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
            continue

        inside_bbox = (
            (fg_coords[:, 0] >= r0)
            & (fg_coords[:, 0] <= r1)
            & (fg_coords[:, 1] >= c0)
            & (fg_coords[:, 1] <= c1)
        )
        if not np.all(inside_bbox):
            continue

        border = _border_coords(r0, c0, r1, c1)
        missing = [(r, c) for r, c in border if int(g[r, c]) != color]
        if len(missing) != 1:
            continue

        gap = missing[0]
        if int(g[gap]) != bg:
            continue

        corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
        if gap in corners:
            continue

        if gap[0] == r0:
            side = "top"
        elif gap[0] == r1:
            side = "bottom"
        elif gap[1] == c0:
            side = "left"
        elif gap[1] == c1:
            side = "right"
        else:
            continue

        candidates.append((color, (r0, c0, r1, c1), gap, side))

    assert len(candidates) == 1, "expected exactly one gapped rectangular frame"
    return candidates[0]


def _draw_frame(out: np.ndarray, frame_color: int, bbox: BBox, gap: Coord) -> None:
    r0, c0, r1, c1 = bbox
    for r, c in _border_coords(r0, c0, r1, c1):
        if (r, c) != gap:
            out[r, c] = frame_color


def _ordered_lane(side: str, bbox: BBox, lane_value: int) -> list[Coord]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return [(r, lane_value) for r in range(r0 + 1, r1)]
    if side == "bottom":
        return [(r, lane_value) for r in range(r1 - 1, r0, -1)]
    if side == "left":
        return [(lane_value, c) for c in range(c0 + 1, c1)]
    if side == "right":
        return [(lane_value, c) for c in range(c1 - 1, c0, -1)]
    raise AssertionError(f"unknown side: {side}")


def _edge_stop(side: str, gap: Coord, shape: tuple[int, int]) -> Coord:
    h, w = shape
    gr, gc = gap
    if side == "top":
        return 0, gc
    if side == "bottom":
        return h - 1, gc
    if side == "left":
        return gr, 0
    if side == "right":
        return gr, w - 1
    raise AssertionError(f"unknown side: {side}")


def _gap_line_value(side: str, gap: Coord) -> int:
    return int(gap[1] if side in ("top", "bottom") else gap[0])


def _line_values(side: str, bbox: BBox) -> list[int]:
    r0, c0, r1, c1 = bbox
    if side in ("top", "bottom"):
        return list(range(c0 + 1, c1))
    return list(range(r0 + 1, r1))


def _particles_on_lane(grid: np.ndarray, lane: list[Coord], bg: int) -> list[int]:
    return [int(grid[r, c]) for r, c in lane if int(grid[r, c]) != bg]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide interior pixels toward the unique frame gap so the front one escapes, the second plugs the hole, and the rest stack behind it.

    Input structure:
    - A dominant background contains a one-pixel-thick monochrome rectangular frame with exactly one missing border cell.
    - All other foreground pixels lie strictly inside that frame; there is no foreground outside it.
    - The missing border cell sits on exactly one side and therefore defines a unique inward-to-outward gravity direction.

    Rule:
    - Keep the frame fixed except for the missing cell, which may be occupied by the second pixel in the gap-aligned queue.
    - For each interior row or column parallel to gravity, stably compact its pixels toward the relevant inner wall while preserving near-to-far order.
    - On the unique row or column aligned with the gap, the nearest pixel exits through the hole and stops at the grid edge, the second moves into the gap cell, and any remaining pixels stack just inside the frame behind that plug.

    Key insight:
    - The single missing border cell simultaneously identifies gravity, opens a one-pixel escape route for exactly one leading pixel, and marks the slot where the next pixel gets trapped as a plug.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)

    frame_color, bbox, gap, side = _find_frame(g, bg)
    r0, c0, r1, c1 = bbox
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame interior must exist"
    assert np.any(g[r0 + 1 : r1, c0 + 1 : c1] != bg), "expected interior pixels"

    out = np.full_like(g, bg)
    _draw_frame(out, frame_color, bbox, gap)

    gap_value = _gap_line_value(side, gap)
    for lane_value in _line_values(side, bbox):
        lane = _ordered_lane(side, bbox, lane_value)
        colors = _particles_on_lane(g, lane, bg)
        if not colors:
            continue

        if lane_value != gap_value:
            for (r, c), color in zip(lane, colors):
                out[r, c] = color
            continue

        edge_r, edge_c = _edge_stop(side, gap, g.shape)
        out[edge_r, edge_c] = colors[0]
        if len(colors) >= 2:
            out[gap] = colors[1]
        for (r, c), color in zip(lane, colors[2:]):
            out[r, c] = color

    return out


# --------------- generator (task-specific) ---------------

_GEN_BG = 0
_GEN_COLORS = tuple(range(1, 10))
_GEN_MARGIN = 2
_GEN_GRID_MIN = 13
_GEN_GRID_MAX = 24
_GEN_FRAME_MIN = 7
_GEN_FRAME_MAX = 12
_GEN_MAX_PARTICLES = 8
_GEN_ATTEMPTS = 400
_GEN_CALL_COUNT = 0
_GEN_SCHEDULE = (
    ("top", 0),
    ("bottom", 1),
    ("left", 2),
    ("right", 3),
    ("top", 1),
    ("bottom", 2),
    ("left", 3),
    ("right", 0),
    ("top", 2),
    ("bottom", 3),
    ("left", 0),
    ("right", 1),
    ("top", 3),
    ("bottom", 0),
    ("left", 1),
    ("right", 2),
)


def _sample_layout() -> tuple[int, int, BBox]:
    h = random.randint(_GEN_GRID_MIN, _GEN_GRID_MAX)
    w = random.randint(_GEN_GRID_MIN, _GEN_GRID_MAX)
    fh = random.randint(_GEN_FRAME_MIN, min(_GEN_FRAME_MAX, h - 2 * _GEN_MARGIN))
    fw = random.randint(_GEN_FRAME_MIN, min(_GEN_FRAME_MAX, w - 2 * _GEN_MARGIN))
    r0 = random.randint(_GEN_MARGIN, h - fh - _GEN_MARGIN)
    c0 = random.randint(_GEN_MARGIN, w - fw - _GEN_MARGIN)
    return h, w, (r0, c0, r0 + fh - 1, c0 + fw - 1)


def _sample_gap(side: str, bbox: BBox) -> Coord:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return r0, random.randint(c0 + 1, c1 - 1)
    if side == "bottom":
        return r1, random.randint(c0 + 1, c1 - 1)
    if side == "left":
        return random.randint(r0 + 1, r1 - 1), c0
    if side == "right":
        return random.randint(r0 + 1, r1 - 1), c1
    raise AssertionError(f"unknown side: {side}")


def _line_capacity(side: str, bbox: BBox) -> int:
    r0, c0, r1, c1 = bbox
    return int(r1 - r0 - 1 if side in ("top", "bottom") else c1 - c0 - 1)


def _aligned_count_for_family(family: int, capacity: int) -> int:
    if family == 0:
        return 0
    if family == 1:
        return 1
    if family == 2:
        return 2
    return random.randint(3, min(4, capacity))


def _sample_indices(length: int, count: int, prefer_spread: bool = True) -> list[int] | None:
    if count > length:
        return None
    for spaced in ((prefer_spread, False) if prefer_spread else (False,)):
        for _ in range(120):
            idxs = sorted(random.sample(range(length), count))
            if not spaced or all(b - a > 1 for a, b in zip(idxs, idxs[1:])):
                return idxs
    return None


def _gap_lane_cells(side: str, bbox: BBox, gap: Coord) -> list[Coord]:
    return _ordered_lane(side, bbox, _gap_line_value(side, gap))


def _choose_stack_lane(side: str, bbox: BBox, gap: Coord) -> int:
    values = _line_values(side, bbox)
    gap_value = _gap_line_value(side, gap)
    far = [value for value in values if value != gap_value and abs(value - gap_value) > 1]
    if far:
        return random.choice(far)
    return random.choice([value for value in values if value != gap_value])


def _same_gap_lane(side: str, gap: Coord, coord: Coord) -> bool:
    return bool(coord[1] == gap[1] if side in ("top", "bottom") else coord[0] == gap[0])


def _chebyshev_clear(coord: Coord, occupied: set[Coord]) -> bool:
    r, c = coord
    return all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in occupied)


def _bg_is_dominant(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    freq = {int(value): int(count) for value, count in zip(values, counts)}
    bg_count = freq.get(bg, 0)
    return all(bg_count > count for value, count in freq.items() if value != bg)


def _build_input(side: str, family: int) -> np.ndarray | None:
    h, w, bbox = _sample_layout()
    frame_color = random.choice(_GEN_COLORS)
    gap = _sample_gap(side, bbox)

    grid = np.full((h, w), _GEN_BG, dtype=int)
    _draw_frame(grid, frame_color, bbox, gap)

    line_capacity = _line_capacity(side, bbox)
    aligned_count = _aligned_count_for_family(family, line_capacity)
    gap_lane = _gap_lane_cells(side, bbox, gap)
    aligned_indices = _sample_indices(len(gap_lane), aligned_count)
    if aligned_indices is None:
        return None

    positions: list[Coord] = [gap_lane[idx] for idx in aligned_indices]
    occupied = set(positions)

    stack_lane_value = _choose_stack_lane(side, bbox, gap)
    stack_lane = _ordered_lane(side, bbox, stack_lane_value)
    stack_count = random.randint(2, min(3, len(stack_lane)))
    stack_indices = _sample_indices(len(stack_lane), stack_count)
    if stack_indices is None:
        return None
    for idx in stack_indices:
        coord = stack_lane[idx]
        if coord in occupied:
            return None
        positions.append(coord)
        occupied.add(coord)

    remaining_budget = _GEN_MAX_PARTICLES - len(positions)
    if remaining_budget < 0:
        return None
    if family == 3:
        single_count = random.randint(0, min(1, remaining_budget))
    else:
        min_single = 1
        max_single = min(3, remaining_budget)
        if max_single < min_single:
            single_count = 0
        else:
            single_count = random.randint(min_single, max_single)

    r0, c0, r1, c1 = bbox
    candidates = [
        (r, c)
        for r in range(r0 + 1, r1)
        for c in range(c0 + 1, c1)
        if not _same_gap_lane(side, gap, (r, c))
        and (r, c) not in occupied
        and _chebyshev_clear((r, c), occupied)
    ]
    random.shuffle(candidates)
    if len(candidates) < single_count:
        return None
    for coord in candidates[:single_count]:
        positions.append(coord)
        occupied.add(coord)

    if len(positions) < 3:
        return None

    particle_colors = [color for color in _GEN_COLORS if color != frame_color]
    random.shuffle(particle_colors)
    if len(positions) > len(particle_colors):
        return None
    for coord, color in zip(positions, particle_colors):
        grid[coord] = color

    if not _bg_is_dominant(grid, _GEN_BG):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    if np.count_nonzero(out != grid) < 3:
        return None
    if not np.any(out == particle_colors[0]):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    side, family = _GEN_SCHEDULE[_GEN_CALL_COUNT % len(_GEN_SCHEDULE)]
    for _ in range(_GEN_ATTEMPTS):
        grid = _build_input(side, family)
        if grid is not None:
            _GEN_CALL_COUNT += 1
            return grid
    raise RuntimeError(f"failed to generate valid input for side={side}, family={family}")
