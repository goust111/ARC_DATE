from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
TOP_COLOR = 1
RED = 2
LEFT_COLOR = 3
BOTTOM_COLOR = 4
PINK = 6
RIGHT_COLOR = 7

FRAME_COLORS = {TOP_COLOR, LEFT_COLOR, BOTTOM_COLOR, RIGHT_COLOR}
COLOR_TO_SIDE = {
    TOP_COLOR: "top",
    BOTTOM_COLOR: "bottom",
    LEFT_COLOR: "left",
    RIGHT_COLOR: "right",
}
SIDE_TO_COLOR = {
    "top": TOP_COLOR,
    "bottom": BOTTOM_COLOR,
    "left": LEFT_COLOR,
    "right": RIGHT_COLOR,
}
SIDE_TO_STEP = {
    "top": (1, 0),
    "bottom": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}
DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    bbox = _bbox(grid != BG)
    assert bbox is not None, "expected a non-empty frame"
    r0, c0, r1, c1 = bbox
    assert r1 - r0 >= 4 and c1 - c0 >= 4, "frame interior must be at least 3x3"
    assert int(grid[r0, c0]) == BG
    assert int(grid[r0, c1]) == BG
    assert int(grid[r1, c0]) == BG
    assert int(grid[r1, c1]) == BG
    assert np.all(grid[r0, c0 + 1 : c1] == TOP_COLOR), "top edge must be blue"
    assert np.all(grid[r1, c0 + 1 : c1] == BOTTOM_COLOR), "bottom edge must be yellow"
    assert np.all(grid[r0 + 1 : r1, c0] == LEFT_COLOR), "left edge must be green"
    assert np.all(grid[r0 + 1 : r1, c1] == RIGHT_COLOR), "right edge must be orange"
    outside = np.ones_like(grid, dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    assert not np.any(grid[outside] != BG), "only the frame-and-markers scene is expected"
    return r0, c0, r1, c1


def _interior_bbox(frame: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = frame
    return r0 + 1, c0 + 1, r1 - 1, c1 - 1


def _extract_markers(grid: np.ndarray, frame: tuple[int, int, int, int]) -> list[tuple[int, int, int]]:
    ir0, ic0, ir1, ic1 = _interior_bbox(frame)
    markers: list[tuple[int, int, int]] = []
    for r in range(ir0, ir1 + 1):
        for c in range(ic0, ic1 + 1):
            color = int(grid[r, c])
            if color == BG:
                continue
            assert color in FRAME_COLORS, "markers must match one of the four frame colors"
            markers.append((r, c, color))
    assert 3 <= len(markers) <= 6, "expected 3 to 6 markers"
    counts = Counter(color for _r, _c, color in markers)
    assert all(counts[color] <= 2 for color in counts), "each frame color may appear at most twice"
    return markers


def _move_marker(
    marker: tuple[int, int, int],
    frame: tuple[int, int, int, int],
    mode: str = "color",
) -> tuple[int, int, int, str]:
    r, c, color = marker
    ir0, ic0, ir1, ic1 = _interior_bbox(frame)
    if mode == "color":
        side = COLOR_TO_SIDE[color]
    elif mode == "nearest":
        dists = {
            "top": r - ir0,
            "bottom": ir1 - r,
            "left": c - ic0,
            "right": ic1 - c,
        }
        side = min(dists, key=lambda key: (dists[key], "top bottom left right".split().index(key)))
    else:
        raise ValueError(mode)

    if side == "top":
        return ir0, c, color, side
    if side == "bottom":
        return ir1, c, color, side
    if side == "left":
        return r, ic0, color, side
    return r, ic1, color, side


def _ray_cells(start: tuple[int, int], side: str, frame: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r, c = start
    dr, dc = SIDE_TO_STEP[side]
    ir0, ic0, ir1, ic1 = _interior_bbox(frame)
    cells: list[tuple[int, int]] = []
    rr, cc = r, c
    while ir0 <= rr <= ir1 and ic0 <= cc <= ic1:
        cells.append((int(rr), int(cc)))
        rr += dr
        cc += dc
    return cells


def _render(
    grid: np.ndarray,
    markers: list[tuple[int, int, int]],
    frame: tuple[int, int, int, int],
    mode: str = "color",
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    ir0, ic0, ir1, ic1 = _interior_bbox(frame)
    out[ir0 : ir1 + 1, ic0 : ic1 + 1] = BG

    moved = [_move_marker(marker, frame, mode=mode) for marker in markers]
    moved_positions = [(r, c) for r, c, _color, _side in moved]
    assert len(set(moved_positions)) == len(moved_positions), "moved markers must not collide"

    cell_sides: dict[tuple[int, int], set[str]] = {}
    cell_color: dict[tuple[int, int], int] = {}
    moved_lookup = {(r, c): color for r, c, color, _side in moved}

    for r, c, color, side in moved:
        out[r, c] = color
        for cell in _ray_cells((r, c), side, frame):
            cell_sides.setdefault(cell, set()).add(side)
            cell_color.setdefault(cell, color)

    for cell, sides in cell_sides.items():
        if cell in moved_lookup:
            out[cell] = moved_lookup[cell]
        elif len(sides) >= 2:
            out[cell] = RED
        else:
            out[cell] = cell_color[cell]

    red_cells = np.argwhere(out == RED)
    for rr, cc in red_cells:
        for dr, dc in DIAGONALS:
            nr = int(rr + dr)
            nc = int(cc + dc)
            if ir0 <= nr <= ir1 and ic0 <= nc <= ic1 and int(out[nr, nc]) == BG:
                out[nr, nc] = PINK
    return out


def _render_without_move(
    grid: np.ndarray,
    markers: list[tuple[int, int, int]],
    frame: tuple[int, int, int, int],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    ir0, ic0, ir1, ic1 = _interior_bbox(frame)
    out[ir0 : ir1 + 1, ic0 : ic1 + 1] = BG

    moved_lookup = {(r, c): color for r, c, color in markers}
    cell_sides: dict[tuple[int, int], set[str]] = {}
    cell_color: dict[tuple[int, int], int] = {}

    for r, c, color in markers:
        side = COLOR_TO_SIDE[color]
        out[r, c] = color
        for cell in _ray_cells((r, c), side, frame):
            cell_sides.setdefault(cell, set()).add(side)
            cell_color.setdefault(cell, color)

    for cell, sides in cell_sides.items():
        if cell in moved_lookup:
            out[cell] = moved_lookup[cell]
        elif len(sides) >= 2:
            out[cell] = RED
        else:
            out[cell] = cell_color[cell]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each color-coded marker to the matching frame side, fire a one-way inward ray, then decorate ray intersections.

    Input structure:
    - A black background contains one rectangular frame whose corners stay black.
    - The frame edges use four fixed colors: blue on top, yellow on bottom, green on left, and orange on right.
    - Inside the frame are 3 to 6 singleton markers whose colors always match one of those four edge colors.

    Rule:
    - Clear the interior markers from their original locations.
    - Reposition each marker to the cell just inside the matching frame side, preserving its orthogonal coordinate: top/bottom markers keep their column, left/right markers keep their row.
    - From each moved marker, extend a single ray only toward the frame interior until the opposite frame side would be reached; keep the moved marker itself in its original color.
    - Repaint any non-marker cell visited by rays from two or more different sides to red.
    - For every red intersection, recolor each diagonal interior black neighbor to pink.

    Key insight:
    - The ray origin is not the marker's original location: the color first selects a destination side, and that moved edge-adjacent position then determines the ray's actual path.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    frame = _find_frame(g)
    markers = _extract_markers(g, frame)
    return _render(g, markers, frame, mode="color")


def _sample_far_side(candidates: list[int], prefer_high: bool) -> int:
    midpoint = len(candidates) // 2
    pool = candidates[midpoint:] if prefer_high else candidates[: max(1, len(candidates) - midpoint)]
    if not pool:
        pool = candidates
    return int(random.choice(pool))


def _is_isolated(cell: tuple[int, int], used: set[tuple[int, int]], gap: int = 1) -> bool:
    r, c = cell
    for rr, cc in used:
        if max(abs(r - rr), abs(c - cc)) <= gap:
            return False
    return True


def _pick_extra_colors() -> list[int]:
    extra_count = random.choices([0, 1, 2], weights=[2, 4, 3])[0]
    if extra_count == 0:
        return []
    if extra_count == 1:
        return [random.choice([TOP_COLOR, BOTTOM_COLOR, LEFT_COLOR, RIGHT_COLOR])]
    return random.sample([TOP_COLOR, BOTTOM_COLOR, LEFT_COLOR, RIGHT_COLOR], 2)


def _wrong_nearest_output(grid: np.ndarray) -> np.ndarray:
    frame = _find_frame(grid)
    markers = _extract_markers(grid, frame)
    return _render(grid, markers, frame, mode="nearest")


def _validate_scene(grid: np.ndarray) -> None:
    frame = _find_frame(grid)
    markers = _extract_markers(grid, frame)
    out = _render(grid, markers, frame, mode="color")
    moved = [_move_marker(marker, frame, mode="color") for marker in markers]
    moved_positions = {(r, c) for r, c, _color, _side in moved}

    assert len(moved_positions) == len(markers)
    assert np.count_nonzero(out == RED) >= 2, "need at least two intersections"
    assert np.count_nonzero(out == PINK) >= 1, "need visible diagonal decoration"
    assert any((r, c) not in moved_positions for r, c, _color in markers), "movement must matter"

    nearest_mismatch = False
    for marker in markers:
        _r, _c, color = marker
        moved_by_color = _move_marker(marker, frame, mode="color")[:2]
        moved_by_nearest = _move_marker(marker, frame, mode="nearest")[:2]
        if moved_by_color != moved_by_nearest:
            nearest_mismatch = True
            break
    assert nearest_mismatch, "at least one marker must reject the nearest-side shortcut"

    no_move = _render_without_move(grid, markers, frame)
    nearest = _wrong_nearest_output(grid)
    assert not np.array_equal(out, no_move), "output must depend on moving before firing"
    assert not np.array_equal(out, nearest), "output must depend on the color-to-side mapping"


def generate_input() -> np.ndarray:
    for _ in range(400):
        height = random.randint(12, 20)
        width = random.randint(12, 20)
        grid = np.zeros((height, width), dtype=int)

        inner_h = random.randint(7, min(11, height - 4))
        inner_w = random.randint(7, min(11, width - 4))
        frame_h = inner_h + 2
        frame_w = inner_w + 2
        top = random.randint(1, height - frame_h - 1)
        left = random.randint(1, width - frame_w - 1)
        bottom = top + frame_h - 1
        right = left + frame_w - 1
        frame = (top, left, bottom, right)

        grid[top, left + 1 : right] = TOP_COLOR
        grid[bottom, left + 1 : right] = BOTTOM_COLOR
        grid[top + 1 : bottom, left] = LEFT_COLOR
        grid[top + 1 : bottom, right] = RIGHT_COLOR

        row_choices = list(range(top + 2, bottom - 1))
        col_choices = list(range(left + 2, right - 1))
        if len(row_choices) < 2 or len(col_choices) < 2:
            continue

        colors = [TOP_COLOR, BOTTOM_COLOR, LEFT_COLOR, RIGHT_COLOR] + _pick_extra_colors()
        random.shuffle(colors)

        vertical_cols: dict[int, list[int]] = {TOP_COLOR: [], BOTTOM_COLOR: []}
        horizontal_rows: dict[int, list[int]] = {LEFT_COLOR: [], RIGHT_COLOR: []}
        used_vertical_cols: set[int] = set()
        used_horizontal_rows: set[int] = set()
        used_cells: set[tuple[int, int]] = set()
        placed: list[tuple[int, int, int]] = []

        ok = True
        for color in colors:
            if color in (TOP_COLOR, BOTTOM_COLOR):
                available_cols = [c for c in col_choices if c not in used_vertical_cols]
                if not available_cols:
                    ok = False
                    break
                prefer_high = color == TOP_COLOR
                col = _sample_far_side(available_cols, prefer_high=False if color == BOTTOM_COLOR else True)
                if color == TOP_COLOR:
                    row_pool = [r for r in row_choices if r >= top + 1 + inner_h // 2]
                    if not row_pool:
                        row_pool = row_choices
                else:
                    row_pool = [r for r in row_choices if r <= bottom - 1 - inner_h // 2]
                    if not row_pool:
                        row_pool = row_choices
                random.shuffle(row_pool)
                cell = None
                for row in row_pool:
                    cand = (int(row), int(col))
                    if _is_isolated(cand, used_cells):
                        cell = cand
                        break
                if cell is None:
                    ok = False
                    break
                used_vertical_cols.add(int(col))
                vertical_cols[color].append(int(col))
            else:
                available_rows = [r for r in row_choices if r not in used_horizontal_rows]
                if not available_rows:
                    ok = False
                    break
                row = _sample_far_side(available_rows, prefer_high=color == RIGHT_COLOR)
                if color == LEFT_COLOR:
                    col_pool = [c for c in col_choices if c >= left + 1 + inner_w // 2]
                    if not col_pool:
                        col_pool = col_choices
                else:
                    col_pool = [c for c in col_choices if c <= right - 1 - inner_w // 2]
                    if not col_pool:
                        col_pool = col_choices
                random.shuffle(col_pool)
                cell = None
                for col in col_pool:
                    cand = (int(row), int(col))
                    if _is_isolated(cand, used_cells):
                        cell = cand
                        break
                if cell is None:
                    ok = False
                    break
                used_horizontal_rows.add(int(row))
                horizontal_rows[color].append(int(row))

            used_cells.add(cell)
            placed.append((cell[0], cell[1], color))

        if not ok:
            continue
        if len(set(vertical_cols[TOP_COLOR] + vertical_cols[BOTTOM_COLOR])) < 2:
            continue
        if len(set(horizontal_rows[LEFT_COLOR] + horizontal_rows[RIGHT_COLOR])) < 2:
            continue

        for r, c, color in placed:
            grid[r, c] = color

        try:
            _validate_scene(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
