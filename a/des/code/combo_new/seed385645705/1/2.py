from __future__ import annotations

from collections import deque
import random

import numpy as np


_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_CORNERS = ("TL", "TR", "BL", "BR")
_EMPTY_OFFSET = {
    "TL": (0, 0),
    "TR": (0, 1),
    "BL": (1, 0),
    "BR": (1, 1),
}


def _background(g: np.ndarray) -> int:
    values, counts = np.unique(g, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        comp: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in _NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _l_empty_corner(cells: list[tuple[int, int]]) -> str | None:
    if len(cells) != 3:
        return None
    r0, c0, r1, c1 = _bbox(cells)
    if r1 - r0 + 1 != 2 or c1 - c0 + 1 != 2:
        return None
    present = {(r - r0, c - c0) for r, c in cells}
    missing = {(0, 0), (0, 1), (1, 0), (1, 1)} - present
    if len(missing) != 1:
        return None
    empty = next(iter(missing))
    for corner, offset in _EMPTY_OFFSET.items():
        if empty == offset:
            return corner
    return None


def _frame_info(cells: list[tuple[int, int]]) -> tuple[int, int, int, int, str] | None:
    r0, c0, r1, c1 = _bbox(cells)
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return None

    expected = set()
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))

    actual = set(cells)
    missing = expected - actual
    if actual - expected or len(missing) != 1:
        return None

    missing_cell = next(iter(missing))
    corner_by_cell = {
        (r0, c0): "TL",
        (r0, c1): "TR",
        (r1, c0): "BL",
        (r1, c1): "BR",
    }
    corner = corner_by_cell.get(missing_cell)
    if corner is None:
        return None
    return int(r0), int(c0), int(r1), int(c1), corner


def _target_l_cells(frame: tuple[int, int, int, int, str]) -> list[tuple[int, int]]:
    r0, c0, r1, c1, corner = frame
    origins = {
        "TL": (r0, c0),
        "TR": (r0, c1 - 1),
        "BL": (r1 - 1, c0),
        "BR": (r1 - 1, c1 - 1),
    }
    or0, oc0 = origins[corner]
    empty = _EMPTY_OFFSET[corner]
    cells = []
    for dr in (0, 1):
        for dc in (0, 1):
            if (dr, dc) != empty:
                cells.append((int(or0 + dr), int(oc0 + dc)))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each colored L fragment into the missing corner of the matching rectangular frame.

    Input structure:
    - The background is the most common color.
    - For each active color, one 4-connected component is a one-pixel-thick rectangular frame with exactly one missing corner pixel.
    - The same color also appears as one separate three-pixel L fragment occupying three cells of a 2x2 box.

    Rule:
    - For each color, identify the frame's missing corner: top-left, top-right, bottom-left, or bottom-right.
    - Clear the scattered same-color L fragment back to background.
    - Rotate the fragment so its empty 2x2 corner faces the same outside direction as the frame's missing corner.
    - Translate that rotated L so its empty cell coincides with the missing frame corner, placing its three cells in the other cells of that inward 2x2 corner block.

    Key insight: The frame's absent corner gives both the destination and the required L rotation; matching is by visible color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background(g)
    out = g.copy()

    placements: list[tuple[int, list[tuple[int, int]]]] = []
    clear_cells: list[tuple[int, int]] = []

    for color in set(map(int, g.ravel())) - {bg}:
        comps = _components(g == int(color))
        frames: list[tuple[int, int, int, int, str]] = []
        pieces: list[list[tuple[int, int]]] = []
        for comp in comps:
            frame = _frame_info(comp)
            if frame is not None:
                frames.append(frame)
                continue
            if _l_empty_corner(comp) is not None:
                pieces.append(comp)

        if not frames:
            continue
        assert len(frames) == 1, "each active color must have exactly one missing-corner frame"
        assert len(pieces) == 1, "each frame color must have exactly one separate L fragment"
        clear_cells.extend(pieces[0])
        placements.append((int(color), _target_l_cells(frames[0])))

    for r, c in clear_cells:
        out[r, c] = bg
    for color, cells in placements:
        for r, c in cells:
            out[r, c] = int(color)
    return out


# ---------------------------- generator ----------------------------

_GEN_RETRIES = 400
_GRID_MIN = 16
_GRID_MAX = 30
_FRAME_H_RANGE = (4, 8)
_FRAME_W_RANGE = (4, 9)
_OBJECT_COUNT_RANGE = (2, 5)


def _all_l_offsets(empty_corner: str) -> list[tuple[int, int]]:
    empty = _EMPTY_OFFSET[empty_corner]
    return [(r, c) for r in (0, 1) for c in (0, 1) if (r, c) != empty]


def _rect_free(blocked: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    H, W = blocked.shape
    rr0 = max(0, int(r0 - pad))
    cc0 = max(0, int(c0 - pad))
    rr1 = min(H, int(r0 + h + pad))
    cc1 = min(W, int(c0 + w + pad))
    return not bool(blocked[rr0:rr1, cc0:cc1].any())


def _mark_rect(blocked: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    H, W = blocked.shape
    rr0 = max(0, int(r0 - pad))
    cc0 = max(0, int(c0 - pad))
    rr1 = min(H, int(r0 + h + pad))
    cc1 = min(W, int(c0 + w + pad))
    blocked[rr0:rr1, cc0:cc1] = True


def _free_locations(blocked: np.ndarray, h: int, w: int, pad: int = 1) -> list[tuple[int, int]]:
    H, W = blocked.shape
    locs = []
    for r in range(0, H - h + 1):
        for c in range(0, W - w + 1):
            if _rect_free(blocked, r, c, h, w, pad):
                locs.append((int(r), int(c)))
    return locs


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int, missing: str) -> None:
    r1, c1 = int(r0 + h - 1), int(c0 + w - 1)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    missing_cell = {
        "TL": (r0, c0),
        "TR": (r0, c1),
        "BL": (r1, c0),
        "BR": (r1, c1),
    }[missing]
    grid[missing_cell] = int(_background(grid))


def _expected_after_move(grid: np.ndarray, records: list[dict], bg: int) -> np.ndarray:
    out = grid.copy()
    for rec in records:
        for r, c in rec["piece_cells"]:
            out[r, c] = int(bg)
    for rec in records:
        for r, c in _target_l_cells(rec["frame"]):
            out[r, c] = int(rec["color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(_GRID_MIN, _GRID_MAX)
    W = random.randint(_GRID_MIN, _GRID_MAX)
    bg = random.choice(range(10))
    colors = [c for c in range(10) if c != bg]
    random.shuffle(colors)

    max_objects = min(_OBJECT_COUNT_RANGE[1], len(colors))
    n_objects = random.randint(_OBJECT_COUNT_RANGE[0], max_objects)
    grid = np.full((H, W), int(bg), dtype=int)
    blocked = np.zeros((H, W), dtype=bool)
    records: list[dict] = []

    for color in colors[:n_objects]:
        fh = random.randint(*_FRAME_H_RANGE)
        fw = random.randint(*_FRAME_W_RANGE)
        if fh > H or fw > W:
            return None
        frame_locs = _free_locations(blocked, fh, fw, pad=1)
        if not frame_locs:
            return None
        fr, fc = random.choice(frame_locs)
        missing = random.choice(_CORNERS)
        _draw_frame(grid, fr, fc, fh, fw, color, missing)
        _mark_rect(blocked, fr, fc, fh, fw, pad=1)

        piece_locs = _free_locations(blocked, 2, 2, pad=1)
        if not piece_locs:
            return None
        pr, pc = random.choice(piece_locs)
        piece_empty = random.choice(_CORNERS)
        piece_cells = []
        for dr, dc in _all_l_offsets(piece_empty):
            grid[pr + dr, pc + dc] = int(color)
            piece_cells.append((int(pr + dr), int(pc + dc)))
        _mark_rect(blocked, pr, pc, 2, 2, pad=1)

        records.append(
            {
                "color": int(color),
                "frame": (int(fr), int(fc), int(fr + fh - 1), int(fc + fw - 1), missing),
                "piece_cells": piece_cells,
            }
        )

    values, counts = np.unique(grid, return_counts=True)
    bg_count = int(counts[np.where(values == int(bg))[0][0]])
    if any(bg_count <= int(count) for value, count in zip(values, counts) if int(value) != int(bg)):
        return None

    expected = _expected_after_move(grid, records, int(bg))
    try:
        produced = main(grid)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to construct a valid sample")
