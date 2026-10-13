from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
MAX_SIDE = 30
GENERATE_RETRIES = 400
PLACEMENT_TRIES = 250


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _normalized_shape(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    shifted = coords - coords.min(axis=0, keepdims=True)
    order = np.lexsort((shifted[:, 1], shifted[:, 0]))
    return tuple((int(shifted[i, 0]), int(shifted[i, 1])) for i in order)


def _frame_candidate(grid: np.ndarray, color: int, comp: np.ndarray, bg: int) -> bool:
    r0, c0, r1, c1 = _bbox(comp)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    if not np.all(grid[r0, c0 : c1 + 1] == color):
        return False
    if not np.all(grid[r1, c0 : c1 + 1] == color):
        return False
    if not np.all(grid[r0 : r1 + 1, c0] == color):
        return False
    if not np.all(grid[r0 : r1 + 1, c1] == color):
        return False
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    return bool(interior.size and np.any(interior == bg))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate each color's outside fragment layout into the matching same-color framed cavity.

    Input structure:
    - The background is black.
    - Several distinct colors each form one one-pixel-thick rectangular border; the border may extend inward with same-color wall cells, leaving black cavity cells inside the rectangle.
    - Outside the frames, the remaining pixels of each such color appear as several small 4-connected fragments.

    Rule:
    - For each color, identify its unique rectangular frame component and collect all same-color fragments outside that frame.
    - The union of those outside fragments matches the pattern of black cavity cells inside the frame up to a single translation, with fragment relative offsets unchanged.
    - Erase the outside fragments and repaint the frame's black cavity cells with that color, leaving all borders and all other pixels unchanged.

    Key insight:
    - Color selects the destination frame, while the fragments' relative positions already encode the cavity layout, so each color group moves by one global translation only.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    out = g.copy()

    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        color_mask = g == color
        comps = _components(color_mask)
        frame_indices = [i for i, comp in enumerate(comps) if _frame_candidate(g, color, comp, bg)]
        assert len(frame_indices) == 1, "each color must have exactly one rectangular frame"

        frame_idx = frame_indices[0]
        frame_coords = comps[frame_idx]
        frame_mask = np.zeros_like(color_mask, dtype=bool)
        frame_mask[frame_coords[:, 0], frame_coords[:, 1]] = True

        outside_comps = [comp for i, comp in enumerate(comps) if i != frame_idx]
        assert outside_comps, "each frame color must have outside fragments"
        assert all(1 <= len(comp) <= 4 for comp in outside_comps), "fragments must be 1-4 cells"

        outside_coords = np.argwhere(color_mask & ~frame_mask)
        r0, c0, r1, c1 = _bbox(frame_coords)
        cavity_local = np.argwhere(g[r0 + 1 : r1, c0 + 1 : c1] == bg)
        assert len(cavity_local) > 0, "frame interior must contain a cavity"
        cavity_coords = cavity_local + np.array([[r0 + 1, c0 + 1]])

        assert len(outside_coords) == len(cavity_coords), "fragment area must match cavity area"
        assert _normalized_shape(outside_coords) == _normalized_shape(cavity_coords), (
            "outside layout must equal cavity layout up to translation"
        )

        out[outside_coords[:, 0], outside_coords[:, 1]] = bg
        out[cavity_coords[:, 0], cavity_coords[:, 1]] = color

    return out


_BASE_SHAPES = [
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1], [1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[0, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=bool),
    np.array([[0, 1], [1, 1], [0, 1]], dtype=bool),
]


def _mark_with_padding(mask: np.ndarray, cells: Iterable[tuple[int, int]]) -> None:
    h, w = mask.shape
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    mask[rr, cc] = True


def _build_cavity(rng: random.Random, interior_h: int, interior_w: int) -> np.ndarray | None:
    target_parts = rng.randint(2, 5)
    occupied = np.zeros((interior_h, interior_w), dtype=bool)
    blocked = np.zeros((interior_h, interior_w), dtype=bool)
    placed = 0

    for _ in range(target_parts):
        success = False
        for _ in range(80):
            shape = rng.choice(_BASE_SHAPES)
            sh, sw = shape.shape
            if sh > interior_h or sw > interior_w:
                continue
            top = rng.randint(0, interior_h - sh)
            left = rng.randint(0, interior_w - sw)
            cells: list[tuple[int, int]] = []
            ok = True
            for rr in range(sh):
                for cc in range(sw):
                    if not shape[rr, cc]:
                        continue
                    ar, ac = top + rr, left + cc
                    if blocked[ar, ac]:
                        ok = False
                        break
                    cells.append((ar, ac))
                if not ok:
                    break
            if not ok:
                continue
            for ar, ac in cells:
                occupied[ar, ac] = True
            _mark_with_padding(blocked, cells)
            placed += 1
            success = True
            break
        if not success:
            break

    if placed < 2:
        return None
    area = int(occupied.sum())
    if area < 3 or area >= interior_h * interior_w:
        return None
    comps = _components(occupied)
    if len(comps) != placed:
        return None
    if not all(1 <= len(comp) <= 4 for comp in comps):
        return None
    rows = np.where(occupied.any(axis=1))[0]
    cols = np.where(occupied.any(axis=0))[0]
    if len(rows) < 2 or len(cols) < 2:
        return None
    return occupied


def _make_object(
    rng: random.Random, color: int, frame_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for _ in range(120):
        if frame_count >= 4:
            interior_h = rng.randint(4, 6)
            interior_w = rng.randint(4, 6)
        elif frame_count == 3:
            interior_h = rng.randint(4, 7)
            interior_w = rng.randint(4, 7)
        else:
            interior_h = rng.randint(5, 8)
            interior_w = rng.randint(5, 8)

        cavity = _build_cavity(rng, interior_h, interior_w)
        if cavity is None:
            continue

        frame = np.full((interior_h + 2, interior_w + 2), color, dtype=int)
        frame[1:-1, 1:-1][cavity] = BG

        cavity_coords = np.argwhere(cavity)
        r0, c0 = cavity_coords.min(axis=0)
        r1, c1 = cavity_coords.max(axis=0)
        group = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
        normalized = cavity[r0 : r1 + 1, c0 : c1 + 1]
        group[normalized] = color

        if _frame_candidate(frame, color, np.argwhere(frame == color), BG):
            return frame, cavity, group
    raise RuntimeError("failed to build a valid frame/group object")


def _find_spot(
    reserved: np.ndarray, box_h: int, box_w: int, pad: int, rng: random.Random
) -> tuple[int, int] | None:
    h, w = reserved.shape
    if box_h + 2 * pad > h or box_w + 2 * pad > w:
        return None
    for _ in range(PLACEMENT_TRIES):
        top = rng.randint(0, h - box_h)
        left = rng.randint(0, w - box_w)
        r0 = max(0, top - pad)
        c0 = max(0, left - pad)
        r1 = min(h, top + box_h + pad)
        c1 = min(w, left + box_w + pad)
        if not reserved[r0:r1, c0:c1].any():
            return top, left
    return None


def _reserve(reserved: np.ndarray, top: int, left: int, box_h: int, box_w: int, pad: int) -> None:
    h, w = reserved.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + box_h + pad)
    c1 = min(w, left + box_w + pad)
    reserved[r0:r1, c0:c1] = True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, MAX_SIDE)
    w = rng.randint(18, MAX_SIDE)
    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    frame_count = rng.randint(2, 4)
    colors = rng.sample(list(range(1, 10)), frame_count)
    records: list[dict[str, np.ndarray | int]] = []

    try:
        objects = [_make_object(rng, color, frame_count) for color in colors]
    except RuntimeError:
        return None

    order = list(range(frame_count))
    rng.shuffle(order)

    for idx in order:
        color = colors[idx]
        frame, cavity, group = objects[idx]
        ft = _find_spot(reserved, frame.shape[0], frame.shape[1], pad=1, rng=rng)
        if ft is None:
            return None
        fr, fc = ft
        grid[fr : fr + frame.shape[0], fc : fc + frame.shape[1]] = frame
        _reserve(reserved, fr, fc, frame.shape[0], frame.shape[1], pad=1)

        gt = _find_spot(reserved, group.shape[0], group.shape[1], pad=1, rng=rng)
        if gt is None:
            return None
        gr, gc = gt
        group_mask = group == color
        grid[gr : gr + group.shape[0], gc : gc + group.shape[1]][group_mask] = color
        _reserve(reserved, gr, gc, group.shape[0], group.shape[1], pad=1)

        cavity_abs = np.argwhere(cavity) + np.array([[fr + 1, fc + 1]])
        group_abs = np.argwhere(group_mask) + np.array([[gr, gc]])

        records.append(
            {
                "color": color,
                "cavity_abs": cavity_abs,
                "group_abs": group_abs,
            }
        )

    expected = grid.copy()
    for rec in records:
        color = int(rec["color"])
        cavity_abs = np.asarray(rec["cavity_abs"], dtype=int)
        group_abs = np.asarray(rec["group_abs"], dtype=int)
        expected[group_abs[:, 0], group_abs[:, 1]] = BG
        expected[cavity_abs[:, 0], cavity_abs[:, 1]] = color

    if _get_bg(grid) != BG:
        return None
    if np.count_nonzero(grid == BG) <= (grid.size // 3):
        return None

    for color in colors:
        color_mask = grid == color
        comps = _components(color_mask)
        frame_indices = [i for i, comp in enumerate(comps) if _frame_candidate(grid, color, comp, BG)]
        if len(frame_indices) != 1:
            return None
        outside_comps = [comp for i, comp in enumerate(comps) if i != frame_indices[0]]
        if len(outside_comps) < 2:
            return None
        if not all(1 <= len(comp) <= 4 for comp in outside_comps):
            return None
        outside_coords = np.argwhere(color_mask)
        frame_mask = np.zeros_like(color_mask, dtype=bool)
        frame_coords = comps[frame_indices[0]]
        frame_mask[frame_coords[:, 0], frame_coords[:, 1]] = True
        outside_count = int(np.count_nonzero(color_mask & ~frame_mask))
        cavity_count = int(np.count_nonzero(grid[_bbox(frame_coords)[0] + 1 : _bbox(frame_coords)[2], _bbox(frame_coords)[1] + 1 : _bbox(frame_coords)[3]] == BG))
        if outside_count != cavity_count:
            return None

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
