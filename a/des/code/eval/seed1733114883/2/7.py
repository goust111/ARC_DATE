from __future__ import annotations

from collections import Counter
import random

import numpy as np


_FRAME_SIZE = 5
_CENTER = 2
_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_OPPOSITE = {(-1, 0): (1, 0), (1, 0): (-1, 0), (0, -1): (0, 1), (0, 1): (0, -1)}
_BORDER_CELLS = [
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
    (1, 0), (1, 4),
    (2, 0), (2, 4),
    (3, 0), (3, 4),
    (4, 0), (4, 1), (4, 2), (4, 3), (4, 4),
]


def _bg_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(grid, dtype=int).ravel()))
    return int(counts.most_common(1)[0][0])


def _uniform_border_color(patch: np.ndarray, bg: int) -> int | None:
    values = [int(patch[r, c]) for r, c in _BORDER_CELLS]
    color = int(values[0])
    if color == bg:
        return None
    if any(int(v) != color for v in values[1:]):
        return None
    return color


def _detect_clock(patch: np.ndarray, bg: int) -> tuple[int, int, tuple[int, int]] | None:
    border = _uniform_border_color(patch, bg)
    if border is None:
        return None

    axle = int(patch[_CENTER, _CENTER])
    if axle in (bg, border):
        return None

    interior = patch[1:4, 1:4]
    if int(np.count_nonzero(interior != bg)) != 2:
        return None
    if any(int(v) not in (bg, axle) for v in interior.ravel()):
        return None

    direction: tuple[int, int] | None = None
    for dr, dc in _DIRS:
        pr = _CENTER + dr
        pc = _CENTER + dc
        if int(patch[pr, pc]) == axle:
            direction = (dr, dc)
            break
    if direction is None:
        return None

    for dr, dc in _DIRS:
        pr = _CENTER + dr
        pc = _CENTER + dc
        if (dr, dc) == direction:
            continue
        if int(patch[pr, pc]) != bg:
            return None

    return border, axle, direction


def _detect_receiver(patch: np.ndarray, bg: int) -> int | None:
    border = _uniform_border_color(patch, bg)
    if border is None:
        return None
    if np.all(patch[1:4, 1:4] == bg):
        return border
    return None


def _scan_scene(grid: np.ndarray, bg: int) -> tuple[list[tuple[int, int, int, int, tuple[int, int]]], list[tuple[int, int]]]:
    h, w = grid.shape
    clocks: list[tuple[int, int, int, int, tuple[int, int]]] = []
    receivers: list[tuple[int, int]] = []
    for r in range(h - _FRAME_SIZE + 1):
        for c in range(w - _FRAME_SIZE + 1):
            patch = grid[r : r + _FRAME_SIZE, c : c + _FRAME_SIZE]
            clock = _detect_clock(patch, bg)
            if clock is not None:
                border, axle, direction = clock
                clocks.append((int(r), int(c), int(border), int(axle), direction))
                continue
            if _detect_receiver(patch, bg) is not None:
                receivers.append((int(r), int(c)))
    return clocks, receivers


def _path_is_clear(grid: np.ndarray, bg: int, src: tuple[int, int], dst: tuple[int, int]) -> bool:
    sr, sc = src
    dr, dc = dst
    if sr == dr:
        left = min(sc, dc) + _FRAME_SIZE
        right = max(sc, dc)
        if right <= left:
            return False
        return bool(np.all(grid[sr : sr + _FRAME_SIZE, left:right] == bg))

    top = min(sr, dr) + _FRAME_SIZE
    bottom = max(sr, dr)
    if bottom <= top:
        return False
    return bool(np.all(grid[top:bottom, sc : sc + _FRAME_SIZE] == bg))


def _receiver_candidates(
    grid: np.ndarray,
    bg: int,
    src: tuple[int, int],
    direction: tuple[int, int],
    receivers: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    sr, sc = src
    dr, dc = direction
    candidates: list[tuple[int, int]] = []
    for rr, cc in receivers:
        if dr == -1 and cc == sc and rr < sr and _path_is_clear(grid, bg, src, (rr, cc)):
            candidates.append((rr, cc))
        elif dr == 1 and cc == sc and rr > sr and _path_is_clear(grid, bg, src, (rr, cc)):
            candidates.append((rr, cc))
        elif dc == -1 and rr == sr and cc < sc and _path_is_clear(grid, bg, src, (rr, cc)):
            candidates.append((rr, cc))
        elif dc == 1 and rr == sr and cc > sc and _path_is_clear(grid, bg, src, (rr, cc)):
            candidates.append((rr, cc))
    if dr != 0:
        candidates.sort(key=lambda pos: abs(pos[0] - sr))
    else:
        candidates.sort(key=lambda pos: abs(pos[1] - sc))
    return candidates


def _flipped_patch(patch: np.ndarray, bg: int, direction: tuple[int, int]) -> np.ndarray:
    out = np.array(patch, dtype=int, copy=True)
    axle = int(out[_CENTER, _CENTER])
    dr, dc = direction
    out[_CENTER + dr, _CENTER + dc] = bg
    odr, odc = _OPPOSITE[(dr, dc)]
    out[_CENTER + odr, _CENTER + odc] = axle
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each 5x5 clock into the hollow frame in its pointer direction, then flip the copied pointer 180 degrees.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    clocks, receivers = _scan_scene(g, bg)

    out = g.copy()
    used_receivers: set[tuple[int, int]] = set()
    for r, c, _border, _axle, direction in clocks:
        candidates = _receiver_candidates(g, bg, (r, c), direction, receivers)
        assert candidates, "expected a receiver in the pointer direction"
        target = candidates[0]
        assert target not in used_receivers, "each receiver should be matched once"
        used_receivers.add(target)
        patch = g[r : r + _FRAME_SIZE, c : c + _FRAME_SIZE]
        out[target[0] : target[0] + _FRAME_SIZE, target[1] : target[1] + _FRAME_SIZE] = _flipped_patch(
            patch, bg, direction
        )
    return out


# --------------- generator ---------------

_GEN_BG = 0
_GEN_RECEIVER = 5
_GEN_COLORS = tuple(color for color in range(10) if color not in (_GEN_BG, _GEN_RECEIVER))
_GEN_H_RANGE = (22, 34)
_GEN_W_RANGE = (22, 34)
_GEN_PAIR_COUNT_RANGE = (1, 4)
_GEN_GAP_MIN = 1
_GEN_GAP_MAX = 8
_GEN_MARGIN = 1
_GEN_RETRIES = 400
_PAIR_PLACE_RETRIES = 200


def _empty_clock_patch(border: int, axle: int, direction: tuple[int, int], bg: int) -> np.ndarray:
    patch = np.full((_FRAME_SIZE, _FRAME_SIZE), bg, dtype=int)
    for r, c in _BORDER_CELLS:
        patch[r, c] = border
    patch[_CENTER, _CENTER] = axle
    dr, dc = direction
    patch[_CENTER + dr, _CENTER + dc] = axle
    return patch


def _empty_receiver_patch(bg: int) -> np.ndarray:
    patch = np.full((_FRAME_SIZE, _FRAME_SIZE), bg, dtype=int)
    for r, c in _BORDER_CELLS:
        patch[r, c] = _GEN_RECEIVER
    return patch


def _rect_clear(mask: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 0) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(mask.shape[0], r0 + h + pad)
    cc1 = min(mask.shape[1], c0 + w + pad)
    return not bool(mask[rr0:rr1, cc0:cc1].any())


def _mark_rect(mask: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 0) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(mask.shape[0], r0 + h + pad)
    cc1 = min(mask.shape[1], c0 + w + pad)
    mask[rr0:rr1, cc0:cc1] = True


def _band_rect(src: tuple[int, int], dst: tuple[int, int]) -> tuple[int, int, int, int]:
    sr, sc = src
    dr, dc = dst
    if sr == dr:
        return sr, min(sc, dc), _FRAME_SIZE, abs(dc - sc) + _FRAME_SIZE
    return min(sr, dr), sc, abs(dr - sr) + _FRAME_SIZE, _FRAME_SIZE


def _place_pair(
    height: int, width: int, occupied: np.ndarray, direction: tuple[int, int]
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    for _ in range(_PAIR_PLACE_RETRIES):
        if direction[0] == 0:
            max_gap = min(_GEN_GAP_MAX, width - 2 * _GEN_MARGIN - 2 * _FRAME_SIZE)
            if max_gap < _GEN_GAP_MIN:
                return None
            gap = random.randint(_GEN_GAP_MIN, max_gap)
            span = 2 * _FRAME_SIZE + gap
            r = random.randint(_GEN_MARGIN, height - _FRAME_SIZE - _GEN_MARGIN)
            left = random.randint(_GEN_MARGIN, width - span - _GEN_MARGIN)
            if direction == (0, 1):
                src = (r, left)
                dst = (r, left + _FRAME_SIZE + gap)
            else:
                dst = (r, left)
                src = (r, left + _FRAME_SIZE + gap)
        else:
            max_gap = min(_GEN_GAP_MAX, height - 2 * _GEN_MARGIN - 2 * _FRAME_SIZE)
            if max_gap < _GEN_GAP_MIN:
                return None
            gap = random.randint(_GEN_GAP_MIN, max_gap)
            span = 2 * _FRAME_SIZE + gap
            c = random.randint(_GEN_MARGIN, width - _FRAME_SIZE - _GEN_MARGIN)
            top = random.randint(_GEN_MARGIN, height - span - _GEN_MARGIN)
            if direction == (1, 0):
                src = (top, c)
                dst = (top + _FRAME_SIZE + gap, c)
            else:
                dst = (top, c)
                src = (top + _FRAME_SIZE + gap, c)

        band = _band_rect(src, dst)
        if not _rect_clear(occupied, *band, pad=1):
            continue
        _mark_rect(occupied, *band, pad=1)
        return src, dst
    return None


def _paste(grid: np.ndarray, patch: np.ndarray, top_left: tuple[int, int]) -> None:
    r, c = top_left
    grid[r : r + patch.shape[0], c : c + patch.shape[1]] = patch


def _expected_output(
    grid: np.ndarray, records: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int], int, int]]
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for src, dst, direction, border, axle in records:
        patch = _empty_clock_patch(border, axle, direction, _GEN_BG)
        out[dst[0] : dst[0] + _FRAME_SIZE, dst[1] : dst[1] + _FRAME_SIZE] = _flipped_patch(
            patch, _GEN_BG, direction
        )
    return out


def _valid_generated_scene(grid: np.ndarray, expected_pairs: int) -> bool:
    bg = _bg_color(grid)
    clocks, receivers = _scan_scene(grid, bg)
    if len(clocks) != expected_pairs or len(receivers) != expected_pairs:
        return False

    used: set[tuple[int, int]] = set()
    for r, c, _border, _axle, direction in clocks:
        forward = _receiver_candidates(grid, bg, (r, c), direction, receivers)
        if len(forward) != 1:
            return False
        if forward[0] in used:
            return False
        used.add(forward[0])
        for other in _DIRS:
            if other == direction:
                continue
            if _receiver_candidates(grid, bg, (r, c), other, receivers):
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*_GEN_H_RANGE)
    width = random.randint(*_GEN_W_RANGE)
    pair_count = random.randint(*_GEN_PAIR_COUNT_RANGE)

    grid = np.full((height, width), _GEN_BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)
    directions = random.sample(_DIRS, k=pair_count)
    records: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int], int, int]] = []

    receiver_patch = _empty_receiver_patch(_GEN_BG)
    for direction in directions:
        placed = _place_pair(height, width, occupied, direction)
        if placed is None:
            return None
        src, dst = placed
        border, axle = random.sample(_GEN_COLORS, 2)
        _paste(grid, _empty_clock_patch(border, axle, direction, _GEN_BG), src)
        _paste(grid, receiver_patch, dst)
        records.append((src, dst, direction, border, axle))

    expected = _expected_output(grid, records)
    if not _valid_generated_scene(grid, pair_count):
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid clock-copy input")
