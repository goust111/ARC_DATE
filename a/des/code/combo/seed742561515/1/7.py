from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BLUE = 1
GREEN = 3
RING_COLORS = (2, 4, 5, 6, 7, 8, 9)

GEN_ATTEMPTS = 400
PAIR_ATTEMPTS = 120
PLACE_ATTEMPTS = 160

BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class Ring:
    color: int
    bbox: BBox
    mode: str


def _most_common(values: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(values, dtype=int), return_counts=True)
    return int(vals[int(counts.argmax())])


def _checkerboard(shape: tuple[int, int], blue_even: bool) -> np.ndarray:
    h, w = map(int, shape)
    parity = np.indices((h, w)).sum(axis=0) % 2
    even = BLUE if blue_even else GREEN
    odd = GREEN if blue_even else BLUE
    return np.where(parity == 0, even, odd).astype(int)


def _infer_background(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    parity = np.indices((h, w)).sum(axis=0) % 2
    even_color = _most_common(grid[parity == 0])
    odd_color = _most_common(grid[parity == 1])
    assert even_color != odd_color
    return np.where(parity == 0, even_color, odd_color).astype(int)


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _draw_border(grid: np.ndarray, bbox: BBox, color: int) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    if r0 > r1 or c0 > c1:
        return
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _ring_key(ring: Ring) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = ring.bbox
    return (int(r0 + r1), int(c0 + c1), int(r0), int(c0))


def _analyze_ring(grid: np.ndarray, bg: np.ndarray, coords: np.ndarray) -> Ring:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    assert h >= 3 and w >= 3

    values = np.unique(grid[rows, cols])
    assert len(values) == 1
    color = int(values[0])
    assert color not in (BLUE, GREEN)

    local = np.zeros((h, w), dtype=bool)
    local[rows - r0, cols - c0] = True
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    assert np.array_equal(local, border)

    patch = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    assert np.all(patch[border] == color)
    assert np.array_equal(patch[1:-1, 1:-1], bg[r0 + 1 : r1, c0 + 1 : c1])

    inner_color = int(patch[1, 1])
    assert inner_color in (BLUE, GREEN)
    mode = "shrink" if inner_color == BLUE else "expand"
    return Ring(color=color, bbox=(r0, c0, r1, c1), mode=mode)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform each colored rectangular ring according to the checkerboard phase hidden inside it.

    Blue at the ring interior's top-left cell means shrink inward by one pixel.
    Green at that cell means expand outward by one pixel.
    """
    g = np.asarray(grid, dtype=int)
    bg = _infer_background(g)
    ring_mask = g != bg

    rings = [_analyze_ring(g, bg, comp) for comp in _components4(ring_mask)]
    out = bg.copy()

    shrinkers = sorted((ring for ring in rings if ring.mode == "shrink"), key=_ring_key)
    expanders = sorted((ring for ring in rings if ring.mode == "expand"), key=_ring_key)

    for ring in shrinkers:
        r0, c0, r1, c1 = ring.bbox
        shrunk = (r0 + 1, c0 + 1, r1 - 1, c1 - 1)
        if shrunk[0] <= shrunk[2] and shrunk[1] <= shrunk[3]:
            _draw_border(out, shrunk, ring.color)

    h, w = g.shape
    for ring in expanders:
        r0, c0, r1, c1 = ring.bbox
        grown = (r0 - 1, c0 - 1, r1 + 1, c1 + 1)
        assert 0 <= grown[0] and 0 <= grown[1] and grown[2] < h and grown[3] < w
        _draw_border(out, grown, ring.color)

    return out


def _grow_box(bbox: BBox, amount: int) -> BBox:
    r0, c0, r1, c1 = bbox
    return (int(r0 - amount), int(c0 - amount), int(r1 + amount), int(c1 + amount))


def _boxes_overlap_with_gap(a: BBox, b: BBox, gap: int) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _in_bounds(bbox: BBox, h: int, w: int) -> bool:
    return 0 <= bbox[0] and 0 <= bbox[1] and bbox[2] < h and bbox[3] < w


def _rand_with_parity(lo: int, hi: int, parity: int) -> int | None:
    first = int(lo) if lo % 2 == parity else int(lo + 1)
    if first > hi:
        return None
    count = ((int(hi) - first) // 2) + 1
    return int(first + 2 * random.randrange(count))


def _mode_parity(blue_even: bool, mode: str) -> int:
    shrink_parity = 0 if blue_even else 1
    return int(shrink_parity if mode == "shrink" else 1 - shrink_parity)


def _safe_bbox(mode: str, bbox: BBox) -> BBox:
    return _grow_box(bbox, 1) if mode == "expand" else bbox


def _sample_dims(mode: str) -> tuple[int, int]:
    if mode == "shrink" and random.random() < 0.35:
        return random.randint(3, 5), random.randint(3, 6)
    return random.randint(4, 8), random.randint(4, 8)


def _candidate_ok(bbox: BBox, mode: str, placed: list[Ring]) -> bool:
    safe = _safe_bbox(mode, bbox)
    for other in placed:
        if _boxes_overlap_with_gap(bbox, other.bbox, 1):
            return False
        if _boxes_overlap_with_gap(safe, _safe_bbox(other.mode, other.bbox), 1):
            return False
    return True


def _place_single_ring(h: int, w: int, blue_even: bool, mode: str, color: int, placed: list[Ring]) -> Ring | None:
    want = _mode_parity(blue_even, mode)
    for _ in range(PLACE_ATTEMPTS):
        rh, rw = _sample_dims(mode)
        max_r0 = h - rh - 1
        max_c0 = w - rw - 1
        if max_r0 < 1 or max_c0 < 1:
            return None
        r0 = random.randint(1, max_r0)
        c0 = _rand_with_parity(1, max_c0, (want - r0) % 2)
        if c0 is None:
            continue
        bbox = (r0, c0, r0 + rh - 1, c0 + rw - 1)
        if _candidate_ok(bbox, mode, placed):
            return Ring(color=color, bbox=bbox, mode=mode)
    return None


def _place_overlap_pair(h: int, w: int, blue_even: bool, colors: tuple[int, int]) -> list[Ring] | None:
    want = _mode_parity(blue_even, "expand")
    for _ in range(PAIR_ATTEMPTS):
        if random.random() < 0.5:
            span_h = random.randint(4, 7)
            left_w = random.choice((3, 5, 7))
            right_w = random.randint(3, 7)
            total_w = left_w + 1 + right_w
            max_r0 = h - span_h - 1
            max_c0 = w - total_w - 1
            if max_r0 < 1 or max_c0 < 1:
                continue
            r0 = random.randint(1, max_r0)
            c0 = _rand_with_parity(1, max_c0, (want - r0) % 2)
            if c0 is None:
                continue
            a = Ring(colors[0], (r0, c0, r0 + span_h - 1, c0 + left_w - 1), "expand")
            b = Ring(
                colors[1],
                (r0, c0 + left_w + 1, r0 + span_h - 1, c0 + total_w - 1),
                "expand",
            )
        else:
            span_w = random.randint(4, 7)
            top_h = random.choice((3, 5, 7))
            bottom_h = random.randint(3, 7)
            total_h = top_h + 1 + bottom_h
            max_r0 = h - total_h - 1
            max_c0 = w - span_w - 1
            if max_r0 < 1 or max_c0 < 1:
                continue
            c0 = random.randint(1, max_c0)
            r0 = _rand_with_parity(1, max_r0, (want - c0) % 2)
            if r0 is None:
                continue
            a = Ring(colors[0], (r0, c0, r0 + top_h - 1, c0 + span_w - 1), "expand")
            b = Ring(
                colors[1],
                (r0 + top_h + 1, c0, r0 + total_h - 1, c0 + span_w - 1),
                "expand",
            )

        sa = _safe_bbox(a.mode, a.bbox)
        sb = _safe_bbox(b.mode, b.bbox)
        if not (_in_bounds(sa, h, w) and _in_bounds(sb, h, w)):
            continue
        if not _boxes_overlap_with_gap(sa, sb, 0):
            continue
        if _boxes_overlap_with_gap(a.bbox, b.bbox, 0):
            continue
        return [a, b]
    return None


def _make_attempt() -> np.ndarray | None:
    h = random.randint(20, 30)
    w = random.randint(20, 30)
    blue_even = bool(random.getrandbits(1))

    ring_count = random.randint(3, 5)
    expand_count = 2 if ring_count <= 4 else random.choice((2, 3))
    shrink_count = ring_count - expand_count

    want_pair = expand_count >= 2 and random.random() < 0.45
    colors = random.sample(RING_COLORS, ring_count)
    placed: list[Ring] = []
    color_idx = 0

    if want_pair:
        pair = _place_overlap_pair(h, w, blue_even, (colors[0], colors[1]))
        if pair is None:
            return None
        placed.extend(pair)
        color_idx = 2
        expand_count -= 2

    modes = ["expand"] * expand_count + ["shrink"] * shrink_count
    random.shuffle(modes)

    for mode in modes:
        ring = _place_single_ring(h, w, blue_even, mode, colors[color_idx], placed)
        if ring is None:
            return None
        placed.append(ring)
        color_idx += 1

    grid = _checkerboard((h, w), blue_even)
    for ring in placed:
        _draw_border(grid, ring.bbox, ring.color)

    if not any(ring.mode == "shrink" for ring in placed):
        return None
    if not any(ring.mode == "expand" for ring in placed):
        return None

    try:
        _ = main(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _make_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
