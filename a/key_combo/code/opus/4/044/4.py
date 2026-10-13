from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


BG = 0
WALL = 2
REF = 5
BAR_COLORS = (1, 3, 4, 7)
REF_SIZE = 2
MAX_TRIES = 400


@dataclass(frozen=True)
class Bar:
    color: int
    col: int
    top: int
    bottom: int

    @property
    def height(self) -> int:
        return self.bottom - self.top + 1

    @property
    def coords(self) -> np.ndarray:
        return np.array([(r, self.col) for r in range(self.top, self.bottom + 1)], dtype=int)


def _extract_reference(grid: np.ndarray) -> np.ndarray:
    ref = np.asarray(grid[:REF_SIZE, :REF_SIZE], dtype=int).copy()
    assert ref.shape == (2, 2), "reference motif must be 2x2"
    assert np.count_nonzero(ref == REF) == 3, "reference motif must contain three grey pixels"
    assert np.count_nonzero(ref == BG) == 1, "reference motif must contain one black pixel"
    assert set(np.unique(ref)).issubset({BG, REF}), "reference motif may only use black and grey"
    return ref


def _find_bars(grid: np.ndarray) -> list[Bar]:
    bars: list[Bar] = []
    colors = [int(c) for c in np.unique(grid) if int(c) not in {BG, WALL, REF}]
    for color in colors:
        coords = np.argwhere(grid == color)
        assert coords.size > 0, "each bar color must appear"
        cols = {int(c) for _, c in coords}
        rows = sorted(int(r) for r, _ in coords)
        assert len(cols) == 1, "each bar must occupy one column"
        assert len(rows) in {2, 3}, "bars must have height 2 or 3"
        assert rows == list(range(rows[0], rows[-1] + 1)), "bar pixels must be consecutive"
        bars.append(Bar(color=color, col=next(iter(cols)), top=rows[0], bottom=rows[-1]))
    assert 3 <= len(bars) <= 4, "expected three or four bars"
    assert len({bar.col for bar in bars}) == len(bars), "bars must use distinct columns"
    return bars


def _slide_bars(bars: list[Bar], shape: tuple[int, int]) -> list[dict]:
    h, w = map(int, shape)
    occupied = np.zeros((h, w), dtype=bool)
    occupied[h - 1, :] = True
    settled: list[dict] = []

    order = sorted(bars, key=lambda bar: (h - 1 - bar.bottom, bar.col))
    for bar in order:
        coords = bar.coords.copy()
        distance = 0
        while True:
            nxt = coords + np.array([1, 0], dtype=int)
            if np.any(nxt[:, 0] >= h) or np.any(occupied[nxt[:, 0], nxt[:, 1]]):
                break
            coords = nxt
            distance += 1
        assert distance >= 1, "every bar must slide at least one row"
        occupied[coords[:, 0], coords[:, 1]] = True
        anchor_idx = int(np.argmax(coords[:, 0]))
        settled.append(
            {
                "color": bar.color,
                "initial": bar,
                "coords": coords,
                "distance": distance,
                "anchor": tuple(int(x) for x in coords[anchor_idx]),
            }
        )
    return settled


def _scaled_stamp(ref: np.ndarray, factor: int, color: int) -> np.ndarray:
    scaled = np.kron(ref, np.ones((factor, factor), dtype=int))
    return np.where(scaled == REF, color, BG).astype(int)


def _place_stamp(dst: np.ndarray, stamp: np.ndarray, anchor: tuple[int, int]) -> None:
    ar, ac = anchor
    h, w = stamp.shape
    r0 = ar - h + 1
    c0 = ac - w + 1
    assert r0 >= 0 and c0 >= 0, "stamp placement must stay in bounds"
    assert ar < dst.shape[0] and ac < dst.shape[1], "anchor must be in bounds"
    dst[r0 : ar + 1, c0 : ac + 1] = stamp


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= 8 and w >= 8, "grid is too small for this puzzle"
    assert np.all(g[h - 1, :] == WALL), "last row must be a full red wall"

    ref = _extract_reference(g)
    bars = _find_bars(g)
    settled = _slide_bars(bars, g.shape)

    out = np.zeros_like(g)
    out[h - 1, :] = WALL
    for info in sorted(settled, key=lambda item: (item["distance"], item["anchor"][1])):
        stamp = _scaled_stamp(ref, int(info["distance"]), int(info["color"]))
        _place_stamp(out, stamp, info["anchor"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each vertical bar down to the red wall and replace it with a reference motif scaled by its slide distance.

    Input structure:
    - Black background with a full red wall on the last row.
    - A 2x2 grey reference motif sits at the top-left corner and contains exactly one black missing corner.
    - Three or four colored vertical bars of width 1 and height 2 or 3 stand above the wall in distinct columns.

    Rule:
    - Keep the red wall fixed, read the top-left 2x2 motif, and erase the original motif and all original bars from the output.
    - Slide the bars straight downward, settling the ones closest to the wall first; each bar stops when the next row would hit the wall or a previously settled bar.
    - Let each bar's actual slide distance D be its scale factor, enlarge the 2x2 motif to a (2D)x(2D) stamp by turning each source cell into a DxD block, and recolor the grey cells with that bar's color.
    - Stamp the enlarged motif so its bottom-right corner aligns with the bar's final bottom cell; if stamps overlap, the later one with larger D overwrites earlier paint.

    Key insight:
    - The physical simulation does not just move the bars: the number of rows each bar falls is reused as the geometric scale for the replacement stamp.
    """
    return _transform(grid)


def _stamp_bbox(anchor: tuple[int, int], factor: int) -> tuple[int, int, int, int]:
    ar, ac = anchor
    size = 2 * factor
    return ar - size + 1, ac - size + 1, ar, ac


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _choose_reference() -> np.ndarray:
    ref = np.full((2, 2), REF, dtype=int)
    missing = random.choice([(0, 0), (0, 1), (1, 0), (1, 1)])
    ref[missing] = BG
    return ref


def _choose_columns(width: int, distances: list[int]) -> list[int] | None:
    order = list(range(len(distances)))
    random.shuffle(order)
    chosen: list[tuple[int, int]] = []

    def backtrack(idx: int) -> bool:
        if idx == len(order):
            return True
        i = order[idx]
        d = distances[i]
        size = 2 * d
        candidates = list(range(size - 1, width))
        random.shuffle(candidates)
        for col in candidates:
            if col in (0, 1):
                continue
            box = (col - size + 1, col)
            ok = True
            for prev_i, prev_col in chosen:
                prev_d = distances[prev_i]
                prev_size = 2 * prev_d
                prev_box = (prev_col - prev_size + 1, prev_col)
                if not (box[1] < prev_box[0] - 1 or prev_box[1] < box[0] - 1):
                    ok = False
                    break
            if not ok:
                continue
            chosen.append((i, col))
            if backtrack(idx + 1):
                return True
            chosen.pop()
        return False

    if not backtrack(0):
        return None
    cols = [0] * len(distances)
    for i, col in chosen:
        cols[i] = col
    return cols


def _generate_instance() -> np.ndarray | None:
    h = random.randint(18, 26)
    w = random.randint(18, 26)
    n_bars = random.choice([3, 4])
    distances = sorted(random.sample([1, 2, 3, 4], k=n_bars))
    cols = _choose_columns(w, distances)
    if cols is None:
        return None

    colors = random.sample(list(BAR_COLORS), k=n_bars)
    heights = [random.choice([2, 3]) for _ in range(n_bars)]
    ref = _choose_reference()

    grid = np.zeros((h, w), dtype=int)
    grid[h - 1, :] = WALL
    grid[:2, :2] = ref

    anchors = [(h - 2, col) for col in cols]
    stamp_boxes = [_stamp_bbox(anchor, d) for anchor, d in zip(anchors, distances)]
    for box in stamp_boxes:
        r0, c0, r1, c1 = box
        if r0 < 0 or c0 < 0 or r1 >= h - 1 or c1 >= w:
            return None
    for i in range(len(stamp_boxes)):
        for j in range(i + 1, len(stamp_boxes)):
            if _boxes_overlap(stamp_boxes[i], stamp_boxes[j]):
                return None

    for color, col, height_bar, distance in zip(colors, cols, heights, distances):
        bottom = h - 2 - distance
        top = bottom - height_bar + 1
        if top < 0:
            return None
        if col in (0, 1):
            return None
        if top <= 1 and col <= 1:
            return None
        grid[top : bottom + 1, col] = color

    if not _validate_input(grid):
        return None
    out = main(grid)
    if not _validate_output(grid, out):
        return None
    return grid


def _validate_input(grid: np.ndarray) -> bool:
    try:
        h, w = grid.shape
        if h > 30 or w > 30:
            return False
        if not np.all(grid[h - 1, :] == WALL):
            return False
        ref = _extract_reference(grid)
        if np.count_nonzero(grid == REF) != 3:
            return False
        bars = _find_bars(grid)
        if len({bar.color for bar in bars}) != len(bars):
            return False
        if any(bar.col in (0, 1) for bar in bars):
            return False
        gaps = [h - 2 - bar.bottom for bar in bars]
        if sorted(gaps) != sorted(set(gaps)):
            return False
        if not all(1 <= gap <= 4 for gap in gaps):
            return False
        for bar in bars:
            if ref[:, min(bar.col, 1)].size and bar.col in (0, 1):
                return False

        boxes = [_stamp_bbox((h - 2, bar.col), h - 2 - bar.bottom) for bar in bars]
        for box in boxes:
            r0, c0, r1, c1 = box
            if r0 < 0 or c0 < 0 or r1 >= h - 1 or c1 >= w:
                return False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if _boxes_overlap(boxes[i], boxes[j]):
                    return False
        _ = ref
        return True
    except AssertionError:
        return False


def _validate_output(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        h, w = inp.shape
        if out.shape != inp.shape:
            return False
        if not np.all(out[h - 1, :] == WALL):
            return False
        if np.any(out[:2, :2] == REF):
            return False

        ref = _extract_reference(inp)
        bars = _find_bars(inp)
        settled = _slide_bars(bars, inp.shape)
        expected = np.zeros_like(inp)
        expected[h - 1, :] = WALL
        seen_colors = set()
        for info in sorted(settled, key=lambda item: (item["distance"], item["anchor"][1])):
            d = int(info["distance"])
            seen_colors.add(int(info["color"]))
            stamp = _scaled_stamp(ref, d, int(info["color"]))
            anchor = info["anchor"]
            r0, c0, r1, c1 = _stamp_bbox(anchor, d)
            if stamp.shape != (2 * d, 2 * d):
                return False
            if not np.array_equal(stamp == int(info["color"]), np.kron(ref == REF, np.ones((d, d), dtype=bool))):
                return False
            if not np.array_equal(stamp == BG, np.kron(ref == BG, np.ones((d, d), dtype=bool))):
                return False
            expected[r0 : r1 + 1, c0 : c1 + 1] = stamp
            if not np.array_equal(out[r0 : r1 + 1, c0 : c1 + 1], stamp):
                return False
        if len(seen_colors) != len(bars):
            return False
        if not np.array_equal(expected, out):
            return False
        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    for _ in range(MAX_TRIES):
        grid = _generate_instance()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
