import random
from collections import deque

import numpy as np


BG = 0
MARKER = 3
SEGMENT_COLORS = (1, 2, 4, 5, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 800


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                x, y = queue.popleft()
                coords.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        queue.append((nx, ny))
            components.append(np.asarray(coords, dtype=int))
    return components


def _detect_marker(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    assert grid[h - 1, 0] == MARKER, "expected a green corner pixel at bottom-left"

    arm_w = 0
    while arm_w < w and grid[h - 1, arm_w] == MARKER:
        arm_w += 1

    arm_h = 0
    while arm_h < h and grid[h - 1 - arm_h, 0] == MARKER:
        arm_h += 1

    expected = {(h - 1, c) for c in range(arm_w)} | {(r, 0) for r in range(h - arm_h, h)}
    actual = set(map(tuple, np.argwhere(grid == MARKER)))
    assert actual == expected, "green pixels must form exactly one bottom-left L marker"
    return arm_w, arm_h


def _extract_segments(grid: np.ndarray) -> list[dict[str, object]]:
    mask = (grid != BG) & (grid != MARKER)
    segments: list[dict[str, object]] = []
    for coords in _find_components(mask):
        rows = coords[:, 0]
        cols = coords[:, 1]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        colors = np.unique(grid[rows, cols])
        assert len(colors) == 1, "each segment must be monochrome"
        color = int(colors[0])

        if r0 == r1:
            assert len(coords) == c1 - c0 + 1, "horizontal segment must be contiguous"
            orient = "h"
            length = c1 - c0 + 1
        else:
            assert c0 == c1, "segment must be horizontal or vertical"
            assert len(coords) == r1 - r0 + 1, "vertical segment must be contiguous"
            orient = "v"
            length = r1 - r0 + 1

        assert length >= 2, "segments must have an unambiguous direction"
        segments.append(
            {
                "color": color,
                "orient": orient,
                "length": length,
                "row": r0,
                "col": c0,
                "coords": coords,
            }
        )
    return segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pack scattered line segments by direction around a bottom-left green L marker.

    Horizontal segments are sorted by (length, original row) and stacked just above
    the marker, one row per segment, with their left ends aligned to column 0.
    Vertical segments are sorted by (length, original column) and stacked just to
    the right of the marker, one column per segment, with their bottom ends aligned
    to the last row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    arm_w, arm_h = _detect_marker(g)
    segments = _extract_segments(g)

    out = g.copy()
    for seg in segments:
        coords = seg["coords"]
        out[coords[:, 0], coords[:, 1]] = BG

    horizontals = sorted(
        (seg for seg in segments if seg["orient"] == "h"),
        key=lambda seg: (int(seg["length"]), int(seg["row"]), int(seg["col"])),
    )
    verticals = sorted(
        (seg for seg in segments if seg["orient"] == "v"),
        key=lambda seg: (int(seg["length"]), int(seg["col"]), int(seg["row"])),
    )

    for i, seg in enumerate(horizontals):
        row = g.shape[0] - arm_h - 1 - i
        length = int(seg["length"])
        assert row >= 0 and length <= g.shape[1]
        assert np.all(out[row, :length] == BG), "packed horizontal segments must not overlap"
        out[row, :length] = int(seg["color"])

    for j, seg in enumerate(verticals):
        col = arm_w + j
        length = int(seg["length"])
        top = g.shape[0] - length
        assert col < g.shape[1] and top >= 0
        assert np.all(out[top:, col] == BG), "packed vertical segments must not overlap"
        out[top:, col] = int(seg["color"])

    return out


def _block_with_margin(blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    h, w = blocked.shape
    blocked[max(0, r0 - 1) : min(h, r1 + 2), max(0, c0 - 1) : min(w, c1 + 2)] = True


def _output_layout_ok(h: int, w: int, arm_w: int, arm_h: int, h_lengths: list[int], v_lengths: list[int]) -> bool:
    occ = np.zeros((h, w), dtype=bool)
    occ[h - arm_h : h, 0] = True
    occ[h - 1, :arm_w] = True

    for i, length in enumerate(sorted(h_lengths)):
        row = h - arm_h - 1 - i
        if row < 0 or length > w:
            return False
        if np.any(occ[row, :length]):
            return False
        occ[row, :length] = True

    for j, length in enumerate(sorted(v_lengths)):
        col = arm_w + j
        top = h - length
        if col >= w or top < 0:
            return False
        if np.any(occ[top:, col]):
            return False
        occ[top:, col] = True

    return True


def _sample_both_case() -> tuple[int, int, int, int, list[int], list[int]] | None:
    n_h = random.randint(1, 3)
    n_v = random.randint(1, 3)
    arm_w = random.randint(2, 4)
    arm_h = random.randint(2, 4)

    h_extra = [random.randint(0, n_v)]
    for _ in range(1, n_h):
        h_extra.append(random.randint(h_extra[-1], n_v))

    v_caps = [sum(extra <= j for extra in h_extra) for j in range(n_v)]
    v_extra: list[int] = []
    prev = 0
    for cap in v_caps:
        if prev > cap:
            return None
        prev = random.randint(prev, cap)
        v_extra.append(prev)

    h_lengths = [arm_w + extra for extra in h_extra]
    v_lengths = [arm_h + extra for extra in v_extra]
    if max(h_lengths) == arm_w and max(v_lengths) == arm_h:
        return None

    min_h = max(arm_h + n_h, max(v_lengths))
    min_w = max(arm_w + n_v, max(h_lengths))
    h = random.randint(min_h + 3, min(min_h + 9, 22))
    w = random.randint(min_w + 3, min(min_w + 9, 22))
    return h, w, arm_w, arm_h, h_lengths, v_lengths


def _sample_single_case(which: str) -> tuple[int, int, int, int, list[int], list[int]]:
    arm_w = random.randint(1, 4)
    arm_h = random.randint(1, 4)

    if which == "h":
        n_h = random.randint(2, 5)
        n_v = 0
        h_lengths = sorted(random.randint(2, 8) for _ in range(n_h))
        v_lengths: list[int] = []
    else:
        n_h = 0
        n_v = random.randint(2, 5)
        h_lengths = []
        v_lengths = sorted(random.randint(2, 8) for _ in range(n_v))

    min_h = max([arm_h + n_h, arm_h + 1, *v_lengths])
    min_w = max([arm_w + n_v, arm_w + 1, *h_lengths])
    h = random.randint(min_h + 3, min(min_h + 10, 24))
    w = random.randint(min_w + 3, min(min_w + 10, 24))
    return h, w, arm_w, arm_h, h_lengths, v_lengths


def _place_source_segments(
    h: int, w: int, arm_w: int, arm_h: int, h_lengths: list[int], v_lengths: list[int]
) -> np.ndarray | None:
    grid = np.zeros((h, w), dtype=int)
    grid[h - arm_h : h, 0] = MARKER
    grid[h - 1, :arm_w] = MARKER

    blocked = np.zeros((h, w), dtype=bool)
    _block_with_margin(blocked, h - arm_h, 0, h - 1, arm_w - 1)

    pieces = [("h", length) for length in h_lengths] + [("v", length) for length in v_lengths]
    random.shuffle(pieces)
    colors = random.sample(SEGMENT_COLORS, k=len(pieces))

    used_rows: set[int] = set()
    used_cols: set[int] = set()

    for color, (orient, length) in zip(colors, pieces):
        placed = False
        for _ in range(500):
            if orient == "h":
                rows = [r for r in range(h) if r not in used_rows]
                if not rows:
                    return None
                row = random.choice(rows)
                col = random.randint(0, w - length)
                if np.any(blocked[row, col : col + length]):
                    continue
                grid[row, col : col + length] = color
                _block_with_margin(blocked, row, col, row, col + length - 1)
                used_rows.add(row)
                placed = True
                break
            else:
                cols = [c for c in range(w) if c not in used_cols]
                if not cols:
                    return None
                col = random.choice(cols)
                row = random.randint(0, h - length)
                if np.any(blocked[row : row + length, col]):
                    continue
                grid[row : row + length, col] = color
                _block_with_margin(blocked, row, col, row + length - 1, col)
                used_cols.add(col)
                placed = True
                break
        if not placed:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        mode = random.choices(("both", "h", "v"), weights=(0.5, 0.25, 0.25), k=1)[0]
        if mode == "both":
            sample = _sample_both_case()
            if sample is None:
                continue
        else:
            sample = _sample_single_case(mode)

        h, w, arm_w, arm_h, h_lengths, v_lengths = sample
        if len(h_lengths) + len(v_lengths) > len(SEGMENT_COLORS):
            continue
        if not _output_layout_ok(h, w, arm_w, arm_h, h_lengths, v_lengths):
            continue

        grid = _place_source_segments(h, w, arm_w, arm_h, h_lengths, v_lengths)
        if grid is None:
            continue

        out = main(grid)
        if np.array_equal(grid, out):
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")
