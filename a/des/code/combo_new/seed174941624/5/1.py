from __future__ import annotations

from collections import Counter

import numpy as np


SIDES = ("top", "right", "bottom", "left")
MAX_GENERATION_ATTEMPTS = 300

_RNG = np.random.default_rng()
_GEN_COUNTER = 0


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)
    assert len(order) > 0
    bg = int(values[int(order[0])])
    if len(order) > 1:
        assert int(counts[int(order[0])]) > int(counts[int(order[1])])
    return bg


def _is_contiguous(indices: np.ndarray) -> bool:
    if indices.size == 0:
        return False
    return int(indices[-1] - indices[0] + 1) == int(indices.size)


def _gap_side_from_missing(missing: np.ndarray) -> str | None:
    h, w = map(int, missing.shape)
    coords = np.argwhere(missing)
    if not (1 <= len(coords) <= 3):
        return None

    candidates: list[str] = []
    rows = coords[:, 0]
    cols = coords[:, 1]

    if bool(np.all(rows == 0)) and bool(np.all((0 < cols) & (cols < w - 1))) and _is_contiguous(np.sort(cols)):
        candidates.append("top")
    if bool(np.all(rows == h - 1)) and bool(np.all((0 < cols) & (cols < w - 1))) and _is_contiguous(np.sort(cols)):
        candidates.append("bottom")
    if bool(np.all(cols == 0)) and bool(np.all((0 < rows) & (rows < h - 1))) and _is_contiguous(np.sort(rows)):
        candidates.append("left")
    if bool(np.all(cols == w - 1)) and bool(np.all((0 < rows) & (rows < h - 1))) and _is_contiguous(np.sort(rows)):
        candidates.append("right")

    if len(candidates) != 1:
        return None
    return candidates[0]


def _find_frame_and_gap(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int], str]:
    candidates: list[tuple[int, int, tuple[int, int, int, int], str]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue

        coords = np.argwhere(grid == color)
        if len(coords) == 0:
            continue
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        mask = sub == color
        border = np.zeros_like(mask, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True

        if bool(np.any(mask & ~border)):
            continue

        missing = border & ~mask
        side = _gap_side_from_missing(missing)
        if side is None:
            continue

        if not np.array_equal(mask, border & ~missing):
            continue

        area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        candidates.append((area, color, (r0, c0, r1, c1), side))

    assert len(candidates) == 1, "expected exactly one gapped rectangular frame"
    _area, frame_color, bbox, side = candidates[0]
    return frame_color, bbox, side


def _mirror_interior(interior: np.ndarray, gap_side: str) -> np.ndarray:
    if gap_side in {"left", "right"}:
        return np.fliplr(interior)
    return np.flipud(interior)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the pattern inside a gapped frame, then scale it by the mirrored marker count.

    Input structure:
    - The background is the most common color in the grid.
    - One single-color rectangular frame is one pixel thick and has exactly one contiguous gap on one side.
    - Inside the frame is a smaller pattern drawn with one body color plus a few marker-color pixels, on background.

    Rule:
    - Find the unique frame and detect which side contains the gap.
    - Remove the frame, then mirror the full frame interior: use a left-right flip for a left/right gap, and an up-down flip for a top/bottom gap.
    - In the mirrored interior, identify the marker color as the less frequent non-background color and count its pixels.
    - Output the mirrored interior scaled by that count, replacing each interior pixel with an equal-colored N x N block.

    Key insight:
    The gap does not mark a destination; it encodes the reflection axis, while the marker pixels simultaneously remain part of the pattern and specify the output scale.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    _frame_color, (r0, c0, r1, c1), gap_side = _find_frame_and_gap(g, bg)
    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    assert interior.size > 0

    mirrored = _mirror_interior(interior, gap_side)
    colors, counts = np.unique(mirrored[mirrored != bg], return_counts=True)
    assert len(colors) == 2, "interior must contain one body color and one marker color"

    order = np.argsort(counts)
    marker_count = int(counts[int(order[0])])
    body_count = int(counts[int(order[1])])
    assert marker_count >= 1
    assert body_count > marker_count

    scale_block = np.ones((marker_count, marker_count), dtype=int)
    out = np.kron(mirrored, scale_block)
    return np.asarray(out, dtype=int)


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _grow_connected_mask(height: int, width: int, target_area: int) -> np.ndarray | None:
    if not (1 <= target_area <= height * width):
        return None

    mask = np.zeros((height, width), dtype=bool)
    start = (int(_RNG.integers(height)), int(_RNG.integers(width)))
    mask[start] = True
    frontier = set(_neighbors4(start[0], start[1], height, width))

    while int(mask.sum()) < target_area:
        if not frontier:
            return None
        options = list(frontier)
        rr, cc = options[int(_RNG.integers(len(options)))]
        frontier.remove((rr, cc))
        mask[rr, cc] = True
        for nbr in _neighbors4(rr, cc, height, width):
            if not mask[nbr]:
                frontier.add(nbr)

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) < 2 or len(cols) < 2:
        return None
    return mask


def _sample_shape(inner_h: int, inner_w: int, marker_count: int, gap_side: str) -> np.ndarray | None:
    interior = np.full((inner_h, inner_w), -1, dtype=int)
    max_box_h = inner_h if inner_h <= 4 else inner_h - int(_RNG.integers(0, 2))
    max_box_w = inner_w if inner_w <= 4 else inner_w - int(_RNG.integers(0, 2))
    max_box_h = max(2, min(inner_h, max_box_h))
    max_box_w = max(2, min(inner_w, max_box_w))

    box_h = int(_RNG.integers(2, max_box_h + 1))
    box_w = int(_RNG.integers(2, max_box_w + 1))
    box_area = int(box_h * box_w)

    min_area = max(2 * marker_count + 2, int(np.ceil(0.4 * box_area)))
    max_area = max(min_area, int(np.floor(0.75 * box_area)))
    if min_area > box_area:
        return None
    target_area = int(_RNG.integers(min_area, min(max_area, box_area) + 1))

    local_mask = _grow_connected_mask(box_h, box_w, target_area)
    if local_mask is None:
        return None

    if int(local_mask.sum()) <= 2 * marker_count:
        return None

    row0 = int(_RNG.integers(0, inner_h - box_h + 1))
    col0 = int(_RNG.integers(0, inner_w - box_w + 1))
    interior[:, :] = 0
    interior[row0 : row0 + box_h, col0 : col0 + box_w][local_mask] = 1

    cells = np.argwhere(interior == 1)
    if len(cells) <= marker_count:
        return None
    marker_idx = _RNG.choice(len(cells), size=marker_count, replace=False)
    for idx in np.atleast_1d(marker_idx):
        rr, cc = map(int, cells[int(idx)])
        interior[rr, cc] = 2

    if int(np.sum(interior == 1)) <= marker_count:
        return None

    mirrored = _mirror_interior(interior, gap_side)
    if np.array_equal(interior, mirrored):
        return None
    return interior


def _draw_frame(
    grid_h: int,
    grid_w: int,
    frame_top: int,
    frame_left: int,
    inner_h: int,
    inner_w: int,
    frame_color: int,
    bg: int,
    gap_side: str,
    gap_len: int,
    gap_start: int,
) -> np.ndarray:
    grid = np.full((grid_h, grid_w), bg, dtype=int)
    r0 = frame_top
    c0 = frame_left
    r1 = frame_top + inner_h + 1
    c1 = frame_left + inner_w + 1

    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color

    if gap_side == "top":
        grid[r0, c0 + 1 + gap_start : c0 + 1 + gap_start + gap_len] = bg
    elif gap_side == "bottom":
        grid[r1, c0 + 1 + gap_start : c0 + 1 + gap_start + gap_len] = bg
    elif gap_side == "left":
        grid[r0 + 1 + gap_start : r0 + 1 + gap_start + gap_len, c0] = bg
    else:
        grid[r0 + 1 + gap_start : r0 + 1 + gap_start + gap_len, c1] = bg

    return grid


def _expected_output(interior: np.ndarray, marker_count: int, gap_side: str, colors: tuple[int, int, int]) -> np.ndarray:
    bg, body_color, marker_color = colors
    colored = np.full(interior.shape, bg, dtype=int)
    colored[interior == 1] = body_color
    colored[interior == 2] = marker_color
    mirrored = _mirror_interior(colored, gap_side)
    return np.kron(mirrored, np.ones((marker_count, marker_count), dtype=int))


def _generate_case(gap_side: str, marker_count: int) -> np.ndarray | None:
    max_inner = min(12, 30 // marker_count)
    if max_inner < 4:
        return None

    inner_h = int(_RNG.integers(4, max_inner + 1))
    inner_w = int(_RNG.integers(4, max_inner + 1))

    if gap_side in {"top", "bottom"}:
        max_gap = min(3, inner_w)
    else:
        max_gap = min(3, inner_h)
    gap_len = int(_RNG.integers(1, max_gap + 1))
    gap_extent = inner_w if gap_side in {"top", "bottom"} else inner_h
    gap_start = int(_RNG.integers(0, gap_extent - gap_len + 1))

    frame_h = inner_h + 2
    frame_w = inner_w + 2
    grid_h = int(_RNG.integers(frame_h + 2, min(30, frame_h + 12) + 1))
    grid_w = int(_RNG.integers(frame_w + 2, min(30, frame_w + 12) + 1))
    frame_top = int(_RNG.integers(1, grid_h - frame_h))
    frame_left = int(_RNG.integers(1, grid_w - frame_w))

    bg, frame_color, body_color, marker_color = map(int, _RNG.choice(10, size=4, replace=False))
    interior_codes = _sample_shape(inner_h, inner_w, marker_count, gap_side)
    if interior_codes is None:
        return None

    grid = _draw_frame(
        grid_h=grid_h,
        grid_w=grid_w,
        frame_top=frame_top,
        frame_left=frame_left,
        inner_h=inner_h,
        inner_w=inner_w,
        frame_color=frame_color,
        bg=bg,
        gap_side=gap_side,
        gap_len=gap_len,
        gap_start=gap_start,
    )

    interior = np.full((inner_h, inner_w), bg, dtype=int)
    interior[interior_codes == 1] = body_color
    interior[interior_codes == 2] = marker_color
    grid[frame_top + 1 : frame_top + 1 + inner_h, frame_left + 1 : frame_left + 1 + inner_w] = interior

    counts = Counter(map(int, grid.reshape(-1)))
    most_common = counts.most_common(2)
    if len(most_common) < 2 or most_common[0][0] != bg or most_common[0][1] <= most_common[1][1]:
        return None

    expected = _expected_output(interior_codes, marker_count, gap_side, (bg, body_color, marker_color))
    if max(expected.shape) > 30:
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_COUNTER

    gap_side = SIDES[_GEN_COUNTER % len(SIDES)]
    marker_count = 1 + ((_GEN_COUNTER // len(SIDES)) % 4)

    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_case(gap_side, marker_count)
        if grid is not None:
            _GEN_COUNTER += 1
            return grid
    raise RuntimeError("failed to generate a valid input")
