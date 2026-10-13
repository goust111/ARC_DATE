from __future__ import annotations

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 300
_RNG = np.random.default_rng()
_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _segments(line: np.ndarray) -> list[tuple[int, int, int]]:
    segs: list[tuple[int, int, int]] = []
    start = 0
    color = int(line[0])
    for i in range(1, len(line) + 1):
        if i == len(line) or int(line[i]) != color:
            segs.append((start, i - 1, color))
            if i < len(line):
                start = i
                color = int(line[i])
    return segs


def _find_fold_marker(grid: np.ndarray) -> tuple[str, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h >= 4 and w >= 4

    inner_h = h - 2
    inner_w = w - 2
    top = g[0, 1:-1]
    bottom = g[h - 1, 1:-1]
    left = g[1:-1, 0]
    right = g[1:-1, w - 1]

    candidates: list[tuple[str, int]] = []

    for s0, e0, color0 in _segments(top):
        if color0 == BG:
            continue
        for s1, e1, color1 in _segments(bottom):
            if color0 != color1:
                continue
            if s0 == s1 and e0 == e1 and s0 + e0 == inner_w - 1:
                assert inner_w % 2 == 0
                candidates.append(("vertical", color0))

    for s0, e0, color0 in _segments(left):
        if color0 == BG:
            continue
        for s1, e1, color1 in _segments(right):
            if color0 != color1:
                continue
            if s0 == s1 and e0 == e1 and s0 + e0 == inner_h - 1:
                assert inner_h % 2 == 0
                candidates.append(("horizontal", color0))

    assert len(candidates) == 1, f"expected one fold marker, got {candidates}"
    return candidates[0]


def _merge_halves(a: np.ndarray, b: np.ndarray, marker_color: int) -> np.ndarray:
    return np.where(a == BG, b, np.where(b == BG, a, int(marker_color))).astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer a horizontal or vertical fold axis from the unique mirrored border segment pair,
    fold along that axis, and merge overlapping pixels.

    Border structure used here:
    - The frame is one cell thick.
    - The fold marker is the unique same-color segment pair on opposite sides whose
      interval is centered on that side.
    - If both folded pixels are non-background, the marker color wins.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    marker_dir, marker_color = _find_fold_marker(g)
    inner = g[1:-1, 1:-1]

    if marker_dir == "vertical":
        _, inner_w = inner.shape
        half = inner_w // 2
        left = inner[:, :half]
        right = np.fliplr(inner[:, half:])
        return _merge_halves(left, right, marker_color)

    inner_h, _ = inner.shape
    half = inner_h // 2
    top = inner[:half, :]
    bottom = np.flipud(inner[half:, :])
    return _merge_halves(top, bottom, marker_color)


def _random_shape(max_h: int, max_w: int, min_area: int, max_area: int) -> np.ndarray:
    h = int(_RNG.integers(1, max_h + 1))
    w = int(_RNG.integers(1, max_w + 1))
    max_cells = min(max_area, h * w)
    area = int(_RNG.integers(min(min_area, max_cells), max_cells + 1))

    mask = np.zeros((h, w), dtype=bool)
    sr = int(_RNG.integers(0, h))
    sc = int(_RNG.integers(0, w))
    mask[sr, sc] = True

    while int(mask.sum()) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in np.argwhere(mask):
            for dr, dc in _N4:
                rr = int(r) + dr
                cc = int(c) + dc
                if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc]:
                    frontier.append((rr, cc))
        rr, cc = frontier[int(_RNG.integers(0, len(frontier)))]
        mask[rr, cc] = True

    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _place_shape(blocked: np.ndarray, shape: np.ndarray) -> tuple[int, int] | None:
    h, w = blocked.shape
    sh, sw = shape.shape
    positions = [(r, c) for r in range(h - sh + 1) for c in range(w - sw + 1)]
    for idx in _RNG.permutation(len(positions)):
        r0, c0 = positions[int(idx)]
        if not bool((blocked[r0 : r0 + sh, c0 : c0 + sw] & shape).any()):
            return int(r0), int(c0)
    return None


def _mark_blocked(blocked: np.ndarray, shape: np.ndarray, r0: int, c0: int) -> None:
    h, w = blocked.shape
    for r, c in np.argwhere(shape):
        rr = int(r0 + r)
        cc = int(c0 + c)
        blocked[max(0, rr - 1) : min(h, rr + 2), max(0, cc - 1) : min(w, cc + 2)] = True


def _pick_color(colors: list[int], forbid: int | None = None) -> int:
    choices = [color for color in colors if color != forbid]
    return int(_RNG.choice(choices))


def _build_fold_layers(shape: tuple[int, int], object_colors: list[int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    first = np.zeros((h, w), dtype=int)
    second = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    area = h * w
    max_clusters = min(8, max(4, area // 6))
    cluster_count = int(_RNG.integers(4, max_clusters + 1))
    modes = ["first", "second", "both"]
    for _ in range(cluster_count - 3):
        modes.append(str(_RNG.choice(["first", "second", "both"])))
    _RNG.shuffle(modes)

    for index, mode in enumerate(modes):
        placed = False
        for _ in range(40):
            shape_mask = _random_shape(min(3, h), min(3, w), 2 if index < 3 else 1, 4)
            pos = _place_shape(blocked, shape_mask)
            if pos is None:
                continue
            r0, c0 = pos
            coords = np.argwhere(shape_mask) + np.array([r0, c0], dtype=int)

            if mode in {"first", "both"}:
                color = _pick_color(object_colors)
                first[coords[:, 0], coords[:, 1]] = color
            if mode in {"second", "both"}:
                if mode == "both":
                    first_color = int(first[coords[0, 0], coords[0, 1]])
                    color = first_color if bool(_RNG.integers(0, 2)) else _pick_color(object_colors, forbid=first_color)
                else:
                    color = _pick_color(object_colors)
                second[coords[:, 0], coords[:, 1]] = color

            _mark_blocked(blocked, shape_mask, r0, c0)
            placed = True
            break
        if not placed:
            raise RuntimeError("failed to place a cluster")

    return first, second


def _random_segmented_line(length: int, colors: list[int]) -> np.ndarray:
    line = np.empty(length, dtype=int)
    pos = 0
    prev: int | None = None
    while pos < length:
        seg_len = int(_RNG.integers(1, min(5, length - pos + 1)))
        color = _pick_color(colors, forbid=prev)
        line[pos : pos + seg_len] = color
        prev = color
        pos += seg_len
    return line


def _line_with_marker(length: int, marker_color: int, other_colors: list[int]) -> np.ndarray:
    max_len = length - 2
    choices = [seg_len for seg_len in range(2, max_len + 1, 2)]
    marker_len = int(_RNG.choice(choices))
    start = (length - marker_len) // 2
    end = start + marker_len

    line = np.empty(length, dtype=int)
    if start > 0:
        line[:start] = _random_segmented_line(start, other_colors)
    line[start:end] = int(marker_color)
    if end < length:
        line[end:] = _random_segmented_line(length - end, other_colors)
    return line


def _build_border(
    inner_shape: tuple[int, int],
    orientation: str,
    marker_color: int,
    border_colors: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    inner_h, inner_w = inner_shape
    if orientation == "vertical":
        top = _line_with_marker(inner_w, marker_color, border_colors)
        bottom = top.copy() if bool(_RNG.integers(0, 2)) else _line_with_marker(inner_w, marker_color, border_colors)
        bottom[(bottom != marker_color)] = _random_segmented_line(int(np.sum(bottom != marker_color)), border_colors)
        left = _random_segmented_line(inner_h, border_colors)
        right = _random_segmented_line(inner_h, border_colors)
    else:
        left = _line_with_marker(inner_h, marker_color, border_colors)
        right = left.copy() if bool(_RNG.integers(0, 2)) else _line_with_marker(inner_h, marker_color, border_colors)
        right[(right != marker_color)] = _random_segmented_line(int(np.sum(right != marker_color)), border_colors)
        top = _random_segmented_line(inner_w, border_colors)
        bottom = _random_segmented_line(inner_w, border_colors)
    return top, bottom, left, right


def _assemble_grid(
    inner: np.ndarray,
    top: np.ndarray,
    bottom: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    corner_colors: list[int],
) -> np.ndarray:
    inner_h, inner_w = inner.shape
    g = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
    g[1:-1, 1:-1] = inner
    g[0, 1:-1] = top
    g[-1, 1:-1] = bottom
    g[1:-1, 0] = left
    g[1:-1, -1] = right
    g[0, 0] = corner_colors[0]
    g[0, -1] = corner_colors[1]
    g[-1, 0] = corner_colors[2]
    g[-1, -1] = corner_colors[3]
    return g


def _gen_attempt() -> np.ndarray | None:
    orientation = str(_RNG.choice(["vertical", "horizontal"]))
    marker_color = int(_RNG.choice(PALETTE))
    remaining = [color for color in PALETTE if color != marker_color]
    _RNG.shuffle(remaining)
    object_colors = remaining[:3]
    border_colors = remaining[3:6]

    if orientation == "vertical":
        out_h = int(_RNG.integers(6, 13))
        out_w = int(_RNG.integers(4, 9))
        first, second = _build_fold_layers((out_h, out_w), object_colors)
        inner = np.concatenate([first, np.fliplr(second)], axis=1)
    else:
        out_h = int(_RNG.integers(4, 9))
        out_w = int(_RNG.integers(6, 13))
        first, second = _build_fold_layers((out_h, out_w), object_colors)
        inner = np.concatenate([first, np.flipud(second)], axis=0)

    expected = _merge_halves(first, second, marker_color)
    top, bottom, left, right = _build_border(inner.shape, orientation, marker_color, border_colors)
    corners = [int(_RNG.choice(border_colors)) for _ in range(4)]
    grid = _assemble_grid(inner, top, bottom, left, right, corners)

    try:
        direction, color = _find_fold_marker(grid)
    except AssertionError:
        return None
    if direction != orientation or color != marker_color:
        return None

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            grid = _gen_attempt()
        except (AssertionError, RuntimeError, ValueError):
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
