from __future__ import annotations

from collections import deque

import numpy as np


BACKGROUND = 0
RAY = 1
DEPTH_COLORS = (0, 3, 4, 7, 2, 6)

GEN_ATTEMPTS = 300
AXIS_ATTEMPTS = 200
MARKER_COUNT_CHOICES = np.array([2, 3, 4], dtype=int)
MARKER_COUNT_WEIGHTS = np.array([0.30, 0.24, 0.46], dtype=float)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    comp_id = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            comp_id += 1
            q = deque([(r, c)])
            labels[r, c] = comp_id
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = comp_id
                        q.append((nr, nc))
    return labels, comp_id


def _extract_marker_axes(grid: np.ndarray) -> tuple[list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    blue = np.argwhere(g == RAY)
    assert blue.size > 0, "expected boundary markers"
    for r, c in blue:
        assert r in (0, h - 1) or c in (0, w - 1), "markers must lie on the boundary"
        assert not (r in (0, h - 1) and c in (0, w - 1)), "markers must not occupy corners"
    assert np.all((g == BACKGROUND) | (g == RAY)), "input should contain only black background and blue markers"

    left_rows = sorted(int(r) for r in np.where(g[:, 0] == RAY)[0])
    right_rows = sorted(int(r) for r in np.where(g[:, w - 1] == RAY)[0])
    top_cols = sorted(int(c) for c in np.where(g[0, :] == RAY)[0])
    bottom_cols = sorted(int(c) for c in np.where(g[h - 1, :] == RAY)[0])

    assert left_rows == right_rows, "left/right marker rows must match"
    assert top_cols == bottom_cols, "top/bottom marker cols must match"
    assert len(left_rows) == len(top_cols), "horizontal and vertical marker counts must match"
    assert 2 <= len(left_rows) <= 4, "each side must contain 2-4 markers"
    return left_rows, top_cols


def _build_ray_masks(shape: tuple[int, int], rows: list[int], cols: list[int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    horizontal = np.zeros((h, w), dtype=bool)
    vertical = np.zeros((h, w), dtype=bool)
    horizontal[rows, :] = True
    vertical[:, cols] = True
    return horizontal, vertical


def _component_depths(horizontal: np.ndarray, vertical: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    ray_mask = horizontal | vertical
    open_mask = ~ray_mask
    labels, n_comp = _label_components(open_mask)
    h, w = labels.shape

    adj = [set() for _ in range(n_comp + 1)]
    exterior = set()

    for r in range(h):
        for c in range(w):
            comp = labels[r, c]
            if comp == 0:
                continue
            if r in (0, h - 1) or c in (0, w - 1):
                exterior.add(int(comp))

    for r in range(h):
        for c in range(w):
            if horizontal[r, c]:
                up = labels[r - 1, c] if r - 1 >= 0 else 0
                down = labels[r + 1, c] if r + 1 < h else 0
                if up and down and up != down:
                    adj[int(up)].add(int(down))
                    adj[int(down)].add(int(up))
            if vertical[r, c]:
                left = labels[r, c - 1] if c - 1 >= 0 else 0
                right = labels[r, c + 1] if c + 1 < w else 0
                if left and right and left != right:
                    adj[int(left)].add(int(right))
                    adj[int(right)].add(int(left))

    depth = {}
    q = deque()
    for comp in sorted(exterior):
        depth[int(comp)] = 0
        q.append(int(comp))

    while q:
        comp = q.popleft()
        for nxt in adj[comp]:
            if nxt not in depth:
                depth[nxt] = depth[comp] + 1
                q.append(nxt)

    assert len(depth) == n_comp, "all background regions must be reachable from the exterior through ray crossings"
    return labels, depth


def _depth_color(depth: int) -> int:
    return int(DEPTH_COLORS[min(depth, len(DEPTH_COLORS) - 1)])


def _render_from_axes(shape: tuple[int, int], rows: list[int], cols: list[int]) -> np.ndarray:
    horizontal, vertical = _build_ray_masks(shape, rows, cols)
    labels, depths = _component_depths(horizontal, vertical)
    out = np.zeros(shape, dtype=int)
    ray_mask = horizontal | vertical
    out[ray_mask] = RAY
    for comp_id, depth in depths.items():
        if depth == 0:
            continue
        out[labels == comp_id] = _depth_color(depth)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the boundary markers into truncated rays and color each compartment by enclosure depth.

    Input structure:
    - The grid is mostly black background with blue single-pixel markers on the four borders.
    - Left/right border markers share the same set of rows, and top/bottom border markers share
      the same set of columns.
    - Each marker emits an inward ray perpendicular to its border; opposite markers on the same
      row or column meet and complete one blue wall.

    Rule:
    - Recover the full blue ray network by extending every marked row horizontally and every marked
      column vertically until the opposing same-line ray is reached, so each marked row/column
      becomes a blue wall.
    - Treat the blue rays as 4-connected barriers that split the black background into regions.
    - Depth 0 regions stay black; depth 1, 2, 3, 4, 5+ are colored green, yellow, orange, red,
      and pink respectively, while blue ray pixels stay blue.

    Key insight:
    The border markers do not mark isolated endpoints; they specify complete horizontal and vertical
    wall lines, and the answer depends on the minimum number of blue walls that must be crossed to
    reach each compartment from the exterior.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    rows, cols = _extract_marker_axes(g)
    return _render_from_axes(g.shape, rows, cols)


def _sample_axis_positions(length: int, count: int, rng: np.random.Generator) -> list[int] | None:
    inner = np.arange(1, length - 1, dtype=int)
    if inner.size < count:
        return None

    min_gap = {2: 5, 3: 4, 4: 3}[count]
    for _ in range(AXIS_ATTEMPTS):
        pos = sorted(int(x) for x in rng.choice(inner, size=count, replace=False))
        if min(pos[i + 1] - pos[i] for i in range(count - 1)) < min_gap:
            continue
        if pos[0] < 1 or pos[-1] > length - 2:
            continue
        if pos[-1] - pos[0] < 2 * min_gap:
            continue
        if count == 3 and not (pos[0] + 2 <= pos[1] <= pos[2] - 2):
            continue
        if count == 4 and not (pos[0] + 2 <= pos[1] < pos[2] - 1 <= pos[3] - 2):
            continue
        return pos
    return None


def _make_input(shape: tuple[int, int], rows: list[int], cols: list[int]) -> np.ndarray:
    h, w = shape
    inp = np.zeros((h, w), dtype=int)
    inp[rows, 0] = RAY
    inp[rows, w - 1] = RAY
    inp[0, cols] = RAY
    inp[h - 1, cols] = RAY
    return inp


def _positive_depth_region_counts(out: np.ndarray) -> dict[int, int]:
    counts: dict[int, int] = {}
    for depth, color in enumerate(DEPTH_COLORS):
        if depth == 0:
            continue
        labels, n = _label_components(out == color)
        if n:
            counts[depth] = int(n)
    return counts


def _validate_candidate(
    inp: np.ndarray,
    rows: list[int],
    cols: list[int],
    expected_max_depth: int,
) -> bool:
    try:
        parsed_rows, parsed_cols = _extract_marker_axes(inp)
    except Exception:
        return False
    if parsed_rows != rows or parsed_cols != cols:
        return False

    expected = _render_from_axes(inp.shape, rows, cols)
    try:
        out = main(inp)
    except Exception:
        return False
    if not np.array_equal(out, expected):
        return False

    horizontal, vertical = _build_ray_masks(inp.shape, rows, cols)
    ray_mask = horizontal | vertical
    if not np.array_equal(out == RAY, ray_mask):
        return False

    labels, depths = _component_depths(horizontal, vertical)
    max_depth = max(depths.values(), default=0)
    if max_depth != expected_max_depth:
        return False

    positive_regions = _positive_depth_region_counts(out)
    if expected_max_depth >= 1 and 1 not in positive_regions:
        return False
    if expected_max_depth >= 2 and 2 not in positive_regions:
        return False

    for comp_id, depth in depths.items():
        if depth == 0:
            if np.any(out[labels == comp_id] != BACKGROUND):
                return False
        else:
            if np.any(out[labels == comp_id] != _depth_color(depth)):
                return False

    if np.array_equal(inp, out):
        return False
    return True


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    count = int(rng.choice(MARKER_COUNT_CHOICES, p=MARKER_COUNT_WEIGHTS))
    expected_max_depth = count // 2

    h_min = {2: 13, 3: 16, 4: 21}[count]
    w_min = {2: 13, 3: 16, 4: 21}[count]
    h = int(rng.integers(h_min, 29))
    w = int(rng.integers(w_min, 29))

    rows = _sample_axis_positions(h, count, rng)
    cols = _sample_axis_positions(w, count, rng)
    if rows is None or cols is None:
        return None

    inp = _make_input((h, w), rows, cols)
    if not _validate_candidate(inp, rows, cols, expected_max_depth):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = np.random.default_rng()
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-depth input")
