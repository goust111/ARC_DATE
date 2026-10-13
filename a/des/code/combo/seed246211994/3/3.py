from __future__ import annotations

from collections import Counter
import random

import numpy as np


NODE_SIZE = 3
GAP_SIZE = 2

INPUT_STEP = 2
LOGICAL_ROWS_RANGE = (5, 7)
LOGICAL_COLS_RANGE = (5, 7)
OUTER_MARGIN_RANGE = (0, 2)
PALETTE_SIZE_OPTIONS = (3, 4)
GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in grid.ravel())
    return int(counts.most_common(1)[0][0])


def _extract_logical_grid(grid: np.ndarray) -> tuple[np.ndarray, int]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    rows = [r for r in range(g.shape[0]) if bool((g[r] != bg).any())]
    cols = [c for c in range(g.shape[1]) if bool((g[:, c] != bg).any())]
    if not rows or not cols:
        return np.zeros((0, 0), dtype=int), bg

    logical = np.full((len(rows), len(cols)), bg, dtype=int)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            val = int(g[r, c])
            if val != bg:
                logical[i, j] = val

    return logical, bg


def _connection_masks(logical: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = logical.shape
    used_h = np.zeros((h, w), dtype=bool)
    h_conn = np.zeros((h, max(0, w - 1)), dtype=bool)

    for r in range(h):
        for c in range(w - 1):
            color = int(logical[r, c])
            if color != bg and int(logical[r, c + 1]) == color:
                h_conn[r, c] = True
                used_h[r, c] = True
                used_h[r, c + 1] = True

    v_conn = np.zeros((max(0, h - 1), w), dtype=bool)
    for r in range(h - 1):
        for c in range(w):
            color = int(logical[r, c])
            if color == bg or int(logical[r + 1, c]) != color:
                continue
            if not bool(used_h[r, c]) and not bool(used_h[r + 1, c]):
                v_conn[r, c] = True

    return h_conn, v_conn, used_h


def _render_output(logical: np.ndarray, bg: int) -> np.ndarray:
    h, w = logical.shape
    if h == 0 or w == 0:
        return np.zeros((0, 0), dtype=int)

    out_h = h * NODE_SIZE + (h - 1) * GAP_SIZE
    out_w = w * NODE_SIZE + (w - 1) * GAP_SIZE
    out = np.full((out_h, out_w), int(bg), dtype=int)

    row_starts = [r * (NODE_SIZE + GAP_SIZE) for r in range(h)]
    col_starts = [c * (NODE_SIZE + GAP_SIZE) for c in range(w)]
    h_conn, v_conn, _used_h = _connection_masks(logical, bg)

    for r in range(h):
        for c in range(w):
            color = int(logical[r, c])
            if color == bg:
                continue
            rs = row_starts[r]
            cs = col_starts[c]
            out[rs : rs + NODE_SIZE, cs : cs + NODE_SIZE] = color

    for r in range(h):
        for c in range(w - 1):
            if not bool(h_conn[r, c]):
                continue
            color = int(logical[r, c])
            rs = row_starts[r]
            left_end = col_starts[c] + NODE_SIZE
            right_start = col_starts[c + 1]
            out[rs : rs + NODE_SIZE, left_end:right_start] = color

    for r in range(h - 1):
        for c in range(w):
            if not bool(v_conn[r, c]):
                continue
            color = int(logical[r, c])
            cs = col_starts[c]
            top_end = row_starts[r] + NODE_SIZE
            bottom_start = row_starts[r + 1]
            out[top_end:bottom_start, cs : cs + NODE_SIZE] = color

    return out


def _feature_summary(logical: np.ndarray, bg: int) -> tuple[int, int, int]:
    h_conn, v_conn, used_h = _connection_masks(logical, bg)
    h_count = int(h_conn.sum())
    v_count = int(v_conn.sum())

    blocked_vertical = 0
    for r in range(logical.shape[0] - 1):
        for c in range(logical.shape[1]):
            color = int(logical[r, c])
            if color == bg or int(logical[r + 1, c]) != color:
                continue
            if not bool(v_conn[r, c]) and (bool(used_h[r, c]) or bool(used_h[r + 1, c])):
                blocked_vertical += 1

    return h_count, v_count, blocked_vertical


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-color neighboring dots on a sparse logical lattice, then enlarge the result.

    Input structure:
    - Background is dominant.
    - Foreground pixels sit on sampled lattice intersections, with background gap rows/cols
      between logical rows and logical columns.

    Rule:
    - Compress the non-background rows/cols into a logical dot grid.
    - Draw every logical dot as a solid block.
    - Connect same-color horizontal logical neighbors first.
    - Connect same-color vertical logical neighbors only when neither endpoint
      participates in any horizontal connection.
    - Expand logical rows/cols and separator gaps into a thicker output image.
    """
    logical, bg = _extract_logical_grid(np.asarray(grid, dtype=int))
    if logical.size == 0:
        return np.asarray(grid, dtype=int).copy()
    return _render_output(logical, bg)


def _choose_palette() -> tuple[int, list[int]]:
    colors = list(range(10))
    random.shuffle(colors)
    palette_size = random.choice(PALETTE_SIZE_OPTIONS)
    bg = int(colors[0])
    palette = [int(c) for c in colors[1 : 1 + palette_size]]
    return bg, palette


def _place_shape(
    logical: np.ndarray,
    occupied: np.ndarray,
    shape: list[tuple[int, int]],
    color: int,
) -> bool:
    h, w = logical.shape
    max_dr = max(dr for dr, _dc in shape)
    max_dc = max(dc for _dr, dc in shape)

    candidates = [
        (r0, c0)
        for r0 in range(h - max_dr)
        for c0 in range(w - max_dc)
    ]
    random.shuffle(candidates)

    for r0, c0 in candidates:
        cells = [(r0 + dr, c0 + dc) for dr, dc in shape]
        if any(bool(occupied[r, c]) for r, c in cells):
            continue
        for r, c in cells:
            logical[r, c] = int(color)
            occupied[r, c] = True
        return True

    return False


def _neighbor_colors(logical: np.ndarray, r: int, c: int, bg: int) -> set[int]:
    seen: set[int] = set()
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < logical.shape[0] and 0 <= cc < logical.shape[1]:
            color = int(logical[rr, cc])
            if color != bg:
                seen.add(color)
    return seen


def _pick_isolated_color(logical: np.ndarray, r: int, c: int, bg: int, palette: list[int]) -> int:
    forbidden = _neighbor_colors(logical, r, c, bg)
    choices = [color for color in palette if color not in forbidden]
    if not choices:
        choices = list(palette)
    return int(random.choice(choices))


def _fill_missing_axes(logical: np.ndarray, bg: int, palette: list[int]) -> bool:
    h, w = logical.shape

    for _ in range(h * w * 3):
        row_counts = np.sum(logical != bg, axis=1)
        col_counts = np.sum(logical != bg, axis=0)
        empty_rows = [r for r in range(h) if int(row_counts[r]) == 0]
        empty_cols = [c for c in range(w) if int(col_counts[c]) == 0]
        if not empty_rows and not empty_cols:
            return True

        candidates: list[tuple[int, int]] = []
        for r in (empty_rows if empty_rows else list(range(h))):
            for c in (empty_cols if empty_cols else list(range(w))):
                if int(logical[r, c]) == bg:
                    candidates.append((r, c))

        if not candidates and empty_rows:
            for r in empty_rows:
                for c in range(w):
                    if int(logical[r, c]) == bg:
                        candidates.append((r, c))

        if not candidates and empty_cols:
            for c in empty_cols:
                for r in range(h):
                    if int(logical[r, c]) == bg:
                        candidates.append((r, c))

        if not candidates:
            return False

        both = [(r, c) for r, c in candidates if r in empty_rows and c in empty_cols]
        if both:
            candidates = both

        r, c = random.choice(candidates)
        logical[r, c] = _pick_isolated_color(logical, r, c, bg, palette)

    return False


def _add_extra_isolated_points(logical: np.ndarray, bg: int, palette: list[int]) -> None:
    extras = random.randint(0, 2)
    cells = [(r, c) for r in range(logical.shape[0]) for c in range(logical.shape[1]) if int(logical[r, c]) == bg]
    random.shuffle(cells)

    for r, c in cells:
        if extras == 0:
            break
        if random.random() < 0.5:
            logical[r, c] = _pick_isolated_color(logical, r, c, bg, palette)
            extras -= 1


def _render_input(logical: np.ndarray, bg: int) -> np.ndarray:
    top = random.randint(*OUTER_MARGIN_RANGE)
    bottom = random.randint(*OUTER_MARGIN_RANGE)
    left = random.randint(*OUTER_MARGIN_RANGE)
    right = random.randint(*OUTER_MARGIN_RANGE)

    h, w = logical.shape
    grid_h = top + (h - 1) * INPUT_STEP + 1 + bottom
    grid_w = left + (w - 1) * INPUT_STEP + 1 + right
    grid = np.full((grid_h, grid_w), int(bg), dtype=int)

    for r in range(h):
        for c in range(w):
            color = int(logical[r, c])
            if color != bg:
                grid[top + r * INPUT_STEP, left + c * INPUT_STEP] = color

    return grid


def _generate_logical() -> tuple[np.ndarray, int] | None:
    bg, palette = _choose_palette()
    h = random.randint(*LOGICAL_ROWS_RANGE)
    w = random.randint(*LOGICAL_COLS_RANGE)

    logical = np.full((h, w), int(bg), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    motif_colors = random.sample(palette, 3)
    horizontal_shape = [(0, dc) for dc in range(random.choice((2, 3)))]
    vertical_shape = [(0, 0), (1, 0)]
    blocked_shape = random.choice(
        [
            [(0, 0), (1, 0), (0, 1)],
            [(0, 1), (1, 1), (0, 0)],
            [(0, 0), (1, 0), (1, 1)],
            [(0, 1), (1, 1), (1, 0)],
        ]
    )

    if not _place_shape(logical, occupied, horizontal_shape, motif_colors[0]):
        return None
    if not _place_shape(logical, occupied, vertical_shape, motif_colors[1]):
        return None
    if not _place_shape(logical, occupied, blocked_shape, motif_colors[2]):
        return None

    if not _fill_missing_axes(logical, bg, palette):
        return None

    _add_extra_isolated_points(logical, bg, palette)

    if len({int(v) for v in logical.ravel() if int(v) != bg}) < 2:
        return None

    point_count = int(np.sum(logical != bg))
    if point_count < max(logical.shape) or point_count > (logical.shape[0] * logical.shape[1]) // 2 + 2:
        return None

    h_count, v_count, blocked_count = _feature_summary(logical, bg)
    if h_count < 1 or v_count < 1 or blocked_count < 1:
        return None

    return logical, bg


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        generated = _generate_logical()
        if generated is None:
            continue

        logical, bg = generated
        inp = _render_input(logical, bg)
        expected = _render_output(logical, bg)
        if np.array_equal(main(inp), expected):
            return inp

    raise RuntimeError("failed to generate a valid sparse-lattice input")
