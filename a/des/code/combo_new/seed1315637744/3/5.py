from __future__ import annotations

import random

import numpy as np


BG = 0
DIR_TO_VEC = {
    "left": (0, -1),
    "right": (0, 1),
    "up": (-1, 0),
    "down": (1, 0),
}
OPPOSITE = {
    "left": "right",
    "right": "left",
    "up": "down",
    "down": "up",
}
HINT_H = 4
ARROW_PATCH_W = 6
MIN_HINT_W = 2 * ARROW_PATCH_W + 5
GEN_RETRIES = 500


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    assert h >= 2 and w >= 2
    assert 0 <= r0 <= r0 + h - 1 < grid.shape[0]
    assert 0 <= c0 <= c0 + w - 1 < grid.shape[1]
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _frame_mask(shape: tuple[int, int], r0: int, c0: int, h: int, w: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    _draw_frame(mask, r0, c0, h, w, True)
    return mask


def _parse_arrow_direction(comp: list[tuple[int, int]]) -> str:
    coords = np.asarray(comp, dtype=int)
    assert coords.shape[0] == 4, "expected a 3-pixel shaft plus 1-pixel tip"

    row_values, row_counts = np.unique(coords[:, 0], return_counts=True)
    for row, count in zip(row_values, row_counts):
        if int(count) != 3:
            continue
        shaft = coords[coords[:, 0] == int(row)]
        cols = sorted(int(c) for c in shaft[:, 1])
        if cols != list(range(cols[0], cols[0] + 3)):
            continue
        tip = coords[coords[:, 0] != int(row)]
        assert tip.shape[0] == 1
        tr, tc = map(int, tip[0])
        assert abs(tr - int(row)) == 1
        if tc == cols[0]:
            return "left"
        if tc == cols[-1]:
            return "right"

    col_values, col_counts = np.unique(coords[:, 1], return_counts=True)
    for col, count in zip(col_values, col_counts):
        if int(count) != 3:
            continue
        shaft = coords[coords[:, 1] == int(col)]
        rows = sorted(int(r) for r in shaft[:, 0])
        if rows != list(range(rows[0], rows[0] + 3)):
            continue
        tip = coords[coords[:, 1] != int(col)]
        assert tip.shape[0] == 1
        tr, tc = map(int, tip[0])
        assert abs(tc - int(col)) == 1
        if tr == rows[0]:
            return "up"
        if tr == rows[-1]:
            return "down"

    raise AssertionError("unable to parse arrow direction")


def _find_composite_start(grid: np.ndarray) -> int:
    occupied_rows = np.flatnonzero(np.any(grid != BG, axis=1))
    assert occupied_rows.size > 0
    gaps = np.diff(occupied_rows)
    assert gaps.size > 0
    gap_index = int(np.argmax(gaps))
    assert int(gaps[gap_index]) > 1, "expected a blank gap between hints and frame composite"
    return int(occupied_rows[gap_index + 1])


def _extract_directions(hint_band: np.ndarray) -> dict[int, str]:
    directions: dict[int, str] = {}
    for color in sorted(int(v) for v in np.unique(hint_band) if int(v) != BG):
        comps = _components4(hint_band == int(color))
        arrow = max(comps, key=len)
        directions[int(color)] = _parse_arrow_direction(arrow)
    assert len(directions) == 2
    return directions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Separate two partially overlapping colored rectangle frames by the number of shared pixels.

    Input structure:
    - Black background with one lower composite made from colors A, B, and C.
    - The lower composite is the overlap of two same-sized one-pixel-thick rectangular frames:
      A plus C reconstructs one full frame, and B plus C reconstructs the other.
    - Above the composite is a direction-hint band containing one hint for color A and one for
      color B; each hint includes a singleton pixel and a same-colored arrow.

    Rule:
    - Use the hint band to identify the two moving frame colors and decode each arrow direction.
    - Treat every C cell as shared by both frames, so the full A-frame is all A-or-C cells and the
      full B-frame is all B-or-C cells.
    - Count the number N of C cells, then translate the full A-frame by N cells along A's arrow and
      the full B-frame by N cells along B's arrow.
    - Keep the hint band unchanged, clear the composite area to black, and draw the two translated
      full frames in colors A and B.

    Key insight:
    - The third color C carries two roles at once: it reveals the hidden portions of both frames and
      its pixel count determines how far the frames must be pulled apart.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    comp_start = _find_composite_start(g)
    hint_band = g[:comp_start, :]
    directions = _extract_directions(hint_band)
    frame_colors = set(directions)
    other_colors = set(int(v) for v in np.unique(g) if int(v) != BG) - frame_colors
    assert len(other_colors) == 1
    mix_color = int(next(iter(other_colors)))
    shift = int(np.count_nonzero(g[comp_start:, :] == mix_color))
    assert shift > 0

    out = np.zeros_like(g)
    out[:comp_start, :] = g[:comp_start, :]

    lower = g[comp_start:, :]
    for color, direction in directions.items():
        mask = np.isin(lower, [int(color), mix_color])
        box = _bbox(mask)
        assert box is not None
        rr0, cc0, rr1, cc1 = box
        h = int(rr1 - rr0 + 1)
        w = int(cc1 - cc0 + 1)
        dr, dc = DIR_TO_VEC[direction]
        abs_r0 = int(comp_start + rr0 + dr * shift)
        abs_c0 = int(cc0 + dc * shift)
        _draw_frame(out, abs_r0, abs_c0, h, w, int(color))

    return out


def _make_arrow_patch(direction: str) -> np.ndarray:
    patch = np.zeros((HINT_H, ARROW_PATCH_W), dtype=int)
    tip_side = random.choice([-1, 1])

    if direction in {"left", "right"}:
        shaft_row = 1 if tip_side == -1 else 2
        patch[shaft_row, 2:5] = 1
        tip_col = 2 if direction == "left" else 4
        patch[shaft_row + tip_side, tip_col] = 1
        patch[shaft_row, 0] = 1
    else:
        top_row = random.randint(0, 1)
        shaft_col = 3
        patch[top_row : top_row + 3, shaft_col] = 1
        tip_row = top_row if direction == "up" else top_row + 2
        tip_col = shaft_col + tip_side
        patch[tip_row, tip_col] = 1
        patch[top_row, 0] = 1

    return patch


def _paste_mask(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = patch.shape
    region = dst[r0 : r0 + h, c0 : c0 + w]
    region[patch.astype(bool)] = int(color)


def _compose_input(
    shape: tuple[int, int],
    a_box: tuple[int, int, int, int],
    b_box: tuple[int, int, int, int],
    colors: tuple[int, int, int],
    hint_specs: list[tuple[int, np.ndarray, int, int]],
) -> np.ndarray:
    a_color, b_color, mix_color = colors
    grid = np.zeros(shape, dtype=int)

    for color, patch, r0, c0 in hint_specs:
        _paste_mask(grid, patch, int(r0), int(c0), int(color))

    ar, ac, ah, aw = a_box
    br, bc, bh, bw = b_box
    mask_a = _frame_mask(shape, ar, ac, ah, aw)
    mask_b = _frame_mask(shape, br, bc, bh, bw)
    overlap = mask_a & mask_b
    grid[mask_a & ~overlap] = int(a_color)
    grid[mask_b & ~overlap] = int(b_color)
    grid[overlap] = int(mix_color)
    return grid


def _compose_output(
    shape: tuple[int, int],
    a_box: tuple[int, int, int, int],
    b_box: tuple[int, int, int, int],
    colors: tuple[int, int, int],
    hint_specs: list[tuple[int, np.ndarray, int, int]],
) -> np.ndarray:
    a_color, b_color, _mix_color = colors
    grid = np.zeros(shape, dtype=int)
    for color, patch, r0, c0 in hint_specs:
        _paste_mask(grid, patch, int(r0), int(c0), int(color))
    ar, ac, ah, aw = a_box
    br, bc, bh, bw = b_box
    _draw_frame(grid, ar, ac, ah, aw, a_color)
    _draw_frame(grid, br, bc, bh, bw, b_color)
    return grid


def _sample_case() -> tuple[np.ndarray, np.ndarray] | None:
    a_dir = random.choice(["left", "right", "up", "down"])
    b_dir = OPPOSITE[a_dir]
    a_color, b_color, mix_color = random.sample(range(1, 10), 3)

    if a_dir in {"left", "right"}:
        h = random.randint(4, 10)
        w = random.randint(4, 7)
        overlap_span = random.randint(1, min(4, w - 1))
        offset = w - overlap_span
    else:
        h = random.randint(4, 6)
        w = random.randint(4, 10)
        overlap_span = random.randint(1, min(3, h - 1))
        offset = h - overlap_span

    shift = 2 * overlap_span
    dr, dc = DIR_TO_VEC[a_dir]

    a_in = (0, 0)
    b_in = (-dr * offset, -dc * offset)
    a_out = (dr * shift, dc * shift)
    b_out = (b_in[0] - dr * shift, b_in[1] - dc * shift)

    positions = [a_in, b_in, a_out, b_out]
    frame_min_r = min(r for r, _ in positions)
    frame_min_c = min(c for _, c in positions)
    frame_max_r = max(r + h - 1 for r, _ in positions)
    frame_max_c = max(c + w - 1 for _, c in positions)
    frame_span_h = int(frame_max_r - frame_min_r + 1)
    frame_span_w = int(frame_max_c - frame_min_c + 1)

    hint_top = random.randint(0, 1)
    gap = random.randint(2, 4)
    frame_top_min = hint_top + HINT_H + gap
    min_h = frame_top_min + frame_span_h + 1
    min_w = max(MIN_HINT_W, frame_span_w + 2)
    if min_h > 30 or min_w > 30:
        return None

    grid_h = random.randint(min_h, min(30, min_h + 4))
    grid_w = random.randint(min_w, min(30, min_w + 6))

    frame_top = random.randint(frame_top_min, grid_h - frame_span_h - 1)
    frame_left = random.randint(1, grid_w - frame_span_w - 1)
    row_shift = frame_top - frame_min_r
    col_shift = frame_left - frame_min_c

    def place(pos: tuple[int, int]) -> tuple[int, int]:
        return int(pos[0] + row_shift), int(pos[1] + col_shift)

    a_in_r, a_in_c = place(a_in)
    b_in_r, b_in_c = place(b_in)
    a_out_r, a_out_c = place(a_out)
    b_out_r, b_out_c = place(b_out)

    hint_left = random.randint(0, 2)
    hint_right = random.randint(grid_w - ARROW_PATCH_W - 2, grid_w - ARROW_PATCH_W)
    if hint_right - hint_left < ARROW_PATCH_W + 3:
        return None

    a_patch = _make_arrow_patch(a_dir)
    b_patch = _make_arrow_patch(b_dir)
    if random.choice([False, True]):
        hint_specs = [
            (a_color, a_patch, hint_top, hint_left),
            (b_color, b_patch, hint_top, hint_right),
        ]
    else:
        hint_specs = [
            (b_color, b_patch, hint_top, hint_left),
            (a_color, a_patch, hint_top, hint_right),
        ]

    out_mask_a = _frame_mask((grid_h, grid_w), a_out_r, a_out_c, h, w)
    out_mask_b = _frame_mask((grid_h, grid_w), b_out_r, b_out_c, h, w)
    if np.any(out_mask_a & out_mask_b):
        return None

    inp = _compose_input(
        (grid_h, grid_w),
        (a_in_r, a_in_c, h, w),
        (b_in_r, b_in_c, h, w),
        (a_color, b_color, mix_color),
        hint_specs,
    )
    expected = _compose_output(
        (grid_h, grid_w),
        (a_out_r, a_out_c, h, w),
        (b_out_r, b_out_c, h, w),
        (a_color, b_color, mix_color),
        hint_specs,
    )

    if np.count_nonzero(inp == mix_color) != shift:
        return None
    if set(int(v) for v in np.unique(inp) if int(v) != BG) != {a_color, b_color, mix_color}:
        return None
    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        case = _sample_case()
        if case is not None:
            return case[0]
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
