from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GRAY = 5
STAMP_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
FRAME_COUNT_CHOICES = (3, 4)
LEFT_WIDTH_RANGE = (8, 12)
RIGHT_WIDTH_RANGE = (11, 16)
FRAME_GAP_RANGE = (1, 3)
FRAME_HEIGHT_CHOICES = {
    3: (5, 7, 9),
    4: (5, 7),
}
FRAME_WIDTH_CHOICES = {
    3: (5, 7, 9),
    4: (5, 7),
}
GRID_LIMIT = 30
GEN_ATTEMPTS = 300


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_separator(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    for c in range(w):
        color = int(grid[0, c])
        if color != BG and bool(np.all(grid[:, c] == color)):
            candidates.append((c, color))
    assert len(candidates) == 1, "expected exactly one full-height separator column"
    sep_col, gray = candidates[0]
    assert 0 < sep_col < w - 1, "separator must split the grid into left and right regions"
    return int(sep_col), int(gray)


def _extract_frames(grid: np.ndarray, sep_col: int, gray: int) -> list[tuple[int, int, int, int]]:
    right = grid[:, sep_col + 1 :]
    comps = _components(right == gray)
    frames: list[tuple[int, int, int, int]] = []
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 5 or w < 5:
            continue
        sub = right[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not bool(np.all(sub[border] == gray)):
            continue
        if not bool(np.all(sub[~border] == BG)):
            continue
        frames.append((int(r0), int(sep_col + 1 + c0), int(r1), int(sep_col + 1 + c1)))
    frames.sort(key=lambda box: (box[0], box[1]))
    assert frames, "expected at least one target frame"
    return frames


def _extract_stamps(
    grid: np.ndarray, sep_col: int, gray: int
) -> list[tuple[int, int, int, int]]:
    left = grid[:, :sep_col]
    comps = _components((left != BG) & (left != gray))
    stamps: list[tuple[int, int, int, int]] = []
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h == 3 and w == 3 and len(comp) == 9, "every colored object must be a solid 3x3 block"
        color = int(left[r0, c0])
        block = left[r0 : r1 + 1, c0 : c1 + 1]
        assert color not in (BG, gray)
        assert bool(np.all(block == color)), "stamp blocks must be monochrome and solid"

        row_counts = [int(np.count_nonzero(left[r0 + dr, :] == gray)) for dr in range(3)]
        nonzero_rows = [(dr, count) for dr, count in enumerate(row_counts) if count > 0]
        assert len(nonzero_rows) == 1, "each stamp must have exactly one stripe-bearing row"
        _stripe_row, target_index = nonzero_rows[0]
        stamps.append((int(r0), int(c0), int(color), int(target_index)))

    stamps.sort(key=lambda item: (item[0], item[1], item[2]))
    assert stamps, "expected at least one left-side stamp"
    return stamps


def _frame_stamp_top_left(frame: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = frame
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    assert inner_h >= 3 and inner_w >= 3
    sr = r0 + 1 + (inner_h - 3) // 2
    sc = c0 + 1 + (inner_w - 3) // 2
    return int(sr), int(sc)


def _solve(grid: np.ndarray) -> np.ndarray:
    sep_col, gray = _find_separator(grid)
    frames = _extract_frames(grid, sep_col, gray)
    stamps = _extract_stamps(grid, sep_col, gray)

    out = grid.copy()
    for _r0, _c0, color, target_index in stamps:
        assert 1 <= target_index <= len(frames), "stripe count must point to an existing frame"
        sr, sc = _frame_stamp_top_left(frames[target_index - 1])
        out[sr : sr + 3, sc : sc + 3] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each left-side colored 3x3 block into the gray target frame indexed by the gray stripe count on its row.

    Input structure:
    - Black background.
    - A full-height gray separator column splits the grid into a left stamp area and a right target area.
    - In the left area, each colored object is a solid 3x3 block, and exactly one of the block's three rows also contains gray single-cell stripes elsewhere on that row.
    - In the right area, gray hollow rectangular frames occupy different columns and are ordered by their top edges from top to bottom.

    Rule:
    - For each left-side colored 3x3 block, count the gray stripe pixels on the unique row that shares the block's row band, ignoring the separator column.
    - Let that count be N; select the N-th gray frame when the frames are sorted by top row.
    - Copy a solid 3x3 block of the source color into the center of the selected frame's interior.
    - Leave the original left area, separator, and frame outlines unchanged.

    Key insight:
    - The gray stripes are not decoration: their visible count is an index that chooses which target frame receives the colored stamp.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    return _solve(g)


def _sample_frame_layout(n_frames: int) -> tuple[int, list[int], list[int]] | None:
    for _ in range(60):
        heights = [random.choice(FRAME_HEIGHT_CHOICES[n_frames]) for _ in range(n_frames)]
        gaps = [random.randint(*FRAME_GAP_RANGE) for _ in range(n_frames - 1)]
        top_margin = random.randint(0, 2)
        bottom_margin = random.randint(0, 2)
        height = top_margin + sum(heights) + sum(gaps) + bottom_margin
        if 12 <= height <= GRID_LIMIT:
            return height, heights, [top_margin, *gaps]
    return None


def _sample_stamp_tops(height: int, count: int) -> list[int] | None:
    tops = list(range(height - 2))
    random.shuffle(tops)
    chosen: list[int] = []
    for top in tops:
        if all(top + 2 < other - 1 or other + 2 < top - 1 for other in chosen):
            chosen.append(top)
            if len(chosen) == count:
                return chosen
    return None


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    grid[r0, c0 : c0 + w] = GRAY
    grid[r0 + h - 1, c0 : c0 + w] = GRAY
    grid[r0 : r0 + h, c0] = GRAY
    grid[r0 : r0 + h, c0 + w - 1] = GRAY


def _validate_generated_input(grid: np.ndarray, target_to_color: dict[int, int]) -> None:
    sep_col, gray = _find_separator(grid)
    frames = _extract_frames(grid, sep_col, gray)
    stamps = _extract_stamps(grid, sep_col, gray)
    assert len(set(target_to_color)) == len(target_to_color)
    assert {target: color for _r0, _c0, color, target in stamps} == target_to_color

    out = _solve(grid)
    for target_index, color in target_to_color.items():
        sr, sc = _frame_stamp_top_left(frames[target_index - 1])
        assert bool(np.all(out[sr : sr + 3, sc : sc + 3] == color))

    unused = set(range(1, len(frames) + 1)) - set(target_to_color)
    for target_index in unused:
        sr, sc = _frame_stamp_top_left(frames[target_index - 1])
        assert bool(np.all(out[sr : sr + 3, sc : sc + 3] == BG))


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.choice(FRAME_COUNT_CHOICES)
    n_stamps = random.randint(2, n_frames)

    layout = _sample_frame_layout(n_frames)
    if layout is None:
        return None
    height, frame_heights, increments = layout

    left_width = random.randint(max(LEFT_WIDTH_RANGE[0], n_frames + 4), LEFT_WIDTH_RANGE[1])
    right_width = random.randint(RIGHT_WIDTH_RANGE[0], RIGHT_WIDTH_RANGE[1])
    width = left_width + 1 + right_width
    if width > GRID_LIMIT:
        return None

    frame_width_choices = [w for w in FRAME_WIDTH_CHOICES[n_frames] if w <= right_width]
    if not frame_width_choices:
        return None
    frame_widths = [random.choice(frame_width_choices) for _ in range(n_frames)]

    c0s: list[int] = []
    used_c0s: set[int] = set()
    for fw in frame_widths:
        options = [c for c in range(right_width - fw + 1) if c not in used_c0s]
        if not options:
            return None
        c0 = random.choice(options)
        used_c0s.add(c0)
        c0s.append(c0)

    grid = np.full((height, width), BG, dtype=int)
    sep_col = left_width
    grid[:, sep_col] = GRAY

    frames: list[tuple[int, int, int, int]] = []
    current_r = increments[0]
    for idx in range(n_frames):
        fh = frame_heights[idx]
        fw = frame_widths[idx]
        c0 = sep_col + 1 + c0s[idx]
        _draw_frame(grid, current_r, c0, fh, fw)
        frames.append((current_r, c0, current_r + fh - 1, c0 + fw - 1))
        if idx < n_frames - 1:
            current_r += fh + increments[idx + 1]

    stamp_tops = _sample_stamp_tops(height, n_stamps)
    if stamp_tops is None:
        return None
    random.shuffle(stamp_tops)

    colors = random.sample(STAMP_COLORS, n_stamps)
    target_indices = random.sample(list(range(1, n_frames + 1)), n_stamps)
    target_to_color: dict[int, int] = {}

    for top, color, target_index in zip(stamp_tops, colors, target_indices):
        square_c0 = random.randint(0, left_width - 3)
        grid[top : top + 3, square_c0 : square_c0 + 3] = color

        stripe_row = top + random.randint(0, 2)
        stripe_options = [c for c in range(left_width) if not (square_c0 <= c < square_c0 + 3)]
        if len(stripe_options) < target_index:
            return None
        stripe_cols = random.sample(stripe_options, target_index)
        grid[stripe_row, stripe_cols] = GRAY
        target_to_color[int(target_index)] = int(color)

    _validate_generated_input(grid, target_to_color)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
