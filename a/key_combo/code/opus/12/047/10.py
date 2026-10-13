from __future__ import annotations

import random
from collections import Counter

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(cells, dtype=int))
    return parts


def _color_patch(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[mask] = int(color)
    return patch


def _ordered_markers(grid: np.ndarray, bg: int, template_color: int) -> tuple[list[tuple[int, int, int]], bool]:
    counts = Counter(map(int, grid.ravel()))
    markers: list[tuple[int, int, int]] = []
    for color, count in counts.items():
        if color in (bg, template_color):
            continue
        assert int(count) == 1, "each marker color must appear exactly once"
        r, c = map(int, np.argwhere(grid == int(color))[0])
        markers.append((r, c, int(color)))

    assert len(markers) >= 2, "expected at least two markers"
    same_row = len({r for r, _c, _color in markers}) == 1
    same_col = len({c for _r, c, _color in markers}) == 1
    assert same_row != same_col, "markers must share exactly one row or one column"

    markers.sort(key=(lambda x: x[1]) if same_row else (lambda x: x[0]))
    return markers, same_row


def _build_output(mask: np.ndarray, markers: list[tuple[int, int, int]], horizontal: bool, bg: int, mode: str = "correct") -> np.ndarray:
    patches: list[np.ndarray] = []
    for idx, (_r, _c, color) in enumerate(markers, start=1):
        rotate = False
        if mode == "correct":
            rotate = (idx % 2 == 0)
        elif mode == "none":
            rotate = False
        elif mode == "all":
            rotate = True
        elif mode == "reverse":
            rotate = (idx % 2 == 1)
        else:
            raise ValueError(mode)

        patch = _color_patch(mask, color, bg)
        if rotate:
            patch = np.rot90(patch, -1)
        patches.append(patch)

    if horizontal:
        out_h = max(int(p.shape[0]) for p in patches)
        out_w = sum(int(p.shape[1]) for p in patches)
        out = np.full((out_h, out_w), int(bg), dtype=int)
        c0 = 0
        for patch in patches:
            h, w = patch.shape
            out[:h, c0 : c0 + w] = patch
            c0 += w
    else:
        out_h = sum(int(p.shape[0]) for p in patches)
        out_w = max(int(p.shape[1]) for p in patches)
        out = np.full((out_h, out_w), int(bg), dtype=int)
        r0 = 0
        for patch in patches:
            h, w = patch.shape
            out[r0 : r0 + h, :w] = patch
            r0 += h
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Copy the template in marker order, rotating only the even-ranked copies before concatenation.

    Input structure:
    - A dominant background contains one multi-pixel template color and several singleton marker colors.
    - The template is a single 4-connected object whose cropped bounding box is rectangular.
    - Every marker color appears exactly once, and all marker pixels lie on one common row or one common column.

    Rule:
    - Find the unique multi-pixel template color, crop it to its bounding box, and keep only its binary mask.
    - Sort markers left-to-right when they share a row, or top-to-bottom when they share a column.
    - For odd ranks, recolor the cropped template with that marker color and keep its orientation.
    - For even ranks, recolor the cropped template with that marker color and rotate that copy 90 degrees clockwise.
    - Concatenate the copies in sorted order; horizontal marker lines build a top-aligned horizontal strip, and vertical marker lines build a left-aligned vertical strip.

    Key insight:
    - The marker pixels encode both palette order and a hidden parity rule: rank decides whether each recolored template stays as-is or is rotated.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    counts = Counter(map(int, g.ravel()))
    fg_colors = [int(color) for color in counts if int(color) != bg]
    template_colors = [int(color) for color in fg_colors if int(counts[color]) > 1]
    assert len(template_colors) == 1, "expected exactly one multi-pixel template color"
    template_color = template_colors[0]

    template_mask = (g == template_color)
    parts = _components(template_mask)
    assert len(parts) == 1, "template color must form a single component"
    box = _bbox(template_mask)
    assert box is not None
    mask = _crop_bbox(template_mask, box)
    assert mask.any()

    markers, horizontal = _ordered_markers(g, bg, template_color)
    return _build_output(mask, markers, horizontal, bg)


_TEMPLATE_COLOR = 1
_MARKER_COLORS = [2, 3, 4, 5, 6, 7, 8, 9]
_SHAPES = [(3, 4), (4, 3)]
_MARKER_COUNTS = [4, 6]
_MAX_ATTEMPTS = 400


def _monotone_path_mask(h: int, w: int) -> np.ndarray:
    start = random.choice([(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)])
    end = (h - 1 - start[0], w - 1 - start[1])
    row_step = 1 if end[0] > start[0] else -1
    col_step = 1 if end[1] > start[1] else -1
    moves = [(row_step, 0)] * abs(end[0] - start[0]) + [(0, col_step)] * abs(end[1] - start[1])
    random.shuffle(moves)

    mask = np.zeros((h, w), dtype=bool)
    r, c = start
    mask[r, c] = True
    for dr, dc in moves:
        r += dr
        c += dc
        mask[r, c] = True
    return mask


def _grow_mask(h: int, w: int, area: int) -> np.ndarray | None:
    mask = _monotone_path_mask(h, w)
    while int(mask.sum()) < area:
        frontier: set[tuple[int, int]] = set()
        for r, c in np.argwhere(mask):
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((int(nr), int(nc)))
        if not frontier:
            return None
        nr, nc = random.choice(list(frontier))
        mask[nr, nc] = True

    if any(mask[r].sum() == 0 for r in range(h)):
        return None
    if any(mask[:, c].sum() == 0 for c in range(w)):
        return None
    return mask


def _sample_template_mask() -> np.ndarray:
    for _ in range(100):
        h, w = random.choice(_SHAPES)
        area = random.randint(max(h + w - 1, 6), min(h * w - 1, 9))
        mask = _grow_mask(h, w, area)
        if mask is None:
            continue
        if _bbox(mask) != (0, 0, h - 1, w - 1):
            continue
        return mask
    raise RuntimeError("failed to generate template mask")


def _sample_positions(length: int, count: int, gap: int) -> list[int] | None:
    positions = []
    current_min = 0
    for remaining in range(count, 0, -1):
        max_start = length - 1 - gap * (remaining - 1)
        if current_min > max_start:
            return None
        pos = random.randint(current_min, max_start)
        positions.append(pos)
        current_min = pos + gap
    random.shuffle(positions)
    positions.sort()
    return positions


def _make_grid(mask: np.ndarray, marker_colors: list[int], horizontal: bool) -> np.ndarray | None:
    th, tw = mask.shape
    marker_count = len(marker_colors)

    if horizontal:
        h = random.randint(10, 16)
        min_w = max(12, marker_count + (marker_count - 1) + 2)
        w = random.randint(min_w, 18)
    else:
        min_h = max(12, marker_count + (marker_count - 1) + 2)
        h = random.randint(min_h, 18)
        w = random.randint(10, 16)

    grid = np.zeros((h, w), dtype=int)
    tr = random.randint(0, h - th)
    tc = random.randint(0, w - tw)
    grid[tr : tr + th, tc : tc + tw][mask] = _TEMPLATE_COLOR

    if horizontal:
        candidate_rows = [r for r in range(h) if r < tr - 1 or r > tr + th]
        if not candidate_rows:
            return None
        marker_row = random.choice(candidate_rows)
        gap = random.choice([2, 3])
        positions = _sample_positions(w, marker_count, gap)
        if positions is None:
            return None
        for c, color in zip(positions, marker_colors):
            grid[marker_row, c] = int(color)
    else:
        candidate_cols = [c for c in range(w) if c < tc - 1 or c > tc + tw]
        if not candidate_cols:
            return None
        marker_col = random.choice(candidate_cols)
        gap = random.choice([2, 3])
        positions = _sample_positions(h, marker_count, gap)
        if positions is None:
            return None
        for r, color in zip(positions, marker_colors):
            grid[r, marker_col] = int(color)

    return grid


def _is_valid_input(grid: np.ndarray) -> bool:
    bg = _get_bg(grid)
    counts = Counter(map(int, grid.ravel()))
    template_colors = [int(color) for color, count in counts.items() if color != bg and int(count) > 1]
    if template_colors != [_TEMPLATE_COLOR]:
        return False
    if len(_components(grid == _TEMPLATE_COLOR)) != 1:
        return False

    marker_colors = [int(color) for color, count in counts.items() if color not in (bg, _TEMPLATE_COLOR)]
    if len(marker_colors) not in _MARKER_COUNTS:
        return False
    if any(int(counts[color]) != 1 for color in marker_colors):
        return False

    markers, horizontal = _ordered_markers(grid, bg, _TEMPLATE_COLOR)
    mask = _crop_bbox(grid == _TEMPLATE_COLOR, _bbox(grid == _TEMPLATE_COLOR))

    expected = _build_output(mask, markers, horizontal, bg, "correct")
    if not np.array_equal(main(grid), expected):
        return False

    wrong_outputs = [
        _build_output(mask, markers, horizontal, bg, "none"),
        _build_output(mask, markers, horizontal, bg, "all"),
        _build_output(mask, markers, horizontal, bg, "reverse"),
    ]
    if any(np.array_equal(expected, wrong) for wrong in wrong_outputs):
        return False

    return max(expected.shape) <= 30


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        mask = _sample_template_mask()
        horizontal = bool(random.getrandbits(1))
        marker_count = random.choice(_MARKER_COUNTS)
        marker_colors = random.sample(_MARKER_COLORS, marker_count)
        grid = _make_grid(mask, marker_colors, horizontal)
        if grid is None:
            continue
        if _is_valid_input(grid):
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
