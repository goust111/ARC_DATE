import random
from collections import deque
from typing import Iterable

import numpy as np


BLACK = 0
RED = 2
PALETTE_COLORS = [1, 3, 4, 5, 6, 7, 8, 9]

# --------------- solver ---------------


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _palette_strip(row: np.ndarray) -> list[int]:
    best: list[int] = []
    start = None
    for idx, value in enumerate(map(int, row)):
        if value not in (BLACK, RED):
            if start is None:
                start = idx
        elif start is not None:
            chunk = list(map(int, row[start:idx]))
            if len(chunk) > len(best):
                best = chunk
            start = None
    if start is not None:
        chunk = list(map(int, row[start:]))
        if len(chunk) > len(best):
            best = chunk

    assert len(best) >= 4, "palette strip must contain at least 4 colors"
    assert len(set(best)) == len(best), "palette strip colors must be distinct"
    return best


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    mask = np.zeros_like(grid, dtype=bool)
    mask[1:] = (grid[1:] != BLACK) & (grid[1:] != RED)
    frames: list[tuple[int, int, int, int, int]] = []
    for comp in _components(mask):
        colors = {int(grid[r, c]) for r, c in comp}
        assert len(colors) == 1, "each frame component must be monochrome"
        color = next(iter(colors))
        r0, c0, r1, c1 = _bbox(comp)
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must enclose an interior"

        expected = {
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)
        }
        assert expected == set(comp), "component must be a 1-cell-thick rectangular frame"
        frames.append((r0, c0, r1, c1, color))

    return sorted(frames)


def _coverage_counts(inner_h: int, inner_w: int, towers: list[tuple[int, int]]) -> np.ndarray:
    counts = np.zeros((inner_h, inner_w), dtype=int)
    row_values = sorted({r for r, _ in towers})
    col_values = sorted({c for _, c in towers})

    for r, c in towers:
        left = 0
        for cc in col_values:
            if cc < c:
                left = cc
            else:
                break

        right = inner_w - 1
        for cc in col_values:
            if cc > c:
                right = cc
                break

        up = 0
        for rr in row_values:
            if rr < r:
                up = rr
            else:
                break

        down = inner_h - 1
        for rr in row_values:
            if rr > r:
                down = rr
                break

        counts[r, left : right + 1] += 1
        counts[up : down + 1, c] += 1

    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color each framed rectangle interior by tower-coverage depth.

    The frame color chooses the start position in the top-row palette strip.
    Inside a frame, each red tower projects a cross that stops at the frame wall
    or the nearest other tower row/column reached by that arm. Background cells
    covered by 1/2/3 towers take the first/second/third palette color in that
    frame's shifted sequence.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= 5 and w >= 4, "grid is too small for this task"

    palette = _palette_strip(g[0])
    palette_index = {color: idx for idx, color in enumerate(palette)}

    out = g.copy()
    for r0, c0, r1, c1, frame_color in _find_frames(g):
        assert frame_color in palette_index, "frame color must appear in the palette strip"

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        assert np.all((interior == BLACK) | (interior == RED)), "frame interior must contain only black/red"

        local_towers = [tuple(map(int, pos)) for pos in np.argwhere(interior == RED)]
        assert 1 <= len(local_towers) <= 3, "each frame must contain 1 to 3 towers"

        counts = _coverage_counts(interior.shape[0], interior.shape[1], local_towers)
        base = palette_index[frame_color]

        for rr in range(interior.shape[0]):
            for cc in range(interior.shape[1]):
                if int(interior[rr, cc]) != BLACK:
                    continue
                cover = int(counts[rr, cc])
                if cover == 0:
                    continue
                assert base + cover - 1 < len(palette), "palette strip is too short for this frame"
                out[r0 + 1 + rr, c0 + 1 + cc] = palette[base + cover - 1]

    return out


# --------------- generator ---------------

PALETTE_LEN_RANGE = (6, 8)
FRAME_COUNT_CHOICES = (1, 2, 3)
TOP_GAP_RANGE = (1, 2)
FRAME_GAP_RANGE = (1, 3)
BOTTOM_GAP_RANGE = (1, 2)
MAX_GRID_HEIGHT = 28
GEN_RETRIES = 400

OUTER_SIZE_BY_TOWERS = {
    1: ((4, 7), (4, 6)),
    2: ((5, 8), (5, 6)),
    3: ((6, 8), (5, 7)),
}


def _sample_palette() -> list[int]:
    return random.sample(PALETTE_COLORS, random.randint(*PALETTE_LEN_RANGE))


def _assign_frame_indices(palette_len: int, tower_counts: list[int]) -> list[int] | None:
    remaining = set(range(palette_len))
    frame_indices = [None] * len(tower_counts)
    order = sorted(range(len(tower_counts)), key=lambda i: -tower_counts[i])
    for i in order:
        max_idx = palette_len - tower_counts[i]
        choices = [idx for idx in remaining if idx <= max_idx]
        if not choices:
            return None
        pick = random.choice(choices)
        frame_indices[i] = pick
        remaining.remove(pick)
    return [int(x) for x in frame_indices]


def _sample_frame_boxes(width: int, tower_counts: list[int]) -> list[dict[str, int]] | None:
    boxes: list[dict[str, int]] = []
    current_top = 1 + random.randint(*TOP_GAP_RANGE)
    for tower_count in tower_counts:
        width_range, height_range = OUTER_SIZE_BY_TOWERS[tower_count]
        min_w = width_range[0]
        max_w = min(width, width_range[1])
        if max_w < min_w:
            return None
        outer_w = random.randint(min_w, max_w)
        outer_h = random.randint(height_range[0], height_range[1])
        left = random.randint(0, width - outer_w)
        boxes.append(
            {
                "top": current_top,
                "left": left,
                "height": outer_h,
                "width": outer_w,
                "tower_count": tower_count,
            }
        )
        current_top += outer_h + random.randint(*FRAME_GAP_RANGE)

    total_height = current_top - random.randint(*FRAME_GAP_RANGE) + random.randint(*BOTTOM_GAP_RANGE)
    if total_height > MAX_GRID_HEIGHT:
        return None
    return boxes


def _place_one_tower(inner_h: int, inner_w: int) -> list[tuple[int, int]]:
    return [(random.randrange(inner_h), random.randrange(inner_w))]


def _place_two_towers(inner_h: int, inner_w: int) -> list[tuple[int, int]]:
    rows = random.sample(range(inner_h), 2)
    cols = random.sample(range(inner_w), 2)
    if random.random() < 0.5:
        return [(rows[0], cols[0]), (rows[1], cols[1])]
    return [(rows[0], cols[1]), (rows[1], cols[0])]


def _place_three_towers(inner_h: int, inner_w: int) -> list[tuple[int, int]] | None:
    choices = []
    if inner_h >= 2 and inner_w >= 3:
        choices.append("shared_row")
    if inner_h >= 3 and inner_w >= 2:
        choices.append("shared_col")
    if not choices:
        return None

    mode = random.choice(choices)
    if mode == "shared_row":
        shared_row, other_row = random.sample(range(inner_h), 2)
        c1, c2, c3 = sorted(random.sample(range(inner_w), 3))
        return [(shared_row, c1), (shared_row, c3), (other_row, c2)]

    shared_col, other_col = random.sample(range(inner_w), 2)
    r1, r2, r3 = sorted(random.sample(range(inner_h), 3))
    return [(r1, shared_col), (r3, shared_col), (r2, other_col)]


def _place_towers(inner_h: int, inner_w: int, tower_count: int) -> list[tuple[int, int]] | None:
    if tower_count == 1:
        return _place_one_tower(inner_h, inner_w)
    if tower_count == 2:
        if inner_h < 2 or inner_w < 2:
            return None
        return _place_two_towers(inner_h, inner_w)
    if tower_count == 3:
        return _place_three_towers(inner_h, inner_w)
    return None


def _tower_pattern_is_informative(inner_h: int, inner_w: int, towers: list[tuple[int, int]]) -> bool:
    counts = _coverage_counts(inner_h, inner_w, towers)
    tower_mask = np.zeros((inner_h, inner_w), dtype=bool)
    for r, c in towers:
        tower_mask[r, c] = True
    bg_counts = counts[~tower_mask]
    if bg_counts.size == 0:
        return False

    tower_count = len(towers)
    if not np.any(bg_counts == 0):
        return False
    for layer in range(1, tower_count + 1):
        if not np.any(bg_counts == layer):
            return False
    return True


def _draw_frame(grid: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    bottom = top + height - 1
    right = left + width - 1
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color


def _gen_attempt() -> np.ndarray | None:
    palette = _sample_palette()
    width = len(palette)
    frame_count = random.choice(FRAME_COUNT_CHOICES)
    tower_counts = [random.randint(1, 3) for _ in range(frame_count)]

    frame_indices = _assign_frame_indices(len(palette), tower_counts)
    if frame_indices is None:
        return None

    boxes = _sample_frame_boxes(width, tower_counts)
    if boxes is None:
        return None

    height = max(box["top"] + box["height"] for box in boxes) + random.randint(*BOTTOM_GAP_RANGE)
    if height > MAX_GRID_HEIGHT:
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    grid[0, :] = np.array(palette, dtype=int)

    for box, frame_idx in zip(boxes, frame_indices):
        frame_color = palette[frame_idx]
        top = box["top"]
        left = box["left"]
        outer_h = box["height"]
        outer_w = box["width"]
        inner_h = outer_h - 2
        inner_w = outer_w - 2

        towers = _place_towers(inner_h, inner_w, box["tower_count"])
        if towers is None or not _tower_pattern_is_informative(inner_h, inner_w, towers):
            return None

        _draw_frame(grid, top, left, outer_h, outer_w, frame_color)
        for tr, tc in towers:
            grid[top + 1 + tr, left + 1 + tc] = RED

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
