import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
SIDE_ORDER = ("top", "right", "bottom", "left")
SIDE_TO_INDEX = {side: idx for idx, side in enumerate(SIDE_ORDER)}

LIBRARY_MASKS = {
    1: (
        np.array(
            [
                [1, 1, 1],
                [0, 1, 0],
                [0, 1, 0],
            ],
            dtype=bool,
        ),
    ),
    2: (
        np.array(
            [
                [1, 1, 0],
                [1, 0, 0],
                [1, 1, 1],
            ],
            dtype=bool,
        ),
    ),
    3: (
        np.array(
            [
                [1, 1, 1],
                [1, 0, 1],
                [0, 0, 1],
            ],
            dtype=bool,
        ),
    ),
}

GEN_RETRIES = 300
RECT_COUNT_OPTIONS = (3, 4)
TOP_HEIGHT_RANGE = (11, 18)
GRID_WIDTH_RANGE = (18, 30)
RECT_SIZE_RANGE = (5, 8)
FRAME_MARGIN = 1
FRAME_PAD = 1
PLACEMENT_TRIES = 120
LIBRARY_HEIGHT = 3


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), max(rows), min(cols), max(cols)


def _find_separator_row(grid: np.ndarray) -> int:
    h, _ = grid.shape
    candidates = [
        r
        for r in range(h)
        if grid[r, 0] != BLACK and np.all(grid[r] == grid[r, 0])
    ]
    for r in candidates:
        if r > 0 and r < h - 1 and np.any(grid[:r] != BLACK) and np.any(grid[r + 1 :] != BLACK):
            return int(r)
    raise AssertionError("expected one full-width non-black separator row")


def _rotate_cw(mask: np.ndarray, quarter_turns: int) -> np.ndarray:
    return np.rot90(mask, -quarter_turns)


def _scale_nearest(mask: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    src_h, src_w = mask.shape
    row_idx = (np.arange(out_h) * src_h) // out_h
    col_idx = (np.arange(out_w) * src_w) // out_w
    return mask[row_idx[:, None], col_idx[None, :]]


def _parse_library(bottom: np.ndarray) -> dict[int, np.ndarray]:
    patterns: dict[int, np.ndarray] = {}
    for color in sorted(int(c) for c in np.unique(bottom) if c != BLACK):
        for comp in _components(bottom == color):
            r0, r1, c0, c1 = _bbox(comp)
            block = bottom[r0 : r1 + 1, c0 : c1 + 1]
            if block.shape != (3, 4):
                continue
            count = int(np.count_nonzero(block[:, 0] == color))
            if count not in (1, 2, 3):
                continue
            if np.any((block[:, 0] != BLACK) & (block[:, 0] != color)):
                continue
            pattern = block[:, 1:] != BLACK
            if pattern.shape != (3, 3) or not np.any(pattern):
                continue
            patterns[count] = pattern
    assert set(patterns) == {1, 2, 3}, "library must contain entries for counts 1, 2, and 3"
    return patterns


def _gap_positions(r0: int, r1: int, c0: int, c1: int) -> dict[str, tuple[int, int]]:
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    return {
        "top": (r0, mid_c),
        "right": (mid_r, c1),
        "bottom": (r1, mid_c),
        "left": (mid_r, c0),
    }


def _frame_info(top: np.ndarray, color: int, comp: list[tuple[int, int]]) -> tuple[int, int, int, int, list[str]] | None:
    points = set(comp)
    r0, r1, c0, c1 = _bbox(comp)
    if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
        return None

    border = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    gaps = [side for side, pos in _gap_positions(r0, r1, c0, c1).items() if pos not in points]
    if not 1 <= len(gaps) <= 3:
        return None

    expected = border - {_gap_positions(r0, r1, c0, c1)[side] for side in gaps}
    if points != expected:
        return None

    interior = top[r0 + 1 : r1, c0 + 1 : c1]
    if np.any(interior != BLACK):
        return None

    if np.any(top[r0 : r1 + 1, c0 : c1 + 1][top[r0 : r1 + 1, c0 : c1 + 1] == color] != color):
        return None
    return r0, r1, c0, c1, gaps


def _rotation_from_gaps(gaps: list[str]) -> int:
    idxs = {SIDE_TO_INDEX[side] for side in gaps}
    if len(idxs) == 1:
        return next(iter(idxs))
    if len(idxs) == 2:
        if idxs == {0, 2}:
            return 0
        if idxs == {1, 3}:
            return 1
        for rot in range(4):
            if idxs == {rot % 4, (rot + 1) % 4}:
                return rot
    if len(idxs) == 3:
        missing = next(iter({0, 1, 2, 3} - idxs))
        return (missing - 2) % 4
    raise AssertionError("unsupported gap configuration")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow frame with the library pattern indexed by its gap count and aligned to the gaps.

    Input structure:
    - Black background. Above a horizontal grey separator sit several medium-sized colored hollow rectangles.
    - Each rectangle border has 1 to 3 one-cell gaps located at side midpoints.
    - Below the separator, a pattern library shows three colored 3x3 motifs; each motif has 1 to 3 same-color marker dots stacked just outside its top-left corner.

    Rule:
    - Count the border gaps on each hollow rectangle.
    - Find the library motif whose marker-dot count matches that gap count.
    - Use the rectangle's gap-side arrangement to choose the motif rotation, then scale the rotated 3x3 mask to the rectangle interior with nearest-neighbor sampling.
    - Recolor the scaled motif with the rectangle border color, draw it into the interior, and discard the library area.
    - Return only the rows above the separator.

    Key insight:
    - The marker-dot stack is a lookup key for how many gaps a frame has, while the gap locations determine how the selected motif must be oriented before it is enlarged and recolored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_row = _find_separator_row(g)
    top = g[:sep_row].copy()
    library = _parse_library(g[sep_row + 1 :])

    for color in sorted(int(c) for c in np.unique(top) if c != BLACK):
        for comp in _components(top == color):
            info = _frame_info(top, color, comp)
            if info is None:
                continue
            r0, r1, c0, c1, gaps = info
            pattern = library[len(gaps)]
            rotated = _rotate_cw(pattern, _rotation_from_gaps(gaps))
            scaled = _scale_nearest(rotated, r1 - r0 - 1, c1 - c0 - 1)
            interior = top[r0 + 1 : r1, c0 + 1 : c1]
            interior[scaled] = color

    return top


def _sample_gap_sides(rng: random.Random, count: int) -> list[str]:
    if count == 1:
        return [rng.choice(SIDE_ORDER)]
    if count == 2:
        if rng.random() < 0.5:
            return ["top", "bottom"] if rng.random() < 0.5 else ["left", "right"]
        start = rng.randrange(4)
        return [SIDE_ORDER[start], SIDE_ORDER[(start + 1) % 4]]
    if count == 3:
        missing = rng.choice(SIDE_ORDER)
        return [side for side in SIDE_ORDER if side != missing]
    raise AssertionError("gap count must be between 1 and 3")


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int, gaps: list[str]) -> None:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    for side, (gr, gc) in _gap_positions(r0, r1, c0, c1).items():
        if side in gaps:
            grid[gr, gc] = BLACK


def _fits(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    top = max(0, r0 - FRAME_PAD)
    left = max(0, c0 - FRAME_PAD)
    bottom = min(occupied.shape[0], r0 + h + FRAME_PAD)
    right = min(occupied.shape[1], c0 + w + FRAME_PAD)
    return not np.any(occupied[top:bottom, left:right])


def _occupy(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    top = max(0, r0 - FRAME_PAD)
    left = max(0, c0 - FRAME_PAD)
    bottom = min(occupied.shape[0], r0 + h + FRAME_PAD)
    right = min(occupied.shape[1], c0 + w + FRAME_PAD)
    occupied[top:bottom, left:right] = True


def _place_frames(rng: random.Random, top_h: int, width: int, counts: list[int], colors: list[int]) -> np.ndarray | None:
    grid = np.zeros((top_h, width), dtype=int)
    occupied = np.zeros_like(grid, dtype=bool)
    for count, color in zip(counts, colors):
        placed = False
        for _ in range(PLACEMENT_TRIES):
            h = rng.randint(*RECT_SIZE_RANGE)
            w = rng.randint(*RECT_SIZE_RANGE)
            max_r0 = top_h - h - FRAME_MARGIN
            max_c0 = width - w - FRAME_MARGIN
            if max_r0 < FRAME_MARGIN or max_c0 < FRAME_MARGIN:
                continue
            r0 = rng.randint(FRAME_MARGIN, max_r0)
            c0 = rng.randint(FRAME_MARGIN, max_c0)
            if not _fits(occupied, r0, c0, h, w):
                continue
            _draw_frame(grid, r0, c0, h, w, color, _sample_gap_sides(rng, count))
            _occupy(occupied, r0, c0, h, w)
            placed = True
            break
        if not placed:
            return None
    return grid


def _draw_library(grid: np.ndarray, top_h: int, width: int, colors_by_count: dict[int, int], masks_by_count: dict[int, np.ndarray], rng: random.Random) -> None:
    order = [1, 2, 3]
    rng.shuffle(order)
    total_w = 3 * 4 + 2
    start_c = rng.randint(0, width - total_w)
    base_r = top_h + 1
    for idx, count in enumerate(order):
        c0 = start_c + idx * 5
        color = colors_by_count[count]
        block = np.zeros((3, 4), dtype=int)
        block[:count, 0] = color
        block[:, 1:4][masks_by_count[count]] = color
        grid[base_r : base_r + 3, c0 : c0 + 4] = block


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    top_h = rng.randint(*TOP_HEIGHT_RANGE)
    width = rng.randint(*GRID_WIDTH_RANGE)
    total_h = top_h + 1 + LIBRARY_HEIGHT
    if total_h > 30:
        return None

    num_rects = rng.choice(RECT_COUNT_OPTIONS)
    counts = [1, 2, 3]
    if num_rects == 4:
        counts.append(rng.randint(1, 3))
    rng.shuffle(counts)

    palette = [c for c in range(1, 10) if c != GREY]
    rng.shuffle(palette)
    rect_colors = palette[:num_rects]
    lib_colors = palette[num_rects : num_rects + 3]
    if len(lib_colors) < 3:
        return None

    top = _place_frames(rng, top_h, width, counts, rect_colors)
    if top is None:
        return None

    grid = np.zeros((total_h, width), dtype=int)
    grid[:top_h] = top
    grid[top_h, :] = GREY

    masks_by_count = {count: LIBRARY_MASKS[count][0] for count in (1, 2, 3)}
    colors_by_count = {count: color for count, color in zip((1, 2, 3), lib_colors)}
    _draw_library(grid, top_h, width, colors_by_count, masks_by_count, rng)

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != (top_h, width):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
