from __future__ import annotations

import random

import numpy as np


BG = 0
BARRIER = 2
GUIDE = 5
MARKER_COLORS = [1, 3, 4, 6, 7, 8, 9]

SIZE_MIN = 12
SIZE_MAX = 18
PAIR_MIN = 3
PAIR_MAX = 5
UNPAIRED_MIN = 2
UNPAIRED_MAX = 4
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _barrier_info(grid: np.ndarray) -> tuple[str, int, int, int]:
    coords = np.argwhere(grid == BARRIER)
    assert coords.size, "expected a red barrier"

    rows = sorted(set(map(int, coords[:, 0])))
    cols = sorted(set(map(int, coords[:, 1])))
    if len(cols) == 1:
        c = cols[0]
        r0, r1 = rows[0], rows[-1]
        assert rows == list(range(r0, r1 + 1)), "vertical barrier must be contiguous"
        return "vertical", c, r0, r1
    if len(rows) == 1:
        r = rows[0]
        c0, c1 = cols[0], cols[-1]
        assert cols == list(range(c0, c1 + 1)), "horizontal barrier must be contiguous"
        return "horizontal", r, c0, c1
    raise AssertionError("barrier must be a straight 1-pixel segment")


def _gravity_from_marker(grid: np.ndarray) -> tuple[str, tuple[int, int]]:
    coords = np.argwhere(grid == GUIDE)
    assert coords.shape[0] == 1, "expected one grey edge marker"
    h, w = grid.shape
    r, c = map(int, coords[0])
    sides: list[str] = []
    if r == 0:
        sides.append("top")
    if r == h - 1:
        sides.append("bottom")
    if c == 0:
        sides.append("left")
    if c == w - 1:
        sides.append("right")
    assert len(sides) == 1, "grey marker must lie on exactly one edge"
    side = sides[0]
    gravity = {
        "top": "down",
        "bottom": "up",
        "left": "right",
        "right": "left",
    }[side]
    step = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }[gravity]
    return gravity, step


def _marker_mask(grid: np.ndarray, bg: int) -> np.ndarray:
    return (grid != bg) & (grid != BARRIER) & (grid != GUIDE)


def _pairings(
    grid: np.ndarray,
    orientation: str,
    axis: int,
    span0: int,
    span1: int,
    bg: int,
) -> tuple[list[dict], list[tuple[int, int]], list[tuple[int, int]]]:
    marker_mask = _marker_mask(grid, bg)
    pairs: list[dict] = []
    crossing_markers: set[tuple[int, int]] = set()
    pair_sources: set[tuple[int, int]] = set()

    if orientation == "vertical":
        barrier_col = axis
        for r in range(span0, span1 + 1):
            left = [(r, c) for c in range(barrier_col) if marker_mask[r, c]]
            right = [(r, c) for c in range(barrier_col + 1, grid.shape[1]) if marker_mask[r, c]]
            crossing_markers.update(left)
            crossing_markers.update(right)
            if len(left) == 1 and len(right) == 1:
                left_src = left[0]
                right_src = right[0]
                pair_sources.add(left_src)
                pair_sources.add(right_src)
                pairs.append(
                    {
                        "line": r,
                        "src_a": left_src,
                        "src_b": right_src,
                        "dst_a": (r, barrier_col + 1),
                        "dst_b": (r, barrier_col - 1),
                        "color_a": int(grid[left_src]),
                        "color_b": int(grid[right_src]),
                    }
                )
    else:
        barrier_row = axis
        for c in range(span0, span1 + 1):
            top = [(r, c) for r in range(barrier_row) if marker_mask[r, c]]
            bottom = [(r, c) for r in range(barrier_row + 1, grid.shape[0]) if marker_mask[r, c]]
            crossing_markers.update(top)
            crossing_markers.update(bottom)
            if len(top) == 1 and len(bottom) == 1:
                top_src = top[0]
                bottom_src = bottom[0]
                pair_sources.add(top_src)
                pair_sources.add(bottom_src)
                pairs.append(
                    {
                        "line": c,
                        "src_a": top_src,
                        "src_b": bottom_src,
                        "dst_a": (barrier_row + 1, c),
                        "dst_b": (barrier_row - 1, c),
                        "color_a": int(grid[top_src]),
                        "color_b": int(grid[bottom_src]),
                    }
                )

    all_markers = set(map(tuple, np.argwhere(marker_mask)))
    crossing_unpaired = sorted(crossing_markers - pair_sources)
    untouched = sorted(all_markers - pair_sources)
    return pairs, crossing_unpaired, untouched


def _apply_swaps(grid: np.ndarray, pairs: list[dict], bg: int) -> tuple[np.ndarray, np.ndarray]:
    out = grid.copy()
    activated = np.zeros_like(grid, dtype=bool)

    for pair in pairs:
        ra, ca = pair["src_a"]
        rb, cb = pair["src_b"]
        out[ra, ca] = bg
        out[rb, cb] = bg

    for pair in pairs:
        da_r, da_c = pair["dst_a"]
        db_r, db_c = pair["dst_b"]
        assert int(out[da_r, da_c]) == bg and int(out[db_r, db_c]) == bg, "swap destinations must be empty"
        out[da_r, da_c] = pair["color_a"]
        out[db_r, db_c] = pair["color_b"]
        activated[da_r, da_c] = True
        activated[db_r, db_c] = True

    return out, activated


def _compact_swapped(grid: np.ndarray, activated: np.ndarray, gravity: str, bg: int) -> np.ndarray:
    out = grid.copy()
    fixed = (grid != bg) & (~activated)
    h, w = grid.shape

    if gravity in ("left", "right"):
        for r in range(h):
            c = 0
            while c < w:
                if fixed[r, c]:
                    c += 1
                    continue
                start = c
                while c < w and not fixed[r, c]:
                    c += 1
                end = c
                values = [int(grid[r, cc]) for cc in range(start, end) if activated[r, cc]]
                out[r, start:end] = bg
                if gravity == "left":
                    for i, value in enumerate(values):
                        out[r, start + i] = value
                else:
                    for i, value in enumerate(values):
                        out[r, end - len(values) + i] = value
    else:
        for c in range(w):
            r = 0
            while r < h:
                if fixed[r, c]:
                    r += 1
                    continue
                start = r
                while r < h and not fixed[r, c]:
                    r += 1
                end = r
                values = [int(grid[rr, c]) for rr in range(start, end) if activated[rr, c]]
                out[start:end, c] = bg
                if gravity == "up":
                    for i, value in enumerate(values):
                        out[start + i, c] = value
                else:
                    for i, value in enumerate(values):
                        out[end - len(values) + i, c] = value

    return out


def _solve_with_trace(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    orientation, axis, span0, span1 = _barrier_info(g)
    gravity, _step = _gravity_from_marker(g)
    if orientation == "vertical":
        assert gravity in ("up", "down"), "vertical barrier requires vertical gravity"
    else:
        assert gravity in ("left", "right"), "horizontal barrier requires horizontal gravity"

    pairs, crossing_unpaired, untouched = _pairings(g, orientation, axis, span0, span1, bg)
    assert pairs, "expected at least one cross-barrier pair"

    swapped, activated = _apply_swaps(g, pairs, bg)
    out = _compact_swapped(swapped, activated, gravity, bg)

    before = [(int(r), int(c), int(swapped[r, c])) for r, c in np.argwhere(activated)]
    after = [(int(r), int(c), int(out[r, c])) for r, c in np.argwhere((out != bg) & (swapped != out) | ((out != bg) & (activated == False) & (swapped == out)))]
    # Track swapped pixels after gravity by color/line count rather than by identity.
    after_swapped = [(int(r), int(c), int(out[r, c])) for r, c in np.argwhere(out != bg) if int(out[r, c]) not in (BARRIER, GUIDE)]
    trace = {
        "bg": bg,
        "orientation": orientation,
        "axis": axis,
        "span": (span0, span1),
        "gravity": gravity,
        "pairs": pairs,
        "crossing_unpaired": crossing_unpaired,
        "untouched": untouched,
        "swapped": swapped,
        "activated": activated,
        "output": out,
        "activated_before": before,
        "all_marker_mask_after_swap": _marker_mask(swapped, bg),
        "swapped_marker_count": int(activated.sum()),
        "marker_positions_after": after_swapped,
    }
    return trace


def main(grid: np.ndarray) -> np.ndarray:
    """Swap paired markers across a straight barrier, then let only the swapped markers fall inward from the grey edge.

    Input structure:
    - The background is black, with one straight red barrier segment that is either horizontal or vertical and does not touch the grid border.
    - Colored singleton markers lie on both sides of the barrier; some crossing rows or columns contain exactly one marker on each side, while others contain markers on only one side.
    - One grey singleton lies on a grid edge and indicates the inward gravity direction.

    Rule:
    - For each row crossing a vertical barrier, or each column crossing a horizontal barrier, check the two sides of the barrier segment.
    - If that crossing line has exactly one colored marker on each side, erase both sources and swap them to the cells immediately adjacent to the opposite side of the barrier.
    - Keep the red barrier, the grey edge marker, and every unpaired colored marker unchanged.
    - Apply gravity only to the newly swapped markers, sliding them inward along the grey-indicated direction until blocked by the border or an already fixed pixel.

    Key insight:
    - The swap step creates the only pixels that are "activated" for gravity, so pairing history, not color alone, determines which markers are allowed to move.
    """
    return _solve_with_trace(grid)["output"]


def _random_palette() -> list[int]:
    count = random.randint(3, 5)
    return random.sample(MARKER_COLORS, count)


def _choose_barrier(size: int) -> tuple[int, int]:
    lo = random.randint(1, 2)
    hi = random.randint(size - 3, size - 2)
    assert hi - lo + 1 >= 8
    return lo, hi


def _pick_lines(span0: int, span1: int) -> tuple[list[int], list[int]]:
    lines = list(range(span0, span1 + 1))
    pair_count = random.randint(PAIR_MIN, min(PAIR_MAX, len(lines) - UNPAIRED_MIN))
    unpaired_count = random.randint(UNPAIRED_MIN, min(UNPAIRED_MAX, len(lines) - pair_count))
    chosen = random.sample(lines, pair_count + unpaired_count)
    random.shuffle(chosen)
    return chosen[:pair_count], chosen[pair_count:]


def _color_seq(count: int, palette: list[int]) -> list[int]:
    out = [random.choice(palette) for _ in range(count)]
    if count >= 2 and len(set(out)) == 1 and len(palette) > 1:
        out[0] = palette[0]
        out[-1] = palette[1]
    return out


def _build_vertical_case(h: int, w: int) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    barrier_col = random.randint(4, w - 5)
    r0, r1 = _choose_barrier(h)
    grid[r0 : r1 + 1, barrier_col] = BARRIER

    marker_col = random.randint(1, w - 2)
    marker_edge = random.choice(["top", "bottom"])
    if marker_edge == "top":
        grid[0, marker_col] = GUIDE
    else:
        grid[h - 1, marker_col] = GUIDE

    pair_rows, unpaired_rows = _pick_lines(r0, r1)
    palette = _random_palette()
    left_colors = _color_seq(len(pair_rows), palette)
    right_colors = _color_seq(len(pair_rows), palette)

    for row, left_color, right_color in zip(pair_rows, left_colors, right_colors):
        left_col = random.randint(1, barrier_col - 2)
        right_col = random.randint(barrier_col + 2, w - 2)
        grid[row, left_col] = left_color
        grid[row, right_col] = right_color

    for row in unpaired_rows:
        side = random.choice(["left", "right"])
        if side == "left":
            col = random.randint(1, barrier_col - 2)
        else:
            col = random.randint(barrier_col + 2, w - 2)
        grid[row, col] = random.choice(palette)

    extras = random.randint(0, 2)
    free_rows = [r for r in range(h) if r < r0 or r > r1]
    for _ in range(extras):
        if not free_rows:
            break
        row = random.choice(free_rows)
        side = random.choice(["left", "right"])
        cols = range(1, barrier_col - 1) if side == "left" else range(barrier_col + 2, w - 1)
        cells = [(row, c) for c in cols if grid[row, c] == BG]
        if cells:
            rr, cc = random.choice(cells)
            grid[rr, cc] = random.choice(palette)

    return grid


def _build_horizontal_case(h: int, w: int) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    barrier_row = random.randint(4, h - 5)
    c0, c1 = _choose_barrier(w)
    grid[barrier_row, c0 : c1 + 1] = BARRIER

    marker_row = random.randint(1, h - 2)
    marker_edge = random.choice(["left", "right"])
    if marker_edge == "left":
        grid[marker_row, 0] = GUIDE
    else:
        grid[marker_row, w - 1] = GUIDE

    pair_cols, unpaired_cols = _pick_lines(c0, c1)
    palette = _random_palette()
    top_colors = _color_seq(len(pair_cols), palette)
    bottom_colors = _color_seq(len(pair_cols), palette)

    for col, top_color, bottom_color in zip(pair_cols, top_colors, bottom_colors):
        top_row = random.randint(1, barrier_row - 2)
        bottom_row = random.randint(barrier_row + 2, h - 2)
        grid[top_row, col] = top_color
        grid[bottom_row, col] = bottom_color

    for col in unpaired_cols:
        side = random.choice(["top", "bottom"])
        if side == "top":
            row = random.randint(1, barrier_row - 2)
        else:
            row = random.randint(barrier_row + 2, h - 2)
        grid[row, col] = random.choice(palette)

    extras = random.randint(0, 2)
    free_cols = [c for c in range(w) if c < c0 or c > c1]
    for _ in range(extras):
        if not free_cols:
            break
        col = random.choice(free_cols)
        side = random.choice(["top", "bottom"])
        rows = range(1, barrier_row - 1) if side == "top" else range(barrier_row + 2, h - 1)
        cells = [(r, col) for r in rows if grid[r, col] == BG]
        if cells:
            rr, cc = random.choice(cells)
            grid[rr, cc] = random.choice(palette)

    return grid


def _all_marker_gravity(swapped: np.ndarray, bg: int, gravity: str) -> np.ndarray:
    movable = _marker_mask(swapped, bg)
    return _compact_swapped(swapped, movable, gravity, bg)


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        trace = _solve_with_trace(grid)
    except Exception:
        return False

    out = trace["output"]
    swapped = trace["swapped"]
    gravity = trace["gravity"]
    bg = trace["bg"]

    if len(trace["pairs"]) < 3:
        return False
    if len(trace["crossing_unpaired"]) < 2:
        return False
    if np.array_equal(out, grid):
        return False
    if np.array_equal(out, swapped):
        return False

    moved_swapped = 0
    for r, c in np.argwhere(trace["activated"]):
        if int(out[r, c]) == bg:
            moved_swapped += 1
    if moved_swapped == 0:
        return False

    naive_all = _all_marker_gravity(swapped, bg, gravity)
    if np.array_equal(naive_all, out):
        return False

    for r, c in trace["untouched"]:
        if int(grid[r, c]) != int(out[r, c]):
            return False

    if not np.array_equal(main(grid), out):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        h = random.randint(SIZE_MIN, SIZE_MAX)
        w = random.randint(SIZE_MIN, SIZE_MAX)
        grid = _build_vertical_case(h, w) if random.random() < 0.5 else _build_horizontal_case(h, w)
        if _semantic_ok(grid):
            return grid
    raise RuntimeError("failed to generate a valid selective-gravity swap puzzle")
