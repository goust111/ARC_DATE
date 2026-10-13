from __future__ import annotations

from collections import deque

import numpy as np

from arc_utils import Color, components_mask, get_bg, neighbors4
import math
import random


def _exterior_background(g: np.ndarray, bg: int) -> np.ndarray:
    h, w = g.shape
    exterior = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    def add(r: int, c: int) -> None:
        if int(g[r, c]) == int(bg) and not exterior[r, c]:
            exterior[r, c] = True
            queue.append((r, c))

    for r in range(h):
        add(r, 0)
        add(r, w - 1)
    for c in range(w):
        add(0, c)
        add(h - 1, c)

    while queue:
        r, c = queue.popleft()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w:
                add(rr, cc)

    return exterior


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill every fully enclosed background region inside red outlines with blue.

    Input structure
    - The grid has a black background and red one-cell-thick outline/barrier pixels.
    - Red outlines may touch each other or the grid boundary, and some red shapes
      may be incomplete.
    - Blue is absent from the input and appears only as the fill color.

    Rule
    - Treat red pixels as walls and trace all background pixels connected to the
      grid boundary by 4-neighbor steps.
    - Recolor each remaining background pixel to blue.
    - Leave red pixels and exterior background unchanged.

    Key insight
    - The blue cells are exactly the background holes: black regions that cannot
      escape to the outer boundary without crossing a red wall.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    assert bg != Color.RED, "red walls must not be the background"
    assert not np.any(g == Color.BLUE), "blue must be absent from the input"
    foreground = set(map(int, np.unique(g))) - {int(bg)}
    assert foreground == {Color.RED}, "expected only red foreground walls"

    exterior = _exterior_background(g, bg)
    enclosed = (g == bg) & ~exterior
    assert np.any(enclosed), "expected at least one enclosed background region"

    out = g.copy()
    out[enclosed] = Color.BLUE
    return out

# --------------- generator (o5x) ---------------

# Generator colors.
GEN_BLACK, GEN_RED, GEN_BLUE = int(Color.BLACK), int(Color.RED), int(Color.BLUE)
# Canvas and frame counts.
GRID_MIN, GRID_MAX = 8, 12
FRAME_COUNTS_BY_GRID = {8: [1, 2], 9: [1, 2], 10: [2, 3], 11: [2, 3], 12: [3]}
# Frame construction and sharing.
INITIAL_FRAME = (0, 0, 5, 5)
FRAME_SIZES = [(5, 5), (5, 5), (4, 5), (5, 4)]
FRAME_SIDES = ("top", "bottom", "left", "right")
FRAME_BUILD_ATTEMPTS, FRAME_LINK_ATTEMPTS = 60, 30
SHARED_WALL_INSET = 2
# Red density and interior dot placement.
MIN_RED_FRACTION, MAX_RED_FRACTION = 0.27, 0.41
MAX_DOTS_PER_GRID, DOT_FRAME_SIZE, DOT_CENTER_OFFSET = 3, 5, 2
# Distractor construction.
DISTRACTOR_PROBABILITY = 0.45
DISTRACTOR_MIN_BORDER_GAP, DISTRACTOR_MIN_SPAN_DELTA = 1, 2
DISTRACTOR_REDS_PER_EXTENSION = 2
# Generation retry budget.
GENERATION_ATTEMPTS = 300
Frame = tuple[int, int, int, int]
DistractorOption = tuple[int, tuple[int, int, int, int, str]]


def _gen_draw_frame(g: np.ndarray, frame: Frame) -> None:
    r, c, h, w = frame
    g[r, c : c + w] = GEN_RED
    g[r + h - 1, c : c + w] = GEN_RED
    g[r : r + h, c] = GEN_RED
    g[r : r + h, c + w - 1] = GEN_RED


def _gen_shared_neighbor(
    frame: Frame, side: str, size: tuple[int, int]
) -> Frame:
    r, c, h, w = frame
    nh, nw = size
    if side == "top":
        nr = r - nh + 1
        lo, hi = c - nw + SHARED_WALL_INSET, c + w - SHARED_WALL_INSET
        nc = random.randint(lo, hi)
    elif side == "bottom":
        nr = r + h - 1
        lo, hi = c - nw + SHARED_WALL_INSET, c + w - SHARED_WALL_INSET
        nc = random.randint(lo, hi)
    elif side == "left":
        nc = c - nw + 1
        lo, hi = r - nh + SHARED_WALL_INSET, r + h - SHARED_WALL_INSET
        nr = random.randint(lo, hi)
    else:
        nc = c + w - 1
        lo, hi = r - nh + SHARED_WALL_INSET, r + h - SHARED_WALL_INSET
        nr = random.randint(lo, hi)
    return nr, nc, nh, nw


def _gen_build_frames(k: int, n: int) -> list[Frame] | None:
    for _ in range(FRAME_BUILD_ATTEMPTS):
        frames = [INITIAL_FRAME]
        for _ in range(k - 1):
            added = False
            for _ in range(FRAME_LINK_ATTEMPTS):
                base = random.choice(frames)
                side = random.choice(FRAME_SIDES)
                new = _gen_shared_neighbor(base, side, random.choice(FRAME_SIZES))
                if new not in frames:
                    frames.append(new)
                    added = True
                    break
            if not added:
                break
        if len(frames) != k:
            continue

        min_r = min(r for r, _, _, _ in frames)
        min_c = min(c for _, c, _, _ in frames)
        max_r = max(r + h - 1 for r, _, h, _ in frames)
        max_c = max(c + w - 1 for _, c, _, w in frames)
        if max_r - min_r + 1 > n or max_c - min_c + 1 > n:
            continue

        dr = random.randint(-min_r, n - 1 - max_r)
        dc = random.randint(-min_c, n - 1 - max_c)
        return [(r + dr, c + dc, h, w) for r, c, h, w in frames]
    return None


def _gen_has_red_2x2(g: np.ndarray) -> bool:
    red = g == GEN_RED
    return bool(np.any(red[:-1, :-1] & red[1:, :-1] & red[:-1, 1:] & red[1:, 1:]))


def _gen_red_limits(n: int) -> tuple[int, int]:
    return math.ceil(MIN_RED_FRACTION * n * n), math.floor(MAX_RED_FRACTION * n * n)


def _gen_red_count(g: np.ndarray) -> int:
    return int(np.count_nonzero(g == GEN_RED))


def _gen_eligible_dot_centers(
    g: np.ndarray, frames: list[Frame]
) -> list[tuple[int, int]]:
    centers: list[tuple[int, int]] = []
    for r, c, h, w in frames:
        if h != DOT_FRAME_SIZE or w != DOT_FRAME_SIZE:
            continue
        rr, cc = r + DOT_CENTER_OFFSET, c + DOT_CENTER_OFFSET
        if int(g[rr, cc]) != GEN_BLACK:
            continue
        neighbors = (g[rr - 1, cc], g[rr + 1, cc], g[rr, cc - 1], g[rr, cc + 1])
        if any(int(v) == GEN_RED for v in neighbors):
            continue
        centers.append((rr, cc))
    return centers


def _gen_add_interior_dots(g: np.ndarray, frames: list[Frame], max_red: int) -> bool:
    max_dots = min(MAX_DOTS_PER_GRID, max_red - _gen_red_count(g))
    dot_centers = _gen_eligible_dot_centers(g, frames)
    if not dot_centers or max_dots < 1:
        return False

    dot_count = random.randint(1, min(max_dots, len(dot_centers)))
    for rr, cc in random.sample(dot_centers, dot_count):
        g[rr, cc] = GEN_RED
    return True


def _gen_span_pairs(start: int, end: int):
    for a in range(start, end - 1):
        for b in range(a + DISTRACTOR_MIN_SPAN_DELTA, end + 1):
            yield a, b


def _gen_distractor_options(g: np.ndarray, frames: list[Frame], n: int) -> list[DistractorOption]:
    opts: list[DistractorOption] = []
    for r, c, h, w in frames:
        r1, c1 = r + h - 1, c + w - 1

        if r >= DISTRACTOR_MIN_BORDER_GAP:
            for a, b in _gen_span_pairs(c, c1):
                if np.all(g[:r, a : b + 1] == GEN_BLACK):
                    opts.append((DISTRACTOR_REDS_PER_EXTENSION * r, (r, a, r, b, "top")))

        if n - 1 - r1 >= DISTRACTOR_MIN_BORDER_GAP:
            for a, b in _gen_span_pairs(c, c1):
                if np.all(g[r1 + 1 :, a : b + 1] == GEN_BLACK):
                    extra = DISTRACTOR_REDS_PER_EXTENSION * (n - 1 - r1)
                    opts.append((extra, (r1, a, r1, b, "bottom")))

        if c >= DISTRACTOR_MIN_BORDER_GAP:
            for a, b in _gen_span_pairs(r, r1):
                if np.all(g[a : b + 1, :c] == GEN_BLACK):
                    opts.append((DISTRACTOR_REDS_PER_EXTENSION * c, (a, c, b, c, "left")))

        if n - 1 - c1 >= DISTRACTOR_MIN_BORDER_GAP:
            for a, b in _gen_span_pairs(r, r1):
                if np.all(g[a : b + 1, c1 + 1 :] == GEN_BLACK):
                    extra = DISTRACTOR_REDS_PER_EXTENSION * (n - 1 - c1)
                    opts.append((extra, (a, c1, b, c1, "right")))
    random.shuffle(opts)
    return opts


def _gen_add_distractor(g: np.ndarray, frames: list[Frame], n: int, need_extra: int, max_extra: int) -> bool:
    for extra, opt in _gen_distractor_options(g, frames, n):
        if extra < need_extra or extra > max_extra:
            continue
        a0, b0, a1, b1, side = opt
        if side == "top":
            g[:a0, b0] = GEN_RED
            g[:a1, b1] = GEN_RED
        elif side == "bottom":
            g[a0 + 1 :, b0] = GEN_RED
            g[a1 + 1 :, b1] = GEN_RED
        elif side == "left":
            g[a0, :b0] = GEN_RED
            g[a1, :b1] = GEN_RED
        else:
            g[a0, b0 + 1 :] = GEN_RED
            g[a1, b1 + 1 :] = GEN_RED
        return True
    return False


def _gen_choose_frame_layout() -> tuple[int, int, list[Frame]] | None:
    n = random.randint(GRID_MIN, GRID_MAX)
    k = random.choice(FRAME_COUNTS_BY_GRID[n])
    frames = _gen_build_frames(k, n)
    if frames is None:
        return None
    return n, k, frames


def _gen_render_frames(n: int, frames: list[Frame]) -> np.ndarray:
    g = np.zeros((n, n), dtype=int)
    for frame in frames:
        _gen_draw_frame(g, frame)
    return g


def _gen_adjust_red_density(
    g: np.ndarray, frames: list[Frame], n: int, min_red: int, max_red: int
) -> bool:
    """Adds dots and optional distractor red cells while staying within density limits."""
    if not _gen_add_interior_dots(g, frames, max_red):
        return False

    red_count = _gen_red_count(g)
    if red_count > max_red:
        return False

    need_extra = max(0, min_red - red_count)
    max_extra = max_red - red_count
    want_distractor = need_extra > 0 or (
        max_extra >= DISTRACTOR_MIN_SPAN_DELTA and random.random() < DISTRACTOR_PROBABILITY
    )
    if want_distractor:
        added = _gen_add_distractor(g, frames, n, need_extra, max_extra)
        if need_extra > 0 and not added:
            return False

    red_count = _gen_red_count(g)
    return min_red <= red_count <= max_red


def _gen_oracle_input(g: np.ndarray, expected_components: int) -> np.ndarray | None:
    """Runs the solver oracle and returns the blue-stripped input grid if valid."""
    try:
        out = main(g)
    except AssertionError:
        return None

    blue_mask = out == GEN_BLUE
    blue_components = components_mask(blue_mask, connectivity=4)
    if len(blue_components) != expected_components:
        return None

    inp = out.copy()
    inp[inp == GEN_BLUE] = GEN_BLACK
    try:
        if not np.array_equal(main(inp), out):
            return None
    except AssertionError:
        return None
    return inp


def _gen_attempt() -> np.ndarray | None:
    layout = _gen_choose_frame_layout()
    if layout is None:
        return None
    n, k, frames = layout

    g = _gen_render_frames(n, frames)
    if _gen_has_red_2x2(g):
        return None

    min_red, max_red = _gen_red_limits(n)
    if not _gen_adjust_red_density(g, frames, n, min_red, max_red):
        return None
    if _gen_has_red_2x2(g):
        return None

    return _gen_oracle_input(g, k)


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid a5313dff input")
