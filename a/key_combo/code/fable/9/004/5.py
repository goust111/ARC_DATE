from __future__ import annotations

from collections import Counter, deque

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
RNG = np.random.default_rng()


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(exclude)
    return sorted(colors)


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _is_frame_component(grid: np.ndarray, coords: np.ndarray, color: int, bg: int) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False

    h = r1 - r0 + 1
    w = c1 - c0 + 1
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True

    mask = np.zeros((h, w), dtype=bool)
    shifted = coords - np.array([r0, c0], dtype=int)
    mask[shifted[:, 0], shifted[:, 1]] = True
    if not np.array_equal(mask, expected):
        return False

    return bool(np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == bg) and np.all(grid[coords[:, 0], coords[:, 1]] == color))


def _find_frame(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in _colors_present(grid, exclude={bg, GREY}):
        for coords in _find_components(grid == color):
            if _is_frame_component(grid, coords, color, bg):
                candidates.append((color, _bbox(coords)))
    assert len(candidates) == 1, "expected exactly one hollow frame"
    return candidates[0]


def _split_bar_at(grid: np.ndarray, bg: int, r: int, c: int) -> tuple[int, int, int] | None:
    h, w = grid.shape
    if not (0 < r < h - 1):
        return None

    color = int(grid[r - 1, c])
    if color in (bg, GREY) or int(grid[r + 1, c]) != color:
        return None

    top = r - 1
    while top - 1 >= 0 and int(grid[top - 1, c]) == color:
        top -= 1
    bottom = r + 1
    while bottom + 1 < h and int(grid[bottom + 1, c]) == color:
        bottom += 1

    a = r - top
    b = bottom - r
    if a < 1 or b < 1:
        return None

    expected = np.array([color] * a + [GREY] + [color] * b, dtype=int)
    if not np.array_equal(grid[top : bottom + 1, c], expected):
        return None

    for rr in range(top, bottom + 1):
        for nc in (c - 1, c + 1):
            if 0 <= nc < w and int(grid[rr, nc]) != bg:
                return None

    return color, a, b


def _extract_bars(grid: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    bars: list[tuple[int, int, int]] = []
    for r, c in np.argwhere(grid == GREY):
        bar = _split_bar_at(grid, bg, int(r), int(c))
        assert bar is not None, "every grey pixel must split one vertical bar"
        bars.append(bar)
    assert bars, "expected at least one split bar"
    return bars


def main(grid: np.ndarray) -> np.ndarray:
    """Decode each grey-split vertical bar into a canvas coordinate and plot its color there.

    Input structure:
    - The grid has a black background and exactly one hollow rectangular frame whose interior is fully empty.
    - Elsewhere there are several single-color vertical bars, each containing exactly one grey splitter pixel.
    - In every such bar, the grey pixel separates an upper segment and a lower segment of the same color.

    Rule:
    - Keep the frame, the bars, and the grey splitter pixels unchanged.
    - For a bar of color c, let a be the length of the upper segment and b be the length of the lower segment.
    - Inside the frame interior, counting from the interior's top-left corner with 1-based indexing, color the cell at row a and column b with color c.

    Key insight:
    - The bars are not objects to move; the grey splitter turns each bar into a pair of lengths, and those two lengths encode the destination row and column inside the frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    _frame_color, (r0, c0, r1, c1) = _find_frame(g, bg)
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1

    out = g.copy()
    used: set[tuple[int, int]] = set()
    for color, a, b in _extract_bars(g, bg):
        assert 1 <= a <= inner_h and 1 <= b <= inner_w
        rr = r0 + a
        cc = c0 + b
        assert (rr, cc) not in used, "bar targets must be distinct"
        used.add((rr, cc))
        out[rr, cc] = color
    return out


GEN_ATTEMPTS = 400
MIN_INTERIOR = 3
MAX_INTERIOR = 8
MIN_BARS = 2
MAX_BARS = 6
PLACEMENT_ATTEMPTS = 300


def _paint_frame(grid: np.ndarray, r0: int, c0: int, inner_h: int, inner_w: int, color: int) -> None:
    r1 = r0 + inner_h + 1
    c1 = c0 + inner_w + 1
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _mark_blocked(blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> None:
    h, w = blocked.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _can_place(blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> bool:
    h, w = blocked.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return not bool(blocked[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _paint_bar(grid: np.ndarray, top: int, col: int, a: int, b: int, color: int) -> None:
    grid[top : top + a, col] = color
    grid[top + a, col] = GREY
    grid[top + a + 1 : top + a + 1 + b, col] = color


def _sample_targets(inner_h: int, inner_w: int, n_bars: int) -> list[tuple[int, int]]:
    all_targets = [(a, b) for a in range(1, inner_h + 1) for b in range(1, inner_w + 1)]
    for _ in range(50):
        idx = RNG.choice(len(all_targets), size=n_bars, replace=False)
        targets = [all_targets[int(i)] for i in idx]
        if len({a for a, _ in targets}) >= 2 and len({b for _, b in targets}) >= 2:
            return targets
    raise RuntimeError("failed to sample diverse bar targets")


def _gen_attempt() -> np.ndarray | None:
    inner_h = int(RNG.integers(MIN_INTERIOR, MAX_INTERIOR + 1))
    inner_w = int(RNG.integers(MIN_INTERIOR, MAX_INTERIOR + 1))
    frame_h = inner_h + 2
    frame_w = inner_w + 2

    max_bars = min(MAX_BARS, inner_h * inner_w, len(PALETTE) - 1)
    n_bars = int(RNG.integers(MIN_BARS, max_bars + 1))
    targets = _sample_targets(inner_h, inner_w, n_bars)
    bar_heights = [a + b + 1 for a, b in targets]

    min_h = max(frame_h + 2, max(bar_heights) + 2)
    min_w = max(frame_w + 2, 3 * n_bars + 2)
    if min_h > 30 or min_w > 30:
        return None

    H = int(RNG.integers(min_h, min(30, min_h + 6) + 1))
    W = int(RNG.integers(min_w, min(30, min_w + 8) + 1))
    grid = np.full((H, W), BLACK, dtype=int)
    blocked = np.zeros((H, W), dtype=bool)

    frame_color = int(RNG.choice(PALETTE))
    bar_colors = list(map(int, RNG.choice([c for c in PALETTE if c != frame_color], size=n_bars, replace=False)))

    max_frame_r0 = H - frame_h - 1
    max_frame_c0 = W - frame_w - 1
    if max_frame_r0 < 1 or max_frame_c0 < 1:
        return None
    frame_r0 = int(RNG.integers(1, max_frame_r0 + 1))
    frame_c0 = int(RNG.integers(1, max_frame_c0 + 1))
    _paint_frame(grid, frame_r0, frame_c0, inner_h, inner_w, frame_color)
    _mark_blocked(blocked, frame_r0, frame_c0, frame_r0 + frame_h - 1, frame_c0 + frame_w - 1)

    order = sorted(range(n_bars), key=lambda i: bar_heights[i], reverse=True)
    placements: list[tuple[int, int, int, int]] = []
    for idx in order:
        a, b = targets[idx]
        height = a + b + 1
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = int(RNG.integers(1, H - height))
            col = int(RNG.integers(1, W - 1))
            if not _can_place(blocked, top, col, top + height - 1, col):
                continue
            _paint_bar(grid, top, col, a, b, bar_colors[idx])
            _mark_blocked(blocked, top, col, top + height - 1, col)
            placements.append((idx, top, col, height))
            placed = True
            break
        if not placed:
            return None

    expected = grid.copy()
    for (a, b), color in zip(targets, bar_colors):
        expected[frame_r0 + a, frame_c0 + b] = color

    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
