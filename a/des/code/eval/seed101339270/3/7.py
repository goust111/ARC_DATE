from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME = 1   # blue
MARKER = 9  # maroon
SEED_COLORS = (2, 3, 4, 5, 6, 7, 8)
MAX_ATTEMPTS = 500


def _frame_border_ok(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    if not np.all(grid[r0, c0 : c1 + 1] == FRAME):
        return False
    if not np.all(grid[r1, c0 : c1 + 1] == FRAME):
        return False
    if not np.all(grid[r0 : r1 + 1, c0] == FRAME):
        return False
    if not np.all(grid[r0 : r1 + 1, c1] == FRAME):
        return False
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    return bool(np.any(inner != BG))


def _find_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    h, w = map(int, grid.shape)
    best: tuple[int, int, int, int] | None = None
    best_area = -1

    for r0 in range(h - 2):
        for c0 in range(w - 2):
            if int(grid[r0, c0]) != FRAME:
                continue
            for r1 in range(r0 + 2, h):
                if int(grid[r1, c0]) != FRAME:
                    continue
                for c1 in range(c0 + 2, w):
                    if int(grid[r0, c1]) != FRAME or int(grid[r1, c1]) != FRAME:
                        continue
                    if not _frame_border_ok(grid, r0, c0, r1, c1):
                        continue
                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    if area > best_area:
                        best_area = area
                        best = (r0, c0, r1, c1)

    assert best is not None, "expected one blue frame"
    return best


def _extract_seed(grid: np.ndarray, frame: tuple[int, int, int, int]) -> tuple[np.ndarray, int, int]:
    r0, c0, r1, c1 = frame
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    coords = np.argwhere(inner != BG)
    assert coords.size > 0, "frame interior must contain a seed pattern"
    sr0 = int(coords[:, 0].min())
    sc0 = int(coords[:, 1].min())
    sr1 = int(coords[:, 0].max())
    sc1 = int(coords[:, 1].max())
    seed = inner[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()
    return seed, int(r0 + 1 + sr0), int(c0 + 1 + sc0)


def _count_markers(grid: np.ndarray, frame: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = frame
    rows, cols = np.where(grid == MARKER)
    up = down = left = right = 0
    for rr, cc in zip(rows.tolist(), cols.tolist(), strict=False):
        if rr < r0:
            up += 1
        elif rr > r1:
            down += 1
        elif cc < c0:
            left += 1
        elif cc > c1:
            right += 1
    return up, down, left, right


def _tile_seed(
    canvas: np.ndarray,
    seed: np.ndarray,
    top: int,
    left: int,
    up: int,
    down: int,
    grow_left: int,
    grow_right: int,
) -> None:
    sh, sw = map(int, seed.shape)
    for tr in range(-up, down + 1):
        for tc in range(-grow_left, grow_right + 1):
            rr = top + tr * sh
            cc = left + tc * sw
            canvas[rr : rr + sh, cc : cc + sw] = seed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the blue frame and maroon counters, then grow the inner seed pattern.

    The seed is treated as a tile. If the frame has N counters above, below, left,
    and right, the output contains a tiled block with N extra tile-rows/columns on
    the corresponding sides of the original seed tile.
    """
    g = np.asarray(grid, dtype=int)
    frame = _find_frame(g)
    seed, top, left = _extract_seed(g, frame)
    up, down, grow_left, grow_right = _count_markers(g, frame)

    out = np.full_like(g, BG)
    _tile_seed(out, seed, top, left, up, down, grow_left, grow_right)
    return out


def _centered_padding() -> tuple[int, int]:
    base = random.randint(1, 3)
    delta = random.randint(0, 1)
    if random.random() < 0.5:
        return base, base + delta
    return base + delta, base


def _make_seed() -> np.ndarray:
    h = random.randint(2, 4)
    w = random.randint(2, 4)
    palette = random.sample(SEED_COLORS, random.randint(2, min(4, len(SEED_COLORS))))

    if random.random() < 0.7:
        seed = np.array([[random.choice(palette) for _ in range(w)] for _ in range(h)], dtype=int)
        if len(set(map(int, seed.reshape(-1)))) == 1:
            seed[random.randrange(h), random.randrange(w)] = random.choice([c for c in palette if c != int(seed[0, 0])])
        return seed

    while True:
        mask = np.random.rand(h, w) < random.uniform(0.55, 0.8)
        if int(mask.sum()) < max(3, (h * w) // 2):
            continue
        if not np.all(mask.any(axis=0)) or not np.all(mask.any(axis=1)):
            continue
        seed = np.zeros((h, w), dtype=int)
        colors = [random.choice(palette) for _ in range(int(mask.sum()))]
        random.shuffle(colors)
        idx = 0
        for r in range(h):
            for c in range(w):
                if mask[r, c]:
                    seed[r, c] = colors[idx]
                    idx += 1
        nonzero_colors = set(map(int, seed[seed != BG].reshape(-1)))
        if len(nonzero_colors) >= 2:
            return seed


def _sample_counts() -> tuple[int, int, int, int]:
    while True:
        counts = tuple(random.randint(0, 3) for _ in range(4))
        if any(counts):
            return counts  # type: ignore[return-value]


def _sample_markers(cells: list[tuple[int, int]], n: int) -> list[tuple[int, int]] | None:
    if n == 0:
        return []
    if len(cells) < n:
        return None

    random.shuffle(cells)
    chosen: list[tuple[int, int]] = []
    for cell in cells:
        if all(abs(cell[0] - rr) + abs(cell[1] - cc) > 1 for rr, cc in chosen):
            chosen.append(cell)
        if len(chosen) == n:
            return chosen

    return cells[:n] if len(cells) >= n else None


def _build_expected(
    shape: tuple[int, int],
    seed: np.ndarray,
    top: int,
    left: int,
    up: int,
    down: int,
    grow_left: int,
    grow_right: int,
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    _tile_seed(out, seed, top, left, up, down, grow_left, grow_right)
    return out


def _generate_candidate() -> np.ndarray | None:
    seed = _make_seed()
    sh, sw = map(int, seed.shape)
    up, down, grow_left, grow_right = _sample_counts()
    top_pad, bottom_pad = _centered_padding()
    left_pad, right_pad = _centered_padding()

    frame_h = top_pad + sh + bottom_pad + 2
    frame_w = left_pad + sw + right_pad + 2

    outer_top_min = max(0, up * sh - top_pad - 1, 1 if up > 0 else 0)
    outer_bottom_min = max(0, down * sh - bottom_pad - 1, 1 if down > 0 else 0)
    outer_left_min = max(0, grow_left * sw - left_pad - 1, 1 if grow_left > 0 else 0)
    outer_right_min = max(0, grow_right * sw - right_pad - 1, 1 if grow_right > 0 else 0)

    min_h = frame_h + outer_top_min + outer_bottom_min
    min_w = frame_w + outer_left_min + outer_right_min
    if min_h > 30 or min_w > 30:
        return None

    extra_h = random.randint(0, min(6, 30 - min_h))
    extra_w = random.randint(0, min(6, 30 - min_w))
    extra_top = random.randint(0, extra_h)
    extra_left = random.randint(0, extra_w)

    outer_top = outer_top_min + extra_top
    outer_bottom = outer_bottom_min + (extra_h - extra_top)
    outer_left = outer_left_min + extra_left
    outer_right = outer_right_min + (extra_w - extra_left)

    h = frame_h + outer_top + outer_bottom
    w = frame_w + outer_left + outer_right
    grid = np.full((h, w), BG, dtype=int)

    frame_r0 = outer_top
    frame_c0 = outer_left
    frame_r1 = frame_r0 + frame_h - 1
    frame_c1 = frame_c0 + frame_w - 1

    grid[frame_r0, frame_c0 : frame_c1 + 1] = FRAME
    grid[frame_r1, frame_c0 : frame_c1 + 1] = FRAME
    grid[frame_r0 : frame_r1 + 1, frame_c0] = FRAME
    grid[frame_r0 : frame_r1 + 1, frame_c1] = FRAME

    seed_top = frame_r0 + 1 + top_pad
    seed_left = frame_c0 + 1 + left_pad
    grid[seed_top : seed_top + sh, seed_left : seed_left + sw] = seed

    top_cells = [(r, c) for r in range(frame_r0) for c in range(frame_c0, frame_c1 + 1)]
    bottom_cells = [(r, c) for r in range(frame_r1 + 1, h) for c in range(frame_c0, frame_c1 + 1)]
    left_cells = [(r, c) for r in range(frame_r0, frame_r1 + 1) for c in range(frame_c0)]
    right_cells = [(r, c) for r in range(frame_r0, frame_r1 + 1) for c in range(frame_c1 + 1, w)]

    placements = [
        _sample_markers(top_cells, up),
        _sample_markers(bottom_cells, down),
        _sample_markers(left_cells, grow_left),
        _sample_markers(right_cells, grow_right),
    ]
    if any(p is None for p in placements):
        return None

    for pts in placements:
        assert pts is not None
        for r, c in pts:
            grid[r, c] = MARKER

    expected = _build_expected((h, w), seed, seed_top, seed_left, up, down, grow_left, grow_right)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        candidate = _generate_candidate()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid input for puzzle 7")
