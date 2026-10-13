from collections import Counter, deque

import numpy as np


BG = 0
FRAME = 7
SEED_PALETTE = [1, 2, 3, 4, 5, 6, 8, 9]

GRID_MIN = 14
GRID_MAX = 30
FRAME_COUNT_RANGE = (1, 4)
INNER_H_RANGE = (2, 10)
INNER_W_RANGE = (2, 10)
PLACE_RETRIES = 80
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        pts: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            pts.append((int(r), int(c)))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(pts)
    return comps


def _bbox(pts: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _is_hollow_frame(pts: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(pts)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return False

    border_size = 2 * height + 2 * width - 4
    pset = set(pts)
    if len(pset) != border_size:
        return False

    for c in range(c0, c1 + 1):
        if (r0, c) not in pset or (r1, c) not in pset:
            return False
    for r in range(r0, r1 + 1):
        if (r, c0) not in pset or (r, c1) not in pset:
            return False
    return True


def _detect_frames(grid: np.ndarray) -> tuple[int, list[tuple[int, int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    counts = Counter(map(int, g.flatten()))

    candidates: list[tuple[int, list[list[tuple[int, int]]]]] = []
    for color, count in counts.items():
        color = int(color)
        if color == BG or count < 8:
            continue
        comps = _components(g == color)
        if comps and all(_is_hollow_frame(comp) for comp in comps):
            candidates.append((color, comps))

    assert len(candidates) == 1, "expected exactly one frame color"
    frame_color, comps = candidates[0]
    frames = [_bbox(comp) for comp in comps]
    frames.sort()
    return int(frame_color), frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each seed inside its rectangular frame into the largest centered solid square allowed by the nearest inner wall.

    Input structure:
    - The background is black.
    - One or more non-overlapping orange one-pixel hollow rectangular frames are scattered across the grid.
    - Each frame interior contains exactly one non-black, non-orange single-pixel seed, with the rest of the interior left black.

    Rule:
    - Keep every orange frame unchanged.
    - For each frame, locate its unique seed and measure the seed's row/column gaps to the four inner walls of that frame.
    - Let m be the smallest of those four gaps, then fill the seed's color on the centered square spanning m cells in every cardinal direction from the seed, clipped to the frame interior.

    Key insight:
    - The seed does not expand freely in the whole canvas; its own containing frame sets four local limits, and the nearest wall determines the square-growth radius.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    frame_color, frames = _detect_frames(g)
    for r0, c0, r1, c1 in frames:
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_locs = np.argwhere((interior != BG) & (interior != frame_color))
        assert seed_locs.shape[0] == 1, "each frame must contain exactly one seed"

        sr = int(seed_locs[0, 0] + r0 + 1)
        sc = int(seed_locs[0, 1] + c0 + 1)
        seed_color = int(g[sr, sc])

        d_up = sr - (r0 + 1)
        d_down = (r1 - 1) - sr
        d_left = sc - (c0 + 1)
        d_right = (c1 - 1) - sc
        radius = min(d_up, d_down, d_left, d_right)

        rr0 = max(r0 + 1, sr - radius)
        rr1 = min(r1 - 1, sr + radius)
        cc0 = max(c0 + 1, sc - radius)
        cc1 = min(c1 - 1, sc + radius)
        out[rr0 : rr1 + 1, cc0 : cc1 + 1] = seed_color

    return out


def _randint(rng: np.random.Generator, bounds: tuple[int, int]) -> int:
    lo, hi = bounds
    return int(rng.integers(lo, hi + 1))


def _choose_frame_count(rng: np.random.Generator, shape: tuple[int, int]) -> int:
    h, w = shape
    max_by_area = max(1, min(FRAME_COUNT_RANGE[1], (h * w) // 70))
    lo = FRAME_COUNT_RANGE[0]
    hi = max(lo, max_by_area)
    return int(rng.integers(lo, hi + 1))


def _sample_inner_size(rng: np.random.Generator) -> tuple[int, int]:
    kind = int(rng.integers(0, 3))
    if kind == 0:
        inner_h = _randint(rng, (2, 6))
        inner_w = _randint(rng, (5, 10))
    elif kind == 1:
        inner_h = _randint(rng, (5, 10))
        inner_w = _randint(rng, (2, 6))
    else:
        inner_h = _randint(rng, INNER_H_RANGE)
        inner_w = _randint(rng, INNER_W_RANGE)
    return inner_h, inner_w


def _overlaps_with_gap(
    bbox: tuple[int, int, int, int], placed: list[tuple[int, int, int, int]], gap: int = 1
) -> bool:
    r0, c0, r1, c1 = bbox
    for pr0, pc0, pr1, pc1 in placed:
        if not (r1 + gap < pr0 or pr1 + gap < r0 or c1 + gap < pc0 or pc1 + gap < c0):
            return True
    return False


def _radius_groups(inner_h: int, inner_w: int) -> dict[int, list[tuple[int, int]]]:
    groups: dict[int, list[tuple[int, int]]] = {}
    for r in range(inner_h):
        for c in range(inner_w):
            radius = min(r, inner_h - 1 - r, c, inner_w - 1 - c)
            groups.setdefault(int(radius), []).append((int(r), int(c)))
    return groups


def _choose_seed(
    rng: np.random.Generator, inner_h: int, inner_w: int
) -> tuple[int, int]:
    groups = _radius_groups(inner_h, inner_w)
    radii = sorted(groups)
    radius = int(rng.choice(radii))
    cells = groups[radius]
    idx = int(rng.integers(0, len(cells)))
    return cells[idx]


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = _randint(rng, (GRID_MIN, GRID_MAX))
    w = _randint(rng, (GRID_MIN, GRID_MAX))
    grid = np.full((h, w), BG, dtype=int)

    frame_count = _choose_frame_count(rng, grid.shape)
    if frame_count > len(SEED_PALETTE):
        return None
    colors = list(rng.choice(SEED_PALETTE, size=frame_count, replace=False))

    placed: list[tuple[int, int, int, int]] = []
    specs: list[tuple[int, int, int, int, int]] = []
    for color in colors:
        placed_this = False
        for _ in range(PLACE_RETRIES):
            inner_h, inner_w = _sample_inner_size(rng)
            outer_h = inner_h + 2
            outer_w = inner_w + 2
            if outer_h > h or outer_w > w:
                continue
            r0 = int(rng.integers(0, h - outer_h + 1))
            c0 = int(rng.integers(0, w - outer_w + 1))
            bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            if _overlaps_with_gap(bbox, placed, gap=1):
                continue
            placed.append(bbox)
            specs.append((r0, c0, inner_h, inner_w, int(color)))
            placed_this = True
            break
        if not placed_this:
            return None

    for r0, c0, inner_h, inner_w, color in specs:
        r1 = r0 + inner_h + 1
        c1 = c0 + inner_w + 1
        grid[r0, c0 : c1 + 1] = FRAME
        grid[r1, c0 : c1 + 1] = FRAME
        grid[r0 : r1 + 1, c0] = FRAME
        grid[r0 : r1 + 1, c1] = FRAME

        sr_in, sc_in = _choose_seed(rng, inner_h, inner_w)
        grid[r0 + 1 + sr_in, c0 + 1 + sc_in] = int(color)

    try:
        out = main(grid)
    except AssertionError:
        return None

    assert out.shape == grid.shape
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
