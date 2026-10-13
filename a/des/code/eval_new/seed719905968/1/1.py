import random
from collections import Counter, deque

import numpy as np


BG = 0
FRAME_COLOR = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRECTIONS = ("up", "down", "left", "right")

GRID_MIN = 15
GRID_MAX = 30
INNER_MIN = 3
INNER_MAX = 7
GEN_ATTEMPTS = 400
PLACEMENT_TRIES = 150
TRUNCATION_PROB = 0.3
TIE_GRID_PROB = 0.6

_RNG = random.Random()


def _bg_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


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
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _border_points(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    pts = set()
    for c in range(c0, c1 + 1):
        pts.add((r0, c))
        pts.add((r1, c))
    for r in range(r0, r1 + 1):
        pts.add((r, c0))
        pts.add((r, c1))
    return pts


def _frame_info(
    grid: np.ndarray,
    coords: list[tuple[int, int]],
    color: int,
    bg: int,
) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 4 or c1 - c0 < 4:
        return None

    border = _border_points(r0, c0, r1, c1)
    comp = set(coords)
    if not comp.issubset(border):
        return None
    missing = list(border - comp)
    if len(missing) != 1:
        return None

    mr, mc = missing[0]
    if mr in (r0, r1) and mc in (c0, c1):
        return None
    if mr == r0:
        direction = "up"
    elif mr == r1:
        direction = "down"
    elif mc == c0:
        direction = "left"
    elif mc == c1:
        direction = "right"
    else:
        return None

    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    seed_positions = np.argwhere(inner != bg)
    if not (2 <= len(seed_positions) <= 4):
        return None

    seed_colors = [int(inner[r, c]) for r, c in seed_positions]
    if len(set(seed_colors)) != len(seed_colors):
        return None
    if any(int(inner[r, c]) == color for r, c in seed_positions):
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "direction": direction,
        "frame_color": color,
    }


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        for comp in _components(grid == color):
            info = _frame_info(grid, comp, color, bg)
            if info is not None:
                frames.append(info)
    frames.sort(key=lambda item: item["bbox"])
    return frames


def _filled_pattern(inner: np.ndarray, bg: int) -> np.ndarray:
    pattern = inner.copy()
    seeds = [(int(r), int(c), int(inner[r, c])) for r, c in np.argwhere(inner != bg)]
    for r, c in np.argwhere(inner == bg):
        r = int(r)
        c = int(c)
        best_dist = None
        best_color = bg
        tie = False
        for sr, sc, color in seeds:
            dist = max(abs(r - sr), abs(c - sc))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_color = color
                tie = False
            elif dist == best_dist:
                tie = True
        if not tie:
            pattern[r, c] = best_color
    return pattern


def _paste_full(out: np.ndarray, block: np.ndarray, top: int, left: int) -> None:
    h, w = out.shape
    bh, bw = block.shape
    for r in range(bh):
        for c in range(bw):
            rr = top + r
            cc = left + c
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = int(block[r, c])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each holed grey frame by Chebyshev-nearest dye expansion, then export the interior pattern through the gap side.

    Input structure:
    - bg = black.
    - One or more thin grey rectangular frames appear on the grid; each frame is missing exactly one border pixel on one side.
    - Inside every frame are 2-4 differently colored single-pixel dye drops on an otherwise black interior.

    Rule:
    - For each frame interior, color every black cell by the color of its nearest dye drop using Chebyshev distance.
    - If two or more dye drops are tied for nearest distance, that interior cell stays black.
    - Keep the original frame, detect which side contains the one-pixel gap, and copy the filled interior rectangle to the outside of that side so the copy touches the frame with no empty gap.
    - Clip any copied pixels that would fall outside the grid.

    Key insight:
    - The puzzle combines two independent cues: the interior is a Voronoi partition under Chebyshev distance, while the frame's missing border pixel encodes the direction in which that finished interior must be exported.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    out = g.copy()

    for frame in _find_frames(g, bg):
        r0, c0, r1, c1 = frame["bbox"]
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        pattern = _filled_pattern(inner, bg)
        out[r0 + 1 : r1, c0 + 1 : c1] = pattern

        ph, pw = pattern.shape
        if frame["direction"] == "up":
            top, left = r0 - ph, c0 + 1
        elif frame["direction"] == "down":
            top, left = r1 + 1, c0 + 1
        elif frame["direction"] == "left":
            top, left = r0 + 1, c0 - pw
        else:
            top, left = r0 + 1, c1 + 1
        _paste_full(out, pattern, top, left)

    return out


def _expand_bbox(
    bbox: tuple[int, int, int, int], pad: int, h: int, w: int
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _clip_bbox(
    bbox: tuple[int, int, int, int], h: int, w: int
) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0)
    cc0 = max(0, c0)
    rr1 = min(h - 1, r1)
    cc1 = min(w - 1, c1)
    if rr0 > rr1 or cc0 > cc1:
        return None
    return rr0, cc0, rr1, cc1


def _union_bbox(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int] | None
) -> tuple[int, int, int, int]:
    if b is None:
        return a
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return min(ar0, br0), min(ac0, bc0), max(ar1, br1), max(ac1, bc1)


def _bbox_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _copy_bbox(
    r0: int,
    c0: int,
    inner_h: int,
    inner_w: int,
    direction: str,
) -> tuple[int, int, int, int]:
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    if direction == "up":
        return r0 - inner_h, c0 + 1, r0 - 1, c0 + inner_w
    if direction == "down":
        return r0 + outer_h, c0 + 1, r0 + outer_h + inner_h - 1, c0 + inner_w
    if direction == "left":
        return r0 + 1, c0 - inner_w, r0 + inner_h, c0 - 1
    return r0 + 1, c0 + outer_w, r0 + inner_h, c0 + outer_w + inner_w - 1


def _sample_position(
    rng: random.Random,
    h: int,
    w: int,
    inner_h: int,
    inner_w: int,
    direction: str,
    truncated: bool,
) -> tuple[int, int] | None:
    outer_h = inner_h + 2
    outer_w = inner_w + 2

    if not truncated:
        row_lo, row_hi = 0, h - outer_h
        col_lo, col_hi = 0, w - outer_w
        if direction == "up":
            row_lo = inner_h
        elif direction == "down":
            row_hi = h - outer_h - inner_h
        elif direction == "left":
            col_lo = inner_w
        else:
            col_hi = w - outer_w - inner_w
        if row_lo > row_hi or col_lo > col_hi:
            return None
        return rng.randint(row_lo, row_hi), rng.randint(col_lo, col_hi)

    if direction in ("up", "down") and inner_h < 2:
        return None
    if direction in ("left", "right") and inner_w < 2:
        return None

    if direction == "up":
        visible = rng.randint(1, inner_h - 1)
        row = visible
        col = rng.randint(0, w - outer_w)
        return row, col
    if direction == "down":
        visible = rng.randint(1, inner_h - 1)
        row = h - outer_h - visible
        col = rng.randint(0, w - outer_w)
        return row, col
    if direction == "left":
        visible = rng.randint(1, inner_w - 1)
        row = rng.randint(0, h - outer_h)
        col = visible
        return row, col
    visible = rng.randint(1, inner_w - 1)
    row = rng.randint(0, h - outer_h)
    col = w - outer_w - visible
    return row, col


def _random_seeds(
    rng: random.Random,
    inner_h: int,
    inner_w: int,
    force_tie: bool,
) -> list[tuple[int, int, int]]:
    if force_tie:
        colors = rng.sample(SEED_COLORS, 2)
        options: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for row in range(inner_h):
            for col in range(inner_w // 2):
                mirror = inner_w - 1 - col
                if mirror != col:
                    options.append(((row, col), (row, mirror)))
        for col in range(inner_w):
            for row in range(inner_h // 2):
                mirror = inner_h - 1 - row
                if mirror != row:
                    options.append(((row, col), (mirror, col)))
        if not options:
            force_tie = False
        else:
            (r0, c0), (r1, c1) = rng.choice(options)
            return [(r0, c0, colors[0]), (r1, c1, colors[1])]

    count = rng.randint(2, 4)
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    for _ in range(60):
        picks = rng.sample(cells, count)
        mindist = min(
            max(abs(r0 - r1), abs(c0 - c1))
            for i, (r0, c0) in enumerate(picks)
            for r1, c1 in picks[i + 1 :]
        )
        if mindist >= 1:
            colors = rng.sample(SEED_COLORS, count)
            return [(r, c, colors[i]) for i, (r, c) in enumerate(picks)]
    colors = rng.sample(SEED_COLORS, count)
    picks = rng.sample(cells, count)
    return [(r, c, colors[i]) for i, (r, c) in enumerate(picks)]


def _paint_frame(
    rng: random.Random,
    grid: np.ndarray,
    r0: int,
    c0: int,
    inner_h: int,
    inner_w: int,
    direction: str,
    seeds: list[tuple[int, int, int]],
) -> None:
    r1 = r0 + inner_h + 1
    c1 = c0 + inner_w + 1
    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR

    if direction == "up":
        gap = (r0, rng.randint(c0 + 1, c1 - 1))
    elif direction == "down":
        gap = (r1, rng.randint(c0 + 1, c1 - 1))
    elif direction == "left":
        gap = (rng.randint(r0 + 1, r1 - 1), c0)
    else:
        gap = (rng.randint(r0 + 1, r1 - 1), c1)
    grid[gap] = BG

    for sr, sc, color in seeds:
        grid[r0 + 1 + sr, c0 + 1 + sc] = color


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    n_frames = rng.choice((1, 2, 2, 3))
    size_floor = GRID_MIN + 3 * (n_frames - 1)
    h = rng.randint(size_floor, GRID_MAX)
    w = rng.randint(size_floor, GRID_MAX)
    grid = np.full((h, w), BG, dtype=int)

    directions = rng.sample(DIRECTIONS, n_frames)
    force_tie_index = rng.randrange(n_frames) if rng.random() < TIE_GRID_PROB else None
    reserved: list[tuple[int, int, int, int]] = []
    placements: list[tuple[int, int, int, int, str, list[tuple[int, int, int]]]] = []

    for idx, direction in enumerate(directions):
        placed = False
        for _ in range(PLACEMENT_TRIES):
            inner_h = rng.randint(INNER_MIN, INNER_MAX)
            inner_w = rng.randint(INNER_MIN, INNER_MAX)
            truncated = n_frames == 1 and rng.random() < TRUNCATION_PROB
            pos = _sample_position(rng, h, w, inner_h, inner_w, direction, truncated)
            if pos is None:
                continue

            r0, c0 = pos
            frame_bbox = (r0, c0, r0 + inner_h + 1, c0 + inner_w + 1)
            raw_copy = _copy_bbox(r0, c0, inner_h, inner_w, direction)
            reserve_bbox = _union_bbox(frame_bbox, _clip_bbox(raw_copy, h, w))
            expanded = _expand_bbox(reserve_bbox, 1, h, w)
            if any(_bbox_intersects(expanded, other) for other in reserved):
                continue

            seeds = _random_seeds(rng, inner_h, inner_w, force_tie=(idx == force_tie_index))
            placements.append((r0, c0, inner_h, inner_w, direction, seeds))
            reserved.append(expanded)
            placed = True
            break
        if not placed:
            return None

    for r0, c0, inner_h, inner_w, direction, seeds in placements:
        _paint_frame(rng, grid, r0, c0, inner_h, inner_w, direction, seeds)

    frames = _find_frames(grid, BG)
    if len(frames) != n_frames:
        return None
    if np.array_equal(main(grid), grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
