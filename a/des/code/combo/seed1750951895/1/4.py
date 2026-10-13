import random
from collections import Counter, deque

import numpy as np


BG = 8
FRAME = 1
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE_COLORS = [0, 2, 3, 4, 5, 6, 7, 9]
LEGEND_ROWS = {
    "up": (0, 0),
    "left": (1, 0),
    "right": (2, 1),
    "down": (3, 0),
}


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
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _legend_candidates(h: int, w: int) -> list[tuple[int, int]]:
    return [(0, 0), (0, w - 2), (h - 4, 0), (h - 4, w - 2)]


def _detect_legend(grid: np.ndarray) -> tuple[dict[str, int], set[tuple[int, int]]]:
    h, w = grid.shape
    best = None
    best_score = -1
    for r0, c0 in _legend_candidates(h, w):
        if r0 < 0 or c0 < 0:
            continue
        win = grid[r0 : r0 + 4, c0 : c0 + 2]
        if win.shape != (4, 2):
            continue
        score = int(np.count_nonzero((win != BG) & (win != FRAME)))
        if score > best_score:
            best_score = score
            best = (r0, c0, win.copy())
    assert best is not None and best_score == 8, "legend not found"
    r0, c0, win = best
    mapping = {
        "up": int(win[0, 1]),
        "left": int(win[1, 1]),
        "right": int(win[2, 0]),
        "down": int(win[3, 1]),
    }
    legend_mask = {(r0 + r, c0 + c) for r in range(4) for c in range(2)}
    return mapping, legend_mask


def _find_gap(line: np.ndarray) -> tuple[int, int] | None:
    gaps = np.flatnonzero(line == BG)
    if gaps.size == 0:
        return None
    if np.any(line[(line != BG)] != FRAME):
        return None
    start = int(gaps[0])
    if not np.array_equal(gaps, np.arange(start, start + gaps.size)):
        return None
    if not (1 <= gaps.size <= 3):
        return None
    return start, int(gaps.size)


def _rectangle_info(grid: np.ndarray, comp: list[tuple[int, int]]) -> tuple[str, int, tuple[int, int, int, int]] | None:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 < 4 or c1 - c0 < 4:
        return None

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    perim = np.zeros_like(patch, dtype=bool)
    perim[0, :] = True
    perim[-1, :] = True
    perim[:, 0] = True
    perim[:, -1] = True
    if np.any((patch == FRAME) & (~perim)):
        return None

    top = _find_gap(patch[0, 1:-1])
    bottom = _find_gap(patch[-1, 1:-1])
    left = _find_gap(patch[1:-1, 0])
    right = _find_gap(patch[1:-1, -1])
    gaps = []
    if top is not None:
        gaps.append(("up", top[1]))
    if bottom is not None:
        gaps.append(("down", bottom[1]))
    if left is not None:
        gaps.append(("left", left[1]))
    if right is not None:
        gaps.append(("right", right[1]))
    if len(gaps) != 1:
        return None

    direction, gap_w = gaps[0]
    for name, line in (
        ("up", patch[0, :]),
        ("down", patch[-1, :]),
        ("left", patch[:, 0]),
        ("right", patch[:, -1]),
    ):
        vals = line[(line != BG)]
        if vals.size == 0 or np.any(vals != FRAME):
            return None
        if name != direction and np.any(line == BG):
            return None

    interior = patch[1:-1, 1:-1]
    if gap_w > min(interior.shape):
        return None
    return direction, gap_w, (r0, c0, r1, c1)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    mapping, legend_mask = _detect_legend(g)
    out = g.copy()

    blue_comps = _components(g == FRAME)
    for comp in blue_comps:
        if any(pt in legend_mask for pt in comp):
            continue
        info = _rectangle_info(g, comp)
        if info is None:
            continue
        direction, gap_w, (r0, c0, r1, c1) = info
        fill = mapping[direction]
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        sr = r0 + 1 + (inner_h - gap_w) // 2
        sc = c0 + 1 + (inner_w - gap_w) // 2
        out[sr : sr + gap_w, sc : sc + gap_w] = fill
    return out


def _place_legend(grid: np.ndarray, rng: random.Random, mapping: dict[str, int]) -> set[tuple[int, int]]:
    h, w = grid.shape
    corner = rng.choice(_legend_candidates(h, w))
    r0, c0 = corner
    marker_colors = rng.sample([c for c in PALETTE_COLORS if c not in mapping.values()], 4)
    marker_by_dir = dict(zip(["up", "down", "left", "right"], marker_colors))
    rows = [
        [marker_by_dir["up"], mapping["up"]],
        [marker_by_dir["left"], mapping["left"]],
        [mapping["right"], marker_by_dir["right"]],
        [marker_by_dir["down"], mapping["down"]],
    ]
    for r in range(4):
        for c in range(2):
            grid[r0 + r, c0 + c] = rows[r][c]
    return {(r0 + r, c0 + c) for r in range(4) for c in range(2)}


def _can_place(occ: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> bool:
    h, w = occ.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return not np.any(occ[rr0 : rr1 + 1, cc0 : cc1 + 1])


def _draw_rect_with_gap(
    grid: np.ndarray,
    occ: np.ndarray,
    r0: int,
    c0: int,
    rh: int,
    cw: int,
    direction: str,
    gap_w: int,
    fill: int,
) -> None:
    r1, c1 = r0 + rh - 1, c0 + cw - 1
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME

    if direction in ("up", "down"):
        start = c0 + 1 + (cw - 2 - gap_w) // 2
        rr = r0 if direction == "up" else r1
        grid[rr, start : start + gap_w] = BG
    else:
        start = r0 + 1 + (rh - 2 - gap_w) // 2
        cc = c0 if direction == "left" else c1
        grid[start : start + gap_w, cc] = BG

    inner_h = rh - 2
    inner_w = cw - 2
    sr = r0 + 1 + (inner_h - gap_w) // 2
    sc = c0 + 1 + (inner_w - gap_w) // 2
    grid[sr : sr + gap_w, sc : sc + gap_w] = BG

    occ[r0 : r1 + 1, c0 : c1 + 1] = True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, 28)
    w = rng.randint(18, 28)
    grid = np.full((h, w), BG, dtype=int)

    fills = rng.sample(PALETTE_COLORS, 4)
    mapping = dict(zip(["up", "down", "left", "right"], fills))
    legend_cells = _place_legend(grid, rng, mapping)

    occ = np.zeros((h, w), dtype=bool)
    for r, c in legend_cells:
        occ[r, c] = True

    n_rects = rng.randint(2, 5)
    directions = ["up", "down", "left", "right"]
    rng.shuffle(directions)
    rect_dirs = directions[: min(4, n_rects)]
    while len(rect_dirs) < n_rects:
        rect_dirs.append(rng.choice(directions))
    rng.shuffle(rect_dirs)

    placed = 0
    for direction in rect_dirs:
        for _ in range(80):
            gap_w = rng.randint(1, 3)
            if direction in ("up", "down"):
                rh = rng.randint(max(5, gap_w + 4), 9)
                cw = rng.randint(max(6, gap_w + 4), 11)
            else:
                rh = rng.randint(max(6, gap_w + 4), 11)
                cw = rng.randint(max(5, gap_w + 4), 9)
            if rh > h - 2 or cw > w - 2:
                continue
            r0 = rng.randint(1, h - rh - 1)
            c0 = rng.randint(1, w - cw - 1)
            r1, c1 = r0 + rh - 1, c0 + cw - 1
            if not _can_place(occ, r0, c0, r1, c1, pad=1):
                continue
            _draw_rect_with_gap(grid, occ, r0, c0, rh, cw, direction, gap_w, mapping[direction])
            placed += 1
            break

    if placed < 2:
        return None

    counts = Counter(grid.flatten().tolist())
    if counts[BG] <= counts[FRAME]:
        return None
    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(300):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
