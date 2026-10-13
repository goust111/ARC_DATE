from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
SEED_COLORS = [1, 2, 3, 4, 6, 7, 8]
PATTERN_LIBRARY = [
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=bool),
    np.array([[0, 0, 1], [0, 1, 1], [0, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1], [1, 0, 0]], dtype=bool),
]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400
_RNG = random.Random()


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


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
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_legend_bbox(g: np.ndarray) -> tuple[int, int, int, int]:
    h, w = g.shape
    candidates: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for comp in _components(g == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        frame = np.zeros_like(sub, dtype=bool)
        frame[0, :] = True
        frame[-1, :] = True
        frame[:, 0] = True
        frame[:, -1] = True
        if not np.array_equal(sub == GREY, frame):
            continue
        inner = sub[1:-1, 1:-1]
        if not np.any(inner == MAROON):
            continue
        corner_dist = min(r0 + c0, r0 + (w - 1 - c1), (h - 1 - r1) + c0, (h - 1 - r1) + (w - 1 - c1))
        area = int(sub.shape[0] * sub.shape[1])
        candidates.append((corner_dist, area, (r0, c0, r1, c1)))
    assert candidates, "expected one grey legend frame containing maroon patterns"
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def _decode_legend(g: np.ndarray, bbox: tuple[int, int, int, int]) -> dict[int, np.ndarray]:
    r0, c0, r1, c1 = bbox
    mapping: dict[int, np.ndarray] = {}
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            color = int(g[r, c])
            if color in (BLACK, GREY, MAROON):
                continue
            candidates: list[np.ndarray] = []
            blocks = [
                (r - 1, c + 1, r + 1, c + 3),
                (r - 1, c - 3, r + 1, c - 1),
                (r + 1, c - 1, r + 3, c + 1),
                (r - 3, c - 1, r - 1, c + 1),
            ]
            for br0, bc0, br1, bc1 in blocks:
                if br0 < r0 + 1 or bc0 < c0 + 1 or br1 > r1 - 1 or bc1 > c1 - 1:
                    continue
                block = g[br0 : br1 + 1, bc0 : bc1 + 1]
                if block.shape != (3, 3):
                    continue
                if not np.isin(block, [BLACK, MAROON]).all():
                    continue
                if not np.any(block == MAROON):
                    continue
                candidates.append(block == MAROON)
            assert len(candidates) == 1, "each legend seed must have exactly one adjacent 3x3 pattern"
            assert color not in mapping, "legend colors must be unique"
            mapping[color] = candidates[0]
    assert mapping, "legend must decode at least one color-to-pattern entry"
    return mapping


def _outside_passable(passable: np.ndarray) -> np.ndarray:
    h, w = passable.shape
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if passable[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if passable[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if passable[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    return outside


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a color-specific crystal pattern from each seed inside its closed grey region.

    The legend frame contains one color pixel per entry and one adjacent 3x3 maroon stamp.
    The stamp repeats on the 3x3 lattice centered at the seed, but only paints black cells
    inside the same enclosed region. The legend is removed in the output.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    legend_bbox = _find_legend_bbox(g)
    pattern_by_color = _decode_legend(g, legend_bbox)

    lr0, lc0, lr1, lc1 = legend_bbox
    out[lr0 : lr1 + 1, lc0 : lc1 + 1] = BLACK
    work = g.copy()
    work[lr0 : lr1 + 1, lc0 : lc1 + 1] = BLACK

    passable = work != GREY
    outside = _outside_passable(passable)
    h, w = g.shape
    seen = outside.copy()

    for r in range(h):
        for c in range(w):
            if not passable[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if passable[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            seeds = [(rr, cc, int(work[rr, cc])) for rr, cc in comp if int(work[rr, cc]) not in (BLACK,)]
            assert len(seeds) == 1, "each closed region must contain exactly one seed"
            sr, sc, color = seeds[0]
            assert color in pattern_by_color, "seed color must appear in the legend"
            stamp = pattern_by_color[color]

            for rr, cc in comp:
                if int(work[rr, cc]) != BLACK:
                    continue
                if stamp[(rr - sr + 1) % 3, (cc - sc + 1) % 3]:
                    out[rr, cc] = color

    return out


def _connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    h, w = mask.shape
    start_r, start_c = map(int, pts[0])
    q = deque([(start_r, start_c)])
    seen = {(start_r, start_c)}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _seed_candidates(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    safe: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            if all(0 <= r + dr < h and 0 <= c + dc < w and mask[r + dr, c + dc] for dr, dc in DIR4):
                safe.append((r, c))
    if safe:
        return safe
    return [tuple(map(int, pt)) for pt in np.argwhere(mask)]


def _random_region_mask(rng: random.Random, h: int, w: int) -> np.ndarray | None:
    for _ in range(80):
        mask = np.ones((h, w), dtype=bool)
        target_cuts = rng.randint(1, 4)
        accepted = 0
        for _ in range(40):
            if accepted >= target_cuts:
                break
            new = mask.copy()
            side = rng.choice(["top", "bottom", "left", "right"])
            cut_h = rng.randint(1, max(1, h // 2))
            cut_w = rng.randint(1, max(1, w // 2))
            if side == "top":
                c0 = rng.randint(0, w - cut_w)
                new[0:cut_h, c0 : c0 + cut_w] = False
            elif side == "bottom":
                c0 = rng.randint(0, w - cut_w)
                new[h - cut_h : h, c0 : c0 + cut_w] = False
            elif side == "left":
                r0 = rng.randint(0, h - cut_h)
                new[r0 : r0 + cut_h, 0:cut_w] = False
            else:
                r0 = rng.randint(0, h - cut_h)
                new[r0 : r0 + cut_h, w - cut_w : w] = False

            if int(new.sum()) < max(10, (h * w) // 2):
                continue
            if not (new[0].any() and new[-1].any() and new[:, 0].any() and new[:, -1].any()):
                continue
            if not _connected(new):
                continue
            mask = new
            accepted += 1

        if accepted == 0 or np.all(mask):
            continue
        if not _seed_candidates(mask):
            continue
        return mask
    return None


def _region_patch(mask: np.ndarray, color: int, rng: random.Random) -> np.ndarray:
    h, w = mask.shape
    padded = np.zeros((h + 2, w + 2), dtype=bool)
    padded[1:-1, 1:-1] = mask
    border = np.zeros_like(padded)
    ph, pw = padded.shape
    for r in range(ph):
        for c in range(pw):
            if padded[r, c]:
                continue
            for nr, nc in _neighbors4(r, c, ph, pw):
                if padded[nr, nc]:
                    border[r, c] = True
                    break

    seed_r, seed_c = rng.choice(_seed_candidates(mask))
    patch = np.full((h + 2, w + 2), BLACK, dtype=int)
    patch[border] = GREY
    patch[seed_r + 1, seed_c + 1] = int(color)
    return patch


def _build_legend(mapping: dict[int, np.ndarray], side: str, rng: random.Random) -> np.ndarray:
    colors = list(mapping)
    rng.shuffle(colors)
    if side in ("right", "left"):
        inner_h = len(colors) * 3 + (len(colors) - 1)
        inner_w = 4
    else:
        inner_h = 4
        inner_w = len(colors) * 3 + (len(colors) - 1)

    legend = np.full((inner_h + 2, inner_w + 2), BLACK, dtype=int)
    legend[0, :] = GREY
    legend[-1, :] = GREY
    legend[:, 0] = GREY
    legend[:, -1] = GREY
    inner = legend[1:-1, 1:-1]

    for idx, color in enumerate(colors):
        stamp = mapping[color]
        if side == "right":
            row = idx * 4
            inner[row + 1, 0] = int(color)
            inner[row : row + 3, 1:4][stamp] = MAROON
        elif side == "left":
            row = idx * 4
            inner[row + 1, 3] = int(color)
            inner[row : row + 3, 0:3][stamp] = MAROON
        elif side == "down":
            col = idx * 4
            inner[0, col + 1] = int(color)
            inner[1:4, col : col + 3][stamp] = MAROON
        else:
            col = idx * 4
            inner[3, col + 1] = int(color)
            inner[0:3, col : col + 3][stamp] = MAROON

    return legend


def _place_with_halo(grid: np.ndarray, patch: np.ndarray, occupied: np.ndarray, r0: int, c0: int) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    if r0 < 0 or c0 < 0 or r0 + ph > h or c0 + pw > w:
        return False
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h, r0 + ph + 1)
    cc1 = min(w, c0 + pw + 1)
    if occupied[rr0:rr1, cc0:cc1].any():
        return False
    grid[r0 : r0 + ph, c0 : c0 + pw] = patch
    occupied[rr0:rr1, cc0:cc1] = True
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    num_colors = rng.randint(2, 4)
    colors = rng.sample(SEED_COLORS, num_colors)
    patterns = rng.sample(PATTERN_LIBRARY, num_colors)
    mapping = {color: pattern.copy() for color, pattern in zip(colors, patterns)}

    side = rng.choice(["right", "left", "down", "up"])
    legend = _build_legend(mapping, side, rng)
    legend_h, legend_w = legend.shape

    num_regions = rng.randint(num_colors, min(5, num_colors + 2))
    grid_h = rng.randint(max(24, legend_h + 8), 34)
    grid_w = rng.randint(max(24, legend_w + 8), 34)
    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    corner = rng.choice(["tl", "tr", "bl", "br"])
    if corner == "tl":
        legend_r0, legend_c0 = 0, 0
    elif corner == "tr":
        legend_r0, legend_c0 = 0, grid_w - legend_w
    elif corner == "bl":
        legend_r0, legend_c0 = grid_h - legend_h, 0
    else:
        legend_r0, legend_c0 = grid_h - legend_h, grid_w - legend_w
    if not _place_with_halo(grid, legend, occupied, legend_r0, legend_c0):
        return None

    region_colors = colors[:]
    while len(region_colors) < num_regions:
        region_colors.append(rng.choice(colors))
    rng.shuffle(region_colors)

    for color in region_colors:
        placed = False
        for _ in range(100):
            inner_h = rng.randint(4, 8)
            inner_w = rng.randint(4, 8)
            mask = _random_region_mask(rng, inner_h, inner_w)
            if mask is None:
                continue
            patch = _region_patch(mask, color, rng)
            ph, pw = patch.shape
            if grid_h - ph - 2 < 1 or grid_w - pw - 2 < 1:
                continue
            r0 = rng.randint(1, grid_h - ph - 1)
            c0 = rng.randint(1, grid_w - pw - 1)
            if _place_with_halo(grid, patch, occupied, r0, c0):
                placed = True
                break
        if not placed:
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exceeded retry budget")
