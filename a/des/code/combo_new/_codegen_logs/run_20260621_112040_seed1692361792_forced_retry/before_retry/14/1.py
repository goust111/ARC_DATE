import random
from collections import deque

import numpy as np

BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), max(rows), min(cols), max(cols)


def _normalize(coords):
    r0, _, c0, _ = _bbox(coords)
    return {(r - r0, c - c0) for r, c in coords}


def _rotate_mask(mask: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(mask, k)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey fenced region with the matching outside pattern, rotated by the seed's quadrant.

    Input structure:
    - Black background with several one-cell-thick grey rectangular fences.
    - Each fence encloses exactly one colored seed pixel on black interior cells.
    - Outside the fences, each seed color appears exactly once as a monochrome connected pattern block.

    Rule:
    - For every fence, match the seed color to the outside pattern of the same color.
    - Determine the seed's quadrant inside the fence interior: upper-left keeps the pattern unchanged,
      upper-right rotates it 90 degrees clockwise, lower-right rotates it 180 degrees, and lower-left
      rotates it 270 degrees clockwise.
    - Stamp the rotated pattern with the seed pixel as the pattern's top-left anchor, clipping anything
      that would fall outside the fence interior.
    - Keep the grey fence and the outside pattern blocks unchanged.

    Key insight:
    - The seed color chooses which outside pattern to copy, while the seed position inside the fence
      independently chooses the rotation.
    """
    out = grid.copy()
    h, w = grid.shape

    fence_infos = []
    seed_positions = set()
    for comp in _components(grid == GREY):
        r0, r1, c0, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        if not np.all(grid[r0, c0 : c1 + 1] == GREY):
            continue
        if not np.all(grid[r1, c0 : c1 + 1] == GREY):
            continue
        if not np.all(grid[r0 : r1 + 1, c0] == GREY):
            continue
        if not np.all(grid[r0 : r1 + 1, c1] == GREY):
            continue
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        seeds = np.argwhere((interior != BLACK) & (interior != GREY))
        if len(seeds) != 1:
            continue
        sr, sc = seeds[0]
        sr += r0 + 1
        sc += c0 + 1
        seed_positions.add((sr, sc))
        fence_infos.append((r0, r1, c0, c1, sr, sc, int(grid[sr, sc])))

    patterns = {}
    for color in np.unique(grid):
        if color in (BLACK, GREY):
            continue
        comps = _components(grid == color)
        for comp in comps:
            if len(comp) == 1 and comp[0] in seed_positions:
                continue
            patterns[int(color)] = comp

    for r0, r1, c0, c1, sr, sc, color in fence_infos:
        pattern = patterns[color]
        pr0, pr1, pc0, pc1 = _bbox(pattern)
        mask = np.zeros((pr1 - pr0 + 1, pc1 - pc0 + 1), dtype=int)
        for rr, cc in pattern:
            mask[rr - pr0, cc - pc0] = 1

        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        rel_r = sr - (r0 + 1)
        rel_c = sc - (c0 + 1)
        top = 2 * rel_r < inner_h
        left = 2 * rel_c < inner_w
        if top and left:
            k = 0
        elif top and not left:
            k = 3
        elif not top and left:
            k = 1
        else:
            k = 2

        rotated = _rotate_mask(mask, k)
        for rr in range(rotated.shape[0]):
            for cc in range(rotated.shape[1]):
                if rotated[rr, cc] == 0:
                    continue
                tr, tc = sr + rr, sc + cc
                if r0 < tr < r1 and c0 < tc < c1 and 0 <= tr < h and 0 <= tc < w:
                    out[tr, tc] = color
    return out


def _random_pattern(rng: random.Random) -> np.ndarray:
    ph = rng.randint(2, 4)
    pw = rng.randint(2, 4)
    total = ph * pw
    target = rng.randint(max(3, total // 2), total)
    cells = {(rng.randrange(ph), rng.randrange(pw))}
    frontier = list(cells)
    while len(cells) < target:
        r, c = rng.choice(frontier)
        nbrs = [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
        rng.shuffle(nbrs)
        for nr, nc in nbrs:
            if 0 <= nr < ph and 0 <= nc < pw and (nr, nc) not in cells:
                cells.add((nr, nc))
                frontier.append((nr, nc))
                break
        else:
            frontier.append((rng.randrange(ph), rng.randrange(pw)))
    mask = np.zeros((ph, pw), dtype=int)
    for r, c in cells:
        mask[r, c] = 1
    rows = np.where(mask.sum(axis=1) > 0)[0]
    cols = np.where(mask.sum(axis=0) > 0)[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        h = rng.randint(18, 28)
        w = rng.randint(18, 28)
        grid = np.zeros((h, w), dtype=int)
        used = np.zeros((h, w), dtype=bool)
        num_fences = rng.randint(1, 3)
        colors = rng.sample(PALETTE, num_fences)
        fence_specs = []

        ok = True
        for color in colors:
            placed = False
            for _ in range(200):
                fh = rng.randint(6, 10)
                fw = rng.randint(6, 10)
                r0 = rng.randint(1, h - fh - 1)
                c0 = rng.randint(1, w - fw - 1)
                r1 = r0 + fh - 1
                c1 = c0 + fw - 1
                rr0 = max(0, r0 - 1)
                rr1 = min(h, r1 + 2)
                cc0 = max(0, c0 - 1)
                cc1 = min(w, c1 + 2)
                if used[rr0:rr1, cc0:cc1].any():
                    continue
                grid[r0, c0 : c1 + 1] = GREY
                grid[r1, c0 : c1 + 1] = GREY
                grid[r0 : r1 + 1, c0] = GREY
                grid[r0 : r1 + 1, c1] = GREY
                used[rr0:rr1, cc0:cc1] = True

                inner_h = fh - 2
                inner_w = fw - 2
                quadrant = rng.choice(["ul", "ur", "ll", "lr"])
                if quadrant == "ul":
                    sr = r0 + 1 + rng.randrange(max(1, (inner_h + 1) // 2))
                    sc = c0 + 1 + rng.randrange(max(1, (inner_w + 1) // 2))
                elif quadrant == "ur":
                    sr = r0 + 1 + rng.randrange(max(1, (inner_h + 1) // 2))
                    sc = c0 + 1 + rng.randrange(inner_w // 2, inner_w)
                elif quadrant == "ll":
                    sr = r0 + 1 + rng.randrange(inner_h // 2, inner_h)
                    sc = c0 + 1 + rng.randrange(max(1, (inner_w + 1) // 2))
                else:
                    sr = r0 + 1 + rng.randrange(inner_h // 2, inner_h)
                    sc = c0 + 1 + rng.randrange(inner_w // 2, inner_w)
                grid[sr, sc] = color
                fence_specs.append((color, r0, r1, c0, c1, sr, sc, _random_pattern(rng)))
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        for color, _, _, _, _, _, _, pattern in fence_specs:
            ph, pw = pattern.shape
            placed = False
            for _ in range(200):
                pr = rng.randint(0, h - ph)
                pc = rng.randint(0, w - pw)
                area = used[max(0, pr - 1) : min(h, pr + ph + 1), max(0, pc - 1) : min(w, pc + pw + 1)]
                if area.any():
                    continue
                cells = np.argwhere(pattern == 1)
                for rr, cc in cells:
                    grid[pr + rr, pc + cc] = color
                used[max(0, pr - 1) : min(h, pr + ph + 1), max(0, pc - 1) : min(w, pc + pw + 1)] = True
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return grid
    raise RuntimeError("failed to generate input")
