import numpy as np


BLACK = 0
MASTER_COLOR = 1
RED = 2


def _components(mask):
    rows, cols = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(rows):
        for c in range(cols):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _stamp(grid, pattern, center):
    rr, cc = center
    h, w = pattern.shape
    r0 = rr - h // 2
    c0 = cc - w // 2
    sub = grid[r0 : r0 + h, c0 : c0 + w]
    sub[pattern] = MASTER_COLOR


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the corner master sprite as a stamp and expand it around each seed according to adjacent red direction markers.

    Input structure:
    - Black background with one 3x3 master sprite in a corner, drawn in a single non-red color.
    - Separate singleton seed pixels of that same color appear elsewhere in the grid.
    - Each seed has one or two red cardinal-neighbor markers indicating growth directions.

    Rule:
    - Identify the unique multi-pixel 3x3 master sprite and treat it as the stamp template, centered on its own middle cell.
    - For every singleton seed, stamp one copy centered on the seed itself.
    - For each red direction marker, stamp another copy three cells away in that direction; if the seed has two directions, stamp one extra copy at the diagonal corner implied by both directions.
    - Remove all red markers and keep the original master sprite.

    Key insight:
    - The corner object is not another target instance: it defines the reusable stamp, while each seed-plus-markers bundle encodes a local layout of copies.
    """
    out = grid.copy()
    comps = _components(grid == MASTER_COLOR)
    master_comp = None
    for comp in comps:
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        if max(rs) - min(rs) == 2 and max(cs) - min(cs) == 2 and len(comp) > 1:
            master_comp = comp
            break
    rs = [p[0] for p in master_comp]
    cs = [p[1] for p in master_comp]
    r0, c0 = min(rs), min(cs)
    pattern = (grid[r0 : r0 + 3, c0 : c0 + 3] == MASTER_COLOR)
    for rr, cc in np.argwhere(grid == RED):
        out[rr, cc] = BLACK
    for comp in comps:
        if comp == master_comp:
            continue
        rr, cc = comp[0]
        dirs = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] == RED:
                dirs.append((dr, dc))
        _stamp(out, pattern, (rr, cc))
        for dr, dc in dirs:
            _stamp(out, pattern, (rr + 3 * dr, cc + 3 * dc))
        if len(dirs) == 2:
            dr = dirs[0][0] + dirs[1][0]
            dc = dirs[0][1] + dirs[1][1]
            _stamp(out, pattern, (rr + 3 * dr, cc + 3 * dc))
    return out


def _random_pattern(rng):
    pat = np.zeros((3, 3), dtype=bool)
    pat[1, 1] = True
    frontier = [(1, 1)]
    target = int(rng.integers(4, 8))
    while pat.sum() < target:
        rr, cc = frontier[int(rng.integers(0, len(frontier)))]
        nbrs = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < 3 and 0 <= nc < 3 and not pat[nr, nc]:
                nbrs.append((nr, nc))
        if not nbrs:
            break
        nr, nc = nbrs[int(rng.integers(0, len(nbrs)))]
        pat[nr, nc] = True
        frontier.append((nr, nc))
    pat[0, 1] = True
    pat[2, 1] = True
    pat[1, 0] = True
    pat[1, 2] = True
    return pat


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    grid = np.zeros((24, 24), dtype=int)
    pattern = _random_pattern(rng)
    corner = rng.choice(["tl", "tr", "bl", "br"])
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, 21
    elif corner == "bl":
        r0, c0 = 21, 0
    else:
        r0, c0 = 21, 21
    grid[r0 : r0 + 3, c0 : c0 + 3][pattern] = MASTER_COLOR

    occupied = np.zeros_like(grid, dtype=bool)
    occupied[max(0, r0 - 1) : min(24, r0 + 4), max(0, c0 - 1) : min(24, c0 + 4)] = True
    seed_count = int(rng.integers(2, 5))
    for _ in range(seed_count):
        for _attempt in range(200):
            rr = int(rng.integers(5, 19))
            cc = int(rng.integers(5, 19))
            dirs_all = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            rng.shuffle(dirs_all)
            dirs = dirs_all[: int(rng.integers(1, 3))]
            centers = [(rr, cc)] + [(rr + 3 * dr, cc + 3 * dc) for dr, dc in dirs]
            if len(dirs) == 2:
                centers.append((rr + 3 * (dirs[0][0] + dirs[1][0]), cc + 3 * (dirs[0][1] + dirs[1][1])))
            box_ok = True
            cells = []
            for cr, cc2 in centers:
                if not (1 <= cr <= 22 and 1 <= cc2 <= 22):
                    box_ok = False
                    break
                for pr in range(cr - 1, cr + 2):
                    for pc in range(cc2 - 1, cc2 + 2):
                        cells.append((pr, pc))
            if not box_ok or any(occupied[pr, pc] for pr, pc in cells):
                continue
            grid[rr, cc] = MASTER_COLOR
            for dr, dc in dirs:
                grid[rr + dr, cc + dc] = RED
            for pr, pc in cells:
                occupied[pr, pc] = True
            occupied[rr, cc] = True
            break
    return grid
