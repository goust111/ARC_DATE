import random
import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = [(r, c)]
            seen[r, c] = True
            pts = []
            while q:
                x, y = q.pop()
                pts.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(pts)
    return comps


def _bbox(pts):
    rs = [p[0] for p in pts]
    cs = [p[1] for p in pts]
    return min(rs), max(rs), min(cs), max(cs)


def main(grid: np.ndarray) -> np.ndarray:
    """Slide sprites along same-colored tracks to endpoint blocks, recoloring when a dye station lies on the route.

    Input structure: black background with colored straight tracks, one same
    color 2x2 endpoint block per track, same-colored sprite components near the
    non-end side, and optional off-color single-pixel dye stations on tracks.
    Rule: pair each sprite to its track color, translate the sprite until its
    leading bbox edge touches the endpoint block's near side, recolor it if the
    swept route crosses a dye station, and erase all tracks and endpoint blocks.
    Key insight: track color pairs motion, but a visible station on the path can
    replace the sprite's final color.
    """
    out = np.zeros_like(grid)
    colors = [int(x) for x in np.unique(grid) if x != 0]
    for color in colors:
        comps = _components(grid == color)
        if len(comps) < 2:
            continue
        track = max(comps, key=len)
        sprites = [c for c in comps if c is not track]
        if not sprites:
            continue
        sprite = max(sprites, key=len)
        tr1, tr2, tc1, tc2 = _bbox(track)
        sr1, sr2, sc1, sc2 = _bbox(sprite)
        horizontal = (tc2 - tc1) >= (tr2 - tr1)
        endpoint_left_or_top = False
        if horizontal:
            left_block = np.sum(grid[tr1:tr2 + 1, tc1:tc1 + 2] == color)
            right_block = np.sum(grid[tr1:tr2 + 1, tc2 - 1:tc2 + 1] == color)
            endpoint_left_or_top = left_block >= right_block
            if endpoint_left_or_top:
                shift = (tc1 + 2) - sc1
            else:
                shift = (tc2 - 2) - sc2
            dr, dc = 0, shift
            path_cols = range(min(sc1, sc1 + dc), max(sc2, sc2 + dc) + 1)
            station = [(r, c) for r in range(tr1, tr2 + 1) for c in path_cols if 0 <= c < grid.shape[1] and grid[r, c] not in (0, color)]
        else:
            top_block = np.sum(grid[tr1:tr1 + 2, tc1:tc2 + 1] == color)
            bot_block = np.sum(grid[tr2 - 1:tr2 + 1, tc1:tc2 + 1] == color)
            endpoint_left_or_top = top_block >= bot_block
            if endpoint_left_or_top:
                shift = (tr1 + 2) - sr1
            else:
                shift = (tr2 - 2) - sr2
            dr, dc = shift, 0
            path_rows = range(min(sr1, sr1 + dr), max(sr2, sr2 + dr) + 1)
            station = [(r, c) for r in path_rows for c in range(tc1, tc2 + 1) if 0 <= r < grid.shape[0] and grid[r, c] not in (0, color)]
        final_color = int(grid[station[0]]) if station else color
        for r, c in sprite:
            nr, nc = r + dr, c + dc
            if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]:
                out[nr, nc] = final_color
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((24, 24), dtype=int)
    color = random.choice([1, 2, 3, 4, 6, 7, 8, 9])
    horizontal = random.random() < 0.5
    endpoint_at_start = random.random() < 0.5
    sprite_shape = [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1)]
    if horizontal:
        r, c1, c2 = random.randint(8, 15), 5, 18
        grid[r, c1:c2 + 1] = color
        if endpoint_at_start:
            grid[r - 1:r + 1, c1:c1 + 2] = color
            sr, sc = r - 2, c2 - 5
        else:
            grid[r - 1:r + 1, c2 - 1:c2 + 1] = color
            sr, sc = r - 2, c1 + 3
        if random.random() < 0.5:
            grid[r, random.randint(c1 + 3, c2 - 3)] = random.choice([x for x in [1, 2, 3, 4, 6, 7, 8, 9] if x != color])
    else:
        c, r1, r2 = random.randint(8, 15), 5, 18
        grid[r1:r2 + 1, c] = color
        if endpoint_at_start:
            grid[r1:r1 + 2, c - 1:c + 1] = color
            sr, sc = r2 - 5, c - 2
        else:
            grid[r2 - 1:r2 + 1, c - 1:c + 1] = color
            sr, sc = r1 + 3, c - 2
        if random.random() < 0.5:
            grid[random.randint(r1 + 3, r2 - 3), c] = random.choice([x for x in [1, 2, 3, 4, 6, 7, 8, 9] if x != color])
    for a, b in sprite_shape:
        grid[sr + a, sc + b] = color
    return grid
