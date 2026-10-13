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


def _quadrant(r, c, h, w):
    return (0 if r < h / 2 else 1, 0 if c < w / 2 else 1)


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate marker-paired outside sprites into a grey frame and flood-fill from frame seeds.

    Input structure: one grey rectangular frame contains distinct colored seed
    pixels; outside the frame, each multicolor sprite has exactly one pixel in a
    matching seed color.
    Rule: pair each seed to the outside sprite containing its color, rotate the
    sprite so the marker pixel's bbox quadrant matches the seed's frame-internal
    quadrant, stamp it with the marker on the seed, flood-fill from each seed to
    recolor its connected stamped region, and output only the frame crop.
    Key insight: the external marker both identifies the seed and tells how the
    sprite must be oriented before seed-color flood propagation.
    """
    grey = 5
    frame = max(_components(grid == grey), key=len)
    fr1, fr2, fc1, fc2 = _bbox(frame)
    result = grid[fr1:fr2 + 1, fc1:fc2 + 1].copy()
    result[(result != 0) & (result != grey)] = 0
    inner = grid[fr1 + 1:fr2, fc1 + 1:fc2]
    seeds = [(fr1 + 1 + r, fc1 + 1 + c, int(inner[r, c])) for r, c in np.argwhere((inner != 0) & (inner != grey))]
    outside_mask = (grid != 0) & (grid != grey)
    outside_mask[fr1:fr2 + 1, fc1:fc2 + 1] = False
    sprites = _components(outside_mask)
    for sr, sc, seed_color in seeds:
        matched = None
        for comp in sprites:
            if any(grid[p] == seed_color for p in comp):
                matched = comp
                break
        if matched is None:
            continue
        r1, r2, c1, c2 = _bbox(matched)
        sprite = grid[r1:r2 + 1, c1:c2 + 1].copy()
        sprite[sprite == grey] = 0
        target_q = _quadrant(sr - (fr1 + 1), sc - (fc1 + 1), fr2 - fr1 - 1, fc2 - fc1 - 1)
        chosen = sprite
        marker = tuple(np.argwhere(sprite == seed_color)[0])
        for k in range(4):
            rot = np.rot90(sprite, k)
            locs = np.argwhere(rot == seed_color)
            if len(locs) and _quadrant(locs[0][0], locs[0][1], rot.shape[0], rot.shape[1]) == target_q:
                chosen = rot
                marker = tuple(locs[0])
                break
        top, left = sr - marker[0], sc - marker[1]
        for r in range(chosen.shape[0]):
            for c in range(chosen.shape[1]):
                color = int(chosen[r, c])
                if color == 0:
                    continue
                gr, gc = top + r, left + c
                if fr1 < gr < fr2 and fc1 < gc < fc2:
                    result[gr - fr1, gc - fc1] = color
    for sr, sc, seed_color in seeds:
        rr, cc = sr - fr1, sc - fc1
        if not (0 <= rr < result.shape[0] and 0 <= cc < result.shape[1]):
            continue
        q = [(rr, cc)]
        seen = {(rr, cc)}
        while q:
            r, c = q.pop()
            if result[r, c] in (0, grey):
                continue
            result[r, c] = seed_color
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < result.shape[0] and 0 <= nc < result.shape[1] and (nr, nc) not in seen and result[nr, nc] not in (0, grey):
                    seen.add((nr, nc))
                    q.append((nr, nc))
    return result


def generate_input() -> np.ndarray:
    grid = np.zeros((24, 24), dtype=int)
    fr1, fc1, fr2, fc2 = 6, 6, 17, 17
    grid[fr1:fr2 + 1, fc1] = 5
    grid[fr1:fr2 + 1, fc2] = 5
    grid[fr1, fc1:fc2 + 1] = 5
    grid[fr2, fc1:fc2 + 1] = 5
    seed_colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(2, 4))
    seed_spots = [(8, 8), (8, 15), (15, 8), (15, 15)]
    random.shuffle(seed_spots)
    sprite_origins = [(1, 1), (1, 17), (18, 1), (18, 17)]
    base_shapes = [[(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)], [(0, 0), (0, 1), (1, 1), (2, 1)], [(0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]]
    for i, color in enumerate(seed_colors):
        sr, sc = seed_spots[i]
        grid[sr, sc] = color
        orr, occ = sprite_origins[i]
        shape = base_shapes[i % len(base_shapes)]
        base = random.choice([x for x in [1, 2, 3, 4, 6, 7, 8, 9] if x != color])
        for a, b in shape:
            grid[orr + a, occ + b] = base
        mr, mc = random.choice(shape)
        grid[orr + mr, occ + mc] = color
    return grid
