import random
import numpy as np


BG = 0
GREY = 5


def _components(mask: np.ndarray):
    seen = set()
    out = []
    H, W = mask.shape
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or (r, c) in seen:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            comp = []
            while stack:
                x, y = stack.pop()
                comp.append((x, y))
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < H and 0 <= ny < W and mask[nx, ny] and (nx, ny) not in seen:
                        seen.add((nx, ny))
                        stack.append((nx, ny))
            out.append(comp)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Join each same-colored horizontal/vertical segment pair by translating the vertical segment to the grey joint.

    Input structure:
    - Every color appears on exactly two straight segments: one horizontal and one vertical.
    - One endpoint of each segment is grey instead of the segment color.

    Rule:
    - Keep the horizontal segment fixed.
    - Translate the vertical segment so its grey endpoint becomes vertically adjacent to
      the horizontal segment's grey endpoint, on the side away from the vertical segment body.
    - Recolor both grey joint pixels to the pair color.

    Key insight:
    - The grey endpoint identifies both the connection end and the body direction, which
      determines on which side the translated joint must land.
    """
    out = np.zeros_like(grid)
    colors = [int(c) for c in np.unique(grid) if c not in (BG, GREY)]
    for color in colors:
        comps = []
        for comp in _components((grid == color) | (grid == GREY)):
            if any(grid[p] == color for p in comp):
                if any(grid[p] == GREY for p in comp):
                    comps.append(comp)
        comps = [c for c in comps if any(grid[p] == color for p in c)]
        uniq = []
        for comp in comps:
            if comp not in uniq:
                uniq.append(comp)
        pair = []
        for comp in uniq:
            if any(grid[p] == color for p in comp):
                rows = {r for r, _ in comp}
                cols = {c for _, c in comp}
                if len(rows) == 1 or len(cols) == 1:
                    pair.append(comp)
        if len(pair) < 2:
            continue
        a, b = pair[:2]
        ha, hb = (a, b) if len({r for r, _ in a}) == 1 else (b, a)
        hgrey = next(p for p in ha if grid[p] == GREY)
        vgrey = next(p for p in hb if grid[p] == GREY)
        vbody = next(p for p in hb if grid[p] == color and p[1] == vgrey[1])
        target = (hgrey[0] + (1 if vbody[0] > vgrey[0] else -1), hgrey[1])
        dr, dc = target[0] - vgrey[0], target[1] - vgrey[1]
        for p in ha:
            out[p] = color
        for r, c in hb:
            nr, nc = r + dr, c + dc
            if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]:
                out[nr, nc] = color
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((22, 22), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(2, 4))
    used = []
    for color in colors:
        for _ in range(200):
            hr = random.randint(2, 18)
            hc = random.randint(2, 14)
            hl = random.randint(3, 5)
            vr = random.randint(2, 14)
            vc = random.randint(2, 18)
            vl = random.randint(3, 5)
            boxes = [(hr, hc, 1, hl), (vr, vc, vl, 1)]
            if all(bx + bw + 4 <= ux or ux + uw + 4 <= bx or by + bh + 4 <= uy or uy + uh + 4 <= by for bx, by, bh, bw in [(hc, hr, 1, hl), (vc, vr, vl, 1)] for ux, uy, uh, uw in used):
                used.extend([(hc, hr, 1, hl), (vc, vr, vl, 1)])
                grid[hr, hc : hc + hl] = color
                grid[vr : vr + vl, vc] = color
                grid[hr, hc] = GREY
                if random.random() < 0.5:
                    grid[vr, vc] = GREY
                else:
                    grid[vr + vl - 1, vc] = GREY
                break
    return grid
