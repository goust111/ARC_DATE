import random
import numpy as np


BG = 0


def _frames(grid: np.ndarray):
    frames = []
    for color in [int(c) for c in np.unique(grid) if c != BG]:
        mask = grid == color
        seen = set()
        for r, c in np.argwhere(mask):
            r = int(r); c = int(c)
            if (r, c) in seen:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            comp = []
            while stack:
                x, y = stack.pop()
                comp.append((x, y))
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and mask[nx, ny] and (nx, ny) not in seen:
                        seen.add((nx, ny))
                        stack.append((nx, ny))
            if len(comp) >= 8:
                rs = [p[0] for p in comp]
                cs = [p[1] for p in comp]
                r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
                if r1 - r0 + 1 >= 5 and c1 - c0 + 1 >= 5:
                    frames.append((color, r0, r1, c0, c1))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each interior sprite to the outside of the gap side of the same-colored target frame.

    Input structure:
    - Colored rectangular frames each have one side gap.
    - Inside every frame is a solid sprite whose color matches some other frame's border color.

    Rule:
    - Use the sprite color to identify the target frame.
    - Detect which side of the target frame contains its gap.
    - Erase the sprite from its source frame and paste it just outside the target gap,
      centered on that gap span.

    Key insight:
    - Sprite color acts as a pointer to another frame, while the target frame contributes
      only a placement side, not a recoloring rule.
    """
    out = grid.copy()
    frames = _frames(grid)
    by_color = {color: (r0, r1, c0, c1) for color, r0, r1, c0, c1 in frames}
    for color, r0, r1, c0, c1 in frames:
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        colors = [int(c) for c in np.unique(inner) if c != BG]
        if not colors:
            continue
        sc = colors[0]
        sprite_pts = np.argwhere(inner == sc)
        sr0, sc0 = sprite_pts.min(axis=0)
        sr1, sc1 = sprite_pts.max(axis=0)
        sprite = (inner[sr0 : sr1 + 1, sc0 : sc1 + 1] == sc).astype(int)
        out[r0 + 1 + sr0 : r0 + 2 + sr1, c0 + 1 + sc0 : c0 + 2 + sc1] = BG
        tr0, tr1, tc0, tc1 = by_color[sc]
        if np.any(grid[tr0, tc0 + 1 : tc1] == BG):
            cols = np.where(grid[tr0, tc0 + 1 : tc1] == BG)[0] + tc0 + 1
            cc = int((cols.min() + cols.max() - sprite.shape[1] + 1) // 2)
            rr = tr0 - sprite.shape[0]
        elif np.any(grid[tr1, tc0 + 1 : tc1] == BG):
            cols = np.where(grid[tr1, tc0 + 1 : tc1] == BG)[0] + tc0 + 1
            cc = int((cols.min() + cols.max() - sprite.shape[1] + 1) // 2)
            rr = tr1 + 1
        elif np.any(grid[tr0 + 1 : tr1, tc0] == BG):
            rows = np.where(grid[tr0 + 1 : tr1, tc0] == BG)[0] + tr0 + 1
            rr = int((rows.min() + rows.max() - sprite.shape[0] + 1) // 2)
            cc = tc0 - sprite.shape[1]
        else:
            rows = np.where(grid[tr0 + 1 : tr1, tc1] == BG)[0] + tr0 + 1
            rr = int((rows.min() + rows.max() - sprite.shape[0] + 1) // 2)
            cc = tc1 + 1
        rr = min(max(0, rr), grid.shape[0] - sprite.shape[0])
        cc = min(max(0, cc), grid.shape[1] - sprite.shape[1])
        out[rr : rr + sprite.shape[0], cc : cc + sprite.shape[1]][sprite == 1] = sc
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((24, 24), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(2, 3))
    used = []
    placed = []
    for color in colors:
        h = random.randint(5, 7)
        w = random.randint(5, 7)
        side = random.choice(["u", "d", "l", "r"])
        for _ in range(200):
            r = random.randint(3, 24 - h - 3)
            c = random.randint(3, 24 - w - 3)
            if all(r + h + 3 <= rr or rr + hh + 3 <= r or c + w + 3 <= cc or cc + ww + 3 <= c for rr, cc, hh, ww in used):
                used.append((r, c, h, w))
                grid[r, c : c + w] = color
                grid[r + h - 1, c : c + w] = color
                grid[r : r + h, c] = color
                grid[r : r + h, c + w - 1] = color
                if side == "u":
                    grid[r, c + w // 2 - 1 : c + w // 2 + 1] = BG
                elif side == "d":
                    grid[r + h - 1, c + w // 2 - 1 : c + w // 2 + 1] = BG
                elif side == "l":
                    grid[r + h // 2 - 1 : r + h // 2 + 1, c] = BG
                else:
                    grid[r + h // 2 - 1 : r + h // 2 + 1, c + w - 1] = BG
                placed.append((color, r, c, h, w))
                break
    if len(placed) < 2:
        return generate_input()
    targets = [p[0] for p in placed[1:]] + [placed[0][0]]
    for (_, r, c, h, w), target in zip(placed, targets):
        sh, sw = random.randint(1, 2), random.randint(1, 2)
        sr = r + 2
        sc = c + 2
        grid[sr : sr + sh, sc : sc + sw] = target
    return grid
