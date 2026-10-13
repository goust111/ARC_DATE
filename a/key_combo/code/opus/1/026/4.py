import random
from collections import deque
import numpy as np


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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


def _chambers(grid: np.ndarray, comp: list[tuple[int, int]]) -> int:
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return 0
    interior = grid[r0 + 1 : r1, c0 + 1 : c1] == 0
    return len(_components(interior))


def main(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    cyan = np.argwhere(grid == 8)
    r0, r1 = int(cyan[:, 0].min()), int(cyan[:, 0].max())
    c0, c1 = int(cyan[:, 1].min()), int(cyan[:, 1].max())

    frame_mask = (grid != 0) & (grid != 8)
    comp_id = {}
    chamber_by_id = {}
    for i, comp in enumerate(_components(frame_mask)):
        chamber_by_id[i] = _chambers(grid, comp)
        for p in comp:
            comp_id[p] = i

    color_for = {1: 2, 2: 4, 3: 6}
    rays = []
    for r in range(r0, r1 + 1):
        rays.append((r, c0 - 1, 0, -1))
        rays.append((r, c1 + 1, 0, 1))
    for c in range(c0, c1 + 1):
        rays.append((r0 - 1, c, -1, 0))
        rays.append((r1 + 1, c, 1, 0))

    h, w = grid.shape
    for sr, sc, dr, dc in rays:
        path = []
        r, c = sr, sc
        while 0 <= r < h and 0 <= c < w:
            if grid[r, c] != 0:
                cid = comp_id.get((r, c))
                if cid is not None:
                    paint = color_for.get(chamber_by_id.get(cid, 0))
                    if paint is not None:
                        for pr, pc in path:
                            if out[pr, pc] == 0:
                                out[pr, pc] = paint
                break
            path.append((r, c))
            r += dr
            c += dc
    return out


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int, chambers: int) -> None:
    r1, c1 = r0 + h - 1, c0 + w - 1
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    if chambers == 2:
        if random.choice([True, False]):
            cc = random.randint(c0 + 2, c1 - 2)
            grid[r0 + 1 : r1, cc] = color
        else:
            rr = random.randint(r0 + 2, r1 - 2)
            grid[rr, c0 + 1 : c1] = color
    elif chambers == 3:
        if random.choice([True, False]):
            for cc in (c0 + 2, c0 + 4):
                grid[r0 + 1 : r1, cc] = color
        else:
            for rr in (r0 + 2, r0 + 4):
                grid[rr, c0 + 1 : c1] = color


def generate_input() -> np.ndarray:
    grid = np.zeros((22, 22), dtype=int)
    grid[10:13, 10:13] = 8
    boxes = [(1, 8, 7, 7), (14, 8, 7, 7), (8, 1, 7, 7), (8, 14, 7, 7)]
    chambers = [1, 2, 3, random.choice([1, 2, 3])]
    random.shuffle(chambers)
    colors = random.sample([1, 3, 7, 9], 4)
    for (r, c, hh, ww), ch, color in zip(boxes, chambers, colors):
        _draw_frame(grid, r, c, hh, ww, color, ch)
    return grid
