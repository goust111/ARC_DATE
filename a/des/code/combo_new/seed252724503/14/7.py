import random
from collections import deque

import numpy as np

BLACK = 0
GREY = 5


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(cells)
    return parts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill half of each framed rectangle from the side indicated by its seed position, or fill all if the seed is central.

    Input structure:
    - Black background with several disjoint grey rectangular outlines.
    - Each outline encloses one colored singleton seed pixel.
    - The seed color is the fill color for that rectangle.

    Rule:
    - For each frame, inspect the seed's position within the interior.
    - If the seed lies in the top, bottom, left, or right outer quarter, fill the corresponding half of the interior from that side with the seed color.
    - If the seed lies in none of those edge quarters, fill the entire interior.

    Key insight:
    - The seed encodes two independent facts at once: its color gives the paint, and its offset within the frame gives the fill direction.
    """
    out = grid.copy()
    for cells in _components(grid == GREY):
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        inner = out[r0 + 1:r1, c0 + 1:c1]
        sr, sc = next((r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1) if grid[r, c] not in (BLACK, GREY))
        color = int(grid[sr, sc])
        ir, ic = sr - (r0 + 1), sc - (c0 + 1)
        ih, iw = inner.shape
        top_q, bottom_q = ih // 4, ih // 4
        left_q, right_q = iw // 4, iw // 4
        if top_q and ir < top_q:
            inner[: ih // 2, :] = color
        elif bottom_q and ir >= ih - bottom_q:
            inner[ih - ih // 2 :, :] = color
        elif left_q and ic < left_q:
            inner[:, : iw // 2] = color
        elif right_q and ic >= iw - right_q:
            inner[:, iw - iw // 2 :] = color
        else:
            inner[:, :] = color
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    colors = [1, 2, 3, 4, 6, 7, 8, 9]
    zones = ["top", "bottom", "left", "right", "center"]
    for _ in range(400):
        h = rng.randint(16, 24)
        w = rng.randint(16, 24)
        grid = np.zeros((h, w), dtype=int)
        occ = set()
        ok = True
        for color in rng.sample(colors, rng.randint(2, 4)):
            fh = rng.randint(5, 8)
            fw = rng.randint(5, 8)
            for _ in range(100):
                r0 = rng.randint(1, h - fh - 1)
                c0 = rng.randint(1, w - fw - 1)
                frame_cells = {(r0, c) for c in range(c0, c0 + fw)} | {(r0 + fh - 1, c) for c in range(c0, c0 + fw)} | {(r, c0) for r in range(r0, r0 + fh)} | {(r, c0 + fw - 1) for r in range(r0, r0 + fh)}
                if any(cell in occ for cell in frame_cells):
                    continue
                for r, c in frame_cells:
                    grid[r, c] = GREY
                occ |= frame_cells
                zone = rng.choice(zones)
                ir0, ic0 = r0 + 1, c0 + 1
                ih, iw = fh - 2, fw - 2
                if zone == "top":
                    sr = ir0 + rng.randint(0, max(0, ih // 4 - 1))
                    sc = ic0 + rng.randint(iw // 4, max(iw // 4, iw - iw // 4 - 1))
                elif zone == "bottom":
                    sr = ir0 + rng.randint(max(0, ih - max(1, ih // 4)), ih - 1)
                    sc = ic0 + rng.randint(iw // 4, max(iw // 4, iw - iw // 4 - 1))
                elif zone == "left":
                    sr = ir0 + rng.randint(ih // 4, max(ih // 4, ih - ih // 4 - 1))
                    sc = ic0 + rng.randint(0, max(0, iw // 4 - 1))
                elif zone == "right":
                    sr = ir0 + rng.randint(ih // 4, max(ih // 4, ih - ih // 4 - 1))
                    sc = ic0 + rng.randint(max(0, iw - max(1, iw // 4)), iw - 1)
                else:
                    sr = ir0 + ih // 2
                    sc = ic0 + iw // 2
                grid[sr, sc] = color
                occ.add((sr, sc))
                break
            else:
                ok = False
                break
        if ok:
            return grid
    raise RuntimeError("failed to generate task 14/7")
