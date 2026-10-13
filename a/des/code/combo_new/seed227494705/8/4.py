import random
from collections import deque

import numpy as np


BG = 0


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            yield pts


def _notch_side(mask):
    sides = []
    if np.any(~mask[0, :]):
        sides.append("top")
    if np.any(~mask[-1, :]):
        sides.append("bottom")
    if np.any(~mask[:, 0]):
        sides.append("left")
    if np.any(~mask[:, -1]):
        sides.append("right")
    return sides[0] if sides else "right"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each notched object by adding a mirrored copy on the notched side.

    Input structure:
    - Black background with separated single-color near-rectangular objects.
    - Each object's bounding rectangle has one rectangular notch cut into one side.
    - Objects have enough empty space on their notched side.

    Rule:
    - Detect the side whose bounding-box edge contains the inward background notch.
    - Mirror the object's mask across the axis perpendicular to that growth direction.
    - Paste the mirrored copy directly adjacent on the notched side, preserving the original object.

    Key insight:
    - The notch is both a direction marker and the geometric lock for the mirrored mate.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for color in [int(x) for x in np.unique(g) if int(x) != BG]:
        for pts in _components(g == color):
            rows = [p[0] for p in pts]
            cols = [p[1] for p in pts]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            patch = g[r0 : r1 + 1, c0 : c1 + 1] == color
            side = _notch_side(patch)
            if side in ("left", "right"):
                copy = np.fliplr(patch)
                nr0 = r0
                nc0 = c0 - patch.shape[1] if side == "left" else c1 + 1
            else:
                copy = np.flipud(patch)
                nr0 = r0 - patch.shape[0] if side == "top" else r1 + 1
                nc0 = c0
            for rr, cc in np.argwhere(copy):
                r, c = nr0 + int(rr), nc0 + int(cc)
                if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
                    out[r, c] = color
    return out


def _make_notched_rect(h, w, side, depth, start, length):
    mask = np.ones((h, w), dtype=bool)
    if side == "top":
        mask[:depth, start : start + length] = False
    elif side == "bottom":
        mask[h - depth :, start : start + length] = False
    elif side == "left":
        mask[start : start + length, :depth] = False
    else:
        mask[start : start + length, w - depth :] = False
    return mask


def generate_input() -> np.ndarray:
    rng = random.Random()
    colors = [1, 2, 3, 4, 6, 7, 8, 9]
    for _ in range(500):
        h = rng.randint(18, 30)
        w = rng.randint(18, 30)
        g = np.zeros((h, w), dtype=int)
        occupied = set()
        ok = True
        for color in rng.sample(colors, rng.randint(2, 4)):
            placed = False
            for _try in range(200):
                oh = rng.randint(4, 7)
                ow = rng.randint(4, 7)
                side = rng.choice(["top", "bottom", "left", "right"])
                depth = rng.randint(1, 2)
                if side in ("top", "bottom"):
                    length = rng.randint(1, max(1, ow - 2))
                    start = rng.randint(1, ow - length - 1)
                else:
                    length = rng.randint(1, max(1, oh - 2))
                    start = rng.randint(1, oh - length - 1)
                mask = _make_notched_rect(oh, ow, side, depth, start, length)
                if side == "left":
                    r0 = rng.randint(1, h - oh - 1)
                    c0 = rng.randint(ow + 1, w - ow - 1)
                    dup_origin = (r0, c0 - ow)
                elif side == "right":
                    r0 = rng.randint(1, h - oh - 1)
                    c0 = rng.randint(1, w - 2 * ow - 1)
                    dup_origin = (r0, c0 + ow)
                elif side == "top":
                    r0 = rng.randint(oh + 1, h - oh - 1)
                    c0 = rng.randint(1, w - ow - 1)
                    dup_origin = (r0 - oh, c0)
                else:
                    r0 = rng.randint(1, h - 2 * oh - 1)
                    c0 = rng.randint(1, w - ow - 1)
                    dup_origin = (r0 + oh, c0)
                cells = {(r0 + r, c0 + c) for r, c in np.argwhere(mask)}
                dup = np.fliplr(mask) if side in ("left", "right") else np.flipud(mask)
                dup_cells = {(dup_origin[0] + int(r), dup_origin[1] + int(c)) for r, c in np.argwhere(dup)}
                footprint = cells | dup_cells
                padded = {(r + dr, c + dc) for r, c in footprint for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
                if padded & occupied:
                    continue
                if any(not (0 <= r < h and 0 <= c < w) for r, c in footprint):
                    continue
                occupied |= padded
                for r, c in cells:
                    g[r, c] = color
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return g
    raise RuntimeError("generate_input failed")
