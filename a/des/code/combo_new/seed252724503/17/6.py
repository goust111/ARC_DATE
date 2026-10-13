import random
from collections import Counter

import numpy as np


BG = 0
BODY_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def _direction_from_pointer(mask: np.ndarray, pointer):
    h, w = mask.shape
    pr, pc = pointer
    roff = pr - (h - 1) / 2.0
    coff = pc - (w - 1) / 2.0
    if abs(roff) > abs(coff):
        return "down" if roff > 0 else "up"
    return "right" if coff > 0 else "left"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project a mirrored copy of each object in the direction indicated by its unique pointer pixel.

    Input structure:
    - The background is black and the foreground consists of several 4-connected objects.
    - Each object is mostly one color but contains exactly one differently colored pointer pixel.
    - The pointer pixel's position inside the object's bounding box determines one of the four cardinal directions.

    Rule:
    - Determine each object's direction from the pointer position relative to the object bbox center.
    - Place a mirrored copy immediately outside the bbox in that direction, flipping along the projection axis.
    - Recolor the whole projected copy with the pointer color and keep the original object unchanged; clip any out-of-bounds copy cells.

    Key insight:
    - The odd-colored pixel simultaneously specifies the travel direction, the reflection axis, and the color of the projected copy.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for cells in _components(g != BG):
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        mask = np.zeros_like(patch, dtype=bool)
        colors = []
        for r, c in cells:
            mask[r - r0, c - c0] = True
            colors.append(int(g[r, c]))
        counts = Counter(colors)
        pointer_color = min(counts, key=counts.get)
        pointer_cell = next((r - r0, c - c0) for r, c in cells if int(g[r, c]) == pointer_color)
        direction = _direction_from_pointer(mask, pointer_cell)
        if direction in ("up", "down"):
            mirrored = np.flipud(mask)
            top = r0 - mask.shape[0] if direction == "up" else r1 + 1
            left = c0
        else:
            mirrored = np.fliplr(mask)
            top = r0
            left = c0 - mask.shape[1] if direction == "left" else c1 + 1
        for rr in range(mirrored.shape[0]):
            for cc in range(mirrored.shape[1]):
                if not mirrored[rr, cc]:
                    continue
                gr, gc = top + rr, left + cc
                if 0 <= gr < out.shape[0] and 0 <= gc < out.shape[1]:
                    out[gr, gc] = pointer_color
    return out


def _random_mask(rng: random.Random):
    h = rng.randint(2, 4)
    w = rng.randint(2, 4)
    target = rng.randint(4, min(8, h * w))
    cells = {(rng.randrange(h), rng.randrange(w))}
    while len(cells) < target:
        r, c = rng.choice(tuple(cells))
        dr, dc = rng.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            cells.add((nr, nc))
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _pointer_candidates(mask: np.ndarray, direction: str):
    h, w = mask.shape
    result = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            roff = r - (h - 1) / 2.0
            coff = c - (w - 1) / 2.0
            if direction == "up" and roff < 0 and abs(roff) > abs(coff):
                result.append((r, c))
            if direction == "down" and roff > 0 and abs(roff) > abs(coff):
                result.append((r, c))
            if direction == "left" and coff < 0 and abs(coff) >= abs(roff):
                result.append((r, c))
            if direction == "right" and coff > 0 and abs(coff) >= abs(roff):
                result.append((r, c))
    return result


def _reserve(mask: np.ndarray, top: int, left: int, direction: str):
    h, w = mask.shape
    if direction == "up":
        return top - h, left, h * 2, w
    if direction == "down":
        return top, left, h * 2, w
    if direction == "left":
        return top, left - w, h, w * 2
    return top, left, h, w * 2


def _gen_attempt(rng: random.Random):
    h = rng.randint(16, 24)
    w = rng.randint(16, 24)
    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros_like(grid, dtype=bool)
    obj_count = rng.randint(2, 4)
    directions = [rng.choice(["up", "down", "left", "right"]) for _ in range(obj_count)]
    colors = rng.sample(BODY_COLORS, obj_count * 2)

    for idx in range(obj_count):
        direction = directions[idx]
        mask = _random_mask(rng)
        candidates = _pointer_candidates(mask, direction)
        if not candidates:
            return None
        pointer = rng.choice(candidates)
        body_color = colors[2 * idx]
        pointer_color = colors[2 * idx + 1]
        mh, mw = mask.shape
        placed = False
        for _ in range(100):
            top = rng.randint(1, h - mh - 1)
            left = rng.randint(1, w - mw - 1)
            rr0, cc0, rh, rw = _reserve(mask, top, left, direction)
            if rr0 < 0 or cc0 < 0 or rr0 + rh > h or cc0 + rw > w:
                continue
            if np.any(reserved[rr0 : rr0 + rh, cc0 : cc0 + rw]):
                continue
            patch = np.full((mh, mw), BG, dtype=int)
            patch[mask] = body_color
            patch[pointer] = pointer_color
            view = grid[top : top + mh, left : left + mw]
            view[patch != BG] = patch[patch != BG]
            reserved[rr0 : rr0 + rh, cc0 : cc0 + rw] = True
            placed = True
            break
        if not placed:
            return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
