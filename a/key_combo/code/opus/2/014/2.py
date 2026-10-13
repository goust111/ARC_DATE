import random
from collections import deque

import numpy as np


def _neighbors4(r, c, h, w):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(cells)
    return parts


def _piece_info(grid, cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    piece = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    colors = [grid[r, c] for r, c in cells if grid[r, c] not in (2, 6)]
    base = max(set(colors), key=colors.count)
    restored = piece.copy()
    restored[restored == 2] = base
    return {
        "bbox": (r0, c0, r1, c1),
        "piece": restored,
        "base": base,
        "anchor": np.any(piece == 6),
    }


def _assemble(grid):
    infos = [_piece_info(grid, cells) for cells in _components(grid != 0)]
    anchor = next(info for info in infos if info["anchor"])
    others = [info for info in infos if not info["anchor"]]
    horizontals = [info for info in others if info["piece"].shape[1] > info["piece"].shape[0]]
    verticals = [info for info in others if info["piece"].shape[0] > info["piece"].shape[1]]
    center = anchor["piece"]
    top = horizontals[0]["piece"]
    bottom = horizontals[1]["piece"]
    left = verticals[0]["piece"]
    right = verticals[1]["piece"]
    h = top.shape[0] + center.shape[0] + bottom.shape[0]
    w = left.shape[1] + center.shape[1] + right.shape[1]
    out = np.zeros((h, w), dtype=int)
    out[: top.shape[0], :] = top
    out[h - bottom.shape[0] :, :] = bottom
    out[top.shape[0] : top.shape[0] + center.shape[0], : left.shape[1]] = left
    out[top.shape[0] : top.shape[0] + center.shape[0], w - right.shape[1] :] = right
    out[
        top.shape[0] : top.shape[0] + center.shape[0],
        left.shape[1] : left.shape[1] + center.shape[1],
    ] = center
    return out


def _reverse_layers(rect):
    out = rect.copy()
    colors = []
    bounds = []
    r0, c0 = 0, 0
    r1, c1 = rect.shape[0] - 1, rect.shape[1] - 1
    while r0 <= r1 and c0 <= c1:
        perim = []
        pos = []
        for c in range(c0, c1 + 1):
            pos.append((r0, c))
        for r in range(r0 + 1, r1 + 1):
            pos.append((r, c1))
        if r1 > r0:
            for c in range(c1 - 1, c0 - 1, -1):
                pos.append((r1, c))
        if c1 > c0:
            for r in range(r1 - 1, r0, -1):
                pos.append((r, c0))
        perim = [rect[r, c] for r, c in pos if rect[r, c] != 6]
        if not perim:
            break
        colors.append(max(set(perim), key=perim.count))
        bounds.append((r0, c0, r1, c1))
        r0 += 1
        c0 += 1
        r1 -= 1
        c1 -= 1
    for idx, (r0, c0, r1, c1) in enumerate(bounds):
        color = colors[-1 - idx]
        for c in range(c0, c1 + 1):
            if out[r0, c] != 6:
                out[r0, c] = color
        for r in range(r0 + 1, r1 + 1):
            if out[r, c1] != 6:
                out[r, c1] = color
        if r1 > r0:
            for c in range(c1 - 1, c0 - 1, -1):
                if out[r1, c] != 6:
                    out[r1, c] = color
        if c1 > c0:
            for r in range(r1 - 1, r0, -1):
                if out[r, c0] != 6:
                    out[r, c0] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Summary:
    Reassemble scattered rectangular sprites into one nested rectangle, then reverse the color order of its concentric layers.

    Input Model:
    One sprite contains a 2x2 magenta anchor block, the remaining sprites are rectangular strips with red connector pixels that become their base color after assembly.

    Algorithm:
    Recover each sprite component, restore connector pixels to the sprite base color, place the anchor piece in the center with horizontal strips above and below and vertical strips on the sides, then peel one-cell rings and repaint them in reverse layer order while leaving magenta untouched.

    Output:
    Return the cropped assembled rectangle after layer reversal, with the magenta anchor preserved as an immovable fixed point.
    """
    rect = _assemble(np.asarray(grid, dtype=int))
    return _reverse_layers(rect)


def _place_piece(canvas, piece, top, left):
    h, w = piece.shape
    if np.any(canvas[top : top + h, left : left + w] != 0):
        return False
    canvas[top : top + h, left : left + w] = piece
    return True


def _generate_case(rng: random.Random):
    outer = rng.choice((1, 3, 4, 5, 7))
    inner = rng.choice([c for c in (1, 3, 4, 5, 7) if c != outer])
    center_h = 4
    center_w = 4
    total_h = center_h + 2
    total_w = center_w + 2
    top = np.full((1, total_w), outer, dtype=int)
    bottom = np.full((1, total_w), outer, dtype=int)
    left = np.full((center_h, 1), outer, dtype=int)
    right = np.full((center_h, 1), outer, dtype=int)
    center = np.full((center_h, center_w), inner, dtype=int)
    mr0 = center_h // 2 - 1
    mc0 = center_w // 2 - 1
    center[mr0 : mr0 + 2, mc0 : mc0 + 2] = 6
    top[0, total_w // 2] = 2
    bottom[0, total_w // 2 - 1] = 2
    left[center_h // 2, 0] = 2
    right[center_h // 2 - 1, 0] = 2
    center[0, center_w // 2] = 2
    center[-1, center_w // 2 - 1] = 2
    center[center_h // 2, 0] = 2
    center[center_h // 2 - 1, -1] = 2
    pieces = [top, bottom, left, right, center]
    side = rng.randint(18, 24)
    grid = np.zeros((side, side), dtype=int)
    for piece in pieces:
        ph, pw = piece.shape
        for _ in range(500):
            r0 = rng.randint(0, side - ph)
            c0 = rng.randint(0, side - pw)
            pad_r0 = max(0, r0 - 1)
            pad_c0 = max(0, c0 - 1)
            pad_r1 = min(side, r0 + ph + 1)
            pad_c1 = min(side, c0 + pw + 1)
            if np.any(grid[pad_r0:pad_r1, pad_c0:pad_c1] != 0):
                continue
            grid[r0 : r0 + ph, c0 : c0 + pw] = piece
            break
    return grid, main(grid)


def generate_input() -> np.ndarray:
    return _generate_case(random.Random(random.randrange(1 << 30)))[0]
