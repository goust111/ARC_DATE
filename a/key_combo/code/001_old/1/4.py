from __future__ import annotations

import random

import numpy as np


def _frame_bbox_for_color(g: np.ndarray, color: int) -> tuple[int, int, int, int] | None:
    pts = np.argwhere(g == color)
    if pts.size == 0:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None
    expected: set[tuple[int, int]] = set()
    for c in range(int(c0), int(c1) + 1):
        expected.add((int(r0), c))
        expected.add((int(r1), c))
    for r in range(int(r0), int(r1) + 1):
        expected.add((r, int(c0)))
        expected.add((r, int(c1)))
    actual = {(int(r), int(c)) for r, c in pts}
    if actual != expected:
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _crop_nonzero(block: np.ndarray) -> np.ndarray | None:
    pts = np.argwhere(block != 0)
    if pts.size == 0:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return block[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the largest complete frame as a blank canvas and stamp smaller frame interiors into relative corners.

    Input structure:
    - A black canvas contains several complete one-cell-thick rectangular frames in distinct colors.
    - The frame with the largest interior area has an empty interior.
    - Other frames contain small colored interior patterns, and sparse isolated noise pixels may also appear.

    Rule:
    - Select the complete frame with largest interior area and output its interior.
    - For every other complete frame, crop the non-black interior pattern.
    - Paste that crop into the output corner matching the donor frame's diagonal position relative to the large frame center.

    Key insight:
    The largest frame defines both the output window and the coordinate origin used to route every donor stamp.
    """
    g = np.asarray(grid, dtype=int)
    frames: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != 0):
        bbox = _frame_bbox_for_color(g, color)
        if bbox is not None:
            r0, c0, r1, c1 = bbox
            area = (r1 - r0 - 1) * (c1 - c0 - 1)
            frames.append((area, bbox))
    if not frames:
        raise ValueError("no complete frames found")
    _area, big = max(frames, key=lambda item: item[0])
    br0, bc0, br1, bc1 = big
    out_h, out_w = br1 - br0 - 1, bc1 - bc0 - 1
    out = np.zeros((out_h, out_w), dtype=int)
    big_center = ((br0 + br1) / 2.0, (bc0 + bc1) / 2.0)

    for _area, bbox in frames:
        if bbox == big:
            continue
        r0, c0, r1, c1 = bbox
        crop = _crop_nonzero(g[r0 + 1 : r1, c0 + 1 : c1])
        if crop is None:
            continue
        ch, cw = crop.shape
        center = ((r0 + r1) / 2.0, (c0 + c1) / 2.0)
        top = center[0] < big_center[0]
        left = center[1] < big_center[1]
        rr = 0 if top else out_h - ch
        cc = 0 if left else out_w - cw
        mask = crop != 0
        out[rr : rr + ch, cc : cc + cw][mask] = crop[mask]
    return out


def _draw_frame(g: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    g[r0, c0 : c0 + w] = color
    g[r0 + h - 1, c0 : c0 + w] = color
    g[r0 : r0 + h, c0] = color
    g[r0 : r0 + h, c0 + w - 1] = color


def _can_place(g: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    rows, cols = g.shape
    if r0 < 0 or c0 < 0 or r0 + h > rows or c0 + w > cols:
        return False
    rr0, cc0 = max(0, r0 - 1), max(0, c0 - 1)
    rr1, cc1 = min(rows, r0 + h + 1), min(cols, c0 + w + 1)
    return not np.any(g[rr0:rr1, cc0:cc1] != 0)


def generate_input() -> np.ndarray:
    for _attempt in range(200):
        n = 18
        g = np.zeros((n, n), dtype=int)
        colors = random.sample(range(1, 10), 9)
        border_colors = colors[:5]
        pattern_palette = colors[5:]
        big_inner = 6
        big_h = big_inner + 2
        big_w = big_inner + 2
        br0 = 5
        bc0 = 5
        _draw_frame(g, br0, bc0, big_h, big_w, border_colors[0])

        quadrants = random.sample(["tl", "tr", "bl", "br"], random.randint(2, 4))
        color_i = 1
        failed = False
        for q in quadrants:
            dh = random.choice([4, 5])
            dw = random.choice([4, 5])
            if q == "tl":
                r0_range = range(0, 2)
                c0_range = range(0, 2)
            elif q == "tr":
                r0_range = range(0, 2)
                c0_range = range(n - dw - 1, n - dw + 1)
            elif q == "bl":
                r0_range = range(n - dh - 1, n - dh + 1)
                c0_range = range(0, 2)
            else:
                r0_range = range(n - dh - 1, n - dh + 1)
                c0_range = range(n - dw - 1, n - dw + 1)
            positions = [(r, c) for r in r0_range for c in c0_range]
            random.shuffle(positions)
            placed = False
            for r0, c0 in positions:
                if _can_place(g, r0, c0, dh, dw):
                    border_color = border_colors[color_i]
                    color_i += 1
                    _draw_frame(g, r0, c0, dh, dw, border_color)
                    interior = [(r, c) for r in range(r0 + 1, r0 + dh - 1) for c in range(c0 + 1, c0 + dw - 1)]
                    start = random.choice(interior)
                    pattern = {start}
                    target_size = random.randint(2, min(5, len(interior)))
                    while len(pattern) < target_size:
                        pr, pc = random.choice(list(pattern))
                        nbrs = [(pr + 1, pc), (pr - 1, pc), (pr, pc + 1), (pr, pc - 1)]
                        nbrs = [p for p in nbrs if p in interior]
                        if nbrs:
                            pattern.add(random.choice(nbrs))
                    pat_colors = random.sample(pattern_palette, random.randint(1, 2))
                    for pr, pc in pattern:
                        g[pr, pc] = random.choice(pat_colors)
                    placed = True
                    break
            if not placed:
                failed = True
                break
        if failed:
            continue

        for _ in range(random.randint(4, 8)):
            for _try in range(80):
                r = random.randrange(n)
                c = random.randrange(n)
                if g[r, c] != 0:
                    continue
                rr0, cc0 = max(0, r - 1), max(0, c - 1)
                rr1, cc1 = min(n, r + 2), min(n, c + 2)
                if not np.any(g[rr0:rr1, cc0:cc1] != 0):
                    g[r, c] = random.choice(pattern_palette)
                    break
        return g
    raise RuntimeError("failed to generate puzzle input")
