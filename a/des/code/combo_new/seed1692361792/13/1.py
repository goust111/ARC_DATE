import random
from collections import deque

import numpy as np


BG = 0
FRAME = 5


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
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
            out.append(comp)
    return out


def _bbox(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(cells):
    r0, c0, _, _ = _bbox(cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _flip_shape(cells, mode: str):
    r0, c0, r1, c1 = _bbox(cells)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    rel = [(r - r0, c - c0) for r, c in cells]
    if mode == "ud":
        return frozenset((h - 1 - r, c) for r, c in rel)
    return frozenset((r, w - 1 - c) for r, c in rel)


def _find_frames(grid: np.ndarray):
    frames = []
    for comp in _components(grid == FRAME):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 != c1 - c0 or r1 - r0 < 5:
            continue
        border = []
        missing = []
        for c in range(c0, c1 + 1):
            border.append((r0, c))
            border.append((r1, c))
        for r in range(r0 + 1, r1):
            border.append((r, c0))
            border.append((r, c1))
        for cell in border:
            if grid[cell] != FRAME:
                missing.append(cell)
        if len(missing) != 1:
            continue
        mr, mc = missing[0]
        if mr == r0:
            side = "top"
        elif mr == r1:
            side = "bottom"
        elif mc == c0:
            side = "left"
        else:
            side = "right"
        frames.append((r0, c0, r1, c1, side))
    return frames


def _missing_cells(inner: np.ndarray, side: str):
    color_cells = {(r, c) for r, c in zip(*np.where(inner != BG), strict=False)}
    h, w = inner.shape
    missing = set()
    if side in {"top", "bottom"}:
        for r, c in list(color_cells):
            mr = h - 1 - r
            if side == "top" and mr < r and (mr, c) not in color_cells:
                missing.add((mr, c))
            if side == "bottom" and mr > r and (mr, c) not in color_cells:
                missing.add((mr, c))
        flip = "ud"
    else:
        for r, c in list(color_cells):
            mc = w - 1 - c
            if side == "left" and mc < c and (r, mc) not in color_cells:
                missing.add((r, mc))
            if side == "right" and mc > c and (r, mc) not in color_cells:
                missing.add((r, mc))
        flip = "lr"
    return missing, flip


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each gapped frame's asymmetric side with the matching outside fragment after mirroring it across the hinted symmetry axis.

    Input structure:
    - Black background with one or more grey square frames; each frame border has exactly one one-cell gap on one side.
    - Inside every frame is a single-color pattern that is almost symmetric, except the side indicated by the gap is missing a connected fragment.
    - Outside the frames are small same-color fragments scattered on the background.

    Rule:
    - Detect each frame and which side contains the border gap.
    - Treat the frame interior as symmetric across the axis perpendicular to that gap direction, and infer the missing interior cells on the gapped side.
    - Find the outside fragment of the same color whose mirrored shape matches that missing set, erase it from the background, and place the mirrored fragment into the missing interior cells.
    - Leave the grey frame, including its gap, unchanged.

    Key insight:
    - The border gap identifies both which side is incomplete and which mirror axis to use when turning the outside fragment into the missing interior piece.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    frames = _find_frames(g)
    frame_mask = np.zeros_like(g, dtype=bool)
    for r0, c0, r1, c1, _side in frames:
        frame_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    for r0, c0, r1, c1, side in frames:
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        colors = sorted(set(map(int, inner.ravel())) - {BG})
        assert len(colors) == 1
        color = colors[0]
        missing, flip = _missing_cells(inner, side)
        assert missing
        target = _normalize([(r0 + 1 + r, c0 + 1 + c) for r, c in missing])
        chosen = None
        for comp in _components((g == color) & (~frame_mask)):
            if _flip_shape(comp, flip) == target:
                chosen = comp
                break
        assert chosen is not None
        for rr, cc in chosen:
            out[rr, cc] = BG
        for rr, cc in missing:
            out[r0 + 1 + rr, c0 + 1 + cc] = color
    return out


def _random_blob(cells, size):
    cells = list(cells)
    start = random.choice(cells)
    blob = {start}
    frontier = {nb for nb in cells if abs(nb[0] - start[0]) + abs(nb[1] - start[1]) == 1}
    while len(blob) < size:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        blob.add(cell)
        for nb in cells:
            if nb not in blob and abs(nb[0] - cell[0]) + abs(nb[1] - cell[1]) == 1:
                frontier.add(nb)
    return blob


def _place_blob(grid, shape, color, forbidden):
    h, w = grid.shape
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    sh = max(rows) + 1
    sw = max(cols) + 1
    options = []
    for r0 in range(1, h - sh):
        for c0 in range(1, w - sw):
            cells = {(r0 + r, c0 + c) for r, c in shape}
            if any(cell in forbidden for cell in cells):
                continue
            if any(grid[r, c] != BG for r, c in cells):
                continue
            options.append((r0, c0))
    if not options:
        return False
    r0, c0 = random.choice(options)
    for r, c in shape:
        grid[r0 + r, c0 + c] = color
    forbidden.update((r0 + r, c0 + c) for r, c in shape)
    return True


def generate_input() -> np.ndarray:
    for _ in range(500):
        h = random.randint(18, 28)
        w = random.randint(18, 28)
        grid = np.zeros((h, w), dtype=int)
        n = random.randint(1, 3)
        colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], n)
        frames = []
        occupied = set()
        ok = True
        for color in colors:
            size = random.choice([6, 8])
            placed = False
            for _attempt in range(80):
                r0 = random.randint(1, h - size - 2)
                c0 = random.randint(1, w - size - 2)
                cells = {(r, c) for r in range(r0 - 1, r0 + size + 1) for c in range(c0 - 1, c0 + size + 1)}
                if cells & occupied:
                    continue
                occupied |= cells
                frames.append((r0, c0, size, color))
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        forbidden = set(occupied)
        for r0, c0, size, color in frames:
            r1 = r0 + size - 1
            c1 = c0 + size - 1
            grid[r0 : r1 + 1, c0] = FRAME
            grid[r0 : r1 + 1, c1] = FRAME
            grid[r0, c0 : c1 + 1] = FRAME
            grid[r1, c0 : c1 + 1] = FRAME
            side = random.choice(["top", "bottom", "left", "right"])
            if side in {"top", "bottom"}:
                gap_c = random.randint(c0 + 1, c1 - 1)
                grid[r0 if side == "top" else r1, gap_c] = BG
                half = {(r, c) for r in range((size - 2) // 2) for c in range(size - 2)} if side == "top" else {(r, c) for r in range((size - 2) // 2, size - 2) for c in range(size - 2)}
                flip = lambda rc: (size - 3 - rc[0], rc[1])
            else:
                gap_r = random.randint(r0 + 1, r1 - 1)
                grid[gap_r, c0 if side == "left" else c1] = BG
                half = {(r, c) for r in range(size - 2) for c in range((size - 2) // 2)} if side == "left" else {(r, c) for r in range(size - 2) for c in range((size - 2) // 2, size - 2)}
                flip = lambda rc: (rc[0], size - 3 - rc[1])

            miss = None
            for _attempt in range(80):
                blob = _random_blob(half, random.randint(2, 5))
                if blob is not None:
                    miss = blob
                    break
            if miss is None:
                ok = False
                break

            pattern = {flip(cell) for cell in miss}
            extras = random.randint(2, 6)
            all_cells = [(r, c) for r in range(size - 2) for c in range(size - 2)]
            for _attempt in range(200):
                cell = random.choice(all_cells)
                mate = flip(cell)
                if cell in miss or mate in miss:
                    continue
                pattern.add(cell)
                pattern.add(mate)
                if len(pattern) >= len(miss) + extras:
                    break

            for rr, cc in pattern:
                grid[r0 + 1 + rr, c0 + 1 + cc] = color

            fragment = _normalize([flip(cell) for cell in miss])
            if not _place_blob(grid, fragment, color, forbidden):
                ok = False
                break
        if ok:
            return grid
    raise RuntimeError("failed to generate valid input")
