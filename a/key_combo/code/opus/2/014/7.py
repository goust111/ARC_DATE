import random
from collections import Counter, defaultdict, deque

import numpy as np


FRAME_COLORS = (1, 2, 3)
MOTIF_COLORS = (4, 7, 5, 8)
MOVE = {4: (0, 1), 7: (-1, 0), 5: (0, -1), 8: (1, 0)}


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


def _frame_infos(grid):
    infos = []
    for cells in _components(np.isin(grid, FRAME_COLORS)):
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        color = int(grid[cells[0]])
        infos.append((color, r0, c0, r1, c1))
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Summary:
    Complete each group of same-colored frames by taking the union of their interior motifs, then move every motif pixel one step according to its color.

    Input Model:
    Hollow frames use the border-color palette, interior motif pixels use the motion palette, and frames with the same border color share one latent complete motif.

    Algorithm:
    Group frames by border color, vote a color for every observed relative motif position within the group, write that complete motif into every frame of the group, then shift the completed motif simultaneously with wrap-around inside the frame interior.

    Output:
    Return the grid with borders unchanged, missing motif pixels restored, and all motif pixels moved to their color-coded destinations.
    """
    grid = np.asarray(grid, dtype=int)
    out = grid.copy()
    groups = defaultdict(list)
    for info in _frame_infos(grid):
        groups[info[0]].append(info)
    for color, frames in groups.items():
        votes = defaultdict(Counter)
        for _, r0, c0, r1, c1 in frames:
            inner = grid[r0 + 1 : r1, c0 + 1 : c1]
            for rr, cc in np.argwhere(inner != 0):
                votes[(int(rr), int(cc))][int(inner[rr, cc])] += 1
        motif = {}
        for pos, counter in votes.items():
            best = sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][0]
            motif[pos] = best
        for _, r0, c0, r1, c1 in frames:
            inner_h = r1 - r0 - 1
            inner_w = c1 - c0 - 1
            out[r0 + 1 : r1, c0 + 1 : c1] = 0
            shifted = {}
            for (rr, cc), mcolor in motif.items():
                dr, dc = MOVE[mcolor]
                nr = (rr + dr) % inner_h
                nc = (cc + dc) % inner_w
                shifted[(nr, nc)] = mcolor
            for (rr, cc), mcolor in shifted.items():
                out[r0 + 1 + rr, c0 + 1 + cc] = mcolor
    return out


def _generate_case(rng: random.Random):
    rows, cols = rng.choice(((2, 3), (3, 3)))
    frame_h = rng.randint(5, 6)
    frame_w = rng.randint(5, 6)
    h = rows * frame_h + (rows - 1)
    w = cols * frame_w + (cols - 1)
    grid = np.zeros((h, w), dtype=int)
    frame_positions = []
    for r in range(rows):
        for c in range(cols):
            r0 = r * (frame_h + 1)
            c0 = c * (frame_w + 1)
            frame_positions.append((r0, c0))
    group_colors = rng.sample(FRAME_COLORS, rng.randint(2, 3))
    assignments = [rng.choice(group_colors) for _ in frame_positions]
    if len(set(assignments)) == 1:
        assignments[0] = group_colors[-1]
    motifs = {}
    for color in set(assignments):
        inner_h = frame_h - 2
        inner_w = frame_w - 2
        while True:
            count = rng.randint(2, min(4, inner_h * inner_w))
            positions = rng.sample([(r, c) for r in range(inner_h) for c in range(inner_w)], count)
            colors = rng.choices(MOTIF_COLORS, k=count)
            shifted = set()
            ok = True
            for (rr, cc), mcolor in zip(positions, colors):
                dr, dc = MOVE[mcolor]
                nr = (rr + dr) % inner_h
                nc = (cc + dc) % inner_w
                if (nr, nc) in shifted:
                    ok = False
                    break
                shifted.add((nr, nc))
            if ok:
                motifs[color] = list(zip(positions, colors))
                break
    incomplete_made = False
    expected = np.zeros_like(grid)
    for (r0, c0), color in zip(frame_positions, assignments):
        grid[r0 : r0 + frame_h, c0 : c0 + frame_w] = 0
        expected[r0 : r0 + frame_h, c0 : c0 + frame_w] = 0
        grid[r0, c0 : c0 + frame_w] = color
        grid[r0 + frame_h - 1, c0 : c0 + frame_w] = color
        grid[r0 : r0 + frame_h, c0] = color
        grid[r0 : r0 + frame_h, c0 + frame_w - 1] = color
        expected[r0, c0 : c0 + frame_w] = color
        expected[r0 + frame_h - 1, c0 : c0 + frame_w] = color
        expected[r0 : r0 + frame_h, c0] = color
        expected[r0 : r0 + frame_h, c0 + frame_w - 1] = color
        motif = motifs[color]
        keep = rng.randint(0, len(motif))
        subset = motif if keep == len(motif) and incomplete_made else rng.sample(motif, keep)
        if len(subset) < len(motif):
            incomplete_made = True
        for (rr, cc), mcolor in subset:
            grid[r0 + 1 + rr, c0 + 1 + cc] = mcolor
        for (rr, cc), mcolor in motif:
            dr, dc = MOVE[mcolor]
            nr = (rr + dr) % (frame_h - 2)
            nc = (cc + dc) % (frame_w - 2)
            expected[r0 + 1 + nr, c0 + 1 + nc] = mcolor
    if not incomplete_made:
        (r0, c0), color = frame_positions[0], assignments[0]
        rr, cc = motifs[color][0][0]
        grid[r0 + 1 + rr, c0 + 1 + cc] = 0
    return grid, main(grid)


def generate_input() -> np.ndarray:
    return _generate_case(random.Random(random.randrange(1 << 30)))[0]
