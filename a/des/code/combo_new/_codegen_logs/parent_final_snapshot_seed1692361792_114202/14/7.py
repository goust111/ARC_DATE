import random
from collections import deque

import numpy as np

AZURE = 8
PALETTE = [1, 2, 3, 4, 5, 6, 7, 9]


def _components(mask: np.ndarray):
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


def _bbox(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), max(rows), min(cols), max(cols)


def _background_components(box_mask):
    h, w = box_mask.shape
    seen = np.zeros_like(box_mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if box_mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            touches_edge = False
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                if rr in (0, h - 1) or cc in (0, w - 1):
                    touches_edge = True
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not box_mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append((comp, touches_edge))
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each ring's enclosed azure hole shape at the target reached by its same-color guide line.

    Input structure:
    - Azure background with several monochrome closed ring outlines, each extended by one straight guide line
      of the same color.
    - The enclosed azure region inside a ring is the hidden source shape.
    - The guide line ends in open background, so the next cell beyond the free endpoint is the stamp anchor.

    Rule:
    - For each connected color component, recover the enclosed azure hole shape inside its bounding box.
    - Trace the unique straight guide-line branch to its free endpoint and extend one more cell in the same
      direction to get the target anchor.
    - Copy the hole shape using the ring's color with the hole bounding box aligned to that anchor.
    - Keep the original ring and guide line unchanged.

    Key insight:
    - The visible ring is only a container; the output shape comes from the ring's negative space, while the
    guide line contributes only the placement anchor.
    """
    h, w = grid.shape
    out = grid.copy()
    for color in np.unique(grid):
        if color == AZURE:
            continue
        for comp in _components(grid == color):
            r0, r1, c0, c1 = _bbox(comp)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in comp:
                mask[rr - r0, cc - c0] = True
            holes = [cells for cells, touches in _background_components(mask) if not touches]
            if not holes:
                continue
            hole = max(holes, key=len)
            hr0, hr1, hc0, hc1 = _bbox(hole)
            rel_hole = [(r - hr0, c - hc0) for r, c in hole]

            comp_set = set(comp)
            degree = {}
            for rr, cc in comp:
                count = 0
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    if (rr + dr, cc + dc) in comp_set:
                        count += 1
                degree[(rr, cc)] = count
            endpoints = [cell for cell, deg in degree.items() if deg == 1]
            if not endpoints:
                continue
            endpoint = endpoints[0]
            path = [endpoint]
            prev = None
            cur = endpoint
            while True:
                nbrs = [
                    (cur[0] + dr, cur[1] + dc)
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                    if (cur[0] + dr, cur[1] + dc) in comp_set and (cur[0] + dr, cur[1] + dc) != prev
                ]
                if len(nbrs) != 1 or degree[cur] != 2:
                    break
                prev, cur = cur, nbrs[0]
                path.append(cur)
            if len(path) < 2:
                continue
            er, ec = path[0]
            sr, sc = path[1]
            dr, dc = er - sr, ec - sc
            anchor_r, anchor_c = er + dr, ec + dc
            for rr, cc in rel_hole:
                tr, tc = anchor_r + rr, anchor_c + cc
                if 0 <= tr < h and 0 <= tc < w and out[tr, tc] == AZURE:
                    out[tr, tc] = color
    return out


def _connected_hole(rng):
    hh = rng.randint(2, 4)
    hw = rng.randint(2, 4)
    target = rng.randint(max(2, hh * hw // 2), hh * hw)
    cells = {(rng.randrange(hh), rng.randrange(hw))}
    frontier = list(cells)
    while len(cells) < target:
        r, c = rng.choice(frontier)
        nbrs = [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
        rng.shuffle(nbrs)
        for nr, nc in nbrs:
            if 0 <= nr < hh and 0 <= nc < hw and (nr, nc) not in cells:
                cells.add((nr, nc))
                frontier.append((nr, nc))
                break
        else:
            frontier.append((rng.randrange(hh), rng.randrange(hw)))
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    norm = {(r - r0, c - c0) for r, c in cells}
    return norm, r1 - r0 + 1, c1 - c0 + 1


def _ring_from_hole(hole, hh, hw):
    shifted = {(r + 1, c + 1) for r, c in hole}
    ring = set()
    for r, c in shifted:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nbr = (r + dr, c + dc)
            if nbr not in shifted:
                ring.add(nbr)
    rows = [r for r, _ in ring]
    cols = [c for _, c in ring]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    norm = {(r - r0, c - c0) for r, c in ring}
    return norm, r1 - r0 + 1, c1 - c0 + 1


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(500):
        h = rng.randint(20, 28)
        w = rng.randint(20, 28)
        grid = np.full((h, w), AZURE, dtype=int)
        used = np.zeros((h, w), dtype=bool)
        num_objs = rng.randint(1, 3)
        colors = rng.sample(PALETTE, num_objs)
        ok = True

        for color in colors:
            hole, hh, hw = _connected_hole(rng)
            ring, rh, rw = _ring_from_hole(hole, hh, hw)
            placed = False
            for _ in range(400):
                top = rng.randint(1, h - rh - 2)
                left = rng.randint(1, w - rw - 2)
                direction = rng.choice(["up", "down", "left", "right"])

                if direction in ("up", "down"):
                    edge_cells = [(r, c) for r, c in ring if (r == 0 if direction == "up" else r == rh - 1)]
                else:
                    edge_cells = [(r, c) for r, c in ring if (c == 0 if direction == "left" else c == rw - 1)]
                attach_r, attach_c = rng.choice(edge_cells)
                line_len = rng.randint(2, 5)

                if direction == "right":
                    anchor_r = top + attach_r
                    anchor_c = left + attach_c + line_len + 1
                    line = [(top + attach_r, left + attach_c + i) for i in range(1, line_len + 1)]
                elif direction == "left":
                    anchor_r = top + attach_r
                    anchor_c = left + attach_c - line_len - 1
                    line = [(top + attach_r, left + attach_c - i) for i in range(1, line_len + 1)]
                elif direction == "down":
                    anchor_r = top + attach_r + line_len + 1
                    anchor_c = left + attach_c
                    line = [(top + attach_r + i, left + attach_c) for i in range(1, line_len + 1)]
                else:
                    anchor_r = top + attach_r - line_len - 1
                    anchor_c = left + attach_c
                    line = [(top + attach_r - i, left + attach_c) for i in range(1, line_len + 1)]

                if not (0 <= anchor_r <= h - hh and 0 <= anchor_c <= w - hw):
                    continue
                ring_cells = [(top + r, left + c) for r, c in ring]
                stamp_box = [(anchor_r + r, anchor_c + c) for r in range(hh) for c in range(hw)]
                all_cells = ring_cells + line + stamp_box
                if any(not (0 <= r < h and 0 <= c < w) for r, c in all_cells):
                    continue
                rr0 = max(0, min(r for r, _ in all_cells) - 1)
                rr1 = min(h, max(r for r, _ in all_cells) + 2)
                cc0 = max(0, min(c for _, c in all_cells) - 1)
                cc1 = min(w, max(c for _, c in all_cells) + 2)
                if used[rr0:rr1, cc0:cc1].any():
                    continue
                for r, c in ring_cells + line:
                    grid[r, c] = color
                used[rr0:rr1, cc0:cc1] = True
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return grid
    raise RuntimeError("failed to generate input")
