import random
from collections import deque

import numpy as np


BG = 0
SHAPES = {
    "right": {(1, 0), (1, 1), (0, 2), (1, 2), (2, 2)},
    "left": {(1, 2), (1, 1), (0, 0), (1, 0), (2, 0)},
    "up": {(2, 1), (1, 1), (0, 0), (0, 1), (0, 2)},
    "down": {(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)},
}
TIP_CELLS = {
    "right": {(0, 2), (1, 2), (2, 2)},
    "left": {(0, 0), (1, 0), (2, 0)},
    "up": {(0, 0), (0, 1), (0, 2)},
    "down": {(2, 0), (2, 1), (2, 2)},
}
TAIL_CELLS = {
    "right": {(1, 0), (1, 1)},
    "left": {(1, 1), (1, 2)},
    "up": {(1, 1), (2, 1)},
    "down": {(0, 1), (1, 1)},
}
DIRS = {"right": (0, 1), "left": (0, -1), "up": (-1, 0), "down": (1, 0)}


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


def _parse_arrows(grid: np.ndarray):
    arrows = []
    for comp in _components(grid != BG):
        r0, c0, r1, c1 = _bbox(comp)
        rel = {(r - r0, c - c0) for r, c in comp}
        orient = next(name for name, shape in SHAPES.items() if rel == shape)
        tip_vals = {int(grid[r0 + r, c0 + c]) for r, c in TIP_CELLS[orient]}
        tail_vals = {int(grid[r0 + r, c0 + c]) for r, c in TAIL_CELLS[orient]}
        assert len(tip_vals) == 1 and len(tail_vals) == 1
        arrows.append(
            {
                "bbox": (r0, c0, r1, c1),
                "orient": orient,
                "tip": next(iter(tip_vals)),
                "tail": next(iter(tail_vals)),
                "center": ((r0 + r1) // 2, (c0 + c1) // 2),
            }
        )
    return arrows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the unique arrow-color relay chain and repaint each next arrow's tail with the current arrow's tip color.

    Input structure:
    - Black background containing multiple two-color arrow sprites, each with a tip color and a different tail color.
    - Every arrow points in one cardinal direction.
    - Exactly one arrow has a tail color that never appears as any arrow tip color.

    Rule:
    - Identify that unique arrow as the start of the chain.
    - Repeatedly look in the current arrow's pointing direction and select the nearest arrow centered on that ray.
    - Recolor the selected arrow's tail cells to the current arrow's tip color, then continue from that selected arrow.
    - Stop when no further arrow lies ahead on the current pointing ray.

    Key insight:
    - The relay start is found by color uniqueness, and the recoloring itself propagates forward geometrically along the arrows' pointing directions.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    arrows = _parse_arrows(g)
    tip_colors = {a["tip"] for a in arrows}
    current = next(a for a in arrows if a["tail"] not in tip_colors)
    seen = {(current["bbox"], current["orient"])}
    while True:
        dr, dc = DIRS[current["orient"]]
        cr, cc = current["center"]
        candidates = []
        for arrow in arrows:
            if arrow is current:
                continue
            ar, ac = arrow["center"]
            if dr != 0 and ac == cc and (ar - cr) * dr > 0:
                candidates.append((abs(ar - cr), arrow))
            if dc != 0 and ar == cr and (ac - cc) * dc > 0:
                candidates.append((abs(ac - cc), arrow))
        if not candidates:
            break
        target = min(candidates, key=lambda item: item[0])[1]
        target_id = (target["bbox"], target["orient"])
        if target_id in seen:
            break
        r0, c0, _r1, _c1 = target["bbox"]
        for r, c in TAIL_CELLS[target["orient"]]:
            out[r0 + r, c0 + c] = current["tip"]
        seen.add(target_id)
        current = target
    return out


def _place_arrow(grid, orient, top, left, tip, tail):
    for r, c in SHAPES[orient]:
        grid[top + r, left + c] = tip if (r, c) in TIP_CELLS[orient] else tail


def _free_exit_orients(centers):
    last_r, last_c = centers[-1]
    free = []
    for orient, (dr, dc) in DIRS.items():
        blocked = False
        for r, c in centers[:-1]:
            if dr != 0 and c == last_c and (r - last_r) * dr > 0:
                blocked = True
            if dc != 0 and r == last_r and (c - last_c) * dc > 0:
                blocked = True
        if not blocked:
            free.append(orient)
    return free


def generate_input() -> np.ndarray:
    for _ in range(500):
        h = random.randint(14, 26)
        w = random.randint(14, 26)
        grid = np.zeros((h, w), dtype=int)
        chain_len = random.randint(3, 5)
        arrows = []
        tip_colors = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], chain_len)
        start_tail = random.choice([c for c in range(1, 10) if c not in tip_colors])
        tail_colors = [start_tail] + tip_colors[:-1]
        orient_choices = []
        for _k in range(chain_len - 1):
            orient_choices.append(random.choice(["right", "down"]))
        orient_choices.append(random.choice(["right", "down", "left", "up"]))
        r = random.randint(1, h // 3)
        c = random.randint(1, w // 3)
        centers = []
        for orient in orient_choices:
            centers.append((r + 1, c + 1))
            dr, dc = DIRS[orient]
            r += dr * random.randint(4, 6)
            c += dc * random.randint(4, 6)
            if not (1 <= r <= h - 4 and 1 <= c <= w - 4):
                break
        if len(centers) != chain_len:
            continue
        free = _free_exit_orients(centers)
        if not free:
            continue
        orient_choices[-1] = random.choice(free)
        ok = True
        for idx, center in enumerate(centers):
            top = center[0] - 1
            left = center[1] - 1
            if top < 0 or left < 0 or top + 3 > h or left + 3 > w:
                ok = False
                break
            cells = {(top + rr, left + cc) for rr, cc in SHAPES[orient_choices[idx]]}
            if any(grid[cell] != BG for cell in cells):
                ok = False
                break
            _place_arrow(grid, orient_choices[idx], top, left, tip_colors[idx], tail_colors[idx])
            arrows.append((top, left))
        if ok:
            return grid
    raise RuntimeError("failed to generate valid input")
