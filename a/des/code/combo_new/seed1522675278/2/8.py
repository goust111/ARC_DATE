import random

import numpy as np


BLACK = 0
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
ARMS = {
    "up": {(1, 0), (1, 1), (1, 2), (2, 1)},
    "down": {(0, 1), (1, 0), (1, 1), (1, 2)},
    "left": {(0, 1), (1, 0), (1, 1), (2, 1)},
    "right": {(0, 1), (1, 1), (1, 2), (2, 1)},
}
STEP = {"up": (-3, 0), "down": (3, 0), "left": (0, -3), "right": (0, 3)}


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(pts)
    return out


def _kind(comp):
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    r0, c0 = min(rs), min(cs)
    rel = {(r - r0, c - c0) for r, c in comp}
    for k, v in ARMS.items():
        if rel == v:
            return k
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Gather same-colored arrows next to their 2x2 anchor, preserving arrow direction and packing them tightly along that direction.

    Input structure:
    - Black background with one 2x2 solid anchor per color.
    - Each color also has several same-colored arrow components using a 3x3 T-shape.
    - All arrows of a color point in the same direction.

    Rule:
    - Count arrows per color and infer their common direction.
    - Remove the scattered arrows from their original positions.
    - Starting adjacent to the anchor on the pointing side, place the arrows in a tight queue one bounding-box step apart.

    Key insight:
    - The arrow shape encodes both identity and queue direction, while the anchor defines the queue origin.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    by_color = {}
    for comp in _components(g != BLACK):
        color = int(g[comp[0]])
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        if len(comp) == 4 and max(rs) - min(rs) <= 2 and max(cs) - min(cs) <= 2:
            by_color.setdefault(color, {"anchor": None, "arrows": []})
            by_color[color]["arrows"].append(comp)
        elif len(comp) == 4 and max(rs) - min(rs) == 1 and max(cs) - min(cs) == 1:
            by_color.setdefault(color, {"anchor": None, "arrows": []})
            by_color[color]["anchor"] = (min(rs), min(cs))
    for color, item in by_color.items():
        if not item["anchor"] or not item["arrows"]:
            continue
        kind = _kind(item["arrows"][0])
        for comp in item["arrows"]:
            for r, c in comp:
                out[r, c] = BLACK
        ar, ac = item["anchor"]
        sr, sc = {"up": (ar - 3, ac), "down": (ar + 2, ac), "left": (ar, ac - 3), "right": (ar, ac + 2)}[kind]
        dr, dc = STEP[kind]
        for idx in range(len(item["arrows"])):
            for rr, cc in ARMS[kind]:
                out[sr + dr * idx + rr, sc + dc * idx + cc] = color
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((24, 24), dtype=int)
    for color, kind, anchor in zip(random.sample(COLORS, 2), random.sample(list(ARMS), 2), [(4, 4), (16, 16)]):
        ar, ac = anchor
        g[ar : ar + 2, ac : ac + 2] = color
        for idx in range(random.randint(1, 3)):
            rr, cc = random.randint(3, 18), random.randint(3, 18)
            for dr, dc in ARMS[kind]:
                g[rr + dr, cc + dc] = color
    return g
