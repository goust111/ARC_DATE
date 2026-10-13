from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 500

ARROW_PATTERNS = {
    frozenset({(0, 0), (1, 0), (1, 1)}): (-1, 0),
    frozenset({(0, 0), (0, 1), (1, 1)}): (0, 1),
    frozenset({(0, 1), (1, 0), (1, 1)}): (1, 0),
    frozenset({(0, 0), (0, 1), (1, 0)}): (0, -1),
}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            parts.append(coords)
    return parts


def _decode_arrow(coords: list[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, c0 = min(rows), min(cols)
    norm = frozenset((r - r0, c - c0) for r, c in coords)
    return ARROW_PATTERNS[norm]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each arrow-marked object by the spike count of its same-colored top ruler.

    Input structure:
    - The background is black.
    - Rows 0 and 1 form a top ruler band: each object color has a horizontal ruler segment on row 1 and same-colored spike pixels immediately above it on row 0.
    - Below the ruler band sit several disjoint interior objects, each made of one non-grey color with an embedded three-pixel grey arrow marker.
    - Different object colors correspond one-to-one with different rulers.

    Rule:
    - For each ruler color, count how many same-colored spike pixels appear on row 0 above the row-1 base segment; that count is the movement distance.
    - For each interior object, read the direction encoded by its grey arrow pattern and identify the object's non-grey body color.
    - Process objects in order of their original top row then left column, clearing each source position and translating the entire object (including grey arrow pixels) by the decoded distance along the decoded direction.
    - Leave the ruler band unchanged.

    Key insight:
    - The transformation is split across two separate cues: direction is local inside the object, while distance must be decoded remotely from the matching same-colored border ruler.
    """
    g = np.asarray(grid, dtype=int)
    move_by_color: dict[int, int] = {}
    for color in [int(v) for v in np.unique(g[:2, :]) if int(v) not in (BG, GREY)]:
        base_cols = [c for c in range(g.shape[1]) if int(g[1, c]) == color]
        if base_cols:
            move_by_color[color] = sum(1 for c in base_cols if int(g[0, c]) == color)

    mask = np.zeros(g.shape, dtype=bool)
    mask[2:, :] = g[2:, :] != BG
    comps = _components(mask)
    objects = []
    for comp in comps:
        colors = {int(g[r, c]) for r, c in comp if int(g[r, c]) != GREY}
        if len(colors) != 1:
            continue
        color = next(iter(colors))
        arrow = [(r, c) for r, c in comp if int(g[r, c]) == GREY]
        if len(arrow) != 3:
            continue
        direction = _decode_arrow(arrow)
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        objects.append(
            {
                "coords": comp,
                "color": color,
                "direction": direction,
                "top": min(rows),
                "left": min(cols),
            }
        )

    objects.sort(key=lambda obj: (int(obj["top"]), int(obj["left"])))
    out = g.copy()
    for obj in objects:
        for r, c in obj["coords"]:
            out[r, c] = BG

    for obj in objects:
        dr, dc = obj["direction"]
        steps = move_by_color[int(obj["color"])]
        for r, c in obj["coords"]:
            out[r + dr * steps, c + dc * steps] = int(g[r, c])
    return out


def _make_patch(color: int, direction: tuple[int, int], rng: random.Random) -> np.ndarray:
    if direction == (-1, 0):
        arrow = np.array([[GREY, BG], [GREY, GREY]], dtype=int)
    elif direction == (1, 0):
        arrow = np.array([[BG, GREY], [GREY, GREY]], dtype=int)
    elif direction == (0, -1):
        arrow = np.array([[GREY, GREY], [GREY, BG]], dtype=int)
    else:
        arrow = np.array([[GREY, GREY], [BG, GREY]], dtype=int)
    ph, pw = rng.choice([(2, 2), (2, 3), (3, 2), (2, 4), (4, 2)])
    patch = np.full((ph, pw), color, dtype=int)
    patch[:2, :2] = np.where(arrow == BG, color, arrow)
    return patch


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        obj_count = rng.randint(2, 4)
        h = rng.randint(12, 20)
        w = rng.randint(14, 24)
        if w < obj_count * 4 + 2:
            continue
        grid = np.zeros((h, w), dtype=int)
        colors = rng.sample(COLORS, obj_count)
        steps_by_color = {color: rng.randint(1, 5) for color in colors}

        col = 1
        ok = True
        for color in colors:
            seg_min = max(3, steps_by_color[color])
            seg_max = min(7, w - col - (obj_count - 1))
            if seg_max < seg_min:
                ok = False
                break
            seg_w = rng.randint(seg_min, seg_max)
            if col + seg_w > w - 1:
                ok = False
                break
            grid[1, col : col + seg_w] = color
            spike_cols = rng.sample(list(range(col, col + seg_w)), steps_by_color[color])
            for sc in spike_cols:
                grid[0, sc] = color
            col += seg_w + rng.randint(1, 2)
        if not ok:
            continue

        source_occ = np.zeros((h, w), dtype=bool)
        target_occ = np.zeros((h, w), dtype=bool)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        rng.shuffle(directions)

        for color in colors:
            direction = directions.pop() if directions else rng.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
            patch = _make_patch(color, direction, rng)
            ph, pw = map(int, patch.shape)
            steps = steps_by_color[color]
            placed = False
            for _try in range(200):
                src_top = rng.randint(3, h - ph - 1)
                src_left = rng.randint(0, w - pw)
                dst_top = src_top + direction[0] * steps
                dst_left = src_left + direction[1] * steps
                if not (3 <= dst_top <= h - ph and 0 <= dst_left <= w - pw):
                    continue
                src_rs = max(0, src_top - 1)
                src_re = min(h, src_top + ph + 1)
                src_cs = max(0, src_left - 1)
                src_ce = min(w, src_left + pw + 1)
                dst_rs = max(0, dst_top - 1)
                dst_re = min(h, dst_top + ph + 1)
                dst_cs = max(0, dst_left - 1)
                dst_ce = min(w, dst_left + pw + 1)
                if source_occ[src_rs:src_re, src_cs:src_ce].any():
                    continue
                if target_occ[dst_rs:dst_re, dst_cs:dst_ce].any():
                    continue
                source_occ[src_top : src_top + ph, src_left : src_left + pw] = patch != BG
                target_occ[dst_top : dst_top + ph, dst_left : dst_left + pw] = patch != BG
                for rr in range(ph):
                    for cc in range(pw):
                        if patch[rr, cc] != BG:
                            grid[src_top + rr, src_left + cc] = int(patch[rr, cc])
                placed = True
                break
            if not placed:
                ok = False
                break

        if ok:
            try:
                out = main(grid)
            except Exception:
                continue
            if out.ndim == 2 and max(out.shape) <= 30:
                return grid

    raise RuntimeError("failed to generate input")
