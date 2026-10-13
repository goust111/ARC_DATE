from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BG = 0
GEN_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _line_info(comp: list[tuple[int, int]]) -> tuple[str, int, int, int]:
    rows = {r for r, _ in comp}
    cols = {c for _, c in comp}
    if len(rows) == 1:
        row = next(iter(rows))
        c0 = min(cols)
        c1 = max(cols)
        assert len(comp) == c1 - c0 + 1
        return ("h", row, c0, c1)
    col = next(iter(cols))
    r0 = min(rows)
    r1 = max(rows)
    assert len(comp) == r1 - r0 + 1
    return ("v", col, r0, r1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each channel object through its nearer matching gate and mirror it across that gate line.

    Input structure:
    - The background is black.
    - Each gate color forms exactly two parallel short line segments, defining a horizontal or vertical channel between them.
    - One or two disconnected objects of some other color lie inside each channel.

    Rule:
    - Pair the two same-colored gate segments and determine whether they form a left-right channel or a top-bottom channel.
    - For every object inside that channel, compare its center distance to the two gates and choose the nearer gate.
    - Remove the object from the channel interior and redraw it outside the chosen gate by mirroring all of its pixels across the gate line.
    - Keep the gate segments unchanged and leave the cleared channel interior black.

    Key insight:
    - The gate pair defines both the valid source region and the transformation: exiting through the nearer gate is not a simple translation, but a reflection across the selected gate segment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    out = g.copy()

    comps_by_color: dict[int, list[list[tuple[int, int]]]] = defaultdict(list)
    for color in (int(v) for v in np.unique(g) if int(v) != BG):
        for comp in _components(g == color):
            comps_by_color[color].append(comp)

    gate_infos = []
    gate_cells: set[tuple[int, int]] = set()
    for color, comps in comps_by_color.items():
        if len(comps) == 2:
            try:
                info1 = _line_info(comps[0])
                info2 = _line_info(comps[1])
            except AssertionError:
                continue
            if info1[0] != info2[0]:
                continue
            gate_infos.append((color, info1, info2))
            gate_cells.update(comps[0])
            gate_cells.update(comps[1])

    object_mask = (g != BG).copy()
    for r, c in gate_cells:
        object_mask[r, c] = False

    object_comps = _components(object_mask)

    for _gate_color, info1, info2 in gate_infos:
        if info1[0] == "v":
            c_left, c_right = sorted([info1[1], info2[1]])
            r0 = min(info1[2], info2[2])
            r1 = max(info1[3], info2[3])
            channel = {(r, c) for r in range(r0, r1 + 1) for c in range(c_left + 1, c_right)}
            channel_objects = [comp for comp in object_comps if set(comp) <= channel]
            for comp in channel_objects:
                for r, c in comp:
                    out[r, c] = BG
                mean_c = sum(c for _, c in comp) / len(comp)
                gate_c = c_left if abs(mean_c - c_left) <= abs(mean_c - c_right) else c_right
                for r, c in comp:
                    nc = 2 * gate_c - c
                    if 0 <= nc < w:
                        out[r, nc] = int(g[r, c])
        else:
            r_top, r_bottom = sorted([info1[1], info2[1]])
            c0 = min(info1[2], info2[2])
            c1 = max(info1[3], info2[3])
            channel = {(r, c) for r in range(r_top + 1, r_bottom) for c in range(c0, c1 + 1)}
            channel_objects = [comp for comp in object_comps if set(comp) <= channel]
            for comp in channel_objects:
                for r, c in comp:
                    out[r, c] = BG
                mean_r = sum(r for r, _ in comp) / len(comp)
                gate_r = r_top if abs(mean_r - r_top) <= abs(mean_r - r_bottom) else r_bottom
                for r, c in comp:
                    nr = 2 * gate_r - r
                    if 0 <= nr < h:
                        out[nr, c] = int(g[r, c])

    return out


def _random_object() -> np.ndarray:
    h = random.randint(2, 4)
    w = random.randint(2, 4)
    mask = np.zeros((h, w), dtype=bool)
    r, c = random.randrange(h), random.randrange(w)
    mask[r, c] = True
    target = random.randint(2, min(h * w - 1, 5))
    while int(mask.sum()) < target:
        frontier = []
        for rr, cc in np.argwhere(mask):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = int(rr + dr), int(cc + dc)
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.append((nr, nc))
        nr, nc = random.choice(frontier)
        mask[nr, nc] = True
    return mask


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        h = random.randint(18, 28)
        w = random.randint(18, 28)
        grid = np.full((h, w), BG, dtype=int)
        pair_count = random.randint(2, 3)
        gate_colors = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], pair_count)
        object_colors = random.sample([c for c in range(1, 10) if c not in gate_colors], pair_count)
        ok = True

        for gate_color, obj_color in zip(gate_colors, object_colors):
            placed = False
            for _ in range(100):
                horizontal_channel = bool(random.getrandbits(1))
                gate_len = random.randint(3, 5)
                if horizontal_channel:
                    gate_rows = range(gate_len)
                    top = random.randint(2, h - gate_len - 3)
                    left_gate = random.randint(2, w - 10)
                    gap = random.randint(4, 8)
                    right_gate = left_gate + gap + 1
                    if right_gate >= w - 2:
                        continue
                    gate1 = [(top + k, left_gate) for k in gate_rows]
                    gate2 = [(top + k, right_gate) for k in gate_rows]
                    if any(grid[r, c] != BG for r, c in gate1 + gate2):
                        continue
                    channel_r0, channel_r1 = top, top + gate_len - 1
                    channel_c0, channel_c1 = left_gate + 1, right_gate - 1
                    obj_mask = _random_object()
                    oh, ow = obj_mask.shape
                    if oh > gate_len or ow > gap - 1:
                        continue
                    or0 = random.randint(channel_r0, channel_r1 - oh + 1)
                    oc0 = random.randint(channel_c0, channel_c1 - ow + 1)
                    obj_cells = [(or0 + rr, oc0 + cc) for rr, cc in np.argwhere(obj_mask)]
                    dest_left = min(2 * left_gate - c for _, c in obj_cells)
                    dest_right = max(2 * right_gate - c for _, c in obj_cells)
                    if dest_left < 0 and dest_right >= w:
                        continue
                else:
                    left = random.randint(2, w - gate_len - 3)
                    top_gate = random.randint(2, h - 10)
                    gap = random.randint(4, 8)
                    bottom_gate = top_gate + gap + 1
                    if bottom_gate >= h - 2:
                        continue
                    gate1 = [(top_gate, left + k) for k in range(gate_len)]
                    gate2 = [(bottom_gate, left + k) for k in range(gate_len)]
                    if any(grid[r, c] != BG for r, c in gate1 + gate2):
                        continue
                    channel_r0, channel_r1 = top_gate + 1, bottom_gate - 1
                    channel_c0, channel_c1 = left, left + gate_len - 1
                    obj_mask = _random_object()
                    oh, ow = obj_mask.shape
                    if oh > gap - 1 or ow > gate_len:
                        continue
                    or0 = random.randint(channel_r0, channel_r1 - oh + 1)
                    oc0 = random.randint(channel_c0, channel_c1 - ow + 1)
                    obj_cells = [(or0 + rr, oc0 + cc) for rr, cc in np.argwhere(obj_mask)]
                    dest_top = min(2 * top_gate - r for r, _ in obj_cells)
                    dest_bottom = max(2 * bottom_gate - r for r, _ in obj_cells)
                    if dest_top < 0 and dest_bottom >= h:
                        continue

                if any(grid[r, c] != BG for r, c in obj_cells):
                    continue
                for r, c in gate1 + gate2:
                    grid[r, c] = gate_color
                for r, c in obj_cells:
                    grid[r, c] = obj_color
                placed = True
                break

            if not placed:
                ok = False
                break

        if ok:
            return grid
    raise RuntimeError("failed to generate a valid instance")
