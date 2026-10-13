from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
SEP = 9
PANEL_ROWS = 3
PANEL_COLS = 3
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8]
GEN_ATTEMPTS = 400


def _components_by_color(panel: np.ndarray, bg: int) -> dict[int, list[np.ndarray]]:
    h, w = panel.shape
    seen = np.zeros((h, w), dtype=bool)
    out: dict[int, list[np.ndarray]] = {}
    for r in range(h):
        for c in range(w):
            color = int(panel[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(panel[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comp = np.array(cells, dtype=int)
            out.setdefault(color, []).append(comp)
    return out


def _find_separator_lines(grid: np.ndarray, axis: int, bg: int) -> list[int]:
    limit = grid.shape[axis]
    lines = []
    for idx in range(limit):
        line = grid[idx, :] if axis == 0 else grid[:, idx]
        color = int(line[0])
        if color != bg and np.all(line == color):
            lines.append(idx)
    return lines


def _split_panels(grid: np.ndarray) -> tuple[list[list[np.ndarray]], int, int]:
    g = np.asarray(grid, dtype=int)
    bg = Counter(map(int, g.ravel())).most_common(1)[0][0]
    sep_rows = _find_separator_lines(g, axis=0, bg=bg)
    sep_cols = _find_separator_lines(g, axis=1, bg=bg)
    assert len(sep_rows) == PANEL_ROWS - 1
    assert len(sep_cols) == PANEL_COLS - 1

    row_cuts = [-1, *sep_rows, g.shape[0]]
    col_cuts = [-1, *sep_cols, g.shape[1]]
    row_spans = [(row_cuts[i] + 1, row_cuts[i + 1]) for i in range(PANEL_ROWS)]
    col_spans = [(col_cuts[i] + 1, col_cuts[i + 1]) for i in range(PANEL_COLS)]
    heights = {r1 - r0 for r0, r1 in row_spans}
    widths = {c1 - c0 for c0, c1 in col_spans}
    assert len(heights) == 1 and len(widths) == 1
    ph = heights.pop()
    pw = widths.pop()
    panels: list[list[np.ndarray]] = []
    for r0, r1 in row_spans:
        row = []
        for c0, c1 in col_spans:
            row.append(g[r0:r1, c0:c1])
        panels.append(row)
    return panels, ph, pw


def _panel_info(panel: np.ndarray, bg: int) -> dict:
    comps = _components_by_color(panel, bg)
    assert len(comps) == 2
    odd_color = None
    odd_comp = None
    major_color = None
    major_comps = None
    for color, parts in comps.items():
        if len(parts) == 1:
            assert odd_color is None
            odd_color = int(color)
            odd_comp = parts[0]
        else:
            major_color = int(color)
            major_comps = parts
    assert odd_color is not None and major_color is not None
    assert len(major_comps) >= 2
    return {
        "odd_color": odd_color,
        "major_color": major_color,
        "odd_cells": odd_comp,
        "major_components": major_comps,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Extract each panel's odd component, recolor it along a hidden adjacency chain, and tight-crop the assembly.

    The input is a 3x3 lattice of equal panels split by solid separator lines.
    Each panel contains several 4-connected components of one repeated color and
    one 4-connected odd component of a different color. The odd color of each
    panel (except the first) equals the repeated color of the previous panel in
    a hidden path through orthogonally adjacent panels.
    """
    g = np.asarray(grid, dtype=int)
    bg = Counter(map(int, g.ravel())).most_common(1)[0][0]
    panels, ph, pw = _split_panels(g)

    infos: list[dict] = []
    for pr in range(PANEL_ROWS):
        for pc in range(PANEL_COLS):
            info = _panel_info(panels[pr][pc], bg)
            info["pos"] = (pr, pc)
            infos.append(info)

    by_pos = {info["pos"]: info for info in infos}
    preds: dict[tuple[int, int], tuple[int, int]] = {}
    succs: dict[tuple[int, int], tuple[int, int]] = {}

    for pr in range(PANEL_ROWS):
        for pc in range(PANEL_COLS):
            cur = by_pos[(pr, pc)]
            for dr, dc in DIRS4:
                nr, nc = pr + dr, pc + dc
                if not (0 <= nr < PANEL_ROWS and 0 <= nc < PANEL_COLS):
                    continue
                nxt = by_pos[(nr, nc)]
                if int(nxt["odd_color"]) == int(cur["major_color"]):
                    assert (pr, pc) not in succs
                    assert (nr, nc) not in preds
                    succs[(pr, pc)] = (nr, nc)
                    preds[(nr, nc)] = (pr, pc)

    all_pos = {(r, c) for r in range(PANEL_ROWS) for c in range(PANEL_COLS)}
    starts = [pos for pos in all_pos if pos not in preds]
    ends = [pos for pos in all_pos if pos not in succs]
    assert len(starts) == 1 and len(ends) == 1

    order = []
    cur = starts[0]
    seen = set()
    while True:
        assert cur not in seen
        seen.add(cur)
        order.append(cur)
        if cur not in succs:
            break
        cur = succs[cur]
    assert len(order) == PANEL_ROWS * PANEL_COLS

    out = np.full((PANEL_ROWS * ph, PANEL_COLS * pw), bg, dtype=int)
    prev_major = None
    for idx, pos in enumerate(order):
        info = by_pos[pos]
        color = int(info["odd_color"]) if idx == 0 else int(prev_major)
        r_off = pos[0] * ph
        c_off = pos[1] * pw
        cells = info["odd_cells"]
        out[cells[:, 0] + r_off, cells[:, 1] + c_off] = color
        prev_major = int(info["major_color"])

    nz = np.argwhere(out != bg)
    r0, c0 = nz.min(axis=0)
    r1, c1 = nz.max(axis=0)
    return out[r0 : r1 + 1, c0 : c1 + 1]


def _neighbors(pos: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = pos
    out = []
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < PANEL_ROWS and 0 <= nc < PANEL_COLS:
            out.append((nr, nc))
    return out


def _random_hamiltonian_path() -> list[tuple[int, int]]:
    all_cells = [(r, c) for r in range(PANEL_ROWS) for c in range(PANEL_COLS)]

    def dfs(path: list[tuple[int, int]], used: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if len(path) == len(all_cells):
            return path[:]
        cur = path[-1]
        nxts = _neighbors(cur)
        random.shuffle(nxts)
        nxts.sort(key=lambda p: sum(nn not in used for nn in _neighbors(p)))
        for nxt in nxts:
            if nxt in used:
                continue
            used.add(nxt)
            path.append(nxt)
            sol = dfs(path, used)
            if sol is not None:
                return sol
            path.pop()
            used.remove(nxt)
        return None

    starts = all_cells[:]
    random.shuffle(starts)
    for start in starts:
        sol = dfs([start], {start})
        if sol is not None:
            return sol
    raise RuntimeError("failed to build a panel path")


def _assign_chain_colors(path: list[tuple[int, int]]) -> tuple[list[int], list[int]]:
    pos_to_idx = {pos: i for i, pos in enumerate(path)}
    odds = [None] * len(path)
    majors = [None] * len(path)
    odds[0] = random.choice(COLOR_POOL)

    def backtrack(i: int) -> bool:
        if i == len(path):
            return True
        cur = path[i]
        prev = path[i - 1] if i > 0 else None
        candidates = COLOR_POOL[:]
        random.shuffle(candidates)
        for color in candidates:
            if color == odds[i]:
                continue
            ok = True
            for nb in _neighbors(cur):
                if nb not in pos_to_idx:
                    continue
                j = pos_to_idx[nb]
                if j < i:
                    if nb != prev and majors[j] == odds[i]:
                        ok = False
                        break
                    if color == odds[j]:
                        ok = False
                        break
            if not ok:
                continue
            majors[i] = color
            if i + 1 < len(path):
                odds[i + 1] = color
            if backtrack(i + 1):
                return True
            majors[i] = None
            if i + 1 < len(path):
                odds[i + 1] = None
        return False

    if not backtrack(0):
        raise RuntimeError("failed to assign chain colors")
    return [int(x) for x in majors], [int(x) for x in odds]


def _random_component(max_h: int, max_w: int, min_cells: int, max_cells: int) -> np.ndarray:
    h = random.randint(max(1, min(2, max_h)), max_h)
    w = random.randint(max(1, min(2, max_w)), max_w)
    target = random.randint(min_cells, min(max_cells, h * w))
    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    mask[start] = True
    frontier = [start]
    while int(mask.sum()) < target:
        base = random.choice(frontier)
        nbrs = []
        for dr, dc in DIRS4:
            nr, nc = base[0] + dr, base[1] + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                nbrs.append((nr, nc))
        if not nbrs:
            frontier.remove(base)
            if not frontier:
                frontier = list(map(tuple, np.argwhere(mask)))
            continue
        pick = random.choice(nbrs)
        mask[pick] = True
        frontier.append(pick)
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _can_place(panel: np.ndarray, shape: np.ndarray, top: int, left: int) -> bool:
    ph, pw = panel.shape
    sh, sw = shape.shape
    for r in range(sh):
        for c in range(sw):
            if not shape[r, c]:
                continue
            rr = top + r
            cc = left + c
            if not (0 <= rr < ph and 0 <= cc < pw):
                return False
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < ph and 0 <= nc < pw and int(panel[nr, nc]) != BG:
                        return False
    return True


def _stamp(panel: np.ndarray, shape: np.ndarray, top: int, left: int, color: int) -> None:
    sh, sw = shape.shape
    for r in range(sh):
        for c in range(sw):
            if shape[r, c]:
                panel[top + r, left + c] = color


def _build_panel(major_color: int, odd_color: int, ph: int, pw: int) -> np.ndarray | None:
    panel = np.full((ph, pw), BG, dtype=int)
    major_count = random.randint(2, 4)
    shapes: list[tuple[np.ndarray, int]] = []
    for _ in range(major_count):
        shapes.append((_random_component(3, 3, 1, 4), major_color))
    shapes.append((_random_component(3, 3, 1, 4), odd_color))
    random.shuffle(shapes)

    for shape, color in shapes:
        spots = []
        for top in range(ph - shape.shape[0] + 1):
            for left in range(pw - shape.shape[1] + 1):
                if _can_place(panel, shape, top, left):
                    spots.append((top, left))
        if not spots:
            return None
        top, left = random.choice(spots)
        _stamp(panel, shape, top, left, color)

    comps = _components_by_color(panel, BG)
    if set(comps) != {major_color, odd_color}:
        return None
    if len(comps[major_color]) < 2 or len(comps[odd_color]) != 1:
        return None
    return panel


def _render_grid(path: list[tuple[int, int]], majors: list[int], odds: list[int], ph: int, pw: int) -> np.ndarray | None:
    h = PANEL_ROWS * ph + (PANEL_ROWS - 1)
    w = PANEL_COLS * pw + (PANEL_COLS - 1)
    grid = np.full((h, w), BG, dtype=int)
    for sr in range(1, PANEL_ROWS):
        grid[sr * ph + (sr - 1), :] = SEP
    for sc in range(1, PANEL_COLS):
        grid[:, sc * pw + (sc - 1)] = SEP

    for idx, (pr, pc) in enumerate(path):
        panel = _build_panel(majors[idx], odds[idx], ph, pw)
        if panel is None:
            return None
        r0 = pr * (ph + 1)
        c0 = pc * (pw + 1)
        grid[r0 : r0 + ph, c0 : c0 + pw] = panel
    return grid


def _gen_attempt() -> np.ndarray | None:
    ph = random.randint(5, 8)
    pw = random.randint(5, 8)
    path = _random_hamiltonian_path()
    try:
        majors, odds = _assign_chain_colors(path)
    except RuntimeError:
        return None
    grid = _render_grid(path, majors, odds, ph, pw)
    if grid is None:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if not isinstance(out, np.ndarray):
        return None
    if out.size == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
