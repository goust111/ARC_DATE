import random
from collections import deque
from typing import Iterable

import numpy as np


BG = 5
SEP = 9
PALETTE = [1, 2, 3, 4, 6, 7, 8]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_ATTEMPTS = 400


def _neighbors(r: int, c: int) -> Iterable[tuple[int, int]]:
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _find_separator_row(g: np.ndarray) -> int:
    rows = np.where(np.all(g == SEP, axis=1))[0]
    assert rows.size == 1, "expected exactly one full-width separator row"
    sep = int(rows[0])
    assert 1 <= sep < g.shape[0] - 1, "separator must split the grid into two non-empty panels"
    assert np.all(g[:sep] != SEP), "separator color must not appear in the upper panel"
    return sep


def _component_from(g: np.ndarray, start: tuple[int, int], used: np.ndarray) -> list[tuple[int, int]]:
    color = int(g[start])
    q = deque([start])
    used[start] = True
    coords: list[tuple[int, int]] = []
    while q:
        r, c = q.popleft()
        coords.append((r, c))
        for nr, nc in _neighbors(r, c):
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and not used[nr, nc] and g[nr, nc] == color:
                used[nr, nc] = True
                q.append((nr, nc))
    return coords


def _monochrome_components(g: np.ndarray, bg: int) -> list[dict]:
    used = np.zeros(g.shape, dtype=bool)
    comps = []
    for r, c in np.argwhere((g != bg) & (g != SEP)):
        rr, cc = int(r), int(c)
        if used[rr, cc]:
            continue
        coords = _component_from(g, (rr, cc), used)
        rows = [p[0] for p in coords]
        cols = [p[1] for p in coords]
        comps.append(
            {
                "color": int(g[rr, cc]),
                "coords": coords,
                "bbox": (min(rows), min(cols), max(rows), max(cols)),
            }
        )
    return comps


def _ordered_template(comp: dict) -> tuple[int, list[tuple[int, int]]]:
    coords = set(comp["coords"])
    degree = {
        cell: sum((nbr in coords) for nbr in _neighbors(*cell))
        for cell in coords
    }
    assert all(d <= 2 for d in degree.values()), "template must be a one-pixel-wide path"
    endpoints = sorted([cell for cell, d in degree.items() if d == 1])
    assert len(endpoints) == 2, "template must be an open path with two endpoints"
    start = endpoints[0]

    ordered = [start]
    prev = None
    cur = start
    while True:
        nxts = [nbr for nbr in _neighbors(*cur) if nbr in coords and nbr != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "path must not branch"
        prev, cur = cur, nxts[0]
        ordered.append(cur)
    assert len(ordered) == len(coords), "path must be connected without cycles"

    r0, c0, _, _ = comp["bbox"]
    rel = [(r - r0, c - c0) for r, c in ordered]
    start_rel = rel[0][0], rel[0][1]
    return comp["color"], [(r - start_rel[0], c - start_rel[1]) for r, c in rel]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert np.any(g == BG), "grey background must be present"

    sep = _find_separator_row(g)
    upper = g[:sep]
    lower = g[sep + 1 :]
    assert np.all((upper == BG) | (upper != SEP))
    assert np.all((lower == BG) | (lower != SEP))

    templates: dict[int, list[tuple[int, int]]] = {}
    for comp in _monochrome_components(upper, BG):
        color, rel_coords = _ordered_template(comp)
        assert color not in templates, "each template color must appear exactly once in the upper panel"
        templates[color] = rel_coords

    assert templates, "expected at least one template"

    lower_fg = lower != BG
    for r, c in np.argwhere(lower_fg):
        rr, cc = int(r), int(c)
        color = int(lower[rr, cc])
        assert color in templates, "every lower-panel seed color must be defined by an upper-panel template"
        same_color = np.argwhere(lower == color)
        for r2, c2 in same_color:
            if abs(int(r2) - rr) + abs(int(c2) - cc) == 1:
                raise AssertionError("lower markers must be isolated single pixels")

    out = np.full(lower.shape, BG, dtype=int)
    for r, c in np.argwhere(lower_fg):
        rr, cc = int(r), int(c)
        color = int(lower[rr, cc])
        for dr, dc in templates[color]:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]:
                out[nr, nc] = color
    return out


def _path_cells(length: int) -> list[tuple[int, int]]:
    cells = [(0, 0)]
    occupied = {(0, 0)}
    while len(cells) < length:
        r, c = cells[-1]
        options = []
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if (nr, nc) in occupied:
                continue
            occupied.add((nr, nc))
            deg = sum((nbr in occupied) for nbr in _neighbors(nr, nc))
            occupied.remove((nr, nc))
            if deg != 1:
                continue
            options.append((nr, nc))
        if not options:
            return []
        weights = []
        prev_dir = None
        if len(cells) >= 2:
            pr, pc = cells[-2]
            prev_dir = (r - pr, c - pc)
        for nr, nc in options:
            step = (nr - r, nc - c)
            weights.append(3 if step != prev_dir else 2)
        nxt = random.choices(options, weights=weights, k=1)[0]
        occupied.add(nxt)
        cells.append(nxt)

    # Reject purely straight paths too often so bends appear frequently.
    turns = 0
    for i in range(2, len(cells)):
        a = (cells[i - 1][0] - cells[i - 2][0], cells[i - 1][1] - cells[i - 2][1])
        b = (cells[i][0] - cells[i - 1][0], cells[i][1] - cells[i - 1][1])
        turns += a != b
    if turns == 0 and random.random() < 0.7:
        return []
    return cells


def _normalize(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return [(r - min_r, c - min_c) for r, c in cells]


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return max(rows) + 1, max(cols) + 1


def _place_template(
    panel: np.ndarray,
    color: int,
    cells: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
) -> bool:
    h, w = _bbox(cells)
    for _ in range(80):
        r0 = random.randint(0, panel.shape[0] - h)
        c0 = random.randint(0, panel.shape[1] - w)
        placed = {(r0 + r, c0 + c) for r, c in cells}
        dilated = {
            nbr
            for cell in placed
            for nbr in ([cell] + [n for n in _neighbors(*cell) if 0 <= n[0] < panel.shape[0] and 0 <= n[1] < panel.shape[1]])
        }
        if occupied & dilated:
            continue
        for r, c in placed:
            panel[r, c] = color
        occupied |= dilated
        return True
    return False


def _can_stamp(
    out_shape: tuple[int, int],
    anchor: tuple[int, int],
    rel_cells: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
) -> bool:
    cells_in = set()
    for dr, dc in rel_cells:
        nr, nc = anchor[0] + dr, anchor[1] + dc
        if 0 <= nr < out_shape[0] and 0 <= nc < out_shape[1]:
            cells_in.add((nr, nc))
    if len(cells_in) < 2:
        return False
    if occupied & cells_in:
        return False
    return True


def _choose_anchor(
    out_shape: tuple[int, int],
    rel_cells: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    blocked_seeds: set[tuple[int, int]],
) -> tuple[int, int] | None:
    offsets = rel_cells
    min_dr = min(dr for dr, _ in offsets)
    max_dr = max(dr for dr, _ in offsets)
    min_dc = min(dc for _, dc in offsets)
    max_dc = max(dc for _, dc in offsets)

    interior = random.random() < 0.5
    for _ in range(120):
        if interior:
            r_lo = max(0, -min_dr)
            r_hi = min(out_shape[0] - 1, out_shape[0] - 1 - max_dr)
            c_lo = max(0, -min_dc)
            c_hi = min(out_shape[1] - 1, out_shape[1] - 1 - max_dc)
            if r_lo > r_hi or c_lo > c_hi:
                break
            anchor = (random.randint(r_lo, r_hi), random.randint(c_lo, c_hi))
        else:
            anchor = (random.randint(0, out_shape[0] - 1), random.randint(0, out_shape[1] - 1))
        if anchor in blocked_seeds:
            continue
        if _can_stamp(out_shape, anchor, rel_cells, occupied):
            return anchor
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(14, 23)
    w = random.randint(14, 24)
    sep = random.randint(5, h - 6)
    upper_h = sep
    lower_h = h - sep - 1
    if upper_h < 4 or lower_h < 4:
        return None

    num_templates = random.randint(2, min(4, len(PALETTE)))
    colors = random.sample(PALETTE, num_templates)

    templates: dict[int, list[tuple[int, int]]] = {}
    upper = np.full((upper_h, w), BG, dtype=int)
    occupied_upper: set[tuple[int, int]] = set()

    for color in colors:
        for _ in range(80):
            length = random.randint(3, min(9, upper_h * w // 4))
            raw = _path_cells(length)
            if not raw:
                continue
            norm = _normalize(raw)
            bh, bw = _bbox(norm)
            if bh > upper_h or bw > w:
                continue
            if _place_template(upper, color, norm, occupied_upper):
                coords = set(norm)
                degree = {cell: sum((nbr in coords) for nbr in _neighbors(*cell)) for cell in coords}
                endpoints = sorted([cell for cell, d in degree.items() if d == 1])
                if len(endpoints) != 2:
                    upper[upper == color] = BG
                    continue
                start = endpoints[0]
                templates[color] = [(r - start[0], c - start[1]) for r, c in norm]
                break
        else:
            return None

    lower = np.full((lower_h, w), BG, dtype=int)
    occupied_out: set[tuple[int, int]] = set()
    blocked_seeds: set[tuple[int, int]] = set()
    seed_count_total = 0

    multi_seed_colors = set(random.sample(colors, k=random.randint(0, min(2, len(colors)))))
    for color in colors:
        seeds_for_color = 2 if color in multi_seed_colors else 1
        if random.random() < 0.2:
            seeds_for_color += 1
        for _ in range(seeds_for_color):
            anchor = _choose_anchor(lower.shape, templates[color], occupied_out, blocked_seeds)
            if anchor is None:
                return None
            lower[anchor] = color
            blocked_seeds.add(anchor)
            for nbr in _neighbors(*anchor):
                if 0 <= nbr[0] < lower.shape[0] and 0 <= nbr[1] < lower.shape[1]:
                    blocked_seeds.add(nbr)
            for dr, dc in templates[color]:
                nr, nc = anchor[0] + dr, anchor[1] + dc
                if 0 <= nr < lower.shape[0] and 0 <= nc < lower.shape[1]:
                    occupied_out.add((nr, nc))
            seed_count_total += 1

    if seed_count_total < num_templates:
        return None

    grid = np.full((h, w), BG, dtype=int)
    grid[:sep] = upper
    grid[sep] = SEP
    grid[sep + 1 :] = lower

    # Validate by round-tripping the solver and ensuring the lower panel is non-trivial.
    out = main(grid)
    if out.shape != lower.shape:
        return None
    if np.array_equal(out, lower):
        return None
    if len(np.unique(out[out != BG])) < 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
