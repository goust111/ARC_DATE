from __future__ import annotations

from collections import Counter

import numpy as np


Coord = tuple[int, int]


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _split_panels(grid: np.ndarray) -> list[np.ndarray]:
    h, w = grid.shape
    for color in np.unique(grid):
        cols = [c for c in range(w) if np.all(grid[:, c] == color)]
        if len(cols) < 2:
            continue
        widths = [cols[0]]
        widths.extend(cols[i] - cols[i - 1] - 1 for i in range(1, len(cols)))
        widths.append(w - cols[-1] - 1)
        if min(widths) < 5 or len(set(widths)) != 1:
            continue
        panel_w = widths[0]
        panels = []
        start = 0
        for sep in cols:
            panels.append(grid[:, start:sep])
            start = sep + 1
        panels.append(grid[:, start:w])
        if len(panels) >= 3 and all(panel.shape == (h, panel_w) for panel in panels):
            return panels
    raise AssertionError("failed to split panels")


def _marker_color(panels: list[np.ndarray]) -> int:
    colors = sorted(set(int(v) for panel in panels for v in panel.ravel()))
    for color in colors:
        coords = []
        ok = True
        for panel in panels:
            pts = np.argwhere(panel == color)
            if len(pts) != 1:
                ok = False
                break
            coords.append(tuple(map(int, pts[0])))
        if ok and len(set(coords)) > 1:
            return color
    raise AssertionError("failed to identify marker color")


def _path_color(panels: list[np.ndarray], marker_color: int) -> int:
    stack = np.stack(panels, axis=0)
    bg = _get_bg(stack.reshape(-1, stack.shape[-1]))
    variable = np.any(stack != stack[:1], axis=0)
    colors = {
        int(v)
        for v in stack[:, variable].ravel()
        if int(v) != bg and int(v) != marker_color
    }
    assert len(colors) == 1
    return next(iter(colors))


def _find_gate(panel: np.ndarray, path_color: int) -> tuple[Coord, set[Coord], Coord]:
    h, w = panel.shape
    for r in range(h):
        for c in range(w):
            if int(panel[r, c]) != path_color:
                continue
            for dr, dc in ((0, 1), (1, 0)):
                a = (r - dr, c - dc)
                b = (r + dr, c + dc)
                if not (_in_bounds(panel.shape, *a) and _in_bounds(panel.shape, *b)):
                    continue
                end_color = int(panel[a])
                if end_color == path_color or end_color != int(panel[b]):
                    continue
                gate_ends = {a, b}
                parking = []
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) in gate_ends or not _in_bounds(panel.shape, nr, nc):
                        continue
                    if int(panel[nr, nc]) == path_color:
                        parking.append((nr, nc))
                if len(parking) == 1:
                    return (r, c), gate_ends, parking[0]
    raise AssertionError("failed to identify gate")


def _route_component(panel: np.ndarray, marker: Coord, path_color: int, marker_color: int) -> list[Coord]:
    allowed = {path_color, marker_color}
    seen = {marker}
    stack = [marker]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if not _in_bounds(panel.shape, nr, nc):
                continue
            if (nr, nc) in seen or int(panel[nr, nc]) not in allowed:
                continue
            seen.add((nr, nc))
            stack.append((nr, nc))
    return list(seen)


def _order_path(route: list[Coord], start: Coord) -> list[Coord]:
    route_set = set(route)
    nbrs = {
        cell: [n for n in _neighbors4(*cell) if n in route_set]
        for cell in route_set
    }
    assert sum(len(v) == 1 for v in nbrs.values()) == 2
    ordered = [start]
    prev = None
    cur = start
    while True:
        nxts = [n for n in nbrs[cur] if n != prev]
        if not nxts:
            return ordered
        assert len(nxts) == 1
        nxt = nxts[0]
        ordered.append(nxt)
        prev, cur = cur, nxt


def main(grid: np.ndarray) -> np.ndarray:
    """Predict the next panel by advancing the marker along the shrinking path, capped at the gate-adjacent stop cell.

    Input structure:
    - The input is a single row of equal panels separated by full-height divider columns.
    - Every panel shares the same static template: background, a one-cell-wide path, a straight end-center-end gate, and small decorative pixels.
    - One marker sits on the path in each panel, and the path behind that marker has already been erased to background.

    Rule:
    - Split the row into panels and recover the path order from the earliest panel, where the visible suffix is longest.
    - Infer the constant per-frame step size from the marker's index difference between consecutive panels.
    - Starting from the last panel, erase the traversed path cells and move the marker forward by that many steps, but stop early on the unique path cell adjacent to the gate center and outside the two gate ends.

    Key insight:
    - The frame-to-frame speed is encoded by how far the marker advances along the same path suffix, while the gate imposes a hard upper bound on the extrapolation.
    """
    panels = _split_panels(np.asarray(grid, dtype=int))
    marker_color = _marker_color(panels)
    path_color = _path_color(panels, marker_color)

    path_counts = [int(np.count_nonzero(panel == path_color)) for panel in panels]
    if any(a < b for a, b in zip(path_counts, path_counts[1:])):
        panels = list(reversed(panels))

    first = panels[0]
    last = panels[-1]
    bg = _get_bg(first)

    marker_positions = [tuple(map(int, np.argwhere(panel == marker_color)[0])) for panel in panels]
    gate_center, _gate_ends, stop_cell = _find_gate(first, path_color)

    route = _route_component(first, marker_positions[0], path_color, marker_color)
    assert gate_center in route and stop_cell in route
    ordered = _order_path(route, marker_positions[0])
    index_of = {cell: i for i, cell in enumerate(ordered)}

    marker_steps = [index_of[pos] for pos in marker_positions]
    diffs = [b - a for a, b in zip(marker_steps, marker_steps[1:])]
    assert diffs and len(set(diffs)) == 1 and diffs[0] > 0
    step = diffs[0]

    last_idx = marker_steps[-1]
    stop_idx = index_of[stop_cell]
    next_idx = min(last_idx + step, stop_idx)

    out = last.copy()
    for idx in range(last_idx, next_idx):
        r, c = ordered[idx]
        out[r, c] = bg
    nr, nc = ordered[next_idx]
    out[nr, nc] = marker_color
    return out


def _render_panel(
    template: np.ndarray,
    path: list[Coord],
    gate_center: Coord,
    marker_idx: int,
    path_color: int,
    marker_color: int,
    bg: int,
) -> np.ndarray:
    panel = template.copy()
    for i in range(marker_idx):
        r, c = path[i]
        panel[r, c] = bg
    mr, mc = path[marker_idx]
    panel[mr, mc] = marker_color
    for i in range(marker_idx + 1, len(path)):
        r, c = path[i]
        panel[r, c] = path_color
    gr, gc = gate_center
    panel[gr, gc] = path_color
    return panel


def _sample_path(
    rng: np.random.Generator,
    shape: tuple[int, int],
    stop_cell: Coord,
    blocked: set[Coord],
    target_len: int,
) -> list[Coord] | None:
    used = {stop_cell}
    grown = [stop_cell]

    def rec() -> bool:
        if len(grown) == target_len:
            return True
        r, c = grown[-1]
        cands = []
        for nr, nc in _neighbors4(r, c):
            if not _in_bounds(shape, nr, nc):
                continue
            if (nr, nc) in used or (nr, nc) in blocked:
                continue
            degree = sum((ar, ac) in used for ar, ac in _neighbors4(nr, nc))
            if degree != 1:
                continue
            cands.append((nr, nc))
        if not cands:
            return False
        for idx in rng.permutation(len(cands)):
            cell = cands[int(idx)]
            used.add(cell)
            grown.append(cell)
            if rec():
                return True
            grown.pop()
            used.remove(cell)
        return False

    if not rec():
        return None
    return list(reversed(grown))


def _place_decor(
    rng: np.random.Generator,
    template: np.ndarray,
    forbidden: set[Coord],
    colors: list[int],
) -> None:
    h, w = template.shape
    decor_count = int(rng.integers(2, 5))
    for _ in range(decor_count):
        color = int(colors[int(rng.integers(len(colors)))])
        size = 1 if rng.random() < 0.6 else 2
        orient = (0, 1) if rng.random() < 0.5 else (1, 0)
        for _attempt in range(50):
            r = int(rng.integers(h))
            c = int(rng.integers(w))
            cells = [(r + k * orient[0], c + k * orient[1]) for k in range(size)]
            if not all(_in_bounds((h, w), rr, cc) for rr, cc in cells):
                continue
            halo = set(cells)
            for rr, cc in cells:
                halo.update((nr, nc) for nr, nc in _neighbors4(rr, cc) if _in_bounds((h, w), nr, nc))
            if halo & forbidden:
                continue
            for rr, cc in cells:
                template[rr, cc] = color
            forbidden.update(halo)
            break


def _build_instance(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    panel_count = int(rng.choice([3, 4]))
    panel_h = int(rng.integers(7, 11))
    panel_w = 6 if panel_count == 4 else int(rng.integers(7, 10))
    if panel_count * panel_w + panel_count - 1 > 30:
        return None

    colors = list(map(int, rng.permutation(np.arange(10))))
    bg, path_color, marker_color, gate_end, divider = colors[:5]
    decor_colors = colors[5:8]

    horizontal_gate = bool(rng.integers(2))
    if horizontal_gate:
        center = (int(rng.integers(1, panel_h - 1)), int(rng.integers(1, panel_w - 1)))
        end1 = (center[0], center[1] - 1)
        end2 = (center[0], center[1] + 1)
        stop_options = [(center[0] - 1, center[1]), (center[0] + 1, center[1])]
    else:
        center = (int(rng.integers(1, panel_h - 1)), int(rng.integers(1, panel_w - 1)))
        end1 = (center[0] - 1, center[1])
        end2 = (center[0] + 1, center[1])
        stop_options = [(center[0], center[1] - 1), (center[0], center[1] + 1)]
    if not all(_in_bounds((panel_h, panel_w), *cell) for cell in (end1, end2)):
        return None
    stop_cell = stop_options[int(rng.integers(len(stop_options)))]
    if not _in_bounds((panel_h, panel_w), *stop_cell):
        return None

    step = int(rng.integers(1, 4))
    last_marker_idx = (panel_count - 1) * step
    stop_branch = bool(rng.integers(2))
    if stop_branch:
        stop_idx = last_marker_idx + int(rng.integers(1, step + 1))
    else:
        stop_idx = last_marker_idx + step + int(rng.integers(1, 4))
    path_len = stop_idx + 1

    blocked = {center, end1, end2}
    blocked.update(cell for cell in _neighbors4(*center) if cell != stop_cell and _in_bounds((panel_h, panel_w), *cell))
    path = _sample_path(rng, (panel_h, panel_w), stop_cell, blocked, path_len)
    if path is None:
        return None

    template = np.full((panel_h, panel_w), bg, dtype=int)
    for r, c in path:
        template[r, c] = path_color
    template[center] = path_color
    template[end1] = gate_end
    template[end2] = gate_end

    forbidden = set(path) | {center, end1, end2}
    for cell in list(forbidden):
        for nbr in _neighbors4(*cell):
            if _in_bounds((panel_h, panel_w), *nbr):
                forbidden.add(nbr)
    _place_decor(rng, template, forbidden, decor_colors)

    panels = []
    for frame in range(panel_count):
        marker_idx = frame * step
        panels.append(_render_panel(template, path, center, marker_idx, path_color, marker_color, bg))

    next_idx = min(stop_idx, panel_count * step)
    expected = _render_panel(template, path, center, next_idx, path_color, marker_color, bg)

    sep = np.full((panel_h, 1), divider, dtype=int)
    pieces = []
    for i, panel in enumerate(panels):
        pieces.append(panel)
        if i + 1 < len(panels):
            pieces.append(sep)
    inp = np.concatenate(pieces, axis=1)
    if not np.array_equal(main(inp), expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        built = _build_instance(rng)
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid input")
