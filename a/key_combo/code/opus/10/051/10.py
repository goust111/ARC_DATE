from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
TOP_COLOR = 1
BOTTOM_COLOR = 2
CLOSED_COLOR = 3
OPEN_COLOR = 4
SEPARATOR_COLOR = 5
N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in N4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _component_has_hole(mask: np.ndarray) -> bool:
    h, w = mask.shape
    exterior = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    def enqueue(r: int, c: int):
        if not mask[r, c] and not exterior[r, c]:
            exterior[r, c] = True
            q.append((r, c))

    for r in range(h):
        enqueue(r, 0)
        enqueue(r, w - 1)
    for c in range(w):
        enqueue(0, c)
        enqueue(h - 1, c)

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if not mask[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((nr, nc))

    return bool(np.any((~mask) & (~exterior)))


def _find_separator_row(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h = g.shape[0]
    candidates = []
    for r in range(1, h - 1):
        if np.all(g[r] == SEPARATOR_COLOR) and r == h - r - 1:
            candidates.append(r)
    assert len(candidates) == 1, "expected exactly one full-width middle separator row"
    return candidates[0]


def _overlay_mask(grid: np.ndarray) -> np.ndarray:
    sep = _find_separator_row(grid)
    top = grid[:sep]
    bottom = grid[sep + 1 :]
    assert top.shape == bottom.shape and top.size > 0
    return (top != BG) | (bottom != BG)


def _classify_overlay(mask: np.ndarray) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=int)
    for coords in _components(mask):
        comp_mask = np.zeros(mask.shape, dtype=bool)
        comp_mask[coords[:, 0], coords[:, 1]] = True
        color = CLOSED_COLOR if _component_has_hole(comp_mask) else OPEN_COLOR
        out[coords[:, 0], coords[:, 1]] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay the two panels, then recolor each composite foreground component by whether it encloses a hole.
    
    Input structure:
    - A black grid is split into equal top and bottom panels by one full-width gray separator row.
    - The top panel contains blue foreground pixels and the bottom panel contains red foreground pixels.
    - Each panel uses the same local coordinate system, and foreground is sparse on black background.
    
    Rule:
    - Remove the gray separator row and overlay the two panels cellwise, keeping any position occupied in either panel.
    - Treat the overlaid foreground as 4-connected components on a black background.
    - Recolor every foreground pixel in a component green if that component encloses at least one black region not connected to the outside; otherwise recolor the whole component yellow.
    - Leave all black cells, including enclosed black holes, unchanged.
    
    Key insight:
    - Closedness is not visible in either panel alone; it emerges only after the two masks are transparently overlaid in shared panel coordinates.
    """
    g = np.asarray(grid, dtype=int)
    mask = _overlay_mask(g)
    return _classify_overlay(mask)


def _rect_ring_cycle(height: int, width: int) -> list[tuple[int, int]]:
    pts = [(0, c) for c in range(width)]
    pts += [(r, width - 1) for r in range(1, height - 1)]
    pts += [(height - 1, c) for c in range(width - 1, -1, -1)]
    pts += [(r, 0) for r in range(height - 2, 0, -1)]
    return pts


def _normalize_path(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in path)
    min_c = min(c for _, c in path)
    return [(r - min_r, c - min_c) for r, c in path]


def _random_open_path(rng: np.random.Generator) -> list[tuple[int, int]]:
    target_len = int(rng.integers(1, 7))
    max_h = int(rng.integers(2, 6))
    max_w = int(rng.integers(2, 7))
    for _ in range(120):
        path = [(0, 0)]
        used = {(0, 0)}
        while len(path) < target_len:
            cr, cc = path[-1]
            options = []
            for dr, dc in N4:
                nr = cr + dr
                nc = cc + dc
                if (nr, nc) in used:
                    continue
                trial = path + [(nr, nc)]
                rows = [r for r, _ in trial]
                cols = [c for _, c in trial]
                if max(rows) - min(rows) + 1 > max_h or max(cols) - min(cols) + 1 > max_w:
                    continue
                options.append((nr, nc))
            if not options:
                break
            nxt = options[int(rng.integers(len(options)))]
            path.append(nxt)
            used.add(nxt)
        if len(path) == target_len:
            return _normalize_path(path)
    return [(0, 0), (0, 1)]


def _dilate_blocked(blocked: np.ndarray, cells: set[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                blocked[rr, cc] = True


def _fits(cells: set[tuple[int, int]], blocked: np.ndarray) -> bool:
    h, w = blocked.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if blocked[r, c]:
            return False
    return True


def _side_outward_options(cycle: list[tuple[int, int]], height: int, width: int):
    options = []
    for idx, (r, c) in enumerate(cycle):
        if r == 0 and 0 < c < width - 1:
            options.append((idx, (-1, 0)))
        elif r == height - 1 and 0 < c < width - 1:
            options.append((idx, (1, 0)))
        elif c == 0 and 0 < r < height - 1:
            options.append((idx, (0, -1)))
        elif c == width - 1 and 0 < r < height - 1:
            options.append((idx, (0, 1)))
    return options


def _place_closed_component(
    rng: np.random.Generator,
    panel_h: int,
    panel_w: int,
    blocked: np.ndarray,
):
    size_choices = []
    for rh in range(3, min(6, panel_h) + 1):
        for rw in range(3, min(7, panel_w) + 1):
            size_choices.append((rh, rw))
    rng.shuffle(size_choices)

    for ring_h, ring_w in size_choices:
        cycle_local = _rect_ring_cycle(ring_h, ring_w)
        perimeter = len(cycle_local)
        min_arc = 2
        placements = [(r0, c0) for r0 in range(panel_h - ring_h + 1) for c0 in range(panel_w - ring_w + 1)]
        rng.shuffle(placements)
        for base_r, base_c in placements:
            cycle = [(base_r + r, base_c + c) for r, c in cycle_local]
            cycle_set = set(cycle)
            if not _fits(cycle_set, blocked):
                continue

            cut_start = int(rng.integers(perimeter))
            arc_len = int(rng.integers(min_arc, perimeter - min_arc + 1))
            rotated = cycle[cut_start:] + cycle[:cut_start]
            arc_a = set(rotated[:arc_len])
            arc_b = cycle_set - arc_a
            if len(arc_b) < min_arc:
                continue

            top_cells = set(arc_a)
            bottom_cells = set(arc_b)
            if rng.random() < 0.5:
                top_cells, bottom_cells = bottom_cells, top_cells

            all_cells = set(cycle_set)
            local_spurs = _side_outward_options(cycle_local, ring_h, ring_w)
            rng.shuffle(local_spurs)
            if rng.random() < 0.45:
                for idx, (dr, dc) in local_spurs:
                    ar, ac = cycle[idx]
                    max_len = 0
                    while True:
                        nr = ar + (max_len + 1) * dr
                        nc = ac + (max_len + 1) * dc
                        if not (0 <= nr < panel_h and 0 <= nc < panel_w):
                            break
                        if blocked[nr, nc] or (nr, nc) in all_cells:
                            break
                        max_len += 1
                    if max_len == 0:
                        continue
                    spur_len = int(rng.integers(1, min(3, max_len) + 1))
                    spur = {(ar + k * dr, ac + k * dc) for k in range(1, spur_len + 1)}
                    if not _fits(spur, blocked):
                        continue
                    all_cells |= spur
                    if (ar, ac) in top_cells:
                        top_cells |= spur
                    else:
                        bottom_cells |= spur
                    break

            return top_cells, bottom_cells, all_cells
    return None


def _place_open_component(
    rng: np.random.Generator,
    panel_h: int,
    panel_w: int,
    blocked: np.ndarray,
):
    for _ in range(160):
        path_local = _random_open_path(rng)
        path_h = max(r for r, _ in path_local) + 1
        path_w = max(c for _, c in path_local) + 1
        placements = [(r0, c0) for r0 in range(panel_h - path_h + 1) for c0 in range(panel_w - path_w + 1)]
        rng.shuffle(placements)
        for base_r, base_c in placements:
            path = [(base_r + r, base_c + c) for r, c in path_local]
            cells = set(path)
            if not _fits(cells, blocked):
                continue

            mode = int(rng.integers(3))
            top_cells: set[tuple[int, int]] = set()
            bottom_cells: set[tuple[int, int]] = set()
            if mode == 0 or len(path) == 1:
                top_cells = cells
            elif mode == 1:
                bottom_cells = cells
            else:
                cut = int(rng.integers(1, len(path)))
                if rng.random() < 0.5:
                    top_cells = set(path[:cut])
                    bottom_cells = set(path[cut:])
                else:
                    top_cells = set(path[cut:])
                    bottom_cells = set(path[:cut])
            return top_cells, bottom_cells, cells
    return None


def _build_expected(panel_h: int, panel_w: int, records: list[tuple[set[tuple[int, int]], bool]]) -> np.ndarray:
    out = np.zeros((panel_h, panel_w), dtype=int)
    for cells, is_closed in records:
        color = CLOSED_COLOR if is_closed else OPEN_COLOR
        for r, c in cells:
            out[r, c] = color
    return out


def _mask_has_closed_component(mask: np.ndarray) -> bool:
    return any(_component_has_hole(_coords_to_mask(mask.shape, set(map(tuple, comp)))) for comp in _components(mask))


def _coords_to_mask(shape: tuple[int, int], cells: set[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _validate(grid: np.ndarray, records: list[tuple[set[tuple[int, int]], bool]]) -> bool:
    panel_h = (grid.shape[0] - 1) // 2
    panel_w = grid.shape[1]
    expected = _build_expected(panel_h, panel_w, records)
    overlay = _overlay_mask(grid)
    top = grid[:panel_h]
    bottom = grid[panel_h + 1 :]

    if _mask_has_closed_component(top != BG):
        return False
    if _mask_has_closed_component(bottom != BG):
        return False

    record_map = {frozenset(cells): is_closed for cells, is_closed in records}
    observed = _components(overlay)
    if len(observed) != len(records):
        return False
    saw_closed = False
    saw_open = False
    for comp in observed:
        cells = frozenset(map(tuple, comp))
        if cells not in record_map:
            return False
        hole = _component_has_hole(_coords_to_mask(overlay.shape, set(cells)))
        if hole != record_map[cells]:
            return False
        saw_closed |= hole
        saw_open |= not hole
    if not (saw_closed and saw_open):
        return False

    return np.array_equal(main(grid), expected)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    panel_h = int(rng.integers(5, 9))
    panel_w = int(rng.integers(8, 15))
    top = np.zeros((panel_h, panel_w), dtype=int)
    bottom = np.zeros((panel_h, panel_w), dtype=int)
    blocked = np.zeros((panel_h, panel_w), dtype=bool)
    records: list[tuple[set[tuple[int, int]], bool]] = []

    max_closed = 2 if panel_h * panel_w < 80 else 3
    target_closed = int(rng.integers(1, max_closed + 1))
    target_open = int(rng.integers(1, 4))

    for _ in range(target_closed):
        placed = _place_closed_component(rng, panel_h, panel_w, blocked)
        if placed is None:
            return None
        top_cells, bottom_cells, all_cells = placed
        for r, c in top_cells:
            top[r, c] = TOP_COLOR
        for r, c in bottom_cells:
            bottom[r, c] = BOTTOM_COLOR
        _dilate_blocked(blocked, all_cells)
        records.append((all_cells, True))

    for _ in range(target_open):
        placed = _place_open_component(rng, panel_h, panel_w, blocked)
        if placed is None:
            return None
        top_cells, bottom_cells, all_cells = placed
        for r, c in top_cells:
            top[r, c] = TOP_COLOR
        for r, c in bottom_cells:
            bottom[r, c] = BOTTOM_COLOR
        _dilate_blocked(blocked, all_cells)
        records.append((all_cells, False))

    grid = np.zeros((panel_h * 2 + 1, panel_w), dtype=int)
    grid[:panel_h] = top
    grid[panel_h, :] = SEPARATOR_COLOR
    grid[panel_h + 1 :] = bottom

    if not _validate(grid, records):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 10")
