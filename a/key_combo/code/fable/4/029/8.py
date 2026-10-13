from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue: deque[Coord] = deque([(r, c)])
            seen[r, c] = True
            pts: list[Coord] = []
            while queue:
                rr, cc = queue.popleft()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _ring_is_wall(grid: np.ndarray, wall: int, box: BBox) -> bool:
    h, w = grid.shape
    r0, c0, r1, c1 = box
    if r0 <= 0 or c0 <= 0 or r1 >= h - 1 or c1 >= w - 1:
        return False
    for r in range(r0 - 1, r1 + 2):
        if int(grid[r, c0 - 1]) != wall or int(grid[r, c1 + 1]) != wall:
            return False
    for c in range(c0 - 1, c1 + 2):
        if int(grid[r0 - 1, c]) != wall or int(grid[r1 + 1, c]) != wall:
            return False
    return True


def _find_wall_and_windows(grid: np.ndarray) -> tuple[int, list[BBox]]:
    counts = Counter(map(int, grid.flat))
    candidates: list[tuple[int, int, tuple[BBox, ...]]] = []
    for wall in counts:
        comps = _components(grid != wall)
        if len(comps) < 2:
            continue
        boxes: list[BBox] = []
        ok = True
        for comp in comps:
            box = _bbox(comp)
            r0, c0, r1, c1 = box
            if len(comp) != (r1 - r0 + 1) * (c1 - c0 + 1):
                ok = False
                break
            if not _ring_is_wall(grid, wall, box):
                ok = False
                break
            boxes.append(box)
        if ok:
            candidates.append((counts[wall], int(wall), tuple(sorted(boxes))))
    assert candidates, "expected a wall color enclosing multiple rectangular cavities"
    max_count = max(item[0] for item in candidates)
    best = [item for item in candidates if item[0] == max_count]
    assert len(best) == 1, "wall color should be unique"
    _, wall, boxes = best[0]
    return wall, list(boxes)


def _window_cells(grid: np.ndarray, windows: list[BBox]) -> list[tuple[int, int, int]]:
    cells: list[tuple[int, int, int]] = []
    for r0, c0, r1, c1 in windows:
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                cells.append((r, c, int(grid[r, c])))
    return cells


def _period_mapping(
    cells: list[tuple[int, int, int]], row_period: int, col_period: int
) -> dict[tuple[int, int], int] | None:
    mapping: dict[tuple[int, int], int] = {}
    for r, c, color in cells:
        key = (r % row_period, c % col_period)
        old = mapping.get(key)
        if old is None:
            mapping[key] = color
        elif old != color:
            return None
    return mapping


def _infer_wallpaper(
    grid: np.ndarray, windows: list[BBox], wall: int
) -> tuple[int, int, int, dict[tuple[int, int], int]]:
    h, w = grid.shape
    cells = _window_cells(grid, windows)
    window_colors = sorted({color for _, _, color in cells})
    candidates: list[tuple[int, int, int, int, dict[tuple[int, int], int]]] = []
    for empty in window_colors:
        if empty == wall:
            continue
        visible = [(r, c, color) for r, c, color in cells if color != empty]
        if not visible:
            continue
        for row_period in range(1, h + 1):
            for col_period in range(1, w + 1):
                mapping = _period_mapping(visible, row_period, col_period)
                if mapping is None:
                    continue
                if len(mapping) != row_period * col_period:
                    continue
                candidates.append((row_period * col_period, row_period, col_period, empty, mapping))
    assert candidates, "expected a shared periodic wallpaper"
    min_area = min(item[0] for item in candidates)
    best = [item for item in candidates if item[0] == min_area]
    assert len(best) == 1, "wallpaper period should be uniquely determined"
    _, row_period, col_period, empty, mapping = best[0]
    return empty, row_period, col_period, mapping


def main(grid: np.ndarray) -> np.ndarray:
    """Fill every rectangular wall window from one globally anchored periodic wallpaper.

    Input structure:
    - One color forms a solid wall that surrounds multiple disjoint rectangular cavities.
    - Inside each cavity, one special empty color marks missing cells and the remaining cells are visible wallpaper fragments.
    - The same wallpaper continues through every cavity in absolute grid coordinates rather than restarting per window.

    Rule:
    - Identify the wall color and the rectangular cavities it encloses.
    - Across all non-empty cavity pixels together, infer the unique smallest row/column period and the color of each residue class in absolute coordinates.
    - Repaint every cavity cell from that shared residue-class wallpaper, leaving the wall unchanged.

    Key insight:
    - No single window is solved independently; only pooled evidence from all windows reveals the wallpaper's global period and phase.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and 0 < g.shape[0] <= 30 and 0 < g.shape[1] <= 30
    wall, windows = _find_wall_and_windows(g)
    _empty, row_period, col_period, mapping = _infer_wallpaper(g, windows, wall)

    out = g.copy()
    for r0, c0, r1, c1 in windows:
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                out[r, c] = mapping[(r % row_period, c % col_period)]
    return out


GEN_MAX_ATTEMPTS = 500


def _tile_has_subperiod(tile: np.ndarray) -> bool:
    h, w = tile.shape
    for row_period in range(1, h + 1):
        for col_period in range(1, w + 1):
            if row_period == h and col_period == w:
                continue
            ok = True
            for r in range(h):
                for c in range(w):
                    if int(tile[r, c]) != int(tile[r % row_period, c % col_period]):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return True
    return False


def _sample_tile(row_period: int, col_period: int, palette: list[int]) -> np.ndarray:
    area = row_period * col_period
    min_colors = 2 if area <= 6 else 3
    for _ in range(200):
        color_count = random.randint(min_colors, min(len(palette), min(5, area)))
        colors = random.sample(palette, color_count)
        flat = [random.choice(colors) for _ in range(area)]
        for i, color in enumerate(colors):
            flat[i] = color
        random.shuffle(flat)
        tile = np.asarray(flat, dtype=int).reshape(row_period, col_period)
        if len(set(map(int, tile.flat))) < min_colors:
            continue
        if _tile_has_subperiod(tile):
            continue
        return tile
    raise RuntimeError("failed to sample a non-degenerate tile")


def _cover_residues(windows: list[BBox], row_period: int, col_period: int) -> bool:
    seen = set()
    for r0, c0, r1, c1 in windows:
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                seen.add((r % row_period, c % col_period))
    return len(seen) == row_period * col_period


def _place_windows(
    height: int, width: int, row_period: int, col_period: int, window_count: int
) -> list[BBox] | None:
    for _ in range(300):
        occupied = np.zeros((height, width), dtype=bool)
        windows: list[BBox] = []
        ok = True
        for _idx in range(window_count):
            placed = False
            for _try in range(120):
                win_h = random.randint(2, row_period - 1)
                win_w = random.randint(2, col_period - 1)
                r0 = random.randint(1, height - win_h - 2)
                c0 = random.randint(1, width - win_w - 2)
                r1 = r0 + win_h - 1
                c1 = c0 + win_w - 1
                if occupied[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2].any():
                    continue
                occupied[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2] = True
                windows.append((r0, c0, r1, c1))
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok and _cover_residues(windows, row_period, col_period):
            return sorted(windows)
    return None


def _choose_visible_cells(windows: list[BBox], row_period: int, col_period: int) -> set[Coord] | None:
    residue_cells: dict[tuple[int, int], list[Coord]] = {
        (rr, cc): []
        for rr in range(row_period)
        for cc in range(col_period)
    }
    per_window: list[list[Coord]] = []
    all_cells: list[Coord] = []
    for r0, c0, r1, c1 in windows:
        cells: list[Coord] = []
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                coord = (r, c)
                cells.append(coord)
                all_cells.append(coord)
                residue_cells[(r % row_period, c % col_period)].append(coord)
        per_window.append(cells)
    if any(not cells for cells in residue_cells.values()):
        return None

    visible = {random.choice(cells) for cells in residue_cells.values()}
    if any(all(cell in visible for cell in cells) for cells in per_window):
        return None

    for cells in per_window:
        target = random.randint(1, min(4, len(cells) - 1))
        candidates = [cell for cell in cells if cell not in visible]
        while sum(cell in visible for cell in cells) < target and candidates:
            pick = random.choice(candidates)
            visible.add(pick)
            candidates.remove(pick)

    density = random.uniform(0.2, 0.45)
    for cell in all_cells:
        if cell not in visible and random.random() < density:
            visible.add(cell)

    protected = set(visible)
    for cells in per_window:
        if all(cell in visible for cell in cells):
            removable = [cell for cell in cells if cell not in protected]
            if not removable:
                return None
            visible.remove(random.choice(removable))
        if not any(cell in visible for cell in cells):
            return None
        if all(cell in visible for cell in cells):
            return None
    return visible


def _window_is_ambiguous(
    input_grid: np.ndarray, box: BBox, empty: int, row_period: int, col_period: int
) -> bool:
    r0, c0, r1, c1 = box
    cells = [
        (r, c, int(input_grid[r, c]))
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if int(input_grid[r, c]) != empty
    ]
    for test_row_period in range(1, row_period + 1):
        for test_col_period in range(1, col_period + 1):
            if (test_row_period, test_col_period) == (row_period, col_period):
                continue
            if _period_mapping(cells, test_row_period, test_col_period) is not None:
                return True
    return False


def _build_output(
    height: int, width: int, wall: int, windows: list[BBox], tile: np.ndarray
) -> np.ndarray:
    out = np.full((height, width), wall, dtype=int)
    row_period, col_period = tile.shape
    for r0, c0, r1, c1 in windows:
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                out[r, c] = int(tile[r % row_period, c % col_period])
    return out


def _gen_attempt() -> np.ndarray | None:
    row_period = random.randint(3, 5)
    col_period = random.randint(3, 5)
    min_height = max(12, row_period * 3 + 2)
    min_width = max(12, col_period * 3 + 2)
    height = random.randint(min_height, min(30, min_height + 8))
    width = random.randint(min_width, min(30, min_width + 8))
    if height > 30 or width > 30:
        return None

    colors = list(range(10))
    wall, empty = random.sample(colors, 2)
    palette = [color for color in colors if color not in (wall, empty)]
    tile = _sample_tile(row_period, col_period, palette)

    window_count = random.randint(3, 5)
    windows = _place_windows(height, width, row_period, col_period, window_count)
    if windows is None:
        return None

    output = _build_output(height, width, wall, windows, tile)
    visible = _choose_visible_cells(windows, row_period, col_period)
    if visible is None:
        return None

    input_grid = np.full((height, width), wall, dtype=int)
    for r0, c0, r1, c1 in windows:
        input_grid[r0 : r1 + 1, c0 : c1 + 1] = empty
    for r, c in visible:
        input_grid[r, c] = output[r, c]

    if int(np.count_nonzero(input_grid == wall)) <= (height * width) // 2:
        return None
    if np.array_equal(input_grid, output):
        return None
    for box in windows:
        r0, c0, r1, c1 = box
        if (r1 - r0 + 1) >= row_period or (c1 - c0 + 1) >= col_period:
            return None
        if not _window_is_ambiguous(input_grid, box, empty, row_period, col_period):
            return None
    try:
        if not np.array_equal(main(input_grid), output):
            return None
    except Exception:
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid shared-wallpaper window puzzle")
