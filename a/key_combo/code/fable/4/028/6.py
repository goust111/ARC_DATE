from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BACKGROUND = 0
GREY = 5
RAY_MODES = ("ghost", "local", "edge")
OBSTACLE_RECT_SHAPES = ((1, 1), (1, 2), (2, 1), (2, 2))
GEN_RETRY_LIMIT = 300

_MODE_BAG: list[str] = []


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    ]


def _find_divider(grid: np.ndarray, bg: int) -> tuple[str, int, int]:
    row_hits = []
    for r in range(grid.shape[0]):
        color = int(grid[r, 0])
        if color != bg and np.all(grid[r, :] == color):
            row_hits.append((r, color))

    col_hits = []
    for c in range(grid.shape[1]):
        color = int(grid[0, c])
        if color != bg and np.all(grid[:, c] == color):
            col_hits.append((c, color))

    assert (len(row_hits) == 1) != (len(col_hits) == 1), "expected exactly one full divider line"
    if row_hits:
        idx, color = row_hits[0]
        return "horizontal", int(idx), int(color)
    idx, color = col_hits[0]
    return "vertical", int(idx), int(color)


def _panel_offsets(shape: tuple[int, int], orientation: str, divider_idx: int) -> list[tuple[int, int]]:
    if orientation == "vertical":
        return [(0, 0), (0, divider_idx + 1)]
    return [(0, 0), (divider_idx + 1, 0)]


def _split_panels(grid: np.ndarray, orientation: str, divider_idx: int) -> tuple[np.ndarray, np.ndarray]:
    if orientation == "vertical":
        left = grid[:, :divider_idx]
        right = grid[:, divider_idx + 1 :]
        assert left.shape == right.shape
        return left, right

    top = grid[:divider_idx, :]
    bottom = grid[divider_idx + 1 :, :]
    assert top.shape == bottom.shape
    return top, bottom


def _mirror_coord(coord: tuple[int, int], panel_shape: tuple[int, int], orientation: str) -> tuple[int, int]:
    r, c = coord
    h, w = panel_shape
    if orientation == "vertical":
        return r, w - 1 - c
    return h - 1 - r, c


def _mirror_mask(mask: np.ndarray, orientation: str) -> np.ndarray:
    return np.fliplr(mask) if orientation == "vertical" else np.flipud(mask)


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors8(cr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            out.append(comp)
    return out


def _find_arrow(
    grid: np.ndarray,
    bg: int,
    divider_color: int,
) -> tuple[int, int, list[tuple[int, int]], tuple[int, int], tuple[int, int], tuple[int, int]]:
    color_totals = {color: int(np.sum(grid == color)) for color in set(map(int, grid.ravel()))}
    candidates = []
    for color in sorted(set(map(int, grid.ravel()))):
        if color in (bg, divider_color):
            continue
        comps = _connected_components(grid == color)
        for comp in comps:
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            if len(comp) != 3 or (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
                continue
            bbox = grid[r0 : r1 + 1, c0 : c1 + 1]
            empty = np.argwhere(bbox == bg)
            if empty.shape != (1, 2):
                continue
            mr, mc = map(int, empty[0])
            dr = -1 if mr == 0 else 1
            dc = -1 if mc == 0 else 1
            start = (r0 + mr + dr, c0 + mc + dc)
            candidates.append((color_totals[color], color, r0, c0, list(comp), (mr, mc), (dr, dc), start))

    exact_color_candidates = [cand for cand in candidates if cand[0] == 3]
    if len(exact_color_candidates) == 1:
        _, color, r0, c0, cells, missing, direction, start = exact_color_candidates[0]
        return int(color), int(r0), int(c0), cells, missing, direction, start

    assert len(candidates) == 1, "expected exactly one directional arrowhead"
    _, color, r0, c0, cells, missing, direction, start = candidates[0]
    return int(color), int(r0), int(c0), cells, missing, direction, start


def _trace_ray(start: tuple[int, int], direction: tuple[int, int], blocked: np.ndarray) -> list[tuple[int, int]]:
    dr, dc = direction
    r, c = start
    path: list[tuple[int, int]] = []
    while _in_bounds(blocked.shape, r, c) and not blocked[r, c]:
        path.append((r, c))
        r += dr
        c += dc
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the arrow ray through one panel while treating mirrored obstacles from the other panel as real blockers.

    Input structure:
    - A full gray divider row or column splits the grid into two equal mirrored panels.
    - Exactly one panel contains a monochrome 2x2 arrowhead with one missing corner indicating a diagonal firing direction.
    - Both panels contain solid obstacle blocks on a black background.

    Rule:
    - Determine the arrow's outgoing diagonal from the missing corner, and start the ray one step beyond that corner.
    - March cell by cell until leaving the panel or hitting an occupied cell, where occupied means either a local obstacle or an obstacle in the opposite panel at the mirrored corresponding position.
    - Color the traversed cells with the arrow color, and also color the mirrored copy of the same path in the other panel.

    Key insight:
    - The two panels depict the same obstacle field under a mirror, so empty space is only truly passable where both a panel cell and its mirrored counterpart are empty.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    orientation, divider_idx, divider_color = _find_divider(g, bg)
    panels = _split_panels(g, orientation, divider_idx)
    panel_shape = panels[0].shape
    offsets = _panel_offsets(g.shape, orientation, divider_idx)

    arrow_color, ar0, ac0, arrow_cells, _missing, direction, _start = _find_arrow(g, bg, divider_color)

    source_idx = None
    source_local_cells: list[tuple[int, int]] | None = None
    source_local_start: tuple[int, int] | None = None
    for idx, (pr0, pc0) in enumerate(offsets):
        h, w = panel_shape
        if pr0 <= ar0 < pr0 + h and pc0 <= ac0 < pc0 + w:
            source_idx = idx
            source_local_cells = [(r - pr0, c - pc0) for r, c in arrow_cells]
            source_local_start = (_start[0] - pr0, _start[1] - pc0)
            break
    assert source_idx is not None and source_local_cells is not None and source_local_start is not None
    other_idx = 1 - source_idx

    source_panel = panels[source_idx]
    other_panel = panels[other_idx]
    source_obstacles = source_panel != bg
    other_obstacles = other_panel != bg
    for r, c in source_local_cells:
        source_obstacles[r, c] = False
    blocked = source_obstacles | _mirror_mask(other_obstacles, orientation)

    path = _trace_ray(source_local_start, direction, blocked)
    mirrored_path = [_mirror_coord(cell, panel_shape, orientation) for cell in path]

    out = g.copy()
    spr0, spc0 = offsets[source_idx]
    opr0, opc0 = offsets[other_idx]
    for r, c in path:
        out[spr0 + r, spc0 + c] = arrow_color
    for r, c in mirrored_path:
        out[opr0 + r, opc0 + c] = arrow_color
    return out


def _next_mode() -> str:
    global _MODE_BAG
    if not _MODE_BAG:
        _MODE_BAG = random.sample(list(RAY_MODES), k=len(RAY_MODES))
    return _MODE_BAG.pop()


def _line_to_edge(
    start: tuple[int, int],
    direction: tuple[int, int],
    panel_shape: tuple[int, int],
) -> list[tuple[int, int]]:
    dr, dc = direction
    r, c = start
    out = []
    while _in_bounds(panel_shape, r, c):
        out.append((r, c))
        r += dr
        c += dc
    return out


def _arrow_candidates(panel_shape: tuple[int, int], direction: tuple[int, int], min_visible: int) -> list[tuple[int, int, list[tuple[int, int]]]]:
    h, w = panel_shape
    dr, dc = direction
    mr = 0 if dr < 0 else 1
    mc = 0 if dc < 0 else 1
    candidates = []
    for r0 in range(h - 1):
        for c0 in range(w - 1):
            start = (r0 + mr + dr, c0 + mc + dc)
            line = _line_to_edge(start, direction, panel_shape)
            if len(line) >= min_visible:
                candidates.append((r0, c0, line))
    return candidates


def _build_arrow_patch(color: int, direction: tuple[int, int]) -> np.ndarray:
    patch = np.full((2, 2), color, dtype=int)
    dr, dc = direction
    patch[0 if dr < 0 else 1, 0 if dc < 0 else 1] = BACKGROUND
    return patch


def _add_rect(mask: np.ndarray, forbidden: np.ndarray) -> bool:
    h, w = mask.shape
    for _ in range(80):
        rh, rw = random.choice(OBSTACLE_RECT_SHAPES)
        r0 = random.randint(0, h - rh)
        c0 = random.randint(0, w - rw)
        if mask[r0 : r0 + rh, c0 : c0 + rw].any():
            continue
        if forbidden[r0 : r0 + rh, c0 : c0 + rw].any():
            continue
        mask[r0 : r0 + rh, c0 : c0 + rw] = True
        return True
    return False


def _global_from_local(
    local_mask: np.ndarray,
    orientation: str,
    divider_idx: int,
    panel_idx: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    if orientation == "vertical":
        offset = (0, 0) if panel_idx == 0 else (0, divider_idx + 1)
    else:
        offset = (0, 0) if panel_idx == 0 else (divider_idx + 1, 0)
    return local_mask, offset


def _paint_panel(global_grid: np.ndarray, panel: np.ndarray, offset: tuple[int, int], transparent: int | None = None) -> None:
    r0, c0 = offset
    h, w = panel.shape
    if transparent is None:
        global_grid[r0 : r0 + h, c0 : c0 + w] = panel
        return
    sub = global_grid[r0 : r0 + h, c0 : c0 + w]
    mask = panel != transparent
    sub[mask] = panel[mask]


def _gen_attempt() -> np.ndarray | None:
    mode = _next_mode()
    orientation = random.choice(("vertical", "horizontal"))
    if orientation == "vertical":
        panel_shape = (random.randint(8, 12), random.randint(7, 12))
        divider_idx = panel_shape[1]
        grid_shape = (panel_shape[0], panel_shape[1] * 2 + 1)
    else:
        panel_shape = (random.randint(7, 12), random.randint(8, 12))
        divider_idx = panel_shape[0]
        grid_shape = (panel_shape[0] * 2 + 1, panel_shape[1])

    assert max(grid_shape) <= 30

    divider_color = GREY
    arrow_color, obstacle_color = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 2)
    direction = random.choice(((-1, -1), (-1, 1), (1, -1), (1, 1)))
    min_visible = 4
    candidates = _arrow_candidates(panel_shape, direction, min_visible)
    if not candidates:
        return None
    arrow_r0, arrow_c0, edge_line = random.choice(candidates)

    source_idx = random.randint(0, 1)
    other_idx = 1 - source_idx
    source_mask = np.zeros(panel_shape, dtype=bool)
    other_mask = np.zeros(panel_shape, dtype=bool)

    arrow_cells = np.ones((2, 2), dtype=bool)
    arrow_cells[0 if direction[0] < 0 else 1, 0 if direction[1] < 0 else 1] = False
    arrow_forbidden = np.zeros(panel_shape, dtype=bool)
    arrow_forbidden[arrow_r0 : arrow_r0 + 2, arrow_c0 : arrow_c0 + 2] = True

    path_cells = edge_line
    mirror_path_cells = [_mirror_coord(cell, panel_shape, orientation) for cell in path_cells]
    source_forbidden = arrow_forbidden.copy()
    other_forbidden = np.zeros(panel_shape, dtype=bool)
    for r, c in path_cells:
        source_forbidden[r, c] = True
    for r, c in mirror_path_cells:
        other_forbidden[r, c] = True

    if mode in ("local", "ghost"):
        max_blocker_idx = len(path_cells) - 1
        if max_blocker_idx < 3:
            return None
        blocker_idx = random.randint(3, max_blocker_idx)
        blocker_local = path_cells[blocker_idx]
        blocker_mirror = mirror_path_cells[blocker_idx]
        if mode == "local":
            source_mask[blocker_local] = True
        else:
            other_mask[blocker_mirror] = True

    source_rect_target = random.randint(1, 3)
    other_rect_target = random.randint(1, 3)
    source_existing = int(source_mask.any())
    other_existing = int(other_mask.any())
    for _ in range(max(0, source_rect_target - source_existing)):
        if not _add_rect(source_mask, source_forbidden):
            break
    for _ in range(max(0, other_rect_target - other_existing)):
        if not _add_rect(other_mask, other_forbidden):
            break

    if not source_mask.any() or not other_mask.any():
        return None
    if np.array_equal(source_mask, _mirror_mask(other_mask, orientation)):
        if not _add_rect(source_mask, source_forbidden):
            return None

    source_panel = np.full(panel_shape, BACKGROUND, dtype=int)
    other_panel = np.full(panel_shape, BACKGROUND, dtype=int)
    source_panel[source_mask] = obstacle_color
    other_panel[other_mask] = obstacle_color
    source_panel[arrow_r0 : arrow_r0 + 2, arrow_c0 : arrow_c0 + 2] = _build_arrow_patch(arrow_color, direction)

    if source_idx == 0:
        panel0 = source_panel
        panel1 = other_panel
    else:
        panel0 = other_panel
        panel1 = source_panel

    grid = np.full(grid_shape, BACKGROUND, dtype=int)
    if orientation == "vertical":
        grid[:, divider_idx] = divider_color
    else:
        grid[divider_idx, :] = divider_color
    _paint_panel(grid, panel0, (0, 0))
    _paint_panel(grid, panel1, (0, divider_idx + 1) if orientation == "vertical" else (divider_idx + 1, 0))

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-ghost-obstacle input")
