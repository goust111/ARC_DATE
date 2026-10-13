from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8

SPRITE_COLORS = [RED, GREEN, YELLOW, PINK, ORANGE, TEAL]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return {(r - r0, c - c0) for r, c in cells}


def _adjacent_cell_edges(cells: set[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r, c in cells:
        for dr, dc in [(1, 0), (0, 1)]:
            nbr = (r + dr, c + dc)
            if nbr in cells:
                a = (r, c)
                b = nbr
                edges.append((a, b))
    return edges


def _random_spanning_tree(cells: set[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    cell_list = list(cells)
    start = random.choice(cell_list)
    seen = {start}
    tree: list[tuple[tuple[int, int], tuple[int, int]]] = []
    while len(seen) < len(cells):
        candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for r, c in list(seen):
            for dr, dc in DIRS4:
                nbr = (r + dr, c + dc)
                if nbr in cells and nbr not in seen:
                    candidates.append(((r, c), nbr))
        assert candidates
        edge = random.choice(candidates)
        tree.append(edge)
        seen.add(edge[1])
    return tree


def _generate_polyomino(n_cells: int, *, max_h: int = 4, max_w: int = 4) -> set[tuple[int, int]] | None:
    for _ in range(200):
        cells = {(0, 0)}
        while len(cells) < n_cells:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in DIRS4:
                    nbr = (r + dr, c + dc)
                    if nbr not in cells:
                        frontier.append(nbr)
            random.shuffle(frontier)
            placed = False
            for nr, nc in frontier:
                trial = _normalize_cells(cells | {(nr, nc)})
                r0, c0, r1, c1 = _bbox(list(trial))
                if r1 + 1 <= max_h and c1 + 1 <= max_w:
                    cells = trial
                    placed = True
                    break
            if not placed:
                break
        if len(cells) != n_cells:
            continue
        norm = _normalize_cells(cells)
        r0, c0, r1, c1 = _bbox(list(norm))
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if area == len(norm):
            continue
        return norm
    return None


def _render_sprite(hole_count: int) -> tuple[np.ndarray, dict]:
    assert 1 <= hole_count <= 4
    min_cells = max(3, hole_count)
    max_cells = min(6, hole_count + 2)
    for _ in range(200):
        n_cells = random.randint(min_cells, max_cells)
        poly = _generate_polyomino(n_cells)
        if poly is None:
            continue
        if len(_adjacent_cell_edges(poly)) < n_cells - 1:
            continue
        tree = _random_spanning_tree(poly)
        open_edges = set()
        need_open = n_cells - hole_count
        for edge in tree[:need_open]:
            open_edges.add(frozenset(edge))

        cell_h = random.randint(2, 4)
        cell_w = random.randint(2, 4)
        poly_h = max(r for r, _ in poly) + 1
        poly_w = max(c for _, c in poly) + 1
        fine_h = poly_h * (cell_h + 1) + 1
        fine_w = poly_w * (cell_w + 1) + 1
        patch = np.zeros((fine_h, fine_w), dtype=int)

        for pr, pc in poly:
            r0 = pr * (cell_h + 1)
            c0 = pc * (cell_w + 1)
            patch[r0, c0 : c0 + cell_w + 2] = 1
            patch[r0 + cell_h + 1, c0 : c0 + cell_w + 2] = 1
            patch[r0 : r0 + cell_h + 2, c0] = 1
            patch[r0 : r0 + cell_h + 2, c0 + cell_w + 1] = 1

        for a, b in _adjacent_cell_edges(poly):
            if frozenset((a, b)) not in open_edges:
                continue
            if a[0] == b[0]:
                pr = a[0]
                left = min(a[1], b[1])
                wall_c = (left + 1) * (cell_w + 1)
                wall_r0 = pr * (cell_h + 1)
                wall_r1 = wall_r0 + cell_h + 1
                patch[wall_r0 + 1 : wall_r1, wall_c] = 0
            else:
                top = min(a[0], b[0])
                pc = a[1]
                wall_r = (top + 1) * (cell_h + 1)
                wall_c0 = pc * (cell_w + 1)
                wall_c1 = wall_c0 + cell_w + 1
                patch[wall_r, wall_c0 + 1 : wall_c1] = 0

        holes = _find_holes_in_patch(patch)
        if len(holes) != hole_count:
            continue
        if not _has_nonrectangular_outline(patch):
            continue
        return patch, {"holes": holes, "cell_h": cell_h, "cell_w": cell_w, "poly": poly}
    raise RuntimeError(f"failed to build sprite with {hole_count} holes")


def _find_holes_in_patch(patch: np.ndarray) -> list[list[tuple[int, int]]]:
    zeros = patch == BLACK
    holes: list[list[tuple[int, int]]] = []
    for comp in _components(zeros):
        if any(
            r == 0 or c == 0 or r == patch.shape[0] - 1 or c == patch.shape[1] - 1
            for r, c in comp
        ):
            continue
        holes.append(comp)
    return holes


def _has_nonrectangular_outline(patch: np.ndarray) -> bool:
    filled = patch != BLACK
    cells = [(r, c) for r, c in zip(*np.where(filled), strict=False)]
    r0, c0, r1, c1 = _bbox(cells)
    bbox_area = (r1 - r0 + 1) * (c1 - c0 + 1)
    return int(filled.sum()) < bbox_area


def _sprite_info(grid: np.ndarray, color: int) -> dict:
    comps = _components(grid == color)
    assert comps, f"missing sprite color {color}"
    assert len(comps) == 1, f"color {color} should form one sprite"
    cells = comps[0]
    r0, c0, r1, c1 = _bbox(cells)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    holes = []
    for comp in _components(sub == BLACK):
        if any(
            r == 0 or c == 0 or r == sub.shape[0] - 1 or c == sub.shape[1] - 1
            for r, c in comp
        ):
            continue
        holes.append([(r + r0, c + c0) for r, c in comp])
    return {
        "color": color,
        "cells": cells,
        "bbox": (r0, c0, r1, c1),
        "holes": holes,
    }


def _analyze_grid(grid: np.ndarray) -> list[dict]:
    colors = [int(v) for v in np.unique(grid) if int(v) not in (BLACK, GREY)]
    infos = [_sprite_info(grid, color) for color in colors]
    infos.sort(key=lambda info: (-len(info["holes"]), info["bbox"][0], info["bbox"][1]))
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep the two hollow sprites with the most enclosed holes, erase the others and all grey paths, then cross-fill each survivor's holes with the other survivor's border color.

    Input structure:
    - Black background contains several separated hollow sprites, each drawn as a 1-pixel-thick single-color border component.
    - Every sprite encloses one or more black 4-connected holes, and different sprites have different hole counts.
    - Thin grey path segments may touch sprite exteriors but never enter any sprite hole.

    Rule:
    - For each sprite, count the black 4-connected regions inside its border that do not touch that sprite's bounding-box boundary.
    - Sort sprites by hole count from highest to lowest and keep only the top two; erase every other sprite and every grey path cell to black.
    - Fill all holes of the first-ranked survivor with the second-ranked survivor's border color, and fill all holes of the second-ranked survivor with the first-ranked survivor's border color.
    - Keep both surviving borders unchanged.

    Key insight:
    - The fill colors are not locally encoded: the hole count determines the global ranking, the ranking determines which two sprites survive, and only then can each survivor's holes inherit the opponent's border color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    infos = _analyze_grid(g)
    assert len(infos) >= 2
    hole_counts = [len(info["holes"]) for info in infos]
    assert len(set(hole_counts)) == len(hole_counts), "hole counts must be unique"

    first, second = infos[:2]
    keep_colors = {first["color"], second["color"]}
    out = g.copy()
    out[out == GREY] = BLACK
    for info in infos[2:]:
        for r, c in info["cells"]:
            out[r, c] = BLACK
    for r, c in [cell for hole in first["holes"] for cell in hole]:
        out[r, c] = second["color"]
    for r, c in [cell for hole in second["holes"] for cell in hole]:
        out[r, c] = first["color"]
    return out


def _can_place(
    occupied: np.ndarray,
    patch_shape: tuple[int, int],
    top: int,
    left: int,
    margin: int = 2,
) -> bool:
    h, w = patch_shape
    H, W = occupied.shape
    if top < 0 or left < 0 or top + h > H or left + w > W:
        return False
    r0 = max(0, top - margin)
    c0 = max(0, left - margin)
    r1 = min(H, top + h + margin)
    c1 = min(W, left + w + margin)
    return not bool(occupied[r0:r1, c0:c1].any())


def _paste_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, color: int) -> None:
    mask = patch != BLACK
    sub = grid[top : top + patch.shape[0], left : left + patch.shape[1]]
    sub[mask] = color


def _sprite_border_neighbors(
    grid: np.ndarray,
    sprite_color: int,
) -> list[tuple[int, int]]:
    coords = list(zip(*np.where(grid == sprite_color), strict=False))
    candidates: set[tuple[int, int]] = set()
    for r, c in coords:
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == BLACK:
                candidates.add((nr, nc))
    return sorted(candidates)


def _shortest_path(
    blocked: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    passable_extra: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int]] | None:
    extra = passable_extra or set()
    q = deque([start])
    prev = {start: None}
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for dr, dc in DIRS4:
            nr, nc = cur[0] + dr, cur[1] + dc
            nxt = (nr, nc)
            if not _in_bounds(blocked.shape, nr, nc) or nxt in prev:
                continue
            if blocked[nr, nc] and nxt not in extra and nxt != goal:
                continue
            prev[nxt] = cur
            q.append(nxt)
    if goal not in prev:
        return None
    path = []
    cur: tuple[int, int] | None = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _draw_paths(grid: np.ndarray, sprite_colors: list[int]) -> bool:
    blocked = grid != BLACK
    segment_count = random.choice([1, 1, 2])
    chosen = random.sample(sprite_colors, random.choice([2, 3]))
    order = chosen[:]
    random.shuffle(order)
    pairs = [(order[i], order[i + 1]) for i in range(len(order) - 1)]
    if segment_count == 1:
        pairs = pairs[:1]
    drawn_any = False
    grey_cells: set[tuple[int, int]] = set()

    for color_a, color_b in pairs:
        starts = _sprite_border_neighbors(grid, color_a)
        goals = _sprite_border_neighbors(grid, color_b)
        random.shuffle(starts)
        random.shuffle(goals)
        best: list[tuple[int, int]] | None = None
        for start in starts[:20]:
            for goal in goals[:20]:
                path = _shortest_path(blocked, start, goal, passable_extra=grey_cells)
                if path is None or len(path) < 2:
                    continue
                if best is None or len(path) < len(best):
                    best = path
            if best is not None:
                break
        if best is None:
            return False
        for r, c in best:
            if int(grid[r, c]) == BLACK:
                grid[r, c] = GREY
                blocked[r, c] = True
                grey_cells.add((r, c))
        drawn_any = True
    return drawn_any and bool(grey_cells)


def _expected_output(grid: np.ndarray) -> np.ndarray:
    return main(grid)


def _validate_instance(inp: np.ndarray) -> None:
    infos = _analyze_grid(inp)
    assert len(infos) == 4
    hole_counts = [len(info["holes"]) for info in infos]
    assert hole_counts == [4, 3, 2, 1], hole_counts
    assert int((inp == GREY).sum()) > 0
    out = _expected_output(inp)
    remaining = sorted(int(v) for v in np.unique(out) if int(v) not in (BLACK, GREY))
    survivor_colors = sorted([infos[0]["color"], infos[1]["color"]])
    assert remaining == survivor_colors
    assert int((out == GREY).sum()) == 0

    for loser in infos[2:]:
        assert not np.any(out == loser["color"])

    first, second = infos[:2]
    for hole in first["holes"]:
        vals = {int(out[r, c]) for r, c in hole}
        assert vals == {second["color"]}
    for hole in second["holes"]:
        vals = {int(out[r, c]) for r, c in hole}
        assert vals == {first["color"]}

    alt_self_fill = out.copy()
    for hole in first["holes"]:
        for r, c in hole:
            alt_self_fill[r, c] = first["color"]
    for hole in second["holes"]:
        for r, c in hole:
            alt_self_fill[r, c] = second["color"]
    assert not np.array_equal(alt_self_fill, out)

    alt_keep_third = out.copy()
    third = infos[2]
    for r, c in third["cells"]:
        alt_keep_third[r, c] = third["color"]
    assert not np.array_equal(alt_keep_third, out)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        H = random.randint(22, 28)
        W = random.randint(22, 28)
        grid = np.zeros((H, W), dtype=int)
        occupied = np.zeros((H, W), dtype=bool)
        hole_counts = [1, 2, 3, 4]
        random.shuffle(hole_counts)
        colors = random.sample(SPRITE_COLORS, 4)

        placements: list[tuple[int, int, np.ndarray, int]] = []
        ok = True
        patches_by_color: dict[int, np.ndarray] = {}
        for hole_count, color in sorted(zip(hole_counts, colors), reverse=True):
            patch, _meta = _render_sprite(hole_count)
            if patch.shape[0] > H - 2 or patch.shape[1] > W - 2:
                ok = False
                break
            placed = False
            for _ in range(300):
                top = random.randint(1, H - patch.shape[0] - 1)
                left = random.randint(1, W - patch.shape[1] - 1)
                if _can_place(occupied, patch.shape, top, left, margin=2):
                    placements.append((top, left, patch, color))
                    occupied[top : top + patch.shape[0], left : left + patch.shape[1]] |= patch != BLACK
                    patches_by_color[color] = patch
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        for top, left, patch, color in placements:
            _paste_patch(grid, patch, top, left, color)

        sprite_colors = [color for _top, _left, _patch, color in placements]
        if not _draw_paths(grid, sprite_colors):
            continue

        try:
            _validate_instance(grid)
        except Exception:
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
