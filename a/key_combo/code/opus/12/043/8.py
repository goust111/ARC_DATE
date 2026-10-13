from __future__ import annotations

from collections import Counter, deque

import numpy as np


BLACK = 0
BLUE = 1
YELLOW = 4
MARKER_PALETTE = (2, 3, 6, 7, 8, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

GEN_HEIGHT_RANGE = (18, 25)
GEN_WIDTH_RANGE = (18, 25)
GEN_RECT_COUNT_CHOICES = (1, 2)
GEN_RECT_HEIGHT_RANGE = (8, 12)
GEN_RECT_WIDTH_RANGE = (8, 12)
GEN_RECT_GAP = 3
GEN_TOP_WALL_ROW_RANGE = (2, 5)
GEN_MAX_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _neighbors8(r: int, c: int):
    for dr, dc in DIR8:
        yield r + dr, c + dc


def _blue_roles(grid: np.ndarray) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    dots: set[tuple[int, int]] = set()
    walls: set[tuple[int, int]] = set()

    for r in range(h):
        for c in range(w):
            if grid[r, c] != BLUE or seen[r, c]:
                continue
            comp = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and grid[nr, nc] == BLUE:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if len(comp) == 1:
                dots.add(comp[0])
            else:
                walls.update(comp)
    return dots, walls


def _find_rectangles(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    yellow_cells = {(int(r), int(c)) for r, c in np.argwhere(grid == YELLOW)}
    rows = sorted({r for r, _ in yellow_cells})
    cols = sorted({c for _, c in yellow_cells})
    rects = set()
    for i, r0 in enumerate(rows):
        for r1 in rows[i + 1 :]:
            for j, c0 in enumerate(cols):
                for c1 in cols[j + 1 :]:
                    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
                    if corners.issubset(yellow_cells):
                        rects.add((r0, c0, r1, c1))
    return sorted(rects)


def _draw_rectangle_edges(
    grid: np.ndarray, rects: list[tuple[int, int, int, int]]
) -> tuple[np.ndarray, set[tuple[int, int]]]:
    out = np.array(grid, copy=True, dtype=int)
    edge_cells: set[tuple[int, int]] = set()
    for r0, c0, r1, c1 in rects:
        for r in range(r0, r1 + 1):
            edge_cells.add((r, c0))
            edge_cells.add((r, c1))
        for c in range(c0, c1 + 1):
            edge_cells.add((r0, c))
            edge_cells.add((r1, c))
    for r, c in edge_cells:
        if out[r, c] == BLACK:
            out[r, c] = YELLOW
    return out, edge_cells


def _pick_winner(adjacent_markers: list[tuple[int, int, int]]) -> tuple[int | None, bool]:
    if not adjacent_markers:
        return None, False
    counts = Counter(color for _, _, color in adjacent_markers)
    max_count = max(counts.values())
    winners = {color for color, count in counts.items() if count == max_count}
    if len(winners) == 1:
        return next(iter(winners)), False
    earliest = min((r, c, color) for r, c, color in adjacent_markers if color in winners)
    return earliest[2], True


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    rects = _find_rectangles(g)
    step1, _ = _draw_rectangle_edges(g, rects)
    blue_dots, wall_cells = _blue_roles(g)
    marker_cells = {
        (int(r), int(c))
        for r, c in np.argwhere((step1 != BLACK) & (step1 != BLUE) & (step1 != YELLOW))
    }

    passable = step1 == BLACK
    for r, c in blue_dots:
        if step1[r, c] == BLUE:
            passable[r, c] = True

    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    regions = []
    for r in range(h):
        for c in range(w):
            if not passable[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            dots = []
            adj_seen = set()
            adjacent_markers = []

            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                if (cr, cc) in blue_dots:
                    dots.append((cr, cc))

                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(g.shape, nr, nc) and passable[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))

                for nr, nc in _neighbors8(cr, cc):
                    if (nr, nc) in marker_cells and (nr, nc) not in adj_seen:
                        adj_seen.add((nr, nc))
                        adjacent_markers.append((nr, nc, int(step1[nr, nc])))

            inside_rect = None
            for idx, (r0, c0, r1, c1) in enumerate(rects):
                if all(r0 <= rr <= r1 and c0 <= cc <= c1 for rr, cc in coords):
                    inside_rect = idx
                    break

            winner, used_tie = _pick_winner(adjacent_markers)
            regions.append(
                {
                    "coords": coords,
                    "dots": dots,
                    "inside_rect": inside_rect,
                    "adjacent_markers": adjacent_markers,
                    "winner": winner,
                    "used_tie": used_tie,
                }
            )

    return {
        "grid": g,
        "rectangles": rects,
        "step1": step1,
        "blue_dots": blue_dots,
        "wall_cells": wall_cells,
        "marker_cells": marker_cells,
        "regions": regions,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor isolated blue dots only inside yellow-corner rectangles using the marker vote of their wall-bounded region.

    Input structure:
    - Black is the background.
    - Blue cells appear as multi-cell wall segments and isolated single-dot cells.
    - Yellow cells appear in groups of four aligned corners; other non-black/non-blue/non-yellow cells are colored markers placed on wall endpoints or corners.

    Rule:
    - First detect every four-corner yellow rectangle and draw its full yellow border without overwriting existing non-background cells.
    - Treat multi-cell blue wall segments, colored markers, and the yellow rectangle borders as barriers; flood-fill the remaining black cells together with isolated blue dots into 4-connected open regions.
    - For each region whose entire extent lies inside one detected rectangle, count the adjacent colored markers in the 8-neighborhood and recolor every isolated blue dot in that region to the majority marker color; ties use the tied marker with smallest row, then smallest column.
    - Regions outside all rectangles keep their isolated blue dots unchanged, and all walls, markers, yellow corners, and yellow borders stay fixed.

    Key insight:
    - The yellow rectangles are not fill targets: they are spatial filters that decide which flood-filled wall regions are eligible for marker-vote recoloring.
    """
    analysis = _analyze(np.asarray(grid, dtype=int))
    out = np.array(analysis["step1"], copy=True, dtype=int)

    for region in analysis["regions"]:
        if region["inside_rect"] is None:
            continue
        if not region["dots"] or region["winner"] is None:
            continue
        for r, c in region["dots"]:
            out[r, c] = region["winner"]
    return out


def _rects_far_apart(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    separated = ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0
    if not separated:
        return False
    if {ar0, ar1} & {br0, br1}:
        return False
    if {ac0, ac1} & {bc0, bc1}:
        return False
    return True


def _choose_rectangles(
    rng: np.random.Generator, h: int, w: int, rect_count: int, top_wall_row: int
) -> list[tuple[int, int, int, int]] | None:
    rects: list[tuple[int, int, int, int]] = []
    top_min = top_wall_row + 3
    for _ in range(120):
        rects.clear()
        ok = True
        for _rect_idx in range(rect_count):
            placed = False
            for _ in range(120):
                rh = int(rng.integers(*GEN_RECT_HEIGHT_RANGE))
                rw = int(rng.integers(*GEN_RECT_WIDTH_RANGE))
                if top_min + rh + 1 >= h or rw + 2 >= w:
                    return None
                r0 = int(rng.integers(top_min, h - rh - 1))
                c0 = int(rng.integers(1, w - rw - 1))
                rect = (r0, c0, r0 + rh - 1, c0 + rw - 1)
                if all(_rects_far_apart(rect, other, GEN_RECT_GAP) for other in rects):
                    rects.append(rect)
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok:
            return sorted(rects)
    return None


def _build_rectangle_pattern(
    rng: np.random.Generator, rect: tuple[int, int, int, int]
) -> dict:
    r0, c0, r1, c1 = rect
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    patterns = ["cross", "double_v", "double_h"]
    if inner_w < 6:
        patterns.remove("double_v")
    if inner_h < 6:
        patterns.remove("double_h")
    pattern = str(rng.choice(patterns))

    wall_cells: set[tuple[int, int]] = set()
    markers: list[tuple[int, int]] = []

    if pattern == "cross":
        hr = int(rng.integers(r0 + 3, r1 - 1))
        vc = int(rng.integers(c0 + 3, c1 - 1))
        for r in range(r0 + 1, r1):
            wall_cells.add((r, vc))
        for c in range(c0 + 1, c1):
            wall_cells.add((hr, c))
        markers = [(r0 + 1, vc), (hr, c0 + 1), (hr, c1 - 1), (r1 - 1, vc)]
    elif pattern == "double_v":
        vc1 = int(rng.integers(c0 + 2, c1 - 4))
        vc2 = int(rng.integers(vc1 + 3, c1 - 1))
        for r in range(r0 + 1, r1):
            wall_cells.add((r, vc1))
            wall_cells.add((r, vc2))
        markers = [(r0 + 1, vc1), (r1 - 1, vc1), (r0 + 1, vc2), (r1 - 1, vc2)]
    else:
        hr1 = int(rng.integers(r0 + 2, r1 - 4))
        hr2 = int(rng.integers(hr1 + 3, r1 - 1))
        for c in range(c0 + 1, c1):
            wall_cells.add((hr1, c))
            wall_cells.add((hr2, c))
        markers = [(hr1, c0 + 1), (hr1, c1 - 1), (hr2, c0 + 1), (hr2, c1 - 1)]

    return {"rect": rect, "pattern": pattern, "wall_cells": wall_cells, "markers": markers}


def _assign_marker_colors(
    rng: np.random.Generator, structures: list[dict], palette: tuple[int, ...]
) -> dict[tuple[int, int], int]:
    colors: dict[tuple[int, int], int] = {}
    distinct = list(rng.choice(np.array(palette), size=min(4, len(palette)), replace=False))
    if len(distinct) < 3:
        distinct = list(palette[:3])
    a, b, c = distinct[0], distinct[1], distinct[2]

    for info in structures:
        markers = info["markers"]
        if info["pattern"] == "cross":
            recipe = [a, a, b, b]
        elif info["pattern"] == "double_v":
            recipe = [a, a, b, c]
        else:
            recipe = [a, a, b, c]
        if rng.random() < 0.5:
            recipe = [recipe[0], recipe[2], recipe[1], recipe[3]]
        for pos, color in zip(markers, recipe):
            colors[pos] = int(color)

    return colors


def _dot_ok(grid: np.ndarray, cell: tuple[int, int], chosen: set[tuple[int, int]]) -> bool:
    if cell in chosen:
        return False
    if grid[cell] != BLACK:
        return False
    for nr, nc in _neighbors4(*cell):
        if _in_bounds(grid.shape, nr, nc):
            if grid[nr, nc] == BLUE or (nr, nc) in chosen:
                return False
    return True


def _choose_dots_from_region(
    rng: np.random.Generator,
    grid: np.ndarray,
    coords: list[tuple[int, int]],
    chosen: set[tuple[int, int]],
    count: int,
) -> list[tuple[int, int]] | None:
    cells = list(coords)
    rng.shuffle(cells)
    out = []
    for cell in cells:
        if _dot_ok(grid, cell, chosen | set(out)):
            out.append(cell)
            if len(out) == count:
                return out
    return None


def _generate_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    h = int(rng.integers(*GEN_HEIGHT_RANGE))
    w = int(rng.integers(*GEN_WIDTH_RANGE))
    top_wall_row = int(rng.integers(*GEN_TOP_WALL_ROW_RANGE))
    rect_count = int(rng.choice(GEN_RECT_COUNT_CHOICES))
    rects = _choose_rectangles(rng, h, w, rect_count, top_wall_row)
    if rects is None:
        return None

    grid = np.zeros((h, w), dtype=int)

    outside_markers = [(top_wall_row, 0), (top_wall_row, w - 1)]
    for c in range(1, w - 1):
        grid[top_wall_row, c] = BLUE

    structures = []
    wall_cells = set()
    for rect in rects:
        info = _build_rectangle_pattern(rng, rect)
        structures.append(info)
        wall_cells.update(info["wall_cells"])

    for r, c in wall_cells:
        grid[r, c] = BLUE

    for r0, c0, r1, c1 in rects:
        for cell in ((r0, c0), (r0, c1), (r1, c0), (r1, c1)):
            grid[cell] = YELLOW

    palette = tuple(rng.choice(np.array(MARKER_PALETTE), size=int(rng.integers(3, 5)), replace=False))
    marker_colors = _assign_marker_colors(rng, structures, palette)
    outside_color_choices = list(rng.choice(np.array(palette), size=2, replace=True))
    marker_colors[outside_markers[0]] = int(outside_color_choices[0])
    marker_colors[outside_markers[1]] = int(outside_color_choices[1])

    for pos, color in marker_colors.items():
        grid[pos] = color

    analysis = _analyze(grid)
    if len(analysis["rectangles"]) != rect_count:
        return None

    inside_by_rect = {idx: [] for idx in range(rect_count)}
    outside_regions = []
    for region in analysis["regions"]:
        idx = region["inside_rect"]
        if idx is None:
            outside_regions.append(region)
        else:
            inside_by_rect[idx].append(region)

    if any(len(regions) < 2 for regions in inside_by_rect.values()):
        return None
    if not outside_regions:
        return None
    if any(not region["adjacent_markers"] for region in sum(inside_by_rect.values(), [])):
        return None

    chosen_dots: set[tuple[int, int]] = set()
    for regions in inside_by_rect.values():
        for region in regions:
            want = 1 + int(len(region["coords"]) >= 16 and rng.random() < 0.45)
            dots = _choose_dots_from_region(rng, grid, region["coords"], chosen_dots, want)
            if dots is None:
                return None
            chosen_dots.update(dots)

    outside_with_votes = [region for region in outside_regions if region["adjacent_markers"]]
    if not outside_with_votes:
        return None
    rng.shuffle(outside_with_votes)
    outside_targets = outside_with_votes[: 1 + int(rng.random() < 0.5)]
    for region in outside_targets:
        want = 1 + int(len(region["coords"]) >= 20 and rng.random() < 0.7)
        dots = _choose_dots_from_region(rng, grid, region["coords"], chosen_dots, want)
        if dots is None:
            return None
        chosen_dots.update(dots)

    for r, c in chosen_dots:
        if grid[r, c] != BLACK:
            return None
        grid[r, c] = BLUE

    analysis = _analyze(grid)
    out = main(grid)

    inside_regions = [region for region in analysis["regions"] if region["inside_rect"] is not None]
    outside_regions = [region for region in analysis["regions"] if region["inside_rect"] is None]
    if not inside_regions or not outside_regions:
        return None
    if any(not region["dots"] for region in inside_regions):
        return None
    if any(not region["adjacent_markers"] for region in inside_regions):
        return None

    if len(analysis["blue_dots"]) != len(chosen_dots) or analysis["blue_dots"] != chosen_dots:
        return None

    tie_regions = [region for region in inside_regions if region["used_tie"]]
    strict_regions = [region for region in inside_regions if not region["used_tie"]]
    if not tie_regions or not strict_regions:
        return None

    recolored_inside = {
        int(out[r, c])
        for region in inside_regions
        for r, c in region["dots"]
    }
    if not recolored_inside or recolored_inside == {BLUE}:
        return None

    unchanged_outside = [
        (r, c)
        for region in outside_regions
        for r, c in region["dots"]
        if int(out[r, c]) == BLUE
    ]
    if not unchanged_outside:
        return None

    if np.array_equal(grid, out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
