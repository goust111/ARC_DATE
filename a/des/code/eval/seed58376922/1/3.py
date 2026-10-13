from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np


GREY = 5
BG = 0
MIN_ROOM_SPAN = 4
GEN_RETRIES = 400
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
_RNG = random.Random()


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _scan_corner_legend(
    grid: np.ndarray,
    bg: int,
    row_order: list[int],
    color_col: int,
    grey_col: int,
    reverse_rows: bool,
) -> tuple[list[int], np.ndarray] | None:
    colors: list[int] = []
    cells: list[tuple[int, int]] = []
    for r in row_order:
        left = int(grid[r, color_col])
        right = int(grid[r, grey_col])
        if left != bg and left != GREY and right == GREY:
            colors.append(left)
            cells.append((r, color_col))
            cells.append((r, grey_col))
        else:
            break
    if len(colors) < 2 or len(set(colors)) != len(colors):
        return None
    if reverse_rows:
        colors = colors[::-1]
    mask = np.zeros_like(grid, dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return colors, mask


def _find_legend(grid: np.ndarray, bg: int) -> tuple[list[int], np.ndarray]:
    h, w = grid.shape
    assert w >= 2
    candidates: list[tuple[int, list[int], np.ndarray]] = []
    for row_order, color_col, grey_col, reverse_rows in (
        (list(range(h)), 0, 1, False),
        (list(range(h)), w - 2, w - 1, False),
        (list(range(h - 1, -1, -1)), 0, 1, True),
        (list(range(h - 1, -1, -1)), w - 2, w - 1, True),
    ):
        cand = _scan_corner_legend(grid, bg, row_order, color_col, grey_col, reverse_rows)
        if cand is not None:
            colors, mask = cand
            candidates.append((len(colors), colors, mask))
    assert candidates, "expected a corner legend"
    _, colors, mask = max(candidates, key=lambda item: item[0])
    return colors, mask


def _side_color_count(grid: np.ndarray, color: int, side_cells: list[tuple[int, int]]) -> int:
    return sum(int(grid[r, c]) == color for r, c in side_cells)


def _room_from_component(
    grid: np.ndarray,
    bg: int,
    coords: np.ndarray,
    max_side_holes: int,
) -> dict[str, object] | None:
    color = int(grid[tuple(coords[0])])
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    coord_set = {tuple(map(int, pt)) for pt in coords}
    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    if not corners.issubset(coord_set):
        return None

    for r, c in coord_set:
        if r0 < r < r1 and c0 < c < c1:
            return None

    top = [(r0, c) for c in range(c0, c1 + 1)]
    bottom = [(r1, c) for c in range(c0, c1 + 1)]
    left = [(r, c0) for r in range(r0, r1 + 1)]
    right = [(r, c1) for r in range(r0, r1 + 1)]
    for side_cells in (top, bottom, left, right):
        values = {int(grid[r, c]) for r, c in side_cells}
        if not values.issubset({bg, color}):
            return None
        if _side_color_count(grid, color, side_cells) < len(side_cells) - max_side_holes:
            return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if interior.size == 0 or np.any(interior != bg):
        return None

    return {
        "color": color,
        "bbox": (r0, c0, r1, c1),
        "area": int(interior.size),
    }


def _extract_rooms(
    grid: np.ndarray,
    bg: int,
    legend_mask: np.ndarray,
    expected_rooms: int,
) -> list[dict[str, object]]:
    rooms: list[dict[str, object]] = []
    max_side_holes = max(1, expected_rooms - 1)
    colors = sorted(set(map(int, grid[~legend_mask].flatten())) - {bg, GREY})
    for color in colors:
        comps = _components4((grid == color) & ~legend_mask)
        for comp in comps:
            room = _room_from_component(grid, bg, comp, max_side_holes)
            if room is not None:
                rooms.append(room)

    assert len(rooms) == expected_rooms, "room count must match legend length"
    areas = [int(room["area"]) for room in rooms]
    assert len(set(areas)) == len(areas), "room areas must be unique"
    return rooms


def _perimeter_lookup(rooms: list[dict[str, object]]) -> dict[tuple[int, int], tuple[int, str]]:
    lookup: dict[tuple[int, int], tuple[int, str]] = {}
    for idx, room in enumerate(rooms):
        r0, c0, r1, c1 = room["bbox"]  # type: ignore[misc]
        for r in range(r0 + 1, r1):
            lookup[(r, c0)] = (idx, "left")
            lookup[(r, c1)] = (idx, "right")
        for c in range(c0 + 1, c1):
            lookup[(r0, c)] = (idx, "top")
            lookup[(r1, c)] = (idx, "bottom")
    return lookup


def _room_doors(
    grid: np.ndarray,
    rooms: list[dict[str, object]],
) -> tuple[list[list[tuple[int, int]]], dict[int, set[int]]]:
    h, w = grid.shape
    lookup = _perimeter_lookup(rooms)
    opposite = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
    outward = {"left": (0, -1), "right": (0, 1), "top": (-1, 0), "bottom": (1, 0)}
    room_doors: list[list[tuple[int, int]]] = [[] for _ in rooms]
    graph: dict[int, set[int]] = defaultdict(set)

    for idx, room in enumerate(rooms):
        r0, c0, r1, c1 = room["bbox"]  # type: ignore[misc]
        side_cells = {
            "left": [(r, c0) for r in range(r0 + 1, r1)],
            "right": [(r, c1) for r in range(r0 + 1, r1)],
            "top": [(r0, c) for c in range(c0 + 1, c1)],
            "bottom": [(r1, c) for c in range(c0 + 1, c1)],
        }
        for side, cells in side_cells.items():
            dr, dc = outward[side]
            want = opposite[side]
            for r, c in cells:
                if int(grid[r, c]) != BG:
                    continue
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                other = lookup.get((nr, nc))
                if other is None or other[0] == idx or other[1] != want:
                    continue
                if int(grid[nr, nc]) != BG:
                    continue
                room_doors[idx].append((r, c))
                graph[idx].add(other[0])
                graph[other[0]].add(idx)

    return room_doors, graph


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill rectangular rooms by ranking their interior areas and reading colors from the corner legend.

    The solver:
    - finds the N×2 legend in a corner,
    - extracts monochrome rectangular room frames outside the legend,
    - pairs opposite-side background gaps to recover room adjacency through doors,
    - ranks rooms by interior area,
    - fills each room interior and its own door pixels with the rank color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    legend_colors, legend_mask = _find_legend(g, bg)
    rooms = _extract_rooms(g, bg, legend_mask, len(legend_colors))
    door_cells, graph = _room_doors(g, rooms)
    assert len(graph) == len(rooms), "all rooms must participate in the door topology"

    ranked = sorted(
        range(len(rooms)),
        key=lambda idx: (
            -int(rooms[idx]["area"]),
            int(rooms[idx]["bbox"][0]),  # type: ignore[index]
            int(rooms[idx]["bbox"][1]),  # type: ignore[index]
        ),
    )
    room_fill = {room_idx: legend_colors[rank] for rank, room_idx in enumerate(ranked)}

    out = g.copy()
    for idx, room in enumerate(rooms):
        fill_color = int(room_fill[idx])
        r0, c0, r1, c1 = room["bbox"]  # type: ignore[misc]
        out[r0 + 1 : r1, c0 + 1 : c1] = fill_color
        for r, c in door_cells[idx]:
            out[r, c] = fill_color
    return out


def _can_split(rect: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = rect
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    return h >= 2 * MIN_ROOM_SPAN or w >= 2 * MIN_ROOM_SPAN


def _split_options(rect: tuple[int, int, int, int]) -> list[tuple[str, int]]:
    r0, c0, r1, c1 = rect
    opts: list[tuple[str, int]] = []
    for split_r in range(r0 + MIN_ROOM_SPAN - 1, r1 - MIN_ROOM_SPAN + 1):
        opts.append(("h", split_r))
    for split_c in range(c0 + MIN_ROOM_SPAN - 1, c1 - MIN_ROOM_SPAN + 1):
        opts.append(("v", split_c))
    return opts


def _partition_rooms(rng: random.Random, count: int) -> list[tuple[int, int, int, int]] | None:
    root_h = rng.randint(9, 15)
    root_w = rng.randint(10, 18)
    rooms = [(0, 0, root_h - 1, root_w - 1)]
    while len(rooms) < count:
        candidates = [idx for idx, rect in enumerate(rooms) if _can_split(rect)]
        if not candidates:
            return None
        idx = rng.choice(candidates)
        rect = rooms.pop(idx)
        split_axis, split_at = rng.choice(_split_options(rect))
        r0, c0, r1, c1 = rect
        if split_axis == "h":
            rooms.append((r0, c0, split_at, c1))
            rooms.append((split_at + 1, c0, r1, c1))
        else:
            rooms.append((r0, c0, r1, split_at))
            rooms.append((r0, split_at + 1, r1, c1))
    return rooms


def _adjacencies(
    rooms: list[tuple[int, int, int, int]]
) -> list[tuple[str, int, int, int, int]]:
    edges: list[tuple[str, int, int, int, int]] = []
    for i, a in enumerate(rooms):
        ar0, ac0, ar1, ac1 = a
        for j in range(i + 1, len(rooms)):
            br0, bc0, br1, bc1 = rooms[j]
            if ac1 + 1 == bc0:
                lo, hi = max(ar0, br0) + 1, min(ar1, br1) - 1
                if lo <= hi:
                    edges.append(("h", i, j, lo, hi))
            elif bc1 + 1 == ac0:
                lo, hi = max(ar0, br0) + 1, min(ar1, br1) - 1
                if lo <= hi:
                    edges.append(("h", j, i, lo, hi))
            elif ar1 + 1 == br0:
                lo, hi = max(ac0, bc0) + 1, min(ac1, bc1) - 1
                if lo <= hi:
                    edges.append(("v", i, j, lo, hi))
            elif br1 + 1 == ar0:
                lo, hi = max(ac0, bc0) + 1, min(ac1, bc1) - 1
                if lo <= hi:
                    edges.append(("v", j, i, lo, hi))
    return edges


def _draw_rooms(
    rooms: list[tuple[int, int, int, int]],
    border_colors: list[int],
    fill_colors: list[int],
    legend_corner: str,
    rng: random.Random,
) -> np.ndarray | None:
    n = len(rooms)
    build_h = max(r1 for r0, c0, r1, c1 in rooms) + 1
    build_w = max(c1 for r0, c0, r1, c1 in rooms) + 1

    top = rng.randint(2, 5)
    bottom = rng.randint(2, 5)
    left = rng.randint(2, 6)
    right = rng.randint(2, 6)
    if legend_corner in ("tl", "tr"):
        top = rng.randint(n + 2, n + 5)
    else:
        bottom = rng.randint(n + 2, n + 5)
    if legend_corner in ("tl", "bl"):
        left = rng.randint(4, 7)
    else:
        right = rng.randint(4, 7)

    h = top + build_h + bottom
    w = left + build_w + right
    grid = np.full((h, w), BG, dtype=int)

    shifted: list[tuple[int, int, int, int]] = []
    for idx, (r0, c0, r1, c1) in enumerate(rooms):
        rr0, cc0, rr1, cc1 = r0 + top, c0 + left, r1 + top, c1 + left
        shifted.append((rr0, cc0, rr1, cc1))
        color = border_colors[idx]
        grid[rr0, cc0 : cc1 + 1] = color
        grid[rr1, cc0 : cc1 + 1] = color
        grid[rr0 : rr1 + 1, cc0] = color
        grid[rr0 : rr1 + 1, cc1] = color

    edges = _adjacencies(shifted)
    if not edges:
        return None

    degree = Counter()
    for kind, a, b, lo, hi in edges:
        degree[a] += 1
        degree[b] += 1
        pos = rng.randint(lo, hi)
        if kind == "h":
            left_room = shifted[a]
            right_room = shifted[b]
            grid[pos, left_room[3]] = BG
            grid[pos, right_room[1]] = BG
        else:
            top_room = shifted[a]
            bottom_room = shifted[b]
            grid[top_room[2], pos] = BG
            grid[bottom_room[0], pos] = BG

    if any(degree[idx] == 0 for idx in range(n)):
        return None

    if legend_corner == "tl":
        start_row, color_col, grey_col = 0, 0, 1
    elif legend_corner == "tr":
        start_row, color_col, grey_col = 0, w - 2, w - 1
    elif legend_corner == "bl":
        start_row, color_col, grey_col = h - n, 0, 1
    else:
        start_row, color_col, grey_col = h - n, w - 2, w - 1

    for idx, color in enumerate(fill_colors):
        grid[start_row + idx, color_col] = color
        grid[start_row + idx, grey_col] = GREY

    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n = rng.choice([2, 3, 4])
    rooms = _partition_rooms(rng, n)
    if rooms is None:
        return None

    areas = [(r1 - r0 - 1) * (c1 - c0 - 1) for r0, c0, r1, c1 in rooms]
    if any(area <= 0 for area in areas) or len(set(areas)) != n:
        return None

    colors = rng.sample(COLOR_POOL, 2 * n)
    border_colors = colors[:n]
    fill_colors = colors[n:]
    legend_corner = rng.choice(["tl", "tr", "bl", "br"])
    grid = _draw_rooms(rooms, border_colors, fill_colors, legend_corner, rng)
    if grid is None:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    if set(fill_colors) - set(map(int, np.unique(out))):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
