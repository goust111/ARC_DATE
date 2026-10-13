from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]


def _mode(values) -> int:
    return int(Counter(map(int, values)).most_common(1)[0][0])


def _bbox(coords: list[Coord]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _components(mask: np.ndarray) -> list[list[Coord]]:
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(height):
        for c in range(width):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < height and 0 <= nc < width and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _normalize(coords: list[Coord]) -> tuple[Coord, ...]:
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _translated(shape: tuple[Coord, ...], r0: int, c0: int) -> set[Coord]:
    return {(int(r0 + r), int(c0 + c)) for r, c in shape}


def _extract_scene(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = g.shape
    bg = int(g[0, 0])

    bottom_non_bg = [int(v) for v in g[-1, :] if int(v) != bg]
    assert bottom_non_bg, "expected a base terrain on the bottom row"
    base = _mode(bottom_non_bg)

    base_rows = np.argwhere(g == base)
    assert base_rows.size, "expected terrain base pixels"
    surface_r = int(base_rows[:, 0].min())

    below = np.zeros_like(g, dtype=bool)
    below[surface_r:, :] = g[surface_r:, :] == bg
    cavity_comps = [
        comp for comp in _components(below)
        if any(r == surface_r for r, _ in comp)
    ]
    assert cavity_comps, "expected upward-open cavities in the terrain"

    cavities = []
    for comp in cavity_comps:
        b = _bbox(comp)
        cavities.append(
            {
                "coords": tuple(sorted(comp)),
                "bbox": b,
                "shape": _normalize(comp),
                "area": len(comp),
            }
        )
    cavities.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))

    upper = g[:surface_r, :]
    colors = sorted(
        int(v)
        for v in np.unique(upper)
        if int(v) not in (bg, base)
    )
    assert len(colors) == len(cavities), "expected one visible color per cavity"
    cluster_mask = np.zeros_like(g, dtype=bool)
    cluster_mask[:surface_r, :] = (g[:surface_r, :] != bg) & (g[:surface_r, :] != base)

    return g, bg, base, surface_r, cavities, colors, cluster_mask


def _candidate_offsets(
    g: np.ndarray,
    shape: tuple[Coord, ...],
    color: int,
    surface_r: int,
    cluster_mask: np.ndarray,
) -> list[Coord]:
    coords_arr = np.argwhere(g == int(color))
    if coords_arr.size == 0 or coords_arr.shape[0] >= len(shape):
        return []

    height, width = g.shape
    visible = [(int(r), int(c)) for r, c in coords_arr]
    visible_set = set(visible)
    offsets: set[Coord] = set()

    first_r, first_c = visible[0]
    for sr, sc in shape:
        offsets.add((int(first_r - sr), int(first_c - sc)))

    valid: list[Coord] = []
    for r0, c0 in offsets:
        cells = _translated(shape, r0, c0)
        if not visible_set.issubset(cells):
            continue
        if any(r < 0 or r >= surface_r or c < 0 or c >= width for r, c in cells):
            continue
        if not all(bool(cluster_mask[r, c]) for r, c in cells):
            continue
        if all(int(g[r, c]) == int(color) for r, c in cells):
            continue
        valid.append((int(r0), int(c0)))
    return sorted(valid)


def _candidate_matrix(g, cavities, colors, surface_r, cluster_mask) -> list[list[int]]:
    rows: list[list[int]] = []
    for cavity in cavities:
        row = []
        for i, color in enumerate(colors):
            if _candidate_offsets(g, cavity["shape"], int(color), surface_r, cluster_mask):
                row.append(i)
        rows.append(row)
    return rows


def _unique_assignment(candidates: list[list[int]]) -> list[int]:
    n = len(candidates)
    assert n > 0
    assert all(candidates), "every cavity needs at least one candidate color"

    row_order = sorted(range(n), key=lambda r: (len(candidates[r]), r))
    used: set[int] = set()
    current = [-1] * n
    solutions: list[list[int]] = []

    def dfs(pos: int) -> None:
        if len(solutions) > 1:
            return
        if pos == n:
            solutions.append(current[:])
            return
        row = row_order[pos]
        for color_i in candidates[row]:
            if color_i in used:
                continue
            used.add(color_i)
            current[row] = int(color_i)
            dfs(pos + 1)
            current[row] = -1
            used.remove(color_i)

    dfs(0)
    assert len(solutions) == 1, "expected a unique global cavity-color assignment"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each upward-open terrain cavity with the color of the overprinted sprite fragment that claims its mold.

    Input structure:
    - The top-left color is the empty background; the bottom row belongs to a large base-colored terrain.
    - The terrain starts at a flat upper surface and contains several background-colored, upward-open cavities.
    - Above the surface is one connected multicolor overprint, made from same-orientation monochrome sprites whose full shapes match the cavities.

    Rule:
    - Extract every cavity shape in terrain coordinates and every visible overprint color above the terrain.
    - A color can claim a cavity if all visible pixels of that color fit inside a translated copy of the cavity shape, while the whole translated shape is covered by the multicolor overprint and the color is not fully visible.
    - Choose the single perfect global assignment from cavities to colors, clear the overprint, restore the base terrain, and fill each cavity with its assigned color.

    Key insight:
    - The cavities are the mold catalog: hidden parts of each sprite are recovered from the only globally consistent shape assignment, not from the visible fragment size or stacking order.
    """
    g, bg, base, surface_r, cavities, colors, cluster_mask = _extract_scene(grid)
    candidates = _candidate_matrix(g, cavities, colors, surface_r, cluster_mask)
    assignment = _unique_assignment(candidates)

    out = np.full(g.shape, int(bg), dtype=int)
    out[surface_r:, :] = int(base)
    for cavity_i, color_i in enumerate(assignment):
        color = int(colors[color_i])
        for r, c in cavities[cavity_i]["coords"]:
            out[int(r), int(c)] = color
    return out


# ----------------------------- generator -----------------------------


_BG = 0
_ALL_COLORS = list(range(1, 10))
_MAX_ATTEMPTS = 500


def _is_connected(cells: set[Coord]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if nb in cells and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return seen == cells


def _random_shape() -> tuple[Coord, ...]:
    for _ in range(100):
        h = random.randint(3, 6)
        w = random.randint(3, 5)
        target = random.randint(6, min(h * w - 1, 13))
        cells: set[Coord] = {(0, random.randrange(w))}

        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        weight = 3 if nr > r else 1
                        frontier.extend([(nr, nc)] * weight)
            if not frontier:
                break
            cells.add(random.choice(frontier))

        coords = list(cells)
        norm = _normalize(coords)
        nr1 = max(r for r, _ in norm)
        nc1 = max(c for _, c in norm)
        if nr1 < 2 or nc1 < 1:
            continue
        if not any(r == 0 for r, _ in norm):
            continue
        if len(norm) >= (nr1 + 1) * (nc1 + 1):
            continue
        if _is_connected(set(norm)):
            return norm
    raise RuntimeError("failed to make a cavity shape")


def _choose_shapes(count: int) -> list[tuple[Coord, ...]]:
    shapes: list[tuple[Coord, ...]] = []
    seen = set()
    for _ in range(200):
        shape = _random_shape()
        if shape in seen:
            continue
        shapes.append(shape)
        seen.add(shape)
        if len(shapes) == count:
            return shapes
    raise RuntimeError("failed to make distinct cavity shapes")


def _shape_size(shape: tuple[Coord, ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _spread_gaps(total_extra: int, slots: int) -> list[int]:
    gaps = [1] * slots
    for _ in range(total_extra - slots):
        gaps[random.randrange(slots)] += 1
    return gaps


def _place_cavities(
    grid: np.ndarray,
    shapes: list[tuple[Coord, ...]],
    surface_r: int,
) -> list[tuple[Coord, ...]]:
    width = grid.shape[1]
    widths = [_shape_size(shape)[1] for shape in shapes]
    spare = width - sum(widths)
    if spare < len(shapes) + 1:
        raise ValueError("cavities do not fit")
    gaps = _spread_gaps(spare, len(shapes) + 1)
    c0 = gaps[0]
    cavity_coords: list[tuple[Coord, ...]] = []
    for shape, gap_after in zip(shapes, gaps[1:]):
        coords = tuple(sorted((surface_r + r, c0 + c) for r, c in shape))
        for r, c in coords:
            grid[r, c] = _BG
        cavity_coords.append(coords)
        c0 += _shape_size(shape)[1] + gap_after
    return cavity_coords


def _footprint(shape: tuple[Coord, ...], pos: Coord) -> set[Coord]:
    return _translated(shape, int(pos[0]), int(pos[1]))


def _connected_union(sets: list[set[Coord]]) -> bool:
    union = set().union(*sets)
    return _is_connected(union)


def _choose_cluster_positions(
    shapes: list[tuple[Coord, ...]],
    surface_r: int,
    width: int,
) -> list[Coord] | None:
    max_h = max(_shape_size(shape)[0] for shape in shapes)
    max_w = max(_shape_size(shape)[1] for shape in shapes)
    if surface_r - max_h < 3 or width - max_w < 4:
        return None

    for _ in range(200):
        anchor_r = random.randint(1, surface_r - max_h - 1)
        anchor_c = random.randint(2, width - max_w - 2)
        positions: list[Coord] = []
        ok = True
        for shape in shapes:
            h, w = _shape_size(shape)
            r = anchor_r + random.randint(-2, 2)
            c = anchor_c + random.randint(-3, 3)
            if r < 0 or c < 0 or r + h >= surface_r or c + w > width:
                ok = False
                break
            positions.append((int(r), int(c)))
        if not ok:
            continue

        fps = [_footprint(shape, pos) for shape, pos in zip(shapes, positions)]
        if not _connected_union(fps):
            continue
        if all(fp & set().union(*(fps[:i] + fps[i + 1:])) for i, fp in enumerate(fps)):
            return positions
    return None


def _assign_visible_cells(
    shapes: list[tuple[Coord, ...]],
    positions: list[Coord],
) -> dict[Coord, int] | None:
    fps = [_footprint(shape, pos) for shape, pos in zip(shapes, positions)]
    union = set().union(*fps)
    cover = {cell: [i for i, fp in enumerate(fps) if cell in fp] for cell in union}

    for _ in range(100):
        owner: dict[Coord, int] = {}
        for cell, who in cover.items():
            if len(who) == 1:
                owner[cell] = who[0]
            else:
                owner[cell] = random.choice(who)

        for i, fp in enumerate(fps):
            if sum(1 for cell in fp if owner.get(cell) == i) == 0:
                choices = [cell for cell in fp if i in cover[cell]]
                for cell in random.sample(choices, min(2, len(choices))):
                    owner[cell] = i

        for i, fp in enumerate(fps):
            visible = [cell for cell in fp if owner.get(cell) == i]
            if len(visible) >= len(fp):
                choices = [cell for cell in fp if len(cover[cell]) > 1]
                if choices:
                    cell = random.choice(choices)
                    owner[cell] = random.choice([j for j in cover[cell] if j != i])

        counts = [sum(1 for cell in fp if owner.get(cell) == i) for i, fp in enumerate(fps)]
        if all(2 <= counts[i] < len(fps[i]) for i in range(len(fps))):
            return owner
    return None


def _expected_output(
    shape: tuple[int, int],
    bg: int,
    base: int,
    surface_r: int,
    cavity_coords: list[tuple[Coord, ...]],
    colors: list[int],
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    out[surface_r:, :] = int(base)
    for coords, color in zip(cavity_coords, colors):
        for r, c in coords:
            out[r, c] = int(color)
    return out


def _semantic_ok(inp: np.ndarray, expected: np.ndarray, colors_by_cavity: list[int]) -> bool:
    try:
        g, _bg, _base, surface_r, cavities, colors, cluster_mask = _extract_scene(inp)
        candidates = _candidate_matrix(g, cavities, colors, surface_r, cluster_mask)
        assignment = _unique_assignment(candidates)
        solved = main(inp)
    except Exception:
        return False
    if not np.array_equal(solved, expected):
        return False

    solved_colors = [colors[i] for i in assignment]
    if solved_colors != colors_by_cavity:
        return False

    for cavity, color in zip(cavities, solved_colors):
        if int(np.count_nonzero(inp == int(color))) >= int(cavity["area"]):
            return False

    # Avoid examples that can be solved by sorting colors, visible sizes, or cavities alone.
    visible_counts = [int(np.count_nonzero(inp == int(color))) for color in colors_by_cavity]
    cavity_areas = [int(cavity["area"]) for cavity in cavities]
    if visible_counts == sorted(visible_counts) and colors_by_cavity == sorted(colors_by_cavity):
        return False
    if visible_counts == cavity_areas:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    count = random.choice([3, 3, 4, 4, 5])
    shapes = _choose_shapes(count)

    max_shape_h = max(_shape_size(shape)[0] for shape in shapes)
    terrain_h = random.randint(max_shape_h + 2, max_shape_h + 6)
    surface_r = random.randint(max(8, max_shape_h + 4), 15)
    height = surface_r + terrain_h
    if height > 30:
        return None

    min_w = sum(_shape_size(shape)[1] for shape in shapes) + count + 1
    width = random.randint(max(16, min_w), 30)

    bg = _BG
    base = random.choice(_ALL_COLORS)
    colors = random.sample([c for c in _ALL_COLORS if c != base], count)
    if random.random() < 0.5:
        random.shuffle(colors)

    grid = np.full((height, width), int(bg), dtype=int)
    grid[surface_r:, :] = int(base)
    try:
        cavity_coords = _place_cavities(grid, shapes, surface_r)
    except ValueError:
        return None

    positions = _choose_cluster_positions(shapes, surface_r, width)
    if positions is None:
        return None
    owner = _assign_visible_cells(shapes, positions)
    if owner is None:
        return None

    for (r, c), owner_i in owner.items():
        if r >= surface_r:
            return None
        grid[r, c] = int(colors[owner_i])

    expected = _expected_output(grid.shape, bg, base, surface_r, cavity_coords, colors)
    if not _semantic_ok(grid, expected, colors):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid inlay-fill overprint input")

