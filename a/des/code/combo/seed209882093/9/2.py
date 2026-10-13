from __future__ import annotations

import random

import numpy as np


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
PALETTE = tuple(range(10))

GRID_MIN = 14
GRID_MAX = 22
MAX_SCENE_ATTEMPTS = 400
MAX_OBJECT_ATTEMPTS = 240
MAX_BASE_ATTEMPTS = 60


def _most_common_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _monochromatic_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            comps.append({"color": color, "coords": tuple(sorted(coords))})

    return comps


def _analyze_scene(
    grid: np.ndarray,
) -> tuple[int, list[dict[str, object]], dict[int, tuple[tuple[int, int], tuple[int, int]]], dict[tuple[int, int], int]]:
    bg = _most_common_color(grid)
    comps = _monochromatic_components(grid, bg)

    by_color: dict[int, list[dict[str, object]]] = {}
    for comp in comps:
        by_color.setdefault(int(comp["color"]), []).append(comp)

    portal_pairs: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
    portal_at: dict[tuple[int, int], int] = {}
    for color, group in by_color.items():
        total = sum(len(comp["coords"]) for comp in group)
        if total != 2 or len(group) != 2:
            continue
        if not all(len(comp["coords"]) == 1 for comp in group):
            continue
        cells = [tuple(comp["coords"][0]) for comp in group]
        if cells[0][0] != cells[1][0] and cells[0][1] != cells[1][1]:
            continue
        portal_pairs[int(color)] = (cells[0], cells[1])
        portal_at[cells[0]] = int(color)
        portal_at[cells[1]] = int(color)

    objects: list[dict[str, object]] = []
    for comp in comps:
        if len(comp["coords"]) > 1 and int(comp["color"]) not in portal_pairs:
            objects.append(comp)

    objects.sort(key=lambda comp: tuple(comp["coords"]))
    return bg, objects, portal_pairs, portal_at


def main(grid: np.ndarray) -> np.ndarray:
    """
    Treat aligned singleton same-color pairs as portals, move the touching object by
    the portal-to-portal offset, and recolor the moved object to the portal color.
    """
    g = np.asarray(grid, dtype=int)
    bg, objects, portal_pairs, portal_at = _analyze_scene(g)
    out = np.full_like(g, bg)

    for obj in objects:
        touching: set[tuple[int, int]] = set()
        for r, c in obj["coords"]:
            for dr, dc in DIRS4:
                nxt = (int(r) + dr, int(c) + dc)
                if nxt in portal_at:
                    touching.add(nxt)

        assert len(touching) == 1, "each object must touch exactly one portal pixel"
        a = next(iter(touching))
        portal_color = portal_at[a]
        p0, p1 = portal_pairs[portal_color]
        b = p1 if a == p0 else p0
        shift = (b[0] - a[0], b[1] - a[1])

        for r, c in obj["coords"]:
            nr = int(r) + shift[0]
            nc = int(c) + shift[1]
            assert 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]
            out[nr, nc] = portal_color

    return out


def _normalize_shape(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _transform_shape(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    pts = set(cells)
    for _ in range(random.randint(0, 3)):
        pts = {(-c, r) for r, c in pts}
    if random.random() < 0.5:
        pts = {(r, -c) for r, c in pts}
    return _normalize_shape(pts)


def _random_shape(max_cells: int) -> tuple[tuple[int, int], ...]:
    target = random.randint(2, max_cells)
    cells = {(0, 0)}
    while len(cells) < target:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in DIRS4:
                frontier.add((r + dr, c + dc))
        frontier.difference_update(cells)
        cells.add(random.choice(tuple(frontier)))
    return _transform_shape(cells)


def _exposed_sides(shape: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    cells = set(shape)
    exposed: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for cell in shape:
        r, c = cell
        for dr, dc in DIRS4:
            if (r + dr, c + dc) not in cells:
                exposed.append((cell, (dr, dc)))
    return exposed


def _shift_cells(cells: set[tuple[int, int]], delta: tuple[int, int]) -> set[tuple[int, int]]:
    dr, dc = delta
    return {(r + dr, c + dc) for r, c in cells}


def _touches(cells: set[tuple[int, int]], other: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        for dr, dc in DIRS4:
            if (r + dr, c + dc) in other:
                return True
    return False


def _adjacent_portals(source_cells: set[tuple[int, int]], portals: set[tuple[int, int]]) -> set[tuple[int, int]]:
    touching: set[tuple[int, int]] = set()
    for r, c in source_cells:
        for dr, dc in DIRS4:
            nxt = (r + dr, c + dc)
            if nxt in portals:
                touching.add(nxt)
    return touching


def _random_shift(h: int, w: int) -> tuple[int, int]:
    if random.random() < 0.5:
        max_len = max(3, min(8, h - 5))
        length = random.randint(3, max_len)
        return (length if random.random() < 0.5 else -length, 0)
    max_len = max(3, min(8, w - 5))
    length = random.randint(3, max_len)
    return (0, length if random.random() < 0.5 else -length)


def _try_place_structure(
    h: int,
    w: int,
    shape: tuple[tuple[int, int], ...],
    shift: tuple[int, int],
    existing_input: set[tuple[int, int]],
    existing_output: set[tuple[int, int]],
) -> dict[str, object] | None:
    source_rel = set(shape)
    output_rel = _shift_cells(source_rel, shift)
    if source_rel & output_rel:
        return None

    exposed = _exposed_sides(shape)
    random.shuffle(exposed)
    reserved = existing_input | existing_output

    for port_cell, port_dir in exposed:
        a_rel = (port_cell[0] + port_dir[0], port_cell[1] + port_dir[1])
        b_rel = (a_rel[0] + shift[0], a_rel[1] + shift[1])
        input_rel = set(source_rel)
        input_rel.add(a_rel)
        input_rel.add(b_rel)
        if input_rel & output_rel:
            continue

        all_rel = input_rel | output_rel
        min_r = min(r for r, _ in all_rel)
        max_r = max(r for r, _ in all_rel)
        min_c = min(c for _, c in all_rel)
        max_c = max(c for _, c in all_rel)

        row_lo = -min_r
        row_hi = h - 1 - max_r
        col_lo = -min_c
        col_hi = w - 1 - max_c
        if row_lo > row_hi or col_lo > col_hi:
            continue

        for _ in range(MAX_BASE_ATTEMPTS):
            base_r = random.randint(row_lo, row_hi)
            base_c = random.randint(col_lo, col_hi)

            source_abs = {(base_r + r, base_c + c) for r, c in source_rel}
            a_abs = (base_r + a_rel[0], base_c + a_rel[1])
            b_abs = (base_r + b_rel[0], base_c + b_rel[1])
            portals_abs = {a_abs, b_abs}
            input_abs = set(source_abs) | portals_abs
            output_abs = {(base_r + r, base_c + c) for r, c in output_rel}

            if input_abs & reserved or output_abs & reserved:
                continue
            if _touches(input_abs, existing_input):
                continue
            if _touches(output_abs, existing_output):
                continue
            if _adjacent_portals(source_abs, portals_abs) != {a_abs}:
                continue

            return {
                "source": source_abs,
                "output": output_abs,
                "portal_a": a_abs,
                "portal_b": b_abs,
            }

    return None


def _build_scene() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    bg = random.choice(PALETTE)

    max_objects = 4 if min(h, w) >= 18 else 3
    n_objects = random.randint(2, max_objects)

    colors = [color for color in PALETTE if color != bg]
    random.shuffle(colors)
    object_colors = colors[:n_objects]
    portal_colors = colors[n_objects : 2 * n_objects]

    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)

    existing_input: set[tuple[int, int]] = set()
    existing_output: set[tuple[int, int]] = set()

    for index in range(n_objects):
        placed = False
        max_cells = 7 if min(h, w) >= 18 else 6
        for _ in range(MAX_OBJECT_ATTEMPTS):
            shape = _random_shape(max_cells)
            shift = _random_shift(h, w)
            placement = _try_place_structure(h, w, shape, shift, existing_input, existing_output)
            if placement is None:
                continue

            for r, c in placement["source"]:
                inp[r, c] = object_colors[index]
            ar, ac = placement["portal_a"]
            br, bc = placement["portal_b"]
            inp[ar, ac] = portal_colors[index]
            inp[br, bc] = portal_colors[index]

            for r, c in placement["output"]:
                out[r, c] = portal_colors[index]

            existing_input.update(placement["source"])
            existing_input.add(placement["portal_a"])
            existing_input.add(placement["portal_b"])
            existing_output.update(placement["output"])
            placed = True
            break

        if not placed:
            return None

    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_SCENE_ATTEMPTS):
        scene = _build_scene()
        if scene is not None:
            return scene[0]
    raise RuntimeError("failed to generate a valid portal-transport scene")
