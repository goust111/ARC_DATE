import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
MAROON = 9
PIECE_COLORS = [1, 2, 3, 4, 6, 7, 8]
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _normalize(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _translate(shape: tuple[tuple[int, int], ...], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in shape}


def _mark_boundaries(crop: np.ndarray, template_mask: np.ndarray, piece_colors: set[int]) -> np.ndarray:
    out = crop.copy()
    h, w = out.shape
    for r in range(h):
        for c in range(w):
            if not template_mask[r, c] or int(crop[r, c]) != GREY:
                continue
            neigh_colors = set()
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w:
                    v = int(crop[nr, nc])
                    if v in piece_colors:
                        neigh_colors.add(v)
            if len(neigh_colors) >= 2:
                out[r, c] = MAROON
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fit scattered colored fragments into the matching holes of a grey template, then mark grey separators between different colors in maroon.

    Input structure:
    - The background is black.
    - One large 4-connected grey region is the template; its interior contains one or more black connected holes.
    - Outside the template are several monochrome colored 4-connected fragments, each matching one hole by translation only.

    Rule:
    - Find the largest grey connected component and crop to its bounding box.
    - Detect each enclosed black hole inside that crop and match it to the unique outside fragment with the same normalized shape.
    - Fill every hole with its fragment color, then recolor any remaining grey pixel to maroon if its 4-neighbors touch at least two different filled fragment colors.

    Key insight:
    The task is a two-stage inference: first solve a translation-only shape-matching puzzle between outside pieces and interior holes, then inspect the completed template to identify grey separator pixels that sit directly between different colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "main expects a 2D grid"

    grey_comps = _components(g == GREY)
    assert grey_comps, "expected at least one grey template component"
    template_cells = max(grey_comps, key=len)
    r0, c0, r1, c1 = _bbox(template_cells)

    crop = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    template_mask = np.zeros_like(crop, dtype=bool)
    for r, c in template_cells:
        template_mask[r - r0, c - c0] = True

    hole_map: dict[tuple[tuple[int, int], ...], list[list[tuple[int, int]]]] = {}
    for comp in _components(crop == BG):
        if any(r == 0 or c == 0 or r == crop.shape[0] - 1 or c == crop.shape[1] - 1 for r, c in comp):
            continue
        sig = _normalize(comp)
        hole_map.setdefault(sig, []).append(comp)
    assert hole_map, "expected at least one enclosed hole in the template"

    piece_map: dict[tuple[tuple[int, int], ...], list[int]] = {}
    for comp in _components((g != BG) & (g != GREY)):
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1, "each fragment must be monochrome"
        color = next(iter(colors))
        sig = _normalize(comp)
        piece_map.setdefault(sig, []).append(color)

    assert set(hole_map) == set(piece_map), "hole shapes and fragment shapes must match exactly"
    for sig in hole_map:
        assert len(hole_map[sig]) == len(piece_map[sig]) == 1, "shape matching must be unambiguous"

    filled = crop.copy()
    piece_colors = set()
    for sig, comps in hole_map.items():
        color = int(piece_map[sig][0])
        piece_colors.add(color)
        for r, c in comps[0]:
            filled[r, c] = color

    return _mark_boundaries(filled, template_mask, piece_colors)


def _rotate90(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return _normalize({(c, -r) for r, c in shape})


def _reflect(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return _normalize({(r, -c) for r, c in shape})


def _build_shape_catalog() -> list[tuple[tuple[int, int], ...]]:
    bases = [
        ((0, 0), (0, 1)),
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 1)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 1), (0, 2), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (1, 1), (1, 2), (2, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1)),
    ]
    seen = set()
    catalog = []
    for base in bases:
        shape = _normalize(base)
        variants = [shape]
        cur = shape
        for _ in range(3):
            cur = _rotate90(cur)
            variants.append(cur)
        variants.extend(_reflect(v) for v in list(variants))
        for variant in variants:
            if variant not in seen:
                seen.add(variant)
                catalog.append(variant)
    catalog.sort(key=lambda s: (len(s), _shape_dims(s), s))
    return catalog


SHAPE_CATALOG = _build_shape_catalog()


def _can_place_hole(cells: set[tuple[int, int]], inner_h: int, inner_w: int, occupied: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        if not (0 <= r < inner_h and 0 <= c < inner_w):
            return False
        if (r, c) in occupied:
            return False
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in occupied:
                return False
    return True


def _place_bridged_pair(
    rng: random.Random,
    inner_h: int,
    inner_w: int,
) -> tuple[str, tuple[int, int], tuple[tuple[int, int], ...], tuple[tuple[int, int], ...], set[tuple[int, int]], set[tuple[int, int]]] | None:
    orientations = ["horizontal", "vertical"]
    for _ in range(300):
        orientation = rng.choice(orientations)
        shape_a, shape_b = rng.sample(SHAPE_CATALOG, 2)
        if orientation == "horizontal":
            if inner_w < 3:
                continue
            br = rng.randrange(inner_h)
            bc = rng.randrange(1, inner_w - 1)
            contacts_a = [cell for cell in shape_a if cell[1] == max(c for _, c in shape_a)]
            contacts_b = [cell for cell in shape_b if cell[1] == min(c for _, c in shape_b)]
            rng.shuffle(contacts_a)
            rng.shuffle(contacts_b)
            for ca in contacts_a:
                cells_a = _translate(shape_a, br - ca[0], bc - 1 - ca[1])
                if not _can_place_hole(cells_a, inner_h, inner_w, set()):
                    continue
                for cb in contacts_b:
                    cells_b = _translate(shape_b, br - cb[0], bc + 1 - cb[1])
                    if _can_place_hole(cells_b, inner_h, inner_w, cells_a):
                        return orientation, (br, bc), shape_a, shape_b, cells_a, cells_b
        else:
            if inner_h < 3:
                continue
            br = rng.randrange(1, inner_h - 1)
            bc = rng.randrange(inner_w)
            contacts_a = [cell for cell in shape_a if cell[0] == max(r for r, _ in shape_a)]
            contacts_b = [cell for cell in shape_b if cell[0] == min(r for r, _ in shape_b)]
            rng.shuffle(contacts_a)
            rng.shuffle(contacts_b)
            for ca in contacts_a:
                cells_a = _translate(shape_a, br - 1 - ca[0], bc - ca[1])
                if not _can_place_hole(cells_a, inner_h, inner_w, set()):
                    continue
                for cb in contacts_b:
                    cells_b = _translate(shape_b, br + 1 - cb[0], bc - cb[1])
                    if _can_place_hole(cells_b, inner_h, inner_w, cells_a):
                        return orientation, (br, bc), shape_a, shape_b, cells_a, cells_b
    return None


def _try_place_extra_hole(
    rng: random.Random,
    shape: tuple[tuple[int, int], ...],
    inner_h: int,
    inner_w: int,
    occupied: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    sh, sw = _shape_dims(shape)
    for _ in range(200):
        dr = rng.randrange(inner_h - sh + 1)
        dc = rng.randrange(inner_w - sw + 1)
        cells = _translate(shape, dr, dc)
        if _can_place_hole(cells, inner_h, inner_w, occupied):
            return cells
    return None


def _piece_and_template_outputs(
    inner_h: int,
    inner_w: int,
    hole_cells: list[set[tuple[int, int]]],
    colors: list[int],
) -> np.ndarray:
    out = np.full((inner_h + 2, inner_w + 2), GREY, dtype=int)
    for cells, color in zip(hole_cells, colors):
        for r, c in cells:
            out[r + 1, c + 1] = color
    template_mask = np.ones_like(out, dtype=bool)
    piece_colors = set(colors)
    return _mark_boundaries(out, template_mask, piece_colors)


def _expand_blocked(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    blocked = set()
    for r, c in cells:
        if 0 <= r < h and 0 <= c < w:
            blocked.add((r, c))
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                blocked.add((nr, nc))
    return blocked


def _render_input(
    rng: random.Random,
    inner_h: int,
    inner_w: int,
    hole_cells: list[set[tuple[int, int]]],
    shapes: list[tuple[tuple[int, int], ...]],
    colors: list[int],
) -> np.ndarray | None:
    template_h = inner_h + 2
    template_w = inner_w + 2

    total_piece_area = sum(len(shape) for shape in shapes)
    min_h = max(template_h + 4, 14)
    min_w = max(template_w + 4, 14)
    max_h = min(30, max(min_h, template_h + 10))
    max_w = min(30, max(min_w, template_w + 10))
    if max_h < min_h or max_w < min_w:
        return None

    for _ in range(150):
        grid_h = rng.randint(min_h, max_h)
        grid_w = rng.randint(min_w, max_w)
        if grid_h * grid_w - template_h * template_w < total_piece_area + 20:
            continue

        top = rng.randint(1, grid_h - template_h - 1)
        left = rng.randint(1, grid_w - template_w - 1)
        grid = np.full((grid_h, grid_w), BG, dtype=int)
        grid[top : top + template_h, left : left + template_w] = GREY
        for cells in hole_cells:
            for r, c in cells:
                grid[top + 1 + r, left + 1 + c] = BG

        template_cells_abs = {
            (r, c)
            for r in range(top, top + template_h)
            for c in range(left, left + template_w)
            if grid[r, c] == GREY
        }
        blocked = _expand_blocked(template_cells_abs, grid_h, grid_w)

        order = list(range(len(shapes)))
        rng.shuffle(order)
        ok = True
        for idx in order:
            shape = shapes[idx]
            color = colors[idx]
            sh, sw = _shape_dims(shape)
            placed = None
            for _ in range(300):
                dr = rng.randrange(grid_h - sh + 1)
                dc = rng.randrange(grid_w - sw + 1)
                cells = _translate(shape, dr, dc)
                if any((r, c) in blocked for r, c in cells):
                    continue
                placed = cells
                break
            if placed is None:
                ok = False
                break
            for r, c in placed:
                grid[r, c] = color
            blocked |= _expand_blocked(placed, grid_h, grid_w)

        if ok:
            return grid
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    inner_h = rng.randint(6, 12)
    inner_w = rng.randint(6, 12)
    hole_count = rng.randint(2, 5)

    bridged = _place_bridged_pair(rng, inner_h, inner_w)
    if bridged is None:
        return None
    _orientation, _bridge_cell, shape_a, shape_b, cells_a, cells_b = bridged

    used_shapes = {shape_a, shape_b}
    hole_cells = [cells_a, cells_b]
    hole_shapes = [shape_a, shape_b]
    occupied = set(cells_a) | set(cells_b)

    candidates = [shape for shape in SHAPE_CATALOG if shape not in used_shapes]
    rng.shuffle(candidates)
    for shape in candidates:
        if len(hole_cells) >= hole_count:
            break
        placed = _try_place_extra_hole(rng, shape, inner_h, inner_w, occupied)
        if placed is None:
            continue
        hole_cells.append(placed)
        hole_shapes.append(shape)
        occupied |= placed

    if len(hole_cells) != hole_count:
        return None

    colors = rng.sample(PIECE_COLORS, hole_count)
    expected = _piece_and_template_outputs(inner_h, inner_w, hole_cells, colors)
    if MAROON not in expected:
        return None

    grid = _render_input(rng, inner_h, inner_w, hole_cells, hole_shapes, colors)
    if grid is None:
        return None

    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid puzzle instance")
