from __future__ import annotations

import itertools
import random
from collections import Counter, deque

import numpy as np


BLACK = 0
RED = 2
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbors(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in DIR4]


def _mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _set_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors(r, c):
            nxt = (nr, nc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return len(seen) == len(cells)


def _dominant_color(values: list[int]) -> int:
    counts = Counter(int(v) for v in values)
    color, amount = counts.most_common(1)[0]
    assert sum(1 for v in counts.values() if v == amount) == 1
    return int(color)


def _has_unique_dominant(values: list[int], color: int) -> bool:
    counts = Counter(int(v) for v in values)
    amount = counts.get(int(color), 0)
    return amount > 0 and sum(1 for v in counts.values() if v == amount) == 1 and amount == max(counts.values())


def _component_info(grid: np.ndarray, cells: list[tuple[int, int]]) -> dict:
    colors = [int(grid[r, c]) for r, c in cells]
    main_color = _dominant_color(colors)
    red_cells = [(r, c) for r, c in cells if int(grid[r, c]) == RED]
    assert len(red_cells) == 1
    return {
        "cells": [(int(r), int(c)) for r, c in cells],
        "cell_set": {(int(r), int(c)) for r, c in cells},
        "red": red_cells[0],
        "main": main_color,
    }


def _move_component(comp: dict, dr: int, dc: int, grid: np.ndarray) -> dict[tuple[int, int], int]:
    moved: dict[tuple[int, int], int] = {}
    for r, c in comp["cells"]:
        color = int(grid[r, c])
        if color == RED:
            color = int(comp["main"])
        pos = (int(r + dr), int(c + dc))
        if pos in moved:
            assert moved[pos] == color
        moved[pos] = color
    return moved


def _repair_symmetry(
    cells: dict[tuple[int, int], int],
    orientation: str,
    axis_sum: int,
    shape: tuple[int, int],
) -> dict[tuple[int, int], int]:
    h, w = shape
    out = dict(cells)
    original = list(out.items())
    for (r, c), color in original:
        if orientation == "H":
            rr, cc = int(r), int(axis_sum - c)
        else:
            rr, cc = int(axis_sum - r), int(c)
        assert 0 <= rr < h and 0 <= cc < w
        if (rr, cc) not in out:
            out[(rr, cc)] = int(color)
        else:
            assert out[(rr, cc)] == int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-dominant-color fragments, slide each pair together until the red connectors touch,
    then mirror-complete the merged body across the connection seam.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    comps = _mask_components(g != BLACK)
    assert comps and len(comps) % 2 == 0

    infos = [_component_info(g, comp) for comp in comps]
    for info in infos:
        assert 3 <= len(info["cells"]) <= 6
        assert info["main"] != BLACK
        assert info["main"] != RED

    by_main: dict[int, list[dict]] = {}
    for info in infos:
        by_main.setdefault(int(info["main"]), []).append(info)
    assert all(len(group) == 2 for group in by_main.values())

    h, w = g.shape
    out = np.full((h, w), BLACK, dtype=int)

    for main_color in sorted(by_main):
        first, second = by_main[main_color]
        ar, ac = first["red"]
        br, bc = second["red"]

        if ar == br:
            left, right = sorted((first, second), key=lambda item: item["red"][1])
            gap = int(right["red"][1] - left["red"][1] - 1)
            assert gap >= 1
            left_shift = gap // 2
            right_shift = gap - left_shift
            moved = _move_component(left, 0, left_shift, g)
            moved_right = _move_component(right, 0, -right_shift, g)
            axis_sum = int(left["red"][1] + left_shift + right["red"][1] - right_shift)
            assert axis_sum % 2 == 1
            assert (left["red"][1] + left_shift) + 1 == right["red"][1] - right_shift
            orientation = "H"
        else:
            assert ac == bc
            top, bottom = sorted((first, second), key=lambda item: item["red"][0])
            gap = int(bottom["red"][0] - top["red"][0] - 1)
            assert gap >= 1
            top_shift = gap // 2
            bottom_shift = gap - top_shift
            moved = _move_component(top, top_shift, 0, g)
            moved_right = _move_component(bottom, -bottom_shift, 0, g)
            axis_sum = int(top["red"][0] + top_shift + bottom["red"][0] - bottom_shift)
            assert axis_sum % 2 == 1
            assert (top["red"][0] + top_shift) + 1 == bottom["red"][0] - bottom_shift
            orientation = "V"

        merged = dict(moved)
        for pos, color in moved_right.items():
            if pos in merged:
                assert merged[pos] == int(color)
            merged[pos] = int(color)

        repaired = _repair_symmetry(merged, orientation, axis_sum, g.shape)
        for (r, c), color in repaired.items():
            assert out[r, c] == BLACK
            out[r, c] = int(color)

    return out


HALF_DOMAIN = [(p, a) for p in range(-2, 3) for a in range(-2, 1)]


def _enumerate_half_shapes() -> list[dict]:
    origin = (0, 0)
    others = [cell for cell in HALF_DOMAIN if cell != origin]
    shapes: list[dict] = []
    for size in range(3, 7):
        for extra in itertools.combinations(others, size - 1):
            cells = {origin, *extra}
            if not any(a < 0 for _, a in cells):
                continue
            rows = [p for p, _ in cells]
            if max(rows) - min(rows) > 3:
                continue
            if not _set_connected({(int(p), int(a)) for p, a in cells}):
                continue
            removable = []
            for cell in cells:
                if cell == origin:
                    continue
                remainder = set(cells)
                remainder.remove(cell)
                if len(remainder) >= 3 and _set_connected({(int(p), int(a)) for p, a in remainder}):
                    removable.append(cell)
            shapes.append(
                {
                    "cells": tuple(sorted((int(p), int(a)) for p, a in cells)),
                    "removable": tuple(sorted((int(p), int(a)) for p, a in removable)),
                }
            )
    return shapes


HALF_SHAPES = _enumerate_half_shapes()
MAIN_COLORS = tuple(color for color in range(10) if color not in {BLACK, RED})
GRID_HEIGHT_RANGE = (14, 24)
GRID_WIDTH_RANGE = (14, 24)
PAIR_COUNT_RANGE = (1, 3)
SLIDE_RANGE = (1, 4)
PLACE_ATTEMPTS = 80
GEN_RETRIES = 400


def _transform_generic(cell: tuple[int, int], orientation: str) -> tuple[int, int]:
    p, axial = cell
    if orientation == "H":
        return int(p), int(axial)
    return int(axial), int(p)


def _mirror_generic(cell: tuple[int, int]) -> tuple[int, int]:
    p, axial = cell
    return int(p), int(1 - axial)


def _choose_half_colors(
    cells: tuple[tuple[int, int], ...],
    main_color: int,
    accent_color: int | None,
) -> dict[tuple[int, int], int]:
    out = {cell: int(main_color) for cell in cells}
    out[(0, 0)] = RED
    if accent_color is not None:
        choices = [cell for cell in cells if cell != (0, 0)]
        out[random.choice(choices)] = int(accent_color)
    return out


def _object_values(
    cells: set[tuple[int, int]],
    base_colors: dict[tuple[int, int], int],
    mirrored: bool = False,
) -> list[int]:
    values = []
    for cell in cells:
        src = (cell[0], 1 - cell[1]) if mirrored else cell
        values.append(int(base_colors[src]))
    return values


def _normalize_pair(
    input_cells: dict[tuple[int, int], int],
    output_cells: dict[tuple[int, int], int],
) -> tuple[np.ndarray, np.ndarray]:
    coords = list(input_cells) + list(output_cells)
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    max_r = max(r for r, _ in coords)
    max_c = max(c for _, c in coords)
    h = int(max_r - min_r + 1)
    w = int(max_c - min_c + 1)

    inp = np.full((h, w), BLACK, dtype=int)
    out = np.full((h, w), BLACK, dtype=int)
    for (r, c), color in input_cells.items():
        inp[int(r - min_r), int(c - min_c)] = int(color)
    for (r, c), color in output_cells.items():
        out[int(r - min_r), int(c - min_c)] = int(color)
    return inp, out


def _build_pair_patch(main_color: int, accent_color: int | None) -> tuple[np.ndarray, np.ndarray] | None:
    orientation = random.choice(("H", "V"))
    defect_side = random.choice(("none", "left", "right"))
    slide = random.randint(*SLIDE_RANGE)

    shape_choices = HALF_SHAPES if defect_side == "none" else [shape for shape in HALF_SHAPES if shape["removable"]]
    shape = random.choice(shape_choices)
    cells = tuple(shape["cells"])
    base_colors = _choose_half_colors(cells, main_color, accent_color)

    removed = None
    if defect_side != "none":
        removed = random.choice(shape["removable"])

    left_cells = set(cells)
    right_cells = {_mirror_generic(cell) for cell in cells}

    if defect_side == "left":
        left_cells.remove(removed)
    elif defect_side == "right":
        right_cells.remove(_mirror_generic(removed))

    if not (3 <= len(left_cells) <= 6 and 3 <= len(right_cells) <= 6):
        return None
    if not _set_connected(left_cells):
        return None
    if not _set_connected(right_cells):
        return None
    if not _has_unique_dominant(_object_values(left_cells, base_colors), main_color):
        return None
    if not _has_unique_dominant(_object_values(right_cells, base_colors, mirrored=True), main_color):
        return None

    input_cells: dict[tuple[int, int], int] = {}
    output_cells: dict[tuple[int, int], int] = {}

    for cell in cells:
        color = int(base_colors[cell])
        left_out = _transform_generic(cell, orientation)
        right_out = _transform_generic(_mirror_generic(cell), orientation)
        output_cells[left_out] = int(main_color if color == RED else color)
        output_cells[right_out] = int(main_color if color == RED else color)

        if cell in left_cells:
            left_in_generic = (cell[0], cell[1] - slide)
            input_cells[_transform_generic(left_in_generic, orientation)] = int(color)
        right_cell = _mirror_generic(cell)
        if right_cell in right_cells:
            right_in_generic = (right_cell[0], right_cell[1] + slide)
            input_cells[_transform_generic(right_in_generic, orientation)] = int(color)

    inp, out = _normalize_pair(input_cells, output_cells)
    return inp, out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    coords = np.argwhere(mask)
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = True
    return out


def _stamp(canvas: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    region = canvas[top : top + h, left : left + w]
    mask = patch != BLACK
    assert not np.any((region != BLACK) & mask)
    region[mask] = patch[mask]


def _place_patch(
    inp: np.ndarray,
    out: np.ndarray,
    reserved: np.ndarray,
    patch_in: np.ndarray,
    patch_out: np.ndarray,
) -> bool:
    ph, pw = patch_in.shape
    h, w = inp.shape
    mask = (patch_in != BLACK) | (patch_out != BLACK)
    padded = np.zeros((ph + 2, pw + 2), dtype=bool)
    padded[1 : ph + 1, 1 : pw + 1] = mask
    padded = _dilate8(padded)

    if ph > h or pw > w:
        return False

    if h - ph < 2 or w - pw < 2:
        return False

    positions = [(r, c) for r in range(1, h - ph) for c in range(1, w - pw)]
    random.shuffle(positions)
    for top, left in positions[:PLACE_ATTEMPTS]:
        window = reserved[top - 1 : top + ph + 1, left - 1 : left + pw + 1]
        if np.any(window & padded):
            continue
        _stamp(inp, patch_in, top, left)
        _stamp(out, patch_out, top, left)
        window |= padded
        return True
    return False


def _generate_candidate() -> tuple[np.ndarray, np.ndarray] | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    main_colors = random.sample(MAIN_COLORS, pair_count)
    accent_pool = [color for color in MAIN_COLORS if color not in main_colors]

    specs: list[tuple[np.ndarray, np.ndarray]] = []
    for main_color in main_colors:
        accent_color = None
        if accent_pool and random.random() < 0.6:
            accent_color = random.choice(accent_pool)
        pair = _build_pair_patch(main_color, accent_color)
        if pair is None:
            return None
        specs.append(pair)

    min_h = max(patch_in.shape[0] for patch_in, _ in specs) + 2
    min_w = max(patch_in.shape[1] for patch_in, _ in specs) + 2
    if min_h > GRID_HEIGHT_RANGE[1] or min_w > GRID_WIDTH_RANGE[1]:
        return None

    h = random.randint(max(GRID_HEIGHT_RANGE[0], min_h), GRID_HEIGHT_RANGE[1])
    w = random.randint(max(GRID_WIDTH_RANGE[0], min_w), GRID_WIDTH_RANGE[1])
    inp = np.full((h, w), BLACK, dtype=int)
    out = np.full((h, w), BLACK, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    random.shuffle(specs)
    for patch_in, patch_out in specs:
        if not _place_patch(inp, out, reserved, patch_in, patch_out):
            return None

    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        pair = _generate_candidate()
        if pair is not None:
            return pair[0]
    raise RuntimeError("failed to generate valid input")
