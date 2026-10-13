from __future__ import annotations

import random

import numpy as np


BLACK = 0
YELLOW = 4
PALETTE = (1, 2, 3, 5, 6, 7, 8, 9)

STAGE_SIZE = 30
MAX_GENERATION_ATTEMPTS = 400
PAIR_COUNT_CHOICES = (1, 2, 3)
SMALL_CORE_CHOICES = ((3, 3), (3, 4), (4, 3))
TRANSFORMS = tuple(range(8))
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(int(r + dr), int(c + dc)) for dr, dc in DIR4]


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(cells: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    r0, c0, _r1, _c1 = _bbox(cells)
    return {(r - r0, c - c0) for r, c in cells}, (int(r0), int(c0))


def _shape_hw(cells: set[tuple[int, int]]) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(cells)
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _interior_cells(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for r, c in cells:
        if all((int(r + dr), int(c + dc)) in cells for dr, dc in DIR4):
            out.append((int(r), int(c)))
    return out


def _transform_point(
    point: tuple[int, int], height: int, width: int, transform: int
) -> tuple[int, int]:
    r, c = point
    if transform == 0:
        return int(r), int(c)
    if transform == 1:
        return int(r), int(width - 1 - c)
    if transform == 2:
        return int(height - 1 - r), int(c)
    if transform == 3:
        return int(height - 1 - r), int(width - 1 - c)
    if transform == 4:
        return int(c), int(r)
    if transform == 5:
        return int(c), int(height - 1 - r)
    if transform == 6:
        return int(width - 1 - c), int(r)
    return int(width - 1 - c), int(height - 1 - r)


def _apply_transform(
    cells: set[tuple[int, int]], marker: tuple[int, int], transform: int
) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    height, width = _shape_hw(cells)
    transformed = {
        _transform_point((int(r), int(c)), int(height), int(width), int(transform))
        for r, c in cells
    }
    transformed_marker = _transform_point(marker, int(height), int(width), int(transform))
    transformed, shift = _normalize(transformed)
    return transformed, (
        int(transformed_marker[0] - shift[0]),
        int(transformed_marker[1] - shift[1]),
    )


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _parse_shapes(grid: np.ndarray) -> dict[int, list[dict[str, object]]]:
    groups: dict[int, list[dict[str, object]]] = {}
    for comp in _components(grid != BLACK):
        coords = {(int(r), int(c)) for r, c in comp}
        values = {int(grid[r, c]) for r, c in coords}
        base_colors = values - {YELLOW}
        assert len(base_colors) == 1, "each shape must have exactly one non-yellow color"
        color = next(iter(base_colors))
        marker_cells = [(int(r), int(c)) for r, c in coords if int(grid[r, c]) == YELLOW]
        assert len(marker_cells) == 1, "each shape must contain exactly one yellow marker"
        groups.setdefault(int(color), []).append(
            {
                "color": int(color),
                "coords": coords,
                "marker": marker_cells[0],
                "area": int(len(coords)),
            }
        )
    return groups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove each small same-color shape by punching its marker-aligned footprint out of the larger partner.

    Input structure:
    - The background is black.
    - Each non-black connected component is a solid shape made of one base color plus exactly one yellow interior marker pixel.
    - For every base color, there are exactly two disconnected shapes: one larger and one smaller.

    Rule:
    - Pair the two shapes that share the same base color.
    - Treat the smaller shape as a stamp, translate it so that its yellow marker lands on the larger shape's yellow marker.
    - Remove every translated stamp cell from the larger shape by turning those cells black, and also erase the smaller shape from its original location.
    - All yellow marker pixels become black in the output.

    Key insight:
    - The visible color match only identifies which shapes belong together; the yellow interior markers encode the exact punch alignment.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    out[out == YELLOW] = BLACK

    for color, shapes in _parse_shapes(g).items():
        assert len(shapes) == 2, f"color {color} must appear in exactly one large/small pair"
        shapes = sorted(shapes, key=lambda rec: int(rec["area"]))
        small = shapes[0]
        large = shapes[1]

        for r, c in small["coords"]:
            out[int(r), int(c)] = BLACK

        dr = int(large["marker"][0] - small["marker"][0])
        dc = int(large["marker"][1] - small["marker"][1])
        for r, c in small["coords"]:
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                out[rr, cc] = BLACK

    out[out == YELLOW] = BLACK
    return out


def _make_small_shape() -> tuple[set[tuple[int, int]], tuple[int, int]]:
    core_h, core_w = random.choice(SMALL_CORE_CHOICES)
    cells = {(r, c) for r in range(core_h) for c in range(core_w)}

    ring = (
        {(r, -1) for r in range(core_h)}
        | {(r, core_w) for r in range(core_h)}
        | {(-1, c) for c in range(core_w)}
        | {(core_h, c) for c in range(core_w)}
    )
    max_extra = 3 if (core_h, core_w) == (3, 3) else 1
    n_extra = random.randint(0, max_extra)
    if n_extra > 0:
        cells |= set(random.sample(sorted(ring), n_extra))

    cells, _shift = _normalize(cells)
    marker = random.choice(_interior_cells(cells))
    transform = random.choice(TRANSFORMS)
    return _apply_transform(cells, marker, transform)


def _grow_big_shape(
    small_cells: set[tuple[int, int]], small_marker: tuple[int, int]
) -> tuple[set[tuple[int, int]], tuple[int, int]] | None:
    small_area = len(small_cells)
    target_area = random.randint(int(4 * small_area), int(4 * small_area + max(2, small_area // 2)))

    for _ in range(80):
        cells = set(small_cells)
        bias = random.choice(
            ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
        )
        max_h = random.randint(max(6, _shape_hw(small_cells)[0] + 3), 10)
        max_w = random.randint(max(6, _shape_hw(small_cells)[1] + 3), 10)

        for _step in range(600):
            if len(cells) >= target_area:
                norm_cells, shift = _normalize(cells)
                norm_marker = (int(small_marker[0] - shift[0]), int(small_marker[1] - shift[1]))
                if norm_marker in _interior_cells(norm_cells):
                    return norm_cells, norm_marker
                break

            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(int(r), int(c)):
                    if (int(nr), int(nc)) not in cells:
                        frontier.add((int(nr), int(nc)))

            valid: list[tuple[int, int]] = []
            weights: list[int] = []
            for cand in frontier:
                new_cells = set(cells)
                new_cells.add(cand)
                h, w = _shape_hw(new_cells)
                if h > max_h or w > max_w:
                    continue
                adj = sum((cand[0] + dr, cand[1] + dc) in cells for dr, dc in DIR4)
                weight = int(max(1, adj))
                if cand[0] * bias[0] + cand[1] * bias[1] > 0:
                    weight += 1
                valid.append(cand)
                weights.append(weight)

            if not valid:
                break
            cells.add(random.choices(valid, weights=weights, k=1)[0])

    return None


def _shape_to_array(
    cells: set[tuple[int, int]], marker: tuple[int, int], color: int
) -> np.ndarray:
    h, w = _shape_hw(cells)
    arr = np.full((h, w), BLACK, dtype=int)
    for r, c in cells:
        arr[int(r), int(c)] = int(color)
    arr[int(marker[0]), int(marker[1])] = YELLOW
    return arr


def _can_place(occupied: np.ndarray, arr: np.ndarray, top: int, left: int, gap: int = 1) -> bool:
    h, w = arr.shape
    if top < 1 or left < 1 or top + h >= STAGE_SIZE or left + w >= STAGE_SIZE:
        return False
    r0 = max(0, int(top - gap))
    c0 = max(0, int(left - gap))
    r1 = min(STAGE_SIZE, int(top + h + gap))
    c1 = min(STAGE_SIZE, int(left + w + gap))
    return not bool(occupied[r0:r1, c0:c1].any())


def _place_array(
    grid: np.ndarray, occupied: np.ndarray, arr: np.ndarray, top: int, left: int
) -> None:
    h, w = arr.shape
    patch = arr != BLACK
    target = grid[top : top + h, left : left + w]
    target[patch] = arr[patch]
    grid[top : top + h, left : left + w] = target
    occupied[top : top + h, left : left + w][patch] = True


def _build_oracle(grid: np.ndarray, pair_specs: list[dict[str, object]]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    out[out == YELLOW] = BLACK
    for spec in pair_specs:
        small_cells = spec["small_abs_cells"]
        for r, c in small_cells:
            out[int(r), int(c)] = BLACK

        big_marker = spec["big_abs_marker"]
        small_marker = spec["small_abs_marker"]
        dr = int(big_marker[0] - small_marker[0])
        dc = int(big_marker[1] - small_marker[1])
        for r, c in small_cells:
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                out[int(rr), int(cc)] = BLACK
    return out


def _generate_attempt() -> np.ndarray | None:
    num_pairs = random.choice(PAIR_COUNT_CHOICES)
    colors = random.sample(PALETTE, num_pairs)

    pairs: list[dict[str, object]] = []
    for color in colors:
        small = _make_small_shape()
        big = _grow_big_shape(*small)
        if big is None:
            return None
        pairs.append(
            {
                "color": int(color),
                "small_cells": small[0],
                "small_marker": small[1],
                "big_cells": big[0],
                "big_marker": big[1],
            }
        )

    grid = np.full((STAGE_SIZE, STAGE_SIZE), BLACK, dtype=int)
    occupied = np.zeros((STAGE_SIZE, STAGE_SIZE), dtype=bool)

    objects: list[dict[str, object]] = []
    for idx, pair in enumerate(pairs):
        objects.append(
            {
                "pair_idx": int(idx),
                "kind": "big",
                "cells": pair["big_cells"],
                "marker": pair["big_marker"],
                "color": pair["color"],
            }
        )
        objects.append(
            {
                "pair_idx": int(idx),
                "kind": "small",
                "cells": pair["small_cells"],
                "marker": pair["small_marker"],
                "color": pair["color"],
            }
        )

    random.shuffle(objects)
    objects.sort(key=lambda rec: _shape_hw(rec["cells"])[0] * _shape_hw(rec["cells"])[1], reverse=True)

    placed_pairs = [
        {
            "color": pair["color"],
            "small_abs_cells": None,
            "small_abs_marker": None,
            "big_abs_marker": None,
        }
        for pair in pairs
    ]

    for obj in objects:
        arr = _shape_to_array(obj["cells"], obj["marker"], int(obj["color"]))
        h, w = arr.shape
        placed = False
        for _ in range(250):
            top = random.randint(1, STAGE_SIZE - h - 1)
            left = random.randint(1, STAGE_SIZE - w - 1)
            if not _can_place(occupied, arr, int(top), int(left)):
                continue
            _place_array(grid, occupied, arr, int(top), int(left))
            abs_cells = {(int(top + r), int(left + c)) for r, c in obj["cells"]}
            abs_marker = (int(top + obj["marker"][0]), int(left + obj["marker"][1]))
            rec = placed_pairs[int(obj["pair_idx"])]
            if obj["kind"] == "small":
                rec["small_abs_cells"] = abs_cells
                rec["small_abs_marker"] = abs_marker
            else:
                rec["big_abs_marker"] = abs_marker
            placed = True
            break
        if not placed:
            return None

    if any(spec["small_abs_cells"] is None or spec["small_abs_marker"] is None or spec["big_abs_marker"] is None for spec in placed_pairs):
        return None

    coords = np.argwhere(grid != BLACK)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    top_margin = random.randint(0, min(3, int(r0)))
    left_margin = random.randint(0, min(3, int(c0)))
    bottom_margin = random.randint(0, min(3, int(STAGE_SIZE - 1 - r1)))
    right_margin = random.randint(0, min(3, int(STAGE_SIZE - 1 - c1)))
    crop_r0 = int(r0 - top_margin)
    crop_c0 = int(c0 - left_margin)
    crop_r1 = int(r1 + bottom_margin)
    crop_c1 = int(c1 + right_margin)
    inp = grid[crop_r0 : crop_r1 + 1, crop_c0 : crop_c1 + 1].copy()

    cropped_specs: list[dict[str, object]] = []
    for spec in placed_pairs:
        cropped_specs.append(
            {
                "small_abs_cells": {
                    (int(r - crop_r0), int(c - crop_c0)) for r, c in spec["small_abs_cells"]
                },
                "small_abs_marker": (
                    int(spec["small_abs_marker"][0] - crop_r0),
                    int(spec["small_abs_marker"][1] - crop_c0),
                ),
                "big_abs_marker": (
                    int(spec["big_abs_marker"][0] - crop_r0),
                    int(spec["big_abs_marker"][1] - crop_c0),
                ),
            }
        )

    expected = _build_oracle(inp, cropped_specs)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid punching-pairs ARC input")
