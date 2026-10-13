from __future__ import annotations

import random
import numpy as np


PALETTE = (1, 2, 3, 4, 7)
AREA_VALUES = tuple(range(2, 8))
GENERATE_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _normalize(coords: list[tuple[int, int]]) -> np.ndarray:
    arr = np.asarray(coords, dtype=int)
    arr[:, 0] -= int(arr[:, 0].min())
    arr[:, 1] -= int(arr[:, 1].min())
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    return arr[order]


def _find_components(grid: np.ndarray, bg: int = 0) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords_arr = np.asarray(coords, dtype=int)
            bbox = (
                int(coords_arr[:, 0].min()),
                int(coords_arr[:, 1].min()),
                int(coords_arr[:, 0].max()),
                int(coords_arr[:, 1].max()),
            )
            components.append(
                {
                    "color": color,
                    "coords": coords_arr,
                    "area": int(len(coords)),
                    "bbox": bbox,
                }
            )

    return components


def _paint_shifted(
    shape: tuple[int, int], components: list[dict], *, descending_area: bool = True
) -> np.ndarray:
    h, w = shape
    out = np.zeros((h, w), dtype=int)
    order = sorted(components, key=lambda comp: int(comp["area"]), reverse=descending_area)
    for comp in order:
        area = int(comp["area"])
        color = int(comp["color"])
        coords = np.asarray(comp["coords"], dtype=int)
        for r, c in coords:
            cc = int(c) + area
            if 0 <= cc < w:
                out[int(r), cc] = color
    return out


def _paint_uniform_shift(shape: tuple[int, int], components: list[dict], shift: int) -> np.ndarray:
    h, w = shape
    out = np.zeros((h, w), dtype=int)
    for comp in components:
        color = int(comp["color"])
        coords = np.asarray(comp["coords"], dtype=int)
        for r, c in coords:
            cc = int(c) + int(shift)
            if 0 <= cc < w:
                out[int(r), cc] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Shift each object right by its own area, then stamp smaller-area objects on top of larger ones.

    Input structure:
    - A black background contains 3-5 separated 4-connected monochromatic objects in distinct colors.
    - Each object is an arbitrary connected polyomino, and object areas are pairwise distinct.
    - The objects start on the left side of the grid and may overlap after their rightward moves.

    Rule:
    - Find every 4-connected monochromatic component and compute its area A.
    - Clear the canvas to black, sort the components by area from large to small, and redraw each one shifted A columns right.
    - Clip cells that move past the right edge, and let later-drawn smaller-area objects overwrite larger ones on overlaps.

    Key insight:
    - The same quantity, area, controls both how far an object moves and which object wins overlaps, so solving requires tracking both effects together.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    components = _find_components(g, bg=0)
    assert 3 <= len(components) <= 5, "expected 3-5 foreground objects"

    areas = [int(comp["area"]) for comp in components]
    assert len(set(areas)) == len(areas), "object areas must be unique"
    for comp in components:
        coords = np.asarray(comp["coords"], dtype=int)
        color = int(comp["color"])
        assert np.all(g[coords[:, 0], coords[:, 1]] == color), "components must be monochromatic"

    return _paint_shifted(g.shape, components, descending_area=True)


def _random_polyomino(area: int) -> np.ndarray:
    for _ in range(200):
        cells = {(0, 0)}
        frontier = {(0, 1), (1, 0), (0, -1), (-1, 0)}
        while len(cells) < area:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            r, c = cell
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) not in cells:
                    frontier.add((nr, nc))

        arr = np.asarray(list(cells), dtype=int)
        rot = random.randrange(4)
        for _ in range(rot):
            arr = np.column_stack((arr[:, 1], -arr[:, 0]))
        if random.random() < 0.5:
            arr[:, 1] *= -1
        arr = _normalize([tuple(cell) for cell in arr])

        bbox_area = (int(arr[:, 0].max()) + 1) * (int(arr[:, 1].max()) + 1)
        if area >= 4 and bbox_area == area:
            continue
        return arr

    raise RuntimeError(f"failed to generate polyomino with area {area}")


def _cells_from_anchor(coords: np.ndarray, anchor: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = anchor
    return {(int(r0 + r), int(c0 + c)) for r, c in coords}


def _visible_shifted_cells(cells: set[tuple[int, int]], area: int, width: int) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        cc = int(c) + int(area)
        if 0 <= cc < width:
            out.add((int(r), cc))
    return out


def _separated8(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r1, c1 in a:
        for r2, c2 in b:
            if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                return False
    return True


def _build_specs(count: int) -> list[dict]:
    areas = random.sample(AREA_VALUES, count)
    colors = random.sample(PALETTE, count)
    specs = []
    for area, color in zip(areas, colors):
        shape = _random_polyomino(area)
        specs.append(
            {
                "area": int(area),
                "color": int(color),
                "shape": shape,
                "h": int(shape[:, 0].max()) + 1,
                "w": int(shape[:, 1].max()) + 1,
            }
        )
    return specs


def _try_place_forced_pair(
    specs: list[dict], height: int, width: int, left_limit: int
) -> tuple[dict[int, tuple[int, int]], tuple[int, int, set[tuple[int, int]]]] | None:
    pair_indices = [(i, j) for i in range(len(specs)) for j in range(len(specs)) if specs[i]["area"] > specs[j]["area"]]
    random.shuffle(pair_indices)

    for big_idx, small_idx in pair_indices:
        big = specs[big_idx]
        small = specs[small_idx]
        big_shape = np.asarray(big["shape"], dtype=int)
        small_shape = np.asarray(small["shape"], dtype=int)
        big_row_max = height - big["h"]
        big_col_max = left_limit - big["w"]
        small_row_max = height - small["h"]
        small_col_max = left_limit - small["w"]
        if min(big_row_max, big_col_max, small_row_max, small_col_max) < 0:
            continue

        big_cells = [tuple(cell) for cell in big_shape]
        small_cells = [tuple(cell) for cell in small_shape]
        candidate_pairs = [(bc, sc) for bc in big_cells for sc in small_cells]
        random.shuffle(candidate_pairs)

        for (br, bc), (sr, sc) in candidate_pairs:
            row_delta = int(br) - int(sr)
            col_delta = int(bc) + int(big["area"]) - int(sc) - int(small["area"])

            row_lo = max(0, -row_delta)
            row_hi = min(big_row_max, small_row_max - row_delta)
            col_lo = max(0, -col_delta)
            col_hi = min(big_col_max, small_col_max - col_delta)
            if row_lo > row_hi or col_lo > col_hi:
                continue

            row_choices = list(range(row_lo, row_hi + 1))
            col_choices = list(range(col_lo, col_hi + 1))
            random.shuffle(row_choices)
            random.shuffle(col_choices)

            for big_r0 in row_choices[: min(len(row_choices), 10)]:
                for big_c0 in col_choices[: min(len(col_choices), 10)]:
                    small_r0 = big_r0 + row_delta
                    small_c0 = big_c0 + col_delta

                    big_input = _cells_from_anchor(big_shape, (big_r0, big_c0))
                    small_input = _cells_from_anchor(small_shape, (small_r0, small_c0))
                    if not _separated8(big_input, small_input):
                        continue

                    big_out = _visible_shifted_cells(big_input, int(big["area"]), width)
                    small_out = _visible_shifted_cells(small_input, int(small["area"]), width)
                    overlap = big_out & small_out
                    if not (1 <= len(overlap) <= 3):
                        continue

                    return {big_idx: (big_r0, big_c0), small_idx: (small_r0, small_c0)}, (
                        big_idx,
                        small_idx,
                        overlap,
                    )

    return None


def _place_remaining_objects(
    specs: list[dict],
    anchors: dict[int, tuple[int, int]],
    height: int,
    width: int,
    left_limit: int,
) -> dict[int, tuple[int, int]] | None:
    occupied_inputs = {
        idx: _cells_from_anchor(np.asarray(spec["shape"], dtype=int), anchor)
        for idx, (spec, anchor) in (
            (idx, (specs[idx], anchors[idx])) for idx in anchors
        )
    }
    occupied_outputs = {
        idx: _visible_shifted_cells(cells, int(specs[idx]["area"]), width) for idx, cells in occupied_inputs.items()
    }

    remaining = [idx for idx in range(len(specs)) if idx not in anchors]
    random.shuffle(remaining)
    for idx in remaining:
        spec = specs[idx]
        shape = np.asarray(spec["shape"], dtype=int)
        row_candidates = list(range(0, height - spec["h"] + 1))
        col_candidates = list(range(0, left_limit - spec["w"] + 1))
        random.shuffle(row_candidates)
        random.shuffle(col_candidates)

        options: list[tuple[int, int]] = []
        for r0 in row_candidates:
            for c0 in col_candidates:
                input_cells = _cells_from_anchor(shape, (r0, c0))
                if any(not _separated8(input_cells, other) for other in occupied_inputs.values()):
                    continue

                output_cells = _visible_shifted_cells(input_cells, int(spec["area"]), width)
                if any(output_cells & other for other in occupied_outputs.values()):
                    continue
                options.append((r0, c0))

        if not options:
            return None

        anchor = random.choice(options)
        anchors[idx] = anchor
        input_cells = _cells_from_anchor(shape, anchor)
        occupied_inputs[idx] = input_cells
        occupied_outputs[idx] = _visible_shifted_cells(input_cells, int(spec["area"]), width)

    return anchors


def _render_input(height: int, width: int, specs: list[dict], anchors: dict[int, tuple[int, int]]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for idx, spec in enumerate(specs):
        color = int(spec["color"])
        for r, c in _cells_from_anchor(np.asarray(spec["shape"], dtype=int), anchors[idx]):
            grid[r, c] = color
    return grid


def _semantic_checks(grid: np.ndarray, left_limit: int) -> bool:
    components = _find_components(grid, bg=0)
    if not (3 <= len(components) <= 5):
        return False

    areas = [int(comp["area"]) for comp in components]
    if len(set(areas)) != len(areas):
        return False
    if any(area < 2 or area > 7 for area in areas):
        return False
    if any(comp["color"] not in PALETTE for comp in components):
        return False
    if any(int(comp["bbox"][3]) >= left_limit for comp in components):
        return False

    actual = main(grid)
    reverse = _paint_shifted(grid.shape, components, descending_area=False)
    if np.array_equal(actual, reverse):
        return False

    smallest = min(areas)
    largest = max(areas)
    if np.array_equal(actual, _paint_uniform_shift(grid.shape, components, smallest)):
        return False
    if np.array_equal(actual, _paint_uniform_shift(grid.shape, components, largest)):
        return False

    output_colors = set(map(int, np.unique(actual))) - {0}
    input_colors = {int(comp["color"]) for comp in components}
    if output_colors != input_colors:
        return False

    total_overlaps = 0
    priority_witness = False
    for i, comp_a in enumerate(components):
        area_a = int(comp_a["area"])
        area_b = -1
        cells_a = _visible_shifted_cells(set(map(tuple, comp_a["coords"])), area_a, grid.shape[1])
        for comp_b in components[i + 1 :]:
            area_b = int(comp_b["area"])
            cells_b = _visible_shifted_cells(set(map(tuple, comp_b["coords"])), area_b, grid.shape[1])
            overlap = cells_a & cells_b
            total_overlaps += len(overlap)
            if not overlap:
                continue
            smaller = comp_a if area_a < area_b else comp_b
            larger = comp_b if smaller is comp_a else comp_a
            if any(int(actual[r, c]) == int(smaller["color"]) and int(reverse[r, c]) == int(larger["color"]) for r, c in overlap):
                priority_witness = True
    if total_overlaps == 0:
        return False
    if total_overlaps > 4:
        return False
    if not priority_witness:
        return False

    return not np.array_equal(actual, grid)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(12, 18)
    width = random.randint(18, 24)
    count = random.randint(3, 5)
    left_limit = max(6, width // 3 + 1)

    specs = _build_specs(count)
    forced = _try_place_forced_pair(specs, height, width, left_limit)
    if forced is None:
        return None
    anchors, _forced_pair = forced

    anchors = _place_remaining_objects(specs, anchors, height, width, left_limit)
    if anchors is None:
        return None

    grid = _render_input(height, width, specs, anchors)
    if not _semantic_checks(grid, left_limit):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
