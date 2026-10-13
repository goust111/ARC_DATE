from __future__ import annotations

import random
from typing import Iterable

import numpy as np


LATTICE = 5
BACKGROUND = 0
INDICATOR_COLORS = (1, 2, 3, 4)
SIDE_TO_DELTA = {
    "N": (-1, 0),
    "S": (1, 0),
    "W": (0, -1),
    "E": (0, 1),
}


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _parse_cells(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    lattice = int(g[0, 0])
    assert lattice == LATTICE
    assert np.all(g[0, :] == lattice)
    assert np.all(g[-1, :] == lattice)
    assert np.all(g[:, 0] == lattice)
    assert np.all(g[:, -1] == lattice)

    comps = _components(g != lattice)
    assert len(comps) == 16

    cells = []
    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        assert int(coords.shape[0]) == (r1 - r0 + 1) * (c1 - c0 + 1)
        cells.append({"bbox": (r0, c0, r1, c1)})

    row_starts = sorted({cell["bbox"][0] for cell in cells})
    col_starts = sorted({cell["bbox"][1] for cell in cells})
    assert len(row_starts) == 4 and len(col_starts) == 4

    for cell in cells:
        r0, c0, r1, c1 = cell["bbox"]
        ri = row_starts.index(r0)
        ci = col_starts.index(c0)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        colors = sorted(set(int(x) for x in patch.ravel()) - {BACKGROUND})
        if colors:
            assert len(colors) == 1
            color = int(colors[0])
            assert color in INDICATOR_COLORS
            locs = np.argwhere(patch == color)
            assert len(locs) == 1
            rr = int(r0 + locs[0, 0])
            cc = int(c0 + locs[0, 1])
            sides = []
            if rr == r0:
                sides.append("N")
            if rr == r1:
                sides.append("S")
            if cc == c0:
                sides.append("W")
            if cc == c1:
                sides.append("E")
            assert len(sides) == 1
            indicator = {"color": color, "pos": (rr, cc), "side": sides[0]}
        else:
            indicator = None
        cell.update({"row": ri, "col": ci, "indicator": indicator})

    return sorted(cells, key=lambda cell: (cell["row"], cell["col"]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split a 4x4 lattice into edge and interior cells, then extend or fill from each indicator pixel.

    Input structure:
    - A 1-pixel grey lattice covers the whole outer border and partitions the canvas into a 4x4 array of rectangular black cell interiors.
    - Some cells contain exactly one colored indicator pixel chosen from blue, red, green, or yellow.
    - Each indicator touches exactly one wall of its own cell interior, so that wall determines a cardinal side.

    Rule:
    - Parse the 16 cell interiors from the lattice and assign each one an array row and column.
    - If an indicated cell lies on the outer row or outer column of the 4x4 array, draw a 1-pixel ray in the indicator color from that pixel straight to the canvas edge in the touched-wall direction, overwriting lattice pixels on the path.
    - If an indicated cell is strictly interior, fill that entire cell interior rectangle with the indicator color.
    - Empty cells stay black, and lattice pixels not crossed by a ray stay grey.

    Key insight:
    - The same visible object, a single wall-touching indicator pixel, changes meaning based on lattice position: edge cells use it as a direction source for an outward ray, while interior cells use it as a fill color for the whole cell.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = out.shape

    for cell in _parse_cells(g):
        indicator = cell["indicator"]
        if indicator is None:
            continue

        color = int(indicator["color"])
        r0, c0, r1, c1 = cell["bbox"]
        is_edge = cell["row"] in {0, 3} or cell["col"] in {0, 3}
        if is_edge:
            dr, dc = SIDE_TO_DELTA[indicator["side"]]
            rr, cc = indicator["pos"]
            while 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = color
                rr += dr
                cc += dc
        else:
            out[r0 : r1 + 1, c0 : c1 + 1] = color

    return out


def _cell_bbox(cell_h: int, cell_w: int, row: int, col: int) -> tuple[int, int, int, int]:
    r0 = 1 + row * (cell_h + 1)
    c0 = 1 + col * (cell_w + 1)
    return r0, c0, r0 + cell_h - 1, c0 + cell_w - 1


def _build_lattice(cell_h: int, cell_w: int) -> np.ndarray:
    h = 4 * cell_h + 5
    w = 4 * cell_w + 5
    grid = np.full((h, w), BACKGROUND, dtype=int)
    for row in range(5):
        grid[row * (cell_h + 1), :] = LATTICE
    for col in range(5):
        grid[:, col * (cell_w + 1)] = LATTICE
    return grid


def _place_indicator(
    grid: np.ndarray,
    cell_h: int,
    cell_w: int,
    row: int,
    col: int,
    side: str,
    color: int,
) -> tuple[int, int]:
    r0, c0, r1, c1 = _cell_bbox(cell_h, cell_w, row, col)
    if side == "N":
        rr = r0
        cc = random.randint(c0 + 1, c1 - 1)
    elif side == "S":
        rr = r1
        cc = random.randint(c0 + 1, c1 - 1)
    elif side == "W":
        rr = random.randint(r0 + 1, r1 - 1)
        cc = c0
    else:
        rr = random.randint(r0 + 1, r1 - 1)
        cc = c1
    grid[rr, cc] = int(color)
    return rr, cc


def _ray_coords(start: tuple[int, int], side: str, shape: tuple[int, int]) -> set[tuple[int, int]]:
    dr, dc = SIDE_TO_DELTA[side]
    h, w = shape
    rr, cc = start
    coords = set()
    while 0 <= rr < h and 0 <= cc < w:
        coords.add((rr, cc))
        rr += dr
        cc += dc
    return coords


def _render_expected(
    inp: np.ndarray,
    cell_h: int,
    cell_w: int,
    specs: Iterable[dict],
) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        color = int(spec["color"])
        row = int(spec["row"])
        col = int(spec["col"])
        side = str(spec["side"])
        rr, cc = spec["pos"]
        if spec["kind"] == "edge":
            dr, dc = SIDE_TO_DELTA[side]
            while 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                out[rr, cc] = color
                rr += dr
                cc += dc
        else:
            r0, c0, r1, c1 = _cell_bbox(cell_h, cell_w, row, col)
            out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, specs: list[dict], cell_h: int, cell_w: int) -> bool:
    if not np.array_equal(main(inp), expected):
        return False

    edge_specs = [spec for spec in specs if spec["kind"] == "edge"]
    interior_specs = [spec for spec in specs if spec["kind"] == "interior"]
    if len(edge_specs) != 2 or len(interior_specs) not in {3, 4}:
        return False

    if len({spec["color"] for spec in specs}) < 2:
        return False

    pair_exists = any(
        edge["color"] == interior["color"] and edge["side"] == interior["side"]
        for edge in edge_specs
        for interior in interior_specs
    )
    if not pair_exists:
        return False

    ray_sets = [_ray_coords(spec["pos"], spec["side"], inp.shape) for spec in edge_specs]
    if ray_sets[0] & ray_sets[1]:
        return False

    for ray in ray_sets:
        for spec in interior_specs:
            r0, c0, r1, c1 = _cell_bbox(cell_h, cell_w, spec["row"], spec["col"])
            interior_box = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
            if ray & interior_box:
                return False

    parsed = _parse_cells(inp)
    parsed_specs = []
    for cell in parsed:
        indicator = cell["indicator"]
        if indicator is None:
            continue
        parsed_specs.append(
            {
                "row": cell["row"],
                "col": cell["col"],
                "color": int(indicator["color"]),
                "side": str(indicator["side"]),
                "kind": "edge" if cell["row"] in {0, 3} or cell["col"] in {0, 3} else "interior",
            }
        )
    expected_specs = [
        {
            "row": int(spec["row"]),
            "col": int(spec["col"]),
            "color": int(spec["color"]),
            "side": str(spec["side"]),
            "kind": str(spec["kind"]),
        }
        for spec in specs
    ]
    if sorted(parsed_specs, key=lambda item: (item["row"], item["col"])) != sorted(
        expected_specs, key=lambda item: (item["row"], item["col"])
    ):
        return False

    untouched_lattice = int(np.sum((inp == LATTICE) & (expected == LATTICE)))
    if untouched_lattice <= 0:
        return False

    return True


def _edge_layout(axis: str) -> list[tuple[int, int, str]]:
    if axis == "vertical":
        north_cell = (random.randrange(4), 0 if random.random() < 0.5 else 3, "N")
        south_cell = (random.randrange(4), 3 if north_cell[1] == 0 else 0, "S")
        return [north_cell, south_cell]

    west_cell = (0 if random.random() < 0.5 else 3, random.randrange(4), "W")
    east_cell = (3 if west_cell[0] == 0 else 0, random.randrange(4), "E")
    return [west_cell, east_cell]


def _gen_attempt() -> np.ndarray | None:
    cell_h = random.randint(3, 5)
    cell_w = random.randint(3, 5)
    grid = _build_lattice(cell_h, cell_w)

    axis = random.choice(["vertical", "horizontal"])
    edge_cells = _edge_layout(axis)
    interior_cells = random.sample([(1, 1), (1, 2), (2, 1), (2, 2)], random.randint(3, 4))

    palette = list(INDICATOR_COLORS)
    random.shuffle(palette)
    pair_color = palette[0]
    other_edge_color = palette[1]
    extra_colors = palette

    specs: list[dict] = []

    pair_row, pair_col, pair_side = edge_cells[0]
    pair_pos = _place_indicator(grid, cell_h, cell_w, pair_row, pair_col, pair_side, pair_color)
    specs.append(
        {
            "row": pair_row,
            "col": pair_col,
            "side": pair_side,
            "color": pair_color,
            "kind": "edge",
            "pos": pair_pos,
        }
    )

    edge_row, edge_col, edge_side = edge_cells[1]
    edge_pos = _place_indicator(grid, cell_h, cell_w, edge_row, edge_col, edge_side, other_edge_color)
    specs.append(
        {
            "row": edge_row,
            "col": edge_col,
            "side": edge_side,
            "color": other_edge_color,
            "kind": "edge",
            "pos": edge_pos,
        }
    )

    paired_interior = interior_cells[0]
    pair_fill_pos = _place_indicator(
        grid, cell_h, cell_w, paired_interior[0], paired_interior[1], pair_side, pair_color
    )
    specs.append(
        {
            "row": paired_interior[0],
            "col": paired_interior[1],
            "side": pair_side,
            "color": pair_color,
            "kind": "interior",
            "pos": pair_fill_pos,
        }
    )

    for row, col in interior_cells[1:]:
        side = random.choice(["N", "S", "W", "E"])
        color = random.choice(extra_colors)
        pos = _place_indicator(grid, cell_h, cell_w, row, col, side, color)
        specs.append(
            {
                "row": row,
                "col": col,
                "side": side,
                "color": color,
                "kind": "interior",
                "pos": pos,
            }
        )

    expected = _render_expected(grid, cell_h, cell_w, specs)
    if not _semantic_checks(grid, expected, specs, cell_h, cell_w):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
