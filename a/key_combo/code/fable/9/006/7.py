from __future__ import annotations

import random
from collections import Counter

import numpy as np


PALETTE = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _parse_separator_table(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    bg = _get_bg(g)

    row_colors = [int(g[r, 0]) if np.all(g[r, :] == g[r, 0]) else None for r in range(h)]
    col_colors = [int(g[0, c]) if np.all(g[:, c] == g[0, c]) else None for c in range(w)]

    row_candidates = {color for color in row_colors if color is not None and color != bg}
    col_candidates = {color for color in col_colors if color is not None and color != bg}
    separator_candidates = row_candidates & col_candidates
    assert len(separator_candidates) == 1, "expected a unique separator color"
    sep = next(iter(separator_candidates))

    sep_rows = [r for r in range(h) if np.all(g[r, :] == sep)]
    sep_cols = [c for c in range(w) if np.all(g[:, c] == sep)]
    cell_rows = len(sep_rows) + 1
    cell_cols = len(sep_cols) + 1

    row_spans: list[tuple[int, int]] = []
    prev = -1
    for stop in sep_rows + [h]:
        row_spans.append((prev + 1, stop - 1))
        prev = stop

    col_spans: list[tuple[int, int]] = []
    prev = -1
    for stop in sep_cols + [w]:
        col_spans.append((prev + 1, stop - 1))
        prev = stop

    assert len(row_spans) == cell_rows and len(col_spans) == cell_cols
    assert all(r1 >= r0 for r0, r1 in row_spans), "empty row band"
    assert all(c1 >= c0 for c0, c1 in col_spans), "empty col band"
    assert all(r1 - r0 + 1 == cell_rows for r0, r1 in row_spans), "cell height must equal row count"
    assert all(c1 - c0 + 1 == cell_cols for c0, c1 in col_spans), "cell width must equal col count"

    return {
        "grid": g,
        "bg": bg,
        "sep": sep,
        "row_spans": row_spans,
        "col_spans": col_spans,
        "cell_rows": cell_rows,
        "cell_cols": cell_cols,
    }


def _extract_candidates(layout: dict) -> list[dict]:
    g = layout["grid"]
    bg = int(layout["bg"])
    sep = int(layout["sep"])
    row_spans = layout["row_spans"]
    col_spans = layout["col_spans"]
    cell_rows = int(layout["cell_rows"])
    cell_cols = int(layout["cell_cols"])

    candidates: list[dict] = []
    for rr, (r0, r1) in enumerate(row_spans):
        for cc, (c0, c1) in enumerate(col_spans):
            cell = g[r0 : r1 + 1, c0 : c1 + 1]
            assert cell.shape == (cell_rows, cell_cols)
            colors = sorted(set(map(int, cell.ravel())) - {bg})
            if not colors:
                continue
            assert sep not in colors, "separator color may not appear inside a cell"
            assert len(colors) == 1, "every occupied cell must contain exactly one monochrome pattern"
            color = colors[0]
            mask = cell == color
            candidates.append({"cell": (rr, cc), "color": color, "mask": mask})
    return candidates


def _build_output(layout: dict, key: dict) -> np.ndarray:
    bg = int(layout["bg"])
    sep = int(layout["sep"])
    row_spans = layout["row_spans"]
    col_spans = layout["col_spans"]
    out = np.full_like(layout["grid"], bg)

    for r0, r1 in row_spans[1:]:
        out[r0 - 1, :] = sep
    for c0, c1 in col_spans[1:]:
        out[:, c0 - 1] = sep

    color = int(key["color"])
    mask = key["mask"]
    for rr, (r0, r1) in enumerate(row_spans):
        for cc, (c0, c1) in enumerate(col_spans):
            if bool(mask[rr, cc]):
                out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the only self-referential mini-map and light the table cells that its map selects.

    Input structure:
    - A full separator scaffold divides the canvas into `R x C` table cells.
    - Every cell interior is exactly `R x C` pixels, so each cell is a mini-map of the whole table.
    - Some cells contain one monochrome pixel pattern of a unique color; the remaining cells are empty background.

    Rule:
    - For each occupied cell, inspect its local `R x C` pattern and test whether it contains the pixel
      whose local coordinates equal that cell's own table coordinates.
    - Exactly one pattern passes this self-reference check; that cell is the key.
    - Preserve the separator lines, erase all original cell contents, and fill every table cell whose
      coordinates are marked by the key pattern with the key color.

    Key insight:
    - The pattern inside a cell is not just decoration: it is a shrunken map of the whole separator grid,
      and only the map that marks its own host coordinates is globally self-consistent.
    """
    layout = _parse_separator_table(np.asarray(grid, dtype=int))
    candidates = _extract_candidates(layout)
    matches = [cand for cand in candidates if bool(cand["mask"][cand["cell"][0], cand["cell"][1]])]
    assert len(matches) == 1, "expected exactly one self-referential pattern"
    return _build_output(layout, matches[0])


def _is_border_cell(cell: tuple[int, int], rows: int, cols: int) -> bool:
    r, c = cell
    return r in {0, rows - 1} or c in {0, cols - 1}


def _choose_key_cell(all_cells: list[tuple[int, int]], rows: int, cols: int) -> tuple[int, int]:
    corners = [cell for cell in all_cells if cell[0] in {0, rows - 1} and cell[1] in {0, cols - 1}]
    edges = [cell for cell in all_cells if _is_border_cell(cell, rows, cols) and cell not in corners]
    interior = [cell for cell in all_cells if not _is_border_cell(cell, rows, cols)]
    groups = [group for group in (corners, edges, interior) if group]
    return random.choice(random.choice(groups))


def _sample_mask_coords(pool: list[tuple[int, int]], size: int) -> list[tuple[int, int]] | None:
    if len(pool) < size:
        return None
    for _ in range(200):
        coords = random.sample(pool, size)
        rows = {r for r, _ in coords}
        cols = {c for _, c in coords}
        if len(rows) >= 2 and len(cols) >= 2:
            return coords
    return None


def _build_target_map(occupied: list[tuple[int, int]], key_cell: tuple[int, int]) -> dict:
    decoys = [cell for cell in occupied if cell != key_cell]
    random.shuffle(decoys)
    target_map = {key_cell: key_cell}
    for idx, cell in enumerate(decoys):
        target_map[cell] = key_cell if idx == len(decoys) - 1 else decoys[idx + 1]
    return target_map


def _layout_geometry(rows: int, cols: int) -> dict:
    row_spans = [(r * (rows + 1), r * (rows + 1) + rows - 1) for r in range(rows)]
    col_spans = [(c * (cols + 1), c * (cols + 1) + cols - 1) for c in range(cols)]
    sep_rows = [r1 + 1 for _r0, r1 in row_spans[:-1]]
    sep_cols = [c1 + 1 for _c0, c1 in col_spans[:-1]]
    height = row_spans[-1][1] + 1
    width = col_spans[-1][1] + 1
    return {
        "row_spans": row_spans,
        "col_spans": col_spans,
        "sep_rows": sep_rows,
        "sep_cols": sep_cols,
        "height": height,
        "width": width,
    }


def _render_case(case: dict) -> np.ndarray:
    geo = case["geometry"]
    rows = int(case["rows"])
    cols = int(case["cols"])
    bg = int(case["bg"])
    sep = int(case["sep"])
    grid = np.full((geo["height"], geo["width"]), bg, dtype=int)
    for rr in geo["sep_rows"]:
        grid[rr, :] = sep
    for cc in geo["sep_cols"]:
        grid[:, cc] = sep

    for cell, pattern in case["patterns"].items():
        r0, r1 = geo["row_spans"][cell[0]]
        c0, c1 = geo["col_spans"][cell[1]]
        patch = np.full((rows, cols), bg, dtype=int)
        patch[pattern["mask"]] = int(pattern["color"])
        grid[r0 : r1 + 1, c0 : c1 + 1] = patch
    return grid


def _expected_output(case: dict) -> np.ndarray:
    rows = int(case["rows"])
    cols = int(case["cols"])
    geo = case["geometry"]
    bg = int(case["bg"])
    sep = int(case["sep"])
    key_cell = case["key_cell"]
    key_pattern = case["patterns"][key_cell]

    out = np.full((geo["height"], geo["width"]), bg, dtype=int)
    for rr in geo["sep_rows"]:
        out[rr, :] = sep
    for cc in geo["sep_cols"]:
        out[:, cc] = sep

    for r in range(rows):
        for c in range(cols):
            if bool(key_pattern["mask"][r, c]):
                r0, r1 = geo["row_spans"][r]
                c0, c1 = geo["col_spans"][c]
                out[r0 : r1 + 1, c0 : c1 + 1] = int(key_pattern["color"])
    return out


def _build_case() -> dict | None:
    rows = random.randint(3, 5)
    cols = random.randint(3, 5)
    all_cells = [(r, c) for r in range(rows) for c in range(cols)]
    key_cell = _choose_key_cell(all_cells, rows, cols)

    max_candidates = min(6, rows * cols - 2)
    if max_candidates < 3:
        return None
    candidate_count = random.randint(3, max_candidates)

    for _ in range(200):
        other_cells = random.sample([cell for cell in all_cells if cell != key_cell], candidate_count - 1)
        occupied = [key_cell] + other_cells
        occupied_rows = {r for r, _ in occupied}
        occupied_cols = {c for _, c in occupied}
        if len(occupied_rows) < 2 or len(occupied_cols) < 2:
            continue

        complement = [cell for cell in all_cells if cell not in occupied]
        if len(complement) < 2:
            continue

        base_size = random.randint(2, min(5, len(complement)))
        base_coords = _sample_mask_coords(complement, base_size)
        if base_coords is None:
            continue

        output_coords = set(base_coords) | {key_cell}
        if len(output_coords) < 3:
            continue
        if all(_is_border_cell(cell, rows, cols) for cell in output_coords) and rows >= 3 and cols >= 3:
            continue
        if not any(_is_border_cell(cell, rows, cols) for cell in output_coords):
            continue

        bg, sep = random.sample(PALETTE, 2)
        remaining = [color for color in PALETTE if color not in {bg, sep}]
        colors = random.sample(remaining, candidate_count)
        target_map = _build_target_map(occupied, key_cell)

        patterns: dict[tuple[int, int], dict] = {}
        for cell, color in zip(occupied, colors):
            mask = np.zeros((rows, cols), dtype=bool)
            for coord in base_coords:
                mask[coord] = True
            mask[target_map[cell]] = True
            patterns[cell] = {"color": int(color), "mask": mask}

        return {
            "rows": rows,
            "cols": cols,
            "bg": int(bg),
            "sep": int(sep),
            "key_cell": key_cell,
            "occupied": occupied,
            "base_coords": base_coords,
            "patterns": patterns,
            "geometry": _layout_geometry(rows, cols),
        }
    return None


def _validate_case(case: dict, grid: np.ndarray) -> None:
    rows = int(case["rows"])
    cols = int(case["cols"])
    expected = _expected_output(case)
    actual = main(grid)
    assert np.array_equal(actual, expected), "solver output must match constructed target"

    parsed = _parse_separator_table(grid)
    assert parsed["cell_rows"] == rows and parsed["cell_cols"] == cols
    candidates = _extract_candidates(parsed)
    assert len(candidates) == len(case["patterns"]) >= 3

    by_cell = {cand["cell"]: cand for cand in candidates}
    assert set(by_cell) == set(case["patterns"])

    key_hits = [cell for cell, cand in by_cell.items() if bool(cand["mask"][cell[0], cell[1]])]
    assert key_hits == [case["key_cell"]], "generated grid must have exactly one self-referential pattern"

    for cell, spec in case["patterns"].items():
        cand = by_cell[cell]
        assert cand["color"] == spec["color"]
        assert np.array_equal(cand["mask"], spec["mask"])

    key_area = int(case["patterns"][case["key_cell"]]["mask"].sum())
    same_area = sum(int(spec["mask"].sum()) == key_area for spec in case["patterns"].values())
    assert same_area >= 2, "key must not be isolated by area"
    assert len(case["base_coords"]) >= 2
    assert np.count_nonzero(expected == case["patterns"][case["key_cell"]]["color"]) > rows * cols


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        case = _build_case()
        if case is None:
            continue
        grid = _render_case(case)
        _validate_case(case, grid)
        return grid
    raise RuntimeError(f"failed to generate a valid puzzle after {MAX_GENERATION_ATTEMPTS} attempts")
