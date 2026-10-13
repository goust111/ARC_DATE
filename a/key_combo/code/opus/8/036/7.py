import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
CELL = 5
STRIDE = CELL + 1
MAX_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _find_lattice_lines(grid: np.ndarray) -> tuple[list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    rows = [int(r) for r in np.where(np.count_nonzero(g == GREY, axis=1) >= W - 2)[0]]
    cols = [int(c) for c in np.where(np.count_nonzero(g == GREY, axis=0) >= H - 2)[0]]
    assert len(rows) >= 2 and len(cols) >= 2, "expected regular lattice lines"
    row_steps = {rows[i + 1] - rows[i] for i in range(len(rows) - 1)}
    col_steps = {cols[i + 1] - cols[i] for i in range(len(cols) - 1)}
    assert row_steps == {STRIDE} and col_steps == {STRIDE}, "expected 5x5 lattice cells"
    return rows, cols


def _find_hole_rectangle(grid: np.ndarray, rows: list[int], cols: list[int]) -> tuple[int, int, int, int]:
    holes = [(r, c) for r in rows for c in cols if int(grid[r, c]) == BLACK]
    assert len(holes) == 4, "expected exactly four missing intersections"
    hole_rows = sorted({r for r, _ in holes})
    hole_cols = sorted({c for _, c in holes})
    assert len(hole_rows) == 2 and len(hole_cols) == 2, "holes must be rectangle corners"
    r0, r1 = hole_rows
    c0, c1 = hole_cols
    assert set(holes) == {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    return r0, c0, r1, c1


def _ring_mask() -> np.ndarray:
    mask = np.zeros((3, 3), dtype=bool)
    mask[0, :] = True
    mask[2, :] = True
    mask[:, 0] = True
    mask[:, 2] = True
    return mask


RING_MASK = _ring_mask()


def _detect_object(cell: np.ndarray) -> tuple[int, int] | None:
    assert cell.shape == (CELL, CELL)
    if np.all(cell == BLACK):
        return None
    if np.any(cell[0, :] != BLACK) or np.any(cell[4, :] != BLACK):
        raise AssertionError("cell perimeter must be empty before growth")
    if np.any(cell[:, 0] != BLACK) or np.any(cell[:, 4] != BLACK):
        raise AssertionError("cell perimeter must be empty before growth")

    block = cell[1:4, 1:4]
    border_vals = block[RING_MASK]
    core_val = int(block[1, 1])
    assert core_val != BLACK, "object core must be colored"
    assert len(set(map(int, border_vals))) == 1, "outer ring must be monochrome"
    border_val = int(border_vals[0])
    assert border_val != BLACK and border_val != core_val, "outer and inner colors must differ"
    expected = np.full((3, 3), border_val, dtype=int)
    expected[1, 1] = core_val
    assert np.array_equal(block, expected), "expected centered 3x3 ring with 1x1 core"
    return border_val, core_val


def _cell_bounds(rows: list[int], cols: list[int], br: int, bc: int) -> tuple[int, int, int, int]:
    top = rows[br] + 1
    left = cols[bc] + 1
    return top, left, top + CELL - 1, left + CELL - 1


def _parse_objects(grid: np.ndarray, rows: list[int], cols: list[int]) -> list[dict]:
    objects = []
    for br in range(len(rows) - 1):
        for bc in range(len(cols) - 1):
            top, left, bottom, right = _cell_bounds(rows, cols, br, bc)
            cell = grid[top : bottom + 1, left : right + 1]
            detected = _detect_object(cell)
            if detected is None:
                continue
            border_color, core_color = detected
            objects.append(
                {
                    "cell": (br, bc),
                    "center": (top + 2, left + 2),
                    "ring_box": (top + 1, left + 1, top + 3, left + 3),
                    "border": border_color,
                    "core": core_color,
                }
            )
    return objects


def _apply_selected_transform(
    out: np.ndarray,
    ring_box: tuple[int, int, int, int],
    border_color: int,
    core_color: int,
) -> None:
    r0, c0, r1, c1 = ring_box
    out[r0 : r1 + 1, c0 : c1 + 1] = core_color
    out[r0 + 1, c0 + 1] = border_color

    er0, ec0, er1, ec1 = r0 - 1, c0 - 1, r1 + 1, c1 + 1
    for r in range(er0, er1 + 1):
        for c in range(ec0, ec1 + 1):
            if r in (er0, er1) or c in (ec0, ec1):
                if int(out[r, c]) == BLACK:
                    out[r, c] = border_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap the concentric square layers inside the hole-marked lattice rectangle and grow one original-border-colored layer without crossing grid lines.

    Input structure:
    - A black background is partitioned by a one-pixel grey square lattice into 5x5 cell interiors.
    - Some cells contain a centered concentric square motif: a 3x3 outer ring in one color with a different 1x1 core color at the center.
    - Exactly four lattice intersections are black holes instead of grey, and those four holes are the corners of one axis-aligned selection rectangle.

    Rule:
    - Find the four missing lattice intersections and use them to define the selection rectangle; only motifs whose centers lie strictly inside that rectangle are selected.
    - For each selected motif, record its original outer-ring color and core color, then recolor the 3x3 ring to the core color and the center pixel to the original ring color.
    - After that swap, grow one more layer around the 3x3 motif using the original ring color, but only onto black pixels, so the growth stops at the surrounding grey lattice lines.
    - Leave motifs outside the hole rectangle, all empty cells, all lattice lines, and the four holes unchanged.

    Key insight:
    - The growth color comes from the motif's pre-swap outer ring, so the solver must use the hole rectangle for selection and track the original border color across the intermediate swap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected 2D grid"
    assert _get_bg(g) == BLACK, "expected black background"

    rows, cols = _find_lattice_lines(g)
    hr0, hc0, hr1, hc1 = _find_hole_rectangle(g, rows, cols)
    objects = _parse_objects(g, rows, cols)

    out = g.copy()
    for obj in objects:
        center_r, center_c = obj["center"]
        if hr0 < center_r < hr1 and hc0 < center_c < hc1:
            _apply_selected_transform(out, obj["ring_box"], obj["border"], obj["core"])

    return out


def _build_lattice(num_rows: int, num_cols: int) -> tuple[np.ndarray, list[int], list[int]]:
    H = num_rows * STRIDE + 1
    W = num_cols * STRIDE + 1
    grid = np.full((H, W), BLACK, dtype=int)
    line_rows = [i * STRIDE for i in range(num_rows + 1)]
    line_cols = [i * STRIDE for i in range(num_cols + 1)]
    for r in line_rows:
        grid[r, :] = GREY
    for c in line_cols:
        grid[:, c] = GREY
    return grid, line_rows, line_cols


def _paint_object(grid: np.ndarray, br: int, bc: int, border_color: int, core_color: int) -> None:
    top = br * STRIDE + 1
    left = bc * STRIDE + 1
    grid[top + 1 : top + 4, left + 1 : left + 4] = border_color
    grid[top + 2, left + 2] = core_color


def _direct_expected(inp: np.ndarray, meta: dict) -> np.ndarray:
    out = inp.copy()
    for br, bc in meta["selected_cells"]:
        border_color, core_color = meta["objects"][(br, bc)]
        top = br * STRIDE + 1
        left = bc * STRIDE + 1
        cell = out[top : top + CELL, left : left + CELL]
        cell[1:4, 1:4] = core_color
        cell[2, 2] = border_color
        for i in range(CELL):
            if int(cell[0, i]) == BLACK:
                cell[0, i] = border_color
            if int(cell[4, i]) == BLACK:
                cell[4, i] = border_color
            if int(cell[i, 0]) == BLACK:
                cell[i, 0] = border_color
            if int(cell[i, 4]) == BLACK:
                cell[i, 4] = border_color
    return out


def _choose_selection(
    rng: random.Random,
    num_rows: int,
    num_cols: int,
) -> tuple[tuple[int, int], tuple[int, int], set[tuple[int, int]]]:
    options = []
    for span_r in (1, 2):
        for span_c in (1, 2):
            if span_r * span_c < 2:
                continue
            if span_r > num_rows or span_c > num_cols:
                continue
            for start_r in range(num_rows - span_r + 1):
                for start_c in range(num_cols - span_c + 1):
                    selected = {
                        (br, bc)
                        for br in range(start_r, start_r + span_r)
                        for bc in range(start_c, start_c + span_c)
                    }
                    if len(selected) >= 2 and len(selected) < num_rows * num_cols:
                        options.append(((start_r, start_r + span_r), (start_c, start_c + span_c), selected))
    assert options, "expected at least one valid selection rectangle"
    return rng.choice(options)


def _build_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray, dict]:
    num_rows = rng.choice((3, 4))
    num_cols = rng.choice((3, 4))
    grid, line_rows, line_cols = _build_lattice(num_rows, num_cols)

    (row_pair, col_pair, selected_cells) = _choose_selection(rng, num_rows, num_cols)
    r0_idx, r1_idx = row_pair
    c0_idx, c1_idx = col_pair
    hole_rows = (line_rows[r0_idx], line_rows[r1_idx])
    hole_cols = (line_cols[c0_idx], line_cols[c1_idx])
    for r in hole_rows:
        for c in hole_cols:
            grid[r, c] = BLACK

    all_cells = [(br, bc) for br in range(num_rows) for bc in range(num_cols)]
    outside_cells = [cell for cell in all_cells if cell not in selected_cells]
    inside_count = rng.randint(2, len(selected_cells))
    outside_count = rng.randint(1, min(len(outside_cells), 4))
    occupied = set(rng.sample(sorted(selected_cells), inside_count))
    occupied.update(rng.sample(outside_cells, outside_count))

    objects: dict[tuple[int, int], tuple[int, int]] = {}
    for cell in occupied:
        border_color, core_color = rng.sample(PALETTE, 2)
        objects[cell] = (border_color, core_color)
        _paint_object(grid, cell[0], cell[1], border_color, core_color)

    selected_occupied = tuple(sorted(cell for cell in occupied if cell in selected_cells))
    expected = _direct_expected(grid, {"selected_cells": selected_occupied, "objects": objects})
    meta = {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "hole_rows": hole_rows,
        "hole_cols": hole_cols,
        "selected_cells": selected_occupied,
        "outside_occupied": tuple(sorted(cell for cell in occupied if cell not in selected_cells)),
        "objects": objects,
        "selected_area": tuple(sorted(selected_cells)),
    }
    return grid, expected, meta


def _cell_pattern_after(border_color: int, core_color: int) -> np.ndarray:
    cell = np.full((CELL, CELL), border_color, dtype=int)
    cell[1:4, 1:4] = core_color
    cell[2, 2] = border_color
    return cell


def _cell_pattern_before(border_color: int, core_color: int) -> np.ndarray:
    cell = np.full((CELL, CELL), BLACK, dtype=int)
    cell[1:4, 1:4] = border_color
    cell[2, 2] = core_color
    return cell


def _validate_generated(inp: np.ndarray, expected: np.ndarray, meta: dict) -> None:
    solved = main(inp)
    assert np.array_equal(solved, expected), "solver output mismatches generator oracle"

    rows, cols = _find_lattice_lines(inp)
    hr0, hc0, hr1, hc1 = _find_hole_rectangle(inp, rows, cols)
    assert (hr0, hr1) == meta["hole_rows"]
    assert (hc0, hc1) == meta["hole_cols"]
    assert len(meta["selected_cells"]) >= 2
    assert len(meta["outside_occupied"]) >= 1

    all_colors = {color for pair in meta["objects"].values() for color in pair}
    assert len(all_colors) >= 3, "need at least three visible object colors"
    border_colors = {border for border, _core in meta["objects"].values()}
    assert len(border_colors) >= 2, "need varied original border colors"
    selected_pairs = {meta["objects"][cell] for cell in meta["selected_cells"]}
    assert len(selected_pairs) >= 1

    for r in rows:
        for c in range(inp.shape[1]):
            if c not in cols or (r, c) not in {(hr0, hc0), (hr0, hc1), (hr1, hc0), (hr1, hc1)}:
                assert int(solved[r, c]) == int(inp[r, c]), "lattice rows must stay unchanged"
    for c in cols:
        for r in range(inp.shape[0]):
            if r not in rows or (r, c) not in {(hr0, hc0), (hr0, hc1), (hr1, hc0), (hr1, hc1)}:
                assert int(solved[r, c]) == int(inp[r, c]), "lattice cols must stay unchanged"

    for (br, bc), (border_color, core_color) in meta["objects"].items():
        top = br * STRIDE + 1
        left = bc * STRIDE + 1
        before = inp[top : top + CELL, left : left + CELL]
        after = solved[top : top + CELL, left : left + CELL]
        assert np.array_equal(before, _cell_pattern_before(border_color, core_color))
        if (br, bc) in meta["selected_cells"]:
            assert np.array_equal(after, _cell_pattern_after(border_color, core_color))
            assert np.count_nonzero(after != before) == 25, "selected motif must both swap and grow"
        else:
            assert np.array_equal(after, before), "outside motifs must not change"

    for br, bc in meta["selected_area"]:
        center_r = br * STRIDE + 3
        center_c = bc * STRIDE + 3
        assert hr0 < center_r < hr1 and hc0 < center_c < hc1
    for br, bc in meta["outside_occupied"]:
        center_r = br * STRIDE + 3
        center_c = bc * STRIDE + 3
        assert not (hr0 < center_r < hr1 and hc0 < center_c < hc1)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        try:
            inp, expected, meta = _build_case(rng)
            _validate_generated(inp, expected, meta)
            return inp
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid lattice swap-grow case")
