import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
PINK = 6
ORANGE = 7
TEAL = 8
CARDINAL = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 500


def _full_color_indices(grid: np.ndarray, color: int, axis: int) -> list[int]:
    view = grid if axis == 0 else grid.T
    return [idx for idx in range(view.shape[0]) if np.all(view[idx] == color)]


def _cell_spans(size: int, dividers: list[int]) -> list[tuple[int, int]]:
    assert len(dividers) == 2, f"expected 2 divider lines, got {dividers}"
    dividers = sorted(int(x) for x in dividers)
    spans = [
        (0, dividers[0] - 1),
        (dividers[0] + 1, dividers[1] - 1),
        (dividers[1] + 1, size - 1),
    ]
    assert all(lo <= hi for lo, hi in spans), f"invalid spans from dividers {dividers}"
    return spans


def _iter_cells(grid: np.ndarray) -> list[dict]:
    row_divs = _full_color_indices(grid, PINK, axis=0)
    col_divs = _full_color_indices(grid, PINK, axis=1)
    row_spans = _cell_spans(grid.shape[0], row_divs)
    col_spans = _cell_spans(grid.shape[1], col_divs)
    cells = []
    for ri, (r0, r1) in enumerate(row_spans):
        for ci, (c0, c1) in enumerate(col_spans):
            cells.append(
                {
                    "index": 3 * ri + ci,
                    "row_band": ri,
                    "col_band": ci,
                    "r0": r0,
                    "r1": r1,
                    "c0": c0,
                    "c1": c1,
                    "height": r1 - r0 + 1,
                    "width": c1 - c0 + 1,
                    "area": (r1 - r0 + 1) * (c1 - c0 + 1),
                }
            )
    return cells


def _classify_cell(cell: np.ndarray) -> dict:
    h, w = cell.shape
    nonblack = np.argwhere(cell != BLACK)
    if nonblack.size == 0:
        return {"state": "empty"}

    reds = np.argwhere(cell == RED)
    blues = np.argwhere(cell == BLUE)
    assert np.all(np.isin(cell, [BLACK, BLUE, RED])), "cells may only contain black/blue/red before recolor"

    if len(reds) == 1 and len(nonblack) == 1:
        cr, cc = map(int, reds[0])
        assert 0 < cr < h - 1 and 0 < cc < w - 1, "center-only copy must allow a full 3x3 completion"
        return {"state": "center_only", "center": (cr, cc)}

    if len(reds) == 1 and len(blues) == 4 and len(nonblack) == 5:
        cr, cc = map(int, reds[0])
        expected_shell = {(cr + dr, cc + dc) for dr, dc in CARDINAL}
        assert all(0 <= r < h and 0 <= c < w for r, c in expected_shell), "complete motif leaves the cell"
        assert set(map(tuple, blues)) == expected_shell, "complete motif shell must be a cardinal cross"
        return {"state": "complete", "center": (cr, cc), "shell": sorted(expected_shell)}

    if len(reds) == 0 and len(blues) == 4 and len(nonblack) == 4:
        shell = {tuple(map(int, rc)) for rc in blues}
        cr = sum(r for r, _c in shell) / 4.0
        cc = sum(c for _r, c in shell) / 4.0
        assert cr.is_integer() and cc.is_integer(), "shell-only copy must have an integral geometric center"
        center = (int(cr), int(cc))
        expected_shell = {(center[0] + dr, center[1] + dc) for dr, dc in CARDINAL}
        assert shell == expected_shell, "shell-only copy must be a cardinal cross"
        assert 0 < center[0] < h - 1 and 0 < center[1] < w - 1, "shell-only copy must fit a full 3x3 motif"
        assert int(cell[center]) == BLACK, "shell-only copy center must be missing"
        return {"state": "shell_only", "center": center, "shell": sorted(shell)}

    raise AssertionError("each non-empty cell must be empty, center-only, shell-only, or complete")


def _absolute_shell(cell_info: dict, center: tuple[int, int]) -> list[tuple[int, int]]:
    return [(cell_info["r0"] + center[0] + dr, cell_info["c0"] + center[1] + dc) for dr, dc in CARDINAL]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete every partial cross motif inside the magenta partition, then recolor the most crowded and roomiest shells by remaining black area.

    Input structure:
    - A black grid is split by two full magenta rows and two full magenta columns into a 3x3 array of rectangular cells.
    - Exactly one cell contains a complete 3x3 motif with a red center and a blue cardinal shell; 4-6 other cells contain partial copies of that same motif, while the remaining cells are empty.
    - Partial copies are either a lone red center or a blue four-pixel shell with its center missing, and every non-empty cell is large enough to hold the full 3x3 motif.

    Rule:
    - Identify the cell partition from the magenta divider lines and treat each cell independently.
    - Complete each center-only copy by adding blue pixels above, below, left, and right of the red center; complete each shell-only copy by placing a red center at the shell's geometric center.
    - After completion, count the black pixels remaining inside every non-empty cell; recolor the shell of the unique smallest-black-area cell to teal and the shell of the unique largest-black-area cell to orange, leaving all other shells blue and every red center unchanged.

    Key insight:
    - The visible partials are incomplete views of a single fixed 3x3 cross motif, and only after restoring those missing pixels do the cells' remaining black areas become the quantity that decides the final shell colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    out = g.copy()

    cells = _iter_cells(g)
    pattern_infos = []
    template_count = 0
    partial_count = 0

    for cell_info in cells:
        r0, r1, c0, c1 = cell_info["r0"], cell_info["r1"], cell_info["c0"], cell_info["c1"]
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        desc = _classify_cell(patch)
        state = desc["state"]
        cell_info["state"] = state
        if state == "empty":
            continue

        center_local = desc["center"]
        center_abs = (r0 + center_local[0], c0 + center_local[1])
        shell_abs = _absolute_shell(cell_info, center_local)

        if state == "center_only":
            for rr, cc in shell_abs:
                assert out[rr, cc] == BLACK, "completion may only write onto black background"
                out[rr, cc] = BLUE
            partial_count += 1
        elif state == "shell_only":
            assert out[center_abs] == BLACK, "completion may only write a missing center onto black background"
            out[center_abs] = RED
            partial_count += 1
        else:
            template_count += 1

        pattern_infos.append(
            {
                "cell_index": cell_info["index"],
                "area": cell_info["area"],
                "center": center_abs,
                "shell": shell_abs,
            }
        )

    assert template_count == 1, f"expected exactly one complete template cell, got {template_count}"
    assert 4 <= partial_count <= 6, f"expected 4-6 partial copies, got {partial_count}"
    assert 5 <= len(pattern_infos) <= 7, f"expected 5-7 non-empty cells, got {len(pattern_infos)}"

    areas_to_cells: dict[int, list[int]] = {}
    for info in pattern_infos:
        cell = cells[info["cell_index"]]
        patch = out[cell["r0"] : cell["r1"] + 1, cell["c0"] : cell["c1"] + 1]
        black_area = int(np.sum(patch == BLACK))
        info["black_area"] = black_area
        areas_to_cells.setdefault(black_area, []).append(info["cell_index"])

    assert len(areas_to_cells) == len(pattern_infos), "remaining black areas must be unique across non-empty cells"

    min_black = min(areas_to_cells)
    max_black = max(areas_to_cells)
    min_cell = areas_to_cells[min_black][0]
    max_cell = areas_to_cells[max_black][0]
    assert min_cell != max_cell, "need distinct most-crowded and roomiest cells"

    for info in pattern_infos:
        if info["cell_index"] == min_cell:
            new_shell_color = TEAL
        elif info["cell_index"] == max_cell:
            new_shell_color = ORANGE
        else:
            new_shell_color = BLUE
        for rr, cc in info["shell"]:
            assert out[rr, cc] == BLUE, "shell recolor must start from blue shell pixels"
            out[rr, cc] = new_shell_color
        assert out[info["center"]] == RED, "center color stays red"

    return out


def _paint_motif(grid: np.ndarray, cell: dict, center: tuple[int, int], *, state: str) -> None:
    cr = cell["r0"] + center[0]
    cc = cell["c0"] + center[1]
    shell = [(cr + dr, cc + dc) for dr, dc in CARDINAL]
    if state in {"complete", "shell_only"}:
        for rr, cc2 in shell:
            grid[rr, cc2] = BLUE
    if state in {"complete", "center_only"}:
        grid[cr, cc] = RED


def _build_base_grid(heights: Iterable[int], widths: Iterable[int]) -> tuple[np.ndarray, list[dict]]:
    heights = list(int(x) for x in heights)
    widths = list(int(x) for x in widths)
    h = sum(heights) + 2
    w = sum(widths) + 2
    grid = np.full((h, w), BLACK, dtype=int)
    row_divs = [heights[0], heights[0] + heights[1] + 1]
    col_divs = [widths[0], widths[0] + widths[1] + 1]
    grid[row_divs[0], :] = PINK
    grid[row_divs[1], :] = PINK
    grid[:, col_divs[0]] = PINK
    grid[:, col_divs[1]] = PINK
    return grid, _iter_cells(grid)


def _choose_pattern_cells(cells: list[dict], total_patterns: int) -> list[dict] | None:
    by_area: dict[int, list[dict]] = {}
    for cell in cells:
        by_area.setdefault(cell["area"], []).append(cell)
    if len(by_area) < total_patterns:
        return None
    chosen_areas = random.sample(list(by_area), total_patterns)
    selected = [random.choice(by_area[area]) for area in chosen_areas]
    return selected


def _assignment_breaks_visible_shortcut(assignments: list[dict]) -> bool:
    true_black = [item["cell"]["area"] - 5 for item in assignments]
    visible_counts = {"template": 5, "center_only": 1, "shell_only": 4}
    visible_black = [item["cell"]["area"] - visible_counts[item["role"]] for item in assignments]

    true_min_cells = {idx for idx, value in enumerate(true_black) if value == min(true_black)}
    true_max_cells = {idx for idx, value in enumerate(true_black) if value == max(true_black)}
    visible_min_cells = {idx for idx, value in enumerate(visible_black) if value == min(visible_black)}
    visible_max_cells = {idx for idx, value in enumerate(visible_black) if value == max(visible_black)}
    return visible_min_cells != true_min_cells or visible_max_cells != true_max_cells


def _pick_roles(selected_cells: list[dict]) -> list[dict] | None:
    partial_count = len(selected_cells) - 1
    assert 4 <= partial_count <= 6
    order = selected_cells[:]
    for _ in range(200):
        random.shuffle(order)
        template_cell = order[0]
        partial_cells = order[1:]
        shell_count = random.randint(1, partial_count - 1)
        shell_indices = set(random.sample(range(partial_count), shell_count))
        assignments = [{"cell": template_cell, "role": "template"}]
        for idx, cell in enumerate(partial_cells):
            role = "shell_only" if idx in shell_indices else "center_only"
            assignments.append({"cell": cell, "role": role})
        if _assignment_breaks_visible_shortcut(assignments):
            return assignments
    return None


def _place_centers(assignments: list[dict]) -> None:
    for item in assignments:
        cell = item["cell"]
        center_r = random.randint(1, cell["height"] - 2)
        center_c = random.randint(1, cell["width"] - 2)
        item["center"] = (center_r, center_c)


def _expected_output_from_assignments(heights: list[int], widths: list[int], assignments: list[dict]) -> np.ndarray:
    grid, _cells = _build_base_grid(heights, widths)
    for item in assignments:
        _paint_motif(grid, item["cell"], item["center"], state="complete")

    black_areas = [item["cell"]["area"] - 5 for item in assignments]
    min_idx = int(np.argmin(black_areas))
    max_idx = int(np.argmax(black_areas))

    for idx, item in enumerate(assignments):
        cr = item["cell"]["r0"] + item["center"][0]
        cc = item["cell"]["c0"] + item["center"][1]
        color = BLUE
        if idx == min_idx:
            color = TEAL
        elif idx == max_idx:
            color = ORANGE
        for dr, dc in CARDINAL:
            grid[cr + dr, cc + dc] = color
        grid[cr, cc] = RED
    return grid


def _validate_input(grid: np.ndarray) -> None:
    cells = _iter_cells(grid)
    states = []
    partial_types = set()
    for cell in cells:
        patch = grid[cell["r0"] : cell["r1"] + 1, cell["c0"] : cell["c1"] + 1]
        desc = _classify_cell(patch)
        states.append(desc["state"])
        if desc["state"] in {"center_only", "shell_only"}:
            partial_types.add(desc["state"])

    nonempty_cells = [state for state in states if state != "empty"]
    assert states.count("complete") == 1
    assert 4 <= len(nonempty_cells) - 1 <= 6
    assert partial_types == {"center_only", "shell_only"}
    assert len({cell["area"] for cell in cells}) >= 3


def _gen_attempt() -> np.ndarray | None:
    heights = random.sample([3, 4, 5, 6], 3)
    widths = random.sample([3, 4, 5, 6, 7], 3)
    base_grid, cells = _build_base_grid(heights, widths)

    total_patterns = random.randint(5, 7)
    selected_cells = _choose_pattern_cells(cells, total_patterns)
    if selected_cells is None:
        return None

    assignments = _pick_roles(selected_cells)
    if assignments is None:
        return None
    _place_centers(assignments)

    grid = base_grid.copy()
    for item in assignments:
        state = "complete" if item["role"] == "template" else item["role"]
        _paint_motif(grid, item["cell"], item["center"], state=state)

    expected = _expected_output_from_assignments(heights, widths, assignments)

    try:
        _validate_input(grid)
        if not np.array_equal(main(grid), expected):
            return None
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
