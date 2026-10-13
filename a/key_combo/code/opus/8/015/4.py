from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
GREY = 5
CONTROL_COLORS = (1, 2, 3, 4)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors4(shape: tuple[int, int], r: int, c: int):
    h, w = shape
    for dr, dc in DIR4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _label_mask(mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=int)
    components: list[list[tuple[int, int]]] = []
    next_id = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != -1:
                continue
            queue = deque([(r, c)])
            labels[r, c] = next_id
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(mask.shape, rr, cc):
                    if mask[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = next_id
                        queue.append((nr, nc))
            components.append(cells)
            next_id += 1
    return labels, components


def _label_monochrome_components(grid: np.ndarray) -> tuple[np.ndarray, list[int], list[list[tuple[int, int]]]]:
    h, w = grid.shape
    labels = np.full((h, w), -1, dtype=int)
    comp_colors: list[int] = []
    comp_cells: list[list[tuple[int, int]]] = []
    next_id = 0
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BLACK or labels[r, c] != -1:
                continue
            queue = deque([(r, c)])
            labels[r, c] = next_id
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(grid.shape, rr, cc):
                    if labels[nr, nc] == -1 and int(grid[nr, nc]) == color:
                        labels[nr, nc] = next_id
                        queue.append((nr, nc))
            comp_colors.append(color)
            comp_cells.append(cells)
            next_id += 1
    return labels, comp_colors, comp_cells


def _recolor_by_row(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    h, w = out.shape
    marker_col = w - 1
    for r in range(h):
        row_mask = out[r, :marker_col] == GREY
        if row_mask.any():
            out[r, :marker_col][row_mask] = int(out[r, marker_col])
    return out


def _fill_component_holes(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    comp_labels, comp_colors, _ = _label_monochrome_components(out)
    black_labels, black_components = _label_mask(out == BLACK)
    del black_labels

    for cells in black_components:
        touches_border = any(
            r == 0 or c == 0 or r == out.shape[0] - 1 or c == out.shape[1] - 1
            for r, c in cells
        )
        if touches_border:
            continue

        neighbor_ids: set[int] = set()
        for r, c in cells:
            for nr, nc in _neighbors4(out.shape, r, c):
                cid = int(comp_labels[nr, nc])
                if cid >= 0:
                    neighbor_ids.add(cid)

        if len(neighbor_ids) != 1:
            continue

        comp_id = next(iter(neighbor_ids))
        fill_color = int(comp_colors[comp_id])
        for r, c in cells:
            out[r, c] = fill_color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor grey pixels row-by-row from the control column, then fill black holes enclosed by fused same-color components.

    Input structure:
    - Black background with grey placeholder shapes drawn in the main area.
    - The rightmost column is a control column; every row containing grey has one non-black, non-grey control pixel there.
    - Different grey shapes may touch the control side indirectly by extending to the penultimate column, while other shapes remain separate.

    Rule:
    - Replace every grey pixel in a row by that row's control color, leaving the control-column pixels unchanged.
    - On the recolored grid, detect 4-connected monochrome foreground components.
    - For each 4-connected black region that does not touch the grid border, fill it only when all of its non-black 4-neighbors belong to one single foreground component.
    - Leave border-reachable black background unchanged.

    Key insight:
    - The control colors change topology as well as color: disconnected grey pieces can fuse through same-colored row assignments, and only those fused monochrome components are allowed to define holes.
    """
    recolored = _recolor_by_row(np.asarray(grid, dtype=int))
    return _fill_component_holes(recolored)


def _pair_cells(top: int, height: int, left: int, right: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    bottom = top + height - 1

    # Two disconnected grey shapes:
    # - upper-right bracket misses the top-left corner and stops one row before the bottom
    # - lower-left bracket misses the top row but reaches the bottom-right corner
    # After recoloring, the control-column bridge fuses them into a single enclosure.
    top_right = {(top, c) for c in range(left + 1, right + 1)}
    top_right |= {(r, right) for r in range(top + 1, bottom - 1)}

    left_bottom = {(r, left) for r in range(top + 1, bottom + 1)}
    left_bottom |= {(bottom, c) for c in range(left, right + 1)}

    interior = {
        (r, c)
        for r in range(top + 1, bottom)
        for c in range(left + 1, right)
    }
    interior.add((bottom - 1, right))
    return top_right, left_bottom, interior


def _paint_cells(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _build_expected_output(grid: np.ndarray, meta: dict) -> np.ndarray:
    expected = _recolor_by_row(grid)
    for pair in meta["hole_pairs"]:
        for r, c in pair["interior"]:
            expected[r, c] = int(pair["color"])
    return expected


def _validate_generated(grid: np.ndarray, meta: dict) -> bool:
    out = main(grid)
    expected = _build_expected_output(grid, meta)
    if not np.array_equal(out, expected):
        return False

    grey_rows = [r for r in range(grid.shape[0]) if np.any(grid[r, :-1] == GREY)]
    used_controls = {int(grid[r, -1]) for r in grey_rows}
    if len(used_controls) < 2:
        return False

    grey_labels, _ = _label_mask(grid[:, :-1] == GREY)
    recolored = _recolor_by_row(grid)
    comp_labels, comp_colors, _ = _label_monochrome_components(recolored)

    filled_cells = 0
    for pair in meta["hole_pairs"]:
        a_sample = next(iter(pair["a"]))
        b_sample = next(iter(pair["b"]))
        a_id = grey_labels[a_sample[0], a_sample[1]]
        b_id = grey_labels[b_sample[0], b_sample[1]]
        if a_id == -1 or b_id == -1 or a_id == b_id:
            return False

        a_comp = int(comp_labels[a_sample[0], a_sample[1]])
        b_comp = int(comp_labels[b_sample[0], b_sample[1]])
        if a_comp == -1 or a_comp != b_comp:
            return False
        if int(comp_colors[a_comp]) != int(pair["color"]):
            return False

        for r, c in pair["interior"]:
            if int(recolored[r, c]) != BLACK or int(out[r, c]) != int(pair["color"]):
                return False
            filled_cells += 1

    if filled_cells == 0:
        return False

    for pair in meta["split_pairs"]:
        for r, c in pair["interior"]:
            if int(out[r, c]) != BLACK:
                return False

    return True


def _sample_layout(rng: np.random.Generator) -> tuple[int, list[dict]]:
    hole_h = int(rng.integers(4, 6))
    split_h = int(rng.integers(4, 6))
    interval_types = ["hole", "split"]
    rng.shuffle(interval_types)
    interval_heights = {"hole": hole_h, "split": split_h}

    gap_top = int(rng.integers(0, 2))
    gap_mid = int(rng.integers(1, 3))
    gap_bottom = int(rng.integers(0, 3))
    base_h = gap_top + gap_mid + gap_bottom + hole_h + split_h
    if base_h > 15:
        return 0, []

    total_h = int(rng.integers(max(10, base_h), 16))
    slack = total_h - base_h
    add_top = int(rng.integers(0, slack + 1))
    slack -= add_top
    add_mid = int(rng.integers(0, slack + 1))
    add_bottom = slack - add_mid

    top0 = gap_top + add_top
    first_type = interval_types[0]
    second_type = interval_types[1]
    second_top = top0 + interval_heights[first_type] + gap_mid + add_mid

    intervals = [
        {"type": first_type, "top": top0, "height": interval_heights[first_type]},
        {"type": second_type, "top": second_top, "height": interval_heights[second_type]},
    ]
    _ = gap_bottom + add_bottom
    return total_h, intervals


def generate_input() -> np.ndarray:
    for _ in range(400):
        rng = np.random.default_rng()
        height, intervals = _sample_layout(rng)
        if not intervals:
            continue

        width = int(rng.integers(10, 15))
        right = width - 2
        max_pair_width = min(6, right)
        if max_pair_width < 4:
            continue

        colors = list(CONTROL_COLORS)
        rng.shuffle(colors)
        hole_color = int(colors[0])
        split_bar_color = int(colors[1])
        split_side_color = int(colors[2])

        grid = np.zeros((height, width), dtype=int)
        meta = {"hole_pairs": [], "split_pairs": []}

        for interval in intervals:
            pair_width = int(rng.integers(4, max_pair_width + 1))
            left = right - pair_width + 1
            a_cells, b_cells, interior = _pair_cells(interval["top"], interval["height"], left, right)
            _paint_cells(grid, a_cells, GREY)
            _paint_cells(grid, b_cells, GREY)

            if interval["type"] == "hole":
                color = hole_color
                for r in range(interval["top"], interval["top"] + interval["height"]):
                    grid[r, -1] = color
                meta["hole_pairs"].append(
                    {
                        "a": a_cells,
                        "b": b_cells,
                        "interior": interior,
                        "color": color,
                    }
                )
            else:
                top = interval["top"]
                bottom = top + interval["height"] - 1
                grid[top, -1] = split_bar_color
                grid[bottom, -1] = split_bar_color
                for r in range(top + 1, bottom):
                    grid[r, -1] = split_side_color
                meta["split_pairs"].append(
                    {
                        "a": a_cells,
                        "b": b_cells,
                        "interior": interior,
                        "bar_color": split_bar_color,
                        "side_color": split_side_color,
                    }
                )

        if _validate_generated(grid, meta):
            return grid

    raise RuntimeError("failed to generate a valid row-control hole-fusion puzzle")
