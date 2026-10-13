from __future__ import annotations

import itertools
import random

import numpy as np


BLACK = 0
YELLOW = 4
GRAY = 5
PINK = 6
KEY_COLORS = (1, 2, 3, 7, 8, 9)

GEN_ATTEMPTS = 500
OUTER_SIZE_2 = (9, 12)
OUTER_SIZE_3 = (9, 11)
INNER_DECOR_MAX = 4
PLACEMENT_PADDING = 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    """arc_utils-style inclusive bbox for an (N,2) array of (r,c) coords."""
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        if r1 != r0:
            cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        if c1 != c0:
            cells.append((r, c1))
    return cells


def _paint_frame(out: np.ndarray, r0: int, c0: int, r1: int, c1: int, *, overwrite_nonblack: bool) -> None:
    for r, c in _frame_cells(r0, c0, r1, c1):
        if overwrite_nonblack or int(out[r, c]) == BLACK:
            out[r, c] = GRAY


def _parse_legend(grid: np.ndarray) -> dict[int, int]:
    legend: dict[int, int] = {}
    for c in range(grid.shape[1]):
        color = int(grid[0, c])
        if color in (BLACK, PINK):
            continue
        depth = 0
        r = 1
        while r < grid.shape[0] and int(grid[r, c]) == PINK:
            depth += 1
            r += 1
        assert 1 <= depth <= 3, "legend depth must be 1..3"
        assert color not in legend, "legend colors must be unique"
        legend[color] = depth
    assert 2 <= len(legend) <= 3, "expected 2 or 3 legend entries"
    return legend


def _extract_groups(grid: np.ndarray, legend: dict[int, int]) -> list[dict[str, int | tuple[int, int, int, int]]]:
    groups: list[dict[str, int | tuple[int, int, int, int]]] = []
    for color, depth in legend.items():
        mask = grid == int(color)
        mask[0, :] = False
        coords = np.argwhere(mask)
        assert coords.shape == (4, 2), "each legend color must appear as exactly four corner markers"
        bbox = bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
        actual = {tuple(map(int, rc)) for rc in coords}
        assert actual == corners, "same-color markers must sit at rectangle corners"
        groups.append({"color": int(color), "depth": int(depth), "bbox": bbox})
    assert 2 <= len(groups) <= 3
    return groups


def _render_frames(
    grid: np.ndarray,
    groups: list[dict[str, int | tuple[int, int, int, int]]],
    *,
    uniform_depth: int | None = None,
    overwrite_nonblack: bool = False,
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for group in groups:
        depth = int(group["depth"] if uniform_depth is None else uniform_depth)
        r0, c0, r1, c1 = group["bbox"]  # type: ignore[index]
        top = r0 + depth
        left = c0 + depth
        bottom = r1 - depth
        right = c1 - depth
        assert top <= bottom and left <= right, "inset frame must stay within marker rectangle"
        _paint_frame(out, top, left, bottom, right, overwrite_nonblack=overwrite_nonblack)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each corner-marker color as an inset depth, then draw the corresponding gray inner frame.

    Input structure:
    - Black is the background.
    - The top row contains 2 or 3 legend entries: a non-black, non-pink color at row 0,
      with 1 to 3 pink cells stacked directly below it in the same column.
    - Below the legend are 2 or 3 non-overlapping four-corner marker rectangles, each using
      one legend color exactly four times.
    - Yellow decoration pixels may appear inside those rectangles and must be preserved.

    Rule:
    - Read each legend column as a mapping from marker color to inset depth.
    - For every same-color four-corner marker group, use its row/column extrema as the outer
      rectangle and look up that color's inset depth.
    - Draw a gray rectangular frame at that inset position.
    - Only paint onto black cells; keep the original markers, legend, and yellow decorations unchanged.

    Key insight:
    - The frame location is not fixed: marker color selects a legend-defined geometric parameter,
      and that decoded depth determines where the inner frame belongs.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    legend = _parse_legend(g)
    groups = _extract_groups(g, legend)
    return _render_frames(g, groups)


def _legend_columns(width: int, n_cols: int) -> tuple[int, ...] | None:
    combos = [
        combo
        for combo in itertools.combinations(range(1, width - 1), n_cols)
        if all(combo[i + 1] - combo[i] >= 2 for i in range(n_cols - 1))
    ]
    if not combos:
        return None
    return random.choice(combos)


def _candidate_positions(
    occupied: np.ndarray, h: int, w: int, top_start: int, border: int = 1
) -> list[tuple[int, int]]:
    H, W = occupied.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(top_start, H - h - border + 1):
        for c0 in range(border, W - w - border + 1):
            rr0 = max(0, r0 - PLACEMENT_PADDING)
            cc0 = max(0, c0 - PLACEMENT_PADDING)
            rr1 = min(H, r0 + h + PLACEMENT_PADDING)
            cc1 = min(W, c0 + w + PLACEMENT_PADDING)
            if np.any(occupied[rr0:rr1, cc0:cc1]):
                continue
            positions.append((r0, c0))
    random.shuffle(positions)
    return positions


def _place_boxes(
    shape: tuple[int, int], boxes: list[tuple[int, int]], top_start: int
) -> list[tuple[int, int]] | None:
    occupied = np.zeros(shape, dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(boxes)
    order = sorted(range(len(boxes)), key=lambda i: boxes[i][0] * boxes[i][1], reverse=True)
    for idx in order:
        h, w = boxes[idx]
        candidates = _candidate_positions(occupied, h, w, top_start)
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        placements[idx] = (r0, c0)
        occupied[r0 : r0 + h, c0 : c0 + w] = True
    return [pos for pos in placements if pos is not None]


def _box_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _choose_decorations(
    bbox: tuple[int, int, int, int], depth: int, *, force_on_path: bool
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    top = r0 + depth
    left = c0 + depth
    bottom = r1 - depth
    right = c1 - depth
    frame = _frame_cells(top, left, bottom, right)
    interior = [
        (r, c)
        for r in range(r0 + 1, r1)
        for c in range(c0 + 1, c1)
    ]
    frame_set = set(frame)
    off_path = [cell for cell in interior if cell not in frame_set]

    chosen: list[tuple[int, int]] = []
    if force_on_path:
        chosen.append(random.choice(frame))

    extra_pool = off_path[:]
    if random.random() < 0.35:
        extra_pool.extend(cell for cell in frame if cell not in chosen)
    random.shuffle(extra_pool)
    extra_count = random.randint(0, min(INNER_DECOR_MAX, len(extra_pool)))
    for cell in extra_pool:
        if len(chosen) >= (1 if force_on_path else 0) + extra_count:
            break
        if cell not in chosen:
            chosen.append(cell)
    return chosen


def _uniform_shortcut_output(grid: np.ndarray, depth: int) -> np.ndarray:
    legend = _parse_legend(grid)
    groups = _extract_groups(grid, legend)
    return _render_frames(grid, groups, uniform_depth=depth)


def _validate_sample(grid: np.ndarray) -> bool:
    try:
        legend = _parse_legend(grid)
        groups = _extract_groups(grid, legend)
        if len({int(group["depth"]) for group in groups}) < 2:
            return False

        boxes = [group["bbox"] for group in groups]
        for i, box_i in enumerate(boxes):
            for box_j in boxes[i + 1 :]:
                if _box_cells(box_i) & _box_cells(box_j):
                    return False

        true_out = main(grid)
        if np.array_equal(true_out, grid):
            return False

        painted_any = False
        has_on_path_yellow = False
        for group in groups:
            depth = int(group["depth"])
            r0, c0, r1, c1 = group["bbox"]
            frame = _frame_cells(r0 + depth, c0 + depth, r1 - depth, c1 - depth)
            new_gray = 0
            for r, c in frame:
                if int(grid[r, c]) == YELLOW:
                    has_on_path_yellow = True
                if int(grid[r, c]) == BLACK and int(true_out[r, c]) == GRAY:
                    new_gray += 1
            if new_gray == 0:
                return False
            painted_any = True
        if not painted_any:
            return False

        for depth in (1, 2, 3):
            if np.array_equal(true_out, _uniform_shortcut_output(grid, depth)):
                return False

        if has_on_path_yellow:
            overwrite_out = _render_frames(grid, groups, overwrite_nonblack=True)
            if np.array_equal(true_out, overwrite_out):
                return False

        return True
    except AssertionError:
        return False


def _sample_group_specs(n_groups: int) -> list[dict[str, int]]:
    colors = random.sample(KEY_COLORS, n_groups)
    depths = random.sample((1, 2, 3), n_groups)
    outer_low, outer_high = OUTER_SIZE_2 if n_groups == 2 else OUTER_SIZE_3
    specs: list[dict[str, int]] = []
    for color, depth in zip(colors, depths):
        h = random.randint(outer_low, outer_high)
        w = random.randint(outer_low, outer_high)
        specs.append({"color": int(color), "depth": int(depth), "h": int(h), "w": int(w)})
    random.shuffle(specs)
    return specs


def _grid_shape(specs: list[dict[str, int]]) -> tuple[int, int]:
    n_groups = len(specs)
    top_start = max(spec["depth"] for spec in specs) + 2
    base_min = 18 if n_groups == 2 else 23
    base_max = 26 if n_groups == 2 else 29
    min_h = max(base_min, top_start + max(spec["h"] for spec in specs) + 2)
    min_w = max(base_min, max(spec["w"] for spec in specs) + 2)
    return random.randint(min_h, base_max), random.randint(min_w, base_max)


def _build_sample() -> np.ndarray | None:
    n_groups = random.randint(2, 3)
    specs = _sample_group_specs(n_groups)
    H, W = _grid_shape(specs)
    legend_cols = _legend_columns(W, n_groups)
    if legend_cols is None:
        return None

    top_start = max(spec["depth"] for spec in specs) + 2
    boxes = [(spec["h"], spec["w"]) for spec in specs]
    placements = _place_boxes((H, W), boxes, top_start)
    if placements is None:
        return None

    grid = np.zeros((H, W), dtype=int)
    for spec, col in zip(specs, legend_cols):
        grid[0, col] = int(spec["color"])
        grid[1 : 1 + int(spec["depth"]), col] = PINK

    forced_path_idx = random.randrange(n_groups)
    for idx, (spec, (r0, c0)) in enumerate(zip(specs, placements)):
        h = int(spec["h"])
        w = int(spec["w"])
        r1 = r0 + h - 1
        c1 = c0 + w - 1
        color = int(spec["color"])
        grid[r0, c0] = color
        grid[r0, c1] = color
        grid[r1, c0] = color
        grid[r1, c1] = color

        bbox = (r0, c0, r1, c1)
        for r, c in _choose_decorations(bbox, int(spec["depth"]), force_on_path=idx == forced_path_idx):
            grid[r, c] = YELLOW

    return grid if _validate_sample(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_sample()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid prefix-coded inset-frame sample")
