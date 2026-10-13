from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BG = 0
SLIDER_COLORS = (1, 2, 3)

GRID_MIN = 13
GRID_MAX = 18
SLIDER_COUNT_CHOICES = (2, 3)
MIN_COL_GAP = 3
GEN_RETRIES = 500


def _ordered_path_rows(height: int, slider: dict, row_count: int) -> list[int]:
    if slider["side"] == "top":
        return list(range(height + 1, row_count - 1))
    return list(range(row_count - height - 2, 0, -1))


def _detect_sliders(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    occupied = np.zeros_like(g, dtype=bool)
    sliders: list[dict] = []
    for c in range(w):
        top_color = int(g[0, c])
        bottom_color = int(g[h - 1, c])
        assert not (top_color and bottom_color), "a slider column cannot touch both top and bottom"

        if top_color:
            height = 2 if h >= 2 and int(g[1, c]) == top_color else 1
            if h > height:
                assert int(g[height, c]) != top_color, "top slider height must be 1 or 2"
            for r in range(height):
                occupied[r, c] = True
            sliders.append({"col": c, "color": top_color, "side": "top", "height": height})

        if bottom_color:
            height = 2 if h >= 2 and int(g[h - 2, c]) == bottom_color else 1
            probe = h - height - 1
            if probe >= 0:
                assert int(g[probe, c]) != bottom_color, "bottom slider height must be 1 or 2"
            for r in range(h - height, h):
                occupied[r, c] = True
            sliders.append({"col": c, "color": bottom_color, "side": "bottom", "height": height})

    sliders.sort(key=lambda item: item["col"])
    return sliders, occupied


def _detect_controls(grid: np.ndarray, occupied: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    controls: list[dict] = []
    for r, c in np.argwhere((g != BG) & (~occupied)):
        rr = int(r)
        cc = int(c)
        color = int(g[rr, cc])
        assert 0 < rr < h - 1, "controls must be strictly inside the grid"
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < h and 0 <= nc < w and not occupied[nr, nc]:
                assert int(g[nr, nc]) == BG, "controls must be isolated single pixels"
        controls.append({"row": rr, "col": cc, "color": color})
    return controls


def _controls_on_path(slider: dict, controls: list[dict]) -> list[dict]:
    path = [ctrl for ctrl in controls if ctrl["col"] == slider["col"]]
    reverse = slider["side"] == "bottom"
    return sorted(path, key=lambda item: item["row"], reverse=reverse)


def _simulate(
    grid: np.ndarray,
    *,
    erase_same: bool = True,
    emit_rays: bool = True,
    order: str = "ltr",
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    sliders, occupied = _detect_sliders(g)
    controls = _detect_controls(g, occupied)
    controls_by_col = {slider["col"]: _controls_on_path(slider, controls) for slider in sliders}

    out = g.copy()
    process = sliders if order == "ltr" else list(reversed(sliders))
    for slider in process:
        col = int(slider["col"])
        color = int(slider["color"])
        path_controls = controls_by_col[col]

        same_hits: list[dict] = []
        diff_hit: dict | None = None
        for ctrl in path_controls:
            if int(ctrl["color"]) == color:
                same_hits.append(ctrl)
            else:
                diff_hit = ctrl
                break

        if slider["side"] == "top":
            stop = int(diff_hit["row"]) if diff_hit is not None else h
            out[:stop, col] = color
        else:
            start = int(diff_hit["row"]) + 1 if diff_hit is not None else 0
            out[start:, col] = color

        if erase_same:
            for ctrl in same_hits:
                out[int(ctrl["row"]), col] = BG

        if diff_hit is not None and emit_rays:
            ray_row = int(diff_hit["row"])
            ray_color = int(diff_hit["color"])
            left_dist = col
            right_dist = w - 1 - col
            if left_dist <= right_dist:
                out[ray_row, : col + 1] = ray_color
            else:
                out[ray_row, col:] = ray_color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Slide each edge-attached color slider along its column, erase same-color controls, and deflect on different-color controls into a horizontal ray.

    Input structure:
    - The grid has a black background, plus 2-3 one-column colored sliders touching
      exactly the top or bottom edge.
    - Each slider is 1-2 cells tall and uses one of blue/red/green; sliders use
      distinct colors.
    - Interior isolated control pixels share those same visible colors and each lies
      on the column-path of exactly one slider.

    Rule:
    - Process sliders from left to right by column.
    - A top slider paints downward; a bottom slider paints upward. Its traveled
      vertical stripe keeps the slider color while the original edge position stays
      colored.
    - When the moving slider reaches a same-color control, that control pixel is
      erased to black and the slide continues.
    - When it reaches a different-color control, the slide stops just before that
      control. The control remains, and from that control cell a horizontal ray in
      the control color is painted toward the nearer side edge (ties go left).
    - Later sliders overwrite earlier painted cells wherever paths cross.

    Key insight:
    - Color matching at the collision point decides between two incompatible
      outcomes: same-color controls become erased holes in the vertical stripe,
      while different-color controls trigger a stop plus a sideways ray.
    """
    return _simulate(grid, erase_same=True, emit_rays=True, order="ltr")


def _candidate_columns(width: int, count: int, direction: str) -> tuple[list[int], int] | None:
    cols = list(range(2, width - 2))
    all_sets = [
        list(choice)
        for choice in combinations(cols, count)
        if all(choice[i + 1] - choice[i] >= MIN_COL_GAP for i in range(count - 1))
    ]
    random.shuffle(all_sets)
    for chosen in all_sets:
        indices: list[int] = []
        if direction == "left":
            indices = [
                idx
                for idx, col in enumerate(chosen)
                if idx > 0 and col <= width - 1 - col
            ]
        else:
            indices = [
                idx
                for idx, col in enumerate(chosen)
                if idx < count - 1 and col > width - 1 - col
            ]
        if indices:
            return chosen, random.choice(indices)
    return None


def _same_only_options(slider: dict, row_count: int, same_count: int) -> list[tuple[int, ...]]:
    ordered = _ordered_path_rows(slider["height"], slider, row_count)
    options: list[tuple[int, ...]] = []
    for idxs in combinations(range(len(ordered)), same_count):
        if all(idxs[i + 1] - idxs[i] >= 2 for i in range(len(idxs) - 1)):
            options.append(tuple(ordered[i] for i in idxs))
    random.shuffle(options)
    return options


def _mixed_options(slider: dict, row_count: int) -> list[tuple[int, int]]:
    ordered = _ordered_path_rows(slider["height"], slider, row_count)
    options: list[tuple[int, int]] = []
    for same_idx in range(len(ordered)):
        for diff_idx in range(same_idx + 2, len(ordered)):
            options.append((ordered[same_idx], ordered[diff_idx]))
    random.shuffle(options)
    return options


def _assign_control_rows(
    sliders: list[dict],
    role_by_index: dict[int, tuple[str, int]],
    row_count: int,
) -> dict[int, tuple[int, ...]] | None:
    option_map: dict[int, list[tuple[int, ...]]] = {}
    for idx, slider in enumerate(sliders):
        role, same_count = role_by_index[idx]
        if role == "mixed":
            option_map[idx] = _mixed_options(slider, row_count)
        else:
            option_map[idx] = _same_only_options(slider, row_count, same_count)
        if not option_map[idx]:
            return None

    order = sorted(option_map, key=lambda key: len(option_map[key]))
    used_rows: set[int] = set()
    assignment: dict[int, tuple[int, ...]] = {}

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        choices = option_map[idx][:]
        random.shuffle(choices)
        for rows in choices:
            if any(row in used_rows for row in rows):
                continue
            assignment[idx] = rows
            used_rows.update(rows)
            if backtrack(pos + 1):
                return True
            for row in rows:
                used_rows.remove(row)
            assignment.pop(idx, None)
        return False

    return assignment if backtrack(0) else None


def _make_sliders(width: int, height: int, count: int) -> tuple[list[dict], str, int] | None:
    direction = random.choice(("left", "right"))
    picked = _candidate_columns(width, count, direction)
    if picked is None:
        return None
    columns, diff_idx = picked
    colors = list(SLIDER_COLORS)
    random.shuffle(colors)
    sliders: list[dict] = []
    for idx, col in enumerate(columns):
        sliders.append(
            {
                "col": col,
                "color": colors[idx],
                "side": random.choice(("top", "bottom")),
                "height": random.choice((1, 2)),
            }
        )
    return sliders, direction, diff_idx


def _build_controls(
    sliders: list[dict],
    direction: str,
    diff_idx: int,
    row_count: int,
) -> list[dict] | None:
    role_by_index: dict[int, tuple[str, int]] = {}
    if len(sliders) == 2:
        same_idx = diff_idx - 1 if direction == "left" else diff_idx + 1
        role_by_index[same_idx] = ("same", random.choice((1, 2)))
        role_by_index[diff_idx] = ("mixed", 1)
    else:
        crossed_idx = diff_idx - 1 if direction == "left" else diff_idx + 1
        role_by_index[crossed_idx] = ("same", random.choice((1, 2)))
        role_by_index[diff_idx] = ("mixed", 1)
        for idx in range(len(sliders)):
            if idx not in role_by_index:
                role_by_index[idx] = ("same", random.choice((1, 2)))

    assignments = _assign_control_rows(sliders, role_by_index, row_count)
    if assignments is None:
        return None

    controls: list[dict] = []
    for idx, slider in enumerate(sliders):
        role, _same_count = role_by_index[idx]
        rows = assignments[idx]
        if role == "mixed":
            same_row, diff_row = rows
            diff_choices = [color for color in SLIDER_COLORS if color != slider["color"]]
            controls.append({"row": same_row, "col": slider["col"], "color": slider["color"]})
            controls.append(
                {
                    "row": diff_row,
                    "col": slider["col"],
                    "color": random.choice(diff_choices),
                }
            )
        else:
            for row in rows:
                controls.append({"row": row, "col": slider["col"], "color": slider["color"]})
    random.shuffle(controls)
    return controls


def _render_input(shape: tuple[int, int], sliders: list[dict], controls: list[dict]) -> np.ndarray:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    for slider in sliders:
        col = int(slider["col"])
        color = int(slider["color"])
        height = int(slider["height"])
        if slider["side"] == "top":
            grid[:height, col] = color
        else:
            grid[h - height :, col] = color
    for ctrl in controls:
        grid[int(ctrl["row"]), int(ctrl["col"])] = int(ctrl["color"])
    return grid


def _path_summary(grid: np.ndarray) -> tuple[list[dict], list[dict], dict[int, list[dict]]]:
    sliders, occupied = _detect_sliders(grid)
    controls = _detect_controls(grid, occupied)
    by_col = {slider["col"]: _controls_on_path(slider, controls) for slider in sliders}
    return sliders, controls, by_col


def _validate_input(grid: np.ndarray) -> bool:
    try:
        sliders, controls, by_col = _path_summary(grid)
    except AssertionError:
        return False

    h, w = grid.shape
    if not (2 <= len(sliders) <= 3):
        return False
    if not (3 <= len(controls) <= 5):
        return False
    if max(h, w) > 30:
        return False

    slider_colors = [int(slider["color"]) for slider in sliders]
    if len(set(slider_colors)) != len(sliders):
        return False
    if any(color not in SLIDER_COLORS for color in slider_colors):
        return False
    if any(int(ctrl["color"]) not in SLIDER_COLORS for ctrl in controls):
        return False
    if len({int(ctrl["row"]) for ctrl in controls}) != len(controls):
        return False

    saw_same = False
    saw_diff = False
    saw_mixed = False
    diff_overlap = False

    for slider in sliders:
        col = int(slider["col"])
        path = by_col[col]
        if not path:
            return False

        diff_hits = [ctrl for ctrl in path if int(ctrl["color"]) != int(slider["color"])]
        if len(diff_hits) > 1:
            return False

        first_diff = diff_hits[0] if diff_hits else None
        same_before = []
        for ctrl in path:
            if first_diff is not None and ctrl["row"] == first_diff["row"]:
                break
            if int(ctrl["color"]) == int(slider["color"]):
                same_before.append(ctrl)

        if same_before:
            saw_same = True
        if first_diff is not None:
            saw_diff = True
            if same_before:
                saw_mixed = True
            ray_row = int(first_diff["row"])
            left_dist = col
            right_dist = w - 1 - col
            if left_dist <= right_dist:
                ray_cols = range(0, col)
            else:
                ray_cols = range(col + 1, w)
            if any(int(ctrl["row"]) == ray_row and int(ctrl["col"]) in ray_cols for ctrl in controls):
                return False
            if any(other["col"] in ray_cols for other in sliders):
                diff_overlap = True

    if not (saw_same and saw_diff and saw_mixed and diff_overlap):
        return False

    try:
        out = main(grid)
        no_erase = _simulate(grid, erase_same=False, emit_rays=True, order="ltr")
        no_ray = _simulate(grid, erase_same=True, emit_rays=False, order="ltr")
        rtl = _simulate(grid, erase_same=True, emit_rays=True, order="rtl")
    except AssertionError:
        return False

    if np.array_equal(out, grid):
        return False
    if np.array_equal(out, no_erase):
        return False
    if np.array_equal(out, no_ray):
        return False
    if np.array_equal(out, rtl):
        return False

    for ctrl in controls:
        row = int(ctrl["row"])
        col = int(ctrl["col"])
        path = by_col[col]
        slider = next(item for item in sliders if int(item["col"]) == col)
        diff_hit = next((item for item in path if int(item["color"]) != int(slider["color"])), None)
        if diff_hit is not None and row == int(diff_hit["row"]):
            if int(out[row, col]) != int(ctrl["color"]):
                return False
        elif int(ctrl["color"]) == int(slider["color"]):
            if int(out[row, col]) != BG:
                return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    slider_count = random.choice(SLIDER_COUNT_CHOICES)

    made = _make_sliders(w, h, slider_count)
    if made is None:
        return None
    sliders, direction, diff_idx = made

    controls = _build_controls(sliders, direction, diff_idx, h)
    if controls is None:
        return None

    grid = _render_input((h, w), sliders, controls)
    return grid if _validate_input(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sliding-toggle ARC input")
