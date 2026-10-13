from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
ORANGE = 7

LABEL_COLORS = (RED, BLUE)
SWATCH_POOL = (3, 4, 6, 8, 9)
SIZE_CHOICES = ((4, 5), (4, 6), (5, 4), (5, 6), (6, 4), (6, 5))
MAX_GENERATION_ATTEMPTS = 400

_CANDIDATE_CACHE: dict[tuple[int, int], dict[str, list[dict[str, object]]]] = {}


def _panel_ranges(separator_lines: list[int], limit: int) -> list[tuple[int, int]]:
    assert len(separator_lines) >= 2
    assert separator_lines[0] == 0 and separator_lines[-1] == limit - 1
    ranges: list[tuple[int, int]] = []
    for a, b in zip(separator_lines, separator_lines[1:]):
        assert b > a + 1, "adjacent separator lines would create an empty panel"
        ranges.append((a + 1, b))
    return ranges


def _uniform_color(panel: np.ndarray) -> int | None:
    values = np.unique(panel)
    if values.size != 1:
        return None
    return int(values[0])


def _trace_bounce(
    height: int, width: int, start: tuple[int, int]
) -> dict[str, object]:
    r, c = map(int, start)
    dr, dc = -1, 1
    visited = {(r, c)}
    ordered = [(r, c)]
    hit_h = False
    hit_v = False

    for _ in range(1, 8 * height * width + 1):
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= height:
            dr = -dr
            nr = r + dr
            hit_h = True
        if nc < 0 or nc >= width:
            dc = -dc
            nc = c + dc
            hit_v = True
        r, c = nr, nc
        ordered.append((r, c))
        visited.add((r, c))
        if (r, c) == start:
            term = "return"
            break
        if r in (0, height - 1) and c in (0, width - 1):
            term = "corner"
            break
    else:
        raise AssertionError("bounce simulation did not terminate")

    mask = np.zeros((height, width), dtype=bool)
    for rr, cc in visited:
        mask[rr, cc] = True
    return {
        "start": (int(start[0]), int(start[1])),
        "mask": mask,
        "term": term,
        "cells": tuple(ordered),
        "count": int(mask.sum()),
        "hit_h": hit_h,
        "hit_v": hit_v,
    }


def _candidate_paths(height: int, width: int) -> dict[str, list[dict[str, object]]]:
    key = (height, width)
    cached = _CANDIDATE_CACHE.get(key)
    if cached is not None:
        return cached

    out: dict[str, list[dict[str, object]]] = {"corner": [], "return": []}
    corners = {
        (0, 0),
        (0, width - 1),
        (height - 1, 0),
        (height - 1, width - 1),
    }
    for r in range(height):
        for c in range(width):
            if (r, c) in corners:
                continue
            info = _trace_bounce(height, width, (r, c))
            if not (bool(info["hit_h"]) and bool(info["hit_v"])):
                continue
            count = int(info["count"])
            if info["term"] == "corner" and count >= 5:
                out["corner"].append(info)
            if info["term"] == "return" and count >= max(6, min(height, width) + 1):
                out["return"].append(info)

    assert out["corner"] and out["return"], "generator requires both termination branches"
    _CANDIDATE_CACHE[key] = out
    return out


def _parse_input(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == GREY)]
    sep_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == GREY)]
    row_ranges = _panel_ranges(sep_rows, g.shape[0])
    col_ranges = _panel_ranges(sep_cols, g.shape[1])
    assert len(row_ranges) == 2 and len(col_ranges) == 3

    panel_heights = {r1 - r0 for r0, r1 in row_ranges}
    panel_widths = {c1 - c0 for c0, c1 in col_ranges}
    assert len(panel_heights) == 1 and len(panel_widths) == 1
    panel_h = panel_heights.pop()
    panel_w = panel_widths.pop()
    assert panel_h >= 4 and panel_w >= 4

    labels: list[int] = []
    columns: list[dict[str, object]] = []

    for ri, (r0, r1) in enumerate(row_ranges):
        label_panel = g[r0:r1, col_ranges[0][0] : col_ranges[0][1]]
        label_color = _uniform_color(label_panel)
        assert label_color is not None
        assert label_color not in {BLACK, GREY, ORANGE}
        labels.append(label_color)

    assert len(set(labels)) == 2

    for ci, (c0, c1) in enumerate(col_ranges[1:], start=1):
        swatch_row = None
        swatch_color = None
        start_row = None
        start_pos = None
        for ri, (r0, r1) in enumerate(row_ranges):
            panel = g[r0:r1, c0:c1]
            pure = _uniform_color(panel)
            if pure is not None:
                assert pure not in {BLACK, GREY, ORANGE}
                assert swatch_row is None, "each active column must have exactly one swatch"
                swatch_row = ri
                swatch_color = pure
                continue

            orange_cells = np.argwhere(panel == ORANGE)
            black_mask = (panel == BLACK) | (panel == ORANGE)
            assert orange_cells.shape[0] == 1 and np.all(black_mask)
            rr, cc = map(int, orange_cells[0])
            assert (rr, cc) not in {
                (0, 0),
                (0, panel_w - 1),
                (panel_h - 1, 0),
                (panel_h - 1, panel_w - 1),
            }
            assert start_row is None, "each active column must have exactly one start panel"
            start_row = ri
            start_pos = (rr, cc)

        assert swatch_row is not None and start_row is not None and start_pos is not None
        assert swatch_row != start_row
        assert swatch_color is not None
        columns.append(
            {
                "input_column": ci,
                "swatch_row": swatch_row,
                "swatch_color": int(swatch_color),
                "start_row": start_row,
                "start_pos": start_pos,
            }
        )

    return {
        "panel_h": int(panel_h),
        "panel_w": int(panel_w),
        "labels": labels,
        "columns": columns,
    }


def _render_output(parsed: dict[str, object]) -> np.ndarray:
    panel_h = int(parsed["panel_h"])
    panel_w = int(parsed["panel_w"])
    labels = list(parsed["labels"])
    columns = list(parsed["columns"])

    out_h = panel_h + 2
    out_w = len(columns) * (panel_w + 2) + (len(columns) - 1)
    out = np.full((out_h, out_w), BLACK, dtype=int)

    for index, col in enumerate(columns):
        start = tuple(col["start_pos"])
        path = _trace_bounce(panel_h, panel_w, start)
        border_color = int(labels[int(col["swatch_row"])])
        fill_color = int(col["swatch_color"])

        panel = np.full((out_h, panel_w + 2), BLACK, dtype=int)
        panel[0, :] = border_color
        panel[-1, :] = border_color
        panel[:, 0] = border_color
        panel[:, -1] = border_color
        panel[1:-1, 1:-1][path["mask"]] = fill_color

        c0 = index * (panel_w + 3)
        out[:, c0 : c0 + panel_w + 2] = panel

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build two colorized output panels from bouncing diagonal masks encoded in a grey-separated panel table.

    Input structure:
    - Grey separator rows and columns split the grid into a 2x3 table of interior panels.
    - The leftmost panel column contains one uniform label color per row.
    - In each remaining panel column, one panel is a solid swatch color and the other is a black field with one orange start pixel.

    Rule:
    - For each non-label panel column, pair the solid swatch with the orange start pixel in the other row.
    - From the start pixel, trace a diagonal ray that begins up-right and reflects off the panel's top, bottom, left, and right interior walls until it returns to the start cell or lands on a corner cell; every visited cell belongs to the mask.
    - Create one output panel per column: wrap the mask in a 1-cell border whose color is the label color of the swatch row, fill mask cells with the swatch color, leave all other cells black, and concatenate the panels left-to-right with one black separator column.

    Key insight:
    The swatch panel chooses the fill color, the swatch row chooses the border color, and the orange pixel's reflected diagonal trajectory generates the mask itself.
    """
    parsed = _parse_input(np.asarray(grid, dtype=int))
    return _render_output(parsed)


def _build_input_from_spec(spec: dict[str, object]) -> np.ndarray:
    panel_h = int(spec["panel_h"])
    panel_w = int(spec["panel_w"])
    grid = np.full((2 * panel_h + 3, 3 * panel_w + 4), GREY, dtype=int)
    row_ranges = [(1, 1 + panel_h), (2 + panel_h, 2 + 2 * panel_h)]
    col_ranges = [
        (1, 1 + panel_w),
        (2 + panel_w, 2 + 2 * panel_w),
        (3 + 2 * panel_w, 3 + 3 * panel_w),
    ]

    for ri, color in enumerate(LABEL_COLORS):
        r0, r1 = row_ranges[ri]
        c0, c1 = col_ranges[0]
        grid[r0:r1, c0:c1] = color

    for col_index, column in enumerate(spec["columns"], start=1):
        swatch_row = int(column["swatch_row"])
        start_row = 1 - swatch_row
        swatch_color = int(column["swatch_color"])
        start_pos = tuple(column["start_pos"])

        sw_r0, sw_r1 = row_ranges[swatch_row]
        c0, c1 = col_ranges[col_index]
        grid[sw_r0:sw_r1, c0:c1] = swatch_color

        st_r0, st_r1 = row_ranges[start_row]
        grid[st_r0:st_r1, c0:c1] = BLACK
        grid[st_r0 + start_pos[0], c0 + start_pos[1]] = ORANGE

    return grid


def _expected_output(spec: dict[str, object]) -> np.ndarray:
    parsed = {
        "panel_h": int(spec["panel_h"]),
        "panel_w": int(spec["panel_w"]),
        "labels": list(LABEL_COLORS),
        "columns": [],
    }
    for column in spec["columns"]:
        parsed["columns"].append(
            {
                "swatch_row": int(column["swatch_row"]),
                "swatch_color": int(column["swatch_color"]),
                "start_pos": tuple(column["start_pos"]),
            }
        )
    return _render_output(parsed)


def _semantic_checks(grid: np.ndarray, spec: dict[str, object]) -> bool:
    parsed = _parse_input(grid)
    if (int(parsed["panel_h"]), int(parsed["panel_w"])) != (
        int(spec["panel_h"]),
        int(spec["panel_w"]),
    ):
        return False
    if list(parsed["labels"]) != list(LABEL_COLORS):
        return False

    recovered = list(parsed["columns"])
    expected_columns = list(spec["columns"])
    if len(recovered) != 2 or len(expected_columns) != 2:
        return False

    terms = []
    masks = []
    for got, want in zip(recovered, expected_columns):
        if int(got["swatch_row"]) != int(want["swatch_row"]):
            return False
        if int(got["swatch_color"]) != int(want["swatch_color"]):
            return False
        if tuple(got["start_pos"]) != tuple(want["start_pos"]):
            return False

        traced = _trace_bounce(int(parsed["panel_h"]), int(parsed["panel_w"]), tuple(got["start_pos"]))
        if traced["term"] != want["term"]:
            return False
        if not (bool(traced["hit_h"]) and bool(traced["hit_v"])):
            return False
        if int(traced["count"]) != int(want["count"]):
            return False
        terms.append(traced["term"])
        masks.append(traced["mask"])

    if set(terms) != {"corner", "return"}:
        return False
    if np.array_equal(masks[0], masks[1]):
        return False

    expected = _expected_output(spec)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    panel_w = int(parsed["panel_w"]) + 2
    left_panel = actual[:, :panel_w]
    right_panel = actual[:, panel_w + 1 :]
    if np.array_equal(left_panel, right_panel):
        return False
    if int(actual[0, 0]) == int(actual[0, panel_w + 1]):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    rng = np.random.default_rng()
    panel_h, panel_w = SIZE_CHOICES[int(rng.integers(0, len(SIZE_CHOICES)))]
    candidates = _candidate_paths(panel_h, panel_w)

    branch_order = ["corner", "return"]
    if int(rng.integers(0, 2)) == 1:
        branch_order.reverse()

    chosen_columns: list[dict[str, object]] = []
    used_masks: set[bytes] = set()
    for branch in branch_order:
        options = list(candidates[branch])
        rng.shuffle(options)
        picked = None
        for option in options:
            mask_key = np.asarray(option["mask"], dtype=np.uint8).tobytes()
            if mask_key in used_masks:
                continue
            picked = option
            used_masks.add(mask_key)
            break
        if picked is None:
            return None
        chosen_columns.append(
            {
                "term": str(picked["term"]),
                "count": int(picked["count"]),
                "start_pos": tuple(picked["start"]),
            }
        )

    swatch_rows = [0, 1] if int(rng.integers(0, 2)) == 0 else [1, 0]
    swatch_colors = rng.choice(np.array(SWATCH_POOL, dtype=int), size=2, replace=False)
    for i in range(2):
        chosen_columns[i]["swatch_row"] = swatch_rows[i]
        chosen_columns[i]["swatch_color"] = int(swatch_colors[i])

    spec = {
        "panel_h": int(panel_h),
        "panel_w": int(panel_w),
        "columns": chosen_columns,
    }
    grid = _build_input_from_spec(spec)
    if not _semantic_checks(grid, spec):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
