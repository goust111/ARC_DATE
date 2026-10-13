from __future__ import annotations

import random
import sys
from itertools import combinations
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, colors_present, get_bg, paste, random_free_location


BACKGROUND = Color.BLACK
COLOR_POOL = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.GREY, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
GRID_HEIGHT_RANGE = (18, 28)
GRID_WIDTH_RANGE = (18, 28)
GROUP_COUNT_RANGE = (2, 4)
MAX_RECTS_TOTAL = 7
GEN_RETRIES = 400

FOCUS_TEMPLATES = [(1, 1), (2, 1), (1, 2), (2, 2), (2, 0), (0, 2)]
FOCUS_WEIGHTS = [4, 2, 2, 1, 1, 1]
SIMPLE_TEMPLATES = [(1, 0), (0, 1), (1, 1)]
GENERAL_TEMPLATES = SIMPLE_TEMPLATES + [(2, 0), (0, 2), (2, 1), (1, 2)]


def _parse_edge_cuts(grid: np.ndarray, bg: int) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    h, w = grid.shape
    row_cuts: dict[int, list[int]] = {}
    col_cuts: dict[int, list[int]] = {}

    for r in range(1, h - 1):
        left = int(grid[r, 0])
        right = int(grid[r, w - 1])
        if left != bg:
            row_cuts.setdefault(left, []).append(r)
        if right != bg:
            row_cuts.setdefault(right, []).append(r)

    for c in range(1, w - 1):
        top = int(grid[0, c])
        bottom = int(grid[h - 1, c])
        if top != bg:
            col_cuts.setdefault(top, []).append(c)
        if bottom != bg:
            col_cuts.setdefault(bottom, []).append(c)

    return row_cuts, col_cuts


def _shift_from_cuts(index: int, cuts: list[int]) -> int:
    delta = 0
    for cut in cuts:
        delta += -1 if index < cut else 1
    return delta


def _apply_cuts(grid: np.ndarray, row_cuts: dict[int, list[int]], col_cuts: dict[int, list[int]], bg: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = np.full_like(g, bg)

    row_sets = {color: set(rows) for color, rows in row_cuts.items()}
    col_sets = {color: set(cols) for color, cols in col_cuts.items()}

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            color = int(g[r, c])
            if color == bg:
                continue

            rows = row_cuts.get(color, [])
            cols = col_cuts.get(color, [])
            if r in row_sets.get(color, set()) or c in col_sets.get(color, set()):
                continue

            nr = r + _shift_from_cuts(r, rows)
            nc = c + _shift_from_cuts(c, cols)
            assert 0 <= nr < h and 0 <= nc < w, "shifted pixel left the grid"
            assert int(out[nr, nc]) == bg, "distinct transformed pixels must not collide"
            out[nr, nc] = color

    return out


def _spec_transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    h, w = g.shape
    row_cuts, col_cuts = _parse_edge_cuts(g, bg)
    out = np.full_like(g, bg)

    sorted_rows = {color: tuple(sorted(rows)) for color, rows in row_cuts.items()}
    sorted_cols = {color: tuple(sorted(cols)) for color, cols in col_cuts.items()}
    row_sets = {color: set(rows) for color, rows in sorted_rows.items()}
    col_sets = {color: set(cols) for color, cols in sorted_cols.items()}

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            color = int(g[r, c])
            if color == bg:
                continue

            rows = sorted_rows.get(color, ())
            cols = sorted_cols.get(color, ())
            if r in row_sets.get(color, set()) or c in col_sets.get(color, set()):
                continue

            lower_rows = sum(cut < r for cut in rows)
            upper_rows = len(rows) - sum(cut <= r for cut in rows)
            lower_cols = sum(cut < c for cut in cols)
            upper_cols = len(cols) - sum(cut <= c for cut in cols)
            nr = r + lower_rows - upper_rows
            nc = c + lower_cols - upper_cols

            assert 0 <= nr < h and 0 <= nc < w, "spec transform moved a pixel out of bounds"
            assert int(out[nr, nc]) == bg, "spec transform produced a collision"
            out[nr, nc] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split same-colored objects by invisible edge-aligned cut lines and push every surviving side one cell away per cut.

    Input structure:
    - The background is black and the interior contains several monochrome solid foreground objects.
    - Non-black singleton markers may appear only on the outer border.
    - Left/right border markers define horizontal cut lines on their rows, while top/bottom border markers define vertical cut lines on their columns.

    Rule:
    - Each cut line acts only on interior pixels whose color matches the marker color.
    - Any matching pixel lying exactly on one of that color's cut rows or cut columns is erased.
    - Every remaining matching pixel is displaced by the sum of all same-color cuts: above a horizontal cut it moves up one, below it moves down one, left of a vertical cut it moves left one, and right of it moves right one.
    - All displacements are computed from the original grid simultaneously, and every border marker is erased in the output.

    Key insight:
    - The border markers do not draw visible cracks themselves; they only specify invisible global row/column knives, and color agreement decides which objects those knives actually move.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))
    row_cuts, col_cuts = _parse_edge_cuts(g, bg)
    return _apply_cuts(g, row_cuts, col_cuts, bg)


def _weighted_choice(options: list[tuple[int, int]], weights: list[int]) -> tuple[int, int]:
    return random.choices(options, weights=weights, k=1)[0]


def _choose_spaced_positions(lo: int, hi: int, count: int) -> list[int] | None:
    if count == 0:
        return []
    values = list(range(lo, hi + 1))
    combos = [combo for combo in combinations(values, count) if all(b - a >= 2 for a, b in zip(combo, combo[1:]))]
    if not combos:
        return None
    return list(random.choice(combos))


def _rect_bbox(rect: dict) -> tuple[int, int, int, int]:
    return rect["r"], rect["c"], rect["r"] + rect["h"] - 1, rect["c"] + rect["w"] - 1


def _intersects_foreign_cut(rect: dict, color: int, row_by_color: dict[int, list[int]], col_by_color: dict[int, list[int]]) -> bool:
    r0, c0, r1, c1 = _rect_bbox(rect)
    for other_color, rows in row_by_color.items():
        if other_color != color and any(r0 <= row <= r1 for row in rows):
            return True
    for other_color, cols in col_by_color.items():
        if other_color != color and any(c0 <= col <= c1 for col in cols):
            return True
    return False


def _random_group_templates() -> list[dict]:
    n_groups = random.randint(*GROUP_COUNT_RANGE)
    templates: list[dict] = []

    focus_rows, focus_cols = _weighted_choice(FOCUS_TEMPLATES, FOCUS_WEIGHTS)
    templates.append({"row_cut_count": focus_rows, "col_cut_count": focus_cols, "affected": True})

    while len(templates) < n_groups:
        if len(templates) == 1:
            if random.random() < 0.8:
                rows, cols = random.choice(SIMPLE_TEMPLATES)
                templates.append({"row_cut_count": rows, "col_cut_count": cols, "affected": True})
            else:
                templates.append({"row_cut_count": 0, "col_cut_count": 0, "affected": False})
            continue

        if random.random() < 0.4:
            templates.append({"row_cut_count": 0, "col_cut_count": 0, "affected": False})
        else:
            rows, cols = random.choice(GENERAL_TEMPLATES)
            templates.append({"row_cut_count": rows, "col_cut_count": cols, "affected": True})

    if not any(template["row_cut_count"] > 0 for template in templates):
        templates[0]["row_cut_count"] = 1
        templates[0]["affected"] = True
    if not any(template["col_cut_count"] > 0 for template in templates):
        templates[min(1, len(templates) - 1)]["col_cut_count"] = 1
        templates[min(1, len(templates) - 1)]["affected"] = True

    if len(templates) >= 3 and all(template["affected"] for template in templates) and random.random() < 0.7:
        templates[-1] = {"row_cut_count": 0, "col_cut_count": 0, "affected": False}

    random.shuffle(templates)
    return templates


def _make_group_specs() -> list[dict]:
    templates = _random_group_templates()
    colors = random.sample(COLOR_POOL, len(templates))
    groups: list[dict] = []
    total_rects = 0

    for color, template in zip(colors, templates):
        row_cuts = int(template["row_cut_count"])
        col_cuts = int(template["col_cut_count"])
        min_h = max(2, 2 * row_cuts + 1)
        min_w = max(2, 2 * col_cuts + 1)
        primary_h = random.randint(min_h, min(7, min_h + 2))
        primary_w = random.randint(min_w, min(7, min_w + 2))
        group = {
            "color": int(color),
            "row_cut_count": row_cuts,
            "col_cut_count": col_cuts,
            "rect_shapes": [{"h": primary_h, "w": primary_w, "primary": True}],
            "rects": [],
        }

        extra_prob = 0.7 if row_cuts + col_cuts > 0 else 0.5
        if random.random() < extra_prob:
            extra_h = random.randint(2, 5)
            extra_w = random.randint(2, 5)
            group["rect_shapes"].append({"h": extra_h, "w": extra_w, "primary": False})

        groups.append(group)
        total_rects += len(group["rect_shapes"])

    while total_rects > MAX_RECTS_TOTAL:
        candidates = [group for group in groups if len(group["rect_shapes"]) > 1]
        if not candidates:
            break
        chosen = random.choice(candidates)
        chosen["rect_shapes"].pop()
        total_rects -= 1

    return groups


def _place_rectangles(groups: list[dict], h: int, w: int) -> bool:
    occupancy = np.zeros((h, w), dtype=int)
    pending = []
    for group_index, group in enumerate(groups):
        for shape in group["rect_shapes"]:
            exp_h = shape["h"] + 2 * group["row_cut_count"]
            exp_w = shape["w"] + 2 * group["col_cut_count"]
            pending.append((exp_h * exp_w, group_index, shape))

    pending.sort(key=lambda item: (-item[0], -item[2]["h"] * item[2]["w"]))

    for _area, group_index, shape in pending:
        patch = np.ones(
            (
                shape["h"] + 2 * groups[group_index]["row_cut_count"],
                shape["w"] + 2 * groups[group_index]["col_cut_count"],
            ),
            dtype=int,
        )
        try:
            top, left = random_free_location(
                occupancy,
                patch,
                transparent=0,
                border=1,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return False

        paste(occupancy, patch, top, left)
        groups[group_index]["rects"].append(
            {
                "r": top + groups[group_index]["row_cut_count"],
                "c": left + groups[group_index]["col_cut_count"],
                "h": shape["h"],
                "w": shape["w"],
                "primary": bool(shape["primary"]),
            }
        )

    return True


def _assign_group_cuts(groups: list[dict], h: int, w: int) -> bool:
    used_row_markers: set[tuple[int, int]] = set()
    used_col_markers: set[tuple[int, int]] = set()

    for group in groups:
        primary = next(rect for rect in group["rects"] if rect["primary"])
        r0, c0, r1, c1 = _rect_bbox(primary)

        rows = _choose_spaced_positions(r0 + 1, r1 - 1, group["row_cut_count"])
        cols = _choose_spaced_positions(c0 + 1, c1 - 1, group["col_cut_count"])
        if rows is None or cols is None:
            return False

        row_markers = []
        for row in rows:
            choices = []
            if (row, 0) not in used_row_markers:
                choices.append((row, 0))
            if (row, w - 1) not in used_row_markers:
                choices.append((row, w - 1))
            if not choices:
                return False
            cell = random.choice(choices)
            used_row_markers.add(cell)
            row_markers.append(cell)

        col_markers = []
        for col in cols:
            choices = []
            if (0, col) not in used_col_markers:
                choices.append((0, col))
            if (h - 1, col) not in used_col_markers:
                choices.append((h - 1, col))
            if not choices:
                return False
            cell = random.choice(choices)
            used_col_markers.add(cell)
            col_markers.append(cell)

        group["row_positions"] = rows
        group["col_positions"] = cols
        group["row_markers"] = row_markers
        group["col_markers"] = col_markers

    return True


def _build_input_grid(groups: list[dict], h: int, w: int) -> np.ndarray:
    grid = np.full((h, w), BACKGROUND, dtype=int)
    for group in groups:
        for rect in group["rects"]:
            r0, c0, r1, c1 = _rect_bbox(rect)
            grid[r0 : r1 + 1, c0 : c1 + 1] = group["color"]
        for r, c in group["row_markers"]:
            grid[r, c] = group["color"]
        for r, c in group["col_markers"]:
            grid[r, c] = group["color"]
    return grid


def _semantic_checks(grid: np.ndarray, groups: list[dict]) -> bool:
    h, w = grid.shape
    if max(h, w) > 30:
        return False
    if int(get_bg(grid)) != BACKGROUND:
        return False
    if any(int(grid[0, corner]) != BACKGROUND for corner in (0, w - 1)):
        return False
    if any(int(grid[h - 1, corner]) != BACKGROUND for corner in (0, w - 1)):
        return False

    interior_colors = colors_present(grid[1:-1, 1:-1], exclude={BACKGROUND})
    if len(interior_colors) < 2:
        return False

    row_cuts = {group["color"]: list(group["row_positions"]) for group in groups if group["row_positions"]}
    col_cuts = {group["color"]: list(group["col_positions"]) for group in groups if group["col_positions"]}

    if not any(rows for rows in row_cuts.values()):
        return False
    if not any(cols for cols in col_cuts.values()):
        return False
    if not any((group["row_cut_count"] > 1 or group["col_cut_count"] > 1 or (group["row_cut_count"] > 0 and group["col_cut_count"] > 0)) for group in groups):
        return False

    foreign_crossing = False
    for group in groups:
        for rect in group["rects"]:
            if _intersects_foreign_cut(rect, group["color"], row_cuts, col_cuts):
                foreign_crossing = True
                break
        if foreign_crossing:
            break
    if not foreign_crossing:
        return False

    expected = _spec_transform(grid)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    return bool(np.array_equal(actual, expected))


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    groups = _make_group_specs()

    if not _place_rectangles(groups, h, w):
        return None
    if not _assign_group_cuts(groups, h, w):
        return None

    grid = _build_input_grid(groups, h, w)
    if not _semantic_checks(grid, groups):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid edge-cut sample")
