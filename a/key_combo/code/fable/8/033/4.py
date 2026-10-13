from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, colors_present, find_components, get_bg


_TEAL = Color.TEAL
_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
_COUNT_CHOICES = [2, 3, 4, 5]
_SIZE_CHOICES = [2, 3, 4]
_MAX_GENERATION_ATTEMPTS = 400

_BBox = tuple[int, int, int, int]


def _bbox_intersects(a: _BBox, b: _BBox) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expand_bbox(b: _BBox, gap: int, shape: tuple[int, int]) -> _BBox:
    h, w = shape
    return (
        max(0, b[0] - gap),
        max(0, b[1] - gap),
        min(h - 1, b[2] + gap),
        min(w - 1, b[3] + gap),
    )


def _render_output_from_specs(
    shape: tuple[int, int],
    bg: int,
    specs: list[dict[str, object]],
    *,
    step_override: int | None = None,
) -> np.ndarray | None:
    out = np.full(shape, bg, dtype=int)
    occupied = np.zeros(shape, dtype=bool)
    h_lim, w_lim = shape

    for spec in specs:
        r0, c0, r1, c1 = map(int, spec["bbox"])
        color = int(spec["color"])
        dr = int(spec["dr"])
        dc = int(spec["dc"])
        step = int(spec["k"] if step_override is None else step_override)

        tr0 = r0 + step * dr
        tc0 = c0 + step * dc
        tr1 = r1 + step * dr
        tc1 = c1 + step * dc
        if not (0 <= tr0 <= tr1 < h_lim and 0 <= tc0 <= tc1 < w_lim):
            return None
        if occupied[tr0 : tr1 + 1, tc0 : tc1 + 1].any():
            return None
        out[tr0 : tr1 + 1, tc0 : tc1 + 1] = color
        occupied[tr0 : tr1 + 1, tc0 : tc1 + 1] = True

    return out


def _parse_specs(grid: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    teal_coords = {tuple(map(int, rc)) for rc in np.argwhere(g == _TEAL)}
    assert teal_coords, "expected at least one teal control pixel"

    specs: list[dict[str, object]] = []
    covered_non_teal: set[tuple[int, int]] = set()
    used_teal: set[tuple[int, int]] = set()

    for color in colors_present(g, exclude={bg, _TEAL}):
        components = find_components(
            g,
            bg=bg,
            connectivity=4,
            monochromatic=True,
            colors=[color],
            return_format="dict",
        )
        assert components, "each rectangle color must appear"

        rectangle: dict[str, object] | None = None
        dots: list[tuple[int, int]] = []
        for comp in components:
            coords = {tuple(map(int, rc)) for rc in comp["coords"]}
            covered_non_teal.update(coords)
            area = int(comp["area"])
            if area == 1:
                dots.append(next(iter(coords)))
                continue

            r0, c0, r1, c1 = map(int, comp["bbox"])
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            assert h >= 2 and w >= 2, "rectangle bbox must be at least 2x2"
            assert h * w == area + 1, "rectangle body must be a filled rectangle with one missing corner"

            region = g[r0 : r1 + 1, c0 : c1 + 1]
            marks = np.argwhere(region == _TEAL)
            assert marks.shape == (1, 2), "each rectangle must have exactly one teal corner"
            mr, mc = map(int, marks[0])
            assert mr in (0, h - 1) and mc in (0, w - 1), "teal control pixel must be a corner"

            expected = np.full((h, w), color, dtype=int)
            expected[mr, mc] = _TEAL
            assert np.array_equal(region, expected), "teal corner must replace exactly one rectangle corner"
            assert rectangle is None, "each color must have exactly one rectangle"

            rectangle = {
                "bbox": (r0, c0, r1, c1),
                "color": int(color),
                "dr": -1 if mr == 0 else 1,
                "dc": -1 if mc == 0 else 1,
                "shape": (h, w),
            }
            used_teal.add((r0 + mr, c0 + mc))

        assert rectangle is not None, "each color needs one teal-cornered rectangle"
        rectangle["dots"] = tuple(sorted(dots))
        rectangle["k"] = len(dots)
        specs.append(rectangle)

    non_bg = {tuple(map(int, rc)) for rc in np.argwhere(g != bg)}
    assert used_teal == teal_coords, "every teal pixel must belong to one rectangle"
    assert covered_non_teal | used_teal == non_bg, "all non-background pixels must be rectangles or matched dots"
    specs.sort(key=lambda spec: int(spec["color"]))
    return bg, specs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each teal-cornered filled rectangle diagonally by the number of same-colored singleton dots.

    Input structure:
    - A dominant background surrounds several filled rectangles of distinct colors.
    - Each rectangle has exactly one corner cell replaced by a teal control pixel.
    - Away from the rectangles, the remaining non-background pixels are isolated singleton dots whose colors match the rectangles.

    Rule:
    - For each rectangle color, count the same-colored singleton dots.
    - Erase the original rectangle footprint, its teal corner, and all same-colored singleton dots.
    - Restore the missing corner to make a full rectangle, then redraw that rectangle translated by k cells along the diagonal indicated by the teal corner, where k is the counted dot total.

    Key insight:
    - Color pairs each dot set with exactly one rectangle, while the teal corner supplies both the missing cell and the diagonal direction; the dot count is the travel distance.
    """
    g = np.asarray(grid, dtype=int)
    bg, specs = _parse_specs(g)

    out = g.copy()
    for spec in specs:
        r0, c0, r1, c1 = map(int, spec["bbox"])
        out[r0 : r1 + 1, c0 : c1 + 1] = bg
        for r, c in spec["dots"]:
            out[r, c] = bg

    repainted = _render_output_from_specs(g.shape, bg, specs)
    assert repainted is not None, "translated rectangles must stay in bounds without overlap"
    assert np.all(out == bg), "input may only contain rectangles and matched singleton dots"
    return repainted


def _candidate_boxes(rows: int, cols: int, h: int, w: int, dr: int, dc: int, k: int) -> list[tuple[_BBox, _BBox]]:
    out_r0_min = 0 if dr == -1 else k
    out_r0_max = rows - h - k if dr == -1 else rows - h
    out_c0_min = 0 if dc == -1 else k
    out_c0_max = cols - w - k if dc == -1 else cols - w
    if out_r0_min > out_r0_max or out_c0_min > out_c0_max:
        return []

    boxes = []
    for out_r0 in range(out_r0_min, out_r0_max + 1):
        for out_c0 in range(out_c0_min, out_c0_max + 1):
            out_bbox = (out_r0, out_c0, out_r0 + h - 1, out_c0 + w - 1)
            in_bbox = (
                out_r0 - k * dr,
                out_c0 - k * dc,
                out_r0 - k * dr + h - 1,
                out_c0 - k * dc + w - 1,
            )
            boxes.append((out_bbox, in_bbox))
    random.shuffle(boxes)
    return boxes


def _compatible_with_placed(
    candidate: dict[str, object],
    placed: list[dict[str, object] | None],
    shape: tuple[int, int],
) -> bool:
    candidate_boxes = [candidate["in_bbox"], candidate["out_bbox"]]
    for other in placed:
        if other is None:
            continue
        other_boxes = [other["in_bbox"], other["out_bbox"]]
        for a in candidate_boxes:
            expanded = _expand_bbox(a, 1, shape)
            for b in other_boxes:
                if _bbox_intersects(expanded, b):
                    return False
    return True


def _place_rectangles(
    shape: tuple[int, int],
    rects: list[dict[str, int]],
) -> list[dict[str, object]] | None:
    rows, cols = shape
    placed: list[dict[str, object] | None] = [None] * len(rects)
    order = sorted(range(len(rects)), key=lambda i: rects[i]["h"] * rects[i]["w"], reverse=True)
    budget = 12000

    def rec(pos: int) -> bool:
        nonlocal budget
        if pos == len(order):
            return True
        if budget <= 0:
            return False
        budget -= 1

        idx = order[pos]
        rect = rects[idx]
        boxes = _candidate_boxes(rows, cols, rect["h"], rect["w"], rect["dr"], rect["dc"], rect["k"])
        for out_bbox, in_bbox in boxes:
            candidate: dict[str, object] = {
                **rect,
                "out_bbox": out_bbox,
                "in_bbox": in_bbox,
            }
            if not _compatible_with_placed(candidate, placed, shape):
                continue
            placed[idx] = candidate
            if rec(pos + 1):
                return True
            placed[idx] = None
        return False

    if not rec(0):
        return None
    return [item for item in placed if item is not None]


def _paint_input(shape: tuple[int, int], bg: int, placed: list[dict[str, object]]) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for spec in placed:
        r0, c0, r1, c1 = map(int, spec["in_bbox"])
        color = int(spec["color"])
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), color, dtype=int)
        mark_r = 0 if int(spec["dr"]) == -1 else patch.shape[0] - 1
        mark_c = 0 if int(spec["dc"]) == -1 else patch.shape[1] - 1
        patch[mark_r, mark_c] = _TEAL
        grid[r0 : r1 + 1, c0 : c1 + 1] = patch
    return grid


def _paint_expected(shape: tuple[int, int], bg: int, placed: list[dict[str, object]]) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for spec in placed:
        r0, c0, r1, c1 = map(int, spec["out_bbox"])
        out[r0 : r1 + 1, c0 : c1 + 1] = int(spec["color"])
    return out


def _place_dots(
    grid: np.ndarray,
    placed: list[dict[str, object]],
) -> dict[int, list[tuple[int, int]]] | None:
    rows, cols = grid.shape
    blocked = np.zeros((rows, cols), dtype=bool)
    for spec in placed:
        for box in (spec["in_bbox"], spec["out_bbox"]):
            r0, c0, r1, c1 = _expand_bbox(box, 1, grid.shape)
            blocked[r0 : r1 + 1, c0 : c1 + 1] = True

    by_color = {int(spec["color"]): [] for spec in placed}
    order = sorted(placed, key=lambda spec: int(spec["k"]), reverse=True)

    for spec in order:
        color = int(spec["color"])
        need = int(spec["k"])
        for _ in range(need):
            candidates = [(r, c) for r in range(rows) for c in range(cols) if not blocked[r, c]]
            if not candidates:
                return None
            r, c = random.choice(candidates)
            by_color[color].append((r, c))
            blocked[max(0, r - 1) : min(rows, r + 2), max(0, c - 1) : min(cols, c + 2)] = True
            grid[r, c] = color

    return by_color


def _semantic_variants_differ(grid: np.ndarray) -> bool:
    bg, specs = _parse_specs(grid)
    expected = _render_output_from_specs(grid.shape, bg, specs)
    if expected is None:
        return False

    if len(specs) < 2:
        return False
    if len({int(spec["k"]) for spec in specs}) < 2:
        return False
    if len({(int(spec["dr"]), int(spec["dc"])) for spec in specs}) < 2:
        return False

    fixed_step = _render_output_from_specs(grid.shape, bg, specs, step_override=1)
    if fixed_step is not None and np.array_equal(fixed_step, expected):
        return False

    fixed_two = _render_output_from_specs(grid.shape, bg, specs, step_override=2)
    if fixed_two is not None and np.array_equal(fixed_two, expected):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(18, 26)
    cols = random.randint(18, 26)
    bg = random.choice([c for c in Color.ALL_COLORS if c != _TEAL])
    rect_count = random.randint(2, 4)

    palette = [c for c in Color.ALL_COLORS if c not in {bg, _TEAL}]
    colors = random.sample(palette, rect_count)
    counts = random.sample(_COUNT_CHOICES, rect_count)
    directions = random.sample(_DIRS, rect_count)

    rects: list[dict[str, int]] = []
    for color, k, (dr, dc) in zip(colors, counts, directions):
        size_choices = [(h, w) for h in _SIZE_CHOICES for w in _SIZE_CHOICES if min(h, w) <= k]
        h, w = random.choice(size_choices)
        rects.append({"color": color, "h": h, "w": w, "dr": dr, "dc": dc, "k": k})

    placed = _place_rectangles((rows, cols), rects)
    if placed is None:
        return None

    inp = _paint_input((rows, cols), bg, placed)
    dots = _place_dots(inp, placed)
    if dots is None:
        return None

    expected = _paint_expected((rows, cols), bg, placed)
    try:
        actual = main(inp)
    except Exception:
        return None

    counts_arr = np.bincount(inp.ravel(), minlength=10)
    if counts_arr[bg] <= max(counts_arr[c] for c in range(10) if c != bg):
        return None
    if not np.array_equal(actual, expected):
        return None
    if not _semantic_variants_differ(inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid scatter-odometer translation input")
