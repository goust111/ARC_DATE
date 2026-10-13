from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

from arc_utils import Color, bbox_h, bbox_w, find_components, get_bg, neighbors4


PALETTE = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.GREY, Color.ORANGE, Color.TEAL, Color.MAROON]
GRID_H_RANGE = (17, 24)
GRID_W_RANGE = (16, 24)
AXIS_ROW_RANGE = (6, 8)
RECT_COUNT_CHOICES = (2, 3)
RECT_W_RANGE = (2, 5)
RECT_H_MAX = 5
MIN_COL_GAP = 2
SIDE_MARGIN = 1
DISTRACTOR_COUNT_RANGE = (2, 5)
GEN_RETRIES = 300


def _find_separator_row(grid: np.ndarray, bg: int) -> tuple[int, int]:
    rows: list[tuple[int, int]] = []
    for r in range(grid.shape[0]):
        vals = np.unique(grid[r, :])
        if vals.size == 1 and int(vals[0]) != int(bg):
            rows.append((int(r), int(vals[0])))
    assert len(rows) == 1, "expected exactly one full-width separator row"
    return rows[0]


def _is_solid_rectangle(grid: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    color = int(comp["color"])
    area = int(comp["area"])
    return area == bbox_h(comp["bbox"]) * bbox_w(comp["bbox"]) and bool(
        np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color)
    )


def _analyze_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    axis_row, axis_color = _find_separator_row(g, bg)

    distractors: dict[tuple[int, int], int] = {}
    rectangles: list[dict] = []

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        color = int(comp["color"])
        if color == axis_color:
            coords = comp["coords"]
            assert int(comp["area"]) == g.shape[1], "separator must be a single full row"
            assert np.all(coords[:, 0] == axis_row), "separator color may not form extra components"
            continue

        area = int(comp["area"])
        if area == 1:
            r, c = map(int, comp["coords"][0])
            distractors[(r, c)] = color
            continue

        assert _is_solid_rectangle(g, comp), "all non-distractor objects must be solid rectangles"
        r0, c0, r1, c1 = map(int, comp["bbox"])
        assert r1 < axis_row, "source rectangles must lie strictly above the separator"
        d = int(axis_row - r1 - 1)
        assert d >= 1, "source rectangles must be separated from the axis"
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        mirror_top = int(axis_row + d + 1)
        mirror_bottom = int(mirror_top + h - 1)
        assert mirror_bottom < g.shape[0], "mirrored rectangle must fit below the separator"

        rectangles.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "h": h,
                "w": w,
                "d": d,
                "mirror_top": mirror_top,
                "mirror_bottom": mirror_bottom,
            }
        )

    assert 2 <= len(rectangles) <= 3, "expected 2 or 3 source rectangles"
    rectangles.sort(key=lambda rect: rect["c0"])
    seen_d: set[int] = set()
    seen_colors: set[int] = set()
    for i, rect in enumerate(rectangles):
        assert rect["color"] not in seen_colors, "source rectangles must use distinct colors"
        seen_colors.add(rect["color"])
        assert rect["d"] not in seen_d, "source rectangles must have distinct distances"
        seen_d.add(rect["d"])
        if i:
            prev = rectangles[i - 1]
            assert prev["c1"] < rect["c0"], "source rectangles may not overlap in columns"

    return {
        "grid": g,
        "bg": bg,
        "axis_row": axis_row,
        "axis_color": axis_color,
        "rectangles": rectangles,
        "distractors": distractors,
    }


def _paint_constant(
    out: np.ndarray,
    keep_mask: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    color: int,
) -> None:
    view = out[r0 : r1 + 1, c0 : c1 + 1]
    free = ~keep_mask[r0 : r1 + 1, c0 : c1 + 1]
    view[free] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the rectangles across the separator row, color the corridor, and encode distance as mirror stripes.

    Input structure:
    - A black background contains exactly one full-width horizontal separator row in a non-background color.
    - Two or three distinct-color solid rectangles lie strictly above that row with different empty-row distances to it.
    - Extra non-separator foreground cells are isolated single-pixel distractors and do not belong to any rectangle.

    Rule:
    - Ignore the isolated distractor pixels: they stay exactly where they are.
    - For each solid rectangle, measure the empty-row distance d from its bottom edge to the separator row.
    - Mirror the rectangle vertically across the separator so the reflected copy starts d rows below the separator.
    - Fill the full same-column corridor between source and mirror with the separator color, keeping distractor pixels unchanged.
    - Inside the mirrored rectangle, repaint only the top d rows from the separator-facing edge in the source color; the remaining mirror rows become black.

    Key insight:
    - The same hidden variable d controls two separate effects at once: corridor height across the axis and the number of filled stripe rows inside the reflected copy.
    """
    scene = _analyze_scene(grid)
    g = scene["grid"]
    out = g.copy()
    keep_mask = np.zeros(g.shape, dtype=bool)
    for r, c in scene["distractors"]:
        keep_mask[r, c] = True

    bg = int(scene["bg"])
    axis_color = int(scene["axis_color"])
    for rect in scene["rectangles"]:
        r0, c0, r1, c1 = rect["bbox"]
        mirror_top = int(rect["mirror_top"])
        mirror_bottom = int(rect["mirror_bottom"])
        d = int(rect["d"])
        color = int(rect["color"])

        _paint_constant(out, keep_mask, r1 + 1, c0, mirror_top - 1, c1, axis_color)
        _paint_constant(out, keep_mask, mirror_top, c0, mirror_bottom, c1, bg)
        stripe_rows = min(d, int(rect["h"]))
        _paint_constant(out, keep_mask, mirror_top, c0, mirror_top + stripe_rows - 1, c1, color)

    return out


def _composition(total: int, parts: int) -> list[int]:
    if total == 0:
        return [0] * parts
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    values: list[int] = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        values.append(cut - prev - 1)
        prev = cut
    return values


def _pack_intervals(length: int, sizes: list[int], margin: int, min_gap: int) -> list[tuple[int, int]] | None:
    required = 2 * margin + sum(sizes) + min_gap * (len(sizes) - 1)
    extra = int(length - required)
    if extra < 0:
        return None
    slack = _composition(extra, len(sizes) + 1)
    pos = int(margin + slack[0])
    intervals: list[tuple[int, int]] = []
    for i, size in enumerate(sizes):
        intervals.append((pos, pos + int(size) - 1))
        pos += int(size)
        if i + 1 < len(sizes):
            pos += int(min_gap + slack[i + 1])
    return intervals


def _choose_specs(axis_row: int, rect_count: int) -> list[dict] | None:
    distances = list(range(1, axis_row - 2))
    if len(distances) < rect_count:
        return None

    long_candidates = []
    short_candidates = []
    for d in distances:
        max_h = min(RECT_H_MAX, axis_row - d - 1)
        if max_h < 2:
            continue
        if min(max_h, d) >= 2:
            long_candidates.append(d)
        if max_h >= d + 1:
            short_candidates.append(d)

    if not long_candidates or not short_candidates:
        return None

    for _ in range(100):
        long_d = random.choice(long_candidates)
        short_d_choices = [d for d in short_candidates if d != long_d]
        if not short_d_choices:
            continue
        short_d = random.choice(short_d_choices)
        chosen = [long_d, short_d]
        remaining = [d for d in distances if d not in chosen]
        if rect_count == 3:
            if not remaining:
                continue
            chosen.append(random.choice(remaining))

        specs: list[dict] = []
        for d in chosen:
            max_h = min(RECT_H_MAX, axis_row - d - 1)
            if d == long_d:
                h = random.randint(2, min(max_h, d))
            elif d == short_d:
                h = random.randint(d + 1, max_h)
            else:
                h = random.randint(2, max_h)
            w = random.randint(*RECT_W_RANGE)
            specs.append({"d": int(d), "h": int(h), "w": int(w)})

        if any(spec["d"] >= spec["h"] for spec in specs) and any(spec["d"] < spec["h"] for spec in specs):
            random.shuffle(specs)
            return specs
    return None


def _place_distractors(
    grid: np.ndarray,
    protected: np.ndarray,
    rectangle_colors: list[int],
) -> None:
    count = random.randint(*DISTRACTOR_COUNT_RANGE)
    extras = [color for color in PALETTE if color not in rectangle_colors]
    color_pool = rectangle_colors + extras
    placed = 0
    attempts = 0
    while placed < count and attempts < 500:
        attempts += 1
        r = random.randrange(grid.shape[0])
        c = random.randrange(grid.shape[1])
        if protected[r, c] or grid[r, c] != Color.BLACK:
            continue
        if any(
            0 <= nr < grid.shape[0]
            and 0 <= nc < grid.shape[1]
            and grid[nr, nc] != Color.BLACK
            for nr, nc in neighbors4(r, c)
        ):
            continue
        color = random.choice(color_pool)
        grid[r, c] = int(color)
        placed += 1

    assert placed == count, "failed to place all distractors"


def _validate_output(inp: np.ndarray, out: np.ndarray) -> None:
    scene = _analyze_scene(inp)
    assert inp.shape == out.shape
    assert np.array_equal(out[scene["axis_row"], :], inp[scene["axis_row"], :])

    touched = np.zeros(inp.shape, dtype=bool)
    touched[scene["axis_row"], :] = True
    for r, c in scene["distractors"]:
        assert int(out[r, c]) == int(inp[r, c]), "distractors must stay unchanged"
        touched[r, c] = True

    for rect in scene["rectangles"]:
        r0, c0, r1, c1 = rect["bbox"]
        mirror_top = int(rect["mirror_top"])
        mirror_bottom = int(rect["mirror_bottom"])
        d = int(rect["d"])
        color = int(rect["color"])
        stripe_rows = min(d, int(rect["h"]))

        assert np.array_equal(out[r0 : r1 + 1, c0 : c1 + 1], inp[r0 : r1 + 1, c0 : c1 + 1])
        touched[r0 : r1 + 1, c0 : c1 + 1] = True

        for rr in range(r1 + 1, mirror_top):
            for cc in range(c0, c1 + 1):
                if (rr, cc) in scene["distractors"]:
                    continue
                assert int(out[rr, cc]) == int(scene["axis_color"])
                touched[rr, cc] = True

        for rr in range(mirror_top, mirror_bottom + 1):
            for cc in range(c0, c1 + 1):
                if (rr, cc) in scene["distractors"]:
                    continue
                expected = color if rr < mirror_top + stripe_rows else int(scene["bg"])
                assert int(out[rr, cc]) == expected
                touched[rr, cc] = True

    untouched = ~touched
    assert np.array_equal(out[untouched], inp[untouched]), "unrelated cells must remain unchanged"


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        axis_row = random.randint(*AXIS_ROW_RANGE)
        rect_count = random.choice(RECT_COUNT_CHOICES)
        specs = _choose_specs(axis_row, rect_count)
        if specs is None:
            continue

        max_mirror_bottom = max(axis_row + spec["d"] + spec["h"] for spec in specs)
        min_h = max(GRID_H_RANGE[0], max_mirror_bottom + 2)
        if min_h > GRID_H_RANGE[1]:
            continue
        height = random.randint(min_h, GRID_H_RANGE[1])

        widths = [spec["w"] for spec in specs]
        min_width = 2 * SIDE_MARGIN + sum(widths) + MIN_COL_GAP * (rect_count - 1)
        if min_width > GRID_W_RANGE[1]:
            continue
        width = random.randint(max(GRID_W_RANGE[0], min_width), GRID_W_RANGE[1])
        intervals = _pack_intervals(width, widths, SIDE_MARGIN, MIN_COL_GAP)
        if intervals is None:
            continue

        colors = random.sample(PALETTE, rect_count)
        grid = np.full((height, width), Color.BLACK, dtype=int)
        grid[axis_row, :] = Color.PINK

        protected = np.zeros_like(grid, dtype=bool)
        planned: list[dict] = []
        for spec, color, (c0, c1) in zip(specs, colors, intervals):
            h = int(spec["h"])
            d = int(spec["d"])
            r1 = int(axis_row - d - 1)
            r0 = int(r1 - h + 1)
            mirror_top = int(axis_row + d + 1)
            mirror_bottom = int(mirror_top + h - 1)
            grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)
            protected[r0 : r1 + 1, c0 : c1 + 1] = True
            protected[r1 + 1 : mirror_bottom + 1, c0 : c1 + 1] = True
            planned.append(
                {
                    "color": int(color),
                    "bbox": (r0, c0, r1, c1),
                    "d": d,
                    "h": h,
                    "w": int(c1 - c0 + 1),
                }
            )

        _place_distractors(grid, protected, colors)

        scene = _analyze_scene(grid)
        decoded = [(rect["color"], rect["bbox"], rect["d"], rect["h"], rect["w"]) for rect in scene["rectangles"]]
        expected = sorted((item["color"], item["bbox"], item["d"], item["h"], item["w"]) for item in planned)
        if sorted(decoded) != expected:
            continue
        if not any(rect["d"] >= rect["h"] for rect in scene["rectangles"]):
            continue
        if not any(rect["d"] < rect["h"] for rect in scene["rectangles"]):
            continue

        out = main(grid)
        _validate_output(grid, out)
        assert max(grid.shape) <= 30 and max(out.shape) <= 30
        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
