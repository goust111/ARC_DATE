import random
from typing import Any

import numpy as np


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _count_runs(flags: np.ndarray) -> int:
    count = 0
    in_run = False
    for value in flags.tolist():
        if value and not in_run:
            count += 1
            in_run = True
        elif not value:
            in_run = False
    return count


def _positive_composition(total: int, parts: int) -> list[int]:
    assert total >= parts >= 1
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    points = [0] + cuts + [total]
    return [points[i + 1] - points[i] for i in range(parts)]


def _paint_rect(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    grid[r0 : r0 + h, c0 : c0 + w] = color


def _object_specs_from_grid(grid: np.ndarray) -> list[dict[str, Any]]:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    specs: list[dict[str, Any]] = []

    for color in sorted(int(c) for c in np.unique(g) if int(c) != bg):
        coords = np.argwhere(g == color)
        assert coords.size > 0, "foreground color must be present"
        r0, c0 = coords.min(axis=0).astype(int)
        r1, c1 = coords.max(axis=0).astype(int)
        box = g[r0 : r1 + 1, c0 : c1 + 1]

        full_bg_rows = np.all(box == bg, axis=1)
        full_bg_cols = np.all(box == bg, axis=0)
        row_runs = _count_runs(np.any(box == color, axis=1))
        col_runs = _count_runs(np.any(box == color, axis=0))

        has_h_cuts = bool(full_bg_rows.any())
        has_v_cuts = bool(full_bg_cols.any())
        assert has_h_cuts ^ has_v_cuts, "each object must have exactly one cut orientation"

        if has_h_cuts:
            direction = "down"
            copies = row_runs
        else:
            direction = "right"
            copies = col_runs

        specs.append(
            {
                "color": color,
                "r0": int(r0),
                "c0": int(c0),
                "h": int(r1 - r0 + 1),
                "w": int(c1 - c0 + 1),
                "direction": direction,
                "copies": int(copies),
            }
        )

    return specs


def _render_output(shape: tuple[int, int], bg: int, specs: list[dict[str, Any]]) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), bg, dtype=int)

    for spec in specs:
        color = int(spec["color"])
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        rect_h = int(spec["h"])
        rect_w = int(spec["w"])
        copies = int(spec["copies"])

        if spec["direction"] == "down":
            step = rect_h + 1
            for i in range(copies):
                rr = r0 + i * step
                if rr + rect_h <= h:
                    _paint_rect(out, rr, c0, rect_h, rect_w, color)
        else:
            step = rect_w + 1
            for i in range(copies):
                cc = c0 + i * step
                if cc + rect_w <= w:
                    _paint_rect(out, r0, cc, rect_h, rect_w, color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild each cut rectangle, then repeat it in the direction perpendicular to its internal cut lines.

    Input structure:
    - The dominant color is the background.
    - Each foreground color forms one solid rectangle that has been split by one or more full background lines inside its bbox.
    - Within one object, the cut lines are all horizontal or all vertical, so the colored remnants form several parallel rectangular segments.

    Rule:
    - For each foreground color, take the bbox of all its pixels as the original rectangle before cutting.
    - Count how many colored segments remain: full background rows imply horizontal cuts and full background columns imply vertical cuts.
    - Fill the original bbox back into a complete rectangle of that color, then place the same rectangle that many times with a 1-cell background gap.
    - Horizontal cuts produce a downward stack of copies, while vertical cuts produce a rightward row of copies; any copy that would go out of bounds is skipped.

    Key insight:
    - The internal background lines do not indicate missing pieces to preserve; they jointly encode the repeat count and the orthogonal expansion direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background_color(g)
    specs = _object_specs_from_grid(g)
    return _render_output(g.shape, bg, specs)


# --------------- generator ---------------

ARC_MAX = 30
GRID_H_RANGE = (16, 30)
GRID_W_RANGE = (16, 30)
OBJECT_COUNT_CHOICES = (1, 2, 3, 4)
OBJECT_COUNT_WEIGHTS = (1, 3, 3, 2)
COPY_RANGE = (2, 4)
RECT_OTHER_AXIS_RANGE = (2, 7)
MAX_PRIMARY_AXIS = 8
CLIP_PROB = 0.35
MARGIN = 1
GEN_RETRIES = 400


def _visible_count_limit(total_space: int, size: int) -> int:
    return 1 + max(0, (total_space - size) // (size + 1))


def _sample_anchor(total_space: int, size: int, copies: int, clip: bool) -> tuple[int, int] | None:
    max_visible = min(copies, _visible_count_limit(total_space, size))
    if max_visible <= 0:
        return None

    if clip:
        if copies == 1 or max_visible >= copies:
            return None
        min_visible = 2 if max_visible >= 2 else 1
        visible = random.randint(min_visible, max_visible)
    else:
        if max_visible < copies:
            return None
        visible = copies

    lower = max(0, total_space - size - visible * (size + 1) + 1)
    upper = min(total_space - size, total_space - size - (visible - 1) * (size + 1))
    if lower > upper:
        return None
    return random.randint(lower, upper), visible


def _sample_object(grid_h: int, grid_w: int) -> dict[str, Any] | None:
    cut_axis = random.choice(("horizontal", "vertical"))
    copies = random.randint(*COPY_RANGE)
    min_primary = 2 * copies - 1
    if min_primary > MAX_PRIMARY_AXIS:
        return None

    if cut_axis == "horizontal":
        rect_h = random.randint(min_primary, MAX_PRIMARY_AXIS)
        rect_w = random.randint(*RECT_OTHER_AXIS_RANGE)
        clip = random.random() < CLIP_PROB
        anchor = _sample_anchor(grid_h, rect_h, copies, clip)
        if anchor is None:
            return None
        r0, visible = anchor
        if rect_w + 2 * MARGIN > grid_w:
            return None
        c0 = random.randint(MARGIN, grid_w - rect_w - MARGIN)
        segment_sizes = _positive_composition(rect_h - (copies - 1), copies)
        return {
            "color": None,
            "cut_axis": cut_axis,
            "copies": copies,
            "visible": visible,
            "r0": r0,
            "c0": c0,
            "h": rect_h,
            "w": rect_w,
            "segments": segment_sizes,
        }

    rect_w = random.randint(min_primary, MAX_PRIMARY_AXIS)
    rect_h = random.randint(*RECT_OTHER_AXIS_RANGE)
    clip = random.random() < CLIP_PROB
    anchor = _sample_anchor(grid_w, rect_w, copies, clip)
    if anchor is None:
        return None
    c0, visible = anchor
    if rect_h + 2 * MARGIN > grid_h:
        return None
    r0 = random.randint(MARGIN, grid_h - rect_h - MARGIN)
    segment_sizes = _positive_composition(rect_w - (copies - 1), copies)
    return {
        "color": None,
        "cut_axis": cut_axis,
        "copies": copies,
        "visible": visible,
        "r0": r0,
        "c0": c0,
        "h": rect_h,
        "w": rect_w,
        "segments": segment_sizes,
    }


def _footprint_bounds(spec: dict[str, Any]) -> tuple[int, int, int, int]:
    r0 = int(spec["r0"])
    c0 = int(spec["c0"])
    h = int(spec["h"])
    w = int(spec["w"])
    visible = int(spec["visible"])

    if spec["cut_axis"] == "horizontal":
        r1 = r0 + (visible - 1) * (h + 1) + h - 1
        c1 = c0 + w - 1
    else:
        r1 = r0 + h - 1
        c1 = c0 + (visible - 1) * (w + 1) + w - 1
    return r0, c0, r1, c1


def _bbox_clear(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    return not occupied[r0 : r1 + 1, c0 : c1 + 1].any()


def _mark_bbox(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    occupied[r0 : r1 + 1, c0 : c1 + 1] = True


def _build_input(shape: tuple[int, int], bg: int, specs: list[dict[str, Any]]) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for spec in specs:
        color = int(spec["color"])
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        h = int(spec["h"])
        w = int(spec["w"])
        _paint_rect(grid, r0, c0, h, w, color)

        if spec["cut_axis"] == "horizontal":
            cursor = r0
            for size in spec["segments"][:-1]:
                cursor += int(size)
                grid[cursor, c0 : c0 + w] = bg
                cursor += 1
        else:
            cursor = c0
            for size in spec["segments"][:-1]:
                cursor += int(size)
                grid[r0 : r0 + h, cursor] = bg
                cursor += 1
    return grid


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    grid_w = random.randint(*GRID_W_RANGE)
    bg = random.randint(0, 9)
    colors = [c for c in range(10) if c != bg]
    random.shuffle(colors)

    target_objects = random.choices(OBJECT_COUNT_CHOICES, weights=OBJECT_COUNT_WEIGHTS, k=1)[0]
    target_objects = min(target_objects, len(colors))

    specs: list[dict[str, Any]] = []
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    for idx in range(target_objects):
        placed = False
        for _ in range(120):
            spec = _sample_object(grid_h, grid_w)
            if spec is None:
                continue
            r0, c0, r1, c1 = _footprint_bounds(spec)
            pad_r0 = max(0, r0 - MARGIN)
            pad_c0 = max(0, c0 - MARGIN)
            pad_r1 = min(grid_h - 1, r1 + MARGIN)
            pad_c1 = min(grid_w - 1, c1 + MARGIN)
            if not _bbox_clear(occupied, pad_r0, pad_c0, pad_r1, pad_c1):
                continue
            spec["color"] = colors[idx]
            specs.append(spec)
            _mark_bbox(occupied, pad_r0, pad_c0, pad_r1, pad_c1)
            placed = True
            break
        if not placed:
            return None

    horizontal_n = sum(spec["cut_axis"] == "horizontal" for spec in specs)
    vertical_n = len(specs) - horizontal_n
    if len(specs) >= 2 and (horizontal_n == 0 or vertical_n == 0):
        return None

    clipped_n = sum(spec["visible"] < spec["copies"] for spec in specs)
    unclipped_n = len(specs) - clipped_n
    if len(specs) >= 2 and (clipped_n == 0 or unclipped_n == 0):
        return None

    grid = _build_input((grid_h, grid_w), bg, specs)
    if np.count_nonzero(grid == bg) <= grid.size // 2:
        return None

    expected_specs = []
    for spec in specs:
        expected_specs.append(
            {
                "color": int(spec["color"]),
                "r0": int(spec["r0"]),
                "c0": int(spec["c0"]),
                "h": int(spec["h"]),
                "w": int(spec["w"]),
                "direction": "down" if spec["cut_axis"] == "horizontal" else "right",
                "copies": int(spec["copies"]),
            }
        )
    expected = _render_output((grid_h, grid_w), bg, expected_specs)

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
