from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, paste


BG = Color.BLACK
SEP = Color.GREY
PALETTE = [color for color in Color.ALL_COLORS if color not in (BG, SEP)]

LAYOUTS = [(3, 5), (3, 6), (3, 7), (3, 8), (4, 5), (4, 6)]
HEIGHT_RANGE = (11, 18)
OBJECT_COUNT_RANGE = (2, 4)
OBJECT_HEIGHT_RANGE = (2, 5)

GEN_RETRIES = 400
PANEL_RETRIES = 80
MASK_RETRIES = 60


def _dominant(values) -> int:
    return int(Counter(map(int, values)).most_common(1)[0][0])


def _separator_info(grid: np.ndarray) -> tuple[int, int, list[tuple[int, int, int]]]:
    height, width = grid.shape
    sep_color = _dominant(grid[:, 0])
    info: list[tuple[int, int, int]] = []
    for col in range(width):
        non_sep = np.flatnonzero(grid[:, col] != sep_color)
        if len(non_sep) == 1:
            row = int(non_sep[0])
            info.append((col, row, int(grid[row, col])))

    assert info and info[0][0] == 0, "column 0 must be a separator"
    if len(info) == 1:
        panel_w = width - 1
    else:
        step = info[1][0] - info[0][0]
        assert step >= 2, "panel width must be positive"
        assert all(info[i + 1][0] - info[i][0] == step for i in range(len(info) - 1))
        panel_w = step - 1

    expected_cols = [i * (panel_w + 1) for i in range(len(info))]
    assert [col for col, _row, _color in info] == expected_cols
    assert width == len(info) * (panel_w + 1)
    assert all(color != sep_color for _col, _row, color in info)
    assert all(int(np.count_nonzero(grid[:, col] == sep_color)) == height - 1 for col, _row, _color in info)
    return sep_color, panel_w, info


def _panel_background(grid: np.ndarray, panel_w: int) -> int:
    step = panel_w + 1
    panel_cols = [col for col in range(grid.shape[1]) if col % step != 0]
    return _dominant(grid[:, panel_cols].ravel())


def _sorted_objects(panel: np.ndarray, bg: int) -> list[dict]:
    objects = find_components(panel, bg=bg, connectivity=4, monochromatic=False)
    assert objects, "each panel must contain foreground objects"
    objects = sorted(objects, key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))

    prev_bottom = -1
    for obj in objects:
        r0, _c0, r1, _c1 = map(int, obj["bbox"])
        assert r0 > prev_bottom, "panel objects must have non-overlapping row spans"
        prev_bottom = r1
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select one object from each separator-controlled panel by dial row, recolor it, and overlay the survivors on one canvas.

    Input structure
    - The grid is split by solid grey separator columns into equal-width panels, and column 0 is also a separator.
    - Each separator column contains exactly one non-grey dial pixel that controls the panel immediately to its right.
    - Inside every panel, several foreground objects sit on a shared background, and their vertical row spans do not overlap.

    Rule
    - For each panel, read the row of the dial pixel on its left separator.
    - Keep the unique panel object whose bbox row interval contains that row, and discard the other objects.
    - Recolor every kept object to the dial pixel's color, then overlay all recolored survivors on a single panel-sized canvas using panel-local coordinates.

    Key insight
    - The grey columns are control tracks rather than picture content: the dial row chooses which object survives, and the dial color determines the survivor's new color before shared-coordinate overlay.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    height, _width = g.shape
    sep_color, panel_w, separators = _separator_info(g)
    bg = _panel_background(g, panel_w)
    out = np.full((height, panel_w), bg, dtype=int)

    for sep_col, dial_row, dial_color in separators:
        panel = g[:, sep_col + 1 : sep_col + 1 + panel_w]
        assert panel.shape == (height, panel_w)
        assert not np.any(panel == sep_color), "separator color should stay in separator columns"

        objects = _sorted_objects(panel, bg)
        hits = [
            obj
            for obj in objects
            if int(obj["bbox"][0]) <= dial_row <= int(obj["bbox"][2])
        ]
        assert len(hits) == 1, "dial row must hit exactly one row span"

        for r, c in hits[0]["coords"]:
            rr, cc = int(r), int(c)
            assert out[rr, cc] == bg, "selected objects from different panels must not overlap"
            out[rr, cc] = int(dial_color)

    return out


def _bounded_composition(total: int, parts: int, lo: int, hi: int) -> list[int] | None:
    if not (parts * lo <= total <= parts * hi):
        return None
    values = []
    remaining = total
    for idx in range(parts):
        slots = parts - idx - 1
        cur_lo = max(lo, remaining - slots * hi)
        cur_hi = min(hi, remaining - slots * lo)
        values.append(random.randint(cur_lo, cur_hi))
        remaining -= values[-1]
    random.shuffle(values)
    return values


def _nonnegative_composition(total: int, parts: int) -> list[int]:
    values = [0] * parts
    for _ in range(total):
        values[random.randrange(parts)] += 1
    return values


def _sample_spans(height: int, count: int) -> list[tuple[int, int]] | None:
    min_rows = OBJECT_HEIGHT_RANGE[0] * count
    max_rows = min(OBJECT_HEIGHT_RANGE[1] * count, height - (count - 1))
    if max_rows < min_rows:
        return None

    total_rows = random.randint(min_rows, max_rows)
    obj_heights = _bounded_composition(
        total_rows,
        count,
        OBJECT_HEIGHT_RANGE[0],
        OBJECT_HEIGHT_RANGE[1],
    )
    if obj_heights is None:
        return None

    slack = height - total_rows - (count - 1)
    extras = _nonnegative_composition(slack, count + 1)

    spans = []
    row = extras[0]
    for idx, obj_h in enumerate(obj_heights):
        spans.append((row, row + obj_h - 1))
        row += obj_h
        if idx < count - 1:
            row += 1 + extras[idx + 1]
    assert row + extras[-1] == height
    return spans


def _mask_coords(mask: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in np.argwhere(mask)]


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _expanded_coords(coords: list[tuple[int, int]], height: int, width: int) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < height and 0 <= cc < width:
                    expanded.add((rr, cc))
    return expanded


def _sample_object_mask(
    height: int,
    panel_w: int,
    span: tuple[int, int],
    *,
    forbidden: set[tuple[int, int]] | None = None,
) -> np.ndarray | None:
    r0, r1 = span
    max_shape_w = min(panel_w, 4)
    min_shape_w = 2

    for _ in range(MASK_RETRIES):
        bbox_w = random.randint(min_shape_w, max_shape_w)
        left = random.randint(0, panel_w - bbox_w)
        right = left + bbox_w - 1

        mask = np.zeros((height, panel_w), dtype=bool)
        seg_start = random.randint(left, right)
        seg_end = random.randint(seg_start, min(right, seg_start + random.randint(0, 2)))

        for row in range(r0, r1 + 1):
            if row > r0:
                overlap = random.randint(seg_start, seg_end)
                seg_start = random.randint(max(left, overlap - 2), overlap)
                seg_end = random.randint(overlap, min(right, overlap + 2))
            mask[row, seg_start : seg_end + 1] = True

            if bbox_w >= 3 and random.random() < 0.35:
                if seg_start > left and random.random() < 0.5:
                    mask[row, seg_start - 1] = True
                elif seg_end < right:
                    mask[row, seg_end + 1] = True

        coords = _mask_coords(mask)
        if forbidden is not None and any(coord in forbidden for coord in coords):
            continue
        return mask
    return None


def _gen_panel(
    height: int,
    panel_w: int,
    selected_forbidden: set[tuple[int, int]],
) -> dict | None:
    for _ in range(PANEL_RETRIES):
        obj_count = random.randint(*OBJECT_COUNT_RANGE)
        spans = _sample_spans(height, obj_count)
        if spans is None:
            continue

        selected_idx = random.randrange(obj_count)
        obj_colors = random.sample(PALETTE, obj_count)
        objects = []
        ok = True

        for idx, span in enumerate(spans):
            mask = _sample_object_mask(
                height,
                panel_w,
                span,
                forbidden=selected_forbidden if idx == selected_idx else None,
            )
            if mask is None:
                ok = False
                break
            coords = _mask_coords(mask)
            objects.append(
                {
                    "mask": mask,
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "color": int(obj_colors[idx]),
                }
            )
        if not ok:
            continue

        panel = np.full((height, panel_w), BG, dtype=int)
        for obj in objects:
            panel[obj["mask"]] = obj["color"]

        selected = objects[selected_idx]
        dial_row = random.randint(selected["bbox"][0], selected["bbox"][2])
        dial_color = random.choice([color for color in PALETTE if color != selected["color"]])
        return {
            "panel": panel,
            "objects": objects,
            "selected_idx": selected_idx,
            "selected_coords": selected["coords"],
            "dial_row": dial_row,
            "dial_color": int(dial_color),
        }
    return None


def _render_input(panels: list[dict]) -> np.ndarray:
    height, panel_w = panels[0]["panel"].shape
    grid = np.full((height, len(panels) * (panel_w + 1)), BG, dtype=int)
    for idx, spec in enumerate(panels):
        sep_col = idx * (panel_w + 1)
        spec["sep_col"] = sep_col
        grid[:, sep_col] = SEP
        grid[spec["dial_row"], sep_col] = spec["dial_color"]
        paste(grid, spec["panel"], 0, sep_col + 1)
    return grid


def _expected_output(panels: list[dict]) -> np.ndarray | None:
    height, panel_w = panels[0]["panel"].shape
    out = np.full((height, panel_w), BG, dtype=int)
    for spec in panels:
        for r, c in spec["selected_coords"]:
            if out[r, c] != BG:
                return None
            out[r, c] = spec["dial_color"]
    return out


def _semantic_checks(grid: np.ndarray, expected: np.ndarray, panels: list[dict]) -> bool:
    try:
        actual = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False

    checked_color = False
    checked_switch = False
    for spec in panels:
        if not checked_color:
            alt_color = next(color for color in PALETTE if color != spec["dial_color"])
            alt_grid = grid.copy()
            alt_grid[spec["dial_row"], spec["sep_col"]] = alt_color
            try:
                alt_out = main(alt_grid)
            except AssertionError:
                return False
            if np.array_equal(alt_out, expected):
                return False
            if not np.array_equal(alt_out != BG, expected != BG):
                return False
            checked_color = True

        alternatives = [
            obj for idx, obj in enumerate(spec["objects"]) if idx != spec["selected_idx"]
        ]
        if alternatives:
            other = random.choice(alternatives)
            alt_row = random.randint(other["bbox"][0], other["bbox"][2])
            alt_grid = grid.copy()
            alt_grid[spec["dial_row"], spec["sep_col"]] = SEP
            alt_grid[alt_row, spec["sep_col"]] = spec["dial_color"]
            try:
                alt_out = main(alt_grid)
            except AssertionError:
                return False
            if np.array_equal(alt_out, expected):
                return False
            checked_switch = True

    return checked_color and checked_switch


def _gen_attempt() -> np.ndarray | None:
    panel_count, panel_w = random.choice(LAYOUTS)
    height = random.randint(*HEIGHT_RANGE)

    panels: list[dict] = []
    selected_forbidden: set[tuple[int, int]] = set()
    selected_indices = []
    object_counts = []

    for _ in range(panel_count):
        spec = _gen_panel(height, panel_w, selected_forbidden)
        if spec is None:
            return None
        panels.append(spec)
        selected_indices.append(int(spec["selected_idx"]))
        object_counts.append(len(spec["objects"]))
        selected_forbidden |= _expanded_coords(spec["selected_coords"], height, panel_w)

    if len(set(selected_indices)) < 2:
        return None
    if max(object_counts) < 3:
        return None
    if len({spec["dial_color"] for spec in panels}) < 2:
        return None

    expected = _expected_output(panels)
    if expected is None:
        return None

    grid = _render_input(panels)
    if not _semantic_checks(grid, expected, panels):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for separator-dial selection")
