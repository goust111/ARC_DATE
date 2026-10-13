from __future__ import annotations

import random
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, get_bg, paste


FG_COLORS = [Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE]
MARKER_COLOR = Color.BLUE
PATH_COLOR = Color.TEAL
BG_COLOR = Color.BLACK
GEN_RETRIES = 400


def _segment_pixels(grid: np.ndarray) -> list[dict]:
    blue = np.argwhere(grid == MARKER_COLOR)
    by_row: dict[int, list[int]] = defaultdict(list)
    by_col: dict[int, list[int]] = defaultdict(list)
    for r, c in blue:
        by_row[int(r)].append(int(c))
        by_col[int(c)].append(int(r))

    bg = int(get_bg(grid))
    segments = []
    for r, cols in by_row.items():
        cols = sorted(cols)
        for left, right in zip(cols, cols[1:]):
            pixels = [(r, c) for c in range(left + 1, right) if int(grid[r, c]) == bg]
            if pixels:
                segments.append(
                    {
                        "orientation": "h",
                        "start": pixels[0],
                        "pixels": pixels,
                    }
                )
    for c, rows in by_col.items():
        rows = sorted(rows)
        for top, bottom in zip(rows, rows[1:]):
            pixels = [(r, c) for r in range(top + 1, bottom) if int(grid[r, c]) == bg]
            if pixels:
                segments.append(
                    {
                        "orientation": "v",
                        "start": pixels[0],
                        "pixels": pixels,
                    }
                )

    segments.sort(key=lambda seg: (seg["start"][0], seg["start"][1], seg["orientation"]))
    for idx, seg in enumerate(segments):
        seg["index"] = idx
    return segments


def _foreground_objects(grid: np.ndarray) -> list[dict]:
    bg = int(get_bg(grid))
    out = []
    for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        color = int(comp["color"])
        if color in {bg, MARKER_COLOR, PATH_COLOR}:
            continue
        out.append(comp)
    out.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["color"]))
    return out


def _pixel_touches_bbox(pixel: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = pixel
    r0, c0, r1, c1 = bbox
    return (
        (r0 <= r <= r1 and c0 <= c <= c1)
        or (r0 <= r <= r1 and c in {c0 - 1, c1 + 1})
        or (c0 <= c <= c1 and r in {r0 - 1, r1 + 1})
    )


def _analyze(grid: np.ndarray) -> dict:
    objects = _foreground_objects(grid)
    segments = _segment_pixels(grid)

    line_groups: list[dict] = []
    segment_to_objects: dict[int, list[dict]] = {seg["index"]: [] for seg in segments}
    isolated = []

    for obj in objects:
        candidates = []
        for seg in segments:
            if any(_pixel_touches_bbox(pixel, obj["bbox"]) for pixel in seg["pixels"]):
                candidates.append(seg["index"])
        obj["candidate_segments"] = tuple(candidates)
        if candidates:
            segment_to_objects[candidates[0]].append(obj)
            obj["assigned_segment"] = candidates[0]
        else:
            isolated.append(obj)
            obj["assigned_segment"] = None

    for seg in segments:
        members = segment_to_objects[seg["index"]]
        if not members:
            continue
        ordered = sorted(members, key=lambda obj: (-int(obj["area"]), obj["bbox"][0], obj["bbox"][1]))
        line_groups.append({"kind": "segment", "segment": seg, "objects": ordered})

    isolated.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
    singleton_groups = [{"kind": "singleton", "objects": [obj]} for obj in isolated]
    return {
        "segments": segments,
        "objects": objects,
        "line_groups": line_groups,
        "singleton_groups": singleton_groups,
    }


def _render_output(grid: np.ndarray, analysis: dict) -> np.ndarray:
    bg = int(get_bg(grid))
    groups = analysis["line_groups"] + analysis["singleton_groups"]
    patches_per_group = []
    heights = []
    widths = []
    for group in groups:
        patches = [crop_bbox(grid, obj["bbox"]) for obj in group["objects"]]
        stack_h = sum(p.shape[0] for p in patches) + max(0, len(patches) - 1)
        group_w = max(p.shape[1] for p in patches)
        patches_per_group.append(patches)
        heights.append(stack_h)
        widths.append(group_w)

    out_h = max(heights)
    out_w = sum(widths) + max(0, len(widths) - 1)
    out = np.full((out_h, out_w), bg, dtype=int)

    c0 = 0
    for patches, stack_h, group_w in zip(patches_per_group, heights, widths):
        r = out_h - stack_h
        for idx, patch in enumerate(patches):
            paste(out, patch, r, c0, transparent=bg)
            r += patch.shape[0]
            if idx != len(patches) - 1:
                r += 1
        c0 += group_w + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Use blue-marker corridors to group nearby objects, then sort and stack each group into bottom-aligned output columns.

    Input structure:
    - The grid is mostly black background, with 4-5 small 4-connected foreground objects in distinct non-blue colors.
    - Blue marker pixels appear in some shared rows or columns; the input itself does not contain teal corridor pixels.
    - Foreground objects do not overlap the blue markers.

    Rule:
    - For every row with at least two blue markers, sort those markers by column and draw a teal horizontal segment on the background cells between each adjacent pair; do the analogous construction for columns.
    - A foreground object is touched by a segment iff at least one teal segment pixel lies inside the object's bbox or is 4-adjacent to that bbox; if multiple segments touch an object, assign it to the earliest segment in top-to-bottom, left-to-right scan order.
    - Each nonempty segment-defined group sorts its objects by descending pixel count, breaking ties by the bbox top-left scan order; untouched objects become rightmost singleton groups.
    - Crop each object to its bbox and stack each group's objects top-to-bottom with one blank row between them, while bottom-aligning every group and placing groups left-to-right with one blank column between groups.

    Key insight:
    - The blue markers are not themselves the grouping labels: their row/column alignments create invisible corridor segments whose geometry determines grouping, and only after that grouping do size-based sorting and compact stacking apply.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    analysis = _analyze(g)
    groups = analysis["line_groups"] + analysis["singleton_groups"]
    assert groups, "expected at least one foreground object"
    return _render_output(g, analysis)


def _mask_is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    seen = {tuple(map(int, coords[0]))}
    stack = [tuple(map(int, coords[0]))]
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and bool(mask[nr, nc]):
                cell = (nr, nc)
                if cell not in seen:
                    seen.add(cell)
                    stack.append(cell)
    return len(seen) == len(coords)


@lru_cache(None)
def _shape_catalog(h: int, w: int, area: int) -> tuple[np.ndarray, ...]:
    masks = []
    n = h * w
    for bits in range(1 << n):
        if bits.bit_count() != area:
            continue
        mask = np.array([(bits >> i) & 1 for i in range(n)], dtype=bool).reshape(h, w)
        if not _mask_is_connected(mask):
            continue
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        masks.append(mask)
    return tuple(masks)


def _grow_patch(h: int, w: int, area: int, color: int) -> np.ndarray:
    catalog = _shape_catalog(h, w, area)
    if not catalog:
        raise RuntimeError("failed to grow patch")
    mask = random.choice(catalog)
    patch = np.full((h, w), BG_COLOR, dtype=int)
    patch[mask] = int(color)
    return patch


def _sample_dims(area: int) -> tuple[int, int]:
    choices = []
    for h in range(2, 4):
        for w in range(2, 5):
            if h * w >= area and _shape_catalog(h, w, area):
                choices.append((h, w))
    return random.choice(choices)


def _make_group_specs(group_sizes: list[int], colors: list[int]) -> tuple[list[dict], bool]:
    specs = []
    needs_sorting = False
    color_iter = iter(colors)
    for idx, size in enumerate(group_sizes):
        objects = []
        if size == 2 and not needs_sorting:
            small_area = random.randint(3, 4)
            large_area = random.randint(small_area + 1, 8)
            sh, sw = _sample_dims(small_area)
            lh, lw = _sample_dims(large_area)
            large_x = random.randint(0, 2)
            small_x = large_x + random.randint(0, 1)
            objects.append(
                {
                    "bbox": (-sh, small_x, -1, small_x + sw - 1),
                    "patch": _grow_patch(sh, sw, small_area, next(color_iter)),
                    "area": small_area,
                }
            )
            objects.append(
                {
                    "bbox": (1, large_x, lh, large_x + lw - 1),
                    "patch": _grow_patch(lh, lw, large_area, next(color_iter)),
                    "area": large_area,
                }
            )
            needs_sorting = True
        else:
            cursor = 0
            side_pattern = random.choice(["same", "split"])
            common_side = random.choice([-1, 1])
            for j in range(size):
                area = random.randint(3, 8)
                h, w = _sample_dims(area)
                x = cursor + random.randint(0, 1)
                if side_pattern == "split":
                    side = -1 if j % 2 == 0 else 1
                else:
                    side = common_side
                r0 = -h if side < 0 else 1
                r1 = -1 if side < 0 else h
                objects.append(
                    {
                        "bbox": (r0, x, r1, x + w - 1),
                        "patch": _grow_patch(h, w, area, next(color_iter)),
                        "area": area,
                    }
                )
                cursor = x + w + random.randint(1, 2)

        min_c = min(obj["bbox"][1] for obj in objects)
        max_c = max(obj["bbox"][3] for obj in objects)
        left_margin = random.randint(1, 2)
        right_margin = random.randint(1, 2)
        specs.append(
            {
                "objects": objects,
                "left_marker": min_c - left_margin,
                "right_marker": max_c + right_margin,
                "sort_anchor": idx,
            }
        )
    return specs, needs_sorting


def _sample_rows(h: int, count: int) -> list[int] | None:
    rows = []
    candidates = list(range(5, h - 5))
    random.shuffle(candidates)
    for r in candidates:
        if all(abs(r - prev) >= 7 for prev in rows):
            rows.append(r)
            if len(rows) == count:
                return sorted(rows)
    return None


def _place_specs_on_grid(h: int, w: int, specs: list[dict]) -> tuple[np.ndarray, dict] | None:
    rows = _sample_rows(h, len(specs))
    if rows is None:
        return None

    grid = np.full((h, w), BG_COLOR, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    marker_rows = set()
    marker_cols = set()
    plan = {"groups": [], "isolated_colors": []}

    for row, spec in zip(rows, specs):
        min_abs_shift = -spec["left_marker"]
        max_abs_shift = w - 1 - spec["right_marker"]
        if min_abs_shift > max_abs_shift:
            return None
        shift = random.randint(min_abs_shift, max_abs_shift)
        left_marker = spec["left_marker"] + shift
        right_marker = spec["right_marker"] + shift
        marker_rows.add(row)
        marker_cols.update({left_marker, right_marker})
        if occupied[row, left_marker] or occupied[row, right_marker]:
            return None
        grid[row, left_marker] = MARKER_COLOR
        grid[row, right_marker] = MARKER_COLOR
        occupied[row, left_marker] = True
        occupied[row, right_marker] = True

        group_colors = []
        for obj in spec["objects"]:
            dr0, dc0, dr1, dc1 = obj["bbox"]
            patch = obj["patch"]
            r0 = row + dr0
            c0 = shift + dc0
            if r0 < 0 or c0 < 0 or r0 + patch.shape[0] > h or c0 + patch.shape[1] > w:
                return None
            mask = patch != BG_COLOR
            if np.any(occupied[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] & mask):
                return None
            paste(grid, patch, r0, c0, transparent=BG_COLOR)
            occupied[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] |= mask
            group_colors.append(int(patch[mask][0]))
        plan["groups"].append({"row": row, "colors": tuple(group_colors)})

    return grid, {"occupied": occupied, "marker_rows": marker_rows, "marker_cols": marker_cols, "plan": plan}


def _place_isolated(grid: np.ndarray, state: dict, color: int) -> bool:
    h, w = grid.shape
    occupied = state["occupied"]
    analysis = _analyze(grid)
    rows_to_avoid = {seg["start"][0] for seg in analysis["segments"]}

    area = random.randint(3, 8)
    ph, pw = _sample_dims(area)
    patch = _grow_patch(ph, pw, area, color)
    for _ in range(300):
        r0 = random.randint(0, h - ph)
        c0 = random.randint(0, w - pw)
        mask = patch != BG_COLOR
        if np.any(occupied[r0 : r0 + ph, c0 : c0 + pw] & mask):
            continue
        bbox = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        if any(seg_r in range(bbox[0] - 1, bbox[2] + 2) for seg_r in rows_to_avoid):
            continue
        paste(grid, patch, r0, c0, transparent=BG_COLOR)
        occupied[r0 : r0 + ph, c0 : c0 + pw] |= mask
        state["plan"]["isolated_colors"].append(int(color))
        return True
    return False


def _add_dummy_markers(grid: np.ndarray, state: dict, count: int) -> bool:
    h, w = grid.shape
    used_rows = set(state["marker_rows"])
    used_cols = set(state["marker_cols"])
    for _ in range(count):
        for _attempt in range(300):
            r = random.randrange(h)
            c = random.randrange(w)
            if r in used_rows or c in used_cols:
                continue
            if int(grid[r, c]) != BG_COLOR:
                continue
            grid[r, c] = MARKER_COLOR
            used_rows.add(r)
            used_cols.add(c)
            break
        else:
            return False
    return True


def _validate_instance(grid: np.ndarray) -> bool:
    if not (isinstance(grid, np.ndarray) and grid.ndim == 2 and max(grid.shape) <= 30):
        return False

    analysis = _analyze(grid)
    objects = analysis["objects"]
    segments = analysis["segments"]
    line_groups = analysis["line_groups"]
    singleton_groups = analysis["singleton_groups"]

    if len(objects) not in {4, 5}:
        return False
    if not (6 <= int(np.sum(grid == MARKER_COLOR)) <= 10):
        return False
    colors = [int(obj["color"]) for obj in objects]
    if len(colors) != len(set(colors)):
        return False
    if any(color not in FG_COLORS for color in colors):
        return False
    if len(line_groups) < 2:
        return False
    if sum(len(group["objects"]) for group in line_groups) < 3:
        return False
    if len(singleton_groups) < 1:
        return False
    if not any(len(group["objects"]) >= 2 for group in line_groups):
        return False
    if any(len(obj["candidate_segments"]) > 1 for obj in objects):
        return False

    sorting_changes_order = False
    for group in line_groups:
        if len(group["objects"]) < 2:
            continue
        scan_order = sorted(group["objects"], key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
        solved_order = group["objects"]
        if [obj["color"] for obj in scan_order] != [obj["color"] for obj in solved_order]:
            sorting_changes_order = True
            break
    if not sorting_changes_order:
        return False

    out = main(grid)
    oracle = _render_output(grid, analysis)
    if not np.array_equal(out, oracle):
        return False
    if max(out.shape) > 30:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    total_objects, group_sizes = random.choice(
        [
            (4, [2, 1]),
            (5, [2, 2]),
            (5, [2, 1, 1]),
        ]
    )
    h = random.randint(21, 26)
    w = random.randint(18, 24)
    colors = random.sample(FG_COLORS, total_objects)
    group_specs, needs_sorting = _make_group_specs(group_sizes, colors[:-1])
    if not needs_sorting:
        return None

    placed = _place_specs_on_grid(h, w, group_specs)
    if placed is None:
        return None
    grid, state = placed
    if not _place_isolated(grid, state, colors[-1]):
        return None

    active_markers = 2 * len(group_sizes)
    target_markers = random.choice([6, 8]) if active_markers <= 6 else random.choice([6, 8, 10])
    dummy_count = max(0, target_markers - active_markers)
    if not _add_dummy_markers(grid, state, dummy_count):
        return None

    if random.random() < 0.5:
        grid = grid.T.copy()
    return grid if _validate_instance(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
