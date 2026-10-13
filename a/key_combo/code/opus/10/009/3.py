from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_TOOLBOX_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_TOOLBOX_DIR) not in sys.path:
    sys.path.append(str(_TOOLBOX_DIR))

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste


BG = 8
BLACK = 0
SEP = 5
LEGEND_SIZE = 5
LEGEND_INNER = 3
MAX_RETRIES = 400


def _find_separator_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == SEP)]
    assert len(cols) == 1, f"expected one full separator column, got {cols}"
    return int(cols[0])


def _legend_candidates(grid: np.ndarray) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    candidates = []
    for comp in components_mask(grid == BLACK, connectivity=4):
        bbox = bbox_coords(comp)
        r0, c0, r1, c1 = bbox
        if r1 - r0 != 4 or c1 - c0 != 4:
            continue
        patch = crop_bbox(grid, bbox)
        if not np.all(patch[0, :] == BLACK):
            continue
        if not np.all(patch[-1, :] == BLACK):
            continue
        if not np.all(patch[:, 0] == BLACK):
            continue
        if not np.all(patch[:, -1] == BLACK):
            continue
        inner = patch[1:4, 1:4]
        vals = [int(v) for v in inner.flat]
        if BLACK in vals:
            continue
        if len(set(vals)) != 9:
            continue
        candidates.append((bbox, inner.copy()))
    return candidates


def _find_legend(grid: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    candidates = _legend_candidates(grid)
    assert len(candidates) == 1, f"expected one legend frame, got {len(candidates)}"
    return candidates[0]


def _shape_components(grid: np.ndarray, sep_col: int, legend_bbox: tuple[int, int, int, int]) -> list[np.ndarray]:
    mask = grid == BLACK
    r0, c0, r1, c1 = legend_bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = False
    mask[:, sep_col:] = False
    comps = components_mask(mask, connectivity=4)
    assert comps, "expected at least one movable black shape"
    return comps


def _translate_component(coords: np.ndarray, delta_c: int) -> np.ndarray:
    moved = coords.copy()
    moved[:, 1] += int(delta_c)
    return moved


def _find_holes_from_shape_coords(shape_coords: np.ndarray, sep_col: int) -> list[np.ndarray]:
    bbox = bbox_coords(shape_coords)
    r0, c0, r1, _c1 = bbox
    local_h = int(r1 - r0 + 1)
    local_w = int(sep_col - c0 + 1)
    local = np.full((local_h, local_w), BG, dtype=int)
    local[:, -1] = SEP
    local[shape_coords[:, 0] - r0, shape_coords[:, 1] - c0] = BLACK

    holes = []
    for comp in components_mask(local == BG, connectivity=4):
        if any(
            rr == 0 or rr == local_h - 1 or cc == 0 or cc == local_w - 1
            for rr, cc in comp
        ):
            continue
        if not any(cc == local_w - 2 for _rr, cc in comp):
            continue
        hole = comp.copy()
        hole[:, 0] += r0
        hole[:, 1] += c0
        holes.append(hole)
    return holes


def _bucket_for_hole(hole_coords: np.ndarray, shape_bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = shape_bbox
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    center_r = int(np.floor(hole_coords[:, 0].mean()))
    center_c = int(np.floor(hole_coords[:, 1].mean()))
    rr = min(2, (center_r - int(r0)) * 3 // height)
    cc = min(2, (center_c - int(c0)) * 3 // width)
    return int(rr), int(cc)


def _solve(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert int(get_bg(g)) == BG

    sep_col = _find_separator_col(g)
    legend_bbox, legend = _find_legend(g)
    shape_components = _shape_components(g, sep_col, legend_bbox)

    out = g.copy()
    for coords in shape_components:
        out[coords[:, 0], coords[:, 1]] = BG

    moved_shapes = []
    for coords in shape_components:
        right = int(coords[:, 1].max())
        delta = int(sep_col - 1 - right)
        assert delta >= 0
        moved = _translate_component(coords, delta)
        out[moved[:, 0], moved[:, 1]] = BLACK
        moved_shapes.append(moved)

    hole_details = []
    for moved in moved_shapes:
        shape_bbox = bbox_coords(moved)
        holes = _find_holes_from_shape_coords(moved, sep_col)
        for hole in holes:
            bucket = _bucket_for_hole(hole, shape_bbox)
            color = int(legend[bucket[0], bucket[1]])
            out[hole[:, 0], hole[:, 1]] = color
            hole_details.append(
                {
                    "shape_bbox": shape_bbox,
                    "hole": hole,
                    "bucket": bucket,
                    "color": color,
                }
            )
    return out, hole_details


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide left black shells onto the grey separator and color the holes they create from a positional legend.

    Input structure:
    - The background is teal (8), with a single full-height grey (5) separator column near the right side.
    - A 5x5 black frame containing a 3x3 array of nine distinct colors acts as the legend, placed in the top-left or bottom-right corner.
    - On the left of the separator there are several 4-connected black shapes whose right-open cavities are not enclosed before sliding.

    Rule:
    - Ignore the black legend frame, but move every other black connected component horizontally right until its rightmost black pixel is flush with the separator's left side.
    - After sliding, find teal regions that are now enclosed by the moved black shape together with the grey separator; these are the holes.
    - For each hole, compute the floored centroid and map it into a 3x3 position inside that shape's bounding box, then fill the whole hole with the legend color from the same 3x3 position.

    Key insight:
    - The separator is not just a target wall for translation: it is the missing wall that turns each right-open cavity into a real hole, and that hole's relative position then indexes the legend.
    """
    out, _ = _solve(grid)
    return out


def _random_gaps(extra: int, n_slots: int) -> list[int]:
    vals = [0] * n_slots
    for _ in range(extra):
        vals[random.randrange(n_slots)] += 1
    return vals


def _bucket_from_rel(rel_hole: np.ndarray, patch_shape: tuple[int, int]) -> tuple[int, int]:
    h, w = patch_shape
    center_r = int(np.floor(rel_hole[:, 0].mean()))
    center_c = int(np.floor(rel_hole[:, 1].mean()))
    rr = min(2, center_r * 3 // h)
    cc = min(2, center_c * 3 // w)
    return int(rr), int(cc)


def _build_shape_patch(n_holes: int) -> tuple[np.ndarray, list[np.ndarray], list[tuple[int, int]]]:
    for _ in range(200):
        if n_holes == 1:
            chamber_heights = [random.randint(1, 3)]
            width = random.randint(3, 6)
        elif n_holes == 2:
            chamber_heights = [random.randint(1, 2), random.randint(1, 2)]
            width = random.randint(3, 6)
        else:
            chamber_heights = [1, 1, 1]
            width = random.randint(3, 5)

        height = sum(chamber_heights) + n_holes + 1
        patch = np.full((height, width), BG, dtype=int)
        patch[0, :] = BLACK
        patch[-1, :] = BLACK

        holes_rel: list[np.ndarray] = []
        cursor = 1
        for hole_idx, chamber_h in enumerate(chamber_heights):
            hole_width = random.randint(1, min(width - 1, 4))
            hole_left = width - hole_width
            if hole_idx > 0 and random.random() < 0.45 and hole_left > 1:
                hole_left -= 1
            coords = []
            for rr in range(cursor, cursor + chamber_h):
                patch[rr, :hole_left] = BLACK
                for cc in range(hole_left, width):
                    coords.append((rr, cc))
            holes_rel.append(np.array(coords, dtype=int))
            cursor += chamber_h
            if hole_idx < n_holes - 1:
                patch[cursor, :] = BLACK
                cursor += 1

        buckets = [_bucket_from_rel(hole, patch.shape) for hole in holes_rel]
        if len(set(buckets)) != len(buckets):
            continue

        black_area = int(np.sum(patch == BLACK))
        if black_area < 8 or black_area > 24:
            continue

        return patch, holes_rel, buckets
    raise AssertionError("failed to build a non-degenerate shape patch")


def _choose_hole_counts(n_shapes: int) -> list[int]:
    for _ in range(200):
        counts = []
        total = 0
        for i in range(n_shapes):
            remaining_shapes = n_shapes - i - 1
            max_for_this = min(3, 6 - total - remaining_shapes)
            choices = [1]
            if max_for_this >= 2:
                choices += [2, 2]
            if max_for_this >= 3:
                choices += [3]
            choice = random.choice(choices)
            counts.append(choice)
            total += choice
        if 2 <= total <= 6:
            return counts
    raise AssertionError("failed to choose valid hole counts")


def _place_rows(height: int, legend_top_left: bool, shape_heights: list[int]) -> list[int] | None:
    top_margin = LEGEND_SIZE + 1 if legend_top_left else 1
    bottom_margin = 1
    min_gap = 1
    usable = height - top_margin - bottom_margin
    need = sum(shape_heights) + min_gap * (len(shape_heights) - 1)
    if usable < need:
        return None
    extra = usable - need
    gaps = _random_gaps(extra, len(shape_heights) + 1)
    rows = []
    r = top_margin + gaps[0]
    for idx, sh in enumerate(shape_heights):
        rows.append(r)
        r += sh
        if idx < len(shape_heights) - 1:
            r += min_gap + gaps[idx + 1]
    return rows


def _legend_patch(used_positions: set[tuple[int, int]]) -> np.ndarray | None:
    all_positions = [(r, c) for r in range(3) for c in range(3)]
    unused = [pos for pos in all_positions if pos not in used_positions]
    if not unused:
        return None

    patch = np.full((LEGEND_SIZE, LEGEND_SIZE), BLACK, dtype=int)
    inner = patch[1:4, 1:4]
    color_positions = {}
    pos_for_bg = random.choice(unused)
    remaining_positions = [pos for pos in all_positions if pos != pos_for_bg]
    remaining_colors = [c for c in range(1, 10) if c != BG]
    random.shuffle(remaining_positions)
    random.shuffle(remaining_colors)
    color_positions[pos_for_bg] = BG
    for pos, color in zip(remaining_positions, remaining_colors):
        color_positions[pos] = color
    for r in range(3):
        for c in range(3):
            inner[r, c] = int(color_positions[(r, c)])
    return patch


def _expected_output(
    grid: np.ndarray,
    legend_inner: np.ndarray,
    shapes: list[dict],
) -> np.ndarray:
    out = grid.copy()
    for shape in shapes:
        original = shape["original_black"]
        out[original[:, 0], original[:, 1]] = BG
    for shape in shapes:
        paste(out, shape["patch"], shape["row"], shape["flush_col"], transparent=BG)
        for hole_rel, bucket in zip(shape["holes_rel"], shape["buckets"]):
            color = int(legend_inner[bucket[0], bucket[1]])
            out[hole_rel[:, 0] + shape["row"], hole_rel[:, 1] + shape["flush_col"]] = color
    return out


def _validate_case(
    grid: np.ndarray,
    expected: np.ndarray,
    legend_bbox: tuple[int, int, int, int],
    shapes: list[dict],
) -> bool:
    if grid.ndim != 2 or expected.ndim != 2:
        return False
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    if np.array_equal(grid, expected):
        return False
    if int(get_bg(grid)) != BG:
        return False
    try:
        found_legend_bbox, found_legend_inner = _find_legend(grid)
        sep_col = _find_separator_col(grid)
    except Exception:
        return False
    if found_legend_bbox != legend_bbox:
        return False

    used_colors = {
        int(found_legend_inner[bucket[0], bucket[1]])
        for shape in shapes
        for bucket in shape["buckets"]
    }
    if BG in used_colors:
        return False

    comps = _shape_components(grid, sep_col, legend_bbox)
    if len(comps) != len(shapes):
        return False

    hole_counts_before = []
    hole_counts_after = []
    detected_buckets = []
    for shape in shapes:
        before = _find_holes_from_shape_coords(shape["original_black"], sep_col)
        after = _find_holes_from_shape_coords(shape["flush_black"], sep_col)
        hole_counts_before.append(len(before))
        hole_counts_after.append(len(after))
        detected_buckets.append(
            sorted(_bucket_for_hole(hole, bbox_coords(shape["flush_black"])) for hole in after)
        )
    if any(count != 0 for count in hole_counts_before):
        return False
    if hole_counts_after != [len(shape["holes_rel"]) for shape in shapes]:
        return False
    if detected_buckets != [sorted(shape["buckets"]) for shape in shapes]:
        return False

    try:
        actual, details = _solve(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False
    if len(details) != sum(len(shape["holes_rel"]) for shape in shapes):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_shapes = random.randint(2, 4)
    hole_counts = _choose_hole_counts(n_shapes)

    shape_specs = []
    used_positions: set[tuple[int, int]] = set()
    for n_holes in hole_counts:
        patch, holes_rel, buckets = _build_shape_patch(n_holes)
        for bucket in buckets:
            used_positions.add(bucket)
        shape_specs.append(
            {
                "patch": patch,
                "holes_rel": holes_rel,
                "buckets": buckets,
                "height": patch.shape[0],
                "width": patch.shape[1],
            }
        )

    legend_patch = _legend_patch(used_positions)
    if legend_patch is None:
        return None
    legend_inner = legend_patch[1:4, 1:4].copy()

    sep_col = random.randint(12, 16)
    width = random.randint(max(21, sep_col + 6), 27)
    legend_top_left = random.random() < 0.5

    min_height = max(16, sum(shape["height"] for shape in shape_specs) + len(shape_specs) + 2)
    if legend_top_left:
        min_height = max(min_height, LEGEND_SIZE + 2 + sum(shape["height"] for shape in shape_specs))
    if min_height > 27:
        return None
    height = random.randint(min_height, 27)

    rows = _place_rows(height, legend_top_left, [shape["height"] for shape in shape_specs])
    if rows is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    grid[:, sep_col] = SEP

    if legend_top_left:
        legend_r0, legend_c0 = 0, 0
    else:
        legend_r0, legend_c0 = height - LEGEND_SIZE, width - LEGEND_SIZE
        if legend_c0 <= sep_col:
            return None
    paste(grid, legend_patch, legend_r0, legend_c0)
    legend_bbox = (legend_r0, legend_c0, legend_r0 + 4, legend_c0 + 4)

    for row, shape in zip(rows, shape_specs):
        flush_col = sep_col - shape["width"]
        if flush_col < 1:
            return None
        shift = random.randint(1, flush_col)
        input_col = flush_col - shift
        paste(grid, shape["patch"], row, input_col, transparent=BG)

        rel_black = np.argwhere(shape["patch"] == BLACK)
        shape["row"] = row
        shape["input_col"] = input_col
        shape["flush_col"] = flush_col
        shape["original_black"] = rel_black + np.array([row, input_col], dtype=int)
        shape["flush_black"] = rel_black + np.array([row, flush_col], dtype=int)

    expected = _expected_output(grid, legend_inner, shape_specs)
    if not _validate_case(grid, expected, legend_bbox, shape_specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
