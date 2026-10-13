from __future__ import annotations

import random
import sys
from typing import Any

import numpy as np

_ARC_UTILS_DIR = "/home/lixinyu/snap/new/solvers"
if _ARC_UTILS_DIR not in sys.path:
    sys.path.append(_ARC_UTILS_DIR)

from arc_utils import Color, bbox_coords, components_mask, crop_bbox, get_bg


BG_COLOR = Color.BLACK
ANCHOR_COLOR = Color.GREY
ALL_PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GEN_RETRIES = 400

RAY_SPECS = (
    ((0, 0), (-1, -1)),
    ((0, 1), (-1, 1)),
    ((1, 0), (1, -1)),
    ((1, 1), (1, 1)),
)


def _non_bg_colors(grid: np.ndarray, bg: int) -> list[int]:
    return sorted(set(map(int, np.unique(grid))) - {int(bg)})


def _stripe_candidates(grid: np.ndarray, bg: int) -> list[tuple[str, int, int, int]]:
    h, w = grid.shape
    candidates: list[tuple[str, int, int, int]] = []

    c = 0
    while c < w:
        color = int(grid[0, c])
        if color == bg or not np.all(grid[:, c] == color):
            c += 1
            continue
        start = c
        while c + 1 < w and int(grid[0, c + 1]) == color and np.all(grid[:, c + 1] == color):
            c += 1
        candidates.append(("v", start, c, color))
        c += 1

    r = 0
    while r < h:
        color = int(grid[r, 0])
        if color == bg or not np.all(grid[r, :] == color):
            r += 1
            continue
        start = r
        while r + 1 < h and int(grid[r + 1, 0]) == color and np.all(grid[r + 1, :] == color):
            r += 1
        candidates.append(("h", start, r, color))
        r += 1

    return candidates


def _find_anchor_bbox(panel: np.ndarray) -> tuple[int, int, int, int] | None:
    anchor_boxes = []
    for coords in components_mask(panel == ANCHOR_COLOR, connectivity=4):
        if int(coords.shape[0]) != 4:
            continue
        r0, c0, r1, c1 = bbox_coords(coords)
        if (r1 - r0 + 1, c1 - c0 + 1) == (2, 2):
            anchor_boxes.append((r0, c0, r1, c1))
    if len(anchor_boxes) != 1:
        return None
    return anchor_boxes[0]


def _trace_legend(
    grid: np.ndarray,
    anchor_bbox: tuple[int, int, int, int],
    panel_bbox: tuple[int, int, int, int],
    bg: int,
) -> tuple[np.ndarray, dict[tuple[int, int], tuple[int, int]], set[tuple[int, int]]]:
    r0, c0, r1, c1 = anchor_bbox
    pr0, pc0, pr1, pc1 = panel_bbox
    starts = (
        (r0, c0),
        (r0, c1),
        (r1, c0),
        (r1, c1),
    )

    legend = np.zeros((2, 2), dtype=int)
    hits: dict[tuple[int, int], tuple[int, int]] = {}
    ray_cells: set[tuple[int, int]] = set()

    for (qr, qc), (dr, dc) in RAY_SPECS:
        sr, sc = starts[qr * 2 + qc]
        r = sr + dr
        c = sc + dc
        hit: tuple[int, int, int] | None = None
        while pr0 <= r <= pr1 and pc0 <= c <= pc1:
            ray_cells.add((r, c))
            value = int(grid[r, c])
            if hit is None and value not in {bg, ANCHOR_COLOR}:
                hit = (r, c, value)
            r += dr
            c += dc
        if hit is None:
            raise AssertionError("each anchor corner ray must hit a colored singleton")
        legend[qr, qc] = hit[2]
        hits[(qr, qc)] = (hit[0], hit[1])

    return legend, hits, ray_cells


def _recolor_data_panel(data_panel: np.ndarray, data_color: int, legend: np.ndarray) -> np.ndarray:
    out = data_panel.copy()
    h, w = out.shape
    half_h, half_w = h // 2, w // 2
    for qr in range(2):
        for qc in range(2):
            rs = slice(qr * half_h, (qr + 1) * half_h)
            cs = slice(qc * half_w, (qc + 1) * half_w)
            block = out[rs, cs]
            block[block == data_color] = int(legend[qr, qc])
    return out


def _analyze_grid(grid: np.ndarray) -> dict[str, Any]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = int(get_bg(g))

    matches = []
    for orientation, start, end, divider_color in _stripe_candidates(g, bg):
        if orientation == "v":
            panel_a = g[:, :start]
            panel_b = g[:, end + 1 :]
            offset_a = (0, 0)
            offset_b = (0, end + 1)
        else:
            panel_a = g[:start, :]
            panel_b = g[end + 1 :, :]
            offset_a = (0, 0)
            offset_b = (end + 1, 0)

        if 0 in panel_a.shape or 0 in panel_b.shape:
            continue

        anchor_a = _find_anchor_bbox(panel_a)
        anchor_b = _find_anchor_bbox(panel_b)
        if (anchor_a is None) == (anchor_b is None):
            continue

        if anchor_a is not None:
            anchor_local = anchor_a
            anchor_offset = offset_a
            data_panel = panel_b
            data_offset = offset_b
        else:
            anchor_local = anchor_b
            anchor_offset = offset_b
            data_panel = panel_a
            data_offset = offset_a
            anchor_panel = panel_b

        data_colors = _non_bg_colors(data_panel, bg)
        if len(data_colors) != 1 or data_colors[0] == ANCHOR_COLOR:
            continue
        if data_panel.shape[0] % 2 or data_panel.shape[1] % 2:
            continue

        ar0, ac0, ar1, ac1 = anchor_local
        anchor_bbox = (
            anchor_offset[0] + ar0,
            anchor_offset[1] + ac0,
            anchor_offset[0] + ar1,
            anchor_offset[1] + ac1,
        )
        if anchor_a is not None:
            anchor_panel = panel_a
        anchor_panel_bbox = (
            anchor_offset[0],
            anchor_offset[1],
            anchor_offset[0] + anchor_panel.shape[0] - 1,
            anchor_offset[1] + anchor_panel.shape[1] - 1,
        )
        data_bbox = (
            data_offset[0],
            data_offset[1],
            data_offset[0] + data_panel.shape[0] - 1,
            data_offset[1] + data_panel.shape[1] - 1,
        )

        try:
            legend, ray_hits, ray_cells = _trace_legend(g, anchor_bbox, anchor_panel_bbox, bg)
        except AssertionError:
            continue

        if not all(
            anchor_panel_bbox[0] <= r <= anchor_panel_bbox[2] and anchor_panel_bbox[1] <= c <= anchor_panel_bbox[3]
            for r, c in ray_hits.values()
        ):
            continue

        matches.append(
            {
                "bg": bg,
                "orientation": orientation,
                "divider": (start, end, divider_color),
                "anchor_bbox": anchor_bbox,
                "data_bbox": data_bbox,
                "data_panel": crop_bbox(g, data_bbox),
                "data_color": data_colors[0],
                "legend": legend,
                "ray_hits": ray_hits,
                "ray_cells": ray_cells,
            }
        )

    assert len(matches) == 1, "expected exactly one valid divider / anchor / data interpretation"
    return matches[0]


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden 2x2 diagonal legend from the gray anchor and recolor the opposite data field by matching quadrants.

    Input structure:
    - A solid gray 2x2 anchor square sits on one side of a full-height or full-width divider bar.
    - Around that anchor are colored singleton pixels; only the four singletons lying on the anchor corners' outward diagonals are meaningful.
    - The opposite side of the divider is an even-sized rectangular data field containing background plus one data color.

    Rule:
    - From each anchor corner, walk outward along its diagonal direction and take the first colored pixel hit as that corner's hidden legend color.
    - Split the divider-separated data field into four equal quadrants.
    - Recolor every data-colored pixel by the legend color from the matching top/bottom and left/right quadrant, keep background unchanged, and output only the cropped data field.

    Key insight:
    - The visible singletons are a decoy cloud; only ray-casting from the gray 2x2 anchor reveals the invisible 2x2 legend that must then be applied to the data field's own quadrant frame.
    """
    info = _analyze_grid(np.asarray(input_grid, dtype=int))
    return _recolor_data_panel(info["data_panel"], info["data_color"], info["legend"])


def _choose_case_dimensions(orientation: str) -> tuple[int, int, int, int]:
    if orientation == "v":
        height = random.choice(range(8, 23, 2))
        data_w = random.choice(range(6, 15, 2))
        max_anchor_w = min(14, 30 - data_w - 1)
        anchor_w = random.randint(7, max_anchor_w)
        max_divider_t = min(3, 30 - data_w - anchor_w)
        divider_t = random.randint(1, max_divider_t)
        return height, anchor_w, divider_t, data_w

    width = random.choice(range(8, 23, 2))
    data_h = random.choice(range(6, 15, 2))
    max_anchor_h = min(14, 30 - data_h - 1)
    anchor_h = random.randint(7, max_anchor_h)
    max_divider_t = min(3, 30 - data_h - anchor_h)
    divider_t = random.randint(1, max_divider_t)
    return width, anchor_h, divider_t, data_h


def _anchor_ray_limits(panel_shape: tuple[int, int], anchor_bbox: tuple[int, int, int, int]) -> dict[tuple[int, int], int]:
    h, w = panel_shape
    r0, c0, r1, c1 = anchor_bbox
    return {
        (0, 0): min(r0, c0),
        (0, 1): min(r0, w - 1 - c1),
        (1, 0): min(h - 1 - r1, c0),
        (1, 1): min(h - 1 - r1, w - 1 - c1),
    }


def _all_ray_cells(panel_shape: tuple[int, int], anchor_bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    h, w = panel_shape
    r0, c0, r1, c1 = anchor_bbox
    starts = {
        (0, 0): (r0, c0),
        (0, 1): (r0, c1),
        (1, 0): (r1, c0),
        (1, 1): (r1, c1),
    }
    cells: set[tuple[int, int]] = set()
    for key, (dr, dc) in zip(starts, ((-1, -1), (-1, 1), (1, -1), (1, 1))):
        r, c = starts[key]
        r += dr
        c += dc
        while 0 <= r < h and 0 <= c < w:
            cells.add((r, c))
            r += dr
            c += dc
    return cells


def _sample_anchor_panel(panel_shape: tuple[int, int], legend: np.ndarray, divider_color: int) -> tuple[np.ndarray, dict[str, Any]] | None:
    h, w = panel_shape
    panel = np.full(panel_shape, BG_COLOR, dtype=int)

    r0 = random.randint(1, h - 4)
    c0 = random.randint(1, w - 4)
    anchor_bbox = (r0, c0, r0 + 1, c0 + 1)
    limits = _anchor_ray_limits(panel_shape, anchor_bbox)
    if min(limits.values()) < 1:
        return None

    panel[r0 : r0 + 2, c0 : c0 + 2] = ANCHOR_COLOR
    ray_cells = _all_ray_cells(panel_shape, anchor_bbox)

    ray_hits: dict[tuple[int, int], tuple[int, int]] = {}
    for qr, qc in ((0, 0), (0, 1), (1, 0), (1, 1)):
        dr = -1 if qr == 0 else 1
        dc = -1 if qc == 0 else 1
        start_r = r0 if qr == 0 else r0 + 1
        start_c = c0 if qc == 0 else c0 + 1
        dist = random.randint(1, limits[(qr, qc)])
        rr = start_r + dr * dist
        cc = start_c + dc * dist
        panel[rr, cc] = int(legend[qr, qc])
        ray_hits[(qr, qc)] = (rr, cc)

    forbidden = ray_cells | {(rr, cc) for rr in range(r0, r0 + 2) for cc in range(c0, c0 + 2)}

    near_candidates = [
        (rr, cc)
        for rr in range(max(0, r0 - 2), min(h, r0 + 4))
        for cc in range(max(0, c0 - 2), min(w, c0 + 4))
        if (rr, cc) not in forbidden
    ]
    if not near_candidates:
        return None

    remaining = [
        (rr, cc)
        for rr in range(h)
        for cc in range(w)
        if (rr, cc) not in forbidden and panel[rr, cc] == BG_COLOR
    ]
    if len(remaining) < 3:
        return None

    decoy_count = random.randint(2, min(6, len(remaining)))
    chosen_decoys = [random.choice(near_candidates)]
    other_choices = [cell for cell in remaining if cell != chosen_decoys[0]]
    chosen_decoys.extend(random.sample(other_choices, decoy_count - 1))

    decoy_palette = [color for color in ALL_PALETTE if color not in {BG_COLOR, ANCHOR_COLOR, divider_color}]
    for rr, cc in chosen_decoys:
        panel[rr, cc] = random.choice(decoy_palette)

    return panel, {
        "anchor_bbox_local": anchor_bbox,
        "ray_hits_local": ray_hits,
        "ray_cells_local": ray_cells,
        "decoys_local": chosen_decoys,
    }


def _sample_data_panel(shape: tuple[int, int], data_color: int) -> np.ndarray | None:
    h, w = shape
    if h % 2 or w % 2:
        return None

    half_h, half_w = h // 2, w // 2
    area = half_h * half_w
    panel = np.full(shape, BG_COLOR, dtype=int)

    for qr in range(2):
        for qc in range(2):
            rs = slice(qr * half_h, (qr + 1) * half_h)
            cs = slice(qc * half_w, (qc + 1) * half_w)
            cells = [(r, c) for r in range(rs.start, rs.stop) for c in range(cs.start, cs.stop)]
            max_fg = max(1, min(area - 1, int(round(area * 0.45))))
            fg_count = random.randint(1, max_fg)
            for rr, cc in random.sample(cells, fg_count):
                panel[rr, cc] = data_color

    if np.any(np.all(panel == data_color, axis=0)) or np.any(np.all(panel == data_color, axis=1)):
        return None
    return panel


def _expected_output(data_panel: np.ndarray, data_color: int, legend: np.ndarray) -> np.ndarray:
    return _recolor_data_panel(data_panel, data_color, legend)


def _generate_case() -> dict[str, Any] | None:
    orientation = random.choice(("v", "h"))

    if orientation == "v":
        height, anchor_w, divider_t, data_w = _choose_case_dimensions("v")
        total_shape = (height, anchor_w + divider_t + data_w)
        anchor_left = bool(random.getrandbits(1))
        anchor_shape = (height, anchor_w)
        data_shape = (height, data_w)
    else:
        width, anchor_h, divider_t, data_h = _choose_case_dimensions("h")
        total_shape = (anchor_h + divider_t + data_h, width)
        anchor_left = bool(random.getrandbits(1))
        anchor_shape = (anchor_h, width)
        data_shape = (data_h, width)

    divider_choices = [color for color in ALL_PALETTE if color != ANCHOR_COLOR]
    divider_color = random.choice(divider_choices)
    available = [color for color in ALL_PALETTE if color not in {BG_COLOR, ANCHOR_COLOR, divider_color}]
    data_color = random.choice(available)
    legend_palette = [color for color in available if color != data_color]
    legend_values = [random.choice(legend_palette) for _ in range(4)]
    if len(set(legend_values)) < 2:
        return None
    legend = np.array(legend_values, dtype=int).reshape(2, 2)

    anchor_sample = _sample_anchor_panel(anchor_shape, legend, divider_color)
    if anchor_sample is None:
        return None
    anchor_panel, anchor_meta = anchor_sample

    data_panel = _sample_data_panel(data_shape, data_color)
    if data_panel is None:
        return None

    grid = np.full(total_shape, BG_COLOR, dtype=int)
    if orientation == "v":
        if anchor_left:
            grid[:, :anchor_shape[1]] = anchor_panel
            grid[:, anchor_shape[1] : anchor_shape[1] + divider_t] = divider_color
            grid[:, anchor_shape[1] + divider_t :] = data_panel
            anchor_offset = (0, 0)
            data_offset = (0, anchor_shape[1] + divider_t)
        else:
            grid[:, : data_shape[1]] = data_panel
            grid[:, data_shape[1] : data_shape[1] + divider_t] = divider_color
            grid[:, data_shape[1] + divider_t :] = anchor_panel
            anchor_offset = (0, data_shape[1] + divider_t)
            data_offset = (0, 0)
    else:
        if anchor_left:
            grid[: anchor_shape[0], :] = anchor_panel
            grid[anchor_shape[0] : anchor_shape[0] + divider_t, :] = divider_color
            grid[anchor_shape[0] + divider_t :, :] = data_panel
            anchor_offset = (0, 0)
            data_offset = (anchor_shape[0] + divider_t, 0)
        else:
            grid[: data_shape[0], :] = data_panel
            grid[data_shape[0] : data_shape[0] + divider_t, :] = divider_color
            grid[data_shape[0] + divider_t :, :] = anchor_panel
            anchor_offset = (data_shape[0] + divider_t, 0)
            data_offset = (0, 0)

    if int(get_bg(grid)) != BG_COLOR:
        return None

    anchor_bbox_local = anchor_meta["anchor_bbox_local"]
    anchor_bbox = (
        anchor_offset[0] + anchor_bbox_local[0],
        anchor_offset[1] + anchor_bbox_local[1],
        anchor_offset[0] + anchor_bbox_local[2],
        anchor_offset[1] + anchor_bbox_local[3],
    )
    ray_hits = {
        key: (anchor_offset[0] + rr, anchor_offset[1] + cc)
        for key, (rr, cc) in anchor_meta["ray_hits_local"].items()
    }
    decoys = [(anchor_offset[0] + rr, anchor_offset[1] + cc) for rr, cc in anchor_meta["decoys_local"]]
    ray_cells = {(anchor_offset[0] + rr, anchor_offset[1] + cc) for rr, cc in anchor_meta["ray_cells_local"]}

    if len([pos for pos in ray_cells if int(grid[pos]) not in {BG_COLOR, ANCHOR_COLOR}]) != 4:
        return None
    if any(pos in ray_cells for pos in decoys):
        return None

    expected = _expected_output(data_panel, data_color, legend)
    return {
        "grid": grid,
        "expected": expected,
        "orientation": orientation,
        "divider_color": divider_color,
        "data_color": data_color,
        "legend": legend,
        "anchor_bbox": anchor_bbox,
        "data_bbox": (
            data_offset[0],
            data_offset[1],
            data_offset[0] + data_shape[0] - 1,
            data_offset[1] + data_shape[1] - 1,
        ),
        "ray_hits": ray_hits,
        "ray_cells": ray_cells,
        "decoys": decoys,
    }


def _validate_case(case: dict[str, Any]) -> None:
    grid = case["grid"]
    out = main(grid)
    assert np.array_equal(out, case["expected"])

    info = _analyze_grid(grid)
    assert np.array_equal(info["legend"], case["legend"])
    assert info["data_color"] == case["data_color"]
    assert info["anchor_bbox"] == case["anchor_bbox"]
    assert info["data_bbox"] == case["data_bbox"]
    assert info["ray_hits"] == case["ray_hits"]
    assert case["ray_cells"] == info["ray_cells"]
    assert all(decoy not in info["ray_cells"] for decoy in case["decoys"])

    data_panel = crop_bbox(grid, case["data_bbox"])
    half_h, half_w = data_panel.shape[0] // 2, data_panel.shape[1] // 2
    for qr in range(2):
        for qc in range(2):
            block = data_panel[qr * half_h : (qr + 1) * half_h, qc * half_w : (qc + 1) * half_w]
            assert np.any(block == case["data_color"]), "every quadrant must use its legend entry"

    assert grid.ndim == 2 and out.ndim == 2
    assert max(grid.shape) <= 30 and max(out.shape) <= 30


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        case = _generate_case()
        if case is None:
            continue
        try:
            _validate_case(case)
        except AssertionError:
            continue
        return case["grid"]
    raise RuntimeError("failed to generate a valid diagonal-legend ARC input")
