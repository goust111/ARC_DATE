from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

LEGEND_H = 2
LEGEND_CELL_W = 3

GEN_RETRIES = 400
GEN_POSITION_TRIES = 160
GEN_MAIN_H_RANGE_2 = (16, 22)
GEN_MAIN_H_RANGE_3 = (18, 24)
GEN_MAIN_WIDTHS_2 = [18, 21, 24, 27]
GEN_MAIN_WIDTHS_3 = [21, 24, 27]
GEN_RADIUS_RANGE = (1, 3)
GEN_OBJ_H_RANGE = (5, 9)
GEN_OBJ_W_RANGE = (6, 11)
GEN_MIN_VIS_HALO = 12
GEN_BBOX_GAP = 5


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray, colors: set[int]) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if visited[r, c] or color not in colors:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                }
            )
    return comps


def _contiguous_span(indices: np.ndarray) -> tuple[int, int]:
    assert indices.size >= 3, "side run must have length >= 3"
    assert np.all(indices[1:] == indices[:-1] + 1), "side run must be contiguous"
    return int(indices[0]), int(indices[-1])


def _centered_span(source_lo: int, source_hi: int, length: int) -> tuple[int, int]:
    center2 = int(source_lo) + int(source_hi)
    lo = (center2 - (int(length) - 1)) // 2
    hi = (center2 + (int(length) - 1)) // 2
    return int(lo), int(hi)


def _paint_shrinking_rows(
    out: np.ndarray,
    row: int,
    direction: int,
    c0: int,
    c1: int,
    halo_color: int,
    step: int,
) -> None:
    h, w = out.shape
    source_len = int(c1) - int(c0) + 1
    distance = 1
    length = source_len - int(step)
    while length > 0:
        rr = int(row) + int(direction) * distance
        if 0 <= rr < h:
            lo, hi = _centered_span(c0, c1, length)
            for cc in range(lo, hi + 1):
                if 0 <= cc < w and int(out[rr, cc]) == BLACK:
                    out[rr, cc] = int(halo_color)
        distance += 1
        length -= int(step)


def _paint_shrinking_cols(
    out: np.ndarray,
    col: int,
    direction: int,
    r0: int,
    r1: int,
    halo_color: int,
    step: int,
) -> None:
    h, w = out.shape
    source_len = int(r1) - int(r0) + 1
    distance = 1
    length = source_len - int(step)
    while length > 0:
        cc = int(col) + int(direction) * distance
        if 0 <= cc < w:
            lo, hi = _centered_span(r0, r1, length)
            for rr in range(lo, hi + 1):
                if 0 <= rr < h and int(out[rr, cc]) == BLACK:
                    out[rr, cc] = int(halo_color)
        distance += 1
        length -= int(step)


def _local_mask_from_component(shape: tuple[int, int], coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for rr, cc in coords:
        local[int(rr) - r0, int(cc) - c0] = True
    assert local[0].any() and local[-1].any() and local[:, 0].any() and local[:, -1].any()
    return local


def _paint_object_halo(
    out: np.ndarray,
    local_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    halo_color: int,
    step: int,
) -> None:
    r0, c0, r1, c1 = bbox
    top_c0, top_c1 = _contiguous_span(np.flatnonzero(local_mask[0]))
    bottom_c0, bottom_c1 = _contiguous_span(np.flatnonzero(local_mask[-1]))
    left_r0, left_r1 = _contiguous_span(np.flatnonzero(local_mask[:, 0]))
    right_r0, right_r1 = _contiguous_span(np.flatnonzero(local_mask[:, -1]))

    _paint_shrinking_rows(out, r0, -1, c0 + top_c0, c0 + top_c1, halo_color, step)
    _paint_shrinking_rows(out, r1, 1, c0 + bottom_c0, c0 + bottom_c1, halo_color, step)
    _paint_shrinking_cols(out, c0, -1, r0 + left_r0, r0 + left_r1, halo_color, step)
    _paint_shrinking_cols(out, c1, 1, r0 + right_r0, r0 + right_r1, halo_color, step)


def _parse_legend(grid: np.ndarray) -> tuple[dict[int, tuple[int, int]], int]:
    h, w = grid.shape
    grey_rows = []
    for r in range(h):
        if np.any(grid[r] == GREY):
            grey_rows.append(r)
        else:
            break
    assert grey_rows == list(range(len(grey_rows))), "legend must be a top prefix"
    legend_h = len(grey_rows)
    assert legend_h == LEGEND_H, "legend must be exactly two rows tall"

    mapping: dict[int, tuple[int, int]] = {}
    for c0 in range(0, w - LEGEND_CELL_W + 1, LEGEND_CELL_W):
        cell = grid[:legend_h, c0 : c0 + LEGEND_CELL_W]
        if np.all(cell == GREY):
            continue
        top = cell[0]
        bottom = cell[1]
        assert int(top[0]) == GREY and int(top[2]) == GREY
        obj_color = int(top[1])
        assert obj_color not in (BLACK, GREY)
        halo_pixels = [(i, int(v)) for i, v in enumerate(bottom) if int(v) != GREY]
        assert len(halo_pixels) == 1, "legend cell must contain exactly one halo marker"
        idx, halo_color = halo_pixels[0]
        assert idx in (0, 2), "halo marker must be on the left or right"
        assert halo_color not in (BLACK, GREY)
        assert int(bottom[1]) == GREY
        mapping[obj_color] = (halo_color, 2 if idx == 0 else 1)
    assert mapping, "legend must contain at least one entry"
    return mapping, legend_h


def _solve_main_area(main_area: np.ndarray, mapping: dict[int, tuple[int, int]]) -> np.ndarray:
    g = np.asarray(main_area, dtype=int)
    out = g.copy()
    comps = _find_components(g, set(mapping))
    assert comps, "main area must contain objects"
    for comp in comps:
        assert comp["color"] in mapping, "every object color must appear in the legend"
    comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
    for comp in comps:
        local_mask = _local_mask_from_component(g.shape, comp["coords"], comp["bbox"])
        halo_color, step = mapping[comp["color"]]
        _paint_object_halo(out, local_mask, comp["bbox"], halo_color, step)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the grey legend and extend each rounded object with a legend-coded shrinking halo.

    Input structure:
    - The background is black, and the top two rows form a grey legend band.
    - Each 3-column legend cell has the object color in the top-center pixel and one halo-color pixel below.
    - The halo pixel sits on the cell's left or right column, while the main area below contains 2-3 large 4-connected rounded monochrome objects in the legend's object colors.

    Rule:
    - Read each legend cell as object_color -> (halo_color, shrink_step), where left means shrink by 2 each layer and right means shrink by 1.
    - Remove the legend rows from the output.
    - For every object, keep its original pixels and, from each of its top/bottom/left/right edge runs, paint outward shrinking runs in the object's halo color.
    - Each new layer is shorter by the legend-coded shrink step, and halo pixels only fill black background cells without overwriting earlier halos or any object pixels.

    Key insight:
    - The legend uses one visible token twice: its color gives the halo paint, and its left-vs-right placement gives the geometric decay rate that changes the halo shape.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    mapping, legend_h = _parse_legend(g)
    main_area = g[legend_h:, :]
    assert main_area.size > 0, "main area must exist below the legend"
    assert not np.any(main_area == GREY), "grey is reserved for the legend"
    return _solve_main_area(main_area, mapping)


def _rounded_mask(height: int, width: int, radius: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for r in range(height):
        edge_distance = min(r, height - 1 - r, radius)
        inset = radius - edge_distance
        mask[r, inset : width - inset] = True
    return mask


def _random_shape() -> np.ndarray:
    for _ in range(80):
        radius = random.randint(*GEN_RADIUS_RANGE)
        min_h = max(GEN_OBJ_H_RANGE[0], 2 * radius + 3)
        min_w = max(GEN_OBJ_W_RANGE[0], 2 * radius + 3)
        if min_h > GEN_OBJ_H_RANGE[1] or min_w > GEN_OBJ_W_RANGE[1]:
            continue
        height = random.randint(min_h, GEN_OBJ_H_RANGE[1])
        width = random.randint(min_w, GEN_OBJ_W_RANGE[1])
        mask = _rounded_mask(height, width, radius)
        side_h = height - 2 * radius
        side_w = width - 2 * radius
        if side_h >= 3 and side_w >= 3:
            return mask
    raise RuntimeError("failed to sample a valid rounded object")


def _embed_mask(grid_shape: tuple[int, int], mask: np.ndarray, pos: tuple[int, int]) -> np.ndarray:
    h, w = grid_shape
    out = np.zeros((h, w), dtype=bool)
    r0, c0 = pos
    mh, mw = mask.shape
    out[r0 : r0 + mh, c0 : c0 + mw] = mask
    return out


def _bbox_gap_ok(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _halo_mask_for_spec(
    grid_shape: tuple[int, int],
    local_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    step: int,
) -> np.ndarray:
    blank = np.zeros(grid_shape, dtype=int)
    _paint_object_halo(blank, local_mask, bbox, 1, step)
    return blank == 1


def _render_main_area(grid_shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    out = np.zeros(grid_shape, dtype=int)
    for spec in specs:
        r0, c0 = spec["bbox"][0], spec["bbox"][1]
        mh, mw = spec["mask"].shape
        region = out[r0 : r0 + mh, c0 : c0 + mw]
        region[spec["mask"]] = int(spec["color"])
        out[r0 : r0 + mh, c0 : c0 + mw] = region
    return out


def _simulate_from_specs(main_area: np.ndarray, specs: list[dict], mapping: dict[int, tuple[int, int]]) -> np.ndarray:
    out = main_area.copy()
    for spec in sorted(specs, key=lambda item: (item["bbox"][0], item["bbox"][1])):
        halo_color, step = mapping[int(spec["color"])]
        _paint_object_halo(out, spec["mask"], spec["bbox"], halo_color, step)
    return out


def _simulate_order_pairing(
    main_area: np.ndarray,
    specs: list[dict],
    legend_entries: list[tuple[int, int, int]],
) -> np.ndarray:
    out = main_area.copy()
    ordered_specs = sorted(specs, key=lambda item: (item["bbox"][0], item["bbox"][1]))
    assert len(ordered_specs) == len(legend_entries)
    for spec, (_obj_color, halo_color, step) in zip(ordered_specs, legend_entries):
        _paint_object_halo(out, spec["mask"], spec["bbox"], halo_color, step)
    return out


def _build_input(main_area: np.ndarray, legend_entries: list[tuple[int, int, int]]) -> np.ndarray:
    h, w = main_area.shape
    legend = np.full((LEGEND_H, w), GREY, dtype=int)
    for idx, (obj_color, halo_color, step) in enumerate(legend_entries):
        c0 = idx * LEGEND_CELL_W
        legend[0, c0 + 1] = int(obj_color)
        legend[1, c0 + (0 if int(step) == 2 else 2)] = int(halo_color)
    return np.vstack([legend, main_area])


def _position_candidates(main_shape: tuple[int, int], mask: np.ndarray, bias_edge: bool) -> list[tuple[int, int]]:
    h, w = main_shape
    mh, mw = mask.shape
    positions = [(r, c) for r in range(h - mh + 1) for c in range(w - mw + 1)]
    random.shuffle(positions)
    if not bias_edge:
        return positions
    edge_positions = []
    inner_positions = []
    for r, c in positions:
        if min(r, c, h - (r + mh), w - (c + mw)) <= 2:
            edge_positions.append((r, c))
        else:
            inner_positions.append((r, c))
    return edge_positions + inner_positions


def _pick_canvas(n_objects: int) -> tuple[int, int]:
    if n_objects == 2:
        return random.randint(*GEN_MAIN_H_RANGE_2), random.choice(GEN_MAIN_WIDTHS_2)
    return random.randint(*GEN_MAIN_H_RANGE_3), random.choice(GEN_MAIN_WIDTHS_3)


def _place_specs(main_shape: tuple[int, int], raw_specs: list[dict]) -> list[dict] | None:
    placed: list[dict] = []
    all_object_cells = np.zeros(main_shape, dtype=bool)
    all_halo_cells = np.zeros(main_shape, dtype=bool)

    random.shuffle(raw_specs)
    edge_index = random.randrange(len(raw_specs))
    for idx, base_spec in enumerate(raw_specs):
        positions = _position_candidates(main_shape, base_spec["mask"], bias_edge=(idx == edge_index and random.random() < 0.45))
        success = False
        for pos in positions[:GEN_POSITION_TRIES]:
            r0, c0 = pos
            mh, mw = base_spec["mask"].shape
            bbox = (r0, c0, r0 + mh - 1, c0 + mw - 1)
            obj_cells = _embed_mask(main_shape, base_spec["mask"], pos)
            halo_cells = _halo_mask_for_spec(main_shape, base_spec["mask"], bbox, int(base_spec["step"]))
            if halo_cells.sum() < GEN_MIN_VIS_HALO:
                continue
            if np.any(obj_cells & all_object_cells):
                continue
            if np.any(halo_cells & (all_object_cells | all_halo_cells)):
                continue
            if np.any(obj_cells & all_halo_cells):
                continue
            if any(not _bbox_gap_ok(bbox, spec["bbox"], GEN_BBOX_GAP) for spec in placed):
                continue
            spec = dict(base_spec)
            spec["bbox"] = bbox
            spec["object_cells"] = obj_cells
            spec["halo_cells"] = halo_cells
            placed.append(spec)
            all_object_cells |= obj_cells
            all_halo_cells |= halo_cells
            success = True
            break
        if not success:
            return None
    return placed


def _sample_specs() -> tuple[list[dict], list[tuple[int, int, int]], tuple[int, int]] | None:
    n_objects = random.choice([2, 3])
    main_shape = _pick_canvas(n_objects)
    colors = random.sample(PALETTE, 2 * n_objects)
    object_colors = colors[:n_objects]
    halo_colors = colors[n_objects:]

    steps = [1, 2]
    while len(steps) < n_objects:
        steps.append(random.choice([1, 2]))
    random.shuffle(steps)
    if 1 not in steps or 2 not in steps:
        return None

    raw_specs = []
    for obj_color, halo_color, step in zip(object_colors, halo_colors, steps):
        raw_specs.append(
            {
                "color": int(obj_color),
                "halo": int(halo_color),
                "step": int(step),
                "mask": _random_shape(),
            }
        )

    placed = _place_specs(main_shape, raw_specs)
    if placed is None:
        return None

    sorted_specs = sorted(placed, key=lambda item: (item["bbox"][0], item["bbox"][1]))
    position_order = [int(spec["color"]) for spec in sorted_specs]

    legend_entries = [(int(spec["color"]), int(spec["halo"]), int(spec["step"])) for spec in placed]
    random.shuffle(legend_entries)
    if [entry[0] for entry in legend_entries] == position_order:
        return None
    return placed, legend_entries, main_shape


def _attempt() -> np.ndarray | None:
    sampled = _sample_specs()
    if sampled is None:
        return None
    specs, legend_entries, main_shape = sampled

    main_area = _render_main_area(main_shape, specs)
    true_mapping = {int(spec["color"]): (int(spec["halo"]), int(spec["step"])) for spec in specs}
    expected = _simulate_from_specs(main_area, specs, true_mapping)
    if np.array_equal(expected, main_area):
        return None

    order_variant = _simulate_order_pairing(main_area, specs, legend_entries)
    if np.array_equal(order_variant, expected):
        return None

    same_one = {int(spec["color"]): (int(spec["halo"]), 1) for spec in specs}
    same_two = {int(spec["color"]): (int(spec["halo"]), 2) for spec in specs}
    flipped = {int(spec["color"]): (int(spec["halo"]), 3 - int(spec["step"])) for spec in specs}
    if np.array_equal(_simulate_from_specs(main_area, specs, same_one), expected):
        return None
    if np.array_equal(_simulate_from_specs(main_area, specs, same_two), expected):
        return None
    if np.array_equal(_simulate_from_specs(main_area, specs, flipped), expected):
        return None

    inp = _build_input(main_area, legend_entries)
    try:
        pred = main(inp)
    except Exception:
        return None
    if not np.array_equal(pred, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
