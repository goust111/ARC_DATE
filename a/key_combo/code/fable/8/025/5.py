from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox, crop_bbox, find_components, get_bg, neighbors4


BG = Color.BLACK
LINE = Color.BLUE
GRAY = Color.GREY
PANEL_COLORS = [Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
SWATCH_COLORS = [Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
GEN_ATTEMPTS = 400


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    mask = np.asarray(mask, dtype=bool)
    return tuple(map(int, mask.shape)), tuple(map(int, mask.ravel()))


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    b = bbox(mask)
    assert b is not None
    return crop_bbox(mask.astype(int), b).astype(bool)


def _unique_variants(mask: np.ndarray) -> list[np.ndarray]:
    base = _normalize_mask(mask)
    out: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], tuple[int, ...]]] = set()
    for k in range(4):
        rot = np.rot90(base, k)
        for variant in (rot, np.fliplr(rot)):
            key = _mask_key(_normalize_mask(variant))
            if key in seen:
                continue
            seen.add(key)
            out.append(_normalize_mask(variant))
    return out


def _orbit_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return min(_mask_key(v) for v in _unique_variants(mask))


def _component_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    coords = np.asarray(comp["coords"], dtype=int)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _touches_local_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = map(int, shape)
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _solid_rect(comp: dict) -> bool:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    return int(comp["area"]) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _find_panel_and_hole(g: np.ndarray) -> tuple[dict, np.ndarray]:
    bg = int(get_bg(g))
    candidates: list[tuple[int, dict, np.ndarray]] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        if int(comp["color"]) in {bg, LINE, GRAY}:
            continue
        region = crop_bbox(g, comp["bbox"])
        colors = set(map(int, np.unique(region)))
        if not colors <= {bg, int(comp["color"])}:
            continue
        bg_mask = region == bg
        if not bg_mask.any():
            continue
        holes = find_components(bg_mask.astype(int), bg=0, connectivity=4, monochromatic=True, colors=[1])
        inner_masks: list[np.ndarray] = []
        for hole in holes:
            coords = np.asarray(hole["coords"], dtype=int)
            if _touches_local_border(coords, bg_mask.shape):
                continue
            hole_mask = np.zeros_like(bg_mask, dtype=bool)
            hole_mask[coords[:, 0], coords[:, 1]] = True
            inner_masks.append(hole_mask)
        if len(inner_masks) == 1:
            candidates.append((int(comp["area"]), comp, inner_masks[0]))
    assert len(candidates) == 1, "expected a unique panel containing one inner hole"
    _area, panel, hole_mask = candidates[0]
    r0, c0, _r1, _c1 = map(int, panel["bbox"])
    abs_hole = np.zeros_like(g, dtype=bool)
    abs_hole[r0 : r0 + hole_mask.shape[0], c0 : c0 + hole_mask.shape[1]] = hole_mask
    return panel, abs_hole


def _analyze_entries(g: np.ndarray, panel: dict) -> list[dict]:
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    non_line = [comp for comp in comps if int(comp["color"]) != LINE]
    id_by_cell: dict[tuple[int, int], int] = {}
    for idx, comp in enumerate(non_line):
        for r, c in np.asarray(comp["coords"], dtype=int):
            id_by_cell[(int(r), int(c))] = idx

    panel_cells = {tuple(map(int, rc)) for rc in np.asarray(panel["coords"], dtype=int)}
    entries: list[dict] = []
    for line in [comp for comp in comps if int(comp["color"]) == LINE]:
        touched_ids: set[int] = set()
        for r, c in np.asarray(line["coords"], dtype=int):
            for nr, nc in neighbors4(int(r), int(c)):
                obj_id = id_by_cell.get((int(nr), int(nc)))
                if obj_id is not None:
                    touched_ids.add(obj_id)
        touched = [non_line[idx] for idx in touched_ids]
        greys = [comp for comp in touched if int(comp["color"]) == GRAY]
        swatches = [
            comp
            for comp in touched
            if int(comp["color"]) not in {GRAY, LINE, int(panel["color"])}
            and tuple(map(int, comp["coords"][0])) not in panel_cells
        ]
        assert len(greys) == 1 and len(swatches) == 1, "each blue link must pair one grey shape and one color swatch"
        entries.append({"shape": greys[0], "swatch": swatches[0], "line": line, "color": int(swatches[0]["color"])})
    assert entries, "expected at least one linked dictionary entry"
    return entries


def _match_color(g: np.ndarray) -> tuple[np.ndarray, int, list[dict]]:
    panel, abs_hole = _find_panel_and_hole(g)
    hole_mask = _normalize_mask(abs_hole)
    hole_keys = {_mask_key(v) for v in _unique_variants(hole_mask)}
    entries = _analyze_entries(g, panel)
    matches = [
        entry
        for entry in entries
        if _mask_key(_normalize_mask(_component_mask(entry["shape"]))) in hole_keys
    ]
    assert len(matches) == 1, "expected one unique linked shape matching the hole"
    return abs_hole, int(matches[0]["color"]), entries


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the panel's negative-space hole by matching it to the linked grey legend shape with the same shape up to rotation or reflection.

    Input structure:
    - A black background contains one large monochrome panel whose interior has one connected black hole.
    - Elsewhere, small solid color squares and grey shapes appear as separate objects, with blue 4-connected line paths linking some square-shape pairs.
    - Some extra color squares and grey shapes are unlinked distractors.

    Rule:
    - Treat each blue line as one dictionary entry pairing exactly one color square with exactly one grey shape.
    - Read the panel's black hole as the query shape and compare it to the linked grey shapes up to any rotation or mirror reflection.
    - Find the unique matching linked shape and fill the hole with its paired square color, leaving every other pixel unchanged.

    Key insight:
    - The target object is invisible positive space: the hole inside the panel must be matched by shape equivalence, and only blue-linked color-shape pairs define the usable dictionary.
    """
    g = np.asarray(grid, dtype=int)
    hole, color, _entries = _match_color(g)
    out = g.copy()
    out[hole] = int(color)
    return out


def _random_shape(rng: np.random.Generator, forbidden: set[tuple[tuple[int, int], tuple[int, ...]]]) -> np.ndarray | None:
    for _ in range(200):
        h = int(rng.integers(3, 6))
        w = int(rng.integers(3, 6))
        area = int(rng.integers(4, min(8, h * w)))
        start = (int(rng.integers(h)), int(rng.integers(w)))
        cells = {start}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        if len(cells) != area:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _normalize_mask(mask)
        if mask.sum() < 4 or mask.sum() == mask.size:
            continue
        if len(find_components(mask.astype(int), bg=0, connectivity=4, monochromatic=True, colors=[1])) != 1:
            continue
        if len(_unique_variants(mask)) < 4:
            continue
        coords = np.argwhere(mask)
        if not (
            np.any(coords[:, 0] == 0)
            and np.any(coords[:, 0] == mask.shape[0] - 1)
            and np.any(coords[:, 1] == 0)
            and np.any(coords[:, 1] == mask.shape[1] - 1)
        ):
            continue
        key = _orbit_key(mask)
        if key in forbidden:
            continue
        return mask
    return None


def _paint_mask(g: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> np.ndarray:
    coords = np.argwhere(mask)
    g[top + coords[:, 0], left + coords[:, 1]] = int(color)
    return coords + np.array([top, left], dtype=int)


def _rect_cells(top: int, left: int, h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(top, top + h) for c in range(left, left + w)}


def _expanded(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = map(int, shape)
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _can_place(cells: set[tuple[int, int]], occupied: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = map(int, shape)
    return all(0 <= r < h and 0 <= c < w for r, c in cells) and not (_expanded(cells, shape) & occupied)


def _place_entry(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    dict_left: int,
    dict_right: int,
    base_top: int,
    shape_mask: np.ndarray,
    swatch_color: int,
    swatch_size: int,
    swatch_on_left: bool,
) -> dict | None:
    h, w = map(int, grid.shape)
    sh, sw = map(int, shape_mask.shape)
    shape_rows = [r for r in range(sh) if np.any(shape_mask[r])]
    for shape_top in range(max(1, base_top - 1), min(h - sh - 1, base_top + 2) + 1):
        if swatch_on_left:
            shape_left = dict_right - sw + 1
            swatch_left = dict_left
        else:
            shape_left = dict_left
            swatch_left = dict_right - swatch_size + 1
        for shape_row in random.sample(shape_rows, len(shape_rows)):
            shape_cols = np.flatnonzero(shape_mask[shape_row])
            for swatch_row in random.sample(range(swatch_size), swatch_size):
                swatch_top = shape_top + shape_row - swatch_row
                if not (1 <= swatch_top and swatch_top + swatch_size <= h - 1):
                    continue
                if swatch_on_left:
                    target_col = shape_left + int(shape_cols.min())
                    line_cols = list(range(swatch_left + swatch_size, target_col))
                else:
                    target_col = shape_left + int(shape_cols.max())
                    line_cols = list(range(target_col + 1, swatch_left))
                if not line_cols:
                    continue
                line_row = shape_top + shape_row
                swatch_cells = _rect_cells(swatch_top, swatch_left, swatch_size, swatch_size)
                shape_cells = {
                    (shape_top + int(r), shape_left + int(c))
                    for r, c in np.argwhere(shape_mask)
                }
                line_cells = {(line_row, int(c)) for c in line_cols}
                all_cells = swatch_cells | shape_cells | line_cells
                if not _can_place(all_cells, occupied, grid.shape):
                    continue
                grid[swatch_top : swatch_top + swatch_size, swatch_left : swatch_left + swatch_size] = int(swatch_color)
                _paint_mask(grid, shape_mask, shape_top, shape_left, GRAY)
                for r, c in line_cells:
                    grid[r, c] = LINE
                occupied.update(all_cells)
                return {
                    "swatch_color": int(swatch_color),
                    "swatch_bbox": (swatch_top, swatch_left, swatch_top + swatch_size - 1, swatch_left + swatch_size - 1),
                    "shape_bbox": (shape_top, shape_left, shape_top + sh - 1, shape_left + sw - 1),
                    "shape_mask": shape_mask.copy(),
                }
    return None


def _hole_distance(hole_coords: np.ndarray, box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = map(int, box)
    cells = np.array([(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)], dtype=int)
    d = np.abs(hole_coords[:, None, :] - cells[None, :, :]).sum(axis=2)
    return int(d.min())


def _choose_near_swatch_pos(
    rng: np.random.Generator,
    occupied: set[tuple[int, int]],
    grid_shape: tuple[int, int],
    panel_box: tuple[int, int, int, int],
    hole_coords: np.ndarray,
    swatch_size: int,
) -> tuple[int, int] | None:
    h, w = map(int, grid_shape)
    pr0, pc0, pr1, pc1 = map(int, panel_box)
    candidates: list[tuple[int, int, int]] = []
    for top in range(1, h - swatch_size):
        for left in range(1, w - swatch_size):
            cells = _rect_cells(top, left, swatch_size, swatch_size)
            if not _can_place(cells, occupied, grid_shape):
                continue
            if any(pr0 <= r <= pr1 and pc0 <= c <= pc1 for r, c in cells):
                continue
            box = (top, left, top + swatch_size - 1, left + swatch_size - 1)
            candidates.append((_hole_distance(hole_coords, box), top, left))
    if not candidates:
        return None
    candidates.sort()
    best = [c for c in candidates if c[0] <= candidates[0][0] + 2]
    _dist, top, left = best[int(rng.integers(len(best)))]
    return int(top), int(left)


def _place_mask_anywhere(
    rng: np.random.Generator,
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    mask: np.ndarray,
) -> tuple[int, int] | None:
    h, w = map(int, grid.shape)
    mh, mw = map(int, mask.shape)
    positions = [(r, c) for r in range(1, h - mh) for c in range(1, w - mw)]
    rng.shuffle(positions)
    for top, left in positions:
        cells = {(top + int(r), left + int(c)) for r, c in np.argwhere(mask)}
        if _can_place(cells, occupied, grid.shape):
            return int(top), int(left)
    return None


def _build_scene(rng: np.random.Generator) -> tuple[np.ndarray, dict] | None:
    entry_count = int(rng.integers(3, 5))
    used_orbits: set[tuple[tuple[int, int], tuple[int, ...]]] = set()
    base_shapes: list[np.ndarray] = []
    for _ in range(entry_count + 1):
        shape = _random_shape(rng, used_orbits)
        if shape is None:
            return None
        used_orbits.add(_orbit_key(shape))
        base_shapes.append(shape)

    linked_bases = base_shapes[:entry_count]
    distractor_base = base_shapes[-1]
    query_index = int(rng.integers(entry_count))
    linked_masks: list[np.ndarray] = []
    for base in linked_bases:
        variants = _unique_variants(base)
        linked_masks.append(variants[int(rng.integers(len(variants)))])
    query_variants = _unique_variants(linked_bases[query_index])
    query_display_key = _mask_key(_normalize_mask(linked_masks[query_index]))
    hole_options = [variant for variant in query_variants if _mask_key(variant) != query_display_key]
    if not hole_options:
        return None
    hole_mask = hole_options[int(rng.integers(len(hole_options)))]

    panel_color = int(rng.choice([c for c in PANEL_COLORS if c != LINE and c != GRAY]))
    swatch_palette = [c for c in SWATCH_COLORS if c not in {panel_color, LINE, GRAY}]
    rng.shuffle(swatch_palette)
    swatch_colors = swatch_palette[:entry_count + 1]
    distractor_swatch_color = int(swatch_colors[-1])
    linked_colors = list(map(int, swatch_colors[:-1]))

    panel_h = int(rng.integers(max(10, hole_mask.shape[0] + 4), 15))
    panel_w = int(rng.integers(max(10, hole_mask.shape[1] + 4), 15))
    dict_width = int(rng.integers(10, 13))
    swatch_sizes = [int(rng.integers(2, 4)) for _ in range(entry_count)]
    entry_heights = [max(mask.shape[0], s) + 3 for mask, s in zip(linked_masks, swatch_sizes)]
    grid_h = max(18, panel_h + 6, sum(entry_heights) + 3)
    grid_h = min(grid_h + int(rng.integers(0, 3)), 29)
    grid_w = panel_w + dict_width + int(rng.integers(5, 8))
    if grid_w > 29:
        return None
    grid = np.zeros((grid_h, grid_w), dtype=int)
    occupied: set[tuple[int, int]] = set()

    panel_on_left = bool(rng.integers(2))
    panel_top = int(rng.integers(1, grid_h - panel_h - 1))
    if panel_on_left:
        panel_left = int(rng.integers(1, 3))
        dict_left = panel_left + panel_w + int(rng.integers(3, 5))
        dict_right = grid_w - 2
        swatch_on_left = True
    else:
        panel_left = grid_w - panel_w - int(rng.integers(2, 4))
        dict_left = 1
        dict_right = panel_left - int(rng.integers(3, 5))
        swatch_on_left = False
    if dict_right - dict_left + 1 < dict_width:
        return None

    panel_box = (panel_top, panel_left, panel_top + panel_h - 1, panel_left + panel_w - 1)
    pr0, pc0, pr1, pc1 = panel_box
    grid[pr0 : pr1 + 1, pc0 : pc1 + 1] = panel_color
    hole_top_hi = pr1 - hole_mask.shape[0] + 1
    hole_left_hi = pc1 - hole_mask.shape[1] + 1
    if hole_top_hi <= pr0 + 1 or hole_left_hi <= pc0 + 1:
        return None
    hole_top = int(rng.integers(pr0 + 1, hole_top_hi))
    hole_left = int(rng.integers(pc0 + 1, hole_left_hi))
    hole_coords = _paint_mask(np.full_like(grid, BG), hole_mask, hole_top, hole_left, 1)
    grid[hole_coords[:, 0], hole_coords[:, 1]] = BG
    occupied.update(_rect_cells(pr0, pc0, panel_h, panel_w))
    occupied -= {tuple(map(int, rc)) for rc in hole_coords}

    entries_meta = []
    row_cursor = 1
    for idx, (shape_mask, color, swatch_size) in enumerate(zip(linked_masks, linked_colors, swatch_sizes)):
        placement = _place_entry(
            grid,
            occupied,
            dict_left,
            dict_right,
            row_cursor,
            shape_mask,
            color,
            swatch_size,
            swatch_on_left,
        )
        if placement is None:
            return None
        placement["is_query"] = idx == query_index
        entries_meta.append(placement)
        row_cursor = max(row_cursor + max(shape_mask.shape[0], swatch_size) + 3, placement["shape_bbox"][2] + 3)

    near_swatch_size = int(rng.integers(2, 4))
    near_pos = _choose_near_swatch_pos(rng, occupied, grid.shape, panel_box, hole_coords, near_swatch_size)
    if near_pos is None:
        return None
    ds_top, ds_left = near_pos
    grid[ds_top : ds_top + near_swatch_size, ds_left : ds_left + near_swatch_size] = distractor_swatch_color
    distractor_swatch_box = (ds_top, ds_left, ds_top + near_swatch_size - 1, ds_left + near_swatch_size - 1)
    occupied.update(_rect_cells(ds_top, ds_left, near_swatch_size, near_swatch_size))

    distractor_variants = _unique_variants(distractor_base)
    distractor_mask = distractor_variants[int(rng.integers(len(distractor_variants)))]
    distractor_pos = _place_mask_anywhere(rng, grid, occupied, distractor_mask)
    if distractor_pos is None:
        return None
    dg_top, dg_left = distractor_pos
    distractor_gray_coords = _paint_mask(grid, distractor_mask, dg_top, dg_left, GRAY)
    occupied.update({tuple(map(int, rc)) for rc in distractor_gray_coords})

    meta = {
        "panel_box": panel_box,
        "panel_color": panel_color,
        "hole_coords": hole_coords.copy(),
        "matched_color": int(linked_colors[query_index]),
        "entries": entries_meta,
        "distractor_swatch_box": distractor_swatch_box,
    }
    return grid, meta


def _validate_scene(inp: np.ndarray, expected: dict) -> None:
    out = main(inp)
    assert isinstance(out, np.ndarray) and out.ndim == 2
    assert max(inp.shape) <= 30 and max(out.shape) <= 30
    hole_coords = np.asarray(expected["hole_coords"], dtype=int)
    assert np.all(out[hole_coords[:, 0], hole_coords[:, 1]] == int(expected["matched_color"]))
    preserved = np.ones(inp.shape, dtype=bool)
    preserved[hole_coords[:, 0], hole_coords[:, 1]] = False
    assert np.array_equal(out[preserved], inp[preserved])

    panel, hole = _find_panel_and_hole(inp)
    assert panel["bbox"] == expected["panel_box"]
    _hole, matched_color, entries = _match_color(inp)
    assert matched_color == int(expected["matched_color"])
    keys = [_orbit_key(_component_mask(entry["shape"])) for entry in entries]
    assert len(keys) == len(set(keys)), "linked grey shapes must be pairwise non-equivalent"

    swatch_boxes = [entry["swatch_bbox"] for entry in expected["entries"]] + [expected["distractor_swatch_box"]]
    distances = [_hole_distance(hole_coords, box) for box in swatch_boxes]
    assert distances[-1] == min(distances) and distances.count(min(distances)) == 1, "nearest swatch must be the unique distractor"
    assert expected["matched_color"] != int(inp[expected["distractor_swatch_box"][0], expected["distractor_swatch_box"][1]])
    query_entry = next(entry for entry in expected["entries"] if entry["is_query"])
    assert not np.array_equal(_normalize_mask(hole), _normalize_mask(query_entry["shape_mask"])), "hole orientation must differ from the matched legend shape"


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = np.random.default_rng()
        built = _build_scene(rng)
        if built is None:
            continue
        inp, meta = built
        try:
            _validate_scene(inp, meta)
        except AssertionError:
            continue
        return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
