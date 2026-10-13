from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components


BG = 0
TARGET_COLOR = 4
MARKER_COLOR = 5
SOURCE_COLORS = (1, 2, 3)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GRID_H_RANGE = (16, 20)
GRID_W_RANGE = (22, 28)
TARGET_H_RANGE = (6, 8)
TARGET_W_RANGE = (6, 8)
SOURCE_H_RANGE = (3, 5)
SOURCE_W_RANGE = (3, 5)
MARKER_COUNT_RANGE = (2, 5)
MAX_GENERATE_ATTEMPTS = 500


def _component_colors(grid: np.ndarray, coords: np.ndarray) -> set[int]:
    return set(map(int, np.unique(grid[tuple(coords.T)])))


def _is_dense_rectangle(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    return bool(mask[r0 : r1 + 1, c0 : c1 + 1].all())


def _domino_mode(coords: list[tuple[int, int]]) -> str:
    assert len(coords) == 2, "domino must have exactly two cells"
    (r0, c0), (r1, c1) = sorted((tuple(map(int, xy)) for xy in coords))
    if r0 == r1 and abs(c0 - c1) == 1:
        return "row"
    if c0 == c1 and abs(r0 - r1) == 1:
        return "col"
    raise AssertionError("domino must be horizontal or vertical")


def _touches_rectangle(domino: list[tuple[int, int]], rect_bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = rect_bbox
    for r, c in domino:
        for dr, dc in DIR4:
            rr, cc = r + dr, c + dc
            if r0 <= rr <= r1 and c0 <= cc <= c1:
                return True
    return False


def _scaled_index(idx: int, src_len: int, dst_len: int) -> int:
    if src_len <= 1:
        return 0
    return int(round(idx * (dst_len - 1) / (src_len - 1)))


def _edge_preserved_index(idx: int, src_len: int, dst_len: int) -> int:
    top = int(idx)
    bottom = int(src_len - 1 - idx)
    if top < bottom:
        return top
    if bottom < top:
        return int(dst_len - 1 - bottom)
    return _scaled_index(idx, src_len, dst_len)


def _map_pattern(
    grey_rel: list[tuple[int, int]],
    src_shape: tuple[int, int],
    dst_shape: tuple[int, int],
    mode: str,
) -> set[tuple[int, int]]:
    src_h, src_w = map(int, src_shape)
    dst_h, dst_w = map(int, dst_shape)
    mapped: set[tuple[int, int]] = set()
    for rr, cc in grey_rel:
        if mode == "row":
            tr = _edge_preserved_index(rr, src_h, dst_h)
            tc = _scaled_index(cc, src_w, dst_w)
        else:
            tr = _scaled_index(rr, src_h, dst_h)
            tc = _edge_preserved_index(cc, src_w, dst_w)
        mapped.add((int(tr), int(tc)))
    return mapped


def _parse_source_component(grid: np.ndarray, comp: dict, bg: int) -> dict:
    bbox = tuple(map(int, comp["bbox"]))
    sub = crop_bbox(grid, bbox)
    mask = np.zeros(sub.shape, dtype=bool)
    rel_coords = comp["coords"] - np.array([[bbox[0], bbox[1]]], dtype=int)
    mask[tuple(rel_coords.T)] = True

    colors = _component_colors(grid, comp["coords"]) - {int(bg)}
    assert MARKER_COLOR in colors and len(colors) == 2, "source component must use base color + grey"
    base_color = int(next(iter(colors - {MARKER_COLOR})))

    total = int(mask.sum())
    candidates: list[dict] = []
    sh, sw = sub.shape
    for r0 in range(sh):
        for r1 in range(r0 + 2, sh):
            for c0 in range(sw):
                for c1 in range(c0 + 2, sw):
                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    if area + 2 != total:
                        continue
                    if not _is_dense_rectangle(mask, r0, c0, r1, c1):
                        continue
                    outside = np.argwhere(mask & ~(
                        np.pad(
                            np.ones((r1 - r0 + 1, c1 - c0 + 1), dtype=bool),
                            ((r0, sh - r1 - 1), (c0, sw - c1 - 1)),
                            constant_values=False,
                        )
                    ))
                    if outside.shape[0] != 2:
                        continue
                    outside_cells = [tuple(map(int, rc)) for rc in outside]
                    if any(int(sub[r, c]) != base_color for r, c in outside_cells):
                        continue
                    try:
                        mode = _domino_mode(outside_cells)
                    except AssertionError:
                        continue
                    if not _touches_rectangle(outside_cells, (r0, c0, r1, c1)):
                        continue

                    rect = sub[r0 : r1 + 1, c0 : c1 + 1]
                    grey_rel = [tuple(map(int, rc)) for rc in np.argwhere(rect == MARKER_COLOR)]
                    if not grey_rel:
                        continue

                    candidates.append(
                        {
                            "base_color": base_color,
                            "mode": mode,
                            "rect_bbox": (bbox[0] + r0, bbox[1] + c0, bbox[0] + r1, bbox[1] + c1),
                            "shape": rect.shape,
                            "grey_rel": sorted(grey_rel),
                        }
                    )

    assert len(candidates) == 1, "source component must contain one rectangle plus one adjacent domino"
    return candidates[0]


def _extract_scene(grid: np.ndarray) -> tuple[dict, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = BG
    components = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert components, "expected non-background components"

    target = None
    sources: list[dict] = []
    for comp in components:
        colors = _component_colors(g, comp["coords"]) - {bg}
        bbox = tuple(map(int, comp["bbox"]))
        sub = crop_bbox(g, bbox)
        if colors == {TARGET_COLOR}:
            assert np.all(sub == TARGET_COLOR), "target must be a solid yellow rectangle"
            assert target is None, "expected exactly one target rectangle"
            target = {"bbox": bbox, "shape": sub.shape}
        else:
            sources.append(_parse_source_component(g, comp, bg))

    assert target is not None, "expected one plain yellow target rectangle"
    assert 2 <= len(sources) <= 3, "expected 2-3 marked source rectangles"
    return target, sources


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the plain yellow rectangle and refill it with grey marker pixels transferred from source rectangles according to adjacent domino orientation.

    Input structure:
    - The background is black.
    - There are 2-3 source components, each formed by one colored dense rectangle containing grey marker pixels plus one same-color adjacent 2-pixel domino outside the rectangle.
    - There is exactly one plain yellow rectangle with no grey marker pixels; it is the target and is at least as tall and wide as every source rectangle.

    Rule:
    - Identify the yellow target rectangle and crop it as the output canvas, initially filled with yellow.
    - For each source component, detect the domino orientation after separating the dense rectangle from the external domino.
    - A horizontal domino means row migration: preserve each grey pixel's top/bottom edge-relative row and map its column proportionally into the target width; a vertical domino means column migration: preserve its left/right edge-relative column and map its row proportionally into the target height.
    - Draw all column-migration grey pixels first, then all row-migration grey pixels, on the yellow crop.

    Key insight:
    - The domino does not point along the travel path; instead its orientation chooses which dimension keeps edge-relative margins and which dimension is resized proportionally.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    target, sources = _extract_scene(g)
    out = np.full(target["shape"], TARGET_COLOR, dtype=int)

    column_first = [src for src in sources if src["mode"] == "col"]
    row_second = [src for src in sources if src["mode"] == "row"]
    for src in column_first + row_second:
        mapped = _map_pattern(src["grey_rel"], src["shape"], target["shape"], src["mode"])
        for r, c in mapped:
            out[r, c] = MARKER_COLOR
    return out


def _allowed_marker_cells(src_h: int, src_w: int, mode: str) -> list[tuple[int, int]]:
    inner_rows = list(range(1, src_h - 1))
    inner_cols = list(range(1, src_w - 1))
    if mode == "row" and src_h % 2 == 1:
        center = src_h // 2
        inner_rows = [r for r in inner_rows if r != center]
    if mode == "col" and src_w % 2 == 1:
        center = src_w // 2
        inner_cols = [c for c in inner_cols if c != center]
    return [(r, c) for r in inner_rows for c in inner_cols]


def _sample_markers(
    src_h: int,
    src_w: int,
    dst_shape: tuple[int, int],
    mode: str,
    *,
    required: list[tuple[int, int]] | None = None,
) -> tuple[list[tuple[int, int]], set[tuple[int, int]]] | None:
    cells = _allowed_marker_cells(src_h, src_w, mode)
    if len(cells) < 2:
        return None
    required = [] if required is None else sorted(set(required))
    if any(cell not in cells for cell in required):
        return None
    avail_rows = {r for r, _ in cells}
    avail_cols = {c for _, c in cells}

    for _ in range(100):
        min_k = max(MARKER_COUNT_RANGE[0], len(required))
        max_k = min(MARKER_COUNT_RANGE[1], len(cells))
        if min_k > max_k:
            return None
        k = random.randint(min_k, max_k)
        extra_pool = [cell for cell in cells if cell not in required]
        extra_n = k - len(required)
        chosen = sorted(required + random.sample(extra_pool, extra_n))
        rows = {r for r, _ in chosen}
        cols = {c for _, c in chosen}
        if len(rows) < min(2, len(avail_rows)):
            continue
        if len(cols) < min(2, len(avail_cols)):
            continue
        mapped = _map_pattern(chosen, (src_h, src_w), dst_shape, mode)
        other = _map_pattern(chosen, (src_h, src_w), dst_shape, "col" if mode == "row" else "row")
        if len(mapped) < 2 or mapped == other:
            continue
        return chosen, mapped
    return None


def _choose_source_specs(target_shape: tuple[int, int]) -> list[dict] | None:
    target_h, target_w = map(int, target_shape)

    def sample_shape_spec(color: int, mode: str) -> dict | None:
        min_h = 4 if mode == "row" else SOURCE_H_RANGE[0]
        min_w = 4 if mode == "col" else SOURCE_W_RANGE[0]
        max_h = min(SOURCE_H_RANGE[1], target_h)
        max_w = min(SOURCE_W_RANGE[1], target_w)
        if min_h > max_h or min_w > max_w:
            return None
        for _ in range(100):
            src_h = random.randint(min_h, max_h)
            src_w = random.randint(min_w, max_w)
            allowed = _allowed_marker_cells(src_h, src_w, mode)
            if len(allowed) < 2:
                continue
            cell_map = {
                cell: next(iter(_map_pattern([cell], (src_h, src_w), target_shape, mode)))
                for cell in allowed
            }
            return {
                "color": int(color),
                "mode": mode,
                "shape": (src_h, src_w),
                "allowed": allowed,
                "cell_map": cell_map,
            }
        return None

    n_sources = 3 if random.random() < 0.35 else 2
    modes = ["row", "col"]
    if n_sources == 3:
        modes.append(random.choice(("row", "col")))
    random.shuffle(modes)

    chosen_colors = random.sample(SOURCE_COLORS, n_sources)
    extra_mode = random.choice(("row", "col")) if n_sources == 3 else None
    _ = extra_mode  # branch probability is encoded by `modes`

    for _attempt in range(200):
        specs: list[dict] = []
        for color, mode in zip(chosen_colors, modes):
            spec = sample_shape_spec(color, mode)
            if spec is None:
                break
            sampled = _sample_markers(*spec["shape"], target_shape, mode)
            if sampled is None:
                break
            spec["grey_rel"], spec["mapped"] = sampled
            specs.append(spec)
        if len(specs) != n_sources:
            continue

        row_union = set().union(*(spec["mapped"] for spec in specs if spec["mode"] == "row"))
        col_union = set().union(*(spec["mapped"] for spec in specs if spec["mode"] == "col"))
        if not row_union or not col_union:
            continue
        if len(row_union | col_union) < 3:
            continue
        for spec in specs:
            spec.pop("allowed", None)
            spec.pop("cell_map", None)
        return specs
    return None


def _component_cells(
    rect_top: int,
    rect_left: int,
    shape: tuple[int, int],
    mode: str,
    side: str,
    offset: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = map(int, shape)
    rect_cells = [(rect_top + r, rect_left + c) for r in range(h) for c in range(w)]
    r1 = rect_top + h - 1
    c1 = rect_left + w - 1

    if mode == "row":
        if side == "top":
            domino = [(rect_top - 1, rect_left + offset), (rect_top - 1, rect_left + offset + 1)]
        elif side == "bottom":
            domino = [(r1 + 1, rect_left + offset), (r1 + 1, rect_left + offset + 1)]
        elif side == "left":
            domino = [(rect_top + offset, rect_left - 2), (rect_top + offset, rect_left - 1)]
        else:
            domino = [(rect_top + offset, c1 + 1), (rect_top + offset, c1 + 2)]
    else:
        if side == "top":
            domino = [(rect_top - 2, rect_left + offset), (rect_top - 1, rect_left + offset)]
        elif side == "bottom":
            domino = [(r1 + 1, rect_left + offset), (r1 + 2, rect_left + offset)]
        elif side == "left":
            domino = [(rect_top + offset, rect_left - 1), (rect_top + offset + 1, rect_left - 1)]
        else:
            domino = [(rect_top + offset, c1 + 1), (rect_top + offset + 1, c1 + 1)]
    return rect_cells, domino


def _all_in_bounds(cells: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = map(int, shape)
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _has_clearance(cells: list[tuple[int, int]], occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    for r, c in cells:
        if occupied[r, c]:
            return False
        for dr, dc in DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and occupied[rr, cc]:
                return False
    return True


def _mark_occupied(occupied: np.ndarray, cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        occupied[r, c] = True


def _unmark_occupied(occupied: np.ndarray, cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        occupied[r, c] = False


def _row_overlap(top_a: int, h_a: int, top_b: int, h_b: int) -> bool:
    return max(top_a, top_b) <= min(top_a + h_a - 1, top_b + h_b - 1)


def _col_overlap(left_a: int, w_a: int, left_b: int, w_b: int) -> bool:
    return max(left_a, left_b) <= min(left_a + w_a - 1, left_b + w_b - 1)


def _sample_target(grid_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    grid_h, grid_w = map(int, grid_shape)
    target_h = random.randint(TARGET_H_RANGE[0], min(TARGET_H_RANGE[1], grid_h - 4))
    target_w = random.randint(TARGET_W_RANGE[0], min(TARGET_W_RANGE[1], grid_w - 4))
    center_top = (grid_h - target_h) // 2
    center_left = (grid_w - target_w) // 2
    top_lo = max(2, center_top - 2)
    top_hi = min(grid_h - target_h - 2, center_top + 2)
    left_lo = max(3, center_left - 3)
    left_hi = min(grid_w - target_w - 3, center_left + 3)
    if top_lo > top_hi or left_lo > left_hi:
        raise RuntimeError("grid too small for target placement")
    top = random.randint(top_lo, top_hi)
    left = random.randint(left_lo, left_hi)
    return top, left, target_h, target_w


def _place_sources(grid: np.ndarray, target_rect: tuple[int, int, int, int], specs: list[dict]) -> bool:
    target_top, target_left, target_h, target_w = target_rect
    occupied = np.zeros_like(grid, dtype=bool)
    _mark_occupied(
        occupied,
        [
            (target_top + r, target_left + c)
            for r in range(target_h)
            for c in range(target_w)
        ],
    )

    def side_options(spec: dict) -> list[tuple[str, int]]:
        src_h, src_w = spec["shape"]
        opts: list[tuple[str, int]] = []
        if spec["mode"] == "row":
            opts.extend([("top", i) for i in range(src_w - 1)])
            opts.extend([("bottom", i) for i in range(src_w - 1)])
            opts.extend([("left", i) for i in range(src_h)])
            opts.extend([("right", i) for i in range(src_h)])
        else:
            opts.extend([("top", i) for i in range(src_w)])
            opts.extend([("bottom", i) for i in range(src_w)])
            opts.extend([("left", i) for i in range(src_h - 1)])
            opts.extend([("right", i) for i in range(src_h - 1)])
        random.shuffle(opts)
        return opts

    def paint(spec: dict, placement: dict) -> None:
        for r, c in placement["rect_cells"]:
            grid[r, c] = spec["color"]
        for rr, cc in spec["grey_rel"]:
            grid[placement["rect_top"] + rr, placement["rect_left"] + cc] = MARKER_COLOR
        for r, c in placement["domino"]:
            grid[r, c] = spec["color"]
        _mark_occupied(occupied, placement["all_cells"])

    def erase(placement: dict) -> None:
        for r, c in placement["all_cells"]:
            grid[r, c] = BG
        _unmark_occupied(occupied, placement["all_cells"])

    def slot_placements(spec: dict, slot: str) -> list[dict]:
        src_h, src_w = spec["shape"]
        gap_values = [2, 3, 4]
        random.shuffle(gap_values)
        placements: list[dict] = []
        if slot in {"left", "right"}:
            overlap_values = list(range(1, min(src_h, target_h) + 1))
            band_modes = ["top", "bottom"]
            random.shuffle(overlap_values)
            random.shuffle(band_modes)
            for gap in gap_values:
                rect_left = target_left - gap - src_w if slot == "left" else target_left + target_w + gap
                if rect_left < 1 or rect_left + src_w >= grid.shape[1] - 1:
                    continue
                for band in band_modes:
                    for overlap in overlap_values:
                        rect_top = target_top - src_h + overlap if band == "top" else target_top + target_h - overlap
                        if not (1 <= rect_top and rect_top + src_h < grid.shape[0] - 1):
                            continue
                        for side, offset in side_options(spec):
                            rect_cells, domino = _component_cells(rect_top, rect_left, spec["shape"], spec["mode"], side, offset)
                            all_cells = rect_cells + domino
                            if not _all_in_bounds(all_cells, grid.shape):
                                continue
                            placements.append(
                                {
                                    "rect_top": rect_top,
                                    "rect_left": rect_left,
                                    "rect_cells": rect_cells,
                                    "domino": domino,
                                    "all_cells": all_cells,
                                }
                            )
        else:
            overlap_values = list(range(1, min(src_w, target_w) + 1))
            band_modes = ["left", "right"]
            random.shuffle(overlap_values)
            random.shuffle(band_modes)
            for gap in gap_values:
                rect_top = target_top - gap - src_h if slot == "top" else target_top + target_h + gap
                if rect_top < 1 or rect_top + src_h >= grid.shape[0] - 1:
                    continue
                for band in band_modes:
                    for overlap in overlap_values:
                        rect_left = target_left - src_w + overlap if band == "left" else target_left + target_w - overlap
                        if not (1 <= rect_left and rect_left + src_w < grid.shape[1] - 1):
                            continue
                        for side, offset in side_options(spec):
                            rect_cells, domino = _component_cells(rect_top, rect_left, spec["shape"], spec["mode"], side, offset)
                            all_cells = rect_cells + domino
                            if not _all_in_bounds(all_cells, grid.shape):
                                continue
                            placements.append(
                                {
                                    "rect_top": rect_top,
                                    "rect_left": rect_left,
                                    "rect_cells": rect_cells,
                                    "domino": domino,
                                    "all_cells": all_cells,
                                }
                            )
        random.shuffle(placements)
        return placements

    ordered = sorted(
        specs,
        key=lambda s: len(s["mapped"]) * max(s["shape"]),
        reverse=True,
    )
    placement_cache = {
        (id(spec), slot): slot_placements(spec, slot)
        for spec in ordered
        for slot in (["left", "right"] if spec["mode"] == "row" else ["top", "bottom"])
    }

    def backtrack(i: int, row_slots: list[str], col_slots: list[str]) -> bool:
        if i == len(ordered):
            return True
        spec = ordered[i]
        slots = row_slots if spec["mode"] == "row" else col_slots
        slot_order = random.sample(slots, len(slots))
        for slot in slot_order:
            for placement in placement_cache[(id(spec), slot)]:
                if not _has_clearance(placement["all_cells"], occupied):
                    continue
                paint(spec, placement)
                next_row_slots = [s for s in row_slots if s != slot] if spec["mode"] == "row" else row_slots
                next_col_slots = [s for s in col_slots if s != slot] if spec["mode"] == "col" else col_slots
                if backtrack(i + 1, next_row_slots, next_col_slots):
                    return True
                erase(placement)
        return False

    return backtrack(0, ["left", "right"], ["top", "bottom"])


def _expected_output(target_shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    out = np.full(target_shape, TARGET_COLOR, dtype=int)
    for mode in ("col", "row"):
        for spec in specs:
            if spec["mode"] != mode:
                continue
            for r, c in spec["mapped"]:
                out[r, c] = MARKER_COLOR
    return out


def _slot_side_choices(mode: str, slot: str) -> list[str]:
    if mode == "row":
        return {
            "left": ["left", "top", "bottom"],
            "right": ["right", "top", "bottom"],
        }[slot]
    return {
        "top": ["top", "left", "right"],
        "bottom": ["bottom", "left", "right"],
    }[slot]


def _overlap_top(target_top: int, target_h: int, src_h: int, overlap: int) -> int:
    return target_top - src_h + overlap


def _overlap_bottom(target_top: int, target_h: int, overlap: int) -> int:
    return target_top + target_h - overlap


def _overlap_left(target_left: int, target_w: int, src_w: int, overlap: int) -> int:
    return target_left - src_w + overlap


def _overlap_right(target_left: int, target_w: int, overlap: int) -> int:
    return target_left + target_w - overlap


def _choose_domino_for_slot(spec: dict, rect_top: int, rect_left: int, slot: str) -> list[tuple[int, int]] | None:
    mode = spec["mode"]
    h, w = spec["shape"]
    side_order = _slot_side_choices(mode, slot)
    random.shuffle(side_order)
    for side in side_order:
        if side in {"top", "bottom"}:
            offsets = list(range(w - 1)) if mode == "row" else list(range(w))
        else:
            offsets = list(range(h)) if mode == "row" else list(range(h - 1))
        random.shuffle(offsets)
        for offset in offsets:
            _rect_cells, domino = _component_cells(rect_top, rect_left, spec["shape"], mode, side, offset)
            return domino
    return None


def _build_layout(specs: list[dict], target_shape: tuple[int, int]) -> np.ndarray | None:
    target_h, target_w = map(int, target_shape)
    row_specs = [dict(spec) for spec in specs if spec["mode"] == "row"]
    col_specs = [dict(spec) for spec in specs if spec["mode"] == "col"]
    for _ in range(80):
        row_slots = ["left", "right"] if len(row_specs) == 2 else [random.choice(["left", "right"])]
        col_slots = ["top", "bottom"] if len(col_specs) == 2 else [random.choice(["top", "bottom"])]
        random.shuffle(row_slots)
        random.shuffle(col_slots)

        left_band = max((spec["shape"][1] + 2 for spec, slot in zip(row_specs, row_slots) if slot == "left"), default=0)
        right_band = max((spec["shape"][1] + 2 for spec, slot in zip(row_specs, row_slots) if slot == "right"), default=0)
        top_band = max((spec["shape"][0] + 2 for spec, slot in zip(col_specs, col_slots) if slot == "top"), default=0)
        bottom_band = max((spec["shape"][0] + 2 for spec, slot in zip(col_specs, col_slots) if slot == "bottom"), default=0)

        gap = 1
        grid_h = 1 + top_band + (gap if top_band else 0) + target_h + (gap if bottom_band else 0) + bottom_band + 1
        grid_w = 1 + left_band + (gap if left_band else 0) + target_w + (gap if right_band else 0) + right_band + 1
        if grid_h > 30 or grid_w > 30:
            continue

        target_top = 1 + top_band + (gap if top_band else 0)
        target_left = 1 + left_band + (gap if left_band else 0)
        grid = np.full((grid_h, grid_w), BG, dtype=int)
        grid[target_top : target_top + target_h, target_left : target_left + target_w] = TARGET_COLOR

        target_bottom = target_top + target_h - 1
        target_right = target_left + target_w - 1
        occupied = np.zeros_like(grid, dtype=bool)
        occupied[target_top : target_top + target_h, target_left : target_left + target_w] = True

        top_col_exists = "top" in col_slots
        left_row_exists = "left" in row_slots
        row_bands = ["bottom" if top_col_exists else "top", "top" if top_col_exists else "bottom"]
        col_bands = ["right" if left_row_exists else "left", "left" if left_row_exists else "right"]

        placements: list[tuple[dict, int, int, list[tuple[int, int]]]] = []
        failed = False

        for idx, (spec, slot) in enumerate(zip(row_specs, row_slots)):
            src_h, src_w = spec["shape"]
            rect_left = target_left - gap - src_w if slot == "left" else target_right + gap + 1
            overlap = random.randint(1, min(src_h, target_h))
            band = row_bands[min(idx, len(row_bands) - 1)]
            rect_top = _overlap_top(target_top, target_h, src_h, overlap) if band == "top" else _overlap_bottom(target_top, target_h, overlap)
            domino = _choose_domino_for_slot(spec, rect_top, rect_left, slot)
            if domino is None:
                failed = True
                break
            all_cells = [(rect_top + r, rect_left + c) for r in range(src_h) for c in range(src_w)] + domino
            if not _all_in_bounds(all_cells, grid.shape) or not _has_clearance(all_cells, occupied):
                failed = True
                break
            placements.append((spec, rect_top, rect_left, domino))
            _mark_occupied(occupied, all_cells)

        if failed:
            continue

        for idx, (spec, slot) in enumerate(zip(col_specs, col_slots)):
            src_h, src_w = spec["shape"]
            rect_top = target_top - gap - src_h if slot == "top" else target_bottom + gap + 1
            overlap = random.randint(1, min(src_w, target_w))
            band = col_bands[min(idx, len(col_bands) - 1)]
            rect_left = _overlap_left(target_left, target_w, src_w, overlap) if band == "left" else _overlap_right(target_left, target_w, overlap)
            domino = _choose_domino_for_slot(spec, rect_top, rect_left, slot)
            if domino is None:
                failed = True
                break
            all_cells = [(rect_top + r, rect_left + c) for r in range(src_h) for c in range(src_w)] + domino
            if not _all_in_bounds(all_cells, grid.shape) or not _has_clearance(all_cells, occupied):
                failed = True
                break
            placements.append((spec, rect_top, rect_left, domino))
            _mark_occupied(occupied, all_cells)

        if failed:
            continue

        for spec, rect_top, rect_left, domino in placements:
            src_h, src_w = spec["shape"]
            grid[rect_top : rect_top + src_h, rect_left : rect_left + src_w] = spec["color"]
            for rr, cc in spec["grey_rel"]:
                grid[rect_top + rr, rect_left + cc] = MARKER_COLOR
            for r, c in domino:
                grid[r, c] = spec["color"]
        return grid
    return None


def _validate_generated(grid: np.ndarray, expected: np.ndarray, specs: list[dict]) -> bool:
    if not np.array_equal(main(grid), expected):
        return False

    try:
        target, parsed_sources = _extract_scene(grid)
    except AssertionError:
        return False

    if tuple(map(int, target["shape"])) != tuple(map(int, expected.shape)):
        return False
    if {src["mode"] for src in parsed_sources} != {"row", "col"}:
        return False

    row_union = set().union(*(_map_pattern(src["grey_rel"], src["shape"], target["shape"], src["mode"]) for src in parsed_sources if src["mode"] == "row"))
    col_union = set().union(*(_map_pattern(src["grey_rel"], src["shape"], target["shape"], src["mode"]) for src in parsed_sources if src["mode"] == "col"))
    if not row_union or not col_union:
        return False
    if len(row_union | col_union) < 3:
        return False

    for spec in specs:
        wrong = _map_pattern(spec["grey_rel"], spec["shape"], expected.shape, "col" if spec["mode"] == "row" else "row")
        if wrong == spec["mapped"]:
            return False
    return True


def _generate_attempt() -> np.ndarray | None:
    target_shape = (
        random.randint(*TARGET_H_RANGE),
        random.randint(*TARGET_W_RANGE),
    )
    specs = _choose_source_specs(target_shape)
    if specs is None:
        return None

    grid = _build_layout(specs, target_shape)
    if grid is None:
        return None

    expected = _expected_output(target_shape, specs)
    if not _validate_generated(grid, expected, specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
