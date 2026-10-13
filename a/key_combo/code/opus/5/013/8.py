from __future__ import annotations

import math
import random

import numpy as np


BG = 0
BORDER = 5
EXAMPLE = 6
ACTIVE_COLORS = (1, 2, 3)

RATIO_SQUARE = (1, 1)
RATIO_TALL = (1, 2)
RATIO_WIDE = (2, 1)
RATIOS = (RATIO_SQUARE, RATIO_TALL, RATIO_WIDE)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_center(coords: np.ndarray) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox_coords(coords)
    return int(round((r0 + r1) / 2.0)), int(round((c0 + c1) / 2.0))


def _coords_to_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in coords}


def _translate_set(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells}


def _ratio_from_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    g = math.gcd(h, w)
    return int(w // g), int(h // g)


def _all_rect_cells(h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(h) for c in range(w)}


def _transform_coords(
    coords: set[tuple[int, int]],
    h: int,
    w: int,
    *,
    flip_r: bool = False,
    flip_c: bool = False,
) -> set[tuple[int, int]]:
    out = set()
    for r, c in coords:
        rr = h - 1 - r if flip_r else r
        cc = w - 1 - c if flip_c else c
        out.add((int(rr), int(cc)))
    return out


def _prototype_for_ratio(ratio: tuple[int, int]) -> dict:
    if ratio == RATIO_SQUARE:
        h, w = 3, 3
        source = {(0, 0), (0, 1), (1, 0)}
        source = _transform_coords(
            source,
            h,
            w,
            flip_r=random.choice([False, True]),
            flip_c=random.choice([False, True]),
        )
    elif ratio == RATIO_TALL:
        h, w = 4, 2
        row0 = random.choice([0, 1])
        col = random.choice([0, 1])
        anchor = {(row0 + k, col) for k in range(3)}
        source = _all_rect_cells(h, w) - anchor
        return {
            "ratio": ratio,
            "rect_h": h,
            "rect_w": w,
            "source": source,
            "anchor": anchor,
        }
    elif ratio == RATIO_WIDE:
        h, w = 2, 4
        col0 = random.choice([0, 1])
        row = random.choice([0, 1])
        anchor = {(row, col0 + k) for k in range(3)}
        source = _all_rect_cells(h, w) - anchor
        return {
            "ratio": ratio,
            "rect_h": h,
            "rect_w": w,
            "source": source,
            "anchor": anchor,
        }
    else:
        raise ValueError("unsupported ratio")

    anchor = _all_rect_cells(h, w) - source
    return {
        "ratio": ratio,
        "rect_h": h,
        "rect_w": w,
        "source": source,
        "anchor": anchor,
    }


def _glyph_cells(ratio: tuple[int, int]) -> list[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    if ratio == RATIO_SQUARE:
        for i in range(5):
            cells.add((2, i))
            cells.add((i, 2))
    elif ratio == RATIO_TALL:
        for i in range(5):
            cells.add((0, i))
            cells.add((i, 2))
    elif ratio == RATIO_WIDE:
        for i in range(5):
            cells.add((i, 0))
            cells.add((4, i))
    else:
        raise ValueError("unsupported glyph ratio")
    return sorted(cells)


def _draw_glyph(out: np.ndarray, center: tuple[int, int], color: int, ratio: tuple[int, int]) -> None:
    h, w = out.shape
    cr, cc = center
    for gr, gc in _glyph_cells(ratio):
        rr = cr + gr - 2
        cc2 = cc + gc - 2
        if 1 <= rr < h - 1 and 1 <= cc2 < w - 1:
            out[rr, cc2] = int(color)


def _completion_candidates(
    grid: np.ndarray,
    color: int,
    source: np.ndarray,
    anchor: np.ndarray,
) -> list[dict]:
    h, w = grid.shape
    src = _coords_to_set(source)
    anc = _coords_to_set(anchor)
    candidates: list[dict] = []

    for dr in range(1 - h, h):
        for dc in range(1 - w, w):
            moved = _translate_set(src, dr, dc)
            if any(r < 1 or r >= h - 1 or c < 1 or c >= w - 1 for r, c in moved):
                continue
            if moved & anc:
                continue
            if any(int(grid[r, c]) != BG for r, c in moved):
                continue

            union = moved | anc
            rows = [r for r, _ in union]
            cols = [c for _, c in union]
            bbox = (min(rows), min(cols), max(rows), max(cols))
            area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
            if len(union) != area:
                continue
            ratio = _ratio_from_bbox(bbox)
            if ratio not in RATIOS:
                continue
            candidates.append(
                {
                    "color": int(color),
                    "dr": int(dr),
                    "dc": int(dc),
                    "bbox": bbox,
                    "ratio": ratio,
                }
            )
    return candidates


def _extract_groups(grid: np.ndarray) -> list[dict]:
    groups: list[dict] = []
    for color in ACTIVE_COLORS:
        comps = _components_mask(grid == int(color))
        if not comps:
            continue
        assert len(comps) == 2, f"color {color} must form exactly two components"
        source_parts = [comp for comp in comps if np.any(comp[:, 1] == 1)]
        anchor_parts = [comp for comp in comps if not np.any(comp[:, 1] == 1)]
        assert len(source_parts) == 1, f"color {color} must have one left-edge fragment"
        assert len(anchor_parts) == 1, f"color {color} must have one anchor fragment"

        source = source_parts[0]
        anchor = anchor_parts[0]
        candidates = _completion_candidates(grid, color, source, anchor)
        assert len(candidates) == 1, f"color {color} must have one unique rectangle completion"

        groups.append(
            {
                "color": int(color),
                "source": source,
                "anchor": anchor,
                "center": _bbox_center(anchor),
                "ratio": candidates[0]["ratio"],
                "completion_bbox": candidates[0]["bbox"],
            }
        )

    ratios = [g["ratio"] for g in groups]
    assert len(groups) in (2, 3), "expected two or three fragment pairs"
    assert len(set(ratios)) == len(ratios), "completion ratios must be pairwise distinct"
    return groups


def _example_component(grid: np.ndarray) -> np.ndarray:
    comps = _components_mask(grid == EXAMPLE)
    assert len(comps) == 1, "expected exactly one magenta example glyph"
    return comps[0]


def _render_output(grid: np.ndarray, groups: list[dict], example: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    for rr, cc in example:
        out[int(rr), int(cc)] = BG
    for group in groups:
        for rr, cc in group["source"]:
            out[int(rr), int(cc)] = BG
        for rr, cc in group["anchor"]:
            out[int(rr), int(cc)] = BG
    for group in groups:
        _draw_glyph(out, group["center"], group["color"], group["ratio"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild color glyphs by using same-color fragment pairs whose rectangle-completion ratio selects the template.

    Input structure:
    - A black (0) field is enclosed by a one-pixel grey (5) border.
    - Two or three colors from blue/red/green each appear as exactly two 4-connected fragments: one fragment touches the inner left edge and the other is an interior anchor fragment.
    - A separate magenta (6) 5x5 glyph example sits on the right side as a removable reference.

    Rule:
    - For each active color, translate the left-edge fragment by the unique offset that makes it complement the anchor fragment into one filled rectangle.
    - Reduce that rectangle's width:height ratio and map it to a 5x5 glyph: 1:1 -> plus, 1:2 -> T, 2:1 -> L.
    - Erase both original fragments of that color and erase the magenta example, then draw the selected glyph in the fragment's color with the anchor fragment bbox center as the glyph center; clip any glyph pixels that would leave the interior.

    Key insight:
    - The fragments are not reconstructed for their own sake: their hidden rectangle ratio is the code that chooses which glyph to redraw at the anchor position.
    """
    g = np.asarray(grid, dtype=int)
    example = _example_component(g)
    groups = _extract_groups(g)
    return _render_output(g, groups, example)


def _source_abs_coords(proto: dict, row_shift: int) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in proto["source"])
    min_c = min(c for _, c in proto["source"])
    dr = int(row_shift - min_r)
    dc = int(1 - min_c)
    return _translate_set(proto["source"], dr, dc)


def _anchor_abs_coords(proto: dict, rect_top: int, rect_left: int) -> set[tuple[int, int]]:
    return _translate_set(proto["anchor"], rect_top, rect_left)


def _anchor_center_abs(proto: dict, rect_top: int, rect_left: int) -> tuple[int, int]:
    anchor = np.array(sorted(_anchor_abs_coords(proto, rect_top, rect_left)), dtype=int)
    return _bbox_center(anchor)


def _mark_box(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 0) -> None:
    h, w = mask.shape
    rr0 = max(0, int(r0) - int(pad))
    cc0 = max(0, int(c0) - int(pad))
    rr1 = min(h - 1, int(r1) + int(pad))
    cc1 = min(w - 1, int(c1) + int(pad))
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _oracle_from_metadata(grid: np.ndarray, groups: list[dict], example_cells: set[tuple[int, int]]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for r, c in example_cells:
        out[r, c] = BG
    for group in groups:
        for r, c in group["source_cells"]:
            out[r, c] = BG
        for r, c in group["anchor_cells"]:
            out[r, c] = BG
        _draw_glyph(out, group["center"], group["color"], group["ratio"])
    return out


def _generate_case() -> tuple[np.ndarray, list[dict], set[tuple[int, int]]] | None:
    n_groups = random.choice([2, 2, 3, 3])
    ratios = random.sample(RATIOS, n_groups)
    colors = random.sample(list(ACTIVE_COLORS), n_groups)
    protos = [_prototype_for_ratio(ratio) for ratio in ratios]

    if n_groups == 2:
        h = random.randint(16, 22)
        w = random.randint(17, 22)
    else:
        h = random.randint(20, 24)
        w = random.randint(21, 25)
    grid = np.full((h, w), BG, dtype=int)
    grid[0, :] = BORDER
    grid[-1, :] = BORDER
    grid[:, 0] = BORDER
    grid[:, -1] = BORDER

    reserve = np.zeros((h, w), dtype=bool)
    reserve[0, :] = True
    reserve[-1, :] = True
    reserve[:, 0] = True
    reserve[:, -1] = True

    example_ratio = random.choice(ratios)
    example_top = random.randint(1, h - 6)
    example_left = w - 6
    example_cells = {
        (example_top + r, example_left + c) for r, c in _glyph_cells(example_ratio)
    }
    for r, c in example_cells:
        grid[r, c] = EXAMPLE
    _mark_box(reserve, example_top, example_left, example_top + 4, example_left + 4, pad=1)

    groups_meta: list[dict] = []

    anchor_order = list(range(n_groups))
    random.shuffle(anchor_order)
    for idx in anchor_order:
        proto = protos[idx]
        rect_h = proto["rect_h"]
        rect_w = proto["rect_w"]
        color = colors[idx]

        candidates: list[tuple[int, int, tuple[int, int]]] = []
        c_max = example_left - rect_w - 2
        if c_max < 2:
            return None
        for rect_top in range(1, h - rect_h):
            for rect_left in range(2, c_max + 1):
                anchor_cells = _anchor_abs_coords(proto, rect_top, rect_left)
                if any(r in (1, h - 2) or c in (1, w - 2) for r, c in anchor_cells):
                    continue
                rect_box = (rect_top, rect_left, rect_top + rect_h - 1, rect_left + rect_w - 1)
                center = _anchor_center_abs(proto, rect_top, rect_left)
                glyph_box = (center[0] - 2, center[1] - 2, center[0] + 2, center[1] + 2)

                blocked = False
                for r, c in anchor_cells:
                    if reserve[r, c]:
                        blocked = True
                        break
                if blocked:
                    continue

                rr0 = max(0, rect_box[0] - 1)
                cc0 = max(0, rect_box[1] - 1)
                rr1 = min(h - 1, rect_box[2] + 1)
                cc1 = min(w - 1, rect_box[3] + 1)
                if reserve[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                    continue

                gr0 = max(0, glyph_box[0])
                gc0 = max(0, glyph_box[1])
                gr1 = min(h - 1, glyph_box[2])
                gc1 = min(w - 1, glyph_box[3])
                if reserve[gr0 : gr1 + 1, gc0 : gc1 + 1].any():
                    continue
                candidates.append((rect_top, rect_left, center))

        if not candidates:
            return None

        rect_top, rect_left, center = random.choice(candidates)
        anchor_cells = _anchor_abs_coords(proto, rect_top, rect_left)
        for r, c in anchor_cells:
            grid[r, c] = color
        _mark_box(reserve, rect_top, rect_left, rect_top + rect_h - 1, rect_left + rect_w - 1, pad=1)
        _mark_box(reserve, center[0] - 2, center[1] - 2, center[0] + 2, center[1] + 2, pad=1)
        groups_meta.append(
            {
                "color": int(color),
                "ratio": proto["ratio"],
                "proto": proto,
                "rect_top": int(rect_top),
                "rect_left": int(rect_left),
                "center": center,
                "anchor_cells": anchor_cells,
            }
        )

    random.shuffle(groups_meta)
    for group in groups_meta:
        proto = group["proto"]
        min_r = min(r for r, _ in proto["source"])
        max_r = max(r for r, _ in proto["source"])
        src_h = max_r - min_r + 1

        rows = list(range(1, h - 1 - src_h + 1))
        random.shuffle(rows)
        placed = None
        for row_shift in rows:
            source_cells = _source_abs_coords(proto, row_shift)
            if any(grid[r, c] != BG for r, c in source_cells):
                continue
            rs = [r for r, _ in source_cells]
            cs = [c for _, c in source_cells]
            r0, c0, r1, c1 = min(rs), min(cs), max(rs), max(cs)
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0)
            rr1 = min(h - 1, r1 + 1)
            cc1 = min(w - 1, c1 + 1)
            if reserve[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                continue
            placed = source_cells
            break
        if placed is None:
            return None
        for r, c in placed:
            grid[r, c] = group["color"]
        rs = [r for r, _ in placed]
        cs = [c for _, c in placed]
        _mark_box(reserve, min(rs), min(cs), max(rs), max(cs), pad=1)
        group["source_cells"] = placed

    return grid, groups_meta, example_cells


def _validate_case(grid: np.ndarray, groups_meta: list[dict], example_cells: set[tuple[int, int]]) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if not (np.all(grid[0, :] == BORDER) and np.all(grid[-1, :] == BORDER) and np.all(grid[:, 0] == BORDER) and np.all(grid[:, -1] == BORDER)):
        return False
    if len(groups_meta) not in (2, 3):
        return False
    if len({g["ratio"] for g in groups_meta}) != len(groups_meta):
        return False
    if len({g["color"] for g in groups_meta}) != len(groups_meta):
        return False
    if len(example_cells) != len(_glyph_cells(random.choice(RATIOS))):
        return False

    try:
        parsed_groups = _extract_groups(grid)
        example = _example_component(grid)
    except AssertionError:
        return False

    parsed_by_color = {g["color"]: g for g in parsed_groups}
    for group in groups_meta:
        parsed = parsed_by_color.get(group["color"])
        if parsed is None:
            return False
        if parsed["ratio"] != group["ratio"]:
            return False
        if parsed["center"] != group["center"]:
            return False

    expected = _oracle_from_metadata(grid, groups_meta, example_cells)
    produced = _render_output(grid, parsed_groups, example)
    if not np.array_equal(produced, expected):
        return False

    colored = {(int(r), int(c)) for r, c in np.argwhere(produced != BG)}
    if not colored:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        case = _generate_case()
        if case is None:
            continue
        grid, groups_meta, example_cells = case
        if _validate_case(grid, groups_meta, example_cells):
            return grid
    raise RuntimeError("failed to generate valid ARC input")
