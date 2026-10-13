from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

from arc_utils import find_components, get_bg


Axis = tuple[str, int]  # ("v", mirror_col_sum) or ("h", mirror_row_sum)
Cell = tuple[int, int]
Rect = tuple[int, int, int, int]  # inclusive bbox


BASE_SHAPES = [
    {(0, 0)},
    {(0, 0), (0, 1)},
    {(0, 0), (1, 0)},
    {(0, 0), (0, 1), (0, 2)},
    {(0, 0), (1, 0), (2, 0)},
    {(0, 0), (1, 0), (1, 1)},
    {(0, 1), (1, 0), (1, 1)},
    {(0, 0), (0, 1), (1, 0)},
    {(0, 0), (0, 1), (1, 1)},
    {(0, 0), (1, 0), (1, 1), (2, 1)},
    {(0, 1), (1, 0), (1, 1), (2, 0)},
]


def _normalize_shape(shape: set[Cell]) -> tuple[Cell, ...]:
    min_r = min(r for r, _ in shape)
    min_c = min(c for _, c in shape)
    norm = sorted((r - min_r, c - min_c) for r, c in shape)
    return tuple(norm)


def _transform_shape(shape: set[Cell], rot: int, flip: bool) -> tuple[Cell, ...]:
    pts = list(shape)
    for _ in range(rot % 4):
        pts = [(c, -r) for r, c in pts]
    if flip:
        pts = [(r, -c) for r, c in pts]
    return _normalize_shape(set(pts))


def _shape_variants() -> list[tuple[Cell, ...]]:
    seen: set[tuple[Cell, ...]] = set()
    out: list[tuple[Cell, ...]] = []
    for shape in BASE_SHAPES:
        for rot in range(4):
            for flip in (False, True):
                variant = _transform_shape(shape, rot, flip)
                if variant not in seen:
                    seen.add(variant)
                    out.append(variant)
    return out


SHAPE_VARIANTS = _shape_variants()


def _rect_cells(rect: Rect) -> set[Cell]:
    r0, c0, r1, c1 = rect
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _is_solid_rectangle(grid: np.ndarray, comp: dict, *, min_side: int = 1) -> bool:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    color = int(comp["color"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    return h >= min_side and w >= min_side and int(comp["area"]) == h * w and np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color)


def _comp_signature(comp: dict) -> tuple[int, int, int, int, int, int]:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    return (int(comp["color"]), r0, c0, r1, c1, int(comp["area"]))


def _mirror_cell(cell: Cell, axis: Axis) -> Cell:
    r, c = map(int, cell)
    kind, mirror_sum = axis
    if kind == "v":
        return (r, int(mirror_sum - c))
    return (int(mirror_sum - r), c)


def _orbit_cells(cells: set[Cell], axes: list[Axis]) -> set[Cell]:
    orbit = set(cells)
    for axis in axes:
        orbit |= {_mirror_cell(cell, axis) for cell in orbit}
    return orbit


def _all_inside_band(coords: np.ndarray, band: Rect) -> bool:
    r0, c0, r1, c1 = band
    return all(r0 <= int(r) <= r1 and c0 <= int(c) <= c1 for r, c in coords)


def _infer_corridors(grid: np.ndarray) -> tuple[list[dict], set[tuple[int, int, int, int, int, int]]]:
    bg = int(get_bg(grid))
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=True)

    rects_by_color: dict[int, list[dict]] = {}
    for comp in comps:
        if _is_solid_rectangle(grid, comp, min_side=2):
            rects_by_color.setdefault(int(comp["color"]), []).append(comp)

    corridors: list[dict] = []
    anchor_signatures: set[tuple[int, int, int, int, int, int]] = set()

    for color, rects in rects_by_color.items():
        if len(rects) != 2:
            continue
        a, b = rects
        ar0, ac0, ar1, ac1 = map(int, a["bbox"])
        br0, bc0, br1, bc1 = map(int, b["bbox"])

        if (ar0, ar1) == (br0, br1):
            left, right = sorted(rects, key=lambda comp: comp["bbox"][1])
            lr0, _lc0, lr1, lc1 = map(int, left["bbox"])
            _rr0, rc0, _rr1, _rc1 = map(int, right["bbox"])
            if lc1 + 1 <= rc0 - 1:
                corridors.append(
                    {
                        "orientation": "h",
                        "band": (lr0, lc1 + 1, lr1, rc0 - 1),
                        "axis": ("v", lc1 + rc0),
                        "color": color,
                    }
                )
                anchor_signatures.add(_comp_signature(left))
                anchor_signatures.add(_comp_signature(right))
        elif (ac0, ac1) == (bc0, bc1):
            top, bottom = sorted(rects, key=lambda comp: comp["bbox"][0])
            _tr0, tc0, tr1, tc1 = map(int, top["bbox"])
            br0, _bc0, _br1, _bc1 = map(int, bottom["bbox"])
            if tr1 + 1 <= br0 - 1:
                corridors.append(
                    {
                        "orientation": "v",
                        "band": (tr1 + 1, tc0, br0 - 1, tc1),
                        "axis": ("h", tr1 + br0),
                        "color": color,
                    }
                )
                anchor_signatures.add(_comp_signature(top))
                anchor_signatures.add(_comp_signature(bottom))

    assert corridors, "expected at least one same-color rectangle corridor"
    return corridors, anchor_signatures


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete corridor-contained motifs by mirroring them across the hidden bisector axes implied by same-color rectangle pairs.

    Input structure:
    - A dominant background contains one or two pairs of same-color solid rectangles.
    - A left-right rectangle pair shares a row span and leaves a background corridor between its facing edges; a top-bottom pair shares a column span and leaves a vertical corridor.
    - Small connected colored motifs may lie wholly inside one corridor, wholly inside the overlap of two corridors, or outside every corridor.

    Rule:
    - Infer one hidden axis per rectangle pair: a left-right pair defines the vertical midpoint of its gap, and a top-bottom pair defines the horizontal midpoint of its gap.
    - Keep rectangles and all out-of-corridor motifs unchanged.
    - For each non-rectangle connected motif entirely inside exactly one corridor, add its mirror image across that corridor's hidden axis.
    - For each motif entirely inside the overlap of a horizontal and a vertical corridor, add its vertical mirror, horizontal mirror, and 180-degree counterpart so the overlap becomes symmetric with respect to both axes.

    Key insight:
    - The axes are never drawn; they are determined only by the empty gaps between matching rectangles, and a motif's behavior depends on whether it belongs to zero, one, or two such corridor domains.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = int(get_bg(g))
    corridors, anchor_signatures = _infer_corridors(g)
    out = g.copy()

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    for comp in comps:
        if _comp_signature(comp) in anchor_signatures and _is_solid_rectangle(g, comp, min_side=2):
            continue

        axes: list[Axis] = []
        for corridor in corridors:
            if _all_inside_band(comp["coords"], corridor["band"]):
                axis = corridor["axis"]
                if axis not in axes:
                    axes.append(axis)
        if not axes:
            continue

        orbit = _orbit_cells({(int(r), int(c)) for r, c in comp["coords"]}, axes)
        for r, c in orbit:
            if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
                out[r, c] = int(comp["color"])

    return out


def _rect_mask(shape: tuple[int, int], rect: Rect) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = rect
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _mask_minus(shape: tuple[int, int], rect: Rect, cut: Rect) -> np.ndarray:
    mask = _rect_mask(shape, rect)
    cr0, cc0, cr1, cc1 = cut
    mask[cr0 : cr1 + 1, cc0 : cc1 + 1] = False
    return mask


def _expand(cells: set[Cell], shape: tuple[int, int]) -> set[Cell]:
    H, W = shape
    out: set[Cell] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    out.add((rr, cc))
    return out


def _paint(grid: np.ndarray, cells: set[Cell], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _placements_for_shape(shape: tuple[Cell, ...], allowed_mask: np.ndarray) -> list[set[Cell]]:
    H, W = allowed_mask.shape
    sh = max(r for r, _ in shape) + 1
    sw = max(c for _, c in shape) + 1
    placements: list[set[Cell]] = []
    for r0 in range(H - sh + 1):
        for c0 in range(W - sw + 1):
            cells = {(r0 + r, c0 + c) for r, c in shape}
            if all(allowed_mask[r, c] for r, c in cells):
                placements.append(cells)
    return placements


def _candidate_shapes(rng: random.Random, preferred: tuple[Cell, ...] | None = None) -> list[tuple[Cell, ...]]:
    variants = SHAPE_VARIANTS[:]
    rng.shuffle(variants)
    if preferred is not None:
        variants = [preferred] + [shape for shape in variants if shape != preferred]
    return variants


def _place_motif(
    rng: random.Random,
    base_grid: np.ndarray,
    input_grid: np.ndarray,
    expected_grid: np.ndarray,
    allowed_mask: np.ndarray,
    axes: list[Axis],
    color: int,
    blocked_input: set[Cell],
    blocked_output: set[Cell],
    motifs: list[dict],
    *,
    require_change: bool = True,
    preferred_shape: tuple[Cell, ...] | None = None,
) -> tuple[Cell, ...] | None:
    bg = int(get_bg(base_grid))
    for shape in _candidate_shapes(rng, preferred=preferred_shape):
        placements = _placements_for_shape(shape, allowed_mask)
        rng.shuffle(placements)
        for cells in placements:
            if any(cell in blocked_input for cell in cells):
                continue
            orbit = _orbit_cells(cells, axes)
            if require_change and orbit == cells:
                continue
            if any(base_grid[r, c] != bg for r, c in orbit):
                continue
            if any(cell in blocked_output for cell in orbit):
                continue

            _paint(input_grid, cells, color)
            _paint(expected_grid, orbit, color)
            blocked_input |= _expand(cells, input_grid.shape)
            blocked_output |= _expand(orbit, expected_grid.shape)
            motifs.append(
                {
                    "color": int(color),
                    "shape": shape,
                    "cells": set(cells),
                    "axes": list(axes),
                    "orbit": set(orbit),
                }
            )
            return shape
    return None


def _sample_colors(rng: random.Random, count: int) -> list[int]:
    return rng.sample(list(range(10)), count)


def _build_horizontal_layout(rng: random.Random) -> tuple[np.ndarray, list[dict], dict]:
    bg, rect_color = _sample_colors(rng, 2)
    row_h = rng.randint(3, 5)
    left_w = rng.randint(2, 4)
    right_w = rng.randint(2, 4)
    gap_w = rng.randint(6, 12)
    top_margin = rng.randint(2, 7)
    bottom_margin = rng.randint(2, 7)
    left_margin = rng.randint(1, 4)
    right_margin = rng.randint(1, 4)

    H = top_margin + row_h + bottom_margin
    W = left_margin + left_w + gap_w + right_w + right_margin
    grid = np.full((H, W), bg, dtype=int)

    left = (top_margin, left_margin, top_margin + row_h - 1, left_margin + left_w - 1)
    right = (top_margin, left[3] + 1 + gap_w, top_margin + row_h - 1, left[3] + gap_w + right_w)

    _paint(grid, _rect_cells(left), rect_color)
    _paint(grid, _rect_cells(right), rect_color)

    corridors = [
        {
            "orientation": "h",
            "band": (left[0], left[3] + 1, left[2], right[1] - 1),
            "axis": ("v", left[3] + right[1]),
            "color": rect_color,
        }
    ]
    return grid, corridors, {"bg": bg, "anchors": [left, right]}


def _build_vertical_layout(rng: random.Random) -> tuple[np.ndarray, list[dict], dict]:
    bg, rect_color = _sample_colors(rng, 2)
    col_w = rng.randint(3, 5)
    top_h = rng.randint(2, 4)
    bottom_h = rng.randint(2, 4)
    gap_h = rng.randint(6, 12)
    top_margin = rng.randint(1, 4)
    bottom_margin = rng.randint(1, 4)
    left_margin = rng.randint(2, 7)
    right_margin = rng.randint(2, 7)

    H = top_margin + top_h + gap_h + bottom_h + bottom_margin
    W = left_margin + col_w + right_margin
    grid = np.full((H, W), bg, dtype=int)

    top = (top_margin, left_margin, top_margin + top_h - 1, left_margin + col_w - 1)
    bottom = (top[2] + 1 + gap_h, left_margin, top[2] + gap_h + bottom_h, left_margin + col_w - 1)

    _paint(grid, _rect_cells(top), rect_color)
    _paint(grid, _rect_cells(bottom), rect_color)

    corridors = [
        {
            "orientation": "v",
            "band": (top[2] + 1, top[1], bottom[0] - 1, top[3]),
            "axis": ("h", top[2] + bottom[0]),
            "color": rect_color,
        }
    ]
    return grid, corridors, {"bg": bg, "anchors": [top, bottom]}


def _build_cross_layout(rng: random.Random) -> tuple[np.ndarray, list[dict], dict]:
    bg, h_color, v_color = _sample_colors(rng, 3)
    row_h = rng.randint(3, 5)
    col_w = rng.randint(3, 5)
    left_w = rng.randint(2, 4)
    right_w = rng.randint(2, 4)
    top_h = rng.randint(2, 4)
    bottom_h = rng.randint(2, 4)

    gap_w = rng.randint(col_w + 6, 12)
    gap_h = rng.randint(row_h + 6, 12)

    left_margin = rng.randint(1, 4)
    right_margin = rng.randint(1, 4)
    top_margin = rng.randint(1, 4)
    bottom_margin = rng.randint(1, 4)

    lc0 = left_margin
    lc1 = lc0 + left_w - 1
    rc0 = lc1 + 1 + gap_w
    rc1 = rc0 + right_w - 1

    tr0 = top_margin
    tr1 = tr0 + top_h - 1
    br0 = tr1 + 1 + gap_h
    br1 = br0 + bottom_h - 1

    gap_c0, gap_c1 = lc1 + 1, rc0 - 1
    gap_r0, gap_r1 = tr1 + 1, br0 - 1

    col_options = [c0 for c0 in range(gap_c0 + 2, gap_c1 - col_w - 1) if c0 + col_w - 1 <= gap_c1 - 2]
    row_options = [r0 for r0 in range(gap_r0 + 2, gap_r1 - row_h - 1) if r0 + row_h - 1 <= gap_r1 - 2]
    assert col_options and row_options
    vc0 = rng.choice(col_options)
    hr0 = rng.choice(row_options)
    vc1 = vc0 + col_w - 1
    hr1 = hr0 + row_h - 1

    H = br1 + 1 + bottom_margin
    W = rc1 + 1 + right_margin
    grid = np.full((H, W), bg, dtype=int)

    left = (hr0, lc0, hr1, lc1)
    right = (hr0, rc0, hr1, rc1)
    top = (tr0, vc0, tr1, vc1)
    bottom = (br0, vc0, br1, vc1)

    _paint(grid, _rect_cells(left), h_color)
    _paint(grid, _rect_cells(right), h_color)
    _paint(grid, _rect_cells(top), v_color)
    _paint(grid, _rect_cells(bottom), v_color)

    corridors = [
        {
            "orientation": "h",
            "band": (hr0, gap_c0, hr1, gap_c1),
            "axis": ("v", lc1 + rc0),
            "color": h_color,
        },
        {
            "orientation": "v",
            "band": (gap_r0, vc0, gap_r1, vc1),
            "axis": ("h", tr1 + br0),
            "color": v_color,
        },
    ]
    return grid, corridors, {"bg": bg, "anchors": [left, right, top, bottom]}


def _outside_mask(shape: tuple[int, int], corridors: list[dict]) -> np.ndarray:
    mask = np.ones(shape, dtype=bool)
    for corridor in corridors:
        r0, c0, r1, c1 = corridor["band"]
        mask[r0 : r1 + 1, c0 : c1 + 1] = False
    return mask


def _generate_case() -> tuple[np.ndarray, np.ndarray, dict]:
    rng = random.Random()
    mode = rng.choices(["hv", "h", "v"], weights=[6, 2, 2], k=1)[0]
    for _ in range(400):
        if mode == "hv":
            base_grid, corridors, meta = _build_cross_layout(rng)
        elif mode == "h":
            base_grid, corridors, meta = _build_horizontal_layout(rng)
        else:
            base_grid, corridors, meta = _build_vertical_layout(rng)

        bg = int(meta["bg"])
        input_grid = base_grid.copy()
        expected_grid = base_grid.copy()

        anchor_cells = set()
        for rect in meta["anchors"]:
            anchor_cells |= _rect_cells(rect)

        blocked_input = _expand(anchor_cells, base_grid.shape)
        blocked_output = _expand(anchor_cells, base_grid.shape)
        motifs: list[dict] = []

        extra_colors = [color for color in range(10) if color not in {bg, *(corridor["color"] for corridor in corridors)}]
        rng.shuffle(extra_colors)
        color_pool = extra_colors[:] or [color for color in range(10) if color != bg]
        color_i = 0

        def next_color() -> int:
            nonlocal color_i
            color = color_pool[color_i % len(color_pool)]
            color_i += 1
            return int(color)

        ok = True
        placed_shapes: list[tuple[int, tuple[Cell, ...]]] = []

        if mode == "hv":
            h_corr = next(c for c in corridors if c["orientation"] == "h")
            v_corr = next(c for c in corridors if c["orientation"] == "v")
            hband = h_corr["band"]
            vband = v_corr["band"]
            intersection = (hband[0], vband[1], hband[2], vband[3])

            shape = _place_motif(
                rng,
                base_grid,
                input_grid,
                expected_grid,
                _rect_mask(base_grid.shape, intersection),
                [h_corr["axis"], v_corr["axis"]],
                next_color(),
                blocked_input,
                blocked_output,
                motifs,
                require_change=True,
            )
            ok &= shape is not None
            if shape is not None:
                placed_shapes.append((motifs[-1]["color"], shape))

            left_arm = (hband[0], hband[1], hband[2], vband[1] - 1)
            right_arm = (hband[0], vband[3] + 1, hband[2], hband[3])
            arm_masks = [rect for rect in (left_arm, right_arm) if rect[1] <= rect[3]]
            rng.shuffle(arm_masks)
            placed = False
            for rect in arm_masks:
                shape = _place_motif(
                    rng,
                    base_grid,
                    input_grid,
                    expected_grid,
                    _rect_mask(base_grid.shape, rect),
                    [h_corr["axis"]],
                    next_color(),
                    blocked_input,
                    blocked_output,
                    motifs,
                    require_change=True,
                )
                if shape is not None:
                    placed_shapes.append((motifs[-1]["color"], shape))
                    placed = True
                    break
            ok &= placed

            top_arm = (vband[0], vband[1], hband[0] - 1, vband[3])
            bottom_arm = (hband[2] + 1, vband[1], vband[2], vband[3])
            arm_masks = [rect for rect in (top_arm, bottom_arm) if rect[0] <= rect[2]]
            rng.shuffle(arm_masks)
            placed = False
            for rect in arm_masks:
                shape = _place_motif(
                    rng,
                    base_grid,
                    input_grid,
                    expected_grid,
                    _rect_mask(base_grid.shape, rect),
                    [v_corr["axis"]],
                    next_color(),
                    blocked_input,
                    blocked_output,
                    motifs,
                    require_change=True,
                )
                if shape is not None:
                    placed_shapes.append((motifs[-1]["color"], shape))
                    placed = True
                    break
            ok &= placed

            if rng.random() < 0.5:
                spare_rects = [rect for rect in (left_arm, right_arm, top_arm, bottom_arm) if rect[0] <= rect[2] and rect[1] <= rect[3]]
                rng.shuffle(spare_rects)
                for rect in spare_rects:
                    axes = [h_corr["axis"]] if rect in (left_arm, right_arm) else [v_corr["axis"]]
                    shape = _place_motif(
                        rng,
                        base_grid,
                        input_grid,
                        expected_grid,
                        _rect_mask(base_grid.shape, rect),
                        axes,
                        next_color(),
                        blocked_input,
                        blocked_output,
                        motifs,
                        require_change=True,
                    )
                    if shape is not None:
                        placed_shapes.append((motifs[-1]["color"], shape))
                        break
        else:
            corridor = corridors[0]
            for _idx in range(rng.randint(2, 3)):
                shape = _place_motif(
                    rng,
                    base_grid,
                    input_grid,
                    expected_grid,
                    _rect_mask(base_grid.shape, corridor["band"]),
                    [corridor["axis"]],
                    next_color(),
                    blocked_input,
                    blocked_output,
                    motifs,
                    require_change=True,
                )
                ok &= shape is not None
                if shape is not None:
                    placed_shapes.append((motifs[-1]["color"], shape))

        outside = _outside_mask(base_grid.shape, corridors)
        clone_target = placed_shapes[:] if placed_shapes else []
        rng.shuffle(clone_target)
        decoy_goal = rng.randint(2, 4)
        decoys = 0
        while decoys < decoy_goal:
            preferred_shape = clone_target[decoys % len(clone_target)][1] if clone_target and decoys < len(clone_target) else None
            preferred_color = clone_target[decoys % len(clone_target)][0] if clone_target and decoys < len(clone_target) else next_color()
            shape = _place_motif(
                rng,
                base_grid,
                input_grid,
                expected_grid,
                outside,
                [],
                preferred_color,
                blocked_input,
                blocked_output,
                motifs,
                require_change=False,
                preferred_shape=preferred_shape,
            )
            if shape is None:
                break
            decoys += 1

        changed = int(np.count_nonzero(input_grid != expected_grid))
        if not ok or changed == 0:
            mode = rng.choices(["hv", "h", "v"], weights=[6, 2, 2], k=1)[0]
            continue

        parsed = main(input_grid)
        if not np.array_equal(parsed, expected_grid):
            mode = rng.choices(["hv", "h", "v"], weights=[6, 2, 2], k=1)[0]
            continue

        meta.update({"mode": mode, "corridors": corridors, "motifs": motifs})
        return input_grid, expected_grid, meta

    raise RuntimeError("failed to generate a valid corridor-mirroring case")


def generate_input() -> np.ndarray:
    return _generate_case()[0]
