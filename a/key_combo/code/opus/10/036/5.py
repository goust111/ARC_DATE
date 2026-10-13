from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


BASE_SHAPES = [
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (2, 1)),
    ((0, 1), (0, 2), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
    ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2)),
]


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize_coords(coords: list[tuple[int, int]] | np.ndarray) -> tuple[tuple[int, int], ...]:
    pts = [(int(r), int(c)) for r, c in coords]
    if not pts:
        return ()
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    return tuple(sorted((r - r0, c - c0) for r, c in pts))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _all_transforms(shape: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    arr = np.zeros(_shape_dims(shape), dtype=bool)
    for r, c in shape:
        arr[r, c] = True
    out: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for k in range(4):
        rot = np.rot90(arr, k)
        for variant in (rot, np.fliplr(rot)):
            coords = _normalize_coords(np.argwhere(variant))
            if coords not in seen:
                seen.add(coords)
                out.append(coords)
    return out


SHAPE_VARIANTS = [_all_transforms(base) for base in BASE_SHAPES]


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _components_by_color(grid: np.ndarray, colors: list[int] | tuple[int, ...] | None = None) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    if colors is None:
        colors = sorted({int(v) for v in g.flat if int(v) != BLACK})
    out: list[dict] = []
    for color in colors:
        comps = _components_from_mask(g == int(color))
        for coords in comps:
            out.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": len(coords),
                }
            )
    return out


def _dilate_mask(shape: tuple[int, int], coords: list[tuple[int, int]], pad: int = 1) -> np.ndarray:
    h, w = shape
    mask = np.zeros(shape, dtype=bool)
    for r, c in coords:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    mask[rr, cc] = True
    return mask


def _is_hollow_rect(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return False
    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    return set(comp["coords"]) == expected


def _parse_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    grey_components = _components_by_color(g, [GREY])
    boxes = [comp for comp in grey_components if _is_hollow_rect(comp)]
    axes = [comp for comp in grey_components if comp not in boxes]
    assert len(boxes) == 1, "expected exactly one gray hollow rectangle"
    assert 1 <= len(axes) <= 2, "expected one or two gray vertical axis segments"

    axis_cols: list[int] = []
    for axis in axes:
        rows = sorted(r for r, _ in axis["coords"])
        cols = {c for _, c in axis["coords"]}
        assert len(cols) == 1, "axis segment must occupy one column"
        assert rows == list(range(rows[0], rows[-1] + 1)), "axis segment must be contiguous"
        axis_cols.append(next(iter(cols)))

    box = boxes[0]
    br0, bc0, br1, bc1 = box["bbox"]
    inner_r0, inner_c0, inner_r1, inner_c1 = br0 + 1, bc0 + 1, br1 - 1, bc1 - 1
    assert inner_r0 <= inner_r1 and inner_c0 <= inner_c1, "box interior must be non-empty"

    ref_mask = np.zeros(g.shape, dtype=bool)
    ref_mask[inner_r0 : inner_r1 + 1, inner_c0 : inner_c1 + 1] = (g[inner_r0 : inner_r1 + 1, inner_c0 : inner_c1 + 1] != BLACK) & (
        g[inner_r0 : inner_r1 + 1, inner_c0 : inner_c1 + 1] != GREY
    )
    ref_components = _components_from_mask(ref_mask)
    assert len(ref_components) == 1, "expected exactly one reference sprite inside the box"
    ref_coords = ref_components[0]
    ref_colors = {int(g[r, c]) for r, c in ref_coords}
    assert len(ref_colors) == 1, "reference sprite must be monochrome"
    ref_color = next(iter(ref_colors))
    ref_shape = _normalize_coords(ref_coords)

    outer_mask = (g != BLACK) & (g != GREY)
    outer_mask[inner_r0 : inner_r1 + 1, inner_c0 : inner_c1 + 1] = False
    outer_components = []
    for color in sorted({int(v) for v in g[outer_mask]}):
        for coords in _components_from_mask(g == color):
            if any(inner_r0 <= r <= inner_r1 and inner_c0 <= c <= inner_c1 for r, c in coords):
                continue
            outer_components.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "shape": _normalize_coords(coords),
                }
            )
    assert 3 <= len(outer_components) <= 5, "expected three to five scattered sprites"
    return {
        "box_bbox": box["bbox"],
        "ref_coords": ref_coords,
        "ref_color": int(ref_color),
        "ref_shape": ref_shape,
        "axis_cols": axis_cols,
        "outer_components": outer_components,
    }


def _unique_nearest_axis(axis_cols: list[int], bbox: tuple[int, int, int, int]) -> int | None:
    _r0, c0, _r1, c1 = bbox
    center = (c0 + c1) / 2.0
    ranked = sorted((abs(center - col), idx) for idx, col in enumerate(axis_cols))
    if len(ranked) > 1 and ranked[0][0] >= ranked[1][0]:
        return None
    return ranked[0][1]


def _nearest_axis(axis_cols: list[int], bbox: tuple[int, int, int, int]) -> int:
    idx = _unique_nearest_axis(axis_cols, bbox)
    assert idx is not None, "nearest axis must be unique"
    return idx


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter same-shaped scattered sprites, reflect only those across the nearest vertical axis, and recolor each reflected copy with the boxed reference color.

    Input structure:
    - A black background contains one gray hollow rectangle with a single colored reference sprite inside.
    - One or two separate gray vertical line segments act as reflection axes.
    - Three to five scattered monochrome 4-connected sprites appear outside the box in various colors.

    Rule:
    - Extract the boxed reference sprite's shape mask and its color.
    - Compare every outside sprite against that reference using only bbox-relative coordinates.
    - If a sprite matches, keep it and add a horizontally reflected copy across its nearest gray axis column, painting the copy with the boxed reference color.
    - If a sprite does not match, erase it to black.
    - Keep the gray box, the gray axes, and the boxed reference sprite unchanged.

    Key insight:
    - The boxed sprite controls two downstream decisions at once: its mask selects which free sprites survive, and its color paints every reflected copy.
    """
    parsed = _parse_grid(np.asarray(grid, dtype=int))
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    ref_shape = parsed["ref_shape"]
    ref_color = parsed["ref_color"]
    axis_cols = parsed["axis_cols"]

    for comp in parsed["outer_components"]:
        coords = comp["coords"]
        if comp["shape"] != ref_shape:
            for r, c in coords:
                out[r, c] = BLACK
            continue

        axis_col = axis_cols[_nearest_axis(axis_cols, comp["bbox"])]
        reflected = [(r, 2 * axis_col - c) for r, c in coords]
        assert all(0 <= r < g.shape[0] and 0 <= c < g.shape[1] for r, c in reflected), "reflected copy must stay in-bounds"
        for r, c in reflected:
            out[r, c] = ref_color

    return out


def _paint_shape(grid: np.ndarray, shape: tuple[tuple[int, int], ...], top: int, left: int, color: int) -> list[tuple[int, int]]:
    coords = [(top + r, left + c) for r, c in shape]
    for r, c in coords:
        grid[r, c] = int(color)
    return coords


def _random_shape_choice() -> tuple[int, tuple[tuple[int, int], ...]]:
    base_idx = random.randrange(len(SHAPE_VARIANTS))
    return base_idx, random.choice(SHAPE_VARIANTS[base_idx])


def _choose_reference_and_negatives() -> tuple[tuple[tuple[int, int], ...], list[tuple[tuple[int, int], ...]], bool]:
    base_idx, ref_shape = _random_shape_choice()
    same_base_variants = [shape for shape in SHAPE_VARIANTS[base_idx] if shape != ref_shape]
    negatives: list[tuple[tuple[int, int], ...]] = []
    hard_negative = False
    if same_base_variants:
        negatives.append(random.choice(same_base_variants))
        hard_negative = True

    other_shapes = [shape for idx, variants in enumerate(SHAPE_VARIANTS) if idx != base_idx for shape in variants]
    random.shuffle(other_shapes)
    for shape in other_shapes:
        if shape != ref_shape and shape not in negatives:
            negatives.append(shape)
        if len(negatives) >= 4:
            break
    return ref_shape, negatives, hard_negative


def _sprite_candidates(
    shape_key: tuple[tuple[int, int], ...],
    axis_cols: list[int],
    axis_idx: int,
    side: str,
    matched: bool,
    blocked: np.ndarray,
) -> list[tuple[list[tuple[int, int]], list[tuple[int, int]]]]:
    h, w = blocked.shape
    sh, sw = _shape_dims(shape_key)
    axis_col = axis_cols[axis_idx]
    candidates = []
    for top in range(h - sh + 1):
        for left in range(w - sw + 1):
            if side == "left" and left + sw - 1 >= axis_col:
                continue
            if side == "right" and left <= axis_col:
                continue
            bbox = (top, left, top + sh - 1, left + sw - 1)
            nearest = _unique_nearest_axis(axis_cols, bbox)
            if nearest is None or nearest != axis_idx:
                continue

            coords = [(top + r, left + c) for r, c in shape_key]
            reflect = [(r, 2 * axis_col - c) for r, c in coords] if matched else []
            all_coords = coords + reflect
            if not all(0 <= r < h and 0 <= c < w for r, c in all_coords):
                continue

            claim = np.zeros_like(blocked)
            for r, c in coords:
                claim[r, c] = True
            if matched:
                for r, c in reflect:
                    if claim[r, c]:
                        claim[r, c] = True
                    else:
                        claim[r, c] = True
            dilated = claim | _dilate_mask(blocked.shape, coords, 1)
            if matched:
                dilated |= _dilate_mask(blocked.shape, reflect, 1)
            if np.any(dilated & blocked):
                continue
            candidates.append((coords, reflect))
    return candidates


def _reserve(blocked: np.ndarray, coords: list[tuple[int, int]]) -> None:
    blocked |= _dilate_mask(blocked.shape, coords, 1)


def _make_instance() -> tuple[np.ndarray, dict]:
    for _ in range(500):
        h = random.randint(15, 22)
        w = random.randint(18, 26)
        grid = np.zeros((h, w), dtype=int)
        blocked = np.zeros((h, w), dtype=bool)

        inner_h = random.randint(4, 5)
        inner_w = random.randint(4, 5)
        box_h = inner_h + 2
        box_w = inner_w + 2
        box_top = random.randint(1, h - box_h - 2)
        box_left = random.randint(1, w - box_w - 2)
        box_coords = []
        for c in range(box_left, box_left + box_w):
            box_coords.append((box_top, c))
            box_coords.append((box_top + box_h - 1, c))
        for r in range(box_top, box_top + box_h):
            box_coords.append((r, box_left))
            box_coords.append((r, box_left + box_w - 1))
        for r, c in set(box_coords):
            grid[r, c] = GREY
        blocked[max(0, box_top - 1) : min(h, box_top + box_h + 1), max(0, box_left - 1) : min(w, box_left + box_w + 1)] = True

        ref_shape, negative_shapes, has_hard_negative = _choose_reference_and_negatives()
        ref_color = random.choice(PALETTE)
        ref_h, ref_w = _shape_dims(ref_shape)
        ref_top = box_top + 1 + random.randint(0, inner_h - ref_h)
        ref_left = box_left + 1 + random.randint(0, inner_w - ref_w)
        ref_coords = _paint_shape(grid, ref_shape, ref_top, ref_left, ref_color)

        axis_count = random.choice([1, 2])
        axis_specs = []
        axis_cols: list[int] = []
        axis_attempts = 0
        while len(axis_specs) < axis_count and axis_attempts < 500:
            axis_attempts += 1
            col = random.randint(2, w - 3)
            if any(abs(col - prev) < 6 for prev in axis_cols):
                continue
            length = random.randint(4, min(8, h - 2))
            top = random.randint(1, h - length - 1)
            coords = [(top + k, col) for k in range(length)]
            dilated = _dilate_mask((h, w), coords, 1)
            if np.any(dilated & blocked):
                continue
            for r, c in coords:
                grid[r, c] = GREY
            blocked |= dilated
            axis_cols.append(col)
            axis_specs.append({"col": col, "coords": coords})
        if len(axis_specs) != axis_count:
            continue

        sprite_count = random.randint(3, 5)
        match_count = random.randint(1, sprite_count - 1)
        nonmatch_count = sprite_count - match_count
        if nonmatch_count < 1:
            continue

        sprite_plan = []
        for _ in range(match_count):
            sprite_plan.append({"shape": ref_shape, "matched": True, "hard_negative": False})

        neg_pool = negative_shapes[:]
        if not neg_pool:
            continue
        if has_hard_negative:
            sprite_plan.append({"shape": neg_pool[0], "matched": False, "hard_negative": True})
            neg_pool = neg_pool[1:] or negative_shapes[:]
        while len(sprite_plan) < sprite_count:
            sprite_plan.append(
                {
                    "shape": random.choice(neg_pool),
                    "matched": False,
                    "hard_negative": False,
                }
            )
        random.shuffle(sprite_plan)

        meta_sprites = []
        ok = True
        for spec in sprite_plan:
            placed = False
            axis_order = list(range(axis_count))
            random.shuffle(axis_order)
            side_order = ["left", "right"]
            random.shuffle(side_order)
            for axis_idx in axis_order:
                for side in side_order:
                    candidates = _sprite_candidates(spec["shape"], axis_cols, axis_idx, side, spec["matched"], blocked)
                    if not candidates:
                        continue
                    coords, reflect = random.choice(candidates)
                    color_choices = [c for c in PALETTE if c != GREY and (not spec["matched"] or c != ref_color)]
                    color = random.choice(color_choices)
                    for r, c in coords:
                        grid[r, c] = color
                    _reserve(blocked, coords)
                    if spec["matched"]:
                        _reserve(blocked, reflect)
                    meta_sprites.append(
                        {
                            "coords": coords,
                            "reflect": reflect,
                            "matched": spec["matched"],
                            "hard_negative": spec["hard_negative"],
                            "color": color,
                            "axis_idx": axis_idx,
                            "shape": spec["shape"],
                        }
                    )
                    placed = True
                    break
                if placed:
                    break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        meta = {
            "grid_shape": (h, w),
            "box_coords": sorted(set(box_coords)),
            "box_bbox": (box_top, box_left, box_top + box_h - 1, box_left + box_w - 1),
            "ref_coords": ref_coords,
            "ref_shape": ref_shape,
            "ref_color": ref_color,
            "axis_cols": axis_cols,
            "axis_specs": axis_specs,
            "sprites": meta_sprites,
        }
        _validate_generated(grid, meta)
        return grid, meta
    raise RuntimeError("failed to generate a valid instance")


def _validate_generated(inp: np.ndarray, meta: dict) -> None:
    out = main(inp)
    h, w = inp.shape
    assert inp.shape == out.shape == tuple(meta["grid_shape"])
    assert 3 <= len(meta["sprites"]) <= 5
    assert any(sprite["matched"] for sprite in meta["sprites"])
    assert any(not sprite["matched"] for sprite in meta["sprites"])
    assert any(sprite["hard_negative"] for sprite in meta["sprites"])

    changed = set()
    for r in range(h):
        for c in range(w):
            if int(inp[r, c]) != int(out[r, c]):
                changed.add((r, c))

    expected_changes = set()
    for r, c in meta["ref_coords"]:
        assert int(inp[r, c]) == meta["ref_color"]
        assert int(out[r, c]) == meta["ref_color"]
    for axis in meta["axis_specs"]:
        for r, c in axis["coords"]:
            assert int(inp[r, c]) == GREY
            assert int(out[r, c]) == GREY

    for sprite in meta["sprites"]:
        coords = {tuple(p) for p in sprite["coords"]}
        reflect = {tuple(p) for p in sprite["reflect"]}
        if sprite["matched"]:
            for r, c in coords:
                assert int(out[r, c]) == int(inp[r, c]), "matched sprite originals must stay"
            for r, c in reflect:
                assert int(inp[r, c]) == BLACK, "matched reflection footprint must be empty in input"
                assert int(out[r, c]) == meta["ref_color"], "matched reflections must use reference color"
                expected_changes.add((r, c))
        else:
            for r, c in coords:
                assert int(out[r, c]) == BLACK, "non-matching sprites must be erased"
                expected_changes.add((r, c))
        assert coords.isdisjoint(reflect)

    assert changed == expected_changes, "output changes must be exactly erasures plus reflected copies"

    parsed = _parse_grid(inp)
    assert parsed["ref_shape"] == meta["ref_shape"]
    assert parsed["ref_color"] == meta["ref_color"]
    assert sorted(parsed["axis_cols"]) == sorted(meta["axis_cols"])
    parsed_shapes = Counter(comp["shape"] == meta["ref_shape"] for comp in parsed["outer_components"])
    assert parsed_shapes[True] >= 1 and parsed_shapes[False] >= 1


def generate_input() -> np.ndarray:
    grid, _meta = _make_instance()
    return grid
