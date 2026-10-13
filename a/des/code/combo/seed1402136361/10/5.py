from __future__ import annotations

import random

import numpy as np


SIDES = ("top", "bottom", "left", "right")
ADJACENT_MISSING = {
    "top": ("left", "right"),
    "bottom": ("left", "right"),
    "left": ("top", "bottom"),
    "right": ("top", "bottom"),
}

GRID_MIN = 18
GRID_MAX = 30
PAIR_COUNT_CHOICES = (2, 3)
OPEN_SIDE_RANGE = (4, 6)
PROVIDER_AXIAL_RANGE = (3, 5)
GAP_RANGE = (1, 3)
OUTER_PAD_RANGE = (1, 4)
GEN_RETRIES = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _components_by_color(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            comps.append({"color": color, "coords": arr, "bbox": _bbox(arr)})
    return comps


def _side_mask(h: int, w: int, side: str) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    if side == "top":
        mask[0, :] = True
    elif side == "bottom":
        mask[-1, :] = True
    elif side == "left":
        mask[:, 0] = True
    else:
        mask[:, -1] = True
    return mask


def _frame_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _mask_from_present_sides(h: int, w: int, present_sides: tuple[str, ...]) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for side in present_sides:
        mask |= _side_mask(h, w, side)
    return mask


def _classify_component(comp: dict[str, object]) -> dict[str, object] | None:
    color = int(comp["color"])
    coords = np.asarray(comp["coords"], dtype=int)
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1

    rel = coords - np.array([[r0, c0]])
    mask = np.zeros((h, w), dtype=bool)
    mask[rel[:, 0], rel[:, 1]] = True

    if np.all(mask):
        return {
            "kind": "closed",
            "shape": "solid",
            "color": color,
            "bbox": (r0, c0, r1, c1),
            "mask": mask,
        }

    if h < 3 or w < 3:
        return None
    border = _frame_mask(h, w)
    if np.any(mask & ~border):
        return None

    present_sides = tuple(side for side in SIDES if np.all(mask[_side_mask(h, w, side)]))
    if len(present_sides) == 4:
        if np.array_equal(mask, border):
            return {
                "kind": "closed",
                "shape": "frame",
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "mask": mask,
            }
        return None

    if len(present_sides) not in (2, 3):
        return None
    if len(present_sides) == 2 and set(present_sides) in ({"top", "bottom"}, {"left", "right"}):
        return None

    expected = _mask_from_present_sides(h, w, present_sides)
    if not np.array_equal(mask, expected):
        return None

    missing_sides = tuple(side for side in SIDES if side not in present_sides)
    return {
        "kind": "open",
        "shape": "frame",
        "color": color,
        "bbox": (r0, c0, r1, c1),
        "mask": mask,
        "missing_sides": missing_sides,
    }


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _directional_distance(open_bbox: tuple[int, int, int, int], closed_bbox: tuple[int, int, int, int], side: str) -> int | None:
    or0, oc0, or1, oc1 = open_bbox
    cr0, cc0, cr1, cc1 = closed_bbox
    if side == "top":
        if not _intervals_overlap(oc0, oc1, cc0, cc1) or cr1 >= or0:
            return None
        return or0 - cr1 - 1
    if side == "bottom":
        if not _intervals_overlap(oc0, oc1, cc0, cc1) or cr0 <= or1:
            return None
        return cr0 - or1 - 1
    if side == "left":
        if not _intervals_overlap(or0, or1, cr0, cr1) or cc1 >= oc0:
            return None
        return oc0 - cc1 - 1
    if side == "right":
        if not _intervals_overlap(or0, or1, cr0, cr1) or cc0 <= oc1:
            return None
        return cc0 - oc1 - 1
    raise ValueError(f"unknown side: {side}")


def _provider_for_open(open_obj: dict[str, object], closed_objs: list[dict[str, object]]) -> dict[str, object]:
    options: list[tuple[int, str, dict[str, object]]] = []
    open_bbox = tuple(map(int, open_obj["bbox"]))
    for side in open_obj["missing_sides"]:
        side_hits: list[tuple[int, dict[str, object]]] = []
        for closed in closed_objs:
            dist = _directional_distance(open_bbox, tuple(map(int, closed["bbox"])), side)
            if dist is not None:
                side_hits.append((dist, closed))
        if not side_hits:
            continue
        best_dist = min(dist for dist, _closed in side_hits)
        best_hits = [(dist, closed) for dist, closed in side_hits if dist == best_dist]
        assert len(best_hits) == 1, "nearest aligned closed shape should be unique in its direction"
        options.append((best_dist, side, best_hits[0][1]))

    assert options, "every open shape must point toward at least one closed shape"
    best_dist = min(dist for dist, _side, _closed in options)
    best = [item for item in options if item[0] == best_dist]
    assert len(best) == 1, "open shape gap direction must be unambiguous"
    return best[0][2]


def _paint_closure(out: np.ndarray, grid: np.ndarray, bg: int, open_obj: dict[str, object], color: int) -> None:
    r0, c0, r1, c1 = map(int, open_obj["bbox"])
    mask = np.asarray(open_obj["mask"], dtype=bool)
    h, w = mask.shape
    patch_in = grid[r0 : r1 + 1, c0 : c1 + 1]
    patch_out = out[r0 : r1 + 1, c0 : c1 + 1]
    fill = np.zeros((h, w), dtype=bool)
    for side in open_obj["missing_sides"]:
        fill |= _side_mask(h, w, side) & ~mask
    assert np.all(patch_in[fill] == bg), "closure cells must be background in the input"
    patch_out[fill] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify open rectangular-border objects, follow each gap toward the nearest aligned
    closed rectangle or frame, and close the missing side(s) using that provider's color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _background_color(g)
    classified: list[dict[str, object]] = []
    for comp in _components_by_color(g, bg):
        info = _classify_component(comp)
        assert info is not None, "all foreground objects must be valid open or closed rectangles"
        classified.append(info)

    open_objs = [obj for obj in classified if obj["kind"] == "open"]
    closed_objs = [obj for obj in classified if obj["kind"] == "closed"]
    assert open_objs and closed_objs, "expected both open and closed objects"

    out = g.copy()
    for open_obj in open_objs:
        provider = _provider_for_open(open_obj, closed_objs)
        _paint_closure(out, g, bg, open_obj, int(provider["color"]))
    return out


def _choose_missing_sides(direction: str) -> tuple[str, ...]:
    if random.random() < 0.5:
        return (direction,)
    extra = random.choice(ADJACENT_MISSING[direction])
    return tuple(side for side in SIDES if side in {direction, extra})


def _open_shape_mask(h: int, w: int, missing_sides: tuple[str, ...]) -> np.ndarray:
    present_sides = tuple(side for side in SIDES if side not in missing_sides)
    return _mask_from_present_sides(h, w, present_sides)


def _sample_pair(open_color: int, provider_color: int) -> dict[str, object]:
    direction = random.choice(SIDES)
    missing_sides = _choose_missing_sides(direction)
    open_h = random.randint(*OPEN_SIDE_RANGE)
    open_w = random.randint(*OPEN_SIDE_RANGE)
    open_mask = _open_shape_mask(open_h, open_w, missing_sides)

    provider_kind = random.choice(("solid", "frame"))
    provider_axial = random.randint(*PROVIDER_AXIAL_RANGE)
    gap = random.randint(*GAP_RANGE)

    if direction in ("top", "bottom"):
        provider_h = provider_axial
        provider_w = open_w
        cluster_h = provider_h + gap + open_h
        cluster_w = open_w
        if direction == "top":
            provider_origin = (0, 0)
            open_origin = (provider_h + gap, 0)
        else:
            open_origin = (0, 0)
            provider_origin = (open_h + gap, 0)
    else:
        provider_h = open_h
        provider_w = provider_axial
        cluster_h = open_h
        cluster_w = provider_w + gap + open_w
        if direction == "left":
            provider_origin = (0, 0)
            open_origin = (0, provider_w + gap)
        else:
            open_origin = (0, 0)
            provider_origin = (0, open_w + gap)

    if provider_kind == "solid":
        provider_mask = np.ones((provider_h, provider_w), dtype=bool)
    else:
        provider_mask = _frame_mask(provider_h, provider_w)

    return {
        "direction": direction,
        "missing_sides": missing_sides,
        "open_mask": open_mask,
        "open_color": open_color,
        "provider_mask": provider_mask,
        "provider_color": provider_color,
        "cluster_h": cluster_h,
        "cluster_w": cluster_w,
        "open_origin": open_origin,
        "provider_origin": provider_origin,
    }


def _layout_pairs(specs: list[dict[str, object]]) -> tuple[int, int, dict[int, tuple[int, int]]] | None:
    count = len(specs)
    row_gaps = [random.randint(*OUTER_PAD_RANGE) for _ in range(count + 1)]
    col_gaps = [random.randint(*OUTER_PAD_RANGE) for _ in range(count + 1)]

    base_h = sum(int(spec["cluster_h"]) for spec in specs) + sum(row_gaps)
    base_w = sum(int(spec["cluster_w"]) for spec in specs) + sum(col_gaps)
    if base_h > GRID_MAX or base_w > GRID_MAX:
        return None

    final_h = random.randint(max(GRID_MIN, base_h), GRID_MAX)
    final_w = random.randint(max(GRID_MIN, base_w), GRID_MAX)
    row_gaps[-1] += final_h - base_h
    col_gaps[-1] += final_w - base_w

    row_order = random.sample(range(count), count)
    col_order = random.sample(range(count), count)
    row_tops = [0] * count
    col_lefts = [0] * count

    cursor = row_gaps[0]
    for gap_idx, spec_idx in enumerate(row_order, start=1):
        row_tops[spec_idx] = cursor
        cursor += int(specs[spec_idx]["cluster_h"]) + row_gaps[gap_idx]

    cursor = col_gaps[0]
    for gap_idx, spec_idx in enumerate(col_order, start=1):
        col_lefts[spec_idx] = cursor
        cursor += int(specs[spec_idx]["cluster_w"]) + col_gaps[gap_idx]

    positions = {idx: (row_tops[idx], col_lefts[idx]) for idx in range(count)}
    return final_h, final_w, positions


def _paint_mask(grid: np.ndarray, origin: tuple[int, int], mask: np.ndarray, color: int, bg: int) -> None:
    r0, c0 = origin
    h, w = mask.shape
    patch = grid[r0 : r0 + h, c0 : c0 + w]
    assert np.all(patch[mask] == bg), "objects must not overlap"
    patch[mask] = color


def _build_example() -> tuple[np.ndarray, np.ndarray] | None:
    pair_count = random.choice(PAIR_COUNT_CHOICES)
    bg = random.randrange(10)
    colors = random.sample([c for c in range(10) if c != bg], 2 * pair_count)
    specs = [_sample_pair(colors[2 * i], colors[2 * i + 1]) for i in range(pair_count)]

    laid_out = _layout_pairs(specs)
    if laid_out is None:
        return None
    height, width, positions = laid_out

    inp = np.full((height, width), bg, dtype=int)
    expected = inp.copy()

    for idx, spec in enumerate(specs):
        cluster_r, cluster_c = positions[idx]
        open_r, open_c = spec["open_origin"]
        provider_r, provider_c = spec["provider_origin"]

        open_origin = (cluster_r + int(open_r), cluster_c + int(open_c))
        provider_origin = (cluster_r + int(provider_r), cluster_c + int(provider_c))

        _paint_mask(inp, provider_origin, np.asarray(spec["provider_mask"], dtype=bool), int(spec["provider_color"]), bg)
        _paint_mask(inp, open_origin, np.asarray(spec["open_mask"], dtype=bool), int(spec["open_color"]), bg)

        _paint_mask(expected, provider_origin, np.asarray(spec["provider_mask"], dtype=bool), int(spec["provider_color"]), bg)
        _paint_mask(expected, open_origin, np.asarray(spec["open_mask"], dtype=bool), int(spec["open_color"]), bg)

        open_mask = np.asarray(spec["open_mask"], dtype=bool)
        fill = np.zeros_like(open_mask)
        for side in spec["missing_sides"]:
            fill |= _side_mask(*open_mask.shape, side) & ~open_mask
        patch = expected[open_origin[0] : open_origin[0] + open_mask.shape[0], open_origin[1] : open_origin[1] + open_mask.shape[1]]
        assert np.all(patch[fill] == bg)
        patch[fill] = int(spec["provider_color"])

    if np.count_nonzero(inp != bg) * 2 >= inp.size:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_example()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid input")
