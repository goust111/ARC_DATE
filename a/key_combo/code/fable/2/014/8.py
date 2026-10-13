from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import TypeAlias

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg


BG = Color.BLACK
GREY = Color.GREY
MARKER_PALETTE = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
MAX_ARC_SIZE = 30
MAX_DEPTH = 2  # 1 -> 2x2 marker, 2 -> recursively nested 4x4 marker
GEN_RETRIES = 400
RECT_COUNT_OPTIONS = [1, 2, 2, 3]
RECT_SIZE_OPTIONS = [4, 8, 12, 16]
CANVAS_MIN = 18
CANVAS_MAX = 30
PLACEMENT_TRIES = 500
OBJECT_GAP = 1
CORNERS = ("tl", "tr", "bl", "br")

Legend: TypeAlias = int | tuple["Legend", "Legend", "Legend", "Legend"]
HalfOpenBox: TypeAlias = tuple[int, int, int, int]
BBox: TypeAlias = tuple[int, int, int, int]


def _is_power_of_two(n: int) -> bool:
    n = int(n)
    return n > 0 and (n & (n - 1)) == 0


def _corner_patch_bounds(rect: BBox, corner: str, side: int) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = rect
    side = int(side)
    if corner == "tl":
        return r0 - side, r0, c0 - side, c0
    if corner == "tr":
        return r0 - side, r0, c1 + 1, c1 + 1 + side
    if corner == "bl":
        return r1 + 1, r1 + 1 + side, c0 - side, c0
    if corner == "br":
        return r1 + 1, r1 + 1 + side, c1 + 1, c1 + 1 + side
    raise ValueError(f"unknown corner: {corner}")


def _corner_coords(rect: BBox, corner: str, side: int) -> list[tuple[int, int]]:
    rs, re, cs, ce = _corner_patch_bounds(rect, corner, side)
    return [(r, c) for r in range(rs, re) for c in range(cs, ce)]


def _extract_marker_patch(grid: np.ndarray, rect: BBox, corner: str, bg: int) -> np.ndarray:
    height, width = grid.shape
    best: np.ndarray | None = None
    side = 1
    while True:
        rs, re, cs, ce = _corner_patch_bounds(rect, corner, side)
        if rs < 0 or cs < 0 or re > height or ce > width:
            break
        patch = grid[rs:re, cs:ce]
        if patch.shape != (side, side):
            break
        if np.any((patch == bg) | (patch == GREY)):
            break
        best = patch.copy()
        side += 1
    assert best is not None, f"missing marker patch at {corner}"
    assert best.shape[0] == best.shape[1]
    assert _is_power_of_two(best.shape[0]), "marker size must be a power of two"
    return best


def _paint_patch(out: np.ndarray, region: HalfOpenBox, patch: np.ndarray) -> None:
    r0, c0, r1, c1 = region
    h = int(r1 - r0)
    w = int(c1 - c0)
    side = int(patch.shape[0])
    assert patch.shape == (side, side)
    assert h > 0 and w > 0
    assert h % side == 0 and w % side == 0

    if side == 1:
        out[r0:r1, c0:c1] = int(patch[0, 0])
        return

    assert side % 2 == 0 and h % 2 == 0 and w % 2 == 0
    rm = r0 + h // 2
    cm = c0 + w // 2
    half = side // 2
    _paint_patch(out, (r0, c0, rm, cm), patch[:half, :half])
    _paint_patch(out, (r0, cm, rm, c1), patch[:half, half:])
    _paint_patch(out, (rm, c0, r1, cm), patch[half:, :half])
    _paint_patch(out, (rm, cm, r1, c1), patch[half:, half:])


def _legend_depth(legend: Legend) -> int:
    if isinstance(legend, int):
        return 0
    return 1 + max(_legend_depth(child) for child in legend)


def _render_legend(legend: Legend, side: int | None = None) -> np.ndarray:
    if isinstance(legend, int):
        side = 1 if side is None else int(side)
        return np.full((side, side), int(legend), dtype=int)

    depth = _legend_depth(legend)
    if side is None:
        side = 1 << depth
    side = int(side)
    assert side % 2 == 0
    half = side // 2
    patch = np.empty((side, side), dtype=int)
    patch[:half, :half] = _render_legend(legend[0], half)
    patch[:half, half:] = _render_legend(legend[1], half)
    patch[half:, :half] = _render_legend(legend[2], half)
    patch[half:, half:] = _render_legend(legend[3], half)
    return patch


def _paint_legend(out: np.ndarray, region: HalfOpenBox, legend: Legend) -> None:
    r0, c0, r1, c1 = region
    if isinstance(legend, int):
        out[r0:r1, c0:c1] = int(legend)
        return

    assert (r1 - r0) % 2 == 0 and (c1 - c0) % 2 == 0
    rm = r0 + (r1 - r0) // 2
    cm = c0 + (c1 - c0) // 2
    _paint_legend(out, (r0, c0, rm, cm), legend[0])
    _paint_legend(out, (r0, cm, rm, c1), legend[1])
    _paint_legend(out, (rm, c0, r1, cm), legend[2])
    _paint_legend(out, (rm, cm, r1, c1), legend[3])


def _legend_signature(legend: Legend):
    if isinstance(legend, int):
        return ("leaf", int(legend))
    return ("node", tuple(_legend_signature(child) for child in legend))


def _sample_legend_exact_depth(depth: int, palette: list[int]) -> Legend:
    depth = int(depth)
    if depth == 0:
        return int(random.choice(palette))

    while True:
        forced = set(random.sample(range(4), random.randint(1, 4)))
        children: list[Legend] = []
        for idx in range(4):
            child_depth = depth - 1 if idx in forced else random.randint(0, depth - 1)
            children.append(_sample_legend_exact_depth(child_depth, palette))
        node = tuple(children)
        patch = _render_legend(node)
        if len({_legend_signature(child) for child in children}) == 1:
            continue
        if len(np.unique(patch)) < 2:
            continue
        return node


def _max_corner_depth(h: int, w: int) -> int:
    half_h = int(h) // 2
    half_w = int(w) // 2
    depth = 0
    side = 2
    while side <= half_h and side <= half_w and half_h % side == 0 and half_w % side == 0 and depth < MAX_DEPTH:
        depth += 1
        side *= 2
    return depth


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each gray rectangle from recursive corner legends and erase the legends afterward.

    Input structure:
    - The background is black.
    - One or more solid gray rectangles appear on the grid.
    - Each rectangle has four diagonal corner legends just outside its bounding box; each legend is a colored square whose side is a power of two, with side 1 meaning a single-pixel marker and larger sides encoding nested 2x2 sub-legends.

    Rule:
    - Remove every external legend and process each gray rectangle independently.
    - Split the rectangle into four equal quadrants matched to the top-left, top-right, bottom-left, and bottom-right legends.
    - A 1x1 legend paints its whole quadrant one color; a larger legend recursively splits the quadrant into four equal sub-quadrants and applies the same rule to the legend's four sub-squares.

    Key insight:
    - The corner marker is not just a color token; it is itself a miniature recursive blueprint whose 2x2 geometry must be read in the same global orientation as the quadrant it controls.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = int(get_bg(g))
    assert bg == BG

    rectangles = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[GREY],
        return_format="dict",
    )
    assert rectangles

    out = np.full_like(g, bg)
    marker_cells: set[tuple[int, int]] = set()

    for rect_obj in rectangles:
        r0, c0, r1, c1 = map(int, rect_obj["bbox"])
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert rect_obj["area"] == h * w
        assert h % 4 == 0 and w % 4 == 0
        assert np.all(g[r0:r1 + 1, c0:c1 + 1] == GREY)

        rect = (r0, c0, r1, c1)
        patches = []
        for corner in CORNERS:
            patch = _extract_marker_patch(g, rect, corner, bg)
            side = int(patch.shape[0])
            assert (h // 2) % side == 0 and (w // 2) % side == 0
            coords = set(_corner_coords(rect, corner, side))
            assert not (marker_cells & coords), "marker patches must not overlap"
            marker_cells |= coords
            patches.append(patch)

        rm = r0 + h // 2
        cm = c0 + w // 2
        regions = [
            (r0, c0, rm, cm),
            (r0, cm, rm, c1 + 1),
            (rm, c0, r1 + 1, cm),
            (rm, cm, r1 + 1, c1 + 1),
        ]
        for region, patch in zip(regions, patches):
            _paint_patch(out, region, patch)

    actual_markers = {
        (int(r), int(c))
        for r, c in np.argwhere((g != bg) & (g != GREY))
    }
    assert actual_markers == marker_cells
    return out


def _bbox_intersects(a: BBox, b: BBox) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expanded_bbox(footprint: BBox, gap: int = OBJECT_GAP) -> BBox:
    return (
        footprint[0] - gap,
        footprint[1] - gap,
        footprint[2] + gap,
        footprint[3] + gap,
    )


def _marker_depth_plan(max_depths: list[int]) -> list[int] | None:
    plans: list[int] = []
    for md in max_depths:
        if md == 0:
            plans.append(0)
            continue
        choices = list(range(md + 1))
        weights = [2, 4, 2][: md + 1]
        plans.append(random.choices(choices, weights=weights, k=1)[0])

    composite_slots = [i for i, md in enumerate(max_depths) if md >= 1]
    deep_slots = [i for i, md in enumerate(max_depths) if md >= 2]
    if not composite_slots:
        return None

    if all(depth == 0 for depth in plans):
        idx = random.choice(composite_slots)
        plans[idx] = 1 if max_depths[idx] == 1 else random.choice([1, 2])

    if all(depth > 0 for depth in plans):
        plans[random.randrange(len(plans))] = 0

    if deep_slots and random.random() < 0.65 and all(plans[i] < 2 for i in deep_slots):
        idx = random.choice(deep_slots)
        plans[idx] = 2

    return plans


def _render_input_and_output(
    shape: tuple[int, int],
    placed: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    H, W = shape
    inp = np.full((H, W), BG, dtype=int)
    out = np.full((H, W), BG, dtype=int)

    for spec in placed:
        r0, c0 = spec["origin"]
        h = int(spec["h"])
        w = int(spec["w"])
        rect = (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1))
        inp[r0:r0 + h, c0:c0 + w] = GREY

        rm = r0 + h // 2
        cm = c0 + w // 2
        regions = [
            (r0, c0, rm, cm),
            (r0, cm, rm, c0 + w),
            (rm, c0, r0 + h, cm),
            (rm, cm, r0 + h, c0 + w),
        ]

        for corner, legend, region in zip(CORNERS, spec["legends"], regions):
            patch = _render_legend(legend)
            side = int(patch.shape[0])
            rs, re, cs, ce = _corner_patch_bounds(rect, corner, side)
            inp[rs:re, cs:ce] = patch
            _paint_legend(out, region, legend)

    return inp, out


def _choose_rectangles() -> list[dict[str, object]] | None:
    rect_count = random.choice(RECT_COUNT_OPTIONS)
    specs: list[dict[str, object]] = []
    max_depths: list[int] = []
    for _ in range(rect_count):
        h = random.choice(RECT_SIZE_OPTIONS)
        w = random.choice(RECT_SIZE_OPTIONS)
        md = _max_corner_depth(h, w)
        if md < 1:
            return None
        specs.append({"h": h, "w": w})
        max_depths.extend([md] * 4)

    depth_plan = _marker_depth_plan(max_depths)
    if depth_plan is None:
        return None

    palette = random.sample(MARKER_PALETTE, random.randint(3, min(6, len(MARKER_PALETTE))))
    offset = 0
    for spec in specs:
        h = int(spec["h"])
        w = int(spec["w"])
        md = _max_corner_depth(h, w)
        legends = []
        depths = depth_plan[offset:offset + 4]
        offset += 4
        if not any(depth == 0 for depth in depths):
            depths[random.randrange(4)] = 0
        if not any(depth > 0 for depth in depths):
            depths[random.randrange(4)] = min(1, md)
        for depth in depths:
            legends.append(_sample_legend_exact_depth(depth, palette))
        spec["legends"] = tuple(legends)
        spec["depths"] = tuple(depths)
    return specs


def _footprint_for_spec(spec: dict[str, object], origin: tuple[int, int]) -> BBox:
    r0, c0 = origin
    h = int(spec["h"])
    w = int(spec["w"])
    sizes = [_render_legend(legend).shape[0] for legend in spec["legends"]]
    tl, tr, bl, br = map(int, sizes)
    top = max(tl, tr)
    bottom = max(bl, br)
    left = max(tl, bl)
    right = max(tr, br)
    return (r0 - top, c0 - left, r0 + h - 1 + bottom, c0 + w - 1 + right)


def _place_specs(shape: tuple[int, int], specs: list[dict[str, object]]) -> list[dict[str, object]] | None:
    H, W = shape
    placed: list[dict[str, object]] = []
    ordered = sorted(specs, key=lambda spec: int(spec["h"]) * int(spec["w"]), reverse=True)

    for spec in ordered:
        h = int(spec["h"])
        w = int(spec["w"])
        sizes = [_render_legend(legend).shape[0] for legend in spec["legends"]]
        tl, tr, bl, br = map(int, sizes)
        top = max(tl, tr)
        bottom = max(bl, br)
        left = max(tl, bl)
        right = max(tr, br)
        if top + h + bottom > H or left + w + right > W:
            return None

        found = False
        for _ in range(PLACEMENT_TRIES):
            r0 = random.randint(top, H - h - bottom)
            c0 = random.randint(left, W - w - right)
            footprint = _footprint_for_spec(spec, (r0, c0))
            expanded = _expanded_bbox(footprint)
            if expanded[0] < 0 or expanded[1] < 0 or expanded[2] >= H or expanded[3] >= W:
                continue
            if any(_bbox_intersects(expanded, other["expanded"]) for other in placed):
                continue
            placed.append({**spec, "origin": (r0, c0), "footprint": footprint, "expanded": expanded})
            found = True
            break
        if not found:
            return None
    return placed


def _grid_is_clear(inp: np.ndarray, out: np.ndarray, specs: list[dict[str, object]]) -> bool:
    if inp.ndim != 2 or out.ndim != 2:
        return False
    if max(inp.shape) > MAX_ARC_SIZE or max(out.shape) > MAX_ARC_SIZE:
        return False
    if get_bg(inp) != BG:
        return False
    if len(np.unique(out[out != BG])) < 3:
        return False
    if not np.any((inp == GREY) & (out != GREY)):
        return False
    return any(depth > 0 for spec in specs for depth in spec["depths"])


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(CANVAS_MIN, CANVAS_MAX)
    W = random.randint(CANVAS_MIN, CANVAS_MAX)
    specs = _choose_rectangles()
    if specs is None:
        return None
    placed = _place_specs((H, W), specs)
    if placed is None:
        return None

    inp, expected = _render_input_and_output((H, W), placed)
    if not _grid_is_clear(inp, expected, placed):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid recursive quadrant-legend input")
