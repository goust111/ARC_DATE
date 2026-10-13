from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import find_components, get_bg


GRID_BG = 0
GRID_MAX = 30
COLORS = tuple(range(1, 10))
BUILD_ATTEMPTS = 400
PAIR_ATTEMPTS = 160


def _overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _bbox_from_center(center_r2: int, center_c2: int, h: int, w: int) -> tuple[int, int, int, int]:
    r0 = (int(center_r2) - (int(h) - 1)) // 2
    c0 = (int(center_c2) - (int(w) - 1)) // 2
    return r0, c0, r0 + int(h) - 1, c0 + int(w) - 1


def _bbox_dims(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return r1 - r0 + 1, c1 - c0 + 1


def _paint_rect(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _is_solid_rect(grid: np.ndarray, obj: dict, color: int) -> bool:
    r0, c0, r1, c1 = obj["bbox"]
    area = int(obj["area"])
    return area == (r1 - r0 + 1) * (c1 - c0 + 1) and np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == int(color))


def _is_block_2x2(grid: np.ndarray, obj: dict, color: int) -> bool:
    r0, c0, r1, c1 = obj["bbox"]
    return (
        int(obj["area"]) == 4
        and r1 - r0 == 1
        and c1 - c0 == 1
        and np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == int(color))
    )


def _normalized_dims(h: int, w: int, axis: str) -> tuple[int, int]:
    if axis == "h":
        return min(h, w), max(h, w)
    return max(h, w), min(h, w)


def _parse_pairs(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    objects = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    assert objects, "expected foreground objects"

    by_color: dict[int, list[dict]] = {}
    for obj in objects:
        by_color.setdefault(int(obj["color"]), []).append(obj)

    pairs: list[dict] = []
    for color, comps in sorted(by_color.items()):
        assert len(comps) == 2, f"color {color} must have one rectangle and one block"
        rects = [obj for obj in comps if _is_solid_rect(g, obj, color) and int(obj["area"]) > 4]
        blocks = [obj for obj in comps if _is_block_2x2(g, obj, color)]
        assert len(rects) == 1, f"color {color} must have exactly one solid rectangle"
        assert len(blocks) == 1, f"color {color} must have exactly one 2x2 block"
        rect = rects[0]
        block = blocks[0]
        h, w = _bbox_dims(rect["bbox"])
        assert h != w, "source rectangle must be non-square"

        rr0, rc0, rr1, rc1 = rect["bbox"]
        br0, bc0, br1, bc1 = block["bbox"]
        row_overlap = _overlap(rr0, rr1, br0, br1)
        col_overlap = _overlap(rc0, rc1, bc0, bc1)
        assert row_overlap ^ col_overlap, "pair must define exactly one sightline axis"
        axis = "h" if row_overlap else "v"

        norm_h, norm_w = _normalized_dims(h, w, axis)
        center_r2 = rr0 + rr1
        center_c2 = rc0 + rc1
        norm_bbox = _bbox_from_center(center_r2, center_c2, norm_h, norm_w)

        pair = {
            "color": int(color),
            "rect_bbox": tuple(map(int, rect["bbox"])),
            "block_bbox": tuple(map(int, block["bbox"])),
            "axis": axis,
            "dims": (int(h), int(w)),
            "norm_dims": (int(norm_h), int(norm_w)),
            "norm_bbox": norm_bbox,
            "swap": (norm_h, norm_w) != (h, w),
        }
        pairs.append(pair)
    assert 3 <= len(pairs) <= 4, "expected 3 to 4 color pairs"
    return pairs


def _final_bbox(pair: dict) -> tuple[int, int, int, int]:
    nr0, nc0, _nr1, _nc1 = pair["norm_bbox"]
    nh, nw = pair["norm_dims"]
    br0, bc0, br1, bc1 = pair["block_bbox"]

    if pair["axis"] == "h":
        block_center_c2 = bc0 + bc1
        rect_center_c2 = nc0 + (nc0 + nw - 1)
        if rect_center_c2 < block_center_c2:
            fc0 = bc0 - nw
        else:
            fc0 = bc1 + 1
        return nr0, fc0, nr0 + nh - 1, fc0 + nw - 1

    block_center_r2 = br0 + br1
    rect_center_r2 = nr0 + (nr0 + nh - 1)
    if rect_center_r2 < block_center_r2:
        fr0 = br0 - nh
    else:
        fr0 = br1 + 1
    return fr0, nc0, fr0 + nh - 1, nc0 + nw - 1


def _solve_from_pairs(grid: np.ndarray, pairs: list[dict]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = np.full(g.shape, bg, dtype=int)
    occupied = np.zeros(g.shape, dtype=bool)

    for pair in pairs:
        color = int(pair["color"])
        fr0, fc0, fr1, fc1 = _final_bbox(pair)
        assert 0 <= fr0 <= fr1 < g.shape[0] and 0 <= fc0 <= fc1 < g.shape[1], "final rectangle must stay in bounds"
        footprint = occupied[fr0 : fr1 + 1, fc0 : fc1 + 1]
        assert not footprint.any(), "final rectangles must not overlap"
        out[fr0 : fr1 + 1, fc0 : fc1 + 1] = color
        occupied[fr0 : fr1 + 1, fc0 : fc1 + 1] = True
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Normalize each same-color rectangle to the sightline axis, then slide it next to its paired 2x2 block and erase the block.

    Input structure:
    - The background is black.
    - There are 3 to 4 foreground colors, each appearing in exactly two monochrome objects.
    - For each color, one object is a non-square solid rectangle larger than 2x2 and the other is a solid 2x2 block.
    - The block overlaps the rectangle only in row range or only in column range, defining a horizontal or vertical sightline.

    Rule:
    - Pair each rectangle with the same-color 2x2 block.
    - If their row ranges overlap, normalize the rectangle to horizontal orientation; if their column ranges overlap, normalize it to vertical orientation.
    - Normalization swaps height and width only when needed, while keeping the rectangle centered on its original center with half-cell cases rounded toward the upper-left.
    - After all rectangles are normalized, slide each one along its sightline axis toward its block until the rectangle is edge-adjacent to the block, then erase every 2x2 block.

    Key insight:
    - The external spatial relation to the same-color block does double duty: it chooses both the movement axis and whether the rectangle must be turned into a wide or tall normalized version before moving.
    """
    pairs = _parse_pairs(np.asarray(grid, dtype=int))
    return _solve_from_pairs(np.asarray(grid, dtype=int), pairs)


def _corridor_clear(grid: np.ndarray, pair: dict) -> bool:
    rr0, rc0, rr1, rc1 = pair["rect_bbox"]
    br0, bc0, br1, bc1 = pair["block_bbox"]
    bg = get_bg(grid)

    if pair["axis"] == "h":
        rows = range(max(rr0, br0), min(rr1, br1) + 1)
        if rc1 < bc0:
            for r in rows:
                if np.any(grid[r, rc1 + 1 : bc0] != bg):
                    return False
        else:
            for r in rows:
                if np.any(grid[r, bc1 + 1 : rc0] != bg):
                    return False
        return True

    cols = range(max(rc0, bc0), min(rc1, bc1) + 1)
    if rr1 < br0:
        for c in cols:
            if np.any(grid[rr1 + 1 : br0, c] != bg):
                return False
    else:
        for c in cols:
            if np.any(grid[br1 + 1 : rr0, c] != bg):
                return False
    return True


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        g = np.asarray(grid, dtype=int)
        assert g.ndim == 2 and max(g.shape) <= GRID_MAX
        pairs = _parse_pairs(g)
        axes = {pair["axis"] for pair in pairs}
        swaps = {bool(pair["swap"]) for pair in pairs}
        assert axes == {"h", "v"}, "generator must cover both sightline axes"
        assert swaps == {True, False}, "generator must cover swap and no-swap branches"

        centers = []
        for pair in pairs:
            rr0, rc0, rr1, rc1 = pair["rect_bbox"]
            nr0, nc0, nr1, nc1 = pair["norm_bbox"]
            fr0, fc0, fr1, fc1 = _final_bbox(pair)
            br0, bc0, br1, bc1 = pair["block_bbox"]

            assert 0 <= nr0 <= nr1 < g.shape[0] and 0 <= nc0 <= nc1 < g.shape[1]
            assert 0 <= fr0 <= fr1 < g.shape[0] and 0 <= fc0 <= fc1 < g.shape[1]
            assert _corridor_clear(g, pair)

            if pair["axis"] == "h":
                assert nr0 == fr0 and nr1 == fr1
                if rc1 < bc0:
                    assert fc0 + pair["norm_dims"][1] == bc0
                else:
                    assert fc0 == bc1 + 1
            else:
                assert nc0 == fc0 and nc1 == fc1
                if rr1 < br0:
                    assert fr0 + pair["norm_dims"][0] == br0
                else:
                    assert fr0 == br1 + 1

            centers.append((rr0 + rr1, rc0 + rc1, nr0 + nr1, nc0 + nc1))

        assert any(r2 != nr2 or c2 != nc2 for r2, c2, nr2, nc2 in centers), "need at least one visible recentering case"

        out = main(g)
        assert out.shape == g.shape and max(out.shape) <= GRID_MAX
        out_pairs = _parse_output(out, len(pairs))
        expected = {(pair["color"], _final_bbox(pair)) for pair in pairs}
        observed = {(item["color"], item["bbox"]) for item in out_pairs}
        assert observed == expected
        return True
    except Exception:
        return False


def _parse_output(grid: np.ndarray, expected_count: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    objects = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    assert len(objects) == expected_count, "output must contain one rectangle per input color"
    parsed: list[dict] = []
    for obj in objects:
        color = int(obj["color"])
        assert _is_solid_rect(g, obj, color), "output objects must remain solid rectangles"
        assert int(obj["area"]) > 4, "output must not contain 2x2 blocks"
        parsed.append({"color": color, "bbox": tuple(map(int, obj["bbox"]))})
    return parsed


_CATEGORY_CANDIDATES: dict[tuple[str, bool], list[tuple[int, int]]] = {("h", True): [], ("h", False): [], ("v", True): [], ("v", False): []}
for _h in range(1, 7):
    for _w in range(1, 7):
        if _h == _w or _h * _w <= 4:
            continue
        if _h > _w:
            _CATEGORY_CANDIDATES[("h", True)].append((_h, _w))
            _CATEGORY_CANDIDATES[("v", False)].append((_h, _w))
        if _w > _h:
            _CATEGORY_CANDIDATES[("h", False)].append((_h, _w))
            _CATEGORY_CANDIDATES[("v", True)].append((_h, _w))


def _choose_dims(axis: str, need_swap: bool, require_mixed_parity: bool) -> tuple[int, int]:
    choices = _CATEGORY_CANDIDATES[(axis, need_swap)]
    if require_mixed_parity:
        mixed = [dims for dims in choices if (dims[0] - dims[1]) % 2 != 0]
        if mixed:
            choices = mixed
    return random.choice(choices)


def _build_pair_zone(
    zone_h: int,
    zone_w: int,
    color: int,
    axis: str,
    need_swap: bool,
    require_mixed_parity: bool,
) -> tuple[np.ndarray, dict] | None:
    for _ in range(PAIR_ATTEMPTS):
        h0, w0 = _choose_dims(axis, need_swap, require_mixed_parity)
        nh, nw = _normalized_dims(h0, w0, axis)
        if zone_h < max(h0, nh, 2) or zone_w < max(w0, nw, 2):
            continue

        nr0 = random.randint(0, zone_h - nh)
        nc0 = random.randint(0, zone_w - nw)
        center_r2 = nr0 + (nr0 + nh - 1)
        center_c2 = nc0 + (nc0 + nw - 1)
        rr0, rc0, rr1, rc1 = _bbox_from_center(center_r2, center_c2, h0, w0)
        if not (0 <= rr0 <= rr1 < zone_h and 0 <= rc0 <= rc1 < zone_w):
            continue

        if axis == "h":
            directions = ["right", "left"]
            random.shuffle(directions)
            for direction in directions:
                max_slide = zone_w - (nc0 + nw + 2) if direction == "right" else nc0 - 2
                if max_slide < 2:
                    continue
                slide = random.randint(2, max_slide)
                final_c0 = nc0 + slide if direction == "right" else nc0 - slide
                block_c0 = final_c0 + nw if direction == "right" else final_c0 - 2
                row_options = [
                    br0
                    for br0 in range(zone_h - 1)
                    if _overlap(br0, br0 + 1, nr0, nr0 + nh - 1) and _overlap(br0, br0 + 1, rr0, rr1)
                ]
                if not row_options:
                    continue
                block_r0 = random.choice(row_options)
                zone = np.zeros((zone_h, zone_w), dtype=int)
                _paint_rect(zone, (rr0, rc0, rr1, rc1), color)
                _paint_rect(zone, (block_r0, block_c0, block_r0 + 1, block_c0 + 1), color)
                meta = {
                    "axis": axis,
                    "swap": need_swap,
                    "dims": (h0, w0),
                    "norm_dims": (nh, nw),
                    "rect_bbox": (rr0, rc0, rr1, rc1),
                    "block_bbox": (block_r0, block_c0, block_r0 + 1, block_c0 + 1),
                    "norm_bbox": (nr0, nc0, nr0 + nh - 1, nc0 + nw - 1),
                }
                if _validate_local_zone(zone, meta):
                    return zone, meta
        else:
            directions = ["down", "up"]
            random.shuffle(directions)
            for direction in directions:
                max_slide = zone_h - (nr0 + nh + 2) if direction == "down" else nr0 - 2
                if max_slide < 2:
                    continue
                slide = random.randint(2, max_slide)
                final_r0 = nr0 + slide if direction == "down" else nr0 - slide
                block_r0 = final_r0 + nh if direction == "down" else final_r0 - 2
                col_options = [
                    bc0
                    for bc0 in range(zone_w - 1)
                    if _overlap(bc0, bc0 + 1, nc0, nc0 + nw - 1) and _overlap(bc0, bc0 + 1, rc0, rc1)
                ]
                if not col_options:
                    continue
                block_c0 = random.choice(col_options)
                zone = np.zeros((zone_h, zone_w), dtype=int)
                _paint_rect(zone, (rr0, rc0, rr1, rc1), color)
                _paint_rect(zone, (block_r0, block_c0, block_r0 + 1, block_c0 + 1), color)
                meta = {
                    "axis": axis,
                    "swap": need_swap,
                    "dims": (h0, w0),
                    "norm_dims": (nh, nw),
                    "rect_bbox": (rr0, rc0, rr1, rc1),
                    "block_bbox": (block_r0, block_c0, block_r0 + 1, block_c0 + 1),
                    "norm_bbox": (nr0, nc0, nr0 + nh - 1, nc0 + nw - 1),
                }
                if _validate_local_zone(zone, meta):
                    return zone, meta
    return None


def _validate_local_zone(zone: np.ndarray, meta: dict) -> bool:
    pair = {
        "color": 1,
        "rect_bbox": meta["rect_bbox"],
        "block_bbox": meta["block_bbox"],
        "axis": meta["axis"],
        "dims": meta["dims"],
        "norm_dims": meta["norm_dims"],
        "norm_bbox": meta["norm_bbox"],
        "swap": meta["swap"],
    }
    try:
        parsed = _parse_pairs(zone)
        assert len(parsed) == 1
    except Exception:
        pass
    try:
        assert _corridor_clear(zone, pair)
        fr0, fc0, fr1, fc1 = _final_bbox(pair)
        assert 0 <= fr0 <= fr1 < zone.shape[0] and 0 <= fc0 <= fc1 < zone.shape[1]
        return True
    except Exception:
        return False


def _choose_specs(pair_count: int) -> list[tuple[str, bool]]:
    if pair_count == 4:
        specs = [("h", True), ("h", False), ("v", True), ("v", False)]
        random.shuffle(specs)
        return specs

    h_spec = ("h", random.choice([True, False]))
    v_spec = ("v", random.choice([True, False]))
    specs = [h_spec, v_spec]
    covered = {spec[1] for spec in specs}
    if covered == {True, False}:
        remaining = [("h", not h_spec[1]), ("v", not v_spec[1])]
        specs.append(random.choice(remaining))
    elif True in covered:
        specs.append(random.choice([("h", False), ("v", False)]))
    else:
        specs.append(random.choice([("h", True), ("v", True)]))
    random.shuffle(specs)
    return specs


def _build_grid() -> np.ndarray | None:
    top_h = random.randint(8, 11)
    bot_h = random.randint(8, 11)
    left_w = random.randint(9, 12)
    right_w = random.randint(9, 12)
    gap_r = random.randint(2, 4)
    gap_c = random.randint(2, 4)
    H = top_h + gap_r + bot_h
    W = left_w + gap_c + right_w
    if max(H, W) > GRID_MAX:
        return None

    zones = [
        (0, 0, top_h, left_w),
        (0, left_w + gap_c, top_h, right_w),
        (top_h + gap_r, 0, bot_h, left_w),
        (top_h + gap_r, left_w + gap_c, bot_h, right_w),
    ]
    pair_count = random.choice([3, 4])
    zone_indices = random.sample(range(4), pair_count)
    specs = _choose_specs(pair_count)
    colors = random.sample(COLORS, pair_count)
    swap_indices = [i for i, spec in enumerate(specs) if spec[1]]
    parity_swap_index = random.choice(swap_indices)

    grid = np.zeros((H, W), dtype=int)
    for idx, zone_index in enumerate(zone_indices):
        zr0, zc0, zh, zw = zones[zone_index]
        axis, need_swap = specs[idx]
        built = _build_pair_zone(
            zh,
            zw,
            colors[idx],
            axis,
            need_swap,
            require_mixed_parity=(idx == parity_swap_index),
        )
        if built is None:
            return None
        zone_grid, _meta = built
        region = grid[zr0 : zr0 + zh, zc0 : zc0 + zw]
        assert not np.any((region != 0) & (zone_grid != 0))
        region[zone_grid != 0] = zone_grid[zone_grid != 0]

    if not _validate_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(BUILD_ATTEMPTS):
        grid = _build_grid()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid instance after {BUILD_ATTEMPTS} attempts")
