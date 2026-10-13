from __future__ import annotations

from collections import Counter
from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_shared_arc_utils",
    str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py")),
).load_module()

Color = _ARC_UTILS.Color
bbox_coords = _ARC_UTILS.bbox_coords
components_mask = _ARC_UTILS.components_mask
in_bounds = _ARC_UTILS.in_bounds


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


SIDE_ORDER = ("top", "right", "bottom", "left")
DIR_BY_SIDE = {
    "top": (1, 0),
    "bottom": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}

FRAME_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW)
FLOOR_COLORS = (Color.PINK, Color.TEAL, Color.MAROON)
CORNER_COLOR = Color.GREY
TRAIL_FINAL_COLOR = Color.ORANGE

GRID_H_RANGE = (14, 22)
GRID_W_RANGE = (14, 22)
FRAME_MARGIN_RANGE = (1, 3)
INNER_H_RANGE = (9, 16)
INNER_W_RANGE = (9, 16)
BLOCK_COUNT_RANGE = (2, 4)
PLACEMENT_RETRIES = 400


def _edge_bg(grid: np.ndarray) -> int:
    border = np.concatenate((grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]))
    return int(Counter(map(int, border)).most_common(1)[0][0])


def _parse_scene(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _edge_bg(g)

    non_bg = np.argwhere(g != bg)
    assert non_bg.size > 0, "expected a framed structure"
    fr0, fc0, fr1, fc1 = bbox_coords(non_bg)
    assert fr1 - fr0 >= 2 and fc1 - fc0 >= 2, "frame too small"

    corner_color = int(g[fr0, fc0])
    corners = [(fr0, fc0), (fr0, fc1), (fr1, fc0), (fr1, fc1)]
    assert all(int(g[r, c]) == corner_color for r, c in corners), "frame corners must agree"

    side_colors = {
        "top": int(g[fr0, fc0 + 1]),
        "right": int(g[fr0 + 1, fc1]),
        "bottom": int(g[fr1, fc0 + 1]),
        "left": int(g[fr0 + 1, fc0]),
    }
    assert np.all(g[fr0, fc0 + 1 : fc1] == side_colors["top"])
    assert np.all(g[fr1, fc0 + 1 : fc1] == side_colors["bottom"])
    assert np.all(g[fr0 + 1 : fr1, fc0] == side_colors["left"])
    assert np.all(g[fr0 + 1 : fr1, fc1] == side_colors["right"])

    inner = (fr0 + 1, fc0 + 1, fr1 - 1, fc1 - 1)
    ir0, ic0, ir1, ic1 = inner
    assert ir0 <= ir1 and ic0 <= ic1

    gate_centers: dict[Coord, dict[str, object]] = {}
    for r in range(ir0, ir1 + 1):
        for c in range(ic0, ic1 + 1):
            if int(g[r, c]) != corner_color:
                continue
            horiz_ok = (
                c - 1 >= ic0
                and c + 1 <= ic1
                and int(g[r, c - 1]) == int(g[r, c + 1])
                and int(g[r, c - 1]) in {side_colors["top"], side_colors["bottom"]}
            )
            vert_ok = (
                r - 1 >= ir0
                and r + 1 <= ir1
                and int(g[r - 1, c]) == int(g[r + 1, c])
                and int(g[r - 1, c]) in {side_colors["left"], side_colors["right"]}
            )
            if horiz_ok == vert_ok:
                continue
            if horiz_ok:
                gate_centers[(r, c)] = {
                    "center": (r, c),
                    "orientation": "horizontal",
                    "end_color": int(g[r, c - 1]),
                    "cells": {(r, c - 1), (r, c), (r, c + 1)},
                }
            else:
                gate_centers[(r, c)] = {
                    "center": (r, c),
                    "orientation": "vertical",
                    "end_color": int(g[r - 1, c]),
                    "cells": {(r - 1, c), (r, c), (r + 1, c)},
                }
    gates = list(gate_centers.values())

    interior = g[ir0 : ir1 + 1, ic0 : ic1 + 1]
    blocked_colors = {bg, corner_color, *side_colors.values()}
    remaining = [
        int(v)
        for v in interior.ravel()
        if int(v) not in blocked_colors
    ]
    assert remaining, "expected a visible floor color inside the frame"
    floor_color = Counter(remaining).most_common(1)[0][0]

    block_mask = np.zeros_like(g, dtype=bool)
    block_mask[ir0 : ir1 + 1, ic0 : ic1 + 1] = interior == bg
    blocks = []
    for coords in components_mask(block_mask, connectivity=4):
        bb = bbox_coords(coords)
        r0, c0, r1, c1 = bb
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == bg), "blocks must be solid rectangles"
        touches = []
        if r0 == ir0:
            touches.append("top")
        if r1 == ir1:
            touches.append("bottom")
        if c0 == ic0:
            touches.append("left")
        if c1 == ic1:
            touches.append("right")
        assert len(touches) == 1, "each block must touch exactly one inner frame side"
        blocks.append({"coords": coords, "bbox": bb, "side": touches[0]})

    assert blocks, "expected at least one movable block"

    return {
        "grid": g,
        "bg": bg,
        "frame_bbox": (fr0, fc0, fr1, fc1),
        "inner_bbox": inner,
        "corner_color": corner_color,
        "side_colors": side_colors,
        "floor_color": floor_color,
        "gates": gates,
        "blocks": blocks,
    }


def _lane_rect(block: dict[str, object], inner_bbox: BBox) -> BBox:
    ir0, ic0, ir1, ic1 = inner_bbox
    r0, c0, r1, c1 = block["bbox"]
    side = block["side"]
    if side in {"top", "bottom"}:
        return (ir0, c0, ir1, c1)
    return (r0, ic0, r1, ic1)


def _gate_for_block(block: dict[str, object], scene: dict[str, object]) -> dict[str, object] | None:
    r0, c0, r1, c1 = block["bbox"]
    side = block["side"]
    candidates = []
    for gate in scene["gates"]:
        gr, gc = gate["center"]
        if side == "top":
            if gate["orientation"] == "horizontal" and c0 <= gc <= c1 and gr > r1:
                candidates.append(gate)
        elif side == "bottom":
            if gate["orientation"] == "horizontal" and c0 <= gc <= c1 and gr < r0:
                candidates.append(gate)
        elif side == "left":
            if gate["orientation"] == "vertical" and r0 <= gr <= r1 and gc > c1:
                candidates.append(gate)
        else:
            if gate["orientation"] == "vertical" and r0 <= gr <= r1 and gc < c0:
                candidates.append(gate)

    if not candidates:
        return None
    if side == "top":
        return min(candidates, key=lambda gate: gate["center"][0])
    if side == "bottom":
        return max(candidates, key=lambda gate: gate["center"][0])
    if side == "left":
        return min(candidates, key=lambda gate: gate["center"][1])
    return max(candidates, key=lambda gate: gate["center"][1])


def _final_bbox(block: dict[str, object], scene: dict[str, object]) -> BBox:
    ir0, ic0, ir1, ic1 = scene["inner_bbox"]
    r0, c0, r1, c1 = block["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    gate = _gate_for_block(block, scene)
    side = block["side"]

    if side == "top":
        nr0 = (gate["center"][0] - h) if gate else (ir1 - h + 1)
        return (nr0, c0, nr0 + h - 1, c1)
    if side == "bottom":
        nr0 = (gate["center"][0] + 1) if gate else ir0
        return (nr0, c0, nr0 + h - 1, c1)
    if side == "left":
        nc0 = (gate["center"][1] - w) if gate else (ic1 - w + 1)
        return (r0, nc0, r1, nc0 + w - 1)
    nc0 = (gate["center"][1] + 1) if gate else ic0
    return (r0, nc0, r1, nc0 + w - 1)


def _block_motion(block: dict[str, object], scene: dict[str, object]) -> tuple[int, int, int]:
    r0, c0, _r1, _c1 = block["bbox"]
    nr0, nc0, _nr1, _nc1 = _final_bbox(block, scene)
    dr, dc = DIR_BY_SIDE[block["side"]]
    steps = abs(nr0 - r0) + abs(nc0 - c0)
    assert steps > 0, "every block must move"
    return dr, dc, steps


def _paint_rect(out: np.ndarray, bbox: BBox, color: int) -> None:
    r0, c0, r1, c1 = bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _transform(scene: dict[str, object]) -> np.ndarray:
    g = np.array(scene["grid"], copy=True)
    out = g.copy()

    for block in scene["blocks"]:
        dr, dc, steps = _block_motion(block, scene)
        r0, c0, r1, c1 = block["bbox"]
        trail_color = scene["side_colors"][block["side"]]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        for step in range(steps):
            rr0 = r0 + dr * step
            cc0 = c0 + dc * step
            _paint_rect(out, (rr0, cc0, rr0 + h - 1, cc0 + w - 1), trail_color)
        _paint_rect(out, _final_bbox(block, scene), TRAIL_FINAL_COLOR)

    fr0, fc0, fr1, fc1 = scene["frame_bbox"]
    out[fr0, fc0] = scene["corner_color"]
    out[fr0, fc1] = scene["corner_color"]
    out[fr1, fc0] = scene["corner_color"]
    out[fr1, fc1] = scene["corner_color"]
    out[fr0, fc0 + 1 : fc1] = scene["side_colors"]["top"]
    out[fr1, fc0 + 1 : fc1] = scene["side_colors"]["bottom"]
    out[fr0 + 1 : fr1, fc0] = scene["side_colors"]["left"]
    out[fr0 + 1 : fr1, fc1] = scene["side_colors"]["right"]
    for gate in scene["gates"]:
        for r, c in gate["cells"]:
            out[r, c] = g[r, c]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each edge-attached black slider across a framed corridor, stopping at gates and painting a side-colored trail.

    Input structure:
    - The grid has black outer background and one rectangular one-pixel frame with four colored sides and shared gray corners.
    - Inside the frame, a uniform visible floor color contains several solid black rectangles touching exactly one inner frame side.
    - The interior may also contain 3-pixel gate motifs whose gray center is flanked by equal side-colored endpoints, perpendicular to a slider lane.

    Rule:
    - Every black rectangle moves straight from its touched side toward the opposite side of the frame interior.
    - If a gate lies on that rectangle's lane, the rectangle stops immediately before the gate's gray center; otherwise it stops adjacent to the opposite inner frame side.
    - The swept cells from the start position up to but not including the final position are recolored to the touched side's frame color, and the final rectangle becomes orange.
    - Frame cells, floor cells outside the swept lanes, and every gate motif remain unchanged.

    Key insight:
    The output is constrained by three independent cues at once: the touched frame side gives both motion direction and trail color, while the first gate on the same lane truncates the travel distance.
    """
    scene = _parse_scene(np.asarray(grid, dtype=int))
    return _transform(scene)


def _nearer_vertical_color(row: int, inner_bbox: BBox, side_colors: dict[str, int]) -> int:
    ir0, _ic0, ir1, _ic1 = inner_bbox
    top_dist = row - ir0
    bottom_dist = ir1 - row
    if top_dist < bottom_dist:
        return side_colors["top"]
    if bottom_dist < top_dist:
        return side_colors["bottom"]
    return random.choice((side_colors["top"], side_colors["bottom"]))


def _nearer_horizontal_color(col: int, inner_bbox: BBox, side_colors: dict[str, int]) -> int:
    _ir0, ic0, _ir1, ic1 = inner_bbox
    left_dist = col - ic0
    right_dist = ic1 - col
    if left_dist < right_dist:
        return side_colors["left"]
    if right_dist < left_dist:
        return side_colors["right"]
    return random.choice((side_colors["left"], side_colors["right"]))


def _free_segments(occupied: np.ndarray, width: int, gap: int = 1) -> list[int]:
    starts = []
    n = occupied.size
    for s in range(0, n - width + 1):
        left = max(0, s - gap)
        right = min(n, s + width + gap)
        if not occupied[left:right].any():
            starts.append(s)
    return starts


def _build_vertical_case(grid: np.ndarray, inner_bbox: BBox, side_colors: dict[str, int]) -> list[dict[str, object]] | None:
    ir0, ic0, ir1, ic1 = inner_bbox
    inner_w = ic1 - ic0 + 1
    usable_w = inner_w - 2
    max_blocks = min(BLOCK_COUNT_RANGE[1], usable_w // 3)
    if max_blocks < 2:
        return None
    n_blocks = random.randint(2, max_blocks)

    gate_count = random.randint(1, n_blocks - 1)
    gated_ids = set(random.sample(range(n_blocks), gate_count))
    sides = [random.choice(("top", "bottom")) for _ in range(n_blocks)]
    if len(set(sides)) == 1:
        sides[random.randrange(n_blocks)] = "bottom" if sides[0] == "top" else "top"

    order = sorted(range(n_blocks), key=lambda i: (i not in gated_ids, random.random()))
    occupied = np.zeros(inner_w, dtype=bool)
    lanes: list[dict[str, object] | None] = [None] * n_blocks

    for idx in order:
        gated = idx in gated_ids
        widths = list(range(3 if gated else 2, 5))
        random.shuffle(widths)
        placed = False
        for width in widths:
            starts = _free_segments(occupied, width, gap=1)
            starts = [s for s in starts if 1 <= s and s + width <= inner_w - 1]
            random.shuffle(starts)
            for start in starts:
                c0 = ic0 + start
                c1 = c0 + width - 1
                h = random.randint(1, 2)
                side = sides[idx]
                if side == "top":
                    r0 = ir0
                    r1 = r0 + h - 1
                else:
                    r1 = ir1
                    r0 = r1 - h + 1
                gate = None
                if gated:
                    gate_col = random.randint(c0 + 1, c1 - 1)
                    if side == "top":
                        gate_rows = list(range(r1 + 2, ir1))
                    else:
                        gate_rows = list(range(ir0 + 1, r0 - 1))
                    if not gate_rows:
                        continue
                    gate_row = random.choice(gate_rows)
                    gate = {
                        "center": (gate_row, gate_col),
                        "orientation": "horizontal",
                        "end_color": _nearer_vertical_color(gate_row, inner_bbox, side_colors),
                    }
                lanes[idx] = {
                    "bbox": (r0, c0, r1, c1),
                    "side": side,
                    "gate": gate,
                }
                occupied[start : start + width] = True
                placed = True
                break
            if placed:
                break
        if not placed:
            return None
    return [lane for lane in lanes if lane is not None]


def _build_horizontal_case(grid: np.ndarray, inner_bbox: BBox, side_colors: dict[str, int]) -> list[dict[str, object]] | None:
    ir0, ic0, ir1, ic1 = inner_bbox
    inner_h = ir1 - ir0 + 1
    usable_h = inner_h - 2
    max_blocks = min(BLOCK_COUNT_RANGE[1], usable_h // 3)
    if max_blocks < 2:
        return None
    n_blocks = random.randint(2, max_blocks)

    gate_count = random.randint(1, n_blocks - 1)
    gated_ids = set(random.sample(range(n_blocks), gate_count))
    sides = [random.choice(("left", "right")) for _ in range(n_blocks)]
    if len(set(sides)) == 1:
        sides[random.randrange(n_blocks)] = "right" if sides[0] == "left" else "left"

    order = sorted(range(n_blocks), key=lambda i: (i not in gated_ids, random.random()))
    occupied = np.zeros(inner_h, dtype=bool)
    lanes: list[dict[str, object] | None] = [None] * n_blocks

    for idx in order:
        gated = idx in gated_ids
        heights = list(range(3 if gated else 2, 5))
        random.shuffle(heights)
        placed = False
        for height in heights:
            starts = _free_segments(occupied, height, gap=1)
            starts = [s for s in starts if 1 <= s and s + height <= inner_h - 1]
            random.shuffle(starts)
            for start in starts:
                r0 = ir0 + start
                r1 = r0 + height - 1
                w = random.randint(1, 2)
                side = sides[idx]
                if side == "left":
                    c0 = ic0
                    c1 = c0 + w - 1
                else:
                    c1 = ic1
                    c0 = c1 - w + 1
                gate = None
                if gated:
                    gate_row = random.randint(r0 + 1, r1 - 1)
                    if side == "left":
                        gate_cols = list(range(c1 + 2, ic1))
                    else:
                        gate_cols = list(range(ic0 + 1, c0 - 1))
                    if not gate_cols:
                        continue
                    gate_col = random.choice(gate_cols)
                    gate = {
                        "center": (gate_row, gate_col),
                        "orientation": "vertical",
                        "end_color": _nearer_horizontal_color(gate_col, inner_bbox, side_colors),
                    }
                lanes[idx] = {
                    "bbox": (r0, c0, r1, c1),
                    "side": side,
                    "gate": gate,
                }
                occupied[start : start + height] = True
                placed = True
                break
            if placed:
                break
        if not placed:
            return None
    return [lane for lane in lanes if lane is not None]


def _draw_frame(grid: np.ndarray, frame_bbox: BBox, side_colors: dict[str, int]) -> None:
    fr0, fc0, fr1, fc1 = frame_bbox
    grid[fr0, fc0] = CORNER_COLOR
    grid[fr0, fc1] = CORNER_COLOR
    grid[fr1, fc0] = CORNER_COLOR
    grid[fr1, fc1] = CORNER_COLOR
    grid[fr0, fc0 + 1 : fc1] = side_colors["top"]
    grid[fr1, fc0 + 1 : fc1] = side_colors["bottom"]
    grid[fr0 + 1 : fr1, fc0] = side_colors["left"]
    grid[fr0 + 1 : fr1, fc1] = side_colors["right"]


def _lane_overlap(a: BBox, b: BBox) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _make_grid() -> np.ndarray | None:
    gh = random.randint(*GRID_H_RANGE)
    gw = random.randint(*GRID_W_RANGE)
    grid = np.full((gh, gw), Color.BLACK, dtype=int)

    inner_h = random.randint(*INNER_H_RANGE)
    inner_w = random.randint(*INNER_W_RANGE)
    mt = random.randint(*FRAME_MARGIN_RANGE)
    ml = random.randint(*FRAME_MARGIN_RANGE)
    if mt + inner_h + 2 >= gh or ml + inner_w + 2 >= gw:
        return None
    fr0 = mt
    fc0 = ml
    fr1 = fr0 + inner_h + 1
    fc1 = fc0 + inner_w + 1
    if fr1 >= gh - 1 or fc1 >= gw - 1:
        return None

    frame_bbox = (fr0, fc0, fr1, fc1)
    inner_bbox = (fr0 + 1, fc0 + 1, fr1 - 1, fc1 - 1)

    perm = random.sample(FRAME_COLORS, 4)
    side_colors = {
        "top": perm[0],
        "right": perm[1],
        "bottom": perm[2],
        "left": perm[3],
    }
    floor_color = random.choice(FLOOR_COLORS)

    _draw_frame(grid, frame_bbox, side_colors)
    ir0, ic0, ir1, ic1 = inner_bbox
    grid[ir0 : ir1 + 1, ic0 : ic1 + 1] = floor_color

    builder = random.choice((_build_vertical_case, _build_horizontal_case))
    lanes = builder(grid, inner_bbox, side_colors)
    if lanes is None:
        return None

    full_lanes = []
    for lane in lanes:
        lane_rect = _lane_rect(lane, inner_bbox)
        if any(_lane_overlap(lane_rect, prev) for prev in full_lanes):
            return None
        full_lanes.append(lane_rect)

    for lane in lanes:
        r0, c0, r1, c1 = lane["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = Color.BLACK
        gate = lane["gate"]
        if gate is None:
            continue
        gr, gc = gate["center"]
        if gate["orientation"] == "horizontal":
            grid[gr, gc - 1] = gate["end_color"]
            grid[gr, gc] = CORNER_COLOR
            grid[gr, gc + 1] = gate["end_color"]
        else:
            grid[gr - 1, gc] = gate["end_color"]
            grid[gr, gc] = CORNER_COLOR
            grid[gr + 1, gc] = gate["end_color"]
    return grid


def _validate_generated_case(inp: np.ndarray) -> None:
    scene = _parse_scene(inp)
    out = main(inp)
    blocks = scene["blocks"]
    sides = {block["side"] for block in blocks}
    assert len(sides) >= 2, "generator must expose at least two touched sides"

    gated = 0
    ungated = 0
    lane_rects = []
    final_rects = []
    for block in blocks:
        gate = _gate_for_block(block, scene)
        if gate is None:
            ungated += 1
        else:
            gated += 1
            gr, gc = gate["center"]
            if gate["orientation"] == "horizontal":
                expected = {
                    scene["side_colors"]["top"],
                    scene["side_colors"]["bottom"],
                }
                assert gate["end_color"] in expected
                near = _nearer_vertical_color(gr, scene["inner_bbox"], scene["side_colors"])
                if (gr - scene["inner_bbox"][0]) != (scene["inner_bbox"][2] - gr):
                    assert gate["end_color"] == near
            else:
                expected = {
                    scene["side_colors"]["left"],
                    scene["side_colors"]["right"],
                }
                assert gate["end_color"] in expected
                near = _nearer_horizontal_color(gc, scene["inner_bbox"], scene["side_colors"])
                if (gc - scene["inner_bbox"][1]) != (scene["inner_bbox"][3] - gc):
                    assert gate["end_color"] == near

        lane_rect = _lane_rect(block, scene["inner_bbox"])
        assert all(not _lane_overlap(lane_rect, other) for other in lane_rects)
        lane_rects.append(lane_rect)

        final_bb = _final_bbox(block, scene)
        final_rects.append(final_bb)
        dr, dc, steps = _block_motion(block, scene)
        assert steps > 0
        trail_color = scene["side_colors"][block["side"]]
        r0, c0, r1, c1 = block["bbox"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        trail_mask = np.zeros_like(out, dtype=bool)
        for step in range(steps):
            rr0 = r0 + dr * step
            cc0 = c0 + dc * step
            trail_mask[rr0 : rr0 + h, cc0 : cc0 + w] = True
        nr0, nc0, nr1, nc1 = final_bb
        trail_mask[nr0 : nr1 + 1, nc0 : nc1 + 1] = False
        assert np.all(out[trail_mask] == trail_color)
        assert np.all(out[nr0 : nr1 + 1, nc0 : nc1 + 1] == TRAIL_FINAL_COLOR)

    assert gated >= 1 and ungated >= 1, "generator must cover gate/no-gate branches"
    assert len({bb for bb in final_rects}) == len(final_rects), "final blocks must stay distinct"

    fr0, fc0, fr1, fc1 = scene["frame_bbox"]
    assert np.array_equal(out[fr0, fc0 : fc1 + 1], inp[fr0, fc0 : fc1 + 1])
    assert np.array_equal(out[fr1, fc0 : fc1 + 1], inp[fr1, fc0 : fc1 + 1])
    assert np.array_equal(out[fr0 : fr1 + 1, fc0], inp[fr0 : fr1 + 1, fc0])
    assert np.array_equal(out[fr0 : fr1 + 1, fc1], inp[fr0 : fr1 + 1, fc1])
    for gate in scene["gates"]:
        for r, c in gate["cells"]:
            assert out[r, c] == inp[r, c], "gates must remain unchanged"


def generate_input() -> np.ndarray:
    for _ in range(PLACEMENT_RETRIES):
        grid = _make_grid()
        if grid is None:
            continue
        _validate_generated_case(grid)
        return grid
    raise RuntimeError("failed to generate a valid grid")
