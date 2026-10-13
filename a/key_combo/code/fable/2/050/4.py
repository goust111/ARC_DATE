from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import find_components, get_bg, paste


BG = 0
MEMBRANE_COLOR = 5
LANE_W = 3
GEN_ATTEMPTS = 500
H_RANGE = (20, 26)
W_RANGE = (18, 26)
FLOOR_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
TOKEN_COLORS = tuple(c for c in range(10) if c not in (BG, MEMBRANE_COLOR))

SOLID = np.ones((3, 3), dtype=bool)
PLUS = np.array(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ],
    dtype=bool,
)
X_SHAPE = np.array(
    [
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1],
    ],
    dtype=bool,
)
SHAPE_MASKS = (SOLID, PLUS, X_SHAPE)
SHAPE_OFFSETS = tuple(tuple(map(tuple, np.argwhere(mask))) for mask in SHAPE_MASKS)


def _shape_patch(shape_id: int, color: int, bg: int) -> np.ndarray:
    patch = np.full((3, 3), bg, dtype=int)
    patch[SHAPE_MASKS[int(shape_id)]] = int(color)
    return patch


def _detect_shape(patch: np.ndarray, color: int, bg: int) -> int | None:
    if not np.all((patch == bg) | (patch == color)):
        return None
    mask = patch == color
    for shape_id, template in enumerate(SHAPE_MASKS):
        if np.array_equal(mask, template):
            return shape_id
    return None


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(int(a0), int(b0)) <= min(int(a1), int(b1))


def _find_floor(grid: np.ndarray, bg: int) -> tuple[int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    floor_color = int(g[h - 1, 0])
    assert floor_color != bg, "bottom row must be a non-background floor"
    assert np.all(g[h - 1, :] == floor_color), "bottom row must be a solid floor"
    top = h - 1
    while top - 1 >= 0 and np.all(g[top - 1, :] == floor_color):
        top -= 1
    return top, floor_color


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    h, w = g.shape
    bg = int(get_bg(g))
    floor_top, floor_color = _find_floor(g, bg)

    components = find_components(
        g, bg=bg, connectivity=8, monochromatic=True, return_format="dict"
    )
    membranes: list[dict] = []
    tokens: list[dict] = []
    for comp in components:
        r0, c0, r1, c1 = comp["bbox"]
        color = int(comp["color"])
        coords = np.asarray(comp["coords"], dtype=int)

        if (
            color == floor_color
            and r0 == floor_top
            and c0 == 0
            and r1 == h - 1
            and c1 == w - 1
            and comp["area"] == (h - floor_top) * w
        ):
            continue

        if r0 == r1 and r0 < floor_top and comp["area"] == c1 - c0 + 1:
            membranes.append(
                {
                    "row": int(r0),
                    "c0": int(c0),
                    "c1": int(c1),
                    "color": color,
                    "coords": coords,
                }
            )
            continue

        if (r1 - r0 + 1, c1 - c0 + 1) == (3, 3):
            shape_id = _detect_shape(g[r0 : r1 + 1, c0 : c1 + 1], color, bg)
            if shape_id is not None:
                tokens.append(
                    {
                        "top": int(r0),
                        "left": int(c0),
                        "shape": int(shape_id),
                        "color": color,
                    }
                )
                continue

        raise AssertionError(
            f"unclassified foreground component at bbox {(r0, c0, r1, c1)} with color {color}"
        )

    assert membranes, "expected at least one membrane line"
    assert tokens, "expected at least one token"
    membrane_colors = {mem["color"] for mem in membranes}
    assert len(membrane_colors) == 1, "all membranes must share one visible color"

    tokens.sort(key=lambda token: (token["top"], token["left"]), reverse=True)
    membranes.sort(key=lambda mem: (mem["row"], mem["c0"]))
    return {
        "grid": g,
        "bg": bg,
        "h": h,
        "w": w,
        "floor_top": floor_top,
        "floor_color": floor_color,
        "membranes": membranes,
        "tokens": tokens,
    }


def _collides(top: int, left: int, shape_id: int, occupied: set[tuple[int, int]], h: int, w: int) -> bool:
    for dr, dc in SHAPE_OFFSETS[int(shape_id)]:
        rr = int(top) + int(dr)
        cc = int(left) + int(dc)
        if not (0 <= rr < h and 0 <= cc < w):
            return True
        if (rr, cc) in occupied:
            return True
    return False


def _drop_tokens(
    tokens: list[dict], membranes: list[dict], h: int, w: int, floor_top: int
) -> list[dict]:
    occupied = {(r, c) for r in range(floor_top, h) for c in range(w)}
    membranes_by_row: defaultdict[int, list[dict]] = defaultdict(list)
    for membrane in membranes:
        membranes_by_row[int(membrane["row"])].append(membrane)

    landed: list[dict] = []
    for token in sorted(tokens, key=lambda item: (item["top"], item["left"]), reverse=True):
        top = int(token["top"])
        left = int(token["left"])
        shape = int(token["shape"])
        crossed_rows: list[int] = []

        while True:
            next_top = top + 1
            next_shape = shape
            step_crossings: list[int] = []
            for membrane in membranes_by_row.get(top, ()):
                if _intervals_overlap(left, left + 2, membrane["c0"], membrane["c1"]):
                    next_shape = (next_shape + 1) % len(SHAPE_MASKS)
                    step_crossings.append(int(membrane["row"]))

            if _collides(next_top, left, next_shape, occupied, h, w):
                break

            top = next_top
            shape = next_shape
            crossed_rows.extend(step_crossings)

        cells = tuple((top + dr, left + dc) for dr, dc in SHAPE_OFFSETS[shape])
        for cell in cells:
            occupied.add(cell)
        landed.append(
            {
                **token,
                "final_top": int(top),
                "final_shape": int(shape),
                "crossings": len(crossed_rows),
                "crossed_rows": tuple(crossed_rows),
                "cells": cells,
            }
        )
    return landed


def _render_scene(scene: dict, landed: list[dict]) -> np.ndarray:
    g = scene["grid"]
    bg = int(scene["bg"])
    h, w = g.shape
    out = np.full((h, w), bg, dtype=int)

    out[scene["floor_top"] :, :] = scene["floor_color"]
    for membrane in scene["membranes"]:
        rr = membrane["coords"][:, 0]
        cc = membrane["coords"][:, 1]
        out[rr, cc] = membrane["color"]

    for token in landed:
        paste(
            out,
            _shape_patch(token["final_shape"], token["color"], bg),
            token["final_top"],
            token["left"],
            transparent=bg,
        )

    # Floor and membranes are visual overlays, not obstacles.
    out[scene["floor_top"] :, :] = g[scene["floor_top"] :, :]
    for membrane in scene["membranes"]:
        rr = membrane["coords"][:, 0]
        cc = membrane["coords"][:, 1]
        out[rr, cc] = membrane["color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Drop 3x3 tokens through horizontal membranes, cycling each token's shape by the number of membranes it fully passes before stacking.

    Input structure:
    - Black background with a solid floor along the bottom edge.
    - Several floating monochrome 3x3 tokens, each using one state of the cycle solid square -> plus -> X.
    - Several horizontal gray line segments above the floor; lines may cover only some columns and are not obstacles.

    Rule:
    - Lower tokens fall first, then higher tokens treat the landed tokens as new floor while still falling straight down.
    - When a token successfully steps downward from top row r to r+1, every overlapping membrane on row r counts as one completed crossing, and the token advances one step along the fixed shape cycle for each such crossing.
    - After all tokens land, clear their original floating positions, keep the floor and membranes unchanged, and draw the landed tokens in their final cycled shapes with their original colors.

    Key insight:
    - Membranes only count actual path crossings, so earlier landed tokens can raise later ones enough to cancel lower membrane crossings; stacking order and shape cycling must be simulated together.
    """
    scene = _parse_scene(grid)
    landed = _drop_tokens(
        scene["tokens"], scene["membranes"], scene["h"], scene["w"], scene["floor_top"]
    )
    return _render_scene(scene, landed)


def _build_membrane(row: int, c0: int, c1: int) -> dict:
    coords = np.array([(row, c) for c in range(c0, c1 + 1)], dtype=int)
    return {
        "row": int(row),
        "c0": int(c0),
        "c1": int(c1),
        "color": MEMBRANE_COLOR,
        "coords": coords,
    }


def _lane_starts(width: int) -> list[int] | None:
    gap1 = random.randint(2, 4)
    gap2 = random.randint(2, 4)
    span = 3 * LANE_W + gap1 + gap2
    if span + 2 > width:
        return None
    left_margin = random.randint(1, width - span - 1)
    return [
        left_margin,
        left_margin + LANE_W + gap1,
        left_margin + 2 * LANE_W + gap1 + gap2,
    ]


def _core_rows(height: int) -> tuple[int, int, int] | None:
    top_row = random.randint(7, 8)
    low_row = height - random.choice((6, 7))
    if low_row <= top_row + 5:
        return None
    mid_row = random.randint(top_row + 3, low_row - 3)
    return top_row, mid_row, low_row


def _make_scene_spec() -> dict | None:
    h = random.randint(*H_RANGE)
    w = random.randint(*W_RANGE)
    lanes = _lane_starts(w)
    rows = _core_rows(h)
    if lanes is None or rows is None:
        return None

    top_row, mid_row, low_row = rows
    stack_upper_top = top_row - 7
    stack_lower_top = top_row - 3
    if stack_upper_top < 0:
        return None

    quiet_count = random.choice((1, 2))
    total_tokens = 4 + (quiet_count - 1)
    floor_color = random.choice(FLOOR_COLORS)
    token_palette = [c for c in TOKEN_COLORS if c != floor_color]
    colors = random.sample(token_palette, total_tokens)

    shapes = [0, 1, 2]
    while len(shapes) < total_tokens:
        shapes.append(random.randrange(3))
    random.shuffle(shapes)

    high_quiet_top = random.randint(0, max(0, top_row - 3))
    low_quiet_top = random.randint(mid_row + 1, min(h - 4, low_row + 1))
    quiet_tops = [high_quiet_top] if quiet_count == 1 and random.random() < 0.5 else [low_quiet_top]
    if quiet_count == 2:
        quiet_tops = [high_quiet_top, low_quiet_top]

    once_top = random.randint(0, max(0, top_row - 3))
    roles_and_positions: list[tuple[str, int, int]] = []
    for idx, top in enumerate(quiet_tops):
        roles_and_positions.append((f"quiet_{idx}", int(top), lanes[0]))
    roles_and_positions.append(("once", int(once_top), lanes[1]))
    roles_and_positions.append(("stack_upper", int(stack_upper_top), lanes[2]))
    roles_and_positions.append(("stack_lower", int(stack_lower_top), lanes[2]))
    random.shuffle(roles_and_positions)

    tokens = []
    for color, shape, (role, top, left) in zip(colors, shapes, roles_and_positions):
        tokens.append(
            {
                "role": role,
                "top": int(top),
                "left": int(left),
                "shape": int(shape),
                "color": int(color),
            }
        )

    membranes = [
        _build_membrane(top_row, lanes[1], lanes[2] + 2),
        _build_membrane(mid_row, lanes[2], lanes[2] + 2),
        _build_membrane(low_row, lanes[2], lanes[2] + 2),
    ]
    return {
        "h": h,
        "w": w,
        "floor_color": floor_color,
        "lanes": lanes,
        "rows": rows,
        "membranes": membranes,
        "tokens": tokens,
    }


def _render_input_from_spec(spec: dict) -> np.ndarray:
    grid = np.full((spec["h"], spec["w"]), BG, dtype=int)
    grid[-1, :] = spec["floor_color"]
    for membrane in spec["membranes"]:
        grid[membrane["row"], membrane["c0"] : membrane["c1"] + 1] = membrane["color"]
    for token in spec["tokens"]:
        paste(
            grid,
            _shape_patch(token["shape"], token["color"], BG),
            token["top"],
            token["left"],
            transparent=BG,
        )
    return grid


def _semantic_checks(spec: dict, grid: np.ndarray, landed: list[dict], out: np.ndarray) -> bool:
    landed_by_role = {token["role"]: token for token in landed}
    solo_by_role = {
        token["role"]: _drop_tokens([token], spec["membranes"], spec["h"], spec["w"], spec["h"] - 1)[0]
        for token in spec["tokens"]
    }

    mods = {token["crossings"] % 3 for token in landed}
    if mods != {0, 1, 2}:
        return False

    if landed_by_role["once"]["crossed_rows"] != (spec["rows"][0],):
        return False
    if set(landed_by_role["stack_upper"]["crossed_rows"]) != {spec["rows"][0], spec["rows"][1]}:
        return False
    if set(landed_by_role["stack_lower"]["crossed_rows"]) != set(spec["rows"]):
        return False

    for role, token in landed_by_role.items():
        if role.startswith("quiet_") and token["crossings"] != 0:
            return False

    if not (landed_by_role["stack_upper"]["crossings"] < solo_by_role["stack_upper"]["crossings"]):
        return False
    if not (landed_by_role["stack_upper"]["final_top"] < solo_by_role["stack_upper"]["final_top"]):
        return False
    if not any(token["final_shape"] != token["shape"] for token in landed):
        return False

    if len({(token["final_top"], token["final_shape"]) for token in landed}) < 3:
        return False

    for token in landed:
        visible = int(np.count_nonzero(out == token["color"]))
        if visible < 3:
            return False

    if np.array_equal(grid, out):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    spec = _make_scene_spec()
    if spec is None:
        return None

    grid = _render_input_from_spec(spec)
    scene = {
        "grid": grid,
        "bg": BG,
        "h": spec["h"],
        "w": spec["w"],
        "floor_top": spec["h"] - 1,
        "floor_color": spec["floor_color"],
        "membranes": spec["membranes"],
        "tokens": sorted(spec["tokens"], key=lambda item: (item["top"], item["left"]), reverse=True),
    }
    landed = _drop_tokens(scene["tokens"], scene["membranes"], scene["h"], scene["w"], scene["floor_top"])
    out = _render_scene(scene, landed)

    if not np.array_equal(main(grid), out):
        return None
    if not _semantic_checks(spec, grid, landed, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_ATTEMPTS} attempts")
