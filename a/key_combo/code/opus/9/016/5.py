from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, find_components, get_bg, paste


CoordArray = np.ndarray

_BG = int(Color.BLACK)
_WALL = int(Color.RED)
_OBJ = int(Color.GREEN)
_H_MIRROR = int(Color.BLUE)
_V_MIRROR = int(Color.TEAL)

_GEN_ROWS = (18, 24)
_GEN_COLS = (18, 24)
_GEN_COUNTS = (2, 3, 4)
_GEN_COUNT_WEIGHTS = (2, 4, 3)
_GEN_SIDES = ("top", "bottom", "left", "right")
_GEN_TAB_SIDES = ("left", "right", "top", "bottom")
_GEN_RETRIES = 400
_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _find_wall(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int], np.ndarray]:
    candidates: list[tuple[str, int]] = []
    h, w = map(int, grid.shape)
    edges = {
        "top": grid[0, :],
        "bottom": grid[h - 1, :],
        "left": grid[:, 0],
        "right": grid[:, w - 1],
    }
    expected_counts = {"top": w, "bottom": w, "left": h, "right": h}
    for side, edge in edges.items():
        color = int(edge[0])
        if color == int(bg):
            continue
        if np.all(edge == color) and int(np.sum(grid == color)) == expected_counts[side]:
            candidates.append((side, color))
    assert len(candidates) == 1, "expected exactly one full edge wall"
    side, color = candidates[0]
    step = {
        "top": (-1, 0),
        "bottom": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }[side]
    return color, step, grid == color


def _settle_key(coords: CoordArray, step: tuple[int, int]) -> tuple[int, int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    if step == (-1, 0):
        primary = int(rows.min())
    elif step == (1, 0):
        primary = int(-rows.max())
    elif step == (0, -1):
        primary = int(cols.min())
    else:
        primary = int(-cols.max())
    return (primary, *map(int, bbox_coords(coords)))


def _coords_to_mask(coords: CoordArray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = map(int, bbox_coords(coords))
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    local = coords - np.array([r0, c0], dtype=int)
    mask[local[:, 0], local[:, 1]] = True
    return mask, (r0, c0, r1, c1)


def _tab_side(mask: np.ndarray) -> str:
    h, w = map(int, mask.shape)
    boundary_counts = {
        "top": int(mask[0, :].sum()),
        "bottom": int(mask[h - 1, :].sum()),
        "left": int(mask[:, 0].sum()),
        "right": int(mask[:, w - 1].sum()),
    }
    sides = [side for side, count in boundary_counts.items() if count == 1]
    assert len(sides) == 1, "expected exactly one one-pixel tab side"
    return sides[0]


def _mirror_spec(mask: np.ndarray, bbox: tuple[int, int, int, int], side: str) -> tuple[np.ndarray, int, int, int]:
    r0, c0, r1, c1 = map(int, bbox)
    h, w = map(int, mask.shape)
    if side in {"left", "right"}:
        reflected = np.fliplr(mask)
        left = c1 + 2 if side == "left" else c0 - w - 1
        return reflected, int(r0), int(left), _H_MIRROR
    reflected = np.flipud(mask)
    top = r1 + 2 if side == "top" else r0 - h - 1
    return reflected, int(top), int(c0), _V_MIRROR


def _paint_mask_background_only(
    canvas: np.ndarray,
    mask: np.ndarray,
    top: int,
    left: int,
    color: int,
    bg: int,
) -> None:
    stamp = np.full(mask.shape, int(bg), dtype=int)
    stamp[mask] = int(color)
    paste(canvas, stamp, int(top), int(left), transparent=int(bg), overwrite=False, clip=True)


def _slide_until_blocked(
    coords: CoordArray,
    step: tuple[int, int],
    occupied: np.ndarray,
    shape: tuple[int, int],
) -> CoordArray:
    moved = coords.copy()
    delta = np.array(step, dtype=int)
    h, w = map(int, shape)
    while True:
        nxt = moved + delta
        if (
            np.any(nxt[:, 0] < 0)
            or np.any(nxt[:, 0] >= h)
            or np.any(nxt[:, 1] < 0)
            or np.any(nxt[:, 1] >= w)
            or np.any(occupied[nxt[:, 0], nxt[:, 1]])
        ):
            return moved
        moved = nxt


def _settled_objects(grid: np.ndarray) -> tuple[int, int, tuple[int, int], np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    wall_color, step, wall_mask = _find_wall(g, bg)
    components = find_components(
        g,
        bg=bg,
        colors=[_OBJ],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert 2 <= len(components) <= 4, "expected 2 to 4 green components"

    settled: list[dict] = []
    occupied = wall_mask.copy()
    for comp in sorted(components, key=lambda item: _settle_key(item["coords"], step)):
        coords = comp["coords"].astype(int)
        final_coords = _slide_until_blocked(coords, step, occupied, g.shape)
        settled.append(
            {
                "input_coords": coords,
                "settled_coords": final_coords,
                "color": int(comp["color"]),
            }
        )
        occupied[final_coords[:, 0], final_coords[:, 1]] = True
    return bg, wall_color, step, wall_mask, settled


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the green tabbed outlines into the red edge wall, then mirror each settled object from its post-gravity position.

    Input structure:
    - The background is black.
    - One full red edge wall occupies exactly one whole grid side and appears nowhere else.
    - The remaining foreground consists of 2 to 4 green 4-connected one-pixel-thick outline objects, each with exactly one one-pixel tab sticking out from one bbox side.

    Rule:
    - The wall side determines gravity: every green object slides rigidly toward that wall.
    - Settle objects from nearest to farthest from the wall; each stops when one more step would hit the wall or an already settled object.
    - Erase the original positions and keep the settled green objects.
    - For each settled object, inspect the tab side on its settled bbox. A left/right tab means mirror the object horizontally and place the blue copy on the opposite bbox side with a one-cell gap; a top/bottom tab means mirror vertically and place the teal copy on the opposite bbox side with a one-cell gap.
    - Paint mirrored copies only onto black background cells; if multiple copies want the same cell, keep the copy belonging to the object nearer the wall.

    Key insight:
    - The tab chooses how to mirror, but the copy is anchored to the object's settled bbox after gravity, so solving the movement stage correctly is necessary before the mirror stage can be placed.
    """
    g = np.asarray(grid, dtype=int)
    bg, wall_color, _step, wall_mask, settled = _settled_objects(g)

    out = np.full_like(g, int(bg))
    out[wall_mask] = int(wall_color)
    for item in settled:
        coords = item["settled_coords"]
        out[coords[:, 0], coords[:, 1]] = _OBJ

    for item in settled:
        mask, bbox = _coords_to_mask(item["settled_coords"])
        side = _tab_side(mask)
        reflected, top, left, color = _mirror_spec(mask, bbox, side)
        _paint_mask_background_only(out, reflected, top, left, color, bg)

    return out


def _draw_outline(patch: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    bottom = int(top) + int(height) - 1
    right = int(left) + int(width) - 1
    patch[top, left : right + 1] = _OBJ
    patch[bottom, left : right + 1] = _OBJ
    patch[top : bottom + 1, left] = _OBJ
    patch[top : bottom + 1, right] = _OBJ


def _make_patch(side: str) -> np.ndarray:
    total_h = random.randint(3, 5)
    total_w = random.randint(3, 5)
    if side in {"left", "right"}:
        patch = np.full((total_h, total_w), _BG, dtype=int)
        frame_left = 1 if side == "left" else 0
        frame_w = total_w - 1
        _draw_outline(patch, 0, frame_left, total_h, frame_w)
        patch[total_h // 2, 0 if side == "left" else total_w - 1] = _OBJ
        return patch
    patch = np.full((total_h, total_w), _BG, dtype=int)
    frame_top = 1 if side == "top" else 0
    frame_h = total_h - 1
    _draw_outline(patch, frame_top, 0, frame_h, total_w)
    patch[0 if side == "top" else total_h - 1, total_w // 2] = _OBJ
    return patch


def _choose_object_tab_sides(count: int) -> list[str]:
    if count == 2:
        sides = [random.choice(("left", "right")), random.choice(("top", "bottom"))]
    elif count == 3:
        sides = [random.choice(("left", "right")), random.choice(("top", "bottom")), random.choice(_GEN_TAB_SIDES)]
    else:
        sides = list(_GEN_TAB_SIDES)
    random.shuffle(sides)
    return sides


def _copy_coords_for_position(
    mask: np.ndarray,
    r0: int,
    c0: int,
    side: str,
    shape: tuple[int, int],
) -> tuple[np.ndarray, int] | None:
    h, w = map(int, mask.shape)
    bbox = (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1))
    reflected, top, left, color = _mirror_spec(mask, bbox, side)
    max_h, max_w = map(int, shape)
    if top < 0 or left < 0 or top + h > max_h or left + w > max_w:
        return None
    local = np.argwhere(reflected)
    coords = local + np.array([top, left], dtype=int)
    return coords, color


def _settled_candidates(
    mask: np.ndarray,
    side: str,
    occupied_green: np.ndarray,
    reserved_copy: np.ndarray,
    require_supported: bool,
) -> list[dict]:
    h, w = map(int, occupied_green.shape)
    patch_h, patch_w = map(int, mask.shape)
    local = np.argwhere(mask)
    wall_mask = np.zeros_like(occupied_green, dtype=bool)
    wall_mask[0, :] = True
    occupied_all = wall_mask | occupied_green
    candidates: list[dict] = []
    for r0 in range(1, h - patch_h + 1):
        for c0 in range(0, w - patch_w + 1):
            coords = local + np.array([r0, c0], dtype=int)
            if np.any(occupied_green[coords[:, 0], coords[:, 1]]):
                continue
            if np.any(reserved_copy[coords[:, 0], coords[:, 1]]):
                continue
            prev = coords + np.array([-1, 0], dtype=int)
            if not np.any(occupied_all[prev[:, 0], prev[:, 1]]):
                continue
            supported = bool(np.any(occupied_green[prev[:, 0], prev[:, 1]]))
            if require_supported and not supported:
                continue
            copy_info = _copy_coords_for_position(mask, r0, c0, side, occupied_green.shape)
            if copy_info is None:
                continue
            copy_coords, copy_color = copy_info
            if np.any(occupied_all[copy_coords[:, 0], copy_coords[:, 1]]):
                continue
            if np.any(reserved_copy[copy_coords[:, 0], copy_coords[:, 1]]):
                continue
            candidates.append(
                {
                    "coords": coords,
                    "copy_coords": copy_coords,
                    "copy_color": int(copy_color),
                    "supported": supported,
                }
            )
    random.shuffle(candidates)
    return candidates


def _mark_with_gap(blocked: np.ndarray, coords: np.ndarray) -> None:
    h, w = map(int, blocked.shape)
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        blocked[rr, cc] = True
        for dr, dc in _N4:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < h and 0 <= nc < w:
                blocked[nr, nc] = True


def _slideback_assignments(settled: list[dict], shape: tuple[int, int]) -> list[dict] | None:
    h, w = map(int, shape)
    assigned: list[dict | None] = [None] * len(settled)

    def rec(idx: int, blocked: np.ndarray, prev_top: int, moved_far: bool) -> bool:
        if idx == len(settled):
            return moved_far
        final_coords = settled[idx]["settled_coords"]
        min_row = int(final_coords[:, 0].min())
        max_row = int(final_coords[:, 0].max())
        max_dist = h - 1 - max_row
        choices = list(range(1, max_dist + 1))
        random.shuffle(choices)
        for dist in choices:
            coords = final_coords + np.array([dist, 0], dtype=int)
            top_row = int(coords[:, 0].min())
            if top_row < 2 or top_row <= prev_top:
                continue
            if np.any(blocked[coords[:, 0], coords[:, 1]]):
                continue
            next_blocked = blocked.copy()
            _mark_with_gap(next_blocked, coords)
            assigned[idx] = {
                **settled[idx],
                "input_coords": coords,
            }
            if rec(idx + 1, next_blocked, top_row, moved_far or dist >= 2):
                return True
            assigned[idx] = None
        return False

    if not rec(0, np.zeros((h, w), dtype=bool), 1, False):
        return None
    return [item for item in assigned if item is not None]


def _render_scene(shape: tuple[int, int], objects: list[dict], include_copies: bool) -> np.ndarray:
    h, w = map(int, shape)
    grid = np.full((h, w), _BG, dtype=int)
    grid[0, :] = _WALL
    for item in objects:
        coords = item["settled_coords"] if include_copies else item["input_coords"]
        grid[coords[:, 0], coords[:, 1]] = _OBJ
    if include_copies:
        for item in objects:
            grid[item["copy_coords"][:, 0], item["copy_coords"][:, 1]] = int(item["copy_color"])
    return grid


def _transform_from_top(grid: np.ndarray, side: str) -> np.ndarray:
    if side == "top":
        return grid.copy()
    if side == "bottom":
        return np.flipud(grid).copy()
    if side == "left":
        return np.rot90(grid, 1).copy()
    if side == "right":
        return np.rot90(grid, -1).copy()
    raise ValueError(side)


def _validate_generated_pair(inp: np.ndarray, expected_out: np.ndarray) -> bool:
    try:
        solved = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(solved, expected_out):
        return False

    bg, _wall_color, step, wall_mask, settled = _settled_objects(inp)
    if int(bg) != _BG:
        return False

    def front_value(coords: np.ndarray) -> int:
        if step == (-1, 0):
            return int(coords[:, 0].min())
        if step == (1, 0):
            return int(coords[:, 0].max())
        if step == (0, -1):
            return int(coords[:, 1].min())
        return int(coords[:, 1].max())

    shift = np.array(step, dtype=int)
    moved = False
    supported = False
    occupied_green = np.zeros(inp.shape, dtype=bool)
    for item in settled:
        if not np.array_equal(item["input_coords"], item["settled_coords"]):
            moved = True
        mask, bbox = _coords_to_mask(item["settled_coords"])
        side = _tab_side(mask)
        reflected, top, left, _color = _mirror_spec(mask, bbox, side)
        h, w = map(int, reflected.shape)
        if top < 0 or left < 0 or top + h > inp.shape[0] or left + w > inp.shape[1]:
            return False
        front = item["settled_coords"] + shift
        if np.any(occupied_green[front[:, 0], front[:, 1]]):
            supported = True
        occupied_green[item["settled_coords"][:, 0], item["settled_coords"][:, 1]] = True

    if not moved or not supported:
        return False

    green_in = find_components(inp, bg=_BG, colors=[_OBJ], connectivity=4, monochromatic=True, return_format="dict")
    if len(green_in) != len(settled):
        return False
    if len({_tab_side(_coords_to_mask(comp["coords"])[0]) for comp in green_in}) < 2:
        return False
    if len({front_value(comp["coords"]) for comp in green_in}) != len(green_in):
        return False

    out = expected_out
    if np.any(out[wall_mask] != _WALL):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*_GEN_ROWS)
    cols = random.randint(*_GEN_COLS)
    count = random.choices(_GEN_COUNTS, weights=_GEN_COUNT_WEIGHTS, k=1)[0]
    object_sides = _choose_object_tab_sides(count)

    occupied_green = np.zeros((rows, cols), dtype=bool)
    reserved_copy = np.zeros((rows, cols), dtype=bool)
    settled: list[dict] = []
    need_supported = True

    for i, side in enumerate(object_sides):
        patch = _make_patch(side)
        mask = patch == _OBJ
        require_supported = bool(need_supported and i > 0)
        candidates = _settled_candidates(mask, side, occupied_green, reserved_copy, require_supported)
        if not candidates:
            return None
        choice = candidates[0]
        coords = choice["coords"]
        occupied_green[coords[:, 0], coords[:, 1]] = True
        reserved_copy[choice["copy_coords"][:, 0], choice["copy_coords"][:, 1]] = True
        settled.append(
            {
                "settled_coords": coords,
                "copy_coords": choice["copy_coords"],
                "copy_color": choice["copy_color"],
            }
        )
        need_supported = need_supported and not choice["supported"]

    if need_supported:
        return None

    assigned = _slideback_assignments(settled, (rows, cols))
    if assigned is None:
        return None

    inp_top = _render_scene((rows, cols), assigned, include_copies=False)
    out_top = _render_scene((rows, cols), assigned, include_copies=True)
    side = random.choice(_GEN_SIDES)
    inp = _transform_from_top(inp_top, side)
    out = _transform_from_top(out_top, side)
    if not _validate_generated_pair(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for ARC task 5")
