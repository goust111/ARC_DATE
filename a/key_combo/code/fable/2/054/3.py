from __future__ import annotations

import math
import random
from collections import Counter
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np

_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")

try:
    _ARC_UTILS = SourceFileLoader(
        "_arc_utils_fable_2_054_3", str(_ARC_UTILS_PATH)
    ).load_module()
    get_bg = _ARC_UTILS.get_bg
    bbox = _ARC_UTILS.bbox
    components_mask = _ARC_UTILS.components_mask
except Exception:
    def get_bg(grid: np.ndarray) -> int:
        return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


    def bbox(mask: np.ndarray):
        coords = np.argwhere(mask)
        if coords.size == 0:
            return None
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)


    def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
        h, w = mask.shape
        seen = np.zeros_like(mask, dtype=bool)
        if int(connectivity) == 4:
            steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            steps = [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1),
            ]
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
                    for dr, dc in steps:
                        nr = rr + dr
                        nc = cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                comps.append(np.array(cells, dtype=int))
        return comps


BODY_OFFSETS = {
    "right": [(-1, 0), (0, -1), (0, 0), (1, 0)],
    "left": [(-1, 0), (0, 0), (0, 1), (1, 0)],
    "up": [(0, -1), (0, 0), (0, 1), (1, 0)],
    "down": [(0, -1), (0, 0), (0, 1), (-1, 0)],
}
TIP_OFFSET = {
    "right": (0, 1),
    "left": (0, -1),
    "up": (-1, 0),
    "down": (1, 0),
}
ALL_COLORS = tuple(range(1, 10))
STEP_OPTIONS = (4, 5)
GEN_ATTEMPTS = 400


def _direction_from_tip(body_box, tip: tuple[int, int]) -> str:
    br0, bc0, br1, bc1 = body_box
    tr, tc = tip
    if tc < bc0:
        return "left"
    if tc > bc1:
        return "right"
    if tr < br0:
        return "up"
    if tr > br1:
        return "down"
    raise AssertionError("tip must lie outside the body bbox along one cardinal axis")


def _stamp_arrow(
    grid: np.ndarray,
    anchor: tuple[int, int],
    direction: str,
    body_color: int,
    tip_color: int,
) -> None:
    ar, ac = map(int, anchor)
    for dr, dc in BODY_OFFSETS[direction]:
        grid[ar + dr, ac + dc] = int(body_color)
    tr, tc = TIP_OFFSET[direction]
    grid[ar + tr, ac + tc] = int(tip_color)


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))

    comps = components_mask(g != bg, connectivity=4)
    arrow_parts = [coords.astype(int) for coords in comps if int(coords.shape[0]) > 1]
    dot_parts = [tuple(map(int, coords[0])) for coords in comps if int(coords.shape[0]) == 1]

    assert len(arrow_parts) == 1, "expected exactly one arrow sprite"
    assert len(dot_parts) >= 4, "expected a non-trivial track of dots"

    arrow_coords = arrow_parts[0]
    arrow_values = g[arrow_coords[:, 0], arrow_coords[:, 1]].astype(int)
    colors, counts = np.unique(arrow_values, return_counts=True)
    assert colors.size == 2, "arrow must have exactly one body color and one tip color"
    tip_colors = colors[counts == 1]
    assert tip_colors.size == 1, "arrow tip color must appear exactly once"
    tip_color = int(tip_colors[0])
    body_color = int(colors[colors != tip_color][0])
    tip_coord = tuple(map(int, arrow_coords[arrow_values == tip_color][0]))

    body_coords = arrow_coords[arrow_values == body_color]
    body_mask = np.zeros_like(g, dtype=bool)
    body_mask[body_coords[:, 0], body_coords[:, 1]] = True
    body_box = bbox(body_mask)
    assert body_box is not None
    direction = _direction_from_tip(body_box, tip_coord)
    horizontal = direction in {"left", "right"}

    arrow_mask = np.zeros_like(g, dtype=bool)
    arrow_mask[arrow_coords[:, 0], arrow_coords[:, 1]] = True
    arrow_box = bbox(arrow_mask)
    assert arrow_box is not None
    r0, c0, r1, c1 = arrow_box
    anchor = ((r0 + r1) // 2, (c0 + c1) // 2)
    assert int(g[anchor]) == body_color, "track slot under the sprite must be a body pixel"

    if horizontal:
        assert all(r == anchor[0] for r, _c in dot_parts), "all dots must lie on one row"
        anchor_axis = anchor[1]
    else:
        assert all(c == anchor[1] for _r, c in dot_parts), "all dots must lie on one column"
        anchor_axis = anchor[0]

    axis_coords = sorted(
        [c if horizontal else r for r, c in dot_parts] + [anchor_axis]
    )
    diffs = [b - a for a, b in zip(axis_coords, axis_coords[1:])]
    assert diffs and all(d > 0 for d in diffs), "track coordinates must be distinct"
    step = diffs[0]
    for d in diffs[1:]:
        step = math.gcd(step, d)
    assert step >= 2, "track spacing must leave the arrow visually separated from other dots"

    slot_coords: dict[int, tuple[int, int]] = {}
    dot_map: dict[int, int] = {}
    for r, c in dot_parts:
        axis = c if horizontal else r
        offset = axis - anchor_axis
        assert offset % step == 0 and offset != 0, "dots must occupy track slots"
        k = int(offset // step)
        slot_coords[k] = (r, c)
        dot_map[k] = int(g[r, c])

    assert len(slot_coords) == len(dot_parts), "track dots must occupy unique slots"
    min_k = min(slot_coords)
    max_k = max(slot_coords)
    expected_slots = set(range(min_k, max_k + 1)) - {0}
    assert set(slot_coords) == expected_slots, "track must be a continuous run of slots"

    forward = 1 if direction in {"right", "down"} else -1
    crossed: list[int] = []
    target_k = None
    k = forward
    while k in dot_map:
        if dot_map[k] == tip_color:
            target_k = k
            break
        crossed.append(k)
        k += forward
    assert target_k is not None, "need a forward target dot matching the tip color"

    return {
        "grid": g,
        "bg": bg,
        "arrow_coords": arrow_coords,
        "arrow_values": arrow_values,
        "body_color": body_color,
        "tip_color": tip_color,
        "tip_coord": tip_coord,
        "direction": direction,
        "horizontal": horizontal,
        "anchor": anchor,
        "step": int(step),
        "slot_coords": slot_coords,
        "dot_map": dot_map,
        "forward": forward,
        "target_k": int(target_k),
        "crossed": crossed,
    }


def _apply_scene(scene: dict, target_k: int | None = None, recolor: bool = True) -> np.ndarray:
    target_k = int(scene["target_k"] if target_k is None else target_k)
    assert target_k != 0 and target_k in scene["slot_coords"]

    out = scene["grid"].copy()
    arrow_coords = scene["arrow_coords"]
    out[arrow_coords[:, 0], arrow_coords[:, 1]] = int(scene["bg"])

    move_sign = 1 if target_k > 0 else -1
    if recolor:
        k = move_sign
        while k != target_k:
            rr, cc = scene["slot_coords"][k]
            if int(out[rr, cc]) != int(scene["tip_color"]):
                out[rr, cc] = int(scene["body_color"])
            k += move_sign

    shift = int(target_k) * int(scene["step"])
    dr, dc = (0, shift) if scene["horizontal"] else (shift, 0)
    moved = arrow_coords + np.array([dr, dc], dtype=int)
    out[moved[:, 0], moved[:, 1]] = scene["arrow_values"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the arrow sprite to the nearest forward dot matching its tip color and recolor the crossed dots.

    Input structure:
    - A black background contains one straight horizontal or vertical track of equally spaced single-pixel dots.
    - One track slot is occupied by a connected arrow sprite whose body uses one color and whose unique tip pixel uses another color.
    - The track has dots on both sides of the sprite, with a closer same-color decoy behind it and closer wrong-color dots ahead of it.

    Rule:
    - Read the motion direction from the tip position relative to the body.
    - Move the whole sprite rigidly along the track to the nearest dot in that forward direction whose color matches the tip color, consuming that dot.
    - Recolor every crossed forward dot with a different color to the sprite body color, clear the original sprite position, and leave all other dots unchanged.

    Key insight:
    - The single tip pixel double-encodes the answer: its geometry chooses the direction, and its color chooses which forward dot is the true landing slot.
    """
    return _apply_scene(_parse_scene(grid))


def _build_instance() -> tuple[np.ndarray, np.ndarray, dict]:
    direction = random.choice(tuple(BODY_OFFSETS))
    sign = 1 if direction in {"right", "down"} else -1
    step = random.choice(STEP_OPTIONS)
    slot_count = random.choice((5, 6, 7) if step == 4 else (5, 6))
    target_dist = random.randint(2, min(3, slot_count - 2))

    if sign == 1:
        arrow_idx = random.randint(1, slot_count - 1 - target_dist)
    else:
        arrow_idx = random.randint(target_dist, slot_count - 2)
    target_idx = arrow_idx + sign * target_dist

    back_limit = min(target_dist - 1, arrow_idx if sign == 1 else slot_count - 1 - arrow_idx)
    back_dist = random.randint(1, back_limit)
    back_idx = arrow_idx - sign * back_dist

    body_color, tip_color, path_color, extra_color = random.sample(ALL_COLORS, 4)
    palette = [body_color, tip_color, path_color, extra_color]
    slot_colors = [random.choice(palette) for _ in range(slot_count)]
    slot_colors[arrow_idx] = None
    slot_colors[target_idx] = tip_color
    slot_colors[back_idx] = tip_color

    crossed_indices = [arrow_idx + sign * d for d in range(1, target_dist)]
    slot_colors[crossed_indices[0]] = path_color
    for idx in crossed_indices[1:]:
        slot_colors[idx] = random.choice([body_color, path_color, extra_color])

    margin_lo = 1
    margin_hi = 2
    if direction in {"left", "right"}:
        left = random.randint(margin_lo, margin_hi)
        right = random.randint(margin_lo, margin_hi)
        width = left + (slot_count - 1) * step + right + 1
        height = random.randint(7, 11)
        track_r = random.randint(1, height - 2)
        slots = [(track_r, left + i * step) for i in range(slot_count)]
    else:
        top = random.randint(margin_lo, margin_hi)
        bottom = random.randint(margin_lo, margin_hi)
        height = top + (slot_count - 1) * step + bottom + 1
        width = random.randint(7, 11)
        track_c = random.randint(1, width - 2)
        slots = [(top + i * step, track_c) for i in range(slot_count)]

    grid = np.zeros((height, width), dtype=int)
    for idx, (r, c) in enumerate(slots):
        if idx == arrow_idx:
            continue
        grid[r, c] = int(slot_colors[idx])

    _stamp_arrow(grid, slots[arrow_idx], direction, body_color, tip_color)

    expected = np.zeros_like(grid)
    expected[:] = grid
    ar, ac = slots[arrow_idx]
    for dr, dc in BODY_OFFSETS[direction]:
        expected[ar + dr, ac + dc] = 0
    tr, tc = TIP_OFFSET[direction]
    expected[ar + tr, ac + tc] = 0
    for idx in crossed_indices:
        rr, cc = slots[idx]
        expected[rr, cc] = int(body_color)
    _stamp_arrow(expected, slots[target_idx], direction, body_color, tip_color)

    latent = {
        "direction": direction,
        "step": step,
        "slots": slots,
        "slot_colors": slot_colors,
        "arrow_idx": arrow_idx,
        "target_idx": target_idx,
        "crossed_indices": crossed_indices,
        "body_color": body_color,
        "tip_color": tip_color,
    }
    return grid, expected, latent


def _validate_instance(grid: np.ndarray, expected: np.ndarray, latent: dict) -> bool:
    scene = _parse_scene(grid)
    anchor = latent["slots"][latent["arrow_idx"]]
    target_k = latent["target_idx"] - latent["arrow_idx"]

    assert scene["direction"] == latent["direction"]
    assert scene["anchor"] == anchor
    assert scene["step"] == latent["step"]
    assert scene["body_color"] == latent["body_color"]
    assert scene["tip_color"] == latent["tip_color"]
    assert scene["target_k"] == target_k
    assert abs(scene["target_k"]) >= 2
    assert len(scene["crossed"]) == abs(scene["target_k"]) - 1
    assert any(
        scene["dot_map"][k] not in {scene["body_color"], scene["tip_color"]}
        for k in scene["crossed"]
    ), "need a visibly recolored intermediate dot"
    assert any(
        k * scene["forward"] < 0
        and scene["dot_map"][k] == scene["tip_color"]
        and abs(k) < abs(scene["target_k"])
        for k in scene["dot_map"]
    ), "need a closer same-color decoy behind the sprite"
    assert scene["dot_map"][scene["forward"]] != scene["tip_color"], "nearest forward dot must be the wrong color"

    nearest_any_forward = scene["forward"]
    nearest_same_any = min(
        (abs(k), k) for k, color in scene["dot_map"].items() if color == scene["tip_color"]
    )[1]
    assert nearest_any_forward != scene["target_k"]
    assert nearest_same_any != scene["target_k"]

    wrong_forward = _apply_scene(scene, target_k=nearest_any_forward, recolor=True)
    wrong_same = _apply_scene(scene, target_k=nearest_same_any, recolor=True)
    wrong_skip_trail = _apply_scene(scene, target_k=scene["target_k"], recolor=False)

    assert not np.array_equal(wrong_forward, expected)
    assert not np.array_equal(wrong_same, expected)
    assert not np.array_equal(wrong_skip_trail, expected)
    assert np.array_equal(main(grid), expected)
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid, expected, latent = _build_instance()
        if _validate_instance(grid, expected, latent):
            return grid
    raise RuntimeError(f"failed to generate a valid puzzle after {GEN_ATTEMPTS} attempts")
