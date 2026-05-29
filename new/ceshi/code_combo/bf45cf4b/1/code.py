# Combo variant of bf45cf4b (index 1)
# combo_spec: D1/1+D6/1
# Name: 全位置旋转瓦片铺贴
#
# Design analysis
# - Original task: fill cells in a monochrome control sprite stamp the same square tile;
#   hole cells expand to plain background.
# - New combo: every control-sprite position receives a tile. Filled positions use a
#   scan-order rotation cycle, while hole positions always use the 180-degree tile.
# - Concretely, scanning the control bbox row-major:
#   Example mask [[1, 0, 1], [1, 1, 0]]
#   Original output tiles: [T, bg, T, T, T, bg]
#   New output tiles:      [T0, T180, T90, T180, T270, T180]
#   where the filled-position counter ignores holes, so the four filled cells rotate
#   as 0/90/180/270 even though holes are interleaved.
# - This creates two simultaneous clues for a human:
#   1. Holes are no longer empty; they systematically show the same rotated tile.
#   2. Filled cells are no longer identical; later copies change in a repeating cycle.
# - Complexity calibration:
#   1. The fill counter ignores holes, so a hole between two filled cells visibly shifts
#      the next filled tile relative to the naive "all positions cycle" hypothesis.
#   2. The generator requires several filled cells after the first hole, so both rules
#      co-occur inside a single example instead of appearing as isolated edge cases.
#   3. The tile is forced to have four distinct quarter-turns; otherwise 90/180-degree
#      rotations could collapse visually and the fusion would look like the base task.
# - Hidden assumptions that changed in the solver:
#   1. "hole -> background block" is no longer valid.
#   2. "all filled cells use the same tile orientation" is no longer valid.
#   3. The tile must be rotation-visible; otherwise the fusion degenerates. The
#      generator therefore forces an asymmetric square tile with four distinct
#      quarter-turns and rejects samples where the combo output matches simpler
#      ablations (plain stamping, holes-only rotation, or counting holes in the cycle).
# - Line-by-line assumption audit:
#   1. Object identification still relies on exactly two disconnected foreground
#      components, with the tile being the only multicolor square object.
#   2. `_compose_combo_output` assumes the control sprite has exactly one fg color and
#      that row-major traversal is the only ordering signal available to the observer.
#   3. The generator guarantees these assumptions by using one monochrome sprite, one
#      multicolor square tile, a background gap of at least one cell, and post-checks
#      that reject degenerate outputs.
# - Placement legality is unchanged: the input still contains exactly two disconnected
#   foreground components, one monochrome control sprite and one multicolor square tile.
# - Visual cleanliness: the two source objects remain spatially separated in the input,
#   while the output is a regular tile grid with no overlaps, so all variation is carried
#   by tile orientation rather than by cluttered object interactions.

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import colors_present, crop_bbox, find_components, get_bg, paste
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[4] / "solvers"))
    from arc_utils import colors_present, crop_bbox, find_components, get_bg, paste


def _identify_objects(grid: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
    comps = find_components(grid, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    assert len(comps) == 2

    infos = []
    for comp in comps:
        patch = crop_bbox(grid, comp["bbox"])
        cols = colors_present(patch, exclude={bg})
        infos.append(
            {
                "patch": patch,
                "cols": cols,
                "is_square": int(patch.shape[0]) == int(patch.shape[1]),
            }
        )

    infos.sort(
        key=lambda d: (
            len(d["cols"]) >= 2,
            len(d["cols"]),
            d["is_square"],
        ),
        reverse=True,
    )
    block = infos[0]["patch"]
    ctrl = infos[1]["patch"]
    assert block.shape[0] == block.shape[1]
    assert len(colors_present(block, exclude={bg})) >= 2
    assert len(colors_present(ctrl, exclude={bg})) == 1
    return block, ctrl


def _compose_combo_output(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    th, tw = map(int, tile.shape)
    ch, cw = map(int, ctrl.shape)
    out = np.full((ch * th, cw * tw), int(bg), dtype=int)

    ctrl_col = int(colors_present(ctrl, exclude={bg})[0])
    mask = (ctrl == int(ctrl_col))
    rotations = [np.rot90(tile, k=k) for k in range(4)]

    filled_seen = 0
    for r in range(ch):
        for c in range(cw):
            if bool(mask[r, c]):
                # Only filled cells advance the 0/90/180/270 cycle.
                patch = rotations[filled_seen % 4]
                filled_seen += 1
            else:
                patch = rotations[2]
            paste(out, patch, r * th, c * tw)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a monochrome control sprite into a dense tile mosaic.

    Input:
    - Exactly two 8-connected foreground components on the background color.
    - One component is a monochrome rectangular control sprite with bg holes.
    - The other is a multicolor square tile.

    Output:
    - Every control-sprite bbox position becomes one tile-sized block.
    - Filled positions use the tile rotated by the count of previous filled positions:
      0, 90, 180, 270 degrees repeating in row-major scan order.
    - Hole positions always use the 180-degree rotation.
    - The result is already tight-cropped because its size is exactly
      (ctrl_h * tile_h, ctrl_w * tile_w).
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    tile, ctrl = _identify_objects(g, bg)
    return _compose_combo_output(tile, ctrl, bg)


def _all_rotations_distinct(tile: np.ndarray) -> bool:
    rots = [np.rot90(tile, k=k) for k in range(4)]
    for i in range(4):
        for j in range(i + 1, 4):
            if np.array_equal(rots[i], rots[j]):
                return False
    return True


def _make_tile(bg: int, fg_colors: list[int], rng: np.random.Generator) -> np.ndarray:
    for _ in range(300):
        size = int(rng.integers(3, 6))
        n_tile_colors = int(rng.integers(2, min(4, len(fg_colors)) + 1))
        tile_colors = [int(x) for x in rng.choice(fg_colors, size=n_tile_colors, replace=False)]

        tile = rng.choice(tile_colors, size=(size, size), replace=True).astype(int)
        for idx, color in enumerate(tile_colors):
            rr = idx % size
            cc = (2 * idx + 1) % size
            tile[rr, cc] = int(color)

        if not _all_rotations_distinct(tile):
            continue
        if any(np.count_nonzero(tile != np.rot90(tile, k=k)) < 2 for k in (1, 2, 3)):
            continue
        return tile
    raise RuntimeError("failed to build asymmetric tile")


def _neighbors8(h: int, w: int, r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w:
                out.append((rr, cc))
    return out


def _make_ctrl(bg: int, color: int, rng: np.random.Generator) -> np.ndarray:
    for _ in range(300):
        h = int(rng.integers(4, 6))
        w = int(rng.integers(4, 6))
        area = h * w
        min_fill = max(6, area // 2)
        max_fill = min(area - 2, int(np.floor(0.72 * area)))
        if min_fill > max_fill:
            continue
        target = int(rng.integers(min_fill, max_fill + 1))

        ar = int(rng.integers(0, h))
        ac = int(rng.integers(0, w))
        mask = np.zeros((h, w), dtype=bool)
        mask[:, ac] = True
        mask[ar, :] = True

        while int(mask.sum()) < target:
            frontier = []
            for r in range(h):
                for c in range(w):
                    if mask[r, c]:
                        continue
                    if any(mask[rr, cc] for rr, cc in _neighbors8(h, w, r, c)):
                        frontier.append((r, c))
            if not frontier:
                break
            pick = frontier[int(rng.integers(len(frontier)))]
            mask[pick] = True

        scan = mask.flatten().tolist()
        first_hole = next((i for i, v in enumerate(scan) if not v), None)
        last_fill = max((i for i, v in enumerate(scan) if v), default=-1)
        later_fill_count = 0 if first_hole is None else sum(1 for i in range(first_hole + 1, len(scan)) if scan[i])
        if int(mask.sum()) < 6:
            continue
        if area - int(mask.sum()) < 2:
            continue
        if first_hole is None or first_hole >= last_fill or later_fill_count < 2:
            continue

        ctrl = np.full((h, w), int(bg), dtype=int)
        ctrl[mask] = int(color)
        return ctrl
    raise RuntimeError("failed to build control sprite")


def _compose_plain_output(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    th, tw = map(int, tile.shape)
    ch, cw = map(int, ctrl.shape)
    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if ctrl[r, c] != bg:
                paste(out, tile, r * th, c * tw)
    return out


def _compose_holes_only_output(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    th, tw = map(int, tile.shape)
    ch, cw = map(int, ctrl.shape)
    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    tile180 = np.rot90(tile, 2)
    for r in range(ch):
        for c in range(cw):
            patch = tile if ctrl[r, c] != bg else tile180
            paste(out, patch, r * th, c * tw)
    return out


def _compose_all_positions_cycle_output(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    th, tw = map(int, tile.shape)
    ch, cw = map(int, ctrl.shape)
    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    rotations = [np.rot90(tile, k=k) for k in range(4)]
    step = 0
    for r in range(ch):
        for c in range(cw):
            if ctrl[r, c] != bg:
                patch = rotations[step % 4]
            else:
                patch = rotations[2]
            paste(out, patch, r * th, c * tw)
            step += 1
    return out


def _behavior_metrics(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> dict[str, int | bool]:
    combo = _compose_combo_output(tile, ctrl, bg)
    plain = _compose_plain_output(tile, ctrl, bg)
    holes_only = _compose_holes_only_output(tile, ctrl, bg)
    all_pos_cycle = _compose_all_positions_cycle_output(tile, ctrl, bg)

    filled = int(np.count_nonzero(ctrl != bg))
    holes = int(np.count_nonzero(ctrl == bg))
    cycle_counts = [0, 0, 0, 0]
    seen = 0
    first_hole_idx = None
    post_hole_filled = 0
    flat = ctrl.flatten()
    for idx, cell in enumerate(flat):
        if int(cell) == int(bg):
            if first_hole_idx is None:
                first_hole_idx = idx
            continue
        if first_hole_idx is not None:
            post_hole_filled += 1
    for cell in flat:
        if int(cell) == int(bg):
            continue
        cycle_counts[seen % 4] += 1
        seen += 1

    return {
        "filled": filled,
        "holes": holes,
        "post_hole_filled": post_hole_filled,
        "cycle_0": cycle_counts[0],
        "cycle_1": cycle_counts[1],
        "cycle_2": cycle_counts[2],
        "cycle_3": cycle_counts[3],
        "same_plain": bool(np.array_equal(combo, plain)),
        "same_holes_only": bool(np.array_equal(combo, holes_only)),
        "same_all_pos_cycle": bool(np.array_equal(combo, all_pos_cycle)),
    }


def _make_input(tile: np.ndarray, ctrl: np.ndarray, bg: int, rng: np.random.Generator) -> np.ndarray:
    horizontal = bool(rng.integers(0, 2))
    gap = 1

    if horizontal:
        if rng.random() < 0.5:
            left_obj, right_obj = tile, ctrl
        else:
            left_obj, right_obj = ctrl, tile
        lh, lw = map(int, left_obj.shape)
        rh, rw = map(int, right_obj.shape)
        H = int(rng.integers(max(12, max(lh, rh) + 2), 19))
        W = int(rng.integers(max(12, lw + rw + gap + 2), 19))
        slack = W - (lw + rw + gap)
        c_left = int(rng.integers(0, slack + 1))
        c_right = int(rng.integers(c_left + lw + gap, W - rw + 1))
        r_left = int(rng.integers(0, H - lh + 1))
        r_right = int(rng.integers(0, H - rh + 1))
        if left_obj is tile:
            tile_pos = (r_left, c_left)
            ctrl_pos = (r_right, c_right)
        else:
            ctrl_pos = (r_left, c_left)
            tile_pos = (r_right, c_right)
    else:
        if rng.random() < 0.5:
            top_obj, bot_obj = tile, ctrl
        else:
            top_obj, bot_obj = ctrl, tile
        th, tw = map(int, top_obj.shape)
        bh, bw = map(int, bot_obj.shape)
        H = int(rng.integers(max(12, th + bh + gap + 2), 19))
        W = int(rng.integers(max(12, max(tw, bw) + 2), 19))
        slack = H - (th + bh + gap)
        r_top = int(rng.integers(0, slack + 1))
        r_bot = int(rng.integers(r_top + th + gap, H - bh + 1))
        c_top = int(rng.integers(0, W - tw + 1))
        c_bot = int(rng.integers(0, W - bw + 1))
        if top_obj is tile:
            tile_pos = (r_top, c_top)
            ctrl_pos = (r_bot, c_bot)
        else:
            ctrl_pos = (r_top, c_top)
            tile_pos = (r_bot, c_bot)

    grid = np.full((H, W), int(bg), dtype=int)
    paste(grid, tile, *tile_pos)
    paste(grid, ctrl, *ctrl_pos, transparent=bg)
    return grid


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    palette = [int(x) for x in rng.choice(np.arange(10, dtype=int), size=6, replace=False)]
    bg = int(palette[0])
    tile_palette = palette[1:5]
    ctrl_color = int(palette[5])

    tile = _make_tile(bg, tile_palette, rng)
    ctrl = _make_ctrl(bg, ctrl_color, rng)
    inp = _make_input(tile, ctrl, bg, rng)
    expected = _compose_combo_output(tile, ctrl, bg)

    comps = find_components(inp, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    if len(comps) != 2:
        return None

    colors_per_comp = []
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        patch = inp[r0 : r1 + 1, c0 : c1 + 1]
        colors_per_comp.append(sorted(set(int(x) for x in patch.flatten()) - {bg}))
    tile_fg_count = len(set(int(x) for x in tile.flatten()) - {bg})
    if sorted(len(cols) for cols in colors_per_comp) != [1, tile_fg_count]:
        return None

    metrics = _behavior_metrics(tile, ctrl, bg)
    if metrics["same_plain"] or metrics["same_holes_only"] or metrics["same_all_pos_cycle"]:
        return None
    if min(int(metrics["cycle_0"]), int(metrics["cycle_1"]), int(metrics["cycle_2"]), int(metrics["cycle_3"])) == 0:
        return None
    if int(metrics["post_hole_filled"]) < 2:
        return None

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid input")
