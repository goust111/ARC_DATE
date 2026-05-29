# Variant of bf45cf4b
# Dimension: D9 — 多步序列与链式处理 (multi_step)
# Change: 单步处理：识别两个分量→区分角色→执行像素放大→裁剪输出... → 两步处理：先像素放大，再对输出执行全局变换（旋转/翻转）
#
# Design analysis
# spec_variant:
#   input_objects:
#     - exactly 2 foreground 8-connected components, same as the source task
#     - one monochrome control sprite with bg holes inside its bbox
#     - one multicolor square tile
#   preserved_step_1:
#     - identify the two components
#     - choose the multicolor square as the tile and the monochrome object as control
#     - expand every filled control cell into a full copy of the tile
#   new_step_2:
#     - inspect the relative placement of tile vs control in the input
#     - use that placement as a global transform code applied to the expanded output
#   direction_code:
#     - tile strictly above control -> flip vertically (top/bottom)
#     - tile strictly below control -> flip horizontally (left/right)
#
# a) Concrete behavior trace
#   Original behavior:
#     - if ctrl is a 3x4 mask and tile is a 3x3 patch, the solver stamps the tile
#       into a 9x12 output according to the filled pixels of ctrl, and returns it.
#   Variant behavior:
#     - perform exactly the same 9x12 stamping first
#     - then inspect where the tile sits relative to ctrl in the input
#     - example: if the tile is placed to the left of ctrl, the 9x12 staged output
#       is rotated clockwise into a 12x9 final output
#   Difference:
#     - the first step alone reproduces the source task
#     - the second step changes orientation globally, so solving now requires
#       discovering a chained transformation rather than a single expansion rule
#
# b) Complexity calibration
#   - Keep the object count at 2 so the task is still locally parsable.
#   - Use only 2 transform categories, both tied to a visible spatial cue.
#   - Avoid adding color remapping or extra markers; the new complexity comes only
#     from the second geometric step.
#   - To keep that step visible, both the tile and the control mask are generated
#     asymmetrically, and samples where the transformed output equals the stage-1
#     output are rejected.
#
# c) Source-code assumptions that must change
#   - In the original main(), returning `out` immediately after stamping is only
#     correct because there is no post-processing step. That assumption is invalid.
#   - The original generator uses a largely symmetric tile; under rotation/flip,
#     many samples would look unchanged. That is harmless in the source task but
#     would hide the new rule here, so the generator must reject symmetric tiles.
#   - Direction inference now relies on the two bboxes being strictly separated
#     along one axis. The generator therefore places the objects in only four
#     axis-coded layouts: above/below.
#
# d) Placement legality
#   - Objects are pasted with at least one bg cell gap, never overlap, and stay
#     inside a 12..18 by 12..18 grid.
#   - Horizontal codes align the two objects on the same top row.
#   - Vertical codes align the two objects on the same left column.
#   - These placements guarantee exactly one unambiguous direction code.
#
# e) Visual observability
#   - A human can first notice the familiar "pixel enlargement" from the source
#     task, then notice that the final output orientation depends on whether the
#     tile is above/below the control in the input.
#   - The control/tile asymmetry makes the second step visibly nontrivial.
#
# f) Visual cleanliness
#   - No extra markers are added.
#   - The two source objects remain disconnected and uncluttered.
#   - The output is still a single compact assembled sprite, only globally turned.
#
# g) Color inference
#   - Colors are not remapped. The only new logic is geometric, so no extra color
#     reasoning burden is introduced.
#
# main() assumption review for the new rule
#   - Component sorting assumes the tile has >1 foreground color and ctrl has 1.
#     The generator enforces this.
#   - Direction decoding assumes one bbox is strictly above/below the
#     other. The generator enforces axis-coded placements only.
#   - The global transform assumes the stage-1 output is intentionally asymmetric
#     so the branch matters. The generator enforces that by rejection sampling.

from __future__ import annotations

import numpy as np

try:
    from arc_utils import colors_present, crop_bbox, find_components, get_bg, paste
except ImportError:  # pragma: no cover - fallback for direct file execution
    from solvers.arc_utils import colors_present, crop_bbox, find_components, get_bg, paste


def _apply_transform(arr: np.ndarray, code: str) -> np.ndarray:
    if code == "above":
        return np.flipud(arr).copy()
    if code == "below":
        return np.fliplr(arr).copy()
    raise ValueError(f"unknown transform code: {code}")


def _direction_from_bboxes(block_bbox: tuple[int, int, int, int], ctrl_bbox: tuple[int, int, int, int]) -> str:
    br0, bc0, br1, bc1 = map(int, block_bbox)
    cr0, cc0, cr1, cc1 = map(int, ctrl_bbox)
    if br1 < cr0:
        return "above"
    if cr1 < br0:
        return "below"
    raise AssertionError("tile/control placement must encode vertical direction")


def _compose_scaled_output(block: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    bh, bw = map(int, block.shape)
    ch, cw = map(int, ctrl.shape)

    ctrl_cols = colors_present(ctrl, exclude={bg})
    if not ctrl_cols:
        return np.full((0, 0), int(bg), dtype=int)

    ctrl_col = int(ctrl_cols[0])
    mask = ctrl == int(ctrl_col)
    out = np.full((ch * bh, cw * bw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if bool(mask[r, c]):
                paste(out, block, r * bh, c * bw)
    return out


def _split_roles(grid: np.ndarray, bg: int) -> tuple[dict, dict]:
    comps = find_components(grid, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    assert len(comps) == 2

    infos = []
    for comp in comps:
        patch = crop_bbox(grid, comp["bbox"])
        cols = colors_present(patch, exclude={bg})
        infos.append({"bbox": comp["bbox"], "patch": patch, "cols": cols})

    infos.sort(
        key=lambda d: (
            len(d["cols"]) >= 2,
            len(d["cols"]),
            int(d["patch"].shape[0]) == int(d["patch"].shape[1]),
        ),
        reverse=True,
    )
    return infos[0], infos[1]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    block_info, ctrl_info = _split_roles(g, bg)
    staged = _compose_scaled_output(block_info["patch"], ctrl_info["patch"], bg)
    code = _direction_from_bboxes(block_info["bbox"], ctrl_info["bbox"])
    return _apply_transform(staged, code)


def _random_points(h: int, w: int, k: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    picks = rng.choice(h * w, size=k, replace=False)
    return [(int(v // w), int(v % w)) for v in picks]


def _make_tile(bg: int, fg_colors: list[int], rng: np.random.Generator) -> np.ndarray:
    for _ in range(300):
        size = int(rng.integers(3, 6))
        n_tile_colors = int(rng.integers(2, min(4, len(fg_colors)) + 1))
        tile_colors = [int(x) for x in rng.choice(fg_colors, size=n_tile_colors, replace=False)]
        tile = rng.choice(tile_colors, size=(size, size), replace=True).astype(int)

        # Force every chosen color to appear.
        for (r, c), color in zip(_random_points(size, size, n_tile_colors, rng), tile_colors):
            tile[r, c] = int(color)

        if len(colors_present(tile, exclude={bg})) < 2:
            continue
        if any(np.array_equal(tile, _apply_transform(tile, code)) for code in ("above", "below")):
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
        h = int(rng.integers(3, 7))
        w = int(rng.integers(3, 7))
        area = h * w
        min_fill = max(4, int(np.ceil(0.45 * area)))
        max_fill = max(min_fill, int(np.floor(0.72 * area)))
        target = int(rng.integers(min_fill, max_fill + 1))

        sr = int(rng.integers(0, h))
        sc = int(rng.integers(0, w))
        mask = np.zeros((h, w), dtype=bool)
        mask[sr, sc] = True

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

        if int(mask.sum()) != target:
            continue
        if int(mask.sum()) >= area:
            continue

        ctrl = np.full((h, w), int(bg), dtype=int)
        ctrl[mask] = int(color)

        if any(np.array_equal(mask, _apply_transform(mask.astype(int), code).astype(bool)) for code in ("above", "below")):
            continue
        return ctrl
    raise RuntimeError("failed to build asymmetric control sprite")


def _make_input(
    tile: np.ndarray,
    ctrl: np.ndarray,
    bg: int,
    direction: str,
    rng: np.random.Generator,
) -> np.ndarray:
    th, tw = map(int, tile.shape)
    ch, cw = map(int, ctrl.shape)
    gap = int(rng.integers(1, 4))

    H = int(rng.integers(max(12, th + ch + gap + 2), 19))
    W = int(rng.integers(max(12, max(tw, cw) + 2), 19))
    col = int(rng.integers(1, W - max(tw, cw)))
    if direction == "above":
        tile_r = int(rng.integers(1, H - (th + gap + ch)))
        ctrl_r = tile_r + th + gap
    else:
        ctrl_r = int(rng.integers(1, H - (ch + gap + th)))
        tile_r = ctrl_r + ch + gap
    tile_pos = (tile_r, col)
    ctrl_pos = (ctrl_r, col)

    grid = np.full((H, W), int(bg), dtype=int)
    paste(grid, tile, *tile_pos)
    paste(grid, ctrl, *ctrl_pos, transparent=bg)
    return grid


def _all_transform_outputs(base: np.ndarray) -> dict[str, np.ndarray]:
    return {code: _apply_transform(base, code) for code in ("above", "below")}


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    palette = [int(x) for x in rng.choice(np.arange(10, dtype=int), size=6, replace=False)]
    bg = int(palette[0])
    tile_fg = palette[1:4]
    ctrl_color = int(palette[4])

    tile = _make_tile(bg, tile_fg, rng)
    ctrl = _make_ctrl(bg, ctrl_color, rng)
    base = _compose_scaled_output(tile, ctrl, bg)
    transform_outputs = _all_transform_outputs(base)

    if any(np.array_equal(base, arr) for arr in transform_outputs.values()):
        return None

    unique_keys: list[np.ndarray] = []
    for arr in transform_outputs.values():
        if not any(np.array_equal(arr, seen) for seen in unique_keys):
            unique_keys.append(arr)
    if len(unique_keys) < 2:
        return None

    direction = str(rng.choice(np.array(["above", "below"], dtype=object)))
    inp = _make_input(tile, ctrl, bg, direction, rng)
    expected = transform_outputs[direction]

    comps = find_components(inp, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    if len(comps) != 2:
        return None

    block_info, ctrl_info = _split_roles(inp, bg)
    if len(block_info["cols"]) < 2 or len(ctrl_info["cols"]) != 1:
        return None
    if _direction_from_bboxes(block_info["bbox"], ctrl_info["bbox"]) != direction:
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
