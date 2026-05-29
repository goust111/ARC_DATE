# Combo variant of bf45cf4b (index 22)
# combo_spec: D9/0+D13/0
# Name: 两步放大后颜色替换
#
# Design analysis
# ---------------
# a) Concrete behavior trace
#    - Source task:
#      one monochrome control sprite acts as a binary mask; each filled control
#      cell expands to one full copy of the multicolor square tile.
#    - Fusion rule here:
#      step 1: perform exactly that same enlargement.
#      step 2: recolor the enlarged result globally by replacing one whole tile
#      color class with the control color.
#    - The replaced color class is not fixed globally. It is encoded by the
#      vertical relation of the two input objects:
#        * tile above ctrl  -> replace the tile's corner accent color.
#        * tile below ctrl  -> replace the tile's center-cross color.
#      Example with a 3x3 tile of colors A/B/C and ctrl color X:
#        tile = A B A
#               B C B
#               A B A
#        ctrl mask = 1 0
#                    1 1
#        step 1 output = [tile][bg]
#                        [tile][tile]
#        step 2:
#          if tile is above ctrl, every A in the enlarged output becomes X.
#          if tile is below ctrl, every C in the enlarged output becomes X.
#    - Difference from the source task:
#      the solver must now infer a chained process rather than a single stamp,
#      and the second step depends on visible spatial context from the input.
#
# b) Complexity calibration
#    - Keep exactly 2 foreground components, as in the source task.
#    - Limit the post-step code to 2 cases only (above/below), so a human can
#      infer it from 2-4 examples.
#    - Use a tile with 3 stable color roles:
#        corner accent, base fill, center cross.
#      The two replaceable roles each repeat many times, so the recolor is
#      visually obvious and not a one-pixel accident.
#    - Avoid left/right or four-way coding; two cases are enough and reduce
#      ambiguity.
#
# c) Hidden assumptions audit
#    - Role identification still assumes one component is monochrome and the
#      other is a multicolor square. The generator preserves that.
#    - The original solver returned immediately after stamping. That assumption
#      is no longer valid because there is now a second global recolor step.
#    - The new recolor logic assumes:
#        1. the tile has a non-background corner color at [0, 0];
#        2. the tile has a distinct center-cross color at [mid, mid];
#        3. these two colors differ from the control color.
#      The generator enforces all three.
#    - Direction decoding assumes the tile and ctrl are strictly vertically
#      separated. The generator therefore only places tile strictly above or
#      strictly below ctrl, never side-by-side.
#
# d) Placement legality
#    - Objects are pasted inside a 12..18 by 12..18 grid with at least one
#      background row between them.
#    - They never overlap and remain 8-connected-disconnected as separate
#      components.
#    - Vertical alignment keeps the direction code unambiguous.
#
# e) Visual observability
#    - A solver can first spot the original "pixel enlargement by tile" rule.
#    - Across examples, they can then compare the output colors with the input
#      tile and notice that one repeated tile role changes into the ctrl color.
#    - The tile being above/below ctrl is an easy visual cue that supports
#      inferring which repeated role gets replaced.
#
# f) Visual cleanliness
#    - No extra markers or auxiliary objects are introduced.
#    - The control sprite and tile remain clearly separated in the input.
#    - The output remains one compact enlarged sprite, with only one coherent
#      recolor class changed.
#
# g) Change inferability
#    - The output color change always maps to the ctrl color, which is present in
#      the input as the only color of the control sprite.
#    - The recolored source class is structural rather than numeric: corners or
#      cross, never "smallest color id" or similar invisible metadata.
#
# main() assumption review
#   - `_identify_parts()` assumes exactly 2 foreground components: generator
#     verifies this.
#   - `_direction_from_bboxes()` assumes strict vertical separation: generator
#     constructs only those layouts.
#   - `_source_color_for_code()` assumes the tile uses the prescribed 3-role
#     palette: generator builds only that family of tiles.
#   - `_recolor_output()` assumes the selected source color actually appears in
#     the enlarged output: generator rejects degenerate samples where stage 2
#     would be inert.

from __future__ import annotations

import numpy as np

try:
    from arc_utils import colors_present, crop_bbox, find_components, get_bg, paste
except ImportError:  # pragma: no cover - fallback for direct execution
    from solvers.arc_utils import colors_present, crop_bbox, find_components, get_bg, paste


def _identify_parts(grid: np.ndarray) -> tuple[int, dict[str, object], dict[str, object]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    assert len(comps) == 2

    infos: list[dict[str, object]] = []
    for comp in comps:
        patch = np.array(crop_bbox(g, comp["bbox"]), dtype=int)
        cols = colors_present(patch, exclude={bg})
        infos.append({"bbox": tuple(comp["bbox"]), "patch": patch, "cols": cols})

    infos.sort(
        key=lambda d: (
            len(d["cols"]) >= 2,
            len(d["cols"]),
            int(d["patch"].shape[0]) == int(d["patch"].shape[1]),
        ),
        reverse=True,
    )
    return bg, infos[0], infos[1]


def _direction_from_bboxes(block_bbox: tuple[int, int, int, int], ctrl_bbox: tuple[int, int, int, int]) -> str:
    br0, bc0, br1, bc1 = map(int, block_bbox)
    cr0, cc0, cr1, cc1 = map(int, ctrl_bbox)
    if br1 < cr0:
        return "above"
    if cr1 < br0:
        return "below"
    raise AssertionError("tile/control placement must encode strict vertical relation")


def _compose_scaled_output(block: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    ctrl_cols = colors_present(ctrl, exclude={bg})
    if not ctrl_cols:
        return np.full((0, 0), int(bg), dtype=int)

    ctrl_color = int(ctrl_cols[0])
    mask = ctrl == int(ctrl_color)
    bh, bw = map(int, block.shape)
    ch, cw = map(int, ctrl.shape)
    out = np.full((ch * bh, cw * bw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if bool(mask[r, c]):
                paste(out, block, r * bh, c * bw)
    return out


def _source_color_for_code(block: np.ndarray, code: str) -> int:
    mid_r = int(block.shape[0] // 2)
    mid_c = int(block.shape[1] // 2)
    if code == "above":
        return int(block[0, 0])
    if code == "below":
        return int(block[mid_r, mid_c])
    raise ValueError(f"unknown code: {code}")


def _recolor_output(arr: np.ndarray, src_color: int, dst_color: int) -> np.ndarray:
    out = np.array(arr, copy=True)
    out[out == int(src_color)] = int(dst_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    bg, block_info, ctrl_info = _identify_parts(grid)
    block = np.array(block_info["patch"], dtype=int)
    ctrl = np.array(ctrl_info["patch"], dtype=int)

    ctrl_cols = colors_present(ctrl, exclude={bg})
    if not ctrl_cols:
        return np.full((0, 0), int(bg), dtype=int)
    ctrl_color = int(ctrl_cols[0])

    staged = _compose_scaled_output(block, ctrl, bg)
    code = _direction_from_bboxes(block_info["bbox"], ctrl_info["bbox"])
    src_color = _source_color_for_code(block, code)
    assert src_color != ctrl_color
    return _recolor_output(staged, src_color, ctrl_color)


def _neighbors8(h: int, w: int, r: int, c: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
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
    h = int(rng.integers(3, 6))
    w = int(rng.integers(3, 6))
    area = h * w
    min_fill = (area + 1) // 2
    max_fill = int(np.floor(0.72 * area))
    target = int(rng.integers(min_fill, max_fill + 1))

    ar = int(rng.integers(0, h))
    ac = int(rng.integers(0, w))
    mask = np.zeros((h, w), dtype=bool)
    mask[:, ac] = True
    mask[ar, :] = True

    while int(mask.sum()) < target:
        frontier: list[tuple[int, int]] = []
        for r in range(h):
            for c in range(w):
                if mask[r, c]:
                    continue
                if any(mask[rr, cc] for rr, cc in _neighbors8(h, w, r, c)):
                    frontier.append((r, c))
        pick = frontier[int(rng.integers(len(frontier)))]
        mask[pick] = True

    ctrl = np.full((h, w), int(bg), dtype=int)
    ctrl[mask] = int(color)
    return ctrl


def _make_tile(bg: int, ctrl_color: int, fg_colors: list[int], rng: np.random.Generator) -> np.ndarray:
    size = 5
    corner_color, base_color, cross_color = [int(x) for x in rng.choice(fg_colors, size=3, replace=False)]

    tile = np.full((size, size), int(base_color), dtype=int)
    tile[0, 0] = int(corner_color)
    tile[0, -1] = int(corner_color)
    tile[-1, 0] = int(corner_color)
    tile[-1, -1] = int(corner_color)

    mid = size // 2
    tile[mid, :] = int(cross_color)
    tile[:, mid] = int(cross_color)

    tile[0, 0] = int(corner_color)
    tile[0, -1] = int(corner_color)
    tile[-1, 0] = int(corner_color)
    tile[-1, -1] = int(corner_color)

    assert int(tile[0, 0]) != int(bg)
    assert int(tile[0, 0]) != int(ctrl_color)
    assert int(tile[mid, mid]) != int(bg)
    assert int(tile[mid, mid]) != int(ctrl_color)
    assert int(tile[0, 0]) != int(tile[mid, mid])
    assert len(colors_present(tile, exclude={bg})) == 3
    assert np.count_nonzero(tile == int(tile[0, 0])) >= 4
    assert np.count_nonzero(tile == int(tile[mid, mid])) >= 5
    return tile


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

    min_h = max(12, th + ch + gap + 2)
    min_w = max(12, max(tw, cw) + 2)
    H = int(rng.integers(min_h, 19))
    W = int(rng.integers(min_w, 19))

    col = int(rng.integers(1, W - max(tw, cw)))
    if direction == "above":
        tile_r = int(rng.integers(1, H - (th + gap + ch)))
        ctrl_r = tile_r + th + gap
    elif direction == "below":
        ctrl_r = int(rng.integers(1, H - (ch + gap + th)))
        tile_r = ctrl_r + ch + gap
    else:
        raise ValueError(f"unknown direction: {direction}")

    grid = np.full((H, W), int(bg), dtype=int)
    paste(grid, tile, tile_r, col)
    paste(grid, ctrl, ctrl_r, col, transparent=bg)
    return grid


def _expected_output(tile: np.ndarray, ctrl: np.ndarray, bg: int, direction: str) -> np.ndarray:
    ctrl_color = int(colors_present(ctrl, exclude={bg})[0])
    staged = _compose_scaled_output(tile, ctrl, bg)
    src_color = _source_color_for_code(tile, direction)
    return _recolor_output(staged, src_color, ctrl_color)


def _behavior_metrics(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> dict[str, int]:
    staged = _compose_scaled_output(tile, ctrl, bg)
    ctrl_color = int(colors_present(ctrl, exclude={bg})[0])
    metrics: dict[str, int] = {}
    for direction in ("above", "below"):
        src_color = _source_color_for_code(tile, direction)
        expected = _recolor_output(staged, src_color, ctrl_color)
        metrics[f"{direction}_changed_cells"] = int(np.count_nonzero(staged != expected))
    metrics["modes_distinct"] = int(
        not np.array_equal(
            _recolor_output(staged, _source_color_for_code(tile, "above"), ctrl_color),
            _recolor_output(staged, _source_color_for_code(tile, "below"), ctrl_color),
        )
    )
    return metrics


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    palette = [int(x) for x in rng.choice(np.arange(10, dtype=int), size=6, replace=False)]
    bg = int(palette[0])
    ctrl_color = int(palette[1])
    tile_colors = palette[2:]

    tile = _make_tile(bg, ctrl_color, tile_colors, rng)
    ctrl = _make_ctrl(bg, ctrl_color, rng)
    direction = str(rng.choice(np.array(["above", "below"], dtype=object)))
    inp = _make_input(tile, ctrl, bg, direction, rng)
    expected = _expected_output(tile, ctrl, bg, direction)

    comps = find_components(inp, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    if len(comps) != 2:
        return None

    bg2, block_info, ctrl_info = _identify_parts(inp)
    if bg2 != bg:
        return None
    if len(block_info["cols"]) != 3 or len(ctrl_info["cols"]) != 1:
        return None
    if tuple(np.array(block_info["patch"]).shape) != tuple(tile.shape):
        return None
    if _direction_from_bboxes(block_info["bbox"], ctrl_info["bbox"]) != direction:
        return None

    metrics = _behavior_metrics(tile, ctrl, bg)
    if metrics["modes_distinct"] != 1:
        return None
    if metrics[f"{direction}_changed_cells"] <= 0:
        return None

    staged = _compose_scaled_output(tile, ctrl, bg)
    if np.array_equal(staged, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid combo variant input")
