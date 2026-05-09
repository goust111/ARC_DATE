# Variant of bf45cf4b
# Dimension: D9 — 多步序列与链式处理 (multi_step)
# Change: 单步处理：识别两个分量→区分角色→执行像素放大→裁剪输出... → 先像素放大，再对输出中特定颜色区域执行flood_fill

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, crop_bbox, find_components, flood_fill, get_bg, paste


# Design analysis for the D9 variant
# ----------------------------------
# a) Concrete behavior and trace
#    Input still has exactly two 8-connected foreground components:
#    - a monochrome control sprite with background holes inside its bbox
#    - a multicolor square tile
#    Step 1 is identical to the base task: every filled control cell expands to one full tile,
#    every control-hole cell expands to a tile-sized background rectangle.
#    Step 2 is new: in that expanded output, background components that do NOT touch the output
#    border are enclosed cavities created by the control sprite's holes. Flood-fill all such
#    closed background regions with the control color.
#    Small trace:
#    - Original task: 3x3 ring control + 4x4 full tile -> center control hole becomes a 4x4 bg hole.
#    - Variant: produce the same 12x12 expanded image first, then recolor the central 4x4 bg hole
#      to the control color because it is fully surrounded by tiles and disconnected from the border.
#    Difference: output is no longer just scaled stamping; it depends on a second global pass over
#    the assembled result.
#
# b) Complexity calibration
#    The flood-fill step is visually obvious because the generator forces the control sprite to
#    contain at least one enclosed hole. The tile is fully non-background, so the only background
#    that survives step 1 comes from control holes. This avoids ambiguity between "tile internals"
#    and "post-scale cavities", while still making step 2 essential.
#
# c) Main() lines whose assumptions change
#    - Component detection and role disambiguation remain valid: still exactly two components, tile
#      still has >=2 colors, control still has exactly one fg color.
#    - The original nested paste loop assumed the assembled output was final. That assumption is no
#      longer valid; after the loop, the solver must detect enclosed background components.
#    - Any logic that would recolor all background cells would now be wrong, because only closed
#      regions should change. Therefore the second pass explicitly excludes border-connected bg.
#
# d) Placement legality
#    Generator still places tile and control as two disconnected foreground components with at least
#    one-cell separation inside a 12-18 by 12-18 canvas. Because the control is pasted with
#    transparency=bg, its internal holes remain holes and do not merge the two objects.
#
# e) Visual observability
#    Across examples, the output shows the same scaled tile mosaic as the base task plus one or more
#    conspicuous monochrome islands exactly where enclosed holes should be after scaling. Since those
#    filled islands use the control color, the relation between the two input objects is visible.
#
# f) Visual cleanliness
#    The tile contains no background, so closed regions are clean rectangles or unions of rectangles
#    carved out by control holes. There is no overdraw or mixed semantics inside a single area.
#
# g) Color inference
#    The new fill color is always the control sprite color. That color already appears as the only
#    color of one input object, so the recoloring rule is inferable from few examples.
#
# h) Variant metric used after implementation
#    For generated samples we measure:
#    - enclosed_bg_cells_before_fill > 0
#    - main(input) differs from pure scale-only output
#    If either fails often, the generator has degenerated back toward the base task.
#
# i) New implicit assumptions introduced by main()
#    - The control must have at least one enclosed background component in its bbox, otherwise the
#      second step would be invisible. Generator guarantees this with frame-like sprites.
#    - The tile must be fully non-background so that closed output cavities correspond only to
#      control holes. Generator guarantees this by never placing bg inside the tile.


def _enclosed_bg_mask(grid: np.ndarray, bg: int) -> np.ndarray:
    marker = np.where(grid == int(bg), 0, 1).astype(int)
    h, w = marker.shape
    for r in range(h):
        if marker[r, 0] == 0:
            flood_fill(marker, r, 0, 2, connectivity=4)
        if marker[r, w - 1] == 0:
            flood_fill(marker, r, w - 1, 2, connectivity=4)
    for c in range(w):
        if marker[0, c] == 0:
            flood_fill(marker, 0, c, 2, connectivity=4)
        if marker[h - 1, c] == 0:
            flood_fill(marker, h - 1, c, 2, connectivity=4)
    return marker == 0


def _scale_only(block: np.ndarray, ctrl: np.ndarray, bg: int, ctrl_col: int) -> np.ndarray:
    bh, bw = map(int, block.shape)
    ch, cw = map(int, ctrl.shape)
    mask = ctrl == int(ctrl_col)
    out = np.full((ch * bh, cw * bw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if bool(mask[r, c]):
                paste(out, block, r * bh, c * bw)
    return out


def _apply_closed_hole_fill(out: np.ndarray, bg: int, fill_color: int) -> np.ndarray:
    result = np.array(out, copy=True)
    enclosed = _enclosed_bg_mask(result, bg)
    result[enclosed] = int(fill_color)
    return result


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    1. Solve the original expansion task.
    2. In the expanded output, flood-fill every background component that is fully enclosed
       by tiles (i.e. not connected to the output border via 4-neighbor background) with
       the control sprite color.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    assert len(comps) == 2

    infos = []
    for comp in comps:
        patch = crop_bbox(g, comp["bbox"])
        cols = colors_present(patch, exclude={bg})
        infos.append({"patch": patch, "cols": cols})

    infos.sort(
        key=lambda d: (
            len(d["cols"]) >= 2,
            len(d["cols"]),
            int(d["patch"].shape[0]) == int(d["patch"].shape[1]),
        ),
        reverse=True,
    )
    block = infos[0]["patch"]
    ctrl = infos[1]["patch"]

    ctrl_cols = colors_present(ctrl, exclude={bg})
    if not ctrl_cols:
        return np.full((0, 0), int(bg), dtype=int)

    ctrl_col = int(ctrl_cols[0])
    out = _scale_only(block, ctrl, bg, ctrl_col)
    return _apply_closed_hole_fill(out, bg, ctrl_col)


def _dihedral_orbits(size: int) -> list[list[tuple[int, int]]]:
    seen: set[tuple[int, int]] = set()
    out: list[list[tuple[int, int]]] = []
    s = int(size)
    for r in range(s):
        for c in range(s):
            if (r, c) in seen:
                continue
            orbit = {
                (r, c),
                (c, s - 1 - r),
                (s - 1 - r, s - 1 - c),
                (s - 1 - c, r),
                (r, s - 1 - c),
                (s - 1 - r, c),
                (c, r),
                (s - 1 - c, s - 1 - r),
            }
            cells = sorted(orbit)
            seen.update(cells)
            out.append(cells)
    return out


def _make_tile(fg_colors: list[int], rng: np.random.Generator) -> np.ndarray:
    size = int(rng.integers(3, 6))
    n_tile_colors = int(rng.integers(2, min(4, len(fg_colors) + 1)))
    tile_colors = [int(x) for x in rng.choice(fg_colors, size=n_tile_colors, replace=False)]
    orbits = _dihedral_orbits(size)
    rng.shuffle(orbits)

    tile = np.zeros((size, size), dtype=int)
    assignments: dict[int, int] = {}
    for i, color in enumerate(tile_colors):
        assignments[i] = int(color)
    for i in range(len(orbits)):
        if i not in assignments:
            assignments[i] = int(rng.choice(tile_colors))
    for i, cells in enumerate(orbits):
        for r, c in cells:
            tile[r, c] = assignments[i]
    return tile


def _frame_ratio(h: int, w: int, extra: int = 0) -> float:
    border = 2 * int(h) + 2 * int(w) - 4
    return (border + int(extra)) / float(h * w)


def _make_ctrl(bg: int, color: int, rng: np.random.Generator) -> np.ndarray:
    while True:
        h = int(rng.integers(4, 8))
        w = int(rng.integers(4, 8))
        if h == 4 and w == 4:
            continue

        mask = np.zeros((h, w), dtype=bool)
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True

        options: list[tuple[str, int]] = [("none", 0)]
        if h - 2 >= 3 and _frame_ratio(h, w, extra=h - 2) <= 0.72:
            options.append(("vertical", h - 2))
        if w - 2 >= 3 and _frame_ratio(h, w, extra=w - 2) <= 0.72:
            options.append(("horizontal", w - 2))

        mode = options[int(rng.integers(len(options)))][0]
        if mode == "vertical":
            col = int(rng.integers(1, w - 1))
            mask[1:-1, col] = True
        elif mode == "horizontal":
            row = int(rng.integers(1, h - 1))
            mask[row, 1:-1] = True

        ctrl = np.full((h, w), int(bg), dtype=int)
        ctrl[mask] = int(color)

        enclosed = _enclosed_bg_mask(ctrl, bg)
        if enclosed.any():
            return ctrl


def _compose_output(tile: np.ndarray, ctrl: np.ndarray, bg: int, ctrl_color: int) -> np.ndarray:
    scaled = _scale_only(tile, ctrl, bg, ctrl_color)
    return _apply_closed_hole_fill(scaled, bg, ctrl_color)


def _make_input(tile: np.ndarray, ctrl: np.ndarray, bg: int, rng: np.random.Generator) -> np.ndarray:
    horizontal = bool(rng.integers(0, 2))
    gap = 1

    if horizontal:
        if rng.random() < 0.5:
            left_obj, right_obj = tile, ctrl
        else:
            left_obj, right_obj = ctrl, tile
        lh, lw = left_obj.shape
        rh, rw = right_obj.shape
        H = int(rng.integers(max(12, max(lh, rh)), 19))
        W = int(rng.integers(max(12, lw + rw + gap), 19))
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
        th, tw = top_obj.shape
        bh, bw = bot_obj.shape
        H = int(rng.integers(max(12, th + bh + gap), 19))
        W = int(rng.integers(max(12, max(tw, bw)), 19))
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
    palette = [int(x) for x in rng.choice(np.arange(10, dtype=int), size=5, replace=False)]
    bg = int(palette[0])
    ctrl_color = int(palette[1])
    tile_colors = palette[2:]

    tile = _make_tile(tile_colors, rng)
    ctrl = _make_ctrl(bg, ctrl_color, rng)
    inp = _make_input(tile, ctrl, bg, rng)

    expected = _compose_output(tile, ctrl, bg, ctrl_color)
    scale_only = _scale_only(tile, ctrl, bg, ctrl_color)
    enclosed_before_fill = _enclosed_bg_mask(scale_only, bg)

    if not enclosed_before_fill.any():
        return None
    if np.array_equal(expected, scale_only):
        return None

    comps = find_components(inp, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    if len(comps) != 2:
        return None

    colors_per_comp = []
    for comp in comps:
        patch = crop_bbox(inp, comp["bbox"])
        colors_per_comp.append(sorted(set(int(x) for x in patch.flatten()) - {bg}))
    if sorted(len(cols) for cols in colors_per_comp) != [1, len(set(int(x) for x in tile.flatten()))]:
        return None

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid input")
