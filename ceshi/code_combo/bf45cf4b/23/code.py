# Combo variant of bf45cf4b (index 23)
# combo_spec: D2/1+D3/1
# Name: 面积区分矩形瓦片非等比放大

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, crop_bbox, find_components, get_bg, paste

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior:
#    - The input still has exactly two disconnected 8-connected foreground
#      components, but both are now multicolor so the original color-count cue is
#      intentionally removed.
#    - Example trace:
#        tile  = a 2x4 rectangle with 8 fg cells and colors {2, 7}
#        ctrl  = a 5x5 square mask with holes, 17 fg cells, colors {1, 3}
#      Original bf45cf4b would prefer the multicolor square component and would
#      therefore misidentify ctrl as the tile. The fused rule instead assigns
#      roles by component area: smaller area -> tile, larger area -> control.
#      Output size becomes (5*2) x (5*4) = 10 x 20. Every non-bg cell in the
#      control bbox stamps one 2x4 copy of the tile; every bg hole becomes a
#      2x4 bg rectangle.
#    - Relative to the seed task, two steps change together:
#      1. role assignment is driven by area rather than color count;
#      2. stamping is anisotropic because tile height and width differ.
#
# b) Complexity calibration:
#    - The tile aspect ratio is drawn from a small catalog of obvious rectangles
#      instead of arbitrary sizes, so humans can see repeated row/column scaling.
#    - The control sprite is forced to be a larger square mask. That gives a
#      stable visual contrast and also makes the old "prefer square" heuristic
#      systematically wrong when both objects use the same number of colors.
#    - Both objects use exactly two fg colors. This is enough to suppress the old
#      single-color control cue without making the scene visually noisy.
#
# c) Implicit assumptions from the original main():
#    - `find_components(... monochromatic=False)` still assumes exactly two whole
#      foreground components. The generator preserves this with a full bg gap.
#    - The original sort by `(is_multicolor, n_colors, is_square)` is invalid:
#      both components are multicolor, and the larger control is square, so that
#      legacy ranking would pick the wrong object.
#    - `ctrl_cols[0]` is no longer meaningful because the control is not
#      monochrome. The correct filled mask is simply `ctrl != bg`.
#    - `bh, bw = block.shape` already supported non-square tiles; in this combo it
#      is now essential because `bh != bw` for every generated sample.
#
# d) Placement legality:
#    - The generator picks a 12..18 by 12..18 canvas and places the two objects
#      either side-by-side or stacked with at least one full bg row/column
#      between them, preventing overlap and accidental 8-connectivity merging.
#
# e) Visual observability:
#    - The smaller object is always a compact rectangle with unequal sides; the
#      larger object is always a square mask with holes. From 2-3 examples a
#      human can see that the output motif copies the smaller rectangle and that
#      horizontal/vertical scale factors differ.
#    - Because both objects are multicolor, size/role must be inferred from
#      geometry, not hidden color-id semantics.
#
# f) Visual cleanliness:
#    - Tile colors and control colors are disjoint, object bboxes are isolated,
#      and the control colorization follows a simple parity pattern. This keeps
#      the input readable even though control colors are ignored by the output.
#
# g) Color inferability:
#    - No color remapping occurs. The tile is copied verbatim into the output.
#      Control colors only indicate filled cells versus bg holes, and the two
#      control colors repeat in a simple structure so humans can treat them as a
#      single mask object.
#
# h) Post-test combo metric:
#    - Every accepted sample must satisfy `tile_h != tile_w`.
#    - Every accepted sample must also satisfy
#      `legacy_color_square_choice != area_choice`, proving the D2 change is
#      active rather than latent.
#
# i) New solver assumptions and generator guarantees:
#    - Assumption: the smaller component by foreground area is the intended tile.
#      Guarantee: `_make_ctrl` is built with strictly larger area than the tile.
#    - Assumption: the control bbox itself is the mask frame to scale. Guarantee:
#      the seed cross touches all four sides, so cropping recovers the full
#      square control bbox.
#    - Assumption: all non-bg cells in the control bbox are "filled". Guarantee:
#      control colors are decorative only; generator never uses bg-like sentinels
#      other than actual holes.


def _component_infos(grid: np.ndarray, bg: int) -> list[dict]:
    comps = find_components(grid, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    infos: list[dict] = []
    for idx, comp in enumerate(comps):
        patch = crop_bbox(grid, comp["bbox"])
        h, w = map(int, patch.shape)
        infos.append(
            {
                "idx": int(idx),
                "patch": patch,
                "area": int(comp["area"]),
                "bbox_area": int(h * w),
                "colors": colors_present(patch, exclude={bg}),
            }
        )
    return infos


def _identify_by_area(infos: list[dict]) -> tuple[dict, dict]:
    """Return (tile, control) using the combo's smaller-area-first rule."""
    ordered = sorted(infos, key=lambda d: (int(d["area"]), int(d["bbox_area"])))
    return ordered[0], ordered[1]


def _identify_by_legacy_rule(infos: list[dict]) -> tuple[dict, dict]:
    ordered = sorted(
        infos,
        key=lambda d: (
            len(d["colors"]) >= 2,
            len(d["colors"]),
            int(d["patch"].shape[0]) == int(d["patch"].shape[1]),
        ),
        reverse=True,
    )
    return ordered[0], ordered[1]


def analyze_input(grid: np.ndarray) -> dict:
    """Return the core geometric facts used to audit the combo behavior."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    infos = _component_infos(g, bg)
    assert len(infos) == 2
    tile, ctrl = _identify_by_area(infos)
    legacy_first, _legacy_second = _identify_by_legacy_rule(infos)
    return {
        "bg": bg,
        "tile_area": int(tile["area"]),
        "ctrl_area": int(ctrl["area"]),
        "tile_shape": tuple(map(int, tile["patch"].shape)),
        "ctrl_shape": tuple(map(int, ctrl["patch"].shape)),
        "tile_colors": len(tile["colors"]),
        "ctrl_colors": len(ctrl["colors"]),
        "legacy_agrees_with_area": bool(legacy_first["idx"] == tile["idx"]),
    }


def combo_behavior_metrics(grid: np.ndarray) -> dict:
    """
    Quantify whether the generated sample really expresses the combo.

    Metrics:
    - `anisotropic_tile`: D3 is active only if tile height != tile width.
    - `area_gap`: D2 is active only if control area is strictly larger.
    - `legacy_would_fail`: the old color-count/square heuristic chooses the
      wrong first object, so the sample does not collapse back to the seed task.
    """
    info = analyze_input(grid)
    th, tw = info["tile_shape"]
    return {
        **info,
        "anisotropic_tile": bool(th != tw),
        "area_gap": int(info["ctrl_area"] - info["tile_area"]),
        "legacy_would_fail": bool(not info["legacy_agrees_with_area"]),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify the smaller foreground component as the tile and stamp it wherever
    the larger component has a non-background cell.

    New assumptions relative to the seed task:
    - role assignment uses foreground area rather than color counts;
    - every non-bg control cell is part of the binary mask, even if the control
      itself is multicolor.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    infos = _component_infos(g, bg)
    assert len(infos) == 2

    tile_info, ctrl_info = _identify_by_area(infos)
    tile = tile_info["patch"]
    ctrl = ctrl_info["patch"]

    th, tw = map(int, tile.shape)
    ch, cw = map(int, ctrl.shape)
    mask = (ctrl != bg)

    out = np.full((ch * th, cw * tw), bg, dtype=int)
    for r in range(ch):
        for c in range(cw):
            if bool(mask[r, c]):
                paste(out, tile, r * th, c * tw)
    return out


def _reflection_orbits(h: int, w: int) -> list[tuple[list[tuple[int, int]], bool]]:
    seen: set[tuple[int, int]] = set()
    out: list[tuple[list[tuple[int, int]], bool]] = []
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            orbit = {
                (r, c),
                (r, w - 1 - c),
                (h - 1 - r, c),
                (h - 1 - r, w - 1 - c),
            }
            cells = sorted(orbit)
            seen.update(cells)
            border = any(rr in (0, h - 1) or cc in (0, w - 1) for rr, cc in cells)
            out.append((cells, border))
    return out


def _make_tile(bg: int, tile_colors: list[int], rng: np.random.Generator) -> np.ndarray:
    dims = [(2, 3), (3, 2), (2, 4), (4, 2), (2, 5), (5, 2), (3, 4), (4, 3)]
    h, w = dims[int(rng.integers(len(dims)))]
    c0, c1 = [int(x) for x in rng.choice(tile_colors, size=2, replace=False)]

    tile = np.full((h, w), bg, dtype=int)
    orbits = _reflection_orbits(h, w)
    border_idxs = [i for i, (_cells, border) in enumerate(orbits) if border]
    inner_idxs = [i for i, (_cells, border) in enumerate(orbits) if not border]
    assert len(border_idxs) >= 2

    rng.shuffle(border_idxs)
    assignments: dict[int, int] = {
        int(border_idxs[0]): c0,
        int(border_idxs[1]): c1,
    }

    for i, (_cells, border) in enumerate(orbits):
        if i in assignments:
            continue
        if border:
            assignments[i] = c0 if bool(rng.integers(0, 2)) else c1
        else:
            palette = [c0, c1, bg] if inner_idxs else [c0, c1]
            assignments[i] = int(palette[int(rng.integers(len(palette)))])

    for i, (cells, _border) in enumerate(orbits):
        for r, c in cells:
            tile[r, c] = assignments[i]

    return tile


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


def _make_ctrl(bg: int, ctrl_colors: list[int], min_area: int, rng: np.random.Generator) -> np.ndarray:
    c0, c1 = [int(x) for x in rng.choice(ctrl_colors, size=2, replace=False)]

    for _ in range(100):
        size = int(rng.integers(5, 8))
        bbox_area = size * size
        base_cross_area = 2 * size - 1
        min_fill = max(base_cross_area, int(np.ceil(0.56 * bbox_area)), int(min_area) + 2)
        max_fill = int(np.floor(0.76 * bbox_area))
        if min_fill > max_fill:
            continue

        target = int(rng.integers(min_fill, max_fill + 1))
        ar = int(rng.integers(0, size))
        ac = int(rng.integers(0, size))
        mask = np.zeros((size, size), dtype=bool)
        mask[:, ac] = True
        mask[ar, :] = True

        while int(mask.sum()) < target:
            frontier = []
            for r in range(size):
                for c in range(size):
                    if mask[r, c]:
                        continue
                    if any(mask[rr, cc] for rr, cc in _neighbors8(size, size, r, c)):
                        frontier.append((r, c))
            if not frontier:
                break
            pick = frontier[int(rng.integers(len(frontier)))]
            mask[pick] = True

        if int(mask.sum()) <= int(min_area):
            continue
        if int(mask.sum()) >= bbox_area:
            continue

        ctrl = np.full((size, size), bg, dtype=int)
        coords = np.argwhere(mask)
        for r, c in coords:
            ctrl[int(r), int(c)] = c0 if (int(r) + int(c)) % 2 == 0 else c1

        if len(colors_present(ctrl, exclude={bg})) < 2:
            rr, cc = map(int, coords[-1])
            ctrl[rr, cc] = c1
        return ctrl

    raise RuntimeError("failed to build control mask with larger area than tile")


def _compose_output(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    th, tw = map(int, tile.shape)
    ch, cw = map(int, ctrl.shape)
    out = np.full((ch * th, cw * tw), bg, dtype=int)
    for r in range(ch):
        for c in range(cw):
            if int(ctrl[r, c]) != bg:
                paste(out, tile, r * th, c * tw)
    return out


def _make_input(tile: np.ndarray, ctrl: np.ndarray, bg: int, rng: np.random.Generator) -> np.ndarray:
    horizontal = bool(rng.integers(0, 2))
    gap = 1

    if horizontal:
        if bool(rng.integers(0, 2)):
            left_obj, right_obj = tile, ctrl
        else:
            left_obj, right_obj = ctrl, tile
        lh, lw = map(int, left_obj.shape)
        rh, rw = map(int, right_obj.shape)
        H = int(rng.integers(max(12, lh, rh), 19))
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
        if bool(rng.integers(0, 2)):
            top_obj, bot_obj = tile, ctrl
        else:
            top_obj, bot_obj = ctrl, tile
        th, tw = map(int, top_obj.shape)
        bh, bw = map(int, bot_obj.shape)
        H = int(rng.integers(max(12, th + bh + gap), 19))
        W = int(rng.integers(max(12, tw, bw), 19))
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

    grid = np.full((H, W), bg, dtype=int)
    paste(grid, tile, *tile_pos)
    paste(grid, ctrl, *ctrl_pos, transparent=bg)
    return grid


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    palette = [int(x) for x in rng.choice(np.arange(10, dtype=int), size=5, replace=False)]
    bg = int(palette[0])
    tile_colors = [int(x) for x in palette[1:3]]
    ctrl_colors = [int(x) for x in palette[3:5]]

    tile = _make_tile(bg, tile_colors, rng)
    tile_area = int(np.count_nonzero(tile != bg))
    ctrl = _make_ctrl(bg, ctrl_colors, tile_area, rng)
    inp = _make_input(tile, ctrl, bg, rng)
    expected = _compose_output(tile, ctrl, bg)

    if int(get_bg(inp)) != bg:
        return None

    infos = _component_infos(inp, bg)
    if len(infos) != 2:
        return None

    metrics = combo_behavior_metrics(inp)

    if metrics["tile_area"] >= metrics["ctrl_area"]:
        return None
    if not metrics["anisotropic_tile"]:
        return None
    if metrics["ctrl_shape"][0] != metrics["ctrl_shape"][1]:
        return None
    if metrics["tile_colors"] != 2 or metrics["ctrl_colors"] != 2:
        return None
    if not metrics["legacy_would_fail"]:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate a sample where area-based identification is necessary."""
    for _ in range(500):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid input")
