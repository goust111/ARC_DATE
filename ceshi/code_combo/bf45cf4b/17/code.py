# Combo variant of bf45cf4b (index 17)
# combo_spec: D4/1+D16/0
# Name: 不规则精灵对角紧贴

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import colors_present, crop_bbox, find_components, get_bg, paste
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[4] / "solvers"))
    from arc_utils import colors_present, crop_bbox, find_components, get_bg, paste


# Design analysis for combo D4/1 + D16/0
# ---------------------------------------
# a) Concrete behavior and trace
# - Original bf45cf4b expects 2 foreground objects that are already separated under
#   multicolor 8-connectivity:
#     1. a monochrome rectangular control bbox containing bg holes
#     2. a multicolor square tile
#   Then it stamps the tile once for every filled control cell.
# - This combo changes two assumptions together:
#   1. D4/1: the control is no longer a holed rectangle. It is a 4-connected
#      non-rectangular sprite such as an L or T. The correct mask is therefore
#      "all non-bg cells inside the control bbox", not "a mostly filled rectangle".
#   2. D16/0: the tile and control are no longer guaranteed to be 8-disconnected.
#      They remain 4-disconnected, but they may touch diagonally.
# - Small traced example:
#     control bbox mask =
#       1 0 0
#       1 0 0
#       1 1 1
#     tile =
#       a b
#       c a
#   Original input would place these two objects with a visible gap. The old solver
#   could split them by `find_components(..., connectivity=8) == 2`.
#   New input places the 2x2 tile at the control's outside corner so one control
#   pixel and one tile pixel meet only by a diagonal. Then:
#     - 8-connectivity sees one merged foreground blob.
#     - 4-connectivity still sees exactly two objects.
#   Output is still the bbox-scaled assembly:
#     every `1` cell in the 3x3 control bbox becomes one 2x2 copy of the tile;
#     every `0` cell becomes one 2x2 bg block.
# - The combined effect is that the solver must both:
#   1. recover object roles from 4-connected splitting rather than 8-connected splitting,
#   2. interpret the monochrome object's bbox as an irregular binary mask.
#
# b) Complexity calibration
# - The control family is deliberately small and structured: only chunky L / T shapes.
#   This keeps the non-rectangular cue obvious to humans and avoids noisy random
#   polyominoes that would be hard to generalize from 2-4 examples.
# - The tile remains a compact square with exactly 2-3 colors and no bg holes, so it
#   stays visually recognizable as the repeated stamp unit.
# - We force diagonal touching in every generated sample; otherwise D16/0 would be only
#   "allowed" rather than observable, and the combo could regress toward the base task.
#
# c) Implicit-assumption audit of the original main()
# - `get_bg(g)`: still valid if bg is most common. The generator keeps a 12-18 canvas
#   and small objects, so bg dominates.
# - `find_components(... connectivity=8, monochromatic=False)` returning exactly 2:
#   invalid now, because diagonal contact merges the objects under 8-connectivity.
# - Sorting those 8-components into block vs control by color count therefore also fails.
# - `ctrl_cols[0]` as the only control color is still valid because the control remains
#   monochrome. But `mask = (ctrl == ctrl_col)` is better expressed as `ctrl != bg`,
#   since the real invariant is the cropped bbox mask of the irregular sprite.
# - New solver assumptions introduced here:
#   1. foreground splits into exactly 2 multicolor components under 4-connectivity;
#   2. one is monochrome and non-rectangular inside its bbox;
#   3. the other is a multicolor square tile.
#   The generator enforces and re-verifies each point.
#
# d) Placement legality
# - Objects are first placed in a roomy local canvas and tested exhaustively.
# - A candidate placement is accepted only if:
#   1. no foreground overlap occurs,
#   2. foreground has exactly 2 components under 4-connectivity,
#   3. foreground has exactly 1 component under 8-connectivity.
#   This is the direct formalization of "4-separated but diagonally touching".
# - The accepted local pair is tight-cropped and then embedded into a 12-18 grid
#   without clipping.
#
# e) Visual observability
# - The diagonal-touch effect is visible because the two objects meet at a corner with
#   no gap. The irregular-mask effect is visible because the control bbox contains large
#   bg regions outside the L/T shape.
# - No hidden color-id ordering is used. Colors only separate "monochrome control" from
#   "multicolor tile", exactly as in the base task.
#
# f) Visual cleanliness
# - Only two semantic objects appear in the input; there are no distractors or overlaps.
# - The tile never sits inside the control bbox. It stays in a diagonal quadrant outside
#   the control's bbox, so the relationship reads as external corner contact rather than
#   nested clutter.
#
# g) Change inferability
# - The output rule itself is unchanged and strongly structured: repeat one square tile on
#   the support of one bbox mask.
# - The changed inference burden is local and human-visible:
#   "use the irregular monochrome bbox mask" + "separate by 4-connectivity because the
#   objects only touch at corners".
# - The generator keeps tile colors to 2-3 and control shapes to a tiny family so the
#   combo remains learnable from a few examples.
#
# 9) New solver assumptions checked against the generator
# - `main()` assumes the 4-connected split yields exactly one monochrome object and one
#   multicolor square. `_gen_attempt()` rejects any sample violating that.
# - `main()` assumes the monochrome crop is truly non-rectangular, i.e. its bbox contains
#   bg cells. `_make_ctrl()` and `_variant_metrics()` guarantee that.
# - `main()` assumes diagonal touching really happens. `_place_diagonal_pair()` enforces
#   4-count = 2 and 8-count = 1, and the validation script later measures that statistic.


def _bbox_of_fg(grid: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid != int(bg))
    if coords.size == 0:
        raise ValueError("foreground is empty")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _count_diag_contacts(mask_a: np.ndarray, mask_b: np.ndarray) -> int:
    h, w = mask_a.shape
    total = 0
    for r in range(h):
        for c in range(w):
            if not mask_a[r, c]:
                continue
            for dr, dc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and mask_b[rr, cc]:
                    total += 1
    return int(total)


def _extract_parts(grid: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    if len(comps) != 2:
        raise ValueError("expected exactly 2 foreground components under 4-connectivity")

    infos = []
    for comp in comps:
        patch = crop_bbox(grid, comp["bbox"])
        cols = colors_present(patch, exclude={bg})
        infos.append(
            {
                "patch": patch,
                "cols": cols,
                "area": int(comp["area"]),
                "square": int(patch.shape[0] == patch.shape[1]),
                "mono": int(len(cols) == 1),
            }
        )

    ctrl_candidates = [info for info in infos if info["mono"] and np.any(info["patch"] == int(bg))]
    block_candidates = [info for info in infos if len(info["cols"]) >= 2 and info["square"]]
    if len(ctrl_candidates) != 1 or len(block_candidates) != 1:
        raise ValueError("failed to distinguish control and tile")

    ctrl = np.array(ctrl_candidates[0]["patch"], dtype=int)
    block = np.array(block_candidates[0]["patch"], dtype=int)
    return ctrl, block


def main(grid: np.ndarray) -> np.ndarray:
    """
    Solve the irregular-control + diagonal-touch variant of bf45cf4b.

    The foreground must be split by 4-connectivity because the two objects can be
    diagonally adjacent. The monochrome non-rectangular object contributes a bbox
    mask (`ctrl != bg`), and each filled mask cell expands into one full copy of
    the multicolor square tile.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    ctrl, block = _extract_parts(g, bg)
    if not np.any(ctrl == int(bg)):
        raise ValueError("control must be non-rectangular inside its bbox")
    if block.shape[0] != block.shape[1]:
        raise ValueError("tile must be square")

    bh, bw = map(int, block.shape)
    ch, cw = map(int, ctrl.shape)
    mask = (ctrl != int(bg))

    out = np.full((ch * bh, cw * bw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if bool(mask[r, c]):
                paste(out, block, r * bh, c * bw)
    return out


def _make_tile(tile_colors: list[int], rng: np.random.Generator) -> np.ndarray:
    size = int(rng.integers(2, 5))
    n_colors = int(rng.integers(2, min(4, len(tile_colors) + 1)))
    palette = [int(x) for x in rng.choice(tile_colors, size=n_colors, replace=False)]

    for _ in range(200):
        tile = np.array(rng.choice(palette, size=(size, size), replace=True), dtype=int)
        if len(colors_present(tile)) < 2:
            continue
        return tile
    raise RuntimeError("failed to build multicolor tile")


def _make_l_mask(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    thick = int(rng.integers(1, min(h, w) // 2 + 1))
    mask = np.zeros((h, w), dtype=bool)
    mask[:, :thick] = True
    mask[h - thick :, :] = True
    return np.rot90(mask, int(rng.integers(0, 4))).copy()


def _make_t_mask(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    bar = int(rng.integers(1, min(3, h)))
    stem = int(rng.integers(1, min(3, w)))
    stem = min(stem, w - 1)
    c0 = int(rng.integers(0, w - stem + 1))
    mask = np.zeros((h, w), dtype=bool)
    mask[:bar, :] = True
    mask[:, c0 : c0 + stem] = True
    return np.rot90(mask, int(rng.integers(0, 4))).copy()


def _make_ctrl(bg: int, color: int, rng: np.random.Generator) -> np.ndarray:
    for _ in range(200):
        family = "L" if rng.random() < 0.55 else "T"
        if family == "L":
            h = int(rng.integers(4, 7))
            w = int(rng.integers(4, 7))
            mask = _make_l_mask(h, w, rng)
        else:
            h = int(rng.integers(5, 7))
            w = int(rng.integers(5, 7))
            mask = _make_t_mask(h, w, rng)

        area = int(mask.sum())
        ratio = area / float(h * w)
        if not (0.28 <= ratio <= 0.75):
            continue

        ctrl = np.full(mask.shape, int(bg), dtype=int)
        ctrl[mask] = int(color)
        return ctrl
    raise RuntimeError("failed to build irregular control")


def _compose_output(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    th, tw = tile.shape
    ch, cw = ctrl.shape
    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if ctrl[r, c] != int(bg):
                paste(out, tile, r * th, c * tw)
    return out


def _place_diagonal_pair(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray | None:
    th, tw = tile.shape
    ch, cw = ctrl.shape
    pad = max(th, tw, ch, cw) + 2
    H = ch + th + 2 * pad
    W = cw + tw + 2 * pad
    cr = pad
    cc = pad

    candidates: list[tuple[int, int, int, int, int]] = []
    for tr in range(H - th + 1):
        for tc in range(W - tw + 1):
            above = tr + th - 1 < cr
            below = tr > cr + ch - 1
            left = tc + tw - 1 < cc
            right = tc > cc + cw - 1
            if not ((above or below) and (left or right)):
                continue

            local = np.full((H, W), int(bg), dtype=int)
            paste(local, ctrl, cr, cc, transparent=bg)
            paste(local, tile, tr, tc)

            comps4 = find_components(local, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
            if len(comps4) != 2:
                continue
            comps8 = find_components(local, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
            if len(comps8) != 1:
                continue

            comp_grids = find_components(local, bg=bg, connectivity=4, monochromatic=False, return_format="objects")
            if len(comp_grids) != 2:
                continue
            masks = [obj != int(bg) for obj in comp_grids]
            diag_contacts = _count_diag_contacts(masks[0], masks[1])
            if diag_contacts <= 0:
                continue

            candidates.append((diag_contacts, abs(tr - cr), abs(tc - cc), tr, tc))

    if not candidates:
        return None

    candidates.sort()
    _diag_contacts, _dr, _dc, tr, tc = candidates[0]
    local = np.full((H, W), int(bg), dtype=int)
    paste(local, ctrl, cr, cc, transparent=bg)
    paste(local, tile, tr, tc)
    return crop_bbox(local, _bbox_of_fg(local, bg))


def _make_input(tile: np.ndarray, ctrl: np.ndarray, bg: int, rng: np.random.Generator) -> np.ndarray | None:
    pair = _place_diagonal_pair(tile, ctrl, bg)
    if pair is None:
        return None

    ph, pw = pair.shape
    min_h = max(12, ph + 1)
    min_w = max(12, pw + 1)
    if min_h > 18 or min_w > 18:
        return None

    H = int(rng.integers(min_h, 19))
    W = int(rng.integers(min_w, 19))
    grid = np.full((H, W), int(bg), dtype=int)
    r0 = int(rng.integers(0, H - ph + 1))
    c0 = int(rng.integers(0, W - pw + 1))
    paste(grid, pair, r0, c0, transparent=bg)
    return grid


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    bg = int(get_bg(grid))
    comps4 = find_components(grid, bg=bg, connectivity=4, monochromatic=False, return_format="objects")
    comps8 = find_components(grid, bg=bg, connectivity=8, monochromatic=False, return_format="objects")

    ctrl_patch = None
    for obj in comps4:
        patch = crop_bbox(obj, _bbox_of_fg(obj, bg))
        cols = colors_present(patch, exclude={bg})
        if len(cols) == 1:
            ctrl_patch = patch
            break
    if ctrl_patch is None:
        raise ValueError("failed to recover control for metrics")

    diag_contacts = 0
    if len(comps4) == 2:
        masks = [obj != int(bg) for obj in comps4]
        diag_contacts = _count_diag_contacts(masks[0], masks[1])

    return {
        "comp4": int(len(comps4)),
        "comp8": int(len(comps8)),
        "ctrl_has_bg_in_bbox": int(np.any(ctrl_patch == int(bg))),
        "diag_contacts": int(diag_contacts),
        "ctrl_h": int(ctrl_patch.shape[0]),
        "ctrl_w": int(ctrl_patch.shape[1]),
    }


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    palette = [int(x) for x in rng.choice(np.arange(10, dtype=int), size=5, replace=False)]
    bg = int(palette[0])
    ctrl_color = int(palette[1])
    tile_colors = [int(x) for x in palette[2:]]

    tile = _make_tile(tile_colors, rng)
    ctrl = _make_ctrl(bg, ctrl_color, rng)
    inp = _make_input(tile, ctrl, bg, rng)
    if inp is None:
        return None

    expected = _compose_output(tile, ctrl, bg)
    metrics = _variant_metrics(inp)
    if metrics["comp4"] != 2 or metrics["comp8"] != 1:
        return None
    if metrics["ctrl_has_bg_in_bbox"] != 1 or metrics["diag_contacts"] <= 0:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid combo input")
