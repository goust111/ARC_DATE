# Combo variant of bf45cf4b (index 3)
# combo_spec: D2/0+D4/0
# Name: 反转角色多色精灵编码
#
# Design analysis
# - Concrete behavior: in the original task, a monochrome control mask selects where a multicolor
#   tile is copied. Here the roles are inverted. Example:
#     tile (monochrome):
#       440
#       444
#       404
#     control (multicolor):
#       223
#       032
#   Original logic would have treated the 2-color object as the tile and the 1-color object as
#   the controller; that assumption is now wrong. New logic treats the monochrome patch as the tile.
#   Control color 2 stamps the tile as-is; control color 3 stamps its horizontal mirror:
#       044
#       444
#       404
#   Background cells in the controller still expand to background-only tile slots.
# - Complexity calibration: the controller uses exactly two action colors, not an open-ended color
#   mapping. That keeps the task learnable while still adding a real extra dimension over the seed.
#   The generator forces both action colors to appear multiple times and forces the tile to be
#   horizontally asymmetric, so the mirror operation is visible instead of collapsing to the seed.
# - Broken original assumptions:
#   1. "More colors => tile" no longer holds; `main()` must explicitly invert the role test.
#   2. The controller is no longer single-color, so `mask = (ctrl == ctrl_col)` is invalid.
#   3. Pasting only the original tile is invalid; one controller color now selects `np.fliplr(tile)`.
#   4. The original square/polychrome tie-break logic is unnecessary and would be misleading here.
# - Placement legality: the generator keeps exactly two 8-connected foreground components, places
#   them with a gap of at least 2 background cells, and keeps both inside a 12-18 by 12-18 grid.
# - Visual observability / cleanliness: controller colors are fixed across samples
#   (`CTRL_ORIG_COLOR=2`, `CTRL_FLIP_COLOR=3`) so the color->action rule is human-visible rather
#   than relying on hidden palette ordering. The tile color is distinct from both controller colors.
#   Objects never overlap, and both objects are tight in their own bounding boxes.
# - New solver assumptions and generator guarantees:
#   1. Exactly one component has 1 foreground color => tile template.
#   2. Exactly one component has 2 foreground colors => controller.
#   3. The controller colors are {2,3} in generated data; `main()` has a sorted-color fallback so
#      it remains well-defined even if that convention is violated.
#   4. The generator guarantees `tile != np.fliplr(tile)` and at least one flip-coded controller
#      cell, so the fusion behavior cannot silently degenerate back to the original task.

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, crop_bbox, find_components, get_bg, paste

CTRL_ORIG_COLOR = 2
CTRL_FLIP_COLOR = 3


def _component_infos(grid: np.ndarray, bg: int) -> list[dict]:
    comps = find_components(grid, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    assert len(comps) == 2

    infos = []
    for comp in comps:
        patch = crop_bbox(grid, comp["bbox"])
        cols = colors_present(patch, exclude={bg})
        infos.append({"bbox": comp["bbox"], "patch": patch, "cols": cols})
    return infos


def _split_roles(grid: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
    infos = _component_infos(grid, bg)
    mono = [info for info in infos if len(info["cols"]) == 1]
    multi = [info for info in infos if len(info["cols"]) >= 2]
    assert len(mono) == 1 and len(multi) == 1
    return mono[0]["patch"], multi[0]["patch"]


def _controller_colors(ctrl: np.ndarray, bg: int) -> tuple[int, int]:
    ctrl_cols = [int(x) for x in colors_present(ctrl, exclude={bg})]
    assert len(ctrl_cols) == 2
    if CTRL_ORIG_COLOR in ctrl_cols and CTRL_FLIP_COLOR in ctrl_cols:
        return CTRL_ORIG_COLOR, CTRL_FLIP_COLOR
    return int(ctrl_cols[0]), int(ctrl_cols[1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Inverted-role variant:
    - monochrome component -> tile template
    - 2-color component -> controller
    - controller color A stamps the tile as-is
    - controller color B stamps the tile horizontally mirrored
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    tile, ctrl = _split_roles(g, bg)
    orig_color, flip_color = _controller_colors(ctrl, bg)
    tile_flip = np.fliplr(tile)

    th, tw = map(int, tile.shape)
    ch, cw = map(int, ctrl.shape)
    out = np.full((ch * th, cw * tw), int(bg), dtype=int)

    for r in range(ch):
        for c in range(cw):
            cell = int(ctrl[r, c])
            if cell == orig_color:
                paste(out, tile, r * th, c * tw)
            elif cell == flip_color:
                paste(out, tile_flip, r * th, c * tw)
    return out


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


def _grow_connected_mask(h: int, w: int, target: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    start = (int(rng.integers(0, h)), int(rng.integers(0, w)))
    mask[start] = True

    while int(mask.sum()) < target:
        frontier = []
        for r in range(h):
            for c in range(w):
                if mask[r, c]:
                    continue
                if any(mask[rr, cc] for rr, cc in _neighbors8(h, w, r, c)):
                    frontier.append((r, c))
        pick = frontier[int(rng.integers(len(frontier)))]
        mask[pick] = True
    return mask


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _make_tile(bg: int, color: int, rng: np.random.Generator) -> np.ndarray:
    for _ in range(400):
        size = int(rng.integers(3, 6))
        area = size * size
        target = int(rng.integers(max(5, int(np.ceil(0.45 * area))), int(np.floor(0.75 * area)) + 1))
        mask = _grow_connected_mask(size, size, target, rng)
        if not _touches_all_sides(mask):
            continue
        if mask.all():
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if int(np.count_nonzero(mask != np.fliplr(mask))) < 3:
            continue

        tile = np.full((size, size), int(bg), dtype=int)
        tile[mask] = int(color)
        return tile
    raise RuntimeError("failed to build asymmetric monochrome tile")


def _assign_control_colors(mask: np.ndarray) -> np.ndarray | None:
    ctrl = np.zeros(mask.shape, dtype=int)
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    for r, c in coords:
        ctrl[r, c] = CTRL_ORIG_COLOR if (r + c) % 2 == 0 else CTRL_FLIP_COLOR

    n_orig = int(np.count_nonzero(ctrl == CTRL_ORIG_COLOR))
    n_flip = int(np.count_nonzero(ctrl == CTRL_FLIP_COLOR))
    if min(n_orig, n_flip) >= 2:
        return ctrl

    ctrl.fill(0)
    coords.sort()
    for i, (r, c) in enumerate(coords):
        ctrl[r, c] = CTRL_ORIG_COLOR if i % 2 == 0 else CTRL_FLIP_COLOR

    n_orig = int(np.count_nonzero(ctrl == CTRL_ORIG_COLOR))
    n_flip = int(np.count_nonzero(ctrl == CTRL_FLIP_COLOR))
    if min(n_orig, n_flip) >= 2:
        return ctrl
    return None


def _make_ctrl(bg: int, rng: np.random.Generator) -> np.ndarray:
    for _ in range(400):
        h = int(rng.integers(3, 7))
        w = int(rng.integers(3, 7))
        area = h * w
        target = int(rng.integers(max(5, int(np.ceil(0.45 * area))), int(np.floor(0.8 * area)) + 1))
        mask = _grow_connected_mask(h, w, target, rng)
        if not _touches_all_sides(mask):
            continue
        if mask.all():
            continue

        colors = _assign_control_colors(mask)
        if colors is None:
            continue

        ctrl = np.full((h, w), int(bg), dtype=int)
        ctrl[mask] = colors[mask]

        if int(np.count_nonzero(ctrl == CTRL_ORIG_COLOR)) < 2:
            continue
        if int(np.count_nonzero(ctrl == CTRL_FLIP_COLOR)) < 2:
            continue
        return ctrl
    raise RuntimeError("failed to build multicolor controller")


def _compose_output(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    th, tw = map(int, tile.shape)
    ch, cw = map(int, ctrl.shape)
    tile_flip = np.fliplr(tile)
    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            cell = int(ctrl[r, c])
            if cell == CTRL_ORIG_COLOR:
                paste(out, tile, r * th, c * tw)
            elif cell == CTRL_FLIP_COLOR:
                paste(out, tile_flip, r * th, c * tw)
    return out


def _compose_baseline_output(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    th, tw = map(int, tile.shape)
    ch, cw = map(int, ctrl.shape)
    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if int(ctrl[r, c]) != int(bg):
                paste(out, tile, r * th, c * tw)
    return out


def _make_input(tile: np.ndarray, ctrl: np.ndarray, bg: int, rng: np.random.Generator) -> np.ndarray:
    gap = 2
    horizontal = bool(rng.integers(0, 2))

    if horizontal:
        if rng.random() < 0.5:
            left_obj, right_obj = tile, ctrl
        else:
            left_obj, right_obj = ctrl, tile
        lh, lw = left_obj.shape
        rh, rw = right_obj.shape
        min_h = max(12, max(lh, rh) + 2)
        min_w = max(12, lw + rw + gap + 2)
        H = int(rng.integers(min_h, 19))
        W = int(rng.integers(min_w, 19))
        slack = W - (lw + rw + gap)
        c_left = int(rng.integers(1, slack))
        c_right = int(rng.integers(c_left + lw + gap, W - rw))
        r_left = int(rng.integers(1, H - lh))
        r_right = int(rng.integers(1, H - rh))
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
        min_h = max(12, th + bh + gap + 2)
        min_w = max(12, max(tw, bw) + 2)
        H = int(rng.integers(min_h, 19))
        W = int(rng.integers(min_w, 19))
        slack = H - (th + bh + gap)
        r_top = int(rng.integers(1, slack))
        r_bot = int(rng.integers(r_top + th + gap, H - bh))
        c_top = int(rng.integers(1, W - tw))
        c_bot = int(rng.integers(1, W - bw))
        if top_obj is tile:
            tile_pos = (r_top, c_top)
            ctrl_pos = (r_bot, c_bot)
        else:
            ctrl_pos = (r_top, c_top)
            tile_pos = (r_bot, c_bot)

    grid = np.full((H, W), int(bg), dtype=int)
    paste(grid, tile, *tile_pos, transparent=bg)
    paste(grid, ctrl, *ctrl_pos, transparent=bg)
    return grid


def _fusion_metrics(inp: np.ndarray) -> dict[str, int]:
    bg = int(get_bg(inp))
    tile, ctrl = _split_roles(inp, bg)
    expected = _compose_output(tile, ctrl, bg)
    baseline = _compose_baseline_output(tile, ctrl, bg)
    return {
        "tile_flip_delta": int(np.count_nonzero(tile != np.fliplr(tile))),
        "ctrl_orig_count": int(np.count_nonzero(ctrl == CTRL_ORIG_COLOR)),
        "ctrl_flip_count": int(np.count_nonzero(ctrl == CTRL_FLIP_COLOR)),
        "output_delta_vs_baseline": int(np.count_nonzero(expected != baseline)),
    }


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    bg_choices = [c for c in range(10) if c not in {CTRL_ORIG_COLOR, CTRL_FLIP_COLOR}]
    bg = int(rng.choice(bg_choices))
    tile_color = int(rng.choice([c for c in range(10) if c not in {bg, CTRL_ORIG_COLOR, CTRL_FLIP_COLOR}]))

    tile = _make_tile(bg, tile_color, rng)
    ctrl = _make_ctrl(bg, rng)
    inp = _make_input(tile, ctrl, bg, rng)
    expected = _compose_output(tile, ctrl, bg)

    comps = find_components(inp, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    if len(comps) != 2:
        return None

    color_counts = []
    for comp in comps:
        patch = crop_bbox(inp, comp["bbox"])
        color_counts.append(len(colors_present(patch, exclude={bg})))
    if sorted(color_counts) != [1, 2]:
        return None

    metrics = _fusion_metrics(inp)
    if metrics["tile_flip_delta"] <= 0:
        return None
    if metrics["ctrl_orig_count"] < 2 or metrics["ctrl_flip_count"] < 2:
        return None
    if metrics["output_delta_vs_baseline"] <= 0:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(500):
        inp = _gen_attempt(rng)
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid input")
