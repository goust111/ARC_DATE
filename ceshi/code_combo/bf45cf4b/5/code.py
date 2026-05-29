# Combo variant of bf45cf4b (index 5)
# combo_spec: D6/0+D7/0
# Name: 奇偶镜像满铺灰度空洞

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, crop_bbox, find_components, get_bg, paste

"""
Design analysis for the D6/0 + D7/0 combo variant
=================================================

spec_variant:
  unchanged_input_core:
    objects: exactly 2 disconnected foreground components under 8-connectivity
    control_sprite:
      type: monochrome rectangular bbox with bg holes
      role: binary controller for which tile family to place
    building_block:
      type: multicolor square patch
      role: source tile whose geometry and palette get repeated
  changed_transform:
    old:
      filled control cell -> original tile
      hole control cell -> all-bg empty tile-sized region
    combo:
      every control-bbox cell now emits a tile patch
      filled control cell -> parity-transformed original tile
      hole control cell -> parity-transformed monochrome tile silhouette
      row parity controls vertical mirroring
      column parity controls horizontal mirroring
      therefore four visible states may occur:
        even row, even col -> original
        even row, odd col  -> horizontal mirror
        odd row, even col  -> vertical mirror
        odd row, odd col   -> both mirrors
    output:
      size: (control_h * tile_h) x (control_w * tile_w)
      crop: tight by construction, with no blank tile slots

a) Concrete behavior trace
--------------------------
Example:
  tile =
    1 2 0
    3 1 2
    1 3 2
  ctrl =
    8 0
    0 8

Old task:
  (0,0) filled -> tile
  (0,1) hole   -> blank bg square
  (1,0) hole   -> blank bg square
  (1,1) filled -> tile

Combo task:
  mono(tile,8) =
    8 8 0
    8 8 8
    8 8 8

  (0,0): filled, even/even -> tile
  (0,1): hole,   even/odd  -> fliplr(mono(tile,8))
  (1,0): hole,   odd/even  -> flipud(mono(tile,8))
  (1,1): filled, odd/odd   -> flipud(fliplr(tile))

Differences from the original:
  1. hole positions are no longer blank; they show the tile silhouette in the control color
  2. the repeated motif changes orientation by row/column parity instead of staying identical
  3. D6 and D7 act on the same placement slots, so every tile slot varies by both structure and orientation

b) Complexity calibration
-------------------------
- The mirror rule is constrained to row/col parity only, which gives a small repeated state space humans can infer.
- Four orientation states are enough to make "horizontal vs vertical vs both" observable without adding arbitrary cases.
- The tile generator rejects tiles whose color pattern or monochrome silhouette is symmetric under left/right, up/down,
  or both flips; otherwise D6 would visually collapse.
- The control generator requires both filled cells and holes across all four parity classes, so each branch repeats.

c) Old implicit assumptions audited
-----------------------------------
Original solver lines and the assumptions they encoded:
- `if mask[r, c]: paste(out, block, ...)`
  Assumed only filled control pixels generate visible output. Broken by D7.
- "Use exactly the same `block` every time"
  Assumed no placement-dependent transform. Broken by D6.
- `out = full(..., bg)`
  Safe as a canvas initializer, but no longer describes semantic emptiness of hole slots.

What still remains valid:
- The input grammar still has exactly two disconnected foreground components.
- The multicolor square-preferred component is still the block.
- The monochrome component still supplies the recolor source for hole tiles.

d) Placement legality
---------------------
- Input generation still places the block and control sprite with at least one bg gap on a larger canvas, preserving
  exactly two 8-connected foreground components.
- The control sprite is pasted with `transparent=bg`, so its holes remain true bg in the input and do not create extras.
- The block mask is required to stay connected and touch all four borders of its square bbox, so cropping recovers the
  intended square tile and every mirrored copy has the same shape and dimensions.

e) Visual observability
-----------------------
- The control color is absent from the tile palette, so hole tiles are immediately distinguishable from filled tiles.
- The row/column parity rule is visible from repeated alternation across the lattice, not from hidden metadata.
- Because both the colored tile and its monochrome silhouette are asymmetric in both axes, horizontal and vertical
  flips are all visible in 2-4 examples instead of silently matching the original.

f) Visual cleanliness
---------------------
- Output stays a single rectangular lattice; there are no extra markers, overlays, or stacked objects.
- The only backgrounds inside the output come from the tile's own silhouette holes, which remain sparse and structured.
- Filled-vs-hole and parity changes are expressed through clear repetition, not clutter.

g) Change inferability
----------------------
- Color change is fully structured: every non-bg cell of a hole tile becomes the one control color.
- Orientation change is fully structured: row parity determines vertical flipping, column parity determines horizontal
  flipping. These two cues combine consistently across the grid.
- No branch depends on numeric color ids, object ordering by screen position, or any non-visual fact.

h) New implicit assumptions introduced by the combo
---------------------------------------------------
- `main()` now assumes the block's mirrored forms are meaningfully different; generator enforces four distinct tile
  orientations and four distinct monochrome-silhouette orientations.
- `main()` assumes the control sprite has one unique non-bg color; generator still builds a monochrome control object.
- `main()` applies both parity dimensions independently, so generator guarantees that filled cells and holes each occur
  in all four `(row % 2, col % 2)` classes, ensuring the new branches are exercised and visible.
"""


def _mono_tile(block: np.ndarray, bg: int, ctrl_color: int) -> np.ndarray:
    return np.where(block != int(bg), int(ctrl_color), int(bg)).astype(int)


def _transform_by_parity(patch: np.ndarray, r: int, c: int) -> np.ndarray:
    out = patch
    if int(r) % 2 == 1:
        out = np.flipud(out)
    if int(c) % 2 == 1:
        out = np.fliplr(out)
    return out


def _all_parity_variants(patch: np.ndarray) -> list[np.ndarray]:
    return [
        patch,
        np.fliplr(patch),
        np.flipud(patch),
        np.flipud(np.fliplr(patch)),
    ]


def _variant_count(patch: np.ndarray) -> int:
    return len({tuple(map(int, arr.flatten())) for arr in _all_parity_variants(patch)})


def _compose_output(block: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    bh, bw = map(int, block.shape)
    ch, cw = map(int, ctrl.shape)

    ctrl_cols = colors_present(ctrl, exclude={bg})
    assert len(ctrl_cols) == 1
    ctrl_color = int(ctrl_cols[0])
    mono_block = _mono_tile(block, bg, ctrl_color)

    out = np.full((ch * bh, cw * bw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            base = block if int(ctrl[r, c]) == ctrl_color else mono_block
            paste(out, _transform_by_parity(base, r, c), r * bh, c * bw)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the control sprite into a full tile lattice.

    Filled control cells use the original multicolor block, holes use a monochrome version
    of that block in the control color, and both branches are mirrored by row/column parity.
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
    return _compose_output(block, ctrl, bg)


def _neighbors8(h: int, w: int, r: int, c: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < h and 0 <= cc < w:
                out.append((rr, cc))
    return out


def _mask_is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False

    h, w = mask.shape
    seed = tuple(map(int, coords[0]))
    stack = [seed]
    seen = {seed}

    while stack:
        r, c = stack.pop()
        for rr, cc in _neighbors8(h, w, r, c):
            if not bool(mask[rr, cc]) or (rr, cc) in seen:
                continue
            seen.add((rr, cc))
            stack.append((rr, cc))
    return len(seen) == int(mask.sum())


def _has_full_square_bbox(mask: np.ndarray) -> bool:
    h, w = mask.shape
    return (
        bool(mask[0, :].any())
        and bool(mask[h - 1, :].any())
        and bool(mask[:, 0].any())
        and bool(mask[:, w - 1].any())
    )


def _make_tile(bg: int, tile_colors: list[int], rng: np.random.Generator) -> np.ndarray:
    for _ in range(1000):
        size = int(rng.integers(3, 6))
        n_colors = int(rng.integers(2, min(4, len(tile_colors) + 1)))
        chosen = [int(x) for x in rng.choice(tile_colors, size=n_colors, replace=False)]

        tile = np.full((size, size), int(bg), dtype=int)
        total = size * size
        max_holes = min(4, total - n_colors - 1)
        if max_holes < 1:
            continue
        n_holes = int(rng.integers(1, max_holes + 1))

        hole_indices = [int(x) for x in rng.choice(np.arange(total, dtype=int), size=n_holes, replace=False)]
        mask = np.ones((size, size), dtype=bool)
        for idx in hole_indices:
            mask[int(idx // size), int(idx % size)] = False

        if not _has_full_square_bbox(mask):
            continue
        if not _mask_is_connected(mask):
            continue
        if _variant_count(mask.astype(int)) != 4:
            continue

        fg_cells = [(r, c) for r in range(size) for c in range(size) if bool(mask[r, c])]
        if len(fg_cells) < n_colors:
            continue
        order = [tuple(map(int, cell)) for cell in rng.permutation(np.array(fg_cells))]

        for color, (r, c) in zip(chosen, order):
            tile[r, c] = int(color)
        for r, c in order[n_colors:]:
            tile[r, c] = int(rng.choice(chosen))

        if len(colors_present(tile, exclude={bg})) < 2:
            continue
        if _variant_count(tile) != 4:
            continue
        return tile
    raise RuntimeError("failed to build parity-visible tile")


def _ctrl_parity_ok(mask: np.ndarray) -> bool:
    all_classes = {(0, 0), (0, 1), (1, 0), (1, 1)}
    filled = {(r % 2, c % 2) for r in range(mask.shape[0]) for c in range(mask.shape[1]) if bool(mask[r, c])}
    holes = {(r % 2, c % 2) for r in range(mask.shape[0]) for c in range(mask.shape[1]) if not bool(mask[r, c])}
    return filled == all_classes and holes == all_classes


def _make_ctrl(bg: int, color: int, rng: np.random.Generator) -> np.ndarray:
    for _ in range(800):
        h = int(rng.integers(3, 6))
        w = int(rng.integers(3, 6))
        area = h * w
        min_fill = max(4, int(np.ceil(0.45 * area)))
        max_fill = min(area - 4, int(np.floor(0.68 * area)))
        if min_fill > max_fill:
            continue
        target = int(rng.integers(min_fill, max_fill + 1))

        mask = np.zeros((h, w), dtype=bool)
        seed = (int(rng.integers(0, h)), int(rng.integers(0, w)))
        mask[seed] = True

        while int(mask.sum()) < target:
            frontier = []
            for r in range(h):
                for c in range(w):
                    if bool(mask[r, c]):
                        continue
                    if any(bool(mask[rr, cc]) for rr, cc in _neighbors8(h, w, r, c)):
                        frontier.append((r, c))
            if not frontier:
                break
            pick = frontier[int(rng.integers(len(frontier)))]
            mask[pick] = True

        if int(mask.sum()) != target:
            continue
        if int((~mask).sum()) < 4:
            continue
        if not _ctrl_parity_ok(mask):
            continue

        ctrl = np.full((h, w), int(bg), dtype=int)
        ctrl[mask] = int(color)
        return ctrl
    raise RuntimeError("failed to build control sprite")


def _make_input(block: np.ndarray, ctrl: np.ndarray, bg: int, rng: np.random.Generator) -> np.ndarray:
    horizontal = bool(rng.integers(0, 2))
    gap = 1

    if horizontal:
        if bool(rng.integers(0, 2)):
            left_obj, right_obj = block, ctrl
        else:
            left_obj, right_obj = ctrl, block
        lh, lw = left_obj.shape
        rh, rw = right_obj.shape
        H = int(rng.integers(max(12, max(lh, rh)), 19))
        W = int(rng.integers(max(12, lw + rw + gap), 19))
        slack = W - (lw + rw + gap)
        c_left = int(rng.integers(0, slack + 1))
        c_right = int(rng.integers(c_left + lw + gap, W - rw + 1))
        r_left = int(rng.integers(0, H - lh + 1))
        r_right = int(rng.integers(0, H - rh + 1))
        if left_obj is block:
            block_pos = (r_left, c_left)
            ctrl_pos = (r_right, c_right)
        else:
            ctrl_pos = (r_left, c_left)
            block_pos = (r_right, c_right)
    else:
        if bool(rng.integers(0, 2)):
            top_obj, bot_obj = block, ctrl
        else:
            top_obj, bot_obj = ctrl, block
        th, tw = top_obj.shape
        bh, bw = bot_obj.shape
        H = int(rng.integers(max(12, th + bh + gap), 19))
        W = int(rng.integers(max(12, max(tw, bw)), 19))
        slack = H - (th + bh + gap)
        r_top = int(rng.integers(0, slack + 1))
        r_bot = int(rng.integers(r_top + th + gap, H - bh + 1))
        c_top = int(rng.integers(0, W - tw + 1))
        c_bot = int(rng.integers(0, W - bw + 1))
        if top_obj is block:
            block_pos = (r_top, c_top)
            ctrl_pos = (r_bot, c_bot)
        else:
            ctrl_pos = (r_top, c_top)
            block_pos = (r_bot, c_bot)

    grid = np.full((H, W), int(bg), dtype=int)
    paste(grid, block, *block_pos)
    paste(grid, ctrl, *ctrl_pos, transparent=bg)
    return grid


def _compose_original(block: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    bh, bw = map(int, block.shape)
    ch, cw = ctrl.shape
    ctrl_color = int(colors_present(ctrl, exclude={bg})[0])
    out = np.full((ch * bh, cw * bw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if int(ctrl[r, c]) == ctrl_color:
                paste(out, block, r * bh, c * bw)
    return out


def _compose_d6_only(block: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    bh, bw = map(int, block.shape)
    ch, cw = ctrl.shape
    ctrl_color = int(colors_present(ctrl, exclude={bg})[0])
    out = np.full((ch * bh, cw * bw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if int(ctrl[r, c]) == ctrl_color:
                paste(out, _transform_by_parity(block, r, c), r * bh, c * bw)
    return out


def _compose_d7_only(block: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    bh, bw = map(int, block.shape)
    ch, cw = ctrl.shape
    ctrl_color = int(colors_present(ctrl, exclude={bg})[0])
    mono_block = _mono_tile(block, bg, ctrl_color)
    out = np.full((ch * bh, cw * bw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            patch = block if int(ctrl[r, c]) == ctrl_color else mono_block
            paste(out, patch, r * bh, c * bw)
    return out


def _behavior_metrics(block: np.ndarray, ctrl: np.ndarray, bg: int) -> dict[str, int]:
    combo = _compose_output(block, ctrl, bg)
    original = _compose_original(block, ctrl, bg)
    d6_only = _compose_d6_only(block, ctrl, bg)
    d7_only = _compose_d7_only(block, ctrl, bg)
    ctrl_color = int(colors_present(ctrl, exclude={bg})[0])
    mono_block = _mono_tile(block, bg, ctrl_color)

    filled_classes = {(r % 2, c % 2) for r in range(ctrl.shape[0]) for c in range(ctrl.shape[1]) if int(ctrl[r, c]) == ctrl_color}
    hole_classes = {(r % 2, c % 2) for r in range(ctrl.shape[0]) for c in range(ctrl.shape[1]) if int(ctrl[r, c]) == bg}

    return {
        "delta_vs_original": int(np.count_nonzero(combo != original)),
        "delta_vs_d6_only": int(np.count_nonzero(combo != d6_only)),
        "delta_vs_d7_only": int(np.count_nonzero(combo != d7_only)),
        "tile_variant_count": int(_variant_count(block)),
        "mono_variant_count": int(_variant_count(mono_block)),
        "filled_class_count": int(len(filled_classes)),
        "hole_class_count": int(len(hole_classes)),
    }


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    palette = [int(x) for x in rng.choice(np.arange(10, dtype=int), size=5, replace=False)]
    bg = int(palette[0])
    ctrl_color = int(palette[1])
    tile_palette = [int(x) for x in palette[2:]]

    block = _make_tile(bg, tile_palette, rng)
    ctrl = _make_ctrl(bg, ctrl_color, rng)
    metrics = _behavior_metrics(block, ctrl, bg)
    if metrics["delta_vs_original"] <= 0:
        return None
    if metrics["delta_vs_d6_only"] <= 0:
        return None
    if metrics["delta_vs_d7_only"] <= 0:
        return None
    if metrics["tile_variant_count"] != 4 or metrics["mono_variant_count"] != 4:
        return None
    if metrics["filled_class_count"] != 4 or metrics["hole_class_count"] != 4:
        return None

    inp = _make_input(block, ctrl, bg, rng)
    expected = _compose_output(block, ctrl, bg)

    comps = find_components(inp, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    if len(comps) != 2:
        return None

    colors_per_comp = []
    for comp in comps:
        patch = crop_bbox(inp, comp["bbox"])
        colors_per_comp.append(sorted(set(int(x) for x in patch.flatten()) - {bg}))

    tile_fg_colors = sorted(set(int(x) for x in block.flatten()) - {bg})
    if sorted(len(cols) for cols in colors_per_comp) != [1, len(tile_fg_colors)]:
        return None
    if ctrl_color in tile_fg_colors:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid input")
