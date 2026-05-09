# Combo variant of bf45cf4b (index 4)
# combo_spec: D4/0+D13/0
# Name: 多色精灵颜色映射放大

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, crop_bbox, find_components, get_bg, paste


# Design analysis for the fused variant
# -------------------------------------
# a) Concrete behavior with a small example:
#    - Let the control bbox be:
#        A B .
#        A A B
#      where "." is bg, A is the dominant control color, and B is the minority
#      control color.
#    - Let the input tile T be a square with a distinctive corner-accent color X.
#    - Original bf45cf4b would ignore control color differences and stamp T at
#      every non-bg control pixel:
#        T T .
#        T T T
#    - This fused variant changes two linked parts at once:
#      1. D4/0: the control sprite is no longer monochrome; two control colors
#         encode two different placement semantics.
#      2. D13/0: one of those semantics is "use T after replacing the whole X
#         color class by the minority control color B".
#    - The output therefore becomes:
#        T recolor(T,B) .
#        T T recolor(T,B)
#      So color is not decorative anymore: it selects whether a tile copy stays
#      original or gets recolored.
# b) Complexity calibration:
#    - If A/B roles were assigned by raw color ids, the rule would be invisible to
#      a human. Instead, A is always the more frequent control color and B the less
#      frequent one. That makes the mapping observable across examples.
#    - The generator forces both colors to appear at least twice and with unequal
#      counts, so both tile behaviors repeat and the majority/minority cue is real.
#    - The tile always has a repeated corner-accent color at all four corners, so
#      recoloring changes a coherent, easy-to-see subset instead of a single pixel.
# c) Hidden-assumption audit versus the original solver:
#    - Original assumption: "the control component is monochrome". Broken. The new
#      solver must read exactly two control colors and assign them semantic roles.
#    - Original assumption: "the tile is the component with more colors". Unsafe
#      after D4/0 because both components are multicolor. The new solver identifies
#      the control by a bbox hole: control crops contain bg, tile crops do not.
#    - Original assumption: "all filled control pixels stamp the same tile". Broken.
#      The new solver chooses between original and recolored tile per control color.
#    - New assumption introduced here: the two control colors have different counts.
#      The generator guarantees that, preventing a majority/minority tie.
#    - New assumption introduced here: tile[0,0] belongs to a foreground color that
#      repeats. The generator makes the four corners share that accent color.
# d) Placement legality:
#    - The generator places exactly two 8-connected foreground components with a bg
#      gap of at least one cell, so no overlap or accidental merging can happen.
#    - Grid sizes stay within 12..18 for the input, while output size is bounded by
#      small control/tile sizes to stay comfortably finite.
# e) Visual observability:
#    - A human can identify the tile as the compact full square and the control as
#      the holed rectangle. In the output, some enlarged cells match the input tile
#      exactly while others share the same corner recoloring, revealing the color ->
#      operation mapping.
# f) Visual cleanliness:
#    - No extra markers are added. The two source objects stay separated in input,
#      and the output is a clean tile assembly with only two tile variants.
# g) Color-change inferability:
#    - The recolored subset is structurally anchored at the four corners of each
#      changed tile. Since B appears in several control cells and the same tile part
#      changes every time, the replacement rule is inferable from a few examples.


def _identify_objects(grid: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
    comps = find_components(grid, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    if len(comps) != 2:
        raise ValueError(f"expected 2 foreground components, found {len(comps)}")

    infos = []
    for comp in comps:
        patch = crop_bbox(grid, comp["bbox"])
        fg_cols = colors_present(patch, exclude={bg})
        infos.append(
            {
                "patch": np.array(patch, dtype=int),
                "fg_cols": fg_cols,
                "has_bg_hole": bool(np.any(patch == int(bg))),
                "is_square": bool(patch.shape[0] == patch.shape[1]),
            }
        )

    holey = [info for info in infos if info["has_bg_hole"]]
    solid = [info for info in infos if not info["has_bg_hole"]]
    if len(holey) == 1 and len(solid) == 1:
        ctrl = holey[0]["patch"]
        tile = solid[0]["patch"]
        return tile, ctrl

    infos.sort(
        key=lambda d: (
            d["has_bg_hole"],
            -(len(d["fg_cols"])),
            not d["is_square"],
        )
    )
    ctrl = infos[-1]["patch"]
    tile = infos[0]["patch"]
    return np.array(tile, dtype=int), np.array(ctrl, dtype=int)


def _control_role_colors(ctrl: np.ndarray, bg: int) -> tuple[int, int]:
    fg_coords = np.argwhere(ctrl != int(bg))
    fg_cols = colors_present(ctrl, exclude={bg})
    if len(fg_cols) != 2:
        raise ValueError(f"expected 2 control colors, found {len(fg_cols)}")

    first_pos: dict[int, tuple[int, int]] = {}
    counts: dict[int, int] = {}
    for color in fg_cols:
        coords = np.argwhere(ctrl == int(color))
        counts[int(color)] = int(coords.shape[0])
        first_pos[int(color)] = tuple(int(x) for x in coords[0])

    ordered = sorted(
        (int(color) for color in fg_cols),
        key=lambda color: (-counts[color], first_pos[color]),
    )
    major = int(ordered[0])
    minor = int(ordered[1])
    if counts[major] == counts[minor]:
        raise ValueError("control colors must have unequal counts for majority/minority mapping")
    if fg_coords.size == 0:
        raise ValueError("empty control foreground")
    return major, minor


def _recolored_tile(tile: np.ndarray, new_color: int) -> np.ndarray:
    src_color = int(tile[0, 0])
    out = np.array(tile, copy=True)
    out[out == src_color] = int(new_color)
    return out


def _compose_variant_output(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    th, tw = map(int, tile.shape)
    ch, cw = map(int, ctrl.shape)
    major, minor = _control_role_colors(ctrl, bg)
    recolored = _recolored_tile(tile, minor)

    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            cell = int(ctrl[r, c])
            if cell == int(bg):
                continue
            patch = tile if cell == major else recolored
            paste(out, patch, r * th, c * tw)
    return out


def _compose_original_output(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    th, tw = map(int, tile.shape)
    ch, cw = map(int, ctrl.shape)
    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if int(ctrl[r, c]) != int(bg):
                paste(out, tile, r * th, c * tw)
    return out


def _compose_all_recolored_output(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    _major, minor = _control_role_colors(ctrl, bg)
    recolored = _recolored_tile(tile, minor)
    th, tw = map(int, recolored.shape)
    ch, cw = map(int, ctrl.shape)
    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if int(ctrl[r, c]) != int(bg):
                paste(out, recolored, r * th, c * tw)
    return out


def _behavior_metrics(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> dict[str, int]:
    major, minor = _control_role_colors(ctrl, bg)
    major_count = int(np.count_nonzero(ctrl == int(major)))
    minor_count = int(np.count_nonzero(ctrl == int(minor)))
    corner_count = int(np.count_nonzero(tile == int(tile[0, 0])))
    variant = _compose_variant_output(tile, ctrl, bg)
    original = _compose_original_output(tile, ctrl, bg)
    all_recolored = _compose_all_recolored_output(tile, ctrl, bg)

    th, tw = map(int, tile.shape)
    seen_tiles: set[bytes] = set()
    for r in range(ctrl.shape[0]):
        for c in range(ctrl.shape[1]):
            if int(ctrl[r, c]) == int(bg):
                continue
            patch = variant[r * th : (r + 1) * th, c * tw : (c + 1) * tw]
            seen_tiles.add(patch.tobytes())

    return {
        "major_count": major_count,
        "minor_count": minor_count,
        "corner_count": corner_count,
        "changed_pixels": int(np.count_nonzero(variant != original)),
        "tile_kinds": int(len(seen_tiles)),
        "differs_from_original": int(not np.array_equal(variant, original)),
        "differs_from_all_recolored": int(not np.array_equal(variant, all_recolored)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify the solid square tile and the holed two-color control sprite.
    The dominant control color stamps the original tile.
    The minority control color stamps a recolored tile where the tile's corner
    accent color class is replaced by that minority color.
    Background control cells expand to background regions.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    tile, ctrl = _identify_objects(g, bg)
    return _compose_variant_output(tile, ctrl, bg)


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


def _make_tile(bg: int, tile_colors: list[int], rng: np.random.Generator) -> np.ndarray:
    size = int(rng.integers(4, 6))
    corner_color, base_color, stripe_color = [int(x) for x in rng.choice(tile_colors, size=3, replace=False)]

    tile = np.full((size, size), int(base_color), dtype=int)
    tile[0, 0] = int(corner_color)
    tile[0, -1] = int(corner_color)
    tile[-1, 0] = int(corner_color)
    tile[-1, -1] = int(corner_color)

    mid = size // 2
    tile[mid, :] = int(stripe_color)
    tile[:, mid] = int(stripe_color)

    tile[0, 0] = int(corner_color)
    tile[0, -1] = int(corner_color)
    tile[-1, 0] = int(corner_color)
    tile[-1, -1] = int(corner_color)

    if size == 4:
        tile[1, 1] = int(base_color)
        tile[1, -2] = int(base_color)
        tile[-2, 1] = int(base_color)
        tile[-2, -2] = int(base_color)

    if len(colors_present(tile, exclude={bg})) != 3:
        raise RuntimeError("tile must use exactly three foreground colors")
    if np.any(tile == int(bg)):
        raise RuntimeError("tile must be solid for object disambiguation")
    if np.count_nonzero(tile == int(tile[0, 0])) < 4:
        raise RuntimeError("tile corner accent must repeat")
    return tile


def _grow_mask(h: int, w: int, target: int, rng: np.random.Generator) -> np.ndarray:
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
        rr, cc = frontier[int(rng.integers(0, len(frontier)))]
        mask[rr, cc] = True
    return mask


def _pick_minority_cells(mask: np.ndarray, target: int, rng: np.random.Generator) -> set[tuple[int, int]]:
    filled = [tuple(int(x) for x in coord) for coord in np.argwhere(mask)]
    seed = filled[int(rng.integers(0, len(filled)))]
    chosen = {seed}

    while len(chosen) < target:
        frontier = []
        for r, c in filled:
            if (r, c) in chosen:
                continue
            if any((rr, cc) in chosen for rr, cc in _neighbors8(mask.shape[0], mask.shape[1], r, c)):
                frontier.append((r, c))
        if not frontier:
            return set()
        chosen.add(frontier[int(rng.integers(0, len(frontier)))])
    return chosen


def _make_ctrl(bg: int, major_color: int, minor_color: int, rng: np.random.Generator) -> np.ndarray:
    for _ in range(200):
        h = int(rng.integers(4, 7))
        w = int(rng.integers(4, 7))
        if h == w:
            continue
        area = h * w
        min_fill = max(h + w - 1, 8, int(np.ceil(0.48 * area)))
        max_fill = min(area - 2, int(np.floor(0.75 * area)))
        if min_fill > max_fill:
            continue
        target = int(rng.integers(min_fill, max_fill + 1))
        mask = _grow_mask(h, w, target, rng)

        fill_count = int(mask.sum())
        hole_count = int(mask.size - fill_count)
        if fill_count != target or hole_count < 1:
            continue

        minor_max = min(fill_count // 3, fill_count - 2)
        if minor_max < 2:
            continue
        minor_target = int(rng.integers(2, minor_max + 1))
        minority_cells = _pick_minority_cells(mask, minor_target, rng)
        if len(minority_cells) != minor_target:
            continue

        ctrl = np.full((h, w), int(bg), dtype=int)
        ctrl[mask] = int(major_color)
        for r, c in minority_cells:
            ctrl[r, c] = int(minor_color)

        major_count = int(np.count_nonzero(ctrl == int(major_color)))
        minor_count = int(np.count_nonzero(ctrl == int(minor_color)))
        if major_count <= minor_count or major_count < 4 or minor_count < 2:
            continue
        if len(colors_present(ctrl, exclude={bg})) != 2:
            continue
        return ctrl
    raise RuntimeError("failed to build two-color control sprite")


def _make_input(tile: np.ndarray, ctrl: np.ndarray, bg: int, rng: np.random.Generator) -> np.ndarray:
    horizontal = bool(rng.integers(0, 2))
    gap = int(rng.integers(1, 3))

    if horizontal:
        if rng.random() < 0.5:
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
        if rng.random() < 0.5:
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

    grid = np.full((H, W), int(bg), dtype=int)
    paste(grid, tile, *tile_pos)
    paste(grid, ctrl, *ctrl_pos, transparent=bg)
    return grid


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    palette = [int(x) for x in rng.choice(np.arange(10, dtype=int), size=6, replace=False)]
    bg = int(palette[0])
    major_color = int(palette[1])
    minor_color = int(palette[2])
    tile_colors = [int(x) for x in palette[3:]]

    tile = _make_tile(bg, tile_colors, rng)
    ctrl = _make_ctrl(bg, major_color, minor_color, rng)
    inp = _make_input(tile, ctrl, bg, rng)
    expected = _compose_variant_output(tile, ctrl, bg)

    comps = find_components(inp, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    if len(comps) != 2:
        return None

    patches = [crop_bbox(inp, comp["bbox"]) for comp in comps]
    holey_count = sum(int(np.any(patch == int(bg))) for patch in patches)
    if holey_count != 1:
        return None

    metrics = _behavior_metrics(tile, ctrl, bg)
    if metrics["minor_count"] < 2 or metrics["major_count"] <= metrics["minor_count"]:
        return None
    if metrics["corner_count"] < 4 or metrics["tile_kinds"] != 2:
        return None
    if metrics["changed_pixels"] <= 0 or metrics["differs_from_original"] != 1:
        return None
    if metrics["differs_from_all_recolored"] != 1:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid fused variant input")
