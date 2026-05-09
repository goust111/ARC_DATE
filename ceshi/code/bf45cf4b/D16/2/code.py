# Variant of bf45cf4b
# Dimension: D16 — 输入前景对象的连通性要求 (src)
# Change: 两个前景对象必须在8-连通下完全分离（至少1像素间隔）... → 三个以上前景对象：真实对象 + 干扰分量

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, crop_bbox, find_components, get_bg, paste, random_free_location


# Design analysis for the D16 variant
# ----------------------------------
# source_task:
#   none
#
# concrete_behavior:
#   original:
#     input has exactly 2 foreground 8-components:
#       1. a monochrome control sprite with bg holes inside its bbox
#       2. a multicolor square tile
#   variant:
#     input has 4 or 5 foreground 8-components:
#       1. the real control sprite
#       2. the real multicolor square tile
#       3-5. 2 or 3 tiny distractors, each area 1-2, disconnected from everything
#     output is unchanged: only the real control sprite is expanded using the real tile.
#
# behavior_trace:
#   small example:
#     - suppose component areas are [13(ctrl), 9(tile), 1(dot), 2(domino)].
#     - original code does `assert len(comps) == 2`, so it fails immediately.
#     - variant code first sorts components by area and keeps the 2 largest.
#     - among those 2, the multicolor square is block and the monochrome patch is ctrl.
#     - output assembly then follows the original stamping rule exactly.
#   difference:
#     the only semantic change is "ignore tiny extra components before object ID";
#     the expansion logic itself stays the same.
#
# complexity_calibration:
#   - distractors are limited to exactly 2-3 components and area <= 2 each.
#   - real objects remain clearly larger: ctrl fill area >= 5, tile foreground area >= 8.
#   - this prevents ambiguity while making the variant visually obvious.
#
# implicit_assumption_audit_of_original_main:
#   - `assert not (len(comps) != 2)`: invalid under D16; must be replaced.
#   - sorting all components directly by color-count/squareness is unsafe now:
#       a 1x1 distractor can tie ctrl on color-count and beat it on squareness.
#     therefore filtering by area must happen before role identification.
#   - `ctrl_cols[0]` assumes ctrl is monochrome; that still holds in this variant.
#   - using `ctrl == ctrl_col` is harmless, but `ctrl != bg` better states the real invariant.
#
# placement_legality:
#   - generator places tile, ctrl, and each distractor with `random_free_location`
#     using `padding=1` under 8-connectivity and `border=1`.
#   - this guarantees no overlap, no touching, no accidental merges, and in-bounds placement.
#
# visual_observability:
#   - each input still has the same two large structured objects as the base task.
#   - the extra tiny dots/dominoes are easy for a human to recognize as non-participating junk,
#     because they never resemble a square tile or a holey rectangle and they disappear in output.
#
# visual_cleanliness:
#   - distractors are tiny, sparse, and disconnected, so they do not obscure the real objects.
#   - colors are reused from the existing foreground palette rather than introducing many new hues.
#
# color_inference:
#   - no new color mapping is added.
#   - distractor colors may match ctrl/tile colors, reinforcing that size/connectivity,
#     not color novelty, is what distinguishes meaningful objects from junk.
#
# new_solver_assumptions_after_change:
#   - the 2 largest foreground components are exactly the real tile and real ctrl.
#   - the tile is the only remaining object with >=2 foreground colors and a square bbox.
#   - generator enforces these assumptions by construction and verifies `main(input) == expected`.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant solver: ignore 2-3 tiny disconnected distractors, then apply the original rule.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    assert len(comps) >= 2

    primary = sorted(comps, key=lambda comp: int(comp["area"]), reverse=True)[:2]

    infos = []
    for comp in primary:
        patch = crop_bbox(g, comp["bbox"])
        cols = colors_present(patch, exclude={bg})
        infos.append({"bbox": comp["bbox"], "patch": patch, "cols": cols, "area": int(comp["area"])})

    infos.sort(
        key=lambda d: (
            len(d["cols"]) >= 2,
            len(d["cols"]),
            int(d["patch"].shape[0]) == int(d["patch"].shape[1]),
            d["area"],
        ),
        reverse=True,
    )
    block = infos[0]["patch"]
    ctrl = infos[1]["patch"]

    bh, bw = map(int, block.shape)
    ch, cw = map(int, ctrl.shape)
    mask = (ctrl != int(bg))

    out = np.full((ch * bh, cw * bw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if bool(mask[r, c]):
                paste(out, block, r * bh, c * bw)
    return out


def _dihedral_orbits(size: int) -> list[tuple[list[tuple[int, int]], bool]]:
    seen: set[tuple[int, int]] = set()
    out: list[tuple[list[tuple[int, int]], bool]] = []
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
            border = any(rr in (0, s - 1) or cc in (0, s - 1) for rr, cc in cells)
            out.append((cells, border))
    return out


def _make_tile(bg: int, tile_palette: list[int], rng: np.random.Generator) -> np.ndarray:
    size = int(rng.integers(3, 6))
    n_tile_colors = int(rng.integers(2, min(4, len(tile_palette)) + 1))
    tile_colors = [int(x) for x in rng.choice(tile_palette, size=n_tile_colors, replace=False)]
    orbits = _dihedral_orbits(size)
    rng.shuffle(orbits)

    tile = np.full((size, size), int(bg), dtype=int)
    assignments: dict[int, int] = {}
    for i, color in enumerate(tile_colors):
        assignments[i] = int(color)

    interior_idxs = [i for i, (_cells, border) in enumerate(orbits) if not border and i not in assignments]
    if interior_idxs and rng.random() < 0.25:
        assignments[int(rng.choice(interior_idxs))] = int(bg)

    for i, (_cells, border) in enumerate(orbits):
        if i not in assignments:
            palette = tile_colors if border else tile_colors + ([bg] if interior_idxs else [])
            assignments[i] = int(rng.choice(palette))

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
        frontier = []
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


def _make_distractor(bg: int, palette: list[int], rng: np.random.Generator) -> np.ndarray:
    color = int(rng.choice(palette))
    kind = int(rng.integers(0, 4))
    if kind == 0:
        patch = np.full((1, 1), int(bg), dtype=int)
        patch[0, 0] = color
        return patch
    if kind == 1:
        patch = np.full((1, 2), int(bg), dtype=int)
        patch[0, :] = color
        return patch
    if kind == 2:
        patch = np.full((2, 1), int(bg), dtype=int)
        patch[:, 0] = color
        return patch

    patch = np.full((2, 2), int(bg), dtype=int)
    if rng.random() < 0.5:
        patch[0, 0] = color
        patch[1, 1] = color
    else:
        patch[0, 1] = color
        patch[1, 0] = color
    return patch


def _compose_output(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    th, tw = tile.shape
    ch, cw = ctrl.shape
    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if ctrl[r, c] != bg:
                paste(out, tile, r * th, c * tw)
    return out


def _make_input(
    tile: np.ndarray,
    ctrl: np.ndarray,
    distractors: list[np.ndarray],
    bg: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    min_h = max(12, tile.shape[0] + 2, ctrl.shape[0] + 2)
    min_w = max(12, tile.shape[1] + 2, ctrl.shape[1] + 2)

    for _ in range(60):
        H = int(rng.integers(min_h, 19))
        W = int(rng.integers(min_w, 19))
        grid = np.full((H, W), int(bg), dtype=int)
        try:
            tile_pos = random_free_location(
                grid, tile, transparent=bg, border=1, padding=1, padding_connectivity=8
            )
            paste(grid, tile, *tile_pos)

            ctrl_pos = random_free_location(
                grid, ctrl, transparent=bg, border=1, padding=1, padding_connectivity=8
            )
            paste(grid, ctrl, *ctrl_pos, transparent=bg)

            for dist in distractors:
                dist_pos = random_free_location(
                    grid, dist, transparent=bg, border=1, padding=1, padding_connectivity=8
                )
                paste(grid, dist, *dist_pos, transparent=bg)
        except ValueError:
            continue
        return grid
    return None


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    palette = [int(x) for x in rng.choice(np.arange(10, dtype=int), size=6, replace=False)]
    bg = int(palette[0])
    ctrl_color = int(palette[1])
    tile_palette = [int(x) for x in palette[2:]]
    distractor_palette = [ctrl_color] + tile_palette

    tile = _make_tile(bg, tile_palette, rng)
    ctrl = _make_ctrl(bg, ctrl_color, rng)
    distractors = [_make_distractor(bg, distractor_palette, rng) for _ in range(int(rng.integers(2, 4)))]

    inp = _make_input(tile, ctrl, distractors, bg, rng)
    if inp is None:
        return None

    expected = _compose_output(tile, ctrl, bg)
    if int(get_bg(inp)) != bg:
        return None

    comps = find_components(inp, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    if len(comps) not in (4, 5):
        return None

    areas = sorted((int(comp["area"]) for comp in comps), reverse=True)
    if not (areas[1] > areas[2] and all(area <= 2 for area in areas[2:])):
        return None

    top2 = sorted(comps, key=lambda comp: int(comp["area"]), reverse=True)[:2]
    top2_color_counts = []
    for comp in top2:
        patch = crop_bbox(inp, comp["bbox"])
        top2_color_counts.append(len(colors_present(patch, exclude={bg})))
    if sorted(top2_color_counts) != [1, max(top2_color_counts)] or max(top2_color_counts) < 2:
        return None

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid D16 input")
