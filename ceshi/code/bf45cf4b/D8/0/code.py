# Variant of bf45cf4b
# Dimension: D8 — 输入中对象的数量与布局 (src)
# Change: 恰好2个前景连通分量（8-连通），在网格中随机放置，8-连通不相邻（至少1像素间隔）... → 3个前景对象：1个控制精灵 + 2个候选瓦片，需选择正确瓦片

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, crop_bbox, find_components, get_bg, paste, random_free_location

"""
Design analysis for the D8 variant

a) Concrete behavior and trace
   Input now contains three disconnected 8-connected foreground objects:
   - one monochrome control sprite with bg holes inside its rectangular bbox
   - two polychrome square candidate tiles
   Output still equals "control sprite expanded by stamping one tile per filled control cell",
   but only one candidate tile is valid. The valid tile is the one whose side length divides
   both control height and control width. Example:
   - control bbox = 6x4, candidates = 3x3 and 4x4
   - original task: there was only one tile, so solver directly stamped it
   - new task: 4 does not divide 6, while 3 does not divide 4, so this pair would be invalid
     and generator must reject it. A valid example is control 6x6 with candidates 3x3 and 4x4:
     3 divides both dimensions, 4 does not, so 3x3 is stamped.
   The transformation step after tile selection is unchanged.

b) Complexity calibration
   The extra object must create a real decision, not noise. Both candidates therefore stay
   "multicolor square tiles", and generator gives them the same foreground color-count so that
   "pick the more colorful object" is not a reliable cue. Tile sizes are limited to 3..5 and
   control sizes to 3..8 so the divisibility relation is visible and physically fits cleanly.

c) Implicit assumptions broken from the original main()
   - `len(comps) == 2` no longer holds; it must become exactly 3.
   - sorting by "multicolor + square" is no longer enough, because two objects satisfy it.
   - assuming `infos[1]` is the control object becomes unsafe.
   The new solver must explicitly classify:
   - exactly one monochrome component -> control
   - exactly two multicolor square components -> candidate tiles
   - exactly one candidate size divides both control dimensions -> chosen block

d) Placement legality
   Generator uses `random_free_location(..., border=1, padding=1)` so the three objects:
   - do not overlap
   - stay inside bounds
   - remain separated under 8-connectivity with at least one bg gap
   Transparent pasting preserves holes inside the control sprite and optional bg inside tiles.

e) Visual observability
   Across examples, the chosen tile is not fixed by position, not fixed by color-count, and not
   fixed by being larger/smaller. What stays invariant is that the output dimensions equal
   `(control_h * chosen_tile_side, control_w * chosen_tile_side)`, and only the tile whose side
   evenly partitions the control bbox can explain that scaling.

f) Visual cleanliness
   Objects are placed with a border and a one-pixel separation, so the control sprite and both
   candidates remain easy to inspect. No extra markers or overlays are added.

g) Color inference
   No new color mapping is introduced. The output copies the chosen tile verbatim, exactly as in
   the source task. Colors stay local to the chosen tile pattern, so humans only need to infer
   the tile-selection rule.

9) New solver assumptions and generator guarantees
   Solver assumptions:
   - exactly 3 foreground components
   - exactly 1 monochrome control component
   - exactly 2 multicolor square candidate tiles
   - exactly 1 candidate tile whose side divides both control dimensions
   Generator guarantees each of these explicitly before accepting a sample.
"""

TILE_SIZES = (3, 4, 5)
CTRL_DIM_RANGE = range(3, 9)


def _component_infos(grid: np.ndarray, bg: int) -> list[dict]:
    comps = find_components(grid, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    infos = []
    for comp in comps:
        patch = crop_bbox(grid, comp["bbox"])
        cols = colors_present(patch, exclude={bg})
        infos.append(
            {
                "bbox": comp["bbox"],
                "patch": patch,
                "cols": cols,
                "shape": tuple(int(x) for x in patch.shape),
            }
        )
    return infos


def _classify_objects(grid: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    infos = _component_infos(grid, bg)
    assert len(infos) == 3

    ctrl_candidates = [info for info in infos if len(info["cols"]) == 1]
    tile_candidates = [
        info
        for info in infos
        if len(info["cols"]) >= 2 and info["shape"][0] == info["shape"][1]
    ]
    assert len(ctrl_candidates) == 1
    assert len(tile_candidates) == 2

    ctrl = ctrl_candidates[0]["patch"]
    ch, cw = ctrl.shape
    valid_tiles = [
        info
        for info in tile_candidates
        if ch % info["shape"][0] == 0 and cw % info["shape"][1] == 0
    ]
    assert len(valid_tiles) == 1

    chosen = valid_tiles[0]["patch"]
    others = [info["patch"] for info in tile_candidates if info is not valid_tiles[0]]
    return ctrl, chosen, others


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule: identify the single monochrome control sprite, then choose from the two
    multicolor square candidates the unique tile whose side length divides both control dimensions.
    Stamp that tile at every filled control cell and leave bg-sized holes elsewhere.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    ctrl, block, _others = _classify_objects(g, bg)
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


def _make_tile(bg: int, fg_colors: list[int], size: int, n_tile_colors: int, rng: np.random.Generator) -> np.ndarray:
    tile_colors = [int(x) for x in rng.choice(fg_colors, size=n_tile_colors, replace=False)]
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


def _make_ctrl(bg: int, color: int, h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    area = h * w
    min_fill = max(h + w - 1, int(np.ceil(0.28 * area)))
    max_fill = min(area - 1, int(np.floor(0.72 * area)))
    assert min_fill <= max_fill
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


def _compose_output(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    th, tw = tile.shape
    ch, cw = ctrl.shape
    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if ctrl[r, c] != bg:
                paste(out, tile, r * th, c * tw)
    return out


def _divides_ctrl(side: int, h: int, w: int) -> bool:
    return h % side == 0 and w % side == 0


def _valid_ctrl_shapes(correct_size: int, distractor_size: int) -> list[tuple[int, int]]:
    return [
        (h, w)
        for h in CTRL_DIM_RANGE
        for w in CTRL_DIM_RANGE
        if _divides_ctrl(correct_size, h, w) and not _divides_ctrl(distractor_size, h, w)
    ]


def _make_input(
    correct_tile: np.ndarray,
    distractor_tile: np.ndarray,
    ctrl: np.ndarray,
    bg: int,
    rng: np.random.Generator,
) -> np.ndarray:
    objects = [correct_tile, distractor_tile, ctrl]
    max_h = max(int(obj.shape[0]) for obj in objects)
    max_w = max(int(obj.shape[1]) for obj in objects)
    h_min = max(12, max_h + 2)
    w_min = max(12, max_w + 2)

    for _ in range(80):
        H = int(rng.integers(h_min, 19))
        W = int(rng.integers(w_min, 19))
        grid = np.full((H, W), int(bg), dtype=int)
        order = list(range(3))
        rng.shuffle(order)
        ok = True
        for idx in order:
            patch = objects[idx]
            try:
                pos = random_free_location(
                    grid,
                    patch,
                    transparent=bg,
                    border=1,
                    padding=1,
                    padding_connectivity=8,
                )
            except ValueError:
                ok = False
                break
            paste(grid, patch, pos[0], pos[1], transparent=bg)
        if ok:
            return grid
    raise RuntimeError("placement failed")


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    palette = [int(x) for x in rng.choice(np.arange(10, dtype=int), size=6, replace=False)]
    bg = int(palette[0])
    ctrl_color = int(palette[1])
    tile_palette = [int(x) for x in palette[2:]]

    correct_size = int(rng.choice(TILE_SIZES))
    distractor_options = [s for s in TILE_SIZES if s != correct_size and _valid_ctrl_shapes(correct_size, s)]
    distractor_size = int(rng.choice(distractor_options))
    ctrl_shapes = _valid_ctrl_shapes(correct_size, distractor_size)
    ch, cw = ctrl_shapes[int(rng.integers(len(ctrl_shapes)))]

    n_tile_colors = int(rng.integers(2, min(4, len(tile_palette)) + 1))
    correct_tile = _make_tile(bg, tile_palette, correct_size, n_tile_colors, rng)
    distractor_tile = _make_tile(bg, tile_palette, distractor_size, n_tile_colors, rng)
    ctrl = _make_ctrl(bg, ctrl_color, ch, cw, rng)
    inp = _make_input(correct_tile, distractor_tile, ctrl, bg, rng)
    expected = _compose_output(correct_tile, ctrl, bg)
    wrong = _compose_output(distractor_tile, ctrl, bg)

    infos = _component_infos(inp, bg)
    if len(infos) != 3:
        return None
    mono = [info for info in infos if len(info["cols"]) == 1]
    multi_square = [
        info for info in infos if len(info["cols"]) >= 2 and info["shape"][0] == info["shape"][1]
    ]
    if len(mono) != 1 or len(multi_square) != 2:
        return None

    ctrl_patch = mono[0]["patch"]
    h, w = ctrl_patch.shape
    divisor_flags = [h % info["shape"][0] == 0 and w % info["shape"][1] == 0 for info in multi_square]
    if sorted(divisor_flags) != [False, True]:
        return None

    if np.array_equal(expected, wrong):
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
