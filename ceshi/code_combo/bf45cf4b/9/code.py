# Combo variant of bf45cf4b (index 9)
# combo_spec: D4/2+D7/0
# Name: 环形精灵相框效果

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, crop_bbox, find_components, get_bg, paste

"""
Design analysis for combo D4/2 + D7/0
=====================================

spec_variant:
  input:
    grid: 12-18 rows x 12-18 cols
    objects: exactly 2 disconnected foreground components under 8-connectivity
    control_sprite:
      type: monochrome hollow rectangular frame, tight-cropped
      size: 4-7 rows x 4-7 cols
      color: fixed grey control color so the "grey interior" is visible and sourced
      structure:
        border cells = control color
        strict interior cells = bg
    building_block:
      type: multicolor square tile
      size: 3-5 x 3-5
      colors: 2-3 foreground colors, all distinct from bg and the grey control color
      may_contain_bg_inside: yes, rarely; monochrome interior fill must preserve that silhouette
  transform:
    filled frame cells -> paste original multicolor tile
    interior hole cells -> paste monochrome tile recolored to the frame/control grey
  output:
    tight assembled lattice of size (frame_h * tile_h) x (frame_w * tile_w)

a) Concrete behavior trace
--------------------------
Example:
  tile =
    2 3
    3 2
  ctrl frame (grey=5) =
    5 5 5 5
    5 0 0 5
    5 5 5 5

Original bf45cf4b:
  border cells paste tile, hole cells stay bg.

This combo:
  border cells paste tile,
  hole cells paste
    5 5
    5 5
  because the tile's non-bg silhouette is recolored to the control grey.

Difference:
  D4/2 changes the mask from an arbitrary sparse monochrome sprite to a strict hollow frame.
  D7/0 changes the frame interior from empty output space to visible monochrome tile copies.
  Together the output reads as a colored frame surrounding grey content.

b) Complexity calibration
-------------------------
- The frame is always hollow and tight-cropped, so every bg cell inside the control bbox has one meaning:
  "inside the frame". That removes ambiguity while still being visually distinct from the original task.
- The control color is fixed to grey. This reconciles the combo description ("grey content") with D7/0's
  "use the control color", and gives the solver repeated evidence that the interior uses one special color.
- Tile colors explicitly exclude grey, so the colored border and grey interior remain separable in 2-4 examples.
- Frame sizes start at 4x4, so the interior contains at least 4 hole cells; the new D7 branch is never a tiny corner case.

c) Implicit assumptions audited against main()
----------------------------------------------
- `find_components(..., monochromatic=False)` assumes there are exactly two foreground components.
  Generator guarantee: tile and frame are placed with a bg gap and rejected unless the whole grid has exactly 2 comps.
- Sorting by `(has >=2 colors, n_colors, square)` assumes the tile is the only multicolor object and square-preferred.
  Generator guarantee: frame is monochrome; tile uses 2-3 colors and is square.
- `crop_bbox` plus "fill every bg control cell with mono tile" assumes the control patch is tight-cropped and has
  no exterior padding in its bbox. Generator guarantee: border is fully filled, so bg cells in the cropped control
  patch are exactly the intended interior hole region.
- `mono_block = where(block != bg, ctrl_col, bg)` assumes the control color is not already part of the tile palette,
  otherwise the two output modes could visually blend. Generator guarantee: tile colors exclude grey/control color.

d) Placement legality
---------------------
- Input placement uses a pre-sized blank canvas, one-tile gap minimum, and opaque tile + transparent frame pasting.
- The frame's interior stays bg in the input, so it remains a single connected ring component.
- Attempt generation is rejected unless the placed grid still has exactly two 8-connected foreground components.

e) Visual observability
-----------------------
- Human-visible cue 1: the control object is no longer an arbitrary sprite; it is clearly a hollow rectangle.
- Human-visible cue 2: the output interior is not empty. It is filled by repeated grey versions of the same tile.
- Because grey also appears as the control object's color in the input, the recolor source is inferable instead of arbitrary.

f) Visual cleanliness
---------------------
- The output is a single regular tile lattice with only two fill modes: multicolor border tiles and grey interior tiles.
- No overlaps, markers, or auxiliary symbols are introduced.

g) Color/change inferability
----------------------------
- Only one recolor rule exists: non-bg cells of hole tiles become the frame's grey color.
- The number of active colors stays small: bg, grey control, and 2-3 tile colors.
- The frame/interior partition provides the structural cue for when recoloring happens.

h) Quantitative non-degeneracy metric
-------------------------------------
- The key fused behavior is "hole cells produce visible monochrome tiles instead of blank bg blocks".
- A direct metric is `delta_pixels = count(combo_output != legacy_output)` where `legacy_output`
  is the original bf45cf4b result on the same input.
- For a true combo sample this must be > 0. In this generator it is actually much larger, because
  every interior frame cell contributes a whole monochrome tile silhouette.

i) Audit of new main() assumptions
----------------------------------
- The loop `for r,c in ctrl bbox` now assumes every bg cell inside the cropped control bbox belongs
  to the intentional frame interior, not to arbitrary exterior padding or random holes. Generator
  enforces a tight hollow frame, so that assumption is safe.
- The branch `patch = block if ctrl[r,c]==ctrl_col else mono_block` assumes only two semantic cell
  types exist in the control bbox: border and interior. Generator never creates extra monochrome
  islands or partial-thickness walls that would introduce a third interpretation.
- The recolor path assumes `mono_block` is visible. Generator enforces both a non-empty frame
  interior and a tile with non-bg cells, so the branch cannot collapse to all-bg output.

Post-change solver assumptions that the generator enforces
----------------------------------------------------------
1. Exactly two foreground components exist.
2. One component is a monochrome hollow frame with full border occupancy and bg-only interior.
3. The other component is a multicolor square tile.
4. The control color is grey and excluded from the tile palette.
5. The combo does not degenerate to the original task: the frame interior is non-empty and the monochrome tile has
   at least one non-bg cell, so the combo output differs from the legacy "leave holes blank" behavior.
"""


GREY = 5


def _mono_tile(block: np.ndarray, bg: int, color: int) -> np.ndarray:
    return np.where(block != int(bg), int(color), int(bg)).astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a hollow monochrome frame with a multicolor tile.

    Frame-border cells receive the original tile. Frame-interior hole cells receive a monochrome
    recolor of the tile using the control frame color (grey in generated tasks).
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

    bh, bw = map(int, block.shape)
    ch, cw = map(int, ctrl.shape)

    ctrl_cols = colors_present(ctrl, exclude={bg})
    if not ctrl_cols:
        return np.full((0, 0), int(bg), dtype=int)

    ctrl_col = int(ctrl_cols[0])
    mono_block = _mono_tile(block, bg, ctrl_col)

    out = np.full((ch * bh, cw * bw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            patch = block if int(ctrl[r, c]) == ctrl_col else mono_block
            paste(out, patch, r * bh, c * bw)
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


def _make_tile(bg: int, tile_colors: list[int], rng: np.random.Generator) -> np.ndarray:
    size = int(rng.integers(3, 6))
    n_tile_colors = int(rng.integers(2, min(4, len(tile_colors) + 1)))
    chosen = [int(x) for x in rng.choice(tile_colors, size=n_tile_colors, replace=False)]
    orbits = _dihedral_orbits(size)
    rng.shuffle(orbits)

    tile = np.full((size, size), int(bg), dtype=int)
    assignments: dict[int, int] = {i: color for i, color in enumerate(chosen)}

    interior_idxs = [i for i, (_cells, border) in enumerate(orbits) if not border and i not in assignments]
    allow_bg_orbit = bool(interior_idxs) and bool(rng.random() < 0.2)
    if allow_bg_orbit:
        assignments[int(rng.choice(interior_idxs))] = int(bg)

    for i, (_cells, border) in enumerate(orbits):
        if i in assignments:
            continue
        palette = chosen if border or not allow_bg_orbit else chosen + [int(bg)]
        assignments[i] = int(rng.choice(palette))

    for i, (cells, _border) in enumerate(orbits):
        for r, c in cells:
            tile[r, c] = assignments[i]
    return tile


def _make_frame_ctrl(bg: int, color: int, rng: np.random.Generator) -> np.ndarray:
    h = int(rng.integers(4, 8))
    w = int(rng.integers(4, 8))
    ctrl = np.full((h, w), int(bg), dtype=int)
    ctrl[0, :] = int(color)
    ctrl[-1, :] = int(color)
    ctrl[:, 0] = int(color)
    ctrl[:, -1] = int(color)
    return ctrl


def _is_hollow_frame(ctrl: np.ndarray, bg: int) -> bool:
    h, w = ctrl.shape
    if h < 3 or w < 3:
        return False
    cols = colors_present(ctrl, exclude={bg})
    if len(cols) != 1:
        return False
    col = int(cols[0])
    if not bool((ctrl[0, :] == col).all()):
        return False
    if not bool((ctrl[-1, :] == col).all()):
        return False
    if not bool((ctrl[:, 0] == col).all()):
        return False
    if not bool((ctrl[:, -1] == col).all()):
        return False
    return bool((ctrl[1:-1, 1:-1] == int(bg)).all())


def _compose_output(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    th, tw = tile.shape
    ch, cw = ctrl.shape
    ctrl_colors = colors_present(ctrl, exclude={bg})
    assert len(ctrl_colors) == 1
    ctrl_color = int(ctrl_colors[0])
    mono_block = _mono_tile(tile, bg, ctrl_color)

    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            patch = tile if int(ctrl[r, c]) == ctrl_color else mono_block
            paste(out, patch, r * th, c * tw)
    return out


def _compose_legacy_output(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    th, tw = tile.shape
    ch, cw = ctrl.shape
    ctrl_colors = colors_present(ctrl, exclude={bg})
    assert len(ctrl_colors) == 1
    ctrl_color = int(ctrl_colors[0])

    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if int(ctrl[r, c]) == ctrl_color:
                paste(out, tile, r * th, c * tw)
    return out


def _combo_delta_pixels(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> int:
    """
    Quantify the fused behavior by comparing against the legacy solver's blank-hole output.
    Positive delta means the frame interior is visibly filled by the combo rule.
    """
    combo = _compose_output(tile, ctrl, bg)
    legacy = _compose_legacy_output(tile, ctrl, bg)
    return int(np.sum(combo != legacy))


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
    non_grey = np.array([c for c in range(10) if c != GREY], dtype=int)
    palette = [int(x) for x in rng.choice(non_grey, size=4, replace=False)]
    bg = int(palette[0])
    ctrl_color = int(GREY)
    tile_palette = [int(x) for x in palette[1:]]

    tile = _make_tile(bg, tile_palette, rng)
    ctrl = _make_frame_ctrl(bg, ctrl_color, rng)
    inp = _make_input(tile, ctrl, bg, rng)
    expected = _compose_output(tile, ctrl, bg)
    delta_pixels = _combo_delta_pixels(tile, ctrl, bg)

    comps = find_components(inp, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    if len(comps) != 2:
        return None

    infos = []
    for comp in comps:
        patch = crop_bbox(inp, comp["bbox"])
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
    tile_patch = infos[0]["patch"]
    ctrl_patch = infos[1]["patch"]
    tile_fg_area = int(np.sum(tile_patch != bg))
    hole_tiles = int(np.sum(ctrl_patch == bg))

    if len(infos[0]["cols"]) < 2:
        return None
    if GREY in infos[0]["cols"]:
        return None
    if infos[1]["cols"] != [GREY]:
        return None
    if not _is_hollow_frame(ctrl_patch, bg):
        return None
    if tile_fg_area == 0 or hole_tiles == 0:
        return None
    if delta_pixels <= 0:
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
