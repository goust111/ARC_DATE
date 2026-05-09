# Combo variant of bf45cf4b (index 18)
# combo_spec: D8/1+D11/1
# Name: 多精灵共享瓦片并行放大

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import colors_present, crop_bbox, find_components, get_bg, paste
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[4] / "solvers"))
    from arc_utils import colors_present, crop_bbox, find_components, get_bg, paste

"""
Design analysis for the fused D8/1 + D11/1 variant.

a) Concrete behavior and trace
- Original bf45cf4b has exactly 2 foreground objects: one multicolor square tile and one
  monochrome holey control. Output is a single enlarged sprite.
- This combo changes both source-object multiplicity and source-target cardinality:
  one shared tile plus 2-3 monochrome controls now produce 2-3 enlarged outputs.
- Trace example: if the tile is 2x2, control A starts at input top-left (1, 6), and control B
  starts at (4, 2), then B is 3 rows lower and 4 columns left of A. In the output, enlarged B
  must be 6 rows lower and 8 columns left of enlarged A, because the same relative offset is
  preserved in tile units.
- Difference from the source task: the solver must distribute one tile to many controls, not
  aggregate one control with one tile into one result.

b) Complexity calibration
- The variable that changes is small and repeated: there are always 2 or 3 controls, never an
  arbitrary count.
- Controls keep the original "monochrome rectangle with bg holes" look, so humans still see them
  as masks rather than free-form distractors.
- Output placement keeps scaled relative spacing instead of naive concatenation, making the fusion
  visually distinct from "just list the enlarged sprites".

c) Hidden-assumption audit of the original solver
- `len(comps) == 2` is invalid: we now need 3-4 components.
- Picking one non-tile component as the control is invalid: there are multiple controls.
- Building one output canvas directly from one control bbox is invalid: output must merge several
  independently enlarged patches.
- New solver assumptions introduced here:
  1. exactly one component is a multicolor square tile,
  2. every other component is monochrome,
  3. control ordering is determined by observable top-left positions,
  4. output offsets are input offsets scaled by tile height/width.

d) Placement legality
- Control sprites are first laid out on an abstract group canvas with 1-3 pixel gaps, ensuring
  they remain disconnected and their relative offsets are well-defined.
- The shared tile is then placed outside that whole group with an additional 2-4 pixel gap.
- Generator re-checks component counts and role signatures after placement.

e) Visual observability
- Humans can see one decorative square tile and multiple differently colored mask-like controls.
- Repetition of the exact same tile across all enlarged regions is the cue for the one-to-many
  shared-tile rule.
- Relative placement survives into the output, so the multi-object relation is visible rather than
  hidden in bookkeeping.

f) Visual cleanliness
- No overlaps are allowed, and tile colors are disjoint from control colors.
- The group layout is limited to row / column / L-shape arrangements, which stays readable and
  avoids clutter from arbitrary scattering.

g) Change inferability
- No invisible ordering tricks or color-id semantics are used.
- The only cross-object rule is structural: all controls share the same tile and preserve their
  relative arrangement after scaling.
- With 2-3 controls per sample, the repeated evidence is enough for few-shot ARC inference.

9) New solver assumptions and generator guarantees
- `main()` assumes a unique multicolor square tile candidate and 2-3 monochrome controls.
- `main()` assumes each control expands independently and the final output contains multiple
  disconnected enlarged components at scaled offsets.
- `generate_input()` guarantees those assumptions and additionally rejects samples unless the
  fusion metrics confirm that extra controls materially affect the output.
"""


def _expand_control(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    th, tw = map(int, tile.shape)
    ch, cw = map(int, ctrl.shape)
    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if int(ctrl[r, c]) != int(bg):
                paste(out, tile, r * th, c * tw)
    return out


def _identify_objects(grid: np.ndarray, bg: int) -> tuple[np.ndarray, list[dict]]:
    comps = find_components(grid, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    infos: list[dict] = []
    for comp in comps:
        patch = crop_bbox(grid, comp["bbox"])
        cols = colors_present(patch, exclude={bg})
        infos.append({"bbox": comp["bbox"], "patch": patch, "cols": cols})

    tile_candidates = [
        info
        for info in infos
        if len(info["cols"]) >= 2 and int(info["patch"].shape[0]) == int(info["patch"].shape[1])
    ]
    assert len(tile_candidates) == 1
    tile_candidates.sort(
        key=lambda d: (len(d["cols"]), d["patch"].shape[0] * d["patch"].shape[1]),
        reverse=True,
    )
    tile_info = tile_candidates[0]

    controls = [info for info in infos if info is not tile_info and len(info["cols"]) == 1]
    assert len(infos) == len(controls) + 1
    controls.sort(key=lambda d: (d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3]))
    assert 2 <= len(controls) <= 3
    return tile_info["patch"], controls


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    tile, controls = _identify_objects(g, bg)
    th, tw = map(int, tile.shape)

    base_r = min(int(info["bbox"][0]) for info in controls)
    base_c = min(int(info["bbox"][1]) for info in controls)

    placements: list[tuple[np.ndarray, int, int]] = []
    max_r = 0
    max_c = 0
    for info in controls:
        patch = _expand_control(tile, info["patch"], bg)
        # Preserve visible control-to-control offsets using top-left bbox anchors,
        # then scale those offsets by the shared tile size.
        rr = (int(info["bbox"][0]) - base_r) * th
        cc = (int(info["bbox"][1]) - base_c) * tw
        placements.append((patch, rr, cc))
        max_r = max(max_r, rr + int(patch.shape[0]))
        max_c = max(max_c, cc + int(patch.shape[1]))

    out = np.full((max_r, max_c), int(bg), dtype=int)
    for patch, rr, cc in placements:
        paste(out, patch, rr, cc)
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


def _make_tile(bg: int, fg_colors: list[int], rng: np.random.Generator) -> np.ndarray:
    size = int(rng.integers(3, 6))
    n_tile_colors = int(rng.integers(2, min(4, len(fg_colors)) + 1))
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

    assert int(tile.shape[0]) == int(tile.shape[1])
    assert len(colors_present(tile, exclude={bg})) >= 2
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
    min_fill = max(4, int(np.ceil(0.45 * area)))
    max_fill = max(min_fill, int(np.floor(0.72 * area)))
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
    assert np.any(ctrl == bg)
    return ctrl


def _layout_controls(ctrls: list[np.ndarray], rng: np.random.Generator) -> list[tuple[int, int]]:
    n = len(ctrls)
    order = list(range(n))
    rng.shuffle(order)
    shapes = [tuple(map(int, ctrls[i].shape)) for i in order]

    if n == 2:
        layout = str(rng.choice(["row", "col"]))
    else:
        layout = str(rng.choice(["row", "col", "l"]))

    positions: list[tuple[int, int]]
    if layout == "row":
        positions = []
        cur_c = 0
        for idx, (h, w) in enumerate(shapes):
            positions.append((0, cur_c))
            cur_c += w + int(rng.integers(1, 4 if idx < n - 1 else 2))
    elif layout == "col":
        positions = []
        cur_r = 0
        for idx, (h, w) in enumerate(shapes):
            positions.append((cur_r, 0))
            cur_r += h + int(rng.integers(1, 4 if idx < n - 1 else 2))
    else:
        (h0, w0), (h1, _w1), (_h2, _w2) = shapes
        gap_h = int(rng.integers(1, 4))
        gap_v = int(rng.integers(1, 4))
        positions = [
            (0, 0),
            (0, w0 + gap_h),
            (max(h0, h1) + gap_v, 0),
        ]

    ordered_positions = [(0, 0)] * n
    for pos, ctrl_idx in zip(positions, order):
        ordered_positions[ctrl_idx] = pos

    max_r = max(r + int(ctrl.shape[0]) for ctrl, (r, c) in zip(ctrls, ordered_positions))
    max_c = max(c + int(ctrl.shape[1]) for ctrl, (r, c) in zip(ctrls, ordered_positions))
    if rng.random() < 0.5:
        ordered_positions = [
            (r, max_c - (c + int(ctrl.shape[1])))
            for ctrl, (r, c) in zip(ctrls, ordered_positions)
        ]
    if rng.random() < 0.5:
        ordered_positions = [
            (max_r - (r + int(ctrl.shape[0])), c)
            for ctrl, (r, c) in zip(ctrls, ordered_positions)
        ]
    return ordered_positions


def _compose_multi_output(
    tile: np.ndarray,
    ctrls: list[np.ndarray],
    ctrl_positions: list[tuple[int, int]],
    bg: int,
) -> np.ndarray:
    th, tw = map(int, tile.shape)
    base_r = min(r for r, _c in ctrl_positions)
    base_c = min(c for _r, c in ctrl_positions)

    placements: list[tuple[np.ndarray, int, int]] = []
    max_r = 0
    max_c = 0
    for ctrl, (r, c) in zip(ctrls, ctrl_positions):
        patch = _expand_control(tile, ctrl, bg)
        rr = (int(r) - base_r) * th
        cc = (int(c) - base_c) * tw
        placements.append((patch, rr, cc))
        max_r = max(max_r, rr + int(patch.shape[0]))
        max_c = max(max_c, cc + int(patch.shape[1]))

    out = np.full((max_r, max_c), int(bg), dtype=int)
    for patch, rr, cc in placements:
        paste(out, patch, rr, cc)
    return out


def _make_input(
    tile: np.ndarray,
    ctrls: list[np.ndarray],
    ctrl_positions: list[tuple[int, int]],
    bg: int,
    rng: np.random.Generator,
) -> np.ndarray:
    th, tw = map(int, tile.shape)
    group_h = max(r + int(ctrl.shape[0]) for ctrl, (r, c) in zip(ctrls, ctrl_positions))
    group_w = max(c + int(ctrl.shape[1]) for ctrl, (r, c) in zip(ctrls, ctrl_positions))
    gap = int(rng.integers(2, 5))

    horizontal = bool(rng.integers(0, 2))
    if horizontal:
        tile_left = bool(rng.integers(0, 2))
        min_h = max(group_h, th) + 2
        min_w = group_w + tw + gap + 2
        H = int(rng.integers(min_h, max(min_h, min_h + 5) + 1))
        W = int(rng.integers(min_w, max(min_w, min_w + 6) + 1))

        r_group = int(rng.integers(1, H - group_h + 1))
        r_tile = int(rng.integers(1, H - th + 1))
        if tile_left:
            c_tile = int(rng.integers(1, W - (tw + gap + group_w) + 1))
            c_group = int(rng.integers(c_tile + tw + gap, W - group_w + 1))
        else:
            c_group = int(rng.integers(1, W - (group_w + gap + tw) + 1))
            c_tile = int(rng.integers(c_group + group_w + gap, W - tw + 1))
    else:
        tile_top = bool(rng.integers(0, 2))
        min_h = group_h + th + gap + 2
        min_w = max(group_w, tw) + 2
        H = int(rng.integers(min_h, max(min_h, min_h + 6) + 1))
        W = int(rng.integers(min_w, max(min_w, min_w + 5) + 1))

        c_group = int(rng.integers(1, W - group_w + 1))
        c_tile = int(rng.integers(1, W - tw + 1))
        if tile_top:
            r_tile = int(rng.integers(1, H - (th + gap + group_h) + 1))
            r_group = int(rng.integers(r_tile + th + gap, H - group_h + 1))
        else:
            r_group = int(rng.integers(1, H - (group_h + gap + th) + 1))
            r_tile = int(rng.integers(r_group + group_h + gap, H - th + 1))

    grid = np.full((H, W), int(bg), dtype=int)
    paste(grid, tile, r_tile, c_tile)
    for ctrl, (r, c) in zip(ctrls, ctrl_positions):
        paste(grid, ctrl, r_group + int(r), c_group + int(c), transparent=bg)
    return grid


def _input_signature_ok(inp: np.ndarray, n_ctrl: int, bg: int) -> bool:
    comps = find_components(inp, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    if len(comps) != n_ctrl + 1:
        return False

    square_multicolor = 0
    single_color = 0
    for comp in comps:
        patch = crop_bbox(inp, comp["bbox"])
        cols = colors_present(patch, exclude={bg})
        if len(cols) >= 2 and int(patch.shape[0]) == int(patch.shape[1]):
            square_multicolor += 1
        elif len(cols) == 1:
            single_color += 1
        else:
            return False

    if square_multicolor != 1 or single_color != n_ctrl:
        return False
    return int(get_bg(inp)) == int(bg)


def _fusion_metrics(
    tile: np.ndarray,
    ctrls: list[np.ndarray],
    ctrl_positions: list[tuple[int, int]],
    bg: int,
) -> dict[str, int]:
    combo = _compose_multi_output(tile, ctrls, ctrl_positions, bg)
    components = find_components(
        combo,
        bg=bg,
        connectivity=8,
        monochromatic=False,
        return_format="dict",
    )
    base_r = min(r for r, _c in ctrl_positions)
    base_c = min(c for _r, c in ctrl_positions)
    patch_fg = [int(np.count_nonzero(_expand_control(tile, ctrl, bg) != int(bg))) for ctrl in ctrls]
    scaled_offset_sum = sum(
        (int(r) - base_r) * int(tile.shape[0]) + (int(c) - base_c) * int(tile.shape[1])
        for r, c in ctrl_positions
    )
    return {
        "n_ctrl": int(len(ctrls)),
        "output_components": int(len(components)),
        "extra_fg_pixels": int(np.count_nonzero(combo != int(bg)) - max(patch_fg)),
        "scaled_offset_sum": int(scaled_offset_sum),
    }


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n_ctrl = int(rng.integers(2, 4))
    n_tile_colors = int(rng.integers(2, 4))
    palette_size = 1 + n_ctrl + n_tile_colors
    palette = [int(x) for x in rng.choice(np.arange(10, dtype=int), size=palette_size, replace=False)]
    bg = int(palette[0])
    ctrl_colors = palette[1 : 1 + n_ctrl]
    tile_colors = palette[1 + n_ctrl :]

    tile = _make_tile(bg, tile_colors, rng)
    ctrls = [_make_ctrl(bg, color, rng) for color in ctrl_colors]
    ctrl_positions = _layout_controls(ctrls, rng)
    inp = _make_input(tile, ctrls, ctrl_positions, bg, rng)
    expected = _compose_multi_output(tile, ctrls, ctrl_positions, bg)
    metrics = _fusion_metrics(tile, ctrls, ctrl_positions, bg)

    if not _input_signature_ok(inp, n_ctrl, bg):
        return None
    if metrics["output_components"] != n_ctrl:
        return None
    if metrics["extra_fg_pixels"] <= 0:
        return None
    if metrics["scaled_offset_sum"] <= 0:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid fused input")
