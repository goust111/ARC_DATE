# Variant of bf45cf4b
# Dimension: D10 — 符号编码与参数传递 (symbol_instructions)
# Change: 无显式符号编码，瓦片和控制精灵通过形态特征隐式区分... → 网格边缘添加颜色图例编码瓦片颜色映射

from __future__ import annotations

import sys
import numpy as np

try:
    from arc_utils import colors_present, crop_bbox, find_components, get_bg, paste
except ImportError:
    sys.path.append("/home/lixinyu/snap/new/solvers")
    from arc_utils import colors_present, crop_bbox, find_components, get_bg, paste

# Design analysis for the D10 variant
# ----------------------------------
# Reference-task note:
# - The prompt asked to read the source task solver for dbff022c first.
# - In the provided workspace, no dbff022c solver file is available, so that step
#   cannot be executed here. The design below therefore uses only the requested
#   original bf45cf4b logic and the D10 description.
#
# a) Concretization and behavior trace
# - Original input: exactly two disconnected components, a polychrome square tile
#   and a monochrome rectangular control sprite with holes.
# - Variant input: exactly three disconnected components:
#   1. the same control sprite,
#   2. the same source tile,
#   3. a thin legend rectangle that touches exactly one grid edge.
# - The legend is either 2×k on the top/bottom edge or k×2 on the left/right edge.
#   Each adjacent pair encodes one substitution "source color -> target color".
# - The output is still a tight expansion of the control sprite, but it stamps the
#   recolored tile, not the raw input tile.
# - Example trace:
#   Original: tile colors {2,3}, ctrl mask cell=filled => stamp tile with {2,3}.
#   Variant: edge legend says 2->8 and 3->6; solver must first recolor the tile to
#   {8,6}, then stamp that recolored version for each filled ctrl pixel.
#   Difference: the geometric expansion stage is unchanged; only the tile palette is
#   parameterized by an external symbol strip.
#
# b) Complexity calibration
# - The legend contains 2-3 mappings, not an unrestricted table. That is enough to
#   make the rule visible while keeping the decoding burden small.
# - Each mapped source color appears in the tile, and each target color is distinct
#   from bg, ctrl, and all source colors. This avoids ambiguous color reuse.
# - The legend is kept as one compact edge object so it is visually read as an
#   instruction block, not a third puzzle object competing with tile/control.
#
# c) Implicit-assumption audit of the original main()
# - `assert len(comps) == 2)` no longer holds because the legend introduces a third
#   disconnected component. The solver must isolate the legend first.
# - Sorting all components by "multicolor + square" is no longer safe because the
#   legend may be more colorful than the tile. The solver must exclude the legend
#   before disambiguating tile vs ctrl.
# - Using the tile directly during stamping is no longer valid; the tile must be
#   copied through a color-mapping stage first.
# - The control-mask logic remains valid because the control object is still a
#   monochrome rectangular sprite with bg holes.
#
# d) Placement legality
# - Generator guarantees all three components are disconnected under 8-connectivity.
# - Only the legend may touch an outer edge; tile and control stay one cell away
#   from every edge so legend detection is unambiguous.
# - A one-cell blank buffer is also kept between the legend band and the area where
#   tile/control may be placed, preventing accidental attachment.
#
# e) Visual observability
# - Humans can compare input tile colors with output tile colors and notice they do
#   not match directly; the only place explaining the change is the edge legend.
# - Using 2-3 repeated mapped colors inside the expanded output gives enough support
#   across a few examples to infer that the strip is a palette instruction.
#
# f) Visual cleanliness
# - The legend is a thin, isolated strip on one edge rather than an overlaid mark.
# - The main scene still contains only one tile object and one control object, so
#   the original puzzle remains easy to parse.
#
# g) Color-change inferability
# - The mapped palette is small (2-3 source colors).
# - All mappings are one-to-one and all targets are distinct from all sources, so
#   the legend can be read directly as a pair list instead of requiring extra logic.
#
# New solver assumptions introduced by this variant
# - Exactly one component is a thin edge-touching legend with one dimension equal to 2.
# - Legend entries cover every non-bg tile color used by the generator.
# - Tile and control remain the only two non-legend foreground components.
# The generator below explicitly enforces all three conditions.


def _legend_spec(comp: dict, shape: tuple[int, int]) -> dict | None:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    H, W = map(int, shape)

    touches = {
        "top": r0 == 0,
        "bottom": r1 == H - 1,
        "left": c0 == 0,
        "right": c1 == W - 1,
    }
    if sum(bool(v) for v in touches.values()) != 1:
        return None

    if h == 2 and w >= 2 and (touches["top"] or touches["bottom"]):
        return {"edge": "top" if touches["top"] else "bottom", "length": w}
    if w == 2 and h >= 2 and (touches["left"] or touches["right"]):
        return {"edge": "left" if touches["left"] else "right", "length": h}
    return None


def _decode_legend(patch: np.ndarray, edge: str, bg: int) -> dict[int, int]:
    if edge == "top":
        srcs = patch[0, :]
        dsts = patch[1, :]
    elif edge == "bottom":
        srcs = patch[1, :]
        dsts = patch[0, :]
    elif edge == "left":
        srcs = patch[:, 0]
        dsts = patch[:, 1]
    elif edge == "right":
        srcs = patch[:, 1]
        dsts = patch[:, 0]
    else:
        raise ValueError(f"unknown edge {edge}")

    mapping: dict[int, int] = {}
    for src, dst in zip(srcs, dsts):
        s = int(src)
        d = int(dst)
        if s == int(bg) or d == int(bg):
            raise ValueError("legend may not contain bg")
        mapping[s] = d
    if len(mapping) != len(srcs):
        raise ValueError("legend source colors must be unique")
    return mapping


def _apply_mapping(tile: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    recolored = np.array(tile, copy=True)
    for src, dst in mapping.items():
        recolored[tile == int(src)] = int(dst)
    return recolored


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - Input contains a control sprite, a source tile, and an edge legend.
    - The edge legend encodes source->target color substitutions for the tile.
    - Recolor the tile using that legend, then expand the control sprite exactly as
      in the original task by stamping the recolored tile for each filled ctrl cell.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    assert len(comps) == 3

    legend_candidates: list[tuple[int, dict]] = []
    for idx, comp in enumerate(comps):
        spec = _legend_spec(comp, g.shape)
        if spec is not None:
            legend_candidates.append((idx, spec))
    assert len(legend_candidates) == 1

    legend_idx, legend_info = legend_candidates[0]
    legend_patch = crop_bbox(g, comps[legend_idx]["bbox"])
    mapping = _decode_legend(legend_patch, legend_info["edge"], bg)

    infos = []
    for idx, comp in enumerate(comps):
        if idx == legend_idx:
            continue
        patch = crop_bbox(g, comp["bbox"])
        cols = colors_present(patch, exclude={bg})
        infos.append({"bbox": comp["bbox"], "patch": patch, "cols": cols})

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

    block = _apply_mapping(block, mapping)

    bh, bw = map(int, block.shape)
    ch, cw = map(int, ctrl.shape)

    if not np.any(ctrl != bg):
        return np.full((0, 0), int(bg), dtype=int)

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


def _make_tile(bg: int, tile_colors: list[int], rng: np.random.Generator) -> np.ndarray:
    size = int(rng.integers(3, 6))
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


def _compose_output(tile: np.ndarray, ctrl: np.ndarray, bg: int, mapping: dict[int, int]) -> np.ndarray:
    transformed = _apply_mapping(tile, mapping)
    th, tw = transformed.shape
    ch, cw = ctrl.shape
    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if ctrl[r, c] != bg:
                paste(out, transformed, r * th, c * tw)
    return out


def _make_legend(mapping: dict[int, int], edge: str, rng: np.random.Generator) -> np.ndarray:
    items = list(mapping.items())
    rng.shuffle(items)
    k = len(items)

    if edge == "top":
        patch = np.zeros((2, k), dtype=int)
        patch[0, :] = [src for src, _dst in items]
        patch[1, :] = [dst for _src, dst in items]
    elif edge == "bottom":
        patch = np.zeros((2, k), dtype=int)
        patch[0, :] = [dst for _src, dst in items]
        patch[1, :] = [src for src, _dst in items]
    elif edge == "left":
        patch = np.zeros((k, 2), dtype=int)
        patch[:, 0] = [src for src, _dst in items]
        patch[:, 1] = [dst for _src, dst in items]
    elif edge == "right":
        patch = np.zeros((k, 2), dtype=int)
        patch[:, 0] = [dst for _src, dst in items]
        patch[:, 1] = [src for src, _dst in items]
    else:
        raise ValueError(f"unknown edge {edge}")
    return patch


def _object_area_bounds(H: int, W: int, edge: str) -> tuple[int, int, int, int]:
    if edge == "top":
        return 3, H - 2, 1, W - 2
    if edge == "bottom":
        return 1, H - 4, 1, W - 2
    if edge == "left":
        return 1, H - 2, 3, W - 2
    if edge == "right":
        return 1, H - 2, 1, W - 4
    raise ValueError(f"unknown edge {edge}")


def _make_input(
    tile: np.ndarray,
    ctrl: np.ndarray,
    legend: np.ndarray,
    bg: int,
    edge: str,
    rng: np.random.Generator,
) -> np.ndarray:
    horizontal = bool(rng.integers(0, 2))
    gap = 1

    th, tw = tile.shape
    ch, cw = ctrl.shape
    lh, lw = legend.shape

    if horizontal:
        content_h = max(th, ch)
        content_w = tw + gap + cw
    else:
        content_h = th + gap + ch
        content_w = max(tw, cw)

    min_h = content_h + (4 if edge in {"top", "bottom"} else 2)
    min_w = content_w + (4 if edge in {"left", "right"} else 2)
    min_h = max(min_h, lh + 2)
    min_w = max(min_w, lw + 2)

    H = int(rng.integers(max(12, min_h), 19))
    W = int(rng.integers(max(12, min_w), 19))
    rmin, rmax, cmin, cmax = _object_area_bounds(H, W, edge)
    area_h = int(rmax - rmin + 1)
    area_w = int(cmax - cmin + 1)

    if horizontal:
        if rng.random() < 0.5:
            left_obj, right_obj = tile, ctrl
            left_name = "tile"
        else:
            left_obj, right_obj = ctrl, tile
            left_name = "ctrl"
        lh0, lw0 = left_obj.shape
        rh0, rw0 = right_obj.shape
        slack_w = area_w - (lw0 + rw0 + gap)
        slack_h_left = area_h - lh0
        slack_h_right = area_h - rh0
        c_left = cmin + int(rng.integers(0, slack_w + 1))
        c_right = cmin + int(rng.integers(c_left - cmin + lw0 + gap, area_w - rw0 + 1))
        r_left = rmin + int(rng.integers(0, slack_h_left + 1))
        r_right = rmin + int(rng.integers(0, slack_h_right + 1))
        if left_name == "tile":
            tile_pos = (r_left, c_left)
            ctrl_pos = (r_right, c_right)
        else:
            ctrl_pos = (r_left, c_left)
            tile_pos = (r_right, c_right)
    else:
        if rng.random() < 0.5:
            top_obj, bot_obj = tile, ctrl
            top_name = "tile"
        else:
            top_obj, bot_obj = ctrl, tile
            top_name = "ctrl"
        th0, tw0 = top_obj.shape
        bh0, bw0 = bot_obj.shape
        slack_h = area_h - (th0 + bh0 + gap)
        slack_w_top = area_w - tw0
        slack_w_bot = area_w - bw0
        r_top = rmin + int(rng.integers(0, slack_h + 1))
        r_bot = rmin + int(rng.integers(r_top - rmin + th0 + gap, area_h - bh0 + 1))
        c_top = cmin + int(rng.integers(0, slack_w_top + 1))
        c_bot = cmin + int(rng.integers(0, slack_w_bot + 1))
        if top_name == "tile":
            tile_pos = (r_top, c_top)
            ctrl_pos = (r_bot, c_bot)
        else:
            ctrl_pos = (r_top, c_top)
            tile_pos = (r_bot, c_bot)

    if edge in {"top", "bottom"}:
        legend_r = 0 if edge == "top" else H - 2
        legend_c = int(rng.integers(1, W - lw))
    else:
        legend_c = 0 if edge == "left" else W - 2
        legend_r = int(rng.integers(1, H - lh))

    grid = np.full((H, W), int(bg), dtype=int)
    paste(grid, legend, legend_r, legend_c)
    paste(grid, tile, *tile_pos)
    paste(grid, ctrl, *ctrl_pos, transparent=bg)
    return grid


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    bg = int(rng.integers(0, 10))
    n_tile_colors = int(rng.integers(2, 4))

    others = [c for c in range(10) if c != bg]
    picked = list(rng.choice(np.array(others, dtype=int), size=1 + 2 * n_tile_colors, replace=False))
    ctrl_color = int(picked[0])
    src_colors = [int(x) for x in picked[1 : 1 + n_tile_colors]]
    dst_colors = [int(x) for x in picked[1 + n_tile_colors : 1 + 2 * n_tile_colors]]
    rng.shuffle(dst_colors)
    mapping = {int(src): int(dst) for src, dst in zip(src_colors, dst_colors)}

    tile = _make_tile(bg, src_colors, rng)
    ctrl = _make_ctrl(bg, ctrl_color, rng)
    edge = str(rng.choice(np.array(["top", "bottom", "left", "right"], dtype=object)))
    legend = _make_legend(mapping, edge, rng)

    inp = _make_input(tile, ctrl, legend, bg, edge, rng)
    expected = _compose_output(tile, ctrl, bg, mapping)

    comps = find_components(inp, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    if len(comps) != 3:
        return None

    legend_candidates = []
    for comp in comps:
        spec = _legend_spec(comp, inp.shape)
        if spec is not None:
            legend_candidates.append((comp, spec))
    if len(legend_candidates) != 1:
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
