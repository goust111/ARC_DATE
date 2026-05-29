# Variant of bf45cf4b
# Dimension: D16 — 输入前景对象的连通性要求 (src)
# Change: 两个前景对象必须在8-连通下完全分离（至少1像素间隔）... → 对象嵌套放置：瓦片位于控制精灵的空洞内部

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, crop_bbox, find_components, get_bg, paste

# Design analysis for the D16 variant
# -----------------------------------
# a) Concrete behavior:
#    Original inputs had two separated foreground components: a monochrome control
#    sprite and a polychrome square tile. In this variant, the tile is pasted
#    inside a rectangular hole of the control sprite's bbox. The tile touches the
#    hole boundary, so under 8-connectivity the whole foreground becomes a single
#    connected component. Output is unchanged: treat only the control-color cells
#    as the mask, and stamp the tile at each filled control pixel.
#    Example trace:
#    - Original: [tile] ...... [control with holes]  -> easy split by CC count=2.
#    - Variant: a control-color rectangular frame contains a multicolor square
#      inside its interior hole; the tile is no longer a standalone component.
#      Solver must first recover the outer monochrome role, then crop the inner
#      square tile from the bbox interior.
# b) Complexity calibration:
#    The tile remains square and multicolor, while the control remains a single
#    monochrome connected object that reaches all four sides of its bbox. This
#    keeps the task inferable: the outer object is still obviously the mask, but
#    segmentation is harder because global CC splitting no longer works.
# c) Hidden assumptions in the original main():
#    - `len(comps) == 2)` assumed two foreground CCs; false here because the tile
#      touches the control and merges into one CC.
#    - Sorting components by number of colors assumed tile/control were already
#      separated. In the variant, the whole foreground is multicolor, so this
#      heuristic is unusable.
#    - `ctrl = infos[1]["patch"]` assumed the control bbox can be read directly as
#      one component crop. Here we must recover the control from its own monochrome
#      component and ignore the embedded tile pixels inside that bbox.
# d) Placement legality:
#    Generator creates a solid monochrome rectangle, carves one interior
#    rectangular hole, and embeds the square tile fully inside that hole. The
#    hole never touches the outer border, the tile never crosses the hole border,
#    and the tile is anchored to at least one hole side so the foreground merges
#    into one 8-connected object.
# e) Visual observability:
#    Across samples, a human sees an outer monochrome frame-like object and a
#    repeated multicolor motif lodged inside its cavity. The output still looks
#    like an enlargement of the outer monochrome pattern by the inner motif, so
#    the nesting relation is visible and relevant.
# f) Visual cleanliness:
#    There are only three semantic regions inside the control bbox: control-color
#    cells, background hole cells, and tile cells. No extra markers or overlaps.
# g) Color inferability:
#    Tile colors are distinct from the control color and from the background. The
#    tile uses only 2-3 colors, enough to be distinctive without making the input
#    noisy.
# 9) New main() assumptions audited against generator:
#    - There is exactly one monochrome component whose bbox is larger than the
#      embedded tile and whose color appears on all four bbox sides.
#    - All non-bg, non-control colors inside that bbox belong to one square tile.
#    - The tile lies strictly inside the control bbox and touches at least one
#      side of the control's interior hole.
#    Generator guarantees each of these conditions explicitly.


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _touching_sides(patch: np.ndarray, ctrl_color: int, tile_bbox: tuple[int, int, int, int]) -> int:
    tr0, tc0, tr1, tc1 = tile_bbox
    touched = 0
    if tr0 > 0 and np.any(patch[tr0 - 1, tc0 : tc1 + 1] == ctrl_color):
        touched += 1
    if tr1 + 1 < patch.shape[0] and np.any(patch[tr1 + 1, tc0 : tc1 + 1] == ctrl_color):
        touched += 1
    if tc0 > 0 and np.any(patch[tr0 : tr1 + 1, tc0 - 1] == ctrl_color):
        touched += 1
    if tc1 + 1 < patch.shape[1] and np.any(patch[tr0 : tr1 + 1, tc1 + 1] == ctrl_color):
        touched += 1
    return int(touched)


def _detect_nested_parts(grid: np.ndarray) -> tuple[int, np.ndarray, np.ndarray, dict]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    mono = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")

    candidates: list[dict] = []
    for comp in mono:
        ctrl_color = int(comp["color"])
        patch = crop_bbox(g, comp["bbox"])
        ph, pw = map(int, patch.shape)

        if not (
            np.any(patch[0] == ctrl_color)
            and np.any(patch[-1] == ctrl_color)
            and np.any(patch[:, 0] == ctrl_color)
            and np.any(patch[:, -1] == ctrl_color)
        ):
            continue

        other_mask = (patch != bg) & (patch != ctrl_color)
        tile_bbox = _bbox_from_mask(other_mask)
        if tile_bbox is None:
            continue

        tr0, tc0, tr1, tc1 = tile_bbox
        if tr0 == 0 or tc0 == 0 or tr1 == ph - 1 or tc1 == pw - 1:
            continue

        tile = crop_bbox(patch, tile_bbox)
        if tile.shape[0] != tile.shape[1]:
            continue
        if np.any(tile == ctrl_color):
            continue

        tile_colors = colors_present(tile, exclude={bg, ctrl_color})
        if len(tile_colors) < 2:
            continue

        touched = _touching_sides(patch, ctrl_color, tile_bbox)
        if touched == 0:
            continue

        candidates.append(
            {
                "ctrl_patch": patch,
                "tile": tile,
                "ctrl_color": ctrl_color,
                "bbox": comp["bbox"],
                "touching_sides": touched,
                "comp_area": int(comp["area"]),
            }
        )

    assert candidates, "failed to identify nested control/tile structure"
    candidates.sort(
        key=lambda d: (
            d["ctrl_patch"].shape[0] * d["ctrl_patch"].shape[1],
            d["comp_area"],
            d["touching_sides"],
        ),
        reverse=True,
    )
    best = candidates[0]
    return bg, best["ctrl_patch"], best["tile"], best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - The input foreground is a single 8-connected nested structure.
    - The outer monochrome object is still the control sprite.
    - A multicolor square tile is embedded in one hole of that control sprite.
    - Ignore the embedded tile when reading the control mask, then enlarge the
      control by stamping the tile at each control-color cell.
    """
    bg, ctrl, tile, info = _detect_nested_parts(grid)
    ctrl_color = int(info["ctrl_color"])
    mask = (ctrl == ctrl_color)

    th, tw = map(int, tile.shape)
    ch, cw = map(int, ctrl.shape)
    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if bool(mask[r, c]):
                paste(out, tile, r * th, c * tw)
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
    n_tile_colors = int(rng.integers(2, 4))
    tile_colors = list(rng.choice(fg_colors, size=n_tile_colors, replace=False))
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


def _make_ctrl_with_hole(
    bg: int, ctrl_color: int, tile_shape: tuple[int, int], rng: np.random.Generator
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    th, tw = map(int, tile_shape)

    h_lo = max(5, th + 3)
    h_hi = min(9, th + 4)
    w_lo = max(5, tw + 3)
    w_hi = min(9, tw + 4)
    h = int(rng.integers(h_lo, h_hi + 1))
    w = int(rng.integers(w_lo, w_hi + 1))

    hole_h_max = min(h - 2, th + 2)
    hole_w_max = min(w - 2, tw + 2)
    hole_h = th
    hole_w = tw
    for _ in range(20):
        hole_h = int(rng.integers(th, hole_h_max + 1))
        hole_w = int(rng.integers(tw, hole_w_max + 1))
        if hole_h > th or hole_w > tw:
            break
    if hole_h == th and hole_w == tw:
        if hole_h_max > th:
            hole_h += 1
        elif hole_w_max > tw:
            hole_w += 1

    hr0_max = h - hole_h - 1
    hc0_max = w - hole_w - 1
    assert hr0_max >= 1 and hc0_max >= 1
    hr0 = int(rng.integers(1, hr0_max + 1))
    hc0 = int(rng.integers(1, hc0_max + 1))
    hole_bbox = (hr0, hc0, hr0 + hole_h - 1, hc0 + hole_w - 1)

    ctrl = np.full((h, w), int(ctrl_color), dtype=int)
    ctrl[hr0 : hr0 + hole_h, hc0 : hc0 + hole_w] = int(bg)
    return ctrl, hole_bbox


def _embed_tile(
    ctrl: np.ndarray, tile: np.ndarray, hole_bbox: tuple[int, int, int, int], bg: int, rng: np.random.Generator
) -> tuple[np.ndarray, tuple[int, int, int, int], int]:
    hr0, hc0, hr1, hc1 = hole_bbox
    hole_h = hr1 - hr0 + 1
    hole_w = hc1 - hc0 + 1
    th, tw = map(int, tile.shape)
    slack_r = hole_h - th
    slack_c = hole_w - tw

    placements: list[tuple[int, int, int]] = []
    for dr in range(slack_r + 1):
        for dc in range(slack_c + 1):
            touch = int(dr == 0) + int(dr == slack_r) + int(dc == 0) + int(dc == slack_c)
            if touch >= 1:
                placements.append((dr, dc, touch))
    assert placements, "no legal embedded-tile placement"

    dr, dc, touch = placements[int(rng.integers(len(placements)))]
    tr0 = hr0 + dr
    tc0 = hc0 + dc

    patch = ctrl.copy()
    paste(patch, tile, tr0, tc0)
    return patch, (tr0, tc0, tr0 + th - 1, tc0 + tw - 1), int(touch)


def _compose_output(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    th, tw = tile.shape
    ch, cw = ctrl.shape
    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    ctrl_colors = colors_present(ctrl, exclude={bg})
    assert len(ctrl_colors) == 1
    ctrl_color = int(ctrl_colors[0])
    for r in range(ch):
        for c in range(cw):
            if ctrl[r, c] == ctrl_color:
                paste(out, tile, r * th, c * tw)
    return out


def _make_input(tile: np.ndarray, ctrl: np.ndarray, composite: np.ndarray, bg: int, rng: np.random.Generator) -> np.ndarray:
    h, w = ctrl.shape
    H = int(rng.integers(max(12, h + 2), 19))
    W = int(rng.integers(max(12, w + 2), 19))
    r0 = int(rng.integers(1, H - h))
    c0 = int(rng.integers(1, W - w))

    grid = np.full((H, W), int(bg), dtype=int)
    paste(grid, composite, r0, c0)
    return grid


def _variant_metrics(inp: np.ndarray) -> dict[str, int]:
    bg, ctrl, tile, info = _detect_nested_parts(inp)
    comps = find_components(inp, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    return {
        "global_fg_components": int(len(comps)),
        "ctrl_h": int(ctrl.shape[0]),
        "ctrl_w": int(ctrl.shape[1]),
        "tile_size": int(tile.shape[0]),
        "touching_sides": int(info["touching_sides"]),
    }


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    palette = list(rng.choice(np.arange(10, dtype=int), size=5, replace=False))
    bg = int(palette[0])
    tile_colors = [int(x) for x in palette[1:4]]
    ctrl_color = int(palette[4])

    tile = _make_tile(bg, tile_colors, rng)
    ctrl, hole_bbox = _make_ctrl_with_hole(bg, ctrl_color, tile.shape, rng)
    composite, _tile_bbox, touch = _embed_tile(ctrl, tile, hole_bbox, bg, rng)
    inp = _make_input(tile, ctrl, composite, bg, rng)
    expected = _compose_output(tile, ctrl, bg)

    try:
        out = main(inp)
        metrics = _variant_metrics(inp)
    except Exception:
        return None

    if not np.array_equal(out, expected):
        return None
    if metrics["global_fg_components"] != 1:
        return None
    if metrics["touching_sides"] < 1:
        return None
    if touch < 1:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid nested input")
