# Combo variant of bf45cf4b (index 13)
# combo_spec: D2/3+D16/1
# Name: 嵌套对象包含关系识别

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import colors_present, crop_bbox, find_components, get_bg, paste
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[4] / "solvers"))
    from arc_utils import colors_present, crop_bbox, find_components, get_bg, paste


# Design analysis for combo D2/3 + D16/1
# ---------------------------------------
# a) Concrete behavior:
#    Original bf45cf4b used two separated objects and a color-count heuristic:
#    multicolor square -> tile, single-color holed rectangle -> control mask.
#    In this combo, the input still contains exactly two foreground components,
#    but they are no longer side-by-side. One hollow outer component encloses a
#    cavity, and the second component sits entirely inside that cavity with a
#    background moat around it.
#    Small trace:
#      - Original input: [tile] .... [control].
#      - Combo input:    [outer monochrome ring with hole]
#                           [bg moat][inner multicolor square][bg moat]
#        so the inner object is truly enclosed, not merely inside the outer bbox.
#      - Original solver path: crop each component, count colors, sort.
#      - Combo solver path: locate the containing bbox pair, take the outer
#        component coords as the control mask, verify that the inner object lies
#        in a sealed cavity of that mask, and crop the inner bbox as tile.
#    D2/3 changes "identify roles by appearance" into "identify roles by spatial
#    relation". Because D16/1 nests the two objects, the meaningful spatial cue
#    is containment rather than left/right or top/bottom ordering: the container
#    is the control source, and the contained object is the tile source.
# b) Complexity calibration:
#    A plain nested frame plus an inner square is easy to see but not trivial:
#    the user must infer that the enclosed object supplies the repeated motif
#    while the enclosing hollow object supplies the binary mask. The outer mask
#    stays monochrome and touches all four sides of its bbox, and the tile stays
#    a small square with 2-3 colors. This avoids ambiguity while making the
#    structural change visually salient.
# c) Hidden assumptions broken in original main():
#    - `len(comps) == 2`: still true, but no longer sufficient for role labeling.
#    - `patch = crop_bbox(g, comp["bbox"])`: for the outer component this crop
#      now includes the enclosed tile, so the crop is no longer monochrome.
#    - Sorting by `(len(cols) >= 2, len(cols), square)` now mislabels the outer
#      object as the tile because the tile colors are visible inside its bbox.
#    - `ctrl_cols = colors_present(ctrl, exclude={bg})` assumes the chosen
#      control crop only contains one foreground color; false after nesting.
#    Therefore the combo solver must recover the outer mask from component
#    coordinates rather than from raw colors in the bbox crop.
# d) Placement legality:
#    Generator first creates a hollow monochrome control object, then places a
#    multicolor square strictly inside the hole with a one-cell background moat.
#    That guarantees no overlap, no out-of-bounds paste, strict enclosure inside
#    one sealed cavity, and two distinct 8-connected components.
# e) Visual observability:
#    In 2-4 examples, a human can see that the inner square is always enclosed
#    and later repeated everywhere the outer mask is filled. The nested relation
#    is the only stable cue explaining which object becomes the stamp and which
#    becomes the template.
# f) Visual cleanliness:
#    The outer bbox contains exactly three visually distinct regions: control
#    color, background hole/moat, and the inner tile. No extra annotation colors
#    or overlaps are introduced.
# g) Change inferability:
#    The decisive clue is geometric containment, not color IDs. Tile colors are
#    kept distinct from the control color only to keep the inner object legible.
# 9) New main() assumptions audited against generator:
#    - Exactly two foreground components exist under 8-connectivity.
#    - One bbox strictly contains the other bbox.
#    - The outer component is monochrome on its own cells and touches all four
#      sides of its bbox, so it is read as the enclosing control object.
#    - The inner component crop is square and has at least two non-bg colors.
#    - The minimum Chebyshev distance between the two components is >= 2, which
#      means there is at least one background cell moat and they are truly
#      disconnected despite being nested.
#    - Starting from the tile cells and moving only through non-control cells
#      inside the outer bbox never reaches the bbox boundary; this is what turns
#      mere bbox nesting into actual "contained inside a hole".
#    Generator enforces each of these assumptions and rejects any attempt that
#    would let the original color-count logic still solve the sample unchanged.


def _bbox_strictly_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return (
        int(outer[0]) < int(inner[0])
        and int(outer[1]) < int(inner[1])
        and int(outer[2]) > int(inner[2])
        and int(outer[3]) > int(inner[3])
    )


def _component_mask_in_bbox(comp: dict, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    coords = np.array(comp["coords"], dtype=int)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _mask_touches_all_sides(mask: np.ndarray) -> bool:
    return bool(
        np.any(mask[0])
        and np.any(mask[-1])
        and np.any(mask[:, 0])
        and np.any(mask[:, -1])
    )


def _is_enclosed_cavity(ctrl_mask: np.ndarray, inner_bbox: tuple[int, int, int, int]) -> bool:
    h, w = map(int, ctrl_mask.shape)
    r0, c0, r1, c1 = map(int, inner_bbox)
    open_mask = ~ctrl_mask
    seen = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if not open_mask[r, c]:
                return False
            stack.append((r, c))
            seen[r, c] = True

    while stack:
        r, c = stack.pop()
        if r in (0, h - 1) or c in (0, w - 1):
            return False
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and open_mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return True


def _min_chebyshev_distance(coords_a: np.ndarray, coords_b: np.ndarray) -> int:
    best: int | None = None
    aa = np.array(coords_a, dtype=int)
    bb = np.array(coords_b, dtype=int)
    for ra, ca in aa:
        d = np.max(np.abs(bb - np.array([ra, ca], dtype=int)), axis=1)
        here = int(d.min())
        if best is None or here < best:
            best = here
            if best == 1:
                break
    return 0 if best is None else int(best)


def _detect_nested_parts(grid: np.ndarray) -> tuple[int, np.ndarray, np.ndarray, dict]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    assert len(comps) == 2, "expected exactly two foreground components"

    candidates: list[dict] = []
    for outer in comps:
        for inner in comps:
            if outer is inner:
                continue
            outer_bbox = tuple(int(x) for x in outer["bbox"])
            inner_bbox = tuple(int(x) for x in inner["bbox"])
            if not _bbox_strictly_contains(outer_bbox, inner_bbox):
                continue

            ctrl_patch = crop_bbox(g, outer_bbox)
            ctrl_mask = _component_mask_in_bbox(outer, outer_bbox)
            tile = crop_bbox(g, inner_bbox)

            outer_colors = sorted({int(g[r, c]) for r, c in np.array(outer["coords"], dtype=int)})
            tile_colors = colors_present(tile, exclude={bg})
            if len(outer_colors) != 1:
                continue
            if not _mask_touches_all_sides(ctrl_mask):
                continue
            if tile.shape[0] != tile.shape[1]:
                continue
            if len(tile_colors) < 2:
                continue
            if int(outer_colors[0]) in tile_colors:
                continue

            moat = _min_chebyshev_distance(np.array(outer["coords"], dtype=int), np.array(inner["coords"], dtype=int))
            if moat < 2:
                continue
            if not _is_enclosed_cavity(ctrl_mask, inner_bbox):
                continue
            candidates.append(
                {
                    "ctrl_patch": ctrl_patch,
                    "ctrl_mask": ctrl_mask,
                    "tile": tile,
                    "ctrl_color": int(outer_colors[0]),
                    "outer_bbox": outer_bbox,
                    "inner_bbox": inner_bbox,
                    "moat": moat,
                    "outer_area": int(outer["area"]),
                }
            )

    assert candidates, "failed to identify nested outer control and inner tile"
    candidates.sort(
        key=lambda d: (
            d["moat"] >= 2,
            d["outer_area"],
            d["ctrl_patch"].shape[0] * d["ctrl_patch"].shape[1],
        ),
        reverse=True,
    )
    best = candidates[0]
    return bg, best["ctrl_mask"], best["tile"], best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - There are exactly two foreground components.
    - One component is fully enclosed inside the other's bbox.
    - The outer component is the control mask; the inner square component is the tile.
    - Expand each filled mask pixel into one copy of the inner tile.
    """
    bg, ctrl_mask, tile, _info = _detect_nested_parts(grid)
    th, tw = map(int, tile.shape)
    ch, cw = map(int, ctrl_mask.shape)
    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if bool(ctrl_mask[r, c]):
                paste(out, tile, r * th, c * tw)
    return out


def _dihedral_orbits(size: int) -> list[list[tuple[int, int]]]:
    seen: set[tuple[int, int]] = set()
    out: list[list[tuple[int, int]]] = []
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
            out.append(cells)
    return out


def _make_tile(bg: int, fg_colors: list[int], rng: np.random.Generator) -> np.ndarray:
    size = int(rng.integers(2, 6))
    n_tile_colors = int(rng.integers(2, min(4, len(fg_colors) + 1)))
    tile_colors = [int(x) for x in rng.choice(fg_colors, size=n_tile_colors, replace=False)]
    orbits = _dihedral_orbits(size)
    rng.shuffle(orbits)

    tile = np.full((size, size), int(tile_colors[0]), dtype=int)
    for idx, cells in enumerate(orbits):
        color = int(tile_colors[idx % len(tile_colors)])
        for r, c in cells:
            tile[r, c] = color

    if len(set(int(x) for x in tile.flatten())) < 2:
        tile[0, 0] = int(tile_colors[-1])
        tile[-1, -1] = int(tile_colors[-1])
    assert len(colors_present(tile, exclude={bg})) >= 2
    return tile


def _make_ctrl_with_hole(
    bg: int, ctrl_color: int, tile_shape: tuple[int, int], rng: np.random.Generator
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    th, tw = map(int, tile_shape)

    hole_h = int(rng.integers(th + 2, th + 5))
    hole_w = int(rng.integers(tw + 2, tw + 5))
    h = hole_h + int(rng.integers(2, 5))
    w = hole_w + int(rng.integers(2, 5))
    h = min(h, 10)
    w = min(w, 10)
    hole_h = min(hole_h, h - 2)
    hole_w = min(hole_w, w - 2)
    assert hole_h >= th + 2 and hole_w >= tw + 2

    top = int(rng.integers(1, h - hole_h))
    left = int(rng.integers(1, w - hole_w))
    hole_bbox = (top, left, top + hole_h - 1, left + hole_w - 1)

    ctrl = np.full((h, w), int(ctrl_color), dtype=int)
    ctrl[top : top + hole_h, left : left + hole_w] = int(bg)
    return ctrl, hole_bbox


def _embed_tile_disconnected(
    ctrl: np.ndarray, tile: np.ndarray, hole_bbox: tuple[int, int, int, int], bg: int, rng: np.random.Generator
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    hr0, hc0, hr1, hc1 = map(int, hole_bbox)
    hole_h = hr1 - hr0 + 1
    hole_w = hc1 - hc0 + 1
    th, tw = map(int, tile.shape)
    slack_r = hole_h - th
    slack_c = hole_w - tw
    assert slack_r >= 2 and slack_c >= 2

    dr = int(rng.integers(1, slack_r))
    dc = int(rng.integers(1, slack_c))
    tr0 = hr0 + dr
    tc0 = hc0 + dc

    composite = ctrl.copy()
    paste(composite, tile, tr0, tc0)
    return composite, (tr0, tc0, tr0 + th - 1, tc0 + tw - 1)


def _compose_output(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    th, tw = map(int, tile.shape)
    ch, cw = map(int, ctrl.shape)
    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if ctrl[r, c] != bg:
                paste(out, tile, r * th, c * tw)
    return out


def _make_input(composite: np.ndarray, bg: int, rng: np.random.Generator) -> np.ndarray:
    h, w = map(int, composite.shape)
    H = int(rng.integers(max(12, h + 2), 19))
    W = int(rng.integers(max(12, w + 2), 19))
    r0 = int(rng.integers(1, H - h))
    c0 = int(rng.integers(1, W - w))
    grid = np.full((H, W), int(bg), dtype=int)
    paste(grid, composite, r0, c0)
    return grid


def _original_main_like(grid: np.ndarray) -> np.ndarray:
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
    assert ctrl_cols
    ctrl_col = int(ctrl_cols[0])
    mask = (ctrl == ctrl_col)

    out = np.full((ch * bh, cw * bw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if bool(mask[r, c]):
                paste(out, block, r * bh, c * bw)
    return out


def _variant_metrics(inp: np.ndarray) -> dict[str, int]:
    bg, ctrl_mask, tile, info = _detect_nested_parts(inp)
    comps = find_components(inp, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    combo_out = main(inp)
    original_failed = 0
    original_same_output = 0
    try:
        original_out = _original_main_like(inp)
        original_same_output = int(np.array_equal(original_out, combo_out))
    except Exception:
        original_failed = 1
    return {
        "components": int(len(comps)),
        "ctrl_h": int(ctrl_mask.shape[0]),
        "ctrl_w": int(ctrl_mask.shape[1]),
        "tile_size": int(tile.shape[0]),
        "moat": int(info["moat"]),
        "nested": int(_bbox_strictly_contains(info["outer_bbox"], info["inner_bbox"])),
        "sealed": int(_is_enclosed_cavity(ctrl_mask, info["inner_bbox"])),
        "original_failed": int(original_failed),
        "original_same_output": int(original_same_output),
    }


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    palette = [int(x) for x in rng.choice(np.arange(10, dtype=int), size=5, replace=False)]
    bg = int(palette[0])
    tile_colors = palette[1:4]
    ctrl_color = int(palette[4])

    tile = _make_tile(bg, tile_colors, rng)
    ctrl, hole_bbox = _make_ctrl_with_hole(bg, ctrl_color, tile.shape, rng)
    composite, _tile_bbox = _embed_tile_disconnected(ctrl, tile, hole_bbox, bg, rng)
    inp = _make_input(composite, bg, rng)
    expected = _compose_output(tile, ctrl, bg)

    try:
        out = main(inp)
        metrics = _variant_metrics(inp)
    except Exception:
        return None

    if not np.array_equal(out, expected):
        return None
    if metrics["components"] != 2:
        return None
    if metrics["nested"] != 1:
        return None
    if metrics["sealed"] != 1:
        return None
    if metrics["moat"] < 2:
        return None
    if metrics["original_same_output"] != 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid combo-nested input")
