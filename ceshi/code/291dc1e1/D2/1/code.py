# Variant of 291dc1e1
# Dimension: D2 — 排序消歧规则 (match_cond)
# Change: 主键为到蓝线距离，次键为到红线距离，再次键为沿蓝线方向坐标和沿红线方向坐标... → 次键为矩形面积

from __future__ import annotations

import numpy as np
from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste


# Design analysis for the D2 variant
# a) Concrete behavior:
#    The scene stays "two edge lines + corner marker + interior rectangles".
#    The original task sorts rectangles mainly by distance to the blue edge and,
#    inside the same blue-distance band, by distance to the red edge. In the
#    variant, a band is still determined by blue-edge distance, but rectangles
#    inside that band are ordered by rectangle area instead of position. Example:
#    if a band contains areas [10, 4, 8] from near-red to far-red, the original
#    stack order is [10, 4, 8] while the variant order is [4, 8, 10].
# b) Complexity calibration:
#    To make the new rule visible, each generated band contains 3-4 rectangles
#    with distinct areas and a deliberately non-monotone left-to-right area
#    sequence. This prevents accidental agreement with either left-to-right or
#    right-to-left positional sorting after flips.
# c) Implicit assumptions review:
#    - Corner detection still assumes the marked corner is the only corner cell
#      whose neighbors have different colors. The generator preserves that.
#    - Rectangle extraction still assumes every object is a solid rectangle with
#      no background inside its bbox. The generator keeps full rectangular fills.
#    - Rotation-on-vertical-blue is unchanged, so object geometry must still
#      survive rotation cleanly. Non-square rectangles are included regularly.
#    - The old solver relied on red-distance as the second key. That assumption
#      is exactly what changes here, so tie-breaking must be retained after area
#      sorting to stay deterministic when equal-area rectangles ever appear.
# d) Placement legality:
#    Rectangles are placed on two separated bands in a canonical layout, with
#    fixed gaps of two cells, inside bounds, and never overlap. After a global
#    flip/rotation transform, those non-overlap guarantees still hold.
# e) Visual observability:
#    Humans can still see that blue-distance bands are preserved, but inside each
#    band the stack now follows size rather than proximity to the red line. The
#    generator rejects any sample where the old and new outputs coincide.
# f) Visual cleanliness:
#    Only three scene elements coexist: the two edge lines, the corner marker,
#    and separated rectangles. Bands are spaced apart so object grouping remains
#    visually readable.
# g) Color inference:
#    Colors are decorative only; no new color mapping is introduced. Minority
#    pixels may appear inside rectangles, but each rectangle remains a complete
#    filled bbox, so area is determined by shape, not by color count.
# h) Generator validity check:
#    Each generated input is checked with both the variant solver and an internal
#    legacy-order solver; the sample is accepted only if the outputs differ.
# i) New assumptions after modification:
#    The solver now assumes area can be computed reliably from the extracted bbox
#    patch. The generator guarantees this by using solid rectangles without any
#    background holes, so bbox area equals rectangle area.


BG = 8
BLUE = 1
RED = 2
MARKER = 0
DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
MINORITY_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)
RECT_SPEC_CATALOG = (
    (2, 2),
    (2, 3),
    (2, 4),
    (2, 5),
    (3, 3),
    (3, 4),
    (3, 5),
)


def _scene_info(g: np.ndarray) -> dict:
    H, W = g.shape
    bg = int(get_bg(g))

    corners = [(0, 0), (0, int(W - 1)), (int(H - 1), 0), (int(H - 1), int(W - 1))]
    corner = None
    for r, c in corners:
        if int(g[r, c]) == int(bg):
            continue
        nbr = []
        if r == 0:
            nbr.append(int(g[r + 1, c]))
        if r == H - 1:
            nbr.append(int(g[r - 1, c]))
        if c == 0:
            nbr.append(int(g[r, c + 1]))
        if c == W - 1:
            nbr.append(int(g[r, c - 1]))
        if nbr and all(int(n) != int(g[r, c]) for n in nbr):
            corner = (int(r), int(c))
            break
    assert corner is not None

    cr, cc = map(int, corner)
    marker = int(g[cr, cc])
    hstep = 1 if cc == 0 else -1
    vstep = 1 if cr == 0 else -1
    hcol = int(g[cr, cc + hstep])
    vcol = int(g[cr + vstep, cc])

    if hcol == 1 or vcol == 1:
        blue_col = 1
        red_col = int(vcol if hcol == 1 else hcol)
    else:
        hlen = 0
        c = int(cc + hstep)
        while 0 <= c < W and int(g[cr, c]) == int(hcol):
            hlen += 1
            c += int(hstep)
        vlen = 0
        r = int(cr + vstep)
        while 0 <= r < H and int(g[r, cc]) == int(vcol):
            vlen += 1
            r += int(vstep)
        if vlen >= hlen:
            blue_col = int(vcol)
            red_col = int(hcol)
        else:
            blue_col = int(hcol)
            red_col = int(vcol)

    blue_orient = "h" if int(hcol) == int(blue_col) else "v"
    red_orient = "h" if int(hcol) == int(red_col) else "v"
    return {
        "bg": int(bg),
        "corner": (int(cr), int(cc)),
        "marker": int(marker),
        "blue_orient": blue_orient,
        "red_orient": red_orient,
        "blue_edge_idx": int(cc if blue_orient == "v" else cr),
        "red_edge_idx": int(cc if red_orient == "v" else cr),
    }


def _dist_to_edge(box: tuple[int, int, int, int], orient: str, edge_idx: int) -> int:
    r0, c0, r1, c1 = map(int, box)
    if orient == "v":
        col = int(edge_idx)
        if int(c0) > col:
            return int(c0 - col - 1)
        if int(c1) < col:
            return int(col - c1 - 1)
        return 0
    row = int(edge_idx)
    if int(r0) > row:
        return int(r0 - row - 1)
    if int(r1) < row:
        return int(row - r1 - 1)
    return 0


def _along_from_corner(box: tuple[int, int, int, int], orient: str, cr: int, cc: int) -> int:
    r0, c0, r1, c1 = map(int, box)
    if orient == "v":
        return int(r0 if cr == 0 else -r1)
    return int(c0 if cc == 0 else -c1)


def _extract_rects(g: np.ndarray, scene: dict) -> list[dict]:
    H, W = g.shape
    bg = int(scene["bg"])
    cr, cc = map(int, scene["corner"])
    marker = int(scene["marker"])

    line_mask = np.zeros((H, W), dtype=bool)
    line_mask[cr, :] = True
    line_mask[:, cc] = True

    mask = (g != int(bg)) & (~line_mask) & (g != int(marker))
    rects = []
    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        b = bbox_coords(coords)
        sub = crop_bbox(g, b)
        if bool((sub == int(bg)).any()):
            continue
        rects.append(
            {
                "box": tuple(map(int, b)),
                "arr": sub.copy(),
                "area": int(sub.shape[0] * sub.shape[1]),
            }
        )

    assert rects

    for rect in rects:
        rect["db"] = int(_dist_to_edge(rect["box"], str(scene["blue_orient"]), int(scene["blue_edge_idx"])))
        rect["dr"] = int(_dist_to_edge(rect["box"], str(scene["red_orient"]), int(scene["red_edge_idx"])))
        rect["ab"] = int(_along_from_corner(rect["box"], str(scene["blue_orient"]), cr, cc))
        rect["ar"] = int(_along_from_corner(rect["box"], str(scene["red_orient"]), cr, cc))
    return rects


def _assemble_output(rects: list[dict], bg: int, blue_orient: str) -> np.ndarray:
    seq = []
    for rect in rects:
        arr = rect["arr"]
        if blue_orient == "v":
            arr = np.rot90(arr, k=1)
        seq.append(arr)

    out_w = int(max(int(arr.shape[1]) for arr in seq))
    out_h = int(sum(int(arr.shape[0]) for arr in seq))
    out = np.full((out_h, out_w), int(bg), dtype=int)

    out_r = 0
    for arr in seq:
        h, w = map(int, arr.shape)
        out_c = int((out_w - w) // 2)
        paste(out, arr, int(out_r), int(out_c))
        out_r += int(h)
    return out


def _solve(grid: np.ndarray, mode: str) -> np.ndarray:
    g = np.array(grid, dtype=int)
    scene = _scene_info(g)
    rects = _extract_rects(g, scene)

    if mode == "variant":
        rects.sort(key=lambda r: (int(r["db"]), int(r["area"]), int(r["dr"]), int(r["ab"]), int(r["ar"])))
    elif mode == "legacy":
        rects.sort(key=lambda r: (int(r["db"]), int(r["dr"]), int(r["ab"]), int(r["ar"])))
    else:
        raise ValueError(f"unknown solve mode: {mode}")

    return _assemble_output(rects, int(scene["bg"]), str(scene["blue_orient"]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder and stack rectangles by proximity to the blue edge, then by area.

    Variant rule:
    - Primary key: distance to blue edge.
    - Secondary key: rectangle area.
    - Remaining keys: legacy spatial tie-breakers for determinism.
    - If the blue edge is vertical, rotate each rectangle 90 degrees CCW.
    """
    return _solve(grid, mode="variant")


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, mode="legacy")


def _transform(grid: np.ndarray, blue_orient: str, corner: str) -> np.ndarray:
    if blue_orient == "h":
        if corner == "tl":
            return grid
        if corner == "tr":
            return np.fliplr(grid)
        if corner == "bl":
            return np.flipud(grid)
        return np.rot90(grid, 2)

    base = np.rot90(grid, 1)
    if corner == "bl":
        return base
    if corner == "tl":
        return np.flipud(base)
    if corner == "tr":
        return np.rot90(grid, 3)
    return np.flipud(np.rot90(grid, 3))


def _sprinkle_minorities(patch: np.ndarray, dominant: int) -> np.ndarray:
    area = int(patch.size)
    if area < 8 or np.random.rand() < 0.6:
        return patch

    max_minor = min(3, area // 4)
    if max_minor < 2:
        return patch

    n_minor = int(np.random.randint(2, max_minor + 1))
    n_colors = int(np.random.randint(1, min(2, n_minor) + 1))
    minority_choices = [int(c) for c in MINORITY_COLORS if int(c) != dominant]
    colors = np.random.choice(minority_choices, size=n_colors, replace=False)
    cells = np.random.choice(area, size=n_minor, replace=False)
    flat = patch.reshape(-1)
    for idx, cell in enumerate(cells):
        flat[int(cell)] = int(colors[idx % n_colors])
    return patch


def _is_monotone(values: list[int]) -> bool:
    return values == sorted(values) or values == sorted(values, reverse=True)


def _pick_band_specs(count: int) -> list[tuple[int, int]]:
    for _ in range(200):
        idxs = np.random.choice(len(RECT_SPEC_CATALOG), size=count, replace=False)
        specs = [RECT_SPEC_CATALOG[int(i)] for i in idxs]
        perm = np.random.permutation(count)
        ordered = [specs[int(i)] for i in perm]
        areas = [int(h * w) for h, w in ordered]
        if not _is_monotone(areas):
            return ordered
    raise ValueError("failed to choose non-monotone band")


def _build_canonical() -> np.ndarray:
    band_counts = [int(np.random.randint(3, 5)), int(np.random.randint(3, 5))]
    bands = [_pick_band_specs(count) for count in band_counts]
    used_widths = [sum(int(w) for _h, w in band) + 2 * (len(band) - 1) for band in bands]

    trailing = int(np.random.randint(1, 5))
    long_dim = int(max(used_widths) + 2 + trailing)
    grid = np.full((10, long_dim), BG, dtype=int)
    grid[0, :] = BLUE
    grid[:, 0] = RED
    grid[0, 0] = MARKER

    for r0, band in zip((2, 6), bands):
        c = 2
        for h, w in band:
            dominant = int(np.random.choice(DOMINANT_COLORS))
            patch = np.full((int(h), int(w)), dominant, dtype=int)
            patch = _sprinkle_minorities(patch, dominant)
            paste(grid, patch, int(r0), int(c))
            c += int(w) + 2

    return grid


def _variant_metrics(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    scene = _scene_info(g)
    rects = _extract_rects(g, scene)
    old_order = sorted(rects, key=lambda r: (int(r["db"]), int(r["dr"]), int(r["ab"]), int(r["ar"])))
    band_areas: dict[int, list[int]] = {}
    for rect in old_order:
        band_areas.setdefault(int(rect["db"]), []).append(int(rect["area"]))
    return {
        "band_area_sequences_in_legacy_order": dict(sorted(band_areas.items())),
        "non_monotone_bands": sum(1 for vals in band_areas.values() if len(vals) >= 3 and not _is_monotone(vals)),
        "differs_from_legacy_output": not np.array_equal(main(g), _legacy_main(g)),
    }


def _gen_attempt() -> np.ndarray:
    canonical = _build_canonical()
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    grid = _transform(canonical, blue_orient, corner)

    metrics = _variant_metrics(grid)
    if int(metrics["non_monotone_bands"]) < 2:
        raise ValueError("band areas became monotone after transform")
    if not bool(metrics["differs_from_legacy_output"]):
        raise ValueError("degenerate sample matches legacy ordering")

    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        try:
            return _gen_attempt()
        except (AssertionError, IndexError, ValueError):
            continue
    raise RuntimeError("failed to generate input for 291dc1e1 D2 variant")
