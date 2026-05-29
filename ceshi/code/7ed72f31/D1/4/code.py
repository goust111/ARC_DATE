# Variant of 7ed72f31
# Dimension: D1 — 反射变换类型 (transform)
# Change: 根据红色标记的几何形状选择反射方式：红线（水平/垂直）触发轴对称镜像，孤立红像素触发点反射（180度旋转）... → 条件性变换：颜色决定反射类型

from __future__ import annotations

import random
import numpy as np
from scipy import ndimage

from arc_utils import STRUCT8, get_bg, label_mask

"""
Design analysis for the D1 variant
==================================

variant_spec:
  core_change:
    old_rule: red marker geometry chose the transform
    new_rule: payload color category chooses the transform
    warm_colors: [4, 6, 7]   # yellow, pink/magenta, orange
    cool_colors: [1, 3, 8]   # blue, green, teal/cyan
    warm_transform: mirror
    cool_transform: point_reflect
  geometry_role_after_change:
    - geometry no longer selects mirror vs point
    - geometry still provides the locus of the transform:
      row-line -> horizontal axis line
      col-line -> vertical axis line
      single red point -> pivot point; if warm, the payload side relative to that point
      determines whether the mirror axis is the point's row or its column
  concrete_pixel_behavior:
    - embedded warm line:
      input has a monochrome payload touching a red row/col line; output mirrors it
      across that same line
    - embedded cool line:
      input still has the same red line geometry, but output is a 180-degree rotation
      around the middle red pixel of the line, not a mirror
    - isolated cool point:
      same as the original isolated-red behavior: point reflection around the red pixel
    - isolated warm point:
      input has a single red pixel diagonally adjacent to one monochrome component;
      if that component sits wholly above/below the red point, mirror across the red
      point's row; if wholly left/right, mirror across the red point's column
  behavior_trace:
    original_embedded_cool_example:
      input: a blue payload above a horizontal red line
      old_output: payload copied below the line with only top/bottom flipped
      new_output: payload copied below the line with both row+column reversed because
        blue is cool and the center is the middle red pixel
      difference: the reflected half changes left-right order, so asymmetric shapes
        visibly differ from the original task
    original_isolated_warm_example:
      input: an orange component diagonally above a lone red point
      old_output: point reflection to the opposite diagonal quadrant
      new_output: mirror across the red point's row, so the copy appears below but
        keeps its left/right arrangement
      difference: same marker geometry, different transform chosen by color
  complexity_calibration:
    - only two transform classes exist: warm=mirror, cool=point
    - generator always includes all four combinations:
      embedded+warm, embedded+cool, isolated+warm, isolated+cool
    - this prevents degeneration back to geometry-only reasoning and gives repeated
      evidence that color class, not marker shape, selects the transform
    - payload masks are rejected if mirror and point outputs would coincide, so the
      D1 change stays visually observable
  implicit_assumption_audit:
    - original main() assumed "red line => mirror" and "single red => point"
      This is invalid now.
    - original embedded loop also tolerated any non-red set inside a component.
      The variant requires monochrome payload components because color category drives
      the transform. The generator guarantees one payload color per sprite, and the
      solver skips ambiguous multi-color marker-components.
    - original isolated-red branch reflected every adjacent color component by point.
      This is invalid for warm colors. The new solver must inspect the adjacent
      component's color first, then choose mirror or point.
    - warm isolated-point mirroring needs an axis orientation even though the marker
      is just one pixel. The new solver therefore assumes the adjacent component lies
      wholly on one side of the red point in either rows or columns. The generator
      enforces exactly that.
    - cool embedded-line point reflection needs a unique integer center. The new
      generator therefore uses odd-length red lines only, so every line has a middle
      red pixel.
  placement_legality:
    - output-first generation is retained
    - each sprite is built as (original payload + red marker + transformed copy)
    - footprints are placed onto background-only regions with 1-cell 8-neighborhood
      padding, preventing accidental component merges
    - all transformed copies are constructed inside local patches first, so no sprite
      self-overlaps and all placements stay within the grid
  visual_observability:
    - asymmetric payloads are required whenever mirror and point could otherwise look
      identical
    - each grid contains both warm and cool sprites and both marker geometries, so a
      solver must jointly reason about visible color class and geometry
  cleanliness:
    - sprites stay small and separated
    - no extra decorative markers are added
    - red remains the only marker color, preserving the original task's visual style
"""

MARK = 2
WARM_COLORS = {4, 6, 7}
COOL_COLORS = {1, 3, 8}

# Canvas
GRID_MIN, GRID_MAX = 18, 28

# Marker counts
TOTAL_MARKERS_MIN, TOTAL_MARKERS_MAX = 4, 7
MAX_EMBEDDED, MAX_ISOLATED = 4, 4

# Embedded payload shape
PAYLOAD_SIDE_MIN, PAYLOAD_SIDE_MAX = 2, 6
PAYLOAD_PX_MIN, PAYLOAD_PX_MAX = 3, 11
PAYLOAD_DENSITY_LO, PAYLOAD_DENSITY_HI = 0.50, 0.80
EMBEDDED_AXIS_LENGTHS = (3, 5)  # odd so cool line sprites have a unique center

# Isolated payload
ISOLATED_PX_MIN, ISOLATED_PX_MAX = 2, 8
ISOLATED_BBOX_MAX = 4

# Color palette
PALETTE_COLORS_MIN, PALETTE_COLORS_MAX = 3, 4

# Layout
PLACEMENT_PADDING = 1
BG_COVERAGE_MIN = 0.88

# Retries
MASK_RETRIES = 50
GEN_RETRIES = 400

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def refl_point(center: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    cr, cc = map(int, center)
    return int(2 * cr - r), int(2 * cc - c)


def refl_row(r0: int, r: int, c: int) -> tuple[int, int]:
    return int(2 * int(r0) - int(r)), int(c)


def refl_col(c0: int, r: int, c: int) -> tuple[int, int]:
    return int(r), int(2 * int(c0) - int(c))


def _random_connected_mask(h: int, w: int, target_pixels: int, *, rng: random.Random) -> np.ndarray | None:
    if target_pixels < 1 or target_pixels > h * w:
        return None
    mask = np.zeros((h, w), dtype=bool)
    mask[rng.randrange(h), rng.randrange(w)] = True
    count = 1
    while count < target_pixels:
        frontier: list[tuple[int, int]] = []
        for r in range(h):
            for c in range(w):
                if mask[r, c]:
                    continue
                for dr, dc in _DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                        frontier.append((r, c))
                        break
        if not frontier:
            return None
        r, c = rng.choice(frontier)
        mask[r, c] = True
        count += 1
    if not mask.any(axis=1).all() or not mask.any(axis=0).all():
        return None
    return mask


def _coords_to_sprite(
    orig_coords: list[tuple[int, int]],
    extra_coords: list[tuple[int, int]],
    red_coords: list[tuple[int, int]],
) -> dict | None:
    orig_set = {(int(r), int(c)) for r, c in orig_coords}
    extra_set = {(int(r), int(c)) for r, c in extra_coords}
    red_set = {(int(r), int(c)) for r, c in red_coords}
    if not orig_set or not red_set:
        return None
    if orig_set & extra_set or orig_set & red_set or extra_set & red_set:
        return None
    pts = list(orig_set | extra_set | red_set)
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    minr, maxr = min(rs), max(rs)
    minc, maxc = min(cs), max(cs)
    h = int(maxr - minr + 1)
    w = int(maxc - minc + 1)
    orig = np.zeros((h, w), dtype=bool)
    extra = np.zeros((h, w), dtype=bool)
    reds: list[tuple[int, int]] = []
    for r, c in orig_set:
        orig[int(r - minr), int(c - minc)] = True
    for r, c in extra_set:
        extra[int(r - minr), int(c - minc)] = True
    for r, c in sorted(red_set):
        nr, nc = int(r - minr), int(c - minc)
        reds.append((nr, nc))
    return {"orig": orig, "extra": extra, "red_pixels": reds}


def _marker_geometry(reds: list[tuple[int, int]]) -> dict | None:
    if len(reds) == 1:
        return {"kind": "point", "point": (int(reds[0][0]), int(reds[0][1]))}

    rs = sorted({int(r) for r, _ in reds})
    cs = sorted({int(c) for _, c in reds})
    if len(rs) == 1:
        row = int(rs[0])
        cols = sorted(int(c) for _, c in reds)
        if cols != list(range(cols[0], cols[-1] + 1)):
            return None
        center = (row, cols[len(cols) // 2]) if len(cols) % 2 == 1 else None
        return {"kind": "row", "axis": row, "center": center}
    if len(cs) == 1:
        col = int(cs[0])
        rows = sorted(int(r) for r, _ in reds)
        if rows != list(range(rows[0], rows[-1] + 1)):
            return None
        center = (rows[len(rows) // 2], col) if len(rows) % 2 == 1 else None
        return {"kind": "col", "axis": col, "center": center}
    return None


def _infer_point_mirror_axis(point: tuple[int, int], payload_coords: list[tuple[int, int]]) -> tuple[str, int] | None:
    pr, pc = map(int, point)
    rows = [int(r) for r, _ in payload_coords]
    cols = [int(c) for _, c in payload_coords]
    if max(rows) < pr or min(rows) > pr:
        return ("row", pr)
    if max(cols) < pc or min(cols) > pc:
        return ("col", pc)
    return None


def _apply_payload_transform(
    payload_coords: list[tuple[int, int]],
    *,
    transform: str,
    geometry: dict,
) -> list[tuple[int, int]] | None:
    if transform == "mirror":
        if geometry["kind"] == "row":
            return [refl_row(int(geometry["axis"]), int(r), int(c)) for r, c in payload_coords]
        if geometry["kind"] == "col":
            return [refl_col(int(geometry["axis"]), int(r), int(c)) for r, c in payload_coords]
        if geometry["kind"] == "point":
            axis = _infer_point_mirror_axis(tuple(geometry["point"]), payload_coords)
            if axis is None:
                return None
            if axis[0] == "row":
                return [refl_row(int(axis[1]), int(r), int(c)) for r, c in payload_coords]
            return [refl_col(int(axis[1]), int(r), int(c)) for r, c in payload_coords]
        return None

    if geometry["kind"] == "row":
        center = geometry.get("center")
        if center is None:
            return None
        return [refl_point(tuple(center), int(r), int(c)) for r, c in payload_coords]
    if geometry["kind"] == "col":
        center = geometry.get("center")
        if center is None:
            return None
        return [refl_point(tuple(center), int(r), int(c)) for r, c in payload_coords]
    if geometry["kind"] == "point":
        return [refl_point(tuple(geometry["point"]), int(r), int(c)) for r, c in payload_coords]
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - Warm payload colors (4/6/7) mirror.
    - Cool payload colors (1/3/8) point-reflect.
    - Red geometry still locates the axis/pivot, but it no longer chooses the transform type.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    def inb(r: int, c: int) -> bool:
        return 0 <= int(r) < H and 0 <= int(c) < W

    out = g.copy()

    lab, n = label_mask(g != int(bg), connectivity=4)
    comps = []
    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if not coords:
            continue
        vals = [int(g[r, c]) for r, c in coords]
        comps.append({"coords": coords, "vals": vals})

    payload_lab, payload_n = label_mask((g != int(bg)) & (g != int(MARK)), connectivity=4)
    payload_comps: dict[int, dict] = {}
    for k in range(1, int(payload_n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(payload_lab == int(k))]
        if not coords:
            continue
        colors = {int(g[r, c]) for r, c in coords}
        if len(colors) != 1:
            continue
        payload_comps[int(k)] = {"coords": coords, "color": int(next(iter(colors)))}

    # Marker-containing mixed components.
    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        if int(MARK) not in vals:
            continue
        if len(coords) == 1 and int(vals[0]) == int(MARK):
            continue

        reds = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) == int(MARK)]
        payload_coords = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) != int(MARK)]
        payload_colors = {int(v) for v in vals if int(v) != int(MARK)}
        if not reds or not payload_coords or len(payload_colors) != 1:
            continue

        geometry = _marker_geometry(reds)
        if geometry is None:
            continue

        color = int(next(iter(payload_colors)))
        if color in WARM_COLORS:
            transform = "mirror"
        elif color in COOL_COLORS:
            transform = "point"
        else:
            continue

        targets = _apply_payload_transform(payload_coords, transform=transform, geometry=geometry)
        if targets is None:
            continue
        for (rr, cc), (_, _c) in zip(targets, payload_coords):
            if inb(int(rr), int(cc)):
                out[int(rr), int(cc)] = int(color)

    # Isolated red pixels with an adjacent payload component.
    for comp in comps:
        if len(comp["coords"]) != 1 or int(comp["vals"][0]) != int(MARK):
            continue
        pr, pc = map(int, comp["coords"][0])

        neigh_labels: set[int] = set()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = int(pr + dr), int(pc + dc)
                if not inb(rr, cc):
                    continue
                lbl = int(payload_lab[rr, cc])
                if lbl != 0:
                    neigh_labels.add(lbl)

        for lbl in sorted(neigh_labels):
            info = payload_comps.get(int(lbl))
            if info is None:
                continue
            color = int(info["color"])
            payload_coords = info["coords"]
            if color in WARM_COLORS:
                transform = "mirror"
            elif color in COOL_COLORS:
                transform = "point"
            else:
                continue

            geometry = {"kind": "point", "point": (int(pr), int(pc))}
            targets = _apply_payload_transform(payload_coords, transform=transform, geometry=geometry)
            if targets is None:
                continue
            for rr, cc in targets:
                if inb(int(rr), int(cc)):
                    out[int(rr), int(cc)] = int(color)

    return out


def _build_embedded_sprite(rng: random.Random, *, transform: str) -> dict | None:
    for _ in range(MASK_RETRIES):
        axis_len = int(rng.choice(EMBEDDED_AXIS_LENGTHS))
        depth = rng.randint(PAYLOAD_SIDE_MIN, PAYLOAD_SIDE_MAX)
        if rng.random() < 0.5:
            h, w = depth, axis_len
            is_row = True
        else:
            h, w = axis_len, depth
            is_row = False

        max_px = min(PAYLOAD_PX_MAX, int(PAYLOAD_DENSITY_HI * h * w))
        min_px = max(PAYLOAD_PX_MIN, int(np.ceil(PAYLOAD_DENSITY_LO * h * w)))
        if min_px > max_px:
            continue
        pmask = _random_connected_mask(h, w, rng.randint(min_px, max_px), rng=rng)
        if pmask is None:
            continue

        side_flip = rng.random() < 0.5
        orig_coords: list[tuple[int, int]] = []
        red_coords: list[tuple[int, int]] = []
        if is_row:
            for r, c in np.argwhere(pmask):
                rr = int(r if not side_flip else r + h + 1)
                orig_coords.append((rr, int(c)))
            red_coords = [(h, c) for c in range(w)]
            geometry = {"kind": "row", "axis": h, "center": (h, w // 2)}
        else:
            for r, c in np.argwhere(pmask):
                cc = int(c if not side_flip else c + w + 1)
                orig_coords.append((int(r), cc))
            red_coords = [(r, w) for r in range(h)]
            geometry = {"kind": "col", "axis": w, "center": (h // 2, w)}

        extra_coords = _apply_payload_transform(orig_coords, transform=transform, geometry=geometry)
        alt_transform = "point" if transform == "mirror" else "mirror"
        alt_coords = _apply_payload_transform(orig_coords, transform=alt_transform, geometry=geometry)
        if extra_coords is None or alt_coords is None:
            continue
        if set(extra_coords) == set(alt_coords):
            continue

        sprite = _coords_to_sprite(orig_coords, extra_coords, red_coords)
        if sprite is not None:
            sprite["geometry_type"] = "embedded"
            sprite["transform"] = transform
            return sprite
    return None


def _build_isolated_sprite(rng: random.Random, *, transform: str) -> dict | None:
    for _ in range(MASK_RETRIES):
        n_px = rng.randint(ISOLATED_PX_MIN, ISOLATED_PX_MAX)
        ph = rng.randint(1, min(ISOLATED_BBOX_MAX, max(1, n_px)))
        pw = rng.randint(1, min(ISOLATED_BBOX_MAX, max(1, n_px)))
        if ph * pw < n_px or n_px < max(ph, pw):
            continue
        if n_px == ph * pw:
            pmask = np.ones((ph, pw), dtype=bool)
        else:
            pmask = _random_connected_mask(ph, pw, n_px, rng=rng)
        if pmask is None:
            continue

        pcoords = [(int(r), int(c)) for r, c in np.argwhere(pmask)]
        pset = set(pcoords)
        minr, maxr = min(r for r, _ in pcoords), max(r for r, _ in pcoords)
        minc, maxc = min(c for _, c in pcoords), max(c for _, c in pcoords)

        axis_kind = rng.choice(["row", "col"])
        candidates: list[dict] = []
        for red_r in range(-1, ph + 1):
            for red_c in range(-1, pw + 1):
                if (red_r, red_c) in pset:
                    continue
                has8 = any(max(abs(red_r - r), abs(red_c - c)) == 1 for r, c in pcoords)
                has4 = any(abs(red_r - r) + abs(red_c - c) == 1 for r, c in pcoords)
                if not has8 or has4:
                    continue
                if axis_kind == "row" and not (maxr < red_r or minr > red_r):
                    continue
                if axis_kind == "col" and not (maxc < red_c or minc > red_c):
                    continue

                geometry = {"kind": "point", "point": (int(red_r), int(red_c))}
                if transform == "mirror":
                    if axis_kind == "row":
                        extra_coords = [refl_row(int(red_r), int(r), int(c)) for r, c in pcoords]
                    else:
                        extra_coords = [refl_col(int(red_c), int(r), int(c)) for r, c in pcoords]
                    alt_coords = [refl_point((int(red_r), int(red_c)), int(r), int(c)) for r, c in pcoords]
                else:
                    extra_coords = [refl_point((int(red_r), int(red_c)), int(r), int(c)) for r, c in pcoords]
                    if axis_kind == "row":
                        alt_coords = [refl_row(int(red_r), int(r), int(c)) for r, c in pcoords]
                    else:
                        alt_coords = [refl_col(int(red_c), int(r), int(c)) for r, c in pcoords]

                if set(extra_coords) == set(alt_coords):
                    continue
                if set(extra_coords) & pset:
                    continue
                if (red_r, red_c) in set(extra_coords):
                    continue
                sprite = _coords_to_sprite(pcoords, extra_coords, [(red_r, red_c)])
                if sprite is None:
                    continue
                candidates.append({"sprite": sprite, "geometry": geometry, "axis_kind": axis_kind})

        if candidates:
            pick = rng.choice(candidates)
            sprite = pick["sprite"]
            sprite["geometry_type"] = "isolated"
            sprite["transform"] = transform
            sprite["axis_kind"] = pick["axis_kind"]
            return sprite
    return None


def _random_free_location(
    canvas: np.ndarray,
    patch_mask: np.ndarray,
    bg: int,
    rng: random.Random,
    *,
    padding: int = PLACEMENT_PADDING,
):
    H, W = canvas.shape
    h, w = patch_mask.shape
    if h > H or w > W:
        return None
    occ = canvas != bg
    if padding > 0:
        occ = ndimage.binary_dilation(occ, structure=STRUCT8, iterations=padding)
    cands = []
    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            if not (occ[r0 : r0 + h, c0 : c0 + w] & patch_mask).any():
                cands.append((r0, c0))
    if not cands:
        return None
    return rng.choice(cands)


def _stamp_sprite(canvas: np.ndarray, spr: dict, r0: int, c0: int) -> None:
    h, w = spr["orig"].shape
    canvas[r0 : r0 + h, c0 : c0 + w][spr["orig"] | spr["extra"]] = int(spr["color"])
    for rr, cc in spr["red_pixels"]:
        canvas[int(r0 + rr), int(c0 + cc)] = int(MARK)


def _erase_extra(canvas: np.ndarray, spr: dict, bg: int) -> None:
    r0, c0 = int(spr["r0"]), int(spr["c0"])
    h, w = spr["extra"].shape
    canvas[r0 : r0 + h, c0 : c0 + w][spr["extra"]] = int(bg)


def _analyze_input(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    lab, n = label_mask(g != int(bg), connectivity=4)
    stats = {"embedded_warm": 0, "embedded_cool": 0, "isolated_warm": 0, "isolated_cool": 0}

    payload_lab, payload_n = label_mask((g != int(bg)) & (g != int(MARK)), connectivity=4)
    payload_color = {}
    for k in range(1, int(payload_n) + 1):
        coords = np.argwhere(payload_lab == int(k))
        if coords.size == 0:
            continue
        colors = {int(g[r, c]) for r, c in coords}
        if len(colors) == 1:
            payload_color[int(k)] = int(next(iter(colors)))

    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        vals = [int(g[r, c]) for r, c in coords]
        if int(MARK) not in vals:
            continue
        if len(coords) == 1 and int(vals[0]) == int(MARK):
            pr, pc = coords[0]
            colors = set()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = pr + dr, pc + dc
                    if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]:
                        lbl = int(payload_lab[rr, cc])
                        if lbl in payload_color:
                            colors.add(payload_color[lbl])
            for color in colors:
                if color in WARM_COLORS:
                    stats["isolated_warm"] += 1
                elif color in COOL_COLORS:
                    stats["isolated_cool"] += 1
            continue

        payload_colors = {int(v) for v in vals if int(v) != int(MARK)}
        if len(payload_colors) != 1:
            continue
        color = int(next(iter(payload_colors)))
        if color in WARM_COLORS:
            stats["embedded_warm"] += 1
        elif color in COOL_COLORS:
            stats["embedded_cool"] += 1
    return stats


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    N = rng.randint(GRID_MIN, GRID_MAX)
    bg = rng.choice([c for c in range(10) if c != MARK])

    roles = [
        ("embedded", "warm"),
        ("embedded", "cool"),
        ("isolated", "warm"),
        ("isolated", "cool"),
    ]
    total_markers = rng.randint(TOTAL_MARKERS_MIN, TOTAL_MARKERS_MAX)
    n_embedded = 2
    n_isolated = 2
    while len(roles) < total_markers:
        cands = []
        if n_embedded < MAX_EMBEDDED:
            cands.extend([("embedded", "warm"), ("embedded", "cool")])
        if n_isolated < MAX_ISOLATED:
            cands.extend([("isolated", "warm"), ("isolated", "cool")])
        if not cands:
            break
        role = rng.choice(cands)
        roles.append(role)
        if role[0] == "embedded":
            n_embedded += 1
        else:
            n_isolated += 1

    sprites: list[dict] = []
    for geometry_type, temp in roles:
        transform = "mirror" if temp == "warm" else "point"
        if geometry_type == "embedded":
            spr = _build_embedded_sprite(rng, transform=transform)
        else:
            spr = _build_isolated_sprite(rng, transform=transform)
        if spr is None:
            return None
        spr["temp"] = temp
        sprites.append(spr)

    warm_choices = [c for c in sorted(WARM_COLORS) if c != bg]
    cool_choices = [c for c in sorted(COOL_COLORS) if c != bg]

    n_colors = rng.randint(PALETTE_COLORS_MIN, PALETTE_COLORS_MAX)
    if n_colors == 3:
        if rng.random() < 0.5:
            chosen_warm = rng.sample(warm_choices, 2)
            chosen_cool = rng.sample(cool_choices, 1)
        else:
            chosen_warm = rng.sample(warm_choices, 1)
            chosen_cool = rng.sample(cool_choices, 2)
    else:
        chosen_warm = rng.sample(warm_choices, 2)
        chosen_cool = rng.sample(cool_choices, 2)

    for spr in sprites:
        if spr["temp"] == "warm":
            spr["color"] = int(rng.choice(chosen_warm))
        else:
            spr["color"] = int(rng.choice(chosen_cool))

    output = np.full((N, N), bg, dtype=int)
    rng.shuffle(sprites)
    for spr in sprites:
        footprint = spr["orig"] | spr["extra"]
        for r, c in spr["red_pixels"]:
            footprint[int(r), int(c)] = True
        loc = _random_free_location(output, footprint, bg, rng)
        if loc is None:
            return None
        r0, c0 = map(int, loc)
        spr["r0"], spr["c0"] = r0, c0
        _stamp_sprite(output, spr, r0, c0)

    inp = output.copy()
    for spr in sprites:
        _erase_extra(inp, spr, bg)

    if (inp == bg).sum() / (N * N) < BG_COVERAGE_MIN:
        return None

    stats = _analyze_input(inp)
    if min(stats.values()) < 1:
        return None

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
