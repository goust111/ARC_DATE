# Variant of 7ed72f31
# Dimension: D14 — 反射结果的边界与碰撞处理 (transform)
# Change: 所有反射必须完全落在网格边界内，且仅落在背景格上（无重叠）... → 允许反射超出边界（截断）

"""
Design analysis for the D14 clipping variant.

a) Concrete behavior and trace
The objects are still the same as the source task:
- embedded marker sprites: a monochrome payload attached to a full red row/col axis;
- isolated red markers: a single red pixel point-reflects one adjacent monochrome component.

The only changed behavior is what happens after the reflection target is computed.
Original generator design forced every reflected pixel to stay inside the grid, so the
`if inb(...)` guards inside `main()` were effectively dormant. In this variant, the input
side remains fully inside the grid, but the reflected side may extend past a border.
Pixels whose reflected coordinates are outside the grid are dropped, while the in-bounds
part is still drawn.

Concrete trace:
- Suppose a horizontal red axis is on row 16 of an 18x18 grid and a payload pixel is at
  (15, 5). Reflection sends it to (17, 5), which survives.
- Another payload pixel at (14, 6) reflects to (18, 6), which is outside the grid and is
  clipped away.
Original task generation would reject that entire placement; this variant keeps the sprite
and produces only the surviving subset of the mirror.

b) Complexity calibration
This is intentionally a narrow D14 change. Axis detection, component semantics, colors,
counts, and connectivity remain the same. To make clipping visually observable, each
generated grid contains both clipped and unclipped examples, and clipped examples are
required to be partial: at least one reflected pixel survives and at least one is lost.
That keeps the task learnable from a few examples instead of degenerating into either
"everything still fits" or "the whole reflection disappears".

c) Implicit-assumption audit of `main()`
Relevant source-task assumptions:
- Reflection target computation (`refl_row/refl_col/refl_point`) is still valid.
- `if inb(rr, cc)` was previously only a safety check; now it defines the new rule.
- Unconditional writes to `out[rr, cc]` relied on the old generator guarantee that every
  reflected target lands on background. That guarantee should remain true for visible
  in-bounds targets, so this variant makes the non-overlap rule explicit with a bg check.
- Component parsing and axis classification still assume the input-side objects are fully
  present in-bounds; the generator preserves that.

d) Placement legality
The generator no longer places a full symmetric output patch. Instead, it places only the
input-side sprite bbox in-bounds, then computes the reflected targets globally and keeps
only their in-bounds subset. Placement tests use the visible output footprint
(input payload + red markers + unclipped mirror pixels), so no visible overlap occurs and
separate sprites stay padded apart. This preserves solver readability while allowing the
theoretical mirror to extend outside the canvas.

e) Visual observability
At least one sprite per grid must clip, and the rest are forced to fit completely. That
contrast makes the new rule legible: some mirrors are full, some are truncated exactly at
the border, with no color remapping or hidden metadata.

f) Visual cleanliness
The visible output footprint is still non-overlapping and padded by at least one cell from
other sprites, so clipping does not create tangled merged components. Red markers remain
the only reflection cues.

g) Color inference
No color transformation is introduced. Payload colors remain unchanged before/after
reflection, so the only new thing to infer is boundary clipping.
"""

from __future__ import annotations

import random

import numpy as np
from scipy import ndimage

from arc_utils import STRUCT8, get_bg, label_mask


MARK = 2

# Canvas
GRID_MIN, GRID_MAX = 18, 28

# Marker counts
TOTAL_MARKERS_MIN, TOTAL_MARKERS_MAX = 3, 7
MAX_EMBEDDED, MAX_ISOLATED = 4, 4

# Embedded payload shape
PAYLOAD_SIDE_MIN, PAYLOAD_SIDE_MAX = 2, 6
PAYLOAD_PX_MIN, PAYLOAD_PX_MAX = 3, 11
PAYLOAD_DENSITY_LO, PAYLOAD_DENSITY_HI = 0.50, 0.80

# Isolated payload
ISOLATED_PX_MIN, ISOLATED_PX_MAX = 1, 8
ISOLATED_BBOX_MAX = 4

# Color palette
PALETTE_COLORS_MIN, PALETTE_COLORS_MAX = 3, 4

# Layout
PLACEMENT_PADDING = 1
BG_COVERAGE_MIN = 0.88
MIN_REQUIRED_CLIPPED = 1
MAX_REQUIRED_CLIPPED = 2

# Retries
MASK_RETRIES = 40
GEN_RETRIES = 300

_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _inb(shape: tuple[int, int], r: int, c: int) -> bool:
    H, W = shape
    return 0 <= int(r) < int(H) and 0 <= int(c) < int(W)


def _refl_point(p: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    pr, pc = map(int, p)
    return int(2 * pr - r), int(2 * pc - c)


def _refl_row(r0: int, r: int, c: int) -> tuple[int, int]:
    return int(2 * int(r0) - int(r)), int(c)


def _refl_col(c0: int, r: int, c: int) -> tuple[int, int]:
    return int(r), int(2 * int(c0) - int(c))


def _extract_components(g: np.ndarray, bg: int) -> list[dict]:
    lab, n = label_mask(g != int(bg), connectivity=4)
    comps = []
    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if not coords:
            continue
        vals = [int(g[r, c]) for r, c in coords]
        rs = [r for r, _ in coords]
        cs = [c for _, c in coords]
        comps.append(
            {
                "coords": coords,
                "vals": vals,
                "bbox": (int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))),
            }
        )
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror colored sprites using red (2) markers as reflection axes.

    D14 variant:
    - Reflected pixels are computed exactly as in the source task.
    - If a reflected pixel falls outside the grid, it is clipped away.
    - Visible reflected pixels are only painted onto background cells.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape
    comps = _extract_components(g, bg)
    out = g.copy()

    def paint(rr: int, cc: int, v: int) -> None:
        if _inb((H, W), int(rr), int(cc)) and int(out[int(rr), int(cc)]) == int(bg):
            out[int(rr), int(cc)] = int(v)

    # 1) Marker-containing mixed components.
    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        if int(MARK) not in set(map(int, vals)):
            continue

        reds = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) == int(MARK)]
        if not reds:
            continue

        rs = {int(r) for r, _ in reds}
        cs = {int(c) for _, c in reds}
        if len(reds) == 1:
            axis = ("point", (int(reds[0][0]), int(reds[0][1])))
        elif len(rs) == 1:
            axis = ("row", int(next(iter(rs))))
        elif len(cs) == 1:
            axis = ("col", int(next(iter(cs))))
        else:
            continue

        r0, c0, r1, c1 = map(int, comp["bbox"])
        payload = {int(v) for v in vals if int(v) != int(MARK)}
        if not payload:
            continue

        if axis[0] == "row":

            def eligible(r: int, c: int, v: int) -> bool:
                return int(v) in payload and int(c0) <= int(c) <= int(c1)

        elif axis[0] == "col":

            def eligible(r: int, c: int, v: int) -> bool:
                return int(v) in payload and int(r0) <= int(r) <= int(r1)

        else:
            rr0 = max(0, int(r0 - 1))
            rr1 = min(int(H - 1), int(r1 + 1))
            cc0 = max(0, int(c0 - 1))
            cc1 = min(int(W - 1), int(c1 + 1))

            def eligible(r: int, c: int, v: int) -> bool:
                return (
                    int(v) in payload
                    and int(rr0) <= int(r) <= int(rr1)
                    and int(cc0) <= int(c) <= int(cc1)
                )

        for r in range(int(H)):
            for c in range(int(W)):
                v = int(g[int(r), int(c)])
                if v == int(bg) or v == int(MARK):
                    continue
                if not eligible(int(r), int(c), int(v)):
                    continue

                if axis[0] == "row":
                    rr, cc = _refl_row(int(axis[1]), int(r), int(c))
                elif axis[0] == "col":
                    rr, cc = _refl_col(int(axis[1]), int(r), int(c))
                else:
                    rr, cc = _refl_point(axis[1], int(r), int(c))

                paint(int(rr), int(cc), int(v))

    # 2) Isolated red points reflect adjacent monochrome 8-components.
    marker_pts = []
    for comp in comps:
        if len(comp["coords"]) == 1 and int(comp["vals"][0]) == int(MARK):
            marker_pts.append(comp["coords"][0])

    for pr, pc in marker_pts:
        pr = int(pr)
        pc = int(pc)
        neigh_colors: set[int] = set()
        neigh_pixels: list[tuple[int, int, int]] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = int(pr + dr), int(pc + dc)
                if not _inb((H, W), rr, cc):
                    continue
                v = int(g[rr, cc])
                if v == int(bg) or v == int(MARK):
                    continue
                neigh_colors.add(int(v))
                neigh_pixels.append((int(rr), int(cc), int(v)))
        if not neigh_colors:
            continue

        for v in neigh_colors:
            labv, _ = label_mask(g == int(v), connectivity=8)
            labs = {
                int(labv[int(rr), int(cc)])
                for rr, cc, vv in neigh_pixels
                if int(vv) == int(v) and int(labv[int(rr), int(cc)]) != 0
            }
            for lbl in labs:
                for r, c in np.argwhere(labv == int(lbl)):
                    rr, cc = _refl_point((int(pr), int(pc)), int(r), int(c))
                    paint(int(rr), int(cc), int(v))

    return out


def _random_connected_mask(h: int, w: int, target_pixels: int, *, rng: random.Random) -> np.ndarray | None:
    if target_pixels < 1 or target_pixels > h * w:
        return None
    mask = np.zeros((h, w), dtype=bool)
    mask[rng.randrange(h), rng.randrange(w)] = True
    count = 1
    while count < target_pixels:
        frontier = []
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


def _build_embedded_sprite(rng: random.Random) -> dict | None:
    ph = rng.randint(PAYLOAD_SIDE_MIN, PAYLOAD_SIDE_MAX)
    pw = rng.randint(PAYLOAD_SIDE_MIN, PAYLOAD_SIDE_MAX)
    max_px = min(PAYLOAD_PX_MAX, int(PAYLOAD_DENSITY_HI * ph * pw))
    min_px = max(PAYLOAD_PX_MIN, int(np.ceil(PAYLOAD_DENSITY_LO * ph * pw)))
    if min_px > max_px:
        return None
    pmask = _random_connected_mask(ph, pw, rng.randint(min_px, max_px), rng=rng)
    if pmask is None:
        return None

    is_row = rng.choice([True, False])
    payload_first = rng.random() < 0.5
    payload_coords = []
    red_pixels = []

    if is_row:
        shape = (ph + 1, pw)
        if payload_first:
            payload_coords = [(int(r), int(c)) for r, c in np.argwhere(pmask)]
            red_pixels = [(ph, int(c)) for c in range(pw)]
        else:
            payload_coords = [(int(r + 1), int(c)) for r, c in np.argwhere(pmask)]
            red_pixels = [(0, int(c)) for c in range(pw)]
        axis = ("row", int(red_pixels[0][0]))
    else:
        shape = (ph, pw + 1)
        if payload_first:
            payload_coords = [(int(r), int(c)) for r, c in np.argwhere(pmask)]
            red_pixels = [(int(r), pw) for r in range(ph)]
        else:
            payload_coords = [(int(r), int(c + 1)) for r, c in np.argwhere(pmask)]
            red_pixels = [(int(r), 0) for r in range(ph)]
        axis = ("col", int(red_pixels[0][1]))

    return {
        "kind": "embedded",
        "shape": shape,
        "payload_coords": payload_coords,
        "red_pixels": red_pixels,
        "axis": axis,
    }


def _is_4connected_to(mask: np.ndarray, r: int, c: int) -> bool:
    H, W = mask.shape
    for dr, dc in _DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W and mask[nr, nc]:
            return True
    return False


def _build_isolated_sprite(rng: random.Random) -> dict | None:
    n_px = rng.randint(ISOLATED_PX_MIN, ISOLATED_PX_MAX)
    pmask = None
    for _ in range(MASK_RETRIES):
        ph = rng.randint(1, min(ISOLATED_BBOX_MAX, max(1, n_px)))
        pw = rng.randint(1, min(ISOLATED_BBOX_MAX, max(1, n_px)))
        if ph * pw < n_px or n_px < max(ph, pw):
            continue
        pmask = np.ones((ph, pw), dtype=bool) if n_px == ph * pw else _random_connected_mask(ph, pw, n_px, rng=rng)
        if pmask is not None:
            break
    if pmask is None:
        return None

    pcoords = [(int(r), int(c)) for r, c in np.argwhere(pmask)]
    rng.shuffle(pcoords)
    pset = set(pcoords)

    for pr, pc in pcoords:
        dirs = list(_DIRS8)
        rng.shuffle(dirs)
        for dr, dc in dirs:
            red_r, red_c = int(pr + dr), int(pc + dc)
            if (red_r, red_c) in pset:
                continue
            refl = [(int(2 * red_r - r), int(2 * red_c - c)) for r, c in pcoords]
            if set(refl) & pset:
                continue
            all_pts = pcoords + [(red_r, red_c)]
            all_rs = [r for r, _ in all_pts]
            all_cs = [c for _, c in all_pts]
            minr, minc = min(all_rs), min(all_cs)
            H = max(all_rs) - minr + 1
            W = max(all_cs) - minc + 1
            payload_coords = [(int(r - minr), int(c - minc)) for r, c in pcoords]
            red_pos = (int(red_r - minr), int(red_c - minc))
            payload_mask = np.zeros((H, W), dtype=bool)
            for r, c in payload_coords:
                payload_mask[r, c] = True
            if payload_mask[red_pos]:
                continue
            if _is_4connected_to(payload_mask, *red_pos):
                continue
            return {
                "kind": "isolated",
                "shape": (H, W),
                "payload_coords": payload_coords,
                "red_pixels": [red_pos],
                "axis": ("point", red_pos),
            }
    return None


def _global_coords(local_coords: list[tuple[int, int]], r0: int, c0: int) -> list[tuple[int, int]]:
    return [(int(r0 + r), int(c0 + c)) for r, c in local_coords]


def _reflected_payload_global(spr: dict, r0: int, c0: int) -> list[tuple[int, int]]:
    payload_global = _global_coords(spr["payload_coords"], r0, c0)
    axis_kind, axis_val = spr["axis"]
    if axis_kind == "row":
        axis_r = int(r0 + axis_val)
        return [_refl_row(axis_r, int(r), int(c)) for r, c in payload_global]
    if axis_kind == "col":
        axis_c = int(c0 + axis_val)
        return [_refl_col(axis_c, int(r), int(c)) for r, c in payload_global]
    pr, pc = spr["red_pixels"][0]
    return [_refl_point((int(r0 + pr), int(c0 + pc)), int(r), int(c)) for r, c in payload_global]


def _candidate_placement(
    spr: dict,
    r0: int,
    c0: int,
    shape: tuple[int, int],
) -> dict | None:
    H, W = shape
    h, w = spr["shape"]
    if not (0 <= int(r0) <= int(H - h) and 0 <= int(c0) <= int(W - w)):
        return None

    payload_global = _global_coords(spr["payload_coords"], r0, c0)
    red_global = _global_coords(spr["red_pixels"], r0, c0)
    refl_global = _reflected_payload_global(spr, r0, c0)

    refl_visible = [(int(r), int(c)) for r, c in refl_global if _inb(shape, int(r), int(c))]
    clipped = int(len(refl_global) - len(refl_visible))

    input_cells = set(payload_global) | set(red_global)
    output_cells = set(input_cells) | set(refl_visible)
    if len(output_cells) != len(input_cells) + len(refl_visible):
        return None
    if not refl_visible:
        return None

    return {
        "r0": int(r0),
        "c0": int(c0),
        "payload_global": payload_global,
        "red_global": red_global,
        "refl_visible": refl_visible,
        "output_cells": output_cells,
        "clipped": clipped,
        "visible_reflection": len(refl_visible),
        "payload_size": len(payload_global),
    }


def _placeable(
    occ: np.ndarray,
    placement: dict,
    *,
    padding: int = PLACEMENT_PADDING,
) -> bool:
    footprint = np.zeros_like(occ, dtype=bool)
    for r, c in placement["output_cells"]:
        footprint[int(r), int(c)] = True
    forbidden = occ
    if padding > 0 and occ.any():
        forbidden = ndimage.binary_dilation(occ, structure=STRUCT8, iterations=padding)
    return not bool((forbidden & footprint).any())


def _coords_mask(shape: tuple[int, int], coords: set[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    return mask


def _choose_location(
    occ: np.ndarray,
    spr: dict,
    shape: tuple[int, int],
    *,
    require_clip: bool,
    rng: random.Random,
) -> dict | None:
    H, W = shape
    h, w = spr["shape"]
    candidates = []
    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            placement = _candidate_placement(spr, int(r0), int(c0), shape)
            if placement is None:
                continue
            if require_clip:
                if not (0 < placement["clipped"] < placement["payload_size"]):
                    continue
            else:
                if placement["clipped"] != 0:
                    continue
            if not _placeable(occ, placement):
                continue
            candidates.append(placement)
    if not candidates:
        return None
    if require_clip:
        max_clip = max(p["clipped"] for p in candidates)
        clipped_band = [p for p in candidates if p["clipped"] >= max(1, max_clip - 1)]
        return rng.choice(clipped_band)
    return rng.choice(candidates)


def _stamp_input(canvas: np.ndarray, spr: dict, placement: dict) -> None:
    for r, c in placement["payload_global"]:
        canvas[int(r), int(c)] = int(spr["color"])
    for r, c in placement["red_global"]:
        canvas[int(r), int(c)] = int(MARK)


def _stamp_output(canvas: np.ndarray, spr: dict, placement: dict) -> None:
    _stamp_input(canvas, spr, placement)
    for r, c in placement["refl_visible"]:
        canvas[int(r), int(c)] = int(spr["color"])


def _analyze_clipping(grid: np.ndarray) -> list[dict]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape
    comps = _extract_components(g, bg)
    info = []

    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        if int(MARK) not in set(map(int, vals)):
            continue

        reds = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) == int(MARK)]
        if not reds:
            continue

        rs = {int(r) for r, _ in reds}
        cs = {int(c) for _, c in reds}
        if len(reds) == 1:
            axis = ("point", (int(reds[0][0]), int(reds[0][1])))
        elif len(rs) == 1:
            axis = ("row", int(next(iter(rs))))
        elif len(cs) == 1:
            axis = ("col", int(next(iter(cs))))
        else:
            continue

        r0, c0, r1, c1 = map(int, comp["bbox"])
        payload = {int(v) for v in vals if int(v) != int(MARK)}
        if not payload:
            continue

        sources = []
        if axis[0] == "row":
            for r in range(H):
                for c in range(W):
                    v = int(g[r, c])
                    if v != int(bg) and v != int(MARK) and int(v) in payload and int(c0) <= int(c) <= int(c1):
                        sources.append((int(r), int(c)))
        elif axis[0] == "col":
            for r in range(H):
                for c in range(W):
                    v = int(g[r, c])
                    if v != int(bg) and v != int(MARK) and int(v) in payload and int(r0) <= int(r) <= int(r1):
                        sources.append((int(r), int(c)))
        else:
            rr0 = max(0, int(r0 - 1))
            rr1 = min(int(H - 1), int(r1 + 1))
            cc0 = max(0, int(c0 - 1))
            cc1 = min(int(W - 1), int(c1 + 1))
            for r in range(H):
                for c in range(W):
                    v = int(g[r, c])
                    if (
                        v != int(bg)
                        and v != int(MARK)
                        and int(v) in payload
                        and int(rr0) <= int(r) <= int(rr1)
                        and int(cc0) <= int(c) <= int(cc1)
                    ):
                        sources.append((int(r), int(c)))

        clipped = 0
        kept = 0
        for r, c in sources:
            if axis[0] == "row":
                rr, cc = _refl_row(int(axis[1]), int(r), int(c))
            elif axis[0] == "col":
                rr, cc = _refl_col(int(axis[1]), int(r), int(c))
            else:
                rr, cc = _refl_point(axis[1], int(r), int(c))
            if _inb((H, W), int(rr), int(cc)):
                kept += 1
            else:
                clipped += 1
        info.append({"kind": "embedded", "clipped": int(clipped), "kept": int(kept)})

    marker_pts = []
    for comp in comps:
        if len(comp["coords"]) == 1 and int(comp["vals"][0]) == int(MARK):
            marker_pts.append(comp["coords"][0])

    for pr, pc in marker_pts:
        neigh_colors: set[int] = set()
        neigh_pixels: list[tuple[int, int, int]] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = int(pr + dr), int(pc + dc)
                if not _inb((H, W), rr, cc):
                    continue
                v = int(g[rr, cc])
                if v == int(bg) or v == int(MARK):
                    continue
                neigh_colors.add(int(v))
                neigh_pixels.append((int(rr), int(cc), int(v)))
        for v in neigh_colors:
            labv, _ = label_mask(g == int(v), connectivity=8)
            labs = {
                int(labv[int(rr), int(cc)])
                for rr, cc, vv in neigh_pixels
                if int(vv) == int(v) and int(labv[int(rr), int(cc)]) != 0
            }
            for lbl in labs:
                pts = [(int(r), int(c)) for r, c in np.argwhere(labv == int(lbl))]
                clipped = 0
                kept = 0
                for r, c in pts:
                    rr, cc = _refl_point((int(pr), int(pc)), int(r), int(c))
                    if _inb((H, W), int(rr), int(cc)):
                        kept += 1
                    else:
                        clipped += 1
                info.append({"kind": "isolated", "clipped": int(clipped), "kept": int(kept)})

    return info


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    N = rng.randint(GRID_MIN, GRID_MAX)
    bg = rng.choice([c for c in range(10) if c != MARK])
    palette = [c for c in range(10) if c != MARK and c != bg]

    total_markers = rng.randint(TOTAL_MARKERS_MIN, TOTAL_MARKERS_MAX)
    n_embedded = rng.randint(max(1, total_markers - MAX_ISOLATED), min(MAX_EMBEDDED, total_markers))
    n_isolated = total_markers - n_embedded

    n_colors = rng.randint(PALETTE_COLORS_MIN, min(PALETTE_COLORS_MAX, len(palette), total_markers))
    chosen_colors = rng.sample(palette, n_colors)

    sprites: list[dict] = []
    for builder, count in [(_build_embedded_sprite, n_embedded), (_build_isolated_sprite, n_isolated)]:
        for _ in range(count):
            spr = builder(rng)
            if spr is None:
                return None
            sprites.append(spr)

    assignment = list(chosen_colors) + [rng.choice(chosen_colors) for _ in range(len(sprites) - n_colors)]
    rng.shuffle(assignment)
    for spr, color in zip(sprites, assignment):
        spr["color"] = int(color)

    n_force_clip = rng.randint(MIN_REQUIRED_CLIPPED, min(MAX_REQUIRED_CLIPPED, len(sprites) - 1))
    clip_flags = [True] * n_force_clip + [False] * (len(sprites) - n_force_clip)
    rng.shuffle(clip_flags)
    for spr, clip in zip(sprites, clip_flags):
        spr["require_clip"] = bool(clip)

    order = list(range(len(sprites)))
    order.sort(
        key=lambda i: (
            0 if sprites[i]["require_clip"] else 1,
            -(len(sprites[i]["payload_coords"])),
        )
    )

    inp = np.full((N, N), bg, dtype=int)
    out = np.full((N, N), bg, dtype=int)
    occ = np.zeros((N, N), dtype=bool)

    for idx in order:
        spr = sprites[idx]
        placement = _choose_location(occ, spr, (N, N), require_clip=bool(spr["require_clip"]), rng=rng)
        if placement is None:
            return None
        spr["placement"] = placement
        _stamp_input(inp, spr, placement)
        _stamp_output(out, spr, placement)
        occ |= _coords_mask((N, N), placement["output_cells"])

    if (inp == bg).sum() / float(N * N) < BG_COVERAGE_MIN:
        return None

    clip_info = _analyze_clipping(inp)
    partial_clipped = [x for x in clip_info if x["clipped"] > 0 and x["kept"] > 0]
    if len(partial_clipped) < MIN_REQUIRED_CLIPPED:
        return None

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
