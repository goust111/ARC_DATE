# Variant of 7ed72f31
# Dimension: D16 — 精灵与标记之间的基数关系 (meta)
# Change: 一对一关系：每个标记对应一个精灵（或一组8邻域精灵），每个精灵有且只有一个标记... → 多对一：多个精灵共享一个反射轴

from __future__ import annotations

import random

import numpy as np
from scipy import ndimage

from arc_utils import STRUCT8, get_bg, label_mask


"""
design_analysis:
  source_task: none
  variant_goal:
    dimension: D16
    change: a single red line is no longer paired with one payload sprite; it serves as a shared
      reflection axis for multiple independent monochrome sprites placed near that line.
  concretization:
    shared_axis_marker:
      shape: an isolated straight red segment, horizontal or vertical
      relation_to_sprites: the line is not 4-connected to the payloads; there is at least one bg cell
        between the line and every payload sprite
      coverage: the line spans the full union extent of the payloads along the parallel axis
    payloads:
      count_per_shared_axis: 2-4 independent 4-connected monochrome blobs
      sides: blobs may appear on either side of the same axis; at least one axis per grid uses both sides
      transform: each blob is mirrored independently across the shared line; originals remain
    unchanged_branch:
      isolated single red pixel still point-reflects adjacent 8-connected monochrome components
  behavior_trace:
    original_small_case:
      input: one red horizontal line touching one payload component
      solver_behavior: component bbox is extracted from one mixed 4-connected object, so only that
        component can be mirrored
    variant_small_case:
      input: one isolated horizontal red line, two separate blue/yellow blobs above it, one green blob below it
      solver_behavior: the red line is first recognized as a standalone axis; then each nearby blob whose
        bbox lies within the axis span and whose mirror landing zone is empty is reflected independently
      visible_difference: the output adds three mirrored copies from one line instead of one
  complexity_calibration:
    kept_small:
      - only two line orientations
      - payload sprites remain monochrome 4-connected blobs
      - isolated point markers keep the old point-reflection rule
    added_complexity:
      - each shared axis has 2-4 sprites, not an unbounded number
      - axis influence is local: only blobs close to the line and fully under its span are eligible
    anti-degeneracy:
      - generator forces at least one shared axis with >=2 sprites
      - most samples show sprites on both sides of the same line
  implicit_assumption_review:
    original_main_risk_1:
      code: label_mask(g != bg, connectivity=4) followed by per-component axis extraction
      old_assumption: a red line and its payload are in the same 4-connected component
      variant_status: false, because the shared line is isolated from all payloads
      fix: add a dedicated pass over pure-red straight components and collect colored-only components separately
    original_main_risk_2:
      code: eligible() restricts candidate pixels to the bbox of the mixed component that contains the red line
      old_assumption: each marker owns exactly one local bbox
      variant_status: false
      fix: eligibility now uses axis span + side + gap + empty mirror landing zone instead of a single mixed bbox
    original_main_risk_3:
      code: isolated singleton red points are found as len(component)==1 in the non-bg component list
      old_assumption: this only refers to point markers
      variant_status: still true if line axes are extracted from pure-red components first
      fix: keep singleton pure-red components as the point-reflection branch
  placement_legality:
    generator_strategy:
      - build each shared-axis arrangement as one local output patch
      - reserve disjoint parallel intervals for each payload so mirrored copies cannot overlap
      - place patches on the global canvas using footprint collision checks with 8-neighborhood padding
    solver_alignment:
      - every shared-axis payload lies entirely on one side of its axis
      - payload bbox stays inside the axis span
      - perpendicular gap to the axis is 1-2 bg cells, which the solver uses as the local ownership test
  visual_observability:
    why_humans_can_see_it:
      - the same red line clearly causes multiple separate copies to appear
      - some examples show activity on both sides of the same axis, making "shared axis" more obvious
      - colors are not used for hidden indexing; color only identifies each sprite's copied pixels
  cleanliness:
    measures:
      - payloads under one axis are separated by bg columns/rows
      - different axis groups are placed with padding so their reflection zones do not touch
      - reflected landings are guaranteed empty in the input
  color_inference:
    rule: colors are preserved under reflection; there is no recoloring
    support: 3-4 non-bg colors are reused across groups so examples do not become visually noisy
  new_solver_assumptions_after_change:
    - a shared line axis is a contiguous pure-red segment, not a bent red shape
    - a payload influenced by a shared axis is monochrome, 4-connected, fully within the axis span,
      and within a small perpendicular gap from the line
    - the mirror landing cells are bg in the input
    generator_guarantee: all three conditions are enforced during group construction and placement
"""


MARK = 2

# Canvas
GRID_MIN, GRID_MAX = 18, 28
BG_COVERAGE_MIN = 0.88

# Group counts
TOTAL_GROUPS_MIN, TOTAL_GROUPS_MAX = 2, 4
SHARED_AXES_MIN, SHARED_AXES_MAX = 1, 2
MAX_ISOLATED_GROUPS = 2

# Shared-axis payloads
SHARED_PAYLOADS_MIN, SHARED_PAYLOADS_MAX = 2, 4
SHARED_SIDE_MIN, SHARED_SIDE_MAX = 2, 4
PAYLOAD_PX_MIN, PAYLOAD_PX_MAX = 3, 10
PAYLOAD_DENSITY_LO, PAYLOAD_DENSITY_HI = 0.45, 0.80
SHARED_AXIS_GAP_MIN, SHARED_AXIS_GAP_MAX = 1, 2
LANE_SEP = 1

# Isolated point payloads
ISOLATED_PX_MIN, ISOLATED_PX_MAX = 1, 7
ISOLATED_BBOX_MAX = 4

# Palette
PALETTE_COLORS_MIN, PALETTE_COLORS_MAX = 3, 4

# Placement / retries
PLACEMENT_PADDING = 1
MASK_RETRIES = 40
GEN_RETRIES = 400

_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant solver for 7ed72f31.

    Added rule:
    - A standalone red line can serve as a shared reflection axis for several nearby monochrome sprites.

    Preserved rule:
    - A singleton isolated red pixel still point-reflects adjacent 8-connected monochrome components.
    - Mixed red+payload components from the original task are still handled by the original local rule.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape
    out = g.copy()

    def inb(r: int, c: int) -> bool:
        return 0 <= int(r) < H and 0 <= int(c) < W

    def refl_point(p: tuple[int, int], r: int, c: int) -> tuple[int, int]:
        pr, pc = map(int, p)
        return int(2 * pr - r), int(2 * pc - c)

    def refl_row(r0: int, r: int, c: int) -> tuple[int, int]:
        return int(2 * r0 - r), int(c)

    def refl_col(c0: int, r: int, c: int) -> tuple[int, int]:
        return int(r), int(2 * c0 - c)

    def line_axis(coords: list[tuple[int, int]]) -> tuple[str, int, int, int] | None:
        if len(coords) <= 1:
            return None
        rs = {int(r) for r, _ in coords}
        cs = {int(c) for _, c in coords}
        if len(rs) == 1:
            row = int(next(iter(rs)))
            c0, c1 = min(cs), max(cs)
            if len(coords) == c1 - c0 + 1:
                return ("row", row, int(c0), int(c1))
        if len(cs) == 1:
            col = int(next(iter(cs)))
            r0, r1 = min(rs), max(rs)
            if len(coords) == r1 - r0 + 1:
                return ("col", col, int(r0), int(r1))
        return None

    # Original mixed-component branch retained for compatibility.
    lab, n = label_mask(g != bg, connectivity=4)
    mixed_comps = []
    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if not coords:
            continue
        vals = [int(g[r, c]) for r, c in coords]
        rs = [r for r, _ in coords]
        cs = [c for _, c in coords]
        mixed_comps.append(
            {
                "coords": coords,
                "vals": vals,
                "bbox": (int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))),
            }
        )

    for comp in mixed_comps:
        coords = comp["coords"]
        vals = comp["vals"]
        if MARK not in set(map(int, vals)):
            continue

        reds = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) == MARK]
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
        payload = {int(v) for v in vals if int(v) != MARK}
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
                v = int(g[r, c])
                if v == bg or v == MARK or not eligible(int(r), int(c), int(v)):
                    continue
                if axis[0] == "row":
                    rr, cc = refl_row(int(axis[1]), int(r), int(c))
                elif axis[0] == "col":
                    rr, cc = refl_col(int(axis[1]), int(r), int(c))
                else:
                    rr, cc = refl_point(axis[1], int(r), int(c))
                if inb(int(rr), int(cc)):
                    out[int(rr), int(cc)] = int(v)

    # Shared-axis branch: pure-red lines now affect several nearby independent payload components.
    red_lab, red_n = label_mask(g == MARK, connectivity=4)
    red_lines: list[tuple[str, int, int, int]] = []
    red_points: list[tuple[int, int]] = []
    for k in range(1, int(red_n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(red_lab == int(k))]
        if len(coords) == 1:
            red_points.append(coords[0])
            continue
        axis = line_axis(coords)
        if axis is not None:
            red_lines.append(axis)

    color_lab, color_n = label_mask((g != bg) & (g != MARK), connectivity=4)
    color_comps = []
    for k in range(1, int(color_n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(color_lab == int(k))]
        if not coords:
            continue
        vals = {int(g[r, c]) for r, c in coords}
        if len(vals) != 1:
            continue
        rs = [r for r, _ in coords]
        cs = [c for _, c in coords]
        color_comps.append(
            {
                "coords": coords,
                "bbox": (int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))),
                "color": int(next(iter(vals))),
            }
        )

    def eligible_shared(axis: tuple[str, int, int, int], comp: dict) -> bool:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        coords = comp["coords"]
        if axis[0] == "row":
            axis_r, span0, span1 = int(axis[1]), int(axis[2]), int(axis[3])
            if not (span0 <= c0 and c1 <= span1):
                return False
            if r1 < axis_r:
                gap = axis_r - r1 - 1
                reflect = lambda rr, cc: refl_row(axis_r, rr, cc)
            elif r0 > axis_r:
                gap = r0 - axis_r - 1
                reflect = lambda rr, cc: refl_row(axis_r, rr, cc)
            else:
                return False
        else:
            axis_c, span0, span1 = int(axis[1]), int(axis[2]), int(axis[3])
            if not (span0 <= r0 and r1 <= span1):
                return False
            if c1 < axis_c:
                gap = axis_c - c1 - 1
                reflect = lambda rr, cc: refl_col(axis_c, rr, cc)
            elif c0 > axis_c:
                gap = c0 - axis_c - 1
                reflect = lambda rr, cc: refl_col(axis_c, rr, cc)
            else:
                return False
        if not (SHARED_AXIS_GAP_MIN <= int(gap) <= SHARED_AXIS_GAP_MAX):
            return False
        for rr, cc in coords:
            tr, tc = reflect(int(rr), int(cc))
            if not inb(int(tr), int(tc)):
                return False
            if int(g[int(tr), int(tc)]) != bg:
                return False
        return True

    for axis in red_lines:
        for comp in color_comps:
            if not eligible_shared(axis, comp):
                continue
            for r, c in comp["coords"]:
                if axis[0] == "row":
                    rr, cc = refl_row(int(axis[1]), int(r), int(c))
                else:
                    rr, cc = refl_col(int(axis[1]), int(r), int(c))
                if inb(int(rr), int(cc)):
                    out[int(rr), int(cc)] = int(comp["color"])

    # Original isolated-red-point branch retained.
    for pr, pc in red_points:
        pr = int(pr)
        pc = int(pc)
        neigh_colors: set[int] = set()
        neigh_pixels: list[tuple[int, int, int]] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = int(pr + dr), int(pc + dc)
                if not inb(rr, cc):
                    continue
                v = int(g[rr, cc])
                if v == bg or v == MARK:
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
                    rr, cc = refl_point((int(pr), int(pc)), int(r), int(c))
                    if inb(rr, cc):
                        out[int(rr), int(cc)] = int(v)

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


def _is_4connected_to(mask: np.ndarray, r: int, c: int) -> bool:
    H, W = mask.shape
    for dr, dc in _DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W and mask[nr, nc]:
            return True
    return False


def _transpose_group(group: dict) -> dict:
    return {
        "red": group["red"].T.copy(),
        "items": [
            {
                "orig": item["orig"].T.copy(),
                "mirror": item["mirror"].T.copy(),
                "side": item["side"],
                "color": item.get("color"),
            }
            for item in group["items"]
        ],
        "kind": group["kind"],
        "orientation": "col" if group["orientation"] == "row" else group["orientation"],
        "shared_count": group.get("shared_count", 1),
        "both_sides": group.get("both_sides", False),
    }


def _build_shared_row_group(rng: random.Random, *, force_both_sides: bool) -> dict | None:
    k = rng.randint(SHARED_PAYLOADS_MIN, SHARED_PAYLOADS_MAX)
    payload_specs = []
    for _ in range(k):
        ph = rng.randint(SHARED_SIDE_MIN, SHARED_SIDE_MAX)
        pw = rng.randint(SHARED_SIDE_MIN, SHARED_SIDE_MAX)
        max_px = min(PAYLOAD_PX_MAX, int(PAYLOAD_DENSITY_HI * ph * pw))
        min_px = max(PAYLOAD_PX_MIN, int(np.ceil(PAYLOAD_DENSITY_LO * ph * pw)))
        if min_px > max_px:
            return None
        pmask = _random_connected_mask(ph, pw, rng.randint(min_px, max_px), rng=rng)
        if pmask is None:
            return None
        payload_specs.append(
            {
                "mask": pmask,
                "gap": rng.randint(SHARED_AXIS_GAP_MIN, SHARED_AXIS_GAP_MAX),
            }
        )

    sides = [rng.choice([-1, 1]) for _ in range(k)]
    if force_both_sides and k >= 2:
        sides[0] = -1
        sides[1] = 1
        rng.shuffle(sides)
    elif k >= 2 and rng.random() < 0.55:
        sides[0] = -1
        sides[1] = 1
        rng.shuffle(sides)

    widths = [int(spec["mask"].shape[1]) for spec in payload_specs]
    axis_w = int(sum(widths) + LANE_SEP * (k - 1))
    # The output contains originals and mirrors, so both sides must reserve the maximum
    # payload reach, regardless of which side currently holds the original.
    top = max(int(spec["mask"].shape[0]) + int(spec["gap"]) for spec in payload_specs)
    bottom = max(int(spec["mask"].shape[0]) + int(spec["gap"]) for spec in payload_specs)
    H = int(top + 1 + bottom)
    W = int(axis_w)
    axis_r = int(top)

    red = np.zeros((H, W), dtype=bool)
    red[axis_r, :] = True
    items = []
    ccur = 0
    for spec, side in zip(payload_specs, sides):
        mask = spec["mask"]
        ph, pw = mask.shape
        gap = int(spec["gap"])
        if side == -1:
            r0 = int(axis_r - gap - ph)
        else:
            r0 = int(axis_r + gap + 1)
        c0 = int(ccur)
        orig = np.zeros((H, W), dtype=bool)
        mirror = np.zeros((H, W), dtype=bool)
        for r, c in np.argwhere(mask):
            rr = int(r0 + r)
            cc = int(c0 + c)
            mr = int(2 * axis_r - rr)
            orig[rr, cc] = True
            mirror[mr, cc] = True
        if (orig & mirror).any() or (orig & red).any() or (mirror & red).any():
            return None
        items.append({"orig": orig, "mirror": mirror, "side": side, "color": None})
        ccur += int(pw + LANE_SEP)

    footprint = red.copy()
    for item in items:
        if (footprint & item["orig"]).any() or (footprint & item["mirror"]).any():
            return None
        footprint |= item["orig"] | item["mirror"]

    return {
        "red": red,
        "items": items,
        "kind": "shared_axis",
        "orientation": "row",
        "shared_count": int(k),
        "both_sides": bool((-1 in sides) and (1 in sides)),
    }


def _build_shared_axis_group(rng: random.Random, *, force_both_sides: bool) -> dict | None:
    row_group = _build_shared_row_group(rng, force_both_sides=force_both_sides)
    if row_group is None:
        return None
    if rng.random() < 0.5:
        return row_group
    return _transpose_group(row_group)


def _build_isolated_group(rng: random.Random) -> dict | None:
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

    pcoords = list(zip(*np.where(pmask)))
    rng.shuffle(pcoords)
    for pr, pc in pcoords:
        dirs = list(_DIRS8)
        rng.shuffle(dirs)
        for dr, dc in dirs:
            red_r, red_c = pr + dr, pc + dc
            refl = [(2 * red_r - r, 2 * red_c - c) for r, c in pcoords]
            all_pts = pcoords + refl + [(red_r, red_c)]
            all_rs = [r for r, _ in all_pts]
            all_cs = [c for _, c in all_pts]
            minr, minc = min(all_rs), min(all_cs)
            H = max(all_rs) - minr + 1
            W = max(all_cs) - minc + 1
            orig = np.zeros((H, W), dtype=bool)
            mirror = np.zeros((H, W), dtype=bool)
            for r, c in pcoords:
                orig[r - minr, c - minc] = True
            for r, c in refl:
                mirror[r - minr, c - minc] = True
            red = np.zeros((H, W), dtype=bool)
            red_pos = (red_r - minr, red_c - minc)
            red[red_pos] = True
            if orig[red_pos] or mirror[red_pos]:
                continue
            if (orig & mirror).any():
                continue
            if _is_4connected_to(orig | mirror, *red_pos):
                continue
            return {
                "red": red,
                "items": [{"orig": orig, "mirror": mirror, "side": 0, "color": None}],
                "kind": "isolated",
                "orientation": "point",
                "shared_count": 1,
                "both_sides": False,
            }
    return None


def _group_output_mask(group: dict) -> np.ndarray:
    mask = group["red"].copy()
    for item in group["items"]:
        mask |= item["orig"] | item["mirror"]
    return mask


def _stamp_group(canvas: np.ndarray, group: dict, r0: int, c0: int, *, include_mirror: bool) -> None:
    h, w = group["red"].shape
    window = canvas[r0 : r0 + h, c0 : c0 + w]
    window[group["red"]] = MARK
    for item in group["items"]:
        window[item["orig"]] = int(item["color"])
        if include_mirror:
            window[item["mirror"]] = int(item["color"])


def _random_free_location(
    canvas: np.ndarray, patch_mask: np.ndarray, bg: int, rng: random.Random, *, padding: int = PLACEMENT_PADDING
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


def _shared_axis_payload_counts(grid: np.ndarray) -> list[int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    red_lab, red_n = label_mask(g == MARK, connectivity=4)
    color_lab, color_n = label_mask((g != bg) & (g != MARK), connectivity=4)

    def _line(coords: list[tuple[int, int]]) -> tuple[str, int, int, int] | None:
        if len(coords) <= 1:
            return None
        rs = {int(r) for r, _ in coords}
        cs = {int(c) for _, c in coords}
        if len(rs) == 1:
            row = int(next(iter(rs)))
            c0, c1 = min(cs), max(cs)
            if len(coords) == c1 - c0 + 1:
                return ("row", row, int(c0), int(c1))
        if len(cs) == 1:
            col = int(next(iter(cs)))
            r0, r1 = min(rs), max(rs)
            if len(coords) == r1 - r0 + 1:
                return ("col", col, int(r0), int(r1))
        return None

    color_comps = []
    for k in range(1, int(color_n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(color_lab == int(k))]
        if not coords:
            continue
        vals = {int(g[r, c]) for r, c in coords}
        if len(vals) != 1:
            continue
        rs = [r for r, _ in coords]
        cs = [c for _, c in coords]
        color_comps.append({"coords": coords, "bbox": (min(rs), min(cs), max(rs), max(cs))})

    counts = []
    for k in range(1, int(red_n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(red_lab == int(k))]
        axis = _line(coords)
        if axis is None:
            continue
        count = 0
        for comp in color_comps:
            r0, c0, r1, c1 = map(int, comp["bbox"])
            if axis[0] == "row":
                ar, span0, span1 = int(axis[1]), int(axis[2]), int(axis[3])
                if span0 <= c0 and c1 <= span1 and ((r1 < ar and SHARED_AXIS_GAP_MIN <= ar - r1 - 1 <= SHARED_AXIS_GAP_MAX) or (r0 > ar and SHARED_AXIS_GAP_MIN <= r0 - ar - 1 <= SHARED_AXIS_GAP_MAX)):
                    count += 1
            else:
                ac, span0, span1 = int(axis[1]), int(axis[2]), int(axis[3])
                if span0 <= r0 and r1 <= span1 and ((c1 < ac and SHARED_AXIS_GAP_MIN <= ac - c1 - 1 <= SHARED_AXIS_GAP_MAX) or (c0 > ac and SHARED_AXIS_GAP_MIN <= c0 - ac - 1 <= SHARED_AXIS_GAP_MAX)):
                    count += 1
        counts.append(int(count))
    return counts


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    N = rng.randint(GRID_MIN, GRID_MAX)
    bg = rng.choice([c for c in range(10) if c != MARK])
    palette = [c for c in range(10) if c not in (bg, MARK)]

    total_groups = rng.randint(TOTAL_GROUPS_MIN, TOTAL_GROUPS_MAX)
    n_shared = rng.randint(SHARED_AXES_MIN, min(SHARED_AXES_MAX, total_groups))
    n_isolated = min(total_groups - n_shared, MAX_ISOLATED_GROUPS)
    if total_groups - n_shared > MAX_ISOLATED_GROUPS:
        return None

    groups: list[dict] = []
    for idx in range(n_shared):
        group = _build_shared_axis_group(rng, force_both_sides=(idx == 0))
        if group is None:
            return None
        groups.append(group)
    for _ in range(n_isolated):
        group = _build_isolated_group(rng)
        if group is None:
            return None
        groups.append(group)

    total_payloads = sum(len(group["items"]) for group in groups)
    if total_payloads < PALETTE_COLORS_MIN:
        return None

    n_colors = rng.randint(PALETTE_COLORS_MIN, min(PALETTE_COLORS_MAX, len(palette), total_payloads))
    chosen_colors = rng.sample(palette, n_colors)
    item_refs = [item for group in groups for item in group["items"]]
    assignment = list(chosen_colors) + [rng.choice(chosen_colors) for _ in range(len(item_refs) - n_colors)]
    rng.shuffle(assignment)
    for item, color in zip(item_refs, assignment):
        item["color"] = int(color)

    output = np.full((N, N), bg, dtype=int)
    inp = np.full((N, N), bg, dtype=int)
    rng.shuffle(groups)
    for group in groups:
        loc = _random_free_location(output, _group_output_mask(group), bg, rng)
        if loc is None:
            return None
        r0, c0 = loc
        _stamp_group(output, group, r0, c0, include_mirror=True)
        _stamp_group(inp, group, r0, c0, include_mirror=False)

    if (inp == bg).sum() / float(N * N) < BG_COVERAGE_MIN:
        return None

    counts = _shared_axis_payload_counts(inp)
    if not counts or max(counts) < 2:
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
