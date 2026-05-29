# Variant of 7ed72f31
# Dimension: D11 — 源-目标基数关系 (meta)
# Change: 1:1 — 每个标记组独立对应一个payload精灵，独立产生一个镜像副本
#         → 一对多：一个payload反射产生多个副本

from __future__ import annotations

from collections import Counter
import random
import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import get_bg, label_mask


# Design analysis
# ===============
# design:
#   source_task: none
#   core_change:
#     original: a marker-bearing mixed component had exactly one actionable red geometry
#       (one row, one column, or one point), so one payload created one reflected copy.
#     variant: the main mixed component is a red cross: one horizontal red line and one
#       vertical red line intersecting, with the payload placed in exactly one quadrant.
#       The same payload must be reflected across both axes, so one source produces three
#       copies: row-mirror, col-mirror, and double-mirror.
#   concrete_visual_form:
#     - Input groups are 4-connected mixed components.
#     - Each group contains:
#       - a red cross centered in a local odd-sized patch,
#       - a monochrome connected payload in one quadrant, touching the cross near the
#         intersection so the whole object is one 4-connected component.
#     - Output keeps the input intact and paints the payload into the other three quadrants.
#   behavior_trace:
#     original_logic:
#       - red pixels not all on one row/one col and not a single point => "ambiguous" => skip.
#     new_logic:
#       - if all red pixels lie on one distinguished row or one distinguished column and both
#         arms are present, classify as a cross centered at their intersection.
#       - reflect every payload pixel across the row, across the column, and across both.
#     example:
#       - payload at upper-left of the cross:
#         input  => payload only in upper-left quadrant
#         output => same shape also appears in upper-right, lower-left, lower-right.
#   complexity_calibration:
#     - The new rule is visible because every actionable group uses the same red-cross motif.
#     - Each grid contains 3-4 such groups, so the one-to-many relation is repeated.
#     - Payloads are small connected non-line shapes; this avoids visually trivial copies.
#   hidden_assumptions_review:
#     - Old code assumed "mixed marker geometry resolves to exactly one axis".
#       That fails for a cross and must be replaced by a multi-axis classifier.
#     - Old bbox-based eligibility could accidentally pull unrelated same-color pixels.
#       This was mostly harmless before because generator colors were often unique.
#       In the variant, exact component payload pixels are used instead; the generator also
#       guarantees unique colors per group.
#     - Painting mirrored pixels remains safe because generator places groups with empty space
#       in all mirror destinations, so no copy needs conflict resolution.
#   placement_legality:
#     - Generation is output-first. Each group is first built as its full output patch.
#     - The patch is placed only if every non-background pixel fits in-bounds and has a
#       one-cell background moat from previously placed groups.
#     - Input is derived by deleting the three mirrored payload copies while keeping the
#       original quadrant payload and the full red cross.
#   visual_observability:
#     - A human can see identical shapes repeated around the same cross in all examples.
#     - Repetition across 3-4 groups makes the two-axis rule inferable from 2-4 examples.
#   visual_cleanliness:
#     - No overlays between groups; each group occupies its own local patch with a moat.
#     - Colors are unique per group, so grouping is immediate.
#   color_inference:
#     - Color is not remapped. Each payload keeps its color in all copies.
#   main_new_assumptions:
#     - Cross groups are 1-cell-thick red unions of one row and one column.
#     - Payload pixels to be mirrored are exactly the non-red pixels in that same 4-connected
#       component.
#     - Generator enforces these assumptions directly.

MARK = 2


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= int(r) < int(h) and 0 <= int(c) < int(w)


def _refl_point(center: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    cr, cc = center
    return int(2 * cr - r), int(2 * cc - c)


def _refl_row(r0: int, r: int, c: int) -> tuple[int, int]:
    return int(2 * r0 - r), int(c)


def _refl_col(c0: int, r: int, c: int) -> tuple[int, int]:
    return int(r), int(2 * c0 - c)


def _classify_red_geometry(reds: list[tuple[int, int]]) -> tuple[str, object] | None:
    if not reds:
        return None
    if len(reds) == 1:
        return ("point", reds[0])

    row_counts = Counter(r for r, _ in reds)
    col_counts = Counter(c for _, c in reds)
    rows = set(row_counts)
    cols = set(col_counts)

    if len(rows) == 1:
        return ("row", next(iter(rows)))
    if len(cols) == 1:
        return ("col", next(iter(cols)))

    r0 = max(sorted(rows), key=lambda r: row_counts[r])
    c0 = max(sorted(cols), key=lambda c: col_counts[c])
    red_set = set(reds)
    if (r0, c0) not in red_set:
        return None
    if not all((r == r0) or (c == c0) for r, c in reds):
        return None
    has_row_arm = any(r == r0 and c != c0 for r, c in reds)
    has_col_arm = any(c == c0 and r != r0 for r, c in reds)
    if has_row_arm and has_col_arm:
        return ("cross", (r0, c0))
    return None


def _analyze_marker_components(grid: np.ndarray) -> list[dict]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    lab, n = label_mask(g != bg, connectivity=4)
    comps: list[dict] = []
    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if not coords:
            continue
        vals = [int(g[r, c]) for r, c in coords]
        if MARK not in vals:
            continue
        reds = [(r, c) for (r, c), v in zip(coords, vals) if int(v) == MARK]
        payload = [(r, c, v) for (r, c), v in zip(coords, vals) if int(v) != MARK]
        geom = _classify_red_geometry(reds)
        comps.append(
            {
                "coords": coords,
                "vals": vals,
                "reds": reds,
                "payload": payload,
                "geometry": geom,
                "single_red_component": len(coords) == 1 and vals[0] == MARK,
            }
        )
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()
    h, w = g.shape

    def paint(r: int, c: int, color: int) -> None:
        if _in_bounds((h, w), r, c):
            out[int(r), int(c)] = int(color)

    comps = _analyze_marker_components(g)

    # Mixed marker-bearing components: reflect the exact payload pixels from the component.
    for comp in comps:
        geom = comp["geometry"]
        payload = comp["payload"]
        if not geom or not payload or comp["single_red_component"]:
            continue

        kind, anchor = geom
        for r, c, color in payload:
            targets: list[tuple[int, int]] = []
            if kind == "row":
                targets.append(_refl_row(int(anchor), int(r), int(c)))
            elif kind == "col":
                targets.append(_refl_col(int(anchor), int(r), int(c)))
            elif kind == "point":
                targets.append(_refl_point(anchor, int(r), int(c)))
            elif kind == "cross":
                cr, cc = anchor
                targets.append(_refl_row(int(cr), int(r), int(c)))
                targets.append(_refl_col(int(cc), int(r), int(c)))
                targets.append(_refl_point((int(cr), int(cc)), int(r), int(c)))

            for rr, cc in set(targets):
                paint(int(rr), int(cc), int(color))

    # Original isolated-red behavior is preserved for compatibility.
    for comp in comps:
        if not comp["single_red_component"]:
            continue
        pr, pc = comp["coords"][0]
        neigh_colors: set[int] = set()
        neigh_pixels: list[tuple[int, int, int]] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = int(pr + dr), int(pc + dc)
                if not _in_bounds((h, w), rr, cc):
                    continue
                v = int(g[rr, cc])
                if v == bg or v == MARK:
                    continue
                neigh_colors.add(v)
                neigh_pixels.append((rr, cc, v))
        for color in neigh_colors:
            labv, _ = label_mask(g == int(color), connectivity=8)
            labels = {
                int(labv[rr, cc])
                for rr, cc, vv in neigh_pixels
                if int(vv) == int(color) and int(labv[rr, cc]) != 0
            }
            for lbl in labels:
                for rr, cc in np.argwhere(labv == int(lbl)):
                    tr, tc = _refl_point((int(pr), int(pc)), int(rr), int(cc))
                    paint(int(tr), int(tc), int(color))

    return out


def _grow_connected_shape(
    rng: random.Random, box_h: int, box_w: int, area: int, seed: tuple[int, int]
) -> set[tuple[int, int]]:
    shape = {seed}
    while len(shape) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in shape:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < box_h and 0 <= cc < box_w and (rr, cc) not in shape:
                    frontier.append((rr, cc))
        if not frontier:
            break
        shape.add(rng.choice(frontier))
    return shape


def _build_cross_group(rng: random.Random, color: int, bg: int) -> tuple[np.ndarray, np.ndarray, dict]:
    box_h = rng.randint(2, 3)
    box_w = rng.randint(2, 3)
    patch_h = 2 * box_h + 1
    patch_w = 2 * box_w + 1
    center_r, center_c = box_h, box_w
    row_sign = rng.choice([-1, 1])
    col_sign = rng.choice([-1, 1])
    near_r = box_h - 1 if row_sign == -1 else 0
    near_c = box_w - 1 if col_sign == -1 else 0
    seed = (near_r, near_c)
    max_area = min(6, box_h * box_w)
    min_area = 3 if max_area <= 3 else 4

    shape: set[tuple[int, int]] | None = None
    for _ in range(50):
        area = rng.randint(min_area, max_area)
        candidate = _grow_connected_shape(rng, box_h, box_w, area, seed)
        rows = {r for r, _ in candidate}
        cols = {c for _, c in candidate}
        if len(rows) >= 2 and len(cols) >= 2:
            shape = candidate
            break
    if shape is None:
        shape = {seed, (seed[0], max(0, seed[1] - 1)), (max(0, seed[0] - 1), seed[1]), (0, 0)}

    inp = np.full((patch_h, patch_w), int(bg), dtype=int)
    out = np.full((patch_h, patch_w), int(bg), dtype=int)
    inp[center_r, :] = MARK
    inp[:, center_c] = MARK
    out[center_r, :] = MARK
    out[:, center_c] = MARK

    payload_coords: list[tuple[int, int]] = []
    for lr, lc in shape:
        pr = int(lr if row_sign == -1 else center_r + 1 + lr)
        pc = int(lc if col_sign == -1 else center_c + 1 + lc)
        payload_coords.append((pr, pc))
        inp[pr, pc] = int(color)
        out[pr, pc] = int(color)

    clone_pixels: set[tuple[int, int]] = set()
    for pr, pc in payload_coords:
        clone_pixels.add(_refl_row(center_r, pr, pc))
        clone_pixels.add(_refl_col(center_c, pr, pc))
        clone_pixels.add(_refl_point((center_r, center_c), pr, pc))
    for rr, cc in clone_pixels:
        out[int(rr), int(cc)] = int(color)

    meta = {
        "type": "cross",
        "payload_area": len(payload_coords),
        "clone_count": 3,
        "painted_new_pixels": int(np.count_nonzero((out != bg) & (inp == bg))),
    }
    return inp, out, meta


def _can_place(occupied: np.ndarray, patch: np.ndarray, bg: int, top: int, left: int, pad: int = 1) -> bool:
    h, w = occupied.shape
    ph, pw = patch.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return False
    for r, c in np.argwhere(patch != int(bg)):
        rr = int(top + r)
        cc = int(left + c)
        r0 = max(0, rr - pad)
        r1 = min(h, rr + pad + 1)
        c0 = max(0, cc - pad)
        c1 = min(w, cc + pad + 1)
        if occupied[r0:r1, c0:c1].any():
            return False
    return True


def _paste(dst: np.ndarray, patch: np.ndarray, bg: int, top: int, left: int) -> None:
    ph, pw = patch.shape
    window = dst[top : top + ph, left : left + pw]
    mask = patch != int(bg)
    window[mask] = patch[mask]


def generate_pair(rng: random.Random | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = rng or random.Random()
    palette = [c for c in range(10) if c != MARK]

    for _attempt in range(200):
        h = rng.randint(18, 28)
        w = rng.randint(18, 28)
        bg = rng.choice(palette)
        colors = [c for c in palette if c != bg]
        rng.shuffle(colors)
        n_groups = rng.randint(3, 4)
        if len(colors) < n_groups:
            continue

        inp = np.full((h, w), int(bg), dtype=int)
        out = np.full((h, w), int(bg), dtype=int)
        occupied = np.zeros((h, w), dtype=bool)
        group_meta: list[dict] = []
        ok = True

        for color in colors[:n_groups]:
            patch_in, patch_out, meta = _build_cross_group(rng, int(color), int(bg))
            ph, pw = patch_out.shape
            placed = False
            for _ in range(120):
                top = rng.randint(0, h - ph)
                left = rng.randint(0, w - pw)
                if not _can_place(occupied, patch_out, int(bg), top, left, pad=1):
                    continue
                _paste(inp, patch_in, int(bg), top, left)
                _paste(out, patch_out, int(bg), top, left)
                occupied[top : top + ph, left : left + pw] |= patch_out != int(bg)
                group_meta.append(meta)
                placed = True
                break
            if not placed:
                ok = False
                break

        if not ok:
            continue
        solved = main(inp)
        if np.array_equal(solved, out):
            return inp, out, {"bg": bg, "groups": group_meta}

    raise RuntimeError("failed to generate a valid D11 variant sample")


def generate_input() -> np.ndarray:
    inp, _out, _meta = generate_pair()
    return inp


def analyze_variant(grid: np.ndarray) -> dict:
    comps = _analyze_marker_components(np.array(grid, dtype=int))
    cross_groups = 0
    payload_areas: list[int] = []
    for comp in comps:
        geom = comp["geometry"]
        if geom and geom[0] == "cross":
            cross_groups += 1
            payload_areas.append(len(comp["payload"]))
    solved = main(np.array(grid, dtype=int))
    changed = int(np.count_nonzero(solved != np.array(grid, dtype=int)))
    return {
        "cross_groups": int(cross_groups),
        "payload_areas": payload_areas,
        "changed_pixels": changed,
    }

