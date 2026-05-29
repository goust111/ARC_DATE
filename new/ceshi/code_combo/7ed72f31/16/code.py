# Combo variant of 7ed72f31 (index 16)
# combo_spec: D5/2+D11/1
# Name: 多副本碰撞消除
#
# Design analysis
# ---------------
# yaml:
#   concretization_and_trace:
#     fused_visual_form:
#       - Each actionable group is a 1-cell-thick red cross.
#       - Around the cross, one monochrome payload occupies several cells from a few
#         4-way symmetric orbits: {(+-dr, +-dc)} with dr>0, dc>0.
#       - A payload cell in exactly one orbit position produces three reflected
#         copies, so the whole 4-position orbit survives in the output.
#       - If an orbit already contains multiple payload cells, the copies generated
#         from those cells land on each other or on existing source cells, and the
#         whole orbit is erased by collision cancellation.
#     small_trace:
#       - Let the red cross center be (5, 5), and consider the orbit
#         {(4,3), (4,7), (6,3), (6,7)}.
#       - Original logic with one source at (4,3):
#         row -> (6,3), col -> (4,7), point -> (6,7); output fills all four cells.
#       - Fused logic with two sources at (4,3) and (6,3):
#         row-reflections hit the opposite existing source cells,
#         col/point reflections both target (4,7) and (6,7),
#         so every cell in the orbit is involved in a collision and becomes bg.
#       - Difference:
#         one occupied orbit slot means "replicate to the other three slots";
#         two or more occupied slots in the same orbit means "all copies cancel".
#   complexity_calibration:
#     - Every sample contains 3-4 red-cross groups, and every group mixes
#       singleton-orbits with collision-orbits. This makes the combo behavior
#       visible repeatedly instead of relying on one subtle corner case.
#     - The generator uses only a small orbit vocabulary near the cross
#       (positions with min(|dr|,|dc|)=1), so humans can visually align the four
#       symmetric slots of each orbit.
#     - Colors are unique per group, which avoids accidental cross-group binding.
#   implicit_assumptions_review:
#     - The original solver assumed one marker geometry per mixed component.
#       Crosses violate that, so geometry classification must recognize a row+col
#       union and emit three transforms from the same payload.
#     - The original row/col solver scanned the whole grid inside a bbox and could
#       capture unrelated same-color pixels. That is unsafe once multiple copies can
#       collide, so the fused solver reflects the exact payload pixels belonging to
#       the marker-bearing component.
#     - Direct in-place painting is no longer valid. All reflected hits must be
#       collected first, then resolved together so mirror-vs-mirror and
#       mirror-vs-source collisions both erase to background.
#     - New solver assumption: each intended multi-axis group is a clean red cross,
#       and inter-group footprints do not overlap. The generator enforces both.
#   placement_legality:
#     - Each group is first sampled as a full reserved footprint:
#       red cross + all four slots of every chosen orbit.
#     - Placement checks that reserved footprint stays in-bounds and keeps a
#       1-cell moat from previously placed groups, so groups never merge and their
#       possible reflected copies never interact across groups.
#     - Input is derived by keeping only selected source slots from each orbit;
#       output is obtained by the collision-aware solver.
#   visual_observability:
#     - A human sees the same red-cross motif repeatedly.
#       In each group, some partially filled orbit completes into all four slots,
#       while some already-multi-occupied orbit disappears instead of duplicating.
#     - The generator rejects degenerate samples where the collision-aware output
#       matches naive overwrite output.
#   cleanliness:
#     - Only one marker style is used: a red cross.
#     - Payload cells lie on a narrow ring around the cross, so the local geometry
#       is structured and easy to parse rather than scattered over the patch.
#   color_inference:
#     - No color remapping occurs. Each surviving copy keeps the payload color.
#   main_new_assumptions_and_generator_support:
#     - Cross groups are unions of one full red row and one full red column.
#       Generator constructs exactly that.
#     - Payload cells never lie on the red axes. Generator samples only off-axis
#       orbit positions.
#     - Inter-group collisions are absent; only intra-group copy collisions matter.
#       Generator places groups by the full reserved footprint with a moat.

from __future__ import annotations

from collections import Counter
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import get_bg, label_mask


MARK = 2


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= int(r) < int(h) and 0 <= int(c) < int(w)


def _refl_point(center: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    cr, cc = map(int, center)
    return int(2 * cr - r), int(2 * cc - c)


def _refl_row(r0: int, r: int, c: int) -> tuple[int, int]:
    return int(2 * int(r0) - int(r)), int(c)


def _refl_col(c0: int, r: int, c: int) -> tuple[int, int]:
    return int(r), int(2 * int(c0) - int(c))


def _classify_red_geometry(reds: list[tuple[int, int]]) -> tuple[str, object] | None:
    if not reds:
        return None
    if len(reds) == 1:
        return ("point", reds[0])

    row_counts = Counter(int(r) for r, _c in reds)
    col_counts = Counter(int(c) for _r, c in reds)
    rows = set(row_counts)
    cols = set(col_counts)
    if len(rows) == 1:
        return ("row", int(next(iter(rows))))
    if len(cols) == 1:
        return ("col", int(next(iter(cols))))

    r0 = max(sorted(rows), key=lambda r: row_counts[int(r)])
    c0 = max(sorted(cols), key=lambda c: col_counts[int(c)])
    red_set = {(int(r), int(c)) for r, c in reds}
    if (int(r0), int(c0)) not in red_set:
        return None
    if not all(int(r) == int(r0) or int(c) == int(c0) for r, c in reds):
        return None
    has_row_arm = any(int(r) == int(r0) and int(c) != int(c0) for r, c in reds)
    has_col_arm = any(int(c) == int(c0) and int(r) != int(r0) for r, c in reds)
    if has_row_arm and has_col_arm:
        return ("cross", (int(r0), int(c0)))
    return None


def _analyze_marker_components(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    lab, n = label_mask(g != int(bg), connectivity=4)
    comps: list[dict[str, object]] = []
    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if not coords:
            continue
        vals = [int(g[r, c]) for r, c in coords]
        if int(MARK) not in vals:
            continue
        reds = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) == int(MARK)]
        payload = [(int(r), int(c), int(v)) for (r, c), v in zip(coords, vals) if int(v) != int(MARK)]
        comps.append(
            {
                "coords": coords,
                "vals": vals,
                "reds": reds,
                "payload": payload,
                "geometry": _classify_red_geometry(reds),
                "single_red_component": len(coords) == 1 and int(vals[0]) == int(MARK),
            }
        )
    return comps


def _collect_reflection_hits(grid: np.ndarray, bg: int) -> dict[tuple[int, int], list[int]]:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    hits: dict[tuple[int, int], list[int]] = {}

    def emit(rr: int, cc: int, color: int) -> None:
        if _in_bounds((int(h), int(w)), int(rr), int(cc)):
            hits.setdefault((int(rr), int(cc)), []).append(int(color))

    comps = _analyze_marker_components(g)

    # Mixed marker-bearing components use their exact payload pixels.
    for comp in comps:
        if comp["single_red_component"]:
            continue
        geom = comp["geometry"]
        payload = comp["payload"]
        if not geom or not payload:
            continue
        kind, anchor = geom
        for r, c, color in payload:
            targets: set[tuple[int, int]] = set()
            if kind == "row":
                targets.add(_refl_row(int(anchor), int(r), int(c)))
            elif kind == "col":
                targets.add(_refl_col(int(anchor), int(r), int(c)))
            elif kind == "point":
                targets.add(_refl_point(anchor, int(r), int(c)))
            elif kind == "cross":
                cr, cc = anchor
                targets.add(_refl_row(int(cr), int(r), int(c)))
                targets.add(_refl_col(int(cc), int(r), int(c)))
                targets.add(_refl_point((int(cr), int(cc)), int(r), int(c)))
            for rr, cc in targets:
                emit(int(rr), int(cc), int(color))

    # Preserve isolated-red point reflection from the original task.
    for comp in comps:
        if not comp["single_red_component"]:
            continue
        pr, pc = map(int, comp["coords"][0])
        neigh_colors: set[int] = set()
        neigh_pixels: list[tuple[int, int, int]] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = int(pr + dr), int(pc + dc)
                if not _in_bounds((int(h), int(w)), int(rr), int(cc)):
                    continue
                v = int(g[int(rr), int(cc)])
                if v == int(bg) or v == int(MARK):
                    continue
                neigh_colors.add(int(v))
                neigh_pixels.append((int(rr), int(cc), int(v)))

        for color in neigh_colors:
            labv, _ = label_mask(g == int(color), connectivity=8)
            labels = {
                int(labv[int(rr), int(cc)])
                for rr, cc, vv in neigh_pixels
                if int(vv) == int(color) and int(labv[int(rr), int(cc)]) != 0
            }
            for lbl in labels:
                for rr, cc in np.argwhere(labv == int(lbl)):
                    tr, tc = _refl_point((int(pr), int(pc)), int(rr), int(cc))
                    emit(int(tr), int(tc), int(color))

    return hits


def _resolve_hits(grid: np.ndarray, bg: int, hits: dict[tuple[int, int], list[int]], *, collision: bool) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for (r, c), vals in hits.items():
        if collision:
            if int(out[int(r), int(c)]) != int(bg) or len(vals) >= 2:
                out[int(r), int(c)] = int(bg)
            else:
                out[int(r), int(c)] = int(vals[0])
        else:
            out[int(r), int(c)] = int(vals[-1])
    return out


def _overlay_main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    hits = _collect_reflection_hits(g, int(bg))
    return _resolve_hits(g, int(bg), hits, collision=False)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect payloads from red markers, but resolve all mirrored writes with
    collision cancellation. For red crosses this means one payload creates
    row/col/point copies simultaneously, and any overlapping copies disappear.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    hits = _collect_reflection_hits(g, int(bg))
    return _resolve_hits(g, int(bg), hits, collision=True)


def _orbit_positions(dr: int, dc: int) -> list[tuple[int, int]]:
    return [
        (-int(dr), -int(dc)),
        (-int(dr), int(dc)),
        (int(dr), -int(dc)),
        (int(dr), int(dc)),
    ]


def _ring_orbits(qh: int, qw: int) -> list[tuple[int, int]]:
    return [
        (int(dr), int(dc))
        for dr in range(1, int(qh) + 1)
        for dc in range(1, int(qw) + 1)
        if min(int(dr), int(dc)) == 1
    ]


def _build_cross_group(rng: random.Random, color: int, bg: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    qh = rng.randint(3, 4)
    qw = rng.randint(3, 4)
    patch_h = 2 * int(qh) + 1
    patch_w = 2 * int(qw) + 1
    center_r = int(qh)
    center_c = int(qw)

    orbits = _ring_orbits(int(qh), int(qw))
    rng.shuffle(orbits)
    n_single = rng.randint(1, 2)
    n_collision = rng.randint(1, 2)
    total_orbits = int(n_single + n_collision)
    chosen = orbits[:total_orbits]
    single_orbits = chosen[:n_single]
    collision_orbits = chosen[n_single:]

    patch_in = np.full((patch_h, patch_w), int(bg), dtype=int)
    patch_reserved = np.full((patch_h, patch_w), int(bg), dtype=int)
    patch_in[center_r, :] = int(MARK)
    patch_in[:, center_c] = int(MARK)
    patch_reserved[center_r, :] = int(MARK)
    patch_reserved[:, center_c] = int(MARK)

    payload_count = 0
    survivor_slots = 0
    collision_slots = 0

    for dr, dc in single_orbits:
        orbit = _orbit_positions(int(dr), int(dc))
        src = rng.choice(orbit)
        rr = int(center_r + src[0])
        cc = int(center_c + src[1])
        patch_in[int(rr), int(cc)] = int(color)
        payload_count += 1
        survivor_slots += 4
        for odr, odc in orbit:
            patch_reserved[int(center_r + odr), int(center_c + odc)] = int(color)

    for dr, dc in collision_orbits:
        orbit = _orbit_positions(int(dr), int(dc))
        picked = rng.sample(orbit, 2)
        for src in picked:
            rr = int(center_r + src[0])
            cc = int(center_c + src[1])
            patch_in[int(rr), int(cc)] = int(color)
            payload_count += 1
        collision_slots += 4
        for odr, odc in orbit:
            patch_reserved[int(center_r + odr), int(center_c + odc)] = int(color)

    patch_out = main(patch_in)
    overlay = _overlay_main(patch_in)
    if np.array_equal(patch_out, overlay):
        raise RuntimeError("degenerate local group without collision effect")

    meta = {
        "type": "cross",
        "color": int(color),
        "payload_pixels": int(payload_count),
        "single_orbits": int(len(single_orbits)),
        "collision_orbits": int(len(collision_orbits)),
        "survivor_slots": int(survivor_slots),
        "collision_slots": int(collision_slots),
        "delta_vs_overlay": int(np.count_nonzero(patch_out != overlay)),
    }
    return patch_in, patch_out, patch_reserved, meta


def _can_place(occupied: np.ndarray, patch: np.ndarray, bg: int, top: int, left: int, pad: int = 1) -> bool:
    h, w = occupied.shape
    ph, pw = patch.shape
    if int(top) < 0 or int(left) < 0 or int(top + ph) > int(h) or int(left + pw) > int(w):
        return False
    for r, c in np.argwhere(patch != int(bg)):
        rr = int(top + r)
        cc = int(left + c)
        r0 = max(0, int(rr - pad))
        r1 = min(int(h), int(rr + pad + 1))
        c0 = max(0, int(cc - pad))
        c1 = min(int(w), int(cc + pad + 1))
        if occupied[r0:r1, c0:c1].any():
            return False
    return True


def _paste(dst: np.ndarray, patch: np.ndarray, bg: int, top: int, left: int) -> None:
    ph, pw = patch.shape
    window = dst[int(top) : int(top + ph), int(left) : int(left + pw)]
    mask = patch != int(bg)
    window[mask] = patch[mask]


def generate_case(rng: random.Random | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    rng = rng or random.Random()
    palette = [c for c in range(10) if c != int(MARK)]

    for _attempt in range(400):
        h = rng.randint(18, 28)
        w = rng.randint(18, 28)
        bg = int(rng.choice(palette))
        colors = [c for c in palette if int(c) != int(bg)]
        rng.shuffle(colors)

        n_groups = rng.randint(3, 4)
        if len(colors) < int(n_groups):
            continue

        inp = np.full((h, w), int(bg), dtype=int)
        occupied = np.zeros((h, w), dtype=bool)
        group_meta: list[dict[str, object]] = []
        ok = True

        for color in colors[:n_groups]:
            try:
                patch_in, _patch_out, patch_reserved, meta = _build_cross_group(rng, int(color), int(bg))
            except RuntimeError:
                ok = False
                break
            ph, pw = patch_reserved.shape
            placed = False
            for _ in range(150):
                top = rng.randint(0, int(h - ph))
                left = rng.randint(0, int(w - pw))
                if not _can_place(occupied, patch_reserved, int(bg), int(top), int(left), pad=1):
                    continue
                _paste(inp, patch_in, int(bg), int(top), int(left))
                occupied[int(top) : int(top + ph), int(left) : int(left + pw)] |= patch_reserved != int(bg)
                meta = dict(meta)
                meta["top"] = int(top)
                meta["left"] = int(left)
                group_meta.append(meta)
                placed = True
                break
            if not placed:
                ok = False
                break

        if not ok:
            continue

        out = main(inp)
        overlay = _overlay_main(inp)
        if np.array_equal(out, overlay):
            continue

        collision_deleted_pixels = int(np.count_nonzero((overlay != int(bg)) & (out == int(bg))))
        erased_source_pixels = int(
            np.count_nonzero((inp != int(bg)) & (inp != int(MARK)) & (out == int(bg)))
        )
        if collision_deleted_pixels < 8:
            continue
        if erased_source_pixels < n_groups:
            continue

        if not np.array_equal(main(inp), out):
            continue

        meta = {
            "bg": int(bg),
            "grid_shape": (int(h), int(w)),
            "group_count": int(n_groups),
            "groups": group_meta,
            "collision_deleted_pixels": int(collision_deleted_pixels),
            "erased_source_pixels": int(erased_source_pixels),
            "delta_vs_overlay": int(np.count_nonzero(out != overlay)),
        }
        return inp, out, meta

    raise RuntimeError("failed to generate a valid multi-copy collision case")


def generate_input() -> np.ndarray:
    inp, _out, _meta = generate_case()
    return inp


def analyze_case(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    out = main(g)
    overlay = _overlay_main(g)
    comps = _analyze_marker_components(g)
    cross_groups = 0
    payload_pixels = 0
    for comp in comps:
        geom = comp["geometry"]
        if geom and geom[0] == "cross":
            cross_groups += 1
            payload_pixels += len(comp["payload"])
    return {
        "cross_groups": int(cross_groups),
        "payload_pixels": int(payload_pixels),
        "collision_deleted_pixels": int(np.count_nonzero((overlay != int(bg)) & (out == int(bg)))),
        "erased_source_pixels": int(np.count_nonzero((g != int(bg)) & (g != int(MARK)) & (out == int(bg)))),
        "delta_vs_overlay": int(np.count_nonzero(out != overlay)),
    }
