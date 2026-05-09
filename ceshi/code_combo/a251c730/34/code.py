# Combo variant of a251c730 (index 34)
# combo_spec: D9/0+D13/0+D5/2
# Name: 边框色区分+重叠+全网格输出

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location

"""
Design analysis for the combo variant
=====================================

a) Concrete behavior and trace:
   - Original task: two separated uniform-border rectangles sit on striped wallpaper. The source
     rectangle is larger and contains multi-pixel sprites; the destination rectangle contains
     singleton anchor markers; output is a crop of the destination rectangle only.
   - Combo target: keep the border-vs-wallpaper role cue from D9, make the rectangles overlap
     through a partial shared wall from D13, and return the entire 30x30 canvas from D5.
   - Pixel-level concretization used here:
     1. The source rectangle keeps a fully uniform border color that is absent from the wallpaper.
     2. The destination rectangle keeps a wallpaper-colored border on its exposed sides, but on the
        border segment where it literally shares cells with the source wall, those cells must use
        the source border color because one ARC cell cannot display two colors at once.
     3. The destination therefore has a mixed border: wallpaper-color on the non-overlapped parts,
        source-border-color on the shared segment. This preserves visible overlap and still leaves a
        strong border-membership cue.
     4. Output starts from the full input grid, clears only the destination interior, then stamps
        matched sprites there while preserving wallpaper, the source frame, and the shared wall.
   - Example trace:
     original: "find two disjoint frames -> classify by component type / area -> output only dst crop"
     combo: "read wallpaper from the untouched outer margin -> find the uniform non-wallpaper source
     frame -> search for a partially shared destination frame whose exposed border color belongs to
     the wallpaper -> stamp into that destination but return the full 30x30 scene".

b) Complexity calibration:
   - The overlap is constrained to a single straight shared wall segment of length >= 4 and strictly
     shorter than both participating sides, so the change is visually obvious but not chaotic.
   - The destination still contains 2-3 singleton markers and the source still contains 2-4 copies
     of 1-2 sprite types, so the anchor-color matching rule has repeated support in each sample.
   - Area is intentionally unconstrained: source may be smaller, larger, or equal in area. This
     prevents silent fallback to the original "source area > destination area" heuristic.

c) Implicit assumptions audit:
   - Old assumption broken: both rectangles have uniform borders. In the combo, only the source
     remains fully uniform; the destination border is mixed because the shared wall is visible in
     the source border color.
   - Old assumption broken: rectangles are disjoint. The combo instead requires a positive shared
     border segment and zero interior overlap.
   - Old assumption broken: wallpaper must be inferred from cells outside both rectangles. The combo
     generator now reserves a 1-cell global margin so the outer frame of the 30x30 grid is pure
     wallpaper, giving the solver a stable place to recover stripe orientation and palette.
   - Old assumption broken: output is a cropped destination frame. The new solver must mutate the
     destination in-place on a full-grid canvas.

d) Placement legality:
   - The generator places both rectangles inside a 28x28 usable window, leaving a 1-cell wallpaper
     margin on every side of the full 30x30 canvas.
   - Side-by-side layouts share one border column; stacked layouts share one border row.
   - Shared length is always >= 4 and < min(side lengths), so the wall overlap is partial.
   - The two interiors never overlap. Markers are still placed output-first so every stamped sprite
     fits entirely inside the destination interior.

e) Visual observability:
   - A human sees three simultaneous novelties across examples:
     1. the full wallpaper scene is preserved in the output,
     2. the rectangles touch via a visibly shared wall segment,
     3. role identity comes from border-vs-wallpaper membership instead of area.
   - The source border color never appears in the wallpaper, while the destination border color does
     appear repeatedly in the wallpaper stripes on its exposed sides, making the distinction legible.

f) Visual cleanliness:
   - Only one straight border segment is shared; no sprite or marker is hidden under another object.
   - The destination interior stays a clean marker field in the input and a clean stamped field in
     the output, while the source remains untouched.

g) Change inferability:
   - No step depends on numeric color ordering. Anchor detection uses the uniquely least-frequent
     color within each sprite component, which the generator guarantees to be a singleton pixel.
   - The mixed destination border is structural, not arbitrary: exactly the overlapped wall segment
     switches to the source border color, and nowhere else.

h) New solver assumptions and generator guards:
   - main() assumes a pure wallpaper outer margin exists; generate_input() enforces that.
   - main() assumes exactly one uniform non-wallpaper source frame and one valid destination frame
     adjacent through a partial shared wall; generate_input() verifies each sample by calling main().
   - main() assumes every destination marker color matches some source anchor color; generator
     constructs markers directly from source sprite types.
"""

# === CFG: canvas / colors ===
GRID_SIZE = 30
OUTER_MARGIN = 1
WALLPAPER_COLORS = (4, 5)
TOTAL_COLORS_TARGET = (7, 8, 9)

# === CFG: rectangle sizes ===
SRC_H_RANGE = (9, 19)
SRC_W_RANGE = (8, 18)
DST_H_RANGE = (9, 19)
DST_W_RANGE = (8, 20)
MIN_RECT_H = 6
MIN_RECT_W = 6
SHARED_SEGMENT_MIN = 4
AREA_RELATION_CHOICES = ("src_smaller", "src_larger", "equal")
DIMENSION_RELATION_RETRIES = 80

# === CFG: sprite / placement ===
SPRITE_TYPE_CHOICES = (1, 2)
MARKER_COUNT_RANGE = (2, 3)
SOURCE_SPRITE_RANGE = (2, 4)
MARKER_PLACE_RETRIES = 25
SOURCE_SPRITE_PADDING = 1

# === CFG: generation retries ===
MAX_GEN_ATTEMPTS = 500

SPRITE_SHAPES = [
    (("AB.", "BBB"), (0, 0)),
    (("ABB", "BBB"), (0, 0)),
    ((".A.", "BBB", ".B."), (0, 1)),
    (("AB.", "BBB", ".BB"), (0, 0)),
    (("ABB", "BBB", "..B"), (0, 0)),
    ((".AB", "BBB", "BB."), (0, 1)),
    (("B.B", "BBB", ".A."), (2, 1)),
    ((".B.", "BAB", ".B."), (1, 1)),
    (("BBB", "BAB"), (1, 1)),
    (("AB.", "BBB", ".B."), (0, 0)),
    (("BAB", "BBB"), (0, 1)),
]


def _render_sprite(shape_rows: tuple[str, ...], body: int, anchor: int, bg: int) -> np.ndarray:
    char_map = {"B": body, "A": anchor}
    rows = [[char_map.get(ch, bg) for ch in row] for row in shape_rows]
    return np.array(rows, dtype=int)


def _make_wallpaper(direction: str, palette: list[int]) -> np.ndarray:
    period = len(palette)
    stripe = np.array(palette * ((GRID_SIZE + period - 1) // period), dtype=int)[:GRID_SIZE]
    if direction == "vertical":
        return np.tile(stripe, (GRID_SIZE, 1))
    return np.tile(stripe.reshape(-1, 1), (1, GRID_SIZE))


def _build_frame(h: int, w: int, border: int, bg: int) -> np.ndarray:
    rect = np.full((h, w), bg, dtype=int)
    rect[0, :] = rect[-1, :] = border
    rect[:, 0] = rect[:, -1] = border
    return rect


def _choose_counts(n_total: int, n_types: int) -> list[int]:
    counts = [1] * n_types
    for _ in range(n_total - n_types):
        counts[random.randrange(n_types)] += 1
    return counts


def _periodic_palette(seq: list[int]) -> list[int] | None:
    for p in WALLPAPER_COLORS:
        if len(seq) < p:
            continue
        head = seq[:p]
        if len(set(head)) != p:
            continue
        if all(int(seq[i]) == int(head[i % p]) for i in range(len(seq))):
            return [int(v) for v in head]
    return None


def _recover_wallpaper(g: np.ndarray) -> tuple[str, list[int]]:
    top = [int(v) for v in g[0, :]]
    palette = _periodic_palette(top)
    if palette is not None:
        return "vertical", palette
    left = [int(v) for v in g[:, 0]]
    palette = _periodic_palette(left)
    if palette is not None:
        return "horizontal", palette
    raise AssertionError("failed to recover wallpaper from outer margin")


def _interior_bg(g: np.ndarray, rect: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = map(int, rect)
    inner = g[r0 + 1 : r1, c0 + 1 : c1]
    return int(Counter(map(int, inner.flatten())).most_common(1)[0][0])


def _rect_components(g: np.ndarray, rect: tuple[int, int, int, int]) -> tuple[int, list[dict]]:
    r0, c0, r1, c1 = map(int, rect)
    inner = g[r0 + 1 : r1, c0 + 1 : c1]
    ibg = int(Counter(map(int, inner.flatten())).most_common(1)[0][0])
    comps: list[dict] = []

    for coords in components_mask(inner != ibg, connectivity=4):
        if coords.size == 0:
            continue
        rr0, cc0, rr1, cc1 = bbox_coords(coords)
        mask = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
        mask[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        patch = inner[rr0 : rr1 + 1, cc0 : cc1 + 1].copy()
        patch[~mask] = 0
        vals = [int(inner[int(r), int(c)]) for r, c in coords]
        comps.append(
            {
                "size": int(coords.shape[0]),
                "colors": dict(Counter(vals)),
                "bbox": (int(rr0), int(cc0), int(rr1), int(cc1)),
                "mask": mask,
                "patch": patch,
                "point": (int(coords[0, 0]), int(coords[0, 1])),
            }
        )

    comps.sort(key=lambda d: (-int(d["size"]), int(d["bbox"][0]), int(d["bbox"][1])))
    return int(ibg), comps


def _extract_anchor(comp: dict) -> tuple[int, tuple[int, int]]:
    counts = {int(k): int(v) for k, v in comp["colors"].items()}
    if not counts:
        raise AssertionError("sprite component has no colors")
    min_count = min(counts.values())
    anchor_colors = [color for color, count in counts.items() if int(count) == int(min_count)]
    if len(anchor_colors) != 1:
        raise AssertionError("sprite anchor color must be uniquely least frequent")
    anchor_color = int(anchor_colors[0])
    anchors = np.argwhere((comp["patch"] == int(anchor_color)) & comp["mask"])
    if anchors.shape[0] != 1:
        raise AssertionError("anchor color must occur exactly once")
    ar, ac = map(int, anchors[0])
    return anchor_color, (ar, ac)


def _build_templates(src_comps: list[dict]) -> dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]]:
    templates: dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]] = {}
    for comp in src_comps:
        anchor_color, anchor_rc = _extract_anchor(comp)
        if anchor_color not in templates or int(comp["mask"].sum()) > int(templates[anchor_color][1].sum()):
            templates[anchor_color] = (comp["patch"].copy(), comp["mask"].copy(), anchor_rc)
    return templates


def _find_source_candidates(g: np.ndarray, wallpaper_palette: list[int]) -> list[dict]:
    H, W = g.shape
    candidates: list[dict] = []

    for r0 in range(H):
        for c0 in range(W):
            bc = int(g[r0, c0])
            if bc in wallpaper_palette:
                continue
            for r1 in range(r0 + MIN_RECT_H - 1, H):
                for c1 in range(c0 + MIN_RECT_W - 1, W):
                    if int(g[r0, c1]) != bc or int(g[r1, c0]) != bc or int(g[r1, c1]) != bc:
                        continue
                    if not bool((g[r0, c0 : c1 + 1] == bc).all()):
                        continue
                    if not bool((g[r1, c0 : c1 + 1] == bc).all()):
                        continue
                    if not bool((g[r0 : r1 + 1, c0] == bc).all()):
                        continue
                    if not bool((g[r0 : r1 + 1, c1] == bc).all()):
                        continue
                    rect = (int(r0), int(c0), int(r1), int(c1))
                    ibg = _interior_bg(g, rect)
                    if ibg == bc:
                        continue
                    ibg2, comps = _rect_components(g, rect)
                    sizes = [int(comp["size"]) for comp in comps]
                    if not sizes or not all(sz > 1 for sz in sizes):
                        continue
                    if not (2 <= len(sizes) <= 4):
                        continue
                    candidates.append(
                        {
                            "rect": rect,
                            "bc": int(bc),
                            "bg": int(ibg2),
                            "area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
                            "comps": comps,
                        }
                    )
    return candidates


def _interval_overlap(a0: int, a1: int, b0: int, b1: int) -> tuple[int, int, int]:
    lo = max(int(a0), int(b0))
    hi = min(int(a1), int(b1))
    if lo > hi:
        return 0, 0, 0
    return int(lo), int(hi), int(hi - lo + 1)


def _evaluate_destination(
    g: np.ndarray,
    rect: tuple[int, int, int, int],
    shared_side: str,
    src_rect: tuple[int, int, int, int],
    src_bc: int,
    wallpaper_palette: list[int],
) -> dict | None:
    r0, c0, r1, c1 = map(int, rect)
    sr0, sc0, sr1, sc1 = map(int, src_rect)
    if r1 - r0 + 1 < MIN_RECT_H or c1 - c0 + 1 < MIN_RECT_W:
        return None

    if shared_side == "left":
        if c0 != sc1:
            return None
        shared_lo, shared_hi, shared_len = _interval_overlap(r0, r1, sr0, sr1)
        local_shared = (int(shared_lo - r0), int(shared_hi - r0))
    elif shared_side == "right":
        if c1 != sc0:
            return None
        shared_lo, shared_hi, shared_len = _interval_overlap(r0, r1, sr0, sr1)
        local_shared = (int(shared_lo - r0), int(shared_hi - r0))
    elif shared_side == "top":
        if r0 != sr1:
            return None
        shared_lo, shared_hi, shared_len = _interval_overlap(c0, c1, sc0, sc1)
        local_shared = (int(shared_lo - c0), int(shared_hi - c0))
    else:
        if r1 != sr0:
            return None
        shared_lo, shared_hi, shared_len = _interval_overlap(c0, c1, sc0, sc1)
        local_shared = (int(shared_lo - c0), int(shared_hi - c0))

    if shared_len < SHARED_SEGMENT_MIN:
        return None
    side_len = int((r1 - r0 + 1) if shared_side in ("left", "right") else (c1 - c0 + 1))
    src_side_len = int((sr1 - sr0 + 1) if shared_side in ("left", "right") else (sc1 - sc0 + 1))
    if shared_len >= min(side_len, src_side_len):
        return None

    non_shared_border: list[int] = []
    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            if rr not in (r0, r1) and cc not in (c0, c1):
                continue
            on_shared = False
            if shared_side == "left" and cc == c0 and shared_lo <= rr <= shared_hi:
                on_shared = True
            elif shared_side == "right" and cc == c1 and shared_lo <= rr <= shared_hi:
                on_shared = True
            elif shared_side == "top" and rr == r0 and shared_lo <= cc <= shared_hi:
                on_shared = True
            elif shared_side == "bottom" and rr == r1 and shared_lo <= cc <= shared_hi:
                on_shared = True

            val = int(g[rr, cc])
            if on_shared:
                if val != int(src_bc):
                    return None
            else:
                non_shared_border.append(val)

    if not non_shared_border:
        return None
    if len(set(non_shared_border)) != 1:
        return None
    dst_bc = int(non_shared_border[0])
    if dst_bc not in wallpaper_palette or dst_bc == int(src_bc):
        return None

    ibg = _interior_bg(g, rect)
    if ibg == dst_bc or ibg == src_bc:
        return None
    ibg2, comps = _rect_components(g, rect)
    sizes = [int(comp["size"]) for comp in comps]
    if not sizes or not all(sz == 1 for sz in sizes):
        return None
    if not (2 <= len(sizes) <= 3):
        return None

    return {
        "rect": rect,
        "bc": int(dst_bc),
        "bg": int(ibg2),
        "area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
        "comps": comps,
        "shared_side": shared_side,
        "shared_len": int(shared_len),
        "shared_local": local_shared,
    }


def _destination_candidates_for_source(g: np.ndarray, src: dict, wallpaper_palette: list[int]) -> list[dict]:
    H, W = g.shape
    sr0, sc0, sr1, sc1 = map(int, src["rect"])
    out: list[dict] = []

    for dr0 in range(0, H - MIN_RECT_H + 1):
        for dr1 in range(dr0 + MIN_RECT_H - 1, H):
            for dc1 in range(sc1 + MIN_RECT_W - 1, W):
                cand = _evaluate_destination(g, (dr0, sc1, dr1, dc1), "left", src["rect"], src["bc"], wallpaper_palette)
                if cand is not None:
                    out.append(cand)
            for dc0 in range(0, sc0 - MIN_RECT_W + 2):
                cand = _evaluate_destination(g, (dr0, dc0, dr1, sc0), "right", src["rect"], src["bc"], wallpaper_palette)
                if cand is not None:
                    out.append(cand)

    for dc0 in range(0, W - MIN_RECT_W + 1):
        for dc1 in range(dc0 + MIN_RECT_W - 1, W):
            for dr1 in range(sr1 + MIN_RECT_H - 1, H):
                cand = _evaluate_destination(g, (sr1, dc0, dr1, dc1), "top", src["rect"], src["bc"], wallpaper_palette)
                if cand is not None:
                    out.append(cand)
            for dr0 in range(0, sr0 - MIN_RECT_H + 2):
                cand = _evaluate_destination(g, (dr0, dc0, sr0, dc1), "bottom", src["rect"], src["bc"], wallpaper_palette)
                if cand is not None:
                    out.append(cand)

    return out


def _analyze_scene(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    direction, wallpaper_palette = _recover_wallpaper(g)
    best: dict | None = None
    best_score: tuple[int, int, int, int] | None = None

    for src in _find_source_candidates(g, wallpaper_palette):
        try:
            templates = _build_templates(src["comps"])
        except AssertionError:
            continue
        if not templates:
            continue
        for dst in _destination_candidates_for_source(g, src, wallpaper_palette):
            marker_colors = []
            valid = True
            for comp in dst["comps"]:
                cols = list(comp["colors"].keys())
                if len(cols) != 1:
                    valid = False
                    break
                marker_colors.append(int(cols[0]))
            if not valid or any(color not in templates for color in marker_colors):
                continue
            area_delta = abs(int(src["area"]) - int(dst["area"]))
            score = (int(dst["shared_len"]), -int(area_delta), int(src["area"] + dst["area"]), len(marker_colors))
            if best_score is None or score > best_score:
                best_score = score
                best = {
                    "direction": direction,
                    "wallpaper_palette": wallpaper_palette,
                    "src": src,
                    "dst": dst,
                    "templates": templates,
                }

    if best is None:
        raise AssertionError("failed to isolate source and destination rectangles")
    return best


def _stamp_full_grid(grid: np.ndarray, scene: dict) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    dst = scene["dst"]
    dr0, dc0, dr1, dc1 = map(int, dst["rect"])
    out[dr0 + 1 : dr1, dc0 + 1 : dc1] = int(dst["bg"])

    for comp in dst["comps"]:
        cols = list(comp["colors"].keys())
        if len(cols) != 1:
            raise AssertionError("destination marker must be monochrome singleton")
        marker_color = int(cols[0])
        patch, mask, (ar, ac) = scene["templates"][marker_color]
        rr, cc = map(int, comp["point"])
        top = int(dr0 + 1 + rr - ar)
        left = int(dc0 + 1 + cc - ac)
        paste(out, patch, top, left, mask=mask, overwrite=True, clip=True)

    return out


def _fusion_metrics_from_grid(grid: np.ndarray) -> dict:
    scene = _analyze_scene(np.array(grid, dtype=int))
    src = scene["src"]
    dst = scene["dst"]
    sr0, sc0, sr1, sc1 = map(int, src["rect"])
    dr0, dc0, dr1, dc1 = map(int, dst["rect"])
    overlap_r0 = max(sr0, dr0)
    overlap_c0 = max(sc0, dc0)
    overlap_r1 = min(sr1, dr1)
    overlap_c1 = min(sc1, dc1)
    bbox_overlap_area = 0
    if overlap_r0 <= overlap_r1 and overlap_c0 <= overlap_c1:
        bbox_overlap_area = int((overlap_r1 - overlap_r0 + 1) * (overlap_c1 - overlap_c0 + 1))
    src_area = int(src["area"])
    dst_area = int(dst["area"])
    if src_area < dst_area:
        area_relation = "src_smaller"
    elif src_area > dst_area:
        area_relation = "src_larger"
    else:
        area_relation = "equal"
    return {
        "shared_len": int(dst["shared_len"]),
        "shared_side": dst["shared_side"],
        "bbox_overlap_area": int(bbox_overlap_area),
        "area_relation": area_relation,
        "src_border_in_wallpaper": bool(int(src["bc"]) in scene["wallpaper_palette"]),
        "dst_border_in_wallpaper": bool(int(dst["bc"]) in scene["wallpaper_palette"]),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp source sprites into an overlapped destination frame while preserving the full wallpaper scene.

    Input structure:
    - A 30x30 striped wallpaper contains a uniform-border source rectangle and a partially overlapped destination rectangle.
    - The source border color is absent from the wallpaper palette; the destination's exposed border color belongs to the wallpaper palette.
    - The destination shares one straight wall segment with the source, so that shared segment visibly uses the source border color.

    Rule:
    - Recover the wallpaper from the untouched outer margin, detect the uniform non-wallpaper source frame, and find the adjacent destination frame that has singleton markers and a partially shared wall.
    - Match each destination marker color to the source sprite with the same anchor color, clear only the destination interior, and stamp those sprites there with anchor alignment.
    - Return the entire 30x30 grid, leaving wallpaper, source rectangle, and the shared wall intact.

    Key insight:
    - Rectangle identity now comes from the interaction of wallpaper membership and overlap geometry: the source is the only fully uniform non-wallpaper frame, while the destination is the marker frame whose exposed border matches the wallpaper but whose shared wall is borrowed from the source.
    """
    scene = _analyze_scene(np.array(grid, dtype=int))
    return _stamp_full_grid(np.array(grid, dtype=int), scene)


def _assign_palette(n_types: int) -> tuple[list[int], int, int, list[int], list[int], int | None] | None:
    wallpaper_n = 5 if n_types == 1 else random.choice(list(WALLPAPER_COLORS))
    base_total = wallpaper_n + 1 + n_types
    allowed_totals = [total for total in TOTAL_COLORS_TARGET if base_total <= total <= base_total + 1]
    if not allowed_totals:
        return None
    used_total_target = random.choice(allowed_totals)
    body_extra = used_total_target == base_total + 1

    all_colors = list(range(10))
    wallpaper = random.sample(all_colors, wallpaper_n)
    remaining = [c for c in all_colors if c not in wallpaper]
    random.shuffle(remaining)

    src_border = remaining.pop()
    anchors = [remaining.pop() for _ in range(n_types)]
    dst_border = random.choice(wallpaper)

    body_pool = wallpaper[:]
    forced_body: int | None = None
    if body_extra:
        if not remaining:
            return None
        forced_body = remaining.pop()
        body_pool.append(forced_body)

    return wallpaper, src_border, dst_border, anchors, body_pool, forced_body


def _place_markers(
    sprite_types: list[dict], marker_type_ids: list[int], dst_h: int, dst_w: int, dst_bg: int
) -> tuple[np.ndarray, list[tuple[int, int, int]]] | None:
    inner = np.full((dst_h - 2, dst_w - 2), dst_bg, dtype=int)
    marker_points: list[tuple[int, int, int]] = []
    for type_id in marker_type_ids:
        sprite = sprite_types[type_id]
        patch = _render_sprite(sprite["shape_rows"], sprite["body"], sprite["anchor"], dst_bg)
        placed = False
        for _ in range(MARKER_PLACE_RETRIES):
            try:
                r0, c0 = random_free_location(inner, patch, transparent=dst_bg)
            except ValueError:
                return None
            mr = int(r0 + sprite["anchor_rc"][0])
            mc = int(c0 + sprite["anchor_rc"][1])
            if any(abs(mr - pr) + abs(mc - pc) <= 1 for pr, pc, _ in marker_points):
                continue
            paste(inner, patch, r0, c0, transparent=dst_bg)
            marker_points.append((mr, mc, type_id))
            placed = True
            break
        if not placed:
            return None
    return inner, marker_points


def _place_sources(
    sprite_types: list[dict], source_type_ids: list[int], src_h: int, src_w: int, src_bg: int
) -> np.ndarray | None:
    inner = np.full((src_h - 2, src_w - 2), src_bg, dtype=int)
    for type_id in source_type_ids:
        sprite = sprite_types[type_id]
        patch = _render_sprite(sprite["shape_rows"], sprite["body"], sprite["anchor"], src_bg)
        try:
            r0, c0 = random_free_location(
                inner,
                patch,
                transparent=src_bg,
                border=SOURCE_SPRITE_PADDING,
                padding=SOURCE_SPRITE_PADDING,
            )
        except ValueError:
            return None
        paste(inner, patch, r0, c0, transparent=src_bg)
    return inner


def _interval_pairs(len_a: int, len_b: int, span: int) -> list[tuple[int, int, int]]:
    pairs: list[tuple[int, int, int]] = []
    for a0 in range(0, span - len_a + 1):
        a1 = a0 + len_a - 1
        for b0 in range(0, span - len_b + 1):
            b1 = b0 + len_b - 1
            overlap = min(a1, b1) - max(a0, b0) + 1
            if overlap < SHARED_SEGMENT_MIN:
                continue
            if overlap >= min(len_a, len_b):
                continue
            pairs.append((int(a0), int(b0), int(overlap)))
    return pairs


def _dims_overlap_feasible(src_h: int, src_w: int, dst_h: int, dst_w: int) -> bool:
    usable = GRID_SIZE - 2 * OUTER_MARGIN
    return bool(src_w + dst_w - 1 <= usable or src_h + dst_h - 1 <= usable)


def _choose_dimensions(area_relation: str) -> tuple[int, int, int, int]:
    if area_relation == "equal":
        h_lo = max(SRC_H_RANGE[0], DST_H_RANGE[0])
        h_hi = min(SRC_H_RANGE[1], DST_H_RANGE[1])
        w_lo = max(SRC_W_RANGE[0], DST_W_RANGE[0])
        w_hi = min(SRC_W_RANGE[1], DST_W_RANGE[1])
        if h_lo > h_hi or w_lo > w_hi:
            raise ValueError("equal-area dimensions unavailable")
        for _ in range(DIMENSION_RELATION_RETRIES):
            h = random.randint(h_lo, h_hi)
            w = random.randint(w_lo, w_hi)
            if _dims_overlap_feasible(h, w, h, w):
                return h, w, h, w
        raise ValueError("failed to sample feasible equal-area dimensions")

    for _ in range(DIMENSION_RELATION_RETRIES):
        src_h = random.randint(*SRC_H_RANGE)
        src_w = random.randint(*SRC_W_RANGE)
        dst_h = random.randint(*DST_H_RANGE)
        dst_w = random.randint(*DST_W_RANGE)
        if not _dims_overlap_feasible(src_h, src_w, dst_h, dst_w):
            continue
        src_area = int(src_h * src_w)
        dst_area = int(dst_h * dst_w)
        if area_relation == "src_smaller" and src_area < dst_area:
            return src_h, src_w, dst_h, dst_w
        if area_relation == "src_larger" and src_area > dst_area:
            return src_h, src_w, dst_h, dst_w

    raise ValueError(f"failed to sample dimensions for relation: {area_relation}")


def _place_rectangles(
    src_h: int, src_w: int, dst_h: int, dst_w: int
) -> tuple[tuple[int, int], tuple[int, int], str, str, int]:
    usable = GRID_SIZE - 2 * OUTER_MARGIN
    options: list[tuple[str, str, list[tuple[int, int, int]]]] = []
    if src_w + dst_w - 1 <= usable:
        pairs = _interval_pairs(src_h, dst_h, usable)
        if pairs:
            options.append(("side_by_side", "src_left", pairs))
            options.append(("side_by_side", "src_right", pairs))
    if src_h + dst_h - 1 <= usable:
        pairs = _interval_pairs(src_w, dst_w, usable)
        if pairs:
            options.append(("stacked", "src_top", pairs))
            options.append(("stacked", "src_bottom", pairs))
    if not options:
        raise ValueError("no overlapping placement fits")

    layout, mode, pairs = random.choice(options)
    a0, b0, shared = random.choice(pairs)

    if layout == "side_by_side":
        union_w = src_w + dst_w - 1
        left_margin = random.randint(0, usable - union_w)
        if mode == "src_left":
            src_pos = (OUTER_MARGIN + a0, OUTER_MARGIN + left_margin)
            dst_pos = (OUTER_MARGIN + b0, OUTER_MARGIN + left_margin + src_w - 1)
            shared_side = "left"
        else:
            dst_pos = (OUTER_MARGIN + b0, OUTER_MARGIN + left_margin)
            src_pos = (OUTER_MARGIN + a0, OUTER_MARGIN + left_margin + dst_w - 1)
            shared_side = "right"
    else:
        union_h = src_h + dst_h - 1
        top_margin = random.randint(0, usable - union_h)
        if mode == "src_top":
            src_pos = (OUTER_MARGIN + top_margin, OUTER_MARGIN + a0)
            dst_pos = (OUTER_MARGIN + top_margin + src_h - 1, OUTER_MARGIN + b0)
            shared_side = "top"
        else:
            dst_pos = (OUTER_MARGIN + top_margin, OUTER_MARGIN + b0)
            src_pos = (OUTER_MARGIN + top_margin + dst_h - 1, OUTER_MARGIN + a0)
            shared_side = "bottom"

    return src_pos, dst_pos, layout, shared_side, int(shared)


def _shared_local_interval(
    src_pos: tuple[int, int], src_shape: tuple[int, int], dst_pos: tuple[int, int], dst_shape: tuple[int, int], shared_side: str
) -> tuple[int, int]:
    sr0, sc0 = map(int, src_pos)
    dr0, dc0 = map(int, dst_pos)
    src_h, src_w = map(int, src_shape)
    dst_h, dst_w = map(int, dst_shape)
    if shared_side in ("left", "right"):
        lo = max(sr0, dr0)
        hi = min(sr0 + src_h - 1, dr0 + dst_h - 1)
        return int(lo - dr0), int(hi - dr0)
    lo = max(sc0, dc0)
    hi = min(sc0 + src_w - 1, dc0 + dst_w - 1)
    return int(lo - dc0), int(hi - dc0)


def _apply_shared_wall(rect: np.ndarray, shared_side: str, local_interval: tuple[int, int], shared_color: int) -> None:
    lo, hi = map(int, local_interval)
    if shared_side == "left":
        rect[lo : hi + 1, 0] = int(shared_color)
    elif shared_side == "right":
        rect[lo : hi + 1, -1] = int(shared_color)
    elif shared_side == "top":
        rect[0, lo : hi + 1] = int(shared_color)
    else:
        rect[-1, lo : hi + 1] = int(shared_color)


def _gen_attempt() -> np.ndarray | None:
    n_types = random.choice(list(SPRITE_TYPE_CHOICES))

    palette_result = _assign_palette(n_types)
    if palette_result is None:
        return None
    wallpaper, src_border, dst_border, anchors, body_pool, forced_body = palette_result

    src_bg = random.choice(wallpaper)
    dst_bg_choices = [c for c in wallpaper if c != dst_border]
    if not dst_bg_choices:
        return None
    dst_bg = random.choice(dst_bg_choices)

    sprite_specs = random.sample(SPRITE_SHAPES, n_types)
    bodies: list[int] = []
    for i in range(n_types):
        choices = [c for c in body_pool if c not in anchors and c != src_bg and c != dst_bg]
        if not choices:
            return None
        if i == 0 and forced_body is not None and forced_body in choices:
            bodies.append(forced_body)
        else:
            bodies.append(random.choice(choices))

    sprite_types = [
        {
            "shape_rows": shape_rows,
            "anchor_rc": anchor_rc,
            "body": body,
            "anchor": anchor,
            "h": len(shape_rows),
            "w": len(shape_rows[0]),
        }
        for (shape_rows, anchor_rc), body, anchor in zip(sprite_specs, bodies, anchors)
    ]

    n_markers = random.randint(*MARKER_COUNT_RANGE)
    marker_type_ids = [i for i, ct in enumerate(_choose_counts(n_markers, n_types)) for _ in range(ct)]
    random.shuffle(marker_type_ids)

    n_source = random.randint(max(2, n_types), SOURCE_SPRITE_RANGE[1])
    source_type_ids = [i for i, ct in enumerate(_choose_counts(n_source, n_types)) for _ in range(ct)]
    random.shuffle(source_type_ids)

    target_area_relation = random.choice(list(AREA_RELATION_CHOICES))
    try:
        src_h, src_w, dst_h, dst_w = _choose_dimensions(target_area_relation)
    except ValueError:
        return None
    try:
        src_pos, dst_pos, _layout, shared_side, shared_len = _place_rectangles(src_h, src_w, dst_h, dst_w)
    except ValueError:
        return None
    if shared_len < SHARED_SEGMENT_MIN:
        return None

    marker_result = _place_markers(sprite_types, marker_type_ids, dst_h, dst_w, dst_bg)
    if marker_result is None:
        return None
    out_inner, marker_points = marker_result

    source_inner = _place_sources(sprite_types, source_type_ids, src_h, src_w, src_bg)
    if source_inner is None:
        return None

    local_shared = _shared_local_interval(src_pos, (src_h, src_w), dst_pos, (dst_h, dst_w), shared_side)

    source_rect = _build_frame(src_h, src_w, src_border, src_bg)
    paste(source_rect[1:-1, 1:-1], source_inner, 0, 0)

    dest_input = _build_frame(dst_h, dst_w, dst_border, dst_bg)
    _apply_shared_wall(dest_input, shared_side, local_shared, src_border)
    for mr, mc, type_id in marker_points:
        dest_input[1 + mr, 1 + mc] = int(sprite_types[type_id]["anchor"])

    dest_output = _build_frame(dst_h, dst_w, dst_border, dst_bg)
    _apply_shared_wall(dest_output, shared_side, local_shared, src_border)
    paste(dest_output[1:-1, 1:-1], out_inner, 0, 0)

    direction = random.choice(["vertical", "horizontal"])
    grid = _make_wallpaper(direction, wallpaper)
    paste(grid, source_rect, src_pos[0], src_pos[1])
    paste(grid, dest_input, dst_pos[0], dst_pos[1])

    output = np.array(grid, copy=True)
    paste(output, dest_output, dst_pos[0], dst_pos[1], overwrite=True)

    try:
        metrics = _fusion_metrics_from_grid(grid)
    except AssertionError:
        return None
    if metrics["shared_len"] < SHARED_SEGMENT_MIN or metrics["bbox_overlap_area"] <= 0:
        return None
    if metrics["src_border_in_wallpaper"] or not metrics["dst_border_in_wallpaper"]:
        return None

    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a251c730 combo-34 input after {MAX_GEN_ATTEMPTS} attempts")
