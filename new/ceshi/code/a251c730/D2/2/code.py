# Variant of a251c730
# Dimension: D2 — 源对象：精灵结构 (src)
# Change: 源矩形中有2-4个精灵，1-2种类型，每种精灵为2x3或3x3外接矩形、5-7像素、恰好2种颜色（主体色=多数、锚点色=单一像素）→ 对角像素对作为源对象

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import paste


# -----------------------------------------------------------------------------
# Design analysis for the D2 variant
#
# variant_spec:
#   core_change:
#     source_object: each source object is now only a same-color diagonal pair
#       sitting on opposite corners of a 2x3 or 3x3 implied bbox
#     transfer_rule: destination markers still match by color, but the copied
#       thing is no longer an observed dense sprite; instead the solver must
#       infer the bbox from the source pair and stamp the entire implied
#       rectangle in that color
#     anchor_rule: the destination singleton aligns to the upper pixel of the
#       source pair, so TL-BR pairs grow down-right while TR-BL pairs grow
#       down-left
#   a_concretization_and_trace:
#     source_rect:
#       - contains 2-4 implied mini-rectangles
#       - each object shows only two colored pixels
#       - all copies of one type share the same color and geometry
#       - geometry choices are limited to:
#         * 2x3 TL-BR
#         * 2x3 TR-BL
#         * 3x3 TL-BR
#         * 3x3 TR-BL
#     destination_rect:
#       - still contains 2-3 singleton markers keyed by type color
#     output:
#       - destination frame cropped out
#       - each marker expands to a solid 2x3 or 3x3 block
#     trace_example:
#       - old task: a source component already contains a full 3x3 sprite;
#         solver extracts that patch and pastes it at the marker
#       - new task: source only shows (r,c) and (r+2,c+2), both same color;
#         solver infers a 3x3 bbox, treats (r,c) as the anchor, and stamps a
#         3x3 solid block at the destination marker
#       - difference: copying visible pixels becomes reconstructing hidden area
#   b_complexity_calibration:
#     - color still indexes type, preserving the original match structure
#     - geometry has only four visible templates, so the rule stays learnable
#     - repeated copies per type ensure the source is read as an object class,
#       not a one-off coincidence
#     - destination markers are still sparse singletons, preserving the original
#       source/destination contrast
#   c_old_assumptions_to_replace:
#     - old solver assumes source contains multi-cell connected components:
#       invalid now, because source objects are disconnected singleton pixels
#     - old solver finds anchors via least-frequent color inside a component:
#       invalid now, because each source object has only one visible color
#     - old solver classifies destination as "all singleton components":
#       no longer sufficient, because source also contains only singleton pixels
#     - replacement: source is detected by same-color pixels that can be parsed
#       as repeated diagonal pairs with a consistent delta
#   d_placement_legality:
#     - source placement reserves the whole implied bbox plus padding, not only
#       the two visible pixels, to prevent overlaps and cross-pair ambiguities
#     - destination placement reserves the eventual stamped bbox, so outputs do
#       not overlap
#     - same-color destination markers are forbidden from forming source-like
#       diagonal offsets, preventing the destination frame from masquerading as
#       another source frame
#   e_visual_observability:
#     - the transformation is visible in every sample: sparse diagonal clues in
#       the source expand into solid mini-rectangles in the output
#     - left-leaning vs right-leaning pairs matter because the marker aligns to
#       the top pixel, making the stamp grow to different horizontal sides
#   f_visual_tidiness:
#     - only two framed regions remain
#     - source bboxes keep padding, so the disconnected pairs do not visually
#       collide into misleading clusters
#     - destination stamps are placed without overlap
#   g_color_inference:
#     - there is no hidden color remapping; a marker color simply selects a
#       source geometry of the same color
#   new_main_assumptions_checked_against_generator:
#     - for any source type color, all its pixels follow one consistent delta
#       from the set {(1,+2),(1,-2),(2,+2),(2,-2)}
#     - no extra same-color cross-pairs exist
#     - every destination marker color appears in the source templates
#     - the chosen marker positions keep the full inferred stamp in-bounds
# -----------------------------------------------------------------------------


PAIR_DELTAS = ((1, 2), (1, -2), (2, 2), (2, -2))
GEOM_SPECS = (
    {"h": 2, "w": 3, "dc": 2},
    {"h": 2, "w": 3, "dc": -2},
    {"h": 3, "w": 3, "dc": 2},
    {"h": 3, "w": 3, "dc": -2},
)


def _most_common(arr: np.ndarray) -> int:
    return int(Counter(map(int, arr.flatten())).most_common(1)[0][0])


def _find_rectangles(g: np.ndarray) -> list[tuple[int, tuple[int, int, int, int], int, int]]:
    """Brute-force bordered rectangles; returns (area, rect, border_color, inner_bg)."""
    H, W = g.shape
    rects: list[tuple[int, tuple[int, int, int, int], int, int]] = []
    min_h = 6
    min_w = 6
    for r0 in range(H):
        for c0 in range(W):
            bc = int(g[r0, c0])
            for r1 in range(r0 + min_h - 1, H):
                for c1 in range(c0 + min_w - 1, W):
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
                    inner = g[r0 + 1 : r1, c0 + 1 : c1]
                    ibg = _most_common(inner)
                    if ibg == bc:
                        continue
                    area = int((r1 - r0 + 1) * (c1 - c0 + 1))
                    rects.append((area, (r0, c0, r1, c1), bc, ibg))
    rects.sort(reverse=True, key=lambda x: int(x[0]))
    return rects[:6]


def _infer_color_template(coords: list[tuple[int, int]]) -> dict | None:
    """Infer one repeated diagonal-pair template for a single color."""
    if len(coords) < 2 or len(coords) % 2 != 0:
        return None
    pts = {(int(r), int(c)) for r, c in coords}
    best: dict | None = None
    for dr, dc in PAIR_DELTAS:
        tops: list[tuple[int, int]] = []
        covered: set[tuple[int, int]] = set()
        ok = True
        for r, c in pts:
            prev = (r - dr, c - dc)
            nxt = (r + dr, c + dc)
            has_prev = prev in pts
            has_next = nxt in pts
            if has_prev and has_next:
                ok = False
                break
            if has_next:
                tops.append((r, c))
                covered.add((r, c))
                covered.add(nxt)
            elif has_prev:
                continue
            else:
                ok = False
                break
        if not ok or len(covered) != len(pts) or len(tops) * 2 != len(pts):
            continue
        h = int(dr + 1)
        w = int(abs(dc) + 1)
        anchor_c = 0 if dc > 0 else w - 1
        candidate = {
            "pairs": [(r, c, r + dr, c + dc) for r, c in sorted(tops)],
            "h": h,
            "w": w,
            "anchor": (0, anchor_c),
            "delta": (dr, dc),
        }
        if best is not None:
            return None
        best = candidate
    return best


def _analyze_source_rect(g: np.ndarray, rect: tuple[int, int, int, int]) -> dict | None:
    """Return source templates if this rectangle fits the diagonal-pair source grammar."""
    r0, c0, r1, c1 = rect
    inner = g[r0 + 1 : r1, c0 + 1 : c1]
    bg = _most_common(inner)
    color_coords: dict[int, list[tuple[int, int]]] = {}
    for rr, cc in np.argwhere(inner != bg):
        col = int(inner[int(rr), int(cc)])
        color_coords.setdefault(col, []).append((int(rr), int(cc)))
    if not (1 <= len(color_coords) <= 2):
        return None

    templates: dict[int, dict] = {}
    total_pairs = 0
    for col, coords in color_coords.items():
        info = _infer_color_template(coords)
        if info is None:
            return None
        total_pairs += len(info["pairs"])
        templates[int(col)] = info
    if not (2 <= total_pairs <= 4):
        return None
    return {"bg": int(bg), "templates": templates, "non_bg_count": int(sum(len(v) for v in color_coords.values()))}


def _analyze_dest_rect(g: np.ndarray, rect: tuple[int, int, int, int], template_colors: set[int]) -> dict | None:
    """Destination must contain only singleton markers using source template colors."""
    r0, c0, r1, c1 = rect
    inner = g[r0 + 1 : r1, c0 + 1 : c1]
    bg = _most_common(inner)
    markers = [(int(rr), int(cc), int(inner[int(rr), int(cc)])) for rr, cc in np.argwhere(inner != bg)]
    if not (2 <= len(markers) <= 3):
        return None
    if any(col not in template_colors for _, _, col in markers):
        return None
    return {"bg": int(bg), "markers": markers}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand diagonal source pairs into inferred mini-rectangles at color-matched markers.

    Input structure:
    - Two bordered rectangles sit on a striped wallpaper: a source frame and a destination frame.
    - The source frame does not contain full sprites anymore; each type is shown only as repeated
      same-color diagonal corner pairs inside a 2x3 or 3x3 implied box.
    - The destination frame contains 2-3 singleton markers whose colors refer to those source types.

    Rule:
    - Detect which source colors form consistent diagonal pairs, and infer each type's bbox size and
      whether the pair leans down-right or down-left.
    - Use the upper pixel of the pair as the alignment anchor for matching destination markers.
    - For each marker, stamp the entire inferred rectangle of that color into the cleared destination frame.
    - Output the cropped destination rectangle after all inferred stamps are applied.

    Key insight: The visible source object is only a sparse diagonal clue; the actual transferred shape is
    the hidden rectangle implied by that clue's geometry.
    """
    g = np.array(grid, dtype=int)

    rects = _find_rectangles(g)
    assert rects, "must find bordered rectangles"

    src_candidates: list[tuple[int, tuple[int, int, int, int], int, dict]] = []
    for area, rect, bc, _ibg in rects:
        info = _analyze_source_rect(g, rect)
        if info is not None:
            src_candidates.append((area, rect, bc, info))
    assert src_candidates, "must find a source rectangle with diagonal pairs"

    src_candidates.sort(reverse=True, key=lambda x: int(x[0]))
    _src_area, src_rect, _src_bc, src_info = src_candidates[0]
    template_colors = set(src_info["templates"].keys())

    dst = None
    for area, rect, bc, _ibg in rects:
        if rect == src_rect:
            continue
        info = _analyze_dest_rect(g, rect, template_colors)
        if info is not None:
            dst = (area, rect, bc, info)
            break
    assert dst is not None, "must find destination rectangle with singleton markers"

    _dst_area, (dr, dc, dr1, dc1), dst_bc, dst_info = dst
    dst_bg = int(dst_info["bg"])
    out_h = int(dr1 - dr + 1)
    out_w = int(dc1 - dc + 1)
    out = np.full((out_h, out_w), int(dst_bg), dtype=int)
    out[0, :] = int(dst_bc)
    out[-1, :] = int(dst_bc)
    out[:, 0] = int(dst_bc)
    out[:, -1] = int(dst_bc)

    for rr, cc, col in dst_info["markers"]:
        templ = src_info["templates"][int(col)]
        patch = np.full((int(templ["h"]), int(templ["w"])), int(col), dtype=int)
        ar, ac = map(int, templ["anchor"])
        top = 1 + int(rr) - int(ar)
        left = 1 + int(cc) - int(ac)
        paste(out, patch, int(top), int(left), overwrite=True, clip=True)

    return out


# --------------- generator (D2) ---------------

# === CFG: Canvas & layout ===
GRID_SIZE = 30
GAP_RANGE = (1, 4)
MAX_GEN_ATTEMPTS = 300

# === CFG: Rectangle size bounds ===
SRC_H_RANGE = (9, 19)
SRC_W_RANGE = (8, 18)
DST_H_RANGE = (9, 19)
DST_W_RANGE = (8, 20)
DST_W_SIDE_MAX = 20

# === CFG: Object counts ===
TYPE_CHOICES = (1, 2)
MARKER_COUNT_RANGE = (2, 3)
SOURCE_OBJECT_RANGE = (2, 4)
SOURCE_PADDING = 1
PLACE_RETRIES = 80


def _make_wallpaper(direction: str, palette: list[int]) -> np.ndarray:
    """Create a striped wallpaper with the chosen palette."""
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


def _choose_dimensions(layout: str) -> tuple[int, int, int, int]:
    gap = random.randint(*GAP_RANGE)
    if layout == "side_by_side":
        src_w = random.randint(*SRC_W_RANGE)
        dst_w_max = min(DST_W_SIDE_MAX, GRID_SIZE - gap - src_w)
        if dst_w_max < DST_W_RANGE[0]:
            raise ValueError("no side-by-side width fit")
        dst_w = random.randint(DST_W_RANGE[0], dst_w_max)
        src_h = random.randint(*SRC_H_RANGE)
        dst_h = random.randint(*DST_H_RANGE)
    else:
        src_h = random.randint(*SRC_H_RANGE)
        dst_h_max = min(DST_H_RANGE[1], GRID_SIZE - gap - src_h)
        if dst_h_max < DST_H_RANGE[0]:
            raise ValueError("no stacked height fit")
        dst_h = random.randint(DST_H_RANGE[0], dst_h_max)
        src_w = random.randint(*SRC_W_RANGE)
        dst_w = random.randint(*DST_W_RANGE)
    return src_h, src_w, dst_h, dst_w


def _place_rectangles(
    layout: str, src_h: int, src_w: int, dst_h: int, dst_w: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    if layout == "side_by_side":
        gap = random.randint(1, GRID_SIZE - src_w - dst_w)
        slack = GRID_SIZE - src_w - dst_w - gap
        left_margin = random.randint(0, slack)
        src_r = random.randint(0, GRID_SIZE - src_h)
        dst_r = random.randint(0, GRID_SIZE - dst_h)
        if random.random() < 0.5:
            src_c, dst_c = left_margin, left_margin + src_w + gap
        else:
            dst_c, src_c = left_margin, left_margin + dst_w + gap
    else:
        gap = random.randint(1, GRID_SIZE - src_h - dst_h)
        slack = GRID_SIZE - src_h - dst_h - gap
        top_margin = random.randint(0, slack)
        src_c = random.randint(0, GRID_SIZE - src_w)
        dst_c = random.randint(0, GRID_SIZE - dst_w)
        if random.random() < 0.5:
            src_r, dst_r = top_margin, top_margin + src_h + gap
        else:
            dst_r, src_r = top_margin, top_margin + dst_h + gap
    return (src_r, src_c), (dst_r, dst_c)


def _assign_palette(n_types: int) -> tuple[list[int], int, int, list[int]]:
    wallpaper_n = 5 if n_types == 1 else 4
    all_colors = list(range(10))
    wallpaper = random.sample(all_colors, wallpaper_n)
    remaining = [c for c in all_colors if c not in wallpaper]
    random.shuffle(remaining)
    src_border = remaining.pop()
    dst_border = remaining.pop()
    type_colors = [remaining.pop() for _ in range(n_types)]
    return wallpaper, src_border, dst_border, type_colors


def _bbox_clear(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, padding: int = 0) -> bool:
    H, W = occupied.shape
    rr0 = max(0, int(r0) - int(padding))
    cc0 = max(0, int(c0) - int(padding))
    rr1 = min(H, int(r0) + int(h) + int(padding))
    cc1 = min(W, int(c0) + int(w) + int(padding))
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _mark_bbox(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, padding: int = 0) -> None:
    H, W = occupied.shape
    rr0 = max(0, int(r0) - int(padding))
    cc0 = max(0, int(c0) - int(padding))
    rr1 = min(H, int(r0) + int(h) + int(padding))
    cc1 = min(W, int(c0) + int(w) + int(padding))
    occupied[rr0:rr1, cc0:cc1] = True


def _pair_pixels(spec: dict, r0: int, c0: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if int(spec["dc"]) > 0:
        return (int(r0), int(c0)), (int(r0) + int(spec["h"]) - 1, int(c0) + int(spec["w"]) - 1)
    return (int(r0), int(c0) + int(spec["w"]) - 1), (int(r0) + int(spec["h"]) - 1, int(c0))


def _creates_cross_pair(existing: list[tuple[int, int]], new_pts: list[tuple[int, int]]) -> bool:
    deltas = set(PAIR_DELTAS) | {(-dr, -dc) for dr, dc in PAIR_DELTAS}
    for r1, c1 in existing:
        for r2, c2 in new_pts:
            if (int(r2) - int(r1), int(c2) - int(c1)) in deltas:
                return True
    return False


def _place_sources(type_specs: list[dict], source_type_ids: list[int], src_h: int, src_w: int, src_bg: int) -> np.ndarray | None:
    inner_h = src_h - 2
    inner_w = src_w - 2
    inner = np.full((inner_h, inner_w), src_bg, dtype=int)
    occupied = np.zeros((inner_h, inner_w), dtype=bool)
    same_color_pixels: dict[int, list[tuple[int, int]]] = {}

    for type_id in source_type_ids:
        spec = type_specs[type_id]
        placed = False
        for _ in range(PLACE_RETRIES):
            r0 = random.randint(0, inner_h - int(spec["h"]))
            c0 = random.randint(0, inner_w - int(spec["w"]))
            if not _bbox_clear(occupied, r0, c0, int(spec["h"]), int(spec["w"]), padding=SOURCE_PADDING):
                continue
            p1, p2 = _pair_pixels(spec, r0, c0)
            existing = same_color_pixels.get(int(spec["color"]), [])
            if _creates_cross_pair(existing, [p1, p2]):
                continue
            inner[p1] = int(spec["color"])
            inner[p2] = int(spec["color"])
            same_color_pixels.setdefault(int(spec["color"]), []).extend([p1, p2])
            _mark_bbox(occupied, r0, c0, int(spec["h"]), int(spec["w"]), padding=SOURCE_PADDING)
            placed = True
            break
        if not placed:
            return None
    return inner


def _anchor_col(spec: dict) -> int:
    return 0 if int(spec["dc"]) > 0 else int(spec["w"]) - 1


def _dest_marker_bad(existing_markers: list[tuple[int, int]], mr: int, mc: int) -> bool:
    for er, ec in existing_markers:
        if abs(int(mr) - int(er)) in (1, 2) and abs(int(mc) - int(ec)) == 2:
            return True
    return False


def _place_markers(
    type_specs: list[dict], marker_type_ids: list[int], dst_h: int, dst_w: int, dst_bg: int
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int]]] | None:
    inner_h = dst_h - 2
    inner_w = dst_w - 2
    marker_inner = np.full((inner_h, inner_w), dst_bg, dtype=int)
    out_inner = np.full((inner_h, inner_w), dst_bg, dtype=int)
    occupied = np.zeros((inner_h, inner_w), dtype=bool)
    marker_points: list[tuple[int, int, int]] = []
    same_color_markers: dict[int, list[tuple[int, int]]] = {}

    for type_id in marker_type_ids:
        spec = type_specs[type_id]
        placed = False
        for _ in range(PLACE_RETRIES):
            r0 = random.randint(0, inner_h - int(spec["h"]))
            c0 = random.randint(0, inner_w - int(spec["w"]))
            if not _bbox_clear(occupied, r0, c0, int(spec["h"]), int(spec["w"]), padding=0):
                continue
            mr = int(r0)
            mc = int(c0) + _anchor_col(spec)
            if any(abs(int(mr) - pr) + abs(int(mc) - pc) <= 1 for pr, pc, _ in marker_points):
                continue
            existing_same = same_color_markers.get(int(spec["color"]), [])
            if _dest_marker_bad(existing_same, mr, mc):
                continue
            patch = np.full((int(spec["h"]), int(spec["w"])), int(spec["color"]), dtype=int)
            paste(out_inner, patch, int(r0), int(c0), overwrite=True)
            marker_inner[int(mr), int(mc)] = int(spec["color"])
            marker_points.append((int(mr), int(mc), int(type_id)))
            same_color_markers.setdefault(int(spec["color"]), []).append((int(mr), int(mc)))
            _mark_bbox(occupied, r0, c0, int(spec["h"]), int(spec["w"]), padding=0)
            placed = True
            break
        if not placed:
            return None
    return marker_inner, out_inner, marker_points


def _gen_attempt() -> np.ndarray | None:
    n_types = random.choice(list(TYPE_CHOICES))
    wallpaper, src_border, dst_border, type_colors = _assign_palette(n_types)
    src_bg = random.choice(wallpaper)
    dst_bg_choices = [c for c in wallpaper if c != src_bg]
    if not dst_bg_choices:
        return None
    dst_bg = random.choice(dst_bg_choices)

    geom_choices = random.sample(list(GEOM_SPECS), n_types)
    type_specs = []
    for geom, color in zip(geom_choices, type_colors):
        spec = dict(geom)
        spec["color"] = int(color)
        type_specs.append(spec)

    n_markers = random.randint(*MARKER_COUNT_RANGE)
    marker_type_ids = [i for i, ct in enumerate(_choose_counts(n_markers, n_types)) for _ in range(ct)]
    random.shuffle(marker_type_ids)

    n_source = random.randint(max(2, n_types), SOURCE_OBJECT_RANGE[1])
    source_type_ids = [i for i, ct in enumerate(_choose_counts(n_source, n_types)) for _ in range(ct)]
    random.shuffle(source_type_ids)

    layout = random.choice(["side_by_side", "stacked"])
    try:
        src_h, src_w, dst_h, dst_w = _choose_dimensions(layout)
    except ValueError:
        return None
    if src_h * src_w <= dst_h * dst_w:
        return None

    marker_result = _place_markers(type_specs, marker_type_ids, dst_h, dst_w, dst_bg)
    if marker_result is None:
        return None
    marker_inner, out_inner, _marker_points = marker_result

    source_inner = _place_sources(type_specs, source_type_ids, src_h, src_w, src_bg)
    if source_inner is None:
        return None

    output = _build_frame(dst_h, dst_w, dst_border, dst_bg)
    paste(output[1:-1, 1:-1], out_inner, 0, 0, overwrite=True)

    source_rect = _build_frame(src_h, src_w, src_border, src_bg)
    paste(source_rect[1:-1, 1:-1], source_inner, 0, 0, overwrite=True)

    dest_rect = _build_frame(dst_h, dst_w, dst_border, dst_bg)
    paste(dest_rect[1:-1, 1:-1], marker_inner, 0, 0, overwrite=True)

    try:
        src_pos, dst_pos = _place_rectangles(layout, src_h, src_w, dst_h, dst_w)
    except ValueError:
        return None

    grid = _make_wallpaper(random.choice(["vertical", "horizontal"]), wallpaper)
    paste(grid, source_rect, src_pos[0], src_pos[1], overwrite=True)
    paste(grid, dest_rect, dst_pos[0], dst_pos[1], overwrite=True)

    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a251c730 D2 variant after {MAX_GEN_ATTEMPTS} attempts")
