# Variant of a251c730
# Dimension: D13 — 矩形布局：空间排列 (meta)
# Change: 两个矩形在30x30网格中不重叠排列，可以并排或上下堆叠，间距1-4像素... → 矩形部分重叠

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location

"""
Design analysis for the D13 overlap variant
===========================================

variant_spec:
  source_task: none
  core_change:
    - The two rectangles no longer sit apart with a wallpaper gap.
    - They must share a border segment made of the same border color cells.
    - The overlap is only on border cells; the two interiors stay disjoint.
  concretization:
    - The layout is still side-by-side or stacked, but one border column/row is common
      to both rectangles.
    - The shared segment is partial rather than a full side match, so one frame visibly
      sticks into the middle portion of the other frame's side.
    - Because one pixel cannot carry two border colors, both rectangles use the same
      border color in this variant. This is the only non-D13 adaptation needed to make
      literal border sharing possible on the ARC cell grid.
  behavior_trace:
    - Original: find two separated uniform-border rectangles; classify the source by
      "has a multi-cell component" and the destination by "has singleton components".
    - Variant: touching frames can create larger false rectangles or mixed candidates
      around the shared wall, so "max component size > 1" is no longer enough.
    - New logic: enumerate uniform-border rectangles, compute their interior components,
      keep only source-like candidates whose components are all multi-cell and
      destination-like candidates whose components are all singleton markers, then pick
      the pair whose borders share a nontrivial segment while their interiors remain
      separate.
  complexity_calibration:
    - Shared-border length is constrained to at least 4 cells so the new arrangement is
      obvious by eye.
    - The overlap is partial, not a full-side merge, so the two frames remain readable
      as two objects rather than a single large box with an internal divider.
    - Sprite counts, marker counts, sprite shapes, wallpaper style, and output rule stay
      close to the original task to keep D13 as the main novelty.
  implicit_assumptions_review:
    - Original solver assumption: the first large rectangle with any multi-cell
      component is the source. This breaks when a false enclosing rectangle can contain
      both sprites and singleton markers.
    - Original generator assumption: border colors may differ because rectangles never
      share cells. This no longer holds; shared border cells force a common border color.
    - Original placement assumption: a positive gap separates the frames. The new
      generator instead guarantees zero interior overlap but positive border overlap.
  placement_legality:
    - Rectangles are placed so their union fits inside 30x30.
    - They share exactly one border column or row, with overlap strictly smaller than
      both side lengths, ensuring partial overlap.
    - Source sprites still avoid each other and the source edge; destination markers are
      placed through output-first stamping so every marker can receive its full sprite.
  visual_observability:
    - Each sample visibly contains a single-color shared wall segment between the two
      framed regions. This directly contrasts with the original gap-separated examples.
    - The source interior still shows repeated multi-pixel motifs and the destination
      still shows isolated colored anchor dots, so a human can recover which frame plays
      which role even though the borders touch.
  visual_cleanliness:
    - Only border cells overlap. No sprite or marker is hidden under another object.
    - The shared wall is a single-cell-thick straight segment, so the layout stays tidy.
  color_inference:
    - No rule depends on ordering colors by numeric id. Anchor detection uses the unique
      least-frequent color inside each sprite component, which the generator guarantees.
  new_main_assumptions_and_generator_guards:
    - main() assumes there is exactly one source-like rectangle and one destination-like
      rectangle connected by a shared border segment of length >= 4.
    - generate_input() enforces those conditions and verifies them by calling main().
"""


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


def _border_cells(rect: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, rect)
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _interiors_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    ir0 = max(ar0 + 1, br0 + 1)
    ic0 = max(ac0 + 1, bc0 + 1)
    ir1 = min(ar1 - 1, br1 - 1)
    ic1 = min(ac1 - 1, bc1 - 1)
    return ir0 <= ir1 and ic0 <= ic1


def _shared_border_len(a: dict, b: dict) -> int:
    return int(len(a["border_cells"] & b["border_cells"]))


def _find_rect_candidates(g: np.ndarray) -> list[dict]:
    H, W = g.shape
    min_h = 6
    min_w = 6
    candidates: list[dict] = []

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

                    rect = (int(r0), int(c0), int(r1), int(c1))
                    ibg = _interior_bg(g, rect)
                    if ibg == bc:
                        continue
                    ibg2, comps = _rect_components(g, rect)
                    sizes = [int(comp["size"]) for comp in comps]
                    all_multi = bool(sizes) and all(sz > 1 for sz in sizes) and 2 <= len(sizes) <= 4
                    all_single = bool(sizes) and all(sz == 1 for sz in sizes) and 2 <= len(sizes) <= 3
                    candidates.append(
                        {
                            "rect": rect,
                            "area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
                            "bc": int(bc),
                            "bg": int(ibg2),
                            "comps": comps,
                            "is_source_like": all_multi,
                            "is_dest_like": all_single,
                            "border_cells": _border_cells(rect),
                        }
                    )
    return candidates


def _select_pair(candidates: list[dict]) -> tuple[dict, dict]:
    source_candidates = [cand for cand in candidates if cand["is_source_like"]]
    dest_candidates = [cand for cand in candidates if cand["is_dest_like"]]
    pairs: list[tuple[tuple[int, int, int, int], dict, dict]] = []

    for src in source_candidates:
        for dst in dest_candidates:
            if src["rect"] == dst["rect"]:
                continue
            if src["area"] <= dst["area"]:
                continue
            if _interiors_overlap(src["rect"], dst["rect"]):
                continue
            shared = _shared_border_len(src, dst)
            if shared < 4:
                continue
            score = (
                int(shared),
                int(src["area"] + dst["area"]),
                int(src["area"]),
                -int(dst["area"]),
            )
            pairs.append((score, src, dst))

    if not pairs:
        raise AssertionError("must find a source/destination rectangle pair with a shared border segment")

    pairs.sort(reverse=True, key=lambda item: item[0])
    return pairs[0][1], pairs[0][2]


def _extract_anchor(comp: dict) -> tuple[int, tuple[int, int]]:
    counts = {int(k): int(v) for k, v in comp["colors"].items() if int(k) != 0}
    if not counts:
        raise AssertionError("sprite component has no colors")
    min_count = min(counts.values())
    anchor_colors = [color for color, count in counts.items() if count == min_count]
    if len(anchor_colors) != 1:
        raise AssertionError("sprite anchor color must be uniquely least frequent")
    anchor_color = int(anchor_colors[0])
    patch = comp["patch"]
    mask = comp["mask"]
    anchors = np.argwhere((patch == anchor_color) & mask)
    if anchors.size == 0:
        raise AssertionError("anchor pixel missing from sprite patch")
    ar, ac = map(int, anchors[0])
    return anchor_color, (ar, ac)


def _build_templates(src_comps: list[dict]) -> dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]]:
    templates: dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]] = {}
    for comp in src_comps:
        anchor_color, anchor_rc = _extract_anchor(comp)
        if anchor_color not in templates or int(comp["mask"].sum()) > int(templates[anchor_color][1].sum()):
            templates[anchor_color] = (comp["patch"].copy(), comp["mask"].copy(), anchor_rc)
    return templates


def _pair_metrics_from_grid(grid: np.ndarray) -> dict:
    candidates = _find_rect_candidates(np.array(grid, dtype=int))
    src, dst = _select_pair(candidates)
    shared_cells = src["border_cells"] & dst["border_cells"]
    rows = {r for r, _ in shared_cells}
    cols = {c for _, c in shared_cells}
    orientation = "vertical" if len(cols) == 1 else "horizontal"
    return {
        "shared_border_len": int(len(shared_cells)),
        "orientation": orientation,
        "src_rect": src["rect"],
        "dst_rect": dst["rect"],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the destination frame when the two rectangles touch through a shared border wall.

    Input structure:
    - Two framed rectangles lie on striped wallpaper and now share part of one border segment.
    - One rectangle contains several separated multi-pixel sprites; the other contains 2-3 single-pixel anchor markers.

    Rule:
    - Recover which rectangle is the sprite source and which is the marker destination despite the shared border cells.
    - Match each destination marker to the source sprite with the same anchor color, align the sprite's anchor onto the marker, and stamp into a cleared copy of the destination frame.

    Key insight:
    - The decisive clue is the combination of interior component type and a nontrivial shared border segment, not the old gap between rectangles.
    """
    g = np.array(grid, dtype=int)
    candidates = _find_rect_candidates(g)
    src, dst = _select_pair(candidates)

    templates = _build_templates(src["comps"])

    dr0, dc0, dr1, dc1 = map(int, dst["rect"])
    out_h = int(dr1 - dr0 + 1)
    out_w = int(dc1 - dc0 + 1)
    out = np.full((out_h, out_w), int(dst["bg"]), dtype=int)
    out[0, :] = int(dst["bc"])
    out[-1, :] = int(dst["bc"])
    out[:, 0] = int(dst["bc"])
    out[:, -1] = int(dst["bc"])

    for comp in dst["comps"]:
        rr, cc = map(int, comp["point"])
        marker_colors = list(comp["colors"].keys())
        if len(marker_colors) != 1:
            raise AssertionError("destination marker component must contain exactly one color")
        marker_color = int(marker_colors[0])
        if marker_color not in templates:
            continue
        sprite_patch, sprite_mask, (ar, ac) = templates[marker_color]
        top = int(rr + 1 - ar)
        left = int(cc + 1 - ac)
        paste(out, sprite_patch, top, left, mask=sprite_mask, overwrite=True, clip=True)

    return out


# --------------- generator (O2) ---------------

# === CFG: Canvas & layout ===
GRID_SIZE = 30
WALLPAPER_COLORS = (4, 5)
TOTAL_COLORS_TARGET = (7, 8, 9)
SHARED_SEGMENT_MIN = 4

# === CFG: Rectangle size bounds ===
SRC_H_RANGE = (9, 19)
SRC_W_RANGE = (8, 18)
DST_H_RANGE = (9, 19)
DST_W_RANGE = (8, 20)

# === CFG: Sprite & placement ===
SPRITE_TYPE_CHOICES = (1, 2)
MARKER_COUNT_RANGE = (2, 3)
SOURCE_SPRITE_RANGE = (2, 4)
MARKER_PLACE_RETRIES = 25
SOURCE_SPRITE_PADDING = 1

# === CFG: Generation retries ===
MAX_GEN_ATTEMPTS = 400

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


def _assign_palette(n_types: int) -> tuple[list[int], int, list[int], list[int]] | None:
    wallpaper_n = 5 if n_types == 1 else random.choice(list(WALLPAPER_COLORS))
    used_total_target = random.choice(list(TOTAL_COLORS_TARGET))
    base_colors = wallpaper_n + 1 + n_types
    extras_needed = used_total_target - base_colors
    if extras_needed < 0 or extras_needed > n_types:
        return None

    all_colors = list(range(10))
    wallpaper = random.sample(all_colors, wallpaper_n)
    remaining = [c for c in all_colors if c not in wallpaper]
    random.shuffle(remaining)

    border = remaining.pop()
    anchors = [remaining.pop() for _ in range(n_types)]
    extra_bodies = [remaining.pop() for _ in range(extras_needed)]
    return wallpaper, border, anchors, extra_bodies


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


def _interval_pairs(len_a: int, len_b: int) -> list[tuple[int, int, int]]:
    pairs: list[tuple[int, int, int]] = []
    for a0 in range(0, GRID_SIZE - len_a + 1):
        a1 = a0 + len_a - 1
        for b0 in range(0, GRID_SIZE - len_b + 1):
            b1 = b0 + len_b - 1
            overlap = min(a1, b1) - max(a0, b0) + 1
            if overlap < SHARED_SEGMENT_MIN:
                continue
            if overlap >= min(len_a, len_b):
                continue
            pairs.append((int(a0), int(b0), int(overlap)))
    return pairs


def _place_rectangles(
    src_h: int, src_w: int, dst_h: int, dst_w: int
) -> tuple[tuple[int, int], tuple[int, int], str, int]:
    options: list[tuple[str, list[tuple[int, int, int]]]] = []
    if src_w + dst_w - 1 <= GRID_SIZE:
        row_pairs = _interval_pairs(src_h, dst_h)
        if row_pairs:
            options.append(("side_by_side", row_pairs))
    if src_h + dst_h - 1 <= GRID_SIZE:
        col_pairs = _interval_pairs(src_w, dst_w)
        if col_pairs:
            options.append(("stacked", col_pairs))
    if not options:
        raise ValueError("no overlapping placement fits")

    layout, pairs = random.choice(options)
    a0, b0, shared = random.choice(pairs)

    if layout == "side_by_side":
        left_margin = random.randint(0, GRID_SIZE - (src_w + dst_w - 1))
        if random.random() < 0.5:
            src_pos = (a0, left_margin)
            dst_pos = (b0, left_margin + src_w - 1)
        else:
            dst_pos = (b0, left_margin)
            src_pos = (a0, left_margin + dst_w - 1)
    else:
        top_margin = random.randint(0, GRID_SIZE - (src_h + dst_h - 1))
        if random.random() < 0.5:
            src_pos = (top_margin, a0)
            dst_pos = (top_margin + src_h - 1, b0)
        else:
            dst_pos = (top_margin, b0)
            src_pos = (top_margin + dst_h - 1, a0)

    return src_pos, dst_pos, layout, int(shared)


def _gen_attempt() -> np.ndarray | None:
    n_types = random.choice(list(SPRITE_TYPE_CHOICES))

    palette_result = _assign_palette(n_types)
    if palette_result is None:
        return None
    wallpaper, border, anchors, extra_bodies = palette_result

    src_bg = random.choice(wallpaper)
    dst_bg = random.choice(wallpaper)

    sprite_specs = random.sample(SPRITE_SHAPES, n_types)
    forced_extra = extra_bodies[:]
    random.shuffle(forced_extra)
    extra_targets = set(random.sample(range(n_types), len(forced_extra))) if forced_extra else set()
    bodies: list[int] = []
    extra_index = 0
    for type_id in range(n_types):
        if type_id in extra_targets:
            body = int(forced_extra[extra_index])
            extra_index += 1
        else:
            choices = [c for c in wallpaper if c != src_bg and c != dst_bg]
            if not choices:
                return None
            body = int(random.choice(choices))
        bodies.append(body)

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

    src_h = random.randint(*SRC_H_RANGE)
    src_w = random.randint(*SRC_W_RANGE)
    dst_h = random.randint(*DST_H_RANGE)
    dst_w = random.randint(*DST_W_RANGE)
    if src_h * src_w <= dst_h * dst_w:
        return None

    marker_result = _place_markers(sprite_types, marker_type_ids, dst_h, dst_w, dst_bg)
    if marker_result is None:
        return None
    out_inner, marker_points = marker_result

    output = _build_frame(dst_h, dst_w, border, dst_bg)
    paste(output[1:-1, 1:-1], out_inner, 0, 0)

    source_inner = _place_sources(sprite_types, source_type_ids, src_h, src_w, src_bg)
    if source_inner is None:
        return None

    source_rect = _build_frame(src_h, src_w, border, src_bg)
    paste(source_rect[1:-1, 1:-1], source_inner, 0, 0)

    dest_rect = _build_frame(dst_h, dst_w, border, dst_bg)
    for mr, mc, type_id in marker_points:
        dest_rect[1 + mr, 1 + mc] = int(sprite_types[type_id]["anchor"])

    try:
        src_pos, dst_pos, _layout, shared = _place_rectangles(src_h, src_w, dst_h, dst_w)
    except ValueError:
        return None
    if shared < SHARED_SEGMENT_MIN:
        return None

    grid = _make_wallpaper(random.choice(["vertical", "horizontal"]), wallpaper)
    paste(grid, source_rect, src_pos[0], src_pos[1])
    paste(grid, dest_rect, dst_pos[0], dst_pos[1])

    try:
        metrics = _pair_metrics_from_grid(grid)
    except AssertionError:
        return None
    if metrics["shared_border_len"] < SHARED_SEGMENT_MIN:
        return None

    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a251c730 D13 input after {MAX_GEN_ATTEMPTS} attempts")
