# Variant of a251c730
# Dimension: D13 — 矩形布局：空间排列 (meta)
# Change: 两个矩形在30x30网格中不重叠排列，可以并排或上下堆叠，间距1-4像素... → 矩形嵌套

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location

# Design analysis for the D13 variant
# ----------------------------------
# spec_like_design:
#   input:
#     grid: 30x30 striped wallpaper
#     topology:
#       - one large bordered source rectangle is pasted on the wallpaper
#       - one smaller bordered destination rectangle is pasted fully inside the source interior
#       - source sprites occupy only the annulus between the two rectangles
#       - destination markers occupy only the innermost rectangle interior
#   output:
#     - crop the inner destination rectangle
#     - clear its interior to the destination background
#     - stamp matched source sprites so each sprite anchor lands on the same-colored marker
#
# a) concrete behavior trace:
#   - Original task: solver searches two separate rectangles anywhere on the canvas, classifies the
#     multi-cell one as source and the singleton one as destination, then stamps source templates
#     into the destination.
#   - Variant task: solver must first recover a nested pair. Example: an outer 24x22 frame contains
#     a 10x12 inner frame offset inside it. The ring holds several copies of two sprite types; the
#     inner frame contains only 2-3 singleton anchor-colored markers. The output is still the inner
#     frame after stamping, but source extraction now comes from the annulus instead of a separate
#     rectangle. The changed step is therefore rectangle interpretation: the old whole-rectangle
#     component scan would wrongly absorb the inner frame into the source object set.
#
# b) complexity calibration:
#   - Keep 1-2 sprite types and 2-3 markers, as in the original, so matching evidence repeats.
#   - Make annulus thickness 5-6 cells. This is visually obvious and guarantees 3x3 sprites fit.
#   - Inner frame remains large enough for multiple marker placements, so the task still has enough
#     combinatorial variation without becoming visually noisy.
#
# c) implicit assumption audit from the original main():
#   - "Find a source rectangle whose whole interior has multi-cell components" no longer holds,
#     because the outer rectangle interior also contains the nested destination frame.
#   - "Find a destination rectangle whose whole interior contains only singletons" no longer holds
#     for the outer rectangle, only for the inner one.
#   - "Most-common color over the whole source interior is the source background" becomes risky if
#     the inner frame is large. The new solver therefore computes the source background only on the
#     annulus mask, excluding the entire nested rectangle footprint.
#   - Template extraction still assumes each sprite has one unique rarest color. The generator keeps
#     that invariant, so the solver avoids color-id tie breaking and requires a unique rarest color.
#
# d) placement legality:
#   - Generator reserves the full inner-rectangle footprint before placing annulus sprites, so
#     sprites cannot overlap the nested frame.
#   - Annulus sprites are placed with a one-cell border and one-cell padding from other occupied
#     cells, matching the original spacing intent.
#   - Marker placement is still output-first: markers are chosen by placing full sprite patches in
#     the destination interior first, then reducing them to singleton anchors for the input.
#
# e) visual observability:
#   - The nested topology is directly visible: one frame inside another, with sprites around it.
#   - The task rule remains recoverable from 2-4 examples because anchor colors still provide the
#     same cross-zone correspondence, now across concentric regions instead of separated panels.
#
# f) visual cleanliness:
#   - Only three layers appear inside the source object: outer border, annulus sprites on a uniform
#     background, and the nested destination frame. No extra decorations are added.
#   - Thicker annulus margins prevent sprites from crowding the inner frame border.
#
# g) color inference:
#   - No hidden ordering by color id is used in the rule. The generator makes each sprite anchor a
#     unique singleton color, so the solver can identify anchors by frequency alone.


def _mode_color(values: np.ndarray) -> int:
    flat = np.asarray(values, dtype=int).reshape(-1)
    return int(Counter(map(int, flat)).most_common(1)[0][0])


def _find_uniform_rectangles(
    g: np.ndarray, *, min_h: int = 6, min_w: int = 6
) -> list[dict]:
    H, W = g.shape
    rects: list[dict] = []
    for r0 in range(int(H)):
        for c0 in range(int(W)):
            bc = int(g[r0, c0])
            for r1 in range(int(r0 + min_h - 1), int(H)):
                if int(g[r1, c0]) != bc:
                    continue
                for c1 in range(int(c0 + min_w - 1), int(W)):
                    if int(g[r0, c1]) != bc or int(g[r1, c1]) != bc:
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
                    if inner.size == 0:
                        continue
                    ibg = _mode_color(inner)
                    if ibg == bc:
                        continue
                    rects.append(
                        {
                            "rect": (int(r0), int(c0), int(r1), int(c1)),
                            "border": int(bc),
                            "area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
                        }
                    )
    rects.sort(key=lambda d: int(d["area"]), reverse=True)
    return rects


def _component_records(g: np.ndarray, fg_mask: np.ndarray) -> list[dict]:
    comps: list[dict] = []
    for coords in components_mask(fg_mask, connectivity=4):
        if coords.size == 0:
            continue
        rr0, cc0, rr1, cc1 = bbox_coords(coords)
        mask = np.zeros((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), dtype=bool)
        mask[coords[:, 0] - int(rr0), coords[:, 1] - int(cc0)] = True
        patch = g[int(rr0) : int(rr1) + 1, int(cc0) : int(cc1) + 1].copy()
        patch[~mask] = 0
        vals = [int(g[int(r), int(c)]) for r, c in coords]
        comps.append(
            {
                "size": int(coords.shape[0]),
                "colors": dict(Counter(vals)),
                "bbox": (int(rr0), int(cc0), int(rr1), int(cc1)),
                "mask": mask,
                "patch": patch,
            }
        )
    comps.sort(key=lambda d: int(d["size"]), reverse=True)
    return comps


def _anchor_color_from_counts(colors: dict[int, int]) -> int | None:
    usable = {int(k): int(v) for k, v in colors.items() if int(k) != 0}
    if len(usable) < 2:
        return None
    rarest = min(int(v) for v in usable.values())
    anchors = [int(col) for col, count in usable.items() if int(count) == rarest]
    if len(anchors) != 1:
        return None
    return int(anchors[0])


def _find_nested_pair(g: np.ndarray) -> dict:
    rects = _find_uniform_rectangles(g)
    candidates: list[tuple[tuple[int, int, int, int], dict]] = []

    for outer in rects[:12]:
        or0, oc0, or1, oc1 = map(int, outer["rect"])
        for inner in rects:
            if int(inner["area"]) >= int(outer["area"]):
                continue
            ir0, ic0, ir1, ic1 = map(int, inner["rect"])
            if not (or0 + 1 < ir0 and oc0 + 1 < ic0 and ir1 < or1 - 1 and ic1 < oc1 - 1):
                continue

            annulus_mask = np.zeros_like(g, dtype=bool)
            annulus_mask[or0 + 1 : or1, oc0 + 1 : oc1] = True
            annulus_mask[ir0 : ir1 + 1, ic0 : ic1 + 1] = False
            if not bool(annulus_mask.any()):
                continue

            src_bg = _mode_color(g[annulus_mask])
            src_fg = annulus_mask & (g != int(src_bg))
            src_comps = [comp for comp in _component_records(g, src_fg) if int(comp["size"]) > 1]
            if not src_comps:
                continue

            inner_cells = g[ir0 + 1 : ir1, ic0 + 1 : ic1]
            if inner_cells.size == 0:
                continue
            dst_bg = _mode_color(inner_cells)
            dst_fg = inner_cells != int(dst_bg)
            dst_coords = components_mask(dst_fg, connectivity=4)
            if not dst_coords:
                continue
            if max(int(coords.shape[0]) for coords in dst_coords) != 1:
                continue

            marker_colors = {int(inner_cells[int(r), int(c)]) for coords in dst_coords for r, c in coords}
            template_anchors = {
                int(anchor_col)
                for comp in src_comps
                for anchor_col in [_anchor_color_from_counts(comp["colors"])]
                if anchor_col is not None
            }
            if not template_anchors or marker_colors != template_anchors:
                continue

            score = (
                len(marker_colors),
                len(src_comps),
                int(outer["area"]) - int(inner["area"]),
                int(outer["area"]),
            )
            candidates.append(
                (
                    score,
                    {
                        "outer_rect": outer["rect"],
                        "outer_border": int(outer["border"]),
                        "outer_bg": int(src_bg),
                        "inner_rect": inner["rect"],
                        "inner_border": int(inner["border"]),
                        "inner_bg": int(dst_bg),
                        "src_comps": src_comps,
                    },
                )
            )

    assert candidates, "must find a nested source rectangle containing a nested destination rectangle"
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project annulus sprites into a nested inner frame via anchor-colored markers.

    Input structure:
    - One large bordered rectangle encloses a smaller bordered rectangle inside it.
    - Multi-cell sprites sit in the ring between the two rectangles, while singleton markers sit inside the inner frame.

    Rule:
    - Use each sprite's unique least-frequent color as its anchor/key.
    - Match inner-frame markers by that color and stamp the corresponding sprite with anchor aligned to marker.
    - Return only the inner rectangle, rebuilt on its own background before stamping.

    Key insight: The same anchor-color matching rule survives, but the source and destination zones are now separated by nesting rather than by external placement.
    """
    g = np.array(grid, dtype=int)
    pair = _find_nested_pair(g)

    ir0, ic0, ir1, ic1 = map(int, pair["inner_rect"])
    inner_border = int(pair["inner_border"])
    inner_bg = int(pair["inner_bg"])
    src_comps = pair["src_comps"]

    templates: dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]] = {}
    for comp in src_comps:
        anchor_col = _anchor_color_from_counts(comp["colors"])
        if anchor_col is None:
            continue
        patch = np.array(comp["patch"], dtype=int)
        mask = np.array(comp["mask"], dtype=bool)
        anchors = np.argwhere((patch == int(anchor_col)) & mask)
        if anchors.size == 0:
            continue
        ar, ac = map(int, anchors[0])
        if int(anchor_col) not in templates or int(mask.sum()) > int(templates[int(anchor_col)][1].sum()):
            templates[int(anchor_col)] = (patch.copy(), mask.copy(), (int(ar), int(ac)))

    out_h = int(ir1 - ir0 + 1)
    out_w = int(ic1 - ic0 + 1)
    out = np.full((out_h, out_w), int(inner_bg), dtype=int)
    out[0, :] = int(inner_border)
    out[-1, :] = int(inner_border)
    out[:, 0] = int(inner_border)
    out[:, -1] = int(inner_border)

    inner = g[ir0 + 1 : ir1, ic0 + 1 : ic1]
    for rr, cc in np.argwhere(inner != int(inner_bg)):
        marker_color = int(inner[int(rr), int(cc)])
        if marker_color not in templates:
            continue
        patch, mask, (ar, ac) = templates[int(marker_color)]
        top = int(1 + int(rr) - int(ar))
        left = int(1 + int(cc) - int(ac))
        paste(out, patch, int(top), int(left), mask=mask, overwrite=True, clip=True)

    return out


# --------------- generator (nested D13) ---------------

GRID_SIZE = 30
WALLPAPER_COLORS = (4, 5)
TOTAL_COLORS_TARGET = (7, 8, 9)

SRC_RING_RANGE = (5, 6)
DST_H_RANGE = (9, 13)
DST_W_RANGE = (9, 14)

SPRITE_TYPE_CHOICES = (1, 2)
MARKER_COUNT_RANGE = (2, 3)
SOURCE_SPRITE_RANGE = (2, 4)
MARKER_PLACE_RETRIES = 25
SOURCE_SPRITE_PADDING = 1
MAX_GEN_ATTEMPTS = 300

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
    for _ in range(int(n_total - n_types)):
        counts[random.randrange(n_types)] += 1
    return counts


def _choose_nested_dimensions() -> tuple[int, int, int, int, int, int]:
    for _ in range(100):
        dst_h = random.randint(*DST_H_RANGE)
        dst_w = random.randint(*DST_W_RANGE)
        top = random.randint(*SRC_RING_RANGE)
        bottom = random.randint(*SRC_RING_RANGE)
        left = random.randint(*SRC_RING_RANGE)
        right = random.randint(*SRC_RING_RANGE)
        src_h = int(dst_h + top + bottom + 2)
        src_w = int(dst_w + left + right + 2)
        if src_h <= GRID_SIZE and src_w <= GRID_SIZE:
            return src_h, src_w, dst_h, dst_w, top, left
    raise ValueError("no nested dimension fit")


def _assign_palette(n_types: int) -> tuple[list[int], int, int, list[int], list[int], int | None] | None:
    wallpaper_n = 5 if n_types == 1 else random.choice(list(WALLPAPER_COLORS))
    used_total_target = random.choice(list(TOTAL_COLORS_TARGET))
    extras_needed = used_total_target - (wallpaper_n + 1 + n_types)
    if extras_needed < 0 or extras_needed > 2:
        return None

    all_colors = list(range(10))
    wallpaper = random.sample(all_colors, wallpaper_n)
    remaining = [c for c in all_colors if c not in wallpaper]
    random.shuffle(remaining)

    src_border = remaining.pop()
    anchors = [remaining.pop() for _ in range(n_types)]

    dst_border_extra = extras_needed >= 2 or (extras_needed == 1 and random.random() < 0.5)
    body_extra = extras_needed >= 2 or (extras_needed >= 1 and not dst_border_extra)

    if dst_border_extra:
        if not remaining:
            return None
        dst_border = remaining.pop()
    else:
        dst_border = -1

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
            marker_points.append((int(mr), int(mc), int(type_id)))
            placed = True
            break
        if not placed:
            return None
    return inner, marker_points


def _place_sources(
    sprite_types: list[dict],
    source_type_ids: list[int],
    src_h: int,
    src_w: int,
    src_bg: int,
    nested_r: int,
    nested_c: int,
    dst_h: int,
    dst_w: int,
) -> np.ndarray | None:
    inner = np.full((src_h - 2, src_w - 2), src_bg, dtype=int)
    blocker = np.full((dst_h, dst_w), -1, dtype=int)
    paste(inner, blocker, int(nested_r), int(nested_c))
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


def _gen_attempt() -> np.ndarray | None:
    n_types = random.choice(list(SPRITE_TYPE_CHOICES))

    palette_result = _assign_palette(n_types)
    if palette_result is None:
        return None
    wallpaper, src_border, dst_border, anchors, body_pool, forced_body = palette_result

    src_bg = random.choice(wallpaper)
    if dst_border == -1:
        dst_border_choices = [c for c in wallpaper if c != src_bg]
        if not dst_border_choices:
            return None
        dst_border = random.choice(dst_border_choices)

    dst_bg_choices = [c for c in wallpaper if c != dst_border]
    if not dst_bg_choices:
        return None
    dst_bg = random.choice(dst_bg_choices)

    sprite_specs = random.sample(SPRITE_SHAPES, n_types)
    bodies: list[int] = []
    for i in range(n_types):
        choices = [
            c
            for c in body_pool
            if c not in anchors and c != src_bg and c != dst_bg and c != src_border and c != dst_border
        ]
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

    try:
        src_h, src_w, dst_h, dst_w, nested_r, nested_c = _choose_nested_dimensions()
    except ValueError:
        return None

    marker_result = _place_markers(sprite_types, marker_type_ids, dst_h, dst_w, dst_bg)
    if marker_result is None:
        return None
    out_inner, marker_points = marker_result

    output = _build_frame(dst_h, dst_w, dst_border, dst_bg)
    paste(output[1:-1, 1:-1], out_inner, 0, 0)

    source_inner = _place_sources(
        sprite_types,
        source_type_ids,
        src_h,
        src_w,
        src_bg,
        nested_r,
        nested_c,
        dst_h,
        dst_w,
    )
    if source_inner is None:
        return None

    source_rect = _build_frame(src_h, src_w, src_border, src_bg)
    paste(source_rect[1:-1, 1:-1], source_inner, 0, 0)

    dest_rect = _build_frame(dst_h, dst_w, dst_border, dst_bg)
    for mr, mc, type_id in marker_points:
        dest_rect[1 + int(mr), 1 + int(mc)] = int(sprite_types[int(type_id)]["anchor"])

    paste(source_rect[1:-1, 1:-1], dest_rect, int(nested_r), int(nested_c))

    grid = _make_wallpaper(random.choice(["vertical", "horizontal"]), wallpaper)
    src_r = random.randint(0, GRID_SIZE - src_h)
    src_c = random.randint(0, GRID_SIZE - src_w)
    paste(grid, source_rect, int(src_r), int(src_c))

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate nested a251c730 input after {MAX_GEN_ATTEMPTS} attempts")
