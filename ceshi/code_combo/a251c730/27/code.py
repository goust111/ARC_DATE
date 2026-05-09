# Combo variant of a251c730 (index 27)
# combo_spec: D9/1+D13/1
# Name: 三矩形+嵌套布局

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location

# Design analysis for the combo variant
# -------------------------------------
# spec_like_design:
#   input:
#     grid: 30x30 striped wallpaper
#     topology:
#       - one outer bordered source rectangle is pasted on the wallpaper
#       - one middle bordered legend rectangle is nested inside the source interior
#       - one inner bordered destination rectangle is nested inside the legend interior
#       - source sprites occupy only the outer annulus between source and legend
#       - legend entries occupy only the middle annulus between legend and destination
#       - destination markers occupy only the innermost rectangle interior
#   legend_encoding:
#       - each legend entry is a 1x2 horizontal domino
#       - left pixel = destination marker color
#       - right pixel = source anchor color
#   output:
#       - crop the innermost destination rectangle
#       - clear its interior to the destination background
#       - for each destination marker, decode marker_color -> anchor_color from the legend
#       - stamp the matching outer-annulus sprite with its anchor aligned to the marker
#
# a) concrete behavior trace:
#   - Original task: detect two separate rectangles, read anchor-colored singleton markers directly
#     from the destination, and stamp the matching source sprite.
#   - D13 changed the topology: the destination moved inside the source, so the solver had to carve
#     the source annulus out of a nested pair instead of scanning two disjoint panels.
#   - This combo adds D9 on top of that: markers inside the innermost rectangle no longer use source
#     anchor colors directly. Example: the outer annulus may contain a sprite whose unique rare color
#     is 6, while the inner target uses marker color 4. The middle annulus then contains a horizontal
#     legend chip [4, 6]. The solver must first decode that chip, then use marker 4 to request the
#     sprite anchored by 6.
#   - The key difference from both ancestors is that rectangle segmentation and color decoding are now
#     coupled: if the middle rectangle is mistaken for the destination, or if the legend is ignored,
#     stamping uses the wrong key and fails.
#
# b) complexity calibration:
#   - Keep 1-2 sprite types and 2-3 markers so every mapping category repeats, which gives humans
#     evidence that the legend is a reusable correspondence table rather than decoration.
#   - Keep source sprites in 2x3 or 3x3 boxes and use a clean 1x2 legend symbol. This makes the
#     three annuli visually distinct: complex shapes outside, tiny code chips in the middle, and
#     singleton markers inside.
#   - Use only a few mapping entries per sample so the color-translation rule stays inferable from
#     2-4 examples instead of becoming an arbitrary permutation puzzle.
#
# c) implicit assumption audit from the original main():
#   - "The destination rectangle is the only rectangle whose interior contains singletons" no longer
#     holds globally, because the middle legend annulus also contains tiny components. The new solver
#     therefore validates a full triple: outer annulus must contain multi-cell sprites, middle annulus
#     must contain only horizontal size-2 legend entries, and inner destination must contain only
#     singleton markers.
#   - "Marker color equals source anchor color" is intentionally broken. The solver must decode the
#     legend map before template lookup.
#   - "Most-common color over the whole outer interior is the source background" is unsafe because the
#     outer interior also contains the entire legend rectangle. The solver computes backgrounds on each
#     annulus separately after subtracting the nested rectangle footprints.
#   - "Rarest color tie-breaking by numeric id is acceptable" is avoided. The generator guarantees each
#     sprite type has exactly one singleton anchor color, and the solver requires uniqueness.
#
# d) placement legality:
#   - The generator reserves the full middle rectangle footprint before placing source sprites, so no
#     sprite can bleed into the legend or destination zones.
#   - The generator reserves the full inner destination footprint before placing legend chips, and keeps
#     all chips in the top band of the legend annulus so they are disjoint, horizontal, and obvious.
#   - Marker placement remains output-first: full sprites are placed in the destination canvas first to
#     guarantee fit, then reduced to singleton marker colors in the input.
#
# e) visual observability:
#   - A human sees three nested frames. The outer ring contains repeated multi-pixel motifs, the middle
#     ring contains tiny two-cell "dictionary" pairs, and the center contains singleton prompts. Across
#     examples, inner marker colors reappear as left legend colors, while source anchors reappear as
#     right legend colors.
#   - Because the middle code ring is the only place where the marker colors and anchor colors touch,
#     the color remapping rule is observable without any hidden ordering by numeric color id.
#
# f) visual cleanliness:
#   - Each semantic layer has its own annulus, so there is no overdraw between sprites, legend entries,
#     and target markers.
#   - The legend chips are all horizontal and placed in a single band, which prevents the middle annulus
#     from looking like noise and makes the read direction stable.
#
# g) change inferability:
#   - The mapping uses at most two categories and each category appears both in the legend and in at
#     least one target marker. That repeated structure is enough for humans to infer "left legend color
#     selects right legend anchor color", which then drives the same stamping action as the original.
#
# post-change assumption audit for the new solver:
#   - Every valid input must contain exactly one triple whose outer annulus has anchor-bearing sprites,
#     whose middle annulus has 1-2 horizontal two-cell legend components, and whose inner rectangle has
#     singleton markers whose colors are covered by the legend keys.
#   - Every legend value must match a unique source anchor color; the generator enforces this by using
#     non-wallpaper anchor colors and by generating one legend entry per sprite type.
#   - Destination markers may repeat types, but all marker colors must be in the legend domain. The
#     generator uses counts >= 1 per type so every legend entry is supported by at least one marker.


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
                "coords": [(int(r), int(c)) for r, c in coords],
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


def _decode_legend_component(comp: dict, g: np.ndarray) -> tuple[int, int] | None:
    if int(comp["size"]) != 2:
        return None
    rr0, cc0, rr1, cc1 = map(int, comp["bbox"])
    if not (rr0 == rr1 and cc1 == cc0 + 1):
        return None
    left = int(g[rr0, cc0])
    right = int(g[rr0, cc1])
    if left == right:
        return None
    return int(left), int(right)


def _find_nested_triple(g: np.ndarray) -> dict:
    rects = _find_uniform_rectangles(g)
    candidates: list[tuple[tuple[int, ...], dict]] = []

    for outer in rects[:16]:
        or0, oc0, or1, oc1 = map(int, outer["rect"])
        for legend in rects:
            if int(legend["area"]) >= int(outer["area"]):
                continue
            lr0, lc0, lr1, lc1 = map(int, legend["rect"])
            if not (or0 + 1 < lr0 and oc0 + 1 < lc0 and lr1 < or1 - 1 and lc1 < oc1 - 1):
                continue
            for inner in rects:
                if int(inner["area"]) >= int(legend["area"]):
                    continue
                ir0, ic0, ir1, ic1 = map(int, inner["rect"])
                if not (lr0 + 1 < ir0 and lc0 + 1 < ic0 and ir1 < lr1 - 1 and ic1 < lc1 - 1):
                    continue

                src_mask = np.zeros_like(g, dtype=bool)
                src_mask[or0 + 1 : or1, oc0 + 1 : oc1] = True
                src_mask[lr0 : lr1 + 1, lc0 : lc1 + 1] = False
                if not bool(src_mask.any()):
                    continue
                src_bg = _mode_color(g[src_mask])
                src_fg = src_mask & (g != int(src_bg))
                src_comps = [comp for comp in _component_records(g, src_fg) if int(comp["size"]) > 1]
                if not src_comps:
                    continue
                anchor_set = {
                    int(anchor)
                    for comp in src_comps
                    for anchor in [_anchor_color_from_counts(comp["colors"])]
                    if anchor is not None
                }
                if not anchor_set:
                    continue

                legend_mask = np.zeros_like(g, dtype=bool)
                legend_mask[lr0 + 1 : lr1, lc0 + 1 : lc1] = True
                legend_mask[ir0 : ir1 + 1, ic0 : ic1 + 1] = False
                if not bool(legend_mask.any()):
                    continue
                legend_bg = _mode_color(g[legend_mask])
                legend_fg = legend_mask & (g != int(legend_bg))
                legend_comps = _component_records(g, legend_fg)
                if not legend_comps:
                    continue

                legend_map: dict[int, int] = {}
                ok_legend = True
                for comp in legend_comps:
                    pair = _decode_legend_component(comp, g)
                    if pair is None:
                        ok_legend = False
                        break
                    key, value = map(int, pair)
                    if key in legend_map and int(legend_map[key]) != int(value):
                        ok_legend = False
                        break
                    legend_map[int(key)] = int(value)
                if not ok_legend:
                    continue
                if not (1 <= len(legend_map) <= 2):
                    continue
                if set(map(int, legend_map.values())) != anchor_set:
                    continue

                inner_cells = g[ir0 + 1 : ir1, ic0 + 1 : ic1]
                if inner_cells.size == 0:
                    continue
                dst_bg = _mode_color(inner_cells)
                dst_fg = inner_cells != int(dst_bg)
                marker_coords = components_mask(dst_fg, connectivity=4)
                if not marker_coords:
                    continue
                if any(int(coords.shape[0]) != 1 for coords in marker_coords):
                    continue
                marker_colors = {int(inner_cells[int(r), int(c)]) for coords in marker_coords for r, c in coords}
                if not marker_colors.issubset(set(map(int, legend_map.keys()))):
                    continue
                if len(marker_colors) != len(legend_map):
                    continue

                score = (
                    len(legend_map),
                    len(src_comps),
                    int(outer["area"]) - int(legend["area"]),
                    int(legend["area"]) - int(inner["area"]),
                    int(outer["area"]),
                )
                candidates.append(
                    (
                        score,
                        {
                            "outer_rect": outer["rect"],
                            "outer_border": int(outer["border"]),
                            "outer_bg": int(src_bg),
                            "legend_rect": legend["rect"],
                            "legend_border": int(legend["border"]),
                            "legend_bg": int(legend_bg),
                            "inner_rect": inner["rect"],
                            "inner_border": int(inner["border"]),
                            "inner_bg": int(dst_bg),
                            "src_comps": src_comps,
                            "legend_map": legend_map,
                        },
                    )
                )

    assert candidates, "must find one nested source/legend/destination triple"
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a nested legend ring, then stamp outer-ring sprites into the innermost frame.

    Input structure:
    - Three bordered rectangles are nested: an outer source frame, a middle legend frame, and an inner destination frame.
    - Multi-cell sprites live only in the outer annulus, horizontal two-cell legend chips live only in the middle annulus, and singleton markers live only in the inner frame.

    Rule:
    - Read each legend chip left-to-right as marker_color -> anchor_color.
    - Find the source sprite whose unique rare color matches that anchor color.
    - For every inner-frame marker, decode its color through the legend and stamp the matched sprite with its anchor aligned to the marker position.

    Key insight: The middle rectangle is not decoration; it translates inner marker colors into the outer sprites' anchor colors, so solving requires both nested-region segmentation and explicit legend decoding.
    """
    g = np.array(grid, dtype=int)
    triple = _find_nested_triple(g)

    ir0, ic0, ir1, ic1 = map(int, triple["inner_rect"])
    inner_border = int(triple["inner_border"])
    inner_bg = int(triple["inner_bg"])
    src_comps = triple["src_comps"]
    legend_map = {int(k): int(v) for k, v in triple["legend_map"].items()}

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
        if marker_color not in legend_map:
            continue
        anchor_color = int(legend_map[int(marker_color)])
        if anchor_color not in templates:
            continue
        patch, mask, (ar, ac) = templates[int(anchor_color)]
        top = int(1 + int(rr) - int(ar))
        left = int(1 + int(cc) - int(ac))
        paste(out, patch, int(top), int(left), mask=mask, overwrite=True, clip=True)

    return out


# --------------- generator (combo D9 + D13) ---------------

GRID_SIZE = 30
WALLPAPER_COLORS = (4, 5)

SRC_RING_RANGE = (4, 5)
LEGEND_RING_RANGE = (2, 3)
DST_H_RANGE = (8, 10)
DST_W_RANGE = (8, 10)

SCENARIO_CHOICES = ((1, 2), (1, 3), (2, 2), (2, 3))
SOURCE_SPRITE_RANGE = (2, 4)
MARKER_PLACE_RETRIES = 25
SOURCE_SPRITE_PADDING = 1
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
    for _ in range(int(n_total - n_types)):
        counts[random.randrange(n_types)] += 1
    return counts


def _pick_distinct(candidates: list[int], k: int) -> list[int] | None:
    uniq = list(dict.fromkeys(int(x) for x in candidates))
    if len(uniq) < k:
        return None
    return random.sample(uniq, k)


def _choose_nested_dimensions(
    n_markers: int,
) -> tuple[int, int, int, int, int, int, int, int, int, int]:
    for _ in range(200):
        dst_h_lo = max(DST_H_RANGE[0], 9 if n_markers >= 3 else DST_H_RANGE[0])
        dst_w_lo = max(DST_W_RANGE[0], 9 if n_markers >= 3 else DST_W_RANGE[0])
        dst_h = random.randint(dst_h_lo, DST_H_RANGE[1])
        dst_w = random.randint(dst_w_lo, DST_W_RANGE[1])
        lg_top = random.randint(*LEGEND_RING_RANGE)
        lg_bottom = random.randint(*LEGEND_RING_RANGE)
        lg_left = random.randint(*LEGEND_RING_RANGE)
        lg_right = random.randint(*LEGEND_RING_RANGE)
        src_top = random.randint(*SRC_RING_RANGE)
        src_bottom = random.randint(*SRC_RING_RANGE)
        src_left = random.randint(*SRC_RING_RANGE)
        src_right = random.randint(*SRC_RING_RANGE)

        legend_h = int(dst_h + lg_top + lg_bottom + 2)
        legend_w = int(dst_w + lg_left + lg_right + 2)
        src_h = int(legend_h + src_top + src_bottom + 2)
        src_w = int(legend_w + src_left + src_right + 2)
        if src_h <= GRID_SIZE and src_w <= GRID_SIZE:
            return src_h, src_w, legend_h, legend_w, dst_h, dst_w, src_top, src_left, lg_top, lg_left
    raise ValueError("no triple-nested dimension fit")


def _assign_palette(n_types: int) -> dict | None:
    for _ in range(100):
        wallpaper_n = 5 if n_types == 2 else random.choice(list(WALLPAPER_COLORS))
        wallpaper = random.sample(list(range(10)), wallpaper_n)
        remaining = [c for c in range(10) if c not in wallpaper]
        random.shuffle(remaining)
        if len(remaining) < 1 + n_types:
            continue
        src_border = int(remaining.pop())
        anchors = [int(remaining.pop()) for _ in range(n_types)]

        if len(wallpaper) < 4:
            continue
        src_bg, legend_bg, dst_bg = random.sample(wallpaper, 3)
        legend_border_choices = [c for c in wallpaper if c != legend_bg]
        if not legend_border_choices:
            continue
        legend_border = int(random.choice(legend_border_choices))
        dst_border_choices = [c for c in wallpaper if c != dst_bg and c != legend_border]
        if not dst_border_choices:
            dst_border_choices = [c for c in wallpaper if c != dst_bg]
        if not dst_border_choices:
            continue
        dst_border = int(random.choice(dst_border_choices))

        marker_preferred = [c for c in wallpaper if c not in {dst_bg, legend_bg, dst_border}]
        marker_backup = [c for c in wallpaper if c not in {dst_bg, legend_bg}]
        marker_colors = _pick_distinct(marker_preferred, n_types) or _pick_distinct(marker_backup, n_types)
        if marker_colors is None:
            continue

        body_preferred = [c for c in wallpaper if c not in {src_bg, dst_bg, *marker_colors}]
        body_backup = [c for c in wallpaper if c not in {src_bg, dst_bg}]
        bodies = _pick_distinct(body_preferred, n_types) or _pick_distinct(body_backup, n_types)
        if bodies is None:
            continue

        return {
            "wallpaper": [int(c) for c in wallpaper],
            "src_border": int(src_border),
            "legend_border": int(legend_border),
            "dst_border": int(dst_border),
            "src_bg": int(src_bg),
            "legend_bg": int(legend_bg),
            "dst_bg": int(dst_bg),
            "anchors": [int(c) for c in anchors],
            "marker_colors": [int(c) for c in marker_colors],
            "bodies": [int(c) for c in bodies],
        }
    return None


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


def _place_source_sprites(
    sprite_types: list[dict],
    source_type_ids: list[int],
    src_h: int,
    src_w: int,
    src_bg: int,
    legend_r: int,
    legend_c: int,
    legend_h: int,
    legend_w: int,
) -> np.ndarray | None:
    inner = np.full((src_h - 2, src_w - 2), src_bg, dtype=int)
    blocker = np.full((legend_h, legend_w), -1, dtype=int)
    paste(inner, blocker, int(legend_r), int(legend_c))
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


def _place_legend_pairs(
    legend_h: int,
    legend_w: int,
    legend_bg: int,
    dst_h: int,
    dst_w: int,
    dst_r: int,
    dst_c: int,
    marker_colors: list[int],
    anchors: list[int],
) -> np.ndarray | None:
    inner = np.full((legend_h - 2, legend_w - 2), legend_bg, dtype=int)
    blocker = np.full((dst_h, dst_w), -1, dtype=int)
    paste(inner, blocker, int(dst_r), int(dst_c))

    top_band = int(dst_r)
    if top_band <= 0:
        return None
    total_w = int(inner.shape[1])
    n_pairs = len(marker_colors)
    needed = 2 if n_pairs == 1 else 5
    if total_w < needed:
        return None
    row = random.randint(0, top_band - 1)
    if n_pairs == 1:
        cols = [random.randint(0, total_w - 2)]
    else:
        start = random.randint(0, total_w - 5)
        cols = [int(start), int(start + 3)]

    items = list(zip(marker_colors, anchors))
    random.shuffle(items)
    for (marker_color, anchor_color), col in zip(items, cols):
        inner[row, int(col)] = int(marker_color)
        inner[row, int(col) + 1] = int(anchor_color)
    return inner


def _gen_attempt() -> np.ndarray | None:
    n_types, n_markers = random.choice(list(SCENARIO_CHOICES))
    palette = _assign_palette(n_types)
    if palette is None:
        return None

    sprite_specs = random.sample(SPRITE_SHAPES, n_types)
    sprite_types = [
        {
            "shape_rows": shape_rows,
            "anchor_rc": anchor_rc,
            "body": int(body),
            "anchor": int(anchor),
            "h": len(shape_rows),
            "w": len(shape_rows[0]),
        }
        for (shape_rows, anchor_rc), body, anchor in zip(
            sprite_specs, palette["bodies"], palette["anchors"]
        )
    ]

    marker_type_ids = [i for i, ct in enumerate(_choose_counts(n_markers, n_types)) for _ in range(ct)]
    random.shuffle(marker_type_ids)

    n_source = random.randint(max(2, n_types), SOURCE_SPRITE_RANGE[1])
    source_type_ids = [i for i, ct in enumerate(_choose_counts(n_source, n_types)) for _ in range(ct)]
    random.shuffle(source_type_ids)

    try:
        dims = _choose_nested_dimensions(n_markers)
    except ValueError:
        return None
    src_h, src_w, legend_h, legend_w, dst_h, dst_w, src_top, src_left, lg_top, lg_left = dims

    marker_result = _place_markers(sprite_types, marker_type_ids, dst_h, dst_w, palette["dst_bg"])
    if marker_result is None:
        return None
    out_inner, marker_points = marker_result

    output = _build_frame(dst_h, dst_w, palette["dst_border"], palette["dst_bg"])
    paste(output[1:-1, 1:-1], out_inner, 0, 0)

    source_inner = _place_source_sprites(
        sprite_types,
        source_type_ids,
        src_h,
        src_w,
        palette["src_bg"],
        src_top,
        src_left,
        legend_h,
        legend_w,
    )
    if source_inner is None:
        return None

    legend_inner = _place_legend_pairs(
        legend_h,
        legend_w,
        palette["legend_bg"],
        dst_h,
        dst_w,
        lg_top,
        lg_left,
        palette["marker_colors"],
        palette["anchors"],
    )
    if legend_inner is None:
        return None

    dst_rect = _build_frame(dst_h, dst_w, palette["dst_border"], palette["dst_bg"])
    for mr, mc, type_id in marker_points:
        dst_rect[1 + int(mr), 1 + int(mc)] = int(palette["marker_colors"][int(type_id)])

    legend_rect = _build_frame(legend_h, legend_w, palette["legend_border"], palette["legend_bg"])
    paste(legend_rect[1:-1, 1:-1], legend_inner, 0, 0)
    paste(legend_rect[1:-1, 1:-1], dst_rect, int(lg_top), int(lg_left))

    source_rect = _build_frame(src_h, src_w, palette["src_border"], palette["src_bg"])
    paste(source_rect[1:-1, 1:-1], source_inner, 0, 0)
    paste(source_rect[1:-1, 1:-1], legend_rect, int(src_top), int(src_left))

    grid = _make_wallpaper(random.choice(["vertical", "horizontal"]), palette["wallpaper"])
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
    raise RuntimeError(f"failed to generate combo a251c730 input after {MAX_GEN_ATTEMPTS} attempts")
