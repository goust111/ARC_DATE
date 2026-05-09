# Combo variant of a251c730 (index 13)
# combo_spec: D2/0+D8/0
# Name: 三色精灵+几何中心锚点

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location

# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    - The global canvas stays the same: striped wallpaper plus one source frame and one destination frame.
#    - Source sprites are now tri-color, but the two non-body colors are both singletons. This deliberately
#      makes the old "pick the least-frequent color" story unusable: the least frequency is tied.
#    - The operational key is the geometric center cell of each sprite bbox:
#        old tri-color intuition: marker color might try to match a singleton accent
#        new combo rule: marker color matches the center cell, and stamping aligns that center cell
#        Example 2x3 sprite:
#            A B C
#            B B B
#          Here A and C are both singleton accents, so "rarest color" is ambiguous.
#          The bbox center is the top-middle B. A destination marker of color B means
#          "place the whole sprite so that top-middle B sits on the marker".
#    - Difference from the original task is therefore twofold and coupled:
#      the marker color is no longer an accent color, and the paste offset is no longer measured
#      from one of the singleton accent pixels.
# b) Complexity calibration:
#    - Each sprite has color counts (1,1,3) or (1,1,4). This keeps the tri-color change visible while
#      avoiding too many combinatorial cases.
#    - We keep 1-2 sprite types and 2-4 total source copies, so each rule has repeated support.
#    - Center colors must be unique across types; otherwise destination markers would be ambiguous.
# c) Implicit assumption audit against the original `main()`:
#    - Rectangle detection is unaffected and can be reused.
#    - Component extraction still assumes source sprites are disconnected 4-neighbor components; generator
#      preserves padding so this remains true.
#    - The old solver assumed the minimum-frequency color identifies both the lookup key and the anchor point.
#      That assumption is false here because two colors share the minimum count. The new solver must ignore
#      frequency for anchoring and instead read the center cell directly.
#    - The old two-color template logic also collapsed all non-anchor pixels into one role. That is false here:
#      the stamped template must preserve both singleton accents and the majority body color.
# d) Placement legality:
#    - Generation remains output-first. Full sprite patches are first placed into a temporary destination
#      interior; only the center-color marker survives into the input destination frame.
#    - Because marker positions are recorded from legal full placements, replaying the same center alignment
#      inside `main()` is guaranteed to stay in bounds.
#    - Source sprites are placed with border/padding=1, so connected-component parsing is stable.
# e) Visual observability:
#    - Humans can see that source/output sprites use three colors, while destination markers use a common
#      body-like color rather than one of the singleton accents.
#    - Since the two singleton colors tie in frequency, examples visually discourage a "pick the rarest color"
#      heuristic and instead support the center-cell rule.
# f) Visual cleanliness:
#    - Only 2-3 markers appear in the destination input, so the blank frame stays sparse.
#    - All shapes are small (2x3 or 3x3), connected, center-filled, and non-overlapping.
# g) Color inference and non-degeneracy:
#    - Marker colors are exactly the sprite center colors with no hidden remapping.
#    - We force every center cell to be a body-colored cell, and body colors are unique per type.
#      This makes the lookup cue repeated and explicit across copies.
# h) Quantitative combo metric to check after implementation:
#    - Every source sprite should have sorted color counts equal to (1,1,3) or (1,1,4).
#    - On generated samples, the combo solver output should often differ from a counterfactual solver that
#      ties the least-frequent color by color id and aligns that pixel instead of the center.
# i) New solver assumptions that the generator must satisfy:
#    - Every source sprite component has exactly 3 colors and an occupied geometric center cell.
#    - Every sprite's center color is unique across types and appears as a destination marker color.
#    - Repeated copies of a type are identical, so any copy can serve as the template.


def _center_rc_from_shape(h: int, w: int) -> tuple[int, int]:
    """Return floor(center) in bbox-local coordinates."""
    return (int((h - 1) // 2), int((w - 1) // 2))


def _center_rc_from_patch(patch: np.ndarray) -> tuple[int, int]:
    """Return floor(center) for a cropped component patch."""
    h, w = map(int, patch.shape)
    return _center_rc_from_shape(h, w)


def _shape_info(shape_rows: tuple[str, ...]) -> tuple[tuple[int, int], tuple[int, int], Counter]:
    """Return (center_rc, rare_rc, counts) for a valid tri-color combo sprite shape."""
    counts: Counter = Counter()
    rare_hits: list[tuple[int, int]] = []
    for r, row in enumerate(shape_rows):
        for c, ch in enumerate(row):
            if ch == ".":
                continue
            counts[ch] += 1
            if ch == "A":
                rare_hits.append((int(r), int(c)))

    if len(rare_hits) != 1:
        raise ValueError("shape must contain exactly one 'A'")

    center_rc = _center_rc_from_shape(len(shape_rows), len(shape_rows[0]))
    if shape_rows[center_rc[0]][center_rc[1]] != "B":
        raise ValueError("shape center must be an occupied 'B' cell")

    if counts["A"] != 1 or counts["C"] != 1 or counts["B"] not in (3, 4):
        raise ValueError("shape counts must be A=1, C=1, B=3/4")

    occ = np.array([[ch != "." for ch in row] for row in shape_rows], dtype=bool)
    comps = components_mask(occ, connectivity=4)
    if len(comps) != 1:
        raise ValueError("shape must be 4-connected")

    return center_rc, rare_hits[0], counts


def _render_sprite(shape_rows: tuple[str, ...], body: int, rare: int, decor: int, bg: int) -> np.ndarray:
    """Render a tri-color sprite. A=rare accent, B=body, C=decoration, .=bg."""
    char_map = {"A": rare, "B": body, "C": decor}
    rows = [[char_map.get(ch, bg) for ch in row] for row in shape_rows]
    return np.array(rows, dtype=int)


def _is_combo_sprite(comp: dict) -> bool:
    """Return True iff a parsed source component matches the combo sprite family."""
    colors = {int(k): int(v) for k, v in comp["colors"].items()}
    if len(colors) != 3:
        return False
    counts = sorted(int(v) for v in colors.values())
    if counts not in ([1, 1, 3], [1, 1, 4]):
        return False
    patch = comp["patch"]
    mask = comp["mask"]
    ar, ac = _center_rc_from_patch(patch)
    if not bool(mask[int(ar), int(ac)]):
        return False
    center_col = int(patch[int(ar), int(ac)])
    if center_col not in colors:
        return False
    return int(colors[center_col]) == int(max(colors.values()))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the destination frame by matching ambiguous tri-color sprites through their bbox centers.

    Input structure:
    - Two bordered rectangles lie on a striped wallpaper: a source frame with disconnected tri-color sprites
      and a destination frame with sparse singleton markers.
    - Each source sprite has three colors, but its two accent colors are both singleton pixels, so frequency
      does not identify a unique anchor. The center cell of the sprite bbox is always occupied.

    Rule:
    - For each source sprite, read the color at its geometric center cell and use that as the template key.
    - Match each destination marker to the source sprite with the same center-cell color.
    - Stamp the full tri-color sprite so that its geometric center cell lands on the marker, then return the
      destination rectangle alone with its border preserved and its interior rebuilt from scratch.

    Key insight: the three-color source sprites intentionally make "least-frequent color" ambiguous, so both
    lookup and alignment must come from the occupied bbox center instead of from either singleton accent.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    def interior_bg(rect: tuple[int, int, int, int]) -> int:
        r0, c0, r1, c1 = map(int, rect)
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        return int(Counter(map(int, inner.flatten())).most_common(1)[0][0])

    def rect_components(rect: tuple[int, int, int, int]) -> tuple[int, list[dict]]:
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
                }
            )

        comps.sort(key=lambda d: -int(d["size"]))
        return int(ibg), comps

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
                    ibg = interior_bg((r0, c0, r1, c1))
                    if ibg == bc:
                        continue
                    area = int((r1 - r0 + 1) * (c1 - c0 + 1))
                    rects.append((area, (int(r0), int(c0), int(r1), int(c1)), int(bc), int(ibg)))

    assert rects, "must find bordered rectangles"

    rects.sort(reverse=True, key=lambda x: int(x[0]))
    rects = rects[:6]

    src = None
    dst = None
    for _area, rect, bc, _ibg in rects:
        ibg2, comps = rect_components(rect)
        max_sz = max([int(c["size"]) for c in comps], default=0)
        if max_sz > 1 and src is None:
            src = (rect, bc, ibg2, comps)
        if max_sz == 1 and comps and dst is None:
            dst = (rect, bc, ibg2, comps)

    assert src is not None and dst is not None, "must find source and destination rectangles"

    (_sr, _sc, _sr1, _sc1), _src_bc, _src_bg, src_comps = src
    (dr, dc, dr1, dc1), dst_bc, dst_bg, _dst_comps = dst

    templates: dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]] = {}
    for comp in src_comps:
        if not _is_combo_sprite(comp):
            continue
        patch = comp["patch"]
        mask = comp["mask"]
        ar, ac = _center_rc_from_patch(patch)
        center_col = int(patch[int(ar), int(ac)])
        if center_col not in templates or int(mask.sum()) > int(templates[center_col][1].sum()):
            templates[center_col] = (patch.copy(), mask.copy(), (int(ar), int(ac)))

    assert templates, "no center-anchored tri-color templates found"

    out_h = int(dr1 - dr + 1)
    out_w = int(dc1 - dc + 1)
    out = np.full((out_h, out_w), int(dst_bg), dtype=int)
    out[0, :] = int(dst_bc)
    out[-1, :] = int(dst_bc)
    out[:, 0] = int(dst_bc)
    out[:, -1] = int(dst_bc)

    inner = g[dr + 1 : dr1, dc + 1 : dc1]
    for rr, cc in np.argwhere(inner != int(dst_bg)):
        mr = int(dr + 1 + int(rr))
        mc = int(dc + 1 + int(cc))
        mcol = int(g[mr, mc])
        if mcol not in templates:
            continue
        patch, mask, (ar, ac) = templates[mcol]
        top = int((mr - dr) - ar)
        left = int((mc - dc) - ac)
        paste(out, patch, top, left, mask=mask, overwrite=True, clip=True)

    return out


# --------------- generator (O2) ---------------

# === CFG: Canvas & layout ===
GRID_SIZE = 30
WALLPAPER_COLORS = 5
GAP_RANGE = (1, 4)

# === CFG: Rectangle size bounds ===
SRC_H_RANGE = (9, 19)
SRC_W_RANGE = (8, 18)
DST_H_RANGE = (9, 19)
DST_W_RANGE = (8, 20)
DST_W_SIDE_MAX = 20

# === CFG: Sprite & placement ===
SPRITE_TYPE_CHOICES = (1, 2)
MARKER_COUNT_RANGE = (2, 3)
SOURCE_SPRITE_RANGE = (2, 4)
MARKER_PLACE_RETRIES = 25
SOURCE_SPRITE_PADDING = 1

# === CFG: Generation retries ===
MAX_GEN_ATTEMPTS = 300

# Every shape below has:
# - exactly one 'A' singleton accent and one 'C' singleton decoration
# - an occupied geometric center cell containing 'B'
# - counts (1,1,3) or (1,1,4), making least-frequency ambiguous
SPRITE_SHAPES = [
    ("ABC", "BBB"),
    ("ABB", "BBC"),
    ("AB.", "BBC"),
    ("ABB", "BC."),
    ("AB.", "BBB", ".C."),
    (".BA", "BBB", "C.."),
    ("A..", "BBB", ".C."),
    ("..A", "BBB", ".C."),
    ("A..", "BBB", "..C"),
    ("..A", "BBB", "C.."),
    ("A.B", "BBB", ".C."),
    (".BA", "BBB", "..C"),
]

for _shape in SPRITE_SHAPES:
    _center_rc, _rare_rc, _counts = _shape_info(_shape)
    assert len(_shape) in (2, 3)
    assert len(_shape[0]) == 3


def _make_wallpaper(direction: str, palette: list[int]) -> np.ndarray:
    """Create a 30x30 striped wallpaper."""
    period = len(palette)
    stripe = np.array(palette * ((GRID_SIZE + period - 1) // period), dtype=int)[:GRID_SIZE]
    if direction == "vertical":
        return np.tile(stripe, (GRID_SIZE, 1))
    return np.tile(stripe.reshape(-1, 1), (1, GRID_SIZE))


def _build_frame(h: int, w: int, border: int, bg: int) -> np.ndarray:
    """Create an h x w rectangle with a 1-pixel border and interior filled with bg."""
    rect = np.full((h, w), bg, dtype=int)
    rect[0, :] = rect[-1, :] = border
    rect[:, 0] = rect[:, -1] = border
    return rect


def _choose_counts(n_total: int, n_types: int) -> list[int]:
    """Distribute n_total items across n_types bins, each bin >= 1."""
    counts = [1] * n_types
    for _ in range(n_total - n_types):
        counts[random.randrange(n_types)] += 1
    return counts


def _choose_dimensions(layout: str) -> tuple[int, int, int, int]:
    """Pick (src_h, src_w, dst_h, dst_w) that fit the 30x30 canvas for the given layout."""
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
    """Compute non-overlapping (row, col) origins for source and destination rectangles."""
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


def _assign_palette(n_types: int) -> tuple[list[int], int, int, list[int]] | None:
    """Allocate wallpaper, borders, and singleton-accent colors."""
    all_colors = list(range(10))
    wallpaper = random.sample(all_colors, WALLPAPER_COLORS)
    remaining = [c for c in all_colors if c not in wallpaper]
    random.shuffle(remaining)

    needed = 2 + n_types
    if len(remaining) < needed:
        return None

    src_border = remaining.pop()
    dst_border = remaining.pop()
    rare_colors = [remaining.pop() for _ in range(n_types)]
    return wallpaper, src_border, dst_border, rare_colors


def _place_markers(
    sprite_types: list[dict], marker_type_ids: list[int], dst_h: int, dst_w: int, dst_bg: int
) -> tuple[np.ndarray, list[tuple[int, int, int]]] | None:
    """Place full sprite stamps in a temporary output interior and record center-marker positions."""
    inner = np.full((dst_h - 2, dst_w - 2), dst_bg, dtype=int)
    marker_points: list[tuple[int, int, int]] = []
    for type_id in marker_type_ids:
        sprite = sprite_types[type_id]
        patch = _render_sprite(sprite["shape_rows"], sprite["body"], sprite["rare"], sprite["decor"], dst_bg)
        placed = False
        for _ in range(MARKER_PLACE_RETRIES):
            try:
                r0, c0 = random_free_location(inner, patch, transparent=dst_bg)
            except ValueError:
                return None
            mr, mc = r0 + sprite["center_rc"][0], c0 + sprite["center_rc"][1]
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
    """Place tri-color sprites into the source interior."""
    inner = np.full((src_h - 2, src_w - 2), src_bg, dtype=int)
    for type_id in source_type_ids:
        sprite = sprite_types[type_id]
        patch = _render_sprite(sprite["shape_rows"], sprite["body"], sprite["rare"], sprite["decor"], src_bg)
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
    """Single attempt to generate a valid input grid."""
    n_types = random.choice(list(SPRITE_TYPE_CHOICES))

    palette_result = _assign_palette(n_types)
    if palette_result is None:
        return None
    wallpaper, src_border, dst_border, rare_colors = palette_result

    src_bg, dst_bg = random.sample(wallpaper, 2)
    sprite_palette = [c for c in wallpaper if c not in {src_bg, dst_bg}]
    if len(sprite_palette) < n_types + 1:
        return None

    bodies = random.sample(sprite_palette, n_types)
    decor_choices = [c for c in sprite_palette if c not in bodies]
    if not decor_choices:
        return None
    decor = random.choice(decor_choices)

    sprite_specs = random.sample(SPRITE_SHAPES, n_types)
    sprite_types: list[dict] = []
    for shape_rows, body, rare in zip(sprite_specs, bodies, rare_colors):
        center_rc, rare_rc, counts = _shape_info(shape_rows)
        sprite_types.append(
            {
                "shape_rows": shape_rows,
                "center_rc": center_rc,
                "rare_rc": rare_rc,
                "body": int(body),
                "rare": int(rare),
                "decor": int(decor),
                "center_color": int(body),
                "h": len(shape_rows),
                "w": len(shape_rows[0]),
                "area": int(sum(counts.values())),
            }
        )

    if len({int(sprite["center_color"]) for sprite in sprite_types}) != n_types:
        return None

    n_markers = random.randint(*MARKER_COUNT_RANGE)
    marker_type_ids = [i for i, ct in enumerate(_choose_counts(n_markers, n_types)) for _ in range(ct)]
    random.shuffle(marker_type_ids)

    n_source = random.randint(max(2, n_types), SOURCE_SPRITE_RANGE[1])
    source_type_ids = [i for i, ct in enumerate(_choose_counts(n_source, n_types)) for _ in range(ct)]
    random.shuffle(source_type_ids)

    layout = random.choice(["side_by_side", "stacked"])
    try:
        src_h, src_w, dst_h, dst_w = _choose_dimensions(layout)
    except ValueError:
        return None
    if src_h * src_w <= dst_h * dst_w:
        return None

    marker_result = _place_markers(sprite_types, marker_type_ids, dst_h, dst_w, dst_bg)
    if marker_result is None:
        return None
    out_inner, marker_points = marker_result

    output = _build_frame(dst_h, dst_w, dst_border, dst_bg)
    paste(output[1:-1, 1:-1], out_inner, 0, 0)

    source_inner = _place_sources(sprite_types, source_type_ids, src_h, src_w, src_bg)
    if source_inner is None:
        return None

    source_rect = _build_frame(src_h, src_w, src_border, src_bg)
    paste(source_rect[1:-1, 1:-1], source_inner, 0, 0)

    dest_rect = _build_frame(dst_h, dst_w, dst_border, dst_bg)
    for mr, mc, type_id in marker_points:
        dest_rect[1 + mr, 1 + mc] = sprite_types[type_id]["center_color"]

    try:
        src_pos, dst_pos = _place_rectangles(layout, src_h, src_w, dst_h, dst_w)
    except ValueError:
        return None

    grid = _make_wallpaper(random.choice(["vertical", "horizontal"]), wallpaper)
    paste(grid, source_rect, src_pos[0], src_pos[1])
    paste(grid, dest_rect, dst_pos[0], dst_pos[1])

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
    raise RuntimeError(
        "failed to generate a251c730 combo variant D2/0+D8/0 after %d attempts" % MAX_GEN_ATTEMPTS
    )
