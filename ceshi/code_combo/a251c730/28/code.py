# Combo variant of a251c730 (index 28)
# combo_spec: D1/1+D2/0+D8/0
# Name: 主体色匹配+三色精灵+中心锚点

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location

# Design analysis
# a) Concrete behavior and trace:
#    - The scene still has two bordered rectangles on striped wallpaper.
#    - Source sprites are now tri-color: body B is the majority color, decoration C appears twice,
#      and rare anchor A appears once, but A is no longer operational.
#    - A destination marker uses the body color, not the rare color, and the placed alignment cell
#      is the bbox center, not the rare pixel.
#    - Example trace on shape ("AB.", "BCB", "..C"):
#        original task: marker color = A, align local A at (0,0) to the marker.
#        fused variant: marker color = B, align local center C at (1,1) to the marker.
#      So both the lookup cue and the stamped offset differ from the original.
# b) Complexity calibration:
#    - Generation is tightened to exactly 2 sprite types so body-color matching cannot collapse into a
#      one-type trivial case.
#    - Both types always appear in source and destination.
#    - Body colors are unique across types; center cells are occupied and always decoration-colored.
#      This makes the D1 and D8 changes visually obvious and removes match ambiguity.
# c) Implicit assumption review:
#    - Original main() assumed the least-frequent color is both lookup key and placement anchor.
#      That is invalid twice here: the key is the unique max-frequency color, and placement uses the
#      geometric center regardless of the rare pixel location.
#    - Original two-color logic also treated "everything except anchor" as one body bucket; that is
#      invalid because a third decoration color must be preserved in the pasted template.
#    - The new solver therefore accepts only 3-color components with visible counts (1,2,3/4) and an
#      occupied bbox center cell.
# d) Placement legality:
#    - Destination markers are still generated output-first by placing the full sprite patch first and
#      then recording only the center position as the input marker. This guarantees in-bounds replay.
#    - Source sprites are placed with padding so connected-component extraction remains stable.
# e) Visual observability:
#    - The marker color now matches the repeated body color visible in source sprites.
#    - The marker position in the output becomes the center decoration color, not the input marker color
#      and not the rare singleton color, so the center-anchor rule is readable from examples.
# f) Visual cleanliness:
#    - Destination inputs remain sparse singleton markers; no overlapping cues are shown.
#    - Tri-color sprites use only one shared decoration color and two distinct body colors, and the
#      key body/border colors are kept off the wallpaper palette so the important cues stay legible.
# g) Variant metric used after implementation:
#    - On generated samples, compare this combo solver against the original solver on the same input.
#      A valid non-degenerate generator should make those outputs differ consistently because markers use
#      body colors and placement uses centers instead of rare pixels.
# h) New solver assumptions that the generator must satisfy:
#    - Each source component is a connected tri-color sprite with counts exactly (1,2,3) or (1,2,4).
#    - The bbox center cell is occupied.
#    - Body colors are unique across the two sprite types and every destination marker uses one of them.


def _center_rc_from_shape(h: int, w: int) -> tuple[int, int]:
    return (int((h - 1) // 2), int((w - 1) // 2))


def _center_rc_from_patch(patch: np.ndarray) -> tuple[int, int]:
    h, w = map(int, patch.shape)
    return _center_rc_from_shape(h, w)


def _classify_tricolor_sprite(colors: dict[int, int]) -> tuple[int, int, int] | None:
    if len(colors) != 3:
        return None
    ordered = sorted((int(cnt), int(col)) for col, cnt in colors.items())
    counts = [cnt for cnt, _col in ordered]
    if counts[0] != 1 or counts[1] != 2 or counts[2] not in (3, 4):
        return None
    anchor_col = int(ordered[0][1])
    decor_col = int(ordered[1][1])
    body_col = int(ordered[2][1])
    return anchor_col, decor_col, body_col


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp tri-color source sprites into the destination frame via body-colored center anchors.

    Input structure:
    - Two bordered rectangles lie on a striped wallpaper: the source rectangle contains disconnected
      tri-color sprites, while the destination rectangle contains only singleton markers.
    - In each sprite, the dominant body color is the lookup cue, the singleton rare color is decorative,
      and the operational anchor is the occupied geometric center of the sprite's bbox.

    Rule:
    - Parse each source sprite, identify its unique majority body color, and keep its full three-color patch.
    - For every destination marker, select the sprite whose body color matches the marker color.
    - Paste that sprite so its bbox-center cell lands on the marker, then return the reconstructed
      destination rectangle alone.

    Key insight: solving now requires three separate inferences at once: ignore the rare singleton as a
    distractor, use the majority body color as the type key, and align the sprite by its geometric center.
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

    assert rects

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
        roles = _classify_tricolor_sprite({int(k): int(v) for k, v in comp["colors"].items()})
        if roles is None:
            continue
        _anchor_col, _decor_col, body_col = roles
        patch = comp["patch"]
        mask = comp["mask"]
        ar, ac = _center_rc_from_patch(patch)
        if not bool(mask[ar, ac]):
            continue
        if int(body_col) not in templates or int(mask.sum()) > int(templates[int(body_col)][1].sum()):
            templates[int(body_col)] = (patch.copy(), mask.copy(), (int(ar), int(ac)))

    assert templates, "no combo templates found"

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


GRID_SIZE = 30
GAP_RANGE = (1, 4)

SRC_H_RANGE = (9, 19)
SRC_W_RANGE = (8, 18)
DST_H_RANGE = (9, 19)
DST_W_RANGE = (8, 20)
DST_W_SIDE_MAX = 20

SPRITE_TYPE_CHOICES = (2,)
MARKER_COUNT_RANGE = (2, 3)
SOURCE_SPRITE_RANGE = (2, 4)
MARKER_PLACE_RETRIES = 25
SOURCE_SPRITE_PADDING = 1

MAX_GEN_ATTEMPTS = 400

# All templates below are connected tri-color sprites with:
# - counts A=1, C=2, B in {3,4}
# - occupied geometric center
# - center character = C, so center placement is visibly different from body-color matching
SPRITE_SHAPES = [
    ("ABB", "BCC", "..B"),
    (".AB", "BCB", "C.."),
    ("AB.", "BCB", "..C"),
    ("AB.", "BCB", ".BC"),
    ("A..", "BCB", ".BC"),
]


def _shape_info(shape_rows: tuple[str, ...]) -> tuple[tuple[int, int], tuple[int, int], Counter, str]:
    counts: Counter = Counter()
    anchor_rc: tuple[int, int] | None = None
    for r, row in enumerate(shape_rows):
        for c, ch in enumerate(row):
            if ch == ".":
                continue
            counts[ch] += 1
            if ch == "A":
                anchor_rc = (int(r), int(c))
    assert anchor_rc is not None
    center_rc = _center_rc_from_shape(len(shape_rows), len(shape_rows[0]))
    center_ch = shape_rows[center_rc[0]][center_rc[1]]
    assert counts["A"] == 1
    assert counts["C"] == 2
    assert counts["B"] in (3, 4)
    assert center_ch == "C"
    return anchor_rc, center_rc, counts, center_ch


for _shape in SPRITE_SHAPES:
    _anchor_rc, _center_rc, _counts, _center_ch = _shape_info(_shape)
    assert len(_shape) in (2, 3)
    assert len(_shape[0]) == 3
    assert _anchor_rc != _center_rc


def _render_sprite(shape_rows: tuple[str, ...], body: int, anchor: int, decor: int, bg: int) -> np.ndarray:
    char_map = {"A": anchor, "B": body, "C": decor}
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


def _assign_palette() -> tuple[list[int], int, int, int, int, list[int], list[int], int]:
    all_colors = list(range(10))
    wallpaper = random.sample(all_colors, 3)
    remaining = [c for c in all_colors if c not in wallpaper]
    random.shuffle(remaining)

    src_bg, dst_bg = [int(v) for v in random.sample(wallpaper, 2)]
    src_border = int(remaining.pop())
    dst_border = int(remaining.pop())
    bodies = [int(remaining.pop()) for _ in range(2)]
    anchors = [int(remaining.pop()) for _ in range(2)]
    decor = int(remaining.pop())
    return wallpaper, src_border, dst_border, src_bg, dst_bg, bodies, anchors, decor


def _place_markers(
    sprite_types: list[dict], marker_type_ids: list[int], dst_h: int, dst_w: int, dst_bg: int
) -> tuple[np.ndarray, list[tuple[int, int, int]]] | None:
    inner = np.full((dst_h - 2, dst_w - 2), dst_bg, dtype=int)
    marker_points: list[tuple[int, int, int]] = []
    for type_id in marker_type_ids:
        sprite = sprite_types[type_id]
        patch = _render_sprite(sprite["shape_rows"], sprite["body"], sprite["anchor"], sprite["decor"], dst_bg)
        placed = False
        for _ in range(MARKER_PLACE_RETRIES):
            try:
                r0, c0 = random_free_location(inner, patch, transparent=dst_bg)
            except ValueError:
                return None
            mr = int(r0 + sprite["center_rc"][0])
            mc = int(c0 + sprite["center_rc"][1])
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
    sprite_types: list[dict], source_type_ids: list[int], src_h: int, src_w: int, src_bg: int
) -> np.ndarray | None:
    inner = np.full((src_h - 2, src_w - 2), src_bg, dtype=int)
    for type_id in source_type_ids:
        sprite = sprite_types[type_id]
        patch = _render_sprite(sprite["shape_rows"], sprite["body"], sprite["anchor"], sprite["decor"], src_bg)
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
    n_types = 2

    wallpaper, src_border, dst_border, src_bg, dst_bg, bodies, anchors, decor = _assign_palette()
    if src_bg == dst_bg:
        return None
    if len(set(bodies)) != n_types:
        return None
    if decor in bodies or decor in anchors or decor in (src_bg, dst_bg, src_border, dst_border):
        return None
    if set(bodies) & set(anchors):
        return None

    sprite_shapes = random.sample(SPRITE_SHAPES, n_types)
    sprite_types: list[dict] = []
    for type_id, (shape_rows, body, anchor) in enumerate(zip(sprite_shapes, bodies, anchors)):
        anchor_rc, center_rc, counts, _center_ch = _shape_info(shape_rows)
        sprite_types.append(
            {
                "type_id": int(type_id),
                "shape_rows": shape_rows,
                "anchor_rc": anchor_rc,
                "center_rc": center_rc,
                "body": int(body),
                "anchor": int(anchor),
                "decor": int(decor),
                "area": int(sum(counts.values())),
                "h": len(shape_rows),
                "w": len(shape_rows[0]),
            }
        )

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
        dest_rect[1 + mr, 1 + mc] = int(sprite_types[type_id]["body"])

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
    raise RuntimeError(f"failed to generate a251c730 combo variant after {MAX_GEN_ATTEMPTS} attempts")
