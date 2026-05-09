# Combo variant of a251c730 (index 20)
# combo_spec: D3/1+D4/0
# Name: 边框标记+旋转对齐

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


# Design analysis for the combo variant
# -------------------------------------
# a) concretization / behavior trace
#    - Original task:
#      1. detect a strict source rectangle full of separated two-color sprites
#      2. detect a strict destination rectangle whose interior contains singleton markers
#      3. match marker color -> source sprite anchor color
#      4. translate the unrotated sprite so its anchor lands on the marker
#    - This combo changes two stages at once:
#      1. destination markers move from the interior onto non-corner border cells
#      2. the marker side itself now encodes rotation:
#         top=0, right=90cw, bottom=180, left=270cw
#    - Concrete trace:
#      suppose the source sprite is
#          .B.
#          BAB
#          ..B
#      with A as the anchor. In the original task, an interior marker of color A simply
#      receives this shape translated without rotation. In the combo task, if the A-marker
#      sits on the right border, the sprite is first rotated 90 degrees clockwise, then its
#      rotated anchor is aligned to that border cell. Cells beyond the right frame edge are
#      clipped, so the visible remnant points inward from the marked side.
#    - The combined difference is therefore not "border markers" plus an unrelated extra rule:
#      the border side is exactly the control signal that chooses the rotation used before
#      the anchor-alignment paste.
# b) complexity calibration
#    - Marker count is fixed to 3 so one sample can show multiple side codes without
#      overcrowding the border.
#    - The generator fixes a single sprite type, so the same anchor color necessarily
#      repeats across 3 distinct sides. That makes the side-driven rotation cue explicit:
#      orientation changes while color identity stays constant.
#    - Sprite templates are restricted to asymmetric 3x3 shapes whose anchor is not on the
#      outermost row/column. This keeps all quarter-turns visually distinct and guarantees
#      side alignment clips something.
# c) implicit-assumption audit for main()
#    - Original destination detection assumed a fully uniform border and singleton interior
#      components. That fails because the combo destination has an empty interior and the
#      only deviations live on the border.
#    - Original stamping assumed the extracted template can be pasted immediately after
#      color matching. That fails because the anchor coordinates change after rotation.
#    - The rarest-color anchor heuristic still survives unchanged: source extraction still
#      identifies templates by the singleton color inside each source sprite.
#    - The redrawn destination frame remains safe: input markers only perturb the border,
#      while output stamping happens afterward and may overwrite some border cells.
# d) placement legality
#    - The generator is output-first: it chooses marker sides/positions, rotates the chosen
#      source template for that side, checks clipped visibility, checks pairwise non-overlap
#      of visible colored cells, and only then commits the stamp.
#    - Source sprites keep the original separation and padding constraints.
# e) visual observability
#    - Rotation depends only on a human-visible feature: which edge carries the marker.
#    - Because one anchor color repeats on different sides within the same sample, a solver
#      can observe the same source sprite appearing in multiple orientations.
#    - No ordering by raw color id is used anywhere in generation or solving.
# f) visual cleanliness
#    - Stamped footprints are forbidden from overlapping, and markers are kept non-adjacent
#      on the border, so each local border-to-inward imprint remains readable.
# g) inferability / anti-degeneration guard
#    - The generator rejects any sample where the rotated-border output equals the
#      border-without-rotation output. This prevents the combo from silently collapsing back
#      to the D3-only behavior.
# h) post-change assumptions main() now makes, and how generation guarantees them
#    - assumption: exactly one source rectangle still has a strict border and multi-cell
#      components. generator: source sprites stay away from the border.
#    - assumption: destination interior is uniform and border deviations are only the 2-3
#      marker cells. generator: destination input contains only border markers.
#    - assumption: every destination marker color exists among source anchors. generator:
#      markers are sampled from sprite type ids.
#    - assumption: every border marker is non-corner, so its side is unambiguous. generator:
#      candidates never include corners.
#    - assumption: when rotation is applied from the side code, some colored cells remain
#      visible inside the frame. generator: candidate acceptance explicitly checks this.


SIDE_TO_QTURNS_CW = {"top": 0, "right": 1, "bottom": 2, "left": 3}
SIDE_ORDER = ("top", "right", "bottom", "left")


def _interior_bg(g: np.ndarray, rect: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = map(int, rect)
    inner = g[int(r0) + 1 : int(r1), int(c0) + 1 : int(c1)]
    return int(Counter(map(int, inner.flatten())).most_common(1)[0][0])


def _rect_components(g: np.ndarray, rect: tuple[int, int, int, int]) -> tuple[int, list[dict]]:
    r0, c0, r1, c1 = map(int, rect)
    inner = g[int(r0) + 1 : int(r1), int(c0) + 1 : int(c1)]
    ibg = int(Counter(map(int, inner.flatten())).most_common(1)[0][0])
    comps: list[dict] = []

    for coords in components_mask(inner != int(ibg), connectivity=4):
        if coords.size == 0:
            continue
        rr0, cc0, rr1, cc1 = bbox_coords(coords)
        mask = np.zeros((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), dtype=bool)
        mask[coords[:, 0] - int(rr0), coords[:, 1] - int(cc0)] = True
        patch = inner[int(rr0) : int(rr1) + 1, int(cc0) : int(cc1) + 1].copy()
        patch[~mask] = 0

        vals = [int(inner[int(r), int(c)]) for r, c in coords]
        cols = Counter(vals)
        comps.append(
            {
                "size": int(coords.shape[0]),
                "colors": dict(cols),
                "bbox": (int(rr0), int(cc0), int(rr1), int(cc1)),
                "mask": mask,
                "patch": patch,
            }
        )

    comps.sort(key=lambda d: -int(d["size"]))
    return int(ibg), comps


def _border_markers(
    g: np.ndarray, rect: tuple[int, int, int, int], border_color: int
) -> list[tuple[int, int, int, str]]:
    r0, c0, r1, c1 = map(int, rect)
    pts: list[tuple[int, int, int, str]] = []
    for c in range(int(c0) + 1, int(c1)):
        if int(g[int(r0), int(c)]) != int(border_color):
            pts.append((int(r0), int(c), int(g[int(r0), int(c)]), "top"))
        if int(g[int(r1), int(c)]) != int(border_color):
            pts.append((int(r1), int(c), int(g[int(r1), int(c)]), "bottom"))
    for r in range(int(r0) + 1, int(r1)):
        if int(g[int(r), int(c0)]) != int(border_color):
            pts.append((int(r), int(c0), int(g[int(r), int(c0)]), "left"))
        if int(g[int(r), int(c1)]) != int(border_color):
            pts.append((int(r), int(c1), int(g[int(r), int(c1)]), "right"))
    return pts


def _rotate_template(
    patch: np.ndarray, mask: np.ndarray, anchor_rc: tuple[int, int], quarter_turns_cw: int
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    k_ccw = (-int(quarter_turns_cw)) % 4
    anchor_mask = np.zeros_like(mask, dtype=bool)
    anchor_mask[int(anchor_rc[0]), int(anchor_rc[1])] = True
    rot_patch = np.rot90(patch, k=int(k_ccw))
    rot_mask = np.rot90(mask, k=int(k_ccw))
    rot_anchor_mask = np.rot90(anchor_mask, k=int(k_ccw))
    ar, ac = map(int, np.argwhere(rot_anchor_mask)[0])
    return rot_patch, rot_mask, (int(ar), int(ac))


def _scan_rectangles(g: np.ndarray) -> tuple[list[tuple[int, tuple[int, int, int, int], int, int]], list[tuple[int, tuple[int, int, int, int], int, int]]]:
    H, W = g.shape
    strict_rects: list[tuple[int, tuple[int, int, int, int], int, int]] = []
    loose_rects: list[tuple[int, tuple[int, int, int, int], int, int]] = []
    min_h = 6
    min_w = 6

    for r0 in range(int(H)):
        for c0 in range(int(W)):
            bc = int(g[int(r0), int(c0)])
            for r1 in range(int(r0 + min_h - 1), int(H)):
                for c1 in range(int(c0 + min_w - 1), int(W)):
                    if (
                        int(g[int(r0), int(c1)]) != int(bc)
                        or int(g[int(r1), int(c0)]) != int(bc)
                        or int(g[int(r1), int(c1)]) != int(bc)
                    ):
                        continue

                    inner = g[int(r0) + 1 : int(r1), int(c0) + 1 : int(c1)]
                    if inner.size == 0:
                        continue
                    ibg = int(Counter(map(int, inner.flatten())).most_common(1)[0][0])
                    if int(ibg) == int(bc):
                        continue

                    top = g[int(r0), int(c0) : int(c1) + 1]
                    bottom = g[int(r1), int(c0) : int(c1) + 1]
                    left = g[int(r0) : int(r1) + 1, int(c0)]
                    right = g[int(r0) : int(r1) + 1, int(c1)]
                    area = int((int(r1) - int(r0) + 1) * (int(c1) - int(c0) + 1))

                    if bool((top == int(bc)).all()) and bool((bottom == int(bc)).all()) and bool(
                        (left == int(bc)).all()
                    ) and bool((right == int(bc)).all()):
                        strict_rects.append((int(area), (int(r0), int(c0), int(r1), int(c1)), int(bc), int(ibg)))
                        continue

                    mismatch_count = int(np.count_nonzero(top[1:-1] != int(bc)))
                    mismatch_count += int(np.count_nonzero(bottom[1:-1] != int(bc)))
                    mismatch_count += int(np.count_nonzero(left[1:-1] != int(bc)))
                    mismatch_count += int(np.count_nonzero(right[1:-1] != int(bc)))

                    if 2 <= mismatch_count <= 3 and bool((inner == int(ibg)).all()):
                        loose_rects.append((int(area), (int(r0), int(c0), int(r1), int(c1)), int(bc), int(ibg)))

    strict_rects.sort(reverse=True, key=lambda x: int(x[0]))
    loose_rects.sort(reverse=True, key=lambda x: int(x[0]))
    return strict_rects, loose_rects


def _extract_templates(src_comps: list[dict]) -> dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]]:
    templates: dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]] = {}
    for comp in src_comps:
        cols = {int(k): int(v) for k, v in comp["colors"].items() if int(k) != 0}
        if not cols:
            continue
        anchor_col = int(min(cols.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])
        patch = comp["patch"]
        mask = comp["mask"]
        anchors = np.argwhere((patch == int(anchor_col)) & mask)
        if anchors.size == 0:
            continue
        ar, ac = map(int, anchors[0])
        if int(anchor_col) not in templates or int(mask.sum()) > int(templates[int(anchor_col)][1].sum()):
            templates[int(anchor_col)] = (patch.copy(), mask.copy(), (int(ar), int(ac)))
    return templates


def _solve(grid: np.ndarray, *, rotate_markers: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    strict_rects, loose_rects = _scan_rectangles(g)
    assert strict_rects and loose_rects, "must find both strict and loose rectangles"

    src = None
    for _area, rect, bc, _ibg in strict_rects[:10]:
        ibg2, comps = _rect_components(g, rect)
        max_sz = max([int(c["size"]) for c in comps], default=0)
        if max_sz > 1:
            src = (rect, bc, ibg2, comps)
            break

    assert src is not None, "must find the source rectangle"
    (_sr, _sc, _sr1, _sc1), _src_bc, _src_bg, src_comps = src
    templates = _extract_templates(src_comps)
    assert templates, "must extract source templates"

    dst = None
    for _area, rect, bc, ibg in loose_rects[:12]:
        markers = _border_markers(g, rect, int(bc))
        if not (2 <= len(markers) <= 3):
            continue
        if not all(int(col) in templates for _r, _c, col, _side in markers):
            continue
        dst = (rect, bc, ibg, markers)
        break

    assert dst is not None, "must find the destination rectangle with border markers"
    (dr, dc, dr1, dc1), dst_bc, dst_bg, dst_markers = dst

    out_h = int(dr1 - dr + 1)
    out_w = int(dc1 - dc + 1)
    out = np.full((out_h, out_w), int(dst_bg), dtype=int)
    out[0, :] = int(dst_bc)
    out[-1, :] = int(dst_bc)
    out[:, 0] = int(dst_bc)
    out[:, -1] = int(dst_bc)

    for mr, mc, mcol, side in dst_markers:
        patch, mask, anchor_rc = templates[int(mcol)]
        if rotate_markers:
            patch, mask, anchor_rc = _rotate_template(patch, mask, anchor_rc, SIDE_TO_QTURNS_CW[str(side)])
        ar, ac = map(int, anchor_rc)
        top = int((int(mr) - int(dr)) - int(ar))
        left = int((int(mc) - int(dc)) - int(ac))
        paste(out, patch, int(top), int(left), mask=mask, overwrite=True, clip=True)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp source sprites into a destination frame via side-coded border markers.

    Input structure:
    - Two bordered rectangles sit on striped wallpaper: a strict source box with multi-cell
      two-color sprites, and a destination box whose interior is blank but whose border
      contains 2-3 non-corner singleton markers.
    - Each source sprite has one minority-color anchor pixel, and every destination marker
      reuses one of those anchor colors.

    Rule:
    - Match each border marker to the source sprite with the same anchor color.
    - Read the marker's border side as a rotation code (top/right/bottom/left =
      0/90/180/270 clockwise), rotate the sprite first, then align the rotated anchor pixel
      to the marker cell and clip anything outside the destination frame.
    - Clear the destination to its border/background template and output only that cropped
      destination rectangle after all rotated border-aligned stamps.

    Key insight: Color still identifies which sprite to reuse, but the border side carries a
    second instruction: how much to rotate that sprite before anchor alignment.
    """
    return _solve(grid, rotate_markers=True)


# --------------- generator (O2) ---------------

# === CFG: Canvas & layout ===
GRID_SIZE = 30
WALLPAPER_COLORS = (4, 5)
TOTAL_COLORS_TARGET = (7, 8, 9)
GAP_RANGE = (1, 4)

# === CFG: Rectangle size bounds ===
SRC_H_RANGE = (9, 19)
SRC_W_RANGE = (8, 18)
DST_H_RANGE = (9, 19)
DST_W_RANGE = (8, 20)
DST_W_SIDE_MAX = 20

# === CFG: Sprite & placement ===
SPRITE_TYPE_CHOICES = (1,)
MARKER_COUNT_RANGE = (3, 3)
SOURCE_SPRITE_RANGE = (2, 4)
MARKER_LAYOUT_RETRIES = 40
MARKER_PLACE_RETRIES = 40
SOURCE_SPRITE_PADDING = 1

# === CFG: Generation retries ===
MAX_GEN_ATTEMPTS = 300

SPRITE_SHAPES = [
    ((".B.", "BAB", "..B"), (1, 1)),
    (("B..", "BAB", ".BB"), (1, 1)),
    ((".BB", "BA.", "B.B"), (1, 1)),
    (("..B", "BAB", "BB."), (1, 1)),
    (("B.B", ".AB", ".B."), (1, 1)),
    ((".B.", "BAB", "B.."), (1, 1)),
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
            src_r, dst_r = top_margin, top_margin + dst_h + gap
        else:
            dst_r, src_r = top_margin, top_margin + dst_h + gap
    return (src_r, src_c), (dst_r, dst_c)


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
        border_choices = [c for c in wallpaper if c != src_border]
        if not border_choices:
            return None
        dst_border = random.choice(border_choices)

    body_pool = wallpaper[:]
    forced_body: int | None = None
    if body_extra:
        if not remaining:
            return None
        forced_body = remaining.pop()
        body_pool.append(forced_body)

    return wallpaper, src_border, dst_border, anchors, body_pool, forced_body


def _visible_stamp_stats(
    patch: np.ndarray,
    anchor_rc: tuple[int, int],
    marker_rc: tuple[int, int],
    frame_h: int,
    frame_w: int,
    bg: int,
) -> tuple[set[tuple[int, int]], int]:
    ar, ac = map(int, anchor_rc)
    mr, mc = map(int, marker_rc)
    top = int(mr - ar)
    left = int(mc - ac)

    visible: set[tuple[int, int]] = set()
    total_colored = 0
    for r in range(int(patch.shape[0])):
        for c in range(int(patch.shape[1])):
            if int(patch[int(r), int(c)]) == int(bg):
                continue
            total_colored += 1
            rr = int(top + int(r))
            cc = int(left + int(c))
            if 0 <= rr < frame_h and 0 <= cc < frame_w:
                visible.add((int(rr), int(cc)))
    return visible, int(total_colored - len(visible))


def _side_positions(side: str, dst_h: int, dst_w: int) -> list[tuple[int, int]]:
    if side == "top":
        return [(0, int(c)) for c in range(1, int(dst_w) - 1)]
    if side == "bottom":
        return [(int(dst_h) - 1, int(c)) for c in range(1, int(dst_w) - 1)]
    if side == "left":
        return [(int(r), 0) for r in range(1, int(dst_h) - 1)]
    return [(int(r), int(dst_w) - 1) for r in range(1, int(dst_h) - 1)]


def _place_border_markers(
    sprite_types: list[dict], marker_type_ids: list[int], dst_h: int, dst_w: int, dst_bg: int, dst_border: int
) -> tuple[np.ndarray, list[tuple[int, int, int, str]]] | None:
    output = _build_frame(dst_h, dst_w, dst_border, dst_bg)
    occupied: set[tuple[int, int]] = set()
    marker_points: list[tuple[int, int, int, str]] = []

    for _ in range(MARKER_LAYOUT_RETRIES):
        output[:, :] = _build_frame(dst_h, dst_w, dst_border, dst_bg)
        occupied.clear()
        marker_points.clear()

        sides = random.sample(list(SIDE_ORDER), len(marker_type_ids))
        order = list(zip(marker_type_ids, sides))
        random.shuffle(order)
        placed_all = True

        for type_id, side in order:
            sprite = sprite_types[int(type_id)]
            base_patch = _render_sprite(sprite["shape_rows"], sprite["body"], sprite["anchor"], dst_bg)
            base_mask = base_patch != int(dst_bg)
            patch, mask, anchor_rc = _rotate_template(
                base_patch, base_mask, sprite["anchor_rc"], SIDE_TO_QTURNS_CW[str(side)]
            )

            candidates = _side_positions(str(side), dst_h, dst_w)
            random.shuffle(candidates)

            placed = False
            for mr, mc in candidates[: max(1, MARKER_PLACE_RETRIES * 4)]:
                visible, clipped = _visible_stamp_stats(patch, anchor_rc, (mr, mc), dst_h, dst_w, dst_bg)
                if clipped <= 0:
                    continue
                inner_visible = {(r, c) for r, c in visible if 0 < r < dst_h - 1 and 0 < c < dst_w - 1}
                if len(inner_visible) < 2:
                    continue
                if visible & occupied:
                    continue
                if any(abs(int(mr) - pr) + abs(int(mc) - pc) <= 1 for pr, pc, _tid, _side in marker_points):
                    continue

                ar, ac = map(int, anchor_rc)
                paste(output, patch, int(mr - ar), int(mc - ac), transparent=dst_bg, clip=True)
                occupied |= visible
                marker_points.append((int(mr), int(mc), int(type_id), str(side)))
                placed = True
                break

            if not placed:
                placed_all = False
                break

        if placed_all:
            marker_points.sort(key=lambda x: (int(x[0]), int(x[1]), int(x[2])))
            return output.copy(), marker_points.copy()

    return None


def _place_sources(
    sprite_types: list[dict], source_type_ids: list[int], src_h: int, src_w: int, src_bg: int
) -> np.ndarray | None:
    inner = np.full((src_h - 2, src_w - 2), src_bg, dtype=int)
    for type_id in source_type_ids:
        sprite = sprite_types[int(type_id)]
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
        paste(inner, patch, int(r0), int(c0), transparent=src_bg)
    return inner


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
    for i in range(int(n_types)):
        choices = [c for c in body_pool if c not in anchors and c != src_bg and c != dst_bg and c != dst_border]
        if not choices:
            return None
        if i == 0 and forced_body is not None and forced_body in choices:
            bodies.append(int(forced_body))
        else:
            bodies.append(int(random.choice(choices)))

    sprite_types = [
        {
            "shape_rows": shape_rows,
            "anchor_rc": anchor_rc,
            "body": int(body),
            "anchor": int(anchor),
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

    layout = random.choice(["side_by_side", "stacked"])
    try:
        src_h, src_w, dst_h, dst_w = _choose_dimensions(layout)
    except ValueError:
        return None
    if src_h * src_w <= dst_h * dst_w:
        return None

    marker_result = _place_border_markers(sprite_types, marker_type_ids, dst_h, dst_w, dst_bg, dst_border)
    if marker_result is None:
        return None
    output, marker_points = marker_result

    source_inner = _place_sources(sprite_types, source_type_ids, src_h, src_w, src_bg)
    if source_inner is None:
        return None

    source_rect = _build_frame(src_h, src_w, src_border, src_bg)
    paste(source_rect[1:-1, 1:-1], source_inner, 0, 0)

    dest_rect = _build_frame(dst_h, dst_w, dst_border, dst_bg)
    for mr, mc, type_id, _side in marker_points:
        dest_rect[int(mr), int(mc)] = int(sprite_types[int(type_id)]["anchor"])

    try:
        src_pos, dst_pos = _place_rectangles(layout, src_h, src_w, dst_h, dst_w)
    except ValueError:
        return None

    grid = _make_wallpaper(random.choice(["vertical", "horizontal"]), wallpaper)
    try:
        paste(grid, source_rect, int(src_pos[0]), int(src_pos[1]))
        paste(grid, dest_rect, int(dst_pos[0]), int(dst_pos[1]))
    except ValueError:
        return None

    try:
        predicted = main(grid)
        no_rotation = _solve(grid, rotate_markers=False)
    except Exception:
        return None

    if not np.array_equal(predicted, output):
        return None
    if np.array_equal(predicted, no_rotation):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a251c730 combo input after %d attempts" % MAX_GEN_ATTEMPTS)
