# Variant of a251c730
# Dimension: D9 — 矩形识别与区分 (meta)
# Change: 源矩形通过内部含多像素连通分量来识别（精灵），目标矩形通过内部仅含单像素标记来识别；源矩形面积严格大于目标矩形... → 边框颜色区分

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location

# Design analysis for the D9 variant
# a) Concrete behavior and trace:
#    - The grid still contains exactly two non-overlapping framed rectangles on striped wallpaper.
#    - The source rectangle still stores multi-cell sprites; the destination rectangle still stores
#      singleton anchor markers. The visible change is that both rectangles may have the same outer size,
#      so size can no longer identify the role.
#    - Instead, the wallpaper itself becomes the disambiguation cue: the destination border color is one
#      of the wallpaper stripe colors, while the source border color is not.
#    - Example trace:
#      original: solver scans candidate rectangles, sees one with multi-cell components and one with only
#      singleton components, then uses those roles directly while generator also enforced source area > dest.
#      variant: solver first recovers the two framed rectangles, then recovers the wallpaper palette from the
#      cells outside them, then decides "border in wallpaper => destination" and "border outside wallpaper
#      => source", and only after that reads sprites / markers from the chosen rectangles.
# b) Complexity calibration:
#    - The role cue is binary and repeated across every stripe cell, so it is visible in 2-3 examples.
#    - The rectangles keep the old sprite/marker structure, so matching by anchor color still has repeated
#      support and does not collapse into a single brittle case.
#    - Equal outer dimensions are generated frequently enough that area order is visibly useless.
# c) Old assumptions that no longer hold:
#    - Any logic relying on "source area > destination area" must be removed.
#    - Any logic that infers rectangle role before reading the wallpaper palette is unsafe, because the
#      destination border may now visually blend with the wallpaper colors while the source border may not.
#    - Anchor identification must not use color-id tie breaking; the generator guarantees a unique singleton
#      anchor per sprite, so the solver now requires that unique minimum frequency directly.
# d) Placement legality:
#    - The generator keeps rectangles non-overlapping and within the 30x30 canvas.
#    - Markers are still placed by first checking where a full sprite stamp would fit, then reducing the
#      input destination rectangle to singleton anchor pixels. This guarantees that the output stamping is legal.
#    - Source sprites keep a one-cell safety margin to avoid merging into ambiguous components.
# e) Visual observability:
#    - Same-size rectangles force the human solver to look at the border/wallpaper relation instead of size.
#    - The destination border repeats a wallpaper stripe color that is plainly visible outside the rectangles.
# f) Visual cleanliness:
#    - There are still only two framed regions and no overlap.
#    - The wallpaper remains simple stripes, so the border-membership cue is easy to read.
# g) Color-change inferability:
#    - No hidden color arithmetic is introduced. Colors only act as equality keys:
#      anchor color matches marker color; border color either belongs to the wallpaper palette or not.
# h) New solver assumptions audited against the generator:
#    - Exactly two salient bordered rectangles exist: generator builds only those two.
#    - The uncovered background outside both rectangles is a clean stripe wallpaper: generator guarantees it.
#    - Exactly one rectangle border color belongs to the recovered wallpaper palette: generator guarantees
#      dst_border in wallpaper and src_border outside wallpaper.
#    - Source components each have one unique least-frequent color: generator uses one singleton anchor pixel.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp source sprites into a destination frame, with rectangle roles read from border-vs-wallpaper membership.

    Input structure:
    - Two bordered rectangles sit on a striped wallpaper, and they may have the same outer size.
    - The source frame holds multi-cell sprites; the destination frame holds singleton anchor markers.
    - The source border color is absent from the wallpaper palette, while the destination border color is one of the wallpaper stripe colors.

    Rule:
    - Detect the two framed rectangles and recover the wallpaper palette from the cells outside them.
    - Use the border-color convention to decide which rectangle is the sprite library and which is the marker canvas.
    - For each destination marker, stamp the source sprite with the same anchor color so that its anchor lands on the marker, then return the destination rectangle alone.

    Key insight: Rectangle identity no longer comes from relative area or interior component type; it comes from whether the frame color belongs to the ambient wallpaper stripes.
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
            cols = Counter(int(inner[r, c]) for r, c in coords)
            comps.append(
                {
                    "size": int(coords.shape[0]),
                    "colors": dict(cols),
                    "mask": mask,
                    "patch": patch,
                }
            )

        comps.sort(key=lambda d: -int(d["size"]))
        return int(ibg), comps

    def find_rectangles() -> list[dict]:
        rects: list[dict] = []
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
                        rects.append(
                            {
                                "rect": (int(r0), int(c0), int(r1), int(c1)),
                                "bc": int(bc),
                                "ibg": int(ibg),
                                "area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
                            }
                        )
        return rects

    def disjoint(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
        ar0, ac0, ar1, ac1 = a
        br0, bc0, br1, bc1 = b
        return ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0

    def wallpaper_info(rect_a: tuple[int, int, int, int], rect_b: tuple[int, int, int, int]) -> tuple[str, list[int]] | None:
        outside = np.ones((H, W), dtype=bool)
        for r0, c0, r1, c1 in (rect_a, rect_b):
            outside[r0 : r1 + 1, c0 : c1 + 1] = False
        if not bool(outside.any()):
            return None

        def periodic_palette(seq: list[int]) -> list[int] | None:
            for p in (4, 5):
                if len(seq) < p:
                    continue
                head = seq[:p]
                if len(set(head)) != p:
                    continue
                if all(int(seq[i]) == int(head[i % p]) for i in range(len(seq))):
                    return [int(v) for v in head]
            return None

        col_seq: list[int] = []
        vertical_ok = True
        for c in range(W):
            vals = {int(v) for v in g[outside[:, c], c]}
            if len(vals) != 1:
                vertical_ok = False
                break
            col_seq.append(int(next(iter(vals))))
        if vertical_ok:
            palette = periodic_palette(col_seq)
            if palette is not None:
                return "vertical", palette

        row_seq: list[int] = []
        horizontal_ok = True
        for r in range(H):
            vals = {int(v) for v in g[r, outside[r]]}
            if len(vals) != 1:
                horizontal_ok = False
                break
            row_seq.append(int(next(iter(vals))))
        if horizontal_ok:
            palette = periodic_palette(row_seq)
            if palette is not None:
                return "horizontal", palette

        return None

    rects = find_rectangles()
    assert len(rects) >= 2, "must find at least two bordered rectangles"

    chosen: dict | None = None
    best_score = -1
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            a = rects[i]
            b = rects[j]
            if not disjoint(a["rect"], b["rect"]):
                continue
            info = wallpaper_info(a["rect"], b["rect"])
            if info is None:
                continue
            _direction, wallpaper_palette = info
            pair = [a, b]
            src_candidates = [entry for entry in pair if int(entry["bc"]) not in wallpaper_palette]
            dst_candidates = [entry for entry in pair if int(entry["bc"]) in wallpaper_palette]
            if len(src_candidates) != 1 or len(dst_candidates) != 1:
                continue

            src_entry = src_candidates[0]
            dst_entry = dst_candidates[0]
            src_bg, src_comps = rect_components(src_entry["rect"])
            dst_bg, dst_comps = rect_components(dst_entry["rect"])
            if not src_comps or max(int(comp["size"]) for comp in src_comps) <= 1:
                continue
            if not dst_comps or any(int(comp["size"]) != 1 for comp in dst_comps):
                continue

            score = int(a["area"]) + int(b["area"])
            if score > best_score:
                best_score = score
                chosen = {
                    "src_entry": src_entry,
                    "dst_entry": dst_entry,
                    "src_bg": int(src_bg),
                    "src_comps": src_comps,
                    "dst_bg": int(dst_bg),
                    "dst_comps": dst_comps,
                }

    assert chosen is not None, "must isolate the two framed rectangles"

    src_entry = chosen["src_entry"]
    dst_entry = chosen["dst_entry"]
    src_rect = src_entry["rect"]
    dst_rect = dst_entry["rect"]
    src_comps = chosen["src_comps"]
    dst_bg = int(chosen["dst_bg"])
    dst_comps = chosen["dst_comps"]

    templates: dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]] = {}
    for comp in src_comps:
        cols = {int(k): int(v) for k, v in comp["colors"].items()}
        if not cols:
            continue
        min_count = min(cols.values())
        anchor_colors = [col for col, count in cols.items() if int(count) == int(min_count)]
        assert len(anchor_colors) == 1, "source sprite anchor must be uniquely least frequent"
        anchor_col = int(anchor_colors[0])
        patch = comp["patch"]
        mask = comp["mask"]
        anchors = np.argwhere((patch == anchor_col) & mask)
        assert anchors.shape[0] == 1, "anchor color must occur exactly once per sprite"
        ar, ac = map(int, anchors[0])
        if anchor_col not in templates or int(mask.sum()) > int(templates[anchor_col][1].sum()):
            templates[anchor_col] = (patch.copy(), mask.copy(), (ar, ac))

    dr0, dc0, dr1, dc1 = map(int, dst_rect)
    dst_bc = int(dst_entry["bc"])
    out = np.full((dr1 - dr0 + 1, dc1 - dc0 + 1), int(dst_bg), dtype=int)
    out[0, :] = out[-1, :] = dst_bc
    out[:, 0] = out[:, -1] = dst_bc

    inner = g[dr0 + 1 : dr1, dc0 + 1 : dc1]
    for rr, cc in np.argwhere(inner != int(dst_bg)):
        mcol = int(inner[rr, cc])
        if mcol not in templates:
            continue
        patch, mask, (ar, ac) = templates[mcol]
        top = int(1 + rr - ar)
        left = int(1 + cc - ac)
        paste(out, patch, top, left, mask=mask, overwrite=True, clip=True)

    return out


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
SPRITE_TYPE_CHOICES = (1, 2)
MARKER_COUNT_RANGE = (2, 3)
SOURCE_SPRITE_RANGE = (2, 4)
MARKER_PLACE_RETRIES = 25
SOURCE_SPRITE_PADDING = 1

# === CFG: Generation retries ===
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
    for _ in range(n_total - n_types):
        counts[random.randrange(n_types)] += 1
    return counts


def _choose_dimensions(layout: str) -> tuple[int, int, int, int]:
    gap = random.randint(*GAP_RANGE)
    h_min = max(SRC_H_RANGE[0], DST_H_RANGE[0])
    h_max = min(SRC_H_RANGE[1], DST_H_RANGE[1])
    w_min = max(SRC_W_RANGE[0], DST_W_RANGE[0])
    w_max = min(SRC_W_RANGE[1], DST_W_RANGE[1], DST_W_SIDE_MAX)
    if layout == "side_by_side":
        w_cap = min(w_max, (GRID_SIZE - gap) // 2)
        if w_cap < w_min:
            raise ValueError("no side-by-side equal-width fit")
        rect_w = random.randint(w_min, w_cap)
        rect_h = random.randint(h_min, h_max)
    else:
        h_cap = min(h_max, (GRID_SIZE - gap) // 2)
        if h_cap < h_min:
            raise ValueError("no stacked equal-height fit")
        rect_h = random.randint(h_min, h_cap)
        rect_w = random.randint(w_min, w_max)
    return rect_h, rect_w, rect_h, rect_w


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

    layout = random.choice(["side_by_side", "stacked"])
    try:
        src_h, src_w, dst_h, dst_w = _choose_dimensions(layout)
    except ValueError:
        return None
    if src_h * src_w != dst_h * dst_w:
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
        dest_rect[1 + mr, 1 + mc] = sprite_types[type_id]["anchor"]

    try:
        src_pos, dst_pos = _place_rectangles(layout, src_h, src_w, dst_h, dst_w)
    except ValueError:
        return None

    grid = _make_wallpaper(random.choice(["vertical", "horizontal"]), wallpaper)
    paste(grid, source_rect, src_pos[0], src_pos[1])
    paste(grid, dest_rect, dst_pos[0], dst_pos[1])

    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a251c730 D9 input after {MAX_GEN_ATTEMPTS} attempts")
