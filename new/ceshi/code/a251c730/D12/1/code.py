# Variant of a251c730
# Dimension: D12 — 颜色约束体系 (meta)
# Change: 锚点颜色不在壁纸调色板中、不等于源背景色和目标背景色；边框颜色不等于内部背景色；总共7-9种不同颜色... → 最少化颜色数量
#
# Design analysis
# - Concrete change:
#   - The task still contains two bordered rectangles on a striped wallpaper.
#   - The destination rectangle still contains 2-3 singleton markers, and the output still stamps source sprites onto those markers.
#   - The visible change is entirely in the palette system: inputs now use only 5-6 total colors.
#   - To make that reduction obvious, all sprite bodies reuse a wallpaper stripe color, both rectangles share one border color,
#     and some samples also reuse that same border color as one sprite anchor.
# - Behavior trace:
#   - Original pipeline: source sprite -> pick its singleton anchor color -> find marker of the same color -> stamp.
#   - Variant pipeline: source sprite -> still find its singleton anchor color -> marker matches by that color -> stamp.
#   - The rule is unchanged, but the same color can now play multiple roles at once, so the grid is visually more ambiguous.
# - Complexity calibration:
#   - Keeping 1-2 sprite types preserves the original reasoning burden.
#   - Reusing one wallpaper color for every body and sharing the border color across both rectangles makes the palette change
#     easy to notice without introducing extra subcases.
#   - A 2-type branch optionally reuses the border color as an anchor color, adding ambiguity while remaining readable.
# - Hidden-assumption audit for main():
#   - The original template code dropped color 0 from sprite colors. That was only accidentally safe when black did not appear
#     in important roles. The variant allows any ARC color, so color 0 must be treated as a normal color.
#   - The original anchor selection used a hidden color-id tie-break. The variant keeps anchors singleton, so ties should not
#     matter; the solver now requires a unique least-frequent color instead of sorting by color id.
#   - The original rectangle pick simply took the first large source-like and destination-like candidates. With fewer colors and
#     shared border colors, accidental nested rectangles are more plausible, so the solver now chooses the best disjoint pair.
# - Placement legality:
#   - Destination markers are still placed by first stamping the full sprite into an internal scratch canvas, so every marker
#     guarantees an in-bounds final stamp.
#   - Source sprites still use padding from each other and from the frame, so component extraction remains clean even when an
#     anchor reuses the border color.
# - Visual observability:
#   - Across examples, humans can see that the puzzle suddenly uses only 5-6 colors, with wallpaper colors showing up inside
#     sprites and one border color reused on both rectangles.
#   - The transformation itself stays single-rule and consistent; only the color regime changes.
# - Cleanliness:
#   - The output remains a single stamped destination frame with no extra overlays.
#   - Color reuse is concentrated in body/background/border roles rather than adding noisy new marks.

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy source sprites into a destination frame even when the palette is deliberately compressed.

    Input structure:
    - Two bordered rectangles sit on a striped wallpaper, but only 5-6 colors are used overall.
    - Sprite body colors can be wallpaper colors, both rectangles can share a border color, and one anchor can also reuse it.

    Rule:
    - Inside the source rectangle, each sprite still has one uniquely rare anchor pixel.
    - Match destination singleton markers by that anchor color, align anchor-to-marker, and stamp the whole sprite into a cleared destination frame.

    Key insight: Color reuse increases ambiguity globally, but each sprite still advertises its own anchor through internal rarity.
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
                    "mask": mask,
                    "patch": patch,
                }
            )

        comps.sort(key=lambda d: -int(d["size"]))
        return ibg, comps

    rects: list[tuple[int, tuple[int, int, int, int], int]] = []
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
                    rects.append((area, (r0, c0, r1, c1), bc))

    assert rects, "must find bordered rectangles"

    def disjoint(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
        ar0, ac0, ar1, ac1 = a
        br0, bc0, br1, bc1 = b
        return ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0

    candidates: list[dict] = []
    for area, rect, bc in rects:
        bg, comps = rect_components(rect)
        if len(comps) < 2:
            continue
        max_sz = max(int(comp["size"]) for comp in comps)
        candidates.append(
            {
                "area": area,
                "rect": rect,
                "bc": int(bc),
                "bg": int(bg),
                "comps": comps,
                "max_sz": max_sz,
            }
        )

    src_candidates = [cand for cand in candidates if int(cand["max_sz"]) > 1]
    dst_candidates = [cand for cand in candidates if int(cand["max_sz"]) == 1]

    best_pair: tuple[dict, dict] | None = None
    best_score: tuple[int, int, int, int, int] | None = None
    for src in src_candidates:
        for dst in dst_candidates:
            if int(src["area"]) <= int(dst["area"]):
                continue
            if not disjoint(src["rect"], dst["rect"]):
                continue
            score = (
                int(src["area"]) + int(dst["area"]),
                int(src["area"]),
                int(dst["area"]),
                len(src["comps"]),
                len(dst["comps"]),
            )
            if best_score is None or score > best_score:
                best_pair = (src, dst)
                best_score = score

    assert best_pair is not None, "must find disjoint source and destination rectangles"
    src, dst = best_pair
    dr, dc, dr1, dc1 = map(int, dst["rect"])
    dst_bc = int(dst["bc"])
    dst_bg = int(dst["bg"])

    templates: dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]] = {}
    for comp in src["comps"]:
        cols = {int(k): int(v) for k, v in comp["colors"].items()}
        least = min(cols.values())
        anchor_cols = [col for col, count in cols.items() if int(count) == least]
        if len(anchor_cols) != 1:
            continue
        anchor_col = int(anchor_cols[0])
        patch = comp["patch"]
        mask = comp["mask"]
        anchors = np.argwhere((patch == anchor_col) & mask)
        if anchors.size != 2:
            continue
        ar, ac = map(int, anchors[0])
        if anchor_col not in templates or int(mask.sum()) > int(templates[anchor_col][1].sum()):
            templates[anchor_col] = (patch.copy(), mask.copy(), (ar, ac))

    assert templates, "must find source sprite templates"

    out_h = int(dr1 - dr + 1)
    out_w = int(dc1 - dc + 1)
    out = np.full((out_h, out_w), dst_bg, dtype=int)
    out[0, :] = dst_bc
    out[-1, :] = dst_bc
    out[:, 0] = dst_bc
    out[:, -1] = dst_bc

    inner = g[dr + 1 : dr1, dc + 1 : dc1]
    for rr, cc in np.argwhere(inner != dst_bg):
        mr = int(dr + 1 + rr)
        mc = int(dc + 1 + cc)
        mcol = int(g[mr, mc])
        if mcol not in templates:
            continue
        patch, mask, (ar, ac) = templates[mcol]
        top = int((mr - dr) - ar)
        left = int((mc - dc) - ac)
        paste(out, patch, top, left, mask=mask, overwrite=True, clip=True)

    return out


GRID_SIZE = 30
WALLPAPER_COLOR_COUNT = 3
TOTAL_COLORS_TARGET = (5, 6)
GAP_RANGE = (1, 4)

SRC_H_RANGE = (9, 19)
SRC_W_RANGE = (8, 18)
DST_H_RANGE = (9, 19)
DST_W_RANGE = (8, 20)
DST_W_SIDE_MAX = 20

SPRITE_TYPE_CHOICES = (1, 2)
MARKER_COUNT_RANGE = (2, 3)
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


def _assign_palette(n_types: int) -> tuple[list[int], int, int, int, int, list[int], int]:
    all_colors = list(range(10))
    wallpaper = random.sample(all_colors, WALLPAPER_COLOR_COUNT)
    src_bg, dst_bg, body = random.sample(wallpaper, WALLPAPER_COLOR_COUNT)
    remaining = [c for c in all_colors if c not in wallpaper]
    random.shuffle(remaining)
    border = remaining.pop()

    if n_types == 1:
        anchors = [remaining.pop()]
        total_colors = 5
    else:
        if random.random() < 0.5:
            anchors = [border, remaining.pop()]
            random.shuffle(anchors)
            total_colors = 5
        else:
            anchors = [remaining.pop(), remaining.pop()]
            total_colors = 6
    return wallpaper, src_bg, dst_bg, body, border, anchors, total_colors


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


def _variant_checks(
    grid: np.ndarray,
    wallpaper: list[int],
    border: int,
    sprite_types: list[dict],
    target_total_colors: int,
) -> bool:
    colors = {int(x) for x in grid.flatten()}
    if len(colors) != target_total_colors:
        return False
    if target_total_colors not in TOTAL_COLORS_TARGET:
        return False
    if any(int(sprite["body"]) not in wallpaper for sprite in sprite_types):
        return False
    if len({int(sprite["body"]) for sprite in sprite_types}) != 1:
        return False
    if not all(int(sprite["body"]) != int(sprite["anchor"]) for sprite in sprite_types):
        return False
    if any(int(sprite["anchor"]) == int(sprite_types[0]["body"]) for sprite in sprite_types):
        return False
    if not all(int(sprite["anchor"]) != int(border) for sprite in sprite_types) and len(sprite_types) == 1:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_types = random.choice(list(SPRITE_TYPE_CHOICES))
    wallpaper, src_bg, dst_bg, body, border, anchors, target_total = _assign_palette(n_types)

    sprite_specs = random.sample(SPRITE_SHAPES, n_types)
    sprite_types = [
        {
            "shape_rows": shape_rows,
            "anchor_rc": anchor_rc,
            "body": body,
            "anchor": anchor,
            "h": len(shape_rows),
            "w": len(shape_rows[0]),
        }
        for (shape_rows, anchor_rc), anchor in zip(sprite_specs, anchors)
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
        src_pos, dst_pos = _place_rectangles(layout, src_h, src_w, dst_h, dst_w)
    except ValueError:
        return None

    grid = _make_wallpaper(random.choice(["vertical", "horizontal"]), wallpaper)
    paste(grid, source_rect, src_pos[0], src_pos[1])
    paste(grid, dest_rect, dst_pos[0], dst_pos[1])

    if not _variant_checks(grid, wallpaper, border, sprite_types, target_total):
        return None
    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a251c730 D12 input after {MAX_GEN_ATTEMPTS} attempts")
