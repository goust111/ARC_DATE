# Combo variant of a251c730 (index 23)
# combo_spec: D4/1+D15/2
# Name: 镜像翻转+精灵内嵌方向

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location

# Design analysis
# ===============
# fused_design:
#   summary:
#     - Keep the original wallpaper + two bordered rectangles layout.
#     - Preserve color matching: each singleton marker color still selects one sprite type.
#     - Fuse two orientation mechanisms in sequence instead of in parallel:
#       1. source decoding: the source sprite's anchor corner encodes how to normalize it back to a
#          canonical top-left-anchor orientation.
#       2. destination transform: after normalization, the marker position in the destination interior
#          applies horizontal and/or vertical mirroring relative to the destination center.
#   concrete_behavior:
#     input:
#       - Source rectangle: 2-4 disconnected multicolor sprites, 1-2 types.
#       - Each sprite has one unique singleton anchor color at a bbox corner.
#       - Destination rectangle: only singleton markers using those anchor colors.
#     output:
#       - Clear the destination interior to its background.
#       - For each marker color, find the matching source sprite.
#       - Read anchor corner code:
#         * top-left     -> id
#         * top-right    -> h
#         * bottom-left  -> v
#         * bottom-right -> hv (same visual effect as 180-degree rotation)
#       - Apply that same self-inverse transform to recover the canonical sprite.
#       - Then inspect marker position:
#         * marker in left half  -> add horizontal flip
#         * marker in upper half -> add vertical flip
#       - Align the twice-processed anchor onto the marker and stamp.
#   behavior_trace:
#     example:
#       canonical_sprite:
#         rows: ["AB.", "B..", "BBB"]
#         anchor: top-left
#       source_view:
#         if the generator chooses source_mode = hv, the source frame shows the sprite rotated 180°
#         with the anchor at the bottom-right corner.
#       original_task:
#         same marker color would stamp that bottom-right-anchor source view directly, with no reorientation.
#       fused_task:
#         first decode bottom-right -> apply hv -> recover canonical top-left-anchor form.
#         then if the marker sits in the destination's upper-left quadrant, apply hv again because
#         it is both left-of-center and above-center.
#         result: the output equals hv(canonical), not "raw source as seen" and not "canonical only".
#       difference:
#         source-local anchor geometry and destination-global marker geometry both matter, and they matter
#         in a fixed order.
#   complexity_calibration:
#     - Both control variables use only four states (`id/h/v/hv`), so the task is still learnable.
#     - The generator forces at least one non-identity source mode and always includes horizontal-triggered
#       and vertical-triggered destination markers, preventing collapse to either single variant.
#     - Canonical sprites are chosen from asymmetric motifs that differ under h/v/hv, so every transform is visible.
#   implicit_assumption_audit:
#     - Rectangle discovery remains unchanged; the variant does not alter frame syntax.
#     - Template extraction can no longer cache the raw source patch as the stamp template. The solver must
#       normalize by anchor-corner code first; the generator enforces "anchor is exactly one corner pixel".
#     - The original anchor selection used a least-frequency tie break. Here the generator guarantees a unique
#       singleton anchor color, so the solver can require exactly one least-frequent color.
#     - After normalization, the anchor position is known to be top-left, but after destination mirroring it
#       moves again. The solver therefore recomputes anchor coordinates for the second transform instead of
#       assuming they stay fixed.
#     - Marker placement can no longer use an arbitrary free cell: it must reserve the full footprint of the
#       canonical sprite after the marker-induced transform. Source placement must separately use the source-view
#       transform so the anchor-corner legend shown to the solver is legal and visible.
#   placement_legality:
#     - Output-first generation enumerates marker cells whose quadrant implies a desired marker mode, and only
#       accepts cells where the transformed canonical footprint stays in-bounds and non-overlapping.
#     - Source placement uses transformed source patches with one-cell padding, preserving the original spacing rule.
#   visual_observability:
#     - Anchor corner is a human-visible symbol; no color-id ordering or hidden metadata is used.
#     - Marker-position mirroring is visible because at least two different destination quadrants appear per task.
#     - Source decoding is visible because at least one sprite type is shown in a non-identity source orientation.
#   visual_cleanliness:
#     - Destination input still contains only singleton markers, so the cue layer stays sparse.
#     - Output sprites are prevented from overlapping, and input markers stay Manhattan distance >= 2 apart.
#   non_degeneracy_goal:
#     - Compared against "decode only" and "marker-flip only" ablations, generated samples should usually differ
#       from both, demonstrating that the fused behavior is genuinely required.


MODES = ("id", "h", "v", "hv")
CORNER_TO_MODE = {
    "top_left": "id",
    "top_right": "h",
    "bottom_left": "v",
    "bottom_right": "hv",
}


def _apply_mode(arr: np.ndarray, mode: str) -> np.ndarray:
    if mode == "id":
        return arr.copy()
    if mode == "h":
        return np.fliplr(arr).copy()
    if mode == "v":
        return np.flipud(arr).copy()
    if mode == "hv":
        return np.rot90(arr, 2).copy()
    raise ValueError(f"unknown transform mode: {mode}")


def _transform_template(
    patch: np.ndarray, mask: np.ndarray, anchor_rc: tuple[int, int], mode: str
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Apply a self-inverse mirror/180 transform and update the anchor coordinate."""
    out_patch = _apply_mode(patch, mode)
    out_mask = _apply_mode(mask.astype(int), mode).astype(bool)
    ar, ac = map(int, anchor_rc)
    h, w = map(int, patch.shape)
    if mode == "id":
        return out_patch, out_mask, (int(ar), int(ac))
    if mode == "h":
        return out_patch, out_mask, (int(ar), int(w - 1 - ac))
    if mode == "v":
        return out_patch, out_mask, (int(h - 1 - ar), int(ac))
    return out_patch, out_mask, (int(h - 1 - ar), int(w - 1 - ac))


def _mode_from_corner(ar: int, ac: int, h: int, w: int) -> str | None:
    if (int(ar), int(ac)) == (0, 0):
        return "id"
    if (int(ar), int(ac)) == (0, int(w) - 1):
        return "h"
    if (int(ar), int(ac)) == (int(h) - 1, 0):
        return "v"
    if (int(ar), int(ac)) == (int(h) - 1, int(w) - 1):
        return "hv"
    return None


def _normalize_template(
    patch: np.ndarray, mask: np.ndarray, anchor_col: int
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], str] | None:
    """Decode the anchor-corner legend and return the canonical top-left-anchor template."""
    anchors = np.argwhere((patch == int(anchor_col)) & mask)
    if anchors.shape[0] != 1:
        return None
    ar, ac = map(int, anchors[0])
    mode = _mode_from_corner(ar, ac, *map(int, mask.shape))
    if mode is None:
        return None
    norm_patch = _apply_mode(patch, mode)
    norm_mask = _apply_mode(mask.astype(int), mode).astype(bool)
    norm_anchors = np.argwhere((norm_patch == int(anchor_col)) & norm_mask)
    if norm_anchors.shape[0] != 1:
        return None
    nar, nac = map(int, norm_anchors[0])
    return norm_patch, norm_mask, (int(nar), int(nac)), mode


def _marker_mode(rr: int, cc: int, inner_h: int, inner_w: int) -> str:
    flip_lr = int(cc) < (int(inner_w) / 2.0)
    flip_ud = int(rr) < (int(inner_h) / 2.0)
    if flip_lr and flip_ud:
        return "hv"
    if flip_lr:
        return "h"
    if flip_ud:
        return "v"
    return "id"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the destination frame by decoding source sprites and then mirroring them onto markers.

    Input structure:
    - Two bordered rectangles lie on a striped wallpaper: a source frame with full sprites and a destination
      frame with only singleton anchor-colored markers.
    - Each source sprite still has one rare anchor color, but that anchor is forced onto one bbox corner so
      the sprite itself carries an orientation legend.

    Rule:
    - Match each destination marker to the source sprite with the same anchor color.
    - First read the anchor corner and normalize the sprite with the corresponding self-inverse transform
      (top-left=id, top-right=horizontal flip, bottom-left=vertical flip, bottom-right=180°).
    - Then mirror that normalized sprite according to the marker position inside the destination interior:
      left half adds a horizontal flip, upper half adds a vertical flip. Align the final anchor to the marker.

    Key insight: Color still selects the sprite type, but orientation is now factored into two readable cues:
    a local cue inside the source sprite (anchor corner) and a global cue from the marker's location in the
    destination frame.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    def interior_bg(rect: tuple[int, int, int, int]) -> int:
        r0, c0, r1, c1 = map(int, rect)
        inner = g[int(r0) + 1 : int(r1), int(c0) + 1 : int(c1)]
        return int(Counter(map(int, inner.flatten())).most_common(1)[0][0])

    def rect_components(rect: tuple[int, int, int, int]) -> tuple[int, list[dict]]:
        r0, c0, r1, c1 = map(int, rect)
        inner = g[int(r0) + 1 : int(r1), int(c0) + 1 : int(c1)]
        ibg = int(Counter(map(int, inner.flatten())).most_common(1)[0][0])
        comps = []

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

    rects: list[tuple[int, tuple[int, int, int, int], int, int]] = []
    min_h = 6
    min_w = 6
    for r0 in range(int(H)):
        for c0 in range(int(W)):
            bc = int(g[int(r0), int(c0)])
            for r1 in range(int(r0 + min_h - 1), int(H)):
                for c1 in range(int(c0 + min_w - 1), int(W)):
                    if int(g[int(r0), int(c1)]) != int(bc) or int(g[int(r1), int(c0)]) != int(bc) or int(
                        g[int(r1), int(c1)]
                    ) != int(bc):
                        continue
                    if not bool((g[int(r0), int(c0) : int(c1) + 1] == int(bc)).all()):
                        continue
                    if not bool((g[int(r1), int(c0) : int(c1) + 1] == int(bc)).all()):
                        continue
                    if not bool((g[int(r0) : int(r1) + 1, int(c0)] == int(bc)).all()):
                        continue
                    if not bool((g[int(r0) : int(r1) + 1, int(c1)] == int(bc)).all()):
                        continue
                    ibg = interior_bg((int(r0), int(c0), int(r1), int(c1)))
                    if int(ibg) == int(bc):
                        continue
                    area = int((int(r1) - int(r0) + 1) * (int(c1) - int(c0) + 1))
                    rects.append((int(area), (int(r0), int(c0), int(r1), int(c1)), int(bc), int(ibg)))

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
        if max_sz == 1 and len(comps) > 0 and dst is None:
            dst = (rect, bc, ibg2, comps)

    assert src is not None and dst is not None, "must find both source (multi-cell) and destination (singleton) rectangles"

    (_sr, _sc, _sr1, _sc1), _src_bc, _src_bg, src_comps = src
    (dr, dc, dr1, dc1), dst_bc, dst_bg, _dst_comps = dst

    templates: dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]] = {}
    for comp in src_comps:
        cols = {int(k): int(v) for k, v in comp["colors"].items() if int(k) != 0}
        if not cols:
            continue
        min_count = min(cols.values())
        anchor_candidates = [int(col) for col, cnt in cols.items() if int(cnt) == int(min_count)]
        if len(anchor_candidates) != 1:
            continue
        anchor_col = int(anchor_candidates[0])
        normalized = _normalize_template(comp["patch"], comp["mask"], anchor_col)
        if normalized is None:
            continue
        norm_patch, norm_mask, anchor_rc, _source_mode = normalized
        if int(anchor_col) not in templates or int(norm_mask.sum()) > int(templates[int(anchor_col)][1].sum()):
            templates[int(anchor_col)] = (norm_patch.copy(), norm_mask.copy(), anchor_rc)

    out_h = int(dr1 - dr + 1)
    out_w = int(dc1 - dc + 1)
    out = np.full((out_h, out_w), int(dst_bg), dtype=int)
    out[0, :] = int(dst_bc)
    out[-1, :] = int(dst_bc)
    out[:, 0] = int(dst_bc)
    out[:, -1] = int(dst_bc)

    inner = g[int(dr) + 1 : int(dr1), int(dc) + 1 : int(dc1)]
    inner_h, inner_w = map(int, inner.shape)
    for rr, cc in np.argwhere(inner != int(dst_bg)):
        mr = int(dr + 1 + int(rr))
        mc = int(dc + 1 + int(cc))
        mcol = int(g[mr, mc])
        if int(mcol) not in templates:
            continue
        patch, mask, anchor_rc = templates[int(mcol)]
        marker_mode = _marker_mode(int(rr), int(cc), inner_h, inner_w)
        tpatch, tmask, (tar, tac) = _transform_template(patch, mask, anchor_rc, marker_mode)
        top = int((mr - dr) - int(tar))
        left = int((mc - dc) - int(tac))
        paste(out, tpatch, int(top), int(left), mask=tmask, overwrite=True, clip=True)

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
SOURCE_SPRITE_PADDING = 1

# === CFG: Generation retries ===
MAX_GEN_ATTEMPTS = 600

# Canonical motifs: singleton anchor at top-left, and asymmetric under h/v/hv.
CANONICAL_SHAPES = [
    ("AB.", "B..", "BBB"),
    ("AB.", "BBB", "..B"),
    ("ABB", "B.B", "..B"),
    ("AB.", "B.B", "BBB"),
    ("ABB", "..B", ".BB"),
    ("A..", "BBB", "B.B"),
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

    if not remaining:
        return None
    src_border = remaining.pop()

    anchor_pool = [c for c in remaining if c != 0]
    if len(anchor_pool) < n_types:
        return None
    anchors = random.sample(anchor_pool, n_types)
    remaining = [c for c in remaining if c not in anchors]

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

    body_pool = [c for c in wallpaper if c != 0]
    forced_body: int | None = None
    if body_extra:
        nonzero_remaining = [c for c in remaining if c != 0]
        if not nonzero_remaining:
            return None
        forced_body = random.choice(nonzero_remaining)
        body_pool.append(forced_body)

    return wallpaper, src_border, dst_border, anchors, body_pool, forced_body


def _choose_source_modes(n_types: int) -> list[str]:
    non_identity = ["h", "v", "hv"]
    if n_types == 1:
        return [random.choice(non_identity)]
    first = random.choice(non_identity)
    second_pool = [mode for mode in MODES if mode != first]
    second = random.choice(second_pool)
    modes = [first, second]
    random.shuffle(modes)
    return modes


def _assign_marker_modes(sprite_types: list[dict], marker_type_ids: list[int]) -> list[str] | None:
    """
    Pick marker-induced mirror modes with branch coverage and anti-degeneracy constraints.

    Coverage:
    - Always show horizontal- and vertical-triggered marker positions.
    - With three markers, add either an identity or hv case as a third branch.

    Anti-degeneracy:
    - For every sprite type that appears in the destination frame, at least one of its markers must
      request a different mode than the source-embedded orientation. Otherwise the fused output can
      collapse back to the original "stamp the raw source view" behavior for that type.
    """
    base_modes = ["h", "v"]
    if len(marker_type_ids) == 3:
        base_modes.append(random.choice(["id", "hv"]))

    used_types = {int(type_id) for type_id in marker_type_ids}
    for _ in range(32):
        modes = base_modes.copy()
        random.shuffle(modes)
        has_mismatch = {type_id: False for type_id in used_types}
        for type_id, mode in zip(marker_type_ids, modes):
            if mode != str(sprite_types[int(type_id)]["source_mode"]):
                has_mismatch[int(type_id)] = True
        if all(has_mismatch.values()):
            return modes
    return None


def _render_canonical_patch(sprite: dict, bg: int) -> np.ndarray:
    return _render_sprite(sprite["shape_rows"], sprite["body"], sprite["anchor"], bg)


def _render_source_patch(sprite: dict, bg: int) -> np.ndarray:
    base = _render_canonical_patch(sprite, bg)
    return _apply_mode(base, sprite["source_mode"])


def _anchor_rc(patch: np.ndarray, anchor: int) -> tuple[int, int]:
    coords = np.argwhere(patch == int(anchor))
    if coords.shape[0] != 1:
        raise ValueError("sprite anchor must appear exactly once")
    r, c = map(int, coords[0])
    return int(r), int(c)


def _place_markers(
    sprite_types: list[dict], marker_type_ids: list[int], dst_h: int, dst_w: int, dst_bg: int
) -> tuple[np.ndarray, list[tuple[int, int, int, str]]] | None:
    """Place final transformed output sprites into the destination interior."""
    inner_h = int(dst_h - 2)
    inner_w = int(dst_w - 2)
    inner = np.full((inner_h, inner_w), dst_bg, dtype=int)
    occupied = np.zeros((inner_h, inner_w), dtype=bool)
    marker_points: list[tuple[int, int, int, str]] = []

    marker_modes = _assign_marker_modes(sprite_types, marker_type_ids)
    if marker_modes is None:
        return None
    marker_plan = list(zip(marker_type_ids, marker_modes))
    marker_plan.sort(key=lambda item: {"hv": 0, "h": 1, "v": 1, "id": 2}[item[1]])

    for type_id, desired_mode in marker_plan:
        sprite = sprite_types[int(type_id)]
        canonical = _render_canonical_patch(sprite, dst_bg)
        mask = canonical != int(dst_bg)
        patch, pmask, (ar, ac) = _transform_template(canonical, mask, (0, 0), desired_mode)
        ph, pw = map(int, patch.shape)

        candidates: list[tuple[int, int, int, int]] = []
        for mr in range(inner_h):
            for mc in range(inner_w):
                if _marker_mode(mr, mc, inner_h, inner_w) != desired_mode:
                    continue
                if any(abs(int(mr) - pr) + abs(int(mc) - pc) <= 1 for pr, pc, _, _ in marker_points):
                    continue
                top = int(mr - ar)
                left = int(mc - ac)
                if top < 0 or left < 0 or top + ph > inner_h or left + pw > inner_w:
                    continue
                if bool((occupied[top : top + ph, left : left + pw] & pmask).any()):
                    continue
                candidates.append((int(mr), int(mc), int(top), int(left)))

        if not candidates:
            return None

        mr, mc, top, left = random.choice(candidates)
        paste(inner, patch, int(top), int(left), mask=pmask, overwrite=True, clip=False)
        occupied[top : top + ph, left : left + pw] |= pmask
        marker_points.append((int(mr), int(mc), int(type_id), desired_mode))

    return inner, marker_points


def _place_sources(sprite_types: list[dict], source_type_ids: list[int], src_h: int, src_w: int, src_bg: int) -> np.ndarray | None:
    inner = np.full((src_h - 2, src_w - 2), src_bg, dtype=int)
    for type_id in source_type_ids:
        sprite = sprite_types[int(type_id)]
        patch = _render_source_patch(sprite, src_bg)
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

    sprite_specs = random.sample(CANONICAL_SHAPES, n_types)
    source_modes = _choose_source_modes(n_types)

    bodies: list[int] = []
    for i in range(n_types):
        choices = [
            c
            for c in body_pool
            if c not in anchors and c not in {src_bg, dst_bg, src_border, dst_border}
        ]
        if not choices:
            return None
        if i == 0 and forced_body is not None and forced_body in choices:
            bodies.append(forced_body)
        else:
            bodies.append(random.choice(choices))

    sprite_types = [
        {
            "shape_rows": tuple(shape_rows),
            "source_mode": source_mode,
            "body": body,
            "anchor": anchor,
            "h": len(shape_rows),
            "w": len(shape_rows[0]),
        }
        for shape_rows, source_mode, body, anchor in zip(sprite_specs, source_modes, bodies, anchors)
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

    output = _build_frame(dst_h, dst_w, dst_border, dst_bg)
    paste(output[1:-1, 1:-1], out_inner, 0, 0)

    source_inner = _place_sources(sprite_types, source_type_ids, src_h, src_w, src_bg)
    if source_inner is None:
        return None

    source_rect = _build_frame(src_h, src_w, src_border, src_bg)
    paste(source_rect[1:-1, 1:-1], source_inner, 0, 0)

    dest_rect = _build_frame(dst_h, dst_w, dst_border, dst_bg)
    for mr, mc, type_id, _marker_mode_name in marker_points:
        dest_rect[1 + mr, 1 + mc] = sprite_types[int(type_id)]["anchor"]

    try:
        src_pos, dst_pos = _place_rectangles(layout, src_h, src_w, dst_h, dst_w)
    except ValueError:
        return None

    grid = _make_wallpaper(random.choice(["vertical", "horizontal"]), wallpaper)
    paste(grid, source_rect, int(src_pos[0]), int(src_pos[1]))
    paste(grid, dest_rect, int(dst_pos[0]), int(dst_pos[1]))

    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a251c730 combo input after %d attempts" % MAX_GEN_ATTEMPTS)
