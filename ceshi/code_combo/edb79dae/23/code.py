# Combo variant of edb79dae (index 23)
# combo_spec: D2/1+D5/0+D7/1
# Name: 旋转图例匹配+旋转放置+分散图例

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
#    - Base task:
#      1. Placeholder color chooses an outside sprite of the same color.
#      2. A compact local legend gives the output color.
#      3. The sprite is pasted back into the frame without changing orientation.
#    - This combo changes both steps 1->2 and the legend layout:
#      1. The outside sprite still comes from placeholder-color matching.
#      2. When that sprite is pasted inside the frame, it is rotated by quadrant:
#         UL=0, UR=90, LR=180, LL=270 degrees (`np.rot90` counts).
#      3. The recolor legend is no longer a compact block. Each legend entry is split across opposite edges:
#         a source fragment on the top or left edge shows a transformed silhouette in one common legend-ink color,
#         and the opposite bottom or right edge shows the target color bar for the same span.
#      4. The silhouette shown in the source fragment is itself a non-identity rotation/flip of the source sprite,
#         so shape matching must be done up to dihedral symmetry.
#    - Small trace:
#      suppose color 3 selects source sprite S. A placeholder of color 3 in the lower-left quadrant uses rot270(S)
#      when pasted into the frame. On the canvas edge, its legend fragment may display flip(rot90(S)).
#      The solver must reconstruct the split edge legend, match S to that displayed silhouette up to transform,
#      obtain the target color, and then recolor the already-rotated placement.
#      The two rotations live in different stages: one controls placement orientation, one controls legend reading.
#
# b) Complexity calibration:
#    - To keep the two rotation sources visible, every sprite mask is generated with four distinct 90-degree rotations.
#    - To keep D2 unambiguous, masks are unique up to full dihedral signature, so one legend silhouette matches one sprite only.
#    - To keep D5 visible, filled placeholders cover all four quadrants and at least one source color repeats across
#      different quadrants, producing same-shape/different-orientation outputs.
#    - To keep D7 visible, every sample uses both vertical (top+bottom) and horizontal (left+right) split entries.
#
# c) Implicit-assumption audit:
#    - Original legend parsing assumed overlapping local blobs. That is invalid: legend data is now spread over the
#      outer boundary, so the solver must parse boundary runs and dedicated edge bands instead.
#    - Original recolor lookup used source color directly. That also breaks: the edge legend carries no source color
#      identity, only shape identity in a transformed orientation, so the solver must extract sprite masks first and
#      then match them to legend silhouettes up to rotation/flip.
#    - Original placement copied the sprite mask as-is. Now quadrant-based rotation happens before pasting.
#    - New combo-specific assumption: all placeholder bboxes share one square NxN size. D5 needs square rotation and
#      the D7 edge fragments use the same N as their span. The generator enforces this globally.
#    - Another new assumption: no free sprite touches the grid boundary. The generator reserves the whole boundary so
#      every boundary-touching object belongs to the split legend.
#
# d) Placement legality:
#    - The frame is placed away from the outer boundary by at least one legend band.
#    - The full boundary is reserved before random sprite placement, so sprites cannot overlap legends or touch edges.
#    - Top/left source fragments are kept away from corners by `corner_clear = n + 2`, preventing cross-edge overlap.
#    - Rotated placements stay inside their original NxN placeholder bboxes because all sprites are square.
#
# e) Visual observability:
#    - Humans can see two distinct regularities:
#      1. identical source colors yield the same base shape but different in-frame orientations by quadrant;
#      2. edge fragments must be paired across opposite sides, and the displayed legend silhouette often differs from
#         the source sprite's raw orientation.
#    - Neither rule depends on numeric color ids or hidden ordering.
#
# f) Visual cleanliness:
#    - The central frame remains visually separate from the edge legends.
#    - The split legends use thin boundary-aligned structures, leaving the remaining outside area for free sprites.
#    - Each region carries one role only: frame/interior, edge legend, or free sprites.
#
# g) Color inferability and generator guarantees:
#    - Target colors are still given explicitly by opposite-edge bars, so recoloring is structured rather than arbitrary.
#    - The generator guarantees:
#      * every legend silhouette is a non-identity dihedral transform of its sprite;
#      * all four quadrant rotations appear among filled placeholders;
#      * at least one source color repeats across different quadrants;
#      * both vertical and horizontal split legend entries are present;
#      * sprite masks are unique up to dihedral symmetry and visibly change under rotation.


FRAME_COL = 5
SPRITE_N_RANGE = (3, 6)
GRID_ROWS_RANGE = (2, 4)
GRID_COLS_RANGE = (3, 5)
TOTAL_BLOCKS_RANGE = (6, 10)
MAX_EMPTY_SLOTS = 2
NUM_COLORS_RANGE = (2, 5)
SPRITE_FILL_MIN = 0.45
SPRITE_FILL_MAX = 0.8
MIN_FRAME_H = 13
MIN_FRAME_W = 18
MAX_FRAME_DIM = 21
FRAME_MARGIN_SLACK = 4
CANVAS_H_RANGE = (28, 41)
CANVAS_W_RANGE = (28, 41)
PATCH_PADDING = 1
GEN_RETRIES = 500
LAYOUT_RETRIES = 100
MASK_RETRIES = 200


def _find_frame(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    """
    Find a rectangular border component:
      - all bbox-border cells are this color,
      - bbox-interior is not entirely this color.
    Return (frame_color, (r0,c0,r1,c1)).
    """
    candidates = []
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    for col in colors:
        for pts in components_mask(g == col, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            border = np.zeros(sub.shape, dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.all(sub[border] == col):
                continue
            if np.all(sub[~border] == col):
                continue
            candidates.append((col, (r0, c0, r1, c1)))
    assert len(candidates) == 1, f"expected exactly one rectangular frame component, found {len(candidates)}"
    frame_col, bbox = candidates[0]
    return int(frame_col), bbox


def _dihedral_variants(mask: np.ndarray) -> list[tuple[np.ndarray, int, bool]]:
    out = []
    seen = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        for arr, flip in ((rot, False), (np.fliplr(rot), True)):
            key = arr.tobytes()
            if key in seen:
                continue
            seen.add(key)
            out.append((arr.copy(), int(k), bool(flip)))
    return out


def _mask_signature(mask: np.ndarray) -> bytes:
    return min(arr.tobytes() for arr, _k, _flip in _dihedral_variants(mask))


def _legend_match_transform(sprite_mask: np.ndarray, legend_mask: np.ndarray) -> tuple[int, bool] | None:
    if sprite_mask.shape != legend_mask.shape:
        return None
    for arr, k, flip in _dihedral_variants(sprite_mask):
        if np.array_equal(arr, legend_mask):
            return int(k), bool(flip)
    return None


def _rotation_k_for_bbox(bbox: tuple[int, int, int, int], frame_h: int, frame_w: int) -> int:
    r0, c0, r1, c1 = bbox
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    mid_r = (frame_h - 1) / 2.0
    mid_c = (frame_w - 1) / 2.0
    assert center_r != mid_r and center_c != mid_c, "placeholder center lies on a frame midline"
    if center_r < mid_r and center_c < mid_c:
        return 0
    if center_r < mid_r and center_c > mid_c:
        return 1
    if center_r > mid_r and center_c > mid_c:
        return 2
    return 3


def _is_rectangular_border_mask(mask: np.ndarray) -> bool:
    if mask.ndim != 2 or min(mask.shape) < 3:
        return False
    border = np.zeros(mask.shape, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.all(mask[border]) and not np.all(mask[~border]))


def _collect_placeholders(
    crop: np.ndarray, bg: int, frame_col: int
) -> tuple[list[tuple[int, tuple[int, int, int, int]]], dict[int, tuple[int, int]], set[int], int]:
    H, W = crop.shape
    inner_mask = np.zeros((H, W), dtype=bool)
    inner_mask[1:-1, 1:-1] = True
    placeholder_colors = set(map(int, np.unique(crop[inner_mask]))) - {bg, frame_col}
    assert placeholder_colors, "no placeholder colors found inside frame"

    placeholders: list[tuple[int, tuple[int, int, int, int]]] = []
    for col in sorted(placeholder_colors):
        for pts in components_mask((crop == col) & inner_mask, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            placeholders.append((col, (r0, c0, r1, c1)))

    phw: dict[int, tuple[int, int]] = {}
    for col in placeholder_colors:
        dims = [(b[2] - b[0] + 1, b[3] - b[1] + 1) for c, b in placeholders if c == col]
        assert dims, f"no placeholder component found for color {col}"
        unique_dims = sorted(set(dims))
        assert len(unique_dims) == 1, f"ambiguous placeholder sizes for color {col}: {unique_dims}"
        phw[col] = unique_dims[0]

    all_dims = sorted(set(phw.values()))
    assert len(all_dims) == 1, f"combo variant expects one global placeholder size, found {all_dims}"
    n_h, n_w = all_dims[0]
    assert n_h == n_w, f"combo variant requires square placeholders, got {(n_h, n_w)}"
    return placeholders, phw, placeholder_colors, int(n_h)


def _edge_runs(arr: np.ndarray, blocked: set[int]) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    vals = [int(x) for x in np.asarray(arr).tolist()]
    i = 0
    while i < len(vals):
        col = vals[i]
        if col in blocked:
            i += 1
            continue
        j = i
        while j + 1 < len(vals) and vals[j + 1] == col:
            j += 1
        runs.append((i, j, col))
        i = j + 1
    return runs


def _extract_edge_legends(
    g: np.ndarray, bg: int, frame_col: int, n: int
) -> tuple[list[dict], np.ndarray, dict[str, int]]:
    blocked = {bg, frame_col}
    top_runs = _edge_runs(g[0, :], blocked)
    bottom_runs = _edge_runs(g[-1, :], blocked)
    left_runs = _edge_runs(g[:, 0], blocked)
    right_runs = _edge_runs(g[:, -1], blocked)

    assert top_runs or left_runs, "no source-side legend fragments found"
    source_colors = {col for *_span, col in top_runs + left_runs}
    assert len(source_colors) == 1, f"expected one common legend ink color, found {sorted(source_colors)}"
    legend_ink = int(next(iter(source_colors)))

    bottom_by_span = {(c0, c1): int(col) for c0, c1, col in bottom_runs}
    right_by_span = {(r0, r1): int(col) for r0, r1, col in right_runs}
    assert len(bottom_by_span) == len(bottom_runs), "duplicate bottom legend spans"
    assert len(right_by_span) == len(right_runs), "duplicate right legend spans"

    legend_entries: list[dict] = []
    legend_mask = np.zeros(g.shape, dtype=bool)

    for c0, c1, col in top_runs:
        assert int(col) == legend_ink, "top legend fragment must use common legend ink"
        assert c1 - c0 + 1 == n, f"top legend width mismatch: expected {n}, got {c1 - c0 + 1}"
        assert (c0, c1) in bottom_by_span, f"missing bottom legend pair for top span {(c0, c1)}"
        shape_mask = g[1 : n + 1, c0 : c1 + 1] == legend_ink
        assert np.any(shape_mask), "top legend silhouette is empty"
        legend_entries.append({"shape_mask": shape_mask.copy(), "tgt": bottom_by_span[(c0, c1)], "axis": "vertical"})
        legend_mask[0 : n + 1, c0 : c1 + 1] = True
        legend_mask[g.shape[0] - 1 : g.shape[0], c0 : c1 + 1] = True

    for r0, r1, col in left_runs:
        assert int(col) == legend_ink, "left legend fragment must use common legend ink"
        assert r1 - r0 + 1 == n, f"left legend height mismatch: expected {n}, got {r1 - r0 + 1}"
        assert (r0, r1) in right_by_span, f"missing right legend pair for left span {(r0, r1)}"
        shape_mask = g[r0 : r1 + 1, 1 : n + 1] == legend_ink
        assert np.any(shape_mask), "left legend silhouette is empty"
        legend_entries.append({"shape_mask": shape_mask.copy(), "tgt": right_by_span[(r0, r1)], "axis": "horizontal"})
        legend_mask[r0 : r1 + 1, 0 : n + 1] = True
        legend_mask[r0 : r1 + 1, g.shape[1] - 1 : g.shape[1]] = True

    counts = {
        "vertical_entries": len(top_runs),
        "horizontal_entries": len(left_runs),
        "legend_ink": legend_ink,
    }
    return legend_entries, legend_mask, counts


def _extract_sprite_masks(
    g: np.ndarray,
    outside: np.ndarray,
    legend_mask: np.ndarray,
    placeholder_colors: set[int],
    phw: dict[int, tuple[int, int]],
) -> dict[int, np.ndarray]:
    sprite_masks: dict[int, np.ndarray] = {}
    usable = outside & (~legend_mask)

    for src_col in sorted(placeholder_colors):
        comps = []
        for pts in components_mask((g == src_col) & usable, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            comps.append({"pts": [(int(r), int(c)) for r, c in pts], "bbox": (r0, c0, r1, c1)})
        assert comps, f"no source sprite found for color {src_col}"

        want_h, want_w = phw[src_col]
        sized = [
            c
            for c in comps
            if (c["bbox"][2] - c["bbox"][0] + 1, c["bbox"][3] - c["bbox"][1] + 1) == (want_h, want_w)
        ]
        assert len(sized) == 1, f"expected one size-matched sprite for color {src_col}, found {len(sized)}"
        base = sized[0]
        sr0, sc0, sr1, sc1 = base["bbox"]
        sh = sr1 - sr0 + 1
        sw = sc1 - sc0 + 1
        assert sh == sw, f"combo variant requires square sprites, got {(sh, sw)} for color {src_col}"

        pts_set = set(base["pts"])
        for comp in comps:
            if comp is base:
                continue
            r0, c0, r1, c1 = comp["bbox"]
            assert sr0 <= r0 and r1 <= sr1 and sc0 <= c0 and c1 <= sc1, (
                f"extra sprite component for color {src_col} lies outside its main bbox"
            )
            pts_set.update(comp["pts"])

        mask = np.zeros((sh, sw), dtype=bool)
        for r, c in pts_set:
            mask[int(r - sr0), int(c - sc0)] = True
        sprite_masks[src_col] = mask

    return sprite_masks


def _extract_instance(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    frame_col, (fr0, fc0, fr1, fc1) = _find_frame(g, bg)

    crop = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    H, W = crop.shape
    assert H >= 3 and W >= 3, f"frame too small for interior: {(H, W)}"

    placeholders, phw, placeholder_colors, n = _collect_placeholders(crop, bg, frame_col)
    legend_entries, legend_mask, legend_counts = _extract_edge_legends(g, bg, frame_col, n)
    assert len(legend_entries) == len(placeholder_colors), (
        f"legend entry count {len(legend_entries)} does not match source-color count {len(placeholder_colors)}"
    )

    outside = np.ones(g.shape, dtype=bool)
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False
    sprite_masks = _extract_sprite_masks(g, outside, legend_mask, placeholder_colors, phw)

    return {
        "grid": g,
        "bg": int(bg),
        "frame_col": int(frame_col),
        "crop": crop,
        "frame_bbox": (fr0, fc0, fr1, fc1),
        "placeholders": placeholders,
        "phw": phw,
        "placeholder_colors": placeholder_colors,
        "tile_n": n,
        "legend_entries": legend_entries,
        "legend_mask": legend_mask,
        "legend_counts": legend_counts,
        "sprite_masks": sprite_masks,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the framed output by combining split edge legends, shape matching up to transform, and quadrant-based placement rotation.

    Input structure:
    - A gray rectangular frame encloses solid placeholder blocks of one shared square size.
    - Outside the frame there is one source-colored sprite template per placeholder color.
    - Along the outer boundary, legend entries are scattered across different edges: the top or left fragment shows a silhouette in one common legend-ink color, while the opposite bottom or right fragment supplies the target color for the same span.

    Rule:
    - Match each placeholder color to its same-color sprite outside the frame.
    - Reconstruct the split edge legend, then match each source sprite to exactly one legend silhouette up to rotation or reflection to determine the target color.
    - For every placeholder block, rotate its matched sprite by quadrant inside the frame (UL 0, UR 90, LR 180, LL 270), stamp it into the block, and recolor it to the target color from the matched edge legend.

    Key insight: The solver must keep two independent orientation states straight: the sprite's placement orientation comes from in-frame position, while the legend's displayed orientation is a separate transformed version that only serves to identify the recolor mapping.
    """
    info = _extract_instance(np.asarray(grid, dtype=int))
    bg = int(info["bg"])
    crop = info["crop"]
    H, W = crop.shape

    color_map: dict[int, int] = {}
    used_legend_ids: set[int] = set()
    for src_col, sprite_mask in info["sprite_masks"].items():
        hits = []
        for idx, entry in enumerate(info["legend_entries"]):
            tfm = _legend_match_transform(sprite_mask, entry["shape_mask"])
            if tfm is not None:
                hits.append((idx, int(entry["tgt"])))
        assert len(hits) == 1, f"expected one legend match for source color {src_col}, found {len(hits)}"
        idx, tgt = hits[0]
        assert idx not in used_legend_ids, "two source sprites matched the same legend entry"
        used_legend_ids.add(idx)
        color_map[src_col] = int(tgt)
    assert set(color_map) == info["placeholder_colors"]

    out = crop.copy()
    out[1:-1, 1:-1] = bg

    for src_col, bbox in info["placeholders"]:
        pr0, pc0, pr1, pc1 = bbox
        sprite_mask = info["sprite_masks"][src_col]
        sh, sw = sprite_mask.shape
        assert (pr1 - pr0 + 1, pc1 - pc0 + 1) == (sh, sw), "placeholder size does not match sprite size"
        krot = _rotation_k_for_bbox(bbox, H, W)
        rot_mask = np.rot90(sprite_mask, krot)
        sprite = np.full((sh, sw), bg, dtype=int)
        sprite[rot_mask] = int(color_map[src_col])
        paste(out, sprite, pr0, pc0, transparent=bg)

    return out


def _rot_signature(mask: np.ndarray) -> tuple[bytes, ...]:
    return tuple(np.rot90(mask, k).tobytes() for k in range(4))


def _has_four_distinct_rotations(mask: np.ndarray) -> bool:
    return len(set(_rot_signature(mask))) == 4


def _manhattan_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = a
    br, bc = b
    pts = [(int(r), int(c))]
    while (r, c) != (br, bc):
        moves = []
        if r < br:
            moves.append((r + 1, c))
        elif r > br:
            moves.append((r - 1, c))
        if c < bc:
            moves.append((r, c + 1))
        elif c > bc:
            moves.append((r, c - 1))
        r, c = random.choice(moves)
        pts.append((int(r), int(c)))
    return pts


def _sprite_mask(n: int) -> np.ndarray:
    lo = max(int(np.ceil(SPRITE_FILL_MIN * n * n)), 2 * n + 1)
    hi = min(int(np.floor(SPRITE_FILL_MAX * n * n)), n * n - 1)
    for _ in range(MASK_RETRIES):
        target = random.randint(lo, hi)
        center = (random.randrange(n), random.randrange(n))
        anchors = {
            (0, random.randrange(n)),
            (n - 1, random.randrange(n)),
            (random.randrange(n), 0),
            (random.randrange(n), n - 1),
        }
        mask = np.zeros((n, n), dtype=bool)
        for anchor in anchors:
            for r, c in _manhattan_path(center, anchor):
                mask[r, c] = True

        while int(mask.sum()) < target:
            frontier = []
            for r, c in zip(*np.where(mask)):
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < n and 0 <= cc < n and not mask[rr, cc]:
                        frontier.append((int(rr), int(cc)))
            if not frontier:
                break
            rr, cc = random.choice(frontier)
            mask[rr, cc] = True

        if int(mask.sum()) >= target and np.any(~mask):
            if _has_four_distinct_rotations(mask) and not _is_rectangular_border_mask(mask):
                return mask
    raise RuntimeError(f"failed to build a suitable sprite mask for n={n}")


def _colored_sprite(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = int(color)
    return out


def _top_source_patch(mask: np.ndarray, ink: int, bg: int) -> np.ndarray:
    n = mask.shape[0]
    patch = np.full((n + 1, n), bg, dtype=int)
    patch[0, :] = int(ink)
    patch[1:, :][mask] = int(ink)
    return patch


def _left_source_patch(mask: np.ndarray, ink: int, bg: int) -> np.ndarray:
    n = mask.shape[0]
    patch = np.full((n, n + 1), bg, dtype=int)
    patch[:, 0] = int(ink)
    patch[:, 1:][mask] = int(ink)
    return patch


def _target_bar(length: int, color: int, *, vertical: bool) -> np.ndarray:
    if vertical:
        return np.full((1, length), int(color), dtype=int)
    return np.full((length, 1), int(color), dtype=int)


def _content_extent(n: int, grid_rows: int, grid_cols: int) -> tuple[int, int]:
    return (grid_rows * n + grid_rows - 1, grid_cols * n + grid_cols - 1)


def _choose_structure() -> tuple[int, int, int, int, int] | None:
    total = random.randint(*TOTAL_BLOCKS_RANGE)
    options = []
    for n in range(*SPRITE_N_RANGE):
        for rows in range(*GRID_ROWS_RANGE):
            for cols in range(*GRID_COLS_RANGE):
                ch, cw = _content_extent(n, rows, cols)
                base_fh = ch + 4
                base_fw = cw + 4
                slots = rows * cols
                empty = slots - total
                if base_fh <= MAX_FRAME_DIM and base_fw <= MAX_FRAME_DIM and 0 <= empty <= MAX_EMPTY_SLOTS:
                    options.append((n, rows, cols, empty, total))
    return random.choice(options) if options else None


def _slot_bbox(slot: tuple[int, int], n: int, top_margin: int, left_margin: int) -> tuple[int, int, int, int]:
    rr, cc = slot
    r0 = 1 + top_margin + rr * (n + 1)
    c0 = 1 + left_margin + cc * (n + 1)
    return (r0, c0, r0 + n - 1, c0 + n - 1)


def _slot_rotations(
    rows: int, cols: int, n: int, frame_h: int, frame_w: int, top_margin: int, left_margin: int
) -> dict[tuple[int, int], int] | None:
    slot_rot: dict[tuple[int, int], int] = {}
    for rr in range(rows):
        for cc in range(cols):
            bbox = _slot_bbox((rr, cc), n, top_margin, left_margin)
            try:
                slot_rot[(rr, cc)] = _rotation_k_for_bbox(bbox, frame_h, frame_w)
            except AssertionError:
                return None
    if set(slot_rot.values()) != {0, 1, 2, 3}:
        return None
    return slot_rot


def _choose_frame_layout(
    n: int, rows: int, cols: int
) -> tuple[int, int, int, int, dict[tuple[int, int], int]] | None:
    ch, cw = _content_extent(n, rows, cols)
    min_fh = max(MIN_FRAME_H, ch + 4)
    min_fw = max(MIN_FRAME_W, cw + 4)
    max_fh = min(MAX_FRAME_DIM, max(min_fh, ch + 2 + FRAME_MARGIN_SLACK))
    max_fw = min(MAX_FRAME_DIM, max(min_fw, cw + 2 + FRAME_MARGIN_SLACK))
    for _ in range(LAYOUT_RETRIES):
        frame_h = random.randint(min_fh, max_fh)
        frame_w = random.randint(min_fw, max_fw)
        y_margin = frame_h - 2 - ch
        x_margin = frame_w - 2 - cw
        top_margin = random.randint(1, y_margin - 1)
        left_margin = random.randint(1, x_margin - 1)
        slot_rot = _slot_rotations(rows, cols, n, frame_h, frame_w, top_margin, left_margin)
        if slot_rot is not None:
            return frame_h, frame_w, top_margin, left_margin, slot_rot
    return None


def _choose_empty_slots(
    rows: int, cols: int, empty_count: int, slot_rot: dict[tuple[int, int], int]
) -> tuple[set[tuple[int, int]], list[tuple[int, int]]] | None:
    slots = [(r, c) for r in range(rows) for c in range(cols)]
    valid = []
    for combo in itertools.combinations(slots, empty_count):
        empty = set(combo)
        filled = [s for s in slots if s not in empty]
        if len({r for r, _ in filled}) != rows:
            continue
        if len({c for _, c in filled}) != cols:
            continue
        if {slot_rot[s] for s in filled} != {0, 1, 2, 3}:
            continue
        valid.append(empty)
    if not valid:
        return None
    empty_slots = random.choice(valid)
    filled_slots = [s for s in slots if s not in empty_slots]
    random.shuffle(filled_slots)
    return empty_slots, filled_slots


def _assign_cell_colors(
    filled_slots: list[tuple[int, int]], src_cols: list[int], slot_rot: dict[tuple[int, int], int]
) -> dict[tuple[int, int], int] | None:
    cross_rot_pairs = [
        (a, b)
        for i, a in enumerate(filled_slots)
        for b in filled_slots[i + 1 :]
        if slot_rot[a] != slot_rot[b]
    ]
    if not cross_rot_pairs:
        return None

    anchor_a, anchor_b = random.choice(cross_rot_pairs)
    anchor_col = random.choice(src_cols)
    cell_colors = {anchor_a: anchor_col, anchor_b: anchor_col}

    remaining_slots = [s for s in filled_slots if s not in (anchor_a, anchor_b)]
    random.shuffle(remaining_slots)
    other_cols = [c for c in src_cols if c != anchor_col]
    if len(remaining_slots) < len(other_cols):
        return None

    for col, slot in zip(other_cols, remaining_slots):
        cell_colors[slot] = col
    for slot in remaining_slots[len(other_cols) :]:
        cell_colors[slot] = random.choice(src_cols)

    assert set(cell_colors) == set(filled_slots)
    assert set(cell_colors.values()) == set(src_cols)
    repeated_rots = {slot_rot[s] for s, col in cell_colors.items() if col == anchor_col}
    assert len(repeated_rots) >= 2
    return cell_colors


def _make_bordered_frame(h: int, w: int, bg: int) -> np.ndarray:
    frame = np.full((h, w), bg, dtype=int)
    frame[0, :] = frame[-1, :] = FRAME_COL
    frame[:, 0] = frame[:, -1] = FRAME_COL
    return frame


def _slot_starts(limit: int, span: int, count: int, margin: int) -> list[int] | None:
    if count == 0:
        return []
    need = count * span + (count - 1)
    extra = limit - 2 * margin - need
    if extra < 0:
        return None
    offset = margin + random.randint(0, extra)
    return [offset + i * (span + 1) for i in range(count)]


def _place_patch(grid: np.ndarray, reserve: np.ndarray, patch: np.ndarray, bg: int) -> bool:
    blocker = np.full(patch.shape, FRAME_COL, dtype=int)
    try:
        r, c = random_free_location(reserve, blocker, transparent=bg, padding=PATCH_PADDING)
    except ValueError:
        return False
    paste(reserve, blocker, r, c)
    paste(grid, patch, r, c, transparent=bg)
    return True


def _legend_entry_layout(
    src_cols: list[int], n: int, h: int, w: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    corner_clear = n + 2
    if len(src_cols) < 2:
        return None
    vcount = random.randint(1, len(src_cols) - 1)
    hcount = len(src_cols) - vcount
    top_starts = _slot_starts(w, n, vcount, corner_clear)
    left_starts = _slot_starts(h, n, hcount, corner_clear)
    if top_starts is None or left_starts is None:
        return None
    order = src_cols[:]
    random.shuffle(order)
    vertical = [(order[i], top_starts[i]) for i in range(vcount)]
    horizontal = [(order[vcount + i], left_starts[i]) for i in range(hcount)]
    return vertical, horizontal


def _choose_masks(src_cols: list[int], n: int) -> dict[int, np.ndarray] | None:
    masks: dict[int, np.ndarray] = {}
    seen = set()
    for src in src_cols:
        chosen = None
        for _ in range(MASK_RETRIES):
            mask = _sprite_mask(n)
            sig = _mask_signature(mask)
            if sig in seen:
                continue
            variants = _dihedral_variants(mask)
            if len(variants) < 2:
                continue
            chosen = mask
            seen.add(sig)
            break
        if chosen is None:
            return None
        masks[src] = chosen
    return masks


def _choose_legend_variants(
    masks: dict[int, np.ndarray]
) -> tuple[dict[int, np.ndarray], dict[int, tuple[int, bool]]] | None:
    legend_masks: dict[int, np.ndarray] = {}
    legend_tfms: dict[int, tuple[int, bool]] = {}
    for src, mask in masks.items():
        options = [(arr, k, flip) for arr, k, flip in _dihedral_variants(mask) if not np.array_equal(arr, mask)]
        if not options:
            return None
        arr, k, flip = random.choice(options)
        legend_masks[src] = arr.copy()
        legend_tfms[src] = (int(k), bool(flip))
    return legend_masks, legend_tfms


def _partial_output_without_rotation(info: dict, color_map: dict[int, int]) -> np.ndarray:
    bg = int(info["bg"])
    crop = info["crop"]
    out = crop.copy()
    out[1:-1, 1:-1] = bg
    for src_col, bbox in info["placeholders"]:
        pr0, pc0, pr1, pc1 = bbox
        sprite_mask = info["sprite_masks"][src_col]
        sh, sw = sprite_mask.shape
        assert (pr1 - pr0 + 1, pc1 - pc0 + 1) == (sh, sw)
        sprite = np.full((sh, sw), bg, dtype=int)
        sprite[sprite_mask] = int(color_map[src_col])
        paste(out, sprite, pr0, pc0, transparent=bg)
    return out


def variant_metrics(grid: np.ndarray) -> dict[str, object]:
    info = _extract_instance(np.asarray(grid, dtype=int))
    crop = info["crop"]
    H, W = crop.shape

    color_map: dict[int, int] = {}
    legend_transform_by_src: dict[int, tuple[int, bool]] = {}
    used = set()
    for src_col, sprite_mask in info["sprite_masks"].items():
        hits = []
        for idx, entry in enumerate(info["legend_entries"]):
            tfm = _legend_match_transform(sprite_mask, entry["shape_mask"])
            if tfm is not None:
                hits.append((idx, tfm, int(entry["tgt"])))
        assert len(hits) == 1, f"ambiguous legend match for color {src_col}"
        idx, tfm, tgt = hits[0]
        assert idx not in used, "legend entry reused across sources"
        used.add(idx)
        color_map[src_col] = int(tgt)
        legend_transform_by_src[src_col] = tfm

    rotation_hist = {0: 0, 1: 0, 2: 0, 3: 0}
    changed_vs_unrotated = 0
    combined_nontrivial = 0
    for src_col, bbox in info["placeholders"]:
        krot = _rotation_k_for_bbox(bbox, H, W)
        rotation_hist[krot] += 1
        sprite_mask = info["sprite_masks"][src_col]
        if krot != 0 and not np.array_equal(np.rot90(sprite_mask, krot), sprite_mask):
            changed_vs_unrotated += 1
        k_leg, flip_leg = legend_transform_by_src[src_col]
        if (k_leg != 0 or flip_leg) and krot != 0:
            combined_nontrivial += 1

    no_rot_out = _partial_output_without_rotation(info, color_map)
    full_out = main(info["grid"])
    repeated_cross_quad = 0
    for src_col in info["placeholder_colors"]:
        seen_rots = {
            _rotation_k_for_bbox(bbox, H, W)
            for col, bbox in info["placeholders"]
            if col == src_col
        }
        if len(seen_rots) >= 2:
            repeated_cross_quad += 1

    return {
        "num_shapes": len(info["sprite_masks"]),
        "vertical_entries": int(info["legend_counts"]["vertical_entries"]),
        "horizontal_entries": int(info["legend_counts"]["horizontal_entries"]),
        "legend_non_identity": int(sum((k != 0 or flip) for k, flip in legend_transform_by_src.values())),
        "legend_flip_count": int(sum(flip for _k, flip in legend_transform_by_src.values())),
        "rotation_hist": rotation_hist,
        "changed_vs_unrotated": int(changed_vs_unrotated),
        "combined_nontrivial": int(combined_nontrivial),
        "repeated_cross_quad": int(repeated_cross_quad),
        "output_differs_without_rotation": bool(not np.array_equal(no_rot_out, full_out)),
    }


def _gen_attempt() -> np.ndarray | None:
    struct = _choose_structure()
    if struct is None:
        return None
    n, rows, cols, empty_count, total = struct

    layout = _choose_frame_layout(n, rows, cols)
    if layout is None:
        return None
    frame_h, frame_w, top_margin, left_margin, slot_rot = layout

    empty_pick = _choose_empty_slots(rows, cols, empty_count, slot_rot)
    if empty_pick is None:
        return None
    _, filled_slots = empty_pick

    bg = random.choice([c for c in range(10) if c != FRAME_COL])
    usable = [c for c in range(10) if c not in (bg, FRAME_COL)]
    k_hi = min(NUM_COLORS_RANGE[1], total - 1)
    if k_hi < NUM_COLORS_RANGE[0]:
        return None
    k = random.randint(NUM_COLORS_RANGE[0], k_hi)
    src_cols = random.sample(usable, k)

    masks = _choose_masks(src_cols, n)
    if masks is None:
        return None
    legend_pick = _choose_legend_variants(masks)
    if legend_pick is None:
        return None
    legend_masks, legend_tfms = legend_pick

    legend_ink_choices = [c for c in usable if c not in src_cols]
    if not legend_ink_choices:
        return None
    legend_ink = random.choice(legend_ink_choices)
    tgt_pool = [c for c in usable if c != legend_ink]
    if len(set(tgt_pool)) < k:
        return None
    tgt_cols = random.sample(tgt_pool, k)
    color_to_tgt = dict(zip(src_cols, tgt_cols))

    cell_colors = _assign_cell_colors(filled_slots, src_cols, slot_rot)
    if cell_colors is None:
        return None

    input_frame = _make_bordered_frame(frame_h, frame_w, bg)
    output_frame = _make_bordered_frame(frame_h, frame_w, bg)
    for slot, src in cell_colors.items():
        r0, c0, _r1, _c1 = _slot_bbox(slot, n, top_margin, left_margin)
        paste(input_frame, np.full((n, n), src, dtype=int), r0, c0)
        rot_mask = np.rot90(masks[src], slot_rot[slot])
        paste(output_frame, _colored_sprite(rot_mask, color_to_tgt[src], bg), r0, c0, transparent=bg)

    edge_band = n + 1
    corner_clear = n + 2
    min_canvas_h = max(CANVAS_H_RANGE[0], frame_h + 2 * edge_band, 2 * corner_clear + n)
    min_canvas_w = max(CANVAS_W_RANGE[0], frame_w + 2 * edge_band, 2 * corner_clear + n)
    ext_area = k * (2 * n * n + 8 * n + 10)
    possible_dims = [
        (h, w)
        for h in range(min_canvas_h, CANVAS_H_RANGE[1])
        for w in range(min_canvas_w, CANVAS_W_RANGE[1])
        if h * w - frame_h * frame_w >= ext_area
    ]
    if not possible_dims:
        return None
    h, w = random.choice(possible_dims)

    legend_layout = _legend_entry_layout(src_cols, n, h, w)
    if legend_layout is None:
        return None
    vertical_entries, horizontal_entries = legend_layout
    if not vertical_entries or not horizontal_entries:
        return None

    grid = np.full((h, w), bg, dtype=int)
    reserve = np.full((h, w), bg, dtype=int)
    reserve[0, :] = FRAME_COL
    reserve[-1, :] = FRAME_COL
    reserve[:, 0] = FRAME_COL
    reserve[:, -1] = FRAME_COL

    fr_lo = edge_band
    fr_hi = h - frame_h - edge_band
    fc_lo = edge_band
    fc_hi = w - frame_w - edge_band
    if fr_lo > fr_hi or fc_lo > fc_hi:
        return None
    fr0 = random.randint(fr_lo, fr_hi)
    fc0 = random.randint(fc_lo, fc_hi)
    paste(grid, input_frame, fr0, fc0)
    paste(reserve, np.full((frame_h, frame_w), FRAME_COL, dtype=int), fr0, fc0)

    for src, c0 in vertical_entries:
        top_patch = _top_source_patch(legend_masks[src], legend_ink, bg)
        bottom_patch = _target_bar(n, color_to_tgt[src], vertical=True)
        paste(grid, top_patch, 0, c0, transparent=bg)
        paste(grid, bottom_patch, h - 1, c0)
        paste(reserve, np.full(top_patch.shape, FRAME_COL, dtype=int), 0, c0)
        paste(reserve, np.full(bottom_patch.shape, FRAME_COL, dtype=int), h - 1, c0)

    for src, r0 in horizontal_entries:
        left_patch = _left_source_patch(legend_masks[src], legend_ink, bg)
        right_patch = _target_bar(n, color_to_tgt[src], vertical=False)
        paste(grid, left_patch, r0, 0, transparent=bg)
        paste(grid, right_patch, r0, w - 1)
        paste(reserve, np.full(left_patch.shape, FRAME_COL, dtype=int), r0, 0)
        paste(reserve, np.full(right_patch.shape, FRAME_COL, dtype=int), r0, w - 1)

    sprite_patches = [(_colored_sprite(masks[src], src, bg), src) for src in src_cols]
    random.shuffle(sprite_patches)
    for patch, _src in sprite_patches:
        if not _place_patch(grid, reserve, patch, bg):
            return None

    try:
        if not np.array_equal(main(grid), output_frame):
            return None
        metrics = variant_metrics(grid)
    except Exception:
        return None

    if metrics["vertical_entries"] <= 0 or metrics["horizontal_entries"] <= 0:
        return None
    if metrics["legend_non_identity"] != k:
        return None
    if metrics["changed_vs_unrotated"] <= 0:
        return None
    if not metrics["output_differs_without_rotation"]:
        return None
    if metrics["combined_nontrivial"] <= 0:
        return None
    if metrics["repeated_cross_quad"] <= 0:
        return None

    assert all((krot != 0 or flip) for krot, flip in legend_tfms.values())
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        out = _gen_attempt()
        if out is not None:
            return out
    raise RuntimeError("failed to generate valid edb79dae combo input")
