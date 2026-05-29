# Variant of edb79dae
# Dimension: D2 — 第二步匹配条件：形状-颜色图例查找 (match_cond)
# Change: 放置后的精灵形状与左上角图例中的形状模式匹配（s2.shape == d2.shape），确定重着色的目标颜色... → 旋转/翻转后形状匹配

import itertools
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis
# a) Concrete behavior:
#    - Step 1 stays the same: each solid placeholder block inside the gray frame selects an outside sprite
#      by matching source color.
#    - Step 2 changes the legend semantics. Instead of a color-only legend pair, the top-left outside area
#      now contains K square "legend tiles". Each tile is fully filled: target color as the tile background,
#      and one common legend-ink color painting a shape silhouette on top.
#    - The legend silhouette is not necessarily in the same orientation as the source sprite. It may be any
#      dihedral transform (rotation/flip) of that sprite mask. So the solver must compare masks up to all
#      8 square symmetries, not just exact equality.
#    - Small trace:
#      original logic: placeholder color 3 -> outside sprite color 3 -> legend pair containing color 3 says
#      recolor to 7.
#      variant logic: placeholder color 3 -> outside sprite color 3 gives an asymmetric 4x4 shape S ->
#      among legend tiles, find the tile whose ink mask is rot90(S) or flip(S) -> recolor to that tile's
#      background color.
#      difference: source color no longer identifies the recolor target; only shape identity under transform
#      does.
# b) Complexity calibration:
#    - Every sample uses 2-5 colors and 2-5 distinct sprite masks, so the shape legend is supported by
#      repeated placements inside the frame.
#    - Masks are generated asymmetric up to dihedral transforms and unique by dihedral signature, preventing
#      ambiguous matches and making the transform visibly matter.
#    - Legend tiles all share one ink color, so color alone is useless for keying the mapping.
# c) Implicit-assumption review for main():
#    - Original legend parser assumed compact two-color overlaps keyed by placeholder/source color. That
#      assumption is invalid now, so legend extraction is replaced with full rectangular non-bg tiles.
#    - Original recolor lookup used src color -> tgt color. That becomes src color -> sprite mask ->
#      dihedral-match legend tile -> tgt color.
#    - Sprite placement still assumes one bbox size per placeholder color and that source sprites keep the
#      full NxN bbox. The generator preserves that by building connected masks with full-row/full-height
#      support.
# d) Placement legality:
#    - Legend tiles are placed deterministically in the top-left cluster with bg gaps between tiles.
#    - The frame is pushed to the bottom-right and the remaining outside area is reserved for random sprite
#      placement with padding, so no overlap occurs between frame, legends, and sprites.
# e) Visual observability:
#    - The new rule is human-visible because the legend tile silhouette often disagrees with the raw sprite
#      orientation. Exact-match-only logic would fail on these inputs.
#    - Using a common legend-ink color avoids hidden color-id logic; the only stable cue is shape.
# f) Visual cleanliness:
#    - Legends are compact, aligned, and separated from sprites by region and padding.
#    - The output frame remains clean because only one recolored sprite is pasted per placeholder bbox.
# g) Color-change inferability:
#    - Target color is directly shown as the filled background of a legend tile, paired with a recognizable
#      silhouette. The number of distinct colors remains small enough for 2-4 ARC examples.
# Assumptions introduced by the new solver and enforced by the generator:
#    - Each sprite mask is unique up to rotation/flip.
#    - Every legend tile uses the same ink color and exactly one non-ink target color.
#    - Every legend mask is a non-identity dihedral transform of its corresponding sprite mask.


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
    """Return unique rot90/flip variants of a square boolean mask."""
    base = []
    for k in range(4):
        rot = np.rot90(mask, k)
        base.append((rot.copy(), k, False))
        base.append((np.fliplr(rot).copy(), k, True))
    out = []
    seen = set()
    for arr, k, flip in base:
        key = arr.tobytes()
        if key in seen:
            continue
        seen.add(key)
        out.append((arr, k, flip))
    return out


def _mask_signature(mask: np.ndarray) -> bytes:
    """Canonical signature up to dihedral symmetry."""
    return min(arr.tobytes() for arr, _k, _flip in _dihedral_variants(mask))


def _same_up_to_dihedral(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        return False
    return any(np.array_equal(arr, b) for arr, _k, _flip in _dihedral_variants(a))


def _legend_match_transform(sprite_mask: np.ndarray, legend_mask: np.ndarray) -> tuple[int, bool] | None:
    """Return the dihedral transform taking sprite_mask to legend_mask, if any."""
    if sprite_mask.shape != legend_mask.shape:
        return None
    for arr, k, flip in _dihedral_variants(sprite_mask):
        if np.array_equal(arr, legend_mask):
            return int(k), bool(flip)
    return None


def _collect_placeholders(
    crop: np.ndarray, bg: int, frame_col: int
) -> tuple[list[tuple[int, tuple[int, int, int, int]]], dict[int, tuple[int, int]], set[int]]:
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
    return placeholders, phw, placeholder_colors


def _extract_legend_tiles(
    g: np.ndarray,
    outside: np.ndarray,
    bg: int,
    frame_col: int,
    valid_sizes: set[tuple[int, int]],
) -> tuple[list[dict], np.ndarray]:
    """
    Legend tiles are full rectangular non-bg outside components whose bbox size matches a placeholder size.
    Each tile contains exactly two colors and no background.
    """
    legend_tiles = []
    legend_mask = np.zeros_like(g, dtype=bool)

    for pts in components_mask((g != bg) & outside, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if (h, w) not in valid_sizes:
            continue
        if len(pts) != h * w:
            continue
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        sub_colors = set(map(int, np.unique(sub)))
        if bg in sub_colors or frame_col in sub_colors or len(sub_colors) != 2:
            continue
        legend_tiles.append(
            {
                "bbox": (r0, c0, r1, c1),
                "sub": sub.copy(),
                "colors": sub_colors,
            }
        )
        legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    assert legend_tiles, "no legend tiles found"
    shared = set.intersection(*(set(tile["colors"]) for tile in legend_tiles))
    assert len(shared) == 1, f"expected one common legend ink color, found {sorted(shared)}"
    legend_ink = int(next(iter(shared)))

    legend_entries = []
    for tile in legend_tiles:
        others = sorted(tile["colors"] - {legend_ink})
        assert len(others) == 1, "legend tile must have exactly one target color"
        shape_mask = tile["sub"] == legend_ink
        assert np.any(shape_mask) and np.any(~shape_mask), "legend tile must contain both shape and target cells"
        legend_entries.append({"shape_mask": shape_mask.copy(), "tgt": int(others[0]), "bbox": tile["bbox"]})

    return legend_entries, legend_mask


def _extract_sprite_masks(
    g: np.ndarray,
    outside: np.ndarray,
    legend_mask: np.ndarray,
    placeholder_colors: set[int],
    phw: dict[int, tuple[int, int]],
) -> dict[int, np.ndarray]:
    sprite_masks: dict[int, np.ndarray] = {}
    for src_col in sorted(placeholder_colors):
        comps = []
        for pts in components_mask((g == src_col) & outside & (~legend_mask), connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            comps.append({"pts": pts, "bbox": (r0, c0, r1, c1)})
        assert comps, f"no source sprite found for color {src_col}"
        want_h, want_w = phw[src_col]
        sized = [c for c in comps if (c["bbox"][2] - c["bbox"][0] + 1, c["bbox"][3] - c["bbox"][1] + 1) == (want_h, want_w)]
        assert len(sized) == 1, f"expected one size-matched sprite for color {src_col}, found {len(sized)}"
        base = sized[0]
        sr0, sc0, sr1, sc1 = base["bbox"]
        mask = np.zeros((sr1 - sr0 + 1, sc1 - sc0 + 1), dtype=bool)
        for r, c in base["pts"]:
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

    placeholders, phw, placeholder_colors = _collect_placeholders(crop, bg, frame_col)

    outside = np.ones(g.shape, dtype=bool)
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False

    legend_entries, legend_mask = _extract_legend_tiles(g, outside, bg, frame_col, set(phw.values()))
    assert len(legend_entries) == len(placeholder_colors), "legend tile count must equal placeholder color count"

    sprite_masks = _extract_sprite_masks(g, outside, legend_mask, placeholder_colors, phw)
    assert set(sprite_masks) == placeholder_colors

    return {
        "grid": g,
        "bg": int(bg),
        "frame_col": int(frame_col),
        "frame_bbox": (fr0, fc0, fr1, fc1),
        "crop": crop,
        "placeholders": placeholders,
        "phw": phw,
        "placeholder_colors": placeholder_colors,
        "legend_entries": legend_entries,
        "sprite_masks": sprite_masks,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the framed placeholders with matched sprites, then recolor them through a shape legend.

    Input structure:
    - A gray rectangular frame encloses solid source-color placeholder blocks on a bg field.
    - Outside the frame there are source-colored sprite templates and, in the top-left area, square
      legend tiles showing shape silhouettes in a common ink color on target-color backgrounds.

    Rule:
    - Match each placeholder color to its same-color source sprite outside the frame and stamp that mask
      into every placeholder bbox.
    - For each stamped mask, find the legend tile whose silhouette has the same shape up to rotation or
      reflection.
    - Recolor the stamped sprite to the legend tile's background color.

    Key insight: Source color still chooses which shape to place, but final recoloring is keyed by shape
    identity under dihedral symmetry, not by color equality inside the legend.
    """
    info = _extract_instance(np.asarray(grid, dtype=int))
    bg = int(info["bg"])
    crop = info["crop"]
    H, W = crop.shape

    color_map: dict[int, int] = {}
    used_legend_ids: set[int] = set()
    for src_col, sprite_mask in info["sprite_masks"].items():
        matches = [i for i, entry in enumerate(info["legend_entries"]) if _same_up_to_dihedral(sprite_mask, entry["shape_mask"])]
        assert len(matches) == 1, f"expected one legend match for source color {src_col}, found {len(matches)}"
        idx = matches[0]
        assert idx not in used_legend_ids, "two source sprites matched the same legend tile"
        color_map[src_col] = int(info["legend_entries"][idx]["tgt"])
        used_legend_ids.add(idx)
    assert set(color_map) == info["placeholder_colors"]

    out = crop.copy()
    out[1:-1, 1:-1] = bg

    for src_col, (pr0, pc0, pr1, pc1) in info["placeholders"]:
        sprite_mask = info["sprite_masks"][src_col]
        tgt_col = color_map[src_col]
        sh, sw = sprite_mask.shape
        assert (pr1 - pr0 + 1, pc1 - pc0 + 1) == (sh, sw), "placeholder size does not match source sprite"
        for dr, dc in np.argwhere(sprite_mask):
            out[int(pr0 + dr), int(pc0 + dc)] = int(tgt_col)

    return out


# --------------- generator (o5x) ---------------

# === CFG: tunables ===
FRAME_COL = 5
SPRITE_N_RANGE = (3, 6)
GRID_ROWS_RANGE = (2, 4)
GRID_COLS_RANGE = (3, 5)
TOTAL_BLOCKS_RANGE = (6, 10)
MAX_EMPTY_SLOTS = 2
NUM_COLORS_RANGE = (2, 5)
SPRITE_FILL_MIN = 0.45
SPRITE_FILL_MAX = 0.82
MIN_FRAME_H = 13
MIN_FRAME_W = 18
MAX_FRAME_DIM = 21
FRAME_MARGIN_SLACK = 4
CANVAS_H_RANGE = (22, 31)
CANVAS_W_RANGE = (23, 31)
PATCH_PADDING = 1
GEN_RETRIES = 400
LEGEND_MAX_PER_ROW = 3
LAYOUT_GAP = 1
MASK_RETRIES = 120


def _legend_tile(mask: np.ndarray, ink: int, tgt: int) -> np.ndarray:
    """Build an NxN legend tile: target-color background with a common ink silhouette."""
    out = np.full(mask.shape, int(tgt), dtype=int)
    out[mask] = int(ink)
    return out


def _sprite_mask(n: int) -> np.ndarray:
    """Generate a connected NxN binary mask with full NxN bbox and moderate fill."""
    lo = max(int(np.ceil(SPRITE_FILL_MIN * n * n)), n + 1)
    hi = min(int(np.floor(SPRITE_FILL_MAX * n * n)), n * n - 1)
    target = random.randint(lo, hi)

    mask = np.zeros((n, n), dtype=bool)
    c = random.randrange(n)
    for r in range(n):
        c = max(0, min(n - 1, c + random.choice([-1, 0, 1])))
        mask[r, c] = True
    mask[random.randrange(n), :] = True

    if random.random() < 0.5:
        mask = mask.T.copy()
    if random.random() < 0.5:
        mask = np.flipud(mask)
    if random.random() < 0.5:
        mask = np.fliplr(mask)

    while int(mask.sum()) < target:
        dilated = np.zeros_like(mask)
        dilated[1:, :] |= mask[:-1, :]
        dilated[:-1, :] |= mask[1:, :]
        dilated[:, 1:] |= mask[:, :-1]
        dilated[:, :-1] |= mask[:, 1:]
        frontier = list(zip(*np.where(dilated & ~mask)))
        r, c = random.choice(frontier)
        mask[r, c] = True
    return mask


def _colored_sprite(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = int(color)
    return out


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


def _choose_empty_slots(
    rows: int, cols: int, empty_count: int
) -> tuple[set[tuple[int, int]], list[tuple[int, int]]]:
    slots = [(r, c) for r in range(rows) for c in range(cols)]
    valid = []
    for combo in itertools.combinations(slots, empty_count):
        empty = set(combo)
        filled = [s for s in slots if s not in empty]
        if len({r for r, _ in filled}) == rows and len({c for _, c in filled}) == cols:
            valid.append(empty)
    empty_slots = random.choice(valid)
    filled_slots = [s for s in slots if s not in empty_slots]
    random.shuffle(filled_slots)
    return empty_slots, filled_slots


def _make_bordered_frame(h: int, w: int, bg: int) -> np.ndarray:
    frame = np.full((h, w), bg, dtype=int)
    frame[0, :] = frame[-1, :] = FRAME_COL
    frame[:, 0] = frame[:, -1] = FRAME_COL
    return frame


def _place_patch(
    grid: np.ndarray, reserve: np.ndarray, patch: np.ndarray, bg: int, *, transparent: bool
) -> bool:
    blocker = np.full(patch.shape, FRAME_COL, dtype=int)
    try:
        r, c = random_free_location(reserve, blocker, transparent=bg, padding=PATCH_PADDING)
    except ValueError:
        return False
    paste(reserve, blocker, r, c)
    paste(grid, patch, r, c, transparent=bg if transparent else None)
    return True


def _legend_positions(k: int, n: int) -> list[tuple[int, int]]:
    per_row = min(LEGEND_MAX_PER_ROW, k)
    positions = []
    for idx in range(k):
        rr = idx // per_row
        cc = idx % per_row
        positions.append((rr * (n + LAYOUT_GAP), cc * (n + LAYOUT_GAP)))
    return positions


def _choose_masks(src_cols: list[int], n: int) -> dict[int, np.ndarray] | None:
    masks: dict[int, np.ndarray] = {}
    seen = set()
    for src in src_cols:
        chosen = None
        for _ in range(MASK_RETRIES):
            mask = _sprite_mask(n)
            variants = _dihedral_variants(mask)
            if len(variants) < 2:
                continue
            sig = _mask_signature(mask)
            if sig in seen:
                continue
            chosen = mask
            seen.add(sig)
            break
        if chosen is None:
            return None
        masks[src] = chosen
    return masks


def _choose_legend_variants(masks: dict[int, np.ndarray]) -> tuple[dict[int, np.ndarray], dict[int, tuple[int, bool]]] | None:
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


def _gen_attempt() -> np.ndarray | None:
    struct = _choose_structure()
    if struct is None:
        return None
    n, rows, cols, empty_count, _total = struct

    bg = random.choice([c for c in range(10) if c != FRAME_COL])
    usable = [c for c in range(10) if c not in (bg, FRAME_COL)]

    k = random.randint(*NUM_COLORS_RANGE)
    src_cols = random.sample(usable, k)
    legend_choices = [c for c in usable if c not in src_cols]
    assert legend_choices
    legend_ink = random.choice(legend_choices)
    tgt_pool = [c for c in usable if c != legend_ink]
    tgt_cols = random.sample(tgt_pool, k)
    color_to_tgt = dict(zip(src_cols, tgt_cols))

    masks = _choose_masks(src_cols, n)
    if masks is None:
        return None
    legend_pick = _choose_legend_variants(masks)
    if legend_pick is None:
        return None
    legend_masks, legend_tfms = legend_pick

    ch, cw = _content_extent(n, rows, cols)
    min_fh = max(MIN_FRAME_H, ch + 4)
    min_fw = max(MIN_FRAME_W, cw + 4)
    max_fh = min(MAX_FRAME_DIM, max(min_fh, ch + 2 + FRAME_MARGIN_SLACK))
    max_fw = min(MAX_FRAME_DIM, max(min_fw, cw + 2 + FRAME_MARGIN_SLACK))
    frame_h = random.randint(min_fh, max_fh)
    frame_w = random.randint(min_fw, max_fw)
    y_margin = frame_h - 2 - ch
    x_margin = frame_w - 2 - cw
    top_margin = random.randint(1, y_margin - 1)
    left_margin = random.randint(1, x_margin - 1)

    _empty_slots, filled_slots = _choose_empty_slots(rows, cols, empty_count)
    cell_colors = {}
    for slot, col in zip(filled_slots[:k], src_cols):
        cell_colors[slot] = col
    for slot in filled_slots[k:]:
        cell_colors[slot] = random.choice(src_cols)

    input_frame = _make_bordered_frame(frame_h, frame_w, bg)
    output_frame = _make_bordered_frame(frame_h, frame_w, bg)
    for (rr, cc), src in cell_colors.items():
        r0 = 1 + top_margin + rr * (n + 1)
        c0 = 1 + left_margin + cc * (n + 1)
        paste(input_frame, np.full((n, n), src, dtype=int), r0, c0)
        paste(output_frame, _colored_sprite(masks[src], color_to_tgt[src], bg), r0, c0, transparent=bg)

    legend_pos = _legend_positions(k, n)
    legend_h = max(r + n for r, _c in legend_pos)
    legend_w = max(c + n for _r, c in legend_pos)

    ext_area = k * (n * n + 8)
    possible_dims = [
        (h, w)
        for h in range(*CANVAS_H_RANGE)
        for w in range(*CANVAS_W_RANGE)
        if h >= frame_h + legend_h + 1
        and w >= max(frame_w, legend_w)
        and h * w - frame_h * frame_w - legend_h * legend_w >= ext_area
    ]
    if not possible_dims:
        return None
    h, w = random.choice(possible_dims)

    grid = np.full((h, w), bg, dtype=int)
    reserve = np.full((h, w), bg, dtype=int)

    fr0 = h - frame_h
    fc0 = w - frame_w
    paste(grid, input_frame, fr0, fc0)
    paste(reserve, np.full((frame_h, frame_w), FRAME_COL, dtype=int), fr0, fc0)

    for (r0, c0), src in zip(legend_pos, src_cols):
        tile = _legend_tile(legend_masks[src], legend_ink, color_to_tgt[src])
        paste(grid, tile, r0, c0)
        paste(reserve, np.full(tile.shape, FRAME_COL, dtype=int), r0, c0)

    sprite_patches = [_colored_sprite(masks[src], src, bg) for src in src_cols]
    random.shuffle(sprite_patches)
    for patch in sprite_patches:
        if not _place_patch(grid, reserve, patch, bg, transparent=True):
            return None

    try:
        if not np.array_equal(main(grid), output_frame):
            return None
    except Exception:
        return None

    metrics = variant_metrics(grid)
    if metrics["non_identity_matches"] != k:
        return None
    if metrics["flip_matches"] + metrics["rotation_only_matches"] != k:
        return None
    if set(metrics["matched_targets"]) != set(tgt_cols):
        return None

    # Keep the transform metadata alive in generation-time reasoning: every legend tile is a true transform.
    assert all((k_rot != 0 or flip) for k_rot, flip in legend_tfms.values())
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        out = _gen_attempt()
        if out is not None:
            return out
    raise RuntimeError("failed to generate valid edb79dae D2 input")


def variant_metrics(grid: np.ndarray) -> dict[str, object]:
    """
    Quantify whether the generated instance really uses the D2 variant.

    Returns counts of legend matches that require non-identity transforms and whether they use flips.
    """
    info = _extract_instance(np.asarray(grid, dtype=int))
    used = set()
    non_identity = 0
    flip_matches = 0
    rotation_only = 0
    matched_targets = []

    for src_col, sprite_mask in info["sprite_masks"].items():
        hits = []
        for idx, entry in enumerate(info["legend_entries"]):
            tfm = _legend_match_transform(sprite_mask, entry["shape_mask"])
            if tfm is not None:
                hits.append((idx, tfm, int(entry["tgt"])))
        assert len(hits) == 1, f"ambiguous metrics match for color {src_col}"
        idx, (k, flip), tgt = hits[0]
        assert idx not in used, "legend tile reused in metrics"
        used.add(idx)
        matched_targets.append(int(tgt))
        if k != 0 or flip:
            non_identity += 1
        if flip:
            flip_matches += 1
        elif k != 0:
            rotation_only += 1

    return {
        "num_shapes": len(info["sprite_masks"]),
        "non_identity_matches": int(non_identity),
        "flip_matches": int(flip_matches),
        "rotation_only_matches": int(rotation_only),
        "matched_targets": tuple(matched_targets),
    }
