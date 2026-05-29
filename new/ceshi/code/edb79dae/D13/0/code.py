# Variant of edb79dae
# Dimension: D13 — 图例条目与精灵形状的覆盖关系 (meta)
# Change: 图例中的每种形状模式对应一种输出颜色，所有放置的精灵都能在图例中找到匹配的形状条目... → 图例包含干扰条目

import itertools
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the D13 variant
# ==================================
# a) Concrete behavior:
#    - Keep the original large gray frame with solid color placeholders inside.
#    - Keep one outside sprite template per placeholder color; each template still has the
#      same N x N bbox as the matching placeholders and is rendered in the placeholder color.
#    - Replace the old 3x4 color-pair legend with several small gray-bordered legend boxes.
#      Each legend box contains a single colored N x N shape on bg. The shape pattern is the
#      key, and the glyph color is the recolor target.
#    - For every real sprite shape there is exactly one matching legend box. In addition,
#      there are 1+ extra legend boxes whose inner shapes do not match any real sprite.
#    - Example trace: if a blue placeholder uses a zig-zag sprite, solver first recovers the
#      blue zig-zag from outside templates, then scans boxed legend entries. A red boxed
#      zig-zag is valid and recolors all blue placements to red; a yellow boxed T-shape that
#      matches no sprite is ignored. Original logic would have treated every detected legend
#      as meaningful; the new logic filters by exact shape coverage.
# b) Complexity calibration:
#    - Distractors are mandatory, but capped at 1-2 so the scene stays readable.
#    - All real sprite masks are pairwise distinct, and distractor masks are also distinct
#      from real masks with a minimum Hamming difference. This avoids accidental ambiguity.
#    - Legend boxes use a stable visual marker: a gray rectangle with one colored glyph.
#      That keeps the extra reasoning observable without adding many new object types.
# c) Implicit-assumption audit:
#    - The original legend parser assumed every small legend-like object should contribute a
#      mapping. That assumption is invalid now, so the solver must first extract sprite
#      shapes, then accept only legend boxes whose inner glyph exactly matches one of them.
#    - The original sprite lookup excluded legend components via pairwise overlap ids. That
#      no longer works because legend glyphs may share arbitrary colors with sprites. The new
#      solver instead finds gray legend boxes first and excludes every colored component whose
#      bbox lies inside one of those boxes.
#    - Shape matching is only safe if sprite sizes are consistent and signatures are unique.
#      The generator guarantees one bbox size per placeholder color and unique real masks.
# d) Placement legality:
#    - Every outside patch is placed with `random_free_location` and a 1-cell padding buffer,
#      so sprites, legend boxes, and the main frame do not touch or overlap.
#    - Legend boxes are exact gray rectangular borders with N x N interiors, which makes
#      solver detection reliable and keeps legend glyph components fully contained.
# e) Visual observability:
#    - Humans can see that recoloring comes from boxed shape samples, not from placeholder
#      colors, because legend entries literally reuse the sprite silhouettes.
#    - Distractor boxes are visible as extra boxed shapes with no corresponding outside
#      template, so D13 is present in the input rather than hidden in generation metadata.
# f) Visual cleanliness:
#    - The frame, free sprites, and boxed legends are separated by background gaps.
#    - Boxed legends keep legend signals grouped by a shared border motif, reducing clutter.
# g) Color-change inferability:
#    - Output colors are still read directly from legend glyph colors.
#    - Because only a few distinct shapes appear and each used shape has repeated placements
#      inside the frame, 2-4 examples are enough to infer that unmatched legend entries are
#      decoys while matched boxed shapes determine recoloring.
# h) Generator validation target:
#    - Each generated sample must include at least one unmatched legend signature.
#    - `main(input)` must equal the output frame produced by the generator's oracle path.
# i) New solver assumptions checked against generation:
#    - Legend boxes are small gray rectangles with exactly one non-bg inner color.
#    - Every real sprite signature appears in exactly one legend box.
#    - No distractor legend shape equals a real sprite shape.


def _rect_border_mask(shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


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
            border = _rect_border_mask(sub.shape)
            if not np.all(sub[border] == col):
                continue
            if np.all(sub[~border] == col):
                continue
            candidates.append((col, (r0, c0, r1, c1)))
    assert candidates, "no rectangular frame component found"
    areas = [((b[2] - b[0] + 1) * (b[3] - b[1] + 1), idx) for idx, (_col, b) in enumerate(candidates)]
    max_area = max(area for area, _idx in areas)
    winners = [candidates[idx] for area, idx in areas if area == max_area]
    assert len(winners) == 1, f"expected one largest frame component, found {len(winners)}"
    frame_col, bbox = winners[0]
    return int(frame_col), bbox


def _bbox_hw(b: tuple[int, int, int, int]) -> tuple[int, int]:
    return b[2] - b[0] + 1, b[3] - b[1] + 1


def _inside_bbox(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> bool:
    return outer[0] <= inner[0] and inner[2] <= outer[2] and outer[1] <= inner[1] and inner[3] <= outer[3]


def _shape_signature_from_points(
    pts: set[tuple[int, int]] | list[tuple[int, int]], top_left: tuple[int, int], shape: tuple[int, int]
) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0 = top_left
    h, w = shape
    coords = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in pts))
    return int(h), int(w), coords


def _shape_signature_from_mask(mask: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    pts = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    return _shape_signature_from_points(pts, (0, 0), mask.shape)


def _extract_legend_entries(
    g: np.ndarray,
    outside: np.ndarray,
    frame_col: int,
    bg: int,
    allowed_inner_sizes: set[tuple[int, int]],
) -> list[dict]:
    entries: list[dict] = []
    for pts in components_mask((g == frame_col) & outside, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts)
        outer_h, outer_w = _bbox_hw((r0, c0, r1, c1))
        if outer_h < 5 or outer_w < 5:
            continue
        inner_shape = (outer_h - 2, outer_w - 2)
        if inner_shape not in allowed_inner_sizes:
            continue

        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        border = _rect_border_mask(sub.shape)
        if not np.all(sub[border] == frame_col):
            continue
        if np.any(sub[1:-1, 1:-1] == frame_col):
            continue

        inner = sub[1:-1, 1:-1]
        inner_colors = set(map(int, np.unique(inner))) - {bg, frame_col}
        if len(inner_colors) != 1:
            continue
        glyph_col = next(iter(inner_colors))
        glyph_pts = [(int(r), int(c)) for r, c in np.argwhere(inner == glyph_col)]
        assert glyph_pts, "legend glyph unexpectedly empty"
        entries.append(
            {
                "bbox": (r0, c0, r1, c1),
                "inner_bbox": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
                "color": int(glyph_col),
                "signature": _shape_signature_from_points(glyph_pts, (0, 0), inner.shape),
            }
        )
    assert entries, "no legend boxes found"
    return entries


def _extract_sprite_signature(comps: list[dict], src_col: int, want_shape: tuple[int, int]) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    all_candidates = [c for c in comps if c["col"] == src_col]
    assert all_candidates, f"no sprite components found for color {src_col}"
    sized = [c for c in all_candidates if _bbox_hw(c["bbox"]) == want_shape]
    assert len(sized) == 1, f"expected one size-matched sprite for color {src_col}, found {len(sized)}"
    base = sized[0]
    sr0, sc0, sr1, sc1 = base["bbox"]
    pts_set = set(base["pts"])
    for comp in all_candidates:
        if comp["id"] == base["id"]:
            continue
        assert _inside_bbox(comp["bbox"], base["bbox"]), f"extra sprite component for color {src_col} lies outside the base bbox"
        pts_set.update(comp["pts"])
    return _shape_signature_from_points(pts_set, (sr0, sc0), (sr1 - sr0 + 1, sc1 - sc0 + 1))


def _analyze_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    frame_col, (fr0, fc0, fr1, fc1) = _find_frame(g, bg)
    crop = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    H, W = crop.shape
    assert H >= 3 and W >= 3, f"frame too small for interior: {(H, W)}"

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
        dims = [_bbox_hw(b) for c, b in placeholders if c == col]
        assert dims, f"no placeholder component found for color {col}"
        unique_dims = sorted(set(dims))
        assert len(unique_dims) == 1, f"ambiguous placeholder sizes for color {col}: {unique_dims}"
        phw[col] = unique_dims[0]

    outside = np.ones(g.shape, dtype=bool)
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False
    allowed_sizes = set(phw.values())
    legends = _extract_legend_entries(g, outside, frame_col, bg, allowed_sizes)

    comps = []
    colors = [int(c) for c in np.unique(g) if int(c) not in (bg, frame_col)]
    cid = 0
    for col in colors:
        for pts in components_mask((g == col) & outside, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            pts_tuples = [(int(r), int(c)) for r, c in pts]
            bbox = (r0, c0, r1, c1)
            in_legend = any(_inside_bbox(bbox, entry["inner_bbox"]) for entry in legends)
            comps.append(
                {
                    "id": cid,
                    "col": int(col),
                    "pts": pts_tuples,
                    "bbox": bbox,
                    "area": len(pts_tuples),
                    "in_legend": in_legend,
                }
            )
            cid += 1

    sprite_comps = [c for c in comps if not c["in_legend"]]
    sprite_sigs: dict[int, tuple[int, int, tuple[tuple[int, int], ...]]] = {}
    for src_col in sorted(placeholder_colors):
        sprite_sigs[src_col] = _extract_sprite_signature(sprite_comps, src_col, phw[src_col])

    legend_map: dict[tuple[int, int, tuple[tuple[int, int], ...]], int] = {}
    for entry in legends:
        sig = entry["signature"]
        assert sig not in legend_map, f"duplicate legend signature detected: {sig[:2]}"
        legend_map[sig] = int(entry["color"])

    color_map: dict[int, int] = {}
    matched_signatures = set()
    for src_col, sig in sprite_sigs.items():
        assert sig in legend_map, f"no legend entry matches sprite shape for color {src_col}"
        color_map[src_col] = legend_map[sig]
        matched_signatures.add(sig)

    assert set(color_map) == placeholder_colors, "missing shape-to-color legend mapping"

    return {
        "bg": bg,
        "frame_col": frame_col,
        "crop": crop,
        "placeholders": placeholders,
        "phw": phw,
        "sprite_sigs": sprite_sigs,
        "color_map": color_map,
        "legend_entries": legends,
        "matched_legend_count": sum(1 for entry in legends if entry["signature"] in matched_signatures),
        "distractor_legend_count": sum(1 for entry in legends if entry["signature"] not in matched_signatures),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the framed placeholders with external sprites, then recolor them via boxed shape legends.

    Input structure:
    - A gray rectangular frame encloses a grid of solid-color placeholder blocks on background cells.
    - Outside the frame there is one source-colored sprite template for each placeholder color.
    - Outside the frame there are also several small gray legend boxes; each box contains one colored shape,
      and some of those boxed shapes are distractors that do not correspond to any real sprite.

    Rule:
    - Recover the sprite shape associated with each placeholder color by matching color and bbox size.
    - Read every gray legend box as a candidate mapping from shape pattern to output color.
    - Keep only the legend entries whose inner shape exactly matches one of the recovered sprites.
    - Replace each placeholder block with its sprite shape and recolor it to the color of the matching legend box.

    Key insight:
    - Legend validity is determined by exact shape agreement, not by the mere presence of a boxed entry.
      Extra boxed glyphs are decoys and must be filtered out before recoloring.
    """
    info = _analyze_scene(grid)
    crop = info["crop"]
    bg = info["bg"]
    placeholders = info["placeholders"]
    sprite_sigs = info["sprite_sigs"]
    color_map = info["color_map"]

    out = crop.copy()
    out[1:-1, 1:-1] = bg

    for src_col, (pr0, pc0, pr1, pc1) in placeholders:
        assert src_col in sprite_sigs and src_col in color_map, f"placeholder color {src_col} missing sprite or legend mapping"
        tgt_col = int(color_map[src_col])
        sh, sw, coords = sprite_sigs[src_col]
        assert (pr1 - pr0 + 1, pc1 - pc0 + 1) == (sh, sw), f"placeholder size does not match sprite for color {src_col}"
        assert pr0 + sh <= out.shape[0] and pc0 + sw <= out.shape[1], f"sprite for color {src_col} does not fit at placeholder"
        for dr, dc in coords:
            out[pr0 + dr, pc0 + dc] = tgt_col

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
SPRITE_FILL_MIN = 0.5
SPRITE_FILL_MAX = 0.9
MIN_FRAME_H = 13
MIN_FRAME_W = 18
MAX_FRAME_DIM = 21
FRAME_MARGIN_SLACK = 4
CANVAS_H_RANGE = (22, 31)
CANVAS_W_RANGE = (23, 31)
FRAME_POS_JITTER_PROB = 0.25
PATCH_PADDING = 1
GEN_RETRIES = 400
DISTRACTOR_RANGE = (1, 2)
MIN_MASK_DIFF = 3


def _sprite_mask(n: int) -> np.ndarray:
    """Generate a random NxN binary mask with fill ratio in [SPRITE_FILL_MIN, SPRITE_FILL_MAX]."""
    lo = max((n * n + 1) // 2, int(np.ceil(SPRITE_FILL_MIN * n * n)))
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


def _mask_hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a != b))


def _sample_distinct_masks(n: int, count: int, forbidden: list[np.ndarray] | None = None) -> list[np.ndarray] | None:
    chosen: list[np.ndarray] = []
    existing = [m.copy() for m in (forbidden or [])]
    seen = {_shape_signature_from_mask(m) for m in existing}
    attempts = 0
    while len(chosen) < count and attempts < 3000:
        attempts += 1
        mask = _sprite_mask(n)
        sig = _shape_signature_from_mask(mask)
        if sig in seen:
            continue
        if any(_mask_hamming(mask, other) < MIN_MASK_DIFF for other in existing):
            continue
        chosen.append(mask)
        existing.append(mask)
        seen.add(sig)
    return chosen if len(chosen) == count else None


def _colored_sprite(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = int(color)
    return out


def _legend_box(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    h, w = mask.shape
    patch = np.full((h + 2, w + 2), bg, dtype=int)
    patch[0, :] = patch[-1, :] = FRAME_COL
    patch[:, 0] = patch[:, -1] = FRAME_COL
    patch[1:-1, 1:-1][mask] = int(color)
    return patch


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


def _gen_attempt() -> np.ndarray | None:
    struct = _choose_structure()
    if struct is None:
        return None
    n, rows, cols, empty_count, _ = struct

    bg = random.choice([c for c in range(10) if c != FRAME_COL])
    usable = [c for c in range(10) if c not in (bg, FRAME_COL)]

    k = random.randint(*NUM_COLORS_RANGE)
    src_cols = random.sample(usable, k)
    tgt_cols = [random.choice([c for c in usable if c != src]) for src in src_cols]
    color_to_tgt = dict(zip(src_cols, tgt_cols))

    real_masks = _sample_distinct_masks(n, k)
    if real_masks is None:
        return None
    masks = dict(zip(src_cols, real_masks))

    max_distractors = DISTRACTOR_RANGE[1]
    if n >= 5 or k >= 4:
        max_distractors = 1
    distractor_count = random.randint(DISTRACTOR_RANGE[0], max_distractors)
    distractor_masks = _sample_distinct_masks(n, distractor_count, forbidden=real_masks)
    if distractor_masks is None:
        return None

    ch, cw = _content_extent(n, rows, cols)
    min_fh = max(MIN_FRAME_H, ch + 4)
    min_fw = max(MIN_FRAME_W, cw + 4)
    frame_h = random.randint(min_fh, min(MAX_FRAME_DIM, max(min_fh, ch + 2 + FRAME_MARGIN_SLACK)))
    frame_w = random.randint(min_fw, min(MAX_FRAME_DIM, max(min_fw, cw + 2 + FRAME_MARGIN_SLACK)))
    y_margin = frame_h - 2 - ch
    x_margin = frame_w - 2 - cw
    top_margin = random.randint(1, y_margin - 1)
    left_margin = random.randint(1, x_margin - 1)

    _, filled_slots = _choose_empty_slots(rows, cols, empty_count)
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

    sprite_patches: list[tuple[np.ndarray, bool]] = [(_colored_sprite(masks[src], src, bg), True) for src in src_cols]

    legend_entries: list[tuple[np.ndarray, bool]] = []
    for src in src_cols:
        legend_entries.append((_legend_box(masks[src], color_to_tgt[src], bg), False))
    distractor_colors = [random.choice(usable) for _ in range(distractor_count)]
    for mask, col in zip(distractor_masks, distractor_colors):
        legend_entries.append((_legend_box(mask, col, bg), False))

    ext_area = sum(patch.size for patch, _ in sprite_patches + legend_entries) + 12 * (len(sprite_patches) + len(legend_entries))
    possible_dims = [
        (h, w)
        for h in range(*CANVAS_H_RANGE)
        for w in range(*CANVAS_W_RANGE)
        if h >= frame_h and w >= frame_w and h * w - frame_h * frame_w >= ext_area
    ]
    if not possible_dims:
        return None

    h, w = random.choice(possible_dims)
    grid = np.full((h, w), bg, dtype=int)
    reserve = np.full((h, w), bg, dtype=int)

    fr0 = random.choice([0, h - frame_h])
    fc0 = random.choice([0, w - frame_w])
    if h - frame_h > 2 and random.random() < FRAME_POS_JITTER_PROB:
        fr0 = random.randrange(0, h - frame_h + 1)
    if w - frame_w > 2 and random.random() < FRAME_POS_JITTER_PROB:
        fc0 = random.randrange(0, w - frame_w + 1)
    paste(grid, input_frame, fr0, fc0)
    paste(reserve, np.full((frame_h, frame_w), FRAME_COL, dtype=int), fr0, fc0)

    patches = sprite_patches + legend_entries
    random.shuffle(patches)
    patches.sort(key=lambda item: item[0].size, reverse=True)
    for patch, transparent in patches:
        if not _place_patch(grid, reserve, patch, bg, transparent=transparent):
            return None

    try:
        info = _analyze_scene(grid)
        if info["distractor_legend_count"] < 1:
            return None
        if info["matched_legend_count"] != k:
            return None
        if not np.array_equal(main(grid), output_frame):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        out = _gen_attempt()
        if out is not None:
            return out
    raise RuntimeError("failed to generate valid edb79dae D13 input")
