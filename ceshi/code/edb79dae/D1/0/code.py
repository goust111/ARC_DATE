# Variant of edb79dae
# Dimension: D1 — 第一步匹配条件：颜色匹配 (match_cond)
# Change: 精灵颜色 == 框架内目标像素颜色（s.color == d.color），通过颜色一致性将外部精灵与内部单色像素配对... → 形状匹配替代颜色匹配

import itertools
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location

# Design analysis for the D1 variant
# ----------------------------------
# variant_spec:
#   core_change:
#     original: each occupied slot inside the frame is a solid source-colored NxN block; the solver matches that color to the
#       equally colored sprite outside the frame.
#     variant: each occupied slot inside the frame is a monochrome shape marker whose normalized geometry identifies the sprite.
#       The marker color is decoupled from the sprite color, so matching must use shape, not color.
#   concrete_encoding:
#     - The frame, slot grid, legends, and external sprite placement all stay in the original positions/style.
#     - Every source type owns one connected NxN mask with full bbox extent.
#     - Inside the frame, each occupied slot shows that mask in an arbitrary marker color.
#     - Outside the frame, the same mask appears once in the source color, plus the same 3x4 legend mapping source color -> target color.
#     - Output keeps only the frame crop and recolors each stamped sprite to the legend target color.
#   trace_example:
#     - Original local step: a 4x4 solid blue placeholder is found inside the frame -> solver looks for the blue external sprite
#       -> solver pastes that sprite and recolors it by the blue legend.
#     - Variant local step: a 4x4 sparse "zigzag" marker in orange is found inside the frame -> orange is ignored -> solver finds the
#       external sprite whose normalized shape is the same zigzag -> solver uses that sprite's own source color to query the legend.
#     - Difference: the discriminative signal moves from color equality to geometry equality.
#   complexity_calibration:
#     - Source types remain 2-5, and each type can repeat across multiple slots, so shape matching is exercised repeatedly.
#     - Marker colors are drawn from a small reusable pool, forcing collisions across different shapes; this blocks any hidden
#       fallback to color matching and makes the D1 change visually obvious.
#     - Shapes are connected and have bbox size NxN, which keeps component extraction simple and prevents ambiguity from fragmented
#       markers while still making the interior visibly sparse rather than solid.
#   implicit_assumption_audit:
#     - Old assumption: placeholder_colors == source sprite colors. No longer true, so legend decoding cannot depend on interior colors.
#     - Old assumption: placeholder bbox size can be inferred from solid blocks. Still true, but now the bbox comes from the marker's
#       full-extent shape instead of a filled square.
#     - Old assumption: selecting sprites by color and size is sufficient. Replaced with exact normalized shape matching.
#     - New assumption: each source type has a unique normalized shape signature. The generator enforces mask uniqueness.
#     - New assumption: each interior marker is one monochrome 4-connected component. The generator builds connected masks only.
#   placement_legality:
#     - The slot grid and frame dimensions are unchanged, so markers stay non-overlapping inside the frame with 1-cell bg gaps.
#     - External sprites and legends are still placed through random_free_location with padding, so outside objects do not overlap.
#     - All masks use the same NxN bbox as the slot they occupy, so stamping inside the frame is always in-bounds.
#   visual_observability:
#     - A human can compare repeated sparse shapes inside the frame against the external templates; repeated marker colors across
#       different shapes make the "shape, not color" rule legible in 2-4 examples.
#     - The recoloring rule remains exposed by the unchanged legend mechanism, so only one reasoning step changes.
#   cleanliness:
#     - There are only three outside object families: frame, sprites, legends. No extra guide objects are introduced.
#     - Interior regions contain one sparse marker per occupied slot, which is visually cleaner than mixing multiple overlays.
#   color_inference:
#     - Legends still provide the only output-color cue.
#     - Marker colors intentionally have many-to-one / one-to-many reuse against source types, so they are visibly non-semantic.
#   solver_generator_contract:
#     - main() assumes unique sprite shapes, exact shape reuse between inside marker and outside sprite, and one legend per source color.
#     - generate_input() guarantees all three conditions before accepting a sample.


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


def _shape_signature(pts: list[tuple[int, int]] | np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = bbox
    rel = tuple(sorted((int(r - r0), int(c - c0)) for r, c in pts))
    return (int(r1 - r0 + 1), int(c1 - c0 + 1), rel)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor sprite-shaped markers inside a frame by matching each marker's geometry to an external sprite template.

    Input structure:
    - bg background, one rectangular single-color frame, and sparse monochrome shape markers placed on an interior slot grid.
    - Outside the frame: one sprite template per source color plus a compact two-color legend for the final recoloring.

    Rule:
    - Find every interior marker component and normalize its shape within its bbox.
    - Decode the legend outside the frame to obtain source-color -> target-color mappings.
    - Match each marker to the unique external sprite with the same normalized shape, then stamp that sprite into the marker's bbox.
    - Recolor every stamped sprite using the legend target color associated with the matched sprite's source color.

    Key insight: Interior colors are no longer meaningful; the only stable link between an inside marker and an outside sprite is their shared shape signature.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    frame_col, (fr0, fc0, fr1, fc1) = _find_frame(g, bg)

    crop = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    H, W = crop.shape
    assert H >= 3 and W >= 3, f"frame too small for interior: {(H, W)}"

    inner_mask = np.zeros((H, W), dtype=bool)
    inner_mask[1:-1, 1:-1] = True

    placeholders = []
    placeholder_colors = set(map(int, np.unique(crop[inner_mask]))) - {bg, frame_col}
    for col in sorted(placeholder_colors):
        for pts in components_mask((crop == col) & inner_mask, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            sig = _shape_signature(pts, (r0, c0, r1, c1))
            placeholders.append({"col": int(col), "bbox": (r0, c0, r1, c1), "sig": sig})
    assert placeholders, "no interior marker components found"

    outside = np.ones(g.shape, dtype=bool)
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False

    comps = []
    colors = [int(c) for c in np.unique(g) if int(c) not in (bg, frame_col)]
    cid = 0
    for col in colors:
        for pts in components_mask((g == col) & outside, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            pts_tuples = [(int(r), int(c)) for r, c in pts]
            comps.append(
                {
                    "id": cid,
                    "col": int(col),
                    "pts": pts_tuples,
                    "bbox": (r0, c0, r1, c1),
                    "area": len(pts_tuples),
                    "sig": _shape_signature(pts_tuples, (r0, c0, r1, c1)),
                }
            )
            cid += 1

    def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
        ar0, ac0, ar1, ac1 = a
        br0, bc0, br1, bc1 = b
        return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)

    color_map: dict[int, int] = {}
    legend_ids: set[int] = set()
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            ci = comps[i]
            cj = comps[j]
            if ci["col"] == cj["col"]:
                continue
            if not _bbox_overlap(ci["bbox"], cj["bbox"]):
                continue
            r0 = min(ci["bbox"][0], cj["bbox"][0])
            c0 = min(ci["bbox"][1], cj["bbox"][1])
            r1 = max(ci["bbox"][2], cj["bbox"][2])
            c1 = max(ci["bbox"][3], cj["bbox"][3])
            if (r1 - r0 + 1) > 8 or (c1 - c0 + 1) > 8:
                continue
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            sub_colors = set(map(int, np.unique(sub))) - {frame_col}
            if bg in sub_colors or len(sub_colors) != 2:
                continue
            cset = set(sub_colors)
            if ci["col"] not in cset or cj["col"] not in cset:
                continue
            assert ci["area"] != cj["area"], f"ambiguous equal-area legend pair: {ci['col']} and {cj['col']}"
            if ci["area"] > cj["area"]:
                src, tgt = ci["col"], cj["col"]
            else:
                src, tgt = cj["col"], ci["col"]
            assert src not in color_map, f"duplicate legend mapping for source color {src}"
            color_map[src] = int(tgt)
            legend_ids.add(ci["id"])
            legend_ids.add(cj["id"])
    assert color_map, "no legends decoded"

    sprite_coords: dict[int, tuple[tuple[tuple[int, int], ...], int, int]] = {}
    shape_to_src: dict[tuple[int, int, tuple[tuple[int, int], ...]], int] = {}
    for src_col in sorted(color_map):
        candidates = [c for c in comps if c["col"] == src_col and c["id"] not in legend_ids]
        assert len(candidates) == 1, f"expected one non-legend sprite for color {src_col}, found {len(candidates)}"
        base = candidates[0]
        sr0, sc0, sr1, sc1 = base["bbox"]
        coords = tuple(sorted((r - sr0, c - sc0) for r, c in base["pts"]))
        sprite_coords[src_col] = (coords, sr1 - sr0 + 1, sc1 - sc0 + 1)
        assert base["sig"] not in shape_to_src, "duplicate sprite shape signature outside the frame"
        shape_to_src[base["sig"]] = int(src_col)

    out = crop.copy()
    out[1:-1, 1:-1] = bg

    for ph in placeholders:
        sig = ph["sig"]
        assert sig in shape_to_src, "interior marker shape has no matching external sprite"
        src_col = int(shape_to_src[sig])
        tgt_col = int(color_map[src_col])
        coords, sh, sw = sprite_coords[src_col]
        pr0, pc0, pr1, pc1 = ph["bbox"]
        assert (pr1 - pr0 + 1, pc1 - pc0 + 1) == (sh, sw), "marker bbox does not match sprite bbox"
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
SPRITE_FILL_MIN = 0.30
SPRITE_FILL_MAX = 0.65
MIN_FRAME_H = 13
MIN_FRAME_W = 18
MAX_FRAME_DIM = 21
FRAME_MARGIN_SLACK = 4
CANVAS_H_RANGE = (22, 31)
CANVAS_W_RANGE = (23, 31)
FRAME_POS_JITTER_PROB = 0.25
PATCH_PADDING = 1
GEN_RETRIES = 300


def _legend(src: int, tgt: int) -> np.ndarray:
    """Build a 3x4 legend block mapping src color to tgt color."""
    return np.array(
        [
            [src, src, tgt, tgt],
            [src, src, src, tgt],
            [src, src, tgt, tgt],
        ],
        dtype=int,
    )


def _mask_signature(mask: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    return _shape_signature([(int(r), int(c)) for r, c in pts], (r0, c0, r1, c1))


def _shape_mask(n: int) -> np.ndarray:
    """Generate a connected NxN marker/sprite mask with full bbox extent and moderate sparsity."""
    lo = max(n + 1, int(np.ceil(SPRITE_FILL_MIN * n * n)))
    hi = min(int(np.floor(SPRITE_FILL_MAX * n * n)), n * n - 1)
    target = random.randint(lo, hi)

    mask = np.zeros((n, n), dtype=bool)

    c = random.randrange(n)
    mask[0, c] = True
    for r in range(1, n):
        c = max(0, min(n - 1, c + random.choice([-1, 0, 1])))
        mask[r, c] = True

    for edge_col in (0, n - 1):
        pts = list(zip(*np.where(mask)))
        rr, cc = random.choice(pts)
        c0, c1 = sorted((cc, edge_col))
        mask[rr, c0 : c1 + 1] = True

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
        if not frontier:
            break
        r, c = random.choice(frontier)
        mask[r, c] = True

    assert mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()
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
    grid: np.ndarray,
    reserve: np.ndarray,
    patch: np.ndarray,
    bg: int,
    *,
    transparent: bool,
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

    masks = {}
    seen_shapes = set()
    for src in src_cols:
        for _ in range(60):
            mask = _shape_mask(n)
            sig = _mask_signature(mask)
            if sig not in seen_shapes:
                seen_shapes.add(sig)
                masks[src] = mask
                break
        else:
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
    cell_src = {}
    for slot, col in zip(filled_slots[:k], src_cols):
        cell_src[slot] = col
    for slot in filled_slots[k:]:
        cell_src[slot] = random.choice(src_cols)

    guide_palette_size = min(random.randint(2, 3), len(usable))
    guide_palette = random.sample(usable, guide_palette_size)
    marker_colors = {slot: random.choice(guide_palette) for slot in filled_slots}

    color_to_srcs: dict[int, set[int]] = {}
    for slot in filled_slots:
        color_to_srcs.setdefault(marker_colors[slot], set()).add(cell_src[slot])
    if all(marker_colors[slot] == cell_src[slot] for slot in filled_slots):
        return None
    if max(len(srcs) for srcs in color_to_srcs.values()) < 2:
        return None

    input_frame = _make_bordered_frame(frame_h, frame_w, bg)
    output_frame = _make_bordered_frame(frame_h, frame_w, bg)
    for (rr, cc), src in cell_src.items():
        r0 = 1 + top_margin + rr * (n + 1)
        c0 = 1 + left_margin + cc * (n + 1)
        paste(input_frame, _colored_sprite(masks[src], marker_colors[(rr, cc)], bg), r0, c0, transparent=bg)
        paste(output_frame, _colored_sprite(masks[src], color_to_tgt[src], bg), r0, c0, transparent=bg)

    ext_area = k * (12 + n * n + 10)
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

    patches: list[tuple[np.ndarray, bool]] = []
    for src, tgt in zip(src_cols, tgt_cols):
        patches.append((_colored_sprite(masks[src], src, bg), True))
        patches.append((_legend(src, tgt), False))
    random.shuffle(patches)
    patches.sort(key=lambda item: item[0].size, reverse=True)
    for patch, transparent in patches:
        if not _place_patch(grid, reserve, patch, bg, transparent=transparent):
            return None

    try:
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
    raise RuntimeError("failed to generate valid edb79dae input")
