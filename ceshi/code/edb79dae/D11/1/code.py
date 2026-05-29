# Variant of edb79dae
# Dimension: D11 — 灰色框架的结构 (meta)
# Change: 单一矩形灰色框架，框架线条均匀，内部为操作区域... → 框架为非矩形（L形或十字形）

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the D11 variant
# a) Concrete behavior:
#    - The gray frame is no longer the 1-cell border of a rectangle. It is the 1-cell
#      outline around an irregular enclosed region shaped like an L or a cross.
#    - Placeholder placement still uses regular N x N slot anchors, but the visible
#      placeholder cells are "slot box intersect irregular interior". Some slots are
#      clipped by the interior boundary while keeping the same N x N bbox.
#    - Output uses the same anchor boxes. Each outside sprite is stamped into the
#      matching placeholder bbox and clipped against the irregular interior mask.
#      Example: in the rectangular task, a 4 x 4 sprite placed at a 4 x 4 placeholder
#      fills only according to the sprite mask. In this variant, if the top-right
#      corner of that 4 x 4 box lies outside the L/cross interior, those sprite pixels
#      are removed; the bbox stays 4 x 4, but the visible result is truncated.
# b) Complexity calibration:
#    - Shape family is limited to two visible categories: L and cross. This keeps the
#      new rule learnable from a few examples while making the frame visibly different.
#    - We require 6-10 valid placeholder slots and at least one clipped slot so the
#      irregular-boundary rule is exercised instead of degenerating back to rectangles.
# c) Original-code assumptions that break:
#    - `_find_frame` assumed "frame = rectangle bbox border"; this fails on non-rectangular
#      outlines. The new solver instead flood-fills non-frame cells from the bbox boundary
#      and treats the unreached cells as the enclosed operating region.
#    - `inner_mask[1:-1,1:-1] = True` assumed every cell inside the outer rectangle is usable.
#      In the variant, only the flood-fill-enclosed cells are usable; exterior pockets inside
#      the frame bbox must remain background.
#    - Unconditional sprite writes were harmless before because every placeholder bbox was
#      fully inside the rectangular interior. Here writes must be clipped against the irregular
#      interior mask or they would leak across the missing corners / arms.
# d) Placement legality:
#    - Generator creates the irregular interior first, then accepts only slot boxes whose
#      intersection with the interior is 4-connected, non-empty, and still spans the full
#      N x N bbox. This guarantees placeholders remain separable and solver-visible.
#    - External sprites and legends are still placed outside the frame bbox with padding.
# e) Visual observability:
#    - The gray shape change is obvious in the input and output.
#    - At least one placeholder is visibly truncated by the L/cross boundary, so the
#      clipping rule can be seen directly rather than inferred from hidden metadata.
# f) Visual cleanliness:
#    - The interior support mask is built from broad bars, not thin corridors, so clipped
#      placeholders remain legible. Frame, placeholders, legends, and outside sprites stay
#      separated by background gaps.
# g) Color inference:
#    - Color logic is unchanged from the base task: placeholder color selects the sprite and
#      the compact two-color legend selects the recolor target. The new difficulty is spatial,
#      not an added hidden color rule.


def _boundary_reachable(mask: np.ndarray) -> np.ndarray:
    """Return the subset of True cells reachable from the array boundary via 4-neighbors."""
    H, W = mask.shape
    seen = np.zeros((H, W), dtype=bool)
    stack: list[tuple[int, int]] = []

    for r in range(H):
        for c in (0, W - 1):
            if mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
    for c in range(W):
        for r in (0, H - 1):
            if mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return seen


def _find_frame(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int], np.ndarray]:
    """
    Find a single-color non-rectangular frame component by enclosure:
      - take a connected component of one color,
      - within its bbox, flood-fill non-frame cells from the bbox boundary,
      - any unreached non-frame cells are the operating region enclosed by that frame.
    Return (frame_color, bbox, enclosed_mask_within_bbox).
    """
    candidates = []
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    for col in colors:
        for pts in components_mask(g == col, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            frame_mask = sub == col
            non_frame = ~frame_mask
            exterior = _boundary_reachable(non_frame)
            enclosed = non_frame & ~exterior
            if not enclosed.any():
                continue
            inside_parts = components_mask(enclosed, connectivity=4)
            if len(inside_parts) != 1:
                continue
            candidates.append((int(col), (r0, c0, r1, c1), enclosed))
    assert len(candidates) == 1, f"expected exactly one enclosing frame component, found {len(candidates)}"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp recolored sprites into an irregular framed region using placeholder anchors and a color legend.

    Input structure:
    - bg background, one gray non-rectangular enclosing frame, and colored placeholder shapes inside the enclosed region.
    - Outside the frame bbox: sprite shapes (one per placeholder color) and compact two-color legends that specify recoloring.

    Rule:
    - Recover the frame's enclosed operating region by treating the gray component as a barrier and flood-filling from the bbox boundary.
    - For each placeholder color, match the same-color sprite outside the frame and use each placeholder bbox as a placement anchor.
    - Stamp the sprite into that bbox, but keep only the pixels that lie inside the irregular enclosed region.
    - Recolor every stamped sprite according to the outside legend mapping.

    Key insight: The placeholder bbox still tells you where each sprite belongs, but the irregular gray outline defines a second mask that clips the pasted sprite, so placement depends on both color matching and frame geometry.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    frame_col, (fr0, fc0, fr1, fc1), interior_mask = _find_frame(g, bg)

    crop = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    H, W = crop.shape
    assert H >= 3 and W >= 3, f"frame too small for interior: {(H, W)}"

    placeholder_colors = set(map(int, np.unique(crop[interior_mask]))) - {bg, frame_col}
    assert placeholder_colors, "no placeholder colors found inside irregular frame"

    placeholders: list[tuple[int, tuple[int, int, int, int]]] = []
    for col in sorted(placeholder_colors):
        for pts in components_mask((crop == col) & interior_mask, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            placeholders.append((col, (r0, c0, r1, c1)))

    phw: dict[int, tuple[int, int]] = {}
    for col in placeholder_colors:
        dims = [(b[2] - b[0] + 1, b[3] - b[1] + 1) for c, b in placeholders if c == col]
        assert dims, f"no placeholder component found for color {col}"
        unique_dims = sorted(set(dims))
        assert len(unique_dims) == 1, f"ambiguous placeholder sizes for color {col}: {unique_dims}"
        phw[col] = unique_dims[0]

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
            if ci["col"] == cj["col"] or not _bbox_overlap(ci["bbox"], cj["bbox"]):
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
            if ci["col"] not in sub_colors or cj["col"] not in sub_colors:
                continue

            if (ci["col"] in placeholder_colors) ^ (cj["col"] in placeholder_colors):
                src = ci["col"] if ci["col"] in placeholder_colors else cj["col"]
                tgt = cj["col"] if src == ci["col"] else ci["col"]
            else:
                assert ci["area"] != cj["area"], f"ambiguous equal-area legend pair: {ci['col']} and {cj['col']}"
                if ci["area"] > cj["area"]:
                    src, tgt = ci["col"], cj["col"]
                else:
                    src, tgt = cj["col"], ci["col"]

            assert src in placeholder_colors, f"legend source color {src} is not a placeholder"
            assert src not in color_map, f"duplicate legend mapping for placeholder color {src}"
            color_map[src] = int(tgt)
            legend_ids.add(ci["id"])
            legend_ids.add(cj["id"])

    assert set(color_map) == placeholder_colors, (
        f"legend mappings {sorted(color_map)} do not match placeholders {sorted(placeholder_colors)}"
    )

    sprite_coords: dict[int, tuple[tuple[tuple[int, int], ...], int, int]] = {}
    for src_col in sorted(placeholder_colors):
        all_candidates = [c for c in comps if c["col"] == src_col and c["id"] not in legend_ids]
        assert all_candidates, f"no non-legend sprite found for color {src_col}"
        want_h, want_w = phw[src_col]
        sized = [
            c
            for c in all_candidates
            if (c["bbox"][2] - c["bbox"][0] + 1) == want_h and (c["bbox"][3] - c["bbox"][1] + 1) == want_w
        ]
        assert len(sized) == 1, f"expected one size-matched sprite for color {src_col}, found {len(sized)}"
        base = sized[0]
        sr0, sc0, sr1, sc1 = base["bbox"]

        pts_set = set(base["pts"])
        for c in all_candidates:
            if c["id"] == base["id"]:
                continue
            r0, c0, r1, c1 = c["bbox"]
            assert sr0 <= r0 and sr1 >= r1 and sc0 <= c0 and sc1 >= c1, (
                f"extra sprite component for color {src_col} is outside the base bbox"
            )
            pts_set.update(c["pts"])

        coords = tuple(sorted((r - sr0, c - sc0) for r, c in pts_set))
        sprite_coords[src_col] = (coords, sr1 - sr0 + 1, sc1 - sc0 + 1)

    out = crop.copy()
    out[interior_mask] = bg

    for src_col, (pr0, pc0, pr1, pc1) in placeholders:
        assert src_col in sprite_coords and src_col in color_map, f"placeholder color {src_col} missing sprite or color mapping"
        tgt_col = int(color_map[src_col])
        coords, sh, sw = sprite_coords[src_col]
        assert (pr1 - pr0 + 1, pc1 - pc0 + 1) == (sh, sw), (
            f"placeholder size does not match sprite for color {src_col}"
        )
        off_r = pr0
        off_c = pc0
        for dr, dc in coords:
            rr, cc = off_r + dr, off_c + dc
            if 0 <= rr < H and 0 <= cc < W and interior_mask[rr, cc]:
                out[rr, cc] = tgt_col

    return out


# --------------- generator (o5x) ---------------

# === CFG: tunables ===
FRAME_COL = 5
SPRITE_N_RANGE = (3, 5)            # N in {3, 4}
GRID_ROWS_RANGE = (3, 5)           # rows in {3, 4}
GRID_COLS = 5
TOTAL_BLOCKS_RANGE = (6, 10)
NUM_COLORS_RANGE = (2, 5)
SPRITE_FILL_MIN = 0.5
SPRITE_FILL_MAX = 0.9
MIN_VISIBLE_FILL = 0.45            # min visible placeholder fill inside a clipped slot
CANVAS_H_RANGE = (24, 33)
CANVAS_W_RANGE = (28, 35)
FRAME_POS_JITTER_PROB = 0.25
PATCH_PADDING = 1
SHAPE_ATTEMPTS = 200
GEN_RETRIES = 300


def _legend(src: int, tgt: int) -> np.ndarray:
    """Build a 3x4 legend block mapping src color to tgt color."""
    return np.array(
        [[src, src, tgt, tgt],
         [src, src, src, tgt],
         [src, src, tgt, tgt]],
        dtype=int,
    )


def _sprite_mask(n: int) -> np.ndarray:
    """Generate a random N x N binary mask with fill ratio in [SPRITE_FILL_MIN, SPRITE_FILL_MAX]."""
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


def _colored_sprite(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    """Render a binary mask as a colored sprite on bg."""
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = int(color)
    return out


def _content_extent(n: int, grid_rows: int, grid_cols: int) -> tuple[int, int]:
    """Content area for grid_rows x grid_cols of N x N boxes with 1-cell gaps."""
    return (grid_rows * n + grid_rows - 1, grid_cols * n + grid_cols - 1)


def _connected(mask: np.ndarray) -> bool:
    """Whether the True cells in mask form one 4-connected component."""
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    seen = {tuple(pts[0])}
    stack = [tuple(pts[0])]
    H, W = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(pts)


def _slot_candidates(
    content_mask: np.ndarray, n: int, rows: int, cols: int
) -> list[dict[str, object]]:
    """Return valid slot boxes whose intersection with the irregular interior stays bbox-sized."""
    candidates: list[dict[str, object]] = []
    min_area = max(n + 2, int(np.ceil(MIN_VISIBLE_FILL * n * n)))
    for rr in range(rows):
        for cc in range(cols):
            r0 = rr * (n + 1)
            c0 = cc * (n + 1)
            sub = content_mask[r0 : r0 + n, c0 : c0 + n]
            pts = np.argwhere(sub)
            if len(pts) == 0 or int(sub.sum()) < min_area:
                continue
            br0, bc0 = pts.min(axis=0)
            br1, bc1 = pts.max(axis=0)
            if (int(br0), int(bc0), int(br1), int(bc1)) != (0, 0, n - 1, n - 1):
                continue
            if not _connected(sub):
                continue
            candidates.append(
                {
                    "slot": (rr, cc),
                    "mask": sub.copy(),
                    "visible": int(sub.sum()),
                    "clipped": bool(int(sub.sum()) < n * n),
                }
            )
    return candidates


def _random_content_shape(n: int, rows: int, cols: int) -> tuple[np.ndarray, str, list[dict[str, object]]] | None:
    """Build an L or cross interior mask whose valid slot count stays in range."""
    H, W = _content_extent(n, rows, cols)
    for _ in range(SHAPE_ATTEMPTS):
        shape_type = random.choice(["L", "cross"])
        mask = np.zeros((H, W), dtype=bool)
        if shape_type == "cross":
            v_w = random.randint(n, min(W - 2, 2 * n + 1))
            h_w = random.randint(n, min(H - 2, 2 * n + 1))
            v0_lo = max(0, W // 2 - n)
            v0_hi = min(W - v_w, W // 2)
            h0_lo = max(0, H // 2 - n)
            h0_hi = min(H - h_w, H // 2)
            v0 = random.randint(v0_lo, v0_hi)
            h0 = random.randint(h0_lo, h0_hi)
            mask[:, v0 : v0 + v_w] = True
            mask[h0 : h0 + h_w, :] = True
        else:
            v_w = random.randint(n, min(W - 2, 2 * n + 1))
            h_w = random.randint(n, min(H - 2, 2 * n + 1))
            mask[:, :v_w] = True
            mask[H - h_w :, :] = True
            mask = np.rot90(mask, k=random.randrange(4)).copy()

        candidates = _slot_candidates(mask, n, rows, cols)
        slot_count = len(candidates)
        clipped = sum(int(c["clipped"]) for c in candidates)
        rows_seen = {int(c["slot"][0]) for c in candidates}
        cols_seen = {int(c["slot"][1]) for c in candidates}
        if TOTAL_BLOCKS_RANGE[0] <= slot_count <= TOTAL_BLOCKS_RANGE[1] and clipped >= 1:
            if len(rows_seen) == rows and len(cols_seen) >= 4:
                return mask, shape_type, candidates
    return None


def _build_frame_patch(content_mask: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert the irregular interior mask into a gray outline frame patch and local interior mask."""
    Hc, Wc = content_mask.shape
    interior = np.zeros((Hc + 2, Wc + 2), dtype=bool)
    interior[1:-1, 1:-1] = content_mask

    dilated = np.zeros_like(interior)
    dilated |= interior
    dilated[1:, :] |= interior[:-1, :]
    dilated[:-1, :] |= interior[1:, :]
    dilated[:, 1:] |= interior[:, :-1]
    dilated[:, :-1] |= interior[:, 1:]
    dilated[1:, 1:] |= interior[:-1, :-1]
    dilated[1:, :-1] |= interior[:-1, 1:]
    dilated[:-1, 1:] |= interior[1:, :-1]
    dilated[:-1, :-1] |= interior[1:, 1:]
    frame_mask = dilated & ~interior

    patch = np.full(interior.shape, bg, dtype=int)
    patch[frame_mask] = FRAME_COL
    return patch, interior


def _place_patch(
    grid: np.ndarray, reserve: np.ndarray, patch: np.ndarray, bg: int,
    *, transparent: bool,
) -> bool:
    """Find a free location for patch and paste it onto grid/reserve. Returns success."""
    blocker = np.full(patch.shape, FRAME_COL, dtype=int)
    try:
        r, c = random_free_location(reserve, blocker, transparent=bg, padding=PATCH_PADDING)
    except ValueError:
        return False
    paste(reserve, blocker, r, c)
    paste(grid, patch, r, c, transparent=bg if transparent else None)
    return True


def _gen_attempt() -> np.ndarray | None:
    n = random.randrange(*SPRITE_N_RANGE)
    rows = random.randrange(*GRID_ROWS_RANGE)
    cols = GRID_COLS

    shaped = _random_content_shape(n, rows, cols)
    if shaped is None:
        return None
    content_mask, _, candidates = shaped

    bg = random.choice([c for c in range(10) if c != FRAME_COL])
    usable = [c for c in range(10) if c not in (bg, FRAME_COL)]

    total = len(candidates)
    k = random.randint(NUM_COLORS_RANGE[0], min(NUM_COLORS_RANGE[1], total))
    src_cols = random.sample(usable, k)
    tgt_cols = [random.choice([c for c in usable if c != src]) for src in src_cols]
    color_to_tgt = dict(zip(src_cols, tgt_cols))
    masks = {src: _sprite_mask(n) for src in src_cols}

    input_frame, interior = _build_frame_patch(content_mask, bg)
    output_frame = input_frame.copy()

    cell_colors: dict[tuple[int, int], int] = {}
    shuffled = candidates[:]
    random.shuffle(shuffled)
    for info, col in zip(shuffled[:k], src_cols):
        cell_colors[tuple(info["slot"])] = col
    for info in shuffled[k:]:
        cell_colors[tuple(info["slot"])] = random.choice(src_cols)

    info_by_slot = {tuple(info["slot"]): info for info in candidates}
    for slot, src in cell_colors.items():
        rr, cc = slot
        r0 = 1 + rr * (n + 1)
        c0 = 1 + cc * (n + 1)
        visible_mask = np.asarray(info_by_slot[slot]["mask"], dtype=bool)
        input_sub = input_frame[r0 : r0 + n, c0 : c0 + n]
        output_sub = output_frame[r0 : r0 + n, c0 : c0 + n]
        input_sub[visible_mask] = src
        output_sub[masks[src] & visible_mask] = color_to_tgt[src]

    frame_h, frame_w = input_frame.shape
    ext_area = k * (12 + n * n + 12)
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
    raise RuntimeError("failed to generate valid edb79dae D11 input")
