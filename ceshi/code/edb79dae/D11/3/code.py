# Variant of edb79dae
# Dimension: D11 — 灰色框架的结构 (meta)
# Change: 单一矩形灰色框架，框架线条均匀，内部为操作区域... → 框架颜色非灰色（多种框架颜色编码不同规则）

import itertools
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location

# Design analysis:
# spec:
#   core_change:
#     keep one rectangular 1-cell frame, but stop treating its color as inert decoration.
#     use two non-gray frame colors as a visible control code:
#       blue(1)  -> rotate every matched sprite 90 degrees clockwise before stamping
#       green(3) -> stamp the sprite without rotation
#   concrete_grid_effect:
#     inside the frame the placeholders are still solid NxN source-color blocks.
#     outside the frame there are still:
#       - one sprite template per source color
#       - one 3x4 source->target legend per source color
#     the new difference is that the output no longer always copies the outside sprite orientation.
#     instead, the frame color decides whether the sprite orientation is preserved or rotated.
#   behavior_trace:
#     original:
#       placeholder color 2 selects the outside color-2 sprite mask M and legend maps 2->7.
#       output pastes M at each color-2 placeholder and recolors it to 7.
#     variant:
#       if the frame is blue, the same placeholder still selects M and 2->7,
#       but the pasted shape is rot90(M, clockwise) recolored to 7.
#       if the frame is green, the pasted shape stays M recolored to 7.
#     difference:
#       the matching and recoloring pipeline is unchanged; only the frame color injects
#       a transformation stage between sprite lookup and sprite stamping.
#   complexity_calibration:
#     keep exactly two frame-rule categories so humans can infer the mapping from a few examples.
#     more categories would make the control code underdetermined in 2-4 ARC examples.
#     to keep the change visible, generator rejects sprites invariant under 90-degree rotation.
#   implicit_assumptions_audit:
#     _find_frame in the original solver already supports any single-color rectangular border,
#     so it remains valid. The fragile assumptions were elsewhere:
#       - generator hard-coded gray frame color 5
#       - solver treated frame color only as something to exclude from color sets
#       - output stamping assumed sprite orientation is always the template orientation
#     the new solver therefore adds a frame-color->rotation lookup before stamping.
#   placement_legality:
#     generator still uses square NxN placeholders/sprites, so a 90-degree rotation keeps the bbox size.
#     outside patches never use a frame color, preventing legend/sprite/frame ambiguity.
#     reserve placement uses an off-palette marker, so occupancy logic is independent of frame color.
#   visual_observability:
#     frame colors are directly visible and the only branch variable.
#     all sprite masks are forced to differ from their 90-degree rotation, so blue-frame outputs
#     visibly differ from green-frame outputs with the same sprite templates.
#   cleanliness:
#     no extra markers are added; the composition remains "frame + placeholders inside, templates + legends outside".
#     this keeps the variant legible and close to the source task.
#   color_inference:
#     color remapping remains supported by the existing 3x4 legends.
#     frame color does not participate in legend mapping and is never reused by sprites/targets.

FRAME_RULES = {
    1: 1,  # blue frame -> rotate 90 degrees clockwise
    3: 0,  # green frame -> keep orientation
}
FRAME_COLORS = tuple(sorted(FRAME_RULES))
RESERVE_MARK = 10


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


def _frame_turns(frame_col: int) -> int:
    assert frame_col in FRAME_RULES, f"unsupported frame color {frame_col}"
    return int(FRAME_RULES[frame_col])


def _apply_frame_rule(mask: np.ndarray, frame_col: int) -> np.ndarray:
    """Return the sprite mask after applying the frame-coded orientation rule."""
    turns = _frame_turns(frame_col)
    return np.rot90(mask, -turns).copy() if turns else mask.copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp recolored sprites into a framed region, with the frame color controlling sprite orientation.

    Input structure:
    - bg background, one rectangular non-gray frame, and solid colored placeholders inside that frame.
    - Outside the frame: one sprite template per placeholder color plus one compact legend that maps each source color to its final color.

    Rule:
    - Find the frame and the placeholder blocks inside it.
    - Use placeholder color to pick the matching outside sprite and use the legend to pick the final recolor.
    - Before stamping, apply the transformation encoded by the frame color: blue frame rotates sprites 90 degrees clockwise, green frame keeps them unchanged.

    Key insight:
    - The outside legend still explains only the recoloring, while the frame color itself explains the orientation change, so solving the task requires combining both cues.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    frame_col, (fr0, fc0, fr1, fc1) = _find_frame(g, bg)

    crop = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    h, w = crop.shape
    assert h >= 3 and w >= 3, f"frame too small for interior: {(h, w)}"

    inner_mask = np.zeros((h, w), dtype=bool)
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
            if bg in sub_colors:
                continue
            if len(sub_colors) != 2:
                continue
            cset = set(sub_colors)
            if ci["col"] not in cset or cj["col"] not in cset:
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

    sprite_masks: dict[int, np.ndarray] = {}
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
        for comp in all_candidates:
            if comp["id"] == base["id"]:
                continue
            r0, c0, r1, c1 = comp["bbox"]
            assert sr0 <= r0 and sr1 >= r1 and sc0 <= c0 and sc1 >= c1, (
                f"extra sprite component for color {src_col} is outside the base bbox"
            )
            pts_set.update(comp["pts"])

        base_mask = np.zeros((sr1 - sr0 + 1, sc1 - sc0 + 1), dtype=bool)
        for r, c in pts_set:
            base_mask[r - sr0, c - sc0] = True
        sprite_masks[src_col] = _apply_frame_rule(base_mask, frame_col)

    out = crop.copy()
    out[1:-1, 1:-1] = bg

    for src_col, (pr0, pc0, pr1, pc1) in placeholders:
        assert src_col in sprite_masks and src_col in color_map, (
            f"placeholder color {src_col} missing sprite or color mapping"
        )
        tgt_col = int(color_map[src_col])
        mask = sprite_masks[src_col]
        sh, sw = mask.shape
        assert (pr1 - pr0 + 1, pc1 - pc0 + 1) == (sh, sw), (
            f"placeholder size does not match transformed sprite for color {src_col}"
        )
        patch = np.full((sh, sw), bg, dtype=int)
        patch[mask] = tgt_col
        paste(out, patch, pr0, pc0, transparent=bg)

    return out


# --------------- generator (o5x) ---------------

# === CFG: tunables ===
SPRITE_N_RANGE = (3, 6)             # NxN sprite/placeholder size, exclusive upper
GRID_ROWS_RANGE = (2, 4)            # placeholder grid row count, exclusive upper
GRID_COLS_RANGE = (3, 5)            # placeholder grid col count, exclusive upper
TOTAL_BLOCKS_RANGE = (6, 10)        # total filled slots (inclusive)
MAX_EMPTY_SLOTS = 2                 # max empty slots in the grid
NUM_COLORS_RANGE = (2, 5)           # K = number of distinct placeholder colors (inclusive)
SPRITE_FILL_MIN = 0.5               # min sprite fill ratio
SPRITE_FILL_MAX = 0.9               # max sprite fill ratio (clamped to n*n-1)
MIN_FRAME_H = 13                    # minimum frame height
MIN_FRAME_W = 18                    # minimum frame width
MAX_FRAME_DIM = 21                  # maximum frame height or width
FRAME_MARGIN_SLACK = 4              # max extra padding inside frame beyond content
CANVAS_H_RANGE = (22, 31)           # final grid height range, exclusive upper
CANVAS_W_RANGE = (23, 31)           # final grid width range, exclusive upper
FRAME_POS_JITTER_PROB = 0.25        # probability of non-edge frame placement per axis
PATCH_PADDING = 1                   # min bg gap between placed patches
GEN_RETRIES = 300                   # outer retry budget
MASK_RETRIES = 200                  # retries for generating a visibly asymmetric sprite


def _legend(src: int, tgt: int) -> np.ndarray:
    """Build a 3x4 legend block mapping src color to tgt color."""
    return np.array(
        [[src, src, tgt, tgt], [src, src, src, tgt], [src, src, tgt, tgt]],
        dtype=int,
    )


def _sprite_mask(n: int) -> np.ndarray:
    """Generate an NxN binary mask whose 90-degree rotation is visibly different."""
    lo = max((n * n + 1) // 2, int(np.ceil(SPRITE_FILL_MIN * n * n)))
    hi = min(int(np.floor(SPRITE_FILL_MAX * n * n)), n * n - 1)

    for _ in range(MASK_RETRIES):
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

        if not np.array_equal(mask, np.rot90(mask, -1)):
            return mask

    raise RuntimeError(f"failed to generate asymmetric {n}x{n} sprite mask")


def _colored_sprite(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    """Render a binary mask as a colored sprite on bg."""
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = int(color)
    return out


def _content_extent(n: int, grid_rows: int, grid_cols: int) -> tuple[int, int]:
    """Content area needed for grid_rows x grid_cols of NxN blocks with 1-cell gaps."""
    return (grid_rows * n + grid_rows - 1, grid_cols * n + grid_cols - 1)


def _choose_structure() -> tuple[int, int, int, int, int] | None:
    """Pick (n, rows, cols, empty_count, total) satisfying frame size constraints."""
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
    """Choose which grid slots are empty, ensuring every row and column has at least one filled slot."""
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


def _make_bordered_frame(h: int, w: int, bg: int, frame_col: int) -> np.ndarray:
    """Create an h x w frame filled with bg and a 1-cell colored border."""
    frame = np.full((h, w), bg, dtype=int)
    frame[0, :] = frame[-1, :] = frame_col
    frame[:, 0] = frame[:, -1] = frame_col
    return frame


def _place_patch(
    grid: np.ndarray,
    reserve: np.ndarray,
    patch: np.ndarray,
    bg: int,
    *,
    transparent: bool,
) -> bool:
    """Find a free location for patch and paste it onto grid/reserve. Returns success."""
    blocker = np.full(patch.shape, RESERVE_MARK, dtype=int)
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

    frame_col = random.choice(FRAME_COLORS)
    bg = random.choice([c for c in range(10) if c not in FRAME_COLORS])
    usable = [c for c in range(10) if c not in ({bg} | set(FRAME_COLORS))]

    k = random.randint(*NUM_COLORS_RANGE)
    src_cols = random.sample(usable, k)
    tgt_cols = [random.choice([c for c in usable if c != src]) for src in src_cols]
    color_to_tgt = dict(zip(src_cols, tgt_cols))
    masks = {src: _sprite_mask(n) for src in src_cols}

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

    input_frame = _make_bordered_frame(frame_h, frame_w, bg, frame_col)
    output_frame = _make_bordered_frame(frame_h, frame_w, bg, frame_col)
    for (rr, cc), src in cell_colors.items():
        r0 = 1 + top_margin + rr * (n + 1)
        c0 = 1 + left_margin + cc * (n + 1)
        paste(input_frame, np.full((n, n), src, dtype=int), r0, c0)
        transformed = _apply_frame_rule(masks[src], frame_col)
        paste(output_frame, _colored_sprite(transformed, color_to_tgt[src], bg), r0, c0, transparent=bg)

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
    paste(reserve, np.full((frame_h, frame_w), RESERVE_MARK, dtype=int), fr0, fc0)

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
