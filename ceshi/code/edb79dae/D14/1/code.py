# Variant of edb79dae
# Dimension: D14 — 精灵放置时的对齐方式 (meta)
# Change: 精灵形状以目标像素位置为中心放置，替换该像素及周围区域... → 精灵对齐方式由目标像素在框架内的位置决定

# Design analysis for the D14 variant
# -----------------------------------
# spec:
#   core_change:
#     original_effective_solver_behavior:
#       The baseline solver replaces each solid NxN placeholder block by aligning the
#       sprite bbox's top-left corner to the placeholder bbox's top-left corner.
#       That makes placement independent of where the placeholder sits inside the frame.
#     variant_behavior:
#       The framed region now contains single-pixel anchors instead of solid NxN blocks.
#       A source-colored anchor pixel selects a sprite/color mapping exactly as before,
#       but the anchor's quadrant inside the frame determines which sprite corner is
#       attached to that pixel:
#         - top-left quadrant -> sprite top-left corner sits on the anchor
#         - top-right quadrant -> sprite top-right corner sits on the anchor
#         - bottom-left quadrant -> sprite bottom-left corner sits on the anchor
#         - bottom-right quadrant -> sprite bottom-right corner sits on the anchor
#       This means the same sprite can expand down-right, down-left, up-right, or
#       up-left depending only on anchor position.
#   behavior_trace:
#     original_local_step:
#       With a 4x4 placeholder block at (r0:r0+3, c0:c0+3), the old code used
#       off_r = r0 and off_c = c0 for every placement, so all copies landed with
#       the same bbox alignment regardless of location.
#     variant_local_step:
#       With a 4x4 sprite and a single anchor at (r, c):
#         - if (r, c) is in the top-right region, off_r = r and off_c = c - 3
#         - if (r, c) is in the bottom-left region, off_r = r - 3 and off_c = c
#       The anchor is consumed and the sprite occupies a different side of the anchor.
#     visible_difference:
#       Two identical-color anchors in different quadrants produce the same shape but
#       mirrored in where the occupied bbox lies relative to the marked pixel.
#   complexity_calibration:
#     why_single_pixels:
#       If placeholders stayed NxN, changing corner alignment would be invisible because
#       sprite bbox and placeholder bbox have identical size, so all corner alignments
#       collapse to the same placement. Single-pixel anchors make D14 observable.
#     controlled_freedom:
#       The rule only has four cases (one per quadrant), and each sample uses repeated
#       colors and multiple anchors so the solver is not forced to infer many unrelated
#       exceptions.
#   implicit_assumption_review:
#     kept_from_original:
#       - exactly one rectangular frame exists
#       - each placeholder color has one external sprite template
#       - legend detection still relies on compact two-color overlapping glyphs
#     changed_in_main:
#       - placeholder components can no longer be assumed to reveal sprite size
#       - size matching between placeholder bbox and sprite bbox must be removed
#       - placement can no longer blindly use placeholder top-left
#     new_solver_assumptions_and_generator_guarantees:
#       - every in-frame placeholder component has area 1; generator places isolated
#         anchor pixels with at least one-cell slot gaps
#       - each external sprite is a single connected component whose bbox is exactly NxN;
#         generator makes masks connected and touches all four sprite edges
#       - frame midlines must separate slot groups cleanly so quadrant classification by
#         absolute position is stable; generator searches margins/sizes so H//2 and W//2
#         land on the gaps between slot groups
#       - placements must not overlap or touch the frame border; generator uses the same
#         slot bbox as the final sprite bbox, so the chosen corner anchor always yields a
#         legal in-frame paste
#   legality_and_visual_cleanliness:
#     layout:
#       Sprites still occupy regular NxN slots with one background gap, so output remains
#       tidy. Inputs are even cleaner: each slot shows only one colored anchor pixel.
#     separations:
#       External sprites/legends are placed with background padding exactly as before.
#       Inside the frame, anchors are one per slot, preventing ambiguous multi-pixel
#       clusters or overlaps.
#   human_observability:
#     The quadrant rule is visible from 2-3 examples because anchors in different parts
#     of the frame make the same shape grow from different corners. No hidden ordering by
#     color id is used. The legend still gives explicit recoloring evidence, while the
#     frame layout gives explicit spatial evidence for D14.

import itertools
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


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


def _anchor_mode(r: int, c: int, H: int, W: int) -> str:
    """Map an in-frame anchor position to one of TL/TR/BL/BR using frame midlines."""
    vertical = "T" if int(r) < H // 2 else "B"
    horizontal = "L" if int(c) < W // 2 else "R"
    return vertical + horizontal


def _placement_from_anchor(r: int, c: int, sh: int, sw: int, mode: str) -> tuple[int, int]:
    """Return sprite bbox top-left when the anchor touches the chosen sprite corner."""
    if mode == "TL":
        return int(r), int(c)
    if mode == "TR":
        return int(r), int(c) - int(sw) + 1
    if mode == "BL":
        return int(r) - int(sh) + 1, int(c)
    if mode == "BR":
        return int(r) - int(sh) + 1, int(c) - int(sw) + 1
    raise AssertionError(f"unknown anchor mode {mode}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a framed sprite mosaic from single-pixel anchors, then recolor it with the legend.

    Input structure:
    - A background-filled canvas contains one rectangular frame and isolated colored anchor pixels inside it.
    - Outside the frame sit one sprite template per anchor color and one compact two-color legend per anchor color.

    Rule:
    - Detect the frame, its in-frame anchor colors, the matching external sprites, and the legend-based recoloring map.
    - For each anchor pixel, choose the matching sprite by color.
    - Stamp that sprite so the anchor touches a quadrant-dependent sprite corner: top-left, top-right, bottom-left, or bottom-right according to the anchor's region inside the frame.
    - Recolor every stamped sprite to the legend's target color.

    Key insight:
    The legend still controls recoloring, but D14 changes placement: the anchor is no longer a whole placeholder block. Its absolute position inside the frame tells you which sprite corner should lock onto it.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    frame_col, (fr0, fc0, fr1, fc1) = _find_frame(g, bg)

    crop = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    H, W = crop.shape
    assert H >= 3 and W >= 3, f"frame too small for interior: {(H, W)}"

    inner_mask = np.zeros((H, W), dtype=bool)
    inner_mask[1:-1, 1:-1] = True
    placeholder_colors = set(map(int, np.unique(crop[inner_mask]))) - {bg, frame_col}
    assert placeholder_colors, "no placeholder anchors found inside frame"

    placeholders: list[tuple[int, int, int]] = []
    for col in sorted(placeholder_colors):
        for pts in components_mask((crop == col) & inner_mask, connectivity=4):
            assert len(pts) == 1, f"placeholder color {col} must appear as isolated single-pixel anchors"
            r, c = map(int, pts[0])
            placeholders.append((int(col), r, c))

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

            assert src in placeholder_colors, f"legend source color {src} is not an in-frame anchor color"
            assert src not in color_map, f"duplicate legend mapping for anchor color {src}"
            color_map[src] = int(tgt)
            legend_ids.add(ci["id"])
            legend_ids.add(cj["id"])
    assert set(color_map) == placeholder_colors, (
        f"legend mappings {sorted(color_map)} do not match anchors {sorted(placeholder_colors)}"
    )

    sprite_coords: dict[int, tuple[tuple[tuple[int, int], ...], int, int]] = {}
    for src_col in sorted(placeholder_colors):
        candidates = [c for c in comps if c["col"] == src_col and c["id"] not in legend_ids]
        assert len(candidates) == 1, f"expected one non-legend sprite for color {src_col}, found {len(candidates)}"
        comp = candidates[0]
        sr0, sc0, sr1, sc1 = comp["bbox"]
        coords = tuple(sorted((r - sr0, c - sc0) for r, c in comp["pts"]))
        sprite_coords[src_col] = (coords, sr1 - sr0 + 1, sc1 - sc0 + 1)

    out = crop.copy()
    out[1:-1, 1:-1] = bg

    for src_col, pr, pc in placeholders:
        assert src_col in sprite_coords and src_col in color_map, f"anchor color {src_col} missing sprite or color mapping"
        tgt_col = int(color_map[src_col])
        coords, sh, sw = sprite_coords[src_col]
        mode = _anchor_mode(pr, pc, H, W)
        off_r, off_c = _placement_from_anchor(pr, pc, sh, sw, mode)
        assert 1 <= off_r and 1 <= off_c and off_r + sh <= H - 1 and off_c + sw <= W - 1, (
            f"sprite for color {src_col} does not fit for anchor mode {mode}"
        )
        for dr, dc in coords:
            rr, cc = off_r + dr, off_c + dc
            assert out[rr, cc] == bg, "overlapping sprite placements inside frame"
            out[rr, cc] = tgt_col

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


def _legend(src: int, tgt: int) -> np.ndarray:
    """Build a 3x4 legend block mapping src color to tgt color."""
    return np.array(
        [[src, src, tgt, tgt], [src, src, src, tgt], [src, src, tgt, tgt]],
        dtype=int,
    )


def _random_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    """Random Manhattan path used to make connected sprites with full-edge coverage."""
    r, c = a
    tr, tc = b
    out = [(r, c)]
    while (r, c) != (tr, tc):
        moves = []
        if r != tr:
            moves.append((r + (1 if tr > r else -1), c))
        if c != tc:
            moves.append((r, c + (1 if tc > c else -1)))
        r, c = random.choice(moves)
        out.append((r, c))
    return out


def _sprite_mask(n: int) -> np.ndarray:
    """Generate a connected NxN sprite whose bbox is exactly NxN."""
    lo = max((n * n + 1) // 2, int(np.ceil(SPRITE_FILL_MIN * n * n)))
    hi = min(int(np.floor(SPRITE_FILL_MAX * n * n)), n * n - 1)

    for _ in range(200):
        mask = np.zeros((n, n), dtype=bool)
        start = (random.randrange(n), random.randrange(n))
        edge_targets = [
            (0, random.randrange(n)),
            (random.randrange(n), n - 1),
            (n - 1, random.randrange(n)),
            (random.randrange(n), 0),
        ]
        current = start
        mask[current] = True
        for target in random.sample(edge_targets, len(edge_targets)):
            for cell in _random_path(current, target):
                mask[cell] = True
            current = target

        if int(mask.sum()) > hi:
            continue

        target_fill = random.randint(max(lo, int(mask.sum())), hi)
        while int(mask.sum()) < target_fill:
            dilated = np.zeros_like(mask)
            dilated[1:, :] |= mask[:-1, :]
            dilated[:-1, :] |= mask[1:, :]
            dilated[:, 1:] |= mask[:, :-1]
            dilated[:, :-1] |= mask[:, 1:]
            frontier = list(zip(*np.where(dilated & ~mask)))
            assert frontier, "frontier unexpectedly empty for connected sprite growth"
            r, c = random.choice(frontier)
            mask[r, c] = True

        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if rows[0] == 0 and rows[-1] == n - 1 and cols[0] == 0 and cols[-1] == n - 1:
            return mask

    raise RuntimeError(f"failed to generate full-bbox sprite of size {n}")


def _colored_sprite(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = int(color)
    return out


def _content_extent(n: int, grid_rows: int, grid_cols: int) -> tuple[int, int]:
    return (grid_rows * n + grid_rows - 1, grid_cols * n + grid_cols - 1)


def _slot_top_left(top_margin: int, left_margin: int, n: int, rr: int, cc: int) -> tuple[int, int]:
    return (1 + top_margin + rr * (n + 1), 1 + left_margin + cc * (n + 1))


def _slot_mode(rr: int, cc: int, row_split: int, col_split: int) -> str:
    return ("T" if rr < row_split else "B") + ("L" if cc < col_split else "R")


def _anchor_cell(r0: int, c0: int, n: int, mode: str) -> tuple[int, int]:
    if mode == "TL":
        return r0, c0
    if mode == "TR":
        return r0, c0 + n - 1
    if mode == "BL":
        return r0 + n - 1, c0
    if mode == "BR":
        return r0 + n - 1, c0 + n - 1
    raise AssertionError(f"unknown slot mode {mode}")


def _choose_structure() -> tuple[int, int, int, int, int] | None:
    total = random.randint(*TOTAL_BLOCKS_RANGE)
    options = []
    for n in range(*SPRITE_N_RANGE):
        for rows in range(*GRID_ROWS_RANGE):
            for cols in range(*GRID_COLS_RANGE):
                if rows < 2 or cols < 2:
                    continue
                ch, cw = _content_extent(n, rows, cols)
                base_fh = ch + 4
                base_fw = cw + 4
                slots = rows * cols
                empty = slots - total
                if base_fh <= MAX_FRAME_DIM and base_fw <= MAX_FRAME_DIM and 0 <= empty <= MAX_EMPTY_SLOTS:
                    options.append((n, rows, cols, empty, total))
    return random.choice(options) if options else None


def _choose_layout(
    n: int, rows: int, cols: int
) -> tuple[int, int, int, int, int, int, int, int] | None:
    """Choose frame size, margins, and slot split so frame midlines fall on slot gaps."""
    ch, cw = _content_extent(n, rows, cols)
    min_fh = max(MIN_FRAME_H, ch + 4)
    min_fw = max(MIN_FRAME_W, cw + 4)

    options = []
    for row_split in range(1, rows):
        for col_split in range(1, cols):
            for top_margin in range(1, 9):
                for bottom_margin in range(1, 9):
                    frame_h = ch + top_margin + bottom_margin + 2
                    if not (min_fh <= frame_h <= MAX_FRAME_DIM):
                        continue
                    if frame_h // 2 != top_margin + row_split * (n + 1):
                        continue
                    for left_margin in range(1, 9):
                        for right_margin in range(1, 9):
                            frame_w = cw + left_margin + right_margin + 2
                            if not (min_fw <= frame_w <= MAX_FRAME_DIM):
                                continue
                            if frame_w // 2 != left_margin + col_split * (n + 1):
                                continue
                            extra_h = frame_h - (ch + 4)
                            extra_w = frame_w - (cw + 4)
                            if extra_h > FRAME_MARGIN_SLACK or extra_w > FRAME_MARGIN_SLACK:
                                continue
                            options.append(
                                (
                                    frame_h,
                                    frame_w,
                                    top_margin,
                                    bottom_margin,
                                    left_margin,
                                    right_margin,
                                    row_split,
                                    col_split,
                                )
                            )
    return random.choice(options) if options else None


def _choose_empty_slots(
    rows: int,
    cols: int,
    empty_count: int,
    row_split: int,
    col_split: int,
) -> tuple[set[tuple[int, int]], list[tuple[int, int]]]:
    """
    Choose empty slots while keeping every row/column active and every quadrant represented.
    """
    slots = [(r, c) for r in range(rows) for c in range(cols)]
    valid = []
    for combo in itertools.combinations(slots, empty_count):
        empty = set(combo)
        filled = [s for s in slots if s not in empty]
        if len({r for r, _ in filled}) != rows or len({c for _, c in filled}) != cols:
            continue
        quadrants = {_slot_mode(r, c, row_split, col_split) for r, c in filled}
        if quadrants != {"TL", "TR", "BL", "BR"}:
            continue
        valid.append(empty)
    if not valid:
        raise ValueError("no empty-slot pattern keeps all quadrants represented")
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

    layout = _choose_layout(n, rows, cols)
    if layout is None:
        return None
    frame_h, frame_w, top_margin, _bottom_margin, left_margin, _right_margin, row_split, col_split = layout

    bg = random.choice([c for c in range(10) if c != FRAME_COL])
    usable = [c for c in range(10) if c not in (bg, FRAME_COL)]

    k = random.randint(*NUM_COLORS_RANGE)
    src_cols = random.sample(usable, k)
    tgt_cols = [random.choice([c for c in usable if c != src]) for src in src_cols]
    color_to_tgt = dict(zip(src_cols, tgt_cols))
    masks = {src: _sprite_mask(n) for src in src_cols}

    _, filled_slots = _choose_empty_slots(rows, cols, empty_count, row_split, col_split)
    cell_colors = {}
    for slot, col in zip(filled_slots[:k], src_cols):
        cell_colors[slot] = col
    for slot in filled_slots[k:]:
        cell_colors[slot] = random.choice(src_cols)

    input_frame = _make_bordered_frame(frame_h, frame_w, bg)
    output_frame = _make_bordered_frame(frame_h, frame_w, bg)
    quadrant_counts = {"TL": 0, "TR": 0, "BL": 0, "BR": 0}

    for (rr, cc), src in cell_colors.items():
        r0, c0 = _slot_top_left(top_margin, left_margin, n, rr, cc)
        mode = _slot_mode(rr, cc, row_split, col_split)
        quadrant_counts[mode] += 1
        ar, ac = _anchor_cell(r0, c0, n, mode)
        input_frame[ar, ac] = src
        paste(output_frame, _colored_sprite(masks[src], color_to_tgt[src], bg), r0, c0, transparent=bg)

        check_mode = _anchor_mode(ar, ac, frame_h, frame_w)
        assert check_mode == mode, f"slot mode mismatch: expected {mode}, got {check_mode}"

    assert all(v > 0 for v in quadrant_counts.values()), f"quadrant coverage failure: {quadrant_counts}"

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
    raise RuntimeError("failed to generate valid edb79dae D14 input")
