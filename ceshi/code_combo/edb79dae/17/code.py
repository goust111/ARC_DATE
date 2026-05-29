# Combo variant of edb79dae (index 17)
# combo_spec: D6/2+D8/0
# Name: 不重着色+保留源精灵
#
# Design analysis for the fused variant:
# a) Concrete behavior trace:
#    Original task had two effective stages inside the frame: choose the outside sprite
#    by placeholder color, then recolor the placed sprite by reading a same-color legend.
#    This combo removes the second stage and also stops deleting the outside sprite.
#    Small example: if a blue 3x3 solid placeholder is inside the frame, a blue outside
#    sprite matches it, and the legend says blue->red, then the original output would
#    show only a red copy inside the frame crop. The fused output keeps the original full
#    canvas, leaves the outside blue sprite where it is, ignores the legend, and writes a
#    blue copy of the sprite into the blue placeholder slot.
# b) Complexity calibration:
#    The legend is still generated for every placeholder color so the scene still contains
#    the same distractor structure as the seed. The task is simpler than recoloring, but
#    not trivial, because the solver must infer that legends do not affect the output and
#    that the output expands from frame-only to full-canvas keep-source copying.
# c) Implicit-assumption audit against the seed solver:
#    - The original `color_map` assertion is no longer valid because legends are ignored.
#    - We still need legend detection, otherwise N=3 creates a collision: the 3x4 legend's
#      source-color component has a 3x3 bbox, which can be mistaken for the real sprite.
#    - The sprite-size matching assumption remains valid and the generator keeps one real
#      outside sprite per placeholder color.
#    - The old crop-only output assumption is invalid after keep-source; output must now be
#      the full grid with only the frame interior rewritten.
# d) Placement legality:
#    The generator still uses padded rectangle reservation, so frame, legends, and sprites
#    never overlap or leave the canvas. External objects remain separated, which keeps the
#    compact-legend detector reliable.
# e) Visual observability:
#    Across samples, the legend proposes visible alternative target colors, but the output
#    preserves the source colors and keeps the external sprites. Humans can see that the
#    frame is filled by copied same-color shapes while the legend is untouched noise.
# f) Visual cleanliness:
#    Only the frame interior changes in the output. Outside objects are preserved rather
#    than overprinted, so the scene remains partitioned into frame area, sprite templates,
#    and legend blocks with clear background gaps.
# g) Change inferability:
#    Every placeholder color has a repeated role: it appears inside the frame, on the real
#    sprite template, and inside the legend. Since legend target colors disagree with the
#    output, multiple samples expose that color change is not part of the rule.

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


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _collect_outside_components(
    g: np.ndarray,
    outside: np.ndarray,
    bg: int,
    frame_col: int,
) -> list[dict]:
    comps: list[dict] = []
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
    return comps


def _find_legend_ids(
    g: np.ndarray,
    comps: list[dict],
    bg: int,
    frame_col: int,
) -> set[int]:
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
            if ci["col"] in sub_colors and cj["col"] in sub_colors:
                legend_ids.add(int(ci["id"]))
                legend_ids.add(int(cj["id"]))
    return legend_ids


def _extract_sprite_coords(
    comps: list[dict],
    legend_ids: set[int],
    src_col: int,
    want_h: int,
    want_w: int,
) -> tuple[tuple[tuple[int, int], ...], int, int]:
    all_candidates = [c for c in comps if c["col"] == src_col and c["id"] not in legend_ids]
    assert all_candidates, f"no non-legend sprite found for color {src_col}"
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

    coords = tuple(sorted((r - sr0, c - sc0) for r, c in pts_set))
    return coords, sr1 - sr0 + 1, sc1 - sc0 + 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy same-colored sprite templates into a framed placeholder layout while leaving all outside objects visible.

    Input structure:
    - bg background, one rectangular gray frame, and solid colored placeholder blocks inside the frame.
    - Outside the frame: one real sprite template per placeholder color plus a compact two-color legend block for each color.

    Rule:
    - Find the frame and the placeholder blocks inside it.
    - For each placeholder color, locate the matching-color sprite template outside the frame.
    - Replace every placeholder block with a copy of that sprite in the same color.
    - Keep the full canvas, including the original outside sprites and the legend blocks, unchanged elsewhere.

    Key insight: The legend still looks informative, but in this variant it is a decoy. The actual transformation is just same-color copying from each outside sprite into the matching placeholder positions inside the frame.
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
    comps = _collect_outside_components(g, outside, bg, frame_col)
    legend_ids = _find_legend_ids(g, comps, bg, frame_col)

    sprite_coords: dict[int, tuple[tuple[tuple[int, int], ...], int, int]] = {}
    for src_col in sorted(placeholder_colors):
        want_h, want_w = phw[src_col]
        sprite_coords[src_col] = _extract_sprite_coords(comps, legend_ids, src_col, want_h, want_w)

    out = g.copy()
    out[fr0 + 1 : fr1, fc0 + 1 : fc1] = bg

    for src_col, (pr0, pc0, pr1, pc1) in placeholders:
        coords, sh, sw = sprite_coords[src_col]
        assert (pr1 - pr0 + 1, pc1 - pc0 + 1) == (sh, sw), (
            f"placeholder size does not match sprite for color {src_col}"
        )
        off_r = fr0 + pr0
        off_c = fc0 + pc0
        for dr, dc in coords:
            out[off_r + dr, off_c + dc] = int(src_col)

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
GEN_RETRIES = 300


def _legend(src: int, tgt: int) -> np.ndarray:
    """Build a 3x4 legend block that is visually plausible but unused by the rule."""
    return np.array(
        [
            [src, src, tgt, tgt],
            [src, src, src, tgt],
            [src, src, tgt, tgt],
        ],
        dtype=int,
    )


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


def _colored_sprite(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = int(color)
    return out


def _content_extent(n: int, grid_rows: int, grid_cols: int) -> tuple[int, int]:
    return grid_rows * n + grid_rows - 1, grid_cols * n + grid_cols - 1


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
    rows: int,
    cols: int,
    empty_count: int,
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


def _copy_output_frame(
    frame_h: int,
    frame_w: int,
    bg: int,
    cell_colors: dict[tuple[int, int], int],
    masks: dict[int, np.ndarray],
    n: int,
    top_margin: int,
    left_margin: int,
) -> np.ndarray:
    out = _make_bordered_frame(frame_h, frame_w, bg)
    for (rr, cc), src in cell_colors.items():
        r0 = 1 + top_margin + rr * (n + 1)
        c0 = 1 + left_margin + cc * (n + 1)
        paste(out, _colored_sprite(masks[src], src, bg), r0, c0, transparent=bg)
    return out


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
    cell_colors: dict[tuple[int, int], int] = {}
    for slot, col in zip(filled_slots[:k], src_cols):
        cell_colors[slot] = col
    for slot in filled_slots[k:]:
        cell_colors[slot] = random.choice(src_cols)

    input_frame = _make_bordered_frame(frame_h, frame_w, bg)
    for (rr, cc), src in cell_colors.items():
        r0 = 1 + top_margin + rr * (n + 1)
        c0 = 1 + left_margin + cc * (n + 1)
        paste(input_frame, np.full((n, n), src, dtype=int), r0, c0)
    output_frame = _copy_output_frame(frame_h, frame_w, bg, cell_colors, masks, n, top_margin, left_margin)

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

    expected = grid.copy()
    paste(expected, output_frame, fr0, fc0)

    try:
        if not np.array_equal(main(grid), expected):
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
