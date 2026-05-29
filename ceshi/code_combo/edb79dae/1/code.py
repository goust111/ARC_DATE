# Combo variant of edb79dae (index 1)
# combo_spec: D1/0+D2/0
# Name: 形状匹配+面积图例双重几何推理

import itertools
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the combo variant
# -------------------------------------
# combo_spec: D1/0 + D2/0
# summary:
#   - Keep the original scene scaffold: one gray rectangular frame, repeated placement slots inside it,
#     and external helper objects placed on the background.
#   - Replace the old "inside color chooses sprite" step with a geometry-only code:
#     each occupied slot contains a tiny 2-3 pixel marker anchored at the slot's top-left corner.
#     Each external sprite carries the same tiny marker in its own top-left 2x2 corner, so the solver
#     must identify the sprite by matching the micro-shape, not by comparing colors.
#   - Replace the old legend lookup with D2-style counting cards:
#     a fixed 5x6 opaque legend card uses one shared code color repeated N times, where N is the
#     matched sprite's foreground area; the other color on that card is the repaint target.
# behavior_trace:
#   - Original selected baseline:
#       1. a solid NxN placeholder color inside the frame chooses the same-color sprite outside
#       2. the legend mentions that same source color and supplies the target color
#   - This combo:
#       1. a tiny L/dominolike marker inside the frame identifies a sprite whose top-left corner
#          shows the same tiny marker; marker color is irrelevant
#       2. after the sprite is identified, the solver counts that sprite's pixels
#       3. the count points to the unique 5x6 legend card with the same number of shared code-color cells
#       4. the sprite is stamped into the slot and recolored to that legend's target color
#   - Local example:
#       an orange 3-pixel corner marker inside the frame does not map to any orange object.
#       Instead it matches the external green sprite whose top-left 2x2 signature is the same corner.
#       If that sprite has area 14, the solver must then find the 14-dot legend card and use its partner color.
# complexity_calibration:
#   - K is kept at 2-3 source types so the counting legend remains readable and enough palette is left
#     to separate source colors from the non-semantic marker colors.
#   - Every source type appears at least twice inside the frame. This makes repeated geometry visible and
#     gives the solver multiple confirmations that marker color is not the deciding factor.
#   - Marker colors are chosen from a small palette disjoint from sprite source colors, and at least one
#     marker color is reused by multiple source types. This blocks regression to the original color match.
#   - Sprite areas are forced to be distinct. Otherwise the D2 counting legend would be ambiguous or inert.
# implicit_assumption_audit:
#   - _find_frame still assumes exactly one rectangular border component; the generator still builds one.
#   - Interior extraction can no longer assume placeholder bbox == sprite bbox. The new contract is instead:
#       each interior marker's bbox top-left equals the destination slot anchor for the full sprite.
#   - Sprite identification can no longer use color. The solver now assumes that the top-left 2x2 corner of
#     every sprite contains a unique connected 2-3 pixel signature drawn from a small library.
#   - Legend decoding can no longer look for source colors or overlapping glyphs. The solver now assumes that
#     the only opaque 5x6 outside rectangles are legend cards, they all share one code color, and their code
#     color counts exactly match the set of sprite areas.
#   - Because step 1 and step 2 are both color-free, the generator must enforce both uniqueness axes:
#       marker signatures are unique across sprites, and sprite areas are unique across sprites.
# placement_legality:
#   - Interior markers occupy only the top-left of each slot, while output sprites still use the full NxN slot.
#     The slot grid keeps 1-cell background gaps, so stamping cannot overlap or cross the frame border.
#   - External sprites are transparent patches and legend cards are opaque rectangles, all placed with the
#     reserve-grid + padding method, so objects remain separated and parseable.
# visual_observability_and_cleanliness:
#   - The input visibly differs from the original task: the frame contains tiny sparse marks rather than solid blocks,
#     and the legend cards share a repeated count color instead of reusing sprite colors.
#   - The outside world still has only two helper families: large sprites and uniform legend cards.
#     No extra connector objects are added, which keeps the scene readable.
#   - The solver's two cues are visually distinct in kind: first compare tiny shapes, then count pixels.
# inference_guard:
#   - After implementation we verify more than "main does not crash". We also measure whether generated samples
#     actually break the old color-based stage and the old source-color legend stage:
#       marker colors must be non-semantic, and legend cards must be count-based with a shared code color.


FRAME_COL = 5

SPRITE_N_RANGE = (4, 6)  # use 4-5 so the micro marker is visibly smaller than the sprite
GRID_ROWS_RANGE = (2, 4)
GRID_COLS_RANGE = (3, 5)
TOTAL_BLOCKS_RANGE = (6, 10)
MAX_EMPTY_SLOTS = 2
NUM_TYPES_RANGE = (2, 3)
SPRITE_FILL_MIN = 0.45
SPRITE_FILL_MAX = 0.84
MIN_FRAME_H = 13
MIN_FRAME_W = 18
MAX_FRAME_DIM = 21
FRAME_MARGIN_SLACK = 4
CANVAS_H_RANGE = (22, 31)
CANVAS_W_RANGE = (23, 31)
FRAME_POS_JITTER_PROB = 0.25
PATCH_PADDING = 1
GEN_RETRIES = 300

LEGEND_CODE_H = 5
LEGEND_CODE_W = 5
LEGEND_H = LEGEND_CODE_H
LEGEND_W = LEGEND_CODE_W + 1
LEGEND_ORDER = tuple((r, c) for r in range(LEGEND_CODE_H) for c in range(LEGEND_CODE_W))

MARKER_LIBRARY = (
    np.array([[1, 1], [0, 0]], dtype=bool),
    np.array([[1, 0], [1, 0]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
)


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


def _shape_signature(
    pts: list[tuple[int, int]] | np.ndarray,
    bbox: tuple[int, int, int, int],
) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = bbox
    rel = tuple(sorted((int(r - r0), int(c - c0)) for r, c in pts))
    return (int(r1 - r0 + 1), int(c1 - c0 + 1), rel)


def _mask_signature(mask: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    return _shape_signature([(int(r), int(c)) for r, c in pts], (r0, c0, r1, c1))


def _bbox_inside(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> bool:
    ir0, ic0, ir1, ic1 = inner
    or0, oc0, or1, oc1 = outer
    return or0 <= ir0 <= ir1 <= or1 and oc0 <= ic0 <= ic1 <= oc1


def _sprite_corner_signature(pts: list[tuple[int, int]], bbox: tuple[int, int, int, int]):
    sr0, sc0, _sr1, _sc1 = bbox
    corner = [(r - sr0, c - sc0) for r, c in pts if r - sr0 < 2 and c - sc0 < 2]
    assert corner, "sprite is missing its top-left micro marker"
    rr = [r for r, _ in corner]
    cc = [c for _, c in corner]
    cbbox = (min(rr), min(cc), max(rr), max(cc))
    return _shape_signature(corner, cbbox)


def _extract_outside_components(
    g: np.ndarray,
    outside: np.ndarray,
    bg: int,
    frame_col: int,
) -> list[dict]:
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
    return comps


def _decode_area_legend(
    g: np.ndarray,
    outside: np.ndarray,
    bg: int,
    frame_col: int,
) -> tuple[dict[int, int], list[tuple[int, int, int, int]], int]:
    legend_candidates = []
    dense_outside = outside & (g != bg) & (g != frame_col)
    for pts in components_mask(dense_outside, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts)
        if (r1 - r0 + 1, c1 - c0 + 1) != (LEGEND_H, LEGEND_W):
            continue
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(sub == bg) or np.any(sub == frame_col):
            continue
        colors_here = set(map(int, np.unique(sub)))
        if len(colors_here) != 2:
            continue
        legend_candidates.append({"bbox": (r0, c0, r1, c1), "colors": colors_here, "sub": sub.copy()})

    assert legend_candidates, "no legend cards found"
    common_colors = set.intersection(*(cand["colors"] for cand in legend_candidates))
    assert len(common_colors) == 1, f"expected one shared legend code color, found {sorted(common_colors)}"
    code_col = int(next(iter(common_colors)))

    area_to_tgt: dict[int, int] = {}
    legend_boxes = []
    for cand in legend_candidates:
        code_area = int(np.sum(cand["sub"] == code_col))
        tgt_colors = [c for c in cand["colors"] if c != code_col]
        assert len(tgt_colors) == 1, "legend card should contain exactly one target color"
        tgt = int(tgt_colors[0])
        assert code_area not in area_to_tgt, f"duplicate legend count {code_area}"
        area_to_tgt[code_area] = tgt
        legend_boxes.append(cand["bbox"])
    return area_to_tgt, legend_boxes, code_col


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill a framed slot layout by matching tiny interior shape markers to external sprites, then recolor by area-count legend cards.

    Input structure:
    - One background color, one gray rectangular frame, and sparse 2-3 pixel monochrome markers inside the frame.
      Each marker sits at the top-left corner of an otherwise empty sprite slot.
    - Outside the frame are larger sprite templates and several opaque 5x6 legend cards that all share one repeated code color.

    Rule:
    - Find every interior marker component and record both its micro-shape and its top-left anchor position.
    - Decode the 5x6 legend cards outside the frame into area -> target-color mappings by counting how many cells use the shared code color.
    - For each external sprite, read the tiny signature in its own top-left 2x2 corner; match that signature to the unique interior marker shape.
    - Stamp the matched sprite into the frame at the marker's anchor, then recolor it using the legend entry whose code-color count equals the sprite's total area.

    Key insight: Both matching stages have been severed from source color. The first link is geometric but local (tiny marker shape),
    and the second link is numeric (foreground area count), so solving requires chaining two different non-color cues.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    frame_col, (fr0, fc0, fr1, fc1) = _find_frame(g, bg)
    crop = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    H, W = crop.shape
    assert H >= 3 and W >= 3, f"frame too small for interior: {(H, W)}"

    inner_mask = np.zeros((H, W), dtype=bool)
    inner_mask[1:-1, 1:-1] = True

    markers = []
    marker_colors = set(map(int, np.unique(crop[inner_mask]))) - {bg, frame_col}
    for col in sorted(marker_colors):
        for pts in components_mask((crop == col) & inner_mask, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            pts_tuples = [(int(r), int(c)) for r, c in pts]
            markers.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "sig": _shape_signature(pts_tuples, (r0, c0, r1, c1)),
                }
            )
    assert markers, "no interior micro markers found"

    outside = np.ones(g.shape, dtype=bool)
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False

    area_to_tgt, legend_boxes, code_col = _decode_area_legend(g, outside, bg, frame_col)
    comps = _extract_outside_components(g, outside, bg, frame_col)

    sprites_by_sig: dict[tuple[int, int, tuple[tuple[int, int], ...]], dict] = {}
    sprite_areas = set()
    for comp in comps:
        if any(_bbox_inside(comp["bbox"], lb) for lb in legend_boxes):
            continue
        sig = _sprite_corner_signature(comp["pts"], comp["bbox"])
        assert sig not in sprites_by_sig, "duplicate sprite micro signature outside the frame"
        sr0, sc0, sr1, sc1 = comp["bbox"]
        coords = tuple(sorted((r - sr0, c - sc0) for r, c in comp["pts"]))
        sprites_by_sig[sig] = {
            "coords": coords,
            "shape": (sr1 - sr0 + 1, sc1 - sc0 + 1),
            "area": len(coords),
            "color": int(comp["col"]),
        }
        sprite_areas.add(len(coords))

    assert sprites_by_sig, "no sprite templates found outside the frame"
    assert set(area_to_tgt) == sprite_areas, (
        f"legend counts {sorted(area_to_tgt)} do not match sprite areas {sorted(sprite_areas)}"
    )
    assert code_col not in {info["color"] for info in sprites_by_sig.values()}, "legend code color should not be a sprite color"

    out = crop.copy()
    out[1:-1, 1:-1] = bg

    for marker in markers:
        sig = marker["sig"]
        assert sig in sprites_by_sig, "interior marker shape has no matching external sprite"
        sprite = sprites_by_sig[sig]
        tgt_col = int(area_to_tgt[sprite["area"]])
        pr0, pc0, _pr1, _pc1 = marker["bbox"]
        sh, sw = sprite["shape"]
        assert pr0 + sh <= H and pc0 + sw <= W, "stamped sprite would leave the frame interior"
        for dr, dc in sprite["coords"]:
            out[pr0 + dr, pc0 + dc] = tgt_col

    return out


def _legend(area: int, tgt: int, code_col: int) -> np.ndarray:
    patch = np.full((LEGEND_H, LEGEND_W), int(tgt), dtype=int)
    for r, c in LEGEND_ORDER[:area]:
        patch[r, c] = int(code_col)
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


def _sprite_area_bounds(n: int) -> tuple[int, int]:
    lo = max(2 * n - 1, int(np.ceil(SPRITE_FILL_MIN * n * n)))
    hi = min(int(np.floor(SPRITE_FILL_MAX * n * n)), n * n - 1)
    return lo, hi


def _sprite_mask(n: int, target: int, marker: np.ndarray) -> np.ndarray:
    lo, hi = _sprite_area_bounds(n)
    assert lo <= target <= hi, f"target area {target} outside [{lo}, {hi}]"

    mask = np.zeros((n, n), dtype=bool)
    mask[:2, :2] = marker
    allowed = np.ones((n, n), dtype=bool)
    allowed[:2, :2] |= marker
    allowed[:2, :2] &= marker

    marker_pts = list(zip(*np.where(marker)))
    cur_r, cur_c = max(marker_pts, key=lambda rc: (rc[0] + rc[1], rc[0], rc[1]))
    while cur_r < n - 1 or cur_c < n - 1:
        options = []
        if cur_r < n - 1 and allowed[cur_r + 1, cur_c]:
            options.append((cur_r + 1, cur_c))
        if cur_c < n - 1 and allowed[cur_r, cur_c + 1]:
            options.append((cur_r, cur_c + 1))
        assert options, "path construction hit a forbidden corner cell"
        cur_r, cur_c = random.choice(options)
        mask[cur_r, cur_c] = True

    while int(mask.sum()) < target:
        frontier = set()
        for r, c in zip(*np.where(mask)):
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < n and 0 <= nc < n and allowed[nr, nc] and not mask[nr, nc]:
                    frontier.add((nr, nc))
        assert frontier, "sprite growth ran out of frontier cells"
        rr, cc = random.choice(sorted(frontier))
        mask[rr, cc] = True

    assert mask[0, 0], "marker anchor must stay occupied"
    assert mask[-1, :].any() and mask[:, -1].any(), "sprite must span the full bbox"
    assert np.all(mask[:2, :2] == marker), "sprite corner signature drifted"
    return mask


def _colored_sprite(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    struct = _choose_structure()
    if struct is None:
        return None
    n, rows, cols, empty_count, total = struct

    bg = random.choice([c for c in range(10) if c != FRAME_COL])
    legend_code_col = random.choice([c for c in range(10) if c not in (bg, FRAME_COL)])
    role_palette = [c for c in range(10) if c not in (bg, FRAME_COL, legend_code_col)]

    lo, hi = _sprite_area_bounds(n)
    possible_areas = list(range(lo, hi + 1))
    max_k = min(NUM_TYPES_RANGE[1], len(MARKER_LIBRARY), len(possible_areas), total // 2, len(role_palette) // 2)
    if max_k < NUM_TYPES_RANGE[0]:
        return None
    k = random.randint(NUM_TYPES_RANGE[0], max_k)

    src_cols = random.sample(role_palette, k)
    tgt_pool = [c for c in role_palette if c not in src_cols]
    if len(tgt_pool) < k:
        return None
    tgt_cols = random.sample(tgt_pool, k)
    marker_pool = [c for c in tgt_pool]
    if len(marker_pool) < 2:
        return None
    marker_palette = random.sample(marker_pool, 2)

    marker_patterns = random.sample(list(MARKER_LIBRARY), k)
    areas = random.sample(possible_areas, k)

    color_to_tgt = dict(zip(src_cols, tgt_cols))
    color_to_area = dict(zip(src_cols, areas))
    color_to_marker = dict(zip(src_cols, marker_patterns))

    masks = {}
    for src in src_cols:
        for _ in range(80):
            mask = _sprite_mask(n, color_to_area[src], color_to_marker[src])
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
    random.shuffle(filled_slots)

    cell_src = {}
    cursor = 0
    for src in src_cols:
        for _ in range(2):
            cell_src[filled_slots[cursor]] = src
            cursor += 1
    for slot in filled_slots[cursor:]:
        cell_src[slot] = random.choice(src_cols)

    marker_colors = {slot: random.choice(marker_palette) for slot in filled_slots}
    color_to_srcs: dict[int, set[int]] = {}
    for slot in filled_slots:
        color_to_srcs.setdefault(marker_colors[slot], set()).add(cell_src[slot])
    if max(len(srcs) for srcs in color_to_srcs.values()) < 2:
        return None

    input_frame = _make_bordered_frame(frame_h, frame_w, bg)
    output_frame = _make_bordered_frame(frame_h, frame_w, bg)
    for (rr, cc), src in cell_src.items():
        r0 = 1 + top_margin + rr * (n + 1)
        c0 = 1 + left_margin + cc * (n + 1)
        paste(
            input_frame,
            _colored_sprite(color_to_marker[src], marker_colors[(rr, cc)], bg),
            r0,
            c0,
            transparent=bg,
        )
        paste(output_frame, _colored_sprite(masks[src], color_to_tgt[src], bg), r0, c0, transparent=bg)

    ext_area = k * (LEGEND_H * LEGEND_W + n * n + 12)
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
    for src in src_cols:
        patches.append((_colored_sprite(masks[src], src, bg), True))
        patches.append((_legend(color_to_area[src], color_to_tgt[src], legend_code_col), False))
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
    raise RuntimeError("failed to generate valid edb79dae combo input")
