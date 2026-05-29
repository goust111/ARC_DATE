# Variant of edb79dae
# Dimension: D12 — 精灵与目标的基数关系 (meta)
# Change: 一对一映射：每个精灵颜色对应框架内一个或多个同色目标像素，每个目标像素只匹配一个精灵... → 一对多但有选择：精灵只放置到最近的匹配目标

import itertools
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis
# - Concrete behavior:
#   The framed area still contains solid NxN placeholder blocks and the outside still contains
#   one sprite template plus one legend per placeholder color. The visible D12 change is that
#   repeated placeholders of the same color are no longer all filled. Instead, the single outside
#   sprite of that color is copied only into the one inside block that is spatially closest to the
#   sprite. Other same-color blocks vanish to background in the output.
# - Small trace:
#   Suppose red has three inside blocks and one red sprite outside the frame. In the original task,
#   all three red blocks would be replaced by the red sprite shape and then recolored via the legend.
#   In this variant, we compute the sprite-to-block distances, pick the unique nearest red block,
#   place the red sprite only there, and leave the other two red blocks empty in the output.
# - Complexity calibration:
#   To make the change visible, the generator forces repeated placeholder colors. Each color appears
#   at least twice, so every sample contains genuine one-to-many candidates. The outside sprite is
#   placed so its nearest same-color target is unique and measurably closer than the runner-up.
# - Hidden assumptions from the original main():
#   1. Placeholder extraction by color and bbox size is still valid; repeated blocks remain solid NxN.
#   2. Legend parsing is unchanged because D12 does not alter the legend glyphs.
#   3. Sprite extraction is unchanged because there is still exactly one size-matched non-legend
#      sprite per placeholder color.
#   4. The old final loop assumed "stamp on every placeholder of that color". That assumption is
#      invalid now and is replaced with "group placeholders by color, choose the nearest one, stamp
#      exactly once".
# - Placement legality:
#   The generator keeps the original no-overlap reserve system for the frame and outside patches.
#   Sprite placement is additionally filtered by a unique-nearest test, so the produced inputs are
#   both spatially legal and solver-compatible.
# - Visual observability:
#   Distance uses geometry only: Manhattan distance between bbox centers. No color-id ordering or
#   hidden metadata is involved. Because each color repeats and only one instance survives, the
#   difference from the original task is visible in every valid sample.
# - Visual cleanliness:
#   Objects stay separated by background gaps. The frame interior keeps the regular grid, while the
#   outside remains a sparse collection of sprites and legends, so the proximity cue is readable.
# - Color mapping inference:
#   Recoloring still comes from the legend blocks, exactly as in the source task. The new inference
#   burden is only "which same-color block wins", and that is indicated by spatial proximity.
# - New solver assumptions reviewed:
#   The solver now assumes each repeated color has a deterministic nearest placeholder. The generator
#   guarantees this by rejecting placements whose best and second-best distances tie or are too close.


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


def _bbox_center2(b: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = b
    return r0 + r1, c0 + c1


def _center_manhattan2(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar, ac = _bbox_center2(a)
    br, bc = _bbox_center2(b)
    return abs(ar - br) + abs(ac - bc)


def _choose_nearest_placeholder(
    entries: list[dict[str, tuple[int, int, int, int]]], sprite_bbox: tuple[int, int, int, int]
) -> dict[str, tuple[int, int, int, int]]:
    scored = []
    for entry in entries:
        dist = _center_manhattan2(entry["abs_bbox"], sprite_bbox)
        rr0, cc0, _, _ = entry["rel_bbox"]
        scored.append((dist, rr0, cc0, entry))
    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    return scored[0][3]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Place each recolored sprite into only one chosen spot inside the frame.

    Input structure:
    - A gray rectangular frame encloses several solid placeholder blocks, and some colors can repeat
      in multiple grid slots.
    - Outside the frame there is one template sprite for each placeholder color plus a small two-color
      legend that specifies the sprite's output color.

    Rule:
    - Find the frame, the repeated placeholder blocks, the outside sprites, and the legend mappings.
    - For each placeholder color, compare its one outside sprite with all inside blocks of that color
      and keep only the nearest candidate location.
    - Stamp that sprite once at the chosen location and recolor it via the legend; all other same-color
      placeholder blocks are cleared to background.

    Key insight:
    - Color still chooses which sprite shape to use, but repeated placeholders no longer mean repeated
      stamping: the outside sprite's position acts as the proximity cue that selects a single target.
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

    placeholders_by_color: dict[int, list[dict[str, tuple[int, int, int, int]]]] = {
        col: [] for col in sorted(placeholder_colors)
    }
    for col in sorted(placeholder_colors):
        for pts in components_mask((crop == col) & inner_mask, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            placeholders_by_color[col].append(
                {
                    "rel_bbox": (r0, c0, r1, c1),
                    "abs_bbox": (fr0 + r0, fc0 + c0, fr0 + r1, fc0 + c1),
                }
            )

    phw: dict[int, tuple[int, int]] = {}
    for col, entries in placeholders_by_color.items():
        dims = [(b["rel_bbox"][2] - b["rel_bbox"][0] + 1, b["rel_bbox"][3] - b["rel_bbox"][1] + 1) for b in entries]
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

    sprite_data: dict[int, dict[str, object]] = {}
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
        sprite_data[src_col] = {
            "coords": coords,
            "shape": (sr1 - sr0 + 1, sc1 - sc0 + 1),
            "bbox": (sr0, sc0, sr1, sc1),
        }

    out = crop.copy()
    out[1:-1, 1:-1] = bg

    for src_col in sorted(placeholder_colors):
        assert src_col in sprite_data and src_col in color_map, (
            f"placeholder color {src_col} missing sprite or color mapping"
        )
        chosen = _choose_nearest_placeholder(placeholders_by_color[src_col], sprite_data[src_col]["bbox"])
        tgt_col = int(color_map[src_col])
        coords = sprite_data[src_col]["coords"]
        sh, sw = sprite_data[src_col]["shape"]
        pr0, pc0, pr1, pc1 = chosen["rel_bbox"]
        assert (pr1 - pr0 + 1, pc1 - pc0 + 1) == (sh, sw), (
            f"placeholder size does not match sprite for color {src_col}"
        )
        assert pr0 + sh <= H and pc0 + sw <= W, f"sprite for color {src_col} does not fit at placeholder"
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
NUM_COLORS_RANGE = (2, 4)
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
SPRITE_PLACE_RETRIES = 80
MIN_NEAREST_GAP = 4


def _legend(src: int, tgt: int) -> np.ndarray:
    return np.array(
        [[src, src, tgt, tgt], [src, src, src, tgt], [src, src, tgt, tgt]],
        dtype=int,
    )


def _sprite_mask(n: int) -> np.ndarray:
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


def _repeat_counts(total: int, k: int) -> list[int]:
    counts = [2] * k
    for _ in range(total - 2 * k):
        counts[random.randrange(k)] += 1
    random.shuffle(counts)
    return counts


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


def _distance_profile(
    sprite_bbox: tuple[int, int, int, int], candidates: list[tuple[int, int, int, int]]
) -> list[tuple[int, int, int, int, int, int]]:
    scored = []
    for box in candidates:
        dist = _center_manhattan2(sprite_bbox, box)
        scored.append((dist, box[0], box[1], box[2], box[3], len(scored)))
    scored.sort()
    return scored


def _place_sprite_for_color(
    grid: np.ndarray,
    reserve: np.ndarray,
    sprite_patch: np.ndarray,
    bg: int,
    candidate_boxes: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    blocker = np.full(sprite_patch.shape, FRAME_COL, dtype=int)
    h, w = sprite_patch.shape
    for _ in range(SPRITE_PLACE_RETRIES):
        try:
            r, c = random_free_location(reserve, blocker, transparent=bg, padding=PATCH_PADDING)
        except ValueError:
            return None
        bbox = (r, c, r + h - 1, c + w - 1)
        profile = _distance_profile(bbox, candidate_boxes)
        assert len(profile) >= 2, "variant generator expects repeated placeholders per color"
        if profile[1][0] - profile[0][0] < MIN_NEAREST_GAP:
            continue
        paste(reserve, blocker, r, c)
        paste(grid, sprite_patch, r, c, transparent=bg)
        return bbox
    return None


def _gen_attempt() -> np.ndarray | None:
    struct = _choose_structure()
    if struct is None:
        return None
    n, rows, cols, empty_count, total = struct

    max_k = min(NUM_COLORS_RANGE[1], total // 2)
    if max_k < NUM_COLORS_RANGE[0]:
        return None

    bg = random.choice([c for c in range(10) if c != FRAME_COL])
    usable = [c for c in range(10) if c not in (bg, FRAME_COL)]

    k = random.randint(NUM_COLORS_RANGE[0], max_k)
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
    counts = _repeat_counts(total, k)
    bag = []
    for src, count in zip(src_cols, counts):
        bag.extend([src] * count)
    random.shuffle(bag)
    cell_colors = {slot: col for slot, col in zip(filled_slots, bag)}

    input_frame = _make_bordered_frame(frame_h, frame_w, bg)
    output_frame = _make_bordered_frame(frame_h, frame_w, bg)
    placeholder_boxes_abs: dict[int, list[tuple[int, int, int, int]]] = {src: [] for src in src_cols}
    slot_boxes_rel: dict[tuple[int, int], tuple[int, int, int, int]] = {}

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

    for (rr, cc), src in cell_colors.items():
        r0 = 1 + top_margin + rr * (n + 1)
        c0 = 1 + left_margin + cc * (n + 1)
        box_rel = (r0, c0, r0 + n - 1, c0 + n - 1)
        slot_boxes_rel[(rr, cc)] = box_rel
        placeholder_boxes_abs[src].append((fr0 + r0, fc0 + c0, fr0 + r0 + n - 1, fc0 + c0 + n - 1))
        paste(input_frame, np.full((n, n), src, dtype=int), r0, c0)

    paste(grid, input_frame, fr0, fc0)
    paste(reserve, np.full((frame_h, frame_w), FRAME_COL, dtype=int), fr0, fc0)

    chosen_slot_by_color: dict[int, tuple[int, int, int, int]] = {}
    for src in src_cols:
        sprite_patch = _colored_sprite(masks[src], src, bg)
        sprite_bbox = _place_sprite_for_color(grid, reserve, sprite_patch, bg, placeholder_boxes_abs[src])
        if sprite_bbox is None:
            return None
        profile = _distance_profile(sprite_bbox, placeholder_boxes_abs[src])
        best_box_abs = profile[0][1:5]
        found_rel = None
        for box_rel in slot_boxes_rel.values():
            box_abs = (fr0 + box_rel[0], fc0 + box_rel[1], fr0 + box_rel[2], fc0 + box_rel[3])
            if box_abs == best_box_abs:
                found_rel = box_rel
                break
        if found_rel is None:
            return None
        chosen_slot_by_color[src] = found_rel

    for src, tgt in zip(src_cols, tgt_cols):
        if not _place_patch(grid, reserve, _legend(src, tgt), bg, transparent=False):
            return None

    for src, rel_box in chosen_slot_by_color.items():
        r0, c0, _, _ = rel_box
        paste(output_frame, _colored_sprite(masks[src], color_to_tgt[src], bg), r0, c0, transparent=bg)

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
