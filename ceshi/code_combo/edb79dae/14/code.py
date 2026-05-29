# Combo variant of edb79dae (index 14)
# combo_spec: D1/2+D4/0
# Name: 尺寸匹配+空洞嵌入

import itertools
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
#   - Original task: inside the gray frame, each destination was a colored solid block. Its color selected the
#     outside sprite, and the legend only changed the final recolor.
#   - D4 removes those colored destinations entirely: inside the frame we now see a gray tray with rectangular bg
#     cavities. A cavity no longer carries any source color.
#   - D1 then changes the remaining matching step: sprite identity is no longer "same color as the destination",
#     but "same bbox size as the destination cavity".
#   - Fused trace:
#       original: interior red 3x3 block -> find red outside sprite -> recolor by red->yellow legend.
#       combo: interior 3x3 hole -> find the unique outside sprite whose bbox is 3x3, even if that sprite is blue
#              -> use the blue->yellow legend -> stamp the blue shape into the hole and recolor it yellow.
#   - So the full chain is cavity size -> sprite bbox size -> sprite source color -> legend target color.
#     Color disappears from placement and survives only for recolor.
# b) Complexity calibration:
#   - Each sample uses 2-4 distinct cavity sizes selected from a small visible size pool; every chosen size appears
#     at least twice inside the frame. This gives repeated support for the size rule and avoids one-off coincidences.
#   - Source colors and target colors are chosen as disjoint sets. That makes it visually obvious that sprite color
#     is not a destination cue, and it keeps legends easy to parse.
#   - The size pool includes nearby dimensions such as 3x4 vs 4x4 or 2x4 vs 3x4, so exact measurement matters.
# c) Hidden assumptions audited from the original solver:
#   - `placeholder_colors = unique(inner)` breaks because the frame interior now contains only frame color + bg holes.
#   - Collecting inside non-bg components as destinations breaks for the same reason; the relevant objects are bg
#     components, and they must be verified to be full rectangles.
#   - Sprite lookup by `c["col"] == src_col` is no longer valid as a matching rule. In the combo, the solver must
#     first infer legend source colors from the outside legends, then bind those source colors to sprites, and only
#     then invert sprite bbox size to cavity size.
#   - Clearing the whole interior to bg would erase the tray structure that defines the cavities, so the output must
#     preserve the gray panel and only fill the holes.
# d) Placement legality guaranteed by the generator:
#   - Cavities are centered inside a regular variable-cell grid with at least one tray cell between neighboring cells,
#     so bg holes never merge and always stay strictly inside the border.
#   - Outside sprites and legends are placed with `random_free_location` on a reserve map, so they do not overlap
#     each other or the frame and remain visually separable.
# e) Visual observability:
#   - Across examples, the frame interior visibly changes from colored markers to a gray panel with cut-out slots.
#   - The same cavity size repeats multiple times and receives the same final recolored sprite, making the size rule
#     apparent from only a few examples.
#   - Because source colors appear only on external sprites and legends, the old color-matching story is visibly gone.
# f) Visual cleanliness:
#   - Only four visual ingredients remain: bg canvas, one gray framed tray, isolated outside sprites, and isolated
#     3x4 legends. There is no overprinting inside the frame except the final fill of each cavity.
#   - Empty grid cells remain solid tray, not extra holes, so every visible cavity is a valid destination.
# g) Change inferability:
#   - Recolor remains fully structured by the 3x4 legend glyphs: the larger 7-cell color is the source sprite color,
#     the smaller 5-cell color is the output color. No ordering by color id or hidden metadata is used.


FRAME_COL = 5
GRID_ROWS_RANGE = (2, 4)
GRID_COLS_RANGE = (3, 5)
TOTAL_CAVITIES_RANGE = (6, 10)
MAX_EMPTY_SLOTS = 2
NUM_TYPES_RANGE = (2, 4)
SIZE_POOL = [(2, 3), (3, 2), (3, 3), (2, 4), (4, 2), (3, 4), (4, 3), (4, 4)]
SPRITE_FILL_MIN = 0.45
SPRITE_FILL_MAX = 0.8
MIN_FRAME_H = 13
MIN_FRAME_W = 18
MAX_FRAME_DIM = 21
FRAME_MARGIN_SLACK = 4
CANVAS_H_RANGE = (22, 31)
CANVAS_W_RANGE = (23, 31)
FRAME_POS_JITTER_PROB = 0.25
PATCH_PADDING = 1
GEN_RETRIES = 400


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


def _bbox_dims(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _rectangular_cavities(crop: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    """Return all bg cavities strictly inside the frame; each must be a full rectangle."""
    h, w = crop.shape
    inner_mask = np.zeros((h, w), dtype=bool)
    inner_mask[1:-1, 1:-1] = True

    cavities: list[tuple[int, int, int, int]] = []
    for pts in components_mask((crop == bg) & inner_mask, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts)
        assert 0 < r0 <= r1 < h - 1 and 0 < c0 <= c1 < w - 1, "cavity must stay away from the frame border"
        sub = crop[r0 : r1 + 1, c0 : c1 + 1]
        assert int(len(pts)) == int(sub.size), "cavity must be a solid rectangle"
        assert np.all(sub == bg), "cavity bbox must contain only background"
        cavities.append((int(r0), int(c0), int(r1), int(c1)))

    assert cavities, "no inner cavities found"
    return sorted(cavities)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill framed cavities with recolored sprites chosen only by exact cavity size.

    Input structure:
    - A bg canvas contains one gray framed tray whose interior is mostly gray, except for several rectangular bg
      cavities of varying sizes.
    - Outside the tray are monochrome sprite templates and compact two-color legend glyphs. The sprite colors match
      legend source colors, but they do not appear inside the tray.

    Rule:
    - Detect the tray, enumerate the rectangular cavities, and read the outside legends as source-color -> target-color
      mappings.
    - For each legend source color, find its non-legend outside sprite and record that sprite's bbox size.
    - Match every cavity to the unique sprite with the same bbox size, stamp that sprite into the hole, and recolor
      the stamped pixels with the legend target color.

    Key insight: The tray contains no source-colored destination markers anymore. Placement is a pure geometry puzzle:
    bbox size determines which sprite fits a cavity, and the legend only determines the final recolor of that fitted
    sprite.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    frame_col, (fr0, fc0, fr1, fc1) = _find_frame(g, bg)
    crop = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    h, w = crop.shape
    assert h >= 3 and w >= 3, f"frame too small for interior: {(h, w)}"

    cavities = _rectangular_cavities(crop, bg)
    cavity_dims = [_bbox_dims(bbox) for bbox in cavities]
    unique_cavity_dims = sorted(set(cavity_dims))
    assert len(unique_cavity_dims) >= 2, "combo expects at least two cavity sizes"

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
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(pts_tuples)),
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
            if (r1 - r0 + 1, c1 - c0 + 1) != (3, 4):
                continue
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            sub_colors = set(map(int, np.unique(sub))) - {frame_col}
            if bg in sub_colors or len(sub_colors) != 2:
                continue
            if ci["area"] == cj["area"]:
                continue

            if ci["area"] > cj["area"]:
                src, tgt = int(ci["col"]), int(cj["col"])
            else:
                src, tgt = int(cj["col"]), int(ci["col"])

            assert src not in color_map, f"duplicate legend mapping for source color {src}"
            color_map[src] = tgt
            legend_ids.add(int(ci["id"]))
            legend_ids.add(int(cj["id"]))

    assert color_map, "no legend mappings found"

    sprite_coords: dict[int, tuple[tuple[tuple[int, int], ...], int, int]] = {}
    dims_to_src: dict[tuple[int, int], int] = {}
    for src_col in sorted(color_map):
        all_candidates = [c for c in comps if c["col"] == src_col and c["id"] not in legend_ids]
        assert all_candidates, f"no non-legend sprite found for color {src_col}"

        sized = [c for c in all_candidates if _bbox_dims(c["bbox"]) in unique_cavity_dims]
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
        sh, sw = sr1 - sr0 + 1, sc1 - sc0 + 1
        dims = (int(sh), int(sw))
        assert dims not in dims_to_src, f"ambiguous shared cavity size {dims}"
        sprite_coords[src_col] = (coords, int(sh), int(sw))
        dims_to_src[dims] = int(src_col)

    assert set(dims_to_src) == set(unique_cavity_dims), "not every cavity size has a matching sprite"

    out = crop.copy()
    for r0, c0, r1, c1 in cavities:
        dims = (int(r1 - r0 + 1), int(c1 - c0 + 1))
        src_col = dims_to_src[dims]
        tgt_col = int(color_map[src_col])
        coords, sh, sw = sprite_coords[src_col]
        assert dims == (sh, sw), f"cavity size {dims} does not match sprite size {(sh, sw)}"
        for dr, dc in coords:
            out[r0 + dr, c0 + dc] = tgt_col

    return out


def _legend(src: int, tgt: int) -> np.ndarray:
    return np.array(
        [[src, src, tgt, tgt], [src, src, src, tgt], [src, src, tgt, tgt]],
        dtype=int,
    )


def _carve_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> tuple[int, int]:
    """Carve a 4-connected Manhattan path from start to end."""
    r, c = start
    tr, tc = end
    mask[r, c] = True
    while (r, c) != (tr, tc):
        choices = []
        if r != tr:
            choices.append((r + (1 if tr > r else -1), c))
        if c != tc:
            choices.append((r, c + (1 if tc > c else -1)))
        r, c = random.choice(choices)
        mask[r, c] = True
    return r, c


def _is_connected(mask: np.ndarray) -> bool:
    pts = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if not pts:
        return False
    seen = {pts[0]}
    stack = [pts[0]]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(pts)


def _spans_bbox(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _sprite_mask(h: int, w: int) -> np.ndarray:
    """Generate a connected h x w binary mask whose bbox is exactly h x w."""
    area = h * w
    lo = max(4, int(np.ceil(SPRITE_FILL_MIN * area)))
    hi = min(int(np.floor(SPRITE_FILL_MAX * area)), area - 1)
    target = random.randint(lo, max(lo, hi))

    mask = np.zeros((h, w), dtype=bool)
    anchors = [
        (0, random.randrange(w)),
        (h - 1, random.randrange(w)),
        (random.randrange(h), 0),
        (random.randrange(h), w - 1),
    ]
    random.shuffle(anchors)
    cur = anchors[0]
    mask[cur] = True
    for nxt in anchors[1:]:
        cur = _carve_path(mask, cur, nxt)

    while int(mask.sum()) < target:
        frontier = set()
        for r, c in zip(*np.where(mask)):
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))
        assert frontier
        rr, cc = random.choice(sorted(frontier))
        mask[rr, cc] = True

    if mask.all():
        filled = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        random.shuffle(filled)
        for rr, cc in filled:
            trial = mask.copy()
            trial[rr, cc] = False
            if _spans_bbox(trial) and _is_connected(trial):
                mask = trial
                break
    assert _spans_bbox(mask)
    assert not mask.all()
    return mask


def _colored_sprite(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = int(color)
    return out


def _choose_structure() -> tuple[int, int, int, int] | None:
    """Pick (rows, cols, empty_count, total) consistent with the original frame density."""
    options = []
    for rows in range(*GRID_ROWS_RANGE):
        for cols in range(*GRID_COLS_RANGE):
            slots = rows * cols
            for total in range(TOTAL_CAVITIES_RANGE[0], TOTAL_CAVITIES_RANGE[1] + 1):
                empty = slots - total
                if 0 <= empty <= MAX_EMPTY_SLOTS:
                    options.append((rows, cols, empty, total))
    return random.choice(options) if options else None


def _choose_counts(k: int, total: int) -> list[int]:
    counts = [2] * k
    extra = total - 2 * k
    for _ in range(extra):
        counts[random.randrange(k)] += 1
    random.shuffle(counts)
    return counts


def _choose_empty_slots(
    rows: int, cols: int, empty_count: int
) -> tuple[set[tuple[int, int]], list[tuple[int, int]]]:
    slots = [(r, c) for r in range(rows) for c in range(cols)]
    valid = []
    for combo in itertools.combinations(slots, empty_count):
        empty = set(combo)
        filled = [slot for slot in slots if slot not in empty]
        if len({r for r, _ in filled}) == rows and len({c for _, c in filled}) == cols:
            valid.append(empty)
    assert valid
    empty_slots = random.choice(valid)
    filled_slots = [slot for slot in slots if slot not in empty_slots]
    random.shuffle(filled_slots)
    return empty_slots, filled_slots


def _layout_cavities(
    rows: int,
    cols: int,
    cell_types: dict[tuple[int, int], int],
    size_map: dict[int, tuple[int, int]],
) -> tuple[dict[tuple[int, int], tuple[int, int, int, int]], int, int]:
    """Compute centered cavity rectangles inside a regular grid of variable-size cells."""
    row_heights = []
    for r in range(rows):
        row_sizes = [size_map[cell_types[(r, c)]][0] for c in range(cols) if (r, c) in cell_types]
        assert row_sizes
        row_heights.append(max(row_sizes))

    col_widths = []
    for c in range(cols):
        col_sizes = [size_map[cell_types[(r, c)]][1] for r in range(rows) if (r, c) in cell_types]
        assert col_sizes
        col_widths.append(max(col_sizes))

    row_offsets = []
    cur = 0
    for rh in row_heights:
        row_offsets.append(cur)
        cur += rh + 1
    content_h = cur - 1

    col_offsets = []
    cur = 0
    for cw in col_widths:
        col_offsets.append(cur)
        cur += cw + 1
    content_w = cur - 1

    placements: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for (rr, cc), src in cell_types.items():
        ch, cw = size_map[src]
        cell_r0 = row_offsets[rr]
        cell_c0 = col_offsets[cc]
        off_r = cell_r0 + (row_heights[rr] - ch) // 2
        off_c = cell_c0 + (col_widths[cc] - cw) // 2
        placements[(rr, cc)] = (int(off_r), int(off_c), int(ch), int(cw))

    return placements, int(content_h), int(content_w)


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
    rows, cols, empty_count, total = struct

    max_k = min(NUM_TYPES_RANGE[1], total // 2, len(SIZE_POOL), 4)
    if max_k < NUM_TYPES_RANGE[0]:
        return None

    bg = random.choice([c for c in range(10) if c != FRAME_COL])
    usable = [c for c in range(10) if c not in (bg, FRAME_COL)]
    if len(usable) < 2 * max_k:
        return None

    k = random.randint(NUM_TYPES_RANGE[0], max_k)
    counts = _choose_counts(k, total)
    assert all(count >= 2 for count in counts), "each size type must repeat for visual support"

    palette = random.sample(usable, 2 * k)
    src_cols = palette[:k]
    tgt_cols = palette[k:]
    assert set(src_cols).isdisjoint(tgt_cols), "source and target colors should stay visually separate"
    color_to_tgt = dict(zip(src_cols, tgt_cols))

    size_choices = random.sample(SIZE_POOL, k)
    assert len(set(size_choices)) == k, "each source type must own a unique cavity size"
    size_map = {src: dims for src, dims in zip(src_cols, size_choices)}
    masks = {src: _sprite_mask(*size_map[src]) for src in src_cols}

    _, filled_slots = _choose_empty_slots(rows, cols, empty_count)
    cell_types: dict[tuple[int, int], int] = {}
    cursor = 0
    for src, count in zip(src_cols, counts):
        for slot in filled_slots[cursor : cursor + count]:
            cell_types[slot] = src
        cursor += count
    assert len(cell_types) == total

    placements, content_h, content_w = _layout_cavities(rows, cols, cell_types, size_map)

    min_fh = max(MIN_FRAME_H, content_h + 4)
    min_fw = max(MIN_FRAME_W, content_w + 4)
    max_fh = min(MAX_FRAME_DIM, content_h + 4 + FRAME_MARGIN_SLACK)
    max_fw = min(MAX_FRAME_DIM, content_w + 4 + FRAME_MARGIN_SLACK)
    if min_fh > max_fh or min_fw > max_fw:
        return None

    frame_h = random.randint(min_fh, max_fh)
    frame_w = random.randint(min_fw, max_fw)
    inner_h = frame_h - 2
    inner_w = frame_w - 2
    y_slack = inner_h - content_h
    x_slack = inner_w - content_w
    if y_slack < 2 or x_slack < 2:
        return None
    top_margin = random.randint(1, y_slack - 1)
    left_margin = random.randint(1, x_slack - 1)

    input_frame = np.full((frame_h, frame_w), FRAME_COL, dtype=int)
    output_frame = np.full((frame_h, frame_w), FRAME_COL, dtype=int)

    for slot, src in cell_types.items():
        rr, cc, ch, cw = placements[slot]
        r0 = 1 + top_margin + rr
        c0 = 1 + left_margin + cc
        input_frame[r0 : r0 + ch, c0 : c0 + cw] = bg
        output_frame[r0 : r0 + ch, c0 : c0 + cw] = bg
        paste(
            output_frame,
            _colored_sprite(masks[src], color_to_tgt[src], bg),
            r0,
            c0,
            transparent=bg,
        )

    patch_bbox_area = sum(h * w for h, w in size_map.values()) + 20 * k
    possible_dims = [
        (h, w)
        for h in range(*CANVAS_H_RANGE)
        for w in range(*CANVAS_W_RANGE)
        if h >= frame_h and w >= frame_w and h * w - frame_h * frame_w >= patch_bbox_area
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

    if get_bg(grid) != bg:
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
