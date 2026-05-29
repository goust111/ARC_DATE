# Variant of edb79dae
# Dimension: D11 — 灰色框架的结构 (meta)
# Change: 单一矩形灰色框架，框架线条均匀，内部为操作区域... → 多个独立框架

"""
Design analysis for the D11 variant
===================================

a) Concrete behavior and trace
------------------------------
- Input now contains 2-3 disjoint grey rectangular frames instead of one.
- Each frame contains its own regular grid of solid NxN placeholder blocks.
- Outside the union of the frames, the puzzle still shows:
  1. one sprite template per placeholder color, and
  2. one 3x4 legend per placeholder color mapping source color -> target color.
- Output is the tight crop covering all frames together. Each frame keeps its border,
  its own placeholders are replaced by the matched sprite shape, and the shape is
  recolored via the legend.

Small trace:
- Original logic: find one frame -> collect all placeholder colors from that one frame
  -> find matching sprites and legends -> stamp everything inside that single crop.
- New logic: find every grey rectangle first -> for each frame collect only its local
  placeholders -> assign each placeholder color to exactly one frame -> find the global
  sprite/legend for that color -> stamp back into that color's owning frame.
- Core difference: "which frame owns this color?" becomes an explicit reasoning step.

b) Complexity calibration
-------------------------
- The variant must be visually distinguishable from the original, so every generated
  sample uses 2 or 3 frames, never 1.
- To avoid ambiguity, each placeholder color is restricted to exactly one frame, but a
  frame may reuse one or more of its own colors multiple times. This keeps frame
  assignment visible without introducing hidden tie-breakers.
- The number of distinct colors stays small (3-5), matching the original task family.

c) Implicit assumptions from the original solver that break here
----------------------------------------------------------------
- `_find_frame(...)` assumed exactly one rectangular border component. This must become
  `_find_frames(...)` returning 2-3 boxes.
- `outside[...] = False` on one bbox used to define the external search area. With
  multiple frames, every frame bbox must be excluded.
- `placeholder_colors` used to come from one crop. Now colors must be aggregated across
  frames while also remembering which frame owns each color.
- Output used to be one frame crop. Now it must be the union bbox of all frames, with
  bg preserved between frames.

d) Placement legality for generation
------------------------------------
- Frames are arranged as disjoint rectangles inside a single union box, with 2-3 cells
  of bg between neighboring frames.
- External sprites and legends are placed outside that union box with at least one bg
  cell of padding, so they never overlap frames or each other.
- Every placeholder color has exactly one sprite template and one legend, and the sprite
  bbox size matches every placeholder occurrence of that color.

e) Visual observability
-----------------------
- The multi-frame change is visible immediately in every input and every output.
- Color-to-frame assignment is human-readable: if a blue placeholder appears only in the
  top-left frame, then the blue sprite must belong to that frame only.
- The rule does not depend on color id ordering or any invisible numeric convention.

f) Clean visualization
----------------------
- External clue objects are kept outside the combined frame area, so the multi-frame
  region itself stays readable.
- Frames use the same grey border color and have simple rectangular geometry, making the
  structural change easy to parse from a few examples.

g) Color-change inferability
----------------------------
- Recoloring still comes only from the explicit 3x4 legend glyphs, exactly as in the
  original task family.
- The variant changes frame structure, not the legend semantics, so the color mapping
  remains learnable from the same kind of visual evidence.

Variant spec
------------
spec:
  grid: 24-30 rows x 24-30 cols
  bg: any color except 5; frame_col: always 5
  frames:
    count: 2-3 disjoint rectangular grey borders
    arrangement: packed into a shared union box with 2-3 bg cells between neighbors
    output: tight crop of the union box
  placeholders:
    size: NxN with N in {3,4}; if 3 frames are used then N=3
    per-frame layout: rows in {1,2}, cols fixed at 2, 0-1 empty cells only in 2x2 frames
    total blocks across all frames: 6-10
    colors: 3-5 distinct source colors, each color belongs to exactly one frame
  sprites:
    one external sprite per source color, same NxN bbox as its placeholders
  legends:
    one external 3x4 legend per source color, same pattern as the original task
  output:
    bg canvas over the union bbox of all frames; frame borders preserved; each frame
    independently receives its matched recolored sprites
"""

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


def _find_frames(g: np.ndarray, bg: int) -> list[tuple[int, tuple[int, int, int, int]]]:
    """Find all rectangular border components that behave like frames."""
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
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
            candidates.append((int(col), (int(r0), int(c0), int(r1), int(c1))))
    assert 2 <= len(candidates) <= 3, f"expected 2-3 rectangular frames, found {len(candidates)}"
    frame_colors = {col for col, _ in candidates}
    assert len(frame_colors) == 1, f"expected one shared frame color, found {sorted(frame_colors)}"
    return sorted(candidates, key=lambda item: (item[1][0], item[1][1]))


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _bbox_union(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp recolored sprites into multiple framed regions using placeholder positions and a color legend.

    Input structure:
    - bg background, 2-3 separate rectangular grey frames, and solid placeholder blocks inside each frame.
    - Outside the combined framed area: sprite shapes (one per placeholder color) and compact legend glyphs mapping each source color to an output color.

    Rule:
    - Find every grey frame and collect the placeholder colors that appear inside each one.
    - For each placeholder color, locate the matching-color sprite outside all frames and the legend that defines its recolor target.
    - Replace every placeholder occurrence inside its owning frame with the matched sprite shape recolored by the legend.

    Key insight: The external clues are still global, but the placeholders are partitioned across several independent frames, so the solver must first assign each color to the correct frame before stamping the sprites.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    frames = _find_frames(g, bg)
    frame_col = int(frames[0][0])

    frame_infos = []
    placeholder_colors: set[int] = set()
    color_to_frame: dict[int, int] = {}
    phw: dict[int, tuple[int, int]] = {}

    for frame_idx, (_col, (fr0, fc0, fr1, fc1)) in enumerate(frames):
        crop = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
        H, W = crop.shape
        assert H >= 3 and W >= 3, f"frame too small for interior: {(H, W)}"

        inner_mask = np.zeros((H, W), dtype=bool)
        inner_mask[1:-1, 1:-1] = True
        local_colors = set(map(int, np.unique(crop[inner_mask]))) - {bg, frame_col}
        assert local_colors, f"frame {frame_idx} has no placeholder colors"

        placeholders: list[tuple[int, tuple[int, int, int, int]]] = []
        for col in sorted(local_colors):
            assert col not in color_to_frame, f"placeholder color {col} appears in multiple frames"
            color_to_frame[col] = int(frame_idx)
            placeholder_colors.add(int(col))
            dims = []
            for pts in components_mask((crop == col) & inner_mask, connectivity=4):
                r0, c0, r1, c1 = bbox_coords(pts)
                bbox = (int(r0), int(c0), int(r1), int(c1))
                placeholders.append((int(col), bbox))
                dims.append((bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1))
            assert dims, f"no placeholder component found for color {col}"
            unique_dims = sorted(set(dims))
            assert len(unique_dims) == 1, f"ambiguous placeholder sizes for color {col}: {unique_dims}"
            phw[int(col)] = unique_dims[0]

        frame_infos.append(
            {
                "bbox": (int(fr0), int(fc0), int(fr1), int(fc1)),
                "crop": crop,
                "shape": (int(H), int(W)),
                "placeholders": placeholders,
            }
        )

    outside = np.ones(g.shape, dtype=bool)
    for _col, (fr0, fc0, fr1, fc1) in frames:
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
            color_map[int(src)] = int(tgt)
            legend_ids.add(int(ci["id"]))
            legend_ids.add(int(cj["id"]))

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
        for comp in all_candidates:
            if comp["id"] == base["id"]:
                continue
            r0, c0, r1, c1 = comp["bbox"]
            assert sr0 <= r0 and sr1 >= r1 and sc0 <= c0 and sc1 >= c1, (
                f"extra sprite component for color {src_col} is outside the base bbox"
            )
            pts_set.update(comp["pts"])

        coords = tuple(sorted((r - sr0, c - sc0) for r, c in pts_set))
        sprite_coords[src_col] = (coords, sr1 - sr0 + 1, sc1 - sc0 + 1)

    union_box = _bbox_union([info["bbox"] for info in frame_infos])
    ur0, uc0, ur1, uc1 = union_box
    out = np.full((ur1 - ur0 + 1, uc1 - uc0 + 1), bg, dtype=int)

    for frame_idx, info in enumerate(frame_infos):
        fr0, fc0, fr1, fc1 = info["bbox"]
        crop = np.array(info["crop"], dtype=int).copy()
        H, W = info["shape"]
        rr0 = fr0 - ur0
        cc0 = fc0 - uc0
        out[rr0 : rr0 + H, cc0 : cc0 + W] = crop
        out[rr0 + 1 : rr0 + H - 1, cc0 + 1 : cc0 + W - 1] = bg

        for src_col, (pr0, pc0, pr1, pc1) in info["placeholders"]:
            assert color_to_frame[src_col] == frame_idx, f"placeholder color {src_col} assigned to wrong frame"
            assert src_col in sprite_coords and src_col in color_map, (
                f"placeholder color {src_col} missing sprite or color mapping"
            )
            tgt_col = int(color_map[src_col])
            coords, sh, sw = sprite_coords[src_col]
            assert (pr1 - pr0 + 1, pc1 - pc0 + 1) == (sh, sw), (
                f"placeholder size does not match sprite for color {src_col}"
            )
            off_r = rr0 + pr0
            off_c = cc0 + pc0
            for dr, dc in coords:
                out[off_r + dr, off_c + dc] = tgt_col

    return out


# --------------- generator (o5x) ---------------

# === CFG: tunables ===
FRAME_COL = 5
NUM_FRAMES_CHOICES = [2, 3]
SPRITE_N_CHOICES_TWO_FRAMES = [3, 4]
SPRITE_N_CHOICES_THREE_FRAMES = [3]
FRAME_ROWS_CHOICES = [1, 2]
FRAME_COLS = 2
TOTAL_BLOCKS_RANGE = (6, 10)
NUM_COLORS_RANGE = (3, 5)
SPRITE_FILL_MIN = 0.5
SPRITE_FILL_MAX = 0.9
FRAME_GAP_CHOICES = [2, 3]
CANVAS_H_RANGE = (24, 31)
CANVAS_W_RANGE = (24, 31)
FRAME_POS_JITTER_PROB = 0.2
PATCH_PADDING = 1
GEN_RETRIES = 400


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
    return (grid_rows * n + grid_rows - 1, grid_cols * n + grid_cols - 1)


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


def _choose_frame_specs(num_frames: int) -> list[tuple[int, int]]:
    base = [(1, 2), (2, 3), (2, 4)]  # (rows, filled_count)
    options = []
    for combo in itertools.product(base, repeat=num_frames):
        total = sum(filled for _rows, filled in combo)
        if TOTAL_BLOCKS_RANGE[0] <= total <= TOTAL_BLOCKS_RANGE[1]:
            options.append(list(combo))
    return random.choice(options)


def _partitions(total: int, caps: list[int]) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []

    def rec(idx: int, remain: int, acc: list[int]) -> None:
        if idx == len(caps):
            if remain == 0:
                out.append(tuple(acc))
            return
        left = len(caps) - idx - 1
        lo = 1
        hi = min(caps[idx], remain - left)
        for val in range(lo, hi + 1):
            acc.append(val)
            rec(idx + 1, remain - val, acc)
            acc.pop()

    rec(0, total, [])
    return out


def _arrange_frames(
    frames: list[np.ndarray],
) -> tuple[list[tuple[int, int]], tuple[int, int], int]:
    num_frames = len(frames)
    gap = random.choice(FRAME_GAP_CHOICES)
    order = list(range(num_frames))
    random.shuffle(order)

    if num_frames == 2:
        slot_templates = [[(0, 0), (0, 1)], [(0, 0), (1, 0)]]
    else:
        slot_templates = [
            [(0, 0), (0, 1), (1, 0)],
            [(0, 0), (0, 1), (1, 1)],
            [(0, 0), (1, 0), (1, 1)],
            [(0, 1), (1, 0), (1, 1)],
        ]

    slots = random.choice(slot_templates)
    placed = list(zip(order, slots))
    rows = max(r for r, _c in slots) + 1
    cols = max(c for _r, c in slots) + 1
    row_heights = [0] * rows
    col_widths = [0] * cols
    for frame_idx, (sr, sc) in placed:
        h, w = frames[frame_idx].shape
        row_heights[sr] = max(row_heights[sr], int(h))
        col_widths[sc] = max(col_widths[sc], int(w))

    row_starts = []
    cur = 0
    for i, h in enumerate(row_heights):
        row_starts.append(cur)
        cur += h + (gap if i + 1 < rows else 0)

    col_starts = []
    cur = 0
    for i, w in enumerate(col_widths):
        col_starts.append(cur)
        cur += w + (gap if i + 1 < cols else 0)

    positions = [(-1, -1)] * num_frames
    for frame_idx, (sr, sc) in placed:
        positions[frame_idx] = (row_starts[sr], col_starts[sc])

    union_h = row_starts[-1] + row_heights[-1]
    union_w = col_starts[-1] + col_widths[-1]
    return positions, (int(union_h), int(union_w)), int(gap)


def _gen_attempt() -> np.ndarray | None:
    num_frames = random.choice(NUM_FRAMES_CHOICES)
    n_choices = SPRITE_N_CHOICES_THREE_FRAMES if num_frames == 3 else SPRITE_N_CHOICES_TWO_FRAMES
    n = random.choice(n_choices)

    bg = random.choice([c for c in range(10) if c != FRAME_COL])
    usable = [c for c in range(10) if c not in (bg, FRAME_COL)]

    frame_specs = _choose_frame_specs(num_frames)
    total_blocks = sum(filled for _rows, filled in frame_specs)

    k_max = min(NUM_COLORS_RANGE[1], total_blocks)
    k = random.randint(max(NUM_COLORS_RANGE[0], num_frames), k_max)
    color_caps = [filled for _rows, filled in frame_specs]
    color_partition = random.choice(_partitions(k, color_caps))

    src_cols = random.sample(usable, k)
    tgt_cols = [random.choice([c for c in usable if c != src]) for src in src_cols]
    color_to_tgt = dict(zip(src_cols, tgt_cols))
    masks = {src: _sprite_mask(n) for src in src_cols}

    frame_color_groups: list[list[int]] = []
    cursor = 0
    random.shuffle(src_cols)
    for count in color_partition:
        group = src_cols[cursor : cursor + count]
        frame_color_groups.append(group)
        cursor += count

    input_frames: list[np.ndarray] = []
    output_frames: list[np.ndarray] = []

    for frame_idx, ((rows, filled_count), local_srcs) in enumerate(zip(frame_specs, frame_color_groups)):
        cols = FRAME_COLS
        empty_count = rows * cols - filled_count
        _, filled_slots = _choose_empty_slots(rows, cols, empty_count)

        cell_colors: dict[tuple[int, int], int] = {}
        for slot, col in zip(filled_slots[: len(local_srcs)], local_srcs):
            cell_colors[slot] = int(col)
        for slot in filled_slots[len(local_srcs) :]:
            cell_colors[slot] = int(random.choice(local_srcs))

        ch, cw = _content_extent(n, rows, cols)
        frame_h = ch + 4
        frame_w = cw + 4
        input_frame = _make_bordered_frame(frame_h, frame_w, bg)
        output_frame = _make_bordered_frame(frame_h, frame_w, bg)

        for (rr, cc), src in cell_colors.items():
            r0 = 2 + rr * (n + 1)
            c0 = 2 + cc * (n + 1)
            paste(input_frame, np.full((n, n), src, dtype=int), r0, c0)
            paste(output_frame, _colored_sprite(masks[src], color_to_tgt[src], bg), r0, c0, transparent=bg)

        input_frames.append(input_frame)
        output_frames.append(output_frame)

    frame_positions, (union_h, union_w), _gap = _arrange_frames(input_frames)

    ext_area = k * (n * n + 16)
    possible_dims = [
        (h, w)
        for h in range(*CANVAS_H_RANGE)
        for w in range(*CANVAS_W_RANGE)
        if h >= union_h and w >= union_w and h * w - union_h * union_w >= ext_area
    ]
    if not possible_dims:
        return None

    h, w = random.choice(possible_dims)
    grid = np.full((h, w), bg, dtype=int)
    reserve = np.full((h, w), bg, dtype=int)

    ur0 = random.choice([0, h - union_h])
    uc0 = random.choice([0, w - union_w])
    if h - union_h > 2 and random.random() < FRAME_POS_JITTER_PROB:
        ur0 = random.randrange(0, h - union_h + 1)
    if w - union_w > 2 and random.random() < FRAME_POS_JITTER_PROB:
        uc0 = random.randrange(0, w - union_w + 1)

    paste(reserve, np.full((union_h, union_w), FRAME_COL, dtype=int), ur0, uc0)
    expected = np.full((union_h, union_w), bg, dtype=int)

    for input_frame, output_frame, (rr, cc) in zip(input_frames, output_frames, frame_positions):
        paste(grid, input_frame, ur0 + rr, uc0 + cc)
        paste(expected, output_frame, rr, cc)

    patches: list[tuple[np.ndarray, bool]] = []
    for src, tgt in color_to_tgt.items():
        patches.append((_colored_sprite(masks[src], src, bg), True))
        patches.append((_legend(src, tgt), False))
    random.shuffle(patches)
    patches.sort(key=lambda item: item[0].size, reverse=True)
    for patch, transparent in patches:
        if not _place_patch(grid, reserve, patch, bg, transparent=transparent):
            return None

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
    raise RuntimeError("failed to generate valid edb79dae D11 input")
