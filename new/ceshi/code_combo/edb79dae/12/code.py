# Combo variant of edb79dae (index 12)
# combo_spec: D11/0+D1/1
# Name: 多框架+边缘颜色匹配

"""
Design analysis for the combo variant
=====================================

a) Concrete behavior and trace
------------------------------
- The input now contains 2-3 disjoint grey frames, and each frame carries its own
  colored edge markers. The interior no longer uses source colors; it only contains
  uniform guide blocks.
- Outside the union of the frames, there is still one sprite template per source color
  and one 3x4 legend per source color.
- Small trace:
  1. Original task: find one frame -> read placeholder colors from the interior ->
     match same-color external sprites -> recolor via legend.
  2. Combo task: find every marked frame -> read each frame's row/column edge colors ->
     infer which source colors belong to which frame -> use row-mark x col-mark inside
     that frame to recover the guide-block coordinates -> stamp the external sprite
     there -> recolor via legend.
  3. Difference: color matching is now split into two layers. The source color first
     selects a frame through its border markers, then selects coordinates inside that
     frame through edge-position decoding.

b) Complexity calibration
-------------------------
- Every sample uses 2 or 3 frames, never 1, so the structural change is always visible.
- Each source color belongs to exactly one frame, avoiding cross-frame ambiguity.
- Within a frame, source colors use disjoint row bands and disjoint column bands, so
  row-mark x col-mark products never overlap or create fake targets.
- The number of source colors is kept at 3-5 and each frame uses only 1-2 source
  colors. This is enough repetition to support inference without making the border
  markings unreadable.

c) Implicit assumption audit
----------------------------
- The original `_find_frame(...)` assumed one solid-color border. That breaks because
  markers replace some border cells and there are multiple frames. The combo solver
  instead finds all enclosed background regions and expands each by 1 cell to recover
  the marked frame boxes.
- The original solver assumed interior colors identify sprite colors. That is false
  here: interior blocks are just guides, so source colors must come from border marks.
- The original output was one crop. The combo output must be the union crop covering
  all frames, with background preserved between them.
- Legend parsing can no longer use interior placeholder colors to identify the source
  side, so it falls back to the 7-cell vs 5-cell area rule used by the edge-match
  single variant.

d) Placement legality
---------------------
- Frames are packed into a shared union box with 2-3 cells of background between them.
- Guide blocks all share one NxN size, and every external sprite uses the same bbox.
- Border markers are never placed on corners; they sit only on rows/columns aligned to
  guide-block top-left positions.
- External sprites and legends are placed outside the entire union box with at least
  one background cell of padding, so nothing overlaps the frames or each other.

e) Visual observability
-----------------------
- Humans can directly see two independent clues: multiple frames, and border colors
  replacing interior identity colors.
- Because source colors do not appear inside the frames, the solver cannot be using the
  original direct color rule.
- Because each source color appears on only one frame border, the assignment step from
  color -> frame is visible from examples.

f) Clean visualization
----------------------
- Each frame only mixes three kinds of information: grey border with sparse markers,
  uniform guide blocks, and background gaps.
- External clue objects stay fully outside the framed union area, so the core puzzle
  region remains readable.
- Target colors are drawn from a pool that excludes source colors and the guide color,
  which prevents the outside clues from visually collapsing into one another.

g) Change inferability
----------------------
- The recoloring rule is still anchored by explicit 3x4 legends, so target colors are
  not hidden.
- The new part that must be inferred is structured and repeated: same-color markers on
  opposite sides encode row and column sets, and those colors are partitioned by frame.
- Generator guardrails ensure at least one source color in every sample occupies more
  than one guide block, so the edge-decoding behavior is actually observable instead of
  degenerating to only 1x1 placements.
"""

from __future__ import annotations

import itertools
import random
from collections import Counter

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


def _find_marked_frames(g: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    """Find 2-3 marked frames via enclosed background components."""
    boxes: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for pts in components_mask(g == bg, connectivity=4):
        rs = pts[:, 0]
        cs = pts[:, 1]
        if (rs == 0).any() or (rs == g.shape[0] - 1).any() or (cs == 0).any() or (cs == g.shape[1] - 1).any():
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        if r1 - r0 < 1 or c1 - c0 < 1:
            continue
        box = (int(r0 - 1), int(c0 - 1), int(r1 + 1), int(c1 + 1))
        fr0, fc0, fr1, fc1 = box
        assert 0 <= fr0 < fr1 < g.shape[0] and 0 <= fc0 < fc1 < g.shape[1], "expanded frame is out of bounds"
        if box not in seen:
            seen.add(box)
            boxes.append(box)
    assert 2 <= len(boxes) <= 3, f"expected 2-3 marked frames, found {len(boxes)}"
    return sorted(boxes, key=lambda box: (box[0], box[1]))


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _bbox_union(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp recolored sprites into multiple marked frames using edge-color coordinates and a color legend.

    Input structure:
    - bg background, 2-3 separate gray frames whose borders carry colored row/column markers, and uniform guide blocks inside each frame.
    - Outside the combined framed area: sprite shapes (one per source color) and compact legend glyphs mapping each source color to an output color.

    Rule:
    - Find every marked frame, collect its guide-block bboxes, and read same-color markers from opposite edges of that frame.
    - For each source color, first determine which frame owns it from the border markers, then use that frame's marked rows and marked columns to recover all target guide positions.
    - Locate the matching external sprite for each source color, stamp it at every decoded guide location, and recolor it according to the legend.

    Key insight: The task no longer has a single direct color match. Source color now acts as a two-stage code: it selects one frame globally, and inside that frame its edge-marker rows and columns decode the local sprite coordinates.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    frame_boxes = _find_marked_frames(g, bg)

    frame_infos = []
    source_colors: set[int] = set()
    source_to_frame: dict[int, int] = {}
    source_to_size: dict[int, tuple[int, int]] = {}
    frame_colors: list[int] = []

    for frame_idx, (fr0, fc0, fr1, fc1) in enumerate(frame_boxes):
        crop = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
        H, W = crop.shape
        assert H >= 3 and W >= 3, f"frame too small for interior: {(H, W)}"

        border_mask = np.zeros((H, W), dtype=bool)
        border_mask[0, :] = True
        border_mask[-1, :] = True
        border_mask[:, 0] = True
        border_mask[:, -1] = True
        frame_col = int(Counter(map(int, crop[border_mask].flatten())).most_common(1)[0][0])
        frame_colors.append(frame_col)

        inner_mask = np.zeros((H, W), dtype=bool)
        inner_mask[1:-1, 1:-1] = True
        inner_colors = set(map(int, np.unique(crop[inner_mask]))) - {bg}
        assert inner_colors, f"frame {frame_idx} has no guide blocks"

        guides: list[tuple[int, int, int, int]] = []
        for col in sorted(inner_colors):
            for pts in components_mask((crop == col) & inner_mask, connectivity=4):
                guides.append(tuple(map(int, bbox_coords(pts))))
        assert guides, f"frame {frame_idx} has no guide components"

        guide_dims = sorted({(r1 - r0 + 1, c1 - c0 + 1) for r0, c0, r1, c1 in guides})
        assert len(guide_dims) == 1, f"ambiguous guide sizes in frame {frame_idx}: {guide_dims}"
        guide_h, guide_w = guide_dims[0]
        guide_by_topleft = {(r0, c0): (r0, c0, r1, c1) for r0, c0, r1, c1 in guides}

        row_marks: dict[int, list[int]] = {}
        for r in range(1, H - 1):
            left = int(crop[r, 0])
            right = int(crop[r, W - 1])
            if left == right and left not in (bg, frame_col):
                row_marks.setdefault(left, []).append(int(r))

        col_marks: dict[int, list[int]] = {}
        for c in range(1, W - 1):
            top = int(crop[0, c])
            bottom = int(crop[H - 1, c])
            if top == bottom and top not in (bg, frame_col):
                col_marks.setdefault(top, []).append(int(c))

        local_sources = set(row_marks) & set(col_marks)
        assert local_sources, f"frame {frame_idx} has no source colors with both row and column markers"

        targets: dict[int, list[tuple[int, int]]] = {}
        for src_col in sorted(local_sources):
            assert src_col not in source_to_frame, f"source color {src_col} appears in multiple frames"
            pts = sorted((int(r), int(c)) for r in row_marks[src_col] for c in col_marks[src_col])
            assert pts, f"no targets derived for source color {src_col}"
            for pt in pts:
                assert pt in guide_by_topleft, f"marker-derived target {pt} missing guide block in frame {frame_idx}"
            targets[int(src_col)] = pts
            source_colors.add(int(src_col))
            source_to_frame[int(src_col)] = int(frame_idx)
            source_to_size[int(src_col)] = (int(guide_h), int(guide_w))

        all_targets = {pt for pts in targets.values() for pt in pts}
        assert all_targets == set(guide_by_topleft), f"frame {frame_idx} guide blocks do not match decoded targets"

        frame_infos.append(
            {
                "bbox": (int(fr0), int(fc0), int(fr1), int(fc1)),
                "crop": crop.copy(),
                "shape": (int(H), int(W)),
                "targets": targets,
            }
        )

    assert len(set(frame_colors)) == 1, f"expected a shared frame color, found {sorted(set(frame_colors))}"
    frame_col = int(frame_colors[0])

    outside = np.ones(g.shape, dtype=bool)
    for fr0, fc0, fr1, fc1 in frame_boxes:
        outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False

    comps = []
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
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

            assert ci["area"] != cj["area"], f"ambiguous equal-area legend pair: {ci['col']} and {cj['col']}"
            if ci["area"] > cj["area"]:
                src, tgt = ci["col"], cj["col"]
            else:
                src, tgt = cj["col"], ci["col"]

            if src not in source_colors:
                continue
            assert src not in color_map, f"duplicate legend mapping for source color {src}"
            color_map[int(src)] = int(tgt)
            legend_ids.add(int(ci["id"]))
            legend_ids.add(int(cj["id"]))

    assert set(color_map) == source_colors, (
        f"legend mappings {sorted(color_map)} do not match source colors {sorted(source_colors)}"
    )

    sprite_coords: dict[int, tuple[tuple[tuple[int, int], ...], int, int]] = {}
    for src_col in sorted(source_colors):
        all_candidates = [c for c in comps if c["col"] == src_col and c["id"] not in legend_ids]
        assert all_candidates, f"no non-legend sprite found for source color {src_col}"
        want_h, want_w = source_to_size[src_col]
        sized = [
            c
            for c in all_candidates
            if (c["bbox"][2] - c["bbox"][0] + 1, c["bbox"][3] - c["bbox"][1] + 1) == (want_h, want_w)
        ]
        assert len(sized) == 1, f"expected one size-matched sprite for source color {src_col}, found {len(sized)}"
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
        sprite_coords[src_col] = (coords, int(sr1 - sr0 + 1), int(sc1 - sc0 + 1))

    union_box = _bbox_union([info["bbox"] for info in frame_infos])
    ur0, uc0, ur1, uc1 = union_box
    out = np.full((ur1 - ur0 + 1, uc1 - uc0 + 1), bg, dtype=int)

    for frame_idx, info in enumerate(frame_infos):
        fr0, fc0, _fr1, _fc1 = info["bbox"]
        crop = np.asarray(info["crop"], dtype=int).copy()
        H, W = info["shape"]
        rr0 = fr0 - ur0
        cc0 = fc0 - uc0
        out[rr0 : rr0 + H, cc0 : cc0 + W] = crop
        out[rr0 + 1 : rr0 + H - 1, cc0 + 1 : cc0 + W - 1] = bg

        for src_col, pts in info["targets"].items():
            assert source_to_frame[src_col] == frame_idx, f"source color {src_col} assigned to wrong frame"
            tgt_col = int(color_map[src_col])
            coords, sh, sw = sprite_coords[src_col]
            assert (sh, sw) == source_to_size[src_col], f"guide size mismatch for source color {src_col}"
            for pr0, pc0 in pts:
                for dr, dc in coords:
                    out[rr0 + pr0 + dr, cc0 + pc0 + dc] = tgt_col

    return out


# --------------- generator (o5x) ---------------

# === CFG: tunables ===
FRAME_COL = 5
NUM_FRAMES_CHOICES = [2, 3]
SPRITE_N_CHOICES_TWO_FRAMES = [3, 4]
SPRITE_N_CHOICES_THREE_FRAMES = [3]
TOTAL_BLOCKS_RANGE = (6, 10)
NUM_COLORS_RANGE = (3, 5)
SPRITE_FILL_MIN = 0.5
SPRITE_FILL_MAX = 0.9
FRAME_GAP_CHOICES = [2, 3]
CANVAS_H_RANGE = (24, 31)
CANVAS_W_RANGE = (24, 31)
FRAME_POS_JITTER_PROB = 0.2
PATCH_PADDING = 1
GEN_RETRIES = 500
SMALL_FRAME_LAYOUTS = (
    ((1, 2),),
    ((2, 1),),
    ((2, 2),),
    ((1, 1), (1, 1)),
)
EXTRA_TWO_FRAME_LAYOUTS = (
    ((1, 2), (1, 1)),
    ((1, 1), (1, 2)),
    ((2, 1), (1, 1)),
    ((1, 1), (2, 1)),
)


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


def _layout_stats(layout: tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = sum(r for r, _ in layout)
    cols = sum(c for _, c in layout)
    blocks = sum(r * c for r, c in layout)
    colors = len(layout)
    return rows, cols, blocks, colors


def _choose_frame_layouts(num_frames: int) -> list[tuple[tuple[int, int], ...]] | None:
    layout_pool = list(SMALL_FRAME_LAYOUTS)
    if num_frames == 2:
        layout_pool.extend(EXTRA_TWO_FRAME_LAYOUTS)

    options: list[list[tuple[tuple[int, int], ...]]] = []
    for combo in itertools.product(layout_pool, repeat=num_frames):
        total_blocks = sum(_layout_stats(layout)[2] for layout in combo)
        total_colors = sum(_layout_stats(layout)[3] for layout in combo)
        if not (TOTAL_BLOCKS_RANGE[0] <= total_blocks <= TOTAL_BLOCKS_RANGE[1]):
            continue
        if not (NUM_COLORS_RANGE[0] <= total_colors <= NUM_COLORS_RANGE[1]):
            continue
        if not any(r * c > 1 for layout in combo for r, c in layout):
            continue
        options.append(list(combo))
    return random.choice(options) if options else None


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


def _arrange_frames(
    frames: list[np.ndarray],
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
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
    rows = max(r for r, _ in slots) + 1
    cols = max(c for _, c in slots) + 1
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
    return positions, (int(union_h), int(union_w))


def _gen_attempt() -> np.ndarray | None:
    num_frames = random.choice(NUM_FRAMES_CHOICES)
    layouts = _choose_frame_layouts(num_frames)
    if layouts is None:
        return None

    n_choices = SPRITE_N_CHOICES_THREE_FRAMES if num_frames == 3 else SPRITE_N_CHOICES_TWO_FRAMES
    n = random.choice(n_choices)

    bg = random.choice([c for c in range(10) if c != FRAME_COL])
    usable = [c for c in range(10) if c not in (bg, FRAME_COL)]

    total_colors = sum(_layout_stats(layout)[3] for layout in layouts)
    if total_colors > len(usable) - 1:
        return None

    src_cols = random.sample(usable, total_colors)
    random.shuffle(src_cols)
    guide_choices = [c for c in usable if c not in src_cols]
    if not guide_choices:
        return None
    guide_col = random.choice(guide_choices)
    target_pool = [c for c in usable if c not in src_cols and c != guide_col]
    if not target_pool:
        return None

    tgt_cols = [random.choice(target_pool) for _ in src_cols]
    color_to_tgt = dict(zip(src_cols, tgt_cols))
    masks = {src: _sprite_mask(n) for src in src_cols}

    frame_source_groups: list[list[int]] = []
    cursor = 0
    for layout in layouts:
        count = len(layout)
        frame_source_groups.append(src_cols[cursor : cursor + count])
        cursor += count

    input_frames: list[np.ndarray] = []
    output_frames: list[np.ndarray] = []

    for layout, local_srcs in zip(layouts, frame_source_groups):
        rows, cols, _blocks, _colors = _layout_stats(layout)
        ch, cw = _content_extent(n, rows, cols)
        frame_h = ch + 4
        frame_w = cw + 4
        input_frame = _make_bordered_frame(frame_h, frame_w, bg)
        output_frame = input_frame.copy()
        output_frame[1:-1, 1:-1] = bg

        row_cursor = 0
        col_cursor = 0
        for (row_count, col_count), src in zip(layout, local_srcs):
            row_marks = [2 + rr * (n + 1) for rr in range(row_cursor, row_cursor + row_count)]
            col_marks = [2 + cc * (n + 1) for cc in range(col_cursor, col_cursor + col_count)]

            for rr in row_marks:
                input_frame[rr, 0] = src
                input_frame[rr, frame_w - 1] = src
                output_frame[rr, 0] = src
                output_frame[rr, frame_w - 1] = src
            for cc in col_marks:
                input_frame[0, cc] = src
                input_frame[frame_h - 1, cc] = src
                output_frame[0, cc] = src
                output_frame[frame_h - 1, cc] = src

            guide_block = np.full((n, n), guide_col, dtype=int)
            sprite_block = _colored_sprite(masks[src], color_to_tgt[src], bg)
            for rr in range(row_cursor, row_cursor + row_count):
                for cc in range(col_cursor, col_cursor + col_count):
                    pr0 = 2 + rr * (n + 1)
                    pc0 = 2 + cc * (n + 1)
                    paste(input_frame, guide_block, pr0, pc0)
                    paste(output_frame, sprite_block, pr0, pc0, transparent=bg)

            row_cursor += row_count
            col_cursor += col_count

        input_frames.append(input_frame)
        output_frames.append(output_frame)

    frame_positions, (union_h, union_w) = _arrange_frames(input_frames)

    ext_area = total_colors * (n * n + 16)
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
    for src in src_cols:
        patches.append((_colored_sprite(masks[src], src, bg), True))
        patches.append((_legend(src, color_to_tgt[src]), False))
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
    raise RuntimeError("failed to generate valid edb79dae combo input")
