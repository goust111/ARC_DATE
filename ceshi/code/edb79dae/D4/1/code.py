# Variant of edb79dae
# Dimension: D4 — 目标位置：框架内的匹配点 (dst)
# Change: 灰色框架内部的单色像素点，每个点的颜色指示应匹配哪个外部精灵... → 目标为框架边缘的对齐标记

# Design analysis for the D4 variant
# ----------------------------------
# a) Concrete behavior and trace
#    - Original: the frame interior already contains solid NxN color blocks. Each block does two jobs:
#      it says which outside sprite color to use, and it gives the destination bbox directly.
#    - Variant: the frame interior starts blank. A placement color now appears as edge markers:
#      left-of-frame markers encode destination rows, top-of-frame markers encode destination columns.
#      For a given source color, every left marker of that color projects horizontally into the frame,
#      every top marker of that color projects vertically into the frame, and their intersections are
#      the top-left anchors where that color's sprite must be stamped.
#    - Concrete example: if red has one left marker aligned to row 6 and two top markers aligned to
#      columns 3 and 11, then red's sprite is placed twice, with top-left anchors at (6,3) and (6,11).
#      The outside legend still recolors red's sprite after placement.
#    - Difference from original: destination is no longer read from colored content inside the frame.
#      The solver must infer coordinates from projected border markers.
#
# b) Complexity calibration
#    - To avoid trivial one-marker-one-cell samples, total filled slots remain 6-10 and colors remain 2-5.
#    - To avoid ambiguous same-color cartesian products, every source color occupies a single horizontal
#      line (one row, many cols) or a single vertical line (many rows, one col) of slots.
#      That keeps repeated colors visible while making row-set x col-set decoding well-defined.
#
# c) Hidden assumptions from the original solver that must change
#    - `placeholder_colors = unique(frame interior)` no longer works because the interior is blank.
#    - Placeholder component bboxes no longer exist, so sprite size must be inferred from the outside
#      sprite itself rather than from interior blocks.
#    - Legend source detection can no longer rely on "is this color used inside the frame?";
#      instead the fixed 7-vs-5 legend area split is used, so the larger legend component is the source.
#    - Frame detection remains valid because markers are placed adjacent to the frame, not on top of
#      the gray border itself. This keeps the frame a single-color rectangle.
#
# d) Placement legality
#    - Markers are single cells placed in narrow bands above and left of the frame, aligned to valid
#      sprite top-left coordinates inside the interior.
#    - To avoid different colors overwriting one another when they share a row or column, markers live in
#      narrow dedicated bands above and left of the frame; only their alignment, not their distance from
#      the frame, carries meaning.
#    - Frame placement always leaves those top/left marker bands available.
#    - External sprites and legends are placed with padding using the same reserve grid, so they never
#      overlap the frame, markers, or each other.
#
# e) Visual observability
#    - The interior being completely blank in the input makes the D4 change obvious.
#    - The output reveals repeated placements along shared rows or shared columns, making the projection
#      rule inferable from a few examples.
#    - No ordering uses raw color ids; all mappings are revealed structurally by legends and marker
#      positions.
#
# f) Visual cleanliness
#    - Only two marker bands are used: top for columns and left for rows. This avoids clutter from
#      four-sided duplicate markers while keeping the "coordinate" reading clear.
#    - Outside sprites and legends remain separated by background gaps, as in the original task.
#
# g) Color-change inferability
#    - Recoloring stays unchanged from the original task: each source color has one compact two-color
#      legend outside the frame. The legend is still the only cue for output recoloring.
#
# h) New solver assumptions audited against the generator
#    - Every source color has at least one row marker and one column marker: enforced by group-based slot
#      generation.
#    - For each source color, all intended placements equal row_markers[color] x col_markers[color]:
#      enforced by restricting each color to one row-line or one column-line.
#    - The non-legend, non-marker outside cells of each source color belong to exactly one sprite bbox:
#      enforced by generating exactly one sprite template per source color.

from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location

MARKER_BAND_MAX = 5


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp recolored sprites into a framed region using edge-projected coordinates and a color legend.

    Input structure:
    - bg background, one gray rectangular frame, and an empty frame interior.
    - Just outside the frame: colored row markers in a narrow band left of the frame and column markers in a narrow band above it.
    - Elsewhere outside the frame: one sprite template per source color and one compact two-color legend per source color.

    Rule:
    - Find the frame, then read row markers and column markers by color from the left/top marker bands.
    - For each source color, take every same-color row marker and every same-color column marker; their projected intersections are the sprite destinations.
    - Locate the matching outside sprite, stamp it at each inferred destination, and recolor it using the outside legend mapping.

    Key insight: The frame interior gives no placement hints anymore — position is reconstructed from colored edge projections, while the legends still provide the final recoloring.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    frame_col, (fr0, fc0, fr1, fc1) = _find_frame(g, bg)

    crop = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    H, W = crop.shape
    assert H >= 3 and W >= 3, f"frame too small for interior: {(H, W)}"

    outside = np.ones(g.shape, dtype=bool)
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False

    comps = []
    cid = 0
    colors = [int(c) for c in np.unique(g) if int(c) not in (bg, frame_col)]
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
            if bg in sub_colors or len(sub_colors) != 2:
                continue
            assert ci["area"] != cj["area"], f"ambiguous equal-area legend pair: {ci['col']} and {cj['col']}"
            if ci["area"] > cj["area"]:
                src, tgt = int(ci["col"]), int(cj["col"])
            else:
                src, tgt = int(cj["col"]), int(ci["col"])
            assert src not in color_map, f"duplicate legend mapping for source color {src}"
            color_map[src] = tgt
            legend_ids.add(ci["id"])
            legend_ids.add(cj["id"])

    source_colors = set(color_map)
    assert source_colors, "no legend-derived source colors found"

    row_marks: dict[int, set[int]] = {col: set() for col in source_colors}
    col_marks: dict[int, set[int]] = {col: set() for col in source_colors}
    marker_ids: set[int] = set()

    for comp in comps:
        if comp["id"] in legend_ids or comp["col"] not in source_colors or comp["area"] != 1:
            continue
        r, c = comp["pts"][0]
        if 0 < fc0 - c <= MARKER_BAND_MAX and fr0 < r < fr1:
            row_marks[int(comp["col"])].add(int(r - fr0))
            marker_ids.add(comp["id"])
        elif 0 < fr0 - r <= MARKER_BAND_MAX and fc0 < c < fc1:
            col_marks[int(comp["col"])].add(int(c - fc0))
            marker_ids.add(comp["id"])

    sprite_coords: dict[int, tuple[tuple[tuple[int, int], ...], int, int]] = {}
    for src_col in sorted(source_colors):
        pts_set = set()
        for comp in comps:
            if comp["col"] != src_col or comp["id"] in legend_ids or comp["id"] in marker_ids:
                continue
            pts_set.update(comp["pts"])
        assert pts_set, f"no sprite found for source color {src_col}"
        coords_arr = np.array(sorted(pts_set), dtype=int)
        sr0, sc0, sr1, sc1 = bbox_coords(coords_arr)
        sh = sr1 - sr0 + 1
        sw = sc1 - sc0 + 1
        assert sh == sw and sh >= 2, f"sprite for color {src_col} is not a square bbox: {(sh, sw)}"
        coords = tuple(sorted((int(r - sr0), int(c - sc0)) for r, c in pts_set))
        sprite_coords[src_col] = (coords, sh, sw)

    out = crop.copy()
    out[1:-1, 1:-1] = bg

    for src_col in sorted(source_colors):
        assert row_marks[src_col], f"source color {src_col} has no row markers"
        assert col_marks[src_col], f"source color {src_col} has no column markers"
        tgt_col = int(color_map[src_col])
        coords, sh, sw = sprite_coords[src_col]
        for off_r in sorted(row_marks[src_col]):
            for off_c in sorted(col_marks[src_col]):
                assert off_r >= 1 and off_c >= 1, f"marker anchor for color {src_col} is on frame border"
                assert off_r + sh <= H - 1 and off_c + sw <= W - 1, (
                    f"sprite for color {src_col} does not fit at {(off_r, off_c)}"
                )
                for dr, dc in coords:
                    rr, cc = off_r + dr, off_c + dc
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
PATCH_PADDING = 1
GEN_RETRIES = 400


def _legend(src: int, tgt: int) -> np.ndarray:
    """Build a 3x4 legend block mapping src color to tgt color."""
    return np.array(
        [[src, src, tgt, tgt], [src, src, src, tgt], [src, src, tgt, tgt]],
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
    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if np.all(mask[border]) and not np.all(mask):
        return _sprite_mask(n)
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


def _powerset_indices(n: int) -> list[tuple[int, ...]]:
    out = []
    for k in range(1, n + 1):
        out.extend(itertools.combinations(range(n), k))
    return out


def _candidate_groups(rows: int, cols: int) -> list[dict]:
    groups = []
    for r in range(rows):
        for cols_subset in _powerset_indices(cols):
            slots = frozenset((r, c) for c in cols_subset)
            groups.append({"slots": slots, "rows": frozenset([r]), "cols": frozenset(cols_subset), "axis": "h"})
    for c in range(cols):
        for rows_subset in _powerset_indices(rows):
            if len(rows_subset) == 1:
                continue
            slots = frozenset((r, c) for r in rows_subset)
            groups.append({"slots": slots, "rows": frozenset(rows_subset), "cols": frozenset([c]), "axis": "v"})
    random.shuffle(groups)
    return groups


def _choose_groups(rows: int, cols: int, total: int, k: int) -> list[dict] | None:
    groups = _candidate_groups(rows, cols)

    def rec(start: int, chosen: list[dict], used: frozenset[tuple[int, int]], used_count: int) -> list[dict] | None:
        if len(chosen) == k:
            if used_count != total:
                return None
            used_rows = {r for r, _ in used}
            used_cols = {c for _, c in used}
            if len(used_rows) != rows or len(used_cols) != cols:
                return None
            return chosen
        remaining_groups = k - len(chosen)
        remaining_cells = total - used_count
        if remaining_cells < remaining_groups:
            return None
        for idx in range(start, len(groups)):
            grp = groups[idx]
            size = len(grp["slots"])
            if size > remaining_cells:
                continue
            if grp["slots"] & used:
                continue
            result = rec(
                idx + 1,
                chosen + [grp],
                frozenset(set(used) | set(grp["slots"])),
                used_count + size,
            )
            if result is not None:
                return result
        return None

    for _ in range(40):
        random.shuffle(groups)
        found = rec(0, [], frozenset(), 0)
        if found is not None:
            return found
    return None


def _gen_attempt() -> np.ndarray | None:
    struct = _choose_structure()
    if struct is None:
        return None
    n, rows, cols, _empty_count, total = struct

    bg = random.choice([c for c in range(10) if c != FRAME_COL])
    usable = [c for c in range(10) if c not in (bg, FRAME_COL)]

    max_k = min(NUM_COLORS_RANGE[1], total - 1, len(usable))
    if max_k < NUM_COLORS_RANGE[0]:
        return None
    k = random.randint(NUM_COLORS_RANGE[0], max_k)
    groups = _choose_groups(rows, cols, total, k)
    if groups is None:
        return None

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

    input_frame = _make_bordered_frame(frame_h, frame_w, bg)
    output_frame = _make_bordered_frame(frame_h, frame_w, bg)

    marker_rows: dict[int, set[int]] = {src: set() for src in src_cols}
    marker_cols: dict[int, set[int]] = {src: set() for src in src_cols}

    for grp, src in zip(groups, src_cols):
        for rr, cc in grp["slots"]:
            r0 = 1 + top_margin + rr * (n + 1)
            c0 = 1 + left_margin + cc * (n + 1)
            paste(output_frame, _colored_sprite(masks[src], color_to_tgt[src], bg), r0, c0, transparent=bg)
            marker_rows[src].add(r0)
            marker_cols[src].add(c0)

    marker_band = max(k, 1)
    ext_area = k * (12 + n * n + 10) + rows + cols + frame_h + frame_w + marker_band * (frame_h + frame_w)
    possible_dims = [
        (h, w)
        for h in range(*CANVAS_H_RANGE)
        for w in range(*CANVAS_W_RANGE)
        if h >= frame_h + marker_band and w >= frame_w + marker_band and h * w - frame_h * frame_w >= ext_area
    ]
    if not possible_dims:
        return None
    h, w = random.choice(possible_dims)
    grid = np.full((h, w), bg, dtype=int)
    reserve = np.full((h, w), bg, dtype=int)

    fr0 = random.randint(marker_band, h - frame_h)
    fc0 = random.randint(marker_band, w - frame_w)
    paste(grid, input_frame, fr0, fc0)
    paste(reserve, np.full((frame_h, frame_w), FRAME_COL, dtype=int), fr0, fc0)
    reserve[fr0 : fr0 + frame_h, fc0 - marker_band : fc0] = FRAME_COL
    reserve[fr0 - marker_band : fr0, fc0 : fc0 + frame_w] = FRAME_COL

    for band_idx, src in enumerate(src_cols, start=1):
        for r0 in marker_rows[src]:
            grid[fr0 + r0, fc0 - band_idx] = src
            reserve[fr0 + r0, fc0 - band_idx] = FRAME_COL
        for c0 in marker_cols[src]:
            grid[fr0 - band_idx, fc0 + c0] = src
            reserve[fr0 - band_idx, fc0 + c0] = FRAME_COL

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
    raise RuntimeError("failed to generate valid edb79dae D4 input")
