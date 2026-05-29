# Combo variant of edb79dae (index 6)
# combo_spec: D4/1+D14/1
# Name: 边缘投影定位+位置依赖对齐
#
# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - Original: solid NxN blocks already exist inside the frame. A block's color selects the outside
#      sprite, and the block bbox directly gives the paste location.
#    - Combo: the frame interior is blank. Instead, single-cell markers sit in a narrow band left of
#      the frame and another band above it. Same-color row markers and column markers project inward;
#      each row/column intersection is now a single-pixel anchor.
#    - The second step is no longer fixed top-left placement. After an anchor is inferred, its quadrant
#      inside the frame decides which sprite corner locks onto that anchor:
#        TL -> sprite top-left touches the anchor
#        TR -> sprite top-right touches the anchor
#        BL -> sprite bottom-left touches the anchor
#        BR -> sprite bottom-right touches the anchor
#    - Example: a source color with a top-half row marker and two column markers on opposite sides of
#      the vertical split yields two anchors. The left anchor uses TL alignment, while the right anchor
#      uses TR alignment, so the same sprite grows in different directions from the two intersections.
#
# b) Complexity calibration
#    - The D4 cartesian-product decoding is kept controlled by using one horizontal line or one vertical
#      line of intended slots per color. That prevents ambiguous extra intersections.
#    - The D14 quadrant rule stays visually meaningful because anchors are single pixels, not full NxN
#      blocks, and every generated sample covers all four quadrants.
#    - At least one source color spans multiple quadrant modes, so the fusion does not collapse into
#      "projection only" or "alignment only" behavior.
#
# c) Hidden assumptions that change
#    - `placeholder_colors = colors inside frame` is invalid because the interior is empty.
#    - Placeholder bboxes no longer exist, so sprite size must come from the external sprite template.
#    - Legend source detection cannot rely on in-frame colors; the fixed 7-vs-5 legend area split is
#      used instead, so the larger legend component is always the source color.
#    - Placement is no longer a simple `off_r = row, off_c = col`: the inferred anchor must be mapped
#      through a quadrant-dependent corner-alignment function.
#
# d) Placement legality
#    - Frame layout is chosen so the frame midlines fall on slot gaps. That makes quadrant classification
#      stable and guarantees the intended anchor corner is unambiguous.
#    - Every intended sprite still occupies a regular NxN slot with one-cell gaps, so quadrant-based
#      alignment never causes overlap or border collisions.
#    - Marker bands are reserved on the top and left of the frame; legends and sprite templates are
#      placed elsewhere with padding, so no outside object overlaps the frame or markers.
#
# e) Visual observability
#    - The blank interior makes the D4 change obvious: destinations must be inferred from edge markers.
#    - The D14 change is also visible because equal-color placements can land in different quadrants and
#      therefore expand from different corners of their projected anchors.
#    - No ordering depends on raw color ids. Shape selection comes from color equality, recoloring comes
#      from explicit legends, and alignment comes from visible spatial position.
#
# f) Visual cleanliness
#    - Only two marker bands are used, avoiding four-sided clutter.
#    - The output remains a clean grid of separated sprite bboxes; the input remains sparse because the
#      frame interior is empty and the markers stay outside the border.
#
# g) Color-change inferability
#    - Recoloring is unchanged structurally from the original task: one compact two-color legend per
#      source color. The fusion only changes where anchors come from and how they control alignment.
#
# h) New solver assumptions audited against the generator
#    - Each source color has at least one row marker and one column marker: guaranteed by line-shaped
#      group generation.
#    - For each source color, intended anchors equal row_markers[color] x col_markers[color]: guaranteed
#      by restricting each color to one horizontal or one vertical line of slots.
#    - At least one source color uses more than one quadrant mode, and the full sample uses all four
#      modes: guaranteed by group selection plus layout constraints.
#    - External non-legend, non-marker cells of one source color belong to exactly one sprite bbox:
#      guaranteed by generating one connected full-bbox sprite template per source color.

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

MARKER_BAND_MAX = 8


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
    Rebuild a framed sprite mosaic from edge-projected anchors, then recolor it with the legend.

    Input structure:
    - A background-filled canvas contains one rectangular frame whose interior starts empty.
    - Just outside the frame, a narrow band on the left carries colored row markers and a narrow band
      on the top carries colored column markers.
    - Elsewhere outside the frame sit one sprite template per source color and one compact two-color
      legend per source color.

    Rule:
    - Detect the frame, infer source colors and recoloring targets from the legends, and read row/column
      marker coordinates by color from the left/top bands.
    - For each source color, take every same-color row marker and every same-color column marker; their
      projected intersections are single-pixel anchors inside the frame.
    - For each inferred anchor, choose the matching sprite by color, decide the anchor's quadrant inside
      the frame, and place the sprite so the quadrant-selected corner touches that anchor.
    - Recolor every placed sprite to the legend's target color.

    Key insight:
    Placement now requires two linked spatial inferences. Edge markers first reconstruct the anchor
    coordinates, and only then does the anchor's region inside the frame determine which sprite corner
    should align to that coordinate.
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
        pts_set: set[tuple[int, int]] = set()
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
        for pr in sorted(row_marks[src_col]):
            for pc in sorted(col_marks[src_col]):
                assert 1 <= pr <= H - 2 and 1 <= pc <= W - 2, f"marker anchor for color {src_col} is on frame border"
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
PATCH_PADDING = 1
GEN_RETRIES = 500


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


def _powerset_indices(n: int) -> list[tuple[int, ...]]:
    out = []
    for k in range(1, n + 1):
        out.extend(itertools.combinations(range(n), k))
    return out


def _candidate_groups(rows: int, cols: int, row_split: int, col_split: int) -> list[dict]:
    groups = []
    for r in range(rows):
        for cols_subset in _powerset_indices(cols):
            slots = frozenset((r, c) for c in cols_subset)
            modes = frozenset(_slot_mode(r, c, row_split, col_split) for c in cols_subset)
            groups.append({"slots": slots, "axis": "h", "modes": modes})
    for c in range(cols):
        for rows_subset in _powerset_indices(rows):
            if len(rows_subset) == 1:
                continue
            slots = frozenset((r, c) for r in rows_subset)
            modes = frozenset(_slot_mode(r, c, row_split, col_split) for r in rows_subset)
            groups.append({"slots": slots, "axis": "v", "modes": modes})
    random.shuffle(groups)
    return groups


def _choose_groups(
    rows: int,
    cols: int,
    total: int,
    k: int,
    row_split: int,
    col_split: int,
) -> list[dict] | None:
    groups = _candidate_groups(rows, cols, row_split, col_split)
    target_quadrants = {"TL", "TR", "BL", "BR"}

    def rec(
        start: int,
        chosen: list[dict],
        used: frozenset[tuple[int, int]],
        used_count: int,
        quadrants: frozenset[str],
        has_cross_mode: bool,
    ) -> list[dict] | None:
        if len(chosen) == k:
            if used_count != total:
                return None
            used_rows = {r for r, _ in used}
            used_cols = {c for _, c in used}
            if len(used_rows) != rows or len(used_cols) != cols:
                return None
            if set(quadrants) != target_quadrants:
                return None
            if not has_cross_mode:
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
            next_quadrants = frozenset(set(quadrants) | set(grp["modes"]))
            result = rec(
                idx + 1,
                chosen + [grp],
                frozenset(set(used) | set(grp["slots"])),
                used_count + size,
                next_quadrants,
                has_cross_mode or len(grp["modes"]) > 1,
            )
            if result is not None:
                return result
        return None

    for _ in range(60):
        random.shuffle(groups)
        found = rec(0, [], frozenset(), 0, frozenset(), False)
        if found is not None:
            return found
    return None


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
    n, rows, cols, _empty_count, total = struct

    layout = _choose_layout(n, rows, cols)
    if layout is None:
        return None
    frame_h, frame_w, top_margin, _bottom_margin, left_margin, _right_margin, row_split, col_split = layout

    bg = random.choice([c for c in range(10) if c != FRAME_COL])
    usable = [c for c in range(10) if c not in (bg, FRAME_COL)]

    max_k = min(NUM_COLORS_RANGE[1], total - 1, len(usable))
    if max_k < NUM_COLORS_RANGE[0]:
        return None

    groups = None
    k = 0
    k_choices = list(range(NUM_COLORS_RANGE[0], max_k + 1))
    random.shuffle(k_choices)
    for k_try in k_choices:
        picked = _choose_groups(rows, cols, total, k_try, row_split, col_split)
        if picked is not None:
            groups = picked
            k = k_try
            break
    if groups is None:
        return None

    src_cols = random.sample(usable, k)
    tgt_cols = [random.choice([c for c in usable if c != src]) for src in src_cols]
    color_to_tgt = dict(zip(src_cols, tgt_cols))
    masks = {src: _sprite_mask(n) for src in src_cols}

    input_frame = _make_bordered_frame(frame_h, frame_w, bg)
    output_frame = _make_bordered_frame(frame_h, frame_w, bg)
    marker_rows: dict[int, set[int]] = {src: set() for src in src_cols}
    marker_cols: dict[int, set[int]] = {src: set() for src in src_cols}
    quadrant_counts = {"TL": 0, "TR": 0, "BL": 0, "BR": 0}
    cross_mode_sources = 0

    for grp, src in zip(groups, src_cols):
        if len(grp["modes"]) > 1:
            cross_mode_sources += 1
        for rr, cc in grp["slots"]:
            r0, c0 = _slot_top_left(top_margin, left_margin, n, rr, cc)
            mode = _slot_mode(rr, cc, row_split, col_split)
            quadrant_counts[mode] += 1
            ar, ac = _anchor_cell(r0, c0, n, mode)
            marker_rows[src].add(ar)
            marker_cols[src].add(ac)
            paste(output_frame, _colored_sprite(masks[src], color_to_tgt[src], bg), r0, c0, transparent=bg)

            check_mode = _anchor_mode(ar, ac, frame_h, frame_w)
            assert check_mode == mode, f"slot mode mismatch: expected {mode}, got {check_mode}"

    assert all(v > 0 for v in quadrant_counts.values()), f"quadrant coverage failure: {quadrant_counts}"
    assert cross_mode_sources > 0, "expected at least one source color to span multiple quadrant modes"

    marker_band = max(k, 1)
    ext_area = k * (12 + n * n + 10) + marker_band * (frame_h + frame_w)
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
        for pr in marker_rows[src]:
            grid[fr0 + pr, fc0 - band_idx] = src
            reserve[fr0 + pr, fc0 - band_idx] = FRAME_COL
        for pc in marker_cols[src]:
            grid[fr0 - band_idx, fc0 + pc] = src
            reserve[fr0 - band_idx, fc0 + pc] = FRAME_COL

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
    raise RuntimeError("failed to generate valid edb79dae combo input")
