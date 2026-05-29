# Combo variant of b5ca7ac4 (index 6)
# combo_spec: D4/0+D15/1
# Name: 虚线边框识别+不完整框架干扰

from __future__ import annotations

import random

import numpy as np

from arc_utils import crop_bbox, get_bg, paste

# Design analysis
# a) Concrete behavior and trace:
#    - The original task has two target groups of complete 5x5 frames, separated
#      by border color: teal frames move left, red frames move right.
#    - This combo keeps the 5x5 geometry and the row-preserving stacking rule,
#      but replaces color-grouping with line-style grouping and adds incomplete
#      frame distractors.
#    - All complete targets now share one visible frame color (teal/8):
#        solid complete:
#          8 8 8 8 8
#          8 i i i 8
#          8 i i i 8
#          8 i i i 8
#          8 8 8 8 8
#        dashed complete:
#          8 . 8 . 8
#          . i i i .
#          8 i i i 8
#          . i i i .
#          8 . 8 . 8
#      where "." is background and "i" is one uniform interior color.
#    - Incomplete distractors are derived from those same two families by
#      deleting one full border side. Example: a dashed frame missing its top
#      side still looks like a broken dashed square-like object, but it must be
#      ignored instead of moved.
#    - Example trace: suppose a solid complete frame at input col 3 and a second
#      solid complete frame at input col 9 overlap in row-span. The first takes
#      output col 0 and the second takes output col 5. A dashed complete frame
#      at input col 16 instead belongs to the right-moving group and is sorted
#      against other dashed completes by decreasing input col. A broken solid or
#      broken dashed frame at any column is discarded before stacking.
#    - Relative to the source task, two different decision points have changed:
#      first decide "complete vs incomplete", then decide "solid vs dashed" for
#      the complete ones.
# b) Complexity calibration:
#    - The variant is not reduced to a trivial detector: every sample contains
#      both complete solid and complete dashed movers, plus broken distractors
#      from both style families.
#    - The style vocabulary is intentionally small: only {solid, dashed} for
#      movers and only "missing one whole side" for distractors. This keeps the
#      task legible in 2-4 ARC examples while still forcing joint reasoning.
#    - Using a shared frame color removes any fallback color cue, so the line
#      pattern must be read directly from the pixels.
# c) Hidden-assumption audit of the original solver:
#    - Original assumption: `border.min() == border.max()` finds every target.
#      Broken here because dashed borders intentionally mix frame color and bg.
#    - Original assumption: frame color alone chooses the destination side.
#      Broken here because all valid movers share one frame color.
#    - New requirement: object detection must recognize exact complete solid and
#      exact complete dashed templates, while rejecting partial versions.
#    - The paste step still relies on `transparent=bg`, which remains correct:
#      dashed gaps and deleted sides are literal background and should not fill
#      in during relocation.
# d) Placement legality:
#    - Generation still treats every object as occupying its full 5x5 bounding
#      box, even when some border cells are bg. This prevents accidental window
#      overlaps or composite fake frames.
#    - Output construction remains output-first with at most two depth lanes per
#      side: left group uses cols {0, 5}, right group uses cols {W-5, W-10}.
#    - Distractors are inserted only into the input, never the output, and must
#      stay non-overlapping with all target 5x5 boxes.
# e) Visual observability:
#    - A human can see three recurring categories directly: continuous squares,
#      interrupted squares, and broken square-like distractors.
#    - Sorting still depends only on visible input columns. There is no hidden
#      ordering based on color ids, generation index, or any non-visual state.
# f) Visual neatness:
#    - Because every object keeps the same 5x5 footprint and all footprints are
#      disjoint, the grid remains sparse and readable despite the added noise.
#    - Restricting distractors to missing a single side avoids clutter from many
#      unrelated defect types.
# g) Inference friendliness:
#    - Interior colors are only identity markers. They remain uniform and avoid
#      both the background and the shared frame color, but they do not encode
#      grouping, completeness, or movement direction.
# h) Anti-degeneration checks:
#    - The generator requires at least one complete solid, one complete dashed,
#      one broken solid, and one broken dashed object in every sample.
#    - After rendering an input, the code rescans all 5x5 windows and verifies
#      that the detected complete/broken objects match the intended placed sets
#      exactly. This prevents accidental regression to the source rule.
# i) New solver assumptions and how generation enforces them:
#    - Assumption: every movable object is an exact 5x5 template. Generation
#      keeps all objects axis-aligned, in-bounds, and bbox-disjoint.
#    - Assumption: incomplete distractors do not combine with neighbors into a
#      fake complete frame. Generation forbids all 5x5 overlap for that reason.
#    - Assumption: stacking never needs a third lane. Generation keeps the same
#      one-edge-plus-one-inward structure as the source task.


GRID_H, GRID_W = 22, 22
SQ = 5
FRAME_COLOR = 8
N_SOLID_RANGE = (3, 4)
N_DASHED_RANGE = (3, 5)
N_DISTRACTOR_RANGE = (2, 3)
MAX_INWARD = 2
INPUT_COL_RETRIES = 50
DISTRACTOR_RETRIES = 80
GEN_RETRIES = 900
MISSING_SIDES = ("top", "bottom", "left", "right")


def _border_mask() -> np.ndarray:
    mask = np.zeros((SQ, SQ), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _side_mask(side: str) -> np.ndarray:
    mask = np.zeros((SQ, SQ), dtype=bool)
    if side == "top":
        mask[0, :] = True
    elif side == "bottom":
        mask[-1, :] = True
    elif side == "left":
        mask[:, 0] = True
    elif side == "right":
        mask[:, -1] = True
    else:
        raise ValueError(f"Unknown side: {side}")
    return mask


BORDER_MASK = _border_mask()
PARITY_MASK = (np.indices((SQ, SQ)).sum(axis=0) % 2) == 0
DASH_COLOR_MASK = BORDER_MASK & PARITY_MASK
DASH_GAP_MASK = BORDER_MASK & (~PARITY_MASK)
SIDE_MASKS = {side: _side_mask(side) for side in MISSING_SIDES}


def _uniform_inner_color(patch: np.ndarray, bg: int) -> int | None:
    inner = patch[1:-1, 1:-1]
    if int(inner.min()) != int(inner.max()):
        return None
    inner_color = int(inner[0, 0])
    if inner_color in {int(bg), FRAME_COLOR}:
        return None
    return inner_color


def _classify_complete_patch(patch: np.ndarray, bg: int) -> str | None:
    inner_color = _uniform_inner_color(patch, bg)
    if inner_color is None:
        return None

    border = patch[BORDER_MASK]
    if bool(np.all(border == FRAME_COLOR)):
        return "solid"

    if bool(np.all(patch[DASH_COLOR_MASK] == FRAME_COLOR)) and bool(
        np.all(patch[DASH_GAP_MASK] == int(bg))
    ):
        return "dashed"

    return None


def _classify_broken_patch(patch: np.ndarray, bg: int) -> str | None:
    inner_color = _uniform_inner_color(patch, bg)
    if inner_color is None:
        return None

    border = patch[BORDER_MASK]
    if not bool(np.isin(border, [int(bg), FRAME_COLOR]).all()):
        return None

    if bool(np.all(patch[DASH_GAP_MASK] == int(bg))) and bool(
        np.isin(patch[DASH_COLOR_MASK], [int(bg), FRAME_COLOR]).all()
    ):
        has_missing_colored = bool(np.any(patch[DASH_COLOR_MASK] == int(bg)))
        has_remaining_colored = bool(np.any(patch[DASH_COLOR_MASK] == FRAME_COLOR))
        if has_missing_colored and has_remaining_colored:
            return "dashed"

    has_missing_border = bool(np.any(border == int(bg)))
    has_remaining_border = bool(np.any(border == FRAME_COLOR))
    has_solid_only_cells = bool(np.any(patch[DASH_GAP_MASK] == FRAME_COLOR))
    if has_missing_border and has_remaining_border and has_solid_only_cells:
        return "solid"

    return None


def _scan_windows(g: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = g.shape
    found: list[dict[str, object]] = []
    for r0 in range(h - SQ + 1):
        for c0 in range(w - SQ + 1):
            patch = crop_bbox(g, (r0, c0, r0 + SQ - 1, c0 + SQ - 1))
            style = _classify_complete_patch(patch, bg)
            if style is not None:
                found.append(
                    {
                        "kind": "complete",
                        "style": style,
                        "r0": int(r0),
                        "c0": int(c0),
                        "patch": patch,
                    }
                )
                continue
            style = _classify_broken_patch(patch, bg)
            if style is not None:
                found.append(
                    {
                        "kind": "broken",
                        "style": style,
                        "r0": int(r0),
                        "c0": int(c0),
                        "patch": patch,
                    }
                )
    return found


def _rows_overlap(r_a: int, r_b: int) -> bool:
    return abs(int(r_a) - int(r_b)) < SQ


def _rects_overlap(r1: int, c1: int, r2: int, c2: int) -> bool:
    return abs(int(r1) - int(r2)) < SQ and abs(int(c1) - int(c2)) < SQ


def _any_pairwise_overlap(positions: list[tuple[int, int, np.ndarray]]) -> bool:
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if _rects_overlap(
                positions[i][0], positions[i][1], positions[j][0], positions[j][1]
            ):
                return True
    return False


def _make_complete_patch(inner_color: int, bg: int, *, solid: bool) -> np.ndarray:
    if solid:
        patch = np.full((SQ, SQ), FRAME_COLOR, dtype=int)
    else:
        patch = np.full((SQ, SQ), bg, dtype=int)
        patch[DASH_COLOR_MASK] = FRAME_COLOR
    patch[1:-1, 1:-1] = inner_color
    return patch


def _make_broken_patch(
    inner_color: int, bg: int, *, solid: bool, missing_side: str
) -> np.ndarray:
    patch = _make_complete_patch(inner_color, bg, solid=solid)
    patch[SIDE_MASKS[missing_side]] = bg
    return patch


def _assign_output_rows(n: int) -> tuple[list[int], list[int]] | None:
    max_inward = min(MAX_INWARD, n - 1)
    n_inward = random.randint(1, max_inward)
    n_edge = n - n_inward

    all_rows = list(range(GRID_H - SQ + 1))
    random.shuffle(all_rows)
    edge_rows: list[int] = []
    for r in all_rows:
        if all(not _rows_overlap(r, edge_r) for edge_r in edge_rows):
            edge_rows.append(r)
            if len(edge_rows) == n_edge:
                break
    if len(edge_rows) < n_edge:
        return None

    random.shuffle(all_rows)
    inward_rows: list[int] = []
    for r in all_rows:
        if not any(_rows_overlap(r, edge_r) for edge_r in edge_rows):
            continue
        if any(_rows_overlap(r, inward_r) for inward_r in inward_rows):
            continue
        inward_rows.append(r)
        if len(inward_rows) == n_inward:
            break
    if len(inward_rows) < n_inward:
        return None
    return edge_rows, inward_rows


def _place_side_squares(
    output: np.ndarray,
    edge_rows: list[int],
    inward_rows: list[int],
    inner_colors: list[int],
    *,
    solid: bool,
    bg: int,
    edge_col: int,
    inward_col: int,
) -> list[tuple[int, int, bool, np.ndarray]]:
    placed: list[tuple[int, int, bool, np.ndarray]] = []
    for i, r in enumerate(edge_rows):
        patch = _make_complete_patch(inner_colors[i], bg, solid=solid)
        paste(output, patch, r, edge_col, transparent=bg)
        placed.append((r, edge_col, True, patch))
    for i, r in enumerate(inward_rows):
        patch = _make_complete_patch(inner_colors[len(edge_rows) + i], bg, solid=solid)
        paste(output, patch, r, inward_col, transparent=bg)
        placed.append((r, inward_col, False, patch))
    return placed


def _derive_input_positions(
    sq_list: list[tuple[int, int, bool, np.ndarray]], *, ascending: bool
) -> list[tuple[int, int, np.ndarray]] | None:
    edges = [(r, out_c, patch) for (r, out_c, is_edge, patch) in sq_list if is_edge]
    inwards = [
        (r, out_c, patch) for (r, out_c, is_edge, patch) in sq_list if not is_edge
    ]
    inward_deps = [
        [i for i, (edge_r, _, _) in enumerate(edges) if _rows_overlap(inward_r, edge_r)]
        for (inward_r, _, _) in inwards
    ]

    n = len(sq_list)
    col_range = list(range(GRID_W - SQ + 1))
    if len(col_range) < n:
        return None

    for _ in range(INPUT_COL_RETRIES):
        edge_indices = list(range(len(edges)))
        random.shuffle(edge_indices)
        order: list[tuple[str, int]] = [("edge", i) for i in edge_indices]

        inward_order = list(range(len(inwards)))
        random.shuffle(inward_order)
        for inward_idx in inward_order:
            latest_pos = max(
                (
                    pos
                    for dep in inward_deps[inward_idx]
                    for pos, (typ, idx) in enumerate(order)
                    if typ == "edge" and idx == dep
                ),
                default=-1,
            )
            insert_pos = random.randint(latest_pos + 1, len(order))
            order.insert(insert_pos, ("inward", inward_idx))

        ordered = [edges[idx] if typ == "edge" else inwards[idx] for typ, idx in order]
        chosen_cols = sorted(random.sample(col_range, n))
        if not ascending:
            chosen_cols = chosen_cols[::-1]

        result: list[tuple[int, int, np.ndarray]] = []
        ok = True
        for i, (r, out_c, patch) in enumerate(ordered):
            in_c = chosen_cols[i]
            if in_c == out_c:
                ok = False
                break
            result.append((r, in_c, patch))
        if ok and not _any_pairwise_overlap(result):
            return result

    return None


def _place_distractors(
    occupied: list[tuple[int, int, np.ndarray]], bg: int
) -> list[tuple[int, int, str, np.ndarray]] | None:
    n_distractors = random.randint(*N_DISTRACTOR_RANGE)
    styles = ["solid", "dashed"]
    while len(styles) < n_distractors:
        styles.append(random.choice(["solid", "dashed"]))
    random.shuffle(styles)

    distractors: list[tuple[int, int, str, np.ndarray]] = []
    taken = [(r, c) for r, c, _ in occupied]

    for style in styles:
        inner_choices = [c for c in range(10) if c not in {bg, FRAME_COLOR}]
        solid = style == "solid"
        placed = False
        for _ in range(DISTRACTOR_RETRIES):
            r = random.randint(0, GRID_H - SQ)
            c = random.randint(0, GRID_W - SQ)
            if any(_rects_overlap(r, c, tr, tc) for tr, tc in taken):
                continue
            patch = _make_broken_patch(
                random.choice(inner_choices),
                bg,
                solid=solid,
                missing_side=random.choice(MISSING_SIDES),
            )
            if _classify_broken_patch(patch, bg) != style:
                continue
            distractors.append((r, c, style, patch))
            taken.append((r, c))
            placed = True
            break
        if not placed:
            return None
    return distractors


def _detected_signature(items: list[dict[str, object]], kind: str) -> set[tuple[int, int, str]]:
    return {
        (int(item["r0"]), int(item["c0"]), str(item["style"]))
        for item in items
        if str(item["kind"]) == kind
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Relocate only complete 5x5 teal frames, grouping them by solid-vs-dashed border style.

    Input structure:
    - The background is the most common color.
    - The grid contains scattered 5x5 teal-framed objects with uniform 3x3 interiors.
    - Some are complete solid frames, some are complete dashed frames with bg gaps, and some are incomplete broken frames missing one border side.

    Rule:
    - Ignore every incomplete frame; only complete 5x5 frames participate.
    - Complete solid frames move to the left edge and complete dashed frames move to the right edge, keeping their original rows.
    - Solid frames are processed by increasing input column, dashed frames by decreasing input column.
    - If a frame overlaps the row-span of a frame already placed in that side's edge lane, shift it inward by one frame width instead.

    Key insight:
    - The solver must combine two tests on the same 5x5 window: completeness decides whether the object matters at all, and border continuity decides which side's stacking rule to apply.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))
    out = np.full_like(g, bg)

    def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
        return not (int(a1) < int(b0) or int(b1) < int(a0))

    scanned = _scan_windows(g, bg)
    complete = [item for item in scanned if str(item["kind"]) == "complete"]

    solid_squares = sorted(
        [item for item in complete if str(item["style"]) == "solid"],
        key=lambda item: int(item["c0"]),
    )
    placed_at_edge: list[tuple[int, int]] = []
    for square in solid_squares:
        r0 = int(square["r0"])
        r1 = r0 + SQ - 1
        has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_edge)
        dest_c = SQ if has_overlap else 0
        paste(out, square["patch"], r0, dest_c, transparent=bg)
        if dest_c == 0:
            placed_at_edge.append((r0, r1))

    dashed_squares = sorted(
        [item for item in complete if str(item["style"]) == "dashed"],
        key=lambda item: -int(item["c0"]),
    )
    placed_at_edge = []
    for square in dashed_squares:
        r0 = int(square["r0"])
        r1 = r0 + SQ - 1
        has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_edge)
        dest_c = w - 2 * SQ if has_overlap else w - SQ
        paste(out, square["patch"], r0, dest_c, transparent=bg)
        if dest_c == w - SQ:
            placed_at_edge.append((r0, r1))

    return out


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([c for c in range(10) if c != FRAME_COLOR])
    n_solid = random.randint(*N_SOLID_RANGE)
    n_dashed = random.randint(*N_DASHED_RANGE)

    solid_assign = _assign_output_rows(n_solid)
    if solid_assign is None:
        return None
    dashed_assign = _assign_output_rows(n_dashed)
    if dashed_assign is None:
        return None
    solid_edge_rows, solid_inward_rows = solid_assign
    dashed_edge_rows, dashed_inward_rows = dashed_assign

    avail_solid = [c for c in range(10) if c not in {bg, FRAME_COLOR}]
    avail_dashed = [c for c in range(10) if c not in {bg, FRAME_COLOR}]
    if len(avail_solid) < n_solid or len(avail_dashed) < n_dashed:
        return None
    solid_colors = random.sample(avail_solid, n_solid)
    dashed_colors = random.sample(avail_dashed, n_dashed)

    output = np.full((GRID_H, GRID_W), bg, dtype=int)
    solid_out = _place_side_squares(
        output,
        solid_edge_rows,
        solid_inward_rows,
        solid_colors,
        solid=True,
        bg=bg,
        edge_col=0,
        inward_col=SQ,
    )
    dashed_out = _place_side_squares(
        output,
        dashed_edge_rows,
        dashed_inward_rows,
        dashed_colors,
        solid=False,
        bg=bg,
        edge_col=GRID_W - SQ,
        inward_col=GRID_W - 2 * SQ,
    )

    solid_input = _derive_input_positions(solid_out, ascending=True)
    if solid_input is None:
        return None
    dashed_input = _derive_input_positions(dashed_out, ascending=False)
    if dashed_input is None:
        return None

    all_targets = solid_input + dashed_input
    if _any_pairwise_overlap(all_targets):
        return None

    distractors = _place_distractors(all_targets, bg)
    if distractors is None:
        return None

    inp = np.full((GRID_H, GRID_W), bg, dtype=int)
    for r, c, patch in all_targets:
        paste(inp, patch, r, c, transparent=bg)
    for r, c, _style, patch in distractors:
        paste(inp, patch, r, c, transparent=bg)

    scanned = _scan_windows(inp, bg)
    expected_complete = {
        (int(r), int(c), "solid") for r, c, _ in solid_input
    } | {(int(r), int(c), "dashed") for r, c, _ in dashed_input}
    expected_broken = {(int(r), int(c), style) for r, c, style, _ in distractors}
    if _detected_signature(scanned, "complete") != expected_complete:
        return None
    if _detected_signature(scanned, "broken") != expected_broken:
        return None
    if not {"solid", "dashed"} <= {style for _, _, style in expected_broken}:
        return None

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
