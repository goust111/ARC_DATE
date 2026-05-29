# Combo variant of b5ca7ac4 (index 30)
# combo_spec: D4/0+D8/1+D15/1
# Name: 虚线边框+不完整干扰+旋转变换

from __future__ import annotations

import random

import numpy as np

from arc_utils import crop_bbox, get_bg, paste

# Design analysis
# a) Concrete behavior and trace:
#    - Valid movers are still 5x5 framed objects, but both groups now share one
#      fixed frame color (teal/8). The visible group cue is border continuity:
#        solid target:
#          8 8 8 8 8
#          8 a a a 8
#          8 a b a 8
#          8 a a c 8
#          8 8 8 8 8
#        dashed target:
#          8 . 8 . 8
#          . a a a .
#          8 a b a 8
#          . a a c .
#          8 . 8 . 8
#      Here "." is background; a/b/c are inner colors. Every valid target uses
#      an asymmetric 3x3 interior, so rotating it by 90 degrees changes the
#      visible accent placement even though the 5x5 frame remains square.
#    - Distractors reuse the same 5x5 footprint and same shared frame color but
#      are incomplete: one whole border side is deleted from an otherwise solid
#      or dashed frame. They stay in the input and vanish from the output.
#    - Trace example: suppose a solid frame at col 3 and another solid frame at
#      col 11 overlap in rows. The leftmost solid is processed first, moves to
#      col 0, and is rotated 90 degrees CCW before pasting. The later solid
#      overlaps that edge lane, so it moves to col 5 and is also rotated. A
#      broken dashed "U" shape nearby is ignored entirely. Compared with the
#      source task, grouping is no longer color-based, filtering now removes
#      incomplete lookalikes, and each moved frame changes orientation.
# b) Complexity calibration:
#    - Only three visually coherent categories exist: complete solid, complete
#      dashed, and incomplete. This is enough to force the fused reasoning
#      chain without exploding the hypothesis space.
#    - Rotation is made observable by asymmetric interiors, but the vocabulary
#      stays small: one shared border color plus 2-color interior motifs.
#    - Each sample contains several solid targets, several dashed targets, and
#      at least one broken solid and one broken dashed distractor, so every
#      sub-rule has repeated support.
# c) Original assumptions that break under the variant:
#    - `border.min() == border.max()` is no longer a valid detector because a
#      dashed border intentionally mixes frame color and background.
#    - `frame in {2, 8}` no longer separates the two moving groups because both
#      groups now use the same frame color.
#    - "Every 5x5 frame-like object moves" is false: incomplete solid/dashed
#      frames must be filtered out before sorting and stacking.
#    - "Moved patch equals detected patch" is false: the solver must rotate each
#      accepted patch before pasting it into the output.
# d) Placement legality:
#    - The generator still treats every object, including dashed and broken
#      ones, as occupying its full 5x5 bbox. This keeps scans unambiguous and
#      avoids overlap-based fake frames.
#    - Output keeps the original two-lane geometry per side: left group at
#      cols {0, 5}, right group at cols {W-5, W-10}. Since rotation preserves
#      5x5 size, no new spatial conflicts are introduced.
# e) Visual observability:
#    - A human can separate the groups from border continuity alone because all
#      complete frames share one color but differ clearly as continuous vs
#      interrupted outlines. Broken frames visibly miss an entire side.
#    - Ordering depends only on visible input columns, and inward stacking still
#      depends on visible row-span overlap. No hidden color-id ordering is used.
#    - Rotation is inferable because the asymmetric interior motifs change
#      orientation consistently across all moved targets.
# f) Visual neatness:
#    - The grid remains sparse and all objects preserve the familiar 5x5 size.
#      Distractors use the same visual language instead of introducing unrelated
#      shapes, so the scene is busier than the source task but still readable.
# g) Variant anti-degeneration check:
#    - Valid targets never use rotation-symmetric interiors, so the rotated
#      output cannot collapse back to the no-rotation behavior.
#    - Every generated sample contains both complete styles and both broken
#      styles, so neither a color-only rule nor a "move every frame-like thing"
#      rule can explain the examples.
# h) New solver assumptions and generator guarantees:
#    - The solver assumes a valid object is an exact 5x5 solid or dashed border
#      template with a 3x3 interior containing no bg/frame cells. The generator
#      enforces that exact template for all movers.
#    - The solver assumes broken distractors do not combine with neighbors into
#      accidental valid windows. The generator enforces pairwise non-overlap of
#      all 5x5 footprints.
#    - The solver assumes only two output depths per side. The generator keeps
#      the original edge-lane + inward-lane construction and validates by
#      re-running `main` on every produced input.


GRID_H, GRID_W = 22, 22
SQ = 5
FRAME_COLOR = 8
N_SOLID_RANGE = (3, 4)
N_DASHED_RANGE = (3, 5)
N_DISTRACTOR_RANGE = (2, 3)
MAX_INWARD = 2
INPUT_COL_RETRIES = 50
DISTRACTOR_RETRIES = 120
GEN_RETRIES = 900
MISSING_SIDES = ("top", "bottom", "left", "right")


def _border_mask() -> np.ndarray:
    mask = np.zeros((SQ, SQ), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


BORDER_MASK = _border_mask()
PARITY_MASK = (np.indices((SQ, SQ)).sum(axis=0) % 2) == 0
DASH_COLOR_MASK = BORDER_MASK & PARITY_MASK
DASH_GAP_MASK = BORDER_MASK & (~PARITY_MASK)
INNER_NONCENTER = [
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
]


def _rotate_patch(patch: np.ndarray) -> np.ndarray:
    return np.rot90(patch, k=1)


def _classify_patch(patch: np.ndarray, bg: int) -> str | None:
    inner = patch[1:-1, 1:-1]
    if bool((inner == int(bg)).any()) or bool((inner == FRAME_COLOR).any()):
        return None

    solid_ok = bool(np.all(patch[BORDER_MASK] == FRAME_COLOR))
    if solid_ok:
        return "solid"

    dashed_ok = bool(np.all(patch[DASH_COLOR_MASK] == FRAME_COLOR)) and bool(
        np.all(patch[DASH_GAP_MASK] == int(bg))
    )
    if dashed_ok:
        return "dashed"

    return None


def _collect_complete_frames(g: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = g.shape
    frames: list[dict[str, object]] = []
    for r0 in range(h - SQ + 1):
        for c0 in range(w - SQ + 1):
            patch = crop_bbox(g, (r0, c0, r0 + SQ - 1, c0 + SQ - 1))
            style = _classify_patch(patch, bg)
            if style is None:
                continue
            frames.append(
                {"style": style, "r0": int(r0), "c0": int(c0), "patch": patch.copy()}
            )
    return frames


def _make_inner_pattern(
    bg: int, used: set[tuple[int, ...]]
) -> np.ndarray:
    palette = [c for c in range(10) if c not in {bg, FRAME_COLOR}]
    for _ in range(300):
        dominant = random.choice(palette)
        accent = random.choice([c for c in palette if c != dominant])
        accent_pos = random.choice(INNER_NONCENTER)
        arr = np.full((3, 3), dominant, dtype=int)
        arr[int(accent_pos[0]), int(accent_pos[1])] = accent
        if np.array_equal(arr, np.rot90(arr, 1)):
            continue
        sig = tuple(int(x) for x in arr.ravel())
        if sig in used:
            continue
        used.add(sig)
        return arr
    raise RuntimeError("Failed to generate a unique asymmetric inner pattern")


def _make_patch(inner: np.ndarray, bg: int, *, solid: bool) -> np.ndarray:
    if solid:
        patch = np.full((SQ, SQ), FRAME_COLOR, dtype=int)
    else:
        patch = np.full((SQ, SQ), bg, dtype=int)
        patch[DASH_COLOR_MASK] = FRAME_COLOR
    patch[1:-1, 1:-1] = inner
    return patch


def _make_broken_patch(
    inner: np.ndarray, bg: int, *, solid: bool, missing_side: str
) -> np.ndarray:
    patch = _make_patch(inner, bg, solid=solid)
    if missing_side == "top":
        patch[0, :] = bg
    elif missing_side == "bottom":
        patch[-1, :] = bg
    elif missing_side == "left":
        patch[:, 0] = bg
    elif missing_side == "right":
        patch[:, -1] = bg
    else:
        raise ValueError(f"Unknown missing_side: {missing_side}")
    return patch


def _rows_overlap(r_a: int, r_b: int) -> bool:
    return abs(int(r_a) - int(r_b)) < SQ


def _rects_overlap(r1: int, c1: int, r2: int, c2: int) -> bool:
    return abs(int(r1) - int(r2)) < SQ and abs(int(c1) - int(c2)) < SQ


def _any_pairwise_overlap(positions: list[tuple[int, int, np.ndarray]]) -> bool:
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if _rects_overlap(
                positions[i][0],
                positions[i][1],
                positions[j][0],
                positions[j][1],
            ):
                return True
    return False


def _assign_output_rows(n: int) -> tuple[list[int], list[int]] | None:
    max_inward = min(MAX_INWARD, n - 1)
    n_inward = random.randint(1, max_inward)
    n_edge = n - n_inward

    all_rows = list(range(GRID_H - SQ + 1))
    random.shuffle(all_rows)
    edge_rows: list[int] = []
    for r in all_rows:
        if all(not _rows_overlap(r, er) for er in edge_rows):
            edge_rows.append(r)
            if len(edge_rows) == n_edge:
                break
    if len(edge_rows) < n_edge:
        return None

    random.shuffle(all_rows)
    inward_rows: list[int] = []
    for r in all_rows:
        if not any(_rows_overlap(r, er) for er in edge_rows):
            continue
        if any(_rows_overlap(r, ir) for ir in inward_rows):
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
    inner_patterns: list[np.ndarray],
    *,
    solid: bool,
    bg: int,
    edge_col: int,
    inward_col: int,
) -> list[tuple[int, int, bool, np.ndarray]]:
    placed: list[tuple[int, int, bool, np.ndarray]] = []
    for i, r in enumerate(edge_rows):
        input_patch = _make_patch(inner_patterns[i], bg, solid=solid)
        paste(output, _rotate_patch(input_patch), r, edge_col, transparent=bg)
        placed.append((r, edge_col, True, input_patch))
    for i, r in enumerate(inward_rows):
        idx = len(edge_rows) + i
        input_patch = _make_patch(inner_patterns[idx], bg, solid=solid)
        paste(output, _rotate_patch(input_patch), r, inward_col, transparent=bg)
        placed.append((r, inward_col, False, input_patch))
    return placed


def _derive_input_positions(
    sq_list: list[tuple[int, int, bool, np.ndarray]],
    *,
    ascending: bool,
) -> list[tuple[int, int, np.ndarray]] | None:
    edges = [(r, oc, p) for (r, oc, is_edge, p) in sq_list if is_edge]
    inwards = [(r, oc, p) for (r, oc, is_edge, p) in sq_list if not is_edge]
    inward_deps = [
        [i for i, (er, _, _) in enumerate(edges) if _rows_overlap(ir, er)]
        for (ir, _, _) in inwards
    ]

    n = len(sq_list)
    col_range = list(range(0, GRID_W - SQ + 1))
    if len(col_range) < n:
        return None

    for _ in range(INPUT_COL_RETRIES):
        edge_indices = list(range(len(edges)))
        random.shuffle(edge_indices)
        order: list[tuple[str, int]] = [("edge", i) for i in edge_indices]

        inward_indices = list(range(len(inwards)))
        random.shuffle(inward_indices)
        for inward_idx in inward_indices:
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
    occupied: list[tuple[int, int, np.ndarray]],
    bg: int,
    used_patterns: set[tuple[int, ...]],
) -> list[tuple[int, int, np.ndarray]] | None:
    n_distractors = random.randint(*N_DISTRACTOR_RANGE)
    styles = ["solid", "dashed"]
    while len(styles) < n_distractors:
        styles.append(random.choice(["solid", "dashed"]))
    random.shuffle(styles)

    distractors: list[tuple[int, int, np.ndarray]] = []
    taken = [(r, c) for r, c, _ in occupied]
    for style in styles:
        placed = False
        for _ in range(DISTRACTOR_RETRIES):
            r = random.randint(0, GRID_H - SQ)
            c = random.randint(0, GRID_W - SQ)
            if any(_rects_overlap(r, c, tr, tc) for tr, tc in taken):
                continue
            inner = _make_inner_pattern(bg, used_patterns)
            patch = _make_broken_patch(
                inner,
                bg,
                solid=(style == "solid"),
                missing_side=random.choice(MISSING_SIDES),
            )
            if _classify_patch(patch, bg) is not None:
                continue
            distractors.append((r, c, patch))
            taken.append((r, c))
            placed = True
            break
        if not placed:
            return None
    return distractors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move complete 5x5 frames to the sides by border style, then rotate them.

    Input structure:
    - The background is the most common color.
    - All frame-like objects use the same teal border color and occupy 5x5 boxes.
    - Some boxes are complete solid frames, some are complete dashed frames, and
      some are incomplete distractors missing one whole border side.

    Rule:
    - Ignore incomplete frame candidates; only exact solid or exact dashed 5x5
      frames participate.
    - Solid frames move to the left edge and dashed frames move to the right
      edge, keeping their original row.
    - Solid frames are processed by increasing input column; dashed frames by
      decreasing input column. Vertical row-span overlap with an already placed
      edge-lane frame pushes the later frame one lane inward.
    - After its destination lane is chosen, each accepted frame is rotated 90
      degrees counterclockwise before it is pasted into the output.

    Key insight:
    - The solver must fuse three cues in sequence: reject incomplete lookalikes,
      separate the remaining frames by solid vs dashed border continuity, and
      only then apply the original edge-stacking rule with a final quarter-turn.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))
    out = np.full_like(g, bg)

    def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
        return not (int(a1) < int(b0) or int(b1) < int(a0))

    frames = _collect_complete_frames(g, bg)

    solid_frames = sorted(
        [frame for frame in frames if frame["style"] == "solid"],
        key=lambda frame: int(frame["c0"]),
    )
    placed_at_edge: list[tuple[int, int]] = []
    for frame in solid_frames:
        r0 = int(frame["r0"])
        r1 = int(r0 + SQ - 1)
        has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_edge)
        dest_c = SQ if has_overlap else 0
        paste(out, _rotate_patch(frame["patch"]), r0, dest_c, transparent=bg)
        if dest_c == 0:
            placed_at_edge.append((r0, r1))

    dashed_frames = sorted(
        [frame for frame in frames if frame["style"] == "dashed"],
        key=lambda frame: -int(frame["c0"]),
    )
    placed_at_edge = []
    for frame in dashed_frames:
        r0 = int(frame["r0"])
        r1 = int(r0 + SQ - 1)
        has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_edge)
        dest_c = w - 2 * SQ if has_overlap else w - SQ
        paste(out, _rotate_patch(frame["patch"]), r0, dest_c, transparent=bg)
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

    used_patterns: set[tuple[int, ...]] = set()
    solid_patterns = [_make_inner_pattern(bg, used_patterns) for _ in range(n_solid)]
    dashed_patterns = [_make_inner_pattern(bg, used_patterns) for _ in range(n_dashed)]

    output = np.full((GRID_H, GRID_W), bg, dtype=int)
    solid_out = _place_side_squares(
        output,
        solid_edge_rows,
        solid_inward_rows,
        solid_patterns,
        solid=True,
        bg=bg,
        edge_col=0,
        inward_col=SQ,
    )
    dashed_out = _place_side_squares(
        output,
        dashed_edge_rows,
        dashed_inward_rows,
        dashed_patterns,
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

    distractors = _place_distractors(all_targets, bg, used_patterns)
    if distractors is None:
        return None

    inp = np.full((GRID_H, GRID_W), bg, dtype=int)
    for r, c, patch in all_targets + distractors:
        paste(inp, patch, r, c, transparent=bg)

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
