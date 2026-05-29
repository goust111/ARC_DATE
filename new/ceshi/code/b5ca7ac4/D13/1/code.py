# Variant of b5ca7ac4
# Dimension: D13 — 符号指令——堆叠方向的编码 (symbol_instructions)
# Change: 无显式符号指令，堆叠方向（青色向左、红色向右）和处理顺序完全靠样例隐式推断
#   → 输入左上角新增小图例：2x2 颜色样块 + 箭头符号，显式指示“该框色向哪一侧堆叠”。

from __future__ import annotations

import random
import numpy as np

from arc_utils import crop_bbox, get_bg, paste

# -----------------------------------------------------------------------------
# Design analysis for the D13 variant
#
# a) Concrete behavior and trace
#    - Input still contains scattered 5x5 framed squares with frame colors 8 and 2.
#    - New element: a fixed top-left legend made of two entries. Each entry is:
#      a 2x2 solid color swatch (the frame color being referred to) plus a small arrow.
#    - Top entry always shows the left-moving group; bottom entry always shows the
#      right-moving group. The swatch colors swap across generated examples, so the
#      legend truly controls which frame color goes left and which goes right.
#    - Output removes the legend and only shows the rearranged 5x5 squares.
#    - Local trace:
#      original logic: frame 8 always sorted by input column ascending and packed to
#      col 0/5; frame 2 always sorted descending and packed to col W-5/W-10.
#      variant logic: first decode legend, e.g. if red swatch is next to the left arrow,
#      then red now uses the "left-group" logic and teal uses the "right-group" logic.
#
# b) Complexity calibration
#    - Only two frame colors exist, so the legend needs just two entries.
#    - The legend changes only one task degree of freedom: color -> side mapping.
#      The overlap/inward-stack mechanism remains identical to the source task.
#    - Mapping is binary and repeated twice in the same legend, which keeps the rule
#      visible without introducing hard-to-learn extra symbols.
#
# c) Old assumptions that break under the variant
#    - `if frame_col not in {2, 8}` still holds.
#    - The original hardcoded split "8 is left group / 2 is right group" no longer
#      holds; `main()` must read a mapping from the legend before grouping squares.
#    - The original solver could scan the whole board for 5x5 squares. That becomes
#      unsafe once the input contains a new symbolic area, so this variant adds a
#      legend keep-out zone and the generator guarantees no real square enters it.
#
# d) Placement legality
#    - The generator keeps the original 22x22 canvas and original 5x5 object sizes.
#    - Legend lives in a fixed 11x7 top-left box; input squares are forbidden from a
#      slightly larger keep-out rectangle around it, so legend pixels cannot merge
#      with framed squares or create accidental 5x5 detections.
#    - Output is still built output-first and validated by calling `main(input)`.
#
# e) Visual observability
#    - The rule change is visible because some examples swap the usual color-to-side
#      assignment. A human can inspect the corner legend and see why.
#    - No hidden use of color ids is added: the arrow shape itself encodes left/right.
#
# f) Visual cleanliness
#    - The legend is isolated in one corner and removed from the output, so the main
#      rearrangement area stays as clean as in the original task.
#
# g) Color inferability
#    - No color transformation occurs. Colors only label groups through an adjacent
#      swatch, which is explicit and directly observable.
#
# 9) New solver assumptions checked against the generator
#    - `main()` assumes the legend is at the fixed top-left layout used below.
#    - `main()` assumes each legend swatch is a uniform 2x2 block in frame color 2 or 8.
#    - `main()` assumes the arrow region is a uniform non-bg symbol whose occupied cells
#      match exactly one of the two masks below.
#    - `generate_input()` enforces all of these assumptions and keeps real squares out
#      of the legend keep-out zone.
# -----------------------------------------------------------------------------


GRID_H, GRID_W = 22, 22
SQ = 5
AZURE, RED = 8, 2
MAX_INWARD = 2
INPUT_COL_RETRIES = 120
GEN_RETRIES = 700
N_AZURE_RANGE = (3, 4)
N_RED_RANGE = (3, 5)

LEGEND_ENTRY_H = 5
LEGEND_ENTRY_W = 7
LEGEND_GAP = 1
LEGEND_H = 2 * LEGEND_ENTRY_H + LEGEND_GAP
LEGEND_W = LEGEND_ENTRY_W
LEGEND_PAD = 1
KEEP_OUT_R1 = LEGEND_H - 1 + LEGEND_PAD
KEEP_OUT_C1 = LEGEND_W - 1 + LEGEND_PAD

SWATCH_CELLS = {(1, 0), (1, 1), (2, 0), (2, 1)}
LEFT_ARROW_MASK = {
    (0, 1),
    (1, 0), (1, 1),
    (2, 0), (2, 1), (2, 2), (2, 3),
    (3, 0), (3, 1),
    (4, 1),
}
RIGHT_ARROW_MASK = {
    (0, 2),
    (1, 2), (1, 3),
    (2, 0), (2, 1), (2, 2), (2, 3),
    (3, 2), (3, 3),
    (4, 2),
}


def _entry_origin(idx: int) -> tuple[int, int]:
    return idx * (LEGEND_ENTRY_H + LEGEND_GAP), 0


def _rects_overlap(r1: int, c1: int, h1: int, w1: int, r2: int, c2: int, h2: int, w2: int) -> bool:
    return not (
        int(r1 + h1 - 1) < int(r2)
        or int(r2 + h2 - 1) < int(r1)
        or int(c1 + w1 - 1) < int(c2)
        or int(c2 + w2 - 1) < int(c1)
    )


def _rows_overlap(r_a: int, r_b: int) -> bool:
    return abs(int(r_a) - int(r_b)) < SQ


def _any_pairwise_overlap(positions: list[tuple[int, int, np.ndarray]]) -> bool:
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if _rects_overlap(positions[i][0], positions[i][1], SQ, SQ,
                              positions[j][0], positions[j][1], SQ, SQ):
                return True
    return False


def _intersects_legend_keepout(r0: int, c0: int) -> bool:
    return _rects_overlap(int(r0), int(c0), SQ, SQ, 0, 0, KEEP_OUT_R1 + 1, KEEP_OUT_C1 + 1)


def _build_legend(bg: int, left_frame: int, right_frame: int, arrow_color: int) -> np.ndarray:
    patch = np.full((LEGEND_H, LEGEND_W), bg, dtype=int)
    for idx, (frame_color, arrow_mask) in enumerate(
        [(left_frame, LEFT_ARROW_MASK), (right_frame, RIGHT_ARROW_MASK)]
    ):
        r0, c0 = _entry_origin(idx)
        for dr, dc in SWATCH_CELLS:
            patch[r0 + dr, c0 + dc] = frame_color
        for dr, dc in arrow_mask:
            patch[r0 + dr, c0 + 3 + dc] = arrow_color
    return patch


def _decode_legend(g: np.ndarray, bg: int) -> dict[int, str]:
    frame_to_side: dict[int, str] = {}
    for idx, expected_side, expected_mask in [
        (0, "left", LEFT_ARROW_MASK),
        (1, "right", RIGHT_ARROW_MASK),
    ]:
        r0, c0 = _entry_origin(idx)
        swatch = g[r0 : r0 + LEGEND_ENTRY_H, c0 : c0 + 2]
        swatch_vals = {int(swatch[dr, dc]) for dr, dc in SWATCH_CELLS}
        swatch_vals.discard(bg)
        assert len(swatch_vals) == 1, "legend swatch must be a single visible frame color"
        frame_color = swatch_vals.pop()
        assert frame_color in {AZURE, RED}, "legend swatch must use one of the square frame colors"

        arrow = g[r0 : r0 + LEGEND_ENTRY_H, c0 + 3 : c0 + 7]
        arrow_mask = {
            (dr, dc)
            for dr in range(LEGEND_ENTRY_H)
            for dc in range(4)
            if int(arrow[dr, dc]) != bg
        }
        assert arrow_mask == expected_mask, "legend arrow does not match expected symbol layout"
        frame_to_side[int(frame_color)] = expected_side

    assert set(frame_to_side.keys()) == {AZURE, RED}, "legend must mention both frame colors exactly once"
    return frame_to_side


def _find_squares(g: np.ndarray, frame_to_side: dict[int, str]) -> list[dict[str, object]]:
    H, W = g.shape
    squares: list[dict[str, object]] = []
    for r0 in range(H - SQ + 1):
        for c0 in range(W - SQ + 1):
            if _rects_overlap(r0, c0, SQ, SQ, 0, 0, KEEP_OUT_R1 + 1, KEEP_OUT_C1 + 1):
                continue
            patch = crop_bbox(g, (r0, c0, r0 + SQ - 1, c0 + SQ - 1))
            border = np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])
            if int(border.min()) != int(border.max()):
                continue
            frame_col = int(border[0])
            if frame_col not in frame_to_side:
                continue
            inner = patch[1:-1, 1:-1]
            if not bool((inner != frame_col).any()):
                continue
            squares.append({"frame": frame_col, "r0": int(r0), "c0": int(c0), "patch": patch})
    return squares


def _place_group(
    out: np.ndarray,
    squares: list[dict[str, object]],
    edge_col: int,
    inward_col: int,
    ascending: bool,
    bg: int,
) -> None:
    ordered = sorted(squares, key=lambda s: int(s["c0"]), reverse=not ascending)
    placed_at_edge: list[tuple[int, int]] = []
    for s in ordered:
        r0 = int(s["r0"])
        r1 = r0 + SQ - 1
        has_overlap = any(not (r1 < pr0 or pr1 < r0) for pr0, pr1 in placed_at_edge)
        dest_c = int(inward_col) if has_overlap else int(edge_col)
        paste(out, s["patch"], r0, dest_c, transparent=bg)
        if dest_c == int(edge_col):
            placed_at_edge.append((r0, r1))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rearrange bordered 5×5 squares according to a corner legend that encodes direction by symbol.

    Input structure:
    - bg = most common color.
    - Scattered 5×5 squares with frame colors 8 and 2, plus a small top-left legend.
    - Each legend entry is a frame-color swatch paired with a left- or right-pointing arrow.

    Rule:
    - Read the legend first to decide which frame color is the left-moving group and which is the right-moving group.
    - Left-moving squares are processed by input column ascending and stacked at col 0, or col 5 if they row-overlap an already placed edge square.
    - Right-moving squares are processed by input column descending and stacked at col W−5, or col W−10 on such overlaps.
    - Every square keeps its original row; the legend itself is instruction-only and does not appear in the output.

    Key insight: The rearrangement mechanism is unchanged from the source task; the new work is decoding the corner swatch+arrow legend before assigning each frame color to the left or right stack.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    frame_to_side = _decode_legend(g, bg)
    squares = _find_squares(g, frame_to_side)

    out = np.full_like(g, bg)
    left_squares = [s for s in squares if frame_to_side[int(s["frame"])] == "left"]
    right_squares = [s for s in squares if frame_to_side[int(s["frame"])] == "right"]

    _place_group(out, left_squares, edge_col=0, inward_col=SQ, ascending=True, bg=bg)
    _place_group(out, right_squares, edge_col=W - SQ, inward_col=W - 2 * SQ, ascending=False, bg=bg)
    return out


def _make_patch(frame_color: int, inner_color: int) -> np.ndarray:
    p = np.full((SQ, SQ), frame_color, dtype=int)
    p[1:-1, 1:-1] = inner_color
    return p


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
    colors: list[int],
    frame_color: int,
    edge_col: int,
    inward_col: int,
) -> list[tuple[int, int, bool, np.ndarray]]:
    placed: list[tuple[int, int, bool, np.ndarray]] = []
    for i, r in enumerate(edge_rows):
        patch = _make_patch(frame_color, colors[i])
        paste(output, patch, r, edge_col)
        placed.append((r, edge_col, True, patch))
    for i, r in enumerate(inward_rows):
        patch = _make_patch(frame_color, colors[len(edge_rows) + i])
        paste(output, patch, r, inward_col)
        placed.append((r, inward_col, False, patch))
    return placed


def _derive_input_positions(
    sq_list: list[tuple[int, int, bool, np.ndarray]],
    ascending: bool,
) -> list[tuple[int, int, np.ndarray]] | None:
    edges = [(r, oc, p) for (r, oc, is_edge, p) in sq_list if is_edge]
    inwards = [(r, oc, p) for (r, oc, is_edge, p) in sq_list if not is_edge]
    inward_deps = [[i for i, (er, _, _) in enumerate(edges) if _rows_overlap(ir, er)]
                   for (ir, _, _) in inwards]

    n = len(sq_list)
    col_range = list(range(0, GRID_W - SQ + 1))
    if len(col_range) < n:
        return None

    for _ in range(INPUT_COL_RETRIES):
        edge_indices = list(range(len(edges)))
        random.shuffle(edge_indices)
        order: list[tuple[str, int]] = [("edge", i) for i in edge_indices]

        inward_order = list(range(len(inwards)))
        random.shuffle(inward_order)
        for iw_idx in inward_order:
            latest_pos = max(
                (
                    pos
                    for dep in inward_deps[iw_idx]
                    for pos, (typ, idx) in enumerate(order)
                    if typ == "edge" and idx == dep
                ),
                default=-1,
            )
            insert_pos = random.randint(latest_pos + 1, len(order))
            order.insert(insert_pos, ("inward", iw_idx))

        ordered_squares = [edges[idx] if typ == "edge" else inwards[idx] for typ, idx in order]
        chosen_cols = sorted(random.sample(col_range, n))
        if not ascending:
            chosen_cols = chosen_cols[::-1]

        result: list[tuple[int, int, np.ndarray]] = []
        ok = True
        for i, (r, oc, p) in enumerate(ordered_squares):
            ic = chosen_cols[i]
            if ic == oc or _intersects_legend_keepout(r, ic):
                ok = False
                break
            result.append((r, ic, p))
        if not ok:
            continue
        if not _any_pairwise_overlap(result):
            return result

    return None


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([c for c in range(10) if c not in {AZURE, RED}])
    arrow_color = random.choice([c for c in range(10) if c not in {bg, AZURE, RED}])
    left_frame = random.choice([AZURE, RED])
    right_frame = RED if left_frame == AZURE else AZURE

    n_azure = random.randint(*N_AZURE_RANGE)
    n_red = random.randint(*N_RED_RANGE)
    frame_counts = {AZURE: n_azure, RED: n_red}

    left_assign = _assign_output_rows(frame_counts[left_frame])
    if left_assign is None:
        return None
    right_assign = _assign_output_rows(frame_counts[right_frame])
    if right_assign is None:
        return None
    left_edge_rows, left_inward_rows = left_assign
    right_edge_rows, right_inward_rows = right_assign

    avail_azure = [c for c in range(10) if c not in {bg, AZURE}]
    avail_red = [c for c in range(10) if c not in {bg, RED}]
    if len(avail_azure) < n_azure or len(avail_red) < n_red:
        return None
    frame_colors = {
        AZURE: random.sample(avail_azure, n_azure),
        RED: random.sample(avail_red, n_red),
    }

    output = np.full((GRID_H, GRID_W), bg, dtype=int)
    group_output = {
        "left": _place_side_squares(
            output,
            left_edge_rows,
            left_inward_rows,
            frame_colors[left_frame],
            left_frame,
            edge_col=0,
            inward_col=SQ,
        ),
        "right": _place_side_squares(
            output,
            right_edge_rows,
            right_inward_rows,
            frame_colors[right_frame],
            right_frame,
            edge_col=GRID_W - SQ,
            inward_col=GRID_W - 2 * SQ,
        ),
    }

    left_input = _derive_input_positions(group_output["left"], ascending=True)
    if left_input is None:
        return None
    right_input = _derive_input_positions(group_output["right"], ascending=False)
    if right_input is None:
        return None

    all_positions = left_input + right_input
    if _any_pairwise_overlap(all_positions):
        return None

    inp = np.full((GRID_H, GRID_W), bg, dtype=int)
    legend = _build_legend(bg, left_frame, right_frame, arrow_color)
    paste(inp, legend, 0, 0)
    for r, c, patch in all_positions:
        if _intersects_legend_keepout(r, c):
            return None
        paste(inp, patch, r, c)

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
