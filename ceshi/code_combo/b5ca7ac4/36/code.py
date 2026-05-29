# Combo variant of b5ca7ac4 (index 36)
# combo_spec: D8/2+D14/2
# Name: 框架复制+颜色互换

from __future__ import annotations

import random
from itertools import combinations

import numpy as np

from arc_utils import crop_bbox, get_bg, paste


# Design analysis
# ---------------
# a) Concrete behavior:
#    - Keep the original object type: isolated 5x5 framed squares with border color 8 or 2
#      and a solid 3x3 interior.
#    - Preserve the original edge-stacking logic as one branch of the fused rule:
#      azure-input squares stack on the left, red-input squares stack on the right,
#      with edge-vs-inward determined by row overlap and near-edge-first ordering.
#    - Apply D14 by duplicating every detected square into a second output position.
#      We choose the "symmetric position" interpretation from the prompt: the second copy
#      is placed at row H-5-r while keeping the original input column.
#    - Apply D8's color-swap semantics to that symmetric copy only. The provided D8/2
#      reference is explicit that the visible transform is a border swap 8<->2 while the
#      3x3 interior stays unchanged, so the fused task uses that concrete recoloring rule.
#    - Example trace:
#      input azure frame at (r=2, c=13) with interior color 4
#      original task -> one azure copy at (2, 0 or 5)
#      fused task    -> one original-color azure copy at (2, 0 or 5)
#                    -> one mirrored red-bordered copy at (22-5-2=15, 13)
#      The solver must infer both the duplication and which of the two destinations swaps color.
# b) Complexity calibration:
#    - The duplication is visually strong because each source square always yields exactly
#      two output copies in two different structural zones: a top stacked zone and a lower
#      mirrored zone.
#    - The color swap stays binary and repeated many times: every mirrored azure copy becomes
#      red-bordered, every mirrored red copy becomes azure-bordered.
#    - To avoid clutter, the input is restricted to the upper band so mirrored copies always
#      land in a separate lower band. This keeps the two effects readable instead of mixing
#      them into one crowded region.
# c) Hidden-assumption audit:
#    - Square detection by scanning 5x5 windows still works because the input itself remains
#      unchanged: only scattered source squares appear there once, with frame colors still in
#      {2, 8}.
#    - The original solver assumed one output paste per square. That assumption breaks under
#      D14, so main() must emit an additional mirrored copy for every detected square.
#    - The D8 single-variant implementation changed the patch that gets pasted into the output.
#      In this combo, blindly pasting the original patch at both destinations would erase the
#      color-swap dimension, so the mirrored branch must use an explicitly recolored copy.
#    - If interiors could use colors 2 or 8, a swapped border could visually blend with the
#      interior and weaken the signal. The generator therefore excludes both frame colors from
#      all interior choices.
# d) Placement legality:
#    - Source rows are restricted to 0..6. Their mirrored rows are then 11..17, so mirrored
#      copies never overlap the stacked copies vertically.
#    - Input columns are partitioned by group: red sources use 0..6, azure sources use 11..17.
#      That separation prevents cross-group collisions in both the input and mirrored zones.
#    - Each side still uses only the original two stack depths (edge and one step inward), and
#      row assignment guarantees that every inward square overlaps at least one edge square.
# e) Visual observability:
#    - A human can compare each input square against two output manifestations:
#      one moved copy with the same colors and one mirrored copy with the swapped border color.
#    - The ordering signal is purely spatial: original columns control processing order, and
#      row overlap controls inward displacement. No invisible metadata is used.
#    - The symmetric copies reuse the original columns, giving a stable geometric cue for the
#      duplication rule across 2-4 examples.
# f) Visual cleanliness:
#    - The upper output band contains the tidy left/right stacks; the lower band contains the
#      mirrored copies. Those zones do not overlap.
#    - Cross-group input columns are split apart, so the lower mirrored copies also form two
#      clean clusters instead of a tangled mixture.
# g) Change inferability:
#    - The only color mapping is 8<->2 on mirrored borders, repeated for every object.
#      Because the stacked copy keeps the source colors, each example directly shows the before
#      and after versions of the same square family, making the recoloring rule learnable.
#    - The generator rejects degenerate layouts by construction: every sample contains both
#      stacked copies and mirrored swapped copies, so the combo cannot collapse back into the
#      original single-output behavior.


GRID_H, GRID_W = 22, 22
SQ = 5
AZURE, RED = 8, 2
SWAP_FRAME = {AZURE: RED, RED: AZURE}
N_AZURE_RANGE = (3, 4)
N_RED_RANGE = (3, 4)
SOURCE_ROW_MAX = GRID_H // 2 - SQ  # 6, so mirrored rows are 11..17
AZURE_INPUT_COLS = list(range(11, GRID_W - SQ + 1))  # 11..17
RED_INPUT_COLS = list(range(0, 7))  # 0..6
INPUT_COL_RETRIES = 120
ROW_ASSIGN_RETRIES = 120
GEN_RETRIES = 1200


def _mirror_row(r0: int, height: int = GRID_H) -> int:
    return int(height - SQ - int(r0))


def _swap_patch_frame(patch: np.ndarray, new_frame_color: int) -> np.ndarray:
    """Return a copy whose border is recolored and whose 3x3 interior is preserved."""
    out = np.array(patch, copy=True)
    out[0, :] = new_frame_color
    out[-1, :] = new_frame_color
    out[:, 0] = new_frame_color
    out[:, -1] = new_frame_color
    return out


def _rows_overlap(r_a: int, r_b: int) -> bool:
    return abs(int(r_a) - int(r_b)) < SQ


def _rects_overlap(r1: int, c1: int, r2: int, c2: int) -> bool:
    return abs(int(r1) - int(r2)) < SQ and abs(int(c1) - int(c2)) < SQ


def _any_pairwise_overlap(positions: list[tuple[int, int, np.ndarray]]) -> bool:
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if _rects_overlap(positions[i][0], positions[i][1], positions[j][0], positions[j][1]):
                return True
    return False


def _make_patch(frame_color: int, inner_color: int) -> np.ndarray:
    patch = np.full((SQ, SQ), frame_color, dtype=int)
    patch[1:-1, 1:-1] = inner_color
    return patch


def _detect_squares(g: np.ndarray) -> list[dict[str, int | np.ndarray]]:
    squares: list[dict[str, int | np.ndarray]] = []
    for r0 in range(g.shape[0] - SQ + 1):
        for c0 in range(g.shape[1] - SQ + 1):
            patch = crop_bbox(g, (int(r0), int(c0), int(r0 + SQ - 1), int(c0 + SQ - 1)))
            border = np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])
            if int(border.min()) != int(border.max()):
                continue
            frame_col = int(border[0])
            if frame_col not in {AZURE, RED}:
                continue
            inner = patch[1:-1, 1:-1]
            if not bool((inner != frame_col).any()):
                continue
            squares.append({"frame": frame_col, "r0": int(r0), "c0": int(c0), "patch": patch})
    return squares


def main(grid: np.ndarray) -> np.ndarray:
    """
    Duplicate each framed 5x5 square into a preserved-color side stack and a mirrored border-swapped copy.

    Input structure:
    - bg is the most common color.
    - The foreground consists of scattered red (2) and azure (8) 5x5 frames with solid 3x3 interiors.

    Rule:
    - Azure squares are still processed by increasing input column and stacked on the left; red squares are processed by decreasing input column and stacked on the right.
    - In that stack, each square keeps its source row and original colors; row-overlap with an already used edge row pushes the square one frame-width inward.
    - Every detected square also produces a second copy at the vertically mirrored row H-5-r, keeping its original input column.
    - This mirrored copy swaps only the border color: azure borders become red and red borders become azure, while the 3x3 interior remains unchanged.

    Key insight:
    - Recover the original frame groups from the input exactly as in the base task, then emit two synchronized outputs per source square: the usual stack copy and a symmetric lower copy with the fixed 8<->2 border swap.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = np.full_like(g, bg)

    squares = _detect_squares(g)

    for s in squares:
        frame = int(s["frame"])
        mirrored_patch = _swap_patch_frame(np.array(s["patch"], copy=True), SWAP_FRAME[frame])
        paste(out, mirrored_patch, _mirror_row(int(s["r0"]), H), int(s["c0"]), transparent=bg)

    def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
        return not (int(a1) < int(b0) or int(b1) < int(a0))

    azure_squares = sorted((s for s in squares if int(s["frame"]) == AZURE), key=lambda s: int(s["c0"]))
    placed_at_edge: list[tuple[int, int]] = []
    for s in azure_squares:
        r0 = int(s["r0"])
        r1 = int(r0 + SQ - 1)
        dest_c = SQ if any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_edge) else 0
        paste(out, s["patch"], r0, dest_c, transparent=bg)
        if dest_c == 0:
            placed_at_edge.append((r0, r1))

    red_squares = sorted((s for s in squares if int(s["frame"]) == RED), key=lambda s: -int(s["c0"]))
    placed_at_edge = []
    for s in red_squares:
        r0 = int(s["r0"])
        r1 = int(r0 + SQ - 1)
        dest_c = int(W - 2 * SQ) if any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_edge) else int(W - SQ)
        paste(out, s["patch"], r0, dest_c, transparent=bg)
        if dest_c == int(W - SQ):
            placed_at_edge.append((r0, r1))

    return out


def _assign_output_rows(n: int) -> tuple[list[int], list[int]] | None:
    n_edge = 2
    n_inward = n - n_edge
    source_rows = list(range(SOURCE_ROW_MAX + 1))
    edge_pairs = [(a, b) for a, b in combinations(source_rows, 2) if not _rows_overlap(a, b)]
    if not edge_pairs:
        return None

    for _ in range(ROW_ASSIGN_RETRIES):
        edge_rows = list(random.choice(edge_pairs))
        random.shuffle(edge_rows)

        inward_rows: list[int] = []
        pool = source_rows[:]
        random.shuffle(pool)
        for r in pool:
            if not any(_rows_overlap(r, er) for er in edge_rows):
                continue
            if any(_rows_overlap(r, ir) for ir in inward_rows):
                continue
            inward_rows.append(r)
            if len(inward_rows) == n_inward:
                break
        if len(inward_rows) == n_inward:
            return edge_rows, inward_rows
    return None


def _place_side_squares(
    edge_rows: list[int],
    inward_rows: list[int],
    inner_colors: list[int],
    frame_color: int,
    edge_col: int,
    inward_col: int,
) -> list[tuple[int, int, bool, np.ndarray]]:
    placed: list[tuple[int, int, bool, np.ndarray]] = []
    for i, r in enumerate(edge_rows):
        patch = _make_patch(frame_color, inner_colors[i])
        placed.append((r, edge_col, True, patch))
    for i, r in enumerate(inward_rows):
        patch = _make_patch(frame_color, inner_colors[len(edge_rows) + i])
        placed.append((r, inward_col, False, patch))
    return placed


def _derive_input_positions(
    sq_list: list[tuple[int, int, bool, np.ndarray]],
    ascending: bool,
    allowed_cols: list[int],
) -> list[tuple[int, int, np.ndarray]] | None:
    edges = [(r, oc, p) for (r, oc, is_edge, p) in sq_list if is_edge]
    inwards = [(r, oc, p) for (r, oc, is_edge, p) in sq_list if not is_edge]
    inward_deps = [[i for i, (er, _, _) in enumerate(edges) if _rows_overlap(ir, er)] for (ir, _, _) in inwards]

    n = len(sq_list)
    if len(allowed_cols) < n:
        return None

    for _ in range(INPUT_COL_RETRIES):
        edge_indices = list(range(len(edges)))
        random.shuffle(edge_indices)
        order: list[tuple[str, int]] = [("edge", i) for i in edge_indices]

        inward_indices = list(range(len(inwards)))
        random.shuffle(inward_indices)
        for iw_idx in inward_indices:
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

        ordered = [edges[idx] if typ == "edge" else inwards[idx] for typ, idx in order]
        chosen_cols = sorted(random.sample(allowed_cols, n))
        if not ascending:
            chosen_cols = chosen_cols[::-1]

        result: list[tuple[int, int, np.ndarray]] = []
        ok = True
        for i, (r, oc, patch) in enumerate(ordered):
            ic = int(chosen_cols[i])
            if ic == oc:
                ok = False
                break
            result.append((int(r), ic, patch))
        if ok and not _any_pairwise_overlap(result):
            return result

    return None


def _build_expected_output(
    bg: int,
    azure_out: list[tuple[int, int, bool, np.ndarray]],
    red_out: list[tuple[int, int, bool, np.ndarray]],
    azure_input: list[tuple[int, int, np.ndarray]],
    red_input: list[tuple[int, int, np.ndarray]],
) -> np.ndarray:
    output = np.full((GRID_H, GRID_W), bg, dtype=int)
    for r, c, patch in azure_input + red_input:
        frame = int(patch[0, 0])
        paste(output, _swap_patch_frame(patch, SWAP_FRAME[frame]), _mirror_row(r), c)
    for r, c, _is_edge, patch in azure_out:
        paste(output, patch, r, c)
    for r, c, _is_edge, patch in red_out:
        paste(output, patch, r, c)
    return output


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([c for c in range(10) if c not in {AZURE, RED}])
    n_azure = random.randint(*N_AZURE_RANGE)
    n_red = random.randint(*N_RED_RANGE)

    azure_assign = _assign_output_rows(n_azure)
    red_assign = _assign_output_rows(n_red)
    if azure_assign is None or red_assign is None:
        return None
    azure_edge_rows, azure_inward_rows = azure_assign
    red_edge_rows, red_inward_rows = red_assign

    avail_inner = [c for c in range(10) if c not in {bg, AZURE, RED}]
    if len(avail_inner) < max(n_azure, n_red):
        return None
    azure_colors = random.sample(avail_inner, n_azure)
    red_colors = random.sample(avail_inner, n_red)

    azure_out = _place_side_squares(azure_edge_rows, azure_inward_rows, azure_colors, AZURE, 0, SQ)
    red_out = _place_side_squares(red_edge_rows, red_inward_rows, red_colors, RED, GRID_W - SQ, GRID_W - 2 * SQ)

    azure_input = _derive_input_positions(azure_out, ascending=True, allowed_cols=AZURE_INPUT_COLS)
    red_input = _derive_input_positions(red_out, ascending=False, allowed_cols=RED_INPUT_COLS)
    if azure_input is None or red_input is None:
        return None

    all_input_positions = azure_input + red_input
    if _any_pairwise_overlap(all_input_positions):
        return None

    inp = np.full((GRID_H, GRID_W), bg, dtype=int)
    for r, c, patch in all_input_positions:
        paste(inp, patch, r, c)

    expected_output = _build_expected_output(bg, azure_out, red_out, azure_input, red_input)
    try:
        if np.array_equal(main(inp), expected_output):
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
