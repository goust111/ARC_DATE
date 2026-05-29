# Variant of b5ca7ac4
# Dimension: D8 — 变换——框架在移动过程中的变化 (transform)
# Change: 框架在从输入位置移动到输出位置时保持完全不变：边框颜色、内部颜色、尺寸均不改变... → 堆叠时两组互换边框颜色

from __future__ import annotations

import random
import numpy as np
from arc_utils import crop_bbox, get_bg, paste


# Design analysis for the D8 variant
# a) Concrete behavior:
#    - Input still contains scattered 5x5 framed squares with teal(8) or red(2) borders and
#      solid 3x3 interiors.
#    - Motion and stacking are unchanged: teal-group squares go to the left side, red-group
#      squares go to the right side, and row overlap decides edge vs inward column.
#    - The only transformed attribute is the border color after movement:
#      teal input frames become red in the output, red input frames become teal in the output.
#    - Example trace: if a teal frame at (r=4,c=9) is processed first among teal squares, the
#      original task pastes that same teal-framed patch at (4,0). This variant instead pastes a
#      copy whose border pixels are recolored to red while keeping the 3x3 interior unchanged.
# b) Complexity calibration:
#    - The swap is global and binary (8 <-> 2), so it is easy to spot across 2-4 examples.
#    - We keep the original row/ordering/stacking structure, preserving enough repeated evidence
#      for the solver to infer that movement and recoloring are coupled.
# c) Hidden-assumption audit:
#    - The 5x5 scan in main() assumes input objects are still identified by frame colors {2,8};
#      this remains true because only output frames swap.
#    - The ordering code depends only on original columns and row overlap, not on output colors.
#    - The old paste(out, s["patch"], ...) implicitly assumed the patch itself stayed valid for
#      output. That is no longer true, so main() must paste a transformed copy instead.
#    - A generator that allowed interior color 2 inside teal frames or 8 inside red frames could
#      make the swapped output lose its visible border. We therefore exclude both 2 and 8 from
#      all interior colors so the color swap stays visually observable.
# d) Placement legality:
#    - Squares remain 5x5, stay in-bounds, and keep the original output-first row assignment.
#    - Input placement still uses distinct columns and pairwise non-overlap checks.
#    - Output placement still has only two depths per side: edge and one-step inward.
# e) Visual observability:
#    - Humans can compare input border color vs output border color directly because the frame
#      shapes are preserved and only their border color flips.
#    - No sorting uses hidden ids or non-visual metadata; all control signals are spatial.
# f) Visual cleanliness:
#    - The output layout remains the original tidy edge-stacking arrangement.
#    - Preventing interiors from using {2,8} avoids ambiguous solid-color 5x5 blocks.
# g) Color-change inferability:
#    - Only two frame colors participate, and each consistently maps to the other while keeping
#      group destination side unchanged. That gives a small, repeated, structurally anchored map.


GRID_H, GRID_W = 22, 22
SQ = 5
AZURE, RED = 8, 2
SWAP_FRAME = {AZURE: RED, RED: AZURE}
N_AZURE_RANGE = (3, 4)
N_RED_RANGE = (3, 5)
MAX_INWARD = 2
INPUT_COL_RETRIES = 50
GEN_RETRIES = 500


def _swap_patch_frame(patch: np.ndarray, new_frame_color: int) -> np.ndarray:
    """Return a copy whose border pixels are recolored and interior is preserved."""
    out = np.array(patch, copy=True)
    out[0, :] = new_frame_color
    out[-1, :] = new_frame_color
    out[:, 0] = new_frame_color
    out[:, -1] = new_frame_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move framed 5x5 squares to side stacks and swap their border colors on arrival.

    Input structure:
    - bg is the most common color.
    - Teal (8) and red (2) 5x5 framed squares with solid interiors are scattered in the grid.

    Rule:
    - Teal-frame squares move to the left side; red-frame squares move to the right side.
    - Each square keeps its original row, and processing order is still near-edge-first:
      teal by increasing input column, red by decreasing input column.
    - A row-overlap with an already placed edge square pushes the square one frame-width inward.
    - During that placement, teal borders become red and red borders become teal; interiors stay fixed.

    Key insight: First recover the original frame groups from the input, then apply the same
    stacking logic as before, but paste a border-recolored copy of each square into the output.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    out = np.full_like(g, bg)

    def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
        return not (int(a1) < int(b0) or int(b1) < int(a0))

    squares = []
    for r0 in range(H - SQ + 1):
        for c0 in range(W - SQ + 1):
            patch = crop_bbox(g, (int(r0), int(c0), int(r0 + SQ - 1), int(c0 + SQ - 1)))
            border = np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])
            if int(border.min()) != int(border.max()):
                continue
            frame_col = int(border[0])
            if frame_col not in {RED, AZURE}:
                continue

            inner = patch[1:-1, 1:-1]
            if not bool((inner != frame_col).any()):
                continue

            squares.append({"frame": frame_col, "r0": int(r0), "c0": int(c0), "patch": patch})

    azure_squares = sorted((s for s in squares if int(s["frame"]) == AZURE), key=lambda s: int(s["c0"]))
    placed_at_default: list[tuple[int, int]] = []
    for s in azure_squares:
        r0 = int(s["r0"])
        r1 = int(r0 + SQ - 1)
        has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_default)
        dest_c = SQ if has_overlap else 0
        paste(out, _swap_patch_frame(s["patch"], SWAP_FRAME[AZURE]), r0, dest_c, transparent=bg)
        if dest_c == 0:
            placed_at_default.append((r0, r1))

    red_squares = sorted((s for s in squares if int(s["frame"]) == RED), key=lambda s: -int(s["c0"]))
    placed_at_default = []
    for s in red_squares:
        r0 = int(s["r0"])
        r1 = int(r0 + SQ - 1)
        has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_default)
        dest_c = int(W - 2 * SQ) if has_overlap else int(W - SQ)
        paste(out, _swap_patch_frame(s["patch"], SWAP_FRAME[RED]), r0, dest_c, transparent=bg)
        if dest_c == int(W - SQ):
            placed_at_default.append((r0, r1))

    return out


def _make_patch(frame_color: int, inner_color: int) -> np.ndarray:
    p = np.full((SQ, SQ), frame_color, dtype=int)
    p[1:-1, 1:-1] = inner_color
    return p


def _rows_overlap(r_a: int, r_b: int) -> bool:
    return abs(r_a - r_b) < SQ


def _rects_overlap(r1: int, c1: int, r2: int, c2: int) -> bool:
    return abs(r1 - r2) < SQ and abs(c1 - c2) < SQ


def _any_pairwise_overlap(positions: list[tuple[int, int, np.ndarray]]) -> bool:
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if _rects_overlap(positions[i][0], positions[i][1], positions[j][0], positions[j][1]):
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
    inner_colors: list[int],
    input_frame_color: int,
    edge_col: int,
    inward_col: int,
) -> list[tuple[int, int, bool, np.ndarray]]:
    """Render swapped-frame output squares and return input-side metadata."""
    placed: list[tuple[int, int, bool, np.ndarray]] = []
    output_frame_color = SWAP_FRAME[input_frame_color]
    for i, r in enumerate(edge_rows):
        input_patch = _make_patch(input_frame_color, inner_colors[i])
        paste(output, _swap_patch_frame(input_patch, output_frame_color), r, edge_col)
        placed.append((r, edge_col, True, input_patch))
    for i, r in enumerate(inward_rows):
        input_patch = _make_patch(input_frame_color, inner_colors[len(edge_rows) + i])
        paste(output, _swap_patch_frame(input_patch, output_frame_color), r, inward_col)
        placed.append((r, inward_col, False, input_patch))
    return placed


def _derive_input_positions(
    sq_list: list[tuple[int, int, bool, np.ndarray]],
    ascending: bool,
) -> list[tuple[int, int, np.ndarray]] | None:
    edges = [(r, oc, p) for (r, oc, is_edge, p) in sq_list if is_edge]
    inwards = [(r, oc, p) for (r, oc, is_edge, p) in sq_list if not is_edge]
    inward_deps = [[i for i, (er, _, _) in enumerate(edges) if _rows_overlap(ir, er)] for (ir, _, _) in inwards]

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
                    for d in inward_deps[iw_idx]
                    for pos, (typ, idx) in enumerate(order)
                    if typ == "edge" and idx == d
                ),
                default=-1,
            )
            insert_pos = random.randint(latest_pos + 1, len(order))
            order.insert(insert_pos, ("inward", iw_idx))

        ordered_squares = [edges[idx] if typ == "edge" else inwards[idx] for typ, idx in order]

        chosen_cols = sorted(random.sample(col_range, n))
        if not ascending:
            chosen_cols = chosen_cols[::-1]

        result = []
        ok = True
        for i, (r, oc, p) in enumerate(ordered_squares):
            ic = chosen_cols[i]
            if ic == oc:
                ok = False
                break
            result.append((r, ic, p))
        if ok and not _any_pairwise_overlap(result):
            return result

    return None


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([c for c in range(10) if c not in {AZURE, RED}])
    n_azure = random.randint(*N_AZURE_RANGE)
    n_red = random.randint(*N_RED_RANGE)

    azure_assign = _assign_output_rows(n_azure)
    if azure_assign is None:
        return None
    red_assign = _assign_output_rows(n_red)
    if red_assign is None:
        return None
    azure_edge_rows, azure_inward_rows = azure_assign
    red_edge_rows, red_inward_rows = red_assign

    avail_inner = [c for c in range(10) if c not in {bg, RED, AZURE}]
    if len(avail_inner) < max(n_azure, n_red):
        return None
    azure_colors = random.sample(avail_inner, n_azure)
    red_colors = random.sample(avail_inner, n_red)

    output = np.full((GRID_H, GRID_W), bg, dtype=int)
    azure_out = _place_side_squares(
        output,
        azure_edge_rows,
        azure_inward_rows,
        azure_colors,
        AZURE,
        edge_col=0,
        inward_col=SQ,
    )
    red_out = _place_side_squares(
        output,
        red_edge_rows,
        red_inward_rows,
        red_colors,
        RED,
        edge_col=GRID_W - SQ,
        inward_col=GRID_W - 2 * SQ,
    )

    azure_input = _derive_input_positions(azure_out, ascending=True)
    if azure_input is None:
        return None
    red_input = _derive_input_positions(red_out, ascending=False)
    if red_input is None:
        return None

    all_positions = azure_input + red_input
    if _any_pairwise_overlap(all_positions):
        return None

    inp = np.full((GRID_H, GRID_W), bg, dtype=int)
    for r, c, p in all_positions:
        paste(inp, p, r, c)

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
