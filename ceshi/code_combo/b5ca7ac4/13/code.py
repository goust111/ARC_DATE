# Combo variant of b5ca7ac4 (index 13)
# combo_spec: D2/0+D9/0
# Name: 交替处理+统一从上到下

from __future__ import annotations

import random

import numpy as np

from arc_utils import crop_bbox, get_bg, paste

# Design analysis for the fused variant
# a) Concrete behavior:
#    Use rows as the only ordering cue. Suppose the topmost three frames are:
#    teal at row 0, red at row 1, teal at row 2. In the original task the red
#    side was independent, so that red frame would still claim the right edge.
#    In this combo we process a single alternating sequence and share the
#    already-claimed edge rows globally, so the red frame at row 1 is forced
#    inward because it vertically overlaps the earlier teal edge frame.
# b) Complexity calibration:
#    If we kept per-color overlap state, D9/0 would be invisible: interleaving
#    colors would not change any decision. To make the combo observable but still
#    readable, the generator organizes the sequence into short row bands where one
#    edge frame can trigger at most one inward teal and one inward red.
# c) Hidden assumptions review:
#    - `main()` now assumes the decisive history is the global set of frames that
#      already occupied the outer edge, not one set per color.
#    - Alternation is driven by the topmost remaining frame color, then both
#      colors are consumed top-to-bottom. The generator gives every frame a
#      distinct start row so this merge rule is unambiguous.
#    - Once one color runs out, the remaining frames of the other color continue
#      top-to-bottom. The generator allows such tails.
# d) Placement legality:
#    Output rows are built from 1-3 frame blocks. Block starts are at least five
#    rows apart, so edge frames never overlap. Within one block, only the block
#    head is edge; later rows overlap that head and therefore go inward.
# e) Visual observability:
#    Each sample contains at least one cross-color trigger: an inward frame whose
#    only overlapping earlier edge frame has the opposite border color. This makes
#    the fused rule visibly different from either the original task or a plain
#    "both colors top-to-bottom but still independent" variant.
# f) Visual cleanliness:
#    On each side there are only two columns (edge and inward). The generator
#    never puts two same-side inward frames in the same overlapping row band, so
#    stacked results stay legible.
# g) Inferability:
#    No hidden numeric ordering such as color-id ranking is used. A human can
#    infer the rule from row order, color alternation, and the fact that opposite
#    colors can now block each other from taking the outer edge.


GRID_H, GRID_W = 22, 22
SQ = 5
AZURE, RED = 8, 2
N_AZURE_RANGE = (3, 4)
N_RED_RANGE = (3, 5)
GEN_RETRIES = 600
LAYOUT_RETRIES = 120
INPUT_RETRIES = 120


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
    patch = np.full((SQ, SQ), int(frame_color), dtype=int)
    patch[1:-1, 1:-1] = int(inner_color)
    return patch


def _detect_squares(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    squares: list[dict[str, object]] = []
    seen: set[tuple[int, int]] = set()
    for r0 in range(H - SQ + 1):
        for c0 in range(W - SQ + 1):
            patch = crop_bbox(g, (int(r0), int(c0), int(r0 + SQ - 1), int(c0 + SQ - 1)))
            border = np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])
            if int(border.min()) != int(border.max()):
                continue
            frame = int(border[0])
            if frame not in {AZURE, RED}:
                continue
            inner = patch[1:-1, 1:-1]
            if not bool((inner != frame).any()):
                continue
            key = (int(r0), int(c0))
            if key in seen:
                continue
            seen.add(key)
            squares.append({"frame": frame, "r0": int(r0), "c0": int(c0), "patch": patch})
    return squares


def _build_alternating_order(squares: list[dict[str, object]]) -> list[dict[str, object]]:
    azure = sorted(
        [s for s in squares if int(s["frame"]) == AZURE],
        key=lambda s: (int(s["r0"]), int(s["c0"])),
    )
    red = sorted(
        [s for s in squares if int(s["frame"]) == RED],
        key=lambda s: (int(s["r0"]), int(s["c0"])),
    )
    if not azure:
        return red
    if not red:
        return azure

    first_azure = (int(azure[0]["r0"]), int(azure[0]["c0"]))
    first_red = (int(red[0]["r0"]), int(red[0]["c0"]))
    next_color = AZURE if first_azure <= first_red else RED

    ai = 0
    ri = 0
    order: list[dict[str, object]] = []
    while ai < len(azure) or ri < len(red):
        if next_color == AZURE and ai < len(azure):
            order.append(azure[ai])
            ai += 1
            next_color = RED
            continue
        if next_color == RED and ri < len(red):
            order.append(red[ri])
            ri += 1
            next_color = AZURE
            continue
        if ai < len(azure):
            order.append(azure[ai])
            ai += 1
            next_color = RED
        else:
            order.append(red[ri])
            ri += 1
            next_color = AZURE
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move bordered 5x5 frames toward their side, using one shared top-to-bottom schedule.

    Input structure:
    - The background is the most common color.
    - The active objects are 5x5 hollow frames with a solid 3x3 interior.
    - Teal frames (8) and red frames (2) are scattered in the grid and keep their original rows.

    Rule:
    - Teal frames move to the left side; red frames move to the right side.
    - Order both colors from top to bottom, then consume them as one alternating sequence:
      topmost remaining color starts, then colors alternate while available.
    - A frame takes the outer edge column unless its 5-row span overlaps any earlier frame that
      already claimed an outer edge position in that global sequence; in that case it shifts one
      frame-width inward on its own side.

    Key insight:
    - The important state is no longer "what teal already did" and "what red already did"
      separately. A single shared history of edge-claiming rows means one color can force the
      other color inward, so the final stack depends on the merged alternating row order.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = np.full_like(g, bg)

    squares = _detect_squares(g)
    order = _build_alternating_order(squares)

    claimed_edge_rows: list[tuple[int, int]] = []
    for sq in order:
        frame = int(sq["frame"])
        r0 = int(sq["r0"])
        r1 = int(r0 + SQ - 1)
        has_overlap = any(not (r1 < er0 or er1 < r0) for er0, er1 in claimed_edge_rows)
        if frame == AZURE:
            edge_col, inward_col = 0, SQ
        else:
            edge_col, inward_col = int(W - SQ), int(W - 2 * SQ)
        dest_c = int(inward_col if has_overlap else edge_col)
        paste(out, sq["patch"], r0, dest_c, transparent=bg)
        if not has_overlap:
            claimed_edge_rows.append((r0, r1))
    return out


def _build_color_sequence(n_azure: int, n_red: int, start_color: int) -> list[int]:
    counts = {AZURE: int(n_azure), RED: int(n_red)}
    seq: list[int] = []
    next_color = int(start_color)
    while counts[AZURE] or counts[RED]:
        if counts[next_color] > 0:
            seq.append(next_color)
            counts[next_color] -= 1
            next_color = RED if next_color == AZURE else AZURE
            continue
        other = RED if next_color == AZURE else AZURE
        seq.extend([other] * counts[other])
        break
    return seq


def _find_block_lengths(seq: list[int]) -> list[int] | None:
    n = len(seq)
    solutions: list[list[int]] = []

    def dfs(i: int, lengths: list[int], starts: list[int], has_cross_trigger: bool) -> None:
        if i == n:
            if starts and AZURE in starts and RED in starts and has_cross_trigger and len(lengths) <= 4:
                solutions.append(lengths.copy())
            return
        if len(lengths) >= 4:
            return
        remaining = n - i
        max_capacity = (4 - len(lengths)) * 3
        if remaining > max_capacity:
            return

        options = [1, 2, 3]
        random.shuffle(options)
        for block_len in options:
            if i + block_len > n:
                continue
            inwards = seq[i + 1 : i + block_len]
            if len(set(inwards)) != len(inwards):
                continue
            lengths.append(block_len)
            starts.append(seq[i])
            cross_here = block_len >= 2 and seq[i + 1] != seq[i]
            dfs(i + block_len, lengths, starts, has_cross_trigger or cross_here)
            starts.pop()
            lengths.pop()

    dfs(0, [], [], False)
    if not solutions:
        return None
    return random.choice(solutions)


def _choose_anchor_rows(n_blocks: int) -> list[int]:
    anchors: list[int] = []
    min_start = 0
    for remaining in range(n_blocks, 0, -1):
        max_start = 15 - 5 * (remaining - 1)
        if min_start > max_start:
            raise ValueError("No legal anchor rows remain")
        anchor = random.randint(min_start, max_start)
        anchors.append(anchor)
        min_start = anchor + 5
    return anchors


def _build_layout(bg: int, n_azure: int, n_red: int) -> tuple[np.ndarray, list[dict[str, object]]] | None:
    start_options = [AZURE, RED]
    random.shuffle(start_options)
    for start_color in start_options:
        seq = _build_color_sequence(n_azure, n_red, start_color)
        block_lengths = _find_block_lengths(seq)
        if block_lengths is None:
            continue

        block_starts: list[int] = []
        idx = 0
        for block_len in block_lengths:
            block_starts.append(idx)
            idx += block_len

        anchors = _choose_anchor_rows(len(block_lengths))

        avail_azure = [c for c in range(10) if c not in {bg, AZURE}]
        avail_red = [c for c in range(10) if c not in {bg, RED}]
        if len(avail_azure) < n_azure or len(avail_red) < n_red:
            return None
        azure_colors = random.sample(avail_azure, n_azure)
        red_colors = random.sample(avail_red, n_red)
        azure_idx = 0
        red_idx = 0

        output = np.full((GRID_H, GRID_W), int(bg), dtype=int)
        records: list[dict[str, object]] = []
        for block_id, start_idx in enumerate(block_starts):
            block_len = block_lengths[block_id]
            anchor = anchors[block_id]
            for offset in range(block_len):
                color = int(seq[start_idx + offset])
                row = int(anchor + offset)
                is_edge = offset == 0
                if color == AZURE:
                    inner = azure_colors[azure_idx]
                    azure_idx += 1
                    edge_col, inward_col = 0, SQ
                else:
                    inner = red_colors[red_idx]
                    red_idx += 1
                    edge_col, inward_col = GRID_W - SQ, GRID_W - 2 * SQ
                out_col = int(edge_col if is_edge else inward_col)
                patch = _make_patch(color, inner)
                paste(output, patch, row, out_col)
                records.append(
                    {
                        "frame": color,
                        "row": row,
                        "out_col": out_col,
                        "patch": patch,
                    }
                )
        return output, records
    return None


def _assign_input_columns(records: list[dict[str, object]]) -> list[tuple[int, int, np.ndarray]] | None:
    n = len(records)
    base_candidates = list(range(GRID_W - SQ + 1))
    conflict_degree = []
    for i, rec in enumerate(records):
        row_i = int(rec["row"])
        degree = sum(1 for j, other in enumerate(records) if i != j and _rows_overlap(row_i, int(other["row"])))
        conflict_degree.append((degree, i))
    order = [i for _, i in sorted(conflict_degree, reverse=True)]

    for _ in range(INPUT_RETRIES):
        assigned: dict[int, int] = {}

        def backtrack(pos: int) -> bool:
            if pos == n:
                return True
            idx = order[pos]
            rec = records[idx]
            row = int(rec["row"])
            out_col = int(rec["out_col"])
            candidates = [c for c in base_candidates if c != out_col]
            random.shuffle(candidates)
            for col in candidates:
                if any(
                    _rects_overlap(row, col, int(records[j]["row"]), assigned[j])
                    for j in assigned
                ):
                    continue
                assigned[idx] = int(col)
                if backtrack(pos + 1):
                    return True
                del assigned[idx]
            return False

        if backtrack(0):
            placed = [
                (int(records[i]["row"]), int(assigned[i]), records[i]["patch"])
                for i in range(n)
            ]
            if not _any_pairwise_overlap(placed):
                return placed
    return None


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([c for c in range(10) if c not in {AZURE, RED}])
    n_azure = random.randint(*N_AZURE_RANGE)
    n_red = random.randint(*N_RED_RANGE)

    for _ in range(LAYOUT_RETRIES):
        built = _build_layout(bg, n_azure, n_red)
        if built is None:
            continue
        output, records = built
        input_positions = _assign_input_columns(records)
        if input_positions is None:
            continue

        inp = np.full((GRID_H, GRID_W), int(bg), dtype=int)
        for row, col, patch in input_positions:
            paste(inp, patch, row, col)

        try:
            if np.array_equal(main(inp), output):
                return inp
        except Exception:
            continue
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
