# Variant of b5ca7ac4
# Dimension: D15 — 干扰项——输入中的非目标对象 (src)
# Change: 输入中除框架和背景外无其他对象，所有非背景像素都属于某个框架... → 增加不完整的框架作为干扰

from __future__ import annotations

import random

import numpy as np

from arc_utils import crop_bbox, get_bg, paste

# Design analysis
# ===============
# variant_spec:
#   grid: 22x22, same as the source task
#   targets:
#     - complete 5x5 bordered squares only
#     - border color fixed to teal/8 or red/2
#     - complete means all 16 border cells share one frame color
#     - interior still contains at least one non-frame pixel
#   distractors:
#     - same 5x5 footprint and same frame-color family {8, 2}
#     - visually close to targets: uniform 3x3 interior plus a frame missing one full side
#     - placed only in the input, never copied to the output
#   output:
#     - unchanged relative to the original task: stack valid teal frames on the left,
#       valid red frames on the right, preserving rows and overlap-driven inward shift
#
# behavior_trace:
#   - Original path: every non-bg object is a valid 5x5 frame, so scanning 5x5 windows
#     and collecting uniform borders is enough to recover every object that must move.
#   - Variant path: the input can now contain a red or teal "U" shape made from a 5x5
#     frame with its top/left/right/bottom side removed. A naive "any 5x5 object of the
#     right colors moves" interpretation would be wrong; only the windows whose full
#     border is intact should be collected. The broken frame disappears in the output.
#   - Difference: the pipeline changes only at object detection. Sorting by column and
#     overlap-based stacking still runs on the retained complete frames.
#
# complexity_guardrails:
#   - Distractors reuse the same size and colors as the targets, so the change is visible.
#   - The defect vocabulary is small (missing exactly one whole side), keeping samples
#     legible and avoiding ambiguous partial objects.
#   - Each sample contains 2-3 distractors, with at least one teal and one red distractor,
#     so the "ignore broken frames" behavior is repeatedly supported.
#
# main_assumptions_review:
#   - The 5x5 scan assumes every movable object occupies one exact 5x5 window. Still true.
#   - Border-uniformity is now a semantic test, not just a convenient detector.
#   - The code assumes broken distractors never combine with nearby objects into a fake
#     complete border. The generator enforces pairwise non-overlap for every 5x5 footprint.
#   - The stacking logic assumes at most two depth lanes per side; the generator still
#     constructs outputs with one edge lane and one inward lane only.
#
# generator_guarantees:
#   - Complete targets are generated exactly as in the source task and validated by main().
#   - Broken distractors are inserted afterward with full-side deletions, so they are never
#     misclassified as complete frames.
#   - All 5x5 footprints are non-overlapping and in-bounds, which keeps both the visuals
#     and the solver's scanning model clean.


# --- Configuration ---
GRID_H, GRID_W = 22, 22
SQ = 5
AZURE, RED = 8, 2
N_AZURE_RANGE = (3, 4)
N_RED_RANGE = (3, 5)
N_DISTRACTOR_RANGE = (2, 3)
MAX_INWARD = 2
INPUT_COL_RETRIES = 50
DISTRACTOR_RETRIES = 80
GEN_RETRIES = 700
MISSING_SIDES = ("top", "bottom", "left", "right")


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


def _make_patch(frame_color: int, inner_color: int) -> np.ndarray:
    patch = np.full((SQ, SQ), frame_color, dtype=int)
    patch[1:-1, 1:-1] = inner_color
    return patch


def _make_broken_patch(
    frame_color: int, inner_color: int, missing_side: str, bg: int
) -> np.ndarray:
    patch = _make_patch(frame_color, inner_color)
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


def _is_complete_square_patch(patch: np.ndarray) -> bool:
    border = np.concatenate(
        [patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]]
    )
    if int(border.min()) != int(border.max()):
        return False
    frame_col = int(border[0])
    if frame_col not in {AZURE, RED}:
        return False
    inner = patch[1:-1, 1:-1]
    return bool((inner != frame_col).any())


def _collect_complete_squares(g: np.ndarray) -> list[dict[str, object]]:
    h, w = g.shape
    squares: list[dict[str, object]] = []
    for r0 in range(h - SQ + 1):
        for c0 in range(w - SQ + 1):
            patch = crop_bbox(g, (r0, c0, r0 + SQ - 1, c0 + SQ - 1))
            if not _is_complete_square_patch(patch):
                continue
            squares.append(
                {"frame": int(patch[0, 0]), "r0": int(r0), "c0": int(c0), "patch": patch}
            )
    return squares


def main(grid: np.ndarray) -> np.ndarray:
    """
    Relocate only complete 5x5 framed squares to the side chosen by frame color.

    Input structure:
    - The background is the most common color.
    - Complete teal/red 5x5 frames are mixed with broken 5x5 frame-like distractors that
      are missing one border side but still have colored interiors.

    Rule:
    - Ignore every incomplete frame candidate; only intact 5x5 borders participate.
    - Teal frames move to the left edge and red frames move to the right edge, keeping rows.
    - Teal frames are processed by increasing input column, red frames by decreasing input column.
    - If a frame row-span overlaps a frame already parked on that side's edge lane, shift it
      inward by one frame width instead.

    Key insight: Completeness of the 5x5 border is the filter that separates real movers from
    distractors; after that, the original near-edge-first stacking rule applies unchanged.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))
    out = np.full_like(g, bg)

    def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
        return not (int(a1) < int(b0) or int(b1) < int(a0))

    squares = _collect_complete_squares(g)

    teal_squares = sorted(
        [s for s in squares if int(s["frame"]) == AZURE], key=lambda s: int(s["c0"])
    )
    placed_at_edge: list[tuple[int, int]] = []
    for square in teal_squares:
        r0 = int(square["r0"])
        r1 = r0 + SQ - 1
        has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_edge)
        dest_c = SQ if has_overlap else 0
        paste(out, square["patch"], r0, dest_c, transparent=bg)
        if dest_c == 0:
            placed_at_edge.append((r0, r1))

    red_squares = sorted(
        [s for s in squares if int(s["frame"]) == RED], key=lambda s: -int(s["c0"])
    )
    placed_at_edge = []
    for square in red_squares:
        r0 = int(square["r0"])
        r1 = r0 + SQ - 1
        has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_edge)
        dest_c = w - 2 * SQ if has_overlap else w - SQ
        paste(out, square["patch"], r0, dest_c, transparent=bg)
        if dest_c == w - SQ:
            placed_at_edge.append((r0, r1))

    return out


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
    sq_list: list[tuple[int, int, bool, np.ndarray]], ascending: bool
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

        ordered = [
            edges[idx] if typ == "edge" else inwards[idx] for typ, idx in order
        ]

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
) -> list[tuple[int, int, np.ndarray]] | None:
    n_distractors = random.randint(*N_DISTRACTOR_RANGE)
    frames = [AZURE, RED]
    while len(frames) < n_distractors:
        frames.append(random.choice([AZURE, RED]))
    random.shuffle(frames)

    distractors: list[tuple[int, int, np.ndarray]] = []
    taken = [(r, c) for r, c, _ in occupied]

    for frame_color in frames:
        inner_choices = [c for c in range(10) if c not in {bg, frame_color}]
        placed = False
        for _ in range(DISTRACTOR_RETRIES):
            r = random.randint(0, GRID_H - SQ)
            c = random.randint(0, GRID_W - SQ)
            if any(_rects_overlap(r, c, tr, tc) for tr, tc in taken):
                continue
            patch = _make_broken_patch(
                frame_color,
                random.choice(inner_choices),
                random.choice(MISSING_SIDES),
                bg,
            )
            if _is_complete_square_patch(patch):
                continue
            distractors.append((r, c, patch))
            taken.append((r, c))
            placed = True
            break
        if not placed:
            return None
    return distractors


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

    avail_azure = [c for c in range(10) if c not in {bg, AZURE}]
    avail_red = [c for c in range(10) if c not in {bg, RED}]
    if len(avail_azure) < n_azure or len(avail_red) < n_red:
        return None
    azure_colors = random.sample(avail_azure, n_azure)
    red_colors = random.sample(avail_red, n_red)

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

    all_targets = azure_input + red_input
    if _any_pairwise_overlap(all_targets):
        return None

    distractors = _place_distractors(all_targets, bg)
    if distractors is None:
        return None

    inp = np.full((GRID_H, GRID_W), bg, dtype=int)
    for r, c, patch in all_targets + distractors:
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
