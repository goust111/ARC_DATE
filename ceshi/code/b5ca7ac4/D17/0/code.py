# Variant of b5ca7ac4
# Dimension: D17 — 堆叠方向与边缘的对应关系 (meta)
# Change: 青色组向左堆叠时先占最左列（col 0），重叠时向右扩展；红色组向右堆叠时先占最右列（col max-4），重叠时向左扩... → 从内向外生长——青色从中间向左扩展

from __future__ import annotations

import random

import numpy as np

from arc_utils import crop_bbox, get_bg, paste


# Design analysis for the D17 variant
# ----------------------------------
# a) Concrete behavior and trace
#    - Objects stay the same as the base task: 5x5 framed squares, teal frame=8, red frame=2,
#      3x3 uniform interior, fixed 22x22 board, background is the modal color.
#    - The only semantic change is which column is treated as the "default" landing slot.
#      Base task:
#        teal default -> col 0, overlapping later teal -> col 5
#        red  default -> col 17, overlapping later red -> col 12
#      D17 variant:
#        teal default -> col 5, overlapping later teal -> col 0
#        red  default -> col 12, overlapping later red -> col 17
#      So both groups now start nearer the center and expand outward toward their edge.
#    - Small trace for teal with two squares whose row spans overlap:
#      input columns 2 and 10, processed ascending.
#      Base task: first square lands at col 0, second detects overlap and lands at col 5.
#      Variant:   first square lands at col 5, second detects overlap and lands at col 0.
#      The ordering signal is unchanged; only the mapping from "first/non-overlap" vs.
#      "later/overlap" to columns is inverted.
# b) Complexity calibration
#    - The visual difference is strong because the majority bucket per side moves from the edge
#      slot to the inner slot. The generator forces 1-2 outward-growing squares per side, so the
#      branching case is present, but the task still only has two depth levels per side.
#    - No extra color coding or asymmetry is needed because the location swap itself is visible.
# c) Implicit-assumption audit for main()
#    - Square detection via 5x5 window scan does not depend on edge-vs-inner placement, so it
#      stays valid.
#    - Sorting by input column is still the control signal; D17 does not change that.
#    - The overlap test originally recorded only squares placed at the edge slot. In D17 it must
#      instead record squares placed at the inner default slot, because later overlapping squares
#      are pushed outward from there.
#    - The solver still assumes at most two column slots per side. The generator preserves that by
#      only creating "default" and "outward" placements, never a third depth.
# d) Placement legality
#    - Output columns are fixed at teal {5,0} and red {12,17}. These bands do not overlap because
#      0..9 and 12..21 are disjoint on a 22-column board.
#    - Rows assigned to the default slot are pairwise non-overlapping. Rows assigned to the outward
#      slot each overlap at least one default row and are pairwise non-overlapping with each other,
#      so no two output squares collide inside a shared column.
#    - Input placement is still derived from a valid processing order and checked for pairwise
#      non-overlap before acceptance.
# e) Visual observability
#    - Humans can infer the rule from 2-4 examples because all teal squares collect on the left
#      half but mostly at col 5 rather than col 0, and red squares mostly at col 12 rather than
#      col 17. Overlapping row cases are the ones that get pushed outward.
#    - No hidden property such as color id ordering is used. Only visible input column order and
#      visible row overlap determine the outcome.
# f) Visual cleanliness
#    - The output still uses two narrow, well-separated side bands. There is no central clutter and
#      no ambiguous overpainting because squares in the same column never share rows.
# g) Color-change audit
#    - There is no new color mapping. Inner colors remain unique within each frame type only to
#      keep object identity readable, matching the base task's style.
# h) Generator guarantee against degeneration
#    - Each side always has at least one default-slot square and at least one outward-slot square.
#      Therefore the same input cannot match the base task's output: at least one square per side
#      occupies the opposite column compared with the original rule.
# i) New solver assumptions checked against generator
#    - main() now assumes that the squares intended for the outward slot appear later than at least
#      one row-overlapping default-slot square in processing order. _derive_input_positions() builds
#      exactly that dependency when it constructs input columns.


# --------------- solver ---------------

GRID_H, GRID_W = 22, 22
SQ = 5
AZURE, RED = 8, 2
LEFT_OUTER_COL = 0
LEFT_INNER_COL = SQ
RIGHT_INNER_COL = GRID_W - 2 * SQ
RIGHT_OUTER_COL = GRID_W - SQ

N_AZURE_RANGE = (3, 4)
N_RED_RANGE = (3, 5)
MAX_OUTWARD = 2
INPUT_COL_RETRIES = 50
GEN_RETRIES = 500


def main(grid: np.ndarray) -> np.ndarray:
    """
    Relocate bordered 5x5 squares into left and right stacks that grow outward from the center.

    Input structure:
    - The background is the most frequent color in a 22x22 grid.
    - Several teal-framed (8) and red-framed (2) 5x5 squares with colored 3x3 interiors are
      scattered without overlap.

    Rule:
    - Teal squares keep their rows and are processed by original column ascending; red squares keep
      their rows and are processed by original column descending.
    - The first non-overlapping teal square takes the inner-left slot at col 5, and later teal
      squares whose row spans overlap an already placed inner-left teal are pushed outward to col 0.
    - The first non-overlapping red square takes the inner-right slot at col 12, and later red
      squares whose row spans overlap an already placed inner-right red are pushed outward to col 17.

    Key insight: The ordering cue is unchanged from the base task, but the default landing slots are
    the center-near columns, so overlap causes outward growth toward the edges instead of inward growth.
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
            if frame_col not in {AZURE, RED}:
                continue
            inner = patch[1:-1, 1:-1]
            if not bool((inner != frame_col).any()):
                continue
            squares.append({"frame": frame_col, "r0": int(r0), "c0": int(c0), "patch": patch})

    teal_squares = sorted((s for s in squares if int(s["frame"]) == AZURE), key=lambda s: int(s["c0"]))
    placed_at_default: list[tuple[int, int]] = []
    for s in teal_squares:
        r0 = int(s["r0"])
        r1 = int(r0 + SQ - 1)
        has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_default)
        dest_c = LEFT_OUTER_COL if has_overlap else LEFT_INNER_COL
        paste(out, s["patch"], r0, dest_c, transparent=bg)
        if dest_c == LEFT_INNER_COL:
            placed_at_default.append((r0, r1))

    red_squares = sorted((s for s in squares if int(s["frame"]) == RED), key=lambda s: -int(s["c0"]))
    placed_at_default = []
    for s in red_squares:
        r0 = int(s["r0"])
        r1 = int(r0 + SQ - 1)
        has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_default)
        dest_c = RIGHT_OUTER_COL if has_overlap else int(W - 2 * SQ)
        paste(out, s["patch"], r0, dest_c, transparent=bg)
        if dest_c == int(W - 2 * SQ):
            placed_at_default.append((r0, r1))

    return out


# --------------- generator ---------------

def _make_patch(frame_color: int, inner_color: int) -> np.ndarray:
    """Create a SQxSQ bordered patch with uniform interior."""
    p = np.full((SQ, SQ), frame_color, dtype=int)
    p[1:-1, 1:-1] = inner_color
    return p


def _rows_overlap(r_a: int, r_b: int) -> bool:
    """Check whether two SQ-tall row spans overlap."""
    return abs(r_a - r_b) < SQ


def _rects_overlap(r1: int, c1: int, r2: int, c2: int) -> bool:
    """Check whether two SQxSQ rectangles overlap."""
    return abs(r1 - r2) < SQ and abs(c1 - c2) < SQ


def _any_pairwise_overlap(positions: list[tuple]) -> bool:
    """Return True if any two (row, col, ...) entries overlap as SQxSQ rectangles."""
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if _rects_overlap(positions[i][0], positions[i][1], positions[j][0], positions[j][1]):
                return True
    return False


def _assign_output_rows(n: int) -> tuple[list[int], list[int]] | None:
    """Pick default-slot rows and outward-slot rows for one color group."""
    max_outward = min(MAX_OUTWARD, n - 1)
    n_outward = random.randint(1, max_outward)
    n_default = n - n_outward

    all_rows = list(range(GRID_H - SQ + 1))
    random.shuffle(all_rows)
    default_rows: list[int] = []
    for r in all_rows:
        if all(not _rows_overlap(r, dr) for dr in default_rows):
            default_rows.append(r)
            if len(default_rows) == n_default:
                break
    if len(default_rows) < n_default:
        return None

    random.shuffle(all_rows)
    outward_rows: list[int] = []
    for r in all_rows:
        if not any(_rows_overlap(r, dr) for dr in default_rows):
            continue
        if any(_rows_overlap(r, orow) for orow in outward_rows):
            continue
        outward_rows.append(r)
        if len(outward_rows) == n_outward:
            break
    if len(outward_rows) < n_outward:
        return None
    return default_rows, outward_rows


def _place_side_squares(
    output: np.ndarray,
    default_rows: list[int],
    outward_rows: list[int],
    colors: list[int],
    frame_color: int,
    default_col: int,
    outward_col: int,
) -> list[tuple[int, int, bool, np.ndarray]]:
    """Render default-slot and outward-slot squares; mark True for default-slot squares."""
    placed: list[tuple[int, int, bool, np.ndarray]] = []
    for i, r in enumerate(default_rows):
        p = _make_patch(frame_color, colors[i])
        paste(output, p, r, default_col)
        placed.append((r, default_col, True, p))
    for i, r in enumerate(outward_rows):
        p = _make_patch(frame_color, colors[len(default_rows) + i])
        paste(output, p, r, outward_col)
        placed.append((r, outward_col, False, p))
    return placed


def _derive_input_positions(
    sq_list: list[tuple[int, int, bool, np.ndarray]],
    ascending: bool,
) -> list[tuple[int, int, np.ndarray]] | None:
    """Assign input columns that reproduce default-first then outward-on-overlap behavior."""
    defaults = [(r, oc, p) for (r, oc, is_default, p) in sq_list if is_default]
    outwards = [(r, oc, p) for (r, oc, is_default, p) in sq_list if not is_default]

    outward_deps = [
        [i for i, (dr, _, _) in enumerate(defaults) if _rows_overlap(orow, dr)]
        for (orow, _, _) in outwards
    ]

    n = len(sq_list)
    col_range = list(range(0, GRID_W - SQ + 1))
    if len(col_range) < n:
        return None

    for _ in range(INPUT_COL_RETRIES):
        default_indices = list(range(len(defaults)))
        random.shuffle(default_indices)
        order: list[tuple[str, int]] = [("default", i) for i in default_indices]

        outward_indices = list(range(len(outwards)))
        random.shuffle(outward_indices)
        for ow_idx in outward_indices:
            latest_pos = max(
                (
                    pos
                    for dep in outward_deps[ow_idx]
                    for pos, (typ, idx) in enumerate(order)
                    if typ == "default" and idx == dep
                ),
                default=-1,
            )
            insert_pos = random.randint(latest_pos + 1, len(order))
            order.insert(insert_pos, ("outward", ow_idx))

        ordered_squares = [
            (defaults[idx] if typ == "default" else outwards[idx]) for typ, idx in order
        ]

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
    """Single attempt to generate a valid input grid."""
    bg = random.choice([c for c in range(10) if c not in {AZURE, RED}])
    n_azure = random.randint(*N_AZURE_RANGE)
    n_red = random.randint(*N_RED_RANGE)

    azure_assign = _assign_output_rows(n_azure)
    if azure_assign is None:
        return None
    red_assign = _assign_output_rows(n_red)
    if red_assign is None:
        return None
    azure_default_rows, azure_outward_rows = azure_assign
    red_default_rows, red_outward_rows = red_assign

    avail_azure = [c for c in range(10) if c not in {bg, AZURE}]
    avail_red = [c for c in range(10) if c not in {bg, RED}]
    if len(avail_azure) < n_azure or len(avail_red) < n_red:
        return None
    azure_colors = random.sample(avail_azure, n_azure)
    red_colors = random.sample(avail_red, n_red)

    output = np.full((GRID_H, GRID_W), bg, dtype=int)
    azure_out = _place_side_squares(
        output,
        azure_default_rows,
        azure_outward_rows,
        azure_colors,
        AZURE,
        default_col=LEFT_INNER_COL,
        outward_col=LEFT_OUTER_COL,
    )
    red_out = _place_side_squares(
        output,
        red_default_rows,
        red_outward_rows,
        red_colors,
        RED,
        default_col=RIGHT_INNER_COL,
        outward_col=RIGHT_OUTER_COL,
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
