# Combo variant of b5ca7ac4 (index 34)
# combo_spec: D12/2+D17/1
# Name: 蛇形堆叠+大量框架

from __future__ import annotations

import random

import numpy as np

from arc_utils import crop_bbox, get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - Base task: teal and red 5x5 framed squares are grouped by frame color, kept on their
#      original rows, and stacked against opposite edges. A row-overlap only creates one inward
#      offset, so the output has at most two columns per side.
#    - Combo rule here:
#      1. Increase the instance size to 10-12 frames total by using 5-6 teal and 5-6 red squares.
#      2. Replace the one-step inward rule with a serpentine chain per color. After sorting by input
#         column (teal ascending, red descending), consecutive same-color squares whose row spans
#         overlap continue the same chain. Their landing slots follow the snake index sequence
#         0, 1, 0, 2, 1, 3, 2, ... .
#      3. Left-group columns are slot*5 from the left edge; right-group columns mirror that from the
#         right edge. With six squares this yields four visible horizontal layers on a 40x40 board.
#    - Small teal trace:
#      input-order rows 0,3,6,9,12 produce output slots 0,1,0,2,1 -> columns 0,5,0,10,5.
#      The same chain now swings inward, back toward the edge, then deeper again, producing the
#      intended zigzag stack instead of the base task's flat two-column pile.
# b) Complexity calibration
#    - The visual change is not cosmetic: long runs now occupy three or four horizontal layers,
#      and the left/right stacks show an obvious sawtooth profile. This prevents degeneration back
#      to the original "edge or one-step inward" behavior.
#    - The control signal stays human-visible: only frame color, input-column order, and row-span
#      overlap matter. No hidden use of color ids or arithmetic on interior colors is introduced.
# c) Implicit-assumption audit for main()
#    - 5x5 window scanning still works because object shape and colors are unchanged.
#    - The base solver implicitly assumed that any overlap case needs only one alternative column.
#      That assumption is broken by D12/2, so the new solver tracks chain position instead.
#    - The new solver assumes the intended snake is expressed by consecutive-overlap chains in the
#      color-sorted order. The generator enforces exactly that, so the rule is stable and visible.
#    - Reusing a previous slot later in the chain assumes those two squares no longer row-overlap.
#      The generator guarantees this with a fixed 3-row step, so repeated slots are collision-free.
# d) Placement legality
#    - The generator uses a 40x40 grid. Left slots 0/5/10/15 occupy cols 0..19, right slots
#      35/30/25/20 occupy cols 20..39, so the two color groups never collide horizontally.
#    - Within a side, consecutive rows differ by 3, so neighbors overlap and extend the snake, while
#      squares that revisit the same slot are at least 6 rows apart and therefore do not overlap.
#    - Input squares use sparse, pre-separated columns and the two colors are placed in disjoint
#      vertical halves, so the scattered source layout is always non-overlapping.
# e) Visual observability
#    - Across 2-4 examples, a human can see that same-color frames are first ordered by their
#      original horizontal position and then packed into an edge-anchored zigzag with repeated
#      inward/backward swings. The 10-12 objects make the alternation obvious rather than subtle.
# f) Visual cleanliness
#    - Even with more objects, each side stays in its own half-width band and one vertical half of
#      the canvas. That keeps the output legible: no mixed-color overpainting and no ambiguous
#      central clutter.
# g) Generator anti-degeneration check
#    - Every generated chain has length at least 5, so each side necessarily reaches slot 2 and can
#      reach slot 3. The original solver would collapse those same inputs into only two columns, so
#      the generated tasks genuinely exercise the fused behavior.


GRID_H, GRID_W = 40, 40
SQ = 5
AZURE, RED = 8, 2
N_AZURE_RANGE = (5, 6)
N_RED_RANGE = (5, 6)
CHAIN_STEP = 3
GEN_RETRIES = 200

AZURE_INPUT_POOL = [1, 7, 13, 19, 25, 31]
RED_INPUT_POOL = [34, 28, 22, 16, 10, 4]


def _rows_overlap(r_a: int, r_b: int) -> bool:
    """Return True iff two SQ-tall row spans overlap."""
    return abs(int(r_a) - int(r_b)) < SQ


def _rects_overlap(r1: int, c1: int, r2: int, c2: int) -> bool:
    """Return True iff two SQxSQ rectangles overlap."""
    return abs(int(r1) - int(r2)) < SQ and abs(int(c1) - int(c2)) < SQ


def _snake_slot(pos: int) -> int:
    """Map chain index 0,1,2,3,... to snake slots 0,1,0,2,1,3,2,... ."""
    pos = int(pos)
    if pos == 0:
        return 0
    return pos // 2 + (1 if pos % 2 == 1 else -1)


def _make_patch(frame_color: int, inner_color: int) -> np.ndarray:
    """Create one 5x5 frame patch with a uniform 3x3 interior."""
    patch = np.full((SQ, SQ), int(frame_color), dtype=int)
    patch[1:-1, 1:-1] = int(inner_color)
    return patch


def _detect_squares(g: np.ndarray) -> list[dict[str, int | np.ndarray]]:
    """Find all valid 5x5 framed squares of the two task colors."""
    H, W = g.shape
    squares: list[dict[str, int | np.ndarray]] = []
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
            squares.append({"frame": frame, "r0": int(r0), "c0": int(c0), "patch": patch})
    return squares


def _stack_group(
    out: np.ndarray,
    squares: list[dict[str, int | np.ndarray]],
    *,
    left_side: bool,
    bg: int,
) -> None:
    """Place one color group using consecutive-overlap snake chains."""
    W = int(out.shape[1])
    placed_by_slot: dict[int, list[tuple[int, int]]] = {}
    prev_span: tuple[int, int] | None = None
    chain_pos = 0

    for idx, sq in enumerate(squares):
        r0 = int(sq["r0"])
        r1 = int(r0 + SQ - 1)
        if idx == 0 or prev_span is None or not (r0 <= prev_span[1] and prev_span[0] <= r1):
            chain_pos = 0
        else:
            chain_pos += 1

        slot = _snake_slot(chain_pos)
        while any(r0 <= rr1 and rr0 <= r1 for rr0, rr1 in placed_by_slot.get(slot, [])):
            slot += 1

        dest_c = int(slot * SQ) if left_side else int(W - SQ * (slot + 1))
        paste(out, sq["patch"], r0, dest_c, transparent=bg)
        placed_by_slot.setdefault(slot, []).append((r0, r1))
        prev_span = (r0, r1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Gather many colored 5x5 frames into left and right edge stacks that zigzag inward in a snake.

    Input structure:
    - The board is larger than the base task and uses the modal color as background.
    - Teal-framed (8) and red-framed (2) 5x5 squares with colored interiors are scattered without
      overlap, with 5-6 frames of each border color.

    Rule:
    - Keep every square on its original row, sort teal by input column ascending and red by input
      column descending, and handle the two colors independently.
    - Within one color, consecutive sorted squares whose 5-row spans overlap continue a snake chain.
      The chain lands in edge-relative slot sequence 0,1,0,2,1,3,..., so teal uses columns
      0,5,0,10,5,15,... and red mirrors that as W-5,W-10,W-5,W-15,W-10,W-20,... .
    - If a sorted square does not overlap the immediately previous square of its color, it starts a
      fresh chain back at slot 0.

    Key insight: The task is no longer a binary "edge vs. one step inward" stack. The visible cue is
    a long overlap chain in the color-sorted order, and that chain oscillates back and forth while
    gradually reaching deeper layers, producing the sawtooth pile-up.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    out = np.full_like(g, bg)

    squares = _detect_squares(g)
    teal = sorted((sq for sq in squares if int(sq["frame"]) == AZURE), key=lambda sq: int(sq["c0"]))
    red = sorted((sq for sq in squares if int(sq["frame"]) == RED), key=lambda sq: -int(sq["c0"]))

    _stack_group(out, teal, left_side=True, bg=bg)
    _stack_group(out, red, left_side=False, bg=bg)
    return out


def _make_chain_rows(n: int, upper_half: bool) -> list[int]:
    """Build one visible overlap chain whose reused slots are collision-free."""
    if upper_half:
        max_start = 15 - CHAIN_STEP * (n - 1)
        start = random.randint(0, max_start)
    else:
        min_start = 20
        max_start = GRID_H - SQ - CHAIN_STEP * (n - 1)
        start = random.randint(min_start, max_start)
    return [int(start + i * CHAIN_STEP) for i in range(n)]


def _output_col(slot: int, *, left_side: bool) -> int:
    """Convert a snake slot index into an output column."""
    return int(slot * SQ) if left_side else int(GRID_W - SQ * (slot + 1))


def _build_side_output(
    output: np.ndarray,
    rows: list[int],
    frame_color: int,
    colors: list[int],
    *,
    left_side: bool,
) -> list[tuple[int, np.ndarray]]:
    """Render one side's snake chain and return (row, patch) in processing order."""
    placed: list[tuple[int, np.ndarray]] = []
    for idx, r0 in enumerate(rows):
        patch = _make_patch(frame_color, colors[idx])
        dest_c = _output_col(_snake_slot(idx), left_side=left_side)
        paste(output, patch, int(r0), dest_c)
        placed.append((int(r0), patch))
    return placed


def _pick_input_cols(n: int, *, ascending: bool) -> list[int]:
    """Pick well-spaced input columns that preserve the intended processing order."""
    pool = AZURE_INPUT_POOL if ascending else RED_INPUT_POOL
    idxs = sorted(random.sample(range(len(pool)), n))
    return [int(pool[i]) for i in idxs]


def _any_pairwise_overlap(positions: list[tuple[int, int, np.ndarray]]) -> bool:
    """Return True iff any two placed SQxSQ patches overlap."""
    for i in range(len(positions)):
        r1, c1, _ = positions[i]
        for j in range(i + 1, len(positions)):
            r2, c2, _ = positions[j]
            if _rects_overlap(r1, c1, r2, c2):
                return True
    return False


def _gen_attempt() -> np.ndarray | None:
    """Generate one valid scattered input whose output shows the fused snake behavior."""
    bg = random.choice([c for c in range(10) if c not in {AZURE, RED}])
    n_azure = random.randint(*N_AZURE_RANGE)
    n_red = random.randint(*N_RED_RANGE)

    azure_upper = bool(random.getrandbits(1))
    azure_rows = _make_chain_rows(n_azure, upper_half=azure_upper)
    red_rows = _make_chain_rows(n_red, upper_half=not azure_upper)

    azure_colors = random.sample([c for c in range(10) if c not in {bg, AZURE}], n_azure)
    red_colors = random.sample([c for c in range(10) if c not in {bg, RED}], n_red)

    output = np.full((GRID_H, GRID_W), bg, dtype=int)
    azure_items = _build_side_output(output, azure_rows, AZURE, azure_colors, left_side=True)
    red_items = _build_side_output(output, red_rows, RED, red_colors, left_side=False)

    azure_cols = _pick_input_cols(n_azure, ascending=True)
    red_cols = _pick_input_cols(n_red, ascending=False)

    positions = [(r0, c0, patch) for (r0, patch), c0 in zip(azure_items, azure_cols)]
    positions.extend((r0, c0, patch) for (r0, patch), c0 in zip(red_items, red_cols))
    if _any_pairwise_overlap(positions):
        return None

    inp = np.full((GRID_H, GRID_W), bg, dtype=int)
    for r0, c0, patch in positions:
        paste(inp, patch, int(r0), int(c0))

    if np.array_equal(main(inp), output):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
