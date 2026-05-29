# Combo variant of b5ca7ac4 (index 31)
# combo_spec: D5/1+D7/0+D10/0
# Name: 垂直堆叠+紧密行排列+紧密裁剪

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

for parent in Path(__file__).resolve().parents:
    candidate = parent / "solvers" / "arc_utils.py"
    if candidate.exists():
        sys.path.insert(0, str(parent / "solvers"))
        break

from arc_utils import crop_bbox, get_bg, paste


# Design analysis for combo 31
# ----------------------------
# a) concrete_behavior_and_trace:
#    - Objects stay the same as the seed task: scattered 5x5 framed squares with
#      frame color 8 (teal) or 2 (red), each with a visible 3x3 interior.
#    - D5/1 rotates the motion axis: teal is organized against the top edge and
#      red against the bottom edge, so processing order must depend on input rows
#      rather than input columns.
#    - D7/0 then removes row preservation. In the fused rule, the k-th teal square
#      in top-to-bottom order is placed at rows 0,5,10,..., while the k-th red
#      square in bottom-to-top order is placed at rows H-5,H-10,H-15,....
#      Columns are kept, so the visible "identity" of each square is its frame,
#      interior color, and preserved horizontal position.
#    - D10/0 crops the final 22x22 canvas to the smallest non-background box.
#      Because D5/1 forces contact with the top and bottom edges, the crop is
#      usually horizontal only: blank outer columns disappear, producing a narrow
#      two-pile result.
#    - Example trace:
#      teal input rows/cols = [(3,2), (8,5), (14,1)] -> sorted top-down unchanged
#      -> output rows become [0,5,10] at cols [2,5,1].
#      red input rows/cols = [(16,13), (11,15), (6,12)] -> processed bottom-up
#      -> output rows become [17,12,7] at cols [13,15,12].
#      The seed task would have preserved those rows and moved columns instead;
#      this combo preserves columns, compacts rows, and finally trims side margins.
# b) complexity_calibration:
#    - Compact vertical packing requires each color group to fit inside 22 rows.
#      Therefore each group is capped at 4 squares; allowing 5 red squares would
#      require 25 rows and break the fused D5+D7 layout.
#    - To keep the change legible, generator columns come from two narrow but
#      distinct interior bands. This makes the crop visible and yields two compact
#      vertical piles instead of a sparse, ambiguous scatter.
#    - Unique interior colors are preserved within each frame class so humans can
#      track individual squares across the transformation.
# c) implicit_assumption_audit:
#    - 5x5 window scanning with monochrome border detection still works because the
#      object shape is unchanged.
#    - Original code assumed only columns changed and row spans decided inward
#      placement. That assumption no longer holds: row preservation is gone, and
#      the decisive relation is row order itself.
#    - The fused solver now assumes a compact top/bottom stack will fit vertically;
#      generator enforces at most 4 squares per color.
#    - The fused solver can overdraw if teal and red preserve overlapping columns
#      while their compact row ranges intersect. Generator therefore guarantees the
#      two color groups occupy disjoint column bands in every input.
# d) placement_legality:
#    - Input squares are placed with row gaps of exactly 5 inside each color group,
#      so they never overlap even when their preserved columns are close.
#    - Teal columns are sampled from {1,2,4,5}; red columns from {12,13,15,16}.
#      These 5x5 spans are disjoint, so cross-color collisions are impossible in
#      both input and output, despite overlapping row intervals after compaction.
#    - Output rows are deterministic: teal uses 0,5,10,15 and red uses
#      H-5,H-10,H-15,H-20 prefixes according to count, all within bounds.
# e) visual_observability:
#    - From a few examples, a human can see three coupled changes:
#      teal rises to the top, red sinks to the bottom, and both colors become
#      vertically compact while staying in the same horizontal lanes.
#    - Cropping is also visible because the outermost used columns are forced away
#      from col 0 and col 21, so the output width is always smaller than 22.
#    - No hidden ordering by color value is used; only visible row order drives it.
# f) visual_cleanliness:
#    - The two color classes live in separated column bands, so the compact stacks
#      remain readable even though teal and red overlap in row ranges.
#    - The final crop removes empty side margins and avoids a cluttered wide canvas.
# g) color_rule_inference:
#    - No new color remapping exists. Frame color still only decides membership:
#      teal means "top compact stack", red means "bottom compact stack".
# new_main_assumptions_checked_against_generator:
#    - `main()` assumes at least one detected square of each type is enough for
#      cropping; generator always produces 3-4 of each type.
#    - `main()` assumes ordering ties are either absent or harmless. Generator uses
#      strictly monotone input rows within each color group, so ties do not occur.
#    - `main()` assumes preserved columns will not create cross-color overwrite in
#      the compact output. Generator's split column bands guarantee that.


GRID_H, GRID_W = 22, 22
SQ = 5
AZURE, RED = 8, 2
N_AZURE_RANGE = (3, 4)
N_RED_RANGE = (3, 4)
GEN_RETRIES = 500
AZURE_COL_CHOICES = (1, 2, 4, 5)
RED_COL_CHOICES = (12, 13, 15, 16)


def _tight_crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    """Crop away border rows and columns containing only background."""
    nz = np.argwhere(grid != int(bg))
    if nz.size == 0:
        return np.full((0, 0), int(bg), dtype=int)
    r0, c0 = nz.min(axis=0)
    r1, c1 = nz.max(axis=0)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _find_squares(grid: np.ndarray) -> list[dict[str, int | np.ndarray]]:
    """Detect all 5x5 framed squares."""
    g = np.array(grid, dtype=int)
    H, W = g.shape
    squares: list[dict[str, int | np.ndarray]] = []
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
    return squares


def _render_compact_vertical(grid: np.ndarray) -> np.ndarray:
    """Render the fused D5+D7 arrangement on the full 22x22 canvas."""
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = np.full((H, W), bg, dtype=int)

    squares = _find_squares(g)

    teal_squares = sorted(
        [s for s in squares if int(s["frame"]) == AZURE],
        key=lambda s: (int(s["r0"]), int(s["c0"])),
    )
    for idx, s in enumerate(teal_squares):
        paste(out, s["patch"], int(idx * SQ), int(s["c0"]), transparent=bg)

    red_squares = sorted(
        [s for s in squares if int(s["frame"]) == RED],
        key=lambda s: (-int(s["r0"]), int(s["c0"])),
    )
    for idx, s in enumerate(red_squares):
        paste(out, s["patch"], int(H - (idx + 1) * SQ), int(s["c0"]), transparent=bg)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repack framed 5x5 squares into vertical color-specific stacks, then tight-crop.

    Input structure:
    - Background is the most frequent color.
    - Teal-framed (8) and red-framed (2) 5x5 squares with colored interiors are scattered on the canvas.

    Rule:
    - Teal squares keep their original columns and are processed from top to bottom; red squares keep their original columns and are processed from bottom to top.
    - Output rows are reassigned compactly: the first teal square goes at row 0, the next at row 5, etc., while red squares fill rows H-5, H-10, H-15, ... upward from the bottom edge.
    - After the compact top/bottom placement is rendered on the full canvas, delete any outer rows or columns that are entirely background.

    Key insight: The horizontal edge logic of the seed task is replaced by a vertical one, and the old row-preservation rule disappears entirely; visible input row order now determines each square's slot in a top or bottom compact stack, after which the answer is just the minimal non-background crop.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    out = _render_compact_vertical(g)
    return _tight_crop_non_bg(out, bg)


def _make_patch(frame_color: int, inner_color: int) -> np.ndarray:
    patch = np.full((SQ, SQ), frame_color, dtype=int)
    patch[1:-1, 1:-1] = inner_color
    return patch


def _sample_group_cols(choices: tuple[int, ...], n: int) -> list[int]:
    """Pick distinct preserved columns from a compact band."""
    cols = random.sample(list(choices), n)
    random.shuffle(cols)
    return [int(c) for c in cols]


def _pick_row_base(n: int, *, align_top: bool) -> int:
    """Pick an arithmetic-progression row base that avoids degenerate no-op stacking."""
    max_base = GRID_H - SQ * n
    forbidden = 0 if align_top else max_base
    candidates = [b for b in range(max_base + 1) if b != forbidden]
    return int(random.choice(candidates))


def _ordered_input_rows(n: int, *, align_top: bool) -> list[int]:
    """Return rows whose visible order matches the solver but differs from output rows."""
    base = _pick_row_base(n, align_top=align_top)
    if align_top:
        return [int(base + SQ * idx) for idx in range(n)]
    return [int(base + SQ * (n - 1 - idx)) for idx in range(n)]


def _build_expected_output(
    bg: int,
    teal_specs: list[dict[str, int | np.ndarray]],
    red_specs: list[dict[str, int | np.ndarray]],
) -> np.ndarray:
    """Render the intended compact full-canvas output, then crop it."""
    full = np.full((GRID_H, GRID_W), int(bg), dtype=int)
    for idx, spec in enumerate(teal_specs):
        paste(full, spec["patch"], int(idx * SQ), int(spec["c0"]))
    for idx, spec in enumerate(red_specs):
        paste(full, spec["patch"], int(GRID_H - (idx + 1) * SQ), int(spec["c0"]))
    return _tight_crop_non_bg(full, int(bg))


def _build_input(
    bg: int,
    teal_specs: list[dict[str, int | np.ndarray]],
    red_specs: list[dict[str, int | np.ndarray]],
) -> np.ndarray:
    """Place squares on the 22x22 input canvas with monotone rows per color."""
    inp = np.full((GRID_H, GRID_W), int(bg), dtype=int)
    teal_rows = _ordered_input_rows(len(teal_specs), align_top=True)
    red_rows = _ordered_input_rows(len(red_specs), align_top=False)

    for spec, row in zip(teal_specs, teal_rows):
        paste(inp, spec["patch"], int(row), int(spec["c0"]))
    for spec, row in zip(red_specs, red_rows):
        paste(inp, spec["patch"], int(row), int(spec["c0"]))
    return inp


def _row_shift_count(inp: np.ndarray) -> int:
    """Count how many squares change row under the fused rule."""
    moved = 0
    squares = _find_squares(inp)
    teal_in = sorted(
        [s for s in squares if int(s["frame"]) == AZURE],
        key=lambda s: (int(s["r0"]), int(s["c0"])),
    )
    red_in = sorted(
        [s for s in squares if int(s["frame"]) == RED],
        key=lambda s: (-int(s["r0"]), int(s["c0"])),
    )
    for idx, s in enumerate(teal_in):
        if int(s["r0"]) != int(idx * SQ):
            moved += 1
    for idx, s in enumerate(red_in):
        if int(s["r0"]) != int(GRID_H - (idx + 1) * SQ):
            moved += 1
    return moved


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([c for c in range(10) if c not in {AZURE, RED}])
    n_azure = random.randint(*N_AZURE_RANGE)
    n_red = random.randint(*N_RED_RANGE)

    avail_azure = [c for c in range(10) if c not in {bg, AZURE}]
    avail_red = [c for c in range(10) if c not in {bg, RED}]
    if len(avail_azure) < n_azure or len(avail_red) < n_red:
        return None

    azure_colors = random.sample(avail_azure, n_azure)
    red_colors = random.sample(avail_red, n_red)
    azure_cols = _sample_group_cols(AZURE_COL_CHOICES, n_azure)
    red_cols = _sample_group_cols(RED_COL_CHOICES, n_red)

    teal_specs = [
        {"frame": AZURE, "c0": int(c), "patch": _make_patch(AZURE, color)}
        for c, color in zip(azure_cols, azure_colors)
    ]
    red_specs = [
        {"frame": RED, "c0": int(c), "patch": _make_patch(RED, color)}
        for c, color in zip(red_cols, red_colors)
    ]

    expected = _build_expected_output(bg, teal_specs, red_specs)
    if expected.shape[1] >= GRID_W:
        return None

    inp = _build_input(bg, teal_specs, red_specs)
    if _row_shift_count(inp) < 4:
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
