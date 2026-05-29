# Combo variant of b5ca7ac4 (index 5)
# combo_spec: D3/1+D8/1
# Name: 图案内部+旋转变换
#
# Design analysis for the fused variant
# a) Concrete trace:
#    In the seed task, a teal-framed 5x5 square with a solid 3x3 interior moves left and keeps
#    its appearance. Here that same object still moves by the original edge-stacking rule, but its
#    3x3 interior is now a two-color asymmetric motif such as an L-shape. After the move, the full
#    5x5 patch is rotated 90 degrees CCW. The border still looks like the same frame, while the
#    interior motif changes orientation, so the motion and the rotation are visible together.
# b) Complexity calibration:
#    The motif library is intentionally small and only contains patterns that are not 90-degree
#    symmetric. This prevents degeneration back into the original pure-fill task, while keeping the
#    hypothesis space small enough that 2-4 ARC examples can support it.
# c) Hidden assumptions changed from the seed:
#    The seed detector only needed "inner != frame somewhere". This variant assumes each true square
#    has a uniform 1-pixel border in {2,8}, an interior that uses exactly two colors, neither equal
#    to the border or background, and a non-uniform 3x3 motif. The generator enforces all of these.
#    The placement order and row-overlap logic from the seed are unchanged, so those assumptions stay
#    compatible with the original output-first construction.
# d) Placement legality:
#    Input squares are still generated as non-overlapping 5x5 patches inside a 22x22 grid. Output
#    uses the original two-column-per-side layout (edge or one-step inward), so all placements stay
#    in bounds and no new overlap mode is introduced by the rotation.
# e) Visual observability:
#    The rule is human-visible because every object keeps its row and group color while its interior
#    motif turns after relocation. No ordering depends on numeric color identities beyond the already
#    visible red/teal group borders.
# f) Visual cleanliness:
#    The grid keeps the seed task's sparse layout. Motifs are confined to the 3x3 interior, so the
#    border grouping signal stays legible and the output edge stacks remain tidy.
# g) Generator anti-degeneration check:
#    Because every chosen motif is asymmetric under 90-degree rotation and uses two distinct colors,
#    the rotated output differs from the seed task's unrotated output for every generated square.

from __future__ import annotations

import random

import numpy as np

from arc_utils import crop_bbox, get_bg, paste


GRID_H, GRID_W = 22, 22
SQ = 5
INNER = 3
AZURE, RED = 8, 2
N_AZURE_RANGE = (3, 4)
N_RED_RANGE = (3, 5)
MAX_INWARD = 2
INPUT_COL_RETRIES = 80
GEN_RETRIES = 800

PATTERN_MASKS = (
    np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=int),
    np.array([[0, 1, 0], [0, 1, 0], [1, 1, 0]], dtype=int),
    np.array([[1, 1, 1], [0, 0, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=int),
)


def _rotated_patch(patch: np.ndarray) -> np.ndarray:
    """Rotate a 5x5 framed patch by 90 degrees CCW."""
    return np.rot90(patch, k=1)


def _rows_overlap(r_a: int, r_b: int) -> bool:
    """Check whether two SQ-tall row spans overlap."""
    return abs(int(r_a) - int(r_b)) < SQ


def _rects_overlap(r1: int, c1: int, r2: int, c2: int) -> bool:
    """Check whether two SQxSQ rectangles overlap."""
    return abs(int(r1) - int(r2)) < SQ and abs(int(c1) - int(c2)) < SQ


def _any_pairwise_overlap(positions: list[tuple[int, int, np.ndarray]]) -> bool:
    """Return True if any two (r, c, patch) entries overlap as SQxSQ rects."""
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if _rects_overlap(positions[i][0], positions[i][1], positions[j][0], positions[j][1]):
                return True
    return False


def _make_patch(frame_color: int, base_color: int, motif_color: int, mask: np.ndarray) -> np.ndarray:
    """Create a SQxSQ framed patch whose 3x3 interior carries a two-color motif."""
    patch = np.full((SQ, SQ), int(frame_color), dtype=int)
    inner = np.full((INNER, INNER), int(base_color), dtype=int)
    inner[mask.astype(bool)] = int(motif_color)
    patch[1:-1, 1:-1] = inner
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move framed 5x5 motif squares to the side assigned by their border color, then rotate them.

    Input structure:
    - bg is the most common color.
    - Each object is a 5x5 square with a uniform teal (8) or red (2) border.
    - The inner 3x3 area is a non-uniform two-color motif, so its orientation is visually meaningful.

    Rule:
    - Teal-framed squares move to the left side; red-framed squares move to the right side.
    - Every square keeps its original row and is processed by the same column ordering as the seed task
      (teal left-to-right, red right-to-left).
    - If a square row-overlaps an already placed edge square from its own group, it shifts one square
      inward instead of taking the outermost column.
    - After choosing the destination slot, rotate the entire 5x5 square 90 degrees CCW before placing it.

    Key insight:
    The border color still decides the destination side, while input column order and row-overlap still
    decide outer vs. inner stacking; the new information is that the interior motif must be preserved and
    turned during relocation, so comparing motif orientation before and after the move reveals the extra step.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = np.full_like(g, bg)

    def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
        return not (int(a1) < int(b0) or int(b1) < int(a0))

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
            inner_colors = {int(v) for v in inner.flatten()}
            if len(inner_colors) != 2:
                continue
            if bg in inner_colors or frame_col in inner_colors:
                continue

            squares.append({"frame": frame_col, "r0": int(r0), "c0": int(c0), "patch": patch})

    teal_squares = sorted((s for s in squares if int(s["frame"]) == AZURE), key=lambda s: int(s["c0"]))
    placed_at_edge: list[tuple[int, int]] = []
    for sq in teal_squares:
        r0 = int(sq["r0"])
        r1 = int(r0 + SQ - 1)
        has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_edge)
        dest_c = SQ if has_overlap else 0
        paste(out, _rotated_patch(np.array(sq["patch"], dtype=int)), r0, int(dest_c), transparent=bg)
        if dest_c == 0:
            placed_at_edge.append((r0, r1))

    red_squares = sorted((s for s in squares if int(s["frame"]) == RED), key=lambda s: -int(s["c0"]))
    placed_at_edge = []
    for sq in red_squares:
        r0 = int(sq["r0"])
        r1 = int(r0 + SQ - 1)
        has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_edge)
        dest_c = int(W - 2 * SQ) if has_overlap else int(W - SQ)
        paste(out, _rotated_patch(np.array(sq["patch"], dtype=int)), r0, dest_c, transparent=bg)
        if dest_c == int(W - SQ):
            placed_at_edge.append((r0, r1))

    return out


def _assign_output_rows(n: int) -> tuple[list[int], list[int]] | None:
    """Pick edge and inward row positions for n squares on one side."""
    max_inward = min(MAX_INWARD, n - 1)
    n_inward = random.randint(1, max_inward)
    n_edge = n - n_inward

    all_rows = list(range(GRID_H - SQ + 1))
    random.shuffle(all_rows)
    edge_rows: list[int] = []
    for r in all_rows:
        if all(not _rows_overlap(r, er) for er in edge_rows):
            edge_rows.append(int(r))
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
        inward_rows.append(int(r))
        if len(inward_rows) == n_inward:
            break
    if len(inward_rows) < n_inward:
        return None
    return edge_rows, inward_rows


def _choose_square_specs(frame_color: int, bg: int, count: int) -> list[tuple[int, int, np.ndarray]]:
    """Choose unique base colors plus motif colors and masks for one side."""
    avail = [c for c in range(10) if c not in {bg, frame_color}]
    if len(avail) < count:
        raise ValueError("not enough colors for unique bases")
    base_colors = random.sample(avail, count)
    masks = random.sample(list(PATTERN_MASKS), count)
    specs = []
    for base_color, mask in zip(base_colors, masks):
        motif_choices = [c for c in avail if c != base_color]
        motif_color = random.choice(motif_choices)
        specs.append((int(base_color), int(motif_color), np.array(mask, dtype=int)))
    return specs


def _place_side_squares(
    output: np.ndarray,
    edge_rows: list[int],
    inward_rows: list[int],
    specs: list[tuple[int, int, np.ndarray]],
    frame_color: int,
    edge_col: int,
    inward_col: int,
) -> list[tuple[int, int, bool, np.ndarray]]:
    """Render rotated output squares and return input-patch metadata."""
    placed: list[tuple[int, int, bool, np.ndarray]] = []
    spec_idx = 0
    for r in edge_rows:
        base_color, motif_color, mask = specs[spec_idx]
        input_patch = _make_patch(frame_color, base_color, motif_color, mask)
        paste(output, _rotated_patch(input_patch), int(r), int(edge_col))
        placed.append((int(r), int(edge_col), True, input_patch))
        spec_idx += 1
    for r in inward_rows:
        base_color, motif_color, mask = specs[spec_idx]
        input_patch = _make_patch(frame_color, base_color, motif_color, mask)
        paste(output, _rotated_patch(input_patch), int(r), int(inward_col))
        placed.append((int(r), int(inward_col), False, input_patch))
        spec_idx += 1
    return placed


def _derive_input_positions(
    sq_list: list[tuple[int, int, bool, np.ndarray]],
    ascending: bool,
) -> list[tuple[int, int, np.ndarray]] | None:
    """Assign random input columns that reproduce the intended edge/inward placement."""
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
        order: list[tuple[str, int]] = [("edge", idx) for idx in edge_indices]

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
        for i, (r, oc, patch) in enumerate(ordered_squares):
            ic = int(chosen_cols[i])
            if ic == int(oc):
                ok = False
                break
            result.append((int(r), ic, patch))
        if ok and not _any_pairwise_overlap(result):
            return result
    return None


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid."""
    bg = random.choice([c for c in range(10) if c not in {AZURE, RED}])
    n_azure = random.randint(*N_AZURE_RANGE)
    n_red = random.randint(*N_RED_RANGE)

    azure_assign = _assign_output_rows(n_azure)
    red_assign = _assign_output_rows(n_red)
    if azure_assign is None or red_assign is None:
        return None
    azure_edge_rows, azure_inward_rows = azure_assign
    red_edge_rows, red_inward_rows = red_assign

    try:
        azure_specs = _choose_square_specs(AZURE, bg, n_azure)
        red_specs = _choose_square_specs(RED, bg, n_red)
    except ValueError:
        return None

    output = np.full((GRID_H, GRID_W), bg, dtype=int)
    azure_out = _place_side_squares(output, azure_edge_rows, azure_inward_rows, azure_specs, AZURE, 0, SQ)
    red_out = _place_side_squares(
        output,
        red_edge_rows,
        red_inward_rows,
        red_specs,
        RED,
        GRID_W - SQ,
        GRID_W - 2 * SQ,
    )

    azure_input = _derive_input_positions(azure_out, ascending=True)
    red_input = _derive_input_positions(red_out, ascending=False)
    if azure_input is None or red_input is None:
        return None

    all_positions = azure_input + red_input
    if _any_pairwise_overlap(all_positions):
        return None

    inp = np.full((GRID_H, GRID_W), bg, dtype=int)
    for r, c, patch in all_positions:
        paste(inp, patch, int(r), int(c))

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
