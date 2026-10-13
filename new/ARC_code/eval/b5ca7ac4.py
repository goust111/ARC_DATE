from __future__ import annotations
import numpy as np
from arc_utils import crop_bbox, get_bg, paste
import random

def main(grid: np.ndarray) -> np.ndarray:
    """
    Move bordered 5×5 squares to their designated grid edges by border color.

    Input structure:
    - bg = most common color.
    - Azure (8) and red (2) bordered 5×5 squares with colored interiors, scattered across the grid.

    Rule:
    - Azure squares → left edge (col 0); red squares → right edge (col W−5).
    - Each square keeps its original row.
    - Process azure left-to-right, red right-to-left (nearest-to-edge first).
    - If a square's rows overlap an already-placed edge square, place it one square-width inward instead.

    Key insight: Near-edge-first ordering plus row-overlap determines which square sits at the edge vs. the second column.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    out = np.full_like(g, bg)

    SQUARE_SIZE = 5
    def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
        return not (int(a1) < int(b0) or int(b1) < int(a0))

    # Detect bordered squares by scanning 5x5 windows. This is robust even if frame-color pixels
    # are connected between squares in the input.
    squares = []
    for r0 in range(H - SQUARE_SIZE + 1):
        for c0 in range(W - SQUARE_SIZE + 1):
            patch = crop_bbox(g, (int(r0), int(c0), int(r0 + SQUARE_SIZE - 1), int(c0 + SQUARE_SIZE - 1)))

            border = np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])
            if int(border.min()) != int(border.max()):
                continue
            frame_col = int(border[0])
            if frame_col not in {2, 8}:
                continue

            inner = patch[1:-1, 1:-1]
            if not bool((inner != int(frame_col)).any()):
                continue

            squares.append({"frame": int(frame_col), "r0": int(r0), "c0": int(c0), "patch": patch})

    # Azure squares: sort by original column ascending (leftmost first).
    azure_squares = sorted([s for s in squares if int(s["frame"]) == 8], key=lambda s: int(s["c0"]))
    placed_at_default: list[tuple[int, int]] = []
    for s in azure_squares:
        r0 = int(s["r0"])
        r1 = int(r0 + SQUARE_SIZE - 1)
        has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_default)
        dest_c = int(SQUARE_SIZE) if has_overlap else 0
        paste(out, s["patch"], r0, dest_c, transparent=bg)
        if dest_c == 0:
            placed_at_default.append((r0, r1))

    # Red squares: sort by original column descending (rightmost first).
    red_squares = sorted([s for s in squares if int(s["frame"]) == 2], key=lambda s: -int(s["c0"]))
    placed_at_default = []
    for s in red_squares:
        r0 = int(s["r0"])
        r1 = int(r0 + SQUARE_SIZE - 1)
        has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_default)
        dest_c = int(W - 2 * SQUARE_SIZE) if has_overlap else int(W - SQUARE_SIZE)
        paste(out, s["patch"], r0, dest_c, transparent=bg)
        if dest_c == int(W - SQUARE_SIZE):
            placed_at_default.append((r0, r1))

    return out

# --------------- generator (c) ---------------

# --- Configuration ---
GRID_H, GRID_W = 22, 22          # canvas dimensions
SQ = 5                            # square side length (frame + 3x3 interior)
AZURE, RED = 8, 2                 # frame colors
N_AZURE_RANGE = (3, 4)            # azure square count range
N_RED_RANGE = (3, 5)              # red square count range
MAX_INWARD = 2                    # max inward squares per side
INPUT_COL_RETRIES = 50            # retries for input column assignment per side
GEN_RETRIES = 500                 # outer generate_input retry limit


def _make_patch(frame_color: int, inner_color: int) -> np.ndarray:
    """Create a SQxSQ bordered patch with uniform interior."""
    p = np.full((SQ, SQ), frame_color, dtype=int)
    p[1:-1, 1:-1] = inner_color
    return p


def _rows_overlap(r_a: int, r_b: int) -> bool:
    """Check whether two SQ-tall row spans overlap."""
    return abs(r_a - r_b) < SQ


def _rects_overlap(r1, c1, r2, c2) -> bool:
    """Check whether two SQxSQ rectangles overlap."""
    return abs(r1 - r2) < SQ and abs(c1 - c2) < SQ


def _any_pairwise_overlap(positions: list[tuple]) -> bool:
    """Return True if any two (r, c, ...) entries overlap as SQxSQ rects."""
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if _rects_overlap(positions[i][0], positions[i][1],
                              positions[j][0], positions[j][1]):
                return True
    return False


def _assign_output_rows(n: int) -> tuple[list[int], list[int]] | None:
    """Pick edge and inward row positions for *n* squares on one side.

    Returns (edge_rows, inward_rows) or None on failure.
    Edge squares have non-overlapping row spans. Each inward square must
    row-overlap at least one edge square.
    """
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
    output: np.ndarray, edge_rows: list[int], inward_rows: list[int],
    colors: list[int], frame_color: int, edge_col: int, inward_col: int,
) -> list[tuple[int, int, bool, np.ndarray]]:
    """Render edge and inward squares onto *output*, return metadata list.

    Each entry is (row, out_col, is_edge, patch).
    """
    placed: list[tuple[int, int, bool, np.ndarray]] = []
    for i, r in enumerate(edge_rows):
        p = _make_patch(frame_color, colors[i])
        paste(output, p, r, edge_col)
        placed.append((r, edge_col, True, p))
    for i, r in enumerate(inward_rows):
        p = _make_patch(frame_color, colors[len(edge_rows) + i])
        paste(output, p, r, inward_col)
        placed.append((r, inward_col, False, p))
    return placed


def _derive_input_positions(
    sq_list: list[tuple[int, int, bool, np.ndarray]], ascending: bool,
) -> list[tuple[int, int, np.ndarray]] | None:
    """Assign random input columns that reproduce the solver's edge/inward placement.

    For ascending (azure), the solver processes squares by input col ascending;
    for descending (red), by input col descending. An inward square must appear
    after at least one of its row-overlapping edge squares in processing order.

    Returns list of (row, input_col, patch) or None on failure.
    """
    edges = [(r, oc, p) for (r, oc, is_edge, p) in sq_list if is_edge]
    inwards = [(r, oc, p) for (r, oc, is_edge, p) in sq_list if not is_edge]

    # For each inward square, find which edge squares it row-overlaps
    inward_deps = [[i for i, (er, _, _) in enumerate(edges) if _rows_overlap(ir, er)]
                   for (ir, _, _) in inwards]

    n = len(sq_list)
    col_range = list(range(0, GRID_W - SQ + 1))
    if len(col_range) < n:
        return None

    for _ in range(INPUT_COL_RETRIES):
        # Build processing order: edges first (shuffled), inwards inserted after deps
        edge_indices = list(range(len(edges)))
        random.shuffle(edge_indices)
        order: list[tuple[str, int]] = [('edge', i) for i in edge_indices]

        inward_order = list(range(len(inwards)))
        random.shuffle(inward_order)
        for iw_idx in inward_order:
            # Find the latest position of any dependency edge in the order
            latest_pos = max(
                (pos for d in inward_deps[iw_idx]
                 for pos, (typ, idx) in enumerate(order)
                 if typ == 'edge' and idx == d),
                default=-1,
            )
            insert_pos = random.randint(latest_pos + 1, len(order))
            order.insert(insert_pos, ('inward', iw_idx))

        # Resolve each entry to (row, out_col, patch) in processing order
        ordered_squares = [
            (edges[idx] if typ == 'edge' else inwards[idx]) for typ, idx in order
        ]

        # Pick n distinct sorted input cols; reverse for descending side
        chosen_cols = sorted(random.sample(col_range, n))
        if not ascending:
            chosen_cols = chosen_cols[::-1]

        # Map cols to squares; reject if any input col == output col
        result = []
        ok = True
        for i, (r, oc, p) in enumerate(ordered_squares):
            ic = chosen_cols[i]
            if ic == oc:
                ok = False
                break
            result.append((r, ic, p))
        if not ok:
            continue

        if not _any_pairwise_overlap(result):
            return result

    return None


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid."""
    bg = random.choice([c for c in range(10) if c not in {AZURE, RED}])
    n_azure = random.randint(*N_AZURE_RANGE)
    n_red = random.randint(*N_RED_RANGE)

    # Assign output row positions for each side
    azure_assign = _assign_output_rows(n_azure)
    if azure_assign is None:
        return None
    red_assign = _assign_output_rows(n_red)
    if red_assign is None:
        return None
    azure_edge_rows, azure_inward_rows = azure_assign
    red_edge_rows, red_inward_rows = red_assign

    # Assign unique inner colors per frame type
    avail_azure = [c for c in range(10) if c not in {bg, AZURE}]
    avail_red = [c for c in range(10) if c not in {bg, RED}]
    if len(avail_azure) < n_azure or len(avail_red) < n_red:
        return None
    azure_colors = random.sample(avail_azure, n_azure)
    red_colors = random.sample(avail_red, n_red)

    # Build output grid
    output = np.full((GRID_H, GRID_W), bg, dtype=int)
    azure_out = _place_side_squares(
        output, azure_edge_rows, azure_inward_rows, azure_colors,
        AZURE, edge_col=0, inward_col=SQ)
    red_out = _place_side_squares(
        output, red_edge_rows, red_inward_rows, red_colors,
        RED, edge_col=GRID_W - SQ, inward_col=GRID_W - 2 * SQ)

    # Derive input column assignments
    azure_input = _derive_input_positions(azure_out, ascending=True)
    if azure_input is None:
        return None
    red_input = _derive_input_positions(red_out, ascending=False)
    if red_input is None:
        return None

    # Cross-side overlap check
    all_positions = azure_input + red_input
    if _any_pairwise_overlap(all_positions):
        return None

    # Build and validate input grid
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
