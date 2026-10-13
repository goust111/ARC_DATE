from __future__ import annotations
import numpy as np
from arc_utils import crop_bbox, get_bg, paste

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
    - If a square's rows overlap or adjoin an already-placed edge square, place it one square-width inward instead.

    Key insight: Near-edge-first ordering plus row-adjacency stacking determines which square sits at the edge vs. the second column.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    out = np.full_like(g, bg)

    SQUARE_SIZE = 5
    OVERLAP_THRESHOLD = 1  # adjacent rows count as overlap

    def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
        gap = max(int(a0 - b1), int(b0 - a1))
        return int(gap) <= int(OVERLAP_THRESHOLD)

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

