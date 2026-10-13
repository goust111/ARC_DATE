import random
from collections import deque
from typing import Dict, List, Tuple

import numpy as np

BLACK = 0
GRAY = 5
YELLOW = 4
MAGENTA = 6

ORI_UP_LEFT = 0
ORI_UP_RIGHT = 1
ORI_DOWN_LEFT = 2
ORI_DOWN_RIGHT = 3


def generate_input() -> np.ndarray:
    """Create one synthetic ARC-like input for this puzzle family.

    The generator builds a black canvas with one grey horizontal/vertical divider,
    four quadrant L-shapes with different solid foreground colors, and a 2x2 palette.

    Palette coding uses yellow and magenta on opposite diagonals; the diagonal choice
    controls which L-orientation pair receives yellow in the expected output.

    Randomness is limited to orientation/color/positions while keeping all constraints
    valid and the board size within 30x30.

    Returns
    -------
    np.ndarray
        Integer grid representing the puzzle input.

    Notes
    -----
    Foreground L colors are drawn from red/blue/green/orange indices, never using
    black/gray/yellow/magenta so the solver can isolate patterns reliably.
    """
    rng = random.Random()

    left_h = rng.randint(8, 12)
    left_w = rng.randint(8, 12)

    h_line = rng.randint(3, left_h - 4)
    v_line = rng.randint(3, left_w - 4)

    gap = rng.randint(1, 2)
    palette_col = left_w + gap
    palette_row = rng.randint(0, max(0, left_h - 3))

    w = left_w + gap + 2
    h = left_h

    grid = np.zeros((h, w), dtype=int)

    # Divider only over the left game area; palette sits to the right of it.
    grid[h_line, :left_w] = GRAY
    grid[:left_h, v_line] = GRAY

    # Palette diagonal coding.
    palette_main_diag = bool(rng.getrandbits(1))
    if palette_main_diag:
        grid[palette_row, palette_col] = YELLOW
        grid[palette_row + 1, palette_col + 1] = YELLOW
        grid[palette_row, palette_col + 1] = MAGENTA
        grid[palette_row + 1, palette_col] = MAGENTA
    else:
        grid[palette_row, palette_col + 1] = YELLOW
        grid[palette_row + 1, palette_col] = YELLOW
        grid[palette_row, palette_col] = MAGENTA
        grid[palette_row + 1, palette_col + 1] = MAGENTA

    # Orientation choices: pairs of opposite diagonals. Exactly two of each pair are used.
    pair_ulrd = bool(rng.getrandbits(1))

    colors = [1, 2, 3, 7]
    rng.shuffle(colors)
    quadrants = [
        ("TL", 0, h_line - 1, 0, v_line - 1),
        ("TR", 0, h_line - 1, v_line + 1, left_w - 1),
        ("BL", h_line + 1, left_h - 1, 0, v_line - 1),
        ("BR", h_line + 1, left_h - 1, v_line + 1, left_w - 1),
    ]
    rng.shuffle(quadrants)

    orientation_by_quadrant: Dict[str, int] = {}

    if pair_ulrd:
        # TL/BR: up-left, down-right; TR/BL: up-right, down-left
        tl_or = rng.choice([ORI_UP_LEFT, ORI_DOWN_RIGHT])
        br_or = ORI_DOWN_RIGHT if tl_or == ORI_UP_LEFT else ORI_UP_LEFT
        tr_or = rng.choice([ORI_UP_RIGHT, ORI_DOWN_LEFT])
        bl_or = ORI_DOWN_LEFT if tr_or == ORI_UP_RIGHT else ORI_UP_RIGHT
        orientation_by_quadrant.update({"TL": tl_or, "BR": br_or, "TR": tr_or, "BL": bl_or})
    else:
        # TL/BR: up-right, down-left; TR/BL: up-left, down-right
        tl_or = rng.choice([ORI_UP_RIGHT, ORI_DOWN_LEFT])
        br_or = ORI_DOWN_LEFT if tl_or == ORI_UP_RIGHT else ORI_UP_RIGHT
        tr_or = rng.choice([ORI_UP_LEFT, ORI_DOWN_RIGHT])
        bl_or = ORI_DOWN_RIGHT if tr_or == ORI_UP_LEFT else ORI_UP_LEFT
        orientation_by_quadrant.update({"TL": tl_or, "BR": br_or, "TR": tr_or, "BL": bl_or})

    # Build each L-shape in its quadrant.
    for (name, color) in zip([q[0] for q in quadrants], colors):
        for q_name, r0, r1, c0, c1 in [
            ("TL", 0, h_line - 1, 0, v_line - 1),
            ("TR", 0, h_line - 1, v_line + 1, left_w - 1),
            ("BL", h_line + 1, left_h - 1, 0, v_line - 1),
            ("BR", h_line + 1, left_h - 1, v_line + 1, left_w - 1),
        ]:
            if q_name != name:
                continue
            orientation = orientation_by_quadrant[q_name]
            coords = _random_l_coords_in_region(rng, orientation, r0, r1, c0, c1)
            for rr, cc in coords:
                grid[rr, cc] = color
            break

    return grid


def _random_l_coords_in_region(
    rng: random.Random,
    orientation: int,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
) -> List[Tuple[int, int]]:
    """Random valid placement for one L-shape with requested corner-orientation."""
    if orientation == ORI_UP_LEFT:
        # corner at top-left of its local 3x2 bbox
        base_r = rng.randint(r0, r1 - 2)
        base_c = rng.randint(c0, c1 - 1)
        rel = [(0, 0), (1, 0), (2, 0), (0, 1)]
    elif orientation == ORI_UP_RIGHT:
        # corner at top-right of its local 3x2 bbox
        base_r = rng.randint(r0, r1 - 2)
        base_c = rng.randint(c0 + 1, c1)
        rel = [(0, 0), (1, 0), (2, 0), (0, -1)]
    elif orientation == ORI_DOWN_LEFT:
        # corner at bottom-left of its local 3x2 bbox
        base_r = rng.randint(r0 + 2, r1)
        base_c = rng.randint(c0, c1 - 1)
        rel = [(0, 0), (-1, 0), (-2, 0), (0, 1)]
    else:  # ORI_DOWN_RIGHT
        # corner at bottom-right of its local 3x2 bbox
        base_r = rng.randint(r0 + 2, r1)
        base_c = rng.randint(c0 + 1, c1)
        rel = [(0, 0), (-1, 0), (-2, 0), (0, -1)]

    return [(base_r + dr, base_c + dc) for dr, dc in rel]


def _find_divider_lines(grid: np.ndarray) -> Tuple[int, int]:
    """Find the grey row and column separators (cross)."""
    gray = np.argwhere(grid == GRAY)
    if gray.size == 0:
        raise ValueError("No grey divider found")
    rows, cols = gray[:, 0], gray[:, 1]
    row_counts = np.bincount(rows, minlength=grid.shape[0])
    col_counts = np.bincount(cols, minlength=grid.shape[1])
    row_line = int(np.argmax(row_counts))
    col_line = int(np.argmax(col_counts))
    return row_line, col_line


def _find_palette_orientation(grid: np.ndarray) -> str | None:
    """Return 'main' if yellow is on tl-br diagonal, 'anti' if tr-bl diagonal."""
    h, w = grid.shape
    for r in range(h - 1):
        for c in range(w - 1):
            block = grid[r : r + 2, c : c + 2]
            if block.shape != (2, 2):
                continue
            vals = block.flatten()
            if set(np.unique(vals).tolist()) != {4, 6}:
                continue
            if np.count_nonzero(vals == 4) != 2:
                continue
            if block[0, 0] == YELLOW and block[1, 1] == YELLOW:
                return "main"
            if block[0, 1] == YELLOW and block[1, 0] == YELLOW:
                return "anti"
    return None


def _color_components(grid: np.ndarray) -> List[Tuple[List[Tuple[int, int]], int]]:
    """Return connected 4-neigh components for all non-bg/non-gray colors."""
    h, w = grid.shape
    comps: List[Tuple[List[Tuple[int, int]], int]] = []
    excluded = {BLACK, GRAY, YELLOW, MAGENTA}
    vis = np.zeros_like(grid, dtype=bool)
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in excluded or vis[r, c]:
                continue
            q = deque([(r, c)])
            vis[r, c] = True
            cells: List[Tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not vis[nr, nc] and int(grid[nr, nc]) == color:
                        vis[nr, nc] = True
                        q.append((nr, nc))
            comps.append((cells, color))
    # Keep only likely L-shapes (size 4).
    return [comp for comp in comps if len(comp[0]) == 4]


def _orientation_of_component(cells: List[Tuple[int, int]]) -> int:
    """Infer one of four corner-orientations from an L connected tetromino."""
    s = set(cells)
    degrees: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for r, c in cells:
        nbrs = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (r + dr, c + dc) in s:
                nbrs.append((dr, dc))
        degrees[(r, c)] = nbrs

    corner = None
    for p, nbrs in degrees.items():
        if len(nbrs) == 2:
            (dr1, dc1), (dr2, dc2) = nbrs
            if abs(dr1) != abs(dr2) and abs(dc1) != abs(dc2):
                corner = p
                break

    if corner is None:
        # Fallback: choose the bbox corner occupied by the component.
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        corner = (min(rs), min(cs))

    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r_min, r_max = min(rs), max(rs)
    c_min, c_max = min(cs), max(cs)
    r, c = corner

    if r == r_min and c == c_min:
        return ORI_UP_LEFT
    if r == r_min and c == c_max:
        return ORI_UP_RIGHT
    if r == r_max and c == c_min:
        return ORI_DOWN_LEFT
    return ORI_DOWN_RIGHT


def _quadrant_of_cells(cells: List[Tuple[int, int]], row_line: int, col_line: int) -> str:
    """Classify a component into one of TL/TR/BL/BR by its centroid."""
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r = sum(rs) / len(rs)
    c = sum(cs) / len(cs)
    if r < row_line and c < col_line:
        return "TL"
    if r < row_line and c > col_line:
        return "TR"
    if r > row_line and c < col_line:
        return "BL"
    return "BR"


def main(grid: np.ndarray) -> np.ndarray:
    """Solve the grouping-by-orientation-and-palette task.

    Parameters
    ----------
    grid : np.ndarray
        Integer ARC grid where black=0, gray divider lines=5, palette=4/6,
        and each quadrant has one L tetromino.

    Process
    -------
    1) detect grey cross lines and palette diagonal markers.
    2) extract each quadrant L-shape, compute its orientation from corner location.
    3) map the two opposite-orientation groups to yellow or magenta.
    4) paint one-pixel borders on each separator-adjacent quadrant edge.

    Returns
    -------
    np.ndarray
        Grid with borders added, keeping lines and shapes unchanged.
    """
    out = np.array(grid, copy=True)
    row_line, col_line = _find_divider_lines(grid)

    palette = _find_palette_orientation(grid)
    if palette == "main":
        color_by_orient = {
            ORI_UP_LEFT: YELLOW,
            ORI_DOWN_RIGHT: YELLOW,
            ORI_UP_RIGHT: MAGENTA,
            ORI_DOWN_LEFT: MAGENTA,
        }
    elif palette == "anti":
        color_by_orient = {
            ORI_UP_LEFT: MAGENTA,
            ORI_DOWN_RIGHT: MAGENTA,
            ORI_UP_RIGHT: YELLOW,
            ORI_DOWN_LEFT: YELLOW,
        }
    else:
        color_by_orient = {
            ORI_UP_LEFT: YELLOW,
            ORI_DOWN_RIGHT: YELLOW,
            ORI_UP_RIGHT: MAGENTA,
            ORI_DOWN_LEFT: MAGENTA,
        }

    comps = _color_components(grid)
    by_orient: Dict[str, int] = {}
    for cells, _color in comps:
        orient = _orientation_of_component(cells)
        q = _quadrant_of_cells(cells, row_line, col_line)
        by_orient[q] = orient

    h, w = grid.shape

    # Left palette-side: color only up to palette start to avoid painting palette cells.
    right_limit = w
    for r in range(h - 1):
        for c in range(w - 1):
            blk = out[r : r + 2, c : c + 2]
            if set(np.unique(blk).tolist()) == {YELLOW, MAGENTA} and np.count_nonzero(blk == YELLOW) == 2:
                if blk[0, 0] in (YELLOW, MAGENTA) and blk[0, 1] in (YELLOW, MAGENTA):
                    right_limit = c
                    break
        if right_limit != w:
            break

    # TL
    tl_cols = range(0, col_line)
    tl_rows = range(0, row_line)
    if "TL" in by_orient:
        cmark = color_by_orient[by_orient["TL"]]
        out[row_line - 1, tl_cols] = cmark
        for rr in tl_rows:
            out[rr, col_line - 1] = cmark

    # TR
    if "TR" in by_orient:
        cmark = color_by_orient[by_orient["TR"]]
        start = col_line + 1
        end = max(col_line + 1, right_limit)
        out[row_line - 1, start:end] = cmark
        for rr in range(0, row_line):
            out[rr, col_line + 1] = cmark

    # BL
    if "BL" in by_orient:
        cmark = color_by_orient[by_orient["BL"]]
        out[row_line + 1, 0:col_line] = cmark
        for rr in range(row_line + 1, h):
            out[rr, col_line - 1] = cmark

    # BR
    if "BR" in by_orient:
        cmark = color_by_orient[by_orient["BR"]]
        start = col_line + 1
        end = max(col_line + 1, right_limit)
        out[row_line + 1, start:end] = cmark
        for rr in range(row_line + 1, h):
            out[rr, col_line + 1] = cmark

    return out


if __name__ == "__main__":
    arr = generate_input()
    print(arr.shape)
    print(main(arr))
