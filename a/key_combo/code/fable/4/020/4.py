import sys
from pathlib import Path

import numpy as np

# Use the shared ARC utility module required by the task prompt.
# Import must work when this file is loaded with SourceFileLoader directly.
_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_PATH) not in sys.path:
    sys.path.append(str(_UTILS_PATH))

from arc_utils import get_bg, find_components


_MARKER_SIDES = (5, 8)
_MIN_SQUARES = 1
_MAX_SQUARES = 4
_MIN_GRID = 12
_MAX_GRID = 30
_SQUARE_PADDING = 2
_GENERATE_ATTEMPTS = 400


def _square_anchors(r0: int, c0: int, r1: int, c1: int):
    """Yield (corner_r, corner_c, dr, dc) for the 4 outward diagonal rays."""
    yield r0, c0, -1, -1  # top-left
    yield r0, c1, -1, 1   # top-right
    yield r1, c0, 1, -1   # bottom-left
    yield r1, c1, 1, 1    # bottom-right


def _first_non_bg_on_ray(grid: np.ndarray, bg: int, start_r: int, start_c: int, dr: int, dc: int):
    """Return first non-background coordinate and color on a 45-degree ray."""
    H, W = grid.shape
    r, c = start_r + dr, start_c + dc
    while 0 <= r < H and 0 <= c < W:
        col = int(grid[r, c])
        if col != bg:
            return (int(r), int(c)), col
        r += dr
        c += dc
    return None


def _locate_pairings(grid: np.ndarray):
    """Pair every solid square with its unique marker by diagonal ray geometry."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)

    components = find_components(g, bg=bg, connectivity=4, monochromatic=False, min_area=1)
    assert components, "expected non-background structures"

    squares = []
    marker_colors: dict[tuple[int, int], int] = {}

    for component in components:
        coords = component["coords"]
        r0, c0, r1, c1 = component["bbox"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        area = int(component["area"])
        if area == 1:
            r, c = map(int, coords[0])
            marker_colors[(r, c)] = int(g[r, c])
            continue

        if h == w and h >= _MARKER_SIDES[0] and area == h * w:
            squares.append(component)
            continue

        raise AssertionError("input must contain only solid squares and singleton markers")

    if len(squares) != len(marker_colors):
        raise AssertionError("each solid square must match exactly one singleton marker")
    if not squares:
        raise AssertionError("expected at least one square")

    pairings = []
    used_markers: set[tuple[int, int]] = set()
    for square in squares:
        r0, c0, r1, c1 = map(int, square["bbox"])
        matches = []
        for anchor_r, anchor_c, dr, dc in _square_anchors(r0, c0, r1, c1):
            hit = _first_non_bg_on_ray(g, bg, anchor_r, anchor_c, dr, dc)
            if hit is None:
                continue
            marker_pos, marker_color = hit
            if marker_pos in marker_colors:
                matches.append((marker_pos, marker_color, (anchor_r, anchor_c)))

        if len(matches) != 1:
            raise AssertionError("each square must see exactly one marker on its outgoing corner ray")

        marker_pos, marker_color, anchor = matches[0]
        if marker_pos in used_markers:
            raise AssertionError("markers must map to exactly one square")
        used_markers.add(marker_pos)

        pairings.append(
            (
                int(r0),
                int(c0),
                int(r1),
                int(c1),
                marker_pos,
                marker_color,
                anchor,
            )
        )

    if len(used_markers) != len(marker_colors):
        raise AssertionError("all markers must be used")

    return pairings, bg


def main(grid: np.ndarray) -> np.ndarray:
    """Carve each solid square with checkerboard parity anchored at its marked corner.

    Input structure:
    - Background is the most common color in the grid.
    - Foreground is a set of non-overlapping solid monochrome squares (at least 5x5).
    - A singleton marker pixel outside each square lies on the outward 45° ray of exactly one square corner.
    - Number of squares equals number of markers.

    Rule:
    - Keep all non-square pixels and all square borders unchanged.
    - For each square, find its unique marker by scanning its four corner outward diagonal rays and choosing the corner whose ray hits that marker first.
    - Inside that square, color interior cells whose (row+col) parity matches the anchor corner parity with the marker color; recolor the rest to background.
    - Markers remain at their original positions and colors.

    Key insight:
    - The pairing is fixed by corner-ray geometry and anchor parity, not by distance to the nearest marker.
    """
    g = np.asarray(grid, dtype=int)
    pairings, bg = _locate_pairings(g)

    out = g.copy()
    for r0, c0, r1, c1, _marker_pos, marker_color, anchor in pairings:
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size == 0:
            continue
        rows = np.arange(r0 + 1, r1)
        cols = np.arange(c0 + 1, c1)
        parity_grid = (rows[:, None] + cols[None, :]) & 1
        target_parity = (anchor[0] + anchor[1]) & 1

        interior[parity_grid == target_parity] = marker_color
        interior[parity_grid != target_parity] = bg

    return out


def _can_place_square(occupancy: np.ndarray, r0: int, c0: int, size: int, padding: int) -> bool:
    H, W = occupancy.shape
    r1 = r0 + size
    c1 = c0 + size

    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(H, r1 + padding)
    cc1 = min(W, c1 + padding)

    return not bool(occupancy[rr0:rr1, cc0:cc1].any())


def _diagonal_candidates(grid: np.ndarray, bg: int, corner: tuple[int, int], dr: int, dc: int):
    H, W = grid.shape
    r0, c0 = corner
    positions: list[tuple[int, int]] = []
    r, c = r0 + dr, c0 + dc
    while 0 <= r < H and 0 <= c < W:
        if int(grid[r, c]) == bg:
            positions.append((int(r), int(c)))
            r += dr
            c += dc
            continue
        break
    return positions


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    palette = list(range(10))

    for _ in range(_GENERATE_ATTEMPTS):
        H = int(rng.integers(_MIN_GRID, _MAX_GRID + 1))
        W = int(rng.integers(_MIN_GRID, _MAX_GRID + 1))
        bg = int(rng.choice(palette))
        other_colors = [c for c in palette if c != bg]

        n_squares = int(rng.integers(_MIN_SQUARES, _MAX_SQUARES + 1))
        if len(other_colors) < n_squares:
            continue

        square_colors = rng.choice(other_colors, size=n_squares, replace=False).astype(int).tolist()
        grid = np.full((H, W), bg, dtype=int)
        occupancy = np.zeros((H, W), dtype=bool)
        squares: list[tuple[int, int, int, int, int]] = []  # r0,c0,size,color,anchor_idx

        for color in square_colors:
            max_side = min(_MARKER_SIDES[1], H - 2 * _SQUARE_PADDING, W - 2 * _SQUARE_PADDING)
            if max_side < _MARKER_SIDES[0]:
                break

            placed = False
            for _ in range(400):
                side = int(rng.integers(_MARKER_SIDES[0], max_side + 1))
                r0 = int(rng.integers(_SQUARE_PADDING, H - side - _SQUARE_PADDING + 1))
                c0 = int(rng.integers(_SQUARE_PADDING, W - side - _SQUARE_PADDING + 1))
                if not _can_place_square(occupancy, r0, c0, side, _SQUARE_PADDING):
                    continue

                grid[r0 : r0 + side, c0 : c0 + side] = color
                occupancy[r0 : r0 + side, c0 : c0 + side] = True
                squares.append((r0, c0, side, color, -1))
                placed = True
                break

            if not placed:
                break

        if len(squares) != n_squares:
            continue

        markers: list[tuple[tuple[int, int], int]] = []
        used_marker_cells: set[tuple[int, int]] = set()
        pairings_ok = True

        rng.shuffle(squares)
        for idx, (r0, c0, side, color, _unused) in enumerate(squares):
            r1 = r0 + side - 1
            c1 = c0 + side - 1
            anchor_options: list[tuple[tuple[int, int], tuple[int, int], int]] = []

            for anchor_r, anchor_c, dr, dc in _square_anchors(r0, c0, r1, c1):
                candidates = _diagonal_candidates(grid, bg, (anchor_r, anchor_c), dr, dc)
                if not candidates:
                    continue

                # Keep both short and long rays for diversity.
                if len(candidates) > 4 and rng.random() < 0.35:
                        if rng.random() < 0.5:
                            cand = candidates[int(rng.integers(0, max(2, len(candidates) // 4 + 1)))]
                        else:
                            cand = candidates[int(rng.integers(max(1, len(candidates) * 3 // 4), len(candidates)))]
                else:
                    cand = candidates[int(rng.integers(0, len(candidates)))]

                if grid[cand] != bg or cand in used_marker_cells:
                    continue
                anchor_options.append((cand, (anchor_r, anchor_c), len(candidates)))

            if not anchor_options:
                pairings_ok = False
                break

            cand_pos, anchor, _cands = anchor_options[int(rng.integers(0, len(anchor_options)))]
            if grid[tuple(cand_pos)] != bg or cand_pos in used_marker_cells:
                pairings_ok = False
                break

            marker_palette = [c for c in other_colors if c != bg and c != color]
            if not marker_palette:
                marker_palette = other_colors
            marker_color = int(rng.choice(marker_palette))

            grid[cand_pos] = marker_color
            occupancy[cand_pos] = True
            used_marker_cells.add(cand_pos)
            squares[idx] = (r0, c0, side, color, anchor)
            markers.append((cand_pos, marker_color))

        if not pairings_ok or len(markers) != n_squares:
            continue

        # Validate by solver assumptions and branching rule before accepting.
        try:
            pairings, _ = _locate_pairings(grid)
            if len(pairings) != n_squares:
                continue
            if len(markers) != len(pairings):
                continue
            # Ensure generated sample is visible and rule-satisfying.
            _ = main(grid)
        except Exception:
            continue

        return grid

    raise RuntimeError("failed to generate a valid input for task 4")
