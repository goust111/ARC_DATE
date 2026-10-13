import numpy as np

BLACK = 0
GREY = 5
MAROON = 9
AZURE = 8
YELLOW = 4
MAGENTA = 6

DIRECTION_BY_LEGEND = {
    MAROON: (-1, 0),  # up
    AZURE: (0, 1),    # right
    YELLOW: (1, 0),   # down
    MAGENTA: (0, -1), # left
}

SEED_COLORS = (1, 2, 3, 7)


def main(grid: np.ndarray) -> np.ndarray:
    """
    1) Purpose: Draw seed growth lines inside each grey-bounded closed region.
    2) Input: Integer ARC-style NumPy 2D grid with black background, grey fences, and a legend row at the bottom.
    3) Output: Grid with each seed extended in its decoded direction up to the enclosed empty-cell count.
    4) Method: Decode legend pairs to direction, find non-grey enclosed components, count interior black cells, then paint along seed direction.
    """
    output = grid.copy()
    if output.ndim != 2 or output.size == 0:
        return output

    if output.shape[0] < 2:
        return output

    legend = _decode_legend(output[-1])
    play_area = output[:-1]
    h, w = play_area.shape

    visited = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if visited[r, c] or play_area[r, c] == GREY:
                continue

            q = [(r, c)]
            visited[r, c] = True
            component = []
            touches_border = False
            black_cells = 0
            seed_cells = []

            while q:
                cr, cc = q.pop()
                component.append((cr, cc))

                value = int(play_area[cr, cc])
                if value == BLACK:
                    black_cells += 1
                elif value != BLACK:
                    seed_cells.append((cr, cc, value))

                if cr == 0 or cc == 0 or cr == h - 1 or cc == w - 1:
                    touches_border = True

                nr = cr - 1
                if 0 <= nr < h and not visited[nr, cc] and play_area[nr, cc] != GREY:
                    visited[nr, cc] = True
                    q.append((nr, cc))
                nr = cr + 1
                if 0 <= nr < h and not visited[nr, cc] and play_area[nr, cc] != GREY:
                    visited[nr, cc] = True
                    q.append((nr, cc))
                nc = cc - 1
                if 0 <= nc < w and not visited[cr, nc] and play_area[cr, nc] != GREY:
                    visited[cr, nc] = True
                    q.append((cr, nc))
                nc = cc + 1
                if 0 <= nc < w and not visited[cr, nc] and play_area[cr, nc] != GREY:
                    visited[cr, nc] = True
                    q.append((cr, nc))

            if touches_border or not seed_cells:
                continue

            for sr, sc, seed_color in seed_cells:
                if seed_color not in legend:
                    continue

                dr, dc = legend[seed_color]
                line_len = black_cells
                rr, cc = sr, sc

                for _ in range(line_len):
                    rr += dr
                    cc += dc

                    if rr < 0 or rr >= h or cc < 0 or cc >= w:
                        break
                    if play_area[rr, cc] == GREY or output[rr, cc] == GREY:
                        break
                    if output[rr, cc] != BLACK:
                        break

                    output[rr, cc] = seed_color

    return output


def _decode_legend(bottom_row: np.ndarray) -> dict[int, tuple[int, int]]:
    legend = {}
    c = 0
    w = bottom_row.shape[0]

    while c < w:
        if c + 1 < w and bottom_row[c] != BLACK and bottom_row[c + 1] != BLACK:
            seed_color = int(bottom_row[c])
            marker = int(bottom_row[c + 1])
            direction = DIRECTION_BY_LEGEND.get(marker)
            if direction is not None:
                legend[seed_color] = direction
                c += 2
                continue
        c += 1

    return legend


def generate_input() -> np.ndarray:
    """Generate one random ARC-like input with one-row legend and multiple closed regions."""
    rng = np.random.default_rng()
    h = int(rng.integers(15, 30))
    w = int(rng.integers(16, 30))

    grid = np.full((h, w), BLACK, dtype=int)
    play_h = h - 1

    occupied = np.zeros((h, w), dtype=bool)
    selected_seeds = list(SEED_COLORS)
    rng.shuffle(selected_seeds)

    num_regions = int(rng.integers(1, min(4, len(selected_seeds)) + 1))
    regions_done = 0
    used_seed_colors = []

    for _ in range(num_regions * 8):
        if regions_done >= num_regions:
            break

        max_region_h = max(3, min(play_h - 3, 8))
        max_region_w = max(3, min(w - 3, 8))
        if max_region_h < 3 or max_region_w < 3:
            break

        region_h = int(rng.integers(3, max_region_h + 1))
        region_w = int(rng.integers(3, max_region_w + 1))
        if region_h < 3 or region_w < 3:
            continue

        if play_h - region_h - 1 <= 1 or w - region_w - 1 <= 1:
            continue

        top = int(rng.integers(1, play_h - region_h - 1))
        left = int(rng.integers(1, w - region_w - 1))
        bottom = top + region_h - 1
        right = left + region_w - 1

        pad_top = max(0, top - 1)
        pad_bottom = min(play_h, bottom + 2)
        pad_left = max(0, left - 1)
        pad_right = min(w, right + 2)
        if np.any(occupied[pad_top:pad_bottom, pad_left:pad_right]):
            continue

        # Draw closed rectangular fence
        grid[top, left:right + 1] = GREY
        grid[bottom, left:right + 1] = GREY
        grid[top:bottom + 1, left] = GREY
        grid[top:bottom + 1, right] = GREY

        occupied[top:bottom + 1, left:right + 1] = True

        seed_r = int(rng.integers(top + 1, bottom))
        seed_c = int(rng.integers(left + 1, right))
        seed_color = selected_seeds[regions_done]
        grid[seed_r, seed_c] = seed_color

        used_seed_colors.append(
            (
                seed_color,
                int(rng.choice(list(DIRECTION_BY_LEGEND.keys()))),
            )
        )
        regions_done += 1

    # Fill legend row with 2x1 (seed, direction) pairs.
    c = 0
    for seed_color, direction_color in used_seed_colors:
        if c + 1 >= w:
            break
        grid[play_h, c] = seed_color
        grid[play_h, c + 1] = direction_color
        c += 2

    return grid
