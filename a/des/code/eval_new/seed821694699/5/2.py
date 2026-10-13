import numpy as np
from collections import defaultdict

BLACK = 0
GREY = 5
AZURE = 8


def _collect_component(grid: np.ndarray, start: tuple[int, int], visited: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Collect all non-black cells connected to `start` by 4-neighborhood.

    Parameters
    ----------
    grid:
        Integer ARC grid.
    start:
        Seed coordinate inside a channel component.
    visited:
        Re-usable visitation mask, updated in-place.

    Returns
    -------
    component:
        List of `(row, col)` coordinates in the connected component.
    azure_cells:
        List of azure coordinates found inside the component.

    Behavior
    -------
    The flood fill excludes black cells and supports any corridor color channel shape.
    It returns coordinates ordered by pop sequence, which is stable for deterministic use only.
    """

    h, w = grid.shape
    stack = [start]
    component: list[tuple[int, int]] = []
    azure_cells: list[tuple[int, int]] = []

    while stack:
        r, c = stack.pop()
        if visited[r, c] or grid[r, c] == BLACK:
            continue
        visited[r, c] = True
        component.append((r, c))
        if grid[r, c] == AZURE:
            azure_cells.append((r, c))

        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and grid[nr, nc] != BLACK:
                stack.append((nr, nc))

    return component, azure_cells


def main(grid: np.ndarray) -> np.ndarray:
    """Transform each gray corridor into ordered color bands.

    Parameters
    ----------
    grid:
        Input ARC board, encoded as `np.ndarray` of integers.

    Returns
    -------
    np.ndarray
        The solved board, same shape as input.

    规则执行
    -------
    For each connected non-black channel: locate the unique AZURE marker (start), detect
    horizontal/vertical orientation, group seeds by color, sort color groups by first
    seed distance from the marker, then paint contiguous bands of width equal to channel
    width and lengths equal to seed counts.

    边界与异常情况
    -------------
    Components that are not clearly 1-cell-thick-or-thickened straight corridors are ignored and
    preserved as-is.
    """

    out = np.array(grid, dtype=int, copy=True)
    h, w = grid.shape
    visited = np.zeros_like(grid, dtype=bool)

    for r0 in range(h):
        for c0 in range(w):
            if grid[r0, c0] == BLACK or visited[r0, c0]:
                continue

            component, azure_cells = _collect_component(grid, (r0, c0), visited)
            if not component or len(azure_cells) == 0:
                continue

            if len(azure_cells) != 1:
                # Ambiguous marker setup; keep original component unchanged.
                continue

            rows = [r for r, _ in component]
            cols = [c for _, c in component]
            rmin, rmax = min(rows), max(rows)
            cmin, cmax = min(cols), max(cols)
            row_span = rmax - rmin + 1
            col_span = cmax - cmin + 1

            if row_span <= 3 and col_span > 3:
                orient = "H"
            elif col_span <= 3 and row_span > 3:
                orient = "V"
            elif max(row_span, col_span) <= 3:
                # tiny component with no clear direction; skip for safety.
                continue
            elif row_span >= col_span:
                orient = "V"
            else:
                orient = "H"

            axis_map: dict[int, list[tuple[int, int]]] = defaultdict(list)
            for p in component:
                if orient == "H":
                    axis_map[p[1]].append(p)
                else:
                    axis_map[p[0]].append(p)

            axis_positions = sorted(axis_map)
            if not axis_positions:
                continue

            # reset whole component to corridor gray first
            for rr, cc in component:
                out[rr, cc] = GREY

            ar, ac = azure_cells[0]
            a_axis = ac if orient == "H" else ar

            if abs(a_axis - axis_positions[0]) <= abs(a_axis - axis_positions[-1]):
                ordered_axis = axis_positions
            else:
                ordered_axis = list(reversed(axis_positions))

            direction = 1 if ordered_axis[0] == axis_positions[0] else -1

            # group seed counts and first-seed distance from azure marker
            color_info: dict[int, tuple[int, int]] = {}
            for rr, cc in component:
                color = int(grid[rr, cc])
                if color in (BLACK, GREY, AZURE):
                    continue
                axis = cc if orient == "H" else rr
                dist = (axis - a_axis) * direction
                if dist <= 0:
                    dist = abs(axis - a_axis)

                cnt, first = color_info.get(color, (0, 10 ** 9))
                color_info[color] = (cnt + 1, first if first <= dist else dist)

            if not color_info:
                continue

            groups = sorted(color_info.items(), key=lambda kv: (kv[1][1], kv[0]))
            axis_i = ordered_axis.index(a_axis) + 1 if a_axis in ordered_axis else 0
            for color, (need, _) in groups:
                for _ in range(need):
                    if axis_i >= len(ordered_axis):
                        break
                    pos = ordered_axis[axis_i]
                    axis_i += 1
                    for rr, cc in axis_map[pos]:
                        out[rr, cc] = color

    return out


def generate_input() -> np.ndarray:
    """Generate one random ARC-style input grid for this puzzle.

    Returns
    -------
    np.ndarray
        Grid size within 30x30, with 1-3 pixel wide horizontal/vertical gray corridors,
        one Azure endpoint marker, and random colored seeds inside each corridor.

    Notes
    -----
    Generator uses NumPy RNG directly and composes multiple independent channels with random
        orientation, width, length, seed counts, and color combinations.
    """

    rng = np.random.default_rng()
    height = int(rng.integers(12, 31))
    width = int(rng.integers(12, 31))
    board = np.zeros((height, width), dtype=int)

    palette = np.array([1, 2, 3, 4, 6, 7, 9], dtype=int)
    channel_count = int(rng.integers(1, 5))

    placed_any = False

    def region_free(r1: int, r2: int, c1: int, c2: int) -> bool:
        rr0 = max(0, r1 - 1)
        rr1 = min(height, r2 + 1)
        cc0 = max(0, c1 - 1)
        cc1 = min(width, c2 + 1)
        return np.all(board[rr0:rr1, cc0:cc1] == BLACK)

    for _ in range(channel_count):
        placed = False
        for _ in range(120):
            orient = "H" if rng.random() < 0.5 else "V"
            corridor_width = int(rng.integers(1, 4))

            if orient == "H":
                if height < corridor_width or width < 6:
                    continue
                max_len = max(3, width - 2)
                corridor_len = int(rng.integers(6, max_len + 1))
                row0 = int(rng.integers(0, height - corridor_width + 1))
                col0 = int(rng.integers(0, width - corridor_len + 1))
                row1 = row0 + corridor_width
                col1 = col0 + corridor_len

                if not region_free(row0, row1, col0, col1):
                    continue

                board[row0:row1, col0:col1] = GREY
                board_anchor = (row0, col0) if rng.random() < 0.5 else (row0, col1 - 1)
                anchor_r = int(rng.integers(row0, row1))
                anchor_c = board_anchor[1]
                direction = 1 if anchor_c == col0 else -1
                available = [(r, c) for r in range(row0, row1) for c in range(col0, col1)]
            else:
                if width < corridor_width or height < 6:
                    continue
                max_len = max(3, height - 2)
                corridor_len = int(rng.integers(6, max_len + 1))
                row0 = int(rng.integers(0, height - corridor_len + 1))
                col0 = int(rng.integers(0, width - corridor_width + 1))
                row1 = row0 + corridor_len
                col1 = col0 + corridor_width

                if not region_free(row0, row1, col0, col1):
                    continue

                board[row0:row1, col0:col1] = GREY
                board_anchor = (row0, col0) if rng.random() < 0.5 else (row1 - 1, col0)
                anchor_r = board_anchor[0]
                anchor_c = int(rng.integers(col0, col1))
                direction = 1 if anchor_r == row0 else -1
                available = [(r, c) for r in range(row0, row1) for c in range(col0, col1)]

            board[anchor_r, anchor_c] = AZURE
            available = [(rr, cc) for rr, cc in available if (rr, cc) != (anchor_r, anchor_c)]
            if not available:
                board[anchor_r, anchor_c] = GREY
                continue

            max_seeds = max(1, len(available))
            total_seeds = int(rng.integers(1, min(max_seeds, 12) + 1))
            seed_positions = rng.choice(len(available), size=total_seeds, replace=False)

            num_colors = int(rng.integers(1, min(4, total_seeds) + 1))
            colors = rng.choice(palette, size=num_colors, replace=False)
            if num_colors == 1:
                counts = [total_seeds]
            else:
                cuts = np.sort(rng.choice(total_seeds - 1, size=num_colors - 1, replace=False) + 1)
                counts = np.diff(np.concatenate(([0], cuts, [total_seeds]))).astype(int).tolist()

            ptr = 0
            for cnt, color in zip(counts, colors):
                for _ in range(cnt):
                    rr, cc = available[int(seed_positions[ptr])]
                    board[rr, cc] = int(color)
                    ptr += 1

            placed = True
            placed_any = True
            break

        if not placed:
            # skip without failing; remaining channels may still fit later
            continue

    if not placed_any:
        # deterministic fallback that always works
        h2, w2 = height // 2, width // 2
        r0 = max(0, h2 - 1)
        c0 = max(0, w2 - 6)
        board[r0 : r0 + 1, c0 : c0 + 6] = GREY
        board[r0, c0] = AZURE
        board[r0, c0 + 2] = 1
        board[r0, c0 + 4] = 2

    return board


if __name__ == "__main__":
    sample = generate_input()
    solved = main(sample)
    assert isinstance(sample, np.ndarray)
    assert isinstance(solved, np.ndarray)
    assert sample.shape == solved.shape

    tiny = np.array(
        [[0, 0, 0, 0, 0, 0, 0], [0, 8, 5, 1, 5, 2, 5], [0, 0, 0, 0, 0, 0, 0]],
        dtype=int,
    )
    tiny_out = main(tiny)
    expected = np.array(
        [[0, 0, 0, 0, 0, 0, 0], [0, 5, 1, 2, 5, 5, 5], [0, 0, 0, 0, 0, 0, 0]],
        dtype=int,
    )
    assert np.array_equal(tiny_out, expected)
    print("self-check ok", sample.shape, tiny_out.tolist())
