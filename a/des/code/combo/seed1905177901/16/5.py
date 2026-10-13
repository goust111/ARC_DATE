import numpy as np


BG = 0
BLUE = 1
GREY = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 300
PANEL_RETRIES = 80
COMPONENT_RETRIES = 60


def _count_components(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                cr, cc = stack.pop()
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return count


def _group_consecutive(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    groups: list[tuple[int, int]] = []
    start = indices[0]
    prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        groups.append((start, prev))
        start = idx
        prev = idx
    groups.append((start, prev))
    return groups


def _regions_from_separator_groups(length: int, groups: list[tuple[int, int]]) -> list[tuple[int, int]]:
    bounds = [(-1, -1)] + groups + [(length, length)]
    regions: list[tuple[int, int]] = []
    for (_, prev_end), (next_start, _) in zip(bounds[:-1], bounds[1:]):
        start = prev_end + 1
        end = next_start
        if start < end:
            regions.append((start, end))
    return regions


def _adjacent_region_indices(group: tuple[int, int], regions: list[tuple[int, int]]) -> list[int]:
    start, end = group
    adjacent: list[int] = []
    for idx, (r0, r1) in enumerate(regions):
        if r1 == start or r0 == end + 1:
            adjacent.append(idx)
    return adjacent


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor panel separator borders using the number of blue 4-connected components inside each panel.

    The bottom row is a 5-color lookup table for component counts 1..5.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    if h == 0 or w == 0:
        return g.copy()

    mapping = [int(v) for v in g[-1] if int(v) != BG]
    if len(mapping) < 5:
        return g.copy()
    mapping = mapping[:5]

    board = g[:-1].copy()
    if board.size == 0:
        out = np.full_like(g, BG)
        return out

    row_groups = _group_consecutive([r for r in range(board.shape[0]) if np.all(board[r] == GREY)])
    col_groups = _group_consecutive([c for c in range(board.shape[1]) if np.all(board[:, c] == GREY)])
    row_regions = _regions_from_separator_groups(board.shape[0], row_groups)
    col_regions = _regions_from_separator_groups(board.shape[1], col_groups)

    if not row_regions or not col_regions:
        out = g.copy()
        out[-1, :] = BG
        return out

    panel_counts = np.zeros((len(row_regions), len(col_regions)), dtype=int)
    for pr, (r0, r1) in enumerate(row_regions):
        for pc, (c0, c1) in enumerate(col_regions):
            panel = board[r0:r1, c0:c1]
            panel_counts[pr, pc] = _count_components(panel == BLUE)

    out_board = board.copy()

    for row_group in row_groups:
        adjacent_rows = _adjacent_region_indices(row_group, row_regions)
        if not adjacent_rows:
            continue
        rs, re = row_group
        for pc, (c0, c1) in enumerate(col_regions):
            k = int(max(panel_counts[pr, pc] for pr in adjacent_rows))
            out_board[rs : re + 1, c0:c1] = mapping[k - 1]

    for col_group in col_groups:
        adjacent_cols = _adjacent_region_indices(col_group, col_regions)
        if not adjacent_cols:
            continue
        cs, ce = col_group
        for pr, (r0, r1) in enumerate(row_regions):
            k = int(max(panel_counts[pr, pc] for pc in adjacent_cols))
            out_board[r0:r1, cs : ce + 1] = mapping[k - 1]

    for row_group in row_groups:
        adjacent_rows = _adjacent_region_indices(row_group, row_regions)
        if not adjacent_rows:
            continue
        rs, re = row_group
        for col_group in col_groups:
            adjacent_cols = _adjacent_region_indices(col_group, col_regions)
            if not adjacent_cols:
                continue
            cs, ce = col_group
            k = int(max(panel_counts[pr, pc] for pr in adjacent_rows for pc in adjacent_cols))
            out_board[rs : re + 1, cs : ce + 1] = mapping[k - 1]

    out = g.copy()
    out[:-1] = out_board
    out[-1, :] = BG
    return out


def _is_too_close(cell: tuple[int, int], occupied: set[tuple[int, int]]) -> bool:
    r, c = cell
    for rr, cc in occupied:
        if max(abs(r - rr), abs(c - cc)) <= 1:
            return True
    return False


def _sample_component(
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
    count_hint: int,
    rng: np.random.Generator,
) -> set[tuple[int, int]] | None:
    starts = [(r, c) for r in range(h) for c in range(w) if (r, c) not in occupied and not _is_too_close((r, c), occupied)]
    if not starts:
        return None

    if count_hint >= 5:
        max_cells = 1
    elif count_hint == 4:
        max_cells = 2
    elif count_hint == 3:
        max_cells = 3
    else:
        max_cells = 4
    target_size = int(rng.integers(1, max_cells + 1))

    start = starts[int(rng.integers(0, len(starts)))]
    component = {start}
    while len(component) < target_size:
        frontier: list[tuple[int, int]] = []
        seen = set()
        for r, c in component:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                nxt = (nr, nc)
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if nxt in component or nxt in occupied or nxt in seen:
                    continue
                if _is_too_close(nxt, occupied):
                    continue
                frontier.append(nxt)
                seen.add(nxt)
        if not frontier:
            break
        component.add(frontier[int(rng.integers(0, len(frontier)))])
    return component


def _random_panel_mask(h: int, w: int, n_components: int, rng: np.random.Generator) -> np.ndarray | None:
    for _ in range(PANEL_RETRIES):
        occupied: set[tuple[int, int]] = set()
        mask = np.zeros((h, w), dtype=bool)
        ok = True
        for _ in range(n_components):
            component = None
            for _ in range(COMPONENT_RETRIES):
                component = _sample_component(h, w, occupied, n_components, rng)
                if component is not None:
                    break
            if component is None:
                ok = False
                break
            for r, c in component:
                mask[r, c] = True
            occupied.update(component)
        if ok and _count_components(mask) == n_components:
            return mask
    return None


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    panel_rows = int(rng.integers(2, 5))
    panel_cols = int(rng.integers(2, 5))
    panel_h = int(rng.integers(5, 9))
    panel_w = int(rng.integers(5, 9))

    board_h = panel_rows * panel_h + panel_rows + 1
    board_w = panel_cols * panel_w + panel_cols + 1
    grid = np.full((board_h + 1, board_w), BG, dtype=int)
    board = grid[:-1]

    row_lines = [i * (panel_h + 1) for i in range(panel_rows + 1)]
    col_lines = [j * (panel_w + 1) for j in range(panel_cols + 1)]
    board[row_lines, :] = GREY
    board[:, col_lines] = GREY

    component_counts = np.zeros((panel_rows, panel_cols), dtype=int)
    for pr in range(panel_rows):
        for pc in range(panel_cols):
            count = int(rng.integers(1, 6))
            mask = _random_panel_mask(panel_h, panel_w, count, rng)
            if mask is None:
                return None
            r0 = row_lines[pr] + 1
            c0 = col_lines[pc] + 1
            board[r0 : r0 + panel_h, c0 : c0 + panel_w][mask] = BLUE
            component_counts[pr, pc] = count

    distinct_counts = {int(v) for v in component_counts.flat}
    has_adjacent_diff = False
    for pr in range(panel_rows):
        for pc in range(panel_cols):
            if pr + 1 < panel_rows and component_counts[pr, pc] != component_counts[pr + 1, pc]:
                has_adjacent_diff = True
            if pc + 1 < panel_cols and component_counts[pr, pc] != component_counts[pr, pc + 1]:
                has_adjacent_diff = True
    if len(distinct_counts) < min(3, panel_rows * panel_cols) or not has_adjacent_diff:
        return None

    palette = np.array([2, 3, 4, 6, 7, 8, 9], dtype=int)
    mapping = rng.choice(palette, size=5, replace=False)
    start = int(rng.integers(0, board_w - 5 + 1))
    grid[-1, start : start + 5] = mapping
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
