import random
from collections import deque

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

# --------------- shared helpers ---------------


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    banned = exclude or set()
    return [int(color) for color in np.unique(grid) if int(color) not in banned]


def _line_positions(mask: np.ndarray) -> tuple[list[int], list[int]]:
    h_rows = [r for r in range(mask.shape[0]) if np.any(mask[r, :-1] & mask[r, 1:])]
    v_cols = [c for c in range(mask.shape[1]) if np.any(mask[:-1, c] & mask[1:, c])]
    return h_rows, v_cols


def _find_border_color(grid: np.ndarray, bg: int) -> tuple[int, list[int], list[int]]:
    best: tuple[int, int, int, int] | None = None
    best_data: tuple[int, list[int], list[int]] | None = None

    for color in colors_present(grid, exclude={bg}):
        mask = grid == int(color)
        h_rows, v_cols = _line_positions(mask)
        row_gaps = [b - a for a, b in zip(h_rows, h_rows[1:])]
        col_gaps = [b - a for a, b in zip(v_cols, v_cols[1:])]
        grid_like = int(
            len(h_rows) >= 3
            and len(v_cols) >= 3
            and all(gap >= 2 for gap in row_gaps)
            and all(gap >= 2 for gap in col_gaps)
        )
        score = (
            grid_like,
            len(h_rows) * len(v_cols),
            len(h_rows) + len(v_cols),
            int(np.count_nonzero(mask)),
        )
        if best is None or score > best:
            best = score
            best_data = (int(color), h_rows, v_cols)

    assert best_data is not None and best is not None and best[0] == 1
    return best_data


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _cell_gap_count(grid: np.ndarray, bg: int, r0: int, r1: int, c0: int, c1: int) -> int:
    return int(
        np.count_nonzero(grid[r0, c0 + 1 : c1] == bg)
        + np.count_nonzero(grid[r1, c0 + 1 : c1] == bg)
        + np.count_nonzero(grid[r0 + 1 : r1, c0] == bg)
        + np.count_nonzero(grid[r0 + 1 : r1, c1] == bg)
    )


def _cell_components(
    grid: np.ndarray, bg: int, border: int, r0: int, r1: int, c0: int, c1: int
) -> list[list[tuple[int, int]]]:
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    mask = (interior != bg) & (interior != border)
    return _components(mask)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the interior pattern whose area matches the number of one-cell border gaps.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    border, h_lines, v_lines = _find_border_color(g, bg)
    assert len(h_lines) >= 3 and len(v_lines) >= 3

    out = g.copy()
    for hi in range(len(h_lines) - 1):
        r0, r1 = int(h_lines[hi]), int(h_lines[hi + 1])
        if r1 - r0 <= 1:
            continue
        for vi in range(len(v_lines) - 1):
            c0, c1 = int(v_lines[vi]), int(v_lines[vi + 1])
            if c1 - c0 <= 1:
                continue
            target = _cell_gap_count(g, bg, r0, r1, c0, c1)
            for comp in _cell_components(g, bg, border, r0, r1, c0, c1):
                if len(comp) == target:
                    continue
                for rr, cc in comp:
                    out[r0 + 1 + rr, c0 + 1 + cc] = bg
    return out


# --------------- generator ---------------

GEN_ATTEMPTS = 400
GRID_ROWS_RANGE = (2, 3)
GRID_COLS_RANGE = (2, 3)
CELL_HEIGHT_RANGE = (6, 7)
CELL_WIDTH_RANGE = (6, 7)
MARGIN_RANGE = (0, 2)
MAX_GAPS_PER_SEGMENT = 2


def _line_positions_from_sizes(sizes: list[int]) -> list[int]:
    positions = [0]
    cursor = 0
    for size in sizes:
        cursor += int(size) + 1
        positions.append(cursor)
    return positions


def _sample_gap_positions(positions: list[int], k: int) -> list[int] | None:
    if k == 0:
        return []
    for _ in range(40):
        chosen = sorted(random.sample(positions, k))
        if all(b - a > 1 for a, b in zip(chosen, chosen[1:])):
            return chosen
    return None


def _choose_gap_count(segment_len: int) -> int:
    max_gaps = min(MAX_GAPS_PER_SEGMENT, (segment_len + 1) // 2)
    if max_gaps <= 1:
        return random.choices([0, 1], weights=[2, 5], k=1)[0]
    return random.choices([0, 1, 2], weights=[2, 4, 3], k=1)[0]


def _paint_random_gaps(
    patch: np.ndarray, bg: int, h_lines: list[int], v_lines: list[int]
) -> tuple[np.ndarray, np.ndarray] | None:
    nr = len(h_lines) - 1
    nc = len(v_lines) - 1

    for _ in range(80):
        g = patch.copy()
        h_gaps: dict[tuple[int, int], list[int]] = {}
        v_gaps: dict[tuple[int, int], list[int]] = {}

        ok = True
        for hi, r in enumerate(h_lines):
            for cj in range(nc):
                cols = list(range(v_lines[cj] + 1, v_lines[cj + 1]))
                k = _choose_gap_count(len(cols))
                chosen = _sample_gap_positions(cols, k)
                if chosen is None:
                    ok = False
                    break
                h_gaps[(hi, cj)] = chosen
                for c in chosen:
                    g[r, c] = bg
            if not ok:
                break
        if not ok:
            continue

        for ri in range(nr):
            for vj, c in enumerate(v_lines):
                rows = list(range(h_lines[ri] + 1, h_lines[ri + 1]))
                k = _choose_gap_count(len(rows))
                chosen = _sample_gap_positions(rows, k)
                if chosen is None:
                    ok = False
                    break
                v_gaps[(ri, vj)] = chosen
                for r in chosen:
                    g[r, c] = bg
            if not ok:
                break
        if not ok:
            continue

        gap_counts = np.zeros((nr, nc), dtype=int)
        for ri in range(nr):
            for cj in range(nc):
                gap_counts[ri, cj] = (
                    len(h_gaps[(ri, cj)])
                    + len(h_gaps[(ri + 1, cj)])
                    + len(v_gaps[(ri, cj)])
                    + len(v_gaps[(ri, cj + 1)])
                )

        flat = gap_counts.ravel()
        if np.all((1 <= flat) & (flat <= 8)) and (flat.size == 1 or len(set(map(int, flat))) >= 2):
            return g, gap_counts
    return None


def _random_polyomino(size: int, max_h: int, max_w: int) -> np.ndarray | None:
    if size == 1:
        return np.ones((1, 1), dtype=bool)

    for _ in range(120):
        cells = {(0, 0)}
        while len(cells) < size:
            frontier = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    nxt = (r + dr, c + dc)
                    if nxt not in cells:
                        frontier.add(nxt)
            choices = list(frontier)
            random.shuffle(choices)

            placed = False
            for nr, nc in choices:
                test = cells | {(nr, nc)}
                rows = [r for r, _ in test]
                cols = [c for _, c in test]
                if max(rows) - min(rows) + 1 <= max_h and max(cols) - min(cols) + 1 <= max_w:
                    cells.add((nr, nc))
                    placed = True
                    break
            if not placed:
                break

        if len(cells) != size:
            continue

        min_r = min(r for r, _ in cells)
        min_c = min(c for _, c in cells)
        norm = [(r - min_r, c - min_c) for r, c in cells]
        h = max(r for r, _ in norm) + 1
        w = max(c for _, c in norm) + 1
        if h > max_h or w > max_w:
            continue
        shape = np.zeros((h, w), dtype=bool)
        for r, c in norm:
            shape[r, c] = True
        return shape
    return None


def _mark_blocked(blocked: np.ndarray, coords: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in coords:
        for nr in range(max(0, r - 1), min(h, r + 2)):
            for nc in range(max(0, c - 1), min(w, c + 2)):
                blocked[nr, nc] = True


def _place_random_shape(cell: np.ndarray, blocked: np.ndarray, size: int, color: int) -> bool:
    h, w = cell.shape
    for _ in range(100):
        shape = _random_polyomino(size, h, w)
        if shape is None:
            continue
        coords = np.argwhere(shape)
        sh, sw = shape.shape
        placements: list[list[tuple[int, int]]] = []
        for top in range(h - sh + 1):
            for left in range(w - sw + 1):
                abs_coords = [(top + int(r), left + int(c)) for r, c in coords]
                if all(not blocked[r, c] for r, c in abs_coords):
                    placements.append(abs_coords)
        if not placements:
            continue

        chosen = random.choice(placements)
        for r, c in chosen:
            cell[r, c] = int(color)
        _mark_blocked(blocked, chosen)
        return True
    return False


def _place_patterns_in_cell(
    grid: np.ndarray, top: int, left: int, height: int, width: int, target_size: int, bg: int, border: int
) -> bool:
    palette = [color for color in range(10) if color not in {bg, border}]

    for _ in range(80):
        cell = np.full((height, width), bg, dtype=int)
        blocked = np.zeros((height, width), dtype=bool)

        if target_size >= 5:
            num_patterns = 2
            other_pool = [1, 2, 3] if target_size >= 7 else [1, 2, 3, 4]
        else:
            num_patterns = random.choice((2, 3))
            other_pool = [size for size in (1, 2, 3, 4) if size != target_size]
        if len(other_pool) < num_patterns - 1:
            continue

        sizes = [int(target_size)] + random.sample(other_pool, num_patterns - 1)
        colors = random.sample(palette, len(sizes))
        items = list(zip(sizes, colors))
        random.shuffle(items)
        items.sort(key=lambda item: item[0], reverse=True)

        ok = True
        for size, color in items:
            if not _place_random_shape(cell, blocked, int(size), int(color)):
                ok = False
                break
        if not ok:
            continue

        grid[top : top + height, left : left + width] = cell
        return True
    return False


def _validate_instance(grid: np.ndarray, bg: int, border: int) -> bool:
    try:
        found_border, h_lines, v_lines = _find_border_color(grid, bg)
    except AssertionError:
        return False
    if found_border != border or len(h_lines) < 3 or len(v_lines) < 3:
        return False

    for hi in range(len(h_lines) - 1):
        r0, r1 = h_lines[hi], h_lines[hi + 1]
        if r1 - r0 <= 1:
            return False
        for vi in range(len(v_lines) - 1):
            c0, c1 = v_lines[vi], v_lines[vi + 1]
            if c1 - c0 <= 1:
                return False
            target = _cell_gap_count(grid, bg, r0, r1, c0, c1)
            comps = _cell_components(grid, bg, border, r0, r1, c0, c1)
            if len(comps) < 2:
                return False
            if sum(len(comp) == target for comp in comps) != 1:
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    bg, border = random.sample(range(10), 2)
    nr = random.randint(*GRID_ROWS_RANGE)
    nc = random.randint(*GRID_COLS_RANGE)
    heights = [random.randint(*CELL_HEIGHT_RANGE) for _ in range(nr)]
    widths = [random.randint(*CELL_WIDTH_RANGE) for _ in range(nc)]

    patch_h = sum(heights) + nr + 1
    patch_w = sum(widths) + nc + 1
    patch = np.full((patch_h, patch_w), bg, dtype=int)

    h_lines = _line_positions_from_sizes(heights)
    v_lines = _line_positions_from_sizes(widths)
    patch[h_lines, :] = border
    patch[:, v_lines] = border

    gapped = _paint_random_gaps(patch, bg, h_lines, v_lines)
    if gapped is None:
        return None
    patch, gap_counts = gapped

    for ri in range(nr):
        for cj in range(nc):
            top = h_lines[ri] + 1
            left = v_lines[cj] + 1
            if not _place_patterns_in_cell(
                patch,
                top,
                left,
                heights[ri],
                widths[cj],
                int(gap_counts[ri, cj]),
                bg,
                border,
            ):
                return None

    mt = random.randint(*MARGIN_RANGE)
    mb = random.randint(*MARGIN_RANGE)
    ml = random.randint(*MARGIN_RANGE)
    mr = random.randint(*MARGIN_RANGE)
    grid = np.full((patch_h + mt + mb, patch_w + ml + mr), bg, dtype=int)
    grid[mt : mt + patch_h, ml : ml + patch_w] = patch

    if not _validate_instance(grid, bg, border):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 3")
