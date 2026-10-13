from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


def _ensure_arc_utils_on_path() -> None:
    here = Path(__file__).resolve()
    candidates = [Path("/home/lixinyu/snap/new/solvers")]
    for parent in [here.parent, *here.parents]:
        candidate = parent / "new" / "solvers"
        if (candidate / "arc_utils.py").exists():
            candidates.insert(0, candidate)
            break
    for candidate in candidates:
        if (candidate / "arc_utils.py").exists():
            path_str = str(candidate)
            if path_str not in sys.path:
                sys.path.append(path_str)
            return
    raise ImportError("unable to locate new/solvers/arc_utils.py")


_ensure_arc_utils_on_path()

from arc_utils import Color, colors_present


BG = int(Color.BLACK)
FRAME = int(Color.GREY)
ROW_LINE = int(Color.YELLOW)
COL_LINE = int(Color.PINK)
MARKER_PALETTE = [int(Color.BLUE), int(Color.RED), int(Color.GREEN), int(Color.ORANGE)]

MAX_ATTEMPTS = 300
LOCAL_ATTEMPTS = 120


def _infer_lattice(grid: np.ndarray) -> tuple[list[int], list[int]]:
    frame_rows = [int(r) for r in np.where(np.all(grid == FRAME, axis=1))[0]]
    frame_cols = [int(c) for c in np.where(np.all(grid == FRAME, axis=0))[0]]
    assert len(frame_rows) >= 2 and len(frame_cols) >= 2

    row_steps = {b - a for a, b in zip(frame_rows, frame_rows[1:])}
    col_steps = {b - a for a, b in zip(frame_cols, frame_cols[1:])}
    assert len(row_steps) == 1 and len(col_steps) == 1
    assert min(row_steps) >= 2 and min(col_steps) >= 2
    return frame_rows, frame_cols


def _extract_markers(grid: np.ndarray) -> tuple[list[tuple[int, int, int]], dict[int, list[tuple[int, int]]]]:
    markers: list[tuple[int, int, int]] = []
    grouped: dict[int, list[tuple[int, int]]] = {}
    for color in colors_present(grid, exclude={BG, FRAME}):
        coords = [tuple(map(int, coord)) for coord in np.argwhere(grid == color)]
        assert len(coords) == 2
        grouped[int(color)] = coords
        for r, c in coords:
            markers.append((r, c, int(color)))
    assert grouped
    return markers, grouped


def _settle_markers(grid: np.ndarray) -> tuple[np.ndarray, dict[int, list[tuple[int, int]]]]:
    markers, _grouped = _extract_markers(grid)
    settled = grid.copy()
    for r, c, _color in markers:
        settled[r, c] = BG

    final_by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c, color in sorted(markers, key=lambda item: (-item[0], item[1])):
        nr = int(r)
        while nr + 1 < settled.shape[0] and int(settled[nr + 1, c]) == BG:
            nr += 1
        settled[nr, c] = int(color)
        final_by_color.setdefault(int(color), []).append((int(nr), int(c)))

    for coords in final_by_color.values():
        coords.sort()
    return settled, final_by_color


def _draw_connectors(
    settled: np.ndarray, final_by_color: dict[int, list[tuple[int, int]]]
) -> np.ndarray:
    out = settled.copy()
    for color in sorted(final_by_color):
        coords = final_by_color[color]
        assert len(coords) == 2
        (r1, c1), (r2, c2) = coords
        if r1 == r2:
            for c in range(min(c1, c2) + 1, max(c1, c2)):
                if int(out[r1, c]) == BG:
                    out[r1, c] = ROW_LINE
        elif c1 == c2:
            for r in range(min(r1, r2) + 1, max(r1, r2)):
                if int(out[r, c1]) == BG:
                    out[r, c1] = COL_LINE
    return out


def _transform(
    grid: np.ndarray, *, return_state: bool = False
) -> np.ndarray | tuple[np.ndarray, dict[int, list[tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    _infer_lattice(g)
    settled, final_by_color = _settle_markers(g)
    out = _draw_connectors(settled, final_by_color)
    if return_state:
        return out, final_by_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Drop paired colored markers inside a grey lattice, then color-code the post-gravity alignments with connector trails.

    Input structure:
    - The grid is a regular grey lattice whose full grey rows and columns divide the canvas into black channel cells.
    - Inside the channel cells are isolated single-pixel markers; each marker color appears exactly twice.
    - Grey lattice pixels are fixed obstacles and all remaining non-lattice pixels are background channels.

    Rule:
    - Move every marker straight downward through black channel pixels, processing lower markers first and stopping above the first lattice pixel, the bottom edge, or an already settled marker.
    - After gravity, inspect each same-colored pair: if the two landed on the same row, paint yellow through the background pixels strictly between them; if they landed on the same column, paint magenta through the background pixels strictly between them.
    - Keep grey lattice pixels, marker pixels, and any off-path background pixels unchanged.

    Key insight:
    - The connector color is not determined from the input endpoints directly; gravity inside the lattice changes where the endpoints land, and only those landed positions decide whether the pair is row-aligned, column-aligned, or unconnected.
    """
    return _transform(grid)


def _build_lattice(n_rows: int, n_cols: int, cell_size: int) -> np.ndarray:
    height = n_rows * cell_size + n_rows + 1
    width = n_cols * cell_size + n_cols + 1
    grid = np.full((height, width), BG, dtype=int)
    grid[np.arange(0, height, cell_size + 1), :] = FRAME
    grid[:, np.arange(0, width, cell_size + 1)] = FRAME
    return grid


def _cell_origin(cell_r: int, cell_c: int, cell_size: int) -> tuple[int, int]:
    return 1 + cell_r * (cell_size + 1), 1 + cell_c * (cell_size + 1)


def _abs_coord(cell_r: int, cell_c: int, local_r: int, local_c: int, cell_size: int) -> tuple[int, int]:
    r0, c0 = _cell_origin(cell_r, cell_c, cell_size)
    return r0 + local_r, c0 + local_c


def _pair_candidates(
    pair_type: str, n_rows: int, n_cols: int, used_cells: set[tuple[int, int]]
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    if pair_type == "row":
        for cell_r in range(n_rows):
            free_cols = [cell_c for cell_c in range(n_cols) if (cell_r, cell_c) not in used_cells]
            for i, c1 in enumerate(free_cols):
                for c2 in free_cols[i + 1 :]:
                    candidates.append(((cell_r, c1), (cell_r, c2)))
    elif pair_type == "col":
        for cell_c in range(n_cols):
            free_rows = [cell_r for cell_r in range(n_rows) if (cell_r, cell_c) not in used_cells]
            for i, r1 in enumerate(free_rows):
                for r2 in free_rows[i + 1 :]:
                    candidates.append(((r1, cell_c), (r2, cell_c)))
    else:
        for r1 in range(n_rows):
            for c1 in range(n_cols):
                if (r1, c1) in used_cells:
                    continue
                for r2 in range(n_rows):
                    for c2 in range(n_cols):
                        if (r2, c2) in used_cells or (r1, c1) == (r2, c2):
                            continue
                        if r1 != r2 and c1 != c2 and (r1, c1) < (r2, c2):
                            candidates.append(((r1, c1), (r2, c2)))
    random.shuffle(candidates)
    return candidates


def _choose_cell_pairs(
    colors: list[int], pair_types: dict[int, str], n_rows: int, n_cols: int
) -> dict[int, tuple[tuple[int, int], tuple[int, int]]] | None:
    used_cells: set[tuple[int, int]] = set()
    chosen: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
    for color in colors:
        pair_type = pair_types[color]
        candidates = _pair_candidates(pair_type, n_rows, n_cols, used_cells)
        if not candidates:
            return None
        pair = candidates[0]
        chosen[color] = pair
        used_cells.add(pair[0])
        used_cells.add(pair[1])
    return chosen


def _random_local_pair(
    pair_type: str, cell_a: tuple[int, int], cell_b: tuple[int, int], cell_size: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    if pair_type == "row":
        local_r1 = random.randrange(cell_size)
        local_r2 = random.randrange(cell_size)
        if cell_size > 1 and local_r1 == cell_size - 1 and local_r2 == cell_size - 1:
            local_r2 = cell_size - 2
        local_c1 = random.randrange(cell_size)
        local_c2 = random.randrange(cell_size)
    elif pair_type == "col":
        local_c1 = random.randrange(cell_size)
        local_c2 = local_c1
        local_r1 = random.randrange(cell_size)
        local_r2 = random.randrange(cell_size)
        if cell_size > 1 and local_r1 == cell_size - 1 and local_r2 == cell_size - 1:
            local_r1 = cell_size - 2
    else:
        local_r1 = random.randrange(cell_size)
        local_c1 = random.randrange(cell_size)
        local_r2 = random.randrange(cell_size)
        local_c2 = random.randrange(cell_size)
    return (
        _abs_coord(cell_a[0], cell_a[1], local_r1, local_c1, cell_size),
        _abs_coord(cell_b[0], cell_b[1], local_r2, local_c2, cell_size),
    )


def _segment_records(
    grid: np.ndarray,
    final_by_color: dict[int, list[tuple[int, int]]],
    pair_types: dict[int, str],
) -> list[tuple[str, int, int, int, int]]:
    segments: list[tuple[str, int, int, int, int]] = []
    final_markers = {coord for coords in final_by_color.values() for coord in coords}
    base_grid = np.where(grid == FRAME, FRAME, BG)
    for color, pair_type in pair_types.items():
        coords = final_by_color[color]
        assert len(coords) == 2
        (r1, c1), (r2, c2) = coords
        actual_type = "row" if r1 == r2 else "col" if c1 == c2 else "none"
        if actual_type != pair_type:
            raise AssertionError(f"expected {pair_type}, got {actual_type}")
        if pair_type == "row":
            left = min(c1, c2) + 1
            right = max(c1, c2) - 1
            if left > right or not np.any(base_grid[r1, left : right + 1] == BG):
                raise AssertionError("row connector would be invisible")
            for coord in final_markers:
                if coord[0] == r1 and left <= coord[1] <= right:
                    raise AssertionError("marker lies on a horizontal connector")
            segments.append(("h", color, r1, left, right))
        elif pair_type == "col":
            top = min(r1, r2) + 1
            bottom = max(r1, r2) - 1
            if top > bottom or not np.any(base_grid[top : bottom + 1, c1] == BG):
                raise AssertionError("column connector would be invisible")
            for coord in final_markers:
                if coord[1] == c1 and top <= coord[0] <= bottom:
                    raise AssertionError("marker lies on a vertical connector")
            segments.append(("v", color, c1, top, bottom))
    for i, seg_a in enumerate(segments):
        for seg_b in segments[i + 1 :]:
            if seg_a[0] == "h" and seg_b[0] == "h":
                if seg_a[2] == seg_b[2] and not (seg_a[4] < seg_b[3] or seg_b[4] < seg_a[3]):
                    raise AssertionError("overlapping horizontal connectors")
            elif seg_a[0] == "v" and seg_b[0] == "v":
                if seg_a[2] == seg_b[2] and not (seg_a[4] < seg_b[3] or seg_b[4] < seg_a[3]):
                    raise AssertionError("overlapping vertical connectors")
            else:
                hseg = seg_a if seg_a[0] == "h" else seg_b
                vseg = seg_a if seg_a[0] == "v" else seg_b
                if hseg[3] <= vseg[2] <= hseg[4] and vseg[3] <= hseg[2] <= vseg[4]:
                    raise AssertionError("crossing connectors")
    return segments


def _validate_generated_grid(
    grid: np.ndarray, pair_types: dict[int, str]
) -> bool:
    try:
        _infer_lattice(grid)
        markers, grouped = _extract_markers(grid)
        assert set(grouped) == set(pair_types)
        assert all(len(coords) == 2 for coords in grouped.values())

        out, final_by_color = _transform(grid, return_state=True)
        _segment_records(grid, final_by_color, pair_types)

        moved = False
        for color, coords in grouped.items():
            before = sorted(coords)
            after = sorted(final_by_color[color])
            if before != after:
                moved = True
            pair_type = pair_types[color]
            if pair_type == "row":
                r = after[0][0]
                c0 = min(after[0][1], after[1][1]) + 1
                c1 = max(after[0][1], after[1][1])
                assert np.any(out[r, c0:c1] == ROW_LINE)
            elif pair_type == "col":
                c = after[0][1]
                r0 = min(after[0][0], after[1][0]) + 1
                r1 = max(after[0][0], after[1][0])
                assert np.any(out[r0:r1, c] == COL_LINE)
        assert moved
        return True
    except AssertionError:
        return False


def _gen_attempt() -> np.ndarray | None:
    cell_size = random.choice([2, 3, 3])
    row_bounds = (4, 8) if cell_size == 2 else (4, 7)
    col_bounds = (4, 8) if cell_size == 2 else (4, 7)
    n_rows = random.randint(*row_bounds)
    n_cols = random.randint(*col_bounds)

    color_count = random.choice([3, 4])
    colors = random.sample(MARKER_PALETTE, color_count)

    base_types = ["row", "col", "none"]
    if color_count == 4:
        base_types.append(random.choice(base_types))
    random.shuffle(base_types)
    pair_types = {color: pair_type for color, pair_type in zip(colors, base_types)}

    cell_pairs = _choose_cell_pairs(colors, pair_types, n_rows, n_cols)
    if cell_pairs is None:
        return None

    grid = _build_lattice(n_rows, n_cols, cell_size)
    for _ in range(LOCAL_ATTEMPTS):
        attempt = grid.copy()
        for color in colors:
            pair_type = pair_types[color]
            cell_a, cell_b = cell_pairs[color]
            coord_a, coord_b = _random_local_pair(pair_type, cell_a, cell_b, cell_size)
            attempt[coord_a] = color
            attempt[coord_b] = color
        if _validate_generated_grid(attempt, pair_types):
            return attempt
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid channel-gravity lattice input")
