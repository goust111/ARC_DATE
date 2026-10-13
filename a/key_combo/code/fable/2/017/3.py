from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import Color, paste
except Exception:
    class Color:
        BLACK = 0

    def paste(
        dst: np.ndarray,
        patch: np.ndarray,
        r0: int,
        c0: int,
        *,
        mask: np.ndarray | None = None,
        transparent: int | None = None,
        overwrite: bool = True,
        clip: bool = False,
    ) -> np.ndarray:
        H, W = dst.shape
        h, w = patch.shape
        if not clip and (r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W):
            raise ValueError("paste out of bounds")
        rr0 = max(0, int(r0))
        cc0 = max(0, int(c0))
        rr1 = min(H, int(r0) + h)
        cc1 = min(W, int(c0) + w)
        if rr0 >= rr1 or cc0 >= cc1:
            return dst
        pr0 = rr0 - int(r0)
        pc0 = cc0 - int(c0)
        pr1 = pr0 + (rr1 - rr0)
        pc1 = pc0 + (cc1 - cc0)
        dst_view = dst[rr0:rr1, cc0:cc1]
        patch_view = patch[pr0:pr1, pc0:pc1]
        use = np.ones_like(patch_view, dtype=bool)
        if mask is not None:
            use &= mask[pr0:pr1, pc0:pc1]
        if transparent is not None:
            use &= patch_view != int(transparent)
            if not overwrite:
                use &= dst_view == int(transparent)
        dst_view[use] = patch_view[use]
        return dst


CELL_SIZE = 3
CELL_CENTER = (1, 1)
STRIDE = CELL_SIZE + 1
SEPARATOR_COLOR = Color.BLACK
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
MIN_GRID_CELLS = 3
MAX_GRID_CELLS = 6
MIN_CHAIN_LEN = 4
MAX_CHAIN_LEN = 8
GEN_ATTEMPTS = 400

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
OFFSET_TO_MARKER = {
    (-1, 0): (0, 1),
    (1, 0): (2, 1),
    (0, -1): (1, 0),
    (0, 1): (1, 2),
}
REQUIRED_DISTRACTORS = ("diagonal_singleton", "double_marker")
NOISE_KINDS = (
    "solid",
    "diagonal_singleton",
    "double_marker",
    "stripe",
    "checker",
    "center_pair",
)


def _extract_spans(length: int, separators: list[int]) -> list[tuple[int, int]] | None:
    separators = sorted(set(int(x) for x in separators))
    if not separators:
        return None
    if len(separators) > 1 and any(b - a != STRIDE for a, b in zip(separators, separators[1:])):
        return None

    spans: list[tuple[int, int]] = []
    if separators[0] == 0:
        if separators[-1] != length - 1:
            return None
        for left, right in zip(separators, separators[1:]):
            spans.append((left + 1, right - 1))
    elif separators[0] == CELL_SIZE:
        if separators[-1] != length - STRIDE:
            return None
        spans.append((0, CELL_SIZE - 1))
        for left, right in zip(separators, separators[1:]):
            spans.append((left + 1, right - 1))
        spans.append((separators[-1] + 1, length - 1))
    else:
        return None

    if not spans:
        return None
    if any(end - start + 1 != CELL_SIZE for start, end in spans):
        return None
    return spans


def _detect_lattice(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    H, W = grid.shape
    solid_rows: dict[int, list[int]] = {}
    solid_cols: dict[int, list[int]] = {}

    for r in range(H):
        color = int(grid[r, 0])
        if bool(np.all(grid[r, :] == color)):
            solid_rows.setdefault(color, []).append(r)
    for c in range(W):
        color = int(grid[0, c])
        if bool(np.all(grid[:, c] == color)):
            solid_cols.setdefault(color, []).append(c)

    candidates = []
    for color in set(solid_rows) & set(solid_cols):
        row_spans = _extract_spans(H, solid_rows[color])
        col_spans = _extract_spans(W, solid_cols[color])
        if row_spans is None or col_spans is None:
            continue
        candidates.append((len(row_spans) * len(col_spans), color, row_spans, col_spans))

    assert candidates, "failed to detect a regular separator lattice"
    candidates.sort(reverse=True)
    _cells, color, row_spans, col_spans = candidates[0]
    return int(color), row_spans, col_spans


def _cell_patch(
    grid: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    cell_rc: tuple[int, int],
) -> np.ndarray:
    cr, cc = cell_rc
    r0, r1 = row_spans[cr]
    c0, c1 = col_spans[cc]
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _parse_chain_cell(cell: np.ndarray) -> dict[str, int | tuple[int, int] | None] | None:
    values, counts = np.unique(cell, return_counts=True)
    if len(values) != 2:
        return None
    if sorted(map(int, counts)) != [1, 8]:
        return None

    marker_index = int(np.argmin(counts))
    base_index = 1 - marker_index
    base_color = int(values[base_index])
    marker_color = int(values[marker_index])
    marker_rc = tuple(map(int, np.argwhere(cell == marker_color)[0]))

    if marker_rc == CELL_CENTER:
        step = None
    else:
        step = (marker_rc[0] - CELL_CENTER[0], marker_rc[1] - CELL_CENTER[1])
        if step not in OFFSET_TO_MARKER:
            return None

    return {"base": base_color, "marker": marker_color, "step": step}


def _recover_chain(grid: np.ndarray) -> list[int]:
    separator_color, row_spans, col_spans = _detect_lattice(grid)
    assert separator_color >= 0

    chain_cells: dict[tuple[int, int], dict[str, int | tuple[int, int] | None]] = {}
    incoming: dict[tuple[int, int], int] = {}
    terminal_count = 0

    for r in range(len(row_spans)):
        for c in range(len(col_spans)):
            parsed = _parse_chain_cell(_cell_patch(grid, row_spans, col_spans, (r, c)))
            if parsed is None:
                continue
            chain_cells[(r, c)] = parsed

    assert chain_cells, "expected at least one pointer cell"

    for cell_rc, parsed in chain_cells.items():
        step = parsed["step"]
        if step is None:
            terminal_count += 1
            continue
        nr = cell_rc[0] + int(step[0])
        nc = cell_rc[1] + int(step[1])
        nxt = (nr, nc)
        assert nxt in chain_cells, "pointer must target another chain cell"
        incoming[nxt] = incoming.get(nxt, 0) + 1
        assert incoming[nxt] == 1, "each chain cell may have only one predecessor"

    assert terminal_count == 1, "expected exactly one terminal cell"
    starts = [cell_rc for cell_rc in chain_cells if incoming.get(cell_rc, 0) == 0]
    assert len(starts) == 1, "expected exactly one unpointed start cell"

    colors: list[int] = []
    seen: set[tuple[int, int]] = set()
    current = starts[0]
    while True:
        assert current not in seen, "chain may not self-intersect"
        seen.add(current)
        parsed = chain_cells[current]
        colors.append(int(parsed["base"]))
        if parsed["step"] is None:
            break
        step = parsed["step"]
        current = (current[0] + int(step[0]), current[1] + int(step[1]))

    assert len(seen) == len(chain_cells), "all chain cells must lie on the same path"
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the unique pointer chain from a separator lattice and straighten its base colors into one column.

    Input structure:
    - Solid separator rows and columns partition the grid into a regular lattice of 3x3 cell interiors.
    - Chain cells are 3x3 patches with one uniform base color plus exactly one off-color marker pixel.
    - A center marker marks the terminal cell; a marker one step above, below, left, or right of center points to the next adjacent lattice cell.
    - All other cells are patterned distractors and must be ignored.

    Rule:
    - Detect every valid chain cell by the exact `8 base + 1 marker` pattern and read each non-center marker as a local direction.
    - Find the unique chain start: the only chain cell that no other chain cell points to.
    - Follow the directed chain cell by cell until the center-marked terminal.
    - Output an `N x 1` column whose entries are the visited chain-cell base colors in traversal order.

    Key insight:
    - The marker color is irrelevant; only the marker's offset inside each 3x3 cell encodes the macro-scale next-step relation, and the global start is defined by indegree zero.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    colors = _recover_chain(g)
    return np.array(colors, dtype=int).reshape(-1, 1)


def _neighbors(cell_rc: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = cell_rc
    out = []
    for dr, dc in DIRS4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            out.append((nr, nc))
    return out


def _has_turn(path: list[tuple[int, int]]) -> bool:
    for i in range(1, len(path) - 1):
        a = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        b = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        if a != b:
            return True
    return False


def _extend_path(
    path: list[tuple[int, int]],
    used: set[tuple[int, int]],
    rows: int,
    cols: int,
    length: int,
) -> bool:
    if len(path) == length:
        return True

    current = path[-1]
    options = [cell for cell in _neighbors(current, rows, cols) if cell not in used]
    random.shuffle(options)
    options.sort(
        key=lambda cell: sum(next_cell not in used for next_cell in _neighbors(cell, rows, cols)),
        reverse=True,
    )

    for nxt in options:
        used.add(nxt)
        path.append(nxt)
        if _extend_path(path, used, rows, cols, length):
            return True
        path.pop()
        used.remove(nxt)
    return False


def _sample_path(rows: int, cols: int, length: int) -> list[tuple[int, int]] | None:
    require_turn = length >= 5 and random.random() < 0.85
    for _ in range(200):
        start = (random.randrange(rows), random.randrange(cols))
        path = [start]
        used = {start}
        if not _extend_path(path, used, rows, cols, length):
            continue
        if require_turn and not _has_turn(path):
            continue
        return path
    return None


def _choice_excluding(*blocked: int) -> int:
    options = [color for color in PALETTE if color not in blocked]
    return int(random.choice(options))


def _chain_patch(base: int, marker: int, step: tuple[int, int] | None) -> np.ndarray:
    patch = np.full((CELL_SIZE, CELL_SIZE), int(base), dtype=int)
    marker_rc = CELL_CENTER if step is None else OFFSET_TO_MARKER[step]
    patch[marker_rc] = int(marker)
    return patch


def _noise_patch(kind: str) -> np.ndarray:
    base = _choice_excluding(SEPARATOR_COLOR)
    accent = _choice_excluding(SEPARATOR_COLOR, base)
    accent2 = _choice_excluding(SEPARATOR_COLOR, base, accent)
    patch = np.full((CELL_SIZE, CELL_SIZE), base, dtype=int)

    if kind == "solid":
        return patch
    if kind == "diagonal_singleton":
        patch[random.choice([(0, 0), (0, 2), (2, 0), (2, 2)])] = accent
        return patch
    if kind == "double_marker":
        patch[random.choice([(0, 1), (1, 0), (1, 2), (2, 1)])] = accent
        patch[random.choice([(0, 0), (0, 2), (2, 0), (2, 2)])] = accent
        return patch
    if kind == "stripe":
        axis = random.choice(["row", "col"])
        if axis == "row":
            patch[0, :] = accent
            patch[2, :] = accent2
        else:
            patch[:, 0] = accent
            patch[:, 2] = accent2
        return patch
    if kind == "checker":
        for r in range(CELL_SIZE):
            for c in range(CELL_SIZE):
                patch[r, c] = (base, accent)[(r + c) % 2]
        patch[1, 1] = accent2
        return patch
    if kind == "center_pair":
        patch[1, 1] = accent
        patch[random.choice([(0, 1), (1, 0), (1, 2), (2, 1)])] = accent2
        return patch
    raise ValueError(kind)


def _render_grid(
    rows: int,
    cols: int,
    path: list[tuple[int, int]],
    base_colors: list[int],
    noise_assignment: dict[tuple[int, int], str],
) -> np.ndarray:
    grid = np.full((rows * STRIDE + 1, cols * STRIDE + 1), SEPARATOR_COLOR, dtype=int)
    path_index = {cell: idx for idx, cell in enumerate(path)}

    for r in range(rows):
        for c in range(cols):
            if (r, c) in path_index:
                idx = path_index[(r, c)]
                step = None
                if idx + 1 < len(path):
                    nxt = path[idx + 1]
                    step = (nxt[0] - r, nxt[1] - c)
                patch = _chain_patch(
                    base_colors[idx],
                    _choice_excluding(SEPARATOR_COLOR, base_colors[idx]),
                    step,
                )
            else:
                patch = _noise_patch(noise_assignment[(r, c)])
            paste(grid, patch, 1 + STRIDE * r, 1 + STRIDE * c)
    return grid


def _expected_output(base_colors: list[int]) -> np.ndarray:
    return np.array(base_colors, dtype=int).reshape(-1, 1)


def _validate_semantics(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    base_colors: list[int],
    noise_assignment: dict[tuple[int, int], str],
) -> bool:
    try:
        separator_color, row_spans, col_spans = _detect_lattice(grid)
        if separator_color != SEPARATOR_COLOR:
            return False
        if len(row_spans) * len(col_spans) != len(path) + len(noise_assignment):
            return False
        path_index = {cell: idx for idx, cell in enumerate(path)}
        observed_distractors = set()
        for r in range(len(row_spans)):
            for c in range(len(col_spans)):
                parsed = _parse_chain_cell(_cell_patch(grid, row_spans, col_spans, (r, c)))
                if (r, c) in path_index:
                    idx = path_index[(r, c)]
                    expected_step = None
                    if idx + 1 < len(path):
                        nxt = path[idx + 1]
                        expected_step = (nxt[0] - r, nxt[1] - c)
                    if parsed is None:
                        return False
                    if int(parsed["base"]) != int(base_colors[idx]):
                        return False
                    if parsed["step"] != expected_step:
                        return False
                else:
                    if parsed is not None:
                        return False
                    observed_distractors.add(noise_assignment[(r, c)])
        for required in REQUIRED_DISTRACTORS:
            if required not in observed_distractors:
                return False
    except Exception:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(MIN_GRID_CELLS, MAX_GRID_CELLS)
    cols = random.randint(MIN_GRID_CELLS, MAX_GRID_CELLS)
    total_cells = rows * cols

    max_len = min(MAX_CHAIN_LEN, total_cells - len(REQUIRED_DISTRACTORS))
    if max_len < MIN_CHAIN_LEN:
        return None
    chain_len = random.randint(MIN_CHAIN_LEN, max_len)
    path = _sample_path(rows, cols, chain_len)
    if path is None:
        return None

    base_colors = random.sample(PALETTE, chain_len)
    path_set = set(path)
    noise_cells = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in path_set]
    random.shuffle(noise_cells)
    if len(noise_cells) < len(REQUIRED_DISTRACTORS):
        return None

    required = list(REQUIRED_DISTRACTORS)
    random.shuffle(required)
    noise_assignment: dict[tuple[int, int], str] = {}
    for cell, kind in zip(noise_cells, required):
        noise_assignment[cell] = kind
    for cell in noise_cells[len(required) :]:
        noise_assignment[cell] = random.choice(NOISE_KINDS)

    grid = _render_grid(rows, cols, path, base_colors, noise_assignment)
    expected = _expected_output(base_colors)
    if not _validate_semantics(grid, path, base_colors, noise_assignment):
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pointer-chain lattice input")
