from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
FRAME = 3
SEPARATOR = 6

GEN_MAX_ATTEMPTS = 400
GEN_SEED_ATTEMPTS = 200


def _separator_lines(grid: np.ndarray, axis: int) -> list[int]:
    if axis == 0:
        return [i for i in range(grid.shape[0]) if np.all(grid[i, :] == SEPARATOR)]
    return [i for i in range(grid.shape[1]) if np.all(grid[:, i] == SEPARATOR)]


def _cell_spans(lines: list[int], limit: int) -> list[tuple[int, int]]:
    assert lines and lines[0] == 0 and lines[-1] == limit - 1
    spans: list[tuple[int, int]] = []
    for a, b in zip(lines, lines[1:]):
        assert b > a + 1
        spans.append((a + 1, b))
    return spans


def _split_grid(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[list[np.ndarray]]]:
    row_lines = _separator_lines(grid, axis=0)
    col_lines = _separator_lines(grid, axis=1)
    assert len(row_lines) >= 5 and len(col_lines) >= 5

    row_spans = _cell_spans(row_lines, grid.shape[0])
    col_spans = _cell_spans(col_lines, grid.shape[1])
    cells: list[list[np.ndarray]] = []
    for r0, r1 in row_spans:
        row_cells = []
        for c0, c1 in col_spans:
            row_cells.append(grid[r0:r1, c0:c1].copy())
        cells.append(row_cells)
    return row_spans, col_spans, cells


def _find_seed(cells: list[list[np.ndarray]]) -> tuple[int, int, np.ndarray]:
    active: list[tuple[int, int, np.ndarray]] = []
    for r, row in enumerate(cells):
        for c, patch in enumerate(row):
            if np.any(patch != BACKGROUND):
                active.append((r, c, patch))
    assert len(active) == 1

    seed_r, seed_c, seed = active[0]
    h, w = seed.shape
    assert h >= 3 and w >= 3 and not (h == 3 and w == 3)
    assert np.all(seed != BACKGROUND)
    assert np.all(seed[0, :] == FRAME)
    assert np.all(seed[-1, :] == FRAME)
    assert np.all(seed[:, 0] == FRAME)
    assert np.all(seed[:, -1] == FRAME)
    inner_colors = set(map(int, np.unique(seed[1:-1, 1:-1])))
    assert inner_colors == {BLUE, RED}
    return seed_r, seed_c, seed


def _axis_indices(src_len: int, dst_len: int, anchor: str = "start") -> list[int]:
    assert src_len >= 3 and dst_len >= 3
    inner = list(range(1, src_len - 1))
    need = dst_len - 2
    base = inner if anchor == "start" else inner[::-1]
    if need <= len(base):
        chosen = base[:need]
    else:
        chosen = [base[i % len(base)] for i in range(need)]
    return [0] + chosen + [src_len - 1]


def _stretch_seed(
    seed: np.ndarray,
    shape: tuple[int, int],
    *,
    row_anchor: str = "start",
    col_anchor: str = "start",
) -> np.ndarray:
    row_idx = _axis_indices(seed.shape[0], int(shape[0]), anchor=row_anchor)
    col_idx = _axis_indices(seed.shape[1], int(shape[1]), anchor=col_anchor)
    return seed[np.ix_(row_idx, col_idx)]


def _solve_from_parts(
    grid: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    seed_pos: tuple[int, int],
    seed: np.ndarray,
    *,
    parity: int | None = None,
    row_anchor: str = "start",
    col_anchor: str = "start",
) -> np.ndarray:
    seed_r, seed_c = seed_pos
    seed_parity = (seed_r + seed_c) % 2 if parity is None else int(parity)
    out = np.array(grid, dtype=int, copy=True)
    for rr, (r0, r1) in enumerate(row_spans):
        for cc, (c0, c1) in enumerate(col_spans):
            if (rr, cc) == (seed_r, seed_c):
                continue
            if (rr + cc) % 2 != seed_parity:
                continue
            out[r0:r1, c0:c1] = _stretch_seed(
                seed,
                (r1 - r0, c1 - c0),
                row_anchor=row_anchor,
                col_anchor=col_anchor,
            )
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stretch the seeded frame pattern to each same-parity cell in a magenta grid.

    Input structure:
    - Magenta separator rows and columns partition the grid into a rectangular array of unequal-sized cells.
    - Exactly one cell is a fully filled seed pattern with a green one-pixel frame and a red/blue interior.
    - Every other cell interior is black, and the separator lines stay fixed.

    Rule:
    - Find the unique non-black seed cell and record its cell-grid parity.
    - For every other cell with the same row+column parity, keep the seed's outer frame bands fixed and resize only the interior bands.
    - In each axis, top/left-to-bottom/right interior lines are truncated when the target is smaller and cycled from the start when the target is larger; then stamp the resized pattern into that target cell.

    Key insight:
    - The copied object is not a rigid sprite: the green border stays one pixel thick while the red/blue interior rows and columns are cropped or repeated inside each differently sized target cell.
    """
    g = np.asarray(grid, dtype=int)
    row_spans, col_spans, cells = _split_grid(g)
    seed_r, seed_c, seed = _find_seed(cells)
    return _solve_from_parts(g, row_spans, col_spans, (seed_r, seed_c), seed)


def _axis_edges(sizes: list[int]) -> list[int]:
    edges = [0]
    pos = 0
    for size in sizes:
        pos += size + 1
        edges.append(pos)
    return edges


def _candidate_seeds(
    row_sizes: list[int],
    col_sizes: list[int],
    mode: str,
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    n_rows, n_cols = len(row_sizes), len(col_sizes)
    for sr in range(n_rows):
        for sc in range(n_cols):
            seed_h, seed_w = row_sizes[sr], col_sizes[sc]
            if seed_h == 3 and seed_w == 3:
                continue
            targets = [
                (r, c)
                for r in range(n_rows)
                for c in range(n_cols)
                if (r, c) != (sr, sc) and (r + c) % 2 == (sr + sc) % 2
            ]
            if len(targets) < 4:
                continue

            target_heights = [row_sizes[r] for r, _ in targets]
            target_widths = [col_sizes[c] for _, c in targets]
            has_h_smaller = any(h < seed_h for h in target_heights)
            has_h_larger = any(h > seed_h for h in target_heights)
            has_w_smaller = any(w < seed_w for w in target_widths)
            has_w_larger = any(w > seed_w for w in target_widths)
            varied_shapes = {(row_sizes[r], col_sizes[c]) for r, c in targets}
            if len(varied_shapes) < 3:
                continue
            if not any(row_sizes[r] != seed_h and col_sizes[c] != seed_w for r, c in targets):
                continue

            if mode == "full":
                if not (has_h_smaller and has_h_larger and has_w_smaller and has_w_larger):
                    continue
            else:
                full_h = has_h_smaller and has_h_larger
                full_w = has_w_smaller and has_w_larger
                if not ((full_h and any(w != seed_w for w in target_widths)) or (full_w and any(h != seed_h for h in target_heights))):
                    continue
            candidates.append((sr, sc))
    return candidates


def _random_layout() -> tuple[list[int], list[int], tuple[int, int]]:
    for _ in range(GEN_MAX_ATTEMPTS):
        mode = random.choices(["full", "edge"], weights=[3, 1], k=1)[0]
        n_rows = random.randint(4, 5)
        n_cols = random.randint(4, 5)
        row_sizes = [random.randint(3, 6) for _ in range(n_rows)]
        col_sizes = [random.randint(3, 6) for _ in range(n_cols)]
        if len(set(row_sizes)) < 2 or len(set(col_sizes)) < 2:
            continue
        if sum(row_sizes) + n_rows + 1 > 30 or sum(col_sizes) + n_cols + 1 > 30:
            continue
        candidates = _candidate_seeds(row_sizes, col_sizes, mode)
        if not candidates:
            continue
        return row_sizes, col_sizes, random.choice(candidates)
    raise RuntimeError("failed to build a non-degenerate grid layout")


def _seed_ok(seed: np.ndarray) -> bool:
    inner = seed[1:-1, 1:-1]
    if set(map(int, np.unique(inner))) != {BLUE, RED}:
        return False
    if np.array_equal(seed, np.flipud(seed)) and np.array_equal(seed, np.fliplr(seed)):
        return False
    if inner.shape[0] > 1 and len({tuple(row.tolist()) for row in inner}) < 2:
        return False
    if inner.shape[1] > 1 and len({tuple(col.tolist()) for col in inner.T}) < 2:
        return False
    if inner.shape[0] > 1 and np.array_equal(inner[0], inner[-1]):
        return False
    if inner.shape[1] > 1 and np.array_equal(inner[:, 0], inner[:, -1]):
        return False
    return True


def _random_seed(height: int, width: int) -> np.ndarray:
    for _ in range(GEN_SEED_ATTEMPTS):
        seed = np.full((height, width), FRAME, dtype=int)
        seed[1:-1, 1:-1] = np.array(
            [[random.choice((BLUE, RED)) for _ in range(width - 2)] for _ in range(height - 2)],
            dtype=int,
        )
        if _seed_ok(seed):
            return seed
    raise RuntimeError("failed to build a valid asymmetric seed")


def _assemble_input(
    row_sizes: list[int],
    col_sizes: list[int],
    seed_pos: tuple[int, int],
    seed: np.ndarray,
) -> np.ndarray:
    row_edges = _axis_edges(row_sizes)
    col_edges = _axis_edges(col_sizes)
    grid = np.full((row_edges[-1] + 1, col_edges[-1] + 1), BACKGROUND, dtype=int)
    grid[row_edges, :] = SEPARATOR
    grid[:, col_edges] = SEPARATOR
    sr, sc = seed_pos
    r0, r1 = row_edges[sr] + 1, row_edges[sr + 1]
    c0, c1 = col_edges[sc] + 1, col_edges[sc + 1]
    grid[r0:r1, c0:c1] = seed
    return grid


def _semantic_signature(patch: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return patch.shape, patch.tobytes()


def _validate_semantics(grid: np.ndarray) -> None:
    row_spans, col_spans, cells = _split_grid(grid)
    seed_r, seed_c, seed = _find_seed(cells)
    seed_h, seed_w = seed.shape
    parity = (seed_r + seed_c) % 2
    targets = [
        (r, c, row_spans[r], col_spans[c])
        for r in range(len(row_spans))
        for c in range(len(col_spans))
        if (r, c) != (seed_r, seed_c) and (r + c) % 2 == parity
    ]
    opposites = [
        (r, c, row_spans[r], col_spans[c])
        for r in range(len(row_spans))
        for c in range(len(col_spans))
        if (r + c) % 2 != parity
    ]
    assert len(set(r1 - r0 for r0, r1 in row_spans)) >= 2
    assert len(set(c1 - c0 for c0, c1 in col_spans)) >= 2
    assert targets
    assert all(np.all(cells[r][c] == BACKGROUND) for r, c, *_ in targets + opposites)

    heights = [r1 - r0 for _, _, (r0, r1), _ in targets]
    widths = [c1 - c0 for _, _, _, (c0, c1) in targets]
    full_h = any(h < seed_h for h in heights) and any(h > seed_h for h in heights)
    full_w = any(w < seed_w for w in widths) and any(w > seed_w for w in widths)
    assert (full_h and any(w != seed_w for w in widths)) or (full_w and any(h != seed_h for h in heights))
    assert any(h != seed_h or w != seed_w for h, w in zip(heights, widths))

    expected = _solve_from_parts(grid, row_spans, col_spans, (seed_r, seed_c), seed)
    assert np.array_equal(main(grid), expected)
    assert np.count_nonzero(expected != grid) > 0

    signatures = set()
    for r, c, (r0, r1), (c0, c1) in targets:
        patch = expected[r0:r1, c0:c1]
        assert np.array_equal(patch, _stretch_seed(seed, (r1 - r0, c1 - c0)))
        signatures.add(_semantic_signature(patch))
    assert len(signatures) >= 2

    for _r, _c, (r0, r1), (c0, c1) in opposites:
        assert np.all(expected[r0:r1, c0:c1] == BACKGROUND)

    swapped = _solve_from_parts(grid, row_spans, col_spans, (seed_r, seed_c), seed, parity=1 - parity)
    anchored = _solve_from_parts(
        grid,
        row_spans,
        col_spans,
        (seed_r, seed_c),
        seed,
        row_anchor="end",
        col_anchor="end",
    )
    assert not np.array_equal(expected, swapped)
    assert not np.array_equal(expected, anchored)


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        try:
            row_sizes, col_sizes, seed_pos = _random_layout()
            seed_h, seed_w = row_sizes[seed_pos[0]], col_sizes[seed_pos[1]]
            seed = _random_seed(seed_h, seed_w)
            grid = _assemble_input(row_sizes, col_sizes, seed_pos, seed)
            _validate_semantics(grid)
        except (AssertionError, RuntimeError):
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
