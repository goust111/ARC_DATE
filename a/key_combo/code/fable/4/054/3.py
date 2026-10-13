import numpy as np
from collections import Counter


COVER_COLOR = 8
PALETTE = tuple(range(10))
RNG = np.random.default_rng()
MAX_GENERATION_ATTEMPTS = 400


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for idx, flag in enumerate(mask.tolist()):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            runs.append((int(start), int(idx - 1)))
            start = None
    if start is not None:
        runs.append((int(start), int(len(mask) - 1)))
    return runs


def _unique_majority(block: np.ndarray) -> int:
    counts = Counter(map(int, block.ravel()))
    ranked = counts.most_common()
    assert ranked, "empty block"
    if len(ranked) > 1:
        assert ranked[0][1] > ranked[1][1], "expected a unique majority color"
    return int(ranked[0][0])


def _find_lattice(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]], int]:
    candidates = []
    for sep in map(int, np.unique(grid)):
        full_rows = np.all(grid == sep, axis=1)
        full_cols = np.all(grid == sep, axis=0)
        row_runs = _runs(~full_rows)
        col_runs = _runs(~full_cols)
        if not row_runs or not col_runs or len(row_runs) != len(col_runs):
            continue
        s = len(row_runs)
        if s < 2:
            continue
        row_sizes = {r1 - r0 + 1 for r0, r1 in row_runs}
        col_sizes = {c1 - c0 + 1 for c0, c1 in col_runs}
        if row_sizes != {s} or col_sizes != {s}:
            continue
        valid = True
        for r0, r1 in row_runs:
            for c0, c1 in col_runs:
                block = grid[r0 : r1 + 1, c0 : c1 + 1]
                if block.shape != (s, s) or np.any(block == sep):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue
        sep_mass = int(np.sum(grid == sep))
        candidates.append((sep_mass, sep, row_runs, col_runs, s))
    assert candidates, "failed to detect the separator lattice"
    _, sep, row_runs, col_runs, s = max(candidates, key=lambda item: item[0])
    return int(sep), row_runs, col_runs, int(s)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the separator lattice to its majority-color coarse map, then recover fully covered cells from minority-pixel records.

    Input structure:
    - Complete separator rows and columns of one color divide the grid into an s-by-s lattice of same-sized blocks.
    - The lattice itself is also s-by-s, and every block is exactly s-by-s internally.
    - Visible blocks have one unique majority color plus a few minority pixels; obscured blocks are filled entirely with color 8.

    Rule:
    - Detect the separator color and extract the non-separator row and column bands.
    - For each visible block, write its unique majority color into the corresponding coarse-grid cell.
    - Every non-majority pixel inside a visible block is a record: its local coordinate (r, c) points to coarse cell (r, c), and its color is that cell's true coarse color. Use these consistent records to fill the coarse cells whose whole blocks were obscured by color 8.

    Key insight:
    - The apparent block noise is actually a distributed registry: block-local coordinates match coarse-grid coordinates, so the hidden coarse colors can be reconstructed from the minority defects scattered through the other blocks.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _sep, row_runs, col_runs, s = _find_lattice(g)
    out = np.full((s, s), -1, dtype=int)
    obscured: list[tuple[int, int]] = []
    records: dict[tuple[int, int], set[int]] = {}

    for br, (r0, r1) in enumerate(row_runs):
        for bc, (c0, c1) in enumerate(col_runs):
            block = g[r0 : r1 + 1, c0 : c1 + 1]
            if np.all(block == COVER_COLOR):
                obscured.append((br, bc))
                continue
            majority = _unique_majority(block)
            out[br, bc] = majority
            for rr in range(s):
                for cc in range(s):
                    color = int(block[rr, cc])
                    if color != majority:
                        records.setdefault((rr, cc), set()).add(color)

    for (rr, cc), values in records.items():
        assert len(values) == 1, "records for one coarse coordinate must agree"
        if out[rr, cc] != -1:
            assert out[rr, cc] == next(iter(values)), "visible coarse cells must match any records about them"

    for br, bc in obscured:
        assert (br, bc) in records, "obscured cell missing recovery records"
        values = records[(br, bc)]
        assert len(values) == 1, "obscured-cell records must agree"
        out[br, bc] = next(iter(values))

    assert np.all(out >= 0), "failed to reconstruct the full coarse grid"
    return out


def _sample_without(exclude: set[int], count: int) -> list[int]:
    choices = [color for color in PALETTE if color not in exclude]
    picked = RNG.choice(choices, size=count, replace=False)
    return [int(color) for color in np.atleast_1d(picked)]


def _compose_grid(blocks: np.ndarray, separator: int, outer_border: bool) -> np.ndarray:
    s = int(blocks.shape[0])
    size = s * s + (s + 1 if outer_border else s - 1)
    grid = np.full((size, size), int(separator), dtype=int)
    offset = 1 if outer_border else 0
    for br in range(s):
        for bc in range(s):
            r0 = offset + br * (s + 1)
            c0 = offset + bc * (s + 1)
            grid[r0 : r0 + s, c0 : c0 + s] = blocks[br, bc]
    return grid


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    outer_border = bool(RNG.integers(0, 2))
    s_choices = np.array([2, 3, 4] if outer_border else [2, 3, 4, 5], dtype=int)
    s = int(RNG.choice(s_choices))
    max_hidden = min(3, (s * s - 1) // 2, s * s - 1)
    if max_hidden < 1:
        return None
    hidden_count = int(RNG.integers(1, max_hidden + 1))

    separator = int(RNG.choice([color for color in PALETTE if color != COVER_COLOR]))
    hidden_palette_size = int(RNG.integers(1, min(3, hidden_count) + 1))
    hidden_palette = _sample_without({separator, COVER_COLOR}, hidden_palette_size)
    visible_palette_size = int(RNG.integers(2, 5))
    visible_palette = _sample_without({separator, COVER_COLOR, *hidden_palette}, visible_palette_size)

    coords = [(r, c) for r in range(s) for c in range(s)]
    hidden_indices = RNG.choice(len(coords), size=hidden_count, replace=False)
    hidden_positions = [coords[int(idx)] for idx in np.atleast_1d(hidden_indices)]
    hidden_set = set(hidden_positions)

    truth = np.empty((s, s), dtype=int)
    for r in range(s):
        for c in range(s):
            if (r, c) in hidden_set:
                truth[r, c] = int(RNG.choice(hidden_palette))
            else:
                truth[r, c] = int(RNG.choice(visible_palette))

    if len(set(map(int, truth.ravel()))) < 2:
        return None

    blocks = np.empty((s, s, s, s), dtype=int)
    for br in range(s):
        for bc in range(s):
            if (br, bc) in hidden_set:
                blocks[br, bc] = np.full((s, s), COVER_COLOR, dtype=int)
                continue
            majority = int(truth[br, bc])
            block = np.full((s, s), majority, dtype=int)
            for rr, cc in hidden_positions:
                block[rr, cc] = int(truth[rr, cc])
            blocks[br, bc] = block

    grid = _compose_grid(blocks, separator, outer_border)
    if max(grid.shape) > 30:
        return None
    if not np.array_equal(main(grid), truth):
        return None
    return grid, truth


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        sample = _gen_attempt()
        if sample is not None:
            grid, _truth = sample
            return grid
    raise RuntimeError("failed to generate valid input")
