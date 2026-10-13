from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np


def _line_positions(grid: np.ndarray, color: int, axis: int) -> list[int]:
    if axis == 0:
        return [r for r in range(grid.shape[0]) if np.all(grid[r, :] == color)]
    return [c for c in range(grid.shape[1]) if np.all(grid[:, c] == color)]


def _cell_slices(lines: list[int]) -> list[tuple[int, int]]:
    slices: list[tuple[int, int]] = []
    for a, b in zip(lines, lines[1:]):
        if b - a <= 1:
            continue
        slices.append((a + 1, b))
    return slices


def _parse_lattice(grid: np.ndarray) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    lattice = int(g[0, 0])

    row_lines = _line_positions(g, lattice, axis=0)
    col_lines = _line_positions(g, lattice, axis=1)
    assert row_lines and col_lines
    assert row_lines[0] == 0 and row_lines[-1] == g.shape[0] - 1
    assert col_lines[0] == 0 and col_lines[-1] == g.shape[1] - 1

    row_steps = {b - a for a, b in zip(row_lines, row_lines[1:])}
    col_steps = {b - a for a, b in zip(col_lines, col_lines[1:])}
    assert len(row_steps) == 1 and len(col_steps) == 1

    row_slices = _cell_slices(row_lines)
    col_slices = _cell_slices(col_lines)
    assert row_slices and col_slices

    cell_h = row_slices[0][1] - row_slices[0][0]
    cell_w = col_slices[0][1] - col_slices[0][0]
    assert all((b - a) == cell_h for a, b in row_slices)
    assert all((b - a) == cell_w for a, b in col_slices)
    return lattice, cell_h, row_slices, col_slices


def _infer_background(grid: np.ndarray, lattice: int) -> int:
    values = [int(v) for v in grid.reshape(-1) if int(v) != lattice]
    assert values
    bg, bg_count = Counter(values).most_common(1)[0]
    assert bg_count > 0
    return int(bg)


def _row_cells(
    grid: np.ndarray, row_slice: tuple[int, int], col_slices: Iterable[tuple[int, int]]
) -> list[np.ndarray]:
    ra, rb = row_slice
    return [grid[ra:rb, ca:cb].copy() for ca, cb in col_slices]


def _majority_and_odd(cells: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    sigs = [cell.tobytes() for cell in cells]
    counts = Counter(sigs)
    majority_sig, majority_count = counts.most_common(1)[0]
    assert majority_count == len(cells) - 1
    odd_sigs = [sig for sig, count in counts.items() if count == 1]
    assert len(odd_sigs) == 1
    majority = next(cell for cell, sig in zip(cells, sigs) if sig == majority_sig)
    odd = next(cell for cell, sig in zip(cells, sigs) if sig == odd_sigs[0])
    return majority, odd


def _diff_info(majority: np.ndarray, odd: np.ndarray, bg: int) -> tuple[int, int, int]:
    diff = np.argwhere(majority != odd)
    assert len(diff) == 1
    r, c = map(int, diff[0])
    majority_color = int(majority[r, c])
    odd_color = int(odd[r, c])
    assert (majority_color == bg) != (odd_color == bg)
    if majority_color == bg:
        return r, c, odd_color
    return r, c, majority_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble a hidden image from each row's one-pixel deviation inside an anomalous lattice cell.

    Input structure:
    - The grid is split by a uniform lattice-line color into equal rectangular cells.
    - Inside each cell, a sparse foreground pattern sits on a shared background color.
    - In every lattice row, exactly one cell differs from the row's majority cell by exactly one interior pixel.

    Rule:
    - Compare cells row by row to find the majority cell and the single anomalous cell.
    - Read the local coordinate of their one-pixel difference.
    - Output one cell-interior-sized grid filled with background, then paint that local coordinate with the non-background color from the differing pair.

    Key insight:
    - The answer never appears as a whole object; each row hides one pixel of it, and missing-pixel rows must be decoded by looking back at the majority cell rather than the anomaly itself.
    """
    g = np.asarray(grid, dtype=int)
    lattice, cell_h, row_slices, col_slices = _parse_lattice(g)
    bg = _infer_background(g, lattice)
    cell_w = col_slices[0][1] - col_slices[0][0]

    out = np.full((cell_h, cell_w), bg, dtype=int)
    seen_coords: set[tuple[int, int]] = set()
    for row_slice in row_slices:
        cells = _row_cells(g, row_slice, col_slices)
        majority, odd = _majority_and_odd(cells)
        r, c, color = _diff_info(majority, odd, bg)
        assert (r, c) not in seen_coords
        seen_coords.add((r, c))
        out[r, c] = color

    return out


def _build_lattice(rows: list[list[np.ndarray]], lattice: int) -> np.ndarray:
    n_rows = len(rows)
    n_cols = len(rows[0])
    cell_h, cell_w = rows[0][0].shape
    out = np.full((n_rows * cell_h + n_rows + 1, n_cols * cell_w + n_cols + 1), lattice, dtype=int)
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            r0 = 1 + r * (cell_h + 1)
            c0 = 1 + c * (cell_w + 1)
            out[r0 : r0 + cell_h, c0 : c0 + cell_w] = cell
    return out


def _unique_coords(rng: np.random.Generator, cell_h: int, cell_w: int, n_rows: int) -> list[tuple[int, int]]:
    all_coords = [(r, c) for r in range(cell_h) for c in range(cell_w)]
    edge_coords = [(r, c) for r, c in all_coords if r in (0, cell_h - 1) or c in (0, cell_w - 1)]
    inner_coords = [coord for coord in all_coords if coord not in edge_coords]

    chosen: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()
    if edge_coords:
        coord = edge_coords[int(rng.integers(len(edge_coords)))]
        chosen.append(coord)
        used.add(coord)
    if n_rows >= 2 and inner_coords:
        coord = inner_coords[int(rng.integers(len(inner_coords)))]
        chosen.append(coord)
        used.add(coord)

    remaining = [coord for coord in all_coords if coord not in used]
    rng.shuffle(remaining)
    chosen.extend(remaining[: n_rows - len(chosen)])
    rng.shuffle(chosen)
    return chosen


def _sample_palette(rng: np.random.Generator, excluded: set[int], count: int) -> list[int]:
    choices = [c for c in range(10) if c not in excluded]
    rng.shuffle(choices)
    return [int(c) for c in choices[:count]]


def _make_majority_cell(
    rng: np.random.Generator,
    cell_h: int,
    cell_w: int,
    bg: int,
    fg_palette: list[int],
    target_coord: tuple[int, int],
    target_color: int,
    polarity: str,
    used_majorities: set[bytes],
) -> np.ndarray | None:
    area = cell_h * cell_w
    target_r, target_c = target_coord
    max_fg = min(area - 1, max(5, area // 2))
    for _ in range(200):
        cell = np.full((cell_h, cell_w), bg, dtype=int)
        row_palette = fg_palette[:]
        rng.shuffle(row_palette)
        row_palette = row_palette[: int(rng.integers(1, min(3, len(row_palette)) + 1))]
        if target_color not in row_palette:
            row_palette[0] = target_color

        total_fg = int(rng.integers(3, max_fg + 1))
        if polarity == "missing":
            cell[target_r, target_c] = target_color
            other_count = total_fg - 1
            pool = [(r, c) for r in range(cell_h) for c in range(cell_w) if (r, c) != target_coord]
        else:
            other_count = total_fg
            pool = [(r, c) for r in range(cell_h) for c in range(cell_w) if (r, c) != target_coord]

        if other_count > len(pool):
            continue
        rng.shuffle(pool)
        for r, c in pool[:other_count]:
            cell[r, c] = int(row_palette[int(rng.integers(len(row_palette)))])

        if polarity == "extra":
            assert int(cell[target_r, target_c]) == bg
        else:
            assert int(cell[target_r, target_c]) == target_color

        if np.count_nonzero(cell != bg) < 3:
            continue
        if cell.tobytes() in used_majorities:
            continue
        return cell
    return None


def _polarity_mix(rng: np.random.Generator, n_rows: int) -> list[str]:
    extras = int(rng.integers(1, n_rows))
    polarities = ["extra"] * extras + ["missing"] * (n_rows - extras)
    rng.shuffle(polarities)
    return polarities


def _semantic_validate(inp: np.ndarray, expected: np.ndarray) -> bool:
    lattice, cell_h, row_slices, col_slices = _parse_lattice(inp)
    bg = _infer_background(inp, lattice)
    cell_w = col_slices[0][1] - col_slices[0][0]
    assert expected.shape == (cell_h, cell_w)

    rebuilt = np.full((cell_h, cell_w), bg, dtype=int)
    odd_only = np.full((cell_h, cell_w), bg, dtype=int)
    majority_only = np.full((cell_h, cell_w), bg, dtype=int)
    seen: set[tuple[int, int]] = set()
    polarities: set[str] = set()

    for row_slice in row_slices:
        cells = _row_cells(inp, row_slice, col_slices)
        majority, odd = _majority_and_odd(cells)
        diff = np.argwhere(majority != odd)
        assert len(diff) == 1
        r, c = map(int, diff[0])
        assert (r, c) not in seen
        seen.add((r, c))

        majority_color = int(majority[r, c])
        odd_color = int(odd[r, c])
        if majority_color == bg:
            polarities.add("extra")
            rebuilt[r, c] = odd_color
            odd_only[r, c] = odd_color
        else:
            polarities.add("missing")
            rebuilt[r, c] = majority_color
            majority_only[r, c] = majority_color

    assert polarities == {"extra", "missing"}
    assert np.array_equal(rebuilt, expected)
    assert not np.array_equal(odd_only, expected)
    assert not np.array_equal(majority_only, expected)
    assert np.array_equal(main(inp), expected)
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    cell_h = int(rng.choice([4, 5]))
    cell_w = int(rng.choice([4, 5]))
    n_rows = 4 if cell_h == 5 else int(rng.integers(4, 6))
    n_cols = int(rng.integers(3, 5))

    lattice, bg = map(int, rng.choice(10, size=2, replace=False))
    fg_palette = _sample_palette(rng, {lattice, bg}, count=int(rng.integers(3, 6)))

    coords = _unique_coords(rng, cell_h, cell_w, n_rows)
    polarities = _polarity_mix(rng, n_rows)
    answer_colors = [fg_palette[int(rng.integers(len(fg_palette)))] for _ in range(n_rows)]

    expected = np.full((cell_h, cell_w), bg, dtype=int)
    rows: list[list[np.ndarray]] = []
    used_majorities: set[bytes] = set()

    for row_idx in range(n_rows):
        coord = coords[row_idx]
        color = int(answer_colors[row_idx])
        polarity = polarities[row_idx]
        expected[coord] = color

        majority = _make_majority_cell(
            rng, cell_h, cell_w, bg, fg_palette, coord, color, polarity, used_majorities
        )
        if majority is None:
            return None
        used_majorities.add(majority.tobytes())

        odd = majority.copy()
        if polarity == "extra":
            odd[coord] = color
        else:
            odd[coord] = bg

        odd_col = int(rng.integers(n_cols))
        row_cells = [majority.copy() for _ in range(n_cols)]
        row_cells[odd_col] = odd
        rows.append(row_cells)

    inp = _build_lattice(rows, lattice)

    non_lattice = [int(v) for v in inp.reshape(-1) if int(v) != lattice]
    counts = Counter(non_lattice).most_common(2)
    if len(counts) > 1 and counts[0][1] <= counts[1][1]:
        return None

    try:
        _semantic_validate(inp, expected)
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after 400 attempts")
