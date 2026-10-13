from __future__ import annotations

from collections import Counter
import random

import numpy as np


PLACEHOLDER = 5
GEN_RETRIES = 400


def _placeholder_suffix(grid: np.ndarray) -> tuple[int, int, int]:
    """Return (placeholder_color, r_cut, c_cut) for a bottom/right suffix union."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[int, int, int]] = []
    for color in np.unique(g):
        color = int(color)
        rows_all = np.all(g == color, axis=1)
        cols_all = np.all(g == color, axis=0)

        r_cut = h
        while r_cut > 0 and bool(rows_all[r_cut - 1]):
            r_cut -= 1
        c_cut = w
        while c_cut > 0 and bool(cols_all[c_cut - 1]):
            c_cut -= 1

        if c_cut == 0:
            continue
        expected = np.zeros((h, w), dtype=bool)
        if r_cut < h:
            expected[r_cut:, :] = True
        if c_cut < w:
            expected[:, c_cut:] = True
        if c_cut < w and np.array_equal(g == color, expected):
            candidates.append((color, r_cut, c_cut))

    assert len(candidates) == 1, "expected a unique suffix placeholder color"
    return candidates[0]


def _detect_lattice(block: np.ndarray) -> tuple[int, list[int], list[int], list[tuple[int, int]], list[tuple[int, int]]]:
    """Detect lattice color, full gridline rows/cols, and the cell slices between them."""
    g = np.asarray(block, dtype=int)
    h, w = g.shape
    best: tuple[int, int, list[int], list[int]] | None = None
    for color in np.unique(g):
        color = int(color)
        row_lines = [r for r in range(h) if np.all(g[r, :] == color)]
        col_lines = [c for c in range(w) if np.all(g[:, c] == color)]
        if len(row_lines) < 2 or len(col_lines) < 2:
            continue
        score = len(row_lines) + len(col_lines)
        if best is None or score > best[0]:
            best = (score, color, row_lines, col_lines)

    assert best is not None, "failed to detect lattice lines"
    lattice = int(best[1])
    row_lines = [int(x) for x in best[2]]
    col_lines = [int(x) for x in best[3]]
    r_slices = [(row_lines[i] + 1, row_lines[i + 1]) for i in range(len(row_lines) - 1) if row_lines[i + 1] > row_lines[i] + 1]
    c_slices = [(col_lines[i] + 1, col_lines[i + 1]) for i in range(len(col_lines) - 1) if col_lines[i + 1] > col_lines[i] + 1]
    assert r_slices and c_slices, "expected at least one visible cell"
    cell_h = r_slices[0][1] - r_slices[0][0]
    cell_w = c_slices[0][1] - c_slices[0][0]
    assert all((b - a) == cell_h for a, b in r_slices)
    assert all((b - a) == cell_w for a, b in c_slices)
    return lattice, row_lines, col_lines, r_slices, c_slices


def _global_majority_cell(cells: list[np.ndarray]) -> np.ndarray:
    sigs = [cell.tobytes() for cell in cells]
    counts = Counter(sigs)
    majority_sig = counts.most_common(1)[0][0]
    return next(cell.copy() for cell in cells if cell.tobytes() == majority_sig)


def _build_completed_grid(
    grid: np.ndarray,
    lattice: int,
    r_slices: list[tuple[int, int]],
    visible_cells: list[list[np.ndarray]],
    last_cells: list[np.ndarray],
    bg: int,
) -> np.ndarray:
    """Fill the right suffix using the inferred final cell for each row."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = np.full((h, w), int(bg), dtype=int)

    row_line_set = {0}
    for a, _ in r_slices:
        row_line_set.add(a - 1)
    row_line_set.add(h - 1)

    for r in range(h):
        if r in row_line_set:
            out[r, :] = int(lattice)

    cell_h = visible_cells[0][0].shape[0]
    full_cols = len(visible_cells[0])
    for row_idx, (ra, rb) in enumerate(r_slices):
        row_cursor = 1
        for cell in visible_cells[row_idx]:
            out[ra:rb, row_cursor : row_cursor + cell_h] = cell
            row_cursor += cell_h
            out[ra:rb, row_cursor] = int(lattice)
            row_cursor += 1
        cell = last_cells[row_idx]
        out[ra:rb, row_cursor : row_cursor + cell_h] = cell
        row_cursor += cell_h
        assert row_cursor == w - 1
        out[ra:rb, row_cursor] = int(lattice)

    assert full_cols + 1 == (w - 1) // (cell_h + 1)
    return out


def _extract_odd_cells(completed: np.ndarray) -> tuple[int, list[tuple[int, int]], list[np.ndarray], int]:
    lattice, row_lines, col_lines, r_slices, c_slices = _detect_lattice(completed)
    bg_candidates = [int(x) for x in completed.ravel() if int(x) != lattice]
    assert bg_candidates
    bg = Counter(bg_candidates).most_common(1)[0][0]

    chosen: list[np.ndarray] = []
    for ra, rb in r_slices:
        row_cells = [completed[ra:rb, ca:cb].copy() for ca, cb in c_slices]
        sigs = [cell.tobytes() for cell in row_cells]
        counts = Counter(sigs)
        odd_indices = [i for i, sig in enumerate(sigs) if counts[sig] == 1]
        assert len(odd_indices) == 1, "each lattice row must have exactly one odd cell"
        chosen.append(row_cells[odd_indices[0]])
    return lattice, r_slices, chosen, int(bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the placeholder-covered lattice suffix, then extract the unique cell from each lattice row.

    Input structure:
    - A lattice of equal square cells appears in the upper-left visible region, separated by one-color gridlines.
    - The right suffix is covered by a single placeholder color, so the last cell of each lattice row is only partially visible.
    - Within every lattice row, all cells share one interior pattern except for exactly one odd cell.

    Rule:
    - Detect the lattice geometry from the visible block and infer the cell pattern that completes the hidden suffix.
    - Reconstruct the full lattice row by row, restoring the final hidden cell and the outer gridline.
    - On the completed lattice, compare cells within each horizontal lattice row and locate the pattern that appears exactly once.
    - Extract those odd cells from top to bottom into a single-column lattice that preserves the original cell size and gridline color.

    Key insight:
    - Some rows appear uniform before completion because the only odd cell is the partially hidden last cell; completing that suffix is required before row-wise odd-one-out extraction becomes possible.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    placeholder, _r_cut, c_cut = _placeholder_suffix(g)
    assert placeholder == PLACEHOLDER
    visible = g[:, :c_cut]
    lattice, _row_lines, _col_lines, r_slices, c_slices = _detect_lattice(visible)

    cell_size = c_slices[0][1] - c_slices[0][0]
    stride = cell_size + 1
    h, w = g.shape
    assert (h - 1) % stride == 0 and (w - 1) % stride == 0
    n_rows = (h - 1) // stride
    n_cols = (w - 1) // stride
    assert n_rows == len(r_slices), "generation uses right-suffix-only placeholders"
    assert len(c_slices) == n_cols - 1, "expected exactly one partially hidden final cell"

    visible_cells: list[list[np.ndarray]] = []
    all_full_cells: list[np.ndarray] = []
    for ra, rb in r_slices:
        row_cells = [visible[ra:rb, ca:cb].copy() for ca, cb in c_slices]
        visible_cells.append(row_cells)
        all_full_cells.extend(row_cells)

    majority_cell = _global_majority_cell(all_full_cells)
    sig_majority = majority_cell.tobytes()
    majority_count = Counter(cell.tobytes() for cell in all_full_cells)
    assert majority_count[sig_majority] >= len(all_full_cells) - n_rows

    partial_start = 1 + (n_cols - 1) * stride
    prefix_w = c_cut - partial_start
    assert 1 <= prefix_w < cell_size

    partial_prefixes = [visible[ra:rb, partial_start:c_cut].copy() for ra, rb in r_slices]
    hidden_candidates: list[np.ndarray] = []
    for cell in all_full_cells:
        if cell.tobytes() == sig_majority:
            continue
        if np.array_equal(cell[:, :prefix_w], partial_prefixes[0]):
            hidden_candidates.append(cell.copy())

    uniq_hidden: list[np.ndarray] = []
    seen_hidden: set[bytes] = set()
    for cell in hidden_candidates:
        sig = cell.tobytes()
        if sig not in seen_hidden:
            uniq_hidden.append(cell)
            seen_hidden.add(sig)
    assert len(uniq_hidden) == 1, "expected one visible odd cell matching the hidden prefix"
    hidden_odd = uniq_hidden[0]

    last_cells: list[np.ndarray] = []
    for row_idx, row_cells in enumerate(visible_cells):
        row_sigs = [cell.tobytes() for cell in row_cells]
        counts = Counter(row_sigs)
        visible_odds = [cell for cell in row_cells if counts[cell.tobytes()] == 1 and cell.tobytes() != sig_majority]
        assert len(visible_odds) <= 1
        prefix = partial_prefixes[row_idx]
        if visible_odds:
            assert np.array_equal(majority_cell[:, :prefix_w], prefix)
            last_cells.append(majority_cell.copy())
        else:
            assert np.array_equal(hidden_odd[:, :prefix_w], prefix)
            last_cells.append(hidden_odd.copy())

    bg = int(Counter(int(x) for x in majority_cell.ravel()).most_common(1)[0][0])
    completed = _build_completed_grid(g, lattice, r_slices, visible_cells, last_cells, bg)
    lattice, comp_r_slices, odd_cells, bg = _extract_odd_cells(completed)

    out = np.full((h, cell_size + 2), int(bg), dtype=int)
    out[:, 0] = int(lattice)
    out[:, -1] = int(lattice)
    for ra, _rb in comp_r_slices:
        out[ra - 1, :] = int(lattice)
    out[-1, :] = int(lattice)
    for (ra, rb), cell in zip(comp_r_slices, odd_cells, strict=True):
        out[ra:rb, 1:-1] = cell
    return out


def _connected_mask(inner_h: int, inner_w: int, n_fg: int) -> np.ndarray | None:
    cells = {(random.randrange(inner_h), random.randrange(inner_w))}
    while len(cells) < n_fg:
        frontier = []
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < inner_h and 0 <= cc < inner_w and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))
    mask = np.zeros((inner_h, inner_w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _make_majority_cell(size: int, bg: int, sp1: int, sp2: int, hidden_cols: int) -> np.ndarray | None:
    inner_h = size - 2
    inner_w = size - 2
    hidden_start = size - hidden_cols
    visible_limit_inner = max(0, hidden_start - 1)

    for _ in range(200):
        n_fg = random.randint(4, min(8, inner_h * inner_w))
        mask = _connected_mask(inner_h, inner_w, n_fg)
        if mask is None:
            continue
        if not all(mask[r, :].any() for r in range(inner_h)):
            continue
        if not all(mask[:, c].any() for c in range(inner_w)):
            continue
        hidden_inner_cols = [c for c in range(inner_w) if c + 1 >= hidden_start]
        visible_inner_cols = [c for c in range(inner_w) if c + 1 < hidden_start]
        if not hidden_inner_cols or not visible_inner_cols:
            continue
        if not any(mask[:, c].any() for c in hidden_inner_cols):
            continue
        if not any(mask[:, c].any() for c in visible_inner_cols):
            continue

        inner = np.full((inner_h, inner_w), int(bg), dtype=int)
        inner[mask] = int(sp1)
        fg_coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
        accent_count = random.randint(1, min(2, len(fg_coords)))
        for r, c in random.sample(fg_coords, accent_count):
            inner[r, c] = int(sp2)

        cell = np.full((size, size), int(bg), dtype=int)
        cell[1:-1, 1:-1] = inner
        if len({int(x) for x in cell.ravel()}) >= 3:
            return cell
    return None


def _variant_from_zone(base: np.ndarray, zone_cols: list[int], bg: int, sp1: int, sp2: int, *, visible_change: bool) -> np.ndarray | None:
    cell = base.copy()
    candidates = []
    for r in range(1, cell.shape[0] - 1):
        for c in zone_cols:
            if 0 < c < cell.shape[1] - 1 and int(cell[r, c]) in {sp1, sp2}:
                candidates.append((r, c))
    if not candidates:
        return None

    random.shuffle(candidates)
    for r, c in candidates:
        out = base.copy()
        out[r, c] = sp2 if int(out[r, c]) == sp1 else sp1
        if np.array_equal(out, base):
            continue
        if visible_change:
            if np.array_equal(out[:, : c + 1], base[:, : c + 1]):
                continue
        return out
    return None


def _render_lattice(rows: list[list[np.ndarray]], lattice: int, bg: int) -> np.ndarray:
    n_rows = len(rows)
    n_cols = len(rows[0])
    cell_size = rows[0][0].shape[0]
    stride = cell_size + 1
    h = 1 + n_rows * stride
    w = 1 + n_cols * stride
    grid = np.full((h, w), int(bg), dtype=int)
    for r in range(0, h, stride):
        grid[r, :] = int(lattice)
    for c in range(0, w, stride):
        grid[:, c] = int(lattice)
    for rr, row in enumerate(rows):
        for cc, cell in enumerate(row):
            r0 = 1 + rr * stride
            c0 = 1 + cc * stride
            grid[r0 : r0 + cell_size, c0 : c0 + cell_size] = cell
    return grid


def _expected_output(full_grid: np.ndarray) -> np.ndarray:
    lattice, r_slices, odd_cells, bg = _extract_odd_cells(full_grid)
    cell_size = odd_cells[0].shape[0]
    out = np.full((full_grid.shape[0], cell_size + 2), int(bg), dtype=int)
    out[:, 0] = int(lattice)
    out[:, -1] = int(lattice)
    for ra, _rb in r_slices:
        out[ra - 1, :] = int(lattice)
    out[-1, :] = int(lattice)
    for (ra, rb), cell in zip(r_slices, odd_cells, strict=True):
        out[ra:rb, 1:-1] = cell
    return out


def _semantic_checks(
    inp: np.ndarray,
    full: np.ndarray,
    majority: np.ndarray,
    hidden_odd: np.ndarray,
    hidden_rows: set[int],
    c_cut: int,
) -> bool:
    try:
        visible = inp[:, :c_cut]
        _lattice, _row_lines, _col_lines, r_slices, c_slices = _detect_lattice(visible)
        cell_size = c_slices[0][1] - c_slices[0][0]
        stride = cell_size + 1
        n_cols = (full.shape[1] - 1) // stride
        if len(c_slices) != n_cols - 1:
            return False

        all_visible_cells = []
        visible_hidden_match = 0
        visible_row_count = 0
        for row_idx, (ra, rb) in enumerate(r_slices):
            row_cells = [visible[ra:rb, ca:cb].copy() for ca, cb in c_slices]
            all_visible_cells.extend(row_cells)
            full_row = [full[ra:rb, 1 + j * stride : 1 + j * stride + cell_size].copy() for j in range(n_cols)]
            counts = Counter(cell.tobytes() for cell in full_row)
            if sorted(counts.values()) != [1, n_cols - 1]:
                return False

            odd_sig = next(sig for sig, cnt in counts.items() if cnt == 1)
            odd_cell = next(cell for cell in full_row if cell.tobytes() == odd_sig)
            visible_odds = sum(1 for cell in row_cells if cell.tobytes() == odd_sig)
            if row_idx in hidden_rows:
                if visible_odds != 0:
                    return False
                if not all(np.array_equal(cell, majority) for cell in row_cells):
                    return False
                if not np.array_equal(odd_cell[:, : c_cut - (1 + (n_cols - 1) * stride)], majority[:, : c_cut - (1 + (n_cols - 1) * stride)]):
                    return False
                visible_hidden_match += 1
            else:
                if visible_odds != 1:
                    return False
                visible_row_count += 1

        if visible_hidden_match < 1 or visible_row_count < 1:
            return False

        majority_sig = majority.tobytes()
        prefix_w = c_cut - (1 + (n_cols - 1) * stride)
        prefix_matches = {
            cell.tobytes()
            for cell in all_visible_cells
            if cell.tobytes() != majority_sig and np.array_equal(cell[:, :prefix_w], hidden_odd[:, :prefix_w])
        }
        if prefix_matches != {hidden_odd.tobytes()}:
            return False

        return np.array_equal(main(inp), _expected_output(full))
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    palette = [c for c in range(10) if c != PLACEHOLDER]
    lattice, bg, sp1, sp2 = random.sample(palette, 4)
    cell_size = random.choice([5, 6])
    n_rows = random.randint(3, 4)
    n_cols = random.randint(4, 6)

    hidden_inside = random.choice([1, 2])
    if hidden_inside >= cell_size:
        return None
    prefix_w = cell_size - hidden_inside
    visible_width = 1 + (n_cols - 1) * (cell_size + 1) + prefix_w
    full_width = 1 + n_cols * (cell_size + 1)
    if visible_width >= full_width:
        return None

    majority = _make_majority_cell(cell_size, bg, sp1, sp2, hidden_inside)
    if majority is None:
        return None

    hidden_zone_cols = list(range(prefix_w, cell_size))
    visible_zone_cols = list(range(1, max(2, prefix_w)))

    hidden_odd = _variant_from_zone(majority, hidden_zone_cols, bg, sp1, sp2, visible_change=False)
    if hidden_odd is None:
        return None
    if not np.array_equal(hidden_odd[:, :prefix_w], majority[:, :prefix_w]):
        return None

    visible_variants: list[np.ndarray] = []
    for _ in range(6):
        var = _variant_from_zone(majority, visible_zone_cols, bg, sp1, sp2, visible_change=True)
        if var is None:
            continue
        if np.array_equal(var[:, :prefix_w], majority[:, :prefix_w]):
            continue
        sig = var.tobytes()
        if sig not in {hidden_odd.tobytes(), *(v.tobytes() for v in visible_variants)}:
            visible_variants.append(var)
    if not visible_variants:
        return None

    hidden_row = random.randrange(n_rows)
    visible_row_for_hidden = (hidden_row + 1) % n_rows
    row_defs: list[tuple[int, np.ndarray]] = []
    hidden_rows = {hidden_row}
    for r in range(n_rows):
        if r == hidden_row:
            row_defs.append((n_cols - 1, hidden_odd))
        elif r == visible_row_for_hidden:
            odd_col = random.randrange(n_cols - 1)
            row_defs.append((odd_col, hidden_odd))
        else:
            odd_col = random.randrange(n_cols - 1)
            row_defs.append((odd_col, random.choice(visible_variants)))

    rows: list[list[np.ndarray]] = []
    for odd_col, odd_cell in row_defs:
        row = [majority.copy() for _ in range(n_cols)]
        row[odd_col] = odd_cell.copy()
        rows.append(row)

    full = _render_lattice(rows, lattice, bg)
    c_cut = visible_width
    inp = full.copy()
    inp[:, c_cut:] = int(PLACEHOLDER)

    if not _semantic_checks(inp, full, majority, hidden_odd, hidden_rows, c_cut):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after repeated attempts")
