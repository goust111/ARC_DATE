from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _line_sequences(start: int, end: int) -> list[list[int]]:
    span = int(end - start)
    seqs: list[list[int]] = []
    for step in range(2, span + 1):
        if span % step != 0:
            continue
        count = span // step + 1
        if 4 <= count <= 5:
            seqs.append([int(start + k * step) for k in range(count)])
    if not seqs:
        for step in range(1, span + 1):
            if span % step == 0:
                seqs.append([int(start + k * step) for k in range(span // step + 1)])
    return seqs


def _infer_frame(gray_mask: np.ndarray) -> tuple[np.ndarray, list[int], list[int], tuple[int, int, int, int]]:
    coords = np.argwhere(gray_mask)
    assert coords.size > 0, "expected grey lattice pixels"
    r0, c0, r1, c1 = _bbox_coords(coords)
    row_seqs = _line_sequences(r0, r1)
    col_seqs = _line_sequences(c0, c1)

    best_key: tuple[int, int] | None = None
    best_rows: list[int] | None = None
    best_cols: list[int] | None = None

    for rows in row_seqs:
        row_mask = np.zeros_like(gray_mask, dtype=bool)
        row_mask[np.array(rows, dtype=int), c0 : c1 + 1] = True
        for cols in col_seqs:
            frame = row_mask.copy()
            frame[r0 : r1 + 1, np.array(cols, dtype=int)] = True
            missed = int(np.count_nonzero(gray_mask & ~frame))
            frame_size = int(np.count_nonzero(frame))
            key = (missed, frame_size)
            if best_key is None or key < best_key:
                best_key = key
                best_rows = rows
                best_cols = cols

    assert best_rows is not None and best_cols is not None
    frame = np.zeros_like(gray_mask, dtype=bool)
    frame[np.array(best_rows, dtype=int), c0 : c1 + 1] = True
    frame[r0 : r1 + 1, np.array(best_cols, dtype=int)] = True
    return frame, best_rows, best_cols, (r0, c0, r1, c1)


def _repair_frame(grid: np.ndarray, frame_mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    r0, c0, r1, c1 = bbox
    row_axis2 = r0 + r1
    col_axis2 = c0 + c1

    changed = True
    while changed:
        changed = False
        for r, c in np.argwhere(frame_mask & (out != GREY)):
            rr = row_axis2 - int(r)
            cc = col_axis2 - int(c)
            restore = False
            if r0 <= rr <= r1 and out[rr, c] == GREY:
                restore = True
            if c0 <= cc <= c1 and out[r, cc] == GREY:
                restore = True
            if restore:
                out[r, c] = GREY
                changed = True
    return out


def _fill_regions(base_grid: np.ndarray, frame_mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    out = np.array(base_grid, dtype=int, copy=True)
    r0, c0, r1, c1 = bbox
    inside = np.zeros_like(frame_mask, dtype=bool)
    inside[r0 : r1 + 1, c0 : c1 + 1] = True
    walk = inside & ~frame_mask
    for coords in _components(walk):
        br0, bc0, br1, bc1 = _bbox_coords(coords)
        fill = GREEN if int(coords.shape[0]) == (br1 - br0 + 1) * (bc1 - bc0 + 1) else YELLOW
        for r, c in coords:
            if out[int(r), int(c)] == BLACK:
                out[int(r), int(c)] = fill
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Infer the symmetric grey lattice, repair every mirror-supported wall gap, then color intact single cells green and persistent merged openings yellow.

    Input structure:
    - A black grid contains a one-pixel-thick grey rectangular lattice with an outer border and evenly spaced inner walls.
    - The intended lattice is symmetric about both the horizontal and vertical mid-axes, but some grey wall pixels are erased to black.
    - A few blue or red marker pixels may sit inside cell interiors; they are not part of the lattice and must be preserved.

    Rule:
    - Use only the surviving grey pixels to infer the lattice bbox, the two reflection axes, and the regularly spaced horizontal and vertical wall lines.
    - For every missing lattice position on those inferred wall lines, restore it when its reflection across the horizontal axis or the vertical axis is still grey; repeat until no more restorations are possible.
    - After repair, treat every non-grey cell inside the lattice as region space, find its 4-connected regions, and compare each region's area with its bounding-box area.
    - Fill black cells of a region green when the region exactly fills one rectangle, otherwise fill its black cells yellow; keep grey walls and existing red/blue markers unchanged.

    Key insight:
    - Repairable gaps are precisely the wall pixels whose symmetry orbit still has a surviving witness, while fully erased symmetric gaps remain open and merge neighboring cells into non-rectangular regions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame_mask, _rows, _cols, bbox = _infer_frame(g == GREY)
    repaired = _repair_frame(g, frame_mask, bbox)
    repaired_frame = repaired == GREY
    return _fill_regions(repaired, repaired_frame, bbox)


def _full_frame_mask(n_cell_rows: int, n_cell_cols: int, cell_h: int, cell_w: int) -> tuple[np.ndarray, list[int], list[int]]:
    h = n_cell_rows * cell_h + (n_cell_rows + 1)
    w = n_cell_cols * cell_w + (n_cell_cols + 1)
    row_lines = [0]
    for _ in range(n_cell_rows):
        row_lines.append(row_lines[-1] + cell_h + 1)
    col_lines = [0]
    for _ in range(n_cell_cols):
        col_lines.append(col_lines[-1] + cell_w + 1)

    frame = np.zeros((h, w), dtype=bool)
    frame[np.array(row_lines, dtype=int), :] = True
    frame[:, np.array(col_lines, dtype=int)] = True
    return frame, row_lines, col_lines


def _segment_candidates(
    n_cell_rows: int,
    n_cell_cols: int,
    cell_h: int,
    cell_w: int,
    row_lines: list[int],
    col_lines: list[int],
) -> list[np.ndarray]:
    parts: list[np.ndarray] = []

    for wall in range(1, n_cell_rows):
        r = row_lines[wall]
        for cell_col in range(n_cell_cols):
            c_start = col_lines[cell_col] + 1
            max_len = max(1, cell_w - 1)
            for length in range(1, max_len + 1):
                for offset in range(0, cell_w - length + 1):
                    coords = [(r, c_start + offset + k) for k in range(length)]
                    parts.append(np.array(coords, dtype=int))

    for wall in range(1, n_cell_cols):
        c = col_lines[wall]
        for cell_row in range(n_cell_rows):
            r_start = row_lines[cell_row] + 1
            max_len = max(1, cell_h - 1)
            for length in range(1, max_len + 1):
                for offset in range(0, cell_h - length + 1):
                    coords = [(r_start + offset + k, c) for k in range(length)]
                    parts.append(np.array(coords, dtype=int))

    return parts


def _orbit(coords: np.ndarray, row_axis2: int, col_axis2: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in coords:
        r = int(r)
        c = int(c)
        out.add((r, c))
        out.add((row_axis2 - r, c))
        out.add((r, col_axis2 - c))
        out.add((row_axis2 - r, col_axis2 - c))
    return out


def _place_markers(
    grid: np.ndarray,
    frame_mask: np.ndarray,
    row_lines: list[int],
    col_lines: list[int],
    rng: np.random.Generator,
) -> None:
    n_rows = len(row_lines) - 1
    n_cols = len(col_lines) - 1
    cells = [(rr, cc) for rr in range(n_rows) for cc in range(n_cols)]
    rng.shuffle(cells)
    marked_cells = int(rng.integers(2, min(len(cells), 7)))

    for cell_r, cell_c in cells[:marked_cells]:
        r_choices = list(range(row_lines[cell_r] + 1, row_lines[cell_r + 1]))
        c_choices = list(range(col_lines[cell_c] + 1, col_lines[cell_c + 1]))
        slots = [(r, c) for r in r_choices for c in c_choices if not frame_mask[r, c]]
        rng.shuffle(slots)
        n_marks = 1 if len(slots) == 1 else int(rng.integers(1, 3))
        for r, c in slots[:n_marks]:
            grid[r, c] = BLUE if rng.random() < 0.5 else RED


def _oracle_output(input_grid: np.ndarray, repaired_frame: np.ndarray) -> np.ndarray:
    bbox = (0, 0, repaired_frame.shape[0] - 1, repaired_frame.shape[1] - 1)
    base = np.full_like(input_grid, BLACK)
    base[repaired_frame] = GREY
    marker_mask = (input_grid == BLUE) | (input_grid == RED)
    base[marker_mask] = input_grid[marker_mask]
    return _fill_regions(base, repaired_frame, bbox)


def _try_generate(rng: np.random.Generator) -> np.ndarray | None:
    n_cell_rows = int(rng.choice([3, 4]))
    n_cell_cols = int(rng.choice([3, 4]))
    cell_h = int(rng.choice([2, 3, 4]))
    cell_w = int(rng.choice([2, 3, 4]))

    frame_mask, row_lines, col_lines = _full_frame_mask(n_cell_rows, n_cell_cols, cell_h, cell_w)
    h, w = frame_mask.shape
    row_axis2 = h - 1
    col_axis2 = w - 1

    candidates = _segment_candidates(n_cell_rows, n_cell_cols, cell_h, cell_w, row_lines, col_lines)
    rng.shuffle(candidates)

    reserved: set[tuple[int, int]] = set()
    removed_input: set[tuple[int, int]] = set()
    persistent_removed: set[tuple[int, int]] = set()
    non_sym_segments = 0
    sym_segments = 0

    target_non_sym = int(rng.integers(2, 5))
    target_sym = int(rng.integers(1, 3))

    for coords in candidates:
        orb = _orbit(coords, row_axis2, col_axis2)
        if orb & reserved:
            continue
        if len(orb) < 2:
            continue
        if sym_segments < target_sym and len(orb) >= 4 and rng.random() < 0.45:
            reserved |= orb
            removed_input |= orb
            persistent_removed |= orb
            sym_segments += 1
        elif non_sym_segments < target_non_sym:
            reserved |= orb
            removed_input |= {tuple(map(int, p)) for p in coords}
            non_sym_segments += 1
        if non_sym_segments >= target_non_sym and sym_segments >= target_sym:
            break

    if non_sym_segments < 2 or sym_segments < 1:
        return None

    observed_frame = frame_mask.copy()
    for r, c in removed_input:
        observed_frame[r, c] = False

    repaired_frame = frame_mask.copy()
    for r, c in persistent_removed:
        repaired_frame[r, c] = False

    if np.array_equal(observed_frame, repaired_frame):
        return None

    inp = np.full((h, w), BLACK, dtype=int)
    inp[observed_frame] = GREY
    _place_markers(inp, frame_mask, row_lines, col_lines, rng)

    expected = _oracle_output(inp, repaired_frame)
    before = _oracle_output(inp, observed_frame)
    if np.array_equal(expected, before):
        return None
    if not np.any(expected == GREEN) or not np.any(expected == YELLOW):
        return None

    repaired_count = int(np.count_nonzero(repaired_frame & ~observed_frame))
    persistent_count = int(np.count_nonzero(frame_mask & ~repaired_frame))
    if repaired_count < 2 or persistent_count < 1:
        return None

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        grid = _try_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
