from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
MAX_ATTEMPTS = 500
GROUP_WIDTH_PATTERNS = (
    (1, 1, 1, 1),
    (1, 2, 1, 1),
    (1, 1, 2, 1),
    (1, 2, 2, 1),
)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _shift_for_distance(distance_from_bottom: int) -> int:
    if distance_from_bottom % 2 == 0:
        return 0
    return -1 if distance_from_bottom % 4 == 1 else 1


def _apply_wave_shift(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    mask = g != BG
    if not np.any(mask):
        return g.copy()

    box = _bbox(mask)
    assert box is not None
    r0, _c0, r1, _c1 = box

    out = np.full_like(g, BG)
    for r in range(r0, r1 + 1):
        cols = np.flatnonzero(mask[r])
        if cols.size == 0:
            continue
        shift = _shift_for_distance(r1 - r)
        new_cols = cols + shift
        keep = (new_cols >= 0) & (new_cols < g.shape[1])
        if np.any(keep):
            out[r, new_cols[keep]] = g[r, cols[keep]]
    return out


def _marker_pair_count(marker_mask: np.ndarray) -> int:
    h, w = marker_mask.shape
    pairs = 0
    for r in range(h):
        count = int(np.count_nonzero(marker_mask[r]))
        if count >= 2:
            pairs += count * (count - 1) // 2
    for c in range(w):
        count = int(np.count_nonzero(marker_mask[:, c]))
        if count >= 2:
            pairs += count * (count - 1) // 2
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Wave-shift the foreground lattice from its bottom row and then connect the shifted red markers by row/column spans.

    Input structure:
    - A black grid contains one 4-connected lattice-like foreground object made of blue cells, with 3-5 red marker cells embedded in that same object.
    - The object is interpreted as the union of blue and red cells, and its bounding-box bottom row acts as the fixed anchor for the horizontal wave.
    - The red markers are sparse clues: in the intended inputs they are not aligned in the original coordinates, but they become aligned after the row-dependent shift.

    Rule:
    - Find the bounding box of all non-black cells and keep its bottom row fixed.
    - For a row at distance k above that bottom row, shift the whole row by 0 when k is even, by 1 left when k mod 4 = 1, and by 1 right when k mod 4 = 3; clip any cells that leave the grid.
    - After shifting, look only at the shifted red marker positions: each row with at least two shifted red markers is filled red between its leftmost and rightmost markers, and each column with at least two shifted red markers is filled red between its topmost and bottommost markers.

    Key insight:
    - The red spans are not determined in the original lattice; the hidden alignments appear only after the bottom-anchored alternating row shift, so the shift and the connector rule must be inferred together.
    """
    shifted = _apply_wave_shift(np.asarray(grid, dtype=int))
    markers = shifted == RED

    out = shifted.copy()
    h, w = out.shape
    for r in range(h):
        cols = np.flatnonzero(markers[r])
        if cols.size >= 2:
            out[r, cols[0] : cols[-1] + 1] = RED
    for c in range(w):
        rows = np.flatnonzero(markers[:, c])
        if rows.size >= 2:
            out[rows[0] : rows[-1] + 1, c] = RED
    return out


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return count


def _build_lattice_patch(
    cell_height: int,
    cell_width: int,
    group_widths: tuple[int, ...],
) -> tuple[np.ndarray, list[int], list[int]]:
    n_bars = 4
    height = n_bars + (n_bars - 1) * int(cell_height)
    width = sum(group_widths) + (len(group_widths) - 1) * int(cell_width)
    patch = np.full((height, width), BG, dtype=int)

    bar_rows = [i * (int(cell_height) + 1) for i in range(n_bars)]
    patch[bar_rows, :] = BLUE

    vertical_cols: list[int] = []
    cursor = 0
    for group_width in group_widths:
        cols = list(range(cursor, cursor + int(group_width)))
        vertical_cols.extend(cols)
        patch[:, cursor : cursor + int(group_width)] = BLUE
        cursor += int(group_width) + int(cell_width)

    return patch, bar_rows, vertical_cols


def _rows_by_shift(height: int, rows: list[int]) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for r in rows:
        shift = _shift_for_distance(height - 1 - int(r))
        groups[shift].append(int(r))
    return groups


def _interior_object_cells(patch: np.ndarray) -> list[tuple[int, int]]:
    h, w = patch.shape
    cells = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(patch[r, c]) == BLUE:
                cells.append((r, c))
    return cells


def _pick_triple(
    width: int,
    shift_rows: dict[int, list[int]],
) -> list[tuple[int, int]] | None:
    if not shift_rows[-1] or not shift_rows[0] or not shift_rows[1] or width < 5:
        return None
    r_neg = random.choice(shift_rows[-1])
    r_zero = random.choice(shift_rows[0])
    r_pos = random.choice(shift_rows[1])

    targets = list(range(2, width - 2))
    random.shuffle(targets)
    for target in targets:
        markers = [(r_neg, target + 1), (r_zero, target), (r_pos, target - 1)]
        cols = [c for _r, c in markers]
        if len(set(cols)) == 3:
            return markers
    return None


def _pick_pair_pair(
    width: int,
    shift_rows: dict[int, list[int]],
) -> list[tuple[int, int]] | None:
    if not shift_rows[-1] or len(shift_rows[0]) < 2 or not shift_rows[1] or width < 7:
        return None
    r_neg = random.choice(shift_rows[-1])
    r_pos = random.choice(shift_rows[1])
    zero_rows = shift_rows[0][:]
    random.shuffle(zero_rows)
    r_zero_a, r_zero_b = zero_rows[:2]

    targets = [(t1, t2) for t1 in range(2, width - 4) for t2 in range(t1 + 3, width - 1)]
    random.shuffle(targets)
    for target_a, target_b in targets:
        markers = [
            (r_neg, target_a + 1),
            (r_zero_a, target_a),
            (r_pos, target_b - 1),
            (r_zero_b, target_b),
        ]
        cols = [c for _r, c in markers]
        if len(set(cols)) == 4:
            return markers
    return None


def _pick_bar_isolated(
    width: int,
    row: int,
    used_cols: set[int],
    blocked_shifted_cols: set[int],
) -> tuple[int, int] | None:
    candidates = list(range(1, width - 1))
    random.shuffle(candidates)
    for col in candidates:
        if col in used_cols or col in blocked_shifted_cols:
            continue
        return row, col
    return None


def _pick_nonbar_isolated(
    patch: np.ndarray,
    vertical_cols: list[int],
    used_rows: set[int],
    used_cols: set[int],
    blocked_shifted_cols: set[int],
) -> tuple[int, int] | None:
    h, w = patch.shape
    candidates: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        if r in used_rows:
            continue
        shift = _shift_for_distance(h - 1 - r)
        for c in vertical_cols:
            if c <= 0 or c >= w - 1:
                continue
            if c in used_cols or int(patch[r, c]) != BLUE:
                continue
            shifted_col = c + shift
            if 0 <= shifted_col < w and shifted_col not in blocked_shifted_cols:
                candidates.append((r, c))
    if not candidates:
        return None
    return random.choice(candidates)


def _choose_markers(
    patch: np.ndarray,
    bar_rows: list[int],
    vertical_cols: list[int],
    mode: str,
) -> list[tuple[int, int]] | None:
    h, w = patch.shape
    shift_rows = _rows_by_shift(h, bar_rows)

    if mode == "triple3":
        return _pick_triple(w, shift_rows)

    if mode == "pair4":
        return _pick_pair_pair(w, shift_rows)

    triple = _pick_triple(w, shift_rows)
    if triple is None:
        return None

    used_rows = {r for r, _c in triple}
    used_cols = {c for _r, c in triple}
    triple_shifted = {c + _shift_for_distance(h - 1 - r) for r, c in triple}
    spare_zero_rows = [r for r in shift_rows[0] if r not in used_rows]
    if not spare_zero_rows:
        return None

    bar_iso = _pick_bar_isolated(w, spare_zero_rows[0], used_cols, triple_shifted)
    if bar_iso is None:
        return None
    markers = triple + [bar_iso]

    if mode == "triple4":
        return markers

    used_rows = {r for r, _c in markers}
    used_cols = {c for _r, c in markers}
    blocked_shifted = {c + _shift_for_distance(h - 1 - r) for r, c in markers}
    nonbar_iso = _pick_nonbar_isolated(patch, vertical_cols, used_rows, used_cols, blocked_shifted)
    if nonbar_iso is None:
        return None
    return markers + [nonbar_iso]


def _place_patch(patch: np.ndarray) -> np.ndarray | None:
    area = int(np.count_nonzero(patch))
    h, w = patch.shape
    margin_options = [
        (top, bottom, left, right)
        for top in (0, 1)
        for bottom in (0, 1)
        for left in (1, 2)
        for right in (1, 2)
    ]
    random.shuffle(margin_options)
    for top, bottom, left, right in margin_options:
        grid_h = h + top + bottom
        grid_w = w + left + right
        ratio = area / float(grid_h * grid_w)
        if 0.40 <= ratio <= 0.60:
            grid = np.full((grid_h, grid_w), BG, dtype=int)
            grid[top : top + h, left : left + w] = patch
            return grid
    return None


def _valid_input(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False
    if not np.all(np.isin(g, [BG, BLUE, RED])):
        return False

    mask = g != BG
    if _component_count(mask) != 1:
        return False

    box = _bbox(mask)
    if box is None:
        return False
    r0, c0, r1, c1 = box
    for r in range(r0, r1 + 1):
        if not np.any(mask[r, c0 : c1 + 1]):
            return False

    markers = np.argwhere(g == RED)
    if not 3 <= len(markers) <= 5:
        return False
    rows = [int(r) for r, _c in markers]
    cols = [int(c) for _r, c in markers]
    if len(set(rows)) != len(rows) or len(set(cols)) != len(cols):
        return False

    ratio = int(mask.sum()) / float(g.size)
    if not (0.40 <= ratio <= 0.60):
        return False

    shifted = _apply_wave_shift(g)
    shifted_markers = shifted == RED
    if _marker_pair_count(shifted_markers) < 2:
        return False

    out = main(g)
    if np.array_equal(out, g):
        return False
    if int(np.count_nonzero(out == RED)) <= int(np.count_nonzero(g == RED)):
        return False
    return True


def _attempt_instance() -> np.ndarray | None:
    cell_height = random.choice([2, 4])
    cell_width = random.choice([1, 2, 2])
    group_widths = random.choice(GROUP_WIDTH_PATTERNS)
    patch, bar_rows, vertical_cols = _build_lattice_patch(cell_height, cell_width, group_widths)

    mode = random.choice(["triple3", "pair4", "triple4", "triple5"])
    markers = _choose_markers(patch, bar_rows, vertical_cols, mode)
    if markers is None:
        return None

    patch = patch.copy()
    for r, c in markers:
        patch[int(r), int(c)] = RED

    grid = _place_patch(patch)
    if grid is None or not _valid_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_instance()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 5")
