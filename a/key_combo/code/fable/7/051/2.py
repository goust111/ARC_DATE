from __future__ import annotations

from collections import Counter

import numpy as np


BG = 0
PLACEHOLDER = 3
LABEL_COLORS = (1, 2, 4, 5, 6, 7, 8, 9)
UNKNOWN = -1
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _assign(arr: np.ndarray, idx: int, color: int, what: str) -> None:
    color = int(color)
    if arr[idx] == UNKNOWN:
        arr[idx] = color
        return
    assert int(arr[idx]) == color, f"conflicting {what} color inference"


def _infer_labels(grid: np.ndarray, bg: int, placeholder: int) -> tuple[np.ndarray, np.ndarray]:
    inner_h = int(grid.shape[0] - 1)
    inner_w = int(grid.shape[1] - 1)
    row_labels = np.full(inner_h, UNKNOWN, dtype=int)
    col_labels = np.full(inner_w, UNKNOWN, dtype=int)

    visible_row_labels = grid[:-1, -1]
    visible_col_labels = grid[-1, :-1]

    for r in range(inner_h):
        color = int(visible_row_labels[r])
        if color != placeholder:
            _assign(row_labels, r, color, "row label")

    for c in range(inner_w):
        color = int(visible_col_labels[c])
        if color != placeholder:
            _assign(col_labels, c, color, "column label")

    interior = grid[:-1, :-1]
    for r in range(inner_h):
        for c in range(inner_w):
            color = int(interior[r, c])
            if color in (bg, placeholder):
                continue
            _assign(row_labels, r, color, "row label")
            _assign(col_labels, c, color, "column label")

    assert np.all(row_labels != UNKNOWN), "some row labels cannot be recovered"
    assert np.all(col_labels != UNKNOWN), "some column labels cannot be recovered"

    for r in range(inner_h):
        for c in range(inner_w):
            color = int(interior[r, c])
            if color == placeholder:
                continue
            expected = int(row_labels[r]) if int(row_labels[r]) == int(col_labels[c]) else bg
            assert color == expected, "visible interior contradicts equality-table rule"

    return row_labels, col_labels


def main(grid: np.ndarray) -> np.ndarray:
    """Recover hidden labels and rebuild the equality table under the green placeholders.

    Input structure:
    - The grid consists of an interior table, one rightmost row-label column, one bottom column-label row, and a background bottom-right corner.
    - Interior cells follow an equality-table pattern: a cell is colored exactly when its row label and column label are the same color.
    - Some labels and some interior cells are replaced by green placeholder cells.

    Rule:
    - Read visible labels directly from the right column and bottom row.
    - Use every visible non-background interior cell as an equation that fixes both its row label and its column label to that cell color.
    - After all labels are recovered, recompute the entire interior table from the equality rule and output the fully repaired grid.

    Key insight:
    - The interior pattern is not an independent drawing; each visible colored intersection pins the shared label color of its row and column, so the hidden labels must be solved by inverting the generation rule.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 2

    bg = _get_bg(g)
    placeholder = PLACEHOLDER

    row_labels, col_labels = _infer_labels(g, bg, placeholder)
    out = np.full_like(g, bg)
    out[:-1, -1] = row_labels
    out[-1, :-1] = col_labels
    out[:-1, :-1] = np.where(row_labels[:, None] == col_labels[None, :], row_labels[:, None], bg)
    out[-1, -1] = bg
    return out


def _sample_sequence(length: int, palette: tuple[int, ...]) -> np.ndarray:
    base = list(palette)
    extras = []
    if length > len(palette):
        extras = list(np.random.choice(palette, size=length - len(palette), replace=True))
    seq = np.array(base + extras, dtype=int)
    np.random.shuffle(seq)
    return seq


def _choose_palette(rows: int, cols: int) -> tuple[int, ...]:
    max_colors = min(4, rows, cols)
    color_count = int(np.random.randint(2, max_colors + 1))
    palette = list(np.random.choice(LABEL_COLORS, size=color_count, replace=False))
    return tuple(int(x) for x in palette)


def _full_grid(row_labels: np.ndarray, col_labels: np.ndarray) -> np.ndarray:
    rows = int(row_labels.shape[0])
    cols = int(col_labels.shape[0])
    out = np.full((rows + 1, cols + 1), BG, dtype=int)
    out[:-1, -1] = row_labels
    out[-1, :-1] = col_labels
    out[:-1, :-1] = np.where(row_labels[:, None] == col_labels[None, :], row_labels[:, None], BG)
    return out


def _pick_joint_pair(row_labels: np.ndarray, col_labels: np.ndarray) -> tuple[int, int]:
    pairs = [(r, c) for r in range(len(row_labels)) for c in range(len(col_labels)) if int(row_labels[r]) == int(col_labels[c])]
    assert pairs
    idx = int(np.random.randint(len(pairs)))
    return pairs[idx]


def _hidden_indices(length: int, required_idx: int) -> set[int]:
    hidden = {int(required_idx)}
    max_extra = max(0, length - 2)
    extra_count = int(np.random.randint(0, max_extra + 1))
    choices = [idx for idx in range(length) if idx != required_idx]
    if extra_count and choices:
        picks = np.random.choice(choices, size=extra_count, replace=False)
        hidden.update(int(x) for x in np.atleast_1d(picks))
    return hidden


def _build_anchor_set(
    row_labels: np.ndarray,
    col_labels: np.ndarray,
    hidden_rows: set[int],
    hidden_cols: set[int],
    joint_pair: tuple[int, int],
) -> set[tuple[int, int]]:
    anchors = {joint_pair}

    for r in hidden_rows:
        same_cols = [c for c in range(len(col_labels)) if int(col_labels[c]) == int(row_labels[r])]
        hidden_same_cols = [c for c in same_cols if c in hidden_cols]
        choices = hidden_same_cols or same_cols
        anchors.add((int(r), int(choices[int(np.random.randint(len(choices)))])))

    for c in hidden_cols:
        same_rows = [r for r in range(len(row_labels)) if int(row_labels[r]) == int(col_labels[c])]
        hidden_same_rows = [r for r in same_rows if r in hidden_rows]
        choices = hidden_same_rows or same_rows
        anchors.add((int(choices[int(np.random.randint(len(choices)))]), int(c)))

    return anchors


def _choose_internal_placeholders(
    row_labels: np.ndarray,
    col_labels: np.ndarray,
    anchors: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    rows = len(row_labels)
    cols = len(col_labels)
    match_cells = []
    mismatch_cells = []
    for r in range(rows):
        for c in range(cols):
            if (r, c) in anchors:
                continue
            if int(row_labels[r]) == int(col_labels[c]):
                match_cells.append((r, c))
            else:
                mismatch_cells.append((r, c))

    if not match_cells or not mismatch_cells:
        return None

    placeholders: set[tuple[int, int]] = set()
    placeholders.add(match_cells[int(np.random.randint(len(match_cells)))])
    placeholders.add(mismatch_cells[int(np.random.randint(len(mismatch_cells)))])

    all_candidates = match_cells + mismatch_cells
    np.random.shuffle(all_candidates)
    target_count = int(np.random.randint(2, max(3, min(len(all_candidates), (rows * cols) // 2) + 1)))
    for cell in all_candidates:
        placeholders.add(cell)
        if len(placeholders) >= target_count:
            break

    return placeholders


def _validate_generated(
    inp: np.ndarray,
    out: np.ndarray,
    hidden_rows: set[int],
    hidden_cols: set[int],
    anchors: set[tuple[int, int]],
    placeholders: set[tuple[int, int]],
    joint_pair: tuple[int, int],
) -> bool:
    if not hidden_rows or not hidden_cols or not placeholders:
        return False

    if int(inp[joint_pair[0], -1]) != PLACEHOLDER or int(inp[-1, joint_pair[1]]) != PLACEHOLDER:
        return False
    if int(inp[joint_pair]) in (BG, PLACEHOLDER):
        return False

    for r in hidden_rows:
        visible = [int(inp[r, c]) for c in range(inp.shape[1] - 1) if int(inp[r, c]) not in (BG, PLACEHOLDER)]
        if not visible or set(visible) != {int(out[r, -1])}:
            return False

    for c in hidden_cols:
        visible = [int(inp[r, c]) for r in range(inp.shape[0] - 1) if int(inp[r, c]) not in (BG, PLACEHOLDER)]
        if not visible or set(visible) != {int(out[-1, c])}:
            return False

    has_hidden_match = any(int(out[r, c]) != BG for r, c in placeholders)
    has_hidden_mismatch = any(int(out[r, c]) == BG for r, c in placeholders)
    if not (has_hidden_match and has_hidden_mismatch):
        return False

    for r, c in anchors:
        if int(inp[r, c]) in (BG, PLACEHOLDER):
            return False

    try:
        produced = main(inp)
    except Exception:
        return False
    return np.array_equal(produced, out)


def _generate_once() -> np.ndarray | None:
    rows = int(np.random.randint(4, 8))
    cols = int(np.random.randint(4, 8))

    palette = _choose_palette(rows, cols)
    row_labels = _sample_sequence(rows, palette)
    col_labels = _sample_sequence(cols, palette)

    if len(set(map(int, row_labels))) < 2 or len(set(map(int, col_labels))) < 2:
        return None
    row_counts = Counter(map(int, row_labels))
    col_counts = Counter(map(int, col_labels))
    if max(row_counts.values()) == 1 and max(col_counts.values()) == 1:
        return None

    out = _full_grid(row_labels, col_labels)
    joint_pair = _pick_joint_pair(row_labels, col_labels)

    hidden_rows = _hidden_indices(rows, joint_pair[0])
    hidden_cols = _hidden_indices(cols, joint_pair[1])
    anchors = _build_anchor_set(row_labels, col_labels, hidden_rows, hidden_cols, joint_pair)
    placeholders = _choose_internal_placeholders(row_labels, col_labels, anchors)
    if placeholders is None:
        return None

    inp = out.copy()
    for r in hidden_rows:
        inp[r, -1] = PLACEHOLDER
    for c in hidden_cols:
        inp[-1, c] = PLACEHOLDER
    for r, c in placeholders:
        inp[r, c] = PLACEHOLDER

    if not _validate_generated(inp, out, hidden_rows, hidden_cols, anchors, placeholders, joint_pair):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid equality-table inversion instance")
