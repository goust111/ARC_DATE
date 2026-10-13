from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
LABEL_COLORS = (1, 2, 3)
GREY = 5

HEIGHT_RANGE = (13, 17)
WIDTH_RANGE = (13, 17)
BAR_LENGTHS = (7, 9)
MAX_GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask as coordinate arrays."""
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.asarray(coords, dtype=int))
    return out


def _extract_bar(grid: np.ndarray) -> tuple[int, int, int]:
    """Find the unique horizontal grey bar inside the non-label area."""
    bar_mask = grid == GREY
    bar_mask[-1, :] = False
    bar_mask[:, -1] = False
    comps = _components(bar_mask)
    assert len(comps) == 1, "expected a single interior grey bar"
    coords = comps[0]
    rows = coords[:, 0]
    cols = coords[:, 1]
    assert int(rows.min()) == int(rows.max()), "grey object must be horizontal"
    bar_row = int(rows[0])
    c0 = int(cols.min())
    c1 = int(cols.max())
    assert int(coords.shape[0]) == c1 - c0 + 1, "grey bar must be solid"
    length = c1 - c0 + 1
    assert length >= 5 and length % 2 == 1, "grey bar must have odd length"
    return bar_row, c0, c1


def _diamond_masks(shape: tuple[int, int], bar_row: int, c0: int, c1: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return full diamond, boundary, bar, and strict interior masks."""
    h, w = map(int, shape)
    length = int(c1 - c0 + 1)
    radius = (length - 1) // 2
    center_col = (int(c0) + int(c1)) // 2

    rr, cc = np.indices((h, w))
    distance = np.abs(rr - int(bar_row)) + np.abs(cc - int(center_col))
    diamond = distance <= radius
    boundary = distance == radius
    bar = (rr == int(bar_row)) & (cc >= int(c0)) & (cc <= int(c1))
    strict_interior = diamond & ~boundary & ~bar

    for mask in (diamond, boundary, bar, strict_interior):
        mask[-1, :] = False
        mask[:, -1] = False
    return diamond, boundary, bar, strict_interior


def _expected_output(grid: np.ndarray) -> np.ndarray:
    """Semantic reference implementation used by main() and generator validation."""
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    assert h >= 5 and w >= 5, "grid must contain an interior plus label row/column"

    row_labels = g[:-1, -1]
    col_labels = g[-1, :-1]
    assert np.all(np.isin(row_labels, LABEL_COLORS)), "row labels must use the label palette"
    assert np.all(np.isin(col_labels, LABEL_COLORS)), "column labels must use the label palette"

    bar_row, c0, c1 = _extract_bar(g)
    diamond, boundary, bar, strict_interior = _diamond_masks(g.shape, bar_row, c0, c1)

    assert not diamond[-1, :].any() and not diamond[:, -1].any(), "diamond must stay out of label border"
    assert np.all(g[:-1, :-1][~bar[:-1, :-1]] == BACKGROUND), "non-bar interior must be black in the input"

    out = g.copy()
    out[boundary] = GREY

    fill = strict_interior[:-1, :-1] & (row_labels[:, None] == col_labels[None, :])
    out[:-1, :-1] = np.where(fill, row_labels[:, None], out[:-1, :-1])
    out[bar] = GREY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill only the label-matching cells that lie strictly inside the diamond induced by the grey bar.

    Input structure:
    - The bottom row (except the last cell) gives non-black column labels, and the rightmost column (except the last cell) gives non-black row labels.
    - The non-label interior is otherwise black except for one horizontal grey solid bar.
    - The grey bar sits fully inside the label border and determines a centered diamond envelope.

    Rule:
    - Keep the label row and label column unchanged.
    - Use the grey bar as the widest row of a diamond; the diamond shrinks by one cell at each end on every row above and below it, and its boundary is grey.
    - For each non-label cell, fill it with its shared row/column label color exactly when it is strictly inside that diamond and not on the boundary or the grey bar itself.

    Key insight:
    - The output is the boolean intersection of two independent constraints: geometric membership in the strict diamond interior and equality of the row/column labels.
    """
    return _expected_output(grid)


def _sample_label_sequence(length: int) -> np.ndarray:
    """Sample a varied label sequence using all three colors."""
    while True:
        seq = np.asarray([random.choice(LABEL_COLORS) for _ in range(int(length))], dtype=int)
        if len(set(map(int, seq.tolist()))) < len(LABEL_COLORS):
            continue
        changes = sum(int(seq[i] != seq[i + 1]) for i in range(len(seq) - 1))
        if changes < max(3, len(seq) // 4):
            continue
        return seq


def _sample_geometry() -> tuple[int, int, int, int, int]:
    """Sample grid size and a bar placement that leaves room around the diamond."""
    while True:
        height = random.randint(*HEIGHT_RANGE)
        width = random.randint(*WIDTH_RANGE)
        length = random.choice(BAR_LENGTHS)
        radius = (length - 1) // 2
        if height < 2 * radius + 5 or width < length + 4:
            continue

        row_min = radius + 1
        row_max = height - radius - 3
        col_min = 1
        col_max = width - length - 2
        if row_min > row_max or col_min > col_max:
            continue

        bar_row = random.randint(row_min, row_max)
        bar_col = random.randint(col_min, col_max)
        return height, width, bar_row, bar_col, length


def _candidate_counts(grid: np.ndarray) -> dict[str, object]:
    """Compute semantic coverage counts for anti-degenerate generation."""
    bar_row, c0, c1 = _extract_bar(grid)
    diamond, boundary, bar, strict_interior = _diamond_masks(grid.shape, bar_row, c0, c1)
    row_labels = grid[:-1, -1]
    col_labels = grid[-1, :-1]
    same = row_labels[:, None] == col_labels[None, :]

    inside_same = same & strict_interior[:-1, :-1]
    outside_same = same & ~diamond[:-1, :-1]
    blocked_same = same & (boundary[:-1, :-1] | bar[:-1, :-1])
    inside_diff = (~same) & strict_interior[:-1, :-1]

    inside_colors = {int(row_labels[r]) for r, c in np.argwhere(inside_same)}
    return {
        "inside_same": int(inside_same.sum()),
        "outside_same": int(outside_same.sum()),
        "blocked_same": int(blocked_same.sum()),
        "inside_diff": int(inside_diff.sum()),
        "inside_color_count": len(inside_colors),
    }


def _valid_candidate(grid: np.ndarray) -> bool:
    """Reject generators that weaken either the geometry or the label constraint."""
    counts = _candidate_counts(grid)
    return (
        counts["inside_same"] >= 2
        and counts["outside_same"] >= 1
        and counts["blocked_same"] >= 1
        and counts["inside_diff"] >= 2
        and counts["inside_color_count"] >= 2
    )


def _generate_candidate() -> np.ndarray:
    """Construct one candidate input grid."""
    height, width, bar_row, bar_col, length = _sample_geometry()
    grid = np.full((height, width), BACKGROUND, dtype=int)
    grid[bar_row, bar_col : bar_col + length] = GREY
    grid[:-1, -1] = _sample_label_sequence(height - 1)
    grid[-1, :-1] = _sample_label_sequence(width - 1)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _generate_candidate()
        if not _valid_candidate(candidate):
            continue
        expected = _expected_output(candidate)
        if np.array_equal(main(candidate), expected):
            return candidate
    raise RuntimeError("failed to generate a valid input for puzzle 10")
