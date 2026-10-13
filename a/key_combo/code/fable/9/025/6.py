from __future__ import annotations

import numpy as np


BLACK = 0
PINK = 6


def _bbox_of_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise AssertionError("expected a non-empty mask")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _panel_masks_and_legend(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.size == 0:
        raise AssertionError("input must be a non-empty 2D grid")

    colors = set(map(int, np.unique(g)))
    if PINK not in colors:
        raise AssertionError("legend must contain pink")

    panel_colors = sorted(c for c in colors if c not in {BLACK, PINK})
    if len(panel_colors) != 2:
        raise AssertionError("expected exactly two non-black, non-pink panel colors")

    panels = []
    for color in panel_colors:
        bbox = _bbox_of_mask(g == color)
        r0, c0, r1, c1 = bbox
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h != w:
            raise AssertionError("each panel must be square")
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        non_black = set(map(int, np.unique(patch))) - {BLACK}
        if non_black != {color}:
            raise AssertionError("each panel may contain only its own foreground color")
        panels.append((bbox, patch == color))

    panels.sort(key=lambda item: (item[0][1], item[0][0]))
    (left_bbox, left_mask), (right_bbox, right_mask) = panels
    lr0, lc0, lr1, lc1 = left_bbox
    rr0, rc0, rr1, rc1 = right_bbox
    panel_size = lr1 - lr0 + 1
    if left_mask.shape != right_mask.shape:
        raise AssertionError("panels must have equal size")
    if (rr1 - rr0 + 1) != panel_size:
        raise AssertionError("panels must have equal size")
    if lr0 != rr0 or lr1 != rr1:
        raise AssertionError("panels must be vertically aligned")
    if rc0 <= lc1:
        raise AssertionError("panels must be side-by-side without overlap")
    if np.any(g[lr0 : lr1 + 1, lc0 : lc1 + 1] == PINK) or np.any(g[rr0 : rr1 + 1, rc0 : rc1 + 1] == PINK):
        raise AssertionError("pink is reserved for the legend")

    pink_mask = g == PINK
    pr0, pc0, pr1, pc1 = _bbox_of_mask(pink_mask)
    if pr1 - pr0 > 1 or pc1 - pc0 > 1:
        raise AssertionError("pink cells must fit inside one 2x2 legend")

    H, W = g.shape
    pair_center = (lc0 + rc1) / 2.0
    expected_legend_c0 = (lc0 + rc1 - 1) // 2
    row_lo = max(0, pr1 - 1)
    row_hi = min(pr0, H - 2)
    col_lo = max(0, pc1 - 1)
    col_hi = min(pc0, W - 2)
    candidates: list[tuple[float, int, int, int, int]] = []
    for r0 in range(row_lo, row_hi + 1):
        for c0 in range(col_lo, col_hi + 1):
            if r0 + 2 >= lr0:
                continue
            window = pink_mask[r0 : r0 + 2, c0 : c0 + 2]
            if int(window.sum()) != int(pink_mask.sum()):
                continue
            col_anchor_delta = abs(c0 - expected_legend_c0)
            center_delta = abs((c0 + 0.5) - pair_center)
            vertical_gap = lr0 - (r0 + 2)
            candidates.append((col_anchor_delta, center_delta, vertical_gap, -r0, r0, c0))
    if not candidates:
        raise AssertionError("failed to locate the 2x2 legend")

    _, _, _, _, legend_r0, legend_c0 = min(candidates)
    legend_patch = g[legend_r0 : legend_r0 + 2, legend_c0 : legend_c0 + 2]
    if not set(map(int, np.unique(legend_patch))) <= {BLACK, PINK}:
        raise AssertionError("legend must use only black and pink")

    return left_mask, right_mask, legend_patch == PINK


def _apply_legend(left_mask: np.ndarray, right_mask: np.ndarray, legend: np.ndarray) -> np.ndarray:
    out = np.full(left_mask.shape, BLACK, dtype=int)
    both = left_mask & right_mask
    left_only = left_mask & ~right_mask
    right_only = ~left_mask & right_mask
    neither = ~left_mask & ~right_mask
    if bool(legend[0, 0]):
        out[both] = PINK
    if bool(legend[0, 1]):
        out[left_only] = PINK
    if bool(legend[1, 0]):
        out[right_only] = PINK
    if bool(legend[1, 1]):
        out[neither] = PINK
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Decode the 2x2 truth-table legend and merge the two side-by-side masks into a pink result.

    Input structure:
    - The grid is a black background containing two equal square panels placed side by side.
    - Each panel uses exactly one non-black foreground color, and the two panel colors differ.
    - Above the panels, separated by black space, sits a 2x2 legend made only of pink and black cells.

    Rule:
    - Align the two panel masks cell by cell.
    - Interpret the legend as a truth table: top-left = both panels occupied, top-right = only left occupied,
      bottom-left = only right occupied, bottom-right = both empty.
    - Output a single square panel of the same size, painting a cell pink exactly when the legend marks that
      occupancy case as foreground; otherwise paint it black.

    Key insight:
    - The top marker is not decoration: it encodes which boolean overlay operator to use, so the same pair of
      masks can yield different outputs in different examples.
    """
    left_mask, right_mask, legend = _panel_masks_and_legend(grid)
    return _apply_legend(left_mask, right_mask, legend)


_VISIBLE_OPERATORS = (
    np.array([[0, 1], [1, 0]], dtype=bool),  # xor
    np.array([[1, 0], [0, 1]], dtype=bool),  # xnor
    np.array([[0, 1], [1, 1]], dtype=bool),  # nand
    np.array([[1, 0], [1, 1]], dtype=bool),  # everything except left-only
    np.array([[1, 1], [0, 1]], dtype=bool),  # everything except right-only
    np.array([[1, 1], [1, 0]], dtype=bool),  # or
)


def _legend_depends_on_both_inputs(legend: np.ndarray) -> bool:
    depends_on_left = bool(legend[0, 0] != legend[1, 0] or legend[0, 1] != legend[1, 1])
    depends_on_right = bool(legend[0, 0] != legend[0, 1] or legend[1, 0] != legend[1, 1])
    return depends_on_left and depends_on_right


def _legend_is_visually_explicit(legend: np.ndarray) -> bool:
    return bool(legend.any() and (~legend).any() and legend[0].any() and legend[1].any() and legend[:, 0].any() and legend[:, 1].any())


def _sample_state_grid(size: int) -> np.ndarray | None:
    for _ in range(200):
        weights = np.random.uniform(0.7, 1.4, size=4)
        probs = weights / weights.sum()
        states = np.random.choice(4, size=(size, size), p=probs)
        if len(set(map(int, np.unique(states)))) != 4:
            continue

        left_mask = (states == 2) | (states == 3)
        right_mask = (states == 1) | (states == 3)
        if not (left_mask[0].any() and left_mask[-1].any() and left_mask[:, 0].any() and left_mask[:, -1].any()):
            continue
        if not (right_mask[0].any() and right_mask[-1].any() and right_mask[:, 0].any() and right_mask[:, -1].any()):
            continue

        left_count = int(left_mask.sum())
        right_count = int(right_mask.sum())
        if not (size + 2 <= left_count <= size * size - 2):
            continue
        if not (size + 2 <= right_count <= size * size - 2):
            continue
        return states
    return None


def _render_input(left_mask: np.ndarray, right_mask: np.ndarray, legend: np.ndarray) -> np.ndarray:
    size = int(left_mask.shape[0])
    left_color, right_color = np.random.choice([1, 2, 3, 4, 5, 7, 8, 9], size=2, replace=False)

    h_gap = int(np.random.randint(1, 5))
    v_gap = int(np.random.randint(1, 4))
    top_margin = int(np.random.randint(0, 3))
    left_margin = int(np.random.randint(0, 4))
    bottom_margin = int(np.random.randint(0, 3))
    right_margin = int(np.random.randint(0, 4))

    left_r0 = top_margin + 2 + v_gap
    left_c0 = left_margin
    right_c0 = left_c0 + size + h_gap
    right_c1 = right_c0 + size - 1
    legend_c0 = (left_c0 + right_c1 - 1) // 2
    legend_r0 = top_margin

    H = left_r0 + size + bottom_margin
    W = right_c0 + size + right_margin
    if max(H, W) > 30:
        raise AssertionError("generated grid exceeds ARC size limit")

    grid = np.full((H, W), BLACK, dtype=int)
    grid[left_r0 : left_r0 + size, left_c0 : left_c0 + size][left_mask] = int(left_color)
    grid[left_r0 : left_r0 + size, right_c0 : right_c0 + size][right_mask] = int(right_color)
    grid[legend_r0 : legend_r0 + 2, legend_c0 : legend_c0 + 2][legend] = PINK
    return grid


def _semantic_checks(grid: np.ndarray, legend: np.ndarray) -> bool:
    left_mask, right_mask, parsed_legend = _panel_masks_and_legend(grid)
    if not np.array_equal(parsed_legend, legend):
        return False

    states = set()
    for left, right in zip(left_mask.flat, right_mask.flat):
        states.add((int(bool(left)), int(bool(right))))
    if states != {(0, 0), (0, 1), (1, 0), (1, 1)}:
        return False
    if not _legend_depends_on_both_inputs(parsed_legend):
        return False
    if not _legend_is_visually_explicit(parsed_legend):
        return False

    out = main(grid)
    if out.shape != left_mask.shape:
        return False
    if not ({BLACK, PINK} >= set(map(int, np.unique(out)))):
        return False
    if not (np.any(out == BLACK) and np.any(out == PINK)):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        size = int(np.random.randint(4, 9))
        states = _sample_state_grid(size)
        if states is None:
            continue

        legend = _VISIBLE_OPERATORS[int(np.random.randint(len(_VISIBLE_OPERATORS)))]
        left_mask = (states == 2) | (states == 3)
        right_mask = (states == 1) | (states == 3)
        grid = _render_input(left_mask, right_mask, legend)

        if not _semantic_checks(grid, legend):
            continue
        return grid
    raise RuntimeError("failed to generate a valid truth-table overlay input")
