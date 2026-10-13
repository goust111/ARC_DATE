from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BG = 0
NON_BG_COLORS = list(range(1, 10))
MAX_GENERATION_ATTEMPTS = 800


def _mirror_mask(mask: np.ndarray, axis_k: int) -> np.ndarray | None:
    h, w = mask.shape
    out = np.zeros((h, w), dtype=bool)
    rows, cols = np.where(mask)
    mirrored_cols = int(axis_k) - cols
    if np.any((mirrored_cols < 0) | (mirrored_cols >= w)):
        return None
    out[rows, mirrored_cols] = True
    return out


def _axis_completion(mask: np.ndarray, axis_k: int) -> tuple[int, int, np.ndarray] | None:
    mirrored = _mirror_mask(mask, axis_k)
    if mirrored is None:
        return None
    rows, cols = np.where(mask)
    matched = int(mask[rows, axis_k - cols].sum())
    completed = mask | mirrored
    added = int(completed.sum() - mask.sum())
    return matched, -added, completed


def _best_vertical_axis(mask: np.ndarray) -> tuple[int, np.ndarray, tuple[int, int], bool]:
    _h, w = mask.shape
    best_k = None
    best_completed = None
    best_score = None
    tied = False
    for axis_k in range(0, 2 * (w - 1) + 1):
        result = _axis_completion(mask, axis_k)
        if result is None:
            continue
        matched, neg_added, completed = result
        score = (matched, neg_added)
        if best_score is None or score > best_score:
            best_k = axis_k
            best_completed = completed
            best_score = score
            tied = False
        elif score == best_score:
            tied = True
    assert best_k is not None and best_completed is not None and best_score is not None
    return int(best_k), best_completed, best_score, tied


def _infer_front_edges(
    grid: np.ndarray,
    layer_masks: dict[int, np.ndarray],
) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for color_a, color_b in combinations(sorted(layer_masks), 2):
        overlap = layer_masks[color_a] & layer_masks[color_b]
        if not overlap.any():
            continue
        witnesses = set(map(int, np.unique(grid[overlap]).tolist()))
        witnesses.discard(BG)
        a_visible = color_a in witnesses
        b_visible = color_b in witnesses
        if a_visible and not b_visible:
            edges.add((color_a, color_b))
        elif b_visible and not a_visible:
            edges.add((color_b, color_a))
    return edges


def _unique_front_order(colors: list[int], edges: set[tuple[int, int]]) -> list[int] | None:
    remaining = set(colors)
    incoming = {color: set() for color in colors}
    for front, back in edges:
        incoming[back].add(front)

    order: list[int] = []
    while remaining:
        front_candidates = [color for color in remaining if not (incoming[color] & remaining)]
        if len(front_candidates) != 1:
            return None
        front = front_candidates[0]
        order.append(front)
        remaining.remove(front)
    return order


def _compose_from_front_order(
    shape: tuple[int, int],
    layer_masks: dict[int, np.ndarray],
    front_order: list[int],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for color in reversed(front_order):
        out[layer_masks[color]] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover each symmetric color layer, reverse the occlusion order, and redraw the full picture.

    Input structure:
    - Black background with two or three colored layers drawn on the same grid.
    - Each color belongs to one full shape that is vertically mirror-symmetric about its own axis.
    - Front layers hide overlapping pixels of back layers, so some visible color masks look incomplete and asymmetric.

    Rule:
    - For each non-black color, find the vertical axis that best explains the visible mask and mirror the visible pixels to reconstruct the full symmetric layer.
    - Compare every pair of reconstructed layers on their overlap cells: whichever color is actually visible there in the input must be in front of the other layer.
    - Reverse the full front-to-back order, then transparently overlay the reconstructed layers in that reversed order.

    Key insight:
    - Missing pixels are not noise: they are occlusion evidence. Vertical symmetry restores each hidden layer, and the surviving overlap colors reveal the original depth order that must then be inverted.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    colors = [int(color) for color in np.unique(g) if int(color) != BG]
    layer_masks: dict[int, np.ndarray] = {}
    for color in colors:
        visible = g == int(color)
        axis_k, completed, _score, _tied = _best_vertical_axis(visible)
        layer_masks[int(color)] = completed

    front_order = _unique_front_order(colors, _infer_front_edges(g, layer_masks))
    assert front_order is not None, "layer order must be uniquely recoverable"
    reversed_front_order = list(reversed(front_order))
    return _compose_from_front_order(g.shape, layer_masks, reversed_front_order)


def _segment_bounds(axis_k: int, half_width: int) -> tuple[int, int]:
    if axis_k % 2 == 0:
        center_col = axis_k // 2
        return center_col - half_width, center_col + half_width
    left_center = axis_k // 2
    return left_center - half_width + 1, left_center + half_width


def _build_full_layer(
    shape: tuple[int, int],
    axis_k: int,
    top: int,
    row_widths: list[int],
) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for dr, half_width in enumerate(row_widths):
        row = top + dr
        c0, c1 = _segment_bounds(axis_k, int(half_width))
        mask[row, c0 : c1 + 1] = True
    return mask


def _recoverable_by_symmetry(full_mask: np.ndarray, visible_mask: np.ndarray, axis_k: int) -> bool:
    rows, cols = np.where(full_mask)
    checked: set[tuple[int, int, int]] = set()
    for row, col in zip(rows.tolist(), cols.tolist()):
        mirror_col = int(axis_k) - int(col)
        key = (int(row), min(int(col), mirror_col), max(int(col), mirror_col))
        if key in checked:
            continue
        checked.add(key)
        if not bool(visible_mask[int(row), int(col)]) and not bool(visible_mask[int(row), mirror_col]):
            return False
    return True


def _make_row_widths(max_half_width: int, height: int) -> list[int]:
    widths = [random.randint(1, max_half_width) for _ in range(height)]
    if max_half_width > 1 and len(set(widths)) == 1:
        idx = random.randrange(height)
        replacement = widths[idx]
        while replacement == widths[idx]:
            replacement = random.randint(1, max_half_width)
        widths[idx] = replacement
    return widths


def _choose_axes(width: int, n_layers: int, max_total_span: int | None = None) -> list[int] | None:
    center_k = width - 1
    candidates = []
    for axis_k in range(0, 2 * (width - 1) + 1):
        if 0.3 * width <= axis_k / 2.0 <= 0.7 * width:
            c0, c1 = _segment_bounds(axis_k, 2)
            if 0 <= c0 <= c1 < width:
                candidates.append(axis_k)
    if len(candidates) < n_layers:
        return None

    for _ in range(100):
        axes = sorted(random.sample(candidates, n_layers), key=lambda k: (abs(k - center_k), k))
        axes = axes[:n_layers]
        if min(abs(a - b) for a, b in combinations(axes, 2)) < 2:
            continue
        if max_total_span is not None and max(axes) - min(axes) > max_total_span:
            continue
        return list(axes)
    return None


def _safe_half_width(axis_k: int, width: int) -> int:
    if axis_k % 2 == 0:
        center_col = axis_k // 2
        return min(center_col, width - 1 - center_col)
    left_center = axis_k // 2
    return min(left_center + 1, width - 1 - left_center)


def _generate_two_layer_specs(shape: tuple[int, int], colors: list[int]) -> list[dict] | None:
    h, w = shape
    axes = _choose_axes(w, len(colors), max_total_span=8)
    if axes is None:
        return None

    band_top = random.randint(1, max(1, h - 11))
    specs = []
    for color, axis_k in zip(colors, axes):
        safe_half = _safe_half_width(axis_k, w)
        max_half_width = random.randint(2, min(5, safe_half))
        height = random.randint(5, min(9, h - 2))
        top = max(0, min(h - height, band_top + random.randint(-2, 2)))
        row_widths = _make_row_widths(max_half_width, height)
        full_mask = _build_full_layer(shape, axis_k, top, row_widths)
        specs.append(
            {
                "color": int(color),
                "axis_k": int(axis_k),
                "mask": full_mask,
            }
        )
    return specs


def _generate_three_layer_specs(shape: tuple[int, int], colors: list[int]) -> list[dict] | None:
    h, w = shape
    axes = _choose_axes(w, 3, max_total_span=10)
    if axes is None:
        return None
    axes = sorted(axes)

    layer_height = random.randint(7, 8)
    delta = layer_height - 2
    max_base_top = h - (2 * delta + layer_height)
    if max_base_top < 0:
        return None
    base_top = random.randint(0, max_base_top)
    tops = [base_top, base_top + delta, base_top + 2 * delta]

    specs = []
    for color, axis_k, top in zip(colors, axes, tops):
        safe_half = _safe_half_width(axis_k, w)
        max_half_width = random.randint(2, min(4, safe_half))
        row_widths = _make_row_widths(max_half_width, layer_height)
        full_mask = _build_full_layer(shape, axis_k, top, row_widths)
        specs.append(
            {
                "color": int(color),
                "axis_k": int(axis_k),
                "mask": full_mask,
            }
        )
    return specs


def _compose_input(shape: tuple[int, int], specs: list[dict], front_order: list[int]) -> np.ndarray:
    masks = {int(spec["color"]): spec["mask"] for spec in specs}
    return _compose_from_front_order(shape, masks, front_order)


def _validate_case(grid: np.ndarray, specs: list[dict], front_order: list[int]) -> bool:
    if np.count_nonzero(grid) == 0:
        return False

    masks = {int(spec["color"]): spec["mask"] for spec in specs}
    visible_masks = {color: grid == color for color in masks}

    if any(int(visible.sum()) < 6 for visible in visible_masks.values()):
        return False
    if not any(int(visible_masks[color].sum()) < int(masks[color].sum()) for color in masks):
        return False

    asymmetric_visible = False
    recovered_masks: dict[int, np.ndarray] = {}
    for spec in specs:
        color = int(spec["color"])
        axis_k = int(spec["axis_k"])
        full_mask = spec["mask"]
        visible = visible_masks[color]
        mirrored_visible = _mirror_mask(visible, axis_k)
        if mirrored_visible is None:
            return False
        if not np.array_equal(visible, mirrored_visible):
            asymmetric_visible = True
        if not _recoverable_by_symmetry(full_mask, visible, axis_k):
            return False
        best_axis, completed, _score, tied = _best_vertical_axis(visible)
        if tied or best_axis != axis_k:
            return False
        if not np.array_equal(completed, full_mask):
            return False
        recovered_masks[color] = completed

    if not asymmetric_visible:
        return False

    edges = _infer_front_edges(grid, recovered_masks)
    inferred_order = _unique_front_order(sorted(recovered_masks), edges)
    if inferred_order != front_order:
        return False
    if len(edges) < len(front_order) - 1:
        return False

    out = main(grid)
    expected = _compose_from_front_order(grid.shape, recovered_masks, list(reversed(front_order)))
    if not np.array_equal(out, expected):
        return False
    if np.array_equal(out, grid):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    n_layers = random.choice([2, 3])
    if n_layers == 2:
        height = random.randint(15, 24)
        width = random.randint(15, 24)
    else:
        height = random.randint(19, 24)
        width = random.randint(18, 24)
    colors = random.sample(NON_BG_COLORS, n_layers)

    if n_layers == 2:
        specs = _generate_two_layer_specs((height, width), colors)
    else:
        specs = _generate_three_layer_specs((height, width), colors)
    if specs is None:
        return None

    if n_layers == 2:
        front_order = [int(spec["color"]) for spec in random.sample(specs, len(specs))]
    else:
        front_order = [int(spec["color"]) for spec in specs]
        if random.random() < 0.5:
            front_order = list(reversed(front_order))
    grid = _compose_input((height, width), specs, front_order)
    if not _validate_case(grid, specs, front_order):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid occlusion-reversal example")
