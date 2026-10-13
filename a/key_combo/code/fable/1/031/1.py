from __future__ import annotations

import random

import numpy as np


class Color:
    BLACK = 0
    YELLOW = 4
    GREY = 5


GRID_HEIGHT_RANGE = (12, 24)
GRID_WIDTH_RANGE = (12, 24)
RECT_COUNT_CHOICES = (1, 2, 3)
FIRST_RECT_H_RANGE = (3, 7)
FIRST_RECT_W_RANGE = (3, 8)
OTHER_RECT_H_RANGE = (2, 6)
OTHER_RECT_W_RANGE = (2, 7)
MIN_CENTER_CHEBYSHEV = 3
MIN_COLOR_RATIO = 0.12
MAX_OCCLUSION_RATIO = 0.28
SECOND_CENTER_RECT_PROB = 0.4
PLACEMENT_TRIES = 80
GEN_RETRY_LIMIT = 300


def _all_centers(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    rr, cc = np.indices((h, w))
    return np.stack([rr, cc], axis=-1).reshape(-1, 2)


def _chebyshev_parity_vectors(centers: np.ndarray, coords: np.ndarray) -> np.ndarray:
    return (
        np.maximum(
            np.abs(centers[:, None, 0] - coords[None, :, 0]),
            np.abs(centers[:, None, 1] - coords[None, :, 1]),
        )
        & 1
    ).astype(np.uint8)


def _pattern_from_centers(shape: tuple[int, int], a: tuple[int, int], b: tuple[int, int]) -> np.ndarray:
    h, w = shape
    rr = np.arange(h)[:, None]
    cc = np.arange(w)[None, :]
    pa = np.maximum(np.abs(rr - a[0]), np.abs(cc - a[1])) & 1
    pb = np.maximum(np.abs(rr - b[0]), np.abs(cc - b[1])) & 1
    return np.where(pa == pb, Color.YELLOW, Color.BLACK).astype(int)


def _matching_center_pairs(grid: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    visible_mask = grid != Color.GREY
    visible_coords = np.argwhere(visible_mask)
    assert visible_coords.size > 0, "expected at least one visible cell"

    visible_values = grid[visible_mask]
    valid_visible = (visible_values == Color.BLACK) | (visible_values == Color.YELLOW)
    assert np.all(valid_visible), "expected only black, yellow, and grey cells"

    centers = _all_centers(grid.shape)
    signatures = _chebyshev_parity_vectors(centers, visible_coords)
    packed_signatures = np.packbits(signatures, axis=1)
    packed_need = np.packbits((visible_values == Color.BLACK).astype(np.uint8))

    lookup: dict[bytes, list[int]] = {}
    for idx, row in enumerate(packed_signatures):
        lookup.setdefault(row.tobytes(), []).append(idx)

    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for left_idx, row in enumerate(packed_signatures):
        partner_key = np.bitwise_xor(row, packed_need).tobytes()
        for right_idx in lookup.get(partner_key, ()):
            if right_idx <= left_idx:
                continue
            pairs.append(
                (
                    tuple(map(int, centers[left_idx])),
                    tuple(map(int, centers[right_idx])),
                )
            )
    return pairs


def _completion_grids(grid: np.ndarray) -> list[np.ndarray]:
    completions: dict[bytes, np.ndarray] = {}
    for a, b in _matching_center_pairs(grid):
        full = _pattern_from_centers(grid.shape, a, b)
        completions.setdefault(full.tobytes(), full)
    return list(completions.values())


def _rect_clear(mask: np.ndarray, rect: tuple[int, int, int, int], margin: int = 1) -> bool:
    h, w = mask.shape
    r0, c0, r1, c1 = rect
    return not mask[
        max(0, r0 - margin) : min(h, r1 + margin),
        max(0, c0 - margin) : min(w, c1 + margin),
    ].any()


def _paint_rect(mask: np.ndarray, rect: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = rect
    mask[r0:r1, c0:c1] = True


def _rand_span(span_range: tuple[int, int], limit: int) -> int:
    low, high = span_range
    low = min(low, limit)
    high = min(high, limit)
    return random.randint(low, high)


def _sample_covering_rect(
    shape: tuple[int, int],
    center: tuple[int, int],
    h_range: tuple[int, int],
    w_range: tuple[int, int],
) -> tuple[int, int, int, int]:
    h, w = shape
    rect_h = _rand_span(h_range, h)
    rect_w = _rand_span(w_range, w)
    r0 = random.randint(max(0, center[0] - rect_h + 1), min(center[0], h - rect_h))
    c0 = random.randint(max(0, center[1] - rect_w + 1), min(center[1], w - rect_w))
    return r0, c0, r0 + rect_h, c0 + rect_w


def _sample_free_rect(
    shape: tuple[int, int],
    h_range: tuple[int, int],
    w_range: tuple[int, int],
) -> tuple[int, int, int, int]:
    h, w = shape
    rect_h = _rand_span(h_range, h)
    rect_w = _rand_span(w_range, w)
    r0 = random.randint(0, h - rect_h)
    c0 = random.randint(0, w - rect_w)
    return r0, c0, r0 + rect_h, c0 + rect_w


def _sample_rectangles(
    shape: tuple[int, int],
    a: tuple[int, int],
    b: tuple[int, int],
) -> list[tuple[int, int, int, int]] | None:
    mask = np.zeros(shape, dtype=bool)
    rects: list[tuple[int, int, int, int]] = []
    n_rects = random.choice(RECT_COUNT_CHOICES)

    targets: list[tuple[int, int] | None] = [random.choice([a, b])]
    if n_rects >= 2 and random.random() < SECOND_CENTER_RECT_PROB:
        first = targets[0]
        assert first is not None
        targets.append(b if first == a else a)
    while len(targets) < n_rects:
        targets.append(None)

    for idx, target in enumerate(targets):
        h_range = FIRST_RECT_H_RANGE if idx == 0 else OTHER_RECT_H_RANGE
        w_range = FIRST_RECT_W_RANGE if idx == 0 else OTHER_RECT_W_RANGE
        placed = False
        for _ in range(PLACEMENT_TRIES):
            rect = (
                _sample_covering_rect(shape, target, h_range, w_range)
                if target is not None
                else _sample_free_rect(shape, h_range, w_range)
            )
            if not _rect_clear(mask, rect):
                continue
            _paint_rect(mask, rect)
            rects.append(rect)
            placed = True
            break
        if not placed:
            return None

    return rects


def _occlusion_mask(shape: tuple[int, int], rects: list[tuple[int, int, int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for rect in rects:
        _paint_rect(mask, rect)
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore grey occluders by reconstructing the dual-center Chebyshev parity interference pattern.

    Input structure:
    - A black canvas contains yellow cells from an unseen two-center interference rule.
    - Visible cells are only black or yellow, and one or more solid grey rectangles occlude part of the pattern.
    - The two generating centers lie on grid cells but are never drawn explicitly.

    Rule:
    - Find the unique unordered pair of grid cells A and B whose visible parity-interference pattern matches every non-grey cell.
    - A cell is yellow exactly when its Chebyshev distances to A and B have the same parity; otherwise it is black.
    - Replace each grey cell by the color dictated by that rule and leave every visible cell unchanged.

    Key insight:
    - Each candidate center induces a visible Chebyshev-parity signature; the true pair is the unique pair of signatures whose equality/inequality pattern matches the observed yellow/black mask.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    completions = _completion_grids(g)
    assert completions, "no center pair matches the visible pattern"
    assert len(completions) == 1, "ambiguous dual-center completion"

    out = g.copy()
    full = completions[0]
    occluded = g == Color.GREY
    out[occluded] = full[occluded]
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    a, b = random.sample([(r, c) for r in range(h) for c in range(w)], 2)
    if max(abs(a[0] - b[0]), abs(a[1] - b[1])) < MIN_CENTER_CHEBYSHEV:
        return None

    full = _pattern_from_centers((h, w), a, b)
    min_color_count = max(8, int(full.size * MIN_COLOR_RATIO))
    yellow_count = int(np.count_nonzero(full == Color.YELLOW))
    black_count = int(np.count_nonzero(full == Color.BLACK))
    if min(yellow_count, black_count) < min_color_count:
        return None
    if len(_matching_center_pairs(full)) != 1:
        return None

    rects = _sample_rectangles((h, w), a, b)
    if rects is None:
        return None

    mask = _occlusion_mask((h, w), rects)
    if int(mask.sum()) > int(full.size * MAX_OCCLUSION_RATIO):
        return None
    if len(np.unique(full[mask])) != 2:
        return None

    grid = full.copy()
    grid[mask] = Color.GREY
    visible_values = grid[grid != Color.GREY]
    if len(np.unique(visible_values)) != 2:
        return None
    if len(_matching_center_pairs(grid)) != 1:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dual-center occlusion input")
