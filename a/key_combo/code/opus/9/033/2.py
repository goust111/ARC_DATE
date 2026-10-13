from __future__ import annotations

import hashlib
import random

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
BLACK = 0
MAROON = 9

MAX_GENERATION_ATTEMPTS = 600
_SEEN_INPUT_SIGNATURES: set[str] = set()
_SEEN_OUTPUT_SIGNATURES: set[str] = set()


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _densify_sparse_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    box = _bbox_from_mask(mask)
    if box is None:
        return out

    r0, c0, r1, c1 = box
    coords = np.argwhere(mask)
    rel = coords - np.array([r0, c0])
    if np.any(rel % 2 != 0):
        raise ValueError("sparse sprite must live on a two-step lattice")

    logical_h = (r1 - r0) // 2 + 1
    logical_w = (c1 - c0) // 2 + 1
    logical = np.zeros((logical_h, logical_w), dtype=bool)
    logical[rel[:, 0] // 2, rel[:, 1] // 2] = True

    patch = np.kron(logical, np.ones((2, 2), dtype=bool))[: 2 * logical_h - 1, : 2 * logical_w - 1]
    out[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = patch
    return out


def _classify_dense_masks(left_dense: np.ndarray, right_dense: np.ndarray) -> np.ndarray:
    out = np.zeros(left_dense.shape, dtype=int)
    both = left_dense & right_dense
    left_only = left_dense & ~right_dense
    right_only = right_dense & ~left_dense
    out[both] = RED
    out[left_only] = GREEN
    out[right_only] = YELLOW
    return out


def _align_bbox_to_origin(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    box = _bbox_from_mask(mask)
    if box is None:
        return out
    r0, c0, r1, c1 = box
    crop = mask[r0 : r1 + 1, c0 : c1 + 1]
    out[: crop.shape[0], : crop.shape[1]] = crop
    return out


def _trimmed(mask: np.ndarray) -> bool:
    return (
        bool(mask[0].any())
        and bool(mask[-1].any())
        and bool(mask[:, 0].any())
        and bool(mask[:, -1].any())
    )


def _has_adjacent_pair(mask: np.ndarray) -> bool:
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if rr + 1 < h and mask[rr + 1, cc]:
            return True
        if cc + 1 < w and mask[rr, cc + 1]:
            return True
    return False


def _monotone_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = a
    tr, tc = b
    path = [(int(r), int(c))]
    while (r, c) != (tr, tc):
        options = []
        if r < tr:
            options.append((1, 0))
        elif r > tr:
            options.append((-1, 0))
        if c < tc:
            options.append((0, 1))
        elif c > tc:
            options.append((0, -1))
        dr, dc = random.choice(options)
        r += dr
        c += dc
        path.append((int(r), int(c)))
    return path


def _random_connected_trimmed_mask(logical_h: int, logical_w: int) -> np.ndarray | None:
    total = logical_h * logical_w
    for _ in range(200):
        mask = np.zeros((logical_h, logical_w), dtype=bool)
        border_points = [
            (0, random.randrange(logical_w)),
            (logical_h - 1, random.randrange(logical_w)),
            (random.randrange(logical_h), 0),
            (random.randrange(logical_h), logical_w - 1),
        ]
        random.shuffle(border_points)

        current = border_points[0]
        mask[current] = True
        for target in border_points[1:]:
            for cell in _monotone_path(current, target):
                mask[cell] = True
            current = target

        base_cells = int(mask.sum())
        if base_cells < total - 1:
            target_cells = random.randint(max(base_cells + 1, total // 3), total - 1)
            while int(mask.sum()) < target_cells:
                frontier = []
                for r, c in np.argwhere(mask):
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr = int(r + dr)
                        nc = int(c + dc)
                        if 0 <= nr < logical_h and 0 <= nc < logical_w and not mask[nr, nc]:
                            frontier.append((nr, nc))
                if not frontier:
                    break
                nr, nc = random.choice(frontier)
                mask[nr, nc] = True

        if mask.all():
            continue
        if not _trimmed(mask):
            continue
        if not _has_adjacent_pair(mask):
            continue
        return mask
    return None


def _stamp_sparse_logical(zone: np.ndarray, logical: np.ndarray, anchor: tuple[int, int]) -> None:
    ar, ac = anchor
    for r, c in np.argwhere(logical):
        zone[int(ar + 2 * r), int(ac + 2 * c)] = MAROON


def _choose_common_delta(max_dr: int, max_dc: int) -> tuple[int, int] | None:
    deltas = []
    for dr in range(-min(2, max_dr), min(2, max_dr) + 1):
        for dc in range(-min(2, max_dc), min(2, max_dc) + 1):
            if dr == 0 and dc == 0:
                continue
            if abs(dr) >= max_dr + 1 or abs(dc) >= max_dc + 1:
                continue
            if dr == 0 or dc == 0 or abs(dr) == abs(dc):
                deltas.append((dr, dc))
    if not deltas:
        return None
    return random.choice(deltas)


def _pick_anchors(
    zone_h: int, zone_w: int, dense_h: int, dense_w: int
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    max_r = zone_h - dense_h
    max_c = zone_w - dense_w
    if max_r < 1 and max_c < 1:
        return None

    delta = _choose_common_delta(max_r, max_c)
    if delta is None:
        return None
    dr, dc = delta

    if dr >= 0:
        left_r = random.randint(0, max_r - dr)
        right_r = left_r + dr
    else:
        right_r = random.randint(0, max_r + dr)
        left_r = right_r - dr

    if dc >= 0:
        left_c = random.randint(0, max_c - dc)
        right_c = left_c + dc
    else:
        right_c = random.randint(0, max_c + dc)
        left_c = right_c - dc

    return (left_r, left_c), (right_r, right_c)


def _dense_from_logical(
    logical: np.ndarray, zone_h: int, zone_w: int, anchor: tuple[int, int]
) -> np.ndarray:
    zone = np.zeros((zone_h, zone_w), dtype=bool)
    sparse = np.zeros((zone_h, zone_w), dtype=bool)
    ar, ac = anchor
    for r, c in np.argwhere(logical):
        sparse[int(ar + 2 * r), int(ac + 2 * c)] = True
    zone[:] = _densify_sparse_mask(sparse)
    return zone


def main(grid: np.ndarray) -> np.ndarray:
    """
    Densify the sparse maroon sprites on both sides of the blue separator and merge them with four-way pixelwise logic.

    Input structure:
    - The grid has black background and exactly one full-height blue separator column.
    - The separator splits the canvas into equal-width left and right regions.
    - Each side contains one maroon sparse sprite whose occupied cells lie on a two-step lattice.

    Rule:
    - Find the blue separator and discard it from further processing.
    - Densify the maroon sprite in each side independently by filling the in-between pixels implied by its two-step lattice occupancy.
    - Compare the two densified side-regions at the same region coordinates.
    - Output a single-region grid: red where both sides are maroon, green where only the left side is maroon, yellow where only the right side is maroon, and black elsewhere.

    Key insight:
    - The separator defines a shared region coordinate system, and the sparse sprites must be densified before comparison because overlap is determined by the filled-in shapes, not by the original lattice points alone.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    h, w = g.shape
    separator_cols = [c for c in range(w) if np.all(g[:, c] == BLUE)]
    if len(separator_cols) != 1:
        raise ValueError("expected exactly one full-height blue separator")

    c_bar = separator_cols[0]
    left = g[:, :c_bar]
    right = g[:, c_bar + 1 :]
    if left.shape != right.shape:
        raise ValueError("left and right regions must have equal shape")

    left_dense = _densify_sparse_mask(left == MAROON)
    right_dense = _densify_sparse_mask(right == MAROON)
    return _classify_dense_masks(left_dense, right_dense)


def _semantic_checks(
    grid: np.ndarray,
    left_logical: np.ndarray,
    right_logical: np.ndarray,
    left_anchor: tuple[int, int],
    right_anchor: tuple[int, int],
) -> bool:
    h, w = grid.shape
    sep = w // 2
    zone_h, zone_w = h, sep
    left_sparse = grid[:, :sep] == MAROON
    right_sparse = grid[:, sep + 1 :] == MAROON
    left_dense = _densify_sparse_mask(left_sparse)
    right_dense = _densify_sparse_mask(right_sparse)
    output = _classify_dense_masks(left_dense, right_dense)

    colors, counts = np.unique(output, return_counts=True)
    count_map = {int(color): int(count) for color, count in zip(colors, counts)}
    for color in (BLACK, RED, GREEN, YELLOW):
        if count_map.get(color, 0) < 2:
            return False

    sparse_direct = _classify_dense_masks(left_sparse, right_sparse)
    if np.array_equal(output, sparse_direct):
        return False
    if int(np.count_nonzero(output)) <= int(np.count_nonzero(sparse_direct)):
        return False

    bbox_aligned = _classify_dense_masks(
        _align_bbox_to_origin(left_dense),
        _align_bbox_to_origin(right_dense),
    )
    if np.array_equal(output, bbox_aligned):
        return False

    left_box = _bbox_from_mask(left_dense)
    right_box = _bbox_from_mask(right_dense)
    if left_box is None or right_box is None:
        return False
    if (left_box[2] - left_box[0], left_box[3] - left_box[1]) != (
        right_box[2] - right_box[0],
        right_box[3] - right_box[1],
    ):
        return False

    expected_left = _dense_from_logical(left_logical, zone_h, zone_w, left_anchor)
    expected_right = _dense_from_logical(right_logical, zone_h, zone_w, right_anchor)
    if not np.array_equal(left_dense, expected_left):
        return False
    if not np.array_equal(right_dense, expected_right):
        return False

    if not np.array_equal(main(grid), output):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    logical_h = random.randint(3, 5)
    logical_w = random.randint(3, 5)
    dense_h = 2 * logical_h - 1
    dense_w = 2 * logical_w - 1

    zone_h_min = max(9, dense_h + 2)
    zone_h_max = min(15, dense_h + 6)
    zone_w_min = max(9, dense_w + 2)
    zone_w_max = min(14, dense_w + 6)
    if zone_h_min > zone_h_max or zone_w_min > zone_w_max:
        return None

    zone_h = random.randint(zone_h_min, zone_h_max)
    zone_w = random.randint(zone_w_min, zone_w_max)
    if 2 * zone_w + 1 > 30 or 2 * zone_w + 1 <= zone_h:
        return None

    left_logical = _random_connected_trimmed_mask(logical_h, logical_w)
    right_logical = _random_connected_trimmed_mask(logical_h, logical_w)
    if left_logical is None or right_logical is None:
        return None

    overlap = left_logical & right_logical
    left_only = left_logical & ~right_logical
    right_only = right_logical & ~left_logical
    if not (overlap.any() and left_only.any() and right_only.any() and (~(left_logical | right_logical)).any()):
        return None

    anchors = _pick_anchors(zone_h, zone_w, dense_h, dense_w)
    if anchors is None:
        return None
    left_anchor, right_anchor = anchors

    grid = np.zeros((zone_h, 2 * zone_w + 1), dtype=int)
    grid[:, zone_w] = BLUE
    _stamp_sparse_logical(grid[:, :zone_w], left_logical, left_anchor)
    _stamp_sparse_logical(grid[:, zone_w + 1 :], right_logical, right_anchor)

    if not _semantic_checks(grid, left_logical, right_logical, left_anchor, right_anchor):
        return None

    input_sig = hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()
    output = main(grid)
    output_sig = hashlib.md5(output.tobytes() + str(output.shape).encode()).hexdigest()
    if input_sig in _SEEN_INPUT_SIGNATURES or output_sig in _SEEN_OUTPUT_SIGNATURES:
        return None

    _SEEN_INPUT_SIGNATURES.add(input_sig)
    _SEEN_OUTPUT_SIGNATURES.add(output_sig)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sparse-densify logical-merge input")
