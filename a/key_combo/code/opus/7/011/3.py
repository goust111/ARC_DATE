from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5


def _find_separator_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(cols) == 1, f"expected one grey separator column, got {cols}"
    return int(cols[0])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _normalize_shape(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    arr = np.array(cells, dtype=int)
    arr[:, 0] -= arr[:, 0].min()
    arr[:, 1] -= arr[:, 1].min()
    pts = sorted((int(r), int(c)) for r, c in arr.tolist())
    return tuple(pts)


def _shape_variants(cells: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    pts = np.array(cells, dtype=int)
    variants: set[tuple[tuple[int, int], ...]] = set()
    cur = pts.copy()
    for _ in range(4):
        for flip in (False, True):
            arr = cur.copy()
            if flip:
                arr[:, 1] *= -1
            variants.add(_normalize_shape(tuple(map(tuple, arr.tolist()))))
        cur = np.stack([cur[:, 1], -cur[:, 0]], axis=1)
    return sorted(variants)


BASE_MIXED_SHAPES = [
    ((0, 0), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0)),
]
BOUNDARY_SHAPES = [
    ((0, 0), (0, 1)),
    ((0, 0), (0, 1), (0, 2)),
]
SHARED_BASE_SHAPES = [
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
]

MIXED_VARIANTS = [v for s in BASE_MIXED_SHAPES for v in _shape_variants(s)]
MIXED_VARIANTS = [
    v
    for v in MIXED_VARIANTS
    if max(r for r, _ in v) == 1 and {r for r, _ in v} == {0, 1}
]
IRREGULAR_MIXED_VARIANTS = [v for v in MIXED_VARIANTS if len(v) == 3]
SIMPLE_MIXED_VARIANTS = [v for v in MIXED_VARIANTS if len(v) == 2]
BOUNDARY_VARIANTS = [v for s in BOUNDARY_SHAPES for v in _shape_variants(s)]
BOUNDARY_VARIANTS = [v for v in BOUNDARY_VARIANTS if max(r for r, _ in v) == 0]
SHARED_VARIANTS_BY_SIZE: dict[int, list[tuple[tuple[int, int], ...]]] = {}
for shape in SHARED_BASE_SHAPES:
    for variant in _shape_variants(shape):
        SHARED_VARIANTS_BY_SIZE.setdefault(len(variant), []).append(variant)
for size in SHARED_VARIANTS_BY_SIZE:
    uniq = []
    seen = set()
    for variant in SHARED_VARIANTS_BY_SIZE[size]:
        if variant not in seen:
            seen.add(variant)
            uniq.append(variant)
    SHARED_VARIANTS_BY_SIZE[size] = uniq


def _solve_from_panels(left_fg: np.ndarray, right_fg: np.ndarray) -> np.ndarray:
    h, w = left_fg.shape
    left_only = left_fg & ~right_fg
    right_only = right_fg & ~left_fg
    out = np.zeros((h, w), dtype=int)
    for r in range(h):
        src_cols = np.flatnonzero(left_only[r] | right_only[r]).tolist()
        for i, c in enumerate(src_cols):
            if left_only[r, c]:
                stop = src_cols[i + 1] if i + 1 < len(src_cols) else w
                if c + 1 < stop:
                    out[r, c + 1 : stop] = RED
            else:
                stop = src_cols[i - 1] if i > 0 else -1
                if stop + 1 < c:
                    out[r, stop + 1 : c] = RED
        for c in src_cols:
            out[r, c] = GREEN
    return out


def _oracle_output(left_fg: np.ndarray, right_fg: np.ndarray) -> np.ndarray:
    h, w = left_fg.shape
    left_only = left_fg & ~right_fg
    right_only = right_fg & ~left_fg
    source_mask = left_only | right_only
    out = np.zeros((h, w), dtype=int)
    for r, c in np.argwhere(left_only):
        cc = c + 1
        while cc < w and not source_mask[r, cc]:
            out[r, cc] = RED
            cc += 1
    for r, c in np.argwhere(right_only):
        cc = c - 1
        while cc >= 0 and not source_mask[r, cc]:
            out[r, cc] = RED
            cc -= 1
    out[source_mask] = GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Compare the two side panels by XOR, then stretch each difference horizontally toward the opposite side until the next difference blocks it.

    Input structure:
    - The grid has a black background with one full grey separator column in the middle.
    - A left panel and a right panel of equal size sit on the two sides of that separator.
    - Panel foreground pixels are blue and are compared at matching `(row, col)` coordinates.

    Rule:
    - Remove the separator and compare the two panels cell by cell.
    - Cells where both panels agree become black in the output.
    - A left-only foreground cell becomes a green source that stretches red to the right until the next same-row XOR source or the edge.
    - A right-only foreground cell becomes a green source that stretches red to the left until the previous same-row XOR source or the edge.

    Key insight:
    - The XOR mask is not the final image: each surviving difference also carries a direction determined by which panel owns it, and those same XOR sources are the obstacles that stop the stretches.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    sep = _find_separator_col(g)
    left = g[:, :sep]
    right = g[:, sep + 1 :]
    assert left.shape == right.shape, "panels must have equal shape"
    assert left.shape[1] > 0, "expected non-empty panels"
    assert not np.any(left == GREY) and not np.any(right == GREY), "separator color must stay in the separator column"

    left_fg = left != BLACK
    right_fg = right != BLACK
    return _solve_from_panels(left_fg, right_fg)


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return 1 + max(r for r, _ in shape), 1 + max(c for _, c in shape)


def _placeable(
    own_mask: np.ndarray,
    occupied_mask: np.ndarray,
    shape: tuple[tuple[int, int], ...],
    anchor_r: int,
    anchor_c: int,
) -> bool:
    h, w = own_mask.shape
    cells = {(anchor_r + dr, anchor_c + dc) for dr, dc in shape}
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if occupied_mask[r, c]:
            return False
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < h and 0 <= nc < w and own_mask[nr, nc] and (nr, nc) not in cells:
                    return False
    return True


def _stamp(mask: np.ndarray, shape: tuple[tuple[int, int], ...], anchor_r: int, anchor_c: int) -> None:
    for dr, dc in shape:
        mask[anchor_r + dr, anchor_c + dc] = True


def _row_source_counts(left_only: np.ndarray, right_only: np.ndarray) -> np.ndarray:
    return np.sum(left_only | right_only, axis=1)


def _collision_rows(left_only: np.ndarray, right_only: np.ndarray) -> list[int]:
    rows = []
    for r in range(left_only.shape[0]):
        left_cols = np.flatnonzero(left_only[r])
        right_cols = np.flatnonzero(right_only[r])
        if len(left_cols) and len(right_cols) and int(left_cols.min()) < int(right_cols.max()):
            rows.append(r)
    return rows


def _boundary_rows(left_only: np.ndarray, right_only: np.ndarray) -> list[int]:
    rows = []
    for r in range(left_only.shape[0]):
        has_left = bool(left_only[r].any())
        has_right = bool(right_only[r].any())
        if has_left ^ has_right:
            rows.append(r)
    return rows


def _shared_size_plan(target: int) -> list[int]:
    plans = {
        4: [[4], [2, 2]],
        5: [[2, 3]],
        6: [[3, 3], [2, 4]],
    }
    return random.choice(plans[target])


def _pick_pair_starts(height: int) -> tuple[int, int] | None:
    starts = [(a, b) for a in range(height - 1) for b in range(a + 3, height - 1)]
    return None if not starts else random.choice(starts)


def _try_place_mixed_pair(
    left_only: np.ndarray,
    right_only: np.ndarray,
    row_start: int,
    left_shape: tuple[tuple[int, int], ...],
    right_shape: tuple[tuple[int, int], ...],
) -> bool:
    h, w = left_only.shape
    lh, lw = _shape_dims(left_shape)
    rh, rw = _shape_dims(right_shape)
    assert lh == rh == 2

    for _ in range(120):
        gap = random.randint(1, max(1, w // 3))
        max_left_c = w - (lw + gap + rw)
        if max_left_c < 0:
            return False
        left_c = random.randint(0, max_left_c)
        right_c = random.randint(left_c + lw + gap, w - rw)

        if not _placeable(left_only, left_only | right_only, left_shape, row_start, left_c):
            continue
        if not _placeable(right_only, left_only | right_only, right_shape, row_start, right_c):
            continue

        trial_left = left_only.copy()
        trial_right = right_only.copy()
        _stamp(trial_left, left_shape, row_start, left_c)
        _stamp(trial_right, right_shape, row_start, right_c)
        if np.any(_row_source_counts(trial_left, trial_right) > 3):
            continue
        rows = range(row_start, row_start + 2)
        if not all(
            trial_left[r].any() and trial_right[r].any() and np.flatnonzero(trial_left[r]).min() < np.flatnonzero(trial_right[r]).max()
            for r in rows
        ):
            continue

        _stamp(left_only, left_shape, row_start, left_c)
        _stamp(right_only, right_shape, row_start, right_c)
        return True
    return False


def _try_place_boundary_component(left_only: np.ndarray, right_only: np.ndarray) -> bool:
    h, w = left_only.shape
    side_order = ["left", "right"]
    random.shuffle(side_order)
    for side in side_order:
        own = left_only if side == "left" else right_only
        other = right_only if side == "left" else left_only
        rows = list(range(h))
        random.shuffle(rows)
        shapes = BOUNDARY_VARIANTS[:]
        random.shuffle(shapes)

        for row in rows:
            if other[row].any():
                continue
            for shape in shapes:
                _, sw = _shape_dims(shape)
                if sw > w:
                    continue
                starts = list(range(w - sw + 1))
                random.shuffle(starts)
                for col in starts:
                    if not _placeable(own, left_only | right_only, shape, row, col):
                        continue
                    trial_left = left_only.copy()
                    trial_right = right_only.copy()
                    target = trial_left if side == "left" else trial_right
                    _stamp(target, shape, row, col)
                    if np.any(_row_source_counts(trial_left, trial_right) > 3):
                        continue
                    left_only[:, :] = trial_left
                    right_only[:, :] = trial_right
                    return True
    return False


def _try_place_shared_components(left_fg: np.ndarray, right_fg: np.ndarray, total_shared: int) -> bool:
    plan = _shared_size_plan(total_shared)
    for size in plan:
        variants = SHARED_VARIANTS_BY_SIZE[size][:]
        random.shuffle(variants)
        placed = False
        for shape in variants:
            sh, sw = _shape_dims(shape)
            positions = [(r, c) for r in range(left_fg.shape[0] - sh + 1) for c in range(left_fg.shape[1] - sw + 1)]
            random.shuffle(positions)
            for r, c in positions:
                if not _placeable(left_fg, left_fg, shape, r, c):
                    continue
                if not _placeable(right_fg, right_fg, shape, r, c):
                    continue
                _stamp(left_fg, shape, r, c)
                _stamp(right_fg, shape, r, c)
                placed = True
                break
            if placed:
                break
        if not placed:
            return False
    return True


def _build_input(left_fg: np.ndarray, right_fg: np.ndarray) -> np.ndarray:
    h, w = left_fg.shape
    grid = np.zeros((h, 2 * w + 1), dtype=int)
    grid[:, w] = GREY
    grid[:, :w][left_fg] = BLUE
    grid[:, w + 1 :][right_fg] = BLUE
    return grid


def _validate_instance(left_fg: np.ndarray, right_fg: np.ndarray) -> bool:
    left_only = left_fg & ~right_fg
    right_only = right_fg & ~left_fg
    shared = left_fg & right_fg

    left_count = int(left_fg.sum())
    right_count = int(right_fg.sum())
    shared_count = int(shared.sum())
    if not (8 <= left_count <= 15 and 8 <= right_count <= 15):
        return False
    if shared_count < 4:
        return False
    overlap_ratio = shared_count / min(left_count, right_count)
    if not (0.30 <= overlap_ratio <= 0.50):
        return False
    if np.any(_row_source_counts(left_only, right_only) > 3):
        return False
    if len(_collision_rows(left_only, right_only)) < 2:
        return False
    if not _boundary_rows(left_only, right_only):
        return False
    if not np.any(np.sum(left_only, axis=1) >= 2) and not np.any(np.sum(right_only, axis=1) >= 2):
        return False

    for mask in (left_fg, right_fg):
        comps = _components(mask)
        if not comps:
            return False
        if any(not (2 <= len(comp) <= 5) for comp in comps):
            return False

    expected = _oracle_output(left_fg, right_fg)
    if not np.array_equal(_solve_from_panels(left_fg, right_fg), expected):
        return False
    if not np.any(expected == RED) or not np.any(expected == GREEN):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(10, 13)
    panel_w = random.randint(10, 12)
    pair_starts = _pick_pair_starts(height)
    if pair_starts is None:
        return None

    left_only = np.zeros((height, panel_w), dtype=bool)
    right_only = np.zeros((height, panel_w), dtype=bool)

    left_shapes = [random.choice(IRREGULAR_MIXED_VARIANTS), random.choice(MIXED_VARIANTS)]
    right_shapes = [random.choice(SIMPLE_MIXED_VARIANTS), random.choice(MIXED_VARIANTS)]
    if random.random() < 0.5:
        left_shapes, right_shapes = right_shapes, left_shapes

    for row_start, lshape, rshape in zip(pair_starts, left_shapes, right_shapes):
        if not _try_place_mixed_pair(left_only, right_only, row_start, lshape, rshape):
            return None

    if not _try_place_boundary_component(left_only, right_only):
        return None

    left_fg = left_only.copy()
    right_fg = right_only.copy()
    shared_total = random.randint(4, 6)
    if not _try_place_shared_components(left_fg, right_fg, shared_total):
        return None
    if not _validate_instance(left_fg, right_fg):
        return None

    inp = _build_input(left_fg, right_fg)
    if not np.array_equal(main(inp), _oracle_output(left_fg, right_fg)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dual-panel XOR stretch input")
