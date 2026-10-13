from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
ORANGE = 7
PAIR_COLORS = (1, 2, 3, 4, 5, 6, 8, 9)
MAX_GENERATION_ATTEMPTS = 400
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _neighbors4(r: int, c: int, shape: tuple[int, int]):
    h, w = shape
    for dr, dc in NEIGHBORS4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _components(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[set[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp = set()
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                comp.add((rr, cc))
                for nr, nc in _neighbors4(rr, cc, mask.shape):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mirror_coords(
    coords: set[tuple[int, int]],
    orientation: str,
    axis: int,
    shape: tuple[int, int],
) -> tuple[set[tuple[int, int]], int]:
    h, w = shape
    mirrored: set[tuple[int, int]] = set()
    invalid = 0
    for r, c in coords:
        if orientation == "vertical":
            rr, cc = r, 2 * axis - c
        else:
            rr, cc = 2 * axis - r, c
        if 0 <= rr < h and 0 <= cc < w:
            mirrored.add((rr, cc))
        else:
            invalid += 1
    return mirrored, invalid


def _axis_side_penalty(
    a: set[tuple[int, int]],
    b: set[tuple[int, int]],
    orientation: str,
    axis: int,
) -> int:
    if orientation == "vertical":
        left_right = sum(c >= axis for _, c in a) + sum(c <= axis for _, c in b)
        right_left = sum(c <= axis for _, c in a) + sum(c >= axis for _, c in b)
    else:
        left_right = sum(r >= axis for r, _ in a) + sum(r <= axis for r, _ in b)
        right_left = sum(r <= axis for r, _ in a) + sum(r >= axis for r, _ in b)
    return min(left_right, right_left)


def _axis_center_penalty(
    bbox_a: tuple[int, int, int, int],
    bbox_b: tuple[int, int, int, int],
    orientation: str,
    axis: int,
) -> int:
    r0a, c0a, r1a, c1a = bbox_a
    r0b, c0b, r1b, c1b = bbox_b
    if orientation == "vertical":
        return abs(4 * axis - (c0a + c1a + c0b + c1b))
    return abs(4 * axis - (r0a + r1a + r0b + r1b))


def _best_axis(
    a: set[tuple[int, int]],
    b: set[tuple[int, int]],
    shape: tuple[int, int],
) -> tuple[str, int]:
    h, w = shape
    bbox_a = _bbox(a)
    bbox_b = _bbox(b)
    best: tuple[tuple[int, int, int, int], str, int] | None = None

    for orientation, size in (("vertical", w), ("horizontal", h)):
        for axis in range(size):
            mirrored_a, invalid_a = _mirror_coords(a, orientation, axis, shape)
            mirrored_b, invalid_b = _mirror_coords(b, orientation, axis, shape)
            score = (
                _axis_side_penalty(a, b, orientation, axis),
                invalid_a + invalid_b,
                len(mirrored_a.symmetric_difference(b)) + len(mirrored_b.symmetric_difference(a)),
                _axis_center_penalty(bbox_a, bbox_b, orientation, axis),
            )
            candidate = (score, orientation, axis)
            if best is None or candidate < best:
                best = candidate

    assert best is not None
    return best[1], best[2]


def _paint_axis(
    out: np.ndarray,
    a: set[tuple[int, int]],
    b: set[tuple[int, int]],
    orientation: str,
    axis: int,
) -> None:
    r0a, c0a, r1a, c1a = _bbox(a)
    r0b, c0b, r1b, c1b = _bbox(b)
    if orientation == "vertical":
        start = max(r0a, r0b)
        end = min(r1a, r1b)
        for r in range(start, end + 1):
            if out[r, axis] == BACKGROUND:
                out[r, axis] = ORANGE
    else:
        start = max(c0a, c0b)
        end = min(c1a, c1b)
        for c in range(start, end + 1):
            if out[axis, c] == BACKGROUND:
                out[axis, c] = ORANGE


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find same-color near-mirror pairs, infer the symmetry axis, repair missing
    mirror pixels, erase orange decoys, and draw the true axis in orange.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    clean = g.copy()
    clean[clean == ORANGE] = BACKGROUND
    out = clean.copy()

    colors = [int(color) for color in np.unique(clean) if int(color) not in {BACKGROUND, ORANGE}]
    for color in colors:
        comps = _components(clean == color)
        assert len(comps) == 2, "each non-orange color must form exactly one object pair"
        a, b = comps
        orientation, axis = _best_axis(a, b, clean.shape)
        mirror_a, _ = _mirror_coords(a, orientation, axis, clean.shape)
        mirror_b, _ = _mirror_coords(b, orientation, axis, clean.shape)
        repaired_a = a | mirror_b
        repaired_b = b | mirror_a
        for r, c in repaired_a | repaired_b:
            out[r, c] = color
        _paint_axis(out, repaired_a, repaired_b, orientation, axis)

    return out


def _mask_connected(mask: np.ndarray) -> bool:
    return mask.any() and len(_components(mask)) == 1


def _preserves_full_bbox(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _removable_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    removable: list[tuple[int, int]] = []
    for r, c in zip(*np.nonzero(mask)):
        trial = mask.copy()
        trial[r, c] = False
        if trial.sum() >= 5 and _preserves_full_bbox(trial) and _mask_connected(trial):
            removable.append((int(r), int(c)))
    return removable


def _valid_removal_set(mask: np.ndarray, removed: set[tuple[int, int]]) -> bool:
    trial = mask.copy()
    for r, c in removed:
        trial[r, c] = False
    return trial.sum() >= 5 and _preserves_full_bbox(trial) and _mask_connected(trial)


def _generate_shape() -> np.ndarray | None:
    for _ in range(100):
        h = random.randint(3, 6)
        w = random.randint(3, 6)
        mask = np.ones((h, w), dtype=bool)

        steps = random.randint(1, max(1, (h * w) // 3))
        for _ in range(steps):
            cells = [(int(r), int(c)) for r, c in zip(*np.nonzero(mask))]
            random.shuffle(cells)
            for r, c in cells:
                trial = mask.copy()
                trial[r, c] = False
                if trial.sum() < 5:
                    continue
                if _preserves_full_bbox(trial) and _mask_connected(trial):
                    mask = trial
                    break

        if np.all(mask):
            continue
        if np.array_equal(mask, np.fliplr(mask)) or np.array_equal(mask, np.flipud(mask)) or np.array_equal(mask, np.rot90(mask, 2)):
            continue
        if len(_removable_cells(mask)) < 4:
            continue
        return mask
    return None


def _sample_defect_sets(mask: np.ndarray) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    removable = _removable_cells(mask)
    if len(removable) < 2:
        return None

    for _ in range(200):
        count_a = random.choice((1, 2))
        count_b = random.choice((1, 2))
        if count_a + count_b > len(removable):
            continue
        removed_a = set(random.sample(removable, count_a))
        remaining = [cell for cell in removable if cell not in removed_a]
        if len(remaining) < count_b:
            continue
        removed_b = set(random.sample(remaining, count_b))
        if _valid_removal_set(mask, removed_a) and _valid_removal_set(mask, removed_b):
            return removed_a, removed_b
    return None


def _valid_region_positions(reserved: np.ndarray, region_h: int, region_w: int) -> list[tuple[int, int]]:
    h, w = reserved.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(h - region_h + 1):
        for c0 in range(w - region_w + 1):
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(h, r0 + region_h + 1)
            cc1 = min(w, c0 + region_w + 1)
            if not reserved[rr0:rr1, cc0:cc1].any():
                positions.append((r0, c0))
    return positions


def _mark_reserved(reserved: np.ndarray, r0: int, c0: int, region_h: int, region_w: int) -> None:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(reserved.shape[0], r0 + region_h + 1)
    cc1 = min(reserved.shape[1], c0 + region_w + 1)
    reserved[rr0:rr1, cc0:cc1] = True


def _build_pair_coords(
    mask: np.ndarray,
    orientation: str,
    gap: int,
    top: int,
    left: int,
) -> tuple[int, set[tuple[int, int]], set[tuple[int, int]], tuple[int, int]]:
    h, w = mask.shape
    perfect_a: set[tuple[int, int]] = set()
    perfect_b: set[tuple[int, int]] = set()
    if orientation == "vertical":
        axis = left + w + gap
        for r, c in zip(*np.nonzero(mask)):
            rr = top + int(r)
            cc = left + int(c)
            perfect_a.add((rr, cc))
            perfect_b.add((rr, 2 * axis - cc))
        return axis, perfect_a, perfect_b, (h, 2 * w + 2 * gap + 1)

    axis = top + h + gap
    for r, c in zip(*np.nonzero(mask)):
        rr = top + int(r)
        cc = left + int(c)
        perfect_a.add((rr, cc))
        perfect_b.add((2 * axis - rr, cc))
    return axis, perfect_a, perfect_b, (2 * h + 2 * gap + 1, w)


def _removed_global_cells(
    removed: set[tuple[int, int]],
    orientation: str,
    gap: int,
    top: int,
    left: int,
    mask_shape: tuple[int, int],
    side: str,
) -> set[tuple[int, int]]:
    h, w = mask_shape
    removed_global: set[tuple[int, int]] = set()
    if orientation == "vertical":
        axis = left + w + gap
        for r, c in removed:
            rr = top + r
            if side == "a":
                cc = left + c
            else:
                cc = 2 * axis - (left + c)
            removed_global.add((rr, cc))
        return removed_global

    axis = top + h + gap
    for r, c in removed:
        cc = left + c
        if side == "a":
            rr = top + r
        else:
            rr = 2 * axis - (top + r)
        removed_global.add((rr, cc))
    return removed_global


def _render_expected_pair(
    out: np.ndarray,
    color: int,
    perfect_a: set[tuple[int, int]],
    perfect_b: set[tuple[int, int]],
    orientation: str,
    axis: int,
) -> None:
    for r, c in perfect_a | perfect_b:
        out[r, c] = color
    _paint_axis(out, perfect_a, perfect_b, orientation, axis)


def _place_noise(inp: np.ndarray, reserved: np.ndarray) -> bool:
    blocked = reserved.copy()
    h, w = inp.shape
    candidates = [(r, c) for r in range(h) for c in range(w) if inp[r, c] == BACKGROUND and not blocked[r, c]]
    random.shuffle(candidates)
    target = random.randint(1, min(6, max(1, len(candidates) // 12)))
    placed = 0
    for r, c in candidates:
        if blocked[r, c] or inp[r, c] != BACKGROUND:
            continue
        inp[r, c] = ORANGE
        placed += 1
        blocked[r, c] = True
        for nr, nc in _neighbors4(r, c, inp.shape):
            blocked[nr, nc] = True
        if placed == target:
            break
    return placed >= 1


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(14, 24)
    w = random.randint(14, 24)
    pair_count = random.choice((1, 2, 2, 3))
    if pair_count > len(PAIR_COLORS):
        return None

    colors = random.sample(PAIR_COLORS, pair_count)
    inp = np.full((h, w), BACKGROUND, dtype=int)
    expected = np.full_like(inp, BACKGROUND)
    reserved = np.zeros((h, w), dtype=bool)

    for color in colors:
        placed = False
        for _ in range(120):
            mask = _generate_shape()
            if mask is None:
                continue
            defects = _sample_defect_sets(mask)
            if defects is None:
                continue
            removed_a, removed_b = defects
            orientation = random.choice(("vertical", "horizontal"))
            gap = random.randint(1, 2)
            mh, mw = mask.shape
            region_h = mh if orientation == "vertical" else 2 * mh + 2 * gap + 1
            region_w = 2 * mw + 2 * gap + 1 if orientation == "vertical" else mw
            positions = _valid_region_positions(reserved, region_h, region_w)
            if not positions:
                continue
            top, left = random.choice(positions)
            axis, perfect_a, perfect_b, _ = _build_pair_coords(mask, orientation, gap, top, left)

            defect_a = perfect_a - _removed_global_cells(removed_a, orientation, gap, top, left, mask.shape, "a")
            defect_b = perfect_b - _removed_global_cells(removed_b, orientation, gap, top, left, mask.shape, "b")

            defect_mask_a = np.zeros_like(mask, dtype=bool)
            defect_mask_b = np.zeros_like(mask, dtype=bool)
            for r, c in defect_a:
                defect_mask_a[r - top, c - left] = True
            if orientation == "vertical":
                right_left = left + mw + 2 * gap + 1
                for r, c in defect_b:
                    defect_mask_b[r - top, c - right_left] = True
            else:
                lower_top = top + mh + 2 * gap + 1
                for r, c in defect_b:
                    defect_mask_b[r - lower_top, c - left] = True
            if not _mask_connected(defect_mask_a) or not _mask_connected(defect_mask_b):
                continue

            for r, c in defect_a | defect_b:
                inp[r, c] = color
            _render_expected_pair(expected, color, perfect_a, perfect_b, orientation, axis)
            _mark_reserved(reserved, top, left, region_h, region_w)
            placed = True
            break
        if not placed:
            return None

    if not _place_noise(inp, reserved):
        return None
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-repair input")
