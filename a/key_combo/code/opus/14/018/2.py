from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
SEPARATOR = 5
MARKER = 7
FOREGROUND_POOL = (1, 2, 3, 4, 6, 8, 9)
QUADRANT_SIZE_OPTIONS = (5, 6)
GENERATION_ATTEMPTS = 500
QUADRANT_ATTEMPTS = 180
LOCAL_RECT_MAX = 4


def _d4_orbit(r: int, c: int, n: int) -> set[tuple[int, int]]:
    last = n - 1
    return {
        (r, c),
        (r, last - c),
        (last - r, c),
        (last - r, last - c),
        (c, r),
        (c, last - r),
        (last - c, r),
        (last - c, last - r),
    }


def _find_cross(grid: np.ndarray) -> tuple[int, int]:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == SEPARATOR)]
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == SEPARATOR)]
    assert len(rows) == 1 and len(cols) == 1, "expected one grey separator row and column"
    return int(rows[0]), int(cols[0])


def _quadrant_index(r: int, c: int, mid: int) -> int:
    assert r != mid and c != mid, "quadrant cell cannot lie on the separator cross"
    if r < mid and c < mid:
        return 0
    if r < mid and c > mid:
        return 1
    if r > mid and c < mid:
        return 2
    return 3


def _quadrant_slices(mid: int, idx: int) -> tuple[slice, slice]:
    if idx == 0:
        return slice(0, mid), slice(0, mid)
    if idx == 1:
        return slice(0, mid), slice(mid + 1, 2 * mid + 1)
    if idx == 2:
        return slice(mid + 1, 2 * mid + 1), slice(0, mid)
    return slice(mid + 1, 2 * mid + 1), slice(mid + 1, 2 * mid + 1)


def _crop_quadrant(grid: np.ndarray, mid: int, idx: int) -> np.ndarray:
    rs, cs = _quadrant_slices(mid, idx)
    return grid[rs, cs].copy()


def _local_to_global(mid: int, idx: int, r: int, c: int) -> tuple[int, int]:
    if idx == 0:
        return int(r), int(c)
    if idx == 1:
        return int(r), int(mid + 1 + c)
    if idx == 2:
        return int(mid + 1 + r), int(c)
    return int(mid + 1 + r), int(mid + 1 + c)


def _mirror_upper_triangle(upper: np.ndarray) -> np.ndarray:
    q = int(upper.shape[0])
    out = upper.copy()
    for r in range(q):
        for c in range(r):
            out[r, c] = out[c, r]
    return out


def _build_full_grid(tl_quadrant: np.ndarray) -> np.ndarray:
    q = int(tl_quadrant.shape[0])
    side = 2 * q + 1
    mid = q
    grid = np.full((side, side), BACKGROUND, dtype=int)
    grid[mid, :] = SEPARATOR
    grid[:, mid] = SEPARATOR
    grid[:q, :q] = tl_quadrant
    grid[:q, mid + 1 :] = np.fliplr(tl_quadrant)
    grid[mid + 1 :, :q] = np.flipud(tl_quadrant)
    grid[mid + 1 :, mid + 1 :] = np.flipud(np.fliplr(tl_quadrant))
    return grid


def _d4_consistent(grid: np.ndarray) -> bool:
    n = int(grid.shape[0])
    mid = n // 2
    for r in range(n):
        for c in range(n):
            if r == mid or c == mid:
                continue
            values = {int(grid[rr, cc]) for rr, cc in _d4_orbit(r, c, n) if rr != mid and cc != mid}
            if len(values) != 1:
                return False
    return True


def _make_tl_quadrant(q: int, palette: tuple[int, ...]) -> np.ndarray | None:
    base_choices = (BACKGROUND,) + palette
    base_weights = [3] + [2] * len(palette)

    for _ in range(QUADRANT_ATTEMPTS):
        upper = np.zeros((q, q), dtype=int)
        for r in range(q):
            for c in range(r, q):
                neighbor_values: list[int] = []
                for nr, nc in ((r - 1, c), (r, c - 1), (r - 1, c - 1), (r - 1, c + 1)):
                    if nr < 0 or nc < 0 or nr >= q or nc >= q or nr > nc:
                        continue
                    if nr < r or (nr == r and nc < c):
                        neighbor_values.append(int(upper[nr, nc]))
                if neighbor_values and random.random() < 0.72:
                    upper[r, c] = random.choice(neighbor_values)
                else:
                    upper[r, c] = random.choices(base_choices, weights=base_weights, k=1)[0]

        quadrant = _mirror_upper_triangle(upper)

        for overlay_idx in range(random.randint(2, 4)):
            h = random.randint(1, q - 1)
            w = random.randint(1, q - 1)
            r0 = random.randint(0, q - h)
            c0 = random.randint(0, q - w)
            mask = np.zeros((q, q), dtype=bool)
            mask[r0 : r0 + h, c0 : c0 + w] = True
            mask |= mask.T
            if overlay_idx == 0:
                color = random.choice(palette)
            else:
                color = random.choice(base_choices)
            quadrant[mask] = color

        used_foregrounds = {int(v) for v in np.unique(quadrant) if int(v) != BACKGROUND}
        if len(used_foregrounds) < 2:
            continue
        if len(palette) == 3 and len(used_foregrounds) < 3:
            continue

        fill_ratio = float(np.count_nonzero(quadrant)) / float(quadrant.size)
        if not (0.38 <= fill_ratio <= 0.88):
            continue

        if np.array_equal(quadrant, np.fliplr(quadrant)):
            continue
        if np.array_equal(quadrant, np.flipud(quadrant)):
            continue
        if np.array_equal(quadrant, np.rot90(quadrant, 2)):
            continue

        return quadrant
    return None


def _touches_cross_edge(idx: int, r0: int, c0: int, h: int, w: int, q: int) -> bool:
    top = r0 == 0
    left = c0 == 0
    bottom = r0 + h == q
    right = c0 + w == q
    if idx == 0:
        return bottom or right
    if idx == 1:
        return bottom or left
    if idx == 2:
        return top or right
    return top or left


def _distance_to_rect(r: int, c: int, rect: tuple[int, int, int, int]) -> int:
    r0, c0, h, w = rect
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    dr = 0 if r0 <= r <= r1 else min(abs(r - r0), abs(r - r1))
    dc = 0 if c0 <= c <= c1 else min(abs(c - c0), abs(c - c1))
    return int(dr + dc)


def _rectangle_candidates(patch: np.ndarray, idx: int) -> list[tuple[int, int, int, int]]:
    q = int(patch.shape[0])
    candidates: list[tuple[int, int, int, int]] = []
    max_side = min(LOCAL_RECT_MAX, q - 1)
    for h in range(2, max_side + 1):
        for w in range(2, max_side + 1):
            for r0 in range(q - h + 1):
                for c0 in range(q - w + 1):
                    if _touches_cross_edge(idx, r0, c0, h, w, q):
                        continue
                    block = patch[r0 : r0 + h, c0 : c0 + w]
                    if np.any(block == BACKGROUND):
                        continue
                    if len({int(v) for v in np.unique(block)}) < 2:
                        continue
                    outside = patch.copy()
                    outside[r0 : r0 + h, c0 : c0 + w] = -1
                    if int(np.count_nonzero(outside == BACKGROUND)) < 2:
                        continue
                    candidates.append((r0, c0, h, w))
    random.shuffle(candidates)
    return candidates


def _pick_marker(patch: np.ndarray, rect: tuple[int, int, int, int]) -> tuple[int, int] | None:
    r0, c0, h, w = rect
    rect_colors = {int(v) for v in np.unique(patch[r0 : r0 + h, c0 : c0 + w])}
    candidates = [
        (r, c)
        for r in range(patch.shape[0])
        for c in range(patch.shape[1])
        if int(patch[r, c]) != BACKGROUND and not (r0 <= r < r0 + h and c0 <= c < c0 + w)
    ]
    if not candidates:
        return None
    far_candidates = [cell for cell in candidates if _distance_to_rect(*cell, rect) >= 2]
    color_diverse = [cell for cell in far_candidates if int(patch[cell]) not in rect_colors]
    if color_diverse:
        return random.choice(color_diverse)
    if far_candidates:
        return random.choice(far_candidates)
    color_diverse = [cell for cell in candidates if int(patch[cell]) not in rect_colors]
    if color_diverse:
        return random.choice(color_diverse)
    return random.choice(candidates)


def _all_quadrant_outputs_distinct(grid: np.ndarray) -> bool:
    mid = grid.shape[0] // 2
    outputs = []
    for idx in range(4):
        patch = _crop_quadrant(grid, mid, idx)
        outputs.append((patch.shape, patch.tobytes()))
    return len(set(outputs)) == 4


def _repairable_mask(grid: np.ndarray, mid: int, idx: int) -> np.ndarray:
    rs, cs = _quadrant_slices(mid, idx)
    patch = grid[rs, cs]
    mask = np.zeros_like(patch, dtype=bool)
    for r in range(patch.shape[0]):
        for c in range(patch.shape[1]):
            value = int(patch[r, c])
            if value not in {BACKGROUND, MARKER}:
                continue
            gr, gc = _local_to_global(mid, idx, r, c)
            orbit_colors = {
                int(grid[rr, cc])
                for rr, cc in _d4_orbit(gr, gc, grid.shape[0])
                if rr != mid and cc != mid and int(grid[rr, cc]) not in {BACKGROUND, SEPARATOR, MARKER}
            }
            if len(orbit_colors) == 1:
                mask[r, c] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the marked quadrant with D4 symmetry and crop that repaired quadrant.

    Input structure:
    - The grid is a square with an odd side length and a grey full-row/full-column separator cross through the center.
    - Away from the cross, the colored pattern is D4-symmetric across the whole grid; black cells are either true background or a missing patch.
    - Exactly one orange marker lies in one quadrant on top of a non-background pattern cell, and that same quadrant also contains a black rectangular missing region.

    Rule:
    - Find the unique grey separator row and column and use the orange marker to select the target quadrant.
    - For each black or orange cell inside that quadrant, inspect its full-grid D4 orbit, ignore separator/background/marker values, and restore the unique remaining pattern color when one exists.
    - Crop out the repaired target quadrant, excluding the grey separator cross.

    Key insight:
    - The orange pixel is both the quadrant selector and a damaged pattern cell, so the solver must select the quadrant first, then use global symmetry to repair it before cropping.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height == width and height % 2 == 1, "expected an odd square grid"

    cross_row, cross_col = _find_cross(g)
    assert cross_row == cross_col == height // 2, "separator cross must be centered"
    mid = cross_row

    marker_coords = np.argwhere(g == MARKER)
    assert marker_coords.shape == (1, 2), "expected exactly one orange marker"
    marker_r, marker_c = map(int, marker_coords[0])
    target_idx = _quadrant_index(marker_r, marker_c, mid)

    repaired = g.copy()
    rs, cs = _quadrant_slices(mid, target_idx)
    for r in range(rs.start, rs.stop):
        for c in range(cs.start, cs.stop):
            value = int(g[r, c])
            if value not in {BACKGROUND, MARKER}:
                continue
            orbit_colors = {
                int(g[rr, cc])
                for rr, cc in _d4_orbit(r, c, height)
                if rr != mid and cc != mid and int(g[rr, cc]) not in {BACKGROUND, SEPARATOR, MARKER}
            }
            if orbit_colors:
                assert len(orbit_colors) == 1, "repairable orbit must carry one visible color"
                repaired[r, c] = orbit_colors.pop()

    out = repaired[rs, cs].copy()
    assert MARKER not in set(map(int, out.flat)), "marker should be repaired away before cropping"
    return out


def _generate_attempt() -> tuple[np.ndarray, np.ndarray, int, tuple[int, int, int, int], tuple[int, int]] | None:
    q = random.choice(QUADRANT_SIZE_OPTIONS)
    palette_size = random.choices((2, 3), weights=(2, 3), k=1)[0]
    palette = tuple(random.sample(FOREGROUND_POOL, palette_size))
    tl_quadrant = _make_tl_quadrant(q, palette)
    if tl_quadrant is None:
        return None

    clean = _build_full_grid(tl_quadrant)
    if not _d4_consistent(clean):
        return None
    if not _all_quadrant_outputs_distinct(clean):
        return None

    mid = q
    valid_choices: list[tuple[int, tuple[int, int, int, int], tuple[int, int]]] = []
    for idx in range(4):
        patch = _crop_quadrant(clean, mid, idx)
        for rect in _rectangle_candidates(patch, idx):
            marker = _pick_marker(patch, rect)
            if marker is None:
                continue
            valid_choices.append((idx, rect, marker))
            break
    if not valid_choices:
        return None

    target_idx, rect, marker_local = random.choice(valid_choices)
    inp = clean.copy()

    r0, c0, h, w = rect
    for dr in range(h):
        for dc in range(w):
            gr, gc = _local_to_global(mid, target_idx, r0 + dr, c0 + dc)
            inp[gr, gc] = BACKGROUND

    marker_global = _local_to_global(mid, target_idx, marker_local[0], marker_local[1])
    if (r0 <= marker_local[0] < r0 + h) and (c0 <= marker_local[1] < c0 + w):
        return None
    inp[marker_global] = MARKER
    return inp, clean, target_idx, rect, marker_local


def _validate_instance(
    inp: np.ndarray,
    clean: np.ndarray,
    target_idx: int,
    rect: tuple[int, int, int, int],
    marker_local: tuple[int, int],
) -> bool:
    try:
        assert inp.shape == clean.shape
        assert inp.ndim == 2 and inp.shape[0] <= 30
        assert _d4_consistent(clean)
        assert _all_quadrant_outputs_distinct(clean)

        mid = clean.shape[0] // 2
        cross_row, cross_col = _find_cross(inp)
        assert cross_row == cross_col == mid

        marker_coords = np.argwhere(inp == MARKER)
        assert marker_coords.shape == (1, 2)
        assert _quadrant_index(int(marker_coords[0, 0]), int(marker_coords[0, 1]), mid) == target_idx
        assert tuple(marker_coords[0]) == _local_to_global(mid, target_idx, marker_local[0], marker_local[1])

        diff = np.argwhere(inp != clean)
        r0, c0, h, w = rect
        rect_globals = {
            _local_to_global(mid, target_idx, r0 + dr, c0 + dc)
            for dr in range(h)
            for dc in range(w)
        }
        expected_diffs = set(rect_globals)
        expected_diffs.add(_local_to_global(mid, target_idx, marker_local[0], marker_local[1]))
        assert {tuple(map(int, cell)) for cell in diff} == expected_diffs

        repairable = _repairable_mask(inp, mid, target_idx)
        local_patch = _crop_quadrant(inp, mid, target_idx)
        rect_mask = np.zeros_like(local_patch, dtype=bool)
        rect_mask[r0 : r0 + h, c0 : c0 + w] = True
        assert np.all(repairable[rect_mask]), "every removed rectangle cell must be symmetry-repairable"
        assert repairable[marker_local], "marker pixel must be symmetry-repairable"
        assert int(np.count_nonzero((local_patch == BACKGROUND) & ~rect_mask)) >= 2

        expected = _crop_quadrant(clean, mid, target_idx)
        actual = main(inp)
        assert np.array_equal(actual, expected)
        return True
    except Exception:
        return False


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        attempt = _generate_attempt()
        if attempt is None:
            continue
        inp, clean, target_idx, rect, marker_local = attempt
        if _validate_instance(inp, clean, target_idx, rect, marker_local):
            return inp
    raise RuntimeError("failed to generate a valid symmetry-repair quadrant puzzle")
