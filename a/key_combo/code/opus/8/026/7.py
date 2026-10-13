from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5

_NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_NEIGHBORS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

_OUTLINE_LIBRARY: dict[int, list[tuple[np.ndarray, bool]]] | None = None


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_nonzero(patch: np.ndarray) -> np.ndarray:
    coords = np.argwhere(patch != BLACK)
    r0, c0, r1, c1 = _bbox_coords(coords)
    return patch[r0 : r1 + 1, c0 : c1 + 1].copy()


def _component_coords(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    steps = _NEIGHBORS4 if int(connectivity) == 4 else _NEIGHBORS8
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(int(r), int(c))])
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in steps:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _find_color_components(grid: np.ndarray, color: int, connectivity: int = 8) -> list[np.ndarray]:
    return _component_coords(np.asarray(grid) == int(color), connectivity=connectivity)


def _row_span_interior_cells(grid: np.ndarray, coords: np.ndarray) -> list[tuple[int, int]]:
    occupied = {(int(r), int(c)) for r, c in coords}
    rows = sorted({int(r) for r, _c in occupied})
    interior: list[tuple[int, int]] = []
    for r in rows:
        cols = sorted(c for rr, c in occupied if rr == r)
        if len(cols) < 2:
            continue
        for c in range(cols[0] + 1, cols[-1]):
            if int(grid[r, c]) == BLACK:
                interior.append((int(r), int(c)))
    return interior


def _outside_reachable_black(patch: np.ndarray) -> np.ndarray:
    h, w = patch.shape
    wall = patch == GREY
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not bool(wall[r, c]) and not bool(seen[r, c]):
                seen[r, c] = True
                q.append((int(r), int(c)))
    for c in range(w):
        for r in (0, h - 1):
            if not bool(wall[r, c]) and not bool(seen[r, c]):
                seen[r, c] = True
                q.append((int(r), int(c)))
    while q:
        r, c = q.popleft()
        for dr, dc in _NEIGHBORS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and not bool(wall[nr, nc]) and not bool(seen[nr, nc]):
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _enclosed_black_cells(patch: np.ndarray) -> list[tuple[int, int]]:
    reachable = _outside_reachable_black(patch)
    enclosed = (patch == BLACK) & ~reachable
    coords = np.argwhere(enclosed)
    return [(int(r), int(c)) for r, c in coords]


def _best_vertical_axis(grid: np.ndarray) -> tuple[int, list[int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    blue_coords = np.argwhere(g == BLUE)
    assert blue_coords.size > 0, "expected blue pixels"

    scores: list[int] = []
    for axis in range(w):
        paired = 0
        for r, c in blue_coords:
            mc = 2 * int(axis) - int(c)
            if 0 <= mc < w and int(g[int(r), int(mc)]) == BLUE:
                paired += 1
        scores.append(int(paired))
    best = max(scores)
    axes = [i for i, score in enumerate(scores) if score == best]
    return int(axes[0]), scores


def _breaker_reds(grid: np.ndarray, axis: int) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out: list[tuple[int, int]] = []
    for r, c in np.argwhere(g == RED):
        mc = 2 * int(axis) - int(c)
        if not (0 <= mc < w) or int(g[int(r), int(mc)]) != RED:
            out.append((int(r), int(c)))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase asymmetric red symmetry-breakers and refill the unique grey cavity whose row-span area matches their count.

    Input structure:
    - Black background surrounds a left-side blue motif and several right-side grey outline objects.
    - The blue motif is nearly vertically symmetric, but some red pixels are placed asymmetrically on or above it.
    - Each grey object is an 8-connected closed outline whose cavity is measured row by row between the leftmost and rightmost grey pixels.

    Rule:
    - Infer the blue motif's vertical symmetry axis as the column whose reflection pairing produces the most blue-blue matches.
    - Any red pixel whose reflected position across that axis is not also red is a symmetry-breaker; erase all such red pixels to black and count them.
    - For every grey outline, compute its row-span cavity cells: on each occupied row, count black cells strictly between the leftmost and rightmost grey cells.
    - Find the unique grey outline whose cavity size equals the erased-red count, and fill that cavity with red.

    Key insight:
    - The red pixels are identified by violating an inferred symmetry, and only their erased count determines which distant outline cavity should be filled.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    axis, scores = _best_vertical_axis(g)
    assert scores.count(max(scores)) == 1, "expected a unique best blue symmetry axis"

    breakers = _breaker_reds(g, axis)
    erased = len(breakers)
    assert erased > 0, "expected at least one red symmetry-breaker"

    grey_components = _find_color_components(g, GREY, connectivity=8)
    matches: list[list[tuple[int, int]]] = []
    for coords in grey_components:
        interior = _row_span_interior_cells(g, coords)
        if len(interior) == erased:
            matches.append(sorted(interior))

    assert len(matches) == 1, "expected exactly one grey cavity matching the erased-red count"

    out = g.copy()
    for r, c in breakers:
        out[r, c] = BLACK
    for r, c in matches[0]:
        out[r, c] = RED
    return out


def _rectangle_outline(area: int) -> list[np.ndarray]:
    patches: list[np.ndarray] = []
    for ih in range(1, area + 1):
        if area % ih != 0:
            continue
        iw = area // ih
        h = ih + 2
        w = iw + 2
        patch = np.zeros((h, w), dtype=int)
        patch[0, :] = GREY
        patch[-1, :] = GREY
        patch[:, 0] = GREY
        patch[:, -1] = GREY
        patches.append(patch)
    patches.sort(key=lambda arr: (max(arr.shape), arr.shape[0] * arr.shape[1], arr.shape))
    return patches


def _is_rectangle_outline(patch: np.ndarray) -> bool:
    h, w = patch.shape
    rect = np.zeros((h, w), dtype=int)
    rect[0, :] = GREY
    rect[-1, :] = GREY
    rect[:, 0] = GREY
    rect[:, -1] = GREY
    return np.array_equal(patch, rect)


def _outline_patch_from_rows(rows: list[tuple[int, int]]) -> np.ndarray:
    h = len(rows)
    w = max(r for _l, r in rows) + 1
    patch = np.zeros((h, w), dtype=int)
    for idx, (l, r) in enumerate(rows):
        if idx in (0, h - 1):
            patch[idx, l : r + 1] = GREY
        else:
            patch[idx, l] = GREY
            patch[idx, r] = GREY
    return _crop_nonzero(patch)


def _build_outline_library() -> dict[int, list[tuple[np.ndarray, bool]]]:
    library: dict[int, list[tuple[np.ndarray, bool]]] = defaultdict(list)
    seen: dict[int, set[tuple[tuple[int, int], bytes]]] = defaultdict(set)

    def add_patch(patch: np.ndarray) -> None:
        patch = _crop_nonzero(np.asarray(patch, dtype=int))
        if not np.any(patch == GREY):
            return
        grey_components = _find_color_components(patch, GREY, connectivity=8)
        if len(grey_components) != 1:
            return
        row_interior = _row_span_interior_cells(patch, grey_components[0])
        area = len(row_interior)
        if not (2 <= area <= 8):
            return
        enclosed = _enclosed_black_cells(patch)
        if len(enclosed) != area:
            return
        if len(_component_coords(((patch == BLACK) & ~_outside_reachable_black(patch)), connectivity=4)) != 1:
            return
        key = (patch.shape, patch.tobytes())
        if key in seen[area]:
            return
        seen[area].add(key)
        library[area].append((patch, _is_rectangle_outline(patch)))

    for area in range(2, 9):
        for patch in _rectangle_outline(area):
            add_patch(patch)

    manual_rows = [
        [(1, 2), (0, 3), (1, 2)],  # area 2
        [(1, 3), (0, 4), (1, 3)],  # area 3
        [(1, 2), (0, 3), (0, 3), (1, 2)],  # area 4
        [(1, 2), (0, 3), (1, 3), (0, 3), (1, 2)],  # area 5
        [(1, 2), (0, 3), (0, 3), (0, 3), (1, 2)],  # area 6
        [(1, 2), (0, 3), (1, 3), (0, 3), (0, 3), (1, 2)],  # area 7
        [(1, 2), (0, 3), (0, 3), (0, 3), (0, 3), (1, 2)],  # area 8
    ]
    for rows in manual_rows:
        add_patch(_outline_patch_from_rows(rows))

    for area in range(2, 9):
        library[area].sort(
            key=lambda item: (
                item[1],
                max(item[0].shape),
                item[0].shape[0] * item[0].shape[1],
                item[0].shape,
            )
        )
        assert library[area], f"missing outline patches for area {area}"
    return dict(library)


def _outline_library() -> dict[int, list[tuple[np.ndarray, bool]]]:
    global _OUTLINE_LIBRARY
    if _OUTLINE_LIBRARY is None:
        _OUTLINE_LIBRARY = _build_outline_library()
    return _OUTLINE_LIBRARY


def _choose_outline_patch(area: int) -> np.ndarray:
    variants = _outline_library()[int(area)]
    non_rect = [patch for patch, is_rect in variants if not is_rect]
    rect = [patch for patch, is_rect in variants if is_rect]
    if non_rect and random.random() < 0.65:
        return random.choice(non_rect).copy()
    if rect and random.random() < 0.5:
        return random.choice(rect).copy()
    return random.choice([patch for patch, _flag in variants]).copy()


def _make_blue_patch() -> tuple[np.ndarray, int]:
    while True:
        top_pad = random.randint(1, 2)
        core_h = random.randint(5, 8)
        half_w = random.randint(2, 4)
        axis = half_w

        widths = [random.randint(1, half_w)]
        for _ in range(core_h - 1):
            widths.append(max(1, min(half_w, widths[-1] + random.choice([-1, 0, 1]))))
        if len(set(widths)) == 1 and half_w > 1:
            idx = random.randrange(core_h)
            widths[idx] = max(1, min(half_w, widths[idx] + random.choice([-1, 1])))
        if sum(widths) < 6:
            continue

        patch = np.zeros((top_pad + core_h, 2 * half_w + 1), dtype=int)
        for i, width in enumerate(widths):
            r = top_pad + i
            patch[r, axis - width : axis + width + 1] = BLUE
        return patch, axis


def _breaker_candidates(blue_patch: np.ndarray, axis: int, side: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    overlay: list[tuple[int, int]] = []
    above: list[tuple[int, int]] = []
    h, w = blue_patch.shape
    if side < 0:
        overlay = [(int(r), int(c)) for r, c in np.argwhere(blue_patch == BLUE) if int(c) < axis]
        for r in range(h):
            blue_cols = [c for c in range(w) if int(blue_patch[r, c]) == BLUE]
            if not blue_cols:
                continue
            left = min(blue_cols)
            for rr in range(r):
                if int(blue_patch[rr, left]) == BLACK:
                    above.append((int(rr), int(left)))
    else:
        overlay = [(int(r), int(c)) for r, c in np.argwhere(blue_patch == BLUE) if int(c) > axis]
        for r in range(h):
            blue_cols = [c for c in range(w) if int(blue_patch[r, c]) == BLUE]
            if not blue_cols:
                continue
            right = max(blue_cols)
            for rr in range(r):
                if int(blue_patch[rr, right]) == BLACK:
                    above.append((int(rr), int(right)))

    overlay.sort()
    above = sorted(set(above))
    return overlay, above


def _sample_breakers(blue_patch: np.ndarray, axis: int, count: int, side: int) -> list[tuple[int, int]]:
    overlay, above = _breaker_candidates(blue_patch, axis, side)
    assert len(overlay) >= count, "blue patch must expose enough overlay breaker cells"

    chosen: list[tuple[int, int]] = []
    if above and random.random() < 0.7:
        chosen.append(random.choice(above))

    overlay_pool = [cell for cell in overlay if cell not in chosen]
    random.shuffle(overlay_pool)
    chosen.extend(overlay_pool[: count - len(chosen)])
    assert len(chosen) == count
    return chosen


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, transparent: int = BLACK) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def _expected_output(
    grid: np.ndarray,
    breakers: list[tuple[int, int]],
    matched_interior: list[tuple[int, int]],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for r, c in breakers:
        out[r, c] = BLACK
    for r, c in matched_interior:
        out[r, c] = RED
    return out


def _validate_instance(
    grid: np.ndarray,
    planted_axis: int,
    expected_breakers: list[tuple[int, int]],
    matched_interior: list[tuple[int, int]],
) -> None:
    inferred_axis, scores = _best_vertical_axis(grid)
    assert inferred_axis == int(planted_axis), (inferred_axis, planted_axis, scores)
    assert scores.count(max(scores)) == 1, scores

    actual_breakers = sorted(_breaker_reds(grid, inferred_axis))
    assert actual_breakers == sorted(expected_breakers), (actual_breakers, expected_breakers)
    assert len(actual_breakers) == int(np.count_nonzero(grid == RED))

    grey_components = _find_color_components(grid, GREY, connectivity=8)
    assert 2 <= len(grey_components) <= 3
    cavity_sizes = [len(_row_span_interior_cells(grid, coords)) for coords in grey_components]
    assert len(set(cavity_sizes)) == len(cavity_sizes), cavity_sizes
    assert cavity_sizes.count(len(expected_breakers)) == 1, cavity_sizes

    matched_sorted = sorted(matched_interior)
    found_match = False
    for coords in grey_components:
        interior = sorted(_row_span_interior_cells(grid, coords))
        if len(interior) == len(expected_breakers):
            assert interior == matched_sorted
            found_match = True
        for r, c in interior:
            assert int(grid[r, c]) == BLACK
    assert found_match

    expected = _expected_output(grid, expected_breakers, matched_interior)
    actual = main(grid)
    assert np.array_equal(actual, expected)
    assert not np.array_equal(actual, grid)


def generate_input() -> np.ndarray:
    for _attempt in range(400):
        matched_area = random.randint(3, 6)
        outline_count = random.choice([2, 3])
        other_areas = random.sample([area for area in range(2, 9) if area != matched_area], outline_count - 1)
        outline_areas = [matched_area, *other_areas]
        random.shuffle(outline_areas)
        patches = [_choose_outline_patch(area) for area in outline_areas]

        blue_patch, local_axis = _make_blue_patch()
        breaker_side = random.choice([-1, 1])
        local_breakers = _sample_breakers(blue_patch, local_axis, matched_area, breaker_side)

        mid_gap = random.randint(3, 5)
        left_margin = random.randint(1, 2)
        right_margin = random.randint(1, 2)
        right_region_width = max(patch.shape[1] for patch in patches) + random.randint(1, 3)
        width = left_margin + blue_patch.shape[1] + mid_gap + right_region_width + right_margin

        vertical_gaps = [random.randint(1, 2) for _ in range(max(0, outline_count - 1))]
        stacked_height = sum(patch.shape[0] for patch in patches) + sum(vertical_gaps)
        height = max(
            random.randint(14, 18),
            blue_patch.shape[0] + random.randint(3, 6),
            stacked_height + random.randint(2, 5),
        )
        if height > 30 or width > 30:
            continue

        grid = np.zeros((height, width), dtype=int)

        motif_r0 = random.randint(1, height - blue_patch.shape[0] - 1)
        motif_c0 = left_margin
        _paste(grid, blue_patch, motif_r0, motif_c0)
        planted_axis = motif_c0 + local_axis

        expected_breakers = [(motif_r0 + r, motif_c0 + c) for r, c in local_breakers]
        for r, c in expected_breakers:
            grid[r, c] = RED

        right_start = motif_c0 + blue_patch.shape[1] + mid_gap
        row = random.randint(1, height - stacked_height - 1)
        matched_interior: list[tuple[int, int]] | None = None
        for idx, patch in enumerate(patches):
            max_shift = right_region_width - patch.shape[1]
            c0 = right_start + (0 if max_shift == 0 else random.randint(0, max_shift))
            _paste(grid, patch, row, c0)
            coords = np.argwhere(patch == GREY) + np.array([row, c0])
            interior = [(row + r, c0 + c) for r, c in _row_span_interior_cells(patch, np.argwhere(patch == GREY))]
            if len(interior) == matched_area:
                matched_interior = interior
            row += patch.shape[0]
            if idx < len(vertical_gaps):
                row += vertical_gaps[idx]

        assert matched_interior is not None
        if np.count_nonzero(grid == GREY) == 0 or np.count_nonzero(grid == BLUE) == 0:
            continue

        try:
            _validate_instance(grid, planted_axis, expected_breakers, matched_interior)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
