from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

MARKER_COLORS = {RED, ORANGE}
OUTLINE_COLORS = [1, GREEN, PINK, TEAL, MAROON]
VARIANTS = ("top", "bottom", "left", "right")

GRID_SIDE_RANGE = (17, 23)
OBJECT_COUNT_RANGE = (2, 3)
PLACEMENT_GAP = 2
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _contiguous(values: list[int]) -> bool:
    return values == list(range(values[0], values[-1] + 1))


def _axis_info(coords: np.ndarray) -> tuple[str, int, int, int]:
    r0, c0, r1, c1 = _bbox_coords(coords)
    if r0 == r1:
        cols = sorted(int(c) for c in coords[:, 1])
        assert _contiguous(cols), "yellow horizontal axis must be contiguous"
        return "h", int(r0), int(c0), int(c1)
    if c0 == c1:
        rows = sorted(int(r) for r in coords[:, 0])
        assert _contiguous(rows), "yellow vertical axis must be contiguous"
        return "v", int(c0), int(r0), int(r1)
    raise AssertionError("yellow component must be a straight axis")


def _expand_contiguous(flags: np.ndarray, index: int) -> tuple[int, int]:
    lo = int(index)
    hi = int(index)
    while lo > 0 and bool(flags[lo - 1]):
        lo -= 1
    while hi + 1 < len(flags) and bool(flags[hi + 1]):
        hi += 1
    return lo, hi


def _object_bbox(grid: np.ndarray, orient: str, axis: int, span0: int, span1: int) -> tuple[int, int, int, int]:
    if orient == "h":
        row_has = np.any(grid[:, span0 : span1 + 1] != BLACK, axis=1)
        r0, r1 = _expand_contiguous(row_has, axis)
        return int(r0), int(span0), int(r1), int(span1)
    col_has = np.any(grid[span0 : span1 + 1, :] != BLACK, axis=0)
    c0, c1 = _expand_contiguous(col_has, axis)
    return int(span0), int(c0), int(span1), int(c1)


def _majority_marker(values: np.ndarray) -> int:
    counts = Counter(int(v) for v in values if int(v) in MARKER_COLORS)
    assert counts, "enclosed region must contain marker pixels"
    ranked = counts.most_common()
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "marker majority must be unique"
    return int(ranked[0][0])


def _enclosed_regions(crop: np.ndarray, outline_color: int) -> list[np.ndarray]:
    walls = (crop == int(outline_color)) | (crop == YELLOW)
    passable = ~walls
    h, w = crop.shape
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if passable[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if passable[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    return _components_mask(passable & ~outside)


def _mirror_crop(crop: np.ndarray, orient: str, axis: int, source_sign: int) -> np.ndarray:
    out = crop.copy()
    h, w = crop.shape
    if orient == "h":
        src_rows = range(0, axis) if source_sign < 0 else range(axis + 1, h)
        for sr in src_rows:
            dr = 2 * axis - sr
            if 0 <= dr < h:
                out[dr, :] = crop[sr, :]
        out[axis, :] = YELLOW
    else:
        src_cols = range(0, axis) if source_sign < 0 else range(axis + 1, w)
        for sc in src_cols:
            dc = 2 * axis - sc
            if 0 <= dc < w:
                out[:, dc] = crop[:, sc]
        out[:, axis] = YELLOW
    return out


def _fill_enclosed_regions(crop: np.ndarray, outline_color: int) -> np.ndarray:
    out = crop.copy()
    for coords in _enclosed_regions(out, outline_color):
        rr, cc = coords[:, 0], coords[:, 1]
        fill_color = _majority_marker(out[rr, cc])
        out[rr, cc] = fill_color
    return out


def _solve_crop(crop: np.ndarray, orient: str, axis: int, outline_color: int) -> np.ndarray:
    marker_mask = np.isin(crop, [RED, ORANGE])
    if orient == "h":
        neg_count = int(marker_mask[:axis, :].sum())
        pos_count = int(marker_mask[axis + 1 :, :].sum())
    else:
        neg_count = int(marker_mask[:, :axis].sum())
        pos_count = int(marker_mask[:, axis + 1 :].sum())
    assert neg_count != pos_count, "one side must have strictly more markers"
    source_sign = -1 if neg_count > pos_count else 1

    mirrored = _mirror_crop(crop, orient, axis, source_sign)
    return _fill_enclosed_regions(mirrored, outline_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Vote which side of each yellow-split outline is trustworthy, mirror that side, then fill the enclosed regions by marker majority.

    Input structure:
    - The background is black.
    - Each object contains one monochrome outline color, a contiguous yellow horizontal or vertical axis, and red/orange single-pixel markers on both sides of that axis.
    - The yellow axis spans the full width or height of the object's tight bounding box, and different objects do not overlap.

    Rule:
    - For each yellow axis object, recover its bounding box and count how many red/orange markers lie on each side of the axis.
    - The side with the strictly larger marker count is the source side; reflect that side's entire contents across the axis, copying outline pixels, markers, and background onto the opposite side while keeping the axis yellow.
    - After mirroring, within that bounding box find every 4-connected passable region that cannot reach the box boundary without crossing the outline color or the yellow axis, and fill each such enclosed region with the unique majority marker color already inside it.

    Key insight:
    - The same markers serve two roles: first they vote for the trusted half to mirror, then the mirrored marker distribution inside each newly closed chamber determines the chamber's fill color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    axis_components = _components_mask(g == YELLOW)
    assert axis_components, "expected at least one yellow axis"

    ordered_axes = sorted(axis_components, key=lambda coords: tuple(map(int, coords.min(axis=0))))
    for axis_coords in ordered_axes:
        orient, axis_value, span0, span1 = _axis_info(axis_coords)
        r0, c0, r1, c1 = _object_bbox(g, orient, axis_value, span0, span1)
        crop = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        local_axis = int(axis_value - r0) if orient == "h" else int(axis_value - c0)
        colors = set(int(v) for v in np.unique(crop)) - {BLACK, RED, YELLOW, ORANGE}
        assert len(colors) == 1, "each object must have exactly one outline color"
        solved = _solve_crop(crop, orient, local_axis, next(iter(colors)))
        out[r0 : r1 + 1, c0 : c1 + 1] = solved

    return out


def _transform_patch(patch: np.ndarray, variant: str) -> np.ndarray:
    if variant == "top":
        return patch.copy()
    if variant == "bottom":
        return np.flipud(patch).copy()
    if variant == "left":
        return patch.T.copy()
    if variant == "right":
        return np.fliplr(patch.T).copy()
    raise ValueError(f"unknown variant {variant}")


def _build_expected_from_source(source_patch: np.ndarray) -> np.ndarray:
    expected = np.zeros_like(source_patch)
    expected[:2, :] = source_patch[:2, :]
    expected[2, :] = YELLOW
    expected[3, :] = source_patch[1, :]
    expected[4, :] = source_patch[0, :]
    outline_color = int(next(iter(set(np.unique(source_patch)) - {BLACK, RED, YELLOW, ORANGE})))
    return _fill_enclosed_regions(expected, outline_color)


def _canonical_object(outline_color: int, majority_color: int) -> tuple[np.ndarray, np.ndarray]:
    patch = np.zeros((5, 7), dtype=int)
    source_only = np.zeros_like(patch)

    margin = random.randint(0, 2)
    wall_cols = list(range(margin, margin + 5))
    chamber = [(1, margin + 1), (1, margin + 2), (1, margin + 3)]
    outline = {(0, c) for c in wall_cols}
    outline |= {(1, margin), (1, margin + 4)}

    for r, c in outline:
        patch[r, c] = outline_color
        source_only[r, c] = outline_color
    patch[2, :] = YELLOW
    source_only[2, :] = YELLOW

    minority_color = ORANGE if majority_color == RED else RED
    chamber_order = chamber[:]
    random.shuffle(chamber_order)
    patch[chamber_order[0]] = majority_color
    patch[chamber_order[1]] = majority_color
    patch[chamber_order[2]] = minority_color
    source_only[chamber_order[0]] = majority_color
    source_only[chamber_order[1]] = majority_color
    source_only[chamber_order[2]] = minority_color

    source_free = [
        (r, c)
        for r in range(2)
        for c in range(7)
        if patch[r, c] == BLACK
    ]
    random.shuffle(source_free)
    source_extra = random.randint(1, 2)
    for idx in range(source_extra):
        color = random.choice([majority_color, minority_color])
        r, c = source_free[idx]
        patch[r, c] = color
        source_only[r, c] = color

    target_margin = random.randint(0, 2)
    damaged_outline = {
        (4, target_margin),
        (4, target_margin + 1),
        (4, target_margin + 2),
        (3, target_margin + 2),
    }
    if random.random() < 0.5:
        damaged_outline.add((3, target_margin))
    for r, c in damaged_outline:
        patch[r, c] = outline_color

    target_free = [
        (r, c)
        for r in range(3, 5)
        for c in range(7)
        if patch[r, c] == BLACK
    ]
    random.shuffle(target_free)
    target_total = random.randint(2, min(3, len(target_free)))
    for idx in range(target_total):
        if idx == 0:
            color = majority_color
        elif idx == 1:
            color = minority_color
        else:
            color = random.choice([majority_color, minority_color])
        r, c = target_free[idx]
        patch[r, c] = color

    expected = _build_expected_from_source(source_only)
    return patch, expected


def _bbox_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _place_objects(
    grid_shape: tuple[int, int],
    patches: list[np.ndarray],
    expected_patches: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = grid_shape
    placed_boxes: list[tuple[int, int, int, int]] = []
    placements: list[tuple[int, int]] = []

    for patch in patches:
        ph, pw = patch.shape
        candidates = [
            (r0, c0)
            for r0 in range(h - ph + 1)
            for c0 in range(w - pw + 1)
        ]
        random.shuffle(candidates)

        placed = False
        for r0, c0 in candidates:
            box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            if all(_bbox_separated(box, old, PLACEMENT_GAP) for old in placed_boxes):
                placed_boxes.append(box)
                placements.append((r0, c0))
                placed = True
                break
        if not placed:
            return None

    inp = np.zeros(grid_shape, dtype=int)
    out = np.zeros(grid_shape, dtype=int)
    for patch, exp_patch, (r0, c0) in zip(patches, expected_patches, placements):
        ph, pw = patch.shape
        mask = patch != BLACK
        inp[r0 : r0 + ph, c0 : c0 + pw][mask] = patch[mask]
        mask = exp_patch != BLACK
        out[r0 : r0 + ph, c0 : c0 + pw][mask] = exp_patch[mask]
    return inp, out


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    n_obj = random.randint(*OBJECT_COUNT_RANGE)
    variants = random.sample(VARIANTS, n_obj)
    colors = random.sample(OUTLINE_COLORS, n_obj)
    majorities = [random.choice([RED, ORANGE]) for _ in range(n_obj)]
    if n_obj > 1 and len(set(majorities)) == 1:
        majorities[-1] = ORANGE if majorities[-1] == RED else RED

    base_patches = []
    base_expected = []
    for outline_color, majority in zip(colors, majorities):
        patch, expected = _canonical_object(outline_color, majority)
        base_patches.append(patch)
        base_expected.append(expected)

    patches = [_transform_patch(patch, variant) for patch, variant in zip(base_patches, variants)]
    expected_patches = [
        _transform_patch(patch, variant) for patch, variant in zip(base_expected, variants)
    ]

    grid_h = random.randint(*GRID_SIDE_RANGE)
    grid_w = random.randint(*GRID_SIDE_RANGE)
    placed = _place_objects((grid_h, grid_w), patches, expected_patches)
    if placed is None:
        return None
    return placed


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_case()
        if built is None:
            continue
        inp, expected = built
        actual = main(inp)
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid input")
