from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
GREY = 5
SEPARATOR = 9
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

LEGEND_COUNT_CHOICES = (3, 4)
TARGET_EXTRA_RANGE = (1, 4)
SHAPE_HEIGHT_RANGE = (2, 5)
SHAPE_WIDTH_RANGE = (2, 5)
SHAPE_AREA_RANGE = (4, 7)
UPPER_HEIGHT_RANGE = (7, 12)
LOWER_HEIGHT_RANGE = (8, 18)
WIDTH_RANGE = (18, 30)
LEGEND_SIDES = ("right", "left", "down", "up")
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 1
GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 250


def _normalize_coords(
    coords: tuple[tuple[int, int], ...] | list[tuple[int, int]]
) -> tuple[tuple[tuple[int, int], ...], int, int]:
    pts = [(int(r), int(c)) for r, c in coords]
    assert pts, "shape must be nonempty"
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    norm = tuple(sorted((r - r0, c - c0) for r, c in pts))
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    return norm, int(h), int(w)


def _rot90(
    coords: tuple[tuple[int, int], ...] | list[tuple[int, int]]
) -> tuple[tuple[tuple[int, int], ...], int, int]:
    norm, h, _w = _normalize_coords(coords)
    rot = tuple((c, h - 1 - r) for r, c in norm)
    return _normalize_coords(rot)


def _flip_h(
    coords: tuple[tuple[int, int], ...] | list[tuple[int, int]]
) -> tuple[tuple[tuple[int, int], ...], int, int]:
    norm, _h, w = _normalize_coords(coords)
    flip = tuple((r, w - 1 - c) for r, c in norm)
    return _normalize_coords(flip)


def _dihedral_variants(
    coords: tuple[tuple[int, int], ...] | list[tuple[int, int]]
) -> list[tuple[tuple[tuple[int, int], ...], int, int, int, bool]]:
    base = _normalize_coords(coords)[0]
    seen: set[tuple[tuple[int, int], ...]] = set()
    out: list[tuple[tuple[tuple[int, int], ...], int, int, int, bool]] = []

    cur = base
    for rot_k in range(4):
        norm, h, w = _normalize_coords(cur)
        if norm not in seen:
            seen.add(norm)
            out.append((norm, h, w, int(rot_k), False))
        cur = _rot90(cur)[0]

    cur = _flip_h(base)[0]
    for rot_k in range(4):
        norm, h, w = _normalize_coords(cur)
        if norm not in seen:
            seen.add(norm)
            out.append((norm, h, w, int(rot_k), True))
        cur = _rot90(cur)[0]

    return out


def _shape_signature(
    coords: tuple[tuple[int, int], ...] | list[tuple[int, int]]
) -> tuple[tuple[int, int], ...]:
    return min(norm for norm, _h, _w, _rot, _flip in _dihedral_variants(coords))


def _render_shape(
    coords: tuple[tuple[int, int], ...] | list[tuple[int, int]], color: int
) -> np.ndarray:
    norm, h, w = _normalize_coords(coords)
    out = np.full((h, w), BACKGROUND, dtype=int)
    for r, c in norm:
        out[r, c] = int(color)
    return out


def _components_of_color(grid: np.ndarray, color: int) -> list[tuple[tuple[int, int], ...]]:
    mask = np.asarray(grid == int(color), dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[tuple[tuple[int, int], ...]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(tuple(sorted(cells)))

    return comps


def _bbox(coords: tuple[tuple[int, int], ...] | list[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = [(int(r), int(c)) for r, c in coords]
    return (
        min(r for r, _ in pts),
        min(c for _, c in pts),
        max(r for r, _ in pts),
        max(c for _, c in pts),
    )


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    row_gap = max(0, b[0] - a[2] - 1, a[0] - b[2] - 1)
    col_gap = max(0, b[1] - a[3] - 1, a[1] - b[3] - 1)
    return int(row_gap + col_gap)


def _separator_bounds(grid: np.ndarray) -> tuple[int, int]:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == SEPARATOR)]
    assert rows, "expected a full maroon separator row"
    start, end = rows[0], rows[-1]
    assert rows == list(range(start, end + 1)), "separator rows must form one contiguous band"
    assert 0 < start and end < grid.shape[0] - 1, "separator must split legend and target areas"
    return int(start), int(end)


def _extract_legend_mapping(upper: np.ndarray) -> dict[tuple[tuple[int, int], ...], int]:
    grey_comps = [comp for comp in _components_of_color(upper, GREY) if len(comp) >= 2]
    color_comps: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    for color in sorted(set(map(int, np.unique(upper))) - {BACKGROUND, GREY, SEPARATOR}):
        for comp in _components_of_color(upper, color):
            color_comps.append((int(color), comp))

    assert 3 <= len(grey_comps) <= 4, "legend must contain 3-4 grey sprites"
    assert len(color_comps) == len(grey_comps), "legend shapes and color blocks must pair one-to-one"

    grey_boxes = [_bbox(comp) for comp in grey_comps]
    color_boxes = [_bbox(comp) for _color, comp in color_comps]

    grey_to_color: list[int] = []
    for box in grey_boxes:
        dists = sorted((_bbox_gap(box, other), idx) for idx, other in enumerate(color_boxes))
        if len(dists) > 1:
            assert dists[0][0] < dists[1][0], "legend pairing must be unambiguous"
        assert dists[0][0] <= 2, "legend color block must be adjacent to its grey shape"
        grey_to_color.append(int(dists[0][1]))

    color_to_grey: list[int] = []
    for box in color_boxes:
        dists = sorted((_bbox_gap(box, other), idx) for idx, other in enumerate(grey_boxes))
        if len(dists) > 1:
            assert dists[0][0] < dists[1][0], "legend pairing must be unambiguous"
        color_to_grey.append(int(dists[0][1]))

    mapping: dict[tuple[tuple[int, int], ...], int] = {}
    used_colors: set[int] = set()
    for grey_idx, color_idx in enumerate(grey_to_color):
        assert color_to_grey[color_idx] == grey_idx, "legend pairing must be mutual nearest"
        assert color_idx not in used_colors, "each color block must pair once"
        used_colors.add(color_idx)
        signature = _shape_signature(grey_comps[grey_idx])
        color = int(color_comps[color_idx][0])
        assert signature not in mapping, "legend shapes must have unique normalized signatures"
        mapping[signature] = color

    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor grey target sprites using a legend of shape-to-color examples above a maroon divider.

    Input structure:
    - The background is black.
    - One full maroon row separates the grid into an upper legend area and a lower target area.
    - In the legend area, each entry contains one grey sprite and one adjacent colored block that indicates the sprite's color label.
    - In the target area, several grey sprites appear in arbitrary rotations or mirrorings of the legend shapes.

    Rule:
    - Read each legend entry as a mapping from the grey sprite's normalized shape to the nearby colored block's color.
    - Normalize shapes up to rotation and horizontal reflection, so all dihedral variants share one signature.
    - For every grey sprite below the separator, find the matching legend signature and repaint that whole sprite with the mapped color.
    - Leave the legend area, the maroon separator, and the black background unchanged.

    Key insight:
    - The visible color cue is attached only to the legend examples; solving requires abstracting each grey object to a rotation/flip-invariant shape identity before recoloring the targets.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep_top, sep_bottom = _separator_bounds(g)
    upper = g[:sep_top, :]
    lower = g[sep_bottom + 1 :, :]
    mapping = _extract_legend_mapping(upper)

    out = g.copy()
    target_comps = [comp for comp in _components_of_color(lower, GREY) if len(comp) >= 2]
    assert target_comps, "expected at least one grey target sprite"
    for comp in target_comps:
        signature = _shape_signature(comp)
        assert signature in mapping, "every target shape must appear in the legend"
        color = int(mapping[signature])
        for r, c in comp:
            out[sep_bottom + 1 + r, c] = color

    return out


def _random_shape() -> tuple[tuple[int, int], ...] | None:
    for _ in range(80):
        h = random.randint(*SHAPE_HEIGHT_RANGE)
        w = random.randint(*SHAPE_WIDTH_RANGE)
        max_area = min(SHAPE_AREA_RANGE[1], h * w - 1)
        min_area = SHAPE_AREA_RANGE[0]
        if max_area < min_area:
            continue
        target_area = random.randint(min_area, max_area)
        seed = (random.randrange(h), random.randrange(w))
        cells = {seed}

        while len(cells) < target_area:
            frontier = []
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        norm, hh, ww = _normalize_coords(tuple(cells))
        if len(norm) != target_area:
            continue
        if hh < 2 or ww < 2:
            continue
        if len(norm) == hh * ww:
            continue
        if len(_dihedral_variants(norm)) < 2:
            continue
        return norm
    return None


def _sample_unique_shapes(count: int) -> list[tuple[tuple[int, int], ...]] | None:
    shapes: list[tuple[tuple[int, int], ...]] = []
    signatures: set[tuple[tuple[int, int], ...]] = set()
    for _ in range(200):
        shape = _random_shape()
        if shape is None:
            continue
        signature = _shape_signature(shape)
        if signature in signatures:
            continue
        signatures.add(signature)
        shapes.append(shape)
        if len(shapes) == count:
            return shapes
    return None


def _contact_offsets(coords: tuple[tuple[int, int], ...], side: str) -> list[int]:
    norm, h, w = _normalize_coords(coords)
    if side == "right":
        boundary = {r for r, c in norm if c == w - 1}
        limit = h - 2
    elif side == "left":
        boundary = {r for r, c in norm if c == 0}
        limit = h - 2
    elif side == "down":
        boundary = {c for r, c in norm if r == h - 1}
        limit = w - 2
    else:
        boundary = {c for r, c in norm if r == 0}
        limit = w - 2

    starts = [start for start in range(limit + 1) if any(start <= x < start + 2 for x in boundary)]
    return starts or [0]


def _make_legend_patch(
    shape: tuple[tuple[int, int], ...], color: int
) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    norm, h, w, _rot, _flip = random.choice(_dihedral_variants(shape))
    side = random.choice(LEGEND_SIDES)
    shape_patch = _render_shape(norm, GREY)
    start = random.choice(_contact_offsets(norm, side))

    if side == "right":
        patch = np.full((h, w + 2), BACKGROUND, dtype=int)
        patch[:h, :w] = shape_patch
        patch[start : start + 2, w : w + 2] = int(color)
    elif side == "left":
        patch = np.full((h, w + 2), BACKGROUND, dtype=int)
        patch[:h, 2 : 2 + w] = shape_patch
        patch[start : start + 2, 0:2] = int(color)
    elif side == "down":
        patch = np.full((h + 2, w), BACKGROUND, dtype=int)
        patch[:h, :w] = shape_patch
        patch[h : h + 2, start : start + 2] = int(color)
    else:
        patch = np.full((h + 2, w), BACKGROUND, dtype=int)
        patch[2 : 2 + h, :w] = shape_patch
        patch[0:2, start : start + 2] = int(color)

    return patch, norm


def _make_target_patch(
    shape: tuple[tuple[int, int], ...],
    legend_orientation: tuple[tuple[int, int], ...] | None = None,
    force_different: bool = False,
) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    variants = _dihedral_variants(shape)
    choices = variants
    if force_different and legend_orientation is not None:
        different = [item for item in variants if item[0] != legend_orientation]
        if different:
            choices = different
    norm, _h, _w, _rot, _flip = random.choice(choices)
    return _render_shape(norm, GREY), norm


def _place_patches(
    height: int, width: int, patches: list[np.ndarray]
) -> tuple[np.ndarray, dict[int, tuple[int, int]]] | None:
    canvas = np.full((height, width), BACKGROUND, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)
    order = sorted(
        range(len(patches)),
        key=lambda idx: patches[idx].shape[0] * patches[idx].shape[1],
        reverse=True,
    )
    positions: dict[int, tuple[int, int]] = {}

    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        r_min = PLACEMENT_BORDER
        c_min = PLACEMENT_BORDER
        r_max = height - ph - PLACEMENT_BORDER
        c_max = width - pw - PLACEMENT_BORDER
        if r_max < r_min or c_max < c_min:
            return None

        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = random.randint(r_min, r_max)
            c0 = random.randint(c_min, c_max)
            rr0 = max(0, r0 - PLACEMENT_PADDING)
            cc0 = max(0, c0 - PLACEMENT_PADDING)
            rr1 = min(height, r0 + ph + PLACEMENT_PADDING)
            cc1 = min(width, c0 + pw + PLACEMENT_PADDING)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue

            sub = canvas[r0 : r0 + ph, c0 : c0 + pw]
            mask = patch != BACKGROUND
            if np.any(sub[mask] != BACKGROUND):
                continue
            sub[mask] = patch[mask]
            occupied[rr0:rr1, cc0:cc1] = True
            positions[idx] = (int(r0), int(c0))
            placed = True
            break

        if not placed:
            return None

    return canvas, positions


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    legend_count = random.choice(LEGEND_COUNT_CHOICES)
    shapes = _sample_unique_shapes(legend_count)
    if shapes is None:
        return None

    colors = random.sample(COLOR_POOL, legend_count)
    legend_patches: list[np.ndarray] = []
    legend_orientations: list[tuple[tuple[int, int], ...]] = []
    for shape, color in zip(shapes, colors, strict=False):
        patch, orientation = _make_legend_patch(shape, int(color))
        legend_patches.append(patch)
        legend_orientations.append(orientation)

    target_count = legend_count + random.randint(*TARGET_EXTRA_RANGE)
    assignments = list(range(legend_count))
    assignments.extend(random.randrange(legend_count) for _ in range(target_count - legend_count))
    random.shuffle(assignments)

    target_patches: list[np.ndarray] = []
    target_orientations: list[tuple[tuple[int, int], ...]] = []
    target_colors: list[int] = []
    for legend_idx in assignments:
        patch, orientation = _make_target_patch(shapes[legend_idx])
        target_patches.append(patch)
        target_orientations.append(orientation)
        target_colors.append(int(colors[legend_idx]))

    if not any(
        target_orientations[i] != legend_orientations[assignments[i]]
        for i in range(target_count)
    ):
        for i, legend_idx in enumerate(assignments):
            patch, orientation = _make_target_patch(
                shapes[legend_idx],
                legend_orientation=legend_orientations[legend_idx],
                force_different=True,
            )
            if orientation != legend_orientations[legend_idx]:
                target_patches[i] = patch
                target_orientations[i] = orientation
                break

    max_legend_h = max(p.shape[0] for p in legend_patches)
    max_legend_w = max(p.shape[1] for p in legend_patches)
    max_target_h = max(p.shape[0] for p in target_patches)
    max_target_w = max(p.shape[1] for p in target_patches)

    upper_min = max(UPPER_HEIGHT_RANGE[0], max_legend_h + 2)
    lower_min = max(LOWER_HEIGHT_RANGE[0], max_target_h + 2)
    if upper_min + 1 + lower_min > 30:
        return None

    upper_max = min(UPPER_HEIGHT_RANGE[1], 30 - 1 - lower_min)
    if upper_min > upper_max:
        return None
    upper_h = random.randint(upper_min, upper_max)

    lower_max = min(LOWER_HEIGHT_RANGE[1], 30 - upper_h - 1)
    if lower_min > lower_max:
        return None
    lower_h = random.randint(lower_min, lower_max)

    width_min = max(
        WIDTH_RANGE[0],
        max_legend_w + 2,
        max_target_w + 2,
        5 * legend_count + 4,
    )
    width_max = min(WIDTH_RANGE[1], 30)
    if width_min > width_max:
        return None
    width = random.randint(width_min, width_max)

    upper_result = _place_patches(upper_h, width, legend_patches)
    lower_result = _place_patches(lower_h, width, target_patches)
    if upper_result is None or lower_result is None:
        return None
    upper, _legend_pos = upper_result
    lower, target_pos = lower_result

    grid = np.full((upper_h + 1 + lower_h, width), BACKGROUND, dtype=int)
    grid[:upper_h, :] = upper
    grid[upper_h, :] = SEPARATOR
    grid[upper_h + 1 :, :] = lower

    expected = grid.copy()
    for idx, patch in enumerate(target_patches):
        r0, c0 = target_pos[idx]
        ph, pw = patch.shape
        mask = patch != BACKGROUND
        sub = expected[upper_h + 1 + r0 : upper_h + 1 + r0 + ph, c0 : c0 + pw]
        sub[mask] = int(target_colors[idx])

    if not np.array_equal(main(grid), expected):
        return None

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        case = _generate_case()
        if case is not None:
            grid, _expected = case
            return grid
    raise RuntimeError("failed to generate a valid input grid")
