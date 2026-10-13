from __future__ import annotations

import random

import numpy as np


BG = 0
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
ARC_MAX = 30


def _components_for_color(g: np.ndarray, color: int) -> list[dict]:
    mask = g == color
    seen = np.zeros_like(mask, dtype=bool)
    components: list[dict] = []
    h, w = mask.shape

    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue

        stack = [(sr, sc)]
        seen[sr, sc] = True
        coords: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            coords.append((r, c))
            for dr, dc in DIRS4:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))

        rows = [r for r, _c in coords]
        cols = [c for _r, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in coords:
            local[r - r0, c - c0] = True
        components.append(
            {
                "mask": local,
                "area": len(coords),
                "bbox": (r0, c0, r1, c1),
            }
        )

    return components


def _merged_rectangle_dims(left: np.ndarray, right: np.ndarray, offset: int) -> tuple[int, int] | None:
    if left.shape[0] != right.shape[0]:
        return None
    h = int(left.shape[0])
    w_left = int(left.shape[1])
    w_right = int(right.shape[1])
    width = max(w_left, offset + w_right)

    canvas = np.zeros((h, width), dtype=bool)
    canvas[:, :w_left] = left
    target = canvas[:, offset : offset + w_right]
    if np.any(target & right):
        return None
    target |= right
    if not np.all(canvas):
        return None
    return h, width


def _rectangle_dims_from_pair(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[int, int]:
    dims: set[tuple[int, int]] = set()
    for original, mirrored_piece in ((mask_a, mask_b), (mask_b, mask_a)):
        right = np.fliplr(mirrored_piece)
        for offset in range(original.shape[1] + 1):
            found = _merged_rectangle_dims(original, right, offset)
            if found is not None:
                dims.add(found)

    assert dims, "same-colored fragments must mirror-fit into a solid rectangle"
    assert len(dims) == 1, "fragment pair must have an unambiguous rectangle"
    return next(iter(dims))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair mirrored color fragments into solid rectangles and stack them by area.

    Input structure:
    - The background is black.
    - Each non-background color appears as exactly two 4-connected fragments.
    - For each color, one fragment can be reflected across a vertical axis and fitted to the right of the other.

    Rule:
    - Pair the two fragments of each color.
    - Mirror one fragment horizontally and merge it with the other at the unique right-side offset that makes a solid rectangle.
    - Sort the reconstructed rectangles by increasing area and stack them top to bottom, left aligned, with one black row between rectangles.

    Key insight:
    - Color gives the pairing, but the output order is controlled by the hidden area of the completed mirrored rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    colors = [int(c) for c in np.unique(g) if int(c) != BG]
    assert colors, "expected at least one foreground color"

    rectangles: list[tuple[int, int, int, int]] = []
    for color in colors:
        comps = _components_for_color(g, color)
        assert len(comps) == 2, "each foreground color must form exactly two objects"
        assert all(int(comp["area"]) <= 12 for comp in comps), "each object must be small"
        h, w = _rectangle_dims_from_pair(comps[0]["mask"], comps[1]["mask"])
        rectangles.append((h * w, h, w, color))

    areas = [area for area, _h, _w, _color in rectangles]
    assert len(set(areas)) == len(areas), "rectangle areas must define a unique order"
    rectangles.sort(key=lambda item: item[0])

    out_h = sum(h for _area, h, _w, _color in rectangles) + len(rectangles) - 1
    out_w = max(w for _area, _h, w, _color in rectangles)
    assert 1 <= out_h <= ARC_MAX and 1 <= out_w <= ARC_MAX

    out = np.full((out_h, out_w), BG, dtype=int)
    row = 0
    for _area, h, w, color in rectangles:
        out[row : row + h, :w] = color
        row += h + 1
    return out


# --------------- generator ---------------

COLOR_POOL = list(range(1, 10))
PAIR_COUNT_RANGE = (3, 5)
RECT_HEIGHT_RANGE = (2, 5)
RECT_WIDTH_RANGE = (3, 8)
MAX_OBJECT_AREA = 12
MAX_GENERATION_ATTEMPTS = 300
MAX_RECT_ATTEMPTS = 200
MAX_PLACE_ATTEMPTS = 200


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def _seam_to_masks(seam: list[int], width: int) -> tuple[np.ndarray, np.ndarray]:
    height = len(seam)
    left_w = max(seam)
    right_start = min(seam)
    left = np.zeros((height, left_w), dtype=bool)
    right = np.zeros((height, width - right_start), dtype=bool)

    for r, cut in enumerate(seam):
        left[r, :cut] = True
        right[r, cut - right_start :] = True

    return left, np.fliplr(right)


def _sample_seam(height: int, width: int, jagged: bool) -> list[int] | None:
    for _ in range(MAX_RECT_ATTEMPTS):
        if jagged:
            seam = [random.randint(1, width - 1)]
            for _r in range(1, height):
                seam.append(_clamp(seam[-1] + random.choice([-1, 0, 1]), 1, width - 1))
            if len(set(seam)) == 1:
                idx = random.randrange(height)
                choices = [v for v in range(1, width) if v != seam[idx]]
                seam[idx] = random.choice(choices)
        else:
            seam = [random.randint(1, width - 1)] * height

        left_area = sum(seam)
        right_area = height * width - left_area
        if left_area <= MAX_OBJECT_AREA and right_area <= MAX_OBJECT_AREA:
            return seam
    return None


def _sample_piece_pair(force_jagged: bool) -> tuple[int, int, np.ndarray, np.ndarray, bool] | None:
    for _ in range(MAX_RECT_ATTEMPTS):
        height = random.randint(*RECT_HEIGHT_RANGE)
        max_width = min(RECT_WIDTH_RANGE[1], (2 * MAX_OBJECT_AREA) // height)
        if max_width < RECT_WIDTH_RANGE[0]:
            continue
        width = random.randint(RECT_WIDTH_RANGE[0], max_width)
        jagged = force_jagged or random.random() < 0.6
        seam = _sample_seam(height, width, jagged)
        if seam is None and jagged and not force_jagged:
            jagged = False
            seam = _sample_seam(height, width, jagged)
        if seam is None:
            continue

        mask_a, mask_b = _seam_to_masks(seam, width)
        if int(mask_a.sum()) > MAX_OBJECT_AREA or int(mask_b.sum()) > MAX_OBJECT_AREA:
            continue
        try:
            if _rectangle_dims_from_pair(mask_a, mask_b) != (height, width):
                continue
        except AssertionError:
            continue
        return height, width, mask_a, mask_b, len(set(seam)) > 1
    return None


def _sample_rectangles(n_pairs: int, colors: list[int]) -> list[dict] | None:
    specs: list[dict] = []
    used_areas: set[int] = set()
    saw_jagged = False

    for idx, color in enumerate(colors):
        force_jagged = idx == 0
        for _ in range(MAX_RECT_ATTEMPTS):
            sampled = _sample_piece_pair(force_jagged)
            if sampled is None:
                return None
            height, width, mask_a, mask_b, jagged = sampled
            area = height * width
            if area in used_areas:
                continue
            used_areas.add(area)
            saw_jagged = saw_jagged or jagged
            specs.append(
                {
                    "color": color,
                    "height": height,
                    "width": width,
                    "area": area,
                    "masks": [mask_a, mask_b],
                }
            )
            break
        else:
            return None

    if not saw_jagged:
        return None
    out_h = sum(int(spec["height"]) for spec in specs) + n_pairs - 1
    if out_h > ARC_MAX:
        return None
    return specs


def _expected_output(specs: list[dict]) -> np.ndarray:
    ordered = sorted(specs, key=lambda spec: int(spec["area"]))
    out_h = sum(int(spec["height"]) for spec in ordered) + len(ordered) - 1
    out_w = max(int(spec["width"]) for spec in ordered)
    out = np.full((out_h, out_w), BG, dtype=int)
    row = 0
    for spec in ordered:
        h = int(spec["height"])
        w = int(spec["width"])
        out[row : row + h, :w] = int(spec["color"])
        row += h + 1
    return out


def _can_place(blocked: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> bool:
    h, w = mask.shape
    return not bool(blocked[r0 : r0 + h, c0 : c0 + w].any())


def _mark_blocked(blocked: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    row0 = max(0, r0 - 1)
    col0 = max(0, c0 - 1)
    row1 = min(blocked.shape[0], r0 + h + 1)
    col1 = min(blocked.shape[1], c0 + w + 1)
    blocked[row0:row1, col0:col1] = True


def _place_fragment(grid: np.ndarray, blocked: np.ndarray, color: int, mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h > grid.shape[0] or w > grid.shape[1]:
        return False
    positions = [
        (r, c)
        for r in range(grid.shape[0] - h + 1)
        for c in range(grid.shape[1] - w + 1)
        if _can_place(blocked, mask, r, c)
    ]
    if not positions:
        return False

    r0, c0 = random.choice(positions)
    patch = grid[r0 : r0 + h, c0 : c0 + w]
    patch[mask] = color
    _mark_blocked(blocked, r0, c0, h, w)
    return True


def _render_input(specs: list[dict]) -> np.ndarray | None:
    fragments: list[tuple[int, np.ndarray]] = []
    for spec in specs:
        for mask in spec["masks"]:
            fragments.append((int(spec["color"]), mask))
    random.shuffle(fragments)
    fragments.sort(key=lambda item: int(np.prod(item[1].shape)), reverse=True)

    min_side = max(12 + 2 * len(specs), max(max(mask.shape) for _color, mask in fragments) + 2)
    if min_side > ARC_MAX:
        return None
    grid_h = random.randint(min_side, ARC_MAX)
    grid_w = random.randint(min_side, ARC_MAX)
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    blocked = np.zeros_like(grid, dtype=bool)

    for color, mask in fragments:
        if not _place_fragment(grid, blocked, color, mask):
            return None
    return grid


def _gen_attempt() -> np.ndarray | None:
    n_pairs = random.randint(*PAIR_COUNT_RANGE)
    colors = random.sample(COLOR_POOL, n_pairs)
    specs = _sample_rectangles(n_pairs, colors)
    if specs is None:
        return None

    expected = _expected_output(specs)
    if max(expected.shape) > ARC_MAX:
        return None

    for _ in range(MAX_PLACE_ATTEMPTS):
        grid = _render_input(specs)
        if grid is None:
            continue
        try:
            solved = main(grid)
        except AssertionError:
            continue
        if np.array_equal(solved, expected):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
