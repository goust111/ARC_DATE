from __future__ import annotations

import random

import numpy as np


BG = 0
ARC_MAX = 30
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        coords: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            coords.append((cr, cc))
            for dr, dc in NEIGHBORS4:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _normalize(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    assert coords.size > 0
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0 = int(rows.min())
    c0 = int(cols.min())
    return tuple(sorted((int(r) - r0, int(c) - c0) for r, c in coords))


def _candidate_rectangles(
    a_coords: tuple[tuple[int, int], ...],
    b_coords: tuple[tuple[int, int], ...],
) -> list[tuple[int, int, int, int]]:
    a = set(a_coords)
    total = len(a_coords) + len(b_coords)
    candidates: list[tuple[int, int, int, int]] = []

    for dr in range(-ARC_MAX, ARC_MAX + 1):
        for dc in range(-ARC_MAX, ARC_MAX + 1):
            shifted = {(r + dr, c + dc) for r, c in b_coords}
            if a & shifted:
                continue
            union = a | shifted
            rows = [r for r, _c in union]
            cols = [c for _r, c in union]
            height = max(rows) - min(rows) + 1
            width = max(cols) - min(cols) + 1
            if height > ARC_MAX or width > ARC_MAX:
                continue
            if height * width == total:
                candidates.append((height, width, dr, dc))
    return candidates


def _completed_rectangle_shape(comp_a: np.ndarray, comp_b: np.ndarray) -> tuple[int, int]:
    a = _normalize(comp_a)
    b = _normalize(comp_b)
    candidates = _candidate_rectangles(a, b)
    assert candidates, "same-color fragments must complete a solid rectangle"

    dims = sorted({(height, width) for height, width, _dr, _dc in candidates})
    assert len(dims) == 1, "the completed rectangle shape must be unique"
    return dims[0]


def _foreground_colors_in_scan_order(g: np.ndarray) -> list[int]:
    colors: list[int] = []
    seen: set[int] = set()
    for value in g.ravel():
        color = int(value)
        if color != BG and color not in seen:
            seen.add(color)
            colors.append(color)
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct each same-color fragment pair as a solid rectangle and pack rectangles by increasing area.

    Input structure:
    - The background is black.
    - Every non-background color appears in exactly two separate 4-connected fragments.
    - The two fragments of each color are complementary under translation only:
      with no rotation or reflection, they can be shifted together into one solid rectangle.
    - Fragments of different colors are scattered with clear background gaps between them.

    Rule:
    - For each color, find its two 4-connected fragments and infer the unique
      translated placement that makes their union a filled rectangle.
    - Compute the completed rectangle's height, width, and area.
    - Draw one solid rectangle per color in a new black grid, ordered left to
      right by increasing rectangle area, with one black separator column.
    - All output rectangles are top-aligned at row 0.

    Key insight:
    The source fragments keep only enough shape information to identify their
    missing complementary partner; after each pair is completed, the final
    arrangement is controlled by rectangle area rather than original position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    specs: list[dict[str, int]] = []
    for scan_index, color in enumerate(_foreground_colors_in_scan_order(g)):
        comps = _components(g == int(color))
        assert len(comps) == 2, "each color must have exactly two fragments"
        height, width = _completed_rectangle_shape(comps[0], comps[1])
        specs.append(
            {
                "scan": int(scan_index),
                "color": int(color),
                "height": int(height),
                "width": int(width),
                "area": int(height * width),
            }
        )

    assert specs
    ordered = sorted(specs, key=lambda item: (item["area"], item["scan"]))
    out_h = max(item["height"] for item in ordered)
    out_w = sum(item["width"] for item in ordered) + len(ordered) - 1
    assert out_h <= ARC_MAX and out_w <= ARC_MAX

    out = np.zeros((out_h, out_w), dtype=int)
    col = 0
    for item in ordered:
        h = item["height"]
        w = item["width"]
        out[:h, col : col + w] = item["color"]
        col += w + 1
    return out


# --------------- generator ---------------

RECT_COUNT_RANGE = (2, 4)
RECT_SIDE_RANGE = (3, 7)
COLOR_POOL = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 500
MAX_SPLIT_ATTEMPTS = 200
MAX_PLACE_ATTEMPTS = 500


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _mask_coords(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(r), int(c)) for r, c in np.argwhere(mask)))


def _is_one_component(mask: np.ndarray) -> bool:
    return len(_components(mask)) == 1


def _is_solid_rect(mask: np.ndarray) -> bool:
    cropped = _crop_mask(mask)
    return int(cropped.sum()) == int(cropped.size)


def _split_rectangle(height: int, width: int) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(MAX_SPLIT_ATTEMPTS):
        vertical = random.choice([False, True])
        a = np.zeros((height, width), dtype=bool)

        if vertical:
            cuts = [random.randint(1, width - 1) for _r in range(height)]
            if len(set(cuts)) == 1:
                continue
            for r, cut in enumerate(cuts):
                a[r, :cut] = True
        else:
            cuts = [random.randint(1, height - 1) for _c in range(width)]
            if len(set(cuts)) == 1:
                continue
            for c, cut in enumerate(cuts):
                a[:cut, c] = True

        b = ~a
        if not (_is_one_component(a) and _is_one_component(b)):
            continue
        if _is_solid_rect(a) or _is_solid_rect(b):
            continue

        ca = _crop_mask(a)
        cb = _crop_mask(b)
        candidates = _candidate_rectangles(_mask_coords(ca), _mask_coords(cb))
        dims = {(h, w) for h, w, _dr, _dc in candidates}
        if dims == {(height, width)}:
            return ca, cb
    return None


def _sample_rectangles(n: int) -> list[tuple[int, int]]:
    choices = [
        (h, w)
        for h in range(RECT_SIDE_RANGE[0], RECT_SIDE_RANGE[1] + 1)
        for w in range(RECT_SIDE_RANGE[0], RECT_SIDE_RANGE[1] + 1)
    ]
    for _ in range(200):
        dims = random.sample(choices, n)
        areas = [h * w for h, w in dims]
        if len(set(areas)) != n:
            continue
        if sum(w for _h, w in dims) + n - 1 <= ARC_MAX:
            return dims
    raise RuntimeError("failed to sample rectangle dimensions")


def _mask_to_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _expected_output(specs: list[dict[str, object]]) -> np.ndarray:
    ordered = sorted(specs, key=lambda item: int(item["area"]))
    out_h = max(int(item["height"]) for item in ordered)
    out_w = sum(int(item["width"]) for item in ordered) + len(ordered) - 1
    out = np.zeros((out_h, out_w), dtype=int)
    col = 0
    for item in ordered:
        h = int(item["height"])
        w = int(item["width"])
        color = int(item["color"])
        out[:h, col : col + w] = color
        col += w + 1
    return out


def _can_place(occupied: np.ndarray, coords: np.ndarray, r0: int, c0: int) -> bool:
    h, w = occupied.shape
    for rr, cc in coords:
        r = int(r0 + rr)
        c = int(c0 + cc)
        if not (0 <= r < h and 0 <= c < w):
            return False
        r_start = max(0, r - 1)
        r_stop = min(h, r + 2)
        c_start = max(0, c - 1)
        c_stop = min(w, c + 2)
        if bool(occupied[r_start:r_stop, c_start:c_stop].any()):
            return False
    return True


def _place_fragments(
    shape: tuple[int, int],
    fragments: list[tuple[int, np.ndarray]],
) -> np.ndarray | None:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    order = list(fragments)
    random.shuffle(order)
    order.sort(key=lambda item: int(item[1].size), reverse=True)

    for color, mask in order:
        coords = np.argwhere(mask)
        mh, mw = mask.shape
        if mh > h or mw > w:
            return None

        for _ in range(MAX_PLACE_ATTEMPTS):
            r0 = random.randint(0, h - mh)
            c0 = random.randint(0, w - mw)
            if not _can_place(occupied, coords, r0, c0):
                continue
            abs_rows = coords[:, 0] + r0
            abs_cols = coords[:, 1] + c0
            grid[abs_rows, abs_cols] = int(color)
            occupied[abs_rows, abs_cols] = True
            break
        else:
            return None

    return grid


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*RECT_COUNT_RANGE)
    dims = _sample_rectangles(n)
    colors = random.sample(COLOR_POOL, n)

    specs: list[dict[str, object]] = []
    fragments: list[tuple[int, np.ndarray]] = []
    for color, (height, width) in zip(colors, dims):
        split = _split_rectangle(height, width)
        if split is None:
            return None
        first, second = split
        specs.append(
            {
                "color": int(color),
                "height": int(height),
                "width": int(width),
                "area": int(height * width),
            }
        )
        if random.choice([False, True]):
            first, second = second, first
        fragments.append((int(color), first))
        fragments.append((int(color), second))

    expected = _expected_output(specs)
    if max(expected.shape) > ARC_MAX:
        return None

    min_h = max(12, max(mask.shape[0] for _color, mask in fragments) + 2)
    min_w = max(12, max(mask.shape[1] for _color, mask in fragments) + 2)
    if min_h > ARC_MAX or min_w > ARC_MAX:
        return None
    grid_h = random.randint(min_h, ARC_MAX)
    grid_w = random.randint(min_w, ARC_MAX)
    grid = _place_fragments((grid_h, grid_w), fragments)
    if grid is None:
        return None

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
