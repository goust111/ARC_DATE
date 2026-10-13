import numpy as np
from collections import deque


BG = 0
PALETTE = tuple(range(1, 10))
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_RETRIES = 400
GRID_H_RANGE = (24, 33)
GRID_W_RANGE = (24, 33)
PAIR_COUNT_CHOICES = (2, 3, 4)
RECT_SIDE_MIN = 5
RECT_SIDE_MAX = 9
MIN_AREA_RATIO = 0.25
PAD = 1

_RNG = np.random.default_rng()


def _find_components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
            mask[arr[:, 0] - r0, arr[:, 1] - c0] = True
            comps.append(
                {
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": mask,
                }
            )

    return comps


def _all_rows_nonempty(mask: np.ndarray) -> bool:
    return bool(np.all(mask.any(axis=1)))


def _all_cols_nonempty(mask: np.ndarray) -> bool:
    return bool(np.all(mask.any(axis=0)))


def _rectangle_shapes_from_pair(mask_a: np.ndarray, mask_b: np.ndarray) -> set[tuple[int, int]]:
    shapes: set[tuple[int, int]] = set()

    def try_vertical(left_mask: np.ndarray, right_mask: np.ndarray) -> None:
        if left_mask.shape[0] != right_mask.shape[0]:
            return
        if not (_all_rows_nonempty(left_mask) and _all_rows_nonempty(right_mask)):
            return

        h, left_w = left_mask.shape
        right_w = right_mask.shape[1]
        for shift in range(1, left_w + 1):
            width = max(left_w, shift + right_w)
            canvas = np.zeros((h, width), dtype=bool)
            canvas[:, :left_w] = left_mask
            window = canvas[:, shift : shift + right_w]
            if np.any(window & right_mask):
                continue
            window |= right_mask
            if bool(canvas.all()):
                shapes.add((int(h), int(width)))

    def try_horizontal(top_mask: np.ndarray, bottom_mask: np.ndarray) -> None:
        if top_mask.shape[1] != bottom_mask.shape[1]:
            return
        if not (_all_cols_nonempty(top_mask) and _all_cols_nonempty(bottom_mask)):
            return

        top_h, w = top_mask.shape
        bottom_h = bottom_mask.shape[0]
        for shift in range(1, top_h + 1):
            height = max(top_h, shift + bottom_h)
            canvas = np.zeros((height, w), dtype=bool)
            canvas[:top_h, :] = top_mask
            window = canvas[shift : shift + bottom_h, :]
            if np.any(window & bottom_mask):
                continue
            window |= bottom_mask
            if bool(canvas.all()):
                shapes.add((int(height), int(w)))

    try_vertical(mask_a, mask_b)
    try_vertical(mask_b, mask_a)
    try_horizontal(mask_a, mask_b)
    try_horizontal(mask_b, mask_a)
    return shapes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-color fragments, reassemble each pair into a solid rectangle,
    and place that rectangle with its top-left corner at the lexicographically
    upper-left fragment's bbox origin.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = np.zeros_like(g)

    colors = [int(color) for color in np.unique(g) if int(color) != BG]
    for color in sorted(colors):
        comps = _find_components_of_color(g, color)
        assert len(comps) == 2, "each color must appear as exactly two fragments"

        shapes = _rectangle_shapes_from_pair(comps[0]["mask"], comps[1]["mask"])
        assert len(shapes) == 1, "fragment pair must reconstruct a unique rectangle"
        rect_h, rect_w = next(iter(shapes))

        anchor = min(comps, key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
        r0, c0 = anchor["bbox"][:2]
        assert r0 + rect_h <= h and c0 + rect_w <= w, "reconstructed rectangle must stay in bounds"
        region = out[r0 : r0 + rect_h, c0 : c0 + rect_w]
        assert not bool(np.any(region != BG)), "output rectangles must not overlap"
        region[:] = color

    return out


def _clear(occ: np.ndarray, r0: int, c0: int, hh: int, ww: int, pad: int = 0) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occ.shape[0], r0 + hh + pad)
    cc1 = min(occ.shape[1], c0 + ww + pad)
    return not bool(np.any(occ[rr0:rr1, cc0:cc1]))


def _mark(occ: np.ndarray, r0: int, c0: int, hh: int, ww: int, pad: int = 0) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occ.shape[0], r0 + hh + pad)
    cc1 = min(occ.shape[1], c0 + ww + pad)
    occ[rr0:rr1, cc0:cc1] = True


def _random_profile(length: int, low: int, high: int, rng: np.random.Generator) -> np.ndarray | None:
    if length < 2 or high - low + 1 < 2:
        return None

    run_count = int(rng.integers(2, min(5, length) + 1))
    cut_points = sorted(int(x) for x in rng.choice(np.arange(1, length), size=run_count - 1, replace=False))
    bounds = [0, *cut_points, length]

    values = []
    prev = None
    for _ in range(run_count):
        choices = [v for v in range(low, high + 1) if v != prev]
        if not choices:
            return None
        val = int(rng.choice(choices))
        values.append(val)
        prev = val

    profile = np.zeros(length, dtype=int)
    for idx in range(run_count):
        profile[bounds[idx] : bounds[idx + 1]] = values[idx]

    if len(set(int(x) for x in profile)) < 2:
        return None
    return profile


def _pair_is_good(mask0: np.ndarray, mask1: np.ndarray, rect_h: int, rect_w: int) -> bool:
    area0 = int(mask0.sum())
    area1 = int(mask1.sum())
    total = rect_h * rect_w
    if min(area0, area1) < int(np.ceil(total * MIN_AREA_RATIO)):
        return False
    return _rectangle_shapes_from_pair(mask0, mask1) == {(rect_h, rect_w)}


def _make_vertical_pair(rng: np.random.Generator) -> dict | None:
    for _ in range(120):
        rect_h = int(rng.integers(RECT_SIDE_MIN, RECT_SIDE_MAX + 1))
        rect_w = int(rng.integers(RECT_SIDE_MIN, RECT_SIDE_MAX + 1))
        profile = _random_profile(rect_h, 2, rect_w - 2, rng)
        if profile is None:
            continue

        left_w = int(profile.max())
        right_offset = int(profile.min())
        right_w = int(rect_w - right_offset)

        left = np.zeros((rect_h, left_w), dtype=bool)
        right = np.zeros((rect_h, right_w), dtype=bool)
        for r, x in enumerate(profile):
            left[r, : int(x)] = True
            right[r, int(x - right_offset) :] = True

        if _pair_is_good(left, right, rect_h, rect_w):
            return {"orientation": "vertical", "rect_shape": (rect_h, rect_w), "masks": [left, right]}
    return None


def _make_horizontal_pair(rng: np.random.Generator) -> dict | None:
    for _ in range(120):
        rect_h = int(rng.integers(RECT_SIDE_MIN, RECT_SIDE_MAX + 1))
        rect_w = int(rng.integers(RECT_SIDE_MIN, RECT_SIDE_MAX + 1))
        profile = _random_profile(rect_w, 2, rect_h - 2, rng)
        if profile is None:
            continue

        top_h = int(profile.max())
        bottom_offset = int(profile.min())
        bottom_h = int(rect_h - bottom_offset)

        top = np.zeros((top_h, rect_w), dtype=bool)
        bottom = np.zeros((bottom_h, rect_w), dtype=bool)
        for c, y in enumerate(profile):
            top[: int(y), c] = True
            bottom[int(y - bottom_offset) :, c] = True

        if _pair_is_good(top, bottom, rect_h, rect_w):
            return {"orientation": "horizontal", "rect_shape": (rect_h, rect_w), "masks": [top, bottom]}
    return None


def _make_pair(rng: np.random.Generator) -> dict | None:
    if int(rng.integers(2)) == 0:
        return _make_vertical_pair(rng)
    return _make_horizontal_pair(rng)


def _draw_patch(grid: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> None:
    hh, ww = mask.shape
    region = grid[top : top + hh, left : left + ww]
    assert not bool(np.any(region[mask] != BG))
    region[mask] = color


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    pair_count = int(rng.choice(PAIR_COUNT_CHOICES))
    colors = [int(x) for x in rng.choice(PALETTE, size=pair_count, replace=False)]

    items = []
    for color in colors:
        spec = _make_pair(rng)
        if spec is None:
            return None
        spec["color"] = color
        spec["anchor_idx"] = int(rng.integers(2))
        items.append(spec)

    rect_area_sum = sum(int(h * w) for h, w in (item["rect_shape"] for item in items))
    h = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1]))
    w = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1]))
    if h * w < rect_area_sum * 2:
        return None

    output_occ = np.zeros((h, w), dtype=bool)
    input_occ = np.zeros((h, w), dtype=bool)

    for item in sorted(items, key=lambda cur: cur["rect_shape"][0] * cur["rect_shape"][1], reverse=True):
        rect_h, rect_w = item["rect_shape"]
        anchor_mask = item["masks"][item["anchor_idx"]]
        partner_mask = item["masks"][1 - item["anchor_idx"]]
        anchor_h, anchor_w = anchor_mask.shape
        partner_h, partner_w = partner_mask.shape

        max_r = min(h - rect_h, h - partner_h - 1)
        max_c = min(w - rect_w, w - partner_w - 1)
        if max_r < 0 or max_c < 0:
            return None

        candidates = []
        for r0 in range(max_r + 1):
            for c0 in range(max_c + 1):
                if not _clear(output_occ, r0, c0, rect_h, rect_w, pad=PAD):
                    continue
                if not _clear(input_occ, r0, c0, anchor_h, anchor_w, pad=PAD):
                    continue
                candidates.append((r0, c0))

        if not candidates:
            return None

        item["anchor_pos"] = tuple(int(x) for x in candidates[int(rng.integers(len(candidates)))])
        _mark(output_occ, item["anchor_pos"][0], item["anchor_pos"][1], rect_h, rect_w, pad=PAD)
        _mark(input_occ, item["anchor_pos"][0], item["anchor_pos"][1], anchor_h, anchor_w, pad=PAD)

    for item in sorted(items, key=lambda cur: int(cur["masks"][1 - cur["anchor_idx"]].sum()), reverse=True):
        partner_mask = item["masks"][1 - item["anchor_idx"]]
        partner_h, partner_w = partner_mask.shape
        anchor_r, anchor_c = item["anchor_pos"]

        candidates = []
        for r0 in range(anchor_r + 1, h - partner_h + 1):
            for c0 in range(anchor_c + 1, w - partner_w + 1):
                if not _clear(input_occ, r0, c0, partner_h, partner_w, pad=PAD):
                    continue
                if not _clear(output_occ, r0, c0, partner_h, partner_w, pad=0):
                    continue
                candidates.append((r0, c0))

        if not candidates:
            return None

        item["partner_pos"] = tuple(int(x) for x in candidates[int(rng.integers(len(candidates)))])
        _mark(input_occ, item["partner_pos"][0], item["partner_pos"][1], partner_h, partner_w, pad=PAD)

    grid = np.zeros((h, w), dtype=int)
    expected = np.zeros((h, w), dtype=int)

    for item in items:
        color = int(item["color"])
        rect_h, rect_w = item["rect_shape"]
        anchor_idx = int(item["anchor_idx"])
        partner_idx = 1 - anchor_idx
        anchor_r, anchor_c = item["anchor_pos"]
        partner_r, partner_c = item["partner_pos"]

        _draw_patch(grid, item["masks"][anchor_idx], color, anchor_r, anchor_c)
        _draw_patch(grid, item["masks"][partner_idx], color, partner_r, partner_c)

        target = expected[anchor_r : anchor_r + rect_h, anchor_c : anchor_c + rect_w]
        if bool(np.any(target != BG)):
            return None
        target[:] = color

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
