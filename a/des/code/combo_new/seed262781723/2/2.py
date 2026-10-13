from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
ARC_MAX_DIM = 30
SEARCH_RADIUS = ARC_MAX_DIM
GENERATION_ATTEMPTS = 400
PAIR_ATTEMPTS = 120
MIN_RECT_SIDE = 3
SIZE_RANGE_BY_PAIR_COUNT = {
    2: (6, 11),
    3: (6, 8),
    4: (5, 8),
}
LAYOUTS_BY_PIECE_COUNT = {
    4: [(2, 2)],
    6: [(2, 3), (3, 2)],
    8: [(3, 3)],
}


def _component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue: deque[tuple[int, int]] = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(coords)

    return components


def _is_connected(coords: set[tuple[int, int]]) -> bool:
    if not coords:
        return False
    start = next(iter(coords))
    queue: deque[tuple[int, int]] = deque([start])
    seen = {start}
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (r + dr, c + dc)
            if nxt in coords and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return len(seen) == len(coords)


def _bbox(coords: set[tuple[int, int]] | list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: set[tuple[int, int]] | list[tuple[int, int]]) -> set[tuple[int, int]]:
    r0, c0, _r1, _c1 = _bbox(coords)
    return {(r - r0, c - c0) for r, c in coords}


def _coords_to_patch(coords: set[tuple[int, int]], color: int) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    for r, c in coords:
        patch[r - r0, c - c0] = int(color)
    return patch


def _colors_in_scan_order(grid: np.ndarray) -> list[int]:
    colors: list[int] = []
    seen: set[int] = set()
    for value in map(int, grid.ravel()):
        if value == BG or value in seen:
            continue
        seen.add(value)
        colors.append(value)
    return colors


def _find_rectangle_assemblies(
    a_coords: set[tuple[int, int]],
    b_coords: set[tuple[int, int]],
) -> list[tuple[int, int, int, int]]:
    a_rows = [r for r, _ in a_coords]
    a_cols = [c for _, c in a_coords]
    b_list = list(b_coords)
    total_area = len(a_coords) + len(b_coords)

    valid: list[tuple[int, int, int, int]] = []
    for dr in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1):
        for dc in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1):
            shifted = {(r + dr, c + dc) for r, c in b_list}
            if a_coords & shifted:
                continue
            rows = a_rows + [r for r, _ in shifted]
            cols = a_cols + [c for _, c in shifted]
            h = max(rows) - min(rows) + 1
            w = max(cols) - min(cols) + 1
            if h * w == total_area:
                valid.append((dr, dc, h, w))
    return valid


def _assembled_rectangle_dims(
    a_coords: set[tuple[int, int]],
    b_coords: set[tuple[int, int]],
) -> tuple[int, int]:
    valid = _find_rectangle_assemblies(a_coords, b_coords)
    assert len(valid) == 1, "same-color pieces must assemble into one unique rectangle"
    _dr, _dc, h, w = valid[0]
    return int(h), int(w)


def main(grid: np.ndarray) -> np.ndarray:
    """Pair each color's complementary fragments into a solid rectangle, then center-stack the rectangles by area.

    Input structure:
    - The background is black.
    - Several colored 4-connected components are scattered on the canvas.
    - Each non-black color appears in exactly two components, and those two shapes are complementary fragments of one axis-aligned rectangle without rotation or flipping.

    Rule:
    - Pair the two components of each color and find the unique translation that makes their union a gap-free filled rectangle.
    - Compute each reconstructed rectangle's area.
    - Create a black output canvas whose size matches the largest reconstructed rectangle.
    - Paint the rectangles centered on that canvas from largest area to smallest area, using top-left bias when centering is half-cell ambiguous; later, smaller rectangles overwrite earlier ones.

    Key insight:
    - The task has two separate steps: first recover each hidden rectangle by translation-only complementary fitting, then ignore the scattered layout and reorder the completed rectangles solely by area for nested stacking.
    """
    g = np.asarray(grid, dtype=int)
    colors = _colors_in_scan_order(g)
    assert colors

    rectangles: list[dict[str, int]] = []
    for color in colors:
        components = _component_coords(g == int(color))
        assert len(components) == 2, "each color must appear in exactly two components"
        comp_a = _normalize(components[0])
        comp_b = _normalize(components[1])
        h, w = _assembled_rectangle_dims(comp_a, comp_b)
        rectangles.append(
            {
                "color": int(color),
                "h": int(h),
                "w": int(w),
                "area": int(h * w),
            }
        )

    largest = max(rectangles, key=lambda item: item["area"])
    out_h = int(largest["h"])
    out_w = int(largest["w"])
    out = np.full((out_h, out_w), BG, dtype=int)

    for rect in sorted(rectangles, key=lambda item: item["area"], reverse=True):
        h = int(rect["h"])
        w = int(rect["w"])
        assert h <= out_h and w <= out_w
        r0 = (out_h - h) // 2
        c0 = (out_w - w) // 2
        out[r0 : r0 + h, c0 : c0 + w] = int(rect["color"])

    return out


def _random_profile(length: int, limit: int) -> list[int]:
    assert length >= 1 and limit >= 2
    values = [random.randint(1, limit - 1)]
    for _ in range(length - 1):
        step = random.choice([-2, -1, 0, 1, 2])
        nxt = min(limit - 1, max(1, values[-1] + step))
        values.append(nxt)
    if len(set(values)) == 1:
        idx = random.randrange(length)
        values[idx] = 1 if values[idx] != 1 else limit - 1
    return values


def _split_rectangle(h: int, w: int, vertical: bool) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    a_coords: set[tuple[int, int]] = set()
    b_coords: set[tuple[int, int]] = set()

    if vertical:
        profile = _random_profile(h, w)
        for r, boundary in enumerate(profile):
            for c in range(w):
                (a_coords if c < boundary else b_coords).add((r, c))
    else:
        profile = _random_profile(w, h)
        for c, boundary in enumerate(profile):
            for r in range(h):
                (a_coords if r < boundary else b_coords).add((r, c))

    return a_coords, b_coords


def _make_piece_pair(h: int, w: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    for _ in range(PAIR_ATTEMPTS):
        a_coords, b_coords = _split_rectangle(h, w, vertical=bool(random.getrandbits(1)))
        if not _is_connected(a_coords) or not _is_connected(b_coords):
            continue
        norm_a = _normalize(a_coords)
        norm_b = _normalize(b_coords)
        valid = _find_rectangle_assemblies(norm_a, norm_b)
        if len(valid) != 1:
            continue
        _dr, _dc, out_h, out_w = valid[0]
        if (int(out_h), int(out_w)) != (int(h), int(w)):
            continue
        return a_coords, b_coords
    return None


def _choose_nested_rectangles(pair_count: int) -> list[tuple[int, int]] | None:
    low, high = SIZE_RANGE_BY_PAIR_COUNT[pair_count]

    def backtrack(
        prev_h: int,
        prev_w: int,
        used_areas: set[int],
        remaining: int,
    ) -> list[tuple[int, int]] | None:
        if remaining == 0:
            return []
        candidates = [
            (h, w)
            for h in range(MIN_RECT_SIDE, prev_h + 1)
            for w in range(MIN_RECT_SIDE, prev_w + 1)
            if (h, w) != (prev_h, prev_w) and h * w < prev_h * prev_w and h * w not in used_areas
        ]
        random.shuffle(candidates)
        for h, w in candidates:
            used_areas.add(h * w)
            tail = backtrack(h, w, used_areas, remaining - 1)
            if tail is not None:
                return [(h, w)] + tail
            used_areas.remove(h * w)
        return None

    for _ in range(120):
        largest_h = random.randint(low, high)
        largest_w = random.randint(low, high)
        tail = backtrack(largest_h, largest_w, {largest_h * largest_w}, pair_count - 1)
        if tail is None:
            continue
        dims = [(largest_h, largest_w)] + tail
        if pair_count >= 3:
            exact_center = False
            biased_center = False
            for h, w in dims[1:]:
                if (largest_h - h) % 2 == 0 and (largest_w - w) % 2 == 0:
                    exact_center = True
                else:
                    biased_center = True
            if not (exact_center and biased_center):
                continue
        return dims
    return None


def _render_expected(rectangles: list[dict[str, int]]) -> np.ndarray:
    largest = max(rectangles, key=lambda item: item["area"])
    out_h = int(largest["h"])
    out_w = int(largest["w"])
    out = np.full((out_h, out_w), BG, dtype=int)
    for rect in sorted(rectangles, key=lambda item: item["area"], reverse=True):
        h = int(rect["h"])
        w = int(rect["w"])
        r0 = (out_h - h) // 2
        c0 = (out_w - w) // 2
        out[r0 : r0 + h, c0 : c0 + w] = int(rect["color"])
    return out


def _pad_grid(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    extra_h = ARC_MAX_DIM - h
    extra_w = ARC_MAX_DIM - w
    top = random.randint(0, extra_h)
    bottom = random.randint(0, extra_h - top)
    left = random.randint(0, extra_w)
    right = random.randint(0, extra_w - left)
    return np.pad(grid, ((top, bottom), (left, right)), constant_values=BG)


def _build_input(
    pieces: list[dict[str, object]],
) -> np.ndarray | None:
    count = len(pieces)
    layouts = list(LAYOUTS_BY_PIECE_COUNT[count])
    random.shuffle(layouts)
    for rows, cols in layouts:
        max_h = max(int(piece["patch"].shape[0]) for piece in pieces)
        max_w = max(int(piece["patch"].shape[1]) for piece in pieces)
        step_h = max_h + 2
        step_w = max_w + 2
        base_h = rows * step_h - 1
        base_w = cols * step_w - 1
        if base_h > ARC_MAX_DIM or base_w > ARC_MAX_DIM:
            continue

        grid = np.full((base_h, base_w), BG, dtype=int)
        slots = [(r, c) for r in range(rows) for c in range(cols)]
        random.shuffle(slots)
        order = pieces[:]
        random.shuffle(order)

        for piece, (slot_r, slot_c) in zip(order, slots):
            patch = piece["patch"]
            ph, pw = map(int, patch.shape)
            anchor_r = slot_r * step_h
            anchor_c = slot_c * step_w
            max_off_r = step_h - ph - 1
            max_off_c = step_w - pw - 1
            assert max_off_r >= 0 and max_off_c >= 0
            off_r = random.randint(0, max_off_r)
            off_c = random.randint(0, max_off_c)
            r0 = anchor_r + off_r
            c0 = anchor_c + off_c
            mask = patch != BG
            grid[r0 : r0 + ph, c0 : c0 + pw][mask] = patch[mask]

        return _pad_grid(grid)
    return None


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(2, 4)
    dims = _choose_nested_rectangles(pair_count)
    if dims is None:
        return None

    colors = random.sample(PALETTE, pair_count)
    rectangles: list[dict[str, int]] = []
    pieces: list[dict[str, object]] = []

    for color, (h, w) in zip(colors, dims):
        pair = _make_piece_pair(h, w)
        if pair is None:
            return None
        a_coords, b_coords = pair
        rectangles.append(
            {
                "color": int(color),
                "h": int(h),
                "w": int(w),
                "area": int(h * w),
            }
        )
        pieces.append({"color": int(color), "patch": _coords_to_patch(a_coords, int(color))})
        pieces.append({"color": int(color), "patch": _coords_to_patch(b_coords, int(color))})

    inp = _build_input(pieces)
    if inp is None:
        return None
    if max(inp.shape) > ARC_MAX_DIM:
        return None

    expected = _render_expected(rectangles)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
