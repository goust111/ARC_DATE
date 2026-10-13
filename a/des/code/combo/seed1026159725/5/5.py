from __future__ import annotations

import random

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
SIDE_TO_K = {
    "up": 0,
    "right": 3,
    "down": 2,
    "left": 1,
}


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _component_from(seed: tuple[int, int], mask: np.ndarray, seen: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    stack = [seed]
    seen[seed] = True
    coords: list[tuple[int, int]] = []
    while stack:
        r, c = stack.pop()
        coords.append((int(r), int(c)))
        for dr, dc in DIRS4:
            nr = int(r + dr)
            nc = int(c + dc)
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return np.array(coords, dtype=int)


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros(mask.shape, dtype=bool)
    parts: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        parts.append(_component_from((rr, cc), mask, seen))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _component_patch(coords: np.ndarray, color: int, bg: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for r, c in coords:
        patch[int(r) - r0, int(c) - c0] = int(color)
    return patch, (r0, c0, r1, c1)


def _side_from_marker(center: tuple[int, int], marker: tuple[int, int]) -> str:
    cr, cc = center
    mr, mc = marker
    if mc == cc and mr < cr:
        return "up"
    if mr == cr and mc > cc:
        return "right"
    if mc == cc and mr > cr:
        return "down"
    if mr == cr and mc < cc:
        return "left"
    raise AssertionError("marker must be axis-aligned with the object center")


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each colored object according to the side of its same-color marker, then move it to that marker.

    Input structure
    - The dominant color is the background.
    - Each non-background color appears in exactly two 4-connected components:
      one multi-pixel object and one singleton marker.
    - The singleton lies strictly above, right, below, or left of the object's center.

    Rule
    - Pair each multi-pixel object with the singleton of the same color.
    - Marker above  -> rotate 0 degrees.
    - Marker right  -> rotate 90 degrees clockwise.
    - Marker below  -> rotate 180 degrees.
    - Marker left   -> rotate 90 degrees counterclockwise.
    - Erase the original object and marker, then place the rotated object so its
      center aligns with the marker position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    out = np.full_like(g, int(bg))

    colors = [int(color) for color in np.unique(g) if int(color) != int(bg)]
    assert colors, "expected foreground colors"

    for color in colors:
        comps = _find_components(g == int(color))
        singletons = [coords for coords in comps if len(coords) == 1]
        objects = [coords for coords in comps if len(coords) > 1]
        assert len(singletons) == 1, f"expected one marker for color {color}"
        assert len(objects) == 1, f"expected one object for color {color}"

        obj_coords = objects[0]
        marker_r, marker_c = map(int, singletons[0][0])
        patch, (r0, c0, r1, c1) = _component_patch(obj_coords, color, bg)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        assert h % 2 == 1 and w % 2 == 1, "object bbox must have an integer center"

        center = (int(r0 + h // 2), int(c0 + w // 2))
        side = _side_from_marker(center, (marker_r, marker_c))
        rotated = np.rot90(patch, int(SIDE_TO_K[side]))
        rh, rw = map(int, rotated.shape)

        dst_r0 = int(marker_r - rh // 2)
        dst_c0 = int(marker_c - rw // 2)
        assert 0 <= dst_r0 and 0 <= dst_c0
        assert dst_r0 + rh <= g.shape[0] and dst_c0 + rw <= g.shape[1]

        for rr, cc in np.argwhere(rotated != int(bg)):
            tr = int(dst_r0 + int(rr))
            tc = int(dst_c0 + int(cc))
            assert out[tr, tc] == int(bg), "rotated objects must not overlap"
            out[tr, tc] = int(color)

    return out


# --------------- generator ---------------

GEN_SIDE_RANGE = (14, 22)
GEN_OBJECT_COUNT_RANGE = (1, 4)
GEN_ATTEMPTS = 400
GEN_PADDING = 1
GEN_MARKER_GAP_RANGE = (2, 6)
GEN_COLORS = tuple(range(10))
GEN_DIRECTIONS = ("up", "right", "down", "left")


def _normalize(coords: set[tuple[int, int]]) -> np.ndarray:
    pts = np.array(sorted(coords), dtype=int)
    r0, c0 = pts.min(axis=0)
    pts[:, 0] -= int(r0)
    pts[:, 1] -= int(c0)
    r1, c1 = pts.max(axis=0)
    patch = np.zeros((int(r1) + 1, int(c1) + 1), dtype=bool)
    patch[pts[:, 0], pts[:, 1]] = True
    return patch


def _shape_is_valid(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 3 or w < 3 or h > 5 or w > 5:
        return False
    if h % 2 == 0 or w % 2 == 0:
        return False
    if int(mask.sum()) < 4:
        return False
    if not bool(mask[h // 2, w // 2]):
        return False
    return all(not np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3))


def _random_shape() -> np.ndarray:
    for _ in range(200):
        target = random.randint(4, 9)
        cells = {(0, 0)}
        while len(cells) < target:
            frontier = set()
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr = int(r + dr)
                    nc = int(c + dc)
                    if -3 <= nr <= 3 and -3 <= nc <= 3:
                        frontier.add((nr, nc))
            frontier -= cells
            cells.add(random.choice(tuple(frontier)))
        mask = _normalize(cells)
        if _shape_is_valid(mask):
            return mask
    raise RuntimeError("failed to build a valid object shape")


def _side_order(count: int) -> list[str]:
    order = list(GEN_DIRECTIONS)
    random.shuffle(order)
    return order[:count]


def _expanded(cells: set[tuple[int, int]], shape: tuple[int, int], pad: int) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _patch_cells(mask: np.ndarray, r0: int, c0: int) -> set[tuple[int, int]]:
    cells = set()
    for rr, cc in np.argwhere(mask):
        cells.add((int(r0 + int(rr)), int(c0 + int(cc))))
    return cells


def _marker_from_center(center: tuple[int, int], side: str, gap: int) -> tuple[int, int]:
    r, c = center
    if side == "up":
        return int(r - gap), int(c)
    if side == "right":
        return int(r), int(c + gap)
    if side == "down":
        return int(r + gap), int(c)
    if side == "left":
        return int(r), int(c - gap)
    raise AssertionError(side)


def _touches_marker(marker: tuple[int, int], obj_cells: set[tuple[int, int]]) -> bool:
    mr, mc = marker
    for dr, dc in DIRS4:
        if (int(mr + dr), int(mc + dc)) in obj_cells:
            return True
    return False


def _attempt_layout() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(*GEN_SIDE_RANGE)
    w = random.randint(*GEN_SIDE_RANGE)
    bg = random.choice(GEN_COLORS)
    count = random.randint(*GEN_OBJECT_COUNT_RANGE)
    palette = [color for color in GEN_COLORS if color != bg]
    colors = random.sample(palette, count)
    sides = _side_order(count)

    inp = np.full((h, w), int(bg), dtype=int)
    expected = np.full_like(inp, int(bg))
    input_reserved: set[tuple[int, int]] = set()
    output_reserved: set[tuple[int, int]] = set()

    for color, side in zip(colors, sides):
        placed = False
        for _ in range(200):
            mask = _random_shape()
            patch = np.full(mask.shape, int(bg), dtype=int)
            patch[mask] = int(color)
            rotated = np.rot90(patch, int(SIDE_TO_K[side]))
            ph, pw = map(int, patch.shape)
            rh, rw = map(int, rotated.shape)

            max_r0 = h - ph
            max_c0 = w - pw
            if max_r0 < 0 or max_c0 < 0:
                return None

            src_r0 = random.randint(0, max_r0)
            src_c0 = random.randint(0, max_c0)
            center = (int(src_r0 + ph // 2), int(src_c0 + pw // 2))
            gap = random.randint(*GEN_MARKER_GAP_RANGE)
            marker = _marker_from_center(center, side, gap)
            mr, mc = marker
            if not (0 <= mr < h and 0 <= mc < w):
                continue

            dst_r0 = int(mr - rh // 2)
            dst_c0 = int(mc - rw // 2)
            if dst_r0 < 0 or dst_c0 < 0 or dst_r0 + rh > h or dst_c0 + rw > w:
                continue

            obj_cells = _patch_cells(mask, src_r0, src_c0)
            if marker in obj_cells or _touches_marker(marker, obj_cells):
                continue

            dst_cells = _patch_cells(rotated != bg, dst_r0, dst_c0)
            if _expanded(obj_cells | {marker}, (h, w), GEN_PADDING) & input_reserved:
                continue
            if _expanded(dst_cells, (h, w), GEN_PADDING) & output_reserved:
                continue

            for r, c in obj_cells:
                inp[r, c] = int(color)
            inp[mr, mc] = int(color)
            for r, c in dst_cells:
                expected[r, c] = int(color)

            input_reserved |= _expanded(obj_cells | {marker}, (h, w), GEN_PADDING)
            output_reserved |= _expanded(dst_cells, (h, w), GEN_PADDING)
            placed = True
            break

        if not placed:
            return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        result = _attempt_layout()
        if result is not None:
            inp, _expected = result
            return inp
    raise RuntimeError("failed to generate a valid input")
