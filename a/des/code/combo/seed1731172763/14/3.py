from __future__ import annotations

from collections import deque
import random

import numpy as np


BACKGROUND = 8
COMPLEMENT = {
    1: 4,  # blue -> yellow
    2: 3,  # red -> green
    3: 2,  # green -> red
    4: 1,  # yellow -> blue
    5: 6,  # grey -> magenta
    6: 5,  # magenta -> grey
    7: 9,  # orange -> maroon
    9: 7,  # maroon -> orange
}
CORNERS = ("tl", "tr", "bl", "br")
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIRS8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)


def _dominant_color(grid: np.ndarray) -> int:
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(np.argmax(counts))


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIRS4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _find_objects(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            objects.append({"color": color, "coords": arr, "bbox": (r0, c0, r1, c1)})

    return objects


def _corner_distances(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> list[tuple[float, str]]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    anchors = {
        "tl": (0.0, 0.0),
        "tr": (0.0, float(w - 1)),
        "bl": (float(h - 1), 0.0),
        "br": (float(h - 1), float(w - 1)),
    }
    out = []
    for corner in CORNERS:
        rr, cc = anchors[corner]
        dist2 = (center_r - rr) ** 2 + (center_c - cc) ** 2
        out.append((dist2, corner))
    out.sort(key=lambda item: (item[0], CORNERS.index(item[1])))
    return out


def _nearest_corner(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> str:
    return _corner_distances(bbox, shape)[0][1]


def _to_local(corner: str, r: int, c: int, h: int, w: int) -> tuple[int, int]:
    if corner == "tl":
        return r, c
    if corner == "tr":
        return r, w - 1 - c
    if corner == "bl":
        return h - 1 - r, c
    if corner == "br":
        return h - 1 - r, w - 1 - c
    raise AssertionError(f"unknown corner: {corner}")


def _from_local(corner: str, u: int, v: int, h: int, w: int) -> tuple[int, int]:
    if corner == "tl":
        return u, v
    if corner == "tr":
        return u, w - 1 - v
    if corner == "bl":
        return h - 1 - u, v
    if corner == "br":
        return h - 1 - u, w - 1 - v
    raise AssertionError(f"unknown corner: {corner}")


def _reflected_coords(coords: np.ndarray, corner: str, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    reflected: list[tuple[int, int]] = []
    for r_raw, c_raw in coords:
        r = int(r_raw)
        c = int(c_raw)
        u, v = _to_local(corner, r, c, h, w)
        rr, cc = _from_local(corner, v, u, h, w)
        assert 0 <= rr < h and 0 <= cc < w, "reflected copy must stay in bounds"
        reflected.append((rr, cc))
    return reflected


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each foreground object toward its nearest corner and recolor the copy by a fixed complement map.

    Input structure
    - The background is a uniform canvas.
    - Each foreground object is a 4-connected monochromatic component.
    - Each object's bounding-box center is closest to exactly one grid corner.

    Rule
    - Find the corner nearest to each object's bounding-box center.
    - If the nearest corner is top-left or bottom-right, reflect the object by swapping row/column offsets
      from that corner (main-diagonal flip).
    - If the nearest corner is top-right or bottom-left, do the analogous swap in that corner's local frame
      (anti-diagonal flip in global coordinates).
    - Draw the reflected copy with the object's complementary color and keep the original unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _dominant_color(g)
    out = g.copy()
    for obj in _find_objects(g, bg):
        color = int(obj["color"])
        assert color in COMPLEMENT, "unexpected foreground color"
        corner = _nearest_corner(obj["bbox"], g.shape)
        new_color = COMPLEMENT[color]
        for rr, cc in _reflected_coords(obj["coords"], corner, g.shape):
            out[rr, cc] = new_color
    return out


# --------------- generator ---------------

GRID_SIZES = (17, 19, 21, 23)
OBJECT_COUNT_RANGE = (4, 6)
OBJECT_SIZE_RANGE = (2, 5)
LOCAL_DIAGONAL_GAP = 2
MAX_PLACEMENT_ATTEMPTS = 80
MAX_GENERATION_ATTEMPTS = 400
SOURCE_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]


def _mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _grow_connected_shape(mask: np.ndarray, size: int) -> list[tuple[int, int]] | None:
    components = [comp for comp in _mask_components(mask) if len(comp) >= size]
    if not components:
        return None

    comp = random.choice(components)
    comp_set = set(comp)
    seed = random.choice(comp)
    chosen = {seed}
    frontier = {cell for cell in (_neighbors4(seed[0], seed[1], *mask.shape)) if cell in comp_set}

    while len(chosen) < size:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        chosen.add(cell)
        for nb in _neighbors4(cell[0], cell[1], *mask.shape):
            if nb in comp_set and nb not in chosen:
                frontier.add(nb)

    coords = list(chosen)
    random.shuffle(coords)
    return coords


def _side_mask(local_size: int, side: str) -> np.ndarray:
    rows, cols = np.indices((local_size, local_size))
    if side == "upper":
        return cols - rows >= LOCAL_DIAGONAL_GAP
    if side == "lower":
        return rows - cols >= LOCAL_DIAGONAL_GAP
    raise AssertionError(f"unknown side: {side}")


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    local_size = blocked.shape[0]
    for u, v in cells:
        for a, b in ((u, v), (v, u)):
            for du, dc in DIRS8:
                uu = a + du
                vv = b + dc
                if 0 <= uu < local_size and 0 <= vv < local_size:
                    blocked[uu, vv] = True
            blocked[a, b] = True


def _local_cells_to_global(corner: str, cells: list[tuple[int, int]], n: int) -> list[tuple[int, int]]:
    return [_from_local(corner, u, v, n, n) for u, v in cells]


def _place_local_shape(blocked: np.ndarray, corner: str, n: int, size: int) -> list[tuple[int, int]] | None:
    local_size = blocked.shape[0]
    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        side = random.choice(("upper", "lower"))
        available = _side_mask(local_size, side) & ~blocked & ~blocked.T
        cells = _grow_connected_shape(available, size)
        if cells is None:
            continue

        global_cells = _local_cells_to_global(corner, cells, n)
        bbox = (
            min(r for r, _ in global_cells),
            min(c for _, c in global_cells),
            max(r for r, _ in global_cells),
            max(c for _, c in global_cells),
        )
        ranked = _corner_distances(bbox, (n, n))
        if len(ranked) > 1 and abs(ranked[0][0] - ranked[1][0]) < 1e-9:
            continue
        if ranked[0][1] != corner:
            continue

        mirror_cells = [(v, u) for u, v in cells]
        if set(cells) & set(mirror_cells):
            continue
        return cells
    return None


def _render_objects(
    n: int,
    placements: list[tuple[str, list[tuple[int, int]], int]],
) -> tuple[np.ndarray, np.ndarray] | None:
    inp = np.full((n, n), BACKGROUND, dtype=int)
    out = inp.copy()

    for corner, cells, color in placements:
        for r, c in _local_cells_to_global(corner, cells, n):
            if int(inp[r, c]) != BACKGROUND:
                return None
            inp[r, c] = color
            out[r, c] = color

    for corner, cells, color in placements:
        mirror_color = COMPLEMENT[color]
        for r, c in _local_cells_to_global(corner, [(v, u) for u, v in cells], n):
            if int(out[r, c]) != BACKGROUND:
                return None
            out[r, c] = mirror_color

    return inp, out


def _attempt_generate() -> tuple[np.ndarray, np.ndarray] | None:
    n = random.choice(GRID_SIZES)
    total_objects = random.randint(*OBJECT_COUNT_RANGE)
    extra_corners = random.sample(CORNERS, total_objects - len(CORNERS))

    corner_counts = {corner: 1 for corner in CORNERS}
    for corner in extra_corners:
        corner_counts[corner] += 1

    local_size = n // 2
    blocked = {corner: np.zeros((local_size, local_size), dtype=bool) for corner in CORNERS}
    colors = random.sample(SOURCE_COLORS, total_objects)
    corner_order = [corner for corner in CORNERS for _ in range(corner_counts[corner])]
    random.shuffle(corner_order)

    placements: list[tuple[str, list[tuple[int, int]], int]] = []
    for color, corner in zip(colors, corner_order):
        max_size = OBJECT_SIZE_RANGE[1] if corner_counts[corner] == 1 else OBJECT_SIZE_RANGE[1] - 1
        size = random.randint(OBJECT_SIZE_RANGE[0], max_size)
        cells = _place_local_shape(blocked[corner], corner, n, size)
        if cells is None:
            return None
        _mark_blocked(blocked[corner], cells)
        placements.append((corner, cells, color))

    rendered = _render_objects(n, placements)
    if rendered is None:
        return None
    inp, expected = rendered

    objects = _find_objects(inp, BACKGROUND)
    if len(objects) != total_objects:
        return None
    for obj in objects:
        ranked = _corner_distances(obj["bbox"], inp.shape)
        if len(ranked) > 1 and abs(ranked[0][0] - ranked[1][0]) < 1e-9:
            return None

    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        generated = _attempt_generate()
        if generated is None:
            continue
        inp, expected = generated
        if np.array_equal(main(inp), expected):
            return inp
    raise RuntimeError("failed to generate a valid input for puzzle 3")
