from __future__ import annotations

import random

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
BACKGROUND = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]

MISSING_TO_DIRECTION = {
    (0, 1): "up",
    (1, 1): "right",
    (1, 0): "down",
    (0, 0): "left",
}
DIRECTION_TO_MISSING = {direction: missing for missing, direction in MISSING_TO_DIRECTION.items()}

GRID_ROWS_RANGE = (18, 26)
GRID_COLS_RANGE = (18, 26)
POINTER_COUNT_CHOICES = [1, 2, 3, 4]
SQUARE_SIZE_CHOICES = [2, 4, 6]
MAX_PLACEMENT_ATTEMPTS = 250
MAX_GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack = [(r, c)]
        coords: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            coords.append((int(rr), int(cc)))
            for dr, dc in DIR4:
                nr = int(rr + dr)
                nc = int(cc + dc)
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _background_color(grid: np.ndarray) -> int:
    if bool((grid == BACKGROUND).any()):
        return BACKGROUND
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _pointer_info(coords: np.ndarray) -> dict[str, int] | None:
    if int(coords.shape[0]) != 3:
        return None
    r0 = int(coords[:, 0].min())
    r1 = int(coords[:, 0].max())
    c0 = int(coords[:, 1].min())
    c1 = int(coords[:, 1].max())
    if r1 != r0 + 1 or c1 != c0 + 1:
        return None

    local = {(int(r - r0), int(c - c0)) for r, c in coords}
    full = {(0, 0), (0, 1), (1, 0), (1, 1)}
    missing = list(full - local)
    if len(missing) != 1:
        return None
    direction = MISSING_TO_DIRECTION[missing[0]]
    return {
        "r0": r0,
        "c0": c0,
        "missing_r": int(missing[0][0]),
        "missing_c": int(missing[0][1]),
        "direction": direction,
    }


def _ray_cells(pointer: dict[str, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    r0 = int(pointer["r0"])
    c0 = int(pointer["c0"])
    direction = pointer["direction"]
    if direction == "up":
        return [(r, c0 + 1) for r in range(r0 - 1, -1, -1)]
    if direction == "right":
        return [(r0 + 1, c) for c in range(c0 + 2, w)]
    if direction == "down":
        return [(r, c0) for r in range(r0 + 2, h)]
    return [(r0, c) for c in range(c0 - 1, -1, -1)]


def _square_top_left(pointer_r0: int, pointer_c0: int, side: int) -> tuple[int, int]:
    return int(pointer_r0 + 1 - side // 2), int(pointer_c0 + 1 - side // 2)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each 3-pixel L-shaped pointer plus its same-color ray markers by a centered solid square.

    The missing corner of the pointer's 2x2 box determines the search direction:
    - missing top-right -> up
    - missing bottom-right -> right
    - missing bottom-left -> down
    - missing top-left -> left

    All same-color singleton pixels on that directed row/column are counted, erased, and replaced
    by a solid square of that color whose side length equals the count.
    """
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    bg = _background_color(g)
    out = np.full((h, w), bg, dtype=int)

    colors = [int(color) for color in np.unique(g) if int(color) != int(bg)]
    for color in colors:
        comps = _components(g == int(color))
        pointer: dict[str, int] | None = None
        markers: list[tuple[int, int]] = []

        for comp in comps:
            info = _pointer_info(comp)
            if info is not None:
                assert pointer is None, "each color should have exactly one pointer"
                pointer = info
            else:
                assert int(comp.shape[0]) == 1, "non-pointer objects must be singleton markers"
                markers.append((int(comp[0, 0]), int(comp[0, 1])))

        assert pointer is not None, "expected one pointer per color"
        assert markers, "pointer must have at least one marker"

        allowed = set(_ray_cells(pointer, g.shape))
        marker_set = set(markers)
        assert marker_set <= allowed, "markers must lie on the pointer ray"

        side = len(markers)
        sr, sc = _square_top_left(int(pointer["r0"]), int(pointer["c0"]), side)
        assert 0 <= sr and 0 <= sc and sr + side <= h and sc + side <= w, "output square must fit"
        assert bool((out[sr : sr + side, sc : sc + side] == bg).all()), "output squares must not overlap"
        out[sr : sr + side, sc : sc + side] = int(color)

    return out


def _pointer_cells(pointer_r0: int, pointer_c0: int, direction: str) -> list[tuple[int, int]]:
    missing_r, missing_c = DIRECTION_TO_MISSING[direction]
    cells = []
    for dr in range(2):
        for dc in range(2):
            if dr == missing_r and dc == missing_c:
                continue
            cells.append((int(pointer_r0 + dr), int(pointer_c0 + dc)))
    return cells


def _reserve_square(mask: np.ndarray, top: int, left: int, side: int, pad: int = 1) -> None:
    r0 = max(0, int(top - pad))
    c0 = max(0, int(left - pad))
    r1 = min(mask.shape[0], int(top + side + pad))
    c1 = min(mask.shape[1], int(left + side + pad))
    mask[r0:r1, c0:c1] = True


def _square_is_free(mask: np.ndarray, top: int, left: int, side: int, pad: int = 1) -> bool:
    r0 = max(0, int(top - pad))
    c0 = max(0, int(left - pad))
    r1 = min(mask.shape[0], int(top + side + pad))
    c1 = min(mask.shape[1], int(left + side + pad))
    return not bool(mask[r0:r1, c0:c1].any())


def _pick_marker_cells(
    slots: list[tuple[int, int]], occupied: np.ndarray, count: int
) -> list[tuple[int, int]] | None:
    by_parity: list[list[tuple[int, int]]] = [[], []]
    for idx, (r, c) in enumerate(slots):
        if not occupied[int(r), int(c)]:
            by_parity[idx % 2].append((int(r), int(c)))

    choices = [cells for cells in by_parity if len(cells) >= count]
    if not choices:
        return None
    picked = random.sample(random.choice(choices), count)
    return sorted((int(r), int(c)) for r, c in picked)


def _place_pointer(
    h: int,
    w: int,
    direction: str,
    color: int,
    side: int,
    occupied_in: np.ndarray,
    reserved_out: np.ndarray,
) -> dict[str, object] | None:
    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        pointer_r0 = random.randint(0, h - 2)
        pointer_c0 = random.randint(0, w - 2)
        square_r0, square_c0 = _square_top_left(pointer_r0, pointer_c0, side)
        if not (0 <= square_r0 and 0 <= square_c0 and square_r0 + side <= h and square_c0 + side <= w):
            continue
        if not _square_is_free(reserved_out, square_r0, square_c0, side):
            continue

        pointer_cells = _pointer_cells(pointer_r0, pointer_c0, direction)
        if any(occupied_in[int(r), int(c)] for r, c in pointer_cells):
            continue

        ray = _ray_cells({"r0": pointer_r0, "c0": pointer_c0, "direction": direction}, (h, w))
        if len(ray) < 2 * side - 1:
            continue
        markers = _pick_marker_cells(ray, occupied_in, side)
        if markers is None:
            continue

        for r, c in pointer_cells:
            occupied_in[int(r), int(c)] = True
        for r, c in markers:
            occupied_in[int(r), int(c)] = True
        _reserve_square(reserved_out, square_r0, square_c0, side)

        return {
            "color": int(color),
            "direction": direction,
            "pointer_r0": int(pointer_r0),
            "pointer_c0": int(pointer_c0),
            "markers": markers,
            "square_r0": int(square_r0),
            "square_c0": int(square_c0),
            "side": int(side),
        }
    return None


def _render_input(h: int, w: int, specs: list[dict[str, object]]) -> np.ndarray:
    grid = np.full((h, w), BACKGROUND, dtype=int)
    for spec in specs:
        color = int(spec["color"])
        direction = str(spec["direction"])
        pointer_r0 = int(spec["pointer_r0"])
        pointer_c0 = int(spec["pointer_c0"])
        for r, c in _pointer_cells(pointer_r0, pointer_c0, direction):
            grid[int(r), int(c)] = color
        for r, c in spec["markers"]:
            grid[int(r), int(c)] = color
    return grid


def _render_expected(h: int, w: int, specs: list[dict[str, object]]) -> np.ndarray:
    grid = np.full((h, w), BACKGROUND, dtype=int)
    for spec in specs:
        color = int(spec["color"])
        side = int(spec["side"])
        r0 = int(spec["square_r0"])
        c0 = int(spec["square_c0"])
        grid[r0 : r0 + side, c0 : c0 + side] = color
    return grid


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_ROWS_RANGE)
    w = random.randint(*GRID_COLS_RANGE)
    n_pointers = random.choice(POINTER_COUNT_CHOICES)
    directions = random.sample(list(DIRECTION_TO_MISSING.keys()), n_pointers)
    colors = random.sample(PALETTE, n_pointers)

    sides = [random.choice(SQUARE_SIZE_CHOICES) for _ in range(n_pointers)]
    order = sorted(range(n_pointers), key=lambda idx: sides[idx], reverse=True)

    occupied_in = np.zeros((h, w), dtype=bool)
    reserved_out = np.zeros((h, w), dtype=bool)
    specs: list[dict[str, object] | None] = [None] * n_pointers

    for idx in order:
        spec = _place_pointer(
            h=h,
            w=w,
            direction=directions[idx],
            color=colors[idx],
            side=sides[idx],
            occupied_in=occupied_in,
            reserved_out=reserved_out,
        )
        if spec is None:
            return None
        specs[idx] = spec

    final_specs = [spec for spec in specs if spec is not None]
    inp = _render_input(h, w, final_specs)
    expected = _render_expected(h, w, final_specs)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    """
    Generate diverse pointer-marker scenes.

    Square sizes are kept even so the center of the output square matches the geometric center
    of the 2x2 pointer bbox without discretization ambiguity.
    """
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid pointer-square sample")
