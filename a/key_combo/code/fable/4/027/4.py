from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
COUNT_OPTIONS = (3, 4, 5, 6)
GRID_SIZE_RANGE = (20, 30)
HOLE_COUNT_OPTIONS = (2, 3)
GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 300

SHAPE_LIBRARY = {
    1: [
        np.array([[1]], dtype=int),
    ],
    2: [
        np.array([[1, 1]], dtype=int),
        np.array([[1], [1]], dtype=int),
    ],
    3: [
        np.array([[1, 1, 1]], dtype=int),
        np.array([[1], [1], [1]], dtype=int),
        np.array([[1, 0], [1, 1]], dtype=int),
        np.array([[0, 1], [1, 1]], dtype=int),
    ],
    4: [
        np.array([[1, 1, 1, 1]], dtype=int),
        np.array([[1], [1], [1], [1]], dtype=int),
        np.array([[1, 1], [1, 1]], dtype=int),
        np.array([[1, 0], [1, 0], [1, 1]], dtype=int),
        np.array([[0, 1], [0, 1], [1, 1]], dtype=int),
    ],
    5: [
        np.array([[1, 1, 1, 1, 1]], dtype=int),
        np.array([[1], [1], [1], [1], [1]], dtype=int),
        np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
        np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=int),
    ],
    6: [
        np.array([[1, 1, 1, 1, 1, 1]], dtype=int),
        np.array([[1], [1], [1], [1], [1], [1]], dtype=int),
        np.array([[1, 1, 1], [1, 1, 1]], dtype=int),
        np.array([[1, 1], [1, 1], [1, 1]], dtype=int),
        np.array([[1, 0], [1, 0], [1, 0], [1, 1]], dtype=int),
    ],
}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        q = deque([(r, c)])
        comp: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            comp.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        out.append(comp)
    return out


def _touches_boundary(comp: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)


def _color_components(grid: np.ndarray, bg: int):
    comps_by_color: dict[int, list[dict]] = {}
    coord_to_comp: dict[int, dict[tuple[int, int], int]] = {}
    for color in sorted(int(x) for x in np.unique(grid) if int(x) != bg):
        comps = []
        mapping: dict[tuple[int, int], int] = {}
        for idx, comp in enumerate(_components(grid == color)):
            for cell in comp:
                mapping[cell] = idx
            comps.append({"color": color, "coords": comp, "area": len(comp)})
        comps_by_color[color] = comps
        coord_to_comp[color] = mapping
    return comps_by_color, coord_to_comp


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each enclosed hole via a border-color count followed by a unique-area scatter lookup.

    Input structure:
    - The most common color is the background.
    - Several hollow monochrome shapes enclose background-colored holes, and their border colors are distinct.
    - Elsewhere the canvas contains separated 4-connected solid monochrome scatter objects, including extra scatter objects in each border color.

    Rule:
    - For each enclosed background hole, read its unique adjacent border color C.
    - Count the 4-connected C-colored components that are not the enclosing border component; call that count k.
    - Among all solid scatter objects excluding the hollow borders, find the unique object whose area is exactly k, and fill the entire hole with that object's color.
    - Leave every border and scatter object unchanged.

    Key insight:
    - The border color is only the first key: it maps to a component count, and that count then maps to a unique scatter-object area/color pair.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    holes = [comp for comp in _components(g == bg) if not _touches_boundary(comp, g.shape)]
    assert holes, "expected at least one enclosed background hole"

    comps_by_color, coord_to_comp = _color_components(g, bg)
    border_keys: set[tuple[int, int]] = set()
    hole_specs: list[tuple[list[tuple[int, int]], int]] = []

    for hole in holes:
        adj_colors: set[int] = set()
        border_ids: set[int] = set()
        for r, c in hole:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and g[nr, nc] != bg:
                    color = int(g[nr, nc])
                    adj_colors.add(color)
                    border_ids.add(coord_to_comp[color][(nr, nc)])

        assert len(adj_colors) == 1, "each hole must have one border color"
        border_color = next(iter(adj_colors))
        assert len(border_ids) == 1, "each hole must touch one enclosing border component"
        border_idx = next(iter(border_ids))
        border_keys.add((border_color, border_idx))
        hole_specs.append((hole, len(comps_by_color[border_color]) - 1))

    scatter_components = []
    for color, comps in comps_by_color.items():
        for idx, comp in enumerate(comps):
            if (color, idx) not in border_keys:
                scatter_components.append(comp)

    out = g.copy()
    for hole, lookup_area in hole_specs:
        matches = [comp for comp in scatter_components if int(comp["area"]) == int(lookup_area)]
        assert len(matches) == 1, f"expected one scatter object with area {lookup_area}"
        fill_color = int(matches[0]["color"])
        for r, c in hole:
            out[r, c] = fill_color
    return out


def _frame_patch(hole_h: int, hole_w: int, color: int) -> np.ndarray:
    patch = np.zeros((hole_h + 2, hole_w + 2), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    return patch


def _colorize(mask: np.ndarray, color: int) -> np.ndarray:
    return np.where(mask != 0, color, 0).astype(int)


def _random_shape(area: int, color: int, rng: random.Random) -> np.ndarray:
    return _colorize(rng.choice(SHAPE_LIBRARY[area]), color)


def _can_place_box(
    grid: np.ndarray, reserved: np.ndarray, top: int, left: int, h: int, w: int, padding: int
) -> bool:
    rows, cols = grid.shape
    if top < 0 or left < 0 or top + h > rows or left + w > cols:
        return False
    if reserved[top : top + h, left : left + w].any():
        return False
    r0 = max(0, top - padding)
    r1 = min(rows, top + h + padding)
    c0 = max(0, left - padding)
    c1 = min(cols, left + w + padding)
    return not bool((grid[r0:r1, c0:c1] != BG).any())


def _place_frame(
    grid: np.ndarray, reserved: np.ndarray, patch: np.ndarray, rng: random.Random
) -> tuple[int, int] | None:
    h, w = patch.shape
    for _ in range(PLACE_ATTEMPTS):
        top = rng.randrange(0, grid.shape[0] - h + 1)
        left = rng.randrange(0, grid.shape[1] - w + 1)
        if not _can_place_box(grid, reserved, top, left, h, w, padding=1):
            continue
        mask = patch != 0
        grid[top : top + h, left : left + w][mask] = patch[mask]
        reserved[top + 1 : top + h - 1, left + 1 : left + w - 1] = True
        return top, left
    return None


def _can_place_patch(
    grid: np.ndarray, reserved: np.ndarray, patch: np.ndarray, top: int, left: int
) -> bool:
    rows, cols = grid.shape
    h, w = patch.shape
    if top < 0 or left < 0 or top + h > rows or left + w > cols:
        return False
    mask = patch != 0
    coords = np.argwhere(mask)
    for pr, pc in coords:
        r = top + int(pr)
        c = left + int(pc)
        if reserved[r, c] or grid[r, c] != BG:
            return False
        for nr in range(max(0, r - 1), min(rows, r + 2)):
            for nc in range(max(0, c - 1), min(cols, c + 2)):
                if grid[nr, nc] != BG:
                    return False
    return True


def _place_patch(
    grid: np.ndarray, reserved: np.ndarray, patch: np.ndarray, rng: random.Random
) -> tuple[int, int] | None:
    h, w = patch.shape
    for _ in range(PLACE_ATTEMPTS):
        top = rng.randrange(0, grid.shape[0] - h + 1)
        left = rng.randrange(0, grid.shape[1] - w + 1)
        if not _can_place_patch(grid, reserved, patch, top, left):
            continue
        mask = patch != 0
        grid[top : top + h, left : left + w][mask] = patch[mask]
        return top, left
    return None


def _extra_colors(used: set[int], rng: random.Random, count: int) -> list[int]:
    available = [color for color in COLORS if color not in used]
    rng.shuffle(available)
    return available[:count]


def _build_case(rng: random.Random):
    rows = rng.randint(*GRID_SIZE_RANGE)
    cols = rng.randint(*GRID_SIZE_RANGE)
    grid = np.full((rows, cols), BG, dtype=int)
    reserved = np.zeros_like(grid, dtype=bool)

    hole_count = rng.choice(HOLE_COUNT_OPTIONS)
    counts = rng.sample(COUNT_OPTIONS, hole_count)
    palette = rng.sample(COLORS, 2 * hole_count)
    border_colors = palette[:hole_count]
    fill_colors = palette[hole_count:]

    frame_specs = []
    for border_color, fill_color, lookup_area in zip(border_colors, fill_colors, counts, strict=False):
        hole_h = rng.randint(2, 5)
        hole_w = rng.randint(2, 6)
        patch = _frame_patch(hole_h, hole_w, border_color)
        frame_specs.append(
            {
                "border_color": border_color,
                "fill_color": fill_color,
                "lookup_area": lookup_area,
                "hole_h": hole_h,
                "hole_w": hole_w,
                "patch": patch,
            }
        )

    rng.shuffle(frame_specs)
    placed_frames = []
    for spec in frame_specs:
        pos = _place_frame(grid, reserved, spec["patch"], rng)
        if pos is None:
            return None
        top, left = pos
        spec = dict(spec)
        spec["top"] = top
        spec["left"] = left
        placed_frames.append(spec)

    for spec in placed_frames:
        patch = _random_shape(spec["lookup_area"], spec["fill_color"], rng)
        if _place_patch(grid, reserved, patch, rng) is None:
            return None

    for spec in placed_frames:
        for _ in range(spec["lookup_area"]):
            patch = _random_shape(rng.choice((1, 2)), spec["border_color"], rng)
            if _place_patch(grid, reserved, patch, rng) is None:
                return None

    used_colors = set(border_colors) | set(fill_colors)
    decoy_colors = _extra_colors(used_colors, rng, rng.randint(1, max(1, 9 - len(used_colors))))
    rng.shuffle(decoy_colors)
    for color in decoy_colors[: rng.randint(1, len(decoy_colors))]:
        repeats = rng.randint(1, 2)
        for _ in range(repeats):
            patch = _random_shape(rng.choice((1, 2)), color, rng)
            if _place_patch(grid, reserved, patch, rng) is None:
                return None

    if _get_bg(grid) != BG:
        return None

    expected = grid.copy()
    for spec in placed_frames:
        r0 = spec["top"] + 1
        c0 = spec["left"] + 1
        expected[r0 : r0 + spec["hole_h"], c0 : c0 + spec["hole_w"]] = spec["fill_color"]

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _build_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
