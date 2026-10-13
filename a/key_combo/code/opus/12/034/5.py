from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
BORDER_COLORS = (1, 3, 4, 7)
NON_BORDER_RESULT_COLORS = (2, 5, 6, 8, 9)
PALETTE_ROWS = 2
PALETTE_LEN = 4
SQUARE_SIZES = (5, 7)
GRID_MIN = 18
GRID_MAX = 25
MAX_GENERATION_ATTEMPTS = 500


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _draw_hollow_square(grid: np.ndarray, r0: int, c0: int, size: int, color: int) -> None:
    r1 = r0 + size - 1
    c1 = c0 + size - 1
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _find_palette(grid: np.ndarray) -> tuple[dict[int, int], int]:
    assert grid.shape[0] >= 2
    cols = np.where((grid[0] != BG) | (grid[1] != BG))[0]
    assert len(cols) == PALETTE_LEN, "expected a 2x4 palette on the top edge"
    assert np.all(cols == np.arange(cols[0], cols[0] + PALETTE_LEN)), "palette must be contiguous"
    c0 = int(cols[0])
    assert np.all(grid[0, c0 : c0 + PALETTE_LEN] != BG)
    assert np.all(grid[1, c0 : c0 + PALETTE_LEN] != BG)

    mapping: dict[int, int] = {}
    for dc in range(PALETTE_LEN):
        border = int(grid[0, c0 + dc])
        result = int(grid[1, c0 + dc])
        assert border not in mapping, "palette border keys must be unique"
        mapping[border] = result
    assert len(mapping) == PALETTE_LEN
    return mapping, c0


def _validate_hollow_square(body: np.ndarray, color: int, coords: np.ndarray) -> tuple[int, int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    size = r1 - r0 + 1
    assert size == c1 - c0 + 1, "object must have a square bbox"
    assert size in SQUARE_SIZES and size % 2 == 1, "square size must be odd and supported"
    expected = np.zeros((size, size), dtype=int)
    _draw_hollow_square(expected, 0, 0, size, color)
    actual = body[r0 : r1 + 1, c0 : c1 + 1]
    assert np.array_equal(actual, expected), "object must be a hollow square border"
    return r0, c0, size


def _expected_output(
    shape: tuple[int, int],
    objects: list[dict[str, int]],
    mapping: dict[int, int],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for obj in objects:
        color = int(obj["color"])
        r0 = int(obj["r0"])
        c0 = int(obj["c0"])
        size = int(obj["size"])
        _draw_hollow_square(out, r0, c0, size, color)
        result = int(mapping[color])
        if result == color:
            center = size // 2
            out[r0 + center, c0 + center] = result
        else:
            out[r0 + 1 : r0 + size - 1, c0 + 1 : c0 + size - 1] = result
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the top-edge palette so each hollow square either gets fully filled or only gets its center marked.

    Input structure:
    - The grid has black background, four separated hollow odd-sized square sprites, and one contiguous 2x4 palette strip on the top edge.
    - The palette's first row lists border colors and the second row lists the corresponding result colors.
    - Each sprite is a single-color hollow square whose border color appears exactly once in the palette.

    Rule:
    - Read the palette as a border-color to result-color mapping.
    - For each hollow square, look up its border color's result color.
    - If the result color differs from the border color, fill the whole interior with that result color; if it matches, color only the unique center cell.
    - Erase the palette strip to black while keeping every square border unchanged.

    Key insight:
    - The palette encodes both a color mapping and an operation choice: equality means "mark center", inequality means "fill interior".
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    mapping, palette_c0 = _find_palette(g)
    body = g.copy()
    body[0:PALETTE_ROWS, palette_c0 : palette_c0 + PALETTE_LEN] = BG

    out = body.copy()
    components: list[tuple[int, np.ndarray]] = []
    for color in sorted(int(v) for v in np.unique(body) if int(v) != BG):
        for coords in _components_mask(body == color):
            components.append((color, coords))

    assert len(components) == 4, "expected exactly four hollow squares"
    seen_colors = set()
    for color, coords in components:
        assert color in mapping, "every sprite border color must appear in the palette"
        assert color not in seen_colors, "each sprite border color must be unique"
        seen_colors.add(color)

        r0, c0, size = _validate_hollow_square(body, color, coords)
        result = int(mapping[color])
        if result == color:
            center = size // 2
            assert out[r0 + center, c0 + center] == BG
            out[r0 + center, c0 + center] = result
        else:
            out[r0 + 1 : r0 + size - 1, c0 + 1 : c0 + size - 1] = result

    assert seen_colors == set(mapping), "palette keys and sprite border colors must match exactly"
    return out


def _candidate_positions(blocked: np.ndarray, size: int) -> list[tuple[int, int]]:
    h, w = blocked.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(3, h - size + 1):
        for c0 in range(0, w - size + 1):
            if not blocked[r0 : r0 + size, c0 : c0 + size].any():
                positions.append((r0, c0))
    return positions


def _place_objects(
    rng: np.random.Generator,
    h: int,
    w: int,
    sizes_by_color: dict[int, int],
) -> list[dict[str, int]] | None:
    blocked = np.zeros((h, w), dtype=bool)
    blocked[:3, :] = True
    order = sorted(
        BORDER_COLORS,
        key=lambda color: (-sizes_by_color[color], float(rng.random())),
    )
    placed: dict[int, dict[str, int]] = {}

    for color in order:
        size = sizes_by_color[color]
        positions = _candidate_positions(blocked, size)
        if not positions:
            return None
        r0, c0 = positions[int(rng.integers(len(positions)))]
        placed[color] = {"color": color, "r0": r0, "c0": c0, "size": size}
        blocked[max(0, r0 - 1) : min(h, r0 + size + 1), max(0, c0 - 1) : min(w, c0 + size + 1)] = True

    return [placed[color] for color in BORDER_COLORS]


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_MIN, GRID_MAX))
    w = int(rng.integers(GRID_MIN, GRID_MAX))
    palette_c0 = int(rng.integers(0, w - PALETTE_LEN + 1))

    sizes = [int(rng.choice(SQUARE_SIZES)) for _ in BORDER_COLORS]
    if len(set(sizes)) == 1 and rng.random() < 0.7:
        sizes[int(rng.integers(0, len(sizes)))] = 12 - sizes[0]
    sizes_by_color = {color: size for color, size in zip(BORDER_COLORS, sizes, strict=False)}

    objects = _place_objects(rng, h, w, sizes_by_color)
    if objects is None:
        return None

    same_colors = set(int(v) for v in rng.choice(BORDER_COLORS, size=2, replace=False))
    diff_sources = [color for color in BORDER_COLORS if color not in same_colors]
    diff_results = list(int(v) for v in rng.choice(NON_BORDER_RESULT_COLORS, size=2, replace=False))
    rng.shuffle(diff_results)

    mapping: dict[int, int] = {}
    for color in BORDER_COLORS:
        mapping[color] = color if color in same_colors else diff_results[diff_sources.index(color)]

    palette_order = list(int(v) for v in rng.permutation(BORDER_COLORS))

    inp = np.zeros((h, w), dtype=int)
    inp[0, palette_c0 : palette_c0 + PALETTE_LEN] = np.array(palette_order, dtype=int)
    inp[1, palette_c0 : palette_c0 + PALETTE_LEN] = np.array([mapping[color] for color in palette_order], dtype=int)
    for obj in objects:
        _draw_hollow_square(inp, int(obj["r0"]), int(obj["c0"]), int(obj["size"]), int(obj["color"]))

    expected = _expected_output((h, w), objects, mapping)

    if len(same_colors) != 2:
        return None
    if sum(mapping[color] != color for color in BORDER_COLORS) != 2:
        return None
    if not any(obj["size"] == 5 for obj in objects) and not any(obj["size"] == 7 for obj in objects):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid palette-double-encoding sample")
