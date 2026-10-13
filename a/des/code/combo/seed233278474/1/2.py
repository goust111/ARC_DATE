import random

import numpy as np


BG = 0
GREY = 5
INNER = 3
RING = 5
GEN_RETRIES = 300
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]

_RING_BORDER = np.ones((RING, RING), dtype=bool)
_RING_BORDER[1:-1, 1:-1] = False

_BASE_TEMPLATES = [
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1], [0, 0, 0]], dtype=bool),
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
]


def _build_template_library() -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    for base in _BASE_TEMPLATES:
        for k in range(4):
            rotated = np.rot90(base, k)
            for flipped in (False, True):
                mask = rotated[:, ::-1] if flipped else rotated
                sig = tuple(mask.astype(int).ravel())
                if sig in seen:
                    continue
                seen.add(sig)
                variants.append(mask.copy())
    return variants


TEMPLATE_LIBRARY = _build_template_library()


def _extract_palette(grid: np.ndarray) -> dict[int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h >= 4 and w >= 3

    mapping: dict[int, np.ndarray] = {}
    for c in range(1, w - 1):
        color = int(g[0, c])
        if color in (BG, GREY):
            continue
        if int(g[0, c - 1]) != BG or int(g[0, c + 1]) != BG:
            continue
        patch = g[1:4, c - 1 : c + 2]
        if patch.shape != (INNER, INNER):
            continue
        if not np.all((patch == BG) | (patch == GREY)):
            continue
        mask = patch == GREY
        if not np.any(mask):
            continue
        assert color not in mapping, "palette colors must be unique"
        mapping[color] = mask

    assert mapping, "expected at least one palette entry"
    return mapping


def _ring_marker_color(patch: np.ndarray) -> int | None:
    if patch.shape != (RING, RING):
        return None
    if not np.all(patch[_RING_BORDER] == GREY):
        return None

    inner = patch[1:-1, 1:-1]
    if np.any(inner == GREY):
        return None

    marker_coords = np.argwhere(inner != BG)
    if len(marker_coords) != 1:
        return None

    mr, mc = map(int, marker_coords[0])
    color = int(inner[mr, mc])
    if color in (BG, GREY):
        return None
    return color


def _find_rings(grid: np.ndarray) -> list[tuple[int, int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    rings: list[tuple[int, int, int]] = []

    for r in range(h - RING + 1):
        for c in range(w - RING + 1):
            color = _ring_marker_color(g[r : r + RING, c : c + RING])
            if color is not None:
                rings.append((r, c, color))

    assert rings, "expected at least one ring"
    return rings


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    palette = _extract_palette(g)
    rings = _find_rings(g)

    seen_colors: set[int] = set()
    out = g.copy()
    for r, c, color in rings:
        assert color in palette, "every ring marker must match a palette color"
        assert color not in seen_colors, "ring marker colors must be distinct"
        seen_colors.add(color)

        fill = np.zeros((INNER, INNER), dtype=int)
        fill[palette[color]] = color
        out[r + 1 : r + 4, c + 1 : c + 4] = fill

    return out


def _random_template_masks(rng: random.Random, count: int) -> list[np.ndarray]:
    indices = rng.sample(range(len(TEMPLATE_LIBRARY)), count)
    return [TEMPLATE_LIBRARY[i].copy() for i in indices]


def _boxes_conflict(a: tuple[int, int], b: tuple[int, int], pad: int = 1) -> bool:
    ar0, ac0 = a
    br0, bc0 = b
    ar1 = ar0 + RING - 1
    ac1 = ac0 + RING - 1
    br1 = br0 + RING - 1
    bc1 = bc0 + RING - 1
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _place_rings(
    rng: random.Random, height: int, width: int, top_row: int, count: int
) -> list[tuple[int, int]] | None:
    if top_row > height - RING:
        return None

    positions: list[tuple[int, int]] = []
    for _ in range(count):
        for _ in range(200):
            r = rng.randint(top_row, height - RING)
            c = rng.randint(0, width - RING)
            if any(_boxes_conflict((r, c), prev) for prev in positions):
                continue
            positions.append((r, c))
            break
        else:
            return None
    return positions


def _build_ring(color: int, marker_r: int, marker_c: int) -> np.ndarray:
    patch = np.zeros((RING, RING), dtype=int)
    patch[0, :] = GREY
    patch[-1, :] = GREY
    patch[:, 0] = GREY
    patch[:, -1] = GREY
    patch[1 + marker_r, 1 + marker_c] = color
    return patch


def _pick_marker(mask: np.ndarray, prefer_on_shape: bool, rng: random.Random) -> tuple[int, int]:
    candidates = np.argwhere(mask if prefer_on_shape else ~mask)
    if len(candidates) == 0:
        candidates = np.argwhere(~mask if prefer_on_shape else mask)
    idx = rng.randrange(len(candidates))
    return tuple(map(int, candidates[idx]))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    ring_count = rng.randint(3, 5)
    palette_count = rng.randint(ring_count, min(6, len(COLOR_POOL)))
    palette_colors = rng.sample(COLOR_POOL, palette_count)
    used_colors = rng.sample(palette_colors, ring_count)
    template_masks = _random_template_masks(rng, palette_count)

    template_lefts: list[int] = []
    cursor = rng.randint(1, 3)
    for i in range(palette_count):
        template_lefts.append(cursor)
        if i < palette_count - 1:
            cursor += INNER + rng.randint(1, 3)
    last_template_right = template_lefts[-1] + INNER - 1

    min_width = max(18, last_template_right + rng.randint(2, 4))
    width = rng.randint(min_width, min_width + 8)
    ring_top = 4 + rng.randint(1, 3)
    min_height = max(15, ring_top + RING + 2)
    height = rng.randint(min_height, min_height + 8)

    positions = _place_rings(rng, height, width, ring_top, ring_count)
    if positions is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    palette_masks: dict[int, np.ndarray] = {}

    for color, left, mask in zip(palette_colors, template_lefts, template_masks):
        center = left + 1
        grid[0, center] = color
        patch = grid[1:4, left : left + INNER]
        patch[mask] = GREY
        palette_masks[color] = mask

    branch_flags = [True, False]
    while len(branch_flags) < ring_count:
        branch_flags.append(rng.random() < 0.5)
    rng.shuffle(branch_flags)

    for (r, c), color, marker_on_shape in zip(positions, used_colors, branch_flags):
        marker_r, marker_c = _pick_marker(palette_masks[color], marker_on_shape, rng)
        grid[r : r + RING, c : c + RING] = _build_ring(color, marker_r, marker_c)

    parsed_palette = _extract_palette(grid)
    if set(parsed_palette) != set(palette_masks):
        return None

    rings = _find_rings(grid)
    if len(rings) != ring_count:
        return None
    if {color for _, _, color in rings} != set(used_colors):
        return None

    out = main(grid)
    for r, c, color in rings:
        expected = np.zeros((INNER, INNER), dtype=int)
        expected[palette_masks[color]] = color
        if not np.array_equal(out[r + 1 : r + 4, c + 1 : c + 4], expected):
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
