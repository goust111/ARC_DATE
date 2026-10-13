import random
from collections import Counter, deque

import numpy as np


_BG = 0
_BOUNDARY = 5
_PALETTE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
_DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_DIRECTION_PATTERNS = {
    "horizontal": frozenset({(1, 0), (1, 1), (1, 2)}),
    "vertical": frozenset({(0, 1), (1, 1), (2, 1)}),
    "diag_main": frozenset({(0, 0), (1, 1), (2, 2)}),
    "diag_anti": frozenset({(0, 2), (1, 1), (2, 0)}),
}
_GEN_RETRIES = 400
_SHAPE_RETRIES = 80


def _background_color(grid: np.ndarray) -> int:
    corners = [int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]
    return int(Counter(corners).most_common(1)[0][0])


def _legend_block(color: int, direction: str) -> np.ndarray:
    block = np.full((3, 3), _BG, dtype=int)
    for r, c in _DIRECTION_PATTERNS[direction]:
        block[r, c] = int(color)
    return block


def _parse_legend(grid: np.ndarray, bg: int) -> tuple[dict[int, str], np.ndarray]:
    h, w = grid.shape
    assert h >= 3 and w >= 3
    mapping: dict[int, str] = {}
    legend_mask = np.zeros_like(grid, dtype=bool)
    for c0 in range(w - 2):
        block = grid[0:3, c0 : c0 + 3]
        colors = [int(v) for v in np.unique(block) if int(v) != bg]
        if len(colors) != 1:
            continue
        color = colors[0]
        coords = frozenset((int(r), int(c)) for r, c in np.argwhere(block == color))
        direction = None
        for name, pattern in _DIRECTION_PATTERNS.items():
            if coords == pattern:
                direction = name
                break
        if direction is None:
            continue
        if color in mapping:
            assert mapping[color] == direction
        mapping[color] = direction
        legend_mask[0:3, c0 : c0 + 3] = True
    assert mapping
    return mapping, legend_mask


def _boundary_color(grid: np.ndarray, bg: int, legend: dict[int, str]) -> int:
    colors = {int(v) for v in np.unique(grid)}
    candidates = colors - {int(bg)} - set(legend)
    assert len(candidates) == 1
    return int(next(iter(candidates)))


def _components(mask: np.ndarray) -> list[tuple[list[tuple[int, int]], bool]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[tuple[list[tuple[int, int]], bool]] = []
    for r0, c0 in np.argwhere(mask):
        r0 = int(r0)
        c0 = int(c0)
        if seen[r0, c0]:
            continue
        q = deque([(r0, c0)])
        seen[r0, c0] = True
        cells: list[tuple[int, int]] = []
        touches_edge = False
        while q:
            r, c = q.popleft()
            cells.append((r, c))
            if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                touches_edge = True
            for dr, dc in _DIR4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        components.append((cells, touches_edge))
    return components


def _parse_regions(
    grid: np.ndarray,
) -> tuple[int, int, dict[int, str], np.ndarray, list[tuple[list[tuple[int, int]], int, int, int, str]]]:
    bg = _background_color(grid)
    legend, legend_mask = _parse_legend(grid, bg)
    boundary = _boundary_color(grid, bg, legend)
    passable = (grid != boundary) & ~legend_mask
    regions: list[tuple[list[tuple[int, int]], int, int, int, str]] = []
    for cells, touches_edge in _components(passable):
        if touches_edge:
            continue
        seeds = [(r, c, int(grid[r, c])) for r, c in cells if int(grid[r, c]) != bg]
        assert len(seeds) == 1
        sr, sc, color = seeds[0]
        assert color in legend
        regions.append((cells, sr, sc, color, legend[color]))
    assert regions
    return bg, boundary, legend, legend_mask, regions


def _on_stripe(r: int, c: int, sr: int, sc: int, direction: str) -> bool:
    if direction == "horizontal":
        return (r - sr) % 2 == 0
    if direction == "vertical":
        return (c - sc) % 2 == 0
    if direction == "diag_main":
        return ((r - c) - (sr - sc)) % 2 == 0
    assert direction == "diag_anti"
    return ((r + c) - (sr + sc)) % 2 == 0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey-enclosed region with stripes whose direction comes from the
    matching color in the top 3x3 legend.
    """
    g = np.asarray(grid, dtype=int)
    bg, _boundary, _legend, _legend_mask, regions = _parse_regions(g)
    out = g.copy()
    for cells, sr, sc, color, direction in regions:
        for r, c in cells:
            out[r, c] = int(color) if _on_stripe(r, c, sr, sc, direction) else int(bg)
    return out


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[rows][:, cols]


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _make_blob(rng: random.Random) -> np.ndarray | None:
    for _ in range(_SHAPE_RETRIES):
        h = rng.randint(4, 7)
        w = rng.randint(4, 8)
        min_area = max(7, (h * w) // 3)
        max_area = max(min_area, (2 * h * w) // 3)
        target = rng.randint(min_area, max_area)

        mask = np.zeros((h, w), dtype=bool)
        start = (rng.randrange(1, h - 1), rng.randrange(1, w - 1))
        mask[start] = True

        while int(mask.sum()) < target:
            candidates: list[tuple[int, int]] = []
            weights: list[int] = []
            for r in range(h):
                for c in range(w):
                    if mask[r, c]:
                        continue
                    neighbors = 0
                    for dr, dc in _DIR4:
                        nr = r + dr
                        nc = c + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                            neighbors += 1
                    if neighbors:
                        candidates.append((r, c))
                        weights.append(neighbors * neighbors)
            if not candidates:
                break
            r, c = rng.choices(candidates, weights=weights, k=1)[0]
            mask[r, c] = True

        mask = _trim_mask(mask)
        if mask.shape[0] < 3 or mask.shape[1] < 3:
            continue
        if int(mask.sum()) < 7:
            continue
        if bool(mask.all()) and rng.random() < 0.6:
            continue
        return mask
    return None


def _make_region_patch(rng: random.Random, seed_color: int) -> np.ndarray | None:
    mask = _make_blob(rng)
    if mask is None:
        return None

    h, w = mask.shape
    interior = np.zeros((h + 2, w + 2), dtype=bool)
    interior[1 : h + 1, 1 : w + 1] = mask
    boundary = _dilate4(interior) & ~interior
    patch = np.full(interior.shape, _BG, dtype=int)
    patch[boundary] = _BOUNDARY

    seed_cells = [(int(r), int(c)) for r, c in np.argwhere(interior)]
    sr, sc = rng.choice(seed_cells)
    patch[sr, sc] = int(seed_color)
    return patch


def _place_patch(grid: np.ndarray, blocked: np.ndarray, patch: np.ndarray, top_min: int, rng: random.Random) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    row_lo = top_min
    row_hi = h - ph - 1
    col_lo = 1
    col_hi = w - pw - 1
    if row_lo > row_hi or col_lo > col_hi:
        return False

    positions = [(r, c) for r in range(row_lo, row_hi + 1) for c in range(col_lo, col_hi + 1)]
    rng.shuffle(positions)
    for r0, c0 in positions:
        if blocked[r0 : r0 + ph, c0 : c0 + pw].any():
            continue
        view = grid[r0 : r0 + ph, c0 : c0 + pw]
        if np.any(view[patch != _BG] != _BG):
            continue
        view[patch != _BG] = patch[patch != _BG]
        rr0 = max(0, r0 - 1)
        rr1 = min(h, r0 + ph + 1)
        cc0 = max(0, c0 - 1)
        cc1 = min(w, c0 + pw + 1)
        blocked[rr0:rr1, cc0:cc1] = True
        return True
    return False


def _validate_grid(grid: np.ndarray, expected_regions: int) -> None:
    bg, boundary, legend, legend_mask, regions = _parse_regions(grid)
    assert bg == _BG
    assert boundary == _BOUNDARY
    assert len(legend) == 4
    assert set(legend.values()) == set(_DIRECTION_PATTERNS)
    assert len(regions) == expected_regions
    assert set(direction for _cells, _sr, _sc, _color, direction in regions) == set(_DIRECTION_PATTERNS)

    out = main(grid)
    assert out.shape == grid.shape
    assert not np.array_equal(out, grid)
    assert np.array_equal(out[legend_mask], grid[legend_mask])
    assert np.array_equal(out[grid == boundary], grid[grid == boundary])


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    extra_regions = rng.randint(0, 2)
    total_regions = 4 + extra_regions
    height = rng.randint(20 + extra_regions, 26 + extra_regions)
    width = rng.randint(22, 30)
    grid = np.full((height, width), _BG, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    colors = rng.sample(_PALETTE_COLORS, 4)
    directions = list(_DIRECTION_PATTERNS)
    rng.shuffle(directions)
    color_to_direction = dict(zip(colors, directions))

    legend_order = colors[:]
    rng.shuffle(legend_order)
    gap = rng.choice((1, 2))
    legend_width = 3 * len(legend_order) + gap * (len(legend_order) - 1)
    if width < legend_width + 2:
        return None
    left = rng.randint(1, width - legend_width - 1)
    for idx, color in enumerate(legend_order):
        c0 = left + idx * (3 + gap)
        block = _legend_block(color, color_to_direction[color])
        block_mask = block != _BG
        grid[0:3, c0 : c0 + 3][block_mask] = block[block_mask]
    blocked[:5, :] = True

    region_colors = colors[:] + [rng.choice(colors) for _ in range(extra_regions)]
    rng.shuffle(region_colors)
    patches = []
    for color in region_colors:
        patch = _make_region_patch(rng, color)
        if patch is None:
            return None
        patches.append(patch)
    patches.sort(key=lambda patch: patch.shape[0] * patch.shape[1], reverse=True)

    for patch in patches:
        if not _place_patch(grid, blocked, patch, top_min=5, rng=rng):
            return None

    _validate_grid(grid, total_regions)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stripe-fill puzzle input")
