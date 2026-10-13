from __future__ import annotations

import numpy as np


BLACK = 0
GREY = 5
AZURE = 8
MAROON = 9
SEED_COLORS = (1, 2, 3, 4, 6, 7)

SQUARE = 0
TALL = 1
WIDE = 2

_RNG = np.random.default_rng()
_MAX_GEN_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.asarray(coords, dtype=int))
    return out


def _classify_shape(height: int, width: int) -> int:
    if height == width:
        return SQUARE
    return TALL if height > width else WIDE


def _tile_pattern(pattern: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    ph, pw = pattern.shape
    h, w = shape
    rows = np.arange(h)[:, None] % ph
    cols = np.arange(w)[None, :] % pw
    return pattern[rows, cols]


def _extract_dictionary(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[np.ndarray]]:
    azure_mask = grid == AZURE
    box = _bbox(azure_mask)
    assert box is not None, "missing azure texture dictionary"
    r0, c0, r1, c1 = box
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    h, w = patch.shape
    assert h >= 3 and w >= 7
    assert np.all(patch[0, :] == AZURE)
    assert np.all(patch[-1, :] == AZURE)

    divider_cols = [col for col in range(w) if np.all(patch[:, col] == AZURE)]
    assert len(divider_cols) == 4, "dictionary must have outer borders and two dividers"
    assert divider_cols[0] == 0 and divider_cols[-1] == w - 1
    cell_widths = [divider_cols[i + 1] - divider_cols[i] - 1 for i in range(3)]
    assert cell_widths[0] > 0 and cell_widths[0] == cell_widths[1] == cell_widths[2]

    textures: list[np.ndarray] = []
    for idx in range(3):
        left = divider_cols[idx] + 1
        right = divider_cols[idx + 1]
        inner = patch[1:-1, left:right]
        texture = inner == MAROON
        assert inner.size > 0 and np.any(texture)
        textures.append(texture)
    return box, textures


def _find_regions(grid: np.ndarray) -> list[tuple[int, int, int, int, int, int]]:
    regions: list[tuple[int, int, int, int, int, int]] = []
    for coords in _components(grid == GREY):
        (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
        r0, c0, r1, c1 = int(r0), int(c0), int(r1), int(c1)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        h, w = patch.shape
        assert h >= 3 and w >= 3, "grey border must enclose a non-empty interior"

        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        assert np.array_equal(patch == GREY, border), "grey component must be a 1-cell-thick rectangle"

        interior = patch[1:-1, 1:-1]
        seed_cells = np.argwhere(interior != BLACK)
        assert seed_cells.shape[0] == 1, "each region interior must contain exactly one seed cell"
        sr, sc = map(int, seed_cells[0])
        seed_color = int(interior[sr, sc])
        assert seed_color != GREY
        kind = _classify_shape(int(interior.shape[0]), int(interior.shape[1]))
        regions.append((r0 + 1, c0 + 1, r1 - 1, c1 - 1, seed_color, kind))

    assert regions, "missing grey bordered regions"
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    dict_box, textures = _extract_dictionary(g)
    out = g.copy()

    for ir0, ic0, ir1, ic1, seed_color, kind in _find_regions(g):
        interior = out[ir0 : ir1 + 1, ic0 : ic1 + 1]
        interior[:, :] = BLACK
        tiled = _tile_pattern(textures[kind], interior.shape)
        interior[tiled] = seed_color

    r0, c0, r1, c1 = dict_box
    out[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    return out


def _random_pattern(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    area = height * width
    lo = max(1, area // 4)
    hi = max(lo, area - max(1, area // 3))
    count = int(rng.integers(lo, hi + 1))
    chosen = rng.choice(area, size=count, replace=False)
    mask = np.zeros((height, width), dtype=bool)
    mask.reshape(-1)[chosen] = True
    return mask


def _sample_textures(rng: np.random.Generator) -> list[np.ndarray]:
    height = int(rng.integers(2, 5))
    width = int(rng.integers(2, 6))
    textures: list[np.ndarray] = []
    seen: set[bytes] = set()
    while len(textures) < 3:
        texture = _random_pattern(height, width, rng)
        signature = texture.tobytes()
        if signature in seen:
            continue
        seen.add(signature)
        textures.append(texture)
    return textures


def _choose_inner_dims(kind: int, rng: np.random.Generator) -> tuple[int, int]:
    if kind == SQUARE:
        side = int(rng.integers(2, 7))
        return side, side
    if kind == TALL:
        width = int(rng.integers(2, 5))
        height = int(rng.integers(width + 1, width + 5))
        return height, width
    height = int(rng.integers(2, 5))
    width = int(rng.integers(height + 1, height + 5))
    return height, width


def _mark_with_margin(mask: np.ndarray, box: tuple[int, int, int, int], margin: int = 1) -> None:
    h, w = mask.shape
    r0, c0, r1, c1 = box
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(h - 1, r1 + margin)
    cc1 = min(w - 1, c1 + margin)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _can_place(mask: np.ndarray, box: tuple[int, int, int, int], margin: int = 1) -> bool:
    h, w = mask.shape
    r0, c0, r1, c1 = box
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return False
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(h - 1, r1 + margin)
    cc1 = min(w - 1, c1 + margin)
    return not bool(mask[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _draw_dictionary(
    grid: np.ndarray,
    top: int,
    left: int,
    textures: list[np.ndarray],
) -> tuple[int, int, int, int]:
    height, width = textures[0].shape
    dict_h = height + 2
    dict_w = 3 * width + 4
    bottom = top + dict_h - 1
    right = left + dict_w - 1

    grid[top, left : right + 1] = AZURE
    grid[bottom, left : right + 1] = AZURE
    for col in (left, left + width + 1, left + 2 * width + 2, right):
        grid[top : bottom + 1, col] = AZURE

    for idx, texture in enumerate(textures):
        cell_left = left + 1 + idx * (width + 1)
        cell = grid[top + 1 : bottom, cell_left : cell_left + width]
        cell[texture] = MAROON

    return top, left, bottom, right


def _draw_region(
    grid: np.ndarray,
    top: int,
    left: int,
    inner_h: int,
    inner_w: int,
    seed_color: int,
    seed_pos: tuple[int, int],
) -> tuple[int, int, int, int]:
    bottom = top + inner_h + 1
    right = left + inner_w + 1
    grid[top, left : right + 1] = GREY
    grid[bottom, left : right + 1] = GREY
    grid[top : bottom + 1, left] = GREY
    grid[top : bottom + 1, right] = GREY
    sr, sc = seed_pos
    grid[top + 1 + sr, left + 1 + sc] = seed_color
    return top, left, bottom, right


def _oracle_output(
    grid: np.ndarray,
    textures: list[np.ndarray],
    dict_box: tuple[int, int, int, int],
    regions: list[dict[str, object]],
) -> np.ndarray:
    out = grid.copy()
    for region in regions:
        ir0, ic0, ir1, ic1 = region["inner_box"]  # type: ignore[index]
        seed_color = int(region["seed_color"])  # type: ignore[index]
        kind = int(region["kind"])  # type: ignore[index]
        interior = out[ir0 : ir1 + 1, ic0 : ic1 + 1]
        interior[:, :] = BLACK
        tiled = _tile_pattern(textures[kind], interior.shape)
        interior[tiled] = seed_color
    r0, c0, r1, c1 = dict_box
    out[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(20, 31))
    width = int(rng.integers(22, 35))
    grid = np.full((height, width), BLACK, dtype=int)

    textures = _sample_textures(rng)
    texture_h, texture_w = textures[0].shape
    dict_h = texture_h + 2
    dict_w = 3 * texture_w + 4
    if dict_w + 2 > width or dict_h + 5 > height:
        return None

    bottom_margin = int(rng.integers(1, 3))
    dict_top = height - dict_h - bottom_margin
    if dict_top < 6:
        return None
    dict_left = int(rng.integers(1, width - dict_w))

    occupied = np.zeros_like(grid, dtype=bool)
    dict_box = (dict_top, dict_left, dict_top + dict_h - 1, dict_left + dict_w - 1)
    _mark_with_margin(occupied, dict_box, margin=1)
    _draw_dictionary(grid, dict_top, dict_left, textures)

    region_count = int(rng.integers(3, 7))
    kinds = [SQUARE, TALL, WIDE]
    while len(kinds) < region_count:
        kinds.append(int(rng.integers(0, 3)))
    rng.shuffle(kinds)

    seed_colors = list(rng.permutation(np.array(SEED_COLORS, dtype=int))[:region_count])
    regions: list[dict[str, object]] = []
    top_band_limit = dict_top - 2

    for idx, kind in enumerate(kinds):
        inner_h, inner_w = _choose_inner_dims(kind, rng)
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        if outer_h > top_band_limit or outer_w > width - 2:
            return None

        placed = False
        for _ in range(120):
            top = int(rng.integers(1, top_band_limit - outer_h + 2))
            left = int(rng.integers(1, width - outer_w))
            box = (top, left, top + outer_h - 1, left + outer_w - 1)
            if not _can_place(occupied, box, margin=1):
                continue
            seed_r = int(rng.integers(0, inner_h))
            seed_c = int(rng.integers(0, inner_w))
            seed_color = int(seed_colors[idx])
            _draw_region(grid, top, left, inner_h, inner_w, seed_color, (seed_r, seed_c))
            _mark_with_margin(occupied, box, margin=1)
            regions.append(
                {
                    "kind": kind,
                    "seed_color": seed_color,
                    "inner_box": (top + 1, left + 1, top + inner_h, left + inner_w),
                }
            )
            placed = True
            break
        if not placed:
            return None

    expected = _oracle_output(grid, textures, dict_box, regions)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
