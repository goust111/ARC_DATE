from __future__ import annotations

from fractions import Fraction

import numpy as np


BG = 0
YELLOW = 4
GREY = 5
SPRITE_COLORS = (1, 2, 3, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GENERATE_RETRIES = 500
SPRITE_TRIES = 220


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _normalize(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _random_polyomino(size: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < size:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in DIR4:
                cell = (r + dr, c + dc)
                if cell not in cells:
                    frontier.append(cell)
        choice = frontier[int(rng.integers(0, len(frontier)))]
        cells.add(choice)
    return _normalize(cells)


def _corner_info(shape: tuple[int, int], coord: tuple[int, int]) -> tuple[int, tuple[int, int]]:
    h, w = shape
    corners = {
        (0, 0): (1, (0, 0)),
        (0, w - 1): (-1, (0, w - 1)),
        (h - 1, 0): (-1, (h - 1, 0)),
        (h - 1, w - 1): (1, (h - 1, w - 1)),
    }
    assert coord in corners, "yellow marker must occupy a corner"
    return corners[coord]


def _distance_key(coords: np.ndarray, source: tuple[int, int]) -> tuple[Fraction, Fraction, Fraction]:
    n = len(coords)
    sum_r = int(coords[:, 0].sum())
    sum_c = int(coords[:, 1].sum())
    sr, sc = source
    dist = Fraction(abs(sum_r - n * sr) + abs(sum_c - n * sc), n)
    centroid_r = Fraction(sum_r, n)
    centroid_c = Fraction(sum_c, n)
    return dist, centroid_r, centroid_c


def _project_coords(coords: np.ndarray, platform_row: int, sign: int) -> np.ndarray:
    dy = int(platform_row - 1 - coords[:, 0].max())
    assert dy >= 1, "sprites must start above the platform"
    offset = np.array([dy, sign * dy], dtype=int)
    projected = coords + offset
    assert int(projected[:, 0].max()) == platform_row - 1
    return projected


def _extract_sprites(grid: np.ndarray, platform_row: int) -> list[dict[str, object]]:
    sprites: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in {BG, YELLOW, GREY}):
        mask = grid == color
        comps = _components(mask)
        assert len(comps) == 1, "each sprite color must form exactly one component"
        coords = comps[0]
        assert 2 <= len(coords) <= 5, "sprite size must stay in the intended range"
        assert int(coords[:, 0].max()) < platform_row, "sprites must lie above the platform"
        sprites.append({"color": color, "coords": coords})
    assert sprites, "expected at least one sprite"
    return sprites


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h >= 8 and w >= 8
    assert int(np.count_nonzero(g == YELLOW)) == 1, "expected exactly one yellow light marker"

    yellow_r, yellow_c = map(int, np.argwhere(g == YELLOW)[0])
    sign, source = _corner_info((h, w), (yellow_r, yellow_c))

    platform_rows = [r for r in range(h) if np.all(g[r, :] == GREY)]
    assert len(platform_rows) == 1, "expected exactly one grey platform row"
    platform_row = int(platform_rows[0])

    sprites = _extract_sprites(g, platform_row)
    placements: list[tuple[tuple[Fraction, Fraction, Fraction], int, np.ndarray, np.ndarray]] = []
    out = g.copy()

    for sprite in sprites:
        color = int(sprite["color"])
        coords = np.asarray(sprite["coords"], dtype=int)
        projected = _project_coords(coords, platform_row, sign)
        assert bool(
            np.all(
                (0 <= projected[:, 0])
                & (projected[:, 0] < h)
                & (0 <= projected[:, 1])
                & (projected[:, 1] < w)
            )
        ), "projected sprite must stay in bounds"
        out[coords[:, 0], coords[:, 1]] = BG
        placements.append((_distance_key(coords, source), color, coords, projected))

    for _dist, color, _coords, projected in sorted(placements, key=lambda item: item[0]):
        out[projected[:, 0], projected[:, 1]] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each colored sprite along the 45-degree projection family implied by the
    corner light until the sprite's lowest pixel sits one row above the grey platform.
    When projected sprites overlap, the one farther from the light source is painted last.
    """
    return _solve(grid)


def _paint(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def _mark_blocked(blocked: np.ndarray, coords: np.ndarray, padding: int = 1) -> None:
    h, w = blocked.shape
    for r, c in coords:
        r0 = max(0, int(r) - padding)
        r1 = min(h - 1, int(r) + padding)
        c0 = max(0, int(c) - padding)
        c1 = min(w - 1, int(c) + padding)
        blocked[r0 : r1 + 1, c0 : c1 + 1] = True


def _coords_from_offsets(top: int, left: int, offsets: list[tuple[int, int]]) -> np.ndarray:
    return np.asarray([(top + dr, left + dc) for dr, dc in offsets], dtype=int)


def _overlap_size(a: np.ndarray, b: np.ndarray) -> int:
    return len({(int(r), int(c)) for r, c in a.tolist()} & {(int(r), int(c)) for r, c in b.tolist()})


def _paint_expected(
    shape: tuple[int, int],
    platform_row: int,
    source: tuple[int, int],
    sprites: list[dict[str, object]],
) -> np.ndarray:
    h, w = shape
    sign, _ = _corner_info(shape, source)
    grid = np.full(shape, BG, dtype=int)
    grid[platform_row, :] = GREY
    grid[source] = YELLOW
    placements: list[tuple[tuple[Fraction, Fraction, Fraction], int, np.ndarray]] = []
    for sprite in sprites:
        placements.append(
            (
                _distance_key(np.asarray(sprite["input_coords"], dtype=int), source),
                int(sprite["color"]),
                np.asarray(sprite["projected_coords"], dtype=int),
            )
        )
    for _dist, color, projected in sorted(placements, key=lambda item: item[0]):
        grid[projected[:, 0], projected[:, 1]] = color
    return grid


def _place_sprite(
    rng: np.random.Generator,
    *,
    grid_shape: tuple[int, int],
    platform_row: int,
    sign: int,
    blocked_input: np.ndarray,
    existing_projected: list[np.ndarray],
    require_overlap_with: int | None,
) -> dict[str, object] | None:
    h, w = grid_shape
    for _ in range(SPRITE_TRIES):
        size = int(rng.integers(2, 6))
        offsets = _random_polyomino(size, rng)
        shape_h = 1 + max(r for r, _ in offsets)
        shape_w = 1 + max(c for _, c in offsets)
        output_top = platform_row - shape_h
        if output_top < 2:
            continue

        if sign > 0:
            col_start = 2
            col_stop = w - shape_w + 1
        else:
            col_start = 0
            col_stop = w - shape_w - 1
        if col_start >= col_stop:
            continue

        output_left = int(rng.integers(col_start, col_stop))
        dy_max = min(
            output_top - 1,
            output_left if sign > 0 else w - shape_w - output_left,
        )
        if dy_max < 2:
            continue
        dy = int(rng.integers(2, dy_max + 1))

        input_top = output_top - dy
        input_left = output_left - sign * dy
        input_coords = _coords_from_offsets(input_top, input_left, offsets)
        projected_coords = _coords_from_offsets(output_top, output_left, offsets)
        if np.any(blocked_input[input_coords[:, 0], input_coords[:, 1]]):
            continue

        if require_overlap_with is None:
            if any(_overlap_size(projected_coords, other) for other in existing_projected):
                continue
        else:
            if require_overlap_with >= len(existing_projected):
                continue
            overlap = _overlap_size(projected_coords, existing_projected[require_overlap_with])
            if overlap not in (1, 2):
                continue
            if any(
                idx != require_overlap_with and _overlap_size(projected_coords, other)
                for idx, other in enumerate(existing_projected)
            ):
                continue

        return {
            "offsets": offsets,
            "input_coords": input_coords,
            "projected_coords": projected_coords,
            "dy": dy,
        }
    return None


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(12, 19))
    w = int(rng.integers(13, 21))
    corner = int(rng.integers(0, 4))
    corners = ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1))
    source = corners[corner]
    sign, _ = _corner_info((h, w), source)

    platform_low = max(7, h - 4)
    platform_high = h - 2
    if platform_low > platform_high:
        return None
    platform_row = int(rng.integers(platform_low, platform_high + 1))

    sprite_count = int(rng.integers(3, 6))
    colors = [int(v) for v in rng.choice(np.asarray(SPRITE_COLORS), size=sprite_count, replace=False)]
    want_overlap = bool(rng.integers(0, 2))

    blocked_input = np.zeros((h, w), dtype=bool)
    blocked_input[platform_row, :] = True
    _mark_blocked(blocked_input, np.asarray([source], dtype=int), padding=1)

    sprites: list[dict[str, object]] = []
    projected: list[np.ndarray] = []

    first = _place_sprite(
        rng,
        grid_shape=(h, w),
        platform_row=platform_row,
        sign=sign,
        blocked_input=blocked_input,
        existing_projected=projected,
        require_overlap_with=None,
    )
    if first is None:
        return None
    first["color"] = colors[0]
    sprites.append(first)
    projected.append(np.asarray(first["projected_coords"], dtype=int))
    _mark_blocked(blocked_input, np.asarray(first["input_coords"], dtype=int), padding=1)

    for idx in range(1, sprite_count):
        require_overlap = 0 if want_overlap and idx == 1 else None
        spec = _place_sprite(
            rng,
            grid_shape=(h, w),
            platform_row=platform_row,
            sign=sign,
            blocked_input=blocked_input,
            existing_projected=projected,
            require_overlap_with=require_overlap,
        )
        if spec is None:
            return None
        spec["color"] = colors[idx]
        sprites.append(spec)
        projected.append(np.asarray(spec["projected_coords"], dtype=int))
        _mark_blocked(blocked_input, np.asarray(spec["input_coords"], dtype=int), padding=1)

    distances = [_distance_key(np.asarray(sprite["input_coords"], dtype=int), source)[0] for sprite in sprites]
    if len(set(distances)) != len(distances):
        return None

    overlap_pairs = 0
    total_overlap_cells = 0
    for i in range(len(projected)):
        for j in range(i + 1, len(projected)):
            overlap = _overlap_size(projected[i], projected[j])
            if overlap:
                overlap_pairs += 1
                total_overlap_cells += overlap
    if want_overlap:
        if overlap_pairs != 1 or total_overlap_cells not in (1, 2):
            return None
    elif overlap_pairs != 0:
        return None

    inp = np.full((h, w), BG, dtype=int)
    inp[platform_row, :] = GREY
    inp[source] = YELLOW
    for sprite in sprites:
        _paint(inp, np.asarray(sprite["input_coords"], dtype=int), int(sprite["color"]))

    expected = _paint_expected((h, w), platform_row, source, sprites)
    if any(int(np.count_nonzero(expected == int(sprite["color"]))) == 0 for sprite in sprites):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid shadow-projection puzzle")
