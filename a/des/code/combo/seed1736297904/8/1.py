import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))

# Generator configuration.
GRID_H_RANGE = (14, 26)
GRID_W_RANGE = (14, 28)
FRAME_COUNT_RANGE = (2, 5)
PLACEMENT_TRIES = 200
GEN_ATTEMPTS = 400


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    result: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            result.append(np.array(coords, dtype=int))
    return result


def _color_components(grid: np.ndarray) -> list[dict[str, object]]:
    comps: list[dict[str, object]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == BG:
            continue
        for coords in _components(grid == color):
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                }
            )
    return comps


def _is_rect_frame(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    local = np.zeros((h, w), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return np.array_equal(local, expected)


def _is_small_solid_square(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h != w or h not in (1, 2):
        return False
    return int(coords.shape[0]) == h * w


def _frame_infos(grid: np.ndarray) -> list[dict[str, object]]:
    infos: list[dict[str, object]] = []
    for comp in _color_components(grid):
        coords = comp["coords"]
        assert isinstance(coords, np.ndarray)
        if not _is_rect_frame(coords):
            continue
        frame_color = int(comp["color"])
        r0, c0, r1, c1 = comp["bbox"]
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        mask = interior != BG
        inner_parts = _components(mask)
        assert len(inner_parts) == 1, "each frame must contain exactly one inner object"
        inner_local = inner_parts[0]
        inner_values = {
            int(interior[r, c])
            for r, c in inner_local
        }
        assert len(inner_values) == 1, "inner object must be monochrome"
        inner_color = inner_values.pop()
        assert inner_color != frame_color, "frame and inner colors must differ"
        assert _is_small_solid_square(inner_local), "inner object must be a 1x1 or 2x2 solid square"
        inner_coords = inner_local.copy()
        inner_coords[:, 0] += r0 + 1
        inner_coords[:, 1] += c0 + 1
        infos.append(
            {
                "frame_color": frame_color,
                "frame_coords": coords,
                "inner_color": inner_color,
                "inner_coords": inner_coords,
            }
        )
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap each rectangular frame's color with the color of the single small square inside it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    infos = _frame_infos(g)
    assert infos, "expected at least one framed region"

    out = g.copy()
    for info in infos:
        frame_color = int(info["frame_color"])
        inner_color = int(info["inner_color"])
        frame_coords = info["frame_coords"]
        inner_coords = info["inner_coords"]
        assert isinstance(frame_coords, np.ndarray)
        assert isinstance(inner_coords, np.ndarray)
        out[frame_coords[:, 0], frame_coords[:, 1]] = inner_color
        out[inner_coords[:, 0], inner_coords[:, 1]] = frame_color
    return out


def _draw_frame(grid: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    grid[top, left : left + width] = color
    grid[top + height - 1, left : left + width] = color
    grid[top : top + height, left] = color
    grid[top : top + height, left + width - 1] = color


def _sample_block_sizes(count: int, rng: random.Random) -> list[int]:
    if count == 1:
        return [rng.choice((1, 2))]
    if rng.random() < 0.7:
        sizes = [1, 2]
        sizes.extend(rng.choice((1, 2)) for _ in range(count - 2))
        rng.shuffle(sizes)
        return sizes
    return [rng.choice((1, 2)) for _ in range(count)]


def _random_derangement(values: list[int], rng: random.Random) -> list[int]:
    perm = values[:]
    for _ in range(100):
        rng.shuffle(perm)
        if all(a != b for a, b in zip(values, perm)):
            return perm
    raise RuntimeError("failed to sample derangement")


def _assign_inner_colors(frame_colors: list[int], rng: random.Random) -> list[int]:
    extras = [color for color in PALETTE if color not in frame_colors]
    if len(frame_colors) == 1:
        return [rng.choice(extras)]

    mode = rng.choice(("cycle", "mixed", "extra"))
    if mode == "cycle":
        return _random_derangement(frame_colors, rng)

    if mode == "extra":
        return [rng.choice(extras) for _ in frame_colors]

    perm = _random_derangement(frame_colors, rng)
    inner_colors: list[int] = []
    used_perm = False
    used_extra = False
    for i in range(len(frame_colors)):
        use_perm = rng.random() < 0.5
        if use_perm:
            inner_colors.append(perm[i])
            used_perm = True
        else:
            inner_colors.append(rng.choice(extras))
            used_extra = True
    if not used_perm:
        inner_colors[0] = perm[0]
    if not used_extra:
        inner_colors[-1] = rng.choice(extras)
    return inner_colors


def _expected_output(grid: np.ndarray, specs: list[dict[str, int]]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        top = spec["top"]
        left = spec["left"]
        height = spec["height"]
        width = spec["width"]
        frame_color = spec["frame_color"]
        inner_color = spec["inner_color"]
        block_top = spec["block_top"]
        block_left = spec["block_left"]
        block_size = spec["block_size"]

        out[top, left : left + width] = inner_color
        out[top + height - 1, left : left + width] = inner_color
        out[top : top + height, left] = inner_color
        out[top : top + height, left + width - 1] = inner_color
        out[block_top : block_top + block_size, block_left : block_left + block_size] = frame_color
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    frame_count = rng.randint(*FRAME_COUNT_RANGE)
    min_side = max(14, 4 * frame_count + 4)
    height = rng.randint(min_side, GRID_H_RANGE[1])
    width = rng.randint(min_side, GRID_W_RANGE[1])

    grid = np.full((height, width), BG, dtype=int)
    frame_colors = rng.sample(list(PALETTE), frame_count)
    block_sizes = _sample_block_sizes(frame_count, rng)

    specs: list[dict[str, int]] = []
    for frame_color, block_size in zip(frame_colors, block_sizes):
        min_h = block_size + 4
        min_w = block_size + 4
        max_h = min(10, height - 2)
        max_w = min(11, width - 2)
        if min_h > max_h or min_w > max_w:
            return None
        outer_h = rng.randint(min_h, max_h)
        outer_w = rng.randint(min_w, max_w)
        if rng.random() < 0.3:
            outer_w = outer_h if min_w <= outer_h <= max_w else outer_w
        specs.append(
            {
                "frame_color": frame_color,
                "block_size": block_size,
                "height": outer_h,
                "width": outer_w,
            }
        )

    inner_colors = _assign_inner_colors(frame_colors, rng)
    for spec, inner_color in zip(specs, inner_colors):
        spec["inner_color"] = inner_color

    specs.sort(key=lambda spec: spec["height"] * spec["width"], reverse=True)
    blocked = np.zeros_like(grid, dtype=bool)
    for spec in specs:
        height = spec["height"]
        width = spec["width"]
        placed = False
        for _ in range(PLACEMENT_TRIES):
            top = rng.randint(1, grid.shape[0] - height - 1)
            left = rng.randint(1, grid.shape[1] - width - 1)
            if blocked[top - 1 : top + height + 1, left - 1 : left + width + 1].any():
                continue
            blocked[top - 1 : top + height + 1, left - 1 : left + width + 1] = True
            spec["top"] = top
            spec["left"] = left
            placed = True
            break
        if not placed:
            return None

    for spec in specs:
        top = spec["top"]
        left = spec["left"]
        height = spec["height"]
        width = spec["width"]
        frame_color = spec["frame_color"]
        block_size = spec["block_size"]
        inner_color = spec["inner_color"]

        _draw_frame(grid, top, left, height, width, frame_color)
        block_top = rng.randint(top + 2, top + height - block_size - 2)
        block_left = rng.randint(left + 2, left + width - block_size - 2)
        grid[block_top : block_top + block_size, block_left : block_left + block_size] = inner_color
        spec["block_top"] = block_top
        spec["block_left"] = block_left

    infos = _frame_infos(grid)
    if len(infos) != frame_count:
        return None
    if any(int(info["frame_coords"].shape[0]) < 8 for info in infos):
        return None

    expected = _expected_output(grid, specs)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
