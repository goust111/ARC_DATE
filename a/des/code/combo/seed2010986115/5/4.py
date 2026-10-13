from __future__ import annotations

import random

import numpy as np


BG_COLOR = 0
CROSS_COLOR = 8
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
QUADRANT_ORDER = ("TL", "TR", "BL", "BR")

# For a source object in quadrant q:
# 1) use the corner on the same horizontal edge,
# 2) then use the corner on the same vertical edge.
#
# The puzzle description leaves the row/column-corner disambiguation implicit.
# This implementation resolves it with the two opposite edge-end corners for
# each quadrant, which makes the composition depend on the diagonal color pairs.
HSTEP_CORNER = {"TL": "tr", "TR": "tl", "BL": "br", "BR": "bl"}
VSTEP_CORNER = {"TL": "bl", "TR": "br", "BL": "tl", "BR": "tr"}

GEN_COLOR_POOL = tuple(c for c in range(10) if c not in {BG_COLOR, CROSS_COLOR})
GRID_SIDES = (13, 15, 17, 19)
HEAVY_AREA_RANGE = (4, 6)
LIGHT_AREA_RANGE = (1, 3)
EXTRA_AREA_RANGE = (2, 4)
MAX_SHAPE_SPAN = 4
GEN_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(counts.argmax())


def _solid_block_color(block: np.ndarray) -> int:
    values = np.unique(block)
    assert len(values) == 1
    return int(values[0])


def _protected_mask(height: int, width: int, center_r: int, center_c: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[center_r, :] = True
    mask[:, center_c] = True
    mask[:2, :2] = True
    mask[:2, -2:] = True
    mask[-2:, :2] = True
    mask[-2:, -2:] = True
    return mask


def _quadrant_of_cell(row: int, col: int, center_r: int, center_c: int) -> str:
    assert row != center_r and col != center_c
    vertical = "T" if row < center_r else "B"
    horizontal = "L" if col < center_c else "R"
    return vertical + horizontal


def _parse_scene(grid: np.ndarray) -> tuple[int, int, int, int, dict[str, int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = g.shape
    assert height % 2 == 1 and width % 2 == 1
    assert height >= 9 and width >= 9

    center_r = height // 2
    center_c = width // 2
    cross_color = int(g[center_r, center_c])
    assert np.all(g[center_r, :] == cross_color)
    assert np.all(g[:, center_c] == cross_color)

    corners = {
        "tl": _solid_block_color(g[:2, :2]),
        "tr": _solid_block_color(g[:2, -2:]),
        "bl": _solid_block_color(g[-2:, :2]),
        "br": _solid_block_color(g[-2:, -2:]),
    }
    assert corners["tl"] == corners["br"]
    assert corners["tr"] == corners["bl"]
    assert corners["tl"] != corners["tr"]
    assert cross_color not in set(corners.values())

    protected = _protected_mask(height, width, center_r, center_c)
    background = _most_common_color(g)
    return height, width, center_r, center_c, corners, protected


def _extract_objects(
    grid: np.ndarray,
    background: int,
    protected: np.ndarray,
    center_r: int,
    center_c: int,
) -> list[dict[str, object]]:
    height, width = grid.shape
    seen = np.zeros((height, width), dtype=bool)
    objects: list[dict[str, object]] = []

    for row in range(height):
        for col in range(width):
            if protected[row, col] or seen[row, col] or int(grid[row, col]) == background:
                continue
            color = int(grid[row, col])
            stack = [(row, col)]
            seen[row, col] = True
            coords: list[tuple[int, int]] = []

            while stack:
                cur_r, cur_c = stack.pop()
                coords.append((cur_r, cur_c))
                for dr, dc in NEIGHBORS4:
                    nxt_r = cur_r + dr
                    nxt_c = cur_c + dc
                    if not (0 <= nxt_r < height and 0 <= nxt_c < width):
                        continue
                    if seen[nxt_r, nxt_c] or protected[nxt_r, nxt_c]:
                        continue
                    if int(grid[nxt_r, nxt_c]) != color:
                        continue
                    seen[nxt_r, nxt_c] = True
                    stack.append((nxt_r, nxt_c))

            quadrant = _quadrant_of_cell(coords[0][0], coords[0][1], center_r, center_c)
            assert all(_quadrant_of_cell(r, c, center_r, center_c) == quadrant for r, c in coords)
            coords.sort()
            objects.append(
                {
                    "color": color,
                    "coords": tuple(coords),
                    "quadrant": quadrant,
                    "anchor": coords[0],
                }
            )

    objects.sort(key=lambda obj: (QUADRANT_ORDER.index(obj["quadrant"]), obj["anchor"]))  # type: ignore[index]
    return objects


def _apply_step(
    coords: tuple[tuple[int, int], ...],
    action: str,
    is_horizontal_step: bool,
    center_r: int,
    center_c: int,
) -> tuple[tuple[int, int], ...]:
    moved: list[tuple[int, int]] = []
    if action == "mirror":
        if is_horizontal_step:
            moved = [(2 * center_r - row, col) for row, col in coords]
        else:
            moved = [(row, 2 * center_c - col) for row, col in coords]
    else:
        assert action == "rotate"
        moved = [(2 * center_r - row, 2 * center_c - col) for row, col in coords]
    moved.sort()
    return tuple(moved)


def _simulate_with_mapping(
    grid: np.ndarray,
    objects: list[dict[str, object]],
    corners: dict[str, int],
    protected: np.ndarray,
    center_r: int,
    center_c: int,
    action_by_color: dict[int, str],
) -> tuple[np.ndarray, tuple[int, int, int]]:
    out = grid.copy()
    moved_pixels = 0
    moved_objects = 0

    for quadrant in QUADRANT_ORDER:
        for obj in objects:
            if obj["quadrant"] != quadrant:
                continue
            coords = obj["coords"]  # type: ignore[assignment]
            first_action = action_by_color[corners[HSTEP_CORNER[quadrant]]]
            second_action = action_by_color[corners[VSTEP_CORNER[quadrant]]]
            transformed = _apply_step(coords, first_action, True, center_r, center_c)
            transformed = _apply_step(transformed, second_action, False, center_r, center_c)

            if transformed != coords:
                moved_objects += 1
                moved_pixels += len(coords)

            color = int(obj["color"])
            for row, col in transformed:
                if 0 <= row < out.shape[0] and 0 <= col < out.shape[1] and not protected[row, col]:
                    out[row, col] = color

    delta = int(np.sum(out != grid))
    return out, (delta, moved_pixels, moved_objects)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform each quadrant's objects by the two corner-guided operations and overlay the result.

    The input itself contains only the two indicator colors, not their action labels, so
    `main()` evaluates both possible color->action assignments and keeps the one that yields
    the richer visible overlay.
    """
    g = np.asarray(grid, dtype=int)
    background = _most_common_color(g)
    height, width, center_r, center_c, corners, protected = _parse_scene(g)
    del height, width

    objects = _extract_objects(g, background, protected, center_r, center_c)
    color_a = corners["tl"]
    color_b = corners["tr"]
    candidates: list[tuple[tuple[int, int, int], np.ndarray]] = []

    for mirror_color, rotate_color in ((color_a, color_b), (color_b, color_a)):
        action_by_color = {mirror_color: "mirror", rotate_color: "rotate"}
        out, score = _simulate_with_mapping(
            g,
            objects,
            corners,
            protected,
            center_r,
            center_c,
            action_by_color,
        )
        candidates.append((score, out))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _rotate_or_flip_shape(
    cells: tuple[tuple[int, int], ...],
    height: int,
    width: int,
) -> tuple[tuple[tuple[int, int], ...], int, int]:
    mask = np.zeros((height, width), dtype=bool)
    for row, col in cells:
        mask[row, col] = True
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    turns = random.randrange(4)
    if turns:
        mask = np.rot90(mask, turns)
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    coords.sort()
    return tuple(coords), int(mask.shape[0]), int(mask.shape[1])


def _random_polyomino(area: int) -> tuple[tuple[tuple[int, int], ...], int, int]:
    assert area >= 1
    for _ in range(200):
        cells = {(0, 0)}
        while len(cells) < area:
            row, col = random.choice(tuple(cells))
            dr, dc = random.choice(NEIGHBORS4)
            cells.add((row + dr, col + dc))

        min_r = min(row for row, _col in cells)
        min_c = min(col for _row, col in cells)
        norm = {(row - min_r, col - min_c) for row, col in cells}
        height = max(row for row, _col in norm) + 1
        width = max(col for _row, col in norm) + 1
        if max(height, width) > MAX_SHAPE_SPAN:
            continue

        coords = tuple(sorted(norm))
        return _rotate_or_flip_shape(coords, height, width)
    raise RuntimeError("failed to sample a compact polyomino")


def _quadrant_box(height: int, width: int, quadrant: str) -> tuple[int, int, int, int]:
    center_r = height // 2
    center_c = width // 2
    if quadrant == "TL":
        return 0, center_r - 1, 0, center_c - 1
    if quadrant == "TR":
        return 0, center_r - 1, center_c + 1, width - 1
    if quadrant == "BL":
        return center_r + 1, height - 1, 0, center_c - 1
    assert quadrant == "BR"
    return center_r + 1, height - 1, center_c + 1, width - 1


def _place_shape(
    shape: tuple[tuple[tuple[int, int], ...], int, int],
    height: int,
    width: int,
    quadrant: str,
    occupied: set[tuple[int, int]],
) -> tuple[tuple[int, int], ...] | None:
    cells, shape_h, shape_w = shape
    row0_min, row0_max, col0_min, col0_max = _quadrant_box(height, width, quadrant)
    if row0_max - row0_min + 1 < shape_h or col0_max - col0_min + 1 < shape_w:
        return None

    placements = [
        (row0, col0)
        for row0 in range(row0_min, row0_max - shape_h + 2)
        for col0 in range(col0_min, col0_max - shape_w + 2)
    ]
    random.shuffle(placements)

    for row0, col0 in placements:
        coords = {(row0 + dr, col0 + dc) for dr, dc in cells}
        if coords & occupied:
            continue
        if any((row < 2 and col < 2) for row, col in coords):
            continue
        if any((row < 2 and col >= width - 2) for row, col in coords):
            continue
        if any((row >= height - 2 and col < 2) for row, col in coords):
            continue
        if any((row >= height - 2 and col >= width - 2) for row, col in coords):
            continue
        placed = tuple(sorted(coords))
        return placed
    return None


def _corner_layout(color_a: int, color_b: int, use_a_on_main_diagonal: bool) -> dict[str, int]:
    if use_a_on_main_diagonal:
        return {"tl": color_a, "tr": color_b, "bl": color_b, "br": color_a}
    return {"tl": color_b, "tr": color_a, "bl": color_a, "br": color_b}


def _render_input(
    height: int,
    width: int,
    corners: dict[str, int],
    objects: list[tuple[int, tuple[tuple[int, int], ...]]],
) -> np.ndarray:
    grid = np.full((height, width), BG_COLOR, dtype=int)
    center_r = height // 2
    center_c = width // 2
    grid[center_r, :] = CROSS_COLOR
    grid[:, center_c] = CROSS_COLOR
    grid[:2, :2] = corners["tl"]
    grid[:2, -2:] = corners["tr"]
    grid[-2:, :2] = corners["bl"]
    grid[-2:, -2:] = corners["br"]

    for color, coords in objects:
        for row, col in coords:
            grid[row, col] = color
    return grid


def _rotating_pair(corners: dict[str, int], mirror_color: int) -> tuple[tuple[str, str], tuple[str, str]]:
    main_diagonal_color = corners["tl"]
    if mirror_color == main_diagonal_color:
        return ("TR", "BL"), ("TL", "BR")
    return ("TL", "BR"), ("TR", "BL")


def _candidate_input() -> np.ndarray | None:
    height = random.choice(GRID_SIDES)
    width = random.choice(GRID_SIDES)
    indicator_a, indicator_b = random.sample(GEN_COLOR_POOL, 2)
    corners = _corner_layout(indicator_a, indicator_b, random.random() < 0.5)
    mirror_color = random.choice((indicator_a, indicator_b))
    rotate_color = indicator_b if mirror_color == indicator_a else indicator_a
    truth_actions = {mirror_color: "mirror", rotate_color: "rotate"}
    false_actions = {rotate_color: "mirror", mirror_color: "rotate"}

    rotating_pair, static_pair = _rotating_pair(corners, mirror_color)
    object_colors = [c for c in GEN_COLOR_POOL if c not in {indicator_a, indicator_b}]
    random.shuffle(object_colors)

    occupied: set[tuple[int, int]] = set()
    objects: list[tuple[int, tuple[tuple[int, int], ...]]] = []

    for quadrant in rotating_pair:
        shape = _random_polyomino(random.randint(*HEAVY_AREA_RANGE))
        coords = _place_shape(shape, height, width, quadrant, occupied)
        if coords is None:
            return None
        occupied.update(coords)
        objects.append((object_colors.pop(), coords))

    if random.random() < 0.35:
        extra_quadrant = random.choice(rotating_pair)
        shape = _random_polyomino(random.randint(*EXTRA_AREA_RANGE))
        coords = _place_shape(shape, height, width, extra_quadrant, occupied)
        if coords is None:
            return None
        occupied.update(coords)
        objects.append((object_colors.pop(), coords))

    for quadrant in static_pair:
        if random.random() < 0.7:
            shape = _random_polyomino(random.randint(*LIGHT_AREA_RANGE))
            coords = _place_shape(shape, height, width, quadrant, occupied)
            if coords is None:
                return None
            occupied.update(coords)
            objects.append((object_colors.pop(), coords))

    grid = _render_input(height, width, corners, objects)
    _h, _w, center_r, center_c, parsed_corners, protected = _parse_scene(grid)
    detected = _extract_objects(grid, BG_COLOR, protected, center_r, center_c)
    truth_out, truth_score = _simulate_with_mapping(
        grid,
        detected,
        parsed_corners,
        protected,
        center_r,
        center_c,
        truth_actions,
    )
    false_out, false_score = _simulate_with_mapping(
        grid,
        detected,
        parsed_corners,
        protected,
        center_r,
        center_c,
        false_actions,
    )
    if truth_score <= false_score:
        return None
    if truth_score[0] < 4:
        return None
    if np.array_equal(truth_out, false_out):
        return None
    if not np.array_equal(main(grid), truth_out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        candidate = _candidate_input()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid puzzle instance")
