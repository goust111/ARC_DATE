from __future__ import annotations

import itertools
import random

import numpy as np


BACKGROUND = 0
GREY = 5
ORANGE = 7
TEMPLATE_SIZE = 4
SOURCE_COUNT_OPTIONS = (3, 4)
SOURCE_COLOR_POOL = (1, 2, 3, 4, 6, 8, 9)
GRID_SIDE_RANGE = (18, 26)
GENERATION_ATTEMPTS = 800
LAYOUT_ATTEMPTS = 300
TEMPLATE_ATTEMPTS = 500

LEFT = "left"
RIGHT = "right"
UP = "up"
DOWN = "down"
DIRECTIONS = (LEFT, RIGHT, UP, DOWN)

DIRECTION_DELTAS = {
    LEFT: (0, -1),
    RIGHT: (0, 1),
    UP: (-1, 0),
    DOWN: (1, 0),
}

DIRECTION_TO_TRANSFORM = {
    LEFT: "hmirror",
    RIGHT: "identity",
    UP: "ccw",
    DOWN: "cw",
}


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _rectangles_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _cell_in_box(cell: tuple[int, int], box: tuple[int, int, int, int]) -> bool:
    r, c = cell
    return box[0] <= r <= box[2] and box[1] <= c <= box[3]


def _transform_mask(mask: np.ndarray, mode: str) -> np.ndarray:
    if mode == "identity":
        return mask.copy()
    if mode == "hmirror":
        return np.fliplr(mask)
    if mode == "ccw":
        return np.rot90(mask, 1)
    if mode == "cw":
        return np.rot90(mask, -1)
    raise ValueError(f"unknown transform mode: {mode}")


def _template_ok(mask: np.ndarray) -> bool:
    if mask.shape != (TEMPLATE_SIZE, TEMPLATE_SIZE):
        return False
    count = int(mask.sum())
    if not 6 <= count <= 10:
        return False
    if len(_components(mask)) != 1:
        return False
    if np.array_equal(mask, np.fliplr(mask)):
        return False
    if np.array_equal(mask, np.flipud(mask)):
        return False
    if np.array_equal(mask, np.rot90(mask, 1)) or np.array_equal(mask, np.rot90(mask, -1)):
        return False

    variants = [_transform_mask(mask, DIRECTION_TO_TRANSFORM[d]) for d in DIRECTIONS]
    if len({variant.tobytes() for variant in variants}) != 4:
        return False

    anchor_hits = [bool(variant[0, 0]) for variant in variants]
    if len(set(anchor_hits)) != 2:
        return False
    return True


def _grow_connected_template(cell_count: int) -> np.ndarray:
    mask = np.zeros((TEMPLATE_SIZE, TEMPLATE_SIZE), dtype=bool)
    start = (random.randrange(TEMPLATE_SIZE), random.randrange(TEMPLATE_SIZE))
    mask[start] = True
    cells = {start}
    while len(cells) < cell_count:
        frontier = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < TEMPLATE_SIZE and 0 <= nc < TEMPLATE_SIZE and not mask[nr, nc]:
                    frontier.add((nr, nc))
        nr, nc = random.choice(tuple(frontier))
        mask[nr, nc] = True
        cells.add((nr, nc))
    return mask


def _make_template() -> np.ndarray:
    for _ in range(TEMPLATE_ATTEMPTS):
        mask = _grow_connected_template(random.randint(6, 10))
        if _template_ok(mask):
            return mask
    raise RuntimeError("failed to generate a valid asymmetric template")


def _choose_directions(template_mask: np.ndarray, count: int) -> list[str]:
    if count == 4:
        chosen = list(DIRECTIONS)
        random.shuffle(chosen)
        return chosen

    valid = []
    for combo in itertools.combinations(DIRECTIONS, count):
        anchor_hits = {
            bool(_transform_mask(template_mask, DIRECTION_TO_TRANSFORM[d])[0, 0]) for d in combo
        }
        if len(anchor_hits) == 2:
            valid.append(combo)
    chosen = list(random.choice(valid))
    random.shuffle(chosen)
    return chosen


def _candidate_anchors(height: int, width: int, direction: str) -> list[tuple[int, int]]:
    dr, dc = DIRECTION_DELTAS[direction]
    template_box = (0, 0, TEMPLATE_SIZE - 1, TEMPLATE_SIZE - 1)
    anchors = []
    for r in range(height - TEMPLATE_SIZE + 1):
        for c in range(width - TEMPLATE_SIZE + 1):
            box = (r, c, r + TEMPLATE_SIZE - 1, c + TEMPLATE_SIZE - 1)
            if _rectangles_overlap(box, template_box):
                continue
            mr, mc = r + dr, c + dc
            if not (0 <= mr < height and 0 <= mc < width):
                continue
            if mr < TEMPLATE_SIZE and mc < TEMPLATE_SIZE:
                continue
            anchors.append((r, c))
    return anchors


def _pick_layout(height: int, width: int, directions: list[str]) -> list[dict[str, int | str]]:
    template_box = (0, 0, TEMPLATE_SIZE - 1, TEMPLATE_SIZE - 1)
    candidates = {d: _candidate_anchors(height, width, d) for d in DIRECTIONS}
    if any(not candidates[d] for d in directions):
        raise RuntimeError("insufficient room for requested directions")

    for _ in range(LAYOUT_ATTEMPTS):
        placed: list[dict[str, int | str]] = []
        boxes: list[tuple[int, int, int, int]] = []
        markers: list[tuple[int, int]] = []

        order = directions[:]
        random.shuffle(order)
        ok = True
        for direction in order:
            sample = candidates[direction][:]
            random.shuffle(sample)
            chosen = None
            for r, c in sample:
                dr, dc = DIRECTION_DELTAS[direction]
                marker = (r + dr, c + dc)
                box = (r, c, r + TEMPLATE_SIZE - 1, c + TEMPLATE_SIZE - 1)

                if any(abs(r - int(item["r"])) + abs(c - int(item["c"])) < 6 for item in placed):
                    continue
                if any(_rectangles_overlap(box, other) for other in boxes):
                    continue
                if _cell_in_box(marker, template_box) or marker in markers:
                    continue
                if any(_cell_in_box(marker, other) for other in boxes):
                    continue
                if any(_cell_in_box(prev_marker, box) for prev_marker in markers):
                    continue
                if any((r, c) == (int(item["r"]), int(item["c"])) for item in placed):
                    continue

                chosen = {"r": r, "c": c, "direction": direction, "marker_r": marker[0], "marker_c": marker[1]}
                break

            if chosen is None:
                ok = False
                break

            placed.append(chosen)
            boxes.append(
                (
                    int(chosen["r"]),
                    int(chosen["c"]),
                    int(chosen["r"]) + TEMPLATE_SIZE - 1,
                    int(chosen["c"]) + TEMPLATE_SIZE - 1,
                )
            )
            markers.append((int(chosen["marker_r"]), int(chosen["marker_c"])))

        if not ok:
            continue

        source_cells = {(int(item["r"]), int(item["c"])) for item in placed}
        marker_cells = {(int(item["marker_r"]), int(item["marker_c"])) for item in placed}
        if len(source_cells) != len(placed) or len(marker_cells) != len(placed):
            continue
        if any(marker in source_cells for marker in marker_cells):
            continue

        ambiguous = False
        for marker in marker_cells:
            touching = sum(abs(marker[0] - r) + abs(marker[1] - c) == 1 for r, c in source_cells)
            if touching != 1:
                ambiguous = True
                break
        if ambiguous:
            continue
        return placed

    raise RuntimeError("failed to place non-overlapping sources and markers")


def _render_input(
    height: int,
    width: int,
    template_mask: np.ndarray,
    placed: list[dict[str, int | str]],
    colors: list[int],
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    grid[:TEMPLATE_SIZE, :TEMPLATE_SIZE][template_mask] = GREY
    for item, color in zip(placed, colors):
        r = int(item["r"])
        c = int(item["c"])
        mr = int(item["marker_r"])
        mc = int(item["marker_c"])
        grid[r, c] = int(color)
        grid[mr, mc] = ORANGE
    return grid


def _expected_output(
    shape: tuple[int, int],
    template_mask: np.ndarray,
    placed: list[dict[str, int | str]],
    colors: list[int],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    out[:TEMPLATE_SIZE, :TEMPLATE_SIZE][template_mask] = GREY
    ordered = sorted(zip(placed, colors), key=lambda item: (int(item[0]["r"]), int(item[0]["c"])))
    for item, color in ordered:
        mask = _transform_mask(template_mask, DIRECTION_TO_TRANSFORM[str(item["direction"])])
        r = int(item["r"])
        c = int(item["c"])
        out[r : r + TEMPLATE_SIZE, c : c + TEMPLATE_SIZE][mask] = int(color)
    return out


def _validate_instance(
    grid: np.ndarray,
    expected: np.ndarray,
    template_mask: np.ndarray,
    placed: list[dict[str, int | str]],
    colors: list[int],
) -> None:
    assert grid.shape == expected.shape
    assert max(grid.shape) <= 30
    assert set(colors).isdisjoint({BACKGROUND, GREY, ORANGE})
    assert len(colors) == len(set(colors))
    assert len(placed) in SOURCE_COUNT_OPTIONS

    transformed = [_transform_mask(template_mask, DIRECTION_TO_TRANSFORM[str(item["direction"])]) for item in placed]
    anchor_hits = {bool(mask[0, 0]) for mask in transformed}
    assert len(anchor_hits) == 2, "used directions must include both anchor-hit and anchor-miss stamps"

    template_only = np.zeros_like(grid, dtype=bool)
    template_only[:TEMPLATE_SIZE, :TEMPLATE_SIZE] = True
    assert np.array_equal(grid[:TEMPLATE_SIZE, :TEMPLATE_SIZE], np.where(template_mask, GREY, BACKGROUND))
    assert np.count_nonzero((grid == GREY) & ~template_only) == 0
    assert np.count_nonzero(expected == ORANGE) == 0
    assert np.array_equal(expected[:TEMPLATE_SIZE, :TEMPLATE_SIZE], grid[:TEMPLATE_SIZE, :TEMPLATE_SIZE])

    occupied = np.zeros_like(grid, dtype=bool)
    occupied[:TEMPLATE_SIZE, :TEMPLATE_SIZE] = True
    for item, color, mask in zip(placed, colors, transformed):
        r = int(item["r"])
        c = int(item["c"])
        mr = int(item["marker_r"])
        mc = int(item["marker_c"])
        block = expected[r : r + TEMPLATE_SIZE, c : c + TEMPLATE_SIZE]
        assert np.all(block[mask] == int(color))
        assert np.all(block[~mask] == BACKGROUND)
        assert grid[r, c] == int(color)
        assert grid[mr, mc] == ORANGE
        assert expected[mr, mc] != ORANGE
        occupied[r : r + TEMPLATE_SIZE, c : c + TEMPLATE_SIZE] = True
        if not mask[0, 0]:
            assert expected[r, c] == BACKGROUND, "source trigger pixels must be cleared before stamping"

    rest = expected.copy()
    rest[occupied] = BACKGROUND
    assert np.count_nonzero(rest) == 0, "output should only contain the preserved template and stamps"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp recolored transformed copies of the top-left template according to each source pixel's orange direction marker.

    Input structure:
    - The grid has a black background, with a single 4x4 grey template in the top-left corner.
    - Elsewhere, 3 or 4 colored source pixels each have exactly one adjacent orange marker in the four-neighborhood.
    - Grey appears only inside the template area, and orange appears only as direction markers beside sources.

    Rule:
    - Read the grey cells inside the top-left 4x4 box as the template sprite.
    - For each source pixel, use the marker position relative to the source to choose the template transform:
      marker on the left -> horizontal mirror, right -> unchanged, above -> 90 degrees counterclockwise,
      below -> 90 degrees clockwise.
    - Recolor the transformed template by replacing grey with the source color, erase all source/marker trigger pixels
      back to black, then stamp the recolored template with its 4x4 top-left corner aligned to the source position.
    - Keep the original grey template unchanged in the top-left corner and process sources in row-major order.

    Key insight:
    - The orange marker is a four-way orientation code, so the output requires decoding marker direction before stamping
      the template with the source color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= TEMPLATE_SIZE and w >= TEMPLATE_SIZE

    template_mask = g[:TEMPLATE_SIZE, :TEMPLATE_SIZE] == GREY
    assert template_mask.any(), "template must contain grey pixels"

    template_area = np.zeros_like(g, dtype=bool)
    template_area[:TEMPLATE_SIZE, :TEMPLATE_SIZE] = True
    assert np.count_nonzero((g == GREY) & ~template_area) == 0, "grey should appear only in the template"
    assert np.count_nonzero((g != BACKGROUND) & template_area & (g != GREY)) == 0, "template area may contain only grey"

    markers = {(int(r), int(c)) for r, c in np.argwhere(g == ORANGE)}
    source_coords = [(int(r), int(c)) for r, c in np.argwhere((g != BACKGROUND) & (g != GREY) & (g != ORANGE))]
    assert 3 <= len(source_coords) <= 4, "expected 3 or 4 colored source pixels"

    items = []
    for r, c in source_coords:
        adjacent = [(nr, nc) for nr, nc in _neighbors4(r, c) if (nr, nc) in markers]
        assert len(adjacent) == 1, "each source must have exactly one adjacent marker"
        marker = adjacent[0]
        delta = (marker[0] - r, marker[1] - c)
        direction = None
        for name, offset in DIRECTION_DELTAS.items():
            if offset == delta:
                direction = name
                break
        assert direction is not None, "marker must be orthogonally adjacent to its source"
        items.append((r, c, int(g[r, c]), direction))

    for marker in markers:
        touching = sum(abs(marker[0] - r) + abs(marker[1] - c) == 1 for r, c in source_coords)
        assert touching == 1, "markers must identify sources unambiguously"

    out = np.zeros_like(g)
    out[:TEMPLATE_SIZE, :TEMPLATE_SIZE] = g[:TEMPLATE_SIZE, :TEMPLATE_SIZE]
    for r, c, color, direction in sorted(items):
        assert r + TEMPLATE_SIZE <= h and c + TEMPLATE_SIZE <= w, "stamp bbox must fit in the grid"
        stamp_mask = _transform_mask(template_mask, DIRECTION_TO_TRANSFORM[direction])
        out[r : r + TEMPLATE_SIZE, c : c + TEMPLATE_SIZE][stamp_mask] = color
    return out


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        height = random.randint(*GRID_SIDE_RANGE)
        width = random.randint(*GRID_SIDE_RANGE)
        count = 4 if random.random() < 0.7 else 3
        template_mask = _make_template()
        directions = _choose_directions(template_mask, count)
        colors = random.sample(SOURCE_COLOR_POOL, count)

        try:
            placed = _pick_layout(height, width, directions)
        except RuntimeError:
            continue

        grid = _render_input(height, width, template_mask, placed, colors)
        expected = _expected_output((height, width), template_mask, placed, colors)

        try:
            _validate_instance(grid, expected, template_mask, placed, colors)
            if not np.array_equal(main(grid), expected):
                continue
        except Exception:
            continue
        return grid

    raise RuntimeError("failed to generate a valid input for puzzle 4")
