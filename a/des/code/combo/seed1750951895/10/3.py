import random
from collections import Counter

import numpy as np


BG_COLOR = 7
SIDE_CYCLE = {1: 9, 3: 1, 5: 3, 7: 5, 9: 7}
SIDE_CHOICES = [1, 3, 5, 7, 9]
GRID_HEIGHT_RANGE = (15, 24)
GRID_WIDTH_RANGE = (15, 24)
OBJECT_COUNT_RANGE = (2, 5)
OUTPUT_GAP = 1
INPUT_GAP = 1
MAX_GENERATION_ATTEMPTS = 300
FOREGROUND_COLORS = [0, 1, 2, 3, 4, 5, 6, 8, 9]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords = []

            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )

    return components


def _bbox(center_r: int, center_c: int, side: int) -> tuple[int, int, int, int]:
    radius = side // 2
    return center_r - radius, center_c - radius, center_r + radius, center_c + radius


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _clear(box: tuple[int, int, int, int], others: list[tuple[int, int, int, int]], gap: int) -> bool:
    return all(_boxes_separated(box, other, gap) for other in others)


def _draw_square(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shrink each odd-sized solid square by two, wrapping a single pixel to 9x9.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    components = _find_components(g, bg)
    assert components, "expected at least one foreground square"

    out = np.full_like(g, bg)
    painted = np.zeros_like(g, dtype=bool)
    h, w = g.shape

    seen_sides = set()
    for comp in components:
        color = int(comp["color"])
        r0, c0, r1, c1 = comp["bbox"]
        side = r1 - r0 + 1
        assert side == c1 - c0 + 1, "each object must have a square bounding box"
        assert side in SIDE_CHOICES, "square side must be one of 1, 3, 5, 7, 9"
        assert side % 2 == 1, "square side must be odd"
        assert side not in seen_sides, "square side lengths must be distinct"
        seen_sides.add(side)
        assert int(comp["area"]) == side * side, "square must fill its bounding box"
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color), "square must be solid"

        new_side = SIDE_CYCLE[side]
        center_r = (r0 + r1) // 2
        center_c = (c0 + c1) // 2
        nr0, nc0, nr1, nc1 = _bbox(center_r, center_c, new_side)
        assert 0 <= nr0 <= nr1 < h and 0 <= nc0 <= nc1 < w, "transformed square must fit"

        region = (slice(nr0, nr1 + 1), slice(nc0, nc1 + 1))
        assert not painted[region].any(), "transformed squares must not overlap"
        out[region] = color
        painted[region] = True

    return out


def _sample_side_set() -> list[int]:
    include_one = random.random() < 0.5
    max_count = 5 if include_one else 4
    count = random.randint(OBJECT_COUNT_RANGE[0], min(OBJECT_COUNT_RANGE[1], max_count))
    if include_one:
        others = random.sample([3, 5, 7, 9], count - 1)
        sides = [1] + others
    else:
        sides = random.sample([3, 5, 7, 9], count)
    random.shuffle(sides)
    return sides


def _sample_specs() -> list[dict] | None:
    sides = _sample_side_set()
    colors = random.sample(FOREGROUND_COLORS, len(sides))
    specs = []
    for side, color in zip(sides, colors):
        out_side = SIDE_CYCLE[side]
        specs.append(
            {
                "input_side": side,
                "output_side": out_side,
                "color": color,
                "footprint": max(side, out_side),
            }
        )

    specs.sort(key=lambda spec: (spec["footprint"], spec["output_side"], spec["input_side"]), reverse=True)
    return specs


def _place_specs(height: int, width: int, specs: list[dict]) -> list[dict] | None:
    placed = []
    input_boxes = []
    output_boxes = []

    for spec in specs:
        margin_r = spec["footprint"] // 2
        margin_c = spec["footprint"] // 2
        centers = [(r, c) for r in range(margin_r, height - margin_r) for c in range(margin_c, width - margin_c)]
        random.shuffle(centers)

        for center_r, center_c in centers:
            in_box = _bbox(center_r, center_c, spec["input_side"])
            out_box = _bbox(center_r, center_c, spec["output_side"])
            if _clear(in_box, input_boxes, INPUT_GAP) and _clear(out_box, output_boxes, OUTPUT_GAP):
                placed.append(
                    {
                        "color": spec["color"],
                        "input_box": in_box,
                        "output_box": out_box,
                    }
                )
                input_boxes.append(in_box)
                output_boxes.append(out_box)
                break
        else:
            return None

    return placed


def _render_input(height: int, width: int, placed: list[dict]) -> np.ndarray:
    grid = np.full((height, width), BG_COLOR, dtype=int)
    for obj in placed:
        _draw_square(grid, obj["input_box"], int(obj["color"]))
    return grid


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    specs = _sample_specs()
    placed = _place_specs(height, width, specs)
    if placed is None:
        return None

    grid = _render_input(height, width, placed)

    counts = Counter(map(int, grid.flat))
    if counts[BG_COLOR] <= max(count for color, count in counts.items() if color != BG_COLOR):
        return None

    if len({obj["color"] for obj in placed}) != len(placed):
        return None

    transformed = main(grid)
    if np.array_equal(transformed, np.full_like(grid, BG_COLOR)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
