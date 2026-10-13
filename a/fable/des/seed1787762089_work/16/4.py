import random
from collections import Counter

import numpy as np


BG = 0
SEP = 5
RECT_COLORS = (1, 3, 7, 9)
MARKER_COLORS = (2, 4)
TOP_STRIP_HEIGHT = 2
MAX_ATTEMPTS = 400

GRID_HEIGHT_RANGE = (16, 22)
GRID_WIDTH_RANGE = (16, 22)


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_separator_row(grid: np.ndarray) -> tuple[int, int, int]:
    bg = _background_color(grid)
    rows = [
        r
        for r in range(grid.shape[0])
        if np.all(grid[r, :] == grid[r, 0]) and int(grid[r, 0]) != bg
    ]
    assert len(rows) == 1, "expected one non-background full-width separator row"
    row = rows[0]
    return row, int(grid[row, 0]), bg


def _connected_rectangles(scene: np.ndarray, bg: int) -> list[dict[str, int | tuple[int, int, int, int]]]:
    h, w = scene.shape
    seen = np.zeros((h, w), dtype=bool)
    rectangles = []

    for r in range(h):
        for c in range(w):
            color = int(scene[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(scene[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [cell[0] for cell in cells]
            cols = [cell[1] for cell in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            block = scene[r0 : r1 + 1, c0 : c1 + 1]
            assert np.all(block == color), "foreground component must be a solid rectangle"
            rectangles.append(
                {
                    "color": color,
                    "area": int(block.size),
                    "bbox": (r0, c0, r1, c1),
                }
            )

    return rectangles


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    sep_row, sep_color, bg = _find_separator_row(g)

    strip = g[:sep_row, :]
    marker_counts = Counter(
        int(value) for value in strip.reshape(-1) if int(value) not in (bg, sep_color)
    )
    assert marker_counts, "count strip must contain markers"
    assert len(set(marker_counts.values())) == len(marker_counts), "marker counts must be distinct"

    scene = g[sep_row + 1 :, :].copy()
    rectangles = _connected_rectangles(scene, bg)
    assert rectangles, "main region must contain rectangles"
    assert len({int(rect["area"]) for rect in rectangles}) == len(rectangles), "rectangle areas must be distinct"

    ranked = sorted(
        rectangles,
        key=lambda rect: (
            -int(rect["area"]),
            int(rect["bbox"][0]),
            int(rect["bbox"][1]),
        ),
    )
    for color, count in marker_counts.items():
        assert 1 <= count <= len(ranked), "marker count must index an existing rank"
        r0, c0, r1, c1 = ranked[count - 1]["bbox"]
        scene[r0 : r1 + 1, c0 : c1 + 1] = int(color)

    return scene


def _choose_marker_plan() -> dict[int, int]:
    if random.choice((False, True)):
        color = random.choice(MARKER_COLORS)
        return {color: random.randint(1, 4)}

    counts = random.sample((1, 2, 3, 4), 2)
    return {MARKER_COLORS[0]: counts[0], MARKER_COLORS[1]: counts[1]}


def _paint_marker_strip(grid: np.ndarray, marker_plan: dict[int, int]) -> bool:
    total = sum(marker_plan.values())
    parity = random.choice((0, 1))
    candidate_cols = list(range(parity, grid.shape[1], 2))
    if len(candidate_cols) < total:
        return False

    chosen_cols = random.sample(candidate_cols, total)
    random.shuffle(chosen_cols)
    index = 0
    for color, count in marker_plan.items():
        for _ in range(count):
            row = random.randrange(TOP_STRIP_HEIGHT)
            col = chosen_cols[index]
            index += 1
            grid[row, col] = color
    return True


def _regions(scene_h: int, scene_w: int) -> list[tuple[int, int, int, int]] | None:
    if scene_h < 8 or scene_w < 8:
        return None

    split_r = random.randint(3, scene_h - 4)
    split_c = random.randint(3, scene_w - 4)

    regions = [
        (1, split_r - 1, 1, split_c - 1),
        (1, split_r - 1, split_c + 1, scene_w - 2),
        (split_r + 1, scene_h - 2, 1, split_c - 1),
        (split_r + 1, scene_h - 2, split_c + 1, scene_w - 2),
    ]
    if any(r0 > r1 or c0 > c1 for r0, r1, c0, c1 in regions):
        return None
    if any((r1 - r0 + 1) < 2 or (c1 - c0 + 1) < 2 for r0, r1, c0, c1 in regions):
        return None
    return regions


def _place_rectangles(scene_h: int, scene_w: int) -> list[dict[str, int | tuple[int, int, int, int]]] | None:
    for _ in range(100):
        regions = _regions(scene_h, scene_w)
        if regions is None:
            continue

        colors = random.sample(list(RECT_COLORS), len(regions))
        rectangles = []
        areas = set()
        ok = True

        for color, (r0, r1, c0, c1) in zip(colors, regions):
            max_h = r1 - r0 + 1
            max_w = c1 - c0 + 1
            height = random.randint(2, max_h)
            width = random.randint(2, max_w)
            area = height * width
            if area in areas:
                ok = False
                break

            top = random.randint(r0, r1 - height + 1)
            left = random.randint(c0, c1 - width + 1)
            areas.add(area)
            rectangles.append(
                {
                    "color": int(color),
                    "area": area,
                    "bbox": (top, left, top + height - 1, left + width - 1),
                }
            )

        if ok:
            return rectangles

    return None


def _expected_output(
    shape: tuple[int, int],
    rectangles: list[dict[str, int | tuple[int, int, int, int]]],
    marker_plan: dict[int, int],
) -> np.ndarray:
    scene_h = shape[0] - TOP_STRIP_HEIGHT - 1
    out = np.full((scene_h, shape[1]), BG, dtype=int)
    for rect in rectangles:
        r0, c0, r1, c1 = rect["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(rect["color"])

    ranked = sorted(
        rectangles,
        key=lambda rect: (
            -int(rect["area"]),
            int(rect["bbox"][0]),
            int(rect["bbox"][1]),
        ),
    )
    for color, count in marker_plan.items():
        r0, c0, r1, c1 = ranked[count - 1]["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    grid = np.full((height, width), BG, dtype=int)
    grid[TOP_STRIP_HEIGHT, :] = SEP

    marker_plan = _choose_marker_plan()
    if not _paint_marker_strip(grid, marker_plan):
        return None

    scene_h = height - TOP_STRIP_HEIGHT - 1
    rectangles = _place_rectangles(scene_h, width)
    if rectangles is None:
        return None

    for rect in rectangles:
        r0, c0, r1, c1 = rect["bbox"]
        grid[TOP_STRIP_HEIGHT + 1 + r0 : TOP_STRIP_HEIGHT + 2 + r1, c0 : c1 + 1] = int(rect["color"])

    expected = _expected_output(grid.shape, rectangles, marker_plan)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
