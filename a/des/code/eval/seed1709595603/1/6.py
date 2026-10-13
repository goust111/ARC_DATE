from __future__ import annotations

import random

import numpy as np


LINE_COLOR = 5
MARKER_COLOR = 7
TEMPLATE_HEIGHT = 7
TEMPLATE_INNER = 5
TEMPLATE_STRIDE = 6
MAX_GENERATION_ATTEMPTS = 400

PATTERN_LIBRARY: tuple[tuple[tuple[int, int], ...], ...] = (
    ((-1, 0), (1, 0), (0, -1), (0, 1)),
    ((-1, -1), (-1, 1), (1, -1), (1, 1)),
    ((-1, 0), (1, 0)),
    ((0, -1), (0, 1)),
    ((-1, 0), (0, 1)),
    ((-1, 0), (0, -1), (0, 1)),
    ((-1, -1), (1, 1)),
    ((-1, 1), (1, -1)),
    ((-1, -1), (-1, 0), (-1, 1)),
    ((-1, 0), (0, 1), (1, 0)),
)


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _template_bounds(grid: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    nonempty_rows = np.where(np.any(grid != bg, axis=1))[0]
    assert len(nonempty_rows) > 0, "expected at least one foreground row"
    bottom = int(nonempty_rows[-1])
    top = bottom
    while top > 0 and np.any(grid[top - 1] != bg):
        top -= 1

    block = grid[top : bottom + 1]
    nonempty_cols = np.where(np.any(block != bg, axis=0))[0]
    assert len(nonempty_cols) > 0, "expected nonempty template columns"
    left = int(nonempty_cols[0])
    right = int(nonempty_cols[-1])
    return top, bottom, left, right


def _parse_templates(grid: np.ndarray, bg: int) -> tuple[int, dict[int, tuple[tuple[int, int], ...]]]:
    top, bottom, left, right = _template_bounds(grid, bg)
    bbox = grid[top : bottom + 1, left : right + 1]
    height, width = map(int, bbox.shape)
    assert height == TEMPLATE_HEIGHT, "template strip must be one row of 5x5 boxes"
    assert (width - 1) % TEMPLATE_STRIDE == 0, "template width must match 5x5 boxes plus grey dividers"

    line_color = int(bbox[0, 0])
    assert line_color != bg, "template outline color must differ from background"
    assert np.all(bbox[0] == line_color) and np.all(bbox[-1] == line_color), "template needs grey top/bottom borders"

    n_templates = (width - 1) // TEMPLATE_STRIDE
    mapping: dict[int, tuple[tuple[int, int], ...]] = {}
    for idx in range(n_templates):
        col0 = idx * TEMPLATE_STRIDE
        assert np.all(bbox[:, col0] == line_color), "template cells must be separated by grey vertical lines"
        inner = bbox[1:6, col0 + 1 : col0 + 6]
        center_color = int(inner[2, 2])
        assert center_color not in (bg, line_color), "template center must be a growth color"

        directions: list[tuple[int, int]] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                if int(inner[2 + dr, 2 + dc]) == center_color:
                    directions.append((dr, dc))

        assert directions, "each template must show at least one growth direction"
        assert center_color not in mapping, "template colors must be unique"
        mapping[center_color] = tuple(directions)

    assert np.all(bbox[:, -1] == line_color), "template strip must end with a grey border"
    return top, mapping


def _marker_color(grid: np.ndarray, template_top: int, bg: int, template_colors: set[int]) -> int | None:
    main_zone = grid[:template_top]
    colors = {int(v) for v in np.unique(main_zone) if int(v) != bg}
    extras = colors - template_colors
    assert len(extras) <= 1, "expected at most one marker color"
    return None if not extras else extras.pop()


def _count_markers(grid: np.ndarray, row: int, col: int, marker_color: int | None) -> int:
    if marker_color is None:
        return 0
    count = 0
    rr = row - 2
    while rr >= 0 and int(grid[rr, col]) == marker_color:
        count += 1
        rr -= 1
    return count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read bottom templates as per-color growth directions, then expand each seed by
    the number of orange markers stacked above it.
    """
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    template_top, mapping = _parse_templates(g, bg)
    template_colors = set(mapping)
    marker_color = _marker_color(g, template_top, bg, template_colors)

    out = g.copy()
    if marker_color is not None:
        marker_mask = np.zeros_like(out, dtype=bool)
        marker_mask[:template_top] = g[:template_top] == marker_color
        out[marker_mask] = bg

    seed_positions = [
        (int(row), int(col))
        for row, col in np.argwhere(np.isin(g[:template_top], list(template_colors)))
    ]
    assert seed_positions, "expected at least one seed in the main area"
    seed_positions.sort()

    height, width = map(int, out.shape)
    for row, col in seed_positions:
        color = int(g[row, col])
        generations = _count_markers(g, row, col, marker_color)
        if generations <= 0:
            continue

        frontier = {(row, col)}
        for _ in range(generations):
            nxt: set[tuple[int, int]] = set()
            for rr, cc in frontier:
                for dr, dc in mapping[color]:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < height and 0 <= nc < width and int(out[nr, nc]) == bg:
                        out[nr, nc] = color
                        nxt.add((nr, nc))
            if not nxt:
                break
            frontier = nxt

    return out


def _territory_offsets(directions: tuple[tuple[int, int], ...], generations: int) -> set[tuple[int, int]]:
    occupied = {(0, 0)}
    frontier = {(0, 0)}
    for _ in range(generations):
        nxt: set[tuple[int, int]] = set()
        for row, col in frontier:
            for dr, dc in directions:
                cell = (row + dr, col + dc)
                if cell not in occupied:
                    occupied.add(cell)
                    nxt.add(cell)
        if not nxt:
            break
        frontier = nxt
    return occupied


def _build_template_strip(colors: list[int], patterns: list[tuple[tuple[int, int], ...]], bg: int) -> np.ndarray:
    width = len(colors) * TEMPLATE_STRIDE + 1
    strip = np.full((TEMPLATE_HEIGHT, width), bg, dtype=int)
    strip[0, :] = LINE_COLOR
    strip[-1, :] = LINE_COLOR
    for col in range(0, width, TEMPLATE_STRIDE):
        strip[:, col] = LINE_COLOR

    for idx, (color, directions) in enumerate(zip(colors, patterns)):
        col0 = idx * TEMPLATE_STRIDE + 1
        strip[3, col0 + 2] = color
        for dr, dc in directions:
            strip[3 + dr, col0 + 2 + dc] = color
    return strip


def _seed_specs(colors: list[int], mapping: dict[int, tuple[tuple[int, int], ...]]) -> list[dict[str, object]]:
    n_templates = len(colors)
    n_seeds = random.randint(n_templates, min(6, n_templates + 3))

    generations: list[int] = []
    if n_seeds >= 4 and random.random() < 0.8:
        generations.extend([0, 1, 2, 3])
    while len(generations) < n_seeds:
        generations.append(random.randint(0, 3))
    if all(value == 0 for value in generations):
        generations[random.randrange(len(generations))] = random.randint(1, 3)
    random.shuffle(generations)

    seed_colors = colors[:]
    while len(seed_colors) < n_seeds:
        seed_colors.append(random.choice(colors))
    random.shuffle(seed_colors)

    specs: list[dict[str, object]] = []
    for color, generations_count in zip(seed_colors, generations):
        directions = mapping[color]
        offsets = _territory_offsets(directions, generations_count)
        specs.append(
            {
                "color": color,
                "generations": generations_count,
                "directions": directions,
                "offsets": offsets,
                "size": len(offsets),
            }
        )
    specs.sort(key=lambda item: (int(item["size"]), int(item["generations"])), reverse=True)
    return specs


def _place_specs(main_h: int, total_w: int, specs: list[dict[str, object]]) -> list[dict[str, object]] | None:
    output_occupied: set[tuple[int, int]] = set()
    input_occupied: set[tuple[int, int]] = set()
    placements: list[dict[str, object]] = []

    for spec in specs:
        offsets = set(spec["offsets"])
        min_r = min(row for row, _ in offsets)
        max_r = max(row for row, _ in offsets)
        min_c = min(col for _, col in offsets)
        max_c = max(col for _, col in offsets)
        generations = int(spec["generations"])

        marker_min_row = generations + 1 if generations > 0 else 0
        row_lo = max(-min_r, marker_min_row)
        row_hi = main_h - 1 - max_r
        col_lo = -min_c
        col_hi = total_w - 1 - max_c
        if row_lo > row_hi or col_lo > col_hi:
            return None

        placed = None
        for _ in range(300):
            row = random.randint(row_lo, row_hi)
            col = random.randint(col_lo, col_hi)
            territory = {(row + dr, col + dc) for dr, dc in offsets}
            markers = {(row - 2 - idx, col) for idx in range(generations)}

            if territory & output_occupied:
                continue
            if territory & input_occupied:
                continue
            if markers & output_occupied:
                continue
            if markers & input_occupied:
                continue

            placed = {
                "row": row,
                "col": col,
                "color": int(spec["color"]),
                "generations": generations,
                "directions": spec["directions"],
                "offsets": offsets,
                "markers": markers,
            }
            break

        if placed is None:
            return None

        placements.append(placed)
        output_occupied |= {(placed["row"] + dr, placed["col"] + dc) for dr, dc in offsets}
        input_occupied.add((int(placed["row"]), int(placed["col"])))
        input_occupied |= set(placed["markers"])

    return placements


def _render_pair(
    bg: int,
    main_h: int,
    gap_h: int,
    total_w: int,
    template_left: int,
    template_strip: np.ndarray,
    placements: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    total_h = main_h + gap_h + TEMPLATE_HEIGHT
    inp = np.full((total_h, total_w), bg, dtype=int)
    out = np.full_like(inp, bg)

    template_top = main_h + gap_h
    template_w = int(template_strip.shape[1])
    inp[template_top:, template_left : template_left + template_w] = template_strip
    out[template_top:, template_left : template_left + template_w] = template_strip

    for placed in placements:
        row = int(placed["row"])
        col = int(placed["col"])
        color = int(placed["color"])
        inp[row, col] = color
        out[row, col] = color
        for mr, mc in placed["markers"]:
            inp[mr, mc] = MARKER_COLOR

    for placed in placements:
        color = int(placed["color"])
        row = int(placed["row"])
        col = int(placed["col"])
        for dr, dc in placed["offsets"]:
            out[row + dr, col + dc] = color

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([0, 1, 2, 3, 4, 6, 8, 9])
    usable_colors = [color for color in range(10) if color not in {bg, LINE_COLOR, MARKER_COLOR}]
    n_templates = random.randint(2, min(4, len(usable_colors)))
    colors = random.sample(usable_colors, n_templates)
    patterns = random.sample(list(PATTERN_LIBRARY), n_templates)
    mapping = {color: pattern for color, pattern in zip(colors, patterns)}

    template_strip = _build_template_strip(colors, patterns, bg)
    template_w = int(template_strip.shape[1])

    total_w = max(random.randint(18, 30), template_w + random.randint(0, 8))
    main_h = random.randint(11, 18)
    gap_h = random.randint(2, 4)

    specs = _seed_specs(colors, mapping)
    placements = _place_specs(main_h, total_w, specs)
    if placements is None:
        return None

    template_left = random.randint(0, total_w - template_w)
    inp, expected = _render_pair(bg, main_h, gap_h, total_w, template_left, template_strip, placements)

    if _dominant_color(inp) != bg:
        return None
    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
