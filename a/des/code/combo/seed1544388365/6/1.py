import random

import numpy as np


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(pts)
    return parts


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _mode_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(colors[np.argmax(counts)])


def _is_rect_frame(points: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(points)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    point_set = set(points)
    expected: set[tuple[int, int]] = set()
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    return point_set == expected


def _find_frames(grid: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int, int, int]]]:
    best_color = None
    best_frames: list[tuple[int, int, int, int]] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        frames: list[tuple[int, int, int, int]] = []
        for comp in _components(grid == color):
            if _is_rect_frame(comp):
                frames.append(_bbox(comp))
        if len(frames) > len(best_frames):
            best_color = color
            best_frames = frames
    assert best_color is not None and best_frames, "expected at least one rectangular frame color"
    best_frames.sort()
    return best_color, best_frames


def _legend_mapping(grid: np.ndarray, top_limit: int, bg: int, frame_color: int) -> dict[int, int]:
    strip = grid[:top_limit, :]
    blocks: list[dict[str, tuple[int, int, int, int] | int]] = []
    for color in map(int, np.unique(strip)):
        if color in (bg, frame_color):
            continue
        for comp in _components(strip == color):
            r0, c0, r1, c1 = _bbox(comp)
            patch = strip[r0 : r1 + 1, c0 : c1 + 1]
            assert np.all(patch == color), "legend block should be a solid rectangle"
            blocks.append({"color": color, "bbox": (r0, c0, r1, c1)})

    assert len(blocks) >= 2 and len(blocks) % 2 == 0, "expected an even number of legend blocks"
    blocks.sort(key=lambda item: (int(item["bbox"][1]), int(item["bbox"][0])))

    mapping: dict[int, int] = {}
    for i in range(0, len(blocks), 2):
        left = blocks[i]
        right = blocks[i + 1]
        lr0, lc0, lr1, lc1 = left["bbox"]
        rr0, rc0, rr1, rc1 = right["bbox"]
        assert lr0 == rr0 and lr1 == rr1, "paired legend blocks should share vertical span"
        assert lc1 < rc0, "legend blocks must be ordered left-to-right"
        gap = strip[lr0 : lr1 + 1, lc1 + 1 : rc0]
        if gap.size:
            assert np.all((gap == bg) | (gap == frame_color)), "unexpected pixels between legend blocks"

        a = int(left["color"])
        b = int(right["color"])
        assert a != b and a not in mapping and b not in mapping, "legend colors must form disjoint pairs"
        mapping[a] = b
        mapping[b] = a

    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each framed fragment to the partner color from the top legend pair."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _mode_color(g)
    frame_color, frames = _find_frames(g, bg)
    top_limit = min(r0 for r0, _, _, _ in frames)
    legend = _legend_mapping(g, top_limit, bg, frame_color)

    out = g.copy()
    for r0, c0, r1, c1 in frames:
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must have an interior"
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        fragment_colors = [int(color) for color in np.unique(inner) if int(color) not in (bg, frame_color)]
        assert len(fragment_colors) == 1, "each frame should contain exactly one fragment color"
        src = fragment_colors[0]
        assert src in legend, "fragment color must appear in a legend pair"
        dst = legend[src]
        out_inner = out[r0 + 1 : r1, c0 + 1 : c1]
        out_inner[inner == src] = dst

    return out


# --------------- generator ---------------

BG = 0
FRAME = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

PAIR_COUNT_RANGE = (2, 4)
FRAME_COUNT_RANGE = (2, 4)
LEGEND_BLOCK_SIZES = (2, 3)

FRAME_OUTER_HEIGHT_RANGE = (6, 9)
FRAME_OUTER_WIDTH_RANGE = (6, 9)
TOP_MARGIN_RANGE = (1, 2)
BOTTOM_MARGIN_RANGE = (1, 3)
SIDE_SLACK_RANGE = (2, 4)
VERTICAL_GAP_RANGE = (2, 4)
HORIZONTAL_GAP_RANGE = (2, 4)

MIN_GRID_SIDE = 16
MAX_GRID_SIDE = 32
GEN_RETRIES = 300


def _row_partitions(count: int) -> list[list[int]]:
    if count == 2:
        return [[2], [1, 1]]
    if count == 3:
        return [[3], [1, 2], [2, 1]]
    return [[2, 2], [1, 3], [3, 1]]


def _grow_blob(h: int, w: int, target: int) -> list[tuple[int, int]] | None:
    cells = {(random.randrange(h), random.randrange(w))}
    while len(cells) < target:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))

    pts = sorted(cells)
    r0, c0, r1, c1 = _bbox(pts)
    if len(pts) == (r1 - r0 + 1) * (c1 - c0 + 1):
        return None
    return pts


def _frame_assignments(pair_count: int, frame_count: int) -> tuple[list[int], list[int]]:
    pair_ids = [random.randrange(pair_count) for _ in range(frame_count)]
    if frame_count >= 2 and pair_count >= 2 and len(set(pair_ids)) == 1:
        pair_ids[-1] = (pair_ids[-1] + 1) % pair_count

    sides = [random.randrange(2) for _ in range(frame_count)]
    if frame_count >= 2 and len(set(sides)) == 1:
        sides[0] = 1 - sides[0]

    return pair_ids, sides


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    legend_colors = random.sample(COLOR_POOL, 2 * pair_count)
    pairs = [tuple(legend_colors[2 * i : 2 * i + 2]) for i in range(pair_count)]

    frame_count = random.randint(*FRAME_COUNT_RANGE)
    frame_sizes = [
        (
            random.randint(*FRAME_OUTER_HEIGHT_RANGE),
            random.randint(*FRAME_OUTER_WIDTH_RANGE),
        )
        for _ in range(frame_count)
    ]
    random.shuffle(frame_sizes)

    row_partition = random.choice(_row_partitions(frame_count))
    rows: list[list[tuple[int, int]]] = []
    cursor = 0
    for row_len in row_partition:
        row = frame_sizes[cursor : cursor + row_len]
        cursor += row_len
        rows.append(row)

    row_widths: list[int] = []
    row_heights: list[int] = []
    row_gaps: list[list[int]] = []
    for row in rows:
        gaps = [random.randint(*HORIZONTAL_GAP_RANGE) for _ in range(max(0, len(row) - 1))]
        row_gaps.append(gaps)
        row_widths.append(sum(w for _, w in row) + sum(gaps))
        row_heights.append(max(h for h, _ in row))

    block = random.choice(LEGEND_BLOCK_SIZES)
    legend_w = pair_count * 2 * block + (pair_count - 1)
    legend_h = block

    top_margin = random.randint(*TOP_MARGIN_RANGE)
    bottom_margin = random.randint(*BOTTOM_MARGIN_RANGE)
    legend_gap = random.randint(*VERTICAL_GAP_RANGE)
    row_gap = random.randint(*VERTICAL_GAP_RANGE) if len(rows) > 1 else 0
    side_slack = random.randint(*SIDE_SLACK_RANGE)

    grid_h = top_margin + legend_h + legend_gap + sum(row_heights) + row_gap * (len(rows) - 1) + bottom_margin
    grid_w = max(legend_w, max(row_widths)) + side_slack
    if not (MIN_GRID_SIDE <= grid_h <= MAX_GRID_SIDE and MIN_GRID_SIDE <= grid_w <= MAX_GRID_SIDE):
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)

    legend_x = random.randint(1, grid_w - legend_w - 1)
    legend_y = top_margin
    x = legend_x
    for idx, (left, right) in enumerate(pairs):
        grid[legend_y : legend_y + block, x : x + block] = left
        x += block
        grid[legend_y : legend_y + block, x : x + block] = right
        x += block
        if idx + 1 < pair_count:
            grid[legend_y : legend_y + block, x] = FRAME
            x += 1

    pair_ids, sides = _frame_assignments(pair_count, frame_count)

    frame_index = 0
    row_top = top_margin + legend_h + legend_gap
    for row_idx, row in enumerate(rows):
        row_width = row_widths[row_idx]
        row_height = row_heights[row_idx]
        row_x = random.randint(1, grid_w - row_width - 1)
        cursor_x = row_x

        for item_idx, (outer_h, outer_w) in enumerate(row):
            r0 = row_top + random.randint(0, row_height - outer_h)
            c0 = cursor_x
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1
            _draw_frame(grid, (r0, c0, r1, c1))

            pair_id = pair_ids[frame_index]
            src_color = pairs[pair_id][sides[frame_index]]

            inner_h = outer_h - 2
            inner_w = outer_w - 2
            usable_r = r0 + 2
            usable_c = c0 + 2
            usable_h = inner_h - 2
            usable_w = inner_w - 2
            if usable_h < 2 or usable_w < 2:
                return None

            usable_area = usable_h * usable_w
            max_cells = min(usable_area - 1, max(3, int(round(usable_area * 0.6))))
            if max_cells < 3:
                return None
            target = random.randint(3, max_cells)
            blob = _grow_blob(usable_h, usable_w, target)
            if blob is None:
                return None
            for br, bc in blob:
                grid[usable_r + br, usable_c + bc] = src_color

            frame_index += 1
            if item_idx < len(row) - 1:
                cursor_x += outer_w + row_gaps[row_idx][item_idx]

        row_top += row_height
        if row_idx + 1 < len(rows):
            row_top += row_gap

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
