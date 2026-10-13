import random

import numpy as np


GREY = 5


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _hole_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            touches_border = False
            while stack:
                rr, cc = stack.pop()
                if rr == 0 or cc == 0 or rr == h - 1 or cc == w - 1:
                    touches_border = True
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if not touches_border:
                holes += 1
    return holes


def _legend_row_and_mapping(grid: np.ndarray, bg: int) -> tuple[int, dict[int, int]]:
    h, _w = grid.shape
    for r in range(h - 1, -1, -1):
        row = grid[r]
        if np.any(row == GREY):
            continue
        entries = [(c, int(v)) for c, v in enumerate(row.tolist()) if int(v) != bg]
        if len(entries) != 4:
            continue
        colors = [color for _c, color in entries]
        if len(set(colors)) != 4:
            continue
        mapping = {i + 1: color for i, color in enumerate(colors)}
        return r, mapping
    raise AssertionError("failed to find the bottom legend row")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey closed outline by its hole count using the bottom legend row.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    legend_row, mapping = _legend_row_and_mapping(g, bg)

    out = g.copy()
    out[legend_row, :] = bg

    grey_mask = g == GREY
    for comp in _components(grey_mask):
        r0, c0, r1, c1 = _bbox(comp)
        submask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            submask[r - r0, c - c0] = True
        holes = _hole_count(submask)
        assert holes in mapping, f"unsupported hole count: {holes}"
        color = int(mapping[holes])
        for r, c in comp:
            out[r, c] = color

    return out


# --------------- generator ---------------

COLOR_POOL = tuple(range(10))
NON_GREY_COLORS = tuple(color for color in COLOR_POOL if color != GREY)

MAX_GENERATION_ATTEMPTS = 500
GRID_HEIGHT_RANGE = (18, 30)
GRID_WIDTH_RANGE = (18, 30)
OBJECT_COUNT_RANGE = (2, 4)
LEGEND_GAP_RANGE = (1, 3)
ROOM_HEIGHT_RANGE = (2, 4)
ROOM_WIDTH_RANGE = (2, 4)
CELL_GRID_MAX_SIDE = 4
LEGEND_MIN_SEP = 2
PLACEMENT_RETRIES = 300
CANVAS_RETRIES = 60


def _true_runs(values: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for i, value in enumerate(values.tolist()):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(values) - 1))
    return runs


def _partition_grid(cell_rows: int, cell_cols: int, room_count: int) -> np.ndarray | None:
    leaves = [(0, cell_rows, 0, cell_cols)]
    while len(leaves) < room_count:
        candidates: list[tuple[int, str, int]] = []
        for idx, (r0, r1, c0, c1) in enumerate(leaves):
            if r1 - r0 >= 2:
                for cut in range(r0 + 1, r1):
                    candidates.append((idx, "h", cut))
            if c1 - c0 >= 2:
                for cut in range(c0 + 1, c1):
                    candidates.append((idx, "v", cut))
        if not candidates:
            return None
        idx, orient, cut = random.choice(candidates)
        r0, r1, c0, c1 = leaves.pop(idx)
        if orient == "h":
            leaves.append((r0, cut, c0, c1))
            leaves.append((cut, r1, c0, c1))
        else:
            leaves.append((r0, r1, c0, cut))
            leaves.append((r0, r1, cut, c1))

    random.shuffle(leaves)
    rooms = np.full((cell_rows, cell_cols), -1, dtype=int)
    for room_id, (r0, r1, c0, c1) in enumerate(leaves):
        rooms[r0:r1, c0:c1] = room_id
    return rooms


def _render_rooms(rooms: np.ndarray) -> np.ndarray:
    cell_rows, cell_cols = rooms.shape
    row_sizes = [random.randint(*ROOM_HEIGHT_RANGE) for _ in range(cell_rows)]
    col_sizes = [random.randint(*ROOM_WIDTH_RANGE) for _ in range(cell_cols)]

    row_edges = [0]
    for size in row_sizes:
        row_edges.append(row_edges[-1] + size + 1)
    col_edges = [0]
    for size in col_sizes:
        col_edges.append(col_edges[-1] + size + 1)

    h = row_edges[-1] + 1
    w = col_edges[-1] + 1
    mask = np.zeros((h, w), dtype=bool)

    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True

    for r in range(cell_rows - 1):
        diff = rooms[r, :] != rooms[r + 1, :]
        for c0, c1 in _true_runs(diff):
            y = row_edges[r + 1]
            x0 = col_edges[c0]
            x1 = col_edges[c1 + 1]
            mask[y, x0 : x1 + 1] = True

    for c in range(cell_cols - 1):
        diff = rooms[:, c] != rooms[:, c + 1]
        for r0, r1 in _true_runs(diff):
            x = col_edges[c + 1]
            y0 = row_edges[r0]
            y1 = row_edges[r1 + 1]
            mask[y0 : y1 + 1, x] = True

    return mask


def _make_object(room_count: int) -> np.ndarray | None:
    candidates = [
        (rows, cols)
        for rows in range(1, CELL_GRID_MAX_SIDE + 1)
        for cols in range(1, CELL_GRID_MAX_SIDE + 1)
        if rows * cols >= room_count
    ]
    random.shuffle(candidates)
    for cell_rows, cell_cols in candidates:
        rooms = _partition_grid(cell_rows, cell_cols, room_count)
        if rooms is None:
            continue
        mask = _render_rooms(rooms)
        turns = random.randrange(4)
        if turns:
            mask = np.rot90(mask, turns)
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        if random.random() < 0.5:
            mask = np.flipud(mask)
        if _hole_count(mask) == room_count:
            return np.asarray(mask, dtype=bool)
    return None


def _sample_hole_counts(object_count: int) -> list[int]:
    while True:
        counts = [random.randint(1, 4) for _ in range(object_count)]
        if len(set(counts)) >= min(2, object_count):
            return counts


def _sample_legend_positions(width: int) -> list[int] | None:
    cols = list(range(1, width - 1))
    if len(cols) < 4:
        return None
    for _ in range(100):
        picked = sorted(random.sample(cols, 4))
        if min(b - a for a, b in zip(picked, picked[1:], strict=False)) >= LEGEND_MIN_SEP:
            return picked
    return None


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _place_objects(
    grid: np.ndarray,
    objects: list[tuple[int, np.ndarray]],
    usable_height: int,
) -> list[tuple[int, int, int, np.ndarray]] | None:
    h, w = grid.shape
    placed_boxes: list[tuple[int, int, int, int]] = []
    placements: list[tuple[int, int, int, np.ndarray]] = []

    for hole_count, mask in sorted(objects, key=lambda item: item[1].shape[0] * item[1].shape[1], reverse=True):
        sh, sw = mask.shape
        if sh > usable_height or sw > w:
            return None
        for _ in range(PLACEMENT_RETRIES):
            r0 = random.randint(0, usable_height - sh)
            c0 = random.randint(0, w - sw)
            padded = (r0 - 1, c0 - 1, r0 + sh, c0 + sw)
            if any(_boxes_overlap(padded, other) for other in placed_boxes):
                continue
            patch = grid[r0 : r0 + sh, c0 : c0 + sw]
            patch[mask] = GREY
            placed_boxes.append(padded)
            placements.append((r0, c0, hole_count, mask))
            break
        else:
            return None

    return placements


def _expected_output(
    shape: tuple[int, int],
    bg: int,
    legend_colors: list[int],
    placements: list[tuple[int, int, int, np.ndarray]],
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for r0, c0, hole_count, mask in placements:
        patch = out[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
        patch[mask] = int(legend_colors[hole_count - 1])
    return out


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(NON_GREY_COLORS)
    legend_colors = random.sample([c for c in NON_GREY_COLORS if c != bg], 4)
    object_count = random.randint(*OBJECT_COUNT_RANGE)
    hole_counts = _sample_hole_counts(object_count)

    objects: list[tuple[int, np.ndarray]] = []
    for hole_count in hole_counts:
        mask = _make_object(hole_count)
        if mask is None:
            return None
        objects.append((hole_count, mask))

    max_h = max(mask.shape[0] for _, mask in objects)
    max_w = max(mask.shape[1] for _, mask in objects)
    padded_area = sum((mask.shape[0] + 2) * (mask.shape[1] + 2) for _, mask in objects)

    for _ in range(CANVAS_RETRIES):
        legend_gap = random.randint(*LEGEND_GAP_RANGE)
        min_h = max(GRID_HEIGHT_RANGE[0], max_h + legend_gap + 1)
        min_w = max(GRID_WIDTH_RANGE[0], max_w, 10)
        if min_h > GRID_HEIGHT_RANGE[1] or min_w > GRID_WIDTH_RANGE[1]:
            return None

        grid_h = random.randint(min_h, GRID_HEIGHT_RANGE[1])
        grid_w = random.randint(min_w, GRID_WIDTH_RANGE[1])
        if grid_h * grid_w < padded_area + 40:
            continue

        legend_positions = _sample_legend_positions(grid_w)
        if legend_positions is None:
            continue

        grid = np.full((grid_h, grid_w), bg, dtype=int)
        legend_row = grid_h - 1
        for c, color in zip(legend_positions, legend_colors, strict=False):
            grid[legend_row, c] = color

        usable_height = legend_row - legend_gap
        placements = _place_objects(grid, objects, usable_height)
        if placements is None:
            continue

        expected = _expected_output(grid.shape, bg, legend_colors, placements)
        if _most_common_color(grid) != bg:
            continue
        if not np.array_equal(main(grid), expected):
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
