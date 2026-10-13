from __future__ import annotations

import random

import numpy as np


BG = 0
SEP = 5
ARROW_COLOR = 2
PATTERN_COLORS = (1, 3, 4, 6, 7, 8, 9)
CARDINAL4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
ARROW_PATTERNS = {
    "up": frozenset({(0, 1), (1, 0), (1, 2)}),
    "down": frozenset({(0, 0), (0, 2), (1, 1)}),
    "left": frozenset({(0, 1), (1, 0), (2, 1)}),
    "right": frozenset({(0, 0), (1, 1), (2, 0)}),
}
LAYOUT_CHOICES = tuple(
    (n_rows, n_cols, room_h, room_w)
    for n_rows in range(2, 6)
    for n_cols in range(2, 6)
    for room_h in range(4, 7)
    for room_w in range(4, 7)
    if n_rows * room_h + n_rows + 1 <= 30 and n_cols * room_w + n_cols + 1 <= 30
)


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _room_spans(lines: list[int], limit: int) -> list[tuple[int, int]]:
    bounds = [-1] + sorted({int(x) for x in lines}) + [int(limit)]
    spans: list[tuple[int, int]] = []
    for left, right in zip(bounds, bounds[1:]):
        if right - left > 1:
            spans.append((int(left + 1), int(right - 1)))
    return spans


def _find_separator_color(grid: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = grid.shape
    best: tuple[int, list[tuple[int, int]], list[tuple[int, int]]] | None = None
    best_score: tuple[int, int] | None = None
    for color in map(int, np.unique(grid)):
        if color == int(bg):
            continue
        row_lines = [r for r in range(h) if bool(np.all(grid[r, :] == int(color)))]
        col_lines = [c for c in range(w) if bool(np.all(grid[:, c] == int(color)))]
        if not row_lines or not col_lines:
            continue
        row_spans = _room_spans(row_lines, h)
        col_spans = _room_spans(col_lines, w)
        room_count = len(row_spans) * len(col_spans)
        if room_count < 2:
            continue
        line_cells = len(row_lines) * w + len(col_lines) * h - len(row_lines) * len(col_lines)
        score = (int(room_count), int(line_cells))
        if best_score is None or score > best_score:
            best = (int(color), row_spans, col_spans)
            best_score = score
    if best is None:
        raise ValueError("could not find a room separator color")
    return best


def _room_patch(grid: np.ndarray, row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]], room: tuple[int, int]) -> np.ndarray:
    ri, ci = room
    r0, r1 = row_spans[int(ri)]
    c0, c1 = col_spans[int(ci)]
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _write_room(
    grid: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    room: tuple[int, int],
    patch: np.ndarray,
) -> None:
    ri, ci = room
    r0, r1 = row_spans[int(ri)]
    c0, c1 = col_spans[int(ci)]
    grid[r0 : r1 + 1, c0 : c1 + 1] = patch


def _fit_patch(src: np.ndarray, dst_shape: tuple[int, int], bg: int) -> np.ndarray:
    out = np.full(dst_shape, int(bg), dtype=int)
    h = min(int(src.shape[0]), int(dst_shape[0]))
    w = min(int(src.shape[1]), int(dst_shape[1]))
    out[:h, :w] = src[:h, :w]
    return out


def _detect_arrow(patch: np.ndarray, bg: int, sep: int) -> tuple[str, int] | None:
    mask = (patch != int(bg)) & (patch != int(sep))
    coords = np.argwhere(mask)
    if int(coords.shape[0]) != 3:
        return None
    colors = {int(patch[int(r), int(c)]) for r, c in coords}
    if len(colors) != 1:
        return None
    r0, c0 = coords.min(axis=0)
    norm = frozenset((int(r - r0), int(c - c0)) for r, c in coords)
    for direction, shape in ARROW_PATTERNS.items():
        if norm == shape:
            return direction, next(iter(colors))
    return None


def _chain_rooms(direction: str, arrow_room: tuple[int, int], n_rows: int, n_cols: int) -> list[tuple[int, int]]:
    ri, ci = arrow_room
    if direction == "right":
        return [(int(ri), c) for c in range(int(ci), int(n_cols))]
    if direction == "left":
        return [(int(ri), c) for c in range(int(ci), -1, -1)]
    if direction == "down":
        return [(r, int(ci)) for r in range(int(ri), int(n_rows))]
    return [(r, int(ci)) for r in range(int(ri), -1, -1)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push room contents along the row or column indicated by a triangular arrow room.

    Input structure:
    - bg is the most common color.
    - Grey single-pixel separator lines split the grid into rectangular rooms.
    - Some rooms contain small colored patterns on the background.
    - Exactly one room contains only a 3-pixel triangular arrow shape in one color, pointing up, down, left, or right.

    Rule:
    - Detect the separator color and decode the rectangular room layout.
    - Find the unique arrow room and its direction.
    - Starting from the arrow room and continuing toward the pointed edge, shift each room's full interior content into the next room on that chain; the farthest room's old content is discarded.
    - After its content has been pushed out, replace the arrow room interior with a solid fill of the arrow color.
    - Separator lines and rooms not on the directed chain stay unchanged.

    Key insight:
    - The arrow is not just a sprite; it is an instruction embedded inside one room, and the real action is a one-sided queue push across the whole aligned room chain.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("grid must be 2D")

    bg = _mode_color(g)
    sep, row_spans, col_spans = _find_separator_color(g, bg)

    room_patches: dict[tuple[int, int], np.ndarray] = {}
    arrow_candidates: list[tuple[int, int, str, int]] = []
    for ri in range(len(row_spans)):
        for ci in range(len(col_spans)):
            room = (int(ri), int(ci))
            patch = _room_patch(g, row_spans, col_spans, room)
            room_patches[room] = patch
            arrow = _detect_arrow(patch, bg, sep)
            if arrow is not None:
                direction, arrow_color = arrow
                arrow_candidates.append((int(ri), int(ci), direction, int(arrow_color)))

    if not arrow_candidates:
        raise ValueError("no arrow room found")
    if len(arrow_candidates) == 1:
        arrow_ri, arrow_ci, direction, arrow_color = arrow_candidates[0]
    else:
        red_candidates = [cand for cand in arrow_candidates if int(cand[3]) == ARROW_COLOR]
        arrow_ri, arrow_ci, direction, arrow_color = (red_candidates[0] if len(red_candidates) == 1 else arrow_candidates[0])

    out = g.copy()
    chain = _chain_rooms(direction, (arrow_ri, arrow_ci), len(row_spans), len(col_spans))
    for idx in range(len(chain) - 1, 0, -1):
        src_room = chain[idx - 1]
        dst_room = chain[idx]
        dst_shape = room_patches[dst_room].shape
        moved = _fit_patch(room_patches[src_room], dst_shape, bg)
        _write_room(out, row_spans, col_spans, dst_room, moved)

    arrow_shape = room_patches[(arrow_ri, arrow_ci)].shape
    _write_room(
        out,
        row_spans,
        col_spans,
        (arrow_ri, arrow_ci),
        np.full(arrow_shape, int(arrow_color), dtype=int),
    )
    return out


def _random_connected_cells(height: int, width: int, n_cells: int) -> set[tuple[int, int]]:
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    while len(cells) < n_cells:
        frontier = {
            (r + dr, c + dc)
            for r, c in cells
            for dr, dc in CARDINAL4
            if 0 <= r + dr < height and 0 <= c + dc < width and (r + dr, c + dc) not in cells
        }
        if not frontier:
            break
        cells.add(random.choice(tuple(frontier)))
    return cells


def _make_room_pattern(room_h: int, room_w: int) -> np.ndarray:
    patch = np.full((room_h, room_w), BG, dtype=int)
    bbox_h = random.randint(2, room_h - 1)
    bbox_w = random.randint(2, room_w - 1)
    max_cells = min(8, bbox_h * bbox_w)
    n_cells = random.randint(4, max_cells)
    cells = list(_random_connected_cells(bbox_h, bbox_w, n_cells))
    base_color = random.choice(PATTERN_COLORS)
    accent_count = 0
    if len(cells) >= 5 and random.random() < 0.45:
        accent_count = random.randint(1, min(2, len(cells) - 1))
    off_r = random.randint(0, room_h - bbox_h)
    off_c = random.randint(0, room_w - bbox_w)
    for r, c in cells:
        patch[int(off_r + r), int(off_c + c)] = int(base_color)
    if accent_count:
        accent_color = random.choice([c for c in PATTERN_COLORS if c != int(base_color)])
        for r, c in random.sample(cells, accent_count):
            patch[int(off_r + r), int(off_c + c)] = int(accent_color)
    return patch


def _make_arrow_room(room_h: int, room_w: int, direction: str) -> np.ndarray:
    patch = np.full((room_h, room_w), BG, dtype=int)
    shape = ARROW_PATTERNS[direction]
    shape_h = max(r for r, _ in shape) + 1
    shape_w = max(c for _, c in shape) + 1
    off_r = random.randint(0, room_h - shape_h)
    off_c = random.randint(0, room_w - shape_w)
    for r, c in shape:
        patch[int(off_r + r), int(off_c + c)] = ARROW_COLOR
    return patch


def _draw_layout(n_rows: int, n_cols: int, room_h: int, room_w: int) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    height = n_rows * room_h + n_rows + 1
    width = n_cols * room_w + n_cols + 1
    grid = np.full((height, width), BG, dtype=int)
    sep_rows = [i * (room_h + 1) for i in range(n_rows + 1)]
    sep_cols = [i * (room_w + 1) for i in range(n_cols + 1)]
    grid[sep_rows, :] = SEP
    grid[:, sep_cols] = SEP
    row_spans = _room_spans(sep_rows, height)
    col_spans = _room_spans(sep_cols, width)
    return grid, row_spans, col_spans


def _valid_arrow_rooms(direction: str, n_rows: int, n_cols: int) -> list[tuple[int, int]]:
    out = []
    for ri in range(n_rows):
        for ci in range(n_cols):
            if direction == "up" and ri == 0:
                continue
            if direction == "down" and ri == n_rows - 1:
                continue
            if direction == "left" and ci == 0:
                continue
            if direction == "right" and ci == n_cols - 1:
                continue
            out.append((int(ri), int(ci)))
    return out


def _gen_attempt() -> np.ndarray | None:
    n_rows, n_cols, room_h, room_w = random.choice(LAYOUT_CHOICES)
    grid, row_spans, col_spans = _draw_layout(n_rows, n_cols, room_h, room_w)

    direction = random.choice(("up", "down", "left", "right"))
    arrow_room = random.choice(_valid_arrow_rooms(direction, n_rows, n_cols))
    _write_room(grid, row_spans, col_spans, arrow_room, _make_arrow_room(room_h, room_w, direction))

    all_rooms = [(ri, ci) for ri in range(n_rows) for ci in range(n_cols) if (ri, ci) != arrow_room]
    chain_tail = _chain_rooms(direction, arrow_room, n_rows, n_cols)[1:]
    selected: set[tuple[int, int]] = set()
    if chain_tail:
        selected.add(random.choice(chain_tail))
    outside_chain = [room for room in all_rooms if room not in chain_tail]
    if outside_chain:
        selected.add(random.choice(outside_chain))

    max_patterns = min(len(all_rooms), 2 + len(all_rooms) // 2)
    target_count = random.randint(max(2, len(selected)), max(len(selected), max_patterns))
    remaining = [room for room in all_rooms if room not in selected]
    random.shuffle(remaining)
    for room in remaining[: max(0, target_count - len(selected))]:
        selected.add(room)

    for room in selected:
        _write_room(grid, row_spans, col_spans, room, _make_room_pattern(room_h, room_w))

    if _mode_color(grid) != BG:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(grid, out):
        return None
    if not np.all(out.shape == grid.shape):
        return None
    if not np.all(out[[i * (room_h + 1) for i in range(n_rows + 1)], :] == SEP):
        return None
    if not np.all(out[:, [i * (room_w + 1) for i in range(n_cols + 1)]] == SEP):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
