from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BG = 0
WALL = 5
PALETTE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

CELL_ROWS_RANGE = (2, 5)
CELL_COLS_RANGE = (2, 5)
CELL_SIZE_RANGE = (2, 4)
ROOM_COUNT_MAX = 6
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            q = deque([(r, c)])
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_boundary(comp: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)


def _find_doors(body: np.ndarray) -> list[tuple[int, int]]:
    h, w = body.shape
    doors: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(body[r, c]) != BG:
                continue
            vertical_gap = (
                int(body[r - 1, c]) == WALL
                and int(body[r + 1, c]) == WALL
                and int(body[r, c - 1]) == BG
                and int(body[r, c + 1]) == BG
            )
            horizontal_gap = (
                int(body[r, c - 1]) == WALL
                and int(body[r, c + 1]) == WALL
                and int(body[r - 1, c]) == BG
                and int(body[r + 1, c]) == BG
            )
            if vertical_gap ^ horizontal_gap:
                doors.append((r, c))
    return doors


def _analyze_rooms(body: np.ndarray) -> tuple[list[tuple[list[tuple[int, int]], int]], list[tuple[int, int]]]:
    doors = _find_doors(body)
    door_set = set(doors)
    sealed = np.array(body, copy=True)
    for r, c in doors:
        sealed[r, c] = WALL

    room_info: list[tuple[list[tuple[int, int]], int]] = []
    for comp in _components(sealed == BG):
        if _touches_boundary(comp, *sealed.shape):
            continue
        touching_doors: set[tuple[int, int]] = set()
        for r, c in comp:
            for dr, dc in DIR4:
                pos = (r + dr, c + dc)
                if pos in door_set:
                    touching_doors.add(pos)
        room_info.append((comp, len(touching_doors)))
    return room_info, doors


def _read_palette(row: np.ndarray) -> list[int]:
    cols = [int(c) for c in np.flatnonzero(row != BG)]
    assert cols, "expected a non-empty palette strip on the bottom row"
    assert all(cols[i + 1] == cols[i] + 1 for i in range(len(cols) - 1)), "palette strip must be contiguous"
    return [int(row[c]) for c in cols]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each walled room by counting its door gaps and looking up the matching palette color.

    Input structure:
    - The background is black, and thin gray walls partition the area above the last row.
    - Some wall segments contain single-pixel black gaps that act as doors between rooms.
    - The bottom row is a contiguous palette strip whose 1st, 2nd, 3rd, ... colors encode
      the fill color for rooms with 1, 2, 3, ... doors.

    Rule:
    - Ignore the bottom palette row and detect every single-pixel door gap in the gray walls.
    - Temporarily close those gaps to recover the enclosed black room interiors.
    - For each enclosed room, count how many detected door pixels touch its boundary, then fill
      the room with the palette color at that 1-based index.
    - Keep all gray wall pixels and the bottom palette row unchanged.

    Key insight:
    - The black door pixels are not ordinary room interior: they are isolated missing wall pixels,
      so the solver must first reconstruct the walls before counting doors and recoloring rooms.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 3, "expected a 2D grid with a palette row"

    palette = _read_palette(g[-1])
    body = g[:-1]
    room_info, _doors = _analyze_rooms(body)

    out = np.array(g, copy=True)
    for comp, door_count in room_info:
        assert 1 <= door_count <= len(palette), "room door count must index the palette"
        color = palette[door_count - 1]
        for r, c in comp:
            out[r, c] = color
    return out


def _neighbors_cell(r: int, c: int, rows: int, cols: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            out.append((nr, nc))
    return out


def _grow_partition(rows: int, cols: int, room_count: int) -> np.ndarray:
    labels = -np.ones((rows, cols), dtype=int)
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    seeds = random.sample(cells, room_count)
    frontier: list[tuple[int, int]] = []
    for room_id, (r, c) in enumerate(seeds):
        labels[r, c] = room_id
        frontier.extend(_neighbors_cell(r, c, rows, cols))

    remaining = rows * cols - room_count
    while remaining:
        idx = random.randrange(len(frontier))
        r, c = frontier.pop(idx)
        if int(labels[r, c]) != -1:
            continue
        neighbor_labels = {
            int(labels[nr, nc])
            for nr, nc in _neighbors_cell(r, c, rows, cols)
            if int(labels[nr, nc]) != -1
        }
        if not neighbor_labels:
            frontier.append((r, c))
            continue
        labels[r, c] = random.choice(sorted(neighbor_labels))
        remaining -= 1
        for nr, nc in _neighbors_cell(r, c, rows, cols):
            if int(labels[nr, nc]) == -1:
                frontier.append((nr, nc))
    return labels


class _DSU:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        self.parent[rb] = ra
        return True


def _cell_start(idx: int, cell_size: int) -> int:
    return 1 + idx * (cell_size + 1)


def _boundary_segments(labels: np.ndarray, cell_h: int, cell_w: int) -> list[dict[str, object]]:
    rows, cols = labels.shape
    segments: list[dict[str, object]] = []

    for r in range(rows):
        for c in range(cols - 1):
            a = int(labels[r, c])
            b = int(labels[r, c + 1])
            if a == b:
                continue
            r0 = _cell_start(r, cell_h)
            wall_c = (c + 1) * (cell_w + 1)
            coords = [(r0 + dr, wall_c) for dr in range(cell_h)]
            segments.append({"rooms": tuple(sorted((a, b))), "coords": coords})

    for r in range(rows - 1):
        for c in range(cols):
            a = int(labels[r, c])
            b = int(labels[r + 1, c])
            if a == b:
                continue
            wall_r = (r + 1) * (cell_h + 1)
            c0 = _cell_start(c, cell_w)
            coords = [(wall_r, c0 + dc) for dc in range(cell_w)]
            segments.append({"rooms": tuple(sorted((a, b))), "coords": coords})

    return segments


def _select_segments(
    segments: list[dict[str, object]],
    room_count: int,
    max_palette_len: int,
) -> tuple[set[int], list[int]] | None:
    pair_to_ids: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, seg in enumerate(segments):
        pair_to_ids[tuple(seg["rooms"])].append(idx)

    pair_keys = list(pair_to_ids)
    random.shuffle(pair_keys)
    dsu = _DSU(room_count)
    selected: set[int] = set()
    for a, b in pair_keys:
        if dsu.union(a, b):
            selected.add(random.choice(pair_to_ids[(a, b)]))

    if len(selected) != room_count - 1:
        return None

    remaining = [idx for idx in range(len(segments)) if idx not in selected]
    random.shuffle(remaining)
    extra_count = random.randint(0, min(len(remaining), room_count + 1))
    for idx in remaining[:extra_count]:
        selected.add(idx)

    counts = [0] * room_count
    for idx in selected:
        a, b = tuple(segments[idx]["rooms"])
        counts[a] += 1
        counts[b] += 1

    if min(counts) < 1 or max(counts) > max_palette_len:
        return None
    return selected, counts


def _render_body(labels: np.ndarray, cell_h: int, cell_w: int, door_pixels: set[tuple[int, int]]) -> np.ndarray:
    rows, cols = labels.shape
    h = rows * cell_h + rows + 1
    w = cols * cell_w + cols + 1
    body = np.full((h, w), WALL, dtype=int)

    for r in range(rows):
        for c in range(cols):
            r0 = _cell_start(r, cell_h)
            c0 = _cell_start(c, cell_w)
            body[r0 : r0 + cell_h, c0 : c0 + cell_w] = BG

    for r in range(rows):
        for c in range(cols - 1):
            if int(labels[r, c]) != int(labels[r, c + 1]):
                continue
            r0 = _cell_start(r, cell_h)
            wall_c = (c + 1) * (cell_w + 1)
            body[r0 : r0 + cell_h, wall_c] = BG

    for r in range(rows - 1):
        for c in range(cols):
            if int(labels[r, c]) != int(labels[r + 1, c]):
                continue
            wall_r = (r + 1) * (cell_h + 1)
            c0 = _cell_start(c, cell_w)
            body[wall_r, c0 : c0 + cell_w] = BG

    for r, c in door_pixels:
        body[r, c] = BG
    return body


def _generate_attempt() -> np.ndarray | None:
    cell_rows = random.randint(*CELL_ROWS_RANGE)
    cell_cols = random.randint(*CELL_COLS_RANGE)
    cell_h = random.randint(*CELL_SIZE_RANGE)
    cell_w = random.randint(*CELL_SIZE_RANGE)

    body_h = cell_rows * cell_h + cell_rows + 1
    body_w = cell_cols * cell_w + cell_cols + 1
    if body_h + 1 > 30 or body_w > 30:
        return None

    room_cap = min(ROOM_COUNT_MAX, cell_rows * cell_cols)
    room_count = random.randint(2, room_cap)
    labels = _grow_partition(cell_rows, cell_cols, room_count)
    segments = _boundary_segments(labels, cell_h, cell_w)
    if not segments:
        return None

    selected = _select_segments(segments, room_count, min(len(PALETTE_COLORS), body_w))
    if selected is None:
        return None
    selected_ids, expected_counts = selected

    door_pixels = {
        random.choice(list(segments[idx]["coords"]))
        for idx in selected_ids
    }
    if len(door_pixels) != len(selected_ids):
        return None

    body = _render_body(labels, cell_h, cell_w, door_pixels)
    room_info, detected_doors = _analyze_rooms(body)
    found_counts = sorted(count for _comp, count in room_info)
    if set(detected_doors) != door_pixels:
        return None
    if len(room_info) != room_count:
        return None
    if found_counts != sorted(expected_counts):
        return None

    palette_len = max(found_counts)
    palette = random.sample(PALETTE_COLORS, palette_len)
    palette_row = np.full((1, body_w), BG, dtype=int)
    start = random.randint(0, body_w - palette_len)
    palette_row[0, start : start + palette_len] = np.array(palette, dtype=int)

    grid = np.vstack([body, palette_row])
    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-and-door puzzle")
