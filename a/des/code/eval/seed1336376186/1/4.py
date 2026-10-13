import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
PATTERN_COLORS = (3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _extract_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for comp in _components(grid == GREY):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r0 <= 0 or c0 <= 0 or r1 >= grid.shape[0] - 1 or c1 >= grid.shape[1] - 1:
            continue
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        box = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros_like(box, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not np.all(box[border] == GREY):
            continue
        if np.any(box[~border] == GREY):
            continue
        if len(comp) != int(border.sum()):
            continue
        frames.append((r0, r1, c0, c1))
    frames.sort()
    return frames


def _read_marker(segment: np.ndarray) -> int | None:
    colors = {int(v) for v in segment.ravel() if int(v) not in (BLACK, GREY)}
    if not colors:
        return None
    if len(colors) != 1:
        raise ValueError(f"ambiguous marker segment: {colors}")
    return next(iter(colors))


def _transform_region(region: np.ndarray, up: int | None, left: int | None) -> np.ndarray:
    if up is None and left is None:
        out = region.copy()
    elif up == RED and left is None:
        out = np.fliplr(region)
    elif up is None and left == RED:
        out = np.flipud(region)
    elif up == RED and left == RED:
        out = np.rot90(region, 2)
    elif up == BLUE and left is None:
        out = np.rot90(region, -1)
    elif up is None and left == BLUE:
        out = np.rot90(region, 1)
    else:
        raise ValueError(f"unsupported instruction pair up={up}, left={left}")
    if out.shape != region.shape:
        raise ValueError("90-degree rotations require square regions")
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read border-encoded instructions and transform each grey-framed enclosed region.

    Input structure:
    - Background is black.
    - Grey one-pixel rectangular frames define enclosed regions.
    - The top border encodes a per-column-band instruction color; the left border encodes
      a per-row-band instruction color.

    Rule:
    - For each enclosed region, read U from the top border segment above its columns and
      L from the left border segment beside its rows.
    - Apply the transformation determined by (U, L) to the full region interior.
    - Keep grey frames and the outer border unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    for r0, r1, c0, c1 in _extract_frames(g):
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        up = _read_marker(g[0, c0 + 1 : c1])
        left = _read_marker(g[r0 + 1 : r1, 0])
        out[r0 + 1 : r1, c0 + 1 : c1] = _transform_region(inner, up, left)
    return out


def _valid_pair(up: int | None, left: int | None) -> bool:
    return (up, left) in {
        (None, None),
        (RED, None),
        (None, RED),
        (RED, RED),
        (BLUE, None),
        (None, BLUE),
    }


def _neighbor_cells(cell: tuple[int, int], size: int) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < size and 0 <= nc < size:
            out.append((nr, nc))
    return out


def _make_pattern(size: int, rng: random.Random) -> np.ndarray:
    cells_all = [(r, c) for r in range(size) for c in range(size)]
    max_area = min(size * size - 1, size + 4)
    for _ in range(200):
        area = rng.randint(3, max_area)
        start = (rng.randrange(size), rng.randrange(size))
        cells = {start}
        frontier = set(_neighbor_cells(start, size))
        while len(cells) < area and frontier:
            cell = rng.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for nxt in _neighbor_cells(cell, size):
                if nxt not in cells:
                    frontier.add(nxt)
        if len(cells) < 3:
            continue

        arr = np.zeros((size, size), dtype=int)
        color_count = rng.randint(1, 3)
        colors = rng.sample(PATTERN_COLORS, color_count)
        for idx, (r, c) in enumerate(rng.sample(list(cells), len(cells))):
            if color_count == 1:
                arr[r, c] = colors[0]
            else:
                arr[r, c] = colors[idx % color_count]

        transforms = (
            np.fliplr(arr),
            np.flipud(arr),
            np.rot90(arr, 2),
            np.rot90(arr, 1),
            np.rot90(arr, -1),
        )
        if any(np.array_equal(arr, t) for t in transforms):
            continue
        if len({int(arr[r, c]) for r, c in cells}) == 1 and len(cells) < 4:
            continue
        if all(r == next(iter(cells))[0] for r, _ in cells):
            continue
        if all(c == next(iter(cells))[1] for _, c in cells):
            continue
        return arr

    raise RuntimeError("failed to generate asymmetric pattern")


def _layout_bounds(starts: list[int], spans: list[int]) -> list[tuple[int, int]]:
    return [(start, start + span) for start, span in zip(starts, spans)]


def _compute_starts(spans: list[int], gaps: list[int]) -> list[int]:
    starts = [1]
    for i in range(1, len(spans)):
        starts.append(starts[-1] + spans[i - 1] + gaps[i - 1])
    return starts


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    row_labels = [None, RED, BLUE]
    col_labels = [None, RED, BLUE]
    rng.shuffle(row_labels)
    rng.shuffle(col_labels)

    row_spans = [rng.randint(7, 9) for _ in range(3)]
    col_spans = [rng.randint(7, 9) for _ in range(3)]
    row_gaps = [rng.randint(1, 2) for _ in range(2)]
    col_gaps = [rng.randint(1, 2) for _ in range(2)]

    row_starts = _compute_starts(row_spans, row_gaps)
    col_starts = _compute_starts(col_spans, col_gaps)
    h = row_starts[-1] + row_spans[-1] + 1
    w = col_starts[-1] + col_spans[-1] + 1
    if h > 30 or w > 30:
        return None

    g = np.zeros((h, w), dtype=int)

    for label, start, span in zip(row_labels, row_starts, row_spans):
        if label is not None:
            g[start : start + span, 0] = label
    for label, start, span in zip(col_labels, col_starts, col_spans):
        if label is not None:
            g[0, start : start + span] = label

    placed = 0
    for ri, (r0, r1) in enumerate(_layout_bounds(row_starts, row_spans)):
        for ci, (c0, c1) in enumerate(_layout_bounds(col_starts, col_spans)):
            left = row_labels[ri]
            up = col_labels[ci]
            if not _valid_pair(up, left):
                continue
            band_h = r1 - r0
            band_w = c1 - c0
            outer = rng.randint(5, min(band_h, band_w))
            room_r0 = r0 + rng.randint(0, band_h - outer)
            room_c0 = c0 + rng.randint(0, band_w - outer)
            room_r1 = room_r0 + outer - 1
            room_c1 = room_c0 + outer - 1

            g[room_r0 : room_r1 + 1, room_c0] = GREY
            g[room_r0 : room_r1 + 1, room_c1] = GREY
            g[room_r0, room_c0 : room_c1 + 1] = GREY
            g[room_r1, room_c0 : room_c1 + 1] = GREY

            pattern = _make_pattern(outer - 2, rng)
            g[room_r0 + 1 : room_r1, room_c0 + 1 : room_c1] = pattern
            placed += 1

    if placed != 6:
        return None
    if len(_extract_frames(g)) != 6:
        return None
    return g


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
