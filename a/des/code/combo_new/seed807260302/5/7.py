from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
RED = 2
GREY = 5
SPECIAL = {BLACK, RED, GREY}
PALETTE_COLORS = tuple(c for c in range(10) if c not in SPECIAL)
MAX_SIZE = 30
GENERATION_ATTEMPTS = 500


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or not bool(mask[sr, sc]):
                continue
            seen[sr, sc] = True
            q: deque[tuple[int, int]] = deque([(sr, sc)])
            cells: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc] and bool(mask[rr, cc]):
                        seen[rr, cc] = True
                        q.append((rr, cc))
            comps.append(np.array(cells, dtype=int))

    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _find_palette(g: np.ndarray) -> list[int]:
    non_special = np.array([[int(x) not in SPECIAL for x in row] for row in g], dtype=bool)
    coords = np.argwhere(non_special)

    if int(coords.shape[0]) == 6:
        rows = sorted({int(r) for r, _c in coords})
        cols = sorted({int(c) for _r, c in coords})
        if len(rows) == 1 and cols == list(range(cols[0], cols[0] + 6)):
            r = rows[0]
            vals = [int(g[r, c]) for c in cols]
            assert len(set(vals)) == 6, "palette colors must be distinct"
            return vals
        if len(cols) == 1 and rows == list(range(rows[0], rows[0] + 6)):
            c = cols[0]
            vals = [int(g[r, c]) for r in rows]
            assert len(set(vals)) == 6, "palette colors must be distinct"
            return vals

    h, w = g.shape
    candidates: list[tuple[int, int, list[int]]] = []
    for r in range(h):
        for c in range(w - 5):
            vals = [int(g[r, c + k]) for k in range(6)]
            if len(set(vals)) == 6 and all(v not in SPECIAL for v in vals):
                at_corner = int(r in (0, h - 1) and c in (0, w - 6))
                candidates.append((1 - at_corner, r * w + c, vals))
    for r in range(h - 5):
        for c in range(w):
            vals = [int(g[r + k, c]) for k in range(6)]
            if len(set(vals)) == 6 and all(v not in SPECIAL for v in vals):
                at_corner = int(c in (0, w - 1) and r in (0, h - 6))
                candidates.append((1 - at_corner, r * w + c, vals))

    assert candidates, "expected one 1x6 palette strip"
    candidates.sort(key=lambda item: (item[0], item[1]))
    return [int(v) for v in candidates[0][2]]


def _find_dice_frames(g: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for coords in _components(g == GREY):
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h != w or h < 3:
            continue

        comp_mask = np.zeros((h, w), dtype=bool)
        rel = coords - np.array([r0, c0])
        comp_mask[rel[:, 0], rel[:, 1]] = True

        perimeter = np.zeros((h, w), dtype=bool)
        perimeter[0, :] = True
        perimeter[-1, :] = True
        perimeter[:, 0] = True
        perimeter[:, -1] = True
        if not bool(np.array_equal(comp_mask, perimeter)):
            continue

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        allowed = (interior == BLACK) | (interior == RED)
        if not bool(np.all(allowed)):
            continue
        pip_count = int(np.sum(interior == RED))
        if not 1 <= pip_count <= 6:
            continue

        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "coords": coords,
                "pip_count": pip_count,
                "side": h,
            }
        )

    assert frames, "expected at least one grey die frame"
    sides = {int(frame["side"]) for frame in frames}
    assert len(sides) == 1, "all die frames must have the same size"
    frames.sort(key=lambda frame: (frame["bbox"][0], frame["bbox"][1]))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey die frame by using its red pip count as an index into the shared palette.

    Input structure:
    - The background is black.
    - Several separated, same-size hollow square frames are drawn in grey.
    - Each frame interior contains 1 to 6 red single-pixel pips in a standard die-face layout.
    - A 1x6 strip of six distinct non-black, non-grey, non-red colors sits in a corner as the palette for counts 1 through 6.

    Rule:
    - Count the red pips inside each grey frame.
    - Select the Nth palette color for a frame with N red pips.
    - Repaint only that frame's grey border pixels with the selected color.
    - Keep all red pips, black background cells, and palette-strip pixels unchanged.

    Key insight:
    - The red pips are not decoration; their die-face count is the ordinal lookup key into the small corner palette.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    palette = _find_palette(g)
    frames = _find_dice_frames(g)

    out = g.copy()
    for frame in frames:
        color = int(palette[int(frame["pip_count"]) - 1])
        coords = frame["coords"]
        out[coords[:, 0], coords[:, 1]] = color
    return out


def _positive_partition(total: int, parts: int, minimum: int) -> list[int]:
    assert total >= parts * minimum
    values = [minimum] * parts
    for _ in range(total - parts * minimum):
        values[random.randrange(parts)] += 1
    return values


def _layout_options(count: int, side: int) -> list[tuple[int, int, int, int]]:
    options: list[tuple[int, int, int, int]] = []
    for rows in range(1, count + 1):
        for cols in range(1, count + 1):
            if rows * cols < count:
                continue
            min_h = rows * side + (rows + 1) * 2
            min_w = cols * side + (cols + 1) * 2
            if min_h <= MAX_SIZE and min_w <= MAX_SIZE:
                spare_slots = rows * cols - count
                options.append((spare_slots, rows, cols, min_h, min_w))
    options.sort()
    return [(rows, cols, min_h, min_w) for _spare, rows, cols, min_h, min_w in options]


def _frame_positions(count: int, side: int) -> tuple[int, int, list[tuple[int, int]]] | None:
    options = _layout_options(count, side)
    if not options:
        return None

    rows, cols, min_h, min_w = random.choice(options[: min(len(options), 8)])
    height = random.randint(min_h, MAX_SIZE)
    width = random.randint(min_w, MAX_SIZE)
    row_gaps = _positive_partition(height - rows * side, rows + 1, minimum=2)
    col_gaps = _positive_partition(width - cols * side, cols + 1, minimum=2)

    row_starts: list[int] = []
    cur = row_gaps[0]
    for r in range(rows):
        row_starts.append(cur)
        cur += side + row_gaps[r + 1]

    col_starts: list[int] = []
    cur = col_gaps[0]
    for c in range(cols):
        col_starts.append(cur)
        cur += side + col_gaps[c + 1]

    slots = [(r0, c0) for r0 in row_starts for c0 in col_starts]
    random.shuffle(slots)
    return height, width, slots[:count]


def _pip_cells(count: int) -> list[tuple[int, int]]:
    diag = random.choice((0, 1))
    if count == 1:
        return [(1, 1)]
    if count == 2:
        return [(0, 0), (2, 2)] if diag == 0 else [(0, 2), (2, 0)]
    if count == 3:
        return [(0, 0), (1, 1), (2, 2)] if diag == 0 else [(0, 2), (1, 1), (2, 0)]
    if count == 4:
        return [(0, 0), (0, 2), (2, 0), (2, 2)]
    if count == 5:
        return [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]
    assert count == 6
    return [(0, 0), (1, 0), (2, 0), (0, 2), (1, 2), (2, 2)]


def _draw_die(g: np.ndarray, r0: int, c0: int, side: int, count: int) -> None:
    r1 = r0 + side - 1
    c1 = c0 + side - 1
    g[r0, c0 : c1 + 1] = GREY
    g[r1, c0 : c1 + 1] = GREY
    g[r0 : r1 + 1, c0] = GREY
    g[r0 : r1 + 1, c1] = GREY

    marks = [1, side // 2, side - 2]
    for pr, pc in _pip_cells(count):
        g[r0 + marks[pr], c0 + marks[pc]] = RED


def _draw_palette(g: np.ndarray, palette: list[int]) -> None:
    h, w = g.shape
    corner = random.choice(("tl", "tr", "bl", "br"))
    row = 0 if corner[0] == "t" else h - 1
    col = 0 if corner[1] == "l" else w - 6
    g[row, col : col + 6] = np.array(palette, dtype=int)


def _expected_output(inp: np.ndarray, palette: list[int]) -> np.ndarray:
    out = inp.copy()
    for frame in _find_dice_frames(inp):
        color = int(palette[int(frame["pip_count"]) - 1])
        coords = frame["coords"]
        out[coords[:, 0], coords[:, 1]] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    side = random.choice((5, 7))
    count = random.randint(6, 9)
    layout = _frame_positions(count, side)
    if layout is None:
        return None
    height, width, positions = layout

    palette = random.sample(PALETTE_COLORS, 6)
    dice_counts = list(range(1, 7))
    dice_counts.extend(random.randint(1, 6) for _ in range(count - 6))
    random.shuffle(dice_counts)

    grid = np.full((height, width), BLACK, dtype=int)
    _draw_palette(grid, palette)

    for (r0, c0), pip_count in zip(positions, dice_counts, strict=True):
        _draw_die(grid, r0, c0, side, pip_count)

    try:
        expected = _expected_output(grid, palette)
        actual = main(grid)
    except Exception:
        return None
    if actual.shape != grid.shape or max(actual.shape) > MAX_SIZE:
        return None
    if not bool(np.array_equal(actual, expected)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate valid input")
