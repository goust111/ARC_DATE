from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
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
            stack = [(sr, sc)]
            cells: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _find_blue_frames(g: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []

    for coords in _components(g == BLUE):
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue

        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        perimeter = np.zeros((h, w), dtype=bool)
        perimeter[0, :] = True
        perimeter[-1, :] = True
        perimeter[:, 0] = True
        perimeter[:, -1] = True

        comp_mask = np.zeros((h, w), dtype=bool)
        rel = coords - np.array([r0, c0])
        comp_mask[rel[:, 0], rel[:, 1]] = True

        if not bool(np.array_equal(comp_mask, perimeter)):
            continue
        if not bool(np.all(patch[perimeter] == BLUE)):
            continue
        if bool(np.any(patch[1:-1, 1:-1] == BLUE)):
            continue

        frames.append((r0, c0, r1, c1))

    frames.sort(key=lambda b: (b[0], b[1]))
    assert frames, "expected at least one blue rectangular frame"

    heights = {r1 - r0 + 1 for r0, _c0, r1, _c1 in frames}
    widths = {c1 - c0 + 1 for _r0, c0, _r1, c1 in frames}
    assert len(heights) == 1 and len(widths) == 1, "all frames must share one size"
    return frames


def _template_frame(g: np.ndarray, frames: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    templates = []
    for frame in frames:
        r0, c0, r1, c1 = frame
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        if bool(np.any(interior == RED)):
            templates.append(frame)
    assert len(templates) == 1, "expected exactly one frame with a red seed pattern"
    return templates[0]


def _partition_for_thumbnail(inner_h: int, inner_w: int, count: int) -> tuple[int, int, int, int]:
    if inner_h * inner_w == count:
        return inner_h, inner_w, 1, 1

    candidates: list[tuple[int, int, int, int]] = []
    for rows in range(1, count + 1):
        if count % rows != 0:
            continue
        cols = count // rows
        if inner_h % rows == 0 and inner_w % cols == 0:
            cell_h = inner_h // rows
            cell_w = inner_w // cols
            candidates.append((rows, cols, cell_h, cell_w))

    assert candidates, "template interior must divide into one subcell per frame"

    def score(item: tuple[int, int, int, int]) -> tuple[int, int, int]:
        rows, cols, cell_h, cell_w = item
        return (abs(cell_h - cell_w), abs(rows - cols), cell_h * cell_w)

    return min(candidates, key=score)


def _active_indices(g: np.ndarray, template: tuple[int, int, int, int], count: int) -> list[int]:
    r0, c0, r1, c1 = template
    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    inner_h, inner_w = interior.shape
    thumb_h, thumb_w, cell_h, cell_w = _partition_for_thumbnail(inner_h, inner_w, count)
    assert thumb_h * thumb_w == count, "thumbnail cell count must match frame count"

    active: list[int] = []
    for idx in range(count):
        tr = idx // thumb_w
        tc = idx % thumb_w
        block = interior[tr * cell_h : (tr + 1) * cell_h, tc * cell_w : (tc + 1) * cell_w]
        if bool(np.any(block == RED)):
            active.append(idx)
    assert active, "seed pattern must activate at least one frame"
    return active


def _paint_selected_frame(out: np.ndarray, frame: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = frame
    out[r0, c0 : c1 + 1] = GREEN
    out[r1, c0 : c1 + 1] = GREEN
    out[r0 : r1 + 1, c0] = GREEN
    out[r0 : r1 + 1, c1] = GREEN
    out[r0 + 1 : r1, c0 + 1 : c1] = RED


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the red seed pattern inside one blue frame as an address table selecting which same-shaped frames to fill.

    Input structure:
    - The background is black.
    - The grid contains several separated, same-size, hollow rectangular blue frames.
    - Exactly one frame contains a red pattern inside its otherwise black interior; every other frame interior is black.

    Rule:
    - Sort all blue frames by their top-left position in row-major order.
    - Read the red pattern inside the template frame as a binary thumbnail whose equal subcells match the number of frames.
    - Each activated thumbnail subcell selects the frame with the same row-major index; selected frames become green-bordered and red-filled.
    - Unselected frames, including the template frame when its own address is off, remain unchanged.

    Key insight:
    - The red pattern is not copied as a drawing; it is a compact address map from local template cells to globally ordered frames.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = _find_blue_frames(g)
    template = _template_frame(g, frames)
    active = _active_indices(g, template, len(frames))

    out = g.copy()
    for idx in active:
        _paint_selected_frame(out, frames[int(idx)])
    return out


def _draw_frame(grid: np.ndarray, r0: int, c0: int, height: int, width: int, color: int = BLUE) -> None:
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _positive_partition(total: int, parts: int, minimum: int = 1) -> list[int]:
    assert total >= parts * minimum
    values = [minimum] * parts
    for _ in range(total - parts * minimum):
        values[random.randrange(parts)] += 1
    return values


def _layout_options(count: int, frame_h: int, frame_w: int) -> list[tuple[int, int, int, int]]:
    options: list[tuple[int, int, int, int]] = []
    for rows in range(1, count + 1):
        cols = (count + rows - 1) // rows
        min_h = rows * frame_h + rows + 1
        min_w = cols * frame_w + cols + 1
        if min_h <= MAX_SIZE and min_w <= MAX_SIZE:
            options.append((rows, cols, min_h, min_w))
    return options


def _thumbnail_options() -> list[tuple[int, int, int, int]]:
    options: list[tuple[int, int, int, int]] = []
    for thumb_h, thumb_w in ((2, 2), (2, 3), (3, 2), (2, 4), (4, 2), (3, 3), (2, 5), (5, 2), (3, 4), (4, 3)):
        for cell_size in (1, 2):
            frame_h = thumb_h * cell_size + 2
            frame_w = thumb_w * cell_size + 2
            count = thumb_h * thumb_w
            if _layout_options(count, frame_h, frame_w):
                options.append((thumb_h, thumb_w, cell_size, cell_size))
    return options


def _frame_positions(count: int, frame_h: int, frame_w: int) -> tuple[int, int, list[tuple[int, int]]]:
    rows, cols, min_h, min_w = random.choice(_layout_options(count, frame_h, frame_w))
    height = random.randint(min_h, MAX_SIZE)
    width = random.randint(min_w, MAX_SIZE)

    row_gaps = _positive_partition(height - rows * frame_h, rows + 1, minimum=1)
    col_gaps = _positive_partition(width - cols * frame_w, cols + 1, minimum=1)

    row_starts: list[int] = []
    cur = row_gaps[0]
    for gr in range(rows):
        row_starts.append(cur)
        cur += frame_h + row_gaps[gr + 1]

    col_starts: list[int] = []
    cur = col_gaps[0]
    for gc in range(cols):
        col_starts.append(cur)
        cur += frame_w + col_gaps[gc + 1]

    positions = []
    for idx in range(count):
        gr = idx // cols
        gc = idx % cols
        positions.append((row_starts[gr], col_starts[gc]))
    return height, width, positions


def _sample_active_indices(count: int, template_idx: int) -> set[int]:
    template_active = random.choice((True, False))
    if template_active:
        active_count = random.randint(1, count)
        active = {int(template_idx)}
        remaining = [idx for idx in range(count) if idx != template_idx]
        active.update(random.sample(remaining, active_count - 1))
        return active

    active_count = random.randint(1, count - 1)
    return set(random.sample([idx for idx in range(count) if idx != template_idx], active_count))


def _render_expected(inp: np.ndarray, frames: list[tuple[int, int, int, int]], active: set[int]) -> np.ndarray:
    out = inp.copy()
    for idx in sorted(active):
        _paint_selected_frame(out, frames[int(idx)])
    return out


def _gen_attempt() -> np.ndarray | None:
    thumb_h, thumb_w, cell_h, cell_w = random.choice(_thumbnail_options())
    count = thumb_h * thumb_w
    frame_h = thumb_h * cell_h + 2
    frame_w = thumb_w * cell_w + 2

    height, width, positions = _frame_positions(count, frame_h, frame_w)
    grid = np.full((height, width), BLACK, dtype=int)

    for r0, c0 in positions:
        _draw_frame(grid, r0, c0, frame_h, frame_w, BLUE)

    frames = [(r0, c0, r0 + frame_h - 1, c0 + frame_w - 1) for r0, c0 in positions]
    frames.sort(key=lambda b: (b[0], b[1]))

    template_idx = random.randrange(count)
    active = _sample_active_indices(count, template_idx)

    tr0, tc0, _tr1, _tc1 = frames[template_idx]
    for idx in sorted(active):
        thumb_r = idx // thumb_w
        thumb_c = idx % thumb_w
        rr = tr0 + 1 + thumb_r * cell_h + random.randrange(cell_h)
        cc = tc0 + 1 + thumb_c * cell_w + random.randrange(cell_w)
        grid[rr, cc] = RED
        if cell_h * cell_w > 1 and random.random() < 0.35:
            block_cells = [
                (tr0 + 1 + thumb_r * cell_h + dr, tc0 + 1 + thumb_c * cell_w + dc)
                for dr in range(cell_h)
                for dc in range(cell_w)
            ]
            for ar, ac in random.sample(block_cells, random.randint(1, len(block_cells))):
                grid[ar, ac] = RED

    expected = _render_expected(grid, frames, active)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid seed807260302/5/1 input")
