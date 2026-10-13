from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

SIDE_TO_DIR = {
    "top": "right",
    "right": "down",
    "bottom": "left",
    "left": "up",
}

OPPOSITE_DIR = {
    "right": "left",
    "left": "right",
    "up": "down",
    "down": "up",
}

GEN_ROWS_RANGE = (15, 20)
GEN_COLS_RANGE = (15, 20)
GEN_FRAME_COUNTS = (2, 3, 4)
GEN_FRAME_SIDE_RANGE = (3, 7)
GEN_MAX_ATTEMPTS = 500


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _marker_side(grid: np.ndarray) -> str:
    blue = grid == BLUE
    sides: list[str] = []
    if np.all(grid[0, :] == BLUE):
        sides.append("top")
    if np.all(grid[:, -1] == BLUE):
        sides.append("right")
    if np.all(grid[-1, :] == BLUE):
        sides.append("bottom")
    if np.all(grid[:, 0] == BLUE):
        sides.append("left")
    assert len(sides) == 1, "expected exactly one full-edge blue marker"
    side = sides[0]
    expected = np.zeros_like(blue, dtype=bool)
    if side == "top":
        expected[0, :] = True
    elif side == "right":
        expected[:, -1] = True
    elif side == "bottom":
        expected[-1, :] = True
    else:
        expected[:, 0] = True
    assert np.array_equal(blue, expected), "blue pixels must be exactly the marker line"
    return side


def _frame_bboxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for coords in _components(grid == RED):
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0 = int(r0)
        c0 = int(c0)
        r1 = int(r1)
        c1 = int(c1)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        mask = sub == RED
        hh, ww = mask.shape
        assert hh >= 3 and ww >= 3, "red objects must be frames with non-empty strict interior"
        border = np.zeros_like(mask, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        assert np.array_equal(mask, border), "every red component must be a 1-cell-thick hollow rectangle"
        assert np.all(sub[1:-1, 1:-1] == BLACK), "frame interiors must be black"
        frames.append((r0, c0, r1, c1))
    assert 2 <= len(frames) <= 4, "expected 2 to 4 red frames"
    return frames


def _filled_grid(grid: np.ndarray) -> tuple[np.ndarray, str, list[tuple[int, int, int, int]]]:
    side = _marker_side(grid)
    frames = _frame_bboxes(grid)
    out = np.full_like(grid, BLACK)
    for r0, c0, r1, c1 in frames:
        out[r0 + 1 : r1, c0 + 1 : c1] = GREEN
    return out, side, frames


def _apply_gravity(grid: np.ndarray, direction: str) -> np.ndarray:
    out = np.array(grid, copy=True)
    greens = np.argwhere(out == GREEN)
    if direction == "down":
        order = sorted((tuple(map(int, rc)) for rc in greens), key=lambda rc: (-rc[0], rc[1]))
        delta = (1, 0)
    elif direction == "up":
        order = sorted((tuple(map(int, rc)) for rc in greens), key=lambda rc: (rc[0], rc[1]))
        delta = (-1, 0)
    elif direction == "right":
        order = sorted((tuple(map(int, rc)) for rc in greens), key=lambda rc: (-rc[1], rc[0]))
        delta = (0, 1)
    elif direction == "left":
        order = sorted((tuple(map(int, rc)) for rc in greens), key=lambda rc: (rc[1], rc[0]))
        delta = (0, -1)
    else:
        raise AssertionError(direction)

    dr, dc = delta
    h, w = out.shape
    for r, c in order:
        if out[r, c] != GREEN:
            continue
        rr, cc = r, c
        while True:
            nr = rr + dr
            nc = cc + dc
            if not (0 <= nr < h and 0 <= nc < w) or out[nr, nc] != BLACK:
                break
            rr, cc = nr, nc
        if (rr, cc) != (r, c):
            out[r, c] = BLACK
            out[rr, cc] = GREEN
    return out


def _line_counts(grid: np.ndarray, direction: str) -> np.ndarray:
    mask = grid == GREEN
    return mask.sum(axis=1 if direction in ("left", "right") else 0)


def _interior_sizes(frames: list[tuple[int, int, int, int]]) -> tuple[list[int], list[int], list[int]]:
    heights = [r1 - r0 - 1 for r0, _c0, r1, _c1 in frames]
    widths = [c1 - c0 - 1 for _r0, c0, _r1, c1 in frames]
    areas = [h * w for h, w in zip(heights, widths)]
    return heights, widths, areas


def _has_overlap_on_gravity_axis(frames: list[tuple[int, int, int, int]], direction: str) -> bool:
    if direction in ("left", "right"):
        spans = [(r0 + 1, r1 - 1) for r0, _c0, r1, _c1 in frames]
    else:
        spans = [(c0 + 1, c1 - 1) for _r0, c0, _r1, c1 in frames]
    for i, (a0, a1) in enumerate(spans):
        for b0, b1 in spans[i + 1 :]:
            if max(a0, b0) <= min(a1, b1):
                return True
    return False


def _validate_generated(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    h, w = grid.shape
    assert GEN_ROWS_RANGE[0] <= h <= GEN_ROWS_RANGE[1]
    assert GEN_COLS_RANGE[0] <= w <= GEN_COLS_RANGE[1]
    assert set(map(int, np.unique(grid))) <= {BLACK, BLUE, RED}

    filled, side, frames = _filled_grid(grid)
    direction = SIDE_TO_DIR[side]
    out = _apply_gravity(filled, direction)
    heights, widths, areas = _interior_sizes(frames)

    assert np.array_equal(out, main(grid))
    assert set(map(int, np.unique(out))) <= {BLACK, GREEN}
    assert np.any(out == GREEN), "filled area must be non-empty"
    assert not np.array_equal(out, filled), "gravity must have a visible effect"
    assert not np.array_equal(out, _apply_gravity(filled, OPPOSITE_DIR[direction])), "marker direction must matter"
    assert len(set(areas)) >= 2, "frame interiors should not all have the same area"
    assert max(areas) >= 4, "at least one frame should have a non-trivial interior block"
    assert sum(areas) <= (h * w) // 3, "output should stay visually sparse enough"
    assert _has_overlap_on_gravity_axis(frames, direction), "at least one pair must merge under gravity"

    counts = _line_counts(out, direction)
    nonzero = [int(v) for v in counts if int(v) > 0]
    assert len(nonzero) >= 2, "output should occupy multiple lines"
    assert len(set(nonzero)) >= 2, "output should not collapse to a uniform bar"
    if direction in ("left", "right"):
        assert max(nonzero) > max(widths), "some output row must merge material from multiple frames"
    else:
        assert max(nonzero) > max(heights), "some output column must merge material from multiple frames"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase red hollow frames, fill their interiors green, and gravity-pack the fill according to the blue edge marker.

    Input structure:
    - The grid has a black background, 2 to 4 red 1-cell-thick hollow rectangular frames strictly inside the border, and one full-edge blue marker line.
    - Each frame encloses a black rectangular strict interior; frames are disconnected and separated from each other.
    - The blue line occupies exactly one whole outer edge: top, right, bottom, or left.

    Rule:
    - Map the marker edge to gravity by clockwise border travel: top -> right, right -> down, bottom -> left, left -> up.
    - Erase every red frame border to black, fill each frame's strict interior with green, and erase the blue marker line.
    - Treat all green pixels as free material and slide them in the gravity direction until blocked by the grid edge or previously settled green pixels.
    - Return the final grid, which contains only black background and the gravity-compressed green pixels.

    Key insight:
    - The red frames are only carriers for hidden green rectangles; once the borders disappear, those filled interiors merge and stack under a direction that is encoded solely by the blue edge marker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"
    assert set(map(int, np.unique(g))) <= {BLACK, BLUE, RED}, "input should contain only black, blue, and red"

    filled, side, _frames = _filled_grid(g)
    return _apply_gravity(filled, SIDE_TO_DIR[side])


def _boxes_clear(box: tuple[int, int, int, int], placed: list[tuple[int, int, int, int]]) -> bool:
    r0, c0, r1, c1 = box
    for pr0, pc0, pr1, pc1 in placed:
        if not (r1 + 1 < pr0 or pr1 + 1 < r0 or c1 + 1 < pc0 or pc1 + 1 < c0):
            return False
    return True


def _interior_rows_overlap(r0a: int, h1: int, r0b: int, h2: int) -> bool:
    return max(r0a + 1, r0b + 1) <= min(r0a + h1 - 2, r0b + h2 - 2)


def _interior_cols_overlap(c0a: int, w1: int, c0b: int, w2: int) -> bool:
    return max(c0a + 1, c0b + 1) <= min(c0a + w1 - 2, c0b + w2 - 2)


def _sample_sizes(h: int, w: int, count: int, direction: str) -> list[tuple[int, int]] | None:
    for _ in range(120):
        sizes = [
            (
                random.randint(*GEN_FRAME_SIDE_RANGE),
                random.randint(*GEN_FRAME_SIDE_RANGE),
            )
            for _ in range(count)
        ]
        interior_areas = [(hh - 2) * (ww - 2) for hh, ww in sizes]
        if len(set(interior_areas)) < 2 or max(interior_areas) < 4:
            continue
        if sum(interior_areas) > (h * w) // 3:
            continue
        if direction in ("left", "right"):
            widths = sorted((ww for _hh, ww in sizes), reverse=True)
            if widths[0] + widths[1] + 3 > w:
                continue
        else:
            heights = sorted((hh for hh, _ww in sizes), reverse=True)
            if heights[0] + heights[1] + 3 > h:
                continue
        return sizes
    return None


def _place_forced_pair(
    h: int,
    w: int,
    size_a: tuple[int, int],
    size_b: tuple[int, int],
    direction: str,
) -> list[tuple[int, int, int, int]] | None:
    ha, wa = size_a
    hb, wb = size_b

    if direction in ("left", "right"):
        if wa + wb + 3 > w:
            return None
        left_first = random.choice([True, False])
        if left_first:
            (h1, w1), (h2, w2) = (ha, wa), (hb, wb)
            ids = (0, 1)
        else:
            (h1, w1), (h2, w2) = (hb, wb), (ha, wa)
            ids = (1, 0)

        c1 = random.randint(1, w - w1 - w2 - 2)
        c2 = random.randint(c1 + w1 + 1, w - w2 - 1)
        r1 = random.randint(1, h - h1 - 1)
        feasible_r2 = [r for r in range(1, h - h2) if _interior_rows_overlap(r1, h1, r, h2)]
        if not feasible_r2:
            return None
        r2 = random.choice(feasible_r2)

        boxes_lr = [(r1, c1, r1 + h1 - 1, c1 + w1 - 1), (r2, c2, r2 + h2 - 1, c2 + w2 - 1)]
        boxes = [boxes_lr[ids.index(i)] for i in range(2)]
    else:
        if ha + hb + 3 > h:
            return None
        top_first = random.choice([True, False])
        if top_first:
            (h1, w1), (h2, w2) = (ha, wa), (hb, wb)
            ids = (0, 1)
        else:
            (h1, w1), (h2, w2) = (hb, wb), (ha, wa)
            ids = (1, 0)

        r1 = random.randint(1, h - h1 - h2 - 2)
        r2 = random.randint(r1 + h1 + 1, h - h2 - 1)
        c1 = random.randint(1, w - w1 - 1)
        feasible_c2 = [c for c in range(1, w - w2) if _interior_cols_overlap(c1, w1, c, w2)]
        feasible_c2 = [c for c in feasible_c2 if not (c + w2 - 1 + 1 < c1 or c1 + w1 - 1 + 1 < c)]
        if not feasible_c2:
            feasible_c2 = [c for c in range(1, w - w2) if _interior_cols_overlap(c1, w1, c, w2)]
        if not feasible_c2:
            return None
        c2 = random.choice(feasible_c2)

        boxes_tb = [(r1, c1, r1 + h1 - 1, c1 + w1 - 1), (r2, c2, r2 + h2 - 1, c2 + w2 - 1)]
        boxes = [boxes_tb[ids.index(i)] for i in range(2)]

    if not _boxes_clear(boxes[0], [boxes[1]]) or not _boxes_clear(boxes[1], [boxes[0]]):
        return None
    return boxes


def _place_remaining_boxes(
    h: int,
    w: int,
    sizes: list[tuple[int, int]],
    placed: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]] | None:
    boxes = placed[:]
    remaining = list(enumerate(sizes[len(placed) :], start=len(placed)))
    remaining.sort(key=lambda item: item[1][0] * item[1][1], reverse=True)

    for _idx, (hh, ww) in remaining:
        candidates = [
            (r0, c0, r0 + hh - 1, c0 + ww - 1)
            for r0 in range(1, h - hh)
            for c0 in range(1, w - ww)
        ]
        random.shuffle(candidates)
        chosen = None
        for box in candidates:
            if _boxes_clear(box, boxes):
                chosen = box
                break
        if chosen is None:
            return None
        boxes.append(chosen)
    return boxes


def _draw_input(h: int, w: int, side: str, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)
    for r0, c0, r1, c1 in boxes:
        grid[r0, c0 : c1 + 1] = RED
        grid[r1, c0 : c1 + 1] = RED
        grid[r0 : r1 + 1, c0] = RED
        grid[r0 : r1 + 1, c1] = RED
    if side == "top":
        grid[0, :] = BLUE
    elif side == "right":
        grid[:, -1] = BLUE
    elif side == "bottom":
        grid[-1, :] = BLUE
    else:
        grid[:, 0] = BLUE
    return grid


def _attempt_generate() -> np.ndarray | None:
    try:
        h = random.randint(*GEN_ROWS_RANGE)
        w = random.randint(*GEN_COLS_RANGE)
        side = random.choice(["top", "right", "bottom", "left"])
        direction = SIDE_TO_DIR[side]
        count = random.choice(GEN_FRAME_COUNTS)
        sizes = _sample_sizes(h, w, count, direction)
        if sizes is None:
            return None
        if direction in ("left", "right"):
            sizes = sorted(sizes, key=lambda hw: (hw[1] - 2, hw[0] - 2), reverse=True)
        else:
            sizes = sorted(sizes, key=lambda hw: (hw[0] - 2, hw[1] - 2), reverse=True)

        forced = _place_forced_pair(h, w, sizes[0], sizes[1], direction)
        if forced is None:
            return None
        boxes = _place_remaining_boxes(h, w, sizes, forced)
        if boxes is None:
            return None

        grid = _draw_input(h, w, side, boxes)
        _validate_generated(grid)
        return grid
    except AssertionError:
        return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
