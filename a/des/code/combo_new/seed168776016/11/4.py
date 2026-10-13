from collections import Counter, deque
import random

import numpy as np


BLACK = 0
DIVIDER = 5
SHAPE = 8
PALETTE_COLORS = (1, 2, 3, 4, 6, 7, 9)

GRID_H_RANGE = (18, 25)
LEFT_W_RANGE = (10, 18)
RIGHT_W_RANGE = (4, 6)
SHAPE_COUNT_RANGE = (2, 4)
LEGEND_ROWS = 4
LEGEND_STEP = 3
LEGEND_HEIGHT = LEGEND_ROWS * 2 + (LEGEND_ROWS - 1)
SHAPE_RETRIES = 120
PLACEMENT_RETRIES = 250
GEN_RETRIES = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _enclosed_holes(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    holes: list[list[tuple[int, int]]] = []
    for comp in _components(~mask):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        holes.append(comp)
    return holes


def _find_divider(grid: np.ndarray) -> int:
    candidates = [
        c
        for c in range(grid.shape[1])
        if grid[0, c] != BLACK and np.all(grid[:, c] == grid[0, c])
    ]
    assert len(candidates) == 1, "expected exactly one full-height divider column"
    return int(candidates[0])


def _legend_entries(grid: np.ndarray, divider_c: int) -> list[tuple[int, int]]:
    right = grid[:, divider_c + 1 :]
    entries: list[tuple[int, int, int]] = []
    for comp in _components(right != BLACK):
        counts = Counter(int(right[r, c]) for r, c in comp)
        assert sorted(counts.values()) == [1, 4], "each legend row must be a 2x2 block plus a singleton"
        body = next(color for color, count in counts.items() if count == 4)
        hole = next(color for color, count in counts.items() if count == 1)
        top = min(r for r, _ in comp)
        entries.append((int(top), int(body), int(hole)))
    entries.sort()
    return [(body, hole) for _top, body, hole in entries]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each left-side closed shape from a right-side legend indexed by its hole count.

    Input structure:
    - The background is black, and a single full-height grey divider column splits the grid into left and right regions.
    - The left region contains several disconnected azure closed shapes, each possibly enclosing 0 to 3 black hole regions.
    - The right region contains a vertical legend: every entry is a 2x2 color block with an adjacent 1x1 color cell, and entries are ordered from top to bottom.

    Rule:
    - Find the hole count H of each left-side shape by counting enclosed black 4-connected regions inside the shape's bounding box.
    - Read the H-th legend entry from the top; its 2x2 block color is the new color for the shape's solid pixels, and its singleton color is the fill color for that shape's holes.
    - Recolor each shape accordingly while leaving the grey divider and the right-side legend unchanged.

    Key insight:
    - The legend is not keyed by the shape's size or position; the only lookup key is how many enclosed holes the azure component contains, and the two legend colors play different roles for body and hole pixels.
    """
    g = np.asarray(grid, dtype=int)
    divider_c = _find_divider(g)
    legend = _legend_entries(g, divider_c)

    out = g.copy()
    for comp in _components(g[:, :divider_c] != BLACK):
        r0, c0, r1, c1 = _bbox(comp)
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            local[r - r0, c - c0] = True
        holes = _enclosed_holes(local)
        body_color, hole_color = legend[len(holes)]
        for r, c in comp:
            out[r, c] = int(body_color)
        for region in holes:
            for rr, cc in region:
                out[r0 + rr, c0 + cc] = int(hole_color)
    return out


def _make_shape(hole_count: int) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(SHAPE_RETRIES):
        if hole_count == 0:
            base_h = random.randint(3, 6)
            base_w = random.randint(3, 6)
        else:
            base_h = random.randint(5, 7)
            base_w = random.randint(5, 7)

        canvas = np.zeros((base_h + 2, base_w + 2), dtype=bool)
        canvas[1 : 1 + base_h, 1 : 1 + base_w] = True

        holes: list[tuple[int, int]] = []
        if hole_count:
            interior = [(r, c) for r in range(2, base_h) for c in range(2, base_w)]
            random.shuffle(interior)
            for cell in interior:
                if all(abs(cell[0] - r) + abs(cell[1] - c) > 1 for r, c in holes):
                    holes.append(cell)
                if len(holes) == hole_count:
                    break
            if len(holes) != hole_count:
                continue
        for r, c in holes:
            canvas[r, c] = False

        sides = ["top", "bottom", "left", "right"]
        random.shuffle(sides)
        for side in sides[: random.randint(0, 2)]:
            if side in ("top", "bottom"):
                span = random.randint(1, min(3, base_w))
                start = random.randint(1, base_w - span + 1)
                row = 0 if side == "top" else base_h + 1
                canvas[row, start : start + span] = True
            else:
                span = random.randint(1, min(3, base_h))
                start = random.randint(1, base_h - span + 1)
                col = 0 if side == "left" else base_w + 1
                canvas[start : start + span, col] = True

        mask = _crop_mask(canvas)
        hole_regions = _enclosed_holes(mask)
        if len(_components(mask)) != 1 or len(hole_regions) != hole_count:
            continue

        hole_mask = np.zeros_like(mask)
        for region in hole_regions:
            for r, c in region:
                hole_mask[r, c] = True
        return mask, hole_mask
    raise RuntimeError(f"failed to build shape with {hole_count} holes")


def _choose_palette() -> tuple[list[int], list[int]]:
    body_colors = random.sample(list(PALETTE_COLORS), LEGEND_ROWS)
    for _ in range(40):
        hole_colors = random.sample(list(PALETTE_COLORS), LEGEND_ROWS)
        if all(body_colors[i] != hole_colors[i] for i in range(LEGEND_ROWS)):
            return body_colors, hole_colors
    shift = random.randint(1, LEGEND_ROWS - 1)
    return body_colors, body_colors[shift:] + body_colors[:shift]


def _place_shapes(
    grid: np.ndarray,
    left_w: int,
    shapes: list[dict[str, np.ndarray | int]],
) -> list[tuple[int, int, int]] | None:
    h = grid.shape[0]
    blocked = np.zeros((h, left_w), dtype=bool)
    placements: list[tuple[int, int, int]] = []
    order = sorted(
        range(len(shapes)),
        key=lambda i: int(np.asarray(shapes[i]["mask"], dtype=bool).sum()),
        reverse=True,
    )
    for idx in order:
        mask = np.asarray(shapes[idx]["mask"], dtype=bool)
        sh, sw = mask.shape
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            top = random.randint(0, h - sh)
            left = random.randint(0, left_w - sw)
            if blocked[top : top + sh, left : left + sw].any():
                continue
            grid[top : top + sh, left : left + sw][mask] = SHAPE
            blocked[
                max(0, top - 1) : min(h, top + sh + 1),
                max(0, left - 1) : min(left_w, left + sw + 1),
            ] = True
            placements.append((idx, int(top), int(left)))
            placed = True
            break
        if not placed:
            return None
    placements.sort()
    return placements


def _expected_output(
    grid: np.ndarray,
    shapes: list[dict[str, np.ndarray | int]],
    placements: list[tuple[int, int, int]],
    body_colors: list[int],
    hole_colors: list[int],
) -> np.ndarray:
    out = grid.copy()
    for idx, top, left in placements:
        mask = np.asarray(shapes[idx]["mask"], dtype=bool)
        hole_mask = np.asarray(shapes[idx]["hole_mask"], dtype=bool)
        hole_count = int(shapes[idx]["hole_count"])
        block = out[top : top + mask.shape[0], left : left + mask.shape[1]]
        block[mask] = int(body_colors[hole_count])
        block[hole_mask] = int(hole_colors[hole_count])
    return out


def _draw_legend(grid: np.ndarray, divider_c: int, body_colors: list[int], hole_colors: list[int]) -> None:
    h, w = grid.shape
    right_w = w - divider_c - 1
    top = random.randint(0, h - LEGEND_HEIGHT)
    left = divider_c + 1 + random.randint(0, right_w - 3)
    for holes in range(LEGEND_ROWS):
        r0 = top + holes * LEGEND_STEP
        grid[r0 : r0 + 2, left : left + 2] = int(body_colors[holes])
        grid[r0 + random.randint(0, 1), left + 2] = int(hole_colors[holes])


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    right_w = random.randint(*RIGHT_W_RANGE)
    left_w = random.randint(*LEFT_W_RANGE)
    grid_w = left_w + 1 + right_w
    if grid_w > 30 or grid_h < LEGEND_HEIGHT:
        return None

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    grid[:, left_w] = DIVIDER

    body_colors, hole_colors = _choose_palette()
    _draw_legend(grid, left_w, body_colors, hole_colors)

    shape_count = random.randint(*SHAPE_COUNT_RANGE)
    hole_counts = random.sample(list(range(LEGEND_ROWS)), shape_count)
    shapes: list[dict[str, np.ndarray | int]] = []
    for hole_count in hole_counts:
        mask, hole_mask = _make_shape(hole_count)
        if mask.shape[0] > grid_h or mask.shape[1] > left_w:
            return None
        shapes.append(
            {
                "mask": mask,
                "hole_mask": hole_mask,
                "hole_count": int(hole_count),
            }
        )

    placements = _place_shapes(grid, left_w, shapes)
    if placements is None:
        return None

    expected = _expected_output(grid, shapes, placements, body_colors, hole_colors)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
