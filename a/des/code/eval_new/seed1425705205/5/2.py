from __future__ import annotations

from collections import Counter
import random

import numpy as np


MAX_GEN_ATTEMPTS = 400
GRID_SIDE_RANGE = (18, 30)
TEMPLATE_H_RANGE = (10, 17)
TEMPLATE_W_RANGE = (10, 17)
HOLE_COUNT_CHOICES = (2, 3, 4)
HOLE_GAP = 1
OUTER_GAP = 1
PLACEMENT_ATTEMPTS = 80


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _unique_most_common_color(grid: np.ndarray) -> int | None:
    values, counts = np.unique(grid, return_counts=True)
    order = np.argsort(counts)[::-1]
    best = int(values[order[0]])
    if len(order) > 1 and int(counts[order[0]]) == int(counts[order[1]]):
        return None
    return best


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
    result: list[list[tuple[int, int]]] = []
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
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            result.append(comp)
    return result


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_template(grid: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], int]:
    candidates: list[tuple[int, int, tuple[int, int, int, int], int]] = []
    for color in map(int, np.unique(grid)):
        if color == int(bg):
            continue
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            if h < 3 or w < 3:
                continue
            perimeter = 2 * (h + w) - 4
            if len(comp) != perimeter:
                continue
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            if not (
                np.all(sub[0, :] == color)
                and np.all(sub[-1, :] == color)
                and np.all(sub[:, 0] == color)
                and np.all(sub[:, -1] == color)
            ):
                continue
            inner = sub[1:-1, 1:-1]
            if inner.size == 0 or np.all(inner == color):
                continue
            candidates.append((h * w, perimeter, (r0, c0, r1, c1), color))

    assert candidates, "expected a rectangular template frame"
    candidates.sort(reverse=True)
    best_area = candidates[0][0]
    best = [cand for cand in candidates if cand[0] == best_area]
    assert len(best) == 1, "expected a unique largest template frame"
    _, _, bbox, color = best[0]
    return bbox, color


def _infer_checker_colors(sub: np.ndarray, bg: int, frame_color: int) -> tuple[int, int]:
    inner = sub[1:-1, 1:-1]
    visible = [
        (r, c, int(inner[r, c]))
        for r in range(inner.shape[0])
        for c in range(inner.shape[1])
        if int(inner[r, c]) not in (int(bg), int(frame_color))
    ]
    colors = sorted({value for _r, _c, value in visible})
    assert len(colors) == 2, "expected two checkerboard colors inside the template"
    a, b = colors
    score_ab = 0
    score_ba = 0
    for r, c, value in visible:
        if value == (a if (r + c) % 2 == 0 else b):
            score_ab += 1
        if value == (b if (r + c) % 2 == 0 else a):
            score_ba += 1
    return (a, b) if score_ab >= score_ba else (b, a)


def _render_template(shape: tuple[int, int], frame_color: int, color0: int, color1: int) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), int(frame_color), dtype=int)
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            out[r, c] = int(color0 if (r + c) % 2 == 0 else color1)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the full checkerboard template by fitting outside fragments back into its matching holes.

    Input structure:
    - The background is the most common color.
    - One large rectangular template is outlined by a one-cell frame color and contains a two-color 1x1 checkerboard.
    - Some connected background holes are carved out of the template interior, and monochrome fragments with matching shapes lie outside the template.

    Rule:
    - Find the unique largest rectangular frame and crop to its bounding box for the output.
    - Infer the two checkerboard colors and their parity from the visible template cells; the outside fragments only indicate which holes are fillable.
    - Restore every missing interior cell to the checkerboard color dictated by its position, keeping the frame unchanged.

    Key insight:
    - Shape matching identifies which outside fragment belongs to each hole, but after placement every fragment discards its original color and simply reveals the template's underlying checkerboard.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    (r0, c0, r1, c1), frame_color = _find_template(g, bg)
    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    color0, color1 = _infer_checker_colors(sub, bg, frame_color)
    return _render_template(sub.shape, frame_color, color0, color1)


def _shape_library() -> dict[int, list[np.ndarray]]:
    return {
        3: [
            np.array([[1, 1], [1, 0]], dtype=bool),
            np.array([[1, 1, 1]], dtype=bool),
        ],
        4: [
            np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
            np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
            np.array([[1, 1], [1, 1]], dtype=bool),
        ],
        5: [
            np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
            np.array([[1, 1, 1], [1, 0, 1]], dtype=bool),
            np.array([[1, 1, 1], [0, 0, 1], [0, 0, 1]], dtype=bool),
        ],
        6: [
            np.array([[1, 1, 1], [1, 0, 1], [1, 0, 0]], dtype=bool),
            np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool),
            np.array([[0, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=bool),
        ],
        7: [
            np.array([[0, 1, 1], [1, 1, 0], [1, 1, 1]], dtype=bool),
            np.array([[1, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=bool),
        ],
    }


SHAPE_LIBRARY = _shape_library()


def _mask_coords(mask: np.ndarray) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in np.argwhere(mask)]


def _distinct_rotations(mask: np.ndarray) -> list[np.ndarray]:
    seen: set[tuple[tuple[int, int], bytes]] = set()
    result: list[np.ndarray] = []
    for k in range(4):
        rot = np.rot90(mask, k)
        key = (rot.shape, rot.tobytes())
        if key in seen:
            continue
        seen.add(key)
        result.append(rot)
    return result


def _can_place_with_gap(
    occupied: np.ndarray, coords: list[tuple[int, int]], top: int, left: int, gap: int
) -> bool:
    h, w = occupied.shape
    cells: list[tuple[int, int]] = []
    for dr, dc in coords:
        r = int(top + dr)
        c = int(left + dc)
        if not (0 <= r < h and 0 <= c < w):
            return False
        if occupied[r, c]:
            return False
        cells.append((r, c))
    for r, c in cells:
        for rr in range(max(0, r - gap), min(h, r + gap + 1)):
            for cc in range(max(0, c - gap), min(w, c + gap + 1)):
                if occupied[rr, cc]:
                    return False
    return True


def _mark_mask(canvas: np.ndarray, mask: np.ndarray, top: int, left: int, value: int) -> None:
    for dr, dc in _mask_coords(mask):
        canvas[top + dr, left + dc] = int(value)


def _place_holes(inner_h: int, inner_w: int) -> list[tuple[np.ndarray, int, int]] | None:
    hole_count = random.choice(HOLE_COUNT_CHOICES)
    areas = random.sample(sorted(SHAPE_LIBRARY), hole_count)
    occupied = np.zeros((inner_h, inner_w), dtype=bool)
    holes: list[tuple[np.ndarray, int, int]] = []

    for area in areas:
        base = random.choice(SHAPE_LIBRARY[area])
        shape = random.choice(_distinct_rotations(base))
        coords = _mask_coords(shape)
        positions = []
        for top in range(inner_h - shape.shape[0] + 1):
            for left in range(inner_w - shape.shape[1] + 1):
                if _can_place_with_gap(occupied, coords, top, left, HOLE_GAP):
                    positions.append((top, left))
        if not positions:
            return None
        top, left = random.choice(positions)
        holes.append((shape, top, left))
        for dr, dc in coords:
            occupied[top + dr, left + dc] = True

    visible_counts = Counter((r + c) % 2 for r in range(inner_h) for c in range(inner_w) if not occupied[r, c])
    if visible_counts.get(0, 0) == 0 or visible_counts.get(1, 0) == 0:
        return None
    if int(occupied.sum()) * 2 >= inner_h * inner_w:
        return None
    return holes


def _place_piece(
    grid: np.ndarray, occupied: np.ndarray, template_bbox: tuple[int, int, int, int], mask: np.ndarray, color: int
) -> bool:
    h, w = grid.shape
    r0, c0, r1, c1 = template_bbox
    coords = _mask_coords(mask)
    positions = []
    for top in range(h - mask.shape[0] + 1):
        for left in range(w - mask.shape[1] + 1):
            if top <= r1 and top + mask.shape[0] - 1 >= r0 and left <= c1 and left + mask.shape[1] - 1 >= c0:
                continue
            if _can_place_with_gap(occupied, coords, top, left, OUTER_GAP):
                positions.append((top, left))
    if not positions:
        return False
    top, left = random.choice(positions)
    _mark_mask(grid, mask, top, left, color)
    for dr, dc in coords:
        occupied[top + dr, left + dc] = True
    return True


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_SIDE_RANGE)
    grid_w = random.randint(*GRID_SIDE_RANGE)
    template_h = random.randint(*TEMPLATE_H_RANGE)
    template_w = random.randint(*TEMPLATE_W_RANGE)
    if template_h >= grid_h - 3 or template_w >= grid_w - 3:
        return None

    colors = random.sample(list(range(10)), 7)
    bg, frame_color, checker_a, checker_b = map(int, colors[:4])

    max_top = grid_h - template_h - 1
    max_left = grid_w - template_w - 1
    if max_top < 1 or max_left < 1:
        return None
    top = random.randint(1, max_top)
    left = random.randint(1, max_left)

    expected = _render_template((template_h, template_w), frame_color, checker_a, checker_b)
    holes = _place_holes(template_h - 2, template_w - 2)
    if holes is None:
        return None

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    grid[top : top + template_h, left : left + template_w] = expected
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    occupied[top : top + template_h, left : left + template_w] = True

    for shape, hole_top, hole_left in holes:
        for dr, dc in _mask_coords(shape):
            grid[top + 1 + hole_top + dr, left + 1 + hole_left + dc] = bg

    piece_palette = [c for c in range(10) if c not in {bg, frame_color, checker_a, checker_b}]
    piece_colors = random.sample(piece_palette, len(holes))

    template_bbox = (top, left, top + template_h - 1, left + template_w - 1)
    for piece_color, (shape, _hole_top, _hole_left) in zip(piece_colors, holes):
        piece_shape = random.choice(_distinct_rotations(shape))
        if not _place_piece(grid, occupied, template_bbox, piece_shape, piece_color):
            return None

    if _unique_most_common_color(grid) != bg:
        return None
    if np.array_equal(grid, expected):
        return None
    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
