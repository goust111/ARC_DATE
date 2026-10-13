from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_HELPER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_HELPER_DIR) not in sys.path:
    sys.path.append(str(_HELPER_DIR))

from arc_utils import Color, bbox_coords, bbox_h, bbox_w, components_mask, in_bounds


BBox = tuple[int, int, int, int]
Rect = tuple[int, int, int, int]


_GEN_ATTEMPTS = 500
_GEN_GRID_RANGE = (18, 28)
_GEN_OCC_H_RANGE = (7, 12)
_GEN_OCC_W_RANGE = (7, 12)
_GEN_SQUARE_COUNT = (1, 2, 3)
_GEN_MIN_SIDE = 4
_GEN_RED_DISTRACTOR_SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (1, 0)),
)
_GEN_FRAGMENT_TYPES = ("h", "v", "corner")


def _corner_marker_coords(bbox: BBox) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return [
        (r0 - 1, c0),
        (r0, c0 - 1),
        (r0 - 1, c1),
        (r0, c1 + 1),
        (r1 + 1, c0),
        (r1, c0 - 1),
        (r1 + 1, c1),
        (r1, c1 + 1),
    ]


def _border_coords(bbox: BBox) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
        coords.append((r1, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c0))
        coords.append((r, c1))
    return coords


def _inside_rect(r: int, c: int, rect: Rect) -> bool:
    r0, c0, r1, c1 = rect
    return r0 <= r <= r1 and c0 <= c <= c1


def _draw_border(grid: np.ndarray, bbox: BBox, color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _border_background_color(grid: np.ndarray) -> int:
    border = np.concatenate((grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]))
    values, counts = np.unique(border, return_counts=True)
    return int(values[np.argmax(counts)])


def _touches_outer_border(rect: Rect, shape: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = rect
    h, w = shape
    return r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1


def _largest_solid_rectangle(grid: np.ndarray) -> tuple[int, Rect]:
    bg = _border_background_color(grid)
    all_rects: list[tuple[int, int, Rect]] = []
    for color in map(int, np.unique(grid)):
        if color == int(Color.RED):
            continue
        for coords in components_mask(grid == color, connectivity=4):
            bbox = bbox_coords(coords)
            h = bbox_h(bbox)
            w = bbox_w(bbox)
            area = int(coords.shape[0])
            if h > 1 and w > 1 and area == h * w:
                all_rects.append((area, color, bbox))
    rects = [
        item
        for item in all_rects
        if item[1] != bg and not _touches_outer_border(item[2], grid.shape)
    ]
    if not rects:
        rects = [item for item in all_rects if item[1] != bg]
    if not rects:
        rects = [item for item in all_rects if not _touches_outer_border(item[2], grid.shape)]
    assert rects, "expected a solid rectangular occluder"
    max_area = max(area for area, _color, _bbox in rects)
    largest = [(color, bbox) for area, color, bbox in rects if area == max_area]
    assert len(largest) == 1, "largest solid rectangle must be unique"
    return largest[0]


def _visible_border_color(grid: np.ndarray, bbox: BBox, occluder: tuple[int, Rect]) -> int | None:
    occ_color, occ_rect = occluder
    visible_values: set[int] = set()
    hidden_count = 0
    visible_count = 0
    for r, c in _border_coords(bbox):
        if _inside_rect(r, c, occ_rect):
            hidden_count += 1
            continue
        visible_count += 1
        visible_values.add(int(grid[r, c]))
    if hidden_count == 0 or visible_count == 0:
        return None
    if len(visible_values) != 1:
        return None
    color = next(iter(visible_values))
    if color in {int(Color.RED), int(occ_color)}:
        return None
    return color


def _find_square_witnesses(grid: np.ndarray) -> list[tuple[BBox, int]]:
    g = np.asarray(grid, dtype=int)
    occluder = _largest_solid_rectangle(g)
    red = int(Color.RED)
    h, w = g.shape
    found: list[tuple[BBox, int]] = []

    max_side = min(h, w) - 2
    for side in range(2, max_side + 1):
        for r0 in range(1, h - side):
            r1 = r0 + side - 1
            for c0 in range(1, w - side):
                c1 = c0 + side - 1
                bbox = (r0, c0, r1, c1)
                markers = _corner_marker_coords(bbox)
                if not all(in_bounds(g.shape, r, c) and int(g[r, c]) == red for r, c in markers):
                    continue
                color = _visible_border_color(g, bbox, occluder)
                if color is None:
                    continue
                found.append((bbox, color))
    return found


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct each occluded square frame whose four corners are witnessed by red outside markers.

    Input structure:
    - A large solid rectangle hides parts of one or more 1-pixel square frames.
    - Every genuine square has exactly the full eight red witness pixels just outside
      its four corners.
    - Extra red clumps and unmarked fragments may also appear and must be ignored.

    Rule:
    - Group the red pixels by square geometry.
    - For every successful eight-pixel corner witness set, recover that square's color
      from the visible border fragments and redraw the full square frame, including the
      portion hidden under the rectangle.
    - Leave the occluder elsewhere, the red markers, and all distractors unchanged.
    """
    g = np.asarray(input_grid, dtype=int)
    out = g.copy()

    for bbox, color in _find_square_witnesses(g):
        _draw_border(out, bbox, color)

    return out


def _enumerate_candidate_bboxes(shape: tuple[int, int], occluder: Rect) -> list[BBox]:
    h, w = map(int, shape)
    out: list[BBox] = []
    max_side = min(h, w) - 2
    for side in range(_GEN_MIN_SIDE, max_side + 1):
        for r0 in range(1, h - side):
            r1 = r0 + side - 1
            for c0 in range(1, w - side):
                c1 = c0 + side - 1
                bbox = (r0, c0, r1, c1)
                markers = _corner_marker_coords(bbox)
                if not all(not _inside_rect(r, c, occluder) for r, c in markers):
                    continue
                border = _border_coords(bbox)
                hidden = sum(_inside_rect(r, c, occluder) for r, c in border)
                if hidden == 0 or hidden == len(border):
                    continue
                out.append(bbox)
    return out


def _bbox_signature(bbox: BBox) -> set[tuple[int, int]]:
    return set(_border_coords(bbox) + _corner_marker_coords(bbox))


def _place_red_distractors(
    grid: np.ndarray,
    blocked: np.ndarray,
    count: int,
    bg: int,
) -> bool:
    h, w = grid.shape
    for _ in range(count):
        placed = False
        for _try in range(60):
            shape = random.choice(_GEN_RED_DISTRACTOR_SHAPES)
            shape_h = max(r for r, _c in shape) + 1
            shape_w = max(c for _r, c in shape) + 1
            r0 = random.randint(0, h - shape_h)
            c0 = random.randint(0, w - shape_w)
            cells = [(r0 + dr, c0 + dc) for dr, dc in shape]
            if any(blocked[r, c] for r, c in cells):
                continue
            if any(int(grid[r, c]) != int(bg) for r, c in cells):
                continue
            for r, c in cells:
                grid[r, c] = int(Color.RED)
                blocked[r, c] = True
            placed = True
            break
        if not placed:
            return False
    return True


def _place_fragment_distractors(
    grid: np.ndarray,
    blocked: np.ndarray,
    palette: list[int],
    count: int,
    bg: int,
    occluder: Rect,
) -> bool:
    h, w = grid.shape
    for _ in range(count):
        placed = False
        for _try in range(80):
            kind = random.choice(_GEN_FRAGMENT_TYPES)
            color = random.choice(palette)
            if kind == "h":
                seg_len = random.randint(2, 5)
                rr = random.randint(0, h - 1)
                cc = random.randint(0, w - seg_len)
                cells = [(rr, cc + k) for k in range(seg_len)]
            elif kind == "v":
                seg_len = random.randint(2, 5)
                rr = random.randint(0, h - seg_len)
                cc = random.randint(0, w - 1)
                cells = [(rr + k, cc) for k in range(seg_len)]
            else:
                rr = random.randint(0, h - 2)
                cc = random.randint(0, w - 2)
                arms = random.choice(
                    (
                        [(0, 0), (0, 1), (1, 0)],
                        [(0, 0), (0, 1), (1, 1)],
                        [(0, 0), (1, 0), (1, 1)],
                        [(0, 1), (1, 0), (1, 1)],
                    )
                )
                cells = [(rr + dr, cc + dc) for dr, dc in arms]
            if any(blocked[r, c] for r, c in cells):
                continue
            if any(_inside_rect(r, c, occluder) for r, c in cells):
                continue
            if any(int(grid[r, c]) != bg for r, c in cells):
                continue
            for r, c in cells:
                grid[r, c] = int(color)
                blocked[r, c] = True
            placed = True
            break
        if not placed:
            return False
    return True


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(_GEN_ATTEMPTS):
        h = random.randint(*_GEN_GRID_RANGE)
        w = random.randint(*_GEN_GRID_RANGE)
        non_red = [c for c in range(10) if c != int(Color.RED)]
        bg, occ_color = random.sample(non_red, 2)
        grid = np.full((h, w), bg, dtype=int)

        occ_h = random.randint(*_GEN_OCC_H_RANGE)
        occ_w = random.randint(*_GEN_OCC_W_RANGE)
        if occ_h >= h - 4 or occ_w >= w - 4:
            continue
        occ_r0 = random.randint(2, h - occ_h - 2)
        occ_c0 = random.randint(2, w - occ_w - 2)
        occluder = (occ_r0, occ_c0, occ_r0 + occ_h - 1, occ_c0 + occ_w - 1)

        candidates = _enumerate_candidate_bboxes((h, w), occluder)
        random.shuffle(candidates)
        if not candidates:
            continue

        square_colors = random.sample([c for c in range(10) if c not in {bg, occ_color, int(Color.RED)}], 4)
        border_reserved = np.zeros((h, w), dtype=bool)
        marker_reserved = np.zeros((h, w), dtype=bool)
        squares: list[tuple[BBox, int]] = []

        target_count = random.choice(_GEN_SQUARE_COUNT)
        for bbox in candidates:
            coords = _bbox_signature(bbox)
            if any(border_reserved[r, c] or marker_reserved[r, c] for r, c in coords):
                continue
            color = square_colors[len(squares) % len(square_colors)]
            for r, c in _border_coords(bbox):
                border_reserved[r, c] = True
            for r, c in _corner_marker_coords(bbox):
                marker_reserved[r, c] = True
            squares.append((bbox, color))
            if len(squares) == target_count:
                break

        if not squares:
            continue

        full = grid.copy()
        for bbox, color in squares:
            _draw_border(full, bbox, color)
            for r, c in _corner_marker_coords(bbox):
                full[r, c] = int(Color.RED)

        inp = full.copy()
        r0, c0, r1, c1 = occluder
        inp[r0 : r1 + 1, c0 : c1 + 1] = occ_color
        for bbox, _color in squares:
            for r, c in _corner_marker_coords(bbox):
                inp[r, c] = int(Color.RED)

        blocked = border_reserved | marker_reserved
        blocked[r0 : r1 + 1, c0 : c1 + 1] = True

        fragment_palette = [c for c in range(10) if c not in {bg, occ_color, int(Color.RED)}]
        if not _place_fragment_distractors(
            inp,
            blocked,
            fragment_palette,
            count=random.randint(1, 4),
            bg=bg,
            occluder=occluder,
        ):
            continue
        if not _place_red_distractors(inp, blocked, count=random.randint(1, 4), bg=bg):
            continue

        expected = inp.copy()
        for bbox, color in squares:
            _draw_border(expected, bbox, color)

        found = _find_square_witnesses(inp)
        if len(found) != len(squares):
            continue
        truth = {bbox: color for bbox, color in squares}
        observed = {bbox: color for bbox, color in found}
        if truth != observed:
            continue
        if not np.array_equal(main(inp), expected):
            continue
        return inp, expected

    raise RuntimeError(f"generate_input: failed after {_GEN_ATTEMPTS} attempts")


def generate_input() -> np.ndarray:
    return _generate_case()[0]
