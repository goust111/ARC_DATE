from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


Coord = tuple[int, int]
DIRS4: tuple[Coord, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))

_RNG = random.Random()


MAX_GENERATION_ATTEMPTS = 400
GRID_MIN = 24
GRID_MAX = 30
MAX_OUTPUT_SIDE = 30
PAIR_COUNT_CHOICES = (2, 2, 3, 3, 4)
EDGE_NAMES = ("top", "bottom", "left", "right")
PALETTE = tuple(range(10))

ROW_BLOCK_MIN = 1
ROW_BLOCK_MAX = 4
COL_BLOCK_MIN = 1
COL_BLOCK_MAX = 4

MACRO_A = ((0, 0), (1, 0), (1, 1))
MACRO_B = ((0, 1), (0, 2), (1, 2))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _normalize(shape: set[Coord] | frozenset[Coord]) -> frozenset[Coord]:
    rows = [r for r, _c in shape]
    cols = [c for _r, c in shape]
    min_r = min(rows)
    min_c = min(cols)
    return frozenset((r - min_r, c - min_c) for r, c in shape)


def _shape_dims(shape: set[Coord] | frozenset[Coord]) -> tuple[int, int]:
    rows = [r for r, _c in shape]
    cols = [c for _r, c in shape]
    return max(rows) + 1, max(cols) + 1


def _rotate90(shape: frozenset[Coord]) -> frozenset[Coord]:
    h, _w = _shape_dims(shape)
    return _normalize({(c, h - 1 - r) for r, c in shape})


def _unique_rotations(shape: frozenset[Coord]) -> list[frozenset[Coord]]:
    out: list[frozenset[Coord]] = []
    seen: set[frozenset[Coord]] = set()
    cur = _normalize(shape)
    for _ in range(4):
        if cur not in seen:
            out.append(cur)
            seen.add(cur)
        cur = _rotate90(cur)
    return out


def _is_proper_l(shape: frozenset[Coord]) -> bool:
    h, w = _shape_dims(shape)
    if h < 2 or w < 2:
        return False
    full = {(r, c) for r in range(h) for c in range(w)}
    missing = full - set(shape)
    if not missing:
        return False
    rows = [r for r, _c in missing]
    cols = [c for _r, c in missing]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if len(missing) != (r1 - r0 + 1) * (c1 - c0 + 1):
        return False
    return bool(
        (r0 == 0 and c0 == 0)
        or (r0 == 0 and c1 == w - 1)
        or (r1 == h - 1 and c0 == 0)
        or (r1 == h - 1 and c1 == w - 1)
    )


def _detect_direction_line(grid: np.ndarray, bg: int) -> tuple[str, int]:
    h, w = grid.shape
    candidates: list[tuple[str, int]] = []
    top = int(grid[0, 0])
    if top != bg and np.all(grid[0, :] == top):
        candidates.append(("top", top))
    bottom = int(grid[h - 1, 0])
    if bottom != bg and np.all(grid[h - 1, :] == bottom):
        candidates.append(("bottom", bottom))
    left = int(grid[0, 0])
    if left != bg and np.all(grid[:, 0] == left):
        candidates.append(("left", left))
    right = int(grid[0, w - 1])
    if right != bg and np.all(grid[:, w - 1] == right):
        candidates.append(("right", right))
    assert len(candidates) == 1
    return candidates[0]


def _find_monochrome_components(
    grid: np.ndarray, bg: int, ignored_colors: set[int]
) -> list[dict[str, object]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r0 in range(h):
        for c0 in range(w):
            if visited[r0, c0]:
                continue
            color = int(grid[r0, c0])
            if color == bg or color in ignored_colors:
                continue

            q: deque[Coord] = deque([(r0, c0)])
            visited[r0, c0] = True
            coords: list[Coord] = []

            while q:
                r, c = q.popleft()
                coords.append((r, c))
                for dr, dc in DIRS4:
                    nr = r + dr
                    nc = c + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))

            min_r = min(r for r, _c in coords)
            min_c = min(c for _r, c in coords)
            max_r = max(r for r, _c in coords)
            max_c = max(c for _r, c in coords)
            shape = frozenset((r - min_r, c - min_c) for r, c in coords)
            components.append(
                {
                    "color": color,
                    "coords": tuple(coords),
                    "bbox": (min_r, min_c, max_r, max_c),
                    "shape": shape,
                }
            )

    return components


def _candidate_rectangle_dims(shape_a: frozenset[Coord], shape_b: frozenset[Coord]) -> set[tuple[int, int]]:
    candidates: set[tuple[int, int]] = set()
    rots_a = _unique_rotations(shape_a)
    rots_b = _unique_rotations(shape_b)

    for rot_a in rots_a:
        ha, wa = _shape_dims(rot_a)
        for rot_b in rots_b:
            hb, wb = _shape_dims(rot_b)
            for dr in range(-hb, ha + 1):
                for dc in range(-wb, wa + 1):
                    shifted_b = {(r + dr, c + dc) for r, c in rot_b}
                    if set(rot_a) & shifted_b:
                        continue
                    union = set(rot_a) | shifted_b
                    norm_union = _normalize(union)
                    uh, uw = _shape_dims(norm_union)
                    if len(norm_union) == uh * uw:
                        candidates.add((min(uh, uw), max(uh, uw)))
    return candidates


def _reconstruct_rectangle_dims(shape_a: frozenset[Coord], shape_b: frozenset[Coord]) -> tuple[int, int]:
    candidates = _candidate_rectangle_dims(shape_a, shape_b)

    assert candidates
    assert len(candidates) == 1
    return next(iter(candidates))


def _pack_rectangles(rectangles: list[tuple[int, int, int]], bg: int, direction_edge: str) -> np.ndarray:
    ordered = sorted(rectangles, key=lambda item: (item[0] * item[1], item[0]))
    assert ordered

    if direction_edge in {"top", "bottom"}:
        out_h = sum(h for h, _w, _color in ordered)
        out_w = max(w for _h, w, _color in ordered)
        out = np.full((out_h, out_w), bg, dtype=int)
        if direction_edge == "top":
            cursor = out_h
            for h, w, color in ordered:
                cursor -= h
                out[cursor : cursor + h, :w] = color
        else:
            cursor = 0
            for h, w, color in ordered:
                out[cursor : cursor + h, :w] = color
                cursor += h
        return out

    out_h = max(h for h, _w, _color in ordered)
    out_w = sum(w for _h, w, _color in ordered)
    out = np.full((out_h, out_w), bg, dtype=int)
    if direction_edge == "left":
        cursor = out_w
        for h, w, color in ordered:
            cursor -= w
            out[out_h - h :, cursor : cursor + w] = color
    else:
        cursor = 0
        for h, w, color in ordered:
            out[out_h - h :, cursor : cursor + w] = color
            cursor += w
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    direction_edge, line_color = _detect_direction_line(g, bg)

    components = _find_monochrome_components(g, bg=bg, ignored_colors={line_color})
    by_color: dict[int, list[dict[str, object]]] = defaultdict(list)
    for comp in components:
        shape = comp["shape"]
        assert isinstance(shape, frozenset)
        assert _is_proper_l(shape)
        by_color[int(comp["color"])].append(comp)

    rectangles: list[tuple[int, int, int]] = []
    for color, group in by_color.items():
        assert len(group) == 2
        shape_a = group[0]["shape"]
        shape_b = group[1]["shape"]
        assert isinstance(shape_a, frozenset) and isinstance(shape_b, frozenset)
        rect_h, rect_w = _reconstruct_rectangle_dims(shape_a, shape_b)
        rectangles.append((rect_h, rect_w, int(color)))

    assert rectangles
    return _pack_rectangles(rectangles, bg=bg, direction_edge=direction_edge)


def _expand_macro_pair(row_blocks: tuple[int, int], col_blocks: tuple[int, int, int]) -> tuple[frozenset[Coord], frozenset[Coord]]:
    row_starts = (0, row_blocks[0])
    col_starts = (0, col_blocks[0], col_blocks[0] + col_blocks[1])

    def block_cells(block_r: int, block_c: int) -> set[Coord]:
        r0 = row_starts[block_r]
        c0 = col_starts[block_c]
        return {
            (r, c)
            for r in range(r0, r0 + row_blocks[block_r])
            for c in range(c0, c0 + col_blocks[block_c])
        }

    shape_a: set[Coord] = set()
    shape_b: set[Coord] = set()
    for block_r, block_c in MACRO_A:
        shape_a |= block_cells(block_r, block_c)
    for block_r, block_c in MACRO_B:
        shape_b |= block_cells(block_r, block_c)

    norm_a = _normalize(shape_a)
    norm_b = _normalize(shape_b)
    assert _is_proper_l(norm_a)
    assert _is_proper_l(norm_b)
    rect_h = sum(row_blocks)
    rect_w = sum(col_blocks)
    assert rect_h <= rect_w
    return norm_a, norm_b


def _render_shape(shape: frozenset[Coord], color: int, bg: int) -> np.ndarray:
    h, w = _shape_dims(shape)
    arr = np.full((h, w), bg, dtype=int)
    for r, c in shape:
        arr[r, c] = color
    return arr


def _build_valid_descriptor_pool() -> tuple[dict[str, object], ...]:
    pool: list[dict[str, object]] = []
    for row_a in range(ROW_BLOCK_MIN, ROW_BLOCK_MAX + 1):
        for row_b in range(ROW_BLOCK_MIN, ROW_BLOCK_MAX + 1):
            row_blocks = (row_a, row_b)
            for col_a in range(COL_BLOCK_MIN, COL_BLOCK_MAX + 1):
                for col_b in range(COL_BLOCK_MIN, COL_BLOCK_MAX + 1):
                    for col_c in range(COL_BLOCK_MIN, COL_BLOCK_MAX + 1):
                        col_blocks = (col_a, col_b, col_c)
                        rect_h = sum(row_blocks)
                        rect_w = sum(col_blocks)
                        if rect_h > rect_w:
                            continue
                        shape_a, shape_b = _expand_macro_pair(row_blocks, col_blocks)
                        if _candidate_rectangle_dims(shape_a, shape_b) != {(rect_h, rect_w)}:
                            continue
                        pool.append(
                            {
                                "rect_dims": (rect_h, rect_w),
                                "shape_a": shape_a,
                                "shape_b": shape_b,
                            }
                        )
    assert pool
    return tuple(pool)


def _has_useful_tie(rectangles: list[tuple[int, int]]) -> bool:
    by_area: dict[int, set[int]] = defaultdict(set)
    for h, w in rectangles:
        by_area[h * w].add(h)
    return any(len(heights) >= 2 for heights in by_area.values())


def _sample_descriptors(pair_count: int, direction_edge: str) -> list[dict[str, object]] | None:
    force_tie = _RNG.random() < 0.5 and pair_count >= 2

    for _ in range(200):
        descriptors = [_RNG.choice(VALID_PAIR_DESCRIPTORS) for _ in range(pair_count)]
        rect_dims = [tuple(desc["rect_dims"]) for desc in descriptors]
        assert all(isinstance(item, tuple) for item in rect_dims)

        if force_tie and not _has_useful_tie(rect_dims):
            continue
        if not force_tie and len(set(rect_dims)) == 1:
            continue

        probe = _pack_rectangles([(h, w, 1) for h, w in rect_dims], bg=0, direction_edge=direction_edge)
        if probe.shape[0] > MAX_OUTPUT_SIDE or probe.shape[1] > MAX_OUTPUT_SIDE:
            continue
        return descriptors
    return None


def _place_piece(
    grid: np.ndarray,
    occupied: np.ndarray,
    shape: frozenset[Coord],
    color: int,
    direction_edge: str,
) -> bool:
    h, w = grid.shape
    sh, sw = _shape_dims(shape)

    row_min = 1
    row_max = h - sh - 1
    col_min = 1
    col_max = w - sw - 1

    if direction_edge == "top":
        row_min = 2
    elif direction_edge == "bottom":
        row_max = h - sh - 2
    elif direction_edge == "left":
        col_min = 2
    else:
        col_max = w - sw - 2

    if row_min > row_max or col_min > col_max:
        return False

    candidates: list[Coord] = []
    for r0 in range(row_min, row_max + 1):
        for c0 in range(col_min, col_max + 1):
            rr0 = r0 - 1
            rr1 = r0 + sh + 1
            cc0 = c0 - 1
            cc1 = c0 + sw + 1
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            candidates.append((r0, c0))

    if not candidates:
        return False

    r0, c0 = _RNG.choice(candidates)
    for dr, dc in shape:
        grid[r0 + dr, c0 + dc] = color
    occupied[r0 - 1 : r0 + sh + 1, c0 - 1 : c0 + sw + 1] = True
    return True


def _random_rotated(shape: frozenset[Coord]) -> frozenset[Coord]:
    return _RNG.choice(_unique_rotations(shape))


VALID_PAIR_DESCRIPTORS = _build_valid_descriptor_pool()


def _gen_attempt() -> np.ndarray | None:
    pair_count = _RNG.choice(PAIR_COUNT_CHOICES)
    direction_edge = _RNG.choice(EDGE_NAMES)
    descriptors = _sample_descriptors(pair_count, direction_edge)
    if descriptors is None:
        return None

    bg = _RNG.choice(PALETTE)
    remaining_colors = [c for c in PALETTE if c != bg]
    _RNG.shuffle(remaining_colors)
    direction_color = remaining_colors[0]
    pair_colors = remaining_colors[1 : 1 + pair_count]
    if len(pair_colors) != pair_count:
        return None

    rectangles: list[tuple[int, int, int]] = []
    pieces: list[tuple[int, frozenset[Coord]]] = []
    for color, desc in zip(pair_colors, descriptors):
        rect_h, rect_w = desc["rect_dims"]
        shape_a = desc["shape_a"]
        shape_b = desc["shape_b"]
        assert isinstance(rect_h, int) and isinstance(rect_w, int)
        assert isinstance(shape_a, frozenset) and isinstance(shape_b, frozenset)
        rectangles.append((rect_h, rect_w, int(color)))
        pieces.append((int(color), _random_rotated(shape_a)))
        pieces.append((int(color), _random_rotated(shape_b)))

    expected = _pack_rectangles(rectangles, bg=bg, direction_edge=direction_edge)
    if expected.shape[0] > MAX_OUTPUT_SIDE or expected.shape[1] > MAX_OUTPUT_SIDE:
        return None

    max_piece_h = max(_shape_dims(shape)[0] for _color, shape in pieces)
    max_piece_w = max(_shape_dims(shape)[1] for _color, shape in pieces)
    min_h = max(GRID_MIN, max_piece_h + 4)
    min_w = max(GRID_MIN, max_piece_w + 4)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    grid_h = _RNG.randint(min_h, GRID_MAX)
    grid_w = _RNG.randint(min_w, GRID_MAX)
    grid = np.full((grid_h, grid_w), bg, dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    if direction_edge == "top":
        grid[0, :] = direction_color
        occupied[0, :] = True
    elif direction_edge == "bottom":
        grid[grid_h - 1, :] = direction_color
        occupied[grid_h - 1, :] = True
    elif direction_edge == "left":
        grid[:, 0] = direction_color
        occupied[:, 0] = True
    else:
        grid[:, grid_w - 1] = direction_color
        occupied[:, grid_w - 1] = True

    _RNG.shuffle(pieces)
    for color, shape in pieces:
        if not _place_piece(grid, occupied, shape, color, direction_edge):
            return None

    if _get_bg(grid) != bg:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {MAX_GENERATION_ATTEMPTS} attempts")
