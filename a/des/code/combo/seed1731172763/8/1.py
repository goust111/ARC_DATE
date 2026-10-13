import random
from itertools import combinations

import numpy as np


BG = 0
MAROON = 9
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8)

GRID_H_RANGE = (14, 24)
GRID_W_RANGE = (14, 24)
BOX_MIN_SIZE = 4
BOX_MAX_SIZE = 9
CORE_SIZE_RANGE = (2, 3)
SHORT_EXTRA_RANGE = (0, 2)
LONG_EXTRA_RANGE = (1, 4)
GEN_RETRIES = 400
BOX_RETRIES = 120
MARKER_RETRIES = 120
MIN_VISIBLE_OVERLAP = 3
MIN_EXCLUSIVE_FILL = 1


def _intersection(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    r0 = max(int(box_a[0]), int(box_b[0]))
    c0 = max(int(box_a[1]), int(box_b[1]))
    r1 = min(int(box_a[2]), int(box_b[2]))
    c1 = min(int(box_a[3]), int(box_b[3]))
    if r0 > r1 or c0 > c1:
        return None
    return (r0, c0, r1, c1)


def _triple_intersection(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
    assert len(boxes) == 3
    r0 = max(int(box[0]) for box in boxes)
    c0 = max(int(box[1]) for box in boxes)
    r1 = min(int(box[2]) for box in boxes)
    c1 = min(int(box[3]) for box in boxes)
    if r0 > r1 or c0 > c1:
        return None
    return (r0, c0, r1, c1)


def _box_area(box: tuple[int, int, int, int]) -> int:
    return (int(box[2]) - int(box[0]) + 1) * (int(box[3]) - int(box[1]) + 1)


def _iter_box_cells(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return [(r, c) for r in range(int(r0), int(r1) + 1) for c in range(int(c0), int(c1) + 1)]


def _expanded_intersects(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int], pad: int = 1) -> bool:
    a = (int(box_a[0]) - pad, int(box_a[1]) - pad, int(box_a[2]) + pad, int(box_a[3]) + pad)
    b = (int(box_b[0]) - pad, int(box_b[1]) - pad, int(box_b[2]) + pad, int(box_b[3]) + pad)
    return _intersection(a, b) is not None


def _random_box(height: int, width: int) -> tuple[int, int, int, int]:
    box_h = random.randint(BOX_MIN_SIZE, min(BOX_MAX_SIZE, height - 1))
    box_w = random.randint(BOX_MIN_SIZE, min(BOX_MAX_SIZE, width - 1))
    r0 = random.randint(0, height - box_h)
    c0 = random.randint(0, width - box_w)
    return (r0, c0, r0 + box_h - 1, c0 + box_w - 1)


def _sample_disjoint_box(
    height: int,
    width: int,
    existing: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    for _ in range(80):
        box = _random_box(height, width)
        if all(not _expanded_intersects(box, other, pad=1) for other in existing):
            return box
    return None


def _sample_core(
    height: int,
    width: int,
    need_top: int,
    need_bottom: int,
    need_left: int,
    need_right: int,
) -> tuple[int, int, int, int] | None:
    core_h = random.randint(CORE_SIZE_RANGE[0], CORE_SIZE_RANGE[1])
    core_w = random.randint(CORE_SIZE_RANGE[0], CORE_SIZE_RANGE[1])
    if need_top + core_h + need_bottom > height or need_left + core_w + need_right > width:
        return None
    r0 = random.randint(need_top, height - core_h - need_bottom)
    c0 = random.randint(need_left, width - core_w - need_right)
    return (r0, c0, r0 + core_h - 1, c0 + core_w - 1)


def _sample_box_from_core(
    height: int,
    width: int,
    core: tuple[int, int, int, int],
    up_range: tuple[int, int],
    down_range: tuple[int, int],
    left_range: tuple[int, int],
    right_range: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    for _ in range(40):
        max_up = min(int(up_range[1]), int(core[0]))
        max_down = min(int(down_range[1]), height - 1 - int(core[2]))
        max_left = min(int(left_range[1]), int(core[1]))
        max_right = min(int(right_range[1]), width - 1 - int(core[3]))
        if max_up < int(up_range[0]) or max_down < int(down_range[0]) or max_left < int(left_range[0]) or max_right < int(right_range[0]):
            return None

        up = random.randint(int(up_range[0]), max_up)
        down = random.randint(int(down_range[0]), max_down)
        left = random.randint(int(left_range[0]), max_left)
        right = random.randint(int(right_range[0]), max_right)

        box = (int(core[0]) - up, int(core[1]) - left, int(core[2]) + down, int(core[3]) + right)
        if _box_area(box) >= BOX_MIN_SIZE * BOX_MIN_SIZE:
            return box
    return None


def _pair_edges(boxes: list[tuple[int, int, int, int]]) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for i, j in combinations(range(len(boxes)), 2):
        inter = _intersection(boxes[i], boxes[j])
        if inter is not None and _box_area(inter) >= 4:
            edges.add((int(i), int(j)))
    return edges


def _coverage_map(shape: tuple[int, int], boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    coverage = np.zeros(shape, dtype=int)
    for r0, c0, r1, c1 in boxes:
        coverage[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] += 1
    return coverage


def _boxes_match_mode(
    boxes: list[tuple[int, int, int, int]],
    mode: str,
    shape: tuple[int, int],
) -> bool:
    coverage = _coverage_map(shape, boxes)
    edges = _pair_edges(boxes)

    if mode == "none":
        if edges:
            return False
    elif mode == "pair":
        if len(boxes) == 2:
            if edges != {(0, 1)}:
                return False
        else:
            if len(edges) != 1:
                return False
    elif mode == "chain":
        if len(boxes) != 3 or len(edges) != 2:
            return False
        degree = [0, 0, 0]
        for i, j in edges:
            degree[i] += 1
            degree[j] += 1
        if sorted(degree) != [1, 1, 2]:
            return False
        if _triple_intersection(boxes) is not None:
            return False
    elif mode == "triple":
        if len(boxes) != 3 or edges != {(0, 1), (0, 2), (1, 2)}:
            return False
        inter3 = _triple_intersection(boxes)
        if inter3 is None or _box_area(inter3) < 4:
            return False
    else:
        raise AssertionError(f"unknown mode: {mode}")

    if mode != "none" and int(np.sum(coverage >= 2)) < 4:
        return False

    for box in boxes:
        own_only = coverage[int(box[0]) : int(box[2]) + 1, int(box[1]) : int(box[3]) + 1] == 1
        if int(np.sum(own_only)) < 4:
            return False
    return True


def _generate_boxes_none(height: int, width: int, n_colors: int) -> list[tuple[int, int, int, int]] | None:
    boxes: list[tuple[int, int, int, int]] = []
    for _ in range(n_colors):
        box = _sample_disjoint_box(height, width, boxes)
        if box is None:
            return None
        boxes.append(box)
    return boxes


def _generate_boxes_pair(height: int, width: int, n_colors: int) -> list[tuple[int, int, int, int]] | None:
    core = _sample_core(height, width, 1, 1, 1, 1)
    if core is None:
        return None

    box_a = _sample_box_from_core(
        height,
        width,
        core,
        up_range=LONG_EXTRA_RANGE,
        down_range=SHORT_EXTRA_RANGE,
        left_range=LONG_EXTRA_RANGE,
        right_range=SHORT_EXTRA_RANGE,
    )
    box_b = _sample_box_from_core(
        height,
        width,
        core,
        up_range=SHORT_EXTRA_RANGE,
        down_range=LONG_EXTRA_RANGE,
        left_range=SHORT_EXTRA_RANGE,
        right_range=LONG_EXTRA_RANGE,
    )
    if box_a is None or box_b is None or box_a == box_b:
        return None

    boxes = [box_a, box_b]
    if n_colors == 3:
        extra = _sample_disjoint_box(height, width, boxes)
        if extra is None:
            return None
        boxes.append(extra)
        random.shuffle(boxes)
    return boxes


def _generate_boxes_chain(height: int, width: int) -> list[tuple[int, int, int, int]] | None:
    middle = _random_box(height, width)
    if _box_area(middle) < 36:
        return None

    mr0, mc0, mr1, mc1 = middle
    if mr1 - mr0 < 4 or mc1 - mc0 < 4:
        return None

    core1_h = random.randint(CORE_SIZE_RANGE[0], CORE_SIZE_RANGE[1])
    core1_w = random.randint(CORE_SIZE_RANGE[0], CORE_SIZE_RANGE[1])
    core2_h = random.randint(CORE_SIZE_RANGE[0], CORE_SIZE_RANGE[1])
    core2_w = random.randint(CORE_SIZE_RANGE[0], CORE_SIZE_RANGE[1])

    max_core1_r0 = mr0 + max(1, (mr1 - mr0 + 1) // 2 - core1_h)
    max_core1_c0 = mc0 + max(1, (mc1 - mc0 + 1) // 2 - core1_w)
    min_core2_r0 = mr0 + (mr1 - mr0 + 1) // 2
    min_core2_c0 = mc0 + (mc1 - mc0 + 1) // 2

    if max_core1_r0 < mr0 or max_core1_c0 < mc0 or min_core2_r0 + core2_h - 1 > mr1 or min_core2_c0 + core2_w - 1 > mc1:
        return None

    core1_r0 = random.randint(mr0, max_core1_r0)
    core1_c0 = random.randint(mc0, max_core1_c0)
    core2_r0 = random.randint(min_core2_r0, mr1 - core2_h + 1)
    core2_c0 = random.randint(min_core2_c0, mc1 - core2_w + 1)
    core1 = (core1_r0, core1_c0, core1_r0 + core1_h - 1, core1_c0 + core1_w - 1)
    core2 = (core2_r0, core2_c0, core2_r0 + core2_h - 1, core2_c0 + core2_w - 1)

    box_a = _sample_box_from_core(
        height,
        width,
        core1,
        up_range=LONG_EXTRA_RANGE,
        down_range=(0, 1),
        left_range=LONG_EXTRA_RANGE,
        right_range=(0, 1),
    )
    box_c = _sample_box_from_core(
        height,
        width,
        core2,
        up_range=(0, 1),
        down_range=LONG_EXTRA_RANGE,
        left_range=(0, 1),
        right_range=LONG_EXTRA_RANGE,
    )
    if box_a is None or box_c is None:
        return None

    boxes = [box_a, middle, box_c]
    random.shuffle(boxes)
    return boxes


def _generate_boxes_triple(height: int, width: int) -> list[tuple[int, int, int, int]] | None:
    core = _sample_core(height, width, 1, 1, 1, 1)
    if core is None:
        return None

    box_a = _sample_box_from_core(
        height,
        width,
        core,
        up_range=LONG_EXTRA_RANGE,
        down_range=SHORT_EXTRA_RANGE,
        left_range=LONG_EXTRA_RANGE,
        right_range=SHORT_EXTRA_RANGE,
    )
    box_b = _sample_box_from_core(
        height,
        width,
        core,
        up_range=LONG_EXTRA_RANGE,
        down_range=SHORT_EXTRA_RANGE,
        left_range=SHORT_EXTRA_RANGE,
        right_range=LONG_EXTRA_RANGE,
    )
    box_c = _sample_box_from_core(
        height,
        width,
        core,
        up_range=SHORT_EXTRA_RANGE,
        down_range=LONG_EXTRA_RANGE,
        left_range=(0, 1),
        right_range=(0, 1),
    )
    if box_a is None or box_b is None or box_c is None:
        return None

    boxes = [box_a, box_b, box_c]
    random.shuffle(boxes)
    return boxes


def _generate_boxes(height: int, width: int, n_colors: int, mode: str) -> list[tuple[int, int, int, int]] | None:
    for _ in range(BOX_RETRIES):
        if mode == "none":
            boxes = _generate_boxes_none(height, width, n_colors)
        elif mode == "pair":
            boxes = _generate_boxes_pair(height, width, n_colors)
        elif mode == "chain":
            boxes = _generate_boxes_chain(height, width)
        elif mode == "triple":
            boxes = _generate_boxes_triple(height, width)
        else:
            raise AssertionError(f"unknown mode: {mode}")

        if boxes is None:
            continue
        if _boxes_match_mode(boxes, mode, (height, width)):
            return boxes
    return None


def _corner_patterns(box: tuple[int, int, int, int]) -> list[tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = box
    corners = [(int(r0), int(c0)), (int(r0), int(c1)), (int(r1), int(c0)), (int(r1), int(c1))]
    patterns = [tuple(corners)]
    for combo in combinations(corners, 3):
        patterns.append(tuple(combo))
    random.shuffle(patterns)
    return patterns


def _choose_base_markers(boxes: list[tuple[int, int, int, int]]) -> list[set[tuple[int, int]]] | None:
    patterns = [_corner_patterns(box) for box in boxes]

    def dfs(index: int, used: set[tuple[int, int]], chosen: list[set[tuple[int, int]]]) -> list[set[tuple[int, int]]] | None:
        if index == len(boxes):
            return [set(points) for points in chosen]
        for pattern in patterns[index]:
            cells = set(pattern)
            if cells & used:
                continue
            chosen.append(cells)
            result = dfs(index + 1, used | cells, chosen)
            if result is not None:
                return result
            chosen.pop()
        return None

    return dfs(0, set(), [])


def _box_boundary(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells: list[tuple[int, int]] = []
    for r in range(int(r0), int(r1) + 1):
        for c in range(int(c0), int(c1) + 1):
            if r in (int(r0), int(r1)) or c in (int(c0), int(c1)):
                cells.append((r, c))
    return cells


def _add_extra_markers(
    boxes: list[tuple[int, int, int, int]],
    markers: list[set[tuple[int, int]]],
    coverage: np.ndarray,
) -> None:
    used = set().union(*markers)
    order = list(range(len(boxes)))
    random.shuffle(order)
    for idx in order:
        if random.random() < 0.35:
            continue

        candidates_overlap = [
            cell
            for cell in _iter_box_cells(boxes[idx])
            if int(coverage[cell]) >= 2 and cell not in used
        ]
        candidates_other = [cell for cell in _box_boundary(boxes[idx]) if cell not in used]
        candidates = candidates_overlap if candidates_overlap and random.random() < 0.6 else candidates_other
        if not candidates:
            continue
        cell = random.choice(candidates)
        markers[idx].add(cell)
        used.add(cell)


def _render_from_boxes(
    grid: np.ndarray,
    colors: list[int],
    boxes: list[tuple[int, int, int, int]],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    coverage = _coverage_map(out.shape, boxes)
    for color, (r0, c0, r1, c1) in zip(colors, boxes, strict=False):
        patch_in = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
        patch_out = out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
        patch_out[patch_in == BG] = int(color)
    out[(coverage >= 2) & (grid == BG)] = MAROON
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group colored marker pixels by color, fill each color's bounding box, and mark overlaps in maroon.

    Input structure:
    - Black background.
    - Two or three colors of scattered single-pixel markers.
    - Maroon does not appear in the input.

    Rule:
    - For each non-background color, compute the minimal axis-aligned bounding box of its markers.
    - Fill background pixels inside that box with the same color.
    - Any background cell covered by two or more boxes becomes Maroon.
    - Original marker pixels keep their original colors even inside overlaps.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    colors = [int(color) for color in np.unique(g) if int(color) != BG]
    if not colors:
        return out

    boxes: list[tuple[int, int, int, int]] = []
    for color in colors:
        coords = np.argwhere(g == int(color))
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        boxes.append((int(r0), int(c0), int(r1), int(c1)))

    coverage = _coverage_map(g.shape, boxes)
    for color, (r0, c0, r1, c1) in zip(colors, boxes, strict=False):
        patch_in = g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
        patch_out = out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
        patch_out[patch_in == BG] = int(color)

    out[(coverage >= 2) & (g == BG)] = MAROON
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GRID_H_RANGE[0], GRID_H_RANGE[1])
    width = random.randint(GRID_W_RANGE[0], GRID_W_RANGE[1])
    n_colors = random.choice((2, 3))
    mode = random.choice(("none", "pair") if n_colors == 2 else ("none", "pair", "chain", "triple"))
    colors = random.sample(PALETTE, n_colors)

    boxes = _generate_boxes(height, width, n_colors, mode)
    if boxes is None:
        return None

    coverage = _coverage_map((height, width), boxes)
    for _ in range(MARKER_RETRIES):
        markers = _choose_base_markers(boxes)
        if markers is None:
            return None
        markers = [set(cells) for cells in markers]
        _add_extra_markers(boxes, markers, coverage)

        grid = np.zeros((height, width), dtype=int)
        for color, points in zip(colors, markers, strict=False):
            for r, c in points:
                if grid[int(r), int(c)] != BG:
                    break
                grid[int(r), int(c)] = int(color)
            else:
                continue
            break
        else:
            ok = True
            for color, box, points in zip(colors, boxes, markers, strict=False):
                coords = np.argwhere(grid == int(color))
                if len(coords) < 3:
                    ok = False
                    break
                r0, c0 = coords.min(axis=0)
                r1, c1 = coords.max(axis=0)
                if (int(r0), int(c0), int(r1), int(c1)) != box:
                    ok = False
                    break

                own_fill = 0
                for r in range(int(box[0]), int(box[2]) + 1):
                    for c in range(int(box[1]), int(box[3]) + 1):
                        if (r, c) in points:
                            continue
                        if int(coverage[r, c]) == 1:
                            own_fill += 1
                if own_fill < MIN_EXCLUSIVE_FILL:
                    ok = False
                    break

            if not ok:
                continue

            out = _render_from_boxes(grid, colors, boxes)
            if not np.array_equal(main(grid), out):
                continue
            if MAROON in colors or int(np.any(grid == MAROON)):
                continue
            if mode == "none":
                if int(np.any(out == MAROON)):
                    continue
            else:
                maroon_count = int(np.sum(out == MAROON))
                if maroon_count < MIN_VISIBLE_OVERLAP:
                    continue
            if any(not bool(np.any((out == int(color)) & (grid == BG))) for color in colors):
                continue
            return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate a valid input after {GEN_RETRIES} attempts")
