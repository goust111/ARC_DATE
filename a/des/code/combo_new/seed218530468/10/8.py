import random
from collections import deque

import numpy as np


NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# --------------- solver helpers ---------------


def _most_common(values: np.ndarray) -> int:
    palette, counts = np.unique(np.asarray(values, dtype=int), return_counts=True)
    return int(palette[np.argmax(counts)])


def _is_single_component(coords: np.ndarray) -> bool:
    pts = {tuple(map(int, cell)) for cell in np.asarray(coords, dtype=int)}
    if not pts:
        return False
    seen = set()
    queue = deque([next(iter(pts))])
    while queue:
        cell = queue.popleft()
        if cell in seen:
            continue
        seen.add(cell)
        r, c = cell
        for dr, dc in NEIGHBORS4:
            nxt = (r + dr, c + dc)
            if nxt in pts and nxt not in seen:
                queue.append(nxt)
    return len(seen) == len(pts)


def _parse_pairs(
    g: np.ndarray,
    sep_col: int,
    sep_color: int,
    left_bg: int,
    right_bg: int,
) -> list[dict]:
    colors = set(map(int, np.unique(g))) - {int(sep_color), int(left_bg), int(right_bg)}
    assert colors, "expected at least one paired color"

    pairs = []
    left_objects = 0
    right_objects = 0
    marker_cells = set()

    for color in colors:
        coords = np.argwhere(g == int(color))
        left_coords = coords[coords[:, 1] < sep_col]
        right_coords = coords[coords[:, 1] > sep_col]
        assert left_coords.size and right_coords.size, "each paired color must appear on both sides"

        if len(left_coords) == 1 and len(right_coords) >= 2:
            marker = tuple(map(int, left_coords[0]))
            obj_coords = np.asarray(right_coords, dtype=int)
            target_bg = int(left_bg)
            object_side = "right"
            right_objects += 1
        elif len(right_coords) == 1 and len(left_coords) >= 2:
            marker = tuple(map(int, right_coords[0]))
            obj_coords = np.asarray(left_coords, dtype=int)
            target_bg = int(right_bg)
            object_side = "left"
            left_objects += 1
        else:
            raise AssertionError("each color must form one object and one singleton marker")

        assert _is_single_component(obj_coords), "object cells must form one 4-connected component"
        r0 = int(obj_coords[:, 0].min())
        c0 = int(obj_coords[:, 1].min())
        offsets = {(int(r - r0), int(c - c0)) for r, c in obj_coords}

        marker_cells.add(marker)
        pairs.append(
            {
                "color": int(color),
                "object_side": object_side,
                "marker": marker,
                "target_bg": target_bg,
                "offsets": offsets,
            }
        )

    assert left_objects > 0 and right_objects > 0, "both sides must contain at least one object"
    assert len(marker_cells) == len(pairs), "marker locations must be unique"
    return pairs


def _parse_instance(grid: np.ndarray) -> tuple[int, int, int, int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape

    candidates = []
    for sep_col in range(1, width - 1):
        if not np.all(g[:, sep_col] == g[0, sep_col]):
            continue
        sep_color = int(g[0, sep_col])
        left = g[:, :sep_col]
        right = g[:, sep_col + 1 :]
        if left.size == 0 or right.size == 0:
            continue
        left_bg = _most_common(left)
        right_bg = _most_common(right)
        if len({int(sep_color), int(left_bg), int(right_bg)}) != 3:
            continue
        try:
            pairs = _parse_pairs(g, sep_col, sep_color, left_bg, right_bg)
        except AssertionError:
            continue
        candidates.append((int(sep_col), int(sep_color), int(left_bg), int(right_bg), pairs))

    assert len(candidates) == 1, "expected one valid vertical separator layout"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each monochrome object as a background-colored stamp on the opposite side of the separator.

    Input structure:
    - A one-cell-wide vertical separator column splits the grid into a left region and a right region with different background colors.
    - Each side contains one or more 4-connected monochrome objects in colors distinct from both region backgrounds and from the separator.
    - For every object color, the opposite side contains exactly one singleton marker of that same color.

    Rule:
    - Detect the separator column and the dominant background color on each side.
    - For each paired color, treat the larger same-color component as the source object and the opposite singleton as the destination anchor.
    - Convert the source object's full pixel set into offsets from its bounding box top-left, then paint those offsets at the marker position.
    - The copied shape is recolored to the destination side's background color, clipped at the grid boundary and the separator, while existing marker pixels stay unchanged.

    Key insight:
    - Color only tells you which object matches which opposite-side marker; the copied geometry comes from the object shape, and the copied color comes from the destination partition rather than the source object.
    """
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    sep_col, _sep_color, _left_bg, _right_bg, pairs = _parse_instance(g)

    out = g.copy()
    marker_cells = {pair["marker"] for pair in pairs}

    for pair in pairs:
        marker_r, marker_c = pair["marker"]
        for dr, dc in pair["offsets"]:
            rr = int(marker_r + dr)
            cc = int(marker_c + dc)
            if not (0 <= rr < height and 0 <= cc < width):
                continue
            if cc == sep_col or (rr, cc) in marker_cells:
                continue
            out[rr, cc] = int(pair["target_bg"])

    return out


# --------------- generator helpers ---------------

PAIR_COUNT_CHOICES = (1, 2, 3)
BOX_SIZE_CHOICES = (
    (2, 3),
    (3, 2),
    (2, 4),
    (4, 2),
    (3, 3),
    (3, 4),
    (4, 3),
    (3, 5),
    (5, 3),
    (4, 4),
    (4, 5),
    (5, 4),
)
GRID_HEIGHT_RANGE = (12, 24)
SIDE_WIDTH_RANGE = (7, 13)
RECT_GAP = 1
PAIR_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 120
GENERATION_ATTEMPTS = 400


def _shortest_path(
    start: tuple[int, int],
    goal: tuple[int, int],
    height: int,
    width: int,
) -> set[tuple[int, int]]:
    if start == goal:
        return {start}

    queue = deque([start])
    parent = {start: None}
    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            break
        steps = list(NEIGHBORS4)
        random.shuffle(steps)
        for dr, dc in steps:
            rr = r + dr
            cc = c + dc
            nxt = (rr, cc)
            if not (0 <= rr < height and 0 <= cc < width):
                continue
            if nxt == (0, 0) or nxt in parent:
                continue
            parent[nxt] = (r, c)
            queue.append(nxt)

    assert goal in parent, "anchors must be connectable without using the top-left corner"

    path = set()
    cur = goal
    while cur is not None:
        path.add(cur)
        cur = parent[cur]
    return path


def _random_shape(height: int, width: int) -> set[tuple[int, int]]:
    assert height >= 2 and width >= 2
    top = (0, random.randint(1, width - 1))
    left = (random.randint(1, height - 1), 0)
    bottom = (height - 1, random.randint(0, width - 1))
    right = (random.randint(0, height - 1), width - 1)

    anchors = list(dict.fromkeys([top, left, bottom, right]))
    random.shuffle(anchors)
    cells = {anchors[0]}
    for anchor in anchors[1:]:
        source = random.choice(tuple(cells))
        cells |= _shortest_path(source, anchor, height, width)

    max_area = height * width - 1
    target_area = random.randint(max(3, len(cells)), max(max(3, len(cells)), max_area))
    while len(cells) < target_area:
        frontier = set()
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                rr = r + dr
                cc = c + dc
                nxt = (rr, cc)
                if 0 <= rr < height and 0 <= cc < width and nxt != (0, 0) and nxt not in cells:
                    frontier.add(nxt)
        if not frontier:
            break
        cells.add(random.choice(tuple(frontier)))

    rows = {r for r, _c in cells}
    cols = {c for _r, c in cells}
    assert 0 in rows and height - 1 in rows and 0 in cols and width - 1 in cols
    assert (0, 0) not in cells
    return cells


def _shape_pair(height: int, width: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    for _ in range(PAIR_ATTEMPTS):
        left_shape = _random_shape(height, width)
        right_shape = _random_shape(height, width)
        overlap = len(left_shape & right_shape)
        if (
            left_shape != right_shape
            and 0 < overlap < len(left_shape)
            and 0 < overlap < len(right_shape)
        ):
            return left_shape, right_shape
    raise RuntimeError("failed to build a compatible shape pair")


def _rectangles_far_enough(
    rect_a: tuple[int, int, int, int],
    rect_b: tuple[int, int, int, int],
    gap: int,
) -> bool:
    ar, ac, ah, aw = rect_a
    br, bc, bh, bw = rect_b
    return (
        ar + ah + gap <= br
        or br + bh + gap <= ar
        or ac + aw + gap <= bc
        or bc + bw + gap <= ac
    )


def _place_rectangles(
    sizes: list[tuple[int, int]],
    height: int,
    width: int,
) -> list[tuple[int, int]] | None:
    order = list(range(len(sizes)))
    order.sort(key=lambda idx: sizes[idx][0] * sizes[idx][1], reverse=True)
    if random.random() < 0.5:
        random.shuffle(order)

    placed: list[tuple[int, int] | None] = [None] * len(sizes)
    rects: list[tuple[int, int, int, int]] = []

    for idx in order:
        h, w = sizes[idx]
        candidates = [
            (r, c)
            for r in range(height - h + 1)
            for c in range(width - w + 1)
            if all(
                _rectangles_far_enough((r, c, h, w), other, RECT_GAP)
                for other in rects
            )
        ]
        if not candidates:
            return None
        r, c = random.choice(candidates)
        placed[idx] = (r, c)
        rects.append((r, c, h, w))

    return [pos for pos in placed if pos is not None]


def _build_grid() -> np.ndarray | None:
    pair_count = random.choice(PAIR_COUNT_CHOICES)
    pair_sizes = [random.choice(BOX_SIZE_CHOICES) for _ in range(pair_count)]
    pair_shapes = [_shape_pair(h, w) for h, w in pair_sizes]

    max_h = max(h for h, _w in pair_sizes)
    max_w = max(w for _h, w in pair_sizes)

    for _ in range(PLACEMENT_ATTEMPTS):
        height = random.randint(max(12, max_h + 2), GRID_HEIGHT_RANGE[1])
        left_width = random.randint(max(7, max_w + 1), SIDE_WIDTH_RANGE[1])
        right_width = random.randint(max(7, max_w + 1), SIDE_WIDTH_RANGE[1])
        if left_width + right_width + 1 > 30:
            continue

        left_boxes = _place_rectangles(pair_sizes, height, left_width)
        right_boxes = _place_rectangles(pair_sizes, height, right_width)
        if left_boxes is None or right_boxes is None:
            continue

        palette = random.sample(list(range(10)), 2 * pair_count + 3)
        sep_color, left_bg, right_bg = palette[:3]
        left_colors = palette[3 : 3 + pair_count]
        right_colors = palette[3 + pair_count :]

        grid = np.full((height, left_width + 1 + right_width), left_bg, dtype=int)
        sep_col = left_width
        grid[:, sep_col] = sep_color
        grid[:, sep_col + 1 :] = right_bg

        for idx, ((left_shape, right_shape), (h, w)) in enumerate(zip(pair_shapes, pair_sizes)):
            lr, lc = left_boxes[idx]
            rr, rc = right_boxes[idx]
            left_marker_color = right_colors[idx]
            right_marker_color = left_colors[idx]

            grid[lr, lc] = left_marker_color
            for dr, dc in left_shape:
                grid[lr + dr, lc + dc] = left_colors[idx]

            grid[rr, sep_col + 1 + rc] = right_marker_color
            for dr, dc in right_shape:
                grid[rr + dr, sep_col + 1 + rc + dc] = right_colors[idx]

        out = main(grid)
        if np.array_equal(out, grid):
            continue

        for idx, ((left_shape, right_shape), (lr, lc), (rr, rc)) in enumerate(
            zip(pair_shapes, left_boxes, right_boxes)
        ):
            left_color = left_colors[idx]
            right_color = right_colors[idx]
            assert int(np.count_nonzero(grid == left_color)) > int(np.count_nonzero(out == left_color))
            assert int(np.count_nonzero(grid == right_color)) > int(np.count_nonzero(out == right_color))
            assert grid[lr, lc] == right_color == out[lr, lc]
            assert grid[rr, sep_col + 1 + rc] == left_color == out[rr, sep_col + 1 + rc]

        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _build_grid()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid split-stamp input")
