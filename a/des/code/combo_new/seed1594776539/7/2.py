import math
from functools import lru_cache

import numpy as np


RNG = np.random.default_rng()
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
PALETTE = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _normalize_coords(coords) -> tuple[tuple[int, int], ...]:
    min_r = min(int(r) for r, _ in coords)
    min_c = min(int(c) for _, c in coords)
    return tuple(sorted((int(r) - min_r, int(c) - min_c) for r, c in coords))


def _shape_size(coords: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(int(r) for r, _ in coords) + 1, max(int(c) for _, c in coords) + 1


def _components_by_color(grid: np.ndarray, bg: int) -> list[dict]:
    height, width = grid.shape
    seen = np.zeros((height, width), dtype=bool)
    components: list[dict] = []

    for row in range(height):
        for col in range(width):
            color = int(grid[row, col])
            if color == bg or seen[row, col]:
                continue

            stack = [(row, col)]
            seen[row, col] = True
            coords = []
            while stack:
                cur_r, cur_c = stack.pop()
                coords.append((cur_r, cur_c))
                for d_r, d_c in NEIGHBORS4:
                    nxt_r = cur_r + d_r
                    nxt_c = cur_c + d_c
                    if not (0 <= nxt_r < height and 0 <= nxt_c < width):
                        continue
                    if seen[nxt_r, nxt_c] or int(grid[nxt_r, nxt_c]) != color:
                        continue
                    seen[nxt_r, nxt_c] = True
                    stack.append((nxt_r, nxt_c))

            components.append({"color": color, "coords": tuple(coords)})

    return components


def _factor_pairs(area: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for height in range(1, int(math.isqrt(area)) + 1):
        if area % height != 0:
            continue
        width = area // height
        pairs.append((height, width))
        if height != width:
            pairs.append((width, height))
    return pairs


@lru_cache(maxsize=None)
def _placement_masks(
    piece: tuple[tuple[int, int], ...], height: int, width: int
) -> tuple[int, ...]:
    piece_h, piece_w = _shape_size(piece)
    if piece_h > height or piece_w > width:
        return ()

    placements = []
    for top in range(height - piece_h + 1):
        for left in range(width - piece_w + 1):
            mask = 0
            for row, col in piece:
                bit = (top + int(row)) * width + left + int(col)
                mask |= 1 << bit
            placements.append(mask)
    return tuple(placements)


def _count_packings(
    pieces: tuple[tuple[tuple[int, int], ...], ...],
    height: int,
    width: int,
    limit: int = 2,
) -> int:
    ordered = tuple(
        sorted(
            (tuple(piece) for piece in pieces),
            key=lambda piece: (-len(piece), -_shape_size(piece)[0] * _shape_size(piece)[1], piece),
        )
    )
    placement_lists = []
    for piece in ordered:
        placements = _placement_masks(piece, height, width)
        if not placements:
            return 0
        placement_lists.append(placements)

    full_mask = (1 << (height * width)) - 1
    piece_count = len(ordered)
    memo: dict[tuple[int, int], int] = {}

    def dfs(occupied: int, used_mask: int) -> int:
        key = (occupied, used_mask)
        if key in memo:
            return memo[key]
        if occupied == full_mask:
            return 1

        uncovered = full_mask & ~occupied
        pivot = uncovered & -uncovered
        total = 0

        for index in range(piece_count):
            if used_mask & (1 << index):
                continue
            for placement in placement_lists[index]:
                if (placement & pivot) == 0 or (placement & occupied) != 0:
                    continue
                total += dfs(occupied | placement, used_mask | (1 << index))
                if total >= limit:
                    memo[key] = limit
                    return limit

        memo[key] = total
        return total

    return dfs(0, 0)


def _candidate_rectangles(
    pieces: tuple[tuple[tuple[int, int], ...], ...]
) -> list[tuple[int, int, int]]:
    area = sum(len(piece) for piece in pieces)
    candidates = []
    for height, width in _factor_pairs(area):
        count = _count_packings(pieces, int(height), int(width), limit=2)
        if count > 0:
            candidates.append((int(height), int(width), int(count)))
    candidates.sort(key=lambda item: (item[0] * item[1], item[0], item[1]))
    return candidates


def _infer_rectangle_dims(
    pieces: tuple[tuple[tuple[int, int], ...], ...]
) -> tuple[int, int]:
    candidates = _candidate_rectangles(pieces)
    dims = {(height, width) for height, width, _ in candidates}
    assert len(dims) == 1, f"ambiguous rectangle assembly: {candidates}"
    return next(iter(dims))


def _unique_rectangle_solution(
    pieces: tuple[tuple[tuple[int, int], ...], ...]
) -> tuple[int, int] | None:
    candidates = _candidate_rectangles(pieces)
    if len(candidates) != 1:
        return None
    height, width, count = candidates[0]
    if count != 1:
        return None
    return height, width


def _build_nested(rectangles: list[tuple[int, int, int, int]]) -> np.ndarray:
    ordered = sorted(rectangles, key=lambda item: (item[0], item[2], item[3]))
    _outer_area, outer_color, outer_h, outer_w = ordered[-1]
    out = np.full((outer_h, outer_w), int(outer_color), dtype=int)

    cur_top = 0
    cur_left = 0
    cur_h = outer_h
    cur_w = outer_w
    for _area, color, height, width in reversed(ordered[:-1]):
        assert cur_h - height >= 2 and cur_w - width >= 2
        cur_top += (cur_h - height) // 2
        cur_left += (cur_w - width) // 2
        out[cur_top : cur_top + height, cur_left : cur_left + width] = int(color)
        cur_h = height
        cur_w = width

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble each color's translated fragments into a solid rectangle, then nest the rectangles by area.

    Input structure:
    - Black is the background.
    - Each non-background color appears as 2 to 4 separate 4-connected fragments.
    - For any one color, translating its fragments without rotation or reflection exactly tiles one solid rectangle.

    Rule:
    - For each non-background color, recover the unique rectangle that its fragments can tile and record that rectangle's height and width.
    - Sort the recovered rectangles by area from small to large.
    - Build the output on the largest rectangle's canvas, then place each next smaller solid rectangle centered inside the previous one, breaking non-exact centering toward the top-left.

    Key insight:
    - The puzzle is two-stage: infer hidden rectangles by translation-only packing, then redraw those rectangles as a size-ordered concentric nesting.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    pieces_by_color: dict[int, list[tuple[tuple[int, int], ...]]] = {}
    for component in _components_by_color(g, bg):
        color = int(component["color"])
        piece = _normalize_coords(component["coords"])
        pieces_by_color.setdefault(color, []).append(piece)

    rectangles = []
    for color, pieces in pieces_by_color.items():
        assert 2 <= len(pieces) <= 4
        height, width = _infer_rectangle_dims(tuple(pieces))
        rectangles.append((int(height * width), int(color), int(height), int(width)))

    assert rectangles, "at least one foreground color is required"
    return _build_nested(rectangles)


def _random_connected_partition(
    height: int, width: int, part_count: int
) -> list[tuple[tuple[int, int], ...]] | None:
    labels = np.full((height, width), -1, dtype=int)
    cells = [(row, col) for row in range(height) for col in range(width)]
    seed_ids = RNG.choice(len(cells), size=part_count, replace=False)

    for label, seed_id in enumerate(seed_ids):
        row, col = cells[int(seed_id)]
        labels[row, col] = int(label)

    unassigned = {(row, col) for row, col in cells if labels[row, col] < 0}
    while unassigned:
        frontier = []
        for row, col in unassigned:
            neighbor_labels = []
            for d_r, d_c in NEIGHBORS4:
                nxt_r = row + d_r
                nxt_c = col + d_c
                if 0 <= nxt_r < height and 0 <= nxt_c < width and labels[nxt_r, nxt_c] >= 0:
                    neighbor_labels.append(int(labels[nxt_r, nxt_c]))
            if neighbor_labels:
                frontier.append((row, col, tuple(sorted(set(neighbor_labels)))))

        if not frontier:
            return None

        row, col, neighbor_labels = frontier[int(RNG.integers(len(frontier)))]
        label = int(neighbor_labels[int(RNG.integers(len(neighbor_labels)))])
        labels[row, col] = label
        unassigned.remove((row, col))

    groups = [[] for _ in range(part_count)]
    for row in range(height):
        for col in range(width):
            groups[int(labels[row, col])].append((row, col))
    return [tuple(group) for group in groups]


def _generate_fragments_for_rectangle(
    height: int, width: int
) -> tuple[tuple[tuple[int, int], ...], ...] | None:
    area = height * width
    max_parts = min(4, area // 2)
    part_counts = list(range(2, max_parts + 1))
    RNG.shuffle(part_counts)

    for part_count in part_counts:
        for _ in range(160):
            groups = _random_connected_partition(height, width, int(part_count))
            if groups is None:
                continue

            pieces = tuple(
                sorted(
                    (_normalize_coords(group) for group in groups),
                    key=lambda piece: (len(piece), _shape_size(piece)[0], _shape_size(piece)[1], piece),
                )
            )
            if min(len(piece) for piece in pieces) < 2:
                continue
            if len(set(pieces)) != len(pieces):
                continue
            if _unique_rectangle_solution(pieces) == (height, width):
                return pieces

    return None


def _sample_dimension_chain() -> list[tuple[int, int]] | None:
    color_count = int(RNG.integers(2, 5))
    for _ in range(120):
        base_h = int(RNG.integers(2, 5))
        base_w = int(RNG.integers(2, 6))
        if base_h * base_w < 6:
            continue

        dims = [(base_h, base_w)]
        ok = True
        for index in range(1, color_count):
            remaining = color_count - index - 1
            prev_h, prev_w = dims[-1]
            max_h = 12 - 2 * remaining
            max_w = 12 - 2 * remaining
            min_h = prev_h + 2
            min_w = prev_w + 2
            if min_h > max_h or min_w > max_w:
                ok = False
                break
            next_h = int(RNG.integers(min_h, min(prev_h + 4, max_h) + 1))
            next_w = int(RNG.integers(min_w, min(prev_w + 4, max_w) + 1))
            dims.append((next_h, next_w))

        if ok and len({height * width for height, width in dims}) == len(dims):
            return dims

    return None


def _piece_bbox_area(piece: tuple[tuple[int, int], ...]) -> int:
    height, width = _shape_size(piece)
    return height * width


def _scatter_fragments(records: list[dict]) -> np.ndarray | None:
    total_fg = sum(len(record["piece"]) for record in records)
    padded_bbox_sum = sum((_shape_size(record["piece"])[0] + 2) * (_shape_size(record["piece"])[1] + 2) for record in records)
    min_h = max(16, max(_shape_size(record["piece"])[0] for record in records) + 2)
    min_w = max(16, max(_shape_size(record["piece"])[1] for record in records) + 2)

    for _ in range(120):
        height = int(RNG.integers(min_h, 31))
        width = int(RNG.integers(min_w, 31))
        if height * width < max(total_fg * 3, padded_bbox_sum * 2):
            continue

        grid = np.zeros((height, width), dtype=int)
        blocked = np.zeros((height, width), dtype=bool)
        order = list(range(len(records)))
        RNG.shuffle(order)
        order.sort(
            key=lambda idx: (_piece_bbox_area(records[idx]["piece"]), len(records[idx]["piece"])),
            reverse=True,
        )

        ok = True
        for idx in order:
            piece = records[idx]["piece"]
            color = int(records[idx]["color"])
            piece_h, piece_w = _shape_size(piece)
            positions = [(top, left) for top in range(height - piece_h + 1) for left in range(width - piece_w + 1)]
            for pos_id in RNG.permutation(len(positions)):
                top, left = positions[int(pos_id)]
                r0 = max(0, top - 1)
                r1 = min(height, top + piece_h + 1)
                c0 = max(0, left - 1)
                c1 = min(width, left + piece_w + 1)
                if np.any(blocked[r0:r1, c0:c1]):
                    continue

                for row, col in piece:
                    grid[top + int(row), left + int(col)] = color
                blocked[r0:r1, c0:c1] = True
                break
            else:
                ok = False
                break

        if ok:
            return grid

    return None


def _gen_attempt() -> np.ndarray | None:
    dims = _sample_dimension_chain()
    if dims is None:
        return None

    colors = [int(color) for color in RNG.choice(np.array(PALETTE, dtype=int), size=len(dims), replace=False)]
    records = []
    expected_rectangles = []
    for color, (height, width) in zip(colors, dims):
        pieces = _generate_fragments_for_rectangle(int(height), int(width))
        if pieces is None:
            return None
        piece_list = list(pieces)
        RNG.shuffle(piece_list)
        for piece in piece_list:
            records.append({"color": int(color), "piece": piece})
        expected_rectangles.append((int(height * width), int(color), int(height), int(width)))

    grid = _scatter_fragments(records)
    if grid is None:
        return None

    bg = _get_bg(grid)
    components = _components_by_color(grid, bg)
    counts_by_color: dict[int, int] = {}
    for component in components:
        color = int(component["color"])
        counts_by_color[color] = counts_by_color.get(color, 0) + 1

    if set(counts_by_color) != set(colors):
        return None
    if any(counts_by_color[color] < 2 or counts_by_color[color] > 4 for color in counts_by_color):
        return None

    expected = _build_nested(expected_rectangles)
    actual = main(grid)
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
