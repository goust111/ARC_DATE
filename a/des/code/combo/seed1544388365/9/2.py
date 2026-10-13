import random

import numpy as np


BG = 0
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_MAX_ATTEMPTS = 300
ALL_DIM_OPTIONS = [
    (2, 2),
    (2, 3),
    (3, 3),
    (3, 4),
    (4, 4),
    (4, 5),
    (5, 5),
    (5, 6),
    (6, 6),
    (6, 7),
    (7, 7),
    (7, 8),
]


def _same_color_degree(mask: np.ndarray, r: int, c: int) -> int:
    height, width = mask.shape
    degree = 0
    for dr, dc in NEIGHBORS4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < height and 0 <= cc < width and bool(mask[rr, cc]):
            degree += 1
    return degree


def _rectangle_dims_from_perimeter(perimeter: int) -> tuple[int, int]:
    assert perimeter >= 4 and perimeter % 2 == 0, "rectangle outline perimeter must be even and at least 4"
    side_sum = perimeter // 2 + 2
    width = side_sum // 2
    height = side_sum - width
    assert 2 <= width <= height
    assert 2 * (width + height) - 4 == perimeter
    return width, height


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group same-color fragments, infer the encoded rectangle size from the
    fragment pixel count, and redraw the rectangle outline at its anchor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)

    out = np.zeros_like(g)
    for color in sorted(int(c) for c in np.unique(g) if int(c) != BG):
        mask = g == color
        points = np.argwhere(mask)
        anchors = []
        for r, c in points:
            rr, cc = int(r), int(c)
            if _same_color_degree(mask, rr, cc) == 0:
                anchors.append((rr, cc))
        assert len(anchors) == 1, f"color {color} must have exactly one isolated anchor"

        anchor_r, anchor_c = anchors[0]
        perimeter = int(points.shape[0] - 1)
        rect_w, rect_h = _rectangle_dims_from_perimeter(perimeter)
        bottom = anchor_r + rect_h - 1
        right = anchor_c + rect_w - 1
        assert bottom < height and right < width, "anchored rectangle must fit inside the grid"

        out[anchor_r, anchor_c : right + 1] = color
        out[bottom, anchor_c : right + 1] = color
        out[anchor_r : bottom + 1, anchor_c] = color
        out[anchor_r : bottom + 1, right] = color

    return out


def _boundary_cells(rect_w: int, rect_h: int) -> set[tuple[int, int]]:
    return {
        (r, c)
        for r in range(rect_h)
        for c in range(rect_w)
        if r in (0, rect_h - 1) or c in (0, rect_w - 1)
    }


def _segment_candidates(rect_w: int, rect_h: int) -> list[tuple[tuple[int, int], ...]]:
    candidates: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()

    for r in sorted({0, rect_h - 1}):
        for c0 in range(rect_w):
            for seg_len in range(2, 6):
                if c0 + seg_len > rect_w:
                    continue
                seg = tuple((r, c) for c in range(c0, c0 + seg_len))
                if seg not in seen:
                    seen.add(seg)
                    candidates.append(seg)

    for c in sorted({0, rect_w - 1}):
        for r0 in range(rect_h):
            for seg_len in range(2, 6):
                if r0 + seg_len > rect_h:
                    continue
                seg = tuple((r, c) for r in range(r0, r0 + seg_len))
                if seg not in seen:
                    seen.add(seg)
                    candidates.append(seg)

    return candidates


def _partition_outline(rect_w: int, rect_h: int) -> list[tuple[tuple[int, int], ...]]:
    boundary = _boundary_cells(rect_w, rect_h)
    candidates = _segment_candidates(rect_w, rect_h)
    random.shuffle(candidates)
    candidate_sets = [set(seg) for seg in candidates]
    by_cell = {cell: [idx for idx, seg in enumerate(candidate_sets) if cell in seg] for cell in boundary}
    dead_states: set[frozenset[tuple[int, int]]] = set()

    def dfs(remaining: set[tuple[int, int]]) -> list[tuple[tuple[int, int], ...]] | None:
        key = frozenset(remaining)
        if key in dead_states:
            return None
        if not remaining:
            return []

        pivot = min(
            remaining,
            key=lambda cell: sum(1 for idx in by_cell[cell] if candidate_sets[idx] <= remaining),
        )
        options = [idx for idx in by_cell[pivot] if candidate_sets[idx] <= remaining]
        random.shuffle(options)
        if random.random() < 0.5:
            options.sort(key=lambda idx: -len(candidates[idx]))

        for idx in options:
            answer = dfs(remaining - candidate_sets[idx])
            if answer is not None:
                return [candidates[idx]] + answer

        dead_states.add(key)
        return None

    answer = dfs(set(boundary))
    assert answer is not None, f"failed to partition {rect_w}x{rect_h} outline"
    return answer


def _box_cells(top: int, left: int, rect_h: int, rect_w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(top, top + rect_h) for c in range(left, left + rect_w)}


def _halo(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    height, width = shape
    halo = set()
    for r, c in cells:
        for rr in range(max(0, r - 1), min(height, r + 2)):
            for cc in range(max(0, c - 1), min(width, c + 2)):
                halo.add((rr, cc))
    return halo


def _segment_orientation(segment: tuple[tuple[int, int], ...]) -> str:
    rows = {r for r, _ in segment}
    return "h" if len(rows) == 1 else "v"


def _place_segment(
    shape: tuple[int, int],
    seg_len: int,
    orientation: str,
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    height, width = shape
    candidates: list[list[tuple[int, int]]] = []

    if orientation == "h":
        for r in range(height):
            for c0 in range(width - seg_len + 1):
                cells = [(r, c) for c in range(c0, c0 + seg_len)]
                if all(cell not in blocked for cell in cells):
                    candidates.append(cells)
    else:
        for r0 in range(height - seg_len + 1):
            for c in range(width):
                cells = [(r, c) for r in range(r0, r0 + seg_len)]
                if all(cell not in blocked for cell in cells):
                    candidates.append(cells)

    if not candidates:
        return None
    return random.choice(candidates)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r, c in np.argwhere(mask):
        rr, cc = int(r), int(c)
        if bool(seen[rr, cc]):
            continue
        stack = [(rr, cc)]
        seen[rr, cc] = True
        comp = []

        while stack:
            cr, cc2 = stack.pop()
            comp.append((cr, cc2))
            for dr, dc in NEIGHBORS4:
                nr = cr + dr
                nc = cc2 + dc
                if 0 <= nr < height and 0 <= nc < width and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                    seen[nr, nc] = True
                    stack.append((nr, nc))

        components.append(comp)

    return components


def _is_straight_segment(cells: list[tuple[int, int]]) -> bool:
    if not 2 <= len(cells) <= 5:
        return False

    rows = {r for r, _ in cells}
    cols = {c for _, c in cells}
    if len(rows) == 1:
        ordered = sorted(c for _, c in cells)
        return ordered == list(range(ordered[0], ordered[-1] + 1))
    if len(cols) == 1:
        ordered = sorted(r for r, _ in cells)
        return ordered == list(range(ordered[0], ordered[-1] + 1))
    return False


def _validate_input(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        return False

    height, width = g.shape
    for r, c in np.argwhere(g != BG):
        rr, cc = int(r), int(c)
        color = int(g[rr, cc])
        for nr in range(max(0, rr - 1), min(height, rr + 2)):
            for nc in range(max(0, cc - 1), min(width, cc + 2)):
                if (nr, nc) == (rr, cc):
                    continue
                other = int(g[nr, nc])
                if other != BG and other != color:
                    return False

    for color in sorted(int(c) for c in np.unique(g) if int(c) != BG):
        mask = g == color
        anchors = []
        for r, c in np.argwhere(mask):
            rr, cc = int(r), int(c)
            if _same_color_degree(mask, rr, cc) == 0:
                anchors.append((rr, cc))
        if len(anchors) != 1:
            return False

        frag_mask = mask.copy()
        frag_mask[anchors[0]] = False
        for comp in _components(frag_mask):
            if not _is_straight_segment(comp):
                return False

        try:
            rect_w, rect_h = _rectangle_dims_from_perimeter(int(frag_mask.sum()))
        except AssertionError:
            return False
        anchor_r, anchor_c = anchors[0]
        if anchor_r + rect_h > height or anchor_c + rect_w > width:
            return False

    return True


def _render_expected(shape: tuple[int, int], objects: list[dict[str, object]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for obj in objects:
        color = int(obj["color"])
        top, left = obj["anchor"]
        rect_w = int(obj["w"])
        rect_h = int(obj["h"])
        bottom = top + rect_h - 1
        right = left + rect_w - 1
        out[top, left : right + 1] = color
        out[bottom, left : right + 1] = color
        out[top : bottom + 1, left] = color
        out[top : bottom + 1, right] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    num_colors = random.randint(2, 4)
    height = random.randint(20, 28)
    width = random.randint(20, 28)
    side_limit = {2: 8, 3: 7, 4: 6}[num_colors]
    dim_options = [dims for dims in ALL_DIM_OPTIONS if dims[1] <= side_limit + 1]
    colors = random.sample(list(range(1, 10)), num_colors)

    objects: list[dict[str, object]] = []
    for color in colors:
        rect_w, rect_h = random.choice(dim_options)
        segments = _partition_outline(rect_w, rect_h)
        objects.append({"color": color, "w": rect_w, "h": rect_h, "segments": segments})

    objects.sort(key=lambda obj: (-int(obj["w"]) * int(obj["h"]), -int(obj["h"]), -int(obj["w"])))
    reserved: set[tuple[int, int]] = set()
    shape = (height, width)

    for obj in objects:
        rect_w = int(obj["w"])
        rect_h = int(obj["h"])
        if height <= rect_h + 2 or width <= rect_w + 2:
            return None

        placed = False
        for _ in range(200):
            top = random.randint(1, height - rect_h - 1)
            left = random.randint(1, width - rect_w - 1)
            box = _box_cells(top, left, rect_h, rect_w)
            halo = _halo(box, shape)
            if halo & reserved:
                continue
            obj["anchor"] = (top, left)
            reserved |= halo
            placed = True
            break
        if not placed:
            return None

    grid = np.zeros(shape, dtype=int)
    blocked = set(reserved)
    for obj in objects:
        top, left = obj["anchor"]
        grid[top, left] = int(obj["color"])

    fragment_specs: list[tuple[int, str, int]] = []
    for obj in objects:
        color = int(obj["color"])
        for segment in obj["segments"]:
            fragment_specs.append((len(segment), _segment_orientation(segment), color))

    random.shuffle(fragment_specs)
    fragment_specs.sort(key=lambda spec: -spec[0])

    for seg_len, orientation, color in fragment_specs:
        placed_cells = _place_segment(shape, seg_len, orientation, blocked)
        if placed_cells is None:
            return None
        for r, c in placed_cells:
            grid[r, c] = color
        blocked |= _halo(set(placed_cells), shape)

    expected = _render_expected(shape, objects)
    if not _validate_input(grid):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_MAX_ATTEMPTS} attempts")
