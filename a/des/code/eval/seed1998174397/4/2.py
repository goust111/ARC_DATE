import random
from collections import defaultdict

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _normalize(cells) -> tuple[tuple[int, int], ...]:
    coords = [(int(r), int(c)) for r, c in cells]
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return tuple(sorted((r - min_r, c - min_c) for r, c in coords))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _transforms(shape: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    variants = []
    seen = set()
    points = [(int(r), int(c)) for r, c in shape]
    for rotations in range(4):
        rotated = points
        for _ in range(rotations):
            rotated = [(c, -r) for r, c in rotated]
        for flip in (False, True):
            transformed = [(r, -c) if flip else (r, c) for r, c in rotated]
            normalized = _normalize(transformed)
            if normalized not in seen:
                seen.add(normalized)
                variants.append(normalized)
    return variants


def _is_l_shape(shape: tuple[tuple[int, int], ...]) -> bool:
    height, width = _shape_dims(shape)
    cells = set(shape)
    for corner_r in (0, height - 1):
        for corner_c in (0, width - 1):
            arm = {(corner_r, c) for c in range(width)} | {(r, corner_c) for r in range(height)}
            if arm == cells:
                return True
    return False


def _components_for_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = grid == color
    h, w = grid.shape
    seen = np.zeros_like(mask, dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(comp)
    return components


def _placements_in_rectangle(
    shape: tuple[tuple[int, int], ...], rect_h: int, rect_w: int
) -> list[set[tuple[int, int]]]:
    placements = []
    for variant in _transforms(shape):
        shape_h, shape_w = _shape_dims(variant)
        if shape_h > rect_h or shape_w > rect_w:
            continue
        for r0 in range(rect_h - shape_h + 1):
            for c0 in range(rect_w - shape_w + 1):
                placements.append({(r0 + r, c0 + c) for r, c in variant})
    return placements


def _candidate_rectangles(
    shape_a: tuple[tuple[int, int], ...], shape_b: tuple[tuple[int, int], ...]
) -> list[tuple[int, int]]:
    total_area = len(shape_a) + len(shape_b)
    candidates = set()
    for rect_h in range(2, total_area + 1):
        if total_area % rect_h != 0:
            continue
        rect_w = total_area // rect_h
        full = {(r, c) for r in range(rect_h) for c in range(rect_w)}
        placements_a = _placements_in_rectangle(shape_a, rect_h, rect_w)
        placements_b = _placements_in_rectangle(shape_b, rect_h, rect_w)
        found = False
        for cells_a in placements_a:
            for cells_b in placements_b:
                if cells_a.isdisjoint(cells_b) and cells_a | cells_b == full:
                    candidates.add((rect_h, rect_w))
                    found = True
                    break
            if found:
                break
    return sorted(candidates)


def _rectangles_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _unique_assignment(
    specs: list[dict], grid_h: int, grid_w: int
) -> dict[int, tuple[int, int]] | None:
    order = sorted(specs, key=lambda item: (len(item["candidates"]), item["anchor"], item["color"]))
    active_rects = []
    chosen: dict[int, tuple[int, int]] = {}
    solutions = []

    def dfs(index: int) -> None:
        if len(solutions) > 1:
            return
        if index == len(order):
            solutions.append(chosen.copy())
            return

        spec = order[index]
        left = int(spec["anchor"])
        for rect_h, rect_w in spec["candidates"]:
            top = grid_h - 1 - rect_h
            if top < 0 or left + rect_w > grid_w:
                continue
            rect = (top, left, grid_h - 2, left + rect_w - 1)
            if any(_rectangles_overlap(rect, other) for other in active_rects):
                continue
            active_rects.append(rect)
            chosen[int(spec["color"])] = (rect_h, rect_w)
            dfs(index + 1)
            active_rects.pop()
            chosen.pop(int(spec["color"]))

    dfs(0)
    if len(solutions) != 1:
        return None
    return solutions[0]


def _build_output(
    grid_h: int, grid_w: int, bg: int, specs: list[dict], assignment: dict[int, tuple[int, int]]
) -> np.ndarray:
    out = np.full((grid_h, grid_w), bg, dtype=int)
    for spec in specs:
        color = int(spec["color"])
        rect_h, rect_w = assignment[color]
        left = int(spec["anchor"])
        top = grid_h - 1 - rect_h
        out[top : grid_h - 1, left : left + rect_w] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored L-shaped fragments, reassemble each pair into a solid rectangle,
    and place that rectangle above the matching bottom-row marker.
    """
    g = np.asarray(grid, dtype=int)
    grid_h, grid_w = g.shape
    bg = _get_bg(g)

    specs = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) != bg):
        components = _components_for_color(g, color)
        markers = [comp for comp in components if len(comp) == 1 and comp[0][0] == grid_h - 1]
        fragments = [comp for comp in components if comp not in markers]
        assert len(markers) == 1, "each foreground color must have exactly one bottom-row marker"
        assert len(fragments) == 2, "each foreground color must have exactly two fragments"

        shapes = []
        for fragment in fragments:
            shape = _normalize(fragment)
            assert 3 <= len(shape) <= 5, "fragment size must be between 3 and 5"
            assert _is_l_shape(shape), "fragment must be L-shaped"
            shapes.append(shape)

        candidates = _candidate_rectangles(shapes[0], shapes[1])
        assert candidates, "fragments must tile at least one rectangle"
        specs.append(
            {
                "color": color,
                "anchor": int(markers[0][0][1]),
                "candidates": candidates,
            }
        )

    assignment = _unique_assignment(specs, grid_h, grid_w)
    assert assignment is not None, "rectangle placement must be globally unique"
    return _build_output(grid_h, grid_w, bg, specs, assignment)


GEN_MAX_ATTEMPTS = 400
GRID_HEIGHTS = (4, 5)
GRID_WIDTH_MAX = 30
FRAGMENT_GAP_ATTEMPTS = 200

BASE_SHAPES = {
    3: _normalize([(0, 0), (0, 1), (1, 0)]),
    4: _normalize([(0, 0), (0, 1), (0, 2), (1, 0)]),
    5: _normalize([(0, 0), (0, 1), (0, 2), (0, 3), (1, 0)]),
}

SIZE_PROFILES = {
    4: [(3, 4), (3, 5), (3, 4, 5)],
    5: [(5, 3), (5, 4), (5, 3, 4)],
}


def _target_orientation(size: int, grid_h: int) -> str:
    if grid_h == 4:
        return "V" if size == 3 else "H"
    return "H" if size == 5 else "V"


def _target_dims(size: int, grid_h: int) -> tuple[int, int]:
    if _target_orientation(size, grid_h) == "H":
        return 2, size
    return size, 2


def _random_partition(total: int, slots: int) -> list[int]:
    parts = [0] * slots
    for _ in range(total):
        parts[random.randrange(slots)] += 1
    return parts


def _anchor_specs(colors: list[int], sizes: list[int], grid_h: int) -> list[dict]:
    specs = []
    for color, size in zip(colors, sizes):
        target_h, target_w = _target_dims(size, grid_h)
        candidates = [(2, size)]
        if grid_h - 1 >= size:
            candidates.append((size, 2))
        specs.append(
            {
                "color": int(color),
                "size": int(size),
                "target": (target_h, target_w),
                "candidates": sorted(set(candidates)),
            }
        )
    return specs


def _choose_anchors(specs: list[dict], grid_h: int, grid_w: int) -> dict[int, int] | None:
    target_width_total = sum(spec["target"][1] for spec in specs)
    gap_total = len(specs) - 1
    min_band_width = target_width_total + gap_total
    if min_band_width > grid_w:
        return None

    for _ in range(200):
        order = specs[:]
        random.shuffle(order)
        extra = grid_w - min_band_width
        pad_before, *gap_extras, pad_after = _random_partition(extra, len(order) + 1)
        band_start = pad_before
        cursor = band_start
        anchors = {}
        for index, spec in enumerate(order):
            anchors[int(spec["color"])] = cursor
            cursor += spec["target"][1]
            if index < len(order) - 1:
                cursor += 1 + gap_extras[index]
        cursor += pad_after
        assert cursor == grid_w

        candidate_specs = [
            {"color": spec["color"], "anchor": anchors[int(spec["color"])], "candidates": spec["candidates"]}
            for spec in specs
        ]
        unique = _unique_assignment(candidate_specs, grid_h, grid_w)
        if unique is None:
            continue
        target_assignment = {int(spec["color"]): spec["target"] for spec in specs}
        if unique == target_assignment:
            return anchors
    return None


def _mark_blocked(blocked: np.ndarray, cells: set[tuple[int, int]]) -> None:
    grid_h, grid_w = blocked.shape
    for r, c in cells:
        for dr, dc in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid_h and 0 <= nc < grid_w:
                blocked[nr, nc] = True


def _place_fragments(
    grid: np.ndarray,
    colors: list[int],
    sizes: list[int],
    anchors: dict[int, int],
) -> bool:
    grid_h, grid_w = grid.shape
    blocked = np.zeros_like(grid, dtype=bool)

    for color in colors:
        marker = {(grid_h - 1, int(anchors[int(color)]))}
        grid[grid_h - 1, int(anchors[int(color)])] = int(color)
        _mark_blocked(blocked, marker)

    fragments = []
    for color, size in zip(colors, sizes):
        for _ in range(2):
            fragments.append({"color": int(color), "shape": BASE_SHAPES[int(size)], "size": int(size)})
    random.shuffle(fragments)
    fragments.sort(key=lambda item: (-item["size"], item["color"]))

    for fragment in fragments:
        options = []
        for variant in _transforms(fragment["shape"]):
            shape_h, shape_w = _shape_dims(variant)
            if shape_h > grid_h - 1 or shape_w > grid_w:
                continue
            for r0 in range(grid_h - shape_h):
                for c0 in range(grid_w - shape_w + 1):
                    cells = {(r0 + r, c0 + c) for r, c in variant}
                    if any(blocked[r, c] for r, c in cells):
                        continue
                    options.append(cells)
        if not options:
            return False
        cells = random.choice(options)
        for r, c in cells:
            grid[r, c] = fragment["color"]
        _mark_blocked(blocked, cells)

    return True


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.choice(GRID_HEIGHTS)
    sizes = list(random.choice(SIZE_PROFILES[grid_h]))
    random.shuffle(sizes)

    bg = random.randrange(10)
    color_pool = [color for color in range(10) if color != bg]
    colors = random.sample(color_pool, len(sizes))

    base_specs = _anchor_specs(colors, sizes, grid_h)
    min_width = sum(spec["target"][1] for spec in base_specs) + len(base_specs) - 1
    width_lo = max(min_width + 4, 14 + len(sizes) * 2)
    width_hi = GRID_WIDTH_MAX
    if width_lo > width_hi:
        return None

    for _ in range(FRAGMENT_GAP_ATTEMPTS):
        grid_w = random.randint(width_lo, width_hi)
        anchors = _choose_anchors(base_specs, grid_h, grid_w)
        if anchors is None:
            continue

        grid = np.full((grid_h, grid_w), bg, dtype=int)
        if not _place_fragments(grid, colors, sizes, anchors):
            continue

        solve_specs = [
            {
                "color": int(color),
                "anchor": int(anchors[int(color)]),
                "candidates": _candidate_rectangles(BASE_SHAPES[int(size)], BASE_SHAPES[int(size)]),
            }
            for color, size in zip(colors, sizes)
        ]
        assignment = _unique_assignment(solve_specs, grid_h, grid_w)
        target_assignment = {int(color): _target_dims(int(size), grid_h) for color, size in zip(colors, sizes)}
        if assignment != target_assignment:
            continue

        expected = _build_output(grid_h, grid_w, bg, solve_specs, target_assignment)
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
