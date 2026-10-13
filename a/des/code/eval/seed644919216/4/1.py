import numpy as np


BG = 0
RNG = np.random.default_rng()

GEN_RETRIES = 400
GRID_SIDE_RANGE = (24, 31)
FENCE_COUNT_OPTIONS = (4, 5)
OUTER_H_RANGE = (5, 10)
OUTER_W_RANGE = (5, 10)
PLACEMENT_RETRIES = 250
PLACEMENT_MARGIN = 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                if rr > 0 and mask[rr - 1, cc] and not seen[rr - 1, cc]:
                    seen[rr - 1, cc] = True
                    stack.append((rr - 1, cc))
                if rr + 1 < h and mask[rr + 1, cc] and not seen[rr + 1, cc]:
                    seen[rr + 1, cc] = True
                    stack.append((rr + 1, cc))
                if cc > 0 and mask[rr, cc - 1] and not seen[rr, cc - 1]:
                    seen[rr, cc - 1] = True
                    stack.append((rr, cc - 1))
                if cc + 1 < w and mask[rr, cc + 1] and not seen[rr, cc + 1]:
                    seen[rr, cc + 1] = True
                    stack.append((rr, cc + 1))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _is_rect_frame(coords: np.ndarray) -> bool:
    if coords.shape[0] < 8:
        return False
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    mask = np.zeros((h, w), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    want = np.zeros((h, w), dtype=bool)
    want[0, :] = True
    want[-1, :] = True
    want[:, 0] = True
    want[:, -1] = True
    return bool(np.array_equal(mask, want))


def _find_fences(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    fences: list[dict] = []
    colors = [int(c) for c in np.unique(g) if int(c) != BG]
    for color in colors:
        frame_comps = [coords for coords in _components(g == color) if _is_rect_frame(coords)]
        assert len(frame_comps) == 1, "each border color must define exactly one fence"
        r0, c0, r1, c1 = _bbox(frame_comps[0])
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        seeds = [(int(rr), int(cc), int(inner[rr, cc])) for rr, cc in np.argwhere(inner != BG)]
        seeds.sort()
        assert 1 <= len(seeds) <= 3, "each fence must contain 1 to 3 seeds"
        seed_colors = [color0 for _, _, color0 in seeds]
        assert len(seed_colors) == len(set(seed_colors)), "seed colors must be distinct inside a fence"
        fences.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "inner_shape": inner.shape,
                "seeds": seeds,
                "pattern": [(rr, cc) for rr, cc, _ in seeds],
            }
        )

    fences.sort(key=lambda item: item["bbox"])
    color_to_fence = {int(item["color"]): item for item in fences}
    assert len(color_to_fence) == len(fences), "fence border colors must be unique"

    for fence in fences:
        for _, _, seed_color in fence["seeds"]:
            assert seed_color in color_to_fence, "seed color must point to another fence color"
            assert seed_color != fence["color"], "seed color must point to a different fence"

    return fences


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow color-coded fills inside rectangular fences.

    Each seed color points to the fence whose border has the same color. That
    referenced fence contributes a template: the set of seed coordinates inside
    its interior, measured from that interior's top-left corner.

    For a fence F with n seeds, every seed paints copies of its referenced
    template in a square tile neighborhood of radius n - 1. The seed position is
    used as the top-left anchor of the central copy. Later seeds overwrite
    earlier ones, using row-major seed order inside the fence.
    """
    g = np.asarray(grid, dtype=int)
    fences = _find_fences(g)
    templates = {
        int(fence["color"]): (tuple(fence["pattern"]), tuple(map(int, fence["inner_shape"])))
        for fence in fences
    }

    out = g.copy()
    for fence in fences:
        r0, c0, r1, c1 = map(int, fence["bbox"])
        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        inner[:, :] = BG
        radius = len(fence["seeds"]) - 1

        for seed_r, seed_c, seed_color in fence["seeds"]:
            pattern, tile_shape = templates[int(seed_color)]
            tile_h, tile_w = map(int, tile_shape)
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    base_r = int(seed_r) + dr * tile_h
                    base_c = int(seed_c) + dc * tile_w
                    for pr, pc in pattern:
                        rr = base_r + int(pr)
                        cc = base_c + int(pc)
                        if 0 <= rr < inner.shape[0] and 0 <= cc < inner.shape[1]:
                            inner[rr, cc] = int(seed_color)

    return out


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _sample_pattern(inner_h: int, inner_w: int, count: int) -> list[tuple[int, int]]:
    assert count >= 1
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w) if (r, c) != (0, 0)]
    points = [(0, 0)]
    if count == 1:
        return points

    # Keep templates visibly nontrivial and anchor every template at its top-left.
    preferred = [(r, c) for r, c in cells if r + c >= 2]
    pool = preferred if len(preferred) >= count - 1 else cells

    for _ in range(100):
        choice_idx = RNG.choice(len(pool), size=count - 1, replace=False)
        extra = [pool[int(i)] for i in np.atleast_1d(choice_idx)]
        pts = sorted(points + extra)
        rows = {r for r, _ in pts}
        cols = {c for _, c in pts}
        if count == 2 or (len(rows) > 1 and len(cols) > 1):
            return pts

    extra = [cells[int(i)] for i in np.atleast_1d(RNG.choice(len(cells), size=count - 1, replace=False))]
    return sorted(points + extra)


def _cycle_targets(n_fences: int) -> list[int]:
    order = list(map(int, RNG.permutation(n_fences)))
    target = [0] * n_fences
    for idx, src in enumerate(order):
        target[src] = order[(idx + 1) % n_fences]
    return target


def _place_rectangles(
    grid_h: int,
    grid_w: int,
    sizes: list[tuple[int, int]],
) -> list[tuple[int, int, int, int]] | None:
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placed: list[tuple[int, int, int, int] | None] = [None] * len(sizes)
    order = sorted(range(len(sizes)), key=lambda idx: sizes[idx][0] * sizes[idx][1], reverse=True)

    for idx in order:
        rect_h, rect_w = sizes[idx]
        if grid_h - rect_h < 2 or grid_w - rect_w < 2:
            return None
        success = False
        for _ in range(PLACEMENT_RETRIES):
            r0 = int(RNG.integers(1, grid_h - rect_h))
            c0 = int(RNG.integers(1, grid_w - rect_w))
            r1 = r0 + rect_h - 1
            c1 = c0 + rect_w - 1
            rr0 = max(0, r0 - PLACEMENT_MARGIN)
            rr1 = min(grid_h, r1 + PLACEMENT_MARGIN + 1)
            cc0 = max(0, c0 - PLACEMENT_MARGIN)
            cc1 = min(grid_w, c1 + PLACEMENT_MARGIN + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            occupied[r0 : r1 + 1, c0 : c1 + 1] = True
            placed[idx] = (r0, c0, r1, c1)
            success = True
            break
        if not success:
            return None

    return [rect for rect in placed if rect is not None]


def _counts_for_fences(n_fences: int) -> list[int]:
    counts = [1, 2, 3]
    while len(counts) < n_fences:
        counts.append(int(RNG.integers(1, 4)))
    RNG.shuffle(counts)
    return counts


def _generate_candidate() -> np.ndarray | None:
    n_fences = int(FENCE_COUNT_OPTIONS[int(RNG.integers(0, len(FENCE_COUNT_OPTIONS)))])
    grid_h = int(RNG.integers(*GRID_SIDE_RANGE))
    grid_w = int(RNG.integers(*GRID_SIDE_RANGE))

    colors = list(map(int, RNG.choice(np.arange(1, 10), size=n_fences, replace=False)))
    counts = _counts_for_fences(n_fences)
    sizes = [
        (
            int(RNG.integers(*OUTER_H_RANGE)),
            int(RNG.integers(*OUTER_W_RANGE)),
        )
        for _ in range(n_fences)
    ]
    rects = _place_rectangles(grid_h, grid_w, sizes)
    if rects is None or len(rects) != n_fences:
        return None

    patterns: list[list[tuple[int, int]]] = []
    for idx, rect in enumerate(rects):
        r0, c0, r1, c1 = rect
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        patterns.append(_sample_pattern(inner_h, inner_w, counts[idx]))

    target_idx = _cycle_targets(n_fences)
    seed_colors: list[list[int]] = []
    for idx in range(n_fences):
        used = [colors[target_idx[idx]]]
        available = [colors[j] for j in range(n_fences) if j != idx and colors[j] not in used]
        extra_needed = counts[idx] - 1
        if extra_needed > 0:
            extra = list(map(int, RNG.choice(np.array(available, dtype=int), size=extra_needed, replace=False)))
            used.extend(extra)
        RNG.shuffle(used)
        seed_colors.append(used)

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for idx, rect in enumerate(rects):
        _draw_frame(grid, rect, colors[idx])
        r0, c0, r1, c1 = rect
        for (pr, pc), seed_color in zip(patterns[idx], seed_colors[idx]):
            grid[r0 + 1 + int(pr), c0 + 1 + int(pc)] = int(seed_color)

    return grid


def _valid_candidate(grid: np.ndarray) -> bool:
    try:
        fences = _find_fences(grid)
        if len(fences) not in FENCE_COUNT_OPTIONS:
            return False
        counts = {len(fence["seeds"]) for fence in fences}
        if counts != {1, 2, 3}:
            return False
        usage = {int(fence["color"]): 0 for fence in fences}
        for fence in fences:
            for _, _, seed_color in fence["seeds"]:
                usage[int(seed_color)] += 1
        if any(v == 0 for v in usage.values()):
            return False
        out = main(grid)
        changed = int(np.count_nonzero(out != grid))
        if changed < 8:
            return False
        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_candidate()
        if grid is not None and _valid_candidate(grid):
            return grid
    raise RuntimeError("failed to generate a valid input")
