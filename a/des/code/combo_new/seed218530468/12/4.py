import random
from collections import Counter

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_BG = 0
GEN_GREY = 5
SEED_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in DIR4 if 0 <= r + dr < h and 0 <= c + dc < w]


def _majority_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _path_color(grid: np.ndarray, bg: int) -> int:
    counts = Counter(int(v) for v in grid.flat if int(v) != bg)
    assert counts, "expected a non-background path color"
    max_count = max(counts.values())
    winners = [color for color, count in counts.items() if count == max_count]
    assert len(winners) == 1, "path color must be the unique largest non-background color"
    return int(winners[0])


def _grey_neighbors(pt: tuple[int, int], grey: set[tuple[int, int]]) -> list[tuple[int, int]]:
    r, c = pt
    return [(r + dr, c + dc) for dr, dc in DIR4 if (r + dr, c + dc) in grey]


def _leaf_distance(leaf: tuple[int, int], degrees: dict[tuple[int, int], int],
                   grey: set[tuple[int, int]]) -> int:
    nbrs = _grey_neighbors(leaf, grey)
    assert len(nbrs) == 1, "leaf must have exactly one grey neighbor"
    prev = leaf
    cur = nbrs[0]
    dist = 1
    while degrees[cur] == 2:
        nxt = [pt for pt in _grey_neighbors(cur, grey) if pt != prev]
        assert len(nxt) == 1, "degree-2 path node must have a single forward continuation"
        prev, cur = cur, nxt[0]
        dist += 1
    return dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each seed away from its grey leaf by the length of that leaf's terminal branch.

    Input structure:
    - The background is the dominant color.
    - One non-background color forms a single-pixel-wide grey tree or forest made of horizontal and vertical path segments with no cycles.
    - Leaf pixels on that grey structure have degree 1, branch points have degree 3 or 4, and each leaf has a colored seed immediately outside it on the side away from the tree.

    Rule:
    - For each colored seed, identify the adjacent grey leaf that points toward it.
    - Walk inward from that leaf along the grey path until reaching the first branch point, or the opposite leaf if that connected component has no branch point, and count the number of grey steps D.
    - Keep the grey structure unchanged and extend the seed's color farther in the same outward direction so that the seed plus its new continuation occupy D cells in total, stopping early at the grid boundary.

    Key insight:
    - The growth length is encoded by graph topology rather than raw geometry: each seed copies the distance from its leaf to the nearest topological decision point on the grey tree.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _majority_color(g)
    grey_color = _path_color(g, bg)

    grey = {(int(r), int(c)) for r, c in np.argwhere(g == grey_color)}
    assert grey, "expected at least one grey path pixel"
    degrees = {pt: len(_grey_neighbors(pt, grey)) for pt in grey}
    leaves = {pt for pt, deg in degrees.items() if deg == 1}
    assert leaves, "expected at least one leaf"

    out = g.copy()
    seed_total = int(((g != bg) & (g != grey_color)).sum())
    seen = 0

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or color == grey_color:
                continue

            candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
            for lr, lc in _neighbors4(r, c, h, w):
                leaf = (int(lr), int(lc))
                if leaf not in leaves:
                    continue
                dr = int(r - lr)
                dc = int(c - lc)
                inward = (int(lr - dr), int(lc - dc))
                nbrs = _grey_neighbors(leaf, grey)
                if len(nbrs) == 1 and nbrs[0] == inward:
                    candidates.append((leaf, (dr, dc)))

            assert len(candidates) == 1, "each colored pixel must be a seed outside exactly one leaf"
            leaf, (dr, dc) = candidates[0]
            dist = _leaf_distance(leaf, degrees, grey)
            for step in range(1, dist):
                rr = int(r + dr * step)
                cc = int(c + dc * step)
                if not (0 <= rr < h and 0 <= cc < w):
                    break
                out[rr, cc] = color
            seen += 1

    assert seen == seed_total, "all non-grey foreground pixels must be valid seeds"
    return out


def _rotate_point(pt: tuple[int, int], turns: int, mirror: bool) -> tuple[int, int]:
    r, c = pt
    if mirror:
        c = -c
    for _ in range(turns % 4):
        r, c = c, -r
    return (int(r), int(c))


def _normalize_component(
    grey: set[tuple[int, int]],
    leaves: list[dict[str, tuple[int, int] | int]],
    palette: list[int],
) -> dict | None:
    turns = random.randrange(4)
    mirror = random.random() < 0.5

    grey_t = {_rotate_point(pt, turns, mirror) for pt in grey}
    leaves_t = []
    for item in leaves:
        leaf = _rotate_point(item["leaf"], turns, mirror)
        prev = _rotate_point(item["prev"], turns, mirror)
        leaves_t.append({"leaf": leaf, "prev": prev, "dist": int(item["dist"])})

    input_cells: dict[tuple[int, int], int] = {pt: GEN_GREY for pt in grey_t}
    output_cells: dict[tuple[int, int], int] = {pt: GEN_GREY for pt in grey_t}
    colors = [random.choice(palette) for _ in leaves_t]
    if len(leaves_t) >= 2 and len(set(palette)) >= 2 and len(set(colors)) == 1 and random.random() < 0.7:
        idx = random.randrange(len(colors))
        options = [color for color in palette if color != colors[idx]]
        colors[idx] = random.choice(options)

    for item, color in zip(leaves_t, colors):
        leaf = item["leaf"]
        prev = item["prev"]
        dist = int(item["dist"])
        dr = int(leaf[0] - prev[0])
        dc = int(leaf[1] - prev[1])
        seed = (int(leaf[0] + dr), int(leaf[1] + dc))
        if seed in input_cells:
            return None
        input_cells[seed] = int(color)
        for step in range(dist):
            pt = (int(seed[0] + dr * step), int(seed[1] + dc * step))
            if pt in output_cells:
                return None
            output_cells[pt] = int(color)

    all_pts = list(input_cells) + list(output_cells)
    min_r = min(r for r, _ in all_pts)
    min_c = min(c for _, c in all_pts)

    def shift_point(pt: tuple[int, int]) -> tuple[int, int]:
        return (int(pt[0] - min_r), int(pt[1] - min_c))

    input_list = sorted((shift_point(pt)[0], shift_point(pt)[1], val) for pt, val in input_cells.items())
    output_list = sorted((shift_point(pt)[0], shift_point(pt)[1], val) for pt, val in output_cells.items())
    in_h = max(r for r, _, _ in input_list) + 1
    in_w = max(c for _, c, _ in input_list) + 1
    out_h = max(r for r, _, _ in output_list) + 1
    out_w = max(c for _, c, _ in output_list) + 1

    return {
        "input": input_list,
        "output": output_list,
        "in_h": int(in_h),
        "in_w": int(in_w),
        "out_h": int(out_h),
        "out_w": int(out_w),
        "area": int(out_h * out_w),
    }


def _simple_path_component(palette: list[int]) -> dict | None:
    family = random.choices(["straight", "elbow", "zigzag"], weights=[3, 3, 2])[0]

    if family == "straight":
        total = random.randint(2, 6)
        path = [(0, i) for i in range(total + 1)]
    elif family == "elbow":
        total = random.randint(3, 6)
        first = random.randint(1, total - 1)
        second = total - first
        path = [(0, i) for i in range(first + 1)]
        path.extend((r, first) for r in range(1, second + 1))
    else:
        total = random.randint(4, 7)
        first = random.randint(1, total - 2)
        second = random.randint(1, total - first - 1)
        third = total - first - second
        path = [(0, i) for i in range(first + 1)]
        path.extend((r, first) for r in range(1, second + 1))
        path.extend((second, first + c) for c in range(1, third + 1))

    grey = set(path)
    leaves = [
        {"leaf": path[0], "prev": path[1], "dist": len(path) - 1},
        {"leaf": path[-1], "prev": path[-2], "dist": len(path) - 1},
    ]
    return _normalize_component(grey, leaves, palette)


def _star_component(palette: list[int]) -> dict | None:
    arm_count = random.choice([3, 4])
    arms = random.sample(DIR4, arm_count)
    grey = {(0, 0)}
    leaves = []
    for dr, dc in arms:
        length = random.randint(1, 4)
        arm = [(int(dr * step), int(dc * step)) for step in range(1, length + 1)]
        grey.update(arm)
        prev = arm[-2] if length > 1 else (0, 0)
        leaves.append({"leaf": arm[-1], "prev": prev, "dist": length})
    return _normalize_component(grey, leaves, palette)


def _bent_branch_component(palette: list[int]) -> dict | None:
    up = random.randint(1, 4)
    left = random.randint(1, 4)
    stem = random.randint(1, 3)
    tail = random.randint(1, 3)
    tail_side = random.choice([-1, 1])
    right = random.randint(1, 4) if random.random() < 0.5 else 0

    grey = {(0, 0)}
    grey.update((int(-step), 0) for step in range(1, up + 1))
    grey.update((0, int(-step)) for step in range(1, left + 1))
    grey.update((int(step), 0) for step in range(1, stem + 1))
    grey.update((int(stem), int(tail_side * step)) for step in range(1, tail + 1))
    if right:
        grey.update((0, int(step)) for step in range(1, right + 1))

    leaves = [
        {"leaf": (-up, 0), "prev": (-(up - 1), 0) if up > 1 else (0, 0), "dist": up},
        {"leaf": (0, -left), "prev": (0, -(left - 1)) if left > 1 else (0, 0), "dist": left},
        {
            "leaf": (stem, tail_side * tail),
            "prev": (stem, tail_side * (tail - 1)) if tail > 1 else (stem - 1, 0),
            "dist": stem + tail,
        },
    ]
    if right:
        leaves.append({"leaf": (0, right), "prev": (0, right - 1) if right > 1 else (0, 0), "dist": right})
    return _normalize_component(grey, leaves, palette)


def _make_component(kind: str, palette: list[int]) -> dict | None:
    if kind == "simple":
        return _simple_path_component(palette)
    if kind == "star":
        return _star_component(palette)
    if kind == "bent":
        return _bent_branch_component(palette)
    return random.choice([_simple_path_component, _star_component, _bent_branch_component])(palette)


def _dilate_points(points: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in points:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _place_component(
    grid: np.ndarray,
    blocked: set[tuple[int, int]],
    comp: dict,
    allow_clip: bool,
) -> bool:
    h, w = grid.shape
    max_r = h - (comp["in_h"] if allow_clip else comp["out_h"])
    max_c = w - (comp["in_w"] if allow_clip else comp["out_w"])
    if max_r < 0 or max_c < 0:
        return False

    offsets = [(r0, c0) for r0 in range(max_r + 1) for c0 in range(max_c + 1)]
    random.shuffle(offsets)

    for r0, c0 in offsets:
        out_pts = [(int(r0 + r), int(c0 + c)) for r, c, _ in comp["output"]]
        in_bounds_out = {(r, c) for r, c in out_pts if 0 <= r < h and 0 <= c < w}
        if any(pt in blocked for pt in in_bounds_out):
            continue

        for r, c, color in comp["input"]:
            grid[int(r0 + r), int(c0 + c)] = int(color)
        blocked.update(_dilate_points(in_bounds_out, h, w))
        return True
    return False


def _counts_ok(grid: np.ndarray) -> bool:
    counts = Counter(int(v) for v in grid.flat if int(v) != GEN_BG)
    if GEN_GREY not in counts:
        return False
    grey_count = counts.pop(GEN_GREY)
    return all(grey_count > count for count in counts.values())


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(16, 30)
    w = random.randint(16, 30)
    palette = random.sample(SEED_POOL, random.randint(2, 5))
    component_kinds = ["simple", random.choice(["star", "bent"])]
    if random.random() < 0.6:
        component_kinds.append(random.choice(["simple", "star", "bent"]))

    components = []
    for kind in component_kinds:
        comp = _make_component(kind, palette)
        if comp is None:
            return None
        components.append(comp)
    components.sort(key=lambda item: -item["area"])

    grid = np.full((h, w), GEN_BG, dtype=int)
    blocked: set[tuple[int, int]] = set()
    clip_index = random.randrange(len(components)) if random.random() < 0.35 else None

    for idx, comp in enumerate(components):
        allow_clip = idx == clip_index
        if not _place_component(grid, blocked, comp, allow_clip):
            return None

    if not _counts_ok(grid):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
