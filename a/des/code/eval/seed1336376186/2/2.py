import random
from collections import Counter, deque

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
BRANCHES = ("fill", "vertical", "horizontal", "none")

GRID_MIN = 18
GRID_MAX = 30
GROUP_COUNT_RANGE = (1, 4)
INNER_SIZE_RANGE = (3, 6)  # interior height/width of the innermost frame
NEST_GAP_RANGE = (1, 2)
PLACEMENT_TRIES = 200
GEN_RETRIES = 300

_RNG = random.Random()


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    border = {(r0, c) for c in range(c0, c1 + 1)}
    border.update((r1, c) for c in range(c0, c1 + 1))
    border.update((r, c0) for r in range(r0 + 1, r1))
    border.update((r, c1) for r in range(r0 + 1, r1))
    return border


def _extract_frames_and_seeds(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    components = _components4(g != BG)
    frames = []
    seeds = []
    for pts in components:
        if len(pts) == 1:
            seeds.append(pts[0])
            continue

        rows = [r for r, _ in pts]
        cols = [c for _, c in pts]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "non-seed component must be at least a 3x3 frame"

        border = _border_cells(r0, c0, r1, c1)
        assert len(border) == len(pts), "non-seed component must be a hollow rectangle"
        assert set(pts) == border, "non-seed component must exactly match its rectangular border"

        top = g[r0, c0 : c1 + 1]
        bottom = g[r1, c0 : c1 + 1]
        left = g[r0 + 1 : r1, c0]
        right = g[r0 + 1 : r1, c1]
        assert left.size > 0 and right.size > 0
        assert np.all(top == top[0]) and top[0] != BG
        assert np.all(bottom == bottom[0]) and bottom[0] != BG
        assert np.all(left == left[0]) and left[0] != BG
        assert np.all(right == right[0]) and right[0] != BG

        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "top": int(top[0]),
                "bottom": int(bottom[0]),
                "left": int(left[0]),
                "right": int(right[0]),
                "area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
            }
        )

    return frames, seeds


def _direct_frame(frames: list[dict], seed: tuple[int, int]) -> dict:
    sr, sc = seed
    containing = [
        frame
        for frame in frames
        if frame["bbox"][0] < sr < frame["bbox"][2] and frame["bbox"][1] < sc < frame["bbox"][3]
    ]
    assert containing, "every seed must be enclosed by a frame"
    return min(containing, key=lambda frame: frame["area"])


def _growth_mode(frame: dict) -> str:
    top = frame["top"]
    bottom = frame["bottom"]
    left = frame["left"]
    right = frame["right"]

    if top == bottom == left == right:
        return "fill"

    vertical_pair = top == bottom and top != left and top != right
    horizontal_pair = left == right and left != top and left != bottom

    # The prompt duplicates the opposite-pair case with conflicting wording.
    # We use the unique matching pair when only one axis is unambiguous; if both
    # axes match, we follow the first explicit clause and grow vertically.
    if vertical_pair and horizontal_pair:
        return "vertical"
    if vertical_pair:
        return "vertical"
    if horizontal_pair:
        return "horizontal"
    return "none"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each single-pixel seed according to the color symmetry of its direct frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames, seeds = _extract_frames_and_seeds(g)
    out = g.copy()

    for seed in seeds:
        sr, sc = seed
        frame = _direct_frame(frames, seed)
        mode = _growth_mode(frame)
        r0, c0, r1, c1 = frame["bbox"]
        seed_color = int(g[sr, sc])

        if mode == "fill":
            out[r0 + 1 : r1, c0 + 1 : c1] = seed_color
        elif mode == "vertical":
            out[r0 + 1 : r1, sc] = seed_color
        elif mode == "horizontal":
            out[sr, c0 + 1 : c1] = seed_color

    return out


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], sides: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    top, bottom, left, right = sides
    grid[r0, c0 : c1 + 1] = top
    grid[r1, c0 : c1 + 1] = bottom
    grid[r0 + 1 : r1, c0] = left
    grid[r0 + 1 : r1, c1] = right


def _random_outer_sides(rng: random.Random) -> tuple[int, int, int, int]:
    return tuple(rng.choice(PALETTE) for _ in range(4))


def _branch_sides(rng: random.Random, branch: str) -> tuple[int, int, int, int]:
    if branch == "fill":
        color = rng.choice(PALETTE)
        return color, color, color, color
    if branch == "vertical":
        a, b, c = rng.sample(PALETTE, 3)
        return a, a, b, c
    if branch == "horizontal":
        a, b, c = rng.sample(PALETTE, 3)
        return b, c, a, a

    pattern = rng.choice(("three_same", "all_distinct", "adjacent_pair"))
    if pattern == "three_same":
        a, b = rng.sample(PALETTE, 2)
        return a, a, a, b
    if pattern == "adjacent_pair":
        a, b, c = rng.sample(PALETTE, 3)
        return a, b, a, c
    a, b, c, d = rng.sample(PALETTE, 4)
    return a, b, c, d


def _make_group_spec(rng: random.Random, branch: str) -> dict:
    depth = rng.choices([1, 2, 3], weights=[3, 3, 1], k=1)[0]
    inner_h = rng.randint(*INNER_SIZE_RANGE)
    inner_w = rng.randint(*INNER_SIZE_RANGE)
    root_h = inner_h + 2
    root_w = inner_w + 2
    expansions: list[tuple[int, int, int, int]] = []

    for _ in range(depth - 1):
        gaps = tuple(rng.randint(*NEST_GAP_RANGE) for _ in range(4))
        gt, gb, gl, gr = gaps
        root_h += gt + gb + 2
        root_w += gl + gr + 2
        expansions.append(gaps)

    frame_sides = [_random_outer_sides(rng) for _ in range(depth - 1)]
    frame_sides.append(_branch_sides(rng, branch))

    return {
        "branch": branch,
        "depth": depth,
        "root_h": root_h,
        "root_w": root_w,
        "expansions": expansions,
        "frame_sides": frame_sides,
    }


def _frame_bboxes(spec: dict, origin: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    r0, c0 = origin
    bbox = (r0, c0, r0 + spec["root_h"] - 1, c0 + spec["root_w"] - 1)
    bboxes = [bbox]
    for gt, gb, gl, gr in reversed(spec["expansions"]):
        bbox = (bbox[0] + 1 + gt, bbox[1] + 1 + gl, bbox[2] - 1 - gb, bbox[3] - 1 - gr)
        bboxes.append(bbox)
    return bboxes


def _choose_seed_color(
    rng: random.Random, used_seed_colors: set[int], forbidden: set[int]
) -> int | None:
    choices = [color for color in PALETTE if color not in used_seed_colors and color not in forbidden]
    if not choices:
        return None
    color = rng.choice(choices)
    used_seed_colors.add(color)
    return color


def _branch_list(rng: random.Random, count: int) -> list[str]:
    order = list(BRANCHES)
    rng.shuffle(order)
    return order[:count]


def _place_specs(
    rng: random.Random, height: int, width: int, specs: list[dict]
) -> list[tuple[int, int]] | None:
    occupied = np.zeros((height, width), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(specs)
    order = sorted(range(len(specs)), key=lambda i: specs[i]["root_h"] * specs[i]["root_w"], reverse=True)

    for idx in order:
        spec = specs[idx]
        if spec["root_h"] > height or spec["root_w"] > width:
            return None

        placed = False
        for _ in range(PLACEMENT_TRIES):
            r0 = rng.randint(0, height - spec["root_h"])
            c0 = rng.randint(0, width - spec["root_w"])
            rr0 = max(0, r0 - 1)
            rr1 = min(height, r0 + spec["root_h"] + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(width, c0 + spec["root_w"] + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            occupied[rr0:rr1, cc0:cc1] = True
            placements[idx] = (r0, c0)
            placed = True
            break

        if not placed:
            return None

    return [placement for placement in placements if placement is not None]


def _validate_generated_grid(grid: np.ndarray, expected_branches: Counter) -> bool:
    try:
        frames, seeds = _extract_frames_and_seeds(grid)
        if not seeds:
            return False

        seed_colors = [int(grid[r, c]) for r, c in seeds]
        if len(seed_colors) != len(set(seed_colors)):
            return False

        actual = Counter()
        for seed in seeds:
            actual[_growth_mode(_direct_frame(frames, seed))] += 1
        if actual != expected_branches:
            return False

        _ = main(grid)
    except Exception:
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(GRID_MIN, GRID_MAX)
    width = rng.randint(GRID_MIN, GRID_MAX)
    group_count = rng.randint(*GROUP_COUNT_RANGE)
    branches = _branch_list(rng, group_count)
    specs = [_make_group_spec(rng, branch) for branch in branches]
    placements = _place_specs(rng, height, width, specs)
    if placements is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    used_seed_colors: set[int] = set()
    for spec, origin in zip(specs, placements):
        bboxes = _frame_bboxes(spec, origin)
        for bbox, sides in zip(bboxes, spec["frame_sides"]):
            _draw_frame(grid, bbox, sides)

        inner_bbox = bboxes[-1]
        seed_color = _choose_seed_color(rng, used_seed_colors, set(spec["frame_sides"][-1]))
        if seed_color is None:
            return None
        sr = rng.randint(inner_bbox[0] + 1, inner_bbox[2] - 1)
        sc = rng.randint(inner_bbox[1] + 1, inner_bbox[3] - 1)
        grid[sr, sc] = seed_color

    if not _validate_generated_grid(grid, Counter(branches)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
