import random
from collections import Counter

import numpy as np


BG = 0
OBJECT_COLORS = (1, 2, 3, 4, 7)
LEGEND_FRAME = 5
MARKER = 8
DIR_STEPS = {
    "down": (1, 0),
    "up": (-1, 0),
    "right": (0, 1),
    "left": (0, -1),
}

GEN_ATTEMPTS = 500
GEN_H_RANGE = (15, 20)
GEN_W_RANGE = (15, 20)
GEN_OBJECT_COUNT = (5, 8)
GEN_SHAPE_SIZE = (3, 7)
GEN_LEGEND_OUTER = (4, 5)


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def detect_legend(grid: np.ndarray) -> tuple[tuple[int, int, int, int], set[int]]:
    grey_comps = components_mask(grid == LEGEND_FRAME)
    candidates = []
    for coords in grey_comps:
        r0, c0, r1, c1 = bbox_coords(coords)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        border = {
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)
        }
        if border != set(map(tuple, coords.tolist())):
            continue
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        colors = set(int(x) for x in np.unique(inner) if int(x) not in (BG, LEGEND_FRAME))
        if 2 <= len(colors) <= 3:
            candidates.append(((r0, c0, r1, c1), colors))
    assert candidates, "expected a framed legend"
    candidates.sort(key=lambda item: (item[0][0] + item[0][1], item[0][0], item[0][1]))
    return candidates[0]


def detect_marker(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    pts = np.argwhere(grid == MARKER)
    assert len(pts) == 1, "expected a single edge marker"
    r, c = map(int, pts[0])
    h, w = grid.shape
    assert (r in (0, h - 1) or c in (0, w - 1)) and not (
        r in (0, h - 1) and c in (0, w - 1)
    ), "marker must be on a non-corner edge"
    if r == 0:
        return (r, c), DIR_STEPS["down"]
    if r == h - 1:
        return (r, c), DIR_STEPS["up"]
    if c == 0:
        return (r, c), DIR_STEPS["right"]
    return (r, c), DIR_STEPS["left"]


def parse_objects(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    legend_bbox, target_colors = detect_legend(g)
    marker_pos, main_step = detect_marker(g)
    opp_step = (-main_step[0], -main_step[1])

    mask = (g != BG) & (g != LEGEND_FRAME) & (g != MARKER)
    r0, c0, r1, c1 = legend_bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = False
    mask[marker_pos] = False

    objects = []
    for coords in components_mask(mask):
        colors = np.asarray([int(g[r, c]) for r, c in coords], dtype=int)
        bbox = bbox_coords(coords)
        objects.append(
            {
                "coords": coords.copy(),
                "colors": colors,
                "bbox": bbox,
                "color_set": set(map(int, colors.tolist())),
            }
        )

    return {
        "grid": g,
        "legend_bbox": legend_bbox,
        "target_colors": set(target_colors),
        "marker_pos": marker_pos,
        "main_step": main_step,
        "opp_step": opp_step,
        "objects": objects,
    }


def edge_distance(bbox: tuple[int, int, int, int], step: tuple[int, int], shape: tuple[int, int]) -> int:
    r0, c0, r1, c1 = bbox
    h, w = shape
    if step == DIR_STEPS["down"]:
        return h - 1 - r1
    if step == DIR_STEPS["up"]:
        return r0
    if step == DIR_STEPS["right"]:
        return w - 1 - c1
    return c0


def _shift_until_blocked(
    coords: np.ndarray, step: tuple[int, int], occupied: set[tuple[int, int]], shape: tuple[int, int]
) -> np.ndarray:
    h, w = shape
    cur = coords.copy()
    while True:
        nxt = cur + np.asarray(step, dtype=int)
        ok = True
        for nr, nc in nxt:
            rr = int(nr)
            cc = int(nc)
            if rr < 0 or rr >= h or cc < 0 or cc >= w or (rr, cc) in occupied:
                ok = False
                break
        if not ok:
            return cur
        cur = nxt


def settle_objects(
    objects: list[dict], step: tuple[int, int], shape: tuple[int, int], *, reverse: bool = False
) -> list[dict]:
    ordered = sorted(
        objects,
        key=lambda obj: (
            edge_distance(obj["bbox"], step, shape),
            obj["bbox"][0],
            obj["bbox"][1],
            obj["bbox"][2],
            obj["bbox"][3],
        ),
        reverse=reverse,
    )
    occupied: set[tuple[int, int]] = set()
    placed = []
    for obj in ordered:
        final_coords = _shift_until_blocked(obj["coords"], step, occupied, shape)
        for r, c in final_coords:
            occupied.add((int(r), int(c)))
        placed.append(
            {
                "start_coords": obj["coords"],
                "coords": final_coords,
                "colors": obj["colors"],
                "bbox": bbox_coords(final_coords),
                "color_set": set(obj["color_set"]),
                "moved": tuple(map(int, final_coords[0] - obj["coords"][0])),
            }
        )
    return placed


def render_objects(shape: tuple[int, int], placed_groups: list[list[dict]]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    occupied: set[tuple[int, int]] = set()
    for group in placed_groups:
        for obj in group:
            for (r, c), color in zip(obj["coords"], obj["colors"].tolist()):
                rr = int(r)
                cc = int(c)
                assert (rr, cc) not in occupied, "placed groups must remain disjoint"
                occupied.add((rr, cc))
                out[rr, cc] = int(color)
    return out


def solve_from_info(info: dict, *, reverse: bool = False) -> np.ndarray:
    target_colors = info["target_colors"]
    matches = [obj for obj in info["objects"] if target_colors <= obj["color_set"]]
    non_matches = [obj for obj in info["objects"] if not (target_colors <= obj["color_set"])]
    placed_matches = settle_objects(matches, info["main_step"], info["grid"].shape, reverse=reverse)
    placed_non = settle_objects(non_matches, info["opp_step"], info["grid"].shape, reverse=reverse)
    return render_objects(info["grid"].shape, [placed_matches, placed_non])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Separate scattered multicolor objects toward opposite edges based on whether they fully match the legend colors.

    Input structure:
    - The background is black.
    - A small grey rectangular frame in the upper-left encloses 2 or 3 legend colors.
    - One cyan edge marker indicates the primary gravity direction, and the remaining colored cells form disjoint 4-connected objects.

    Rule:
    - Read the set S of non-background, non-grey colors inside the legend frame.
    - For each foreground object, test whether its color set is a superset of S.
    - Objects that contain all legend colors fall rigidly in the marker-indicated inward direction; objects that do not fall rigidly in the opposite direction.
    - Within each class, process objects from nearest to the destination edge to farthest, stop against the grid edge or already placed same-class objects, and remove the legend plus the marker from the output.

    Key insight:
    - The legend is not an erase mask: it partitions objects into two gravity classes, while the edge marker selects which class moves toward which side.
    """
    info = parse_objects(grid)
    return solve_from_info(info, reverse=False)


def random_connected_shape(size: int) -> np.ndarray:
    pts = {(0, 0)}
    frontier = {(0, 0)}
    while len(pts) < size:
        base_r, base_c = random.choice(tuple(frontier))
        nbrs = [(base_r + dr, base_c + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        random.shuffle(nbrs)
        added = False
        for nr, nc in nbrs:
            if (nr, nc) not in pts:
                pts.add((nr, nc))
                frontier.add((nr, nc))
                added = True
                break
        if not added:
            frontier.remove((base_r, base_c))
            if not frontier:
                frontier = set(pts)
    arr = np.asarray(sorted(pts), dtype=int)
    arr[:, 0] -= arr[:, 0].min()
    arr[:, 1] -= arr[:, 1].min()
    return arr


def assign_colors(shape: np.ndarray, target_colors: list[int], is_match: bool, mode: str) -> np.ndarray:
    size = len(shape)
    if is_match:
        extras = [c for c in OBJECT_COLORS if c not in target_colors]
        if mode == "extra" and extras and size >= len(target_colors) + 1:
            palette = target_colors + [random.choice(extras)]
        else:
            k = random.randint(len(target_colors), min(len(OBJECT_COLORS), len(target_colors) + 2))
            pool = list(target_colors)
            more = [c for c in OBJECT_COLORS if c not in pool]
            random.shuffle(more)
            palette = pool + more[: max(0, k - len(pool))]
        colors = list(target_colors)
        while len(colors) < size:
            colors.append(random.choice(palette))
        random.shuffle(colors)
        return np.asarray(colors[:size], dtype=int)

    target_subset = list(target_colors)
    random.shuffle(target_subset)
    if mode == "partial":
        keep_count = random.randint(1, len(target_colors) - 1)
        kept = target_subset[:keep_count]
    else:
        keep_count = random.randint(0, len(target_colors) - 1)
        kept = target_subset[:keep_count]
    palette = list(kept)
    extras = [c for c in OBJECT_COLORS if c not in target_colors]
    extra_count = random.randint(1, max(1, min(2, len(extras))))
    random.shuffle(extras)
    palette.extend(extras[:extra_count])
    if not palette:
        palette = extras[:1]
    colors = list(kept)
    while len(colors) < size:
        colors.append(random.choice(palette))
    random.shuffle(colors)
    return np.asarray(colors[:size], dtype=int)


def _expand_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    blocked[nr, nc] = True


def place_object(
    blocked: np.ndarray,
    shape: np.ndarray,
    *,
    fixed_axis: int | None,
    fixed_value: int | None,
) -> np.ndarray | None:
    h, w = blocked.shape
    sh = int(shape[:, 0].max()) + 1
    sw = int(shape[:, 1].max()) + 1
    row_choices = list(range(2, h - sh - 1))
    col_choices = list(range(2, w - sw - 1))
    if not row_choices or not col_choices:
        return None
    if fixed_axis == 0 and fixed_value is not None:
        row_choices = [r for r in row_choices if abs(r - fixed_value) <= 1]
        if not row_choices:
            row_choices = [fixed_value] if 2 <= fixed_value <= h - sh - 2 else []
    if fixed_axis == 1 and fixed_value is not None:
        col_choices = [c for c in col_choices if abs(c - fixed_value) <= 1]
        if not col_choices:
            col_choices = [fixed_value] if 2 <= fixed_value <= w - sw - 2 else []
    random.shuffle(row_choices)
    random.shuffle(col_choices)
    tries = [(r, c) for r in row_choices for c in col_choices]
    random.shuffle(tries)
    for top, left in tries[:200]:
        abs_coords = shape + np.asarray([top, left], dtype=int)
        if any(blocked[int(r), int(c)] for r, c in abs_coords):
            continue
        return abs_coords
    return None


def interval_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _classify(objects: list[dict], target_colors: set[int]) -> tuple[list[dict], list[dict]]:
    matches = [obj for obj in objects if target_colors <= obj["color_set"]]
    non_matches = [obj for obj in objects if not (target_colors <= obj["color_set"])]
    return matches, non_matches


def semantic_checks(grid: np.ndarray) -> bool:
    info = parse_objects(grid)
    objects = info["objects"]
    target_colors = info["target_colors"]
    matches, non_matches = _classify(objects, target_colors)
    if not (5 <= len(objects) <= 8):
        return False
    if len(matches) < 2 or len(non_matches) < 2:
        return False
    if not any(obj["color_set"] > target_colors for obj in matches):
        return False
    if not any(obj["color_set"] & target_colors and not (target_colors <= obj["color_set"]) for obj in non_matches):
        return False
    if any(len(obj["coords"]) < 3 or len(obj["coords"]) > 7 for obj in objects):
        return False

    main_step = info["main_step"]
    perp = 1 if main_step[0] else 0
    lane_overlap = False
    for cls in (matches, non_matches):
        dists = [edge_distance(obj["bbox"], main_step if cls is matches else info["opp_step"], grid.shape) for obj in cls]
        if len(dists) != len(set(dists)):
            return False
        for i in range(len(cls)):
            for j in range(i + 1, len(cls)):
                a = cls[i]["bbox"]
                b = cls[j]["bbox"]
                if perp == 1 and interval_overlap(a[1], a[3], b[1], b[3]):
                    lane_overlap = True
                if perp == 0 and interval_overlap(a[0], a[2], b[0], b[2]):
                    lane_overlap = True
    if not lane_overlap:
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False

    placed_matches = settle_objects(matches, info["main_step"], grid.shape, reverse=False)
    placed_non = settle_objects(non_matches, info["opp_step"], grid.shape, reverse=False)
    if any(obj["moved"] == (0, 0) for obj in placed_matches + placed_non):
        return False
    if np.count_nonzero(out) != sum(len(obj["coords"]) for obj in objects):
        return False

    try:
        reverse_out = solve_from_info(info, reverse=True)
    except AssertionError:
        reverse_out = None
    if reverse_out is None or np.array_equal(reverse_out, out):
        return False

    exact_matches = [obj for obj in objects if obj["color_set"] == target_colors]
    exact_non = [obj for obj in objects if obj["color_set"] != target_colors]
    try:
        exact_out = render_objects(
            grid.shape,
            [
                settle_objects(exact_matches, info["main_step"], grid.shape, reverse=False),
                settle_objects(exact_non, info["opp_step"], grid.shape, reverse=False),
            ],
        )
    except AssertionError:
        exact_out = None
    if exact_out is not None and np.array_equal(exact_out, out):
        return False

    any_matches = [obj for obj in objects if obj["color_set"] & target_colors]
    any_non = [obj for obj in objects if not (obj["color_set"] & target_colors)]
    try:
        any_out = render_objects(
            grid.shape,
            [
                settle_objects(any_matches, info["main_step"], grid.shape, reverse=False),
                settle_objects(any_non, info["opp_step"], grid.shape, reverse=False),
            ],
        )
    except AssertionError:
        any_out = None
    if any_out is not None and np.array_equal(any_out, out):
        return False
    return True


def _build_attempt() -> np.ndarray | None:
    h = random.randint(*GEN_H_RANGE)
    w = random.randint(*GEN_W_RANGE)
    grid = np.full((h, w), BG, dtype=int)

    legend_outer = random.choice(GEN_LEGEND_OUTER)
    legend_r0, legend_c0 = 1, 1
    legend_r1 = legend_r0 + legend_outer - 1
    legend_c1 = legend_c0 + legend_outer - 1
    grid[legend_r0 : legend_r1 + 1, legend_c0] = LEGEND_FRAME
    grid[legend_r0 : legend_r1 + 1, legend_c1] = LEGEND_FRAME
    grid[legend_r0, legend_c0 : legend_c1 + 1] = LEGEND_FRAME
    grid[legend_r1, legend_c0 : legend_c1 + 1] = LEGEND_FRAME

    target_count = random.choice((2, 3))
    target_colors = random.sample(list(OBJECT_COLORS), target_count)
    inner_h = legend_outer - 2
    inner_w = legend_outer - 2
    inner_cells = [(r, c) for r in range(legend_r0 + 1, legend_r1) for c in range(legend_c0 + 1, legend_c1)]
    fill_colors = list(target_colors)
    while len(fill_colors) < len(inner_cells):
        fill_colors.append(random.choice(target_colors))
    random.shuffle(fill_colors)
    for (r, c), color in zip(inner_cells, fill_colors):
        grid[r, c] = color

    side = random.choice(("top", "bottom", "left", "right"))
    if side == "top":
        choices = [c for c in range(1, w - 1) if abs(c - ((legend_c0 + legend_c1) // 2)) > legend_outer]
        if not choices:
            return None
        marker_pos = (0, random.choice(choices))
    elif side == "bottom":
        marker_pos = (h - 1, random.randint(1, w - 2))
    elif side == "left":
        choices = [r for r in range(1, h - 1) if abs(r - ((legend_r0 + legend_r1) // 2)) > legend_outer]
        if not choices:
            return None
        marker_pos = (random.choice(choices), 0)
    else:
        marker_pos = (random.randint(1, h - 2), w - 1)
    grid[marker_pos] = MARKER

    blocked = np.zeros((h, w), dtype=bool)
    legend_cells = [
        (r, c)
        for r in range(legend_r0, legend_r1 + 1)
        for c in range(legend_c0, legend_c1 + 1)
    ]
    _expand_blocked(blocked, legend_cells + [marker_pos])

    object_count = random.randint(*GEN_OBJECT_COUNT)
    match_count = random.randint(2, object_count - 2)
    labels = [True] * match_count + [False] * (object_count - match_count)
    random.shuffle(labels)

    force_class = random.choice((True, False))
    forced_indices = [i for i, flag in enumerate(labels) if flag == force_class][:2]
    if len(forced_indices) < 2:
        return None

    special_match_idx = next((i for i, flag in enumerate(labels) if flag), None)
    partial_non_idx = next((i for i, flag in enumerate(labels) if not flag), None)
    if special_match_idx is None or partial_non_idx is None:
        return None

    vertical_motion = side in ("top", "bottom")
    fixed_axis = 1 if vertical_motion else 0
    if fixed_axis == 1:
        fixed_value = random.randint(max(3, legend_c1 + 2), max(3, min(w - 6, w // 2 + 2)))
    else:
        fixed_value = random.randint(max(3, legend_r1 + 2), max(3, min(h - 6, h // 2 + 2)))

    objects = []
    for idx, is_match in enumerate(labels):
        size = random.randint(*GEN_SHAPE_SIZE)
        shape = random_connected_shape(size)
        mode = ""
        if idx == special_match_idx:
            mode = "extra"
        if idx == partial_non_idx:
            mode = "partial"
        colors = assign_colors(shape, target_colors, is_match, mode)
        coords = place_object(
            blocked,
            shape,
            fixed_axis=fixed_axis if idx in forced_indices else None,
            fixed_value=fixed_value if idx in forced_indices else None,
        )
        if coords is None:
            return None
        for (r, c), color in zip(coords.tolist(), colors.tolist()):
            grid[r, c] = color
        _expand_blocked(blocked, [tuple(map(int, rc)) for rc in coords.tolist()])
        objects.append({"coords": coords, "colors": colors, "is_match": is_match})

    if Counter(grid.ravel())[BG] <= (h * w) // 2:
        return None
    if not semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
