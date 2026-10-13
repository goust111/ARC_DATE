import numpy as np

from arc_utils import find_components, paste
import random


def _bbox_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and inner[2] <= outer[2] and inner[3] <= outer[3]


def _wholly_in_three_sided_cavity(container: dict, inner: dict) -> bool:
    r0, c0, r1, c1 = container["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    coords = container["coords"]
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True

    for rr, cc in inner["coords"]:
        r = int(rr - r0)
        c = int(cc - c0)
        if not (0 <= r < mask.shape[0] and 0 <= c < mask.shape[1]) or mask[r, c]:
            return False
        support = (
            int(np.any(mask[:r, c]))
            + int(np.any(mask[r + 1 :, c]))
            + int(np.any(mask[r, :c]))
            + int(np.any(mask[r, c + 1 :]))
        )
        if support < 3:
            return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Delete each foreign-color object whose bounding box is contained by a larger object of another color.

    Input structure
    - The black pixels are the empty canvas, and every other color forms one or more 8-connected monochromatic objects.
    - Some larger foreground objects act as loose U/C-like containers around smaller objects of different colors.
    Rule
    - Find all 8-connected monochromatic non-black objects.
    - Recolor an object to black exactly when its bounding box is contained in the bounding box of a larger object of another color.
    - Keep all other pixels unchanged.
    Key insight
    - The removable objects are identified by containment in another color's rectangular extent, not by their own color or shape.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert np.any(g == 0)

    bg = 0
    objects = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    assert objects

    out = g.copy()
    removed_any = False
    for obj in objects:
        contained = [
            other
            for other in objects
            if other is not obj
            and other["color"] != obj["color"]
            and other["area"] > obj["area"]
            and _bbox_contains(other["bbox"], obj["bbox"])
        ]
        if contained:
            assert any(_wholly_in_three_sided_cavity(other, obj) for other in contained)
            coords = obj["coords"]
            out[coords[:, 0], coords[:, 1]] = bg
            removed_any = True

    assert removed_any
    return out

# --------------- generator (att2) ---------------

# Canvas and palette ranges.
_GEN_SMALL_GRID_PROB = 0.18
_GEN_SMALL_GRID_SIZE_RANGE, _GEN_LARGE_GRID_SIZE_RANGE = (3, 5), (10, 15)
_GEN_PALETTE_SIZE_RANGE, _GEN_FOREGROUND_COLORS = (2, 5), range(1, 10)
# Container shape constraints and retry limits.
_GEN_CONTAINER_COUNT_PROB = 0.45
_GEN_CONTAINER_MIN_GRID_AREA, _GEN_CONTAINER_MIN_SIDE_FOR_PAIR = 36, 4
_GEN_CONTAINER_AREA_RANGE, _GEN_CONTAINER_FILL_RATIO_RANGE = (7, 21), (0.25, 0.78)
_GEN_ARCHETYPE_TRIES = 8

# Inner object counts and size limits.
_GEN_TWO_CONTAINER_EXTRA_INNER_PROB = 0.45
_GEN_INNER_COUNT_RANGE, _GEN_INNER_AREA_MAX = (1, 2), 4
_GEN_INNER_SPRITE_TRIES = 12
# Standalone and adjacent distractor controls.
_GEN_OUTSIDE_ADJACENT_MIN_AREA, _GEN_OUTSIDE_ADJACENT_PROB = 80, 0.25
_GEN_ADJACENT_AREA_RANGE = range(2, 6)
_GEN_ADJACENT_SPRITE_SIZE_MAX, _GEN_ADJACENT_SPRITE_TRIES = (3, 4), 12
_GEN_DENSE_GRID_MIN_AREA, _GEN_DENSE_STANDALONE_PROB, _GEN_DENSE_DISTRACTOR_PROB = 120, 0.12, 0.08
_GEN_DENSE_STANDALONE_COUNT_RANGE = (6, 9)
_GEN_STANDALONE_COUNT_CHOICES, _GEN_STANDALONE_COUNT_WEIGHTS = [0, 1, 2, 3], [35, 30, 25, 10]
_GEN_SMALL_DISTRACTOR_AREA_MAX, _GEN_LARGE_DISTRACTOR_AREA_MAX = 3, 11
_GEN_DISTRACTOR_SPRITE_TRIES = 10
# Geometric thresholds and final retry limit.
_GEN_STEP_DIAGONAL_BREAK_PROB, _GEN_CAVITY_SUPPORT_MIN, _GEN_MAX_ATTEMPTS = 0.35, 3, 300


def _gen_sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _gen_bbox_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and inner[2] <= outer[2] and inner[3] <= outer[3]


def _gen_bboxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _gen_abs_bbox(r: int, c: int, patch: np.ndarray) -> tuple[int, int, int, int]:
    return int(r), int(c), int(r + patch.shape[0] - 1), int(c + patch.shape[1] - 1)


def _gen_step_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r, c = map(int, start)
    er, ec = map(int, end)
    mask[r, c] = True
    while (r, c) != (er, ec):
        dr = _gen_sign(er - r)
        dc = _gen_sign(ec - c)
        if dr and dc and random.random() < _GEN_STEP_DIAGONAL_BREAK_PROB:
            if random.random() < 0.5:
                dc = 0
            else:
                dr = 0
        r += dr
        c += dc
        mask[r, c] = True


def _gen_grow_mask(mask: np.ndarray, target_area: int) -> bool:
    h, w = mask.shape
    while int(mask.sum()) < int(target_area):
        frontier: list[tuple[int, int]] = []
        for r, c in np.argwhere(mask):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = int(r + dr), int(c + dc)
                    if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc]:
                        frontier.append((rr, cc))
        if not frontier:
            return False
        mask[random.choice(frontier)] = True
    return True


def _gen_support_count(mask: np.ndarray, r: int, c: int) -> int:
    return (
        int(np.any(mask[:r, c]))
        + int(np.any(mask[r + 1 :, c]))
        + int(np.any(mask[r, :c]))
        + int(np.any(mask[r, c + 1 :]))
    )


def _gen_cavity_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    return [
        (int(r), int(c))
        for r, c in np.argwhere(~mask)
        if _gen_support_count(mask, int(r), int(c)) >= _GEN_CAVITY_SUPPORT_MIN
    ]


_GEN_CONTAINER_ARCHETYPES = (
    ("#.#", "#.#", "###"),
    ("..#####", "##.....", "##.....", ".#####."),
    ("..###", "##...", "#....", "#####"),
    ("..#...", ".#....", "#.....", "#....#", ".#..#.", "..##.."),
    ("..##..", ".#..#.", "#....#", "....#.", "...#..", "..#..."),
    ("..##", "##..", "##..", ".###"),
    (".#######.", "##.....#.", "#......##", "#......##", "#......##", "#......#."),
    (".##..", "#..#.", "#...#", ".#...", "..#.."),
)


def _gen_mask_from_rows(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[ch == "#" for ch in row] for row in rows], dtype=bool)


def _gen_container_dims() -> list[tuple[int, int]]:
    return [(len(rows), len(rows[0])) for rows in _GEN_CONTAINER_ARCHETYPES]


def _gen_archetype_mask(rows: tuple[str, ...]) -> np.ndarray:
    mask = _gen_mask_from_rows(rows)
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    if random.random() < 0.5:
        mask = np.flipud(mask)
    return mask.copy()


def _gen_sparse_container(h: int, w: int, color: int) -> np.ndarray | None:
    min_area, max_area = _GEN_CONTAINER_AREA_RANGE
    min_fill, max_fill = _GEN_CONTAINER_FILL_RATIO_RANGE
    lo = max(min_area, int(np.ceil(min_fill * h * w)))
    hi = min(max_area, int(np.floor(max_fill * h * w)))
    if lo > hi:
        return None

    choices = [rows for rows in _GEN_CONTAINER_ARCHETYPES if len(rows) == h and len(rows[0]) == w]
    random.shuffle(choices)
    for rows in choices:
        for _ in range(_GEN_ARCHETYPE_TRIES):
            mask = _gen_archetype_mask(rows)
            area = int(mask.sum())
            if not (lo <= area <= hi):
                continue
            if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
                continue
            if not _gen_cavity_cells(mask):
                continue
            patch = np.zeros((h, w), dtype=int)
            patch[mask] = int(color)
            return patch
    return None


def _gen_small_sprite(area: int, color: int, max_h: int, max_w: int) -> np.ndarray | None:
    area = int(area)
    if area < 1:
        return None
    dims = []
    for h in range(1, min(max_h, area) + 1):
        for w in range(1, min(max_w, area) + 1):
            if h * w >= area and max(h, w) <= area:
                dims.append((h, w))
    if not dims:
        return None
    h, w = random.choice(dims)
    mask = np.zeros((h, w), dtype=bool)
    _gen_step_path(mask, (0, 0), (h - 1, w - 1))
    if int(mask.sum()) > area:
        return None
    if not _gen_grow_mask(mask, area):
        return None
    patch = np.zeros((h, w), dtype=int)
    patch[mask] = int(color)
    return patch


def _gen_no_same_color_touch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, color: int) -> bool:
    h, w = patch.shape
    mask = patch != 0
    for pr, pc in np.argwhere(mask):
        ar, ac = int(r0 + pr), int(c0 + pc)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = ar + dr, ac + dc
                if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]:
                    if not (r0 <= rr < r0 + h and c0 <= cc < c0 + w) and int(grid[rr, cc]) == int(color):
                        return False
    return True


def _gen_place_patch(
    grid: np.ndarray,
    patch: np.ndarray,
    occupied_bboxes: list[tuple[int, int, int, int]],
    *,
    forbidden_bboxes: list[tuple[int, int, int, int]] | None = None,
) -> tuple[int, int, tuple[int, int, int, int]] | None:
    forbidden_bboxes = [] if forbidden_bboxes is None else forbidden_bboxes
    h, w = patch.shape
    color = int(np.max(patch))
    positions = [(r, c) for r in range(grid.shape[0] - h + 1) for c in range(grid.shape[1] - w + 1)]
    random.shuffle(positions)
    for r, c in positions:
        bb = _gen_abs_bbox(r, c, patch)
        if any(_gen_bboxes_overlap(bb, old) for old in occupied_bboxes):
            continue
        if any(_gen_bboxes_overlap(bb, old) for old in forbidden_bboxes):
            continue
        view = grid[r : r + h, c : c + w]
        if np.any((patch != 0) & (view != 0)):
            continue
        if not _gen_no_same_color_touch(grid, patch, r, c, color):
            continue
        paste(grid, patch, r, c, transparent=0)
        return int(r), int(c), bb
    return None


def _gen_place_container(
    grid: np.ndarray,
    color: int,
    occupied_bboxes: list[tuple[int, int, int, int]],
) -> dict | None:
    dims = [
        (h, w)
        for h, w in _gen_container_dims()
        if h <= grid.shape[0] and w <= grid.shape[1]
    ]
    random.shuffle(dims)
    for h, w in dims:
        patch = _gen_sparse_container(h, w, color)
        if patch is None:
            continue
        placed = _gen_place_patch(grid, patch, occupied_bboxes)
        if placed is None:
            continue
        r, c, bb = placed
        return {"bbox": bb, "area": int(np.count_nonzero(patch)), "color": int(color), "r": r, "c": c}
    return None


def _gen_object_infos(grid: np.ndarray) -> list[dict]:
    return find_components(grid, bg=0, connectivity=8, monochromatic=True, return_format="dict")


def _gen_has_unwanted_removal(grid: np.ndarray) -> bool:
    objs = _gen_object_infos(grid)
    for obj in objs:
        for other in objs:
            if other is obj:
                continue
            if other["color"] != obj["color"] and other["area"] > obj["area"]:
                if _gen_bbox_contains(other["bbox"], obj["bbox"]):
                    return True
    return False


def _gen_place_distractor(
    grid: np.ndarray,
    color: int,
    occupied_bboxes: list[tuple[int, int, int, int]],
    remaining_after: int,
) -> bool:
    free_cells = [
        (r, c)
        for r in range(grid.shape[0])
        for c in range(grid.shape[1])
        if grid[r, c] == 0 and not any(_gen_bbox_contains(bb, (r, c, r, c)) for bb in occupied_bboxes)
    ]
    cluttered_large_grid = grid.shape[0] * grid.shape[1] >= _GEN_DENSE_GRID_MIN_AREA and random.random() < _GEN_DENSE_DISTRACTOR_PROB
    area_cap = _GEN_LARGE_DISTRACTOR_AREA_MAX if cluttered_large_grid else _GEN_SMALL_DISTRACTOR_AREA_MAX
    max_area = min(area_cap, len(free_cells) - int(remaining_after))
    if max_area < 1:
        return False

    small_areas = list(range(1, min(_GEN_SMALL_DISTRACTOR_AREA_MAX, max_area) + 1))
    random.shuffle(small_areas)
    large_areas = list(range(4, max_area + 1))
    random.shuffle(large_areas)
    areas = small_areas + large_areas
    for area in areas:
        for _ in range(_GEN_DISTRACTOR_SPRITE_TRIES):
            patch = _gen_small_sprite(area, color, min(grid.shape[0], area), min(grid.shape[1], area))
            if patch is None:
                continue
            before = grid.copy()
            placed = _gen_place_patch(grid, patch, occupied_bboxes)
            if placed is None:
                continue
            if _gen_has_unwanted_removal(grid):
                grid[:, :] = before
                continue
            occupied_bboxes.append(placed[2])
            return True
    return False


def _gen_cell_in_container_cavity(grid: np.ndarray, container: dict, r: int, c: int) -> bool:
    r0, c0, r1, c1 = container["bbox"]
    lr, lc = int(r - r0), int(c - c0)
    if not (0 <= lr <= r1 - r0 and 0 <= lc <= c1 - c0):
        return False
    container_mask = grid[r0 : r1 + 1, c0 : c1 + 1] == int(container["color"])
    return not container_mask[lr, lc] and _gen_support_count(container_mask, lr, lc) >= _GEN_CAVITY_SUPPORT_MIN


def _gen_patch_touches_container(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, container: dict) -> bool:
    r_min, c_min, r_max, c_max = container["bbox"]
    for pr, pc in np.argwhere(patch != 0):
        ar, ac = int(r0 + pr), int(c0 + pc)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = ar + dr, ac + dc
                if r_min <= rr <= r_max and c_min <= cc <= c_max and int(grid[rr, cc]) == int(container["color"]):
                    return True
    return False


def _gen_place_outside_adjacent_distractor(
    grid: np.ndarray,
    color: int,
    occupied_bboxes: list[tuple[int, int, int, int]],
    containers: list[dict],
) -> bool:
    targets = [container for container in containers if int(container["color"]) != int(color)]
    random.shuffle(targets)
    for container in targets:
        target_bbox = container["bbox"]
        for area in random.sample(_GEN_ADJACENT_AREA_RANGE, len(_GEN_ADJACENT_AREA_RANGE)):
            for _ in range(_GEN_ADJACENT_SPRITE_TRIES):
                max_h, max_w = _GEN_ADJACENT_SPRITE_SIZE_MAX
                patch = _gen_small_sprite(area, color, min(max_h, area), min(max_w, area))
                if patch is None:
                    continue
                h, w = patch.shape
                positions = [(r, c) for r in range(grid.shape[0] - h + 1) for c in range(grid.shape[1] - w + 1)]
                random.shuffle(positions)
                for r, c in positions:
                    bb = _gen_abs_bbox(r, c, patch)
                    if _gen_bbox_contains(target_bbox, bb):
                        continue
                    if any(old != target_bbox and _gen_bboxes_overlap(bb, old) for old in occupied_bboxes):
                        continue
                    if not _gen_patch_touches_container(grid, patch, r, c, container):
                        continue
                    if any(
                        _gen_cell_in_container_cavity(grid, other, int(r + pr), int(c + pc))
                        for other in containers
                        for pr, pc in np.argwhere(patch != 0)
                    ):
                        continue

                    view = grid[r : r + h, c : c + w]
                    if np.any((patch != 0) & (view != 0)):
                        continue
                    if not _gen_no_same_color_touch(grid, patch, r, c, color):
                        continue
                    before = grid.copy()
                    paste(grid, patch, r, c, transparent=0)
                    if _gen_has_unwanted_removal(grid):
                        grid[:, :] = before
                        continue
                    occupied_bboxes.append(bb)
                    return True
    return False


def _gen_place_inner(
    grid: np.ndarray,
    container: dict,
    color: int,
    remaining_in_container: int,
) -> bool:
    r0, c0, r1, c1 = container["bbox"]
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    if inner_h <= 0 or inner_w <= 0:
        return False

    free_inside = [
        (r, c)
        for r in range(r0 + 1, r1)
        for c in range(c0 + 1, c1)
        if grid[r, c] == 0
    ]
    max_area = min(_GEN_INNER_AREA_MAX, container["area"] - 1, len(free_inside) - int(remaining_in_container))
    if max_area < 1:
        return False

    container_mask = grid[r0 : r1 + 1, c0 : c1 + 1] == int(container["color"])

    areas = list(range(1, max_area + 1))
    random.shuffle(areas)
    for area in areas:
        for _ in range(_GEN_INNER_SPRITE_TRIES):
            patch = _gen_small_sprite(area, color, min(inner_h, area), min(inner_w, area))
            if patch is None:
                continue
            h, w = patch.shape
            positions = [
                (r, c)
                for r in range(r0 + 1, r1 - h + 1)
                for c in range(c0 + 1, c1 - w + 1)
            ]
            random.shuffle(positions)
            for r, c in positions:
                view = grid[r : r + h, c : c + w]
                if np.any((patch != 0) & (view != 0)):
                    continue
                if not all(
                    _gen_support_count(container_mask, int(r + pr - r0), int(c + pc - c0)) >= _GEN_CAVITY_SUPPORT_MIN
                    for pr, pc in np.argwhere(patch != 0)
                ):
                    continue
                if not _gen_no_same_color_touch(grid, patch, r, c, color):
                    continue
                paste(grid, patch, r, c, transparent=0)
                return True
    return False


def _gen_canvas_shape() -> tuple[int, int]:
    if random.random() < _GEN_SMALL_GRID_PROB:
        return random.randint(*_GEN_SMALL_GRID_SIZE_RANGE), random.randint(*_GEN_SMALL_GRID_SIZE_RANGE)
    return random.randint(*_GEN_LARGE_GRID_SIZE_RANGE), random.randint(*_GEN_LARGE_GRID_SIZE_RANGE)


def _gen_container_count(height: int, width: int) -> int:
    return 2 if (
        height * width >= _GEN_CONTAINER_MIN_GRID_AREA
        and min(height, width) >= _GEN_CONTAINER_MIN_SIDE_FOR_PAIR
        and random.random() < _GEN_CONTAINER_COUNT_PROB
    ) else 1


def _gen_place_containers(
    out: np.ndarray,
    palette: list[int],
    container_count: int,
    occupied_bboxes: list[tuple[int, int, int, int]],
) -> list[dict] | None:
    containers: list[dict] = []
    for i in range(container_count):
        placed = _gen_place_container(out, palette[i % len(palette)], occupied_bboxes)
        if placed is None:
            return None
        containers.append(placed)
        occupied_bboxes.append(placed["bbox"])
    return containers


def _gen_inner_counts(container_count: int) -> list[int]:
    if container_count == 1:
        return [random.randint(*_GEN_INNER_COUNT_RANGE)]
    if random.random() < 1.0 - _GEN_TWO_CONTAINER_EXTRA_INNER_PROB:
        return [1, 1]
    inner_counts = [1, 1]
    inner_counts[random.randrange(2)] = 2
    return inner_counts


def _gen_plan_inner_objects(
    palette: list[int],
    containers: list[dict],
    inner_counts: list[int],
) -> tuple[list[tuple[int, int]], set[int]]:
    present_colors = {c["color"] for c in containers}
    inner_plan: list[tuple[int, int]] = []
    unused = [c for c in palette if c not in present_colors]
    for ci, n_inner in enumerate(inner_counts):
        container_color = containers[ci]["color"]
        for _ in range(n_inner):
            choices = [c for c in palette if c != container_color]
            preferred = [c for c in unused if c in choices]
            color = random.choice(preferred if preferred else choices)
            if color in unused:
                unused.remove(color)
            present_colors.add(color)
            inner_plan.append((ci, color))
    return inner_plan, present_colors


def _gen_place_optional_adjacent_distractor(
    out: np.ndarray,
    palette: list[int],
    containers: list[dict],
    occupied_bboxes: list[tuple[int, int, int, int]],
    present_colors: set[int],
) -> bool:
    height, width = out.shape
    outside_adjacent_placed = False
    want_outside_adjacent = height * width >= _GEN_OUTSIDE_ADJACENT_MIN_AREA and random.random() < _GEN_OUTSIDE_ADJACENT_PROB
    if want_outside_adjacent:
        adjacent_colors = [c for c in palette if any(c != container["color"] for container in containers)]
        random.shuffle(adjacent_colors)
        for color in adjacent_colors:
            if _gen_place_outside_adjacent_distractor(out, color, occupied_bboxes, containers):
                present_colors.add(color)
                outside_adjacent_placed = True
                break
    return outside_adjacent_placed


def _gen_standalone_count(height: int, width: int) -> int:
    dense_standalone = height * width >= _GEN_DENSE_GRID_MIN_AREA and random.random() < _GEN_DENSE_STANDALONE_PROB
    if dense_standalone:
        return random.randint(*_GEN_DENSE_STANDALONE_COUNT_RANGE)
    return random.choices(_GEN_STANDALONE_COUNT_CHOICES, weights=_GEN_STANDALONE_COUNT_WEIGHTS, k=1)[0]


def _gen_place_standalone_distractors(
    out: np.ndarray,
    palette: list[int],
    occupied_bboxes: list[tuple[int, int, int, int]],
    present_colors: set[int],
    already_standalone: int,
) -> bool:
    height, width = out.shape
    missing_colors = [c for c in palette if c not in present_colors]
    target_standalone_count = _gen_standalone_count(height, width)
    standalone_count = max(0, max(target_standalone_count, len(missing_colors)) - already_standalone)
    distractor_colors = missing_colors + [random.choice(palette) for _ in range(standalone_count - len(missing_colors))]
    random.shuffle(distractor_colors)
    for i, color in enumerate(distractor_colors):
        if not _gen_place_distractor(out, color, occupied_bboxes, len(distractor_colors) - i - 1):
            return False
    return True


def _gen_place_inner_objects(
    inp: np.ndarray,
    containers: list[dict],
    inner_counts: list[int],
    inner_plan: list[tuple[int, int]],
) -> bool:
    placed_per_container = [0 for _ in containers]
    for ci, color in inner_plan:
        remaining = inner_counts[ci] - placed_per_container[ci] - 1
        if not _gen_place_inner(inp, containers[ci], color, remaining):
            return False
        placed_per_container[ci] += 1
    return True


def _gen_attempt() -> np.ndarray | None:
    height, width = _gen_canvas_shape()
    out = np.zeros((height, width), dtype=int)
    palette = random.sample(_GEN_FOREGROUND_COLORS, random.randint(*_GEN_PALETTE_SIZE_RANGE))

    container_count = _gen_container_count(height, width)
    occupied_bboxes: list[tuple[int, int, int, int]] = []
    containers = _gen_place_containers(out, palette, container_count, occupied_bboxes)
    if containers is None:
        return None

    if _gen_has_unwanted_removal(out):
        return None

    inner_counts = _gen_inner_counts(container_count)
    inner_plan, present_colors = _gen_plan_inner_objects(palette, containers, inner_counts)
    outside_adjacent_placed = _gen_place_optional_adjacent_distractor(out, palette, containers, occupied_bboxes, present_colors)
    if not _gen_place_standalone_distractors(out, palette, occupied_bboxes, present_colors, int(outside_adjacent_placed)):
        return None

    if _gen_has_unwanted_removal(out):
        return None

    inp = out.copy()
    if not _gen_place_inner_objects(inp, containers, inner_counts, inner_plan):
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if np.array_equal(solved, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
