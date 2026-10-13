from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
WALL = 6

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
TARGET_COLORS = (BLUE, RED)

GEN_RETRIES = 500


def neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def target_slots(center: tuple[int, int]) -> list[tuple[int, int]]:
    cr, cc = center
    slots = []
    for r in range(cr - 1, cr + 2):
        for c in range(cc - 1, cc + 2):
            if (r, c) != center:
                slots.append((r, c))
    return slots


def component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def label_passable(mask: np.ndarray) -> tuple[np.ndarray, dict[int, list[tuple[int, int]]]]:
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=int)
    groups: dict[int, list[tuple[int, int]]] = {}
    nxt = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] >= 0:
                continue
            q = deque([(r, c)])
            labels[r, c] = nxt
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and labels[nr, nc] < 0:
                        labels[nr, nc] = nxt
                        q.append((nr, nc))
            groups[nxt] = cells
            nxt += 1
    return labels, groups


def detect_targets(grid: np.ndarray) -> dict[int, tuple[int, int]]:
    centers: dict[int, tuple[int, int]] = {}
    for color in TARGET_COLORS:
        pts = np.argwhere(grid == color)
        if len(pts) != 1:
            raise AssertionError(f"expected exactly one center of color {color}")
        centers[color] = tuple(map(int, pts[0]))
    return centers


def analyze_candidates(
    grid: np.ndarray,
) -> tuple[
    dict[int, tuple[int, int]],
    dict[int, list[tuple[int, int]]],
    list[tuple[int, int]],
    dict[tuple[int, int], tuple[int, ...]],
    dict[tuple[int, int], int],
]:
    centers = detect_targets(grid)
    green_components = component_coords(grid == GREEN)
    candidates: list[tuple[int, int]] = []
    isolated: list[tuple[int, int]] = []
    for comp in green_components:
        if len(comp) == 1:
            isolated.extend(comp)
        else:
            candidates.extend(comp)

    passable = (grid == BLACK) | (grid == GREEN)
    labels, _groups = label_passable(passable)

    slot_component_ids: dict[int, set[int]] = {}
    for color, center in centers.items():
        ids = set()
        for r, c in target_slots(center):
            lab = int(labels[r, c])
            if lab >= 0:
                ids.add(lab)
        slot_component_ids[color] = ids

    reachable: dict[tuple[int, int], tuple[int, ...]] = {}
    assigned: dict[tuple[int, int], int] = {}
    for cell in candidates + isolated:
        r, c = cell
        lab = int(labels[r, c])
        reach = []
        if lab >= 0:
            for color in TARGET_COLORS:
                if lab in slot_component_ids[color]:
                    reach.append(color)
        reach.sort(key=lambda color: (abs(r - centers[color][0]) + abs(c - centers[color][1]), centers[color][0], centers[color][1]))
        reachable[cell] = tuple(reach)
        if cell in candidates and reach:
            assigned[cell] = reach[0]

    return centers, {color: target_slots(center) for color, center in centers.items()}, isolated, reachable, assigned


def main(grid: np.ndarray) -> np.ndarray:
    """
    Gather non-isolated green pixels into the two target blocks and recolor them to the target-center colors.

    Input structure:
    - The grid is a mostly black canvas containing exactly one blue center and one red center, each surrounded by an empty 3x3 target block.
    - Green pixels form a mix of isolated singletons and 4-connected clusters of size at least two.
    - Magenta wall cells block movement; candidates may travel only through black and green cells.

    Rule:
    - Treat every green 4-connected component of area 1 as non-candidate and leave it unchanged.
    - Every green pixel belonging to a larger 4-connected component is a candidate; determine which target blocks it can reach through 4-connected black/green paths that do not cross walls.
    - Assign each reachable candidate pixel to its nearest reachable target center by Manhattan distance, then keep the 8 closest assigned pixels for each target.
    - Fill the 8 non-center cells of each target block in row-major order with that target's center color, erase the moved source pixels to black, and leave all other cells unchanged.

    Key insight:
    - The transformation has two gates before any movement happens: a green pixel must belong to a multi-pixel component and it must be path-reachable to a target through the wall-constrained black/green region; only then does distance ranking decide which pixels actually enter each 3x3 block.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    centers, slots_by_color, _isolated, reachable, assigned = analyze_candidates(g)

    chosen_by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for cell, color in assigned.items():
        r, c = cell
        chosen_by_color[color].append(cell)

    for color in TARGET_COLORS:
        center = centers[color]
        chosen_by_color[color].sort(
            key=lambda cell: (
                abs(cell[0] - center[0]) + abs(cell[1] - center[1]),
                cell[0],
                cell[1],
            )
        )

    moved: set[tuple[int, int]] = set()
    for color in TARGET_COLORS:
        slots = slots_by_color[color]
        chosen = chosen_by_color[color][: len(slots)]
        moved.update(chosen)
        for (rr, cc), (_sr, _sc) in zip(slots, chosen):
            out[rr, cc] = color

    for r, c in moved:
        out[r, c] = BLACK

    return out


def random_partition(total: int, *, lo: int = 2, hi: int = 4) -> list[int] | None:
    if total == 0:
        return []

    options: list[list[int]] = []

    def rec(rem: int, current: list[int]) -> None:
        if rem == 0:
            options.append(current[:])
            return
        for size in range(lo, hi + 1):
            if size > rem:
                continue
            if rem - size == 1:
                continue
            current.append(size)
            rec(rem - size, current)
            current.pop()

    rec(total, [])
    if not options:
        return None
    return random.choice(options)


def free_for_blob(grid: np.ndarray, r: int, c: int, blob: set[tuple[int, int]]) -> bool:
    if grid[r, c] != BLACK:
        return False
    for nr, nc in neighbors4(r, c):
        if in_bounds(grid.shape, nr, nc) and grid[nr, nc] == GREEN and (nr, nc) not in blob:
            return False
    return True


def grow_blob(
    grid: np.ndarray,
    allowed: set[tuple[int, int]],
    size: int,
    *,
    seed_pool: set[tuple[int, int]] | None = None,
    tries: int = 80,
) -> list[tuple[int, int]] | None:
    if seed_pool is None:
        seed_pool = allowed
    starts = [cell for cell in seed_pool if free_for_blob(grid, cell[0], cell[1], set())]
    random.shuffle(starts)
    for start in starts[:tries]:
        blob = {start}
        while len(blob) < size:
            frontier = []
            for r, c in list(blob):
                for nr, nc in neighbors4(r, c):
                    if (nr, nc) in allowed and (nr, nc) not in blob and free_for_blob(grid, nr, nc, blob):
                        frontier.append((nr, nc))
            frontier = list(dict.fromkeys(frontier))
            if not frontier:
                break
            blob.add(random.choice(frontier))
        if len(blob) == size:
            return sorted(blob)
    return None


def place_blob_group(
    grid: np.ndarray,
    sizes: list[int],
    primary_allowed: set[tuple[int, int]],
    secondary_allowed: set[tuple[int, int]],
) -> list[list[tuple[int, int]]] | None:
    placed: list[list[tuple[int, int]]] = []
    for size in sizes:
        blob = grow_blob(grid, primary_allowed, size)
        if blob is None:
            blob = grow_blob(grid, secondary_allowed, size)
        if blob is None:
            return None
        for r, c in blob:
            grid[r, c] = GREEN
        placed.append(blob)
    return placed


def place_isolated(grid: np.ndarray, cells: list[tuple[int, int]], count: int) -> bool:
    chosen = 0
    for r, c in cells:
        if grid[r, c] != BLACK:
            continue
        if any(in_bounds(grid.shape, nr, nc) and grid[nr, nc] == GREEN for nr, nc in neighbors4(r, c)):
            continue
        grid[r, c] = GREEN
        chosen += 1
        if chosen == count:
            return True
    return False


def build_layout() -> np.ndarray:
    h = random.randint(14, 17)
    w = random.randint(14, 17)
    grid = np.zeros((h, w), dtype=int)

    wall_c = w // 2 + random.choice([-1, 0, 0, 1])
    wall_c = max(5, min(w - 6, wall_c))

    blue_r = random.randint(3, 5)
    red_r = random.randint(h - 6, h - 4)
    if red_r - blue_r < 6:
        red_r = blue_r + 6
    red_r = min(red_r, h - 4)

    blue_c = wall_c - random.choice([3, 4])
    red_c = wall_c + random.choice([3, 4])
    blue_c = max(2, blue_c)
    red_c = min(w - 3, red_c)

    blue_center = (blue_r, blue_c)
    red_center = (red_r, red_c)

    top_gap = random.randint(1, blue_r - 2)
    bottom_gap = random.randint(red_r + 2, h - 2)
    blue_wall_r = blue_r + 2
    red_wall_r = red_r - 2

    for r in range(h):
        if r not in (top_gap, bottom_gap):
            grid[r, wall_c] = WALL

    grid[blue_wall_r, : wall_c + 1] = WALL
    grid[red_wall_r, wall_c:] = WALL

    for center, color in ((blue_center, BLUE), (red_center, RED)):
        cr, cc = center
        grid[cr, cc] = color
        for rr, cc2 in target_slots(center):
            grid[rr, cc2] = BLACK

    # A few short decorative wall stubs that do not change the basic reachability split.
    stub_choices = [
        (top_gap + 1, wall_c + 2, 0, random.randint(2, 3)),
        (bottom_gap - 1, wall_c - 3, 0, random.randint(2, 3)),
        (blue_wall_r - 2, max(1, wall_c - 4), 1, random.randint(2, 3)),
        (red_wall_r + 1, min(w - 2, wall_c + 4), 1, random.randint(2, 3)),
    ]
    random.shuffle(stub_choices)
    for sr, sc, axis, length in stub_choices[: random.randint(1, 3)]:
        for k in range(length):
            rr = sr + (k if axis == 0 else 0)
            cc = sc + (0 if axis == 0 else k)
            if not in_bounds(grid.shape, rr, cc):
                break
            if grid[rr, cc] != BLACK:
                break
            blocked = False
            for center in (blue_center, red_center):
                if abs(rr - center[0]) <= 1 and abs(cc - center[1]) <= 1:
                    blocked = True
                    break
            if blocked:
                break
            grid[rr, cc] = WALL

    return grid


def region_sets(
    grid: np.ndarray,
) -> tuple[
    dict[int, tuple[int, int]],
    dict[str, set[tuple[int, int]]],
    dict[tuple[int, int], tuple[int, ...]],
]:
    centers = detect_targets(grid)
    slots_by_color = {color: set(target_slots(center)) for color, center in centers.items()}
    passable = grid == BLACK
    labels, _groups = label_passable(passable)
    reach_by_cell: dict[tuple[int, int], tuple[int, ...]] = {}
    regions = {
        "blue_only": set(),
        "red_only": set(),
        "both": set(),
        "neither": set(),
    }
    slot_components: dict[int, set[int]] = {}
    for color, slots in slots_by_color.items():
        ids = {int(labels[r, c]) for r, c in slots if int(labels[r, c]) >= 0}
        slot_components[color] = ids

    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if not passable[r, c]:
                continue
            cell = (r, c)
            if cell in slots_by_color[BLUE] or cell in slots_by_color[RED]:
                continue
            lab = int(labels[r, c])
            reach = []
            for color in TARGET_COLORS:
                if lab in slot_components[color]:
                    reach.append(color)
            reach_t = tuple(reach)
            reach_by_cell[cell] = reach_t
            if reach_t == (BLUE,):
                regions["blue_only"].add(cell)
            elif reach_t == (RED,):
                regions["red_only"].add(cell)
            elif reach_t == (BLUE, RED):
                regions["both"].add(cell)
            else:
                regions["neither"].add(cell)
    return centers, regions, reach_by_cell


def sorted_cells(
    cells: set[tuple[int, int]],
    center: tuple[int, int],
) -> list[tuple[int, int]]:
    return sorted(cells, key=lambda cell: (abs(cell[0] - center[0]) + abs(cell[1] - center[1]), cell[0], cell[1]))


def semantic_checks(inp: np.ndarray, out: np.ndarray) -> bool:
    centers, slots_by_color, isolated, reachable, assigned = analyze_candidates(inp)
    if len(isolated) < 3:
        return False

    for color, center in centers.items():
        if inp[center] != color:
            return False
        for cell in slots_by_color[color]:
            if inp[cell] != BLACK:
                return False
            if out[cell] != color:
                return False

    assigned_by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for cell, color in assigned.items():
        assigned_by_color[color].append(cell)

    # Distance ranking must matter in at least one branch.
    if max(len(assigned_by_color[BLUE]), len(assigned_by_color[RED])) <= 8:
        return False

    selected_by_color: dict[int, list[tuple[int, int]]] = {}
    for color in TARGET_COLORS:
        center = centers[color]
        ranked = sorted(
            assigned_by_color[color],
            key=lambda cell: (
                abs(cell[0] - center[0]) + abs(cell[1] - center[1]),
                cell[0],
                cell[1],
            ),
        )
        if len(ranked) < 8:
            return False
        selected = ranked[:8]
        selected_by_color[color] = selected
        for cell in selected:
            if out[cell] != BLACK:
                return False
        for cell in ranked[8:]:
            if out[cell] != GREEN:
                return False

    # Area gate counterexample: at least one isolated pixel is reachable and closer than a moved candidate.
    area_counterexample = False
    for cell in isolated:
        reach = reachable.get(cell, ())
        for color in reach:
            center = centers[color]
            d_iso = abs(cell[0] - center[0]) + abs(cell[1] - center[1])
            worst_selected = max(abs(r - center[0]) + abs(c - center[1]) for r, c in selected_by_color[color])
            if d_iso < worst_selected:
                area_counterexample = True
                break
        if area_counterexample:
            break
    if not area_counterexample:
        return False

    # Reachability counterexample: nearest-by-geometry disagrees with the assigned target for at least one candidate.
    reach_counterexample = False
    other = {BLUE: RED, RED: BLUE}
    for cell, color in assigned.items():
        if len(reachable[cell]) != 1:
            continue
        d_self = abs(cell[0] - centers[color][0]) + abs(cell[1] - centers[color][1])
        d_other = abs(cell[0] - centers[other[color]][0]) + abs(cell[1] - centers[other[color]][1])
        if d_other < d_self:
            reach_counterexample = True
            break
    if not reach_counterexample:
        return False

    # Unmoved isolated pixels stay green.
    for cell in isolated:
        if out[cell] != GREEN:
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = build_layout()
        centers, regions, _reach_by_cell = region_sets(grid)

        blue_only = regions["blue_only"]
        red_only = regions["red_only"]
        both = regions["both"]
        neither = regions["neither"]
        if len(blue_only) < 20 or len(red_only) < 20:
            continue

        # Enforce wall-driven misassignment examples by reserving cells on the "wrong" side.
        blue_cross = {
            cell
            for cell in blue_only
            if cell[1] > detect_targets(grid)[RED][1] - 1
            and abs(cell[0] - centers[RED][0]) + abs(cell[1] - centers[RED][1])
            < abs(cell[0] - centers[BLUE][0]) + abs(cell[1] - centers[BLUE][1])
        }
        red_cross = {
            cell
            for cell in red_only
            if cell[1] < detect_targets(grid)[BLUE][1] + 1
            and abs(cell[0] - centers[BLUE][0]) + abs(cell[1] - centers[BLUE][1])
            < abs(cell[0] - centers[RED][0]) + abs(cell[1] - centers[RED][1])
        }
        if len(blue_cross) < 4 or len(red_cross) < 4:
            continue

        blue_sorted = sorted_cells(blue_only, centers[BLUE])
        red_sorted = sorted_cells(red_only, centers[RED])

        blue_far = set(blue_sorted[max(8, len(blue_sorted) // 2) :])
        red_far = set(red_sorted[max(8, len(red_sorted) // 2) :])
        blue_near = set(blue_sorted[: max(20, len(blue_sorted) // 2)])
        red_near = set(red_sorted[: max(20, len(red_sorted) // 2)])

        blue_total = random.choice([8, 9, 10])
        red_total = random.choice([8, 9, 10])
        if blue_total == 8 and red_total == 8:
            if random.random() < 0.5:
                blue_total = 9
            else:
                red_total = 9

        blue_sizes = random_partition(blue_total - 2)
        red_sizes = random_partition(red_total - 2)
        if blue_sizes is None or red_sizes is None:
            continue
        random.shuffle(blue_sizes)
        random.shuffle(red_sizes)

        # Required misleading cross blobs.
        blue_cross_blob = grow_blob(grid, blue_cross, 2, seed_pool=blue_cross)
        if blue_cross_blob is None:
            continue
        for r, c in blue_cross_blob:
            grid[r, c] = GREEN

        red_cross_blob = grow_blob(grid, red_cross, 2, seed_pool=red_cross)
        if red_cross_blob is None:
            continue
        for r, c in red_cross_blob:
            grid[r, c] = GREEN

        # Remaining candidate blobs: near blobs first, optional far overflow later.
        blue_extra = blue_total - 8
        red_extra = red_total - 8
        blue_far_budget = 2 if blue_extra >= 1 else 0
        red_far_budget = 2 if red_extra >= 1 else 0

        def split_sizes(sizes: list[int], far_budget: int) -> tuple[list[int], list[int]]:
            near_sizes = sizes[:]
            far_sizes: list[int] = []
            if far_budget > 0:
                for i, size in enumerate(near_sizes):
                    if size >= far_budget:
                        far_sizes = [far_budget]
                        near_sizes[i] = size - far_budget
                        break
            near_sizes = [size for size in near_sizes if size > 0]
            return near_sizes, far_sizes

        blue_near_sizes, blue_far_sizes = split_sizes(blue_sizes, blue_far_budget)
        red_near_sizes, red_far_sizes = split_sizes(red_sizes, red_far_budget)

        if place_blob_group(grid, blue_near_sizes, blue_near, blue_only) is None:
            continue
        if place_blob_group(grid, red_near_sizes, red_near, red_only) is None:
            continue
        if place_blob_group(grid, blue_far_sizes, blue_far, blue_only) is None:
            continue
        if place_blob_group(grid, red_far_sizes, red_far, red_only) is None:
            continue

        # Optional unreachable multi-pixel distractor to exercise the reachability gate.
        if len(neither) >= 8 and random.random() < 0.45:
            blob = grow_blob(grid, neither, random.choice([2, 3]), seed_pool=neither)
            if blob is not None:
                for r, c in blob:
                    grid[r, c] = GREEN

        # Isolated pixels: force one near a target, then add more elsewhere.
        near_iso_candidates = sorted(
            [
                cell
                for cell in blue_only | red_only | both
                if min(
                    abs(cell[0] - centers[BLUE][0]) + abs(cell[1] - centers[BLUE][1]),
                    abs(cell[0] - centers[RED][0]) + abs(cell[1] - centers[RED][1]),
                )
                <= 4
            ],
            key=lambda cell: (
                min(
                    abs(cell[0] - centers[BLUE][0]) + abs(cell[1] - centers[BLUE][1]),
                    abs(cell[0] - centers[RED][0]) + abs(cell[1] - centers[RED][1]),
                ),
                cell[0],
                cell[1],
            ),
        )
        if not place_isolated(grid, near_iso_candidates, 1):
            continue

        all_black = [(r, c) for r in range(grid.shape[0]) for c in range(grid.shape[1]) if grid[r, c] == BLACK]
        random.shuffle(all_black)
        if not place_isolated(grid, all_black, random.choice([2, 3])):
            continue

        out = main(grid)
        if semantic_checks(grid, out):
            return grid

    raise RuntimeError("failed to generate a valid input for puzzle 7")
