from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
TEAL = 8

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_RETRY_LIMIT = 500


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _label_mask(mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=int)
    components: list[list[tuple[int, int]]] = []
    next_id = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != -1:
                continue
            q = deque([(r, c)])
            labels[r, c] = next_id
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = next_id
                        q.append((nr, nc))
            components.append(coords)
            next_id += 1
    return labels, components


def _classify_green_objects(grid: np.ndarray) -> tuple[list[list[tuple[int, int]]], set[int], set[int]]:
    green_labels, green_objects = _label_mask(grid == GREEN)
    _ = green_labels
    nonblack_labels, _ = _label_mask(grid != BLACK)
    comp_to_green_ids: dict[int, list[int]] = defaultdict(list)
    for obj_id, coords in enumerate(green_objects):
        comp_id = nonblack_labels[coords[0][0], coords[0][1]]
        for r, c in coords[1:]:
            if nonblack_labels[r, c] != comp_id:
                raise ValueError("green object split across non-black components")
        comp_to_green_ids[int(comp_id)].append(obj_id)

    emitters: set[int] = set()
    for obj_ids in comp_to_green_ids.values():
        if len(obj_ids) >= 2:
            emitters.update(obj_ids)
    blockers = set(range(len(green_objects))) - emitters
    return green_objects, emitters, blockers


def _render_with_emitters(
    grid: np.ndarray,
    green_objects: list[list[tuple[int, int]]],
    emitter_ids: set[int],
) -> tuple[np.ndarray, int]:
    out = np.array(grid, dtype=int, copy=True)
    emitter_cells: list[tuple[int, int]] = []
    for obj_id, coords in enumerate(green_objects):
        if obj_id in emitter_ids:
            emitter_cells.extend(coords)
        else:
            for r, c in coords:
                out[r, c] = GREY

    stop_count = 0
    h, w = out.shape
    for r, c in emitter_cells:
        for dr, dc in DIRS4:
            rr, cc = r + dr, c + dc
            while 0 <= rr < h and 0 <= cc < w:
                value = int(out[rr, cc])
                if value == GREY:
                    stop_count += 1
                    break
                if value in (BLACK, RED, BLUE):
                    out[rr, cc] = BLUE
                rr += dr
                cc += dc
    return out, stop_count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use bridge-defined connectivity to decide which green objects emit rays and which become blockers.

    Input structure:
    - A black background contains 3 to 5 small 4-connected green objects plus some teal bridge paths.
    - Teal pixels may connect multiple green objects into one mixed-color 4-connected component.
    - Red singleton markers lie on empty background cells and are not 4-adjacent to the green or teal structure.

    Rule:
    - Build 4-connected components over all non-black pixels, ignoring color differences.
    - Treat each monochromatic green 4-connected object as an emitter iff its non-black component contains at least
      one other green object; otherwise treat it as a blocker.
    - Recolor blocker green pixels to grey.
    - From every emitter pixel, cast rays up, down, left, and right; rays paint black and red cells blue, pass
      through green emitter pixels and teal bridge pixels, and stop immediately before grey blocker pixels or the grid edge.

    Key insight:
    - Green objects share the same visible color, so only the mixed-color connectivity created by teal bridges reveals
      which ones emit rays and which ones instead act as the walls that stop those rays.
    """
    g = np.asarray(grid, dtype=int)
    green_objects, emitter_ids, _ = _classify_green_objects(g)
    out, _ = _render_with_emitters(g, green_objects, emitter_ids)
    return out


def _bbox_of_shape(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _absolute_cells(shape: tuple[tuple[int, int], ...], top: int, left: int) -> list[tuple[int, int]]:
    return [(top + r, left + c) for r, c in shape]


def _random_shape() -> tuple[tuple[int, int], ...]:
    for _ in range(200):
        area = random.randint(3, 6)
        coords = {(2, 2)}
        while len(coords) < area:
            frontier = []
            for r, c in coords:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < 5 and 0 <= nc < 5 and (nr, nc) not in coords:
                        frontier.append((nr, nc))
            coords.add(random.choice(frontier))

        min_r = min(r for r, _ in coords)
        min_c = min(c for _, c in coords)
        norm = tuple(sorted((r - min_r, c - min_c) for r, c in coords))
        h, w = _bbox_of_shape(norm)
        full_rect = len(norm) == h * w
        line_like = min(h, w) == 1
        if full_rect and random.random() < 0.85:
            continue
        if line_like and random.random() < 0.45:
            continue
        return norm
    raise RuntimeError("failed to create a green shape")


def _can_place_shape(grid: np.ndarray, shape: tuple[tuple[int, int], ...], top: int, left: int) -> bool:
    h, w = grid.shape
    cells = _absolute_cells(shape, top, left)
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    if min(rows) < 2 or min(cols) < 2 or max(rows) > h - 3 or max(cols) > w - 3:
        return False
    cell_set = set(cells)
    for r, c in cells:
        if grid[r, c] != BLACK:
            return False
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cell_set and grid[nr, nc] != BLACK:
                    return False
    return True


def _sorted_positions(
    shape: tuple[tuple[int, int], ...],
    grid_shape: tuple[int, int],
    target: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = _bbox_of_shape(shape)
    gh, gw = grid_shape
    positions = [(r, c) for r in range(2, gh - h - 1) for c in range(2, gw - w - 1)]
    random.shuffle(positions)
    positions.sort(key=lambda rc: abs(rc[0] - target[0]) + abs(rc[1] - target[1]))
    return positions


def _place_shape_near(
    grid: np.ndarray,
    shape: tuple[tuple[int, int], ...],
    target: tuple[int, int],
) -> list[tuple[int, int]] | None:
    for top, left in _sorted_positions(shape, grid.shape, target):
        if _can_place_shape(grid, shape, top, left):
            cells = _absolute_cells(shape, top, left)
            for r, c in cells:
                grid[r, c] = GREEN
            return cells
    return None


def _green_owner(objects: list[list[tuple[int, int]]], shape: tuple[int, int]) -> np.ndarray:
    owner = np.full(shape, -1, dtype=int)
    for obj_id, coords in enumerate(objects):
        for r, c in coords:
            owner[r, c] = obj_id
    return owner


def _bridge_path(
    grid: np.ndarray,
    owner: np.ndarray,
    start_obj: int,
    end_obj: int,
) -> list[tuple[int, int]] | None:
    h, w = grid.shape

    def adjacent_green_ids(r: int, c: int) -> set[int]:
        ids = set()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and owner[nr, nc] != -1:
                ids.add(int(owner[nr, nc]))
        return ids

    def adjacent_nonblack(r: int, c: int) -> bool:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != BLACK:
                return True
        return False

    starts = []
    ends = set()
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if grid[r, c] != BLACK:
                continue
            adj_ids = adjacent_green_ids(r, c)
            if adj_ids == {start_obj} and not any(
                0 <= nr < h and 0 <= nc < w and grid[nr, nc] == TEAL for nr, nc in _neighbors4(r, c)
            ):
                starts.append((r, c))
            if adj_ids == {end_obj} and not any(
                0 <= nr < h and 0 <= nc < w and grid[nr, nc] == TEAL for nr, nc in _neighbors4(r, c)
            ):
                ends.add((r, c))

    if not starts or not ends:
        return None

    random.shuffle(starts)
    starts.sort(key=lambda rc: min(abs(rc[0] - er) + abs(rc[1] - ec) for er, ec in ends))
    for start in starts:
        q = deque([start])
        parent = {start: None}
        while q:
            r, c = q.popleft()
            if (r, c) in ends and (r, c) != start:
                path = []
                cur = (r, c)
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                if len(path) >= 2:
                    return path
                break
            for nr, nc in _neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w) or (nr, nc) in parent:
                    continue
                if grid[nr, nc] != BLACK:
                    continue
                if (nr, nc) in ends:
                    parent[(nr, nc)] = (r, c)
                    q.append((nr, nc))
                elif not adjacent_nonblack(nr, nc):
                    parent[(nr, nc)] = (r, c)
                    q.append((nr, nc))
    return None


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]]) -> None:
    for r, c in path:
        grid[r, c] = TEAL


def _is_hit_by_emitter(
    grid: np.ndarray,
    cells: set[tuple[int, int]],
    emitter_cells: list[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    for r, c in emitter_cells:
        for dr, dc in DIRS4:
            rr, cc = r + dr, c + dc
            saw_black = False
            while 0 <= rr < h and 0 <= cc < w:
                if (rr, cc) in cells:
                    return saw_black
                if grid[rr, cc] not in (BLACK, GREEN, TEAL):
                    break
                saw_black = saw_black or grid[rr, cc] == BLACK
                rr += dr
                cc += dc
    return False


def _red_safe_cells(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    safe = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != BLACK:
                continue
            if any(0 <= nr < h and 0 <= nc < w and grid[nr, nc] in (GREEN, TEAL) for nr, nc in _neighbors4(r, c)):
                continue
            safe.append((r, c))
    return safe


def _pick_scattered(candidates: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    candidates = candidates[:]
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    for r, c in candidates:
        if all(max(abs(r - rr), abs(c - cc)) >= 2 for rr, cc in chosen):
            chosen.append((r, c))
            if len(chosen) == count:
                break
    return chosen


def _analysis(grid: np.ndarray, *, require_reds: bool = True) -> dict[str, object] | None:
    green_objects, emitters, blockers = _classify_green_objects(grid)
    if not (3 <= len(green_objects) <= 5):
        return None
    if any(not (3 <= len(obj) <= 6) for obj in green_objects):
        return None

    nonblack_labels, nonblack_components = _label_mask(grid != BLACK)
    group_sizes = []
    for coords in nonblack_components:
        comp_id = nonblack_labels[coords[0][0], coords[0][1]]
        obj_count = 0
        seen = set()
        for r, c in coords:
            if grid[r, c] == GREEN:
                seen.add((r, c))
        if seen:
            for obj in green_objects:
                if any(cell in seen for cell in obj):
                    obj_count += 1
            group_sizes.append(obj_count)

    if not any(size >= 2 for size in group_sizes):
        return None
    if 1 not in group_sizes:
        return None
    if max(group_sizes) > 3:
        return None

    teal_labels, teal_components = _label_mask(grid == TEAL)
    _ = teal_labels
    if not teal_components or any(len(comp) < 2 for comp in teal_components):
        return None

    for r, c in np.argwhere(grid == RED):
        if any(
            0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] in (GREEN, TEAL)
            for nr, nc in _neighbors4(int(r), int(c))
        ):
            return None

    true_out, stop_count = _render_with_emitters(grid, green_objects, emitters)
    all_emitters_out, _ = _render_with_emitters(grid, green_objects, set(range(len(green_objects))))
    all_blockers_out, _ = _render_with_emitters(grid, green_objects, set())
    if np.array_equal(true_out, all_emitters_out) or np.array_equal(true_out, all_blockers_out):
        return None

    red_total = int(np.sum(grid == RED))
    red_painted = int(np.sum((grid == RED) & (true_out == BLUE)))
    red_preserved = int(np.sum((grid == RED) & (true_out == RED)))
    if require_reds and (red_total < 2 or red_painted < 1 or red_preserved < 1):
        return None
    if stop_count < 1:
        return None
    if int(np.sum(true_out == BLUE)) < 8:
        return None
    if int(np.sum(true_out == GREY)) < 3:
        return None

    return {
        "green_objects": green_objects,
        "emitters": emitters,
        "blockers": blockers,
        "output": true_out,
        "stop_count": stop_count,
    }


def _generation_attempt() -> np.ndarray | None:
    h = random.randint(15, 18)
    w = random.randint(15, 18)
    grid = np.zeros((h, w), dtype=int)

    total_objects = random.randint(3, 5)
    bridge_group_size = random.randint(2, min(3, total_objects - 1))
    axis = random.choice(["h", "v"])
    step = random.randint(4, 5)

    shapes = [_random_shape() for _ in range(total_objects)]
    if total_objects > bridge_group_size and random.random() < 0.7:
        shapes[bridge_group_size] = shapes[random.randrange(bridge_group_size)]

    group_objects: list[list[tuple[int, int]]] = []
    if axis == "h":
        base_r = random.randint(3, max(3, h - 8))
        base_c = random.randint(3, max(3, w - (bridge_group_size - 1) * step - 6))
        targets = [(base_r + random.randint(-1, 1), base_c + i * step + random.randint(-1, 1)) for i in range(bridge_group_size)]
        order_key = lambda cells: sum(c for _, c in cells) / len(cells)
    else:
        base_r = random.randint(3, max(3, h - (bridge_group_size - 1) * step - 6))
        base_c = random.randint(3, max(3, w - 8))
        targets = [(base_r + i * step + random.randint(-1, 1), base_c + random.randint(-1, 1)) for i in range(bridge_group_size)]
        order_key = lambda cells: sum(r for r, _ in cells) / len(cells)

    for shape, target in zip(shapes[:bridge_group_size], targets):
        cells = _place_shape_near(grid, shape, target)
        if cells is None:
            return None
        group_objects.append(cells)

    ordered_group = sorted(range(len(group_objects)), key=lambda idx: order_key(group_objects[idx]))
    owner = _green_owner(group_objects, grid.shape)
    bridge_lengths: list[int] = []
    for a, b in zip(ordered_group, ordered_group[1:]):
        path = _bridge_path(grid, owner, a, b)
        if path is None:
            return None
        bridge_lengths.append(len(path))
        _paint_path(grid, path)

    if not bridge_lengths or any(length < 2 for length in bridge_lengths):
        return None

    emitter_cells = [cell for coords in group_objects for cell in coords]
    all_objects = group_objects[:]

    remaining_shapes = shapes[bridge_group_size:]
    if not remaining_shapes:
        return None

    placed_remaining = 0
    for need_hit, shape in zip(
        [True] + [False] * max(0, len(remaining_shapes) - 1),
        remaining_shapes,
    ):
        hshape, wshape = _bbox_of_shape(shape)
        positions = [(r, c) for r in range(2, h - hshape - 1) for c in range(2, w - wshape - 1)]
        random.shuffle(positions)
        chosen = None
        for top, left in positions:
            if not _can_place_shape(grid, shape, top, left):
                continue
            cells = set(_absolute_cells(shape, top, left))
            hit = _is_hit_by_emitter(grid, cells, emitter_cells)
            if need_hit != hit:
                continue
            chosen = sorted(cells)
            break
        if chosen is None:
            return None
        for r, c in chosen:
            grid[r, c] = GREEN
        all_objects.append(chosen)
        placed_remaining += 1

    if placed_remaining != total_objects - bridge_group_size:
        return None

    base_analysis = _analysis(grid, require_reds=False)
    if base_analysis is None:
        return None
    base_output = np.asarray(base_analysis["output"], dtype=int)

    safe_red = _red_safe_cells(grid)
    on_ray = [cell for cell in safe_red if base_output[cell] == BLUE]
    off_ray = [cell for cell in safe_red if base_output[cell] == BLACK]
    if not on_ray or not off_ray:
        return None

    red_count_on = random.randint(1, min(3, len(on_ray)))
    red_count_off = random.randint(1, min(2, len(off_ray)))
    chosen_on = _pick_scattered(on_ray, red_count_on)
    chosen_off = _pick_scattered([cell for cell in off_ray if cell not in chosen_on], red_count_off)
    if len(chosen_on) != red_count_on or len(chosen_off) != red_count_off:
        return None

    for r, c in chosen_on + chosen_off:
        grid[r, c] = RED

    return grid if _analysis(grid) is not None else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _generation_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid connectivity-ray instance")
