from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


MAX_GENERATION_ATTEMPTS = 400
AXIS_GAP_MIN = 3
PALETTE = tuple(range(10))
ORTHO_DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(np.asarray(grid, dtype=int))))
    if exclude:
        colors -= set(exclude)
    return sorted(colors)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid, dtype=int)[r0 : r1 + 1, c0 : c1 + 1].copy()


def _crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    box = _bbox(np.asarray(grid, dtype=int) != int(bg))
    assert box is not None
    return _crop_bbox(grid, box)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _axis_scores(points: list[tuple[int, int]], width: int) -> dict[int, int]:
    pts = {(int(r), int(c)) for r, c in points}
    scores: dict[int, int] = {}
    for k in range(2 * (width - 1) + 1):
        score = 0
        for r, c in pts:
            cc = k - c
            if 0 <= cc < width and (r, cc) in pts:
                score += 1
        scores[k] = score
    return scores


def _best_axis(points: list[tuple[int, int]], width: int) -> tuple[int, int, int]:
    scores = _axis_scores(points, width)
    best_k = 0
    best_score = -1
    for k in range(2 * (width - 1) + 1):
        score = scores[k]
        if score > best_score:
            best_k = k
            best_score = score
    second = max(score for k, score in scores.items() if k != best_k)
    return int(best_k), int(best_score), int(second)


def _connected(coords: set[tuple[int, int]]) -> bool:
    if not coords:
        return False
    start = next(iter(coords))
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for rr, cc in _neighbors4(r, c):
            if (rr, cc) in coords and (rr, cc) not in seen:
                seen.add((rr, cc))
                queue.append((rr, cc))
    return len(seen) == len(coords)


def _random_connected_mask(h: int, w: int, area: int) -> np.ndarray:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    frontier = set(_neighbors4(*start))
    while len(cells) < area:
        options = [(r, c) for r, c in frontier if 0 <= r < h and 0 <= c < w and (r, c) not in cells]
        assert options
        weights = []
        for r, c in options:
            neigh = sum((rr, cc) in cells for rr, cc in _neighbors4(r, c))
            weights.append(neigh + 1)
        choice = random.choices(options, weights=weights, k=1)[0]
        cells.add(choice)
        frontier.remove(choice)
        for rr, cc in _neighbors4(*choice):
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                frontier.add((rr, cc))
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _traversal_order(mask: np.ndarray) -> list[tuple[int, int]]:
    coords = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    start = random.choice(tuple(coords))
    order: list[tuple[int, int]] = []
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        order.append((r, c))
        neighbors = [(rr, cc) for rr, cc in _neighbors4(r, c) if (rr, cc) in coords and (rr, cc) not in seen]
        random.shuffle(neighbors)
        for nxt in neighbors:
            seen.add(nxt)
            stack.append(nxt)
    assert len(order) == len(coords)
    return order


def _sample_group_sizes(total: int, n_groups: int) -> list[int]:
    assert total >= 2 * n_groups
    sizes = [2] * n_groups
    for _ in range(total - 2 * n_groups):
        sizes[random.randrange(n_groups)] += 1
    random.shuffle(sizes)
    return sizes


def _generate_hidden_output(colors: list[int], bg: int) -> tuple[np.ndarray, dict[int, list[tuple[int, int]]]]:
    n_colors = len(colors)
    for _ in range(200):
        base_h = random.randint(4, 7)
        base_w = random.randint(4, 8)
        max_area = min(16, base_h * base_w - 1)
        area = random.randint(max(2 * n_colors, 7), max_area)
        mask = _random_connected_mask(base_h, base_w, area)
        cropped = _crop_non_bg(mask.astype(int), 0).astype(bool)
        h, w = cropped.shape
        if min(h, w) < 2:
            continue
        if area == h * w:
            continue
        order = _traversal_order(cropped)
        sizes = _sample_group_sizes(len(order), n_colors)
        output = np.full(cropped.shape, bg, dtype=int)
        by_color: dict[int, list[tuple[int, int]]] = {}
        idx = 0
        ok = True
        shuffled_colors = list(colors)
        random.shuffle(shuffled_colors)
        for color, size in zip(shuffled_colors, sizes):
            pts = order[idx : idx + size]
            idx += size
            by_color[int(color)] = pts
            for r, c in pts:
                output[r, c] = int(color)
            cols = [c for _r, c in pts]
            if max(cols) - min(cols) > 4:
                ok = False
                break
        if not ok:
            continue
        union = {tuple(map(int, rc)) for rc in np.argwhere(output != bg)}
        if not _connected(union):
            continue
        return output, by_color
    raise RuntimeError("failed to build hidden output")


def _assign_sides(colors: list[int]) -> dict[int, str]:
    shuffled = list(colors)
    random.shuffle(shuffled)
    sides: dict[int, str] = {}
    if len(shuffled) == 2:
        sides[shuffled[0]] = "left"
        sides[shuffled[1]] = "right"
    else:
        heavy_side = random.choice(["left", "right"])
        light_side = "right" if heavy_side == "left" else "left"
        sides[shuffled[0]] = heavy_side
        sides[shuffled[1]] = heavy_side
        sides[shuffled[2]] = light_side
    return sides


def _interval_clear(interval: tuple[int, int], others: list[tuple[int, int]], pad: int = 1) -> bool:
    a0, a1 = interval
    for b0, b1 in others:
        if not (a1 + pad < b0 or b1 + pad < a0):
            return False
    return True


def _choose_axis_assignments(
    target_by_color: dict[int, list[tuple[int, int]]],
    sides: dict[int, str],
    width: int,
    fig_c0: int,
    fig_c1: int,
) -> dict[int, dict]:
    colors = sorted(target_by_color, key=lambda color: len(target_by_color[color]), reverse=True)
    used_k: set[int] = set()
    intervals_by_side = {"left": [], "right": []}
    assignments: dict[int, dict] = {}

    def backtrack(i: int) -> bool:
        if i == len(colors):
            return True
        color = colors[i]
        pts = target_by_color[color]
        tcols = [c for _r, c in pts]
        tmin = min(tcols)
        tmax = max(tcols)
        span = tmax - tmin
        side = sides[color]
        candidates: list[tuple[tuple[int, int], int]] = []
        if side == "left":
            start_hi = fig_c0 - span - 3
            for start in range(1, start_hi + 1):
                interval = (start, start + span)
                k = start + tmax
                if k in used_k:
                    continue
                if _interval_clear(interval, intervals_by_side[side], pad=1):
                    candidates.append((interval, k))
        else:
            end_lo = fig_c1 + span + 3
            for end in range(end_lo, width - 1):
                interval = (end - span, end)
                k = end + tmin
                if k in used_k:
                    continue
                if _interval_clear(interval, intervals_by_side[side], pad=1):
                    candidates.append((interval, k))
        random.shuffle(candidates)
        for interval, k in candidates:
            assignments[color] = {
                "side": side,
                "k": int(k),
                "interval": interval,
            }
            used_k.add(int(k))
            intervals_by_side[side].append(interval)
            if backtrack(i + 1):
                return True
            intervals_by_side[side].pop()
            used_k.remove(int(k))
            assignments.pop(color, None)
        return False

    if not backtrack(0):
        raise RuntimeError("failed to assign axes")
    return assignments


def _reflect_point(k: int, point: tuple[int, int]) -> tuple[int, int]:
    r, c = point
    return int(r), int(k - c)


def _build_color_layer(
    color: int,
    target_points: list[tuple[int, int]],
    assignment: dict,
    shape: tuple[int, int],
    target_set: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    h, w = shape
    k = int(assignment["k"])
    side = assignment["side"]
    orphan_points = {_reflect_point(k, point) for point in target_points}
    if len(orphan_points) != len(target_points):
        raise RuntimeError("duplicate orphan points")

    def in_bounds(point: tuple[int, int]) -> bool:
        r, c = point
        return 0 <= r < h and 0 <= c < w

    def on_source_side(point: tuple[int, int]) -> bool:
        _r, c = point
        return 2 * c < k if side == "left" else 2 * c > k

    forbidden = set(target_set) | set(occupied) | set(orphan_points)
    if not all(in_bounds(point) for point in orphan_points):
        raise RuntimeError("orphan out of bounds")
    if orphan_points & target_set:
        raise RuntimeError("orphan overlaps hidden figure")
    if orphan_points & occupied:
        raise RuntimeError("orphan overlaps other color")

    row_values = [r for r, _c in target_points]
    row_lo = max(0, min(row_values) - 2)
    row_hi = min(h - 1, max(row_values) + 2)

    side_cells: set[tuple[int, int]] = set()
    taken = set(forbidden)

    for orphan in orphan_points:
        candidates = list(ORTHO_DELTAS)
        random.shuffle(candidates)
        placed = False
        for dr, dc in candidates:
            a = (orphan[0] + dr, orphan[1] + dc)
            b = _reflect_point(k, a)
            if not (in_bounds(a) and in_bounds(b)):
                continue
            if not on_source_side(a):
                continue
            if a in taken or b in taken:
                continue
            side_cells.add(a)
            taken.add(a)
            taken.add(b)
            placed = True
            break
        if not placed:
            raise RuntimeError("failed to anchor orphan")

    target_pairs = random.randint(len(orphan_points) + 2, len(orphan_points) + 5)
    frontier = set()
    for point in side_cells:
        frontier.update(_neighbors4(*point))
    while len(side_cells) < target_pairs:
        options = []
        for point in frontier:
            r, c = point
            mirror = _reflect_point(k, point)
            if not (row_lo <= r <= row_hi):
                continue
            if not (in_bounds(point) and in_bounds(mirror)):
                continue
            if not on_source_side(point):
                continue
            if point in taken or mirror in taken:
                continue
            touch = sum(neigh in side_cells for neigh in _neighbors4(r, c))
            options.append((touch, point))
        if not options:
            raise RuntimeError("frontier exhausted")
        best_touch = max(touch for touch, _point in options)
        shortlist = [point for touch, point in options if touch >= best_touch - 1]
        choice = random.choice(shortlist)
        mirror = _reflect_point(k, choice)
        side_cells.add(choice)
        taken.add(choice)
        taken.add(mirror)
        for point in (choice, mirror):
            for neigh in _neighbors4(*point):
                frontier.add(neigh)

    center_cells: set[tuple[int, int]] = set()
    if k % 2 == 0:
        axis_col = k // 2
        axis_rows = list(range(row_lo, row_hi + 1))
        random.shuffle(axis_rows)
        for r in axis_rows[: random.randint(0, 2)]:
            point = (r, axis_col)
            if point not in taken and point not in target_set:
                center_cells.add(point)
                taken.add(point)

    mirrored_side = {_reflect_point(k, point) for point in side_cells}
    layer_points = set(orphan_points) | side_cells | mirrored_side | center_cells
    if len(layer_points) != len(orphan_points) + len(side_cells) + len(mirrored_side) + len(center_cells):
        raise RuntimeError("layer overlap")

    coords = list(layer_points)
    best_k, best_score, second_score = _best_axis(coords, w)
    if best_k != k or best_score - second_score < AXIS_GAP_MIN:
        raise RuntimeError("axis not unique enough")

    mirrored_orphans = set()
    pts = {(int(r), int(c)) for r, c in coords}
    for r, c in orphan_points:
        cc = k - c
        if 0 <= cc < w and (r, cc) not in pts:
            mirrored_orphans.add((r, cc))
    if mirrored_orphans != set(target_points):
        raise RuntimeError("mirrored orphan targets mismatch")

    for point in layer_points:
        if point in target_set:
            raise RuntimeError("target point leaked into input")
    return layer_points, orphan_points


def _render_points(shape: tuple[int, int], bg: int, by_color: dict[int, set[tuple[int, int]]]) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for color, points in by_color.items():
        for r, c in points:
            if grid[r, c] != bg:
                raise RuntimeError("overlap while rendering")
            grid[r, c] = int(color)
    return grid


def _oracle_targets(grid: np.ndarray) -> dict[int, set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    h, w = g.shape
    targets: dict[int, set[tuple[int, int]]] = {}
    for color in _colors_present(g, exclude={bg}):
        points = [tuple(map(int, rc)) for rc in np.argwhere(g == color)]
        best_k, _best_score, _second = _best_axis(points, w)
        point_set = set(points)
        mirrored: set[tuple[int, int]] = set()
        for r, c in points:
            cc = best_k - c
            if 0 <= cc < w and (r, cc) not in point_set:
                mirrored.add((r, cc))
        targets[int(color)] = mirrored
    return targets


def _orphan_crop(grid: np.ndarray, bg: int, orphans_by_color: dict[int, set[tuple[int, int]]]) -> np.ndarray:
    tmp = np.full(grid.shape, bg, dtype=int)
    for color, points in orphans_by_color.items():
        for r, c in points:
            tmp[r, c] = int(color)
    return _crop_non_bg(tmp, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reveal the hidden figure by mirroring each color's orphan pixels across that color's vertical axis.

    Input structure:
    - A dominant background contains several non-background color layers.
    - Each color layer is mostly vertically symmetric around its own axis, but also has a few same-color orphan pixels with no mirror partner.
    - Different colors use different vertical axes, and the true mirrored destinations of the orphan pixels are still background in the input.

    Rule:
    - Process each non-background color independently and find the vertical axis that best explains most of that color's pixels.
    - For every pixel of that color that lacks a same-color mirror partner across that axis, place a new same-color pixel at its mirrored position.
    - Combine all mirrored orphan pixels from all colors and return the minimum bounding-box crop of that revealed multi-color figure on the original background color.

    Key insight:
    - The asymmetric pixels are not noise to delete; they are encoded fragments that must first be reflected within their own color layer before the hidden picture appears.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = _get_bg(g)
    h, w = g.shape
    targets: list[tuple[int, int, int]] = []

    for color in _colors_present(g, exclude={bg}):
        points = [tuple(map(int, rc)) for rc in np.argwhere(g == color)]
        assert points, "each listed color must be present"
        best_k, _best_score, _second_score = _best_axis(points, w)
        point_set = set(points)
        for r, c in points:
            cc = best_k - c
            if 0 <= cc < w and (r, cc) not in point_set:
                targets.append((r, cc, int(color)))

    assert targets, "expected at least one orphan-derived target"
    rows = [r for r, _c, _color in targets]
    cols = [c for _r, c, _color in targets]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)

    out = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c, color in targets:
        out[r - r0, c - c0] = int(color)
    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        bg = random.choice(PALETTE)
        n_colors = random.choice([2, 3])
        colors = random.sample([color for color in PALETTE if color != bg], n_colors)
        try:
            expected_output, local_targets = _generate_hidden_output(colors, bg)
        except RuntimeError:
            continue

        out_h, out_w = expected_output.shape
        h = random.randint(max(out_h + 6, 12), min(24, 30))
        min_w = max(out_w + 14, 22)
        if min_w > 30:
            continue
        w = random.randint(min_w, 30)
        fig_r0 = random.randint(2, h - out_h - 2)
        fig_c0 = random.randint(5, w - out_w - 6)
        fig_c1 = fig_c0 + out_w - 1

        target_by_color: dict[int, list[tuple[int, int]]] = {}
        target_set: set[tuple[int, int]] = set()
        for color, points in local_targets.items():
            global_points = [(fig_r0 + r, fig_c0 + c) for r, c in points]
            target_by_color[int(color)] = global_points
            target_set.update(global_points)

        sides = _assign_sides(colors)
        try:
            assignments = _choose_axis_assignments(target_by_color, sides, w, fig_c0, fig_c1)
        except RuntimeError:
            continue

        occupied: set[tuple[int, int]] = set()
        layers: dict[int, set[tuple[int, int]]] = {}
        orphan_points: dict[int, set[tuple[int, int]]] = {}
        success = True
        for color in random.sample(colors, len(colors)):
            try:
                layer, orphans = _build_color_layer(
                    int(color),
                    target_by_color[int(color)],
                    assignments[int(color)],
                    (h, w),
                    target_set,
                    occupied,
                )
            except RuntimeError:
                success = False
                break
            layers[int(color)] = layer
            orphan_points[int(color)] = orphans
            occupied.update(layer)
        if not success:
            continue

        try:
            grid = _render_points((h, w), bg, layers)
        except RuntimeError:
            continue

        oracle_targets = _oracle_targets(grid)
        if set(oracle_targets) != set(target_by_color):
            continue
        if any(oracle_targets[color] != set(target_by_color[color]) for color in target_by_color):
            continue

        if not np.array_equal(main(grid), expected_output):
            continue
        if np.array_equal(_orphan_crop(grid, bg, orphan_points), expected_output):
            continue
        if len({assignments[color]["k"] for color in assignments}) != len(assignments):
            continue
        if len({sides[color] for color in sides}) < 2:
            continue
        if max(grid.shape) > 30 or max(expected_output.shape) > 30:
            continue

        return grid

    raise RuntimeError("failed to generate a valid mirror-reveal input")
