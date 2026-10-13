from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
ACTIVE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GENERATE_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr = cr + dr
                        nc = cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _classify_scene(grid: np.ndarray):
    towers: list[dict[str, object]] = []
    receivers: list[dict[str, object]] = []
    objects: list[dict[str, object]] = []

    for coords in _components(grid != BG):
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        assert coords.shape[0] == 9, "every object must occupy exactly 9 cells"
        assert (r1 - r0, c1 - c0) == (2, 2), "every object must be a 3x3 block"
        window = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
        assert np.all(window != BG), "objects must be fully non-black"

        center = (r0 + 1, c0 + 1)
        flat = window.reshape(-1)
        ring = np.array(
            [
                window[0, 0],
                window[0, 1],
                window[0, 2],
                window[1, 0],
                window[1, 2],
                window[2, 0],
                window[2, 1],
                window[2, 2],
            ],
            dtype=int,
        )

        if np.all(flat == flat[0]):
            tower = {
                "kind": "tower",
                "bbox": (r0, c0, r1, c1),
                "center": center,
                "color": int(flat[0]),
            }
            towers.append(tower)
            objects.append(tower)
            continue

        assert np.all(ring == ring[0]), "receiver ring must be monochrome"
        assert int(window[1, 1]) != int(ring[0]), "receiver center must differ from the ring"
        receiver = {
            "kind": "receiver",
            "bbox": (r0, c0, r1, c1),
            "center": center,
            "ring": int(ring[0]),
            "filter": int(window[1, 1]),
        }
        receivers.append(receiver)
        objects.append(receiver)

    assert towers, "expected at least one signal tower"
    return towers, receivers, objects


def _objects_on_ray(
    start: tuple[int, int],
    step: tuple[int, int],
    objects: list[dict[str, object]],
    source_center: tuple[int, int],
) -> list[tuple[tuple[int, int], tuple[int, int], dict[str, object]]]:
    start_r, start_c = start
    dr, dc = step
    hits: list[tuple[tuple[int, int], tuple[int, int], dict[str, object]]] = []

    for obj in objects:
        if tuple(obj["center"]) == source_center:
            continue
        r0, c0, r1, c1 = obj["bbox"]
        if dr == 0:
            if not (r0 <= start_r <= r1):
                continue
            if dc > 0:
                if c0 <= start_c:
                    continue
                near = (start_r, c0)
                far = (start_r, c1)
            else:
                if c1 >= start_c:
                    continue
                near = (start_r, c1)
                far = (start_r, c0)
        else:
            if not (c0 <= start_c <= c1):
                continue
            if dr > 0:
                if r0 <= start_r:
                    continue
                near = (r0, start_c)
                far = (r1, start_c)
            else:
                if r1 >= start_r:
                    continue
                near = (r1, start_c)
                far = (r0, start_c)
        hits.append((near, far, obj))

    if dr != 0:
        hits.sort(key=lambda item: item[0][0], reverse=dr < 0)
    else:
        hits.sort(key=lambda item: item[0][1], reverse=dc < 0)
    return hits


def _paint_between(
    out: np.ndarray,
    object_mask: np.ndarray,
    current: tuple[int, int],
    stop: tuple[int, int],
    step: tuple[int, int],
    color: int,
) -> None:
    h, w = out.shape
    dr, dc = step
    rr = current[0] + dr
    cc = current[1] + dc
    while (rr, cc) != stop:
        if 0 <= rr < h and 0 <= cc < w and not object_mask[rr, cc]:
            out[rr, cc] = color
        rr += dr
        cc += dc


def _paint_to_edge(
    out: np.ndarray,
    object_mask: np.ndarray,
    current: tuple[int, int],
    step: tuple[int, int],
    color: int,
) -> None:
    h, w = out.shape
    dr, dc = step
    rr = current[0] + dr
    cc = current[1] + dc
    while 0 <= rr < h and 0 <= cc < w:
        if not object_mask[rr, cc]:
            out[rr, cc] = color
        rr += dr
        cc += dc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Broadcast colored rays from solid 3x3 towers and recolor each ray after every 3x3 receiver it crosses.

    Input structure:
    - The background is black.
    - Each signal tower is a solid 3x3 monochrome block.
    - Each receiver is a filled 3x3 block whose outer ring is one color and whose center pixel is a different filter color.
    - Towers and receivers are separated by black space.

    Rule:
    - Keep every input object unchanged in the output.
    - Process towers in top-to-bottom, then left-to-right order. From the center cell of each tower's four sides, emit a one-pixel-wide ray to the grid edge in that cardinal direction.
    - Rays paint only black cells: if a ray reaches any tower or receiver cell, it skips those object cells and continues past them.
    - A ray starts in the tower color. Whenever it crosses a receiver's 3x3 region, the color changes to the receiver center color immediately after that receiver, and stays that way until another receiver is crossed.
    - If multiple rays paint the same black cell, the later tower in row-major order overwrites the earlier one.

    Key insight:
    - The puzzle combines two independent mechanisms: full-row/full-column broadcasting from the 3x3 towers, and stateful color modulation determined by each crossed receiver's center color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    towers, _receivers, objects = _classify_scene(g)
    object_mask = g != BG
    out = g.copy()

    towers = sorted(towers, key=lambda obj: tuple(obj["center"]))
    for tower in towers:
        tr, tc = tower["center"]
        base_color = int(tower["color"])
        starts = [
            ((tr - 1, tc), (-1, 0)),
            ((tr + 1, tc), (1, 0)),
            ((tr, tc - 1), (0, -1)),
            ((tr, tc + 1), (0, 1)),
        ]
        for start, step in starts:
            ray_color = base_color
            current = start
            for near, far, obj in _objects_on_ray(start, step, objects, tower["center"]):
                _paint_between(out, object_mask, current, near, step, ray_color)
                if obj["kind"] == "receiver":
                    ray_color = int(obj["filter"])
                current = far
            _paint_to_edge(out, object_mask, current, step, ray_color)

    return out


def _sample_axis(length: int, count: int, min_gap: int = 5) -> list[int] | None:
    if length < 3 + min_gap * (count - 1):
        return None
    positions: list[int] = []
    low = 1
    for remaining in range(count, 0, -1):
        high = (length - 2) - min_gap * (remaining - 1)
        if low > high:
            return None
        pos = random.randint(low, high)
        positions.append(pos)
        low = pos + min_gap
    return positions


def _receiver_counts(
    towers: list[dict[str, object]],
    receivers: list[dict[str, object]],
) -> list[int]:
    counts: list[int] = []
    for tower in towers:
        tr, tc = tower["center"]
        counts.append(sum(receiver["center"][0] == tr and receiver["center"][1] < tc for receiver in receivers))
        counts.append(sum(receiver["center"][0] == tr and receiver["center"][1] > tc for receiver in receivers))
        counts.append(sum(receiver["center"][1] == tc and receiver["center"][0] < tr for receiver in receivers))
        counts.append(sum(receiver["center"][1] == tc and receiver["center"][0] > tr for receiver in receivers))
    return counts


def _tower_colors(count: int) -> list[int]:
    colors = ACTIVE_COLORS[:]
    random.shuffle(colors)
    return colors[:count]


def _filter_colors(count: int, tower_colors: list[int]) -> list[int]:
    palette = [color for color in ACTIVE_COLORS if color not in tower_colors]
    if not palette:
        palette = ACTIVE_COLORS[:]
    random.shuffle(palette)
    colors: list[int] = []
    while len(colors) < count:
        colors.extend(palette)
        random.shuffle(palette)
    return colors[:count]


def _render_scene(
    shape: tuple[int, int],
    towers: list[dict[str, object]],
    receivers: list[dict[str, object]],
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for tower in towers:
        r, c = tower["center"]
        grid[r - 1 : r + 2, c - 1 : c + 2] = int(tower["color"])
    for receiver in receivers:
        r, c = receiver["center"]
        grid[r - 1 : r + 2, c - 1 : c + 2] = GREY
        grid[r, c] = int(receiver["filter"])
    return grid


def _build_attempt(branch: str) -> np.ndarray | None:
    if branch == "simple":
        tower_count = random.choice([2, 3])
    else:
        tower_count = random.choice([3, 4])

    min_side = max(16, 3 + 5 * (tower_count - 1))
    h = random.randint(min_side, 30)
    w = random.randint(min_side, 30)
    rows = _sample_axis(h, tower_count)
    cols = _sample_axis(w, tower_count)
    if rows is None or cols is None:
        return None

    perm = list(range(tower_count))
    random.shuffle(perm)
    tower_colors = _tower_colors(tower_count)
    towers = [
        {"center": (rows[i], cols[perm[i]]), "color": tower_colors[i]}
        for i in range(tower_count)
    ]

    tower_pairs = {(i, perm[i]) for i in range(tower_count)}
    free_pairs = [
        (i, j)
        for i in range(tower_count)
        for j in range(tower_count)
        if (i, j) not in tower_pairs
    ]
    max_receivers = min(len(free_pairs) - 1, 5)
    if max_receivers < 1:
        return None

    chosen_pairs: set[tuple[int, int]] = set()
    if branch == "simple":
        target = 1 if tower_count == 2 else random.randint(1, min(3, max_receivers))
        while len(chosen_pairs) < target:
            chosen_pairs.add(random.choice(free_pairs))
    elif branch == "row_multi":
        extreme_rows = [i for i in range(tower_count) if perm[i] in (0, tower_count - 1)]
        row_i = random.choice(extreme_rows)
        same_ray = [j for j in range(tower_count) if j != perm[row_i]]
        random.shuffle(same_ray)
        for j in same_ray[: random.randint(2, len(same_ray))]:
            chosen_pairs.add((row_i, j))
        extras = [pair for pair in free_pairs if pair not in chosen_pairs]
        extra_cap = min(max_receivers - len(chosen_pairs), 2)
        if extras and extra_cap > 0:
            extra_count = random.randint(0, extra_cap)
            if extra_count:
                chosen_pairs.update(random.sample(extras, extra_count))
    else:
        target = random.randint(2, max_receivers)
        chosen_pairs = set(random.sample(free_pairs, target))

    if not chosen_pairs or len(chosen_pairs) >= len(free_pairs):
        return None

    receiver_pairs = sorted(chosen_pairs)
    filters = _filter_colors(len(receiver_pairs), tower_colors)
    receivers = [
        {"center": (rows[i], cols[j]), "filter": filters[idx]}
        for idx, (i, j) in enumerate(receiver_pairs)
    ]

    counts = _receiver_counts(towers, receivers)
    if not any(count == 0 for count in counts):
        return None
    if not any(count > 0 for count in counts):
        return None
    if branch == "row_multi" and not any(count > 1 for count in counts):
        return None

    grid = _render_scene((h, w), towers, receivers)
    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    if np.count_nonzero(out != grid) < 12:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        branch = random.choice(["simple", "row_multi", "cross_mix"])
        grid = _build_attempt(branch)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-tower puzzle")
