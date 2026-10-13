from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

MIN_OBJECTS = 3
MAX_OBJECTS = 5
TOP_LENGTHS = (6, 7, 8)
TAIL_LENGTHS = (2, 3)
SCENE_PADDING_MIN = 2
SCENE_PADDING_MAX = 6
GAP_ROWS = (1, 2, 3)
GEN_ATTEMPTS = 400
PATH_ATTEMPTS = 1200
TAIL_ATTEMPTS = 1200


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in NEIGHBORS4]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        pts: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            pts.append((cr, cc))
            for dr, dc in NEIGHBORS4:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(pts)
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    comps = _components(mask)
    return len(comps) == 1


def _shortest_path(
    mask: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = mask.shape
    queue: deque[tuple[int, int]] = deque([start])
    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while queue:
        cur = queue.popleft()
        if cur == goal:
            break
        for dr, dc in NEIGHBORS4:
            nr = cur[0] + dr
            nc = cur[1] + dc
            nxt = (nr, nc)
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and nxt not in prev:
                prev[nxt] = cur
                queue.append(nxt)
    assert goal in prev, "expected a connector path inside the upper shape"

    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _read_band(grid: np.ndarray) -> tuple[list[int], np.ndarray]:
    band_cols = np.flatnonzero(grid[-1] != BG)
    assert band_cols.size > 0, "expected a bottom color band"
    assert bool(np.all(np.diff(band_cols) == 1)), "band must be contiguous"
    order = [int(grid[-1, c]) for c in band_cols]
    assert len(order) == len(set(order)), "band colors must be unique"
    return order, band_cols


def _adjacent_upper_cells(
    comp: list[tuple[int, int]],
    upper_mask: np.ndarray,
) -> set[tuple[int, int]]:
    h, w = upper_mask.shape
    out: set[tuple[int, int]] = set()
    for r, c in comp:
        for dr, dc in NEIGHBORS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and upper_mask[nr, nc]:
                out.add((nr, nc))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover each path-like shape by connecting its two visible pieces through the
    next higher layer, then repaint the recovered shapes in reverse order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert int(g.min()) >= 0

    h, w = g.shape
    assert h >= 4 and w >= 4

    order, band_cols = _read_band(g)
    scene = g[:-1]
    present = {int(v) for v in np.unique(scene) if int(v) != BG}
    assert present <= set(order), "scene colors must come from the band"

    full_masks: dict[int, np.ndarray] = {}
    top_color = order[-1]
    top_mask = scene == top_color
    assert top_mask.any(), "top shape must be visible"
    assert _is_connected(top_mask), "top shape must be connected"
    full_masks[top_color] = top_mask

    for idx in range(len(order) - 2, -1, -1):
        color = order[idx]
        upper_color = order[idx + 1]

        seed_mask = scene == color
        assert seed_mask.any(), "every shape must remain at least partly visible"
        seed_components = _components(seed_mask)
        assert len(seed_components) == 2, "lower shapes should appear as two visible pieces"

        upper_mask = full_masks[upper_color]
        attachments: list[tuple[int, int]] = []
        for comp in seed_components:
            adj = _adjacent_upper_cells(comp, upper_mask)
            assert len(adj) == 1, "each visible piece should touch the hidden upper segment once"
            attachments.append(next(iter(adj)))

        connector = _shortest_path(upper_mask, attachments[0], attachments[1])
        full_mask = seed_mask.copy()
        for r, c in connector:
            full_mask[r, c] = True

        assert _is_connected(full_mask), "recovered shape must be connected"
        full_masks[color] = full_mask

    out = np.full_like(g, BG)
    canvas = out[:-1]
    for color in reversed(order):
        canvas[full_masks[color]] = color
    out[-1, band_cols] = np.array(order[::-1], dtype=int)
    return out


def _grow_path(length: int, blocked: set[tuple[int, int]] | None = None) -> list[tuple[int, int]] | None:
    blocked = set() if blocked is None else set(blocked)
    for _ in range(PATH_ATTEMPTS):
        path = [(0, 0)]
        used = {(0, 0)}
        while len(path) < length:
            tip = path[-1]
            dirs = list(NEIGHBORS4)
            random.shuffle(dirs)
            candidates: list[tuple[int, int]] = []
            for dr, dc in dirs:
                nxt = (tip[0] + dr, tip[1] + dc)
                if nxt in used or nxt in blocked:
                    continue
                if {p for p in _neighbors4(nxt) if p in used} != {tip}:
                    continue
                if any(p in blocked for p in _neighbors4(nxt)):
                    continue
                candidates.append(nxt)
            if not candidates:
                break
            choice = random.choice(candidates)
            path.append(choice)
            used.add(choice)
        if len(path) == length:
            return path
    return None


def _grow_tail(
    base: tuple[int, int],
    length: int,
    blocked_above: set[tuple[int, int]],
    used_now: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    blocked_above = set(blocked_above)
    used_base = set(used_now)
    for _ in range(TAIL_ATTEMPTS):
        tail: list[tuple[int, int]] = []
        used = set(used_base)
        tip = base
        ok = True
        for _step in range(length):
            dirs = list(NEIGHBORS4)
            random.shuffle(dirs)
            candidates: list[tuple[int, int]] = []
            for dr, dc in dirs:
                nxt = (tip[0] + dr, tip[1] + dc)
                if nxt in used or nxt in blocked_above:
                    continue
                if {p for p in _neighbors4(nxt) if p in used} != {tip}:
                    continue
                touching_above = {p for p in _neighbors4(nxt) if p in blocked_above}
                allowed_above = {tip} if tip in blocked_above else set()
                if touching_above != allowed_above:
                    continue
                candidates.append(nxt)
            if not candidates:
                ok = False
                break
            tip = random.choice(candidates)
            tail.append(tip)
            used.add(tip)
        if ok:
            return tail
    return None


def _valid_overlap_segments(host_segment: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    host_set = set(host_segment)
    candidates: list[list[tuple[int, int]]] = []
    for i in range(len(host_segment) - 1):
        a = host_segment[i]
        b = host_segment[i + 1]
        good = True
        for endpoint, other in ((a, b), (b, a)):
            found_exit = False
            for nbr in _neighbors4(endpoint):
                if nbr == other or nbr in host_set:
                    continue
                if any(side in host_set for side in _neighbors4(nbr) if side != endpoint):
                    continue
                found_exit = True
                break
            if not found_exit:
                good = False
                break
        if good:
            candidates.append([a, b])
    return candidates


def _shift_path(path: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in path]


def _render_paths(
    paths_by_color: dict[int, list[tuple[int, int]]],
    order: list[int],
    height: int,
    width: int,
    band_start: int,
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((height, width), BG, dtype=int)
    out = np.full((height, width), BG, dtype=int)

    for color in order:
        for r, c in paths_by_color[color]:
            inp[r, c] = color

    for color in reversed(order):
        for r, c in paths_by_color[color]:
            out[r, c] = color

    band_end = band_start + len(order)
    inp[-1, band_start:band_end] = np.array(order, dtype=int)
    out[-1, band_start:band_end] = np.array(order[::-1], dtype=int)
    return inp, out


def _build_chain() -> tuple[list[int], dict[int, list[tuple[int, int]]]] | None:
    count = random.randint(MIN_OBJECTS, MAX_OBJECTS)
    order = random.sample(COLORS, count)

    level_to_path: dict[int, list[tuple[int, int]]] = {}
    visible_segments: dict[int, list[list[tuple[int, int]]]] = {}

    top_path = _grow_path(random.choice(TOP_LENGTHS))
    if top_path is None:
        return None
    level_to_path[count] = top_path
    visible_segments[count] = [top_path]

    for level in range(count - 1, 0, -1):
        overlap_options: list[list[tuple[int, int]]] = []
        for seg in visible_segments[level + 1]:
            if len(seg) >= 2:
                overlap_options.extend(_valid_overlap_segments(seg))
        random.shuffle(overlap_options)

        built = False
        for overlap in overlap_options:
            t1_len = random.choice(TAIL_LENGTHS)
            t2_len = random.choice(TAIL_LENGTHS)
            used_now = set(overlap)
            upper_path = set(level_to_path[level + 1])

            tail1 = _grow_tail(overlap[0], t1_len, upper_path, used_now)
            if tail1 is None:
                continue
            used_now.update(tail1)

            tail2 = _grow_tail(overlap[1], t2_len, upper_path, used_now)
            if tail2 is None:
                continue

            level_to_path[level] = list(reversed(tail1)) + overlap + tail2
            visible_segments[level] = [list(reversed(tail1)), tail2]
            built = True
            break

        if not built:
            return None

    paths_by_color = {
        order[level - 1]: level_to_path[level]
        for level in range(1, count + 1)
    }
    return order, paths_by_color


def _place_chain(
    order: list[int],
    paths_by_color: dict[int, list[tuple[int, int]]],
) -> tuple[np.ndarray, np.ndarray]:
    all_cells = [cell for path in paths_by_color.values() for cell in path]
    rows = [r for r, _ in all_cells]
    cols = [c for _, c in all_cells]
    box_h = max(rows) - min(rows) + 1
    box_w = max(cols) - min(cols) + 1

    scene_h = random.randint(box_h + SCENE_PADDING_MIN, box_h + SCENE_PADDING_MAX)
    scene_w_min = max(box_w + SCENE_PADDING_MIN, len(order) + 4)
    scene_w = random.randint(scene_w_min, scene_w_min + (SCENE_PADDING_MAX - SCENE_PADDING_MIN))

    r_shift = random.randint(0, scene_h - box_h) - min(rows)
    c_shift = random.randint(0, scene_w - box_w) - min(cols)

    shifted = {
        color: _shift_path(path, r_shift, c_shift)
        for color, path in paths_by_color.items()
    }

    gap = random.choice(GAP_ROWS)
    height = scene_h + gap + 1
    width = scene_w
    band_start = random.randint(0, width - len(order))
    return _render_paths(shifted, order, height, width, band_start)


def _gen_attempt() -> np.ndarray | None:
    built = _build_chain()
    if built is None:
        return None
    order, paths_by_color = built
    inp, expected = _place_chain(order, paths_by_color)

    try:
        actual = main(inp)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
