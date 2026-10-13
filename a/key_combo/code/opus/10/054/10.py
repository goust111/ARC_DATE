from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
TRACK = 8
GATE = 7
ENDPOINT = 9
OBJECT_COLORS = (1, 2, 3, 4, 5, 6)
RESERVED = {BG, GATE, TRACK, ENDPOINT}

GEN_RETRIES = 500
COARSE_RANGE = (4, 6)
COARSE_PATH_LEN = (6, 9)
OBJECT_COUNT_RANGE = (3, 5)
MIN_TURNS = 3


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        coords = []
        while q:
            rr, cc = q.popleft()
            coords.append((rr, cc))
            for nr, nc in _neighbors4(rr, cc):
                if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _extract_objects(grid: np.ndarray) -> list[dict]:
    mask = ~np.isin(grid, list(RESERVED))
    objects = []
    for coords in _component_coords(mask):
        rs = coords[:, 0]
        cs = coords[:, 1]
        box = (int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max()))
        patch = _crop(grid, box)
        colors = sorted(set(map(int, patch.ravel())) - {BG})
        assert box[2] - box[0] == 2 and box[3] - box[1] == 2, "expected 3x3 objects"
        assert int(coords.shape[0]) == 9, "expected filled 3x3 objects"
        assert len(colors) == 2, "expected exactly two colors per object"
        objects.append({"coords": coords, "bbox": box, "colors": tuple(colors)})
    return objects


def _ordered_path(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    endpoints = sorted(tuple(map(int, p)) for p in np.argwhere(g == ENDPOINT))
    assert len(endpoints) == 2, "expected exactly two endpoints"
    path_mask = np.isin(g, [TRACK, GATE, ENDPOINT])
    path_cells = {tuple(map(int, p)) for p in np.argwhere(path_mask)}
    assert path_cells, "expected a path"

    degrees = {}
    for cell in path_cells:
        deg = sum((nbr in path_cells) for nbr in _neighbors4(*cell))
        degrees[cell] = deg

    assert degrees[endpoints[0]] == 1 and degrees[endpoints[1]] == 1, "endpoints must be leaves"
    for cell, deg in degrees.items():
        if cell not in endpoints:
            assert deg == 2, "track must be a single simple path"

    start, end = endpoints
    ordered = [start]
    seen = {start}
    prev = None
    cur = start
    while cur != end:
        nxts = [nbr for nbr in _neighbors4(*cur) if nbr in path_cells and nbr != prev]
        assert len(nxts) == 1, "path must be unique"
        nxt = nxts[0]
        assert nxt not in seen or nxt == end, "path cannot branch or cycle"
        ordered.append(nxt)
        seen.add(nxt)
        prev, cur = cur, nxt

    assert len(seen) == len(path_cells), "all path cells must lie on the unique route"
    return ordered


def _object_anchor(path_set: set[tuple[int, int]], obj: dict) -> tuple[int, int]:
    adj = set()
    for r, c in obj["coords"]:
        for nr, nc in _neighbors4(int(r), int(c)):
            if (nr, nc) in path_set:
                adj.add((nr, nc))
    assert len(adj) == 1, "each object must touch exactly one path position"
    return next(iter(adj))


def _swap_patch(patch: np.ndarray, colors: tuple[int, int]) -> np.ndarray:
    a, b = colors
    out = patch.copy()
    out[patch == a] = b
    out[patch == b] = a
    return out


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    ordered = _ordered_path(g)
    path_set = set(ordered)
    objects = _extract_objects(g)
    anchor_to_object = {}
    for obj in objects:
        anchor = _object_anchor(path_set, obj)
        assert int(g[anchor]) != GATE, "objects cannot be adjacent to gates"
        anchor_to_object[anchor] = obj

    out = g.copy()
    swap_on = False
    for cell in ordered:
        if int(g[cell]) == GATE:
            swap_on = not swap_on
        obj = anchor_to_object.get(cell)
        if obj is not None and swap_on:
            patch = _crop(out, obj["bbox"])
            swapped = _swap_patch(patch, obj["colors"])
            r0, c0, _, _ = obj["bbox"]
            _paste(out, swapped, r0, c0)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Traverse the maroon-to-maroon route and conditionally swap each nearby双色 3x3 object after gate toggles.

    Input structure:
    - A black background contains one 4-connected teal/orange track with exactly two maroon endpoint pixels.
    - Teal (8) cells are ordinary track cells and orange (7) cells are gate cells embedded in the same route.
    - Several filled 3x3 objects sit beside the route; each object uses exactly two non-background, non-track colors.
    - Every object touches exactly one path position, and no object touches an orange gate.

    Rule:
    - Choose the maroon endpoint with the smaller (row, col) coordinate as the start, then follow the unique simple 4-connected path to the other endpoint.
    - Keep a swap switch that starts OFF and flips each time the traversal steps onto an orange gate cell.
    - Whenever the current path position is the unique position adjacent to a 3x3 object, swap that object's two colors if and only if the switch is ON.
    - Leave the background, teal/orange route, and maroon endpoints unchanged.

    Key insight:
    - The local color swap is controlled by global path order: an object changes only when the number of earlier gates on the traversed route is odd.
    """
    return _solve(grid)


def _turn_count(path: list[tuple[int, int]]) -> int:
    turns = 0
    for i in range(1, len(path) - 1):
        a = path[i - 1]
        b = path[i]
        c = path[i + 1]
        d1 = (b[0] - a[0], b[1] - a[1])
        d2 = (c[0] - b[0], c[1] - b[1])
        turns += d1 != d2
    return turns


def _random_coarse_path() -> list[tuple[int, int]] | None:
    coarse_h = random.randint(*COARSE_RANGE)
    coarse_w = random.randint(*COARSE_RANGE)
    target_len = random.randint(*COARSE_PATH_LEN)

    def search(path: list[tuple[int, int]], seen: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if len(path) == target_len:
            return path if _turn_count(path) >= MIN_TURNS else None

        cur = path[-1]
        nbrs = []
        for nr, nc in _neighbors4(*cur):
            if 0 <= nr < coarse_h and 0 <= nc < coarse_w and (nr, nc) not in seen:
                nbrs.append((nr, nc))
        random.shuffle(nbrs)
        for nxt in nbrs:
            res = search(path + [nxt], seen | {nxt})
            if res is not None:
                return res
        return None

    starts = [(r, c) for r in range(coarse_h) for c in range(coarse_w)]
    random.shuffle(starts)
    for start in starts:
        result = search([start], {start})
        if result is not None:
            return result
    return None


def _expand_path(coarse: list[tuple[int, int]]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    coarse_h = max(r for r, _ in coarse) + 1
    coarse_w = max(c for _, c in coarse) + 1
    h = 4 * coarse_h
    w = 4 * coarse_w
    grid = np.full((h, w), BG, dtype=int)

    actual_nodes = [(2 + 4 * r, 2 + 4 * c) for r, c in coarse]
    for (r0, c0), (r1, c1) in zip(actual_nodes, actual_nodes[1:]):
        dr = 0 if r0 == r1 else (1 if r1 > r0 else -1)
        dc = 0 if c0 == c1 else (1 if c1 > c0 else -1)
        rr, cc = r0, c0
        grid[rr, cc] = TRACK
        while (rr, cc) != (r1, c1):
            rr += dr
            cc += dc
            grid[rr, cc] = TRACK

    first = actual_nodes[0]
    second = actual_nodes[1]
    last = actual_nodes[-1]
    prev = actual_nodes[-2]
    start_dir = (second[0] - first[0], second[1] - first[1])
    end_dir = (last[0] - prev[0], last[1] - prev[1])
    start_dir = (0 if start_dir[0] == 0 else start_dir[0] // abs(start_dir[0]), 0 if start_dir[1] == 0 else start_dir[1] // abs(start_dir[1]))
    end_dir = (0 if end_dir[0] == 0 else end_dir[0] // abs(end_dir[0]), 0 if end_dir[1] == 0 else end_dir[1] // abs(end_dir[1]))

    start_ep = (first[0] - start_dir[0], first[1] - start_dir[1])
    end_ep = (last[0] + end_dir[0], last[1] + end_dir[1])
    assert _in_bounds(grid.shape, *start_ep) and _in_bounds(grid.shape, *end_ep)
    grid[start_ep] = ENDPOINT
    grid[end_ep] = ENDPOINT
    ordered = _ordered_path(grid)
    return grid, ordered


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[2] + 1 < b[0] or b[2] + 1 < a[0] or a[3] + 1 < b[1] or b[3] + 1 < a[1]


def _candidate_objects(base: np.ndarray, ordered: list[tuple[int, int]]) -> list[dict]:
    path_set = set(ordered)
    candidates = []
    for idx, (r, c) in enumerate(ordered):
        for r0, c0 in ((r - 2, c + 1), (r, c + 1), (r - 2, c - 3), (r, c - 3)):
            box = (r0, c0, r0 + 2, c0 + 2)
            if r0 < 0 or c0 < 0 or box[2] >= base.shape[0] or box[3] >= base.shape[1]:
                continue
            patch = base[r0 : r0 + 3, c0 : c0 + 3]
            if np.any(patch != BG):
                continue
            adj = set()
            for rr in range(r0, r0 + 3):
                for cc in range(c0, c0 + 3):
                    for nbr in _neighbors4(rr, cc):
                        if nbr in path_set:
                            adj.add(nbr)
            if adj == {ordered[idx]}:
                candidates.append({"anchor_idx": idx, "anchor": ordered[idx], "bbox": box})
    return candidates


def _gate_adjacency(coord: tuple[int, int], boxes: list[tuple[int, int, int, int]]) -> bool:
    r, c = coord
    for r0, c0, r1, c1 in boxes:
        if r0 <= r <= r1 and c0 <= c <= c1:
            return True
        for nr, nc in _neighbors4(r, c):
            if r0 <= nr <= r1 and c0 <= nc <= c1:
                return True
    return False


def _assign_gates(
    ordered: list[tuple[int, int]],
    selection: list[dict],
    boxes: list[tuple[int, int, int, int]],
) -> tuple[list[bool], list[int]] | None:
    gate_slots = [
        idx
        for idx, cell in enumerate(ordered)
        if 0 < idx < len(ordered) - 1 and not _gate_adjacency(cell, boxes)
    ]

    start_choices = [False, True]
    random.shuffle(start_choices)
    for start_state in start_choices:
        desired = [(start_state if i % 2 == 0 else not start_state) for i in range(len(selection))]
        current = False
        gate_idxs: list[int] = []
        ok = True
        prev_anchor = 0
        for obj, want in zip(selection, desired):
            if current != want:
                lo = prev_anchor + 1
                hi = obj["anchor_idx"] - 1
                slots = [idx for idx in gate_slots if lo <= idx <= hi and idx not in gate_idxs]
                if not slots:
                    ok = False
                    break
                gate_idxs.append(random.choice(slots))
                current = not current
            prev_anchor = obj["anchor_idx"]
        if ok and len(gate_idxs) >= 2:
            return desired, gate_idxs
    return None


def _choose_selection(base: np.ndarray, ordered: list[tuple[int, int]]) -> tuple[list[dict], list[bool], list[int]] | None:
    candidates = _candidate_objects(base, ordered)
    by_anchor: dict[int, list[dict]] = {}
    for cand in candidates:
        by_anchor.setdefault(cand["anchor_idx"], []).append(cand)
    anchor_ids = sorted(by_anchor)
    if len(anchor_ids) < OBJECT_COUNT_RANGE[0]:
        return None

    target_count = random.randint(OBJECT_COUNT_RANGE[0], min(OBJECT_COUNT_RANGE[1], len(anchor_ids)))

    def search(pos: int, chosen: list[dict]) -> tuple[list[dict], list[bool], list[int]] | None:
        if len(chosen) == target_count:
            boxes = [item["bbox"] for item in chosen]
            assigned = _assign_gates(ordered, chosen, boxes)
            if assigned is None:
                return None
            desired, gate_idxs = assigned
            return chosen, desired, gate_idxs

        remaining_anchors = len(anchor_ids) - pos
        if len(chosen) + remaining_anchors < target_count:
            return None

        indices = list(range(pos, len(anchor_ids)))
        random.shuffle(indices)
        for i in indices:
            if len(chosen) + (len(anchor_ids) - i) < target_count:
                continue
            anchor = anchor_ids[i]
            options = by_anchor[anchor][:]
            random.shuffle(options)
            for cand in options:
                if any(not _boxes_separated(cand["bbox"], prev["bbox"]) for prev in chosen):
                    continue
                result = search(i + 1, chosen + [cand])
                if result is not None:
                    return result
        return None

    return search(0, [])


def _random_patch(colors: tuple[int, int]) -> np.ndarray:
    a, b = colors
    minority = random.choice([a, b])
    majority = b if minority == a else a
    count = random.randint(2, 4)
    flat = [majority] * 9
    for idx in random.sample(range(9), count):
        flat[idx] = minority
    return np.array(flat, dtype=int).reshape(3, 3)


def _build_expected(inp: np.ndarray, objects: list[dict], swap_flags: list[bool]) -> np.ndarray:
    out = inp.copy()
    for obj, do_swap in zip(objects, swap_flags):
        if not do_swap:
            continue
        r0, c0, _, _ = obj["bbox"]
        patch = _crop(out, obj["bbox"])
        _paste(out, _swap_patch(patch, obj["colors"]), r0, c0)
    return out


def _attempt_generate() -> np.ndarray | None:
    coarse = _random_coarse_path()
    if coarse is None:
        return None

    base, ordered = _expand_path(coarse)
    chosen_pack = _choose_selection(base, ordered)
    if chosen_pack is None:
        return None
    chosen, desired, gate_idxs = chosen_pack

    inp = base.copy()
    for gate_idx in gate_idxs:
        r, c = ordered[gate_idx]
        inp[r, c] = GATE

    color_pairs = random.sample(
        [(a, b) for i, a in enumerate(OBJECT_COLORS) for b in OBJECT_COLORS[i + 1 :]],
        len(chosen),
    )
    object_specs = []
    for geom, pair in zip(chosen, color_pairs):
        patch = _random_patch(pair)
        r0, c0, _, _ = geom["bbox"]
        _paste(inp, patch, r0, c0)
        object_specs.append({"bbox": geom["bbox"], "colors": pair, "anchor_idx": geom["anchor_idx"]})

    gate_counts = [sum(g < obj["anchor_idx"] for g in gate_idxs) for obj in object_specs]
    assert any(desired) and any(not flag for flag in desired)
    assert any((not flag) and gates_before > 0 for flag, gates_before in zip(desired, gate_counts))

    expected = _build_expected(inp, object_specs, desired)
    if np.array_equal(expected, inp):
        return None
    if np.array_equal(_build_expected(inp, object_specs, [True] * len(object_specs)), expected):
        return None
    if np.array_equal(main(inp), expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
