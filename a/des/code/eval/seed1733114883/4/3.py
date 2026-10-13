from __future__ import annotations

import random

import numpy as np


GREY = 5
MAROON = 9
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _most_common_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        rr = int(r + dr)
        cc = int(c + dc)
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc), h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _detect_gate_color(grid: np.ndarray, r: int, c: int, bg: int) -> int | None:
    h, w = grid.shape
    matches: list[int] = []

    if 0 < c < w - 1:
        col = int(grid[r, c - 1])
        if col == int(grid[r, c + 1]) and col not in (bg, GREY, MAROON):
            matches.append(int(col))
    if 0 < r < h - 1:
        col = int(grid[r - 1, c])
        if col == int(grid[r + 1, c]) and col not in (bg, GREY, MAROON):
            matches.append(int(col))

    uniq = sorted(set(matches))
    assert len(uniq) <= 1, "a path pixel may not belong to multiple gates"
    return None if not uniq else int(uniq[0])


def _trace_path(component: list[tuple[int, int]], start: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    comp_set = set(component)
    degree: dict[tuple[int, int], int] = {}
    for r, c in component:
        degree[(int(r), int(c))] = sum((nr, nc) in comp_set for nr, nc in _neighbors4(int(r), int(c), h, w))

    endpoints = [pt for pt, deg in degree.items() if deg == 1]
    assert len(component) >= 2, "expected paths with two distinct endpoints"
    assert len(endpoints) == 2, "each path must have exactly two endpoints"
    assert degree[start] == 1, "maroon start must sit on an endpoint"
    assert all(deg in (1, 2) for deg in degree.values()), "paths may not branch"

    order = [start]
    prev: tuple[int, int] | None = None
    current = start
    while True:
        options = [
            (nr, nc)
            for nr, nc in _neighbors4(current[0], current[1], h, w)
            if (nr, nc) in comp_set and (nr, nc) != prev
        ]
        if not options:
            break
        assert len(options) == 1, "path traversal encountered a branch"
        prev, current = current, options[0]
        order.append(current)

    assert len(order) == len(component), "path traversal must visit the whole component"
    return order


def _analyze_paths(grid: np.ndarray) -> tuple[int, dict[tuple[int, int], int], list[list[tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    path_mask = (g == GREY) | (g == MAROON)
    comps = _components(path_mask)
    assert comps, "expected at least one path"

    gate_by_cell: dict[tuple[int, int], int] = {}
    ordered_paths: list[list[tuple[int, int]]] = []

    for comp in comps:
        maroons = [(r, c) for r, c in comp if int(g[r, c]) == MAROON]
        assert len(maroons) == 1, "each path component must contain exactly one maroon start"
        start = maroons[0]
        ordered = _trace_path(comp, start, g.shape)
        ordered_paths.append(ordered)
        for r, c in ordered:
            gate = _detect_gate_color(g, int(r), int(c), bg)
            if gate is not None:
                gate_by_cell[(int(r), int(c))] = int(gate)

    return bg, gate_by_cell, ordered_paths


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    _bg, gate_by_cell, ordered_paths = _analyze_paths(g)

    for ordered in ordered_paths:
        current_color = GREY
        start = ordered[0]
        for cell in ordered:
            if cell in gate_by_cell:
                current_color = int(gate_by_cell[cell])
            r, c = cell
            if cell == start:
                out[r, c] = MAROON
            else:
                out[r, c] = int(current_color)

    return out


GRID_H_RANGE = (15, 25)
GRID_W_RANGE = (15, 25)
PATH_COUNT_CHOICES = (1, 2, 3)
ROOM_H_RANGE = (7, 13)
ROOM_W_RANGE = (7, 13)
PATH_LEN_MIN = 7
PATH_LEN_MAX = 22
PATH_ATTEMPTS = 120
PATCH_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 120
GENERATE_ATTEMPTS = 400


def _can_extend(path_set: set[tuple[int, int]], prev: tuple[int, int], nxt: tuple[int, int], h: int, w: int) -> bool:
    r, c = nxt
    if not (1 <= r < h - 1 and 1 <= c < w - 1):
        return False
    if nxt in path_set:
        return False
    for nb in _neighbors4(r, c, h, w):
        if nb in path_set and nb != prev:
            return False
    return True


def _grow_path(h: int, w: int, min_len: int, max_len: int) -> list[tuple[int, int]] | None:
    for _ in range(PATH_ATTEMPTS):
        start = (random.randint(1, h - 2), random.randint(1, w - 2))
        path = [start]
        path_set = {start}
        straight_bias = random.uniform(0.35, 0.75)

        while len(path) < max_len:
            current = path[-1]
            candidates = [
                nb for nb in _neighbors4(current[0], current[1], h, w) if _can_extend(path_set, current, nb, h, w)
            ]

            if len(path) >= min_len:
                if not candidates or random.random() < 0.18:
                    break
            elif not candidates:
                break

            if len(path) >= 2:
                prev = path[-2]
                dr = current[0] - prev[0]
                dc = current[1] - prev[1]
                straight = (current[0] + dr, current[1] + dc)
                if straight in candidates and random.random() < straight_bias:
                    chosen = straight
                else:
                    chosen = random.choice(candidates)
            else:
                chosen = random.choice(candidates)

            path.append(chosen)
            path_set.add(chosen)

        if len(path) >= min_len:
            return [(int(r), int(c)) for r, c in path]
    return None


def _gate_candidates(path: list[tuple[int, int]], h: int, w: int) -> list[dict[str, object]]:
    path_set = set(path)
    candidates: list[dict[str, object]] = []
    for i in range(1, len(path) - 1):
        prev = path[i - 1]
        cur = path[i]
        nxt = path[i + 1]

        pair: tuple[tuple[int, int], tuple[int, int]] | None = None
        if prev[0] == cur[0] == nxt[0]:
            up = (cur[0] - 1, cur[1])
            down = (cur[0] + 1, cur[1])
            if up not in path_set and down not in path_set and 0 <= up[0] < h and down[0] < h:
                pair = (up, down)
        elif prev[1] == cur[1] == nxt[1]:
            left = (cur[0], cur[1] - 1)
            right = (cur[0], cur[1] + 1)
            if left not in path_set and right not in path_set and 0 <= left[1] < w and right[1] < w:
                pair = (left, right)

        if pair is not None:
            candidates.append({"index": int(i), "cell": (int(cur[0]), int(cur[1])), "pair": pair})
    return candidates


def _gate_capacity(candidates: list[dict[str, object]]) -> int:
    last = -10**9
    count = 0
    for cand in sorted(candidates, key=lambda x: int(x["index"])):
        idx = int(cand["index"])
        if idx - last <= 1:
            continue
        last = idx
        count += 1
    return int(count)


def _select_gates(candidates: list[dict[str, object]], count: int) -> list[dict[str, object]] | None:
    if count == 0:
        return []
    for _ in range(80):
        chosen: list[dict[str, object]] = []
        used_pairs: set[tuple[int, int]] = set()
        shuffled = candidates[:]
        random.shuffle(shuffled)
        for cand in shuffled:
            idx = int(cand["index"])
            pair = tuple(cand["pair"])
            if any(abs(idx - int(other["index"])) <= 1 for other in chosen):
                continue
            if used_pairs & set(pair):
                continue
            chosen.append(cand)
            used_pairs.update(pair)
            if len(chosen) == count:
                return chosen
    return None


def _crop_with_padding(grid: np.ndarray, bg: int, pad: int = 1) -> np.ndarray:
    pts = np.argwhere(grid != bg)
    assert pts.size > 0
    r0 = max(0, int(pts[:, 0].min()) - pad)
    c0 = max(0, int(pts[:, 1].min()) - pad)
    r1 = min(grid.shape[0] - 1, int(pts[:, 0].max()) + pad)
    c1 = min(grid.shape[1] - 1, int(pts[:, 1].max()) + pad)
    return np.array(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)


def _build_patch(bg: int, gate_palette: list[int], gate_target: int) -> tuple[np.ndarray, np.ndarray, int] | None:
    if len(gate_palette) < gate_target:
        return None

    for _ in range(PATCH_ATTEMPTS):
        h = random.randint(*ROOM_H_RANGE)
        w = random.randint(*ROOM_W_RANGE)
        max_len = min(PATH_LEN_MAX, max(PATH_LEN_MIN, ((h - 2) * (w - 2)) // 2))
        min_len = min(max_len, random.randint(PATH_LEN_MIN, min(PATH_LEN_MAX, max_len)))
        path = _grow_path(h, w, min_len, max_len)
        if path is None:
            continue

        candidates = _gate_candidates(path, h, w)
        if gate_target > _gate_capacity(candidates):
            continue
        selected = _select_gates(candidates, gate_target)
        if selected is None:
            continue

        start_at_head = random.random() < 0.5
        ordered = path if start_at_head else list(reversed(path))
        start = ordered[0]

        inp = np.full((h, w), bg, dtype=int)
        out = np.full((h, w), bg, dtype=int)
        for r, c in path:
            inp[r, c] = GREY
            out[r, c] = GREY
        inp[start] = MAROON
        out[start] = MAROON

        gate_colors = random.sample(gate_palette, gate_target)
        gate_by_cell: dict[tuple[int, int], int] = {}
        for cand, color in zip(selected, gate_colors):
            pair = tuple(cand["pair"])
            gate_by_cell[tuple(cand["cell"])] = int(color)
            for r, c in pair:
                inp[int(r), int(c)] = int(color)
                out[int(r), int(c)] = int(color)

        current = GREY
        for cell in ordered:
            if cell in gate_by_cell:
                current = int(gate_by_cell[cell])
            if cell != start:
                out[cell] = int(current)

        inp = _crop_with_padding(inp, bg, pad=1)
        out = _crop_with_padding(out, bg, pad=1)
        if np.array_equal(main(inp), out):
            return inp, out, gate_target

    return None


def _gate_targets(num_paths: int) -> list[int]:
    targets = [random.choice((0, 1, 2)) for _ in range(num_paths)]
    if num_paths == 1 and random.random() < 0.35:
        targets[0] = 3
    elif num_paths >= 2 and random.random() < 0.35:
        idx = random.randrange(num_paths)
        targets[idx] = min(3, targets[idx] + 1)
    if sum(targets) == 0 and random.random() < 0.5:
        targets[random.randrange(num_paths)] = 1
    while sum(targets) > 6:
        idx = random.randrange(num_paths)
        if targets[idx] > 0:
            targets[idx] -= 1
    return targets


def _can_place_patch(forbidden: np.ndarray, patch: np.ndarray, bg: int, top: int, left: int) -> bool:
    h, w = forbidden.shape
    ph, pw = patch.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return False
    pts = np.argwhere(patch != bg)
    for pr, pc in pts:
        if bool(forbidden[top + int(pr), left + int(pc)]):
            return False
    return True


def _mark_forbidden(forbidden: np.ndarray, patch: np.ndarray, bg: int, top: int, left: int) -> None:
    h, w = forbidden.shape
    for pr, pc in np.argwhere(patch != bg):
        rr = top + int(pr)
        cc = left + int(pc)
        forbidden[rr, cc] = True
        for nr, nc in _neighbors4(rr, cc, h, w):
            forbidden[nr, nc] = True


def _place_patch(dst: np.ndarray, patch: np.ndarray, bg: int, top: int, left: int) -> None:
    pts = np.argwhere(patch != bg)
    for pr, pc in pts:
        dst[top + int(pr), left + int(pc)] = int(patch[int(pr), int(pc)])


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    bg = random.choice([c for c in range(10) if c not in (GREY, MAROON)])
    num_paths = random.choice(PATH_COUNT_CHOICES)
    gate_targets = _gate_targets(num_paths)

    gate_palette = [c for c in range(10) if c not in (bg, GREY, MAROON)]
    random.shuffle(gate_palette)

    patches: list[tuple[np.ndarray, np.ndarray]] = []
    color_cursor = 0
    for gate_target in gate_targets:
        patch = _build_patch(bg, gate_palette[color_cursor:], gate_target)
        if patch is None:
            return None
        inp_patch, out_patch, used = patch
        patches.append((inp_patch, out_patch))
        color_cursor += int(used)

    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)
    forbidden = np.zeros((h, w), dtype=bool)

    for inp_patch, out_patch in random.sample(patches, len(patches)):
        ph, pw = inp_patch.shape
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = random.randint(0, h - ph)
            left = random.randint(0, w - pw)
            if not _can_place_patch(forbidden, inp_patch, bg, top, left):
                continue
            _place_patch(inp, inp_patch, bg, top, left)
            _place_patch(out, out_patch, bg, top, left)
            _mark_forbidden(forbidden, inp_patch, bg, top, left)
            placed = True
            break
        if not placed:
            return None

    bg_count = int(np.count_nonzero(inp == bg))
    fg_count = int(inp.size - bg_count)
    if bg_count <= fg_count:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATE_ATTEMPTS} attempts")
