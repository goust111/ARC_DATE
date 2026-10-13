from __future__ import annotations

from collections import deque
import random
import numpy as np


BG = 0
MARKER = 1
FG = 7
CHAIN_COLOR = 8

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
DIAG4 = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 120


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _label_mask(mask: np.ndarray, connectivity: int) -> tuple[np.ndarray, int]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    dirs = DIR4 if connectivity == 4 else DIR8
    cur = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            cur += 1
            q = deque([(r, c)])
            labels[r, c] = cur
            while q:
                rr, cc = q.popleft()
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = cur
                        q.append((nr, nc))
    return labels, cur


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _trace_path(pts: set[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]] | None:
    path = [start]
    prev = None
    cur = start
    while True:
        nxt = []
        for dr, dc in DIR4:
            cand = (cur[0] + dr, cur[1] + dc)
            if cand in pts and cand != prev:
                nxt.append(cand)
        if not nxt:
            break
        if len(nxt) != 1:
            return None
        prev, cur = cur, nxt[0]
        path.append(cur)
        if len(path) > len(pts):
            return None
    return path if len(path) == len(pts) else None


def _is_single_staircase(coords: np.ndarray) -> bool:
    pts = {tuple(map(int, p)) for p in coords}
    if len(pts) < 4:
        return False
    degrees = {}
    for r, c in pts:
        degree = sum((r + dr, c + dc) in pts for dr, dc in DIR4)
        if degree == 0 or degree > 2:
            return False
        degrees[(r, c)] = degree
    ends = [p for p, degree in degrees.items() if degree == 1]
    if len(ends) != 2:
        return False

    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 2 or w < 2 or len(pts) != h + w - 1:
        return False

    options = [
        ((r0, c0), (r1, c1), {(1, 0), (0, 1)}),
        ((r0, c1), (r1, c0), {(1, 0), (0, -1)}),
    ]
    for start, end, allowed in options:
        if set(ends) != {start, end}:
            continue
        order = _trace_path(pts, start)
        if order is None or order[-1] != end:
            continue
        ok = True
        for i in range(len(order) - 1):
            step = (order[i + 1][0] - order[i][0], order[i + 1][1] - order[i][1])
            if step not in allowed:
                ok = False
                break
        if ok:
            return True
    return False


def _diagonal_contacts(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r, c in a:
        for dr, dc in DIAG4:
            if (r + dr, c + dc) in b:
                return True
    return False


def _is_diagonal_chain(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    lab4, n4 = _label_mask(mask, connectivity=4)
    if n4 == 1:
        return _is_single_staircase(coords)

    parts = []
    for k in range(1, n4 + 1):
        part_coords = np.argwhere(lab4 == k)
        parts.append({tuple(map(int, p)) for p in part_coords})

    graph = [set() for _ in range(n4)]
    for i in range(n4):
        for j in range(i + 1, n4):
            if _diagonal_contacts(parts[i], parts[j]):
                graph[i].add(j)
                graph[j].add(i)

    if any(len(nei) > 2 for nei in graph):
        return False

    seen = {0}
    q = deque([0])
    while q:
        i = q.popleft()
        for j in graph[i]:
            if j not in seen:
                seen.add(j)
                q.append(j)
    return len(seen) == n4


def _marker_assignments(grid: np.ndarray, labels8: np.ndarray, n8: int) -> tuple[dict[int, list[tuple[int, int]]], list[tuple[int, int]]]:
    assigned = {k: [] for k in range(1, n8 + 1)}
    isolated = []
    h, w = grid.shape
    for r, c in map(tuple, np.argwhere(grid == MARKER)):
        owners = set()
        for dr, dc in DIR8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                lab = int(labels8[nr, nc])
                if lab > 0:
                    owners.add(lab)
        if len(owners) == 1:
            assigned[next(iter(owners))].append((int(r), int(c)))
        else:
            isolated.append((int(r), int(c)))
    return assigned, isolated


def _analyze(grid: np.ndarray) -> list[dict]:
    orange = grid == FG
    labels8, n8 = _label_mask(orange, connectivity=8)
    assigned, _isolated = _marker_assignments(grid, labels8, n8)
    info = []
    for k in range(1, n8 + 1):
        mask = labels8 == k
        coords = np.argwhere(mask)
        markers = assigned[k]
        info.append(
            {
                "label": k,
                "coords": coords,
                "marker_coords": markers,
                "marker_count": len(markers),
                "is_chain": _is_diagonal_chain(mask),
            }
        )
    return info


def main(grid: np.ndarray) -> np.ndarray:
    """Classify orange regions by diagonal-chain topology, then recolor or erase them based on attached blue markers.

    Input structure:
    - The grid is mostly black background with orange foreground regions and blue marker pixels.
    - Orange pixels form several separated 8-connected regions; some blue pixels sit diagonally adjacent to exactly one orange region, while other blue singletons may be isolated.

    Rule:
    - For each orange 8-connected region, split its orange pixels into 4-connected lobes.
    - If those lobes form a connected non-branching diagonal chain or ring, or the whole region is a single monotone diagonal staircase, recolor that region and its attached blue markers to cyan.
    - Otherwise count the attached blue markers: regions with two or more are erased to black, while regions with zero or one stay unchanged.
    - Blue pixels not attached to any orange region remain where they are.

    Key insight:
    - The same blue marker is either irrelevant decoration or a deletion trigger depending on the orange region's topology, so topology must be resolved before counting markers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    orange = g == FG
    labels8, n8 = _label_mask(orange, connectivity=8)
    assigned, _isolated = _marker_assignments(g, labels8, n8)

    out = g.copy()
    for k in range(1, n8 + 1):
        mask = labels8 == k
        coords = np.argwhere(mask)
        markers = assigned[k]
        if _is_diagonal_chain(mask):
            out[coords[:, 0], coords[:, 1]] = CHAIN_COLOR
            for r, c in markers:
                out[r, c] = CHAIN_COLOR
        elif len(markers) >= 2:
            out[coords[:, 0], coords[:, 1]] = BG
            for r, c in markers:
                out[r, c] = BG
    return out


def _normalize(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return {(r - min_r, c - min_c) for r, c in coords}


def _coords_to_mask(coords: set[tuple[int, int]]) -> np.ndarray:
    coords = _normalize(coords)
    max_r = max(r for r, _ in coords)
    max_c = max(c for _, c in coords)
    mask = np.zeros((max_r + 1, max_c + 1), dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


def _mask_to_coords(mask: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, p)) for p in np.argwhere(mask)}


def _random_transform(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    mask = _coords_to_mask(coords)
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    mask = np.rot90(mask, random.randrange(4))
    return _mask_to_coords(mask)


def _staircase_shape() -> set[tuple[int, int]]:
    h = random.randint(2, 4)
    w = random.randint(2, 5)
    moves = [(1, 0)] * (h - 1) + [(0, 1)] * (w - 1)
    random.shuffle(moves)
    r = c = 0
    pts = {(r, c)}
    for dr, dc in moves:
        r += dr
        c += dc
        pts.add((r, c))
    return _random_transform(pts)


def _singleton_chain_shape() -> set[tuple[int, int]]:
    length = random.randint(4, 6)
    col = 0
    pts = [(0, col)]
    for r in range(1, length):
        col += random.choice([-1, 1])
        pts.append((r, col))
    min_c = min(c for _, c in pts)
    pts = {(r, c - min_c) for r, c in pts}
    return _random_transform(pts)


def _rect_shape() -> set[tuple[int, int]]:
    h = random.randint(2, 3)
    w = random.randint(2, 4)
    return {(r, c) for r in range(h) for c in range(w)}


def _plus_shape() -> set[tuple[int, int]]:
    return _random_transform({(1, 0), (1, 1), (1, 2), (0, 1), (2, 1)})


def _tee_shape() -> set[tuple[int, int]]:
    return _random_transform({(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)})


def _branching_diagonal_shape() -> set[tuple[int, int]]:
    return _random_transform({(1, 1), (0, 0), (0, 2), (2, 2), (2, 0)})


def _make_chain_shape() -> set[tuple[int, int]]:
    for _ in range(30):
        coords = _staircase_shape() if random.random() < 0.5 else _singleton_chain_shape()
        mask = _coords_to_mask(coords)
        if _is_diagonal_chain(mask):
            return coords
    raise RuntimeError("failed to make chain shape")


def _make_nonchain_shape() -> set[tuple[int, int]]:
    makers = [_rect_shape, _plus_shape, _tee_shape, _branching_diagonal_shape]
    for _ in range(40):
        coords = random.choice(makers)()
        mask = _coords_to_mask(coords)
        if not _is_diagonal_chain(mask):
            return coords
    raise RuntimeError("failed to make non-chain shape")


def _marker_slots(coords: set[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = _normalize(coords)
    max_r = max(r for r, _ in pts)
    max_c = max(c for _, c in pts)
    slots = []
    for r in range(-1, max_r + 2):
        for c in range(-1, max_c + 2):
            if (r, c) in pts:
                continue
            has_diag = any((r + dr, c + dc) in pts for dr, dc in DIAG4)
            has_card = any((r + dr, c + dc) in pts for dr, dc in DIR4)
            if has_diag and not has_card:
                slots.append((r, c))
    random.shuffle(slots)
    return slots


def _build_patch(kind: str, marker_count: int) -> dict | None:
    for _ in range(80):
        orange = _make_chain_shape() if kind == "chain" else _make_nonchain_shape()
        orange = _normalize(orange)
        slots = _marker_slots(orange)
        if len(slots) < marker_count:
            continue
        chosen_markers = slots[:marker_count]
        all_cells = set(orange) | set(chosen_markers)
        min_r = min(r for r, _ in all_cells)
        min_c = min(c for _, c in all_cells)
        orange = {(r - min_r, c - min_c) for r, c in orange}
        markers = {(r - min_r, c - min_c) for r, c in chosen_markers}
        all_cells = orange | markers
        h = max(r for r, _ in all_cells) + 1
        w = max(c for _, c in all_cells) + 1
        patch = np.zeros((h, w), dtype=int)
        for r, c in orange:
            patch[r, c] = FG
        for r, c in markers:
            patch[r, c] = MARKER
        orange_mask = patch == FG
        if _is_diagonal_chain(orange_mask) != (kind == "chain"):
            continue
        return {"patch": patch, "orange": orange, "markers": markers, "kind": kind, "marker_count": marker_count}
    return None


def _can_place(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    h, w = patch.shape
    gh, gw = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > gh or c0 + w > gw:
        return False
    for pr in range(h):
        for pc in range(w):
            if patch[pr, pc] == BG:
                continue
            gr, gc = r0 + pr, c0 + pc
            for rr in range(gr - 1, gr + 2):
                for cc in range(gc - 1, gc + 2):
                    if 0 <= rr < gh and 0 <= cc < gw and grid[rr, cc] != BG:
                        return False
    return True


def _place_patch(grid: np.ndarray, patch: np.ndarray) -> bool:
    gh, gw = grid.shape
    h, w = patch.shape
    positions = [(r, c) for r in range(gh - h + 1) for c in range(gw - w + 1)]
    random.shuffle(positions)
    for r0, c0 in positions[:PLACE_ATTEMPTS]:
        if not _can_place(grid, patch, r0, c0):
            continue
        mask = patch != BG
        grid[r0 : r0 + h, c0 : c0 + w][mask] = patch[mask]
        return True
    return False


def _place_isolated_markers(grid: np.ndarray) -> None:
    gh, gw = grid.shape
    count = random.randint(0, 2)
    cells = [(r, c) for r in range(gh) for c in range(gw)]
    random.shuffle(cells)
    placed = 0
    for r, c in cells:
        if grid[r, c] != BG:
            continue
        near_fg = False
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < gh and 0 <= cc < gw and grid[rr, cc] != BG:
                    near_fg = True
                    break
            if near_fg:
                break
        if near_fg:
            continue
        grid[r, c] = MARKER
        placed += 1
        if placed == count:
            break


def _generation_ok(grid: np.ndarray) -> bool:
    info = _analyze(grid)
    if not (5 <= len(info) <= 7):
        return False
    chain = [x for x in info if x["is_chain"]]
    nonchain = [x for x in info if not x["is_chain"]]
    if len(chain) < 2 or len(nonchain) < 2:
        return False
    if not any(x["marker_count"] < 2 for x in nonchain):
        return False
    if not any(x["marker_count"] >= 2 for x in nonchain):
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if np.count_nonzero(out == CHAIN_COLOR) == 0:
        return False
    if np.count_nonzero(out == FG) == 0:
        return False
    if np.count_nonzero(grid == FG) - np.count_nonzero(out == FG) <= 0:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        h = random.randint(16, 20)
        w = random.randint(16, 20)
        grid = np.zeros((h, w), dtype=int)

        n_objects = random.randint(5, 7)
        n_chain = random.randint(2, min(3, n_objects - 2))
        n_nonchain = n_objects - n_chain

        specs: list[tuple[str, int]] = []
        specs.append(("chain", random.randint(0, 3)))
        specs.append(("chain", random.randint(0, 3)))
        for _i in range(n_chain - 2):
            specs.append(("chain", random.randint(0, 3)))

        specs.append(("nonchain", random.choice([0, 1])))
        specs.append(("nonchain", random.choice([2, 3])))
        for _i in range(n_nonchain - 2):
            if random.random() < 0.5:
                specs.append(("nonchain", random.choice([0, 1])))
            else:
                specs.append(("nonchain", random.choice([2, 3])))
        random.shuffle(specs)

        patches = []
        failed = False
        for kind, marker_count in specs:
            patch = _build_patch(kind, marker_count)
            if patch is None:
                failed = True
                break
            patches.append(patch["patch"])
        if failed:
            continue

        patches.sort(key=lambda x: (x.shape[0] * x.shape[1], x.shape[0] + x.shape[1]), reverse=True)
        if any(not _place_patch(grid, patch) for patch in patches):
            continue

        _place_isolated_markers(grid)
        if _generation_ok(grid):
            return grid

    raise RuntimeError("generate_input failed")
