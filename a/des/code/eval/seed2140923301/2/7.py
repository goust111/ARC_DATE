import random
from collections import deque

import numpy as np


BG = 0
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 300
PATCH_RETRIES = 200
PATH_RETRIES = 80


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _monochrome_components(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(h, w, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": color, "coords": coords})
    return comps


def _simple_path_endpoints(coords: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    if len(coords) < 2:
        return None
    coord_set = set(coords)
    endpoints = []
    for r, c in coords:
        degree = sum((nr, nc) in coord_set for nr, nc in _neighbors4(r, c))
        if degree == 1:
            endpoints.append((r, c))
        elif degree != 2:
            return None
    if len(endpoints) != 2:
        return None
    return sorted(endpoints)


def _extract_chains(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    chains: list[dict[str, object]] = []
    for comp in _monochrome_components(grid):
        coords = list(comp["coords"])
        endpoints = _simple_path_endpoints(coords)
        if endpoints is None:
            continue
        coord_set = set(coords)
        anchors = []
        for r, c in endpoints:
            candidates = []
            for nr, nc in _neighbors4(r, c):
                if _in_bounds(h, w, nr, nc) and (nr, nc) not in coord_set and int(grid[nr, nc]) != BG:
                    candidates.append((nr, nc))
            assert len(candidates) == 1, "each chain endpoint must touch exactly one anchor"
            ar, ac = candidates[0]
            assert int(grid[ar, ac]) != int(comp["color"]), "anchor color must differ from chain color"
            anchors.append((ar, ac))
        assert anchors[0] != anchors[1], "two endpoints cannot share an anchor"
        chains.append(
            {
                "color": int(comp["color"]),
                "coords": coords,
                "coord_set": coord_set,
                "endpoints": endpoints,
                "anchors": anchors,
                "anchor_colors": (int(grid[anchors[0]]), int(grid[anchors[1]])),
            }
        )
    chains.sort(key=lambda item: (item["endpoints"][0], item["endpoints"][1], item["color"]))
    return chains


def _horizontal_then_vertical_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    (r1, c1), (r2, c2) = sorted([a, b])
    path = []
    step_c = 1 if c2 >= c1 else -1
    for c in range(c1, c2 + step_c, step_c):
        path.append((r1, c))
    step_r = 1 if r2 >= r1 else -1
    for r in range(r1 + step_r, r2 + step_r, step_r):
        path.append((r, c2))
    return path


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    chains = _extract_chains(g)

    for chain in chains:
        for r, c in chain["coords"]:
            out[r, c] = BG

    for chain in chains:
        color = int(chain["color"])
        (e1, e2) = chain["endpoints"]
        a1_color, a2_color = chain["anchor_colors"]

        if a1_color == a2_color:
            target = _horizontal_then_vertical_path(e1, e2)
        else:
            r0, r1 = sorted([e1[0], e2[0]])
            c0, c1 = sorted([e1[1], e2[1]])
            target = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]

        for r, c in target:
            out[r, c] = color

    return out


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _reachable(ih: int, iw: int, start: tuple[int, int], end: tuple[int, int], blocked: set[tuple[int, int]]) -> bool:
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        if (r, c) == end:
            return True
        for nr, nc in _neighbors4(r, c):
            if not (1 <= nr <= ih and 1 <= nc <= iw):
                continue
            if (nr, nc) in seen:
                continue
            if (nr, nc) in blocked and (nr, nc) != end:
                continue
            seen.add((nr, nc))
            q.append((nr, nc))
    return False


def _random_simple_path(
    ih: int,
    iw: int,
    start: tuple[int, int],
    end: tuple[int, int],
    min_cells: int,
    max_cells: int,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    for _ in range(PATH_RETRIES):
        path = [start]
        visited = {start}

        def dfs(cell: tuple[int, int]) -> bool:
            if len(path) > max_cells:
                return False
            if cell == end:
                return len(path) >= min_cells

            options = []
            for nr, nc in _neighbors4(*cell):
                nxt = (nr, nc)
                if not (1 <= nr <= ih and 1 <= nc <= iw):
                    continue
                if nxt in visited:
                    continue
                min_possible = len(path) + _manhattan(nxt, end)
                if min_possible + 1 > max_cells:
                    continue
                blocked = visited | {nxt}
                if not _reachable(ih, iw, nxt, end, blocked):
                    continue
                options.append(nxt)

            rng.shuffle(options)
            if len(path) < min_cells - 1:
                options.sort(key=lambda p: (_manhattan(p, end), rng.random()), reverse=True)
            else:
                options.sort(key=lambda p: (_manhattan(p, end), rng.random()))

            for nxt in options:
                visited.add(nxt)
                path.append(nxt)
                if dfs(nxt):
                    return True
                path.pop()
                visited.remove(nxt)
            return False

        if dfs(start):
            return path
    return None


def _pick_edge_point(edge: str, ih: int, iw: int, rng: random.Random) -> tuple[tuple[int, int], tuple[int, int]]:
    if edge == "top":
        c = rng.randint(2, iw - 1)
        return (1, c), (0, c)
    if edge == "bottom":
        c = rng.randint(2, iw - 1)
        return (ih, c), (ih + 1, c)
    if edge == "left":
        r = rng.randint(2, ih - 1)
        return (r, 1), (r, 0)
    if edge == "right":
        r = rng.randint(2, ih - 1)
        return (r, iw), (r, iw + 1)
    raise ValueError(edge)


def _build_patch(kind: str, chain_color: int, rng: random.Random) -> np.ndarray | None:
    for _ in range(PATCH_RETRIES):
        ih = rng.randint(4, 8)
        iw = rng.randint(4, 8)
        edge_a, edge_b = rng.sample(["top", "bottom", "left", "right"], 2)
        start, anchor_a = _pick_edge_point(edge_a, ih, iw, rng)
        end, anchor_b = _pick_edge_point(edge_b, ih, iw, rng)

        if start == end or anchor_a == anchor_b:
            continue
        if kind == "different" and (start[0] == end[0] or start[1] == end[1]):
            continue

        shortest = _manhattan(start, end) + 1
        free_cells = ih * iw
        feasible_extras = [extra for extra in (0, 2, 4, 6) if shortest + extra <= free_cells]
        if kind == "same":
            feasible_extras = [extra for extra in feasible_extras if extra >= 2]
        if not feasible_extras:
            continue
        extra = rng.choice(feasible_extras)
        min_cells = shortest + extra
        max_cells = min(free_cells, min_cells + (2 if kind == "same" else 4))
        path = _random_simple_path(ih, iw, start, end, min_cells, max_cells, rng)
        if path is None:
            continue

        canonical = set(_horizontal_then_vertical_path(start, end))
        if kind == "same" and set(path) == canonical:
            continue

        choices = [color for color in range(1, 10) if color != chain_color]
        if kind == "same":
            anchor_color_a = anchor_color_b = rng.choice(choices)
        else:
            anchor_color_a, anchor_color_b = rng.sample(choices, 2)

        patch = np.zeros((ih + 2, iw + 2), dtype=int)
        for r, c in path:
            patch[r, c] = chain_color
        patch[anchor_a] = anchor_color_a
        patch[anchor_b] = anchor_color_b

        chains = _extract_chains(patch)
        if len(chains) != 1:
            continue
        return patch
    return None


def _branch_plan(count: int, rng: random.Random) -> list[str]:
    if count == 2:
        kinds = ["same", "different"]
    elif count == 3:
        kinds = ["same", "different", rng.choice(["same", "different"])]
    else:
        kinds = ["same", "same", "different", "different"]
    rng.shuffle(kinds)
    return kinds


def _pack_patches(patches: list[np.ndarray], rng: random.Random) -> np.ndarray | None:
    ordered = patches[:]
    rng.shuffle(ordered)
    k = len(ordered)

    for _ in range(100):
        if k <= 2 and rng.random() < 0.5:
            groups = [ordered]
        else:
            if k == 3:
                split = rng.choice([1, 2])
            else:
                split = k // 2
            groups = [ordered[:split], ordered[split:]]
        groups = [group for group in groups if group]

        gap_x = rng.randint(2, 4)
        gap_y = rng.randint(2, 4)
        row_widths = [sum(p.shape[1] for p in group) + gap_x * (len(group) - 1) for group in groups]
        row_heights = [max(p.shape[0] for p in group) for group in groups]

        min_w = max(row_widths) + 2
        min_h = sum(row_heights) + gap_y * (len(groups) - 1) + 2
        if min_w > 30 or min_h > 30:
            continue

        max_w = min(30, min_w + 6)
        max_h = min(30, min_h + 6)
        if max_w < 14 or max_h < 14:
            continue

        w = rng.randint(max(14, min_w), max_w)
        h = rng.randint(max(14, min_h), max_h)
        grid = np.zeros((h, w), dtype=int)

        top_shift = rng.randint(0, h - min_h)
        y = 1 + top_shift
        for row_h, row_w, group in zip(row_heights, row_widths, groups):
            x = 1 + rng.randint(0, w - row_w - 2)
            for patch in group:
                dy = rng.randint(0, row_h - patch.shape[0])
                ph, pw = patch.shape
                block = grid[y + dy : y + dy + ph, x : x + pw]
                if np.any((block != BG) & (patch != BG)):
                    return None
                block[patch != BG] = patch[patch != BG]
                x += pw + gap_x
            y += row_h + gap_y

        return grid
    return None


def _validate_generated_grid(grid: np.ndarray, expected_chains: int) -> bool:
    chains = _extract_chains(grid)
    if len(chains) != expected_chains:
        return False
    if len(_monochrome_components(grid)) != expected_chains * 3:
        return False

    same_flags = [chain["anchor_colors"][0] == chain["anchor_colors"][1] for chain in chains]
    if not any(same_flags) or not any(not flag for flag in same_flags):
        return False

    anchors = {anchor for chain in chains for anchor in chain["anchors"]}
    if len(anchors) != expected_chains * 2:
        return False

    out = main(grid)
    for r, c in anchors:
        if int(out[r, c]) != int(grid[r, c]):
            return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        chain_count = rng.randint(2, 4)
        chain_colors = rng.sample(range(1, 10), chain_count)
        patches = []
        failed = False
        for kind, chain_color in zip(_branch_plan(chain_count, rng), chain_colors):
            patch = _build_patch(kind, chain_color, rng)
            if patch is None:
                failed = True
                break
            patches.append(patch)
        if failed:
            continue

        grid = _pack_patches(patches, rng)
        if grid is None:
            continue
        if _validate_generated_grid(grid, chain_count):
            return grid
    raise RuntimeError("generate_input failed after repeated retries")
