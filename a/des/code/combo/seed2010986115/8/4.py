from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
DIRS: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
COLOR_POOL: tuple[int, ...] = tuple(range(2, 10))
GEN_MAX_ATTEMPTS = 400


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in DIRS]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
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
                for nr, nc in _neighbors4((int(cr), int(cc))):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _path_adjacency(coords: list[tuple[int, int]]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    coord_set = set(coords)
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for cell in coords:
        adj[cell] = [nbr for nbr in _neighbors4(cell) if nbr in coord_set]
    return adj


def _trace_path(
    adj: dict[tuple[int, int], list[tuple[int, int]]],
    start: tuple[int, int],
) -> list[tuple[int, int]]:
    path = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        nxt = [nbr for nbr in adj[cur] if nbr != prev]
        if not nxt:
            break
        assert len(nxt) == 1, "component must be a simple path"
        prev, cur = cur, nxt[0]
        path.append(cur)
    return path


def _extract_segments(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    segments: list[dict[str, object]] = []
    for comp in _components(g != BLACK):
        assert 3 <= len(comp) <= 6, "every scattered piece must be a short path segment"

        color_counts: dict[int, int] = {}
        for r, c in comp:
            color = int(g[r, c])
            color_counts[color] = color_counts.get(color, 0) + 1
        assert len(color_counts) == 2, "each segment must contain body pixels and one marker pixel"

        colors_by_count = sorted((cnt, color) for color, cnt in color_counts.items())
        marker_count, marker_color = colors_by_count[0]
        body_count, body_color = colors_by_count[1]
        assert marker_count == 1 and body_count == len(comp) - 1

        marker = next((cell for cell in comp if int(g[cell]) == int(marker_color)))
        adj = _path_adjacency(comp)
        degrees = {cell: len(nbrs) for cell, nbrs in adj.items()}
        endpoints = [cell for cell, deg in degrees.items() if deg == 1]
        assert len(endpoints) == 2, "segment must have exactly two endpoints"
        assert marker in endpoints, "the marker pixel must sit on one endpoint"
        assert all(deg in (1, 2) for deg in degrees.values())

        start = endpoints[0] if endpoints[1] == marker else endpoints[1]
        path = _trace_path(adj, start)
        assert len(path) == len(comp) and path[-1] == marker

        sr, sc = start
        local_path = tuple((int(r - sr), int(c - sc)) for r, c in path)
        start_dir = (
            int(local_path[1][0] - local_path[0][0]),
            int(local_path[1][1] - local_path[0][1]),
        )
        end_dir = (
            int(local_path[-1][0] - local_path[-2][0]),
            int(local_path[-1][1] - local_path[-2][1]),
        )

        segments.append(
            {
                "body_color": int(body_color),
                "marker_color": int(marker_color),
                "path_local": local_path,
                "start_dir": start_dir,
                "end_dir": end_dir,
            }
        )

    assert segments, "expected at least one segment"
    return segments


def _build_chain(segments: list[dict[str, object]]) -> list[dict[str, object]]:
    body_map: dict[int, dict[str, object]] = {}
    for seg in segments:
        color = int(seg["body_color"])
        assert color not in body_map, "body colors must be unique"
        body_map[color] = seg

    ref_counts = {color: 0 for color in body_map}
    for seg in segments:
        marker_color = int(seg["marker_color"])
        if marker_color in ref_counts:
            ref_counts[marker_color] += 1

    assert all(cnt in (0, 1) for cnt in ref_counts.values())
    starts = [body_map[color] for color, cnt in ref_counts.items() if cnt == 0]
    assert len(starts) == 1, "the color relay must have a unique head segment"

    chain: list[dict[str, object]] = []
    seen: set[int] = set()
    cur = starts[0]
    while True:
        body_color = int(cur["body_color"])
        assert body_color not in seen, "relay chain must not cycle"
        chain.append(cur)
        seen.add(body_color)

        nxt = body_map.get(int(cur["marker_color"]))
        if nxt is None:
            break
        cur = nxt

    assert len(chain) == len(segments), "relay chain must use all segments"
    return chain


def _translate_path(
    path_local: tuple[tuple[int, int], ...],
    origin: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    orow, ocol = origin
    return tuple((int(orow + r), int(ocol + c)) for r, c in path_local)


def _normalize_coords(coords: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    min_r = min(rows)
    min_c = min(cols)
    return tuple(sorted((int(r - min_r), int(c - min_c)) for r, c in coords))


def _is_simple_path(coords: set[tuple[int, int]]) -> bool:
    if not coords:
        return False

    start = next(iter(coords))
    stack = [start]
    seen = {start}
    while stack:
        cur = stack.pop()
        for nbr in _neighbors4(cur):
            if nbr in coords and nbr not in seen:
                seen.add(nbr)
                stack.append(nbr)
    if len(seen) != len(coords):
        return False

    degs = []
    for cell in coords:
        deg = sum((nbr in coords) for nbr in _neighbors4(cell))
        if deg not in (1, 2):
            return False
        degs.append(deg)
    return degs.count(1) == 2 and degs.count(2) == len(coords) - 2


def _partial_degree_ok(
    occupied: set[tuple[int, int]],
    new_path: tuple[tuple[int, int], ...],
) -> bool:
    merged = set(occupied)
    merged.update(new_path)

    to_check = set(new_path)
    for cell in new_path:
        to_check.update(_neighbors4(cell))

    for cell in to_check:
        if cell not in merged:
            continue
        deg = sum((nbr in merged) for nbr in _neighbors4(cell))
        if deg > 2:
            return False
    return True


def _candidate_key(
    coords: set[tuple[int, int]],
    straight_joins: int,
) -> tuple[object, ...]:
    norm = _normalize_coords(coords)
    height = max(r for r, _ in norm) + 1
    width = max(c for _, c in norm) + 1
    return (-int(straight_joins), int(height * width), int(height), int(width), norm)


def _assemble_chain(chain: list[dict[str, object]]) -> set[tuple[int, int]]:
    placed0 = _translate_path(chain[0]["path_local"], (0, 0))
    occupied0 = set(placed0)

    best_key: tuple[object, ...] | None = None
    best_coords: set[tuple[int, int]] | None = None

    def dfs(
        idx: int,
        occupied: set[tuple[int, int]],
        prev_path: tuple[tuple[int, int], ...],
        straight_joins: int,
    ) -> None:
        nonlocal best_key, best_coords

        if idx == len(chain):
            if not _is_simple_path(occupied):
                return
            key = _candidate_key(occupied, straight_joins)
            if best_key is None or key < best_key:
                best_key = key
                best_coords = set(occupied)
            return

        prev_seg = chain[idx - 1]
        next_seg = chain[idx]
        prev_end = prev_path[-1]
        next_start_dir = next_seg["start_dir"]
        prev_end_dir = prev_seg["end_dir"]

        for dr, dc in DIRS:
            next_origin = (int(prev_end[0] + dr), int(prev_end[1] + dc))
            next_path = _translate_path(next_seg["path_local"], next_origin)
            if any(cell in occupied for cell in next_path):
                continue
            if not _partial_degree_ok(occupied, next_path):
                continue

            next_occupied = set(occupied)
            next_occupied.update(next_path)
            bonus = int((dr, dc) == prev_end_dir and (dr, dc) == next_start_dir)
            dfs(idx + 1, next_occupied, next_path, straight_joins + bonus)

    dfs(1, occupied0, placed0, 0)
    assert best_coords is not None, "failed to assemble the relay chain into a valid path"
    return best_coords


def _render_blue(coords: set[tuple[int, int]]) -> np.ndarray:
    norm = _normalize_coords(coords)
    height = max(r for r, _ in norm) + 1
    width = max(c for _, c in norm) + 1
    out = np.full((height, width), BLACK, dtype=int)
    for r, c in norm:
        out[r, c] = BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble colored path fragments by following the marker-color relay chain.

    Each non-black connected component is a short simple path with exactly one
    endpoint recolored as a marker. The marker color names the next segment's
    body color, so the segments form a unique linked list. Reorder them by that
    color relay, translate them into one connected path, then recolor the whole
    assembled path blue and crop to its tight bounding box.
    """
    segments = _extract_segments(np.asarray(grid, dtype=int))
    chain = _build_chain(segments)
    assembled = _assemble_chain(chain)
    return _render_blue(assembled)


def _perpendiculars(direction: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    dr, dc = direction
    return ((-dc, dr), (dc, -dr))


def _segment_options(direction: tuple[int, int]) -> list[tuple[tuple[tuple[int, int], ...], tuple[int, int]]]:
    dr, dc = direction
    options: list[tuple[tuple[tuple[int, int], ...], tuple[int, int]]] = []

    for length in range(3, 7):
        path = tuple((int(step * dr), int(step * dc)) for step in range(length))
        options.append((path, direction))

    for turn_dir in _perpendiculars(direction):
        tdr, tdc = turn_dir
        for arm1 in range(2, 6):
            for arm2 in range(2, 6):
                if arm1 + arm2 - 1 > 6:
                    continue
                corner = ((arm1 - 1) * dr, (arm1 - 1) * dc)
                first_leg = [(int(step * dr), int(step * dc)) for step in range(arm1)]
                second_leg = [
                    (int(corner[0] + step * tdr), int(corner[1] + step * tdc))
                    for step in range(1, arm2)
                ]
                options.append((tuple(first_leg + second_leg), turn_dir))

    return options


def _can_extend_path(
    occupied: set[tuple[int, int]],
    path: tuple[tuple[int, int], ...],
    entry_neighbor: tuple[int, int] | None,
) -> bool:
    built: set[tuple[int, int]] = set()
    prev = entry_neighbor
    for cell in path:
        if cell in occupied or cell in built:
            return False
        touching = {nbr for nbr in _neighbors4(cell) if nbr in occupied or nbr in built}
        expected = set()
        if prev is not None:
            expected.add(prev)
        if touching != expected:
            return False
        built.add(cell)
        prev = cell
    return True


def _generate_full_segments(
    rng: random.Random,
    segment_count: int,
) -> list[tuple[tuple[int, int], ...]] | None:
    start_dir = rng.choice(DIRS)
    current_dir = start_dir
    current_start = (0, 0)
    occupied: set[tuple[int, int]] = set()
    segments: list[tuple[tuple[int, int], ...]] = []

    for idx in range(segment_count):
        options = _segment_options(current_dir)
        rng.shuffle(options)

        entry_neighbor = None
        if segments:
            prev_end = segments[-1][-1]
            entry_neighbor = prev_end

        placed = False
        for local_path, end_dir in options:
            abs_path = tuple(
                (int(current_start[0] + r), int(current_start[1] + c))
                for r, c in local_path
            )
            if not _can_extend_path(occupied, abs_path, entry_neighbor):
                continue

            segments.append(abs_path)
            occupied.update(abs_path)
            current_dir = end_dir
            if idx < segment_count - 1:
                current_start = (
                    int(abs_path[-1][0] + end_dir[0]),
                    int(abs_path[-1][1] + end_dir[1]),
                )
            placed = True
            break

        if not placed:
            return None

    return segments


def _colored_patch(
    path: tuple[tuple[int, int], ...],
    body_color: int,
    marker_color: int,
) -> tuple[int, int, list[tuple[int, int, int]]]:
    rows = [r for r, _ in path]
    cols = [c for _, c in path]
    min_r = min(rows)
    min_c = min(cols)
    max_r = max(rows)
    max_c = max(cols)

    cells: list[tuple[int, int, int]] = []
    for idx, (r, c) in enumerate(path):
        color = marker_color if idx == len(path) - 1 else body_color
        cells.append((int(r - min_r), int(c - min_c), int(color)))
    return int(max_r - min_r + 1), int(max_c - min_c + 1), cells


def _can_place_patch(
    grid: np.ndarray,
    patch_cells: list[tuple[int, int, int]],
    top: int,
    left: int,
) -> bool:
    h, w = map(int, grid.shape)
    placed = {(int(top + r), int(left + c)) for r, c, _ in patch_cells}

    for rr, cc in placed:
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        if int(grid[rr, cc]) != BLACK:
            return False

    for rr, cc in placed:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = rr + dr
                nc = cc + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in placed:
                    continue
                if int(grid[nr, nc]) != BLACK:
                    return False
    return True


def _scatter_patches(
    rng: random.Random,
    patches: list[tuple[int, int, list[tuple[int, int, int]]]],
) -> np.ndarray | None:
    max_h = max(ph for ph, _pw, _cells in patches)
    max_w = max(pw for _ph, pw, _cells in patches)
    lower = max(14, max_h + 3, max_w + 3, 2 * len(patches) + 8)
    upper = max(lower + 2, 30)
    grid_h = rng.randint(lower, upper)
    grid_w = rng.randint(lower, upper)
    grid = np.full((grid_h, grid_w), BLACK, dtype=int)

    order = list(range(len(patches)))
    rng.shuffle(order)

    for patch_idx in order:
        ph, pw, patch_cells = patches[patch_idx]
        candidates = [
            (r, c)
            for r in range(grid_h - ph + 1)
            for c in range(grid_w - pw + 1)
        ]
        rng.shuffle(candidates)

        placed = False
        for top, left in candidates:
            if not _can_place_patch(grid, patch_cells, top, left):
                continue
            for r, c, color in patch_cells:
                grid[top + r, left + c] = int(color)
            placed = True
            break
        if not placed:
            return None

    return grid


def _expected_output(segments: list[tuple[tuple[int, int], ...]]) -> np.ndarray:
    coords = set()
    for segment in segments:
        coords.update(segment)
    return _render_blue(coords)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    segment_count = rng.randint(2, 6)
    if segment_count + 1 > len(COLOR_POOL):
        return None

    colors = rng.sample(COLOR_POOL, segment_count + 1)
    body_colors = colors[:segment_count]
    terminal_color = colors[-1]

    segments = _generate_full_segments(rng, segment_count)
    if segments is None:
        return None

    marker_colors = body_colors[1:] + [terminal_color]
    patches = [
        _colored_patch(path, body_color, marker_color)
        for path, body_color, marker_color in zip(segments, body_colors, marker_colors)
    ]
    inp = _scatter_patches(rng, patches)
    if inp is None:
        return None

    expected = _expected_output(segments)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid relay-path input")
