from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
GREY = 5
MAROON = 9
PATH_COLORS = [1, 3, 4, 6, 7, 8]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _bbox(points: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    dirs = DIRS4 if connectivity == 4 else DIRS8
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[tuple[int, int]] = []
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in dirs:
                    nr = rr + dr
                    nc = cc + dc
                    if _in_bounds(h, w, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _neighbor_set(points: set[tuple[int, int]], cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in DIRS8:
        nxt = (r + dr, c + dc)
        if nxt in points:
            out.append(nxt)
    return out


def _find_divider(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    assert len(rows) == 1
    return rows[0]


def _find_template_bbox(grid: np.ndarray, divider: int) -> tuple[int, int, int, int]:
    upper = grid[:divider, :]
    points = list(map(tuple, np.argwhere(upper == MAROON)))
    assert points
    r0, c0, r1, c1 = _bbox(points)
    sub = upper[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(sub[0, :] == MAROON)
    assert np.all(sub[-1, :] == MAROON)
    assert np.all(sub[:, 0] == MAROON)
    assert np.all(sub[:, -1] == MAROON)
    return r0, c0, r1, c1


def _find_endpoints(points: set[tuple[int, int]]) -> list[tuple[int, int]]:
    endpoints = [p for p in points if len(_neighbor_set(points, p)) == 1]
    assert len(endpoints) == 2
    return endpoints


def _trace_path(points: set[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]]:
    order = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        nexts = [p for p in _neighbor_set(points, cur) if p != prev]
        if not nexts:
            break
        assert len(nexts) == 1
        prev, cur = cur, nexts[0]
        order.append(cur)
    assert len(order) == len(points)
    return order


def _endpoint_marker(
    lower: np.ndarray,
    color: int,
    endpoint: tuple[int, int],
) -> int:
    h, w = lower.shape
    r, c = endpoint
    markers = []
    for dr, dc in DIRS8:
        nr = r + dr
        nc = c + dc
        if not _in_bounds(h, w, nr, nc):
            continue
        val = int(lower[nr, nc])
        if val == BLACK or val == color:
            continue
        markers.append((nr, nc))
    assert len(markers) == 1
    mr, mc = markers[0]
    return int(lower[mr, mc])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a fragment chain from endpoint markers and map its pixel colors onto template anchors.
    """
    g = np.asarray(grid, dtype=int)
    divider = _find_divider(g)
    tr0, tc0, tr1, tc1 = _find_template_bbox(g, divider)

    anchors = [
        (r, c)
        for r in range(tr0 + 1, tr1)
        for c in range(tc0 + 1, tc1)
        if int(g[r, c]) == GREY
    ]
    anchors.sort()

    lower = g[divider + 1 :, :]
    fragments: dict[int, dict[str, object]] = {}
    for color in sorted(set(map(int, np.unique(lower)))):
        if color in (BLACK, RED, GREY, MAROON):
            continue
        comps = _components(lower == color, connectivity=8)
        bodies = [comp for comp in comps if len(comp) >= 2]
        if not bodies:
            continue
        assert len(bodies) == 1
        points = set(bodies[0])
        endpoints = _find_endpoints(points)
        markers = {ep: _endpoint_marker(lower, color, ep) for ep in endpoints}
        fragments[color] = {
            "points": points,
            "markers": markers,
        }

    assert fragments
    start_colors = [
        color
        for color, data in fragments.items()
        if RED in set(data["markers"].values())  # type: ignore[index]
    ]
    assert len(start_colors) == 1

    ordered_colors: list[int] = []
    current = start_colors[0]
    prev_marker = RED
    visited: set[int] = set()
    for index in range(len(fragments)):
        data = fragments[current]
        markers: dict[tuple[int, int], int] = data["markers"]  # type: ignore[assignment]
        starts = [ep for ep, marker in markers.items() if marker == prev_marker]
        assert len(starts) == 1
        path = _trace_path(data["points"], starts[0])  # type: ignore[arg-type]
        ordered_colors.extend([current] * len(path))
        visited.add(current)
        if index == len(fragments) - 1:
            break
        next_color = markers[path[-1]]
        assert next_color in fragments and next_color not in visited
        prev_marker, current = current, next_color

    assert len(ordered_colors) == len(anchors)
    out = g[tr0 : tr1 + 1, tc0 : tc1 + 1].copy()
    out[1:-1, 1:-1] = BLACK
    for (r, c), color in zip(anchors, ordered_colors, strict=True):
        out[r - tr0, c - tc0] = color
    return out


def _path_candidates(
    cur: tuple[int, int],
    used: set[tuple[int, int]],
    inner_h: int,
    inner_w: int,
) -> list[tuple[int, int]]:
    out = []
    r, c = cur
    for dr, dc in DIRS8:
        nr = r + dr
        nc = c + dc
        nxt = (nr, nc)
        if not _in_bounds(inner_h, inner_w, nr, nc) or nxt in used:
            continue
        ok = True
        for other in used:
            if other == cur:
                continue
            if max(abs(other[0] - nr), abs(other[1] - nc)) <= 1:
                ok = False
                break
        if ok:
            out.append(nxt)
    return out


def _build_walk(
    path: list[tuple[int, int]],
    used: set[tuple[int, int]],
    target_len: int,
    inner_h: int,
    inner_w: int,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    if len(path) == target_len:
        return path[:]

    cur = path[-1]
    candidates = _path_candidates(cur, used, inner_h, inner_w)
    rng.shuffle(candidates)
    candidates.sort(
        key=lambda cell: len(_path_candidates(cell, used | {cell}, inner_h, inner_w)),
        reverse=True,
    )
    for nxt in candidates:
        path.append(nxt)
        used.add(nxt)
        solved = _build_walk(path, used, target_len, inner_h, inner_w, rng)
        if solved is not None:
            return solved
        used.remove(nxt)
        path.pop()
    return None


def _marker_options(
    endpoint: tuple[int, int],
    path_set: set[tuple[int, int]],
    inner_h: int,
    inner_w: int,
) -> list[tuple[int, int]]:
    out = []
    er, ec = endpoint
    for dr, dc in DIRS8:
        mr = er + dr
        mc = ec + dc
        marker = (mr, mc)
        if not _in_bounds(inner_h, inner_w, mr, mc) or marker in path_set:
            continue
        touching = [p for p in path_set if max(abs(p[0] - mr), abs(p[1] - mc)) <= 1]
        if touching == [endpoint] or set(touching) == {endpoint}:
            out.append(marker)
    return out


def _generate_fragment_shape(length: int, rng: random.Random) -> dict[str, object] | None:
    for _ in range(120):
        inner_h = rng.randint(max(3, min(length, 4)), max(5, min(7, length + 2)))
        inner_w = rng.randint(max(3, min(length, 4)), max(5, min(7, length + 2)))
        if inner_h * inner_w < length + 2:
            continue
        starts = [(r, c) for r in range(inner_h) for c in range(inner_w)]
        rng.shuffle(starts)
        for start in starts[:20]:
            walk = _build_walk([start], {start}, length, inner_h, inner_w, rng)
            if walk is None:
                continue
            path_set = set(walk)
            start_opts = _marker_options(walk[0], path_set, inner_h, inner_w)
            end_opts = _marker_options(walk[-1], path_set, inner_h, inner_w)
            rng.shuffle(start_opts)
            rng.shuffle(end_opts)
            for start_marker in start_opts:
                for end_marker in end_opts:
                    if start_marker == end_marker:
                        continue
                    if max(abs(start_marker[0] - walk[-1][0]), abs(start_marker[1] - walk[-1][1])) <= 1:
                        continue
                    if max(abs(end_marker[0] - walk[0][0]), abs(end_marker[1] - walk[0][1])) <= 1:
                        continue
                    cells = walk + [start_marker, end_marker]
                    r0, c0, r1, c1 = _bbox(cells)
                    shift_r = -r0
                    shift_c = -c0
                    return {
                        "path": [(r + shift_r, c + shift_c) for r, c in walk],
                        "start_marker": (start_marker[0] + shift_r, start_marker[1] + shift_c),
                        "end_marker": (end_marker[0] + shift_r, end_marker[1] + shift_c),
                        "shape": (r1 - r0 + 1, c1 - c0 + 1),
                    }
    return None


def _make_fragment_patch(
    spec: dict[str, object],
    body_color: int,
    start_marker_color: int,
    end_marker_color: int,
) -> np.ndarray:
    h, w = spec["shape"]  # type: ignore[assignment]
    patch = np.full((h, w), BLACK, dtype=int)
    for r, c in spec["path"]:  # type: ignore[index]
        patch[r, c] = body_color
    sr, sc = spec["start_marker"]  # type: ignore[assignment]
    er, ec = spec["end_marker"]  # type: ignore[assignment]
    patch[sr, sc] = start_marker_color
    patch[er, ec] = end_marker_color
    return patch


def _place_patches(
    patches: list[np.ndarray],
    height: int,
    width: int,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    order = sorted(range(len(patches)), key=lambda idx: int(np.prod(patches[idx].shape)), reverse=True)
    placements: list[tuple[int, int] | None] = [None] * len(patches)
    occupied = np.zeros((height, width), dtype=bool)
    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        if ph > height or pw > width:
            return None
        placed = False
        for _ in range(250):
            r0 = rng.randint(0, height - ph)
            c0 = rng.randint(0, width - pw)
            rr0 = max(0, r0 - 1)
            rr1 = min(height, r0 + ph + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(width, c0 + pw + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            occupied[r0 : r0 + ph, c0 : c0 + pw] = True
            placements[idx] = (r0, c0)
            placed = True
            break
        if not placed:
            return None
    return [pos for pos in placements if pos is not None]


def _expected_output(
    frame_outer_shape: tuple[int, int],
    anchor_positions: list[tuple[int, int]],
    ordered_colors: list[int],
) -> np.ndarray:
    h, w = frame_outer_shape
    out = np.full((h, w), BLACK, dtype=int)
    out[0, :] = MAROON
    out[-1, :] = MAROON
    out[:, 0] = MAROON
    out[:, -1] = MAROON
    for (r, c), color in zip(anchor_positions, ordered_colors, strict=True):
        out[r, c] = color
    return out


def _generate_case(rng: random.Random) -> np.ndarray | None:
    fragment_count = rng.randint(2, 5)
    fragment_colors = rng.sample(PATH_COLORS, fragment_count)
    lengths = [rng.randint(2, 6) for _ in range(fragment_count)]
    total_pixels = sum(lengths)

    shape_specs = []
    for length in lengths:
        spec = _generate_fragment_shape(length, rng)
        if spec is None:
            return None
        shape_specs.append(spec)

    outer_candidates = []
    for inner_h in range(4, 10):
        for inner_w in range(4, 13):
            area = inner_h * inner_w
            if total_pixels <= area <= total_pixels + 14:
                outer_candidates.append((inner_h + 2, inner_w + 2))
    if not outer_candidates:
        return None
    frame_h, frame_w = rng.choice(outer_candidates)
    inner_h = frame_h - 2
    inner_w = frame_w - 2

    anchor_pool = [(r, c) for r in range(1, frame_h - 1) for c in range(1, frame_w - 1)]
    anchor_positions = sorted(rng.sample(anchor_pool, total_pixels))

    outgoing = fragment_colors[1:] + [rng.choice(fragment_colors)]
    patches = [
        _make_fragment_patch(
            spec,
            fragment_colors[idx],
            RED if idx == 0 else fragment_colors[idx - 1],
            outgoing[idx],
        )
        for idx, spec in enumerate(shape_specs)
    ]

    patch_area = sum(int(np.prod(patch.shape)) for patch in patches)
    min_width = max(frame_w + 2, max(patch.shape[1] for patch in patches) + 2, 18)
    max_width = min(30, max(min_width, int(np.ceil(np.sqrt(patch_area) * 2.2))))
    if min_width > max_width:
        max_width = 30
    if min_width > 30:
        return None
    width = rng.randint(min_width, max_width)

    lower_min_h = max(
        max(patch.shape[0] for patch in patches) + 2,
        int(np.ceil(patch_area / max(1, width - 2))) + 4,
        8,
    )
    lower_max_h = max(lower_min_h, min(18, lower_min_h + 5))
    if lower_min_h > lower_max_h:
        return None
    lower_h = rng.randint(lower_min_h, lower_max_h)

    placements = _place_patches(patches, lower_h, width, rng)
    if placements is None:
        return None

    upper_h = frame_h + rng.randint(1, 4)
    grid_h = upper_h + 1 + lower_h
    grid = np.full((grid_h, width), BLACK, dtype=int)

    frame_r0 = rng.randint(0, upper_h - frame_h)
    frame_c0 = rng.randint(0, width - frame_w)
    frame_r1 = frame_r0 + frame_h - 1
    frame_c1 = frame_c0 + frame_w - 1
    grid[frame_r0, frame_c0 : frame_c1 + 1] = MAROON
    grid[frame_r1, frame_c0 : frame_c1 + 1] = MAROON
    grid[frame_r0 : frame_r1 + 1, frame_c0] = MAROON
    grid[frame_r0 : frame_r1 + 1, frame_c1] = MAROON
    for r, c in anchor_positions:
        grid[frame_r0 + r, frame_c0 + c] = GREY

    divider = upper_h
    grid[divider, :] = GREY

    lower_r0 = divider + 1
    for patch, (pr, pc) in zip(patches, placements, strict=True):
        ph, pw = patch.shape
        grid[lower_r0 + pr : lower_r0 + pr + ph, pc : pc + pw] = patch

    ordered_colors = []
    for color, length in zip(fragment_colors, lengths, strict=True):
        ordered_colors.extend([color] * length)
    expected = _expected_output((frame_h, frame_w), anchor_positions, ordered_colors)

    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _generate_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
