from __future__ import annotations

import random

import numpy as np


DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
OPPOSITE = {d: (-d[0], -d[1]) for d in DIRS}
FRAGMENT_PALETTE = [1, 2, 3, 5, 6, 7, 8, 9]
GEN_RETRIES = 500


def _add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return int(a[0] + b[0]), int(a[1] + b[1])


def _sub(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return int(a[0] - b[0]), int(a[1] - b[1])


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS:
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
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _infer_marker_color(grid: np.ndarray, bg: int) -> int:
    candidates: list[int] = []
    for color in sorted(set(map(int, grid.ravel())) - {int(bg)}):
        comps = _components(grid == int(color))
        if comps and all(len(comp) == 1 for comp in comps):
            candidates.append(int(color))
    assert len(candidates) == 1, "expected a unique singleton marker color"
    return int(candidates[0])


def _normalize_coords(coords: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    local = sorted((int(r - min_r), int(c - min_c)) for r, c in coords)
    return local, (int(min_r), int(min_c))


def _endpoint_data(
    grid: np.ndarray,
    coords_abs: list[tuple[int, int]],
    coords_local: list[tuple[int, int]],
    marker: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int] | None]]:
    h, w = grid.shape
    local_set = set(coords_local)
    local_to_abs = {local: coords_abs[i] for i, local in enumerate(coords_local)}
    endpoints_local: list[tuple[int, int]] = []
    endpoints_abs: list[tuple[int, int]] = []
    endpoint_dirs: list[tuple[int, int] | None] = []

    for local in coords_local:
        deg = 0
        for dr, dc in DIRS:
            if (int(local[0] + dr), int(local[1] + dc)) in local_set:
                deg += 1
        assert deg <= 2, "fragment pixels must form a path, not a branch"
        if deg == 1:
            abs_pos = local_to_abs[local]
            marker_dirs = []
            for dr, dc in DIRS:
                rr = int(abs_pos[0] + dr)
                cc = int(abs_pos[1] + dc)
                if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == int(marker):
                    marker_dirs.append((int(dr), int(dc)))
            assert len(marker_dirs) <= 1, "each endpoint may have at most one marker"
            endpoints_local.append(local)
            endpoints_abs.append(abs_pos)
            endpoint_dirs.append(marker_dirs[0] if marker_dirs else None)

    assert len(endpoints_local) == 2, "each fragment must have exactly two endpoints"
    return endpoints_local, endpoints_abs, endpoint_dirs


def _parse_fragments(grid: np.ndarray) -> tuple[int, int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    marker = _infer_marker_color(g, bg)

    fragments: list[dict] = []
    used_marker_positions: set[tuple[int, int]] = set()

    for color in sorted(set(map(int, g.ravel())) - {int(bg), int(marker)}):
        for comp in _components(g == int(color)):
            local_coords, offset = _normalize_coords(comp)
            coords_abs = [_add(local, offset) for local in local_coords]
            endpoints_local, endpoints_abs, endpoint_dirs = _endpoint_data(g, coords_abs, local_coords, marker)
            n_open = sum(direction is None for direction in endpoint_dirs)
            assert n_open in (0, 1), "only start/end fragments may have an unmarked endpoint"
            for abs_pos, direction in zip(endpoints_abs, endpoint_dirs):
                if direction is not None:
                    used_marker_positions.add(_add(abs_pos, direction))
            fragments.append(
                {
                    "color": int(color),
                    "coords": local_coords,
                    "endpoint_coords": endpoints_local,
                    "endpoint_dirs": endpoint_dirs,
                }
            )

    marker_positions = {tuple(map(int, pos)) for pos in np.argwhere(g == int(marker))}
    assert marker_positions == used_marker_positions, "markers must be attached to fragment endpoints"

    start_end_frags = [frag for frag in fragments if sum(d is None for d in frag["endpoint_dirs"]) == 1]
    assert len(start_end_frags) == 2, "expected exactly one start fragment and one end fragment"
    assert fragments, "expected at least one fragment"
    return bg, marker, fragments


def _cross_adjacencies(
    occupied: set[tuple[int, int]],
    new_cells: set[tuple[int, int]],
) -> set[tuple[tuple[int, int], tuple[int, int]]]:
    links: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for r, c in new_cells:
        for dr, dc in DIRS:
            nbr = (int(r + dr), int(c + dc))
            if nbr in occupied:
                links.add((nbr, (int(r), int(c))))
    return links


def _can_attach(
    occupied: set[tuple[int, int]],
    new_cells: set[tuple[int, int]],
    current_exit: tuple[int, int],
    next_entry: tuple[int, int],
) -> bool:
    if occupied & new_cells:
        return False
    return _cross_adjacencies(occupied, new_cells) == {(current_exit, next_entry)}


def _solve_fragments(fragments: list[dict], max_solutions: int = 16) -> list[list[tuple[int, int, tuple[int, int]]]]:
    orientations: list[list[dict]] = []
    for frag_idx, frag in enumerate(fragments):
        frag_orients = []
        for entry_idx in (0, 1):
            exit_idx = 1 - entry_idx
            frag_orients.append(
                {
                    "frag_idx": int(frag_idx),
                    "entry_idx": int(entry_idx),
                    "exit_idx": int(exit_idx),
                    "entry_dir": frag["endpoint_dirs"][entry_idx],
                    "exit_dir": frag["endpoint_dirs"][exit_idx],
                }
            )
        orientations.append(frag_orients)

    solutions: list[list[tuple[int, int, tuple[int, int]]]] = []
    n = len(fragments)

    def dfs(
        current_idx: int,
        current_orient_idx: int,
        current_offset: tuple[int, int],
        used: set[int],
        occupied: set[tuple[int, int]],
        placements: list[tuple[int, int, tuple[int, int]]],
    ) -> None:
        if len(solutions) >= int(max_solutions):
            return

        orient = orientations[current_idx][current_orient_idx]
        exit_dir = orient["exit_dir"]

        if len(used) == n:
            if exit_dir is None:
                solutions.append(list(placements))
            return

        if exit_dir is None:
            return

        current_exit = _add(fragments[current_idx]["endpoint_coords"][orient["exit_idx"]], current_offset)
        need_entry = OPPOSITE[tuple(exit_dir)]

        for next_idx in range(n):
            if next_idx in used:
                continue
            for next_orient_idx, next_orient in enumerate(orientations[next_idx]):
                if next_orient["entry_dir"] != need_entry:
                    continue
                next_entry_local = fragments[next_idx]["endpoint_coords"][next_orient["entry_idx"]]
                next_entry_abs = _add(current_exit, tuple(exit_dir))
                next_offset = _sub(next_entry_abs, next_entry_local)
                next_cells = {_add(cell, next_offset) for cell in fragments[next_idx]["coords"]}
                if not _can_attach(occupied, next_cells, current_exit, next_entry_abs):
                    continue

                used.add(int(next_idx))
                placements.append((int(next_idx), int(next_orient_idx), next_offset))
                dfs(
                    next_idx,
                    next_orient_idx,
                    next_offset,
                    used,
                    occupied | next_cells,
                    placements,
                )
                placements.pop()
                used.remove(int(next_idx))

    for start_idx in range(n):
        for orient_idx, orient in enumerate(orientations[start_idx]):
            if orient["entry_dir"] is not None:
                continue
            start_offset = (0, 0)
            start_cells = {_add(cell, start_offset) for cell in fragments[start_idx]["coords"]}
            dfs(
                start_idx,
                orient_idx,
                start_offset,
                {int(start_idx)},
                start_cells,
                [(int(start_idx), int(orient_idx), start_offset)],
            )

    assert solutions, "expected at least one valid assembly"
    return solutions


def _render_solution(
    fragments: list[dict],
    solution: list[tuple[int, int, tuple[int, int]]],
    bg: int,
) -> np.ndarray:
    painted: list[tuple[int, int, int]] = []
    for frag_idx, _orient_idx, offset in solution:
        frag = fragments[int(frag_idx)]
        for r, c in frag["coords"]:
            rr, cc = _add((int(r), int(c)), offset)
            painted.append((int(rr), int(cc), int(frag["color"])))

    min_r = min(r for r, _c, _color in painted)
    max_r = max(r for r, _c, _color in painted)
    min_c = min(c for _r, c, _color in painted)
    max_c = max(c for _r, c, _color in painted)

    out = np.full((max_r - min_r + 1, max_c - min_c + 1), int(bg), dtype=int)
    for r, c, color in painted:
        out[int(r - min_r), int(c - min_c)] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify colored path fragments and assemble them into one continuous polyline
    by matching yellow endpoint directions, then crop to the path's tight bbox.
    """
    bg, _marker, fragments = _parse_fragments(grid)
    solutions = _solve_fragments(fragments)
    rendered: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
    for solution in solutions:
        out = _render_solution(fragments, solution, bg)
        rendered.setdefault((tuple(map(int, out.shape)), out.tobytes()), out)
        if len(rendered) > 1:
            break
    assert len(rendered) == 1, "assembly output must be unique"
    return next(iter(rendered.values()))


def _normalize_sequence(seq: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in seq)
    min_c = min(c for _, c in seq)
    return [(int(r - min_r), int(c - min_c)) for r, c in seq]


def _random_fragment_sequence(length: int) -> list[tuple[int, int]]:
    if random.random() < 0.5:
        dr, dc = random.choice(DIRS)
        seq = [(int(step * dr), int(step * dc)) for step in range(length)]
        return _normalize_sequence(seq)

    first_len = random.randint(2, length - 1)
    second_len = length - first_len + 1
    d1 = random.choice(DIRS)
    d2 = random.choice([d for d in DIRS if (d[0] == 0) != (d1[0] == 0)])

    seq = [(0, 0)]
    r, c = 0, 0
    for _ in range(first_len - 1):
        r += d1[0]
        c += d1[1]
        seq.append((int(r), int(c)))
    for _ in range(second_len - 1):
        r += d2[0]
        c += d2[1]
        seq.append((int(r), int(c)))
    return _normalize_sequence(seq)


def _allowed_dirs_for_sequence(seq: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    start_step = _sub(seq[1], seq[0])
    end_step = _sub(seq[-2], seq[-1])
    entry_allowed = [d for d in DIRS if d != start_step]
    exit_allowed = [d for d in DIRS if d != end_step]
    return entry_allowed, exit_allowed


def _path_degrees(cells: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    degrees: dict[tuple[int, int], int] = {}
    for cell in cells:
        deg = 0
        for dr, dc in DIRS:
            if (int(cell[0] + dr), int(cell[1] + dc)) in cells:
                deg += 1
        degrees[cell] = int(deg)
    return degrees


def _build_generated_fragments() -> list[dict]:
    n_fragments = random.randint(3, 6)
    colors = random.sample(FRAGMENT_PALETTE, n_fragments)
    fragments: list[dict] = []
    for color in colors:
        length = random.randint(3, 6)
        seq = _random_fragment_sequence(length)
        entry_allowed, exit_allowed = _allowed_dirs_for_sequence(seq)
        fragments.append(
            {
                "color": int(color),
                "coords": list(seq),
                "endpoint_coords": [seq[0], seq[-1]],
                "entry_allowed": entry_allowed,
                "exit_allowed": exit_allowed,
            }
        )
    return fragments


def _assemble_generated_chain(
    fragments: list[dict],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], set[tuple[int, int]]] | None:
    offsets = [(0, 0)]
    connectors: list[tuple[int, int]] = []
    occupied = set(fragments[0]["coords"])

    def dfs(i: int, current_occupied: set[tuple[int, int]]) -> bool:
        if i == len(fragments) - 1:
            return True

        current = fragments[i]
        nxt = fragments[i + 1]
        current_exit = _add(current["endpoint_coords"][1], offsets[i])
        options = [
            direction
            for direction in current["exit_allowed"]
            if OPPOSITE[direction] in nxt["entry_allowed"]
        ]
        random.shuffle(options)

        for direction in options:
            next_entry_abs = _add(current_exit, direction)
            next_offset = _sub(next_entry_abs, nxt["endpoint_coords"][0])
            next_cells = {_add(cell, next_offset) for cell in nxt["coords"]}
            if not _can_attach(current_occupied, next_cells, current_exit, next_entry_abs):
                continue

            offsets.append(next_offset)
            connectors.append(direction)
            if dfs(i + 1, current_occupied | next_cells):
                return True
            connectors.pop()
            offsets.pop()

        return False

    if not dfs(0, occupied):
        return None

    all_cells: set[tuple[int, int]] = set()
    for frag, offset in zip(fragments, offsets):
        for cell in frag["coords"]:
            all_cells.add(_add(cell, offset))

    degrees = _path_degrees(all_cells)
    if any(deg > 2 for deg in degrees.values()):
        return None
    if sum(deg == 1 for deg in degrees.values()) != 2:
        return None

    return offsets, connectors, all_cells


def _render_generated_output(
    fragments: list[dict],
    offsets: list[tuple[int, int]],
    bg: int,
) -> np.ndarray:
    solution = [(idx, 0, offset) for idx, offset in enumerate(offsets)]
    return _render_solution(fragments, solution, bg)


def _fragment_object_cells(
    fragment: dict,
    entry_dir: tuple[int, int] | None,
    exit_dir: tuple[int, int] | None,
    marker: int,
) -> tuple[list[tuple[int, int, int]], int, int]:
    cells = [(int(r), int(c), int(fragment["color"])) for r, c in fragment["coords"]]
    if entry_dir is not None:
        p = fragment["endpoint_coords"][0]
        cells.append((int(p[0] + entry_dir[0]), int(p[1] + entry_dir[1]), int(marker)))
    if exit_dir is not None:
        p = fragment["endpoint_coords"][1]
        cells.append((int(p[0] + exit_dir[0]), int(p[1] + exit_dir[1]), int(marker)))

    min_r = min(r for r, _c, _color in cells)
    min_c = min(c for _r, c, _color in cells)
    norm = [(int(r - min_r), int(c - min_c), int(color)) for r, c, color in cells]
    height = max(r for r, _c, _color in norm) + 1
    width = max(c for _r, c, _color in norm) + 1
    return norm, int(height), int(width)


def _can_place_object(
    occupied: np.ndarray,
    obj_cells: list[tuple[int, int, int]],
    top: int,
    left: int,
    h: int,
    w: int,
    pad: int = 1,
) -> bool:
    H, W = occupied.shape
    if top < 0 or left < 0 or top + h > H or left + w > W:
        return False
    for dr, dc, _color in obj_cells:
        rr = int(top + dr)
        cc = int(left + dc)
        r0 = max(0, rr - pad)
        r1 = min(H, rr + pad + 1)
        c0 = max(0, cc - pad)
        c1 = min(W, cc + pad + 1)
        if bool(occupied[r0:r1, c0:c1].any()):
            return False
    return True


def _place_object(
    grid: np.ndarray,
    occupied: np.ndarray,
    obj_cells: list[tuple[int, int, int]],
    top: int,
    left: int,
) -> None:
    for dr, dc, color in obj_cells:
        rr = int(top + dr)
        cc = int(left + dc)
        grid[rr, cc] = int(color)
        occupied[rr, cc] = True


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    marker = 4

    fragments = _build_generated_fragments()
    assembled = _assemble_generated_chain(fragments)
    if assembled is None:
        return None
    offsets, connectors, all_cells = assembled

    min_r = min(r for r, _ in all_cells)
    max_r = max(r for r, _ in all_cells)
    min_c = min(c for _, c in all_cells)
    max_c = max(c for _, c in all_cells)
    out_h = max_r - min_r + 1
    out_w = max_c - min_c + 1
    if out_h > 18 or out_w > 18:
        return None

    object_specs = []
    for idx, fragment in enumerate(fragments):
        entry_dir = None if idx == 0 else OPPOSITE[connectors[idx - 1]]
        exit_dir = None if idx == len(fragments) - 1 else connectors[idx]
        cells, height, width = _fragment_object_cells(fragment, entry_dir, exit_dir, marker)
        object_specs.append((cells, height, width))

    low_h = max(max(height for _cells, height, _width in object_specs) + 2, out_h + 4)
    low_w = max(max(width for _cells, _height, width in object_specs) + 2, out_w + 4)
    high_h = max(low_h, min(28, low_h + 8))
    high_w = max(low_w, min(28, low_w + 8))
    if low_h > 28 or low_w > 28:
        return None

    H = random.randint(low_h, high_h)
    W = random.randint(low_w, high_w)
    grid = np.full((H, W), bg, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    order = list(range(len(object_specs)))
    random.shuffle(order)
    for idx in order:
        cells, height, width = object_specs[idx]
        positions = [
            (r, c)
            for r in range(H - height + 1)
            for c in range(W - width + 1)
        ]
        random.shuffle(positions)
        placed = False
        for top, left in positions:
            if _can_place_object(occupied, cells, top, left, height, width):
                _place_object(grid, occupied, cells, top, left)
                placed = True
                break
        if not placed:
            return None

    expected = _render_generated_output(fragments, offsets, bg)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
