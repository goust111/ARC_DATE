from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BG = 0
AZURE = 8
MAROON = 9
PALETTE = (1, 2, 3, 4, 5, 6, 7)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def _add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return a[0] + b[0], a[1] + b[1]


def _mul(d: tuple[int, int], k: int) -> tuple[int, int]:
    return d[0] * k, d[1] * k


def _neg(d: tuple[int, int]) -> tuple[int, int]:
    return -d[0], -d[1]


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            queue = deque([(r, c)])
            while queue:
                cell = queue.popleft()
                comp.append(cell)
                for nb in _neighbors4(cell):
                    if _in_bounds((h, w), nb) and mask[nb] and not seen[nb]:
                        seen[nb] = True
                        queue.append(nb)
            comps.append(comp)
    return comps


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        step = 1 if b[1] >= a[1] else -1
        return [(a[0], c) for c in range(a[1], b[1] + step, step)]
    if a[1] == b[1]:
        step = 1 if b[0] >= a[0] else -1
        return [(r, a[1]) for r in range(a[0], b[0] + step, step)]
    raise AssertionError("line cells require row or column alignment")


def _find_ports(grid: np.ndarray) -> dict[tuple[int, int], dict[str, object]]:
    azure_mask = grid == AZURE
    ports: dict[tuple[int, int], dict[str, object]] = {}
    for comp_id, comp in enumerate(_components(azure_mask)):
        comp_set = set(comp)
        degrees = {
            cell: sum(nb in comp_set for nb in _neighbors4(cell))
            for cell in comp
        }
        assert all(deg <= 2 for deg in degrees.values())
        endpoints = [cell for cell, deg in degrees.items() if deg == 1]
        assert len(endpoints) == 2
        for endpoint in endpoints:
            candidates = []
            for nb in _neighbors4(endpoint):
                if not _in_bounds(grid.shape, nb):
                    continue
                color = int(grid[nb])
                if color in (BG, AZURE):
                    continue
                candidates.append(nb)
            assert len(candidates) == 1
            port = candidates[0]
            assert port not in ports
            ports[port] = {
                "color": int(grid[port]),
                "endpoint": endpoint,
                "component_id": comp_id,
            }
    return ports


def _build_pipe(
    grid: np.ndarray,
    ports: dict[tuple[int, int], dict[str, object]],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int], int]:
    pipe = set(map(tuple, np.argwhere(grid == AZURE)))
    port_cells = set(ports)
    pipe |= port_cells

    by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for port, info in ports.items():
        by_color[int(info["color"])].append(port)

    unmatched: list[tuple[int, int]] = []
    for color, group in by_color.items():
        if len(group) == 1:
            unmatched.append(group[0])
            continue
        assert len(group) == 2, f"ambiguous port color {color}"
        a, b = group
        cells = _line_cells(a, b)
        for cell in cells[1:-1]:
            assert int(grid[cell]) == BG
        pipe.update(cells)

    assert len(unmatched) == 2
    start_port = None
    end_port = None
    start_marker = None
    end_marker = None
    seed_color = None
    for port in unmatched:
        markers = []
        for nb in _neighbors4(port):
            if not _in_bounds(grid.shape, nb):
                continue
            if nb in pipe:
                continue
            color = int(grid[nb])
            if color == BG:
                continue
            markers.append(nb)
        assert len(markers) == 1
        marker = markers[0]
        if int(grid[marker]) == MAROON:
            assert end_port is None
            end_port = port
            end_marker = marker
        else:
            assert start_port is None
            start_port = port
            start_marker = marker
            seed_color = int(grid[marker])

    assert start_port is not None and end_port is not None
    assert start_marker is not None and end_marker is not None
    assert seed_color is not None
    return pipe, port_cells, start_port, end_port, start_marker, end_marker, seed_color


def _ordered_path(
    pipe: set[tuple[int, int]],
    start: tuple[int, int],
    end: tuple[int, int],
) -> list[tuple[int, int]]:
    adj = {
        cell: [nb for nb in _neighbors4(cell) if nb in pipe]
        for cell in pipe
    }
    assert len(adj[start]) == 1 and len(adj[end]) == 1
    for cell in pipe:
        if cell in (start, end):
            continue
        assert len(adj[cell]) == 2

    path = [start]
    prev = None
    cur = start
    while cur != end:
        nxts = [nb for nb in adj[cur] if nb != prev]
        assert len(nxts) == 1
        nxt = nxts[0]
        path.append(nxt)
        prev, cur = cur, nxt
    assert len(path) == len(pipe)
    return path


def _find_filters(
    grid: np.ndarray,
    pipe: set[tuple[int, int]],
    port_cells: set[tuple[int, int]],
    start_marker: tuple[int, int],
    end_marker: tuple[int, int],
    path_index: dict[tuple[int, int], int],
) -> dict[int, tuple[int, int]]:
    mask = np.zeros_like(grid, dtype=bool)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            cell = (r, c)
            color = int(grid[cell])
            if color in (BG, AZURE):
                continue
            if cell in port_cells or cell == start_marker or cell == end_marker:
                continue
            mask[cell] = True

    transitions: dict[int, tuple[int, int]] = {}
    for comp in _components(mask):
        assert len(comp) == 2
        a, b = sorted(comp)
        horizontal = a[0] == b[0] and abs(a[1] - b[1]) == 1
        vertical = a[1] == b[1] and abs(a[0] - b[0]) == 1
        assert horizontal or vertical

        choices = []
        if horizontal:
            for dr in (-1, 1):
                p1 = (a[0] + dr, a[1])
                p2 = (b[0] + dr, b[1])
                if p1 in pipe and p2 in pipe and abs(path_index[p1] - path_index[p2]) == 1:
                    choices.append(((a, p1), (b, p2)))
        else:
            for dc in (-1, 1):
                p1 = (a[0], a[1] + dc)
                p2 = (b[0], b[1] + dc)
                if p1 in pipe and p2 in pipe and abs(path_index[p1] - path_index[p2]) == 1:
                    choices.append(((a, p1), (b, p2)))

        assert len(choices) == 1
        mapping = {}
        for filter_cell, pipe_cell in choices[0]:
            mapping[path_index[pipe_cell]] = int(grid[filter_cell])
        idxs = sorted(mapping)
        assert idxs[1] == idxs[0] + 1
        transitions[idxs[1]] = (mapping[idxs[0]], mapping[idxs[1]])
    return transitions


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    ports = _find_ports(g)
    pipe, port_cells, start_port, end_port, start_marker, end_marker, seed_color = _build_pipe(g, ports)
    path = _ordered_path(pipe, start_port, end_port)
    path_index = {cell: idx for idx, cell in enumerate(path)}
    transitions = _find_filters(g, pipe, port_cells, start_marker, end_marker, path_index)

    out = g.copy()
    color = seed_color
    for idx, cell in enumerate(path):
        if idx > 0 and idx in transitions and color == transitions[idx][0]:
            color = transitions[idx][1]
        out[cell] = AZURE if cell in port_cells else color
    return out


def _choice_without(seq: tuple[int, ...] | list[int], banned: set[int]) -> int:
    options = [x for x in seq if x not in banned]
    return random.choice(options)


def _fragment_cells(
    input_port: tuple[int, int],
    input_facing: tuple[int, int],
    kind: str,
) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    forward = _neg(input_facing)
    if kind == "straight":
        body_len = random.randint(3, 5)
        body = [_add(input_port, _mul(forward, step)) for step in range(1, body_len + 1)]
        output_port = _add(body[-1], forward)
        return body, output_port, forward

    output_facing = random.choice([d for d in DIRS if d[0] * forward[0] + d[1] * forward[1] == 0])
    arm1 = random.randint(2, 4)
    arm2 = random.randint(2, 4)
    first_arm = [_add(input_port, _mul(forward, step)) for step in range(1, arm1 + 1)]
    corner = first_arm[-1]
    second_arm = [_add(corner, _mul(output_facing, step)) for step in range(1, arm2)]
    body = first_arm + second_arm
    output_port = _add(body[-1], output_facing)
    return body, output_port, output_facing


def _cells_clear(
    cells: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> bool:
    if cells & reserved:
        return False
    if cells & occupied:
        return False
    for cell in cells:
        for nb in _neighbors8(cell):
            if nb in occupied and nb not in cells:
                return False
    return True


def _corridor_clear(
    gap_cells: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    src_port: tuple[int, int],
) -> bool:
    gap_set = set(gap_cells)
    if gap_set & occupied or gap_set & reserved:
        return False
    for cell in gap_cells:
        for nb in _neighbors4(cell):
            if nb == src_port:
                continue
            if nb in occupied:
                return False
    return True


def _filter_ok(
    filter_cells: tuple[tuple[int, int], tuple[int, int]],
    attached: tuple[tuple[int, int], tuple[int, int]],
    pipe_set: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> bool:
    fc_set = set(filter_cells)
    if fc_set & occupied or fc_set & reserved:
        return False
    for fc, pc in zip(filter_cells, attached):
        adjs = {nb for nb in _neighbors4(fc) if nb in pipe_set}
        if adjs != {pc}:
            return False
        for nb in _neighbors4(fc):
            if nb in occupied and nb not in attached and nb not in fc_set:
                return False
    return True


def _assemble_path(fragments: list[dict[str, object]]) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = [fragments[0]["input_port"]]  # type: ignore[index]
    for i, frag in enumerate(fragments):
        path.extend(frag["body"])  # type: ignore[arg-type]
        path.append(frag["output_port"])  # type: ignore[arg-type]
        if i + 1 < len(fragments):
            path.extend(frag["gap"])  # type: ignore[arg-type]
            path.append(fragments[i + 1]["input_port"])  # type: ignore[index]
    return path


def _gen_attempt() -> np.ndarray | None:
    n_fragments = random.randint(3, 6)
    unique_port_colors = random.sample(PALETTE, n_fragments + 1)
    start_port_color = unique_port_colors[0]
    middle_colors = unique_port_colors[1:-1]
    end_port_color = unique_port_colors[-1]
    seed_color = _choice_without(PALETTE, {start_port_color})

    input_port = (0, 0)
    input_facing = random.choice(DIRS)
    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()
    fragments: list[dict[str, object]] = []

    for idx in range(n_fragments):
        placed = False
        for _ in range(80):
            kind = "turn" if random.random() < 0.6 else "straight"
            body, output_port, output_facing = _fragment_cells(input_port, input_facing, kind)
            extras = {input_port, output_port}
            if idx == 0:
                extras.add(_add(input_port, input_facing))
            if idx == n_fragments - 1:
                extras.add(_add(output_port, output_facing))
            candidate = set(body) | extras
            if not _cells_clear(candidate, occupied, reserved):
                continue

            gap_len = random.randint(1, 3) if idx < n_fragments - 1 else 0
            gap_cells = [_add(output_port, _mul(output_facing, step)) for step in range(1, gap_len + 1)]
            next_input = _add(output_port, _mul(output_facing, gap_len + 1)) if idx < n_fragments - 1 else None
            if idx < n_fragments - 1 and not _corridor_clear(gap_cells, occupied | candidate, reserved, output_port):
                continue

            fragments.append(
                {
                    "input_port": input_port,
                    "input_facing": input_facing,
                    "input_color": start_port_color if idx == 0 else middle_colors[idx - 1],
                    "body": body,
                    "output_port": output_port,
                    "output_facing": output_facing,
                    "output_color": end_port_color if idx == n_fragments - 1 else middle_colors[idx],
                    "gap": gap_cells,
                }
            )
            occupied |= candidate
            reserved |= set(gap_cells)
            if idx == 0:
                occupied.add(_add(input_port, input_facing))
            if idx == n_fragments - 1:
                occupied.add(_add(output_port, output_facing))
            if idx < n_fragments - 1:
                input_port = next_input  # type: ignore[assignment]
                input_facing = _neg(output_facing)
            placed = True
            break
        if not placed:
            return None

    path = _assemble_path(fragments)
    if len(path) != len(set(path)):
        return None
    path_index = {cell: idx for idx, cell in enumerate(path)}
    pipe_set = set(path)

    filter_candidates = []
    for frag in fragments:
        body = frag["body"]  # type: ignore[assignment]
        for i in range(1, len(body) - 2):
            a = body[i]
            b = body[i + 1]
            if a[0] == b[0]:
                offsets = [(-1, 0), (1, 0)]
            else:
                offsets = [(0, -1), (0, 1)]
            for off in offsets:
                filter_cells = (_add(a, off), _add(b, off))
                if _filter_ok(filter_cells, (a, b), pipe_set, occupied, reserved):
                    filter_candidates.append((path_index[a], a, b, filter_cells))

    if not filter_candidates:
        return None

    random.shuffle(filter_candidates)
    target_filters = random.randint(1, min(3, len(filter_candidates)))
    selected = []
    used_pipe: set[tuple[int, int]] = set()
    used_filter_cells: set[tuple[int, int]] = set()
    for start_idx, a, b, fcells in filter_candidates:
        if len(selected) >= target_filters:
            break
        if a in used_pipe or b in used_pipe:
            continue
        if set(fcells) & used_filter_cells:
            continue
        selected.append((start_idx, a, b, fcells))
        used_pipe.update((a, b))
        used_filter_cells.update(fcells)
    if not selected:
        return None
    selected.sort()

    filter_colors: dict[tuple[int, int], int] = {}
    current = seed_color
    for _, a, b, fcells in selected:
        trigger = random.random() < 0.5
        if trigger:
            in_color = current
            out_color = _choice_without(PALETTE, {current})
            current = out_color
        else:
            in_color = _choice_without(PALETTE, {current})
            out_color = _choice_without(PALETTE, {in_color})
        filter_colors[fcells[0]] = in_color
        filter_colors[fcells[1]] = out_color

    all_cells = set()
    for frag in fragments:
        all_cells.add(frag["input_port"])  # type: ignore[arg-type]
        all_cells.update(frag["body"])  # type: ignore[arg-type]
        all_cells.add(frag["output_port"])  # type: ignore[arg-type]
    start_marker = _add(fragments[0]["input_port"], fragments[0]["input_facing"])  # type: ignore[index]
    end_marker = _add(fragments[-1]["output_port"], fragments[-1]["output_facing"])  # type: ignore[index]
    all_cells |= set(filter_colors)
    all_cells.add(start_marker)
    all_cells.add(end_marker)

    rows = [r for r, _ in all_cells]
    cols = [c for _, c in all_cells]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    if max_r - min_r + 1 > 26 or max_c - min_c + 1 > 26:
        return None

    top = random.randint(1, 3)
    left = random.randint(1, 3)
    bottom = random.randint(1, 3)
    right = random.randint(1, 3)
    shift = (top - min_r, left - min_c)
    h = max_r - min_r + 1 + top + bottom
    w = max_c - min_c + 1 + left + right
    grid = np.full((h, w), BG, dtype=int)

    for frag in fragments:
        for cell in frag["body"]:  # type: ignore[index]
            rr, cc = _add(cell, shift)
            grid[rr, cc] = AZURE

    for frag in fragments:
        rr, cc = _add(frag["input_port"], shift)  # type: ignore[index]
        grid[rr, cc] = int(frag["input_color"])
        rr, cc = _add(frag["output_port"], shift)  # type: ignore[index]
        grid[rr, cc] = int(frag["output_color"])

    for cell, color in filter_colors.items():
        rr, cc = _add(cell, shift)
        grid[rr, cc] = color

    sr, sc = _add(start_marker, shift)
    er, ec = _add(end_marker, shift)
    grid[sr, sc] = seed_color
    grid[er, ec] = MAROON

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if np.count_nonzero(grid == AZURE) < 6:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid pipe puzzle input")
