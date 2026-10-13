from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GATE = 1
PATH = 3
SEPARATOR = 4
MARKER = 5

MAX_GENERATION_ATTEMPTS = 400

Coord = tuple[int, int]


def _neighbors4(shape: tuple[int, int], r: int, c: int) -> list[Coord]:
    h, w = shape
    out: list[Coord] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[Coord] = deque([(r, c)])
            seen[r, c] = True
            comp: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(mask.shape, rr, cc):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _segment(a: Coord, b: Coord) -> list[Coord]:
    if a == b:
        return [a]
    if a[0] == b[0]:
        step = 1 if b[1] > a[1] else -1
        return [(a[0], c) for c in range(a[1], b[1] + step, step)]
    if a[1] == b[1]:
        step = 1 if b[0] > a[0] else -1
        return [(r, a[1]) for r in range(a[0], b[0] + step, step)]
    raise AssertionError("segments must be axis-aligned")


def _concat_segments(points: list[Coord]) -> list[Coord]:
    out: list[Coord] = []
    for i in range(len(points) - 1):
        seg = _segment(points[i], points[i + 1])
        out.extend(seg if i == 0 else seg[1:])
    return out


def _find_separator(g: np.ndarray) -> int:
    candidates = [r for r in range(g.shape[0]) if np.all(g[r, :] == SEPARATOR)]
    assert len(candidates) == 1, "expected exactly one full-width separator row"
    return int(candidates[0])


def _trace_path(order_start: Coord, coords_set: set[Coord], shape: tuple[int, int]) -> list[Coord]:
    order = [order_start]
    prev: Coord | None = None
    cur = order_start
    while True:
        nxts = [n for n in _neighbors4(shape, cur[0], cur[1]) if n in coords_set and n != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "path must not branch"
        prev, cur = cur, nxts[0]
        order.append(cur)
    assert len(order) == len(coords_set), "component must be a single simple path"
    return order


def _parse_paths(g: np.ndarray, sep: int) -> list[dict[str, object]]:
    upper = g[:sep, :]
    mask = (upper == PATH) | (upper == MARKER)
    paths: list[dict[str, object]] = []
    for comp in _components(mask):
        coords_set = set(comp)
        markers = [p for p in comp if int(g[p]) == MARKER]
        assert len(markers) == 1, "each path component must have exactly one marker"
        marker = markers[0]

        degrees = {
            p: sum(1 for n in _neighbors4(g.shape, p[0], p[1]) if n in coords_set)
            for p in comp
        }
        assert max(degrees.values()) <= 2, "path component must be non-branching"
        endpoints = [p for p, deg in degrees.items() if deg == 1]
        assert len(endpoints) == 2, "path component must have two endpoints"
        assert marker in endpoints, "marker must sit at a path endpoint"

        order = _trace_path(marker, coords_set, g.shape)
        paths.append(
            {
                "marker": marker,
                "order": order,
                "coords_set": coords_set,
                "green_set": {p for p in comp if int(g[p]) == PATH},
            }
        )

    paths.sort(key=lambda item: tuple(item["marker"]))  # deterministic pairing for validation
    assert paths, "expected at least one upper-half path"
    return paths


def _parse_gates(g: np.ndarray, sep: int) -> list[Coord]:
    h, _w = g.shape
    gates = [tuple(map(int, coord)) for coord in np.argwhere(g == GATE)]
    for r, c in gates:
        assert sep + 1 < r < h - 1, "gate center must stay away from separator and border"
        assert int(g[r - 1, c]) == BG and int(g[r + 1, c]) == BG, "gate must be black-blue-black"
    gates.sort()
    assert gates, "expected at least one gate"
    return gates


def _match_targets(
    g: np.ndarray, sep: int, paths: list[dict[str, object]], gates: list[Coord]
) -> tuple[list[Coord], set[int]]:
    targets: list[Coord] = []
    used_gate_indices: set[int] = set()

    for path in paths:
        green_set: set[Coord] = path["green_set"]  # type: ignore[assignment]
        order: list[Coord] = path["order"]  # type: ignore[assignment]

        exact: list[tuple[int, Coord]] = []
        for gate_idx, (gr, gc) in enumerate(gates):
            mirror = (2 * sep - gr, gc)
            if mirror in green_set:
                exact.append((gate_idx, mirror))

        if exact:
            assert len(exact) == 1, "path must hit exactly one mirrored gate position"
            gate_idx, target = exact[0]
        else:
            fallback: list[tuple[int, Coord]] = []
            for gate_idx, (_gr, gc) in enumerate(gates):
                green_on_column = [p for p in order[1:] if int(g[p]) == PATH and p[1] == gc]
                if len(green_on_column) == 1:
                    fallback.append((gate_idx, green_on_column[0]))
            assert len(fallback) == 1, "path must have exactly one gate-determined stop column"
            gate_idx, target = fallback[0]

        assert gate_idx not in used_gate_indices, "different paths must use different gates"
        used_gate_indices.add(gate_idx)
        targets.append(target)

    return targets, used_gate_indices


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect lower-half gate locations across the separator to decide where each upper marker stops on its path.

    Input structure:
    - Black background with one full-width horizontal yellow separator row.
    - Above the separator are 2-3 disjoint non-branching green paths; each path has one gray marker on exactly one endpoint.
    - Below the separator are blue gate centers, each implicitly forming a black-blue-black vertical gate.

    Rule:
    - Reflect every gate center across the yellow separator, keeping its column and mirroring its row.
    - For each path, find the unique mirrored gate position it hits; if only the reflected column is exposed, use the unique green path cell on that column as the stop cell.
    - Slide the gray marker along its own path until it reaches that stop cell.
    - Erase every traversed green path cell before the stop cell to black, recolor the stop cell gray, and leave the untraversed tail, separator, and lower-half gates unchanged.

    Key insight:
    - The blue gates do not directly touch the paths; their reflected positions across the separator identify the path-local stopping cells that control the subsequent erase-while-sliding move.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    sep = _find_separator(g)
    paths = _parse_paths(g, sep)
    gates = _parse_gates(g, sep)
    targets, _used = _match_targets(g, sep, paths, gates)

    out = g.copy()
    for path, target in zip(paths, targets):
        order: list[Coord] = path["order"]  # type: ignore[assignment]
        assert target in order and target != order[0], "marker must move forward to a green stop cell"
        stop_idx = order.index(target)
        for r, c in order[:stop_idx]:
            out[r, c] = BG
        out[target] = MARKER

    return out


def _composition(total: int, parts: int) -> list[int]:
    if total <= 0:
        return [0] * parts
    return list(np.random.multinomial(int(total), [1.0 / parts] * parts))


def _target_row_bounds(h: int, sep: int) -> tuple[int, int] | None:
    low = max(0, 2 * sep - (h - 2))
    high = sep - 2
    if low > high:
        return None
    return int(low), int(high)


def _sample_band_intervals(width: int, n_paths: int) -> list[tuple[int, int]] | None:
    widths = [4] * n_paths
    required = sum(widths) + (n_paths - 1)
    if required > width:
        return None

    extra = width - required
    widenable = [i for i in range(n_paths)]
    random.shuffle(widenable)
    while extra > 0 and widenable:
        idx = random.choice(widenable)
        if widths[idx] < 5:
            widths[idx] += 1
            extra -= 1
        else:
            widenable.remove(idx)

    gaps = _composition(extra, n_paths + 1)
    intervals: list[tuple[int, int]] = []
    c = gaps[0]
    for i, band_width in enumerate(widths):
        intervals.append((int(c), int(c + band_width - 1)))
        c += band_width
        if i + 1 < n_paths:
            c += 1 + gaps[i + 1]
    return intervals


def _has_turn(order: list[Coord]) -> bool:
    for i in range(1, len(order) - 1):
        dr1 = order[i][0] - order[i - 1][0]
        dc1 = order[i][1] - order[i - 1][1]
        dr2 = order[i + 1][0] - order[i][0]
        dc2 = order[i + 1][1] - order[i][1]
        if (dr1, dc1) != (dr2, dc2):
            return True
    return False


def _is_induced_path(order: list[Coord], shape: tuple[int, int], occupied: set[Coord]) -> bool:
    if len(order) != len(set(order)):
        return False

    coords_set = set(order)
    for i, (r, c) in enumerate(order):
        if not (0 <= r < shape[0] and 0 <= c < shape[1]):
            return False
        if (r, c) in occupied:
            return False
        if any(n in occupied for n in _neighbors4(shape, r, c)):
            return False

        allowed_neighbors = set()
        if i > 0:
            allowed_neighbors.add(order[i - 1])
        if i + 1 < len(order):
            allowed_neighbors.add(order[i + 1])
        actual_neighbors = {n for n in _neighbors4(shape, r, c) if n in coords_set}
        if actual_neighbors != allowed_neighbors:
            return False

    return True


def _build_bridge_path(
    h: int,
    w: int,
    sep: int,
    band: tuple[int, int],
    occupied: set[Coord],
) -> dict[str, object] | None:
    row_bounds = _target_row_bounds(h, sep)
    if row_bounds is None:
        return None
    row_low, row_high = row_bounds
    if row_low > row_high:
        return None

    c0, c1 = band
    left_col, right_col = c0, c1
    if right_col - left_col < 3:
        return None

    all_rows = list(range(sep))
    target_row = random.randint(row_low, row_high)
    start_on_left = bool(random.getrandbits(1))
    start_col = left_col if start_on_left else right_col
    end_col = right_col if start_on_left else left_col

    prefer_bent = random.random() < 0.8
    if prefer_bent:
        non_mid_rows = [r for r in all_rows if r != target_row]
        if len(non_mid_rows) < 2:
            return None
        start_row = random.choice(non_mid_rows)
        end_row = random.choice(non_mid_rows)
    else:
        start_row = random.choice(all_rows)
        end_row = random.choice(all_rows)
        if start_row == target_row and end_row == target_row:
            return None

    waypoints = [
        (start_row, start_col),
        (target_row, start_col),
        (target_row, end_col),
        (end_row, end_col),
    ]
    order = _concat_segments(waypoints)
    if len(order) < 5:
        return None
    if not _is_induced_path(order, (h, w), occupied):
        return None

    target_columns = list(range(min(start_col, end_col) + 1, max(start_col, end_col)))
    if not target_columns:
        return None
    target_col = random.choice(target_columns)
    target = (target_row, target_col)
    if target not in order:
        return None

    green_columns = [c for _r, c in order[1:]]
    if green_columns.count(target_col) != 1:
        return None

    target_idx = order.index(target)
    if target_idx <= 0 or target_idx >= len(order) - 1:
        return None

    gate_center = (2 * sep - target_row, target_col)
    if not (sep + 1 < gate_center[0] < h - 1):
        return None

    return {
        "order": order,
        "marker": order[0],
        "target": target,
        "target_idx": target_idx,
        "gate_center": gate_center,
        "has_turn": _has_turn(order),
    }


def _pick_extra_gate_columns(w: int, specs: list[dict[str, object]]) -> list[int]:
    used_gate_cols = {int(spec["gate_center"][1]) for spec in specs}  # type: ignore[index]
    path_cols = {c for spec in specs for _r, c in spec["order"]}  # type: ignore[index]
    candidates = [c for c in range(w) if c not in used_gate_cols and c not in path_cols]
    random.shuffle(candidates)

    max_extra = min(2, len(candidates))
    if max_extra == 0 or random.random() < 0.35:
        return []
    want = random.randint(1, max_extra)

    chosen: list[int] = []
    for c in candidates:
        if all(abs(c - other) >= 2 for other in chosen):
            chosen.append(c)
        if len(chosen) == want:
            break
    return chosen[:want]


def _render_case(
    h: int,
    w: int,
    sep: int,
    specs: list[dict[str, object]],
    extra_gates: list[Coord],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((h, w), BG, dtype=int)
    inp[sep, :] = SEPARATOR

    for spec in specs:
        order: list[Coord] = spec["order"]  # type: ignore[assignment]
        inp[order[0]] = MARKER
        for coord in order[1:]:
            inp[coord] = PATH
        gate_center: Coord = spec["gate_center"]  # type: ignore[assignment]
        inp[gate_center] = GATE

    for gate in extra_gates:
        inp[gate] = GATE

    expected = inp.copy()
    for spec in specs:
        order: list[Coord] = spec["order"]  # type: ignore[assignment]
        target_idx = int(spec["target_idx"])
        for coord in order[:target_idx]:
            expected[coord] = BG
        expected[order[target_idx]] = MARKER

    return inp, expected


def _validate_generated_case(
    inp: np.ndarray,
    expected: np.ndarray,
    specs: list[dict[str, object]],
    extra_gates: list[Coord],
) -> bool:
    sep = _find_separator(inp)
    parsed_paths = _parse_paths(inp, sep)
    parsed_gates = _parse_gates(inp, sep)
    parsed_targets, used_gate_indices = _match_targets(inp, sep, parsed_paths, parsed_gates)

    if len(parsed_paths) != len(specs):
        return False
    if len(parsed_gates) != len(specs) + len(extra_gates):
        return False
    if len(used_gate_indices) != len(specs):
        return False

    specs_sorted = sorted(specs, key=lambda item: tuple(item["marker"]))  # type: ignore[arg-type]
    for parsed, spec, target in zip(parsed_paths, specs_sorted, parsed_targets):
        if tuple(parsed["marker"]) != tuple(spec["marker"]):  # type: ignore[index]
            return False
        if tuple(target) != tuple(spec["target"]):  # type: ignore[index]
            return False
        order: list[Coord] = spec["order"]  # type: ignore[assignment]
        target_idx = int(spec["target_idx"])
        if target_idx <= 0 or target_idx >= len(order) - 1:
            return False

    counts = np.bincount(inp.ravel(), minlength=10)
    if int(np.argmax(counts)) != BG:
        return False
    if set(map(int, np.unique(inp))) - {BG, GATE, PATH, SEPARATOR, MARKER}:
        return False
    if not any(bool(spec["has_turn"]) for spec in specs):
        return False

    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(14, 19)
    w = random.randint(15, 20)
    sep = random.randint(5, min(8, h - 5))
    n_paths = random.randint(2, 3)

    if _target_row_bounds(h, sep) is None:
        return None

    bands = _sample_band_intervals(w, n_paths)
    if bands is None:
        return None

    specs: list[dict[str, object]] = []
    occupied: set[Coord] = set()
    for band in bands:
        spec = _build_bridge_path(h, w, sep, band, occupied)
        if spec is None:
            return None
        specs.append(spec)
        occupied.update(spec["order"])  # type: ignore[arg-type]

    if not any(bool(spec["has_turn"]) for spec in specs):
        return None

    extra_cols = _pick_extra_gate_columns(w, specs)
    extra_gates = [
        (random.randint(sep + 2, h - 2), c)
        for c in extra_cols
    ]

    inp, expected = _render_case(h, w, sep, specs, extra_gates)
    if not _validate_generated_case(inp, expected, specs, extra_gates):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid reflection-gated slide input")
