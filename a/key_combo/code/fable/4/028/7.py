from __future__ import annotations

import random

import numpy as np


BG = 0
DIR_ORDER = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # N, E, S, W
DIR_TO_INDEX = {d: i for i, d in enumerate(DIR_ORDER)}

EMITTER_CATALOG_EAST = [
    [(-3, 0), (-2, 0), (-2, 1), (-1, 0), (-1, 1), (-1, 2)],
    [(-3, 1), (-2, 0), (-2, 1), (-1, 0), (-1, 1)],
    [(-3, 0), (-2, 0), (-2, 1), (-2, 2), (-1, 0), (-1, 2)],
]

GEN_RETRY_LIMIT = 400
MIN_RAY_DOTS = 4
ROWCOL_GAP = 5


def _rotate_point(pt: tuple[int, int], k: int) -> tuple[int, int]:
    r, c = pt
    for _ in range(k % 4):
        r, c = c, -r
    return int(r), int(c)


def _rotate_offsets(offsets: list[tuple[int, int]], k: int) -> list[tuple[int, int]]:
    return [_rotate_point(pt, k) for pt in offsets]


def _longest_step2_run(values: list[int]) -> list[int]:
    if not values:
        return []
    best = [values[0]]
    cur = [values[0]]
    for v in values[1:]:
        if v - cur[-1] == 2:
            cur.append(v)
        else:
            if (len(cur), cur[-1] - cur[0]) > (len(best), best[-1] - best[0]):
                best = cur[:]
            cur = [v]
    if (len(cur), cur[-1] - cur[0]) > (len(best), best[-1] - best[0]):
        best = cur[:]
    return best


def _extract_ray(coords: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    pts = [tuple(map(int, rc)) for rc in coords]
    by_row: dict[int, list[int]] = {}
    by_col: dict[int, list[int]] = {}
    for r, c in pts:
        by_row.setdefault(r, []).append(c)
        by_col.setdefault(c, []).append(r)

    best_axis = None
    best_fixed = None
    best_run: list[int] = []

    for r, cols in by_row.items():
        run = _longest_step2_run(sorted(cols))
        if len(run) >= 3 and (len(run), run[-1] - run[0]) > (len(best_run), best_run[-1] - best_run[0] if best_run else -1):
            best_axis, best_fixed, best_run = "h", r, run
    for c, rows in by_col.items():
        run = _longest_step2_run(sorted(rows))
        if len(run) >= 3 and (len(run), run[-1] - run[0]) > (len(best_run), best_run[-1] - best_run[0] if best_run else -1):
            best_axis, best_fixed, best_run = "v", c, run

    assert best_axis is not None and len(best_run) >= 3, "expected one dotted ray per color"
    if best_axis == "h":
        line = [(int(best_fixed), int(c)) for c in best_run]
    else:
        line = [(int(r), int(best_fixed)) for r in best_run]
    line_set = set(line)
    extras = [pt for pt in pts if pt not in line_set]
    assert extras, "each color must have emitter cells or one residue cell"
    return line, extras


def _endpoint_direction(line: list[tuple[int, int]], at_start: bool) -> tuple[int, int]:
    if at_start:
        a, b = line[0], line[1]
    else:
        a, b = line[-1], line[-2]
    dr = int(np.sign(b[0] - a[0]))
    dc = int(np.sign(b[1] - a[1]))
    return dr, dc


def _attached_endpoint(extras: list[tuple[int, int]], line: list[tuple[int, int]]) -> tuple[int, int]:
    endpoints = [line[0], line[-1]]
    scores = []
    for endpoint in endpoints:
        scores.append(min(abs(endpoint[0] - r) + abs(endpoint[1] - c) for r, c in extras))
    assert scores[0] != scores[1], "example emitter must identify one endpoint"
    return endpoints[int(scores[1] < scores[0])]


def _candidate_emitter(
    exemplar_offsets: list[tuple[int, int]],
    exemplar_dir: tuple[int, int],
    endpoint: tuple[int, int],
    ray_dir: tuple[int, int],
) -> list[tuple[int, int]]:
    k = (DIR_TO_INDEX[ray_dir] - DIR_TO_INDEX[exemplar_dir]) % 4
    offsets = _rotate_offsets(exemplar_offsets, k)
    er, ec = endpoint
    return [(er + dr, ec + dc) for dr, dc in offsets]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore each missing emitter by matching the intact example's pose to the residue beside its dotted ray.

    Input structure:
    - A black grid contains several monochrome dotted rays made of same-color cells spaced one blank apart.
    - Exactly one color also includes a full asymmetric emitter sprite attached to one endpoint of its ray.
    - Every other color has its emitter erased, leaving only one same-color residue pixel near the correct endpoint.

    Rule:
    - For each color, separate the longest step-2 horizontal or vertical run as the ray and treat the remaining same-color cells as emitter matter.
    - From the intact example, read which ray endpoint owns the emitter, the emitter cell offsets relative to that endpoint, and the ray direction that defines the pose.
    - For every residue-only color, try the two ray endpoints; rotate the learned emitter offsets to that endpoint's outward ray direction and place the emitter where the resulting footprint covers the residue pixel.

    Key insight:
    - The intact emitter gives a relative endpoint-anchored template, while the lone residue simultaneously selects the correct end of an otherwise ambiguous dotted ray and confirms the exact rotated placement.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = g.copy()

    colors = [int(color) for color in np.unique(g) if int(color) != BG]
    parsed: dict[int, dict[str, object]] = {}
    for color in colors:
        coords = np.argwhere(g == color)
        line, extras = _extract_ray(coords)
        parsed[color] = {"line": line, "extras": extras}

    exemplar_colors = [color for color, item in parsed.items() if len(item["extras"]) > 1]
    assert len(exemplar_colors) == 1, "expected exactly one intact emitter"
    exemplar_color = exemplar_colors[0]
    exemplar_line = parsed[exemplar_color]["line"]
    exemplar_extras = parsed[exemplar_color]["extras"]
    exemplar_endpoint = _attached_endpoint(exemplar_extras, exemplar_line)
    exemplar_at_start = exemplar_endpoint == exemplar_line[0]
    exemplar_dir = _endpoint_direction(exemplar_line, exemplar_at_start)
    epr, epc = exemplar_endpoint
    exemplar_offsets = [(r - epr, c - epc) for r, c in exemplar_extras]

    h, w = g.shape
    for color, item in parsed.items():
        if color == exemplar_color:
            continue
        line = item["line"]
        extras = item["extras"]
        assert len(extras) == 1, "non-example colors must leave one residue"
        residue = extras[0]

        candidates = []
        for endpoint, at_start in ((line[0], True), (line[-1], False)):
            ray_dir = _endpoint_direction(line, at_start)
            cells = _candidate_emitter(exemplar_offsets, exemplar_dir, endpoint, ray_dir)
            if not all(0 <= r < h and 0 <= c < w for r, c in cells):
                continue
            if residue in cells:
                candidates.append(cells)

        assert len(candidates) == 1, "residue must determine one emitter placement"
        for r, c in candidates[0]:
            out[r, c] = color
    return out


def _safe_lanes(size: int, margin: int) -> list[int]:
    return list(range(margin, size - margin, ROWCOL_GAP))


def _line_cells(endpoint: tuple[int, int], direction: tuple[int, int], steps: int) -> list[tuple[int, int]]:
    r0, c0 = endpoint
    dr, dc = direction
    return [(r0 + 2 * i * dr, c0 + 2 * i * dc) for i in range(steps)]


def _expanded_cells(cells: list[tuple[int, int]], shape: tuple[int, int], pad: int = 1) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _shape_margin(offsets: list[tuple[int, int]]) -> int:
    vals = [abs(v) for pt in offsets for v in pt]
    return max(vals) if vals else 0


def _sample_horizontal_endpoint(width: int, direction: tuple[int, int], both_edges: bool) -> int:
    if direction == (0, 1):
        if both_edges:
            return 0
        choices = list(range(2, width - 2 * (MIN_RAY_DOTS - 1), 2))
        return random.choice(choices)
    if both_edges:
        return width - 1
    choices = list(range(2 * (MIN_RAY_DOTS - 1), width - 2, 2))
    return random.choice(choices)


def _sample_vertical_endpoint(height: int, direction: tuple[int, int], both_edges: bool) -> int:
    if direction == (1, 0):
        if both_edges:
            return 0
        choices = list(range(2, height - 2 * (MIN_RAY_DOTS - 1), 2))
        return random.choice(choices)
    if both_edges:
        return height - 1
    choices = list(range(2 * (MIN_RAY_DOTS - 1), height - 2, 2))
    return random.choice(choices)


def _build_spec(
    family: str,
    lane: int,
    direction: tuple[int, int],
    both_edges: bool,
    shape_east: list[tuple[int, int]],
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
) -> dict[str, object] | None:
    h, w = shape
    east_dir = (0, 1)
    k = (DIR_TO_INDEX[direction] - DIR_TO_INDEX[east_dir]) % 4
    emitter_offsets = _rotate_offsets(shape_east, k)

    if family == "horizontal":
        ep_c = _sample_horizontal_endpoint(w, direction, both_edges)
        endpoint = (lane, ep_c)
        steps = ((w - 1 - ep_c) // 2 + 1) if direction == (0, 1) else (ep_c // 2 + 1)
    else:
        ep_r = _sample_vertical_endpoint(h, direction, both_edges)
        endpoint = (ep_r, lane)
        steps = ((h - 1 - ep_r) // 2 + 1) if direction == (1, 0) else (ep_r // 2 + 1)

    ray = _line_cells(endpoint, direction, steps)
    emitter = [(endpoint[0] + dr, endpoint[1] + dc) for dr, dc in emitter_offsets]
    if not all(0 <= r < h and 0 <= c < w for r, c in emitter):
        return None

    cells = ray + emitter
    if len(set(cells)) != len(cells):
        return None
    padded = _expanded_cells(cells, shape, pad=1)
    if padded & blocked:
        return None
    return {
        "endpoint": endpoint,
        "direction": direction,
        "ray": ray,
        "emitter": emitter,
        "blocked": padded,
        "both_edges": both_edges,
    }


def _render_output(shape: tuple[int, int], specs: list[dict[str, object]], colors: list[int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for spec, color in zip(specs, colors):
        for r, c in spec["ray"]:
            grid[r, c] = color
        for r, c in spec["emitter"]:
            grid[r, c] = color
    return grid


def _erase_emitters(
    full_grid: np.ndarray,
    specs: list[dict[str, object]],
    exemplar_idx: int,
    residue_offset: tuple[int, int],
    shape_east: list[tuple[int, int]],
    colors: list[int],
) -> np.ndarray:
    inp = full_grid.copy()
    east_dir = (0, 1)
    for idx, (spec, color) in enumerate(zip(specs, colors)):
        if idx == exemplar_idx:
            continue
        endpoint = spec["endpoint"]
        direction = spec["direction"]
        k = (DIR_TO_INDEX[direction] - DIR_TO_INDEX[east_dir]) % 4
        rotated_residue = _rotate_point(residue_offset, k)
        residue = (endpoint[0] + rotated_residue[0], endpoint[1] + rotated_residue[1])
        for r, c in spec["emitter"]:
            inp[r, c] = BG
        inp[residue] = color
    return inp


def _gen_attempt() -> np.ndarray | None:
    family = random.choice(["horizontal", "vertical"])
    h = random.randrange(21, 30, 2)
    w = random.randrange(21, 30, 2)

    shape_east = [tuple(pt) for pt in random.choice(EMITTER_CATALOG_EAST)]
    residue_offset = random.choice(shape_east)
    margin = _shape_margin(shape_east)
    lanes = _safe_lanes(h if family == "horizontal" else w, margin)
    max_rays = min(5, len(lanes))
    if max_rays < 3:
        return None

    n_rays = random.randint(3, max_rays)
    chosen_lanes = sorted(random.sample(lanes, n_rays))
    colors = random.sample(list(range(1, 10)), n_rays)
    exemplar_idx = random.randrange(n_rays)

    if family == "horizontal":
        dir_pool = [(0, 1), (0, -1)]
    else:
        dir_pool = [(1, 0), (-1, 0)]

    directions = [random.choice(dir_pool) for _ in range(n_rays)]
    if n_rays >= 2 and len(set(directions)) == 1 and random.random() < 0.7:
        directions[random.randrange(n_rays)] = dir_pool[1 - dir_pool.index(directions[0])]

    both_flags = [False] * n_rays
    both_flags[random.randrange(n_rays)] = True
    for i in range(n_rays):
        if not both_flags[i] and random.random() < 0.35:
            both_flags[i] = True

    specs: list[dict[str, object]] = []
    blocked: set[tuple[int, int]] = set()
    for lane, direction, both_edges in zip(chosen_lanes, directions, both_flags):
        spec = _build_spec(
            family=family,
            lane=lane,
            direction=direction,
            both_edges=both_edges,
            shape_east=shape_east,
            shape=(h, w),
            blocked=blocked,
        )
        if spec is None:
            return None
        specs.append(spec)
        blocked |= spec["blocked"]

    out = _render_output((h, w), specs, colors)
    inp = _erase_emitters(out, specs, exemplar_idx, residue_offset, shape_east, colors)

    parsed = []
    for color in colors:
        line, extras = _extract_ray(np.argwhere(inp == color))
        parsed.append((line, extras))
    if sum(len(extras) > 1 for _, extras in parsed) != 1:
        return None
    if not any(spec["both_edges"] for spec in specs):
        return None
    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
