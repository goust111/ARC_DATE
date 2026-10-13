from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
RAY_COLORS = [3, 4, 5, 6, 7, 8, 9]
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
MAX_GEN_ATTEMPTS = 500


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors(r: int, c: int, connectivity: int) -> list[tuple[int, int]]:
    if connectivity == 4:
        return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if dr != 0 or dc != 0
    ]


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in neighbors(rr, cc, connectivity):
                    if in_bounds((h, w), nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def marker_kind(coords: np.ndarray) -> str:
    r0, c0, r1, c1 = bbox_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h == 1 and w == 1:
        return "point"
    if h == 1 and w in (3, 4) and len(coords) == w:
        return "horizontal"
    if w == 1 and h in (3, 4) and len(coords) == h:
        return "vertical"
    raise AssertionError("invalid red marker geometry")


def parse_seed(coords: np.ndarray, shape: tuple[int, int]) -> dict:
    n = len(coords)
    assert n in (2, 3), "seed length must be 2 or 3"
    r0, c0, r1, c1 = bbox_coords(coords)
    assert r1 - r0 + 1 == n and c1 - c0 + 1 == n, "seed must be a short 45-degree line"
    rel = {(int(r - r0), int(c - c0)) for r, c in coords}
    diag1 = {(i, i) for i in range(n)}
    diag2 = {(i, n - 1 - i) for i in range(n)}
    assert rel in (diag1, diag2), "seed pixels must form a contiguous diagonal segment"

    pts = sorted((int(r), int(c)) for r, c in coords)
    end_a = min(
        pts,
        key=lambda rc: sum(
            1 for rr, cc in pts if max(abs(rc[0] - rr), abs(rc[1] - cc)) == 1
        ),
    )
    end_b = max(
        pts,
        key=lambda rc: sum(
            1 for rr, cc in pts if max(abs(rc[0] - rr), abs(rc[1] - cc)) == 1
        ),
    )
    if end_a == end_b:
        end_a, end_b = pts[0], pts[-1]

    candidates = []
    for start, other in ((end_a, end_b), (end_b, end_a)):
        dr = start[0] - other[0]
        dc = start[1] - other[1]
        dr = 1 if dr > 0 else -1
        dc = 1 if dc > 0 else -1
        nr, nc = start[0] + dr, start[1] + dc
        if in_bounds(shape, nr, nc):
            candidates.append((start, (dr, dc)))
    assert len(candidates) == 1, "seed launch endpoint must be uniquely determined by the border"
    start, direction = candidates[0]
    return {
        "cells": {tuple(map(int, rc)) for rc in coords},
        "start": start,
        "dir": direction,
    }


def extract_scene(grid: np.ndarray) -> tuple[list[dict], list[dict], set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    shape = g.shape

    marker_comps = components_mask(g == RED, connectivity=8)
    markers: list[dict] = []
    marker_cells: set[tuple[int, int]] = set()
    for idx, comp in enumerate(marker_comps):
        kind = marker_kind(comp)
        cells = {tuple(map(int, rc)) for rc in comp}
        marker_cells |= cells
        markers.append({"id": idx, "kind": kind, "cells": cells})

    seed_components: list[dict] = []
    seed_cells: set[tuple[int, int]] = set()
    for color in sorted(int(v) for v in np.unique(g) if int(v) not in (BG, RED, BLUE)):
        for comp in components_mask(g == color, connectivity=8):
            seed = parse_seed(comp, shape)
            seed["color"] = color
            seed_components.append(seed)
            seed_cells |= seed["cells"]

    assert 2 <= len(seed_components) <= 3, "grid must contain 2 or 3 ray seeds"
    assert 3 <= len(markers) <= 5, "grid must contain 3 to 5 red markers"
    return markers, seed_components, seed_cells


def reflect(direction: tuple[int, int], kind: str) -> tuple[int, int]:
    dr, dc = direction
    if kind == "horizontal":
        return -dr, dc
    if kind == "vertical":
        return dr, -dc
    if kind == "point":
        return -dr, -dc
    raise AssertionError("unknown marker type")


def trace_ray(
    shape: tuple[int, int],
    marker_by_cell: dict[tuple[int, int], int],
    markers: list[dict],
    seed: dict,
    kind_override: str | None = None,
) -> dict:
    pos = seed["start"]
    direction = seed["dir"]
    path = [pos]
    path_set = {pos}
    post_first: set[tuple[int, int]] = set()
    reflections: list[str] = []
    seen_states: set[tuple[int, int, int, int]] = set()

    reflected_once = False
    while True:
        state = (pos[0], pos[1], direction[0], direction[1])
        assert state not in seen_states, "ray entered a reflection loop"
        seen_states.add(state)

        nr = pos[0] + direction[0]
        nc = pos[1] + direction[1]
        if not in_bounds(shape, nr, nc):
            break

        if (nr, nc) in marker_by_cell:
            marker = markers[marker_by_cell[(nr, nc)]]
            kind = marker["kind"] if kind_override is None else kind_override
            reflections.append(marker["kind"])
            direction = reflect(direction, kind)
            reflected_once = True
            continue

        pos = (nr, nc)
        path.append(pos)
        path_set.add(pos)
        if reflected_once:
            post_first.add(pos)

    return {
        "path": path,
        "path_set": path_set,
        "post_first": post_first,
        "reflections": reflections,
    }


def render_output(grid: np.ndarray, kind_override: str | None = None) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    markers, seeds, seed_cells = extract_scene(g)
    marker_by_cell = {}
    for marker in markers:
        for cell in marker["cells"]:
            marker_by_cell[cell] = marker["id"]

    traces = [trace_ray(g.shape, marker_by_cell, markers, seed, kind_override) for seed in seeds]
    coverage: dict[tuple[int, int], set[int]] = defaultdict(set)
    red_cells = set(marker_by_cell)
    for idx, trace in enumerate(traces):
        for cell in trace["path_set"]:
            if cell in seed_cells or cell in red_cells:
                continue
            coverage[cell].add(idx)

    out = g.copy()
    for cell, owners in coverage.items():
        if len(owners) >= 2:
            out[cell] = BLUE
        else:
            out[cell] = seeds[next(iter(owners))]["color"]
    return out, traces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend diagonal ray seeds, reflect them on red markers whose geometry encodes the reflection type, and mark shared path cells blue.

    Input structure:
    - The grid uses black background with 2 or 3 non-red diagonal ray seeds, each a 2- or 3-pixel 45-degree monochrome segment of a distinct color.
    - Red (2) foreground components are isolated markers of exactly one of three geometries: a horizontal 3-4 cell segment, a vertical 3-4 cell segment, or a single pixel.

    Rule:
    - For each seed, use the endpoint whose forward diagonal extension stays inside the grid as the launch point, and fire along the diagonal from the opposite endpoint toward it.
    - Advance the ray one diagonal cell at a time; if the next cell is a red marker, do not enter it and instead reflect according to that marker's geometry: horizontal flips the row direction, vertical flips the column direction, and a single pixel flips both.
    - Continue until the next step would leave the grid, combining all ray paths afterward.
    - Red marker cells and original seed cells stay unchanged; every other covered cell keeps the unique ray color if exactly one ray reaches it, or becomes blue (1) if multiple rays reach it.

    Key insight:
    - The red geometry is functional rather than decorative: line orientation decides which motion component flips, and only the accumulated post-reflection ray paths reveal the blue collision pattern.
    """
    out, _ = render_output(np.asarray(grid, dtype=int))
    return out


def line_cells_until_boundary(
    start: tuple[int, int],
    direction: tuple[int, int],
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    cells = []
    r, c = start
    while in_bounds(shape, r, c):
        cells.append((r, c))
        r += direction[0]
        c += direction[1]
    return cells


def boundary_seed_for_target(
    target: tuple[int, int],
    direction: tuple[int, int],
    shape: tuple[int, int],
    length: int,
) -> dict | None:
    line = line_cells_until_boundary(target, (-direction[0], -direction[1]), shape)
    boundary_entry = line[-1]
    dist = max(abs(target[0] - boundary_entry[0]), abs(target[1] - boundary_entry[1]))
    if dist < length + 1:
        return None
    start = (
        boundary_entry[0] + (length - 1) * direction[0],
        boundary_entry[1] + (length - 1) * direction[1],
    )
    cells = {
        (
            start[0] - k * direction[0],
            start[1] - k * direction[1],
        )
        for k in range(length)
    }
    return {"start": start, "dir": direction, "cells": cells, "length": length}


def halo(cells: set[tuple[int, int]], shape: tuple[int, int], radius: int = 1) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = r + dr, c + dc
                if in_bounds(shape, rr, cc):
                    out.add((rr, cc))
    return out


def make_segment(
    hit_cell: tuple[int, int],
    kind: str,
    length: int,
    shape: tuple[int, int],
    rng: random.Random,
) -> set[tuple[int, int]] | None:
    hr, hc = hit_cell
    shifts = list(range(length))
    rng.shuffle(shifts)
    for shift in shifts:
        if kind == "horizontal":
            cells = {(hr, hc - shift + i) for i in range(length)}
        elif kind == "vertical":
            cells = {(hr - shift + i, hc) for i in range(length)}
        else:
            cells = {hit_cell}
        if all(in_bounds(shape, r, c) for r, c in cells):
            return cells
    return None


def candidate_seed_starts(shape: tuple[int, int]) -> list[dict]:
    h, w = shape
    out = []
    for direction in DIAG_DIRS:
        for length in (2, 3):
            for r in range(h):
                for c in range(w):
                    start = (r, c)
                    forward = (r + direction[0], c + direction[1])
                    if not in_bounds(shape, forward[0], forward[1]):
                        continue
                    cells = {
                        (r - k * direction[0], c - k * direction[1])
                        for k in range(length)
                    }
                    if not all(in_bounds(shape, rr, cc) for rr, cc in cells):
                        continue
                    tail = (r - (length - 1) * direction[0], c - (length - 1) * direction[1])
                    pred = (tail[0] - direction[0], tail[1] - direction[1])
                    if in_bounds(shape, pred[0], pred[1]):
                        continue
                    out.append({"start": start, "dir": direction, "cells": cells, "length": length})
    return out


def build_input_grid(
    shape: tuple[int, int],
    markers: list[dict],
    seeds: list[dict],
    colors: list[int],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for marker in markers:
        for r, c in marker["cells"]:
            grid[r, c] = RED
    for seed, color in zip(seeds, colors, strict=True):
        for r, c in seed["cells"]:
            grid[r, c] = color
    return grid


def semantic_checks(grid: np.ndarray) -> bool:
    try:
        correct, traces = render_output(grid)
    except AssertionError:
        return False

    markers, seeds, seed_cells = extract_scene(grid)
    marker_by_cell = {}
    for marker in markers:
        for cell in marker["cells"]:
            marker_by_cell[cell] = marker["id"]

    if len({seed["color"] for seed in seeds}) != len(seeds):
        return False

    hit_types = set()
    for trace in traces:
        if not trace["reflections"]:
            return False
        hit_types.update(trace["reflections"])
    if hit_types != {"horizontal", "vertical", "point"}:
        return False

    pairwise_post_overlap = False
    for i in range(len(traces)):
        for j in range(i + 1, len(traces)):
            if traces[i]["post_first"] & traces[j]["post_first"]:
                pairwise_post_overlap = True
    if not pairwise_post_overlap:
        return False

    coverage: dict[tuple[int, int], int] = defaultdict(int)
    marker_cells = set(marker_by_cell)
    for trace in traces:
        for cell in trace["path_set"]:
            if cell in seed_cells or cell in marker_cells:
                continue
            coverage[cell] += 1
    overlap_cells = [cell for cell, count in coverage.items() if count >= 2]
    if not overlap_cells or len(overlap_cells) > max(10, len(coverage) // 2):
        return False

    wrong_different = 0
    for override in ("horizontal", "vertical", "point"):
        try:
            wrong = render_output(grid, kind_override=override)[0]
        except AssertionError:
            wrong_different += 1
            continue
        if not np.array_equal(correct, wrong):
            wrong_different += 1
    if wrong_different < 2:
        return False

    if np.array_equal(correct, grid):
        return False
    return True


def generate_base_layout(rng: random.Random) -> tuple[np.ndarray, list[dict], list[dict]] | None:
    shape = (rng.randint(18, 24), rng.randint(18, 24))
    direction = rng.choice(DIAG_DIRS)
    incoming_a = (-direction[0], direction[1])
    incoming_b = (direction[0], -direction[1])

    candidates = candidate_seed_starts(shape)
    candidates_by_key = {(cand["start"], cand["dir"], cand["length"]): cand for cand in candidates}

    for _ in range(200):
        jr = rng.randint(4, shape[0] - 5)
        jc = rng.randint(4, shape[1] - 5)
        join = (jr, jc)
        prev = (join[0] - direction[0], join[1] - direction[1])
        hit_a = (prev[0] + incoming_a[0], prev[1] + incoming_a[1])
        hit_b = (prev[0] + incoming_b[0], prev[1] + incoming_b[1])
        if not all(in_bounds(shape, r, c) for r, c in (join, prev, hit_a, hit_b)):
            continue

        max_forward = 0
        probe = join
        while in_bounds(shape, probe[0] + direction[0], probe[1] + direction[1]):
            max_forward += 1
            probe = (probe[0] + direction[0], probe[1] + direction[1])
        if max_forward < 3:
            continue
        single_dist = rng.randint(2, max_forward)
        point_hit = (join[0] + single_dist * direction[0], join[1] + single_dist * direction[1])

        seg_a = make_segment(hit_a, "horizontal", rng.choice((3, 4)), shape, rng)
        seg_b = make_segment(hit_b, "vertical", rng.choice((3, 4)), shape, rng)
        if seg_a is None or seg_b is None:
            continue

        marker_specs = [
            {"kind": "horizontal", "cells": seg_a},
            {"kind": "vertical", "cells": seg_b},
            {"kind": "point", "cells": {point_hit}},
        ]
        occupied_markers: set[tuple[int, int]] = set()
        occupied_halo: set[tuple[int, int]] = set()
        ok = True
        for marker in marker_specs:
            if marker["cells"] & occupied_halo:
                ok = False
                break
            occupied_markers |= marker["cells"]
            occupied_halo |= halo(marker["cells"], shape)
        if not ok:
            continue
        if join in occupied_markers or prev in occupied_markers:
            continue

        len_a = rng.choice((2, 3))
        len_b = rng.choice((2, 3))
        seed_a = boundary_seed_for_target(prev, incoming_a, shape, len_a)
        seed_b = boundary_seed_for_target(prev, incoming_b, shape, len_b)
        if seed_a is None or seed_b is None:
            continue
        if seed_a["cells"] & occupied_halo or seed_b["cells"] & occupied_halo:
            continue
        if seed_a["cells"] & halo(seed_b["cells"], shape):
            continue

        seeds = [seed_a, seed_b]
        colors = rng.sample(RAY_COLORS, 3)
        grid = build_input_grid(shape, marker_specs, seeds, colors[:2])
        try:
            _, traces = render_output(grid)
        except AssertionError:
            continue

        if {trace["reflections"][0] for trace in traces} != {"horizontal", "vertical"}:
            continue
        if "point" not in traces[0]["reflections"] or "point" not in traces[1]["reflections"]:
            continue
        if not (traces[0]["post_first"] & traces[1]["post_first"]):
            continue

        return grid, marker_specs, seeds
    return None


def maybe_add_third_seed(
    grid: np.ndarray,
    markers: list[dict],
    seeds: list[dict],
    rng: random.Random,
) -> tuple[np.ndarray, list[dict]] | tuple[None, None]:
    shape = grid.shape
    existing_cells = set()
    for marker in markers:
        existing_cells |= marker["cells"]
    for seed in seeds:
        existing_cells |= seed["cells"]
    forbidden = halo(existing_cells, shape)

    _, base_traces = render_output(grid)
    post_union = set().union(*(trace["post_first"] for trace in base_traces))

    candidates = candidate_seed_starts(shape)
    rng.shuffle(candidates)
    used_starts = {seed["start"] for seed in seeds}
    for cand in candidates:
        if cand["start"] in used_starts:
            continue
        if cand["cells"] & forbidden:
            continue
        trial_seeds = seeds + [cand]
        color_pool = [int(v) for v in np.unique(grid) if int(v) not in (BG, RED)]
        for color in RAY_COLORS:
            if color in color_pool:
                continue
            trial_grid = grid.copy()
            for r, c in cand["cells"]:
                trial_grid[r, c] = color
            try:
                _, traces = render_output(trial_grid)
            except AssertionError:
                continue
            third = traces[-1]
            if not third["reflections"]:
                continue
            if not (third["post_first"] & post_union):
                continue
            return trial_grid, trial_seeds
    return None, None


def add_extra_markers(
    grid: np.ndarray,
    markers: list[dict],
    seeds: list[dict],
    rng: random.Random,
) -> tuple[np.ndarray, list[dict]] | tuple[None, None]:
    shape = grid.shape
    try:
        _, traces = render_output(grid)
    except AssertionError:
        return None, None
    ray_cells = set().union(*(trace["path_set"] for trace in traces))

    count_target = rng.randint(3, 5)
    extras_needed = count_target - len(markers)
    if extras_needed <= 0:
        return grid, markers

    occupied = set()
    for marker in markers:
        occupied |= marker["cells"]
    for seed in seeds:
        occupied |= seed["cells"]
    occupied_halo = halo(occupied, shape)

    new_markers = list(markers)
    work = grid.copy()
    for _ in range(extras_needed):
        placed = False
        options = ["horizontal", "vertical", "point"]
        rng.shuffle(options)
        for _ in range(100):
            kind = rng.choice(options)
            if kind == "point":
                cells = {(rng.randrange(shape[0]), rng.randrange(shape[1]))}
            elif kind == "horizontal":
                length = rng.choice((3, 4))
                r = rng.randrange(shape[0])
                c = rng.randrange(shape[1] - length + 1)
                cells = {(r, c + i) for i in range(length)}
            else:
                length = rng.choice((3, 4))
                r = rng.randrange(shape[0] - length + 1)
                c = rng.randrange(shape[1])
                cells = {(r + i, c) for i in range(length)}
            if cells & occupied_halo:
                continue
            if cells & ray_cells:
                continue
            occupied |= cells
            occupied_halo = halo(occupied, shape)
            marker = {"kind": kind, "cells": cells}
            new_markers.append(marker)
            for rr, cc in cells:
                work[rr, cc] = RED
            placed = True
            break
        if not placed:
            return None, None
    return work, new_markers


def random_marker_cells(
    kind: str,
    shape: tuple[int, int],
    rng: random.Random,
    central: bool,
) -> set[tuple[int, int]]:
    h, w = shape
    margin = 2
    if kind == "point":
        if central:
            r_lo, r_hi = max(margin, h // 4), min(h - margin - 1, (3 * h) // 4)
            c_lo, c_hi = max(margin, w // 4), min(w - margin - 1, (3 * w) // 4)
        else:
            r_lo, r_hi = margin, h - margin - 1
            c_lo, c_hi = margin, w - margin - 1
        return {(rng.randint(r_lo, r_hi), rng.randint(c_lo, c_hi))}

    length = rng.choice((3, 4))
    if kind == "horizontal":
        if central:
            r_lo, r_hi = max(margin, h // 4), min(h - margin - 1, (3 * h) // 4)
            c_lo, c_hi = margin, w - margin - length
        else:
            r_lo, r_hi = margin, h - margin - 1
            c_lo, c_hi = margin, w - margin - length
        r = rng.randint(r_lo, r_hi)
        c = rng.randint(c_lo, c_hi)
        return {(r, c + i) for i in range(length)}

    if central:
        r_lo, r_hi = margin, h - margin - length
        c_lo, c_hi = max(margin, w // 4), min(w - margin - 1, (3 * w) // 4)
    else:
        r_lo, r_hi = margin, h - margin - length
        c_lo, c_hi = margin, w - margin - 1
    r = rng.randint(r_lo, r_hi)
    c = rng.randint(c_lo, c_hi)
    return {(r + i, c) for i in range(length)}


def sample_marker_layout(shape: tuple[int, int], rng: random.Random) -> list[dict] | None:
    kinds = ["horizontal", "vertical", "point"]
    while len(kinds) < rng.randint(3, 5):
        kinds.append(rng.choice(["horizontal", "vertical", "point"]))

    markers: list[dict] = []
    occupied_halo: set[tuple[int, int]] = set()
    for idx, kind in enumerate(kinds):
        placed = False
        for _ in range(200):
            cells = random_marker_cells(kind, shape, rng, central=idx < 3)
            if cells & occupied_halo:
                continue
            markers.append({"id": len(markers), "kind": kind, "cells": cells})
            occupied_halo |= halo(cells, shape)
            placed = True
            break
        if not placed:
            return None
    return markers


def seed_group_is_separated(shape: tuple[int, int], seeds: list[dict], traces: list[dict]) -> bool:
    occupied_halo: set[tuple[int, int]] = set()
    for idx, seed in enumerate(seeds):
        if seed["cells"] & occupied_halo:
            return False
        occupied_halo |= halo(seed["cells"], shape)
        for jdx, other in enumerate(seeds):
            if idx == jdx:
                continue
            if traces[idx]["path_set"] & other["cells"]:
                return False
    return True


def search_seed_set(shape: tuple[int, int], markers: list[dict], rng: random.Random) -> list[dict] | None:
    marker_by_cell = {}
    marker_cells: set[tuple[int, int]] = set()
    for marker in markers:
        for cell in marker["cells"]:
            marker_by_cell[cell] = marker["id"]
            marker_cells.add(cell)
    blocked = halo(marker_cells, shape)

    entries: list[tuple[dict, dict]] = []
    for cand in candidate_seed_starts(shape):
        if cand["cells"] & blocked:
            continue
        try:
            trace = trace_ray(shape, marker_by_cell, markers, cand)
        except AssertionError:
            continue
        if not trace["reflections"] or not trace["post_first"]:
            continue
        entries.append((cand, trace))

    if len(entries) < 2:
        return None

    order = list(range(len(entries)))
    rng.shuffle(order)
    want_three = rng.random() < 0.45

    for ii, i in enumerate(order):
        seed_i, trace_i = entries[i]
        for j in order[ii + 1 :]:
            seed_j, trace_j = entries[j]
            pair_seeds = [seed_i, seed_j]
            pair_traces = [trace_i, trace_j]
            overlap = trace_i["post_first"] & trace_j["post_first"]
            if not overlap or len(overlap) > 8:
                continue
            hit_types = set(trace_i["reflections"]) | set(trace_j["reflections"])
            if hit_types != {"horizontal", "vertical", "point"}:
                continue
            if not seed_group_is_separated(shape, pair_seeds, pair_traces):
                continue

            if want_three:
                union_post = trace_i["post_first"] | trace_j["post_first"]
                for k in order:
                    if k in (i, j):
                        continue
                    seed_k, trace_k = entries[k]
                    trio_seeds = pair_seeds + [seed_k]
                    trio_traces = pair_traces + [trace_k]
                    if not (trace_k["post_first"] & union_post):
                        continue
                    if not seed_group_is_separated(shape, trio_seeds, trio_traces):
                        continue
                    return trio_seeds
            return pair_seeds
    return None


def generate_input() -> np.ndarray:
    for attempt in range(MAX_GEN_ATTEMPTS):
        rng = random.Random(random.randrange(1 << 30) + attempt)
        shape = (rng.randint(18, 24), rng.randint(18, 24))
        markers = sample_marker_layout(shape, rng)
        if markers is None:
            continue
        seeds = search_seed_set(shape, markers, rng)
        if seeds is None:
            continue
        colors = rng.sample(RAY_COLORS, len(seeds))
        grid = build_input_grid(shape, markers, seeds, colors)
        if semantic_checks(grid):
            return grid
    raise RuntimeError("failed to generate a valid ray-reflection ARC input")
