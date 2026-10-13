from __future__ import annotations

from collections import deque

import numpy as np
import random


BLACK = 0
GREY = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _flood_fill_passable(
    passable: np.ndarray,
    starts: list[tuple[int, int]],
) -> set[tuple[int, int]]:
    h, w = passable.shape
    seen: set[tuple[int, int]] = set()
    q: deque[tuple[int, int]] = deque()
    for r, c in starts:
        if 0 <= r < h and 0 <= c < w and bool(passable[r, c]) and (r, c) not in seen:
            seen.add((r, c))
            q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and bool(passable[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return seen


def _candidate_gaps(grid: np.ndarray) -> dict[tuple[int, int], str]:
    h, w = grid.shape
    gaps: dict[tuple[int, int], str] = {}
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) != BLACK:
                continue
            lr = int(grid[r, c - 1]) == GREY and int(grid[r, c + 1]) == GREY
            ud = int(grid[r - 1, c]) == GREY and int(grid[r + 1, c]) == GREY
            if lr ^ ud:
                gaps[(r, c)] = "lr" if lr else "ud"
    return gaps


def _nearest_cardinal(seed: tuple[int, int], gap: tuple[int, int]) -> tuple[int, int] | None:
    dr = int(gap[0] - seed[0])
    dc = int(gap[1] - seed[1])
    if abs(dr) > abs(dc):
        return (1 if dr > 0 else -1, 0)
    if abs(dc) > abs(dr):
        return (0, 1 if dc > 0 else -1)
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow seed-colored rays out through fence gaps, with ray length set by each fenced cavity's area.

    Input structure:
    - The background is black.
    - Grey pixels form one-pixel-thick rectilinear fences around black cavities, but
      each fence has one or more single-cell gaps on straight fence segments.
    - Each fenced cavity contains exactly one colored seed pixel, and fences do not overlap.

    Rule:
    - Identify every gap cell and temporarily close those gaps to recover each seed's
      enclosed cavity.
    - For each seed, count the black cells in its recovered cavity, excluding the seed
      itself; call that value N.
    - For every gap belonging to that cavity, determine the outward direction away from
      the cavity interior and draw a straight ray of the seed's color starting at the
      gap cell. The ray extends for N cells unless it reaches another non-black cell first.

    Key insight:
    - The missing fence cell supplies the ray direction, while the cavity area behind the
      fence independently supplies the ray length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    seeds = [(int(r), int(c)) for r, c in np.argwhere((g != BLACK) & (g != GREY))]
    assert seeds, "expected at least one seed"

    gaps = _candidate_gaps(g)
    wall = g == GREY
    for r, c in gaps:
        wall[r, c] = True

    out = g.copy()
    seed_set = set(seeds)

    for sr, sc in sorted(seeds):
        seed_color = int(g[sr, sc])
        passable = np.zeros((h, w), dtype=bool)
        for r in range(h):
            for c in range(w):
                if bool(wall[r, c]):
                    continue
                if (r, c) in seed_set and (r, c) != (sr, sc):
                    continue
                passable[r, c] = True

        region = _flood_fill_passable(passable, [(sr, sc)])
        area = sum(int(g[r, c]) == BLACK for r, c in region)
        assert area > 0, "expected a nonempty cavity behind each seed"

        owned_gaps: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for (gr, gc), kind in gaps.items():
            outward: tuple[int, int] | None = None
            if kind == "lr":
                up_in = (gr - 1, gc) in region
                down_in = (gr + 1, gc) in region
                if up_in ^ down_in:
                    outward = (1, 0) if up_in else (-1, 0)
            else:
                left_in = (gr, gc - 1) in region
                right_in = (gr, gc + 1) in region
                if left_in ^ right_in:
                    outward = (0, 1) if left_in else (0, -1)

            if outward is None:
                continue

            seed_dir = _nearest_cardinal((sr, sc), (gr, gc))
            if seed_dir is not None:
                assert seed_dir == outward, "gap direction should agree with seed-to-gap direction"
            owned_gaps.append(((gr, gc), outward))

        assert owned_gaps, "expected each seed cavity to own at least one gap"

        for (gr, gc), (dr, dc) in owned_gaps:
            r, c = gr, gc
            steps = 0
            while steps < area and 0 <= r < h and 0 <= c < w:
                if int(g[r, c]) != BLACK:
                    break
                if int(out[r, c]) != BLACK:
                    break
                out[r, c] = seed_color
                steps += 1
                r += dr
                c += dc

    return out


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    assert c0 == c1, "segments must be orthogonal"
    step = 1 if r1 >= r0 else -1
    return [(r, c0) for r in range(r0, r1 + step, step)]


def _normalize_vertices(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in vertices)
    min_c = min(c for _, c in vertices)
    return [(r - min_r, c - min_c) for r, c in vertices]


def _boundary_from_vertices(
    vertices: list[tuple[int, int]],
) -> tuple[set[tuple[int, int]], list[list[tuple[int, int]]], tuple[int, int]]:
    boundary: set[tuple[int, int]] = set()
    segments: list[list[tuple[int, int]]] = []
    for i in range(len(vertices) - 1):
        cells = _segment_cells(vertices[i], vertices[i + 1])
        boundary.update(cells)
        segments.append(cells)

    max_r = max(r for r, _ in boundary)
    max_c = max(c for _, c in boundary)
    return boundary, segments, (max_r + 1, max_c + 1)


def _interior_from_boundary(
    boundary: set[tuple[int, int]],
    shape: tuple[int, int],
) -> set[tuple[int, int]]:
    h, w = shape
    mask = np.zeros((h + 2, w + 2), dtype=bool)
    for r, c in boundary:
        mask[r + 1, c + 1] = True

    outside = _flood_fill_passable(~mask, [(0, 0)])
    interior: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            pr, pc = r + 1, c + 1
            if (pr, pc) not in outside and not bool(mask[pr, pc]):
                interior.add((r, c))
    return interior


def _shape_gap_options(
    boundary: set[tuple[int, int]],
    interior: set[tuple[int, int]],
    segments: list[list[tuple[int, int]]],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    options: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for seg in segments:
        if len(seg) < 3:
            continue
        inner_cells = seg[1:-1]
        horizontal = seg[0][0] == seg[-1][0]
        for r, c in inner_cells:
            if horizontal:
                above = (r - 1, c) in interior
                below = (r + 1, c) in interior
                if above ^ below:
                    outward = (1, 0) if above else (-1, 0)
                    options.append(((r, c), outward))
            else:
                left = (r, c - 1) in interior
                right = (r, c + 1) in interior
                if left ^ right:
                    outward = (0, 1) if left else (0, -1)
                    options.append(((r, c), outward))
    dedup: dict[tuple[int, int], tuple[int, int]] = {}
    for cell, outward in options:
        dedup[cell] = outward
    return sorted(dedup.items())


def _sample_shape() -> dict[str, object] | None:
    kind = random.choice(["rect", "bulge"])
    base_h = random.randint(5, 8)
    base_w = random.randint(5, 8)

    if kind == "rect":
        vertices = [(0, 0), (0, base_w - 1), (base_h - 1, base_w - 1), (base_h - 1, 0), (0, 0)]
    else:
        side = random.choice(["right", "left", "top", "bottom"])
        ext = random.randint(1, 3)
        if side in ("right", "left"):
            if base_h < 6:
                return None
            a = random.randint(1, base_h - 4)
            b = random.randint(a + 2, base_h - 2)
            if side == "right":
                vertices = [
                    (0, 0),
                    (0, base_w - 1),
                    (a, base_w - 1),
                    (a, base_w - 1 + ext),
                    (b, base_w - 1 + ext),
                    (b, base_w - 1),
                    (base_h - 1, base_w - 1),
                    (base_h - 1, 0),
                    (0, 0),
                ]
            else:
                vertices = [
                    (0, 0),
                    (0, base_w - 1),
                    (base_h - 1, base_w - 1),
                    (base_h - 1, 0),
                    (b, 0),
                    (b, -ext),
                    (a, -ext),
                    (a, 0),
                    (0, 0),
                ]
        else:
            if base_w < 6:
                return None
            a = random.randint(1, base_w - 4)
            b = random.randint(a + 2, base_w - 2)
            if side == "top":
                vertices = [
                    (0, 0),
                    (0, a),
                    (-ext, a),
                    (-ext, b),
                    (0, b),
                    (0, base_w - 1),
                    (base_h - 1, base_w - 1),
                    (base_h - 1, 0),
                    (0, 0),
                ]
            else:
                vertices = [
                    (0, 0),
                    (0, base_w - 1),
                    (base_h - 1, base_w - 1),
                    (base_h - 1, b),
                    (base_h - 1 + ext, b),
                    (base_h - 1 + ext, a),
                    (base_h - 1, a),
                    (base_h - 1, 0),
                    (0, 0),
                ]

    vertices = _normalize_vertices(vertices)
    boundary, segments, shape = _boundary_from_vertices(vertices)
    interior = _interior_from_boundary(boundary, shape)
    if len(interior) < 2:
        return None

    gap_options = _shape_gap_options(boundary, interior, segments)
    if len(gap_options) < 2:
        return None

    interior_list = sorted(interior)
    boundary_list = sorted(boundary)

    scores: list[tuple[int, tuple[int, int]]] = []
    for cell in interior_list:
        dist = min(abs(cell[0] - br) + abs(cell[1] - bc) for br, bc in boundary_list)
        scores.append((dist, cell))
    max_dist = max(score for score, _ in scores)
    seed_choices = [cell for score, cell in scores if score == max_dist]
    seed = random.choice(seed_choices)

    valid_gap_options: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for cell, outward in gap_options:
        nearest = _nearest_cardinal(seed, cell)
        if nearest is not None and nearest == outward:
            valid_gap_options.append((cell, outward))
    if len(valid_gap_options) < 1:
        return None

    return {
        "boundary_local": boundary,
        "interior_local": interior,
        "gap_options_local": valid_gap_options,
        "seed_local": seed,
        "bbox_shape": shape,
        "area": len(interior) - 1,
        "kind": kind,
    }


def _translate_cells(cells: set[tuple[int, int]] | list[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells}


def _place_objects(
    canvas_shape: tuple[int, int],
    specs: list[dict[str, object]],
) -> list[dict[str, object]] | None:
    h, w = canvas_shape
    placed: list[dict[str, object]] = []
    blocked_pref = random.choice([False, True]) and len(specs) >= 2

    for idx, spec in enumerate(specs):
        sh, sw = spec["bbox_shape"]
        candidates = [(r, c) for r in range(1, h - sh) for c in range(1, w - sw)]
        random.shuffle(candidates)
        placed_one = None
        for r0, c0 in candidates:
            bbox = (r0, c0, r0 + sh - 1, c0 + sw - 1)
            ok = True
            for prev in placed:
                pr0, pc0, pr1, pc1 = prev["bbox"]
                if not (bbox[2] < pr0 - 1 or bbox[0] > pr1 + 1 or bbox[3] < pc0 - 1 or bbox[1] > pc1 + 1):
                    ok = False
                    break
            if not ok:
                continue

            if blocked_pref and idx == 1:
                prev = placed[0]
                sr, sc = prev["seed_global"]
                overlap_row = bbox[0] <= sr <= bbox[2]
                overlap_col = bbox[1] <= sc <= bbox[3]
                near = abs(c0 - prev["bbox"][3]) <= 8 or abs(r0 - prev["bbox"][2]) <= 8
                if not ((overlap_row or overlap_col) and near):
                    continue

            placed_one = {
                "bbox": bbox,
                "boundary_global": _translate_cells(spec["boundary_local"], r0, c0),
                "interior_global": _translate_cells(spec["interior_local"], r0, c0),
                "gap_options_global": [((gr + r0, gc + c0), outward) for (gr, gc), outward in spec["gap_options_local"]],
                "seed_global": (spec["seed_local"][0] + r0, spec["seed_local"][1] + c0),
                "area": int(spec["area"]),
                "kind": spec["kind"],
            }
            break
        if placed_one is None:
            return None
        placed.append(placed_one)

    return placed


def _ray_clearance(
    closed_grid: np.ndarray,
    gap: tuple[int, int],
    outward: tuple[int, int],
) -> tuple[int, str]:
    h, w = closed_grid.shape
    r, c = gap
    dr, dc = outward
    steps = 0
    while 0 <= r < h and 0 <= c < w:
        if (r, c) != gap and int(closed_grid[r, c]) != BLACK:
            return steps, "obstacle"
        steps += 1
        r += dr
        c += dc
    return steps, "edge"


def _choose_gaps(
    closed_grid: np.ndarray,
    objects: list[dict[str, object]],
) -> list[list[tuple[int, int]]] | None:
    blocked_candidates: list[tuple[int, tuple[int, int]]] = []
    free_candidates: list[tuple[int, tuple[int, int]]] = []

    option_info: list[list[tuple[tuple[int, int], tuple[int, int], bool]]] = []
    for idx, obj in enumerate(objects):
        area = int(obj["area"])
        infos: list[tuple[tuple[int, int], tuple[int, int], bool]] = []
        for gap, outward in obj["gap_options_global"]:
            clearance, reason = _ray_clearance(closed_grid, gap, outward)
            if clearance >= area:
                infos.append((gap, outward, False))
                free_candidates.append((idx, gap))
            elif reason == "obstacle" and clearance >= 2:
                infos.append((gap, outward, True))
                blocked_candidates.append((idx, gap))
        if not infos:
            return None
        option_info.append(infos)

    need_blocked = random.choice([False, True]) and bool(blocked_candidates)
    chosen: list[list[tuple[int, int]]] = [[] for _ in objects]

    if need_blocked:
        idx, gap = random.choice(blocked_candidates)
        chosen[idx].append(gap)

    for idx, infos in enumerate(option_info):
        target_count = random.choice([1, 2]) if len(infos) >= 2 else 1
        if chosen[idx]:
            target_count = max(target_count, 1)

        random.shuffle(infos)
        for gap, _outward, _blocked in infos:
            if gap in chosen[idx]:
                continue
            chosen[idx].append(gap)
            if len(chosen[idx]) >= target_count:
                break

        if len(chosen[idx]) < target_count:
            return None

    all_gaps = [gap for group in chosen for gap in group]
    if len(set(all_gaps)) != len(all_gaps):
        return None

    return chosen


def _render_closed_grid(
    shape: tuple[int, int],
    objects: list[dict[str, object]],
) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for idx, obj in enumerate(objects):
        color = SEED_COLORS[idx]
        for r, c in obj["boundary_global"]:
            grid[r, c] = GREY
        sr, sc = obj["seed_global"]
        grid[sr, sc] = color
    return grid


def _validate_input(inp: np.ndarray, expected_closed: np.ndarray) -> bool:
    try:
        out = main(inp)
    except Exception:
        return False
    if out.shape != inp.shape:
        return False
    if np.array_equal(inp, out):
        return False

    # Every removed fence cell should be a detectable gap.
    actual_gaps = _candidate_gaps(inp)
    removed = set(map(tuple, np.argwhere((expected_closed == GREY) & (inp == BLACK))))
    if removed != set(actual_gaps):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        h = random.randint(18, 30)
        w = random.randint(18, 30)
        object_count = random.choice([1, 2, 2, 3])

        specs: list[dict[str, object]] = []
        for _obj in range(object_count):
            spec = _sample_shape()
            if spec is None:
                break
            specs.append(spec)
        if len(specs) != object_count:
            continue

        placed = _place_objects((h, w), specs)
        if placed is None:
            continue

        closed = _render_closed_grid((h, w), placed)
        chosen_gaps = _choose_gaps(closed, placed)
        if chosen_gaps is None:
            continue

        inp = closed.copy()
        for gaps in chosen_gaps:
            for r, c in gaps:
                inp[r, c] = BLACK

        if not _validate_input(inp, closed):
            continue
        return inp

    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
