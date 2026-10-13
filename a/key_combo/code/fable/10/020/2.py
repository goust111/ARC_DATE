from __future__ import annotations

from collections import Counter, deque
import random
from typing import Iterable

import numpy as np


GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
BLOCK_SIZES = [(2, 2), (2, 3), (3, 2), (1, 1)]
LAUNCH_KINDS = ("row", "col", "corner")
GEN_RETRY_LIMIT = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < int(shape[0]) and 0 <= c < int(shape[1])


def _neighbors(r: int, c: int, connectivity: int) -> Iterable[tuple[int, int]]:
    if connectivity == 4:
        yield r - 1, c
        yield r + 1, c
        yield r, c - 1
        yield r, c + 1
        return

    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    seen = np.zeros(mask.shape, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    h, w = mask.shape

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors(rr, cc, connectivity):
                    if _in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(cells)
    return out


def _forward_run(shape: tuple[int, int], cell: tuple[int, int], direction: tuple[int, int]) -> int:
    h, w = shape
    r, c = cell
    dr, dc = direction
    row_room = r if dr < 0 else h - 1 - r
    col_room = c if dc < 0 else w - 1 - c
    return int(min(row_room, col_room))


def _absorber_cells(grid: np.ndarray, bg: int) -> tuple[set[tuple[int, int]], dict[tuple[int, int], int]]:
    cells = set()
    colors: dict[tuple[int, int], int] = {}
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            val = int(grid[r, c])
            if val not in {bg, GREY}:
                cells.add((r, c))
                colors[(r, c)] = val
    return cells, colors


def _trace_to_absorber(
    grid: np.ndarray,
    bg: int,
    bar_cells: set[tuple[int, int]],
    outer: tuple[int, int],
    direction: tuple[int, int],
) -> dict[str, object] | None:
    absorber_set, absorber_color = _absorber_cells(grid, bg)
    h, w = grid.shape
    r, c = outer
    dr, dc = direction
    seen_states = {(r, c, dr, dc)}
    path: list[tuple[int, int]] = []
    bounce_events = 0

    for _ in range(8 * h * w):
        bounced = False
        if not (0 <= r + dr < h):
            dr *= -1
            bounced = True
        if not (0 <= c + dc < w):
            dc *= -1
            bounced = True
        if bounced:
            bounce_events += 1

        nr, nc = r + dr, c + dc
        assert _in_bounds(grid.shape, nr, nc)

        if (nr, nc) in absorber_set:
            return {
                "path": path,
                "color": int(absorber_color[(nr, nc)]),
                "endpoint": (nr, nc),
                "bounces": bounce_events,
            }

        val = int(grid[nr, nc])
        if val == GREY and (nr, nc) not in bar_cells:
            return None
        if val != bg:
            return None

        path.append((nr, nc))
        r, c = nr, nc
        state = (r, c, dr, dc)
        if state in seen_states:
            return None
        seen_states.add(state)

    return None


def _pick_bar_orientation(
    grid: np.ndarray,
    bg: int,
    bar: list[tuple[int, int]],
) -> dict[str, object]:
    assert len(bar) == 2
    a, b = sorted(bar)
    assert abs(a[0] - b[0]) == 1 and abs(a[1] - b[1]) == 1

    candidates = []
    for inner, outer in ((a, b), (b, a)):
        direction = (outer[0] - inner[0], outer[1] - inner[1])
        trace = _trace_to_absorber(grid, bg, set(bar), outer, direction)
        candidates.append(
            {
                "inner": inner,
                "outer": outer,
                "dir": direction,
                "run": _forward_run(grid.shape, outer, direction),
                "trace": trace,
            }
        )

    valid = [cand for cand in candidates if cand["trace"] is not None]
    assert valid, "each bar must define a ray that eventually hits an absorber"
    if len(valid) == 1:
        return valid[0]

    valid.sort(key=lambda item: int(item["run"]))
    if int(valid[0]["run"]) != int(valid[1]["run"]):
        return valid[0]
    raise AssertionError("bar orientation is ambiguous")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint each grey diagonal launch bar and its reflected diagonal ray with the color of the absorber block it eventually hits.

    Input structure:
    - The background is dominant.
    - Each start object is a grey 8-connected component of exactly two diagonally adjacent cells.
    - Other foreground objects are solid monochrome absorber blocks in distinct colors.

    Rule:
    - Each grey bar emits one diagonal ray: start from the bar's outer cell and continue in the diagonal direction defined by the two cells.
    - When the ray would leave the canvas, reflect it off the touched border like a mirror and keep moving diagonally; when it reaches an absorber block, stop immediately.
    - Recolor every background cell visited before absorption, and also recolor both grey start-bar cells, using the color of the absorber block that was hit; keep all absorber blocks unchanged.

    Key insight:
    - The launch bar itself does not tell you its final color: you must simulate the whole reflected future path first, because the endpoint absorber determines the color of the entire trajectory.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)

    grey_components = _components(g == GREY, connectivity=8)
    assert grey_components, "expected at least one grey start bar"
    for comp in grey_components:
        assert len(comp) == 2, "every grey component must be a 2-cell bar"
        (r0, c0), (r1, c1) = comp
        assert abs(r0 - r1) == 1 and abs(c0 - c1) == 1, "grey bar cells must be diagonally adjacent"

    out = g.copy()
    for bar in grey_components:
        picked = _pick_bar_orientation(g, bg, bar)
        trace = picked["trace"]
        assert trace is not None
        color = int(trace["color"])
        for r, c in bar:
            out[r, c] = color
        for r, c in trace["path"]:
            assert int(g[r, c]) == bg
            out[r, c] = color
    return out


def _path_without_absorbers(
    shape: tuple[int, int],
    outer: tuple[int, int],
    direction: tuple[int, int],
) -> list[dict[str, object]]:
    h, w = shape
    r, c = outer
    dr, dc = direction
    seen_states = {(r, c, dr, dc)}
    seen_cells: set[tuple[int, int]] = set()
    bounce_events = 0
    path: list[dict[str, object]] = []

    for _ in range(8 * h * w):
        bounced = False
        if not (0 <= r + dr < h):
            dr *= -1
            bounced = True
        if not (0 <= c + dc < w):
            dc *= -1
            bounced = True
        if bounced:
            bounce_events += 1

        r += dr
        c += dc
        if (r, c) in seen_cells:
            break
        seen_cells.add((r, c))
        path.append({"cell": (r, c), "dir": (dr, dc), "bounces": bounce_events})
        state = (r, c, dr, dc)
        if state in seen_states:
            break
        seen_states.add(state)
    return path


def _margin_cells(shape: tuple[int, int], cells: Iterable[tuple[int, int]], margin: int = 1) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-margin, margin + 1):
            for dc in range(-margin, margin + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _block_cells_for_endpoint(
    shape: tuple[int, int],
    endpoint: tuple[int, int],
    direction: tuple[int, int],
    size: tuple[int, int],
) -> list[tuple[int, int]] | None:
    h, w = size
    r, c = endpoint
    dr, dc = direction
    r0 = r if dr > 0 else r - h + 1
    c0 = c if dc > 0 else c - w + 1
    cells = [(rr, cc) for rr in range(r0, r0 + h) for cc in range(c0, c0 + w)]
    if not cells:
        return None
    if not all(_in_bounds(shape, rr, cc) for rr, cc in cells):
        return None
    prev = (r - dr, c - dc)
    if prev in cells:
        return None
    return cells


def _build_expected(inp: np.ndarray, rays: list[dict[str, object]]) -> np.ndarray:
    out = inp.copy()
    for ray in rays:
        color = int(ray["color"])
        for r, c in ray["bar_cells"]:
            out[r, c] = color
        for r, c in ray["path"]:
            out[r, c] = color
    return out


def _sample_outer(shape: tuple[int, int], direction: tuple[int, int], kind: str) -> tuple[int, int]:
    h, w = shape
    dr, dc = direction

    max_row = max(3, min(5, h - 3))
    max_col = max(3, min(5, w - 3))

    if kind == "corner":
        dist = random.randint(1, min(max_row, max_col) - 1)
        row_dist = dist
        col_dist = dist
    elif kind == "row":
        row_dist = random.randint(1, max_row - 1)
        col_dist = random.randint(min(row_dist + 2, max_col - 1), max_col - 1)
    else:
        col_dist = random.randint(1, max_col - 1)
        row_dist = random.randint(min(col_dist + 2, max_row - 1), max_row - 1)

    r = row_dist if dr < 0 else h - 1 - row_dist
    c = col_dist if dc < 0 else w - 1 - col_dist
    return int(r), int(c)


def _try_place_ray(
    shape: tuple[int, int],
    required_bounces: int,
    color: int,
    exact_blocked: set[tuple[int, int]],
    object_margin_blocked: set[tuple[int, int]],
) -> dict[str, object] | None:
    for _ in range(120):
        direction = random.choice(DIRS)
        launch_kind = random.choice(LAUNCH_KINDS)
        outer = _sample_outer(shape, direction, launch_kind)
        inner = (outer[0] - direction[0], outer[1] - direction[1])
        if not _in_bounds(shape, *inner):
            continue

        bar_cells = [inner, outer]
        if any(cell in exact_blocked or cell in object_margin_blocked for cell in bar_cells):
            continue

        opposite_run = _forward_run(shape, inner, (-direction[0], -direction[1]))
        if _forward_run(shape, outer, direction) >= opposite_run:
            continue

        raw_path = _path_without_absorbers(shape, outer, direction)
        if len(raw_path) < 6:
            continue

        prefix: list[tuple[int, int]] = []
        candidates: list[dict[str, object]] = []
        for step_idx, step in enumerate(raw_path, start=1):
            endpoint = step["cell"]
            if endpoint in exact_blocked:
                break
            if step_idx < 5 or int(step["bounces"]) < required_bounces:
                prefix.append(endpoint)
                continue
            if any(cell in exact_blocked for cell in prefix):
                break

            sizes = BLOCK_SIZES[:]
            random.shuffle(sizes)
            for size in sizes:
                block_cells = _block_cells_for_endpoint(shape, endpoint, step["dir"], size)
                if block_cells is None:
                    continue
                if any(cell in exact_blocked for cell in block_cells):
                    continue
                if any(cell in prefix for cell in block_cells):
                    continue
                if any(cell in object_margin_blocked for cell in block_cells):
                    continue
                candidates.append(
                    {
                        "bar_cells": tuple(bar_cells),
                        "outer": outer,
                        "dir": direction,
                        "path": tuple(prefix),
                        "endpoint": endpoint,
                        "color": color,
                        "absorber_cells": tuple(block_cells),
                        "bounces": int(step["bounces"]),
                        "launch_kind": launch_kind,
                    }
                )
                break
            prefix.append(endpoint)

        if candidates:
            weighted = [cand for cand in candidates if cand["bounces"] == required_bounces]
            pool = weighted if weighted else candidates
            return random.choice(pool)
    return None


def _place_extra_absorber(
    shape: tuple[int, int],
    color: int,
    rays: list[dict[str, object]],
    exact_blocked: set[tuple[int, int]],
    object_margin_blocked: set[tuple[int, int]],
) -> dict[str, object] | None:
    bounced_rays = [ray for ray in rays if int(ray["bounces"]) >= 1]
    if not bounced_rays:
        return None

    random.shuffle(bounced_rays)
    h, w = shape
    for ray in bounced_rays:
        outer = ray["outer"]
        true_dist = abs(int(ray["endpoint"][0]) - outer[0]) + abs(int(ray["endpoint"][1]) - outer[1])
        sizes = BLOCK_SIZES[:]
        random.shuffle(sizes)
        cells = [(r, c) for r in range(h) for c in range(w)]
        random.shuffle(cells)
        for endpoint in cells:
            close_dist = abs(endpoint[0] - outer[0]) + abs(endpoint[1] - outer[1])
            if close_dist >= true_dist:
                continue
            for direction in DIRS:
                for size in sizes:
                    block_cells = _block_cells_for_endpoint(shape, endpoint, direction, size)
                    if block_cells is None:
                        continue
                    if any(cell in exact_blocked for cell in block_cells):
                        continue
                    if any(cell in object_margin_blocked for cell in block_cells):
                        continue
                    return {"color": color, "cells": tuple(block_cells), "near_ray": ray}
    return None


def _validate_generated(
    inp: np.ndarray,
    rays: list[dict[str, object]],
    extras: list[dict[str, object]],
) -> bool:
    expected = _build_expected(inp, rays)
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    all_absorber_cells = {cell for extra in extras for cell in extra["cells"]}
    all_absorber_cells |= {cell for ray in rays for cell in ray["absorber_cells"]}
    if any(cell in all_absorber_cells for ray in rays for cell in ray["path"]):
        return False

    path_cells: set[tuple[int, int]] = set()
    for ray in rays:
        for cell in ray["path"]:
            if cell in path_cells:
                return False
            path_cells.add(cell)
        picked = _pick_bar_orientation(inp, _get_bg(inp), list(ray["bar_cells"]))
        if picked["outer"] != ray["outer"] or picked["dir"] != ray["dir"]:
            return False

    bounced = [ray for ray in rays if int(ray["bounces"]) >= 1]
    if not bounced:
        return False

    closer_extra_exists = False
    for extra in extras:
        ray = extra["near_ray"]
        outer = ray["outer"]
        true_dist = abs(int(ray["endpoint"][0]) - outer[0]) + abs(int(ray["endpoint"][1]) - outer[1])
        extra_dist = min(abs(r - outer[0]) + abs(c - outer[1]) for r, c in extra["cells"])
        if extra_dist < true_dist:
            closer_extra_exists = True
            break
    return closer_extra_exists


def _gen_attempt() -> tuple[np.ndarray, list[dict[str, object]], list[dict[str, object]]] | None:
    h = random.randint(14, 22)
    w = random.randint(14, 22)
    shape = (h, w)

    ray_count = random.choice([2, 3, 3, 4])
    extra_count = 1 if ray_count <= 3 else 2
    total_colors = ray_count + extra_count
    if total_colors > len(PALETTE):
        return None
    colors = random.sample(PALETTE, total_colors)

    if ray_count == 2:
        bounce_reqs = [1, 2]
    elif ray_count == 3:
        bounce_reqs = [1, 1, 2]
    else:
        bounce_reqs = [1, 1, 2, 2]
    random.shuffle(bounce_reqs)

    rays: list[dict[str, object]] = []
    exact_blocked: set[tuple[int, int]] = set()
    object_margin_blocked: set[tuple[int, int]] = set()

    for idx in range(ray_count):
        ray = _try_place_ray(shape, bounce_reqs[idx], colors[idx], exact_blocked, object_margin_blocked)
        if ray is None:
            return None
        rays.append(ray)
        ray_cells = set(ray["bar_cells"]) | set(ray["path"]) | set(ray["absorber_cells"])
        exact_blocked |= ray_cells
        object_margin_blocked |= _margin_cells(shape, set(ray["bar_cells"]) | set(ray["absorber_cells"]), margin=1)

    extras: list[dict[str, object]] = []
    for idx in range(extra_count):
        extra = _place_extra_absorber(shape, colors[ray_count + idx], rays, exact_blocked, object_margin_blocked)
        if extra is None:
            return None
        extras.append(extra)
        exact_blocked |= set(extra["cells"])
        object_margin_blocked |= _margin_cells(shape, set(extra["cells"]), margin=1)

    grid = np.zeros(shape, dtype=int)
    for ray in rays:
        for r, c in ray["bar_cells"]:
            grid[r, c] = GREY
        for r, c in ray["absorber_cells"]:
            grid[r, c] = int(ray["color"])
    for extra in extras:
        for r, c in extra["cells"]:
            grid[r, c] = int(extra["color"])

    if not _validate_generated(grid, rays, extras):
        return None
    return grid, rays, extras


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        built = _gen_attempt()
        if built is not None:
            grid, _rays, _extras = built
            return grid
    raise RuntimeError("failed to generate a valid reflected-ray ARC input")
