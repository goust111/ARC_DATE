from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
AXIS = 5
CONTAINER = 7
MARKER_COLORS = (1, 2, 3, 4)

_GEN_COUNTER = 0


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((int(rr), int(cc)))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(np.array(pts, dtype=int))
    return comps


def _find_axis_row(g: np.ndarray) -> int:
    rows = [r for r in range(g.shape[0]) if bool(np.all(g[r, :] == AXIS))]
    assert len(rows) == 1, f"expected exactly one full-width axis row, got {rows}"
    return int(rows[0])


def _find_markers(g: np.ndarray, axis_r: int) -> list[tuple[int, int, int]]:
    markers: list[tuple[int, int, int]] = []
    cols_seen: set[int] = set()
    for r in range(axis_r):
        for c in range(g.shape[1]):
            color = int(g[r, c])
            if color in (BG, AXIS, CONTAINER):
                continue
            markers.append((color, int(r), int(c)))
            assert c not in cols_seen, "marker columns must be distinct"
            cols_seen.add(int(c))
    assert 3 <= len(markers) <= 4, f"expected 3-4 markers, got {len(markers)}"
    return markers


def _find_containers(g: np.ndarray, axis_r: int) -> list[dict]:
    mask = np.zeros_like(g, dtype=bool)
    mask[axis_r + 1 :, :] = g[axis_r + 1 :, :] == CONTAINER
    comps = _components_mask(mask)
    containers: list[dict] = []
    for coords in comps:
        r0, c0, r1, c1 = _bbox_coords(coords)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        assert w == 3, f"container width must be 3, got {w}"
        assert h in (2, 3), f"container height must be 2 or 3, got {h}"
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert bool(np.all(patch[:, 0] == CONTAINER))
        assert bool(np.all(patch[:, 2] == CONTAINER))
        assert bool(np.all(patch[-1, :] == CONTAINER))
        assert bool(np.all(patch[:-1, 1] == BG))
        interior = [(rr, c0 + 1) for rr in range(r0, r1)]
        containers.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "mid_col": int(c0 + 1),
                "interior": [(int(rr), int(cc)) for rr, cc in interior],
                "interior_rows": set(range(int(r0), int(r1))),
                "wall_cells": {(int(rr), int(c0)) for rr in range(r0, r1 + 1)}
                | {(int(rr), int(c1)) for rr in range(r0, r1 + 1)}
                | {(int(r1), int(cc)) for cc in range(c0, c1 + 1)},
            }
        )
    containers.sort(key=lambda item: (item["mid_col"], item["bbox"][0]))
    assert 2 <= len(containers) <= 3, f"expected 2-3 containers, got {len(containers)}"
    return containers


def _container_hit(containers: list[dict], r: int, c: int) -> int | None:
    matches = [
        idx
        for idx, container in enumerate(containers)
        if int(c) == int(container["mid_col"]) and int(r) in container["interior_rows"]
    ]
    assert len(matches) <= 1, "a reflected point cannot be inside multiple containers"
    return None if not matches else int(matches[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect markers across a horizontal axis, but turn any reflection that lands inside a U-container into a full container fill.

    Input structure:
    - Black background with one full-width horizontal grey axis.
    - Several single-cell colored markers sit above the axis in distinct columns.
    - Two or three orange U-shaped containers sit below the axis; each is 3 cells wide and 2 or 3 cells tall.

    Rule:
    - Reflect every marker across the grey axis, keeping its column and mirroring its row distance.
    - If the reflected position falls in a container's interior cell range, erase the source marker and fill that container's entire interior with the marker color instead of placing the reflected pixel.
    - Otherwise erase the source marker and place the reflected pixel at the mirrored location, even if that location lies on a container wall.
    - Leave the grey axis unchanged.

    Key insight:
    - Reflection alone is not the full rule: the mirrored landing coordinate is tested against container interiors, and that spatial coincidence switches the effect from "place one pixel" to "fill one container."
    """
    g = np.asarray(grid, dtype=int)
    axis_r = _find_axis_row(g)
    markers = _find_markers(g, axis_r)
    containers = _find_containers(g, axis_r)

    out = g.copy()
    used_containers: set[int] = set()
    placed_targets: set[tuple[int, int]] = set()
    for color, r, c in markers:
        out[r, c] = BG
        rr = int(2 * axis_r - r)
        assert 0 <= rr < g.shape[0], "reflected row must stay in bounds"
        hit = _container_hit(containers, rr, c)
        if hit is not None:
            assert hit not in used_containers, "at most one marker may be captured by a container"
            used_containers.add(int(hit))
            for ir, ic in containers[hit]["interior"]:
                out[ir, ic] = int(color)
            continue

        target = (int(rr), int(c))
        assert target not in placed_targets, "uncaptured reflected pixels must be distinct"
        placed_targets.add(target)
        out[rr, c] = int(color)

    return out


def _sample_spaced_columns(start: int, stop: int, count: int, min_gap: int) -> list[int] | None:
    cols = list(range(start, stop + 1))
    random.shuffle(cols)
    chosen: list[int] = []
    for c in cols:
        if all(abs(c - prev) >= min_gap for prev in chosen):
            chosen.append(int(c))
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _stamp_container(g: np.ndarray, top: int, mid: int, height: int) -> dict:
    left = int(mid - 1)
    right = int(mid + 1)
    bottom = int(top + height - 1)
    g[top : bottom + 1, left] = CONTAINER
    g[top : bottom + 1, right] = CONTAINER
    g[bottom, left : right + 1] = CONTAINER
    return {
        "top": int(top),
        "bottom": int(bottom),
        "mid": int(mid),
        "left": int(left),
        "right": int(right),
        "height": int(height),
        "interior_rows": list(range(int(top), int(bottom))),
    }


def _reflect_row(axis_r: int, delta: int) -> int:
    return int(axis_r + delta)


def _marker_row(axis_r: int, delta: int) -> int:
    return int(axis_r - delta)


def _free_column(w: int, forbidden: set[int]) -> int | None:
    candidates = [c for c in range(1, w - 1) if c not in forbidden]
    return None if not candidates else int(random.choice(candidates))


def _scenario_specs(containers: list[dict], axis_r: int, w: int, mode: int) -> list[dict] | None:
    specs: list[dict] = []
    used_cols: set[int] = set()

    capture_candidates = [
        (idx, delta)
        for idx, container in enumerate(containers)
        for delta in range(1, 4)
        if _reflect_row(axis_r, delta) in container["interior_rows"]
    ]
    if not capture_candidates:
        return None
    cap_idx, cap_delta = random.choice(capture_candidates)
    cap = containers[cap_idx]
    specs.append(
        {
            "kind": "capture",
            "delta": int(cap_delta),
            "col": int(cap["mid"]),
            "container": int(cap_idx),
        }
    )
    used_cols.add(int(cap["mid"]))

    miss_candidates = [
        (idx, delta)
        for idx, container in enumerate(containers)
        for delta in range(1, 4)
        if _reflect_row(axis_r, delta) not in container["interior_rows"]
    ]
    wall_candidates = [
        (idx, side, delta)
        for idx, container in enumerate(containers)
        for side in ("left", "right")
        for delta in range(1, 4)
        if _reflect_row(axis_r, delta) in range(int(container["top"]), int(container["bottom"]) + 1)
    ]

    free_col = _free_column(w, used_cols | {c["left"] for c in containers} | {c["right"] for c in containers})
    if free_col is None:
        return None
    free_delta = random.randint(1, 3)
    free_rr = _reflect_row(axis_r, free_delta)
    for _ in range(10):
        if all(
            free_col != container["mid"] or free_rr not in container["interior_rows"]
            for container in containers
        ):
            break
        free_delta = random.randint(1, 3)
        free_rr = _reflect_row(axis_r, free_delta)
    else:
        return None
    specs.append({"kind": "free", "delta": int(free_delta), "col": int(free_col)})
    used_cols.add(int(free_col))

    if mode == 0:
        miss_pool = [item for item in miss_candidates if int(containers[item[0]]["mid"]) not in used_cols]
        if not miss_pool or not wall_candidates:
            return None
        miss_idx, miss_delta = random.choice(miss_pool)
        miss = containers[miss_idx]
        specs.append(
            {
                "kind": "same_col_miss",
                "delta": int(miss_delta),
                "col": int(miss["mid"]),
                "container": int(miss_idx),
            }
        )
        used_cols.add(int(miss["mid"]))

        wall_pool = [item for item in wall_candidates if int(containers[item[0]][item[1]]) not in used_cols]
        if not wall_pool:
            return None
        wall_idx, side, wall_delta = random.choice(wall_pool)
        specs.append(
            {
                "kind": "wall",
                "delta": int(wall_delta),
                "col": int(containers[wall_idx][side]),
                "container": int(wall_idx),
            }
        )
    elif mode == 1:
        miss_pool = [item for item in miss_candidates if int(containers[item[0]]["mid"]) not in used_cols]
        if not miss_pool:
            return None
        miss_idx, miss_delta = random.choice(miss_pool)
        miss = containers[miss_idx]
        specs.append(
            {
                "kind": "same_col_miss",
                "delta": int(miss_delta),
                "col": int(miss["mid"]),
                "container": int(miss_idx),
            }
        )
    else:
        second_cap_pool = [
            (idx, delta)
            for idx, delta in capture_candidates
            if idx != cap_idx and containers[idx]["mid"] not in used_cols
        ]
        if not second_cap_pool or not wall_candidates:
            return None
        cap2_idx, cap2_delta = random.choice(second_cap_pool)
        cap2 = containers[cap2_idx]
        specs.append(
            {
                "kind": "capture",
                "delta": int(cap2_delta),
                "col": int(cap2["mid"]),
                "container": int(cap2_idx),
            }
        )
        used_cols.add(int(cap2["mid"]))

        wall_pool = [item for item in wall_candidates if int(containers[item[0]][item[1]]) not in used_cols]
        if not wall_pool:
            return None
        wall_idx, side, wall_delta = random.choice(wall_pool)
        specs.append(
            {
                "kind": "wall",
                "delta": int(wall_delta),
                "col": int(containers[wall_idx][side]),
                "container": int(wall_idx),
            }
        )

    return specs


def _analyze_grid(g: np.ndarray) -> dict:
    axis_r = _find_axis_row(g)
    markers = _find_markers(g, axis_r)
    containers = _find_containers(g, axis_r)

    captures = 0
    uncaptured = 0
    same_col_miss = 0
    wall_hits = 0
    hit_ids: set[int] = set()
    for _color, r, c in markers:
        rr = int(2 * axis_r - r)
        hit = _container_hit(containers, rr, c)
        if hit is not None:
            captures += 1
            hit_ids.add(int(hit))
            continue
        uncaptured += 1
        for container in containers:
            if int(c) == int(container["mid_col"]) and rr not in container["interior_rows"]:
                same_col_miss += 1
            if (int(rr), int(c)) in container["wall_cells"]:
                wall_hits += 1

    return {
        "captures": int(captures),
        "uncaptured": int(uncaptured),
        "same_col_miss": int(same_col_miss),
        "wall_hits": int(wall_hits),
        "hit_ids": hit_ids,
    }


def _build_example(mode: int) -> np.ndarray | None:
    h = random.randint(13, 18)
    w = random.randint(13, 18)
    axis_r = random.randint(5, h - 6)
    g = np.full((h, w), BG, dtype=int)
    g[axis_r, :] = AXIS

    n_containers = 2 if mode == 1 else random.randint(2, 3)
    mids = _sample_spaced_columns(2, w - 3, n_containers, min_gap=4)
    if mids is None:
        return None

    tops = []
    heights = []
    for idx in range(n_containers):
        if idx == 0:
            top = axis_r + random.randint(1, 2)
            height = random.choice((2, 3))
        elif idx == 1:
            top = axis_r + random.randint(2, 4)
            height = random.choice((2, 3))
        else:
            top = axis_r + random.randint(1, 4)
            height = random.choice((2, 3))
        if top + height - 1 >= h:
            return None
        tops.append(int(top))
        heights.append(int(height))

    containers = [
        _stamp_container(g, int(top), int(mid), int(height))
        for top, mid, height in zip(tops, mids, heights)
    ]

    specs = _scenario_specs(containers, axis_r, w, mode)
    if specs is None:
        return None

    n_markers = len(specs)
    colors = random.sample(list(MARKER_COLORS), n_markers)
    random.shuffle(colors)
    for color, spec in zip(colors, specs):
        r = _marker_row(axis_r, int(spec["delta"]))
        c = int(spec["col"])
        if not (0 <= r < axis_r):
            return None
        if int(g[r, c]) != BG:
            return None
        g[r, c] = int(color)

    analysis = _analyze_grid(g)
    if analysis["captures"] < 1 or analysis["uncaptured"] < 1:
        return None
    if len(analysis["hit_ids"]) != analysis["captures"]:
        return None
    if mode == 0 and (analysis["same_col_miss"] < 1 or analysis["wall_hits"] < 1):
        return None
    if mode == 1 and analysis["same_col_miss"] < 1:
        return None
    if mode == 2 and analysis["wall_hits"] < 1:
        return None

    out = main(g)
    filled = 0
    placed = 0
    for container in _find_containers(g, _find_axis_row(g)):
        interior_colors = {int(out[r, c]) for r, c in container["interior"]}
        if any(color not in (BG, CONTAINER) for color in interior_colors):
            filled += 1
    for color, r, c in _find_markers(g, _find_axis_row(g)):
        rr = int(2 * _find_axis_row(g) - r)
        if out[rr, c] == color:
            placed += 1
    if filled < 1 or placed < 1:
        return None

    return g


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    for _ in range(400):
        mode = _GEN_COUNTER % 3
        _GEN_COUNTER += 1
        g = _build_example(mode)
        if g is not None:
            return g
    raise RuntimeError("failed to generate a valid reflection-container example")
