import random
from collections import deque

import numpy as np


BLACK = 0
TRACK = 5
ORANGE = 7

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
COLOR_POOL = [1, 2, 3, 4, 6, 8, 9]
GEN_RETRIES = 300


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(grid == color):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        while q:
            rr, cc = q.popleft()
            comp.append((rr, cc))
            for dr, dc in DIR4:
                nr = rr + dr
                nc = cc + dc
                if _in_bounds((h, w), nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _portal_axis_and_track_cell(
    grid: np.ndarray, cells: list[tuple[int, int]]
) -> tuple[str, tuple[int, int], tuple[int, int]]:
    rows = {r for r, _ in cells}
    cols = {c for _, c in cells}
    domino_axis = "v" if len(cols) == 1 else "h"
    expected_track_axis = "h" if domino_axis == "v" else "v"
    comp = set(cells)
    best = None
    for r, c in cells:
        hcnt = 0
        vcnt = 0
        for dc in (-1, 1):
            nr = r
            nc = c + dc
            if _in_bounds(grid.shape, nr, nc) and (nr, nc) not in comp and int(grid[nr, nc]) not in (BLACK, ORANGE):
                hcnt += 1
        for dr in (-1, 1):
            nr = r + dr
            nc = c
            if _in_bounds(grid.shape, nr, nc) and (nr, nc) not in comp and int(grid[nr, nc]) not in (BLACK, ORANGE):
                vcnt += 1
        axis_score = hcnt if expected_track_axis == "h" else vcnt
        total_score = hcnt + vcnt
        score = (axis_score, total_score)
        if best is None or score > best[0]:
            best = (score, (r, c), hcnt, vcnt)
    assert best is not None
    _, track_cell, hcnt, vcnt = best
    track_axis = expected_track_axis if (hcnt or vcnt) else expected_track_axis
    if hcnt and not vcnt:
        track_axis = "h"
    elif vcnt and not hcnt:
        track_axis = "v"
    offtrack = cells[0] if cells[1] == track_cell else cells[1]
    return track_axis, track_cell, offtrack


def _parse_scene(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    portals_by_color: dict[int, list[dict[str, object]]] = {}
    portal_track_to_info: dict[tuple[int, int], dict[str, object]] = {}
    portal_cells: set[tuple[int, int]] = set()

    for color in np.unique(g):
        color = int(color)
        if color in (BLACK, TRACK, ORANGE):
            continue
        for comp in _components_of_color(g, color):
            if len(comp) != 2:
                continue
            rows = {r for r, _ in comp}
            cols = {c for _, c in comp}
            if not (len(rows) == 1 and len(cols) == 2) and not (len(rows) == 2 and len(cols) == 1):
                continue
            axis, track_cell, offtrack = _portal_axis_and_track_cell(g, comp)
            info = {
                "color": color,
                "cells": tuple(comp),
                "axis": axis,
                "track": track_cell,
                "offtrack": offtrack,
            }
            portals_by_color.setdefault(color, []).append(info)
            portal_track_to_info[track_cell] = info
            portal_cells.update(comp)

    assert portals_by_color, "expected at least one portal pair"
    for color, infos in portals_by_color.items():
        assert len(infos) == 2, f"portal color {color} must appear exactly twice"

    particle_mask = (g != BLACK) & (g != TRACK) & (g != ORANGE)
    particles: list[dict[str, object]] = []
    used_oranges: set[tuple[int, int]] = set()
    for r, c in np.argwhere(particle_mask):
        r = int(r)
        c = int(c)
        if (r, c) in portal_cells:
            continue
        oranges = []
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if _in_bounds(g.shape, nr, nc) and int(g[nr, nc]) == ORANGE:
                oranges.append((nr, nc))
        assert len(oranges) == 1, "each particle must have exactly one adjacent orange marker"
        or_r, or_c = oranges[0]
        direction = (r - or_r, c - or_c)
        assert abs(direction[0]) + abs(direction[1]) == 1
        particles.append({"pos": (r, c), "color": int(g[r, c]), "dir": direction})
        used_oranges.add((or_r, or_c))

    assert particles, "expected at least one particle"

    track_cells = {tuple(map(int, rc)) for rc in np.argwhere(g == TRACK)}
    track_cells.update(portal_track_to_info.keys())
    track_cells.update(tuple(p["pos"]) for p in particles)

    for p in particles:
        r, c = p["pos"]
        dr, dc = p["dir"]
        assert (r + dr, c + dc) in track_cells, "particle direction must point into a track"

    return portals_by_color, portal_track_to_info, particles, track_cells


def _follow_particle(
    start: tuple[int, int],
    direction: tuple[int, int],
    track_cells: set[tuple[int, int]],
    portal_track_to_info: dict[tuple[int, int], dict[str, object]],
    pair_by_track: dict[tuple[int, int], dict[str, object]],
) -> tuple[int, int]:
    cur = start
    dr, dc = direction
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    max_steps = len(track_cells) * 4 + 10
    for _ in range(max_steps):
        state = (cur, (dr, dc))
        assert state not in seen, "teleport path should not loop"
        seen.add(state)

        nr = cur[0] + dr
        nc = cur[1] + dc
        if (nr, nc) not in track_cells:
            return cur
        if (nr, nc) in portal_track_to_info:
            dst = pair_by_track[(nr, nc)]
            sign = dc if dc else dr
            if dst["axis"] == "h":
                dr, dc = 0, 1 if sign > 0 else -1
            else:
                dr, dc = 1 if sign > 0 else -1, 0
            exit_cell = (int(dst["track"][0]) + dr, int(dst["track"][1]) + dc)
            assert exit_cell in track_cells, "portal exit must feed into its destination track"
            cur = exit_cell
        else:
            cur = (nr, nc)
    raise AssertionError("particle walk exceeded step budget")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Teleport each particle through the matching portal pair, preserve signed motion,
    and leave the particle at the end of its post-teleport track.
    """
    g = np.asarray(grid, dtype=int)
    portals_by_color, portal_track_to_info, particles, track_cells = _parse_scene(g)

    pair_by_track: dict[tuple[int, int], dict[str, object]] = {}
    for infos in portals_by_color.values():
        a, b = infos
        pair_by_track[tuple(a["track"])] = b
        pair_by_track[tuple(b["track"])] = a

    out = g.copy()
    out[out == ORANGE] = BLACK
    for p in particles:
        r, c = p["pos"]
        out[r, c] = TRACK

    finals: list[tuple[tuple[int, int], int]] = []
    for p in particles:
        final_cell = _follow_particle(
            tuple(p["pos"]),
            tuple(p["dir"]),
            track_cells,
            portal_track_to_info,
            pair_by_track,
        )
        finals.append((final_cell, int(p["color"])))

    final_positions = [pos for pos, _ in finals]
    assert len(final_positions) == len(set(final_positions)), "particles must not collide at their endpoints"
    for (r, c), color in finals:
        assert int(out[r, c]) == TRACK, "final particle location must be a track endpoint"
        out[r, c] = color
    return out


def _segment_points(spec: dict[str, object], anchor: tuple[int, int]) -> dict[str, object]:
    axis = str(spec["axis"])
    length = int(spec["length"])
    sign = int(spec["sign"])
    portal_idx = int(spec["portal_idx"])
    portal_side = int(spec["portal_side"])

    track_cells: list[tuple[int, int]] = []
    if axis == "h":
        r, c0 = anchor
        track_cells = [(r, c0 + i) for i in range(length)]
        particle = (r, c0 if sign > 0 else c0 + length - 1) if spec["kind"] == "src" else None
        orange = (particle[0], particle[1] - 1) if spec["kind"] == "src" and sign > 0 else None
        if spec["kind"] == "src" and sign < 0:
            orange = (particle[0], particle[1] + 1)
        portal_track = (r, c0 + portal_idx)
        portal_off = (r + portal_side, c0 + portal_idx)
    else:
        r0, c = anchor
        track_cells = [(r0 + i, c) for i in range(length)]
        particle = (r0 if sign > 0 else r0 + length - 1, c) if spec["kind"] == "src" else None
        orange = (particle[0] - 1, particle[1]) if spec["kind"] == "src" and sign > 0 else None
        if spec["kind"] == "src" and sign < 0:
            orange = (particle[0] + 1, particle[1])
        portal_track = (r0 + portal_idx, c)
        portal_off = (r0 + portal_idx, c + portal_side)

    all_cells = set(track_cells)
    all_cells.add(portal_off)
    if particle is not None:
        all_cells.add(particle)
        all_cells.add(orange)
    return {
        "track_cells": track_cells,
        "particle": particle,
        "orange": orange,
        "portal_track": portal_track,
        "portal_off": portal_off,
        "all_cells": all_cells,
    }


def _valid_anchor_ranges(spec: dict[str, object], h: int, w: int) -> tuple[range, range]:
    axis = str(spec["axis"])
    length = int(spec["length"])
    sign = int(spec["sign"])
    portal_side = int(spec["portal_side"])
    if axis == "h":
        row_lo = 1 if portal_side < 0 else 0
        row_hi = h - 2 if portal_side > 0 else h - 1
        col_lo = 1 if spec["kind"] == "src" and sign > 0 else 0
        col_hi = w - length - 1 if spec["kind"] == "src" and sign < 0 else w - length
        return range(row_lo, row_hi + 1), range(col_lo, col_hi + 1)
    row_lo = 1 if spec["kind"] == "src" and sign > 0 else 0
    row_hi = h - length - 1 if spec["kind"] == "src" and sign < 0 else h - length
    col_lo = 1 if portal_side < 0 else 0
    col_hi = w - 2 if portal_side > 0 else w - 1
    return range(row_lo, row_hi + 1), range(col_lo, col_hi + 1)


def _buffer_cells(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    buf: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if _in_bounds(shape, nr, nc):
                    buf.add((nr, nc))
    return buf


def _pick_portal_index(length: int, sign: int, kind: str) -> int:
    if kind == "src":
        endpoint = length - 1 if sign > 0 else 0
        valid = list(range(1, length)) if sign > 0 else list(range(0, length - 1))
    else:
        endpoint = 0 if sign > 0 else length - 1
        valid = list(range(0, length - 1)) if sign > 0 else list(range(1, length))
    interior = [idx for idx in valid if idx != endpoint]
    if interior and random.random() < 0.5:
        return random.choice(interior)
    return endpoint


def _paint_system(grid: np.ndarray, spec: dict[str, object], placed: dict[str, object]) -> None:
    for r, c in placed["track_cells"]:
        grid[r, c] = TRACK
    pr, pc = placed["portal_track"]
    or_r, or_c = placed["portal_off"]
    grid[pr, pc] = int(spec["portal_color"])
    grid[or_r, or_c] = int(spec["portal_color"])
    if spec["kind"] == "src":
        rr, cc = placed["particle"]
        mr, mc = placed["orange"]
        grid[rr, cc] = int(spec["particle_color"])
        grid[mr, mc] = ORANGE


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(15, 23)
    w = random.randint(15, 23)
    num_pairs = random.choice([1, 2, 2, 3])

    if len(COLOR_POOL) < num_pairs:
        return None
    portal_colors = random.sample(COLOR_POOL, num_pairs)
    particle_choices = [c for c in COLOR_POOL if c not in portal_colors]
    if len(particle_choices) < num_pairs:
        return None
    particle_colors = random.sample(particle_choices, num_pairs)

    combo_pool = [("h", "h"), ("h", "v"), ("v", "h"), ("v", "v")]
    random.shuffle(combo_pool)

    specs: list[dict[str, object]] = []
    for idx in range(num_pairs):
        src_axis, dst_axis = combo_pool[idx % len(combo_pool)]
        if random.random() < 0.5:
            src_axis = random.choice(["h", "v"])
            dst_axis = random.choice(["h", "v"])
        sign = random.choice([-1, 1])
        src_len = random.randint(5, 9)
        dst_len = random.randint(4, 9)
        src_spec = {
            "kind": "src",
            "axis": src_axis,
            "length": src_len,
            "sign": sign,
            "portal_idx": _pick_portal_index(src_len, sign, "src"),
            "portal_side": random.choice([-1, 1]),
            "portal_color": portal_colors[idx],
            "particle_color": particle_colors[idx],
        }
        dst_spec = {
            "kind": "dst",
            "axis": dst_axis,
            "length": dst_len,
            "sign": sign,
            "portal_idx": _pick_portal_index(dst_len, sign, "dst"),
            "portal_side": random.choice([-1, 1]),
            "portal_color": portal_colors[idx],
            "particle_color": particle_colors[idx],
        }
        specs.extend([src_spec, dst_spec])

    random.shuffle(specs)
    blocked: set[tuple[int, int]] = set()
    placed_specs: list[tuple[dict[str, object], dict[str, object]]] = []
    for spec in specs:
        rows, cols = _valid_anchor_ranges(spec, h, w)
        if not rows or not cols:
            return None
        anchors = [(r, c) for r in rows for c in cols]
        random.shuffle(anchors)
        placed = None
        for anchor in anchors[:200]:
            candidate = _segment_points(spec, anchor)
            if any(not _in_bounds((h, w), r, c) for r, c in candidate["all_cells"]):
                continue
            if len(candidate["all_cells"]) != len(set(candidate["all_cells"])):
                continue
            if any(cell in blocked for cell in candidate["all_cells"]):
                continue
            if spec["kind"] == "src" and candidate["portal_track"] == candidate["particle"]:
                continue
            placed = candidate
            break
        if placed is None:
            return None
        blocked.update(_buffer_cells(set(placed["all_cells"]), (h, w)))
        placed_specs.append((spec, placed))

    grid = np.full((h, w), BLACK, dtype=int)
    for spec, placed in placed_specs:
        _paint_system(grid, spec, placed)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if int(np.count_nonzero(grid == ORANGE)) != num_pairs:
        return None
    if int(np.count_nonzero(out == ORANGE)) != 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid teleport-portal scene")
