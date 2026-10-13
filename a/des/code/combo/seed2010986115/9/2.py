from __future__ import annotations

import random

import numpy as np


BG = 5
TRACK = 8
FRAGMENT_COLORS = [0, 1, 2, 3, 4, 6, 7, 9]
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_GRID_MIN = 12
GEN_GRID_MAX = 18
GEN_TRACK_COUNT_CHOICES = [2, 3, 4]
GEN_TRACK_COUNT_WEIGHTS = [5, 8, 4]
GEN_POS_GAP = 2
GEN_MAX_FRAGMENTS = 8
GEN_MAX_ATTEMPTS = 400


Cell = tuple[int, int]


def _neighbors(cell: Cell, shape: tuple[int, int]) -> list[Cell]:
    r, c = cell
    h, w = shape
    out: list[Cell] = []
    for dr, dc in DIRS:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _follow_track_from_endpoint(grid: np.ndarray, endpoint: Cell, entry: Cell) -> list[Cell]:
    path = [entry]
    visited = {entry}
    prev = endpoint
    cur = entry

    while True:
        move = (cur[0] - prev[0], cur[1] - prev[1])
        next_cells = [nxt for nxt in _neighbors(cur, grid.shape) if int(grid[nxt]) == TRACK and nxt != prev]
        if not next_cells:
            return path
        if len(next_cells) == 1:
            nxt = next_cells[0]
        else:
            straight = [nxt for nxt in next_cells if (nxt[0] - cur[0], nxt[1] - cur[1]) == move]
            assert len(straight) == 1, "crossings must be resolved by continuing straight"
            nxt = straight[0]
        assert nxt not in visited, "track may not loop"
        path.append(nxt)
        visited.add(nxt)
        prev, cur = cur, nxt


def _parse_tracks(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    colors = sorted(set(map(int, np.unique(g))) - {BG, TRACK})
    assert colors, "expected at least one endpoint color"
    tracks: list[dict] = []

    for color in colors:
        coords = [tuple(map(int, coord)) for coord in np.argwhere(g == color)]
        candidates: list[dict] = []

        for endpoint in coords:
            entries = [nbr for nbr in _neighbors(endpoint, g.shape) if int(g[nbr]) == TRACK]
            if len(entries) != 1:
                continue
            entry = entries[0]

            try:
                endpoint_to_start = _follow_track_from_endpoint(g, endpoint, entry)
            except AssertionError:
                continue
            if len(endpoint_to_start) < 2:
                continue
            step0 = (entry[0] - endpoint[0], entry[1] - endpoint[1])
            step1 = (
                endpoint_to_start[1][0] - entry[0],
                endpoint_to_start[1][1] - entry[1],
            )
            if step0 != step1:
                continue

            path = list(reversed(endpoint_to_start))
            path_index = {cell: idx for idx, cell in enumerate(path)}
            fragment_cells: list[Cell] = []
            fragment_indices: list[int] = []

            ok = True
            for coord in coords:
                if coord == endpoint:
                    continue
                touched = [nbr for nbr in _neighbors(coord, g.shape) if nbr in path_index]
                if len(touched) != 1:
                    ok = False
                    break
                fragment_cells.append(coord)
                fragment_indices.append(path_index[touched[0]])
            if not ok:
                continue

            candidates.append(
                {
                    "color": color,
                    "endpoint": endpoint,
                    "path": path,
                    "fragment_cells": fragment_cells,
                    "fragment_indices": sorted(fragment_indices),
                }
            )

        assert len(candidates) == 1, f"could not uniquely identify endpoint for color {color}"
        tracks.append(candidates[0])

    return tracks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each colored fragment onto the azure track cells nearest its same-colored endpoint.

    Tracks are reconstructed by following the azure path away from each candidate endpoint,
    continuing straight through plus-shaped crossings and turning only at degree-2 corners.
    Fragments of the same color compact into the suffix of that track nearest the endpoint.
    When two compacted suffixes want the same shared track cell, the track whose endpoint
    is earlier in reading order stays visible.
    """
    g = np.asarray(grid, dtype=int)
    tracks = _parse_tracks(g)
    out = g.copy()

    for track in tracks:
        for cell in track["fragment_cells"]:
            out[cell] = BG

    by_priority = sorted(tracks, key=lambda item: item["endpoint"])
    for track in reversed(by_priority):
        count = len(track["fragment_cells"])
        if count == 0:
            continue
        path = track["path"]
        assert count <= len(path), "track must have room for all fragments"
        for cell in path[-count:]:
            out[cell] = int(track["color"])

    return out


def _spaced_positions(n: int, lo: int, hi: int, min_gap: int) -> list[int] | None:
    span = hi - lo + 1
    need = n + (n - 1) * min_gap
    if need > span:
        return None
    picks = sorted(random.sample(range(span - (n - 1) * min_gap), n))
    return [lo + picks[i] + i * min_gap for i in range(n)]


def _orientation(path: list[Cell], idx: int) -> str | None:
    if len(path) < 2:
        return None
    if idx == 0:
        a = path[0]
        b = path[1]
        return "h" if a[0] == b[0] else "v"
    if idx == len(path) - 1:
        a = path[-2]
        b = path[-1]
        return "h" if a[0] == b[0] else "v"

    prev = path[idx - 1]
    cur = path[idx]
    nxt = path[idx + 1]
    if prev[0] == cur[0] == nxt[0]:
        return "h"
    if prev[1] == cur[1] == nxt[1]:
        return "v"
    return None


def _build_horizontal_path(h: int, w: int, row: int, endpoint_side: str, shape: str, bend_col: int | None, start_side: str | None) -> tuple[Cell, list[Cell]]:
    if endpoint_side == "left":
        endpoint = (row, 0)
        if shape == "straight":
            path = [(row, c) for c in range(w - 2, 0, -1)]
        else:
            assert bend_col is not None and start_side in {"top", "bottom"}
            if start_side == "top":
                vertical = [(r, bend_col) for r in range(1, row + 1)]
            else:
                vertical = [(r, bend_col) for r in range(h - 2, row - 1, -1)]
            horizontal = [(row, c) for c in range(bend_col - 1, 0, -1)]
            path = vertical + horizontal
    else:
        assert endpoint_side == "right"
        endpoint = (row, w - 1)
        if shape == "straight":
            path = [(row, c) for c in range(1, w - 1)]
        else:
            assert bend_col is not None and start_side in {"top", "bottom"}
            if start_side == "top":
                vertical = [(r, bend_col) for r in range(1, row + 1)]
            else:
                vertical = [(r, bend_col) for r in range(h - 2, row - 1, -1)]
            horizontal = [(row, c) for c in range(bend_col + 1, w - 1)]
            path = vertical + horizontal
    return endpoint, path


def _build_vertical_path(h: int, w: int, col: int, endpoint_side: str, shape: str, bend_row: int | None, start_side: str | None) -> tuple[Cell, list[Cell]]:
    if endpoint_side == "top":
        endpoint = (0, col)
        if shape == "straight":
            path = [(r, col) for r in range(h - 2, 0, -1)]
        else:
            assert bend_row is not None and start_side in {"left", "right"}
            if start_side == "left":
                horizontal = [(bend_row, c) for c in range(1, col + 1)]
            else:
                horizontal = [(bend_row, c) for c in range(w - 2, col - 1, -1)]
            vertical = [(r, col) for r in range(bend_row - 1, 0, -1)]
            path = horizontal + vertical
    else:
        assert endpoint_side == "bottom"
        endpoint = (h - 1, col)
        if shape == "straight":
            path = [(r, col) for r in range(1, h - 1)]
        else:
            assert bend_row is not None and start_side in {"left", "right"}
            if start_side == "left":
                horizontal = [(bend_row, c) for c in range(1, col + 1)]
            else:
                horizontal = [(bend_row, c) for c in range(w - 2, col - 1, -1)]
            vertical = [(r, col) for r in range(bend_row + 1, h - 1)]
            path = horizontal + vertical
    return endpoint, path


def _track_degrees(all_track_cells: set[Cell], shape: tuple[int, int]) -> dict[Cell, int]:
    return {
        cell: sum(1 for nbr in _neighbors(cell, shape) if nbr in all_track_cells)
        for cell in all_track_cells
    }


def _candidate_fragment_sites(base: np.ndarray, path: list[Cell], all_track_cells: set[Cell], endpoint_cells: set[Cell]) -> dict[int, list[Cell]]:
    h, w = base.shape
    degrees = _track_degrees(all_track_cells, base.shape)
    candidates: dict[int, list[Cell]] = {}

    for idx, cell in enumerate(path):
        if idx == len(path) - 1:
            continue
        if degrees[cell] != 2:
            continue
        orient = _orientation(path, idx)
        if orient is None:
            continue

        r, c = cell
        side_cells = [(r - 1, c), (r + 1, c)] if orient == "h" else [(r, c - 1), (r, c + 1)]
        valid: list[Cell] = []
        for pos in side_cells:
            rr, cc = pos
            if not (1 <= rr < h - 1 and 1 <= cc < w - 1):
                continue
            if int(base[rr, cc]) != BG:
                continue
            if pos in endpoint_cells or pos in all_track_cells:
                continue
            touched = [nbr for nbr in _neighbors(pos, base.shape) if nbr in all_track_cells]
            if touched == [cell] or set(touched) == {cell}:
                valid.append(pos)
        if valid:
            candidates[idx] = valid

    return candidates


def _occupied_suffixes(tracks: list[dict]) -> list[set[Cell]]:
    occupied: list[set[Cell]] = []
    for track in tracks:
        count = int(track["count"])
        occupied.append(set(track["path"][-count:]))
    return occupied


def _choose_counts(tracks: list[dict], want_overlap: bool) -> bool:
    shared_options: list[tuple[int, int, int, int]] = []
    for i, track_i in enumerate(tracks):
        for j in range(i + 1, len(tracks)):
            track_j = tracks[j]
            shared = set(track_i["path"]) & set(track_j["path"])
            for cell in shared:
                idx_i = track_i["path_index"][cell]
                idx_j = track_j["path_index"][cell]
                need_i = len(track_i["path"]) - idx_i
                need_j = len(track_j["path"]) - idx_j
                if need_i <= track_i["max_count"] and need_j <= track_j["max_count"]:
                    shared_options.append((i, j, need_i, need_j))

    if want_overlap:
        if not shared_options:
            return False
        i, j, need_i, need_j = random.choice(shared_options)
        for idx, track in enumerate(tracks):
            lo = 1
            hi = int(track["max_count"])
            if idx == i:
                lo = max(lo, need_i)
            if idx == j:
                lo = max(lo, need_j)
            if lo > hi:
                return False
            track["count"] = random.randint(lo, hi)
        if not any(_occupied_suffixes(tracks)[a] & _occupied_suffixes(tracks)[b] for a in range(len(tracks)) for b in range(a + 1, len(tracks))):
            return False
        return True

    for track in tracks:
        safe_cap = int(track["max_count"])
        for need in track["shared_needs"]:
            safe_cap = min(safe_cap, need - 1)
        if safe_cap < 1:
            safe_cap = 1
        track["count"] = random.randint(1, safe_cap)

    return not any(_occupied_suffixes(tracks)[a] & _occupied_suffixes(tracks)[b] for a in range(len(tracks)) for b in range(a + 1, len(tracks)))


def _touches_any(pos: Cell, existing: list[Cell]) -> bool:
    return any(abs(pos[0] - rr) + abs(pos[1] - cc) == 1 or pos == (rr, cc) for rr, cc in existing)


def _choose_fragment_positions(track: dict, existing: list[Cell]) -> list[Cell] | None:
    count = int(track["count"])
    path = track["path"]
    sites = track["candidate_sites"]
    preferred = [idx for idx in sites if idx < len(path) - count]
    others = [idx for idx in sites if idx >= len(path) - count]

    for _ in range(80):
        chosen: list[Cell] = []
        preferred_order = preferred[:]
        other_order = others[:]
        random.shuffle(preferred_order)
        random.shuffle(other_order)
        index_order = preferred_order + other_order
        for idx in index_order:
            pos_options = sites[idx][:]
            random.shuffle(pos_options)
            for pos in pos_options:
                if _touches_any(pos, existing) or _touches_any(pos, chosen):
                    continue
                chosen.append(pos)
                break
            if len(chosen) == count:
                return chosen
    return None


def _render_expected(base: np.ndarray, tracks: list[dict]) -> np.ndarray:
    out = base.copy()
    by_priority = sorted(tracks, key=lambda item: item["endpoint"])
    for track in reversed(by_priority):
        for cell in track["path"][-int(track["count"]):]:
            out[cell] = int(track["color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    w = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    n_tracks = random.choices(GEN_TRACK_COUNT_CHOICES, weights=GEN_TRACK_COUNT_WEIGHTS, k=1)[0]
    n_horizontal = random.randint(1, n_tracks - 1)
    n_vertical = n_tracks - n_horizontal

    rows = _spaced_positions(n_horizontal, 2, h - 3, GEN_POS_GAP)
    cols = _spaced_positions(n_vertical, 2, w - 3, GEN_POS_GAP)
    if rows is None or cols is None:
        return None

    colors = random.sample(FRAGMENT_COLORS, n_tracks)
    base = np.full((h, w), BG, dtype=int)

    tracks: list[dict] = []
    row_values = rows[:]
    col_values = cols[:]
    random.shuffle(row_values)
    random.shuffle(col_values)

    horiz_meta = [{"row": row, "color": colors[i]} for i, row in enumerate(row_values)]
    vert_meta = [{"col": col, "color": colors[n_horizontal + i]} for i, col in enumerate(col_values)]

    h_bend_pool = [c for c in range(2, w - 2) if c not in cols]
    v_bend_pool = [r for r in range(2, h - 2) if r not in rows]
    random.shuffle(h_bend_pool)
    random.shuffle(v_bend_pool)

    h_l_count = min(len(horiz_meta), len(h_bend_pool), random.randint(0, len(horiz_meta)))
    v_l_count = min(len(vert_meta), len(v_bend_pool), random.randint(0, len(vert_meta)))
    if h_l_count == 0 and v_l_count == 0 and random.random() < 0.6:
        if h_bend_pool and horiz_meta:
            h_l_count = 1
        elif v_bend_pool and vert_meta:
            v_l_count = 1

    random.shuffle(horiz_meta)
    random.shuffle(vert_meta)
    h_l_indices = set(range(h_l_count))
    v_l_indices = set(range(v_l_count))

    for idx, item in enumerate(horiz_meta):
        row = int(item["row"])
        endpoint_side = random.choice(["left", "right"])
        if idx in h_l_indices:
            bend_col = h_bend_pool.pop()
            start_side = random.choice(["top", "bottom"])
            endpoint, path = _build_horizontal_path(h, w, row, endpoint_side, "L", bend_col, start_side)
        else:
            endpoint, path = _build_horizontal_path(h, w, row, endpoint_side, "straight", None, None)
        tracks.append({"color": int(item["color"]), "endpoint": endpoint, "path": path})

    for idx, item in enumerate(vert_meta):
        col = int(item["col"])
        endpoint_side = random.choice(["top", "bottom"])
        if idx in v_l_indices:
            bend_row = v_bend_pool.pop()
            start_side = random.choice(["left", "right"])
            endpoint, path = _build_vertical_path(h, w, col, endpoint_side, "L", bend_row, start_side)
        else:
            endpoint, path = _build_vertical_path(h, w, col, endpoint_side, "straight", None, None)
        tracks.append({"color": int(item["color"]), "endpoint": endpoint, "path": path})

    all_track_cells = {cell for track in tracks for cell in track["path"]}
    endpoint_cells = {track["endpoint"] for track in tracks}

    for track in tracks:
        for cell in track["path"]:
            base[cell] = TRACK
        base[track["endpoint"]] = int(track["color"])

    for track in tracks:
        track["path_index"] = {cell: idx for idx, cell in enumerate(track["path"])}
        track["candidate_sites"] = _candidate_fragment_sites(base, track["path"], all_track_cells, endpoint_cells)
        track["max_count"] = min(len(track["path"]), len(track["candidate_sites"]), GEN_MAX_FRAGMENTS)
        if track["max_count"] < 1:
            return None
        track["shared_needs"] = []

    for i, track_i in enumerate(tracks):
        for j in range(i + 1, len(tracks)):
            track_j = tracks[j]
            for cell in set(track_i["path"]) & set(track_j["path"]):
                need_i = len(track_i["path"]) - track_i["path_index"][cell]
                need_j = len(track_j["path"]) - track_j["path_index"][cell]
                track_i["shared_needs"].append(need_i)
                track_j["shared_needs"].append(need_j)

    want_overlap = random.random() < 0.5
    if not _choose_counts(tracks, want_overlap):
        return None

    inp = base.copy()
    chosen_fragments: list[Cell] = []
    for track in sorted(tracks, key=lambda item: (len(item["candidate_sites"]), -int(item["count"]))):
        positions = _choose_fragment_positions(track, chosen_fragments)
        if positions is None:
            return None
        track["fragment_cells"] = positions
        chosen_fragments.extend(positions)
        for pos in positions:
            inp[pos] = int(track["color"])

    expected = _render_expected(base, tracks)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if not np.any(inp != expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid conveyor-track puzzle")
