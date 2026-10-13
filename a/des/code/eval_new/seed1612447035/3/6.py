from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


GREY = 5
MAROON = 9
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 300


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.reshape(-1)))
    return int(counts.most_common(1)[0][0])


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    comps: list[list[tuple[int, int]]] = []
    next_id = 1
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or labels[r, c] != 0:
                continue
            q = deque([(r, c)])
            labels[r, c] = next_id
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if bool(mask[nr, nc]) and labels[nr, nc] == 0:
                        labels[nr, nc] = next_id
                        q.append((nr, nc))
            comps.append(pts)
            next_id += 1
    return labels, comps


def _trace_path(
    neighbors: dict[tuple[int, int], list[tuple[int, int]]],
    start: tuple[int, int],
    expected_len: int,
) -> list[tuple[int, int]]:
    order = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        nxts = [pt for pt in neighbors[cur] if pt != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "path must not branch"
        prev, cur = cur, nxts[0]
        order.append(cur)
    assert len(order) == expected_len, "failed to traverse full path"
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor directed grey paths by the ordered stations they encounter, then enlarge each station by path degree.

    Input structure:
    - bg is the dominant color; grey pixels form one-pixel-wide 4-connected paths with no branches.
    - Every path has exactly two endpoints, and one endpoint has an adjacent maroon pixel that marks the start direction.
    - Non-bg pixels that are neither grey nor maroon are single-pixel coloring stations adjacent to one or more paths.

    Rule:
    - Split the grey pixels into path components and orient each path from the endpoint touching maroon to the opposite endpoint.
    - Along each ordered path, keep the path grey until the first adjacent station is encountered; from that path pixel onward recolor with that station's color, then overwrite again when later stations are encountered.
    - Count how many distinct paths are 4-neighbor-adjacent to every station: degree 1 stays single-pixel, degree 2 becomes a centered 1x3 horizontal segment unless that would hit existing non-bg cells, in which case it becomes a centered 3x1 vertical segment, and degree 3+ becomes a centered 3x3 square; expansions only fill original bg cells. Remove all maroon start markers.

    Key insight:
    - The maroon marker turns every thin grey component into a directed walk, so station colors act as ordered overwrite events along that walk, while the station's own final shape depends only on how many different paths touch it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _get_bg(g)

    path_mask = g == GREY
    path_labels, path_comps = _label_components(path_mask)
    assert path_comps, "expected at least one grey path"

    stations: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color in (bg, GREY, MAROON):
                continue
            stations.append({"pos": (r, c), "color": color})
    assert stations, "expected at least one station"

    path_orders: dict[int, list[tuple[int, int]]] = {}
    path_index: dict[int, dict[tuple[int, int], int]] = {}
    for pid, comp in enumerate(path_comps, start=1):
        comp_set = set(comp)
        neighbor_map: dict[tuple[int, int], list[tuple[int, int]]] = {}
        endpoints: list[tuple[int, int]] = []
        for cell in comp:
            r, c = cell
            neigh = [pt for pt in _neighbors4(r, c, h, w) if pt in comp_set]
            neighbor_map[cell] = neigh
            if len(neigh) == 1:
                endpoints.append(cell)
            else:
                assert len(neigh) == 2, "every non-endpoint path pixel must have degree 2"
        assert len(endpoints) == 2, "every path must have exactly two endpoints"
        starts = [
            pt
            for pt in endpoints
            if any(int(g[nr, nc]) == MAROON for nr, nc in _neighbors4(pt[0], pt[1], h, w))
        ]
        assert len(starts) == 1, "every path must have exactly one maroon-marked start"
        order = _trace_path(neighbor_map, starts[0], len(comp))
        path_orders[pid] = order
        path_index[pid] = {pt: idx for idx, pt in enumerate(order)}

    station_infos: list[dict[str, object]] = []
    events_by_path: dict[int, list[tuple[int, tuple[int, int], int]]] = {pid: [] for pid in path_orders}
    for station in stations:
        r, c = station["pos"]  # type: ignore[assignment]
        color = int(station["color"])  # type: ignore[arg-type]
        adj_pids: set[int] = set()
        by_path: dict[int, list[int]] = {}
        for nr, nc in _neighbors4(r, c, h, w):
            pid = int(path_labels[nr, nc])
            if pid == 0:
                continue
            adj_pids.add(pid)
            by_path.setdefault(pid, []).append(path_index[pid][(nr, nc)])
        station_infos.append({"pos": (r, c), "color": color, "adj_pids": adj_pids})
        for pid, idxs in by_path.items():
            events_by_path[pid].append((min(idxs), (r, c), color))

    out = g.copy()
    for pid, order in path_orders.items():
        events = sorted(events_by_path[pid], key=lambda item: (item[0], item[1][0], item[1][1]))
        current = GREY
        event_idx = 0
        for idx, (r, c) in enumerate(order):
            while event_idx < len(events) and events[event_idx][0] == idx:
                current = int(events[event_idx][2])
                event_idx += 1
            out[r, c] = current

    out[g == MAROON] = bg

    def can_fill(cells: list[tuple[int, int]]) -> bool:
        for rr, cc in cells:
            if not (0 <= rr < h and 0 <= cc < w):
                return False
            if int(g[rr, cc]) != bg:
                return False
            if int(out[rr, cc]) != bg:
                return False
        return True

    for info in sorted(station_infos, key=lambda item: item["pos"]):  # type: ignore[arg-type]
        r, c = info["pos"]  # type: ignore[assignment]
        color = int(info["color"])  # type: ignore[arg-type]
        degree = len(info["adj_pids"])  # type: ignore[arg-type]
        if degree == 2:
            horiz = [(r, c - 1), (r, c + 1)]
            vert = [(r - 1, c), (r + 1, c)]
            cells = horiz if can_fill(horiz) else vert
            if can_fill(cells):
                for rr, cc in cells:
                    out[rr, cc] = color
        elif degree >= 3:
            for rr in range(r - 1, r + 2):
                for cc in range(c - 1, c + 2):
                    if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == bg and int(out[rr, cc]) == bg:
                        out[rr, cc] = color

    return out


def _transform_coords(
    coords: list[tuple[int, int]],
    h: int,
    w: int,
    rot: int,
    flip: bool,
) -> tuple[list[tuple[int, int]], int, int]:
    pts = list(coords)
    hh, ww = h, w
    if flip:
        pts = [(r, ww - 1 - c) for r, c in pts]
    for _ in range(rot % 4):
        pts = [(c, hh - 1 - r) for r, c in pts]
        hh, ww = ww, hh
    return pts, hh, ww


def _transform_module(module: dict[str, object], rot: int, flip: bool) -> dict[str, object]:
    h = int(module["h"])
    w = int(module["w"])

    paths = []
    for path in module["paths"]:  # type: ignore[index]
        pts, hh, ww = _transform_coords(path, h, w, rot, flip)  # type: ignore[arg-type]
        paths.append(pts)

    markers, hh, ww = _transform_coords(module["markers"], h, w, rot, flip)  # type: ignore[arg-type]
    station_pts = [station["pos"] for station in module["stations"]]  # type: ignore[index]
    station_pts_t, _, _ = _transform_coords(station_pts, h, w, rot, flip)
    stations = []
    for pos, station in zip(station_pts_t, module["stations"]):  # type: ignore[index]
        stations.append({"pos": pos, "color": int(station["color"])})  # type: ignore[index]

    return {"h": hh, "w": ww, "paths": paths, "markers": markers, "stations": stations}


def _make_seq_module(colors: list[int]) -> dict[str, object]:
    assert len(colors) in (2, 3)
    variant = random.choice(["straight", "turn"])
    if variant == "straight":
        length = random.randint(6, 8)
        path = [(2, c) for c in range(1, length + 1)]
        marker = (1, 1)
        cols = sorted(random.sample(list(range(3, length)), len(colors)))
        stations = []
        for idx, (col, color) in enumerate(zip(cols, colors)):
            row = 1 if idx % 2 == 0 else 3
            stations.append({"pos": (row, col), "color": color})
        module = {"h": 5, "w": length + 2, "paths": [path], "markers": [marker], "stations": stations}
    else:
        horiz = random.randint(4, 6)
        vert = random.randint(4, 6)
        path = [(vert, c) for c in range(1, horiz + 1)] + [(r, horiz) for r in range(vert - 1, 0, -1)]
        marker = (vert - 1, 1)
        stations = [
            {"pos": (vert - 1, random.randint(3, horiz - 1)), "color": colors[0]},
            {"pos": (random.randint(1, vert - 2), horiz - 1), "color": colors[1]},
        ]
        module = {"h": vert + 2, "w": horiz + 2, "paths": [path], "markers": [marker], "stations": stations}
    return _transform_module(module, rot=random.randrange(4), flip=bool(random.getrandbits(1)))


def _make_degree2_module(color: int, horizontal_available: bool) -> dict[str, object]:
    if horizontal_available:
        up_len = random.randint(2, 4)
        down_len = random.randint(2, 4)
        station_r = up_len + 1
        station_c = 2
        top_path = [(r, station_c) for r in range(1, up_len + 1)]
        bot_path = [(r, station_c) for r in range(station_r + down_len, station_r, -1)]
        module = {
            "h": up_len + down_len + 3,
            "w": 5,
            "paths": [top_path, bot_path],
            "markers": [(0, station_c), (up_len + down_len + 2, station_c)],
            "stations": [{"pos": (station_r, station_c), "color": color}],
        }
    else:
        left_len = random.randint(2, 4)
        right_len = random.randint(2, 4)
        station_r = 2
        station_c = left_len + 1
        left_path = [(station_r, c) for c in range(1, left_len + 1)]
        right_path = [(station_r, c) for c in range(station_c + right_len, station_c, -1)]
        module = {
            "h": 5,
            "w": left_len + right_len + 3,
            "paths": [left_path, right_path],
            "markers": [(station_r, 0), (station_r, left_len + right_len + 2)],
            "stations": [{"pos": (station_r, station_c), "color": color}],
        }
    return _transform_module(module, rot=random.choice([0, 2]), flip=bool(random.getrandbits(1)))


def _make_degree3_module(color: int) -> dict[str, object]:
    lens = {side: random.randint(2, 3) for side in ("left", "right", "up", "down")}
    station_r = lens["up"] + 1
    station_c = lens["left"] + 1
    h = lens["up"] + lens["down"] + 3
    w = lens["left"] + lens["right"] + 3
    missing = random.choice(["left", "right", "up", "down"])
    paths: list[list[tuple[int, int]]] = []
    markers: list[tuple[int, int]] = []

    if missing != "left":
        paths.append([(station_r, c) for c in range(1, lens["left"] + 1)])
        markers.append((station_r, 0))
    if missing != "right":
        paths.append([(station_r, c) for c in range(station_c + lens["right"], station_c, -1)])
        markers.append((station_r, w - 1))
    if missing != "up":
        paths.append([(r, station_c) for r in range(1, lens["up"] + 1)])
        markers.append((0, station_c))
    if missing != "down":
        paths.append([(r, station_c) for r in range(station_r + lens["down"], station_r, -1)])
        markers.append((h - 1, station_c))

    module = {"h": h, "w": w, "paths": paths, "markers": markers, "stations": [{"pos": (station_r, station_c), "color": color}]}
    return _transform_module(module, rot=random.randrange(4), flip=bool(random.getrandbits(1)))


def _stamp_module(grid: np.ndarray, module: dict[str, object]) -> None:
    for path in module["paths"]:  # type: ignore[index]
        for r, c in path:  # type: ignore[misc]
            assert int(grid[r, c]) == _get_bg(grid)
            grid[r, c] = GREY
    for r, c in module["markers"]:  # type: ignore[index]
        assert int(grid[r, c]) == _get_bg(grid)
        grid[r, c] = MAROON
    for station in module["stations"]:  # type: ignore[index]
        r, c = station["pos"]  # type: ignore[index]
        assert int(grid[r, c]) == _get_bg(grid)
        grid[r, c] = int(station["color"])  # type: ignore[index]


def _place_modules(bg: int, modules: list[dict[str, object]]) -> np.ndarray | None:
    slot_ids = list(range(4))
    random.shuffle(slot_ids)
    assigned = {slot: mod for slot, mod in zip(slot_ids[: len(modules)], modules)}

    row_heights = [0, 0]
    col_widths = [0, 0]
    for slot, mod in assigned.items():
        row = slot // 2
        col = slot % 2
        row_heights[row] = max(row_heights[row], int(mod["h"]))
        col_widths[col] = max(col_widths[col], int(mod["w"]))

    top_margin = random.randint(1, 2)
    left_margin = random.randint(1, 2)
    bottom_margin = random.randint(1, 2)
    right_margin = random.randint(1, 2)
    gap_r = random.randint(2, 3)
    gap_c = random.randint(2, 3)

    h = top_margin + row_heights[0] + gap_r + row_heights[1] + bottom_margin
    w = left_margin + col_widths[0] + gap_c + col_widths[1] + right_margin
    if h > 30 or w > 30:
        return None

    grid = np.full((h, w), bg, dtype=int)
    row_starts = [top_margin, top_margin + row_heights[0] + gap_r]
    col_starts = [left_margin, left_margin + col_widths[0] + gap_c]

    for slot, mod in assigned.items():
        row = slot // 2
        col = slot % 2
        cell_h = row_heights[row]
        cell_w = col_widths[col]
        mod_h = int(mod["h"])
        mod_w = int(mod["w"])
        off_r = random.randint(0, cell_h - mod_h)
        off_c = random.randint(0, cell_w - mod_w)
        r0 = row_starts[row] + off_r
        c0 = col_starts[col] + off_c

        placed = {
            "h": mod_h,
            "w": mod_w,
            "paths": [[(r0 + r, c0 + c) for r, c in path] for path in mod["paths"]],  # type: ignore[index]
            "markers": [(r0 + r, c0 + c) for r, c in mod["markers"]],  # type: ignore[index]
            "stations": [
                {"pos": (r0 + station["pos"][0], c0 + station["pos"][1]), "color": int(station["color"])}  # type: ignore[index]
                for station in mod["stations"]  # type: ignore[index]
            ],
        }
        _stamp_module(grid, placed)

    return grid


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([c for c in range(10) if c not in (GREY, MAROON)])
    seq_station_count = random.choice([2, 3])
    station_palette = [c for c in range(10) if c not in (bg, GREY, MAROON)]
    station_colors = random.sample(station_palette, seq_station_count + 2)

    seq_mod = _make_seq_module(station_colors[:seq_station_count])
    deg2_mod = _make_degree2_module(station_colors[seq_station_count], horizontal_available=bool(random.getrandbits(1)))
    deg3_mod = _make_degree3_module(station_colors[seq_station_count + 1])

    grid = _place_modules(bg, [seq_mod, deg2_mod, deg3_mod])
    if grid is None:
        return None

    if _get_bg(grid) != bg:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if out.ndim != 2 or max(out.shape) > 30:
        return None
    if np.any(grid == MAROON) and np.any(out == MAROON):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
