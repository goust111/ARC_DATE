from __future__ import annotations

import random

import numpy as np


RED = 2
GREY = 5
MAROON = 9
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_ATTEMPTS = 400
PATH_ATTEMPTS = 200
PLACEMENT_ATTEMPTS = 120


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_divider(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(cols) == 1, "expected exactly one grey divider column"
    return int(cols[0])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _vec_axis(vec: tuple[int, int]) -> int:
    return 0 if vec[0] != 0 else 1


def _extract_segments(grid: np.ndarray) -> tuple[int, int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    divider = _find_divider(g)

    left_mask = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(divider):
            val = int(g[r, c])
            if val not in (bg, RED, MAROON):
                left_mask[r, c] = True

    comps = _components(left_mask)
    assert comps, "expected at least one path segment"

    segments: list[dict] = []
    for comp in comps:
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1, "each segment must be monochrome"
        color = int(next(iter(colors)))
        coord_set = set(comp)

        degrees = {
            cell: sum((nr, nc) in coord_set for nr, nc in _neighbors4(cell[0], cell[1]))
            for cell in comp
        }
        endpoints = [cell for cell, deg in degrees.items() if deg == 1]
        assert len(endpoints) == 2, "each segment must be a simple path with two endpoints"
        assert all(deg in (1, 2) for deg in degrees.values()), "segment may not branch"

        start = None
        end = None
        start_vec = None
        end_vec = None
        for cell in endpoints:
            red_hits = []
            maroon_hits = []
            for nr, nc in _neighbors4(cell[0], cell[1]):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if int(g[nr, nc]) == RED:
                    red_hits.append((nr, nc))
                if int(g[nr, nc]) == MAROON:
                    maroon_hits.append((nr, nc))
            if red_hits:
                assert start is None and len(red_hits) == 1
                start = cell
                start_vec = (red_hits[0][0] - cell[0], red_hits[0][1] - cell[1])
            if maroon_hits:
                assert end is None and len(maroon_hits) == 1
                end = cell
                end_vec = (maroon_hits[0][0] - cell[0], maroon_hits[0][1] - cell[1])

        assert start is not None and end is not None, "missing endpoint markers"
        assert start_vec is not None and end_vec is not None

        segments.append(
            {
                "color": color,
                "coords": tuple(sorted(comp)),
                "coord_set": coord_set,
                "start": start,
                "end": end,
                "start_vec": start_vec,
                "end_vec": end_vec,
            }
        )

    return bg, divider, segments


def _is_successor(seg_a: dict, seg_b: dict) -> bool:
    if seg_a["end_vec"] != (-seg_b["start_vec"][0], -seg_b["start_vec"][1]):
        return False
    if _vec_axis(seg_a["end_vec"]) != _vec_axis(seg_b["start_vec"]):
        return False
    if seg_a["end_vec"][0] == 0:
        return seg_a["end"][0] == seg_b["start"][0]
    return seg_a["end"][1] == seg_b["start"][1]


def _segment_order(segments: list[dict]) -> list[int]:
    n = len(segments)
    preds = [[] for _ in range(n)]
    succs = [[] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if _is_successor(segments[i], segments[j]):
                succs[i].append(j)
                preds[j].append(i)

    assert all(len(items) <= 1 for items in preds), "ambiguous predecessor relation"
    assert all(len(items) <= 1 for items in succs), "ambiguous successor relation"

    starts = [i for i in range(n) if not preds[i]]
    assert len(starts) == 1, "expected exactly one global start segment"

    order: list[int] = []
    seen: set[int] = set()
    cur = starts[0]
    while True:
        assert cur not in seen, "cycle in segment chain"
        seen.add(cur)
        order.append(cur)
        if not succs[cur]:
            break
        cur = succs[cur][0]

    assert len(order) == n, "segment chain must cover all segments"
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble marked path segments into a chained polyline on the blank right canvas.
    """
    g = np.asarray(grid, dtype=int)
    bg, divider, segments = _extract_segments(g)
    order = _segment_order(segments)

    out = np.full_like(g, bg)
    out[:, divider] = g[:, divider]

    anchor = (0, divider + 1)
    for idx in order:
        seg = segments[idx]
        dr = anchor[0] - seg["start"][0]
        dc = anchor[1] - seg["start"][1]
        for r, c in seg["coords"]:
            rr = r + dr
            cc = c + dc
            assert 0 <= rr < g.shape[0] and divider < cc < g.shape[1], "segment spills out of canvas"
            assert int(out[rr, cc]) == bg, "reconstructed path may not overlap"
            out[rr, cc] = seg["color"]
        end_r = seg["end"][0] + dr
        end_c = seg["end"][1] + dc
        anchor = (end_r + seg["end_vec"][0], end_c + seg["end_vec"][1])

    return out


def _add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return (a[0] + b[0], a[1] + b[1])


def _sub(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return (a[0] - b[0], a[1] - b[1])


def _dir_axis(vec: tuple[int, int]) -> str:
    return "v" if vec[0] else "h"


def _valid_extension(
    cell: tuple[int, int],
    prev: tuple[int, int],
    occupied: set[tuple[int, int]],
    h: int,
    w: int,
) -> bool:
    r, c = cell
    if not (0 <= r < h and 0 <= c < w):
        return False
    if cell in occupied:
        return False
    for nr, nc in _neighbors4(r, c):
        if (nr, nc) in occupied and (nr, nc) != prev:
            return False
    return True


def _generate_path(h: int, w: int, total_len: int) -> list[tuple[int, int]] | None:
    for _ in range(PATH_ATTEMPTS):
        path = [(0, 0)]
        occupied = {(0, 0)}
        prev_dir: tuple[int, int] | None = None
        while len(path) < total_len:
            tail = path[-1]
            candidates: list[tuple[tuple[int, int], int]] = []
            for dr, dc in DIRS:
                nxt = (tail[0] + dr, tail[1] + dc)
                if not _valid_extension(nxt, tail, occupied, h, w):
                    continue
                if len(path) + 1 < total_len:
                    onward = 0
                    for odr, odc in DIRS:
                        nxt2 = (nxt[0] + odr, nxt[1] + odc)
                        if _valid_extension(nxt2, nxt, occupied | {nxt}, h, w):
                            onward += 1
                    if onward == 0:
                        continue
                weight = 4 if prev_dir == (dr, dc) else 3
                weight += 1 if dr == 1 or dc == 1 else 0
                candidates.append((nxt, weight))
            if not candidates:
                break
            cells = [cell for cell, _wgt in candidates]
            weights = [wgt for _cell, wgt in candidates]
            nxt = random.choices(cells, weights=weights, k=1)[0]
            prev_dir = _sub(nxt, tail)
            path.append(nxt)
            occupied.add(nxt)
        if len(path) == total_len:
            return path
    return None


def _segment_lengths(n_segments: int) -> list[int]:
    return [random.randint(3, 6) for _ in range(n_segments)]


def _choose_free_marker(
    used_step: tuple[int, int],
    options: list[tuple[int, int]],
) -> tuple[int, int]:
    free = [vec for vec in options if vec != used_step]
    assert free
    return random.choice(free)


def _build_segments_from_path(
    path: list[tuple[int, int]],
    lengths: list[int],
    colors: list[int],
) -> list[dict]:
    boundaries: list[tuple[int, int]] = []
    running = 0
    for length in lengths[:-1]:
        running += length
        boundaries.append(_sub(path[running], path[running - 1]))

    segments: list[dict] = []
    start_idx = 0
    for i, length in enumerate(lengths):
        cells = path[start_idx : start_idx + length]
        start = cells[0]
        local = [_sub(cell, start) for cell in cells]
        first_step = _sub(cells[1], cells[0])
        last_step_from_end = _sub(cells[-2], cells[-1])
        if i == 0:
            start_vec = _choose_free_marker(first_step, DIRS)
        else:
            start_vec = (-boundaries[i - 1][0], -boundaries[i - 1][1])
            assert start_vec != first_step
        if i == len(lengths) - 1:
            end_vec = _choose_free_marker(last_step_from_end, DIRS)
        else:
            end_vec = boundaries[i]
            assert end_vec != last_step_from_end
        segments.append(
            {
                "color": colors[i],
                "coords_local": tuple(local),
                "end_local": local[-1],
                "start_vec": start_vec,
                "end_vec": end_vec,
            }
        )
        start_idx += length
    return segments


def _rel_positions(seg: dict) -> set[tuple[int, int]]:
    rel = set(seg["coords_local"])
    rel.add(seg["start_vec"])
    rel.add(_add(seg["end_local"], seg["end_vec"]))
    return rel


def _translated_positions(seg: dict, start_pos: tuple[int, int]) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    cells = {_add(start_pos, rel) for rel in seg["coords_local"]}
    markers = {
        _add(start_pos, seg["start_vec"]),
        _add(start_pos, _add(seg["end_local"], seg["end_vec"])),
    }
    return cells, markers


def _fits_left(seg: dict, start_pos: tuple[int, int], h: int, left_w: int) -> bool:
    cells, markers = _translated_positions(seg, start_pos)
    all_pts = cells | markers
    return all(0 <= r < h and 0 <= c < left_w for r, c in all_pts) and len(all_pts) == len(cells) + len(markers)


def _too_close(candidate: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in candidate:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in occupied:
                    return True
    return False


def _place_segments_left(segments: list[dict], h: int, left_w: int) -> list[tuple[int, int]] | None:
    feasible = []
    for seg in segments:
        rel = _rel_positions(seg)
        rows = [r for r, _c in rel]
        cols = [c for _r, c in rel]
        feasible.append(
            (
                -min(rows),
                h - 1 - max(rows),
                -min(cols),
                left_w - 1 - max(cols),
            )
        )

    for _ in range(PLACEMENT_ATTEMPTS):
        starts: list[tuple[int, int]] = []
        occupied: set[tuple[int, int]] = set()

        r0_lo, r0_hi, c0_lo, c0_hi = feasible[0]
        first_candidates = [(r, c) for r in range(r0_lo, r0_hi + 1) for c in range(c0_lo, c0_hi + 1)]
        random.shuffle(first_candidates)

        placed = False
        for start_pos in first_candidates:
            cells, markers = _translated_positions(segments[0], start_pos)
            pts = cells | markers
            if _too_close(pts, occupied):
                continue
            starts.append(start_pos)
            occupied |= pts
            placed = True
            break
        if not placed:
            continue

        ok = True
        for i in range(1, len(segments)):
            prev = segments[i - 1]
            prev_start = starts[i - 1]
            prev_end = _add(prev_start, prev["end_local"])
            r_lo, r_hi, c_lo, c_hi = feasible[i]

            candidates: list[tuple[int, int]] = []
            if _dir_axis(segments[i]["start_vec"]) == "h":
                fixed_r = prev_end[0]
                if not (r_lo <= fixed_r <= r_hi):
                    ok = False
                    break
                cols = list(range(c_lo, c_hi + 1))
                random.shuffle(cols)
                candidates = [(fixed_r, c) for c in cols]
            else:
                fixed_c = prev_end[1]
                if not (c_lo <= fixed_c <= c_hi):
                    ok = False
                    break
                rows = list(range(r_lo, r_hi + 1))
                random.shuffle(rows)
                candidates = [(r, fixed_c) for r in rows]

            placed = False
            for start_pos in candidates:
                cells, markers = _translated_positions(segments[i], start_pos)
                pts = cells | markers
                if _too_close(pts, occupied):
                    continue
                starts.append(start_pos)
                occupied |= pts
                placed = True
                break
            if not placed:
                ok = False
                break

        if ok:
            return starts
    return None


def _render_input(
    h: int,
    side: int,
    bg: int,
    segments: list[dict],
    starts: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((h, 2 * side + 1), bg, dtype=int)
    grid[:, side] = GREY
    for seg, start_pos in zip(segments, starts, strict=True):
        cells, markers = _translated_positions(seg, start_pos)
        for r, c in cells:
            grid[r, c] = seg["color"]
        marker_list = list(markers)
        start_marker = _add(start_pos, seg["start_vec"])
        end_marker = _add(start_pos, _add(seg["end_local"], seg["end_vec"]))
        assert start_marker in markers and end_marker in markers
        grid[start_marker[0], start_marker[1]] = RED
        grid[end_marker[0], end_marker[1]] = MAROON
    return grid


def _render_output(h: int, side: int, bg: int, segments: list[dict]) -> np.ndarray:
    out = np.full((h, 2 * side + 1), bg, dtype=int)
    out[:, side] = GREY
    anchor = (0, side + 1)
    for seg in segments:
        for rel in seg["coords_local"]:
            rr = anchor[0] + rel[0]
            cc = side + 1 + (anchor[1] - (side + 1)) + rel[1]
            out[rr, cc] = seg["color"]
        end_pos = _add(anchor, seg["end_local"])
        anchor = _add(end_pos, seg["end_vec"])
    return out


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(11, 17)
    side = random.randint(11, 16)
    n_segments = random.randint(3, 5)

    bg_choices = [0, 1, 3, 4, 6, 7, 8]
    bg = random.choice(bg_choices)
    segment_pool = [color for color in bg_choices if color != bg]
    colors = random.sample(segment_pool, n_segments)

    lengths = _segment_lengths(n_segments)
    total_len = sum(lengths)
    path = _generate_path(h, side, total_len)
    if path is None:
        return None

    boundaries = []
    offset = 0
    for length in lengths[:-1]:
        offset += length
        boundaries.append(_sub(path[offset], path[offset - 1]))
    boundary_axes = {_dir_axis(vec) for vec in boundaries}
    if len(boundary_axes) < 2:
        return None

    segments = _build_segments_from_path(path, lengths, colors)
    starts = _place_segments_left(segments, h, side)
    if starts is None:
        return None

    inp = _render_input(h, side, bg, segments, starts)
    out = _render_output(h, side, bg, segments)
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_case()
        if built is None:
            continue
        inp, expected = built
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid input")
