from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


BG = 0
MARKER = 9
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PATH_COLORS = [1, 2, 3, 4, 5, 6, 7, 8]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as coordinate arrays."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIRS4:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(sorted(coords), dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _sort_segment_coords(coords: np.ndarray) -> tuple[list[tuple[int, int]], str]:
    r0, c0, r1, c1 = _bbox(coords)
    if r0 == r1:
        ordered = sorted(((int(r), int(c)) for r, c in coords), key=lambda p: p[1])
        return ordered, "h"
    if c0 == c1:
        ordered = sorted(((int(r), int(c)) for r, c in coords), key=lambda p: p[0])
        return ordered, "v"
    raise AssertionError("each path segment must be a horizontal or vertical line")


def _dir_from_cells(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return int(b[0] - a[0]), int(b[1] - a[1])


def _turn_left(d: tuple[int, int]) -> tuple[int, int]:
    dr, dc = d
    return -dc, dr


def _turn_right(d: tuple[int, int]) -> tuple[int, int]:
    dr, dc = d
    return dc, -dr


def _translate_cells(cells: list[tuple[int, int]], delta: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = delta
    return [(int(r + dr), int(c + dc)) for r, c in cells]


def _dilate(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _anchor_key(seg: dict) -> tuple[int, int]:
    r0, c0, _r1, _c1 = seg["bbox"]
    return int(r0), int(c0)


def _parse_segments(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], int]]:
    seg_mask = (grid != BG) & (grid != MARKER)
    comps = _components(seg_mask)
    assert comps, "expected at least one segment"

    segments: list[dict] = []
    cell_to_seg: dict[tuple[int, int], int] = {}

    for coords in comps:
        ordered, orientation = _sort_segment_coords(coords)
        length = len(ordered)
        assert length >= 3, "generator keeps segments length >= 3"

        interior_colors = {int(grid[r, c]) for r, c in ordered[1:-1]}
        assert len(interior_colors) == 1, "interior of each segment must be monochrome"
        main = int(next(iter(interior_colors)))

        connector_by_end: dict[int, int] = {}
        for end_idx, cell in ((0, ordered[0]), (1, ordered[-1])):
            color = int(grid[cell])
            if color != main:
                connector_by_end[int(end_idx)] = int(color)

        seg = {
            "bbox": _bbox(coords),
            "ordered": ordered,
            "orientation": orientation,
            "length": int(length),
            "main": int(main),
            "connector_by_end": connector_by_end,
            "markers": [],
        }
        seg_idx = len(segments)
        segments.append(seg)
        for cell in ordered:
            cell_to_seg[cell] = int(seg_idx)

    marker_coords = np.argwhere(grid == MARKER)
    for mr_raw, mc_raw in marker_coords:
        mr = int(mr_raw)
        mc = int(mc_raw)
        hosts: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
        for dr, dc in DIRS4:
            rr = int(mr - dr)
            cc = int(mc - dc)
            seg_idx = cell_to_seg.get((rr, cc))
            if seg_idx is None:
                continue
            seg = segments[seg_idx]
            if (rr, cc) in (seg["ordered"][0], seg["ordered"][-1]):
                continue
            side = (int(dr), int(dc))
            if seg["orientation"] == "h":
                assert side in {(-1, 0), (1, 0)}
            else:
                assert side in {(0, -1), (0, 1)}
            hosts.append((int(seg_idx), (int(rr), int(cc)), side))

        assert len(hosts) == 1, "each maroon marker must touch exactly one non-endpoint segment cell"
        seg_idx, host_cell, side = hosts[0]
        segments[seg_idx]["markers"].append(
            {
                "marker": (int(mr), int(mc)),
                "host": host_cell,
                "side": side,
            }
        )

    return segments, cell_to_seg


def _build_chain(segments: list[dict]) -> tuple[list[int], dict[int, int]]:
    n = len(segments)
    by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for seg_idx, seg in enumerate(segments):
        for end_idx, color in seg["connector_by_end"].items():
            by_color[int(color)].append((int(seg_idx), int(end_idx)))

    adjacency: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for color, endpoints in by_color.items():
        assert len(endpoints) == 2, "each connector color must appear on exactly two endpoints"
        (a, ea), (b, eb) = endpoints
        adjacency[int(a)].append((int(b), int(ea), int(eb)))
        adjacency[int(b)].append((int(a), int(eb), int(ea)))

    if n == 1:
        return [0], {0: 0}

    degrees = {idx: len(adjacency.get(idx, [])) for idx in range(n)}
    ends = [idx for idx, deg in degrees.items() if deg == 1]
    assert len(ends) == 2, "segment graph must be a single chain"
    assert all(deg in (1, 2) for deg in degrees.values()), "segment graph must not branch"

    root = min(ends, key=lambda idx: _anchor_key(segments[idx]))
    order: list[int] = []
    entry_end_by_seg: dict[int, int] = {}

    prev = None
    cur = int(root)
    while True:
        seg = segments[cur]
        nbrs = [item for item in adjacency.get(cur, []) if item[0] != prev]
        used = [item for item in adjacency.get(cur, []) if item[0] == prev]

        if prev is None:
            if not adjacency.get(cur):
                entry_end = 0
            else:
                connector_end = adjacency[cur][0][1]
                entry_end = 1 - int(connector_end)
        else:
            assert len(used) == 1
            entry_end = int(used[0][1])

        entry_end_by_seg[int(cur)] = int(entry_end)
        order.append(int(cur))

        if not nbrs:
            break
        assert len(nbrs) == 1, "chain traversal must have a unique next segment"
        prev, cur = cur, int(nbrs[0][0])

    assert len(order) == n, "chain traversal must visit every segment"
    return order, entry_end_by_seg


def _ordered_from_entry(seg: dict, entry_end: int) -> list[tuple[int, int]]:
    return list(seg["ordered"] if int(entry_end) == 0 else seg["ordered"][::-1])


def _grow_branches(
    canvas: np.ndarray,
    markers: list[dict],
) -> np.ndarray:
    out = canvas.copy()
    h, w = out.shape

    for marker in markers:
        mr, mc = marker["marker"]
        dr, dc = marker["side"]
        color = int(marker["color"])
        length = int(marker["length"])
        for step in range(length):
            rr = int(mr + dr * step)
            cc = int(mc + dc * step)
            if not (0 <= rr < h and 0 <= cc < w):
                break
            if int(out[rr, cc]) != BG:
                break
            out[rr, cc] = int(color)

    return out


def _assemble_segments(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    segments, _cell_to_seg = _parse_segments(grid)
    order, entry_end_by_seg = _build_chain(segments)

    h, w = grid.shape
    out = np.full((h, w), BG, dtype=int)
    occupied: set[tuple[int, int]] = set()
    placed_markers: list[dict] = []
    placed_coords: dict[int, list[tuple[int, int]]] = {}

    for pos, seg_idx in enumerate(order):
        seg = segments[seg_idx]
        src = _ordered_from_entry(seg, entry_end_by_seg[seg_idx])

        if pos == 0:
            delta = (0, 0)
        else:
            prev_coords = placed_coords[order[pos - 1]]
            prev_dir = _dir_from_cells(prev_coords[-2], prev_coords[-1])
            target_start = (int(prev_coords[-1][0] + prev_dir[0]), int(prev_coords[-1][1] + prev_dir[1]))
            delta = (int(target_start[0] - src[0][0]), int(target_start[1] - src[0][1]))

        dst = _translate_cells(src, delta)
        for rr, cc in dst:
            assert 0 <= rr < h and 0 <= cc < w, "assembled path must stay within the canvas"
            assert (rr, cc) not in occupied, "assembled segments must not overlap"
            occupied.add((int(rr), int(cc)))
            out[rr, cc] = int(seg["main"])

        placed_coords[seg_idx] = dst

        for marker in seg["markers"]:
            mr, mc = marker["marker"]
            host = marker["host"]
            translated_marker = (int(mr + delta[0]), int(mc + delta[1]))
            translated_host = (int(host[0] + delta[0]), int(host[1] + delta[1]))
            side = marker["side"]
            assert translated_marker == (translated_host[0] + side[0], translated_host[1] + side[1])
            placed_markers.append(
                {
                    "marker": translated_marker,
                    "host": translated_host,
                    "side": side,
                    "color": int(seg["main"]),
                    "length": int(seg["length"]),
                }
            )

    return out, placed_markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble tagged straight segments into one continuous chain, then emit branches from maroon side markers.

    Visible rules handled by this solver:
    - Black is background; Maroon singleton pixels are side markers.
    - Every non-marker component is a 1-pixel-thick horizontal or vertical segment.
    - Segment interiors reveal the segment's main color; endpoint colors different from the main color are connector tags.
    - Equal connector colors pair two endpoints. Those pairings form one non-branching chain.
    - The upper-left chain-end segment stays fixed; every later segment is translated so its entry endpoint sits one step
      beyond the previous segment's exit direction. Connector endpoints revert to the segment main colors after assembly.
    - Markers move together with their host segment. Each marker emits a same-color branch outward from its side, with
      length equal to the host segment length. The branch repaints the marker cell and stops at the border or any
      existing non-background pixel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assembled, markers = _assemble_segments(g)
    return _grow_branches(assembled, markers)


def _build_chain_geometry(rng: random.Random) -> list[dict] | None:
    seg_count = rng.choices([2, 3, 4], weights=[2, 3, 3], k=1)[0]
    lengths = [rng.randint(3, 6) for _ in range(seg_count)]

    occupied: set[tuple[int, int]] = set()
    segments: list[dict] = []
    prev_exit: tuple[int, int] | None = None
    prev_dir: tuple[int, int] | None = None

    for idx in range(seg_count):
        if idx == 0:
            options = DIRS4[:]
            rng.shuffle(options)
            start = (0, 0)
        else:
            assert prev_exit is not None and prev_dir is not None
            options = [prev_dir, _turn_left(prev_dir), _turn_right(prev_dir)]
            rng.shuffle(options)
            start = (int(prev_exit[0] + prev_dir[0]), int(prev_exit[1] + prev_dir[1]))

        found = False
        for d in options:
            cells = [(int(start[0] + k * d[0]), int(start[1] + k * d[1])) for k in range(lengths[idx])]
            ok = True
            for j, cell in enumerate(cells):
                if cell in occupied:
                    ok = False
                    break
                allowed = set()
                if idx > 0 and j == 0:
                    allowed.add(prev_exit)
                for dr, dc in DIRS4:
                    nb = (int(cell[0] + dr), int(cell[1] + dc))
                    if nb in occupied and nb not in allowed:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue

            for cell in cells:
                occupied.add(cell)
            segments.append({"coords_chain": cells, "dir": d, "length": int(lengths[idx])})
            prev_exit = cells[-1]
            prev_dir = d
            found = True
            break

        if not found:
            return None

    return segments


def _valid_branch_candidate(
    marker: tuple[int, int],
    branch: list[tuple[int, int]],
    host: tuple[int, int],
    path_set: set[tuple[int, int]],
) -> bool:
    if marker in path_set:
        return False
    if any(cell in path_set for cell in branch):
        return False

    for step, cell in enumerate(branch):
        allowed = {host} if step == 0 else set()
        for dr, dc in DIRS4:
            nb = (int(cell[0] + dr), int(cell[1] + dc))
            if nb in path_set and nb not in allowed:
                return False
    return True


def _choose_markers(
    segments: list[dict],
    rng: random.Random,
) -> list[dict] | None:
    path_set = {cell for seg in segments for cell in seg["coords_chain"]}
    candidates_by_seg: dict[int, list[dict]] = {}

    for seg_idx, seg in enumerate(segments):
        d = seg["dir"]
        side_dirs = [_turn_left(d), _turn_right(d)]
        options: list[dict] = []
        for host in seg["coords_chain"][1:-1]:
            for side in side_dirs:
                marker = (int(host[0] + side[0]), int(host[1] + side[1]))
                branch = [(int(marker[0] + k * side[0]), int(marker[1] + k * side[1])) for k in range(seg["length"])]
                if _valid_branch_candidate(marker, branch, host, path_set):
                    options.append(
                        {
                            "seg_idx": int(seg_idx),
                            "host": host,
                            "marker_rel": marker,
                            "side": side,
                            "branch_rel": branch,
                        }
                    )
        if options:
            candidates_by_seg[int(seg_idx)] = options

    if not candidates_by_seg:
        return None

    usable = list(candidates_by_seg)
    max_markers = min(len(usable), len(segments))
    target = rng.randint(1, max_markers)
    rng.shuffle(usable)
    chosen: list[dict] = []

    for seg_idx in usable:
        if len(chosen) == target:
            break
        options = candidates_by_seg[seg_idx][:]
        rng.shuffle(options)
        picked = None
        for cand in options:
            cand_cells = set(cand["branch_rel"])
            ok = True
            for other in chosen:
                other_cells = set(other["branch_rel"])
                if cand_cells & other_cells:
                    ok = False
                    break
                for cell in cand_cells:
                    for dr, dc in DIRS4:
                        if (int(cell[0] + dr), int(cell[1] + dc)) in other_cells:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                picked = cand
                break
        if picked is not None:
            chosen.append(picked)

    return chosen if chosen else None


def _shift_model(
    segments: list[dict],
    markers: list[dict],
    rng: random.Random,
) -> tuple[list[dict], list[dict], tuple[int, int, int, int], set[tuple[int, int]]]:
    output_cells = {cell for seg in segments for cell in seg["coords_chain"]}
    for marker in markers:
        output_cells.update(marker["branch_rel"])

    pts = np.array(sorted(output_cells), dtype=int)
    r0, c0, r1, c1 = _bbox(pts)
    top_margin = rng.randint(1, 3)
    left_margin = rng.randint(1, 3)
    delta = (int(top_margin - r0), int(left_margin - c0))

    shifted_segments: list[dict] = []
    for seg in segments:
        shifted_segments.append(
            {
                **seg,
                "coords_chain": _translate_cells(seg["coords_chain"], delta),
            }
        )

    shifted_markers: list[dict] = []
    for marker in markers:
        shifted_markers.append(
            {
                **marker,
                "host": (int(marker["host"][0] + delta[0]), int(marker["host"][1] + delta[1])),
                "marker_rel": (int(marker["marker_rel"][0] + delta[0]), int(marker["marker_rel"][1] + delta[1])),
                "branch_rel": _translate_cells(marker["branch_rel"], delta),
            }
        )

    shifted_cells = {cell for seg in shifted_segments for cell in seg["coords_chain"]}
    for marker in shifted_markers:
        shifted_cells.update(marker["branch_rel"])
    pts2 = np.array(sorted(shifted_cells), dtype=int)
    bbox = _bbox(pts2)
    return shifted_segments, shifted_markers, bbox, shifted_cells


def _paint_expected_output(
    shape: tuple[int, int],
    segments: list[dict],
    markers: list[dict],
    main_colors: list[int],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for seg_idx, seg in enumerate(segments):
        for rr, cc in seg["coords_chain"]:
            out[int(rr), int(cc)] = int(main_colors[seg_idx])

    branch_specs = [
        {
            "marker": marker["marker_rel"],
            "side": marker["side"],
            "color": int(main_colors[marker["seg_idx"]]),
            "length": int(segments[marker["seg_idx"]]["length"]),
        }
        for marker in markers
    ]
    return _grow_branches(out, branch_specs)


def _segment_sorted_coords(seg: dict) -> list[tuple[int, int]]:
    coords = np.array(seg["coords_chain"], dtype=int)
    ordered, _orientation = _sort_segment_coords(coords)
    return ordered


def _sorted_end_indices(seg: dict) -> tuple[int, int]:
    sorted_coords = _segment_sorted_coords(seg)
    start_cell = seg["coords_chain"][0]
    start_end = 0 if start_cell == sorted_coords[0] else 1
    return int(start_end), int(1 - start_end)


def _object_cells(seg_coords: list[tuple[int, int]], marker_cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return list(seg_coords) + list(marker_cells)


def _can_place_object(
    cells: list[tuple[int, int]],
    delta: tuple[int, int],
    h: int,
    w: int,
    blocked: np.ndarray,
) -> bool:
    translated = _translate_cells(cells, delta)
    for rr, cc in translated:
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        if bool(blocked[rr, cc]):
            return False
    return True


def _stamp_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for rr, cc in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = int(rr + dr)
                nc = int(cc + dc)
                if 0 <= nr < h and 0 <= nc < w:
                    blocked[nr, nc] = True


def _build_input_from_model(
    segments: list[dict],
    markers: list[dict],
    bbox: tuple[int, int, int, int],
    output_cells: set[tuple[int, int]],
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray] | None:
    seg_count = len(segments)
    color_pool = PATH_COLORS[:]
    rng.shuffle(color_pool)
    assert seg_count + (seg_count - 1) <= len(color_pool)
    main_colors = color_pool[:seg_count]
    connector_colors = color_pool[seg_count : seg_count + seg_count - 1]

    r0, c0, r1, c1 = bbox
    reserve_h = int(r1 - r0 + 1)
    reserve_w = int(c1 - c0 + 1)
    extra_h = rng.randint(7, 12)
    extra_w = rng.randint(7, 12)
    h = int(max(18, r1 + 1 + extra_h))
    w = int(max(18, c1 + 1 + extra_w))
    if h > 32 or w > 32:
        return None

    expected = _paint_expected_output((h, w), segments, markers, main_colors)

    reserved = np.zeros((h, w), dtype=bool)
    for rr, cc in output_cells:
        reserved[int(rr), int(cc)] = True
    blocked = _dilate(reserved)

    root_seg = segments[0]
    root_bbox = _bbox(np.array(root_seg["coords_chain"], dtype=int))
    root_r0, root_c0, _root_r1, _root_c1 = root_bbox

    placed_seg_coords: list[list[tuple[int, int]]] = []
    placed_marker_coords: list[list[tuple[int, int]]] = []

    markers_by_seg: dict[int, list[dict]] = defaultdict(list)
    for marker in markers:
        markers_by_seg[int(marker["seg_idx"])].append(marker)

    for seg_idx, seg in enumerate(segments):
        final_seg_cells = list(seg["coords_chain"])
        final_marker_cells = [m["marker_rel"] for m in markers_by_seg.get(seg_idx, [])]

        if seg_idx == 0:
            placed_seg_coords.append(final_seg_cells)
            placed_marker_coords.append(final_marker_cells)
            continue

        obj_cells = _object_cells(final_seg_cells, final_marker_cells)
        obj_arr = np.array(obj_cells, dtype=int)
        or0, oc0, or1, oc1 = _bbox(obj_arr)
        oh = int(or1 - or0 + 1)
        ow = int(oc1 - oc0 + 1)

        placed = False
        for _ in range(120):
            band = rng.choice(["below", "right"])
            if band == "below":
                row_lo = int(r1 + 2)
                row_hi = int(h - oh)
                col_lo = 0
                col_hi = int(w - ow)
            else:
                row_lo = int(root_r0)
                row_hi = int(h - oh)
                col_lo = int(c1 + 2)
                col_hi = int(w - ow)

            if row_hi < row_lo or col_hi < col_lo:
                continue

            nr0 = rng.randint(row_lo, row_hi)
            nc0 = rng.randint(col_lo, col_hi)
            delta = (int(nr0 - or0), int(nc0 - oc0))

            if not _can_place_object(obj_cells, delta, h, w, blocked):
                continue

            translated_seg = _translate_cells(final_seg_cells, delta)
            translated_markers = _translate_cells(final_marker_cells, delta)
            placed_seg_coords.append(translated_seg)
            placed_marker_coords.append(translated_markers)
            _stamp_blocked(blocked, translated_seg + translated_markers)
            placed = True
            break

        if not placed:
            return None

    inp = np.full((h, w), BG, dtype=int)
    marker_pos_to_seg: dict[tuple[int, int], int] = {}

    for seg_idx, coords in enumerate(placed_seg_coords):
        for rr, cc in coords:
            inp[int(rr), int(cc)] = int(main_colors[seg_idx])

        start_end_idx, exit_end_idx = _sorted_end_indices(segments[seg_idx])
        sorted_coords = _segment_sorted_coords({"coords_chain": coords})

        if seg_idx > 0:
            prev_color = connector_colors[seg_idx - 1]
            inp[sorted_coords[start_end_idx]] = int(prev_color)
        if seg_idx < seg_count - 1:
            next_color = connector_colors[seg_idx]
            inp[sorted_coords[exit_end_idx]] = int(next_color)

        for marker_cell in placed_marker_coords[seg_idx]:
            inp[int(marker_cell[0]), int(marker_cell[1])] = MARKER
            marker_pos_to_seg[marker_cell] = int(seg_idx)

    try:
        oracle = main(inp)
    except Exception:
        return None
    if not np.array_equal(oracle, expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(500):
        rng = random.Random(random.randrange(1 << 30))
        segments = _build_chain_geometry(rng)
        if segments is None:
            continue

        markers = _choose_markers(segments, rng)
        if markers is None:
            continue

        shifted_segments, shifted_markers, bbox, output_cells = _shift_model(segments, markers, rng)
        built = _build_input_from_model(shifted_segments, shifted_markers, bbox, output_cells, rng)
        if built is None:
            continue

        inp, _expected = built
        if np.count_nonzero(inp == MARKER) == 0:
            continue
        return inp

    raise RuntimeError("generate_input: failed to build a valid sample")
