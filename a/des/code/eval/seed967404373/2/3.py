from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
MAGENTA = 6
SOURCE_COLORS = (1, 2, 3, 7, 8, 9)
MAX_GRID_SIZE = 30
GENERATION_RETRIES = 400


def _runs1d(mask_1d: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(mask_1d):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask_1d) - 1))
    return runs


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _detect_bars(mask: np.ndarray) -> tuple[list[dict], np.ndarray, np.ndarray]:
    h, w = mask.shape
    bars: list[dict] = []

    active_h: dict[tuple[int, int], int] = {}
    for r in range(h):
        segs = _runs1d(mask[r])
        segset = set(segs)
        for key, r0 in list(active_h.items()):
            if key in segset:
                continue
            c0, c1 = key
            r1 = r - 1
            thickness = r1 - r0 + 1
            length = c1 - c0 + 1
            if 1 <= thickness <= 3 and length > thickness:
                bars.append(
                    {"orient": "H", "r0": r0, "r1": r1, "c0": c0, "c1": c1, "width": thickness}
                )
            del active_h[key]
        for key in segs:
            active_h.setdefault(key, r)
    for key, r0 in active_h.items():
        c0, c1 = key
        r1 = h - 1
        thickness = r1 - r0 + 1
        length = c1 - c0 + 1
        if 1 <= thickness <= 3 and length > thickness:
            bars.append({"orient": "H", "r0": r0, "r1": r1, "c0": c0, "c1": c1, "width": thickness})

    active_v: dict[tuple[int, int], int] = {}
    for c in range(w):
        segs = _runs1d(mask[:, c])
        segset = set(segs)
        for key, c0 in list(active_v.items()):
            if key in segset:
                continue
            r0, r1 = key
            c1 = c - 1
            thickness = c1 - c0 + 1
            length = r1 - r0 + 1
            if 1 <= thickness <= 3 and length > thickness:
                bars.append(
                    {"orient": "V", "r0": r0, "r1": r1, "c0": c0, "c1": c1, "width": thickness}
                )
            del active_v[key]
        for key in segs:
            active_v.setdefault(key, c)
    for key, c0 in active_v.items():
        r0, r1 = key
        c1 = w - 1
        thickness = c1 - c0 + 1
        length = r1 - r0 + 1
        if 1 <= thickness <= 3 and length > thickness:
            bars.append({"orient": "V", "r0": r0, "r1": r1, "c0": c0, "c1": c1, "width": thickness})

    h_cover = np.zeros_like(mask, dtype=bool)
    v_cover = np.zeros_like(mask, dtype=bool)
    for bar_id, bar in enumerate(bars):
        bar["id"] = bar_id
        rr = slice(bar["r0"], bar["r1"] + 1)
        cc = slice(bar["c0"], bar["c1"] + 1)
        if bar["orient"] == "H":
            h_cover[rr, cc] = True
        else:
            v_cover[rr, cc] = True

    assert bars, "expected at least one pipe bar"
    return bars, h_cover, v_cover


def _parse_network(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    pipe = g == GREY
    assert pipe.any(), "expected grey pipe pixels"

    bars, h_cover, v_cover = _detect_bars(pipe)
    assert np.array_equal(pipe, h_cover | v_cover), "pipe pixels must be covered by detected bars"

    junction_mask = pipe & h_cover & v_cover
    junction_comps = _components4(junction_mask)

    node_kind: list[str] = []
    node_marker_color: list[int | None] = []
    node_marker_cell: list[tuple[int, int] | None] = []
    incident: dict[int, list[int]] = defaultdict(list)
    junction_node_by_cell: dict[tuple[int, int], int] = {}
    junction_bbox: dict[int, tuple[int, int, int, int]] = {}

    for comp in junction_comps:
        nid = len(node_kind)
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        node_kind.append("junction")
        node_marker_color.append(None)
        node_marker_cell.append(None)
        junction_bbox[nid] = (min(rows), min(cols), max(rows), max(cols))
        for cell in comp:
            junction_node_by_cell[cell] = nid

    def face_markers(face_cells: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int, int]]:
        marks: list[tuple[int, int, int]] = []
        seen: set[tuple[int, int]] = set()
        for r, c in face_cells:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]):
                continue
            if int(g[rr, cc]) in (BLACK, GREY):
                continue
            if (rr, cc) not in seen:
                seen.add((rr, cc))
                marks.append((rr, cc, int(g[rr, cc])))
        return marks

    def add_endpoint(marker_color: int, marker_cell: tuple[int, int]) -> int:
        nid = len(node_kind)
        node_kind.append("endpoint")
        node_marker_color.append(int(marker_color))
        node_marker_cell.append(marker_cell)
        return nid

    edges: list[dict] = []
    for bar in bars:
        events: list[tuple[float, int]] = []
        touched_junctions = {
            junction_node_by_cell[(r, c)]
            for r in range(bar["r0"], bar["r1"] + 1)
            for c in range(bar["c0"], bar["c1"] + 1)
            if (r, c) in junction_node_by_cell
        }
        for nid in touched_junctions:
            jr0, jc0, jr1, jc1 = junction_bbox[nid]
            if bar["orient"] == "H":
                events.append((((jc0 + jc1) / 2.0), nid))
            else:
                events.append((((jr0 + jr1) / 2.0), nid))

        if bar["orient"] == "H":
            left_face = [(r, bar["c0"]) for r in range(bar["r0"], bar["r1"] + 1)]
            if not any(cell in junction_node_by_cell for cell in left_face):
                marks = face_markers(left_face, 0, -1)
                assert len(marks) == 1, "each horizontal dead-end must have one marker"
                mr, mc, color = marks[0]
                events.append((bar["c0"] - 0.5, add_endpoint(color, (mr, mc))))

            right_face = [(r, bar["c1"]) for r in range(bar["r0"], bar["r1"] + 1)]
            if not any(cell in junction_node_by_cell for cell in right_face):
                marks = face_markers(right_face, 0, 1)
                assert len(marks) == 1, "each horizontal dead-end must have one marker"
                mr, mc, color = marks[0]
                events.append((bar["c1"] + 0.5, add_endpoint(color, (mr, mc))))
        else:
            top_face = [(bar["r0"], c) for c in range(bar["c0"], bar["c1"] + 1)]
            if not any(cell in junction_node_by_cell for cell in top_face):
                marks = face_markers(top_face, -1, 0)
                assert len(marks) == 1, "each vertical dead-end must have one marker"
                mr, mc, color = marks[0]
                events.append((bar["r0"] - 0.5, add_endpoint(color, (mr, mc))))

            bottom_face = [(bar["r1"], c) for c in range(bar["c0"], bar["c1"] + 1)]
            if not any(cell in junction_node_by_cell for cell in bottom_face):
                marks = face_markers(bottom_face, 1, 0)
                assert len(marks) == 1, "each vertical dead-end must have one marker"
                mr, mc, color = marks[0]
                events.append((bar["r1"] + 0.5, add_endpoint(color, (mr, mc))))

        events.sort()
        assert len(events) >= 2, "each bar must connect at least two events"
        for i in range(len(events) - 1):
            pos0, u = events[i]
            pos1, v = events[i + 1]
            assert pos0 < pos1, "bar events must be strictly ordered"
            eid = len(edges)
            edges.append({"u": u, "v": v, "width": int(bar["width"])})
            incident[u].append(eid)
            incident[v].append(eid)

    sources: list[int] = []
    terminals: list[int] = []
    for nid, kind in enumerate(node_kind):
        if kind != "endpoint":
            continue
        color = node_marker_color[nid]
        cell = node_marker_cell[nid]
        assert color is not None and cell is not None, "endpoint must carry a marker"
        assert len(incident[nid]) == 1, "source/terminal nodes must be dead-ends"
        if color == YELLOW:
            terminals.append(nid)
        else:
            sources.append(nid)

    assert sources, "expected at least one source"
    assert terminals, "expected at least one terminal"
    return {
        "node_kind": node_kind,
        "node_marker_color": node_marker_color,
        "node_marker_cell": node_marker_cell,
        "incident": incident,
        "edges": edges,
        "sources": sources,
        "terminals": terminals,
    }


def _propagate(parsed: dict) -> dict[int, set[int]]:
    arrivals: dict[int, set[int]] = defaultdict(set)
    edges = parsed["edges"]
    incident = parsed["incident"]
    node_kind = parsed["node_kind"]
    marker_colors = parsed["node_marker_color"]
    terminals = set(parsed["terminals"])

    for source in parsed["sources"]:
        color = int(marker_colors[source])
        stack = [(source, -1)]
        seen: set[tuple[int, int, int]] = set()
        while stack:
            node, prev_edge = stack.pop()
            state = (node, prev_edge, color)
            if state in seen:
                continue
            seen.add(state)

            if node in terminals:
                arrivals[node].add(color)

            options = [eid for eid in incident[node] if eid != prev_edge]
            if not options:
                continue
            max_width = max(int(edges[eid]["width"]) for eid in options)
            for eid in options:
                if int(edges[eid]["width"]) != max_width:
                    continue
                edge = edges[eid]
                nxt = int(edge["v"]) if int(edge["u"]) == node else int(edge["u"])
                stack.append((nxt, eid))

    return arrivals


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each source color through the grey pipe network.

    At every junction, the signal may continue into every outgoing branch whose
    width is maximal among the non-backtracking choices. Terminals hit by one
    color inherit that color; terminals hit by multiple colors become magenta.
    """
    g = np.asarray(grid, dtype=int)
    parsed = _parse_network(g)
    arrivals = _propagate(parsed)

    out = g.copy()
    for nid in parsed["terminals"]:
        colors = arrivals.get(nid)
        if not colors:
            continue
        rr, cc = parsed["node_marker_cell"][nid]
        out[rr, cc] = MAGENTA if len(colors) >= 2 else next(iter(colors))
    return out


def _draw_h(grid: np.ndarray, top: int, left: int, length: int, width: int) -> None:
    grid[top : top + width, left : left + length] = GREY


def _draw_v(grid: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    grid[top : top + height, left : left + width] = GREY


def _h_marker_row(top: int, width: int) -> int:
    return top + width // 2


def _v_marker_col(left: int, width: int) -> int:
    return left + width // 2


def _choose_attach_positions(main_length: int, widths: list[int], end_margin: int = 3, gap: int = 3) -> list[int] | None:
    for _ in range(200):
        starts: list[int | None] = [None] * len(widths)
        intervals: list[tuple[int, int]] = []
        order = list(range(len(widths)))
        random.shuffle(order)
        ok = True
        for idx in order:
            width = widths[idx]
            lo = end_margin
            hi = main_length - end_margin - width
            if hi < lo:
                ok = False
                break
            candidates = list(range(lo, hi + 1))
            random.shuffle(candidates)
            chosen = None
            for start in candidates:
                end = start + width - 1
                if all(end + gap < prev_start or start > prev_end + gap for prev_start, prev_end in intervals):
                    chosen = start
                    intervals.append((start, end))
                    starts[idx] = start
                    break
            if chosen is None:
                ok = False
                break
        if ok:
            return [int(v) for v in starts]
    return None


def _pad_and_transform(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k = random.randrange(4)
    inp = np.rot90(inp, k).copy()
    out = np.rot90(out, k).copy()
    if random.random() < 0.5:
        inp = np.fliplr(inp).copy()
        out = np.fliplr(out).copy()
    if random.random() < 0.5:
        inp = np.flipud(inp).copy()
        out = np.flipud(out).copy()

    h, w = inp.shape
    extra_h = random.randint(0, min(6, MAX_GRID_SIZE - h))
    extra_w = random.randint(0, min(6, MAX_GRID_SIZE - w))
    top = random.randint(0, extra_h)
    left = random.randint(0, extra_w)
    bottom = extra_h - top
    right = extra_w - left

    padded_in = np.zeros((h + top + bottom, w + left + right), dtype=int)
    padded_out = np.zeros_like(padded_in)
    padded_in[top : top + h, left : left + w] = inp
    padded_out[top : top + h, left : left + w] = out
    return padded_in, padded_out


def _family_unique() -> tuple[np.ndarray, np.ndarray] | None:
    main_width = random.randint(2, 3)
    main_length = random.randint(13, 17)
    branch_count = random.randint(1, 2)
    branch_widths = [random.randint(1, main_width - 1) for _ in range(branch_count)]
    branch_lengths = [main_width + random.randint(4, 8) for _ in range(branch_count)]
    branch_sides = [random.choice(("up", "down")) for _ in range(branch_count)]
    attach = _choose_attach_positions(main_length, branch_widths)
    if attach is None:
        return None

    up_extra = max(
        [branch_lengths[i] - main_width for i in range(branch_count) if branch_sides[i] == "up"] + [0]
    )
    down_extra = max(
        [branch_lengths[i] - main_width for i in range(branch_count) if branch_sides[i] == "down"] + [0]
    )
    height = up_extra + main_width + down_extra + 4
    width = main_length + 4
    inp = np.zeros((height, width), dtype=int)
    out = np.zeros_like(inp)

    main_top = 2 + up_extra
    main_left = 2
    _draw_h(inp, main_top, main_left, main_length, main_width)
    _draw_h(out, main_top, main_left, main_length, main_width)

    source = random.choice(SOURCE_COLORS)
    row = _h_marker_row(main_top, main_width)
    inp[row, main_left - 1] = source
    out[row, main_left - 1] = source
    inp[row, main_left + main_length] = YELLOW
    out[row, main_left + main_length] = source

    for i in range(branch_count):
        branch_left = main_left + attach[i]
        branch_width = branch_widths[i]
        branch_height = branch_lengths[i]
        if branch_sides[i] == "up":
            branch_top = main_top - (branch_height - main_width)
            term_row = branch_top - 1
        else:
            branch_top = main_top
            term_row = branch_top + branch_height
        _draw_v(inp, branch_top, branch_left, branch_height, branch_width)
        _draw_v(out, branch_top, branch_left, branch_height, branch_width)
        term_col = _v_marker_col(branch_left, branch_width)
        inp[term_row, term_col] = YELLOW
        out[term_row, term_col] = YELLOW

    return _pad_and_transform(inp, out)


def _family_split() -> tuple[np.ndarray, np.ndarray] | None:
    main_width = random.randint(1, 3)
    main_length = random.randint(12, 16)
    stem_width = random.randint(1, 3)
    stem_height = main_width + random.randint(4, 8)
    extra_width = random.randint(1, main_width - 1) if main_width > 1 and random.random() < 0.5 else None
    widths = [stem_width] + ([extra_width] if extra_width is not None else [])
    attach = _choose_attach_positions(main_length, widths)
    if attach is None:
        return None

    stem_side = random.choice(("up", "down"))
    extra_side = random.choice(("up", "down")) if extra_width is not None else None
    extra_height = main_width + random.randint(4, 7) if extra_width is not None else None
    up_extra = max(
        [
            stem_height - main_width if stem_side == "up" else 0,
            extra_height - main_width if extra_width is not None and extra_side == "up" else 0,
            0,
        ]
    )
    down_extra = max(
        [
            stem_height - main_width if stem_side == "down" else 0,
            extra_height - main_width if extra_width is not None and extra_side == "down" else 0,
            0,
        ]
    )
    height = up_extra + main_width + down_extra + 4
    width = main_length + 4
    inp = np.zeros((height, width), dtype=int)
    out = np.zeros_like(inp)

    main_top = 2 + up_extra
    main_left = 2
    _draw_h(inp, main_top, main_left, main_length, main_width)
    _draw_h(out, main_top, main_left, main_length, main_width)

    source = random.choice(SOURCE_COLORS)
    left_row = _h_marker_row(main_top, main_width)
    inp[left_row, main_left - 1] = YELLOW
    out[left_row, main_left - 1] = source
    inp[left_row, main_left + main_length] = YELLOW
    out[left_row, main_left + main_length] = source

    stem_left = main_left + attach[0]
    if stem_side == "up":
        stem_top = main_top - (stem_height - main_width)
        stem_mark_row = stem_top - 1
    else:
        stem_top = main_top
        stem_mark_row = stem_top + stem_height
    _draw_v(inp, stem_top, stem_left, stem_height, stem_width)
    _draw_v(out, stem_top, stem_left, stem_height, stem_width)
    stem_mark_col = _v_marker_col(stem_left, stem_width)
    inp[stem_mark_row, stem_mark_col] = source
    out[stem_mark_row, stem_mark_col] = source

    if extra_width is not None:
        extra_left = main_left + attach[1]
        if extra_side == "up":
            extra_top = main_top - (extra_height - main_width)
            extra_mark_row = extra_top - 1
        else:
            extra_top = main_top
            extra_mark_row = extra_top + extra_height
        _draw_v(inp, extra_top, extra_left, extra_height, extra_width)
        _draw_v(out, extra_top, extra_left, extra_height, extra_width)
        extra_mark_col = _v_marker_col(extra_left, extra_width)
        inp[extra_mark_row, extra_mark_col] = YELLOW
        out[extra_mark_row, extra_mark_col] = YELLOW

    return _pad_and_transform(inp, out)


def _family_collision() -> tuple[np.ndarray, np.ndarray] | None:
    main_width = random.randint(2, 3)
    main_length = random.randint(13, 17)
    stem_width = random.randint(1, main_width - 1)
    stem_height = main_width + random.randint(4, 8)
    extra_width = random.randint(1, main_width - 1) if random.random() < 0.5 else None
    widths = [stem_width] + ([extra_width] if extra_width is not None else [])
    attach = _choose_attach_positions(main_length, widths)
    if attach is None:
        return None

    stem_side = random.choice(("up", "down"))
    extra_side = random.choice(("up", "down")) if extra_width is not None else None
    extra_height = main_width + random.randint(4, 7) if extra_width is not None else None
    up_extra = max(
        [
            stem_height - main_width if stem_side == "up" else 0,
            extra_height - main_width if extra_width is not None and extra_side == "up" else 0,
            0,
        ]
    )
    down_extra = max(
        [
            stem_height - main_width if stem_side == "down" else 0,
            extra_height - main_width if extra_width is not None and extra_side == "down" else 0,
            0,
        ]
    )
    height = up_extra + main_width + down_extra + 4
    width = main_length + 4
    inp = np.zeros((height, width), dtype=int)
    out = np.zeros_like(inp)

    main_top = 2 + up_extra
    main_left = 2
    _draw_h(inp, main_top, main_left, main_length, main_width)
    _draw_h(out, main_top, main_left, main_length, main_width)

    color_a, color_b = random.sample(SOURCE_COLORS, 2)
    row = _h_marker_row(main_top, main_width)
    inp[row, main_left - 1] = color_a
    out[row, main_left - 1] = color_a
    inp[row, main_left + main_length] = YELLOW
    out[row, main_left + main_length] = MAGENTA

    stem_left = main_left + attach[0]
    if stem_side == "up":
        stem_top = main_top - (stem_height - main_width)
        stem_mark_row = stem_top - 1
    else:
        stem_top = main_top
        stem_mark_row = stem_top + stem_height
    _draw_v(inp, stem_top, stem_left, stem_height, stem_width)
    _draw_v(out, stem_top, stem_left, stem_height, stem_width)
    stem_mark_col = _v_marker_col(stem_left, stem_width)
    inp[stem_mark_row, stem_mark_col] = color_b
    out[stem_mark_row, stem_mark_col] = color_b

    if extra_width is not None:
        extra_left = main_left + attach[1]
        if extra_side == "up":
            extra_top = main_top - (extra_height - main_width)
            extra_mark_row = extra_top - 1
        else:
            extra_top = main_top
            extra_mark_row = extra_top + extra_height
        _draw_v(inp, extra_top, extra_left, extra_height, extra_width)
        _draw_v(out, extra_top, extra_left, extra_height, extra_width)
        extra_mark_col = _v_marker_col(extra_left, extra_width)
        inp[extra_mark_row, extra_mark_col] = YELLOW
        out[extra_mark_row, extra_mark_col] = YELLOW

    return _pad_and_transform(inp, out)


def _gen_attempt() -> np.ndarray | None:
    family = random.choice((_family_unique, _family_split, _family_collision))
    result = family()
    if result is None:
        return None
    inp, expected = result
    if inp.shape[0] > MAX_GRID_SIZE or inp.shape[1] > MAX_GRID_SIZE:
        return None
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pipe-routing input")
