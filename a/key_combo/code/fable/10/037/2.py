from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 600


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _inner_bbox(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return r0 + 1, c0 + 1, r1 - 1, c1 - 1


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _frame_kind(box: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = box
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h == w:
        return "S"
    return "H" if w > h else "V"


def _color_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _is_rectangular_frame(coords: list[tuple[int, int]]) -> bool:
    if len(coords) < 8:
        return False
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    border_area = 2 * (h + w) - 4
    if len(coords) != border_area:
        return False
    coord_set = set(coords)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if on_border != ((r, c) in coord_set):
                return False
    return True


def _find_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    colors = sorted({int(x) for x in np.unique(g)} - {BG})
    frames: list[dict] = []
    for color in colors:
        for coords in _color_components(g, color):
            if not _is_rectangular_frame(coords):
                continue
            box = _bbox_from_coords(coords)
            frames.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": box,
                    "inner": _inner_bbox(box),
                    "kind": _frame_kind(box),
                }
            )
    assert frames, "expected at least one hollow rectangular frame"
    frames.sort(key=lambda fr: fr["bbox"])
    return frames


def _assign_destinations(frames: list[dict]) -> list[int | None]:
    dests: list[int | None] = []
    for i, src in enumerate(frames):
        kind = src["kind"]
        if kind == "S":
            dests.append(None)
            continue

        sr0, sc0, sr1, sc1 = src["bbox"]
        sir0, sic0, sir1, sic1 = src["inner"]
        candidates: list[tuple[int, int]] = []
        for j, dst in enumerate(frames):
            if i == j:
                continue
            dr0, dc0, dr1, dc1 = dst["bbox"]
            dir0, dic0, dir1, dic1 = dst["inner"]
            if kind == "H":
                if dc0 <= sc1:
                    continue
                if not _intervals_overlap(sir0, sir1, dir0, dir1):
                    continue
                candidates.append((dc0 - sc1, j))
            else:
                if dr0 <= sr1:
                    continue
                if not _intervals_overlap(sic0, sic1, dic0, dic1):
                    continue
                candidates.append((dr0 - sr1, j))

        assert candidates, f"frame {src['bbox']} has no aligned successor"
        best_gap = min(gap for gap, _ in candidates)
        best = [j for gap, j in candidates if gap == best_gap]
        assert len(best) == 1, f"frame {src['bbox']} has ambiguous successor"
        dests.append(best[0])
    return dests


def _sink_index(start: int, dests: list[int | None]) -> int:
    seen: set[int] = set()
    cur = start
    while dests[cur] is not None:
        assert cur not in seen, "routing graph must be acyclic"
        seen.add(cur)
        cur = int(dests[cur])
    return cur


def _frame_border_mask(shape: tuple[int, int], frames: list[dict]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for fr in frames:
        for r, c in fr["coords"]:
            mask[r, c] = True
    return mask


def _motifs_by_frame(grid: np.ndarray, frames: list[dict]) -> list[list[tuple[int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    border_mask = _frame_border_mask(g.shape, frames)
    motifs: list[list[tuple[int, int, int]]] = [[] for _ in frames]

    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            color = int(g[r, c])
            if color == BG or border_mask[r, c]:
                continue
            owners = []
            for i, fr in enumerate(frames):
                ir0, ic0, ir1, ic1 = fr["inner"]
                if ir0 <= r <= ir1 and ic0 <= c <= ic1:
                    owners.append(i)
            assert len(owners) == 1, f"motif pixel {(r, c)} must belong to exactly one frame"
            owner = owners[0]
            ir0, ic0, _ir1, _ic1 = frames[owner]["inner"]
            motifs[owner].append((r - ir0, c - ic0, color))
    return motifs


def main(grid: np.ndarray) -> np.ndarray:
    """Route each interior motif through aligned rectangular frames until it stops in a square frame.

    Input structure:
    - The background is black.
    - Several 1-cell-thick monochrome hollow rectangles appear on the canvas; some are wide,
      some tall, and some square.
    - Some frame interiors contain small colored motifs, while other interiors are empty.

    Rule:
    - A wide frame sends its entire interior motif to the nearest frame strictly to its right
      whose vertical span overlaps it; a tall frame sends its motif to the nearest frame
      strictly below whose horizontal span overlaps it.
    - Motifs keep the same offsets relative to the destination frame's inner top-left corner,
      then the destination frame's own aspect ratio determines the next hop.
    - A square frame is a sink: any motif that reaches it stays there. Clear all original motif
      locations, keep every frame border, and draw motifs only inside their final square sinks.

    Key insight:
    - The rectangles are routers rather than containers: each non-square frame contributes one
      mandatory hop, so the correct sink is determined by following the whole right/down chain.
    """
    g = np.asarray(grid, dtype=int)
    frames = _find_frames(g)
    dests = _assign_destinations(frames)
    motifs = _motifs_by_frame(g, frames)

    out = np.zeros_like(g)
    for fr in frames:
        for r, c in fr["coords"]:
            out[r, c] = int(fr["color"])

    for i, motif in enumerate(motifs):
        if not motif:
            continue
        sink = frames[_sink_index(i, dests)]
        sir0, sic0, sir1, sic1 = sink["inner"]
        for dr, dc, color in motif:
            rr = sir0 + dr
            cc = sic0 + dc
            assert sir0 <= rr <= sir1 and sic0 <= cc <= sic1, "motif must fit every routed frame"
            assert int(out[rr, cc]) == BG, "constructed examples must avoid sink overlap"
            out[rr, cc] = int(color)

    return out


def _node_size(kind: str, core: int) -> tuple[int, int]:
    if kind == "H":
        return core + 2, core + 4
    if kind == "V":
        return core + 4, core + 2
    if kind == "S":
        return core + 4, core + 4
    raise ValueError(kind)


def _outer_box(r0: int, c0: int, h: int, w: int) -> tuple[int, int, int, int]:
    return r0, c0, r0 + h - 1, c0 + w - 1


def _boxes_clear(
    used: list[tuple[int, int, int, int]], box: tuple[int, int, int, int], pad: int = 1
) -> bool:
    r0, c0, r1, c1 = box
    for ur0, uc0, ur1, uc1 in used:
        if max(r0 - pad, ur0 - pad) <= min(r1 + pad, ur1 + pad) and max(c0 - pad, uc0 - pad) <= min(c1 + pad, uc1 + pad):
            return False
    return True


def _place_sinks(h: int, w: int, core: int) -> list[tuple[int, int, int, int]] | None:
    sh, sw = _node_size("S", core)
    for _ in range(120):
        left_col_max = max(12, w - 2 * sw - 8)
        s0_c = random.randint(9, left_col_max)
        s1_c_min = s0_c + sw + 4
        s1_c_max = w - sw - 2
        if s1_c_min > s1_c_max:
            continue
        s1_c = random.randint(s1_c_min, s1_c_max)
        s0_r = random.randint(9, h - sh - 2)
        s1_r = random.randint(9, h - sh - 2)
        box0 = _outer_box(s0_r, s0_c, sh, sw)
        box1 = _outer_box(s1_r, s1_c, sh, sw)
        if _boxes_clear([], box0, pad=2) and _boxes_clear([box0], box1, pad=2):
            return [box0, box1]
    return None


def _try_place_parent(
    kind: str,
    child_box: tuple[int, int, int, int],
    used: list[tuple[int, int, int, int]],
    canvas_h: int,
    canvas_w: int,
    core: int,
) -> tuple[int, int, int, int] | None:
    ph, pw = _node_size(kind, core)
    cr0, cc0, cr1, cc1 = child_box
    for _ in range(120):
        gap = random.randint(2, 5)
        if kind == "H":
            c0 = cc0 - gap - pw
            if c0 < 1:
                continue
            row_lo = max(1, cr0 - ph + 3)
            row_hi = min(canvas_h - ph - 1, cr1 - 2)
            if row_lo > row_hi:
                continue
            r0 = random.randint(row_lo, row_hi)
        else:
            r0 = cr0 - gap - ph
            if r0 < 1:
                continue
            col_lo = max(1, cc0 - pw + 3)
            col_hi = min(canvas_w - pw - 1, cc1 - 2)
            if col_lo > col_hi:
                continue
            c0 = random.randint(col_lo, col_hi)
        box = _outer_box(r0, c0, ph, pw)
        if _boxes_clear(used, box, pad=1):
            return box
    return None


def _render_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _render_from_layout(layout: dict, mode: str) -> np.ndarray:
    h = layout["grid_shape"][0]
    w = layout["grid_shape"][1]
    grid = np.zeros((h, w), dtype=int)
    for node in layout["nodes"].values():
        _render_frame(grid, node["bbox"], node["color"])

    for source_name, motif in layout["motifs"].items():
        if mode == "input":
            dest_name = source_name
        elif mode == "first_hop":
            node = layout["nodes"][source_name]
            dest_name = node["dest"] if node["dest"] is not None else source_name
        elif mode == "output":
            dest_name = layout["sink_by_source"][source_name]
        else:
            raise ValueError(mode)
        ir0, ic0, _ir1, _ic1 = _inner_bbox(layout["nodes"][dest_name]["bbox"])
        for dr, dc, color in motif:
            grid[ir0 + dr, ic0 + dc] = int(color)
    return grid


def _layout_actual_destinations(layout: dict) -> dict[str, str | None]:
    nodes = list(layout["nodes"].items())
    boxes = [node["bbox"] for _, node in nodes]
    frames = [
        {
            "bbox": box,
            "inner": _inner_bbox(box),
            "kind": node["kind"],
        }
        for (_, node), box in zip(nodes, boxes)
    ]
    raw_dests = _assign_destinations(frames)
    name_by_idx = {i: name for i, (name, _) in enumerate(nodes)}
    actual: dict[str, str | None] = {}
    for i, (name, _node) in enumerate(nodes):
        actual[name] = None if raw_dests[i] is None else name_by_idx[int(raw_dests[i])]
    return actual


def _layout_sink(name: str, layout: dict) -> str:
    seen: set[str] = set()
    cur = name
    while layout["nodes"][cur]["dest"] is not None:
        assert cur not in seen
        seen.add(cur)
        cur = str(layout["nodes"][cur]["dest"])
    return cur


def _validate_layout(layout: dict) -> None:
    actual = _layout_actual_destinations(layout)
    for name, node in layout["nodes"].items():
        assert actual[name] == node["dest"], f"geometry mismatch for {name}"

    sources = list(layout["motifs"])
    assert sources, "need at least one source motif"
    assert any(layout["nodes"][src]["kind"] == "H" for src in sources), "need a wide-frame source"
    assert any(layout["nodes"][src]["kind"] == "V" for src in sources), "need a tall-frame source"

    sink_counts = Counter(layout["sink_by_source"][src] for src in sources)
    assert len(sink_counts) >= 2, "motifs must end in at least two distinct square sinks"

    def path_len(src: str) -> int:
        steps = 0
        cur = src
        while layout["nodes"][cur]["dest"] is not None:
            steps += 1
            cur = str(layout["nodes"][cur]["dest"])
        return steps

    assert any(path_len(src) >= 2 for src in sources), "need a redirected multi-hop chain"

    expected = _render_from_layout(layout, mode="output")
    first_hop = _render_from_layout(layout, mode="first_hop")
    assert not np.array_equal(expected, first_hop), "one-hop shortcut must fail"

    for source_name, motif in layout["motifs"].items():
        cur = source_name
        while True:
            ir0, ic0, ir1, ic1 = _inner_bbox(layout["nodes"][cur]["bbox"])
            for dr, dc, _color in motif:
                assert ir0 + dr <= ir1 and ic0 + dc <= ic1, "motif must fit routed frame"
            nxt = layout["nodes"][cur]["dest"]
            if nxt is None:
                break
            cur = str(nxt)


def _build_layout() -> dict | None:
    core = 2
    grid_h = random.randint(27, 30)
    grid_w = random.randint(27, 30)
    h_h, w_h = _node_size("H", core)
    h_v, w_v = _node_size("V", core)
    h_s, w_s = _node_size("S", core)

    min_sink_r = max(10, h_v + 4)
    if min_sink_r > grid_h - h_s - 2:
        return None

    for _ in range(120):
        sink1_c = grid_w - w_s - random.randint(1, 2)
        sink0_c_min = max(9, w_h + 3)
        sink0_c_max = sink1_c - w_s - 3
        if sink0_c_min > sink0_c_max:
            continue
        sink0_c = random.randint(sink0_c_min, sink0_c_max)
        sink0_r = random.randint(min_sink_r, grid_h - h_s - 2)
        sink1_r = random.randint(min_sink_r, grid_h - h_s - 2)

        sink0 = _outer_box(sink0_r, sink0_c, h_s, w_s)
        sink1 = _outer_box(sink1_r, sink1_c, h_s, w_s)
        used = [sink0, sink1]
        if not _boxes_clear([sink0], sink1, pad=2):
            continue

        gap_a = random.randint(2, 4)
        gap_b = random.randint(2, 4)
        gap_c = random.randint(2, 4)
        gap_d = random.randint(2, 4)

        v0_c = sink0_c + random.randint(0, w_s - w_v)
        v0_r = sink0_r - gap_a - h_v
        h0_r = v0_r + random.randint(0, max(0, h_v - h_h - 1))
        h0_c = v0_c - gap_b - w_h

        h1_r = sink1_r + 1
        h1_c = sink1_c - gap_c - w_h
        v1_r = h1_r - gap_d - h_v
        v1_c = h1_c + random.randint(0, w_h - w_v)

        base_boxes = {
            "v0": _outer_box(v0_r, v0_c, h_v, w_v),
            "h0": _outer_box(h0_r, h0_c, h_h, w_h),
            "h1": _outer_box(h1_r, h1_c, h_h, w_h),
            "v1": _outer_box(v1_r, v1_c, h_v, w_v),
        }
        if any(r0 < 1 or c0 < 1 or r1 >= grid_h - 1 or c1 >= grid_w - 1 for r0, c0, r1, c1 in base_boxes.values()):
            continue
        if any(not _boxes_clear(used + [box for name, box in base_boxes.items() if name != cur], base_boxes[cur], pad=1) for cur in base_boxes):
            continue

        nodes: dict[str, dict] = {
            "sink0": {"kind": "S", "bbox": sink0, "dest": None},
            "sink1": {"kind": "S", "bbox": sink1, "dest": None},
            "v0": {"kind": "V", "bbox": base_boxes["v0"], "dest": "sink0"},
            "h0": {"kind": "H", "bbox": base_boxes["h0"], "dest": "v0"},
            "h1": {"kind": "H", "bbox": base_boxes["h1"], "dest": "sink1"},
            "v1": {"kind": "V", "bbox": base_boxes["v1"], "dest": "h1"},
        }
        sources = ["h0", "v1"]

        extra_choice = random.choice(("none", "h_to_sink0", "v_to_sink1"))
        if extra_choice == "h_to_sink0":
            gap_e = random.randint(2, 4)
            extra_box = _outer_box(sink0_r + 1, sink0_c - gap_e - w_h, h_h, w_h)
            if not (extra_box[0] >= 1 and extra_box[1] >= 1 and extra_box[2] < grid_h - 1 and extra_box[3] < grid_w - 1):
                continue
            if not _boxes_clear([node["bbox"] for node in nodes.values()], extra_box, pad=1):
                continue
            nodes["h2"] = {"kind": "H", "bbox": extra_box, "dest": "sink0"}
            sources.append("h2")
        elif extra_choice == "v_to_sink1":
            gap_e = random.randint(2, 4)
            extra_box = _outer_box(sink1_r - gap_e - h_v, sink1_c + (w_s - w_v), h_v, w_v)
            if not (extra_box[0] >= 1 and extra_box[1] >= 1 and extra_box[2] < grid_h - 1 and extra_box[3] < grid_w - 1):
                continue
            if not _boxes_clear([node["bbox"] for node in nodes.values()], extra_box, pad=1):
                continue
            nodes["v2"] = {"kind": "V", "bbox": extra_box, "dest": "sink1"}
            sources.append("v2")

        break
    else:
        return None

    frame_palette = random.sample(PALETTE, k=3)
    remaining = [c for c in PALETTE if c not in frame_palette]
    motif_palette = random.sample(remaining, k=min(4, len(remaining)))
    for node in nodes.values():
        node["color"] = random.choice(frame_palette)

    sink_by_source = {src: _layout_sink(src, {"nodes": nodes}) for src in sources}

    motifs: dict[str, list[tuple[int, int, int]]] = {}
    sink_slots = {name: [(r, c) for r in range(core) for c in range(core)] for name in ("sink0", "sink1")}
    for slots in sink_slots.values():
        random.shuffle(slots)

    for src in sources:
        sink = sink_by_source[src]
        max_cells = 1 if core == 2 else 2
        count = 1 if len(sink_slots[sink]) == 1 else random.randint(1, min(max_cells, len(sink_slots[sink])))
        picked = [sink_slots[sink].pop() for _ in range(count)]
        colors = random.sample(motif_palette, k=min(count, len(motif_palette)))
        while len(colors) < count:
            colors.append(random.choice(motif_palette))
        motifs[src] = [(dr, dc, color) for (dr, dc), color in zip(picked, colors)]

    layout = {
        "grid_shape": (grid_h, grid_w),
        "core": core,
        "nodes": nodes,
        "motifs": motifs,
        "sink_by_source": sink_by_source,
    }

    _validate_layout(layout)
    return layout


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    layout = _build_layout()
    if layout is None:
        return None
    inp = _render_from_layout(layout, mode="input")
    expected = _render_from_layout(layout, mode="output")

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        case = _build_case()
        if case is None:
            continue
        inp, _expected = case
        return inp
    raise AssertionError("failed to generate a valid aspect-ratio-routing input")
