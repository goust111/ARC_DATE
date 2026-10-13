from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
NONBLACK = tuple(range(1, 10))

Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

_GEN_RETRY_LIMIT = 700
_GEN_CALL_COUNT = 0
_GEN_SEEN_INPUTS: set[tuple[tuple[int, int], bytes]] = set()
_GEN_SEEN_OUTPUTS: set[tuple[tuple[int, int], bytes]] = set()


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    out: list[np.ndarray] = []
    for r0_raw, c0_raw in np.argwhere(mask):
        r0 = int(r0_raw)
        c0 = int(c0_raw)
        if seen[r0, c0]:
            continue
        cells: list[Coord] = []
        queue: deque[Coord] = deque([(r0, c0)])
        seen[r0, c0] = True
        while queue:
            r, c = queue.popleft()
            cells.append((r, c))
            for rr, cc in _neighbors4(r, c):
                if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]) and not seen[rr, cc]:
                    seen[rr, cc] = True
                    queue.append((rr, cc))
        out.append(np.array(cells, dtype=int))
    return out


def _bbox(coords: np.ndarray | list[Coord]) -> BBox:
    arr = np.asarray(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _border_cells(b: BBox) -> set[Coord]:
    r0, c0, r1, c1 = b
    cells: set[Coord] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _gap_side_and_outward(b: BBox, cell: Coord) -> tuple[str, Coord] | None:
    r0, c0, r1, c1 = b
    r, c = cell
    is_corner = r in (r0, r1) and c in (c0, c1)
    if is_corner:
        return None
    if r == r0 and c0 < c < c1:
        return "top", (-1, 0)
    if r == r1 and c0 < c < c1:
        return "bottom", (1, 0)
    if c == c0 and r0 < r < r1:
        return "left", (0, -1)
    if c == c1 and r0 < r < r1:
        return "right", (0, 1)
    return None


def _frame_from_component(g: np.ndarray, coords: np.ndarray, color: int, bg: int) -> dict | None:
    b = _bbox(coords)
    r0, c0, r1, c1 = b
    if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 4:
        return None

    present = {(int(r), int(c)) for r, c in coords}
    border = _border_cells(b)
    if not present <= border:
        return None

    missing = sorted(border - present)
    if len(missing) not in (0, 1, 2):
        return None

    gaps: list[dict] = []
    for pos in missing:
        if int(g[pos]) != int(bg):
            return None
        side_out = _gap_side_and_outward(b, pos)
        if side_out is None:
            return None
        side, outward = side_out
        gaps.append({"pos": pos, "side": side, "out": outward, "color": int(color)})

    if len(gaps) == 2:
        a, b_gap = gaps
        ar, ac = a["out"]
        br, bc = b_gap["out"]
        if (ar, ac) != (-br, -bc):
            return None
        if ar and a["pos"][1] != b_gap["pos"][1]:
            return None
        if ac and a["pos"][0] != b_gap["pos"][0]:
            return None

    return {"bbox": b, "color": int(color), "cells": present, "gaps": gaps}


def _find_frames(g: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(set(map(int, g.flat)) - {int(bg)}):
        for coords in _components(g == int(color)):
            frame = _frame_from_component(g, coords, int(color), int(bg))
            if frame is not None:
                frames.append(frame)

    frames.sort(key=lambda f: (f["bbox"][0], f["bbox"][1], f["bbox"][2], f["bbox"][3]))
    occupied: set[Coord] = set()
    for frame in frames:
        overlap = occupied & frame["cells"]
        assert not overlap, "frames must not overlap"
        occupied.update(frame["cells"])
    return frames


def _inside_bbox(coords: np.ndarray, b: BBox) -> bool:
    r0, c0, r1, c1 = b
    rows = coords[:, 0]
    cols = coords[:, 1]
    return bool(np.all((r0 < rows) & (rows < r1) & (c0 < cols) & (cols < c1)))


def _object_records(g: np.ndarray, frame_cells: set[Coord], bg: int) -> list[dict]:
    mask = g != int(bg)
    for r, c in frame_cells:
        mask[r, c] = False

    records: list[dict] = []
    for coords in _components(mask):
        values = np.array([int(g[int(r), int(c)]) for r, c in coords], dtype=int)
        records.append({"coords": coords, "values": values})
    records.sort(key=lambda o: _bbox(o["coords"]))
    return records


def _aligned_with_gap(coords: np.ndarray, gap: dict) -> bool:
    gr, gc = gap["pos"]
    dr, dc = gap["out"]
    if dr:
        return bool(np.all(coords[:, 1] == int(gc)))
    assert dc
    return bool(np.all(coords[:, 0] == int(gr)))


def _gap_crossed_by_direction(gap: dict, direction: Coord) -> bool:
    dr, dc = direction
    gr, gc = gap["out"]
    return (dr, dc) == (gr, gc) or (dr, dc) == (-gr, -gc)


def _classify_movers(frames: list[dict], objects: list[dict]) -> list[dict]:
    by_frame: dict[int, list[dict]] = {i: [] for i in range(len(frames))}
    for obj in objects:
        containing = [i for i, frame in enumerate(frames) if _inside_bbox(obj["coords"], frame["bbox"])]
        assert len(containing) <= 1, "objects must not be ambiguously inside nested frames"
        if containing:
            by_frame[containing[0]].append(obj)

    movers: list[dict] = []
    for idx, frame in enumerate(frames):
        gaps = frame["gaps"]
        contained = by_frame[idx]
        if not gaps or not contained:
            continue
        assert len(contained) == 1, "an open starting frame must contain exactly one object"
        obj = contained[0]
        candidates = [gap for gap in gaps if _aligned_with_gap(obj["coords"], gap)]
        assert len(candidates) == 1, "the contained object must identify a unique exit gap"
        gap = candidates[0]
        movers.append({"frame": frame, "object": obj, "gap": gap, "dir": gap["out"]})

    return movers


def _solve_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = BLACK
    assert np.any(g == bg), "expected a black background"

    frames = _find_frames(g, bg)
    assert frames, "expected at least one hollow rectangle frame"
    frame_cells: set[Coord] = set()
    gap_by_pos: dict[Coord, dict] = {}
    for frame in frames:
        frame_cells.update(frame["cells"])
        for gap in frame["gaps"]:
            pos = gap["pos"]
            assert pos not in gap_by_pos, "two frame gaps must not share one cell"
            gap_by_pos[pos] = gap

    objects = _object_records(g, frame_cells, bg)
    movers = _classify_movers(frames, objects)
    assert movers, "expected at least one aligned object in an open frame"

    base = g.copy()
    for mover in movers:
        coords = mover["object"]["coords"]
        base[coords[:, 0], coords[:, 1]] = bg

    h, w = g.shape
    finals: list[tuple[np.ndarray, np.ndarray]] = []
    closed: dict[Coord, int] = {}

    for mover in movers:
        direction = mover["dir"]
        dr, dc = direction
        current = [(int(r), int(c)) for r, c in mover["object"]["coords"]]
        passed: set[Coord] = set()
        moved = False

        while True:
            nxt = [(r + dr, c + dc) for r, c in current]
            if any(not (0 <= r < h and 0 <= c < w) for r, c in nxt):
                break
            if any(int(base[r, c]) != bg for r, c in nxt):
                break

            current = nxt
            moved = True
            for pos in current:
                gap = gap_by_pos.get(pos)
                if gap is not None and _gap_crossed_by_direction(gap, direction):
                    passed.add(pos)

        assert moved, "each active object must be able to move"
        assert mover["gap"]["pos"] in passed, "active object must pass through its starting door"
        for pos in passed:
            closed[pos] = int(gap_by_pos[pos]["color"])

        finals.append((np.array(current, dtype=int), mover["object"]["values"]))

    out = base.copy()
    for (r, c), color in closed.items():
        out[r, c] = int(color)

    occupied: set[Coord] = set()
    for coords, values in finals:
        for (r_raw, c_raw), color in zip(coords, values):
            r = int(r_raw)
            c = int(c_raw)
            assert 0 <= r < h and 0 <= c < w, "final object position must stay in bounds"
            assert (r, c) not in occupied, "moved objects must not overlap"
            assert int(out[r, c]) == bg, "moved object must stop on an empty cell"
            occupied.add((r, c))
            out[r, c] = int(color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each enclosed object out through aligned frame gaps, closing every passed door behind it.

    Input structure:
    - A black background contains several colored hollow rectangular frames.
    - A frame may be closed, have one one-cell side gap, or have a pair of
      opposite aligned side gaps that form a straight through-door.
    - Some open frames contain exactly one small object whose row or column is
      aligned with one visible gap; other open or closed frames may be empty or
      contain stationary distractor objects.

    Rule:
    - For every object in an open frame, use the uniquely aligned gap as its
      exit direction and slide the whole object straight outward.
    - The object continues through any later gap met head-on, including paired
      through-gaps of intermediate frames, and stops just before a wall, another
      non-background figure, or the grid edge.
    - Clear the object's original cells to black, repaint every traversed gap
      with that gap's own frame color, and stamp the object at its final stop.
      Frames and contents whose gaps are not traversed remain unchanged.

    Key insight:
    - The gaps are doors in a straight cascade: the lasting evidence of the
      path is not a trail, but the set of formerly open doors that have been
      closed after the migrating object passed through them.
    """
    return _solve_grid(grid)


def _rand_color(exclude: set[int] | None = None) -> int:
    excluded = set() if exclude is None else set(map(int, exclude))
    choices = [c for c in NONBLACK if c not in excluded]
    assert choices, "no colors left to sample"
    return int(random.choice(choices))


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int, gaps: set[Coord]) -> None:
    b = (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1))
    for r, c in _border_cells(b):
        if (r, c) in gaps:
            assert int(grid[r, c]) == BLACK
            continue
        assert int(grid[r, c]) == BLACK
        grid[r, c] = int(color)


def _rect_clear(grid: np.ndarray, r0: int, c0: int, h: int, w: int, margin: int = 1) -> bool:
    gh, gw = grid.shape
    rr0 = max(0, int(r0) - int(margin))
    cc0 = max(0, int(c0) - int(margin))
    rr1 = min(gh, int(r0) + int(h) + int(margin))
    cc1 = min(gw, int(c0) + int(w) + int(margin))
    return bool(np.all(grid[rr0:rr1, cc0:cc1] == BLACK))


def _choose_chain_rows(height: int, count: int) -> list[tuple[int, int, int]] | None:
    chosen: list[tuple[int, int, int]] = []
    for _ in range(count):
        options: list[tuple[int, int, int]] = []
        for h in (5, 7):
            for gap_offset in range(1, h - 1):
                for lane_r in range(gap_offset + 1, height - (h - gap_offset)):
                    top = lane_r - gap_offset
                    bottom = top + h - 1
                    if top < 1 or bottom > height - 2:
                        continue
                    if all(bottom + 2 < prev_top or top - 2 > prev_bottom for prev_top, prev_bottom, _ in chosen):
                        options.append((top, bottom, gap_offset))
        if not options:
            return None
        top, bottom, gap_offset = random.choice(options)
        chosen.append((top, bottom, gap_offset))
    random.shuffle(chosen)
    return chosen


def _chain_transit_count(kind: str) -> int:
    if kind == "transit_terminal":
        return random.randint(1, 2)
    if kind in ("edge", "blocker") and random.random() < 0.55:
        return 1
    return 0


def _place_chain(grid: np.ndarray, kind: str, row_spec: tuple[int, int, int]) -> dict | None:
    height, width = grid.shape
    top, bottom, gap_offset = row_spec
    frame_h = bottom - top + 1
    lane_r = top + gap_offset
    transit_count = _chain_transit_count(kind)
    has_terminal = kind in ("terminal", "transit_terminal")
    has_blocker = kind == "blocker"

    for _ in range(120):
        frame_widths = [random.choice((5, 7))]
        frame_widths.extend(random.choice((5, 7)) for _ in range(transit_count))
        if has_terminal:
            frame_widths.append(random.choice((5, 7)))

        spacings = [random.randint(2, 4) for _ in range(max(1, len(frame_widths)))]
        total = sum(frame_widths) + sum(spacings[: max(0, len(frame_widths) - 1)])
        if has_blocker:
            total += spacings[-1] + 1
        elif not has_terminal:
            total += 3

        if total > width - 3:
            continue
        start_c0 = random.randint(1, width - total - 1)
        c0 = start_c0
        frame_specs: list[dict] = []

        start_w = frame_widths[0]
        start_c1 = c0 + start_w - 1
        frame_specs.append({"role": "start", "c0": c0, "w": start_w})
        c0 = start_c1

        for i in range(transit_count):
            c0 = c0 + spacings[i] + 1
            w = frame_widths[1 + i]
            frame_specs.append({"role": "transit", "c0": c0, "w": w})
            c0 = c0 + w - 1

        terminal_spec = None
        if has_terminal:
            c0 = c0 + spacings[transit_count] + 1
            w = frame_widths[-1]
            terminal_spec = {"role": "terminal", "c0": c0, "w": w}
            frame_specs.append(terminal_spec)
            c0 = c0 + w - 1

        blocker_pos = None
        if has_blocker:
            c0 = c0 + spacings[transit_count] + 1
            blocker_pos = (lane_r, c0)
            if c0 >= width - 1:
                continue
        elif not has_terminal and c0 > width - 4:
            continue

        if any(not _rect_clear(grid, top, int(spec["c0"]), frame_h, int(spec["w"]), margin=0) for spec in frame_specs):
            continue
        if blocker_pos is not None and int(grid[blocker_pos]) != BLACK:
            continue

        closed_gaps: list[tuple[Coord, int]] = []
        for spec in frame_specs:
            c_left = int(spec["c0"])
            c_right = c_left + int(spec["w"]) - 1
            color = _rand_color()
            if spec["role"] == "start":
                gaps = {(lane_r, c_right)}
                closed_gaps.append(((lane_r, c_right), color))
            elif spec["role"] == "transit":
                gaps = {(lane_r, c_left), (lane_r, c_right)}
                closed_gaps.append(((lane_r, c_left), color))
                closed_gaps.append(((lane_r, c_right), color))
            else:
                gaps = {(lane_r, c_left)}
                closed_gaps.append(((lane_r, c_left), color))
            _draw_frame(grid, top, c_left, frame_h, int(spec["w"]), color, gaps)

        start_obj = (lane_r, start_c1 - 1)
        object_color = _rand_color(exclude={int(grid[top, start_c1])})
        assert int(grid[start_obj]) == BLACK
        grid[start_obj] = object_color

        if blocker_pos is not None:
            blocker_color = _rand_color(exclude={object_color})
            grid[blocker_pos] = blocker_color
            final = (lane_r, blocker_pos[1] - 1)
        elif terminal_spec is not None:
            terminal_c1 = int(terminal_spec["c0"]) + int(terminal_spec["w"]) - 1
            final = (lane_r, terminal_c1 - 1)
        else:
            final = (lane_r, width - 1)

        assert int(grid[final]) == BLACK
        min_closed = 4 if kind == "transit_terminal" else 1
        if len(closed_gaps) < min_closed:
            continue

        return {
            "kind": kind,
            "start": start_obj,
            "final": final,
            "object_color": object_color,
            "closed": closed_gaps,
            "lane_row": lane_r,
        }

    return None


def _sample_gap_for_side(r0: int, c0: int, h: int, w: int, side: str) -> Coord:
    if side == "top":
        return r0, c0 + random.randint(1, w - 2)
    if side == "bottom":
        return r0 + h - 1, c0 + random.randint(1, w - 2)
    if side == "left":
        return r0 + random.randint(1, h - 2), c0
    assert side == "right"
    return r0 + random.randint(1, h - 2), c0 + w - 1


def _place_distractor(
    grid: np.ndarray,
    lane_rows: set[int],
    kind: str,
    untouched_gaps: list[Coord],
    stationary_cells: list[Coord],
) -> bool:
    height, width = grid.shape
    for _ in range(160):
        h = random.choice((5, 7))
        w = random.choice((5, 7))
        if height - h - 2 < 1 or width - w - 2 < 1:
            continue
        r0 = random.randint(1, height - h - 1)
        c0 = random.randint(1, width - w - 1)
        if any(r0 - 1 <= lane <= r0 + h for lane in lane_rows):
            continue
        if not _rect_clear(grid, r0, c0, h, w, margin=1):
            continue

        color = _rand_color()
        gaps: set[Coord] = set()
        if kind == "open":
            gap = _sample_gap_for_side(r0, c0, h, w, random.choice(("top", "bottom", "left", "right")))
            gaps.add(gap)
            untouched_gaps.append(gap)
        elif kind == "through":
            if random.random() < 0.5:
                offset = random.randint(1, h - 2)
                gaps.add((r0 + offset, c0))
                gaps.add((r0 + offset, c0 + w - 1))
            else:
                offset = random.randint(1, w - 2)
                gaps.add((r0, c0 + offset))
                gaps.add((r0 + h - 1, c0 + offset))
            untouched_gaps.extend(sorted(gaps))

        _draw_frame(grid, r0, c0, h, w, color, gaps)
        if kind == "closed":
            rr = r0 + random.randint(1, h - 2)
            cc = c0 + random.randint(1, w - 2)
            cell_color = _rand_color(exclude={color})
            assert int(grid[rr, cc]) == BLACK
            grid[rr, cc] = cell_color
            stationary_cells.append((rr, cc))
        return True
    return False


def _expected_from_records(inp: np.ndarray, records: list[dict]) -> np.ndarray:
    out = inp.copy()
    for record in records:
        out[record["start"]] = BLACK
    for record in records:
        for pos, color in record["closed"]:
            out[pos] = int(color)
    for record in records:
        out[record["final"]] = int(record["object_color"])
    return out


def _transform_grid(grid: np.ndarray, op: int) -> np.ndarray:
    if op == 0:
        out = grid
    elif op == 1:
        out = np.rot90(grid, 1)
    elif op == 2:
        out = np.rot90(grid, 2)
    elif op == 3:
        out = np.rot90(grid, 3)
    elif op == 4:
        out = np.fliplr(grid)
    elif op == 5:
        out = np.flipud(grid)
    elif op == 6:
        out = grid.T
    else:
        out = np.fliplr(grid).T
    return np.ascontiguousarray(out)


def _mode_kinds(mode: int) -> list[str]:
    if mode == 0:
        return ["edge"]
    if mode == 1:
        return ["terminal"]
    if mode == 2:
        return ["transit_terminal"]
    return ["blocker", random.choice(("terminal", "transit_terminal"))]


def _gen_attempt(mode: int) -> np.ndarray | None:
    kinds = _mode_kinds(mode)
    chain_count = len(kinds)
    height = random.randint(22, 30) if chain_count > 1 else random.randint(17, 27)
    width = random.randint(24, 30)
    grid = np.zeros((height, width), dtype=int)

    row_specs = _choose_chain_rows(height, chain_count)
    if row_specs is None:
        return None

    records: list[dict] = []
    lane_rows: set[int] = set()
    for kind, row_spec in zip(kinds, row_specs):
        record = _place_chain(grid, kind, row_spec)
        if record is None:
            return None
        records.append(record)
        lane_rows.add(int(record["lane_row"]))

    untouched_gaps: list[Coord] = []
    stationary_cells: list[Coord] = []
    required = ["open", "closed"]
    if mode == 2:
        required.append("through")
    random.shuffle(required)
    for kind in required:
        if not _place_distractor(grid, lane_rows, kind, untouched_gaps, stationary_cells):
            return None

    for _ in range(random.randint(0, 2)):
        _place_distractor(
            grid,
            lane_rows,
            random.choice(("open", "closed", "through")),
            untouched_gaps,
            stationary_cells,
        )

    expected = _expected_from_records(grid, records)
    if not np.array_equal(_solve_grid(grid), expected):
        return None

    for record in records:
        assert int(expected[record["start"]]) == BLACK
        assert int(expected[record["final"]]) == int(record["object_color"])
        assert len(record["closed"]) >= 1
        if record["kind"] == "transit_terminal":
            assert len(record["closed"]) >= 4
    for gap in untouched_gaps:
        assert int(expected[gap]) == BLACK
    for cell in stationary_cells:
        assert int(expected[cell]) == int(grid[cell])

    op = random.randrange(8)
    inp_t = _transform_grid(grid, op)
    expected_t = _transform_grid(expected, op)
    try:
        actual = main(inp_t)
    except Exception:
        return None
    if not np.array_equal(actual, expected_t):
        return None
    return inp_t


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    mode = _GEN_CALL_COUNT % 4
    _GEN_CALL_COUNT += 1

    for attempt in range(_GEN_RETRY_LIMIT):
        chosen_mode = mode if attempt < _GEN_RETRY_LIMIT // 2 else random.randrange(4)
        grid = _gen_attempt(chosen_mode)
        if grid is None:
            continue
        out = main(grid)
        inp_sig = (tuple(map(int, grid.shape)), grid.tobytes())
        out_sig = (tuple(map(int, out.shape)), out.tobytes())
        if inp_sig in _GEN_SEEN_INPUTS or out_sig in _GEN_SEEN_OUTPUTS:
            continue
        _GEN_SEEN_INPUTS.add(inp_sig)
        _GEN_SEEN_OUTPUTS.add(out_sig)
        return grid

    raise RuntimeError("failed to generate a valid door-chain migration input")
