import numpy as np
from collections import deque


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

PIECE_COLORS = (MAGENTA, ORANGE, AZURE)
STRUCTURE_COLORS = {BG, BLUE, RED, GREEN, YELLOW, GREY, MAROON}
TRACK_COLORS = {GREY, BLUE, RED, GREEN, MAROON}
SIGNAL_COLORS = {BLUE, RED, GREEN}

DIRS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
DIR_LIST = list(DIRS.values())
SIDE_TO_FINAL_DIR = {
    "left": DIRS["right"],
    "right": DIRS["left"],
    "top": DIRS["down"],
    "bottom": DIRS["up"],
}

SHAPES = (
    ((0, 0), (1, 0)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (3, 0)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
)

GEN_RETRIES = 400
PATH_RETRIES = 300


def _add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return a[0] + b[0], a[1] + b[1]


def _sub(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return a[0] - b[0], a[1] - b[1]


def _turn_left(d: tuple[int, int]) -> tuple[int, int]:
    return -d[1], d[0]


def _turn_right(d: tuple[int, int]) -> tuple[int, int]:
    return d[1], -d[0]


def _neighbors4(r: int, c: int):
    for dr, dc in DIR_LIST:
        yield r + dr, c + dc


def _halo(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out = set(cells)
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                out.add((nr, nc))
    return out


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords):
    r0, c0, _, _ = _bbox(list(coords))
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _rotate90(coords):
    return tuple((c, -r) for r, c in coords)


def _rotations(coords):
    out = []
    seen = set()
    cur = tuple(coords)
    for _ in range(4):
        norm = _normalize(cur)
        if norm not in seen:
            seen.add(norm)
            out.append(norm)
        cur = _rotate90(cur)
    return out


def _find_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    yellow = np.argwhere(grid == YELLOW)
    assert yellow.size > 0, "missing yellow frame"
    coords = [(int(r), int(c)) for r, c in yellow]
    r0, c0, r1, c1 = _bbox(coords)
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame too small"
    assert np.all(grid[r0, c0 : c1 + 1] == YELLOW)
    assert np.all(grid[r1, c0 : c1 + 1] == YELLOW)
    assert np.all(grid[r0 : r1 + 1, c0] == YELLOW)
    assert np.all(grid[r0 : r1 + 1, c1] == YELLOW)
    return r0, c0, r1, c1


def _extract_slots(grid: np.ndarray, frame: tuple[int, int, int, int]):
    r0, c0, r1, c1 = frame
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    comps = _components(inner != GREY)
    slots = []
    slot_lookup: dict[tuple[int, int], int] = {}
    for idx, comp in enumerate(comps):
        abs_coords = [(r + r0 + 1, c + c0 + 1) for r, c in comp]
        markers = [
            (r, c, int(grid[r, c]))
            for r, c in abs_coords
            if int(grid[r, c]) not in {BG}
        ]
        assert len(markers) == 1, "each slot must contain exactly one marker"
        mr, mc, marker_color = markers[0]
        for coord in abs_coords:
            slot_lookup[coord] = idx
        slots.append(
            {
                "coords": abs_coords,
                "bbox": _bbox(abs_coords),
                "marker": (mr, mc),
                "color": marker_color,
            }
        )
    assert slots, "expected at least one slot"
    return slots, slot_lookup


def _find_pieces(grid: np.ndarray, frame: tuple[int, int, int, int]):
    r0, c0, r1, c1 = frame
    outside = np.ones_like(grid, dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    pieces = []
    for color in np.unique(grid):
        color = int(color)
        if color in STRUCTURE_COLORS:
            continue
        comps = _components((grid == color) & outside)
        for comp in comps:
            if 2 <= len(comp) <= 4:
                pieces.append({"color": color, "coords": comp, "bbox": _bbox(comp)})
    assert pieces, "expected movable pieces"
    return pieces


def _piece_anchor_and_dir(piece, maroons: set[tuple[int, int]]):
    piece_cells = set(piece["coords"])
    matches = []
    for mr, mc in maroons:
        adjacent = []
        for r, c in piece["coords"]:
            if abs(r - mr) + abs(c - mc) == 1:
                adjacent.append((r, c))
        if len(adjacent) == 1:
            anchor = adjacent[0]
            d = (mr - anchor[0], mc - anchor[1])
            if d in DIR_LIST:
                matches.append((anchor, d, (mr, mc)))
    assert len(matches) == 1, "piece must have exactly one maroon indicator"
    anchor, d, maroon = matches[0]
    for r, c in piece_cells:
        if (r, c) == anchor:
            continue
        assert abs(r - maroon[0]) + abs(c - maroon[1]) != 1, "indicator must touch only anchor cell"
    return anchor, d, maroon


def _trace_endpoint(grid: np.ndarray, anchor: tuple[int, int], direction: tuple[int, int]) -> tuple[int, int]:
    h, w = grid.shape
    pos = anchor
    d = direction
    seen = {(pos, d)}
    while True:
        nxt = _add(pos, d)
        nr, nc = nxt
        if not (0 <= nr < h and 0 <= nc < w):
            return pos
        if int(grid[nr, nc]) not in TRACK_COLORS:
            return pos
        pos = nxt
        cell = int(grid[pos])
        nd = d
        if cell == RED:
            nd = _turn_right(d)
        elif cell == BLUE:
            nd = _turn_left(d)
        elif cell == GREEN:
            nd = d
        if cell in SIGNAL_COLORS:
            probe = _add(pos, nd)
            pr, pc = probe
            if not (0 <= pr < h and 0 <= pc < w):
                return pos
            if int(grid[pr, pc]) not in TRACK_COLORS:
                return pos
        state = (pos, nd)
        assert state not in seen, "track may not loop"
        seen.add(state)
        d = nd


def _slot_for_endpoint(
    endpoint: tuple[int, int],
    frame: tuple[int, int, int, int],
    slot_lookup: dict[tuple[int, int], int],
    slots,
):
    r0, c0, r1, c1 = frame
    r, c = endpoint
    if c == c0 - 1 and r0 + 1 <= r <= r1 - 1:
        inner = (r, c0 + 1)
    elif c == c1 + 1 and r0 + 1 <= r <= r1 - 1:
        inner = (r, c1 - 1)
    elif r == r0 - 1 and c0 + 1 <= c <= c1 - 1:
        inner = (r0 + 1, c)
    elif r == r1 + 1 and c0 + 1 <= c <= c1 - 1:
        inner = (r1 - 1, c)
    else:
        raise AssertionError("endpoint must align to one frame side")
    assert inner in slot_lookup, "endpoint must point at a slot"
    return slots[slot_lookup[inner]]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each colored fragment along the grey rail network, turn at signal lights,
    then insert it into the frame slot reached at the frame edge.
    """
    g = np.asarray(grid, dtype=int)
    frame = _find_frame(g)
    slots, slot_lookup = _extract_slots(g, frame)
    pieces = _find_pieces(g, frame)
    maroons = {(int(r), int(c)) for r, c in np.argwhere(g == MAROON)}

    assert len(pieces) == len(maroons) == len(slots), "piece/indicator/slot counts must match"

    out = g.copy()
    for r, c in maroons:
        out[r, c] = GREY
    for piece in pieces:
        for r, c in piece["coords"]:
            out[r, c] = BG

    for piece in pieces:
        anchor, d, _maroon = _piece_anchor_and_dir(piece, maroons)
        endpoint = _trace_endpoint(g, anchor, d)
        slot = _slot_for_endpoint(endpoint, frame, slot_lookup, slots)
        assert slot["color"] == piece["color"], "slot marker color must match piece color"
        marker = slot["marker"]
        delta = _sub(marker, anchor)
        for r, c in piece["coords"]:
            rr, cc = r + delta[0], c + delta[1]
            out[rr, cc] = piece["color"]
    return out


def _choose_shape_for_direction(rng: np.random.Generator, direction: tuple[int, int]):
    shape_order = list(rng.permutation(len(SHAPES)))
    for shape_idx in shape_order:
        variants = _rotations(SHAPES[shape_idx])
        variant_order = list(rng.permutation(len(variants)))
        for vidx in variant_order:
            coords = variants[vidx]
            anchor_options = []
            coord_set = set(coords)
            for ar, ac in coords:
                maroon = (ar + direction[0], ac + direction[1])
                if maroon in coord_set:
                    continue
                touching = 0
                for rr, cc in coords:
                    if abs(rr - maroon[0]) + abs(cc - maroon[1]) == 1:
                        touching += 1
                if touching == 1:
                    rel = tuple(sorted((rr - ar, cc - ac) for rr, cc in coords))
                    anchor_options.append(rel)
            if anchor_options:
                return anchor_options[int(rng.integers(len(anchor_options)))]
    raise RuntimeError("no shape supports requested direction")


def _build_dirs(final_dir: tuple[int, int], turn_colors: list[int]):
    dirs = [None] * (len(turn_colors) + 1)
    dirs[-1] = final_dir
    for i in range(len(turn_colors) - 1, -1, -1):
        if turn_colors[i] == RED:
            dirs[i] = _turn_left(dirs[i + 1])
        else:
            dirs[i] = _turn_right(dirs[i + 1])
    return dirs


def _signal_requirements(
    signal: tuple[int, int],
    color: int,
    incoming: tuple[int, int],
    outgoing: tuple[int, int],
):
    if color == GREEN:
        return [(_add(signal, _turn_left(incoming))), (_add(signal, _turn_right(incoming)))]
    return [(_add(signal, incoming)), (_add(signal, (-outgoing[0], -outgoing[1])))]


def _path_from_spec(start: tuple[int, int], dirs, lengths):
    path = [start]
    turn_points: list[tuple[int, int]] = []
    step_dirs: list[tuple[int, int]] = []
    pos = start
    for i, d in enumerate(dirs):
        for step in range(lengths[i]):
            pos = _add(pos, d)
            path.append(pos)
            step_dirs.append(d)
        if i < len(dirs) - 1:
            turn_points.append(pos)
    return path, turn_points, step_dirs


def _green_candidates(path):
    candidates = []
    for i in range(1, len(path) - 1):
        d0 = _sub(path[i], path[i - 1])
        d1 = _sub(path[i + 1], path[i])
        if d0 == d1:
            candidates.append((i, d0))
    return candidates


def _pick_marker(
    rng: np.random.Generator,
    slot_bbox: tuple[int, int, int, int],
    rel_coords,
    side: str,
):
    r0, c0, r1, c1 = slot_bbox
    min_r = min(r for r, _ in rel_coords)
    max_r = max(r for r, _ in rel_coords)
    min_c = min(c for _, c in rel_coords)
    max_c = max(c for _, c in rel_coords)
    row_lo = r0 - min_r
    row_hi = r1 - max_r
    col_lo = c0 - min_c
    col_hi = c1 - max_c
    assert row_lo <= row_hi and col_lo <= col_hi
    rows = list(range(row_lo, row_hi + 1))
    cols = list(range(col_lo, col_hi + 1))

    if side == "left":
        cols = sorted(cols)[: min(3, len(cols))]
    elif side == "right":
        cols = sorted(cols)[-min(3, len(cols)) :]
    elif side == "top":
        rows = sorted(rows)[: min(3, len(rows))]
    else:
        rows = sorted(rows)[-min(3, len(rows)) :]

    rr = rows[int(rng.integers(len(rows)))]
    cc = cols[int(rng.integers(len(cols)))]
    return rr, cc


def _frame_layout(
    rng: np.random.Generator,
    h: int,
    w: int,
    piece_count: int,
    side: str,
):
    slot_h = int(rng.integers(5, 8))
    slot_w = int(rng.integers(5, 8))
    if side in {"left", "right"}:
        frame_h = 2 + piece_count * slot_h + (piece_count - 1)
        frame_w = 2 + slot_w
    else:
        frame_h = 2 + slot_h
        frame_w = 2 + piece_count * slot_w + (piece_count - 1)

    if frame_h > h - 4 or frame_w > w - 4:
        return None

    if side == "left":
        c_low = max(8, w // 2 - 1)
        c_high = w - frame_w - 2
        r_low = 1
        r_high = h - frame_h - 2
    elif side == "right":
        c_low = 1
        c_high = min(w // 2 - frame_w, w - frame_w - 8)
        r_low = 1
        r_high = h - frame_h - 2
    elif side == "top":
        r_low = max(8, h // 2 - 1)
        r_high = h - frame_h - 2
        c_low = 1
        c_high = w - frame_w - 2
    else:
        r_low = 1
        r_high = min(h // 2 - frame_h, h - frame_h - 8)
        c_low = 1
        c_high = w - frame_w - 2

    if r_low > r_high or c_low > c_high:
        return None
    fr0 = int(rng.integers(r_low, r_high + 1))
    fc0 = int(rng.integers(c_low, c_high + 1))
    fr1 = fr0 + frame_h - 1
    fc1 = fc0 + frame_w - 1

    slots = []
    if side in {"left", "right"}:
        for i in range(piece_count):
            sr0 = fr0 + 1 + i * (slot_h + 1)
            sr1 = sr0 + slot_h - 1
            sc0 = fc0 + 1
            sc1 = fc1 - 1
            slots.append((sr0, sc0, sr1, sc1))
    else:
        for i in range(piece_count):
            sr0 = fr0 + 1
            sr1 = fr1 - 1
            sc0 = fc0 + 1 + i * (slot_w + 1)
            sc1 = sc0 + slot_w - 1
            slots.append((sr0, sc0, sr1, sc1))
    return (fr0, fc0, fr1, fc1), slots


def _draw_frame(grid: np.ndarray, frame: tuple[int, int, int, int], slots):
    fr0, fc0, fr1, fc1 = frame
    grid[fr0, fc0 : fc1 + 1] = YELLOW
    grid[fr1, fc0 : fc1 + 1] = YELLOW
    grid[fr0 : fr1 + 1, fc0] = YELLOW
    grid[fr0 : fr1 + 1, fc1] = YELLOW
    if len(slots) >= 2:
        if slots[0][0] != slots[1][0]:
            for i in range(len(slots) - 1):
                divider = slots[i][2] + 1
                grid[divider, fc0 + 1 : fc1] = GREY
        else:
            for i in range(len(slots) - 1):
                divider = slots[i][3] + 1
                grid[fr0 + 1 : fr1, divider] = GREY


def _entry_point(rng: np.random.Generator, slot_bbox: tuple[int, int, int, int], frame, side: str):
    fr0, fc0, fr1, fc1 = frame
    sr0, sc0, sr1, sc1 = slot_bbox
    if side == "left":
        rr = int(rng.integers(sr0, sr1 + 1))
        return rr, fc0 - 1
    if side == "right":
        rr = int(rng.integers(sr0, sr1 + 1))
        return rr, fc1 + 1
    if side == "top":
        cc = int(rng.integers(sc0, sc1 + 1))
        return fr0 - 1, cc
    cc = int(rng.integers(sc0, sc1 + 1))
    return fr1 + 1, cc


def _generate_piece_path(
    rng: np.random.Generator,
    grid_shape: tuple[int, int],
    endpoint: tuple[int, int],
    final_dir: tuple[int, int],
    blocked: set[tuple[int, int]],
    spacer: set[tuple[int, int]],
    force_signal: int | None,
):
    h, w = grid_shape
    for _ in range(PATH_RETRIES):
        turn_count = int(rng.integers(0, 3))
        if force_signal in {RED, BLUE}:
            turn_count = max(1, turn_count)
        turn_colors = [RED if rng.random() < 0.5 else BLUE for _ in range(turn_count)]
        if force_signal in {RED, BLUE}:
            turn_colors[int(rng.integers(turn_count))] = force_signal
        dirs = _build_dirs(final_dir, turn_colors)
        lengths = [int(rng.integers(2, 6)) for _ in dirs]
        if force_signal == GREEN:
            lengths[int(rng.integers(len(lengths)))] = max(3, lengths[int(rng.integers(len(lengths)))])

        total = np.array([0, 0], dtype=int)
        for d, length in zip(dirs, lengths):
            total += np.array(d) * length
        maroon = (endpoint[0] - int(total[0]), endpoint[1] - int(total[1]))
        path, turn_points, _step_dirs = _path_from_spec(maroon, dirs, lengths)
        if len(set(path)) != len(path):
            continue

        greens = _green_candidates(path)
        signal_map = {turn_points[i]: turn_colors[i] for i in range(len(turn_points))}
        if force_signal == GREEN and not greens:
            continue
        green_budget = 0
        if greens and (force_signal == GREEN or rng.random() < 0.7):
            green_budget = 1 + int(rng.random() < 0.25 and len(greens) > 1)
        rng.shuffle(greens)
        for idx, _d in greens[:green_budget]:
            signal_map[path[idx]] = GREEN
        if force_signal == GREEN and GREEN not in signal_map.values():
            continue

        start_dir = dirs[0]
        rel_shape = _choose_shape_for_direction(rng, start_dir)
        anchor = _sub(maroon, start_dir)
        shape_cells = {(anchor[0] + dr, anchor[1] + dc) for dr, dc in rel_shape}
        if any(not (0 <= r < h and 0 <= c < w) for r, c in shape_cells):
            continue

        if any(cell in blocked or cell in spacer for cell in path):
            continue
        if any(cell in blocked or cell in spacer for cell in shape_cells):
            continue
        if shape_cells & set(path):
            continue

        branch_cells: set[tuple[int, int]] = set()
        ok = True
        index_by_pos = {pos: i for i, pos in enumerate(path)}
        for signal_pos, color in signal_map.items():
            idx = index_by_pos[signal_pos]
            prev_d = _sub(path[idx], path[idx - 1])
            next_d = _sub(path[idx + 1], path[idx]) if idx + 1 < len(path) else prev_d
            needed = _signal_requirements(signal_pos, color, prev_d, next_d)
            for cell in needed:
                r, c = cell
                if not (0 <= r < h and 0 <= c < w):
                    ok = False
                    break
                if cell in blocked or cell in spacer or cell in shape_cells or cell in branch_cells:
                    ok = False
                    break
                if cell in path:
                    ok = False
                    break
                branch_cells.add(cell)
            if not ok:
                break
        if not ok:
            continue

        return {
            "anchor": anchor,
            "maroon": maroon,
            "rel_shape": rel_shape,
            "shape_cells": shape_cells,
            "path": path,
            "signals": signal_map,
            "branches": branch_cells,
            "start_dir": start_dir,
        }
    return None


def _gen_attempt(rng: np.random.Generator):
    h = int(rng.integers(22, 31))
    w = int(rng.integers(22, 31))
    piece_count = int(rng.integers(2, 4))
    side = ("left", "right", "top", "bottom")[int(rng.integers(4))]
    layout = _frame_layout(rng, h, w, piece_count, side)
    if layout is None:
        return None
    frame, slot_bboxes = layout
    grid = np.full((h, w), BG, dtype=int)
    _draw_frame(grid, frame, slot_bboxes)

    blocked = {
        (r, c)
        for r in range(frame[0], frame[2] + 1)
        for c in range(frame[1], frame[3] + 1)
    }
    spacer: set[tuple[int, int]] = set()

    colors = list(rng.permutation(PIECE_COLORS))[:piece_count]
    featured_signal = [RED, BLUE, GREEN][int(rng.integers(3))]
    final_dir = SIDE_TO_FINAL_DIR[side]
    records = []

    for i, (color, slot_bbox) in enumerate(zip(colors, slot_bboxes)):
        endpoint = _entry_point(rng, slot_bbox, frame, side)
        force_signal = featured_signal if i == 0 else None
        info = _generate_piece_path(rng, grid.shape, endpoint, final_dir, blocked, spacer, force_signal)
        if info is None:
            return None

        marker = _pick_marker(rng, slot_bbox, info["rel_shape"], side)
        piece_cells = {(marker[0] + dr, marker[1] + dc) for dr, dc in info["rel_shape"]}
        if not all(slot_bbox[0] <= r <= slot_bbox[2] and slot_bbox[1] <= c <= slot_bbox[3] for r, c in piece_cells):
            return None

        records.append(
            {
                "color": color,
                "slot_bbox": slot_bbox,
                "marker": marker,
                **info,
            }
        )
        blocked |= info["shape_cells"]
        blocked |= set(info["path"])
        blocked |= info["branches"]
        spacer |= _halo(set(info["shape_cells"]) | {info["maroon"]}, grid.shape)

    for rec in records:
        grid[rec["marker"]] = rec["color"]

    for rec in records:
        for cell in rec["path"]:
            if cell == rec["maroon"]:
                grid[cell] = MAROON
            elif cell in rec["signals"]:
                grid[cell] = rec["signals"][cell]
            else:
                grid[cell] = GREY
        for cell in rec["branches"]:
            grid[cell] = GREY
        for cell in rec["shape_cells"]:
            grid[cell] = rec["color"]

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is None:
            continue
        return grid
    raise RuntimeError("failed to generate a valid rail-routing puzzle")
