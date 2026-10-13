from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
LINE_COLORS = (BLUE, GREEN, TEAL, MAROON)
OBJECT_COLORS = (YELLOW, PINK, ORANGE)
SHAPES = (
    ((0, 0), (0, 1), (1, 0)),  # triomino L
    ((0, 0), (1, 0), (1, 1), (1, 2)),  # tetromino L
    ((0, 0), (0, 1), (1, 1), (1, 2)),  # tetromino Z
    ((0, 0), (0, 1), (0, 2), (1, 1)),  # tetromino T
    ((0, 0), (0, 1), (1, 0), (1, 1)),  # square
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),  # pentomino P-like
)


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_of_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds((h, w), nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _find_components(grid: np.ndarray, *, exclude: set[int]) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(set(map(int, np.unique(grid))) - set(exclude)):
        for coords in _components_of_mask(grid == color):
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _straight_orientation(coords: np.ndarray) -> str | None:
    rows = set(map(int, coords[:, 0]))
    cols = set(map(int, coords[:, 1]))
    if len(rows) == 1 and len(cols) >= 2:
        return "h"
    if len(cols) == 1 and len(rows) >= 2:
        return "v"
    return None


def _endpoint_set(comp: dict) -> set[tuple[int, int]]:
    coords = comp["coords"]
    orient = _straight_orientation(coords)
    if orient == "h":
        row = int(coords[0, 0])
        return {(row, int(coords[:, 1].min())), (row, int(coords[:, 1].max()))}
    if orient == "v":
        col = int(coords[0, 1])
        return {(int(coords[:, 0].min()), col), (int(coords[:, 0].max()), col)}
    return set()


def _classify_scene(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    comps = _find_components(grid, exclude={BLACK, RED})
    lines: list[dict] = []
    objects: list[dict] = []
    for comp in comps:
        orient = _straight_orientation(comp["coords"])
        if orient is not None:
            comp = dict(comp)
            comp["orientation"] = orient
            lines.append(comp)
        else:
            objects.append(comp)
    return lines, objects


def _repair_grid(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int, copy=True)
    red_coords = np.argwhere(g == RED)
    h, w = g.shape

    for r, c in map(tuple, red_coords):
        for nr, nc in neighbors4(r, c):
            assert not (in_bounds((h, w), nr, nc) and g[nr, nc] == RED), "red markers must be isolated"

    for r, c in sorted(map(tuple, red_coords)):
        line_frags = _find_components(g, exclude={BLACK, RED})
        endpoint_info: dict[tuple[int, int], tuple[int, str]] = {}
        for comp in line_frags:
            orient = _straight_orientation(comp["coords"])
            if orient is None:
                continue
            for endpoint in _endpoint_set(comp):
                endpoint_info[endpoint] = (int(comp["color"]), orient)

        candidates: list[tuple[int, int, str, tuple[int, int]]] = []
        for dr, dc in DIRS4:
            rr, cc = r + dr, c + dc
            dist = 1
            while in_bounds((h, w), rr, cc) and int(g[rr, cc]) in (BLACK, RED):
                rr += dr
                cc += dc
                dist += 1
            if not in_bounds((h, w), rr, cc):
                continue
            key = (rr, cc)
            if key in endpoint_info:
                color, orient = endpoint_info[key]
                candidates.append((dist, color, orient, key))

        assert candidates, "each red marker must see a line endpoint"
        best_dist = min(item[0] for item in candidates)
        best = [item for item in candidates if item[0] == best_dist]
        assert len(best) == 1, "nearest endpoint must be unique"
        _, line_color, orient, _ = best[0]

        if orient == "h":
            left = None
            right = None
            for cc in range(c - 1, -1, -1):
                val = int(g[r, cc])
                if val == line_color:
                    left = cc
                    break
                if val not in (BLACK, RED):
                    break
            for cc in range(c + 1, w):
                val = int(g[r, cc])
                if val == line_color:
                    right = cc
                    break
                if val not in (BLACK, RED):
                    break
            assert left is not None and right is not None and left < c < right
            g[r, left + 1 : right] = line_color
        else:
            up = None
            down = None
            for rr in range(r - 1, -1, -1):
                val = int(g[rr, c])
                if val == line_color:
                    up = rr
                    break
                if val not in (BLACK, RED):
                    break
            for rr in range(r + 1, h):
                val = int(g[rr, c])
                if val == line_color:
                    down = rr
                    break
                if val not in (BLACK, RED):
                    break
            assert up is not None and down is not None and up < r < down
            g[up + 1 : down, c] = line_color

    return g


def _line_lookup(lines: list[dict]) -> dict[tuple[int, int], int]:
    lookup: dict[tuple[int, int], int] = {}
    for idx, comp in enumerate(lines):
        for r, c in map(tuple, comp["coords"]):
            lookup[(int(r), int(c))] = idx
    return lookup


def _frontier(coords: set[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return sorted((r, c) for (r, c) in coords if (r + dr, c + dc) not in coords)


def _infer_move(grid: np.ndarray, obj: dict, line_at: dict[tuple[int, int], int]) -> tuple[tuple[int, int], int, int]:
    h, w = grid.shape
    coords = {tuple(map(int, rc)) for rc in obj["coords"]}
    candidates: list[tuple[tuple[int, int], int, int, int]] = []
    for dr, dc in DIRS4:
        front = _frontier(coords, dr, dc)
        if not front:
            continue
        hits: list[int] = []
        target = None
        ok = True
        for r, c in front:
            rr, cc = r + dr, c + dc
            dist = 1
            while in_bounds((h, w), rr, cc) and int(grid[rr, cc]) == BLACK:
                rr += dr
                cc += dc
                dist += 1
            if not in_bounds((h, w), rr, cc):
                ok = False
                break
            lid = line_at.get((rr, cc))
            if lid is None:
                ok = False
                break
            if target is None:
                target = lid
            elif target != lid:
                ok = False
                break
            hits.append(dist)
        if ok and hits:
            shift = min(hits) - 1
            if shift >= 1:
                candidates.append(((dr, dc), int(target), int(shift), min(hits)))
    assert len(candidates) == 1, "each object must face exactly one repaired wall"
    move_dir, target, shift, _ = candidates[0]
    return move_dir, target, shift


def _apply_moves(repaired: np.ndarray, objects: list[dict], moves: dict[int, tuple[tuple[int, int], int, int]]) -> np.ndarray:
    out = repaired.copy()
    ordered = sorted(objects, key=lambda comp: (int(comp["bbox"][0]), int(comp["bbox"][1]), int(comp["color"])))
    for obj in ordered:
        color = int(obj["color"])
        (dr, dc), _target, shift = moves[color]
        coords = [tuple(map(int, rc)) for rc in obj["coords"]]
        for step in range(shift):
            for r, c in coords:
                out[r + step * dr, c + step * dc] = BLACK
        for r, c in coords:
            out[r + shift * dr, c + shift * dc] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Repair marked gaps in straight wall fragments, then slide each nearby object until it docks against the repaired wall and leave a black trail.

    Input structure:
    - The background is black.
    - Several non-red components are one-pixel-wide horizontal or vertical wall fragments, each containing one or two black gaps marked by isolated red pixels.
    - Several other non-red components are small monochrome movable objects that are not straight line segments.

    Rule:
    - For every isolated red marker, ray-cast along its row and column to the unique nearest visible wall-fragment endpoint, copy that wall color onto the marker, and fill the whole collinear gap so the two fragment ends reconnect.
    - After all gaps are repaired, treat every continuous straight segment as a wall.
    - For each movable object, inspect the four cardinal directions from its forward boundary pixels and find the unique direction where every first non-background hit belongs to the same repaired wall.
    - Translate the whole object in that direction until its front becomes orthogonally adjacent to the wall, paint every intermediate occupied cell black, and keep the repaired walls unchanged.

    Key insight:
    - The red pixels are not noise: they restore missing wall cells, and only the repaired wall topology creates the unique stopping boundary that determines where each object docks.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    repaired = _repair_grid(g)
    lines, objects = _classify_scene(repaired)
    assert 2 <= len(lines) <= 3
    assert 2 <= len(objects) <= 3
    line_at = _line_lookup(lines)

    moves: dict[int, tuple[tuple[int, int], int, int]] = {}
    for obj in objects:
        moves[int(obj["color"])] = _infer_move(repaired, obj, line_at)
    return _apply_moves(repaired, objects, moves)


def _normalize_shape(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _rotate_shape(shape: tuple[tuple[int, int], ...], k: int) -> tuple[tuple[int, int], ...]:
    coords = list(shape)
    for _ in range(k % 4):
        h = max(r for r, _ in coords) + 1
        coords = [(c, h - 1 - r) for r, c in coords]
        coords = list(_normalize_shape(coords))
    return _normalize_shape(coords)


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _flat_front(shape: tuple[tuple[int, int], ...], move: tuple[int, int]) -> tuple[list[tuple[int, int]], int] | None:
    dr, dc = move
    cells = set(shape)
    front = sorted((r, c) for (r, c) in cells if (r + dr, c + dc) not in cells)
    if dr != 0:
        vals = {r for r, _ in front}
    else:
        vals = {c for _, c in front}
    if len(vals) != 1:
        return None
    return front, next(iter(vals))


def _expand(coords: set[tuple[int, int]], h: int, w: int, pad: int = 1) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in coords:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _sample_gap_intervals(length: int, main_gap: tuple[int, int]) -> list[tuple[int, int]]:
    intervals = [main_gap]
    if length < 9 or random.random() >= 0.55:
        return intervals

    main_start, main_end = main_gap
    side = random.choice(("left", "right"))
    gap_len = random.randint(2, 3)
    if side == "left":
        max_end = main_start - 3
        if max_end - gap_len + 1 >= 1:
            start = random.randint(1, max_end - gap_len + 1)
            intervals.append((start, start + gap_len - 1))
    else:
        min_start = main_end + 3
        if min_start + gap_len <= length - 2:
            start = random.randint(min_start, length - gap_len - 1)
            intervals.append((start, start + gap_len - 1))
    return sorted(intervals)


def _marker_pos(start: int, end: int) -> int:
    return start if random.random() < 0.5 else end


def _pair_candidate(
    h: int,
    w: int,
    line_color: int,
    object_color: int,
) -> dict | None:
    orientation = random.choice(("h", "v"))
    if orientation == "h":
        move = random.choice(((1, 0), (-1, 0)))
    else:
        move = random.choice(((0, 1), (0, -1)))

    options: list[tuple[tuple[int, int], ...]] = []
    for base in SHAPES:
        for rot in range(4):
            shape = _rotate_shape(base, rot)
            if shape not in options and _flat_front(shape, move) is not None:
                options.append(shape)
    if not options:
        return None
    shape = random.choice(options)
    flat = _flat_front(shape, move)
    assert flat is not None
    front, face_level = flat
    obj_h, obj_w = _shape_dims(shape)
    shift = random.randint(2, 5)

    if orientation == "h":
        front_cols = [c for _, c in front]
        gap_left = min(front_cols)
        gap_right = max(front_cols)
        gap_len = gap_right - gap_left + 1
        line_len = random.randint(max(7, gap_len + 4), min(12, w - 2))
        main_gap_start = random.randint(2, line_len - gap_len - 2)
        main_gap = (main_gap_start, main_gap_start + gap_len - 1)
        gap_intervals = _sample_gap_intervals(line_len, main_gap)
        line_row = random.randint(2, h - 3)
        line_c0 = random.randint(1, w - line_len - 1)
        obj_c0 = line_c0 + main_gap_start - gap_left
        if move == (1, 0):
            final_r0 = line_row - 1 - face_level
        else:
            final_r0 = line_row + 1 - face_level
        init_r0 = final_r0 - shift * move[0]
        if not (0 <= init_r0 and init_r0 + obj_h <= h and 0 <= obj_c0 and obj_c0 + obj_w <= w):
            return None
    else:
        front_rows = [r for r, _ in front]
        gap_top = min(front_rows)
        gap_bottom = max(front_rows)
        gap_len = gap_bottom - gap_top + 1
        line_len = random.randint(max(7, gap_len + 4), min(12, h - 2))
        main_gap_start = random.randint(2, line_len - gap_len - 2)
        main_gap = (main_gap_start, main_gap_start + gap_len - 1)
        gap_intervals = _sample_gap_intervals(line_len, main_gap)
        line_col = random.randint(2, w - 3)
        line_r0 = random.randint(1, h - line_len - 1)
        obj_r0 = line_r0 + main_gap_start - gap_top
        if move == (0, 1):
            final_c0 = line_col - 1 - face_level
        else:
            final_c0 = line_col + 1 - face_level
        init_c0 = final_c0 - shift * move[1]
        if not (0 <= obj_r0 and obj_r0 + obj_h <= h and 0 <= init_c0 and init_c0 + obj_w <= w):
            return None

    line_full: set[tuple[int, int]] = set()
    line_input: dict[tuple[int, int], int] = {}
    markers: list[tuple[int, int]] = []
    shape_cells = [(int(r), int(c)) for r, c in shape]

    if orientation == "h":
        for off in range(line_len):
            cell = (line_row, line_c0 + off)
            line_full.add(cell)
            if any(start <= off <= end for start, end in gap_intervals):
                line_input[cell] = BLACK
            else:
                line_input[cell] = line_color
        for start, end in gap_intervals:
            markers.append((line_row, line_c0 + _marker_pos(start, end)))
        object_init = {(init_r0 + r, obj_c0 + c) for r, c in shape_cells}
        object_final = {(final_r0 + r, obj_c0 + c) for r, c in shape_cells}
    else:
        for off in range(line_len):
            cell = (line_r0 + off, line_col)
            line_full.add(cell)
            if any(start <= off <= end for start, end in gap_intervals):
                line_input[cell] = BLACK
            else:
                line_input[cell] = line_color
        for start, end in gap_intervals:
            markers.append((line_r0 + _marker_pos(start, end), line_col))
        object_init = {(obj_r0 + r, init_c0 + c) for r, c in shape_cells}
        object_final = {(obj_r0 + r, final_c0 + c) for r, c in shape_cells}

    swept: set[tuple[int, int]] = set(line_full)
    cur = set(object_init)
    for _ in range(shift + 1):
        swept |= cur
        cur = {(r + move[0], c + move[1]) for r, c in cur}

    if line_full & object_init:
        return None
    for marker in markers:
        if marker in object_init:
            return None

    return {
        "line_color": line_color,
        "object_color": object_color,
        "move": move,
        "line_full": line_full,
        "line_input": line_input,
        "markers": markers,
        "object_init": object_init,
        "object_final": object_final,
        "swept": swept,
    }


def _draw_scene(h: int, w: int, pairs: list[dict]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for pair in pairs:
        for (r, c), color in pair["line_input"].items():
            if color != BLACK:
                grid[r, c] = color
        for r, c in pair["markers"]:
            grid[r, c] = RED
        for r, c in pair["object_init"]:
            grid[r, c] = pair["object_color"]
    return grid


def _validate_generated(inp: np.ndarray) -> bool:
    try:
        repaired = _repair_grid(inp)
        lines, objects = _classify_scene(repaired)
        if not (2 <= len(lines) <= 3 and 2 <= len(objects) <= 3):
            return False
        if len({int(obj["color"]) for obj in objects}) != len(objects):
            return False
        line_at = _line_lookup(lines)
        repaired_moves: dict[int, tuple[tuple[int, int], int, int]] = {}
        for obj in objects:
            repaired_moves[int(obj["color"])] = _infer_move(repaired, obj, line_at)

        raw_lines, raw_objects = _classify_scene(inp)
        raw_line_at = _line_lookup(raw_lines)
        raw_by_color = {int(obj["color"]): obj for obj in raw_objects}
        for color, repaired_move in repaired_moves.items():
            if color not in raw_by_color:
                return False
            try:
                raw_move = _infer_move(inp, raw_by_color[color], raw_line_at)
            except Exception:
                raw_move = None
            if raw_move == repaired_move:
                return False

        moved = _apply_moves(repaired, objects, repaired_moves)
        oracle = main(inp)
        if not np.array_equal(moved, oracle):
            return False

        trail_mask = (repaired != moved) & (moved == BLACK)
        if trail_mask.sum() == 0:
            return False
        return True
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(16, 22)
    w = random.randint(16, 22)
    count = random.choice((2, 2, 3))
    line_colors = random.sample(LINE_COLORS, count)
    object_colors = random.sample(OBJECT_COLORS, count)

    pairs: list[dict] = []
    reserved: set[tuple[int, int]] = set()
    for idx in range(count):
        placed = False
        for _ in range(200):
            pair = _pair_candidate(h, w, line_colors[idx], object_colors[idx])
            if pair is None:
                continue
            expanded = _expand(pair["swept"], h, w, pad=1)
            if reserved & expanded:
                continue
            reserved |= expanded
            pairs.append(pair)
            placed = True
            break
        if not placed:
            return None

    inp = _draw_scene(h, w, pairs)
    if len(np.argwhere(inp == RED)) < count:
        return None
    if not _validate_generated(inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
