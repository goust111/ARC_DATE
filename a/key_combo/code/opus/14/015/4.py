import itertools
import random
from collections import Counter

import numpy as np


BG = 0
FRAME = 5
CORRIDOR = 7
SPRITE_COLORS = (1, 2, 3, 6, 8, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
TURN_PATTERNS = [
    (0, 2),
    (1, 3),
    (0, 1, 3),
    (0, 2, 3),
    (1, 2, 3),
    (0, 1, 2),
]
GEN_ATTEMPTS = 400
_GEN_CALL_COUNT = 0


def _in_bounds(shape, r, c):
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r, c):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _components(mask):
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros(mask.shape, dtype=bool)
    comps = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        coords = []
        while stack:
            cr, cc = stack.pop()
            coords.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc):
                if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(sorted(coords), dtype=int))
    return comps


def _bbox(coords):
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid, box):
    r0, c0, r1, c1 = box
    return np.array(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)


def _append_segment(path, a, b):
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        cells = [(r0, c) for c in range(c0, c1 + step, step)]
    else:
        step = 1 if r1 >= r0 else -1
        cells = [(r, c0) for r in range(r0, r1 + step, step)]
    if path and cells and path[-1] == cells[0]:
        cells = cells[1:]
    path.extend(cells)


def _trace_simple_path(coords):
    cells = {tuple(map(int, rc)) for rc in coords}
    nbrs = {cell: [nb for nb in _neighbors4(*cell) if nb in cells] for cell in cells}
    endpoints = [cell for cell, nbs in nbrs.items() if len(nbs) == 1]
    assert len(endpoints) == 2, "corridor must be a simple path with two endpoints"
    path = [min(endpoints)]
    prev = None
    cur = path[0]
    while True:
        nxts = [nb for nb in nbrs[cur] if nb != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "corridor branches are not allowed"
        prev, cur = cur, nxts[0]
        path.append(cur)
        assert len(path) <= len(cells), "corridor revisits a cell"
    assert len(path) == len(cells), "corridor must be fully traceable"
    return path


def _count_turns(path):
    dirs = []
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        dirs.append((r1 - r0, c1 - c0))
    return sum(dirs[i] != dirs[i - 1] for i in range(1, len(dirs)))


def _frame_border_mask():
    mask = np.ones((5, 5), dtype=bool)
    mask[1:4, 1:4] = False
    return mask


FRAME_BORDER_MASK = _frame_border_mask()


def _frame_border_cells(r0, c0):
    cells = []
    for dr in range(5):
        for dc in range(5):
            if FRAME_BORDER_MASK[dr, dc]:
                cells.append((r0 + dr, c0 + dc))
    return cells


def _find_frames(grid):
    frames = []
    used = np.zeros(grid.shape, dtype=bool)
    H, W = grid.shape
    for r0 in range(H - 4):
        for c0 in range(W - 4):
            win = grid[r0 : r0 + 5, c0 : c0 + 5]
            if not np.all(win[1:4, 1:4] == BG):
                continue
            border = win[FRAME_BORDER_MASK]
            if np.count_nonzero(border == FRAME) != 15:
                continue
            others = border[border != FRAME]
            if others.size != 1:
                continue
            marker_color = int(others[0])
            if marker_color in (BG, FRAME, CORRIDOR, 4):
                continue
            coords = [(r0 + dr, c0 + dc) for dr in range(5) for dc in range(5) if FRAME_BORDER_MASK[dr, dc]]
            if any(used[r, c] for r, c in coords):
                continue
            marker_pos = None
            for dr in range(5):
                for dc in range(5):
                    if FRAME_BORDER_MASK[dr, dc] and int(win[dr, dc]) == marker_color:
                        marker_pos = (r0 + dr, c0 + dc)
                        break
                if marker_pos is not None:
                    break
            for r, c in coords:
                used[r, c] = True
            frames.append(
                {
                    "bbox": (r0, c0, r0 + 4, c0 + 4),
                    "marker_color": marker_color,
                    "marker_pos": marker_pos,
                    "border_cells": set(coords),
                }
            )
    assert len(frames) in (2, 3), "expected 2 or 3 hollow frames"
    return frames


def _find_sprites(grid):
    sprites = {}
    special = {BG, FRAME, CORRIDOR}
    for color in sorted(set(map(int, np.unique(grid))) - special):
        comps = _components(grid == color)
        big = [coords for coords in comps if 3 <= len(coords) <= 6]
        singles = [coords for coords in comps if len(coords) == 1]
        assert len(big) == 1, f"color {color} must have exactly one sprite component"
        assert len(singles) == 1, f"color {color} must have exactly one frame marker"
        coords = big[0]
        box = _bbox(coords)
        patch = _crop(grid, box)
        assert patch.shape == (3, 3), "sprite must occupy a 3x3 bounding box"
        rots = {np.rot90(patch, -k).tobytes() for k in range(4)}
        assert len(rots) == 4, "sprite must expose all four rotations distinctly"
        sprites[color] = {"coords": coords, "bbox": box, "patch": patch}
    assert len(sprites) in (2, 3), "expected 2 or 3 sprites"
    return sprites


def _parse_scene(grid):
    g = np.asarray(grid, dtype=int)
    frames = _find_frames(g)
    sprites = _find_sprites(g)
    assert len(frames) == len(sprites), "sprite and frame counts must match"

    frame_by_color = {}
    frame_border_owner = {}
    for idx, frame in enumerate(frames):
        color = int(frame["marker_color"])
        assert color in sprites, "frame marker must match some sprite color"
        assert color not in frame_by_color, "frame marker colors must be unique"
        frame_by_color[color] = idx
        for rc in frame["border_cells"]:
            frame_border_owner[rc] = idx

    sprite_owner = {}
    for color, sprite in sprites.items():
        for rc in sprite["coords"]:
            sprite_owner[tuple(map(int, rc))] = int(color)

    corridors = _components(g == CORRIDOR)
    assert len(corridors) == len(sprites), "corridor count must match sprite count"
    corridor_by_color = {}
    all_turns = set()
    for coords in corridors:
        path = _trace_simple_path(coords)
        end_meta = []
        for endpoint in (path[0], path[-1]):
            sprite_hits = {sprite_owner[nb] for nb in _neighbors4(*endpoint) if nb in sprite_owner}
            frame_hits = {frame_border_owner[nb] for nb in _neighbors4(*endpoint) if nb in frame_border_owner}
            end_meta.append((sprite_hits, frame_hits))

        if len(end_meta[0][0]) == 1 and not end_meta[0][1] and not end_meta[1][0] and len(end_meta[1][1]) == 1:
            sprite_color = next(iter(end_meta[0][0]))
            frame_idx = next(iter(end_meta[1][1]))
            ordered_path = path
        elif len(end_meta[1][0]) == 1 and not end_meta[1][1] and not end_meta[0][0] and len(end_meta[0][1]) == 1:
            sprite_color = next(iter(end_meta[1][0]))
            frame_idx = next(iter(end_meta[0][1]))
            ordered_path = list(reversed(path))
        else:
            raise AssertionError("corridor endpoints must uniquely touch one sprite and one frame")

        assert frame_by_color[sprite_color] == frame_idx, "corridor must connect a sprite to its matching frame"
        turns = _count_turns(ordered_path)
        assert 0 <= turns <= 3, "corridor turns must stay within the allowed range"
        assert turns not in all_turns, "corridor turn counts must be distinct"
        all_turns.add(turns)
        assert sprite_color not in corridor_by_color, "each sprite needs exactly one corridor"
        corridor_by_color[sprite_color] = {
            "path": ordered_path,
            "turns": turns,
            "frame_idx": frame_idx,
            "coords": coords,
        }

    records = []
    for color, sprite in sorted(sprites.items()):
        frame = frames[frame_by_color[color]]
        corridor = corridor_by_color[color]
        records.append(
            {
                "color": int(color),
                "sprite_coords": sprite["coords"],
                "sprite_bbox": sprite["bbox"],
                "sprite_patch": sprite["patch"],
                "frame_bbox": frame["bbox"],
                "frame_marker_pos": frame["marker_pos"],
                "turns": int(corridor["turns"]),
                "corridor_coords": corridor["coords"],
            }
        )
    return records


def _render_from_records(grid, records, turn_rule):
    out = np.array(grid, dtype=int).copy()
    for rec in records:
        rr = rec["sprite_coords"][:, 0]
        cc = rec["sprite_coords"][:, 1]
        out[rr, cc] = BG
        cr = rec["corridor_coords"][:, 0]
        cc2 = rec["corridor_coords"][:, 1]
        out[cr, cc2] = BG
    for rec in records:
        k = int(turn_rule(int(rec["turns"]))) % 4
        rotated = np.rot90(rec["sprite_patch"], -k)
        r0, c0, _r1, _c1 = rec["frame_bbox"]
        window = out[r0 + 1 : r0 + 4, c0 + 1 : c0 + 4]
        assert np.all(window == BG), "frame interior must be empty before nesting"
        mask = rotated != BG
        window[mask] = rotated[mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Slide each sprite along its corridor, rotate it by the corridor's bend count, and nest it inside the matching frame.

    Input structure:
    - The background is black. The grid contains 2-3 hollow 5x5 grey square frames whose 3x3 interiors are empty.
    - Each frame border has exactly one non-grey marker pixel; that marker color matches exactly one remote monochrome sprite with a 3x3 bounding box.
    - Each sprite is attached to its matching frame by a non-branching orange orthogonal corridor that stays outside all objects.

    Rule:
    - Pair every sprite with the unique frame whose border marker has the same color.
    - Trace the orange corridor from the sprite side to the frame side and count how many times its direction changes by 90 degrees.
    - Rotate the sprite clockwise by that turn count times 90 degrees, erase the original sprite and all corridors, then place the rotated sprite into the paired frame's 3x3 interior.

    Key insight:
    - The corridor is not just a connector: its geometry encodes the rotation amount, so both the path and the frame-marker pairing are required to determine the final nested sprite.
    """
    records = _parse_scene(np.asarray(grid, dtype=int))
    return _render_from_records(grid, records, lambda turns: turns)


def _shape_connected(mask):
    return len(_components(mask)) == 1


def _build_shape_pool():
    pool = []
    for area in range(3, 7):
        for cells in itertools.combinations(range(9), area):
            mask = np.zeros((3, 3), dtype=bool)
            for idx in cells:
                mask[idx // 3, idx % 3] = True
            coords = np.argwhere(mask)
            if coords.size == 0:
                continue
            r0, c0, r1, c1 = _bbox(coords)
            if (r0, c0, r1, c1) != (0, 0, 2, 2):
                continue
            if not _shape_connected(mask):
                continue
            if len({np.rot90(mask, -k).tobytes() for k in range(4)}) != 4:
                continue
            pool.append(mask.astype(int))
    return pool


SHAPE_POOL = _build_shape_pool()


def _facing_rows(shape, side):
    col = 2 if side == "right" else 0
    return [r for r in range(3) if int(shape[r, col]) == 1]


def _path_from_waypoints(points):
    path = []
    for a, b in zip(points, points[1:]):
        assert a[0] == b[0] or a[1] == b[1], "corridor segments must be orthogonal"
        _append_segment(path, a, b)
    assert len(path) == len(set(path)), "corridor path must not self-intersect"
    return path


def _make_corridor(start, end, turns, row_lo, row_hi):
    rs, cs = start
    re, ce = end
    assert cs != ce, "horizontal separation is required"
    x_min = min(cs, ce)
    x_max = max(cs, ce)
    if turns == 0:
        assert rs == re
        return _path_from_waypoints([start, end])
    if turns == 1:
        assert rs != re
        if random.random() < 0.5:
            return _path_from_waypoints([start, (re, cs), end])
        return _path_from_waypoints([start, (rs, ce), end])
    if turns == 2:
        assert rs != re
        mids = [x for x in range(x_min + 2, x_max - 1)]
        assert mids
        mid_x = random.choice(mids)
        return _path_from_waypoints([start, (rs, mid_x), (re, mid_x), end])
    assert turns == 3
    mids = [x for x in range(x_min + 2, x_max - 1)]
    mid_rows = [r for r in range(row_lo, row_hi + 1) if r not in (rs, re)]
    assert mids and mid_rows
    mid_x = random.choice(mids)
    mid_r = random.choice(mid_rows)
    return _path_from_waypoints([start, (rs, mid_x), (mid_r, mid_x), (mid_r, ce), end])


def _frame_entry_border(frame_left, frame_top, sprite_left, row):
    if sprite_left:
        return (row, frame_left)
    return (row, frame_left + 4)


def _choose_rows(shape, sprite_top, frame_top, turns):
    sprite_side = random.choice(["left", "right"])
    frame_rows = [frame_top + r for r in range(5)]
    sprite_rows = [sprite_top + r for r in _facing_rows(shape, sprite_side)]
    if turns == 0:
        common = sorted(set(sprite_rows) & set(frame_rows))
        if not common:
            return None
        row = random.choice(common)
        return sprite_side, row, row
    pairs = [(sr, fr) for sr in sprite_rows for fr in frame_rows if sr != fr]
    if turns in (1, 2):
        if not pairs:
            return None
        sr, fr = random.choice(pairs)
        return sprite_side, sr, fr
    common = sorted(set(sprite_rows) & set(frame_rows))
    if common and random.random() < 0.5:
        row = random.choice(common)
        return sprite_side, row, row
    if not pairs:
        return None
    sr, fr = random.choice(pairs)
    return sprite_side, sr, fr


def _choose_layout_dims(count):
    width = random.randint(22, 28)
    min_height = 8 * count + 3
    max_height = min(30, 8 * count + 6)
    height = random.randint(min_height, max_height)
    return height, width


def _paint_shape(grid, top, left, shape, color):
    for r in range(3):
        for c in range(3):
            if int(shape[r, c]):
                grid[top + r, left + c] = int(color)


def _place_instance(shape, color, turns, band_top, width):
    sprite_left = random.random() < 0.5
    frame_top = band_top + random.choice([0, 1])
    sprite_top = band_top + random.choice([1, 2, 3])
    if sprite_left:
        sprite_leftcol = random.choice([1, 2])
        frame_leftcol = width - 6 - random.choice([0, 1])
    else:
        frame_leftcol = random.choice([1, 2])
        sprite_leftcol = width - 4 - random.choice([0, 1])

    row_choice = _choose_rows(shape, sprite_top, frame_top, turns)
    if row_choice is None:
        return None
    sprite_side, sprite_row, frame_row = row_choice

    if sprite_left:
        start = (sprite_row, sprite_leftcol - 1 if sprite_side == "left" else sprite_leftcol + 3)
        end = (frame_row, frame_leftcol - 1)
    else:
        start = (sprite_row, sprite_leftcol - 1 if sprite_side == "left" else sprite_leftcol + 3)
        end = (frame_row, frame_leftcol + 5)

    band_lo = band_top
    band_hi = band_top + 5
    path = _make_corridor(start, end, turns, band_lo, band_hi)

    if sprite_left and start[1] >= end[1]:
        return None
    if (not sprite_left) and start[1] <= end[1]:
        return None
    if any(not _in_bounds((band_hi + 1, width), r, c) for r, c in path):
        return None
    if any(r < band_lo or r > band_hi for r, c in path):
        return None

    entry_border = _frame_entry_border(frame_leftcol, frame_top, sprite_left, frame_row)
    marker_options = [rc for rc in _frame_border_cells(frame_top, frame_leftcol) if rc != entry_border]
    marker_pos = random.choice(marker_options)
    return {
        "color": int(color),
        "shape": np.array(shape, dtype=int),
        "turns": int(turns),
        "frame_top": int(frame_top),
        "frame_left": int(frame_leftcol),
        "sprite_top": int(sprite_top),
        "sprite_left": int(sprite_leftcol),
        "marker_pos": marker_pos,
        "path": path,
    }


def _assemble_grid(records, height, width):
    grid = np.zeros((height, width), dtype=int)
    occupied = np.zeros((height, width), dtype=bool)
    for rec in records:
        fr, fc = rec["frame_top"], rec["frame_left"]
        sr, sc = rec["sprite_top"], rec["sprite_left"]
        for r, c in _frame_border_cells(fr, fc):
            if occupied[r, c]:
                return None
            grid[r, c] = FRAME
            occupied[r, c] = True
        mr, mc = rec["marker_pos"]
        grid[mr, mc] = rec["color"]
        for r in range(3):
            for c in range(3):
                if int(rec["shape"][r, c]) == 0:
                    continue
                rr, cc = sr + r, sc + c
                if occupied[rr, cc]:
                    return None
                grid[rr, cc] = rec["color"]
                occupied[rr, cc] = True
        for rr, cc in rec["path"]:
            if occupied[rr, cc]:
                return None
            grid[rr, cc] = CORRIDOR
            occupied[rr, cc] = True
    return grid


def _validate_grid(grid):
    records = _parse_scene(grid)
    true_out = _render_from_records(grid, records, lambda turns: turns)
    assert np.array_equal(true_out, main(grid))
    assert len(records) in (2, 3)
    assert len({rec["color"] for rec in records}) == len(records)
    assert len({rec["turns"] for rec in records}) == len(records)
    assert any(rec["turns"] >= 2 for rec in records), "need a multi-bend counterexample"

    no_rotation = _render_from_records(grid, records, lambda turns: 0)
    any_bend = _render_from_records(grid, records, lambda turns: 1 if turns > 0 else 0)
    assert not np.array_equal(no_rotation, true_out), "non-zero rotation must matter"
    assert not np.array_equal(any_bend, true_out), "multi-bend examples must reject the binary-bend shortcut"

    hashes = Counter()
    for rec in records:
        rotated = np.rot90(rec["sprite_patch"], -rec["turns"])
        hashes[(rec["color"], rotated.tobytes())] += 1
    assert all(v == 1 for v in hashes.values()), "rotated inserts must stay distinguishable"
    return True


def _gen_attempt():
    global _GEN_CALL_COUNT
    turns = list(TURN_PATTERNS[_GEN_CALL_COUNT % len(TURN_PATTERNS)])
    _GEN_CALL_COUNT += 1
    random.shuffle(turns)
    count = len(turns)
    height, width = _choose_layout_dims(count)
    base_span = 6 * count + 2 * (count - 1)
    top_margin = (height - base_span) // 2
    colors = random.sample(list(SPRITE_COLORS), count)
    shapes = random.sample(SHAPE_POOL, count)

    records = []
    for idx, (shape, color, bend_count) in enumerate(zip(shapes, colors, turns)):
        band_top = top_margin + idx * 8
        rec = _place_instance(shape, color, bend_count, band_top, width)
        if rec is None:
            return None
        records.append(rec)

    grid = _assemble_grid(records, height, width)
    if grid is None:
        return None
    if 4 in set(map(int, np.unique(grid))):
        return None
    if max(grid.shape) > 30:
        return None
    try:
        _validate_grid(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid corridor-rotation-nesting input")
