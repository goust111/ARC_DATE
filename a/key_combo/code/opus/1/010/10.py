from __future__ import annotations

import itertools
import random
from collections import deque

import numpy as np


BACKGROUND = 0
FRAME = 1
SPRITE = 7
FILL = 8

DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}

GRID_MIN = 20
GRID_MAX = 26
MAX_GENERATION_ATTEMPTS = 400
INTERSECTION_CHOICES = [1, 1, 1, 2, 2, 3, 3]

SPRITE_MASKS: list[np.ndarray] = []


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
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
                for dr, dc in DIRS.values():
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _mask_is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    comp = _components(mask.astype(bool))
    return len(comp) == 1


def _build_sprite_masks() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for area in range(3, 6):
        for cells in itertools.combinations(range(9), area):
            mask = np.zeros((3, 3), dtype=bool)
            for idx in cells:
                mask[idx // 3, idx % 3] = True
            if _mask_is_connected(mask):
                masks.append(mask)
    return masks


SPRITE_MASKS = _build_sprite_masks()


def _frame_border_cells(frame: dict) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame["bbox"]
    gap = frame["gap"]
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return [cell for cell in cells if cell != gap]


def _frame_signature(frame: dict) -> tuple[int, int, int, int, int, int, str]:
    r0, c0, r1, c1 = frame["bbox"]
    gr, gc = frame["gap"]
    return (r0, c0, r1, c1, gr, gc, frame["dir"])


def _parse_frames(grid: np.ndarray) -> list[dict]:
    comps = _components(grid == FRAME)
    frames: list[dict] = []
    for comp in comps:
        rs = comp[:, 0]
        cs = comp[:, 1]
        r0, r1 = int(rs.min()), int(rs.max())
        c0, c1 = int(cs.min()), int(cs.max())
        hh, ww = r1 - r0 + 1, c1 - c0 + 1
        assert 3 <= hh <= 5 and 3 <= ww <= 5, "frame bbox must be 3..5 in each dimension"

        border = {(r0, c) for c in range(c0, c1 + 1)}
        border |= {(r1, c) for c in range(c0, c1 + 1)}
        border |= {(r, c0) for r in range(r0 + 1, r1)}
        border |= {(r, c1) for r in range(r0 + 1, r1)}
        cells = set(map(tuple, comp))
        missing = border - cells
        extra = cells - border
        assert not extra, "blue pixels must lie on rectangle borders"
        assert len(missing) == 1, "each frame must have exactly one border gap"
        gap = next(iter(missing))
        assert gap not in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}, "gap cannot be a corner"
        if gap[0] == r0:
            direction = "U"
        elif gap[0] == r1:
            direction = "D"
        elif gap[1] == c0:
            direction = "L"
        else:
            direction = "R"
        frames.append({"bbox": (r0, c0, r1, c1), "gap": gap, "dir": direction})

    assert 3 <= len(frames) <= 4, "expected 3 or 4 blue frames"
    return sorted(frames, key=_frame_signature)


def _extract_sprite(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    orange = np.argwhere(grid == SPRITE)
    assert 3 <= len(orange) <= 5, "sprite must have 3..5 orange cells"

    h, w = grid.shape
    windows = [(0, 0), (0, w - 3), (h - 3, 0), (h - 3, w - 3)]
    orange_set = set(map(tuple, orange))
    candidates = []
    for r0, c0 in windows:
        cells = {(r, c) for r in range(r0, r0 + 3) for c in range(c0, c0 + 3)}
        if orange_set <= cells:
            candidates.append((r0, c0))
    assert len(candidates) == 1, "sprite must occupy exactly one corner 3x3 window"
    r0, c0 = candidates[0]
    tile = grid[r0 : r0 + 3, c0 : c0 + 3].copy()
    assert np.all((tile == BACKGROUND) | (tile == SPRITE)), "corner tile must contain only sprite and background"
    assert _mask_is_connected(tile == SPRITE), "sprite must be 4-connected"
    return tile, (r0, c0)


def _blue_mask(shape: tuple[int, int], frames: list[dict]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for frame in frames:
        for r, c in _frame_border_cells(frame):
            mask[r, c] = True
    return mask


def _ray_coverage(shape: tuple[int, int], frames: list[dict]) -> np.ndarray:
    h, w = shape
    blue = _blue_mask(shape, frames)
    counts = np.zeros(shape, dtype=int)
    for frame in frames:
        r, c = frame["gap"]
        dr, dc = DIRS[frame["dir"]]
        while 0 <= r < h and 0 <= c < w and not blue[r, c]:
            counts[r, c] += 1
            r += dr
            c += dc
    return counts


def _sorted_intersections(counts: np.ndarray) -> list[tuple[int, int]]:
    points = [(int(r), int(c)) for r, c in np.argwhere(counts >= 2)]
    return sorted(points, key=lambda rc: (rc[1], rc[0]))


def _transform_from_parts(
    grid: np.ndarray,
    frames: list[dict],
    sprite_tile: np.ndarray,
    sprite_origin: tuple[int, int],
) -> np.ndarray:
    out = grid.copy()
    blue = _blue_mask(grid.shape, frames)

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = FILL

    counts = _ray_coverage(grid.shape, frames)
    out[(counts > 0) & (~blue)] = FILL

    sprite_cells = [(int(r), int(c)) for r, c in np.argwhere(sprite_tile == SPRITE)]
    for ir, ic in _sorted_intersections(counts):
        dests: list[tuple[int, int]] = []
        ok = True
        for sr, sc in sprite_cells:
            rr = ir + sr - 1
            cc = ic + sc - 1
            if not (0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]) or blue[rr, cc]:
                ok = False
                break
            dests.append((rr, cc))
        if ok:
            for rr, cc in dests:
                out[rr, cc] = SPRITE

    sr0, sc0 = sprite_origin
    sprite_mask = sprite_tile == SPRITE
    corner = out[sr0 : sr0 + 3, sc0 : sc0 + 3]
    corner[sprite_mask] = BACKGROUND
    out[blue] = FRAME
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each gapped blue frame, extend cyan rays from its gap, and stamp the corner sprite at ray intersections.

    Input structure:
    - Black background with 3 or 4 blue hollow rectangular frames, each 3..5 cells wide and tall, with exactly one missing non-corner border cell.
    - One orange 3x3 sprite template sits in a corner 3x3 window; only 3..5 cells of that window are orange and they form one 4-connected shape.
    - Frames are separated from each other and from the corner sprite.

    Rule:
    - For every blue frame, fill its strict interior with cyan.
    - The missing border cell determines an outward ray direction: top gap shoots up, bottom down, left gap left, right gap right.
    - Each ray covers its gap cell and continues cell by cell until the grid edge or until the next blue border cell, without passing through blue.
    - Any cell covered by at least two different rays is an intersection; center the extracted 3x3 sprite there and paste only its orange cells, skipping that stamp if any orange cell would leave the grid or hit blue.
    - Remove the original corner sprite and keep all blue frame borders unchanged.

    Key insight:
    - The orange stamps are not explicitly marked in the input; they emerge only after globally tracing all gap-directed rays and locating their intersections.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 3, "expected a 2D grid"
    frames = _parse_frames(g)
    sprite_tile, sprite_origin = _extract_sprite(g)
    return _transform_from_parts(g, frames, sprite_tile, sprite_origin)


def _pick_spaced_values(low: int, high: int, count: int, min_gap: int) -> list[int] | None:
    choices = list(range(low, high + 1))
    random.shuffle(choices)
    picked: list[int] = []
    for value in choices:
        if all(abs(value - other) >= min_gap for other in picked):
            picked.append(value)
            if len(picked) == count:
                return sorted(picked)
    return None


def _reserve_mask(h: int, w: int, intersections: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r0, c0 in [(0, 0), (0, w - 3), (h - 3, 0), (h - 3, w - 3)]:
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(h, r0 + 4)
        cc1 = min(w, c0 + 4)
        mask[rr0:rr1, cc0:cc1] = True
    for r, c in intersections:
        mask[r - 1 : r + 2, c - 1 : c + 2] = True
    return mask


def _frame_candidates(spec: dict, h: int, w: int, reserved: np.ndarray) -> list[dict]:
    gap_r, gap_c = spec["gap"]
    direction = spec["dir"]
    candidates: list[dict] = []
    for hh in range(3, 6):
        for ww in range(3, 6):
            if direction == "U":
                r0, r1 = gap_r, gap_r + hh - 1
                if r1 >= h:
                    continue
                offsets = range(1, ww - 1)
                for off in offsets:
                    c0, c1 = gap_c - off, gap_c - off + ww - 1
                    if c0 < 0 or c1 >= w:
                        continue
                    if reserved[r0 : r1 + 1, c0 : c1 + 1].any():
                        continue
                    candidates.append({"bbox": (r0, c0, r1, c1), "gap": (gap_r, gap_c), "dir": direction})
            elif direction == "D":
                r0, r1 = gap_r - hh + 1, gap_r
                if r0 < 0:
                    continue
                offsets = range(1, ww - 1)
                for off in offsets:
                    c0, c1 = gap_c - off, gap_c - off + ww - 1
                    if c0 < 0 or c1 >= w:
                        continue
                    if reserved[r0 : r1 + 1, c0 : c1 + 1].any():
                        continue
                    candidates.append({"bbox": (r0, c0, r1, c1), "gap": (gap_r, gap_c), "dir": direction})
            elif direction == "L":
                c0, c1 = gap_c, gap_c + ww - 1
                if c1 >= w:
                    continue
                offsets = range(1, hh - 1)
                for off in offsets:
                    r0, r1 = gap_r - off, gap_r - off + hh - 1
                    if r0 < 0 or r1 >= h:
                        continue
                    if reserved[r0 : r1 + 1, c0 : c1 + 1].any():
                        continue
                    candidates.append({"bbox": (r0, c0, r1, c1), "gap": (gap_r, gap_c), "dir": direction})
            else:
                c0, c1 = gap_c - ww + 1, gap_c
                if c0 < 0:
                    continue
                offsets = range(1, hh - 1)
                for off in offsets:
                    r0, r1 = gap_r - off, gap_r - off + hh - 1
                    if r0 < 0 or r1 >= h:
                        continue
                    if reserved[r0 : r1 + 1, c0 : c1 + 1].any():
                        continue
                    candidates.append({"bbox": (r0, c0, r1, c1), "gap": (gap_r, gap_c), "dir": direction})
    random.shuffle(candidates)
    return candidates


def _expanded_bbox(bbox: tuple[int, int, int, int], h: int, w: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return max(0, r0 - 1), max(0, c0 - 1), min(h - 1, r1 + 1), min(w - 1, c1 + 1)


def _place_frames(specs: list[dict], h: int, w: int, reserved: np.ndarray) -> list[dict] | None:
    by_idx = [None] * len(specs)
    candidates = [_frame_candidates(spec, h, w, reserved) for spec in specs]
    if any(not cand for cand in candidates):
        return None
    order = sorted(range(len(specs)), key=lambda i: len(candidates[i]))
    blocked = np.zeros((h, w), dtype=bool)

    def search(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        for frame in candidates[idx]:
            er0, ec0, er1, ec1 = _expanded_bbox(frame["bbox"], h, w)
            if blocked[er0 : er1 + 1, ec0 : ec1 + 1].any():
                continue
            by_idx[idx] = frame
            blocked[er0 : er1 + 1, ec0 : ec1 + 1] = True
            if search(pos + 1):
                return True
            blocked[er0 : er1 + 1, ec0 : ec1 + 1] = False
            by_idx[idx] = None
        return False

    if not search(0):
        return None
    return [frame for frame in by_idx if frame is not None]


def _plan_vbundle(h: int, w: int) -> tuple[list[dict], list[tuple[int, int]]] | None:
    intersections_n = random.choice(INTERSECTION_CHOICES)
    if intersections_n == 1:
        total_frames = 3
    elif intersections_n == 3:
        total_frames = 4
    else:
        total_frames = random.choice([3, 4])
    extras = total_frames - (intersections_n + 1)

    row_low = 8 if extras == 2 else 6
    row_high = h - 9 if extras == 2 else h - 7
    if row_low > row_high:
        return None

    rows = _pick_spaced_values(row_low, row_high, intersections_n, 4)
    if rows is None:
        return None
    col = random.randint(6, w - 7)

    vertical_dir = random.choice(["U", "D"])
    if vertical_dir == "U":
        low, high = max(rows) + 4, h - 3
    else:
        low, high = 2, min(rows) - 4
    if low > high:
        return None
    specs = [{"gap": (random.randint(low, high), col), "dir": vertical_dir}]

    for row in rows:
        direction = random.choice(["L", "R"])
        if direction == "L":
            low, high = col + 4, w - 3
        else:
            low, high = 2, col - 4
        if low > high:
            return None
        specs.append({"gap": (row, random.randint(low, high)), "dir": direction})

    if extras:
        extra_cols = _pick_spaced_values(6, w - 7, extras, 4)
        if extra_cols is None or any(abs(extra_col - col) < 4 for extra_col in extra_cols):
            return None
        side_specs: list[tuple[str, tuple[int, int]]] = []
        above_low, above_high = 2, min(rows) - 6
        below_low, below_high = max(rows) + 6, h - 3
        if extras == 2:
            if above_low > above_high or below_low > below_high:
                return None
            side_specs = [
                ("U", (above_low, above_high)),
                ("D", (below_low, below_high)),
            ]
        else:
            choices = []
            if above_low <= above_high:
                choices.append(("U", (above_low, above_high)))
            if below_low <= below_high:
                choices.append(("D", (below_low, below_high)))
            if not choices:
                return None
            side_specs = [random.choice(choices)]
        for extra_col, (direction, (low, high)) in zip(extra_cols, side_specs, strict=False):
            specs.append({"gap": (random.randint(low, high), extra_col), "dir": direction})

    intersections = [(row, col) for row in rows]
    return specs, intersections


def _plan_hbundle(h: int, w: int) -> tuple[list[dict], list[tuple[int, int]]] | None:
    intersections_n = random.choice(INTERSECTION_CHOICES)
    if intersections_n == 1:
        total_frames = 3
    elif intersections_n == 3:
        total_frames = 4
    else:
        total_frames = random.choice([3, 4])
    extras = total_frames - (intersections_n + 1)

    col_low = 8 if extras == 2 else 6
    col_high = w - 9 if extras == 2 else w - 7
    if col_low > col_high:
        return None

    cols = _pick_spaced_values(col_low, col_high, intersections_n, 4)
    if cols is None:
        return None
    row = random.randint(6, h - 7)

    horizontal_dir = random.choice(["L", "R"])
    if horizontal_dir == "L":
        low, high = max(cols) + 4, w - 3
    else:
        low, high = 2, min(cols) - 4
    if low > high:
        return None
    specs = [{"gap": (row, random.randint(low, high)), "dir": horizontal_dir}]

    for col in cols:
        direction = random.choice(["U", "D"])
        if direction == "U":
            low, high = row + 4, h - 3
        else:
            low, high = 2, row - 4
        if low > high:
            return None
        specs.append({"gap": (random.randint(low, high), col), "dir": direction})

    if extras:
        extra_rows = _pick_spaced_values(6, h - 7, extras, 4)
        if extra_rows is None or any(abs(extra_row - row) < 4 for extra_row in extra_rows):
            return None
        side_specs: list[tuple[str, tuple[int, int]]] = []
        left_low, left_high = 2, min(cols) - 6
        right_low, right_high = max(cols) + 6, w - 3
        if extras == 2:
            if left_low > left_high or right_low > right_high:
                return None
            side_specs = [
                ("R", (left_low, left_high)),
                ("L", (right_low, right_high)),
            ]
        else:
            choices = []
            if left_low <= left_high:
                choices.append(("R", (left_low, left_high)))
            if right_low <= right_high:
                choices.append(("L", (right_low, right_high)))
            if not choices:
                return None
            side_specs = [random.choice(choices)]
        for extra_row, (direction, (low, high)) in zip(extra_rows, side_specs, strict=False):
            specs.append({"gap": (extra_row, random.randint(low, high)), "dir": direction})

    intersections = [(row, col) for col in cols]
    return specs, intersections


def _choose_plan(h: int, w: int) -> tuple[list[dict], list[tuple[int, int]]] | None:
    return (_plan_vbundle if random.random() < 0.5 else _plan_hbundle)(h, w)


def _corner_windows(h: int, w: int) -> list[tuple[int, int]]:
    return [(0, 0), (0, w - 3), (h - 3, 0), (h - 3, w - 3)]


def _render_input(shape: tuple[int, int], frames: list[dict], sprite_tile: np.ndarray, sprite_origin: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    for frame in frames:
        for r, c in _frame_border_cells(frame):
            grid[r, c] = FRAME
    sr0, sc0 = sprite_origin
    corner = grid[sr0 : sr0 + 3, sc0 : sc0 + 3]
    corner[sprite_tile == SPRITE] = SPRITE
    return grid


def _sprite_window_clear(frame_cells: set[tuple[int, int]], sprite_origin: tuple[int, int]) -> bool:
    sr0, sc0 = sprite_origin
    for r in range(sr0, sr0 + 3):
        for c in range(sc0, sc0 + 3):
            if any(max(abs(r - fr), abs(c - fc)) <= 1 for fr, fc in frame_cells):
                return False
    return True


def _oracle_output(
    inp: np.ndarray,
    frames: list[dict],
    sprite_tile: np.ndarray,
    sprite_origin: tuple[int, int],
) -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray]:
    counts = _ray_coverage(inp.shape, frames)
    out = inp.copy()
    blue = _blue_mask(inp.shape, frames)

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = FILL

    for r, c in np.argwhere(counts > 0):
        rr, cc = int(r), int(c)
        if not blue[rr, cc]:
            out[rr, cc] = FILL

    intersections = _sorted_intersections(counts)
    orange_cells = [(int(r), int(c)) for r, c in np.argwhere(sprite_tile == SPRITE)]
    for ir, ic in intersections:
        placements = []
        valid = True
        for sr, sc in orange_cells:
            rr = ir + sr - 1
            cc = ic + sc - 1
            if not (0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]) or blue[rr, cc]:
                valid = False
                break
            placements.append((rr, cc))
        if valid:
            for rr, cc in placements:
                out[rr, cc] = SPRITE

    sr0, sc0 = sprite_origin
    corner = out[sr0 : sr0 + 3, sc0 : sc0 + 3]
    corner[sprite_tile == SPRITE] = BACKGROUND
    out[blue] = FRAME
    return out, intersections, counts


def _validate_semantics(
    inp: np.ndarray,
    frames: list[dict],
    sprite_tile: np.ndarray,
    sprite_origin: tuple[int, int],
    planned_intersections: list[tuple[int, int]],
) -> np.ndarray | None:
    parsed_frames = _parse_frames(inp)
    if sorted(_frame_signature(frame) for frame in parsed_frames) != sorted(_frame_signature(frame) for frame in frames):
        return None

    parsed_tile, parsed_origin = _extract_sprite(inp)
    if parsed_origin != sprite_origin or not np.array_equal(parsed_tile, sprite_tile):
        return None

    frame_cells = {cell for frame in frames for cell in _frame_border_cells(frame)}
    if not _sprite_window_clear(frame_cells, sprite_origin):
        return None

    counts = _ray_coverage(inp.shape, parsed_frames)
    intersections = _sorted_intersections(counts)
    if intersections != sorted(planned_intersections, key=lambda rc: (rc[1], rc[0])):
        return None
    if not (1 <= len(intersections) <= 3):
        return None
    if len({frame["dir"] for frame in frames}) < 2:
        return None

    orange_cells = [(int(r), int(c)) for r, c in np.argwhere(sprite_tile == SPRITE)]
    blue = _blue_mask(inp.shape, frames)
    for ir, ic in intersections:
        for sr, sc in orange_cells:
            rr = ir + sr - 1
            cc = ic + sc - 1
            if not (0 <= rr < inp.shape[0] and 0 <= cc < inp.shape[1]) or blue[rr, cc]:
                return None

    expected, _, _ = _oracle_output(inp, frames, sprite_tile, sprite_origin)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return expected


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)

    plan = _choose_plan(h, w)
    if plan is None:
        return None
    specs, planned_intersections = plan

    reserved = _reserve_mask(h, w, planned_intersections)
    frames = _place_frames(specs, h, w, reserved)
    if frames is None:
        return None
    frames = sorted(frames, key=_frame_signature)

    counts = _ray_coverage((h, w), frames)
    actual_intersections = _sorted_intersections(counts)
    if actual_intersections != sorted(planned_intersections, key=lambda rc: (rc[1], rc[0])):
        return None

    for (r0, c0), (r1, c1) in itertools.combinations(actual_intersections, 2):
        if max(abs(r0 - r1), abs(c0 - c1)) < 3:
            return None

    sprite_mask = random.choice(SPRITE_MASKS)
    sprite_tile = np.full((3, 3), BACKGROUND, dtype=int)
    sprite_tile[sprite_mask] = SPRITE

    ray_mask = counts > 0
    frame_cells = {cell for frame in frames for cell in _frame_border_cells(frame)}
    sprite_candidates = []
    for origin in _corner_windows(h, w):
        sr0, sc0 = origin
        window = ray_mask[sr0 : sr0 + 3, sc0 : sc0 + 3]
        if window.any():
            continue
        if not _sprite_window_clear(frame_cells, origin):
            continue
        sprite_candidates.append(origin)
    if not sprite_candidates:
        return None
    sprite_origin = random.choice(sprite_candidates)

    inp = _render_input((h, w), frames, sprite_tile, sprite_origin)
    if _validate_semantics(inp, frames, sprite_tile, sprite_origin, planned_intersections) is None:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 10")
