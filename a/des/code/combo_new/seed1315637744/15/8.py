import random
from collections import defaultdict

import numpy as np


BG = 0
MIRROR = 4
MARKER = 5
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
FRAME_COLORS = [1, 2, 3, 6, 7, 8, 9]
INNER_SIZES = (3, 5)
CANVAS_SIZE = 30
MIN_SEG_LEN = 2
MAX_SEG_LEN = 5
MAX_GENERATION_ATTEMPTS = 400
CROP_MARGIN_MAX = 2


def _rot_cw_dir(d: int) -> int:
    return (d + 1) % 4


def _opp_dir(d: int) -> int:
    return (d + 2) % 4


def _rot_cw(arr: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(arr, -(turns % 4))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
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
                for dr, dc in DIRS:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def _is_rect_frame(coords: np.ndarray) -> tuple[bool, tuple[int, int, int, int] | None]:
    r0, r1, c0, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return False, None
    if int(coords.shape[0]) != 2 * (height + width) - 4:
        return False, None
    coord_set = {tuple(map(int, xy)) for xy in coords}
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if on_border != ((r, c) in coord_set):
                return False, None
    return True, (r0, r1, c0, c1)


def _marker_for_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[tuple[int, int], int] | None:
    r0, r1, c0, c1 = bbox
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    candidates = []
    for (r, c), d in [
        ((r0 - 1, mid_c), 0),
        ((mid_r, c1 + 1), 1),
        ((r1 + 1, mid_c), 2),
        ((mid_r, c0 - 1), 3),
    ]:
        if _in_bounds(grid.shape, r, c) and int(grid[r, c]) == MARKER:
            candidates.append(((r, c), d))
    if len(candidates) != 1:
        return None
    return candidates[0]


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames = []
    colors = set(map(int, np.unique(grid))) - {BG, MIRROR, MARKER}
    for color in colors:
        for comp in _components(grid == color):
            ok, box = _is_rect_frame(comp)
            if not ok or box is None:
                continue
            marker_info = _marker_for_bbox(grid, box)
            if marker_info is None:
                continue
            r0, r1, c0, c1 = box
            interior = grid[r0 + 1 : r1, c0 + 1 : c1].copy()
            kind = "receiver" if np.all(interior == BG) else "emitter"
            frames.append(
                {
                    "color": int(color),
                    "bbox": box,
                    "marker": marker_info[0],
                    "side_dir": int(marker_info[1]),
                    "kind": kind,
                    "interior": interior,
                }
            )
    return frames


def _trace_turns(
    grid: np.ndarray,
    start: tuple[int, int],
    start_dir: int,
    target: tuple[int, int],
    expected_dir: int,
) -> int:
    h, w = grid.shape
    pos = start
    direction = start_dir
    turns = 0
    seen = {(pos, direction)}
    while True:
        dr, dc = DIRS[direction]
        pos = (pos[0] + dr, pos[1] + dc)
        r, c = pos
        assert 0 <= r < h and 0 <= c < w, "beam left the grid"
        if pos == target:
            assert direction == expected_dir, "beam reached receiver from the wrong side"
            return turns
        cell = int(grid[r, c])
        assert cell in (BG, MIRROR), "beam crossed a non-path object"
        if cell == MIRROR:
            direction = _rot_cw_dir(direction)
            turns += 1
        state = (pos, direction)
        assert state not in seen, "beam path must not cycle"
        seen.add(state)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each emitter's interior pattern into its same-color receiver after following the mirror-guided path and rotating on every clockwise reflection.

    Input structure:
    - The background is black.
    - Several colored hollow rectangular frames appear in same-color emitter/receiver pairs.
    - Each emitter has a non-empty interior pattern, while its paired receiver has an empty black interior.
    - A single gray marker sits just outside the midpoint of one frame side, and yellow singleton mirror pixels lie on the travel path.

    Rule:
    - Pair frames by their border color; within each pair, the non-empty frame is the emitter and the empty frame is the receiver.
    - The emitter marker side gives the initial outward travel direction.
    - Follow the path through black cells; every time it hits a yellow mirror pixel, rotate the travel direction 90 degrees clockwise and rotate the carried pattern 90 degrees clockwise as well.
    - When the path reaches the paired receiver marker from the outside, fill the receiver interior with the final rotated pattern while leaving emitters, borders, markers, and mirrors unchanged.

    Key insight:
    - The task combines three linked cues: same-color frame pairing, gray side markers for path endpoints, and yellow mirror pixels that simultaneously turn both the route and the transmitted pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames = _find_frames(g)
    assert frames, "expected at least one emitter/receiver pair"

    by_color: dict[int, list[dict]] = defaultdict(list)
    for frame in frames:
        by_color[int(frame["color"])].append(frame)

    out = g.copy()
    for color, group in by_color.items():
        emitters = [item for item in group if item["kind"] == "emitter"]
        receivers = [item for item in group if item["kind"] == "receiver"]
        assert len(group) == 2 and len(emitters) == 1 and len(receivers) == 1, (
            f"color {color} must form exactly one emitter and one receiver"
        )
        emitter = emitters[0]
        receiver = receivers[0]

        turns = _trace_turns(
            g,
            start=emitter["marker"],
            start_dir=emitter["side_dir"],
            target=receiver["marker"],
            expected_dir=_opp_dir(receiver["side_dir"]),
        )
        rotated = _rot_cw(emitter["interior"], turns)

        r0, r1, c0, c1 = receiver["bbox"]
        target = out[r0 + 1 : r1, c0 + 1 : c1]
        assert rotated.shape == target.shape, "receiver interior must match the rotated pattern size"
        out[r0 + 1 : r1, c0 + 1 : c1] = rotated

    return out


def _frame_bbox_from_marker(
    marker: tuple[int, int],
    side_dir: int,
    outer_h: int,
    outer_w: int,
) -> tuple[int, int, int, int]:
    r, c = marker
    if side_dir == 0:
        r0 = r + 1
        c0 = c - outer_w // 2
    elif side_dir == 1:
        r0 = r - outer_h // 2
        c0 = c - outer_w
    elif side_dir == 2:
        r0 = r - outer_h
        c0 = c - outer_w // 2
    else:
        r0 = r - outer_h // 2
        c0 = c + 1
    return r0, r0 + outer_h - 1, c0, c0 + outer_w - 1


def _box_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, r1, c0, c1 = bbox
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _perimeter_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, r1, c0, c1 = bbox
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _expand(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    expanded = set()
    for r, c in cells:
        for rr in range(max(0, r - 1), min(shape[0], r + 2)):
            for cc in range(max(0, c - 1), min(shape[1], c + 2)):
                expanded.add((rr, cc))
    return expanded


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, r1, c0, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _walk_path(
    start: tuple[int, int],
    start_dir: int,
    lengths: list[int],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[int]] | None:
    pos = start
    path = [start]
    special = [start]
    segment_dirs = []
    seen = {start}
    direction = start_dir
    for index, steps in enumerate(lengths):
        segment_dirs.append(direction)
        for _ in range(steps):
            dr, dc = DIRS[direction]
            pos = (pos[0] + dr, pos[1] + dc)
            if pos in seen:
                return None
            seen.add(pos)
            path.append(pos)
        special.append(pos)
        if index + 1 < len(lengths):
            direction = _rot_cw_dir(direction)
    return special, path, segment_dirs


def _make_pattern(shape: tuple[int, int], palette: list[int], turns: int) -> np.ndarray:
    height, width = shape
    color_count = 1 if random.random() < 0.55 else 2
    color_count = min(color_count, len(palette))
    for _ in range(200):
        colors = random.sample(palette, color_count)
        fill_prob = random.uniform(0.35, 0.7)
        arr = np.zeros((height, width), dtype=int)
        for r in range(height):
            for c in range(width):
                if random.random() < fill_prob:
                    arr[r, c] = random.choice(colors)
        filled = int(np.count_nonzero(arr))
        if filled == 0 or filled == arr.size:
            continue
        if turns % 4 and np.array_equal(arr, _rot_cw(arr, turns)):
            continue
        return arr

    arr = np.zeros((height, width), dtype=int)
    color = palette[0]
    arr[:, 0] = color
    arr[-1, :] = color
    if turns % 4 and np.array_equal(arr, _rot_cw(arr, turns)):
        arr[0, -1] = palette[-1]
    return arr


def _crop(grid: np.ndarray) -> np.ndarray:
    coords = np.argwhere(grid != BG)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    top = random.randint(0, min(CROP_MARGIN_MAX, int(r0)))
    bottom = random.randint(0, min(CROP_MARGIN_MAX, int(grid.shape[0] - 1 - r1)))
    left = random.randint(0, min(CROP_MARGIN_MAX, int(c0)))
    right = random.randint(0, min(CROP_MARGIN_MAX, int(grid.shape[1] - 1 - c1)))
    return grid[r0 - top : r1 + bottom + 1, c0 - left : c1 + right + 1]


def _gen_attempt() -> np.ndarray | None:
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    blocked = np.zeros_like(canvas, dtype=bool)

    pair_count = random.randint(1, 3)
    frame_colors = random.sample(FRAME_COLORS, pair_count)
    pattern_palette = [color for color in FRAME_COLORS if color not in frame_colors]

    for frame_color in frame_colors:
        placed = False
        for _ in range(600):
            inner_h = random.choice(INNER_SIZES)
            inner_w = random.choice(INNER_SIZES)
            turns = random.randint(0, 3)
            emitter_outer = (inner_h + 2, inner_w + 2)
            receiver_inner = (inner_h, inner_w) if turns % 2 == 0 else (inner_w, inner_h)
            receiver_outer = (receiver_inner[0] + 2, receiver_inner[1] + 2)

            start_dir = random.randrange(4)
            lengths = [random.randint(MIN_SEG_LEN, MAX_SEG_LEN) for _ in range(turns + 1)]
            start = (
                random.randint(1, CANVAS_SIZE - 2),
                random.randint(1, CANVAS_SIZE - 2),
            )

            walked = _walk_path(start, start_dir, lengths)
            if walked is None:
                continue
            special, path, segment_dirs = walked
            if not all(_in_bounds(canvas.shape, r, c) for r, c in path):
                continue

            end = special[-1]
            final_dir = segment_dirs[-1]
            emitter_bbox = _frame_bbox_from_marker(start, start_dir, emitter_outer[0], emitter_outer[1])
            receiver_side = _opp_dir(final_dir)
            receiver_bbox = _frame_bbox_from_marker(
                end,
                receiver_side,
                receiver_outer[0],
                receiver_outer[1],
            )

            boxes = [emitter_bbox, receiver_bbox]
            if not all(
                0 <= r0 and r1 < CANVAS_SIZE and 0 <= c0 and c1 < CANVAS_SIZE
                for r0, r1, c0, c1 in boxes
            ):
                continue

            emitter_area = _box_cells(emitter_bbox)
            receiver_area = _box_cells(receiver_bbox)
            path_cells = set(path)
            if emitter_area & receiver_area:
                continue
            if (emitter_area | receiver_area) & path_cells:
                continue

            reserve = emitter_area | receiver_area | path_cells
            halo = _expand(reserve, canvas.shape)
            if any(blocked[r, c] for r, c in halo):
                continue

            pattern = _make_pattern((inner_h, inner_w), pattern_palette, turns)

            _draw_frame(canvas, emitter_bbox, frame_color)
            _draw_frame(canvas, receiver_bbox, frame_color)
            canvas[start] = MARKER
            canvas[end] = MARKER
            for mirror in special[1:-1]:
                canvas[mirror] = MIRROR

            er0, er1, ec0, ec1 = emitter_bbox
            canvas[er0 + 1 : er1, ec0 + 1 : ec1] = pattern

            for r, c in halo:
                blocked[r, c] = True
            placed = True
            break

        if not placed:
            return None

    cropped = _crop(canvas)
    try:
        solved = main(cropped)
    except AssertionError:
        return None
    if solved.shape != cropped.shape or not np.any(solved != cropped):
        return None
    return cropped


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-transmission ARC input")
