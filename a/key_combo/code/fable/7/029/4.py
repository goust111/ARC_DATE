import random
from collections import Counter

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9


DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
FRAME_COLORS = [Color.BLUE, Color.RED, Color.YELLOW, Color.GREY, Color.ORANGE, Color.TEAL, Color.MAROON]
MARKER_COLOR = Color.PINK

GEN_SIZE_RANGE = (14, 24)
GEN_TURN_COUNT_RANGE = (1, 5)
GEN_SEGMENT_LENGTHS = (1, 2, 3, 4, 5, 6)
GEN_FRAME_LONG = (4, 5, 6)
GEN_FRAME_SHORT = (3, 4, 5, 6)
GEN_MAX_ATTEMPTS = 500


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= int(r) < int(h) and 0 <= int(c) < int(w)


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid)[r0 : r1 + 1, c0 : c1 + 1].copy()


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _turn_left(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return -dc, dr


def _turn_right(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return dc, -dr


def _frame_wall_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _bbox_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _is_frame_component(grid: np.ndarray, color: int, coords: np.ndarray) -> bool:
    bbox = bbox_coords(coords)
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if len(coords) != 2 * h + 2 * w - 4:
        return False

    patch = crop_bbox(grid, bbox)
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.all(patch[border] == int(color)):
        return False
    if np.any(patch[~border] == int(color)):
        return False
    return True


def _extract_scene(grid: np.ndarray) -> tuple[int, tuple[int, int], dict[tuple[int, int], dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)

    starts = np.argwhere(g == Color.GREEN)
    assert len(starts) == 1, "expected exactly one green start pixel"
    start = tuple(map(int, starts[0]))

    frames: list[dict] = []
    markers: list[tuple[int, int, int]] = []
    for color in sorted(set(map(int, g.ravel())) - {bg}):
        comps = components_mask(g == color)
        for coords in comps:
            if color == Color.GREEN:
                assert len(coords) == 1 and tuple(map(int, coords[0])) == start, "unexpected extra green cells"
                continue
            if len(coords) == 1:
                r, c = map(int, coords[0])
                markers.append((r, c, color))
                continue
            assert _is_frame_component(g, color, coords), "unexpected non-frame component"
            bbox = bbox_coords(coords)
            frames.append(
                {
                    "color": int(color),
                    "bbox": bbox,
                    "walls": set(_frame_wall_cells(bbox)),
                    "marker": None,
                }
            )

    assert frames, "expected at least one frame"
    assert len(markers) == len(frames), "expected one marker per frame"

    for r, c, color in markers:
        owner = None
        for frame in frames:
            r0, c0, r1, c1 = frame["bbox"]
            if r0 < r < r1 and c0 < c < c1:
                if owner is not None:
                    raise AssertionError("marker lies inside multiple frames")
                owner = frame
        assert owner is not None, "marker must lie inside a frame"
        assert owner["marker"] is None, "frame must have exactly one marker"
        owner["marker"] = (r, c, color)

    for frame in frames:
        assert frame["marker"] is not None, "missing frame marker"

    wall_to_frame: dict[tuple[int, int], dict] = {}
    for frame in frames:
        for cell in frame["walls"]:
            assert cell not in wall_to_frame, "frames may not overlap"
            wall_to_frame[cell] = frame

    return bg, start, wall_to_frame, frames


def _initial_direction(grid: np.ndarray, bg: int, start: tuple[int, int], wall_to_frame: dict[tuple[int, int], dict]) -> tuple[int, int]:
    hits = []
    sr, sc = start
    for dr, dc in DIRS:
        r, c = sr + dr, sc + dc
        while in_bounds(grid.shape, r, c):
            if int(grid[r, c]) != bg:
                if (r, c) in wall_to_frame:
                    hits.append((dr, dc))
                break
            r += dr
            c += dc
    assert len(hits) == 1, "start must face exactly one frame wall"
    return hits[0]


def _marker_turn_is_left(frame: dict, direction: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = frame["bbox"]
    mr, mc, _ = frame["marker"]
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    dr, dc = direction
    left_r, left_c = -dc, dr
    signed = (mr - center_r) * left_r + (mc - center_c) * left_c
    assert signed != 0, "marker may not lie on the heading centerline"
    return signed > 0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace a green path that turns at hollow rectangular signboards according to each internal marker's side relative to the incoming heading.

    Input structure:
    - A black background contains exactly one green start pixel and one or more monochrome hollow rectangular frames.
    - Each frame encloses exactly one marker pixel strictly inside the frame and not on the heading-relative centerline used when that frame is read.
    - The start has exactly one cardinal sightline whose first non-background cell is a frame wall.

    Rule:
    - Move from the start in that unique initial direction, painting every visited background cell green.
    - If the next cell is a frame wall, do not enter the frame; instead, read whether the enclosed marker lies on the left or right half of the frame interior relative to the current heading, then turn left or right in place by 90 degrees.
    - Keep all frame walls, marker pixels, and the original start unchanged, and stop when the next step would leave the grid.

    Key insight:
    - The marker does not encode a fixed global up/down or left/right instruction; its meaning must be recomputed in the traveler’s local coordinate frame at the moment a specific wall is hit.
    """
    g = np.asarray(grid, dtype=int)
    bg, start, wall_to_frame, _frames = _extract_scene(g)
    direction = _initial_direction(g, bg, start, wall_to_frame)

    out = g.copy()
    r, c = start
    seen_states: set[tuple[int, int, int, int]] = set()

    while True:
        state = (r, c, direction[0], direction[1])
        assert state not in seen_states, "path must terminate rather than loop"
        seen_states.add(state)

        nr = r + direction[0]
        nc = c + direction[1]
        if not in_bounds(g.shape, nr, nc):
            return out

        value = int(g[nr, nc])
        if value == bg:
            out[nr, nc] = Color.GREEN
            r, c = nr, nc
            continue

        assert (nr, nc) in wall_to_frame, "only frame walls may redirect the path"
        frame = wall_to_frame[(nr, nc)]
        direction = _turn_left(direction) if _marker_turn_is_left(frame, direction) else _turn_right(direction)


def _distance_to_edge(h: int, w: int, pos: tuple[int, int], direction: tuple[int, int]) -> int:
    r, c = pos
    dr, dc = direction
    if dr < 0:
        return r
    if dr > 0:
        return h - 1 - r
    if dc < 0:
        return c
    return w - 1 - c


def _line_cells(pos: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    r, c = pos
    dr, dc = direction
    return [(r + dr * k, c + dc * k) for k in range(1, int(length) + 1)]


def _candidate_marker_cells(
    bbox: tuple[int, int, int, int], direction: tuple[int, int], turn_sign: int
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    dr, dc = direction
    left_r, left_c = -dc, dr
    want_positive = turn_sign < 0

    cells = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            signed = (r - center_r) * left_r + (c - center_c) * left_c
            if signed == 0:
                continue
            if (signed > 0) == want_positive:
                cells.append((r, c))
    return cells


def _frame_candidates(
    h: int, w: int, hit_cell: tuple[int, int], direction: tuple[int, int], turn_sign: int
) -> list[dict]:
    hr, hc = hit_cell
    candidates = []

    if direction in {(0, 1), (0, -1)}:
        heights = GEN_FRAME_LONG
        widths = GEN_FRAME_SHORT
        for fh in heights:
            for fw in widths:
                for offset in range(1, fh - 1):
                    r0 = hr - offset
                    c0 = hc if direction == (0, 1) else hc - fw + 1
                    bbox = (r0, c0, r0 + fh - 1, c0 + fw - 1)
                    if bbox[0] < 0 or bbox[1] < 0 or bbox[2] >= h or bbox[3] >= w:
                        continue
                    marker_cells = _candidate_marker_cells(bbox, direction, turn_sign)
                    if marker_cells:
                        candidates.append({"bbox": bbox, "marker": random.choice(marker_cells)})
    else:
        heights = GEN_FRAME_SHORT
        widths = GEN_FRAME_LONG
        for fh in heights:
            for fw in widths:
                for offset in range(1, fw - 1):
                    r0 = hr if direction == (1, 0) else hr - fh + 1
                    c0 = hc - offset
                    bbox = (r0, c0, r0 + fh - 1, c0 + fw - 1)
                    if bbox[0] < 0 or bbox[1] < 0 or bbox[2] >= h or bbox[3] >= w:
                        continue
                    marker_cells = _candidate_marker_cells(bbox, direction, turn_sign)
                    if marker_cells:
                        candidates.append({"bbox": bbox, "marker": random.choice(marker_cells)})

    random.shuffle(candidates)
    return candidates


def _visible_hit_directions(grid: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    bg = Color.BLACK
    hits = []
    sr, sc = start
    for dr, dc in DIRS:
        r, c = sr + dr, sc + dc
        while in_bounds(grid.shape, r, c):
            if int(grid[r, c]) != bg:
                hits.append((dr, dc))
                break
            r += dr
            c += dc
    return hits


def _build_path(
    h: int,
    w: int,
    start: tuple[int, int],
    direction: tuple[int, int],
    turns: tuple[int, ...],
) -> tuple[list[tuple[int, int]], list[dict]] | None:
    pos = start
    blocked = {start}
    path_cells: list[tuple[int, int]] = []
    frames: list[dict] = []

    for turn_sign in turns:
        placed = False
        for _ in range(40):
            length = random.choice(GEN_SEGMENT_LENGTHS)
            segment = _line_cells(pos, direction, length)
            if any(not in_bounds((h, w), r, c) for r, c in segment):
                continue
            if any(cell in blocked for cell in segment):
                continue

            end = segment[-1]
            hit = (end[0] + direction[0], end[1] + direction[1])
            if not in_bounds((h, w), hit[0], hit[1]):
                continue

            next_direction = _turn_left(direction) if turn_sign < 0 else _turn_right(direction)
            next_step = (end[0] + next_direction[0], end[1] + next_direction[1])
            if not in_bounds((h, w), next_step[0], next_step[1]):
                continue

            frame_options = _frame_candidates(h, w, hit, direction, turn_sign)
            random.shuffle(frame_options)
            for candidate in frame_options[:20]:
                bbox_cells = _bbox_cells(candidate["bbox"])
                if any(cell in blocked for cell in bbox_cells):
                    continue
                if next_step in bbox_cells:
                    continue

                blocked |= set(segment) | bbox_cells
                path_cells.extend(segment)
                frames.append(candidate)
                pos = end
                direction = next_direction
                placed = True
                break

            if placed:
                break

        if not placed:
            return None

    final_len = _distance_to_edge(h, w, pos, direction)
    if not 1 <= final_len <= 10:
        return None
    tail = _line_cells(pos, direction, final_len)
    if any(cell in blocked for cell in tail):
        return None
    return path_cells + tail, frames


def _render_scene(
    h: int,
    w: int,
    start: tuple[int, int],
    path_cells: list[tuple[int, int]],
    frames: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.zeros((h, w), dtype=int)
    out = np.zeros((h, w), dtype=int)
    inp[start] = Color.GREEN
    out[start] = Color.GREEN

    frame_colors = random.sample(FRAME_COLORS, len(frames))
    for frame, color in zip(frames, frame_colors):
        for cell in _frame_wall_cells(frame["bbox"]):
            inp[cell] = int(color)
            out[cell] = int(color)
        mr, mc = frame["marker"]
        inp[mr, mc] = MARKER_COLOR
        out[mr, mc] = MARKER_COLOR

    for cell in path_cells:
        out[cell] = Color.GREEN
    return inp, out


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GEN_SIZE_RANGE)
    w = random.randint(*GEN_SIZE_RANGE)
    turn_count = random.randint(*GEN_TURN_COUNT_RANGE)
    turns = tuple(random.choice((-1, 1)) for _ in range(turn_count))

    start = (random.randint(1, h - 2), random.randint(1, w - 2))
    direction = random.choice(DIRS)

    built = _build_path(h, w, start, direction, turns)
    if built is None:
        return None

    path_cells, frames = built
    if len(path_cells) < turn_count + 2:
        return None

    inp, expected = _render_scene(h, w, start, path_cells, frames)
    if len(_visible_hit_directions(inp, start)) != 1:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid door-sign path puzzle")
