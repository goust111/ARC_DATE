import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7

FRAME_COLORS = (BLUE, GREEN, ORANGE)
PATTERN_COLORS = (RED, GREY)

GRID_MIN = 20
GRID_MAX = 27  # exclusive
FRAME_COUNT_CHOICES = (2, 3)
INTERIOR_SIZE_CHOICES = (3, 4, 5)
MAX_RETRIES = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _rects_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _clip_rect(
    top: int, left: int, height: int, width: int, shape: tuple[int, int]
) -> tuple[int, int, int, int] | None:
    h, w = shape
    r0 = max(0, top)
    c0 = max(0, left)
    r1 = min(h - 1, top + height - 1)
    c1 = min(w - 1, left + width - 1)
    if r0 > r1 or c0 > c1:
        return None
    return (r0, c0, r1, c1)


def _frame_border_cells(frame: dict) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = frame["bbox"]
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _find_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    frames: list[dict] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != BLACK):
        coords = np.argwhere(g == color)
        if coords.size == 0:
            continue
        rows = sorted(set(int(r) for r in coords[:, 0]))
        cols = sorted(set(int(c) for c in coords[:, 1]))
        candidates: list[dict] = []
        for i, r0 in enumerate(rows):
            for r1 in rows[i + 1 :]:
                if r1 - r0 < 2:
                    continue
                for j, c0 in enumerate(cols):
                    for c1 in cols[j + 1 :]:
                        if c1 - c0 < 2:
                            continue
                        if (
                            np.all(g[r0, c0 : c1 + 1] == color)
                            and np.all(g[r1, c0 : c1 + 1] == color)
                            and np.all(g[r0 : r1 + 1, c0] == color)
                            and np.all(g[r0 : r1 + 1, c1] == color)
                        ):
                            candidates.append(
                                {
                                    "color": color,
                                    "bbox": (r0, c0, r1, c1),
                                    "pattern": g[r0 + 1 : r1, c0 + 1 : c1].copy(),
                                }
                            )
        if candidates:
            assert len(candidates) == 1, f"ambiguous frame color {color}"
            frames.append(candidates[0])
    frames.sort(key=lambda f: f["bbox"])
    return frames


def _find_markers(grid: np.ndarray, frames: list[dict]) -> list[tuple[int, int, int]]:
    frame_by_color = {int(frame["color"]): frame for frame in frames}
    frame_cells = set()
    for frame in frames:
        frame_cells |= _frame_border_cells(frame)

    markers: list[tuple[int, int, int]] = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in frame_by_color and (r, c) not in frame_cells:
                markers.append((r, c, color))
    return markers


def _stamp(
    out: np.ndarray, painted: np.ndarray, pattern: np.ndarray, top: int, left: int
) -> None:
    ph, pw = pattern.shape
    for pr in range(ph):
        for pc in range(pw):
            value = int(pattern[pr, pc])
            if value == BLACK:
                continue
            rr = top + pr
            cc = left + pc
            if _in_bounds(out.shape, rr, cc):
                out[rr, cc] = value
                painted[rr, cc] = True


def _render_expected(
    grid: np.ndarray, frames: list[dict], markers: list[tuple[int, int, int]]
) -> tuple[np.ndarray, dict]:
    frame_by_color = {int(frame["color"]): frame for frame in frames}
    out = np.array(grid, dtype=int, copy=True)
    painted = np.zeros(out.shape, dtype=bool)
    overlap_conflict = False
    clipped_count = 0

    for r, c, color in markers:
        pattern = frame_by_color[color]["pattern"]
        ph, pw = pattern.shape
        if r + ph > out.shape[0] or c + pw > out.shape[1]:
            clipped_count += 1
        for pr in range(ph):
            for pc in range(pw):
                value = int(pattern[pr, pc])
                if value == BLACK:
                    continue
                rr = r + pr
                cc = c + pc
                if not _in_bounds(out.shape, rr, cc):
                    continue
                if painted[rr, cc] and int(out[rr, cc]) != value:
                    overlap_conflict = True
                out[rr, cc] = value
                painted[rr, cc] = True

    erased_count = 0
    for r, c, _color in markers:
        if not painted[r, c]:
            out[r, c] = BLACK
            erased_count += 1

    stats = {
        "overlap_conflict": overlap_conflict,
        "clipped_count": clipped_count,
        "erased_count": erased_count,
        "painted": painted,
    }
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """Crop each colored frame interior and stamp it at matching-color marker pixels.

    Input structure
    The grid has a black background, 2-3 complete one-cell-thick monochrome hollow
    rectangles of distinct colors, and single-pixel markers outside the frames whose
    colors match those frame colors. Each frame encloses a small pattern.

    Rule
    Detect every complete hollow rectangle. For each external marker pixel, find the
    same-colored frame, crop the frame's strict interior, and paste that crop with its
    marker as the top-left corner. Black crop cells are transparent, stamps are applied
    in row-major marker order, later non-black pixels overwrite earlier ones, and any
    part leaving the grid is clipped. After stamping, erase marker pixels not covered
    by any non-transparent stamp cell.

    Key insight
    Marker color is the pairing key: it selects which frame to crop, and that cropped
    interior then becomes the transparent sprite that must be stamped in scan order.
    """
    g = np.asarray(grid, dtype=int)
    frames = _find_frames(g)
    assert 2 <= len(frames) <= 3, "expected 2-3 complete hollow frames"
    markers = _find_markers(g, frames)
    out, _stats = _render_expected(g, frames, markers)
    return out


def _make_pattern(height: int, width: int, force_top_left: int | None) -> np.ndarray:
    pattern = np.zeros((height, width), dtype=int)
    cells = [(r, c) for r in range(height) for c in range(width)]
    random.shuffle(cells)

    nonzero_count = random.randint(max(3, (height * width) // 3), max(4, (2 * height * width) // 3))
    nonzero_count = min(nonzero_count, height * width - (1 if force_top_left == BLACK else 0))
    chosen = cells[:nonzero_count]
    for r, c in chosen:
        pattern[r, c] = random.choice(PATTERN_COLORS)

    if force_top_left is not None:
        pattern[0, 0] = force_top_left

    if not np.any(pattern != BLACK):
        rr, cc = random.choice(cells[1:] if force_top_left == BLACK and len(cells) > 1 else cells)
        pattern[rr, cc] = random.choice(PATTERN_COLORS)

    if not np.any(pattern == RED):
        rr, cc = random.choice(cells)
        if not (force_top_left == BLACK and (rr, cc) == (0, 0)):
            pattern[rr, cc] = RED
    if not np.any(pattern == GREY):
        rr, cc = random.choice(cells)
        if not (force_top_left == BLACK and (rr, cc) == (0, 0)):
            pattern[rr, cc] = GREY
    return pattern


def _pattern_key(pattern: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return pattern.shape, pattern.tobytes()


def _choose_overlap_delta(pattern_a: np.ndarray, pattern_b: np.ndarray) -> tuple[int, int] | None:
    ha, wa = pattern_a.shape
    hb, wb = pattern_b.shape
    options: list[tuple[int, int]] = []
    for dr in range(-(hb - 1), ha):
        for dc in range(-(wb - 1), wa):
            if dr < 0 or (dr == 0 and dc <= 0):
                continue
            conflict = False
            overlap = False
            for ra in range(ha):
                for ca in range(wa):
                    rb = ra - dr
                    cb = ca - dc
                    if 0 <= rb < hb and 0 <= cb < wb:
                        overlap = True
                        va = int(pattern_a[ra, ca])
                        vb = int(pattern_b[rb, cb])
                        if va != BLACK and vb != BLACK and va != vb:
                            conflict = True
            if overlap and conflict:
                options.append((dr, dc))
    return random.choice(options) if options else None


def _bbox_contains(bbox: tuple[int, int, int, int], cell: tuple[int, int]) -> bool:
    r, c = cell
    return bbox[0] <= r <= bbox[2] and bbox[1] <= c <= bbox[3]


def _valid_marker_cell(
    cell: tuple[int, int],
    pattern: np.ndarray,
    grid_shape: tuple[int, int],
    frame_bboxes: list[tuple[int, int, int, int]],
    used_cells: set[tuple[int, int]],
) -> bool:
    if cell in used_cells:
        return False
    if any(_bbox_contains(bbox, cell) for bbox in frame_bboxes):
        return False
    clipped = _clip_rect(cell[0], cell[1], pattern.shape[0], pattern.shape[1], grid_shape)
    if clipped is None:
        return False
    return all(not _rects_intersect(clipped, bbox) for bbox in frame_bboxes)


def _stamp_painted_cells(pattern: np.ndarray, top: int, left: int, shape: tuple[int, int]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r in range(pattern.shape[0]):
        for c in range(pattern.shape[1]):
            if int(pattern[r, c]) == BLACK:
                continue
            rr = top + r
            cc = left + c
            if _in_bounds(shape, rr, cc):
                cells.add((rr, cc))
    return cells


def _choose_frame_positions(frames: list[dict], shape: tuple[int, int]) -> bool:
    h, w = shape
    placed: list[tuple[int, int, int, int]] = []
    for frame in frames:
        outer_h = frame["pattern"].shape[0] + 2
        outer_w = frame["pattern"].shape[1] + 2
        candidates = [
            (r0, c0)
            for r0 in range(1, h - outer_h)
            for c0 in range(1, w - outer_w)
        ]
        random.shuffle(candidates)
        choice = None
        for r0, c0 in candidates:
            bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            expanded = (bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2)
            if all(not _rects_intersect(expanded, prev) for prev in placed):
                choice = bbox
                break
        if choice is None:
            return False
        frame["bbox"] = choice
        placed.append(choice)
    return True


def _semantic_check(
    grid: np.ndarray,
    frames_meta: list[dict],
    markers_meta: list[tuple[int, int, int]],
) -> bool:
    detected = _find_frames(grid)
    if len(detected) != len(frames_meta):
        return False

    meta_by_color = {int(frame["color"]): frame for frame in frames_meta}
    det_by_color = {int(frame["color"]): frame for frame in detected}
    if set(meta_by_color) != set(det_by_color):
        return False

    for color, meta in meta_by_color.items():
        det = det_by_color[color]
        if det["bbox"] != meta["bbox"]:
            return False
        if det["pattern"].shape != meta["pattern"].shape:
            return False
        if not np.array_equal(det["pattern"], meta["pattern"]):
            return False
        ph, pw = det["pattern"].shape
        if ph not in INTERIOR_SIZE_CHOICES or pw not in INTERIOR_SIZE_CHOICES:
            return False

    if len({_pattern_key(frame["pattern"]) for frame in frames_meta}) != len(frames_meta):
        return False

    detected_markers = _find_markers(grid, detected)
    if Counter(detected_markers) != Counter(markers_meta):
        return False

    counts = Counter(color for _r, _c, color in markers_meta)
    if any(count < 1 or count > 3 for count in counts.values()):
        return False
    if set(counts) != set(meta_by_color):
        return False

    bboxes = [frame["bbox"] for frame in frames_meta]
    for r, c, color in markers_meta:
        pattern = meta_by_color[color]["pattern"]
        clipped = _clip_rect(r, c, pattern.shape[0], pattern.shape[1], grid.shape)
        if clipped is None:
            return False
        if any(_rects_intersect(clipped, bbox) for bbox in bboxes):
            return False

    expected, stats = _render_expected(grid, detected, detected_markers)
    if not stats["overlap_conflict"]:
        return False
    if stats["clipped_count"] < 1:
        return False
    if stats["erased_count"] < 1:
        return False
    if np.array_equal(expected, grid):
        return False
    if not np.array_equal(main(grid), expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    grid_shape = (
        random.randrange(GRID_MIN, GRID_MAX),
        random.randrange(GRID_MIN, GRID_MAX),
    )
    frame_count = random.choice(FRAME_COUNT_CHOICES)
    colors = random.sample(FRAME_COLORS, frame_count)
    random.shuffle(colors)

    frames: list[dict] = []
    seen_patterns: set[tuple[tuple[int, int], bytes]] = set()
    for idx, color in enumerate(colors):
        interior_h = random.choice(INTERIOR_SIZE_CHOICES)
        interior_w = random.choice(INTERIOR_SIZE_CHOICES)
        if idx == 0:
            force_top_left = BLACK
        elif idx == 1:
            force_top_left = random.choice(PATTERN_COLORS)
        else:
            force_top_left = random.choice((None, BLACK, RED, GREY))
        for _ in range(50):
            pattern = _make_pattern(interior_h, interior_w, force_top_left)
            key = _pattern_key(pattern)
            if key not in seen_patterns:
                seen_patterns.add(key)
                frames.append({"color": color, "pattern": pattern})
                break
        else:
            return None

    if not _choose_frame_positions(frames, grid_shape):
        return None

    blank_colors = [f["color"] for f in frames if int(f["pattern"][0, 0]) == BLACK]
    filled_colors = [f["color"] for f in frames if int(f["pattern"][0, 0]) != BLACK]
    if not blank_colors or not filled_colors:
        return None
    blank_color = random.choice(blank_colors)
    filled_color = random.choice(filled_colors)

    overlap_options: list[tuple[int, int, int, tuple[int, int]]] = []
    for frame_a in frames:
        for frame_b in frames:
            if frame_a["color"] == frame_b["color"]:
                continue
            delta = _choose_overlap_delta(frame_a["pattern"], frame_b["pattern"])
            if delta is not None:
                overlap_options.append((frame_a["color"], frame_b["color"], delta[0], delta[1]))
    if not overlap_options:
        return None
    overlap_a_color, overlap_b_color, dr, dc = random.choice(overlap_options)

    frame_by_color = {frame["color"]: frame for frame in frames}
    frame_bboxes = [frame["bbox"] for frame in frames]
    used_cells: set[tuple[int, int]] = set()
    markers: list[tuple[int, int, int]] = []

    pattern_a = frame_by_color[overlap_a_color]["pattern"]
    pattern_b = frame_by_color[overlap_b_color]["pattern"]
    bases = [
        (r, c)
        for r in range(grid_shape[0])
        for c in range(grid_shape[1])
    ]
    random.shuffle(bases)
    overlap_pair = None
    for r, c in bases:
        first = (r, c)
        second = (r + dr, c + dc)
        if not _valid_marker_cell(first, pattern_a, grid_shape, frame_bboxes, used_cells):
            continue
        if not _valid_marker_cell(second, pattern_b, grid_shape, frame_bboxes, used_cells | {first}):
            continue
        overlap_pair = (first, second)
        break
    if overlap_pair is None:
        return None

    markers.append((overlap_pair[0][0], overlap_pair[0][1], overlap_a_color))
    markers.append((overlap_pair[1][0], overlap_pair[1][1], overlap_b_color))
    used_cells.update(overlap_pair)

    clip_pattern = frame_by_color[filled_color]["pattern"]
    clip_candidates = [
        (r, c)
        for r in range(grid_shape[0])
        for c in range(grid_shape[1])
        if _valid_marker_cell((r, c), clip_pattern, grid_shape, frame_bboxes, used_cells)
        and (r + clip_pattern.shape[0] > grid_shape[0] or c + clip_pattern.shape[1] > grid_shape[1])
    ]
    random.shuffle(clip_candidates)
    if not clip_candidates:
        return None
    clip_cell = clip_candidates[0]
    markers.append((clip_cell[0], clip_cell[1], filled_color))
    used_cells.add(clip_cell)

    current_painted = set()
    for r, c, color in markers:
        current_painted |= _stamp_painted_cells(frame_by_color[color]["pattern"], r, c, grid_shape)

    blank_pattern = frame_by_color[blank_color]["pattern"]
    blank_candidates = [
        (r, c)
        for r in range(grid_shape[0])
        for c in range(grid_shape[1])
        if _valid_marker_cell((r, c), blank_pattern, grid_shape, frame_bboxes, used_cells)
        and (r, c) not in current_painted
    ]
    random.shuffle(blank_candidates)
    if not blank_candidates:
        return None
    blank_cell = blank_candidates[0]
    markers.append((blank_cell[0], blank_cell[1], blank_color))
    used_cells.add(blank_cell)

    color_counts = Counter(color for _r, _c, color in markers)
    for color in colors:
        if color in color_counts:
            continue
        pattern = frame_by_color[color]["pattern"]
        candidates = [
            (r, c)
            for r in range(grid_shape[0])
            for c in range(grid_shape[1])
            if _valid_marker_cell((r, c), pattern, grid_shape, frame_bboxes, used_cells)
            and blank_cell not in _stamp_painted_cells(pattern, r, c, grid_shape)
        ]
        random.shuffle(candidates)
        if not candidates:
            return None
        cell = candidates[0]
        markers.append((cell[0], cell[1], color))
        used_cells.add(cell)
        color_counts[color] += 1

    extra_target = random.randint(0, len(colors))
    for _ in range(extra_target):
        choices = [color for color in colors if color_counts[color] < 3]
        if not choices:
            break
        color = random.choice(choices)
        pattern = frame_by_color[color]["pattern"]
        candidates = [
            (r, c)
            for r in range(grid_shape[0])
            for c in range(grid_shape[1])
            if _valid_marker_cell((r, c), pattern, grid_shape, frame_bboxes, used_cells)
            and blank_cell not in _stamp_painted_cells(pattern, r, c, grid_shape)
        ]
        random.shuffle(candidates)
        if not candidates:
            continue
        cell = candidates[0]
        markers.append((cell[0], cell[1], color))
        used_cells.add(cell)
        color_counts[color] += 1

    grid = np.zeros(grid_shape, dtype=int)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        color = frame["color"]
        grid[r0, c0 : c1 + 1] = color
        grid[r1, c0 : c1 + 1] = color
        grid[r0 : r1 + 1, c0] = color
        grid[r0 : r1 + 1, c1] = color
        grid[r0 + 1 : r1, c0 + 1 : c1] = frame["pattern"]

    for r, c, color in markers:
        if int(grid[r, c]) != BLACK:
            return None
        grid[r, c] = color

    markers.sort()
    if not _semantic_check(grid, frames, markers):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
