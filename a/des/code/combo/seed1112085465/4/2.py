import random

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_DIR = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _frame_border(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    border = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))
    return border


def _frame_info(grid: np.ndarray, bg: int) -> dict[int, dict[str, object]]:
    info: dict[int, dict[str, object]] = {}
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        comps = _components(grid == color)
        if not comps:
            continue
        comps.sort(key=lambda comp: (-len(comp), _bbox(comp)))
        frame = comps[0]
        assert len(frame) > 1, f"color {color} must contain a frame component"
        box = _bbox(frame)
        r0, c0, r1, c1 = box
        assert r1 - r0 >= 2 and c1 - c0 >= 2, f"color {color} frame too small"
        frame_set = set(frame)
        assert frame_set == _frame_border(box), f"color {color} component is not a thin rectangular frame"
        markers = [tuple(comp[0]) for comp in comps[1:] if len(comp) == 1]
        info[color] = {"frame": frame_set, "bbox": box, "markers": markers}
    return info


def _marker_target(
    grid: np.ndarray, bg: int, marker: tuple[int, int], frame: set[tuple[int, int]]
) -> tuple[int, int]:
    h, w = grid.shape
    r, c = marker
    candidates: list[tuple[int, int]] = []

    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        while 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == bg:
            nr += dr
            nc += dc
        if 0 <= nr < h and 0 <= nc < w and (nr, nc) in frame:
            candidates.append((nr - dr, nc - dc))

    assert len(candidates) == 1, f"marker {marker} must have exactly one visible same-color frame direction"
    return candidates[0]


def _paint_segment(
    grid: np.ndarray, start: tuple[int, int], end: tuple[int, int], color: int
) -> None:
    r0, c0 = start
    r1, c1 = end
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        grid[r0, lo : hi + 1] = color
        return
    assert c0 == c1
    lo, hi = sorted((r0, r1))
    grid[lo : hi + 1, c0] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pull each singleton marker toward the uniquely visible edge of its same-color frame.

    The frame stays fixed. The marker colors the straight path from its source cell to the
    background cell immediately outside the frame edge it can see.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    info = _frame_info(g, bg)

    out = g.copy()
    for color, payload in info.items():
        frame = payload["frame"]
        markers = payload["markers"]
        assert isinstance(frame, set)
        assert isinstance(markers, list)
        for marker in markers:
            target = _marker_target(g, bg, marker, frame)
            _paint_segment(out, marker, target, color)
    return out


BG = 7
PALETTE = [1, 2, 3, 4, 5, 6, 8, 9]
GRID_SIZE_RANGE = (10, 24)
FRAME_COUNT_WEIGHTS = [(1, 3), (2, 4), (3, 3)]
FRAME_SIZE_RANGE = (4, 8)
FRAME_SEPARATION = 1
MAX_MARKERS_PER_FRAME = 3
MAX_MARKER_GAP = 5
GEN_RETRIES = 400


def _line_cells(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = start
    r1, c1 = end
    cells: list[tuple[int, int]] = []
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        for c in range(c0, c1 + step, step):
            cells.append((r0, c))
        return cells
    assert c0 == c1
    step = 1 if r1 >= r0 else -1
    for r in range(r0, r1 + step, step):
        cells.append((r, c0))
    return cells


def _boxes_overlap(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _draw_frame(
    grid: np.ndarray, bbox: tuple[int, int, int, int], color: int
) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _sample_frame_count() -> int:
    expanded = []
    for count, weight in FRAME_COUNT_WEIGHTS:
        expanded.extend([count] * weight)
    return random.choice(expanded)


def _place_frames(
    h: int, w: int, frame_count: int
) -> list[tuple[int, int, int, int, int]] | None:
    frames: list[tuple[int, int, int, int, int]] = []
    colors = random.sample(PALETTE, frame_count)
    max_side = max(FRAME_SIZE_RANGE[0], FRAME_SIZE_RANGE[1] - max(0, frame_count - 2))

    for color in colors:
        placed = False
        for _ in range(120):
            fh = random.randint(FRAME_SIZE_RANGE[0], min(max_side, h - 2))
            fw = random.randint(FRAME_SIZE_RANGE[0], min(max_side, w - 2))
            if fh >= h or fw >= w:
                continue
            r0 = random.randint(1, h - fh - 1)
            c0 = random.randint(1, w - fw - 1)
            bbox = (r0, c0, r0 + fh - 1, c0 + fw - 1)
            if any(_boxes_overlap(bbox, other[:4], FRAME_SEPARATION) for other in frames):
                continue
            frames.append((*bbox, color))
            placed = True
            break
        if not placed:
            return None
    return frames


def _marker_counts(frame_count: int) -> list[int]:
    if frame_count == 1:
        total = random.randint(2, MAX_MARKERS_PER_FRAME)
    else:
        total = frame_count + random.randint(0, frame_count + 1)
    total = min(total, frame_count * MAX_MARKERS_PER_FRAME)
    counts = [1] * frame_count
    extras = total - frame_count
    while extras > 0:
        idx = random.randrange(frame_count)
        if counts[idx] < MAX_MARKERS_PER_FRAME:
            counts[idx] += 1
            extras -= 1
    random.shuffle(counts)
    return counts


def _candidate_corridors(
    struct: np.ndarray,
    reserved: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> dict[str, list[list[tuple[int, int]]]]:
    h, w = struct.shape
    r0, c0, r1, c1 = bbox
    candidates: dict[str, list[list[tuple[int, int]]]] = {side: [] for side in SIDE_TO_DIR}

    for col in range(c0, c1 + 1):
        max_gap = min(MAX_MARKER_GAP, r0 - 1)
        for gap in range(1, max_gap + 1):
            start = (r0 - 1 - gap, col)
            end = (r0 - 1, col)
            cells = _line_cells(start, end)
            if all(int(struct[r, c]) == BG and not reserved[r, c] for r, c in cells):
                candidates["top"].append(cells)

        max_gap = min(MAX_MARKER_GAP, h - r1 - 2)
        for gap in range(1, max_gap + 1):
            start = (r1 + 1 + gap, col)
            end = (r1 + 1, col)
            cells = _line_cells(start, end)
            if all(int(struct[r, c]) == BG and not reserved[r, c] for r, c in cells):
                candidates["bottom"].append(cells)

    for row in range(r0, r1 + 1):
        max_gap = min(MAX_MARKER_GAP, c0 - 1)
        for gap in range(1, max_gap + 1):
            start = (row, c0 - 1 - gap)
            end = (row, c0 - 1)
            cells = _line_cells(start, end)
            if all(int(struct[r, c]) == BG and not reserved[r, c] for r, c in cells):
                candidates["left"].append(cells)

        max_gap = min(MAX_MARKER_GAP, w - c1 - 2)
        for gap in range(1, max_gap + 1):
            start = (row, c1 + 1 + gap)
            end = (row, c1 + 1)
            cells = _line_cells(start, end)
            if all(int(struct[r, c]) == BG and not reserved[r, c] for r, c in cells):
                candidates["right"].append(cells)

    return candidates


def _choose_side(
    candidates: dict[str, list[list[tuple[int, int]]]], side_counts: dict[str, int]
) -> str | None:
    feasible = [side for side, items in candidates.items() if items]
    if not feasible:
        return None
    best = min(side_counts[side] for side in feasible)
    choices = [side for side in feasible if side_counts[side] == best]
    return random.choice(choices)


def _build_expected(
    struct: np.ndarray, placements: list[tuple[int, list[tuple[int, int]]]]
) -> np.ndarray:
    out = struct.copy()
    for color, cells in placements:
        for r, c in cells:
            out[r, c] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    frame_count = _sample_frame_count()
    min_side = {1: 10, 2: 14, 3: 18}[frame_count]
    h = random.randint(min_side, GRID_SIZE_RANGE[1])
    w = random.randint(min_side, GRID_SIZE_RANGE[1])

    frames = _place_frames(h, w, frame_count)
    if frames is None:
        return None

    struct = np.full((h, w), BG, dtype=int)
    for r0, c0, r1, c1, color in frames:
        _draw_frame(struct, (r0, c0, r1, c1), color)

    target_counts = _marker_counts(frame_count)
    reserved = np.zeros((h, w), dtype=bool)
    placements: list[tuple[int, list[tuple[int, int]]]] = []
    side_counts = {side: 0 for side in SIDE_TO_DIR}

    frame_order = list(range(frame_count))
    random.shuffle(frame_order)
    for frame_idx in frame_order:
        r0, c0, r1, c1, color = frames[frame_idx]
        count = target_counts[frame_idx]
        for _ in range(count):
            candidates = _candidate_corridors(struct, reserved, (r0, c0, r1, c1))
            side = _choose_side(candidates, side_counts)
            if side is None:
                return None
            cells = random.choice(candidates[side])
            for r, c in cells:
                reserved[r, c] = True
            placements.append((color, cells))
            side_counts[side] += 1

    inp = struct.copy()
    for color, cells in placements:
        sr, sc = cells[0]
        inp[sr, sc] = color

    expected = _build_expected(struct, placements)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for task 2")
