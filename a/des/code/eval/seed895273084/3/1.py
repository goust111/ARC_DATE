import numpy as np


COLOR_IDS = tuple(range(10))
GRID_H_RANGE = (14, 30)
GRID_W_RANGE = (14, 30)
OBJECT_COUNT_MAX = 5
OUTER_H_RANGE = (5, 12)
OUTER_W_RANGE = (5, 12)
FRAME_GAP = 1
PLACEMENT_TRIES = 80
GENERATION_TRIES = 300


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
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _mode_color(values: np.ndarray) -> int:
    flat = np.asarray(values, dtype=int).reshape(-1)
    colors, counts = np.unique(flat, return_counts=True)
    return int(colors[counts.argmax()])


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _is_rect_frame(points: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(points)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False

    expected = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
    if len(points) != expected:
        return False

    point_set = set(points)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_boundary = r == r0 or r == r1 or c == c0 or c == c1
            if on_boundary != ((r, c) in point_set):
                return False
    return True


def _find_frames(grid: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int, int, int]]]:
    best_color = -1
    best_score = -1
    best_boxes: list[tuple[int, int, int, int]] = []

    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        boxes = []
        score = 0
        for comp in _components(grid == color):
            if _is_rect_frame(comp):
                boxes.append(_bbox(comp))
                score += len(comp)
        if boxes and score > best_score:
            best_color = color
            best_score = score
            best_boxes = sorted(boxes)

    assert best_boxes, "expected at least one rectangular frame"
    return best_color, best_boxes


def _band_distance(r: int, c: int, bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return min(r - r0, r1 - r, c - c0, c1 - c)


def _paint_band(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int, thickness: int) -> None:
    r0, c0, r1, c1 = bbox
    if thickness <= 0 or r1 - r0 < 2 or c1 - c0 < 2:
        return

    rr = np.arange(r0 + 1, r1)[:, None]
    cc = np.arange(c0 + 1, c1)[None, :]
    dist = np.minimum(np.minimum(rr - r0, r1 - rr), np.minimum(cc - c0, c1 - cc))
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    inner[dist <= thickness] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each seed's distance to the nearest rectangle side as the thickness of
    a same-colored band grown inward from that rectangle's inner border.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    frame_color, boxes = _find_frames(g, bg)

    out = g.copy()
    for bbox in boxes:
        r0, c0, r1, c1 = bbox
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_locs = np.argwhere((inner != bg) & (inner != frame_color))
        assert len(seed_locs) == 1, "each frame must contain exactly one seed"

        sr = int(seed_locs[0, 0] + r0 + 1)
        sc = int(seed_locs[0, 1] + c0 + 1)
        seed_color = int(g[sr, sc])
        thickness = _band_distance(sr, sc, bbox)
        _paint_band(out, bbox, seed_color, thickness)

    return out


def _seed_candidates(bbox: tuple[int, int, int, int], thickness: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if _band_distance(r, c, bbox) == thickness:
                cells.append((r, c))
    return cells


def _stamp_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], frame_color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color


def _try_place_rectangles(
    rng: np.random.Generator,
    h: int,
    w: int,
    n_objects: int,
) -> list[tuple[int, int, int, int]] | None:
    placed: list[tuple[int, int, int, int]] = []
    blocked = np.zeros((h, w), dtype=bool)

    for _ in range(n_objects):
        success = False
        for _ in range(PLACEMENT_TRIES):
            outer_h = int(rng.integers(OUTER_H_RANGE[0], min(OUTER_H_RANGE[1], h - 1) + 1))
            outer_w = int(rng.integers(OUTER_W_RANGE[0], min(OUTER_W_RANGE[1], w - 1) + 1))
            if outer_h > h - 2 or outer_w > w - 2:
                continue

            r0 = int(rng.integers(1, h - outer_h))
            c0 = int(rng.integers(1, w - outer_w))
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1

            rr0 = max(0, r0 - FRAME_GAP)
            cc0 = max(0, c0 - FRAME_GAP)
            rr1 = min(h, r1 + FRAME_GAP + 1)
            cc1 = min(w, c1 + FRAME_GAP + 1)
            if blocked[rr0:rr1, cc0:cc1].any():
                continue

            placed.append((r0, c0, r1, c1))
            blocked[rr0:rr1, cc0:cc1] = True
            success = True
            break
        if not success:
            return None

    return placed


def _build_example(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray] | None:
    h = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    w = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))

    colors = list(COLOR_IDS)
    rng.shuffle(colors)
    bg = colors[0]
    frame_color = colors[1]

    max_objects = min(OBJECT_COUNT_MAX, len(colors) - 2, max(1, (h * w) // 70))
    n_objects = int(rng.integers(1, max_objects + 1))
    seed_colors = colors[2 : 2 + n_objects]

    boxes = _try_place_rectangles(rng, h, w, n_objects)
    if boxes is None:
        return None

    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)

    for bbox, seed_color in zip(boxes, seed_colors, strict=False):
        r0, c0, r1, c1 = bbox
        _stamp_frame(inp, bbox, frame_color)
        _stamp_frame(out, bbox, frame_color)

        max_thickness = min((r1 - r0) // 2, (c1 - c0) // 2)
        if max_thickness < 1:
            return None
        thickness = int(rng.integers(1, max_thickness + 1))
        candidates = _seed_candidates(bbox, thickness)
        if not candidates:
            return None
        sr, sc = candidates[int(rng.integers(len(candidates)))]

        inp[sr, sc] = seed_color
        _paint_band(out, bbox, seed_color, thickness)

    colors_present, counts = np.unique(inp, return_counts=True)
    order = np.argsort(-counts)
    if int(colors_present[order[0]]) != bg:
        return None

    non_bg = inp[inp != bg]
    if non_bg.size == 0 or _mode_color(non_bg) != frame_color:
        return None

    if not np.array_equal(main(inp), out):
        return None

    return inp, out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_TRIES):
        built = _build_example(rng)
        if built is not None:
            inp, _ = built
            return inp
    raise RuntimeError(f"generate_input failed after {GENERATION_TRIES} attempts")
