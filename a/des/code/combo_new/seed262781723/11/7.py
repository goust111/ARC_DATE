import random

import numpy as np


BACKGROUND = 0
MARKER_COLOR = 4
SIDES = ("top", "bottom", "left", "right")
FRAME_COLORS = [1, 2, 3, 5, 6, 7, 8, 9]


def _border_values(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> np.ndarray:
    top = grid[r0, c0 : c1 + 1]
    bottom = grid[r1, c0 : c1 + 1]
    left = grid[r0 + 1 : r1, c0]
    right = grid[r0 + 1 : r1, c1]
    return np.concatenate((top, bottom, left, right))


def _marker_side(marker: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r, c = marker
    r0, c0, r1, c1 = bbox
    if r == r0:
        return "top"
    if r == r1:
        return "bottom"
    if c == c0:
        return "left"
    if c == c1:
        return "right"
    raise ValueError("marker is not on the frame border")


def _flip_for_side(patch: np.ndarray, side: str) -> np.ndarray:
    if side in ("top", "bottom"):
        return np.flipud(patch)
    return np.fliplr(patch)


def _validate_frame(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    marker: tuple[int, int],
) -> dict | None:
    r0, c0, r1, c1 = bbox
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    if not (r0 <= marker[0] <= r1 and c0 <= marker[1] <= c1):
        return None

    border = _border_values(grid, r0, c0, r1, c1)
    if np.any(border == BACKGROUND):
        return None
    if int(np.count_nonzero(border == MARKER_COLOR)) != 1:
        return None
    if int(grid[marker]) != MARKER_COLOR:
        return None

    frame_vals = border[border != MARKER_COLOR]
    frame_colors = np.unique(frame_vals)
    if frame_colors.size != 1:
        return None
    frame_color = int(frame_colors[0])
    if frame_color in (BACKGROUND, MARKER_COLOR):
        return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if interior.size == 0 or not np.any(interior != BACKGROUND):
        return None
    if np.any(interior == frame_color):
        return None

    try:
        side = _marker_side(marker, bbox)
    except ValueError:
        return None

    return {
        "bbox": bbox,
        "side": side,
        "frame_color": frame_color,
        "inner": interior.copy(),
        "area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
    }


def _scan_horizontal_extent(grid: np.ndarray, r: int, c: int, color: int) -> tuple[int, int]:
    c0 = c - 1
    while c0 - 1 >= 0 and int(grid[r, c0 - 1]) == color:
        c0 -= 1
    c1 = c + 1
    while c1 + 1 < grid.shape[1] and int(grid[r, c1 + 1]) == color:
        c1 += 1
    return c0, c1


def _scan_vertical_extent(grid: np.ndarray, r: int, c: int, color: int) -> tuple[int, int]:
    r0 = r - 1
    while r0 - 1 >= 0 and int(grid[r0 - 1, c]) == color:
        r0 -= 1
    r1 = r + 1
    while r1 + 1 < grid.shape[0] and int(grid[r1 + 1, c]) == color:
        r1 += 1
    return r0, r1


def _candidate_rectangles_from_marker(grid: np.ndarray, marker: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    r, c = marker
    h, w = grid.shape
    candidates: list[tuple[int, int, int, int]] = []

    if 0 < c < w - 1:
        left = int(grid[r, c - 1])
        right = int(grid[r, c + 1])
        if left == right and left not in (BACKGROUND, MARKER_COLOR):
            frame_color = left
            c0, c1 = _scan_horizontal_extent(grid, r, c, frame_color)
            for rr in range(r + 2, h):
                if not np.all(grid[rr, c0 : c1 + 1] == frame_color):
                    continue
                if np.all(grid[r + 1 : rr, c0] == frame_color) and np.all(grid[r + 1 : rr, c1] == frame_color):
                    candidates.append((r, c0, rr, c1))
                    break
            for rr in range(r - 2, -1, -1):
                if not np.all(grid[rr, c0 : c1 + 1] == frame_color):
                    continue
                if np.all(grid[rr + 1 : r, c0] == frame_color) and np.all(grid[rr + 1 : r, c1] == frame_color):
                    candidates.append((rr, c0, r, c1))
                    break

    if 0 < r < h - 1:
        up = int(grid[r - 1, c])
        down = int(grid[r + 1, c])
        if up == down and up not in (BACKGROUND, MARKER_COLOR):
            frame_color = up
            r0, r1 = _scan_vertical_extent(grid, r, c, frame_color)
            for cc in range(c + 2, w):
                if not np.all(grid[r0 : r1 + 1, cc] == frame_color):
                    continue
                if np.all(grid[r0, c + 1 : cc] == frame_color) and np.all(grid[r1, c + 1 : cc] == frame_color):
                    candidates.append((r0, c, r1, cc))
                    break
            for cc in range(c - 2, -1, -1):
                if not np.all(grid[r0 : r1 + 1, cc] == frame_color):
                    continue
                if np.all(grid[r0, cc + 1 : c] == frame_color) and np.all(grid[r1, cc + 1 : c] == frame_color):
                    candidates.append((r0, cc, r1, c))
                    break

    return candidates


def _find_frames(grid: np.ndarray) -> list[dict]:
    markers = [tuple(map(int, rc)) for rc in np.argwhere(grid == MARKER_COLOR)]
    frames_by_bbox: dict[tuple[int, int, int, int], dict] = {}

    for marker in markers:
        for bbox in _candidate_rectangles_from_marker(grid, marker):
            info = _validate_frame(grid, bbox, marker)
            if info is not None:
                frames_by_bbox[info["bbox"]] = info

    frames = list(frames_by_bbox.values())
    assert frames, "expected at least one framed pattern"
    return frames


def _stack_frames(frames: list[dict]) -> np.ndarray:
    ordered = sorted(frames, key=lambda item: int(item["area"]), reverse=True)
    out_h, out_w = ordered[0]["inner"].shape
    out = np.zeros((int(out_h), int(out_w)), dtype=int)

    for item in ordered:
        patch = _flip_for_side(item["inner"], item["side"])
        ph, pw = patch.shape
        top = (out_h - ph) // 2
        left = (out_w - pw) // 2
        mask = patch != BACKGROUND
        out[top : top + ph, left : left + pw][mask] = patch[mask]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flip each framed inner pattern according to its yellow edge marker, then center-stack the results by frame size.

    Input structure:
    - The background is black.
    - Several non-overlapping hollow rectangular frames appear in different colors, each with a single yellow border pixel.
    - Each frame encloses a smaller multicolor pattern inside its interior area.

    Rule:
    - Detect every frame and identify which border side contains the yellow marker.
    - Flip the frame's interior pattern across the normal axis of that side: top/bottom markers mean a vertical flip,
      left/right markers mean a horizontal flip.
    - Create an output grid whose size matches the interior of the largest frame, then place all flipped patterns
      centered on that canvas in descending frame-area order so later smaller patterns overwrite earlier ones.
    - Do not draw the frames themselves in the output.

    Key insight:
    - The tiny yellow anomaly is the instruction that selects the flip axis, while the final scene is a centered layered
      composite ordered by frame area rather than an in-place transformation.
    """
    grid = np.asarray(grid, dtype=int)
    assert grid.ndim == 2, "expected a 2D grid"
    frames = _find_frames(grid)
    return _stack_frames(frames)


SHAPE_LIBRARY = [
    np.array([[1]], dtype=int),
    np.array([[1, 1]], dtype=int),
    np.array([[1], [1]], dtype=int),
    np.array([[1, 0], [1, 1]], dtype=int),
    np.array([[1, 1, 1]], dtype=int),
    np.array([[1], [1], [1]], dtype=int),
    np.array([[1, 1], [1, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=int),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
]

GRID_RANGE = (24, 30)
FRAME_COUNT_RANGE = (2, 4)
LARGEST_INNER_RANGE = (7, 10)
SMALLER_INNER_MIN = 3
PATTERN_COLOR_COUNT_RANGE = (1, 3)
EXTRA_STAMP_RANGE = (1, 4)
MAX_GENERATION_ATTEMPTS = 300


def _transform_shape(shape: np.ndarray) -> np.ndarray:
    out = np.array(shape, copy=True)
    if random.random() < 0.5:
        out = np.fliplr(out)
    return np.rot90(out, random.randrange(4))


def _stamp_positions(h: int, w: int, sh: int, sw: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(h - sh + 1) for c in range(w - sw + 1)]


def _place_shape(patch: np.ndarray, shape: np.ndarray, color: int, prefer_center: bool) -> bool:
    h, w = patch.shape
    sh, sw = shape.shape
    if sh > h or sw > w:
        return False

    positions = _stamp_positions(h, w, sh, sw)
    if prefer_center:
        center_r = h / 2.0
        center_c = w / 2.0
        positions.sort(
            key=lambda pos: (
                abs((pos[0] + sh / 2.0) - center_r) + abs((pos[1] + sw / 2.0) - center_c),
                random.random(),
            )
        )
    else:
        random.shuffle(positions)

    for r0, c0 in positions:
        cells = shape.astype(bool)
        view = patch[r0 : r0 + sh, c0 : c0 + sw]
        overlap = int(np.count_nonzero((view != BACKGROUND) & cells))
        if overlap > 1:
            continue
        view[cells] = color
        return True
    return False


def _ensure_asymmetry(patch: np.ndarray, side: str, colors: list[int]) -> None:
    h, w = patch.shape
    if side in ("top", "bottom"):
        choices = [(r, c) for r in range(h) for c in range(w) if r != h - 1 - r]
        mirror = lambda r, c: (h - 1 - r, c)
    else:
        choices = [(r, c) for r in range(h) for c in range(w) if c != w - 1 - c]
        mirror = lambda r, c: (r, w - 1 - c)

    random.shuffle(choices)
    for r, c in choices:
        mr, mc = mirror(r, c)
        if patch[r, c] == BACKGROUND and patch[mr, mc] == BACKGROUND:
            patch[r, c] = random.choice(colors)
            return
    for r, c in choices:
        mr, mc = mirror(r, c)
        if patch[r, c] == patch[mr, mc]:
            alternatives = [color for color in colors if color != int(patch[mr, mc])]
            patch[r, c] = random.choice(alternatives) if alternatives else BACKGROUND
            return


def _make_pattern(h: int, w: int, side: str, colors: list[int]) -> np.ndarray:
    for _ in range(120):
        patch = np.zeros((h, w), dtype=int)
        center_r = random.randrange(max(1, h // 2) - 1, min(h, h // 2 + 2))
        center_c = random.randrange(max(1, w // 2) - 1, min(w, w // 2 + 2))
        patch[center_r, center_c] = colors[0]

        n_stamps = random.randint(*EXTRA_STAMP_RANGE)
        for stamp_i in range(n_stamps):
            shape = _transform_shape(random.choice(SHAPE_LIBRARY))
            color = random.choice(colors)
            _place_shape(patch, shape, color, prefer_center=(stamp_i == 0))

        _ensure_asymmetry(patch, side, colors)

        if np.count_nonzero(patch) < 3:
            continue
        if np.array_equal(patch, _flip_for_side(patch, side)):
            continue
        return patch
    raise RuntimeError("failed to create a non-symmetric pattern")


def _choose_inner_sizes(n_frames: int) -> list[tuple[int, int]]:
    largest_h = random.randint(*LARGEST_INNER_RANGE)
    largest_w = random.randint(*LARGEST_INNER_RANGE)
    parity_h = largest_h % 2
    parity_w = largest_w % 2

    candidates = [
        (h, w)
        for h in range(SMALLER_INNER_MIN, largest_h + 1)
        for w in range(SMALLER_INNER_MIN, largest_w + 1)
        if h % 2 == parity_h and w % 2 == parity_w
    ]
    area_seen: set[int] = set()
    by_area: dict[int, list[tuple[int, int]]] = {}
    for h, w in candidates:
        area = (h + 2) * (w + 2)
        by_area.setdefault(area, []).append((h, w))
    areas = sorted(by_area.keys(), reverse=True)
    sizes: list[tuple[int, int]] = []
    for area in areas:
        choices = [pair for pair in by_area[area] if pair not in sizes]
        if not choices:
            continue
        pair = random.choice(choices)
        sizes.append(pair)
        area_seen.add(area)
        if len(sizes) == n_frames:
            break

    if len(sizes) < n_frames:
        raise RuntimeError("not enough unique frame areas")
    sizes.sort(key=lambda pair: (pair[0] + 2) * (pair[1] + 2), reverse=True)
    return sizes


def _choose_sides(n_frames: int) -> list[str]:
    base = list(SIDES)
    random.shuffle(base)
    sides = base[: min(n_frames, len(base))]
    while len(sides) < n_frames:
        sides.append(random.choice(SIDES))
    random.shuffle(sides)
    return sides


def _marker_location(side: str, h: int, w: int) -> tuple[int, int]:
    if side == "top":
        return 0, random.randint(1, w - 2)
    if side == "bottom":
        return h - 1, random.randint(1, w - 2)
    if side == "left":
        return random.randint(1, h - 2), 0
    return random.randint(1, h - 2), w - 1


def _free_positions(occupied: np.ndarray, h: int, w: int, padding: int = 1) -> list[tuple[int, int]]:
    rows, cols = occupied.shape
    out: list[tuple[int, int]] = []
    for r in range(rows - h + 1):
        for c in range(cols - w + 1):
            rr0 = max(0, r - padding)
            cc0 = max(0, c - padding)
            rr1 = min(rows, r + h + padding)
            cc1 = min(cols, c + w + padding)
            if not occupied[rr0:rr1, cc0:cc1].any():
                out.append((r, c))
    return out


def _render_input(frames: list[dict], shape: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for item in frames:
        r0 = item["top"]
        c0 = item["left"]
        h = item["outer_h"]
        w = item["outer_w"]
        frame_color = item["frame_color"]
        marker_r, marker_c = item["marker"]
        grid[r0, c0 : c0 + w] = frame_color
        grid[r0 + h - 1, c0 : c0 + w] = frame_color
        grid[r0 : r0 + h, c0] = frame_color
        grid[r0 : r0 + h, c0 + w - 1] = frame_color
        grid[r0 + marker_r, c0 + marker_c] = MARKER_COLOR
        inner = item["inner"]
        ih, iw = inner.shape
        mask = inner != BACKGROUND
        grid[r0 + 1 : r0 + 1 + ih, c0 + 1 : c0 + 1 + iw][mask] = inner[mask]
    return grid


def _expected_output(frames: list[dict]) -> np.ndarray:
    solver_frames = []
    for item in frames:
        solver_frames.append(
            {
                "inner": item["inner"],
                "side": item["side"],
                "area": int(item["outer_h"] * item["outer_w"]),
            }
        )
    return _stack_frames(solver_frames)


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*GRID_RANGE)
    cols = random.randint(*GRID_RANGE)
    n_frames = random.randint(*FRAME_COUNT_RANGE)

    try:
        inner_sizes = _choose_inner_sizes(n_frames)
    except RuntimeError:
        return None

    sides = _choose_sides(n_frames)
    frame_colors = random.sample(FRAME_COLORS, n_frames)
    pattern_palette = [color for color in FRAME_COLORS if color not in frame_colors]
    if len(pattern_palette) < 3:
        return None

    frames: list[dict] = []
    occupied = np.zeros((rows, cols), dtype=bool)

    for (inner_h, inner_w), side, frame_color in zip(inner_sizes, sides, frame_colors):
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        positions = _free_positions(occupied, outer_h, outer_w, padding=1)
        if not positions:
            return None
        top, left = random.choice(positions)
        occupied[top : top + outer_h, left : left + outer_w] = True

        n_colors = random.randint(*PATTERN_COLOR_COUNT_RANGE)
        colors = random.sample(pattern_palette, n_colors)
        inner = _make_pattern(inner_h, inner_w, side, colors)
        marker = _marker_location(side, outer_h, outer_w)
        frames.append(
            {
                "top": top,
                "left": left,
                "outer_h": outer_h,
                "outer_w": outer_w,
                "frame_color": frame_color,
                "side": side,
                "marker": marker,
                "inner": inner,
            }
        )

    grid = _render_input(frames, (rows, cols))
    expected = _expected_output(frames)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
