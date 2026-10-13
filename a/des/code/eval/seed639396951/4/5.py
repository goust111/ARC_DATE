import numpy as np


BG = 0
RED = 2
GREY = 5
FRAME_PALETTE = np.array([1, 3, 4, 6, 7, 8, 9], dtype=int)
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 200


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _frame_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    _paint_frame(mask, bbox, True)
    return mask


def _indicator_candidates(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    candidates = []
    for r, c in ((1, 1), (1, w - 2), (h - 2, 1), (h - 2, w - 2)):
        if 0 < r < h - 1 and 0 < c < w - 1:
            candidates.append((r, c))
    return candidates


def _detect_indicator(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], set[tuple[int, int]]]:
    candidates: list[tuple[tuple[int, int], tuple[int, int], set[tuple[int, int]]]] = []
    for r, c in _indicator_candidates(grid.shape):
        if int(grid[r, c]) != GREY:
            continue
        red_dirs = []
        ok = True
        for dr, dc in DIRS:
            value = int(grid[r + dr, c + dc])
            if value == RED:
                red_dirs.append((dr, dc))
            elif value != BG:
                ok = False
                break
        if not ok or len(red_dirs) != 1:
            continue
        for dr in (-1, 1):
            for dc in (-1, 1):
                if int(grid[r + dr, c + dc]) != BG:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        patch = {(rr, cc) for rr in range(r - 1, r + 2) for cc in range(c - 1, c + 2)}
        candidates.append(((r, c), red_dirs[0], patch))
    assert len(candidates) == 1, "expected exactly one corner direction indicator"
    return candidates[0]


def _extract_frames(grid: np.ndarray, indicator_patch: set[tuple[int, int]]) -> list[dict[str, object]]:
    work = np.array(grid, copy=True)
    for r, c in indicator_patch:
        work[r, c] = BG

    colors = [int(color) for color in np.unique(work) if int(color) != BG]
    assert 2 <= len(colors) <= 4, "expected 2 to 4 frame colors"

    frames = []
    for color in colors:
        mask = work == color
        bbox = _bbox_from_mask(mask)
        assert bbox is not None
        r0, c0, r1, c1 = bbox
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must enclose an interior"
        expected = _frame_mask(work.shape, bbox)
        assert np.array_equal(mask, expected), "each frame color must form one 1px rectangular ring"
        frames.append({"color": color, "bbox": bbox})

    frames.sort(
        key=lambda frame: (
            (frame["bbox"][2] - frame["bbox"][0] + 1) * (frame["bbox"][3] - frame["bbox"][1] + 1),
            -frame["bbox"][0],
            -frame["bbox"][1],
        ),
        reverse=True,
    )

    for i in range(1, len(frames)):
        outer = frames[i - 1]["bbox"]
        inner = frames[i]["bbox"]
        assert inner[0] == outer[0] + 2
        assert inner[1] == outer[1] + 2
        assert inner[2] == outer[2] - 2
        assert inner[3] == outer[3] - 2

    return frames


def _translated_frames(
    frames: list[dict[str, object]], direction: tuple[int, int]
) -> list[tuple[int, tuple[int, int, int, int], int]]:
    dr, dc = direction
    translated = []
    for depth, frame in enumerate(frames, start=1):
        offset = (depth - 1) * 2
        r0, c0, r1, c1 = frame["bbox"]
        bbox = (r0 + dr * offset, c0 + dc * offset, r1 + dr * offset, c1 + dc * offset)
        translated.append((int(frame["color"]), bbox, depth))
    return translated


def main(grid: np.ndarray) -> np.ndarray:
    """
    Separate nested rectangular rings by translating each depth inward along the arrow direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape[0] >= 7 and g.shape[1] >= 7

    _center, direction, indicator_patch = _detect_indicator(g)
    frames = _extract_frames(g, indicator_patch)
    translated = _translated_frames(frames, direction)

    min_r = min(bbox[0] for _color, bbox, _depth in translated)
    min_c = min(bbox[1] for _color, bbox, _depth in translated)
    max_r = max(bbox[2] for _color, bbox, _depth in translated)
    max_c = max(bbox[3] for _color, bbox, _depth in translated)

    out = np.full((max_r - min_r + 3, max_c - min_c + 3), BG, dtype=int)
    for color, bbox, _depth in translated:
        r0, c0, r1, c1 = bbox
        shifted = (r0 - min_r + 1, c0 - min_c + 1, r1 - min_r + 1, c1 - min_c + 1)
        _paint_frame(out, shifted, color)
    return out


def _make_indicator(grid: np.ndarray, corner: int, direction: tuple[int, int]) -> None:
    h, w = grid.shape
    centers = [(1, 1), (1, w - 2), (h - 2, 1), (h - 2, w - 2)]
    r, c = centers[corner]
    grid[r, c] = GREY
    dr, dc = direction
    grid[r + dr, c + dc] = RED


def _balanced_margin(rng: np.random.Generator) -> tuple[int, int]:
    base = int(rng.integers(4, 9))
    delta = int(rng.integers(-1, 2))
    first = base
    second = max(4, base + delta)
    return first, second


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    depth_count = int(rng.integers(2, 5))
    inner_h = int(rng.integers(3, 9))
    inner_w = int(rng.integers(3, 11))
    outer_h = inner_h + 4 * (depth_count - 1)
    outer_w = inner_w + 4 * (depth_count - 1)

    top_margin, bottom_margin = _balanced_margin(rng)
    left_margin, right_margin = _balanced_margin(rng)

    h = outer_h + top_margin + bottom_margin
    w = outer_w + left_margin + right_margin
    grid = np.full((h, w), BG, dtype=int)

    top = top_margin
    left = left_margin
    colors = rng.choice(FRAME_PALETTE, size=depth_count, replace=False)
    for depth, color in enumerate(colors):
        inset = depth * 2
        bbox = (top + inset, left + inset, top + outer_h - 1 - inset, left + outer_w - 1 - inset)
        _paint_frame(grid, bbox, int(color))

    corner = int(rng.integers(0, 4))
    direction = DIRS[int(rng.integers(0, 4))]
    _make_indicator(grid, corner, direction)

    try:
        _center, found_direction, indicator_patch = _detect_indicator(grid)
        assert found_direction == direction
        frames = _extract_frames(grid, indicator_patch)
        assert len(frames) == depth_count
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated attempts")
