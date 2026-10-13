import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

MARKER_COLORS = (RED, BLUE, GREEN, YELLOW)
SEED_COLORS = (6, 7, 8, 9)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        q = deque([(sr, sc)])
        pts: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            pts.append((r, c))
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _frame_bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(points)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    if set(points) != expected:
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _find_frames(g: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for pts in _components(g == GREY):
        bbox = _frame_bbox(pts)
        assert bbox is not None, "grey pixels must form hollow rectangular frames"
        frames.append(bbox)
    assert 2 <= len(frames) <= 4, "expected two to four nested grey frames"

    frames.sort(key=lambda b: (b[2] - b[0] + 1) * (b[3] - b[1] + 1))
    for inner, outer in zip(frames, frames[1:]):
        ir0, ic0, ir1, ic1 = inner
        or0, oc0, or1, oc1 = outer
        assert or0 < ir0 and oc0 < ic0 and ir1 < or1 and ic1 < oc1, "frames must be nested"
    return frames


def _apply_transform(arr: np.ndarray, color: int) -> np.ndarray:
    if color == RED:
        return np.fliplr(arr)
    if color == BLUE:
        return np.flipud(arr)
    if color == GREEN:
        return np.rot90(arr, -1)
    if color == YELLOW:
        return np.rot90(arr, 2)
    raise AssertionError("marker color must be red, blue, green, or yellow")


def _crop_nonblack(arr: np.ndarray) -> np.ndarray:
    pts = np.argwhere(arr != BLACK)
    assert len(pts) > 0, "pattern must contain at least one colored pixel"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _extract_seed(g: np.ndarray, inner_frame: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = inner_frame
    interior = g[r0 + 1 : r1, c0 + 1 : c1].copy()
    assert interior.size > 0, "innermost frame must have an interior"
    interior[0, 0] = BLACK
    mask = (interior != BLACK) & (interior != GREY)
    pts = np.argwhere(mask)
    assert len(pts) > 0, "expected a colored seed pattern inside the innermost frame"
    sr0, sc0 = pts.min(axis=0)
    sr1, sc1 = pts.max(axis=0)
    return interior[int(sr0) : int(sr1) + 1, int(sc0) : int(sc1) + 1].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode nested frame corner markers and apply their transforms to the central seed pattern.

    Input structure:
    - The background is black.
    - The center contains a small multicolor, asymmetric seed pattern.
    - Around it are two to four nested grey hollow rectangular frames separated by black gaps.
    - The inside top-left corner cell of each frame is a single colored marker: red, blue, green, or yellow.

    Rule:
    - Identify the grey rectangular frames and sort them from innermost to outermost.
    - Crop the non-black, non-grey seed pattern from inside the innermost frame, ignoring that frame's corner marker.
    - Read each frame's top-left interior marker from inside to outside.
    - Apply the marked transform cumulatively: red = horizontal flip, blue = vertical flip, green = clockwise 90 degree rotation, yellow = 180 degree rotation.
    - Return only the final transformed seed pattern cropped to its tight non-black bounding box.

    Key insight: The grey frames and corner pixels are an ordered instruction stack, not output content; transform order is read from the seed outward.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = _find_frames(g)
    markers: list[int] = []
    for r0, c0, _r1, _c1 in frames:
        marker = int(g[r0 + 1, c0 + 1])
        assert marker in MARKER_COLORS, "each frame must have a valid top-left interior marker"
        markers.append(marker)

    out = _extract_seed(g, frames[0])
    for marker in markers:
        out = _apply_transform(out, marker)
    return _crop_nonblack(out)


def _is_asymmetric(seed: np.ndarray) -> bool:
    tests = [np.fliplr(seed), np.flipud(seed), np.rot90(seed, 2)]
    if seed.shape[0] == seed.shape[1]:
        tests.extend([np.rot90(seed, 1), np.rot90(seed, -1)])
    return all(not np.array_equal(seed, other) for other in tests)


def _random_seed(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(3, 5)
    w = rng.randint(3, 5)
    colors = list(SEED_COLORS)
    rng.shuffle(colors)
    color_count = rng.randint(2, min(4, len(colors)))
    palette = colors[:color_count]

    for _ in range(80):
        flat = [rng.choice(palette) for _ in range(h * w)]
        for i, color in enumerate(palette):
            flat[i] = color
        rng.shuffle(flat)
        seed = np.array(flat, dtype=int).reshape(h, w)
        if _is_asymmetric(seed):
            return seed
    return None


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY


def _shift_bbox(bbox: tuple[int, int, int, int], dr: int, dc: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return r0 + dr, c0 + dc, r1 + dr, c1 + dc


def _marker_sequence(layer_count: int, rng: random.Random) -> list[int]:
    colors = list(MARKER_COLORS)
    rng.shuffle(colors)
    return colors[:layer_count]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    seed = _random_seed(rng)
    if seed is None:
        return None

    layer_count = rng.randint(2, 4)
    inner_gap_r = rng.randint(1, 2)
    inner_gap_c = rng.randint(1, 2)

    seed_h, seed_w = seed.shape
    inner_h = seed_h + 2 * inner_gap_r + 2
    inner_w = seed_w + 2 * inner_gap_c + 2
    frames = [(0, 0, inner_h - 1, inner_w - 1)]

    for _ in range(1, layer_count):
        grow_r = rng.randint(2, 3)
        grow_c = rng.randint(2, 3)
        r0, c0, r1, c1 = frames[-1]
        frames.append((r0 - grow_r, c0 - grow_c, r1 + grow_r, c1 + grow_c))

    min_r = min(b[0] for b in frames)
    min_c = min(b[1] for b in frames)
    frames = [_shift_bbox(b, -min_r, -min_c) for b in frames]

    outer = frames[-1]
    struct_h = outer[2] + 1
    struct_w = outer[3] + 1
    if struct_h > 30 or struct_w > 30:
        return None

    extra_h = rng.randint(0, 30 - struct_h)
    extra_w = rng.randint(0, 30 - struct_w)
    top = max(0, min(extra_h, extra_h // 2 + rng.choice([-1, 0, 1])))
    left = max(0, min(extra_w, extra_w // 2 + rng.choice([-1, 0, 1])))
    h = struct_h + extra_h
    w = struct_w + extra_w

    frames = [_shift_bbox(b, top, left) for b in frames]
    grid = np.full((h, w), BLACK, dtype=int)

    for bbox in reversed(frames):
        _draw_frame(grid, bbox)

    markers = _marker_sequence(layer_count, rng)
    for bbox, marker in zip(frames, markers):
        r0, c0, _r1, _c1 = bbox
        grid[r0 + 1, c0 + 1] = int(marker)

    ir0, ic0, _ir1, _ic1 = frames[0]
    sr0 = ir0 + 1 + inner_gap_r
    sc0 = ic0 + 1 + inner_gap_c
    grid[sr0 : sr0 + seed_h, sc0 : sc0 + seed_w] = seed

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.ndim != 2 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt(random)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame transform input")
