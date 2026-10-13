from __future__ import annotations

from collections import Counter, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8

SIDE_TO_FILL = {
    4: BLUE,
    5: TEAL,
    6: YELLOW,
    7: ORANGE,
    8: PINK,
}

GEN_MIN_OBJECTS = 5
GEN_MAX_OBJECTS = 8
GEN_MIN_FRAMES = 3
GEN_MAX_FRAMES = 4
GEN_MIN_SIDE = 4
GEN_MAX_SIDE = 8
GEN_MIN_GAP = 1
GEN_GRID_ROWS = (20, 27)
GEN_GRID_COLS = (20, 27)
GEN_MAX_ATTEMPTS = 400

_RNG = np.random.default_rng()


def _background_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _marker_offset(side: int) -> int:
    return (int(side) - 1) // 2


def _frame_patch(side: int) -> np.ndarray:
    patch = np.full((side, side), BLACK, dtype=int)
    patch[0, :] = GREY
    patch[-1, :] = GREY
    patch[:, 0] = GREY
    patch[:, -1] = GREY
    k = _marker_offset(side)
    patch[k, k] = RED
    return patch


def _filled_frame_patch(side: int) -> np.ndarray:
    patch = _frame_patch(side)
    patch[patch == BLACK] = SIDE_TO_FILL[int(side)]
    return patch


def _classify_grey_component(grid: np.ndarray, coords: np.ndarray) -> dict:
    bbox = _bbox_from_coords(coords)
    r0, c0, r1, c1 = bbox
    side_h = r1 - r0 + 1
    side_w = c1 - c0 + 1
    assert side_h == side_w, "every grey object must be square"
    side = int(side_h)
    assert GEN_MIN_SIDE <= side <= GEN_MAX_SIDE, "unexpected square side length"

    patch = _crop(grid, bbox)
    area = int(coords.shape[0])
    all_red = np.argwhere(patch == RED)

    if area == side * side:
        assert np.all(patch == GREY), "solid square must be fully grey"
        assert all_red.size == 0, "solid square must not contain a red marker"
        return {"kind": "solid", "bbox": bbox, "side": side}

    border = np.zeros((side, side), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    grey_mask = patch == GREY
    assert np.array_equal(grey_mask, border), "frame grey cells must form a 1-pixel border"
    assert area == 4 * side - 4, "frame area must equal square perimeter"

    interior = patch[1:-1, 1:-1]
    assert interior.size > 0
    marker_locs = np.argwhere(interior == RED)
    assert marker_locs.shape == (1, 2), "frame interior must contain exactly one red marker"
    mk = _marker_offset(side) - 1
    assert tuple(map(int, marker_locs[0])) == (mk, mk), "marker must sit at the specified centered cell"
    allowed = {BLACK, RED}
    assert set(map(int, np.unique(interior))).issubset(allowed), "frame interior may only contain black and red"
    assert int(np.count_nonzero(interior == BLACK)) == (side - 2) * (side - 2) - 1, "frame must have a black hole"

    filled = patch.copy()
    filled[filled == BLACK] = SIDE_TO_FILL[side]
    return {
        "kind": "frame",
        "bbox": bbox,
        "side": side,
        "filled_patch": filled,
        "marker": (r0 + 1 + mk, c0 + 1 + mk),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Filter out solid grey squares, fill each retained hollow frame by its side length, then pack the frames into one sorted strip.

    Input structure:
    - A black background contains separated grey square objects with side length 4-8.
    - Some grey objects are 1-pixel-wide hollow square frames whose interiors are black except for one centered red marker pixel.
    - The remaining grey objects are solid grey squares with no red marker, and object bounding boxes do not overlap.

    Rule:
    - Find every 4-connected grey component and classify it as either a hollow frame or a solid square from its square bbox and interior structure.
    - Discard all solid squares. For each hollow frame, fill every interior black cell with the color determined by the outer side length: 4->blue, 5->teal, 6->yellow, 7->orange, 8->pink; keep the grey border and red marker unchanged.
    - Sort the filled frames by outer side length ascending, breaking ties by the original left column of the frame bbox, then paste them left-to-right with shared vertical borders, top-aligned, and black padding below shorter frames.

    Key insight:
    - The same geometric property, the hollow frame's outer side length, simultaneously decides which objects survive filtering, which fill color appears inside each frame, and where each finished frame lands in the final packed strip.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert _background_color(g) == BLACK, "background must be black"
    assert set(map(int, np.unique(g))).issubset({BLACK, RED, GREY}), "unexpected input colors"

    frames: list[dict] = []
    used_markers: set[tuple[int, int]] = set()
    for coords in _find_components_of_color(g, GREY):
        item = _classify_grey_component(g, coords)
        if item["kind"] == "frame":
            frames.append(item)
            used_markers.add(item["marker"])

    assert frames, "expected at least one hollow frame"
    red_pixels = {tuple(map(int, rc)) for rc in np.argwhere(g == RED)}
    assert red_pixels == used_markers, "every red pixel must be the marker of one retained frame"

    frames.sort(key=lambda item: (int(item["side"]), int(item["bbox"][1])))
    max_side = max(int(item["side"]) for item in frames)
    out_w = sum(int(item["side"]) for item in frames) - (len(frames) - 1)
    out = np.full((max_side, out_w), BLACK, dtype=int)

    col = 0
    for item in frames:
        patch = item["filled_patch"]
        side = int(item["side"])
        view = out[:side, col : col + side]
        overlap = view != BLACK
        if np.any(overlap):
            assert np.array_equal(view[overlap], patch[overlap]), "shared borders must be consistent"
        view[:, :] = patch
        col += side - 1

    return out


def _randint(lo: int, hi: int) -> int:
    return int(_RNG.integers(int(lo), int(hi) + 1))


def _sample_frame_sides(n_frames: int) -> list[int]:
    all_sides = np.arange(GEN_MIN_SIDE, GEN_MAX_SIDE + 1, dtype=int)
    while True:
        if _RNG.random() < 0.6:
            distinct_count = _randint(2, min(3, n_frames))
            distinct = _RNG.choice(all_sides, size=distinct_count, replace=False)
            sides = [int(x) for x in _RNG.choice(distinct, size=n_frames, replace=True)]
            if len(set(sides)) < 2 or len(set(sides)) == n_frames:
                continue
        else:
            sides = [int(x) for x in _RNG.choice(all_sides, size=n_frames, replace=False)]
        if len(set(sides)) >= 2:
            return sides


def _sample_solid_sides(n_solids: int, frame_sides: list[int]) -> list[int]:
    sides = [int(_RNG.choice(frame_sides))]
    while len(sides) < n_solids:
        if _RNG.random() < 0.45:
            sides.append(int(_RNG.choice(frame_sides)))
        else:
            sides.append(_randint(GEN_MIN_SIDE, GEN_MAX_SIDE))
    _RNG.shuffle(sides)
    return sides


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = GEN_MIN_GAP) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _place_objects(sides: list[int], shape: tuple[int, int]) -> list[tuple[int, int, int, int]] | None:
    h, w = shape
    order = list(enumerate(sides))
    order.sort(key=lambda item: int(item[1]), reverse=True)

    for _ in range(80):
        placed: dict[int, tuple[int, int, int, int]] = {}
        ok = True
        for idx, side in order:
            candidates: list[tuple[int, int, int, int]] = []
            for r in range(h - side + 1):
                for c in range(w - side + 1):
                    box = (r, c, r + side - 1, c + side - 1)
                    if all(_boxes_separated(box, other) for other in placed.values()):
                        candidates.append(box)
            if not candidates:
                ok = False
                break
            choice = candidates[_randint(0, len(candidates) - 1)]
            placed[idx] = choice
        if ok:
            return [placed[i] for i in range(len(sides))]
    return None


def _draw_square(grid: np.ndarray, box: tuple[int, int, int, int], *, hollow: bool) -> None:
    r0, c0, r1, c1 = box
    side = r1 - r0 + 1
    if hollow:
        patch = _frame_patch(side)
    else:
        patch = np.full((side, side), GREY, dtype=int)
    grid[r0 : r1 + 1, c0 : c1 + 1] = patch


def _semantic_summary(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    frames: list[dict] = []
    solids: list[dict] = []
    for coords in _find_components_of_color(grid, GREY):
        item = _classify_grey_component(grid, coords)
        if item["kind"] == "frame":
            frames.append(item)
        else:
            solids.append(item)
    return frames, solids


def _verify_sample(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if _background_color(grid) != BLACK:
        return False
    if not set(map(int, np.unique(grid))).issubset({BLACK, RED, GREY}):
        return False

    try:
        frames, solids = _semantic_summary(grid)
    except AssertionError:
        return False

    if not (GEN_MIN_FRAMES <= len(frames) <= GEN_MAX_FRAMES):
        return False
    if len(solids) < 2:
        return False
    if not (GEN_MIN_OBJECTS <= len(frames) + len(solids) <= GEN_MAX_OBJECTS):
        return False
    frame_sides = [int(item["side"]) for item in frames]
    solid_sides = [int(item["side"]) for item in solids]
    if len(set(frame_sides)) < 2:
        return False
    if not any(side in set(frame_sides) for side in solid_sides):
        return False

    ordered = sorted(frames, key=lambda item: (int(item["side"]), int(item["bbox"][1])))
    if ordered == frames and len(set(frame_sides)) == len(frame_sides):
        return False

    try:
        out = main(grid)
    except AssertionError:
        return False

    if out.ndim != 2 or max(out.shape) > 30:
        return False
    expected_h = max(frame_sides)
    expected_w = sum(frame_sides) - (len(frame_sides) - 1)
    if out.shape != (expected_h, expected_w):
        return False
    if len(set(map(int, np.unique(out))) - {BLACK, GREY, RED}) < 2:
        return False

    col = 0
    for item in ordered:
        patch = _filled_frame_patch(int(item["side"]))
        side = int(item["side"])
        if not np.array_equal(out[:side, col : col + side], patch):
            return False
        if side < expected_h and np.any(out[side:, col : col + side - 1] != BLACK):
            return False
        col += side - 1

    return True


def _attempt_generate() -> np.ndarray | None:
    n_frames = _randint(GEN_MIN_FRAMES, GEN_MAX_FRAMES)
    n_solids = _randint(2, GEN_MAX_OBJECTS - n_frames)
    frame_sides = _sample_frame_sides(n_frames)
    solid_sides = _sample_solid_sides(n_solids, frame_sides)
    all_sides = frame_sides + solid_sides

    base_h = max(max(all_sides) + 4, int(np.ceil(np.sqrt(sum((side + 1) ** 2 for side in all_sides)))))
    base_w = base_h
    grid_h = _randint(max(GEN_GRID_ROWS[0], base_h), GEN_GRID_ROWS[1])
    grid_w = _randint(max(GEN_GRID_COLS[0], base_w), GEN_GRID_COLS[1])

    boxes = _place_objects(all_sides, (grid_h, grid_w))
    if boxes is None:
        return None

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    for box in boxes[:n_frames]:
        _draw_square(grid, box, hollow=True)
    for box in boxes[n_frames:]:
        _draw_square(grid, box, hollow=False)

    if not _verify_sample(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid input grid")
