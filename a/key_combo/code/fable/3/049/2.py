from __future__ import annotations

import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    winners = values[counts == counts.max()]
    assert winners.size == 1, "expected a unique dominant background color"
    return int(winners[0])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    out: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        stack = [(sr, sc)]
        seen[sr, sc] = True
        coords: list[Coord] = []
        while stack:
            r, c = stack.pop()
            coords.append((r, c))
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_rectangular_frame(coords: np.ndarray) -> tuple[bool, BBox, int, int]:
    box = _bbox(coords)
    r0, c0, r1, c1 = box
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return False, box, height, width

    local = np.zeros((height, width), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    expected = np.zeros((height, width), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return bool(np.array_equal(local, expected)), box, height, width


def _parse_scene(grid: np.ndarray) -> tuple[int, list[dict], int, list[Coord]]:
    bg = _get_bg(grid)
    frames: list[dict] = []
    singletons: dict[int, list[Coord]] = {}

    for color in _colors_present(grid, exclude={bg}):
        for coords in _components(grid == color):
            is_frame, box, height, width = _is_rectangular_frame(coords)
            if is_frame:
                frames.append(
                    {
                        "color": int(color),
                        "bbox": box,
                        "height": int(height),
                        "width": int(width),
                        "size": int(max(height, width)),
                    }
                )
            elif int(coords.shape[0]) == 1:
                r, c = map(int, coords[0])
                singletons.setdefault(int(color), []).append((r, c))
            else:
                raise AssertionError("expected only rectangular frames and isolated marker pixels")

    assert len(frames) >= 2, "expected multiple rectangular frames"
    frame_colors = [frame["color"] for frame in frames]
    assert len(set(frame_colors)) == len(frame_colors), "frame colors must be unique"
    sizes = [frame["size"] for frame in frames]
    assert len(set(sizes)) == len(sizes), "frame sizes must be unique"

    marker_items = [(color, coords) for color, coords in singletons.items() if color not in set(frame_colors)]
    assert len(marker_items) == 1, "expected one shared marker color"
    marker_color, markers = marker_items[0]
    assert len(markers) > 0, "expected marker pixels"

    for color in singletons:
        assert color == marker_color, "only the marker color may appear as isolated pixels"

    for r, c in markers:
        assert int(grid[r, c]) == marker_color
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                assert int(grid[nr, nc]) != marker_color, "marker pixels must be isolated"

    return bg, frames, int(marker_color), markers


def _annotate_counts(frames: list[dict], markers: list[Coord]) -> list[dict]:
    counted = [dict(frame) for frame in frames]
    for frame in counted:
        r0, c0, r1, c1 = frame["bbox"]
        frame["count"] = sum(1 for r, c in markers if r0 < r < r1 and c0 < c < c1)
        assert frame["count"] > 0, "each frame must contain at least one marker"

    for marker in markers:
        containing = 0
        r, c = marker
        for frame in counted:
            r0, c0, r1, c1 = frame["bbox"]
            containing += int(r0 < r < r1 and c0 < c < c1)
        assert containing <= 1, "a marker may be inside at most one frame"

    return counted


def _render_rings(bg: int, frames: list[dict]) -> np.ndarray:
    layers = sorted(frames, key=lambda frame: frame["size"], reverse=True)
    total = sum(int(frame["count"]) for frame in layers)
    assert total > 0 and 2 * total - 1 <= 30, "output must fit ARC bounds"

    side = 2 * total - 1
    out = np.full((side, side), int(bg), dtype=int)
    inset = 0
    for frame in layers:
        count = int(frame["count"])
        assert count > 0
        out[inset : side - inset, inset : side - inset] = int(frame["color"])
        inset += count
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Rebuild scattered hollow rectangles as centered concentric square rings whose thicknesses are marker counts.

    Input structure:
    - A dominant background contains several non-touching hollow rectangular frames with distinct colors and distinct sizes.
    - A single marker color appears as isolated pixels, with some pixels strictly inside frames and some outside all frames.
    - Frame interiors may contain only background and marker pixels; frame positions do not determine the output positions.

    Rule:
    - Identify each rectangular frame by its colored perimeter and count marker pixels strictly inside its bbox.
    - Ignore marker pixels outside every frame.
    - Sort frame colors from outer to inner by decreasing frame side length and render centered square rings.
      Each color occupies as many consecutive layers as the marker count inside its source frame, and the output side
      length is twice the total inside-marker count minus one.

    Key insight:
    - The frame color supplies the ring color and the frame size supplies the ordering, while the number of enclosed
      marker pixels, not the frame thickness or position, encodes that color band's thickness.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg, frames, _marker_color, markers = _parse_scene(g)
    counted = _annotate_counts(frames, markers)
    return _render_rings(bg, counted)


# --------------- generator ---------------

ALL_COLORS = list(range(10))
FRAME_COUNT_CHOICES = (3, 4)
FRAME_SIZE_CHOICES = list(range(7, 14))
GRID_SIDE_BOUNDS = (24, 30)
COUNT_BOUNDS = (1, 4)
OUTSIDE_DOT_BOUNDS = (1, 5)
FRAME_GAP = 1
GENERATION_ATTEMPTS = 500


def _draw_frame(grid: np.ndarray, r0: int, c0: int, size: int, color: int) -> None:
    r1 = r0 + size - 1
    c1 = c0 + size - 1
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _valid_frame_positions(occupied: np.ndarray, size: int) -> list[Coord]:
    height, width = occupied.shape
    valid: list[Coord] = []
    for r in range(1, height - size):
        for c in range(1, width - size):
            r0 = max(0, r - FRAME_GAP)
            c0 = max(0, c - FRAME_GAP)
            r1 = min(height, r + size + FRAME_GAP)
            c1 = min(width, c + size + FRAME_GAP)
            if not occupied[r0:r1, c0:c1].any():
                valid.append((r, c))
    random.shuffle(valid)
    return valid


def _mark_frame_occupied(occupied: np.ndarray, r: int, c: int, size: int) -> None:
    height, width = occupied.shape
    r0 = max(0, r - FRAME_GAP)
    c0 = max(0, c - FRAME_GAP)
    r1 = min(height, r + size + FRAME_GAP)
    c1 = min(width, c + size + FRAME_GAP)
    occupied[r0:r1, c0:c1] = True


def _place_frames(height: int, width: int, sizes: list[int]) -> list[Coord] | None:
    occupied = np.zeros((height, width), dtype=bool)
    placements: list[Coord | None] = [None] * len(sizes)
    order = list(range(len(sizes)))
    random.shuffle(order)
    order.sort(key=lambda idx: sizes[idx], reverse=True)
    for idx in order:
        valid = _valid_frame_positions(occupied, sizes[idx])
        if not valid:
            return None
        r, c = random.choice(valid)
        placements[idx] = (r, c)
        _mark_frame_occupied(occupied, r, c, sizes[idx])
    return [p for p in placements if p is not None]


def _sparse_local_points(size: int, count: int) -> list[Coord] | None:
    candidates = [(r, c) for r in range(2, size - 2) for c in range(2, size - 2)]
    random.shuffle(candidates)
    chosen: list[Coord] = []

    def search(start: int) -> bool:
        if len(chosen) == count:
            return True
        remaining = count - len(chosen)
        if len(candidates) - start < remaining:
            return False
        order = list(range(start, len(candidates)))
        random.shuffle(order)
        for i in order:
            r, c = candidates[i]
            if any(max(abs(r - pr), abs(c - pc)) <= 1 for pr, pc in chosen):
                continue
            chosen.append((r, c))
            if search(i + 1):
                return True
            chosen.pop()
        return False

    return chosen.copy() if search(0) else None


def _choose_counts(count: int) -> list[int] | None:
    for _ in range(200):
        counts = [random.randint(*COUNT_BOUNDS) for _ in range(count)]
        if sum(counts) > 15:
            continue
        if max(counts) == 1 or len(set(counts)) == 1:
            continue
        if counts == sorted(counts) or counts == sorted(counts, reverse=True):
            continue
        return counts
    return None


def _inside_any_frame(r: int, c: int, boxes: list[BBox]) -> bool:
    return any(r0 < r < r1 and c0 < c < c1 for r0, c0, r1, c1 in boxes)


def _clear_outside_marker_cell(grid: np.ndarray, r: int, c: int, bg: int, boxes: list[BBox]) -> bool:
    if int(grid[r, c]) != bg or _inside_any_frame(r, c, boxes):
        return False
    height, width = grid.shape
    for nr, nc in _neighbors8(r, c):
        if 0 <= nr < height and 0 <= nc < width and int(grid[nr, nc]) != bg:
            return False
    return True


def _add_outside_markers(grid: np.ndarray, marker: int, boxes: list[BBox], count: int, bg: int) -> bool:
    candidates = [
        (r, c)
        for r in range(grid.shape[0])
        for c in range(grid.shape[1])
        if _clear_outside_marker_cell(grid, r, c, bg, boxes)
    ]
    random.shuffle(candidates)
    placed: list[Coord] = []
    for r, c in candidates:
        if any(max(abs(r - pr), abs(c - pc)) <= 1 for pr, pc in placed):
            continue
        grid[r, c] = marker
        placed.append((r, c))
        if len(placed) == count:
            return True
    return False


def _build_expected(bg: int, sizes: list[int], colors: list[int], counts: list[int]) -> np.ndarray:
    frames = [
        {"size": int(size), "color": int(color), "count": int(count)}
        for size, color, count in zip(sizes, colors, counts)
    ]
    return _render_rings(bg, frames)


def _generated_sample_is_semantic(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        bg, frames, _marker, markers = _parse_scene(grid)
        counted = _annotate_counts(frames, markers)
        actual = _render_rings(bg, counted)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected) or not np.array_equal(main(grid), expected):
        return False

    counts_by_size = [frame["count"] for frame in sorted(counted, key=lambda frame: frame["size"], reverse=True)]
    if max(counts_by_size) <= 1 or len(set(counts_by_size)) <= 1:
        return False
    if counts_by_size == sorted(counts_by_size) or counts_by_size == sorted(counts_by_size, reverse=True):
        return False

    outside = 0
    for r, c in markers:
        if not any(frame["bbox"][0] < r < frame["bbox"][2] and frame["bbox"][1] < c < frame["bbox"][3] for frame in counted):
            outside += 1
    return outside > 0


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.choice(FRAME_COUNT_CHOICES)
    sizes = sorted(random.sample(FRAME_SIZE_CHOICES, frame_count), reverse=True)
    counts = _choose_counts(frame_count)
    if counts is None:
        return None

    height = random.randint(*GRID_SIDE_BOUNDS)
    width = random.randint(*GRID_SIDE_BOUNDS)
    placements = _place_frames(height, width, sizes)
    if placements is None:
        return None

    bg = random.choice(ALL_COLORS)
    palette = [color for color in ALL_COLORS if color != bg]
    random.shuffle(palette)
    marker = palette[0]
    frame_colors = palette[1 : frame_count + 1]
    if len(frame_colors) != frame_count:
        return None

    grid = np.full((height, width), bg, dtype=int)
    boxes: list[BBox] = []
    for (r, c), size, color in zip(placements, sizes, frame_colors):
        _draw_frame(grid, r, c, size, color)
        boxes.append((r, c, r + size - 1, c + size - 1))

    for (r, c), size, count in zip(placements, sizes, counts):
        local_points = _sparse_local_points(size, count)
        if local_points is None:
            return None
        for dr, dc in local_points:
            grid[r + dr, c + dc] = marker

    outside_count = random.randint(*OUTSIDE_DOT_BOUNDS)
    if not _add_outside_markers(grid, marker, boxes, outside_count, bg):
        return None

    expected = _build_expected(bg, sizes, frame_colors, counts)
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    if not _generated_sample_is_semantic(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid thickness-encoded ring input")
