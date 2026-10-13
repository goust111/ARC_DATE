import random
from collections import deque

import numpy as np


ORTHOGONALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Generator configuration.
COLOR_IDS = tuple(range(10))
GRID_HEIGHT_RANGE = (14, 24)
GRID_WIDTH_RANGE = (14, 24)
OBJECT_COUNT_CHOICES = (1, 2, 3, 4)
OBJECT_COUNT_WEIGHTS = (2, 3, 3, 2)
FRAME_HEIGHT_RANGE = (3, 8)
FRAME_WIDTH_RANGE = (3, 9)
FRAME_SIZE_PAIRS = [
    (h, w)
    for h in range(FRAME_HEIGHT_RANGE[0], FRAME_HEIGHT_RANGE[1] + 1)
    for w in range(FRAME_WIDTH_RANGE[0], FRAME_WIDTH_RANGE[1] + 1)
]
OUTER_MARGIN = 1
MAX_MARKERS_PER_FRAME = 4
FOREGROUND_RATIO_MAX = 0.45
GENERATION_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in ORTHOGONALS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(coords)

    return parts


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _rect_frame_bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None

    expected = 2 * (height + width) - 4
    if len(coords) != expected:
        return None

    pixels = set(coords)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_perimeter = r in (r0, r1) or c in (c0, c1)
            if ((r, c) in pixels) != on_perimeter:
                return None
    return r0, c0, r1, c1


def _analyze_grid(
    grid: np.ndarray,
    *,
    require_unique_marker_colors: bool = False,
) -> tuple[int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    frame_mask = np.zeros_like(g, dtype=bool)
    frames: list[dict[str, object]] = []

    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        for coords in _components(g == color):
            box = _rect_frame_bbox(coords)
            if box is None:
                continue
            for r, c in coords:
                assert not frame_mask[r, c], "frame components must not overlap"
                frame_mask[r, c] = True
            frames.append(
                {
                    "bbox": box,
                    "frame_color": color,
                    "frame_coords": coords,
                }
            )

    assert frames, "expected at least one rectangular frame"
    frames.sort(key=lambda frame: frame["bbox"])

    marker_mask = np.zeros_like(g, dtype=bool)
    marker_colors: list[int] = []
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        marker_local = np.argwhere(interior != bg)
        assert len(marker_local) > 0, "every frame must contain interior marker pixels"
        assert len(marker_local) <= MAX_MARKERS_PER_FRAME, "expected at most four interior markers"

        colors = {int(interior[rr, cc]) for rr, cc in marker_local}
        assert len(colors) == 1, "markers inside a frame must share one color"
        marker_color = next(iter(colors))
        assert marker_color != frame["frame_color"], "marker color must differ from frame color"

        marker_coords: list[tuple[int, int]] = []
        for rr, cc in marker_local:
            r = int(r0 + 1 + rr)
            c = int(c0 + 1 + cc)
            assert not frame_mask[r, c], "markers must stay inside the frame interior"
            assert not marker_mask[r, c], "frames must not share marker pixels"
            marker_mask[r, c] = True
            marker_coords.append((r, c))

        frame["marker_color"] = marker_color
        frame["marker_coords"] = marker_coords
        marker_colors.append(marker_color)

    covered = frame_mask | marker_mask
    assert np.array_equal(g != bg, covered), "unexpected foreground pixels outside frames or markers"
    if require_unique_marker_colors:
        assert len(set(marker_colors)) == len(marker_colors), "marker colors should be unique across frames"

    return bg, frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap each rectangular frame's color with the shared color of the marker pixels inside it.
    """
    g = np.asarray(grid, dtype=int)
    _, frames = _analyze_grid(g)

    out = g.copy()
    for frame in frames:
        frame_color = int(frame["frame_color"])
        marker_color = int(frame["marker_color"])
        for r, c in frame["frame_coords"]:
            out[r, c] = marker_color
        for r, c in frame["marker_coords"]:
            out[r, c] = frame_color
    return out


def _draw_frame(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _candidate_positions(blocked: np.ndarray, height: int, width: int) -> list[tuple[int, int]]:
    h, w = blocked.shape
    if height + 2 * OUTER_MARGIN > h or width + 2 * OUTER_MARGIN > w:
        return []

    positions: list[tuple[int, int]] = []
    for r0 in range(OUTER_MARGIN, h - height - OUTER_MARGIN + 1):
        for c0 in range(OUTER_MARGIN, w - width - OUTER_MARGIN + 1):
            if np.any(blocked[r0 - 1 : r0 + height + 1, c0 - 1 : c0 + width + 1]):
                continue
            positions.append((r0, c0))
    return positions


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    separated = ar1 + 1 < br0 or br1 + 1 < ar0 or ac1 + 1 < bc0 or bc1 + 1 < ac0
    return not separated


def _place_frame(
    grid: np.ndarray,
    blocked: np.ndarray,
    *,
    frame_color: int,
    marker_color: int,
) -> tuple[int, int, int, int] | None:
    size_pairs = FRAME_SIZE_PAIRS[:]
    random.shuffle(size_pairs)

    for height, width in size_pairs:
        positions = _candidate_positions(blocked, height, width)
        if not positions:
            continue

        r0, c0 = random.choice(positions)
        r1, c1 = r0 + height - 1, c0 + width - 1
        _draw_frame(grid, r0, c0, r1, c1, frame_color)

        interior_cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
        marker_count = random.randint(1, min(MAX_MARKERS_PER_FRAME, len(interior_cells)))
        for r, c in random.sample(interior_cells, marker_count):
            grid[r, c] = marker_color

        # Reserve a one-cell moat so frames remain visually separate.
        blocked[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2] = True
        return r0, c0, r1, c1

    return None


def _expected_output_from_frames(grid: np.ndarray, frames: list[dict[str, object]]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for frame in frames:
        frame_color = int(frame["frame_color"])
        marker_color = int(frame["marker_color"])
        for r, c in frame["frame_coords"]:
            out[r, c] = marker_color
        for r, c in frame["marker_coords"]:
            out[r, c] = frame_color
    return out


def _validate_generated_grid(
    grid: np.ndarray,
    *,
    expected_frame_count: int,
    expected_frame_color: int,
) -> None:
    bg, frames = _analyze_grid(grid, require_unique_marker_colors=True)
    assert len(frames) == expected_frame_count, "unexpected frame count"
    assert {int(frame["frame_color"]) for frame in frames} == {expected_frame_color}, "all frames must share one input color"
    assert np.count_nonzero(grid != bg) < grid.size / 2, "background must be dominant"

    boxes = [frame["bbox"] for frame in frames]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            assert not _boxes_too_close(boxes[i], boxes[j]), "frames must not touch"

    expected = _expected_output_from_frames(grid, frames)
    actual = main(grid)
    assert np.array_equal(actual, expected), "main() must perform the intended swap"


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    bg, frame_color = random.sample(COLOR_IDS, 2)
    frame_count = random.choices(OBJECT_COUNT_CHOICES, weights=OBJECT_COUNT_WEIGHTS, k=1)[0]

    marker_palette = [color for color in COLOR_IDS if color not in {bg, frame_color}]
    if frame_count > len(marker_palette):
        return None
    marker_colors = random.sample(marker_palette, frame_count)

    grid = np.full((height, width), bg, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []
    for marker_color in marker_colors:
        box = _place_frame(grid, blocked, frame_color=frame_color, marker_color=marker_color)
        if box is None:
            return None
        boxes.append(box)

    try:
        _validate_generated_grid(
            grid,
            expected_frame_count=frame_count,
            expected_frame_color=frame_color,
        )
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")
