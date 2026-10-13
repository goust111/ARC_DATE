from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
MARKER = 2
FRAME_COLORS = (1, 3, 4)
OUTPUT_SIZE = 15

GRID_MIN = 11
GRID_MAX = 14
FRAME_COUNT_OPTIONS = (2, 3)
INTERIOR_HEIGHTS = (2, 3, 4)
INTERIOR_WIDTHS = (2, 3, 4, 5)
FRAME_GAP = 1
DISTRACTOR_RANGE = (2, 6)
MIN_VISIBLE_PIXELS = 3
MAX_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.asarray(grid).ravel()))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rect_border_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    return mask


def _boxes_clear(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
    gap: int = 0,
) -> bool:
    r0, c0, r1, c1 = first
    rr0, cc0, rr1, cc1 = second
    return (
        r1 + gap < rr0
        or rr1 + gap < r0
        or c1 + gap < cc0
        or cc1 + gap < c0
    )


def _line_positions(count: int) -> tuple[int, ...]:
    return tuple(range(int(count), OUTPUT_SIZE, int(count) + 1))


def _parse_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    assert bg == BG, "expected a black background"

    extra_colors = set(_colors_present(g, exclude={bg, MARKER}))
    assert extra_colors and extra_colors.issubset(set(FRAME_COLORS)), "unexpected non-frame colors"

    frames: list[dict] = []
    for color in sorted(extra_colors):
        mask = g == int(color)
        box = _bbox(mask)
        assert box is not None, "missing frame bbox"
        r0, c0, r1, c1 = box
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must have a non-empty interior"

        border = _rect_border_mask(g.shape, box)
        assert np.array_equal(mask, border), "frame must be a single-pixel hollow rectangle"

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        assert interior.size > 0
        assert np.all((interior == bg) | (interior == MARKER)), "frame interior may only contain marker or background"

        frames.append(
            {
                "color": int(color),
                "box": box,
                "area": int(interior.size),
                "count": int(np.count_nonzero(interior == MARKER)),
                "positions": _line_positions(int(np.count_nonzero(interior == MARKER))),
            }
        )

    assert 2 <= len(frames) <= 3, "expected two or three frames"
    counts = [frame["count"] for frame in frames]
    assert len(set(counts)) == len(counts), "frame marker counts must be distinct"
    assert all(2 <= count <= 5 for count in counts), "frame counts must be in 2..5"

    areas = [frame["area"] for frame in frames]
    assert len(set(areas)) == len(areas), "frame interior areas must be distinct"

    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            assert _boxes_clear(frames[i]["box"], frames[j]["box"], gap=0), "frames may not overlap or nest"

    marker_outside = (g == MARKER)
    for frame in frames:
        r0, c0, r1, c1 = frame["box"]
        marker_outside[r0 + 1 : r1, c0 + 1 : c1] = False
    assert np.any(marker_outside), "expected at least one distractor marker outside the frames"
    return frames


def _render_from_frames(frames: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    owner = np.full((OUTPUT_SIZE, OUTPUT_SIZE), -1, dtype=int)
    order = sorted(range(len(frames)), key=lambda idx: frames[idx]["area"], reverse=True)
    for idx in order:
        positions = np.array(frames[idx]["positions"], dtype=int)
        owner[positions, :] = idx
        owner[:, positions] = idx

    out = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=int)
    for idx, frame in enumerate(frames):
        out[owner == idx] = int(frame["color"])
    return out, owner


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count red markers inside each rectangular frame and overlay the resulting parameterized lattice lines on one shared canvas.

    Input structure:
    - The background is black.
    - The grid contains two or three non-overlapping, one-pixel-thick rectangular frames, each in a distinct frame color.
    - Red single-pixel markers are scattered both inside frame interiors and on the outside background; only strict interior markers belong to a frame.

    Rule:
    - For each frame, count the red markers strictly inside it to obtain N.
    - Create a black 15x15 output and paint that frame's color on every row and column at indices N, N+(N+1), N+2(N+1), ... while in bounds.
    - When multiple frame lattices cover the same output pixel, keep the color of the frame with the smaller interior area.

    Key insight:
    - The red markers are not rendered directly; each interior count becomes a spacing parameter for a lattice, and overlapping lattices are resolved by comparing the visible frame sizes.
    """
    frames = _parse_frames(np.asarray(grid, dtype=int))
    out, _owner = _render_from_frames(frames)
    return out


def _catalog_dims() -> list[tuple[int, int, int]]:
    dims: list[tuple[int, int, int]] = []
    for height in INTERIOR_HEIGHTS:
        for width in INTERIOR_WIDTHS:
            dims.append((int(height), int(width), int(height * width)))
    return dims


DIM_CATALOG = _catalog_dims()


def _sample_frame_specs(frame_count: int) -> list[dict] | None:
    counts = random.sample([2, 3, 4, 5], frame_count)
    for _ in range(200):
        dims = random.sample(DIM_CATALOG, frame_count)
        areas = [area for _h, _w, area in dims]
        if len(set(areas)) != frame_count:
            continue

        random.shuffle(dims)
        specs = []
        ok = True
        for count, (inner_h, inner_w, area) in zip(counts, dims):
            if area < count + 2:
                ok = False
                break
            specs.append(
                {
                    "count": int(count),
                    "inner_h": int(inner_h),
                    "inner_w": int(inner_w),
                    "area": int(area),
                    "outer_h": int(inner_h + 2),
                    "outer_w": int(inner_w + 2),
                }
            )
        if not ok:
            continue

        area_order = tuple(idx for idx, _ in sorted(enumerate(specs), key=lambda item: item[1]["area"]))
        count_order = tuple(idx for idx, _ in sorted(enumerate(specs), key=lambda item: item[1]["count"]))
        if area_order == count_order:
            continue

        trial_frames = [
            {
                "color": FRAME_COLORS[idx],
                "area": spec["area"],
                "count": spec["count"],
                "positions": _line_positions(spec["count"]),
            }
            for idx, spec in enumerate(specs)
        ]
        _trial_out, owner = _render_from_frames(trial_frames)
        if any(int(np.count_nonzero(owner == idx)) < MIN_VISIBLE_PIXELS for idx in range(frame_count)):
            continue
        return specs
    return None


def _place_boxes(
    height: int,
    width: int,
    specs: list[dict],
) -> list[tuple[int, int, int, int]] | None:
    order = sorted(range(len(specs)), key=lambda idx: (specs[idx]["outer_h"] * specs[idx]["outer_w"]), reverse=True)
    placed: dict[int, tuple[int, int, int, int]] = {}

    candidates: dict[int, list[tuple[int, int, int, int]]] = {}
    for idx, spec in enumerate(specs):
        outer_h = spec["outer_h"]
        outer_w = spec["outer_w"]
        boxes = []
        for r0 in range(height - outer_h + 1):
            for c0 in range(width - outer_w + 1):
                boxes.append((r0, c0, r0 + outer_h - 1, c0 + outer_w - 1))
        random.shuffle(boxes)
        candidates[idx] = boxes

    def search(step: int) -> bool:
        if step == len(order):
            return True
        idx = order[step]
        for box in candidates[idx]:
            if all(_boxes_clear(box, other, gap=FRAME_GAP) for other in placed.values()):
                placed[idx] = box
                if search(step + 1):
                    return True
                placed.pop(idx)
        return False

    if not search(0):
        return None
    return [placed[idx] for idx in range(len(specs))]


def _place_markers(grid: np.ndarray, box: tuple[int, int, int, int], count: int) -> None:
    r0, c0, r1, c1 = box
    coords = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    chosen = random.sample(coords, int(count))
    for r, c in chosen:
        grid[r, c] = MARKER


def _outside_cells(grid: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int]]:
    outside = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] != BG:
                continue
            if any(r0 <= r <= r1 and c0 <= c <= c1 for r0, c0, r1, c1 in boxes):
                continue
            outside.append((r, c))
    return outside


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        frames = _parse_frames(grid)
        out, owner = _render_from_frames(frames)
        if not np.array_equal(main(grid), out):
            return False

        area_order = tuple(idx for idx, _ in sorted(enumerate(frames), key=lambda item: item[1]["area"]))
        count_order = tuple(idx for idx, _ in sorted(enumerate(frames), key=lambda item: item[1]["count"]))
        if area_order == count_order:
            return False

        for idx in range(len(frames)):
            if int(np.count_nonzero(owner == idx)) < MIN_VISIBLE_PIXELS:
                return False

        for i in range(len(frames)):
            for j in range(i + 1, len(frames)):
                mask_i = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=bool)
                mask_j = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=bool)
                pos_i = np.array(frames[i]["positions"], dtype=int)
                pos_j = np.array(frames[j]["positions"], dtype=int)
                mask_i[pos_i, :] = True
                mask_i[:, pos_i] = True
                mask_j[pos_j, :] = True
                mask_j[:, pos_j] = True
                overlap = mask_i & mask_j
                if not np.any(overlap):
                    return False
                winner = i if frames[i]["area"] < frames[j]["area"] else j
                if not np.any((owner == winner) & overlap):
                    return False

        return True
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.choice(FRAME_COUNT_OPTIONS)
    specs = _sample_frame_specs(frame_count)
    if specs is None:
        return None

    colors = random.sample(list(FRAME_COLORS), frame_count)
    for spec, color in zip(specs, colors):
        spec["color"] = int(color)

    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    boxes = _place_boxes(height, width, specs)
    if boxes is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    for spec, box in zip(specs, boxes):
        r0, c0, r1, c1 = box
        color = int(spec["color"])
        grid[r0 : r1 + 1, c0] = color
        grid[r0 : r1 + 1, c1] = color
        grid[r0, c0 : c1 + 1] = color
        grid[r1, c0 : c1 + 1] = color
        _place_markers(grid, box, spec["count"])

    outside = _outside_cells(grid, boxes)
    distractor_count = random.randint(*DISTRACTOR_RANGE)
    if len(outside) < distractor_count:
        return None
    for r, c in random.sample(outside, distractor_count):
        grid[r, c] = MARKER

    return grid if _semantic_checks(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
