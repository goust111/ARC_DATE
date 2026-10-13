from __future__ import annotations

import random
from collections import Counter
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_shared_032_2",
    str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py")),
).load_module()

Color = _ARC_UTILS.Color
find_components = _ARC_UTILS.find_components
paste = _ARC_UTILS.paste


BG = Color.BLACK
SOURCE = Color.GREY
MAJ_COLOR = Color.TEAL
MIN_COLOR = Color.RED
NWSE = "NWSE"
NESW = "NESW"

GRID_MIN = 24
GRID_MAX = 30
SEGMENT_COUNT_MIN = 5
SEGMENT_COUNT_MAX = 8
SEGMENT_SIZES = (3, 4, 5)
SEGMENT_SIZE_WEIGHTS = (0.45, 0.35, 0.20)
MAX_GENERATION_ATTEMPTS = 600
MAX_SEGMENT_PLACEMENT_ATTEMPTS = 250
REQUIRE_OVERLAP_PROB = 0.35
MAX_OVERLAP_PIXELS = 18
MIN_SPECIAL_PIXELS = 2


def _diag_rel_coords(orientation: str, size: int) -> np.ndarray:
    if orientation == NWSE:
        return np.array([(i, i) for i in range(size)], dtype=int)
    if orientation == NESW:
        return np.array([(i, size - 1 - i) for i in range(size)], dtype=int)
    raise ValueError(f"unknown orientation: {orientation}")


def _make_segment(orientation: str, size: int, r0: int, c0: int) -> dict:
    rel = _diag_rel_coords(orientation, size)
    bbox = (int(r0), int(c0), int(r0 + size - 1), int(c0 + size - 1))
    return {
        "orientation": orientation,
        "size": int(size),
        "bbox": bbox,
        "rel": rel,
    }


def _patch_from_rel(rel: np.ndarray, size: int, color: int) -> np.ndarray:
    patch = np.full((size, size), BG, dtype=int)
    patch[rel[:, 0], rel[:, 1]] = int(color)
    return patch


def _interval_gap(a0: int, a1: int, b0: int, b1: int) -> int:
    if a1 < b0:
        return int(b0 - a1 - 1)
    if b1 < a0:
        return int(a0 - b1 - 1)
    return 0


def _bbox_spacing_ok(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    row_gap = _interval_gap(a[0], a[2], b[0], b[2])
    col_gap = _interval_gap(a[1], a[3], b[1], b[3])
    return row_gap >= 2 or col_gap >= 2


def _extract_segments(grid: np.ndarray) -> list[dict]:
    comps = find_components(grid, bg=BG, connectivity=8, monochromatic=True)
    segments: list[dict] = []
    for comp in comps:
        color = int(comp["color"])
        assert color == SOURCE, "expected grey diagonal segments only"

        r0, c0, r1, c1 = map(int, comp["bbox"])
        size_r = r1 - r0 + 1
        size_c = c1 - c0 + 1
        assert size_r == size_c and 3 <= size_r <= 5, "segment bbox must be square 3..5"

        coords = np.asarray(comp["coords"], dtype=int)
        assert coords.shape[0] == size_r, "each segment must be a contiguous diagonal line"
        rel = coords - np.array([[r0, c0]], dtype=int)

        row_counts = np.bincount(rel[:, 0], minlength=size_r)
        col_counts = np.bincount(rel[:, 1], minlength=size_c)
        assert np.all(row_counts == 1) and np.all(col_counts == 1), "segment must use one cell per row/col"

        ordered = rel[np.argsort(rel[:, 0])]
        cols_by_row = ordered[:, 1].tolist()
        if cols_by_row == list(range(size_r)):
            orientation = NWSE
        elif cols_by_row == list(range(size_r - 1, -1, -1)):
            orientation = NESW
        else:
            raise AssertionError("segment must follow a 45-degree diagonal")

        segments.append(
            {
                "orientation": orientation,
                "size": int(size_r),
                "bbox": (r0, c0, r1, c1),
                "rel": rel,
            }
        )

    segments.sort(key=lambda rec: (rec["bbox"][1], rec["bbox"][0]))
    return segments


def _majority_orientation(segments: list[dict]) -> str:
    counts = Counter(seg["orientation"] for seg in segments)
    return NWSE if counts[NWSE] >= counts[NESW] else NESW


def _render(segments: list[dict], shape: tuple[int, int], majority_orientation: str) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for seg in sorted(segments, key=lambda rec: (rec["bbox"][1], rec["bbox"][0])):
        size = int(seg["size"])
        step = size + 1
        horizontal = seg["orientation"] == majority_orientation
        for idx in range(4):
            rr = int(seg["bbox"][0]) + (0 if horizontal else idx * step)
            cc = int(seg["bbox"][1]) + (idx * step if horizontal else 0)
            color = SOURCE
            if idx == 2:
                color = MAJ_COLOR if horizontal else MIN_COLOR
            patch = _patch_from_rel(seg["rel"], size, color)
            paste(out, patch, rr, cc, transparent=BG, clip=True)
    return out


def _occupancy_counts(segments: list[dict], shape: tuple[int, int], majority_orientation: str) -> np.ndarray:
    counts = np.zeros(shape, dtype=int)
    h, w = shape
    for seg in sorted(segments, key=lambda rec: (rec["bbox"][1], rec["bbox"][0])):
        size = int(seg["size"])
        step = size + 1
        horizontal = seg["orientation"] == majority_orientation
        for idx in range(4):
            base_r = int(seg["bbox"][0]) + (0 if horizontal else idx * step)
            base_c = int(seg["bbox"][1]) + (idx * step if horizontal else 0)
            for dr, dc in seg["rel"]:
                rr = base_r + int(dr)
                cc = base_c + int(dc)
                if 0 <= rr < h and 0 <= cc < w:
                    counts[rr, cc] += 1
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Classify diagonal segments by orientation, then stamp the majority class horizontally and the minority class vertically.

    Input structure:
    - A black background contains 5 to 8 disconnected grey diagonal line segments.
    - Each segment is a straight 45-degree run of length 3 to 5, with either NW-SE or NE-SW orientation.
    - Both orientations are present, and the segments are separated so their original bounding boxes do not touch.

    Rule:
    - Detect each grey segment and classify its diagonal orientation from its own bounding box coordinates.
    - Count the two orientations; the majority orientation is M and the other is m, with NW-SE winning ties.
    - For every M segment, stamp four total copies of that segment at its original row, spaced rightward by bbox_width + 1; only copy index 2 is recolored teal and the others stay grey.
    - For every m segment, stamp four total copies of that segment at its original column, spaced downward by bbox_height + 1; only copy index 2 is recolored red and the others stay grey, with later segments overwriting earlier ones on overlaps.

    Key insight:
    - The global majority vote over diagonal orientation controls two downstream effects at once: which axis each segment propagates along, and which special recolor appears on its third stamped copy.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "grid must fit ARC size limits"

    segments = _extract_segments(g)
    majority_orientation = _majority_orientation(segments)
    return _render(segments, g.shape, majority_orientation)


def _sample_segment_size() -> int:
    return int(random.choices(SEGMENT_SIZES, weights=SEGMENT_SIZE_WEIGHTS, k=1)[0])


def _sample_counts(total: int) -> tuple[str, int, int]:
    majority_orientation = random.choice((NWSE, NESW))
    minority_count = random.randint(1, total // 2)
    majority_count = total - minority_count
    return majority_orientation, majority_count, minority_count


def _place_segments(shape: tuple[int, int], orientations: list[str], sizes: list[int]) -> list[dict] | None:
    h, w = shape
    majority_orientation = NWSE if orientations.count(NWSE) > orientations.count(NESW) else NESW
    placed: list[dict] = []

    order = list(range(len(orientations)))
    order.sort(key=lambda idx: sizes[idx], reverse=True)

    for idx in order:
        orientation = orientations[idx]
        size = sizes[idx]
        horizontal = orientation == majority_orientation
        max_r0 = h - size if horizontal else h - (4 * size + 3)
        max_c0 = w - (4 * size + 3) if horizontal else w - size
        if max_r0 < 0 or max_c0 < 0:
            return None

        placed_here = False
        for _ in range(MAX_SEGMENT_PLACEMENT_ATTEMPTS):
            if horizontal:
                c0 = random.randint(0, max_c0)
                r0 = random.randint(0, h - size)
            else:
                r0 = random.randint(0, max_r0)
                c0 = random.randint(0, w - size)

            candidate = _make_segment(orientation, size, r0, c0)
            if all(_bbox_spacing_ok(candidate["bbox"], other["bbox"]) for other in placed):
                placed.append(candidate)
                placed_here = True
                break
        if not placed_here:
            return None

    return placed


def _input_grid_from_segments(shape: tuple[int, int], segments: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for seg in segments:
        patch = _patch_from_rel(seg["rel"], int(seg["size"]), SOURCE)
        paste(grid, patch, int(seg["bbox"][0]), int(seg["bbox"][1]), transparent=BG, clip=False)
    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    segments = _extract_segments(grid)
    if not (SEGMENT_COUNT_MIN <= len(segments) <= SEGMENT_COUNT_MAX):
        return False

    counts = Counter(seg["orientation"] for seg in segments)
    if counts[NWSE] == 0 or counts[NESW] == 0 or counts[NWSE] == counts[NESW]:
        return False

    for i, seg_a in enumerate(segments):
        for seg_b in segments[i + 1 :]:
            if not _bbox_spacing_ok(seg_a["bbox"], seg_b["bbox"]):
                return False

    majority_orientation = _majority_orientation(segments)
    minority_orientation = NESW if majority_orientation == NWSE else NWSE

    out = _render(segments, grid.shape, majority_orientation)
    alt = _render(segments, grid.shape, minority_orientation)
    if np.array_equal(out, alt):
        return False
    if np.array_equal(out, grid):
        return False

    teal_count = int(np.count_nonzero(out == MAJ_COLOR))
    red_count = int(np.count_nonzero(out == MIN_COLOR))
    if teal_count < MIN_SPECIAL_PIXELS or red_count < MIN_SPECIAL_PIXELS:
        return False

    overlap_pixels = int(np.count_nonzero(_occupancy_counts(segments, grid.shape, majority_orientation) > 1))
    if overlap_pixels > MAX_OVERLAP_PIXELS:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    total = random.randint(SEGMENT_COUNT_MIN, SEGMENT_COUNT_MAX)
    majority_orientation, majority_count, minority_count = _sample_counts(total)
    minority_orientation = NESW if majority_orientation == NWSE else NWSE

    orientations = [majority_orientation] * majority_count + [minority_orientation] * minority_count
    random.shuffle(orientations)
    sizes = [_sample_segment_size() for _ in orientations]

    segments = _place_segments((h, w), orientations, sizes)
    if segments is None:
        return None

    grid = _input_grid_from_segments((h, w), segments)
    parsed = _extract_segments(grid)
    majority = _majority_orientation(parsed)
    overlaps = int(np.count_nonzero(_occupancy_counts(parsed, grid.shape, majority) > 1))

    require_overlap = random.random() < REQUIRE_OVERLAP_PROB
    if require_overlap and overlaps == 0:
        return None
    if overlaps > MAX_OVERLAP_PIXELS:
        return None

    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diagonal-stamping ARC input")
