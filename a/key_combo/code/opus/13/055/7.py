from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4

GEN_HEIGHT_RANGE = (13, 18)
GEN_WIDTH_RANGE = (13, 18)
GEN_SEGMENT_COUNT_RANGE = (3, 5)
GEN_LENGTH_OPTIONS = (2, 3, 4, 5, 6)
GEN_MAX_ATTEMPTS = 800


def _find_red_segments(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return maximal down-right red segments, indexed from each top-left endpoint."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    red = g == RED
    seen = np.zeros_like(red, dtype=bool)
    segments: list[list[tuple[int, int]]] = []

    for r, c in np.argwhere(red):
        r = int(r)
        c = int(c)
        if r > 0 and c > 0 and red[r - 1, c - 1]:
            continue
        seg: list[tuple[int, int]] = []
        rr, cc = r, c
        while rr < h and cc < w and red[rr, cc]:
            seg.append((rr, cc))
            seen[rr, cc] = True
            rr += 1
            cc += 1
        segments.append(seg)

    assert np.array_equal(seen, red), "all red pixels must belong to maximal down-right segments"
    return segments


def _segment_border(center: tuple[int, int], radius: int) -> set[tuple[int, int]]:
    cr, cc = center
    border: set[tuple[int, int]] = set()
    if radius == 0:
        border.add((cr, cc))
        return border
    for c in range(cc - radius, cc + radius + 1):
        border.add((cr - radius, c))
        border.add((cr + radius, c))
    for r in range(cr - radius + 1, cr + radius):
        border.add((r, cc - radius))
        border.add((r, cc + radius))
    return border


def _paint_border(out: np.ndarray, center: tuple[int, int], radius: int, color: int) -> None:
    for r, c in _segment_border(center, radius):
        out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor odd-indexed pixels in each red down-right segment, then draw a green hollow square whose size is driven by the yellow-count of that segment.

    Input structure:
    - A black background contains 3-5 maximal red diagonal segments that run in the down-right direction.
    - Each segment is one cell wide, length 2-6, and indexed from its top-left endpoint.
    - Segments are spaced so every square can fit inside the grid and segment bodies remain distinguishable.

    Rule:
    - Find every maximal contiguous red segment along direction (row+1, col+1).
    - Within each segment, keep even indices red and recolor odd indices to yellow.
    - Let Y be the number of yellow cells in that segment, and let the segment center be index floor(L/2).
    - Draw a green 1-pixel-thick hollow square centered on that middle cell with side length 2Y+1.
    - If square borders overlap, the one from the larger Y is on top; ties use smaller center row, then smaller center column.
    - Red/yellow segment pixels remain the top layer and are never hidden by green borders.

    Key insight:
    - The alternating recolor step is the measurement step: the number of newly yellow pixels, not the raw segment length alone, determines each square radius.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    segments = _find_red_segments(g)

    square_specs = []
    for seg in segments:
        length = len(seg)
        center = seg[length // 2]
        radius = length // 2
        square_specs.append((radius, center))

    # Lower-priority borders first, higher-priority borders last.
    for radius, center in sorted(square_specs, key=lambda item: (item[0], -item[1][0], -item[1][1])):
        _paint_border(out, center, radius, GREEN)

    # Segment pixels always stay on top.
    for seg in segments:
        for idx, (r, c) in enumerate(seg):
            out[r, c] = YELLOW if idx % 2 == 1 else RED

    return out


def _segment_from_center(center: tuple[int, int], length: int) -> list[tuple[int, int]]:
    cr, cc = center
    start_offset = length // 2
    start_r = cr - start_offset
    start_c = cc - start_offset
    return [(start_r + i, start_c + i) for i in range(length)]


def _box_bbox(center: tuple[int, int], radius: int) -> tuple[int, int, int, int]:
    cr, cc = center
    return cr - radius, cc - radius, cr + radius, cc + radius


def _boxes_overlap(a_center: tuple[int, int], a_radius: int, b_center: tuple[int, int], b_radius: int) -> bool:
    return bool(_segment_border(a_center, a_radius) & _segment_border(b_center, b_radius))


def _length_choices(count: int, require_equal_radius_pair: bool) -> list[int]:
    for _ in range(300):
        lengths = [random.choice(GEN_LENGTH_OPTIONS) for _ in range(count)]
        if len(set(lengths)) < 2:
            continue
        if require_equal_radius_pair:
            radii = [length // 2 for length in lengths]
            if len(radii) == len(set(radii)):
                continue
        return lengths
    raise RuntimeError("failed to sample non-degenerate segment lengths")


def _center_candidates(height: int, width: int, length: int) -> list[tuple[int, int]]:
    radius = length // 2
    return [
        (r, c)
        for r in range(radius, height - radius)
        for c in range(radius, width - radius)
    ]


def _segment_body_hits_border(
    segment: Iterable[tuple[int, int]],
    other_center: tuple[int, int],
    other_radius: int,
) -> bool:
    border = _segment_border(other_center, other_radius)
    return any(cell in border for cell in segment)


def _whole_square_contains_segment(
    segment: Iterable[tuple[int, int]],
    other_center: tuple[int, int],
    other_radius: int,
) -> bool:
    r0, c0, r1, c1 = _box_bbox(other_center, other_radius)
    for r, c in segment:
        if r0 <= r <= r1 and c0 <= c <= c1:
            return True
    return False


def _is_valid_placement(
    candidate: dict,
    placed: list[dict],
    height: int,
    width: int,
) -> bool:
    segment = candidate["segment"]
    center = candidate["center"]
    radius = candidate["radius"]

    for r, c in segment:
        if not (0 <= r < height and 0 <= c < width):
            return False

    # Avoid accidental merge or branching in the red geometry.
    occupied = {cell for item in placed for cell in item["segment"]}
    if any(cell in occupied for cell in segment):
        return False

    for other in placed:
        other_center = other["center"]
        other_radius = other["radius"]

        if abs(center[0] - other_center[0]) + abs(center[1] - other_center[1]) < 8:
            return False

        for r, c in segment:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nbr = (r + dr, c + dc)
                    if nbr in other["segment"]:
                        return False

        if _segment_body_hits_border(segment, other_center, other_radius):
            return False
        if _segment_body_hits_border(other["segment"], center, radius):
            return False

        # Keep segments visually outside each other's square interiors.
        if _whole_square_contains_segment(segment, other_center, other_radius):
            return False
        if _whole_square_contains_segment(other["segment"], center, radius):
            return False

    return True


def _attempt_layout() -> tuple[int, int, list[dict]] | None:
    height = random.randint(*GEN_HEIGHT_RANGE)
    width = random.randint(*GEN_WIDTH_RANGE)
    count = random.randint(*GEN_SEGMENT_COUNT_RANGE)

    want_overlap = random.random() < 0.45
    want_equal_radius_pair = random.random() < 0.65
    lengths = _length_choices(count, require_equal_radius_pair=want_equal_radius_pair)

    if want_overlap and max(lengths) < 4:
        lengths[random.randrange(count)] = random.choice((4, 5, 6))
        if len(set(lengths)) < 2:
            lengths[(lengths.index(max(lengths)) + 1) % count] = 2

    order = sorted(range(count), key=lambda idx: lengths[idx], reverse=True)
    placed: list[dict] = []
    for idx in order:
        length = lengths[idx]
        radius = length // 2
        candidates = _center_candidates(height, width, length)
        random.shuffle(candidates)
        found = False
        for center in candidates:
            candidate = {
                "length": length,
                "radius": radius,
                "center": center,
                "segment": _segment_from_center(center, length),
            }
            if _is_valid_placement(candidate, placed, height, width):
                placed.append(candidate)
                found = True
                break
        if not found:
            return None

    if want_overlap:
        overlap_count = 0
        for i in range(len(placed)):
            for j in range(i + 1, len(placed)):
                if _boxes_overlap(placed[i]["center"], placed[i]["radius"], placed[j]["center"], placed[j]["radius"]):
                    overlap_count += 1
        if overlap_count == 0:
            return None

    if len({item["length"] for item in placed}) < 2:
        return None

    if max(item["radius"] for item in placed) < 2:
        return None

    return height, width, placed


def _render_input(height: int, width: int, placed: list[dict]) -> np.ndarray:
    grid = np.full((height, width), BLACK, dtype=int)
    for item in placed:
        for r, c in item["segment"]:
            grid[r, c] = RED
    return grid


def _green_only(grid: np.ndarray) -> np.ndarray:
    return (grid == GREEN).astype(np.uint8)


def _validate_instance(inp: np.ndarray, placed: list[dict]) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if not np.all(np.isin(inp, [BLACK, RED])):
        return False

    detected = _find_red_segments(inp)
    detected_signatures = sorted(tuple(seg) for seg in detected)
    placed_signatures = sorted(tuple(item["segment"]) for item in placed)
    if detected_signatures != placed_signatures:
        return False

    lengths = [len(seg) for seg in detected]
    if not (GEN_SEGMENT_COUNT_RANGE[0] <= len(lengths) <= GEN_SEGMENT_COUNT_RANGE[1]):
        return False
    if min(lengths) < 2 or max(lengths) > 6 or len(set(lengths)) < 2:
        return False

    out = main(inp)
    if out.shape != inp.shape:
        return False
    if not np.all(np.isin(out, [BLACK, RED, GREEN, YELLOW])):
        return False

    # Independent semantic checks for recoloring and box drawing.
    rebuilt = np.full(inp.shape, BLACK, dtype=int)
    specs = []
    for seg in detected:
        length = len(seg)
        center = seg[length // 2]
        radius = length // 2
        specs.append((radius, center, seg))

    for radius, center, _seg in sorted(specs, key=lambda item: (item[0], -item[1][0], -item[1][1])):
        _paint_border(rebuilt, center, radius, GREEN)
    for _radius, _center, seg in specs:
        for idx, (r, c) in enumerate(seg):
            rebuilt[r, c] = YELLOW if idx % 2 == 1 else RED

    if not np.array_equal(out, rebuilt):
        return False

    # Ensure green structure is not degenerate and depends on the actual segment lengths.
    green_mask = _green_only(out)
    if green_mask.sum() == 0:
        return False
    if np.array_equal(green_mask, _green_only(main(np.where(inp == RED, BLACK, inp)))):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        layout = _attempt_layout()
        if layout is None:
            continue
        height, width, placed = layout
        inp = _render_input(height, width, placed)
        if _validate_instance(inp, placed):
            return inp
    raise RuntimeError("generate_input: failed to build a valid puzzle instance")
