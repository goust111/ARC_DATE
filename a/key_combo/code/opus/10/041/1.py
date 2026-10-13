from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _compressed(values: list[int]) -> list[int]:
    out: list[int] = []
    for value in values:
        if not out or out[-1] != value:
            out.append(int(value))
    return out


def _is_rectangular_perimeter(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if len(coords) != 2 * h + 2 * w - 4:
        return False
    on_border = (
        (coords[:, 0] == r0)
        | (coords[:, 0] == r1)
        | (coords[:, 1] == c0)
        | (coords[:, 1] == c1)
    )
    return bool(np.all(on_border))


def _strict_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = outer
    rr0, cc0, rr1, cc1 = inner
    return r0 < rr0 and c0 < cc0 and rr1 < r1 and cc1 < c1


def _extract_band_sequence(g: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], list[int]]:
    candidates: list[tuple[int, int, int, int]] = []
    for coords in _components(g != bg):
        r0, c0, r1, c1 = _bbox(coords)
        if c0 != c1:
            continue
        expected_rows = np.arange(r0, r1 + 1)
        if len(coords) != len(expected_rows):
            continue
        if not np.array_equal(np.sort(coords[:, 0]), expected_rows):
            continue
        candidates.append((r0, c0, r1, c1))
    assert len(candidates) == 1, "expected a unique vertical color band"
    r0, c0, r1, c1 = candidates[0]
    seq = _compressed([int(g[r, c0]) for r in range(r0, r1 + 1)])
    assert len(seq) >= 2
    return candidates[0], seq


def _find_frames(
    g: np.ndarray, bg: int, band_bbox: tuple[int, int, int, int]
) -> list[dict[str, object]]:
    work = g.copy()
    r0, c0, r1, c1 = band_bbox
    work[r0 : r1 + 1, c0 : c1 + 1] = bg

    frames: list[dict[str, object]] = []
    for color in sorted(set(map(int, work.ravel())) - {bg}):
        comps = _components(work == color)
        assert len(comps) == 1, "each frame color must form one component"
        coords = comps[0]
        assert _is_rectangular_perimeter(coords), "foreground component must be a hollow rectangle"
        frames.append({"color": int(color), "coords": coords, "bbox": _bbox(coords)})

    assert len(frames) >= 2
    return frames


def _frame_depths(frames: list[dict[str, object]]) -> dict[int, int]:
    depths: dict[int, int] = {}
    boxes = [frame["bbox"] for frame in frames]
    for i, box in enumerate(boxes):
        depth = 1 + sum(_strict_contains(other, box) for j, other in enumerate(boxes) if i != j)
        depths[i] = int(depth)

    expected = list(range(1, len(frames) + 1))
    assert sorted(depths.values()) == expected, "frames must form one strict nesting chain"
    return depths


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint each nested frame's background chamber using the color-band entry at the same nesting depth.

    Input structure:
    - The dominant background is black, with one narrow vertical color band on the right.
    - The left side contains 2-4 distinct monochrome rectangular perimeters nested strictly inside each other.
    - The band is a single 1-pixel-wide connected column whose consecutive same-color runs encode an ordered sequence.

    Rule:
    - Compress the right-side band top-to-bottom by keeping one entry per maximal same-color run.
    - Compute each rectangle's strict-containment depth: outermost is depth 1, the next is depth 2, and so on.
    - For every frame, fill only the background cells inside that frame with the compressed band color at the same depth, while preserving all frame borders and leaving the band unchanged.

    Key insight:
    - The color band is a lookup table, while the nested geometry supplies the lookup index; neither analysis alone determines the repaint color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    band_bbox, seq = _extract_band_sequence(g, bg)
    frames = _find_frames(g, bg, band_bbox)
    depths = _frame_depths(frames)
    assert len(seq) == len(frames), "band length must match frame depth count"

    out = g.copy()
    for i, frame in sorted(enumerate(frames), key=lambda item: depths[item[0]], reverse=True):
        r0, c0, r1, c1 = frame["bbox"]
        color = seq[depths[i] - 1]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[interior == bg] = color
    return out


_BG = 0
_DEPTH_RANGE = (2, 4)
_INNER_SIDE_RANGE = (3, 5)
_EXPAND_RANGE = (2, 3)
_FRAME_MARGIN_RANGE = (1, 4)
_STRIP_MARGIN_RANGE = (1, 4)
_STRIP_GAP_RANGE = (2, 5)
_STRIP_SEGMENT_HEIGHT_RANGE = (1, 3)
_MAX_ATTEMPTS = 500


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _build_boxes(depth: int) -> list[tuple[int, int, int, int]]:
    inner_h = random.randint(*_INNER_SIDE_RANGE)
    inner_w = random.randint(*_INNER_SIDE_RANGE)
    boxes = [(0, 0, inner_h - 1, inner_w - 1)]
    for _ in range(depth - 1):
        cr0, cc0, cr1, cc1 = boxes[-1]
        top = random.randint(*_EXPAND_RANGE)
        bottom = random.randint(*_EXPAND_RANGE)
        left = random.randint(*_EXPAND_RANGE)
        right = random.randint(*_EXPAND_RANGE)
        boxes.append((cr0 - top, cc0 - left, cr1 + bottom, cc1 + right))

    or0, oc0, _, _ = boxes[-1]
    shift_r = -or0
    shift_c = -oc0
    normalized = [
        (r0 + shift_r, c0 + shift_c, r1 + shift_r, c1 + shift_c) for r0, c0, r1, c1 in boxes
    ]
    return list(reversed(normalized))


def _expected_output(
    grid: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    strip_bbox: tuple[int, int, int, int],
    seq: list[int],
) -> np.ndarray:
    out = grid.copy()
    for i, box in enumerate(boxes):
        child = boxes[i + 1] if i + 1 < len(boxes) else None
        r0, c0, r1, c1 = box
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if child is not None:
                    cr0, cc0, cr1, cc1 = child
                    if cr0 <= r <= cr1 and cc0 <= c <= cc1:
                        continue
                if out[r, c] == _BG:
                    out[r, c] = seq[i]

    sr0, sc0, sr1, sc1 = strip_bbox
    out[sr0 : sr1 + 1, sc0 : sc1 + 1] = grid[sr0 : sr1 + 1, sc0 : sc1 + 1]
    return out


def _validate_generated(
    grid: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    strip_bbox: tuple[int, int, int, int],
    seq: list[int],
    frame_colors: list[int],
) -> bool:
    counts = Counter(map(int, grid.ravel()))
    if counts[_BG] != max(counts.values()):
        return False

    try:
        found_strip_bbox, found_seq = _extract_band_sequence(grid, _BG)
        found_frames = _find_frames(grid, _BG, found_strip_bbox)
        found_depths = _frame_depths(found_frames)
    except AssertionError:
        return False

    if found_strip_bbox != strip_bbox or found_seq != seq:
        return False

    found_boxes = [frame["bbox"] for frame in found_frames]
    if set(found_boxes) != set(boxes):
        return False

    ordered_boxes = [
        frame["bbox"]
        for _, frame in sorted(enumerate(found_frames), key=lambda item: found_depths[item[0]])
    ]
    if ordered_boxes != boxes:
        return False

    if sorted(int(frame["color"]) for frame in found_frames) != sorted(frame_colors):
        return False

    expected = _expected_output(grid, boxes, strip_bbox, seq)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    sr0, sc0, sr1, sc1 = strip_bbox
    if not np.array_equal(actual[sr0 : sr1 + 1, sc0 : sc1 + 1], grid[sr0 : sr1 + 1, sc0 : sc1 + 1]):
        return False

    for i, box in enumerate(boxes):
        child = boxes[i + 1] if i + 1 < len(boxes) else None
        region: list[tuple[int, int]] = []
        r0, c0, r1, c1 = box
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if child is not None:
                    cr0, cc0, cr1, cc1 = child
                    if cr0 <= r <= cr1 and cc0 <= c <= cc1:
                        continue
                region.append((r, c))
        if not region:
            return False
        if not all(int(actual[r, c]) == seq[i] for r, c in region):
            return False

    for color in frame_colors:
        if np.any(actual[grid == color] != color):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    depth = random.randint(*_DEPTH_RANGE)
    boxes = _build_boxes(depth)
    outer_r0, outer_c0, outer_r1, outer_c1 = boxes[0]
    outer_h = outer_r1 - outer_r0 + 1
    outer_w = outer_c1 - outer_c0 + 1

    strip_heights = [random.randint(*_STRIP_SEGMENT_HEIGHT_RANGE) for _ in range(depth)]
    strip_h = sum(strip_heights)

    frame_top = random.randint(*_FRAME_MARGIN_RANGE)
    frame_left = random.randint(*_FRAME_MARGIN_RANGE)
    frame_bottom = random.randint(*_FRAME_MARGIN_RANGE)
    strip_top = random.randint(*_STRIP_MARGIN_RANGE)
    strip_bottom = random.randint(*_STRIP_MARGIN_RANGE)
    strip_gap = random.randint(*_STRIP_GAP_RANGE)
    right_margin = random.randint(*_STRIP_MARGIN_RANGE)

    height = max(frame_top + outer_h + frame_bottom, strip_top + strip_h + strip_bottom)
    width = frame_left + outer_w + strip_gap + 1 + right_margin
    if height > 30 or width > 30:
        return None

    palette = random.sample(list(range(1, 10)), 2 * depth)
    frame_colors = palette[:depth]
    strip_colors = palette[depth:]

    grid = np.zeros((height, width), dtype=int)

    placed_boxes: list[tuple[int, int, int, int]] = []
    for color, (r0, c0, r1, c1) in zip(frame_colors, boxes):
        shifted = (r0 + frame_top, c0 + frame_left, r1 + frame_top, c1 + frame_left)
        placed_boxes.append(shifted)
        _draw_frame(grid, shifted, color)

    strip_col = frame_left + outer_w + strip_gap
    strip_row = strip_top
    cursor = strip_row
    for color, seg_h in zip(strip_colors, strip_heights):
        grid[cursor : cursor + seg_h, strip_col] = color
        cursor += seg_h
    strip_bbox = (strip_row, strip_col, cursor - 1, strip_col)

    if not _validate_generated(grid, placed_boxes, strip_bbox, strip_colors, frame_colors):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-depth color-band input")
