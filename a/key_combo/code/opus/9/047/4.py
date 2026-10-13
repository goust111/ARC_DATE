from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5
ORANGE = 7

GEN_ATTEMPTS = 500
GRID_MIN = 15
GRID_MAX = 24
FRAME_MIN = 6
FRAME_MAX = 9
CANVAS_MARGIN = 1
FRAME_GAP = 1

BBox = tuple[int, int, int, int]
Coord = tuple[int, int]


def _neighbors4(r: int, c: int) -> Iterable[Coord]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox_from_coords(coords: list[Coord]) -> BBox:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _find_color_components(grid: np.ndarray, color: int) -> list[list[Coord]]:
    height, width = map(int, grid.shape)
    seen = np.zeros((height, width), dtype=bool)
    components: list[list[Coord]] = []
    for r in range(height):
        for c in range(width):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _frame_bbox_from_component(grid: np.ndarray, coords: list[Coord], color: int) -> BBox | None:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None
    border = _border_mask(height, width)
    comp_mask = np.zeros((height, width), dtype=bool)
    for r, c in coords:
        comp_mask[r - r0, c - c0] = True
    if not np.array_equal(comp_mask, border):
        return None
    if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1][border] == color):
        return None
    return r0, c0, r1, c1


def _find_frame_bboxes(grid: np.ndarray, color: int) -> list[BBox]:
    frames: list[BBox] = []
    for coords in _find_color_components(grid, color):
        bbox = _frame_bbox_from_component(grid, coords, color)
        if bbox is not None:
            frames.append(bbox)
    return sorted(frames)


def _control_category(control: Coord, inner_h: int, inner_w: int) -> str:
    r, c = control
    if r == 0 or c == 0 or r == inner_h - 1 or c == inner_w - 1:
        return "edge"
    return "inner"


def _filled_interior(inner_h: int, inner_w: int, control: Coord) -> np.ndarray:
    mr, mc = control
    interior = np.full((inner_h, inner_w), ORANGE, dtype=int)
    for r in range(inner_h):
        for c in range(inner_w):
            distance = max(abs(r - mr), abs(c - mc))
            if distance == 0:
                interior[r, c] = RED
            elif distance % 2 == 1:
                interior[r, c] = YELLOW
            else:
                interior[r, c] = ORANGE
    return interior


def _build_source_patch(frame_h: int, frame_w: int, control: Coord) -> np.ndarray:
    patch = np.full((frame_h, frame_w), BLUE, dtype=int)
    patch[1:-1, 1:-1] = _filled_interior(frame_h - 2, frame_w - 2, control)
    return patch


def _build_target_patch(frame_h: int, frame_w: int, control: Coord) -> np.ndarray:
    patch = np.full((frame_h, frame_w), BLACK, dtype=int)
    patch[_border_mask(frame_h, frame_w)] = GREY
    patch[1 + control[0], 1 + control[1]] = RED
    return patch


def _validate_source_interior(grid: np.ndarray, bbox: BBox) -> Coord:
    r0, c0, r1, c1 = bbox
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    markers = np.argwhere(interior == RED)
    assert markers.shape[0] == 1, "source frame must contain exactly one red control pixel"
    assert np.all(np.isin(interior, [RED, YELLOW, ORANGE])), "source interior must be the demonstrated ring pattern"
    mr, mc = map(int, markers[0])
    expected = _filled_interior(interior.shape[0], interior.shape[1], (mr, mc))
    assert np.array_equal(interior, expected), "source interior must exactly demonstrate the ring rule"
    return r0 + 1 + mr, c0 + 1 + mc


def _validate_target_interior(grid: np.ndarray, bbox: BBox) -> Coord:
    r0, c0, r1, c1 = bbox
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    markers = np.argwhere(interior == RED)
    assert markers.shape[0] == 1, "each target frame must contain exactly one red control pixel"
    assert np.all(np.isin(interior, [BLACK, RED])), "target interiors must be blank except for their control pixel"
    mr, mc = map(int, markers[0])
    return r0 + 1 + mr, c0 + 1 + mc


def _far_enough(a: BBox, b: BBox, pad: int = FRAME_GAP) -> bool:
    return (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _parse_grid(grid: np.ndarray) -> tuple[BBox, Coord, list[BBox], list[Coord]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    source_frames = _find_frame_bboxes(g, BLUE)
    assert len(source_frames) == 1, "expected exactly one blue source frame"
    source_bbox = source_frames[0]
    source_control = _validate_source_interior(g, source_bbox)
    source_h = source_bbox[2] - source_bbox[0] + 1
    source_w = source_bbox[3] - source_bbox[1] + 1

    grey_frames = _find_frame_bboxes(g, GREY)
    target_bboxes = [
        bbox
        for bbox in grey_frames
        if (bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1) == (source_h, source_w)
    ]
    assert len(target_bboxes) in (1, 2), "expected one or two same-size grey target frames"
    assert len(target_bboxes) == len(grey_frames), "all grey frames must match the source size"

    target_controls = [_validate_target_interior(g, bbox) for bbox in target_bboxes]

    all_bboxes = [source_bbox, *target_bboxes]
    for idx, bbox_a in enumerate(all_bboxes):
        for bbox_b in all_bboxes[idx + 1 :]:
            assert _far_enough(bbox_a, bbox_b), "frames must not overlap or touch"

    return source_bbox, source_control, target_bboxes, target_controls


def _paint_target(output: np.ndarray, bbox: BBox, control: Coord) -> None:
    r0, c0, r1, c1 = bbox
    frame_h = r1 - r0 + 1
    frame_w = c1 - c0 + 1
    output[r0 : r1 + 1, c0 : c1 + 1][_border_mask(frame_h, frame_w)] = BLUE

    inner_r = control[0] - (r0 + 1)
    inner_c = control[1] - (c0 + 1)
    fill = _filled_interior(frame_h - 2, frame_w - 2, (inner_r, inner_c))
    output[r0 + 1 : r1, c0 + 1 : c1] = fill


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every matching grey frame to the source border color and copy the source's concentric ring rule into each target interior.

    Input structure:
    - The background is black.
    - Exactly one blue 1-pixel-thick rectangular frame contains a completed demonstration: one red control pixel and yellow/orange Chebyshev rings filling the whole interior.
    - One or two grey 1-pixel-thick rectangular frames have the same outer size as the blue frame; each target interior is otherwise black except for one red control pixel.

    Rule:
    - Identify the unique blue source frame and read both of its clues: the border must stay blue, and the interior demonstrates a parity-based Chebyshev ring fill centered on the red control pixel.
    - Find every grey frame with the same outer dimensions as the source.
    - For each target, repaint its border blue and fill every interior black cell by Chebyshev distance to that target's own red control: odd distances become yellow, even positive distances become orange, and the red control stays red.
    - Leave the source frame and all background pixels unchanged.

    Key insight:
    - The source frame is a live template with two independent messages at once: its border color tells you how to recolor matching frames, while its completed interior tells you how to fill each target around its own control pixel.
    """
    g = np.asarray(grid, dtype=int)
    _source_bbox, _source_control, target_bboxes, target_controls = _parse_grid(g)

    out = g.copy()
    for bbox, control in zip(target_bboxes, target_controls):
        _paint_target(out, bbox, control)
    return out


def _sample_controls(inner_h: int, inner_w: int, target_count: int) -> tuple[Coord, list[Coord]] | None:
    all_controls = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    if len(all_controls) < target_count + 1:
        return None

    for _ in range(80):
        source = random.choice(all_controls)
        remaining = [pos for pos in all_controls if pos != source]

        first_pool = [
            pos
            for pos in remaining
            if _control_category(pos, inner_h, inner_w) != _control_category(source, inner_h, inner_w)
        ]
        if not first_pool:
            first_pool = remaining
        target_controls = [random.choice(first_pool)]

        if target_count == 2:
            remaining2 = [pos for pos in remaining if pos != target_controls[0]]
            second_pool = [
                pos
                for pos in remaining2
                if _control_category(pos, inner_h, inner_w) != _control_category(target_controls[0], inner_h, inner_w)
            ]
            if not second_pool:
                second_pool = remaining2
            target_controls.append(random.choice(second_pool))

        if len({source, *target_controls}) != target_count + 1:
            continue
        if len({_control_category(pos, inner_h, inner_w) for pos in [source, *target_controls]}) == 1:
            continue
        return source, target_controls
    return None


def _sample_bboxes(canvas_h: int, canvas_w: int, frame_h: int, frame_w: int, count: int) -> list[BBox] | None:
    rows = list(range(CANVAS_MARGIN, canvas_h - frame_h - CANVAS_MARGIN + 1))
    cols = list(range(CANVAS_MARGIN, canvas_w - frame_w - CANVAS_MARGIN + 1))
    if not rows or not cols:
        return None

    positions = [(r, c) for r in rows for c in cols]
    for _ in range(80):
        random.shuffle(positions)
        bboxes: list[BBox] = []
        for r0, c0 in positions:
            bbox = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)
            if all(_far_enough(bbox, other) for other in bboxes):
                bboxes.append(bbox)
                if len(bboxes) == count:
                    return bboxes
    return None


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _validate_generated_case(inp: np.ndarray, out: np.ndarray) -> None:
    source_bbox, source_control, target_bboxes, target_controls = _parse_grid(inp)
    solved = main(inp)
    assert np.array_equal(solved, out), "internal validation must match solver output"
    assert np.array_equal(inp[source_bbox[0] : source_bbox[2] + 1, source_bbox[1] : source_bbox[3] + 1], out[source_bbox[0] : source_bbox[2] + 1, source_bbox[1] : source_bbox[3] + 1]), "source frame must stay unchanged"
    assert len({source_control, *target_controls}) == 1 + len(target_controls), "controls should be distinct to prevent copy shortcuts"

    frame_h = source_bbox[2] - source_bbox[0] + 1
    frame_w = source_bbox[3] - source_bbox[1] + 1
    border = _border_mask(frame_h, frame_w)
    transformed_crops: list[np.ndarray] = []
    for bbox, control in zip(target_bboxes, target_controls):
        r0, c0, r1, c1 = bbox
        before = inp[r0 : r1 + 1, c0 : c1 + 1]
        after = out[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(before[border] == GREY), "targets must begin grey"
        assert np.all(after[border] == BLUE), "target borders must recolor to blue"
        assert np.any(before[~border] == BLACK), "targets must contain blank pixels to fill"
        assert np.any(after[~border] != before[~border]), "interior fill must change target interiors"
        assert tuple(np.argwhere(after == RED)[0]) == (control[0] - r0, control[1] - c0), "control must stay fixed"
        transformed_crops.append(after.copy())

    if len(transformed_crops) == 2:
        assert not np.array_equal(transformed_crops[0], transformed_crops[1]), "different targets should not collapse to the same transformed crop"


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        canvas_h = random.randint(GRID_MIN, GRID_MAX)
        canvas_w = random.randint(GRID_MIN, GRID_MAX)
        target_count = random.choice([1, 2])

        max_h = min(FRAME_MAX, canvas_h - 2 * CANVAS_MARGIN)
        max_w = min(FRAME_MAX, canvas_w - 2 * CANVAS_MARGIN)
        if max_h < FRAME_MIN or max_w < FRAME_MIN:
            continue

        frame_h = random.randint(FRAME_MIN, max_h)
        frame_w = random.randint(FRAME_MIN, max_w)
        inner_h = frame_h - 2
        inner_w = frame_w - 2

        bboxes = _sample_bboxes(canvas_h, canvas_w, frame_h, frame_w, target_count + 1)
        if bboxes is None:
            continue

        controls = _sample_controls(inner_h, inner_w, target_count)
        if controls is None:
            continue
        source_control, target_controls = controls

        grid = np.zeros((canvas_h, canvas_w), dtype=int)
        random.shuffle(bboxes)
        source_bbox = bboxes[0]
        target_bboxes = bboxes[1:]

        source_patch = _build_source_patch(frame_h, frame_w, source_control)
        _paste(grid, source_patch, source_bbox[0], source_bbox[1])

        for bbox, control in zip(target_bboxes, target_controls):
            target_patch = _build_target_patch(frame_h, frame_w, control)
            _paste(grid, target_patch, bbox[0], bbox[1])

        out = main(grid)
        _validate_generated_case(grid, out)
        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
