from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

GRID_ROWS_RANGE = (13, 18)
GRID_COLS_RANGE = (13, 18)
FRAME_COUNT_CHOICES = (1, 2)
FRAME_COUNT_WEIGHTS = (1, 3)
FRAME_SIDE_RANGE = (5, 9)  # inclusive-exclusive for randint helper below
FRAME_GAP = 2
OUTSIDE_GAP = 1
EXTRA_EDGE_MARKERS_PROB = 0.6
GEN_ATTEMPTS = 400


def _randint(lo: int, hi: int) -> int:
    return int(np.random.randint(lo, hi + 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _ring_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _span_fill_red(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    red = g == RED
    h, w = g.shape

    for r in range(h):
        cols = np.flatnonzero(red[r])
        if len(cols) >= 2:
            out[r, cols[0] : cols[-1] + 1] = RED

    for c in range(w):
        rows = np.flatnonzero(red[:, c])
        if len(rows) >= 2:
            out[rows[0] : rows[-1] + 1, c] = RED

    return out


def _find_rectangular_frames(grid_after_fill: np.ndarray) -> list[tuple[int, int, int, int]]:
    red = grid_after_fill == RED
    frames: list[tuple[int, int, int, int]] = []
    for coords in _components(red):
        if coords.shape[0] < 8:
            continue
        box = _bbox_coords(coords)
        r0, c0, r1, c1 = box
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        component_mask = np.zeros_like(red, dtype=bool)
        component_mask[coords[:, 0], coords[:, 1]] = True
        if np.array_equal(component_mask, _ring_mask(red.shape, box)):
            frames.append(box)
    assert frames, "expected at least one rectangular red frame after span fill"
    return sorted(frames)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Span-fill red markers into emergent rectangular frames, then swap blue and green only inside those frames.

    Input structure:
    - The grid has a black background, scattered red marker pixels, and blue/green pixels both inside and outside the future frames.
    - The red pixels are sparse markers rather than complete outlines; rows or columns with multiple red markers define the missing red spans.
    - After span filling, the red structure forms one or more one-cell-thick rectangular frames.

    Rule:
    - Using only the original red markers, fill each qualifying row from its leftmost red marker to its rightmost red marker.
    - Using only the original red markers, fill each qualifying column from its topmost red marker to its bottommost red marker.
    - Detect every rectangular red frame in the filled result, take each frame's strict interior, and swap blue with green there.
    - Preserve black interior cells, preserve the red frames, and leave all blue/green pixels outside the frame interiors unchanged.

    Key insight:
    - The editable regions are not explicit in the input: they appear only after the red marker spans are completed into rectangular rings.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    spanned = _span_fill_red(g)
    out = spanned.copy()

    for r0, c0, r1, c1 in _find_rectangular_frames(spanned):
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        region = out[r0 + 1 : r1, c0 + 1 : c1]
        region[interior == BLUE] = GREEN
        region[interior == GREEN] = BLUE

    return out


def _boxes_share_boundary(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] in (b[0], b[2]) or a[2] in (b[0], b[2]) or a[1] in (b[1], b[3]) or a[3] in (b[1], b[3])


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    return not (
        a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1]
    )


def _sample_boxes(h: int, w: int, frame_count: int) -> list[tuple[int, int, int, int]] | None:
    for _ in range(120):
        boxes: list[tuple[int, int, int, int]] = []
        for _frame_idx in range(frame_count):
            fh = _randint(FRAME_SIDE_RANGE[0], FRAME_SIDE_RANGE[1])
            fw = _randint(FRAME_SIDE_RANGE[0], FRAME_SIDE_RANGE[1])
            if fh >= h - 1 or fw >= w - 1:
                break
            r0 = _randint(1, h - fh - 1)
            c0 = _randint(1, w - fw - 1)
            box = (r0, c0, r0 + fh - 1, c0 + fw - 1)
            if any(_boxes_share_boundary(box, other) or _boxes_too_close(box, other, FRAME_GAP) for other in boxes):
                break
            boxes.append(box)
        if len(boxes) == frame_count:
            return sorted(boxes)
    return None


def _place_interior_colors(grid: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    n = inner_h * inner_w
    black_count = _randint(max(1, n // 5), max(1, n // 2))
    remaining = n - black_count
    blue_count = _randint(2, remaining - 2)
    green_count = remaining - blue_count
    if green_count < 2:
        green_count = 2
        blue_count = remaining - green_count
    vals = [BLACK] * black_count + [BLUE] * blue_count + [GREEN] * green_count
    np.random.shuffle(vals)
    grid[r0 + 1 : r1, c0 + 1 : c1] = np.array(vals, dtype=int).reshape(inner_h, inner_w)


def _outside_positions(h: int, w: int, boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            inside_frame = False
            on_border = False
            for r0, c0, r1, c1 in boxes:
                if r0 < r < r1 and c0 < c < c1:
                    inside_frame = True
                    break
                if r0 <= r <= r1 and c0 <= c <= c1:
                    on_border = True
            if not inside_frame and not on_border:
                positions.append((r, c))
    return positions


def _between_box_positions(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int]]:
    if len(boxes) != 2:
        return []
    union_r0 = min(box[0] for box in boxes)
    union_c0 = min(box[1] for box in boxes)
    union_r1 = max(box[2] for box in boxes)
    union_c1 = max(box[3] for box in boxes)
    positions: list[tuple[int, int]] = []
    for r in range(union_r0, union_r1 + 1):
        for c in range(union_c0, union_c1 + 1):
            if any(r0 <= r <= r1 and c0 <= c <= c1 for r0, c0, r1, c1 in boxes):
                continue
            positions.append((r, c))
    return positions


def _scatter_outside_colors(grid: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> bool:
    h, w = grid.shape
    outside = _outside_positions(h, w, boxes)
    if len(outside) < 8:
        return False

    targeted = _between_box_positions(boxes)
    chosen: list[tuple[int, int]] = []
    if targeted:
        np.random.shuffle(targeted)
        chosen.extend(targeted[: min(4, len(targeted))])

    remaining_pool = [pos for pos in outside if pos not in set(chosen)]
    np.random.shuffle(remaining_pool)
    target_total = _randint(8, min(18, len(outside)))
    needed_more = max(0, target_total - len(chosen))
    chosen.extend(remaining_pool[:needed_more])
    if len(chosen) < 4:
        return False

    colors = [BLUE, GREEN] * ((len(chosen) + 1) // 2)
    colors = colors[: len(chosen)]
    np.random.shuffle(colors)
    for (r, c), color in zip(chosen, colors):
        grid[r, c] = color

    return np.count_nonzero(grid == BLUE) > 0 and np.count_nonzero(grid == GREEN) > 0


def _input_markers_for_box(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    markers = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]

    if random.random() < EXTRA_EDGE_MARKERS_PROB and c1 - c0 >= 4:
        markers.append((r0, _randint(c0 + 1, c1 - 1)))
    if random.random() < EXTRA_EDGE_MARKERS_PROB and c1 - c0 >= 4:
        markers.append((r1, _randint(c0 + 1, c1 - 1)))
    if random.random() < EXTRA_EDGE_MARKERS_PROB and r1 - r0 >= 4:
        markers.append((_randint(r0 + 1, r1 - 1), c0))
    if random.random() < EXTRA_EDGE_MARKERS_PROB and r1 - r0 >= 4:
        markers.append((_randint(r0 + 1, r1 - 1), c1))

    dedup: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for coord in markers:
        if coord not in seen:
            seen.add(coord)
            dedup.append(coord)
    return dedup


def _build_input_from_boxes(output_like: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    inp = output_like.copy()
    inp[inp == RED] = BLACK
    for box in boxes:
        for r, c in _input_markers_for_box(box):
            inp[r, c] = RED
    return inp


def _validate_example(inp: np.ndarray) -> None:
    out = main(inp)
    spanned = _span_fill_red(inp)
    frames = _find_rectangular_frames(spanned)
    assert len(frames) in FRAME_COUNT_CHOICES
    assert np.count_nonzero(inp == RED) < np.count_nonzero(spanned == RED), "span fill must add new red cells"

    changed_outside = np.zeros(inp.shape, dtype=bool)
    interior_mask = np.zeros(inp.shape, dtype=bool)
    for r0, c0, r1, c1 in frames:
        interior_mask[r0 + 1 : r1, c0 + 1 : c1] = True
        interior = inp[r0 + 1 : r1, c0 + 1 : c1]
        swapped = out[r0 + 1 : r1, c0 + 1 : c1]
        assert np.count_nonzero(interior == BLUE) >= 2
        assert np.count_nonzero(interior == GREEN) >= 2
        assert np.any(interior == BLACK)
        assert np.array_equal(swapped[interior == BLACK], interior[interior == BLACK])
        assert np.all(swapped[interior == BLUE] == GREEN)
        assert np.all(swapped[interior == GREEN] == BLUE)
        changed_outside[r0 + 1 : r1, c0 + 1 : c1] = True

    outside_mask = ~interior_mask
    assert np.any(inp[outside_mask] == BLUE)
    assert np.any(inp[outside_mask] == GREEN)
    assert np.array_equal(out[outside_mask & (inp == BLUE)], inp[outside_mask & (inp == BLUE)])
    assert np.array_equal(out[outside_mask & (inp == GREEN)], inp[outside_mask & (inp == GREEN)])
    assert np.array_equal(out[spanned == RED], spanned[spanned == RED])

    if len(frames) == 2:
        between = _between_box_positions(frames)
        assert any(inp[r, c] in (BLUE, GREEN) for r, c in between), "need distractors between the two frames"


def _gen_attempt() -> np.ndarray | None:
    h = _randint(*GRID_ROWS_RANGE)
    w = _randint(*GRID_COLS_RANGE)
    frame_count = random.choices(FRAME_COUNT_CHOICES, weights=FRAME_COUNT_WEIGHTS, k=1)[0]
    boxes = _sample_boxes(h, w, frame_count)
    if boxes is None:
        return None

    out_like = np.zeros((h, w), dtype=int)
    for box in boxes:
        out_like[_ring_mask((h, w), box)] = RED
        _place_interior_colors(out_like, box)

    if not _scatter_outside_colors(out_like, boxes):
        return None

    inp = _build_input_from_boxes(out_like, boxes)

    try:
        _validate_example(inp)
    except Exception:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 6")
