from __future__ import annotations

import itertools
from collections import Counter

import numpy as np


Color = int
BBox = tuple[int, int, int, int]
Coord = tuple[int, int]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, grid.flatten()))
    if exclude is not None:
        colors -= set(exclude)
    return sorted(colors)


def bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int) -> tuple[Coord, ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def clipped_frame_mask(shape: tuple[int, int], box: BBox) -> np.ndarray:
    h, w = shape
    r0, c0, r1, c1 = box
    mask = np.zeros(shape, dtype=bool)
    if r0 > r1 or c0 > c1:
        return mask

    clip_c0 = max(0, c0)
    clip_c1 = min(w - 1, c1)
    if clip_c0 <= clip_c1:
        if 0 <= r0 < h:
            mask[r0, clip_c0 : clip_c1 + 1] = True
        if 0 <= r1 < h:
            mask[r1, clip_c0 : clip_c1 + 1] = True

    clip_r0 = max(0, r0)
    clip_r1 = min(h - 1, r1)
    if clip_r0 <= clip_r1:
        if 0 <= c0 < w:
            mask[clip_r0 : clip_r1 + 1, c0] = True
        if 0 <= c1 < w:
            mask[clip_r0 : clip_r1 + 1, c1] = True
    return mask


def visible_interior_mask(shape: tuple[int, int], box: BBox) -> np.ndarray:
    h, w = shape
    r0, c0, r1, c1 = box
    mask = np.zeros(shape, dtype=bool)
    rr0 = max(0, r0 + 1)
    rr1 = min(h - 1, r1 - 1)
    cc0 = max(0, c0 + 1)
    cc1 = min(w - 1, c1 - 1)
    if rr0 <= rr1 and cc0 <= cc1:
        mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    return mask


def is_full_rect_frame(shape: tuple[int, int], coords: np.ndarray) -> bool:
    box = bbox_coords(coords)
    r0, c0, r1, c1 = box
    if r0 < 0 or c0 < 0 or r1 >= shape[0] or c1 >= shape[1]:
        return False
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return False
    actual = np.zeros(shape, dtype=bool)
    actual[coords[:, 0], coords[:, 1]] = True
    expected = clipped_frame_mask(shape, box)
    return np.array_equal(actual, expected)


def add_box(box: BBox, step: BBox, sign: int = 1) -> BBox:
    return tuple(int(box[i] + sign * step[i]) for i in range(4))


def diff_box(a: BBox, b: BBox) -> BBox:
    return tuple(int(b[i] - a[i]) for i in range(4))


def order_arithmetic_boxes(boxes: list[BBox]) -> tuple[list[BBox], BBox]:
    assert len(boxes) >= 2
    for perm in itertools.permutations(range(len(boxes))):
        seq = [boxes[i] for i in perm]
        step = diff_box(seq[0], seq[1])
        if step == (0, 0, 0, 0):
            continue
        if all(diff_box(seq[i], seq[i + 1]) == step for i in range(len(seq) - 1)):
            return seq, step
    raise AssertionError("frame boxes do not form a single arithmetic sequence")


def _identify_roles(grid: np.ndarray) -> tuple[int, int, list[BBox]]:
    bg = get_bg(grid)
    assert bg == 0, "expected black background"
    fg_colors = colors_present(grid, exclude={bg})
    assert len(fg_colors) == 2, "expected exactly two foreground colors"

    candidates: list[tuple[int, list[BBox]]] = []
    for color in fg_colors:
        comps = components_mask(grid == color)
        boxes = [bbox_coords(comp) for comp in comps]
        if len(boxes) >= 2 and all(is_full_rect_frame(grid.shape, comp) for comp in comps):
            candidates.append((int(color), boxes))

    assert len(candidates) == 1, "expected exactly one frame color"
    frame_color, frame_boxes = candidates[0]
    fill_color = next(color for color in fg_colors if color != frame_color)
    return int(frame_color), int(fill_color), frame_boxes


def _render_solution(grid: np.ndarray, frame_color: int, fill_color: int, box: BBox) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    bg = get_bg(out)
    out[out == fill_color] = bg
    frame_mask = clipped_frame_mask(out.shape, box)
    interior_mask = visible_interior_mask(out.shape, box)
    out[frame_mask] = frame_color
    out[interior_mask] = fill_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Choose the one-step frame extension whose visible interior area matches the scattered-pixel count.

    Input structure:
    - A black background contains one foreground color that forms several disjoint hollow rectangular frames.
    - The frame bounding boxes are consecutive members of one arithmetic progression in bbox coordinates.
    - A second foreground color appears as scattered witness pixels away from the frame family.

    Rule:
    - Extrapolate the frame sequence by one bbox at each end.
    - For each candidate, compute the number of in-bounds cells in its strict rectangle interior.
    - The total witness-pixel count matches exactly one candidate interior area; erase the witness pixels, draw that clipped frame, and fill its visible interior with the witness color.

    Key insight:
    - The frame family itself only determines two possible next boxes; the scattered pixels act as an area certificate that selects the correct extension end.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frame_color, fill_color, boxes = _identify_roles(g)
    ordered, step = order_arithmetic_boxes(boxes)
    candidates = [add_box(ordered[0], step, sign=-1), add_box(ordered[-1], step)]

    witness_count = int(np.count_nonzero(g == fill_color))
    assert witness_count > 0

    areas = [int(visible_interior_mask(g.shape, box).sum()) for box in candidates]
    matches = [idx for idx, area in enumerate(areas) if area == witness_count]
    assert len(matches) == 1, "witness count must identify exactly one candidate"

    return _render_solution(g, frame_color, fill_color, candidates[matches[0]])


# --------------- generator ---------------

GEN_ATTEMPTS = 400
PALETTE = tuple(range(1, 10))
MODES = ("translate", "concentric")


def _box_inside(shape: tuple[int, int], box: BBox) -> bool:
    r0, c0, r1, c1 = box
    return 0 <= r0 <= r1 < shape[0] and 0 <= c0 <= c1 < shape[1]


def _boxes_are_separate(shape: tuple[int, int], boxes: list[BBox]) -> bool:
    masks = [clipped_frame_mask(shape, box) for box in boxes]
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            if np.any(masks[i] & masks[j]):
                return False
    return True


def _candidate_is_visible(shape: tuple[int, int], box: BBox) -> bool:
    return bool(clipped_frame_mask(shape, box).any())


def _full_interior_area(box: BBox) -> int:
    r0, c0, r1, c1 = box
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    return max(0, h - 2) * max(0, w - 2)


def _expanded_forbidden(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask)
    rows, cols = np.where(mask)
    for r, c in zip(rows.tolist(), cols.tolist(), strict=False):
        r0 = max(0, r - radius)
        r1 = min(h - 1, r + radius)
        c0 = max(0, c - radius)
        c1 = min(w - 1, c + radius)
        out[r0 : r1 + 1, c0 : c1 + 1] = True
    return out


def _place_scattered(shape: tuple[int, int], forbidden: np.ndarray, count: int) -> np.ndarray | None:
    available = np.argwhere(~forbidden)
    if len(available) < count:
        return None
    order = np.random.permutation(len(available))
    chosen: list[Coord] = []
    blocked = forbidden.copy()
    for idx in order.tolist():
        r, c = map(int, available[idx])
        if blocked[r, c]:
            continue
        chosen.append((r, c))
        r0 = max(0, r - 1)
        r1 = min(shape[0] - 1, r + 1)
        c0 = max(0, c - 1)
        c1 = min(shape[1] - 1, c + 1)
        blocked[r0 : r1 + 1, c0 : c1 + 1] = True
        if len(chosen) == count:
            out = np.zeros(shape, dtype=bool)
            for rr, cc in chosen:
                out[rr, cc] = True
            return out
    return None


def _draw_frames(shape: tuple[int, int], boxes: list[BBox], color: int) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for box in boxes:
        grid[clipped_frame_mask(shape, box)] = color
    return grid


def _build_input(
    shape: tuple[int, int],
    boxes: list[BBox],
    step: BBox,
    choose_prev: bool,
    frame_color: int,
    fill_color: int,
) -> np.ndarray | None:
    candidates = [add_box(boxes[0], step, sign=-1), add_box(boxes[-1], step)]
    areas = [int(visible_interior_mask(shape, box).sum()) for box in candidates]
    if areas[0] <= 0 or areas[1] <= 0 or areas[0] == areas[1]:
        return None

    chosen_idx = 0 if choose_prev else 1
    witness_count = areas[chosen_idx]
    base = _draw_frames(shape, boxes, frame_color)

    forbidden = base != 0
    for box in candidates:
        forbidden |= clipped_frame_mask(shape, box)
        forbidden |= visible_interior_mask(shape, box)
    forbidden = _expanded_forbidden(forbidden, radius=1)
    scatter = _place_scattered(shape, forbidden, witness_count)
    if scatter is None:
        return None

    inp = base.copy()
    inp[scatter] = fill_color
    expected = _render_solution(inp, frame_color, fill_color, candidates[chosen_idx])
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def _gen_translate() -> np.ndarray | None:
    h = int(np.random.randint(4, 8))
    w = int(np.random.randint(4, 8))
    n = int(np.random.randint(2, 5))
    direction = (int(np.random.randint(-1, 2)), int(np.random.randint(-1, 2)))
    if direction == (0, 0):
        return None

    gap_r = int(np.random.randint(2, 5))
    gap_c = int(np.random.randint(2, 5))
    dr = direction[0] * (h + gap_r if direction[0] else 0)
    dc = direction[1] * (w + gap_c if direction[1] else 0)
    step = (dr, dc, dr, dc)

    hh = h + abs(dr) * (n - 1) + int(np.random.randint(4, 9))
    ww = w + abs(dc) * (n - 1) + int(np.random.randint(4, 9))
    if max(hh, ww) > 30:
        return None
    shape = (hh, ww)

    min_r = max(0, -(n - 1) * dr)
    max_r = min(hh - h, hh - h - (n - 1) * dr)
    min_c = max(0, -(n - 1) * dc)
    max_c = min(ww - w, ww - w - (n - 1) * dc)
    if min_r > max_r or min_c > max_c:
        return None

    anchor_r = int(np.random.randint(min_r, max_r + 1))
    anchor_c = int(np.random.randint(min_c, max_c + 1))
    boxes = [
        (anchor_r + i * dr, anchor_c + i * dc, anchor_r + i * dr + h - 1, anchor_c + i * dc + w - 1)
        for i in range(n)
    ]
    if not all(_box_inside(shape, box) for box in boxes):
        return None
    if not _boxes_are_separate(shape, boxes):
        return None

    prev_box = add_box(boxes[0], step, sign=-1)
    next_box = add_box(boxes[-1], step)
    if not (_candidate_is_visible(shape, prev_box) and _candidate_is_visible(shape, next_box)):
        return None
    # Keep translation cases informative by making at least one end clipped.
    full_area = _full_interior_area(boxes[0])
    areas = [int(visible_interior_mask(shape, prev_box).sum()), int(visible_interior_mask(shape, next_box).sum())]
    if areas[0] == full_area and areas[1] == full_area:
        return None

    colors = np.random.choice(PALETTE, size=2, replace=False).astype(int).tolist()
    choose_prev = bool(np.random.randint(2))
    return _build_input(shape, boxes, step, choose_prev, colors[0], colors[1])


def _gen_concentric() -> np.ndarray | None:
    n = int(np.random.randint(2, 5))
    mode = int(np.random.randint(3))
    step_size = int(np.random.randint(2, 4))
    if mode == 0:
        step = (-step_size, -step_size, step_size, step_size)
        min_h = 2 * step_size + 3
        min_w = 2 * step_size + 3
    elif mode == 1:
        step = (-step_size, 0, step_size, 0)
        min_h = 2 * step_size + 3
        min_w = 4
    else:
        step = (0, -step_size, 0, step_size)
        min_h = 4
        min_w = 2 * step_size + 3

    base_h = int(np.random.randint(min_h, max(min_h + 1, 10)))
    base_w = int(np.random.randint(min_w, max(min_w + 1, 10)))
    anchor_r = int(np.random.randint(1, 8))
    anchor_c = int(np.random.randint(1, 8))
    boxes = []
    for i in range(n):
        boxes.append(
            (
                anchor_r + i * step[0],
                anchor_c + i * step[1],
                anchor_r + base_h - 1 + i * step[2],
                anchor_c + base_w - 1 + i * step[3],
            )
        )

    all_boxes = boxes + [add_box(boxes[0], step, sign=-1), add_box(boxes[-1], step)]
    min_row = min(box[0] for box in all_boxes)
    min_col = min(box[1] for box in all_boxes)
    shift_r = int(np.random.randint(0, 30)) - min_row
    shift_c = int(np.random.randint(0, 30)) - min_col
    shifted = [add_box(box, (shift_r, shift_c, shift_r, shift_c)) for box in boxes]
    step_box = step

    max_row = max(box[2] for box in shifted)
    max_col = max(box[3] for box in shifted)
    min_row = min(box[0] for box in shifted)
    min_col = min(box[1] for box in shifted)
    hh = max_row - min_row + 1 + int(np.random.randint(3, 8))
    ww = max_col - min_col + 1 + int(np.random.randint(3, 8))
    if max(hh, ww) > 30:
        return None
    shift_inside_r = int(np.random.randint(0, hh - (max_row - min_row + 1) + 1)) - min_row
    shift_inside_c = int(np.random.randint(0, ww - (max_col - min_col + 1) + 1)) - min_col
    boxes = [add_box(box, (shift_inside_r, shift_inside_c, shift_inside_r, shift_inside_c)) for box in shifted]
    shape = (hh, ww)

    if not all(_box_inside(shape, box) for box in boxes):
        return None
    if not _boxes_are_separate(shape, boxes):
        return None

    prev_box = add_box(boxes[0], step_box, sign=-1)
    next_box = add_box(boxes[-1], step_box)
    if not (_candidate_is_visible(shape, prev_box) and _candidate_is_visible(shape, next_box)):
        return None
    areas = [int(visible_interior_mask(shape, prev_box).sum()), int(visible_interior_mask(shape, next_box).sum())]
    if areas[0] <= 0 or areas[1] <= 0 or areas[0] == areas[1]:
        return None

    colors = np.random.choice(PALETTE, size=2, replace=False).astype(int).tolist()
    choose_prev = bool(np.random.randint(2))
    return _build_input(shape, boxes, step_box, choose_prev, colors[0], colors[1])


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        mode = MODES[int(np.random.randint(len(MODES)))]
        if mode == "translate":
            grid = _gen_translate()
        else:
            grid = _gen_concentric()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
