from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
YELLOW = 4
SUPPORT_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)
GEN_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _coords_bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _scale(arr: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(np.asarray(arr, dtype=int), np.ones((factor, factor), dtype=int))


def _connected(mask: np.ndarray) -> bool:
    comps = _components(mask)
    return len(comps) == 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer the template's yellow missing-mask onto every larger scaled copy of the template body.

    Input structure:
    - A black background contains one small complete template sprite whose yellow cells mark the missing-mask.
    - Elsewhere there are integer-upscaled copies of the template's non-yellow body, with the yellow part omitted.
    - Extra distractor pixels may appear, but they do not form a valid scaled body copy.

    Rule:
    - Use the bounding box of all yellow cells to crop the template window.
    - For each other non-background connected component, infer a scale s from its bbox if that bbox is exactly s times the template size.
    - When the component's bbox matches the template with yellow cells removed and then scaled by s, paint the corresponding scaled yellow blocks into that bbox.

    Key insight:
    - The copies are recognized by exact agreement with the template body at their own scale, and the yellow mask must be upscaled by the same factor before being stamped back.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(np.bincount(g.ravel()).argmax())
    yellow_box = _bbox(g == YELLOW)
    assert yellow_box is not None, "expected yellow template cells"

    template = _crop(g, yellow_box)
    fill_mask = template == YELLOW
    assert np.any(fill_mask), "template must contain yellow cells"
    th, tw = map(int, template.shape)

    partial_template = template.copy()
    partial_template[fill_mask] = bg

    out = g.copy()
    matched = 0
    for comp in _components(g != bg):
        rr, cc = comp[:, 0], comp[:, 1]
        if np.any(g[rr, cc] == YELLOW):
            continue

        box = _coords_bbox(comp)
        r0, c0, r1, c1 = box
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h % th != 0 or w % tw != 0:
            continue
        scale = h // th
        if scale < 1 or w // tw != scale:
            continue

        window = _crop(g, box)
        expected_partial = _scale(partial_template, scale)
        if not np.array_equal(window, expected_partial):
            continue

        scaled_fill = _scale(fill_mask.astype(int), scale).astype(bool)
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[scaled_fill] = YELLOW
        matched += 1

    assert matched >= 1, "expected at least one scaled partial copy"
    return out


def _sample_fill_mask(h: int, w: int) -> np.ndarray:
    cells = [(r, c) for r in range(h) for c in range(w)]
    target_min = 4
    target_max = min(h * w - 5, max(4, (h * w) // 2))
    assert target_min <= target_max

    for _ in range(300):
        target = random.randint(target_min, target_max)
        fill = {
            (0, random.randrange(w)),
            (h - 1, random.randrange(w)),
            (random.randrange(h), 0),
            (random.randrange(h), w - 1),
        }
        remaining = [cell for cell in cells if cell not in fill]
        random.shuffle(remaining)
        for cell in remaining[: max(0, target - len(fill))]:
            fill.add(cell)

        mask = np.zeros((h, w), dtype=bool)
        for r, c in fill:
            mask[r, c] = True
        support = ~mask
        if support.sum() < 5:
            continue
        if not mask[0, :].any() or not mask[-1, :].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if not support[0, :].any() or not support[-1, :].any() or not support[:, 0].any() or not support[:, -1].any():
            continue
        if not _connected(support):
            continue
        return mask
    raise RuntimeError("failed to sample a valid fill mask")


def _build_template(h: int, w: int) -> np.ndarray:
    fill_mask = _sample_fill_mask(h, w)
    support_coords = [(r, c) for r in range(h) for c in range(w) if not fill_mask[r, c]]
    color_count = random.randint(2, min(3, len(support_coords)))
    palette = random.sample(SUPPORT_COLORS, color_count)
    seeds = random.sample(support_coords, color_count)

    template = np.full((h, w), YELLOW, dtype=int)
    for r, c in support_coords:
        best = min(
            range(color_count),
            key=lambda i: (abs(r - seeds[i][0]) + abs(c - seeds[i][1]), i),
        )
        template[r, c] = palette[best]
    return template


def _can_place(reserved: np.ndarray, top: int, left: int, h: int, w: int, pad: int = 1) -> bool:
    rows, cols = reserved.shape
    if top < 0 or left < 0 or top + h > rows or left + w > cols:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(rows, top + h + pad)
    c1 = min(cols, left + w + pad)
    return not reserved[r0:r1, c0:c1].any()


def _reserve(reserved: np.ndarray, top: int, left: int, h: int, w: int, pad: int = 1) -> None:
    rows, cols = reserved.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(rows, top + h + pad)
    c1 = min(cols, left + w + pad)
    reserved[r0:r1, c0:c1] = True


def _paste_non_bg(canvas: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int = BG) -> None:
    view = canvas[top : top + patch.shape[0], left : left + patch.shape[1]]
    mask = patch != bg
    view[mask] = patch[mask]


def _try_place_boxes(shape: tuple[int, int], boxes: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    rows, cols = shape
    reserved = np.zeros((rows, cols), dtype=bool)
    placements: list[tuple[int, int]] = [(-1, -1)] * len(boxes)
    order = sorted(range(len(boxes)), key=lambda i: boxes[i][0] * boxes[i][1], reverse=True)

    for idx in order:
        h, w = boxes[idx]
        candidates = [(r, c) for r in range(rows - h + 1) for c in range(cols - w + 1) if _can_place(reserved, r, c, h, w)]
        if not candidates:
            return None
        top, left = random.choice(candidates)
        _reserve(reserved, top, left, h, w)
        placements[idx] = (top, left)
    return placements


def _scatter_distractors(grid: np.ndarray, reserved: np.ndarray, count: int) -> bool:
    rows, cols = grid.shape
    attempts = 0
    placed = 0
    while placed < count and attempts < 200:
        attempts += 1
        h = 1 if random.random() < 0.7 else random.randint(1, 2)
        w = 1 if h == 1 or random.random() < 0.7 else 2
        shape_type = random.choice(("dot", "domino", "corner"))
        patch = np.zeros((h, w), dtype=int)
        color = random.choice([c for c in SUPPORT_COLORS if c != YELLOW])
        if shape_type == "dot" or (h, w) == (1, 1):
            patch[0, 0] = color
        elif shape_type == "domino":
            if random.random() < 0.5:
                patch = np.array([[color, color]], dtype=int)
            else:
                patch = np.array([[color], [color]], dtype=int)
        else:
            patch = np.array([[color, 0], [color, color]], dtype=int)

        ph, pw = patch.shape
        candidates = [(r, c) for r in range(rows - ph + 1) for c in range(cols - pw + 1) if _can_place(reserved, r, c, ph, pw)]
        if not candidates:
            return placed == count
        top, left = random.choice(candidates)
        _reserve(reserved, top, left, ph, pw)
        _paste_non_bg(grid, patch, top, left)
        placed += 1
    return placed == count


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(3, 5)
    w = random.randint(3, 5)
    template = _build_template(h, w)
    partial_template = template.copy()
    partial_template[partial_template == YELLOW] = BG

    max_scale = min(4, 28 // max(h, w))
    if max_scale < 2:
        return None
    scale_choices = list(range(2, max_scale + 1))
    copy_count = random.randint(2, min(4, len(scale_choices) + 1))
    if len(scale_choices) >= copy_count and random.random() < 0.75:
        scales = random.sample(scale_choices, copy_count)
    else:
        scales = [random.choice(scale_choices) for _ in range(copy_count)]

    object_boxes = [(h, w)] + [(h * s, w * s) for s in scales]
    max_h = max(box_h for box_h, _ in object_boxes)
    max_w = max(box_w for _, box_w in object_boxes)
    min_rows = max(max_h + 2, min(30, max_h + h + 2))
    min_cols = max(max_w + 2, min(30, max_w + w + 2))
    if min_rows > 30 or min_cols > 30:
        return None

    for _ in range(80):
        rows = random.randint(min_rows, 30)
        cols = random.randint(min_cols, 30)
        placements = _try_place_boxes((rows, cols), object_boxes)
        if placements is None:
            continue

        inp = np.zeros((rows, cols), dtype=int)
        out = np.zeros((rows, cols), dtype=int)
        reserved = np.zeros((rows, cols), dtype=bool)

        src_top, src_left = placements[0]
        _paste_non_bg(inp, template, src_top, src_left)
        _paste_non_bg(out, template, src_top, src_left)
        _reserve(reserved, src_top, src_left, h, w)

        for i, scale in enumerate(scales, start=1):
            top, left = placements[i]
            partial = _scale(partial_template, scale)
            full = _scale(template, scale)
            _paste_non_bg(inp, partial, top, left)
            _paste_non_bg(out, full, top, left)
            _reserve(reserved, top, left, partial.shape[0], partial.shape[1])

        distractor_count = random.randint(3, 8)
        if not _scatter_distractors(inp, reserved, distractor_count):
            continue
        _paste_non_bg(out, inp, 0, 0)
        for i, scale in enumerate(scales, start=1):
            top, left = placements[i]
            full = _scale(template, scale)
            _paste_non_bg(out, full, top, left)

        try:
            produced = main(inp)
        except Exception:
            continue
        if np.array_equal(produced, out):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
