from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
FOREGROUND_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GENERATION_ATTEMPTS = 500
SHAPE_ATTEMPTS = 300
MASK_ATTEMPTS = 300
PLACEMENT_ATTEMPTS = 300


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _component_mask(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _same_color_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) != color:
                continue
            queue = deque([(r, c)])
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _sprite_components(grid: np.ndarray) -> list[dict]:
    components: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (BLACK, GREY)):
        for coords in _same_color_components(grid, color):
            bbox = _bbox_from_coords(coords)
            components.append({"color": color, "coords": coords, "bbox": bbox})
    return components


def _is_hollow_rectangle(coords: np.ndarray) -> bool:
    bbox = _bbox_from_coords(coords)
    mask = _component_mask(coords, bbox)
    h, w = mask.shape
    if h < 3 or w < 3:
        return False
    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return np.array_equal(mask, border)


def _find_frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    frames = []
    for coords in _same_color_components(grid, GREY):
        if _is_hollow_rectangle(coords):
            frames.append(_bbox_from_coords(coords))
    assert len(frames) == 1, "expected exactly one gray hollow frame"
    return frames[0]


def _strictly_inside(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = inner
    or0, oc0, or1, oc1 = outer
    return or0 < r0 and oc0 < c0 and r1 < or1 and c1 < oc1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the hidden full sprite from aligned damaged copies, then fill it inside the frame and recolor it outside.

    Input structure:
    - A black background contains one gray hollow rectangle.
    - Inside the frame sits one monochrome damaged sprite; outside the frame are several more damaged copies of the same shape in other colors.
    - Every copy keeps the four bbox-corner pixels of the true shape, so their tight bboxes align uniquely.

    Rule:
    - Find the sprite inside the gray frame and all same-orientation copies outside it.
    - Crop every sprite to its own tight bbox and take the union of those boolean masks to recover the full shape.
    - Repaint that full shape at every sprite location: use the inside sprite's color for the repaired framed copy and also for every repaired outside copy; keep the gray frame unchanged.

    Key insight:
    - The complete shape is never shown directly: shape information comes from the union of all damaged copies, while color information flows outward from the framed exemplar.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame_bbox = _find_frame_bbox(g)
    sprites = _sprite_components(g)
    assert len(sprites) >= 3, "expected one framed copy and at least two outside copies"

    inside = [sprite for sprite in sprites if _strictly_inside(sprite["bbox"], frame_bbox)]
    assert len(inside) == 1, "expected exactly one sprite inside the frame"
    inside_sprite = inside[0]

    dims = {(b[2] - b[0] + 1, b[3] - b[1] + 1) for b in (sprite["bbox"] for sprite in sprites)}
    assert len(dims) == 1, "all copies must share the same bbox dimensions"
    shape_h, shape_w = next(iter(dims))

    full_mask = np.zeros((shape_h, shape_w), dtype=bool)
    for sprite in sprites:
        full_mask |= _component_mask(sprite["coords"], sprite["bbox"])

    assert full_mask[0, 0] and full_mask[0, -1] and full_mask[-1, 0] and full_mask[-1, -1]
    inside_color = int(inside_sprite["color"])

    out = g.copy()
    for sprite in sprites:
        coords = sprite["coords"]
        out[coords[:, 0], coords[:, 1]] = BLACK
    for sprite in sprites:
        r0, c0, r1, c1 = sprite["bbox"]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[full_mask] = inside_color

    return out


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(int(v) for v in coords[0])
    queue = deque([start])
    seen = {start}
    while queue:
        r, c = queue.popleft()
        for dr, dc in NEIGHBORS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == int(mask.sum())


def _shape_is_interesting(mask: np.ndarray) -> bool:
    if mask.all():
        return False
    border_holes = 0
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                continue
            if r in (0, h - 1) or c in (0, w - 1):
                if (r, c) not in {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}:
                    border_holes += 1
    interior_holes = 0
    if h > 2 and w > 2:
        interior_holes = int((~mask[1:-1, 1:-1]).sum())
    return border_holes + interior_holes >= 3 and (border_holes >= 1 or interior_holes >= 2)


def _make_full_shape() -> np.ndarray:
    for _ in range(SHAPE_ATTEMPTS):
        h = random.randint(5, 7)
        w = random.randint(5, 7)
        mask = np.ones((h, w), dtype=bool)
        removable = [(r, c) for r in range(h) for c in range(w) if (r, c) not in {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}]
        random.shuffle(removable)
        target_removed = random.randint(max(5, (h * w) // 4), max(7, (h * w) // 3))
        removed = 0
        for r, c in removable:
            if removed >= target_removed:
                break
            trial = mask.copy()
            trial[r, c] = False
            if trial.sum() < max(14, h + w):
                continue
            if not _is_connected(trial):
                continue
            mask = trial
            removed += 1
        non_corner_cells = int(mask.sum()) - 4
        if removed >= 4 and non_corner_cells >= 10 and _shape_is_interesting(mask):
            return mask
    raise RuntimeError("failed to build a non-degenerate full shape")


def _make_damaged_copies(full_mask: np.ndarray, total_copies: int) -> list[np.ndarray]:
    h, w = full_mask.shape
    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    cells = [tuple(int(v) for v in coord) for coord in np.argwhere(full_mask)]
    safe_cells = []
    for cell in cells:
        if cell in corners:
            continue
        trial = full_mask.copy()
        trial[cell] = False
        if _is_connected(trial):
            safe_cells.append(cell)
    if len(safe_cells) < total_copies + 2:
        raise RuntimeError("shape does not have enough safe removable cells")

    for _ in range(MASK_ATTEMPTS):
        private_cells = random.sample(safe_cells, total_copies)
        masks: list[np.ndarray] = []
        ok = True
        for i in range(total_copies):
            mask = full_mask.copy()
            order = [private_cells[j] for j in range(total_copies) if j != i]
            random.shuffle(order)
            for cell in order:
                trial = mask.copy()
                trial[cell] = False
                if not _is_connected(trial):
                    ok = False
                    break
                mask = trial
            if not ok:
                break
            masks.append(mask)
        if not ok:
            continue

        for i in range(total_copies):
            candidates = [cell for cell in safe_cells if cell != private_cells[i] and masks[i][cell]]
            random.shuffle(candidates)
            extra_target = random.randint(1, min(3, len(candidates)))
            extra_done = 0
            for cell in candidates:
                if not any(masks[j][cell] for j in range(total_copies) if j != i):
                    continue
                trial = masks[i].copy()
                trial[cell] = False
                if not _is_connected(trial):
                    continue
                masks[i] = trial
                extra_done += 1
                if extra_done >= extra_target:
                    break

        if not all(_is_connected(mask) for mask in masks):
            continue
        if any(np.array_equal(mask, full_mask) for mask in masks):
            continue
        if len({mask.tobytes() for mask in masks}) != total_copies:
            continue
        union = np.zeros_like(full_mask)
        for mask in masks:
            union |= mask
        if not np.array_equal(union, full_mask):
            continue
        indispensable = True
        for i in range(total_copies):
            others = np.zeros_like(full_mask)
            for j, mask in enumerate(masks):
                if i != j:
                    others |= mask
            if np.array_equal(others, full_mask):
                indispensable = False
                break
        if indispensable:
            return masks

    raise RuntimeError("failed to build damaged copies with the required semantics")


def _rect_separated(
    rect: tuple[int, int, int, int],
    placed: list[tuple[int, int, int, int]],
    margin: int,
) -> bool:
    r, c, h, w = rect
    for pr, pc, ph, pw in placed:
        if not (r + h + margin <= pr or pr + ph + margin <= r or c + w + margin <= pc or pc + pw + margin <= c):
            return False
    return True


def _place_layout(
    shape_h: int,
    shape_w: int,
    frame_h: int,
    frame_w: int,
    outside_count: int,
) -> tuple[int, int, tuple[int, int], list[tuple[int, int]]]:
    total_objects = outside_count + 1
    base_rows = 2 if total_objects <= 4 else 3
    min_h = max(frame_h + 4, base_rows * (shape_h + 3) + 2)
    min_w = max(frame_w + 4, 2 * (shape_w + 3) + 6)
    if min_h > 30 or min_w > 30:
        raise RuntimeError("layout would exceed ARC limits")

    for _ in range(PLACEMENT_ATTEMPTS):
        grid_h = random.randint(min_h, 30)
        grid_w = random.randint(min_w, 30)

        placed: list[tuple[int, int, int, int]] = []
        frame_pos = None
        sprite_positions: list[tuple[int, int]] = []

        frame_candidates = [
            (r, c)
            for r in range(1, grid_h - frame_h)
            for c in range(1, grid_w - frame_w)
        ]
        random.shuffle(frame_candidates)
        for r, c in frame_candidates:
            rect = (r, c, frame_h, frame_w)
            if _rect_separated(rect, placed, margin=1):
                placed.append(rect)
                frame_pos = (r, c)
                break
        if frame_pos is None:
            continue

        for _idx in range(outside_count):
            candidates = [
                (r, c)
                for r in range(1, grid_h - shape_h)
                for c in range(1, grid_w - shape_w)
                if _rect_separated((r, c, shape_h, shape_w), placed, margin=1)
            ]
            if not candidates:
                break
            pos = random.choice(candidates)
            sprite_positions.append(pos)
            placed.append((pos[0], pos[1], shape_h, shape_w))
        if len(sprite_positions) == outside_count:
            return grid_h, grid_w, frame_pos, sprite_positions

    raise RuntimeError("failed to place the frame and all copies")


def _paint_mask(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    h, w = mask.shape
    patch = grid[top : top + h, left : left + w]
    patch[mask] = color


def _expected_output(
    shape_mask: np.ndarray,
    frame_pos: tuple[int, int],
    frame_h: int,
    frame_w: int,
    inner_pos: tuple[int, int],
    outside_positions: list[tuple[int, int]],
    inside_color: int,
    grid_shape: tuple[int, int],
) -> np.ndarray:
    out = np.zeros(grid_shape, dtype=int)
    fr, fc = frame_pos
    out[fr, fc : fc + frame_w] = GREY
    out[fr + frame_h - 1, fc : fc + frame_w] = GREY
    out[fr : fr + frame_h, fc] = GREY
    out[fr : fr + frame_h, fc + frame_w - 1] = GREY

    _paint_mask(out, inner_pos[0], inner_pos[1], shape_mask, inside_color)
    for top, left in outside_positions:
        _paint_mask(out, top, left, shape_mask, inside_color)
    return out


def _validate_generated_scene(
    inp: np.ndarray,
    out: np.ndarray,
    shape_mask: np.ndarray,
    masks: list[np.ndarray],
    inside_index: int,
    colors: list[int],
) -> None:
    assert inp.ndim == 2 and out.ndim == 2
    assert max(inp.shape) <= 30 and max(out.shape) <= 30
    assert not np.array_equal(inp, out)
    assert np.array_equal(np.logical_or.reduce(masks), shape_mask)
    assert len({mask.tobytes() for mask in masks}) == len(masks)
    assert len(colors) == len(set(colors))
    assert colors[inside_index] not in {GREY, BLACK}
    assert all(color not in {GREY, BLACK} for color in colors)
    assert all(mask[0, 0] and mask[0, -1] and mask[-1, 0] and mask[-1, -1] for mask in masks)
    assert all(not np.array_equal(mask, shape_mask) for mask in masks)
    assert any(np.any(shape_mask & ~masks[inside_index]) for _ in [0])
    for i, mask in enumerate(masks):
        assert np.any(shape_mask & ~mask), f"copy {i} must be incomplete"
        others = np.zeros_like(shape_mask)
        for j, other in enumerate(masks):
            if i != j:
                others |= other
        assert not np.array_equal(others, shape_mask), f"copy {i} must contribute unique shape information"


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        try:
            shape_mask = _make_full_shape()
            shape_h, shape_w = shape_mask.shape
            outside_count = random.randint(2, 4)
            total_copies = outside_count + 1
            masks = _make_damaged_copies(shape_mask, total_copies)

            colors = random.sample(FOREGROUND_COLORS, total_copies)
            inside_index = random.randrange(total_copies)
            inside_color = colors[inside_index]

            pad_top = random.randint(1, 2)
            pad_bottom = random.randint(1, 2)
            pad_left = random.randint(1, 2)
            pad_right = random.randint(1, 2)
            frame_h = shape_h + pad_top + pad_bottom + 2
            frame_w = shape_w + pad_left + pad_right + 2

            grid_h, grid_w, frame_pos, outside_positions = _place_layout(
                shape_h, shape_w, frame_h, frame_w, outside_count
            )
            inner_pos = (frame_pos[0] + 1 + pad_top, frame_pos[1] + 1 + pad_left)

            inp = np.zeros((grid_h, grid_w), dtype=int)
            fr, fc = frame_pos
            inp[fr, fc : fc + frame_w] = GREY
            inp[fr + frame_h - 1, fc : fc + frame_w] = GREY
            inp[fr : fr + frame_h, fc] = GREY
            inp[fr : fr + frame_h, fc + frame_w - 1] = GREY

            placements: list[tuple[int, int]] = [inner_pos] + outside_positions
            ordered_masks = [masks[inside_index]] + [masks[i] for i in range(total_copies) if i != inside_index]
            ordered_colors = [colors[inside_index]] + [colors[i] for i in range(total_copies) if i != inside_index]
            for (top, left), mask, color in zip(placements, ordered_masks, ordered_colors):
                _paint_mask(inp, top, left, mask, color)

            out = _expected_output(
                shape_mask,
                frame_pos,
                frame_h,
                frame_w,
                inner_pos,
                outside_positions,
                inside_color,
                inp.shape,
            )
            _validate_generated_scene(inp, out, shape_mask, ordered_masks, 0, ordered_colors)
            if not np.array_equal(main(inp), out):
                continue
            return inp
        except RuntimeError:
            continue
    raise RuntimeError("failed to generate a valid puzzle instance")
