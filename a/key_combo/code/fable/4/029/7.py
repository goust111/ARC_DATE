from __future__ import annotations

from collections import Counter

import numpy as np


BACKGROUND = 0
FRAME_COLOR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _normalized_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    shifted = coords - np.array([r0, c0], dtype=int)
    out[shifted[:, 0], shifted[:, 1]] = True
    return out


def _count_components(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                cr, cc = stack.pop()
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return count


def _find_color_components(grid: np.ndarray, *, skip_colors: set[int]) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if seen[r, c] or color in skip_colors:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                }
            )
    return comps


def _parse_frame(mask: np.ndarray) -> tuple[int, int] | None:
    h, w = mask.shape
    full_rows = [r for r in range(h) if int(mask[r].sum()) == w]
    full_cols = [c for c in range(w) if int(mask[:, c].sum()) == h]
    if len(full_rows) != 1 or len(full_cols) != 1:
        return None
    frame_row = full_rows[0]
    frame_col = full_cols[0]
    if frame_row not in (0, h - 1) or frame_col not in (0, w - 1):
        return None
    expected = np.zeros_like(mask, dtype=bool)
    expected[frame_row, :] = True
    expected[:, frame_col] = True
    if not np.array_equal(mask, expected):
        return None
    if h == w:
        return None
    return frame_row, frame_col


def _mirror_for_longer_leg(mask: np.ndarray, frame_h: int, frame_w: int) -> np.ndarray:
    if frame_h > frame_w:
        return np.fliplr(mask)
    return np.flipud(mask)


def _inside_bbox(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return (
        (coords[:, 0] >= r0)
        & (coords[:, 0] <= r1)
        & (coords[:, 1] >= c0)
        & (coords[:, 1] <= c1)
    )


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor the unique outside copy that matches the template after the frame-directed mirror.

    Input structure:
    - A black background contains one grey one-cell-thick L-frame and several colored objects.
    - The frame encloses one asymmetric template object in the rectangle opposite the L corner.
    - Outside the frame are exactly one direct copy, one left-right mirror, one up-down mirror, plus unrelated distractors.

    Rule:
    - Read the template mask inside the grey L-frame and normalize it by its tight bounding box.
    - If the frame's vertical leg is longer, reflect the template left-right; otherwise reflect it up-down.
    - Find the unique outside monochromatic component with that normalized mirrored mask and recolor the whole component grey.

    Key insight:
    - The L-frame is both the template marker and the code for which mirror axis to use, so plain shape equality is the wrong selector.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)
    assert bg != FRAME_COLOR

    comps = _find_color_components(g, skip_colors={bg})
    frames = []
    for comp in comps:
        if int(comp["color"]) != FRAME_COLOR:
            continue
        r0, c0, r1, c1 = comp["bbox"]
        local_mask = g[r0 : r1 + 1, c0 : c1 + 1] == FRAME_COLOR
        parsed = _parse_frame(local_mask)
        if parsed is not None:
            frames.append((comp, parsed))
    assert len(frames) == 1

    frame_comp, (frame_row, frame_col) = frames[0]
    fr0, fc0, fr1, fc1 = frame_comp["bbox"]
    frame_h = fr1 - fr0 + 1
    frame_w = fc1 - fc0 + 1
    local = g[fr0 : fr1 + 1, fc0 : fc1 + 1]

    row_slice = slice(1, frame_h) if frame_row == 0 else slice(0, frame_h - 1)
    col_slice = slice(1, frame_w) if frame_col == 0 else slice(0, frame_w - 1)
    template_region = local[row_slice, col_slice]
    template_mask = template_region != bg
    assert template_mask.any()
    assert _count_components(template_mask) == 1
    template_mask = _crop_mask(template_mask)
    target_mask = _mirror_for_longer_leg(template_mask, frame_h, frame_w)

    frame_bbox = (fr0, fc0, fr1, fc1)
    matches = []
    for comp in comps:
        color = int(comp["color"])
        if color in (bg, FRAME_COLOR):
            continue
        coords = comp["coords"]
        if np.any(_inside_bbox(coords, frame_bbox)):
            continue
        if np.array_equal(_normalized_mask(coords), target_mask):
            matches.append(comp)

    assert len(matches) == 1
    out = g.copy()
    coords = matches[0]["coords"]
    out[coords[:, 0], coords[:, 1]] = FRAME_COLOR
    return out


def _random_connected_mask(rng: np.random.Generator, h: int, w: int) -> np.ndarray | None:
    area_min = max(4, int(np.ceil(h * w * 0.4)))
    area_max = min(h * w - 1, int(np.floor(h * w * 0.7)))
    if area_min > area_max:
        return None
    target = int(rng.integers(area_min, area_max + 1))
    start = (int(rng.integers(h)), int(rng.integers(w)))
    cells = {start}

    while len(cells) < target:
        frontier = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        cells.add(tuple(frontier.pop()) if len(frontier) == 1 else frontier_tuple(rng, frontier))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
        return None
    if _count_components(mask) != 1:
        return None
    variants = {mask.tobytes(), np.fliplr(mask).tobytes(), np.flipud(mask).tobytes()}
    if len(variants) != 3:
        return None
    return mask


def frontier_tuple(rng: np.random.Generator, frontier: set[tuple[int, int]]) -> tuple[int, int]:
    items = list(frontier)
    return items[int(rng.integers(len(items)))]


def _draw_shape(canvas: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    coords = np.argwhere(mask)
    canvas[top + coords[:, 0], left + coords[:, 1]] = color


def _draw_frame_patch(mask: np.ndarray, color: int, orientation: str) -> np.ndarray:
    h, w = mask.shape
    patch = np.zeros((h + 1, w + 1), dtype=int)
    if "t" in orientation:
        patch[0, :] = FRAME_COLOR
        row_slice = slice(1, h + 1)
    else:
        patch[-1, :] = FRAME_COLOR
        row_slice = slice(0, h)
    if "l" in orientation:
        patch[:, 0] = FRAME_COLOR
        col_slice = slice(1, w + 1)
    else:
        patch[:, -1] = FRAME_COLOR
        col_slice = slice(0, w)
    patch[row_slice, col_slice][mask] = color
    return patch


def _bbox_clear(existing: list[tuple[int, int, int, int]], bbox: tuple[int, int, int, int], gap: int = 1) -> bool:
    r0, c0, r1, c1 = bbox
    for er0, ec0, er1, ec1 in existing:
        if not (r1 + gap < er0 or er1 + gap < r0 or c1 + gap < ec0 or ec1 + gap < c0):
            return False
    return True


def _place_bbox(
    rng: np.random.Generator,
    existing: list[tuple[int, int, int, int]],
    grid_h: int,
    grid_w: int,
    obj_h: int,
    obj_w: int,
) -> tuple[int, int] | None:
    for _ in range(120):
        top = int(rng.integers(0, grid_h - obj_h + 1))
        left = int(rng.integers(0, grid_w - obj_w + 1))
        bbox = (top, left, top + obj_h - 1, left + obj_w - 1)
        if _bbox_clear(existing, bbox):
            existing.append(bbox)
            return top, left
    return None


def _sample_template_mask(rng: np.random.Generator, vertical_branch: bool) -> np.ndarray | None:
    for _ in range(80):
        if vertical_branch:
            h = int(rng.integers(4, 7))
            w = int(rng.integers(3, min(h, 6)))
        else:
            w = int(rng.integers(4, 7))
            h = int(rng.integers(3, min(w, 6)))
        if h == w:
            continue
        mask = _random_connected_mask(rng, h, w)
        if mask is not None:
            return mask
    return None


def _sample_distractor_mask(
    rng: np.random.Generator,
    forbidden: list[np.ndarray],
) -> np.ndarray | None:
    for _ in range(80):
        h = int(rng.integers(2, 6))
        w = int(rng.integers(2, 6))
        mask = _random_connected_mask(rng, h, w)
        if mask is None:
            continue
        cropped = _crop_mask(mask)
        if any(np.array_equal(cropped, other) for other in forbidden):
            continue
        return cropped
    return None


def _gen_attempt(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    vertical_branch = bool(rng.integers(2))
    template_mask = _sample_template_mask(rng, vertical_branch)
    if template_mask is None:
        return None

    exact_mask = template_mask
    lr_mask = np.fliplr(template_mask)
    ud_mask = np.flipud(template_mask)
    target_mask = lr_mask if vertical_branch else ud_mask
    wrong_mask = ud_mask if vertical_branch else lr_mask

    family_masks = [exact_mask, lr_mask, ud_mask]
    distractor_count = int(rng.integers(1, 4))
    distractor_masks: list[np.ndarray] = []
    for _ in range(distractor_count):
        mask = _sample_distractor_mask(rng, family_masks + distractor_masks)
        if mask is None:
            return None
        distractor_masks.append(mask)

    shape_count = 4 + distractor_count
    colors = rng.choice(np.array(PALETTE, dtype=int), size=shape_count, replace=False)
    template_color = int(colors[0])
    exact_color = int(colors[1])
    lr_color = int(colors[2])
    ud_color = int(colors[3])
    distractor_colors = [int(c) for c in colors[4:]]

    frame_orientation = ("br", "bl", "tr", "tl")[int(rng.integers(4))]
    frame_patch = _draw_frame_patch(template_mask, template_color, frame_orientation)

    obj_specs = [
        ("frame", frame_patch.shape[0], frame_patch.shape[1]),
        ("exact", exact_mask.shape[0], exact_mask.shape[1]),
        ("lr", lr_mask.shape[0], lr_mask.shape[1]),
        ("ud", ud_mask.shape[0], ud_mask.shape[1]),
    ]
    for i, mask in enumerate(distractor_masks):
        obj_specs.append((f"distractor_{i}", mask.shape[0], mask.shape[1]))

    min_h = max(14, sum(h for _name, h, _w in obj_specs[:2]) + 3)
    min_w = max(14, sum(w for _name, _h, w in obj_specs[2:4]) + 3)
    grid_h = int(rng.integers(min_h, 31))
    grid_w = int(rng.integers(min_w, 31))
    grid = np.full((grid_h, grid_w), BACKGROUND, dtype=int)
    expected = grid.copy()

    placed: list[tuple[int, int, int, int]] = []
    placements: dict[str, tuple[int, int]] = {}
    for name, obj_h, obj_w in rng.permutation(obj_specs).tolist():
        pos = _place_bbox(rng, placed, grid_h, grid_w, int(obj_h), int(obj_w))
        if pos is None:
            return None
        placements[str(name)] = pos

    frame_top, frame_left = placements["frame"]
    fr_h, fr_w = frame_patch.shape
    grid[frame_top : frame_top + fr_h, frame_left : frame_left + fr_w] = np.maximum(
        grid[frame_top : frame_top + fr_h, frame_left : frame_left + fr_w],
        frame_patch,
    )
    expected[frame_top : frame_top + fr_h, frame_left : frame_left + fr_w] = np.maximum(
        expected[frame_top : frame_top + fr_h, frame_left : frame_left + fr_w],
        frame_patch,
    )

    shape_info = [
        ("exact", exact_mask, exact_color),
        ("lr", lr_mask, lr_color),
        ("ud", ud_mask, ud_color),
    ]
    target_name = "lr" if vertical_branch else "ud"
    for name, mask, color in shape_info:
        top, left = placements[name]
        _draw_shape(grid, top, left, mask, color)
        _draw_shape(expected, top, left, mask, FRAME_COLOR if name == target_name else color)

    for i, mask in enumerate(distractor_masks):
        top, left = placements[f"distractor_{i}"]
        _draw_shape(grid, top, left, mask, distractor_colors[i])
        _draw_shape(expected, top, left, mask, distractor_colors[i])

    out = main(grid)
    if not np.array_equal(out, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _gen_attempt(rng)
        if result is not None:
            grid, _expected = result
            return grid
    raise RuntimeError("failed to generate a valid mirrored-template ARC input")
