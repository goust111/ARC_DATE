import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 400


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    """BBox of an (N,2) coords array as (r0,c0,r1,c1), inclusive."""
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Crop a grid to an inclusive bbox."""
    r0, c0, r1, c1 = map(int, box)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int = BLACK) -> np.ndarray:
    """Paste a patch onto a destination grid, ignoring transparent cells."""
    h, w = map(int, patch.shape)
    r0 = int(r0)
    c0 = int(c0)
    view = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != int(transparent)
    view[mask] = patch[mask]
    return dst


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    """4-connected components of a boolean mask, each as an (N,2) coord array."""
    h, w = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def border_coords(shape: tuple[int, int]) -> list[tuple[int, int]]:
    """Clockwise perimeter coordinates starting from the upper-left corner."""
    h, w = map(int, shape)
    coords: list[tuple[int, int]] = []
    for c in range(w):
        coords.append((0, c))
    for r in range(1, h - 1):
        coords.append((r, w - 1))
    for c in range(w - 1, -1, -1):
        coords.append((h - 1, c))
    for r in range(h - 2, 0, -1):
        coords.append((r, 0))
    return coords


def _border_model(grid: np.ndarray) -> tuple[list[tuple[int, int]], np.ndarray, list[tuple[int, int]], int]:
    coords = border_coords(grid.shape)
    values = np.array([int(grid[r, c]) for r, c in coords], dtype=int)
    colors = sorted(set(map(int, values.tolist())))
    assert len(colors) == 2, "border must use exactly two colors"

    best_expected = None
    best_mismatches = None
    second_best = None
    best_coords = None
    for start in colors:
        other = colors[1] if start == colors[0] else colors[0]
        expected = np.array([start if i % 2 == 0 else other for i in range(len(coords))], dtype=int)
        mismatch_mask = values != expected
        mismatch_coords = [coords[i] for i in np.flatnonzero(mismatch_mask)]
        count = len(mismatch_coords)
        if best_mismatches is None or count < best_mismatches:
            second_best = best_mismatches
            best_mismatches = count
            best_expected = expected
            best_coords = mismatch_coords
        elif second_best is None or count < second_best:
            second_best = count

    assert best_expected is not None and best_mismatches is not None and best_coords is not None
    if second_best is None:
        second_best = best_mismatches
    return coords, best_expected, best_coords, second_best


def _detect_defects(grid: np.ndarray) -> tuple[list[tuple[int, int]], np.ndarray, tuple[int, int], tuple[int, int]]:
    coords, expected, mismatches, second_best = _border_model(grid)
    assert len(mismatches) == 2, "expected exactly two border anomalies"
    assert second_best > 2, "border phase should be uniquely determined"

    h, w = map(int, grid.shape)
    horizontal = [p for p in mismatches if p[0] in (0, h - 1) and p[1] not in (0, w - 1)]
    vertical = [p for p in mismatches if p[1] in (0, w - 1) and p[0] not in (0, h - 1)]
    assert len(horizontal) == 1 and len(vertical) == 1, "need one horizontal and one vertical defect"
    return coords, expected, horizontal[0], vertical[0]


def _extract_anchor_sprite(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int], tuple[int, int]]:
    inner = np.asarray(grid, dtype=int).copy()
    for r, c in border_coords(inner.shape):
        inner[r, c] = BLACK

    components = components_mask(inner != BLACK)
    assert len(components) >= 2, "expected the anchor sprite plus distractors"

    anchor_component = None
    anchor_box = None
    for comp in components:
        has_grey = any(int(inner[r, c]) == GREY for r, c in comp)
        if has_grey:
            assert anchor_component is None, "grey anchor must identify exactly one sprite"
            anchor_component = comp
            anchor_box = bbox_coords(comp)

    assert anchor_component is not None and anchor_box is not None
    sprite = crop_bbox(inner, anchor_box)
    anchor_pixels = np.argwhere(sprite == GREY)
    assert anchor_pixels.shape == (1, 2), "anchor sprite must contain exactly one grey pixel"
    anchor_local = tuple(map(int, anchor_pixels[0]))
    return inner, anchor_component, anchor_box, anchor_local


def _render_moved(grid: np.ndarray, top_left: tuple[int, int]) -> np.ndarray:
    coords, expected, _horizontal_defect, _vertical_defect = _detect_defects(grid)
    inner, anchor_component, anchor_box, _anchor_local = _extract_anchor_sprite(grid)
    sprite = crop_bbox(inner, anchor_box)

    out = np.asarray(grid, dtype=int).copy()
    for (r, c), value in zip(coords, expected):
        out[r, c] = int(value)
    out[anchor_component[:, 0], anchor_component[:, 1]] = BLACK
    paste(out, sprite, int(top_left[0]), int(top_left[1]), transparent=BLACK)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the grey-anchor sprite to the crosshair defined by the two periodic-border defects.

    Input structure:
    - The outer border is a strict two-color alternation around the full perimeter, except for exactly
      two phase-flipped defect cells.
    - One defect lies on a horizontal edge and the other lies on a vertical edge.
    - Inside the border are several separated sprite copies; exactly one copy contains a single grey anchor pixel.

    Rule:
    - Recover the intended alternating border phase and identify the two mismatching perimeter cells.
    - Use the vertical-edge defect's row and the horizontal-edge defect's column as the target intersection.
    - Rigidly translate only the sprite containing the grey anchor so that the grey pixel lands on that
      intersection, clear its old position to black, leave distractors unchanged, and repair the two border defects.

    Key insight:
    - The two anomalous border cells are not destinations by themselves: each contributes one coordinate
      component, and only their row/column crosshair determines where the anchored sprite must move.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    _coords, expected, horizontal_defect, vertical_defect = _detect_defects(g)
    inner, anchor_component, anchor_box, anchor_local = _extract_anchor_sprite(g)
    sprite = crop_bbox(inner, anchor_box)

    target_anchor = (int(vertical_defect[0]), int(horizontal_defect[1]))
    target_top = (target_anchor[0] - anchor_local[0], target_anchor[1] - anchor_local[1])
    assert 0 <= target_top[0] and 0 <= target_top[1]
    assert target_top[0] + sprite.shape[0] <= g.shape[0] and target_top[1] + sprite.shape[1] <= g.shape[1]

    out = g.copy()
    for (r, c), value in zip(border_coords(g.shape), expected):
        out[r, c] = int(value)
    out[anchor_component[:, 0], anchor_component[:, 1]] = BLACK
    paste(out, sprite, target_top[0], target_top[1], transparent=BLACK)
    return out


def _random_connected_mask() -> np.ndarray | None:
    for _ in range(80):
        h = random.randint(3, 6)
        w = random.randint(3, 6)
        max_area = min(h * w - 1, 12)
        area = random.randint(5, max_area)

        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        if not mask.any():
            continue
        box = bbox_coords(np.argwhere(mask))
        mask = mask[box[0] : box[2] + 1, box[1] : box[3] + 1]
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if int(mask.sum()) < 5 or int(mask.sum()) == mask.size:
            continue
        return mask
    return None


def _make_anchor_sprite(mask: np.ndarray, body_colors: list[int]) -> tuple[np.ndarray, tuple[int, int]]:
    coords = np.argwhere(mask)
    anchor_r, anchor_c = map(int, coords[random.randrange(len(coords))])
    sprite = np.zeros(mask.shape, dtype=int)
    for r, c in coords:
        sprite[int(r), int(c)] = int(body_colors[(int(r) + 2 * int(c)) % len(body_colors)])
    sprite[anchor_r, anchor_c] = GREY
    return sprite, (anchor_r, anchor_c)


def _make_distractor(anchor_sprite: np.ndarray, body_colors: list[int]) -> np.ndarray:
    distractor = anchor_sprite.copy()
    anchor_r, anchor_c = map(int, np.argwhere(distractor == GREY)[0])
    distractor[anchor_r, anchor_c] = int(body_colors[(anchor_r + anchor_c + 1) % len(body_colors)])
    return distractor


def _build_clean_border(h: int, w: int, colors: tuple[int, int]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    start = random.choice(colors)
    other = colors[1] if start == colors[0] else colors[0]
    for i, (r, c) in enumerate(border_coords((h, w))):
        grid[r, c] = int(start if i % 2 == 0 else other)
    return grid


def _boxes_far(box: tuple[int, int, int, int], others: list[tuple[int, int, int, int]], gap: int = 1) -> bool:
    r0, c0, r1, c1 = map(int, box)
    for rr0, cc0, rr1, cc1 in others:
        if r0 <= rr1 + gap and r1 + gap >= rr0 and c0 <= cc1 + gap and c1 + gap >= cc0:
            return False
    return True


def _random_position(
    shape: tuple[int, int],
    patch_shape: tuple[int, int],
    boxes: list[tuple[int, int, int, int]],
) -> tuple[int, int] | None:
    h, w = map(int, shape)
    ph, pw = map(int, patch_shape)
    candidates = []
    for r0 in range(1, h - ph):
        for c0 in range(1, w - pw):
            box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            if _boxes_far(box, boxes, gap=1):
                candidates.append((r0, c0))
    if not candidates:
        return None
    return random.choice(candidates)


def _flip_border_cell(grid: np.ndarray, coord: tuple[int, int], border_colors: tuple[int, int]) -> None:
    r, c = map(int, coord)
    grid[r, c] = int(border_colors[1] if int(grid[r, c]) == int(border_colors[0]) else border_colors[0])


def _validate_scene(
    inp: np.ndarray,
    out: np.ndarray,
    source_anchor: tuple[int, int],
    target_anchor: tuple[int, int],
    source_top: tuple[int, int],
    target_top: tuple[int, int],
    sprite: np.ndarray,
    expected_horizontal: tuple[int, int],
    expected_vertical: tuple[int, int],
) -> bool:
    if inp.ndim != 2 or out.ndim != 2 or max(inp.shape) > 30 or max(out.shape) > 30:
        return False

    try:
        coords, expected, horizontal_defect, vertical_defect = _detect_defects(inp)
        if tuple(horizontal_defect) != tuple(expected_horizontal) or tuple(vertical_defect) != tuple(expected_vertical):
            return False
        _clean_coords, _clean_expected, clean_mismatches, _second = _border_model(out)
        if len(clean_mismatches) != 0:
            return False

        _inner, _anchor_component, anchor_box, anchor_local = _extract_anchor_sprite(inp)
        if tuple(anchor_local) != (source_anchor[0] - source_top[0], source_anchor[1] - source_top[1]):
            return False
        if tuple(anchor_box[:2]) != tuple(source_top):
            return False
        if source_anchor[0] == target_anchor[0] or source_anchor[1] == target_anchor[1]:
            return False
        if tuple(target_top) == tuple(source_top):
            return False

        target_view = inp[target_top[0] : target_top[0] + sprite.shape[0], target_top[1] : target_top[1] + sprite.shape[1]]
        if target_view.shape != sprite.shape or np.any(target_view[sprite != BLACK] != BLACK):
            return False

        repair_only = inp.copy()
        for (r, c), value in zip(coords, expected):
            repair_only[r, c] = int(value)
        if np.array_equal(repair_only, out):
            return False

        row_only = _render_moved(inp, (target_top[0], source_top[1]))
        col_only = _render_moved(inp, (source_top[0], target_top[1]))
        if np.array_equal(row_only, out) or np.array_equal(col_only, out):
            return False

        if not np.array_equal(main(inp), out):
            return False
    except Exception:
        return False

    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(16, 24)
    w = random.randint(16, 24)

    border_colors = tuple(random.sample(COLOR_POOL, 2))
    body_pool = [c for c in COLOR_POOL if c not in border_colors]
    body_colors = random.sample(body_pool, 2)

    mask = _random_connected_mask()
    if mask is None:
        return None
    sprite, anchor_local = _make_anchor_sprite(mask, body_colors)
    ph, pw = map(int, sprite.shape)

    target_out = _build_clean_border(h, w, border_colors)
    placed_boxes: list[tuple[int, int, int, int]] = []

    target_pos = _random_position((h, w), sprite.shape, placed_boxes)
    if target_pos is None:
        return None
    target_top = tuple(map(int, target_pos))
    target_box = (target_top[0], target_top[1], target_top[0] + ph - 1, target_top[1] + pw - 1)
    target_anchor = (target_top[0] + anchor_local[0], target_top[1] + anchor_local[1])
    placed_boxes.append(target_box)
    paste(target_out, sprite, target_top[0], target_top[1], transparent=BLACK)

    distractor_count = random.randint(2, 4)
    distractor_patch = _make_distractor(sprite, body_colors)
    for _ in range(distractor_count):
        pos = _random_position((h, w), distractor_patch.shape, placed_boxes)
        if pos is None:
            return None
        r0, c0 = map(int, pos)
        box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        placed_boxes.append(box)
        paste(target_out, distractor_patch, r0, c0, transparent=BLACK)

    source_pos = _random_position((h, w), sprite.shape, placed_boxes)
    if source_pos is None:
        return None
    source_top = tuple(map(int, source_pos))
    source_anchor = (source_top[0] + anchor_local[0], source_top[1] + anchor_local[1])
    if source_anchor[0] == target_anchor[0] or source_anchor[1] == target_anchor[1]:
        return None

    inp = target_out.copy()
    target_view = inp[target_top[0] : target_top[0] + ph, target_top[1] : target_top[1] + pw]
    target_view[sprite != BLACK] = BLACK
    paste(inp, sprite, source_top[0], source_top[1], transparent=BLACK)

    horizontal_side = random.choice(("top", "bottom"))
    vertical_side = random.choice(("left", "right"))
    horizontal_defect = (0, target_anchor[1]) if horizontal_side == "top" else (h - 1, target_anchor[1])
    vertical_defect = (target_anchor[0], 0) if vertical_side == "left" else (target_anchor[0], w - 1)
    _flip_border_cell(inp, horizontal_defect, border_colors)
    _flip_border_cell(inp, vertical_defect, border_colors)

    out = target_out.copy()
    if not _validate_scene(
        inp,
        out,
        source_anchor=source_anchor,
        target_anchor=target_anchor,
        source_top=source_top,
        target_top=target_top,
        sprite=sprite,
        expected_horizontal=horizontal_defect,
        expected_vertical=vertical_defect,
    ):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid double-defect targeting input")
