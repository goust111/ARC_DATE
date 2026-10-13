from __future__ import annotations

import random
from collections import deque

import numpy as np


HOLE_COLOR = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _tight_crop_bool(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), mask.astype(np.uint8).tobytes()


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_monochrome_components(grid: np.ndarray, ignored: set[int]) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in ignored or visited[r, c]:
                continue
            visited[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox_from_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return components


def _find_separator_row(grid: np.ndarray, bg: int) -> tuple[int, int]:
    rows = [r for r in range(1, grid.shape[0] - 1) if np.all(grid[r] == grid[r, 0]) and int(grid[r, 0]) != bg]
    assert len(rows) == 1, "expected exactly one non-background separator row"
    r = rows[0]
    return r, int(grid[r, 0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the missing mirrored patch in the top panel, then crop the lower sprite whose shape matches that patch.

    Input structure:
    - One solid separator row splits the grid into an upper mirror puzzle and a lower candidate area.
    - The upper area is vertically mirror-symmetric except for one solid black rectangular hole; the mirrored rectangle on the opposite side remains visible.
    - The lower area contains separated monochrome candidate sprites on the shared background.

    Rule:
    - Detect the black rectangular hole in the upper area and find its horizontally mirrored rectangle.
    - Flip that intact counterpart horizontally to recover the missing patch, then take the recovered patch's non-background shape.
    - In the lower area, find the unique candidate sprite whose shape matches exactly without rotation or reflection, and return its tight crop with its own color preserved.

    Key insight:
    - The missing patch is only a key: the answer comes from first undoing the upper mirror deletion, then using the recovered shape to select a lower sprite by geometry alone.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    assert bg != HOLE_COLOR, "black is reserved for the missing rectangle"
    sep_r, sep_color = _find_separator_row(g, bg)
    top = g[:sep_r]
    bottom = g[sep_r + 1 :]
    assert top.size > 0 and bottom.size > 0, "expected content on both sides of the separator"

    hole_coords = np.argwhere(top == HOLE_COLOR)
    assert hole_coords.size > 0, "expected a black missing rectangle in the top panel"
    hr0, hc0, hr1, hc1 = _bbox_from_coords(hole_coords)
    hole_mask = top == HOLE_COLOR
    assert np.all(hole_mask[hr0 : hr1 + 1, hc0 : hc1 + 1]), "missing area must be one solid rectangle"

    mirror_c0 = top.shape[1] - 1 - hc1
    mirror_c1 = top.shape[1] - 1 - hc0
    assert 0 <= mirror_c0 <= mirror_c1 < top.shape[1], "mirrored patch must stay in bounds"
    counterpart = top[hr0 : hr1 + 1, mirror_c0 : mirror_c1 + 1]
    assert not np.any(counterpart == HOLE_COLOR), "only one side may be removed"

    recovered = np.fliplr(counterpart)
    recovered_shape = _tight_crop_bool(recovered != bg)

    matches: list[np.ndarray] = []
    for comp in _find_monochrome_components(bottom, ignored={bg, sep_color, HOLE_COLOR}):
        r0, c0, r1, c1 = comp["bbox"]
        crop = bottom[r0 : r1 + 1, c0 : c1 + 1].copy()
        if crop.shape == recovered_shape.shape and np.array_equal(crop != bg, recovered_shape):
            matches.append(crop)

    assert len(matches) == 1, "expected exactly one candidate sprite with the recovered shape"
    return matches[0]


def _random_connected_shape() -> np.ndarray:
    for _ in range(100):
        h = random.randint(2, 5)
        w = random.randint(2, 5)
        max_cells = min(h * w - 1, 8)
        if max_cells < 3:
            continue
        target = random.randint(3, max_cells)
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != target:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _tight_crop_bool(mask)
        if int(mask.sum()) < 3:
            continue
        if mask.shape == (1, 1):
            continue
        if int(mask.sum()) == int(mask.size) and random.random() < 0.8:
            continue
        return mask
    raise RuntimeError("failed to sample a connected target shape")


def _embed_shape_in_box(shape: np.ndarray, bg: int, color: int) -> np.ndarray:
    sh, sw = shape.shape
    top_pad = random.randint(0, 2)
    bottom_pad = random.randint(0, 2)
    left_pad = random.randint(0, 2)
    right_pad = random.randint(0, 2)
    box = np.full((sh + top_pad + bottom_pad, sw + left_pad + right_pad), bg, dtype=int)
    view = box[top_pad : top_pad + sh, left_pad : left_pad + sw]
    view[shape] = color
    return box


def _colorize_shape(shape: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(shape.shape, bg, dtype=int)
    patch[shape] = color
    return patch


def _random_decor_patch(bg: int, color: int) -> np.ndarray:
    kind = random.choice(["dot", "rect", "hline", "vline", "ell", "plus"])
    if kind == "dot":
        return np.array([[color]], dtype=int)
    if kind == "rect":
        return np.full((random.randint(1, 3), random.randint(1, 3)), color, dtype=int)
    if kind == "hline":
        return np.full((1, random.randint(2, 4)), color, dtype=int)
    if kind == "vline":
        return np.full((random.randint(2, 4), 1), color, dtype=int)
    if kind == "ell":
        h = random.randint(2, 3)
        w = random.randint(2, 3)
        patch = np.full((h, w), bg, dtype=int)
        patch[:, 0] = color
        patch[-1, :] = color
        return patch
    patch = np.full((3, 3), bg, dtype=int)
    patch[1, :] = color
    patch[:, 1] = color
    return patch


def _bbox_is_clear(reserved: np.ndarray, r0: int, c0: int, h: int, w: int, padding: int = 1) -> bool:
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(reserved.shape[0], r0 + h + padding)
    cc1 = min(reserved.shape[1], c0 + w + padding)
    return not bool(reserved[rr0:rr1, cc0:cc1].any())


def _reserve_bbox(reserved: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    reserved[r0 : r0 + h, c0 : c0 + w] = True


def _can_place_patch(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int, padding: int = 1) -> bool:
    h, w = patch.shape
    if r0 < 0 or c0 < 0 or r0 + h > canvas.shape[0] or c0 + w > canvas.shape[1]:
        return False
    coords = np.argwhere(patch != bg)
    if coords.size == 0:
        return False
    for dr, dc in coords:
        rr = r0 + int(dr)
        cc = c0 + int(dc)
        if int(canvas[rr, cc]) != bg:
            return False
        rr0 = max(0, rr - padding)
        cc0 = max(0, cc - padding)
        rr1 = min(canvas.shape[0], rr + padding + 1)
        cc1 = min(canvas.shape[1], cc + padding + 1)
        if np.any(canvas[rr0:rr1, cc0:cc1] != bg):
            return False
    return True


def _paste_patch(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = patch.shape
    view = canvas[r0 : r0 + h, c0 : c0 + w]
    mask = patch != bg
    view[mask] = patch[mask]


def _random_free_location(canvas: np.ndarray, patch: np.ndarray, bg: int, border: int = 1, padding: int = 1) -> tuple[int, int] | None:
    h, w = patch.shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(border, canvas.shape[0] - h - border + 1):
        for c0 in range(border, canvas.shape[1] - w - border + 1):
            if _can_place_patch(canvas, patch, r0, c0, bg, padding=padding):
                candidates.append((r0, c0))
    if not candidates:
        return None
    return random.choice(candidates)


def _build_candidate_masks(target_shape: np.ndarray, count: int) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    seen = {_shape_signature(target_shape)}

    specials = []
    for variant in (np.fliplr(target_shape), np.rot90(target_shape), np.rot90(target_shape, 3)):
        variant = _tight_crop_bool(np.asarray(variant, dtype=bool))
        sig = _shape_signature(variant)
        if sig not in seen:
            specials.append(variant)
            seen.add(sig)
    random.shuffle(specials)
    while specials and len(masks) < count - 1 and random.random() < 0.8:
        masks.append(specials.pop())

    while len(masks) < count - 1:
        candidate = _random_connected_shape()
        sig = _shape_signature(candidate)
        if sig in seen:
            continue
        seen.add(sig)
        masks.append(candidate)
    return masks


def _place_bottom_candidates(width: int, height: int, patches: list[np.ndarray], bg: int) -> np.ndarray | None:
    bottom = np.full((height, width), bg, dtype=int)
    order = sorted(range(len(patches)), key=lambda i: patches[i].shape[0] * patches[i].shape[1], reverse=True)
    for idx in order:
        loc = _random_free_location(bottom, patches[idx], bg, border=1, padding=1)
        if loc is None:
            return None
        _paste_patch(bottom, patches[idx], loc[0], loc[1], bg)
    return bottom


def _gen_attempt() -> np.ndarray | None:
    bg, sep_color, patch_color = random.sample(PALETTE, 3)

    target_shape = _random_connected_shape()
    hole_box = _embed_shape_in_box(target_shape, bg, patch_color)
    candidate_count = random.randint(3, 5)
    distractor_masks = _build_candidate_masks(target_shape, candidate_count)
    target_slot = random.randrange(candidate_count)
    candidate_masks = distractor_masks[:]
    candidate_masks.insert(target_slot, target_shape)

    candidate_palette = [c for c in PALETTE if c not in {bg, sep_color, patch_color}]
    candidate_colors = random.sample(candidate_palette, candidate_count)
    candidate_patches = [_colorize_shape(mask, candidate_colors[i], bg) for i, mask in enumerate(candidate_masks)]
    expected = candidate_patches[target_slot].copy()

    min_width = max(15, hole_box.shape[1] * 2 + 3, max(p.shape[1] for p in candidate_patches) + 4)
    width_choices = [w for w in range(min_width, 28) if w % 2 == 1]
    if not width_choices:
        return None
    width = random.choice(width_choices)

    top_h_min = max(8, hole_box.shape[0] + 2)
    bottom_h_min = max(8, max(p.shape[0] for p in candidate_patches) + 2)
    if top_h_min > 13:
        return None
    top_h = random.randint(top_h_min, 13)
    bottom_h_max = min(15, 29 - top_h)
    if bottom_h_min > bottom_h_max:
        return None
    bottom_h = random.randint(bottom_h_min, bottom_h_max)

    top = np.full((top_h, width), bg, dtype=int)
    reserved = np.zeros_like(top, dtype=bool)
    mid = width // 2
    max_left = mid - hole_box.shape[1]
    if max_left < 1:
        return None
    hole_r0 = random.randint(1, top_h - hole_box.shape[0] - 1)
    hole_c0 = random.randint(1, max_left)
    hole_right_c0 = width - hole_c0 - hole_box.shape[1]

    hole_side = random.choice(["left", "right"])
    if hole_side == "left":
        left_box = hole_box
        right_box = np.fliplr(hole_box)
    else:
        left_box = np.fliplr(hole_box)
        right_box = hole_box

    top[hole_r0 : hole_r0 + left_box.shape[0], hole_c0 : hole_c0 + left_box.shape[1]] = left_box
    top[hole_r0 : hole_r0 + right_box.shape[0], hole_right_c0 : hole_right_c0 + right_box.shape[1]] = right_box
    _reserve_bbox(reserved, hole_r0, hole_c0, left_box.shape[0], left_box.shape[1])
    _reserve_bbox(reserved, hole_r0, hole_right_c0, right_box.shape[0], right_box.shape[1])

    decor_count = random.randint(1, 4)
    decor_palette = [c for c in PALETTE if c not in {bg, sep_color}]
    for _ in range(decor_count):
        placed = False
        for _ in range(40):
            color = random.choice(decor_palette)
            patch = _random_decor_patch(bg, color)
            ph, pw = patch.shape
            max_left = mid - pw
            if max_left < 0 or top_h - ph < 0:
                continue
            r0 = random.randint(0, top_h - ph)
            c0 = random.randint(0, max_left)
            c1 = width - c0 - pw
            if not _bbox_is_clear(reserved, r0, c0, ph, pw):
                continue
            if not _bbox_is_clear(reserved, r0, c1, ph, pw):
                continue
            top[r0 : r0 + ph, c0 : c0 + pw] = patch
            top[r0 : r0 + ph, c1 : c1 + pw] = np.fliplr(patch)
            _reserve_bbox(reserved, r0, c0, ph, pw)
            _reserve_bbox(reserved, r0, c1, ph, pw)
            placed = True
            break
        if not placed:
            continue

    if random.random() < 0.8:
        center_color = random.choice(decor_palette)
        run = random.randint(1, min(4, top_h))
        start = random.randint(0, top_h - run)
        if not reserved[start : start + run, mid].any():
            top[start : start + run, mid] = center_color
            reserved[start : start + run, mid] = True

    if hole_side == "left":
        top[hole_r0 : hole_r0 + hole_box.shape[0], hole_c0 : hole_c0 + hole_box.shape[1]] = HOLE_COLOR
    else:
        top[hole_r0 : hole_r0 + hole_box.shape[0], hole_right_c0 : hole_right_c0 + hole_box.shape[1]] = HOLE_COLOR

    bottom = _place_bottom_candidates(width, bottom_h, candidate_patches, bg)
    if bottom is None:
        return None

    grid = np.vstack(
        [
            top,
            np.full((1, width), sep_color, dtype=int),
            bottom,
        ]
    )
    if max(grid.shape) > 30:
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-key ARC input")
