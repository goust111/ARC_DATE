from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

# Small connected hole shapes that never consume an entire card row/column.
SHAPE_LIBRARY = (
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
)

GEN_RETRIES = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
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
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bands(indices: np.ndarray) -> list[tuple[int, int]]:
    if indices.size == 0:
        return []
    vals = [int(x) for x in indices]
    out: list[tuple[int, int]] = []
    start = prev = vals[0]
    for x in vals[1:]:
        if x == prev + 1:
            prev = x
        else:
            out.append((start, prev))
            start = prev = x
    out.append((start, prev))
    return out


def _shape_key(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _crop_true(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _enclosed_holes(object_mask: np.ndarray) -> list[np.ndarray]:
    black = ~object_mask
    h, w = black.shape
    seen = np.zeros((h, w), dtype=bool)
    holes: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not black[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            touches_edge = False
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                if rr in (0, h - 1) or cc in (0, w - 1):
                    touches_edge = True
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if black[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_edge:
                holes.append(np.array(coords, dtype=int))
    return holes


def _single_hole_key(object_mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    holes = _enclosed_holes(object_mask)
    assert len(holes) == 1, f"expected exactly one enclosed hole, got {len(holes)}"
    return _shape_key(holes[0])


def _card_mapping(card: np.ndarray) -> dict[tuple[tuple[int, int], ...], int]:
    sep_cols = np.where(np.all(card == GREY, axis=0))[0]
    sep_bands = _bands(sep_cols)
    assert all(c0 == c1 for c0, c1 in sep_bands), "separators must be single grey columns"

    color_cols = np.where(np.any((card != BLACK) & (card != GREY), axis=0))[0]
    cell_bands = _bands(color_cols)
    if len(cell_bands) != len(sep_bands) + 1:
        side = card.shape[0]
        assert len(sep_bands) >= 1, "expected at least two card cells"
        start = sep_bands[0][0] - side
        cell_bands = [
            (start + i * (side + 1), start + i * (side + 1) + side - 1)
            for i in range(len(sep_bands) + 1)
        ]

    mapping: dict[tuple[tuple[int, int], ...], int] = {}
    for c0, c1 in cell_bands:
        assert 0 <= c0 <= c1 < card.shape[1]
        patch = card[:, c0 : c1 + 1]
        colors = set(map(int, np.unique(patch))) - {BLACK, GREY}
        assert len(colors) == 1, "each card cell must have one base color"
        base = colors.pop()
        black_coords = np.argwhere(patch == BLACK)
        assert black_coords.size > 0, "each card cell must contain a black exemplar shape"
        key = _shape_key(black_coords)
        assert key not in mapping, "card shapes must be unique"
        mapping[key] = base
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each upper object by the shape of its enclosed black hole using the bottom color card.

    Input structure:
    - The background is black.
    - Near the bottom sits a color-card strip made of small colored cells separated by single grey columns; each cell contains a black shape exemplar.
    - Above the card are several 4-connected monochrome objects, and each object contains exactly one enclosed black hole.

    Rule:
    - Extract the relative black shape inside every color-card cell and map that hole-shape to the cell's base color.
    - For each upper object, extract its unique enclosed black hole shape, matching only by translation with the card exemplars.
    - Recolor every pixel of that object to the matched card color, while leaving the color card itself unchanged.

    Key insight:
    - The visible foreground colors are distractions; both the card and the scattered objects are indexed by negative space, so the object must be classified by its internal black hole rather than by its outer contour or original color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    grey_rows = np.where(np.any(g == GREY, axis=1))[0]
    assert grey_rows.size > 0, "missing grey color-card separators"
    assert int(grey_rows[-1]) == g.shape[0] - 1, "color card must sit at the bottom"
    assert np.array_equal(grey_rows, np.arange(int(grey_rows[0]), int(grey_rows[-1]) + 1))
    card_top = int(grey_rows[0])

    mapping = _card_mapping(g[card_top:, :])
    upper = g[:card_top, :]
    object_mask = (upper != BLACK) & (upper != GREY)
    comps = _components(object_mask)
    assert comps, "expected at least one object above the color card"

    out = g.copy()
    for comp in comps:
        r0 = int(comp[:, 0].min())
        c0 = int(comp[:, 1].min())
        r1 = int(comp[:, 0].max())
        c1 = int(comp[:, 1].max())
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        local[comp[:, 0] - r0, comp[:, 1] - c0] = True
        hole_key = _single_hole_key(local)
        assert hole_key in mapping, "object hole shape must appear in the card"
        target = mapping[hole_key]
        out[comp[:, 0], comp[:, 1]] = target
    return out


def _render_card_cell(shape: np.ndarray, color: int, side: int) -> np.ndarray:
    h, w = shape.shape
    cell = np.full((side, side), color, dtype=int)
    r0 = random.randint(0, side - h)
    c0 = random.randint(0, side - w)
    cell[r0 : r0 + h, c0 : c0 + w][shape] = BLACK
    return cell


def _render_card(shape_infos: list[tuple[np.ndarray, int]], side: int) -> np.ndarray:
    width = len(shape_infos) * side + (len(shape_infos) - 1)
    card = np.zeros((side, width), dtype=int)
    c = 0
    for i, (shape, color) in enumerate(shape_infos):
        card[:, c : c + side] = _render_card_cell(shape, color, side)
        c += side
        if i + 1 < len(shape_infos):
            card[:, c] = GREY
            c += 1
    return card


def _hole_key_from_shape(shape: np.ndarray) -> tuple[tuple[int, int], ...]:
    return _shape_key(np.argwhere(shape))


def _object_hole_key_if_valid(mask: np.ndarray) -> tuple[tuple[int, int], ...] | None:
    holes = _enclosed_holes(mask)
    if len(holes) != 1:
        return None
    return _shape_key(holes[0])


def _object_patch(shape: np.ndarray, color: int) -> np.ndarray | None:
    hole_h, hole_w = shape.shape
    base_h = hole_h + 2
    base_w = hole_w + 2
    pad_top = random.randint(0, 2)
    pad_bottom = random.randint(0, 2)
    pad_left = random.randint(0, 2)
    pad_right = random.randint(0, 2)

    obj = np.zeros((base_h + pad_top + pad_bottom, base_w + pad_left + pad_right), dtype=bool)
    fixed_hole = np.zeros_like(obj, dtype=bool)
    r0 = pad_top
    c0 = pad_left
    obj[r0 : r0 + base_h, c0 : c0 + base_w] = True
    fixed_hole[r0 + 1 : r0 + 1 + hole_h, c0 + 1 : c0 + 1 + hole_w] = shape
    obj[fixed_hole] = False

    base_area = int(obj.sum())
    if not (6 <= base_area <= 15):
        return None

    target_area = random.randint(base_area, min(15, base_area + 4))
    base_key = _hole_key_from_shape(shape)
    while int(obj.sum()) < target_area:
        frontier = []
        h, w = obj.shape
        for rr in range(h):
            for cc in range(w):
                if obj[rr, cc] or fixed_hole[rr, cc]:
                    continue
                if any(obj[nr, nc] for nr, nc in _neighbors4(rr, cc, h, w)):
                    frontier.append((rr, cc))
        random.shuffle(frontier)
        placed = False
        for rr, cc in frontier:
            test = obj.copy()
            test[rr, cc] = True
            if _object_hole_key_if_valid(test) == base_key:
                obj = test
                placed = True
                break
        if not placed:
            break

    patch = np.zeros(obj.shape, dtype=int)
    patch[obj] = color
    coords = np.argwhere(patch != BLACK)
    r1, c1 = coords.min(axis=0)
    r2, c2 = coords.max(axis=0)
    patch = patch[int(r1) : int(r2) + 1, int(c1) : int(c2) + 1]

    local_mask = patch != BLACK
    if _object_hole_key_if_valid(local_mask) != base_key:
        return None
    if not (6 <= int(local_mask.sum()) <= 15):
        return None
    return patch


def _place_patch(grid: np.ndarray, patch: np.ndarray, r: int, c: int, color_override: int | None = None) -> None:
    mask = patch != BLACK
    if color_override is None:
        grid[r : r + patch.shape[0], c : c + patch.shape[1]][mask] = patch[mask]
    else:
        grid[r : r + patch.shape[0], c : c + patch.shape[1]][mask] = color_override


def _gen_attempt() -> np.ndarray | None:
    side = random.choice((3, 4))
    n_classes = random.randint(2, 4)
    shapes = random.sample(list(SHAPE_LIBRARY), n_classes)
    target_colors = random.sample(PALETTE, n_classes)
    shape_infos = list(zip(shapes, target_colors))
    gap = random.randint(1, 3)
    total_h = random.randint(16, 29)
    if total_h - gap - side < 8:
        return None
    upper_h = total_h - gap - side
    card = _render_card(shape_infos, side)
    min_w = max(card.shape[1] + 1, 16)
    width = random.randint(min_w, 30)
    left_margin = random.randint(0, width - card.shape[1])

    inp = np.zeros((total_h, width), dtype=int)
    out = np.zeros((total_h, width), dtype=int)
    inp[-side:, left_margin : left_margin + card.shape[1]] = card
    out[:, :] = inp

    object_count = n_classes + random.randint(1, 3)
    class_ids = list(range(n_classes)) + random.choices(range(n_classes), k=object_count - n_classes)
    random.shuffle(class_ids)

    color_pool = random.sample(PALETTE, random.randint(2, 4))
    specs = []
    for class_id in class_ids:
        obj_color = random.choice(color_pool)
        patch = _object_patch(shapes[class_id], obj_color)
        if patch is None:
            return None
        specs.append((patch, target_colors[class_id]))

    if len({int(patch[patch != BLACK][0]) for patch, _ in specs}) < 2:
        return None

    specs.sort(key=lambda item: -int((item[0] != BLACK).sum()))
    reserved = np.zeros((upper_h, width), dtype=bool)
    changed = 0
    for patch, target_color in specs:
        ph, pw = patch.shape
        if ph > upper_h or pw > width:
            return None
        placed = False
        for _ in range(80):
            r = random.randint(0, upper_h - ph)
            c = random.randint(0, width - pw)
            rr0 = max(0, r - 1)
            rr1 = min(upper_h, r + ph + 1)
            cc0 = max(0, c - 1)
            cc1 = min(width, c + pw + 1)
            if reserved[rr0:rr1, cc0:cc1].any():
                continue
            _place_patch(inp, patch, r, c)
            _place_patch(out, patch, r, c, color_override=target_color)
            reserved[rr0:rr1, cc0:cc1] = True
            if int(patch[patch != BLACK][0]) != target_color:
                changed += 1
            placed = True
            break
        if not placed:
            return None

    if changed == 0:
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
