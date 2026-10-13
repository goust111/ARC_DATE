from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
ARROW = 9
SIDES = ("top", "right", "bottom", "left")
SCAN_ORDER = ("top", "right", "bottom", "left")
MISSING_TO_FLIP = {
    (0, 1): "horizontal",
    (1, 0): "vertical",
    (0, 0): "main_diag",
    (1, 1): "anti_diag",
}
FLIP_TO_MISSING = {value: key for key, value in MISSING_TO_FLIP.items()}


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(coords)
    return parts


def _edge_marker_positions(shape: tuple[int, int]) -> dict[str, tuple[int, int]]:
    h, w = shape
    positions = {
        "top": (0, w // 2),
        "right": (h // 2, w - 1),
        "bottom": (h - 1, w // 2),
        "left": (h // 2, 0),
    }
    assert len(set(positions.values())) == 4, "edge marker positions must be distinct"
    return positions


def _touching_sides(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> list[str]:
    r0, c0, r1, c1 = bbox
    h, w = shape
    touched = []
    if r0 == 0:
        touched.append("top")
    if c1 == w - 1:
        touched.append("right")
    if r1 == h - 1:
        touched.append("bottom")
    if c0 == 0:
        touched.append("left")
    assert touched, "every object bbox must touch at least one grid side"
    return touched


def _contact_color(touched: list[str], edge_colors: dict[str, int]) -> int:
    for side in SCAN_ORDER:
        if side in touched:
            return int(edge_colors[side])
    raise AssertionError("unreachable")


def _transform_mask(mask: np.ndarray, flip_kind: str) -> np.ndarray:
    if flip_kind == "horizontal":
        return np.fliplr(mask)
    if flip_kind == "vertical":
        return np.flipud(mask)
    if flip_kind == "main_diag":
        assert mask.shape[0] == mask.shape[1], "diagonal flips require a square bbox"
        return mask.T
    if flip_kind == "anti_diag":
        assert mask.shape[0] == mask.shape[1], "diagonal flips require a square bbox"
        return np.rot90(mask, 2).T
    raise AssertionError(f"unknown flip kind: {flip_kind}")


def _parse_object(g: np.ndarray, coords: list[tuple[int, int]]) -> dict[str, object]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    for r, c in coords:
        patch[r - r0, c - c0] = int(g[r, c])

    non_bg = set(map(int, patch[patch != BG].tolist()))
    object_colors = non_bg - {ARROW}
    assert len(object_colors) == 1, "each object must have one base color plus the maroon arrow"
    object_color = object_colors.pop()
    occ = patch != BG

    arrow_candidates = []
    for rr in range(patch.shape[0] - 1):
        for cc in range(patch.shape[1] - 1):
            block = patch[rr : rr + 2, cc : cc + 2]
            if not bool(np.all(occ[rr : rr + 2, cc : cc + 2])):
                continue
            arrow_count = int(np.sum(block == ARROW))
            base_count = int(np.sum(block == object_color))
            if arrow_count == 3 and base_count == 1:
                miss = tuple(int(x) for x in np.argwhere(block == object_color)[0])
                arrow_candidates.append((rr, cc, miss))
    assert len(arrow_candidates) == 1, "each object must contain exactly one 3-cell arrow block"
    arrow_r, arrow_c, missing_corner = arrow_candidates[0]
    flip_kind = MISSING_TO_FLIP[missing_corner]

    return {
        "bbox": (r0, c0, r1, c1),
        "patch": patch,
        "mask": occ,
        "color": int(object_color),
        "arrow_pos": (arrow_r, arrow_c),
        "flip_kind": flip_kind,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flip each object inside its bbox according to the embedded maroon L marker,
    then recolor it from the edge markers selected by bbox boundary contact.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 5 and g.shape[1] >= 5

    edge_positions = _edge_marker_positions(g.shape)
    edge_colors = {side: int(g[pos]) for side, pos in edge_positions.items()}

    object_mask = g != BG
    for pos in edge_positions.values():
        object_mask[pos] = False

    objects = _components(object_mask)
    assert objects, "expected at least one object"

    out = g.copy()
    for coords in objects:
        for r, c in coords:
            out[r, c] = BG

    for coords in objects:
        parsed = _parse_object(g, coords)
        r0, c0, r1, c1 = parsed["bbox"]
        mask = parsed["mask"]
        flip_kind = parsed["flip_kind"]
        transformed = _transform_mask(mask, flip_kind)
        color = _contact_color(_touching_sides((r0, c0, r1, c1), g.shape), edge_colors)
        for rr, cc in np.argwhere(transformed):
            out[r0 + int(rr), c0 + int(cc)] = color

    return out


GRID_SIZES = (11, 13, 15, 17, 19, 21)
OBJECT_COUNT_CHOICES = [1, 2, 3, 4]
OBJECT_COUNT_WEIGHTS = [1, 2, 2, 1]
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8]
CONTACT_TYPES = ("top", "right", "bottom", "left", "tr", "br", "bl", "tl")
FLIP_KINDS = ("horizontal", "vertical", "main_diag", "anti_diag")
GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 60
MASK_ATTEMPTS = 80
PADDING = 1


def _reserve_marker_buffer(reserved: np.ndarray) -> None:
    for r, c in _edge_marker_positions(reserved.shape).values():
        for rr in range(max(0, r - 1), min(reserved.shape[0], r + 2)):
            for cc in range(max(0, c - 1), min(reserved.shape[1], c + 2)):
                reserved[rr, cc] = True


def _bbox_clear(reserved: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - PADDING)
    cc0 = max(0, c0 - PADDING)
    rr1 = min(reserved.shape[0] - 1, r1 + PADDING)
    cc1 = min(reserved.shape[1] - 1, c1 + PADDING)
    return not bool(np.any(reserved[rr0 : rr1 + 1, cc0 : cc1 + 1]))


def _reserve_bbox(reserved: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - PADDING)
    cc0 = max(0, c0 - PADDING)
    rr1 = min(reserved.shape[0] - 1, r1 + PADDING)
    cc1 = min(reserved.shape[1] - 1, c1 + PADDING)
    reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _monotone_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    out = [(r, c)]
    while (r, c) != end:
        moves = []
        if r < end[0]:
            moves.append((1, 0))
        elif r > end[0]:
            moves.append((-1, 0))
        if c < end[1]:
            moves.append((0, 1))
        elif c > end[1]:
            moves.append((0, -1))
        dr, dc = random.choice(moves)
        r += dr
        c += dc
        out.append((r, c))
    return out


def _arrow_core(h: int, w: int) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    arrow_r = random.randint(1, h - 4)
    arrow_c = random.randint(1, w - 4)
    block = {(arrow_r + dr, arrow_c + dc) for dr in range(2) for dc in range(2)}
    ring = {
        (arrow_r - 1, arrow_c),
        (arrow_r - 1, arrow_c + 1),
        (arrow_r + 2, arrow_c),
        (arrow_r + 2, arrow_c + 1),
        (arrow_r, arrow_c - 1),
        (arrow_r + 1, arrow_c - 1),
        (arrow_r, arrow_c + 2),
        (arrow_r + 1, arrow_c + 2),
    }
    return block | ring, (arrow_r, arrow_c)


def _grow_mask(cells: set[tuple[int, int]], h: int, w: int, target: int) -> None:
    while len(cells) < target:
        frontier = []
        weights = []
        for r in range(h):
            for c in range(w):
                if (r, c) in cells:
                    continue
                deg = sum((nr, nc) in cells for nr, nc in _neighbors4(r, c))
                if deg == 0:
                    continue
                frontier.append((r, c))
                weights.append(float(deg * deg))
        if not frontier:
            return
        cells.add(random.choices(frontier, weights=weights, k=1)[0])


def _make_local_mask(h: int, w: int, flip_kind: str) -> tuple[np.ndarray, tuple[int, int]] | None:
    for _ in range(MASK_ATTEMPTS):
        cells, arrow_pos = _arrow_core(h, w)
        side_targets = {
            "top": (0, random.randint(0, w - 1)),
            "bottom": (h - 1, random.randint(0, w - 1)),
            "left": (random.randint(0, h - 1), 0),
            "right": (random.randint(0, h - 1), w - 1),
        }

        for target in side_targets.values():
            best_dist = min(abs(r - target[0]) + abs(c - target[1]) for r, c in cells)
            starts = [cell for cell in cells if abs(cell[0] - target[0]) + abs(cell[1] - target[1]) == best_dist]
            for pt in _monotone_path(random.choice(starts), target):
                cells.add(pt)

        min_area = max(len(cells) + 1, int(np.ceil(0.35 * h * w)))
        max_area = min(h * w - 1, int(np.floor(0.72 * h * w)))
        if min_area > max_area:
            max_area = min(h * w - 1, len(cells) + max(1, (h * w) // 8))
            min_area = min(len(cells) + 1, max_area)
        if min_area <= max_area:
            _grow_mask(cells, h, w, random.randint(min_area, max_area))

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        if not bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if np.array_equal(mask, _transform_mask(mask, flip_kind)):
            continue
        return mask, arrow_pos
    return None


def _size_candidates(contact: str, flip_kind: str, h: int, w: int) -> list[tuple[int, int]]:
    mid_r = h // 2
    mid_c = w // 2
    sizes: list[tuple[int, int]] = []

    if flip_kind in {"main_diag", "anti_diag"}:
        base_sizes = [(n, n) for n in range(5, min(h, w, 8) + 1)]
    else:
        base_sizes = [(hh, ww) for hh in range(5, min(h, 8) + 1) for ww in range(5, min(w, 9) + 1)]

    for hh, ww in base_sizes:
        if contact == "top":
            if hh < h and ww <= w - 2:
                sizes.append((hh, ww))
        elif contact == "bottom":
            if hh < h and ww <= w - 2:
                sizes.append((hh, ww))
        elif contact == "left":
            if ww < w and hh <= h - 2:
                sizes.append((hh, ww))
        elif contact == "right":
            if ww < w and hh <= h - 2:
                sizes.append((hh, ww))
        elif contact == "tl":
            if hh <= mid_r and ww <= mid_c:
                sizes.append((hh, ww))
        elif contact == "tr":
            if hh <= mid_r and ww <= w - mid_c - 1:
                sizes.append((hh, ww))
        elif contact == "br":
            if hh <= h - mid_r - 1 and ww <= w - mid_c - 1:
                sizes.append((hh, ww))
        else:
            if hh <= h - mid_r - 1 and ww <= mid_c:
                sizes.append((hh, ww))

    random.shuffle(sizes)
    return sizes


def _candidate_bboxes(contact: str, hh: int, ww: int, h: int, w: int) -> list[tuple[int, int, int, int]]:
    mid_r = h // 2
    mid_c = w // 2
    bboxes: list[tuple[int, int, int, int]] = []

    if contact == "top":
        for c0 in range(1, w - ww):
            if not (c0 <= mid_c <= c0 + ww - 1):
                bboxes.append((0, c0, hh - 1, c0 + ww - 1))
    elif contact == "bottom":
        r0 = h - hh
        for c0 in range(1, w - ww):
            if not (c0 <= mid_c <= c0 + ww - 1):
                bboxes.append((r0, c0, h - 1, c0 + ww - 1))
    elif contact == "left":
        for r0 in range(1, h - hh):
            if not (r0 <= mid_r <= r0 + hh - 1):
                bboxes.append((r0, 0, r0 + hh - 1, ww - 1))
    elif contact == "right":
        c0 = w - ww
        for r0 in range(1, h - hh):
            if not (r0 <= mid_r <= r0 + hh - 1):
                bboxes.append((r0, c0, r0 + hh - 1, w - 1))
    elif contact == "tl":
        bboxes.append((0, 0, hh - 1, ww - 1))
    elif contact == "tr":
        c0 = w - ww
        bboxes.append((0, c0, hh - 1, w - 1))
    elif contact == "br":
        r0 = h - hh
        c0 = w - ww
        bboxes.append((r0, c0, h - 1, w - 1))
    else:
        r0 = h - hh
        bboxes.append((r0, 0, h - 1, ww - 1))

    random.shuffle(bboxes)
    return bboxes


def _paint_arrow(patch: np.ndarray, arrow_pos: tuple[int, int], flip_kind: str) -> None:
    miss_r, miss_c = FLIP_TO_MISSING[flip_kind]
    ar, ac = arrow_pos
    for dr in range(2):
        for dc in range(2):
            if (dr, dc) != (miss_r, miss_c):
                patch[ar + dr, ac + dc] = ARROW


def _build_input_patch(mask: np.ndarray, color: int, arrow_pos: tuple[int, int], flip_kind: str) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(color)
    _paint_arrow(patch, arrow_pos, flip_kind)
    return patch


def _expected_patch(mask: np.ndarray, flip_kind: str, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[_transform_mask(mask, flip_kind)] = int(color)
    return patch


def _place_patch(canvas: np.ndarray, patch: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    assert patch.shape == (r1 - r0 + 1, c1 - c0 + 1)
    target = canvas[r0 : r1 + 1, c0 : c1 + 1]
    assert not bool(np.any((target != BG) & (patch != BG)))
    canvas[r0 : r1 + 1, c0 : c1 + 1][patch != BG] = patch[patch != BG]


def _gen_object(
    inp: np.ndarray,
    expected: np.ndarray,
    reserved: np.ndarray,
    contact: str,
    flip_kind: str,
    base_color: int,
    edge_colors: dict[str, int],
) -> bool:
    h, w = inp.shape
    sizes = _size_candidates(contact, flip_kind, h, w)
    for hh, ww in sizes[:PLACE_ATTEMPTS]:
        bbox_candidates = _candidate_bboxes(contact, hh, ww, h, w)
        for bbox in bbox_candidates[:PLACE_ATTEMPTS]:
            if not _bbox_clear(reserved, bbox):
                continue
            made = _make_local_mask(hh, ww, flip_kind)
            if made is None:
                continue
            mask, arrow_pos = made
            input_patch = _build_input_patch(mask, base_color, arrow_pos, flip_kind)
            touched = _touching_sides(bbox, inp.shape)
            final_color = _contact_color(touched, edge_colors)
            output_patch = _expected_patch(mask, flip_kind, final_color)
            _place_patch(inp, input_patch, bbox)
            _place_patch(expected, output_patch, bbox)
            _reserve_bbox(reserved, bbox)
            return True
    return False


def _choose_grid_shape(n_objects: int) -> tuple[int, int]:
    min_side = 11 if n_objects <= 2 else 15
    valid = [size for size in GRID_SIZES if size >= min_side]
    return random.choice(valid), random.choice(valid)


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.choices(OBJECT_COUNT_CHOICES, weights=OBJECT_COUNT_WEIGHTS, k=1)[0]
    h, w = _choose_grid_shape(n_objects)
    all_colors = random.sample(PALETTE, len(PALETTE))
    edge_palette = all_colors[:4]
    object_palette = all_colors[4 : 4 + n_objects]
    edge_colors = dict(zip(SIDES, edge_palette, strict=False))

    inp = np.full((h, w), BG, dtype=int)
    expected = np.full((h, w), BG, dtype=int)
    for side, pos in _edge_marker_positions(inp.shape).items():
        inp[pos] = edge_colors[side]
        expected[pos] = edge_colors[side]

    reserved = np.zeros((h, w), dtype=bool)
    _reserve_marker_buffer(reserved)

    contacts = list(CONTACT_TYPES)
    random.shuffle(contacts)
    contacts = contacts[:n_objects]
    contacts.sort(key=lambda name: 0 if len(name) == 2 else 1)

    flip_pool = list(FLIP_KINDS)
    random.shuffle(flip_pool)
    flips = [flip_pool[i % len(flip_pool)] for i in range(n_objects)]
    random.shuffle(flips)

    for contact, flip_kind, color in zip(contacts, flips, object_palette, strict=False):
        if not _gen_object(inp, expected, reserved, contact, flip_kind, color, edge_colors):
            return None

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
