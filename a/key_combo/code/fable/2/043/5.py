from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]
ShapeKey = tuple[tuple[int, int], bytes]

BG = 0
ALL_COLORS = tuple(range(1, 10))
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBORS8 = tuple(
    (dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)
)

SHAPE_SIZES = ((3, 3), (3, 4), (4, 3), (4, 4), (3, 5), (5, 3))
MIN_HOLE_AREA = 4
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 200


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def bbox(mask: np.ndarray) -> BBox | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return bbox_coords(coords)


def crop_bbox(grid: np.ndarray, box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = map(int, box)
    return np.asarray(grid)[r0 : r1 + 1, c0 : c1 + 1].copy()


def components_mask(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    neighbors = NEIGHBORS4 if int(connectivity) == 4 else NEIGHBORS8

    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            cells: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in neighbors:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(cells, dtype=int))
    return components


def find_components(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    connectivity: int = 8,
    monochromatic: bool = True,
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g) if bg is None else bg)

    if monochromatic:
        components: list[dict] = []
        for color in sorted(set(map(int, g.ravel())) - {bg}):
            for coords in components_mask(g == color, connectivity=connectivity):
                components.append(
                    {
                        "color": int(color),
                        "coords": coords,
                        "bbox": bbox_coords(coords),
                        "area": int(coords.shape[0]),
                    }
                )
        return components

    fg = g != bg
    return [
        {
            "color": None,
            "coords": coords,
            "bbox": bbox_coords(coords),
            "area": int(coords.shape[0]),
        }
        for coords in components_mask(fg, connectivity=connectivity)
    ]


def _shape_key(mask: np.ndarray) -> ShapeKey:
    mask = np.asarray(mask, dtype=bool)
    return (tuple(map(int, mask.shape)), mask.astype(np.uint8, copy=False).tobytes())


def _tight_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _touches_edge(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = map(int, shape)
    return bool(
        np.any(
            (coords[:, 0] == 0)
            | (coords[:, 1] == 0)
            | (coords[:, 0] == h - 1)
            | (coords[:, 1] == w - 1)
        )
    )


def _extract_holes(grid: np.ndarray, host: dict, bg: int) -> list[dict]:
    hr0, hc0, hr1, hc1 = map(int, host["bbox"])
    patch = crop_bbox(grid, host["bbox"])
    holes: list[dict] = []
    for local in components_mask(patch == int(bg), connectivity=8):
        if _touches_edge(local, patch.shape):
            continue
        local_mask = _tight_mask(local)
        global_coords = local + np.array([hr0, hc0], dtype=int)
        holes.append(
            {
                "coords": global_coords,
                "mask": local_mask,
                "shape": _shape_key(local_mask),
                "area": int(local_mask.sum()),
            }
        )
    return holes


def _extract_pieces(grid: np.ndarray, components: list[dict], host: dict, bg: int) -> list[dict]:
    pieces: list[dict] = []
    for comp in components:
        if comp is host:
            continue
        patch = crop_bbox(grid, comp["bbox"])
        color = int(comp["color"])
        positive = patch == color
        if not np.all((patch == int(bg)) | positive):
            raise AssertionError("piece bbox must contain only its own color and background")
        negative = ~positive
        if not bool(negative.any()):
            raise AssertionError("piece bbox complement must be non-empty")
        pieces.append(
            {
                "color": color,
                "coords": comp["coords"],
                "negative": negative,
                "shape": _shape_key(negative),
            }
        )
    return pieces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each host hole with the color of the uniquely matching negative-shape piece.

    Input structure:
    - Black is the background; the largest non-black component is a single-color host object.
    - Inside the host are one or more black holes.
    - Outside the host are separate single-color pieces, one per hole.

    Rule:
    - Extract every enclosed black hole inside the host.
    - For each outside piece, crop its bbox and treat the unpainted cells inside that bbox as
      the true matching shape; compare that negative shape to the hole shapes without rotation
      or reflection.
    - Fill each hole with the matched piece color, erase every outside piece to background, and
      leave the host otherwise unchanged.

    Key insight:
    - The painted part of each piece is a decoy; only the bbox complement of the piece matches
      the target hole.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = int(get_bg(g))
    components = find_components(g, bg=bg, connectivity=8, monochromatic=True)
    assert components, "expected a host object"
    host = max(components, key=lambda comp: int(comp["area"]))

    holes = _extract_holes(g, host, bg)
    pieces = _extract_pieces(g, components, host, bg)
    assert holes, "expected at least one enclosed hole"
    assert len(holes) == len(pieces), "hole count and piece count must match"

    piece_by_shape: dict[ShapeKey, dict] = {}
    for piece in pieces:
        key = piece["shape"]
        assert key not in piece_by_shape, "piece negative shapes must be unique"
        piece_by_shape[key] = piece

    hole_shapes = {hole["shape"] for hole in holes}
    assert hole_shapes == set(piece_by_shape), "holes and piece negative shapes must match exactly"

    out = g.copy()
    for piece in pieces:
        coords = piece["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    for hole in holes:
        matched = piece_by_shape[hole["shape"]]
        coords = hole["coords"]
        out[coords[:, 0], coords[:, 1]] = int(matched["color"])

    return out


def _is_connected(mask: np.ndarray, connectivity: int = 8) -> bool:
    return len(components_mask(mask, connectivity=connectivity)) == 1


def _valid_hole_mask(mask: np.ndarray) -> bool:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    area = int(mask.sum())
    total = int(mask.size)
    if area < MIN_HOLE_AREA or area > total - MIN_HOLE_AREA:
        return False
    if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
        return False
    if mask[0].all() or mask[-1].all() or mask[:, 0].all() or mask[:, -1].all():
        return False
    if not _is_connected(mask, connectivity=8):
        return False
    if not _is_connected(~mask, connectivity=8):
        return False
    return True


def _build_hole_library() -> dict[tuple[int, int, int], list[np.ndarray]]:
    buckets: dict[tuple[int, int, int], list[np.ndarray]] = defaultdict(list)
    for h, w in SHAPE_SIZES:
        total = h * w
        for bits in range(1 << total):
            if bits.bit_count() < MIN_HOLE_AREA or total - bits.bit_count() < MIN_HOLE_AREA:
                continue
            flat = np.array([(bits >> i) & 1 for i in range(total)], dtype=bool)
            mask = flat.reshape(h, w)
            if _valid_hole_mask(mask):
                buckets[(h, w, int(mask.sum()))].append(mask.copy())
    return {bucket: masks for bucket, masks in buckets.items() if len(masks) >= 2}


HOLE_LIBRARY = _build_hole_library()
AMBIGUOUS_BUCKETS = tuple(sorted(HOLE_LIBRARY))
ALL_LIBRARY_ITEMS = tuple(
    (bucket, mask.copy())
    for bucket, masks in sorted(HOLE_LIBRARY.items())
    for mask in masks
)


def _choose_hole_masks(count: int) -> list[np.ndarray]:
    same_bucket_count = 2 if count == 2 else random.choice((2, 2, 3))
    candidate_buckets = [
        bucket for bucket in AMBIGUOUS_BUCKETS if len(HOLE_LIBRARY[bucket]) >= same_bucket_count
    ]
    shared_bucket = random.choice(candidate_buckets)

    chosen = [mask.copy() for mask in random.sample(HOLE_LIBRARY[shared_bucket], same_bucket_count)]
    hole_keys = {_shape_key(mask) for mask in chosen}
    positive_keys = {_shape_key(~mask) for mask in chosen}

    while len(chosen) < count:
        _bucket, mask = random.choice(ALL_LIBRARY_ITEMS)
        mask = mask.copy()
        hole_key = _shape_key(mask)
        positive_key = _shape_key(~mask)
        if hole_key in hole_keys or hole_key in positive_keys or positive_key in hole_keys:
            continue
        chosen.append(mask)
        hole_keys.add(hole_key)
        positive_keys.add(positive_key)

    random.shuffle(chosen)
    return chosen


def _can_place_bbox(reserved: np.ndarray, r: int, c: int, h: int, w: int, pad: int) -> bool:
    hh, ww = reserved.shape
    if r < 0 or c < 0 or r + h > hh or c + w > ww:
        return False
    r0 = max(0, r - pad)
    c0 = max(0, c - pad)
    r1 = min(hh, r + h + pad)
    c1 = min(ww, c + w + pad)
    return not bool(reserved[r0:r1, c0:c1].any())


def _reserve_bbox(reserved: np.ndarray, r: int, c: int, h: int, w: int, pad: int) -> None:
    hh, ww = reserved.shape
    r0 = max(0, r - pad)
    c0 = max(0, c - pad)
    r1 = min(hh, r + h + pad)
    c1 = min(ww, c + w + pad)
    reserved[r0:r1, c0:c1] = True


def _place_bboxes(
    canvas_shape: tuple[int, int],
    box_shapes: list[tuple[int, int]],
    *,
    border: int,
    pad: int,
    attempts: int,
) -> list[tuple[int, int]] | None:
    hh, ww = map(int, canvas_shape)
    reserved = np.zeros((hh, ww), dtype=bool)
    order = sorted(range(len(box_shapes)), key=lambda idx: -(box_shapes[idx][0] * box_shapes[idx][1]))
    placed: list[tuple[int, int] | None] = [None] * len(box_shapes)

    for idx in order:
        h, w = map(int, box_shapes[idx])
        low_r = int(border)
        low_c = int(border)
        high_r = hh - h - int(border)
        high_c = ww - w - int(border)
        if high_r < low_r or high_c < low_c:
            return None

        for _ in range(attempts):
            r = random.randint(low_r, high_r)
            c = random.randint(low_c, high_c)
            if not _can_place_bbox(reserved, r, c, h, w, pad):
                continue
            _reserve_bbox(reserved, r, c, h, w, pad)
            placed[idx] = (int(r), int(c))
            break
        else:
            return None

    return [(int(r), int(c)) for r, c in placed if r is not None and c is not None]


def _render_host(host_shape: tuple[int, int], host_color: int, holes: list[tuple[np.ndarray, tuple[int, int]]]) -> np.ndarray:
    host = np.full(host_shape, int(host_color), dtype=int)
    for hole_mask, (r0, c0) in holes:
        h, w = hole_mask.shape
        view = host[r0 : r0 + h, c0 : c0 + w]
        view[hole_mask] = BG
    return host


def _piece_patch(hole_mask: np.ndarray, color: int) -> np.ndarray:
    return np.where(~hole_mask, int(color), BG).astype(int)


def _explicit_output(
    input_grid: np.ndarray,
    host_pos: tuple[int, int],
    hole_specs: list[tuple[np.ndarray, tuple[int, int], int]],
    piece_specs: list[tuple[np.ndarray, tuple[int, int]]],
) -> np.ndarray:
    out = np.asarray(input_grid, dtype=int).copy()
    for piece_patch, (r0, c0) in piece_specs:
        ph, pw = piece_patch.shape
        piece_mask = piece_patch != BG
        out[r0 : r0 + ph, c0 : c0 + pw][piece_mask] = BG

    host_r, host_c = host_pos
    for hole_mask, (r0, c0), color in hole_specs:
        h, w = hole_mask.shape
        out[host_r + r0 : host_r + r0 + h, host_c + c0 : host_c + c0 + w][hole_mask] = int(color)
    return out


def _semantic_checks(
    input_grid: np.ndarray,
    expected: np.ndarray,
    chosen_holes: list[np.ndarray],
    hole_specs: list[tuple[np.ndarray, tuple[int, int], int]],
    piece_specs: list[tuple[np.ndarray, tuple[int, int]]],
) -> bool:
    if not np.array_equal(main(input_grid), expected):
        return False

    hole_keys = [_shape_key(mask) for mask in chosen_holes]
    if len(set(hole_keys)) != len(hole_keys):
        return False

    same_size_counts = Counter((mask.shape[0], mask.shape[1], int(mask.sum())) for mask in chosen_holes)
    if not any(count >= 2 for count in same_size_counts.values()):
        return False

    positive_keys = [_shape_key(piece_patch != BG) for piece_patch, _pos in piece_specs]
    negative_keys = [_shape_key(piece_patch == BG) for piece_patch, _pos in piece_specs]
    if set(negative_keys) != set(hole_keys):
        return False
    if any(key in set(hole_keys) for key in positive_keys):
        return False

    components = find_components(input_grid, bg=BG, connectivity=8, monochromatic=True)
    host = max(components, key=lambda comp: int(comp["area"]))
    holes = _extract_holes(input_grid, host, BG)
    pieces = _extract_pieces(input_grid, components, host, BG)
    if len(holes) != len(chosen_holes) or len(pieces) != len(chosen_holes):
        return False
    if {hole["shape"] for hole in holes} != set(hole_keys):
        return False
    if {piece["shape"] for piece in pieces} != set(hole_keys):
        return False

    expected_colors = {int(color) for _mask, _pos, color in hole_specs}
    actual_colors = {int(piece["color"]) for piece in pieces}
    return expected_colors == actual_colors


def _gen_attempt() -> np.ndarray | None:
    hole_count = random.choice((2, 3, 4))
    hole_masks = _choose_hole_masks(hole_count)
    hole_boxes = [tuple(map(int, mask.shape)) for mask in hole_masks]

    max_h = max(h for h, _w in hole_boxes)
    max_w = max(w for _h, w in hole_boxes)
    host_h = random.randint(max(10, max_h + 6), min(16, max(10, max_h + 9)))
    host_w = random.randint(max(11, max_w + 7), min(19, max(11, max_w + 11)))

    hole_positions = _place_bboxes(
        (host_h, host_w),
        hole_boxes,
        border=1,
        pad=1,
        attempts=PLACEMENT_ATTEMPTS,
    )
    if hole_positions is None:
        return None

    host_color, *piece_colors = random.sample(ALL_COLORS, hole_count + 1)
    hole_specs = [
        (mask, pos, color)
        for mask, pos, color in zip(hole_masks, hole_positions, piece_colors, strict=True)
    ]
    host_patch = _render_host((host_h, host_w), host_color, [(mask, pos) for mask, pos, _color in hole_specs])

    piece_patches = [_piece_patch(mask, color) for mask, _pos, color in hole_specs]
    piece_specs_shuffled = list(zip(piece_patches, piece_colors, strict=True))
    random.shuffle(piece_specs_shuffled)
    outer_boxes = [host_patch.shape] + [tuple(map(int, patch.shape)) for patch, _color in piece_specs_shuffled]

    piece_bbox_area = sum(h * w for h, w in outer_boxes[1:])
    min_canvas_h = max(host_h + 6, max(max_h + 4, 15))
    min_canvas_w = max(host_w + 6, max(max_w + 4, 15))
    max_canvas_h = min(30, max(min_canvas_h, host_h + 12))
    max_canvas_w = min(30, max(min_canvas_w, host_w + 12))

    for _ in range(PLACEMENT_ATTEMPTS):
        grid_h = random.randint(min_canvas_h, max_canvas_h)
        grid_w = random.randint(min_canvas_w, max_canvas_w)
        if grid_h * grid_w - host_h * host_w < piece_bbox_area + 25:
            continue

        outer_positions = _place_bboxes(
            (grid_h, grid_w),
            outer_boxes,
            border=1,
            pad=1,
            attempts=PLACEMENT_ATTEMPTS,
        )
        if outer_positions is None:
            continue

        host_pos = outer_positions[0]
        piece_positions = outer_positions[1:]
        input_grid = np.full((grid_h, grid_w), BG, dtype=int)

        host_r, host_c = host_pos
        input_grid[host_r : host_r + host_h, host_c : host_c + host_w] = host_patch

        placed_piece_specs: list[tuple[np.ndarray, tuple[int, int]]] = []
        for (piece_patch, _color), (r0, c0) in zip(piece_specs_shuffled, piece_positions, strict=True):
            ph, pw = piece_patch.shape
            piece_mask = piece_patch != BG
            input_grid[r0 : r0 + ph, c0 : c0 + pw][piece_mask] = piece_patch[piece_mask]
            placed_piece_specs.append((piece_patch, (r0, c0)))

        expected = _explicit_output(input_grid, host_pos, hole_specs, placed_piece_specs)
        if _semantic_checks(input_grid, expected, hole_masks, hole_specs, placed_piece_specs):
            return input_grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid negative-shape hole puzzle")
