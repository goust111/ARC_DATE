import random
from typing import Iterable

import numpy as np


BG = 0
FRAME_COLOR = 7
SPRITE_COLORS = (1, 2, 3, 4)
GENERAL_LOGICAL_SHAPES = (
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
)
NON_MONOTONIC_SIDE_PATTERNS = (
    ("L", "R", "L", "R"),
    ("R", "L", "R", "L"),
    ("L", "R", "R", "L"),
    ("R", "L", "L", "R"),
)
MAX_GENERATION_ATTEMPTS = 500


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _solidify_sparse_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    rel = coords - np.array([r0, c0])
    assert np.all(rel % 2 == 0), "sparse sprite pixels must lie on a two-step lattice"

    logical_h = (r1 - r0) // 2 + 1
    logical_w = (c1 - c0) // 2 + 1
    logical = np.zeros((logical_h, logical_w), dtype=bool)
    for dr, dc in rel:
        logical[int(dr) // 2, int(dc) // 2] = True

    return np.kron(logical, np.ones((2, 2), dtype=bool))[: 2 * logical_h - 1, : 2 * logical_w - 1]


def _solidify_logical(logical: np.ndarray) -> np.ndarray:
    logical = np.asarray(logical, dtype=bool)
    return np.kron(logical, np.ones((2, 2), dtype=bool))[: 2 * logical.shape[0] - 1, : 2 * logical.shape[1] - 1]


def _sparse_rect_perimeter_ok(coords: np.ndarray) -> bool:
    if coords.size == 0:
        return False
    r0, c0, r1, c1 = _bbox(coords)
    rel = coords - np.array([r0, c0])
    if np.any(rel % 2 != 0):
        return False

    logical_h = (r1 - r0) // 2 + 1
    logical_w = (c1 - c0) // 2 + 1
    if logical_h < 2 or logical_w < 2:
        return False

    expected = 2 * logical_h + 2 * logical_w - 4
    if len(coords) != expected:
        return False

    points = {tuple(map(int, pt)) for pt in coords}
    for r in range(r0, r1 + 1, 2):
        for c in range(c0, c1 + 1, 2):
            on_border = r in (r0, r1) or c in (c0, c1)
            if ((r, c) in points) != on_border:
                return False
    return True


def _paint_mask(dst: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    view[mask] = int(color)


def _extract_frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == FRAME_COLOR)
    assert coords.size > 0, "expected an orange sparse frame"
    assert _sparse_rect_perimeter_ok(coords), "orange pixels must form one sparse rectangular perimeter"
    bbox = _bbox(coords)
    assert (bbox[2] - bbox[0]) > (bbox[3] - bbox[1]), "frame must be taller than it is wide"
    return bbox


def _sprite_from_grid(grid: np.ndarray, color: int, frame_bbox: tuple[int, int, int, int]) -> dict:
    coords = np.argwhere(grid == color)
    assert coords.size > 0, f"missing sprite color {color}"
    bbox = _bbox(coords)
    assert not _boxes_overlap(bbox, frame_bbox), "sprites must lie outside the frame bbox"
    dense = _solidify_sparse_coords(coords)
    center_r = float(np.median(coords[:, 0]))
    center_c = float(np.median(coords[:, 1]))
    return {
        "color": int(color),
        "coords": coords,
        "bbox": bbox,
        "dense": dense,
        "center_r": center_r,
        "center_c": center_c,
    }


def _ordered_sprites_from_grid(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[dict]]:
    frame_bbox = _extract_frame_bbox(grid)
    colors = [int(c) for c in np.unique(grid) if int(c) not in (BG, FRAME_COLOR)]
    assert len(colors) == 4, "expected exactly four external sprites"
    sprites = [_sprite_from_grid(grid, color, frame_bbox) for color in colors]

    row_keys = [(sp["center_r"], sp["center_c"]) for sp in sprites]
    col_keys = [(sp["center_c"], sp["center_r"]) for sp in sprites]
    assert len(set(row_keys)) == 4, "sprite centers must be row-distinct"
    assert len(set(col_keys)) == 4, "sprite centers must be col-distinct"
    sprites.sort(key=lambda sp: (sp["center_r"], sp["center_c"]))
    return frame_bbox, sprites


def _render_output(shape: tuple[int, int], frame_bbox: tuple[int, int, int, int], sprites: Iterable[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    r0, c0, r1, c1 = frame_bbox
    out[r0, c0 : c1 + 1] = FRAME_COLOR
    out[r1, c0 : c1 + 1] = FRAME_COLOR
    out[r0 : r1 + 1, c0] = FRAME_COLOR
    out[r0 : r1 + 1, c1] = FRAME_COLOR

    sprites = list(sprites)
    interior_h = r1 - r0 - 1
    interior_w = c1 - c0 - 1
    rotated_masks = [np.rot90(sp["dense"], -rank) for rank, sp in enumerate(sprites)]

    total_h = sum(mask.shape[0] for mask in rotated_masks)
    slack = interior_h - total_h
    assert slack > 0 and slack % (len(sprites) + 1) == 0, "frame height must yield equal vertical gaps"
    gap = slack // (len(sprites) + 1)

    cursor = r0 + 1 + gap
    for sp, mask in zip(sprites, rotated_masks):
        ph, pw = mask.shape
        cross_slack = interior_w - pw
        assert cross_slack >= 0 and cross_slack % 2 == 0, "sprite must center horizontally"
        cc = c0 + 1 + cross_slack // 2
        _paint_mask(out, mask, cursor, cc, sp["color"])
        cursor += ph + gap

    return out


def _render_output_no_rotation(shape: tuple[int, int], frame_bbox: tuple[int, int, int, int], sprites: Iterable[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    r0, c0, r1, c1 = frame_bbox
    out[r0, c0 : c1 + 1] = FRAME_COLOR
    out[r1, c0 : c1 + 1] = FRAME_COLOR
    out[r0 : r1 + 1, c0] = FRAME_COLOR
    out[r0 : r1 + 1, c1] = FRAME_COLOR

    sprites = list(sprites)
    interior_h = r1 - r0 - 1
    interior_w = c1 - c0 - 1
    masks = [sp["dense"] for sp in sprites]
    total_h = sum(mask.shape[0] for mask in masks)
    slack = interior_h - total_h
    assert slack > 0 and slack % (len(sprites) + 1) == 0
    gap = slack // (len(sprites) + 1)

    cursor = r0 + 1 + gap
    for sp, mask in zip(sprites, masks):
        ph, pw = mask.shape
        cross_slack = interior_w - pw
        assert cross_slack >= 0 and cross_slack % 2 == 0
        cc = c0 + 1 + cross_slack // 2
        _paint_mask(out, mask, cursor, cc, sp["color"])
        cursor += ph + gap
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Densify the sparse outer sprites, rotate them by top-to-bottom rank, and pack them evenly into the sparse frame.

    Input structure:
    - Black background with one tall orange sparse rectangular frame whose border
      pixels appear every two cells along the perimeter.
    - Exactly four monochrome sparse sprites outside the frame, one each in red,
      blue, green, and yellow, with sprite pixels also lying on a two-step lattice.
    - The four sprite center rows and center columns are all distinct, so their
      positional ordering is unambiguous.

    Rule:
    - Recover the orange frame bbox and thicken that sparse border into a solid
      one-pixel rectangular border.
    - For each non-frame sprite, fill the gaps implied by its two-step lattice so
      the sparse sprite becomes a contiguous dense mask.
    - Sort the sprites from top to bottom by the median of their input row
      coordinates; sprite rank k rotates clockwise by k quarter-turns.
    - Clear the frame interior and all external sprite pixels, then place the
      rotated dense sprites inside the frame from top to bottom with equal vertical
      gaps, using the same gap above the first sprite and below the last one; each
      sprite is horizontally centered and keeps its original color.

    Key insight:
    - The external layout encodes instructions: vertical source order determines
      rotation amount, while the sparse lattice geometry determines the dense
      sprite body that must be inserted into the frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    frame_bbox, sprites = _ordered_sprites_from_grid(g)
    return _render_output(g.shape, frame_bbox, sprites)


def _sprite_metadata(logical: np.ndarray, color: int) -> dict:
    logical = np.asarray(logical, dtype=bool)
    rel = np.argwhere(logical).astype(int) * 2
    rows = rel[:, 0]
    cols = rel[:, 1]
    bbox_h = int(rows.max() - rows.min() + 1)
    bbox_w = int(cols.max() - cols.min() + 1)
    dense = _solidify_logical(logical)
    return {
        "color": int(color),
        "logical": logical,
        "rel_sparse": rel,
        "bbox_h": bbox_h,
        "bbox_w": bbox_w,
        "median_off_r": float(np.median(rows)),
        "median_off_c": float(np.median(cols)),
        "dense": dense,
    }


def _random_rank_sprites() -> list[dict] | None:
    colors = random.sample(list(SPRITE_COLORS), 4)
    sprites = []
    dense_signatures = set()

    for rank in range(4):
        candidates = []
        for template in GENERAL_LOGICAL_SHAPES:
            for turns in range(4):
                logical = np.rot90(template, turns)
                dense = _solidify_logical(logical)
                if rank > 0 and np.array_equal(np.rot90(dense, -rank), dense):
                    continue
                candidates.append(logical)
        if not candidates:
            return None
        chosen = random.choice(candidates)
        sprite = _sprite_metadata(chosen, colors[rank])
        sprites.append(sprite)
        dense_signatures.add(sprite["dense"].tobytes())

    if len(dense_signatures) < 2:
        return None
    return sprites


def _choose_frame_dims(rank_sprites: list[dict]) -> tuple[int, int, int] | None:
    total_h = sum(np.rot90(sp["dense"], -rank).shape[0] for rank, sp in enumerate(rank_sprites))
    gap_choices = [gap for gap in (1, 3) if total_h + 5 * gap + 2 <= 25]
    if not gap_choices:
        return None
    gap = random.choice(gap_choices)
    interior_h = total_h + 5 * gap

    max_w = max(np.rot90(sp["dense"], -rank).shape[1] for rank, sp in enumerate(rank_sprites))
    width_choices = [w for w in (5, 7, 9) if w >= max_w and w < interior_h]
    if not width_choices:
        return None
    interior_w = random.choice(width_choices)
    return interior_h + 2, interior_w + 2, gap


def _choose_row_anchors(rank_sprites: list[dict], H: int) -> list[int] | None:
    for _ in range(250):
        meds = sorted(random.sample(range(H), 4))
        anchors = []
        ok = True
        for med_r, sp in zip(meds, rank_sprites):
            top = int(round(med_r - sp["median_off_r"]))
            if top < 0 or top + sp["bbox_h"] > H:
                ok = False
                break
            anchors.append(top)
        if ok:
            return anchors
    return None


def _choose_col_anchors(
    rank_sprites: list[dict],
    W: int,
    frame_bbox: tuple[int, int, int, int],
    row_anchors: list[int],
) -> list[int] | None:
    r0, c0, r1, c1 = frame_bbox
    pattern = random.choice(NON_MONOTONIC_SIDE_PATTERNS)

    feasible_cols = []
    for side, sp in zip(pattern, rank_sprites):
        if side == "L":
            start = 0
            stop = c0 - sp["bbox_w"] - 1
        else:
            start = c1 + 2
            stop = W - sp["bbox_w"]
        if start > stop:
            return None
        feasible_cols.append(list(range(start, stop + 1)))

    for _ in range(500):
        anchors = [random.choice(choices) for choices in feasible_cols]
        centers_c = []
        boxes = []
        ok = True
        for top, left, sp in zip(row_anchors, anchors, rank_sprites):
            box = (top, left, top + sp["bbox_h"] - 1, left + sp["bbox_w"] - 1)
            if _boxes_overlap(box, frame_bbox, pad=1):
                ok = False
                break
            boxes.append(box)
            centers_c.append(sp["median_off_c"] + left)

        if not ok or len(set(centers_c)) != 4:
            continue

        if any(_boxes_overlap(boxes[i], boxes[j], pad=1) for i in range(4) for j in range(i + 1, 4)):
            continue

        col_order = sorted(range(4), key=lambda k: (centers_c[k], row_anchors[k]))
        if col_order == [0, 1, 2, 3]:
            continue
        return anchors

    return None


def _stamp_sparse_frame(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = frame_bbox
    grid[r0 : r1 + 1 : 2, c0] = FRAME_COLOR
    grid[r0 : r1 + 1 : 2, c1] = FRAME_COLOR
    grid[r0, c0 : c1 + 1 : 2] = FRAME_COLOR
    grid[r1, c0 : c1 + 1 : 2] = FRAME_COLOR


def _build_input(
    H: int,
    W: int,
    frame_bbox: tuple[int, int, int, int],
    rank_sprites: list[dict],
    row_anchors: list[int],
    col_anchors: list[int],
) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    _stamp_sparse_frame(grid, frame_bbox)
    for top, left, sp in zip(row_anchors, col_anchors, rank_sprites):
        rr = top + sp["rel_sparse"][:, 0]
        cc = left + sp["rel_sparse"][:, 1]
        grid[rr, cc] = sp["color"]
    return grid


def _semantic_ok(inp: np.ndarray) -> bool:
    frame_bbox, ordered = _ordered_sprites_from_grid(inp)
    expected = _render_output(inp.shape, frame_bbox, ordered)

    by_col = sorted(ordered, key=lambda sp: (sp["center_c"], sp["center_r"]))
    try:
        no_rotation = _render_output_no_rotation(inp.shape, frame_bbox, ordered)
    except AssertionError:
        no_rotation = None
    try:
        wrong_order = _render_output(inp.shape, frame_bbox, by_col)
    except AssertionError:
        wrong_order = None

    if no_rotation is not None and np.array_equal(expected, no_rotation):
        return False
    if wrong_order is not None and np.array_equal(expected, wrong_order):
        return False
    if not np.array_equal(main(inp), expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rank_sprites = _random_rank_sprites()
    if rank_sprites is None:
        return None

    frame_dims = _choose_frame_dims(rank_sprites)
    if frame_dims is None:
        return None
    frame_h, frame_w, _gap = frame_dims

    left_margin = random.randint(6, 8)
    right_margin = random.randint(6, 8)
    W = frame_w + left_margin + right_margin
    if W > 30:
        return None

    H = min(30, max(18, frame_h) + random.randint(0, 4))
    if H < frame_h:
        return None

    frame_r0 = random.randint(0, H - frame_h)
    frame_c0 = left_margin
    frame_bbox = (frame_r0, frame_c0, frame_r0 + frame_h - 1, frame_c0 + frame_w - 1)

    row_anchors = _choose_row_anchors(rank_sprites, H)
    if row_anchors is None:
        return None
    col_anchors = _choose_col_anchors(rank_sprites, W, frame_bbox, row_anchors)
    if col_anchors is None:
        return None

    inp = _build_input(H, W, frame_bbox, rank_sprites, row_anchors, col_anchors)
    if not _semantic_ok(inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample for puzzle 9")
