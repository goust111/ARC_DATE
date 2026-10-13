from __future__ import annotations

import numpy as np


BG = 0
FRAME_COLOR = 3
MARKER_COLOR = 7
SPRITE_COLOR_POOL = (1, 2, 4, 5)
BAND_COLOR_POOL = (6, 8, 9)
GEN_ATTEMPTS = 400


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _dense_mask_from_sparse_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    rel = coords - np.array([r0, c0], dtype=int)
    assert np.all(rel % 2 == 0), "sparse sprite must lie on a two-step lattice"
    logical_h = (r1 - r0) // 2 + 1
    logical_w = (c1 - c0) // 2 + 1
    logical = np.zeros((logical_h, logical_w), dtype=bool)
    for dr, dc in rel:
        logical[int(dr) // 2, int(dc) // 2] = True
    return np.kron(logical, np.ones((2, 2), dtype=bool))[: 2 * logical_h - 1, : 2 * logical_w - 1]


def _place_dense_sprites(
    frame_bbox: tuple[int, int, int, int],
    sprites: list[dict[str, object]],
) -> list[dict[str, object]]:
    r0, c0, r1, c1 = frame_bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    ordered = sorted(sprites, key=lambda sp: (int(sp["bbox"][1]), int(sp["bbox"][0])))
    total_w = sum(int(np.asarray(sp["mask"]).shape[1]) for sp in ordered)
    gap = (inner_w - total_w) // (len(ordered) + 1)
    assert gap >= 1, "frame interior must leave at least one column between sprites"

    placements: list[dict[str, object]] = []
    cursor = c0 + 1 + gap
    for sp in ordered:
        mask = np.asarray(sp["mask"], dtype=bool)
        h, w = mask.shape
        assert h <= inner_h, "sprite must fit inside frame interior height"
        placements.append(
            {
                "color": int(sp["color"]),
                "bbox": tuple(sp["bbox"]),
                "mask": mask,
                "top": r0 + 1,
                "left": int(cursor),
            }
        )
        cursor += w + gap
    return placements


def _stamp_dense(out: np.ndarray, color: int, top: int, left: int, mask: np.ndarray) -> None:
    rows, cols = np.where(mask)
    out[top + rows, left + cols] = int(color)


def _inserted_sprite_mask(shape: tuple[int, int], placements: list[dict[str, object]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for sp in placements:
        sprite_mask = np.asarray(sp["mask"], dtype=bool)
        top = int(sp["top"])
        left = int(sp["left"])
        rows, cols = np.where(sprite_mask)
        mask[top + rows, left + cols] = True
    return mask


def _sprite_specs_from_input(
    grid: np.ndarray, frame_bbox: tuple[int, int, int, int]
) -> list[dict[str, object]]:
    r0, c0, r1, c1 = frame_bbox
    outside = np.ones(grid.shape, dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    colors = sorted(set(map(int, grid[outside].ravel())) - {BG, FRAME_COLOR, MARKER_COLOR})
    sprites: list[dict[str, object]] = []
    for color in colors:
        coords = np.argwhere((grid == color) & outside)
        if coords.size == 0:
            continue
        sprites.append(
            {
                "color": int(color),
                "bbox": _bbox(coords),
                "mask": _dense_mask_from_sparse_coords(coords),
            }
        )
    assert 2 <= len(sprites) <= 3, "expected 2-3 external sparse sprites"
    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Thicken the sparse frame, insert the densified external sprites into it, and let marked columns propagate upward until a sprite blocks them.

    Input structure:
    - The grid has a black background, one green sparse rectangular frame on a two-step lattice, and 2-3 external monochrome sparse sprites.
    - Inside the frame sit 2-3 orange bottom markers, plus a few other colored pixels above them that act as band endpoints.
    - The frame is wider than it is tall, and at least one marked column will overlap the eventual column span of an inserted sprite.

    Rule:
    - Find the green sparse frame, clear its original sparse pixels and the external sparse sprite pixels, and replace the frame by a solid one-pixel green border on the same bounding box.
    - Densify each external sparse sprite by filling the two-step lattice gaps, then place the dense sprites inside the frame from left to right with equal horizontal gaps and top alignment.
    - For each orange marker column, scan upward from that marker while remembering the most recent non-black color; fill black cells with that remembered color, update the memory when another non-black endpoint is hit, and stop before the solid frame or before the first inserted sprite pixel in that column.

    Key insight:
    - The inserted sprites are not just decorative content inside the thickened frame: their placed column spans determine where the upward color propagation is cut off.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frame_coords = np.argwhere(g == FRAME_COLOR)
    assert frame_coords.size > 0, "expected a sparse green frame"
    frame_bbox = _bbox(frame_coords)

    out = g.copy()
    out[g == FRAME_COLOR] = BG

    sprites = _sprite_specs_from_input(g, frame_bbox)
    for sp in sprites:
        color = int(sp["color"])
        out[g == color] = BG

    r0, c0, r1, c1 = frame_bbox
    out[r0, c0 : c1 + 1] = FRAME_COLOR
    out[r1, c0 : c1 + 1] = FRAME_COLOR
    out[r0 : r1 + 1, c0] = FRAME_COLOR
    out[r0 : r1 + 1, c1] = FRAME_COLOR

    placed = _place_dense_sprites(frame_bbox, sprites)
    for sp in placed:
        _stamp_dense(out, int(sp["color"]), int(sp["top"]), int(sp["left"]), np.asarray(sp["mask"], dtype=bool))

    sprite_mask = _inserted_sprite_mask(g.shape, placed)
    markers = np.argwhere(g == MARKER_COLOR)
    marker_by_col: dict[int, int] = {}
    for r, c in markers:
        rr = int(r)
        cc = int(c)
        if r0 < rr < r1 and c0 < cc < c1:
            marker_by_col[cc] = max(rr, marker_by_col.get(cc, -1))
    assert 2 <= len(marker_by_col) <= 3, "expected 2-3 markers inside the frame"

    for c, marker_r in sorted(marker_by_col.items()):
        current = MARKER_COLOR
        for r in range(marker_r - 1, r0, -1):
            if sprite_mask[r, c]:
                break
            value = int(out[r, c])
            if value == BG:
                out[r, c] = current
            else:
                current = value

    return out


SPRITE_TEMPLATES = (
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
)


def _dense_shape_from_logical(logical: np.ndarray) -> np.ndarray:
    return np.kron(logical.astype(bool), np.ones((2, 2), dtype=bool))[: 2 * logical.shape[0] - 1, : 2 * logical.shape[1] - 1]


def _draw_sparse_frame(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = frame_bbox
    grid[r0, c0 : c1 + 1 : 2] = FRAME_COLOR
    grid[r1, c0 : c1 + 1 : 2] = FRAME_COLOR
    grid[r0 : r1 + 1 : 2, c0] = FRAME_COLOR
    grid[r0 : r1 + 1 : 2, c1] = FRAME_COLOR


def _stamp_sparse_sprite(grid: np.ndarray, logical: np.ndarray, top: int, left: int, color: int) -> None:
    rows, cols = np.where(logical)
    grid[top + 2 * rows, left + 2 * cols] = int(color)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + margin < br0
        or br1 + margin < ar0
        or ac1 + margin < bc0
        or bc1 + margin < ac0
    )


def _sample_external_boxes(
    rng: np.random.Generator,
    shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    logical_masks: list[np.ndarray],
) -> list[tuple[int, int]] | None:
    h, w = shape
    placed_boxes: list[tuple[int, int, int, int]] = []
    anchors: list[tuple[int, int]] = []

    for logical in logical_masks:
        dense = _dense_shape_from_logical(logical)
        hh, ww = dense.shape
        candidates = [
            (r, c)
            for r in range(0, h - hh + 1, 2)
            for c in range(0, w - ww + 1, 2)
        ]
        rng.shuffle(candidates)
        picked = None
        for top, left in candidates:
            box = (top, left, top + hh - 1, left + ww - 1)
            if _boxes_overlap(box, frame_bbox, margin=1):
                continue
            if any(_boxes_overlap(box, other, margin=1) for other in placed_boxes):
                continue
            picked = (top, left)
            anchors.append(picked)
            placed_boxes.append(box)
            break
        if picked is None:
            return None

    lefts = [left for _top, left in anchors]
    if len(set(lefts)) != len(lefts):
        return None
    return anchors


def _sample_marker_columns(
    rng: np.random.Generator,
    frame_bbox: tuple[int, int, int, int],
    sprite_mask: np.ndarray,
) -> list[int] | None:
    r0, c0, r1, c1 = frame_bbox
    interior_cols = list(range(c0 + 1, c1))
    blocked_cols = [c for c in interior_cols if np.any(sprite_mask[r0 + 1 : r1, c])]
    free_cols = [c for c in interior_cols if c not in set(blocked_cols)]
    if not blocked_cols or not free_cols:
        return None

    marker_count = int(rng.integers(2, 4))
    blocked_count = min(len(blocked_cols), marker_count - 1, 1 + int(rng.integers(0, 2)))
    free_count = marker_count - blocked_count
    if free_count < 1 or len(free_cols) < free_count:
        return None

    blocked_pick = list(rng.choice(blocked_cols, size=blocked_count, replace=False))
    free_pick = list(rng.choice(free_cols, size=free_count, replace=False))
    cols = sorted(int(c) for c in blocked_pick + free_pick)
    if len(set(cols)) != marker_count:
        return None
    return cols


def _place_band_endpoints(
    rng: np.random.Generator,
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    sprite_mask: np.ndarray,
    marker_cols: list[int],
) -> bool:
    r0, _c0, r1, _c1 = frame_bbox
    marker_r = r1 - 1
    for c in marker_cols:
        blocked_rows = np.where(sprite_mask[:, c])[0]
        if blocked_rows.size:
            lo = int(blocked_rows.max()) + 1
        else:
            lo = r0 + 1
        hi = marker_r - 1
        if lo > hi:
            return False

        row_pool = list(range(lo, hi + 1))
        count = 1 if len(row_pool) == 1 else int(rng.integers(1, 3))
        count = min(count, len(row_pool))
        rows = sorted((int(r) for r in rng.choice(row_pool, size=count, replace=False)), reverse=True)
        colors = list(rng.choice(BAND_COLOR_POOL, size=count, replace=len(BAND_COLOR_POOL) < count))
        for r, color in zip(rows, colors):
            grid[r, c] = int(color)
    return True


def _build_candidate_input(rng: np.random.Generator) -> np.ndarray | None:
    sprite_count = int(rng.integers(2, 4))
    sprite_colors = list(rng.choice(SPRITE_COLOR_POOL, size=sprite_count, replace=False))
    logical_masks = [np.array(SPRITE_TEMPLATES[int(rng.integers(0, len(SPRITE_TEMPLATES)))], dtype=bool) for _ in range(sprite_count)]
    dense_masks = [_dense_shape_from_logical(mask) for mask in logical_masks]

    total_w = sum(mask.shape[1] for mask in dense_masks)
    gap_options = [
        gap
        for gap in (1, 2, 3)
        if (total_w + gap * (sprite_count + 1)) % 2 == 1
    ]
    if not gap_options:
        return None
    gap = int(rng.choice(gap_options))

    interior_w = total_w + gap * (sprite_count + 1)
    interior_h = max(mask.shape[0] for mask in dense_masks) + int(rng.choice(np.array([2, 4], dtype=int)))
    frame_h = interior_h + 2
    frame_w = interior_w + 2
    if frame_h > 21 or frame_w > 25:
        return None

    h_lo = max(16, frame_h + 5)
    h_hi = min(29, frame_h + 10)
    w_lo = max(18, frame_w + 3)
    w_hi = min(30, frame_w + 7)
    if h_lo > h_hi or w_lo > w_hi:
        return None

    grid_h = int(rng.integers(h_lo, h_hi + 1))
    grid_w = int(rng.integers(w_lo, w_hi + 1))
    if grid_h <= frame_h + 2 or grid_w <= frame_w + 2:
        return None

    top = int(rng.integers(2, grid_h - frame_h))
    left = int(rng.integers(1, grid_w - frame_w - 1))
    if top % 2 == 0:
        top += 1
    if left % 2 == 0:
        left += 1
    if top + frame_h > grid_h or left + frame_w > grid_w:
        return None
    frame_bbox = (top, left, top + frame_h - 1, left + frame_w - 1)
    if frame_w <= frame_h:
        return None

    sprite_anchors = _sample_external_boxes(rng, (grid_h, grid_w), frame_bbox, logical_masks)
    if sprite_anchors is None:
        return None

    specs = []
    for logical, color, anchor in zip(logical_masks, sprite_colors, sprite_anchors):
        dense = _dense_shape_from_logical(logical)
        top0, left0 = anchor
        specs.append(
            {
                "color": int(color),
                "bbox": (top0, left0, top0 + dense.shape[0] - 1, left0 + dense.shape[1] - 1),
                "mask": dense,
            }
        )
    placed = _place_dense_sprites(frame_bbox, specs)
    sprite_mask = _inserted_sprite_mask((grid_h, grid_w), placed)

    marker_cols = _sample_marker_columns(rng, frame_bbox, sprite_mask)
    if marker_cols is None:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    _draw_sparse_frame(grid, frame_bbox)
    for logical, color, (sr, sc) in zip(logical_masks, sprite_colors, sprite_anchors):
        _stamp_sparse_sprite(grid, logical, sr, sc, int(color))

    marker_r = frame_bbox[2] - 1
    for c in marker_cols:
        grid[marker_r, c] = MARKER_COLOR

    if not _place_band_endpoints(rng, grid, frame_bbox, sprite_mask, marker_cols):
        return None

    output = main(grid)
    if np.array_equal(grid, output):
        return None

    changed = int(np.count_nonzero(grid != output))
    if changed < 20:
        return None

    blocked_cols = [c for c in marker_cols if np.any(sprite_mask[:, c])]
    free_cols = [c for c in marker_cols if not np.any(sprite_mask[:, c])]
    if not blocked_cols or not free_cols:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = np.random.default_rng()
        candidate = _build_candidate_input(rng)
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid sparse-frame blocked-propagation input")
