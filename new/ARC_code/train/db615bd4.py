from __future__ import annotations

from collections import Counter

import numpy as np

from arc_utils import bbox_coords, colors_present, get_bg, paste
import random


def _bbox_area(b: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = b
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _bboxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar0 <= br1 and br0 <= ar1 and ac0 <= bc1 and bc0 <= ac1


def _is_sparse_rect_perimeter(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = bbox_coords(coords)
    rel = coords - np.array([r0, c0])
    if np.any(rel % 2 != 0):
        return False

    h = (r1 - r0) // 2 + 1
    w = (c1 - c0) // 2 + 1
    expected = h * w if h == 1 or w == 1 else 2 * h + 2 * w - 4
    if coords.shape[0] != expected:
        return False

    points = {tuple(map(int, p)) for p in coords}
    for r in range(r0, r1 + 1, 2):
        for c in range(c0, c1 + 1, 2):
            on_border = h == 1 or w == 1 or r in (r0, r1) or c in (c0, c1)
            if ((r, c) in points) != on_border:
                return False
    return True


def _solidified_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(coords)
    rel = coords - np.array([r0, c0])
    assert np.all(rel % 2 == 0), "sprite pixels must sit on a two-step lattice"

    h = (r1 - r0) // 2 + 1
    w = (c1 - c0) // 2 + 1
    logical = np.zeros((h, w), dtype=bool)
    for dr, dc in rel:
        logical[int(dr) // 2, int(dc) // 2] = True

    return np.kron(logical, np.ones((2, 2), dtype=bool))[: 2 * h - 1, : 2 * w - 1]


def _place_sprites_in_order(
    out: np.ndarray,
    sprites: list[dict],
    frame_bbox: tuple[int, int, int, int],
    horizontal: bool,
) -> None:
    r0, c0, r1, c1 = frame_bbox
    interior_h = r1 - r0 - 1
    interior_w = c1 - c0 - 1
    assert interior_h > 0 and interior_w > 0, "frame needs an interior"

    if horizontal:
        order_keys = [(sp["bbox"][1], sp["bbox"][0]) for sp in sprites]
        assert len(set(order_keys)) == len(order_keys), "source x-order must be unique"
        sprites.sort(key=lambda sp: (sp["bbox"][1], sp["bbox"][0]))

        total_w = sum(sp["mask"].shape[1] for sp in sprites)
        slack = interior_w - total_w
        assert slack > 0 and slack % (len(sprites) + 1) == 0, "sprites must leave equal horizontal gaps"
        gap = slack // (len(sprites) + 1)
        cursor = c0 + 1 + gap

        for sp in sprites:
            mask = sp["mask"]
            ph, pw = mask.shape
            cross_slack = interior_h - ph
            assert cross_slack >= 0 and cross_slack % 2 == 0, "sprite must center vertically"
            rr = r0 + 1 + cross_slack // 2
            patch = np.full(mask.shape, int(sp["color"]), dtype=int)
            paste(out, patch, rr, cursor, mask=mask)
            cursor += pw + gap
    else:
        order_keys = [(sp["bbox"][0], sp["bbox"][1]) for sp in sprites]
        assert len(set(order_keys)) == len(order_keys), "source y-order must be unique"
        sprites.sort(key=lambda sp: (sp["bbox"][0], sp["bbox"][1]))

        total_h = sum(sp["mask"].shape[0] for sp in sprites)
        slack = interior_h - total_h
        assert slack > 0 and slack % (len(sprites) + 1) == 0, "sprites must leave equal vertical gaps"
        gap = slack // (len(sprites) + 1)
        cursor = r0 + 1 + gap

        for sp in sprites:
            mask = sp["mask"]
            ph, pw = mask.shape
            cross_slack = interior_w - pw
            assert cross_slack >= 0 and cross_slack % 2 == 0, "sprite must center horizontally"
            cc = c0 + 1 + cross_slack // 2
            patch = np.full(mask.shape, int(sp["color"]), dtype=int)
            paste(out, patch, cursor, cc, mask=mask)
            cursor += ph + gap


def main(grid: np.ndarray) -> np.ndarray:
    """
    Thicken the sparse frame and stamp the external sparse sprites into it in positional order.

    Input structure
    The grid has a dominant background color, a regular lattice color, one large
    sparse rectangular frame, and one or more external monochrome sprites whose
    pixels also sit on a two-step lattice.

    Rule
    Use the largest sparse rectangular perimeter as the frame. Clear that frame
    region and the external sprite pixels to background, draw a solid one-pixel
    border in the frame color, solidify each sparse sprite into its contiguous
    two-step mask, and place the sprites inside the frame with equal gaps.
    Wide frames receive sprites left-to-right by source position; tall frames
    receive them top-to-bottom by source position.

    Key insight
    The frame supplies ordered slots: the sprite-to-slot assignment is by visible
    source order, not by numeric color or by matching individual cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    counts = Counter(map(int, g.ravel()))
    non_bg = colors_present(g, exclude={bg})
    assert non_bg, "expected foreground colors"

    max_non_bg_count = max(counts[c] for c in non_bg)
    lattice_candidates = [c for c in non_bg if counts[c] == max_non_bg_count]
    assert len(lattice_candidates) == 1, "lattice color must be uniquely most frequent"
    lattice_color = lattice_candidates[0]

    special_colors = [c for c in non_bg if c != lattice_color]
    assert len(special_colors) >= 2, "expected one frame and at least one sprite"

    coords_by_color = {c: np.argwhere(g == c) for c in special_colors}
    frame_candidates = []
    for color, coords in coords_by_color.items():
        if _is_sparse_rect_perimeter(coords):
            frame_candidates.append((color, bbox_coords(coords)))
    assert frame_candidates, "expected a sparse rectangular frame"

    max_area = max(_bbox_area(b) for _color, b in frame_candidates)
    largest_frames = [(color, b) for color, b in frame_candidates if _bbox_area(b) == max_area]
    assert len(largest_frames) == 1, "frame must be the unique largest sparse perimeter"
    frame_color, frame_bbox = largest_frames[0]

    source_colors = [c for c in special_colors if c != frame_color]
    r0, c0, r1, c1 = frame_bbox
    interior_h = r1 - r0 - 1
    interior_w = c1 - c0 - 1
    assert interior_h != interior_w, "frame must have a unique long axis"

    sprites = []
    for color in source_colors:
        coords = coords_by_color[color]
        b = bbox_coords(coords)
        assert not _bboxes_overlap(b, frame_bbox), "sprites must be outside the frame"
        mask = _solidified_mask(coords)
        assert mask.any(), "sprite mask must be nonempty"
        sprites.append({"color": color, "bbox": b, "mask": mask})

    out = g.copy()
    for color in source_colors:
        coords = coords_by_color[color]
        out[coords[:, 0], coords[:, 1]] = bg

    out[r0 : r1 + 1, c0 : c1 + 1] = bg
    out[r0, c0 : c1 + 1] = frame_color
    out[r1, c0 : c1 + 1] = frame_color
    out[r0 : r1 + 1, c0] = frame_color
    out[r0 : r1 + 1, c1] = frame_color

    _place_sprites_in_order(out, sprites, frame_bbox, horizontal=interior_w > interior_h)
    return out

# --------------- generator (att1) ---------------

# Canvas and lattice geometry.
GRID_SIZE, LATTICE_STEP = 25, 2
ODD_ANCHOR_START, EVEN_ANCHOR_START = 1, 0

# Palette roles.
COLOR_RANGE, BASE_COLOR_COUNT = range(10), 3

# Sprite and frame logical dimensions.
SPRITE_COUNTS = [2, 3]
SPRITE_DIM_MIN, SPRITE_DIM_MAX = 1, 3
ALONG_SUM_MIN, ALONG_SUM_MAX = 3, 7
FRAME_DIM_MIN, FRAME_DIM_MAX = 5, 9
FRAME_MARGIN_DIMS = 2
MAX_GENERATION_ATTEMPTS = 300


def _choose_along_dims(n_sprites: int) -> list[int] | None:
    along_dims = []
    for i in range(n_sprites):
        remaining = n_sprites - i - 1
        current_sum = sum(along_dims)
        lo = max(SPRITE_DIM_MIN, ALONG_SUM_MIN - current_sum - SPRITE_DIM_MAX * remaining)
        hi = min(SPRITE_DIM_MAX, ALONG_SUM_MAX - current_sum - SPRITE_DIM_MIN * remaining)
        if lo > hi:
            return None
        along_dims.append(random.randint(lo, hi))
    return along_dims


def _choose_frame_shape(horizontal: bool, n_sprites: int) -> tuple[int, int, list[tuple[int, int]]] | None:
    cross_dim = random.randint(SPRITE_DIM_MIN, SPRITE_DIM_MAX)
    along_dims = _choose_along_dims(n_sprites)
    if along_dims is None:
        return None

    frame_along = sum(along_dims) + FRAME_MARGIN_DIMS
    min_cross = max(FRAME_DIM_MIN, cross_dim + 1)
    possible_cross = [v for v in range(min_cross, FRAME_DIM_MAX + 1) if v != frame_along]
    if not possible_cross:
        return None

    frame_cross = random.choice(possible_cross)
    if horizontal:
        return frame_cross, frame_along, [(cross_dim, w_l) for w_l in along_dims]
    return frame_along, frame_cross, [(h_l, cross_dim) for h_l in along_dims]


def _pixel_bbox(sr: int, sc: int, h_l: int, w_l: int) -> tuple[int, int, int, int]:
    return (sr, sc, sr + LATTICE_STEP * (h_l - 1), sc + LATTICE_STEP * (w_l - 1))


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _choose_odd_bbox(h_l: int, w_l: int) -> tuple[int, int, int, int] | None:
    max_r = GRID_SIZE - 1 - LATTICE_STEP * (h_l - 1)
    max_c = GRID_SIZE - 1 - LATTICE_STEP * (w_l - 1)
    valid_r = list(range(ODD_ANCHOR_START, max_r + 1, LATTICE_STEP))
    valid_c = list(range(ODD_ANCHOR_START, max_c + 1, LATTICE_STEP))
    if not valid_r or not valid_c:
        return None

    r0 = random.choice(valid_r)
    c0 = random.choice(valid_c)
    return _pixel_bbox(r0, c0, h_l, w_l)


def _stamp_sparse_frame(grid: np.ndarray, frame_bbox: tuple[int, int, int, int], frame_color: int) -> None:
    r0, c0, r1, c1 = frame_bbox
    grid[r0 : r1 + 1 : LATTICE_STEP, c0] = frame_color
    grid[r0 : r1 + 1 : LATTICE_STEP, c1] = frame_color
    grid[r0, c0 : c1 + 1 : LATTICE_STEP] = frame_color
    grid[r1, c0 : c1 + 1 : LATTICE_STEP] = frame_color


def _choose_sprite_anchors(sprite_dims: list[tuple[int, int]], frame_bbox: tuple[int, int, int, int]) -> list[tuple[int, int]] | None:
    placed_bboxes = []
    sprite_anchors = []

    for h_l, w_l in sprite_dims:
        max_sr = GRID_SIZE - 1 - LATTICE_STEP * (h_l - 1)
        max_sc = GRID_SIZE - 1 - LATTICE_STEP * (w_l - 1)
        candidates = []
        for sr in range(EVEN_ANCHOR_START, max_sr + 1, LATTICE_STEP):
            for sc in range(EVEN_ANCHOR_START, max_sc + 1, LATTICE_STEP):
                sb = _pixel_bbox(sr, sc, h_l, w_l)
                if _overlap(sb, frame_bbox):
                    continue
                if any(_overlap(sb, pb) for pb in placed_bboxes):
                    continue
                candidates.append((sr, sc))
        if not candidates:
            return None

        sr, sc = random.choice(candidates)
        placed_bboxes.append(_pixel_bbox(sr, sc, h_l, w_l))
        sprite_anchors.append((sr, sc))

    return sprite_anchors


def _has_unique_ordering(sprite_anchors: list[tuple[int, int]], horizontal: bool) -> bool:
    keys = [(sc, sr) for sr, sc in sprite_anchors] if horizontal else [(sr, sc) for sr, sc in sprite_anchors]
    return len(set(keys)) == len(keys)


def _stamp_sparse_sprites(grid: np.ndarray, sprite_dims: list[tuple[int, int]], sprite_anchors: list[tuple[int, int]], sprite_colors: list[int]) -> None:
    for i, (h_l, w_l) in enumerate(sprite_dims):
        sr, sc = sprite_anchors[i]
        grid[sr : sr + LATTICE_STEP * h_l : LATTICE_STEP, sc : sc + LATTICE_STEP * w_l : LATTICE_STEP] = sprite_colors[i]


def _lattice_is_unique_non_bg_max(grid: np.ndarray, bg: int, lattice_color: int) -> bool:
    counts = Counter(grid.ravel().tolist())
    non_bg = [c for c in counts if c != bg]
    if not non_bg:
        return False

    max_count = max(counts[c] for c in non_bg)
    lattice_cands = [c for c in non_bg if counts[c] == max_count]
    return len(lattice_cands) == 1 and lattice_cands[0] == lattice_color


def _gen_attempt() -> np.ndarray | None:
    n_sprites = random.choice(SPRITE_COUNTS)
    palette = random.sample(COLOR_RANGE, BASE_COLOR_COUNT + n_sprites)
    bg, lattice_color, frame_color = palette[0], palette[1], palette[2]
    sprite_colors = palette[3:]
    horizontal = random.choice([True, False])

    frame_and_sprites = _choose_frame_shape(horizontal, n_sprites)
    if frame_and_sprites is None:
        return None
    frame_H, frame_W, sprite_dims = frame_and_sprites

    frame_bbox = _choose_odd_bbox(frame_H, frame_W)
    if frame_bbox is None:
        return None

    grid = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    grid[ODD_ANCHOR_START::LATTICE_STEP, ODD_ANCHOR_START::LATTICE_STEP] = lattice_color
    _stamp_sparse_frame(grid, frame_bbox, frame_color)

    sprite_anchors = _choose_sprite_anchors(sprite_dims, frame_bbox)
    if sprite_anchors is None or not _has_unique_ordering(sprite_anchors, horizontal):
        return None

    _stamp_sparse_sprites(grid, sprite_dims, sprite_anchors, sprite_colors)

    if not _lattice_is_unique_non_bg_max(grid, bg, lattice_color):
        return None

    try:
        main(grid)
    except Exception:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
