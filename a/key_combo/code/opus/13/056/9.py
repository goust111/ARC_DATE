from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
SPRITE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != BLACK
    view[mask] = patch[mask]


def _component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
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
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].astype(bool)


def _mask_key(mask: np.ndarray) -> tuple[str, ...]:
    return tuple("".join("1" if cell else "0" for cell in row) for row in mask)


def _shape_signature(mask: np.ndarray) -> tuple[str, ...]:
    base = _tight_mask(mask)
    variants: list[tuple[str, ...]] = []
    for k in range(4):
        rotated = np.rot90(base, k)
        variants.append(_mask_key(_tight_mask(rotated)))
        variants.append(_mask_key(_tight_mask(np.fliplr(rotated))))
    return min(variants)


def _dihedral_equal(mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
    return _shape_signature(mask_a) == _shape_signature(mask_b)


def _find_hollow_rectangles(grid: np.ndarray) -> list[dict]:
    rectangles: list[dict] = []
    for coords in _component_coords(grid == GREY):
        bbox = _bbox_from_coords(coords)
        crop = _crop_bbox(grid, bbox)
        h, w = crop.shape
        if h < 4 or w < 4:
            continue
        if np.any(crop[0, :] != GREY) or np.any(crop[-1, :] != GREY):
            continue
        if np.any(crop[:, 0] != GREY) or np.any(crop[:, -1] != GREY):
            continue
        interior = crop[1:-1, 1:-1]
        if interior.size == 0 or np.any(interior == GREY):
            continue
        if not np.all((crop == BLACK) | (crop == GREY)):
            continue
        if sum(grid[r, c] == GREY for r, c in coords) != 2 * h + 2 * w - 4:
            continue
        hole_mask = interior == BLACK
        rectangles.append(
            {
                "bbox": bbox,
                "hole_mask": hole_mask,
                "signature": _shape_signature(hole_mask),
            }
        )
    rectangles.sort(key=lambda item: item["bbox"])
    return rectangles


def _find_sprites(grid: np.ndarray) -> list[dict]:
    sprites: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in {BLACK, GREY}):
        for coords in _component_coords(grid == color):
            bbox = _bbox_from_coords(coords)
            crop = _crop_bbox(grid, bbox)
            sprites.append(
                {
                    "color": int(color),
                    "bbox": bbox,
                    "mask": crop == int(color),
                    "bbox_area": int(crop.shape[0] * crop.shape[1]),
                    "area": int(np.count_nonzero(crop == int(color))),
                }
            )
    sprites.sort(key=lambda item: (item["bbox"], item["color"]))
    return sprites


def _best_sprite_matches(rectangles: list[dict], sprites: list[dict]) -> list[int | None]:
    matches: list[int | None] = []
    for rect in rectangles:
        candidates: list[tuple[int, int]] = []
        for idx, sprite in enumerate(sprites):
            if _dihedral_equal(rect["hole_mask"], sprite["mask"]):
                candidates.append((sprite["bbox_area"], idx))
        matches.append(min(candidates)[1] if candidates else None)
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow gray rectangle only when one outside sprite has the same hole shape up to rotation or reflection.

    Input structure:
    - A black background contains several gray one-pixel-thick hollow rectangles.
    - Each rectangle encloses a black interior hole and does not touch the grid edge.
    - Additional non-black, non-gray monochrome sprites sit outside the rectangles.

    Rule:
    - Detect every gray hollow rectangle and extract its interior black-hole mask.
    - Detect every colored sprite, crop it to its tight bounding box, and compare its mask against each hole mask up to the 8 dihedral transforms.
    - If a rectangle has at least one matching sprite, choose the matching sprite with the smallest bounding-box area and fill only that rectangle's interior black cells with the sprite's color.
    - Keep every gray outline and every original sprite pixel unchanged.

    Key insight:
    - The sprite is not moved; its normalized shape merely acts as a gate that decides whether a rectangle interior gets filled, and the fill color comes from the matching sprite itself.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    rectangles = _find_hollow_rectangles(g)
    sprites = _find_sprites(g)
    matches = _best_sprite_matches(rectangles, sprites)

    for rect, match_idx in zip(rectangles, matches):
        if match_idx is None:
            continue
        color = sprites[match_idx]["color"]
        r0, c0, r1, c1 = rect["bbox"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[interior == BLACK] = int(color)

    return out


def _rectangle_patch(hole_h: int, hole_w: int) -> np.ndarray:
    patch = np.full((hole_h + 2, hole_w + 2), GREY, dtype=int)
    patch[1:-1, 1:-1] = BLACK
    return patch


def _corner_path_mask(h: int, w: int) -> np.ndarray:
    start = random.choice([(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)])
    end = (h - 1 - start[0], w - 1 - start[1])
    moves = [(1 if end[0] > start[0] else -1, 0)] * abs(end[0] - start[0])
    moves += [(0, 1 if end[1] > start[1] else -1)] * abs(end[1] - start[1])
    random.shuffle(moves)
    mask = np.zeros((h, w), dtype=bool)
    r, c = start
    mask[r, c] = True
    for dr, dc in moves:
        r += dr
        c += dc
        mask[r, c] = True
    return mask


def _mask_to_patch(mask: np.ndarray, color: int) -> np.ndarray:
    return np.where(mask, int(color), BLACK).astype(int)


def _bboxes_too_close(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], padding: int = 1
) -> bool:
    return not (
        a[2] + padding < b[0]
        or b[2] + padding < a[0]
        or a[3] + padding < b[1]
        or b[3] + padding < a[1]
    )


def _can_place_bbox(
    bbox: tuple[int, int, int, int], existing: list[tuple[int, int, int, int]], padding: int = 1
) -> bool:
    return all(not _bboxes_too_close(bbox, other, padding) for other in existing)


def _place_rectangles(
    grid: np.ndarray, rects: list[dict], existing: list[tuple[int, int, int, int]]
) -> bool:
    h, w = grid.shape
    margin = 4
    for rect in sorted(rects, key=lambda item: item["patch"].size, reverse=True):
        patch = rect["patch"]
        ph, pw = patch.shape
        if h - ph - margin < margin or w - pw - margin < margin:
            return False
        candidates = [
            (r, c)
            for r in range(margin, h - ph - margin + 1)
            for c in range(margin, w - pw - margin + 1)
        ]
        random.shuffle(candidates)
        placed = False
        for r0, c0 in candidates:
            bbox = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            if not _can_place_bbox(bbox, existing, padding=1):
                continue
            _paste(grid, patch, r0, c0)
            existing.append(bbox)
            rect["bbox"] = bbox
            placed = True
            break
        if not placed:
            return False
    return True


def _edge_candidates(
    grid_shape: tuple[int, int], patch_shape: tuple[int, int], band: int = 4
) -> list[tuple[int, int]]:
    h, w = grid_shape
    ph, pw = patch_shape
    candidates: set[tuple[int, int]] = set()
    if ph <= band:
        for r0 in range(0, band - ph + 1):
            for c0 in range(0, w - pw + 1):
                candidates.add((r0, c0))
        for r0 in range(h - band, h - ph + 1):
            for c0 in range(0, w - pw + 1):
                candidates.add((r0, c0))
    if pw <= band:
        for c0 in range(0, band - pw + 1):
            for r0 in range(0, h - ph + 1):
                candidates.add((r0, c0))
        for c0 in range(w - band, w - pw + 1):
            for r0 in range(0, h - ph + 1):
                candidates.add((r0, c0))
    out = list(candidates)
    random.shuffle(out)
    return out


def _place_sprites(
    grid: np.ndarray, sprites: list[dict], existing: list[tuple[int, int, int, int]]
) -> bool:
    for sprite in sorted(sprites, key=lambda item: item["patch"].size, reverse=True):
        patch = sprite["patch"]
        ph, pw = patch.shape
        placed = False
        for r0, c0 in _edge_candidates(grid.shape, patch.shape, band=4):
            bbox = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            if not _can_place_bbox(bbox, existing, padding=1):
                continue
            _paste(grid, patch, r0, c0)
            existing.append(bbox)
            sprite["bbox"] = bbox
            placed = True
            break
        if not placed:
            return False
    return True


def _build_expected_output(grid: np.ndarray, rects: list[dict]) -> np.ndarray:
    out = grid.copy()
    for rect in rects:
        color = rect["fill_color"]
        if color is None:
            continue
        r0, c0, r1, c1 = rect["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = int(color)
    return out


def _validate_generated_scene(grid: np.ndarray, rects: list[dict], sprites: list[dict]) -> None:
    detected_rects = _find_hollow_rectangles(grid)
    detected_sprites = _find_sprites(grid)
    assert len(detected_rects) == len(rects)
    assert len(detected_sprites) == len(sprites)

    expected = _build_expected_output(grid, rects)
    actual = main(grid)
    assert np.array_equal(actual, expected)

    matched_rects = [rect for rect in rects if rect["fill_color"] is not None]
    assert len(matched_rects) == 2
    assert len({rect["fill_color"] for rect in matched_rects}) == 2
    assert any(rect["rotated_match"] for rect in matched_rects)
    assert any(rect["fill_color"] is None for rect in rects)

    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        if rect["fill_color"] is None:
            assert np.all(actual[r0 + 1 : r1, c0 + 1 : c1] == BLACK)
        else:
            assert np.all(actual[r0 + 1 : r1, c0 + 1 : c1] == int(rect["fill_color"]))
        assert np.all(actual[r0, c0 : c1 + 1] == GREY)
        assert np.all(actual[r1, c0 : c1 + 1] == GREY)
        assert np.all(actual[r0 : r1 + 1, c0] == GREY)
        assert np.all(actual[r0 : r1 + 1, c1] == GREY)

    used_match_indices = {idx for idx in _best_sprite_matches(detected_rects, detected_sprites) if idx is not None}
    assert len(used_match_indices) == 2
    assert len(detected_rects) - len(used_match_indices) >= 1
    assert len(detected_sprites) - len(used_match_indices) >= 1

    hole_shapes = [tuple(rect["hole_shape"]) for rect in rects]
    unmatched_bbox_decoy = False
    for idx, sprite in enumerate(detected_sprites):
        if idx in used_match_indices:
            continue
        shp = tuple(int(v) for v in sprite["mask"].shape)
        if any(shp == hole or shp == hole[::-1] for hole in hole_shapes):
            unmatched_bbox_decoy = True
            break
    assert unmatched_bbox_decoy

    diff = actual != grid
    expected_diff = expected != grid
    assert np.array_equal(diff, expected_diff)


def _attempt_generate() -> np.ndarray | None:
    n_rectangles = random.choice([3, 4])
    min_side = 18 if n_rectangles == 3 else 20
    h = random.randint(min_side, 24)
    w = random.randint(min_side, 24)
    grid = np.zeros((h, w), dtype=int)

    colors = random.sample(SPRITE_COLORS, 5 if random.random() < 0.5 else 4)
    square_color = colors[0]
    nonsquare_color = colors[1]

    nonsquare_hole = random.choice([(2, 3), (3, 2)])
    unmatched_holes = [random.choice([(3, 3), (3, 4), (4, 3)])]
    if n_rectangles == 4:
        extra_choices = [(3, 5), (5, 3), (4, 4), (4, 5), (5, 4), (5, 5)]
        unmatched_holes.append(random.choice(extra_choices))

    rects = [
        {
            "hole_shape": (2, 2),
            "patch": _rectangle_patch(2, 2),
            "fill_color": square_color,
            "rotated_match": False,
        },
        {
            "hole_shape": nonsquare_hole,
            "patch": _rectangle_patch(*nonsquare_hole),
            "fill_color": nonsquare_color,
            "rotated_match": True,
        },
    ]
    for hole_shape in unmatched_holes:
        rects.append(
            {
                "hole_shape": hole_shape,
                "patch": _rectangle_patch(*hole_shape),
                "fill_color": None,
                "rotated_match": False,
            }
        )
    random.shuffle(rects)

    sprites = [
        {
            "patch": _mask_to_patch(np.ones((2, 2), dtype=bool), square_color),
            "matched": True,
        },
        {
            "patch": _mask_to_patch(np.ones(nonsquare_hole[::-1], dtype=bool), nonsquare_color),
            "matched": True,
        },
        {
            "patch": _mask_to_patch(_corner_path_mask(*nonsquare_hole[::-1]), colors[2]),
            "matched": False,
        },
        {
            "patch": _mask_to_patch(_corner_path_mask(*unmatched_holes[0]), colors[3]),
            "matched": False,
        },
    ]
    if len(colors) == 5:
        extra_shape = random.choice([(2, 2), (2, 3), (3, 2), (3, 3)])
        sprites.append(
            {
                "patch": _mask_to_patch(_corner_path_mask(*extra_shape), colors[4]),
                "matched": False,
            }
        )
    random.shuffle(sprites)

    placed_bboxes: list[tuple[int, int, int, int]] = []
    if not _place_rectangles(grid, rects, placed_bboxes):
        return None
    if not _place_sprites(grid, sprites, placed_bboxes):
        return None

    _validate_generated_scene(grid, rects, sprites)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
