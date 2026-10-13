from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
GEN_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, grid.ravel()))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_h(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[2] - bbox[0] + 1)


def _bbox_w(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[3] - bbox[1] + 1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _on_bbox_perimeter(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    on_edge = (
        (coords[:, 0] == r0)
        | (coords[:, 0] == r1)
        | (coords[:, 1] == c0)
        | (coords[:, 1] == c1)
    )
    return bool(np.all(on_edge))


def _frame_candidates(grid: np.ndarray, bg: int) -> list[tuple[int, tuple[int, int, int, int]]]:
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in _colors_present(grid, exclude={bg}):
        coords = np.argwhere(grid == color)
        if coords.shape[0] < 4:
            continue
        bbox = _bbox_coords(coords)
        if _bbox_h(bbox) < 3 or _bbox_w(bbox) < 3:
            continue
        if _on_bbox_perimeter(coords, bbox):
            candidates.append((int(color), bbox))
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the latent sparse frame, complete its border, and crop the scene to that viewfinder.

    Input structure:
    - The grid has a black background, one sparse latent frame color, several other scattered single-pixel colors, and one solid monochrome rectangle used as a decoy.
    - For each non-background color, its visible pixels define a color-specific bounding box.
    - Exactly one color has a non-degenerate bounding box whose every visible pixel lies on that box perimeter.

    Rule:
    - Test each color independently for the "all pixels lie on their own bounding-box perimeter" property, keeping only colors whose bbox height and width are both at least three.
    - The unique passing color is the hidden frame color; crop the input to that color's bounding box.
    - Repaint the entire outer ring of the crop with the frame color and leave the cropped interior unchanged.

    Key insight:
    - The obvious solid rectangle is irrelevant: the real locator is the only sparse color whose pixels all share one latent rectangular perimeter, even though most of that frame is missing in the input.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    candidates = _frame_candidates(g, bg)
    assert len(candidates) == 1, "expected exactly one latent-frame color"
    frame_color, frame_bbox = candidates[0]

    out = _crop_bbox(g, frame_bbox)
    out[0, :] = frame_color
    out[-1, :] = frame_color
    out[:, 0] = frame_color
    out[:, -1] = frame_color
    return out


def _bbox_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int] | None) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if bbox is None:
        return mask
    r0, c0, r1, c1 = bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _expand_bbox(
    bbox: tuple[int, int, int, int], pad: int, shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    return (
        max(0, r0 - pad),
        max(0, c0 - pad),
        min(h - 1, r1 + pad),
        min(w - 1, c1 + pad),
    )


def _shrink_bbox(
    bbox: tuple[int, int, int, int], pad: int
) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = bbox
    shrunk = (r0 + pad, c0 + pad, r1 - pad, c1 - pad)
    if shrunk[0] > shrunk[2] or shrunk[1] > shrunk[3]:
        return None
    return shrunk


def _zone_inside(shape: tuple[int, int], frame_bbox: tuple[int, int, int, int], pad: int) -> np.ndarray:
    return _bbox_mask(shape, _shrink_bbox(frame_bbox, pad))


def _zone_outside(shape: tuple[int, int], frame_bbox: tuple[int, int, int, int], pad: int) -> np.ndarray:
    mask = np.ones(shape, dtype=bool)
    er = _expand_bbox(frame_bbox, pad, shape)
    mask[er[0] : er[2] + 1, er[1] : er[3] + 1] = False
    return mask


def _reserve(reserved: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> None:
    rr = _expand_bbox(bbox, pad, reserved.shape)
    reserved[rr[0] : rr[2] + 1, rr[1] : rr[3] + 1] = True


def _paste_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> tuple[int, int, int, int]:
    h, w = patch.shape
    view = patch != BG
    grid[top : top + h, left : left + w][view] = patch[view]
    return int(top), int(left), int(top + h - 1), int(left + w - 1)


def _candidate_positions(
    grid: np.ndarray,
    reserved: np.ndarray,
    zone_mask: np.ndarray,
    patch_shape: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = grid.shape
    ph, pw = patch_shape
    positions: list[tuple[int, int]] = []
    for top in range(0, h - ph + 1):
        for left in range(0, w - pw + 1):
            slr = slice(top, top + ph)
            slc = slice(left, left + pw)
            if not bool(zone_mask[slr, slc].all()):
                continue
            if bool(reserved[slr, slc].any()):
                continue
            if bool((grid[slr, slc] != BG).any()):
                continue
            positions.append((int(top), int(left)))
    return positions


def _place_patch(
    grid: np.ndarray,
    reserved: np.ndarray,
    zone_mask: np.ndarray,
    patch: np.ndarray,
    rng: random.Random,
    *,
    pad: int = 1,
) -> tuple[int, int, int, int] | None:
    positions = _candidate_positions(grid, reserved, zone_mask, patch.shape)
    if not positions:
        return None
    top, left = rng.choice(positions)
    bbox = _paste_patch(grid, patch, top, left)
    _reserve(reserved, bbox, pad=pad)
    return bbox


def _make_line_patch(color: int, rng: random.Random) -> np.ndarray:
    point_count = rng.choice([4, 5, 6])
    length = 2 * point_count - 1
    patch = np.zeros((1, length), dtype=int)
    patch[0, ::2] = color
    if rng.random() < 0.5:
        patch = patch.T
    return patch


def _make_interior_patch(color: int, rng: random.Random) -> np.ndarray:
    h = rng.choice([5, 6, 7])
    w = rng.choice([5, 6, 7])
    patch = np.zeros((h, w), dtype=int)
    patch[0, 0] = color
    patch[0, w - 1] = color
    patch[h - 1, 0] = color
    patch[h - 1, w - 1] = color
    center_r = rng.randint(2, h - 3)
    center_c = rng.randint(2, w - 3)
    patch[center_r, center_c] = color
    return patch


def _make_decoy_patch(color: int, rng: random.Random) -> np.ndarray:
    h = rng.randint(3, 5)
    w = rng.randint(3, 6)
    return np.full((h, w), color, dtype=int)


def _sample_frame_points(
    bbox: tuple[int, int, int, int], color: int, rng: random.Random
) -> list[tuple[int, int, int]]:
    r0, c0, r1, c1 = bbox
    top = [(r0, c) for c in range(c0, c1 + 1)]
    bottom = [(r1, c) for c in range(c0, c1 + 1)]
    left = [(r, c0) for r in range(r0 + 1, r1)]
    right = [(r, c1) for r in range(r0 + 1, r1)]

    chosen = {
        rng.choice(top),
        rng.choice(bottom),
        rng.choice(left),
        rng.choice(right),
    }

    border = top + bottom + left + right
    extras = [cell for cell in border if cell not in chosen]
    rng.shuffle(extras)
    perimeter = 2 * (_bbox_h(bbox) + _bbox_w(bbox)) - 4
    target = rng.randint(max(6, perimeter // 3), max(7, (2 * perimeter) // 3))
    target = min(target, perimeter - 2)
    for cell in extras[: max(0, target - len(chosen))]:
        chosen.add(cell)
    return [(r, c, color) for r, c in sorted(chosen)]


def _paint_frame_points(grid: np.ndarray, points: list[tuple[int, int, int]]) -> None:
    for r, c, color in points:
        grid[r, c] = color


def _build_expected_output(
    grid: np.ndarray, frame_color: int, frame_bbox: tuple[int, int, int, int]
) -> np.ndarray:
    out = _crop_bbox(grid, frame_bbox)
    out[0, :] = frame_color
    out[-1, :] = frame_color
    out[:, 0] = frame_color
    out[:, -1] = frame_color
    return out


def _validate_scene(
    grid: np.ndarray,
    frame_color: int,
    frame_bbox: tuple[int, int, int, int],
    decoy_bbox: tuple[int, int, int, int],
    mode: str,
) -> bool:
    qualifiers = _frame_candidates(grid, BG)
    if qualifiers != [(frame_color, frame_bbox)]:
        return False

    frame_coords = np.argwhere(grid == frame_color)
    perimeter = 2 * (_bbox_h(frame_bbox) + _bbox_w(frame_bbox)) - 4
    if int(frame_coords.shape[0]) >= perimeter:
        return False

    r0, c0, r1, c1 = decoy_bbox
    decoy_color = int(grid[r0, c0])
    if decoy_color == BG:
        return False
    if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == decoy_color):
        return False
    if _bbox_h(decoy_bbox) < 3 or _bbox_w(decoy_bbox) < 3:
        return False

    crop = _crop_bbox(grid, frame_bbox)
    inner = crop[1:-1, 1:-1]
    if inner.size == 0 or not bool(np.any(inner != BG)):
        return False

    inside_decoy = (
        frame_bbox[0] < decoy_bbox[0]
        and frame_bbox[1] < decoy_bbox[1]
        and decoy_bbox[2] < frame_bbox[2]
        and decoy_bbox[3] < frame_bbox[3]
    )
    if mode == "decoy_outside" and inside_decoy:
        return False
    if mode != "decoy_outside" and not inside_decoy:
        return False

    out = _build_expected_output(grid, frame_color, frame_bbox)
    try:
        solved = main(grid)
    except AssertionError:
        return False
    return np.array_equal(solved, out)


def _attempt_generate() -> np.ndarray | None:
    rng = random.Random()
    height = rng.randint(22, 30)
    width = rng.randint(22, 30)
    grid = np.zeros((height, width), dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    frame_h = rng.randint(11, min(17, height - 4))
    frame_w = rng.randint(11, min(17, width - 4))
    frame_top = rng.randint(1, height - frame_h - 1)
    frame_left = rng.randint(1, width - frame_w - 1)
    frame_bbox = (
        int(frame_top),
        int(frame_left),
        int(frame_top + frame_h - 1),
        int(frame_left + frame_w - 1),
    )

    colors = rng.sample(PALETTE, rng.randint(4, 6))
    frame_color = colors[0]
    decoy_color = colors[1]
    line_colors = [colors[2]]
    interior_colors = [colors[3]]
    extra_pool = colors[4:]

    mode = rng.choice(["both_inside", "decoy_inside_only", "decoy_outside"])
    inside_kind = rng.choice(["line", "interior"])

    inner_zone = _zone_inside(grid.shape, frame_bbox, pad=2)
    outside_zone = _zone_outside(grid.shape, frame_bbox, pad=1)
    if not bool(inner_zone.any()) or not bool(outside_zone.any()):
        return None

    decoy_patch = _make_decoy_patch(decoy_color, rng)
    decoy_zone = inner_zone if mode != "decoy_outside" else outside_zone
    decoy_bbox = _place_patch(grid, reserved, decoy_zone, decoy_patch, rng, pad=1)
    if decoy_bbox is None:
        return None

    line_inside = mode in {"both_inside", "decoy_outside"} and inside_kind == "line"
    interior_inside = mode in {"both_inside", "decoy_outside"} and inside_kind == "interior"

    line_patch = _make_line_patch(line_colors[0], rng)
    line_bbox = _place_patch(
        grid,
        reserved,
        inner_zone if line_inside else outside_zone,
        line_patch,
        rng,
        pad=1,
    )
    if line_bbox is None:
        return None

    interior_patch = _make_interior_patch(interior_colors[0], rng)
    interior_bbox = _place_patch(
        grid,
        reserved,
        inner_zone if interior_inside else outside_zone,
        interior_patch,
        rng,
        pad=1,
    )
    if interior_bbox is None:
        return None

    for extra_color in extra_pool:
        if rng.random() < 0.45:
            continue
        patch = _make_line_patch(extra_color, rng) if rng.random() < 0.5 else _make_interior_patch(extra_color, rng)
        bbox = _place_patch(grid, reserved, outside_zone, patch, rng, pad=1)
        if bbox is None:
            return None

    frame_points = _sample_frame_points(frame_bbox, frame_color, rng)
    for r, c, _color in frame_points:
        if grid[r, c] != BG:
            return None
    _paint_frame_points(grid, frame_points)

    if not _validate_scene(grid, frame_color, frame_bbox, decoy_bbox, mode):
        return None

    line_coords = np.argwhere(grid == line_colors[0])
    interior_coords = np.argwhere(grid == interior_colors[0])
    if _bbox_h(_bbox_coords(line_coords)) != 1 and _bbox_w(_bbox_coords(line_coords)) != 1:
        return None
    if _on_bbox_perimeter(interior_coords, _bbox_coords(interior_coords)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid latent-frame input")
