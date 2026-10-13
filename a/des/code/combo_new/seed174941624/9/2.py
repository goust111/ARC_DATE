from __future__ import annotations

from collections import deque
import random

import numpy as np


_BG = 0
_COLORS = tuple(range(1, 10))
_MIN_GRID = 18
_MAX_GRID = 30
_MIN_INTERIOR = 3
_MAX_INTERIOR = 7
_MIN_PIECE_AREA = 3
_PLACEMENT_GAP = 1
_GEN_ATTEMPTS = 400
_PARTITION_ATTEMPTS = 120


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(coords)

    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _component_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        out[r - r0, c - c0] = True
    return out


def _is_hollow_rect_frame(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if len(coords) != 2 * (h + w) - 4:
        return False

    coord_set = set(coords)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if ((r, c) in coord_set) != on_border:
                return False
    return True


def _iter_translations(mask: np.ndarray, target_shape: tuple[int, int]):
    target_h, target_w = target_shape
    h, w = mask.shape
    for r0 in range(target_h - h + 1):
        for c0 in range(target_w - w + 1):
            canvas = np.zeros((target_h, target_w), dtype=bool)
            canvas[r0 : r0 + h, c0 : c0 + w] |= mask
            yield canvas


def _can_reconstruct(mask_a: np.ndarray, mask_b: np.ndarray, target_shape: tuple[int, int]) -> bool:
    target_h, target_w = target_shape
    if int(mask_a.sum() + mask_b.sum()) != target_h * target_w:
        return False

    for keep, rotate_src in ((mask_a, mask_b), (mask_b, mask_a)):
        rotated = np.rot90(rotate_src, 2)
        for placed_keep in _iter_translations(keep, target_shape):
            for placed_rot in _iter_translations(rotated, target_shape):
                if np.any(placed_keep & placed_rot):
                    continue
                if np.all(placed_keep | placed_rot):
                    return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each color's two fragments and use their 180-degree complementarity to fill the matching hollow frame.

    Input structure:
    - The background is black.
    - Several colors each form one one-pixel-thick hollow rectangular frame.
    - The same color also appears in exactly two separate 4-connected fragment objects elsewhere.
    - For each color, the two fragments can tile that frame's interior when one is rotated 180 degrees.

    Rule:
    - Group connected components by color and identify the hollow rectangular frame for each color.
    - Crop the two non-frame fragments to their bounding boxes.
    - Check that one crop and a 180-degree rotation of the other can be translated inside the frame interior so their union covers the whole interior with no overlap.
    - Clear the original fragment pixels and fill the frame interior with the frame color, leaving the border unchanged.

    Key insight:
    - Color links each frame to exactly two fragments, but the crucial geometric relation is only revealed after rotating one fragment by 180 degrees and testing how the pair complements the rectangular interior.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    out = g.copy()

    for color in sorted(int(c) for c in np.unique(g) if int(c) != bg):
        components = _find_components(g == color)
        assert len(components) == 3, "each color must form one frame and two fragments"

        frame_components = [comp for comp in components if _is_hollow_rect_frame(comp)]
        assert len(frame_components) == 1, "expected a unique hollow frame per color"
        frame = frame_components[0]
        pieces = [comp for comp in components if comp is not frame]
        assert len(pieces) == 2

        fr0, fc0, fr1, fc1 = _bbox(frame)
        interior_shape = (fr1 - fr0 - 1, fc1 - fc0 - 1)
        piece_masks = [_component_mask(comp) for comp in pieces]
        assert _can_reconstruct(piece_masks[0], piece_masks[1], interior_shape)

        for comp in pieces:
            for r, c in comp:
                out[r, c] = bg
        out[fr0 + 1 : fr1, fc0 + 1 : fc1] = color

    return out


def _connected(mask: np.ndarray) -> bool:
    return len(_find_components(mask)) == 1


def _random_partition(h: int, w: int) -> tuple[np.ndarray, np.ndarray] | None:
    axis = random.randrange(2)
    if axis == 0:
        cut = [random.randint(1, h - 1)]
        for _ in range(1, w):
            prev = cut[-1]
            cut.append(random.randint(max(1, prev - 2), min(h - 1, prev + 2)))
        if len(set(cut)) < 2:
            return None
        top = np.zeros((h, w), dtype=bool)
        for c, k in enumerate(cut):
            top[:k, c] = True
        mask_a = top
    else:
        cut = [random.randint(1, w - 1)]
        for _ in range(1, h):
            prev = cut[-1]
            cut.append(random.randint(max(1, prev - 2), min(w - 1, prev + 2)))
        if len(set(cut)) < 2:
            return None
        left = np.zeros((h, w), dtype=bool)
        for r, k in enumerate(cut):
            left[r, :k] = True
        mask_a = left

    mask_b = ~mask_a
    if random.random() < 0.5:
        mask_a = np.flipud(mask_a)
        mask_b = np.flipud(mask_b)
    if random.random() < 0.5:
        mask_a = np.fliplr(mask_a)
        mask_b = np.fliplr(mask_b)

    if not _connected(mask_a) or not _connected(mask_b):
        return None
    if min(int(mask_a.sum()), int(mask_b.sum())) < _MIN_PIECE_AREA:
        return None

    piece_a = _crop_mask(mask_a)
    piece_b = _crop_mask(mask_b)
    if random.random() < 0.5:
        return piece_a, np.rot90(piece_b, 2)
    return np.rot90(piece_a, 2), piece_b


def _frame_patch(interior_h: int, interior_w: int, color: int) -> np.ndarray:
    patch = np.full((interior_h + 2, interior_w + 2), _BG, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    return patch


def _piece_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, _BG, dtype=int)
    patch[mask] = color
    return patch


def _can_place_bbox(
    placed_boxes: list[tuple[int, int, int, int]],
    r0: int,
    c0: int,
    h: int,
    w: int,
    gap: int,
) -> bool:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    for pr0, pc0, pr1, pc1 in placed_boxes:
        if not (r1 + gap < pr0 or pr1 + gap < r0 or c1 + gap < pc0 or pc1 + gap < c0):
            return False
    return True


def _paint_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    h, w = patch.shape
    target = grid[r0 : r0 + h, c0 : c0 + w]
    mask = patch != _BG
    if np.any(target[mask] != _BG):
        return False
    target[mask] = patch[mask]
    return True


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    grid_h = random.randint(_MIN_GRID, _MAX_GRID)
    grid_w = random.randint(_MIN_GRID, _MAX_GRID)
    n_colors = random.randint(1, 3)
    colors = random.sample(_COLORS, n_colors)

    color_specs: list[tuple[int, int, int, np.ndarray, np.ndarray]] = []
    for color in colors:
        interior_h = random.randint(_MIN_INTERIOR, _MAX_INTERIOR)
        interior_w = random.randint(_MIN_INTERIOR, _MAX_INTERIOR)

        pieces = None
        for _ in range(_PARTITION_ATTEMPTS):
            candidate = _random_partition(interior_h, interior_w)
            if candidate is None:
                continue
            piece_a, piece_b = candidate
            if _can_reconstruct(piece_a, piece_b, (interior_h, interior_w)):
                pieces = (piece_a, piece_b)
                break
        if pieces is None:
            return None

        color_specs.append((color, interior_h, interior_w, pieces[0], pieces[1]))

    objects: list[tuple[str, int, np.ndarray]] = []
    for color, interior_h, interior_w, piece_a, piece_b in color_specs:
        objects.append(("frame", color, _frame_patch(interior_h, interior_w, color)))
        objects.append(("piece", color, _piece_patch(piece_a, color)))
        objects.append(("piece", color, _piece_patch(piece_b, color)))

    objects.sort(key=lambda item: (item[2].shape[0] * item[2].shape[1], max(item[2].shape)), reverse=True)

    inp = np.full((grid_h, grid_w), _BG, dtype=int)
    out = np.full((grid_h, grid_w), _BG, dtype=int)
    placed_boxes: list[tuple[int, int, int, int]] = []
    placements: list[tuple[str, int, np.ndarray, int, int]] = []

    for kind, color, patch in objects:
        h, w = patch.shape
        positions = [
            (r0, c0)
            for r0 in range(1, grid_h - h)
            for c0 in range(1, grid_w - w)
            if _can_place_bbox(placed_boxes, r0, c0, h, w, _PLACEMENT_GAP)
        ]
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        ok = _paint_patch(inp, patch, r0, c0)
        assert ok
        placed_boxes.append((r0, c0, r0 + h - 1, c0 + w - 1))
        placements.append((kind, color, patch, r0, c0))

    out[:] = inp
    for kind, color, patch, r0, c0 in placements:
        h, w = patch.shape
        if kind == "piece":
            out[r0 : r0 + h, c0 : c0 + w][patch != _BG] = _BG
        else:
            out[r0 + 1 : r0 + h - 1, c0 + 1 : c0 + w - 1] = color

    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        case = _generate_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid fragment-to-frame ARC instance")
