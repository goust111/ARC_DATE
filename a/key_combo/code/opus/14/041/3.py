from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
FRAME = 1
SPRITE_COLORS = [2, 3, 4, 5, 6, 7, 8, 9]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _component_patch(grid: np.ndarray, coords: np.ndarray, fill: int = BG) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(fill), dtype=int)
    for rr, cc in coords:
        patch[int(rr) - r0, int(cc) - c0] = int(grid[int(rr), int(cc)])
    return patch


def _scale_patch(patch: np.ndarray, factor: int) -> np.ndarray:
    k = int(factor)
    return np.kron(patch, np.ones((k, k), dtype=int)).astype(int)


def _largest_blue_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    comps = _components_from_mask(grid == FRAME)
    assert comps
    comps.sort(key=lambda coords: (-len(coords), _bbox(coords)))
    if len(comps) > 1:
        assert len(comps[0]) > len(comps[1])
    return _bbox(comps[0])


def _rectangular_cavities(frame_crop: np.ndarray) -> list[dict]:
    cavities = []
    for coords in _components_from_mask(frame_crop == BG):
        r0, c0, r1, c1 = _bbox(coords)
        if r0 == 0 or c0 == 0 or r1 == frame_crop.shape[0] - 1 or c1 == frame_crop.shape[1] - 1:
            continue
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if len(coords) != h * w:
            continue
        cavities.append({"bbox": (r0, c0, r1, c1), "shape": (h, w), "area": h * w})
    cavities.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return cavities


def _outer_sprites(grid: np.ndarray, frame_box: tuple[int, int, int, int]) -> list[dict]:
    masked = np.array(grid, dtype=int)
    r0, c0, r1, c1 = frame_box
    masked[r0 : r1 + 1, c0 : c1 + 1] = BG
    sprites = []
    for color in sorted(set(map(int, masked.flatten())) - {BG, FRAME}):
        for coords in _components_from_mask(masked == int(color)):
            patch = _component_patch(masked, coords, fill=BG)
            sprites.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "count": int(len(coords)),
                    "patch": patch,
                    "bbox": _bbox(coords),
                }
            )
    sprites.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["color"]))
    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scale each outside sprite by its own pixel count, then fit it into the unique cavity of matching size.

    Input structure:
    - Black background surrounds one large blue template region.
    - Inside the blue template bbox are several black rectangular cavities.
    - Outside that bbox are several disconnected monochrome sprites in non-black, non-blue colors.

    Rule:
    - Use the largest blue connected component as the template and crop to its bbox.
    - For each outside sprite, count its pixels and scale its whole bbox by that factor, including black cells inside the bbox.
    - Match each scaled sprite to the unique cavity whose rectangle has the same height and width, then place it left-aligned and top-aligned into that cavity.

    Key insight:
    - The cavities match the sprites only after each sprite is independently magnified by its own area, so counting, scaling, and size-matching are all required.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frame_box = _largest_blue_bbox(g)
    frame_crop = _crop_bbox(g, frame_box)
    cavities = _rectangular_cavities(frame_crop)
    assert cavities
    cavity_by_shape = {}
    for cavity in cavities:
        assert cavity["shape"] not in cavity_by_shape
        cavity_by_shape[cavity["shape"]] = cavity

    sprites = _outer_sprites(g, frame_box)
    assert len(sprites) == len(cavities)

    out = frame_crop.copy()
    for sprite in sprites:
        factor = int(sprite["count"])
        scaled = _scale_patch(sprite["patch"], factor)
        shape = tuple(map(int, scaled.shape))
        assert shape in cavity_by_shape
        cavity = cavity_by_shape.pop(shape)
        r0, c0, r1, c1 = cavity["bbox"]
        assert (r1 - r0 + 1, c1 - c0 + 1) == shape
        out[r0 : r1 + 1, c0 : c1 + 1] = scaled

    assert not cavity_by_shape
    return out


def _patch_from_cells(cells: set[tuple[int, int]], color: int) -> np.ndarray:
    coords = np.array(sorted(cells), dtype=int)
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in cells:
        patch[r - r0, c - c0] = int(color)
    return patch


def _choose_sprite_patches() -> list[np.ndarray]:
    colors = random.sample(SPRITE_COLORS, 3)
    domino = random.choice(
        [
            _patch_from_cells({(0, 0), (0, 1)}, colors[0]),
            _patch_from_cells({(0, 0), (1, 0)}, colors[0]),
        ]
    )
    line = random.choice(
        [
            _patch_from_cells({(0, 0), (0, 1), (0, 2)}, colors[1]),
            _patch_from_cells({(0, 0), (1, 0), (2, 0)}, colors[1]),
        ]
    )
    l_cells = random.choice(
        [
            {(0, 0), (0, 1), (1, 0)},
            {(0, 0), (0, 1), (1, 1)},
            {(0, 0), (1, 0), (1, 1)},
            {(0, 1), (1, 0), (1, 1)},
        ]
    )
    corner = _patch_from_cells(l_cells, colors[2])
    return [domino, line, corner]


def _expanded_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _place_holes(shapes: list[tuple[int, int]]) -> tuple[int, int, list[tuple[int, int, int, int]]]:
    hs = [h for h, _ in shapes]
    ws = [w for _, w in shapes]
    min_h = max(max(hs) + 2, (sum(hs) + 1) // 2 + 4)
    min_w = max(max(ws) + 2, (sum(ws) + 1) // 2 + 4)
    for _ in range(400):
        frame_h = random.randint(min_h, min(18, min_h + 5))
        frame_w = random.randint(min_w, min(18, min_w + 5))
        order = list(range(len(shapes)))
        order.sort(key=lambda i: -(shapes[i][0] * shapes[i][1]))
        placed: dict[int, tuple[int, int, int, int]] = {}
        ok = True
        for idx in order:
            h, w = shapes[idx]
            cells = []
            for r0 in range(1, frame_h - h):
                for c0 in range(1, frame_w - w):
                    box = (r0, c0, r0 + h - 1, c0 + w - 1)
                    if any(_expanded_overlap(box, other, pad=1) for other in placed.values()):
                        continue
                    cells.append(box)
            if not cells:
                ok = False
                break
            placed[idx] = random.choice(cells)
        if ok:
            return frame_h, frame_w, [placed[i] for i in range(len(shapes))]
    raise RuntimeError("failed to place cavity rectangles")


def _place_frame_on_canvas(frame_h: int, frame_w: int) -> tuple[int, int, int, int]:
    h = random.randint(max(frame_h + 6, 18), min(frame_h + 11, 28))
    w = random.randint(max(frame_w + 6, 18), min(frame_w + 11, 28))
    r0 = random.randint(2, h - frame_h - 2)
    c0 = random.randint(2, w - frame_w - 2)
    return h, w, r0, c0


def _place_outer_sprites(
    grid: np.ndarray,
    frame_box: tuple[int, int, int, int],
    sprite_patches: list[np.ndarray],
) -> None:
    h, w = grid.shape
    occupied = [frame_box]
    order = list(range(len(sprite_patches)))
    random.shuffle(order)
    for idx in order:
        patch = sprite_patches[idx]
        ph, pw = patch.shape
        candidates = []
        for r0 in range(h - ph + 1):
            for c0 in range(w - pw + 1):
                box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
                if any(_expanded_overlap(box, other, pad=1) for other in occupied):
                    continue
                if np.any(grid[r0 : r0 + ph, c0 : c0 + pw] != BG):
                    continue
                candidates.append((r0, c0))
        if not candidates:
            raise RuntimeError("failed to place outer sprites")
        r0, c0 = random.choice(candidates)
        mask = patch != BG
        grid[r0 : r0 + ph, c0 : c0 + pw][mask] = patch[mask]
        occupied.append((r0, c0, r0 + ph - 1, c0 + pw - 1))


def _expected_output_from_parts(
    frame_h: int,
    frame_w: int,
    hole_boxes: list[tuple[int, int, int, int]],
    scaled_patches: list[np.ndarray],
) -> np.ndarray:
    out = np.full((frame_h, frame_w), FRAME, dtype=int)
    for box, patch in zip(hole_boxes, scaled_patches):
        r0, c0, r1, c1 = box
        assert patch.shape == (r1 - r0 + 1, c1 - c0 + 1)
        out[r0 : r1 + 1, c0 : c1 + 1] = patch
    return out


def _semantic_checks(inp: np.ndarray, expected: np.ndarray) -> bool:
    out = main(inp)
    if not np.array_equal(out, expected):
        return False

    frame_box = _largest_blue_bbox(inp)
    frame_crop = _crop_bbox(inp, frame_box)
    cavities = _rectangular_cavities(frame_crop)
    if len(cavities) != 3:
        return False
    if len({tuple(c["shape"]) for c in cavities}) != 3:
        return False

    sprites = _outer_sprites(inp, frame_box)
    if len(sprites) != 3:
        return False
    counts = sorted(int(s["count"]) for s in sprites)
    if counts != [2, 3, 3]:
        return False

    scaled_shapes = []
    raw_shapes = []
    for sprite in sprites:
        scaled = _scale_patch(sprite["patch"], sprite["count"])
        scaled_shapes.append(tuple(map(int, scaled.shape)))
        raw_shapes.append(tuple(map(int, sprite["patch"].shape)))
    cavity_shapes = [tuple(map(int, c["shape"])) for c in cavities]

    if Counter(scaled_shapes) != Counter(cavity_shapes):
        return False
    if any(shape in cavity_shapes for shape in raw_shapes):
        return False

    # Reject accidental global-scaling shortcuts: no single factor may explain every cavity.
    all_ok = []
    for factor in range(1, 5):
        guessed = [tuple(int(dim * factor) for dim in sprite["patch"].shape) for sprite in sprites]
        all_ok.append(Counter(guessed) == Counter(cavity_shapes))
    if sum(all_ok) != 0:
        return False

    return True


def _generate_instance() -> tuple[np.ndarray, np.ndarray]:
    sprite_patches = _choose_sprite_patches()
    scaled_patches = [_scale_patch(patch, int(np.count_nonzero(patch))) for patch in sprite_patches]
    hole_shapes = [tuple(map(int, patch.shape)) for patch in scaled_patches]

    frame_h, frame_w, hole_boxes = _place_holes(hole_shapes)
    expected = _expected_output_from_parts(frame_h, frame_w, hole_boxes, scaled_patches)

    grid_h, grid_w, frame_r0, frame_c0 = _place_frame_on_canvas(frame_h, frame_w)
    grid = np.zeros((grid_h, grid_w), dtype=int)
    frame = np.full((frame_h, frame_w), FRAME, dtype=int)
    for box in hole_boxes:
        r0, c0, r1, c1 = box
        frame[r0 : r1 + 1, c0 : c1 + 1] = BG
    grid[frame_r0 : frame_r0 + frame_h, frame_c0 : frame_c0 + frame_w] = frame
    frame_box = (frame_r0, frame_c0, frame_r0 + frame_h - 1, frame_c0 + frame_w - 1)

    _place_outer_sprites(grid, frame_box, sprite_patches)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(300):
        inp, expected = _generate_instance()
        if _semantic_checks(inp, expected):
            return inp
    raise RuntimeError("failed to generate a valid non-degenerate instance")
