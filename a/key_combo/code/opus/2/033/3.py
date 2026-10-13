from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
FRAME_COLOR = 5
SPRITE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GEN_ATTEMPTS = 400
_SHAPE_POOL: list[np.ndarray] | None = None


def _get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    m = np.asarray(mask, dtype=bool)
    h, w = m.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not m[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and m[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    m = np.asarray(mask, dtype=bool)
    return (int(m.shape[0]), int(m.shape[1])), tuple(int(x) for x in m.ravel())


def _transform_mask(mask: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(np.asarray(mask, dtype=bool), k)
    if flip:
        out = np.fliplr(out)
    return out.copy()


def _variant_masks(mask: np.ndarray) -> list[tuple[int, bool, np.ndarray]]:
    seen: set[tuple[tuple[int, int], tuple[int, ...]]] = set()
    variants: list[tuple[int, bool, np.ndarray]] = []
    for k in range(4):
        for flip in (False, True):
            out = _transform_mask(mask, k, flip)
            key = _mask_key(out)
            if key in seen:
                continue
            seen.add(key)
            variants.append((int(k), bool(flip), out))
    return variants


def _canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return min(_mask_key(variant) for _k, _flip, variant in _variant_masks(mask))


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(np.asarray(mask, dtype=bool))
    r0, c0, r1, c1 = _bbox_coords(coords)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool).copy()


def _shape_pool() -> list[np.ndarray]:
    global _SHAPE_POOL
    if _SHAPE_POOL is not None:
        return _SHAPE_POOL

    seen: set[tuple[tuple[int, int], tuple[int, ...]]] = set()
    pool: list[np.ndarray] = []
    for area in range(4, 9):
        for cells in itertools.combinations(range(16), area):
            mask = np.zeros((4, 4), dtype=bool)
            mask.flat[list(cells)] = True
            if len(_components(mask)) != 1:
                continue
            tight = _normalize_mask(mask)
            if min(tight.shape) == 1 or bool(tight.all()):
                continue
            if len(_variant_masks(tight)) != 8:
                continue
            canon = _canonical_key(tight)
            if canon in seen:
                continue
            seen.add(canon)
            pool.append(tight)

    random.shuffle(pool)
    _SHAPE_POOL = pool
    return _SHAPE_POOL


def _cavity_specs(frame_crop: np.ndarray) -> list[dict]:
    cavities = []
    h, w = frame_crop.shape
    for coords in _components(frame_crop == BG):
        if np.any(coords[:, 0] == 0) or np.any(coords[:, 0] == h - 1):
            continue
        if np.any(coords[:, 1] == 0) or np.any(coords[:, 1] == w - 1):
            continue
        r0, c0, r1, c1 = _bbox_coords(coords)
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        local[coords[:, 0] - r0, coords[:, 1] - c0] = True
        cavities.append({"bbox": (r0, c0, r1, c1), "mask": local})
    cavities.sort(key=lambda item: item["bbox"])
    return cavities


def _sprite_specs(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> list[dict]:
    work = np.asarray(grid, dtype=int).copy()
    r0, c0, r1, c1 = frame_bbox
    work[r0 : r1 + 1, c0 : c1 + 1] = BG
    sprites = []
    colors = [int(c) for c in np.unique(work) if int(c) not in (BG, FRAME_COLOR)]
    for color in colors:
        for coords in _components(work == int(color)):
            br0, bc0, br1, bc1 = _bbox_coords(coords)
            local = np.zeros((br1 - br0 + 1, bc1 - bc0 + 1), dtype=bool)
            local[coords[:, 0] - br0, coords[:, 1] - bc0] = True
            sprites.append(
                {
                    "color": int(color),
                    "bbox": (br0, bc0, br1, bc1),
                    "mask": local,
                }
            )
    sprites.sort(key=lambda item: (item["bbox"], item["color"]))
    return sprites


def _match_sprites_to_cavities(sprites: list[dict], cavities: list[dict]) -> dict[int, tuple[int, int, bool]]:
    options: list[list[tuple[int, int, bool]]] = []
    for sprite in sprites:
        by_cavity: dict[int, tuple[int, bool]] = {}
        for cavity_index, cavity in enumerate(cavities):
            for k in range(4):
                for flip in (False, True):
                    variant = _transform_mask(sprite["mask"], k, flip)
                    if variant.shape == cavity["mask"].shape and np.array_equal(variant, cavity["mask"]):
                        by_cavity.setdefault(int(cavity_index), (int(k), bool(flip)))
        opts = [(cavity_index, k, flip) for cavity_index, (k, flip) in sorted(by_cavity.items())]
        options.append(opts)

    order = sorted(range(len(sprites)), key=lambda i: len(options[i]))
    assignment: dict[int, tuple[int, int, bool]] = {}
    used_cavities: set[int] = set()

    def search(pos: int) -> bool:
        if pos == len(order):
            return True
        sprite_index = order[pos]
        for cavity_index, k, flip in options[sprite_index]:
            if cavity_index in used_cavities:
                continue
            assignment[sprite_index] = (int(cavity_index), int(k), bool(flip))
            used_cavities.add(int(cavity_index))
            if search(pos + 1):
                return True
            used_cavities.remove(int(cavity_index))
            del assignment[sprite_index]
        return False

    assert all(options), "every sprite must match at least one cavity"
    assert search(0), "expected a unique sprite/cavity matching"
    return assignment


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each irregular cavity in the gray template with the uniquely matching rotated/flipped outside sprite.

    Input structure:
    - The grid has a black background, one large gray 4-connected template region, and 3-4 small colored sprites outside the template bbox.
    - Inside the template bbox, the gray region surrounds several enclosed black cavities; each cavity is a 4-connected non-rectangular hole.
    - Each outside sprite is a single-color 4-connected shape whose mask matches exactly one cavity up to rotation/reflection.

    Rule:
    - Find the largest gray component and crop to its bounding box.
    - Inside that crop, collect every enclosed black 4-connected cavity that does not touch the crop border.
    - For each outside colored sprite, test all 8 D4 variants (4 rotations x optional horizontal flip) against every cavity mask.
    - Use the unique sprite-to-cavity pairing and paint each matched cavity with the sprite's color in the matched orientation.
    - Return only the template bounding-box region: gray frame plus the filled cavities.

    Key insight:
    - Size alone is insufficient because multiple cavities can share similar boxes; the decisive signal is exact pixel-mask equality after trying the full rotation/reflection orbit.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame_components = _components(g == FRAME_COLOR)
    assert frame_components, "expected a gray template"
    frame_coords = max(frame_components, key=lambda comp: int(comp.shape[0]))
    frame_bbox = _bbox_coords(frame_coords)
    frame_crop = _crop_bbox(g, frame_bbox)

    cavities = _cavity_specs(frame_crop)
    sprites = _sprite_specs(g, frame_bbox)
    assert cavities, "expected enclosed cavities inside the template"
    assert len(cavities) == len(sprites), "sprite and cavity counts must match"

    assignment = _match_sprites_to_cavities(sprites, cavities)
    out = frame_crop.copy()
    for sprite_index, (cavity_index, k, flip) in assignment.items():
        cavity = cavities[cavity_index]
        variant = _transform_mask(sprites[sprite_index]["mask"], k, flip)
        r0, c0, r1, c1 = cavity["bbox"]
        view = out[r0 : r1 + 1, c0 : c1 + 1]
        view[variant] = int(sprites[sprite_index]["color"])
    return out


def _pick_shapes(n: int) -> list[np.ndarray]:
    pool = list(_shape_pool())
    random.shuffle(pool)
    return [pool[i].copy() for i in range(n)]


def _pick_transform(mask: np.ndarray, kind: str) -> tuple[int, bool]:
    if kind == "identity":
        return 0, False

    candidates: list[tuple[int, bool]] = []
    for k in range(4):
        for flip in (False, True):
            if kind == "rotate" and (flip or k == 0):
                continue
            if kind == "flip" and not flip:
                continue
            if kind == "any" and k == 0 and not flip:
                continue
            if np.array_equal(_transform_mask(mask, k, flip), mask):
                continue
            candidates.append((int(k), bool(flip)))

    assert candidates, f"no transform available for kind={kind}"
    return random.choice(candidates)


def _hole_layout(masks: list[np.ndarray], frame_h: int, frame_w: int) -> list[tuple[int, int]] | None:
    order = sorted(
        range(len(masks)),
        key=lambda idx: masks[idx].shape[0] * masks[idx].shape[1],
        reverse=True,
    )
    placements: list[tuple[int, int] | None] = [None] * len(masks)

    def search(pos: int, reserved: np.ndarray) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        mask = masks[idx]
        h, w = mask.shape
        candidates = []
        for r in range(1, frame_h - h):
            for c in range(1, frame_w - w):
                rr0 = max(0, r - 1)
                rr1 = min(frame_h, r + h + 1)
                cc0 = max(0, c - 1)
                cc1 = min(frame_w, c + w + 1)
                if reserved[rr0:rr1, cc0:cc1].any():
                    continue
                candidates.append((int(r), int(c)))
        random.shuffle(candidates)
        for r, c in candidates:
            next_reserved = reserved.copy()
            next_reserved[max(0, r - 1) : min(frame_h, r + h + 1), max(0, c - 1) : min(frame_w, c + w + 1)] = True
            placements[idx] = (int(r), int(c))
            if search(pos + 1, next_reserved):
                return True
            placements[idx] = None
        return False

    reserved = np.zeros((frame_h, frame_w), dtype=bool)
    reserved[0, :] = True
    reserved[-1, :] = True
    reserved[:, 0] = True
    reserved[:, -1] = True
    if not search(0, reserved):
        return None
    return [placement for placement in placements if placement is not None]


def _build_frame(masks: list[np.ndarray], colors: list[int]) -> tuple[np.ndarray, np.ndarray] | None:
    padded_area = sum((mask.shape[0] + 2) * (mask.shape[1] + 2) for mask in masks)
    base = max(12, int(np.ceil(np.sqrt(padded_area))) + 2)
    for _ in range(40):
        frame_h = random.randint(base, min(16, base + 3))
        frame_w = random.randint(base, min(16, base + 3))
        layout = _hole_layout(masks, frame_h, frame_w)
        if layout is None or len(layout) != len(masks):
            continue

        frame = np.full((frame_h, frame_w), FRAME_COLOR, dtype=int)
        expected = frame.copy()
        for mask, color, (r0, c0) in zip(masks, colors, layout):
            h, w = mask.shape
            frame[r0 : r0 + h, c0 : c0 + w][mask] = BG
            expected[r0 : r0 + h, c0 : c0 + w][mask] = int(color)
        return frame, expected
    return None


def _sprite_patches(masks: list[np.ndarray], colors: list[int]) -> list[np.ndarray]:
    plans = ["identity", "rotate", "flip"]
    if len(masks) == 4:
        plans.append(random.choice(["rotate", "flip", "any"]))
    random.shuffle(plans)

    patches = []
    for mask, color, plan in zip(masks, colors, plans):
        k, flip = _pick_transform(mask, plan)
        variant = _transform_mask(mask, k, flip)
        patch = np.zeros(variant.shape, dtype=int)
        patch[variant] = int(color)
        patches.append(patch)
    return patches


def _place_sprites(
    canvas: np.ndarray,
    patches: list[np.ndarray],
    frame_pos: tuple[int, int],
    frame_shape: tuple[int, int],
) -> bool:
    h, w = canvas.shape
    frame_r, frame_c = frame_pos
    frame_h, frame_w = frame_shape
    reserved = np.zeros((h, w), dtype=bool)
    reserved[
        max(0, frame_r - 1) : min(h, frame_r + frame_h + 1),
        max(0, frame_c - 1) : min(w, frame_c + frame_w + 1),
    ] = True

    order = sorted(
        range(len(patches)),
        key=lambda idx: patches[idx].shape[0] * patches[idx].shape[1],
        reverse=True,
    )
    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        candidates = []
        for r in range(1, h - ph):
            for c in range(1, w - pw):
                rr0 = r - 1
                rr1 = r + ph + 1
                cc0 = c - 1
                cc1 = c + pw + 1
                if reserved[rr0:rr1, cc0:cc1].any():
                    continue
                candidates.append((int(r), int(c)))
        if not candidates:
            return False
        r0, c0 = random.choice(candidates)
        canvas[r0 : r0 + ph, c0 : c0 + pw][patch != BG] = patch[patch != BG]
        reserved[r0 - 1 : r0 + ph + 1, c0 - 1 : c0 + pw + 1] = True
    return True


def _gen_attempt() -> np.ndarray | None:
    n = random.choice((3, 4))
    masks = _pick_shapes(n)
    colors = random.sample(SPRITE_COLORS, n)
    frame_pair = _build_frame(masks, colors)
    if frame_pair is None:
        return None
    frame, expected = frame_pair

    patches = _sprite_patches(masks, colors)
    frame_h, frame_w = frame.shape
    min_h = max(18, frame_h + 8)
    max_h = min(30, frame_h + 13)
    min_w = max(18, frame_w + 8)
    max_w = min(30, frame_w + 13)
    for _ in range(30):
        canvas_h = random.randint(min_h, max_h)
        canvas_w = random.randint(min_w, max_w)
        frame_r = random.randint(2, canvas_h - frame_h - 2)
        frame_c = random.randint(2, canvas_w - frame_w - 2)

        grid = np.zeros((canvas_h, canvas_w), dtype=int)
        grid[frame_r : frame_r + frame_h, frame_c : frame_c + frame_w] = frame
        if not _place_sprites(grid, patches, (frame_r, frame_c), frame.shape):
            continue
        if np.array_equal(main(grid), expected):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
