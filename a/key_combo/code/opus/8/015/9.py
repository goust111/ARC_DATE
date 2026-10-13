from __future__ import annotations

from collections import Counter
import itertools
from typing import Iterable

import numpy as np


BG = 0
GUIDE = 5
SPRITE_SIZE = 3
OUTPUT_SIZE = 9
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 500


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _bbox_from_coords(coords: Iterable[Coord]) -> BBox:
    pts = list(coords)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_bbox(grid: np.ndarray, box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _find_components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _contains(outer: BBox, inner: BBox) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 <= ir0 and oc0 <= ic0 and ir1 <= or1 and ic1 <= oc1


def _sprite_objects(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    sprite_mask = np.zeros_like(grid, dtype=bool)
    sprites: list[dict] = []
    for comp in _find_components(grid != BG):
        box = _bbox_from_coords(comp)
        patch = _crop_bbox(grid, box)
        if len(comp) == 9 and patch.shape == (SPRITE_SIZE, SPRITE_SIZE) and np.all(patch != BG):
            for r, c in comp:
                sprite_mask[r, c] = True
            sprites.append({"bbox": box, "patch": patch})
    return sprites, sprite_mask


def _guide_box(grid: np.ndarray, sprite_mask: np.ndarray) -> BBox:
    guide_coords = [tuple(map(int, rc)) for rc in np.argwhere((grid == GUIDE) & ~sprite_mask)]
    assert guide_coords, "expected guide pixels"
    return _bbox_from_coords(guide_coords)


def _major_color(template: np.ndarray) -> int:
    counts = Counter(map(int, template.reshape(-1)))
    color, freq = counts.most_common(1)[0]
    assert list(counts.values()).count(freq) == 1, "major color must be unique"
    return int(color)


def _stamp_from_template(template: np.ndarray) -> np.ndarray:
    major = _major_color(template)
    out = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=int)
    for rr, cc in np.argwhere(template == major):
        _paste(out, template, int(rr) * SPRITE_SIZE, int(cc) * SPRITE_SIZE)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the uniquely guide-enclosed 3x3 sprite and stamp it into a 9x9 array at its own major-color positions.

    Input structure:
    - A black background contains 3 or 4 separated multicolor 3x3 sprites whose 9 cells are all non-black.
    - Sparse grey guide pixels lie outside the sprites and define one guide-mask bounding box.
    - Exactly one sprite bbox is fully contained in that guide bbox.

    Rule:
    - Find every foreground object whose tight bbox is 3x3 and whose crop is fully non-black.
    - Build the guide mask from grey pixels that are not part of those sprites, then take the guide mask bbox.
    - Select the unique sprite fully contained in that bbox and crop its 3x3 pattern.
    - In that pattern, find the color with strictly highest frequency.
    - Return a 9x9 black grid split into 3x3 slots; wherever the major color appears in the template, copy the whole 3x3 template into the corresponding slot.

    Key insight:
    - The guide mask only chooses which existing sprite becomes the template; that same chosen template then determines both stamp content and stamp placement via its own most-frequent-color cells.
    """
    g = np.asarray(grid, dtype=int)
    sprites, sprite_mask = _sprite_objects(g)
    assert len(sprites) >= 2, "expected multiple sprite candidates"

    guide_box = _guide_box(g, sprite_mask)
    selected = [sprite for sprite in sprites if _contains(guide_box, sprite["bbox"])]
    assert len(selected) == 1, "guide bbox must select exactly one sprite"
    return _stamp_from_template(selected[0]["patch"])


def _bbox_from_top_left(r0: int, c0: int) -> BBox:
    return r0, c0, r0 + SPRITE_SIZE - 1, c0 + SPRITE_SIZE - 1


def _boxes_disjoint(a: BBox, b: BBox) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0


def _sprite_gap_ok(a: BBox, b: BBox, gap: int = 2) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _expanded(box: BBox, pad: int, shape: tuple[int, int]) -> BBox:
    h, w = shape
    r0, c0, r1, c1 = box
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _random_major_masks(rng: np.random.Generator, count: int) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    while len(masks) < count:
        size = int(rng.integers(4, 7))
        flat = np.zeros(9, dtype=bool)
        flat[rng.choice(9, size=size, replace=False)] = True
        key = tuple(int(v) for v in flat)
        if key in seen:
            continue
        seen.add(key)
        masks.append(flat.reshape(3, 3))
    return masks


def _fill_template(rng: np.random.Generator, major_mask: np.ndarray) -> np.ndarray:
    template = np.empty((3, 3), dtype=int)
    colors = list(map(int, rng.choice(PALETTE, size=3, replace=False)))
    major_color = colors[0]
    template[major_mask] = major_color

    remainder = [(int(r), int(c)) for r, c in np.argwhere(~major_mask)]
    rng.shuffle(remainder)
    major_count = int(major_mask.sum())
    rem = len(remainder)
    use_three = major_count == 4 or (rem >= 2 and rng.random() < 0.45)

    if use_three:
        low = max(1, rem - (major_count - 1))
        high = min(rem - 1, major_count - 1)
        split = int(rng.integers(low, high + 1))
        for r, c in remainder[:split]:
            template[r, c] = colors[1]
        for r, c in remainder[split:]:
            template[r, c] = colors[2]
    else:
        template[~major_mask] = colors[1]

    assert 2 <= len(np.unique(template)) <= 3
    _major_color(template)
    return template


def _make_guide_cells(rng: np.random.Generator, box: BBox) -> list[Coord]:
    r0, c0, r1, c1 = box
    perimeter = [
        (r0, c) for c in range(c0, c1 + 1)
    ] + [
        (r, c1) for r in range(r0 + 1, r1 + 1)
    ] + [
        (r1, c) for c in range(c1 - 1, c0 - 1, -1)
    ] + [
        (r, c0) for r in range(r1 - 1, r0, -1)
    ]
    required = {
        (r0, int(rng.integers(c0, c1 + 1))),
        (r1, int(rng.integers(c0, c1 + 1))),
        (int(rng.integers(r0, r1 + 1)), c0),
        (int(rng.integers(r0, r1 + 1)), c1),
    }
    extras_pool = [cell for cell in perimeter if cell not in required]
    extra_count = int(rng.integers(2, min(7, len(extras_pool)) + 1))
    extras = rng.choice(len(extras_pool), size=extra_count, replace=False)
    cells = set(required)
    for idx in np.atleast_1d(extras):
        cells.add(extras_pool[int(idx)])
    return sorted(cells)


def _candidate_positions(
    shape: tuple[int, int], reserved: BBox, selected_box: BBox
) -> list[Coord]:
    h, w = shape
    blocked = _expanded(reserved, 1, shape)
    positions: list[Coord] = []
    for r in range(h - SPRITE_SIZE + 1):
        for c in range(w - SPRITE_SIZE + 1):
            box = _bbox_from_top_left(r, c)
            if box == selected_box:
                continue
            if not _boxes_disjoint(box, blocked):
                continue
            positions.append((r, c))
    return positions


def _pick_other_boxes(
    rng: np.random.Generator,
    shape: tuple[int, int],
    selected_box: BBox,
    guide_box: BBox,
    need: int,
) -> list[BBox] | None:
    coords = _candidate_positions(shape, guide_box, selected_box)
    rng.shuffle(coords)
    chosen = [selected_box]
    others: list[BBox] = []
    for r, c in coords:
        box = _bbox_from_top_left(r, c)
        if all(_sprite_gap_ok(box, prev) for prev in chosen):
            chosen.append(box)
            others.append(box)
            if len(others) == need:
                return others
    return None


def _analyze_input(grid: np.ndarray) -> dict:
    sprites, sprite_mask = _sprite_objects(grid)
    guide_box = _guide_box(grid, sprite_mask)
    selected = [sprite for sprite in sprites if _contains(guide_box, sprite["bbox"])]
    return {"sprites": sprites, "sprite_mask": sprite_mask, "guide_box": guide_box, "selected": selected}


def _validate_semantics(grid: np.ndarray) -> bool:
    info = _analyze_input(grid)
    sprites = info["sprites"]
    sprite_mask = info["sprite_mask"]
    guide_box = info["guide_box"]
    selected = info["selected"]

    if len(sprites) not in (3, 4) or len(selected) != 1:
        return False
    if np.any((grid == GUIDE) & sprite_mask):
        return False

    guide_mask = (grid == GUIDE) & ~sprite_mask
    if not np.any(guide_mask):
        return False
    if _bbox_from_coords([tuple(map(int, rc)) for rc in np.argwhere(guide_mask)]) != guide_box:
        return False

    major_masks = []
    outputs = []
    for sprite in sprites:
        patch = sprite["patch"]
        if patch.shape != (3, 3) or np.any(patch == BG):
            return False
        colors = np.unique(patch)
        if not (2 <= len(colors) <= 3):
            return False
        major = _major_color(patch)
        major_mask = tuple(int(v) for v in (patch == major).reshape(-1))
        major_masks.append(major_mask)
        outputs.append(_stamp_from_template(patch))

    if len(set(major_masks)) != len(major_masks):
        return False

    hashes = {out.tobytes() for out in outputs}
    if len(hashes) != len(outputs):
        return False

    boxes = [sprite["bbox"] for sprite in sprites]
    for a, b in itertools.combinations(boxes, 2):
        if not _sprite_gap_ok(a, b):
            return False
    if sum(1 for box in boxes if _contains(guide_box, box)) != 1:
        return False

    selected_patch = selected[0]["patch"]
    if not np.array_equal(main(grid), _stamp_from_template(selected_patch)):
        return False
    return True


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(14, 19))
    w = int(rng.integers(14, 19))
    count = int(rng.choice([3, 4]))

    sel_r = int(rng.integers(1, h - 3))
    sel_c = int(rng.integers(1, w - 3))
    selected_box = _bbox_from_top_left(sel_r, sel_c)

    max_top = min(3, sel_r)
    max_left = min(3, sel_c)
    max_bottom = min(3, h - 1 - selected_box[2])
    max_right = min(3, w - 1 - selected_box[3])
    if min(max_top, max_left, max_bottom, max_right) < 1:
        return None

    top = int(rng.integers(1, max_top + 1))
    left = int(rng.integers(1, max_left + 1))
    bottom = int(rng.integers(1, max_bottom + 1))
    right = int(rng.integers(1, max_right + 1))
    guide_box = (
        selected_box[0] - top,
        selected_box[1] - left,
        selected_box[2] + bottom,
        selected_box[3] + right,
    )

    other_boxes = _pick_other_boxes(rng, (h, w), selected_box, guide_box, count - 1)
    if other_boxes is None:
        return None

    masks = _random_major_masks(rng, count)
    templates = [_fill_template(rng, mask) for mask in masks]
    order = np.arange(count)
    rng.shuffle(order)
    boxes = [selected_box] + other_boxes

    grid = np.zeros((h, w), dtype=int)
    for idx, box in zip(order, boxes):
        r0, c0, _r1, _c1 = box
        _paste(grid, templates[int(idx)], r0, c0)

    guide_cells = _make_guide_cells(rng, guide_box)
    for r, c in guide_cells:
        if grid[r, c] != BG:
            return None
        grid[r, c] = GUIDE

    if not _validate_semantics(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
