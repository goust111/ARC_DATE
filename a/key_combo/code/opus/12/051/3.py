from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8)

GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 80
GRID_SIDE_RANGE = (18, 26)


SMALL_SHAPES = (
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0], [1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1]], dtype=bool),
    np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1], [0, 1], [1, 1]], dtype=bool),
)


@dataclass(frozen=True)
class ArcObject:
    color: int
    coords: tuple[tuple[int, int], ...]
    bbox: tuple[int, int, int, int]
    mask: np.ndarray

    @property
    def area(self) -> int:
        return len(self.coords)


def neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("empty mask")
    r0, c0, r1, c1 = bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy().astype(bool)


def scale_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(mask.astype(int), np.ones((factor, factor), dtype=int)).astype(bool)


def canonical_mask(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    cropped = trim_mask(mask)
    return tuple(map(int, cropped.shape)), tuple(map(int, cropped.astype(int).ravel()))


def component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(comp, dtype=int))
    return components


def extract_objects(grid: np.ndarray) -> list[ArcObject]:
    objs: list[ArcObject] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BLACK):
        comps = component_coords(grid == color)
        for comp in comps:
            r0, c0, r1, c1 = bbox_coords(comp)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[comp[:, 0] - r0, comp[:, 1] - c0] = True
            objs.append(
                ArcObject(
                    color=color,
                    coords=tuple((int(r), int(c)) for r, c in comp),
                    bbox=(r0, c0, r1, c1),
                    mask=mask,
                )
            )
    return objs


def exact_scaled_match(smaller: np.ndarray, larger: np.ndarray) -> bool:
    return (
        larger.shape[0] == 2 * smaller.shape[0]
        and larger.shape[1] == 2 * smaller.shape[1]
        and np.array_equal(scale_mask(smaller, 2), larger)
    )


def find_scale_pairs(objects: list[ArcObject]) -> list[tuple[ArcObject, ArcObject]]:
    matches: list[tuple[int, int]] = []
    for i, small in enumerate(objects):
        for j, big in enumerate(objects):
            if i == j:
                continue
            if exact_scaled_match(small.mask, big.mask):
                matches.append((i, j))

    small_to_big: dict[int, list[int]] = {}
    big_to_small: dict[int, list[int]] = {}
    for i, j in matches:
        small_to_big.setdefault(i, []).append(j)
        big_to_small.setdefault(j, []).append(i)

    assert len(small_to_big) == 2, "expected exactly two smaller matched objects"
    assert len(big_to_small) == 2, "expected exactly two larger matched objects"
    assert set(small_to_big).isdisjoint(set(big_to_small)), "objects cannot be both small and large matches"
    assert all(len(v) == 1 for v in small_to_big.values()), "each small object must match one larger object"
    assert all(len(v) == 1 for v in big_to_small.values()), "each large object must match one smaller object"

    pairs = [(objects[small_i], objects[big_js[0]]) for small_i, big_js in small_to_big.items()]
    pairs.sort(key=lambda pair: pair[0].bbox)
    return pairs


def fixed_corner_color(base_color: int, marker_color: int) -> int:
    return GREY if base_color == marker_color else marker_color


def render_output(shape: tuple[int, int], pairs: list[tuple[ArcObject, ArcObject]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for small, big in pairs:
        for r, c in small.coords:
            out[r, c] = small.color
        r0, c0, r1, c1 = small.bbox
        markers = (
            (r0 - 1, c0 - 1, big.color),
            (r0 - 1, c1 + 1, fixed_corner_color(small.color, RED)),
            (r1 + 1, c0 - 1, fixed_corner_color(small.color, GREEN)),
            (r1 + 1, c1 + 1, fixed_corner_color(small.color, YELLOW)),
        )
        for r, c, color in markers:
            assert 0 <= r < shape[0] and 0 <= c < shape[1], "corner marker out of bounds"
            assert out[r, c] == BLACK, "corner marker overlap"
            out[r, c] = color
    return out


def validate_instance(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if not set(map(int, np.unique(grid))).issubset(set(PALETTE) | {BLACK}):
        return False

    objects = extract_objects(grid)
    if not (5 <= len(objects) <= 7):
        return False
    if len({obj.color for obj in objects}) != len(objects):
        return False

    for i, obj_a in enumerate(objects):
        r0a, c0a, r1a, c1a = obj_a.bbox
        if r0a < 1 or c0a < 1 or r1a >= grid.shape[0] - 1 or c1a >= grid.shape[1] - 1:
            return False
        for obj_b in objects[i + 1 :]:
            r0b, c0b, r1b, c1b = obj_b.bbox
            row_clear = r1a + 2 < r0b or r1b + 2 < r0a
            col_clear = c1a + 2 < c0b or c1b + 2 < c0a
            if not (row_clear or col_clear):
                return False

    try:
        pairs = find_scale_pairs(objects)
    except AssertionError:
        return False

    small_shapes = [canonical_mask(small.mask) for small, _big in pairs]
    if small_shapes[0] == small_shapes[1]:
        return False

    matched_small_colors = {small.color for small, _big in pairs}
    matched_big_colors = {big.color for _small, big in pairs}
    if matched_small_colors & matched_big_colors:
        return False
    if len(objects) - len(matched_small_colors) - len(matched_big_colors) not in (1, 2, 3):
        return False

    for small, _big in pairs:
        r0, c0, r1, c1 = small.bbox
        for r, c in ((r0 - 1, c0 - 1), (r0 - 1, c1 + 1), (r1 + 1, c0 - 1), (r1 + 1, c1 + 1)):
            if grid[r, c] != BLACK:
                return False

    expected = render_output(grid.shape, pairs)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep the smaller member of each 2x scale-matched pair, erase everything else, and decorate each survivor with transferred corner colors.

    Input structure:
    - The grid has black background and 5-7 single-color 4-connected objects whose colors are all distinct.
    - Exactly two pairs are related by a strict factor-2 shape match: one object's cropped mask is the other object's cropped mask scaled by 2.
    - The remaining 1-3 objects are distractors, and each matched smaller object has empty diagonal outside-corner cells around its bounding box.

    Rule:
    - Compare every object pair by cropped boolean mask, matching a smaller object to a larger one only when the larger mask equals the smaller mask scaled by 2.
    - Erase all distractors and both larger matched objects, preserving only the two smaller matched objects in place.
    - Around each preserved smaller object's bounding box, paint the northwest diagonal corner with the erased larger partner's color, the northeast with red, the southwest with green, and the southeast with yellow; if a fixed marker color equals the small object's own color, use grey instead for that marker.

    Key insight:
    - The northwest corner marker is not fixed: it can only be determined after solving the cross-scale shape matching and identifying which larger partner is paired with each surviving smaller object.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    objects = extract_objects(g)
    pairs = find_scale_pairs(objects)
    return render_output(g.shape, pairs)


def random_transform(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    out = np.rot90(out, random.randrange(4))
    return trim_mask(out)


def pick_small_colors() -> tuple[int, int]:
    special = [RED, GREEN, YELLOW]
    ordinary = [1, 6, 7, 8]
    mode = random.choices(("none", "one", "two"), weights=(1, 4, 2), k=1)[0]
    if mode == "none":
        colors = random.sample(ordinary, 2)
    elif mode == "one":
        colors = [random.choice(special), random.choice(ordinary)]
        random.shuffle(colors)
    else:
        colors = random.sample(special, 2)
    return int(colors[0]), int(colors[1])


def mask_has_any_scale_match(mask: np.ndarray, others: list[np.ndarray]) -> bool:
    return any(exact_scaled_match(mask, other) or exact_scaled_match(other, mask) for other in others)


def random_distractor_mask(forbidden: list[np.ndarray]) -> np.ndarray:
    for _ in range(80):
        base = random.choice(SMALL_SHAPES)
        mask = random_transform(base)
        if random.random() < 0.45:
            extra_row = np.zeros((1, mask.shape[1]), dtype=bool)
            extra_col = np.zeros((mask.shape[0], 1), dtype=bool)
            edge = random.randrange(4)
            if edge == 0:
                candidates = np.flatnonzero(mask[0])
                extra_row[0, int(random.choice(candidates))] = True
                mask = np.vstack([extra_row, mask])
            elif edge == 1:
                candidates = np.flatnonzero(mask[-1])
                extra_row[0, int(random.choice(candidates))] = True
                mask = np.vstack([mask, extra_row])
            elif edge == 2:
                candidates = np.flatnonzero(mask[:, 0])
                extra_col[int(random.choice(candidates)), 0] = True
                mask = np.hstack([extra_col, mask])
            else:
                candidates = np.flatnonzero(mask[:, -1])
                extra_col[int(random.choice(candidates)), 0] = True
                mask = np.hstack([mask, extra_col])
            mask = trim_mask(mask)
        if not mask_has_any_scale_match(mask, forbidden):
            return mask
    raise RuntimeError("failed to sample distractor")


def mask_to_sprite(mask: np.ndarray, color: int) -> np.ndarray:
    sprite = np.zeros(mask.shape, dtype=int)
    sprite[mask] = int(color)
    return sprite


def paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    patch = grid[top : top + h, left : left + w]
    patch[mask] = int(color)


def place_objects(specs: list[dict]) -> np.ndarray | None:
    for _ in range(PLACE_ATTEMPTS):
        height = random.randint(*GRID_SIDE_RANGE)
        width = random.randint(*GRID_SIDE_RANGE)
        grid = np.zeros((height, width), dtype=int)
        placed_boxes: list[tuple[int, int, int, int]] = []

        success = True
        for spec in sorted(specs, key=lambda item: item["mask"].shape[0] * item["mask"].shape[1], reverse=True):
            mask = spec["mask"]
            h, w = mask.shape
            candidates: list[tuple[int, int]] = []
            for r0 in range(1, height - h):
                for c0 in range(1, width - w):
                    r1 = r0 + h - 1
                    c1 = c0 + w - 1
                    ok = True
                    for pr0, pc0, pr1, pc1 in placed_boxes:
                        row_clear = r1 + 2 < pr0 or pr1 + 2 < r0
                        col_clear = c1 + 2 < pc0 or pc1 + 2 < c0
                        if not (row_clear or col_clear):
                            ok = False
                            break
                    if ok:
                        candidates.append((r0, c0))
            if not candidates:
                success = False
                break

            top, left = random.choice(candidates)
            paint_mask(grid, mask, top, left, spec["color"])
            placed_boxes.append((top, left, top + h - 1, left + w - 1))

        if success:
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        distractor_count = random.randint(1, 3)
        small_color_1, small_color_2 = pick_small_colors()
        remaining = [color for color in PALETTE if color not in {small_color_1, small_color_2}]
        other_colors = random.sample(remaining, 2 + distractor_count)
        big_color_1, big_color_2 = other_colors[:2]
        distractor_colors = other_colors[2:]

        base_idx_1, base_idx_2 = random.sample(range(len(SMALL_SHAPES)), 2)
        small_mask_1 = random_transform(SMALL_SHAPES[base_idx_1])
        small_mask_2 = random_transform(SMALL_SHAPES[base_idx_2])
        if canonical_mask(small_mask_1) == canonical_mask(small_mask_2):
            continue

        big_mask_1 = scale_mask(small_mask_1, 2)
        big_mask_2 = scale_mask(small_mask_2, 2)

        forbidden = [small_mask_1, small_mask_2, big_mask_1, big_mask_2]
        distractor_masks: list[np.ndarray] = []
        try:
            for _idx, _color in enumerate(distractor_colors):
                mask = random_distractor_mask(forbidden + distractor_masks)
                distractor_masks.append(mask)
        except RuntimeError:
            continue

        specs = [
            {"color": small_color_1, "mask": small_mask_1},
            {"color": big_color_1, "mask": big_mask_1},
            {"color": small_color_2, "mask": small_mask_2},
            {"color": big_color_2, "mask": big_mask_2},
        ]
        for color, mask in zip(distractor_colors, distractor_masks):
            specs.append({"color": int(color), "mask": mask})

        grid = place_objects(specs)
        if grid is None:
            continue
        if validate_instance(grid):
            return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
