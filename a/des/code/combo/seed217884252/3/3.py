from __future__ import annotations

import math
import random

import numpy as np


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _component_from_cells(cells: list[tuple[int, int]], color: int) -> dict:
    coords = np.asarray(cells, dtype=int)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return {
        "color": int(color),
        "area": int(mask.sum()),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "mask": mask,
    }


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == 0 or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            objects.append(_component_from_cells(cells, color))
    return objects


def _normalized_shape(mask: np.ndarray) -> np.ndarray:
    """Compress exact kxk foreground/background blocks to their primitive shape."""
    h, w = mask.shape
    for scale in range(min(h, w), 1, -1):
        if h % scale != 0 or w % scale != 0:
            continue
        blocks = mask.reshape(h // scale, scale, w // scale, scale)
        coarse = blocks[:, 0, :, 0]
        if np.all(blocks == coarse[:, None, :, None]):
            return coarse
    return mask


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    primitive = _normalized_shape(mask)
    return tuple(map(int, primitive.shape)), tuple(int(v) for v in primitive.ravel())


def main(grid: np.ndarray) -> np.ndarray:
    """Sort same-shape differently colored objects by area and emit their colors as one column.

    Input structure
    - The background is black.
    - The foreground consists of 4-connected monochrome objects.
    - One shape family appears multiple times in different colors and different integer scales.

    Rule
    - Normalize each object by dividing exact kxk foreground/background blocks back to a primitive shape.
    - Keep the unique repeated normalized shape family.
    - Sort that family's objects by area from small to large.
    - Output an N x 1 column of the sorted colors.

    Key insight
    - Shared normalized shape identifies the relevant objects; their areas provide the order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    objects = _find_components(g)
    assert objects, "expected foreground objects"

    groups: dict[tuple[tuple[int, int], tuple[int, ...]], list[dict]] = {}
    for obj in objects:
        groups.setdefault(_shape_signature(obj["mask"]), []).append(obj)

    repeated = [group for group in groups.values() if len(group) >= 2]
    assert repeated, "expected at least one repeated normalized shape"

    repeated.sort(
        key=lambda group: (len(group), len({obj["area"] for obj in group})),
        reverse=True,
    )
    best = repeated[0]
    if len(repeated) > 1:
        top_score = (len(best), len({obj["area"] for obj in best}))
        next_score = (len(repeated[1]), len({obj["area"] for obj in repeated[1]}))
        assert top_score > next_score, "ambiguous repeated shape family"

    areas = [int(obj["area"]) for obj in best]
    assert len(set(areas)) == len(areas), "target objects must have distinct areas"
    colors = {int(obj["color"]) for obj in best}
    assert len(colors) == len(best), "target objects must have distinct colors"

    ordered = sorted(best, key=lambda obj: int(obj["area"]))
    return np.array([[int(obj["color"])] for obj in ordered], dtype=int)


# --------------- generator ---------------

TARGET_COUNT_RANGE = (2, 5)
DISTRACTOR_PROB = 0.5
DISTRACTOR_COUNT_RANGE = (1, 3)
BASE_DIM_RANGE = (2, 4)
BASE_AREA_MIN = 3
SCALE_VALUES = (1, 2, 3, 4, 5)
GRID_SLACK_RANGE = (2, 8)
PLACEMENT_MARGIN = 1
MAX_GEN_ATTEMPTS = 400
MASK_GROW_ATTEMPTS = 80
DISTRACTOR_SCALE_VALUES = (1, 2, 3)


def _random_connected_mask(min_dim: int = 2, max_dim: int = 4) -> np.ndarray:
    for _ in range(MASK_GROW_ATTEMPTS):
        h = random.randint(min_dim, max_dim)
        w = random.randint(min_dim, max_dim)
        target_area = random.randint(BASE_AREA_MIN, h * w)

        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True
        frontier = {start}

        while int(mask.sum()) < target_area:
            candidates: set[tuple[int, int]] = set()
            for r, c in frontier:
                for nr, nc in _neighbors4(r, c, h, w):
                    if not mask[nr, nc]:
                        candidates.add((nr, nc))
            if not candidates:
                break
            chosen = random.choice(tuple(candidates))
            mask[chosen] = True
            frontier.add(chosen)

        coords = np.argwhere(mask)
        if coords.size == 0:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        cropped = mask[r0 : r1 + 1, c0 : c1 + 1]
        if min(cropped.shape) < 2:
            continue
        if not cropped[0, :].any() or not cropped[-1, :].any():
            continue
        if not cropped[:, 0].any() or not cropped[:, -1].any():
            continue
        if int(cropped.sum()) < BASE_AREA_MIN:
            continue
        if not np.array_equal(_normalized_shape(cropped), cropped):
            continue
        return cropped
    raise RuntimeError("failed to sample a primitive connected mask")


def _scale_mask(mask: np.ndarray, scale: int) -> np.ndarray:
    return np.kron(mask.astype(np.uint8), np.ones((scale, scale), dtype=np.uint8)).astype(bool)


def _valid_positions(blocked: np.ndarray, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = blocked.shape
    ph, pw = map(int, shape)
    positions: list[tuple[int, int]] = []
    for r in range(PLACEMENT_MARGIN, h - ph - PLACEMENT_MARGIN + 1):
        for c in range(PLACEMENT_MARGIN, w - pw - PLACEMENT_MARGIN + 1):
            r0 = r - PLACEMENT_MARGIN
            r1 = r + ph + PLACEMENT_MARGIN
            c0 = c - PLACEMENT_MARGIN
            c1 = c + pw + PLACEMENT_MARGIN
            if not blocked[r0:r1, c0:c1].any():
                positions.append((r, c))
    return positions


def _paint_object(grid: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> None:
    h, w = mask.shape
    window = grid[top : top + h, left : left + w]
    window[mask] = int(color)


def _place_objects(
    grid: np.ndarray,
    objects: list[dict],
) -> bool:
    blocked = np.zeros_like(grid, dtype=bool)
    order = sorted(
        range(len(objects)),
        key=lambda idx: (
            objects[idx]["mask"].shape[0] * objects[idx]["mask"].shape[1],
            random.random(),
        ),
        reverse=True,
    )
    for idx in order:
        obj = objects[idx]
        positions = _valid_positions(blocked, obj["mask"].shape)
        if not positions:
            return False
        top, left = random.choice(positions)
        _paint_object(grid, obj["mask"], obj["color"], top, left)
        h, w = obj["mask"].shape
        blocked[
            top - PLACEMENT_MARGIN : top + h + PLACEMENT_MARGIN,
            left - PLACEMENT_MARGIN : left + w + PLACEMENT_MARGIN,
        ] = True
    return True


def _expected_output(targets: list[dict]) -> np.ndarray:
    ordered = sorted(targets, key=lambda obj: int(obj["area"]))
    return np.array([[int(obj["color"])] for obj in ordered], dtype=int)


def _sample_target_objects() -> list[dict]:
    for _ in range(MASK_GROW_ATTEMPTS):
        primitive = _random_connected_mask()
        target_count = random.randint(*TARGET_COUNT_RANGE)
        scales = sorted(random.sample(SCALE_VALUES, target_count))
        masks = [_scale_mask(primitive, scale) for scale in scales]
        footprint = sum((m.shape[0] + 2) * (m.shape[1] + 2) for m in masks)
        side_est = max(
            max(m.shape[0] for m in masks) + 2,
            max(m.shape[1] for m in masks) + 2,
            int(math.ceil(math.sqrt(footprint * 1.6))),
        )
        if side_est <= 32:
            colors = random.sample(range(1, 10), target_count)
            return [
                {
                    "mask": mask,
                    "color": color,
                    "area": int(mask.sum()),
                    "signature": _shape_signature(mask),
                }
                for mask, color in zip(masks, colors)
            ]
    raise RuntimeError("failed to sample target objects")


def _sample_distractors(
    target_signature: tuple[tuple[int, int], tuple[int, ...]],
    used_colors: set[int],
) -> list[dict]:
    if random.random() >= DISTRACTOR_PROB:
        return []

    distractor_count = random.randint(*DISTRACTOR_COUNT_RANGE)
    distractors: list[dict] = []
    used_signatures = {target_signature}
    remaining_colors = [color for color in range(1, 10) if color not in used_colors]

    for _ in range(distractor_count):
        if not remaining_colors:
            return []
        built = False
        for _ in range(MASK_GROW_ATTEMPTS):
            primitive = _random_connected_mask()
            scale = random.choice(DISTRACTOR_SCALE_VALUES)
            mask = _scale_mask(primitive, scale)
            signature = _shape_signature(mask)
            if signature in used_signatures:
                continue
            color = random.choice(remaining_colors)
            remaining_colors.remove(color)
            distractors.append(
                {
                    "mask": mask,
                    "color": color,
                    "area": int(mask.sum()),
                    "signature": signature,
                }
            )
            used_signatures.add(signature)
            built = True
            break
        if not built:
            return []
    return distractors


def _sample_grid_shape(objects: list[dict]) -> tuple[int, int]:
    max_h = max(int(obj["mask"].shape[0]) for obj in objects)
    max_w = max(int(obj["mask"].shape[1]) for obj in objects)
    footprint = sum((obj["mask"].shape[0] + 2) * (obj["mask"].shape[1] + 2) for obj in objects)
    base = max(
        max_h + 2 * PLACEMENT_MARGIN + 1,
        max_w + 2 * PLACEMENT_MARGIN + 1,
        int(math.ceil(math.sqrt(footprint * 1.7))),
    )
    h = base + random.randint(*GRID_SLACK_RANGE)
    w = base + random.randint(*GRID_SLACK_RANGE)
    if random.random() < 0.5:
        h += random.randint(0, 4)
    else:
        w += random.randint(0, 4)
    return h, w


def _gen_attempt() -> np.ndarray | None:
    targets = _sample_target_objects()
    target_signature = targets[0]["signature"]
    distractors = _sample_distractors(target_signature, {int(obj["color"]) for obj in targets})
    objects = targets + distractors

    h, w = _sample_grid_shape(objects)
    grid = np.zeros((h, w), dtype=int)
    if not _place_objects(grid, objects):
        return None

    expected = _expected_output(targets)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if np.array_equal(actual, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
