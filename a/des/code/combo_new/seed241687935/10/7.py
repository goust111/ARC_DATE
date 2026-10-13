from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
BROWN = 9

REPAIR_BY_HEIGHT = {
    1: RED,
    2: BLUE,
    3: GREEN,
    4: YELLOW,
    5: MAGENTA,
}

GEN_ATTEMPTS = 400
OBJECT_COUNT_RANGE = (3, 5)
HALF_WIDTH_RANGE = (8, 14)
GRID_HEIGHT_RANGE = (10, 24)
OBJECT_COLOR_POOL = [BLUE, RED, GREEN, YELLOW, GREY, MAGENTA, ORANGE, BROWN]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_mirror_col(g: np.ndarray) -> int:
    cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == AZURE)]
    assert len(cols) == 1, "expected one full-height azure mirror line"
    mirror_col = int(cols[0])
    assert 2 * mirror_col + 1 == g.shape[1], "mirror line must split the grid into equal halves"
    return mirror_col


def _extract_left_objects(g: np.ndarray, mirror_col: int) -> list[dict]:
    h = g.shape[0]
    visited = np.zeros((h, mirror_col), dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(mirror_col):
            color = int(g[r, c])
            if color in (BLACK, AZURE) or visited[r, c]:
                continue

            queue = deque([(r, c)])
            visited[r, c] = True
            coords: list[tuple[int, int]] = []

            while queue:
                rr, cc = queue.popleft()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if not (0 <= nr < h and 0 <= nc < mirror_col):
                        continue
                    if visited[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    queue.append((int(nr), int(nc)))

            arr = np.array(coords, dtype=int)
            rows = arr[:, 0]
            cols = arr[:, 1]
            objects.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (
                        int(rows.min()),
                        int(cols.min()),
                        int(rows.max()),
                        int(cols.max()),
                    ),
                }
            )

    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the missing pixels in each right-side mirror object using the height-coded repair color of its left partner.

    Input structure:
    - Background is black.
    - A full-height azure column at the center is the vertical mirror line.
    - The left half contains several 4-connected monochrome objects.
    - The right half is the horizontal reflection of those objects, except that each mirrored copy has 1 to 3 black missing pixels.

    Rule:
    - Find every complete object on the left half.
    - Reflect each occupied left-half pixel across the azure mirror line to locate the corresponding right-half pixel.
    - Measure the left object's projection height H, meaning the number of rows in its bounding box.
    - Whenever a reflected right-half pixel is black, fill it with the repair color for that height:
      H=1 -> red, H=2 -> blue, H=3 -> green, H=4 -> yellow, H=5 -> magenta.
    - Leave the mirror line, the left half, and every already-present right-half object pixel unchanged.

    Key insight:
    - The geometry tells you where the missing pixels are, but the color used to repair them comes from the left object's height rather than from the object's original color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    mirror_col = _find_mirror_col(g)
    out = g.copy()

    for obj in _extract_left_objects(g, mirror_col):
        r0, _, r1, _ = obj["bbox"]
        height = int(r1 - r0 + 1)
        assert height in REPAIR_BY_HEIGHT, "object height must be in the learned repair palette"
        repair_color = int(REPAIR_BY_HEIGHT[height])
        object_color = int(obj["color"])

        for r, c in obj["coords"]:
            mirrored_c = int(2 * mirror_col - c)
            cell = int(out[r, mirrored_c])
            assert cell in (BLACK, object_color), "right half must be a possibly-erased mirror of the left object"
            if cell == BLACK:
                out[r, mirrored_c] = repair_color

    return out


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[np.ix_(rows, cols)]


def _has_hole(mask: np.ndarray) -> bool:
    canvas = np.pad(mask.astype(bool), 1, constant_values=False)
    seen = np.zeros_like(canvas, dtype=bool)
    queue = deque([(0, 0)])
    seen[0, 0] = True

    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < canvas.shape[0] and 0 <= nc < canvas.shape[1]):
                continue
            if seen[nr, nc] or canvas[nr, nc]:
                continue
            seen[nr, nc] = True
            queue.append((int(nr), int(nc)))

    return bool(np.any((~canvas) & (~seen)))


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _make_shape(height: int, rng: random.Random) -> np.ndarray:
    if height == 1:
        width = rng.randint(4, 6)
        return np.ones((1, width), dtype=bool)

    for _ in range(80):
        area = rng.randint(max(4, height), 8)
        extra = area - height
        min_width = 1 if extra == 0 else 2
        max_width = min(4, extra + 1)
        width_choices = [w for w in range(min_width, max_width + 1) if height * w >= area]
        if not width_choices:
            continue
        width = rng.choice(width_choices)

        mask = np.zeros((height, width), dtype=bool)
        spine_col = rng.randrange(width)
        mask[:, spine_col] = True

        while int(mask.sum()) < area:
            frontier: set[tuple[int, int]] = set()
            for r, c in np.argwhere(mask):
                for nr, nc in _neighbors4(int(r), int(c)):
                    if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                        frontier.add((int(nr), int(nc)))
            if not frontier:
                break
            nr, nc = rng.choice(tuple(frontier))
            mask[nr, nc] = True

        if int(mask.sum()) != area:
            continue

        mask = _trim_mask(mask)
        if mask.shape[0] != height or _has_hole(mask):
            continue

        if mask.shape[0] > 1 and mask.shape[1] > 1 and bool(mask.all()) and rng.random() < 0.85:
            continue
        if rng.random() < 0.5:
            mask = np.fliplr(mask)
        return mask

    raise RuntimeError(f"failed to build a shape of height {height}")


def _assign_colors(heights: list[int], rng: random.Random) -> list[int]:
    forbidden = [REPAIR_BY_HEIGHT[h] for h in heights]
    for _ in range(200):
        colors = rng.sample(OBJECT_COLOR_POOL, len(heights))
        if all(colors[i] != forbidden[i] for i in range(len(heights))):
            return [int(c) for c in colors]
    raise RuntimeError("failed to assign distinct object colors")


def _place_objects(objects: list[dict], grid_h: int, half_w: int, rng: random.Random) -> bool:
    occupied = np.zeros((grid_h, half_w), dtype=bool)

    for obj in sorted(objects, key=lambda item: (-int(item["mask"].sum()), -item["mask"].shape[1])):
        mask = obj["mask"]
        h, w = mask.shape
        c0 = int(half_w - obj["distance"] - (w - 1))
        if c0 < 0 or c0 + w > half_w:
            return False

        rows = list(range(0, grid_h - h + 1))
        rng.shuffle(rows)
        dilated = _dilate8(occupied)

        for r0 in rows:
            if not bool((dilated[r0 : r0 + h, c0 : c0 + w] & mask).any()):
                obj["r0"] = int(r0)
                obj["c0"] = int(c0)
                occupied[r0 : r0 + h, c0 : c0 + w] |= mask
                break
        else:
            return False

    return True


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    rows, cols = np.nonzero(mask)
    grid[rows + r0, cols + c0] = int(color)


def _build_sample(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    n_objects = rng.randint(*OBJECT_COUNT_RANGE)
    heights = rng.sample([1, 2, 3, 4, 5], n_objects)
    distances = rng.sample([1, 2, 3, 4, 5], n_objects)
    colors = _assign_colors(heights, rng)

    objects = []
    for height, distance, color in zip(heights, distances, colors):
        mask = _make_shape(int(height), rng)
        objects.append(
            {
                "height": int(height),
                "distance": int(distance),
                "color": int(color),
                "mask": mask,
            }
        )

    required_half = max(obj["distance"] + obj["mask"].shape[1] - 1 for obj in objects)
    min_half = max(HALF_WIDTH_RANGE[0], int(required_half))
    if min_half > HALF_WIDTH_RANGE[1]:
        return None
    half_w = rng.randint(min_half, HALF_WIDTH_RANGE[1])

    min_grid_h = max(GRID_HEIGHT_RANGE[0], max(obj["mask"].shape[0] for obj in objects) + 2, n_objects + 6)
    if min_grid_h > GRID_HEIGHT_RANGE[1]:
        return None
    grid_h = rng.randint(min_grid_h, GRID_HEIGHT_RANGE[1])
    if not _place_objects(objects, grid_h, half_w, rng):
        return None

    grid_w = 2 * half_w + 1
    inp = np.full((grid_h, grid_w), BLACK, dtype=int)
    out = np.full((grid_h, grid_w), BLACK, dtype=int)
    inp[:, half_w] = AZURE
    out[:, half_w] = AZURE

    for obj in objects:
        mask = obj["mask"]
        r0 = int(obj["r0"])
        c0 = int(obj["c0"])
        color = int(obj["color"])
        repair_color = int(REPAIR_BY_HEIGHT[obj["height"]])

        _paint_mask(inp, mask, r0, c0, color)
        _paint_mask(out, mask, r0, c0, color)

        local_rows, local_cols = np.nonzero(mask)
        abs_rows = local_rows + r0
        abs_cols = local_cols + c0
        mirrored_cols = 2 * half_w - abs_cols
        area = len(abs_rows)
        missing_count = rng.randint(1, min(3, area - 1))
        missing_idx = set(rng.sample(range(area), missing_count))

        for idx in range(area):
            rr = int(abs_rows[idx])
            cc = int(mirrored_cols[idx])
            out[rr, cc] = color
            if idx in missing_idx:
                out[rr, cc] = repair_color
            else:
                inp[rr, cc] = color

    left_objects = _extract_left_objects(inp, half_w)
    if len(left_objects) != n_objects:
        return None
    if sorted(int(obj["bbox"][2] - obj["bbox"][0] + 1) for obj in left_objects) != sorted(heights):
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        built = _build_sample(rng)
        if built is not None:
            inp, _ = built
            return inp
    raise RuntimeError(f"generate_input: failed after {GEN_ATTEMPTS} attempts")
