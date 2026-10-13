from __future__ import annotations

import random

import numpy as np


BG = 8
COLOR_POOL = (0, 1, 2, 3, 4, 5, 6, 7, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Generator configuration.
GRID_HEIGHT_RANGE = (18, 28)
GRID_WIDTH_RANGE = (18, 28)
OBJECT_COUNT_RANGE = (2, 5)
BBOX_HEIGHT_RANGE = (4, 8)
BBOX_WIDTH_RANGE = (4, 8)
EDGE_MARGIN = 1
OBJECT_GAP = 1
PLACEMENT_ATTEMPTS = 80
GENERATION_ATTEMPTS = 400
SHAPE_STYLES = ("frame", "cross", "h_bridge", "v_bridge")


def _components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == int(bg) or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            objects.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )

    objects.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
    return objects


def _scatter_indices(box_area: int, count: int) -> list[int]:
    assert 0 < count <= box_area
    step = max(1, box_area // count)
    indices = [i * step for i in range(count)]
    assert indices[-1] < box_area
    return indices


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dissolve each foreground object into same-color single-pixel dots spread across its bbox.

    The prompt describes rectangular blocks, which makes the transform degenerate for
    fully filled rectangles. This implementation applies the stated bbox-and-count rule
    to each 4-connected monochrome foreground object, and it still behaves correctly for
    solid rectangles as the identity case.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = BG if np.any(g == BG) else int(np.bincount(g.ravel()).argmax())
    out = g.copy()

    for obj in _components(g, bg):
        color = int(obj["color"])
        r0, c0, r1, c1 = obj["bbox"]
        count = int(obj["area"])
        box_h = int(r1 - r0 + 1)
        box_w = int(c1 - c0 + 1)
        box_area = int(box_h * box_w)
        assert count <= box_area

        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
        for idx in _scatter_indices(box_area, count):
            rr = int(r0 + idx // box_w)
            cc = int(c0 + idx % box_w)
            out[rr, cc] = int(color)

    return out


def _shape_mask(h: int, w: int, style: str) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)

    if style == "frame":
        mask[0, :] = True
        mask[h - 1, :] = True
        mask[:, 0] = True
        mask[:, w - 1] = True
    elif style == "cross":
        rr = random.randrange(h)
        cc = random.randrange(w)
        mask[rr, :] = True
        mask[:, cc] = True
    elif style == "h_bridge":
        rr = random.randrange(h)
        mask[:, 0] = True
        mask[:, w - 1] = True
        mask[rr, :] = True
    elif style == "v_bridge":
        cc = random.randrange(w)
        mask[0, :] = True
        mask[h - 1, :] = True
        mask[:, cc] = True
    else:
        raise AssertionError(f"unknown style: {style}")

    assert mask[0, :].any()
    assert mask[h - 1, :].any()
    assert mask[:, 0].any()
    assert mask[:, w - 1].any()
    assert 0 < int(mask.sum()) < h * w
    return mask


def _spaced_apart(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _sample_object_specs(count: int) -> list[tuple[int, int, str, int]]:
    colors = random.sample(COLOR_POOL, count)
    specs: list[tuple[int, int, str, int]] = []
    for color in colors:
        h = random.randint(*BBOX_HEIGHT_RANGE)
        w = random.randint(*BBOX_WIDTH_RANGE)
        style = random.choice(SHAPE_STYLES)
        specs.append((h, w, style, color))
    random.shuffle(specs)
    return specs


def _place_objects(
    height: int,
    width: int,
    specs: list[tuple[int, int, str, int]],
) -> np.ndarray | None:
    grid = np.full((height, width), BG, dtype=int)
    placed_bboxes: list[tuple[int, int, int, int]] = []

    for obj_h, obj_w, style, color in specs:
        if height < obj_h + 2 * EDGE_MARGIN or width < obj_w + 2 * EDGE_MARGIN:
            return None

        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = random.randint(EDGE_MARGIN, height - obj_h - EDGE_MARGIN)
            c0 = random.randint(EDGE_MARGIN, width - obj_w - EDGE_MARGIN)
            bbox = (r0, c0, r0 + obj_h - 1, c0 + obj_w - 1)
            if not all(_spaced_apart(bbox, other, OBJECT_GAP) for other in placed_bboxes):
                continue

            mask = _shape_mask(obj_h, obj_w, style)
            patch = np.full((obj_h, obj_w), BG, dtype=int)
            patch[mask] = int(color)
            grid[r0 : r0 + obj_h, c0 : c0 + obj_w] = patch
            placed_bboxes.append(bbox)
            placed = True
            break

        if not placed:
            return None

    return grid


def _valid_generated_grid(grid: np.ndarray) -> bool:
    if grid is None:
        return False
    if not np.any(grid == BG):
        return False

    objects = _components(grid, BG)
    if not (OBJECT_COUNT_RANGE[0] <= len(objects) <= OBJECT_COUNT_RANGE[1]):
        return False
    if len({int(obj["color"]) for obj in objects}) != len(objects):
        return False

    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        box_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if int(obj["area"]) >= box_area:
            return False

    try:
        out = main(grid)
    except Exception:
        return False

    if np.array_equal(out, grid):
        return False
    return True


def generate_input() -> np.ndarray:
    """Generate non-degenerate foreground objects whose bboxes strictly exceed their pixel counts."""
    for _ in range(GENERATION_ATTEMPTS):
        height = random.randint(*GRID_HEIGHT_RANGE)
        width = random.randint(*GRID_WIDTH_RANGE)
        count = random.randint(*OBJECT_COUNT_RANGE)
        specs = _sample_object_specs(count)
        grid = _place_objects(height, width, specs)
        if _valid_generated_grid(grid):
            return grid
    raise RuntimeError("failed to generate a valid input grid")
