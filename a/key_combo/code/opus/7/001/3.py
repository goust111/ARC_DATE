import itertools
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREY = 5

_NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    """Inclusive bbox in the shared (r, c) convention used by arc_utils."""
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    """Minimal connected-components helper matching arc_utils semantics."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 8:
        neighbors = _NEIGHBORS8
    else:
        neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1))

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in neighbors:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _is_lr_symmetric(mask: np.ndarray) -> bool:
    return bool(np.array_equal(mask, np.fliplr(mask)))


def _enumerate_shape_catalogs() -> tuple[list[np.ndarray], list[np.ndarray]]:
    dims = (
        (2, 2),
        (2, 3),
        (2, 4),
        (2, 5),
        (3, 2),
        (3, 3),
        (3, 4),
        (4, 2),
        (4, 3),
    )
    symmetric: list[np.ndarray] = []
    asymmetric: list[np.ndarray] = []

    for h, w in dims:
        cell_ids = list(range(h * w))
        for area in range(3, min(6, h * w) + 1):
            for choice in itertools.combinations(cell_ids, area):
                mask = np.zeros((h, w), dtype=bool)
                for idx in choice:
                    mask[idx // w, idx % w] = True
                if len(components_mask(mask, connectivity=8)) != 1:
                    continue
                if not mask[0].any() or not mask[-1].any():
                    continue
                if not mask[:, 0].any() or not mask[:, -1].any():
                    continue
                bucket = symmetric if _is_lr_symmetric(mask) else asymmetric
                bucket.append(mask)

    return symmetric, asymmetric


SYMMETRIC_MASKS, ASYMMETRIC_MASKS = _enumerate_shape_catalogs()


def _randomize_mask(mask: np.ndarray, keep_symmetric: bool) -> np.ndarray:
    out = mask.copy()
    if random.random() < 0.5:
        out = np.flipud(out)
    if not keep_symmetric and random.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _extract_objects(grid: np.ndarray) -> list[dict]:
    comps = components_mask(grid == BLUE, connectivity=8)
    markers = {tuple(map(int, pos)) for pos in np.argwhere(grid == RED)}
    objects = []
    for coords in comps:
        r0, c0, r1, c1 = bbox_coords(coords)
        marker = (r0, c0 - 1)
        if marker not in markers:
            raise AssertionError("missing marker aligned to object bbox")
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        rr = coords[:, 0] - r0
        cc = coords[:, 1] - c0
        mask[rr, cc] = True
        objects.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "marker": marker,
                "mask": mask,
                "symmetric": _is_lr_symmetric(mask),
            }
        )
    if len(markers) != len(objects):
        raise AssertionError("markers and objects must match one-to-one")
    objects.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
    return objects


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Draw rightward rays whose stripe pattern is determined by each blue object's left-right symmetry.

    Input structure:
    - A black background contains 3 to 5 small blue 8-connected objects.
    - Each object's tight bbox has a single red marker one column to its left, on the bbox's top row.
    - Exactly one blue object is not mirror-symmetric left-to-right inside its tight bbox.

    Rule:
    - For each blue object, crop its tight bbox mask and test whether that mask equals its left-right reflection.
    - Starting from the aligned red marker, extend a horizontal ray to the right edge.
    - If the object is symmetric, fill background cells on that ray with alternating red and grey by column offset from the marker; if it is asymmetric, fill every background cell red.
    - Never overwrite blue object pixels; skip them and continue with the same offset sequence.

    Key insight:
    - The ray is not chosen by symmetry; symmetry becomes a label that controls whether the emitted ray is striped or solid.
    """
    grid = np.asarray(input_grid, dtype=int)
    out = grid.copy()
    h, w = out.shape

    for obj in _extract_objects(grid):
        r, c_start = obj["marker"]
        symmetric = obj["symmetric"]
        for c in range(c_start, w):
            if c == c_start:
                out[r, c] = RED
                continue
            if out[r, c] != BG:
                continue
            offset = c - c_start
            if symmetric:
                out[r, c] = RED if offset % 2 == 0 else GREY
            else:
                out[r, c] = RED
    return out


def _allocate_rows(heights: list[int], height: int) -> list[int]:
    gaps = len(heights) + 1
    base = [1] + [3] * (len(heights) - 1) + [1]
    extra = height - (sum(heights) + sum(base))
    for _ in range(extra):
        base[random.randrange(gaps)] += 1

    starts = []
    r = base[0]
    for idx, obj_h in enumerate(heights):
        starts.append(r)
        r += obj_h
        if idx + 1 < len(heights):
            r += base[idx + 1]
    return starts


def _render_specs(height: int, width: int, specs: list[dict]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for spec in specs:
        mask = spec["mask"]
        r0 = spec["r0"]
        c0 = spec["c0"]
        rr, cc = np.argwhere(mask).T
        grid[r0 + rr, c0 + cc] = BLUE
        grid[r0, c0 - 1] = RED
    return grid


def _expected_output_from_specs(height: int, width: int, specs: list[dict]) -> np.ndarray:
    out = _render_specs(height, width, specs)
    for spec in specs:
        r = spec["r0"]
        c_start = spec["c0"] - 1
        for c in range(c_start, width):
            if c == c_start:
                out[r, c] = RED
                continue
            if out[r, c] != BG:
                continue
            offset = c - c_start
            if spec["symmetric"]:
                out[r, c] = RED if offset % 2 == 0 else GREY
            else:
                out[r, c] = RED
    return out


def _sample_specs() -> tuple[int, int, list[dict]]:
    count = random.randint(3, 5)
    asym_index = random.randrange(count)
    specs = []

    for idx in range(count):
        if idx == asym_index:
            mask = random.choice(ASYMMETRIC_MASKS)
            mask = _randomize_mask(mask, keep_symmetric=False)
            symmetric = False
        else:
            mask = random.choice(SYMMETRIC_MASKS)
            mask = _randomize_mask(mask, keep_symmetric=True)
            symmetric = True
        specs.append({"mask": mask, "symmetric": symmetric})

    heights = [int(spec["mask"].shape[0]) for spec in specs]
    max_width = max(int(spec["mask"].shape[1]) for spec in specs)
    min_height = sum(heights) + 3 * (count - 1) + 2
    if min_height > 30 or max_width + 8 > 30:
        raise RuntimeError("sampled objects do not fit the ARC size bound")
    height = random.randint(max(14, min_height), min(30, min_height + 6))
    width = random.randint(max(15, max_width + 8), min(30, max_width + 16))

    row_starts = _allocate_rows(heights, height)
    candidates = list(range(2, max(3, width - 5)))
    random.shuffle(candidates)

    for spec, r0 in zip(specs, row_starts):
        obj_w = int(spec["mask"].shape[1])
        valid_cols = [c for c in candidates if c + obj_w <= width - 4]
        if not valid_cols:
            raise RuntimeError("no valid column placement")
        spec["r0"] = int(r0)
        spec["c0"] = int(random.choice(valid_cols))

    random.shuffle(specs)
    return height, width, specs


def _validate_specs(specs: list[dict], grid: np.ndarray, expected: np.ndarray) -> None:
    objects = _extract_objects(grid)
    if not (3 <= len(objects) <= 5):
        raise AssertionError("unexpected object count")

    asym_count = sum(not obj["symmetric"] for obj in objects)
    if asym_count != 1:
        raise AssertionError("expected exactly one asymmetric object")

    bboxes = [obj["bbox"] for obj in objects]
    for i in range(len(bboxes) - 1):
        gap = bboxes[i + 1][0] - bboxes[i][2] - 1
        if gap < 3:
            raise AssertionError("row gap too small")

    for obj in objects:
        r0, c0, _r1, _c1 = obj["bbox"]
        if grid[r0, c0 - 1] != RED:
            raise AssertionError("marker not aligned to bbox top-left exterior")
        if not (3 <= int(obj["mask"].sum()) <= 6):
            raise AssertionError("invalid object area")

    if not np.array_equal(main(grid), expected):
        raise AssertionError("solver disagrees with oracle")

    striped_rows = 0
    solid_rows = 0
    for spec in specs:
        row = expected[spec["r0"]]
        tail = row[spec["c0"] - 1 :]
        if spec["symmetric"]:
            striped_rows += int(GREY in tail)
        else:
            solid_rows += int(np.all(np.isin(tail[tail != BLUE], [RED])))
    if striped_rows == 0 or solid_rows == 0:
        raise AssertionError("missing visible branch evidence")


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            height, width, specs = _sample_specs()
        except RuntimeError:
            continue
        grid = _render_specs(height, width, specs)
        expected = _expected_output_from_specs(height, width, specs)
        try:
            _validate_specs(specs, grid, expected)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
