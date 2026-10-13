from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask, crop_bbox


BG = int(Color.BLACK)
SEP = int(Color.GREY)
OBJECT_COLORS = (int(Color.BLUE), int(Color.RED), int(Color.GREEN))
GEN_RETRIES = 400

_BASE_SHAPES: dict[int, tuple[np.ndarray, ...]] = {
    0: (
        np.array([[1, 1, 0], [0, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 1, 1], [1, 0, 1], [1, 0, 0]], dtype=bool),
        np.array([[1, 0, 0], [1, 1, 1], [0, 1, 1]], dtype=bool),
    ),
    1: (
        np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
        np.array([[1, 1, 1, 0], [1, 0, 1, 1], [1, 1, 1, 0]], dtype=bool),
        np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 0]], dtype=bool),
    ),
    2: (
        np.array([[0, 1, 1, 1, 0], [1, 0, 1, 0, 1], [1, 1, 1, 1, 1]], dtype=bool),
    ),
}


def _hole_count(mask: np.ndarray) -> int:
    """Count 4-connected enclosed background components inside a sprite bbox."""
    h, w = mask.shape
    bg_mask = ~mask
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r0 in range(h):
        for c0 in range(w):
            if not bg_mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            touches_border = False
            while stack:
                r, c = stack.pop()
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches_border = True
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < h and 0 <= cc < w and bg_mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            if not touches_border:
                holes += 1
    return holes


def _separator_col(grid: np.ndarray) -> int:
    grey_cols = np.flatnonzero(np.all(grid == SEP, axis=0))
    assert grey_cols.size == 1, "expected exactly one full-height grey separator column"
    return int(grey_cols[0])


def _object_patch(grid: np.ndarray, coords: np.ndarray, color: int) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(coords)
    return crop_bbox(grid == color, (r0, c0, r1, c1))


def _extract_left_objects(grid: np.ndarray, sep: int) -> list[dict]:
    left = grid[:, :sep]
    colors = sorted(set(map(int, np.unique(left))) - {BG, SEP})
    objects: list[dict] = []
    for color in colors:
        comps = components_mask(left == color, connectivity=4)
        assert len(comps) == 1, "each sprite color must form exactly one 4-connected object"
        coords = comps[0]
        bbox = bbox_coords(coords)
        assert bbox[3] <= sep - 2, "sprites must stay at least one cell away from the separator"
        patch = _object_patch(left, coords, color).astype(bool)
        holes = _hole_count(patch)
        objects.append(
            {
                "color": int(color),
                "coords": coords,
                "bbox": bbox,
                "patch": patch,
                "holes": int(holes),
            }
        )
    assert 2 <= len(objects) <= 3, "expected two or three left-hand sprites"
    return sorted(objects, key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))


def _paint_order(objects: list[dict]) -> list[dict]:
    return sorted(
        objects,
        key=lambda obj: (obj["holes"], -obj["bbox"][0], -obj["bbox"][1]),
    )


def _paint_order_indices(objects: list[dict]) -> list[int]:
    return sorted(
        range(len(objects)),
        key=lambda idx: (objects[idx]["holes"], -objects[idx]["bbox"][0], -objects[idx]["bbox"][1]),
    )


def _render_right(objects: list[dict], shape: tuple[int, int], sep: int) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros(shape, dtype=int)
    owner = np.full(shape, -1, dtype=int)
    by_priority = _paint_order(objects)
    for idx, obj in enumerate(by_priority):
        patch = np.fliplr(obj["patch"])
        rows, cols = np.nonzero(patch)
        rr = rows + int(obj["bbox"][0])
        cc = cols + sep + 1
        out[rr, cc] = int(obj["color"])
        owner[rr, cc] = idx
    return out, owner


def _render_right_with_order(objects: list[dict], shape: tuple[int, int], sep: int, order: list[int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for idx in order:
        obj = objects[idx]
        patch = np.fliplr(obj["patch"])
        rows, cols = np.nonzero(patch)
        out[rows + int(obj["bbox"][0]), cols + sep + 1] = int(obj["color"])
    return out


def _overlap_count(a: np.ndarray, top_a: int, b: np.ndarray, top_b: int) -> int:
    fa = np.fliplr(a)
    fb = np.fliplr(b)
    r0 = max(int(top_a), int(top_b))
    r1 = min(int(top_a) + fa.shape[0], int(top_b) + fb.shape[0])
    if r0 >= r1:
        return 0
    overlap_w = min(fa.shape[1], fb.shape[1])
    aa = fa[r0 - int(top_a) : r1 - int(top_a), :overlap_w]
    bb = fb[r0 - int(top_b) : r1 - int(top_b), :overlap_w]
    return int(np.count_nonzero(aa & bb))


def _transformed_shape(hole_count: int, rng: np.random.Generator) -> np.ndarray:
    base = _BASE_SHAPES[hole_count][int(rng.integers(0, len(_BASE_SHAPES[hole_count])))]
    patch = base.copy()
    patch = np.rot90(patch, int(rng.integers(0, 4)))
    if bool(rng.integers(0, 2)):
        patch = np.fliplr(patch)
    if bool(rng.integers(0, 2)):
        patch = np.flipud(patch)
    return patch.astype(bool)


def _place_left_objects(
    shape_specs: list[dict], height: int, sep: int, rng: np.random.Generator
) -> list[tuple[int, int]] | None:
    order = sorted(range(len(shape_specs)), key=lambda i: (-shape_specs[i]["shape"].shape[1], -shape_specs[i]["shape"].shape[0]))
    blocked = np.zeros((height, sep), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(shape_specs)

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        sprite = shape_specs[idx]["shape"]
        top = int(shape_specs[idx]["top"])
        h, w = sprite.shape
        if top < 0 or top + h > height or w > sep - 1:
            return False
        cols = list(range(0, sep - w))
        rng.shuffle(cols)
        rows, cols_local = np.nonzero(sprite)
        for left in cols:
            rr = rows + top
            cc = cols_local + left
            if np.any(blocked[rr, cc]):
                continue
            snapshot = blocked.copy()
            for r, c in zip(rr, cc):
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr = int(r + dr)
                        nc = int(c + dc)
                        if 0 <= nr < height and 0 <= nc < sep:
                            blocked[nr, nc] = True
            placements[idx] = (top, int(left))
            if dfs(pos + 1):
                return True
            blocked[:, :] = snapshot
            placements[idx] = None
        return False

    if not dfs(0):
        return None
    return [placement for placement in placements if placement is not None]


def _semantic_check(grid: np.ndarray) -> bool:
    try:
        sep = _separator_col(grid)
        if not np.all(grid[:, sep + 1 :] == BG):
            return False
        objects = _extract_left_objects(grid, sep)
        holes = [obj["holes"] for obj in objects]
        if len(set(holes)) != len(holes):
            return False
        right, _ = _render_right(objects, grid.shape, sep)
        output = main(grid)
        if not np.array_equal(output[:, sep], np.zeros(grid.shape[0], dtype=int)):
            return False
        if not np.array_equal(output[:, sep + 1 :], right[:, sep + 1 :]):
            return False
        overlap_pixels = 0
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                overlap_pixels += _overlap_count(
                    objects[i]["patch"],
                    objects[i]["bbox"][0],
                    objects[j]["patch"],
                    objects[j]["bbox"][0],
                )
        if overlap_pixels <= 0:
            return False
        order = _paint_order_indices(objects)
        reverse = _render_right_with_order(
            objects,
            grid.shape,
            sep,
            list(reversed(order)),
        )
        if np.array_equal(right, reverse):
            return False
        for obj in objects:
            visible = int(np.count_nonzero((right == obj["color"]) & (output == obj["color"])))
            if visible <= 0:
                return False
        total_stamp_pixels = sum(int(obj["patch"].sum()) for obj in objects)
        if overlap_pixels * 3 >= total_stamp_pixels * 2:
            return False
        return True
    except AssertionError:
        return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each left sprite inside its own bbox across the separator and stamp the mirrored copies into the shared right mirror zone, with hole-rich sprites on top.

    Input structure:
    - Black is the background and one full-height grey column is the separator.
    - Two or three single-color 4-connected sprites lie on the left of the separator and the right side is empty.
    - Each sprite has a distinct number of enclosed holes when counted inside its own bounding box.

    Rule:
    - For each left sprite, count its enclosed holes.
    - Crop the sprite to its own bbox, flip that crop left-right, and stamp it on the right starting immediately after the separator while keeping the same top row.
    - If mirrored stamps overlap, the sprite with more holes covers the one with fewer holes; if hole counts tie, the higher original sprite wins.
    - Erase the grey separator to black and keep the left-side sprites unchanged.

    Key insight:
    - The separator creates one shared mirror-stamping lane on the right, but topology decides the visible z-order wherever mirrored copies collide.
    """
    g = np.asarray(grid, dtype=int)
    sep = _separator_col(g)
    objects = _extract_left_objects(g, sep)

    out = g.copy()
    out[:, sep] = BG
    right, _owner = _render_right(objects, g.shape, sep)
    mask = right != BG
    out[mask] = right[mask]
    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        n_objects = int(rng.choice([2, 3]))
        hole_counts = list(rng.choice(np.array([0, 1, 2]), size=n_objects, replace=False))
        rng.shuffle(hole_counts)

        shape_specs: list[dict] = []
        max_w = 0
        for holes in hole_counts:
            shape = _transformed_shape(int(holes), rng)
            shape_specs.append({"holes": int(holes), "shape": shape})
            max_w = max(max_w, int(shape.shape[1]))

        height = int(rng.integers(max(11, max(shape["shape"].shape[0] for shape in shape_specs) + 4), 17))
        width = int(rng.integers(max(18, max_w + 12), 25))
        sep_low = max(max_w + 3, width // 2 - 1)
        sep_high = min(width - max_w - 2, width // 2 + 2)
        if sep_low > sep_high:
            continue
        sep = int(rng.integers(sep_low, sep_high + 1))

        pair = tuple(rng.choice(np.arange(n_objects), size=2, replace=False))
        first, second = int(pair[0]), int(pair[1])
        shape_a = shape_specs[first]["shape"]
        shape_b = shape_specs[second]["shape"]
        top_a = int(rng.integers(0, height - shape_a.shape[0] + 1))
        candidate_tops = []
        for top_b in range(0, height - shape_b.shape[0] + 1):
            overlap = _overlap_count(shape_a, top_a, shape_b, top_b)
            if 1 <= overlap <= 5:
                candidate_tops.append(top_b)
        if not candidate_tops:
            continue
        shape_specs[first]["top"] = top_a
        shape_specs[second]["top"] = int(candidate_tops[int(rng.integers(0, len(candidate_tops)))])

        for idx, spec in enumerate(shape_specs):
            if idx in pair:
                continue
            spec["top"] = int(rng.integers(0, height - spec["shape"].shape[0] + 1))

        placements = _place_left_objects(shape_specs, height, sep, rng)
        if placements is None:
            continue

        grid = np.zeros((height, width), dtype=int)
        grid[:, sep] = SEP
        colors = list(rng.permutation(np.array(OBJECT_COLORS))[:n_objects])
        for idx, ((top, left), spec, color) in enumerate(zip(placements, shape_specs, colors)):
            patch = spec["shape"]
            rows, cols = np.nonzero(patch)
            grid[rows + int(top), cols + int(left)] = int(color)

        if _semantic_check(grid):
            return grid

    raise RuntimeError("failed to generate a valid separator mirror-overlay puzzle")
