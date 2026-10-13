from __future__ import annotations

import random
from collections import Counter

import numpy as np


NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
PALETTE = tuple(range(10))
GEN_ATTEMPTS = 500
PLACEMENT_ATTEMPTS = 200


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _components_by_color(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or visited[r, c]:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(g[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            pts = np.array(coords, dtype=int)
            r0 = int(pts[:, 0].min())
            c0 = int(pts[:, 1].min())
            r1 = int(pts[:, 0].max())
            c1 = int(pts[:, 1].max())
            components.append(
                {
                    "color": color,
                    "coords": pts,
                    "bbox": (r0, c0, r1, c1),
                }
            )

    return components


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress each foreground object along its short axis and keep the result in place.

    Wide objects (bbox width > height) become a single vertical column at the bbox
    center column. Tall objects (bbox height > width) become a single horizontal row
    at the bbox center row. Rows/columns that contain no object pixels remain bg.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _background_color(g)
    components = _components_by_color(g, bg)

    out = g.copy()
    for comp in components:
        coords = np.asarray(comp["coords"], dtype=int)
        color = int(comp["color"])
        r0, c0, r1, c1 = map(int, comp["bbox"])
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        assert height != width, "each object bbox must be non-square"

        out[coords[:, 0], coords[:, 1]] = bg

        if width > height:
            target_c = c0 + width // 2
            row_counts = Counter(int(r) for r in coords[:, 0])
            for r in range(r0, r1 + 1):
                if row_counts.get(r, 0) > 0:
                    out[r, target_c] = color
        else:
            target_r = r0 + height // 2
            col_counts = Counter(int(c) for c in coords[:, 1])
            for c in range(c0, c1 + 1):
                if col_counts.get(c, 0) > 0:
                    out[target_r, c] = color

    return out


def _monotone_ribbon(height: int, width: int) -> np.ndarray:
    """Create a 4-connected wide ribbon whose bbox uses the full height/width."""
    assert width > height >= 2
    direction = random.choice((-1, 1))
    row = 0 if direction > 0 else height - 1
    change_points = set(random.sample(range(1, width), height - 1))

    path_rows = []
    for c in range(width):
        path_rows.append(row)
        if c + 1 in change_points:
            row += direction

    mask = np.zeros((height, width), dtype=bool)
    mask[path_rows[0], 0] = True
    for c in range(1, width):
        prev_row = path_rows[c - 1]
        cur_row = path_rows[c]
        lo = min(prev_row, cur_row)
        hi = max(prev_row, cur_row)
        mask[lo : hi + 1, c] = True

    for c in range(width):
        rows = np.flatnonzero(mask[:, c])
        lo = int(rows.min())
        hi = int(rows.max())
        if lo > 0 and random.random() < 0.45:
            lo -= 1
        if hi < height - 1 and random.random() < 0.45:
            hi += 1
        mask[lo : hi + 1, c] = True

    return mask


def _wide_mask(height: int, width: int) -> np.ndarray:
    assert width > height >= 2
    modes = ["ribbon", "ribbon", "ribbon", "solid"]
    if height >= 3 and width >= 4:
        modes.append("frame")
    mode = random.choice(modes)

    if mode == "solid":
        return np.ones((height, width), dtype=bool)

    if mode == "frame":
        mask = np.zeros((height, width), dtype=bool)
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True
        return mask

    return _monotone_ribbon(height, width)


def _object_mask(height: int, width: int, orientation: str) -> np.ndarray:
    if orientation == "wide":
        return _wide_mask(height, width)
    if orientation == "tall":
        return _wide_mask(width, height).T
    raise ValueError(f"unknown orientation: {orientation}")


def _odd_choices(lo: int, hi: int) -> list[int]:
    return [value for value in range(lo, hi + 1) if value % 2 == 1]


def _sample_dims(orientation: str) -> tuple[int, int]:
    short = random.randint(2, 5)
    long_choices = _odd_choices(short + 1, min(11, short + 6))
    long_side = random.choice(long_choices)
    if orientation == "wide":
        return short, long_side
    return long_side, short


def _bbox_is_clear(
    placed: list[tuple[int, int, int, int]],
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    margin: int = 1,
) -> bool:
    for or0, oc0, or1, oc1 in placed:
        if not (
            r1 < or0 - margin
            or r0 > or1 + margin
            or c1 < oc0 - margin
            or c0 > oc1 + margin
        ):
            return False
    return True


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    rows, cols = np.nonzero(mask)
    grid[r0 + rows, c0 + cols] = color


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.randint(2, 5)
    n_wide = random.randint(1, n_objects - 1)
    orientations = ["wide"] * n_wide + ["tall"] * (n_objects - n_wide)
    random.shuffle(orientations)

    bg = random.choice(PALETTE)
    colors = [color for color in PALETTE if color != bg]
    random.shuffle(colors)
    obj_colors = colors[:n_objects]

    specs = []
    total_bbox_area = 0
    for orientation, color in zip(orientations, obj_colors):
        height, width = _sample_dims(orientation)
        mask = _object_mask(height, width, orientation)
        total_bbox_area += height * width
        specs.append(
            {
                "orientation": orientation,
                "color": color,
                "height": height,
                "width": width,
                "mask": mask,
            }
        )

    min_side = max(max(int(spec["height"]) for spec in specs), max(int(spec["width"]) for spec in specs)) + 4
    area_side = int(np.ceil(np.sqrt(total_bbox_area * 2.6)))
    H = random.randint(max(12, min_side, area_side - 2), min(24, max(14, area_side + 4, min_side + 4)))
    W = random.randint(max(12, min_side, area_side - 2), min(24, max(14, area_side + 4, min_side + 4)))
    if H < min_side or W < min_side:
        return None

    placed_bboxes: list[tuple[int, int, int, int]] = []
    placements: list[tuple[int, int]] = []
    for spec in specs:
        height = int(spec["height"])
        width = int(spec["width"])
        found = None
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = random.randint(0, H - height)
            c0 = random.randint(0, W - width)
            r1 = r0 + height - 1
            c1 = c0 + width - 1
            if _bbox_is_clear(placed_bboxes, r0, c0, r1, c1, margin=1):
                found = (r0, c0)
                placed_bboxes.append((r0, c0, r1, c1))
                placements.append((r0, c0))
                break
        if found is None:
            return None

    grid = np.full((H, W), bg, dtype=int)
    for spec, (r0, c0) in zip(specs, placements):
        _paint_mask(grid, np.asarray(spec["mask"], dtype=bool), r0, c0, int(spec["color"]))

    if _background_color(grid) != bg:
        return None
    if int(np.count_nonzero(grid != bg)) * 2 >= int(grid.size):
        return None

    components = _components_by_color(grid, bg)
    if len(components) != n_objects:
        return None

    seen_orientations = {"wide": 0, "tall": 0}
    seen_colors = set()
    for comp in components:
        color = int(comp["color"])
        if color in seen_colors:
            return None
        seen_colors.add(color)

        r0, c0, r1, c1 = map(int, comp["bbox"])
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        if height == width:
            return None
        seen_orientations["wide" if width > height else "tall"] += 1

    if seen_orientations["wide"] == 0 or seen_orientations["tall"] == 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
