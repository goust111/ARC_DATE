from __future__ import annotations

import random

import numpy as np


_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_N8 = _N4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]

_BG = 0
_COLORS = tuple(range(1, 10))
_MAX_RETRIES = 400


def _background_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(flags.tolist()):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 1))
    return runs


def _bbox(points: np.ndarray) -> tuple[int, int, int, int]:
    rows = points[:, 0]
    cols = points[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _crop_true(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(pts)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    deltas = _N8 if connectivity == 8 else _N4
    parts: list[np.ndarray] = []

    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if visited[rr, cc]:
            continue
        stack = [(rr, cc)]
        visited[rr, cc] = True
        pts: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            pts.append((cr, cc))
            for dr, dc in deltas:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    stack.append((nr, nc))
        parts.append(np.array(pts, dtype=int))
    return parts


def _component_count(mask: np.ndarray, connectivity: int) -> int:
    return len(_components(mask, connectivity))


def _hole_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    holes = 0
    for pts in _components(~mask, connectivity=4):
        if np.any(
            (pts[:, 0] == 0)
            | (pts[:, 0] == h - 1)
            | (pts[:, 1] == 0)
            | (pts[:, 1] == w - 1)
        ):
            continue
        holes += 1
    return holes


def _sorted_parts(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    parts = _components(mask, connectivity)
    return sorted(parts, key=lambda pts: (_bbox(pts)[1], _bbox(pts)[0]))


def _extract_objects(band: np.ndarray, bg: int) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    for pts in _sorted_parts(band != bg, connectivity=8):
        r0, c0, r1, c1 = _bbox(pts)
        colors = {int(band[r, c]) for r, c in pts.tolist()}
        assert len(colors) == 1, "each top object must be monochrome"
        color = next(iter(colors))
        mask = band[r0 : r1 + 1, c0 : c1 + 1] == color
        objects.append({"bbox": (r0, c0, r1, c1), "mask": mask, "color": color})
    return objects


def _extract_legend_blocks(band: np.ndarray, bg: int) -> list[dict[str, object]]:
    blocks: list[dict[str, object]] = []
    for pts in _sorted_parts(band != bg, connectivity=8):
        r0, c0, r1, c1 = _bbox(pts)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h == w and h in (1, 2), "legend blocks must be 1x1 or 2x2 squares"
        colors = {int(band[r, c]) for r, c in pts.tolist()}
        assert len(colors) == 1, "legend block must be monochrome"
        assert len(pts) == h * w, "legend block must be solid"
        blocks.append({"bbox": (r0, c0, r1, c1), "color": next(iter(colors))})
    return blocks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate each top object according to its hole count and recolor it from the aligned legend block.

    Input structure:
    - The most common color is the background.
    - The upper band contains a left-to-right row of separated monochrome 8-connected objects.
    - The lower band contains the same number of separated 1x1 or 2x2 monochrome square legend blocks, aligned with the objects above them.

    Rule:
    - Split the non-background content into the upper object band and the lower legend band.
    - Sort upper objects and lower legend blocks from left to right, then pair them by order.
    - For each paired object, count enclosed background holes inside the object's bbox using 4-connected background regions.
    - If an object has H holes, output H horizontal copies of its bbox crop, separated by 1 background column and recolored to the paired legend-block color.
    - Omit objects with H=0, and separate different surviving object groups by 2 background columns.

    Key insight:
    - The object's topology supplies the copy count, while the lower aligned color block supplies the output color, so both cues must be combined.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)

    row_runs = _runs(np.any(g != bg, axis=1))
    assert len(row_runs) == 2, "expected one object band and one legend band"
    (top_r0, top_r1), (bot_r0, bot_r1) = row_runs

    objects = _extract_objects(g[top_r0 : top_r1 + 1, :], bg)
    blocks = _extract_legend_blocks(g[bot_r0 : bot_r1 + 1, :], bg)
    assert len(objects) == len(blocks) and objects, "object and legend counts must match"

    groups: list[np.ndarray] = []
    max_h = 0
    for obj, block in zip(objects, blocks):
        mask = np.asarray(obj["mask"], dtype=bool)
        holes = _hole_count(mask)
        if holes == 0:
            continue
        color = int(block["color"])
        sprite = np.where(mask, color, bg)
        h, w = sprite.shape
        group = np.full((h, holes * w + (holes - 1)), bg, dtype=int)
        for idx in range(holes):
            start = idx * (w + 1)
            group[:, start : start + w] = sprite
        groups.append(group)
        max_h = max(max_h, h)

    if not groups:
        return np.full((1, 1), bg, dtype=int)

    out_w = sum(part.shape[1] for part in groups) + 2 * (len(groups) - 1)
    out = np.full((max_h, out_w), bg, dtype=int)
    cursor = 0
    for idx, part in enumerate(groups):
        h, w = part.shape
        out[:h, cursor : cursor + w] = part
        cursor += w
        if idx + 1 < len(groups):
            cursor += 2
    return out


def _sample_hole_counts(count: int) -> list[int]:
    counts = [random.randint(1, 3) for _ in range(count)]
    if random.random() < 0.5:
        zero_count = 1 if count == 2 else random.randint(1, 2)
        for idx in random.sample(range(count), zero_count):
            counts[idx] = 0
    if all(value == 0 for value in counts):
        counts[random.randrange(count)] = random.randint(1, 3)
    return counts


def _decorate_exterior(mask: np.ndarray) -> np.ndarray:
    canvas = np.pad(mask, 1, constant_values=False)
    for _ in range(random.randint(0, 3)):
        h, w = canvas.shape
        side = random.choice(["top", "bottom", "left", "right"])
        if side in ("top", "bottom"):
            border_row = 1 if side == "top" else h - 2
            outer_row = 0 if side == "top" else h - 1
            cols = [c for c in range(1, w - 1) if canvas[border_row, c]]
            if cols:
                canvas[outer_row, random.choice(cols)] = True
        else:
            border_col = 1 if side == "left" else w - 2
            outer_col = 0 if side == "left" else w - 1
            rows = [r for r in range(1, h - 1) if canvas[r, border_col]]
            if rows:
                canvas[random.choice(rows), outer_col] = True
    return _crop_true(canvas)


def _make_solid_mask() -> np.ndarray | None:
    height = random.randint(2, 5)
    width = random.randint(2, 6)
    mask = np.ones((height, width), dtype=bool)

    corners = [(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)]
    random.shuffle(corners)
    for r, c in corners[: random.randint(0, min(2, len(corners)))]:
        trial = mask.copy()
        trial[r, c] = False
        if trial.any() and _component_count(trial, 8) == 1 and _hole_count(trial) == 0:
            mask = trial

    mask = _decorate_exterior(mask)
    if _component_count(mask, 8) == 1 and _hole_count(mask) == 0:
        return mask
    return None


def _make_holey_mask(hole_count: int) -> np.ndarray | None:
    orientation = random.choice(["horizontal", "vertical"])

    if orientation == "horizontal":
        hole_heights = [random.randint(1, 2) for _ in range(hole_count)]
        height = max(hole_heights) + 2 + random.randint(0, 1)
        width = 2 * hole_count + 1
        mask = np.ones((height, width), dtype=bool)
        for idx, hole_h in enumerate(hole_heights):
            col = 1 + 2 * idx
            row = random.randint(1, height - hole_h - 1)
            mask[row : row + hole_h, col] = False
    else:
        hole_widths = [random.randint(1, 2) for _ in range(hole_count)]
        height = 2 * hole_count + 1
        width = max(hole_widths) + 2 + random.randint(0, 1)
        mask = np.ones((height, width), dtype=bool)
        for idx, hole_w in enumerate(hole_widths):
            row = 1 + 2 * idx
            col = random.randint(1, width - hole_w - 1)
            mask[row, col : col + hole_w] = False

    mask = _decorate_exterior(mask)
    if _component_count(mask, 8) == 1 and _hole_count(mask) == hole_count:
        return mask
    return None


def _make_object_mask(hole_count: int) -> np.ndarray | None:
    for _ in range(50):
        mask = _make_solid_mask() if hole_count == 0 else _make_holey_mask(hole_count)
        if mask is None:
            continue
        h, w = mask.shape
        if h <= 10 and w <= 10:
            return mask
    return None


def _place_mask(grid: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> None:
    h, w = mask.shape
    sub = grid[top : top + h, left : left + w]
    sub[mask] = color


def _candidate_grid() -> np.ndarray | None:
    count = random.randint(2, 4)
    hole_counts = _sample_hole_counts(count)
    masks = []
    for hole_count in hole_counts:
        mask = _make_object_mask(hole_count)
        if mask is None:
            return None
        masks.append(mask)

    object_widths = [mask.shape[1] for mask in masks]
    object_heights = [mask.shape[0] for mask in masks]
    obj_gaps = [random.randint(2, 4) for _ in range(count - 1)]
    top_span = sum(object_widths) + sum(obj_gaps)

    positive_widths = [
        holes * mask.shape[1] + (holes - 1)
        for holes, mask in zip(hole_counts, masks)
        if holes > 0
    ]
    if not positive_widths:
        return None
    out_width = sum(positive_widths) + 2 * (len(positive_widths) - 1)
    if top_span > 28 or out_width > 30:
        return None

    colors = random.sample(_COLORS, 2 * count)
    object_colors = colors[:count]
    legend_colors = colors[count:]
    legend_sizes = [random.choice([1, 2]) for _ in range(count)]

    top_height = max(object_heights)
    legend_height = max(legend_sizes)
    row_gap = random.randint(2, 5)
    top_margin = random.randint(0, 3)
    bottom_margin = random.randint(0, 3)
    grid_height = top_margin + top_height + row_gap + legend_height + bottom_margin
    if grid_height > 30:
        return None

    min_width = max(top_span, out_width)
    width_slack = max(0, 30 - min_width)
    grid_width = min_width + random.randint(0, min(6, width_slack))
    left_margin = random.randint(0, grid_width - top_span)

    grid = np.full((grid_height, grid_width), _BG, dtype=int)

    x_positions: list[int] = []
    cursor = left_margin
    for idx, mask in enumerate(masks):
        x_positions.append(cursor)
        _place_mask(grid, mask, object_colors[idx], top_margin, cursor)
        cursor += mask.shape[1]
        if idx + 1 < count:
            cursor += obj_gaps[idx]

    legend_top = top_margin + top_height + row_gap
    for idx, size in enumerate(legend_sizes):
        obj_left = x_positions[idx]
        obj_width = masks[idx].shape[1]
        block_left = obj_left + random.randint(0, obj_width - size)
        grid[legend_top : legend_top + size, block_left : block_left + size] = legend_colors[idx]

    try:
        out = main(grid)
    except Exception:
        return None

    if out.ndim != 2 or max(out.shape) > 30:
        return None
    if _background_color(grid) != _BG:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _candidate_grid()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
