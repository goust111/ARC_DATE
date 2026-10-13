from __future__ import annotations

import numpy as np


GREY = 5
TRANSFORM_MARKERS = {
    8: "copy",    # azure
    9: "hflip",   # maroon
    7: "vflip",   # orange
    4: "rot180",  # yellow
}
MARKER_FOR_TRANSFORM = {v: k for k, v in TRANSFORM_MARKERS.items()}


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        stack = [(rr, cc)]
        seen[rr, cc] = True
        pts: list[tuple[int, int]] = []
        while stack:
            cr, cur_c = stack.pop()
            pts.append((cr, cur_c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = cr + dr
                nc = cur_c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _apply_transform(mask: np.ndarray, transform: str) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    if transform == "copy":
        return arr.copy()
    if transform == "hflip":
        return np.fliplr(arr)
    if transform == "vflip":
        return np.flipud(arr)
    if transform == "rot180":
        return np.flipud(np.fliplr(arr))
    raise ValueError(f"unknown transform: {transform}")


def _parse_legends(grid: np.ndarray, bg: int) -> dict[int, str]:
    h, w = grid.shape
    legends: dict[int, str] = {}
    for row in (0, h - 1):
        c = 0
        while c + 1 < w:
            left = int(grid[row, c])
            right = int(grid[row, c + 1])
            if left != bg and right in TRANSFORM_MARKERS:
                left_ok = c == 0 or int(grid[row, c - 1]) == bg
                right_ok = c + 2 >= w or int(grid[row, c + 2]) == bg
                vertical_ok = True
                other_row = 1 if row == 0 else h - 2
                if 0 <= other_row < h:
                    vertical_ok = int(grid[other_row, c]) == bg and int(grid[other_row, c + 1]) == bg
                if left_ok and right_ok and vertical_ok:
                    legends[left] = TRANSFORM_MARKERS[right]
                    c += 2
                    continue
            c += 1
    return legends


def _find_portal_lines(grid: np.ndarray, portal_color: int) -> list[dict[str, int]]:
    lines: list[dict[str, int]] = []
    for comp in _components(grid == portal_color):
        if len(comp) < 2:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if r0 == r1:
            cols = sorted(c for _, c in comp)
            if cols == list(range(c0, c1 + 1)):
                lines.append(
                    {
                        "orientation": 0,  # horizontal
                        "fixed": r0,
                        "start": c0,
                        "end": c1,
                        "length": c1 - c0 + 1,
                    }
                )
        elif c0 == c1:
            rows = sorted(r for r, _ in comp)
            if rows == list(range(r0, r1 + 1)):
                lines.append(
                    {
                        "orientation": 1,  # vertical
                        "fixed": c0,
                        "start": r0,
                        "end": r1,
                        "length": r1 - r0 + 1,
                    }
                )
    lines.sort(key=lambda item: (-item["length"], item["orientation"], item["fixed"], item["start"]))
    if len(lines) < 2:
        return []
    dominant_orientation = lines[0]["orientation"]
    filtered = [line for line in lines if line["orientation"] == dominant_orientation]
    filtered.sort(key=lambda item: (item["fixed"], item["start"]))
    return filtered[:2]


def _paint_transferred_pattern(
    out: np.ndarray,
    patch: np.ndarray,
    target_line: dict[str, int],
    target_side: str,
    line_offset: int,
    gap: int,
) -> None:
    h, w = patch.shape
    if target_line["orientation"] == 1:
        r0 = target_line["start"] + line_offset
        if target_side == "right":
            c0 = target_line["fixed"] + gap
        else:
            c1 = target_line["fixed"] - gap
            c0 = c1 - w + 1
    else:
        c0 = target_line["start"] + line_offset
        if target_side == "down":
            r0 = target_line["fixed"] + gap
        else:
            r1 = target_line["fixed"] - gap
            r0 = r1 - h + 1
    for rr in range(h):
        for cc in range(w):
            if patch[rr, cc]:
                out[r0 + rr, c0 + cc] = GREY


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find matched portal lines, read the edge legend, and copy the grey pattern
    from the occupied outer side to the empty outer side with the required
    geometric transform.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    legends = _parse_legends(g, bg)
    out = g.copy()
    grey_points = [(int(r), int(c)) for r, c in np.argwhere(g == GREY)]

    for portal_color, transform in legends.items():
        lines = _find_portal_lines(g, portal_color)
        if len(lines) != 2:
            continue

        first, second = lines
        if first["orientation"] != second["orientation"]:
            continue

        if first["orientation"] == 1:
            left, right = sorted(lines, key=lambda item: item["fixed"])
            side_limit = right["fixed"] - left["fixed"] + 2
            left_cells = [
                (r, c)
                for r, c in grey_points
                if left["start"] <= r <= left["end"] and 0 < left["fixed"] - c <= side_limit
            ]
            right_cells = [
                (r, c)
                for r, c in grey_points
                if right["start"] <= r <= right["end"] and 0 < c - right["fixed"] <= side_limit
            ]
            if bool(left_cells) == bool(right_cells):
                continue
            if left_cells:
                source_line, target_line = left, right
                source_side, target_side = "left", "right"
                source_cells = left_cells
            else:
                source_line, target_line = right, left
                source_side, target_side = "right", "left"
                source_cells = right_cells
            r0, c0, r1, c1 = _bbox(source_cells)
            mask = g[r0 : r1 + 1, c0 : c1 + 1] == GREY
            transformed = _apply_transform(mask, transform)
            line_offset = r0 - source_line["start"]
            gap = source_line["fixed"] - c1 if source_side == "left" else c0 - source_line["fixed"]
            _paint_transferred_pattern(out, transformed, target_line, target_side, line_offset, gap)
        else:
            top, bottom = sorted(lines, key=lambda item: item["fixed"])
            side_limit = bottom["fixed"] - top["fixed"] + 2
            top_cells = [
                (r, c)
                for r, c in grey_points
                if top["start"] <= c <= top["end"] and 0 < top["fixed"] - r <= side_limit
            ]
            bottom_cells = [
                (r, c)
                for r, c in grey_points
                if bottom["start"] <= c <= bottom["end"] and 0 < r - bottom["fixed"] <= side_limit
            ]
            if bool(top_cells) == bool(bottom_cells):
                continue
            if top_cells:
                source_line, target_line = top, bottom
                source_side, target_side = "up", "down"
                source_cells = top_cells
            else:
                source_line, target_line = bottom, top
                source_side, target_side = "down", "up"
                source_cells = bottom_cells
            r0, c0, r1, c1 = _bbox(source_cells)
            mask = g[r0 : r1 + 1, c0 : c1 + 1] == GREY
            transformed = _apply_transform(mask, transform)
            line_offset = c0 - source_line["start"]
            gap = source_line["fixed"] - r1 if source_side == "up" else r0 - source_line["fixed"]
            _paint_transferred_pattern(out, transformed, target_line, target_side, line_offset, gap)

    return out


def _random_mask(rng: np.random.Generator, h: int, w: int, transform: str) -> np.ndarray:
    max_components = int(rng.integers(1, 4))
    for _ in range(400):
        mask = rng.random((h, w)) < rng.uniform(0.35, 0.72)
        if int(mask.sum()) < max(3, min(h, w)):
            continue
        if bool(mask.all()):
            continue
        if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if len(_components(mask)) > max_components:
            continue
        if transform != "copy" and np.array_equal(mask, _apply_transform(mask, transform)):
            continue
        return mask

    mask = np.zeros((h, w), dtype=bool)
    mask[:, 0] = True
    mask[-1, :] = True
    mask[0, -1] = True
    if h > 2 and w > 2:
        mask[1, -1] = True
    if transform != "copy" and np.array_equal(mask, _apply_transform(mask, transform)):
        mask[0, min(1, w - 1)] = True
    return mask


def _make_vertical_module(
    rng: np.random.Generator, bg: int, portal_color: int, transform: str
) -> dict[str, object]:
    line_len = int(rng.integers(5, 9))
    shape_h = int(rng.integers(2, min(5, line_len) + 1))
    shape_w = int(rng.integers(2, 5))
    gap = int(rng.integers(1, 3))
    corridor = int(rng.integers(2, 4))
    width = 2 * gap + 2 * shape_w + corridor
    left_col = gap + shape_w - 1
    right_col = left_col + corridor + 1
    grid = np.full((line_len, width), bg, dtype=int)
    grid[:, left_col] = portal_color
    grid[:, right_col] = portal_color

    row_offset = int(rng.integers(0, line_len - shape_h + 1))
    source_left = bool(rng.integers(0, 2))
    mask = _random_mask(rng, shape_h, shape_w, transform)
    if source_left:
        c0 = left_col - gap - shape_w + 1
    else:
        c0 = right_col + gap
    patch = grid[row_offset : row_offset + shape_h, c0 : c0 + shape_w]
    patch[mask] = GREY
    return {"grid": grid, "portal_color": portal_color, "transform": transform}


def _make_horizontal_module(
    rng: np.random.Generator, bg: int, portal_color: int, transform: str
) -> dict[str, object]:
    line_len = int(rng.integers(5, 9))
    shape_h = int(rng.integers(2, 5))
    shape_w = int(rng.integers(2, min(5, line_len) + 1))
    gap = int(rng.integers(1, 3))
    corridor = int(rng.integers(2, 4))
    height = 2 * gap + 2 * shape_h + corridor
    top_row = gap + shape_h - 1
    bottom_row = top_row + corridor + 1
    grid = np.full((height, line_len), bg, dtype=int)
    grid[top_row, :] = portal_color
    grid[bottom_row, :] = portal_color

    col_offset = int(rng.integers(0, line_len - shape_w + 1))
    source_up = bool(rng.integers(0, 2))
    mask = _random_mask(rng, shape_h, shape_w, transform)
    if source_up:
        r0 = top_row - gap - shape_h + 1
    else:
        r0 = bottom_row + gap
    patch = grid[r0 : r0 + shape_h, col_offset : col_offset + shape_w]
    patch[mask] = GREY
    return {"grid": grid, "portal_color": portal_color, "transform": transform}


def _paste(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> bool:
    h, w = patch.shape
    sub = canvas[r0 : r0 + h, c0 : c0 + w]
    mask = patch != bg
    if np.any(sub[mask] != bg):
        return False
    sub[mask] = patch[mask]
    return True


def _legend_starts(rng: np.random.Generator, width: int, count: int) -> list[int] | None:
    if width < 3 * count - 1:
        return None
    for _ in range(200):
        starts = sorted(int(v) for v in rng.choice(width - 1, size=count, replace=False))
        if all(starts[i + 1] - starts[i] >= 3 for i in range(count - 1)):
            return starts
    return None


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    bg = int(rng.choice([0, 1, 2]))
    portal_colors = [color for color in (0, 1, 2, 3, 6) if color != bg]
    transforms = list(MARKER_FOR_TRANSFORM)
    rng.shuffle(portal_colors)
    rng.shuffle(transforms)

    modules = []
    for portal_color, transform in zip(portal_colors, transforms):
        if bool(rng.integers(0, 2)):
            module = _make_vertical_module(rng, bg, int(portal_color), str(transform))
        else:
            module = _make_horizontal_module(rng, bg, int(portal_color), str(transform))
        modules.append(module)

    rng.shuffle(modules)
    slot00, slot01, slot10, slot11 = modules
    h1 = max(slot00["grid"].shape[0], slot01["grid"].shape[0])
    h2 = max(slot10["grid"].shape[0], slot11["grid"].shape[0])
    w1 = max(slot00["grid"].shape[1], slot10["grid"].shape[1])
    w2 = max(slot01["grid"].shape[1], slot11["grid"].shape[1])

    row_gap = int(rng.integers(6, 9))
    col_gap = int(rng.integers(6, 9))
    left_margin = int(rng.integers(1, 4))
    right_margin = int(rng.integers(1, 4))
    top_or_bottom_margin = int(rng.integers(1, 4))
    legend_gap = int(rng.integers(2, 4))
    legend_top = bool(rng.integers(0, 2))

    width = left_margin + w1 + col_gap + w2 + right_margin
    if legend_top:
        height = 1 + legend_gap + h1 + row_gap + h2 + top_or_bottom_margin
        top_base = 1 + legend_gap
        legend_row = 0
    else:
        height = top_or_bottom_margin + h1 + row_gap + h2 + legend_gap + 1
        top_base = top_or_bottom_margin
        legend_row = height - 1

    if height > 40 or width > 40:
        return None

    grid = np.full((height, width), bg, dtype=int)
    left_x = left_margin
    right_x = left_margin + w1 + col_gap
    lower_y = top_base + h1 + row_gap

    slots = [
        (slot00, top_base, left_x, h1, w1),
        (slot01, top_base, right_x, h1, w2),
        (slot10, lower_y, left_x, h2, w1),
        (slot11, lower_y, right_x, h2, w2),
    ]

    for module, slot_r, slot_c, slot_h, slot_w in slots:
        patch = module["grid"]
        dr = int(rng.integers(0, slot_h - patch.shape[0] + 1))
        dc = int(rng.integers(0, slot_w - patch.shape[1] + 1))
        if not _paste(grid, patch, slot_r + dr, slot_c + dc, bg):
            return None

    starts = _legend_starts(rng, width, len(modules))
    if starts is None:
        return None
    legend_order = [modules[i] for i in rng.permutation(len(modules))]
    for start, module in zip(starts, legend_order):
        portal_color = int(module["portal_color"])
        marker = MARKER_FOR_TRANSFORM[str(module["transform"])]
        grid[legend_row, start] = portal_color
        grid[legend_row, start + 1] = marker

    if _get_bg(grid) != bg:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    fixed_mask = (grid != bg) & (grid != GREY)
    if not np.array_equal(out[fixed_mask], grid[fixed_mask]):
        return None
    if int(np.sum(out == GREY)) != 2 * int(np.sum(grid == GREY)):
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid portal puzzle sample")
