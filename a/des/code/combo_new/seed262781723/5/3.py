from __future__ import annotations

import numpy as np


def _runs(values: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(values.tolist()):
        if value and start is None:
            start = i
        elif (not value) and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(values) - 1))
    return runs


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if (not mask[r, c]) or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.asarray(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _count_holes(mask: np.ndarray) -> int:
    h, w = mask.shape
    holes = 0
    for coords in _components(~mask):
        touches_edge = np.any(
            (coords[:, 0] == 0)
            | (coords[:, 0] == h - 1)
            | (coords[:, 1] == 0)
            | (coords[:, 1] == w - 1)
        )
        if not touches_edge:
            holes += 1
    return holes


def _is_connected(mask: np.ndarray) -> bool:
    return bool(mask.any()) and len(_components(mask)) == 1


def _find_separator_column(grid: np.ndarray) -> int:
    h, w = grid.shape
    candidates: list[int] = []
    for c in range(w):
        color = int(grid[0, c])
        if color == 0 or not np.all(grid[:, c] == color):
            continue
        if c == 0 or c == w - 1:
            continue
        if np.any(grid[:, :c] != 0) and np.any(grid[:, c + 1 :] != 0):
            candidates.append(c)
    assert len(candidates) == 1, "expected one full-height separator column"
    return candidates[0]


def _extract_templates(left: np.ndarray) -> list[np.ndarray]:
    templates: list[np.ndarray] = []
    for r0, r1 in _runs(np.any(left != 0, axis=1)):
        band = left[r0 : r1 + 1, :]
        cols = np.any(band != 0, axis=0)
        c0 = int(np.argmax(cols))
        c1 = int(len(cols) - 1 - np.argmax(cols[::-1]))
        templates.append(band[:, c0 : c1 + 1] != 0)
    assert templates, "template area must contain at least one stamp"
    return templates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each right-side object by the indexed stamp selected by its hole count.

    Input structure:
    - The background is black.
    - A full-height gray separator column splits the grid into a left legend area and a right object area.
    - The left area contains small monochrome stamp templates stacked top-to-bottom, with black rows separating consecutive templates.
    - The right area contains larger monochrome colored objects whose interiors may contain one or more fully enclosed black holes.

    Rule:
    - Read the left templates in top-to-bottom order so template 0 is the top stamp, template 1 is the next stamp, and so on.
    - For each 4-connected object in the right area, count the number of 4-connected black regions inside its bbox that do not touch that bbox boundary.
    - Select the template whose index equals that hole count.
    - Erase the original object and redraw the selected template at the object's top-left corner, using the object's own color instead of the template's original legend color.
    - Drop the entire left legend area and the separator; the output keeps only the right-side area.

    Key insight:
    - The left column of stamps is a lookup table indexed by enclosed-hole count, while output colors come from the right-side objects rather than from the legend itself.
    """
    g = np.asarray(grid, dtype=int)
    sep_col = _find_separator_column(g)
    left = g[:, :sep_col]
    right = g[:, sep_col + 1 :]
    templates = _extract_templates(left)

    out = np.zeros_like(right)
    for coords in _components(right != 0):
        values = right[coords[:, 0], coords[:, 1]]
        color = int(values[0])
        assert np.all(values == color), "right-side objects must be monochrome"
        r0, c0, r1, c1 = _bbox(coords)
        obj_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        obj_mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
        hole_count = _count_holes(obj_mask)
        assert 0 <= hole_count < len(templates), "missing template for hole count"
        stamp = templates[hole_count]
        sh, sw = stamp.shape
        out[r0 : r0 + sh, c0 : c0 + sw][stamp] = color
    return out


_TEMPLATE_LIBRARY = [
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1], [1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1]], dtype=bool),
]

_RNG = np.random.default_rng()
_MAX_ATTEMPTS = 400


def _random_partition(total: int, parts: int) -> list[int]:
    values = [0] * parts
    for _ in range(total):
        values[int(_RNG.integers(parts))] += 1
    return values


def _sample_templates(count: int) -> list[np.ndarray]:
    chosen: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for _ in range(200):
        if len(chosen) == count:
            break
        base = np.array(_TEMPLATE_LIBRARY[int(_RNG.integers(len(_TEMPLATE_LIBRARY)))], copy=True)
        mask = np.rot90(base, int(_RNG.integers(4)))
        if bool(_RNG.integers(2)):
            mask = np.fliplr(mask)
        mask = _normalize_mask(mask)
        if np.any(mask.sum(axis=1) == 0) or np.any(mask.sum(axis=0) == 0):
            continue
        sig = (tuple(map(int, mask.shape)), mask.tobytes())
        if sig in seen:
            continue
        chosen.append(mask)
        seen.add(sig)
    assert len(chosen) == count, "failed to sample enough distinct templates"
    return chosen


def _make_solid_object(min_h: int, min_w: int) -> np.ndarray:
    h = int(_RNG.integers(max(3, min_h), min(10, max(3, min_h) + 4)))
    w = int(_RNG.integers(max(3, min_w), min(10, max(3, min_w) + 4)))
    for _ in range(30):
        mask = np.ones((h, w), dtype=bool)
        notch_count = int(_RNG.integers(0, 3))
        for _ in range(notch_count):
            side = int(_RNG.integers(4))
            if side < 2 and w >= 4:
                notch_w = int(_RNG.integers(1, w))
                if notch_w >= w:
                    notch_w = w - 1
                start = int(_RNG.integers(0, w - notch_w + 1))
                depth = int(_RNG.integers(1, min(4, h)))
                if side == 0:
                    mask[:depth, start : start + notch_w] = False
                else:
                    mask[h - depth :, start : start + notch_w] = False
            elif side >= 2 and h >= 4:
                notch_h = int(_RNG.integers(1, h))
                if notch_h >= h:
                    notch_h = h - 1
                start = int(_RNG.integers(0, h - notch_h + 1))
                depth = int(_RNG.integers(1, min(4, w)))
                if side == 2:
                    mask[start : start + notch_h, :depth] = False
                else:
                    mask[start : start + notch_h, w - depth :] = False
        if _is_connected(mask) and _count_holes(mask) == 0:
            return mask
    return np.ones((max(3, min_h), max(3, min_w)), dtype=bool)


def _make_partitioned_ring(hole_count: int, min_h: int, min_w: int) -> np.ndarray:
    assert hole_count >= 1
    orientation = "vertical" if hole_count == 1 else ["vertical", "horizontal"][int(_RNG.integers(2))]
    if orientation == "vertical":
        min_h = max(min_h, 4)
        min_w = max(min_w, 2 * hole_count + 1)
        h = int(_RNG.integers(min_h, min(11, min_h + 4)))
        w = int(_RNG.integers(min_w, min(11, min_w + 4)))
        mask = np.zeros((h, w), dtype=bool)
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True
        inner_width = w - 2
        hole_widths = np.ones(hole_count, dtype=int)
        extra = inner_width - (hole_count + hole_count - 1)
        for _ in range(extra):
            hole_widths[int(_RNG.integers(hole_count))] += 1
        c = 1
        for i in range(hole_count):
            c += int(hole_widths[i])
            if i < hole_count - 1:
                mask[1:-1, c] = True
                c += 1
        return mask

    min_h = max(min_h, 2 * hole_count + 1)
    min_w = max(min_w, 4)
    h = int(_RNG.integers(min_h, min(11, min_h + 4)))
    w = int(_RNG.integers(min_w, min(11, min_w + 4)))
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    inner_height = h - 2
    hole_heights = np.ones(hole_count, dtype=int)
    extra = inner_height - (hole_count + hole_count - 1)
    for _ in range(extra):
        hole_heights[int(_RNG.integers(hole_count))] += 1
    r = 1
    for i in range(hole_count):
        r += int(hole_heights[i])
        if i < hole_count - 1:
            mask[r, 1:-1] = True
            r += 1
    return mask


def _make_object_mask(hole_count: int, stamp: np.ndarray) -> np.ndarray:
    need_h = int(stamp.shape[0]) + int(_RNG.integers(1, 4))
    need_w = int(stamp.shape[1]) + int(_RNG.integers(1, 4))
    if hole_count == 0:
        return _make_solid_object(need_h, need_w)
    return _make_partitioned_ring(hole_count, need_h, need_w)


def _pack_objects(objects: list[dict[str, object]], max_h: int, max_w: int) -> tuple[int, int, dict[int, tuple[int, int]]]:
    count = len(objects)
    base_margin = 1
    base_gap = 2
    for _ in range(300):
        order = list(map(int, _RNG.permutation(count)))
        if count <= 3 and bool(_RNG.integers(2)):
            rows = [order]
        else:
            split = int(_RNG.integers(1, count))
            rows = [order[:split], order[split:]]
            if not rows[1]:
                rows = [order]

        row_heights = [max(int(objects[i]["mask"].shape[0]) for i in row) for row in rows]
        row_width_mins = []
        for row in rows:
            width_sum = sum(int(objects[i]["mask"].shape[1]) for i in row)
            row_width_mins.append(2 * base_margin + width_sum + base_gap * (len(row) - 1))
        min_w = max(row_width_mins)
        min_h = 2 * base_margin + sum(row_heights) + base_gap * (len(rows) - 1)
        if min_w > max_w or min_h > max_h:
            continue

        right_w = int(_RNG.integers(min_w, min(max_w, min_w + 5) + 1))
        right_h = int(_RNG.integers(min_h, min(max_h, min_h + 5) + 1))

        v_slack = _random_partition(right_h - min_h, len(rows) + 1)
        y = base_margin + v_slack[0]
        positions: dict[int, tuple[int, int]] = {}
        ok = True
        for ridx, row in enumerate(rows):
            row_h = row_heights[ridx]
            row_w_min = row_width_mins[ridx]
            h_slack = _random_partition(right_w - row_w_min, len(row) + 1)
            x = base_margin + h_slack[0]
            for j, obj_idx in enumerate(row):
                mask = objects[obj_idx]["mask"]
                obj_h, obj_w = map(int, mask.shape)
                y0 = y + int(_RNG.integers(0, row_h - obj_h + 1))
                positions[obj_idx] = (y0, x)
                x += obj_w
                if j < len(row) - 1:
                    x += base_gap + h_slack[j + 1]
            if x > right_w:
                ok = False
                break
            y += row_h
            if ridx < len(rows) - 1:
                y += base_gap + v_slack[ridx + 1]
        if ok:
            return right_h, right_w, positions
    raise RuntimeError("failed to pack objects")


def _build_expected(height: int, right_w: int, objects: list[dict[str, object]], templates: list[np.ndarray]) -> np.ndarray:
    out = np.zeros((height, right_w), dtype=int)
    for obj in objects:
        y, x = obj["pos"]
        hole_count = int(obj["holes"])
        color = int(obj["color"])
        stamp = templates[hole_count]
        sh, sw = stamp.shape
        out[y : y + sh, x : x + sw][stamp] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    template_count = int(_RNG.integers(3, 5))
    templates = _sample_templates(template_count)

    separator_color = int(_RNG.integers(1, 10))
    remaining = [c for c in range(1, 10) if c != separator_color]
    template_colors = list(map(int, _RNG.choice(remaining, size=template_count, replace=False)))
    object_palette = [c for c in remaining if c not in template_colors]
    if len(object_palette) < 2:
        return None

    object_count = template_count + int(_RNG.integers(0, 2))
    hole_counts = list(range(template_count))
    while len(hole_counts) < object_count:
        hole_counts.append(int(_RNG.integers(template_count)))
    _RNG.shuffle(hole_counts)

    replace = object_count > len(object_palette)
    object_colors = list(map(int, _RNG.choice(object_palette, size=object_count, replace=replace)))
    if object_count > 1 and len(set(object_colors)) < 2:
        return None

    objects: list[dict[str, object]] = []
    for holes, color in zip(hole_counts, object_colors):
        mask = _make_object_mask(int(holes), templates[int(holes)])
        if _count_holes(mask) != int(holes):
            return None
        objects.append({"holes": int(holes), "color": int(color), "mask": mask})

    max_template_w = max(int(mask.shape[1]) for mask in templates)
    left_w = max_template_w + 2 + int(_RNG.integers(0, 2))
    if left_w >= 29:
        return None

    try:
        right_h, right_w, packed = _pack_objects(objects, max_h=30, max_w=30 - left_w - 1)
    except RuntimeError:
        return None

    legend_inner_h = sum(int(mask.shape[0]) for mask in templates) + (template_count - 1)
    min_total_h = max(right_h, legend_inner_h)
    if min_total_h > 30:
        return None
    total_h = int(_RNG.integers(min_total_h, min(30, min_total_h + 4) + 1))

    right_y_shift = int(_RNG.integers(0, total_h - right_h + 1))
    legend_top = int(_RNG.integers(0, total_h - legend_inner_h + 1))

    grid = np.zeros((total_h, left_w + 1 + right_w), dtype=int)
    sep_col = left_w
    grid[:, sep_col] = separator_color

    current_r = legend_top
    for color, mask in zip(template_colors, templates):
        th, tw = map(int, mask.shape)
        x0 = int(_RNG.integers(0, left_w - tw + 1))
        block = grid[current_r : current_r + th, x0 : x0 + tw]
        block[mask] = int(color)
        current_r += th + 1

    for idx, obj in enumerate(objects):
        y0, x0 = packed[idx]
        y0 += right_y_shift
        x0 += sep_col + 1
        mask = obj["mask"]
        oh, ow = map(int, mask.shape)
        block = grid[y0 : y0 + oh, x0 : x0 + ow]
        block[mask] = int(obj["color"])
        obj["pos"] = (y0, x0 - (sep_col + 1))

    expected = _build_expected(total_h, right_w, objects, templates)
    if np.array_equal(main(grid), expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
