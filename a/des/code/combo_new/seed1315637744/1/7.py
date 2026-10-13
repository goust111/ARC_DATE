import random

import numpy as np


BG = 0
FENCE = 9
COLORS = (1, 2, 3, 4, 5, 6, 7, 8)
GEN_ATTEMPTS = 400
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRECTION_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIRS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _opening_direction(grid: np.ndarray, color: int, bbox: tuple[int, int, int, int]) -> str | None:
    r0, c0, r1, c1 = bbox
    top_missing = [c for c in range(c0, c1 + 1) if int(grid[r0, c]) != color]
    bottom_missing = [c for c in range(c0, c1 + 1) if int(grid[r1, c]) != color]
    left_missing = [r for r in range(r0, r1 + 1) if int(grid[r, c0]) != color]
    right_missing = [r for r in range(r0, r1 + 1) if int(grid[r, c1]) != color]

    def is_gap(xs: list[int]) -> bool:
        return len(xs) == 2 and xs[1] == xs[0] + 1

    sides = {
        "up": top_missing,
        "down": bottom_missing,
        "left": left_missing,
        "right": right_missing,
    }
    good = [name for name, missing in sides.items() if is_gap(missing)]
    if len(good) != 1:
        return None
    for name, missing in sides.items():
        if name != good[0] and missing:
            return None
    return good[0]


def _fence_components(grid: np.ndarray, bg: int) -> list[dict]:
    fences: list[dict] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for coords in _components(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 4 or w < 4:
                continue
            expected = 2 * (h + w) - 6
            if int(coords.shape[0]) != expected:
                continue
            direction = _opening_direction(grid, color, (r0, c0, r1, c1))
            if direction is None:
                continue
            fences.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "direction": direction,
                }
            )
    return fences


def _inside_objects(grid: np.ndarray, fence: dict, bg: int) -> list[dict]:
    r0, c0, r1, c1 = fence["bbox"]
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    mask = (interior != bg) & (interior != int(fence["color"]))
    objects: list[dict] = []
    for coords in _components(mask):
        global_coords = coords + np.array([r0 + 1, c0 + 1], dtype=int)
        rr, cc = map(int, global_coords[0])
        objects.append(
            {
                "color": int(grid[rr, cc]),
                "coords": global_coords,
            }
        )
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Release every animal object from its fence in the opening direction by a distance equal to the number of animals inside.

    Input structure:
    - The background is black.
    - One or more thin rectangular fences appear as hollow 1-pixel frames with one 2-pixel gap on exactly one side.
    - Each fence encloses 1 to 4 small colored 4-connected monochrome objects, and the outside background contains no other objects.

    Rule:
    - Detect each fence, its opening side, and the number N of enclosed objects.
    - Remove the enclosed objects from their original interior positions.
    - Translate every enclosed object by N cells in the opening direction, preserving each object's shape and color.
    - Keep the fence unchanged, including the opening itself.

    Key insight:
    - The gap tells you only the direction; the movement distance comes from counting the fence's companions inside rather than from object size or gap size.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = g.shape
    bg = _get_bg(g)
    fences = _fence_components(g, bg)
    fence_mask = np.zeros_like(g, dtype=bool)
    for fence in fences:
        r0, c0, r1, c1 = fence["bbox"]
        fence_mask[r0 : r1 + 1, c0 : c1 + 1] |= g[r0 : r1 + 1, c0 : c1 + 1] == int(fence["color"])

    shifted: list[tuple[int, np.ndarray]] = []
    for fence in fences:
        objects = _inside_objects(g, fence, bg)
        if not objects:
            continue
        dr, dc = DIRECTION_VECTORS[fence["direction"]]
        delta = np.array([dr * len(objects), dc * len(objects)], dtype=int)
        for obj in objects:
            for r, c in obj["coords"]:
                out[int(r), int(c)] = bg
            shifted.append((int(obj["color"]), obj["coords"] + delta))

    for color, coords in shifted:
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            if 0 <= rr < h and 0 <= cc < w and not fence_mask[rr, cc]:
                out[rr, cc] = color
    return out


def _normalize_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in shape)
    min_c = min(c for _, c in shape)
    return tuple(sorted((r - min_r, c - min_c) for r, c in shape))


def _rotate_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    height = max(r for r, _ in shape) + 1
    rotated = tuple((c, height - 1 - r) for r, c in shape)
    return _normalize_shape(rotated)


def _build_shapes() -> list[tuple[tuple[int, int], ...]]:
    bases = [
        ((0, 0), (0, 1)),
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 1), (1, 0), (1, 1), (1, 2)),
        ((0, 1), (0, 2), (1, 0), (1, 1)),
    ]
    seen: set[tuple[tuple[int, int], ...]] = set()
    shapes: list[tuple[tuple[int, int], ...]] = []
    for base in bases:
        shape = _normalize_shape(base)
        for _ in range(4):
            if shape not in seen:
                seen.add(shape)
                shapes.append(shape)
            shape = _rotate_shape(shape)
    return shapes


SHAPES = _build_shapes()
DOMINO_H = _normalize_shape(((0, 0), (0, 1)))
DOMINO_V = _normalize_shape(((0, 0), (1, 0)))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    height = max(r for r, _ in shape) + 1
    width = max(c for _, c in shape) + 1
    return height, width


def _split_extra(rng: random.Random, total: int, parts: int) -> list[int]:
    out = [0] * parts
    for _ in range(total):
        out[rng.randrange(parts)] += 1
    return out


def _count_choices(rng: random.Random, k: int) -> list[int]:
    if rng.random() < 0.7:
        values = [1, 2, 3, 4]
        rng.shuffle(values)
        return values[:k]
    counts = [rng.randint(1, 4) for _ in range(k)]
    if len(set(counts)) == 1:
        counts[rng.randrange(k)] = (counts[0] % 4) + 1
    return counts


def _direction_choices(rng: random.Random, k: int) -> list[str]:
    directions = ["up", "down", "left", "right"]
    if rng.random() < 0.5:
        rng.shuffle(directions)
        return directions[:k]
    return [rng.choice(directions) for _ in range(k)]


def _pick_shapes(
    rng: random.Random, direction: str, count: int
) -> list[tuple[tuple[int, int], ...]]:
    if count == 1:
        return [DOMINO_H if direction in ("up", "down") else DOMINO_V]
    band = count - 1
    if direction in ("up", "down"):
        allowed = [shape for shape in SHAPES if _shape_dims(shape)[0] <= band]
    else:
        allowed = [shape for shape in SHAPES if _shape_dims(shape)[1] <= band]
    return [rng.choice(allowed) for _ in range(count)]


def _make_fence_spec(rng: random.Random, direction: str, count: int) -> dict:
    shapes = _pick_shapes(rng, direction, count)
    colors = rng.sample(COLORS, count)
    band = 1 if count == 1 else count - 1

    if direction in ("up", "down"):
        spans = [_shape_dims(shape)[1] for shape in shapes]
        inner_w_min = sum(spans) + count + 1
        inner_w = inner_w_min + rng.randint(0, 3)
        inner_h = band + rng.randint(2, 5)
    else:
        spans = [_shape_dims(shape)[0] for shape in shapes]
        inner_h_min = sum(spans) + count + 1
        inner_h = inner_h_min + rng.randint(0, 3)
        inner_w = band + rng.randint(2, 5)

    patch = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
    patch[0, :] = FENCE
    patch[-1, :] = FENCE
    patch[:, 0] = FENCE
    patch[:, -1] = FENCE

    if direction in ("up", "down"):
        widths = [_shape_dims(shape)[1] for shape in shapes]
        extra = inner_w - (sum(widths) + count + 1)
        gaps = _split_extra(rng, extra, count + 1)
        col = 1 + gaps[0]
        for idx, (shape, color, width) in enumerate(zip(shapes, colors, widths)):
            height = _shape_dims(shape)[0]
            if direction == "up":
                row = rng.randint(0, band - height) if band > height else 0
            else:
                row_base = inner_h - band
                row = row_base + (rng.randint(0, band - height) if band > height else 0)
            for dr, dc in shape:
                patch[row + 1 + dr, col + dc] = color
            col += width + 1 + gaps[idx + 1]

        object_mask = (patch != BG) & (patch != FENCE)
        cols = np.where(object_mask)[1]
        if count == 1:
            gap_start = int(cols.min())
        else:
            low = max(1, int(cols.min()) - 1)
            high = min(inner_w - 1, int(cols.max()))
            if low > high:
                low = high = max(1, min(inner_w - 1, int(cols.min())))
            gap_start = rng.randint(low, high)
        gap_start = max(1, min(gap_start, patch.shape[1] - 3))
        gap_row = 0 if direction == "up" else patch.shape[0] - 1
        patch[gap_row, gap_start : gap_start + 2] = BG
    else:
        heights = [_shape_dims(shape)[0] for shape in shapes]
        extra = inner_h - (sum(heights) + count + 1)
        gaps = _split_extra(rng, extra, count + 1)
        row = 1 + gaps[0]
        for idx, (shape, color, height) in enumerate(zip(shapes, colors, heights)):
            width = _shape_dims(shape)[1]
            if direction == "left":
                col = rng.randint(0, band - width) if band > width else 0
            else:
                col_base = inner_w - band
                col = col_base + (rng.randint(0, band - width) if band > width else 0)
            for dr, dc in shape:
                patch[row + dr, col + 1 + dc] = color
            row += height + 1 + gaps[idx + 1]

        object_mask = (patch != BG) & (patch != FENCE)
        rows = np.where(object_mask)[0]
        if count == 1:
            gap_start = int(rows.min())
        else:
            low = max(1, int(rows.min()) - 1)
            high = min(inner_h - 1, int(rows.max()))
            if low > high:
                low = high = max(1, min(inner_h - 1, int(rows.min())))
            gap_start = rng.randint(low, high)
        gap_start = max(1, min(gap_start, patch.shape[0] - 3))
        gap_col = 0 if direction == "left" else patch.shape[1] - 1
        patch[gap_start : gap_start + 2, gap_col] = BG

    extra_top = count - 1 if direction == "up" and count > 1 else 0
    extra_bottom = count - 1 if direction == "down" and count > 1 else 0
    extra_left = count - 1 if direction == "left" and count > 1 else 0
    extra_right = count - 1 if direction == "right" and count > 1 else 0

    return {
        "patch": patch,
        "count": count,
        "direction": direction,
        "extra_top": extra_top,
        "extra_bottom": extra_bottom,
        "extra_left": extra_left,
        "extra_right": extra_right,
        "fp_h": patch.shape[0] + extra_top + extra_bottom,
        "fp_w": patch.shape[1] + extra_left + extra_right,
    }


def _pack_specs(rng: random.Random, specs: list[dict]) -> np.ndarray | None:
    for _ in range(40):
        row_count = rng.randint(1, min(3, len(specs)))
        row_items: list[list[tuple[dict, int]]] = [[] for _ in range(row_count)]
        row_widths = [0] * row_count
        row_heights = [0] * row_count

        order = specs[:]
        rng.shuffle(order)
        order.sort(key=lambda spec: max(spec["fp_h"], spec["fp_w"]), reverse=True)
        for spec in order:
            choices = list(range(row_count))
            rng.shuffle(choices)
            row_idx = min(choices, key=lambda idx: (row_widths[idx], row_heights[idx]))
            gap = rng.randint(1, 3) if row_items[row_idx] else 0
            row_items[row_idx].append((spec, gap))
            row_widths[row_idx] += gap + int(spec["fp_w"])
            row_heights[row_idx] = max(row_heights[row_idx], int(spec["fp_h"]))

        rows = [row for row in row_items if row]
        if not rows:
            continue
        row_widths = [sum(gap + int(spec["fp_w"]) for spec, gap in row) for row in rows]
        row_heights = [max(int(spec["fp_h"]) for spec, _ in row) for row in rows]
        row_gaps = [rng.randint(1, 3) for _ in range(max(0, len(rows) - 1))]

        top = rng.randint(1, 3)
        bottom = rng.randint(1, 3)
        left = rng.randint(1, 3)
        right = rng.randint(1, 3)
        height = top + bottom + sum(row_heights) + sum(row_gaps)
        width = left + right + max(row_widths)
        if height > 30 or width > 30:
            continue

        grid = np.zeros((height, width), dtype=int)
        y = top
        ok = True
        for row_idx, row in enumerate(rows):
            row_height = row_heights[row_idx]
            x = left
            for spec, gap in row:
                x += gap
                vertical_slack = row_height - int(spec["fp_h"])
                fp_top = y + (rng.randint(0, vertical_slack) if vertical_slack else 0)
                fp_left = x
                patch = spec["patch"]
                patch_top = fp_top + int(spec["extra_top"])
                patch_left = fp_left + int(spec["extra_left"])
                region = grid[patch_top : patch_top + patch.shape[0], patch_left : patch_left + patch.shape[1]]
                if np.any((patch != BG) & (region != BG)):
                    ok = False
                    break
                region[patch != BG] = patch[patch != BG]
                x += int(spec["fp_w"])
            if not ok:
                break
            if row_idx < len(row_gaps):
                y += row_height + row_gaps[row_idx]
        if ok:
            return grid
    return None


def _valid_example(grid: np.ndarray) -> bool:
    out = main(grid)
    if out.shape != grid.shape:
        return False
    if not np.array_equal(grid == FENCE, out == FENCE):
        return False
    if np.array_equal(grid, out):
        return False
    for fence in _fence_components(grid, BG):
        r0, c0, r1, c1 = fence["bbox"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        if np.any((interior != BG) & (interior != int(fence["color"]))):
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    fence_count = rng.randint(2, 4)
    counts = _count_choices(rng, fence_count)
    directions = _direction_choices(rng, fence_count)
    specs = [_make_fence_spec(rng, direction, count) for direction, count in zip(directions, counts)]
    grid = _pack_specs(rng, specs)
    if grid is None:
        return None
    if not _valid_example(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid example")
