from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GEN_ATTEMPTS = 400
GRID_MAX_SIZE = 30
PAIR_COUNT_CHOICES = (1, 2, 3)
MAX_SINGLE_COUNT = 2
PAIR_RECT_RANGE = (3, 7)
SINGLE_RECT_RANGE = (2, 6)
BRIDGE_GAP_RANGE = (1, 6)
PATCH_ATTEMPTS = 100
LAYOUT_ATTEMPTS = 200
ROW_COUNT_MAX = 3
ROW_GAP_RANGE = (0, 2)
COL_GAP_RANGE = (0, 2)
OUTER_MARGIN_RANGE = (0, 3)
EXTRA_PAD_RANGE = (0, 4)


def _neighbors4(r: int, c: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
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


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, (h, w)):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.asarray(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return (
        int(rows.min()),
        int(cols.min()),
        int(rows.max()),
        int(cols.max()),
    )


def _sprite_from_coords(grid: np.ndarray, coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    sprite = np.full((r1 - r0 + 1, c1 - c0 + 1), BLACK, dtype=int)
    color = int(grid[coords[0, 0], coords[0, 1]])
    sprite[coords[:, 0] - r0, coords[:, 1] - c0] = color
    return sprite


def _is_straight_line(coords: np.ndarray) -> bool:
    rows = coords[:, 0]
    cols = coords[:, 1]
    if np.all(rows == rows[0]):
        return int(cols.max()) - int(cols.min()) + 1 == len(coords)
    if np.all(cols == cols[0]):
        return int(rows.max()) - int(rows.min()) + 1 == len(coords)
    return False


def _analyze_scene(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    object_labels = np.full(g.shape, -1, dtype=int)
    objects: list[dict] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) not in (BLACK, GREY)):
        for coords in _connected_components(g == color):
            obj_id = len(objects)
            object_labels[coords[:, 0], coords[:, 1]] = obj_id
            objects.append(
                {
                    "id": obj_id,
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(coords.shape[0]),
                    "sprite": _sprite_from_coords(g, coords),
                }
            )

    bridge_degree = [0] * len(objects)
    bridges: list[dict] = []
    for coords in _connected_components(g == GREY):
        assert _is_straight_line(coords), "each grey bridge must be a straight 1-pixel-wide segment"
        touched: set[int] = set()
        for r, c in coords:
            for nr, nc in _neighbors4(int(r), int(c), g.shape):
                obj_id = int(object_labels[nr, nc])
                if obj_id >= 0:
                    touched.add(obj_id)

        assert len(touched) == 2, "each grey bridge must connect exactly two objects"
        a, b = sorted(touched)
        bridge_degree[a] += 1
        bridge_degree[b] += 1
        assert bridge_degree[a] <= 1 and bridge_degree[b] <= 1, "each object may participate in at most one bridge"
        bridges.append({"coords": coords, "pair": (a, b)})

    return objects, bridges


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find object pairs connected by grey bridges and stamp the smaller object into the larger one.

    Input structure
    - The background is black.
    - Foreground objects are monochrome 4-connected shapes; task instances use solid rectangles.
    - Some pairs of objects are linked by a one-pixel-wide straight grey bridge.

    Rule
    - Each bridge defines exactly one pair of objects.
    - Within each pair, compare areas to choose a smaller source object and a larger target object.
    - Erase the smaller object and all grey bridge cells to black.
    - Stamp the smaller object's normalized mask into the center of the larger object's bbox using the smaller object's color.
    - Objects not participating in a bridge stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    objects, bridges = _analyze_scene(g)

    out = g.copy()
    out[g == GREY] = BLACK

    for bridge in bridges:
        left_id, right_id = bridge["pair"]
        obj_a = objects[left_id]
        obj_b = objects[right_id]
        assert obj_a["area"] != obj_b["area"], "paired objects must have different areas"
        small, large = (obj_a, obj_b) if obj_a["area"] < obj_b["area"] else (obj_b, obj_a)

        out[small["coords"][:, 0], small["coords"][:, 1]] = BLACK

        sr0, sc0, sr1, sc1 = _bbox(small["coords"])
        lr0, lc0, lr1, lc1 = large["bbox"]
        sh = sr1 - sr0 + 1
        sw = sc1 - sc0 + 1
        lh = lr1 - lr0 + 1
        lw = lc1 - lc0 + 1
        assert sh <= lh and sw <= lw, "the stamped object must fit inside the larger object's bbox"
        assert (lh - sh) % 2 == 0 and (lw - sw) % 2 == 0, "object centers must align on grid cells"

        top = lr0 + (lh - sh) // 2
        left = lc0 + (lw - sw) // 2
        target = out[top : top + sh, left : left + sw]
        sprite = small["sprite"]
        mask = sprite != BLACK
        target[mask] = sprite[mask]

    return out


def _sample_small_dims(big_h: int, big_w: int) -> tuple[int, int] | None:
    options = [
        (h, w)
        for h in range(1, big_h + 1)
        for w in range(1, big_w + 1)
        if (big_h - h) % 2 == 0
        and (big_w - w) % 2 == 0
        and h * w >= 2
        and h * w < big_h * big_w
    ]
    if not options:
        return None
    return random.choice(options)


def _choose_overlapping_offsets(size_a: int, size_b: int) -> tuple[int, int, int] | None:
    for _ in range(PATCH_ATTEMPTS):
        extent = max(size_a, size_b) + random.randint(0, 3)
        off_a = random.randint(0, extent - size_a)
        off_b = random.randint(0, extent - size_b)
        lo = max(off_a, off_b)
        hi = min(off_a + size_a - 1, off_b + size_b - 1)
        if lo <= hi:
            return extent, off_a, off_b
    return None


def _make_pair_patch(small_color: int, large_color: int) -> np.ndarray:
    for _ in range(PATCH_ATTEMPTS):
        big_h = random.randint(*PAIR_RECT_RANGE)
        big_w = random.randint(*PAIR_RECT_RANGE)
        small_dims = _sample_small_dims(big_h, big_w)
        if small_dims is None:
            continue
        small_h, small_w = small_dims

        orientation = random.choice(("H", "V"))
        gap = random.randint(*BRIDGE_GAP_RANGE)
        small_first = bool(random.getrandbits(1))

        if small_first:
            first_h, first_w, first_color = small_h, small_w, small_color
            second_h, second_w, second_color = big_h, big_w, large_color
        else:
            first_h, first_w, first_color = big_h, big_w, large_color
            second_h, second_w, second_color = small_h, small_w, small_color

        if orientation == "H":
            overlap = _choose_overlapping_offsets(first_h, second_h)
            if overlap is None:
                continue
            inner_h, top_a, top_b = overlap
            inner_w = first_w + gap + second_w
            patch = np.full((inner_h, inner_w), BLACK, dtype=int)
            patch[top_a : top_a + first_h, :first_w] = first_color
            second_c0 = first_w + gap
            patch[top_b : top_b + second_h, second_c0 : second_c0 + second_w] = second_color
            bridge_row = random.randint(max(top_a, top_b), min(top_a + first_h - 1, top_b + second_h - 1))
            patch[bridge_row, first_w:second_c0] = GREY
        else:
            overlap = _choose_overlapping_offsets(first_w, second_w)
            if overlap is None:
                continue
            inner_w, left_a, left_b = overlap
            inner_h = first_h + gap + second_h
            patch = np.full((inner_h, inner_w), BLACK, dtype=int)
            patch[:first_h, left_a : left_a + first_w] = first_color
            second_r0 = first_h + gap
            patch[second_r0 : second_r0 + second_h, left_b : left_b + second_w] = second_color
            bridge_col = random.randint(max(left_a, left_b), min(left_a + first_w - 1, left_b + second_w - 1))
            patch[first_h:second_r0, bridge_col] = GREY

        return np.pad(patch, 1, mode="constant", constant_values=BLACK)

    raise RuntimeError("failed to build a valid pair patch")


def _make_single_patch(color: int) -> np.ndarray:
    h = random.randint(*SINGLE_RECT_RANGE)
    w = random.randint(*SINGLE_RECT_RANGE)
    patch = np.full((h + 2, w + 2), BLACK, dtype=int)
    patch[1 : 1 + h, 1 : 1 + w] = color
    return patch


def _random_partition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    bounds = [0] + cuts + [total]
    return [bounds[i + 1] - bounds[i] for i in range(parts)]


def _paste_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    target = grid[top : top + h, left : left + w]
    mask = patch != BLACK
    assert np.all(target[mask] == BLACK), "layout patches must not overlap"
    target[mask] = patch[mask]


def _layout_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    count = len(patches)
    for _ in range(LAYOUT_ATTEMPTS):
        ordered = patches[:]
        random.shuffle(ordered)
        row_count = random.randint(1, min(ROW_COUNT_MAX, count))
        row_sizes = _random_partition(count, row_count)

        rows: list[list[np.ndarray]] = []
        index = 0
        for size in row_sizes:
            rows.append(ordered[index : index + size])
            index += size

        row_gaps = [random.randint(*ROW_GAP_RANGE) for _ in range(row_count - 1)]
        margin_top = random.randint(*OUTER_MARGIN_RANGE)
        margin_bottom = random.randint(*OUTER_MARGIN_RANGE)
        margin_left = random.randint(*OUTER_MARGIN_RANGE)
        margin_right = random.randint(*OUTER_MARGIN_RANGE)

        row_specs: list[tuple[list[np.ndarray], list[int], int, int]] = []
        max_row_width = 0
        total_height = margin_top + margin_bottom + sum(row_gaps)
        for row in rows:
            col_gaps = [random.randint(*COL_GAP_RANGE) for _ in range(len(row) - 1)]
            row_height = max(int(patch.shape[0]) for patch in row)
            row_width = sum(int(patch.shape[1]) for patch in row) + sum(col_gaps)
            row_specs.append((row, col_gaps, row_height, row_width))
            max_row_width = max(max_row_width, row_width)
            total_height += row_height

        total_width = margin_left + margin_right + max_row_width
        if total_height > GRID_MAX_SIZE or total_width > GRID_MAX_SIZE:
            continue

        grid = np.full((total_height, total_width), BLACK, dtype=int)
        y = margin_top
        for row_index, (row, col_gaps, row_height, row_width) in enumerate(row_specs):
            x = margin_left + random.randint(0, max_row_width - row_width)
            for patch_index, patch in enumerate(row):
                y_offset = random.randint(0, row_height - patch.shape[0])
                _paste_patch(grid, patch, y + y_offset, x)
                x += patch.shape[1]
                if patch_index < len(col_gaps):
                    x += col_gaps[patch_index]
            if row_index < len(row_gaps):
                y += row_height + row_gaps[row_index]

        extra_h = random.randint(EXTRA_PAD_RANGE[0], min(EXTRA_PAD_RANGE[1], GRID_MAX_SIZE - total_height))
        extra_w = random.randint(EXTRA_PAD_RANGE[0], min(EXTRA_PAD_RANGE[1], GRID_MAX_SIZE - total_width))
        pad_top = random.randint(0, extra_h)
        pad_bottom = extra_h - pad_top
        pad_left = random.randint(0, extra_w)
        pad_right = extra_w - pad_left
        if extra_h or extra_w:
            grid = np.pad(grid, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=BLACK)
        return grid

    return None


def _valid_generated_input(grid: np.ndarray, pair_count: int, single_count: int) -> bool:
    try:
        objects, bridges = _analyze_scene(grid)
        out = main(grid)
    except Exception:
        return False

    if len(bridges) != pair_count:
        return False
    paired_ids = {obj_id for bridge in bridges for obj_id in bridge["pair"]}
    if len(objects) - len(paired_ids) != single_count:
        return False
    if len({int(obj["color"]) for obj in objects}) != len(objects):
        return False
    if np.array_equal(out, grid):
        return False
    return out.shape == grid.shape


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        pair_count = random.choice(PAIR_COUNT_CHOICES)
        max_single = min(MAX_SINGLE_COUNT, len(OBJECT_COLORS) - 2 * pair_count)
        single_count = random.randint(0, max_single)
        total_objects = 2 * pair_count + single_count
        colors = random.sample(list(OBJECT_COLORS), total_objects)

        patches: list[np.ndarray] = []
        color_index = 0
        for _pair in range(pair_count):
            small_color = colors[color_index]
            large_color = colors[color_index + 1]
            color_index += 2
            patches.append(_make_pair_patch(small_color, large_color))
        for _single in range(single_count):
            patches.append(_make_single_patch(colors[color_index]))
            color_index += 1

        grid = _layout_patches(patches)
        if grid is not None and _valid_generated_input(grid, pair_count, single_count):
            return grid

    raise RuntimeError("failed to generate a valid input")
