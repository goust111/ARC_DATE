import random
from collections import defaultdict

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GRID_SIDE_RANGE = (16, 30)
PAIR_COUNT_CHOICES = (2, 3, 4)
MAX_GENERATION_ATTEMPTS = 400
MAX_PAIR_ATTEMPTS = 140
MAX_SHAPE_ATTEMPTS = 100
PLACEMENT_PADDING = 1
MAX_GAP = 4


def _bbox(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords):
    r0, c0, _, _ = _bbox(coords)
    return frozenset((r - r0, c - c0) for r, c in coords)


def _shape_dims(shape):
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _hflip(shape, width):
    return frozenset((r, width - 1 - c) for r, c in shape)


def _vflip(shape, height):
    return frozenset((height - 1 - r, c) for r, c in shape)


def _translate(shape, dr, dc):
    return tuple((r + dr, c + dc) for r, c in sorted(shape))


def _encode_shape(shape):
    h, w = _shape_dims(shape)
    return h, w, tuple(sorted(shape))


def _family_key(shape):
    h, w = _shape_dims(shape)
    variants = (
        shape,
        _hflip(shape, w),
        _vflip(shape, h),
        _vflip(_hflip(shape, w), h),
    )
    return min(_encode_shape(variant) for variant in variants)


def _paint(grid, coords, color):
    for r, c in coords:
        grid[r, c] = color


def _can_place(occupied, coords, pad):
    h, w = occupied.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        r0 = max(0, r - pad)
        r1 = min(h, r + pad + 1)
        c0 = max(0, c - pad)
        c1 = min(w, c + pad + 1)
        if occupied[r0:r1, c0:c1].any():
            return False
    return True


def _mark(occupied, coords):
    for r, c in coords:
        occupied[r, c] = True


def _random_shape(rng, used_families):
    for _ in range(MAX_SHAPE_ATTEMPTS):
        height = rng.randint(2, 4)
        width = rng.randint(2, 4)
        max_area = min(height * width - 1, 7)
        min_area = max(3, (height * width + 2) // 3)
        if min_area > max_area:
            continue
        target_area = rng.randint(min_area, max_area)

        cells = {(rng.randrange(height), rng.randrange(width))}
        while len(cells) < target_area:
            frontier = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))

        shape = _normalize(cells)
        if len(shape) != target_area:
            continue
        sh, sw = _shape_dims(shape)
        if sh != height or sw != width:
            continue
        if _hflip(shape, width) == shape or _vflip(shape, height) == shape:
            continue
        key = _family_key(shape)
        if key in used_families:
            continue
        return shape
    return None


def _find_objects(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            box = _bbox(coords)
            objects.append(
                {
                    "color": color,
                    "coords": tuple(coords),
                    "bbox": box,
                    "shape": _normalize(coords),
                    "height": box[2] - box[0] + 1,
                    "width": box[3] - box[1] + 1,
                }
            )
    return objects


def _pair_options(objects):
    options = []
    count = len(objects)
    for i in range(count):
        for j in range(i + 1, count):
            ai = objects[i]
            aj = objects[j]

            left, right = sorted((i, j), key=lambda idx: (objects[idx]["bbox"][1], objects[idx]["bbox"][0]))
            a = objects[left]
            b = objects[right]
            if (
                a["bbox"][0] == b["bbox"][0]
                and a["bbox"][2] == b["bbox"][2]
                and a["bbox"][3] < b["bbox"][1]
                and _hflip(a["shape"], a["width"]) == b["shape"]
            ):
                options.append({"kind": "vertical", "a": left, "b": right})

            top, bottom = sorted((i, j), key=lambda idx: (objects[idx]["bbox"][0], objects[idx]["bbox"][1]))
            a = objects[top]
            b = objects[bottom]
            if (
                a["bbox"][1] == b["bbox"][1]
                and a["bbox"][3] == b["bbox"][3]
                and a["bbox"][2] < b["bbox"][0]
                and _vflip(a["shape"], a["height"]) == b["shape"]
            ):
                options.append({"kind": "horizontal", "a": top, "b": bottom})
    return options


def _solve_matching(objects):
    options = _pair_options(objects)
    by_object = defaultdict(list)
    for option in options:
        by_object[option["a"]].append(option)
        by_object[option["b"]].append(option)
    for key in by_object:
        by_object[key].sort(key=lambda opt: (opt["kind"], opt["a"], opt["b"]))

    def search(remaining):
        if not remaining:
            return []
        idx = min(remaining)
        for option in by_object.get(idx, []):
            other = option["b"] if option["a"] == idx else option["a"]
            if other not in remaining:
                continue
            next_remaining = remaining - {idx, other}
            tail = search(next_remaining)
            if tail is not None:
                return [option] + tail
        return None

    return search(set(range(len(objects))))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair mirrored fragments by shape, then slide each pair together onto its symmetry axis to make a two-color whole.

    Input structure:
    - The background is black.
    - The grid contains an even number of small non-overlapping monochrome objects, each with its own color.
    - Objects come in pairs: within each pair the two shapes match exactly after either a horizontal flip or a vertical flip.
    - A left-right pair already shares the same row span, and a top-bottom pair already shares the same column span.

    Rule:
    - Find the unique partner of each object by mirror-shape matching.
    - For a horizontal-mirror pair, keep both row positions fixed and translate the left and right pieces inward until they become adjacent on their vertical midline.
    - For a vertical-mirror pair, keep both column positions fixed and translate the top and bottom pieces inward until they become adjacent on their horizontal midline.
    - Clear the original scattered fragments and keep only the merged two-color objects.

    Key insight:
    - Colors do not identify the pairs; the pairing key is the shared binary shape under a single mirror flip, and the final placement is the midpoint between the two matching fragments.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    objects = _find_objects(g)
    if not objects:
        return np.zeros_like(g)
    assert len(objects) % 2 == 0

    matching = _solve_matching(objects)
    assert matching is not None and len(matching) * 2 == len(objects)

    out = np.zeros_like(g)
    for option in matching:
        first = objects[option["a"]]
        second = objects[option["b"]]

        if option["kind"] == "vertical":
            left, right = first, second
            new_left_right = (left["bbox"][3] + right["bbox"][1] - 1) // 2
            dx_left = new_left_right - left["bbox"][3]
            dx_right = new_left_right + 1 - right["bbox"][1]
            moved_left = ((r, c + dx_left) for r, c in left["coords"])
            moved_right = ((r, c + dx_right) for r, c in right["coords"])
        else:
            top, bottom = first, second
            new_top_bottom = (top["bbox"][2] + bottom["bbox"][0] - 1) // 2
            dy_top = new_top_bottom - top["bbox"][2]
            dy_bottom = new_top_bottom + 1 - bottom["bbox"][0]
            moved_left = ((r + dy_top, c) for r, c in top["coords"])
            moved_right = ((r + dy_bottom, c) for r, c in bottom["coords"])

        for r, c in moved_left:
            assert out[r, c] == BG
            out[r, c] = first["color"]
        for r, c in moved_right:
            assert out[r, c] == BG
            out[r, c] = second["color"]

    return out


def _orientation_plan(rng, num_pairs):
    num_vertical = num_pairs // 2
    num_horizontal = num_pairs // 2
    if num_pairs % 2 == 1:
        if rng.random() < 0.5:
            num_vertical += 1
        else:
            num_horizontal += 1
    plan = ["vertical"] * num_vertical + ["horizontal"] * num_horizontal
    rng.shuffle(plan)
    return plan


def _build_pair(shape, orientation, anchor_r, anchor_c, gap):
    height, width = _shape_dims(shape)
    if orientation == "vertical":
        mirrored = _hflip(shape, width)
        input_a = _translate(shape, anchor_r, anchor_c - gap)
        input_b = _translate(mirrored, anchor_r, anchor_c + width + gap)
        output_a = _translate(shape, anchor_r, anchor_c)
        output_b = _translate(mirrored, anchor_r, anchor_c + width)
    else:
        mirrored = _vflip(shape, height)
        input_a = _translate(shape, anchor_r - gap, anchor_c)
        input_b = _translate(mirrored, anchor_r + height + gap, anchor_c)
        output_a = _translate(shape, anchor_r, anchor_c)
        output_b = _translate(mirrored, anchor_r + height, anchor_c)
    return input_a, input_b, output_a, output_b


def _sample_pair_layout(rng, shape, orientation, grid_shape):
    height, width = _shape_dims(shape)
    h, w = grid_shape
    if orientation == "vertical":
        max_gap = min(MAX_GAP, (w - 2 * width) // 2)
        if max_gap < 1 or height > h:
            return None
        gap = rng.randint(1, max_gap)
        anchor_r = rng.randint(0, h - height)
        anchor_c = rng.randint(gap, w - (2 * width + gap))
    else:
        max_gap = min(MAX_GAP, (h - 2 * height) // 2)
        if max_gap < 1 or width > w:
            return None
        gap = rng.randint(1, max_gap)
        anchor_r = rng.randint(gap, h - (2 * height + gap))
        anchor_c = rng.randint(0, w - width)
    return _build_pair(shape, orientation, anchor_r, anchor_c, gap)


def _gen_attempt(rng):
    height = rng.randint(*GRID_SIDE_RANGE)
    width = rng.randint(*GRID_SIDE_RANGE)
    grid = np.zeros((height, width), dtype=int)
    expected = np.zeros_like(grid)
    input_occupied = np.zeros_like(grid, dtype=bool)
    output_occupied = np.zeros_like(grid, dtype=bool)

    num_pairs = rng.choice(PAIR_COUNT_CHOICES)
    orientations = _orientation_plan(rng, num_pairs)
    colors = rng.sample(PALETTE, 2 * num_pairs)
    used_families = set()

    for pair_idx, orientation in enumerate(orientations):
        placed = False
        for _ in range(MAX_PAIR_ATTEMPTS):
            shape = _random_shape(rng, used_families)
            if shape is None:
                break
            layout = _sample_pair_layout(rng, shape, orientation, grid.shape)
            if layout is None:
                continue
            input_a, input_b, output_a, output_b = layout
            input_coords = input_a + input_b
            output_coords = output_a + output_b
            if not _can_place(input_occupied, input_coords, PLACEMENT_PADDING):
                continue
            if not _can_place(output_occupied, output_coords, PLACEMENT_PADDING):
                continue

            color_a = colors[2 * pair_idx]
            color_b = colors[2 * pair_idx + 1]
            _paint(grid, input_a, color_a)
            _paint(grid, input_b, color_b)
            _paint(expected, output_a, color_a)
            _paint(expected, output_b, color_b)
            _mark(input_occupied, input_coords)
            _mark(output_occupied, output_coords)
            used_families.add(_family_key(shape))
            placed = True
            break
        if not placed:
            return None

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirrored-fragment sample")
