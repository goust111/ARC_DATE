from __future__ import annotations

import random

import numpy as np


Coord = tuple[int, int]
Shape = tuple[Coord, ...]
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
COLOR_POOL = tuple(range(10))
GENERATION_RETRIES = 400
SHAPE_RETRIES = 300
PLACEMENT_RETRIES = 300


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[counts.argmax()])


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
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _normalize(coords: np.ndarray) -> Shape:
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _shape_dims(shape: Shape) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _render_shape(grid: np.ndarray, top: int, left: int, shape: Shape, color: int) -> None:
    for dr, dc in shape:
        grid[top + dr, left + dc] = color


def _component_info(grid: np.ndarray, coords: np.ndarray) -> dict[str, object]:
    r0, c0, r1, c1 = _bbox(coords)
    colors = {int(grid[r, c]) for r, c in coords}
    assert len(colors) == 1, "foreground objects must be monochrome"
    shape = _normalize(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    return {
        "coords": coords,
        "r0": r0,
        "c0": c0,
        "r1": r1,
        "c1": c1,
        "h": height,
        "w": width,
        "shape": shape,
        "color": next(iter(colors)),
    }


def _column_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    c = 0
    width = len(mask)
    while c < width:
        if not mask[c]:
            c += 1
            continue
        c0 = c
        while c + 1 < width and mask[c + 1]:
            c += 1
        runs.append((c0, c))
        c += 1
    return runs


def _parse_legend(left: np.ndarray, bg: int) -> dict[Shape, tuple[Shape, int]]:
    comps = _components(left != bg)
    assert len(comps) >= 4 and len(comps) % 2 == 0, "legend must contain row pairs"
    infos = [_component_info(left, coords) for coords in comps]
    infos.sort(key=lambda item: (int(item["r0"]), int(item["c0"])))

    groups: list[list[dict[str, object]]] = []
    current = [infos[0]]
    current_bottom = int(infos[0]["r1"])
    for info in infos[1:]:
        if int(info["r0"]) <= current_bottom:
            current.append(info)
            current_bottom = max(current_bottom, int(info["r1"]))
        else:
            groups.append(current)
            current = [info]
            current_bottom = int(info["r1"])
    groups.append(current)

    mapping: dict[Shape, tuple[Shape, int]] = {}
    for group in groups:
        assert len(group) == 2, "each legend row must contain exactly two templates"
        first, second = sorted(group, key=lambda item: int(item["c0"]))
        assert int(first["c1"]) < int(second["c0"]), "legend templates must be separated horizontally"
        assert (int(first["r0"]), int(first["r1"])) == (int(second["r0"]), int(second["r1"]))
        assert (int(first["h"]), int(first["w"])) == (int(second["h"]), int(second["w"]))
        left_shape = first["shape"]
        right_shape = second["shape"]
        assert left_shape != right_shape, "legend row must translate to a different shape"
        assert left_shape not in mapping, "left-column template shapes must be unique"
        mapping[left_shape] = (right_shape, int(second["color"]))

    return mapping


def _apply_mapping(grid: np.ndarray, bg: int, mapping: dict[Shape, tuple[Shape, int]], right_start: int) -> np.ndarray:
    right = grid[:, right_start:]
    comps = _components(right != bg)
    assert comps, "target area must contain objects"

    objects = []
    for coords in comps:
        info = _component_info(right, coords)
        source_shape = info["shape"]
        assert source_shape in mapping, "each target object must match a left legend shape"
        target_shape, target_color = mapping[source_shape]
        assert _shape_dims(source_shape) == _shape_dims(target_shape), "paired legend templates must share a bbox size"
        objects.append((info, target_shape, target_color))

    out = grid.copy()
    for info, _, _ in objects:
        coords = np.asarray(info["coords"], dtype=int)
        out[coords[:, 0], right_start + coords[:, 1]] = bg

    for info, target_shape, target_color in objects:
        top = int(info["r0"])
        left = right_start + int(info["c0"])
        for dr, dc in target_shape:
            rr = top + dr
            cc = left + dc
            assert out[rr, cc] == bg, "translated objects must not overlap"
            out[rr, cc] = target_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each right-side object by the right-column legend template whose row is
    keyed by that object's shape.

    Input structure:
    - A two-column legend sits on the left.
    - A blank vertical gap separates the legend from the target region on the right.
    - Each right-side foreground object has the same normalized mask as one left-column legend template.

    Rule:
    - Match each target object to the unique left-column template with the same normalized shape.
    - Replace the object with the corresponding right-column template.
    - Place the replacement at the original object's bbox top-left.
    - Use the right-column template's own color.
    - Keep the legend unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = _most_common_color(g)

    full_bg_cols = np.all(g == bg, axis=0)
    outputs: list[np.ndarray] = []
    for c0, c1 in _column_runs(full_bg_cols):
        if c0 == 0 or c1 == g.shape[1] - 1:
            continue
        if not np.any(g[:, :c0] != bg) or not np.any(g[:, c1 + 1 :] != bg):
            continue
        try:
            mapping = _parse_legend(g[:, :c0], bg)
            outputs.append(_apply_mapping(g, bg, mapping, c1 + 1))
        except AssertionError:
            continue

    assert outputs, "failed to identify a valid legend/target split"
    first = outputs[0]
    for candidate in outputs[1:]:
        assert np.array_equal(candidate, first), "multiple valid parses disagree"
    return first


def _connected(mask: np.ndarray) -> bool:
    comps = _components(mask)
    return len(comps) == 1


def _grow_path(mask: np.ndarray, start: Coord, end: Coord) -> None:
    r, c = start
    er, ec = end
    mask[r, c] = True
    while (r, c) != (er, ec):
        choices = []
        if er < r:
            choices.append((r - 1, c))
        if er > r:
            choices.append((r + 1, c))
        if ec < c:
            choices.append((r, c - 1))
        if ec > c:
            choices.append((r, c + 1))
        r, c = random.choice(choices)
        mask[r, c] = True


def _random_shape(height: int, width: int, banned: set[Shape]) -> Shape | None:
    min_area = max(height + width - 2, (height * width + 2) // 3)
    for _ in range(SHAPE_RETRIES):
        seeds = {
            (0, random.randrange(width)),
            (height - 1, random.randrange(width)),
            (random.randrange(height), 0),
            (random.randrange(height), width - 1),
        }
        seed_list = list(seeds)
        random.shuffle(seed_list)

        mask = np.zeros((height, width), dtype=bool)
        mask[seed_list[0]] = True
        for seed in seed_list[1:]:
            current_cells = [tuple(map(int, cell)) for cell in np.argwhere(mask)]
            anchor = random.choice(current_cells)
            _grow_path(mask, seed, anchor)

        initial_area = int(mask.sum())
        if initial_area > height * width - 1:
            continue
        target_area = random.randint(initial_area, height * width - 1)
        while int(mask.sum()) < target_area:
            frontier = {
                (r + dr, c + dc)
                for r, c in map(tuple, np.argwhere(mask))
                for dr, dc in NEIGHBORS4
                if 0 <= r + dr < height and 0 <= c + dc < width and not mask[r + dr, c + dc]
            }
            if not frontier:
                break
            cell = random.choice(list(frontier))
            mask[cell] = True

        if int(mask.sum()) < min_area or mask.all():
            continue
        if not (mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if not _connected(mask):
            continue

        shape = _normalize(np.argwhere(mask))
        if shape in banned:
            continue
        return shape
    return None


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _place_objects(area_h: int, area_w: int, dims: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    placed_boxes: list[tuple[int, int, int, int]] = []
    placements: list[tuple[int, int]] = []
    order = list(range(len(dims)))
    order.sort(key=lambda idx: dims[idx][0] * dims[idx][1], reverse=True)

    slots: list[tuple[int, int] | None] = [None] * len(dims)
    for idx in order:
        height, width = dims[idx]
        for _ in range(PLACEMENT_RETRIES):
            r0 = random.randint(0, area_h - height)
            c0 = random.randint(0, area_w - width)
            box = (r0, c0, r0 + height - 1, c0 + width - 1)
            if any(not _boxes_separated(box, other) for other in placed_boxes):
                continue
            placed_boxes.append(box)
            slots[idx] = (r0, c0)
            break
        else:
            return None
    return [slot for slot in slots if slot is not None]


def _build_output(inp: np.ndarray, placements: list[dict[str, object]], right_start: int) -> np.ndarray:
    out = inp.copy()
    for obj in placements:
        top = int(obj["top"])
        left = right_start + int(obj["left"])
        source_shape = obj["source_shape"]
        for dr, dc in source_shape:
            out[top + dr, left + dc] = int(obj["bg"])
    for obj in placements:
        top = int(obj["top"])
        left = right_start + int(obj["left"])
        for dr, dc in obj["target_shape"]:
            out[top + dr, left + dc] = int(obj["target_color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(COLOR_POOL)
    palette = [color for color in COLOR_POOL if color != bg]
    random.shuffle(palette)

    row_count = random.randint(2, 4)
    if len(palette) < 2 * row_count:
        return None

    row_specs = []
    used_shapes: set[Shape] = set()
    for _ in range(row_count):
        height = random.randint(3, 5)
        width = random.randint(3, 5)
        left_shape = _random_shape(height, width, used_shapes)
        if left_shape is None:
            return None
        used_shapes.add(left_shape)
        right_shape = _random_shape(height, width, used_shapes)
        if right_shape is None:
            return None
        used_shapes.add(right_shape)
        left_color = palette.pop()
        right_color = palette.pop()
        row_specs.append(
            {
                "h": height,
                "w": width,
                "left_shape": left_shape,
                "right_shape": right_shape,
                "left_color": left_color,
                "right_color": right_color,
            }
        )

    top_margin = random.randint(1, 2)
    bottom_margin = random.randint(1, 3)
    left_margin = random.randint(1, 2)
    inner_gap = random.randint(1, 2)
    row_gaps = [random.randint(1, 2) for _ in range(row_count - 1)]

    row_tops = []
    cursor = top_margin
    for idx, spec in enumerate(row_specs):
        row_tops.append(cursor)
        cursor += int(spec["h"])
        if idx < row_count - 1:
            cursor += row_gaps[idx]
    legend_height = cursor + bottom_margin
    legend_width = left_margin + max(2 * int(spec["w"]) + inner_gap for spec in row_specs) + random.randint(1, 2)

    base_object_rows = list(range(row_count))
    random.shuffle(base_object_rows)
    if random.random() < 0.5:
        chosen_rows = base_object_rows
    else:
        chosen_rows = base_object_rows[: random.randint(2, row_count)]
    extra_count = random.randint(0, row_count)
    chosen_rows = chosen_rows + [random.randrange(row_count) for _ in range(extra_count)]
    random.shuffle(chosen_rows)

    object_dims = [(int(row_specs[idx]["h"]), int(row_specs[idx]["w"])) for idx in chosen_rows]
    max_h = max(height for height, _ in object_dims)
    max_w = max(width for _, width in object_dims)
    object_area_h = random.randint(max(legend_height, max_h + 3), max(legend_height + 7, max_h + 8))
    object_area_w = random.randint(max(max_w + 6, 11), max(max_w + 12, 19))
    placements = _place_objects(object_area_h, object_area_w, object_dims)
    if placements is None:
        return None

    split_gap = random.randint(2, 4)
    right_margin = random.randint(1, 3)
    grid_h = max(legend_height, object_area_h)
    grid_w = legend_width + split_gap + object_area_w + right_margin
    grid = np.full((grid_h, grid_w), bg, dtype=int)

    for top, spec in zip(row_tops, row_specs, strict=False):
        width = int(spec["w"])
        _render_shape(grid, top, left_margin, spec["left_shape"], int(spec["left_color"]))
        _render_shape(grid, top, left_margin + width + inner_gap, spec["right_shape"], int(spec["right_color"]))

    right_start = legend_width + split_gap
    target_placements = []
    available_target_colors = [color for color in COLOR_POOL if color != bg]
    for (top, left), row_idx in zip(placements, chosen_rows, strict=False):
        spec = row_specs[row_idx]
        color_mode = random.randrange(3)
        if color_mode == 0:
            source_color = int(spec["left_color"])
        elif color_mode == 1:
            source_color = int(spec["right_color"])
        else:
            source_color = random.choice(available_target_colors)
        _render_shape(grid, top, right_start + left, spec["left_shape"], source_color)
        target_placements.append(
            {
                "top": top,
                "left": left,
                "source_shape": spec["left_shape"],
                "target_shape": spec["right_shape"],
                "target_color": int(spec["right_color"]),
                "bg": bg,
            }
        )

    if int(np.count_nonzero(grid != bg)) * 2 >= grid.size:
        return None

    expected = _build_output(grid, target_placements, right_start)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
