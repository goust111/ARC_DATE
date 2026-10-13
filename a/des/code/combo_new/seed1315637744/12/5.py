from __future__ import annotations

import numpy as np


BG = 0
GREY = 5
NON_SPECIAL_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

MIN_OBJECTS = 3
MAX_OBJECTS = 4
MIN_BOX = 3
MAX_BOX_BY_COUNT = {3: 12, 4: 10}
GRID_MIN_BY_COUNT = {3: 20, 4: 24}
GRID_MAX = 30
OUTER_MARGIN = 1
PALETTE_CLEARANCE = 2
OBJECT_CLEARANCE = 1
GENERATION_ATTEMPTS = 400


def _randint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(int(lo), int(hi) + 1))


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
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
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))

            comps.append(np.array(coords, dtype=int))

    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return (
        int(rows.min()),
        int(cols.min()),
        int(rows.max()),
        int(cols.max()),
    )


def _mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    mask = np.zeros((h, w), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0, r1, c1)


def _boxes_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    return not (
        int(a[2]) + int(gap) < int(b[0])
        or int(b[2]) + int(gap) < int(a[0])
        or int(a[3]) + int(gap) < int(b[1])
        or int(b[3]) + int(gap) < int(a[1])
    )


def _find_palette(g: np.ndarray) -> dict:
    candidates = []
    for coords in _components(g == GREY):
        r0, c0, r1, c1 = _bbox(coords)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h != 4 or w < 5:
            continue
        if int(coords.shape[0]) != 2 * (h + w) - 4:
            continue

        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not bool(np.all(sub[border] == GREY)):
            continue

        inner = sub[1:-1, 1:-1]
        if inner.shape[0] != 2:
            continue
        if bool(np.any((inner == BG) | (inner == GREY))):
            continue

        top = tuple(map(int, inner[0]))
        bottom = tuple(map(int, inner[1]))
        if len(set(top)) != len(top) or len(set(bottom)) != len(bottom):
            continue

        mapping = dict(zip(top, bottom))
        if len(mapping) != len(top):
            continue

        candidates.append(
            {
                "bbox": (r0, c0, r1, c1),
                "mapping": mapping,
                "n_cols": int(inner.shape[1]),
            }
        )

    assert len(candidates) == 1, "expected exactly one grey palette frame"
    return candidates[0]


def _extract_objects(g: np.ndarray, palette_bbox: tuple[int, int, int, int]) -> list[dict]:
    mask = g != BG
    r0, c0, r1, c1 = palette_bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = False

    objects = []
    for coords in _components(mask):
        colors = np.unique(g[coords[:, 0], coords[:, 1]])
        assert len(colors) == 1, "each pattern must be monochrome"
        color = int(colors[0])
        assert color != GREY, "only the palette frame may be grey"
        shape, bbox = _mask_from_coords(coords)
        objects.append(
            {
                "color": color,
                "area": int(coords.shape[0]),
                "mask": shape,
                "bbox": bbox,
            }
        )

    assert objects, "expected at least one recolorable pattern"
    areas = [obj["area"] for obj in objects]
    assert len(areas) == len(set(areas)), "pattern areas must be unique"
    return objects


def _stack_objects(objects: list[dict], mapping: dict[int, int]) -> np.ndarray:
    ordered = sorted(objects, key=lambda obj: int(obj["area"]), reverse=True)
    base_h, base_w = map(int, ordered[0]["mask"].shape)
    center_r = base_h // 2
    center_c = base_w // 2

    out = np.zeros((base_h + 2, base_w + 2), dtype=int)
    for obj in ordered:
        src_color = int(obj["color"])
        assert src_color in mapping, "every pattern color must appear in the palette top row"
        dst_color = int(mapping[src_color])
        mask = obj["mask"]
        h, w = map(int, mask.shape)
        assert h <= base_h and w <= base_w, "largest pattern bbox must contain the others when centered"

        top = 1 + center_r - h // 2
        left = 1 + center_c - w // 2
        out[top : top + h, left : left + w][mask] = dst_color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor palette-keyed monochrome patterns and center-stack them from largest area to smallest.

    Input structure:
    - Black background surrounds several disconnected monochrome 4-connected patterns.
    - A single small central palette is enclosed by a grey rectangular frame whose two interior rows give a top-to-bottom color mapping.
    - Every pattern uses one distinct source color that appears exactly once in the palette's top row.

    Rule:
    - Read the palette column by column to map each top-row color to the bottom-row color beneath it.
    - Remove the palette and recolor every pattern with its mapped destination color.
    - Sort the patterns by area from largest to smallest, then place their cropped masks so they share the same center, with smaller patterns painted over larger ones.
    - Return only the largest pattern's bbox size plus a one-cell black border around the centered stack.

    Key insight:
    - The grey frame is not an object to keep; it is a legend that first defines a color substitution and then supplies the common alignment center for rebuilding all other shapes into one nested stack.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    palette = _find_palette(g)
    objects = _extract_objects(g, palette["bbox"])
    obj_colors = {int(obj["color"]) for obj in objects}
    assert obj_colors == set(palette["mapping"]), "palette keys must match the object colors exactly"
    assert len(objects) == int(palette["n_cols"]), "palette width must match the number of patterns"
    return _stack_objects(objects, palette["mapping"])


def _grow_shape(h: int, w: int, target_area: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    spine_r = _randint(rng, 0, h - 1)
    spine_c = _randint(rng, 0, w - 1)
    mask[spine_r, :] = True
    mask[:, spine_c] = True
    assert int(mask.sum()) <= int(target_area)

    frontier: set[tuple[int, int]] = set()

    def add_frontier(rr: int, cc: int) -> None:
        for nr, nc in _neighbors4(rr, cc, h, w):
            if not mask[nr, nc]:
                frontier.add((int(nr), int(nc)))

    for rr, cc in np.argwhere(mask):
        add_frontier(int(rr), int(cc))

    while int(mask.sum()) < int(target_area):
        assert frontier, "frontier must remain nonempty while growing a connected shape"
        options = tuple(frontier)
        pick = options[_randint(rng, 0, len(options) - 1)]
        frontier.remove(pick)
        rr, cc = pick
        mask[rr, cc] = True
        add_frontier(rr, cc)

    return mask


def _sample_boxes(n_objects: int, rng: np.random.Generator) -> list[tuple[int, int]] | None:
    max_box = int(MAX_BOX_BY_COUNT[int(n_objects)])
    boxes_asc: list[tuple[int, int]] = []
    h = _randint(rng, MIN_BOX, 5)
    w = _randint(rng, MIN_BOX, 5)
    boxes_asc.append((h, w))

    while len(boxes_asc) < int(n_objects):
        steps_left = int(n_objects) - len(boxes_asc) - 1
        h_cap = int(max_box) - steps_left
        w_cap = int(max_box) - steps_left
        if h + 1 > h_cap or w + 1 > w_cap:
            return None

        next_h = _randint(rng, h + 1, min(h + 3, h_cap))
        next_w = _randint(rng, w + 1, min(w + 3, w_cap))
        h, w = next_h, next_w
        boxes_asc.append((h, w))

    return list(reversed(boxes_asc))


def _sample_areas(boxes_desc: list[tuple[int, int]], rng: np.random.Generator) -> list[int] | None:
    boxes_asc = list(reversed(boxes_desc))
    prev_area = 0
    areas_asc: list[int] = []

    for h, w in boxes_asc:
        box_area = int(h * w)
        min_area = max(int(h + w - 1), int(prev_area + 1), int(np.ceil(0.5 * box_area)))
        soft_high = max(int(min_area), int(np.floor(0.82 * box_area)))
        high = min(int(box_area), int(soft_high))
        if high < min_area:
            high = box_area
        if high < min_area:
            return None

        area = _randint(rng, min_area, high)
        areas_asc.append(int(area))
        prev_area = int(area)

    return list(reversed(areas_asc))


def _sample_objects(rng: np.random.Generator) -> list[dict] | None:
    n_objects = _randint(rng, MIN_OBJECTS, MAX_OBJECTS)
    boxes_desc = _sample_boxes(n_objects, rng)
    if boxes_desc is None:
        return None

    areas_desc = _sample_areas(boxes_desc, rng)
    if areas_desc is None:
        return None

    objects = []
    for (h, w), area in zip(boxes_desc, areas_desc):
        mask = _grow_shape(int(h), int(w), int(area), rng)
        objects.append({"mask": mask, "area": int(area)})

    return objects


def _sample_palette(n_objects: int, rng: np.random.Generator) -> tuple[list[int], dict[int, int]]:
    chosen = list(map(int, rng.choice(NON_SPECIAL_COLORS, size=2 * int(n_objects), replace=False)))
    top = chosen[:n_objects]
    bottom = chosen[n_objects:]
    rng.shuffle(top)
    rng.shuffle(bottom)
    return top, {int(src): int(dst) for src, dst in zip(top, bottom)}


def _candidate_positions(
    grid_shape: tuple[int, int],
    box_shape: tuple[int, int],
    palette_bbox: tuple[int, int, int, int],
    placed: list[tuple[int, int, int, int]],
) -> list[tuple[int, int]]:
    grid_h, grid_w = map(int, grid_shape)
    h, w = map(int, box_shape)
    positions: list[tuple[int, int]] = []

    for r in range(OUTER_MARGIN, grid_h - h - OUTER_MARGIN + 1):
        for c in range(OUTER_MARGIN, grid_w - w - OUTER_MARGIN + 1):
            cand = (int(r), int(c), int(r + h - 1), int(c + w - 1))
            if _boxes_close(cand, palette_bbox, PALETTE_CLEARANCE):
                continue
            if any(_boxes_close(cand, other, OBJECT_CLEARANCE) for other in placed):
                continue
            positions.append((int(r), int(c)))

    return positions


def _place_objects(
    objects: list[dict],
    grid_shape: tuple[int, int],
    palette_bbox: tuple[int, int, int, int],
    rng: np.random.Generator,
) -> bool:
    placed_boxes: list[tuple[int, int, int, int]] = []
    for obj in sorted(objects, key=lambda item: int(item["area"]), reverse=True):
        h, w = map(int, obj["mask"].shape)
        positions = _candidate_positions(grid_shape, (h, w), palette_bbox, placed_boxes)
        if not positions:
            return False

        rng.shuffle(positions)
        row, col = positions[0]
        obj["top"] = int(row)
        obj["left"] = int(col)
        bbox = (int(row), int(col), int(row + h - 1), int(col + w - 1))
        obj["bbox"] = bbox
        placed_boxes.append(bbox)

    return True


def _render_input(
    grid_shape: tuple[int, int],
    palette_top: list[int],
    mapping: dict[int, int],
    objects: list[dict],
) -> np.ndarray:
    grid_h, grid_w = map(int, grid_shape)
    n_objects = len(objects)
    palette_w = int(n_objects) + 2
    palette_r = grid_h // 2 - 2
    palette_c = grid_w // 2 - palette_w // 2

    g = np.zeros((grid_h, grid_w), dtype=int)
    g[palette_r : palette_r + 4, palette_c : palette_c + palette_w] = GREY
    g[palette_r + 1, palette_c + 1 : palette_c + 1 + n_objects] = np.array(palette_top, dtype=int)
    g[palette_r + 2, palette_c + 1 : palette_c + 1 + n_objects] = np.array(
        [mapping[int(color)] for color in palette_top],
        dtype=int,
    )

    for obj, src_color in zip(objects, palette_top):
        h, w = map(int, obj["mask"].shape)
        top = int(obj["top"])
        left = int(obj["left"])
        g[top : top + h, left : left + w][obj["mask"]] = int(src_color)

    return g


def _expected_output(objects: list[dict], mapping: dict[int, int], palette_top: list[int]) -> np.ndarray:
    descriptors = []
    for obj, src_color in zip(objects, palette_top):
        descriptors.append(
            {
                "color": int(src_color),
                "area": int(obj["area"]),
                "mask": obj["mask"].copy(),
            }
        )
    return _stack_objects(descriptors, mapping)


def _sample_grid_shape(objects: list[dict], rng: np.random.Generator) -> tuple[int, int] | None:
    n_objects = len(objects)
    largest_h, largest_w = map(int, objects[0]["mask"].shape)
    second_h, second_w = map(int, objects[1]["mask"].shape)
    palette_w = n_objects + 2

    min_h = max(int(GRID_MIN_BY_COUNT[int(n_objects)]), int(largest_h + second_h + 8))
    min_w = max(int(GRID_MIN_BY_COUNT[int(n_objects)]), int(largest_w + second_w + 8), int(palette_w + 6))
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    grid_h = _randint(rng, min_h, GRID_MAX)
    grid_w = _randint(rng, min_w, GRID_MAX)
    return int(grid_h), int(grid_w)


def _build_example(rng: np.random.Generator) -> np.ndarray | None:
    objects = _sample_objects(rng)
    if objects is None:
        return None

    grid_shape = _sample_grid_shape(objects, rng)
    if grid_shape is None:
        return None

    n_objects = len(objects)
    palette_top, mapping = _sample_palette(n_objects, rng)

    grid_h, grid_w = grid_shape
    palette_w = n_objects + 2
    palette_bbox = (
        int(grid_h // 2 - 2),
        int(grid_w // 2 - palette_w // 2),
        int(grid_h // 2 + 1),
        int(grid_w // 2 - palette_w // 2 + palette_w - 1),
    )

    if not _place_objects(objects, grid_shape, palette_bbox, rng):
        return None

    inp = _render_input(grid_shape, palette_top, mapping, objects)
    expected = _expected_output(objects, mapping, palette_top)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        inp = _build_example(rng)
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid palette-stack example")
