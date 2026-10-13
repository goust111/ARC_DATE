import random
from collections import Counter, defaultdict, deque

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            min_r = max_r = r
            min_c = max_c = c

            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                min_r = min(min_r, cr)
                max_r = max(max_r, cr)
                min_c = min(min_c, cc)
                max_c = max(max_c, cc)

                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min_r, min_c, max_r, max_c),
                    "area": len(coords),
                }
            )

    return comps


def _center2(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return r0 + r1, c0 + c1


def _center_dist2(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar, ac = _center2(a)
    br, bc = _center2(b)
    return (ar - br) ** 2 + (ac - bc) ** 2


def _outline_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    comps = _find_components(g, bg)
    assert comps, "expected foreground objects"

    by_color: dict[int, list[dict]] = defaultdict(list)
    for comp in comps:
        by_color[int(comp["color"])].append(comp)
    assert len(by_color) == 2, "expected exactly two foreground colors"

    ordered = sorted(by_color.items(), key=lambda item: len(item[1]), reverse=True)
    majority_color, majority_objs = ordered[0]
    minority_color, minority_objs = ordered[1]
    assert len(majority_objs) > len(minority_objs), "object counts must define a unique majority color"

    out = g.copy()

    for obj in minority_objs:
        for r, c in obj["coords"]:
            out[r, c] = bg

    for obj in minority_objs:
        nearest = min(majority_objs, key=lambda maj: _center_dist2(obj["bbox"], maj["bbox"]))
        _outline_bbox(out, obj["bbox"], int(nearest["color"]))

    return out


def _shape_from_rows(*rows: str) -> tuple[tuple[int, int], ...]:
    cells = [(r, c) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch == "#"]
    return _normalize_shape(cells)


def _normalize_shape(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _shape_dims(cells: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return 1 + max(r for r, _ in cells), 1 + max(c for _, c in cells)


def _rotate90(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    h, _ = _shape_dims(cells)
    return _normalize_shape([(c, h - 1 - r) for r, c in cells])


def _flip_h(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    _, w = _shape_dims(cells)
    return _normalize_shape([(r, w - 1 - c) for r, c in cells])


def _random_variant(rng: random.Random, catalog: tuple[tuple[tuple[int, int], ...], ...]) -> tuple[tuple[int, int], ...]:
    cells = rng.choice(catalog)
    if rng.random() < 0.5:
        cells = _flip_h(cells)
    for _ in range(rng.randrange(4)):
        cells = _rotate90(cells)
    return cells


def _bbox_border(h: int, w: int) -> set[tuple[int, int]]:
    border: set[tuple[int, int]] = set()
    for c in range(w):
        border.add((0, c))
        border.add((h - 1, c))
    for r in range(h):
        border.add((r, 0))
        border.add((r, w - 1))
    return border


def _is_nontrivial_minority_shape(cells: tuple[tuple[int, int], ...]) -> bool:
    h, w = _shape_dims(cells)
    if h < 2 or w < 2:
        return False
    return set(cells) != _bbox_border(h, w)


def _can_place(reserved: np.ndarray, top: int, left: int, h: int, w: int, pad: int = 1) -> bool:
    rr0 = max(0, top - pad)
    cc0 = max(0, left - pad)
    rr1 = min(reserved.shape[0], top + h + pad)
    cc1 = min(reserved.shape[1], left + w + pad)
    return not bool(reserved[rr0:rr1, cc0:cc1].any())


def _reserve(reserved: np.ndarray, top: int, left: int, h: int, w: int, pad: int = 1) -> None:
    rr0 = max(0, top - pad)
    cc0 = max(0, left - pad)
    rr1 = min(reserved.shape[0], top + h + pad)
    cc1 = min(reserved.shape[1], left + w + pad)
    reserved[rr0:rr1, cc0:cc1] = True


def _paint_shape(grid: np.ndarray, cells: tuple[tuple[int, int], ...], top: int, left: int, color: int) -> None:
    for r, c in cells:
        grid[top + r, left + c] = color


MAJORITY_SHAPES = (
    _shape_from_rows("##"),
    _shape_from_rows("###"),
    _shape_from_rows("####"),
    _shape_from_rows("##", "##"),
    _shape_from_rows("###", "###"),
    _shape_from_rows("#.", "##"),
    _shape_from_rows("###", ".#."),
    _shape_from_rows("##.", ".##"),
    _shape_from_rows("#..", "###"),
    _shape_from_rows("###", "#.#"),
    _shape_from_rows("##", "##", "#."),
    _shape_from_rows(".#.", "###", ".#."),
    _shape_from_rows("###", "###", "###"),
    _shape_from_rows("###", "#.#", "###"),
)

MINORITY_SHAPES = (
    _shape_from_rows("#.", "##"),
    _shape_from_rows("###", ".#."),
    _shape_from_rows("##.", ".##"),
    _shape_from_rows("#..", "###"),
    _shape_from_rows("##", "##", "#."),
    _shape_from_rows("###", "#.#"),
    _shape_from_rows("#..", "##.", ".##"),
    _shape_from_rows(".#.", "###", ".#."),
    _shape_from_rows("###", "###", "###"),
    _shape_from_rows("####", "####", "####"),
)

GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 200


def _sample_counts(rng: random.Random) -> tuple[int, int]:
    total = rng.randint(4, 6)
    max_minority = min(2, total // 2)
    minority = rng.randint(1, max_minority)
    majority = total - minority
    return majority, minority


def _sample_specs(
    rng: random.Random,
    majority_color: int,
    minority_color: int,
    majority_count: int,
    minority_count: int,
) -> list[dict]:
    specs: list[dict] = []

    for _ in range(majority_count):
        shape = _random_variant(rng, MAJORITY_SHAPES)
        h, w = _shape_dims(shape)
        specs.append({"color": majority_color, "role": "majority", "shape": shape, "h": h, "w": w, "area": len(shape)})

    for _ in range(minority_count):
        for _ in range(50):
            shape = _random_variant(rng, MINORITY_SHAPES)
            if _is_nontrivial_minority_shape(shape):
                h, w = _shape_dims(shape)
                specs.append(
                    {"color": minority_color, "role": "minority", "shape": shape, "h": h, "w": w, "area": len(shape)}
                )
                break
        else:
            raise RuntimeError("failed to sample minority shape")

    rng.shuffle(specs)
    specs.sort(key=lambda spec: (spec["h"] * spec["w"], spec["area"]), reverse=True)
    return specs


def _place_specs(grid: np.ndarray, specs: list[dict], rng: random.Random) -> bool:
    reserved = np.zeros_like(grid, dtype=bool)
    h, w = grid.shape

    for spec in specs:
        shp = spec["shape"]
        sh = int(spec["h"])
        sw = int(spec["w"])
        if sh > h or sw > w:
            return False

        placed = False
        for _ in range(PLACE_ATTEMPTS):
            top = rng.randint(0, h - sh)
            left = rng.randint(0, w - sw)
            if not _can_place(reserved, top, left, sh, sw, pad=1):
                continue
            _paint_shape(grid, shp, top, left, int(spec["color"]))
            _reserve(reserved, top, left, sh, sw, pad=1)
            placed = True
            break

        if not placed:
            return False

    return True


def _minority_shapes_are_visible(comps: list[dict]) -> bool:
    for comp in comps:
        bbox = comp["bbox"]
        r0, c0, _, _ = bbox
        local = _normalize_shape([(r - r0, c - c0) for r, c in comp["coords"]])
        if not _is_nontrivial_minority_shape(local):
            return False
    return True


def _unique_nearest_majorities(majority_objs: list[dict], minority_objs: list[dict]) -> bool:
    for obj in minority_objs:
        dists = sorted(_center_dist2(obj["bbox"], maj["bbox"]) for maj in majority_objs)
        if len(dists) >= 2 and dists[0] == dists[1]:
            return False
    return True


def _valid_generated_grid(grid: np.ndarray) -> bool:
    try:
        bg = _get_bg(grid)
        counts = Counter(map(int, grid.flat))
        fg_colors = [color for color in counts if color != bg]
        if len(fg_colors) != 2:
            return False
        if counts[bg] <= max(counts[color] for color in fg_colors):
            return False

        comps = _find_components(grid, bg)
        by_color: dict[int, list[dict]] = defaultdict(list)
        for comp in comps:
            by_color[int(comp["color"])].append(comp)
        if len(by_color) != 2:
            return False

        ordered = sorted(by_color.items(), key=lambda item: len(item[1]), reverse=True)
        if len(ordered[0][1]) <= len(ordered[1][1]):
            return False

        majority_color, majority_objs = ordered[0]
        minority_color, minority_objs = ordered[1]
        if not _minority_shapes_are_visible(minority_objs):
            return False
        if not _unique_nearest_majorities(majority_objs, minority_objs):
            return False

        out = main(grid)
        if np.array_equal(out, grid):
            return False
        if np.any(out == minority_color):
            return False
        if np.all(out == bg):
            return False
        if np.count_nonzero(out == majority_color) <= np.count_nonzero(grid == majority_color):
            return False
        return True
    except Exception:
        return False


def generate_input() -> np.ndarray:
    rng = random.Random()

    for _ in range(GEN_ATTEMPTS):
        majority_count, minority_count = _sample_counts(rng)
        total = majority_count + minority_count
        min_side = 11 + total
        h = rng.randint(min_side, min_side + 5)
        w = rng.randint(min_side, min_side + 5)

        bg, majority_color, minority_color = rng.sample(range(10), 3)
        grid = np.full((h, w), bg, dtype=int)

        specs = _sample_specs(rng, majority_color, minority_color, majority_count, minority_count)
        if sum(spec["role"] == "majority" for spec in specs) != majority_count:
            continue
        if sum(spec["role"] == "minority" for spec in specs) != minority_count:
            continue

        if not _place_specs(grid, specs, rng):
            continue
        if _valid_generated_grid(grid):
            return grid

    raise RuntimeError("failed to generate a valid input")
