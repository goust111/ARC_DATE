import hashlib
from collections import Counter

import numpy as np


BG = 0
TEMPLATE_COLOR = 5
SOURCE_COLORS = (1, 2, 3, 7)
RNG = np.random.default_rng()
MAX_GENERATION_ATTEMPTS = 500

# Keep examples diverse during a single module lifetime.
_SEEN_INPUT_SIGNATURES: set[str] = set()
_SEEN_OUTPUT_SIGNATURES: set[str] = set()


# Normalized 4-connected polyominoes, all fitting inside a 3x3 box.
SHAPES_BY_AREA = {
    3: [
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (1, 1)),
        ((0, 1), (1, 0), (1, 1)),
    ],
    4: [
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 1)),
        ((0, 1), (0, 2), (1, 0), (1, 1)),
    ],
    5: [
        ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
        ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
        ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
        ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)),
    ],
    6: [
        ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
        ((0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (2, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 1), (2, 0), (2, 1)),
        ((0, 0), (0, 1), (1, 1), (1, 2), (2, 1), (2, 2)),
    ],
}

# Count patterns deliberately include at least one duplicated area so pairing by
# pixel count alone is insufficient.
AREA_PATTERNS = {
    3: [(3, 3, 5), (3, 4, 4), (3, 5, 5), (4, 4, 6), (3, 3, 6)],
    4: [(3, 3, 4, 4), (3, 3, 4, 5), (3, 4, 4, 6), (3, 5, 5, 6), (3, 4, 5, 5)],
}


def _sig(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize_shape(coords) -> tuple[tuple[int, int], ...]:
    arr = np.asarray(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    norm = [(int(r - r0), int(c - c0)) for r, c in arr]
    return tuple(sorted(norm))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rs = [r for r, _ in shape]
    cs = [c for _, c in shape]
    return max(rs) + 1, max(cs) + 1


def _shape_to_array(shape: tuple[tuple[int, int], ...]) -> np.ndarray:
    h, w = _shape_dims(shape)
    arr = np.zeros((h, w), dtype=int)
    for r, c in shape:
        arr[r, c] = 1
    return arr


def _array_to_shape(arr: np.ndarray) -> tuple[tuple[int, int], ...]:
    return _normalize_shape(np.argwhere(arr != 0))


def _dihedral_variants(shape: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    arr = _shape_to_array(shape)
    variants = set()
    for k in range(4):
        rot = np.rot90(arr, k)
        variants.add(_array_to_shape(rot))
        variants.add(_array_to_shape(np.fliplr(rot)))
    return sorted(variants)


def _paint_shape(grid: np.ndarray, shape: tuple[tuple[int, int], ...], top: int, left: int, color: int) -> None:
    for dr, dc in shape:
        grid[top + dr, left + dc] = int(color)


def _extract_templates(grid: np.ndarray) -> list[dict]:
    comps = _components_from_mask(grid == TEMPLATE_COLOR)
    templates = []
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        hole_mask = patch == BG
        hole_comps = _components_from_mask(hole_mask)
        assert len(hole_comps) == 1, "each template must contain exactly one hole"
        hole = hole_comps[0]
        hr0, hc0, hr1, hc1 = _bbox(hole)
        assert hr0 > 0 and hc0 > 0 and hr1 < patch.shape[0] - 1 and hc1 < patch.shape[1] - 1
        templates.append(
            {
                "bbox": (r0, c0, r1, c1),
                "top": r0,
                "shape": _normalize_shape(hole),
                "area": int(hole.shape[0]),
            }
        )
    return sorted(templates, key=lambda item: item["top"])


def _extract_sources(grid: np.ndarray) -> list[dict]:
    colors = sorted(int(v) for v in np.unique(grid) if int(v) not in (BG, TEMPLATE_COLOR))
    sources = []
    for color in colors:
        comps = _components_from_mask(grid == color)
        assert len(comps) == 1, "each source color must form one component"
        comp = comps[0]
        r0, c0, r1, c1 = _bbox(comp)
        sources.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "top": r0,
                "shape": _normalize_shape(comp),
                "area": int(comp.shape[0]),
            }
        )
    return sources


def _build_nested(layers: list[tuple[int, int]]) -> np.ndarray:
    color0, side0 = layers[0]
    out = np.full((side0, side0), int(color0), dtype=int)
    for color, thickness in layers[1:]:
        size = int(out.shape[0] + 2 * thickness)
        nxt = np.full((size, size), int(color), dtype=int)
        nxt[thickness : thickness + out.shape[0], thickness : thickness + out.shape[1]] = out
        out = nxt
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match source shapes to complementary template holes, then use the template order and source sizes to build nested square layers.

    Input structure:
    - The background is black.
    - The left side contains 3-4 gray filled rectangles stacked top-to-bottom, each with exactly one black 4-connected hole shape strictly inside it.
    - The right side contains the same number of colored 4-connected source shapes, one component per color.

    Rule:
    - Extract each template hole as the black cells inside that gray rectangle's bbox, normalized to its own hole bbox.
    - Extract each colored source shape, normalized to its own bbox, and pair it with the unique template hole of exactly the same normalized shape.
    - Read templates from top to bottom; the matched source for the top template becomes the innermost layer, and later templates become progressively outer layers.
    - The matched source area gives the layer size: the first source area is the side length of the center solid square, and each later source area is the thickness of a square frame in that source color.

    Key insight:
    - Shape equality determines which color belongs to which nesting layer, while pixel count determines how large that assigned layer is.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    templates = _extract_templates(g)
    sources = _extract_sources(g)
    assert 3 <= len(templates) <= 4
    assert len(templates) == len(sources)

    by_shape = {}
    for src in sources:
        assert src["shape"] not in by_shape, "source shapes must be unique after normalization"
        by_shape[src["shape"]] = src

    layers = []
    for tpl in templates:
        assert tpl["shape"] in by_shape, "every template hole must have a source match"
        src = by_shape[tpl["shape"]]
        layers.append((int(src["color"]), int(src["area"])))

    return _build_nested(layers)


def _choose_shapes(areas: tuple[int, ...]) -> list[dict]:
    records = []
    next_id = 0
    for area, count in sorted(Counter(areas).items()):
        candidates = sorted({variant for base_shape in SHAPES_BY_AREA[area] for variant in _dihedral_variants(base_shape)})
        picks = RNG.choice(len(candidates), size=count, replace=False)
        for pick in picks:
            variant = candidates[int(pick)]
            records.append(
                {
                    "id": next_id,
                    "area": area,
                    "shape": variant,
                    "shape_dims": _shape_dims(variant),
                }
            )
            next_id += 1
    RNG.shuffle(records)
    colors = list(RNG.choice(SOURCE_COLORS, size=len(records), replace=False))
    RNG.shuffle(colors)
    for record, color in zip(records, colors):
        record["color"] = int(color)
    return records


def _place_templates(grid: np.ndarray, template_records: list[dict], left: int, top: int, box_size: int, gap: int) -> None:
    for i, record in enumerate(template_records):
        r0 = top + i * (box_size + gap)
        c0 = left
        grid[r0 : r0 + box_size, c0 : c0 + box_size] = TEMPLATE_COLOR

        shape = record["shape"]
        sh, sw = record["shape_dims"]
        hole_r = int(RNG.integers(1, box_size - sh))
        hole_c = int(RNG.integers(1, box_size - sw))
        for dr, dc in shape:
            grid[r0 + hole_r + dr, c0 + hole_c + dc] = BG


def _top_rows_for_order(records_in_order: list[dict], height: int) -> list[int]:
    min_total = sum(record["shape_dims"][0] for record in records_in_order) + (len(records_in_order) - 1)
    slack = height - min_total
    assert slack >= 0
    extras = RNG.multinomial(slack, np.full(len(records_in_order) + 1, 1.0 / (len(records_in_order) + 1)))
    rows = []
    cur = int(extras[0])
    for i, record in enumerate(records_in_order):
        rows.append(cur)
        cur += int(record["shape_dims"][0])
        if i < len(records_in_order) - 1:
            cur += 1 + int(extras[i + 1])
    return rows


def _build_expected_output(template_records: list[dict]) -> np.ndarray | None:
    layers = [(int(record["color"]), int(record["area"])) for record in template_records]
    out = _build_nested(layers)
    if max(out.shape) > 30:
        return None
    return out


def _validate_semantics(grid: np.ndarray, expected: np.ndarray, template_order_ids: list[int], source_order_ids: list[int]) -> bool:
    templates = _extract_templates(grid)
    sources = _extract_sources(grid)
    if len(templates) != len(source_order_ids) or len(sources) != len(source_order_ids):
        return False

    area_counts = Counter(src["area"] for src in sources)
    if max(area_counts.values()) < 2:
        return False

    dup_areas = [area for area, count in area_counts.items() if count >= 2]
    if not dup_areas:
        return False
    for area in dup_areas:
        dup_shapes = {src["shape"] for src in sources if src["area"] == area}
        if len(dup_shapes) < area_counts[area]:
            return False

    if template_order_ids == source_order_ids:
        return False

    if len({src["shape"] for src in sources}) != len(sources):
        return False

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    count = int(RNG.choice([3, 4]))
    areas = AREA_PATTERNS[count][int(RNG.integers(0, len(AREA_PATTERNS[count])))]
    records = _choose_shapes(areas)

    template_order_ids = list(RNG.permutation([record["id"] for record in records]))
    template_records = [next(record for record in records if record["id"] == rid) for rid in template_order_ids]
    expected = _build_expected_output(template_records)
    if expected is None:
        return None

    box_size = int(RNG.choice([5, 6]))
    gap = 1
    left = int(RNG.choice([1, 2]))
    top = 1
    height = top + count * box_size + (count - 1) * gap + 1
    if height > 30:
        return None

    right_gap = int(RNG.choice([2, 3, 4]))
    right_width = int(RNG.choice([5, 6, 7]))
    right_margin = int(RNG.choice([1, 2]))
    width = left + box_size + right_gap + right_width + right_margin
    if width > 30:
        return None

    grid = np.zeros((height, width), dtype=int)
    _place_templates(grid, template_records, left=left, top=top, box_size=box_size, gap=gap)

    while True:
        source_order_ids = list(RNG.permutation([record["id"] for record in records]))
        if source_order_ids != template_order_ids:
            break

    right_left = left + box_size + right_gap
    ordered_sources = [next(record for record in records if record["id"] == rid) for rid in source_order_ids]
    source_rows = _top_rows_for_order(ordered_sources, height)

    for row, record in zip(source_rows, ordered_sources):
        sh, sw = record["shape_dims"]
        max_left = right_left + right_width - sw
        if max_left < right_left:
            return None
        col = int(RNG.integers(right_left, max_left + 1))
        _paint_shape(grid, record["shape"], row, col, record["color"])

    if np.any(grid[:, right_left:] == TEMPLATE_COLOR):
        return None

    input_sig = _sig(grid)
    output_sig = _sig(expected)
    if input_sig in _SEEN_INPUT_SIGNATURES or output_sig in _SEEN_OUTPUT_SIGNATURES:
        return None

    sources = _extract_sources(grid)
    shape_to_id = {record["shape"]: record["id"] for record in records}
    source_order_ids_actual = [shape_to_id[src["shape"]] for src in sorted(sources, key=lambda item: item["top"])]
    if source_order_ids_actual == template_order_ids:
        return None

    if not _validate_semantics(grid, expected, template_order_ids, source_order_ids_actual):
        return None

    _SEEN_INPUT_SIGNATURES.add(input_sig)
    _SEEN_OUTPUT_SIGNATURES.add(output_sig)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
