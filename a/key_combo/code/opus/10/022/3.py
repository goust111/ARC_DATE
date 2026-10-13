import random
from typing import Dict, List, Sequence, Tuple

import numpy as np


BLACK = 0
GREY = 5
OBJECT_COLORS = (1, 2, 3, 4)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


Coord = Tuple[int, int]
Shape = Tuple[Coord, ...]


SHAPES: List[Shape] = [
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2)),
    ((0, 0), (1, 0), (1, 1), (1, 2), (2, 2)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
]


def _bbox(coords: Sequence[Coord]) -> Tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize_shape(coords: Sequence[Coord]) -> Shape:
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _shape_size(shape: Shape) -> Tuple[int, int]:
    r0, c0, r1, c1 = _bbox(shape)
    return r1 - r0 + 1, c1 - c0 + 1


def _connected_components(mask: np.ndarray) -> List[List[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: List[List[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: List[Coord] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(sorted(comp))
    return comps


def _paint_shape(grid: np.ndarray, top: int, left: int, shape: Shape, color: int) -> None:
    for dr, dc in shape:
        grid[top + dr, left + dc] = color


def _extract_objects(grid: np.ndarray) -> List[Dict]:
    mask = (grid != BLACK) & (grid != GREY)
    objects: List[Dict] = []
    for coords in _connected_components(mask):
        colors = {int(grid[r, c]) for r, c in coords}
        assert len(colors) == 1, "each falling object must be monochrome"
        r0, c0, r1, c1 = _bbox(coords)
        objects.append(
            {
                "coords": coords,
                "color": colors.pop(),
                "shape": _normalize_shape(coords),
                "area": len(coords),
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "height": r1 - r0 + 1,
                "width": c1 - c0 + 1,
            }
        )
    return sorted(objects, key=lambda obj: (obj["r0"], obj["c0"]))


def _extract_structures(grid: np.ndarray) -> List[Dict]:
    mask = grid == GREY
    structures: List[Dict] = []
    for coords in _connected_components(mask):
        platform_row = min(r for r, _ in coords)
        platform_cols = sorted(c for r, c in coords if r == platform_row)
        assert platform_cols, "platform row missing"
        assert platform_cols == list(range(platform_cols[0], platform_cols[-1] + 1)), "platform must be contiguous"
        key_coords = [(r, c) for r, c in coords if r > platform_row]
        assert key_coords, "every platform needs a key pattern"
        kr0, kc0, kr1, kc1 = _bbox(key_coords)
        structures.append(
            {
                "coords": coords,
                "platform_row": platform_row,
                "c0": platform_cols[0],
                "c1": platform_cols[-1],
                "width": len(platform_cols),
                "key_coords": key_coords,
                "key_shape": _normalize_shape(key_coords),
                "key_area": len(key_coords),
                "key_bbox": (kr0, kc0, kr1, kc1),
            }
        )
    return sorted(structures, key=lambda item: (item["platform_row"], item["c0"]))


def _assigned_structure(obj: Dict, structures: Sequence[Dict]) -> Dict:
    overlaps = [
        s
        for s in structures
        if s["c0"] <= obj["c1"] and obj["c0"] <= s["c1"] and s["platform_row"] > obj["r1"]
    ]
    assert overlaps, "each object must have a platform below its horizontal range"
    overlaps.sort(key=lambda s: s["platform_row"])
    return overlaps[0]


def _landed_coords(obj: Dict, structure: Dict) -> List[Coord]:
    delta_r = structure["platform_row"] - obj["r1"] - 1
    return [(r + delta_r, c) for r, c in obj["coords"]]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each colored object onto its aligned gray platform and color only the matching key below it.

    Input structure:
    - Black is background. The top rows contain several small monochrome 4-connected objects.
    - The middle/lower area contains separate gray structures; each structure is a horizontal platform with a gray key pattern attached directly below it.
    - Every object's horizontal span overlaps exactly one platform span, which determines where that object falls.

    Rule:
    - Move each colored object straight downward without changing its columns until its bbox bottom is one row above the first overlapping gray platform.
    - Compare the object's occupied cells, normalized inside its own bbox, with the attached key pattern's occupied cells, normalized inside the key bbox.
    - If the shapes match exactly, recolor the key cells with the object's color and erase the landed object.
    - If the shapes do not match, keep the gray structure unchanged and leave the landed object resting on the platform.

    Key insight:
    Gravity chooses which platform-key pair an object interacts with; only after that fixed pairing does exact shape equality decide whether color transfers.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    objects = _extract_objects(g)
    structures = _extract_structures(g)

    for obj in objects:
        for r, c in obj["coords"]:
            out[r, c] = BLACK

    for obj in objects:
        structure = _assigned_structure(obj, structures)
        landed = _landed_coords(obj, structure)
        if obj["shape"] == structure["key_shape"]:
            for r, c in structure["key_coords"]:
                out[r, c] = obj["color"]
        else:
            for r, c in landed:
                out[r, c] = obj["color"]
    return out


def _wrong_area_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    objects = _extract_objects(g)
    structures = _extract_structures(g)
    for obj in objects:
        for r, c in obj["coords"]:
            out[r, c] = BLACK
    for obj in objects:
        structure = _assigned_structure(obj, structures)
        landed = _landed_coords(obj, structure)
        if obj["area"] == structure["key_area"]:
            for r, c in structure["key_coords"]:
                out[r, c] = obj["color"]
        else:
            for r, c in landed:
                out[r, c] = obj["color"]
    return out


def _structure_events(grid: np.ndarray) -> List[Dict]:
    objects = _extract_objects(grid)
    structures = _extract_structures(grid)
    events: List[Dict] = []
    used_platforms = set()
    for obj in objects:
        structure = _assigned_structure(obj, structures)
        sid = (structure["platform_row"], structure["c0"], structure["c1"])
        assert sid not in used_platforms, "different objects must land on different platforms"
        used_platforms.add(sid)
        matched = obj["shape"] == structure["key_shape"]
        cross_match = any(other["shape"] == structure["key_shape"] for other in objects if other is not obj)
        events.append(
            {
                "object": obj,
                "structure": structure,
                "matched": matched,
                "cross_match": cross_match,
                "landed": _landed_coords(obj, structure),
            }
        )
    return events


def _validate_case(grid: np.ndarray) -> None:
    objects = _extract_objects(grid)
    structures = _extract_structures(grid)
    assert 3 <= len(objects) <= 4
    assert len(structures) >= len(objects)
    assert len(structures) <= 4

    for obj in objects:
        assert 3 <= obj["area"] <= 5
        assert obj["r0"] <= 2
    for structure in structures:
        assert 5 <= structure["width"] <= 7
        assert 3 <= structure["key_area"] <= 5

    events = _structure_events(grid)
    matched = [event for event in events if event["matched"]]
    mismatched = [event for event in events if not event["matched"]]
    assert matched, "must include at least one successful transfer"
    assert mismatched, "must include at least one failed match"
    assert any(
        event["object"]["area"] == event["structure"]["key_area"] for event in mismatched
    ), "need a same-area mismatch to rule out area shortcuts"
    assert any(event["cross_match"] for event in mismatched), "need a non-local shape distractor"

    out = main(grid)
    for event in events:
        obj = event["object"]
        structure = event["structure"]
        landed = set(event["landed"])
        for r, c in obj["coords"]:
            assert out[r, c] == BLACK, "original object must move away"
        if event["matched"]:
            for r, c in structure["key_coords"]:
                assert out[r, c] == obj["color"], "matching key must inherit object color"
            for r, c in landed:
                assert out[r, c] != obj["color"], "matched object must be absorbed"
        else:
            for r, c in structure["key_coords"]:
                assert out[r, c] == GREY, "mismatched key must stay gray"
            for r, c in landed:
                assert out[r, c] == obj["color"], "mismatched object must remain on the platform"

    assert not np.array_equal(out, _wrong_area_output(grid)), "generator must defeat area-only shortcut"


SHAPES_BY_AREA: Dict[int, List[Shape]] = {}
for _shape in SHAPES:
    SHAPES_BY_AREA.setdefault(len(_shape), []).append(_shape)


def _pick_object_shapes(count: int) -> Tuple[List[Shape], int, int]:
    pair_area = random.choice([area for area, shapes in SHAPES_BY_AREA.items() if len(shapes) >= 2])
    first, second = random.sample(SHAPES_BY_AREA[pair_area], 2)
    chosen = [first, second]
    pool = [shape for shape in SHAPES if shape not in chosen]
    random.shuffle(pool)
    while len(chosen) < count:
        chosen.append(pool.pop())
    random.shuffle(chosen)
    i = chosen.index(first)
    j = chosen.index(second)
    return chosen, i, j


def _build_platform_layout(count: int) -> Tuple[int, List[Tuple[int, int]]]:
    while True:
        widths = [random.randint(5, 6 if count == 4 else 7) for _ in range(count)]
        gaps = [random.randint(1, 2 if count == 4 else 3) for _ in range(count - 1)]
        total = sum(widths) + sum(gaps)
        if total <= 30:
            break
    width = random.randint(total, min(30, total + 3))
    start = random.randint(0, width - total)
    spans: List[Tuple[int, int]] = []
    c = start
    for idx, span_width in enumerate(widths):
        spans.append((c, c + span_width - 1))
        c += span_width
        if idx < len(gaps):
            c += gaps[idx]
    return width, spans


def _generate_once() -> np.ndarray:
    object_count = random.choice([3, 4])
    platform_count = object_count + int(object_count == 3 and random.random() < 0.5)
    width, spans = _build_platform_layout(platform_count)
    height = random.randint(14, 20)
    platform_rows = sorted(random.sample(range(7, height - 3), platform_count))

    object_shapes, mismatch_idx, match_source_idx = _pick_object_shapes(object_count)
    key_shapes: List[Shape] = []
    for idx, shape in enumerate(object_shapes):
        if idx == mismatch_idx:
            key_shapes.append(object_shapes[match_source_idx])
        else:
            key_shapes.append(shape)
    if platform_count > object_count:
        key_shapes.append(random.choice(object_shapes))

    grid = np.zeros((height, width), dtype=int)

    for idx, ((c0, c1), row, key_shape) in enumerate(zip(spans, platform_rows, key_shapes)):
        grid[row, c0 : c1 + 1] = GREY
        kh, kw = _shape_size(key_shape)
        key_left = random.randint(c0, c1 - kw + 1)
        _paint_shape(grid, row + 1, key_left, key_shape, GREY)

    colors = random.sample(list(OBJECT_COLORS), object_count)
    for idx, shape in enumerate(object_shapes):
        c0, c1 = spans[idx]
        oh, ow = _shape_size(shape)
        obj_left = random.randint(c0, c1 - ow + 1)
        obj_top = random.randint(0, 3 - oh)
        _paint_shape(grid, obj_top, obj_left, shape, colors[idx])

    _validate_case(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        try:
            return _generate_once()
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid gravity-key matching input")
