from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BLACK = 0
TEAL = 8
PALETTE = (1, 2, 3, 4, 5, 6, 7, 9)
REFLECTION_KINDS = ("h", "v", "d", "a")
NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

Coord = tuple[int, int]
Shape = tuple[Coord, ...]
BBox = tuple[int, int, int, int]
Axis = tuple[str, int]


def normalize_coords(coords: np.ndarray | set[Coord] | list[Coord] | tuple[Coord, ...]) -> Shape:
    arr = np.asarray(list(coords), dtype=int)
    arr = arr - arr.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in arr))


def bbox_coords(coords: np.ndarray | set[Coord] | list[Coord] | tuple[Coord, ...]) -> BBox:
    arr = np.asarray(list(coords), dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def bbox_union(boxes: list[BBox] | tuple[BBox, ...]) -> BBox:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def crop_grid(grid: np.ndarray, box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def shape_size(shape: Shape) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def shift_shape(shape: Shape, dr: int, dc: int) -> set[Coord]:
    return {(dr + r, dc + c) for r, c in shape}


def reflect_shape(shape: Shape, kind: str) -> Shape:
    pts = np.asarray(shape, dtype=int)
    if kind == "h":
        out = np.column_stack((-pts[:, 0], pts[:, 1]))
    elif kind == "v":
        out = np.column_stack((pts[:, 0], -pts[:, 1]))
    elif kind == "d":
        out = np.column_stack((pts[:, 1], pts[:, 0]))
    elif kind == "a":
        out = np.column_stack((-pts[:, 1], -pts[:, 0]))
    else:
        raise ValueError(f"unknown reflection kind: {kind}")
    return normalize_coords(out)


def reflection_variants(shape: Shape) -> list[Shape]:
    variants: list[Shape] = [shape]
    seen = {shape}
    for kind in REFLECTION_KINDS:
        mirrored = reflect_shape(shape, kind)
        if mirrored not in seen:
            seen.add(mirrored)
            variants.append(mirrored)
    return variants


def canonical_shape_key(shape: Shape) -> Shape:
    return min(reflection_variants(shape))


def find_components(grid: np.ndarray, bg: int = BLACK) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS8:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": bbox_coords(arr),
                    "shape": normalize_coords(arr),
                }
            )
    return components


def transform_abs(coords: set[Coord], axis: Axis) -> set[Coord]:
    kind, param = axis
    if kind == "h":
        return {(2 * param - r, c) for r, c in coords}
    if kind == "v":
        return {(r, 2 * param - c) for r, c in coords}
    if kind == "d":
        return {(c + param, r - param) for r, c in coords}
    if kind == "a":
        return {(param - c, param - r) for r, c in coords}
    raise ValueError(f"unknown axis kind: {kind}")


def detect_translation(coords_a: set[Coord], coords_b: set[Coord]) -> tuple[int, int] | None:
    if len(coords_a) != len(coords_b):
        return None
    box_a = bbox_coords(coords_a)
    box_b = bbox_coords(coords_b)
    if (box_a[2] - box_a[0], box_a[3] - box_a[1]) != (box_b[2] - box_b[0], box_b[3] - box_b[1]):
        return None
    dr = box_b[0] - box_a[0]
    dc = box_b[1] - box_a[1]
    translated = {(r + dr, c + dc) for r, c in coords_a}
    if translated == coords_b:
        return int(dr), int(dc)
    return None


def _detect_horizontal(coords_a: set[Coord], coords_b: set[Coord]) -> Axis | None:
    ra0, ca0, ra1, ca1 = bbox_coords(coords_a)
    rb0, cb0, rb1, cb1 = bbox_coords(coords_b)
    if (ca0, ca1) != (cb0, cb1):
        return None
    if ra0 + rb1 != ra1 + rb0:
        return None
    axis_r = ra0 + rb1
    if axis_r % 2:
        return None
    axis = ("h", axis_r // 2)
    return axis if transform_abs(coords_a, axis) == coords_b else None


def _detect_vertical(coords_a: set[Coord], coords_b: set[Coord]) -> Axis | None:
    ra0, ca0, ra1, ca1 = bbox_coords(coords_a)
    rb0, cb0, rb1, cb1 = bbox_coords(coords_b)
    if (ra0, ra1) != (rb0, rb1):
        return None
    if ca0 + cb1 != ca1 + cb0:
        return None
    axis_c = ca0 + cb1
    if axis_c % 2:
        return None
    axis = ("v", axis_c // 2)
    return axis if transform_abs(coords_a, axis) == coords_b else None


def _detect_main_diagonal(coords_a: set[Coord], coords_b: set[Coord]) -> Axis | None:
    ra0, ca0, ra1, ca1 = bbox_coords(coords_a)
    rb0, cb0, rb1, cb1 = bbox_coords(coords_b)
    diffs = {rb0 - ca0, rb1 - ca1, ra0 - cb0, ra1 - cb1}
    if len(diffs) != 1:
        return None
    axis = ("d", next(iter(diffs)))
    return axis if transform_abs(coords_a, axis) == coords_b else None


def _detect_anti_diagonal(coords_a: set[Coord], coords_b: set[Coord]) -> Axis | None:
    ra0, ca0, ra1, ca1 = bbox_coords(coords_a)
    rb0, cb0, rb1, cb1 = bbox_coords(coords_b)
    sums = {rb0 + ca1, rb1 + ca0, cb0 + ra1, cb1 + ra0}
    if len(sums) != 1:
        return None
    axis = ("a", next(iter(sums)))
    return axis if transform_abs(coords_a, axis) == coords_b else None


def detect_mirror_axis(coords_a: set[Coord], coords_b: set[Coord]) -> Axis | None:
    candidates = []
    for detector in (_detect_horizontal, _detect_vertical, _detect_main_diagonal, _detect_anti_diagonal):
        axis = detector(coords_a, coords_b)
        if axis is not None:
            candidates.append(axis)
    if not candidates:
        return None
    assert len(candidates) == 1, f"ambiguous mirror axis: {candidates}"
    return candidates[0]


def axis_cells(box: BBox, axis: Axis) -> list[Coord]:
    r0, c0, r1, c1 = box
    kind, param = axis
    if kind == "h":
        return [(param, c) for c in range(c0, c1 + 1)]
    if kind == "v":
        return [(r, param) for r in range(r0, r1 + 1)]
    if kind == "d":
        return [(r, r - param) for r in range(r0, r1 + 1) if c0 <= r - param <= c1]
    if kind == "a":
        return [(r, param - r) for r in range(r0, r1 + 1) if c0 <= param - r <= c1]
    raise ValueError(f"unknown axis kind: {kind}")


def shift_axis(axis: Axis, dr: int, dc: int) -> Axis:
    kind, param = axis
    if kind == "h":
        return kind, param - dr
    if kind == "v":
        return kind, param - dc
    if kind == "d":
        return kind, param - dr + dc
    if kind == "a":
        return kind, param - dr - dc
    raise ValueError(f"unknown axis kind: {kind}")


def paint_cells(grid: np.ndarray, cells: set[Coord] | list[Coord], color: int) -> None:
    for r, c in cells:
        grid[int(r), int(c)] = int(color)


def render_pair_patch(pair: dict) -> tuple[np.ndarray, np.ndarray]:
    h = int(pair["height"])
    w = int(pair["width"])
    inp = np.zeros((h, w), dtype=int)
    paint_cells(inp, pair["coords_a"], pair["color_a"])
    paint_cells(inp, pair["coords_b"], pair["color_b"])
    out = inp.copy()
    axis = pair["axis"]
    if axis is not None:
        line = axis_cells((0, 0, h - 1, w - 1), axis)
        assert line
        assert not (set(line) & pair["coords_a"])
        assert not (set(line) & pair["coords_b"])
        paint_cells(out, line, TEAL)
    return inp, out


def analyze_pairs(grid: np.ndarray) -> list[dict]:
    components = find_components(grid, bg=BLACK)
    assert len(components) >= 2 and len(components) % 2 == 0, "expected an even number of objects"

    groups: dict[Shape, list[dict]] = defaultdict(list)
    for comp in components:
        groups[canonical_shape_key(comp["shape"])].append(comp)

    pairs: list[dict] = []
    for key, members in groups.items():
        assert len(members) == 2, f"shape class must contain exactly two objects: {key}"
        first, second = members
        coords_a = {tuple(map(int, coord)) for coord in first["coords"]}
        coords_b = {tuple(map(int, coord)) for coord in second["coords"]}
        translation = detect_translation(coords_a, coords_b)
        axis = detect_mirror_axis(coords_a, coords_b)
        assert (translation is None) != (axis is None), "each pair must be either a translation or a mirror"
        pairs.append(
            {
                "components": (first, second),
                "coords_a": coords_a,
                "coords_b": coords_b,
                "axis": axis,
                "is_translation": translation is not None,
            }
        )
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair sprite objects by normalized shape and draw the hidden axis only for mirrored pairs.

    Input structure:
    - The grid is a black background containing an even number of disconnected monochrome sprite objects.
    - Each sprite shape is non-self-symmetric, and colors are unreliable because unrelated objects may share a color.
    - Every shape-equivalence class under translation plus reflection contains exactly two objects.

    Rule:
    - Group objects by bbox-normalized shape, treating horizontal, vertical, and diagonal mirror variants as equivalent.
    - For each matched pair, decide whether the two objects are mere translations or true reflections across a horizontal, vertical, main-diagonal, or anti-diagonal hidden axis.
    - Preserve all original objects and draw a teal axis segment only for mirror pairs, restricted to the two-object union bbox.

    Key insight:
    - Color is a decoy; the only reliable pairing key is normalized geometry, and the midpoint alone is insufficient because pure translations must be rejected and the two diagonal directions must be distinguished.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "grid exceeds ARC size"
    assert TEAL not in set(map(int, g.flatten())), "teal is reserved for the hidden axes"

    out = g.copy()
    for pair in analyze_pairs(g):
        if pair["is_translation"]:
            continue
        first, second = pair["components"]
        box = bbox_union([first["bbox"], second["bbox"]])
        line = axis_cells(box, pair["axis"])
        assert line, "mirror axis must cross the pair bbox"
        for r, c in line:
            assert int(g[r, c]) == BLACK, "hidden axis must be absent from the input"
            out[r, c] = TEAL
    return out


def _random_connected_shape() -> Shape:
    for _ in range(400):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        max_area = min(6, h * w - 1)
        if max_area < 4:
            continue
        area = random.randint(4, max_area)
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in NEIGHBORS8:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            cells.add(random.choice(frontier))
        shape = normalize_coords(cells)
        sh, sw = shape_size(shape)
        if sh < 2 or sw < 2:
            continue
        if any(reflect_shape(shape, kind) == shape for kind in REFLECTION_KINDS):
            continue
        return shape
    raise RuntimeError("failed to generate a non-self-symmetric shape")


def _sample_axis(kind: str, h: int, w: int) -> Axis:
    if kind == "h":
        return kind, random.randint(1, h - 2)
    if kind == "v":
        return kind, random.randint(1, w - 2)
    if kind == "d":
        while True:
            axis = (kind, random.randint(-(w - 1), h - 1))
            if len(axis_cells((0, 0, h - 1, w - 1), axis)) >= 2:
                return axis
    if kind == "a":
        while True:
            axis = (kind, random.randint(0, h + w - 2))
            if len(axis_cells((0, 0, h - 1, w - 1), axis)) >= 2:
                return axis
    raise ValueError(f"unknown kind: {kind}")


def _in_bounds(coords: set[Coord], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in coords)


def _compact_pair(shape: Shape, coords_a: set[Coord], coords_b: set[Coord], axis: Axis | None) -> dict:
    box = bbox_union([bbox_coords(coords_a), bbox_coords(coords_b)])
    r0, c0, r1, c1 = box
    shifted_a = {(r - r0, c - c0) for r, c in coords_a}
    shifted_b = {(r - r0, c - c0) for r, c in coords_b}
    return {
        "shape": shape,
        "coords_a": shifted_a,
        "coords_b": shifted_b,
        "axis": None if axis is None else shift_axis(axis, r0, c0),
        "height": r1 - r0 + 1,
        "width": c1 - c0 + 1,
    }


def _make_mirror_pair(shape: Shape, kind: str) -> dict:
    sh, sw = shape_size(shape)
    for _ in range(600):
        h = random.randint(max(sh, sw) + 2, min(14, max(sh, sw) + 7))
        w = random.randint(max(sh, sw) + 2, min(14, max(sh, sw) + 7))
        axis = _sample_axis(kind, h, w)
        r0 = random.randint(0, h - sh)
        c0 = random.randint(0, w - sw)
        coords_a = shift_shape(shape, r0, c0)
        coords_b = transform_abs(coords_a, axis)
        if not _in_bounds(coords_b, h, w):
            continue
        if coords_a & coords_b:
            continue
        union_box = bbox_union([bbox_coords(coords_a), bbox_coords(coords_b)])
        line = set(axis_cells(union_box, axis))
        if not line or line & coords_a or line & coords_b:
            continue
        if detect_translation(coords_a, coords_b) is not None:
            continue
        if detect_mirror_axis(coords_a, coords_b) != axis:
            continue
        return _compact_pair(shape, coords_a, coords_b, axis)
    raise RuntimeError(f"failed to place mirror pair for axis {kind}")


def _make_translation_pair(shape: Shape) -> dict:
    sh, sw = shape_size(shape)
    for _ in range(600):
        h = random.randint(sh + 1, min(14, sh + 7))
        w = random.randint(sw + 1, min(14, sw + 7))
        r1 = random.randint(0, h - sh)
        c1 = random.randint(0, w - sw)
        r2 = random.randint(0, h - sh)
        c2 = random.randint(0, w - sw)
        if (r1, c1) == (r2, c2):
            continue
        coords_a = shift_shape(shape, r1, c1)
        coords_b = shift_shape(shape, r2, c2)
        if coords_a & coords_b:
            continue
        if detect_translation(coords_a, coords_b) is None:
            continue
        if detect_mirror_axis(coords_a, coords_b) is not None:
            continue
        return _compact_pair(shape, coords_a, coords_b, None)
    raise RuntimeError("failed to place translation pair")


def _assign_colors(pairs: list[dict]) -> None:
    trap_color = random.choice(PALETTE)
    for pair in pairs:
        pair["color_a"] = random.choice(PALETTE)
        pair["color_b"] = random.choice(PALETTE)
    if len(pairs) >= 2:
        pairs[0]["color_a"] = trap_color
        pairs[1]["color_a"] = trap_color
    if all(pair["color_a"] == pair["color_b"] for pair in pairs):
        pairs[random.randrange(len(pairs))]["color_b"] = random.choice(PALETTE)


def _rectangles_overlap(a: BBox, b: BBox, padding: int) -> bool:
    return not (
        a[2] + padding < b[0]
        or b[2] + padding < a[0]
        or a[3] + padding < b[1]
        or b[3] + padding < a[1]
    )


def _place_patches(pairs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    patch_data = []
    for pair in pairs:
        inp_patch, out_patch = render_pair_patch(pair)
        patch_data.append((pair, inp_patch, out_patch))
    patch_data.sort(key=lambda item: item[1].shape[0] * item[1].shape[1], reverse=True)

    total_h = sum(patch.shape[0] for _, patch, _ in patch_data)
    total_w = sum(patch.shape[1] for _, patch, _ in patch_data)
    min_side = max(max(patch.shape) for _, patch, _ in patch_data) + 2

    for _ in range(400):
        h = random.randint(max(min_side, min(18, total_h // 2 + 6)), 30)
        w = random.randint(max(min_side, min(18, total_w // 2 + 6)), 30)
        inp = np.zeros((h, w), dtype=int)
        out = np.zeros((h, w), dtype=int)
        placed_boxes: list[BBox] = []
        success = True
        for _pair, inp_patch, out_patch in patch_data:
            ph, pw = inp_patch.shape
            candidates = []
            for r0 in range(h - ph + 1):
                for c0 in range(w - pw + 1):
                    box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
                    if all(not _rectangles_overlap(box, prev, padding=2) for prev in placed_boxes):
                        candidates.append((r0, c0, box))
            if not candidates:
                success = False
                break
            r0, c0, box = random.choice(candidates)
            inp[r0 : r0 + ph, c0 : c0 + pw] = np.maximum(inp[r0 : r0 + ph, c0 : c0 + pw], inp_patch)
            out[r0 : r0 + ph, c0 : c0 + pw] = np.maximum(out[r0 : r0 + ph, c0 : c0 + pw], out_patch)
            placed_boxes.append(box)
        if not success:
            continue

        occupied = np.argwhere(inp != BLACK)
        assert occupied.size > 0
        r0, c0 = occupied.min(axis=0)
        r1, c1 = occupied.max(axis=0)
        top = random.randint(0, min(2, int(r0)))
        left = random.randint(0, min(2, int(c0)))
        bottom = random.randint(0, min(2, h - 1 - int(r1)))
        right = random.randint(0, min(2, w - 1 - int(c1)))
        crop_box = (int(r0) - top, int(c0) - left, int(r1) + bottom, int(c1) + right)
        return crop_grid(inp, crop_box), crop_grid(out, crop_box)
    raise RuntimeError("failed to place pair patches on the global canvas")


def _count_relations(pairs: list[dict]) -> tuple[int, int]:
    mirror_count = sum(1 for pair in pairs if pair["axis"] is not None)
    translation_count = len(pairs) - mirror_count
    return mirror_count, translation_count


def _generate_case() -> tuple[np.ndarray, np.ndarray, dict]:
    for _ in range(300):
        pair_count = random.randint(2, 3)
        mirror_kinds = random.sample(REFLECTION_KINDS, k=pair_count - 1)
        relation_kinds = ["t", *mirror_kinds]
        random.shuffle(relation_kinds)

        used_keys: set[Shape] = set()
        pairs: list[dict] = []
        for relation in relation_kinds:
            for _shape_try in range(200):
                shape = _random_connected_shape()
                key = canonical_shape_key(shape)
                if key in used_keys:
                    continue
                used_keys.add(key)
                if relation == "t":
                    pair = _make_translation_pair(shape)
                else:
                    pair = _make_mirror_pair(shape, relation)
                pair["relation"] = relation
                pairs.append(pair)
                break
            else:
                break
        if len(pairs) != len(relation_kinds):
            continue

        _assign_colors(pairs)
        inp, expected = _place_patches(pairs)
        mirror_count, translation_count = _count_relations(pairs)
        if mirror_count < 1 or translation_count < 1:
            continue
        if np.array_equal(inp, expected):
            continue
        if _oracle_accepts(inp, expected):
            return inp, expected, {"pairs": pairs}
    raise RuntimeError("failed to generate a valid puzzle instance")


def _oracle_accepts(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False
    analyzed = analyze_pairs(inp)
    mirror_count = sum(1 for pair in analyzed if not pair["is_translation"])
    translation_count = sum(1 for pair in analyzed if pair["is_translation"])
    if mirror_count < 1 or translation_count < 1:
        return False
    if len({canonical_shape_key(comp["shape"]) for comp in find_components(inp)}) * 2 != len(find_components(inp)):
        return False
    return True


def generate_input() -> np.ndarray:
    return _generate_case()[0]
