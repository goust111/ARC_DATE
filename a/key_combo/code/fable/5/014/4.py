from __future__ import annotations

import random

import numpy as np


Coord = tuple[int, int]
Shape = frozenset[Coord]


def _normalize(cells: set[Coord] | Shape) -> Shape:
    """Translate a nonempty coordinate set so its bounding-box origin is (0, 0)."""
    assert cells
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _shape_size(shape: Shape) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _components(mask: np.ndarray, connectivity: int = 8) -> list[set[Coord]]:
    """Return foreground components using row/column coordinates."""
    remaining = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    result: list[set[Coord]] = []
    if connectivity == 4:
        steps = ((-1, 0), (1, 0), (0, -1), (0, 1))
    else:
        steps = tuple(
            (dr, dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if dr != 0 or dc != 0
        )

    while remaining:
        seed = remaining.pop()
        comp = {seed}
        queue = [seed]
        while queue:
            r, c = queue.pop()
            for dr, dc in steps:
                nxt = (r + dr, c + dc)
                if nxt in remaining:
                    remaining.remove(nxt)
                    comp.add(nxt)
                    queue.append(nxt)
        result.append(comp)
    return result


def _edge_run(flags: np.ndarray, from_start: bool) -> int:
    seq = flags if from_start else flags[::-1]
    count = 0
    for value in seq:
        if not bool(value):
            break
        count += 1
    return count


def _split_regions(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """Find the all-nonblack checker strip and return it in its own coordinates."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    full_cols = np.all(g != 0, axis=0)
    full_rows = np.all(g != 0, axis=1)
    choices: list[tuple[str, int]] = []

    for side, run in (
        ("L", _edge_run(full_cols, True)),
        ("R", _edge_run(full_cols, False)),
        ("T", _edge_run(full_rows, True)),
        ("B", _edge_run(full_rows, False)),
    ):
        if run >= 3:
            choices.append((side, run))

    assert len(choices) == 1, "expected one uniquely identifiable checkerboard side"
    side, depth = choices[0]
    if side == "L":
        checker, candidates = g[:, :depth], g[:, depth:]
    elif side == "R":
        checker, candidates = g[:, -depth:], g[:, :-depth]
    elif side == "T":
        checker, candidates = g[:depth, :], g[depth:, :]
    else:
        checker, candidates = g[-depth:, :], g[:-depth, :]

    assert min(checker.shape) >= 3 and min(candidates.shape) >= 3
    return checker.copy(), candidates.copy(), side


def _infer_key(checker: np.ndarray) -> tuple[Shape, dict[Coord, int], tuple[int, int]]:
    """Infer checker phase, then return normalized defect shape and actual colors."""
    colors = [int(v) for v in np.unique(checker)]
    assert len(colors) == 2, "checkerboard must contain exactly two colors"
    rows, cols = np.indices(checker.shape)
    even = (rows + cols) % 2 == 0

    expected_a = np.where(even, colors[0], colors[1])
    expected_b = np.where(even, colors[1], colors[0])
    errors_a = int(np.count_nonzero(checker != expected_a))
    errors_b = int(np.count_nonzero(checker != expected_b))
    assert errors_a != errors_b, "checkerboard majority must determine a unique phase"
    expected = expected_a if errors_a < errors_b else expected_b
    defect_mask = checker != expected
    defect_coords = [tuple(map(int, rc)) for rc in np.argwhere(defect_mask)]
    assert 2 <= len(defect_coords) < checker.size // 2

    r0 = min(r for r, _ in defect_coords)
    c0 = min(c for _, c in defect_coords)
    key = frozenset((r - r0, c - c0) for r, c in defect_coords)
    actual_colors = {
        (r - r0, c - c0): int(checker[r, c]) for r, c in defect_coords
    }
    assert set(actual_colors.values()) == set(colors), "both violation classes must occur"
    return key, actual_colors, (r0, c0)


def main(grid: np.ndarray) -> np.ndarray:
    """Select the candidate matching the union of two checker-phase violation classes and stamp their colors onto it.

    Input structure:
    - One edge region is a complete two-color checkerboard containing a sparse set of phase-flipped cells.
    - The opposite region is a black canvas of separated monochrome candidate objects, including single-class decoys.

    Rule:
    - Infer the checker phase by majority, combine all flipped cells into one translation-normalized key shape, and find the unique candidate with exactly that shape.
    - Crop to the candidate region, erase every candidate, and recolor the matching object with each aligned defect cell's actual checker color.

    Key insight:
    - Neither violation class alone is the key: their union selects the object, while class membership supplies its two output colors.
    """
    checker, candidate, _side = _split_regions(np.asarray(grid, dtype=int))
    key, actual_colors, _origin = _infer_key(checker)

    objects = _components(candidate != 0, connectivity=8)
    assert len(objects) >= 3, "expected several separated candidate objects"
    matches: list[set[Coord]] = []
    for obj in objects:
        values = {int(candidate[r, c]) for r, c in obj}
        assert len(values) == 1, "each candidate object must be monochrome"
        if _normalize(obj) == key:
            matches.append(obj)
    assert len(matches) == 1, "the full two-class key must match exactly one candidate"

    target = matches[0]
    r0 = min(r for r, _ in target)
    c0 = min(c for _, c in target)
    out = np.zeros_like(candidate)
    for (r, c), color in actual_colors.items():
        assert (r0 + r, c0 + c) in target
        out[r0 + r, c0 + c] = color
    return out


# ------------------------------ generator ------------------------------

_PALETTE = tuple(range(1, 10))
_SIDES = ("L", "R", "T", "B")
_MAX_ATTEMPTS = 600
_SEEN_OUTPUTS: set[tuple[tuple[int, int], bytes]] = set()


def _is_connected(shape: Shape, connectivity: int) -> bool:
    h, w = _shape_size(shape)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in shape:
        mask[r, c] = True
    return len(_components(mask, connectivity=connectivity)) == 1


def _grow_polyomino(area: int) -> Shape:
    cells: set[Coord] = {(0, 0)}
    while len(cells) < area:
        frontier = {
            (r + dr, c + dc)
            for r, c in cells
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if (r + dr, c + dc) not in cells
        }
        cells.add(random.choice(tuple(frontier)))
    return _normalize(cells)


def _dihedral_variants(shape: Shape) -> list[Shape]:
    variants: set[Shape] = set()
    current = set(shape)
    for _ in range(4):
        variants.add(_normalize(current))
        variants.add(_normalize({(r, -c) for r, c in current}))
        current = {(c, -r) for r, c in current}
    return list(variants)


def _make_key() -> tuple[Shape, Shape, Shape, list[Shape]] | None:
    key = _grow_polyomino(random.randint(9, 14))
    h, w = _shape_size(key)
    if not (3 <= h <= 5 and 3 <= w <= 5):
        return None

    parity_parts = [frozenset(p for p in key if (p[0] + p[1]) % 2 == q) for q in (0, 1)]
    if min(map(len, parity_parts)) < 3:
        return None
    if not all(_is_connected(part, 8) for part in parity_parts):
        return None
    class_shapes = [_normalize(set(part)) for part in parity_parts]
    if class_shapes[0] == class_shapes[1]:
        return None

    variants = [
        variant
        for variant in _dihedral_variants(key)
        if variant != key and variant not in class_shapes
    ]
    if len(variants) < 2:
        return None
    random.shuffle(variants)
    return key, class_shapes[0], class_shapes[1], variants


def _rects_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    return (
        ar + ah + 1 <= br
        or br + bh + 1 <= ar
        or ac + aw + 1 <= bc
        or bc + bw + 1 <= ac
    )


def _pack_objects(
    shape_specs: list[tuple[str, Shape]], height: int, width: int, color: int
) -> tuple[np.ndarray, dict[str, list[tuple[int, int, Shape]]]] | None:
    canvas = np.zeros((height, width), dtype=int)
    placed_rects: list[tuple[int, int, int, int]] = []
    placements: dict[str, list[tuple[int, int, Shape]]] = {}
    order = shape_specs[:]
    random.shuffle(order)

    for label, shape in order:
        h, w = _shape_size(shape)
        positions = [
            (r, c)
            for r in range(1, height - h)
            for c in range(1, width - w)
        ]
        random.shuffle(positions)
        chosen: tuple[int, int] | None = None
        for r0, c0 in positions:
            rect = (r0, c0, h, w)
            if all(_rects_separated(rect, old) for old in placed_rects):
                chosen = (r0, c0)
                placed_rects.append(rect)
                break
        if chosen is None:
            return None
        r0, c0 = chosen
        for r, c in shape:
            canvas[r0 + r, c0 + c] = color
        placements.setdefault(label, []).append((r0, c0, shape))

    return canvas, placements


def _render_checker(
    shape: tuple[int, int], key: Shape, x: int, y: int
) -> tuple[np.ndarray, dict[Coord, int], dict[int, Shape]] | None:
    height, width = shape
    key_h, key_w = _shape_size(key)
    if key_h + 2 > height or key_w + 2 > width:
        return None
    r0 = random.randint(1, height - key_h - 1)
    c0 = random.randint(1, width - key_w - 1)
    phase = random.randrange(2)

    rows, cols = np.indices(shape)
    checker = np.where((rows + cols) % 2 == phase, x, y).astype(int)
    actual_map: dict[Coord, int] = {}
    by_actual: dict[int, set[Coord]] = {x: set(), y: set()}
    for r, c in key:
        rr, cc = r0 + r, c0 + c
        checker[rr, cc] = y if checker[rr, cc] == x else x
        actual = int(checker[rr, cc])
        actual_map[(r, c)] = actual
        by_actual[actual].add((r, c))

    if not by_actual[x] or not by_actual[y]:
        return None
    class_shapes = {color: _normalize(cells) for color, cells in by_actual.items()}
    return checker, actual_map, class_shapes


def _assemble(checker: np.ndarray, candidate: np.ndarray, side: str) -> np.ndarray:
    if side == "L":
        return np.concatenate((checker, candidate), axis=1)
    if side == "R":
        return np.concatenate((candidate, checker), axis=1)
    if side == "T":
        return np.concatenate((checker, candidate), axis=0)
    return np.concatenate((candidate, checker), axis=0)


def _semantic_valid(
    inp: np.ndarray,
    expected: np.ndarray,
    key: Shape,
    class_shapes: dict[int, Shape],
    target_top_left: tuple[int, int],
) -> bool:
    """Check visible semantics, uniqueness, anti-shortcuts, and the independent oracle."""
    try:
        checker, candidate, _ = _split_regions(inp)
        parsed_key, actual_map, _ = _infer_key(checker)
        if parsed_key != key or set(actual_map.values()) != set(class_shapes):
            return False

        objects = _components(candidate != 0, connectivity=8)
        normalized = [_normalize(obj) for obj in objects]
        if normalized.count(key) != 1:
            return False
        if any(normalized.count(shape) != 1 for shape in class_shapes.values()):
            return False

        # A one-class shortcut must select a decoy, never the full-key target.
        target_obj = objects[normalized.index(key)]
        target_anchor = (min(r for r, _ in target_obj), min(c for _, c in target_obj))
        if target_anchor != target_top_left:
            return False
        for shape in class_shapes.values():
            if shape == key:
                return False
            shortcut_anchors = [
                (min(r for r, _ in obj), min(c for _, c in obj))
                for obj, norm in zip(objects, normalized)
                if norm == shape
            ]
            if target_anchor in shortcut_anchors:
                return False

        # Both color classes must materially contribute pixels to the result.
        if any(np.count_nonzero(expected == color) == 0 for color in class_shapes):
            return False
        actual = main(inp)
        if not np.array_equal(actual, expected):
            return False
        if np.array_equal(actual, candidate):
            return False
        return max(inp.shape) <= 30 and max(actual.shape) <= 30
    except (AssertionError, ValueError, IndexError):
        return False


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    made = _make_key()
    if made is None:
        return None
    key, parity_a, parity_b, variants = made

    side = random.choice(_SIDES)
    candidate_h = random.randint(18, 20)
    candidate_w = random.randint(18, 20)
    checker_depth = random.randint(7, 10)
    checker_shape = (
        (candidate_h, checker_depth)
        if side in ("L", "R")
        else (checker_depth, candidate_w)
    )

    x, y, object_color = random.sample(_PALETTE, 3)
    rendered = _render_checker(checker_shape, key, x, y)
    if rendered is None:
        return None
    checker, actual_map, class_shapes = rendered
    if set(class_shapes.values()) != {parity_a, parity_b}:
        return None

    distractor_count = random.choice((1, 2))
    specs: list[tuple[str, Shape]] = [
        ("target", key),
        ("class_x", class_shapes[x]),
        ("class_y", class_shapes[y]),
    ]
    specs.extend((f"transform_{i}", variants[i]) for i in range(distractor_count))
    packed = _pack_objects(specs, candidate_h, candidate_w, object_color)
    if packed is None:
        return None
    candidate, placements = packed
    target_r, target_c, _ = placements["target"][0]

    expected = np.zeros_like(candidate)
    for (r, c), color in actual_map.items():
        expected[target_r + r, target_c + c] = color
    inp = _assemble(checker, candidate, side)
    if not _semantic_valid(
        inp, expected, key, class_shapes, (target_r, target_c)
    ):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    """Generate a diverse oracle-checked instance with mandatory anti-shortcut decoys."""
    for _ in range(_MAX_ATTEMPTS):
        made = _gen_attempt()
        if made is None:
            continue
        inp, out = made
        signature = (tuple(map(int, out.shape)), out.tobytes())
        if signature in _SEEN_OUTPUTS:
            continue
        _SEEN_OUTPUTS.add(signature)
        return inp
    raise RuntimeError(f"failed to generate a valid input after {_MAX_ATTEMPTS} attempts")
