from __future__ import annotations

import random

import numpy as np


BG = 8
MAROON = 9
OBJECT_COLORS = (0, 1, 2, 3, 4, 5, 6, 7)
OPS = ("hflip", "vflip", "rot90cw")

MIN_OBJECTS = 3
MAX_OBJECTS = 5
BODY_HEIGHT_RANGE = (11, 24)
GRID_WIDTH_RANGE = (18, 28)
SHAPE_BOX_RANGE = (2, 5)
SHAPE_AREA_RANGE = (4, 8)
GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 200
PLACEMENT_ATTEMPTS = 300


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _signal_op_from_col(col: int, width: int) -> str:
    if 3 * col < width:
        return "hflip"
    if 3 * col < 2 * width:
        return "vflip"
    return "rot90cw"


def _transform_points(points: set[tuple[int, int]], op: str) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in points:
        if op == "hflip":
            out.add((r, -c))
        elif op == "vflip":
            out.add((-r, c))
        elif op == "rot90cw":
            out.add((c, -r))
        else:
            raise AssertionError(op)
    return out


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the top-row signal pixels to flip or rotate each same-colored object around its maroon anchor.

    Input structure:
    - bg = azure background.
    - The top row contains isolated colored signal pixels, one per object color, separated by bg.
    - Rows below contain one 4-connected object per signal color; each object is monochrome except that exactly one of its cells is replaced by a maroon anchor pixel.

    Rule:
    - Read each signal pixel's horizontal third: left third -> horizontal left-right mirror, middle third -> vertical up-down mirror, right third -> 90 degree clockwise rotation.
    - Match each signal color to the body object with the same color and apply that transform around the object's maroon anchor, keeping the anchor cell fixed.
    - Recolor the transformed anchor back to the object's own color and remove the signal row from the output.

    Key insight:
    - The transform type is encoded by where the color appears on the border row, while the transform center is encoded independently by the maroon pixel inside the matching object.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2
    height, width = g.shape

    signal_row = g[0]
    body = g[1:]
    body_h, body_w = body.shape

    ops_by_color: dict[int, str] = {}
    for c, value in enumerate(signal_row.tolist()):
        color = int(value)
        if color == BG:
            continue
        assert color != MAROON
        assert color not in ops_by_color
        ops_by_color[color] = _signal_op_from_col(c, width)
    assert ops_by_color

    components = _connected_components(body != BG)
    objects: dict[int, dict[str, object]] = {}

    for coords in components:
        values = body[coords[:, 0], coords[:, 1]]
        anchor_mask = values == MAROON
        anchor_coords = coords[anchor_mask]
        colors = {int(v) for v in values.tolist() if int(v) != MAROON}

        assert len(anchor_coords) == 1
        assert len(colors) == 1

        color = colors.pop()
        assert color in ops_by_color
        assert color not in objects

        anchor_r, anchor_c = map(int, anchor_coords[0])
        rel_points = {
            (int(r) - anchor_r, int(c) - anchor_c)
            for r, c in coords.tolist()
        }
        objects[color] = {
            "anchor": (anchor_r, anchor_c),
            "points": rel_points,
        }

    assert set(objects) == set(ops_by_color)

    out = np.full((body_h, body_w), BG, dtype=int)
    for color, obj in objects.items():
        anchor_r, anchor_c = obj["anchor"]  # type: ignore[index]
        rel_points = obj["points"]  # type: ignore[index]
        transformed = _transform_points(rel_points, ops_by_color[color])
        for dr, dc in transformed:
            rr = int(anchor_r) + int(dr)
            cc = int(anchor_c) + int(dc)
            assert 0 <= rr < body_h and 0 <= cc < body_w
            assert out[rr, cc] in (BG, color)
            out[rr, cc] = color

    return out


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _randomize_shape(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out = set(cells)
    if random.random() < 0.5:
        max_r = max(r for r, _ in out)
        out = {(max_r - r, c) for r, c in out}
    if random.random() < 0.5:
        max_c = max(c for _, c in out)
        out = {(r, max_c - c) for r, c in out}
    if random.random() < 0.35:
        out = {(c, r) for r, c in out}
    return _normalize(out)


def _grow_shape(height: int, width: int, area: int) -> set[tuple[int, int]] | None:
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))
    return _randomize_shape(cells)


def _degree(shape: set[tuple[int, int]], cell: tuple[int, int]) -> int:
    r, c = cell
    return sum((nr, nc) in shape for nr, nc in _neighbors4(r, c))


def _make_object_shape(op: str) -> set[tuple[int, int]] | None:
    for _ in range(SHAPE_ATTEMPTS):
        box_h = random.randint(*SHAPE_BOX_RANGE)
        box_w = random.randint(*SHAPE_BOX_RANGE)
        max_area = min(SHAPE_AREA_RANGE[1], box_h * box_w)
        if max_area < SHAPE_AREA_RANGE[0]:
            continue
        area = random.randint(SHAPE_AREA_RANGE[0], max_area)
        shape = _grow_shape(box_h, box_w, area)
        if not shape:
            continue

        candidates = list(shape)
        random.shuffle(candidates)
        candidates.sort(key=lambda cell: _degree(shape, cell), reverse=True)

        for anchor_r, anchor_c in candidates:
            rel = {(r - anchor_r, c - anchor_c) for r, c in shape}
            transformed = _transform_points(rel, op)
            if transformed != rel:
                return rel
    return None


def _expand(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                out.add((r + dr, c + dc))
    return out


def _absolute(points: set[tuple[int, int]], anchor: tuple[int, int]) -> set[tuple[int, int]]:
    ar, ac = anchor
    return {(ar + r, ac + c) for r, c in points}


def _place_objects(specs: list[dict[str, object]], body_h: int, width: int) -> list[dict[str, object]] | None:
    placements: list[dict[str, object] | None] = [None] * len(specs)
    input_forbidden: set[tuple[int, int]] = set()
    output_forbidden: set[tuple[int, int]] = set()

    order = sorted(
        range(len(specs)),
        key=lambda idx: -len(specs[idx]["orig"] | specs[idx]["trans"]),  # type: ignore[operator]
    )

    for idx in order:
        orig = specs[idx]["orig"]  # type: ignore[index]
        trans = specs[idx]["trans"]  # type: ignore[index]
        footprint = orig | trans

        min_r = min(r for r, _ in footprint)
        max_r = max(r for r, _ in footprint)
        min_c = min(c for _, c in footprint)
        max_c = max(c for _, c in footprint)

        row_lo = 1 - min_r
        row_hi = body_h - 2 - max_r
        col_lo = 1 - min_c
        col_hi = width - 2 - max_c
        if row_lo > row_hi or col_lo > col_hi:
            return None

        for _ in range(PLACEMENT_ATTEMPTS):
            anchor = (
                random.randint(row_lo, row_hi),
                random.randint(col_lo, col_hi),
            )
            orig_abs = _absolute(orig, anchor)
            trans_abs = _absolute(trans, anchor)
            if orig_abs & input_forbidden:
                continue
            if trans_abs & output_forbidden:
                continue

            placements[idx] = {
                "anchor": anchor,
                "orig_abs": orig_abs,
                "trans_abs": trans_abs,
            }
            input_forbidden.update(_expand(orig_abs))
            output_forbidden.update(_expand(trans_abs))
            break
        else:
            return None

    return [placement for placement in placements if placement is not None]


def _choose_signal_columns(width: int, ops: list[str]) -> dict[str, list[int]] | None:
    counts = {op: ops.count(op) for op in OPS}
    chosen = {op: [] for op in OPS}
    used: list[int] = []

    for op in sorted(OPS, key=lambda name: counts[name], reverse=True):
        need = counts[op]
        if need == 0:
            continue
        pool = [
            c
            for c in range(width)
            if _signal_op_from_col(c, width) == op and 0 < c < width - 1
        ]
        random.shuffle(pool)
        for col in pool:
            if all(abs(col - prev) > 1 for prev in used):
                chosen[op].append(col)
                used.append(col)
                if len(chosen[op]) == need:
                    break
        if len(chosen[op]) < need:
            return None

    for op in OPS:
        random.shuffle(chosen[op])
    return chosen


def _attempt_generate() -> np.ndarray | None:
    object_count = random.randint(MIN_OBJECTS, MAX_OBJECTS)
    body_h = random.randint(*BODY_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)

    colors = random.sample(OBJECT_COLORS, object_count)
    ops = list(OPS) + [random.choice(OPS) for _ in range(object_count - len(OPS))]
    random.shuffle(ops)

    specs: list[dict[str, object]] = []
    for color, op in zip(colors, ops, strict=True):
        orig = _make_object_shape(op)
        if orig is None:
            return None
        trans = _transform_points(orig, op)
        specs.append(
            {
                "color": color,
                "op": op,
                "orig": orig,
                "trans": trans,
            }
        )

    placements = _place_objects(specs, body_h, width)
    if placements is None:
        return None

    signal_cols = _choose_signal_columns(width, ops)
    if signal_cols is None:
        return None

    grid = np.full((body_h + 1, width), BG, dtype=int)
    expected = np.full((body_h, width), BG, dtype=int)

    for spec, placement in zip(specs, placements, strict=True):
        color = int(spec["color"])
        op = spec["op"]

        for r, c in placement["orig_abs"]:  # type: ignore[index]
            grid[r + 1, c] = color
        anchor_r, anchor_c = placement["anchor"]  # type: ignore[index]
        grid[anchor_r + 1, anchor_c] = MAROON

        for r, c in placement["trans_abs"]:  # type: ignore[index]
            expected[r, c] = color

        signal_col = signal_cols[op].pop()
        grid[0, signal_col] = color

    if any(signal_cols[op] for op in OPS):
        return None
    if np.array_equal(expected, grid[1:]):
        return None

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid signal-anchor transform input")
