from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BACKGROUND = 0
MARKER_COLORS = (2, 3, 4)
OBJECT_COLOR = 5
EDGES = ("top", "bottom", "left", "right")

# Fixed visible mapping inferred from examples: color -> slide direction.
DIRECTION_BY_COLOR = {
    2: (0, 1),   # red -> right
    3: (1, 0),   # green -> down
    4: (0, -1),  # yellow -> left
}

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

BASE_SHAPES = [
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
]

GEN_ATTEMPTS = 500
PLACEMENT_ATTEMPTS = 120


def _normalize_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in shape)
    c0 = min(c for _, c in shape)
    return tuple(sorted((r - r0, c - c0) for r, c in shape))


def _transform_shape(
    shape: tuple[tuple[int, int], ...], rotation: int, mirror: bool
) -> tuple[tuple[int, int], ...]:
    coords = list(shape)
    if mirror:
        coords = [(r, -c) for r, c in coords]
    for _ in range(rotation % 4):
        coords = [(c, -r) for r, c in coords]
    return _normalize_shape(tuple(coords))


def _random_shape() -> tuple[tuple[int, int], ...]:
    return _transform_shape(
        random.choice(BASE_SHAPES),
        rotation=random.randrange(4),
        mirror=bool(random.randrange(2)),
    )


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _edge_cells(shape: tuple[int, int], edge: str) -> list[tuple[int, int]]:
    h, w = shape
    if edge == "top":
        return [(0, c) for c in range(w)]
    if edge == "bottom":
        return [(h - 1, c) for c in range(w)]
    if edge == "left":
        return [(r, 0) for r in range(h)]
    return [(r, w - 1) for r in range(h)]


def _detect_marker_edge(grid: np.ndarray) -> str:
    counts = {
        edge: sum(int(grid[r, c] != BACKGROUND) for r, c in _edge_cells(grid.shape, edge))
        for edge in EDGES
    }
    active = [edge for edge, count in counts.items() if count > 0]
    assert len(active) == 1, f"expected one marked edge, got counts={counts}"
    return active[0]


def _alignment_value(edge: str, r: int, c: int) -> int:
    return int(c if edge in ("top", "bottom") else r)


def _extract_markers(
    grid: np.ndarray, edge: str
) -> tuple[dict[int, list[tuple[int, int]]], list[dict[str, int]]]:
    markers: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r, c in _edge_cells(grid.shape, edge):
        color = int(grid[r, c])
        if color == BACKGROUND:
            continue
        assert color in MARKER_COLORS, "only boundary markers may touch the edge"
        markers[color].append((int(r), int(c)))

    assert 2 <= len(markers) <= 3, "expected 2-3 marker colors"
    records: list[dict[str, int]] = []
    for color, pts in markers.items():
        assert len(pts) == 2, f"marker color {color} must appear exactly twice"
        records.extend(
            {"color": int(color), "align": _alignment_value(edge, r, c)}
            for r, c in pts
        )
    return dict(markers), records


def _step_by_color(markers: dict[int, list[tuple[int, int]]], edge: str) -> dict[int, int]:
    steps: dict[int, int] = {}
    for color, pts in markers.items():
        a = _alignment_value(edge, *pts[0])
        b = _alignment_value(edge, *pts[1])
        steps[int(color)] = abs(int(a) - int(b))
        assert steps[int(color)] >= 1
    return steps


def _nearest_marker_color(
    axis_value: float, marker_records: list[dict[str, int]]
) -> tuple[int, float]:
    dists = [abs(axis_value - float(rec["align"])) for rec in marker_records]
    best = min(dists)
    nearest = [rec for rec, dist in zip(marker_records, dists) if abs(dist - best) < 1e-9]
    colors = {int(rec["color"]) for rec in nearest}
    assert len(nearest) == 1 and len(colors) == 1, "object-marker alignment must be unique"
    return int(nearest[0]["color"]), float(best)


def _translate_coords(coords: np.ndarray, dr: int, dc: int) -> np.ndarray:
    return coords + np.array([[int(dr), int(dc)]], dtype=int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each internal grey object by the vector encoded by repeated boundary marker dots.

    Input structure:
    - The background is black.
    - Exactly one grid edge contains non-black single-pixel markers, using 2-3 colors from {red, green, yellow}; the other three edges are empty.
    - Each marker color appears exactly twice on that edge, and the gap between the two same-color markers is that color's step size.
    - Inside the grid are 2-4 grey 4-connected objects; each object's centroid aligns most closely to exactly one marker point on the marked edge.

    Rule:
    - Detect the unique marked edge and use its row/column coordinate as the alignment axis.
    - For each marker color, read its step from the spacing between its two boundary dots.
    - For each grey object, find the uniquely nearest marker point to the object's centroid on the alignment axis; that marker's color selects the motion direction, and its same-color spacing selects the motion length.
    - Translate the whole object rigidly by that vector, clear its old location to black, and keep the boundary markers unchanged.

    Key insight:
    - The boundary dots form a codebook: color chooses direction, while same-color repetition spacing chooses the slide distance, so both the boundary pattern and centroid alignment are necessary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    edge = _detect_marker_edge(g)
    markers, marker_records = _extract_markers(g, edge)
    steps = _step_by_color(markers, edge)
    axis_index = 1 if edge in ("top", "bottom") else 0

    objects = _components(g == OBJECT_COLOR)
    assert 2 <= len(objects) <= 4, "expected 2-4 interior objects"

    out = g.copy()
    out[out == OBJECT_COLOR] = BACKGROUND

    for coords in objects:
        centroid = coords.mean(axis=0)
        color, best_dist = _nearest_marker_color(float(centroid[axis_index]), marker_records)
        assert best_dist <= 1.0 + 1e-9, "object centroid must align to a marker point"
        step = int(steps[color])
        dr, dc = DIRECTION_BY_COLOR[color]
        moved = _translate_coords(coords, dr * step, dc * step)
        for r, c in moved:
            assert 0 <= int(r) < g.shape[0] and 0 <= int(c) < g.shape[1], "translated object left the grid"
            assert int(g[int(r), int(c)]) != BACKGROUND or int(out[int(r), int(c)]) == BACKGROUND
        for r, c in moved:
            out[int(r), int(c)] = OBJECT_COLOR

    return out


def _build_expected(
    shape: tuple[int, int],
    markers: dict[int, list[tuple[int, int]]],
    object_specs: list[dict[str, object]],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for color, pts in markers.items():
        for r, c in pts:
            out[int(r), int(c)] = int(color)
    for spec in object_specs:
        for r, c in spec["dst"]:
            out[int(r), int(c)] = OBJECT_COLOR
    return out


def _sample_marker_pairs(
    edge: str, h: int, w: int, colors: list[int]
) -> dict[int, list[tuple[int, int]]] | None:
    axis_len = w if edge in ("top", "bottom") else h
    max_step = min(6, axis_len - 4)
    if max_step < 2:
        return None

    for _ in range(100):
        placed: dict[int, list[int]] = {}
        used_by_other: list[int] = []
        order = colors[:]
        random.shuffle(order)
        ok = True
        for color in order:
            step = random.randint(2, max_step)
            starts = [
                start
                for start in range(1, axis_len - 1 - step)
                if all(abs(start - other) >= 3 and abs(start + step - other) >= 3 for other in used_by_other)
            ]
            if not starts:
                ok = False
                break
            start = random.choice(starts)
            placed[int(color)] = [int(start), int(start + step)]
            used_by_other.extend(placed[int(color)])
        if not ok:
            continue

        markers: dict[int, list[tuple[int, int]]] = {}
        for color, aligns in placed.items():
            pts: list[tuple[int, int]] = []
            for align in aligns:
                if edge == "top":
                    pts.append((0, align))
                elif edge == "bottom":
                    pts.append((h - 1, align))
                elif edge == "left":
                    pts.append((align, 0))
                else:
                    pts.append((align, w - 1))
            markers[int(color)] = pts
        return markers
    return None


def _candidate_placements(
    shape: tuple[tuple[int, int], ...],
    color: int,
    edge: str,
    marker_records: list[dict[str, int]],
    step: int,
    reserved_any: set[tuple[int, int]],
    grid_shape: tuple[int, int],
) -> list[dict[str, object]]:
    h, w = grid_shape
    dr, dc = DIRECTION_BY_COLOR[color]
    shape_h, shape_w = _shape_dims(shape)
    axis_index = 1 if edge in ("top", "bottom") else 0
    candidates: list[dict[str, object]] = []

    for r0 in range(1, h - shape_h):
        for c0 in range(1, w - shape_w):
            src = {(r0 + r, c0 + c) for r, c in shape}
            dst = {(r + dr * step, c + dc * step) for r, c in src}
            if src & reserved_any or dst & reserved_any:
                continue
            if not src.isdisjoint(dst):
                continue
            if any(not (0 < r < h - 1 and 0 < c < w - 1) for r, c in src | dst):
                continue
            coords = np.array(sorted(src), dtype=float)
            axis_value = float(coords[:, axis_index].mean())
            try:
                nearest_color, dist = _nearest_marker_color(axis_value, marker_records)
            except AssertionError:
                continue
            if nearest_color != int(color) or dist > 1.0 + 1e-9:
                continue
            candidates.append(
                {
                    "src": tuple(sorted(src)),
                    "dst": tuple(sorted(dst)),
                    "axis_value": axis_value,
                    "color": int(color),
                }
            )
    return candidates


def _validate_instance(
    grid: np.ndarray,
    markers: dict[int, list[tuple[int, int]]],
    object_specs: list[dict[str, object]],
    expected: np.ndarray,
) -> bool:
    try:
        edge = _detect_marker_edge(grid)
        extracted_markers, marker_records = _extract_markers(grid, edge)
        steps = _step_by_color(extracted_markers, edge)
        if extracted_markers != markers:
            return False
        if len(object_specs) < 2 or len(object_specs) > 4:
            return False

        comps = _components(grid == OBJECT_COLOR)
        if len(comps) != len(object_specs):
            return False

        spec_by_src = {
            tuple(sorted(tuple(map(int, cell)) for cell in spec["src"])): spec
            for spec in object_specs
        }
        seen_colors = set()
        all_src: set[tuple[int, int]] = set()
        all_dst: set[tuple[int, int]] = set()
        for comp in comps:
            key = tuple(sorted((int(r), int(c)) for r, c in comp))
            if key not in spec_by_src:
                return False
            spec = spec_by_src[key]
            centroid = comp.mean(axis=0)
            axis_value = float(centroid[1 if edge in ("top", "bottom") else 0])
            color, dist = _nearest_marker_color(axis_value, marker_records)
            if color != int(spec["color"]) or dist > 1.0 + 1e-9:
                return False
            if steps[color] != int(spec["step"]):
                return False
            all_src.update(key)
            all_dst.update(tuple(sorted(tuple(map(int, cell)) for cell in spec["dst"])))
            seen_colors.add(color)

        if len(all_src) != sum(len(spec["src"]) for spec in object_specs):
            return False
        if len(all_dst) != sum(len(spec["dst"]) for spec in object_specs):
            return False
        if not np.array_equal(main(grid), expected):
            return False
        if np.array_equal(grid, expected):
            return False
        return seen_colors == set(markers)
    except AssertionError:
        return False


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(16, 22)
    w = random.randint(16, 22)
    edge = random.choice(EDGES)

    color_count = 3 if random.random() < 0.65 else 2
    colors = random.sample(list(MARKER_COLORS), color_count)
    markers = _sample_marker_pairs(edge, h, w, colors)
    if markers is None:
        return None

    marker_records = [
        {"color": color, "align": _alignment_value(edge, r, c)}
        for color, pts in markers.items()
        for r, c in pts
    ]
    steps = _step_by_color(markers, edge)

    object_count = random.randint(len(colors), 4)
    assignments = colors[:] + [random.choice(colors) for _ in range(object_count - len(colors))]
    random.shuffle(assignments)

    grid = np.zeros((h, w), dtype=int)
    reserved_any: set[tuple[int, int]] = set()
    for color, pts in markers.items():
        for r, c in pts:
            grid[r, c] = color
            reserved_any.add((int(r), int(c)))

    object_specs: list[dict[str, object]] = []
    for color in assignments:
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            shape = _random_shape()
            candidates = _candidate_placements(
                shape=shape,
                color=int(color),
                edge=edge,
                marker_records=marker_records,
                step=int(steps[int(color)]),
                reserved_any=reserved_any,
                grid_shape=grid.shape,
            )
            if not candidates:
                continue
            choice = random.choice(candidates)
            src = tuple((int(r), int(c)) for r, c in choice["src"])
            dst = tuple((int(r), int(c)) for r, c in choice["dst"])
            for r, c in src:
                grid[r, c] = OBJECT_COLOR
            reserved_any.update(src)
            reserved_any.update(dst)
            object_specs.append(
                {
                    "color": int(color),
                    "step": int(steps[int(color)]),
                    "src": src,
                    "dst": dst,
                }
            )
            placed = True
            break
        if not placed:
            return None

    expected = _build_expected(grid.shape, markers, object_specs)
    if not _validate_instance(grid, markers, object_specs, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid boundary-marker slide instance")
