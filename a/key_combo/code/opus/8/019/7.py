from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
TEAL = 8

N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
N8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

SHAPE_LIBRARY = [
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 1), (1, 2), (1, 3)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
    ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
]

GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    for dr, dc in N4:
        yield r + dr, c + dc


def _neighbors8(r: int, c: int):
    for dr, dc in N8:
        yield r + dr, c + dc


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(sorted(cells), dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = _bbox(coords)
    shifted = coords - np.array([r0, c0])
    return tuple(sorted((int(r), int(c)) for r, c in shifted))


def _round_half_up(value: float) -> int:
    return int(np.floor(value + 0.5))


def _centroid(coords: np.ndarray) -> tuple[int, int]:
    cr = _round_half_up(float(coords[:, 0].mean()))
    cc = _round_half_up(float(coords[:, 1].mean()))
    return cr, cc


def _expanded_ring_cells(shape: tuple[int, int], box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = box
    cells = []
    for rr in range(max(0, r0 - 1), min(h - 1, r1 + 1) + 1):
        for cc in range(max(0, c0 - 1), min(w - 1, c1 + 1) + 1):
            if r0 <= rr <= r1 and c0 <= cc <= c1:
                continue
            cells.append((rr, cc))
    return cells


def _has_control_mark(grid: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    return any(int(grid[r, c]) == RED for r, c in _expanded_ring_cells(grid.shape, box))


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    template_comps = _components_of_color(g, TEAL)
    blue_comps = _components_of_color(g, BLUE)
    assert len(template_comps) == 1, "expected exactly one teal template object"
    assert 5 <= len(blue_comps) <= 8, "expected 5-8 blue objects"

    template_shape = _normalize(template_comps[0])
    objects = []
    for comp in blue_comps:
        box = _bbox(comp)
        match = _normalize(comp) == template_shape
        marked = _has_control_mark(g, box)
        red_neighbors = [
            (r, c)
            for r, c in _expanded_ring_cells(g.shape, box)
            if int(g[r, c]) == RED
        ]
        objects.append(
            {
                "coords": comp,
                "bbox": box,
                "match": match,
                "marked": marked,
                "centroid": _centroid(comp),
                "red_neighbors": red_neighbors,
            }
        )
    return {"template": template_comps[0], "template_shape": template_shape, "objects": objects}


def _spec_transform(grid: np.ndarray) -> np.ndarray:
    info = _analyze_grid(grid)
    out = np.asarray(grid, dtype=int).copy()

    emitters: list[tuple[int, int]] = []
    for obj in info["objects"]:
        coords = obj["coords"]
        if obj["match"]:
            out[coords[:, 0], coords[:, 1]] = TEAL
            if obj["marked"]:
                emitters.append(obj["centroid"])

    for cr, cc in emitters:
        out[cr, :] = TEAL
        out[:, cc] = TEAL

    out[np.asarray(grid, dtype=int) == RED] = BG
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor blue objects that match the teal template, and let only the marked matches emit full row-and-column teal rays.

    Input structure:
    - Black background contains one teal 4-connected template object, several blue 4-connected objects, and red control pixels.
    - Some blue objects have the same translation-normalized shape as the template; others are distractors.
    - Red control pixels sit one cell outside a blue object's bounding box, touching it only diagonally.

    Rule:
    - Normalize the teal template shape by translation.
    - For each blue 4-connected object, normalize its shape and compare it to the template.
    - Every matching blue object is recolored teal.
    - Among those matches only, any object with at least one red control pixel in its one-cell expanded bounding-box ring emits teal rays through its centroid row and centroid column.
    - Blue objects that do not match stay blue even if they have red markers; all red control pixels are cleared to black in the output.

    Key insight:
    - The red markers are gated by shape matching: match-only recoloring is the first filter, and only marked matches pass the second filter that triggers the rays.
    """
    return _spec_transform(grid)


def _transform_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    pts = np.array(shape, dtype=int)
    if random.random() < 0.5:
        pts = pts[:, ::-1]
    if random.random() < 0.5:
        pts[:, 0] *= -1
    if random.random() < 0.5:
        pts[:, 1] *= -1
    pts -= pts.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in pts))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    arr = np.array(shape, dtype=int)
    return int(arr[:, 0].max()) + 1, int(arr[:, 1].max()) + 1


def _shifted_coords(shape: tuple[tuple[int, int], ...], top: int, left: int) -> np.ndarray:
    return np.array([(top + r, left + c) for r, c in shape], dtype=int)


def _reserve_box(shape: tuple[tuple[int, int], ...], top: int, left: int) -> tuple[int, int, int, int]:
    h, w = _shape_dims(shape)
    return top - 1, left - 1, top + h, left + w


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _mark_candidates(grid: np.ndarray, coords: np.ndarray) -> list[tuple[int, int]]:
    box = _bbox(coords)
    coord_set = {tuple(map(int, cell)) for cell in coords}
    candidates = []
    for r, c in _expanded_ring_cells(grid.shape, box):
        if int(grid[r, c]) != BG:
            continue
        has_diag = False
        has_cardinal = False
        for nr, nc in _neighbors8(r, c):
            if (nr, nc) in coord_set:
                has_diag = True
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in coord_set:
                has_cardinal = True
        if has_diag and not has_cardinal:
            candidates.append((r, c))
    return candidates


def _pick_counts(n_objects: int) -> list[tuple[str, bool]]:
    roles: list[tuple[str, bool]] = [
        ("match", True),
        ("match", False),
        ("distractor", True),
    ]
    while len(roles) < n_objects:
        choice = random.random()
        if choice < 0.25 and sum(kind == "match" and marked for kind, marked in roles) < 3:
            roles.append(("match", True))
        elif choice < 0.55:
            roles.append(("match", False))
        elif choice < 0.8 and sum(kind == "distractor" and marked for kind, marked in roles) < 3:
            roles.append(("distractor", True))
        else:
            roles.append(("distractor", False))
    random.shuffle(roles)
    return roles


def _try_layout() -> np.ndarray | None:
    h = random.randint(15, 21)
    w = random.randint(15, 21)
    grid = np.full((h, w), BG, dtype=int)

    template_shape = _transform_shape(random.choice(SHAPE_LIBRARY))

    n_objects = random.randint(5, 8)
    roles = _pick_counts(n_objects)

    placed: list[dict] = []
    reserved: list[tuple[int, int, int, int]] = []

    items = [{"kind": "template", "shape": template_shape, "marked": False}]
    for kind, marked in roles:
        if kind == "match":
            shape = template_shape
        else:
            shape = template_shape
            for _ in range(20):
                shape = _transform_shape(random.choice(SHAPE_LIBRARY))
                if shape != template_shape:
                    break
            if shape == template_shape:
                return None
        items.append({"kind": kind, "shape": shape, "marked": marked})

    random.shuffle(items)
    items.sort(key=lambda item: 0 if item["kind"] == "template" else 1)

    for item in items:
        shape = item["shape"]
        sh, sw = _shape_dims(shape)
        row_choices = list(range(1, h - sh))
        col_choices = list(range(1, w - sw))
        if not row_choices or not col_choices:
            return None

        positions = [(r0, c0) for r0 in row_choices for c0 in col_choices]
        random.shuffle(positions)
        chosen = None
        for top, left in positions:
            box = _reserve_box(shape, top, left)
            if box[0] < 0 or box[1] < 0 or box[2] >= h or box[3] >= w:
                continue
            if any(_boxes_overlap(box, other) for other in reserved):
                continue
            coords = _shifted_coords(shape, top, left)
            chosen = (coords, box)
            break
        if chosen is None:
            return None

        coords, box = chosen
        color = TEAL if item["kind"] == "template" else BLUE
        grid[coords[:, 0], coords[:, 1]] = color
        entry = {"kind": item["kind"], "marked": item["marked"], "coords": coords, "shape": shape}
        placed.append(entry)
        reserved.append(box)

    emitters = []
    all_centroids = []
    for entry in placed:
        if entry["kind"] == "template":
            continue
        center = _centroid(entry["coords"])
        all_centroids.append(center)
        if entry["kind"] == "match" and entry["marked"]:
            emitters.append(center)

    emitter_rows = [r for r, _c in emitters]
    emitter_cols = [c for _r, c in emitters]
    if len(emitter_rows) != len(set(emitter_rows)) or len(emitter_cols) != len(set(emitter_cols)):
        return None
    for idx, (r, c) in enumerate(all_centroids):
        if (r, c) in emitters:
            continue
        if r in emitter_rows or c in emitter_cols:
            return None

    for entry in placed:
        if entry["kind"] == "template" or not entry["marked"]:
            continue
        candidates = _mark_candidates(grid, entry["coords"])
        if not candidates:
            return None
        filtered = [(r, c) for r, c in candidates if r not in emitter_rows and c not in emitter_cols]
        r, c = random.choice(filtered or candidates)
        grid[r, c] = RED

    return grid


def _validate_generated(grid: np.ndarray) -> None:
    info = _analyze_grid(grid)
    objects = info["objects"]
    matches = [obj for obj in objects if obj["match"]]
    mismatches = [obj for obj in objects if not obj["match"]]
    marked_matches = [obj for obj in matches if obj["marked"]]
    unmarked_matches = [obj for obj in matches if not obj["marked"]]
    marked_mismatches = [obj for obj in mismatches if obj["marked"]]

    assert len(matches) >= 2
    assert marked_matches
    assert unmarked_matches
    assert marked_mismatches

    blue_pixels = set()
    for obj in objects:
        coords = {tuple(map(int, cell)) for cell in obj["coords"]}
        assert blue_pixels.isdisjoint(coords)
        blue_pixels |= coords

    red_pixels = [tuple(map(int, cell)) for cell in np.argwhere(grid == RED)]
    assert red_pixels
    for r, c in red_pixels:
        attached = 0
        for obj in objects:
            if (r, c) not in obj["red_neighbors"]:
                continue
            coord_set = {tuple(map(int, cell)) for cell in obj["coords"]}
            has_diag = any((nr, nc) in coord_set for nr, nc in _neighbors8(r, c))
            has_cardinal = any((nr, nc) in coord_set for nr, nc in _neighbors4(r, c))
            if has_diag and not has_cardinal:
                attached += 1
        assert attached == 1, "each red mark must belong to exactly one blue object"

    emitters = [obj["centroid"] for obj in marked_matches]
    rows = [r for r, _c in emitters]
    cols = [c for _r, c in emitters]
    assert len(rows) == len(set(rows))
    assert len(cols) == len(set(cols))
    for r, c in red_pixels:
        assert r not in rows and c not in cols, "markers should not punch holes through rays"

    expected = _spec_transform(grid)
    actual = main(grid)
    assert np.array_equal(actual, expected)

    for obj in marked_mismatches:
        coords = obj["coords"]
        assert np.all(expected[coords[:, 0], coords[:, 1]] == BLUE), "marked distractors must stay blue"

    for obj in unmarked_matches:
        coords = obj["coords"]
        assert np.all(expected[coords[:, 0], coords[:, 1]] == TEAL), "unmarked matches must still recolor"


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _try_layout()
        if grid is None:
            continue
        try:
            _validate_generated(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid instance for puzzle 7")
