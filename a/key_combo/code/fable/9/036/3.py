import random
from collections import Counter, deque

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


GENERATION_RETRIES = 400
GENERATION_MODES = ("disjoint", "overlap", "nested")
_GEN_COUNTER = 0


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, grid.flat))
    if exclude is not None:
        colors -= set(exclude)
    return sorted(colors)


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> BBox:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def find_components(
    grid: np.ndarray,
    *,
    bg: int,
    colors: list[int] | None = None,
) -> list[dict]:
    h, w = grid.shape
    if colors is None:
        colors = colors_present(grid, exclude={bg})
    color_set = {int(c) for c in colors if int(c) != bg}
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == bg or color not in color_set:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if (
                        in_bounds(grid.shape, nr, nc)
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": bbox_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def rect_interior_mask(shape: tuple[int, int], rect: BBox) -> np.ndarray:
    r0, c0, r1, c1 = rect
    mask = np.zeros(shape, dtype=bool)
    if r1 - r0 >= 2 and c1 - c0 >= 2:
        mask[r0 + 1 : r1, c0 + 1 : c1] = True
    return mask


def rect_from_points(points: list[Coord]) -> BBox:
    rows = [p[0] for p in points]
    cols = [p[1] for p in points]
    return min(rows), min(cols), max(rows), max(cols)


def bbox_contains(outer: BBox, inner: BBox) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
    )


def classify_cells(cells: list[Coord], interiors: list[np.ndarray]) -> tuple[str, int | frozenset[int] | None]:
    rows = np.array([r for r, _ in cells], dtype=int)
    cols = np.array([c for _, c in cells], dtype=int)
    inside = [mask[rows, cols] for mask in interiors]
    any_in = [bool(arr.any()) for arr in inside]
    all_in = [bool(arr.all()) for arr in inside]
    crossed = frozenset(i for i in range(2) if any_in[i] and not all_in[i])

    if all_in[0] and all_in[1]:
        return "erase", None
    if crossed:
        return "cross", crossed
    if all_in[0] and not any_in[1]:
        return "recolor", 0
    if all_in[1] and not any_in[0]:
        return "recolor", 1
    if not any_in[0] and not any_in[1]:
        return "outside", None
    raise AssertionError("unreachable cell classification")


def analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)
    h, w = g.shape

    marker_colors: list[int] = []
    markers: dict[int, list[Coord]] = {}
    for color in colors_present(g, exclude={bg}):
        coords = np.argwhere(g == color)
        if coords.shape[0] != 2:
            continue
        pts = [(int(r), int(c)) for r, c in coords]
        if any(sum(1 for nr, nc in neighbors4(r, c) if in_bounds(g.shape, nr, nc) and int(g[nr, nc]) == color) for r, c in pts):
            continue
        marker_colors.append(color)
        markers[color] = pts
    assert len(marker_colors) == 2, "expected exactly two marker colors"

    rects = [rect_from_points(markers[color]) for color in sorted(marker_colors)]
    for rect in rects:
        r0, c0, r1, c1 = rect
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "markers must define rectangles with nonempty strict interiors"
    interiors = [rect_interior_mask((h, w), rect) for rect in rects]

    fg_colors = colors_present(g, exclude={bg, *marker_colors})
    objects = find_components(g, bg=bg, colors=fg_colors)
    assert objects, "expected foreground objects"
    for obj in objects:
        cells = [tuple(map(int, rc)) for rc in obj["coords"]]
        kind, detail = classify_cells(cells, interiors)
        obj["kind"] = kind
        obj["detail"] = detail
    return {
        "bg": bg,
        "marker_colors": sorted(marker_colors),
        "markers": markers,
        "rects": rects,
        "interiors": interiors,
        "objects": objects,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore two invisible rectangles from their diagonal corner markers and transform each foreground object by region membership.

    Input structure:
    - Two marker colors each appear as exactly two isolated single pixels, and each same-color pair gives one axis-aligned rectangle's diagonal corners.
    - The rectangles themselves are not drawn; only their corner markers are visible.
    - Additional 4-connected foreground objects use colors different from the two marker colors.

    Rule:
    - Reconstruct each rectangle and use only its strict interior for membership tests.
    - Erase any object whose every pixel lies in the strict-interior overlap of both rectangles.
    - Recolor any object whose every pixel lies strictly inside exactly one rectangle to that rectangle's marker color.
    - For any object that crosses a rectangle boundary, delete only the pixels that lie in the strict interior of each crossed rectangle, keeping the remaining pixels in their original color.
    - Keep objects completely outside both rectangles unchanged, and always preserve the four marker pixels.

    Key insight:
    - The hidden rectangles must be inferred from same-color diagonal corner pairs, then objects are processed by a containment taxonomy: double-inside is erased, single-inside is recolored, and boundary-crossing is cropped only on the crossed rectangle interiors.
    """
    g = np.asarray(grid, dtype=int)
    info = analyze_grid(g)
    out = g.copy()
    bg = info["bg"]
    interiors = info["interiors"]
    marker_colors = info["marker_colors"]

    for obj in info["objects"]:
        coords = obj["coords"]
        rows = coords[:, 0]
        cols = coords[:, 1]
        if obj["kind"] == "erase":
            out[rows, cols] = bg
        elif obj["kind"] == "recolor":
            target_color = marker_colors[int(obj["detail"])]
            out[rows, cols] = target_color
        elif obj["kind"] == "cross":
            remove = np.zeros(coords.shape[0], dtype=bool)
            for idx in obj["detail"]:
                remove |= interiors[int(idx)][rows, cols]
            out[rows[remove], cols[remove]] = bg
        else:
            assert obj["kind"] == "outside"
    return out


def random_polyomino_patch(area: int, max_h: int = 4, max_w: int = 4) -> np.ndarray:
    coords = {(0, 0)}
    while len(coords) < area:
        frontier: list[Coord] = []
        for r, c in coords:
            for nr, nc in neighbors4(r, c):
                if (nr, nc) in coords:
                    continue
                trial = coords | {(nr, nc)}
                rows = [p[0] for p in trial]
                cols = [p[1] for p in trial]
                if max(rows) - min(rows) + 1 <= max_h and max(cols) - min(cols) + 1 <= max_w:
                    frontier.append((nr, nc))
        coords.add(random.choice(frontier))
    rows = [p[0] for p in coords]
    cols = [p[1] for p in coords]
    r0, c0 = min(rows), min(cols)
    h = max(rows) - r0 + 1
    w = max(cols) - c0 + 1
    patch = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        patch[r - r0, c - c0] = True
    for _ in range(random.randint(0, 3)):
        patch = np.rot90(patch)
    if random.random() < 0.5:
        patch = patch[:, ::-1]
    return patch


def can_place(cells: list[Coord], occupied: np.ndarray) -> bool:
    cell_set = set(cells)
    if len(cell_set) != len(cells):
        return False
    h, w = occupied.shape
    for r, c in cells:
        if not in_bounds((h, w), r, c) or occupied[r, c]:
            return False
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if in_bounds((h, w), rr, cc) and occupied[rr, cc] and (rr, cc) not in cell_set:
                    return False
    return True


def place_cells(grid: np.ndarray, occupied: np.ndarray, cells: list[Coord], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color
        occupied[r, c] = True


def place_polyomino_in_mask(
    grid: np.ndarray,
    occupied: np.ndarray,
    allowed: np.ndarray,
    color: int,
    *,
    area_range: tuple[int, int] = (2, 5),
    tries: int = 200,
) -> bool:
    h, w = grid.shape
    if int(allowed.sum()) < area_range[0]:
        return False
    true_coords = np.argwhere(allowed)
    rmin, cmin = true_coords.min(axis=0)
    rmax, cmax = true_coords.max(axis=0)
    for _ in range(tries):
        area = random.randint(area_range[0], min(area_range[1], int(allowed.sum())))
        patch = random_polyomino_patch(area)
        ph, pw = patch.shape
        row_lo = max(0, int(rmin) - ph + 1)
        row_hi = min(h - ph, int(rmax))
        col_lo = max(0, int(cmin) - pw + 1)
        col_hi = min(w - pw, int(cmax))
        if row_lo > row_hi or col_lo > col_hi:
            continue
        for _ in range(40):
            r0 = random.randint(row_lo, row_hi)
            c0 = random.randint(col_lo, col_hi)
            cells = [(r0 + r, c0 + c) for r, c in np.argwhere(patch)]
            if all(allowed[r, c] for r, c in cells) and can_place(cells, occupied):
                place_cells(grid, occupied, cells, color)
                return True
    return False


def side_cross_cells(rect: BBox, side: str, anchor: int, inside_len: int, outside_len: int) -> list[Coord]:
    r0, c0, r1, c1 = rect
    if side == "left":
        return [(anchor, c) for c in range(c0 - outside_len + 1, c0 + inside_len + 1)]
    if side == "right":
        return [(anchor, c) for c in range(c1 - inside_len, c1 + outside_len)]
    if side == "top":
        return [(r, anchor) for r in range(r0 - outside_len + 1, r0 + inside_len + 1)]
    if side == "bottom":
        return [(r, anchor) for r in range(r1 - inside_len, r1 + outside_len)]
    raise ValueError(side)


def place_single_cross(
    grid: np.ndarray,
    occupied: np.ndarray,
    rects: list[BBox],
    interiors: list[np.ndarray],
    color: int,
    rect_idx: int,
    *,
    tries: int = 250,
) -> bool:
    rect = rects[rect_idx]
    r0, c0, r1, c1 = rect
    for _ in range(tries):
        side = random.choice(["left", "right", "top", "bottom"])
        inside_len = random.randint(1, 3)
        outside_len = random.randint(1, 3)
        if side in {"left", "right"}:
            if r1 - r0 <= 2:
                continue
            anchor = random.randint(r0 + 1, r1 - 1)
        else:
            if c1 - c0 <= 2:
                continue
            anchor = random.randint(c0 + 1, c1 - 1)
        cells = side_cross_cells(rect, side, anchor, inside_len, outside_len)
        if not all(in_bounds(grid.shape, r, c) for r, c in cells):
            continue
        kind, detail = classify_cells(cells, interiors)
        if kind == "cross" and detail == frozenset({rect_idx}) and can_place(cells, occupied):
            place_cells(grid, occupied, cells, color)
            return True
    return False


def place_cross_both(
    grid: np.ndarray,
    occupied: np.ndarray,
    rects: list[BBox],
    interiors: list[np.ndarray],
    color: int,
    *,
    tries: int = 250,
) -> bool:
    overlap = interiors[0] & interiors[1]
    rows, cols = np.where(overlap)
    if rows.size == 0:
        return False
    for _ in range(tries):
        horizontal = random.random() < 0.5
        if horizontal:
            row = int(random.choice(rows.tolist()))
            left = min(rects[0][1], rects[1][1]) - random.randint(1, 3) + 1
            right = max(rects[0][3], rects[1][3]) + random.randint(1, 3) - 1
            cells = [(row, c) for c in range(left, right + 1)]
        else:
            col = int(random.choice(cols.tolist()))
            top = min(rects[0][0], rects[1][0]) - random.randint(1, 3) + 1
            bottom = max(rects[0][2], rects[1][2]) + random.randint(1, 3) - 1
            cells = [(r, col) for r in range(top, bottom + 1)]
        if not all(in_bounds(grid.shape, r, c) for r, c in cells):
            continue
        kind, detail = classify_cells(cells, interiors)
        if kind == "cross" and detail == frozenset({0, 1}) and can_place(cells, occupied):
            place_cells(grid, occupied, cells, color)
            return True
    return False


def sample_disjoint_rects(h: int, w: int) -> list[BBox]:
    horizontal = random.random() < 0.5
    if horizontal:
        rh1 = random.randint(6, 9)
        rw1 = random.randint(6, 8)
        rh2 = random.randint(6, 9)
        rw2 = random.randint(6, 8)
        gap = random.randint(3, 6)
        total = rw1 + gap + rw2
        if total > w - 4:
            raise ValueError("disjoint rectangles do not fit")
        c01 = random.randint(2, w - total - 2)
        c02 = c01 + rw1 + gap
        r01 = random.randint(2, h - rh1 - 2)
        r02 = random.randint(2, h - rh2 - 2)
        return [
            (r01, c01, r01 + rh1 - 1, c01 + rw1 - 1),
            (r02, c02, r02 + rh2 - 1, c02 + rw2 - 1),
        ]
    rh1 = random.randint(6, 8)
    rw1 = random.randint(6, 9)
    rh2 = random.randint(6, 8)
    rw2 = random.randint(6, 9)
    gap = random.randint(3, 6)
    total = rh1 + gap + rh2
    if total > h - 4:
        raise ValueError("disjoint rectangles do not fit")
    r01 = random.randint(2, h - total - 2)
    r02 = r01 + rh1 + gap
    c01 = random.randint(2, w - rw1 - 2)
    c02 = random.randint(2, w - rw2 - 2)
    return [
        (r01, c01, r01 + rh1 - 1, c01 + rw1 - 1),
        (r02, c02, r02 + rh2 - 1, c02 + rw2 - 1),
    ]


def sample_overlap_rects(h: int, w: int) -> list[BBox]:
    for _ in range(300):
        rh1 = random.randint(7, 10)
        rw1 = random.randint(7, 10)
        rh2 = random.randint(7, 10)
        rw2 = random.randint(7, 10)
        r01 = random.randint(2, h - rh1 - 2)
        c01 = random.randint(2, w - rw1 - 2)
        r02 = random.randint(max(2, r01 - rh2 + 4), min(h - rh2 - 2, r01 + rh1 - 4))
        c02 = random.randint(max(2, c01 - rw2 + 4), min(w - rw2 - 2, c01 + rw1 - 4))
        rects = [
            (r01, c01, r01 + rh1 - 1, c01 + rw1 - 1),
            (r02, c02, r02 + rh2 - 1, c02 + rw2 - 1),
        ]
        interiors = [rect_interior_mask((h, w), rect) for rect in rects]
        overlap = interiors[0] & interiors[1]
        if not overlap.any():
            continue
        if bbox_contains(rects[0], rects[1]) or bbox_contains(rects[1], rects[0]):
            continue
        only0 = interiors[0] & ~interiors[1]
        only1 = interiors[1] & ~interiors[0]
        if int(overlap.sum()) >= 4 and int(only0.sum()) >= 6 and int(only1.sum()) >= 6:
            return rects
    raise RuntimeError("failed to sample overlap rectangles")


def sample_nested_rects(h: int, w: int) -> list[BBox]:
    outer_h = random.randint(10, 14)
    outer_w = random.randint(10, 14)
    r0 = random.randint(2, h - outer_h - 2)
    c0 = random.randint(2, w - outer_w - 2)
    outer = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
    inner_h = random.randint(5, min(8, outer_h - 4))
    inner_w = random.randint(5, min(8, outer_w - 4))
    ir0 = random.randint(outer[0] + 2, outer[2] - inner_h - 1)
    ic0 = random.randint(outer[1] + 2, outer[3] - inner_w - 1)
    inner = (ir0, ic0, ir0 + inner_h - 1, ic0 + inner_w - 1)
    return [outer, inner] if random.random() < 0.5 else [inner, outer]


def sample_rects(mode: str, h: int, w: int) -> list[BBox]:
    if mode == "disjoint":
        return sample_disjoint_rects(h, w)
    if mode == "overlap":
        return sample_overlap_rects(h, w)
    if mode == "nested":
        return sample_nested_rects(h, w)
    raise ValueError(mode)


def place_markers(grid: np.ndarray, occupied: np.ndarray, rects: list[BBox], colors: list[int]) -> None:
    for rect, color in zip(rects, colors):
        r0, c0, r1, c1 = rect
        if random.random() < 0.5:
            pts = [(r0, c0), (r1, c1)]
        else:
            pts = [(r0, c1), (r1, c0)]
        place_cells(grid, occupied, pts, color)


def detect_nested_roles(rects: list[BBox]) -> tuple[int, int]:
    if bbox_contains(rects[0], rects[1]):
        return 0, 1
    if bbox_contains(rects[1], rects[0]):
        return 1, 0
    raise AssertionError("rectangles are not nested")


def gen_attempt(mode: str, variant: int) -> np.ndarray | None:
    h = random.randint(18, 25)
    w = random.randint(18, 25)
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    try:
        rects = sample_rects(mode, h, w)
    except (RuntimeError, ValueError):
        return None

    marker_colors = random.sample(list(range(1, 10)), 2)
    remaining = [c for c in range(1, 10) if c not in marker_colors]
    object_budget = {"disjoint": 5, "overlap": 7, "nested": 5}[mode]
    if object_budget > len(remaining):
        return None
    object_colors = random.sample(remaining, object_budget)

    place_markers(grid, occupied, rects, marker_colors)
    interiors = [rect_interior_mask((h, w), rect) for rect in rects]
    overlap = interiors[0] & interiors[1]
    only0 = interiors[0] & ~interiors[1]
    only1 = interiors[1] & ~interiors[0]
    outside = ~(interiors[0] | interiors[1])

    color_iter = iter(object_colors)

    if mode == "disjoint":
        if not place_polyomino_in_mask(grid, occupied, only0, next(color_iter)):
            return None
        if not place_polyomino_in_mask(grid, occupied, only1, next(color_iter)):
            return None
        if not place_single_cross(grid, occupied, rects, interiors, next(color_iter), 0):
            return None
        if not place_single_cross(grid, occupied, rects, interiors, next(color_iter), 1):
            return None
        if not place_polyomino_in_mask(grid, occupied, outside, next(color_iter), area_range=(2, 4)):
            return None

    elif mode == "overlap":
        if not place_polyomino_in_mask(grid, occupied, overlap, next(color_iter), area_range=(2, 4)):
            return None
        if not place_polyomino_in_mask(grid, occupied, only0, next(color_iter)):
            return None
        if not place_polyomino_in_mask(grid, occupied, only1, next(color_iter)):
            return None
        if not place_single_cross(grid, occupied, rects, interiors, next(color_iter), 0):
            return None
        if not place_single_cross(grid, occupied, rects, interiors, next(color_iter), 1):
            return None
        if not place_cross_both(grid, occupied, rects, interiors, next(color_iter)):
            return None
        if not place_polyomino_in_mask(grid, occupied, outside, next(color_iter), area_range=(2, 4)):
            return None

    else:
        outer_idx, inner_idx = detect_nested_roles(rects)
        only_outer = interiors[outer_idx] & ~interiors[inner_idx]
        if not place_polyomino_in_mask(grid, occupied, overlap, next(color_iter), area_range=(2, 4)):
            return None
        if not place_polyomino_in_mask(grid, occupied, only_outer, next(color_iter), area_range=(2, 5)):
            return None
        if not place_single_cross(grid, occupied, rects, interiors, next(color_iter), inner_idx):
            return None
        if not place_single_cross(grid, occupied, rects, interiors, next(color_iter), outer_idx):
            return None
        if not place_polyomino_in_mask(grid, occupied, outside, next(color_iter), area_range=(2, 4)):
            return None

    if not validate_case(grid, mode):
        return None
    return grid


def validate_case(grid: np.ndarray, mode: str) -> bool:
    try:
        info = analyze_grid(grid)
        out = main(grid)
    except Exception:
        return False

    if np.array_equal(out, grid):
        return False

    marker_colors = info["marker_colors"]
    markers = info["markers"]
    rects = info["rects"]
    interiors = info["interiors"]
    overlap = interiors[0] & interiors[1]
    only0 = interiors[0] & ~interiors[1]
    only1 = interiors[1] & ~interiors[0]

    for color in marker_colors:
        for r, c in markers[color]:
            if int(out[r, c]) != color:
                return False

    counts: Counter = Counter()
    for obj in info["objects"]:
        coords = obj["coords"]
        rows = coords[:, 0]
        cols = coords[:, 1]
        kind = obj["kind"]
        detail = obj["detail"]
        in_color = int(obj["color"])
        if kind == "erase":
            counts["erase"] += 1
            if not np.all(out[rows, cols] == info["bg"]):
                return False
        elif kind == "recolor":
            key = f"recolor_{detail}"
            counts[key] += 1
            if not np.all(out[rows, cols] == marker_colors[int(detail)]):
                return False
        elif kind == "outside":
            counts["outside"] += 1
            if not np.all(out[rows, cols] == in_color):
                return False
        else:
            key = f"cross_{'-'.join(map(str, sorted(detail)))}"
            counts[key] += 1
            remove = np.zeros(coords.shape[0], dtype=bool)
            for idx in detail:
                remove |= interiors[int(idx)][rows, cols]
            if not remove.any() or remove.all():
                return False
            if not np.all(out[rows[remove], cols[remove]] == info["bg"]):
                return False
            if not np.all(out[rows[~remove], cols[~remove]] == in_color):
                return False

    if mode == "disjoint":
        if overlap.any():
            return False
        return (
            counts["recolor_0"] >= 1
            and counts["recolor_1"] >= 1
            and counts["cross_0"] >= 1
            and counts["cross_1"] >= 1
            and counts["outside"] >= 1
            and counts["erase"] == 0
        )

    if mode == "overlap":
        if not overlap.any():
            return False
        if bbox_contains(rects[0], rects[1]) or bbox_contains(rects[1], rects[0]):
            return False
        return (
            counts["erase"] >= 1
            and counts["recolor_0"] >= 1
            and counts["recolor_1"] >= 1
            and counts["cross_0"] >= 1
            and counts["cross_1"] >= 1
            and counts["cross_0-1"] >= 1
            and counts["outside"] >= 1
            and int(only0.sum()) >= 6
            and int(only1.sum()) >= 6
        )

    try:
        outer_idx, inner_idx = detect_nested_roles(rects)
    except AssertionError:
        return False
    only_outer = interiors[outer_idx] & ~interiors[inner_idx]
    only_inner = interiors[inner_idx] & ~interiors[outer_idx]
    return (
        counts["erase"] >= 1
        and counts[f"recolor_{outer_idx}"] >= 1
        and counts[f"cross_{inner_idx}"] >= 1
        and counts[f"cross_{outer_idx}"] >= 1
        and counts["outside"] >= 1
        and int(only_outer.sum()) >= 8
        and int(only_inner.sum()) == 0
    )


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    mode = GENERATION_MODES[_GEN_COUNTER % len(GENERATION_MODES)]
    variant = _GEN_COUNTER
    for _ in range(GENERATION_RETRIES):
        grid = gen_attempt(mode, variant)
        if grid is not None:
            _GEN_COUNTER += 1
            return grid
    raise RuntimeError(f"failed to generate valid input for mode={mode}")
