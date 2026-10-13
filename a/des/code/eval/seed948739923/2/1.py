import random
from collections import Counter

import numpy as np


DIR_TO_VEC = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

SIDE_ORDER = ("top", "right", "bottom", "left")

SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
)

PALETTE = tuple(range(10))
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    pts = set()
    for c in range(c0, c1 + 1):
        pts.add((r0, c))
        pts.add((r1, c))
    for r in range(r0, r1 + 1):
        pts.add((r, c0))
        pts.add((r, c1))
    return pts


def _detect_frame(grid: np.ndarray, comp: list[tuple[int, int]], bg: int):
    r0, c0, r1, c1 = _bbox(comp)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 7 or w < 7:
        return None

    cells = set(comp)
    if any((r not in (r0, r1)) and (c not in (c0, c1)) for r, c in cells):
        return None

    perimeter = _perimeter_cells(r0, c0, r1, c1)
    if len(cells) != len(perimeter) - 4:
        return None

    missing = perimeter - cells
    if len(missing) != 4:
        return None

    gaps_by_side = {}
    side_cells = {
        "top": [pt for pt in missing if pt[0] == r0],
        "bottom": [pt for pt in missing if pt[0] == r1],
        "left": [pt for pt in missing if pt[1] == c0],
        "right": [pt for pt in missing if pt[1] == c1],
    }
    for side, pts in side_cells.items():
        if len(pts) != 1:
            return None
        gr, gc = pts[0]
        if (gr, gc) in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
            return None
        gaps_by_side[side] = pts[0]

    border_color = Counter(int(grid[r, c]) for r, c in comp).most_common(1)[0][0]

    def marker_color_for(side: str):
        gr, gc = gaps_by_side[side]
        if side == "top":
            pts = [(gr, gc - 1), (gr, gc + 1)]
        elif side == "bottom":
            pts = [(gr, gc - 1), (gr, gc + 1)]
        elif side == "left":
            pts = [(gr - 1, gc), (gr + 1, gc)]
        else:
            pts = [(gr - 1, gc), (gr + 1, gc)]
        vals = [int(grid[r, c]) for r, c in pts]
        if len(set(vals)) != 1:
            return None
        val = vals[0]
        if val in (bg, border_color):
            return None
        return val

    markers = {}
    for side in SIDE_ORDER:
        color = marker_color_for(side)
        if color is None:
            return None
        markers[side] = color
    if len(set(markers.values())) != 4:
        return None

    allowed_colors = {border_color, *markers.values()}
    if any(int(grid[r, c]) not in allowed_colors for r, c in comp):
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "border_color": int(border_color),
        "markers": markers,
    }


def _object_sort_key(obj, dr: int, dc: int):
    rows = [r for r, _, _ in obj]
    cols = [c for _, c, _ in obj]
    if dr == -1:
        return min(rows), min(cols)
    if dr == 1:
        return -max(rows), min(cols)
    if dc == -1:
        return min(cols), min(rows)
    return -max(cols), min(rows)


def _slide_objects(
    objects: list[list[tuple[int, int, int]]],
    top: int,
    left: int,
    bottom: int,
    right: int,
    dr: int,
    dc: int,
):
    ordered = sorted(objects, key=lambda obj: _object_sort_key(obj, dr, dc))
    placed: list[list[tuple[int, int, int]]] = []
    occupied: set[tuple[int, int]] = set()
    moved = False

    for obj in ordered:
        step = 0
        while True:
            next_cells = []
            ok = True
            for r, c, color in obj:
                rr = r + dr * (step + 1)
                cc = c + dc * (step + 1)
                if rr < top or rr > bottom or cc < left or cc > right or (rr, cc) in occupied:
                    ok = False
                    break
                next_cells.append((rr, cc, color))
            if not ok:
                break
            step += 1
        if step:
            moved = True
        final_cells = [(r + dr * step, c + dc * step, color) for r, c, color in obj]
        occupied.update((r, c) for r, c, _ in final_cells)
        placed.append(final_cells)

    return placed, moved


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the interior singleton seed color to select one border gap direction,
    then slide every interior object toward that side and stack them against
    the inner wall. Finally restore the whole frame border.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    for comp in _components(g != bg):
        frame = _detect_frame(g, comp, bg)
        if frame is None:
            continue

        r0, c0, r1, c1 = frame["bbox"]
        marker_to_dir = {color: DIR_TO_VEC[side] for side, color in frame["markers"].items()}
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        objects = []
        seed_colors = []

        for obj in _components(interior != bg):
            abs_obj = []
            for rr, cc in obj:
                color = int(interior[rr, cc])
                abs_obj.append((r0 + 1 + rr, c0 + 1 + cc, color))
            objects.append(abs_obj)
            if len(obj) == 1:
                color = abs_obj[0][2]
                if color in marker_to_dir:
                    seed_colors.append(color)

        assert len(seed_colors) == 1, "frame must contain exactly one seed singleton"
        dr, dc = marker_to_dir[seed_colors[0]]
        moved_objects, _ = _slide_objects(objects, r0 + 1, c0 + 1, r1 - 1, c1 - 1, dr, dc)

        out[r0 + 1 : r1, c0 + 1 : c1] = bg
        for c in range(c0, c1 + 1):
            out[r0, c] = frame["border_color"]
            out[r1, c] = frame["border_color"]
        for r in range(r0, r1 + 1):
            out[r, c0] = frame["border_color"]
            out[r, c1] = frame["border_color"]
        for obj in moved_objects:
            for r, c, color in obj:
                out[r, c] = color

    return out


def _can_place(
    cells: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    ih: int,
    iw: int,
) -> bool:
    for r, c in cells:
        if not (1 <= r <= ih - 2 and 1 <= c <= iw - 2):
            return False
        if (r, c) in occupied:
            return False
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in occupied:
                return False
    return True


def _place_shape(
    rng: random.Random,
    shape: tuple[tuple[int, int], ...],
    occupied: set[tuple[int, int]],
    ih: int,
    iw: int,
) -> list[tuple[int, int]] | None:
    max_r = max(r for r, _ in shape)
    max_c = max(c for _, c in shape)
    if ih < max_r + 3 or iw < max_c + 3:
        return None

    for _ in range(80):
        base_r = rng.randint(1, ih - max_r - 2)
        base_c = rng.randint(1, iw - max_c - 2)
        cells = [(base_r + dr, base_c + dc) for dr, dc in shape]
        if _can_place(cells, occupied, ih, iw):
            occupied.update(cells)
            return cells
    return None


def _frame_patch(
    rng: random.Random,
    outer_h: int,
    outer_w: int,
    border_color: int,
    bg: int,
):
    patch = np.full((outer_h, outer_w), bg, dtype=int)
    patch[0, :] = border_color
    patch[-1, :] = border_color
    patch[:, 0] = border_color
    patch[:, -1] = border_color

    marker_colors = rng.sample([c for c in PALETTE if c not in (bg, border_color)], 4)
    markers = dict(zip(SIDE_ORDER, marker_colors))
    gaps = {
        "top": (0, rng.randint(2, outer_w - 3)),
        "bottom": (outer_h - 1, rng.randint(2, outer_w - 3)),
        "left": (rng.randint(2, outer_h - 3), 0),
        "right": (rng.randint(2, outer_h - 3), outer_w - 1),
    }

    for side, color in markers.items():
        gr, gc = gaps[side]
        patch[gr, gc] = bg
        if side in ("top", "bottom"):
            patch[gr, gc - 1] = color
            patch[gr, gc + 1] = color
        else:
            patch[gr - 1, gc] = color
            patch[gr + 1, gc] = color

    direction = rng.choice(SIDE_ORDER)
    seed_color = markers[direction]
    ih = outer_h - 2
    iw = outer_w - 2
    safe_colors = [c for c in PALETTE if c not in {bg, border_color, *marker_colors}]
    occupied: set[tuple[int, int]] = set()

    total_objects = rng.randint(2, 5)
    specs = [{"shape": ((0, 0),), "color": seed_color}]
    for _ in range(total_objects - 1):
        specs.append({"shape": rng.choice(SHAPES), "color": rng.choice(safe_colors)})
    rng.shuffle(specs)

    objects = []
    for spec in specs:
        placed = _place_shape(rng, spec["shape"], occupied, ih, iw)
        if placed is None:
            return None
        objects.append([(r, c, spec["color"]) for r, c in placed])

    dr, dc = DIR_TO_VEC[direction]
    moved_objects, moved = _slide_objects(objects, 0, 0, ih - 1, iw - 1, dr, dc)
    if not moved:
        return None

    for obj in objects:
        for r, c, color in obj:
            patch[r + 1, c + 1] = color

    return patch


def _place_frames(
    rng: random.Random,
    grid: np.ndarray,
    frame_count: int,
    border_color: int,
    bg: int,
) -> bool:
    h, w = grid.shape
    reserved = np.zeros_like(grid, dtype=bool)

    for _ in range(frame_count):
        placed = False
        for _ in range(160):
            outer_h = rng.randint(9, 13)
            outer_w = rng.randint(9, 13)
            if outer_h >= h - 2 or outer_w >= w - 2:
                continue
            patch = _frame_patch(rng, outer_h, outer_w, border_color, bg)
            if patch is None:
                continue
            r0 = rng.randint(1, h - outer_h - 1)
            c0 = rng.randint(1, w - outer_w - 1)
            rr0 = max(0, r0 - 2)
            cc0 = max(0, c0 - 2)
            rr1 = min(h, r0 + outer_h + 2)
            cc1 = min(w, c0 + outer_w + 2)
            if np.any(reserved[rr0:rr1, cc0:cc1]):
                continue
            grid[r0 : r0 + outer_h, c0 : c0 + outer_w] = patch
            reserved[rr0:rr1, cc0:cc1] = True
            placed = True
            break
        if not placed:
            return False

    return True


def _valid_background_dominance(grid: np.ndarray, bg: int) -> bool:
    bg_count = int(np.count_nonzero(grid == bg))
    for color in np.unique(grid):
        if int(color) == bg:
            continue
        if int(np.count_nonzero(grid == color)) >= bg_count:
            return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        h = rng.randint(22, 30)
        w = rng.randint(22, 30)
        bg = rng.choice(PALETTE)
        border_color = rng.choice([c for c in PALETTE if c != bg])
        frame_count = rng.randint(1, 3)

        grid = np.full((h, w), bg, dtype=int)
        if not _place_frames(rng, grid, frame_count, border_color, bg):
            continue
        if not _valid_background_dominance(grid, bg):
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle input")
