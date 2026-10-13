import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
TEAL = 8
MAROON = 9

INPUT_COLORS = (BLUE, GREEN, GREY, MAROON)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
GEN_ATTEMPTS = 600


def neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def neighbors8(r: int, c: int):
    for dr, dc in DIR8:
        yield r + dr, c + dc


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def bbox_of(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def color_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BLACK or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": color, "coords": coords, "bbox": bbox_of(coords)})
    return comps


def enclosed_holes(shape: tuple[int, int], coords: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    r0, c0, r1, c1 = bbox_of(coords)
    hh = r1 - r0 + 1
    ww = c1 - c0 + 1
    mask = np.zeros((hh, ww), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True

    empty = ~mask
    seen = np.zeros_like(empty, dtype=bool)
    holes = []
    for rr in range(hh):
        for cc in range(ww):
            if not empty[rr, cc] or seen[rr, cc]:
                continue
            q = deque([(rr, cc)])
            seen[rr, cc] = True
            region = []
            touches_border = False
            while q:
                cr, cc2 = q.popleft()
                region.append((cr, cc2))
                if cr in (0, hh - 1) or cc2 in (0, ww - 1):
                    touches_border = True
                for nr, nc in neighbors4(cr, cc2):
                    if 0 <= nr < hh and 0 <= nc < ww and empty[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_border:
                holes.append([(r0 + rr2, c0 + cc3) for rr2, cc3 in region])
    return holes


def outline_black_neighbors(out: np.ndarray, solid: set[tuple[int, int]]) -> None:
    h, w = out.shape
    to_paint = set()
    for r, c in solid:
        for nr, nc in neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(out[nr, nc]) == BLACK:
                to_paint.add((nr, nc))
    for r, c in to_paint:
        out[r, c] = RED


def transform_before_fill(
    grid: np.ndarray,
    *,
    fill_holes: bool = True,
    recolor_hollow: bool = True,
    draw_outline: bool = True,
) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    comps = color_components(g)
    infos = []

    for comp in comps:
        coords = comp["coords"]
        holes = enclosed_holes(g.shape, coords)
        has_hole = bool(holes)
        solid = set(coords)
        if has_hole and recolor_hollow:
            for r, c in coords:
                out[r, c] = TEAL
        for hole in holes:
            if fill_holes:
                for r, c in hole:
                    out[r, c] = MAGENTA
                    solid.add((r, c))
        infos.append(
            {
                "color": comp["color"],
                "coords": coords,
                "holes": holes,
                "has_hole": has_hole,
                "solid": solid,
            }
        )

    if draw_outline:
        for info in infos:
            outline_black_neighbors(out, info["solid"])

    return out, infos


def all_black_rectangles(grid: np.ndarray, min_h: int = 3, min_w: int = 3) -> list[tuple[int, int, int, int, int]]:
    mask = (grid == BLACK).astype(int)
    h, w = grid.shape
    prefix = np.pad(mask, ((1, 0), (1, 0)), constant_values=0).cumsum(0).cumsum(1)
    rects = []
    for r0 in range(h):
        for c0 in range(w):
            for r1 in range(r0 + min_h - 1, h):
                for c1 in range(c0 + min_w - 1, w):
                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    total = prefix[r1 + 1, c1 + 1] - prefix[r0, c1 + 1] - prefix[r1 + 1, c0] + prefix[r0, c0]
                    if total == area:
                        rects.append((area, r0, c0, r1, c1))
    rects.sort(key=lambda item: (-item[0], item[1], item[2], item[3], item[4]))
    return rects


def choose_largest_rectangle(grid: np.ndarray) -> tuple[int, int, int, int] | None:
    rects = all_black_rectangles(grid)
    if not rects:
        return None
    _, r0, c0, r1, c1 = rects[0]
    return r0, c0, r1, c1


def fill_rectangle(grid: np.ndarray, rect: tuple[int, int, int, int] | None) -> np.ndarray:
    out = grid.copy()
    if rect is None:
        return out
    r0, c0, r1, c1 = rect
    out[r0 : r1 + 1, c0 : c1 + 1] = YELLOW
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor hollow objects, outline every object, then fill the largest remaining black rectangle yellow.

    Input structure:
    - A black background contains 3-4 non-touching 4-connected foreground objects in distinct colors.
    - At least one object encloses a 4-connected black hole inside its bounding box, and at least one object is solid.
    - After processing the objects, at least one all-black rectangle of size at least 3x3 remains somewhere in the grid.

    Rule:
    - For each foreground object, detect black 4-connected regions inside its bounding box that do not touch that box border.
    - If an object has such enclosed holes, recolor the object to teal (8) and fill the hole cells with magenta (6); solid objects keep their original color.
    - Treat the recolored object together with its filled holes as one solid region, and paint every currently black 8-neighbor of that region red (2).
    - Among all remaining all-black rectangles with height and width at least 3, select the one with maximum area; ties go to the smallest top row, then the smallest left column.
    - Fill that selected rectangle yellow (4).

    Key insight:
    - The final yellow region is determined only after hole handling and red outlining shrink the black background, so the solver must first model object holes and outlines and only then search for the maximal empty rectangle.
    """
    base, _ = transform_before_fill(np.asarray(grid, dtype=int))
    rect = choose_largest_rectangle(base)
    return fill_rectangle(base, rect)


def normalize(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return {(r - r0, c - c0) for r, c in coords}


def transform_shape(coords: set[tuple[int, int]], rotation: int, flip: bool) -> set[tuple[int, int]]:
    pts = np.array(sorted(coords), dtype=int)
    if pts.size == 0:
        return set()
    if flip:
        pts[:, 1] *= -1
    for _ in range(rotation % 4):
        pts = np.stack([pts[:, 1], -pts[:, 0]], axis=1)
    out = {(int(r), int(c)) for r, c in pts}
    return normalize(out)


def random_polyomino(rng: random.Random, target_area: int) -> set[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < target_area:
        r, c = rng.choice(tuple(cells))
        dr, dc = rng.choice(DIR4)
        cells.add((r + dr, c + dc))
    return normalize(cells)


def make_bar(rng: random.Random, orientation: str) -> set[tuple[int, int]]:
    if orientation == "h":
        thickness = 1 if rng.random() < 0.7 else 2
        length = rng.randint(8, 14 if thickness == 1 else 12)
        cells = {(r, c) for r in range(thickness) for c in range(length)}
        if thickness == 1 and rng.random() < 0.6:
            nub_c = rng.randint(1, length - 2)
            cells.add((1, nub_c))
    else:
        thickness = 1 if rng.random() < 0.7 else 2
        length = rng.randint(8, 14 if thickness == 1 else 12)
        cells = {(r, c) for r in range(length) for c in range(thickness)}
        if thickness == 1 and rng.random() < 0.6:
            nub_r = rng.randint(1, length - 2)
            cells.add((nub_r, 1))
    return normalize(cells)


def make_l_shape(rng: random.Random) -> set[tuple[int, int]]:
    arm_r = rng.randint(5, 8)
    arm_c = rng.randint(5, 8)
    thick = 1 if rng.random() < 0.75 else 2
    cells = set()
    for r in range(arm_r):
        for c in range(thick):
            cells.add((r, c))
    for r in range(thick):
        for c in range(arm_c):
            cells.add((r, c))
    if thick == 1 and rng.random() < 0.7:
        ext = rng.randint(1, max(1, arm_r - 2))
        cells.add((ext, 1))
    return normalize(cells)


def make_hollow_shape(rng: random.Random) -> set[tuple[int, int]]:
    hole_dims = [(1, 2), (2, 1), (1, 3), (3, 1), (2, 2), (2, 3), (3, 2)]
    for _ in range(200):
        hole_h, hole_w = rng.choice(hole_dims)
        pad_t = rng.randint(1, 2)
        pad_b = rng.randint(1, 2)
        pad_l = rng.randint(1, 2)
        pad_r = rng.randint(1, 2)
        outer_h = hole_h + pad_t + pad_b
        outer_w = hole_w + pad_l + pad_r
        if outer_h > 7 or outer_w > 7:
            continue
        cells = {(r, c) for r in range(outer_h) for c in range(outer_w)}
        for r in range(pad_t, pad_t + hole_h):
            for c in range(pad_l, pad_l + hole_w):
                cells.remove((r, c))
        side = rng.choice(["top", "bottom", "left", "right", "none"])
        if side != "none":
            if side in ("top", "bottom"):
                width = rng.randint(1, max(1, outer_w - 2))
                start = rng.randint(1, outer_w - width - 1)
                rr = -1 if side == "top" else outer_h
                for cc in range(start, start + width):
                    cells.add((rr, cc))
            else:
                height = rng.randint(1, max(1, outer_h - 2))
                start = rng.randint(1, outer_h - height - 1)
                cc = -1 if side == "left" else outer_w
                for rr in range(start, start + height):
                    cells.add((rr, cc))
        cells = normalize(cells)
        if 8 <= len(cells) <= 25 and is_4_connected(cells):
            holes = enclosed_holes((40, 40), [(r + 10, c + 10) for r, c in cells])
            if len(holes) == 1 and 2 <= len(holes[0]) <= 6:
                return cells
    raise RuntimeError("failed to build hollow shape")


def is_4_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return seen == cells


def place_shape(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    coords: set[tuple[int, int]],
    color: int,
    top: int,
    left: int,
) -> set[tuple[int, int]] | None:
    placed = {(top + r, left + c) for r, c in coords}
    h, w = grid.shape
    for r, c in placed:
        if not (0 <= r < h and 0 <= c < w):
            return None
    expanded = set()
    for r, c in placed:
        expanded.add((r, c))
        for nr, nc in neighbors8(r, c):
            expanded.add((nr, nc))
    if any((r, c) in occupied for r, c in expanded):
        return None
    for r, c in placed:
        grid[r, c] = color
    return placed


def try_place_role(
    rng: random.Random,
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    shape: set[tuple[int, int]],
    color: int,
    role: str,
) -> bool:
    h, w = grid.shape
    shape_h = max(r for r, _ in shape) + 1
    shape_w = max(c for _, c in shape) + 1
    ranges = []
    if role == "top":
        ranges = [(0, max(0, h // 3), 0, max(0, w - shape_w))]
    elif role == "bottom":
        ranges = [(max(0, 2 * h // 3 - shape_h), h - shape_h, 0, max(0, w - shape_w))]
    elif role == "left":
        ranges = [(0, max(0, h - shape_h), 0, max(0, w // 3))]
    elif role == "right":
        ranges = [(0, max(0, h - shape_h), max(0, 2 * w // 3 - shape_w), w - shape_w)]
    elif role == "center":
        ranges = [(max(0, h // 4), max(0, 3 * h // 4 - shape_h), max(0, w // 4), max(0, 3 * w // 4 - shape_w))]
    elif role == "corner":
        ranges = [
            (0, max(0, h // 4), 0, max(0, w // 4)),
            (0, max(0, h // 4), max(0, 3 * w // 4 - shape_w), w - shape_w),
            (max(0, 3 * h // 4 - shape_h), h - shape_h, 0, max(0, w // 4)),
            (max(0, 3 * h // 4 - shape_h), h - shape_h, max(0, 3 * w // 4 - shape_w), w - shape_w),
        ]
    else:
        ranges = [(0, h - shape_h, 0, w - shape_w)]

    for _ in range(100):
        r0a, r0b, c0a, c0b = rng.choice(ranges)
        if r0a > r0b or c0a > c0b:
            continue
        top = rng.randint(r0a, r0b)
        left = rng.randint(c0a, c0b)
        placed = place_shape(grid, occupied, shape, color, top, left)
        if placed is not None:
            occupied.update(placed)
            return True
    return False


def choose_roles(mode: str, count: int) -> list[str]:
    if mode == "tie":
        base = ["top", "left", "right", "bottom"]
    else:
        base = ["top", "left", "center", "right"]
    return base[:count]


def build_candidate_grid(rng: random.Random, mode: str) -> np.ndarray | None:
    h = rng.randint(18, 24)
    w = rng.randint(18, 24)
    grid = np.zeros((h, w), dtype=int)
    occupied: set[tuple[int, int]] = set()

    count = 4 if rng.random() < 0.7 else 3
    colors = list(INPUT_COLORS)
    rng.shuffle(colors)
    roles = choose_roles(mode, count)

    shapes: list[set[tuple[int, int]]] = [make_hollow_shape(rng)]
    solid_bank = [
        make_bar(rng, "h"),
        make_bar(rng, "v"),
        make_l_shape(rng),
        random_polyomino(rng, rng.randint(8, 18)),
    ]
    rng.shuffle(solid_bank)
    shapes.extend(solid_bank[: count - 1])

    oriented = []
    for shape in shapes:
        oriented.append(transform_shape(shape, rng.randint(0, 3), rng.random() < 0.5))

    role_order = [("center", 0)]
    remaining_roles = roles[:]
    if "center" in remaining_roles:
        remaining_roles.remove("center")
    for idx in range(1, count):
        role_order.append((remaining_roles[idx - 1] if idx - 1 < len(remaining_roles) else "any", idx))

    for role, idx in role_order:
        if not try_place_role(rng, grid, occupied, oriented[idx], colors[idx], role):
            return None

    return grid


def make_top_t(width: int, stem_end: int) -> set[tuple[int, int]]:
    center = width // 2
    cells = {(0, c) for c in range(width)}
    for r in range(stem_end + 1):
        cells.add((r, center))
    return cells


def make_center_column(height: int, thickness: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(height) for c in range(thickness)}


def build_tie_grid(rng: random.Random) -> np.ndarray | None:
    w = rng.choice([17, 19])
    h = rng.randint(18, 24)
    stem_end = rng.randint(4, min(6, 25 - w))
    top_t = make_top_t(w, stem_end)
    hollow = transform_shape(make_hollow_shape(rng), rng.randint(0, 1), False)
    hollow_h = max(r for r, _ in hollow) + 1
    hollow_w = max(c for _, c in hollow) + 1
    hollow_top = stem_end + 2
    if hollow_top + hollow_h + 4 >= h:
        return None

    col_thickness = 1 if rng.random() < 0.65 else 2
    col_height = h - (hollow_top + hollow_h + 2)
    if not (8 <= col_height * col_thickness <= 25):
        return None
    center_col = make_center_column(col_height, col_thickness)

    grid = np.zeros((h, w), dtype=int)
    occupied: set[tuple[int, int]] = set()
    colors = list(INPUT_COLORS)
    rng.shuffle(colors)

    if place_shape(grid, occupied, top_t, colors[0], 0, 0) is None:
        return None
    occupied.update({(r, c) for r, c in np.argwhere(grid == colors[0])})

    hollow_left = (w - hollow_w) // 2
    placed = place_shape(grid, occupied, hollow, colors[1], hollow_top, hollow_left)
    if placed is None:
        return None
    occupied.update(placed)

    col_left = w // 2 - (col_thickness - 1) // 2
    col_top = hollow_top + hollow_h + 2
    placed = place_shape(grid, occupied, center_col, colors[2], col_top, col_left)
    if placed is None:
        return None
    occupied.update(placed)

    return grid


def selected_rectangle_from_variant(
    grid: np.ndarray,
    *,
    fill_holes: bool = True,
    recolor_hollow: bool = True,
    draw_outline: bool = True,
) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    base, _ = transform_before_fill(grid, fill_holes=fill_holes, recolor_hollow=recolor_hollow, draw_outline=draw_outline)
    return base, choose_largest_rectangle(base)


def validate_grid(grid: np.ndarray, mode: str) -> bool:
    comps = color_components(grid)
    if len(comps) not in (3, 4):
        return False
    if len({comp["color"] for comp in comps}) != len(comps):
        return False
    if any(not (8 <= len(comp["coords"]) <= 25) for comp in comps):
        return False

    hole_counts = []
    for comp in comps:
        holes = enclosed_holes(grid.shape, comp["coords"])
        hole_counts.append(len(holes))
    if not any(count > 0 for count in hole_counts):
        return False
    if not any(count == 0 for count in hole_counts):
        return False

    base, infos = transform_before_fill(grid)
    rects = all_black_rectangles(base)
    if not rects:
        return False
    max_area = rects[0][0]
    if max_area < 9:
        return False
    max_rects = [rect for rect in rects if rect[0] == max_area]
    if mode == "tie":
        if len(max_rects) < 2:
            return False
    else:
        if len(max_rects) != 1:
            return False
    if len(rects) < 2:
        return False

    final_rect = (rects[0][1], rects[0][2], rects[0][3], rects[0][4])

    no_outline_grid, no_outline_rect = selected_rectangle_from_variant(
        grid, fill_holes=True, recolor_hollow=True, draw_outline=False
    )
    if no_outline_rect == final_rect:
        return False
    if all_black_rectangles(no_outline_grid) and all_black_rectangles(no_outline_grid)[0][0] <= max_area:
        return False

    output = main(grid)
    if not np.any(output == YELLOW):
        return False
    if not np.any(output == RED):
        return False
    if not np.any(output == TEAL):
        return False
    if not np.any(output == MAGENTA):
        return False

    for info in infos:
        if info["has_hole"]:
            for hole in info["holes"]:
                if not all(int(output[r, c]) == MAGENTA for r, c in hole):
                    return False
        else:
            orig = info["color"]
            if not any(int(output[r, c]) == orig for r, c in info["coords"]):
                return False

    ymask = output == YELLOW
    ys = np.argwhere(ymask)
    r0, c0 = ys.min(axis=0)
    r1, c1 = ys.max(axis=0)
    if not np.all(ymask[r0 : r1 + 1, c0 : c1 + 1]):
        return False
    if (r1 - r0 + 1) < 3 or (c1 - c0 + 1) < 3:
        return False
    if (r0, c0, r1, c1) != final_rect:
        return False

    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    mode = "tie" if rng.random() < 0.35 else "unique"
    for _ in range(GEN_ATTEMPTS):
        grid = build_tie_grid(rng) if mode == "tie" else build_candidate_grid(rng, mode)
        if grid is None:
            continue
        if validate_grid(grid, mode):
            return grid
    raise RuntimeError("failed to generate a valid input")
