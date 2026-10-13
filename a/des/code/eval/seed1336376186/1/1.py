import random
from collections import defaultdict, deque

import numpy as np


BG_COLOR = 0
FRAME_COLOR = 5
OVERLAP_COLOR = 9
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ("top", "bottom", "left", "right")


def _find_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    frame_pixels = np.argwhere(grid == FRAME_COLOR)
    if frame_pixels.size == 0:
        raise ValueError("missing grey frame")
    r0, c0 = frame_pixels.min(axis=0)
    r1, c1 = frame_pixels.max(axis=0)
    if r1 - r0 < 2 or c1 - c0 < 2:
        raise ValueError("frame is too small")
    expected = np.zeros_like(grid, dtype=bool)
    expected[r0, c0 : c1 + 1] = True
    expected[r1, c0 : c1 + 1] = True
    expected[r0 : r1 + 1, c0] = True
    expected[r0 : r1 + 1, c1] = True
    if not np.array_equal(grid == FRAME_COLOR, expected):
        raise ValueError("grey pixels must form exactly one rectangular frame")
    return int(r0), int(c0), int(r1), int(c1)


def _inside_frame(frame: tuple[int, int, int, int], r: int, c: int) -> bool:
    r0, c0, r1, c1 = frame
    return r0 < r < r1 and c0 < c < c1


def _outside_frame(frame: tuple[int, int, int, int], r: int, c: int) -> bool:
    r0, c0, r1, c1 = frame
    return r < r0 or r > r1 or c < c0 or c > c1


def _find_components(grid: np.ndarray, frame: tuple[int, int, int, int]) -> list[dict]:
    r0, c0, r1, c1 = frame
    seen = np.zeros_like(grid, dtype=bool)
    components = []
    for sr in range(r0 + 1, r1):
        for sc in range(c0 + 1, c1):
            color = int(grid[sr, sc])
            if color in (BG_COLOR, FRAME_COLOR) or seen[sr, sc]:
                continue
            q = deque([(sr, sc)])
            seen[sr, sc] = True
            coords = []
            while q:
                r, c = q.popleft()
                coords.append((int(r), int(c)))
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if not (r0 < nr < r1 and c0 < nc < c1):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            components.append({"color": color, "coords": coords})
    return components


def _contact_side(coords: list[tuple[int, int]], frame: tuple[int, int, int, int]) -> tuple[str, tuple[int, int]]:
    r0, c0, r1, c1 = frame
    touches = {
        "top": [(r, c) for r, c in coords if r == r0 + 1],
        "bottom": [(r, c) for r, c in coords if r == r1 - 1],
        "left": [(r, c) for r, c in coords if c == c0 + 1],
        "right": [(r, c) for r, c in coords if c == c1 - 1],
    }
    active = [(side, pts) for side, pts in touches.items() if pts]
    if len(active) != 1 or len(active[0][1]) != 1:
        raise ValueError("each block must touch exactly one inner wall at exactly one pixel")
    side, pts = active[0]
    return side, pts[0]


def _mirror_coords(
    coords: list[tuple[int, int]], side: str, frame: tuple[int, int, int, int]
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame
    if side == "top":
        return [(2 * r0 - r, c) for r, c in coords]
    if side == "bottom":
        return [(2 * r1 - r, c) for r, c in coords]
    if side == "left":
        return [(r, 2 * c0 - c) for r, c in coords]
    return [(r, 2 * c1 - c) for r, c in coords]


def _extension_coords(
    coords: list[tuple[int, int]], side: str, frame: tuple[int, int, int, int]
) -> list[tuple[int, int]]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    mirror = _mirror_coords(coords, side, frame)
    mirror_rows = [r for r, _ in mirror]
    mirror_cols = [c for _, c in mirror]
    if side in ("top", "bottom"):
        span = max_c - min_c + 1
        if side == "top":
            start = min(mirror_rows) - span
            end = min(mirror_rows)
            return [(r, c) for r in range(start, end) for c in range(min_c, max_c + 1)]
        start = max(mirror_rows) + 1
        end = max(mirror_rows) + span + 1
        return [(r, c) for r in range(start, end) for c in range(min_c, max_c + 1)]
    span = max_r - min_r + 1
    if side == "left":
        start = min(mirror_cols) - span
        end = min(mirror_cols)
        return [(r, c) for r in range(min_r, max_r + 1) for c in range(start, end)]
    start = max(mirror_cols) + 1
    end = max(mirror_cols) + span + 1
    return [(r, c) for r in range(min_r, max_r + 1) for c in range(start, end)]


def _component_geometry(component: dict, frame: tuple[int, int, int, int]) -> dict:
    coords = component["coords"]
    side, contact = _contact_side(coords, frame)
    mirror = _mirror_coords(coords, side, frame)
    extension = _extension_coords(coords, side, frame)
    return {
        "color": int(component["color"]),
        "coords": list(coords),
        "side": side,
        "contact": contact,
        "mirror": mirror,
        "extension": extension,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Grow mirrored exterior branches from frame-touching interior blocks."""
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("grid must be 2D")

    frame = _find_frame(g)
    components = _find_components(g, frame)
    out = g.copy()
    h, w = g.shape

    mirror_claims: dict[tuple[int, int], set[int]] = defaultdict(set)
    ext_claims: dict[tuple[int, int], set[int]] = defaultdict(set)

    for component in components:
        info = _component_geometry(component, frame)
        for cell in info["mirror"]:
            r, c = cell
            if not (0 <= r < h and 0 <= c < w):
                raise ValueError("mirrored branch leaves the canvas")
            mirror_claims[cell].add(info["color"])
        for cell in info["extension"]:
            r, c = cell
            if not (0 <= r < h and 0 <= c < w):
                raise ValueError("extended branch leaves the canvas")
            ext_claims[cell].add(info["color"])

    for (r, c), colors in mirror_claims.items():
        out[r, c] = OVERLAP_COLOR if len(colors) > 1 else next(iter(colors))

    for (r, c), colors in ext_claims.items():
        value = OVERLAP_COLOR if len(colors) > 1 else next(iter(colors))
        current = int(out[r, c])
        if current in (BG_COLOR, value, OVERLAP_COLOR):
            out[r, c] = value if current != OVERLAP_COLOR else OVERLAP_COLOR
        else:
            out[r, c] = OVERLAP_COLOR

    return out


SHAPES = {
    "v2": ((0, 0), (1, 0)),
    "hook_r": ((0, 0), (1, 0), (1, 1)),
    "hook_l": ((0, 0), (1, -1), (1, 0)),
    "T4": ((0, 0), (1, -1), (1, 0), (1, 1)),
    "L4r": ((0, 0), (1, 0), (2, 0), (2, 1)),
    "L4l": ((0, 0), (1, 0), (2, -1), (2, 0)),
    "step_r": ((0, 0), (1, 0), (1, 1), (2, 1)),
    "step_l": ((0, 0), (1, -1), (1, 0), (2, -1)),
    "fan5": ((0, 0), (1, 0), (2, -1), (2, 0), (2, 1)),
    "sweep_r": ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
    "sweep_l": ((0, 0), (1, 0), (2, -2), (2, -1), (2, 0)),
    "cross5": ((0, 0), (1, -1), (1, 0), (1, 1), (2, 0)),
}

OVERLAP_TEMPLATES = [
    ("step_l", "sweep_l", 2),
    ("sweep_r", "step_r", 2),
    ("cross5", "sweep_l", 3),
    ("sweep_r", "cross5", 3),
]

COLOR_POOL = (1, 2, 3, 4, 6, 7, 8)
GEN_RETRIES = 300
OBJECT_COUNT_RANGE = (3, 6)
INNER_SIZE_RANGE = (8, 12)
MARGIN_RANGE = (6, 9)
PLACE_TRIES = 80


def _actual_cells(
    side: str,
    frame: tuple[int, int, int, int],
    anchor: int,
    shape: tuple[tuple[int, int], ...],
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame
    coords = []
    for depth, offset in shape:
        if side == "top":
            coords.append((r0 + 1 + depth, anchor + offset))
        elif side == "bottom":
            coords.append((r1 - 1 - depth, anchor + offset))
        elif side == "left":
            coords.append((anchor + offset, c0 + 1 + depth))
        else:
            coords.append((anchor + offset, c1 - 1 - depth))
    return coords


def _parallel_bounds(
    side: str,
    frame: tuple[int, int, int, int],
    shapes_with_offsets: list[tuple[tuple[tuple[int, int], ...], int]],
) -> tuple[int, int] | None:
    r0, c0, r1, c1 = frame
    if side in ("top", "bottom"):
        low, high = c0 + 2, c1 - 2
        depth_limit = r1 - r0 - 3
    else:
        low, high = r0 + 2, r1 - 2
        depth_limit = c1 - c0 - 3

    min_p = min(offset + p for shape, offset in shapes_with_offsets for _, p in shape)
    max_p = max(offset + p for shape, offset in shapes_with_offsets for _, p in shape)
    max_d = max(depth for shape, _ in shapes_with_offsets for depth, _p in shape)

    if max_d > depth_limit:
        return None

    lo = low - min_p
    hi = high - max_p
    if lo > hi:
        return None
    return int(lo), int(hi)


def _occupied_conflict(candidate: list[tuple[int, int]], occupied: set[tuple[int, int]], gap: bool) -> bool:
    for r, c in candidate:
        if (r, c) in occupied:
            return True
        if not gap:
            continue
        for dr, dc in DIR4:
            if (r + dr, c + dc) in occupied:
                return True
    return False


def _build_component(color: int, coords: list[tuple[int, int]]) -> dict:
    return {"color": int(color), "coords": sorted((int(r), int(c)) for r, c in coords)}


def _component_claims(
    components: list[dict],
    frame: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> tuple[dict[tuple[int, int], set[int]], dict[tuple[int, int], set[int]]] | None:
    h, w = shape
    mirror_claims: dict[tuple[int, int], set[int]] = defaultdict(set)
    ext_claims: dict[tuple[int, int], set[int]] = defaultdict(set)
    original = {cell for component in components for cell in component["coords"]}
    frame_cells = set()
    r0, c0, r1, c1 = frame
    for c in range(c0, c1 + 1):
        frame_cells.add((r0, c))
        frame_cells.add((r1, c))
    for r in range(r0, r1 + 1):
        frame_cells.add((r, c0))
        frame_cells.add((r, c1))

    for component in components:
        info = _component_geometry(component, frame)
        for bucket, cells in ((mirror_claims, info["mirror"]), (ext_claims, info["extension"])):
            for r, c in cells:
                if not (0 <= r < h and 0 <= c < w):
                    return None
                if not _outside_frame(frame, r, c):
                    return None
                if (r, c) in original or (r, c) in frame_cells:
                    return None
                bucket[(r, c)].add(info["color"])

    if set(mirror_claims) & set(ext_claims):
        return None
    if any(len(colors) > 1 for colors in mirror_claims.values()):
        return None
    if any(len(colors) > 2 for colors in ext_claims.values()):
        return None
    return mirror_claims, ext_claims


def _frame_from_layout(
    top_margin: int, left_margin: int, inner_h: int, inner_w: int
) -> tuple[int, int, int, int]:
    return (
        int(top_margin),
        int(left_margin),
        int(top_margin + inner_h + 1),
        int(left_margin + inner_w + 1),
    )


def _candidate_sides(rng: random.Random, count: int) -> list[str]:
    order = list(SIDES)
    rng.shuffle(order)
    sides = order[: min(count, len(order))]
    while len(sides) < count:
        sides.append(rng.choice(SIDES))
    rng.shuffle(sides)
    return sides


def _choose_anchor(
    rng: random.Random,
    side: str,
    frame: tuple[int, int, int, int],
    shape: tuple[tuple[int, int], ...],
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    bounds = _parallel_bounds(side, frame, [(shape, 0)])
    if bounds is None:
        return None
    lo, hi = bounds
    anchors = list(range(lo, hi + 1))
    rng.shuffle(anchors)
    for anchor in anchors[:PLACE_TRIES]:
        coords = _actual_cells(side, frame, anchor, shape)
        if _occupied_conflict(coords, occupied, gap=True):
            continue
        return coords
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    inner_h = rng.randint(*INNER_SIZE_RANGE)
    inner_w = rng.randint(*INNER_SIZE_RANGE)
    top_margin = rng.randint(*MARGIN_RANGE)
    bottom_margin = rng.randint(*MARGIN_RANGE)
    left_margin = rng.randint(*MARGIN_RANGE)
    right_margin = rng.randint(*MARGIN_RANGE)
    frame = _frame_from_layout(top_margin, left_margin, inner_h, inner_w)
    height = top_margin + inner_h + 2 + bottom_margin
    width = left_margin + inner_w + 2 + right_margin

    overlap_mode = rng.random() < 0.5
    object_count = rng.randint(*OBJECT_COUNT_RANGE)
    if overlap_mode:
        object_count = max(4, object_count)
    if object_count > len(COLOR_POOL):
        return None

    colors = list(COLOR_POOL)
    rng.shuffle(colors)
    components: list[dict] = []
    occupied: set[tuple[int, int]] = set()

    if overlap_mode:
        pair_side = rng.choice(SIDES)
        shape_a_name, shape_b_name, delta = rng.choice(OVERLAP_TEMPLATES)
        shape_a = SHAPES[shape_a_name]
        shape_b = SHAPES[shape_b_name]
        pair_bounds = _parallel_bounds(pair_side, frame, [(shape_a, 0), (shape_b, delta)])
        if pair_bounds is None:
            return None
        lo, hi = pair_bounds
        anchors = list(range(lo, hi + 1))
        rng.shuffle(anchors)
        placed_pair = None
        for anchor in anchors[:PLACE_TRIES]:
            coords_a = _actual_cells(pair_side, frame, anchor, shape_a)
            coords_b = _actual_cells(pair_side, frame, anchor + delta, shape_b)
            if set(coords_a) & set(coords_b):
                continue
            placed_pair = (coords_a, coords_b)
            break
        if placed_pair is None:
            return None
        coords_a, coords_b = placed_pair
        components.append(_build_component(colors.pop(), coords_a))
        components.append(_build_component(colors.pop(), coords_b))
        occupied.update(coords_a)
        occupied.update(coords_b)
        extra_sides = [side for side in _candidate_sides(rng, object_count - 2) if side != pair_side]
        while len(extra_sides) < object_count - 2:
            extra_sides.append(rng.choice([side for side in SIDES if side != pair_side]))
    else:
        extra_sides = _candidate_sides(rng, object_count)

    for side in extra_sides:
        placed = False
        shape_names = list(SHAPES)
        rng.shuffle(shape_names)
        for shape_name in shape_names:
            coords = _choose_anchor(rng, side, frame, SHAPES[shape_name], occupied)
            if coords is None:
                continue
            components.append(_build_component(colors.pop(), coords))
            occupied.update(coords)
            placed = True
            break
        if not placed:
            return None

    grid = np.full((height, width), BG_COLOR, dtype=int)
    r0, c0, r1, c1 = frame
    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR
    for component in components:
        for r, c in component["coords"]:
            grid[r, c] = component["color"]

    claims = _component_claims(components, frame, grid.shape)
    if claims is None:
        return None
    _mirror_claims, ext_claims = claims
    overlap_cells = [cell for cell, colors in ext_claims.items() if len(colors) > 1]
    if overlap_mode:
        if not overlap_cells or len(overlap_cells) > 6:
            return None
    elif overlap_cells:
        return None

    side_count = len({_component_geometry(component, frame)["side"] for component in components})
    if side_count < 2:
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if overlap_mode and OVERLAP_COLOR not in set(map(int, out.ravel())):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
