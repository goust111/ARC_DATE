import hashlib
import random
from collections import Counter

import numpy as np


OUTPUT_CYCLE = (2, 8, 5)
SQUARE_COLOR_POOL = (1, 3, 4, 6, 7, 9)
ALL_OBJECT_COLORS = (1, 2, 3, 4, 5, 6, 7, 8, 9)
GEN_ATTEMPTS = 600
_SEEN_OUTPUT_SIGNATURES: set[str] = set()


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def neighbors4(r: int, c: int):
    yield int(r) - 1, int(c)
    yield int(r) + 1, int(c)
    yield int(r), int(c) - 1
    yield int(r), int(c) + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    comps: list[dict] = []
    for color in sorted(set(map(int, g.ravel())) - {int(bg)}):
        mask = g == int(color)
        coords_all = np.argwhere(mask)
        if coords_all.size == 0:
            continue
        seen: set[tuple[int, int]] = set()
        for start_r, start_c in coords_all:
            start = (int(start_r), int(start_c))
            if start in seen:
                continue
            stack = [start]
            seen.add(start)
            pts: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                pts.append((int(r), int(c)))
                for nr, nc in neighbors4(r, c):
                    if in_bounds(g.shape, nr, nc) and int(g[nr, nc]) == int(color) and (int(nr), int(nc)) not in seen:
                        seen.add((int(nr), int(nc)))
                        stack.append((int(nr), int(nc)))
            coords = np.array(pts, dtype=int)
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def corner_marker_coords(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, bbox)
    return [
        (r0 - 1, c0),
        (r0, c0 - 1),
        (r0 - 1, c1),
        (r0, c1 + 1),
        (r1 + 1, c0),
        (r1, c0 - 1),
        (r1 + 1, c1),
        (r1, c1 + 1),
    ]


def is_complete_square(grid: np.ndarray, comp: dict) -> bool:
    color = int(comp["color"])
    r0, c0, r1, c1 = map(int, comp["bbox"])
    side_h = int(r1 - r0 + 1)
    side_w = int(c1 - c0 + 1)
    if side_h != side_w or side_h < 2:
        return False
    border = np.zeros((side_h, side_w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    box = grid[r0 : r1 + 1, c0 : c1 + 1]
    return bool(np.all(box[border] == color))


def analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    comps = find_components(g, bg)
    squares = [comp for comp in comps if is_complete_square(g, comp)]
    distractors = [comp for comp in comps if not is_complete_square(g, comp)]
    for comp in squares:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        comp["side"] = int(r1 - r0 + 1)
        comp["box_area"] = int(comp["side"] ** 2)
    return {"bg": int(bg), "components": comps, "squares": squares, "distractors": distractors}


def _sorted_squares(squares: list[dict], *, mode: str = "area") -> list[dict]:
    if mode == "area":
        return sorted(squares, key=lambda comp: (int(comp["box_area"]), int(comp["bbox"][0]), int(comp["bbox"][1])))
    if mode == "scan":
        return sorted(squares, key=lambda comp: (int(comp["bbox"][0]), int(comp["bbox"][1])))
    raise ValueError(mode)


def _render_output(
    grid: np.ndarray,
    squares: list[dict],
    *,
    order_mode: str = "area",
    fill_bbox: bool = True,
    marker_use_new_color: bool = True,
) -> np.ndarray:
    bg = get_bg(grid)
    out = np.full_like(grid, int(bg))
    ordered = _sorted_squares(squares, mode=order_mode)
    for idx, comp in enumerate(ordered):
        new_color = int(OUTPUT_CYCLE[idx % len(OUTPUT_CYCLE)])
        marker_color = new_color if marker_use_new_color else int(comp["color"])
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if fill_bbox:
            out[r0 : r1 + 1, c0 : c1 + 1] = new_color
        else:
            out[comp["coords"][:, 0], comp["coords"][:, 1]] = new_color
        for r, c in corner_marker_coords(comp["bbox"]):
            if in_bounds(out.shape, r, c):
                out[int(r), int(c)] = marker_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter complete square objects, sort them by area, recolor them cyclically, and mark their outside corners.

    Input structure:
    - The background is black.
    - The foreground consists of 5-7 monochrome 4-connected objects in distinct colors.
    - Some objects are complete squares: their bounding boxes are square and every bbox-border cell has the object's color; they may be filled or hollow.
    - The remaining objects are distractors such as non-square bars, broken square fragments, or other irregular blobs.

    Rule:
    - Identify every complete square object and discard every distractor.
    - Sort the complete squares by increasing bounding-box area, breaking ties by top-left row then column.
    - Repaint the sorted squares in a repeating cycle of red (2), teal (8), and grey (5), filling each square's entire bounding box.
    - For each repainted square, paint the two orthogonally adjacent outside cells at each corner with that square's new color.

    Key insight:
    - The outside corner marks use the post-sort repaint color, so the solver must first filter valid squares, then order them by size, and only then determine the final marker colors.
    """
    g = np.asarray(grid, dtype=int)
    info = analyze_grid(g)
    return _render_output(g, info["squares"], order_mode="area", fill_bbox=True, marker_use_new_color=True)


def _shape_square(side: int, *, hollow: bool) -> np.ndarray:
    shape = np.ones((side, side), dtype=bool)
    if hollow and side >= 3:
        shape[1:-1, 1:-1] = False
    return shape


def _shape_broken_square(side: int) -> np.ndarray:
    shape = _shape_square(side, hollow=True)
    if side == 2:
        shape[0, 0] = False
    else:
        candidates = [(0, c) for c in range(1, side - 1)] + [(side - 1, c) for c in range(1, side - 1)]
        candidates += [(r, 0) for r in range(1, side - 1)] + [(r, side - 1) for r in range(1, side - 1)]
        rr, cc = random.choice(candidates)
        shape[rr, cc] = False
    return shape


def _shape_rect(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=bool)


def _shape_l(h: int, w: int) -> np.ndarray:
    shape = np.zeros((h, w), dtype=bool)
    shape[:, 0] = True
    shape[-1, :] = True
    return shape


def _place_shape(
    grid: np.ndarray,
    reserved: np.ndarray,
    shape: np.ndarray,
    color: int,
    *,
    require_margin: bool,
) -> tuple[int, int] | None:
    h, w = map(int, shape.shape)
    gh, gw = map(int, grid.shape)
    row_range = range(1 if require_margin else 0, gh - h - (1 if require_margin else 0) + 1)
    col_range = range(1 if require_margin else 0, gw - w - (1 if require_margin else 0) + 1)
    candidates = [(r0, c0) for r0 in row_range for c0 in col_range]
    random.shuffle(candidates)
    for r0, c0 in candidates:
        r1 = r0 + h - 1
        c1 = c0 + w - 1
        pad_r0 = max(0, r0 - 1)
        pad_c0 = max(0, c0 - 1)
        pad_r1 = min(gh - 1, r1 + 1)
        pad_c1 = min(gw - 1, c1 + 1)
        if np.any(reserved[pad_r0 : pad_r1 + 1, pad_c0 : pad_c1 + 1]):
            continue
        if require_margin:
            if r0 == 0 or c0 == 0 or r1 == gh - 1 or c1 == gw - 1:
                continue
            if np.any(reserved[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2]):
                continue
        cells = np.argwhere(shape)
        if cells.size == 0:
            continue
        rr = cells[:, 0] + r0
        cc = cells[:, 1] + c0
        if np.any(grid[rr, cc] != 0):
            continue
        grid[rr, cc] = int(color)
        reserved[pad_r0 : pad_r1 + 1, pad_c0 : pad_c1 + 1] = True
        return int(r0), int(c0)
    return None


def _distractor_kind(comp: dict, grid: np.ndarray) -> str:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h == w and h >= 2:
        return "broken_square_like"
    if h != w:
        return "non_square"
    return "irregular"


def _validate_semantics(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False
    info = analyze_grid(g)
    bg = int(info["bg"])
    if bg != 0:
        return False
    squares = info["squares"]
    distractors = info["distractors"]
    if len(squares) not in (3, 4) or len(distractors) not in (2, 3):
        return False

    sides = [int(comp["side"]) for comp in squares]
    if len(set(sides)) != len(sides):
        return False
    if not any(int(comp["area"]) < int(comp["box_area"]) for comp in squares):
        return False
    if not any(int(comp["area"]) == int(comp["box_area"]) for comp in squares):
        return False
    if any(int(comp["color"]) in OUTPUT_CYCLE for comp in squares):
        return False
    if sorted(int(comp["color"]) for comp in info["components"]) != sorted(set(int(comp["color"]) for comp in info["components"])):
        return False

    for comp in squares:
        for r, c in corner_marker_coords(comp["bbox"]):
            if not in_bounds(g.shape, r, c) or int(g[r, c]) != 0:
                return False

    kinds = {_distractor_kind(comp, g) for comp in distractors}
    if "broken_square_like" not in kinds:
        return False
    if "non_square" not in kinds and "irregular" not in kinds:
        return False

    area_order = [(int(comp["box_area"]), int(comp["bbox"][0]), int(comp["bbox"][1])) for comp in _sorted_squares(squares, mode="area")]
    scan_order = [(int(comp["box_area"]), int(comp["bbox"][0]), int(comp["bbox"][1])) for comp in _sorted_squares(squares, mode="scan")]
    if area_order == scan_order:
        return False

    out = main(g)
    if np.array_equal(out, g):
        return False
    if np.array_equal(out, _render_output(g, squares, order_mode="scan", fill_bbox=True, marker_use_new_color=True)):
        return False
    if np.array_equal(out, _render_output(g, squares, order_mode="area", fill_bbox=True, marker_use_new_color=False)):
        return False
    if np.array_equal(out, _render_output(g, squares, order_mode="area", fill_bbox=False, marker_use_new_color=True)):
        return False

    square_like = []
    for comp in info["components"]:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if int(r1 - r0) != int(c1 - c0) or int(r1 - r0 + 1) < 2:
            continue
        comp2 = dict(comp)
        comp2["side"] = int(r1 - r0 + 1)
        comp2["box_area"] = int(comp2["side"] ** 2)
        square_like.append(comp2)
    if len(square_like) == len(squares):
        return False
    if np.array_equal(out, _render_output(g, square_like, order_mode="area", fill_bbox=True, marker_use_new_color=True)):
        return False

    sig = hashlib.md5(out.tobytes() + str(out.shape).encode()).hexdigest()
    if sig in _SEEN_OUTPUT_SIGNATURES:
        return False
    return True


def _choose_square_shapes(sizes: list[int]) -> list[np.ndarray]:
    large_indices = [i for i, side in enumerate(sizes) if side >= 3]
    if not large_indices:
        return [_shape_square(side, hollow=False) for side in sizes]
    hollow_idx = random.choice(large_indices)
    filled_idx = random.randrange(len(sizes))
    while filled_idx == hollow_idx and len(sizes) > 1:
        filled_idx = random.randrange(len(sizes))
    shapes: list[np.ndarray] = []
    for idx, side in enumerate(sizes):
        if idx == hollow_idx:
            shapes.append(_shape_square(side, hollow=True))
        elif idx == filled_idx:
            shapes.append(_shape_square(side, hollow=False))
        else:
            if side >= 3 and random.random() < 0.45:
                shapes.append(_shape_square(side, hollow=True))
            else:
                shapes.append(_shape_square(side, hollow=False))
    return shapes


def _build_candidate() -> np.ndarray | None:
    h = random.randint(15, 19)
    w = random.randint(15, 19)
    grid = np.zeros((h, w), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    square_count = random.choice((3, 4))
    distractor_count = random.choice((2, 3))
    sizes = random.sample([2, 3, 4, 5, 6], square_count)
    sizes.sort()
    square_shapes = _choose_square_shapes(sizes)

    square_colors = random.sample(SQUARE_COLOR_POOL, square_count)
    distractor_color_pool = [c for c in ALL_OBJECT_COLORS if c not in square_colors]
    distractor_colors = random.sample(distractor_color_pool, distractor_count)

    square_records = list(zip(sizes, square_shapes, square_colors))
    random.shuffle(square_records)
    for _side, shape, color in square_records:
        if _place_shape(grid, reserved, shape, color, require_margin=True) is None:
            return None

    distractor_shapes: list[np.ndarray] = []
    broken_side_choices = [side for side in (3, 4, 5) if side <= min(h, w) - 2]
    if not broken_side_choices:
        return None
    distractor_shapes.append(_shape_broken_square(random.choice(broken_side_choices)))

    if random.random() < 0.5:
        dh = 1 if random.random() < 0.5 else random.randint(2, 4)
        dw = random.randint(3, 6) if dh == 1 else 1
        if dh == 1 and dw == 1:
            dw = 3
        distractor_shapes.append(_shape_rect(dh, dw))
    else:
        distractor_shapes.append(_shape_l(random.randint(2, 4), random.randint(2, 4)))

    while len(distractor_shapes) < distractor_count:
        if random.random() < 0.5:
            dh = random.randint(1, 3)
            dw = random.randint(2, 5)
            if dh == dw:
                dw += 1
            distractor_shapes.append(_shape_rect(dh, dw))
        else:
            distractor_shapes.append(_shape_l(random.randint(2, 4), random.randint(2, 4)))

    distractor_records = list(zip(distractor_shapes, distractor_colors))
    random.shuffle(distractor_records)
    for shape, color in distractor_records:
        if _place_shape(grid, reserved, shape, color, require_margin=False) is None:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_candidate()
        if grid is None:
            continue
        if not _validate_semantics(grid):
            continue
        out = main(grid)
        sig = hashlib.md5(out.tobytes() + str(out.shape).encode()).hexdigest()
        _SEEN_OUTPUT_SIGNATURES.add(sig)
        return grid
    raise RuntimeError("failed to generate a valid puzzle-7 input")
