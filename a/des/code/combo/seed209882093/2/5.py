import random
from collections import defaultdict

import numpy as np


BG = 0
COLORS = list(range(1, 10))

GRID_H_RANGE = (12, 22)
GRID_W_RANGE = (12, 22)
MAX_GENERATION_ATTEMPTS = 400
PAIR_PLACEMENT_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise AssertionError("segment must be axis-aligned")


def _path_cells(
    start: tuple[int, int],
    end: tuple[int, int],
    mode: str,
) -> list[tuple[int, int]]:
    if mode == "hv":
        corner = (start[0], end[1])
    elif mode == "vh":
        corner = (end[0], start[1])
    else:
        raise AssertionError(f"unknown mode {mode}")
    first = _segment_cells(start, corner)
    second = _segment_cells(corner, end)
    return first + second[1:]


def _box_footprint(center: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    return {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _expanded_box_footprint(
    center: tuple[int, int],
    height: int,
    width: int,
) -> set[tuple[int, int]]:
    r, c = center
    cells = set()
    for rr in range(r - 2, r + 3):
        for cc in range(c - 2, c + 3):
            if 0 <= rr < height and 0 <= cc < width:
                cells.add((rr, cc))
    return cells


def _box_patch(color: int, hole: bool) -> np.ndarray:
    patch = np.full((3, 3), color, dtype=int)
    if hole:
        patch[1, 1] = BG
    return patch


def _find_components(g: np.ndarray) -> list[dict]:
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(g[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            r1 = int(arr[:, 0].max())
            c0 = int(arr[:, 1].min())
            c1 = int(arr[:, 1].max())
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(arr.shape[0]),
                }
            )
    return components


def _detect_boxes(g: np.ndarray) -> dict[int, list[dict]]:
    boxes_by_color: dict[int, list[dict]] = defaultdict(list)
    for comp in _find_components(g):
        r0, c0, r1, c1 = comp["bbox"]
        assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3), "every component must be a 3x3 box"
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        color = int(comp["color"])
        ring_mask = np.ones((3, 3), dtype=bool)
        ring_mask[1, 1] = False
        assert np.all(patch[ring_mask] == color), "box border must be monochromatic"
        center = (r0 + 1, c0 + 1)
        center_value = int(patch[1, 1])
        hole = center_value == BG
        if hole:
            assert int(comp["area"]) == 8, "hollow box must have 8 colored pixels"
        else:
            assert center_value == color and int(comp["area"]) == 9, "solid box must be fully filled"
        boxes_by_color[color].append({"center": center, "hole": hole})

    assert boxes_by_color, "expected at least one colored pair"
    for color, boxes in boxes_by_color.items():
        assert len(boxes) == 2, f"color {color} must appear exactly twice"
        centers = [box["center"] for box in boxes]
        assert centers[0][0] != centers[1][0] and centers[0][1] != centers[1][1], "paired boxes must differ in row and column"
    return boxes_by_color


def _route_for_pair(boxes: list[dict]) -> tuple[tuple[int, int], tuple[int, int], str]:
    a, b = boxes
    if a["hole"] and b["hole"]:
        start, end = sorted((a["center"], b["center"]))
        return start, end, "hv"
    if (not a["hole"]) and (not b["hole"]):
        start, end = sorted((a["center"], b["center"]))
        return start, end, "vh"
    hole_box = a if a["hole"] else b
    solid_box = b if a["hole"] else a
    return hole_box["center"], solid_box["center"], "hv"


def _draw_path(out: np.ndarray, color: int, cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        if int(out[r, c]) == BG:
            out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect equal-colored 3x3 boxes with L-shaped paths whose turn order is encoded by holes.

    - Hollow+hollow: horizontal then vertical.
    - Solid+solid: vertical then horizontal.
    - Hollow+solid: horizontal then vertical, starting from the hollow box.

    When both boxes have the same center status, the lexicographically earlier center is
    used as the start to keep the route deterministic.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    boxes_by_color = _detect_boxes(g)
    for color in sorted(boxes_by_color):
        start, end, mode = _route_for_pair(boxes_by_color[color])
        _draw_path(out, color, _path_cells(start, end, mode))

    return out


def _type_sequence(n_pairs: int) -> list[str]:
    base = ["hh", "ss", "hs"]
    if n_pairs >= 3 and random.random() < 0.7:
        extra = [random.choice(base) for _ in range(n_pairs - 3)]
        out = base + extra
        random.shuffle(out)
        return out
    if n_pairs == 2 and random.random() < 0.6:
        return random.sample(base, 2)
    return [random.choice(base) for _ in range(n_pairs)]


def _pair_count(height: int, width: int) -> int:
    area = height * width
    if area < 220:
        return random.randint(2, 3)
    return random.randint(2, 4)


def _pair_boxes(
    color: int,
    kind: str,
    center_a: tuple[int, int],
    center_b: tuple[int, int],
) -> list[dict]:
    if kind == "hh":
        return [
            {"color": color, "center": center_a, "hole": True},
            {"color": color, "center": center_b, "hole": True},
        ]
    if kind == "ss":
        return [
            {"color": color, "center": center_a, "hole": False},
            {"color": color, "center": center_b, "hole": False},
        ]
    if random.random() < 0.5:
        return [
            {"color": color, "center": center_a, "hole": True},
            {"color": color, "center": center_b, "hole": False},
        ]
    return [
        {"color": color, "center": center_a, "hole": False},
        {"color": color, "center": center_b, "hole": True},
    ]


def _box_fits(
    center: tuple[int, int],
    height: int,
    width: int,
    blocked: set[tuple[int, int]],
    box_guard: set[tuple[int, int]],
) -> bool:
    cells = _box_footprint(center)
    return not (cells & blocked) and not (cells & box_guard)


def _path_is_clear(
    path: list[tuple[int, int]],
    own_guard: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    box_guard: set[tuple[int, int]],
) -> bool:
    for cell in path:
        if cell in own_guard:
            continue
        if cell in blocked or cell in box_guard:
            return False
    return True


def _render_input(height: int, width: int, pairs: list[list[dict]]) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for boxes in pairs:
        for box in boxes:
            r, c = box["center"]
            grid[r - 1 : r + 2, c - 1 : c + 2] = _box_patch(int(box["color"]), bool(box["hole"]))
    return grid


def _expected_output(inp: np.ndarray, pairs: list[list[dict]]) -> np.ndarray:
    out = inp.copy()
    for boxes in pairs:
        color = int(boxes[0]["color"])
        start, end, mode = _route_for_pair(boxes)
        _draw_path(out, color, _path_cells(start, end, mode))
    return out


def _sample_pairs(height: int, width: int) -> list[list[dict]] | None:
    n_pairs = _pair_count(height, width)
    colors = random.sample(COLORS, n_pairs)
    kinds = _type_sequence(n_pairs)

    blocked: set[tuple[int, int]] = set()
    box_guard: set[tuple[int, int]] = set()
    pairs: list[list[dict]] = []

    row_choices = list(range(2, height - 2))
    col_choices = list(range(2, width - 2))
    if len(row_choices) < 2 or len(col_choices) < 2:
        return None

    for color, kind in zip(colors, kinds):
        placed = None
        for _ in range(PAIR_PLACEMENT_ATTEMPTS):
            center_a = (random.choice(row_choices), random.choice(col_choices))
            center_b = (random.choice(row_choices), random.choice(col_choices))
            if center_a == center_b:
                continue
            if center_a[0] == center_b[0] or center_a[1] == center_b[1]:
                continue
            if not _box_fits(center_a, height, width, blocked, box_guard):
                continue
            if not _box_fits(center_b, height, width, blocked, box_guard):
                continue

            footprint_a = _box_footprint(center_a)
            footprint_b = _box_footprint(center_b)
            guard_a = _expanded_box_footprint(center_a, height, width)
            guard_b = _expanded_box_footprint(center_b, height, width)
            if footprint_a & guard_b or footprint_b & guard_a:
                continue

            boxes = _pair_boxes(color, kind, center_a, center_b)
            start, end, mode = _route_for_pair(boxes)
            path = _path_cells(start, end, mode)
            own_guard = guard_a | guard_b
            if not _path_is_clear(path, own_guard, blocked, box_guard):
                continue

            placed = boxes
            blocked |= footprint_a | footprint_b | set(path)
            box_guard |= guard_a | guard_b
            break
        if placed is None:
            return None
        pairs.append(placed)

    return pairs


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    pairs = _sample_pairs(height, width)
    if pairs is None:
        return None

    inp = _render_input(height, width, pairs)
    expected = _expected_output(inp, pairs)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
