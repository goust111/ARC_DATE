from __future__ import annotations

import numpy as np


DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

HOLE_SHAPES = [
    (1, 2),
    (2, 1),
    (1, 3),
    (3, 1),
    (2, 2),
    (1, 4),
    (4, 1),
    (1, 5),
    (5, 1),
    (2, 3),
    (3, 2),
]

GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 250
GRID_MIN = 16
GRID_MAX = 30
OBJECT_COUNT_MIN = 2
OBJECT_COUNT_MAX = 4


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS.values():
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _checkerboard(shape: tuple[int, int], color_a: int, color_b: int, offset: int) -> np.ndarray:
    rr, cc = np.indices(shape)
    parity = (rr + cc + int(offset)) % 2
    return np.where(parity == 0, int(color_a), int(color_b)).astype(int)


def _infer_background(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    rr, cc = np.indices((h, w))
    even = ((rr + cc) % 2) == 0
    colors = [int(v) for v in np.unique(grid)]

    best_score = -1
    best_pair = None
    for color_even in colors:
        for color_odd in colors:
            if color_even == color_odd:
                continue
            score = int(np.count_nonzero(grid[even] == color_even))
            score += int(np.count_nonzero(grid[~even] == color_odd))
            if score > best_score:
                best_score = score
                best_pair = (int(color_even), int(color_odd))

    assert best_pair is not None
    return np.where(even, best_pair[0], best_pair[1]).astype(int)


def _enclosed_hole(local_fg: np.ndarray) -> np.ndarray:
    h, w = local_fg.shape
    holes: list[np.ndarray] = []
    for comp in _components(~local_fg):
        touches_edge = bool(
            np.any(comp[:, 0] == 0)
            or np.any(comp[:, 0] == h - 1)
            or np.any(comp[:, 1] == 0)
            or np.any(comp[:, 1] == w - 1)
        )
        if not touches_edge:
            holes.append(comp)
    assert len(holes) == 1
    return holes[0]


def _direction_from_hole(local_shape: tuple[int, int], hole: np.ndarray) -> tuple[int, int]:
    h, w = map(int, local_shape)
    hr0, hc0, hr1, hc1 = _bbox(hole)
    margins = {
        "up": hr0,
        "down": h - 1 - hr1,
        "left": hc0,
        "right": w - 1 - hc1,
    }
    closest = min(margins.values())
    candidates = [name for name, value in margins.items() if value == closest]
    if len(candidates) == 1:
        return DIRS[candidates[0]]

    hole_center = hole.mean(axis=0)
    delta_r = float(hole_center[0]) - (h - 1) / 2.0
    delta_c = float(hole_center[1]) - (w - 1) / 2.0
    if abs(delta_r) > abs(delta_c):
        return DIRS["up" if delta_r < 0 else "down"]
    return DIRS["left" if delta_c < 0 else "right"]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each hollow foreground object toward its internal hole by a distance equal to the hole's area.

    Input structure:
    - The background is a two-color checkerboard that covers all cells not occupied by foreground objects.
    - Each foreground object is a single-color 4-connected block with one enclosed rectangular hole where the checkerboard remains visible.
    - The hole is offset toward exactly one side of the object's bounding box: up, down, left, or right.

    Rule:
    - Recover the checkerboard background pattern and identify each foreground object as the cells that break that pattern.
    - For each object, find its unique enclosed background hole, use the hole's offset side as the movement direction, and count the hole cells to get the movement distance.
    - Move the object's foreground pixels by that many cells in the hole's direction.
    - Restore the object's old cells to the checkerboard, and leave the moved hole showing the checkerboard in its new position.

    Key insight:
    The hole encodes both parts of the motion: which side it sits closer to gives the direction, and how many checkerboard cells it contains gives the translation length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    background = _infer_background(g)
    fg = g != background
    out = background.copy()

    for comp in _components(fg):
        r0, c0, r1, c1 = _bbox(comp)
        local_fg = fg[r0 : r1 + 1, c0 : c1 + 1]
        hole = _enclosed_hole(local_fg)
        dr, dc = _direction_from_hole(local_fg.shape, hole)
        distance = int(hole.shape[0])
        color_values = g[comp[:, 0], comp[:, 1]]
        color = int(color_values[0])
        assert bool(np.all(color_values == color))

        moved = comp + np.array([dr * distance, dc * distance], dtype=int)
        assert int(moved[:, 0].min()) >= 0 and int(moved[:, 0].max()) < g.shape[0]
        assert int(moved[:, 1].min()) >= 0 and int(moved[:, 1].max()) < g.shape[1]
        out[moved[:, 0], moved[:, 1]] = color

    return out


def _rect_clear(rect: tuple[int, int, int, int], others: list[tuple[int, int, int, int]], margin: int) -> bool:
    r0, c0, r1, c1 = rect
    for or0, oc0, or1, oc1 in others:
        if not (r1 + margin < or0 or or1 + margin < r0 or c1 + margin < oc0 or oc1 + margin < c0):
            return False
    return True


def _centered_positions(span: int, size: int) -> list[int]:
    starts = list(range(1, span - size))
    target = (span - 1) / 2.0
    best = min(abs((start + (size - 1) / 2.0) - target) for start in starts)
    return [start for start in starts if abs((start + (size - 1) / 2.0) - target) == best]


def _make_object_mask(
    outer_h: int,
    outer_w: int,
    hole_h: int,
    hole_w: int,
    direction: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, tuple[int, int], int]:
    mask = np.ones((outer_h, outer_w), dtype=bool)
    if direction == "up":
        hole_r = 1
        hole_c = int(rng.choice(_centered_positions(outer_w, hole_w)))
    elif direction == "down":
        hole_r = outer_h - hole_h - 1
        hole_c = int(rng.choice(_centered_positions(outer_w, hole_w)))
    elif direction == "left":
        hole_r = int(rng.choice(_centered_positions(outer_h, hole_h)))
        hole_c = 1
    else:
        hole_r = int(rng.choice(_centered_positions(outer_h, hole_h)))
        hole_c = outer_w - hole_w - 1

    mask[hole_r : hole_r + hole_h, hole_c : hole_c + hole_w] = False
    return mask, (hole_r, hole_c), hole_h * hole_w


def _sample_outer_dims(
    hole_h: int,
    hole_w: int,
    direction: str,
    rng: np.random.Generator,
) -> tuple[int, int]:
    if direction in ("up", "down"):
        h_low, h_high = hole_h + 3, min(10, hole_h + 6)
        w_candidates = [
            width
            for width in range(hole_w + 2, min(10, hole_w + 5) + 1)
            if (width - hole_w) % 2 == 0
        ]
        assert w_candidates
    else:
        h_candidates = [
            height
            for height in range(hole_h + 2, min(10, hole_h + 5) + 1)
            if (height - hole_h) % 2 == 0
        ]
        assert h_candidates
        w_low, w_high = hole_w + 3, min(10, hole_w + 6)
    if direction in ("up", "down"):
        outer_h = int(rng.integers(h_low, h_high + 1))
        outer_w = int(rng.choice(w_candidates))
    else:
        outer_h = int(rng.choice(h_candidates))
        outer_w = int(rng.integers(w_low, w_high + 1))
    return outer_h, outer_w


def _place_objects(
    h: int,
    w: int,
    specs: list[dict],
    rng: np.random.Generator,
) -> bool:
    input_rects: list[tuple[int, int, int, int]] = []
    output_rects: list[tuple[int, int, int, int]] = []

    specs.sort(key=lambda spec: -(max(spec["h"], spec["w"]) + spec["distance"]))
    for spec in specs:
        dr, dc = DIRS[spec["direction"]]
        if dr < 0:
            top_min, top_max = spec["distance"], h - spec["h"]
            left_min, left_max = 0, w - spec["w"]
        elif dr > 0:
            top_min, top_max = 0, h - spec["h"] - spec["distance"]
            left_min, left_max = 0, w - spec["w"]
        elif dc < 0:
            top_min, top_max = 0, h - spec["h"]
            left_min, left_max = spec["distance"], w - spec["w"]
        else:
            top_min, top_max = 0, h - spec["h"]
            left_min, left_max = 0, w - spec["w"] - spec["distance"]

        if top_min > top_max or left_min > left_max:
            return False

        placed = False
        for _ in range(PLACE_ATTEMPTS):
            top = int(rng.integers(top_min, top_max + 1))
            left = int(rng.integers(left_min, left_max + 1))
            out_top = top + dr * spec["distance"]
            out_left = left + dc * spec["distance"]

            input_rect = (top, left, top + spec["h"] - 1, left + spec["w"] - 1)
            output_rect = (
                out_top,
                out_left,
                out_top + spec["h"] - 1,
                out_left + spec["w"] - 1,
            )
            if not _rect_clear(input_rect, input_rects, margin=1):
                continue
            if not _rect_clear(output_rect, output_rects, margin=1):
                continue

            spec["top"] = top
            spec["left"] = left
            input_rects.append(input_rect)
            output_rects.append(output_rect)
            placed = True
            break

        if not placed:
            return False

    return True


def _paint_object(grid: np.ndarray, spec: dict, moved: bool) -> None:
    top = int(spec["top"])
    left = int(spec["left"])
    if moved:
        dr, dc = DIRS[spec["direction"]]
        top += dr * int(spec["distance"])
        left += dc * int(spec["distance"])
    patch = grid[top : top + spec["h"], left : left + spec["w"]]
    patch[spec["mask"]] = int(spec["color"])


def _build_pair(specs: list[dict], bg_a: int, bg_b: int, offset: int, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    inp = _checkerboard(shape, bg_a, bg_b, offset)
    out = _checkerboard(shape, bg_a, bg_b, offset)
    for spec in specs:
        _paint_object(inp, spec, moved=False)
        _paint_object(out, spec, moved=True)
    return inp, out


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    w = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    object_count = int(rng.integers(OBJECT_COUNT_MIN, OBJECT_COUNT_MAX + 1))

    palette = [int(v) for v in rng.choice(np.arange(10), size=2 + object_count, replace=False)]
    bg_a, bg_b = palette[:2]
    colors = palette[2:]
    offset = int(rng.integers(0, 2))

    direction_order = ["up", "down", "left", "right"]
    rng.shuffle(direction_order)
    directions = direction_order[:object_count]

    specs: list[dict] = []
    for color, direction in zip(colors, directions):
        hole_h, hole_w = HOLE_SHAPES[int(rng.integers(0, len(HOLE_SHAPES)))]
        outer_h, outer_w = _sample_outer_dims(hole_h, hole_w, direction, rng)
        mask, hole_rc, distance = _make_object_mask(outer_h, outer_w, hole_h, hole_w, direction, rng)
        specs.append(
            {
                "color": int(color),
                "direction": direction,
                "mask": mask,
                "h": outer_h,
                "w": outer_w,
                "hole_rc": hole_rc,
                "hole_shape": (hole_h, hole_w),
                "distance": int(distance),
            }
        )

    if not _place_objects(h, w, specs, rng):
        return None

    inp, expected = _build_pair(specs, bg_a, bg_b, offset, (h, w))
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if np.array_equal(actual, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        try:
            grid = _generate_attempt(rng)
        except (AssertionError, ValueError):
            continue
        if grid is not None:
            try:
                out = main(grid)
            except Exception:
                continue
            if out.ndim == 2 and max(out.shape) <= 30:
                return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
