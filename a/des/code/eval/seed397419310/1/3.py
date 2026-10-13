from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_H_RANGE = (14, 25)
GRID_W_RANGE = (14, 25)
FRAME_SIDE_OPTIONS = (5, 6)
OBJECT_COUNT_MIN = 2
OBJECT_COUNT_MAX = 5
PLACEMENT_GAP = 1
GENERATE_RETRIES = 400


ARROW_SHAPES: dict[str, frozenset[tuple[int, int]]] = {
    "right": frozenset({(1, 0), (1, 1), (1, 2), (0, 2), (2, 2)}),
    "left": frozenset({(1, 0), (1, 1), (1, 2), (0, 0), (2, 0)}),
    "up": frozenset({(0, 1), (1, 1), (2, 1), (0, 0), (0, 2)}),
    "down": frozenset({(0, 1), (1, 1), (2, 1), (2, 0), (2, 2)}),
}

STEP_BY_DIRECTION: dict[str, tuple[int, int]] = {
    "right": (0, 1),
    "left": (0, -1),
    "up": (-1, 0),
    "down": (1, 0),
}


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []

    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((int(r), int(c)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))

    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return (
        int(rows.min()),
        int(cols.min()),
        int(rows.max()),
        int(cols.max()),
    )


def _rect_border_set(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    border: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        border.add((int(r0), int(c)))
        border.add((int(r1), int(c)))
    for r in range(r0 + 1, r1):
        border.add((int(r), int(c0)))
        border.add((int(r), int(c1)))
    return border


def _normalized(coords: np.ndarray) -> frozenset[tuple[int, int]]:
    r0, c0, _r1, _c1 = _bbox(coords)
    return frozenset((int(r - r0), int(c - c0)) for r, c in coords)


def _distance_to_edge(r: int, c: int, direction: str, shape: tuple[int, int]) -> int:
    h, w = shape
    if direction == "right":
        return int(w - 1 - c)
    if direction == "left":
        return int(c)
    if direction == "up":
        return int(r)
    assert direction == "down"
    return int(h - 1 - r)


def _ray_cells(
    start: tuple[int, int], direction: str, length: int, shape: tuple[int, int]
) -> list[tuple[int, int]]:
    r0, c0 = start
    dr, dc = STEP_BY_DIRECTION[direction]
    h, w = shape
    out: list[tuple[int, int]] = []
    for step in range(1, int(length) + 1):
        r = int(r0 + dr * step)
        c = int(c0 + dc * step)
        if not (0 <= r < h and 0 <= c < w):
            break
        out.append((r, c))
    return out


def _extract_instructions(
    grid: np.ndarray,
) -> tuple[dict[int, tuple[str, int]], list[tuple[int, int, int, int]], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    frames: list[tuple[int, int, int, int]] = []
    frame_area = np.zeros((h, w), dtype=bool)
    instructions: dict[int, tuple[str, int]] = {}

    for comp in _components(g == FRAME):
        r0, c0, r1, c1 = _bbox(comp)
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        assert height >= 5 and width >= 5, "each frame must fit a 3x3 arrow"

        expected_border = _rect_border_set(r0, c0, r1, c1)
        actual_border = {tuple(map(int, rc)) for rc in comp}
        assert actual_border == expected_border, "gray component must be a rectangular border"

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        colors = set(map(int, np.unique(inner))) - {BG}
        assert FRAME not in colors, "frame color cannot appear inside the frame"
        assert len(colors) == 1, "each frame must contain exactly one arrow color"
        color = next(iter(colors))
        assert color not in instructions, "frame colors must be unique"

        arrow_coords = np.argwhere(inner == color)
        assert arrow_coords.shape[0] == 5, "arrow must contain exactly five pixels"
        ar0, ac0, ar1, ac1 = _bbox(arrow_coords)
        assert int(ar1 - ar0 + 1) == 3 and int(ac1 - ac0 + 1) == 3, "arrow must fit a 3x3 box"
        norm = _normalized(arrow_coords)

        direction = None
        for name, shape in ARROW_SHAPES.items():
            if norm == shape:
                direction = name
                break
        assert direction is not None, "unrecognized arrow shape"

        length = inner.shape[1] if direction in ("left", "right") else inner.shape[0]
        instructions[color] = (direction, int(length))
        frames.append((r0, c0, r1, c1))
        frame_area[r0 : r1 + 1, c0 : c1 + 1] = True

    assert frames, "expected at least one frame"
    return instructions, frames, frame_area


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the arrow inside each gray frame as an instruction for the same-colored marker outside the frames.

    Rule:
    - Arrow direction gives the ray direction.
    - Horizontal arrows use frame interior width as length; vertical arrows use frame interior height.
    - Match each instruction to the unique same-colored outside marker.
    - Remove all frames and arrows, then extend each marker by the instructed ray length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    instructions, _frames, frame_area = _extract_instructions(g)

    outside_foreground = np.argwhere((g != BG) & ~frame_area)
    assert outside_foreground.shape[0] == len(instructions), "outside foreground must be exactly the markers"

    markers: dict[int, tuple[int, int]] = {}
    for color in instructions:
        coords = np.argwhere((g == color) & ~frame_area)
        assert coords.shape[0] == 1, "each arrow color must have exactly one outside marker"
        markers[color] = (int(coords[0, 0]), int(coords[0, 1]))

    out = g.copy()
    out[frame_area] = BG

    claimed: dict[tuple[int, int], int] = {}
    for color, (direction, length) in instructions.items():
        marker = markers[color]
        claimed[marker] = color
        for cell in _ray_cells(marker, direction, length, (h, w)):
            prev = claimed.get(cell)
            assert prev in (None, color), "rays must not overlap across colors"
            claimed[cell] = color

    for (r, c), color in claimed.items():
        out[r, c] = int(color)

    return out


def _direction_sample(count: int) -> list[str]:
    base = list(ARROW_SHAPES)
    random.shuffle(base)
    out = base[: min(count, len(base))]
    while len(out) < count:
        out.append(random.choice(base))
    random.shuffle(out)
    return out


def _truncation_sample(count: int) -> list[bool]:
    out = [True, False]
    while len(out) < count:
        out.append(bool(random.getrandbits(1)))
    random.shuffle(out)
    return out[:count]


def _max_objects_for_shape(h: int, w: int) -> int:
    return min(OBJECT_COUNT_MAX, max(OBJECT_COUNT_MIN, (h * w) // 95 + 1))


def _frame_candidates(
    occupied: np.ndarray, frame_h: int, frame_w: int
) -> list[tuple[int, int]]:
    h, w = occupied.shape
    out: list[tuple[int, int]] = []
    for r0 in range(h - frame_h + 1):
        for c0 in range(w - frame_w + 1):
            rr0 = max(0, r0 - PLACEMENT_GAP)
            cc0 = max(0, c0 - PLACEMENT_GAP)
            rr1 = min(h, r0 + frame_h + PLACEMENT_GAP)
            cc1 = min(w, c0 + frame_w + PLACEMENT_GAP)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            out.append((r0, c0))
    return out


def _build_marker_forbidden(
    shape: tuple[int, int], frames: list[dict[str, object]]
) -> np.ndarray:
    h, w = shape
    forbidden = np.zeros((h, w), dtype=bool)
    for item in frames:
        r0, c0, r1, c1 = item["frame"]
        rr0 = max(0, int(r0) - 1)
        cc0 = max(0, int(c0) - 1)
        rr1 = min(h, int(r1) + 2)
        cc1 = min(w, int(c1) + 2)
        forbidden[rr0:rr1, cc0:cc1] = True
    return forbidden


def _expected_output(shape: tuple[int, int], specs: list[dict[str, object]]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    claimed: dict[tuple[int, int], int] = {}
    for spec in specs:
        color = int(spec["color"])
        marker = spec["marker"]
        claimed[marker] = color
        for cell in _ray_cells(marker, spec["direction"], int(spec["length"]), shape):
            prev = claimed.get(cell)
            assert prev in (None, color)
            claimed[cell] = color
    for (r, c), color in claimed.items():
        out[r, c] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    count = random.randint(OBJECT_COUNT_MIN, _max_objects_for_shape(h, w))
    directions = _direction_sample(count)
    truncations = _truncation_sample(count)
    colors = random.sample(PALETTE, count)

    occupied = np.zeros((h, w), dtype=bool)
    specs: list[dict[str, object]] = []

    for idx in range(count):
        frame_h = random.choice(FRAME_SIDE_OPTIONS)
        frame_w = random.choice(FRAME_SIDE_OPTIONS)
        candidates = _frame_candidates(occupied, frame_h, frame_w)
        if not candidates:
            return None

        r0, c0 = random.choice(candidates)
        r1 = r0 + frame_h - 1
        c1 = c0 + frame_w - 1

        rr0 = max(0, r0 - PLACEMENT_GAP)
        cc0 = max(0, c0 - PLACEMENT_GAP)
        rr1 = min(h, r1 + PLACEMENT_GAP + 1)
        cc1 = min(w, c1 + PLACEMENT_GAP + 1)
        occupied[rr0:rr1, cc0:cc1] = True

        inner_h = frame_h - 2
        inner_w = frame_w - 2
        arrow_top = random.randint(0, inner_h - 3)
        arrow_left = random.randint(0, inner_w - 3)
        direction = directions[idx]
        length = inner_w if direction in ("left", "right") else inner_h

        specs.append(
            {
                "color": int(colors[idx]),
                "direction": direction,
                "length": int(length),
                "frame": (int(r0), int(c0), int(r1), int(c1)),
                "arrow_offset": (int(arrow_top), int(arrow_left)),
                "truncated": bool(truncations[idx]),
            }
        )

    marker_forbidden = _build_marker_forbidden((h, w), specs)
    reserved = np.zeros((h, w), dtype=bool)
    order = list(range(count))
    random.shuffle(order)

    for idx in order:
        spec = specs[idx]
        direction = str(spec["direction"])
        length = int(spec["length"])
        want_truncated = bool(spec["truncated"])
        candidates: list[tuple[int, int, list[tuple[int, int]]]] = []

        for r in range(h):
            for c in range(w):
                if marker_forbidden[r, c] or reserved[r, c]:
                    continue
                space = _distance_to_edge(r, c, direction, (h, w))
                if want_truncated:
                    if not (1 <= space < length):
                        continue
                else:
                    if space < length:
                        continue
                ray = _ray_cells((r, c), direction, length, (h, w))
                if want_truncated and len(ray) >= length:
                    continue
                if not want_truncated and len(ray) != length:
                    continue
                if any(reserved[rr, cc] for rr, cc in ray):
                    continue
                candidates.append((r, c, ray))

        if not candidates:
            return None

        r, c, ray = random.choice(candidates)
        spec["marker"] = (int(r), int(c))
        reserved[r, c] = True
        for rr, cc in ray:
            reserved[rr, cc] = True

    inp = np.full((h, w), BG, dtype=int)
    for spec in specs:
        color = int(spec["color"])
        r0, c0, r1, c1 = spec["frame"]
        inp[r0 : r1 + 1, c0] = FRAME
        inp[r0 : r1 + 1, c1] = FRAME
        inp[r0, c0 : c1 + 1] = FRAME
        inp[r1, c0 : c1 + 1] = FRAME

        inner_r0 = r0 + 1 + int(spec["arrow_offset"][0])
        inner_c0 = c0 + 1 + int(spec["arrow_offset"][1])
        for dr, dc in ARROW_SHAPES[str(spec["direction"])]:
            inp[inner_r0 + dr, inner_c0 + dc] = color

        mr, mc = spec["marker"]
        inp[mr, mc] = color

    expected = _expected_output((h, w), specs)

    try:
        actual = main(inp)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
