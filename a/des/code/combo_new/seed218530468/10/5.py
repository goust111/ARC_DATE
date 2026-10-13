from __future__ import annotations

import numpy as np


DIRECTIONS = (
    "up",
    "down",
    "left",
    "right",
    "up_left",
    "up_right",
    "down_left",
    "down_right",
)


FRAME_COUNT_RANGE = (2, 5)
STAR_COUNT_RANGE = (1, 5)
FRAME_SIDE_CHOICES = (5, 7, 9)
GRID_SIDE_RANGE = (18, 31)
FRAME_EDGE_MARGIN = 2
FRAME_SEPARATION = 1
STAR_CLEARANCE = 1
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue

            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))

            rows = [r0 for r0, _ in coords]
            cols = [c0 for _, c0 in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )

    return comps


def _is_hollow_frame(grid: np.ndarray, bg: int, comp: dict) -> bool:
    color = int(comp["color"])
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 5 or w < 5:
        return False
    if int(comp["area"]) != 2 * h + 2 * w - 4:
        return False
    if not np.all(grid[r0 : r1 + 1, c0] == color):
        return False
    if not np.all(grid[r0 : r1 + 1, c1] == color):
        return False
    if not np.all(grid[r0, c0 : c1 + 1] == color):
        return False
    if not np.all(grid[r1, c0 : c1 + 1] == color):
        return False
    if not np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == bg):
        return False
    return True


def _classify_direction(point: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r, c = map(int, point)
    r0, c0, r1, c1 = map(int, bbox)
    vr = 2 * r - (r0 + r1)
    vc = 2 * c - (c0 + c1)

    if abs(vr) > abs(vc):
        return "down" if vr > 0 else "up"
    if abs(vc) > abs(vr):
        return "right" if vc > 0 else "left"
    if vr < 0 and vc < 0:
        return "up_left"
    if vr < 0 and vc > 0:
        return "up_right"
    if vr > 0 and vc < 0:
        return "down_left"
    return "down_right"


def _interior_target(bbox: tuple[int, int, int, int], direction: str) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, bbox)
    ir0, ic0 = int(r0 + 1), int(c0 + 1)
    ir1, ic1 = int(r1 - 1), int(c1 - 1)
    mr = int((ir0 + ir1) // 2)
    mc = int((ic0 + ic1) // 2)

    mapping = {
        "up": (ir0, mc),
        "down": (ir1, mc),
        "left": (mr, ic0),
        "right": (mr, ic1),
        "up_left": (ir0, ic0),
        "up_right": (ir0, ic1),
        "down_left": (ir1, ic0),
        "down_right": (ir1, ic1),
    }
    return mapping[direction]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move same-color exterior stars into their matching hollow frames as directional interior dots.

    Input structure:
    - The background is the most common color.
    - Several unique-colored one-pixel-thick hollow rectangular frames sit on the grid,
      each with a background-filled interior of at least 3x3.
    - Additional foreground cells are isolated one-pixel stars placed strictly outside all
      frames, and each star shares its color with exactly one frame.

    Rule:
    - Match every star to the frame of the same color and erase the exterior star.
    - Determine the star's coarse direction from the frame center using eight sectors:
      up, down, left, right, and the four diagonals.
    - Paint one same-color dot inside the frame at the sector's fixed interior location:
      top-middle, bottom-middle, middle-left, middle-right, or one of the four corners.
    - Keep all frame borders unchanged.

    Key insight:
    - The outer stars encode direction only; the output is obtained by decoding each
      star's relative sector around its same-color frame and drawing the corresponding
      marker inside that frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)

    frames: list[dict] = []
    stars: list[dict] = []
    for comp in _find_components(g, bg):
        if _is_hollow_frame(g, bg, comp):
            frames.append(comp)
        else:
            assert int(comp["area"]) == 1, "expected only frames and single-pixel stars"
            stars.append(comp)

    assert frames, "expected at least one frame"
    frame_by_color: dict[int, dict] = {}
    for frame in frames:
        color = int(frame["color"])
        assert color not in frame_by_color, "frame colors must be unique"
        frame_by_color[color] = frame

    out = g.copy()
    for star in stars:
        sr, sc = star["coords"][0]
        out[int(sr), int(sc)] = bg

    for star in stars:
        color = int(star["color"])
        assert color in frame_by_color, "each star must match a frame color"
        sr, sc = star["coords"][0]
        direction = _classify_direction((int(sr), int(sc)), frame_by_color[color]["bbox"])
        tr, tc = _interior_target(frame_by_color[color]["bbox"], direction)
        out[int(tr), int(tc)] = color

    return out


# --------------- generator (codex) ---------------


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = map(int, a)
    br0, bc0, br1, bc1 = map(int, b)
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _point_in_expanded_box(r: int, c: int, box: tuple[int, int, int, int], pad: int) -> bool:
    r0, c0, r1, c1 = map(int, box)
    return (r0 - pad) <= int(r) <= (r1 + pad) and (c0 - pad) <= int(c) <= (c1 + pad)


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)


def _choose_grid_shape(rng: np.random.Generator, n_frames: int) -> tuple[int, int]:
    min_side = 22 if n_frames >= 4 else 20 if n_frames == 3 else 18
    height = int(rng.integers(min_side, GRID_SIDE_RANGE[1]))
    width = int(rng.integers(min_side, GRID_SIDE_RANGE[1]))
    return height, width


def _sample_frame_sizes(rng: np.random.Generator, n_frames: int) -> list[tuple[int, int]]:
    side_choices = FRAME_SIDE_CHOICES if n_frames <= 3 else FRAME_SIDE_CHOICES[:2]
    sizes = [(int(rng.choice(side_choices)), int(rng.choice(side_choices))) for _ in range(n_frames)]
    sizes.sort(key=lambda hw: hw[0] * hw[1], reverse=True)
    return sizes


def _place_frames(
    rng: np.random.Generator,
    shape: tuple[int, int],
    sizes: list[tuple[int, int]],
) -> list[tuple[int, int, int, int]] | None:
    height, width = map(int, shape)
    boxes: list[tuple[int, int, int, int]] = []

    for fh, fw in sizes:
        positions: list[tuple[int, int, int, int]] = []
        for r0 in range(FRAME_EDGE_MARGIN, height - fh - FRAME_EDGE_MARGIN + 1):
            for c0 in range(FRAME_EDGE_MARGIN, width - fw - FRAME_EDGE_MARGIN + 1):
                box = (int(r0), int(c0), int(r0 + fh - 1), int(c0 + fw - 1))
                if all(not _boxes_overlap(box, other, pad=FRAME_SEPARATION) for other in boxes):
                    positions.append(box)
        if not positions:
            return None
        boxes.append(positions[int(rng.integers(0, len(positions)))])

    rng.shuffle(boxes)
    return boxes


def _cell_is_clear_for_star(
    r: int,
    c: int,
    frame_boxes: list[tuple[int, int, int, int]],
    stars: list[tuple[int, int]],
) -> bool:
    for box in frame_boxes:
        if _point_in_expanded_box(int(r), int(c), box, STAR_CLEARANCE):
            return False
    for sr, sc in stars:
        if max(abs(int(r) - int(sr)), abs(int(c) - int(sc))) <= STAR_CLEARANCE:
            return False
    return True


def _star_candidates(
    shape: tuple[int, int],
    owner_box: tuple[int, int, int, int],
    direction: str,
    frame_boxes: list[tuple[int, int, int, int]],
    stars: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    height, width = map(int, shape)
    candidates: list[tuple[int, int]] = []
    for r in range(height):
        for c in range(width):
            if not _cell_is_clear_for_star(int(r), int(c), frame_boxes, stars):
                continue
            if _classify_direction((int(r), int(c)), owner_box) == direction:
                candidates.append((int(r), int(c)))
    return candidates


def _expected_output(
    inp: np.ndarray,
    bg: int,
    frames: list[dict],
    stars: list[tuple[int, int, int, str]],
) -> np.ndarray:
    out = inp.copy()
    frame_by_color = {int(frame["color"]): frame for frame in frames}
    for r, c, color, direction in stars:
        out[int(r), int(c)] = bg
        tr, tc = _interior_target(frame_by_color[int(color)]["bbox"], direction)
        out[int(tr), int(tc)] = int(color)
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_frames = int(rng.integers(*FRAME_COUNT_RANGE))
    height, width = _choose_grid_shape(rng, n_frames)
    frame_sizes = _sample_frame_sizes(rng, n_frames)
    frame_boxes = _place_frames(rng, (height, width), frame_sizes)
    if frame_boxes is None:
        return None

    palette = rng.choice(np.arange(10), size=n_frames + 1, replace=False)
    bg = int(palette[0])
    frame_colors = [int(color) for color in palette[1:]]

    inp = np.full((height, width), bg, dtype=int)
    frames: list[dict] = []
    for box, color in zip(frame_boxes, frame_colors):
        _draw_frame(inp, box, color)
        frames.append({"bbox": box, "color": int(color)})

    # Ensure every frame has all eight directional sectors available before
    # choosing a random subset of 1-4 stars.
    for frame in frames:
        available = [
            direction
            for direction in DIRECTIONS
            if _star_candidates(inp.shape, frame["bbox"], direction, frame_boxes, [])
        ]
        if len(available) != len(DIRECTIONS):
            return None

    star_specs: list[tuple[int, int, int, str]] = []
    occupied_stars: list[tuple[int, int]] = []
    frame_order = list(range(n_frames))
    rng.shuffle(frame_order)

    for frame_idx in frame_order:
        frame = frames[frame_idx]
        count = int(rng.integers(*STAR_COUNT_RANGE))
        directions = list(rng.permutation(np.array(DIRECTIONS, dtype=object))[:count])
        for direction in directions:
            candidates = _star_candidates(inp.shape, frame["bbox"], str(direction), frame_boxes, occupied_stars)
            if not candidates:
                return None
            star_r, star_c = candidates[int(rng.integers(0, len(candidates)))]
            occupied_stars.append((int(star_r), int(star_c)))
            inp[int(star_r), int(star_c)] = int(frame["color"])
            star_specs.append((int(star_r), int(star_c), int(frame["color"]), str(direction)))

    if _get_bg(inp) != bg:
        return None

    out = _expected_output(inp, bg, frames, star_specs)
    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
