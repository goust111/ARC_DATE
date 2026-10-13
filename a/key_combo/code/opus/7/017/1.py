import random
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if _SOLVER_DIR.is_dir() and str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

try:
    from arc_utils import Color, in_bounds
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        ORANGE = 7
        TEAL = 8
        MAROON = 9

    def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
        return 0 <= r < shape[0] and 0 <= c < shape[1]


FRAME_COLORS = (Color.BLUE, Color.RED, Color.GREEN)
CONTENT_COLORS = (Color.ORANGE, Color.TEAL, Color.MAROON)
CORNER_FROM_NEIGHBORS = {
    frozenset(((1, 0), (0, 1))): "ul",
    frozenset(((1, 0), (0, -1))): "ur",
    frozenset(((-1, 0), (0, 1))): "ll",
    frozenset(((-1, 0), (0, -1))): "lr",
}
DIR_BY_CORNER = {
    "ul": (-1, -1),
    "ur": (-1, 1),
    "ll": (1, -1),
    "lr": (1, 1),
}
GEN_ATTEMPTS = 400
GRID_SIDE_RANGE = (18, 24)
INNER_SIDE_RANGE = (3, 5)
CONTENT_COUNT_RANGE = (3, 5)
EDGE_GAP = 4
QUADRANTS = ("ul", "ur", "ll", "lr")


def _perimeter_coords(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    coords = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
        coords.append((r1, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c0))
        coords.append((r, c1))
    return coords


def _paint_perimeter(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    for r, c in _perimeter_coords(bbox):
        grid[r, c] = int(color)


def _run_to_last_color(grid: np.ndarray, r: int, c: int, dr: int, dc: int, color: int) -> tuple[int, int]:
    rr, cc = r + dr, c + dc
    last_r, last_c = r, c
    while in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == int(color):
        last_r, last_c = rr, cc
        rr += dr
        cc += dc
    return last_r, last_c


def _expected_corner(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> str:
    r0, c0, r1, c1 = bbox
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    grid_r = (shape[0] - 1) / 2.0
    grid_c = (shape[1] - 1) / 2.0
    vertical = "u" if center_r < grid_r else "l"
    horizontal = "l" if center_c < grid_c else "r"
    return vertical + horizontal


def _detect_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    frames = []
    for yr, yc in np.argwhere(g == Color.YELLOW):
        neighbor_info = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = int(yr + dr), int(yc + dc)
            if in_bounds(g.shape, nr, nc) and int(g[nr, nc]) not in (Color.BLACK, Color.YELLOW):
                neighbor_info.append((dr, dc, int(g[nr, nc])))
        assert len(neighbor_info) == 2, "yellow marker must touch exactly two border pixels"
        colors = {col for _, _, col in neighbor_info}
        assert len(colors) == 1, "marker-adjacent border pixels must share one color"
        border_color = colors.pop()
        corner = CORNER_FROM_NEIGHBORS[frozenset((dr, dc) for dr, dc, _ in neighbor_info)]

        if corner == "ul":
            r0, c0 = int(yr), int(yc)
            _, c1 = _run_to_last_color(g, yr, yc, 0, 1, border_color)
            r1, _ = _run_to_last_color(g, yr, yc, 1, 0, border_color)
        elif corner == "ur":
            r0, c1 = int(yr), int(yc)
            _, c0 = _run_to_last_color(g, yr, yc, 0, -1, border_color)
            r1, _ = _run_to_last_color(g, yr, yc, 1, 0, border_color)
        elif corner == "ll":
            r1, c0 = int(yr), int(yc)
            _, c1 = _run_to_last_color(g, yr, yc, 0, 1, border_color)
            r0, _ = _run_to_last_color(g, yr, yc, -1, 0, border_color)
        else:
            r1, c1 = int(yr), int(yc)
            _, c0 = _run_to_last_color(g, yr, yc, 0, -1, border_color)
            r0, _ = _run_to_last_color(g, yr, yc, -1, 0, border_color)

        bbox = (int(r0), int(c0), int(r1), int(c1))
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must enclose a nonempty interior"

        perimeter = _perimeter_coords(bbox)
        yellow_hits = 0
        for r, c in perimeter:
            val = int(g[r, c])
            assert val in (border_color, Color.YELLOW), "frame perimeter must be border color except the marker"
            yellow_hits += val == Color.YELLOW
        assert yellow_hits == 1, "each frame must have exactly one yellow corner marker"

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        content_coords = [(r0 + 1 + r, c0 + 1 + c) for r, c in np.argwhere(inner != Color.BLACK)]
        assert content_coords, "frame interior must contain content"
        content_colors = {int(g[r, c]) for r, c in content_coords}
        assert len(content_colors) == 1, "content inside one frame must be monochrome"
        content_color = content_colors.pop()
        assert content_color not in (border_color, Color.YELLOW), "content color must differ from border and marker"

        frames.append(
            {
                "bbox": bbox,
                "marker": (int(yr), int(yc)),
                "corner": corner,
                "direction": DIR_BY_CORNER[corner],
                "border_color": int(border_color),
                "content_color": int(content_color),
                "content_coords": content_coords,
                "count": len(content_coords),
            }
        )
    frames.sort(key=lambda frame: (frame["bbox"][1], frame["bbox"][0]))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Eject each frame's interior content along the diagonal chosen by its yellow corner and recolor the emptied frame with the content color.

    Input structure:
    - A black background contains 2-3 hollow rectangular frames with 1-pixel borders.
    - Each frame border has one yellow corner marker replacing that corner's original border color.
    - Each frame interior contains a 4-connected monochrome content cluster whose color differs from the frame border.

    Rule:
    - Detect each rectangular frame and read the yellow corner position as one of the four diagonal directions.
    - Count and erase the interior content pixels, then place that many pixels of the content color starting at the marked corner and continuing one step per cell along the chosen diagonal.
    - Recolor the whole frame perimeter to the content color, processing frames from left to right so later diagonal rays overwrite earlier ones where they overlap.

    Key insight:
    - The marked corner simultaneously identifies the owning frame, encodes the diagonal ejection direction, and becomes the first ejected pixel after the frame hands its color identity over to the former content color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    frames = _detect_frames(g)

    for frame in frames:
        for r, c in frame["content_coords"]:
            out[r, c] = Color.BLACK

        dr, dc = frame["direction"]
        mr, mc = frame["marker"]
        for step in range(frame["count"]):
            rr = mr + step * dr
            cc = mc + step * dc
            if not in_bounds(out.shape, rr, cc):
                break
            out[rr, cc] = frame["content_color"]

        _paint_perimeter(out, frame["bbox"], frame["content_color"])

    return out


def _gap_at_least(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _random_connected_shape(
    rng: random.Random, inner_h: int, inner_w: int, count: int
) -> list[tuple[int, int]]:
    seed = (rng.randrange(inner_h), rng.randrange(inner_w))
    shape = {seed}
    while len(shape) < count:
        frontier = []
        for r, c in shape:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < inner_h and 0 <= nc < inner_w and (nr, nc) not in shape:
                    frontier.append((nr, nc))
        shape.add(rng.choice(frontier))
    return sorted(shape)


def _quadrant_ranges(h: int, w: int, outer_h: int, outer_w: int, quadrant: str) -> tuple[range, range]:
    upper_stop = h // 2 - outer_h - 1
    left_stop = w // 2 - outer_w - 1
    lower_start = h // 2 + 1
    right_start = w // 2 + 1
    if quadrant == "ul":
        return range(EDGE_GAP, upper_stop + 1), range(EDGE_GAP, left_stop + 1)
    if quadrant == "ur":
        return range(EDGE_GAP, upper_stop + 1), range(right_start, w - EDGE_GAP - outer_w + 1)
    if quadrant == "ll":
        return range(lower_start, h - EDGE_GAP - outer_h + 1), range(EDGE_GAP, left_stop + 1)
    return range(lower_start, h - EDGE_GAP - outer_h + 1), range(right_start, w - EDGE_GAP - outer_w + 1)


def _frame_from_params(
    rng: random.Random,
    shape: tuple[int, int],
    quadrant: str,
    border_color: int,
    content_color: int,
    existing_frames: list[dict],
    used_ray_cells: set[tuple[int, int]],
) -> dict | None:
    h, w = shape
    inner_h = rng.randint(*INNER_SIDE_RANGE)
    inner_w = rng.randint(*INNER_SIDE_RANGE)
    outer_h, outer_w = inner_h + 2, inner_w + 2
    row_range, col_range = _quadrant_ranges(h, w, outer_h, outer_w, quadrant)
    if not row_range or not col_range:
        return None

    positions = [(r0, c0) for r0 in row_range for c0 in col_range]
    rng.shuffle(positions)
    count_choices = list(range(CONTENT_COUNT_RANGE[0], min(CONTENT_COUNT_RANGE[1], inner_h * inner_w) + 1))
    rng.shuffle(count_choices)

    for r0, c0 in positions:
        bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
        if any(not _gap_at_least(bbox, other["bbox"], EDGE_GAP) for other in existing_frames):
            continue

        if quadrant == "ul":
            marker = (r0, c0)
        elif quadrant == "ur":
            marker = (r0, c0 + outer_w - 1)
        elif quadrant == "ll":
            marker = (r0 + outer_h - 1, c0)
        else:
            marker = (r0 + outer_h - 1, c0 + outer_w - 1)

        direction = DIR_BY_CORNER[quadrant]
        ray_cells = []
        rr, cc = marker
        while in_bounds(shape, rr, cc):
            ray_cells.append((rr, cc))
            rr += direction[0]
            cc += direction[1]

        if any(cell in used_ray_cells for cell in ray_cells):
            continue
        if any(
            other["bbox"][0] <= rr <= other["bbox"][2] and other["bbox"][1] <= cc <= other["bbox"][3]
            for rr, cc in ray_cells[1:]
            for other in existing_frames
        ):
            continue

        for count in count_choices:
            if count > len(ray_cells):
                continue
            content_shape = _random_connected_shape(rng, inner_h, inner_w, count)
            content_coords = [(r0 + 1 + r, c0 + 1 + c) for r, c in content_shape]
            return {
                "bbox": bbox,
                "marker": marker,
                "corner": quadrant,
                "direction": direction,
                "border_color": int(border_color),
                "content_color": int(content_color),
                "content_coords": content_coords,
                "count": count,
                "ray_cells": ray_cells[:count],
            }
    return None


def _render_input(shape: tuple[int, int], frames: list[dict]) -> np.ndarray:
    grid = np.full(shape, Color.BLACK, dtype=int)
    for frame in frames:
        _paint_perimeter(grid, frame["bbox"], frame["border_color"])
        mr, mc = frame["marker"]
        grid[mr, mc] = Color.YELLOW
        for r, c in frame["content_coords"]:
            grid[r, c] = frame["content_color"]
    return grid


def _manual_output(shape: tuple[int, int], frames: list[dict]) -> np.ndarray:
    out = _render_input(shape, frames)
    for frame in sorted(frames, key=lambda item: (item["bbox"][1], item["bbox"][0])):
        for r, c in frame["content_coords"]:
            out[r, c] = Color.BLACK
        mr, mc = frame["marker"]
        dr, dc = frame["direction"]
        for step in range(frame["count"]):
            rr = mr + step * dr
            cc = mc + step * dc
            if not in_bounds(shape, rr, cc):
                break
            out[rr, cc] = frame["content_color"]
        _paint_perimeter(out, frame["bbox"], frame["content_color"])
    return out


def _validate_instance(inp: np.ndarray, frames: list[dict]) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if len(frames) not in (2, 3):
        return False
    if len({frame["border_color"] for frame in frames}) != len(frames):
        return False
    if len({frame["content_color"] for frame in frames}) != len(frames):
        return False
    if len({frame["corner"] for frame in frames}) < 2:
        return False
    if len({(frame["count"], frame["bbox"][2] - frame["bbox"][0], frame["bbox"][3] - frame["bbox"][1]) for frame in frames}) == 1:
        return False

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        if min(r0, c0, inp.shape[0] - 1 - r1, inp.shape[1] - 1 - c1) < EDGE_GAP:
            return False
        if frame["corner"] != _expected_corner(frame["bbox"], inp.shape):
            return False
        if not (CONTENT_COUNT_RANGE[0] <= frame["count"] <= CONTENT_COUNT_RANGE[1]):
            return False

        content = set(frame["content_coords"])
        seen = set()
        stack = [next(iter(content))]
        while stack:
            cell = stack.pop()
            if cell in seen:
                continue
            seen.add(cell)
            r, c = cell
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nxt = (r + dr, c + dc)
                if nxt in content and nxt not in seen:
                    stack.append(nxt)
        if seen != content:
            return False

    for i, left in enumerate(frames):
        for right in frames[i + 1 :]:
            if not _gap_at_least(left["bbox"], right["bbox"], EDGE_GAP):
                return False
            if set(left["ray_cells"]) & set(right["ray_cells"]):
                return False
            for rr, cc in left["ray_cells"][1:]:
                br0, bc0, br1, bc1 = right["bbox"]
                if br0 <= rr <= br1 and bc0 <= cc <= bc1:
                    return False
            for rr, cc in right["ray_cells"][1:]:
                br0, bc0, br1, bc1 = left["bbox"]
                if br0 <= rr <= br1 and bc0 <= cc <= bc1:
                    return False

    if int(np.sum(inp == Color.YELLOW)) != len(frames):
        return False

    try:
        detected = _detect_frames(inp)
    except Exception:
        return False

    if len(detected) != len(frames):
        return False

    want = sorted(
        (frame["bbox"], frame["marker"], frame["corner"], frame["content_color"], frame["count"])
        for frame in frames
    )
    got = sorted(
        (frame["bbox"], frame["marker"], frame["corner"], frame["content_color"], frame["count"])
        for frame in detected
    )
    if want != got:
        return False

    expected = _manual_output(inp.shape, frames)
    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt(rng: random.Random, frame_count: int) -> np.ndarray | None:
    h = rng.randint(*GRID_SIDE_RANGE)
    w = rng.randint(*GRID_SIDE_RANGE)
    quadrants = rng.sample(QUADRANTS, frame_count)
    border_colors = rng.sample(FRAME_COLORS, frame_count)
    content_colors = rng.sample(CONTENT_COLORS, frame_count)

    frames = []
    used_ray_cells: set[tuple[int, int]] = set()
    for quadrant, border_color, content_color in zip(quadrants, border_colors, content_colors):
        frame = _frame_from_params(
            rng,
            (h, w),
            quadrant,
            border_color,
            content_color,
            frames,
            used_ray_cells,
        )
        if frame is None:
            return None
        frames.append(frame)
        used_ray_cells.update(frame["ray_cells"])

    inp = _render_input((h, w), frames)
    if not _validate_instance(inp, frames):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    target_count = rng.choice((2, 3))
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng, target_count)
        if grid is not None:
            return grid
    fallback_count = 5 - target_count
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng, fallback_count)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
