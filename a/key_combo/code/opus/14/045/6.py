from __future__ import annotations

import hashlib
import random
import sys

import numpy as np

ARC_UTILS_DIR = "/home/lixinyu/snap/new/solvers"
if ARC_UTILS_DIR not in sys.path:
    sys.path.append(ARC_UTILS_DIR)

from arc_utils import Color, find_components, get_bg


BG = int(Color.BLACK)
MARKER = int(Color.RED)
TRAIL = int(Color.YELLOW)
BORDER = int(Color.GREY)
SPRITE_COLORS = [1, 3, 6, 7, 8, 9]
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}

# Right-pointing connected shapes with a unique tip at (0, 0).
BASE_RIGHT_SHAPES = [
    ((0, 0), (0, -1), (-1, -1)),
    ((0, 0), (0, -1), (1, -1)),
    ((0, 0), (0, -1), (-1, -1), (1, -1)),
    ((0, 0), (0, -1), (-1, -1), (-1, -2)),
    ((0, 0), (0, -1), (1, -1), (1, -2)),
    ((0, 0), (0, -1), (-1, -1), (1, -1), (-1, -2)),
    ((0, 0), (0, -1), (-1, -1), (1, -1), (1, -2)),
    ((0, 0), (0, -1), (-1, -1), (1, -1), (-1, -2), (1, -2)),
]

GRID_MIN = 13
GRID_MAX = 18
MAX_ATTEMPTS = 400
PER_SPRITE_ATTEMPTS = 120

_CYCLES: dict[str, list] = {}
_SEEN_INPUTS: set[str] = set()
_SEEN_OUTPUTS: set[str] = set()


def _cycled_choice(name: str, options):
    pool = _CYCLES.setdefault(name, [])
    if not pool:
        pool[:] = list(options)
        random.shuffle(pool)
    return pool.pop()


def _grid_signature(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(tuple(grid.shape)).encode()).hexdigest()


def _shift_cells(cells: set[tuple[int, int]], dr: int, dc: int, steps: int) -> set[tuple[int, int]]:
    return {(r + dr * steps, c + dc * steps) for r, c in cells}


def _neighbors4(cell: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r, c = cell
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _rotate_shape(shape: tuple[tuple[int, int], ...], direction: str) -> set[tuple[int, int]]:
    if direction == "right":
        return {(r, c) for r, c in shape}
    if direction == "left":
        return {(-r, -c) for r, c in shape}
    if direction == "down":
        return {(c, -r) for r, c in shape}
    if direction == "up":
        return {(-c, r) for r, c in shape}
    raise AssertionError(direction)


def _tip_of(cells: set[tuple[int, int]], direction: str) -> tuple[int, int]:
    if direction == "right":
        edge = max(c for _r, c in cells)
        tips = [(r, c) for r, c in cells if c == edge]
    elif direction == "left":
        edge = min(c for _r, c in cells)
        tips = [(r, c) for r, c in cells if c == edge]
    elif direction == "down":
        edge = max(r for r, _c in cells)
        tips = [(r, c) for r, c in cells if r == edge]
    else:
        edge = min(r for r, _c in cells)
        tips = [(r, c) for r, c in cells if r == edge]
    assert len(tips) == 1, "leading edge width must be exactly one pixel"
    return tips[0]


def _marker_candidates(cells: set[tuple[int, int]], direction: str) -> list[tuple[int, int]]:
    dr, dc = DIRS[direction]
    candidates = []
    for r, c in cells:
        marker = (r - dr, c - dc)
        if marker in cells:
            continue
        if sum(neigh in cells for neigh in _neighbors4(marker)) != 1:
            continue
        candidates.append(marker)
    return sorted(set(candidates))


def _trail_cells(
    start_cells: set[tuple[int, int]],
    direction: str,
    shift: int,
) -> list[tuple[int, int]]:
    dr, dc = DIRS[direction]
    tip = _tip_of(start_cells, direction)
    final_cells = _shift_cells(start_cells, dr, dc, shift)
    positions = [(tip[0] + dr * t, tip[1] + dc * t) for t in range(1, shift)]
    return [pos for pos in positions if pos not in start_cells and pos not in final_cells]


def _build_layout(
    shape: set[tuple[int, int]],
    direction: str,
    shift: int,
    cross: int,
    h: int,
    w: int,
) -> dict | None:
    rows = [r for r, _c in shape]
    cols = [c for _r, c in shape]
    dr, dc = DIRS[direction]

    if direction == "right":
        tip = (cross, w - 2 - shift)
    elif direction == "left":
        tip = (cross, 1 + shift)
    elif direction == "down":
        tip = (h - 2 - shift, cross)
    else:
        tip = (1 + shift, cross)

    start_cells = {(tip[0] + r, tip[1] + c) for r, c in shape}
    final_cells = _shift_cells(start_cells, dr, dc, shift)
    if not all(1 <= r <= h - 2 and 1 <= c <= w - 2 for r, c in start_cells | final_cells):
        return None

    trail = _trail_cells(start_cells, direction, shift)
    if len(trail) < 3:
        return None

    marker_options = [
        marker
        for marker in _marker_candidates(start_cells, direction)
        if 1 <= marker[0] <= h - 2 and 1 <= marker[1] <= w - 2
    ]
    if not marker_options:
        return None

    swept = set()
    for t in range(shift + 1):
        swept |= _shift_cells(start_cells, dr, dc, t)

    return {
        "direction": direction,
        "shift": shift,
        "start": start_cells,
        "final": final_cells,
        "trail": trail,
        "swept": swept,
        "marker_options": marker_options,
        "tip": _tip_of(start_cells, direction),
        "bbox": (min(rows), min(cols), max(rows), max(cols)),
    }


def _sprite_specs(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert int(get_bg(g)) == BG
    assert np.all(g[0, :] == BORDER) and np.all(g[-1, :] == BORDER)
    assert np.all(g[:, 0] == BORDER) and np.all(g[:, -1] == BORDER)
    assert not np.any(g[1:-1, 1:-1] == BORDER)
    assert not np.any(g == TRAIL), "input must not contain yellow"

    red_pixels = {tuple(map(int, pos)) for pos in np.argwhere(g == MARKER)}
    assert 2 <= len(red_pixels) <= 3

    sprite_colors = [color for color in np.unique(g) if color not in (BG, MARKER, TRAIL, BORDER)]
    sprite_colors = [int(color) for color in sprite_colors]
    assert 2 <= len(sprite_colors) <= 3
    assert len(sprite_colors) == len(red_pixels)

    comps = find_components(g, bg=BG, colors=sprite_colors, connectivity=4, monochromatic=True)
    assert len(comps) == len(sprite_colors)

    infos = []
    used_markers = set()
    all_sprite_cells = set()
    for comp in comps:
        color = int(comp["color"])
        cells = {tuple(map(int, pos)) for pos in comp["coords"]}
        assert 3 <= len(cells) <= 6
        all_sprite_cells |= cells

        touching_markers = set()
        for cell in cells:
            for neigh in _neighbors4(cell):
                if neigh in red_pixels:
                    touching_markers.add(neigh)
        assert len(touching_markers) == 1
        marker = next(iter(touching_markers))
        used_markers.add(marker)

        adjacent_sprite_cells = [neigh for neigh in _neighbors4(marker) if neigh in cells]
        assert len(adjacent_sprite_cells) == 1
        sr, sc = adjacent_sprite_cells[0]
        direction_vec = (sr - marker[0], sc - marker[1])
        direction = next(name for name, vec in DIRS.items() if vec == direction_vec)

        tip = _tip_of(cells, direction)
        if direction == "right":
            shift = w - 2 - tip[1]
        elif direction == "left":
            shift = tip[1] - 1
        elif direction == "down":
            shift = h - 2 - tip[0]
        else:
            shift = tip[0] - 1
        assert shift >= 1

        dr, dc = DIRS[direction]
        final_cells = _shift_cells(cells, dr, dc, shift)
        assert all(1 <= r <= h - 2 and 1 <= c <= w - 2 for r, c in final_cells)

        for r, c in cells:
            for t in range(1, shift + 1):
                pos = (r + dr * t, c + dc * t)
                assert 0 <= pos[0] < h and 0 <= pos[1] < w
                if pos not in cells:
                    assert g[pos] == BG

        trail = _trail_cells(cells, direction, shift)
        info = {
            "color": color,
            "start": cells,
            "marker": marker,
            "direction": direction,
            "shift": shift,
            "tip": tip,
            "final": final_cells,
            "trail": trail,
        }
        infos.append(info)

    assert used_markers == red_pixels
    for info in infos:
        foreign = all_sprite_cells - info["start"]
        assert all(neigh not in foreign for cell in info["start"] for neigh in _neighbors4(cell))
        assert len(info["trail"]) >= 3

    trail_sets = [set(info["trail"]) for info in infos]
    for i in range(len(trail_sets)):
        for j in range(i + 1, len(trail_sets)):
            assert not (trail_sets[i] & trail_sets[j])

    infos.sort(key=lambda item: item["color"])
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each marked sprite away from its red marker until it reaches the grey border, and paint the traveled tip track with alternating colors.

    Input structure:
    - The background is black, with a 1-pixel grey border around the whole grid.
    - Inside the border there are 2-3 small monochrome 4-connected sprites in distinct colors, plus one red marker pixel orthogonally adjacent to each sprite.
    - Each marker sits on exactly one side of its sprite, and each sprite has a one-pixel leading tip in the opposite sliding direction.

    Rule:
    - For each sprite, read the marker side and slide the whole sprite in the opposite cardinal direction until its leading tip becomes orthogonally adjacent to the grey border.
    - The intermediate positions visited by that leading tip, excluding positions occupied by the start sprite and the final sprite, form the trajectory.
    - Recolor the trajectory from the final-near end outward: even indices keep the sprite color, odd indices become yellow; erase every red marker; keep the final sprite in its original color.

    Key insight:
    - The red marker determines the slide direction, while the collision distance to the border determines the trajectory length, so the alternating recolor must be anchored to the true stopping position rather than to a fixed template.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    infos = _sprite_specs(g)

    out = np.full((h, w), BG, dtype=int)
    out[0, :] = BORDER
    out[-1, :] = BORDER
    out[:, 0] = BORDER
    out[:, -1] = BORDER

    for info in infos:
        for idx, (r, c) in enumerate(reversed(info["trail"])):
            out[r, c] = info["color"] if idx % 2 == 0 else TRAIL
        for r, c in info["final"]:
            out[r, c] = info["color"]
    return out


def _alt_start_near_output(grid: np.ndarray) -> np.ndarray:
    infos = _sprite_specs(grid)
    out = np.full_like(grid, BG)
    out[0, :] = BORDER
    out[-1, :] = BORDER
    out[:, 0] = BORDER
    out[:, -1] = BORDER
    for info in infos:
        for idx, (r, c) in enumerate(info["trail"]):
            out[r, c] = info["color"] if idx % 2 == 0 else TRAIL
        for r, c in info["final"]:
            out[r, c] = info["color"]
    return out


def _draw_instance(h: int, w: int, specs: list[dict]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    grid[0, :] = BORDER
    grid[-1, :] = BORDER
    grid[:, 0] = BORDER
    grid[:, -1] = BORDER
    for spec in specs:
        for r, c in spec["start"]:
            grid[r, c] = spec["color"]
        grid[spec["marker"]] = MARKER
    return grid


def _distance_to_borders(info: dict, shape: tuple[int, int]) -> dict[str, int]:
    h, w = shape
    r, c = info["tip"]
    return {
        "up": r - 1,
        "down": h - 2 - r,
        "left": c - 1,
        "right": w - 2 - c,
    }


def _semantic_check(grid: np.ndarray) -> bool:
    try:
        infos = _sprite_specs(grid)
        out = main(grid)
    except Exception:
        return False

    lengths = [len(info["trail"]) for info in infos]
    if len(set(lengths)) < 2:
        return False
    if not any(length % 2 == 0 for length in lengths):
        return False
    if not any(length % 2 == 1 for length in lengths):
        return False
    if not any(length > 3 for length in lengths):
        return False
    if int(np.sum(out == TRAIL)) < len(infos):
        return False
    if np.array_equal(out, _alt_start_near_output(grid)):
        return False

    counterexample = False
    for info in infos:
        dists = _distance_to_borders(info, grid.shape)
        nearest = min(dists.values())
        nearest_dirs = [name for name, dist in dists.items() if dist == nearest]
        if len(nearest_dirs) == 1 and nearest_dirs[0] != info["direction"]:
            counterexample = True
            break
    if not counterexample:
        return False

    return True


def _direction_patterns(count: int) -> tuple[str, ...]:
    if count == 2:
        patterns = [
            ("right", "up"),
            ("left", "down"),
            ("right", "left"),
            ("up", "down"),
            ("right", "down"),
            ("left", "up"),
        ]
        return _cycled_choice("pattern2", patterns)
    patterns = [
        ("right", "left", "up"),
        ("right", "up", "down"),
        ("left", "up", "down"),
        ("right", "left", "down"),
    ]
    return _cycled_choice("pattern3", patterns)


def _choose_cross_values(shape: set[tuple[int, int]], direction: str, h: int, w: int) -> list[int]:
    rows = [r for r, _c in shape]
    cols = [c for _r, c in shape]
    if direction in ("left", "right"):
        lo = 1 - min(rows)
        hi = h - 2 - max(rows)
    else:
        lo = 1 - min(cols)
        hi = w - 2 - max(cols)
    if lo > hi:
        return []
    values = list(range(lo, hi + 1))
    random.shuffle(values)
    return values


def _candidate_shifts(shape: set[tuple[int, int]], direction: str, h: int, w: int) -> list[int]:
    if direction == "right":
        limit = w - 3 + min(c for _r, c in shape)
    elif direction == "left":
        limit = w - 3 - max(c for _r, c in shape)
    elif direction == "down":
        limit = h - 3 + min(r for r, _c in shape)
    else:
        limit = h - 3 - max(r for r, _c in shape)
    if limit < 5:
        return []

    shifts = [shift for shift in range(5, limit + 1) if len(_trail_cells(shape, direction, shift)) >= 3]
    random.shuffle(shifts)
    return shifts


def _accept_layout(
    layout: dict,
    occupied_sprite_cells: set[tuple[int, int]],
    occupied_input: set[tuple[int, int]],
    occupied_swept: set[tuple[int, int]],
) -> dict | None:
    if layout["swept"] & occupied_swept:
        return None
    if layout["start"] & occupied_input:
        return None

    for cell in layout["start"]:
        for neigh in _neighbors4(cell):
            if neigh in occupied_sprite_cells:
                return None

    marker_options = layout["marker_options"][:]
    random.shuffle(marker_options)
    for marker in marker_options:
        if marker in occupied_input or marker in occupied_swept:
            continue
        forbidden = (occupied_input - layout["start"]) | occupied_sprite_cells
        if sum(neigh in layout["start"] for neigh in _neighbors4(marker)) != 1:
            continue
        if any(neigh in forbidden for neigh in _neighbors4(marker)):
            continue
        placed = dict(layout)
        placed["marker"] = marker
        return placed
    return None


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    count = int(_cycled_choice("sprite_count", [2, 3]))
    directions = list(_direction_patterns(count))
    random.shuffle(directions)
    colors = random.sample(SPRITE_COLORS, count)

    occupied_input = {
        *((0, c) for c in range(w)),
        *((h - 1, c) for c in range(w)),
        *((r, 0) for r in range(h)),
        *((r, w - 1) for r in range(h)),
    }
    occupied_sprite_cells: set[tuple[int, int]] = set()
    occupied_swept: set[tuple[int, int]] = set()
    specs = []

    for idx in range(count):
        direction = directions[idx]
        color = colors[idx]
        placed = None

        for _ in range(PER_SPRITE_ATTEMPTS):
            shape = _rotate_shape(random.choice(BASE_RIGHT_SHAPES), direction)
            shifts = _candidate_shifts(shape, direction, h, w)
            if not shifts:
                continue
            crosses = _choose_cross_values(shape, direction, h, w)
            if not crosses:
                continue

            if idx == 0:
                shifts.sort(reverse=True)

            for shift in shifts[: min(len(shifts), 8)]:
                for cross in crosses[: min(len(crosses), 10)]:
                    layout = _build_layout(shape, direction, shift, cross, h, w)
                    if layout is None:
                        continue
                    placed = _accept_layout(layout, occupied_sprite_cells, occupied_input, occupied_swept)
                    if placed is not None:
                        break
                if placed is not None:
                    break
            if placed is not None:
                break

        if placed is None:
            return None

        placed["color"] = color
        specs.append(placed)
        occupied_input |= placed["start"] | {placed["marker"]}
        occupied_sprite_cells |= placed["start"]
        occupied_swept |= placed["swept"]

    grid = _draw_instance(h, w, specs)
    if not _semantic_check(grid):
        return None

    out = main(grid)
    in_sig = _grid_signature(grid)
    out_sig = _grid_signature(out)
    if in_sig in _SEEN_INPUTS or out_sig in _SEEN_OUTPUTS:
        return None
    _SEEN_INPUTS.add(in_sig)
    _SEEN_OUTPUTS.add(out_sig)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sliding alternating-track input")
