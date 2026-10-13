from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
MARKER = 9
SPRITE_COLORS = [1, 2, 3, 4, 5, 6, 7, 8]
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}

SHAPES = {
    "up": {(-1, 0), (0, -1), (0, 0), (0, 1)},
    "down": {(1, 0), (0, -1), (0, 0), (0, 1)},
    "left": {(-1, 0), (0, 0), (1, 0), (0, -1)},
    "right": {(-1, 0), (0, 0), (1, 0), (0, 1)},
}


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _normalized(coords: set[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return frozenset((r - r0, c - c0) for r, c in coords)


ORIENTATION_PATTERNS = {
    _normalized({(r, c) for r, c in cells}): orient for orient, cells in SHAPES.items()
}


def _build_marker_options():
    options = {orient: {push: [] for push in DIRS} for orient in SHAPES}
    for orient, cells in SHAPES.items():
        for r, c in cells:
            for contact, (dr, dc) in DIRS.items():
                marker = (r + dr, c + dc)
                if marker in cells:
                    continue
                touches = sum(abs(rr - marker[0]) + abs(cc - marker[1]) == 1 for rr, cc in cells)
                if touches != 1:
                    continue
                push = OPPOSITE[contact]
                if marker not in options[orient][push]:
                    options[orient][push].append(marker)
    return options


MARKER_OPTIONS = _build_marker_options()


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (BG, MARKER) or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [rr for rr, _ in cells]
            cols = [cc for _, cc in cells]
            components.append(
                {
                    "color": color,
                    "coords": np.array(cells, dtype=int),
                    "area": len(cells),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return components


def _decode_orientation(coords: np.ndarray) -> str:
    pts = {(int(r), int(c)) for r, c in coords.tolist()}
    pattern = _normalized(pts)
    assert len(pts) == 4 and pattern in ORIENTATION_PATTERNS, "expected a 4-cell T sprite"
    return ORIENTATION_PATTERNS[pattern]


def _distance_to_boundary(coords: np.ndarray, push: str, shape: tuple[int, int]) -> int:
    h, w = shape
    if push == "up":
        return int(np.min(coords[:, 0]))
    if push == "down":
        return int(h - 1 - np.max(coords[:, 0]))
    if push == "left":
        return int(np.min(coords[:, 1]))
    return int(w - 1 - np.max(coords[:, 1]))


def _parse_grid(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    marker_cells = {tuple(cell) for cell in np.argwhere(g == MARKER).tolist()}
    used_markers: set[tuple[int, int]] = set()
    infos = []
    for comp in _find_components(g):
        assert comp["area"] == 4, "every sprite must have exactly four cells"
        coords = comp["coords"]
        orientation = _decode_orientation(coords)
        touches = []
        for r, c in coords.tolist():
            for nr, nc in _neighbors4(int(r), int(c)):
                if (nr, nc) in marker_cells:
                    touches.append(((int(r), int(c)), (nr, nc)))
        assert len(touches) == 1, "every sprite must touch exactly one single-cell marker"
        (sr, sc), marker = touches[0]
        assert marker not in used_markers, "a marker cannot serve multiple sprites"
        used_markers.add(marker)
        dr = marker[0] - sr
        dc = marker[1] - sc
        contact = next(name for name, vec in DIRS.items() if vec == (dr, dc))
        push = OPPOSITE[contact]
        bbox = comp["bbox"]
        infos.append(
            {
                "color": int(comp["color"]),
                "coords": coords.copy(),
                "bbox": bbox,
                "left_col": int(bbox[1]),
                "top_row": int(bbox[0]),
                "orientation": orientation,
                "marker": marker,
                "push": push,
                "distance": _distance_to_boundary(coords, push, g.shape),
            }
        )
    return infos


def _simulate(grid: np.ndarray, *, keep_all: bool = False, with_info: bool = False):
    g = np.asarray(grid, dtype=int)
    infos = _parse_grid(g)
    movers = []
    for info in infos:
        matched = info["orientation"] == info["push"]
        if keep_all or matched:
            item = dict(info)
            item["matched"] = matched
            movers.append(item)

    key_pairs = [(item["distance"], item["left_col"]) for item in movers]
    assert len(key_pairs) == len(set(key_pairs)), "survivor settle order must be unique"

    out = np.full_like(g, BG)
    marker_coords = np.argwhere(g == MARKER)
    for r, c in marker_coords:
        out[int(r), int(c)] = MARKER

    occupied = {tuple(cell) for cell in marker_coords.tolist()}
    finals: dict[int, np.ndarray] = {}
    h, w = g.shape
    for info in sorted(movers, key=lambda item: (item["distance"], item["left_col"], item["top_row"], item["color"])):
        dr, dc = DIRS[info["push"]]
        coords = info["coords"].copy()
        while True:
            nxt = coords + np.array([dr, dc], dtype=int)
            ok = True
            for rr, cc in nxt.tolist():
                if not (0 <= rr < h and 0 <= cc < w) or (rr, cc) in occupied:
                    ok = False
                    break
            if not ok:
                break
            coords = nxt
        finals[info["color"]] = coords.copy()
        for rr, cc in coords.tolist():
            occupied.add((int(rr), int(cc)))
            out[int(rr), int(cc)] = int(info["color"])

    if with_info:
        return out, infos, finals
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the sprites whose geometric orientation matches their push direction, then slide the survivors until blocked.

    Input structure:
    - The background is black.
    - Several 4-cell monochrome T-shaped sprites appear in distinct colors, plus single maroon marker pixels.
    - Each sprite touches exactly one maroon marker on one cardinal side, which defines the contact side.

    Rule:
    - For each sprite, push direction is the direction away from its marker contact side.
    - Decode the sprite orientation from the T geometry: from the center of the 3-cell bar toward the 4th protruding cell.
    - If orientation matches push direction, keep the sprite; otherwise erase it while leaving the marker untouched.
    - After all erasures, slide the kept sprites rigidly in order of increasing distance to the target boundary, breaking ties by leftmost column, until the next step would hit the boundary, a marker, or an already settled moved sprite.

    Key insight:
    - The T sprite encodes a direction twice: the marker chooses where it wants to move, while the protruding arm decides whether it is even allowed to survive and move.
    """
    return _simulate(np.asarray(grid, dtype=int), keep_all=False, with_info=False)


def _shape_cells(pivot: tuple[int, int], orientation: str) -> set[tuple[int, int]]:
    pr, pc = pivot
    return {(pr + dr, pc + dc) for dr, dc in SHAPES[orientation]}


def _all_package_candidates(
    h: int,
    w: int,
    orientation: str,
    push: str,
    sprite_cells: set[tuple[int, int]],
    marker_cells: set[tuple[int, int]],
) -> list[dict]:
    candidates = []
    for pr in range(h):
        for pc in range(w):
            shape = _shape_cells((pr, pc), orientation)
            if any(not (0 <= r < h and 0 <= c < w) for r, c in shape):
                continue
            if shape & sprite_cells:
                continue
            blocked = False
            for r, c in shape:
                if (r, c) in marker_cells:
                    blocked = True
                    break
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) in sprite_cells or (nr, nc) in marker_cells:
                        blocked = True
                        break
                if blocked:
                    break
            if blocked:
                continue
            for mr_off, mc_off in MARKER_OPTIONS[orientation][push]:
                marker = (pr + mr_off, pc + mc_off)
                if not (0 <= marker[0] < h and 0 <= marker[1] < w):
                    continue
                if marker in sprite_cells or marker in marker_cells:
                    continue
                touching = [cell for cell in shape if abs(cell[0] - marker[0]) + abs(cell[1] - marker[1]) == 1]
                if len(touching) != 1:
                    continue
                ok = True
                for nr, nc in _neighbors4(*marker):
                    if (nr, nc) in sprite_cells and (nr, nc) not in shape:
                        ok = False
                        break
                    if (nr, nc) in marker_cells:
                        ok = False
                        break
                if not ok:
                    continue
                rows = [r for r, _ in shape]
                cols = [c for _, c in shape]
                candidates.append(
                    {
                        "pivot": (pr, pc),
                        "shape": shape,
                        "marker": marker,
                        "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    }
                )
    return candidates


def _place_package(
    grid: np.ndarray,
    color: int,
    orientation: str,
    push: str,
    sprite_cells: set[tuple[int, int]],
    marker_cells: set[tuple[int, int]],
    rng: random.Random,
    preferred: callable | None = None,
):
    candidates = _all_package_candidates(grid.shape[0], grid.shape[1], orientation, push, sprite_cells, marker_cells)
    if preferred is not None:
        candidates = [cand for cand in candidates if preferred(cand)]
    if not candidates:
        return None
    choice = rng.choice(candidates)
    for r, c in choice["shape"]:
        grid[r, c] = color
        sprite_cells.add((r, c))
    mr, mc = choice["marker"]
    grid[mr, mc] = MARKER
    marker_cells.add((mr, mc))
    return choice


def _place_interaction_pair(grid: np.ndarray, colors: list[int], rng: random.Random, sprite_cells, marker_cells):
    lane_dir = rng.choice(list(DIRS))
    front_orientation = OPPOSITE[lane_dir]
    back_orientation = lane_dir
    h, w = grid.shape

    def front_pref(cand):
        r0, c0, r1, c1 = cand["bbox"]
        if lane_dir == "up":
            return r0 <= max(2, h // 3)
        if lane_dir == "down":
            return r1 >= h - 1 - max(2, h // 3)
        if lane_dir == "left":
            return c0 <= max(2, w // 3)
        return c1 >= w - 1 - max(2, w // 3)

    for _ in range(60):
        front = _place_package(
            grid, colors[0], front_orientation, lane_dir, sprite_cells, marker_cells, rng, preferred=front_pref
        )
        if front is None:
            return None
        pivot = front["pivot"]

        def back_pref(cand):
            if lane_dir in ("up", "down"):
                if cand["pivot"][1] != pivot[1]:
                    return False
                gap = abs(cand["pivot"][0] - pivot[0])
                if gap < 4:
                    return False
                return cand["pivot"][0] > pivot[0] if lane_dir == "up" else cand["pivot"][0] < pivot[0]
            if cand["pivot"][0] != pivot[0]:
                return False
            gap = abs(cand["pivot"][1] - pivot[1])
            if gap < 4:
                return False
            return cand["pivot"][1] > pivot[1] if lane_dir == "left" else cand["pivot"][1] < pivot[1]

        back = _place_package(
            grid, colors[1], back_orientation, lane_dir, sprite_cells, marker_cells, rng, preferred=back_pref
        )
        if back is not None:
            return lane_dir

        for r, c in front["shape"]:
            grid[r, c] = BG
            sprite_cells.remove((r, c))
        grid[front["marker"]] = BG
        marker_cells.remove(front["marker"])
    return None


def _validate_semantics(grid: np.ndarray) -> bool:
    try:
        actual, infos, finals = _simulate(grid, keep_all=False, with_info=True)
        _, _, finals_all = _simulate(grid, keep_all=True, with_info=True)
    except AssertionError:
        return False

    if len(infos) < 5 or len(infos) > 7:
        return False

    matched = [info for info in infos if info["orientation"] == info["push"]]
    erased = [info for info in infos if info["orientation"] != info["push"]]
    if len(matched) < 2 or len(erased) < 2:
        return False

    pushes = {info["push"] for info in infos}
    if not (pushes & {"up", "down"}) or not (pushes & {"left", "right"}):
        return False

    changed = 0
    for info in matched:
        color = info["color"]
        if color in finals_all and not np.array_equal(finals[color], finals_all[color]):
            changed += 1
    if changed == 0:
        return False

    if np.count_nonzero(actual == MARKER) != len(infos):
        return False

    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(11, 14)
    w = rng.randint(14, 17)
    count = rng.randint(5, 7)
    colors = rng.sample(SPRITE_COLORS, count)
    grid = np.zeros((h, w), dtype=int)
    sprite_cells: set[tuple[int, int]] = set()
    marker_cells: set[tuple[int, int]] = set()

    lane_dir = _place_interaction_pair(grid, colors[:2], rng, sprite_cells, marker_cells)
    if lane_dir is None:
        return None

    package_specs = []
    perp_dirs = ["left", "right"] if lane_dir in ("up", "down") else ["up", "down"]
    package_specs.append((rng.choice(perp_dirs), True))
    package_specs.append((rng.choice(list(DIRS)), False))
    while len(package_specs) < count - 2:
        push = rng.choice(list(DIRS))
        match = rng.random() < 0.5
        package_specs.append((push, match))

    for color, (push, match) in zip(colors[2:], package_specs):
        if match:
            orientation = push
        else:
            choices = [name for name in DIRS if name != push]
            orientation = rng.choice(choices)
        placed = _place_package(grid, color, orientation, push, sprite_cells, marker_cells, rng)
        if placed is None:
            return None

    if not _validate_semantics(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for seed in range(600):
        rng = random.Random(random.randrange(1 << 30) ^ (seed * 104729))
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
