from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9

DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= int(r) < h and 0 <= int(c) < w


def _perp(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return -int(dc), int(dr)


def _add(a: tuple[int, int], b: tuple[int, int], k: int = 1) -> tuple[int, int]:
    return int(a[0] + k * b[0]), int(a[1] + k * b[1])


def _proj(p: tuple[int, int], direction: tuple[int, int]) -> int:
    return int(p[0] * direction[0] + p[1] * direction[1])


def _arrow_pixels(tail: tuple[int, int], direction: tuple[int, int]) -> set[tuple[int, int]]:
    mid = _add(tail, direction)
    head = _add(tail, direction, 2)
    side = _perp(direction)
    nose = _add(head, direction)
    return {tail, mid, head, _add(nose, side), _add(nose, side, -1)}


def _components(grid: np.ndarray, color: int, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    if int(connectivity) == 8:
        steps = DIRS + ((-1, -1), (-1, 1), (1, -1), (1, 1))
    else:
        steps = DIRS

    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or int(grid[sr, sc]) != int(color):
                continue
            q = deque([(sr, sc)])
            seen[sr, sc] = True
            pts: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                pts.append((int(r), int(c)))
                for dr, dc in steps:
                    rr = int(r) + dr
                    cc = int(c) + dc
                    if not _in_bounds(grid.shape, rr, cc):
                        continue
                    if seen[rr, cc] or int(grid[rr, cc]) != int(color):
                        continue
                    seen[rr, cc] = True
                    q.append((rr, cc))
            comps.append(pts)
    return comps


def _non_arrow_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for sr in range(h):
        for sc in range(w):
            color = int(grid[sr, sc])
            if color in (BLACK, MAROON) or seen[sr, sc]:
                continue
            q = deque([(sr, sc)])
            seen[sr, sc] = True
            pts: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                pts.append((int(r), int(c)))
                for dr, dc in DIRS:
                    rr = int(r) + dr
                    cc = int(c) + dc
                    if not _in_bounds(grid.shape, rr, cc):
                        continue
                    if seen[rr, cc] or int(grid[rr, cc]) != color:
                        continue
                    seen[rr, cc] = True
                    q.append((rr, cc))
            comps.append({"color": color, "coords": pts})
    return comps


def _decode_arrow(comp: list[tuple[int, int]]) -> dict | None:
    pts = set(comp)
    if len(pts) != 5:
        return None

    matches: list[dict] = []
    for tail in pts:
        for direction in DIRS:
            shape = _arrow_pixels(tail, direction)
            if shape == pts:
                matches.append(
                    {
                        "tail": tail,
                        "head": _add(tail, direction, 2),
                        "direction": direction,
                        "pixels": pts,
                    }
                )
    if len(matches) != 1:
        return None
    return matches[0]


def _is_perpendicular_segment(coords: list[tuple[int, int]], direction: tuple[int, int]) -> bool:
    if not (2 <= len(coords) <= 5):
        return False
    rows = sorted({int(r) for r, _ in coords})
    cols = sorted({int(c) for _, c in coords})

    if int(direction[0]) != 0:
        if len(rows) != 1 or len(cols) != len(coords):
            return False
        return cols == list(range(cols[0], cols[-1] + 1))

    if len(cols) != 1 or len(rows) != len(coords):
        return False
    return rows == list(range(rows[0], rows[-1] + 1))


def _touches_tail(coords: list[tuple[int, int]], tail: tuple[int, int]) -> bool:
    tr, tc = tail
    return any(abs(int(r) - tr) + abs(int(c) - tc) == 1 for r, c in coords)


def _find_anchor(grid: np.ndarray, head: tuple[int, int], direction: tuple[int, int], color: int) -> tuple[int, int] | None:
    p = _add(head, direction)
    while _in_bounds(grid.shape, p[0], p[1]):
        if int(grid[p[0], p[1]]) == int(color):
            return p
        p = _add(p, direction)
    return None


def _translated(coords: list[tuple[int, int]], direction: tuple[int, int], shift: int) -> list[tuple[int, int]]:
    dr, dc = direction
    return [(int(r + shift * dr), int(c + shift * dc)) for r, c in coords]


def _solve_moves(grid: np.ndarray) -> list[dict]:
    arrows: list[dict] = []
    for comp in _components(grid, MAROON, connectivity=8):
        arrow = _decode_arrow(comp)
        assert arrow is not None, "each maroon component must be one arrow"
        arrows.append(arrow)
    assert arrows, "expected at least one arrow"

    colored = _non_arrow_components(grid)
    moves: list[dict] = []
    used_segments: set[tuple[tuple[int, int], ...]] = set()

    for arrow in sorted(arrows, key=lambda a: (a["tail"][0], a["tail"][1], a["direction"])):
        tail = arrow["tail"]
        direction = arrow["direction"]

        candidates: list[dict] = []
        for comp in colored:
            coords = list(comp["coords"])
            if not _touches_tail(coords, tail):
                continue
            if not _is_perpendicular_segment(coords, direction):
                continue
            anchor = _find_anchor(grid, arrow["head"], direction, int(comp["color"]))
            if anchor is None:
                continue
            key = tuple(sorted(coords))
            if key in used_segments:
                continue
            candidates.append({"component": comp, "anchor": anchor, "key": key})

        assert len(candidates) == 1, "each arrow tail must touch exactly one matching segment"
        cand = candidates[0]
        comp = cand["component"]
        seg = list(comp["coords"])
        anchor = cand["anchor"]
        front = max(_proj(p, direction) for p in seg)
        shift = int(_proj(anchor, direction) - front)
        assert shift > 0, "anchor must lie in front of the moving segment"

        final = _translated(seg, direction, shift)
        assert all(_in_bounds(grid.shape, r, c) for r, c in final)

        used_segments.add(cand["key"])
        moves.append(
            {
                "arrow_pixels": set(arrow["pixels"]),
                "segment": seg,
                "color": int(comp["color"]),
                "direction": direction,
                "shift": shift,
                "anchor": anchor,
                "final": final,
            }
        )

    return moves


def _apply_moves(grid: np.ndarray, moves: list[dict]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)

    for move in moves:
        for r, c in move["arrow_pixels"]:
            out[int(r), int(c)] = BLACK
        for r, c in move["segment"]:
            out[int(r), int(c)] = BLACK

    for move in moves:
        direction = move["direction"]
        shift = int(move["shift"])
        for step in range(2, shift, 2):
            for r, c in _translated(move["segment"], direction, step):
                out[int(r), int(c)] = GREY

    for move in moves:
        for r, c in move["final"]:
            out[int(r), int(c)] = int(move["color"])

    for move in moves:
        r, c = move["anchor"]
        out[int(r), int(c)] = int(move["color"])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each colored tail segment forward along its maroon arrow until it reaches the same-color anchor, leaving grey motion shadows.

    Input structure:
    - The background is black.
    - Maroon pixels form one or more five-pixel arrows; each arrow has a three-pixel cardinal shaft and a V-shaped diagonal head.
    - The tail of each arrow touches one straight colored segment of length two to five, perpendicular to the arrow direction.
    - For every segment color there is exactly one same-color single-pixel anchor on the forward extension of that arrow's shaft.

    Rule:
    - Decode each maroon arrow to get its push direction and shaft line.
    - Match the tail-touching perpendicular segment to the same-color anchor found ahead on the shaft extension.
    - Clear the original arrow and original segment, then translate the whole segment forward until its leading row or column aligns with the anchor.
    - Between the start and final positions, draw same-shaped grey shadows at every second translation step; keep the anchor color visible in the output.

    Key insight:
    - The arrow supplies both the direction of motion and the sightline for finding the matching anchor; the grey copies are evenly spaced records of the segment's path.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    moves = _solve_moves(g)
    return _apply_moves(g, moves)


# --------------- generator ---------------

COLOR_POOL = (1, 2, 3, 4, 6, 7, 8)
GRID_SIZE_RANGE = (12, 30)
LINE_LENGTH_RANGE = (2, 5)
ANCHOR_GAP_RANGE = (1, 8)
GENERATION_RETRIES = 600
PLACEMENT_RETRIES = 300


def _scene(
    tail: tuple[int, int],
    direction: tuple[int, int],
    length: int,
    axis_index: int,
    anchor_gap: int,
    color: int,
) -> dict:
    side = _perp(direction)
    base = _add(tail, direction, -1)
    offsets = range(-int(axis_index), int(length) - int(axis_index))
    segment = [_add(base, side, offset) for offset in offsets]
    arrow = _arrow_pixels(tail, direction)
    head = _add(tail, direction, 2)
    anchor = _add(head, direction, int(anchor_gap))
    shift = int(_proj(anchor, direction) - max(_proj(p, direction) for p in segment))
    final = _translated(segment, direction, shift)
    shadows: list[tuple[int, int]] = []
    for step in range(2, shift, 2):
        shadows.extend(_translated(segment, direction, step))

    return {
        "arrow_pixels": arrow,
        "segment": segment,
        "anchor": anchor,
        "direction": direction,
        "shift": shift,
        "final": final,
        "shadows": shadows,
        "color": int(color),
    }


def _all_scene_cells(scene: dict) -> set[tuple[int, int]]:
    cells = set(scene["arrow_pixels"])
    cells.update(scene["segment"])
    cells.add(scene["anchor"])
    cells.update(scene["final"])
    cells.update(scene["shadows"])
    return cells


def _mark_reserved(reserved: np.ndarray, cells: set[tuple[int, int]], radius: int = 1) -> None:
    h, w = reserved.shape
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr = int(r) + dr
                cc = int(c) + dc
                if 0 <= rr < h and 0 <= cc < w:
                    reserved[rr, cc] = True


def _fits_scene(scene: dict, reserved: np.ndarray) -> bool:
    cells = _all_scene_cells(scene)
    if any(not _in_bounds(reserved.shape, r, c) for r, c in cells):
        return False
    return not any(bool(reserved[int(r), int(c)]) for r, c in cells)


def _paint_input(grid: np.ndarray, scene: dict) -> None:
    for r, c in scene["arrow_pixels"]:
        assert int(grid[int(r), int(c)]) == BLACK
        grid[int(r), int(c)] = MAROON
    for r, c in scene["segment"]:
        assert int(grid[int(r), int(c)]) == BLACK
        grid[int(r), int(c)] = int(scene["color"])
    ar, ac = scene["anchor"]
    assert int(grid[int(ar), int(ac)]) == BLACK
    grid[int(ar), int(ac)] = int(scene["color"])


def _expected_output(inp: np.ndarray, scenes: list[dict]) -> np.ndarray:
    moves: list[dict] = []
    for scene in scenes:
        moves.append(
            {
                "arrow_pixels": set(scene["arrow_pixels"]),
                "segment": list(scene["segment"]),
                "color": int(scene["color"]),
                "direction": scene["direction"],
                "shift": int(scene["shift"]),
                "anchor": scene["anchor"],
                "final": list(scene["final"]),
            }
        )
    return _apply_moves(inp, moves)


def _object_count_for_size(h: int, w: int) -> int:
    m = min(int(h), int(w))
    if m < 15:
        max_count = 1
    elif m < 19:
        max_count = 2
    elif m < 24:
        max_count = 3
    else:
        max_count = 4
    return random.randint(1, max_count)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIZE_RANGE)
    w = random.randint(*GRID_SIZE_RANGE)
    count = _object_count_for_size(h, w)
    colors = random.sample(COLOR_POOL, count)
    directions = random.sample(DIRS, count)

    grid = np.zeros((h, w), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)
    scenes: list[dict] = []

    for color, direction in zip(colors, directions):
        placed = None
        for _ in range(PLACEMENT_RETRIES):
            length = random.randint(*LINE_LENGTH_RANGE)
            axis_index = random.randint(0, int(length) - 1)
            anchor_gap = random.randint(*ANCHOR_GAP_RANGE)
            tail = (random.randrange(h), random.randrange(w))
            scene = _scene(tail, direction, length, axis_index, anchor_gap, int(color))
            if not _fits_scene(scene, reserved):
                continue
            placed = scene
            break

        if placed is None:
            return None
        _paint_input(grid, placed)
        _mark_reserved(reserved, _all_scene_cells(placed), radius=1)
        scenes.append(placed)

    expected = _expected_output(grid, scenes)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-push input")
