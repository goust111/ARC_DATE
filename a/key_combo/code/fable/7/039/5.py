from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]


def _background(g: np.ndarray) -> int:
    """Return the unique most frequent color."""
    counts = Counter(map(int, g.ravel()))
    ranked = counts.most_common()
    assert ranked and (len(ranked) == 1 or ranked[0][1] > ranked[1][1])
    return int(ranked[0][0])


def _border_cells(box: tuple[int, int, int, int]) -> set[Coord]:
    r0, c0, r1, c1 = box
    return {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }


def _gap_direction(box: tuple[int, int, int, int], gap: Coord) -> Coord:
    r0, c0, r1, c1 = box
    r, c = gap
    if r == r0 and c0 < c < c1:
        return -1, 0
    if r == r1 and c0 < c < c1:
        return 1, 0
    if c == c0 and r0 < r < r1:
        return 0, -1
    if c == c1 and r0 < r < r1:
        return 0, 1
    raise AssertionError("the gap must be a non-corner border cell")


def _find_frames(g: np.ndarray, bg: int) -> list[dict]:
    """Find all monochrome rectangular perimeters with one non-corner gap."""
    h, w = map(int, g.shape)
    found: list[dict] = []
    visited: set[Coord] = set()
    # A one-gap perimeter is still one 4-connected monochrome component.  Using
    # components avoids guessing a maximum frame size and is much faster than
    # enumerating every possible rectangle in a 30x30 canvas.
    for sr, sc in map(tuple, np.argwhere(g != bg)):
        start = (int(sr), int(sc))
        if start in visited:
            continue
        color = int(g[start])
        stack = [start]
        component: set[Coord] = set()
        visited.add(start)
        while stack:
            r, c = stack.pop()
            component.add((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                q = (rr, cc)
                if 0 <= rr < h and 0 <= cc < w and q not in visited and int(g[q]) == color:
                    visited.add(q)
                    stack.append(q)
        if len(component) < 10:
            continue
        rows = [p[0] for p in component]
        cols = [p[1] for p in component]
        box = (min(rows), min(cols), max(rows), max(cols))
        r0, c0, r1, c1 = box
        if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 4:
            continue
        border = _border_cells(box)
        gaps = border - component
        if len(gaps) != 1 or component != border - gaps:
            continue
        gap = next(iter(gaps))
        try:
            direction = _gap_direction(box, gap)
        except AssertionError:
            continue
        if np.any(g[r0 + 1 : r1, c0 + 1 : c1] != bg):
            continue
        found.append(
            {"box": box, "gap": gap, "direction": direction, "color": color, "walls": frozenset(component)}
        )
    assert found, "expected at least one one-gap rectangular frame"
    all_walls: set[Coord] = set()
    for frame in found:
        assert not (all_walls & set(frame["walls"])), "frames must be distinct"
        all_walls.update(frame["walls"])
    return found


def _keys(g: np.ndarray, bg: int, frames: list[dict]) -> list[dict]:
    wall_cells = set().union(*(set(frame["walls"]) for frame in frames))
    coords = [tuple(map(int, p)) for p in np.argwhere(g != bg)]
    key_cells = [p for p in coords if p not in wall_cells]
    assert len(key_cells) == len(frames), "expected one singleton key per frame"
    for r, c in key_cells:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            q = (r + dr, c + dc)
            assert q not in key_cells, "keys must be separate singleton pixels"
    return [{"pos": p, "color": int(g[p])} for p in key_cells]


def _path_to_gap(key: Coord, frame: dict) -> list[Coord] | None:
    """Return cells strictly between a key and its gap on the outward normal."""
    kr, kc = key
    gr, gc = frame["gap"]
    dr, dc = frame["direction"]
    if dr and (kc != gc or (kr - gr) * dr <= 0):
        return None
    if dc and (kr != gr or (kc - gc) * dc <= 0):
        return None
    distance = abs(kr - gr) + abs(kc - gc)
    return [(gr + dr * step, gc + dc * step) for step in range(1, distance)]


def _candidate_graph(g: np.ndarray, bg: int, keys: list[dict], frames: list[dict]) -> list[list[int]]:
    graph: list[list[int]] = []
    for key in keys:
        edges = []
        for j, frame in enumerate(frames):
            path = _path_to_gap(key["pos"], frame)
            if path is not None and all(int(g[p]) == bg for p in path):
                edges.append(j)
        assert edges, "every key must see at least one exact gap"
        graph.append(edges)
    return graph


def _unique_matching(graph: list[list[int]], n: int) -> tuple[int, ...]:
    """Enumerate up to two perfect matchings; require exactly one."""
    solutions: list[tuple[int, ...]] = []
    assignment = [-1] * n

    def search(remaining: set[int], used: set[int]) -> None:
        if len(solutions) >= 2:
            return
        if not remaining:
            solutions.append(tuple(assignment))
            return
        key_i = min(remaining, key=lambda i: len([j for j in graph[i] if j not in used]))
        choices = [j for j in graph[key_i] if j not in used]
        for frame_j in choices:
            assignment[key_i] = frame_j
            search(remaining - {key_i}, used | {frame_j})
            assignment[key_i] = -1

    search(set(range(n)), set())
    assert len(solutions) == 1, "key-to-gap assignment must be globally unique"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each one-gap rectangular lock from the uniquely matched sightline key.

    Input structure:
    - A uniquely dominant background contains several monochrome rectangular frames.
    - Every frame border has exactly one missing non-corner pixel, and its interior is empty.
    - The same number of isolated colored key pixels lie outside the frames on unobstructed
      outward-normal sightlines to frame gaps.

    Rule:
    - Find the unique global one-to-one assignment using only keys exactly aligned with a
      gap whose intervening cells are background. For each pair, preserve the key and frame
      wall, and paint the intervening channel, the gap, and the full interior in key color.

    Key insight:
    - Alignment is tested against the missing wall cell itself in the frame's coordinate
      system; the bijection chooses whole key-gap pairs before color is injected inward.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size and max(g.shape) <= 30, "expected an ARC-sized 2D grid"
    bg = _background(g)
    frames = _find_frames(g, bg)
    keys = _keys(g, bg, frames)
    graph = _candidate_graph(g, bg, keys, frames)
    matching = _unique_matching(graph, len(frames))

    out = g.copy()
    for key_i, frame_i in enumerate(matching):
        key = keys[key_i]
        frame = frames[frame_i]
        color = key["color"]
        path = _path_to_gap(key["pos"], frame)
        assert path is not None and all(int(g[p]) == bg for p in path)
        for p in path:
            out[p] = color
        out[frame["gap"]] = color
        r0, c0, r1, c1 = frame["box"]
        out[r0 + 1 : r1, c0 + 1 : c1] = color

    return out


# ------------------------------- generator -------------------------------

_SIDES = ("top", "bottom", "left", "right")
_DELTAS = {"top": (-1, 0), "bottom": (1, 0), "left": (0, -1), "right": (0, 1)}


def _gap_for(box: tuple[int, int, int, int], side: str, offset: int) -> Coord:
    r0, c0, r1, c1 = box
    if side == "top":
        return r0, c0 + offset
    if side == "bottom":
        return r1, c0 + offset
    if side == "left":
        return r0 + offset, c0
    return r0 + offset, c1


def _separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + 2 < br0 or br1 + 2 < ar0 or ac1 + 2 < bc0 or bc1 + 2 < ac0


def _render_expected(inp: np.ndarray, specs: list[dict]) -> np.ndarray:
    expected = inp.copy()
    for spec in specs:
        color = spec["key_color"]
        gr, gc = spec["gap"]
        kr, kc = spec["key"]
        dr, dc = spec["direction"]
        distance = abs(kr - gr) + abs(kc - gc)
        for step in range(distance):
            expected[gr + dr * step, gc + dc * step] = color
        r0, c0, r1, c1 = spec["box"]
        expected[r0 + 1 : r1, c0 + 1 : c1] = color
        expected[kr, kc] = color
    return expected


def _attempt() -> np.ndarray | None:
    h = random.randint(18, 30)
    w = random.randint(18, 30)
    count = random.randint(2, 4)
    bg, wall = random.sample(range(10), 2)
    key_palette = random.sample([c for c in range(10) if c not in {bg, wall}], count)
    grid = np.full((h, w), bg, dtype=int)
    specs: list[dict] = []

    # Place empty frames first, so every later ray can be checked against all walls.
    for i in range(count):
        placed = False
        for _ in range(180):
            fh, fw = random.randint(4, 7), random.randint(4, 7)
            side = random.choice(_SIDES)
            margin_top = 3 if side == "top" else 1
            margin_bottom = 3 if side == "bottom" else 1
            margin_left = 3 if side == "left" else 1
            margin_right = 3 if side == "right" else 1
            if h - fh - margin_top - margin_bottom < 0 or w - fw - margin_left - margin_right < 0:
                continue
            r0 = random.randint(margin_top, h - fh - margin_bottom)
            c0 = random.randint(margin_left, w - fw - margin_right)
            box = (r0, c0, r0 + fh - 1, c0 + fw - 1)
            if any(not _separated(box, old["box"]) for old in specs):
                continue
            edge_len = fw if side in ("top", "bottom") else fh
            gap = _gap_for(box, side, random.randint(1, edge_len - 2))
            specs.append(
                {"box": box, "side": side, "gap": gap, "direction": _DELTAS[side], "key_color": key_palette[i]}
            )
            placed = True
            break
        if not placed:
            return None

    for spec in specs:
        r0, c0, r1, c1 = spec["box"]
        grid[r0, c0 : c1 + 1] = wall
        grid[r1, c0 : c1 + 1] = wall
        grid[r0 : r1 + 1, c0] = wall
        grid[r0 : r1 + 1, c1] = wall
        grid[spec["gap"]] = bg

    # Put each key on a nontrivial clear outward ray. Reject accidental extra matches.
    random.shuffle(specs)
    for spec in specs:
        gr, gc = spec["gap"]
        dr, dc = spec["direction"]
        possible = []
        for distance in range(2, 8):
            kr, kc = gr + dr * distance, gc + dc * distance
            if not (0 <= kr < h and 0 <= kc < w):
                break
            cells = [(gr + dr * step, gc + dc * step) for step in range(1, distance + 1)]
            if all(int(grid[p]) == bg for p in cells):
                possible.append((kr, kc))
            else:
                break
        if not possible:
            return None
        key = random.choice(possible)
        spec["key"] = key
        grid[key] = spec["key_color"]

    expected = _render_expected(grid, specs)
    try:
        frames = _find_frames(grid, bg)
        keys = _keys(grid, bg, frames)
        graph = _candidate_graph(grid, bg, keys, frames)
        matching = _unique_matching(graph, count)
        actual = main(grid)
    except (AssertionError, IndexError, ValueError):
        return None
    if len(frames) != count or len(keys) != count or not np.array_equal(actual, expected):
        return None
    # Semantic checks: unique assignment, every intended pair, preserved walls/keys,
    # and both channel and interior are necessary visible changes.
    frame_by_gap = {frame["gap"]: i for i, frame in enumerate(frames)}
    intended = {spec["key"]: frame_by_gap.get(spec["gap"]) for spec in specs}
    for i, key in enumerate(keys):
        if matching[i] != intended.get(key["pos"]):
            return None
    wall_cells = set().union(*(set(frame["walls"]) for frame in frames))
    if any(actual[p] != grid[p] for p in wall_cells):
        return None
    if any(actual[key["pos"]] != key["color"] for key in keys):
        return None
    changed = actual != grid
    if not any(changed[frame["gap"]] for frame in frames):
        return None
    if not all(np.any(changed[f["box"][0] + 1 : f["box"][2], f["box"][1] + 1 : f["box"][3]]) for f in frames):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a varied, independently checked key-and-lock input."""
    for _ in range(400):
        result = _attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid key-and-lock grid")
