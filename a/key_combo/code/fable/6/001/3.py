from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
TRACK = 8
GATE = 5
PALETTE_COLORS = (1, 2, 3, 4, 6, 7, 9)

Coord = tuple[int, int]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def bbox_coords(coords: list[Coord] | np.ndarray) -> tuple[int, int, int, int]:
    arr = np.asarray(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _foreground_components(grid: np.ndarray, bg: int) -> list[list[Coord]]:
    h, w = grid.shape
    seen: set[Coord] = set()
    out: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) == bg or (r, c) in seen:
                continue
            comp: list[Coord] = []
            q = deque([(r, c)])
            seen.add((r, c))
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if not in_bounds((h, w), nr, nc):
                        continue
                    if (nr, nc) in seen or int(grid[nr, nc]) == bg:
                        continue
                    seen.add((nr, nc))
                    q.append((nr, nc))
            out.append(comp)
    return out


def _ordered_simple_path(coords: list[Coord]) -> list[Coord]:
    coord_set = set(coords)
    degrees = {
        p: sum((nr, nc) in coord_set for nr, nc in neighbors4(*p))
        for p in coord_set
    }
    if len(coord_set) < 2:
        raise AssertionError("path must contain at least two cells")
    if any(d not in (1, 2) for d in degrees.values()):
        raise AssertionError("component must be a non-branching 4-connected path")
    endpoints = [p for p, d in degrees.items() if d == 1]
    if len(endpoints) != 2:
        raise AssertionError("path must have exactly two endpoints")

    ordered = [endpoints[0]]
    prev: Coord | None = None
    cur = endpoints[0]
    while True:
        nxts = [
            (nr, nc)
            for nr, nc in neighbors4(*cur)
            if (nr, nc) in coord_set and (nr, nc) != prev
        ]
        if not nxts:
            break
        if len(nxts) != 1:
            raise AssertionError("path ordering must be unique")
        prev, cur = cur, nxts[0]
        ordered.append(cur)

    if len(ordered) != len(coord_set):
        raise AssertionError("path traversal did not cover the full component")
    return ordered


def _path_turns(path: list[Coord]) -> int:
    turns = 0
    for i in range(2, len(path)):
        dr1 = path[i - 1][0] - path[i - 2][0]
        dc1 = path[i - 1][1] - path[i - 2][1]
        dr2 = path[i][0] - path[i - 1][0]
        dc2 = path[i][1] - path[i - 1][1]
        if (dr1, dc1) != (dr2, dc2):
            turns += 1
    return turns


def _expand_chebyshev(coords: set[Coord], radius: int = 1) -> set[Coord]:
    out = set(coords)
    for r, c in list(coords):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                out.add((r + dr, c + dc))
    return out


def _translate(coords: list[Coord], dr: int, dc: int) -> list[Coord]:
    return [(r + dr, c + dc) for r, c in coords]


def _min_manhattan(a: list[Coord], b: Coord) -> int:
    return min(abs(r - b[0]) + abs(c - b[1]) for r, c in a)


def _analyze_track(grid: np.ndarray, coords: list[Coord]) -> dict[str, object]:
    path = _ordered_simple_path(coords)
    colors = [int(grid[r, c]) for r, c in path]

    endpoint_colors = (colors[0], colors[-1])
    if endpoint_colors[0] == endpoint_colors[1]:
        raise AssertionError("track endpoints must have distinct colors")
    if endpoint_colors[0] in (BG, TRACK, GATE) or endpoint_colors[1] in (BG, TRACK, GATE):
        raise AssertionError("track endpoints must use distinct non-special colors")

    gate_indices = [i for i, color in enumerate(colors) if color == GATE]
    special_indices = [0, *gate_indices, len(path) - 1]
    if len(gate_indices) < 2:
        raise AssertionError("track must contain enough gates to create at least three segments")
    if any(colors[i] != GATE for i in gate_indices):
        raise AssertionError("all internal separators must be gates")
    if any(color not in (TRACK, GATE) for color in colors[1:-1]):
        raise AssertionError("track interior must contain only cyan path cells and gray gates")

    segments: list[list[Coord]] = []
    for left, right in zip(special_indices, special_indices[1:]):
        if right - left < 2:
            raise AssertionError("every segment must contain at least one cyan cell")
        segment = path[left + 1 : right]
        if any(int(grid[r, c]) != TRACK for r, c in segment):
            raise AssertionError("segment cells must be cyan before recoloring")
        segments.append(segment)

    return {
        "path": path,
        "colors": colors,
        "segments": segments,
        "endpoint_colors": endpoint_colors,
        "endpoints": (path[0], path[-1]),
        "turns": _path_turns(path),
    }


def _analyze_scene(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")
    bg = get_bg(g)
    if bg != BG:
        raise AssertionError("background must be black")

    components = _foreground_components(g, bg)
    if len(components) != 2:
        raise AssertionError("expected exactly one track component and one palette component")

    track_coords: list[Coord] | None = None
    palette_coords: list[Coord] | None = None
    for comp in components:
        comp_colors = {int(g[r, c]) for r, c in comp}
        has_track_signature = TRACK in comp_colors and GATE in comp_colors
        if has_track_signature:
            if track_coords is not None:
                raise AssertionError("expected a unique track component")
            track_coords = comp
        else:
            if palette_coords is not None:
                raise AssertionError("expected a unique palette component")
            palette_coords = comp

    if track_coords is None or palette_coords is None:
        raise AssertionError("could not separate track and palette")

    track = _analyze_track(g, track_coords)
    palette_path = _ordered_simple_path(palette_coords)
    palette_colors = [int(g[r, c]) for r, c in palette_path]
    if len(palette_path) != len(track["segments"]):
        raise AssertionError("palette length must equal the number of track segments")

    ep0, ep1 = track["endpoint_colors"]
    matches: list[tuple[str, str]] = []
    if palette_colors[0] == ep0:
        matches.append(("track_forward", "palette_forward"))
    if palette_colors[0] == ep1:
        matches.append(("track_reverse", "palette_forward"))
    if palette_colors[-1] == ep0:
        matches.append(("track_forward", "palette_reverse"))
    if palette_colors[-1] == ep1:
        matches.append(("track_reverse", "palette_reverse"))
    if len(matches) != 1:
        raise AssertionError("palette endpoint match must determine a unique reading direction")

    track_dir, palette_dir = matches[0]
    ordered_segments = list(track["segments"])
    if track_dir == "track_reverse":
        ordered_segments = ordered_segments[::-1]
    palette_read = list(palette_colors)
    if palette_dir == "palette_reverse":
        palette_read = palette_read[::-1]

    matched_color = int(palette_read[0])
    other_endpoint_color = ep1 if matched_color == ep0 else ep0
    if other_endpoint_color not in palette_read[1:-1]:
        raise AssertionError("the non-matching endpoint color must appear inside the palette as a decoy")
    if palette_read[-1] in (ep0, ep1):
        raise AssertionError("only one palette end may match an endpoint color")

    return {
        "grid": g,
        "track": track,
        "palette_path": palette_path,
        "palette_colors": palette_colors,
        "palette_read": palette_read,
        "ordered_segments": ordered_segments,
        "matched_endpoint_color": matched_color,
        "other_endpoint_color": other_endpoint_color,
        "track_turns": track["turns"],
        "palette_turns": _path_turns(palette_path),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each cyan track segment in traversal order using the palette strip read from its uniquely matching end.

    Input structure:
    - The background is black.
    - One 4-connected non-branching cyan track forms a single path whose two endpoint pixels have distinct colors.
    - Gray gate pixels lie on the track and split the cyan cells into ordered segments.
    - A detached one-cell-wide palette strip is another simple path whose length equals the number of cyan segments, and exactly one of its two end cells matches one track endpoint color.

    Rule:
    - Start from the track endpoint whose color matches the matching palette end cell.
    - Read the palette from that matching end, and recolor the k-th cyan segment encountered along the track to the k-th palette color.
    - Keep the track endpoints, gray gates, and the palette strip itself unchanged.

    Key insight:
    - The same unique endpoint-color match fixes both the direction along the track and the read direction of the palette strip, while the other endpoint color appears only as an interior decoy.
    """
    scene = _analyze_scene(np.asarray(grid, dtype=int))
    out = scene["grid"].copy()
    for color, segment in zip(scene["palette_read"], scene["ordered_segments"]):
        for r, c in segment:
            out[r, c] = int(color)
    return out


def _neighbor_count(coord: Coord, occupied: set[Coord]) -> int:
    return sum(nb in occupied for nb in neighbors4(*coord))


def _can_extend_from(
    cell: Coord,
    shape: tuple[int, int],
    occupied: set[Coord],
    blocked: set[Coord],
    margin: int,
) -> bool:
    h, w = shape
    for nr, nc in neighbors4(*cell):
        if not (margin <= nr < h - margin and margin <= nc < w - margin):
            continue
        if (nr, nc) in occupied or (nr, nc) in blocked:
            continue
        if _neighbor_count((nr, nc), occupied) == 1:
            return True
    return False


def _gen_induced_path(
    shape: tuple[int, int],
    length: int,
    *,
    blocked: set[Coord] | None = None,
    margin: int = 1,
    min_turns: int = 0,
    attempts: int = 500,
) -> list[Coord] | None:
    h, w = shape
    blocked = set() if blocked is None else set(blocked)
    starts = [
        (r, c)
        for r in range(margin, h - margin)
        for c in range(margin, w - margin)
        if (r, c) not in blocked
    ]
    if not starts:
        return None

    for _ in range(attempts):
        path = [random.choice(starts)]
        occupied = {path[0]}
        last_dir: Coord | None = None

        while len(path) < length:
            cur = path[-1]
            remaining_after_pick = length - len(path) - 1
            candidates: list[Coord] = []
            weights: list[int] = []
            for nr, nc in neighbors4(*cur):
                if not (margin <= nr < h - margin and margin <= nc < w - margin):
                    continue
                if (nr, nc) in occupied or (nr, nc) in blocked:
                    continue
                if _neighbor_count((nr, nc), occupied) != 1:
                    continue

                new_occupied = occupied | {(nr, nc)}
                if remaining_after_pick > 0 and not _can_extend_from(
                    (nr, nc), shape, new_occupied, blocked, margin
                ):
                    continue

                candidates.append((nr, nc))
                new_dir = (nr - cur[0], nc - cur[1])
                if last_dir is None:
                    weights.append(1)
                elif new_dir != last_dir and min_turns > 0:
                    weights.append(4)
                elif new_dir == last_dir and min_turns > 0:
                    weights.append(1)
                else:
                    weights.append(1)

            if not candidates:
                break

            nxt = random.choices(candidates, weights=weights, k=1)[0]
            last_dir = (nxt[0] - cur[0], nxt[1] - cur[1])
            path.append(nxt)
            occupied.add(nxt)

        if len(path) != length:
            continue
        if _path_turns(path) < min_turns:
            continue
        return path

    return None


def _sample_segment_lengths(segment_count: int) -> list[int]:
    lengths = [random.randint(1, 3) for _ in range(segment_count)]
    if segment_count <= 4 and random.random() < 0.45:
        idx = random.randrange(segment_count)
        lengths[idx] += 1
    return lengths


def _sample_palette_read(segment_count: int, matched_color: int, other_color: int) -> list[int]:
    colors = [matched_color] + [0] * (segment_count - 1)
    decoy_idx = random.randint(1, segment_count - 2)
    colors[decoy_idx] = other_color
    free_slots = [i for i in range(1, segment_count) if i != decoy_idx]
    pool = [c for c in PALETTE_COLORS if c not in (matched_color, other_color)]
    picks = random.sample(pool, len(free_slots))
    for idx, color in zip(free_slots, picks):
        colors[idx] = color
    if colors[-1] in (matched_color, other_color):
        raise AssertionError("the far palette end must not match an endpoint color")
    return colors


def _track_colors_along_path(path_len: int, seg_lengths: list[int], endpoint_colors: tuple[int, int]) -> list[int]:
    colors = [TRACK] * path_len
    colors[0], colors[-1] = endpoint_colors
    cursor = 1
    for seg_len in seg_lengths[:-1]:
        cursor += seg_len
        colors[cursor] = GATE
        cursor += 1
    if cursor + seg_lengths[-1] != path_len - 1:
        raise AssertionError("segment layout must exactly fill the path")
    return colors


def _distance_mode_ok(
    palette_path: list[Coord],
    matched_endpoint: Coord,
    other_endpoint: Coord,
    mode: str,
) -> bool:
    d_match = _min_manhattan(palette_path, matched_endpoint)
    d_other = _min_manhattan(palette_path, other_endpoint)
    if d_match == d_other:
        return False
    if mode == "matched_closer":
        return d_match < d_other
    if mode == "other_closer":
        return d_match > d_other
    raise ValueError(f"unknown distance mode: {mode}")


def _render_scene(
    shape: tuple[int, int],
    track_path: list[Coord],
    track_colors: list[int],
    palette_path: list[Coord],
    palette_colors: list[int],
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for (r, c), color in zip(track_path, track_colors):
        grid[r, c] = int(color)
    for (r, c), color in zip(palette_path, palette_colors):
        grid[r, c] = int(color)
    return grid


def _validate_generated_pair(grid: np.ndarray, out: np.ndarray) -> None:
    scene = _analyze_scene(grid)
    track = scene["track"]
    palette_path = scene["palette_path"]
    ordered_segments = scene["ordered_segments"]
    palette_read = scene["palette_read"]

    if scene["track_turns"] == 0 and scene["palette_turns"] == 0:
        raise AssertionError("at least one of the two paths must bend")

    changed = 0
    for segment, color in zip(ordered_segments, palette_read):
        for r, c in segment:
            if int(out[r, c]) != int(color):
                raise AssertionError("each track segment must be repainted to the matching palette color")
            if int(grid[r, c]) != int(out[r, c]):
                changed += 1

    if changed == 0:
        raise AssertionError("the output must visibly differ from the input")

    for r, c in palette_path:
        if int(out[r, c]) != int(grid[r, c]):
            raise AssertionError("the palette strip must remain unchanged")

    for r, c in track["endpoints"]:
        if int(out[r, c]) != int(grid[r, c]):
            raise AssertionError("track endpoints must remain unchanged")

    for (r, c), color in zip(track["path"], track["colors"]):
        if color == GATE and int(out[r, c]) != GATE:
            raise AssertionError("gates must remain gray")


def _generate_candidate() -> np.ndarray | None:
    h = random.randint(15, 26)
    w = random.randint(15, 26)
    shape = (h, w)

    segment_count = random.randint(3, 6)
    seg_lengths = _sample_segment_lengths(segment_count)
    track_len = sum(seg_lengths) + (segment_count - 1) + 2

    track_turn_goal = min(max(1, random.choice([1, 2, 2, 3])), track_len - 2)
    palette_turn_goal = min(random.choice([0, 1, 1, 2]), segment_count - 2)

    if track_turn_goal == 0 and palette_turn_goal == 0:
        track_turn_goal = 1

    track_path = _gen_induced_path(shape, track_len, margin=1, min_turns=track_turn_goal, attempts=600)
    if track_path is None:
        return None

    endpoint_colors = random.sample(list(PALETTE_COLORS), 2)
    match_track_start = random.choice([True, False])
    matched_color = endpoint_colors[0] if match_track_start else endpoint_colors[1]
    other_color = endpoint_colors[1] if match_track_start else endpoint_colors[0]
    palette_read = _sample_palette_read(segment_count, matched_color, other_color)

    match_palette_start = random.choice([True, False])
    palette_colors = palette_read if match_palette_start else palette_read[::-1]
    track_colors = _track_colors_along_path(track_len, seg_lengths, tuple(endpoint_colors))

    track_endpoint_a = track_path[0]
    track_endpoint_b = track_path[-1]
    matched_endpoint = track_endpoint_a if match_track_start else track_endpoint_b
    other_endpoint = track_endpoint_b if match_track_start else track_endpoint_a
    distance_mode = random.choice(["matched_closer", "other_closer"])

    blocked = {
        (r, c)
        for r, c in _expand_chebyshev(set(track_path), radius=1)
        if in_bounds(shape, r, c)
    }

    palette_path: list[Coord] | None = None
    for _ in range(120):
        candidate = _gen_induced_path(
            shape,
            segment_count,
            blocked=blocked,
            margin=1,
            min_turns=palette_turn_goal,
            attempts=40,
        )
        if candidate is None:
            continue
        if not _distance_mode_ok(candidate, matched_endpoint, other_endpoint, distance_mode):
            continue
        palette_path = candidate
        break
    if palette_path is None:
        return None

    grid = _render_scene(shape, track_path, track_colors, palette_path, palette_colors)

    try:
        out = main(grid)
        _validate_generated_pair(grid, out)
    except Exception:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid palette-track ARC input")
