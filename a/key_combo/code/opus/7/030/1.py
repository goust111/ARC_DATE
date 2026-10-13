import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

DIRS = {
    "U": (-1, 0),
    "R": (0, 1),
    "D": (1, 0),
    "L": (0, -1),
}

BASE_TEMPLATES = {
    "hook3": [(0, 0), (1, 0), (1, 1)],
    "tee4": [(0, 0), (1, -1), (1, 0), (1, 1)],
    "arrow5": [(0, 0), (1, -1), (1, 0), (1, 1), (2, 0)],
}

GEN_ATTEMPTS = 500


def _rot90_cw(offsets: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(c, -r) for r, c in offsets]


def _build_templates() -> dict[tuple[str, str], list[tuple[int, int]]]:
    out: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for name, base in BASE_TEMPLATES.items():
        cur = list(base)
        for direction in ("U", "R", "D", "L"):
            out[(name, direction)] = sorted(cur)
            cur = _rot90_cw(cur)
    return out


SPRITE_TEMPLATES = _build_templates()

GLYPH_PATTERNS = {
    "D": [np.array([[1], [1]], dtype=int)],
    "R": [np.array([[0, 1], [1, 1]], dtype=int)],
    "L": [np.array([[1, 0], [1, 1]], dtype=int)],
    "U": [
        np.array([[1, 1], [0, 1]], dtype=int),
        np.array([[1, 1], [1, 0]], dtype=int),
    ],
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _decode_glyphs(grid: np.ndarray) -> list[str]:
    top = np.asarray(grid[:2] == RED, dtype=bool)
    comps = _components(top)
    comps.sort(key=lambda comp: min(c for _, c in comp))
    directions: list[str] = []
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        if (r0, r1) != (0, 1):
            continue
        patch = top[r0 : r1 + 1, c0 : c1 + 1].astype(int)
        if patch.shape == (2, 1) and int(patch.sum()) == 2:
            directions.append("D")
            continue
        if patch.shape == (2, 2) and int(patch.sum()) == 3:
            missing = tuple(map(int, np.argwhere(patch == 0)[0]))
            if missing == (0, 0):
                directions.append("R")
            elif missing == (0, 1):
                directions.append("L")
            else:
                directions.append("U")
    return directions


def _find_sprite_component(grid: np.ndarray) -> set[tuple[int, int]]:
    comps = _components(grid == BLUE)
    if not comps:
        return set()
    best = max(comps, key=len)
    return set(best)


def _match_template(component: set[tuple[int, int]]) -> tuple[tuple[int, int], list[tuple[int, int]]] | None:
    if not component:
        return None
    for head in component:
        hr, hc = head
        for offsets in SPRITE_TEMPLATES.values():
            placed = {(hr + dr, hc + dc) for dr, dc in offsets}
            if placed == component:
                return (hr, hc), list(offsets)
    return None


def _infer_tip_fallback(component: set[tuple[int, int]]) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    def score(direction: str) -> tuple[int, int, int, tuple[int, int]] | None:
        if direction == "U":
            edge = min(r for r, _ in component)
            extreme = [(r, c) for r, c in component if r == edge]
        elif direction == "D":
            edge = max(r for r, _ in component)
            extreme = [(r, c) for r, c in component if r == edge]
        elif direction == "L":
            edge = min(c for _, c in component)
            extreme = [(r, c) for r, c in component if c == edge]
        else:
            edge = max(c for _, c in component)
            extreme = [(r, c) for r, c in component if c == edge]
        if len(extreme) != 1:
            return None
        head = extreme[0]
        head_neighbors = sum((nr, nc) in component for nr, nc in _neighbors4(*head))
        if head_neighbors != 1:
            return None
        if direction in ("U", "D"):
            span = len({c for _, c in component})
        else:
            span = len({r for r, _ in component})
        depth = len(component)
        return span, depth, -head_neighbors, head

    scored = []
    for direction in DIRS:
        cand = score(direction)
        if cand is not None:
            scored.append(cand)
    if not scored:
        head = min(component)
    else:
        head = max(scored)[-1]
    hr, hc = head
    offsets = sorted((r - hr, c - hc) for r, c in component)
    return head, offsets


def _find_sprite(grid: np.ndarray) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    component = _find_sprite_component(grid)
    matched = _match_template(component)
    if matched is not None:
        return matched
    return _infer_tip_fallback(component)


def _halfline_distance(src: tuple[int, int], dst: tuple[int, int], direction: str) -> int | None:
    sr, sc = src
    dr = dst[0] - sr
    dc = dst[1] - sc
    if direction == "U" and dc == 0 and dr < 0:
        return -dr
    if direction == "D" and dc == 0 and dr > 0:
        return dr
    if direction == "L" and dr == 0 and dc < 0:
        return -dc
    if direction == "R" and dr == 0 and dc > 0:
        return dc
    return None


def _nearest_marker(head: tuple[int, int], markers: set[tuple[int, int]], direction: str) -> tuple[int, int] | None:
    candidates: list[tuple[int, tuple[int, int]]] = []
    for marker in markers:
        dist = _halfline_distance(head, marker, direction)
        if dist is not None:
            candidates.append((dist, marker))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def _stamp_sprite(grid: np.ndarray, head: tuple[int, int], offsets: list[tuple[int, int]], color: int) -> None:
    hr, hc = head
    for dr, dc in offsets:
        rr = hr + dr
        cc = hc + dc
        if _in_bounds(grid.shape, rr, cc):
            grid[rr, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the top glyph strip into navigation commands and move the blue sprite through the green markers.

    Input structure:
    - A black background contains a two-row red glyph strip at the top, followed by one black separator row.
    - Below the strip there is one 4-connected blue arrow sprite and several isolated green single-pixel markers.
    - Each red glyph spans two rows and is either a 1-column vertical bar or a 2-column elbow with one missing corner.

    Rule:
    - Read the red glyphs from left to right: a full vertical bar means move downward, a top-left gap means move right, a top-right gap means move left, and either bottom-corner gap means move up.
    - Starting from the sprite tip pixel, follow the decoded directions one by one; for each command, move the whole sprite by translation so that the tip lands on the nearest remaining green marker on that half-line, then remove that marker.
    - Return a black grid of the same size containing only the sprite at its final position plus any markers that were never consumed.

    Key insight:
    - The red strip is a program, not a picture: it decodes into an ordered direction sequence that controls the sprite independently of the sprite's own facing.
    """
    g = np.asarray(grid, dtype=int)
    directions = _decode_glyphs(g)
    head, offsets = _find_sprite(g)
    markers = {tuple(map(int, rc)) for rc in np.argwhere(g == GREEN)}

    for direction in directions:
        target = _nearest_marker(head, markers, direction)
        if target is None:
            continue
        head = target
        markers.remove(target)

    out = np.full_like(g, BLACK)
    _stamp_sprite(out, head, offsets, BLUE)
    for r, c in markers:
        out[r, c] = GREEN
    return out


def _sample_directions(rng: random.Random, n_steps: int) -> list[str] | None:
    need_up = rng.random() < 0.65
    for _ in range(200):
        dirs = [rng.choice(["U", "R", "D", "L"]) for _ in range(n_steps)]
        if "D" not in dirs:
            continue
        if not ({"L", "R"} & set(dirs)):
            continue
        if len(set(dirs)) < 3:
            continue
        if need_up and "U" not in dirs:
            continue
        return dirs
    return None


def _positions_from_steps(directions: list[str], distances: list[int]) -> list[tuple[int, int]]:
    pos = [(0, 0)]
    for direction, distance in zip(directions, distances):
        dr, dc = DIRS[direction]
        last = pos[-1]
        pos.append((last[0] + dr * distance, last[1] + dc * distance))
    return pos


def _body_cells(head: tuple[int, int], offsets: list[tuple[int, int]]) -> set[tuple[int, int]]:
    hr, hc = head
    return {(hr + dr, hc + dc) for dr, dc in offsets}


def _path_is_valid(positions: list[tuple[int, int]], directions: list[str], offsets: list[tuple[int, int]]) -> bool:
    if len(set(positions)) != len(positions):
        return False
    for i, direction in enumerate(directions):
        head = positions[i]
        remaining = positions[i + 1 :]
        candidates = [p for p in remaining if _halfline_distance(head, p, direction) is not None]
        if candidates != [positions[i + 1]]:
            return False
    for i, head in enumerate(positions[:-1]):
        body = _body_cells(head, offsets)
        future_markers = set(positions[i + 1 :])
        if body & future_markers:
            return False
    return True


def _glyph_width(directions: list[str]) -> int:
    widths = [1 if d == "D" else 2 for d in directions]
    return sum(widths) + max(0, len(widths) - 1)


def _encode_glyph_strip(directions: list[str], rng: random.Random) -> np.ndarray:
    strip = np.zeros((2, _glyph_width(directions)), dtype=int)
    c = 0
    for i, direction in enumerate(directions):
        pattern = rng.choice(GLYPH_PATTERNS[direction])
        patch = np.asarray(pattern, dtype=int)
        strip[:, c : c + patch.shape[1]] = patch
        c += patch.shape[1]
        if i + 1 < len(directions):
            c += 1
    return np.where(strip == 1, RED, BLACK)


def _sample_template(rng: random.Random) -> list[tuple[int, int]]:
    weighted = [
        ("hook3", 2),
        ("tee4", 4),
        ("arrow5", 4),
    ]
    bag = [name for name, weight in weighted for _ in range(weight)]
    name = rng.choice(bag)
    direction = rng.choice(["U", "R", "D", "L"])
    return list(SPRITE_TEMPLATES[(name, direction)])


def _region_bbox(positions: list[tuple[int, int]], offsets: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    cells = set(positions)
    for head in positions:
        cells |= _body_cells(head, offsets)
    return _bbox(cells)


def _compose_input(
    positions: list[tuple[int, int]],
    offsets: list[tuple[int, int]],
    directions: list[str],
    rng: random.Random,
) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    strip = _encode_glyph_strip(directions, rng)
    glyph_w = strip.shape[1]
    r0, c0, r1, c1 = _region_bbox(positions, offsets)
    region_h = r1 - r0 + 1
    region_w = c1 - c0 + 1

    min_h = 3 + region_h + 2
    max_h = min(30, min_h + 5)
    min_w = max(glyph_w + 2, region_w + 2)
    max_w = min(30, min_w + 6)
    if min_h > max_h or min_w > max_w:
        return None

    h = rng.randint(min_h, max_h)
    w = rng.randint(min_w, max_w)
    grid = np.zeros((h, w), dtype=int)

    strip_left = rng.randint(0, w - glyph_w)
    grid[:2, strip_left : strip_left + glyph_w] = strip

    top_min = 4
    top_max = h - region_h - 1
    left_min = 1
    left_max = w - region_w - 1
    if top_min > top_max or left_min > left_max:
        return None
    obj_top = rng.randint(top_min, top_max)
    obj_left = rng.randint(left_min, left_max)
    shift_r = obj_top - r0
    shift_c = obj_left - c0

    shifted_positions = [(r + shift_r, c + shift_c) for r, c in positions]
    shifted_head = shifted_positions[0]
    shifted_markers = shifted_positions[1:]

    for marker in shifted_markers:
        if grid[marker] != BLACK:
            return None
        grid[marker] = GREEN

    sprite_cells = _body_cells(shifted_head, offsets)
    for rr, cc in sprite_cells:
        if not _in_bounds(grid.shape, rr, cc) or grid[rr, cc] != BLACK:
            return None
        grid[rr, cc] = BLUE
    return grid, shifted_positions


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_steps = rng.randint(4, 8)
    directions = _sample_directions(rng, n_steps)
    if directions is None:
        return None
    offsets = _sample_template(rng)
    for _ in range(300):
        distances = [rng.randint(2, 5) for _ in range(n_steps)]
        positions = _positions_from_steps(directions, distances)
        if not _path_is_valid(positions, directions, offsets):
            continue
        r0, c0, r1, c1 = _region_bbox(positions, offsets)
        if r1 - r0 + 1 > 22 or c1 - c0 + 1 > 26:
            continue
        composed = _compose_input(positions, offsets, directions, rng)
        if composed is None:
            continue
        grid, shifted_positions = composed
        expected = np.full_like(grid, BLACK)
        _stamp_sprite(expected, shifted_positions[-1], offsets, BLUE)
        if np.array_equal(main(grid), expected):
            return grid
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
