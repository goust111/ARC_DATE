from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

import numpy as np


DIRS: tuple[tuple[int, int], ...] = (
    (-1, 0),   # up
    (-1, 1),   # up-right
    (0, 1),    # right
    (1, 1),    # down-right
    (1, 0),    # down
    (1, -1),   # down-left
    (0, -1),   # left
    (-1, -1),  # up-left
)
DIR_TO_INDEX = {d: i for i, d in enumerate(DIRS)}
ALL_COLORS = list(range(10))


@dataclass(frozen=True)
class Obj:
    kind: str
    top: int
    left: int
    direction: int | None = None
    start: bool = False

    @property
    def center(self) -> tuple[int, int]:
        return self.top + 1, self.left + 1


def _bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _crop3(grid: np.ndarray, top: int, left: int) -> np.ndarray:
    return grid[top : top + 3, left : left + 3].copy()


def _clock_direction(window: np.ndarray, bg: int) -> int | None:
    if window.shape != (3, 3) or np.any(window == bg):
        return None

    needle = int(window[1, 1])
    if needle == bg:
        return None

    hits: list[int] = []
    for k, (dr, dc) in enumerate(DIRS):
        if int(window[1 + dr, 1 + dc]) == needle:
            hits.append(k)
    if len(hits) != 1:
        return None

    if int(np.count_nonzero(window == needle)) != 2:
        return None
    return int(hits[0])


def _checker_remainder(window: np.ndarray, direction: int) -> bool:
    needle = int(window[1, 1])
    colors_by_parity: dict[int, int] = {}
    hand_r, hand_c = 1 + DIRS[direction][0], 1 + DIRS[direction][1]

    for r in range(3):
        for c in range(3):
            if (r, c) == (1, 1) or (r, c) == (hand_r, hand_c):
                continue
            value = int(window[r, c])
            if value == needle:
                return False
            parity = (r + c) & 1
            old = colors_by_parity.get(parity)
            if old is None:
                colors_by_parity[parity] = value
            elif old != value:
                return False

    return len(colors_by_parity) == 2 and colors_by_parity[0] != colors_by_parity[1]


def _is_ring(window: np.ndarray, bg: int) -> bool:
    if window.shape != (3, 3):
        return False
    border = [
        int(window[0, 0]),
        int(window[0, 1]),
        int(window[0, 2]),
        int(window[1, 0]),
        int(window[1, 2]),
        int(window[2, 0]),
        int(window[2, 1]),
        int(window[2, 2]),
    ]
    border_color = border[0]
    return (
        border_color != int(bg)
        and all(v == border_color for v in border)
        and int(window[1, 1]) != border_color
    )


def _find_objects(grid: np.ndarray, bg: int) -> list[Obj]:
    h, w = grid.shape
    objects: list[Obj] = []
    occupied = np.zeros((h, w), dtype=bool)

    for r in range(h - 2):
        for c in range(w - 2):
            window = grid[r : r + 3, c : c + 3]
            direction = _clock_direction(window, bg)
            if direction is not None:
                obj = Obj(
                    "clock",
                    r,
                    c,
                    direction=direction,
                    start=_checker_remainder(window, direction),
                )
            elif _is_ring(window, bg):
                obj = Obj("ring", r, c)
            else:
                continue

            patch = occupied[r : r + 3, c : c + 3]
            assert not np.any(patch), "3x3 objects must be separated"
            patch[:, :] = True
            objects.append(obj)

    starts = [obj for obj in objects if obj.kind == "clock" and obj.start]
    assert len(starts) == 1, "expected exactly one checkerboard clock"
    assert any(obj.kind == "ring" for obj in objects), "expected at least one ring object"
    return objects


def _aligned_delta(src: tuple[int, int], dst: tuple[int, int], heading: int) -> int | None:
    sr, sc = src
    dr, dc = DIRS[heading]
    rr = int(dst[0]) - int(sr)
    cc = int(dst[1]) - int(sc)
    if rr == 0 and cc == 0:
        return None

    if dr == 0:
        if rr != 0 or cc * dc <= 0:
            return None
        return abs(cc)
    if dc == 0:
        if cc != 0 or rr * dr <= 0:
            return None
        return abs(rr)
    if abs(rr) != abs(cc) or rr * dr <= 0 or cc * dc <= 0:
        return None
    return abs(rr)


def _first_hit(objects: list[Obj], current: Obj, heading: int) -> Obj:
    src = current.center
    candidates: list[tuple[int, Obj]] = []
    for obj in objects:
        if obj == current:
            continue
        dist = _aligned_delta(src, obj.center, heading)
        if dist is not None:
            candidates.append((dist, obj))

    assert candidates, "each relay segment must hit a clock or ring"
    candidates.sort(key=lambda item: item[0])
    assert len(candidates) == 1 or candidates[0][0] < candidates[1][0], (
        "the first hit on a heading must be unique"
    )
    return candidates[0][1]


def _trace(objects: list[Obj], start: Obj, *, absolute_after_start: bool = False) -> Obj:
    current = start
    heading = int(start.direction)
    seen: set[tuple[Obj, int]] = set()

    for _ in range(len(objects) + 2):
        state = (current, heading)
        assert state not in seen, "clock relay must not loop"
        seen.add(state)

        hit = _first_hit(objects, current, heading)
        if hit.kind == "ring":
            return hit

        assert hit.kind == "clock" and hit.direction is not None
        current = hit
        if absolute_after_start:
            heading = int(hit.direction)
        else:
            heading = (heading + int(hit.direction)) % 8

    raise AssertionError("clock relay did not terminate")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the ring reached by following checker-clock relays whose later hands are relative turns.

    Input structure:
    - The background is the dominant color.
    - Several separated 3x3 clocks have a center pixel and one same-colored hand pixel pointing in one of eight directions.
    - Exactly one clock has its non-center, non-hand cells filled as a checkerboard; it is the start.
    - Several separated 3x3 ring objects have a solid border and a different center color.

    Rule:
    - Start at the checkerboard clock and use its hand as the first absolute heading.
    - Jump along that row, column, or diagonal to the first 3x3 object whose center lies on the heading line.
    - At every later clock, read the hand as a turn measured clockwise from up and add it to the current heading.
    - When the first object on a segment is a ring, return that ring's unchanged 3x3 crop.

    Key insight:
    - After the start, the hand is a rudder, not a compass; treating those hands as absolute directions leads to a decoy ring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = _bg(g)
    objects = _find_objects(g, bg)
    start = [obj for obj in objects if obj.kind == "clock" and obj.start][0]
    target = _trace(objects, start, absolute_after_start=False)
    assert target.kind == "ring"
    return _crop3(g, target.top, target.left)


# ---------------- generator ----------------

GEN_ATTEMPTS = 900
LAYOUT_ATTEMPTS = 260
MIN_CENTER_GAP = 4


def _rand_choice(seq):
    return seq[random.randrange(len(seq))]


def _dir_between(a: tuple[int, int], b: tuple[int, int]) -> int | None:
    rr = int(b[0]) - int(a[0])
    cc = int(b[1]) - int(a[1])
    if rr == 0 and cc == 0:
        return None
    dr = 0 if rr == 0 else (1 if rr > 0 else -1)
    dc = 0 if cc == 0 else (1 if cc > 0 else -1)
    if rr == 0 or cc == 0 or abs(rr) == abs(cc):
        return DIR_TO_INDEX[(dr, dc)]
    return None


def _centers_separated(a: tuple[int, int], b: tuple[int, int], gap: int = MIN_CENTER_GAP) -> bool:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1]))) >= gap


def _center_in_bounds(center: tuple[int, int], shape: tuple[int, int]) -> bool:
    r, c = center
    h, w = shape
    return 1 <= int(r) < h - 1 and 1 <= int(c) < w - 1


def _step(center: tuple[int, int], direction: int, dist: int) -> tuple[int, int]:
    dr, dc = DIRS[direction]
    return int(center[0]) + dr * dist, int(center[1]) + dc * dist


def _make_clock(direction: int, *, start: bool, colors: dict[str, int]) -> np.ndarray:
    hand = int(colors["hand"])
    if start:
        a = int(colors["checker_a"])
        b = int(colors["checker_b"])
        patch = np.empty((3, 3), dtype=int)
        for r in range(3):
            for c in range(3):
                patch[r, c] = a if ((r + c) & 1) == 0 else b
    else:
        patch = np.full((3, 3), int(colors["face"]), dtype=int)

    patch[1, 1] = hand
    dr, dc = DIRS[direction]
    patch[1 + dr, 1 + dc] = hand
    return patch


def _make_ring(border: int, center: int) -> np.ndarray:
    patch = np.full((3, 3), int(border), dtype=int)
    patch[1, 1] = int(center)
    return patch


def _paint_patch(grid: np.ndarray, center: tuple[int, int], patch: np.ndarray) -> None:
    r, c = center
    grid[int(r) - 1 : int(r) + 2, int(c) - 1 : int(c) + 2] = patch


def _first_hit_from_specs(
    centers: list[tuple[int, int]], current_i: int, heading: int
) -> int | None:
    src = centers[current_i]
    candidates: list[tuple[int, int]] = []
    for i, center in enumerate(centers):
        if i == current_i:
            continue
        dist = _aligned_delta(src, center, heading)
        if dist is not None:
            candidates.append((dist, i))
    if not candidates:
        return None
    candidates.sort()
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        return None
    return candidates[0][1]


def _relative_target_index(
    kinds: list[str], centers: list[tuple[int, int]], directions: list[int]
) -> int | None:
    current = 0
    heading = directions[0]
    seen: set[tuple[int, int]] = set()
    for _ in range(len(centers) + 2):
        state = (current, heading)
        if state in seen:
            return None
        seen.add(state)
        hit = _first_hit_from_specs(centers, current, heading)
        if hit is None:
            return None
        if kinds[hit] == "ring":
            return hit
        current = hit
        heading = (heading + directions[hit]) % 8
    return None


def _absolute_target_index(
    kinds: list[str], centers: list[tuple[int, int]], directions: list[int]
) -> int | None:
    current = 0
    heading = directions[0]
    seen: set[tuple[int, int]] = set()
    for _ in range(len(centers) + 2):
        state = (current, heading)
        if state in seen:
            return None
        seen.add(state)
        hit = _first_hit_from_specs(centers, current, heading)
        if hit is None:
            return None
        if kinds[hit] == "ring":
            return hit
        current = hit
        heading = directions[hit]
    return None


def _route_is_clean(
    kinds: list[str],
    centers: list[tuple[int, int]],
    directions: list[int],
    chain_len: int,
    target_i: int,
    decoy_i: int,
) -> bool:
    current = 0
    heading = directions[0]
    for expected in list(range(1, chain_len)) + [target_i]:
        hit = _first_hit_from_specs(centers, current, heading)
        if hit != expected:
            return False
        if kinds[hit] == "ring":
            break
        current = hit
        heading = (heading + directions[hit]) % 8

    rel = _relative_target_index(kinds, centers, directions)
    abs_hit = _absolute_target_index(kinds, centers, directions)
    return rel == target_i and abs_hit == decoy_i and decoy_i != target_i


def _try_route(shape: tuple[int, int]):
    h, w = shape
    relay_count = random.randint(2, 4)
    chain_len = relay_count + 1  # start plus relay clocks
    segment_count = relay_count + 1

    headings = [random.randint(1, 7)]
    for i in range(1, segment_count):
        choices = list(range(8))
        if i == 1:
            # Keep the first absolute misread away from the previous clock.
            bad_turns = {
                headings[0],
                (headings[0] + 4) % 8,
            }
            choices = [d for d in choices if ((d - headings[0]) % 8) not in bad_turns]
        headings.append(_rand_choice(choices))

    distances = [random.randint(5, 8) for _ in range(segment_count)]
    start = (random.randint(3, h - 4), random.randint(3, w - 4))
    centers = [start]
    for direction, dist in zip(headings, distances):
        nxt = _step(centers[-1], direction, dist)
        if not _center_in_bounds(nxt, shape):
            return None
        if any(not _centers_separated(nxt, old) for old in centers):
            return None
        centers.append(nxt)

    # The final center is the true ring; earlier centers are clocks.
    chain_centers = centers[:-1]
    target_center = centers[-1]
    directions = [0] * len(chain_centers)
    directions[0] = headings[0]
    for i in range(1, len(chain_centers)):
        directions[i] = (headings[i] - headings[i - 1]) % 8

    decoy_direction = directions[1]
    decoy_options = list(range(4, 8))
    random.shuffle(decoy_options)
    decoy_center = None
    for dist in decoy_options:
        cand = _step(chain_centers[1], decoy_direction, dist)
        if not _center_in_bounds(cand, shape):
            continue
        all_chain = chain_centers + [target_center]
        if all(_centers_separated(cand, old) for old in all_chain):
            decoy_center = cand
            break
    if decoy_center is None:
        return None

    kinds = ["clock"] * len(chain_centers) + ["ring", "ring"]
    all_centers = chain_centers + [target_center, decoy_center]
    all_directions = directions + [-1, -1]
    target_i = len(chain_centers)
    decoy_i = len(chain_centers) + 1

    if not _route_is_clean(kinds, all_centers, all_directions, len(chain_centers), target_i, decoy_i):
        return None
    return kinds, all_centers, all_directions, target_i, decoy_i, headings, distances


def _try_place_near_miss(
    kinds: list[str],
    centers: list[tuple[int, int]],
    directions: list[int],
    headings: list[int],
    distances: list[int],
    shape: tuple[int, int],
    target_i: int,
    decoy_i: int,
) -> None:
    options = [
        (i, heading, dist)
        for i, (heading, dist) in enumerate(zip(headings, distances))
        if heading in (0, 2, 4, 6) and dist >= 7
    ]
    random.shuffle(options)
    for seg_i, heading, dist in options:
        dr, dc = DIRS[heading]
        perp = (0, 1) if dc == 0 else (1, 0)
        if random.random() < 0.5:
            perp = (-perp[0], -perp[1])
        t_values = list(range(3, dist - 2))
        random.shuffle(t_values)
        for t in t_values:
            base = _step(centers[seg_i], heading, t)
            cand = (base[0] + perp[0], base[1] + perp[1])
            if not _center_in_bounds(cand, shape):
                continue
            if not all(_centers_separated(cand, old) for old in centers):
                continue

            kinds.append("clock")
            centers.append(cand)
            directions.append(random.randrange(8))
            if _route_is_clean(kinds, centers, directions, target_i, target_i, decoy_i):
                return
            kinds.pop()
            centers.pop()
            directions.pop()
    return


def _add_distractors(
    kinds: list[str],
    centers: list[tuple[int, int]],
    directions: list[int],
    shape: tuple[int, int],
    target_i: int,
    decoy_i: int,
) -> bool:
    extra_count = random.randint(2, 5)
    for _ in range(extra_count):
        want_clock = random.random() < 0.55
        placed = False
        for _try in range(90):
            cand = (random.randint(1, shape[0] - 2), random.randint(1, shape[1] - 2))
            if not all(_centers_separated(cand, old) for old in centers):
                continue
            kinds.append("clock" if want_clock else "ring")
            centers.append(cand)
            directions.append(random.randrange(8) if want_clock else -1)
            if _route_is_clean(kinds, centers, directions, target_i, target_i, decoy_i):
                placed = True
                break
            kinds.pop()
            centers.pop()
            directions.pop()
        if not placed:
            return False
    return True


def _color_plan(bg: int) -> dict[str, int]:
    pool = [c for c in ALL_COLORS if c != bg]
    random.shuffle(pool)
    hand = pool.pop()
    face = pool.pop()
    checker_a = pool.pop()
    checker_b = pool.pop()
    return {
        "hand": hand,
        "face": face,
        "checker_a": checker_a,
        "checker_b": checker_b,
    }


def _render(
    shape: tuple[int, int],
    bg: int,
    kinds: list[str],
    centers: list[tuple[int, int]],
    directions: list[int],
    target_i: int,
    decoy_i: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = np.full(shape, int(bg), dtype=int)
    clock_colors = _color_plan(bg)
    colors = [c for c in ALL_COLORS if c != bg]

    target_border = random.choice(colors)
    target_center = random.choice([c for c in ALL_COLORS if c != target_border])
    decoy_border_options = [c for c in colors if c != target_border]
    decoy_border = random.choice(decoy_border_options or colors)
    decoy_center = random.choice([c for c in ALL_COLORS if c != decoy_border and c != target_center])

    for i, (kind, center) in enumerate(zip(kinds, centers)):
        if kind == "clock":
            patch = _make_clock(directions[i], start=(i == 0), colors=clock_colors)
        else:
            if i == target_i:
                patch = _make_ring(target_border, target_center)
            elif i == decoy_i:
                patch = _make_ring(decoy_border, decoy_center)
            else:
                border = random.choice(colors)
                center_color = random.choice([c for c in ALL_COLORS if c != border])
                patch = _make_ring(border, center_color)
        _paint_patch(grid, center, patch)

    target = _crop3(grid, centers[target_i][0] - 1, centers[target_i][1] - 1)
    decoy = _crop3(grid, centers[decoy_i][0] - 1, centers[decoy_i][1] - 1)
    return grid, target, decoy


def _semantic_checks(grid: np.ndarray, expected: np.ndarray, decoy: np.ndarray) -> bool:
    try:
        bg = _bg(grid)
        objects = _find_objects(grid, bg)
        start = [obj for obj in objects if obj.kind == "clock" and obj.start][0]
        relative = _trace(objects, start, absolute_after_start=False)
        absolute = _trace(objects, start, absolute_after_start=True)
        actual = main(grid)
    except Exception:
        return False

    if not np.array_equal(actual, expected):
        return False
    if not np.array_equal(_crop3(grid, relative.top, relative.left), expected):
        return False
    if np.array_equal(_crop3(grid, absolute.top, absolute.left), expected):
        return False
    if not np.array_equal(_crop3(grid, absolute.top, absolute.left), decoy):
        return False
    if np.array_equal(expected, decoy):
        return False
    clocks = [obj for obj in objects if obj.kind == "clock"]
    rings = [obj for obj in objects if obj.kind == "ring"]
    return len(clocks) >= 4 and len(rings) >= 3


def _gen_attempt() -> np.ndarray | None:
    shape = (random.randint(22, 30), random.randint(22, 30))
    for _ in range(LAYOUT_ATTEMPTS):
        route = _try_route(shape)
        if route is None:
            continue
        kinds, centers, directions, target_i, decoy_i, headings, distances = route
        _try_place_near_miss(
            kinds,
            centers,
            directions,
            headings,
            distances,
            shape,
            target_i,
            decoy_i,
        )
        if not _add_distractors(kinds, centers, directions, shape, target_i, decoy_i):
            continue
        bg = random.choice(ALL_COLORS)
        grid, expected, decoy = _render(shape, bg, kinds, centers, directions, target_i, decoy_i)
        if _semantic_checks(grid, expected, decoy):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid clock relay input")
