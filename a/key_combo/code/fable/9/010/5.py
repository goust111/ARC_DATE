from __future__ import annotations

import random
from collections import Counter

import numpy as np


GREY = 5
RING_OFFSETS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
)
CARDINALS = ((-1, 0), (1, 0), (0, -1), (0, 1))
MARKER_TARGET_INDEX = {
    (-1, 0): 1,
    (0, 1): 3,
    (1, 0): 5,
    (0, -1): 7,
}
PALETTE = tuple(range(10))
GEN_RETRIES = 800


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _ring_coords(center: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = center
    return [(r + dr, c + dc) for dr, dc in RING_OFFSETS]


def _read_sequence(grid: np.ndarray, center: tuple[int, int]) -> np.ndarray:
    return np.array([int(grid[r, c]) for r, c in _ring_coords(center)], dtype=int)


def _write_sequence(grid: np.ndarray, center: tuple[int, int], seq: np.ndarray) -> None:
    for (r, c), color in zip(_ring_coords(center), seq.tolist()):
        grid[r, c] = int(color)


def _detect_rings(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = map(int, grid.shape)
    rings = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) != bg:
                continue
            coords = _ring_coords((r, c))
            values = [int(grid[rr, cc]) for rr, cc in coords]
            if any(v == bg for v in values):
                continue
            if values.count(GREY) != 1:
                continue
            rings.append({"center": (r, c), "seq": np.array(values, dtype=int)})
    assert rings, "expected at least one 3x3 color ring"
    centers = [ring["center"] for ring in rings]
    assert len(set(centers)) == len(centers), "ring centers must be unique"
    return rings


def _build_chain(rings: list[dict], marker: tuple[int, int]) -> tuple[list[dict], tuple[int, int]]:
    centers = [ring["center"] for ring in rings]
    index_of = {center: i for i, center in enumerate(centers)}
    adjacency = {i: [] for i in range(len(rings))}
    for i, (r0, c0) in enumerate(centers):
        for j in range(i + 1, len(rings)):
            r1, c1 = centers[j]
            if abs(r0 - r1) + abs(c0 - c1) == 3 and (r0 == r1 or c0 == c1):
                adjacency[i].append(j)
                adjacency[j].append(i)

    first_candidates = []
    mr, mc = marker
    for idx, (r, c) in enumerate(centers):
        dr = mr - r
        dc = mc - c
        if (dr, dc) in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            first_candidates.append((idx, (dr // 2, dc // 2)))
    assert len(first_candidates) == 1, "marker must identify exactly one first ring"
    first_idx, marker_dir = first_candidates[0]
    target_deg = 0 if len(rings) == 1 else 1
    assert len(adjacency[first_idx]) == target_deg, "first ring must be an endpoint of the chain"

    order = []
    prev = None
    curr = first_idx
    while True:
        order.append(curr)
        nxt = [n for n in adjacency[curr] if n != prev]
        if not nxt:
            break
        assert len(nxt) == 1, "chain must not branch"
        prev, curr = curr, nxt[0]
        assert curr not in order, "chain must not loop"

    assert len(order) == len(rings), "chain must include every ring"
    if len(rings) > 1:
        degs = sorted(len(adjacency[i]) for i in range(len(rings)))
        assert degs.count(1) == 2 and all(d in (1, 2) for d in degs), "rings must form a simple path"
    return [rings[i] for i in order], marker_dir


def _parse(grid: np.ndarray) -> tuple[int, list[dict], tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)
    rings = _detect_rings(g, bg)
    ring_cells = {coord for ring in rings for coord in _ring_coords(ring["center"])}
    marker_candidates = [
        (int(r), int(c))
        for r, c in np.argwhere(g == GREY)
        if (int(r), int(c)) not in ring_cells
    ]
    assert len(marker_candidates) == 1, "expected exactly one external grey marker"
    ordered, marker_dir = _build_chain(rings, marker_candidates[0])
    return bg, ordered, marker_dir


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate a chain of 3x3 color rings so the lead ring aligns its grey tooth to the external grey marker and every later ring counter-rotates by the same step count.

    Input structure:
    - The grid has a dominant background color.
    - The foreground consists of one chain of touching 3x3 rings: each ring has eight non-background perimeter pixels around a background center.
    - Every ring uses eight distinct perimeter colors and contains exactly one grey perimeter pixel.
    - Exactly one grey marker pixel lies two cells away from the first ring in a cardinal direction.

    Rule:
    - On the first ring, rotate the entire 8-pixel perimeter cyclically until its grey pixel reaches the side-center closest to the marker.
    - Choose the shorter of the two rotation directions; the input guarantees the two direction lengths are different.
    - Propagate that same step count along the chain: each next ring rotates in the opposite direction from the previous ring.
    - Keep the marker, every ring center, and the background unchanged.

    Key insight:
    - The marker determines only the lead ring's target tooth and therefore a single step count; every other ring is fixed by alternating gear-like counter-rotation with that same number of perimeter steps.
    """
    g = np.asarray(grid, dtype=int)
    bg, ordered_rings, marker_dir = _parse(g)
    target_idx = MARKER_TARGET_INDEX[marker_dir]

    first_seq = ordered_rings[0]["seq"]
    grey_idx = int(np.flatnonzero(first_seq == GREY)[0])
    cw_steps = (target_idx - grey_idx) % 8
    ccw_steps = (grey_idx - target_idx) % 8
    assert cw_steps != ccw_steps
    assert min(cw_steps, ccw_steps) > 0
    steps = min(cw_steps, ccw_steps)
    first_sign = 1 if cw_steps < ccw_steps else -1

    out = g.copy()
    for i, ring in enumerate(ordered_rings):
        sign = first_sign if i % 2 == 0 else -first_sign
        rotated = np.roll(ring["seq"], sign * steps)
        _write_sequence(out, ring["center"], rotated)
        rr, cc = ring["center"]
        out[rr, cc] = bg
    return out


def _step_dirs(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [
        (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        for i in range(len(path) - 1)
    ]


def _is_bent(path: list[tuple[int, int]]) -> bool:
    dirs = _step_dirs(path)
    return any(dirs[i] != dirs[i - 1] for i in range(1, len(dirs)))


def _generate_path(length: int, require_bend: bool) -> list[tuple[int, int]] | None:
    for _ in range(200):
        path = [(0, 0)]
        used = {(0, 0)}
        last_dir = None
        while len(path) < length:
            cr, cc = path[-1]
            options = []
            for dr, dc in CARDINALS:
                nr, nc = cr + dr, cc + dc
                if (nr, nc) in used:
                    continue
                extra_edge = False
                for pr, pc in used:
                    if (pr, pc) == (cr, cc):
                        continue
                    if abs(pr - nr) + abs(pc - nc) == 1:
                        extra_edge = True
                        break
                if extra_edge:
                    continue
                score = 0
                if last_dir == (dr, dc):
                    score += 2
                options.append(((nr, nc), score))
            if not options:
                break
            random.shuffle(options)
            options.sort(key=lambda item: item[1], reverse=True)
            if require_bend and len(path) >= 2:
                bend_options = [opt for opt in options if last_dir != (opt[0][0] - cr, opt[0][1] - cc)]
                if bend_options and random.random() < 0.7:
                    nxt = random.choice(bend_options)[0]
                else:
                    nxt = random.choice(options[: max(1, len(options) // 2)])[0]
            else:
                pool = options[: max(1, len(options) // 2)] if random.random() < 0.7 else options
                nxt = random.choice(pool)[0]
            last_dir = (nxt[0] - cr, nxt[1] - cc)
            path.append(nxt)
            used.add(nxt)
        if len(path) == length and _is_bent(path) == require_bend:
            return path
    return None


def _random_ring_sequence(bg: int, grey_index: int | None = None) -> np.ndarray:
    colors = [c for c in PALETTE if c not in {bg, GREY}]
    seq = [None] * 8
    if grey_index is None:
        grey_index = random.randrange(8)
    seq[grey_index] = GREY
    others = random.sample(colors, 7)
    random.shuffle(others)
    fill_indices = [i for i in range(8) if i != grey_index]
    for idx, color in zip(fill_indices, others):
        seq[idx] = color
    return np.array(seq, dtype=int)


def _render_input(
    path: list[tuple[int, int]],
    marker_dir: tuple[int, int],
    bg: int,
    first_steps: int,
    first_sign: int,
) -> tuple[np.ndarray, list[tuple[int, int]], tuple[int, int], list[np.ndarray]]:
    min_r = min(r for r, _ in path)
    max_r = max(r for r, _ in path)
    min_c = min(c for _, c in path)
    max_c = max(c for _, c in path)

    top_pad = random.randint(0, 3)
    bottom_pad = random.randint(0, 3)
    left_pad = random.randint(0, 3)
    right_pad = random.randint(0, 3)
    if marker_dir == (-1, 0):
        top_pad = max(top_pad, 1)
    elif marker_dir == (1, 0):
        bottom_pad = max(bottom_pad, 1)
    elif marker_dir == (0, -1):
        left_pad = max(left_pad, 1)
    else:
        right_pad = max(right_pad, 1)

    height = 3 * (max_r - min_r) + 3 + top_pad + bottom_pad
    width = 3 * (max_c - min_c) + 3 + left_pad + right_pad
    grid = np.full((height, width), bg, dtype=int)

    centers = [
        (top_pad + 1 + 3 * (r - min_r), left_pad + 1 + 3 * (c - min_c))
        for r, c in path
    ]
    target_idx = MARKER_TARGET_INDEX[marker_dir]
    first_grey_idx = (target_idx - first_sign * first_steps) % 8
    sequences = [_random_ring_sequence(bg, grey_index=first_grey_idx)]
    for _ in range(1, len(centers)):
        sequences.append(_random_ring_sequence(bg))

    for center, seq in zip(centers, sequences):
        _write_sequence(grid, center, seq)
    fr, fc = centers[0]
    mr = fr + 2 * marker_dir[0]
    mc = fc + 2 * marker_dir[1]
    grid[mr, mc] = GREY
    return grid, centers, (mr, mc), sequences


def _expected_output(
    inp: np.ndarray,
    centers: list[tuple[int, int]],
    sequences: list[np.ndarray],
    steps: int,
    first_sign: int,
) -> np.ndarray:
    out = inp.copy()
    bg = get_bg(inp)
    for i, (center, seq) in enumerate(zip(centers, sequences)):
        sign = first_sign if i % 2 == 0 else -first_sign
        _write_sequence(out, center, np.roll(seq, sign * steps))
        out[center] = bg
    return out


def _wrong_same_direction(
    inp: np.ndarray,
    centers: list[tuple[int, int]],
    sequences: list[np.ndarray],
    steps: int,
    first_sign: int,
) -> np.ndarray:
    out = inp.copy()
    bg = get_bg(inp)
    for center, seq in zip(centers, sequences):
        _write_sequence(out, center, np.roll(seq, first_sign * steps))
        out[center] = bg
    return out


def _wrong_first_only(
    inp: np.ndarray,
    centers: list[tuple[int, int]],
    sequences: list[np.ndarray],
    steps: int,
    first_sign: int,
) -> np.ndarray:
    out = inp.copy()
    bg = get_bg(inp)
    _write_sequence(out, centers[0], np.roll(sequences[0], first_sign * steps))
    out[centers[0]] = bg
    return out


def _semantic_checks(
    inp: np.ndarray,
    centers: list[tuple[int, int]],
    marker: tuple[int, int],
    sequences: list[np.ndarray],
    steps: int,
    first_sign: int,
    require_bend: bool,
) -> bool:
    if max(inp.shape) > 30:
        return False
    expected = _expected_output(inp, centers, sequences, steps, first_sign)
    try:
        actual = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False

    bg, ordered_rings, marker_dir = _parse(inp)
    parsed_centers = [ring["center"] for ring in ordered_rings]
    if parsed_centers != centers:
        return False
    if marker != (centers[0][0] + 2 * marker_dir[0], centers[0][1] + 2 * marker_dir[1]):
        return False
    if _is_bent([(r // 3, c // 3) for r, c in centers]) != require_bend:
        return False

    first_seq = ordered_rings[0]["seq"]
    grey_idx = int(np.flatnonzero(first_seq == GREY)[0])
    target_idx = MARKER_TARGET_INDEX[marker_dir]
    cw_steps = (target_idx - grey_idx) % 8
    ccw_steps = (grey_idx - target_idx) % 8
    if min(cw_steps, ccw_steps) != steps or (1 if cw_steps < ccw_steps else -1) != first_sign:
        return False

    if len(centers) < 2:
        return False
    if np.array_equal(expected, _wrong_same_direction(inp, centers, sequences, steps, first_sign)):
        return False
    if np.array_equal(expected, _wrong_first_only(inp, centers, sequences, steps, first_sign)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    length = random.choice((2, 3, 4, 5))
    require_bend = length >= 3 and random.random() < 0.5
    path = _generate_path(length, require_bend)
    if path is None:
        return None

    if length > 1:
        first_step_dir = (path[1][0] - path[0][0], path[1][1] - path[0][1])
        marker_choices = [d for d in CARDINALS if d != first_step_dir]
    else:
        marker_choices = list(CARDINALS)
    marker_dir = random.choice(marker_choices)

    bg = random.choice([c for c in PALETTE if c != GREY])
    steps = random.choice((1, 2, 3))
    first_sign = random.choice((-1, 1))
    inp, centers, marker, sequences = _render_input(path, marker_dir, bg, steps, first_sign)
    return inp if _semantic_checks(inp, centers, marker, sequences, steps, first_sign, require_bend) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid gear-ring chain input")
