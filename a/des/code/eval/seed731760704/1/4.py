import itertools
import random
from collections import Counter

import numpy as np


BLACK = 0
GREEN = 3
YELLOW = 4
ORANGE = 7
MAROON = 9

TRACK_COLORS = (1, 2, 5, 6, 8)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIR8 = DIR4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))

PLACEMENT_PAD = 4
PLACEMENT_ATTEMPTS = 200
GEN_RETRIES = 400


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIR4 if connectivity == 4 else DIR8
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((int(rr), int(cc)))
                for dr, dc in dirs:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _rect_loop(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        pts.append((int(r0), int(c)))
    for r in range(r0 + 1, r1):
        pts.append((int(r), int(c1)))
    for c in range(c1, c0 - 1, -1):
        pts.append((int(r1), int(c)))
    for r in range(r1 - 1, r0, -1):
        pts.append((int(r), int(c0)))
    return pts


def _cyclic_runs(seq: list[int]) -> list[tuple[int, int]]:
    runs: list[list[int]] = []
    n = len(seq)
    i = 0
    while i < n:
        j = i + 1
        while j < n and int(seq[j]) == int(seq[i]):
            j += 1
        runs.append([int(seq[i]), int(j - i)])
        i = j
    if len(runs) > 1 and int(runs[0][0]) == int(runs[-1][0]):
        runs[0][1] += runs[-1][1]
        runs.pop()
    return [(int(color), int(length)) for color, length in runs]


def _valid_track_sequence(seq: list[int], colors: set[int]) -> bool:
    if set(map(int, seq)) != {int(c) for c in colors}:
        return False
    runs = _cyclic_runs(seq)
    if len(runs) < 2:
        return False
    for color, length in runs:
        if int(color) not in colors:
            return False
        if not (2 <= int(length) <= 5):
            return False
    return True


def _recover_track_candidates(obs: list[int], color_a: int, color_b: int) -> list[tuple[int, ...]]:
    hidden = [i for i, value in enumerate(obs) if int(value) == YELLOW]
    candidates: list[tuple[int, ...]] = []
    for fill in itertools.product((int(color_a), int(color_b)), repeat=len(hidden)):
        seq = list(map(int, obs))
        for idx, color in zip(hidden, fill):
            seq[int(idx)] = int(color)
        if _valid_track_sequence(seq, {int(color_a), int(color_b)}):
            candidates.append(tuple(seq))
    return sorted(set(candidates))


def _component_distance(loop: list[tuple[int, int]], comp: np.ndarray) -> int:
    best = 10**9
    for r, c in loop:
        for rr, cc in comp:
            dist = abs(int(r) - int(rr)) + abs(int(c) - int(cc))
            if dist < best:
                best = dist
    return int(best)


def _assign_components(
    tracks: list[dict],
    comps: list[dict],
) -> list[dict]:
    n = len(tracks)
    assert len(comps) == n
    if n == 0:
        return []
    costs = [
        [_component_distance(track["loop"], comp["coords"]) for comp in comps]
        for track in tracks
    ]
    best_perm = None
    best_cost = None
    for perm in itertools.permutations(range(n)):
        total = sum(int(costs[i][perm[i]]) for i in range(n))
        if best_cost is None or total < best_cost:
            best_cost = total
            best_perm = perm
    assert best_perm is not None
    return [comps[int(best_perm[i])] for i in range(n)]


def _parse_tracks(grid: np.ndarray) -> list[dict]:
    mask = (grid != BLACK) & (grid != ORANGE) & (grid != MAROON)
    tracks: list[dict] = []
    for coords in _components(mask, connectivity=4):
        r0, c0, r1, c1 = _bbox(coords)
        loop = _rect_loop(r0, c0, r1, c1)
        coord_set = {tuple(map(int, pt)) for pt in coords.tolist()}
        assert len(loop) == len(coord_set)
        assert coord_set == set(loop)

        obs = [int(grid[r, c]) for r, c in loop]
        visible = sorted(set(obs) - {YELLOW})
        assert len(visible) == 2
        candidates = _recover_track_candidates(obs, visible[0], visible[1])
        assert candidates
        seq = list(candidates[0])
        counts = Counter(map(int, seq))
        assert counts[int(visible[0])] != counts[int(visible[1])]
        dominant = int(visible[0]) if counts[int(visible[0])] > counts[int(visible[1])] else int(visible[1])

        tracks.append(
            {
                "bbox": (r0, c0, r1, c1),
                "loop": loop,
                "base_seq": seq,
                "car_indices": [int(i) for i, value in enumerate(obs) if int(value) == YELLOW],
                "dominant": dominant,
            }
        )
    return tracks


def _parse_direction_components(grid: np.ndarray) -> list[dict]:
    comps: list[dict] = []
    for coords in _components(grid == MAROON, connectivity=4):
        assert len(coords) == 3
        r0, c0, r1, c1 = _bbox(coords)
        assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2)
        full = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
        cells = {tuple(map(int, pt)) for pt in coords.tolist()}
        missing = list(full - cells)
        assert len(missing) == 1
        opening = "right" if int(missing[0][1]) == int(c1) else "left"
        comps.append({"coords": coords, "opening": opening})
    return comps


def _parse_step_components(grid: np.ndarray) -> list[dict]:
    comps: list[dict] = []
    for coords in _components(grid == ORANGE, connectivity=4):
        r0, c0, r1, c1 = _bbox(coords)
        size = int(len(coords))
        assert 1 <= size <= 4
        assert r0 == r1 or c0 == c1
        assert size == max(r1 - r0 + 1, c1 - c0 + 1)
        comps.append({"coords": coords, "step": size})
    return comps


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    tracks = _parse_tracks(g)
    dirs = _parse_direction_components(g)
    steps = _parse_step_components(g)
    assert len(tracks) == len(dirs) == len(steps)

    dir_assignment = _assign_components(tracks, dirs)
    step_assignment = _assign_components(tracks, steps)

    for track, dir_comp, step_comp in zip(tracks, dir_assignment, step_assignment):
        loop = track["loop"]
        base_seq = track["base_seq"]
        dominant = int(track["dominant"])
        delta = int(step_comp["step"]) if dir_comp["opening"] == "right" else -int(step_comp["step"])

        for idx, (r, c) in enumerate(loop):
            out[int(r), int(c)] = int(base_seq[idx])

        for idx in track["car_indices"]:
            new_idx = (int(idx) + int(delta)) % len(loop)
            rr, cc = loop[new_idx]
            out[int(rr), int(cc)] = YELLOW if int(base_seq[new_idx]) == dominant else GREEN

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move yellow cars around colored rectangular loop tracks.

    Each track is a 1-pixel-wide rectangular loop with alternating runs of two colors.
    A maroon L marker encodes clockwise vs counterclockwise, and an orange line encodes
    the step count. After moving, cars stay yellow on the dominant track color and turn
    green on the smaller-total track color.
    """
    return _solve(grid)


def _random_segment_lengths(total: int, rng: random.Random) -> list[int] | None:
    seg_counts = [n for n in range(4, min(12, total // 2) + 1, 2) if 2 * n <= total <= 5 * n]
    rng.shuffle(seg_counts)
    for seg_count in seg_counts:
        for _ in range(200):
            lengths = [2] * seg_count
            remaining = total - 2 * seg_count
            for _ in range(remaining):
                choices = [i for i, value in enumerate(lengths) if value < 5]
                if not choices:
                    break
                lengths[rng.choice(choices)] += 1
            if max(lengths) >= 3 and sum(lengths[0::2]) != sum(lengths[1::2]):
                return lengths
    return None


def _cyclic_gap(a: int, b: int, n: int) -> int:
    gap = abs(int(a) - int(b))
    return min(gap, int(n - gap))


def _choose_car_positions(
    base_seq: list[int],
    step: int,
    clockwise: bool,
    rng: random.Random,
) -> list[int] | None:
    n = len(base_seq)
    counts = Counter(map(int, base_seq))
    colors = list(counts)
    assert len(colors) == 2
    dominant = colors[0] if counts[colors[0]] > counts[colors[1]] else colors[1]
    delta = int(step) if clockwise else -int(step)

    eligible = [
        i
        for i in range(n)
        if int(base_seq[i - 1]) == int(base_seq[i]) == int(base_seq[(i + 1) % n])
    ]
    if not eligible:
        return None

    max_cars = min(3, len(eligible))
    count_choices = list(range(1, max_cars + 1))
    rng.shuffle(count_choices)

    weighted_choices = sorted(count_choices, key=lambda k: (k == 1, rng.random()))
    for car_count in weighted_choices:
        valid: list[tuple[int, ...]] = []
        for combo in itertools.combinations(eligible, car_count):
            ok = True
            for i in range(car_count):
                for j in range(i + 1, car_count):
                    if _cyclic_gap(combo[i], combo[j], n) <= 1:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue

            outcomes = {int(base_seq[(idx + delta) % n]) == int(dominant) for idx in combo}
            if car_count >= 2 and len(outcomes) == 1 and len(set(base_seq)) == 2:
                continue
            valid.append(tuple(map(int, combo)))

        if valid:
            choice = list(rng.choice(valid))
            choice.sort()
            return choice
    return None


def _l_shape_cells(block_r: int, block_c: int, opening: str, top_missing: bool) -> list[tuple[int, int]]:
    if opening == "right":
        rel = [(0, 0), (1, 0), (1, 1)] if top_missing else [(0, 0), (0, 1), (1, 0)]
    else:
        rel = [(0, 1), (1, 0), (1, 1)] if top_missing else [(0, 0), (0, 1), (1, 1)]
    return [(int(block_r + dr), int(block_c + dc)) for dr, dc in rel]


def _indicator_cells(track: dict, rng: random.Random) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    r0 = int(track["r0"])
    c0 = int(track["c0"])
    h = int(track["h"])
    w = int(track["w"])
    side = track["side"]
    step = int(track["step"])
    opening = "right" if track["clockwise"] else "left"
    top_missing = bool(rng.randrange(2))

    if side == "top":
        area_r0 = r0 - 4
        area_c0 = rng.randint(c0, c0 + w - 4)
        orange_row = area_r0
        orange_c0 = area_c0 + rng.randint(0, 4 - step)
        orange = [(orange_row, int(orange_c0 + k)) for k in range(step)]
        block_r = area_r0 + 2
        block_c = area_c0 + rng.choice((0, 2))
    elif side == "bottom":
        area_r0 = r0 + h
        area_c0 = rng.randint(c0, c0 + w - 4)
        orange_row = area_r0 + 3
        orange_c0 = area_c0 + rng.randint(0, 4 - step)
        orange = [(orange_row, int(orange_c0 + k)) for k in range(step)]
        block_r = area_r0
        block_c = area_c0 + rng.choice((0, 2))
    elif side == "left":
        area_r0 = rng.randint(r0, r0 + h - 4)
        area_c0 = c0 - 4
        orange_col = area_c0
        orange_r0 = area_r0 + rng.randint(0, 4 - step)
        orange = [(int(orange_r0 + k), orange_col) for k in range(step)]
        block_r = area_r0 + rng.choice((0, 2))
        block_c = area_c0 + 2
    else:
        area_r0 = rng.randint(r0, r0 + h - 4)
        area_c0 = c0 + w
        orange_col = area_c0 + 3
        orange_r0 = area_r0 + rng.randint(0, 4 - step)
        orange = [(int(orange_r0 + k), orange_col) for k in range(step)]
        block_r = area_r0 + rng.choice((0, 2))
        block_c = area_c0

    maroon = _l_shape_cells(block_r, block_c, opening, top_missing)
    return maroon, orange


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        int(ar1) < int(br0) - 1
        or int(br1) < int(ar0) - 1
        or int(ac1) < int(bc0) - 1
        or int(bc1) < int(ac0) - 1
    )


def _make_track_spec(h: int, w: int, rng: random.Random) -> dict | None:
    loop_len = len(_rect_loop(0, 0, h - 1, w - 1))
    lengths = _random_segment_lengths(loop_len, rng)
    if lengths is None:
        return None

    color_a, color_b = rng.sample(list(TRACK_COLORS), 2)
    start_color = color_a if rng.random() < 0.5 else color_b
    other_color = color_b if start_color == color_a else color_a

    base_seq: list[int] = []
    for idx, seg_len in enumerate(lengths):
        color = start_color if idx % 2 == 0 else other_color
        base_seq.extend([int(color)] * int(seg_len))

    shift = rng.randrange(loop_len)
    base_seq = base_seq[shift:] + base_seq[:shift]
    counts = Counter(map(int, base_seq))
    if counts[int(color_a)] == counts[int(color_b)]:
        return None

    step = rng.randint(1, 4)
    clockwise = bool(rng.randrange(2))
    car_positions = _choose_car_positions(base_seq, step, clockwise, rng)
    if car_positions is None:
        return None

    return {
        "h": int(h),
        "w": int(w),
        "base_seq": list(map(int, base_seq)),
        "step": int(step),
        "clockwise": clockwise,
        "cars": list(map(int, car_positions)),
        "dominant": int(color_a) if counts[int(color_a)] > counts[int(color_b)] else int(color_b),
        "side": rng.choice(("top", "bottom", "left", "right")),
    }


def _render_input(tracks: list[dict], shape: tuple[int, int], rng: random.Random) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)

    for track in tracks:
        r0 = int(track["r0"])
        c0 = int(track["c0"])
        h = int(track["h"])
        w = int(track["w"])
        loop = _rect_loop(r0, c0, r0 + h - 1, c0 + w - 1)
        track["loop"] = loop

        for idx, (r, c) in enumerate(loop):
            grid[int(r), int(c)] = int(track["base_seq"][idx])

        for idx in track["cars"]:
            r, c = loop[int(idx)]
            grid[int(r), int(c)] = YELLOW

        maroon, orange = _indicator_cells(track, rng)
        track["maroon"] = maroon
        track["orange"] = orange
        for r, c in maroon:
            assert grid[int(r), int(c)] == BLACK
            grid[int(r), int(c)] = MAROON
        for r, c in orange:
            assert grid[int(r), int(c)] == BLACK
            grid[int(r), int(c)] = ORANGE

    return grid


def _expected_output(inp: np.ndarray, tracks: list[dict]) -> np.ndarray:
    out = np.array(inp, copy=True, dtype=int)
    for track in tracks:
        loop = track["loop"]
        base_seq = track["base_seq"]
        delta = int(track["step"]) if track["clockwise"] else -int(track["step"])
        dominant = int(track["dominant"])

        for idx, (r, c) in enumerate(loop):
            out[int(r), int(c)] = int(base_seq[idx])

        for idx in track["cars"]:
            new_idx = (int(idx) + int(delta)) % len(loop)
            rr, cc = loop[new_idx]
            out[int(rr), int(cc)] = YELLOW if int(base_seq[new_idx]) == dominant else GREEN

    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_tracks = rng.choices([1, 2, 3], weights=[2, 3, 2])[0]
    min_side = 15 + 4 * n_tracks
    h = rng.randint(min_side, 30)
    w = rng.randint(min_side, 30)

    tracks: list[dict] = []
    reserved: list[tuple[int, int, int, int]] = []

    for _ in range(n_tracks):
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            track_h = rng.randint(4, min(7, h - 2 * PLACEMENT_PAD))
            track_w = rng.randint(4, min(7, w - 2 * PLACEMENT_PAD))
            spec = _make_track_spec(track_h, track_w, rng)
            if spec is None:
                continue

            r0 = rng.randint(PLACEMENT_PAD, h - track_h - PLACEMENT_PAD)
            c0 = rng.randint(PLACEMENT_PAD, w - track_w - PLACEMENT_PAD)
            box = (
                int(r0 - PLACEMENT_PAD),
                int(c0 - PLACEMENT_PAD),
                int(r0 + track_h - 1 + PLACEMENT_PAD),
                int(c0 + track_w - 1 + PLACEMENT_PAD),
            )
            if any(_boxes_too_close(box, other) for other in reserved):
                continue

            spec["r0"] = int(r0)
            spec["c0"] = int(c0)
            tracks.append(spec)
            reserved.append(box)
            placed = True
            break
        if not placed:
            return None

    inp = _render_input(tracks, (h, w), rng)

    if len(_components(inp == MAROON, connectivity=4)) != n_tracks:
        return None
    if len(_components(inp == ORANGE, connectivity=4)) != n_tracks:
        return None
    if len(_parse_tracks(inp)) != n_tracks:
        return None

    for track in tracks:
        obs = [int(inp[r, c]) for r, c in track["loop"]]
        colors = sorted(set(track["base_seq"]))
        candidates = _recover_track_candidates(obs, colors[0], colors[1])
        if len(candidates) != 1 or list(candidates[0]) != list(track["base_seq"]):
            return None

    expected = _expected_output(inp, tracks)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 4")
