from __future__ import annotations

import numpy as np


BG = 0
TRACK = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
MAX_ATTEMPTS = 400

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_ORDER = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}


def _track_color(grid: np.ndarray) -> int | None:
    colors = [int(c) for c in np.unique(grid) if int(c) != BG]
    best: tuple[int, int, int] | None = None
    for color in colors:
        mask = grid == int(color)
        horiz = int(np.count_nonzero(mask[:, 1:] & mask[:, :-1]))
        vert = int(np.count_nonzero(mask[1:, :] & mask[:-1, :]))
        adj = horiz + vert
        if adj == 0:
            continue
        score = (adj, int(mask.sum()), int(color))
        if best is None or score > best:
            best = score
    return None if best is None else int(best[2])


def _nearest_track_candidates(
    grid: np.ndarray,
    r: int,
    c: int,
    track_color: int,
) -> list[tuple[int, int, int, str, tuple[int, int]]]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int, str, tuple[int, int]]] = []
    for dr, dc in DIRS:
        rr = int(r)
        cc = int(c)
        dist = 0
        while True:
            rr += int(dr)
            cc += int(dc)
            dist += 1
            if not (0 <= rr < h and 0 <= cc < w):
                break
            value = int(grid[rr, cc])
            if value == BG:
                continue
            if value == int(track_color):
                orientation = "h" if dr == 0 else "v"
                candidates.append((dist, rr, cc, orientation, (int(dr), int(dc))))
            break
    candidates.sort(key=lambda item: (item[0], item[1], item[2], DIR_ORDER[item[4]]))
    return candidates


def _find_emitters(
    grid: np.ndarray,
    *,
    require_unique_nearest: bool = False,
) -> tuple[int | None, list[dict]]:
    g = np.asarray(grid, dtype=int)
    track_color = _track_color(g)
    if track_color is None:
        return None, []

    sources = np.argwhere((g != BG) & (g != int(track_color)))
    emitters: list[dict] = []
    for r0, c0 in sources:
        r = int(r0)
        c = int(c0)
        candidates = _nearest_track_candidates(g, r, c, int(track_color))
        if not candidates:
            return None, []
        if require_unique_nearest and len(candidates) >= 2 and int(candidates[0][0]) == int(candidates[1][0]):
            return None, []
        dist, lr, lc, orientation, direction = candidates[0]
        emitters.append(
            {
                "source": (r, c),
                "landing": (int(lr), int(lc)),
                "orientation": orientation,
                "move_dir": direction,
                "distance": int(dist),
                "color": int(g[r, c]),
            }
        )
    emitters.sort(key=lambda item: (item["source"][0], item["source"][1]))
    return int(track_color), emitters


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide isolated colored pixels onto a grey track, then emit perpendicular rays.

    Each non-grey colored source pixel moves in a straight line to the nearest visible
    grey track cell in the same row or column. Once landed, it repaints that track
    cell and emits two opposite rays perpendicular to the track orientation. Rays only
    color cells that were black after all sources were removed; collisions are resolved
    by smaller distance to the landing point, then by smaller landing row.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    track_color, emitters = _find_emitters(g)
    if track_color is None or not emitters:
        return g.copy()

    base = g.copy()
    for emitter in emitters:
        sr, sc = emitter["source"]
        base[int(sr), int(sc)] = BG

    out = base.copy()
    for emitter in emitters:
        lr, lc = emitter["landing"]
        out[int(lr), int(lc)] = int(emitter["color"])

    best_dist = np.full((h, w), h + w + 1, dtype=int)
    best_row = np.full((h, w), h + 1, dtype=int)
    best_col = np.full((h, w), w + 1, dtype=int)
    best_color = np.full((h, w), BG, dtype=int)

    for emitter in emitters:
        lr, lc = emitter["landing"]
        if emitter["orientation"] == "h":
            ray_dirs = [(-1, 0), (1, 0)]
        else:
            ray_dirs = [(0, -1), (0, 1)]

        for dr, dc in ray_dirs:
            rr = int(lr)
            cc = int(lc)
            dist = 0
            while True:
                rr += int(dr)
                cc += int(dc)
                dist += 1
                if not (0 <= rr < h and 0 <= cc < w):
                    break
                if int(base[rr, cc]) != BG:
                    break
                if (
                    int(dist) < int(best_dist[rr, cc])
                    or (
                        int(dist) == int(best_dist[rr, cc])
                        and (
                            int(lr) < int(best_row[rr, cc])
                            or (
                                int(lr) == int(best_row[rr, cc])
                                and int(lc) < int(best_col[rr, cc])
                            )
                        )
                    )
                ):
                    best_dist[rr, cc] = int(dist)
                    best_row[rr, cc] = int(lr)
                    best_col[rr, cc] = int(lc)
                    best_color[rr, cc] = int(emitter["color"])

    paint_mask = best_dist <= h + w
    out[paint_mask] = best_color[paint_mask]
    return out


def _choose_spaced_positions(
    rng: np.random.Generator,
    low: int,
    high: int,
    count: int,
    min_gap: int,
    parity: int | None = None,
) -> list[int] | None:
    values = [v for v in range(int(low), int(high) + 1) if parity is None or (v % 2) == int(parity)]
    if len(values) < count:
        return None
    for _ in range(200):
        picked = sorted(int(v) for v in rng.choice(np.array(values, dtype=int), size=count, replace=False))
        if all(int(b) - int(a) >= int(min_gap) for a, b in zip(picked, picked[1:])):
            return picked
    return None


def _random_side_pair(rng: np.random.Generator) -> set[str]:
    mode = int(rng.integers(0, 3))
    if mode == 0:
        return {"left"}
    if mode == 1:
        return {"right"}
    return {"left", "right"}


def _random_side_pair_vertical(rng: np.random.Generator) -> set[str]:
    mode = int(rng.integers(0, 3))
    if mode == 0:
        return {"top"}
    if mode == 1:
        return {"bottom"}
    return {"top", "bottom"}


def _sample_long_segment(
    rng: np.random.Generator,
    size: int,
    min_len: int,
    shared_side: str | None,
    shared_value: int | None,
) -> tuple[int, int] | None:
    for _ in range(100):
        if shared_side == "start":
            start = int(shared_value)
            end = int(rng.integers(start + min_len - 1, size - 1))
        elif shared_side == "end":
            end = int(shared_value)
            start = int(rng.integers(1, end - min_len + 2))
        else:
            start = int(rng.integers(1, size - min_len))
            end = int(rng.integers(start + min_len - 1, size - 1))
        if 1 <= start < end <= size - 2 and int(end - start + 1) >= int(min_len):
            return int(start), int(end)
    return None


def _paint_tracks(grid: np.ndarray, tracks: list[dict]) -> None:
    for track in tracks:
        if track["ori"] == "h":
            grid[int(track["r"]), int(track["c0"]) : int(track["c1"]) + 1] = TRACK
        else:
            grid[int(track["r0"]) : int(track["r1"]) + 1, int(track["c"])] = TRACK


def _choose_colors(rng: np.random.Generator, count: int) -> list[int]:
    if count <= len(PALETTE):
        return [int(v) for v in rng.choice(PALETTE, size=count, replace=False)]
    return [int(v) for v in rng.choice(PALETTE, size=count, replace=True)]


def _build_grid(h: int, w: int, tracks: list[dict], sources: list[tuple[int, int, int]]) -> np.ndarray:
    grid = np.zeros((int(h), int(w)), dtype=int)
    _paint_tracks(grid, tracks)
    for r, c, color in sources:
        grid[int(r), int(c)] = int(color)
    return grid


def _is_valid_generated(grid: np.ndarray) -> bool:
    track_color, emitters = _find_emitters(grid, require_unique_nearest=True)
    if track_color is None or not emitters:
        return False
    if len({tuple(e["landing"]) for e in emitters}) != len(emitters):
        return False
    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if int(np.count_nonzero((grid == BG) & (out != BG))) == 0:
        return False
    return True


def _gen_horizontal_only(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(10, 19))
    w = int(rng.integers(12, 21))
    max_tracks = min(3, max(1, (h - 2) // 3))
    track_count = int(rng.integers(1, max_tracks + 1))
    collision = bool(track_count >= 2 and rng.random() < 0.55)
    rows = _choose_spaced_positions(
        rng,
        1,
        h - 2,
        track_count,
        4 if collision else 3,
        parity=int(rng.integers(0, 2)) if collision else None,
    )
    if rows is None:
        return None

    min_len = max(5, int(np.ceil(0.55 * w)))
    shared_mode: str | None = None
    shared_value: int | None = None
    if collision:
        if rng.random() < 0.5:
            shared_mode = "start"
            high = w - min_len - 1
            if high < 1:
                return None
            shared_value = int(rng.integers(1, high + 1))
        else:
            shared_mode = "end"
            low = min_len
            high = w - 2
            if low > high:
                return None
            shared_value = int(rng.integers(low, high + 1))

    tracks: list[dict] = []
    for row in rows:
        seg = _sample_long_segment(rng, w, min_len, shared_mode, shared_value)
        if seg is None:
            return None
        c0, c1 = seg
        tracks.append({"ori": "h", "r": int(row), "c0": int(c0), "c1": int(c1)})

    source_slots: list[tuple[int, int]] = []
    for idx, track in enumerate(tracks):
        sides = _random_side_pair(rng)
        if collision and idx < 2:
            if shared_mode == "start":
                sides.add("left")
            else:
                sides.add("right")
        if "left" in sides:
            col = int(rng.integers(0, int(track["c0"])))
            source_slots.append((int(track["r"]), col))
        if "right" in sides:
            col = int(rng.integers(int(track["c1"]) + 1, w))
            source_slots.append((int(track["r"]), col))

    colors = _choose_colors(rng, len(source_slots))
    sources = [(r, c, col) for (r, c), col in zip(source_slots, colors)]
    grid = _build_grid(h, w, tracks, sources)
    return grid if _is_valid_generated(grid) else None


def _gen_vertical_only(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(12, 21))
    w = int(rng.integers(10, 19))
    max_tracks = min(3, max(1, (w - 2) // 3))
    track_count = int(rng.integers(1, max_tracks + 1))
    collision = bool(track_count >= 2 and rng.random() < 0.55)
    cols = _choose_spaced_positions(
        rng,
        1,
        w - 2,
        track_count,
        4 if collision else 3,
        parity=int(rng.integers(0, 2)) if collision else None,
    )
    if cols is None:
        return None

    min_len = max(5, int(np.ceil(0.55 * h)))
    shared_mode: str | None = None
    shared_value: int | None = None
    if collision:
        if rng.random() < 0.5:
            shared_mode = "start"
            high = h - min_len - 1
            if high < 1:
                return None
            shared_value = int(rng.integers(1, high + 1))
        else:
            shared_mode = "end"
            low = min_len
            high = h - 2
            if low > high:
                return None
            shared_value = int(rng.integers(low, high + 1))

    tracks: list[dict] = []
    for col in cols:
        seg = _sample_long_segment(rng, h, min_len, shared_mode, shared_value)
        if seg is None:
            return None
        r0, r1 = seg
        tracks.append({"ori": "v", "c": int(col), "r0": int(r0), "r1": int(r1)})

    source_slots: list[tuple[int, int]] = []
    for idx, track in enumerate(tracks):
        sides = _random_side_pair_vertical(rng)
        if collision and idx < 2:
            if shared_mode == "start":
                sides.add("top")
            else:
                sides.add("bottom")
        if "top" in sides:
            row = int(rng.integers(0, int(track["r0"])))
            source_slots.append((row, int(track["c"])))
        if "bottom" in sides:
            row = int(rng.integers(int(track["r1"]) + 1, h))
            source_slots.append((row, int(track["c"])))

    colors = _choose_colors(rng, len(source_slots))
    sources = [(r, c, col) for (r, c), col in zip(source_slots, colors)]
    grid = _build_grid(h, w, tracks, sources)
    return grid if _is_valid_generated(grid) else None


def _gen_mixed(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(14, 21))
    w = int(rng.integers(14, 21))
    min_h_len = max(5, int(np.ceil(0.5 * w)))
    min_v_len = max(5, int(np.ceil(0.5 * h)))

    h_side = "left" if rng.random() < 0.5 else "right"
    v_side = "above" if rng.random() < 0.5 else "below"

    if h_side == "left":
        c_v_low = min_h_len + 1
        c_v_high = w - 2
    else:
        c_v_low = 1
        c_v_high = w - min_h_len - 2
    if c_v_low > c_v_high:
        return None
    c_v = int(rng.integers(c_v_low, c_v_high + 1))

    if v_side == "above":
        r_h_low = min_v_len + 1
        r_h_high = h - 2
    else:
        r_h_low = 1
        r_h_high = h - min_v_len - 2
    if r_h_low > r_h_high:
        return None
    r_h = int(rng.integers(r_h_low, r_h_high + 1))

    if h_side == "left":
        h_c1 = int(rng.integers(min_h_len, c_v))
        h_c0 = int(rng.integers(1, h_c1 - min_h_len + 2))
    else:
        h_c0 = int(rng.integers(c_v + 1, w - min_h_len))
        h_c1 = int(rng.integers(h_c0 + min_h_len - 1, w - 1))
    if not (1 <= h_c0 < h_c1 <= w - 2):
        return None

    if v_side == "above":
        v_r1 = int(rng.integers(min_v_len, r_h))
        v_r0 = int(rng.integers(1, v_r1 - min_v_len + 2))
    else:
        v_r0 = int(rng.integers(r_h + 1, h - min_v_len))
        v_r1 = int(rng.integers(v_r0 + min_v_len - 1, h - 1))
    if not (1 <= v_r0 < v_r1 <= h - 2):
        return None

    tracks = [
        {"ori": "h", "r": int(r_h), "c0": int(h_c0), "c1": int(h_c1)},
        {"ori": "v", "c": int(c_v), "r0": int(v_r0), "r1": int(v_r1)},
    ]

    source_slots: list[tuple[int, int]] = []
    h_sides = _random_side_pair(rng)
    if "left" in h_sides:
        choices = [c for c in range(0, int(h_c0)) if c != int(c_v)]
        if not choices:
            return None
        source_slots.append((int(r_h), int(rng.choice(np.array(choices, dtype=int)))))
    if "right" in h_sides:
        choices = [c for c in range(int(h_c1) + 1, w) if c != int(c_v)]
        if not choices:
            return None
        source_slots.append((int(r_h), int(rng.choice(np.array(choices, dtype=int)))))

    v_sides = _random_side_pair_vertical(rng)
    if "top" in v_sides:
        choices = [r for r in range(0, int(v_r0)) if r != int(r_h)]
        if not choices:
            return None
        source_slots.append((int(rng.choice(np.array(choices, dtype=int))), int(c_v)))
    if "bottom" in v_sides:
        choices = [r for r in range(int(v_r1) + 1, h) if r != int(r_h)]
        if not choices:
            return None
        source_slots.append((int(rng.choice(np.array(choices, dtype=int))), int(c_v)))

    colors = _choose_colors(rng, len(source_slots))
    sources = [(r, c, col) for (r, c), col in zip(source_slots, colors)]
    grid = _build_grid(h, w, tracks, sources)
    return grid if _is_valid_generated(grid) else None


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    mode = int(rng.integers(0, 3))
    if mode == 0:
        return _gen_horizontal_only(rng)
    if mode == 1:
        return _gen_vertical_only(rng)
    return _gen_mixed(rng)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
