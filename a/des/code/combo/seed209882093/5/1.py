from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

TOP = 0
RIGHT = 1
BOTTOM = 2
LEFT = 3

MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(pts)

    return components


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in points]
    cols = [c for _r, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _frame_mask_from_bbox(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    border: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))
    return border


def _direction_from_seed(sr: int, sc: int, r0: int, c0: int, r1: int, c1: int) -> int:
    gaps = (
        sr - r0 - 1,
        c1 - sc - 1,
        r1 - sr - 1,
        sc - c0 - 1,
    )
    best = min(gaps)
    for direction, gap in enumerate(gaps):
        if gap == best:
            return direction
    raise AssertionError("unreachable")


def _destination(sr: int, sc: int, r0: int, c0: int, r1: int, c1: int, direction: int) -> tuple[int, int]:
    if direction == TOP:
        return r0 - 1, sc
    if direction == RIGHT:
        return sr, c1 + 1
    if direction == BOTTOM:
        return r1 + 1, sc
    if direction == LEFT:
        return sr, c0 - 1
    raise AssertionError(f"unknown direction {direction}")


def _erase_edge(out: np.ndarray, r0: int, c0: int, r1: int, c1: int, direction: int) -> None:
    if direction == TOP:
        out[r0, c0 : c1 + 1] = BLACK
    elif direction == RIGHT:
        out[r0 : r1 + 1, c1] = BLACK
    elif direction == BOTTOM:
        out[r1, c0 : c1 + 1] = BLACK
    elif direction == LEFT:
        out[r0 : r1 + 1, c0] = BLACK
    else:
        raise AssertionError(f"unknown direction {direction}")


def _extract_frames(g: np.ndarray) -> list[tuple[int, int, int, int, int, int, int]]:
    frames: list[tuple[int, int, int, int, int, int, int]] = []
    for pts in _components(g == GREY):
        r0, c0, r1, c1 = _bbox(pts)
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "grey frames must enclose an interior"
        assert set(pts) == _frame_mask_from_bbox(r0, c0, r1, c1), "grey component must be a hollow rectangle"

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_locs = np.argwhere((interior != BLACK) & (interior != GREY))
        assert seed_locs.shape[0] == 1, "each frame must contain exactly one colored seed"
        sr = r0 + 1 + int(seed_locs[0, 0])
        sc = c0 + 1 + int(seed_locs[0, 1])
        seed_color = int(g[sr, sc])

        cleaned = interior.copy()
        cleaned[int(seed_locs[0, 0]), int(seed_locs[0, 1])] = BLACK
        assert np.all(cleaned == BLACK), "frame interiors must be empty except for the seed"
        frames.append((r0, c0, r1, c1, sr, sc, seed_color))

    frames.sort()
    assert frames, "expected at least one grey frame"
    seed_count = int(np.count_nonzero((g != BLACK) & (g != GREY)))
    assert seed_count == len(frames), "all colored pixels must be frame seeds"
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Open the nearest fence edge for each seed and move the seed just outside that edge.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = g.shape

    for r0, c0, r1, c1, sr, sc, color in _extract_frames(g):
        direction = _direction_from_seed(sr, sc, r0, c0, r1, c1)
        tr, tc = _destination(sr, sc, r0, c0, r1, c1, direction)
        assert 0 <= tr < h and 0 <= tc < w, "moved seed must stay inside the grid"
        assert int(out[tr, tc]) == BLACK, "destination outside the opened edge must be empty"

        out[sr, sc] = BLACK
        _erase_edge(out, r0, c0, r1, c1, direction)
        out[tr, tc] = color

    return out


def _seed_candidates(frame_h: int, frame_w: int, direction: int, require_tie: bool) -> list[tuple[int, int]]:
    interior_h = frame_h - 2
    interior_w = frame_w - 2
    candidates: list[tuple[int, int]] = []

    for ir in range(interior_h):
        for ic in range(interior_w):
            gaps = (ir, interior_w - 1 - ic, interior_h - 1 - ir, ic)
            best = min(gaps)
            chosen = next(idx for idx, gap in enumerate(gaps) if gap == best)
            has_tie = sum(gap == best for gap in gaps) > 1
            if chosen == direction and has_tie == require_tie:
                candidates.append((ir, ic))

    return candidates


def _sample_frame_spec(direction: int, color: int, require_tie: bool) -> dict[str, int] | None:
    for _ in range(120):
        frame_h = random.randint(4, 10)
        frame_w = random.randint(4, 10)
        candidates = _seed_candidates(frame_h, frame_w, direction, require_tie)
        if not candidates:
            continue
        ir, ic = random.choice(candidates)
        return {
            "frame_h": frame_h,
            "frame_w": frame_w,
            "seed_r": ir,
            "seed_c": ic,
            "direction": direction,
            "color": color,
        }
    return None


def _assigned_directions(count: int) -> list[int]:
    directions = random.sample([TOP, RIGHT, BOTTOM, LEFT], min(count, 4))
    while len(directions) < count:
        directions.append(random.choice([TOP, RIGHT, BOTTOM, LEFT]))
    random.shuffle(directions)
    return directions


def _choose_specs(count: int) -> list[dict[str, int]] | None:
    colors = random.sample(SEED_COLORS, count)
    specs: list[dict[str, int]] = []
    tie_indices: list[int] = []

    for idx, (direction, color) in enumerate(zip(_assigned_directions(count), colors)):
        can_tie = direction != LEFT
        require_tie = can_tie and random.random() < 0.35
        spec = _sample_frame_spec(direction, color, require_tie)
        if spec is None:
            return None
        specs.append(spec)
        if require_tie:
            tie_indices.append(idx)

    if count >= 3 and not tie_indices and random.random() < 0.5:
        candidates = [idx for idx, spec in enumerate(specs) if spec["direction"] != LEFT]
        if candidates:
            repl_idx = random.choice(candidates)
            new_spec = _sample_frame_spec(specs[repl_idx]["direction"], specs[repl_idx]["color"], True)
            if new_spec is None:
                return None
            specs[repl_idx] = new_spec

    return specs


def _choose_canvas(specs: list[dict[str, int]]) -> tuple[int, int]:
    reserved_area = sum((spec["frame_h"] + 2) * (spec["frame_w"] + 2) for spec in specs)
    base = max(12, int(np.ceil(np.sqrt(reserved_area * random.uniform(1.35, 1.75)))))
    max_frame_h = max(spec["frame_h"] for spec in specs)
    max_frame_w = max(spec["frame_w"] for spec in specs)

    if random.random() < 0.65:
        rows = random.randint(base, base + 5)
        cols = random.randint(base, base + 5)
    else:
        stretch = random.randint(3, 8)
        if random.random() < 0.5:
            rows = random.randint(base, base + 3)
            cols = random.randint(base + stretch, base + stretch + 4)
        else:
            rows = random.randint(base + stretch, base + stretch + 4)
            cols = random.randint(base, base + 3)

    rows = max(rows, max_frame_h + 2)
    cols = max(cols, max_frame_w + 2)
    return rows, cols


def _place_specs(rows: int, cols: int, specs: list[dict[str, int]]) -> list[dict[str, int]] | None:
    occupied = np.zeros((rows, cols), dtype=bool)
    placed: list[dict[str, int] | None] = [None] * len(specs)
    order = sorted(range(len(specs)), key=lambda idx: specs[idx]["frame_h"] * specs[idx]["frame_w"], reverse=True)

    for idx in order:
        spec = specs[idx]
        frame_h = spec["frame_h"]
        frame_w = spec["frame_w"]
        positions: list[tuple[int, int]] = []

        for r0 in range(1, rows - frame_h):
            for c0 in range(1, cols - frame_w):
                if occupied[r0 - 1 : r0 + frame_h + 1, c0 - 1 : c0 + frame_w + 1].any():
                    continue
                positions.append((r0, c0))

        if not positions:
            return None

        r0, c0 = random.choice(positions)
        occupied[r0 - 1 : r0 + frame_h + 1, c0 - 1 : c0 + frame_w + 1] = True

        placed_spec = dict(spec)
        placed_spec["r0"] = r0
        placed_spec["c0"] = c0
        placed[idx] = placed_spec

    return [spec for spec in placed if spec is not None]


def _draw_frame(grid: np.ndarray, r0: int, c0: int, frame_h: int, frame_w: int) -> None:
    r1 = r0 + frame_h - 1
    c1 = c0 + frame_w - 1
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY


def _render_input(rows: int, cols: int, placed_specs: list[dict[str, int]]) -> np.ndarray:
    grid = np.full((rows, cols), BLACK, dtype=int)
    for spec in placed_specs:
        r0 = spec["r0"]
        c0 = spec["c0"]
        frame_h = spec["frame_h"]
        frame_w = spec["frame_w"]
        _draw_frame(grid, r0, c0, frame_h, frame_w)
        sr = r0 + 1 + spec["seed_r"]
        sc = c0 + 1 + spec["seed_c"]
        grid[sr, sc] = spec["color"]
    return grid


def _expected_output(inp: np.ndarray, placed_specs: list[dict[str, int]]) -> np.ndarray:
    out = inp.copy()
    for spec in placed_specs:
        r0 = spec["r0"]
        c0 = spec["c0"]
        r1 = r0 + spec["frame_h"] - 1
        c1 = c0 + spec["frame_w"] - 1
        sr = r0 + 1 + spec["seed_r"]
        sc = c0 + 1 + spec["seed_c"]
        direction = spec["direction"]
        color = spec["color"]

        out[sr, sc] = BLACK
        _erase_edge(out, r0, c0, r1, c1, direction)
        tr, tc = _destination(sr, sc, r0, c0, r1, c1, direction)
        assert int(out[tr, tc]) == BLACK
        out[tr, tc] = color

    return out


def _gen_attempt() -> np.ndarray | None:
    count = random.choice([2, 2, 3, 3, 4, 4, 5])
    specs = _choose_specs(count)
    if specs is None:
        return None

    rows, cols = _choose_canvas(specs)
    placed_specs = _place_specs(rows, cols, specs)
    if placed_specs is None:
        return None

    inp = _render_input(rows, cols, placed_specs)
    expected = _expected_output(inp, placed_specs)

    try:
        actual = main(inp)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
