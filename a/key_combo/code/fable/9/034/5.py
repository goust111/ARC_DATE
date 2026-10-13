from __future__ import annotations

import random
from collections import Counter

import numpy as np


TAIL_COLOR = 5
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GEN_ATTEMPTS = 600


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _rotate(dr: int, dc: int, turn: str) -> tuple[int, int]:
    if turn == "cw":
        return dc, -dr
    if turn == "ccw":
        return -dc, dr
    if turn == "none":
        return dr, dc
    raise ValueError(f"unknown turn mode: {turn}")


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(comp)
    return parts


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _trace_path(
    shape: tuple[int, int],
    seed: tuple[int, int],
    direction: tuple[int, int],
    heights: list[int],
    *,
    blocked: set[tuple[int, int]] | None = None,
    row_max: int | None = None,
    turn: str = "cw",
    forbid_revisit: bool = True,
) -> list[tuple[int, int]] | None:
    H, W = shape
    blocked = set() if blocked is None else set(blocked)
    r, c = seed
    dr, dc = direction
    used = {seed}
    path: list[tuple[int, int]] = []
    for steps in heights:
        for _ in range(int(steps)):
            r += dr
            c += dc
            if not (0 <= r < H and 0 <= c < W):
                return None
            if row_max is not None and r > row_max:
                return None
            if (r, c) in blocked:
                return None
            if forbid_revisit and (r, c) in used:
                return None
            path.append((r, c))
            used.add((r, c))
        dr, dc = _rotate(dr, dc, turn)
    return path


def _render_path(grid: np.ndarray, seed_color: int, path: list[tuple[int, int]]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for r, c in path:
        out[r, c] = int(seed_color)
    return out


def _parse_scene(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = _get_bg(g)

    baseline_candidates = [
        (r, int(g[r, 0]))
        for r in range(H)
        if int(g[r, 0]) != bg and np.all(g[r, :] == g[r, 0])
    ]
    assert baseline_candidates, "expected a full-width baseline row"
    base_r, base_color = max(baseline_candidates, key=lambda item: item[0])

    bar_candidate = None
    for color in sorted(int(v) for v in np.unique(g[:base_r, :]) if int(v) not in {bg, base_color}):
        comps = _components(g[:base_r, :] == int(color))
        if len(comps) < 3:
            continue
        bars = []
        valid = True
        for comp in comps:
            r0, c0, r1, c1 = _bbox(comp)
            if r1 != base_r - 1:
                valid = False
                break
            if len(comp) != (r1 - r0 + 1) * (c1 - c0 + 1):
                valid = False
                break
            bars.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "height": int(r1 - r0 + 1),
                    "left": int(c0),
                    "width": int(c1 - c0 + 1),
                }
            )
        if valid:
            assert bar_candidate is None, "bar color should be uniquely identifiable"
            bar_candidate = (int(color), bars)
    assert bar_candidate is not None, "expected one valid bar set"
    bar_color, bars = bar_candidate
    bars.sort(key=lambda bar: (int(bar["left"]), int(bar["width"])))
    heights = [int(bar["height"]) for bar in bars]

    special = [
        (r, c, int(g[r, c]))
        for r in range(H)
        for c in range(W)
        if int(g[r, c]) not in {bg, base_color, bar_color}
    ]
    assert len(special) == 2, "expected exactly one seed and one tail pixel"
    a, b = special
    assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1, "seed and tail must be edge-adjacent"
    if a[2] == TAIL_COLOR and b[2] != TAIL_COLOR:
        tail = (a[0], a[1])
        seed = (b[0], b[1])
        seed_color = int(b[2])
    elif b[2] == TAIL_COLOR and a[2] != TAIL_COLOR:
        tail = (b[0], b[1])
        seed = (a[0], a[1])
        seed_color = int(a[2])
    else:
        raise AssertionError("tail pixel should be uniquely gray")

    direction = (seed[0] - tail[0], seed[1] - tail[1])
    assert direction in DIRS, "tail must define one cardinal orientation"

    return {
        "bg": int(bg),
        "base_r": int(base_r),
        "base_color": int(base_color),
        "bar_color": int(bar_color),
        "bars": bars,
        "heights": heights,
        "seed": seed,
        "tail": tail,
        "seed_color": int(seed_color),
        "direction": direction,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the bottom histogram as step lengths and draw a clockwise-turning polyline from the seed.

    Input structure:
    - A full-width baseline occupies the bottom row, and several separated same-color vertical bars stand directly above it.
    - Elsewhere in the empty upper area, one colored seed pixel is edge-adjacent to one gray tail pixel.
    - The tail-to-seed direction gives the starting orientation.

    Rule:
    - Read the bar heights from left to right, counting only the cells above the baseline row.
    - Starting at the seed, move forward by the first height and color each traversed cell with the seed color.
    - After every segment, turn 90 degrees clockwise and use the next bar height; keep the baseline, bars, seed, and gray tail unchanged.

    Key insight:
    - The histogram is a read-only instruction strip: bar order gives segment order, bar height gives segment length, and the seed-tail pair fixes the initial heading.
    """
    g = np.asarray(grid, dtype=int)
    scene = _parse_scene(g)
    path = _trace_path(
        g.shape,
        scene["seed"],
        scene["direction"],
        scene["heights"],
        blocked={scene["tail"]},
        turn="cw",
        forbid_revisit=False,
    )
    assert path is not None, "polyline should stay inside the canvas"

    out = np.array(g, dtype=int, copy=True)
    bg = int(scene["bg"])
    seed_color = int(scene["seed_color"])
    protected = {scene["tail"], scene["seed"]}
    for r, c in path:
        if (r, c) in protected:
            continue
        assert int(g[r, c]) in {bg, seed_color}, "constructed polyline must not hit the fixed objects"
        out[r, c] = seed_color
    return out


def _sample_gaps(total_extra: int, slots: int) -> list[int]:
    probs = np.full(slots, 1.0 / slots)
    return [int(x) for x in np.random.multinomial(int(total_extra), probs)]


def _sample_bar_lefts(widths: list[int], W: int) -> list[int] | None:
    needed = sum(widths) + len(widths) - 1
    extra = W - needed
    if extra < 0:
        return None
    gaps = _sample_gaps(extra, len(widths) + 1)
    lefts: list[int] = []
    c = gaps[0]
    for i, width in enumerate(widths):
        lefts.append(int(c))
        c += int(width)
        if i < len(widths) - 1:
            c += 1 + gaps[i + 1]
    return lefts


def _build_grid(
    H: int,
    W: int,
    bg: int,
    base_color: int,
    bar_color: int,
    seed_color: int,
    heights: list[int],
    widths: list[int],
    lefts: list[int],
    seed: tuple[int, int],
    tail: tuple[int, int],
) -> np.ndarray:
    g = np.full((H, W), int(bg), dtype=int)
    g[H - 1, :] = int(base_color)
    for left, width, height in zip(lefts, widths, heights):
        g[H - 1 - int(height) : H - 1, int(left) : int(left) + int(width)] = int(bar_color)
    g[tail] = TAIL_COLOR
    g[seed] = int(seed_color)
    return g


def _semantic_checks(grid: np.ndarray, meta: dict[str, object]) -> bool:
    try:
        scene = _parse_scene(grid)
    except Exception:
        return False

    if scene["heights"] != meta["heights"]:
        return False
    if scene["direction"] != meta["direction"]:
        return False
    if scene["seed"] != meta["seed"] or scene["tail"] != meta["tail"]:
        return False

    expected_path = _trace_path(
        grid.shape,
        scene["seed"],
        scene["direction"],
        scene["heights"],
        blocked={scene["tail"]},
        turn="cw",
        forbid_revisit=True,
    )
    if expected_path is None:
        return False

    actual = main(grid)
    expected = _render_path(grid, int(scene["seed_color"]), expected_path)
    if np.array_equal(actual, grid) or not np.array_equal(actual, expected):
        return False

    alternatives: list[np.ndarray | None] = []

    alt_path = _trace_path(
        grid.shape,
        scene["seed"],
        scene["direction"],
        list(reversed(scene["heights"])),
        blocked={scene["tail"]},
        turn="cw",
        forbid_revisit=True,
    )
    alternatives.append(None if alt_path is None else _render_path(grid, int(scene["seed_color"]), alt_path))

    alt_path = _trace_path(
        grid.shape,
        scene["seed"],
        scene["direction"],
        scene["heights"],
        blocked={scene["tail"]},
        turn="ccw",
        forbid_revisit=True,
    )
    alternatives.append(None if alt_path is None else _render_path(grid, int(scene["seed_color"]), alt_path))

    dr, dc = scene["direction"]
    alt_path = _trace_path(
        grid.shape,
        scene["seed"],
        (-dr, -dc),
        scene["heights"],
        blocked={scene["tail"]},
        turn="cw",
        forbid_revisit=True,
    )
    alternatives.append(None if alt_path is None else _render_path(grid, int(scene["seed_color"]), alt_path))

    alt_path = _trace_path(
        grid.shape,
        scene["seed"],
        scene["direction"],
        scene["heights"],
        blocked={scene["tail"]},
        turn="none",
        forbid_revisit=True,
    )
    alternatives.append(None if alt_path is None else _render_path(grid, int(scene["seed_color"]), alt_path))

    if any(alt is not None and np.array_equal(alt, expected) for alt in alternatives):
        return False

    if len(set(scene["heights"])) < 2:
        return False
    if sum(scene["heights"]) < 8:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([0, 8])
    palette = [c for c in range(10) if c not in {bg, TAIL_COLOR}]
    base_color, bar_color, seed_color = random.sample(palette, 3)

    k = random.randint(3, 6)
    widths = [random.choice([1, 1, 2, 2, 3]) for _ in range(k)]
    min_w = sum(widths) + k - 1
    if min_w > 28:
        return None

    W = random.randint(max(14, min_w), 28)
    H = random.randint(16, 28)

    max_height = min(6, H - 6)
    if max_height < 2:
        return None
    heights = [random.randint(1, max_height) for _ in range(k)]
    if len(set(heights)) < 2:
        return None

    upper_row_max = H - max(heights) - 3
    if upper_row_max < 3:
        return None

    direction = random.choice(DIRS)
    dr, dc = direction
    seeds: list[tuple[int, int, list[tuple[int, int]]]] = []
    for r in range(0, upper_row_max + 1):
        for c in range(W):
            tail = (r - dr, c - dc)
            if not (0 <= tail[0] <= upper_row_max and 0 <= tail[1] < W):
                continue
            path = _trace_path(
                (H, W),
                (r, c),
                direction,
                heights,
                blocked={tail},
                row_max=upper_row_max,
                turn="cw",
                forbid_revisit=True,
            )
            if path is not None:
                seeds.append((r, c, path))
    if not seeds:
        return None

    seed_r, seed_c, _path = random.choice(seeds)
    seed = (int(seed_r), int(seed_c))
    tail = (int(seed_r - dr), int(seed_c - dc))

    lefts = _sample_bar_lefts(widths, W)
    if lefts is None:
        return None

    grid = _build_grid(
        H,
        W,
        bg,
        base_color,
        bar_color,
        seed_color,
        heights,
        widths,
        lefts,
        seed,
        tail,
    )

    meta = {
        "heights": heights,
        "direction": direction,
        "seed": seed,
        "tail": tail,
    }
    return grid if _semantic_checks(grid, meta) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid histogram-polyline input")
