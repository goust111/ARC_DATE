from __future__ import annotations

import itertools
import random
from collections import Counter, deque

import numpy as np


BG = 0
MARKER = 5
OBJECT_COLORS = (1, 2, 3, 4, 7)
DIMENSIONS = (
    (2, 3),
    (3, 2),
    (2, 4),
    (4, 2),
    (2, 5),
    (5, 2),
    (3, 4),
    (4, 3),
    (3, 5),
    (5, 3),
)
RECT_COUNT_RANGE = (3, 5)
GRID_H_RANGE = (12, 18)
GRID_W_RANGE = (15, 22)
PLACEMENT_GAP = 1
GENERATION_ATTEMPTS = 600


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def find_components(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _normalized_dims(height: int, width: int, marker_count: int, mode: str = "parity") -> tuple[int, int]:
    short, long = min(height, width), max(height, width)
    if mode == "parity":
        vertical = marker_count % 2 == 1
    elif mode == "opposite":
        vertical = marker_count % 2 == 0
    elif mode == "horizontal":
        vertical = False
    elif mode == "vertical":
        vertical = True
    else:
        raise ValueError(f"unknown normalization mode: {mode}")
    return (long, short) if vertical else (short, long)


def _render_output(rects: list[dict], bg: int, order_mode: str = "count", norm_mode: str = "parity") -> np.ndarray:
    prepared = []
    for rect in rects:
        out_h, out_w = _normalized_dims(rect["h"], rect["w"], rect["k"], mode=norm_mode)
        prepared.append({**rect, "out_h": out_h, "out_w": out_w})

    if order_mode == "count":
        prepared.sort(key=lambda rect: (rect["k"], rect["c0"], rect["r0"], rect["color"]))
    elif order_mode == "spatial":
        prepared.sort(key=lambda rect: (rect["c0"], rect["r0"], rect["k"], rect["color"]))
    else:
        raise ValueError(f"unknown order mode: {order_mode}")

    out_h = max(rect["out_h"] for rect in prepared)
    out_w = sum(rect["out_w"] for rect in prepared) + len(prepared) - 1
    out = np.full((out_h, out_w), bg, dtype=int)

    cursor = 0
    for rect in prepared:
        rh, rw = rect["out_h"], rect["out_w"]
        out[out_h - rh : out_h, cursor : cursor + rw] = rect["color"]
        cursor += rw + 1
    return out


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    bg = get_bg(grid)
    rects = []
    for coords in find_components(grid, bg):
        r0, c0, r1, c1 = bbox_coords(coords)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert int(coords.shape[0]) == height * width
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        colors = {int(x) for x in np.unique(patch) if int(x) != bg}
        fill_colors = colors - {MARKER}
        assert len(fill_colors) == 1
        rects.append(
            {
                "r0": r0,
                "c0": c0,
                "h": height,
                "w": width,
                "k": int(np.count_nonzero(patch == MARKER)),
                "color": int(next(iter(fill_colors))),
            }
        )
    assert rects
    return rects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count each rectangle's gray markers, normalize its orientation from that count, then pack the rectangles by increasing count.

    Input structure:
    - A mostly black background contains 3 to 5 separated solid-color rectangles.
    - Each rectangle may have several gray marker pixels inside it; the gray cells stay 4-connected to the rectangle and belong to the same object.
    - Every object is a full axis-aligned rectangle when its gray markers are treated as covered foreground cells.

    Rule:
    - Detect each 4-connected non-black object and count how many gray marker pixels it contains.
    - Erase the markers by restoring the rectangle to its single non-gray foreground color.
    - If the marker count is odd, normalize that rectangle to vertical orientation; if it is even, normalize it to horizontal orientation.
    - Sort the normalized rectangles by marker count ascending and place them left-to-right with one black column between neighbors, bottom-aligned on the output canvas.

    Key insight:
    - The same gray-marker count drives two independent downstream choices at once: odd/even decides whether width and height are swapped, while the count magnitude decides the final left-to-right packing order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    return _render_output(_extract_rectangles(g), bg=get_bg(g), order_mode="count", norm_mode="parity")


def _count_choices(n_rects: int) -> list[tuple[int, ...]]:
    combos = []
    for combo in itertools.combinations(range(5), n_rects):
        if not any(k % 2 == 0 for k in combo):
            continue
        if not any(k % 2 == 1 for k in combo):
            continue
        combos.append(combo)
    return combos


def _sample_counts(n_rects: int) -> list[int]:
    choices = _count_choices(n_rects)
    with_zero = [combo for combo in choices if 0 in combo]
    pool = with_zero if with_zero and random.random() < 0.7 else choices
    counts = list(random.choice(pool))
    random.shuffle(counts)
    return counts


def _sample_specs() -> list[dict] | None:
    n_rects = random.randint(*RECT_COUNT_RANGE)
    for _ in range(200):
        dims = [random.choice(DIMENSIONS) for _ in range(n_rects)]
        counts = _sample_counts(n_rects)
        colors = random.sample(list(OBJECT_COLORS), n_rects)
        specs = [
            {"h": h, "w": w, "k": k, "color": color}
            for (h, w), k, color in zip(dims, counts, colors)
        ]
        has_horizontal = any(spec["w"] > spec["h"] for spec in specs)
        has_vertical = any(spec["h"] > spec["w"] for spec in specs)
        odd_flips_to_vertical = any(spec["k"] % 2 == 1 and spec["w"] > spec["h"] for spec in specs)
        even_flips_to_horizontal = any(spec["k"] % 2 == 0 and spec["h"] > spec["w"] for spec in specs)
        if has_horizontal and has_vertical and odd_flips_to_vertical and even_flips_to_horizontal:
            return specs
    return None


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = PLACEMENT_GAP) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _place_specs(specs: list[dict], height: int, width: int) -> list[dict] | None:
    order = sorted(range(len(specs)), key=lambda idx: specs[idx]["h"] * specs[idx]["w"], reverse=True)
    placed: list[dict | None] = [None] * len(specs)
    boxes: list[tuple[int, int, int, int]] = []
    for idx in order:
        spec = specs[idx]
        placed_here = False
        for _ in range(250):
            r0 = random.randint(0, height - spec["h"])
            c0 = random.randint(0, width - spec["w"])
            box = (r0, c0, r0 + spec["h"] - 1, c0 + spec["w"] - 1)
            if any(_boxes_too_close(box, other) for other in boxes):
                continue
            placed[idx] = {**spec, "r0": r0, "c0": c0}
            boxes.append(box)
            placed_here = True
            break
        if not placed_here:
            return None
    return [spec for spec in placed if spec is not None]


def _build_grid(specs: list[dict], height: int, width: int) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for spec in specs:
        r0, c0 = spec["r0"], spec["c0"]
        h, w = spec["h"], spec["w"]
        grid[r0 : r0 + h, c0 : c0 + w] = spec["color"]
        cells = [(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)]
        for r, c in random.sample(cells, spec["k"]):
            grid[r, c] = MARKER
    return grid


def _semantic_checks(specs: list[dict], grid: np.ndarray) -> bool:
    if len({spec["k"] for spec in specs}) != len(specs):
        return False
    if not any(spec["k"] % 2 == 0 for spec in specs):
        return False
    if not any(spec["k"] % 2 == 1 for spec in specs):
        return False

    by_x = sorted(specs, key=lambda spec: (spec["c0"], spec["r0"]))
    by_k = sorted(specs, key=lambda spec: spec["k"])
    if [spec["k"] for spec in by_x] == [spec["k"] for spec in by_k]:
        return False

    expected = _render_output(specs, bg=BG, order_mode="count", norm_mode="parity")
    if np.array_equal(expected, _render_output(specs, bg=BG, order_mode="spatial", norm_mode="parity")):
        return False
    if np.array_equal(expected, _render_output(specs, bg=BG, order_mode="count", norm_mode="opposite")):
        return False
    if np.array_equal(expected, _render_output(specs, bg=BG, order_mode="count", norm_mode="horizontal")):
        return False

    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    if int(np.count_nonzero(grid == BG)) <= int(np.count_nonzero(grid != BG)):
        return False

    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    specs = _sample_specs()
    if specs is None:
        return None

    min_h = max(max(spec["h"] for spec in specs) + 4, GRID_H_RANGE[0])
    min_w = max(max(spec["w"] for spec in specs) + 6, GRID_W_RANGE[0])
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None

    height = random.randint(min_h, GRID_H_RANGE[1])
    width = random.randint(min_w, GRID_W_RANGE[1])
    placed = _place_specs(specs, height, width)
    if placed is None:
        return None

    grid = _build_grid(placed, height, width)
    if not _semantic_checks(placed, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
