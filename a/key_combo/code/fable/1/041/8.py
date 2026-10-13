from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _detect_mirrors(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    h, w = map(int, grid.shape)
    candidates: list[tuple[int, tuple[int, ...], tuple[int, ...]]] = []
    for color in sorted(set(map(int, grid.ravel())) - {int(bg)}):
        row_ids = tuple(r for r in range(h) if bool(np.all(grid[r] == color)))
        col_ids = tuple(c for c in range(w) if bool(np.all(grid[:, c] == color)))
        if len(row_ids) + len(col_ids) not in (1, 2):
            continue
        mask = np.zeros_like(grid, dtype=bool)
        if row_ids:
            mask[list(row_ids), :] = True
        if col_ids:
            mask[:, list(col_ids)] = True
        if bool(np.array_equal(grid == color, mask)):
            candidates.append((int(color), row_ids, col_ids))
    assert len(candidates) == 1, "expected exactly one mirror-line color"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect every foreground sprite across the full-row/full-column mirror lines until closure.

    Input structure:
    - The background is the dominant color.
    - One color forms one or two complete straight mirror lines spanning the whole grid; each mirror is horizontal or vertical.
    - The remaining non-background pixels are foreground sprites whose colors differ from the mirror color.

    Rule:
    - Keep the mirror lines unchanged.
    - For every non-mirror foreground pixel, paint its reflection across every mirror line in the same color.
    - Newly created pixels reflect again across all mirrors; iterate until no new in-bounds pixels appear.

    Key insight:
    - Two mirrors do not mean "reflect once in each"; repeated reflections generate the full orbit, which for parallel mirrors becomes periodic translated copies.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    mirror_color, mirror_rows, mirror_cols = _detect_mirrors(g, bg)

    out = g.copy()
    queue: deque[tuple[int, int, int]] = deque()
    for r_raw, c_raw in np.argwhere((g != bg) & (g != mirror_color)):
        r = int(r_raw)
        c = int(c_raw)
        queue.append((r, c, int(g[r, c])))

    while queue:
        r, c, color = queue.popleft()
        for mr in mirror_rows:
            rr = 2 * int(mr) - int(r)
            cc = int(c)
            if not _in_bounds(g.shape, rr, cc):
                continue
            cell = int(out[rr, cc])
            if cell == bg:
                out[rr, cc] = color
                queue.append((rr, cc, color))
            else:
                assert cell in (mirror_color, color), "conflicting reflected colors"
        for mc in mirror_cols:
            rr = int(r)
            cc = 2 * int(mc) - int(c)
            if not _in_bounds(g.shape, rr, cc):
                continue
            cell = int(out[rr, cc])
            if cell == bg:
                out[rr, cc] = color
                queue.append((rr, cc, color))
            else:
                assert cell in (mirror_color, color), "conflicting reflected colors"

    return out


def _shape(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


_BASE_SHAPES = [
    _shape(["11"]),
    _shape(["111"]),
    _shape(["11", "10"]),
    _shape(["11", "01"]),
    _shape(["111", "100"]),
    _shape(["111", "001"]),
    _shape(["110", "011"]),
    _shape(["10", "11", "01"]),
    _shape(["111", "010"]),
]


def _shape_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[bytes] = set()
    for flip in (False, True):
        cur = np.fliplr(mask) if flip else mask
        for k in range(4):
            turned = np.rot90(cur, k)
            key = turned.shape[0].to_bytes(1, "little") + turned.shape[1].to_bytes(1, "little") + turned.tobytes()
            if key not in seen:
                seen.add(key)
                variants.append(turned.copy())
    return variants


_ALL_SHAPES = [variant for base in _BASE_SHAPES for variant in _shape_variants(base)]
_ASYMMETRIC_SHAPES = [mask for mask in _ALL_SHAPES if not np.array_equal(mask, np.fliplr(mask)) or not np.array_equal(mask, np.flipud(mask))]


def _axis_closure(start: int, mirrors: tuple[int, ...], limit: int) -> tuple[int, ...]:
    seen = {int(start)}
    queue = [int(start)]
    while queue:
        cur = queue.pop()
        for mirror in mirrors:
            nxt = 2 * int(mirror) - int(cur)
            if 0 <= nxt < limit and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return tuple(sorted(seen))


def _orbit_pixels(
    coords: list[tuple[int, int]],
    mirror_rows: tuple[int, ...],
    mirror_cols: tuple[int, ...],
    shape: tuple[int, int],
) -> set[tuple[int, int]]:
    h, w = map(int, shape)
    row_cache: dict[int, tuple[int, ...]] = {}
    col_cache: dict[int, tuple[int, ...]] = {}
    out: set[tuple[int, int]] = set()
    for r, c in coords:
        if r not in row_cache:
            row_cache[r] = _axis_closure(r, mirror_rows, h)
        if c not in col_cache:
            col_cache[c] = _axis_closure(c, mirror_cols, w)
        for rr in row_cache[r]:
            for cc in col_cache[c]:
                out.add((int(rr), int(cc)))
    return out


def _mirror_grid(
    h: int,
    w: int,
    bg: int,
    mirror_color: int,
    mirror_rows: tuple[int, ...],
    mirror_cols: tuple[int, ...],
) -> np.ndarray:
    grid = np.full((h, w), int(bg), dtype=int)
    if mirror_rows:
        grid[list(mirror_rows), :] = int(mirror_color)
    if mirror_cols:
        grid[:, list(mirror_cols)] = int(mirror_color)
    return grid


def _render_expected(
    h: int,
    w: int,
    bg: int,
    mirror_color: int,
    mirror_rows: tuple[int, ...],
    mirror_cols: tuple[int, ...],
    objects: list[dict],
) -> np.ndarray:
    out = _mirror_grid(h, w, bg, mirror_color, mirror_rows, mirror_cols)
    for obj in objects:
        for r, c in obj["orbit"]:
            cell = int(out[r, c])
            assert cell in (bg, obj["color"])
            out[r, c] = int(obj["color"])
    return out


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


def _seed_box(
    config: str,
    h: int,
    w: int,
    mirror_rows: tuple[int, ...],
    mirror_cols: tuple[int, ...],
) -> tuple[int, int, int, int]:
    if config == "single_h":
        row = mirror_rows[0]
        if random.random() < 0.5:
            return 0, int(row), 0, w
        return int(row) + 1, h, 0, w
    if config == "single_v":
        col = mirror_cols[0]
        if random.random() < 0.5:
            return 0, h, 0, int(col)
        return 0, h, int(col) + 1, w
    if config == "double_h":
        return int(mirror_rows[0]) + 1, int(mirror_rows[1]), 0, w
    if config == "double_v":
        return 0, h, int(mirror_cols[0]) + 1, int(mirror_cols[1])
    row = mirror_rows[0]
    col = mirror_cols[0]
    quadrants = [
        (0, int(row), 0, int(col)),
        (0, int(row), int(col) + 1, w),
        (int(row) + 1, h, 0, int(col)),
        (int(row) + 1, h, int(col) + 1, w),
    ]
    return random.choice(quadrants)


def _choose_mirrors(config: str) -> tuple[int, int, tuple[int, ...], tuple[int, ...]]:
    if config == "single_h":
        h = random.randint(9, 18)
        w = random.randint(8, 18)
        return h, w, (random.randint(2, h - 3),), ()
    if config == "single_v":
        h = random.randint(8, 18)
        w = random.randint(9, 18)
        return h, w, (), (random.randint(2, w - 3),)
    if config == "double_h":
        h = random.randint(12, 20)
        w = random.randint(8, 18)
        gap = random.randint(2, 4)
        top = random.randint(2, h - gap - 3)
        return h, w, (top, top + gap), ()
    if config == "double_v":
        h = random.randint(8, 18)
        w = random.randint(12, 20)
        gap = random.randint(2, 4)
        left = random.randint(2, w - gap - 3)
        return h, w, (), (left, left + gap)
    h = random.randint(10, 18)
    w = random.randint(10, 18)
    return h, w, (random.randint(2, h - 3),), (random.randint(2, w - 3),)


def _pick_shape(max_h: int, max_w: int, *, prefer_asymmetric: bool) -> np.ndarray | None:
    pool = _ASYMMETRIC_SHAPES if prefer_asymmetric else _ALL_SHAPES
    candidates = [mask for mask in pool if mask.shape[0] <= max_h and mask.shape[1] <= max_w]
    if not candidates and prefer_asymmetric:
        candidates = [mask for mask in _ALL_SHAPES if mask.shape[0] <= max_h and mask.shape[1] <= max_w]
    if not candidates:
        return None
    return random.choice(candidates).copy()


def _top_left_positions(
    box: tuple[int, int, int, int],
    mask: np.ndarray,
) -> list[tuple[int, int]]:
    r0, r1, c0, c1 = box
    max_r = int(r1) - int(mask.shape[0])
    max_c = int(c1) - int(mask.shape[1])
    if max_r < r0 or max_c < c0:
        return []
    return [(r, c) for r in range(int(r0), max_r + 1) for c in range(int(c0), max_c + 1)]


def _place_objects(
    config: str,
    h: int,
    w: int,
    mirror_rows: tuple[int, ...],
    mirror_cols: tuple[int, ...],
    colors: list[int],
) -> list[dict] | None:
    box = _seed_box(config, h, w, mirror_rows, mirror_cols)
    objects: list[dict] = []
    input_occ = set()
    input_pad = set()
    output_occ = set()
    min_h = int(box[1] - box[0])
    min_w = int(box[3] - box[2])
    want = random.choice([1, 2] if config.startswith("double_") else [1, 2, 3])

    for idx in range(want):
        prefer_asymmetric = idx == 0
        found = False
        for _ in range(80):
            mask = _pick_shape(min_h, min_w, prefer_asymmetric=prefer_asymmetric)
            if mask is None:
                return None
            positions = _top_left_positions(box, mask)
            random.shuffle(positions)
            for top, left in positions:
                coords = [
                    (top + int(r), left + int(c))
                    for r, c in np.argwhere(mask)
                ]
                coord_set = set(coords)
                if coord_set & input_occ:
                    continue
                if coord_set & input_pad:
                    continue
                orbit = _orbit_pixels(coords, mirror_rows, mirror_cols, (h, w))
                if orbit & output_occ:
                    continue
                axis_sizes = []
                if mirror_rows:
                    axis_sizes.extend(len(_axis_closure(r, mirror_rows, h)) for r, _ in coords)
                if mirror_cols:
                    axis_sizes.extend(len(_axis_closure(c, mirror_cols, w)) for _, c in coords)
                if config in ("double_h", "double_v") and max(axis_sizes, default=1) < 4:
                    continue
                objects.append({"color": colors[idx], "coords": coords, "orbit": orbit})
                input_occ |= coord_set
                output_occ |= orbit
                for r, c in coords:
                    input_pad.add((r, c))
                    for rr, cc in _neighbors8(r, c):
                        if 0 <= rr < h and 0 <= cc < w:
                            input_pad.add((rr, cc))
                found = True
                break
            if found:
                break
        if not found:
            return None

    return objects


def _valid_counts(inp: np.ndarray, bg: int, mirror_color: int) -> bool:
    counts = Counter(map(int, inp.ravel()))
    bg_count = counts[int(bg)]
    if any(bg_count <= count for color, count in counts.items() if color != int(bg)):
        return False
    return int(counts[int(mirror_color)]) >= max(inp.shape)


def _attempt() -> np.ndarray | None:
    config = random.choice(["single_h", "single_v", "double_h", "double_v", "cross"])
    h, w, mirror_rows, mirror_cols = _choose_mirrors(config)
    palette = list(range(10))
    bg = random.choice(palette)
    remaining = [color for color in palette if color != bg]
    mirror_color = random.choice(remaining)
    remaining = [color for color in remaining if color != mirror_color]

    object_count = 3 if not config.startswith("double_") else 2
    colors = random.sample(remaining, object_count)
    objects = _place_objects(config, h, w, mirror_rows, mirror_cols, colors)
    if objects is None:
        return None

    inp = _mirror_grid(h, w, bg, mirror_color, mirror_rows, mirror_cols)
    for obj in objects:
        for r, c in obj["coords"]:
            assert int(inp[r, c]) == bg
            inp[r, c] = int(obj["color"])

    expected = _render_expected(h, w, bg, mirror_color, mirror_rows, mirror_cols, objects)
    if np.array_equal(inp, expected):
        return None
    if np.count_nonzero(expected != bg) >= int(0.65 * h * w):
        return None
    if not _valid_counts(inp, bg, mirror_color):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-closure puzzle")
