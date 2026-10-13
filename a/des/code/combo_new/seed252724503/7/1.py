from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME = 5
MAX_SIDE = 30
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
QUADRANTS = ("TL", "TR", "BR", "BL")
ROTATION_K = {"TL": 0, "TR": 3, "BR": 2, "BL": 1}
GEN_RETRIES = 300

PATTERN_SHAPES = [
    np.array([[1, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=int),
    np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 0], [0, 0, 0]], dtype=int),
    np.array([[0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=int),
    np.array([[1, 0, 0], [0, 1, 1], [0, 0, 1]], dtype=int),
]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for comp in _components(grid == FRAME):
        r0, c0, r1, c1 = _bbox(comp)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height < 3 or width < 3:
            continue
        if len(comp) != 2 * (height + width) - 4:
            continue
        if not np.all(grid[r0, c0 : c1 + 1] == FRAME):
            continue
        if not np.all(grid[r1, c0 : c1 + 1] == FRAME):
            continue
        if not np.all(grid[r0 : r1 + 1, c0] == FRAME):
            continue
        if not np.all(grid[r0 : r1 + 1, c1] == FRAME):
            continue
        if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == FRAME):
            continue
        frames.append((int(r0), int(c0), int(r1), int(c1)))
    frames.sort()
    return frames


def _interior_mask(shape: tuple[int, int], frames: list[tuple[int, int, int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r0, c0, r1, c1 in frames:
        mask[r0 + 1 : r1, c0 + 1 : c1] = True
    return mask


def _extract_legend(
    grid: np.ndarray, frames: list[tuple[int, int, int, int]]
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    h, w = grid.shape
    blocked = _interior_mask(grid.shape, frames)
    mapping: dict[int, np.ndarray] = {}
    legend_mask = np.zeros((h, w), dtype=bool)

    for r in range(1, h - 1):
        for c in range(0, w - 3):
            if blocked[r, c]:
                continue
            seed = int(grid[r, c])
            if seed in (BG, FRAME):
                continue
            sub = grid[r - 1 : r + 2, c : c + 4]
            if int(sub[1, 0]) != seed:
                continue
            if np.any(sub[[0, 2], 0] != BG):
                continue
            if c > 0 and int(grid[r, c - 1]) != BG:
                continue
            pattern = sub[:, 1:4]
            colors = sorted({int(v) for v in pattern.ravel() if int(v) != BG})
            if len(colors) != 1:
                continue
            pattern_color = colors[0]
            if pattern_color in (BG, FRAME, seed):
                continue
            if np.count_nonzero(pattern == pattern_color) < 3:
                continue
            if seed in mapping:
                raise AssertionError("legend seed colors must be unique")
            cleaned = np.where(pattern == pattern_color, pattern_color, BG).astype(int)
            mapping[seed] = cleaned
            legend_mask[r, c] = True
            legend_mask[r - 1 : r + 2, c + 1 : c + 4] |= cleaned != BG

    return mapping, legend_mask


def _frame_seed(
    grid: np.ndarray, frame: tuple[int, int, int, int]
) -> tuple[int, int, int]:
    r0, c0, r1, c1 = frame
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    seeds = np.argwhere((interior != BG) & (interior != FRAME))
    assert len(seeds) == 1, "each frame interior must contain exactly one seed pixel"
    sr = int(seeds[0, 0]) + r0 + 1
    sc = int(seeds[0, 1]) + c0 + 1
    return sr, sc, int(grid[sr, sc])


def _quadrant_from_frame(
    frame: tuple[int, int, int, int], seed_pos: tuple[int, int]
) -> str:
    r0, c0, r1, c1 = frame
    sr, sc = seed_pos
    center_r = (r0 + 1 + r1 - 1) / 2.0
    center_c = (c0 + 1 + c1 - 1) / 2.0
    if sr == center_r or sc == center_c:
        raise AssertionError("seed may not sit on a frame center line")
    top = sr < center_r
    left = sc < center_c
    if top and left:
        return "TL"
    if top and not left:
        return "TR"
    if not top and left:
        return "BL"
    return "BR"


def _stamp_pattern(
    out: np.ndarray,
    frame: tuple[int, int, int, int],
    seed_pos: tuple[int, int],
    pattern: np.ndarray,
) -> None:
    r0, c0, r1, c1 = frame
    sr, sc = seed_pos
    for pr, pc in np.argwhere(pattern != BG):
        tr = sr + int(pr) - 1
        tc = sc + int(pc) - 1
        if not (r0 < tr < r1 and c0 < tc < c1):
            continue
        if out[tr, tc] != BG:
            continue
        out[tr, tc] = int(pattern[pr, pc])


def _solve(grid: np.ndarray) -> np.ndarray:
    frames = _find_frames(grid)
    assert frames, "the puzzle must contain at least one grey frame"
    legend_map, legend_mask = _extract_legend(grid, frames)
    assert legend_map, "the puzzle must contain a parseable legend"

    out = np.asarray(grid, dtype=int).copy()
    out[legend_mask] = BG

    for frame in frames:
        sr, sc, seed_color = _frame_seed(grid, frame)
        assert seed_color in legend_map, "every frame seed color must exist in the legend"
        quadrant = _quadrant_from_frame(frame, (sr, sc))
        rotated = np.rot90(legend_map[seed_color], ROTATION_K[quadrant])
        _stamp_pattern(out, frame, (sr, sc), rotated)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each framed region with the legend pattern whose rotation is chosen by the seed pixel's quadrant.

    Input structure:
    - The background is black.
    - The top-left legend is made of several rows; each row has one seed-color pixel and an adjacent 3x3 asymmetric pattern drawn in a different color on black.
    - Elsewhere, several one-pixel-thick grey rectangular frames each enclose exactly one non-black, non-grey seed pixel.

    Rule:
    - Read the legend as a mapping from seed color to its 3x3 pattern in the pattern's original orientation.
    - For each grey frame, find its interior seed pixel and locate the matching legend pattern.
    - Determine whether the seed lies in the frame interior's top-left, top-right, bottom-left, or bottom-right quadrant.
    - Rotate the pattern by 0, 90, 270, or 180 degrees clockwise according to that quadrant, center it on the seed, and paint only black cells inside the frame.
    - Leave the grey frame and the seed pixel unchanged, and erase the legend to black.

    Key insight:
    - The legend supplies the color-to-pattern lookup, but the seed's position inside its own enclosure is the hidden control signal that selects the pattern's final orientation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    return _solve(g)


def _colored_pattern(shape: np.ndarray, color: int) -> np.ndarray:
    return np.where(shape != 0, color, BG).astype(int)


def _pattern_rotations_distinct(shape: np.ndarray) -> bool:
    rots = {tuple(np.rot90(shape, k).ravel().tolist()) for k in range(4)}
    return len(rots) == 4


assert all(_pattern_rotations_distinct(shape) for shape in PATTERN_SHAPES)


def _place_legend(
    canvas: np.ndarray,
    rng: random.Random,
    legend_specs: list[dict[str, object]],
) -> tuple[list[dict[str, object]], tuple[int, int, int, int]]:
    top = rng.randint(1, 2)
    left = rng.randint(1, 2)
    row_top = top
    records: list[dict[str, object]] = []

    for i, spec in enumerate(legend_specs):
        seed_color = int(spec["seed_color"])
        pattern = np.asarray(spec["pattern"], dtype=int)
        seed_r = row_top + 1
        seed_c = left
        canvas[seed_r, seed_c] = seed_color
        pr0 = row_top
        pc0 = left + 1
        block = canvas[pr0 : pr0 + 3, pc0 : pc0 + 3]
        block[pattern != BG] = pattern[pattern != BG]
        records.append({"seed_color": seed_color, "pattern": pattern.copy(), "seed_pos": (seed_r, seed_c)})
        row_top += 3
        if i < len(legend_specs) - 1:
            row_top += rng.randint(1, 2)

    bottom = row_top - 1
    right = left + 3
    return records, (top, left, bottom, right)


def _can_place(occupied: np.ndarray, r0: int, c0: int, height: int, width: int, pad: int = 1) -> bool:
    h, w = occupied.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h, r0 + height + pad)
    cc1 = min(w, c0 + width + pad)
    return not np.any(occupied[rr0:rr1, cc0:cc1])


def _mark_occupied(occupied: np.ndarray, r0: int, c0: int, height: int, width: int, pad: int = 1) -> None:
    h, w = occupied.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h, r0 + height + pad)
    cc1 = min(w, c0 + width + pad)
    occupied[rr0:rr1, cc0:cc1] = True


def _pick_seed_position(
    rng: random.Random,
    frame: tuple[int, int, int, int],
    quadrant: str,
) -> tuple[int, int]:
    r0, c0, r1, c1 = frame
    center_r = (r0 + 1 + r1 - 1) / 2.0
    center_c = (c0 + 1 + c1 - 1) / 2.0
    rows = list(range(r0 + 2, r1 - 1))
    cols = list(range(c0 + 2, c1 - 1))
    if quadrant.startswith("T"):
        rows = [r for r in rows if r < center_r]
    else:
        rows = [r for r in rows if r > center_r]
    if quadrant.endswith("L"):
        cols = [c for c in cols if c < center_c]
    else:
        cols = [c for c in cols if c > center_c]
    assert rows and cols
    return rng.choice(rows), rng.choice(cols)


def _draw_frame(canvas: np.ndarray, frame: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = frame
    canvas[r0, c0 : c1 + 1] = FRAME
    canvas[r1, c0 : c1 + 1] = FRAME
    canvas[r0 : r1 + 1, c0] = FRAME
    canvas[r0 : r1 + 1, c1] = FRAME


def _crop_records(
    records: list[dict[str, object]], top: int, left: int
) -> list[dict[str, object]]:
    cropped: list[dict[str, object]] = []
    for rec in records:
        new_rec = dict(rec)
        if "seed_pos" in rec:
            sr, sc = rec["seed_pos"]
            new_rec["seed_pos"] = (int(sr) - top, int(sc) - left)
        if "bbox" in rec:
            r0, c0, r1, c1 = rec["bbox"]
            new_rec["bbox"] = (int(r0) - top, int(c0) - left, int(r1) - top, int(c1) - left)
        cropped.append(new_rec)
    return cropped


def _render_expected_from_spec(
    grid: np.ndarray,
    legend_records: list[dict[str, object]],
    frame_records: list[dict[str, object]],
) -> np.ndarray:
    out = grid.copy()
    legend_map = {int(rec["seed_color"]): np.asarray(rec["pattern"], dtype=int) for rec in legend_records}

    for rec in legend_records:
        sr, sc = rec["seed_pos"]
        sr = int(sr)
        sc = int(sc)
        out[sr, sc] = BG
        pattern = np.asarray(rec["pattern"], dtype=int)
        for pr, pc in np.argwhere(pattern != BG):
            out[sr - 1 + int(pr), sc + 1 + int(pc)] = BG

    for rec in frame_records:
        bbox = rec["bbox"]
        sr, sc = rec["seed_pos"]
        seed_color = int(rec["seed_color"])
        quadrant = _quadrant_from_frame(bbox, (int(sr), int(sc)))
        rotated = np.rot90(legend_map[seed_color], ROTATION_K[quadrant])
        _stamp_pattern(out, bbox, (int(sr), int(sc)), rotated)

    return out


def _build_attempt(rng: random.Random) -> np.ndarray | None:
    canvas = np.zeros((MAX_SIDE, MAX_SIDE), dtype=int)
    occupied = np.zeros((MAX_SIDE, MAX_SIDE), dtype=bool)

    n_legend = rng.randint(2, 4)
    colors = rng.sample(COLOR_POOL, 2 * n_legend)
    seed_colors = colors[:n_legend]
    pattern_colors = colors[n_legend:]
    shapes = rng.sample(PATTERN_SHAPES, n_legend)

    legend_specs = []
    for seed_color, pattern_color, shape in zip(seed_colors, pattern_colors, shapes, strict=False):
        legend_specs.append(
            {"seed_color": int(seed_color), "pattern": _colored_pattern(np.asarray(shape, dtype=int), int(pattern_color))}
        )
    rng.shuffle(legend_specs)

    legend_records, legend_bbox = _place_legend(canvas, rng, legend_specs)
    lr0, lc0, lr1, lc1 = legend_bbox
    _mark_occupied(occupied, lr0, lc0, lr1 - lr0 + 1, lc1 - lc0 + 1, pad=1)

    min_frames = min(6, max(4, n_legend + 1))
    n_frames = rng.randint(min_frames, 6)

    quadrants = list(QUADRANTS)
    rng.shuffle(quadrants)
    while len(quadrants) < n_frames:
        quadrants.append(rng.choice(QUADRANTS))
    rng.shuffle(quadrants)

    frame_seed_colors = seed_colors[:]
    rng.shuffle(frame_seed_colors)
    while len(frame_seed_colors) < n_frames:
        frame_seed_colors.append(rng.choice(seed_colors))
    rng.shuffle(frame_seed_colors)

    size_max = 11 - max(0, n_frames - 4)
    frame_records: list[dict[str, object]] = []
    for quadrant, seed_color in zip(quadrants, frame_seed_colors, strict=False):
        placed = False
        for _ in range(250):
            height = rng.randint(7, size_max)
            width = rng.randint(7, size_max)
            r0 = rng.randint(0, MAX_SIDE - height)
            c0 = rng.randint(0, MAX_SIDE - width)
            if not _can_place(occupied, r0, c0, height, width, pad=1):
                continue
            frame = (r0, c0, r0 + height - 1, c0 + width - 1)
            seed_pos = _pick_seed_position(rng, frame, quadrant)
            _draw_frame(canvas, frame)
            canvas[seed_pos] = int(seed_color)
            frame_records.append({"bbox": frame, "seed_pos": seed_pos, "seed_color": int(seed_color)})
            _mark_occupied(occupied, r0, c0, height, width, pad=1)
            placed = True
            break
        if not placed:
            return None

    used = np.argwhere(canvas != BG)
    if used.size == 0:
        return None
    r0 = int(used[:, 0].min())
    c0 = int(used[:, 1].min())
    r1 = int(used[:, 0].max())
    c1 = int(used[:, 1].max())
    top = max(0, r0 - rng.randint(0, 1))
    left = max(0, c0 - rng.randint(0, 1))
    bottom = min(MAX_SIDE - 1, r1 + rng.randint(1, 2))
    right = min(MAX_SIDE - 1, c1 + rng.randint(1, 2))
    grid = canvas[top : bottom + 1, left : right + 1].copy()

    cropped_legend = _crop_records(legend_records, top, left)
    cropped_frames = _crop_records(frame_records, top, left)
    expected = _render_expected_from_spec(grid, cropped_legend, cropped_frames)

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None
    if len(_find_frames(grid)) != n_frames:
        return None
    legend_map, _ = _extract_legend(grid, _find_frames(grid))
    if len(legend_map) != n_legend:
        return None
    if sorted(legend_map) != sorted(seed_colors):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _build_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
