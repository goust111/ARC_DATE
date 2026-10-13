from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def get_bg(grid: np.ndarray) -> int:
    if np.any(grid == 0):
        return 0
    counts = Counter(int(v) for v in grid.flat)
    return int(max(counts.items(), key=lambda kv: kv[1])[0])


def components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def bbox_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_frame_component(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> bool:
    r0, c0, r1, c1 = bbox_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    sub = grid[r0 : r1 + 1, c0 : c1 + 1] == color
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.all(sub[border]) and not np.any(sub[~border]))


def _find_frames(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    frames: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        for coords in components_mask(g == color):
            if not _is_frame_component(g, coords, color):
                continue
            r0, c0, r1, c1 = bbox_coords(coords)
            frames.append(
                {
                    "color": int(color),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
                    "inner": g[r0 + 1 : r1, c0 + 1 : c1].copy(),
                }
            )
    frames.sort(key=lambda item: (int(item["area"]), item["bbox"]))
    return frames


def _compose_layers(patches: list[np.ndarray]) -> np.ndarray:
    assert patches
    canvas_h, canvas_w = patches[-1].shape
    out = np.zeros((canvas_h, canvas_w), dtype=int)
    for patch in patches:
        ph, pw = patch.shape
        assert ph <= canvas_h and pw <= canvas_w
        r0 = (canvas_h - ph) // 2
        c0 = (canvas_w - pw) // 2
        window = out[r0 : r0 + ph, c0 : c0 + pw]
        mask = patch != 0
        window[mask] = patch[mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract each complete rectangular frame's interior, sort frames by area, and
    overlay the interiors centered on the largest interior canvas.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = _find_frames(g)
    assert frames, "expected at least one complete frame"

    patches = [np.asarray(frame["inner"], dtype=int) for frame in frames]
    return _compose_layers(patches)


COLOR_VALUES = tuple(range(1, 10))
FRAME_COUNT_RANGE = (2, 5)
GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
INNER_MIN = 3
INNER_MAX = 8
PLACEMENT_MARGIN = 1
PLACEMENT_GAP = 1
GEN_RETRIES = 300
PATTERN_RETRIES = 120


def _overlap_with_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ah, aw = a
    br0, bc0, bh, bw = b
    ar1 = ar0 + ah - 1
    ac1 = ac0 + aw - 1
    br1 = br0 + bh - 1
    bc1 = bc0 + bw - 1
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _sample_dims(frame_count: int, rng: random.Random) -> list[tuple[int, int]] | None:
    if frame_count >= 5:
        largest_h = rng.randint(5, 7)
        largest_w = rng.randint(5, 7)
    elif frame_count == 4:
        largest_h = rng.randint(5, 8)
        largest_w = rng.randint(5, 8)
    else:
        largest_h = rng.randint(5, INNER_MAX)
        largest_w = rng.randint(5, INNER_MAX)

    largest = (largest_h, largest_w)
    used_areas = {(largest_h + 2) * (largest_w + 2)}
    candidates = [
        (h, w)
        for h in range(INNER_MIN, largest_h + 1)
        for w in range(INNER_MIN, largest_w + 1)
        if (h, w) != largest and (h + 2) * (w + 2) not in used_areas
    ]
    rng.shuffle(candidates)
    dims = [largest]
    for h, w in candidates:
        area = (h + 2) * (w + 2)
        if area in used_areas:
            continue
        dims.append((int(h), int(w)))
        used_areas.add(int(area))
        if len(dims) == frame_count:
            break
    if len(dims) != frame_count:
        return None
    dims.sort(key=lambda hw: (hw[0] + 2) * (hw[1] + 2))
    return dims


def _paint_rectangle(patch: np.ndarray, color: int, rng: random.Random) -> None:
    h, w = patch.shape
    rh = rng.randint(1, h)
    rw = rng.randint(1, w)
    r0 = rng.randint(0, h - rh)
    c0 = rng.randint(0, w - rw)
    patch[r0 : r0 + rh, c0 : c0 + rw] = color


def _paint_segment(patch: np.ndarray, color: int, rng: random.Random) -> None:
    h, w = patch.shape
    if rng.random() < 0.5:
        r = rng.randint(0, h - 1)
        c0 = rng.randint(0, w - 1)
        c1 = rng.randint(c0, w - 1)
        patch[r, c0 : c1 + 1] = color
    else:
        c = rng.randint(0, w - 1)
        r0 = rng.randint(0, h - 1)
        r1 = rng.randint(r0, h - 1)
        patch[r0 : r1 + 1, c] = color


def _paint_l_shape(patch: np.ndarray, color: int, rng: random.Random) -> None:
    h, w = patch.shape
    r0 = rng.randint(0, h - 1)
    c0 = rng.randint(0, w - 1)
    r1 = rng.randint(r0, h - 1)
    c1 = rng.randint(c0, w - 1)
    patch[r0 : r1 + 1, c0] = color
    patch[r0, c0 : c1 + 1] = color


def _paint_diagonal(patch: np.ndarray, color: int, rng: random.Random) -> None:
    h, w = patch.shape
    span = rng.randint(2, max(2, min(h, w)))
    r0 = rng.randint(0, h - span)
    c0 = rng.randint(0, w - span)
    if rng.random() < 0.5:
        for k in range(span):
            patch[r0 + k, c0 + k] = color
    else:
        for k in range(span):
            patch[r0 + k, c0 + span - 1 - k] = color


def _contains_frame_like_component(patch: np.ndarray) -> bool:
    for color in sorted(int(v) for v in np.unique(patch) if int(v) != 0):
        for coords in components_mask(patch == color):
            if _is_frame_component(patch, coords, color):
                return True
    return False


def _valid_pattern(patch: np.ndarray) -> bool:
    colors = [int(v) for v in np.unique(patch) if int(v) != 0]
    filled = int(np.count_nonzero(patch))
    area = int(patch.size)
    if len(colors) < 2:
        return False
    if filled < max(3, area // 4):
        return False
    if filled >= area:
        return False
    return not _contains_frame_like_component(patch)


def _make_pattern(h: int, w: int, palette: list[int], rng: random.Random) -> np.ndarray | None:
    for _ in range(PATTERN_RETRIES):
        patch = np.zeros((h, w), dtype=int)
        colors = rng.sample(palette, k=rng.randint(2, min(4, len(palette))))
        family = rng.choice(["blocks", "lines", "diagonal", "mixed"])

        if family == "blocks":
            for _ in range(rng.randint(3, 5)):
                _paint_rectangle(patch, rng.choice(colors), rng)
            _paint_segment(patch, rng.choice(colors), rng)
        elif family == "lines":
            for _ in range(rng.randint(3, 6)):
                _paint_segment(patch, rng.choice(colors), rng)
            _paint_l_shape(patch, rng.choice(colors), rng)
        elif family == "diagonal":
            for color in colors:
                _paint_diagonal(patch, color, rng)
            _paint_rectangle(patch, rng.choice(colors), rng)
        else:
            _paint_rectangle(patch, colors[0], rng)
            _paint_rectangle(patch, colors[1], rng)
            _paint_segment(patch, rng.choice(colors), rng)
            _paint_l_shape(patch, rng.choice(colors), rng)
            if min(h, w) >= 3:
                _paint_diagonal(patch, rng.choice(colors), rng)

        if area := patch.size:
            if area >= 9 and rng.random() < 0.7:
                hole_h = rng.randint(1, max(1, h // 2))
                hole_w = rng.randint(1, max(1, w // 2))
                hole_r = rng.randint(0, h - hole_h)
                hole_c = rng.randint(0, w - hole_w)
                patch[hole_r : hole_r + hole_h, hole_c : hole_c + hole_w] = 0

        if _valid_pattern(patch):
            return patch
    return None


def _force_visible_overlay(patterns: list[np.ndarray], rng: random.Random) -> bool:
    largest = np.asarray(patterns[-1], dtype=int).copy()
    canvas_h, canvas_w = largest.shape

    for idx in range(len(patterns) - 2, -1, -1):
        patch = np.asarray(patterns[idx], dtype=int)
        ph, pw = patch.shape
        top = (canvas_h - ph) // 2
        left = (canvas_w - pw) // 2
        candidates = [(r, c) for r in range(ph) for c in range(pw) if int(patch[r, c]) != 0]
        rng.shuffle(candidates)
        for r, c in candidates:
            rr = top + r
            cc = left + c
            if int(largest[rr, cc]) == 0:
                patterns[-1] = largest
                return True
            old = int(largest[rr, cc])
            largest[rr, cc] = 0
            if _valid_pattern(largest):
                patterns[-1] = largest
                return True
            largest[rr, cc] = old
    return False


def _place_frames(
    dims: list[tuple[int, int]],
    frame_colors: list[int],
    patterns: list[np.ndarray],
    grid_h: int,
    grid_w: int,
    rng: random.Random,
) -> list[dict[str, object]] | None:
    order = sorted(range(len(dims)), key=lambda i: (dims[i][0] + 2) * (dims[i][1] + 2), reverse=True)
    placed_boxes: list[tuple[int, int, int, int]] = []
    specs: list[dict[str, object] | None] = [None] * len(dims)

    for idx in order:
        inner_h, inner_w = dims[idx]
        outer_h, outer_w = inner_h + 2, inner_w + 2
        positions = [
            (r, c)
            for r in range(PLACEMENT_MARGIN, grid_h - outer_h - PLACEMENT_MARGIN + 1)
            for c in range(PLACEMENT_MARGIN, grid_w - outer_w - PLACEMENT_MARGIN + 1)
        ]
        rng.shuffle(positions)

        chosen: tuple[int, int] | None = None
        for r0, c0 in positions:
            box = (int(r0), int(c0), int(outer_h), int(outer_w))
            if any(_overlap_with_gap(box, other, PLACEMENT_GAP) for other in placed_boxes):
                continue
            chosen = (int(r0), int(c0))
            placed_boxes.append(box)
            specs[idx] = {
                "frame_color": int(frame_colors[idx]),
                "bbox": (int(r0), int(c0), int(r0 + outer_h - 1), int(c0 + outer_w - 1)),
                "inner": np.asarray(patterns[idx], dtype=int).copy(),
            }
            break
        if chosen is None:
            return None

    return [spec for spec in specs if spec is not None]


def _render_grid(specs: list[dict[str, object]], grid_h: int, grid_w: int) -> np.ndarray:
    grid = np.zeros((grid_h, grid_w), dtype=int)
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        frame_color = int(spec["frame_color"])
        inner = np.asarray(spec["inner"], dtype=int)
        grid[r0, c0 : c1 + 1] = frame_color
        grid[r1, c0 : c1 + 1] = frame_color
        grid[r0 : r1 + 1, c0] = frame_color
        grid[r0 : r1 + 1, c1] = frame_color
        grid[r0 + 1 : r1, c0 + 1 : c1] = inner
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    frame_count = rng.randint(*FRAME_COUNT_RANGE)
    dims = _sample_dims(frame_count, rng)
    if dims is None:
        return None

    max_outer_h = max(h + 2 for h, _ in dims)
    max_outer_w = max(w + 2 for _, w in dims)
    grid_h_min = max(GRID_H_RANGE[0], max_outer_h + 2 * PLACEMENT_MARGIN + 1, 9 + frame_count * 2)
    grid_w_min = max(GRID_W_RANGE[0], max_outer_w + 2 * PLACEMENT_MARGIN + 1, 9 + frame_count * 2)
    if grid_h_min > GRID_H_RANGE[1] or grid_w_min > GRID_W_RANGE[1]:
        return None

    grid_h = rng.randint(grid_h_min, GRID_H_RANGE[1])
    grid_w = rng.randint(grid_w_min, GRID_W_RANGE[1])

    frame_colors = rng.sample(list(COLOR_VALUES), k=frame_count)
    pattern_palette = [color for color in COLOR_VALUES if color not in frame_colors]

    patterns: list[np.ndarray] = []
    for inner_h, inner_w in dims:
        pattern = _make_pattern(inner_h, inner_w, pattern_palette, rng)
        if pattern is None:
            return None
        patterns.append(pattern)
    if not _force_visible_overlay(patterns, rng):
        return None

    specs = _place_frames(dims, frame_colors, patterns, grid_h, grid_w, rng)
    if specs is None:
        return None

    grid = _render_grid(specs, grid_h, grid_w)
    frames = _find_frames(grid)
    if len(frames) != frame_count:
        return None

    expected = _compose_layers([np.asarray(spec["inner"], dtype=int) for spec in sorted(specs, key=lambda s: (s["bbox"][2] - s["bbox"][0] + 1) * (s["bbox"][3] - s["bbox"][1] + 1))])
    if np.array_equal(expected, np.asarray(patterns[-1], dtype=int)):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
