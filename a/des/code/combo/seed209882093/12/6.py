import random
from typing import Iterable

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


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
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _frame_bbox_from_points(points: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    local = np.zeros((h, w), dtype=bool)
    for r, c in points:
        local[r - r0, c - c0] = True
    frame = np.zeros((h, w), dtype=bool)
    frame[0, :] = True
    frame[-1, :] = True
    frame[:, 0] = True
    frame[:, -1] = True
    if np.array_equal(local, frame):
        return r0, c0, r1, c1
    return None


def _outside_moat_is_black(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    h, w = grid.shape
    r0, c0, r1, c1 = bbox
    for r in range(max(0, r0 - 1), min(h, r1 + 2)):
        for c in range(max(0, c0 - 1), min(w, c1 + 2)):
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            if grid[r, c] != BLACK:
                return False
    return True


def _detect_frames(grid: np.ndarray) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BLACK):
        for points in _components(grid == color):
            bbox = _frame_bbox_from_points(points)
            if bbox is None or not _outside_moat_is_black(grid, bbox):
                continue
            r0, c0, r1, c1 = bbox
            frames.append(
                {
                    "color": color,
                    "bbox": bbox,
                    "area": (r1 - r0 + 1) * (c1 - c0 + 1),
                }
            )
    frames.sort(key=lambda item: (-int(item["area"]), tuple(item["bbox"])))
    return frames


def _find_structure(grid: np.ndarray) -> tuple[dict[str, object], list[dict[str, object]]] | None:
    frames = _detect_frames(grid)
    if len(frames) < 3:
        return None
    large = frames[0]
    stamps = sorted(
        frames[1:],
        key=lambda item: (int(item["bbox"][0]), int(item["bbox"][1])),
    )
    return large, stamps


def _ring_cells(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for rr in range(r - 1, r + h + 1):
        for cc in range(c - 1, c + w + 1):
            if r <= rr < r + h and c <= cc < c + w:
                continue
            cells.append((rr, cc))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find each bordered stamp on the right, match its interior pattern inside the large left frame,
    and draw the stamp border color on the matched subregion's outer ring.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    found = _find_structure(g)
    if found is None:
        return out

    large, stamps = found
    lr0, lc0, lr1, lc1 = large["bbox"]

    for stamp in stamps:
        sr0, sc0, sr1, sc1 = stamp["bbox"]
        out[sr0 : sr1 + 1, sc0 : sc1 + 1] = BLACK

    for stamp in stamps:
        sr0, sc0, sr1, sc1 = stamp["bbox"]
        color = int(stamp["color"])
        pattern = g[sr0 + 1 : sr1, sc0 + 1 : sc1]
        sh, sw = pattern.shape

        match: tuple[int, int] | None = None
        count = 0
        for r in range(lr0 + 1, lr1 - sh + 1):
            for c in range(lc0 + 1, lc1 - sw + 1):
                if np.array_equal(g[r : r + sh, c : c + sw], pattern):
                    match = (r, c)
                    count += 1
        if count != 1 or match is None:
            continue

        mr, mc = match
        for rr, cc in _ring_cells(mr, mc, sh, sw):
            if lr0 < rr < lr1 and lc0 < cc < lc1:
                out[rr, cc] = color

    return out


def _make_pattern(h: int, w: int, colors: list[int], rng: random.Random) -> np.ndarray:
    grid = np.full((h, w), rng.choice(colors), dtype=int)

    for _ in range(rng.randint(4, 7)):
        color = rng.choice(colors)
        r0 = rng.randint(0, h - 2)
        c0 = rng.randint(0, w - 2)
        r1 = rng.randint(r0 + 1, min(h - 1, r0 + max(1, h // 2)))
        c1 = rng.randint(c0 + 1, min(w - 1, c0 + max(1, w // 2)))
        grid[r0 : r1 + 1, c0 : c1 + 1] = color

    for _ in range(rng.randint(4, 7)):
        color = rng.choice(colors)
        if rng.random() < 0.5:
            r = rng.randrange(h)
            c0 = rng.randrange(w)
            c1 = rng.randrange(c0, w)
            grid[r, c0 : c1 + 1] = color
        else:
            c = rng.randrange(w)
            r0 = rng.randrange(h)
            r1 = rng.randrange(r0, h)
            grid[r0 : r1 + 1, c] = color

    for _ in range(rng.randint(5, 10)):
        color = rng.choice(colors)
        r = rng.randrange(h)
        c = rng.randrange(w)
        grid[r, c] = color
        if r + 1 < h and c + 1 < w and rng.random() < 0.55:
            grid[r + 1, c + 1] = color
        if r + 1 < h and c - 1 >= 0 and rng.random() < 0.35:
            grid[r + 1, c - 1] = color
        if r + 1 < h and rng.random() < 0.25:
            grid[r + 1, c] = color
        if c + 1 < w and rng.random() < 0.25:
            grid[r, c + 1] = color

    for color in colors:
        if not np.any(grid == color):
            grid[rng.randrange(h), rng.randrange(w)] = color
    return grid


def _unique_windows(pattern: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = pattern.shape
    candidates: list[tuple[int, int, int, int]] = []
    for sh in range(3, min(5, h) + 1):
        for sw in range(3, min(5, w) + 1):
            buckets: dict[bytes, list[tuple[int, int]]] = {}
            for r in range(h - sh + 1):
                for c in range(w - sw + 1):
                    patch = pattern[r : r + sh, c : c + sw]
                    buckets.setdefault(patch.tobytes(), []).append((r, c))
            for positions in buckets.values():
                if len(positions) != 1:
                    continue
                r, c = positions[0]
                patch = pattern[r : r + sh, c : c + sw]
                if len(np.unique(patch)) >= 2:
                    candidates.append((r, c, sh, sw))
    return candidates


def _touches_edge(window: tuple[int, int, int, int], h: int, w: int) -> bool:
    r, c, sh, sw = window
    return r == 0 or c == 0 or r + sh == h or c + sw == w


def _windows_ring_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ring_a = set(_ring_cells(*a))
    ring_b = set(_ring_cells(*b))
    return bool(ring_a & ring_b)


def _choose_windows(
    candidates: list[tuple[int, int, int, int]],
    stamp_count: int,
    h: int,
    w: int,
    rng: random.Random,
) -> list[tuple[int, int, int, int]] | None:
    if len(candidates) < stamp_count:
        return None

    chosen: list[tuple[int, int, int, int]] = []
    pool = candidates[:]

    if rng.random() < 0.35:
        edge_candidates = [cand for cand in pool if _touches_edge(cand, h, w)]
        if edge_candidates:
            picked = rng.choice(edge_candidates)
            chosen.append(picked)
            pool.remove(picked)

    if not chosen:
        picked = rng.choice(pool)
        chosen.append(picked)
        pool.remove(picked)

    if stamp_count >= 2 and rng.random() < 0.35:
        overlap_candidates = [cand for cand in pool if _windows_ring_overlap(chosen[0], cand)]
        if overlap_candidates:
            picked = rng.choice(overlap_candidates)
            chosen.append(picked)
            pool.remove(picked)

    rng.shuffle(pool)
    for cand in pool:
        if len(chosen) == stamp_count:
            break
        chosen.append(cand)

    return chosen if len(chosen) == stamp_count else None


def _render_expected(
    grid: np.ndarray,
    large_bbox: tuple[int, int, int, int],
    stamps: list[dict[str, int]],
) -> np.ndarray:
    out = grid.copy()
    lr0, lc0, lr1, lc1 = large_bbox
    for stamp in stamps:
        sy, sx = stamp["y"], stamp["x"]
        sh, sw = stamp["h"], stamp["w"]
        out[sy : sy + sh + 2, sx : sx + sw + 2] = BLACK

    for stamp in sorted(stamps, key=lambda item: (item["y"], item["x"])):
        r = stamp["match_r"]
        c = stamp["match_c"]
        sh, sw = stamp["h"], stamp["w"]
        color = stamp["color"]
        for rr, cc in _ring_cells(r, c, sh, sw):
            if lr0 < rr < lr1 and lc0 < cc < lc1:
                out[rr, cc] = color
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    stamp_count = rng.randint(2, 4)
    max_pattern_colors = min(5, len(COLOR_POOL) - stamp_count)
    pattern_color_count = rng.randint(3, max_pattern_colors)

    shuffled = COLOR_POOL[:]
    rng.shuffle(shuffled)
    pattern_colors = shuffled[:pattern_color_count]
    stamp_colors = shuffled[pattern_color_count : pattern_color_count + stamp_count]

    inner_h = rng.randint(10, 15)
    inner_w = rng.randint(10, 15)
    pattern = _make_pattern(inner_h, inner_w, pattern_colors, rng)
    candidates = _unique_windows(pattern)
    chosen = _choose_windows(candidates, stamp_count, inner_h, inner_w, rng)
    if chosen is None:
        return None

    rng.shuffle(chosen)
    stamp_specs = [
        {
            "match_r": r,
            "match_c": c,
            "h": sh,
            "w": sw,
            "color": stamp_colors[idx],
        }
        for idx, (r, c, sh, sw) in enumerate(chosen)
    ]

    frame_h = inner_h + 2
    frame_w = inner_w + 2
    stamp_heights = [spec["h"] + 2 for spec in stamp_specs]
    stamp_widths = [spec["w"] + 2 for spec in stamp_specs]

    vertical_gaps = [rng.randint(2, 4) for _ in range(stamp_count - 1)]
    stamp_block_h = sum(stamp_heights) + sum(vertical_gaps)
    min_h = max(frame_h + 2, stamp_block_h + 2)
    grid_h = rng.randint(min_h, min_h + 5)

    large_y = rng.randint(1, grid_h - frame_h - 1)
    stamp_top = rng.randint(1, grid_h - stamp_block_h - 1)

    left_margin = rng.randint(1, 3)
    gap_x = rng.randint(4, 7)
    stamp_band_extra = rng.randint(1, 3)
    right_margin = rng.randint(1, 3)

    large_x = left_margin
    stamp_band_x = large_x + frame_w + gap_x
    stamp_band_w = max(stamp_widths) + stamp_band_extra
    grid_w = stamp_band_x + stamp_band_w + right_margin

    grid = np.zeros((grid_h, grid_w), dtype=int)
    grid[large_y, large_x : large_x + frame_w] = GREY
    grid[large_y + frame_h - 1, large_x : large_x + frame_w] = GREY
    grid[large_y : large_y + frame_h, large_x] = GREY
    grid[large_y : large_y + frame_h, large_x + frame_w - 1] = GREY
    grid[large_y + 1 : large_y + 1 + inner_h, large_x + 1 : large_x + 1 + inner_w] = pattern

    y = stamp_top
    for idx, spec in enumerate(stamp_specs):
        sh, sw = spec["h"], spec["w"]
        stamp_h = sh + 2
        stamp_w = sw + 2
        max_x = grid_w - right_margin - stamp_w
        min_x = stamp_band_x
        if max_x < min_x:
            return None
        x = rng.randint(min_x, max_x)
        spec["y"] = y
        spec["x"] = x

        grid[y, x : x + stamp_w] = spec["color"]
        grid[y + stamp_h - 1, x : x + stamp_w] = spec["color"]
        grid[y : y + stamp_h, x] = spec["color"]
        grid[y : y + stamp_h, x + stamp_w - 1] = spec["color"]
        patch = pattern[spec["match_r"] : spec["match_r"] + sh, spec["match_c"] : spec["match_c"] + sw]
        grid[y + 1 : y + 1 + sh, x + 1 : x + 1 + sw] = patch

        if idx + 1 < stamp_count:
            y += stamp_h + vertical_gaps[idx]

    large_bbox = (large_y, large_x, large_y + frame_h - 1, large_x + frame_w - 1)
    expected = _render_expected(
        grid,
        large_bbox,
        [
            {
                "y": int(spec["y"]),
                "x": int(spec["x"]),
                "h": int(spec["h"]),
                "w": int(spec["w"]),
                "color": int(spec["color"]),
                "match_r": int(large_y + 1 + spec["match_r"]),
                "match_c": int(large_x + 1 + spec["match_c"]),
            }
            for spec in stamp_specs
        ],
    )

    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(500):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exceeded retries")
