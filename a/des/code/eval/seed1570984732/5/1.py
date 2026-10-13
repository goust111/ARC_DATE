from __future__ import annotations

from collections import Counter, deque

import numpy as np


QUADRANTS = ("UL", "UR", "LL", "LR")
QUAD_DIR = {
    "UL": (-1, -1),
    "UR": (-1, 1),
    "LL": (1, -1),
    "LR": (1, 1),
}
GENERATE_ATTEMPTS = 400
FRAME_ATTEMPTS = 300


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _colors_present(grid: np.ndarray, exclude: set[int]) -> list[int]:
    return [int(v) for v in np.unique(grid) if int(v) not in exclude]


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[tuple[int, int]] = []
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _point_quadrant(
    r: int, c: int, bbox: tuple[int, int, int, int]
) -> str | None:
    r0, c0, r1, c1 = bbox
    cr = (r0 + r1) / 2.0
    cc = (c0 + c1) / 2.0
    if r == cr or c == cc:
        return None
    top = r < cr
    left = c < cc
    if top and left:
        return "UL"
    if top:
        return "UR"
    if left:
        return "LL"
    return "LR"


def _segment_quadrant(
    coords: list[tuple[int, int]], bbox: tuple[int, int, int, int]
) -> str | None:
    counts: Counter[str] = Counter()
    for r, c in coords:
        quad = _point_quadrant(int(r), int(c), bbox)
        if quad is not None:
            counts[quad] += 1
    if not counts:
        return None
    best_quad, best_count = counts.most_common(1)[0]
    if sum(count == best_count for count in counts.values()) != 1:
        return None
    return best_quad


def _frame_border_set(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    border: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0 + 1, r1):
        border.add((r, c0))
        border.add((r, c1))
    return border


def _extract_frames(grid: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    frames: list[dict[str, object]] = []
    for color in _colors_present(g, exclude={bg}):
        pts = [(int(r), int(c)) for r, c in np.argwhere(g == color)]
        if len(pts) < 4:
            continue
        r0, c0, r1, c1 = _bbox(pts)
        if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 4:
            continue
        border = _frame_border_set(r0, c0, r1, c1)
        if any((r, c) not in border for r, c in pts):
            continue
        if any(int(g[r, c]) not in (bg, color) for r, c in border):
            continue
        if len(pts) == len(border):
            continue
        if np.any(g[r0 + 1 : r1, c0 + 1 : c1] == color):
            continue

        local = g[r0 : r1 + 1, c0 : c1 + 1]
        seg_comps = _components(local == color)
        if not (1 <= len(seg_comps) <= 4):
            continue

        segments: list[dict[str, object]] = []
        segment_quads: set[str] = set()
        ok = True
        frame_bbox = (r0, c0, r1, c1)
        for comp in seg_comps:
            abs_coords = [(r + r0, c + c0) for r, c in comp]
            quad = _segment_quadrant(abs_coords, frame_bbox)
            if quad is None or quad in segment_quads:
                ok = False
                break
            segments.append(
                {
                    "coords": abs_coords,
                    "quadrant": quad,
                    "size": len(abs_coords),
                }
            )
            segment_quads.add(quad)
        if not ok:
            continue

        interior_mask = local[1:-1, 1:-1] != bg
        seed_comps = _components(interior_mask)
        seeds: dict[str, dict[str, object]] = {}
        for comp in seed_comps:
            if len(comp) != 1:
                ok = False
                break
            sr, sc = comp[0]
            sr += r0 + 1
            sc += c0 + 1
            seed_color = int(g[sr, sc])
            if seed_color == color:
                ok = False
                break
            quad = _point_quadrant(sr, sc, frame_bbox)
            if quad is None or quad in seeds:
                ok = False
                break
            seeds[quad] = {"coord": (sr, sc), "color": seed_color}
        if not ok or len(seeds) != len(segments) or set(seeds) != segment_quads:
            continue

        frames.append(
            {
                "bbox": frame_bbox,
                "color": int(color),
                "segments": segments,
                "seeds": seeds,
            }
        )
    frames.sort(key=lambda frame: tuple(int(v) for v in frame["bbox"]))
    return bg, frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each cut frame fragment using the seed from the same quadrant,
    then eject that fragment diagonally outward by its own pixel count.
    """
    g = np.asarray(grid, dtype=int)
    bg, frames = _extract_frames(g)
    if not frames:
        return g.copy()

    out = np.full_like(g, bg)
    h, w = g.shape
    for frame in frames:
        seeds = frame["seeds"]
        for segment in frame["segments"]:
            quad = segment["quadrant"]
            dr, dc = QUAD_DIR[quad]
            distance = int(segment["size"])
            color = int(seeds[quad]["color"])
            for r, c in segment["coords"]:
                rr = int(r) + dr * distance
                cc = int(c) + dc * distance
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = color
    return out


def _perimeter_cycle(height: int, width: int) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    for c in range(width):
        pts.append((0, c))
    for r in range(1, height):
        pts.append((r, width - 1))
    for c in range(width - 2, -1, -1):
        pts.append((height - 1, c))
    for r in range(height - 2, 0, -1):
        pts.append((r, 0))
    return pts


def _random_split(
    rng: np.random.Generator, total: int, parts: int, min_each: int
) -> list[int]:
    if total < parts * min_each:
        raise ValueError("total too small for requested split")
    vals = np.full(parts, min_each, dtype=int)
    extra = total - parts * min_each
    if extra > 0:
        vals += rng.multinomial(extra, np.full(parts, 1.0 / parts))
    rng.shuffle(vals)
    return [int(v) for v in vals]


def _seed_cells_for_pattern(height: int, width: int) -> dict[str, list[tuple[int, int]]]:
    bbox = (0, 0, height - 1, width - 1)
    cells = {quad: [] for quad in QUADRANTS}
    for r in range(1, height - 1):
        for c in range(1, width - 1):
            quad = _point_quadrant(r, c, bbox)
            if quad is not None:
                cells[quad].append((r, c))
    return cells


def _make_frame_pattern(
    rng: np.random.Generator, height: int, width: int, n_segments: int
) -> dict[str, object] | None:
    cycle = _perimeter_cycle(height, width)
    perimeter = len(cycle)
    bbox = (0, 0, height - 1, width - 1)
    seed_cells = _seed_cells_for_pattern(height, width)
    max_size = {2: 14, 3: 12, 4: 10}[n_segments]

    for _ in range(FRAME_ATTEMPTS):
        cut_lengths = [int(rng.integers(1, 3)) for _ in range(n_segments)]
        total_arc = perimeter - sum(cut_lengths)
        if total_arc < 3 * n_segments:
            continue
        arc_lengths = _random_split(rng, total_arc, n_segments, 3)
        start = int(rng.integers(perimeter))

        segments: list[dict[str, object]] = []
        used_quads: set[str] = set()
        pos = start
        ok = True
        for arc_len, cut_len in zip(arc_lengths, cut_lengths):
            coords = [cycle[(pos + i) % perimeter] for i in range(arc_len)]
            pos = (pos + arc_len + cut_len) % perimeter
            quad = _segment_quadrant(coords, bbox)
            if quad is None or quad in used_quads or not seed_cells[quad]:
                ok = False
                break
            seg_bbox = _bbox(coords)
            segments.append(
                {
                    "coords": coords,
                    "quadrant": quad,
                    "size": int(arc_len),
                    "bbox": seg_bbox,
                }
            )
            used_quads.add(quad)
        if not ok or max(int(seg["size"]) for seg in segments) > max_size:
            continue

        frame_coords = sorted({pt for seg in segments for pt in seg["coords"]})
        mask = np.zeros((height, width), dtype=bool)
        for r, c in frame_coords:
            mask[r, c] = True
        if len(frame_coords) != sum(int(seg["size"]) for seg in segments):
            continue
        if len(_components(mask)) != n_segments:
            continue

        return {
            "height": int(height),
            "width": int(width),
            "segments": segments,
            "frame_coords": frame_coords,
            "seed_cells": seed_cells,
        }
    return None


def _allowed_origin_range(
    pattern: dict[str, object], grid_h: int, grid_w: int
) -> tuple[int, int, int, int] | None:
    height = int(pattern["height"])
    width = int(pattern["width"])
    r_lo = 0
    r_hi = grid_h - height
    c_lo = 0
    c_hi = grid_w - width
    for segment in pattern["segments"]:
        size = int(segment["size"])
        sr0, sc0, sr1, sc1 = segment["bbox"]
        dr, dc = QUAD_DIR[segment["quadrant"]]
        if dr < 0:
            r_lo = max(r_lo, size - sr0)
        else:
            r_hi = min(r_hi, grid_h - 1 - size - sr1)
        if dc < 0:
            c_lo = max(c_lo, size - sc0)
        else:
            c_hi = min(c_hi, grid_w - 1 - size - sc1)
    if r_lo > r_hi or c_lo > c_hi:
        return None
    return int(r_lo), int(r_hi), int(c_lo), int(c_hi)


def _translated_output_pixels(
    pattern: dict[str, object], top: int, left: int
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for segment in pattern["segments"]:
        size = int(segment["size"])
        dr, dc = QUAD_DIR[segment["quadrant"]]
        for r, c in segment["coords"]:
            out.append((top + int(r) + dr * size, left + int(c) + dc * size))
    return out


def _reserve_coords(
    pattern: dict[str, object],
    top: int,
    left: int,
    out_pixels: list[tuple[int, int]],
    grid_h: int,
    grid_w: int,
) -> set[tuple[int, int]]:
    reserve: set[tuple[int, int]] = set()
    height = int(pattern["height"])
    width = int(pattern["width"])
    for r in range(top - 1, top + height + 1):
        for c in range(left - 1, left + width + 1):
            if 0 <= r < grid_h and 0 <= c < grid_w:
                reserve.add((r, c))
    for r, c in out_pixels:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < grid_h and 0 <= cc < grid_w:
                    reserve.add((rr, cc))
    return reserve


def _place_pattern(
    rng: np.random.Generator,
    grid: np.ndarray,
    blocked: np.ndarray,
    pattern: dict[str, object],
    frame_color: int,
    seed_colors: list[int],
) -> bool:
    h, w = grid.shape
    allowed = _allowed_origin_range(pattern, h, w)
    if allowed is None:
        return False
    r_lo, r_hi, c_lo, c_hi = allowed
    positions = [
        (int(r), int(c))
        for r in range(r_lo, r_hi + 1)
        for c in range(c_lo, c_hi + 1)
    ]
    if not positions:
        return False
    for idx in rng.permutation(len(positions)):
        top, left = positions[int(idx)]
        out_pixels = _translated_output_pixels(pattern, top, left)
        if len(set(out_pixels)) != len(out_pixels):
            continue

        quad_order = [str(seg["quadrant"]) for seg in pattern["segments"]]
        seeds: dict[str, tuple[tuple[int, int], int]] = {}
        for quad, color in zip(quad_order, seed_colors):
            cells = pattern["seed_cells"][quad]
            sr, sc = cells[int(rng.integers(len(cells)))]
            seeds[quad] = ((top + sr, left + sc), int(color))

        reserve = _reserve_coords(pattern, top, left, out_pixels, h, w)
        if any(blocked[r, c] for r, c in reserve):
            continue

        for r, c in pattern["frame_coords"]:
            grid[top + r, left + c] = int(frame_color)
        for (sr, sc), color in seeds.values():
            grid[sr, sc] = int(color)
        for r, c in reserve:
            blocked[r, c] = True
        return True
    return False


def _layout_options() -> list[list[int]]:
    return [
        [2],
        [3],
        [4],
        [2, 2],
        [2, 3],
        [3, 2],
        [2, 4],
        [4, 2],
        [3, 3],
        [3, 4],
        [4, 3],
    ]


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    counts = list(_layout_options()[int(rng.integers(len(_layout_options())))])
    grid_h = int(rng.integers(26, 33 if len(counts) == 1 else 39))
    grid_w = int(rng.integers(26, 33 if len(counts) == 1 else 39))
    bg = int(rng.integers(10))
    grid = np.full((grid_h, grid_w), bg, dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)

    available = [c for c in range(10) if c != bg]
    rng.shuffle(available)
    needed = len(counts) + sum(counts)
    if needed > len(available):
        return None
    cursor = 0
    frame_colors = [int(available[cursor + i]) for i in range(len(counts))]
    cursor += len(counts)

    patterns: list[dict[str, object]] = []
    for n_segments in counts:
        size_limit = {2: (6, 8), 3: (6, 9), 4: (7, 10)}[n_segments]
        pattern = None
        for _ in range(FRAME_ATTEMPTS):
            height = int(rng.integers(size_limit[0], size_limit[1] + 1))
            width = int(rng.integers(size_limit[0], size_limit[1] + 2))
            pattern = _make_frame_pattern(rng, height, width, n_segments)
            if pattern is not None:
                break
        if pattern is None:
            return None
        patterns.append(pattern)

    for frame_color, count, pattern in zip(frame_colors, counts, patterns):
        seed_colors = [int(available[cursor + i]) for i in range(count)]
        cursor += count
        if not _place_pattern(rng, grid, blocked, pattern, frame_color, seed_colors):
            return None

    bg2, frames = _extract_frames(grid)
    if bg2 != bg or len(frames) != len(counts):
        return None
    if sorted(len(frame["segments"]) for frame in frames) != sorted(counts):
        return None
    out = main(grid)
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATE_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated attempts")
