from __future__ import annotations

from collections import Counter
import random

import numpy as np


DIRECTION_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

ALL_COLORS = tuple(range(10))
GENERATION_ATTEMPTS = 300
PLACEMENT_ATTEMPTS = 200


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    return cells


def _parse_frame_component(
    grid: np.ndarray, pts: list[tuple[int, int]], color: int, bg: int
) -> dict | None:
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 6 or w < 6:
        return None

    pset = set(pts)
    border = _border_cells(r0, c0, r1, c1)
    if not pset <= border:
        return None

    missing = list(border - pset)
    if len(missing) != 1:
        return None
    gap = missing[0]
    gr, gc = gap
    if (gr in (r0, r1)) and (gc in (c0, c1)):
        return None
    if int(grid[gr, gc]) != bg:
        return None

    if gr == r0:
        direction = "up"
    elif gr == r1:
        direction = "down"
    elif gc == c0:
        direction = "left"
    else:
        direction = "right"

    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    if inner.size == 0:
        return None

    seed_positions = np.argwhere((inner != bg) & (inner != color))
    if len(seed_positions) == 0:
        return None

    for sr, sc in seed_positions:
        if sr == 0 or sr == inner.shape[0] - 1 or sc == 0 or sc == inner.shape[1] - 1:
            return None

    return {
        "bbox": (r0, c0, r1, c1),
        "gap": gap,
        "direction": direction,
        "seed_points": [(r0 + 1 + int(sr), c0 + 1 + int(sc)) for sr, sc in seed_positions],
    }


def _find_structure(grid: np.ndarray, bg: int) -> tuple[int, list[dict]]:
    candidates: list[tuple[int, list[dict]]] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        comps = _components(grid == color)
        if not comps:
            continue
        parsed: list[dict] = []
        ok = True
        for comp in comps:
            frame = _parse_frame_component(grid, comp, color, bg)
            if frame is None:
                ok = False
                break
            parsed.append(frame)
        if ok:
            candidates.append((color, parsed))
    assert len(candidates) == 1, "expected exactly one structure color"
    return candidates[0]


def _ordered_seeds(grid: np.ndarray, frame: dict) -> list[tuple[int, int, int]]:
    pts = frame["seed_points"]
    direction = frame["direction"]

    if direction == "left":
        key = lambda rc: (rc[0], rc[1])
    elif direction == "right":
        key = lambda rc: (rc[0], -rc[1])
    elif direction == "up":
        key = lambda rc: (rc[1], rc[0])
    else:
        key = lambda rc: (rc[1], -rc[0])

    ordered = sorted(pts, key=key)
    return [(r, c, int(grid[r, c])) for r, c in ordered]


def _draw_growth(out: np.ndarray, structure_color: int, frame: dict, seeds: list[tuple[int, int, int]]) -> None:
    counts = Counter(color for _, _, color in seeds)
    gap_r, gap_c = frame["gap"]
    direction = frame["direction"]
    dr, dc = DIRECTION_VECTORS[direction]
    h, w = out.shape

    for i, (_, _, color) in enumerate(seeds):
        tr = gap_r + dr * (i + 1)
        tc = gap_c + dc * (i + 1)
        assert 0 <= tr < h and 0 <= tc < w, "trunk must fit in-grid"
        out[tr, tc] = structure_color

        branch_len = counts[color]
        neg = branch_len // 2
        pos = (branch_len + 1) // 2

        if direction in ("left", "right"):
            for step in range(1, neg + 1):
                rr = tr - step
                assert 0 <= rr < h
                out[rr, tc] = color
            for step in range(1, pos + 1):
                rr = tr + step
                assert 0 <= rr < h
                out[rr, tc] = color
        else:
            for step in range(1, neg + 1):
                cc = tc - step
                assert 0 <= cc < w
                out[tr, cc] = color
            for step in range(1, pos + 1):
                cc = tc + step
                assert 0 <= cc < w
                out[tr, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a structure-colored trunk out of each frame gap, then attach one colored
    cross-branch per interior seed in scan order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    structure_color, frames = _find_structure(g, bg)

    out = g.copy()
    for frame in frames:
        seeds = _ordered_seeds(g, frame)
        _draw_growth(out, structure_color, frame, seeds)
    return out


def _sample_seed_spec(safe_cells: int, rng: random.Random) -> tuple[int, list[int]]:
    total = rng.randint(3, min(7, safe_cells))
    max_colors = min(4, total - 1, len(ALL_COLORS) - 2)
    color_count = rng.randint(2, max_colors)
    counts = [1] * color_count
    for _ in range(total - color_count):
        counts[rng.randrange(color_count)] += 1
    assert any(v > 1 for v in counts)
    return total, counts


def _sample_seed_positions(
    safe_positions: list[tuple[int, int]], total: int, rng: random.Random
) -> list[tuple[int, int]] | None:
    for _ in range(50):
        chosen = rng.sample(safe_positions, total)
        ok = True
        for i, (r0, c0) in enumerate(chosen):
            for r1, c1 in chosen[i + 1 :]:
                if abs(r0 - r1) + abs(c0 - c1) == 1:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return chosen
    return None


def _patch_geometry(
    frame_h: int,
    frame_w: int,
    gap_idx: int,
    trunk_len: int,
    counts: list[int],
    direction: str,
) -> tuple[int, int, int, int]:
    max_neg = max(v // 2 for v in counts)
    max_pos = max((v + 1) // 2 for v in counts)

    if direction in ("left", "right"):
        row_min_rel = min(0, gap_idx - max_neg)
        row_max_rel = max(frame_h - 1, gap_idx + max_pos)
        if direction == "left":
            col_min_rel = -trunk_len
            col_max_rel = frame_w - 1
        else:
            col_min_rel = 0
            col_max_rel = frame_w - 1 + trunk_len
    else:
        col_min_rel = min(0, gap_idx - max_neg)
        col_max_rel = max(frame_w - 1, gap_idx + max_pos)
        if direction == "up":
            row_min_rel = -trunk_len
            row_max_rel = frame_h - 1
        else:
            row_min_rel = 0
            row_max_rel = frame_h - 1 + trunk_len

    pad = 1
    patch_h = row_max_rel - row_min_rel + 1 + 2 * pad
    patch_w = col_max_rel - col_min_rel + 1 + 2 * pad
    frame_top = pad - row_min_rel
    frame_left = pad - col_min_rel
    return patch_h, patch_w, frame_top, frame_left


def _make_frame_patch(bg: int, structure_color: int, rng: random.Random, direction: str) -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        frame_h = rng.randint(7, 9)
        frame_w = rng.randint(7, 9)
        safe_rows = list(range(2, frame_h - 2))
        safe_cols = list(range(2, frame_w - 2))
        safe_cells = len(safe_rows) * len(safe_cols)
        if safe_cells < 3:
            continue

        trunk_len, counts = _sample_seed_spec(safe_cells, rng)

        if direction in ("left", "right"):
            gap_idx = rng.randint(1, frame_h - 2)
        else:
            gap_idx = rng.randint(1, frame_w - 2)

        patch_h, patch_w, frame_top, frame_left = _patch_geometry(
            frame_h, frame_w, gap_idx, trunk_len, counts, direction
        )

        available_colors = [c for c in ALL_COLORS if c not in (bg, structure_color)]
        rng.shuffle(available_colors)
        seed_colors = available_colors[: len(counts)]
        color_counts = dict(zip(seed_colors, counts))

        patch = np.full((patch_h, patch_w), bg, dtype=int)
        r0 = frame_top
        c0 = frame_left
        r1 = r0 + frame_h - 1
        c1 = c0 + frame_w - 1

        patch[r0, c0 : c1 + 1] = structure_color
        patch[r1, c0 : c1 + 1] = structure_color
        patch[r0 : r1 + 1, c0] = structure_color
        patch[r0 : r1 + 1, c1] = structure_color

        if direction == "left":
            gap = (r0 + gap_idx, c0)
        elif direction == "right":
            gap = (r0 + gap_idx, c1)
        elif direction == "up":
            gap = (r0, c0 + gap_idx)
        else:
            gap = (r1, c0 + gap_idx)
        patch[gap] = bg

        safe_positions = [(r0 + rr, c0 + cc) for rr in safe_rows for cc in safe_cols]
        chosen_positions = _sample_seed_positions(safe_positions, trunk_len, rng)
        if chosen_positions is None:
            continue

        color_list: list[int] = []
        for color, count in color_counts.items():
            color_list.extend([color] * count)
        rng.shuffle(color_list)

        for (sr, sc), color in zip(chosen_positions, color_list):
            patch[sr, sc] = color

        try:
            out = main(patch)
        except AssertionError:
            continue
        if np.array_equal(out, patch):
            continue
        return patch
    raise RuntimeError("failed to generate a valid frame patch")


def _place_patches(patches: list[np.ndarray], bg: int, rng: random.Random) -> np.ndarray | None:
    total_area = sum(int(p.shape[0] * p.shape[1]) for p in patches)
    max_h = max(int(p.shape[0]) for p in patches)
    max_w = max(int(p.shape[1]) for p in patches)

    min_h = max(max_h + 2, int(np.sqrt(total_area)) + 6)
    min_w = max(max_w + 2, int(np.sqrt(total_area)) + 6)
    max_dim = 40

    for _ in range(PLACEMENT_ATTEMPTS):
        upper_h = max(min_h, min(max_dim, min_h + 10))
        upper_w = max(min_w, min(max_dim, min_w + 10))
        h = rng.randint(min_h, upper_h)
        w = rng.randint(min_w, upper_w)
        if h * w < total_area + 20:
            continue

        grid = np.full((h, w), bg, dtype=int)
        occupied = np.zeros((h, w), dtype=bool)
        placements: list[tuple[int, int, np.ndarray]] = []
        ok = True

        for patch in sorted(patches, key=lambda arr: arr.shape[0] * arr.shape[1], reverse=True):
            ph, pw = map(int, patch.shape)
            placed = False
            for _ in range(PLACEMENT_ATTEMPTS):
                top = rng.randint(0, h - ph)
                left = rng.randint(0, w - pw)
                r0 = max(0, top - 1)
                c0 = max(0, left - 1)
                r1 = min(h, top + ph + 1)
                c1 = min(w, left + pw + 1)
                if occupied[r0:r1, c0:c1].any():
                    continue
                placements.append((top, left, patch))
                occupied[top : top + ph, left : left + pw] = True
                placed = True
                break
            if not placed:
                ok = False
                break

        if not ok:
            continue

        for top, left, patch in placements:
            ph, pw = patch.shape
            region = grid[top : top + ph, left : left + pw]
            mask = patch != bg
            region[mask] = patch[mask]

        return grid
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.choice(ALL_COLORS)
    remaining = [c for c in ALL_COLORS if c != bg]
    structure_color = rng.choice(remaining)
    directions = ["up", "down", "left", "right"]
    frame_count = rng.randint(1, 3)
    chosen_directions = [rng.choice(directions) for _ in range(frame_count)]

    try:
        patches = [_make_frame_patch(bg, structure_color, rng, direction) for direction in chosen_directions]
    except RuntimeError:
        return None

    grid = _place_patches(patches, bg, rng)
    if grid is None:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
