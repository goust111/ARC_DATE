from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
SEED_COLORS = [1, 2, 3, 4, 6, 7, 8]
CORNERS = ("TL", "TR", "BL", "BR")
BRANCHES = ("TL", "TR", "BL", "BR", "CENTER")

RECT_COUNT_RANGE = (2, 5)
INNER_H_RANGE = (5, 11)
INNER_W_RANGE = (5, 11)
MAX_GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_rect_frame(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    expected: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0 + 1, r1):
        expected.add((r, c0))
        expected.add((r, c1))
    return set(coords) == expected


def _find_frames(grid: np.ndarray, color: int) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for comp in _components(grid == color):
        if _is_rect_frame(comp):
            boxes.append(_bbox(comp))
    boxes.sort()
    return boxes


def _true_runs(values: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, val in enumerate(values.tolist()):
        if val and start is None:
            start = i
        elif not val and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(values) - 1))
    return runs


def _parse_patterns_by_runs(inner: np.ndarray) -> dict[int, np.ndarray]:
    row_runs = _true_runs(np.any(inner != BLACK, axis=1))
    col_runs = _true_runs(np.any(inner != BLACK, axis=0))
    if not row_runs or not col_runs:
        return {}
    if any(end - start != 2 for start, end in row_runs + col_runs):
        return {}

    patterns: dict[int, np.ndarray] = {}
    for rs, re in row_runs:
        for cs, ce in col_runs:
            patch = inner[rs : re + 1, cs : ce + 1]
            if patch.shape != (3, 3) or not np.any(patch != BLACK):
                continue
            key = int(patch[0, 0])
            if key == BLACK:
                continue
            if key in patterns and not np.array_equal(patterns[key], patch):
                return {}
            patterns[key] = patch.copy()
    return patterns


def _parse_patterns_by_lattice(inner: np.ndarray) -> dict[int, np.ndarray]:
    ih, iw = inner.shape
    best: dict[int, np.ndarray] = {}
    best_score = (-1, -10**9, -1)
    nonblack = inner != BLACK

    for row_offset in range(4):
        for col_offset in range(4):
            covered = np.zeros_like(inner, dtype=bool)
            patterns: dict[int, np.ndarray] = {}
            valid = True
            density = 0
            for r in range(row_offset, ih - 2, 4):
                for c in range(col_offset, iw - 2, 4):
                    patch = inner[r : r + 3, c : c + 3]
                    if not np.any(patch != BLACK):
                        continue
                    key = int(patch[0, 0])
                    if key == BLACK:
                        valid = False
                        break
                    if key in patterns and not np.array_equal(patterns[key], patch):
                        valid = False
                        break
                    patterns[key] = patch.copy()
                    covered[r : r + 3, c : c + 3] = True
                    density += int(np.count_nonzero(patch))
                if not valid:
                    break
            if not valid:
                continue

            uncovered = int(np.count_nonzero(nonblack & ~covered))
            score = (len(patterns), -uncovered, density)
            if score > best_score:
                best = patterns
                best_score = score
    return best


def _parse_dictionary(grid: np.ndarray) -> dict[int, np.ndarray]:
    frames = _find_frames(grid, MAROON)
    assert frames, "expected one maroon dictionary frame"
    frame = max(frames, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    r0, c0, r1, c1 = frame
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]

    patterns = _parse_patterns_by_runs(inner)
    if not patterns:
        patterns = _parse_patterns_by_lattice(inner)
    assert patterns, "failed to parse any 3x3 dictionary patterns"
    return patterns


def _seed_in_frame(grid: np.ndarray, box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    pts = np.argwhere(inner != BLACK)
    assert len(pts) == 1, "each grey frame must contain exactly one seed"
    rr, cc = map(int, pts[0])
    return r0 + 1 + rr, c0 + 1 + cc


def _nearest_corner(box: tuple[int, int, int, int], seed_r: int, seed_c: int) -> str:
    r0, c0, r1, c1 = box
    corners = {
        "TL": (r0 + 1, c0 + 1),
        "TR": (r0 + 1, c1 - 1),
        "BL": (r1 - 1, c0 + 1),
        "BR": (r1 - 1, c1 - 1),
    }
    dists = {
        name: (seed_r - rr) * (seed_r - rr) + (seed_c - cc) * (seed_c - cc)
        for name, (rr, cc) in corners.items()
    }
    best = min(dists.values())
    winners = [name for name in CORNERS if dists[name] == best]
    return "TL" if len(winners) != 1 else winners[0]


def _pattern_index(seed_r: int, seed_c: int, r: int, c: int, direction: str) -> tuple[int, int]:
    if direction in ("TL", "TR"):
        pr = (seed_r - r) % 3
    else:
        pr = (r - seed_r) % 3

    if direction in ("TL", "BL"):
        pc = (seed_c - c) % 3
    else:
        pc = (c - seed_c) % 3

    return pr, pc


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    patterns = _parse_dictionary(g)
    frames = _find_frames(g, GREY)
    out = g.copy()

    for box in frames:
        seed_r, seed_c = _seed_in_frame(g, box)
        seed_color = int(g[seed_r, seed_c])
        assert seed_color in patterns, "missing dictionary entry for seed color"
        patch = patterns[seed_color]
        direction = _nearest_corner(box, seed_r, seed_c)
        r0, c0, r1, c1 = box
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                pr, pc = _pattern_index(seed_r, seed_c, r, c, direction)
                out[r, c] = int(patch[pr, pc])

    return out


def _corner_for_local_point(r: int, c: int, h: int, w: int) -> tuple[str, bool]:
    dists = {
        "TL": r * r + c * c,
        "TR": r * r + (w - 1 - c) * (w - 1 - c),
        "BL": (h - 1 - r) * (h - 1 - r) + c * c,
        "BR": (h - 1 - r) * (h - 1 - r) + (w - 1 - c) * (w - 1 - c),
    }
    best = min(dists.values())
    winners = [name for name in CORNERS if dists[name] == best]
    return ("TL" if len(winners) != 1 else winners[0], len(winners) != 1)


def _sample_seed_local(inner_h: int, inner_w: int, branch: str, rng: random.Random) -> tuple[int, int] | None:
    if branch == "CENTER":
        if inner_h % 2 == 0 or inner_w % 2 == 0:
            return None
        return inner_h // 2, inner_w // 2

    target_corner = {
        "TL": (0, 0),
        "TR": (0, inner_w - 1),
        "BL": (inner_h - 1, 0),
        "BR": (inner_h - 1, inner_w - 1),
    }[branch]

    candidates: list[tuple[int, int, int]] = []
    for r in range(inner_h):
        for c in range(inner_w):
            corner, tie = _corner_for_local_point(r, c, inner_h, inner_w)
            if tie or corner != branch:
                continue
            dist = (r - target_corner[0]) * (r - target_corner[0]) + (c - target_corner[1]) * (c - target_corner[1])
            candidates.append((dist, r, c))
    if not candidates:
        return None

    candidates.sort()
    take = max(1, len(candidates) // 2)
    _, r, c = rng.choice(candidates[:take])
    return r, c


def _direction_preview(mask: np.ndarray, direction: str) -> bytes:
    preview = np.zeros((7, 7), dtype=np.uint8)
    sr = sc = 3
    for r in range(7):
        for c in range(7):
            pr, pc = _pattern_index(sr, sc, r, c, direction)
            preview[r, c] = int(mask[pr, pc])
    return preview.tobytes()


def _random_pattern_mask(rng: random.Random, used: set[tuple[int, ...]]) -> np.ndarray:
    cells = [(r, c) for r in range(3) for c in range(3) if (r, c) != (0, 0)]
    for _ in range(300):
        mask = np.zeros((3, 3), dtype=int)
        mask[0, 0] = 1
        rng.shuffle(cells)
        total_on = rng.randint(4, 6)
        for r, c in cells[: total_on - 1]:
            mask[r, c] = 1
        if not all(mask[r].any() for r in range(3)):
            continue
        if not all(mask[:, c].any() for c in range(3)):
            continue
        sigs = {_direction_preview(mask, direction) for direction in CORNERS}
        if len(sigs) < 4:
            continue
        key = tuple(int(x) for x in mask.reshape(-1))
        if key in used:
            continue
        used.add(key)
        return mask
    raise RuntimeError("failed to sample a sufficiently asymmetric 3x3 mask")


def _corner_origin(corner: str, grid_h: int, grid_w: int, obj_h: int, obj_w: int, inset_r: int, inset_c: int) -> tuple[int, int]:
    if corner == "TL":
        return inset_r, inset_c
    if corner == "TR":
        return inset_r, grid_w - obj_w - inset_c
    if corner == "BL":
        return grid_h - obj_h - inset_r, inset_c
    if corner == "BR":
        return grid_h - obj_h - inset_r, grid_w - obj_w - inset_c
    raise ValueError(corner)


def _can_place(occ: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occ.shape[0], r0 + h + pad)
    cc1 = min(occ.shape[1], c0 + w + pad)
    return not bool(occ[rr0:rr1, cc0:cc1].any())


def _mark_occupied(occ: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    occ[r0 : r0 + h, c0 : c0 + w] = True


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_rects = rng.randint(*RECT_COUNT_RANGE)
    colors = rng.sample(SEED_COLORS, n_rects)
    branches = rng.sample(list(BRANCHES), n_rects)

    n_rows = 1 if n_rects <= 3 and rng.random() < 0.5 else 2
    n_cols = (n_rects + n_rows - 1) // n_rows
    gap = 1
    pad = 1
    content_h = n_rows * 3 + (n_rows - 1) * gap
    content_w = n_cols * 3 + (n_cols - 1) * gap
    dict_h = content_h + 2 * pad + 2
    dict_w = content_w + 2 * pad + 2

    min_h = max(22, dict_h + 8, 8 + 4 * n_rects)
    min_w = max(24, dict_w + 10, 10 + 4 * n_rects)
    grid_h = rng.randint(min_h, min_h + 8)
    grid_w = rng.randint(min_w, min_w + 10)

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    occ = np.zeros((grid_h, grid_w), dtype=bool)

    dict_corner = rng.choice(CORNERS)
    inset_r = rng.randint(0, 1)
    inset_c = rng.randint(0, 1)
    dict_r0, dict_c0 = _corner_origin(dict_corner, grid_h, grid_w, dict_h, dict_w, inset_r, inset_c)
    if not _can_place(occ, dict_r0, dict_c0, dict_h, dict_w, pad=1):
        return None
    _mark_occupied(occ, dict_r0, dict_c0, dict_h, dict_w)
    _draw_frame(grid, (dict_r0, dict_c0, dict_r0 + dict_h - 1, dict_c0 + dict_w - 1), MAROON)

    used_masks: set[tuple[int, ...]] = set()
    pattern_masks = [_random_pattern_mask(rng, used_masks) for _ in range(n_rects)]
    tile_order = list(range(n_rects))
    rng.shuffle(tile_order)
    for slot, idx in enumerate(tile_order):
        rr = slot // n_cols
        cc = slot % n_cols
        tr = dict_r0 + 1 + pad + rr * (3 + gap)
        tc = dict_c0 + 1 + pad + cc * (3 + gap)
        patch = pattern_masks[idx] * colors[idx]
        black_mask = pattern_masks[idx] == 0
        patch = patch.astype(int)
        patch[black_mask] = BLACK
        grid[tr : tr + 3, tc : tc + 3] = patch

    for color, branch in zip(colors, branches):
        placed = False
        for _ in range(160):
            if branch == "CENTER":
                inner_h = rng.choice([v for v in range(INNER_H_RANGE[0], INNER_H_RANGE[1] + 1) if v % 2 == 1])
                inner_w = rng.choice([v for v in range(INNER_W_RANGE[0], INNER_W_RANGE[1] + 1) if v % 2 == 1])
            else:
                inner_h = rng.randint(*INNER_H_RANGE)
                inner_w = rng.randint(*INNER_W_RANGE)
            seed = _sample_seed_local(inner_h, inner_w, branch, rng)
            if seed is None:
                continue

            outer_h = inner_h + 2
            outer_w = inner_w + 2
            r0 = rng.randint(0, grid_h - outer_h)
            c0 = rng.randint(0, grid_w - outer_w)
            if not _can_place(occ, r0, c0, outer_h, outer_w, pad=1):
                continue

            _mark_occupied(occ, r0, c0, outer_h, outer_w)
            _draw_frame(grid, (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1), GREY)
            sr, sc = seed
            grid[r0 + 1 + sr, c0 + 1 + sc] = color
            placed = True
            break
        if not placed:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if int(np.count_nonzero(out != grid)) < 8:
        return None
    if len(_find_frames(grid, GREY)) != n_rects:
        return None
    if len(_parse_dictionary(grid)) != n_rects:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
