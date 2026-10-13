from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
SIDE_NAMES = ("left", "right", "top", "bottom")


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRECTIONS:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _find_mosaic_and_probes(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    mask = grid != BLACK
    comps = _components(mask)
    assert comps, "expected non-black cells"
    mosaic = max(comps, key=lambda pts: int(pts.shape[0]))
    mosaic_mask = np.zeros_like(mask, dtype=bool)
    mosaic_mask[mosaic[:, 0], mosaic[:, 1]] = True
    probes: list[tuple[int, int]] = []
    for pts in comps:
        if pts.shape[0] == mosaic.shape[0] and np.array_equal(pts, mosaic):
            continue
        assert pts.shape[0] == 1, "all non-mosaic objects must be singleton probes"
        probes.append((int(pts[0, 0]), int(pts[0, 1])))
    return mosaic_mask, sorted(probes)


def _ray_count(grid: np.ndarray, mosaic_mask: np.ndarray, r: int, c: int, dr: int, dc: int) -> int:
    h, w = map(int, grid.shape)
    color = int(grid[r, c])
    rr = r + dr
    cc = c + dc
    while 0 <= rr < h and 0 <= cc < w:
        if mosaic_mask[rr, cc]:
            count = 0
            while 0 <= rr < h and 0 <= cc < w and mosaic_mask[rr, cc]:
                if int(grid[rr, cc]) == color:
                    count += 1
                rr += dr
                cc += dc
            return count
        if int(grid[rr, cc]) != BLACK:
            return 0
        rr += dr
        cc += dc
    return 0


def _probe_score(grid: np.ndarray, mosaic_mask: np.ndarray, r: int, c: int) -> int:
    return sum(_ray_count(grid, mosaic_mask, r, c, dr, dc) for dr, dc in DIRECTIONS)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    mosaic_mask, probes = _find_mosaic_and_probes(g)
    out = g.copy()
    h, w = map(int, g.shape)

    for r, c in probes:
        color = int(g[r, c])
        score = _probe_score(g, mosaic_mask, r, c)
        if score <= 0:
            continue
        radius = score - 1
        for rr in range(max(0, r - radius), min(h - 1, r + radius) + 1):
            for cc in range(max(0, c - radius), min(w - 1, c + radius) + 1):
                if not mosaic_mask[rr, cc]:
                    out[rr, cc] = color

    return out


def _line_candidates(mosaic: np.ndarray, side: str, row0: int, col0: int) -> list[dict[str, int | str]]:
    candidates: list[dict[str, int | str]] = []
    if side in {"left", "right"}:
        for local_r in range(mosaic.shape[0]):
            counts = Counter(map(int, mosaic[local_r]))
            colors = [color for color, count in counts.items() if 1 <= int(count) <= 4]
            random.shuffle(colors)
            for color in colors:
                candidates.append(
                    {
                        "side": side,
                        "line": row0 + local_r,
                        "color": int(color),
                        "score": int(counts[int(color)]),
                    }
                )
    else:
        for local_c in range(mosaic.shape[1]):
            counts = Counter(map(int, mosaic[:, local_c]))
            colors = [color for color, count in counts.items() if 1 <= int(count) <= 4]
            random.shuffle(colors)
            for color in colors:
                candidates.append(
                    {
                        "side": side,
                        "line": col0 + local_c,
                        "color": int(color),
                        "score": int(counts[int(color)]),
                    }
                )
    random.shuffle(candidates)
    return candidates


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 < br0 - pad or br1 < ar0 - pad or ac1 < bc0 - pad or bc1 < ac0 - pad


def _place_probe(
    side: str,
    line: int,
    score: int,
    row0: int,
    col0: int,
    row1: int,
    col1: int,
    height: int,
    width: int,
) -> tuple[int, int, tuple[int, int, int, int]] | None:
    radius = score - 1
    gap_min = max(2, radius)
    if side == "left":
        gap_max = col0 - radius - 1
        if gap_max < gap_min:
            return None
        gap = random.randint(gap_min, gap_max)
        r = line
        c = col0 - gap - 1
    elif side == "right":
        gap_max = (width - 1 - radius) - col1 - 1
        if gap_max < gap_min:
            return None
        gap = random.randint(gap_min, gap_max)
        r = line
        c = col1 + gap + 1
    elif side == "top":
        gap_max = row0 - radius - 1
        if gap_max < gap_min:
            return None
        gap = random.randint(gap_min, gap_max)
        r = row0 - gap - 1
        c = line
    else:
        gap_max = (height - 1 - radius) - row1 - 1
        if gap_max < gap_min:
            return None
        gap = random.randint(gap_min, gap_max)
        r = row1 + gap + 1
        c = line

    box = (r - radius, c - radius, r + radius, c + radius)
    if box[0] < 0 or box[1] < 0 or box[2] >= height or box[3] >= width:
        return None
    return int(r), int(c), box


def _build_mosaic(height: int, width: int, palette: tuple[int, ...]) -> np.ndarray | None:
    mosaic = np.zeros((height, width), dtype=int)
    for r in range(height):
        shift = random.randrange(len(palette))
        for c in range(width):
            if random.random() < 0.65:
                mosaic[r, c] = int(palette[(c + shift + random.randrange(2)) % len(palette)])
            else:
                mosaic[r, c] = int(random.choice(palette))
    if set(map(int, np.unique(mosaic))) != set(map(int, palette)):
        return None
    return mosaic


def _validate_generated(grid: np.ndarray, probes: list[dict[str, int | tuple[int, int, int, int]]]) -> bool:
    mosaic_mask, found_probes = _find_mosaic_and_probes(grid)
    if len(found_probes) != len(probes):
        return False

    occupied = np.zeros_like(grid, dtype=int)
    for spec in probes:
        r = int(spec["r"])
        c = int(spec["c"])
        score = int(spec["score"])
        if (r, c) not in found_probes:
            return False
        if _probe_score(grid, mosaic_mask, r, c) != score:
            return False
        radius = score - 1
        for rr in range(r - radius, r + radius + 1):
            for cc in range(c - radius, c + radius + 1):
                if mosaic_mask[rr, cc]:
                    return False
                occupied[rr, cc] += 1
    return bool((occupied <= 1).all())


def _gen_attempt() -> np.ndarray | None:
    mosaic_h = random.randint(8, 12)
    mosaic_w = random.randint(8, 12)
    top_margin = random.randint(6, 10)
    bottom_margin = random.randint(6, 10)
    left_margin = random.randint(6, 10)
    right_margin = left_margin + random.randint(1, 5)

    height = top_margin + mosaic_h + bottom_margin
    width = left_margin + mosaic_w + right_margin
    row0 = top_margin
    col0 = left_margin
    row1 = row0 + mosaic_h - 1
    col1 = col0 + mosaic_w - 1

    palette_size = random.randint(4, 6)
    palette = tuple(random.sample(PALETTE, palette_size))
    mosaic = _build_mosaic(mosaic_h, mosaic_w, palette)
    if mosaic is None:
        return None

    side_candidates = {side: _line_candidates(mosaic, side, row0, col0) for side in SIDE_NAMES}
    if any(not side_candidates[side] for side in SIDE_NAMES):
        return None

    target_probes = random.randint(3, 6)
    required_count = random.randint(2, min(4, target_probes))
    side_plan = random.sample(SIDE_NAMES, required_count)
    side_plan.extend(random.choices(SIDE_NAMES, k=target_probes - required_count))
    random.shuffle(side_plan)

    probes: list[dict[str, int | tuple[int, int, int, int]]] = []
    boxes: list[tuple[int, int, int, int]] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()

    for side in side_plan:
        placed = False
        for cand in side_candidates[side]:
            score = int(cand["score"])
            for _ in range(4):
                placement = _place_probe(
                    side,
                    int(cand["line"]),
                    score,
                    row0,
                    col0,
                    row1,
                    col1,
                    height,
                    width,
                )
                if placement is None:
                    continue
                r, c, box = placement
                if r in used_rows or c in used_cols:
                    continue
                if any(max(abs(r - int(spec["r"])), abs(c - int(spec["c"]))) <= 1 for spec in probes):
                    continue
                if any(not _boxes_separated(box, other, pad=1) for other in boxes):
                    continue
                probes.append(
                    {
                        "side": side,
                        "line": int(cand["line"]),
                        "score": score,
                        "color": int(cand["color"]),
                        "r": r,
                        "c": c,
                        "box": box,
                    }
                )
                boxes.append(box)
                used_rows.add(r)
                used_cols.add(c)
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    grid = np.zeros((height, width), dtype=int)
    grid[row0 : row1 + 1, col0 : col1 + 1] = mosaic
    for spec in probes:
        grid[int(spec["r"]), int(spec["c"])] = int(spec["color"])

    if not _validate_generated(grid, probes):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
