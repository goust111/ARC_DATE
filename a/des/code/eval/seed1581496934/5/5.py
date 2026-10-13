import random
from collections import deque

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
TYPE_ORDER = ("A", "B", "C", "D")
NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

# --------------- solver ---------------


def _connected_components(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS8:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append({"coords": arr, "bbox": (int(r0), int(c0), int(r1), int(c1))})
    return comps


def _legend_colors(grid: np.ndarray) -> dict[str, int]:
    cols = np.flatnonzero(grid[-1] != BG)
    if cols.size != 4:
        raise ValueError("legend row must contain exactly four pixels")
    colors = [int(grid[-1, c]) for c in cols]
    if len(set(colors)) != 4:
        raise ValueError("legend colors must be distinct")
    return dict(zip(TYPE_ORDER, colors))


def _classify_symmetry(patch: np.ndarray) -> str:
    if patch.shape[0] != patch.shape[1] or patch.shape[0] not in (3, 4):
        raise ValueError("seed patch must be a 3x3 or 4x4 square")
    lr = np.array_equal(patch, np.fliplr(patch))
    ud = np.array_equal(patch, np.flipud(patch))
    rot = np.array_equal(patch, np.rot90(patch, 1))
    if rot:
        return "C"
    if lr and not ud:
        return "A"
    if ud and not lr:
        return "B"
    if not lr and not ud:
        return "D"
    raise ValueError("seed symmetry is ambiguous for this task")


def _extract_seeds(grid: np.ndarray) -> list[dict]:
    comps = _connected_components(grid[:-1] != BG)
    seeds: list[dict] = []
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        h, w = patch.shape
        if h != w or h not in (3, 4):
            raise ValueError("every seed must have a 3x3 or 4x4 bounding box")
        seeds.append(
            {
                "bbox": (r0, c0, r1, c1),
                "patch": patch,
                "kind": _classify_symmetry(patch),
            }
        )
    seeds.sort(key=lambda seed: (seed["bbox"][0], seed["bbox"][1]))
    return seeds


def _draw_ray(
    out: np.ndarray,
    blockers: np.ndarray,
    start_r: int,
    start_c: int,
    dr: int,
    dc: int,
    color: int,
) -> None:
    h, w = out.shape
    r, c = start_r, start_c
    while 0 <= r < h - 1 and 0 <= c < w:
        if blockers[r, c]:
            break
        out[r, c] = color
        r += dr
        c += dc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Classify each seed by symmetry, look up the matching legend color, and grow rays.

    Type A: left-right mirror symmetry -> grow horizontal rays from the left and right sides.
    Type B: top-bottom mirror symmetry -> grow vertical rays from the top and bottom sides.
    Type C: 90-degree rotational symmetry -> grow diagonal rays from the four corners.
    Type D: no symmetry -> unchanged.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.shape[0] < 4 or g.shape[1] < 4:
        raise ValueError("grid must be a 2D canvas")

    legend = _legend_colors(g)
    seeds = _extract_seeds(g)
    blockers = g != BG
    blockers[-1, :] = True  # keep the legend row unchanged

    out = g.copy()
    for seed in seeds:
        r0, c0, r1, c1 = seed["bbox"]
        color = legend[seed["kind"]]
        if seed["kind"] == "A":
            for r in range(r0, r1 + 1):
                _draw_ray(out, blockers, r, c0 - 1, 0, -1, color)
                _draw_ray(out, blockers, r, c1 + 1, 0, 1, color)
        elif seed["kind"] == "B":
            for c in range(c0, c1 + 1):
                _draw_ray(out, blockers, r0 - 1, c, -1, 0, color)
                _draw_ray(out, blockers, r1 + 1, c, 1, 0, color)
        elif seed["kind"] == "C":
            _draw_ray(out, blockers, r0 - 1, c0 - 1, -1, -1, color)
            _draw_ray(out, blockers, r0 - 1, c1 + 1, -1, 1, color)
            _draw_ray(out, blockers, r1 + 1, c0 - 1, 1, -1, color)
            _draw_ray(out, blockers, r1 + 1, c1 + 1, 1, 1, color)
    return out


# --------------- generator ---------------

GRID_H_RANGE = (16, 25)
GRID_W_RANGE = (18, 31)
SEED_COUNT_RANGE = (4, 7)
SEED_GAP = 1
GEN_ATTEMPTS = 400
PATTERN_ATTEMPTS = 200


def _rotation_orbits(n: int) -> list[list[tuple[int, int]]]:
    seen: set[tuple[int, int]] = set()
    orbits: list[list[tuple[int, int]]] = []
    for r in range(n):
        for c in range(n):
            if (r, c) in seen:
                continue
            orbit: list[tuple[int, int]] = []
            rr, cc = r, c
            for _ in range(4):
                if (rr, cc) not in orbit:
                    orbit.append((rr, cc))
                rr, cc = cc, n - 1 - rr
            for cell in orbit:
                seen.add(cell)
            orbits.append(orbit)
    return orbits


def _sample_seed_pattern(kind: str, size: int, colors: list[int]) -> np.ndarray:
    for _ in range(PATTERN_ATTEMPTS):
        patch = np.zeros((size, size), dtype=int)
        if kind == "A":
            half = (size + 1) // 2
            for r in range(size):
                for c in range(half):
                    patch[r, c] = random.choice(colors)
                    patch[r, size - 1 - c] = patch[r, c]
        elif kind == "B":
            half = (size + 1) // 2
            for r in range(half):
                for c in range(size):
                    patch[r, c] = random.choice(colors)
                    patch[size - 1 - r, c] = patch[r, c]
        elif kind == "C":
            for orbit in _rotation_orbits(size):
                color = random.choice(colors)
                for r, c in orbit:
                    patch[r, c] = color
        else:
            patch = np.array(
                [[random.choice(colors) for _ in range(size)] for _ in range(size)],
                dtype=int,
            )
        if len(np.unique(patch)) < 2:
            continue
        try:
            sampled_kind = _classify_symmetry(patch)
        except ValueError:
            continue
        if sampled_kind == kind:
            return patch
    raise RuntimeError(f"failed to sample a {kind} seed")


def _seed_margins(kind: str) -> tuple[int, int, int, int]:
    top = 1 if kind in ("B", "C") else 0
    left = 1 if kind in ("A", "C") else 0
    bottom = 1 if kind in ("B", "C") else 0
    right = 1 if kind in ("A", "C") else 0
    return top, left, bottom, right


def _rects_conflict(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _place_seeds(h: int, w: int, seeds: list[dict]) -> list[dict] | None:
    placed: list[dict] = []
    order = sorted(
        range(len(seeds)),
        key=lambda i: (sum(_seed_margins(seeds[i]["kind"])), seeds[i]["patch"].shape[0]),
        reverse=True,
    )
    for idx in order:
        seed = seeds[idx]
        size = int(seed["patch"].shape[0])
        top, left, bottom, right = _seed_margins(seed["kind"])
        r_lo = top
        c_lo = left
        r_hi = h - 2 - size - bottom
        c_hi = w - size - right
        if r_lo > r_hi or c_lo > c_hi:
            return None
        success = False
        for _ in range(200):
            r0 = random.randint(r_lo, r_hi)
            c0 = random.randint(c_lo, c_hi)
            rect = (r0, c0, r0 + size - 1, c0 + size - 1)
            if any(_rects_conflict(rect, other["bbox"], SEED_GAP) for other in placed):
                continue
            placed.append({**seed, "bbox": rect})
            success = True
            break
        if not success:
            return None
    return placed


def _sample_legend_columns(w: int) -> list[int]:
    cols = random.sample(range(w), 4)
    cols.sort()
    return cols


def _build_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    legend_colors = random.sample(list(PALETTE), 4)
    seed_palette = [color for color in PALETTE if color not in legend_colors]

    seed_count = random.randint(*SEED_COUNT_RANGE)
    kinds = list(TYPE_ORDER) + [random.choice(TYPE_ORDER) for _ in range(seed_count - 4)]
    random.shuffle(kinds)

    seeds: list[dict] = []
    for kind in kinds:
        size = random.choice((3, 4))
        palette_size = random.randint(2, min(4, len(seed_palette)))
        colors = random.sample(seed_palette, palette_size)
        seeds.append({"kind": kind, "patch": _sample_seed_pattern(kind, size, colors)})

    placed = _place_seeds(h, w, seeds)
    if placed is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    for seed in placed:
        r0, c0, r1, c1 = seed["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = seed["patch"]

    for col, color in zip(_sample_legend_columns(w), legend_colors):
        grid[h - 1, col] = color

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if not np.array_equal(out[-1], grid[-1]):
        return None

    seen_kinds = sorted(seed["kind"] for seed in _extract_seeds(grid))
    if seen_kinds.count("A") == 0 or seen_kinds.count("B") == 0 or seen_kinds.count("C") == 0 or seen_kinds.count("D") == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid crystal-growth ARC input")
