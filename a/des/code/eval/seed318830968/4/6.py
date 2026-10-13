import random
from typing import Iterable

import numpy as np


BG = 0
WALL = 5
LIGHTHOUSE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAGS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
EXTRA_REGION_RANGE = (0, 3)
MAX_PLACE_TRIES = 200
MAX_GENERATE_ATTEMPTS = 400


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
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(coords)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _detect_bg_and_wall(grid: np.ndarray) -> tuple[int, int]:
    vals, cnts = np.unique(grid, return_counts=True)
    bg = BG if np.any(vals == BG) else int(vals[int(np.argmax(cnts))])
    non_bg = [int(v) for v in vals if int(v) != bg]
    wall = bg if not non_bg else max(non_bg, key=lambda v: int((grid == v).sum()))
    return int(bg), int(wall)


def _ray_lengths(mask: np.ndarray, pos: tuple[int, int], directions: list[tuple[int, int]]) -> list[int]:
    h, w = mask.shape
    r0, c0 = pos
    lengths: list[int] = []
    for dr, dc in directions:
        rr, cc = r0 + dr, c0 + dc
        length = 0
        while 0 <= rr < h and 0 <= cc < w and mask[rr, cc]:
            length += 1
            rr += dr
            cc += dc
        lengths.append(length)
    return lengths


def _best_lighthouse(
    mask: np.ndarray,
    directions: list[tuple[int, int]],
    rng: random.Random,
    require_all: bool,
    min_nonzero: int,
) -> tuple[int, int]:
    pts = [tuple(map(int, p)) for p in np.argwhere(mask)]
    valid: list[tuple[tuple[int, ...], tuple[int, int]]] = []
    fallback: list[tuple[tuple[int, ...], tuple[int, int]]] = []
    center_r = (mask.shape[0] - 1) / 2.0
    center_c = (mask.shape[1] - 1) / 2.0
    for pos in pts:
        lengths = _ray_lengths(mask, pos, directions)
        nonzero = sum(length > 0 for length in lengths)
        score = (
            int(nonzero),
            int(sum(lengths)),
            int(min((length for length in lengths if length > 0), default=0)),
            -int(abs(pos[0] - center_r) + abs(pos[1] - center_c)),
        )
        fallback.append((score, pos))
        if require_all:
            if all(length > 0 for length in lengths):
                valid.append((score, pos))
        elif nonzero >= min_nonzero:
            valid.append((score, pos))

    scored = valid if valid else fallback
    best_score = max(score for score, _ in scored)
    candidates = [pos for score, pos in scored if score == best_score]
    return candidates[rng.randrange(len(candidates))]


def _make_square(rng: random.Random) -> tuple[np.ndarray, tuple[int, int], str]:
    size = rng.randint(4, 7)
    mask = np.ones((size, size), dtype=bool)
    light = _best_lighthouse(mask, DIRS4, rng, require_all=True, min_nonzero=4)
    return mask, light, "square"


def _make_wide_rect(rng: random.Random) -> tuple[np.ndarray, tuple[int, int], str]:
    h = rng.randint(3, 5)
    w = rng.randint(h + 1, min(9, h + 5))
    mask = np.ones((h, w), dtype=bool)
    light = _best_lighthouse(mask, [(0, -1), (0, 1)], rng, require_all=True, min_nonzero=2)
    return mask, light, "wide"


def _make_tall_rect(rng: random.Random) -> tuple[np.ndarray, tuple[int, int], str]:
    w = rng.randint(3, 5)
    h = rng.randint(w + 1, min(9, w + 5))
    mask = np.ones((h, w), dtype=bool)
    light = _best_lighthouse(mask, [(-1, 0), (1, 0)], rng, require_all=True, min_nonzero=2)
    return mask, light, "tall"


def _make_irregular(rng: random.Random) -> tuple[np.ndarray, tuple[int, int], str]:
    h = rng.randint(5, 8)
    w = rng.randint(5, 8)
    mask = np.ones((h, w), dtype=bool)
    if rng.random() < 0.5:
        notch_h = rng.randint(1, h - 2)
        notch_w = rng.randint(1, w - 2)
        corner = rng.choice(["tl", "tr", "bl", "br"])
        if corner == "tl":
            mask[:notch_h, :notch_w] = False
        elif corner == "tr":
            mask[:notch_h, w - notch_w :] = False
        elif corner == "bl":
            mask[h - notch_h :, :notch_w] = False
        else:
            mask[h - notch_h :, w - notch_w :] = False
    else:
        side = rng.choice(["top", "bottom", "left", "right"])
        if side in ("top", "bottom"):
            notch_h = rng.randint(1, h - 2)
            notch_w = rng.randint(1, w - 2)
            start = rng.randint(1, w - notch_w - 1)
            rows = slice(0, notch_h) if side == "top" else slice(h - notch_h, h)
            mask[rows, start : start + notch_w] = False
        else:
            notch_w = rng.randint(1, w - 2)
            notch_h = rng.randint(1, h - 2)
            start = rng.randint(1, h - notch_h - 1)
            cols = slice(0, notch_w) if side == "left" else slice(w - notch_w, w)
            mask[start : start + notch_h, cols] = False

    assert len(_components(mask)) == 1
    assert int(mask.sum()) < mask.size
    light = _best_lighthouse(mask, DIAGS, rng, require_all=False, min_nonzero=2)
    return mask, light, "irregular"


def _make_region(kind: str, rng: random.Random) -> tuple[np.ndarray, tuple[int, int], str]:
    if kind == "square":
        return _make_square(rng)
    if kind == "wide":
        return _make_wide_rect(rng)
    if kind == "tall":
        return _make_tall_rect(rng)
    return _make_irregular(rng)


def _trace_region(
    out: np.ndarray,
    mask: np.ndarray,
    origin: tuple[int, int],
    light: tuple[int, int],
    color: int,
    kind: str,
) -> int:
    if kind == "square":
        directions = DIRS4
    elif kind == "wide":
        directions = [(0, -1), (0, 1)]
    elif kind == "tall":
        directions = [(-1, 0), (1, 0)]
    else:
        directions = DIAGS

    h, w = mask.shape
    r0, c0 = origin
    lr, lc = light
    painted = 0
    for dr, dc in directions:
        rr, cc = lr + dr, lc + dc
        while 0 <= rr < h and 0 <= cc < w and mask[rr, cc]:
            gr, gc = r0 + rr, c0 + cc
            if out[gr, gc] != BG:
                break
            out[gr, gc] = color
            painted += 1
            rr += dr
            cc += dc
    return painted


def _valid_positions(occupied: np.ndarray, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = occupied.shape
    sh, sw = shape
    positions: list[tuple[int, int]] = []
    for r0 in range(h - sh + 1):
        for c0 in range(w - sw + 1):
            if occupied[r0 : r0 + sh, c0 : c0 + sw].any():
                continue
            positions.append((r0, c0))
    return positions


def _reserve(occupied: np.ndarray, origin: tuple[int, int], shape: tuple[int, int]) -> None:
    h, w = occupied.shape
    r0, c0 = origin
    sh, sw = shape
    occupied[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)] = True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    base_kinds = ["square", "wide", "tall", "irregular"]
    extra_count = rng.randint(*EXTRA_REGION_RANGE)
    kinds = base_kinds + [rng.choice(base_kinds) for _ in range(extra_count)]
    rng.shuffle(kinds)
    if len(kinds) > len(LIGHTHOUSE_COLORS):
        return None

    regions = [_make_region(kind, rng) for kind in kinds]
    estimated = sum((mask.shape[0] + 2) * (mask.shape[1] + 2) for mask, _, _ in regions)
    side_floor = max(18, int(np.ceil(np.sqrt(estimated * 1.4))))
    max_h = max(side_floor, GRID_H_RANGE[0])
    max_w = max(side_floor, GRID_W_RANGE[0])
    if max_h > GRID_H_RANGE[1] or max_w > GRID_W_RANGE[1]:
        return None

    grid_h = rng.randint(max_h, GRID_H_RANGE[1])
    grid_w = rng.randint(max_w, GRID_W_RANGE[1])
    grid = np.full((grid_h, grid_w), WALL, dtype=int)
    expected = grid.copy()
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    colors = rng.sample(list(LIGHTHOUSE_COLORS), len(kinds))
    placed_specs: list[tuple[np.ndarray, tuple[int, int], tuple[int, int], int, str]] = []

    for (mask, light, kind), color in zip(regions, colors):
        shape = mask.shape
        positions = _valid_positions(occupied, shape)
        if not positions:
            return None
        placed = False
        for _ in range(MAX_PLACE_TRIES):
            r0, c0 = positions[rng.randrange(len(positions))]
            sh, sw = shape
            sub = grid[r0 : r0 + sh, c0 : c0 + sw]
            if np.any((sub != WALL) & mask):
                continue
            sub[mask] = BG
            lr, lc = light
            grid[r0 + lr, c0 + lc] = color
            expected[r0 : r0 + sh, c0 : c0 + sw][mask] = BG
            expected[r0 + lr, c0 + lc] = color
            _reserve(occupied, (r0, c0), shape)
            placed_specs.append((mask, (r0, c0), light, color, kind))
            placed = True
            break
        if not placed:
            return None

    for mask, origin, light, color, kind in placed_specs:
        if _trace_region(expected, mask, origin, light, color, kind) == 0:
            return None

    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Lighthouse pixels radiate along directions implied by their enclosing region shape.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg, wall = _detect_bg_and_wall(g)
    passable = g != wall
    out = g.copy()

    for comp in _components(passable):
        comp_set = set(comp)
        lights = [(r, c) for r, c in comp if g[r, c] != bg and g[r, c] != wall]
        assert len(lights) == 1, "each region must contain exactly one lighthouse"
        lr, lc = lights[0]
        color = int(g[lr, lc])

        r0, c0, r1, c1 = _bbox(comp)
        box_h = r1 - r0 + 1
        box_w = c1 - c0 + 1
        fills_bbox = len(comp) == box_h * box_w

        if fills_bbox:
            if box_h == box_w:
                directions = DIRS4
            elif box_w > box_h:
                directions = [(0, -1), (0, 1)]
            else:
                directions = [(-1, 0), (1, 0)]
        else:
            directions = DIAGS

        for dr, dc in directions:
            rr, cc = lr + dr, lc + dc
            while 0 <= rr < h and 0 <= cc < w and (rr, cc) in comp_set:
                if out[rr, cc] != bg:
                    break
                out[rr, cc] = color
                rr += dr
                cc += dc

    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATE_ATTEMPTS} attempts")
