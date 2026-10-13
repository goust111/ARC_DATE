from __future__ import annotations

from functools import lru_cache
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GRAY = 5
ORANGE = 7
TEAL = 8

SQUARE = "square"
HBAR = "hbar"
VBAR = "vbar"

PIECE_DEFS = {
    SQUARE: ((0, 0), (0, 1), (1, 0), (1, 1)),
    HBAR: ((0, 0), (0, 1), (0, 2)),
    VBAR: ((0, 0), (1, 0), (2, 0)),
}
PIECE_AREAS = {SQUARE: 4, HBAR: 3, VBAR: 3}
PIECE_COLORS = {SQUARE: TEAL, HBAR: RED, VBAR: BLUE}
KIND_TO_COUNT_INDEX = {SQUARE: 0, HBAR: 1, VBAR: 2}

TEMPLATE_STRINGS = [
    [
        "###.",
        "###.",
        "###.",
        "..##",
        "..##",
    ],
    [
        ".###",
        "####",
        "####",
        "###.",
    ],
    [
        "#####",
        "##.##",
        "##.##",
        "##...",
    ],
    [
        "###..",
        "#####",
        "#####",
        "##...",
    ],
    [
        "...###",
        "..####",
        "######",
        "###...",
    ],
    [
        "...##.",
        "#####.",
        "######",
        "###...",
    ],
]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or not mask[sr, sc]:
                continue
            stack = [(sr, sc)]
            seen[sr, sc] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(sorted(coords), dtype=int))
    return parts


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _shape_kind(mask: np.ndarray) -> str:
    h, w = mask.shape
    if h == 2 and w == 2 and np.all(mask):
        return SQUARE
    if h == 1 and w == 3 and np.all(mask):
        return HBAR
    if h == 3 and w == 1 and np.all(mask):
        return VBAR
    raise AssertionError("unexpected hole shape")


def _find_wall_and_holes(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[tuple[np.ndarray, str]]]:
    orange_coords = np.argwhere(grid == ORANGE)
    assert orange_coords.size > 0, "expected an orange wall"
    r0, c0, r1, c1 = _bbox_from_coords(orange_coords)
    wall = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert wall.shape[0] >= 4 and wall.shape[1] >= 4, "wall must have room for holes"
    assert np.all(wall[0, :] == ORANGE) and np.all(wall[-1, :] == ORANGE)
    assert np.all(wall[:, 0] == ORANGE) and np.all(wall[:, -1] == ORANGE)
    assert set(map(int, np.unique(wall))) <= {BLACK, ORANGE}

    holes_mask = wall == BLACK
    holes = []
    for coords in _components(holes_mask):
        hr0, hc0, hr1, hc1 = _bbox_from_coords(coords)
        assert hr0 > 0 and hc0 > 0 and hr1 < wall.shape[0] - 1 and hc1 < wall.shape[1] - 1
        sub = holes_mask[hr0 : hr1 + 1, hc0 : hc1 + 1]
        kind = _shape_kind(sub)
        holes.append((coords + np.array([r0, c0]), kind))
    assert holes, "expected at least one wall hole"
    return (r0, c0, r1, c1), holes


def _gray_component(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    parts = _components(grid == GRAY)
    assert len(parts) == 1, "expected one gray component"
    coords = parts[0]
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    cropped = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    cropped[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return cropped, (r0, c0)


def _shape_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    return int(mask.shape[0]), int(mask.shape[1]), mask.astype(np.uint8).tobytes()


@lru_cache(maxsize=None)
def _placements_for_shape(key: tuple[int, int, bytes]) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[dict, ...], ...]]:
    h, w, raw = key
    mask = np.frombuffer(raw, dtype=np.uint8).reshape((h, w)).astype(bool)
    cells = tuple((int(r), int(c)) for r, c in np.argwhere(mask))
    index = {cell: i for i, cell in enumerate(cells)}
    cover: list[list[dict]] = [[] for _ in range(len(cells))]
    seen_masks: set[tuple[str, int]] = set()

    for r in range(h):
        for c in range(w):
            for kind, offsets in PIECE_DEFS.items():
                pts = tuple((r + dr, c + dc) for dr, dc in offsets)
                if not all(0 <= rr < h and 0 <= cc < w and mask[rr, cc] for rr, cc in pts):
                    continue
                bitmask = 0
                idxs = []
                for pt in pts:
                    idx = index[pt]
                    idxs.append(idx)
                    bitmask |= 1 << idx
                if (kind, bitmask) in seen_masks:
                    continue
                seen_masks.add((kind, bitmask))
                placement = {
                    "kind": kind,
                    "mask": bitmask,
                    "coords": tuple(sorted(pts)),
                }
                for idx in idxs:
                    cover[idx].append(placement)

    return cells, tuple(tuple(options) for options in cover)


def _count_vectors(mask: np.ndarray) -> dict[tuple[int, int, int], int]:
    key = _shape_key(mask)
    cells, cover = _placements_for_shape(key)
    full = (1 << len(cells)) - 1

    @lru_cache(maxsize=None)
    def rec(used: int) -> tuple[tuple[tuple[int, int, int], int], ...]:
        if used == full:
            return (((0, 0, 0), 1),)
        first = 0
        while used & (1 << first):
            first += 1
        totals: dict[tuple[int, int, int], int] = {}
        for placement in cover[first]:
            if used & placement["mask"]:
                continue
            add = [0, 0, 0]
            add[KIND_TO_COUNT_INDEX[placement["kind"]]] = 1
            for vec, count in rec(used | placement["mask"]):
                new_vec = (vec[0] + add[0], vec[1] + add[1], vec[2] + add[2])
                totals[new_vec] = min(3, totals.get(new_vec, 0) + int(count))
        return tuple(sorted(totals.items()))

    return dict(rec(0))


def _solve_unique(mask: np.ndarray, demand: tuple[int, int, int]) -> list[dict] | None:
    key = _shape_key(mask)
    cells, cover = _placements_for_shape(key)
    full = (1 << len(cells)) - 1
    total_cells = len(cells)

    @lru_cache(maxsize=None)
    def count(used: int, sq: int, hb: int, vb: int) -> int:
        if min(sq, hb, vb) < 0:
            return 0
        uncovered = total_cells - used.bit_count()
        if uncovered != 4 * sq + 3 * hb + 3 * vb:
            return 0
        if used == full:
            return 1 if (sq, hb, vb) == (0, 0, 0) else 0
        first = 0
        while used & (1 << first):
            first += 1
        total = 0
        for placement in cover[first]:
            if used & placement["mask"]:
                continue
            dsq, dhb, dvb = sq, hb, vb
            idx = KIND_TO_COUNT_INDEX[placement["kind"]]
            if idx == 0:
                dsq -= 1
            elif idx == 1:
                dhb -= 1
            else:
                dvb -= 1
            total += count(used | placement["mask"], dsq, dhb, dvb)
            if total > 1:
                return 2
        return total

    if count(0, *demand) != 1:
        return None

    def build(used: int, sq: int, hb: int, vb: int) -> list[dict]:
        if used == full:
            return []
        first = 0
        while used & (1 << first):
            first += 1
        for placement in cover[first]:
            if used & placement["mask"]:
                continue
            dsq, dhb, dvb = sq, hb, vb
            idx = KIND_TO_COUNT_INDEX[placement["kind"]]
            if idx == 0:
                dsq -= 1
            elif idx == 1:
                dhb -= 1
            else:
                dvb -= 1
            if count(used | placement["mask"], dsq, dhb, dvb):
                return [placement] + build(used | placement["mask"], dsq, dhb, dvb)
        raise AssertionError("unique solution vanished")

    return build(0, *demand)


def _render_solution(
    grid: np.ndarray,
    wall_holes: list[tuple[np.ndarray, str]],
    gray_offset: tuple[int, int],
    solution: list[dict],
) -> np.ndarray:
    out = grid.copy()
    for coords, kind in wall_holes:
        out[coords[:, 0], coords[:, 1]] = PIECE_COLORS[kind]
    gr0, gc0 = gray_offset
    out[out == GRAY] = BLACK
    for placement in solution:
        color = PIECE_COLORS[placement["kind"]]
        for r, c in placement["coords"]:
            out[gr0 + r, gc0 + c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the wall-hole inventory to uniquely decompose the gray block, recolor the pieces, and fill the holes.

    Input structure:
    - Black background contains one solid orange rectangular wall with several black holes.
    - Every hole is exactly a 2x2 square, a horizontal 1x3 bar, or a vertical 3x1 bar.
    - Elsewhere there is one 4-connected gray block whose pixels admit multiple legal decompositions
      into those same primitive shapes.

    Rule:
    - Count wall holes by shape to obtain the demanded numbers of 2x2 squares, horizontal 1x3 bars,
      and vertical 3x1 bars.
    - Find the unique decomposition of the gray block whose piece counts match that demand.
    - Recolor matched gray pieces in place: 2x2 squares teal, horizontal bars red, vertical bars blue.
    - Fill each wall hole with the same color determined by its own shape; keep the remaining wall and
      background unchanged.

    Key insight:
    - The gray block alone is underconstrained: the decisive information is the remote hole inventory,
      so the solver must reverse the causal direction and let the wall decide where the cuts go.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _, wall_holes = _find_wall_and_holes(g)
    demand = (
        sum(kind == SQUARE for _coords, kind in wall_holes),
        sum(kind == HBAR for _coords, kind in wall_holes),
        sum(kind == VBAR for _coords, kind in wall_holes),
    )
    gray_mask, gray_offset = _gray_component(g)
    solution = _solve_unique(gray_mask, demand)
    assert solution is not None, "demand must select exactly one decomposition"
    return _render_solution(g, wall_holes, gray_offset, solution)


def _mask_from_strings(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "#" for ch in row] for row in rows], dtype=bool)


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    out = np.rot90(out, random.randint(0, 3))
    return out


def _weighted_demand(mask: np.ndarray) -> tuple[int, int, int] | None:
    stats = _count_vectors(mask)
    total_tilings = sum(stats.values())
    if total_tilings <= 1:
        return None
    candidates = [vec for vec, count in stats.items() if count == 1]
    if not candidates:
        return None

    weights = []
    for sq, hb, vb in candidates:
        types = int(sq > 0) + int(hb > 0) + int(vb > 0)
        if types == 3:
            weights.append(8)
        elif sq > 0 and types == 2:
            weights.append(5)
        else:
            weights.append(2)
    return random.choices(candidates, weights=weights, k=1)[0]


def _hole_rects(demand: tuple[int, int, int]) -> list[tuple[str, int, int]]:
    pieces = [(SQUARE, 2, 2)] * demand[0]
    pieces += [(HBAR, 1, 3)] * demand[1]
    pieces += [(VBAR, 3, 1)] * demand[2]
    random.shuffle(pieces)
    return pieces


def _pack_holes(demand: tuple[int, int, int]) -> np.ndarray | None:
    pieces = _hole_rects(demand)
    max_h = max(h for _kind, h, _w in pieces)
    max_w = max(w for _kind, _h, w in pieces)
    total_area = sum(h * w for _kind, h, w in pieces)

    for _ in range(120):
        inner_h = random.randint(max_h + 1, min(12, max_h + 4 + len(pieces)))
        inner_w = random.randint(max_w + 1, min(14, max_w + 4 + len(pieces)))
        if inner_h * inner_w < total_area + len(pieces):
            continue
        wall = np.full((inner_h + 2, inner_w + 2), ORANGE, dtype=int)
        blocked = np.zeros((inner_h, inner_w), dtype=bool)
        actual_holes: list[tuple[int, int, int, int]] = []
        ok = True
        for kind, ph, pw in pieces:
            choices = []
            for r in range(inner_h - ph + 1):
                for c in range(inner_w - pw + 1):
                    rr0 = max(0, r - 1)
                    cc0 = max(0, c - 1)
                    rr1 = min(inner_h, r + ph + 1)
                    cc1 = min(inner_w, c + pw + 1)
                    if blocked[rr0:rr1, cc0:cc1].any():
                        continue
                    choices.append((r, c))
            if not choices:
                ok = False
                break
            r, c = random.choice(choices)
            actual_holes.append((r + 1, c + 1, ph, pw))
            rr0 = max(0, r - 1)
            cc0 = max(0, c - 1)
            rr1 = min(inner_h, r + ph + 1)
            cc1 = min(inner_w, c + pw + 1)
            blocked[rr0:rr1, cc0:cc1] = True

        if not ok:
            continue
        for r, c, ph, pw in actual_holes:
            wall[r : r + ph, c : c + pw] = BLACK
        return wall
    return None


def _place_scene(wall: np.ndarray, gray_mask: np.ndarray) -> np.ndarray | None:
    gray_h, gray_w = gray_mask.shape
    wall_h, wall_w = wall.shape
    gap = random.randint(2, 5)
    pad_top = random.randint(1, 3)
    pad_left = random.randint(1, 3)
    pad_bottom = random.randint(1, 3)
    pad_right = random.randint(1, 3)

    layout = random.choice(["horizontal", "vertical"])
    if layout == "horizontal":
        middle_h = max(wall_h, gray_h) + random.randint(0, 2)
        middle_w = wall_w + gray_w + gap
        if pad_top + middle_h + pad_bottom > 30 or pad_left + middle_w + pad_right > 30:
            return None
        grid = np.full((pad_top + middle_h + pad_bottom, pad_left + middle_w + pad_right), BLACK, dtype=int)
        if random.random() < 0.5:
            wall_c = pad_left
            gray_c = pad_left + wall_w + gap
        else:
            gray_c = pad_left
            wall_c = pad_left + gray_w + gap
        wall_r = pad_top + random.randint(0, middle_h - wall_h)
        gray_r = pad_top + random.randint(0, middle_h - gray_h)
    else:
        middle_h = wall_h + gray_h + gap
        middle_w = max(wall_w, gray_w) + random.randint(0, 2)
        if pad_top + middle_h + pad_bottom > 30 or pad_left + middle_w + pad_right > 30:
            return None
        grid = np.full((pad_top + middle_h + pad_bottom, pad_left + middle_w + pad_right), BLACK, dtype=int)
        if random.random() < 0.5:
            wall_r = pad_top
            gray_r = pad_top + wall_h + gap
        else:
            gray_r = pad_top
            wall_r = pad_top + gray_h + gap
        wall_c = pad_left + random.randint(0, middle_w - wall_w)
        gray_c = pad_left + random.randint(0, middle_w - gray_w)

    grid[wall_r : wall_r + wall_h, wall_c : wall_c + wall_w] = wall
    gray_rows, gray_cols = np.where(gray_mask)
    grid[gray_r + gray_rows, gray_c + gray_cols] = GRAY
    return grid


def _semantic_valid(grid: np.ndarray) -> bool:
    try:
        _, wall_holes = _find_wall_and_holes(grid)
        gray_mask, gray_offset = _gray_component(grid)
    except AssertionError:
        return False

    stats = _count_vectors(gray_mask)
    total_tilings = sum(stats.values())
    if total_tilings <= 1:
        return False
    demand = (
        sum(kind == SQUARE for _coords, kind in wall_holes),
        sum(kind == HBAR for _coords, kind in wall_holes),
        sum(kind == VBAR for _coords, kind in wall_holes),
    )
    if stats.get(demand, 0) != 1:
        return False
    solution = _solve_unique(gray_mask, demand)
    if solution is None:
        return False
    expected = _render_solution(grid, wall_holes, gray_offset, solution)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    raw = random.choice(TEMPLATE_STRINGS)
    if random.random() < 0.6:
        raw = random.choice(TEMPLATE_STRINGS[-2:])
    gray_mask = _random_transform(_mask_from_strings(raw))
    demand = _weighted_demand(gray_mask)
    if demand is None:
        return None
    wall = _pack_holes(demand)
    if wall is None:
        return None
    grid = _place_scene(wall, gray_mask)
    if grid is None:
        return None
    if not _semantic_valid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
