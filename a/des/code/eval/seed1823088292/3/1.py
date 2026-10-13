import random
from typing import Iterable

import numpy as np


BG = 0
GREY = 5
PALETTE_H = 4
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]

ORIENTATION_TO_INDEX = {"ul": 0, "ur": 1, "ll": 2, "lr": 3}
INDEX_TO_OFFSETS = {
    0: [(0, 0), (0, 1), (1, 0)],
    1: [(0, 1), (0, 2), (1, 2)],
    2: [(1, 0), (2, 0), (2, 1)],
    3: [(1, 2), (2, 1), (2, 2)],
}

GEN_RETRIES = 400
MIN_INNER_H = 3
MAX_INNER_H = 5
MIN_INNER_W = 3
MAX_INNER_W = 6
MIN_FRAMES = 1
MAX_FRAMES = 4
MIN_GRID_H = 14
MAX_GRID_H = 24
MAX_GRID_W = 34
PALETTE_SIDES = ["top", "bottom", "left", "right"]


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
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(points)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _frame_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _find_palette(grid: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    mask = (grid != BG) & (grid != GREY)
    h, w = grid.shape
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for comp in _components(mask):
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 + 1 != PALETTE_H:
            continue
        patch_mask = mask[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(patch_mask):
            continue
        candidates.append((patch_mask.size, (r0, c0, r1, c1)))
    assert candidates, "palette not found"
    _, box = max(candidates, key=lambda item: item[0])
    r0, c0, r1, c1 = box
    return box, grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for color in np.unique(grid):
        color = int(color)
        if color in (BG, GREY):
            continue
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 5 or w < 5:
                continue
            patch = grid[r0 : r1 + 1, c0 : c1 + 1] == color
            if np.array_equal(patch, _frame_mask(h, w)):
                frames.append((r0, c0, r1, c1))
    frames.sort()
    assert frames, "no frames found"
    return frames


def _orientation_index(interior: np.ndarray) -> int:
    pts = [tuple(map(int, pt)) for pt in np.argwhere(interior == GREY)]
    assert len(pts) == 3, "L marker must have 3 grey cells"
    r0, c0, r1, c1 = _bbox(pts)
    assert r1 - r0 == 1 and c1 - c0 == 1, "L marker must occupy a 2x2 corner"
    block = np.zeros((2, 2), dtype=bool)
    for r, c in pts:
        block[r - r0, c - c0] = True
    patterns = {
        ((True, True), (True, False)): 0,
        ((True, True), (False, True)): 1,
        ((True, False), (True, True)): 2,
        ((False, True), (True, True)): 3,
    }
    key = tuple(tuple(bool(v) for v in row) for row in block)
    assert key in patterns, "invalid L marker shape"
    return patterns[key]


def _fill_frame(out: np.ndarray, box: tuple[int, int, int, int], palette_row: np.ndarray) -> None:
    r0, c0, r1, c1 = box
    interior = out[r0 + 1 : r1, c0 + 1 : c1]
    orientation = _orientation_index(interior)
    row = np.asarray(palette_row[orientation], dtype=int)
    targets = [
        (r0 + 1 + rr, c0 + 1 + cc)
        for rr in range(interior.shape[0])
        for cc in range(interior.shape[1])
        if interior[rr, cc] != GREY
    ]
    assert len(targets) == row.shape[0], "palette width does not match fill count"
    for idx, (r, c) in enumerate(targets):
        out[r, c] = int(row[idx])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each frame interior using the palette row selected by the grey L orientation.
    """
    g = np.asarray(grid, dtype=int)
    palette_box, palette = _find_palette(g)
    frames = _find_frames(g)

    out = g.copy()
    for frame_box in frames:
        _fill_frame(out, frame_box, palette)

    pr0, pc0, pr1, pc1 = palette_box
    out[pr0 : pr1 + 1, pc0 : pc1 + 1] = BG
    return out


def _random_palette_rows(rng: random.Random, width: int) -> np.ndarray:
    rows: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    while len(rows) < PALETTE_H:
        if width == 1:
            seq = [rng.choice(COLOR_POOL)]
        else:
            palette_size = rng.randint(2, min(4, len(COLOR_POOL)))
            base = rng.sample(COLOR_POOL, palette_size)
            seq = [rng.choice(base) for _ in range(width)]
            if len(set(seq)) == 1:
                alt = rng.choice([c for c in COLOR_POOL if c != seq[0]])
                seq[rng.randrange(width)] = alt
        key = tuple(seq)
        if key in seen:
            continue
        seen.add(key)
        rows.append(np.array(seq, dtype=int))
    return np.stack(rows, axis=0)


def _place_palette(grid: np.ndarray, occ: np.ndarray, palette: np.ndarray, side: str, rng: random.Random) -> tuple[int, int]:
    h, w = grid.shape
    ph, pw = palette.shape
    if side == "top":
        r0 = 0
        c0 = rng.randint(0, w - pw)
    elif side == "bottom":
        r0 = h - ph
        c0 = rng.randint(0, w - pw)
    elif side == "left":
        r0 = rng.randint(0, h - ph)
        c0 = 0
    else:
        r0 = rng.randint(0, h - ph)
        c0 = w - pw
    grid[r0 : r0 + ph, c0 : c0 + pw] = palette
    occ[max(0, r0 - 1) : min(h, r0 + ph + 1), max(0, c0 - 1) : min(w, c0 + pw + 1)] = True
    return r0, c0


def _candidate_positions(occ: np.ndarray, h: int, w: int) -> list[tuple[int, int]]:
    gh, gw = occ.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(1, gh - h):
        for c0 in range(1, gw - w):
            if occ[r0 - 1 : r0 + h + 1, c0 - 1 : c0 + w + 1].any():
                continue
            positions.append((r0, c0))
    return positions


def _paint_frame(grid: np.ndarray, top: int, left: int, inner_h: int, inner_w: int, color: int) -> None:
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    patch = np.full((outer_h, outer_w), BG, dtype=int)
    border = _frame_mask(outer_h, outer_w)
    patch[border] = color
    grid[top : top + outer_h, left : left + outer_w] = patch


def _paint_l_marker(grid: np.ndarray, top: int, left: int, orientation: int) -> None:
    for dr, dc in INDEX_TO_OFFSETS[orientation]:
        grid[top + dr, left + dc] = GREY


def _expected_from_metadata(
    inp: np.ndarray,
    palette_box: tuple[int, int, int, int],
    palette: np.ndarray,
    frames: list[dict[str, int]],
) -> np.ndarray:
    out = inp.copy()
    for info in frames:
        r0 = info["r0"]
        c0 = info["c0"]
        outer_h = info["inner_h"] + 2
        outer_w = info["inner_w"] + 2
        box = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
        row = palette[info["orientation"]]
        targets = [
            (r0 + 1 + rr, c0 + 1 + cc)
            for rr in range(info["inner_h"])
            for cc in range(info["inner_w"])
            if out[r0 + 1 + rr, c0 + 1 + cc] != GREY
        ]
        for idx, (r, c) in enumerate(targets):
            out[r, c] = int(row[idx])
    pr0, pc0, pr1, pc1 = palette_box
    out[pr0 : pr1 + 1, pc0 : pc1 + 1] = BG
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    inner_h = rng.randint(MIN_INNER_H, MAX_INNER_H)
    inner_w = rng.randint(MIN_INNER_W, MAX_INNER_W)
    fill_len = inner_h * inner_w - 3
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    frame_count = rng.randint(MIN_FRAMES, MAX_FRAMES)
    side = rng.choice(PALETTE_SIDES)

    min_w = max(fill_len + 2, outer_w + 3)
    if min_w > MAX_GRID_W:
        return None
    h = rng.randint(max(MIN_GRID_H, outer_h + 3), MAX_GRID_H)
    w = rng.randint(min_w, MAX_GRID_W)

    grid = np.full((h, w), BG, dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    palette = _random_palette_rows(rng, fill_len)
    pr0, pc0 = _place_palette(grid, occ, palette, side, rng)
    palette_box = (pr0, pc0, pr0 + PALETTE_H - 1, pc0 + fill_len - 1)

    frame_colors = rng.sample(COLOR_POOL, frame_count)
    frames: list[dict[str, int]] = []
    for color in frame_colors:
        positions = _candidate_positions(occ, outer_h, outer_w)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        _paint_frame(grid, r0, c0, inner_h, inner_w, color)

        lr0 = r0 + 1 + rng.randint(0, inner_h - 3)
        lc0 = c0 + 1 + rng.randint(0, inner_w - 3)
        orientation = rng.randrange(4)
        _paint_l_marker(grid, lr0, lc0, orientation)

        occ[r0 - 1 : r0 + outer_h + 1, c0 - 1 : c0 + outer_w + 1] = True
        frames.append(
            {
                "r0": r0,
                "c0": c0,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "orientation": orientation,
            }
        )

    expected = _expected_from_metadata(grid, palette_box, palette, frames)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
