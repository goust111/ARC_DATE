import random
from collections import deque

import numpy as np


RED = 2
GREY = 5
PATTERN_COLORS = [1, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 500


def _neighbors4(h: int, w: int, r: int, c: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _bool_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(h, w, rr, cc):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _pattern_from_period(
    grid: np.ndarray, visible_mask: np.ndarray, period_r: int, period_c: int
) -> np.ndarray | None:
    tile = np.full((period_r, period_c), -1, dtype=int)
    seen = np.zeros((period_r, period_c), dtype=bool)
    coords = np.argwhere(visible_mask)
    for r, c in coords:
        rr = int(r) % period_r
        cc = int(c) % period_c
        color = int(grid[r, c])
        if seen[rr, cc]:
            if int(tile[rr, cc]) != color:
                return None
        else:
            tile[rr, cc] = color
            seen[rr, cc] = True
    if not bool(np.all(seen)):
        return None
    return tile


def _infer_tile(grid: np.ndarray) -> np.ndarray:
    visible = (grid != RED) & (grid != GREY)
    assert bool(np.any(visible)), "expected some visible pattern pixels"
    h, w = grid.shape

    best_key = None
    best_tile = None
    for period_r in range(1, h + 1):
        for period_c in range(1, w + 1):
            tile = _pattern_from_period(grid, visible, period_r, period_c)
            if tile is None:
                continue
            key = (period_r * period_c, period_r + period_c, period_r, period_c)
            if best_key is None or key < best_key:
                best_key = key
                best_tile = tile

    assert best_tile is not None, "failed to infer periodic tile"
    return best_tile


def _tile_to_grid(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    out = np.empty((h, w), dtype=int)
    tr, tc = tile.shape
    for r in range(h):
        for c in range(w):
            out[r, c] = int(tile[r % tr, c % tc])
    return out


def _restoration_mask(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    passable = grid != RED
    red_mask = grid == RED
    restore = np.zeros((h, w), dtype=bool)
    for comp in _bool_components(passable):
        touches_border = False
        has_grey = False
        for r, c in comp:
            if r == 0 or r == h - 1 or c == 0 or c == w - 1:
                touches_border = True
            if int(grid[r, c]) == GREY:
                has_grey = True
        if touches_border or not has_grey:
            continue

        red_seeds = []
        for r, c in comp:
            restore[r, c] = True
            for nr, nc in _neighbors4(h, w, r, c):
                if red_mask[nr, nc]:
                    red_seeds.append((nr, nc))

        red_seen = set(red_seeds)
        q = deque(red_seeds)
        while q:
            r, c = q.popleft()
            restore[r, c] = True
            for nr, nc in _neighbors4(h, w, r, c):
                if red_mask[nr, nc] and (nr, nc) not in red_seen:
                    red_seen.add((nr, nc))
                    q.append((nr, nc))
    return restore


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore only the gray damage that sits inside closed red outlines by reconstructing the hidden periodic tiling underneath.

    Input structure:
    - The canvas is covered by a repeating rectangular color tile whose colors never include red or gray.
    - Gray connected blotches occlude parts of that periodic pattern.
    - Some blotches are surrounded by one-cell-thick red outlines, and those outlines may be closed or may contain a gap.

    Rule:
    - Infer the smallest periodic tile and its phase from all visible non-red, non-gray pixels.
    - Treat red pixels as walls and find every non-red region that is fully cut off from the grid boundary.
    - If such an enclosed region contains gray damage, restore that whole enclosed region and its adjacent red outline from the inferred periodic pattern.
    - Leave gray damage outside closed outlines, and leave incomplete red outlines, unchanged.

    Key insight:
    - Recovery is gated by enclosure: the periodic background must be inferred globally first, but only the gray damage inside boundary-disconnected regions approved by closed red walls is actually repaired.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    tile = _infer_tile(g)
    periodic = _tile_to_grid(tile, g.shape)
    restore = _restoration_mask(g)

    out = g.copy()
    out[restore] = periodic[restore]
    return out


def _has_smaller_period(tile: np.ndarray) -> bool:
    tr, tc = tile.shape
    for pr in range(1, tr + 1):
        for pc in range(1, tc + 1):
            if pr == tr and pc == tc:
                continue
            ok = True
            for r in range(tr):
                for c in range(tc):
                    if int(tile[r, c]) != int(tile[r % pr, c % pc]):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return True
    return False


def _sample_tile() -> np.ndarray:
    for _ in range(200):
        tr = random.randint(2, 4)
        tc = random.randint(2, 4)
        palette = random.sample(PATTERN_COLORS, random.randint(3, min(5, len(PATTERN_COLORS))))
        tile = np.random.choice(palette, size=(tr, tc)).astype(int)
        if len(set(map(int, tile.flat))) < 3:
            continue
        if any(len(set(map(int, row))) == 1 for row in tile):
            continue
        if any(len(set(map(int, tile[:, c]))) == 1 for c in range(tc)):
            continue
        if _has_smaller_period(tile):
            continue
        return tile
    raise RuntimeError("failed to sample a unique periodic tile")


def _make_base_pattern() -> tuple[np.ndarray, np.ndarray]:
    tile = _sample_tile()
    tr, tc = tile.shape
    h = random.randint(max(12, tr * 4), min(24, tr * 7))
    w = random.randint(max(12, tc * 4), min(24, tc * 7))
    off_r = random.randrange(tr)
    off_c = random.randrange(tc)

    base = np.empty((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            base[r, c] = int(tile[(r + off_r) % tr, (c + off_c) % tc])
    return base, tile


def _connected_blob(height: int, width: int) -> np.ndarray:
    area = random.randint(2, max(2, height * width - 1))
    cells = {(random.randrange(height), random.randrange(width))}
    frontier = set(cells)
    while len(cells) < area and frontier:
        r, c = random.choice(tuple(frontier))
        candidates = []
        for nr, nc in _neighbors4(height, width, r, c):
            if (nr, nc) not in cells:
                candidates.append((nr, nc))
        if not candidates:
            frontier.discard((r, c))
            continue
        nxt = random.choice(candidates)
        cells.add(nxt)
        frontier.add(nxt)

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _place_box(
    occupied: np.ndarray, box_h: int, box_w: int, margin: int = 2, padding: int = 2
) -> tuple[int, int] | None:
    h, w = occupied.shape
    for _ in range(200):
        r0 = random.randint(margin, h - box_h - margin)
        c0 = random.randint(margin, w - box_w - margin)
        rr0 = max(0, r0 - padding)
        cc0 = max(0, c0 - padding)
        rr1 = min(h, r0 + box_h + padding)
        cc1 = min(w, c0 + box_w + padding)
        if np.any(occupied[rr0:rr1, cc0:cc1]):
            continue
        occupied[r0 : r0 + box_h, c0 : c0 + box_w] = True
        return r0, c0
    return None


def _add_ring_group(
    inp: np.ndarray,
    expected: np.ndarray,
    occupied: np.ndarray,
    *,
    closed: bool,
) -> bool:
    inner_h = random.randint(3, 6)
    inner_w = random.randint(3, 6)
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    pos = _place_box(occupied, outer_h, outer_w)
    if pos is None:
        return False
    r0, c0 = pos
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1

    inp[r0, c0 : c1 + 1] = RED
    inp[r1, c0 : c1 + 1] = RED
    inp[r0 : r1 + 1, c0] = RED
    inp[r0 : r1 + 1, c1] = RED

    if not closed:
        side = random.choice(["top", "bottom", "left", "right"])
        if side in ("top", "bottom"):
            gap_c = random.randint(c0 + 1, c1 - 1)
            gap_r = r0 if side == "top" else r1
            inp[gap_r, gap_c] = expected[gap_r, gap_c]
        else:
            gap_r = random.randint(r0 + 1, r1 - 1)
            gap_c = c0 if side == "left" else c1
            inp[gap_r, gap_c] = expected[gap_r, gap_c]

    blob = _connected_blob(inner_h, inner_w)
    inner = inp[r0 + 1 : r1, c0 + 1 : c1]
    inner[blob] = GREY

    if not closed:
        expected[r0 : r1 + 1, c0 : c1 + 1] = inp[r0 : r1 + 1, c0 : c1 + 1]
    return True


def _add_bare_blob(inp: np.ndarray, expected: np.ndarray, occupied: np.ndarray) -> bool:
    box_h = random.randint(3, 6)
    box_w = random.randint(3, 6)
    pos = _place_box(occupied, box_h, box_w)
    if pos is None:
        return False
    r0, c0 = pos
    blob = _connected_blob(box_h, box_w)
    patch = inp[r0 : r0 + box_h, c0 : c0 + box_w]
    patch[blob] = GREY
    expected[r0 : r0 + box_h, c0 : c0 + box_w] = patch
    return True


def _all_residues_visible(grid: np.ndarray, tile_shape: tuple[int, int]) -> bool:
    visible = (grid != RED) & (grid != GREY)
    residues = set()
    tr, tc = tile_shape
    for r, c in np.argwhere(visible):
        residues.add((int(r) % tr, int(c) % tc))
    return len(residues) == tr * tc


def _gen_attempt() -> np.ndarray | None:
    base, tile = _make_base_pattern()
    inp = base.copy()
    expected = base.copy()
    occupied = np.zeros(base.shape, dtype=bool)

    if not _add_ring_group(inp, expected, occupied, closed=True):
        return None

    unrestored_kind = random.choice(["open", "bare"])
    if unrestored_kind == "open":
        if not _add_ring_group(inp, expected, occupied, closed=False):
            return None
    else:
        if not _add_bare_blob(inp, expected, occupied):
            return None

    extra_groups = random.randint(0, 2)
    for _ in range(extra_groups):
        kind = random.choice(["closed", "open", "bare"])
        ok = (
            _add_ring_group(inp, expected, occupied, closed=True)
            if kind == "closed"
            else _add_ring_group(inp, expected, occupied, closed=False)
            if kind == "open"
            else _add_bare_blob(inp, expected, occupied)
        )
        if not ok:
            return None

    if not np.any(inp == GREY) or not np.any(inp == RED):
        return None
    if not np.any(expected == GREY):
        return None
    if np.array_equal(inp, expected):
        return None
    if not _all_residues_visible(inp, tile.shape):
        return None

    inferred = _infer_tile(inp)
    if inferred.shape != tile.shape:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
