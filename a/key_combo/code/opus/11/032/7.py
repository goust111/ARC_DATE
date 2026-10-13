from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
TEAL = 8
MAROON = 9

SRC_POOL = (BLUE, RED, GREEN)
TGT_POOL = (ORANGE, TEAL, MAROON)
MAX_GENERATION_ATTEMPTS = 400
STRUCT4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbors4(r: int, c: int):
    for dr, dc in STRUCT4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize(coords: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _shape_patch(shape: tuple[tuple[int, int], ...], color: int) -> np.ndarray:
    h = 1 + max(r for r, _ in shape)
    w = 1 + max(c for _, c in shape)
    patch = np.zeros((h, w), dtype=int)
    for r, c in shape:
        patch[r, c] = int(color)
    return patch


def _find_layout(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    sep_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == YELLOW)]
    sep_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == YELLOW)]
    assert len(sep_rows) >= 3 and len(sep_cols) >= 3, "expected separator rows and cols with outer boundaries"

    row_gaps = [sep_rows[i + 1] - sep_rows[i] - 1 for i in range(len(sep_rows) - 1)]
    col_gaps = [sep_cols[i + 1] - sep_cols[i] - 1 for i in range(len(sep_cols) - 1)]
    assert min(row_gaps) > 0 and min(col_gaps) > 0, "separators must leave non-empty cells"
    assert len(set(row_gaps)) == 1 and len(set(col_gaps)) == 1, "all cells must have equal size"

    row_keep = [r for r in range(sep_rows[0] + 1, sep_rows[-1]) if r not in set(sep_rows)]
    col_keep = [c for c in range(sep_cols[0] + 1, sep_cols[-1]) if c not in set(sep_cols)]
    compressed = g[np.ix_(row_keep, col_keep)].copy()

    outside = np.zeros_like(g, dtype=bool)
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if r < sep_rows[0] or r > sep_rows[-1] or c < sep_cols[0] or c > sep_cols[-1]:
                outside[r, c] = True
    outside &= g != YELLOW

    return {
        "grid": g,
        "sep_rows": sep_rows,
        "sep_cols": sep_cols,
        "cell_h": row_gaps[0],
        "cell_w": col_gaps[0],
        "compressed": compressed,
        "outside": outside,
    }


def _parse_legends(grid: np.ndarray, outside: np.ndarray) -> dict[int, int]:
    g = np.asarray(grid, dtype=int)
    singletons: dict[tuple[int, int], int] = {}
    for color in sorted(int(c) for c in np.unique(g[outside]) if int(c) not in (BLACK, GREY, YELLOW)):
        comps = _components((g == color) & outside)
        for comp in comps:
            if len(comp) == 1:
                r, c = map(int, comp[0])
                singletons[(r, c)] = color

    legends: dict[int, int] = {}
    used_positions: set[tuple[int, int]] = set()
    for (r, c), src in sorted(singletons.items()):
        if src not in SRC_POOL:
            continue
        below = (r + 1, c)
        if below not in singletons:
            continue
        tgt = int(singletons[below])
        assert tgt not in (BLACK, GREY, YELLOW) and tgt != src, "invalid legend target"
        assert (r, c) not in used_positions and below not in used_positions, "legend pixels must not overlap"
        assert src not in legends, "duplicate legend for source color"
        legends[src] = tgt
        used_positions.add((r, c))
        used_positions.add(below)

    assert len(legends) >= 2, "expected at least two color legends"
    return legends


def _parse_sprites(grid: np.ndarray, outside: np.ndarray, src_colors: set[int]) -> dict[int, tuple[tuple[int, int], ...]]:
    g = np.asarray(grid, dtype=int)
    sprites: dict[int, tuple[tuple[int, int], ...]] = {}
    for color in sorted(src_colors):
        comps = _components((g == color) & outside)
        big = [comp for comp in comps if 3 <= len(comp) <= 5]
        assert len(big) == 1, f"expected one sprite component for color {color}"
        comp = big[0]
        r0, c0, r1, c1 = _bbox(comp)
        assert r1 - r0 + 1 <= 3 and c1 - c0 + 1 <= 3, "sprite bbox too large"
        rel = tuple(sorted((int(r - r0), int(c - c0)) for r, c in comp))
        sprites[color] = rel
    assert set(sprites) == set(src_colors), "missing sprite colors"
    return sprites


def _stamp(
    compressed: np.ndarray,
    sprites: dict[int, tuple[tuple[int, int], ...]],
    legends: dict[int, int],
) -> np.ndarray:
    base = compressed.copy()
    placeholders = [(int(r), int(c), int(base[r, c])) for r, c in np.argwhere(np.isin(base, list(legends.keys())))]
    base[base == GREY] = BLACK
    base[np.isin(base, list(legends.keys()))] = BLACK

    painted = np.zeros(base.shape, dtype=bool)
    for r, c, src in placeholders:
        shape = sprites[src]
        tgt = legends[src]
        for dr, dc in shape:
            rr = r + dr
            cc = c + dc
            assert 0 <= rr < base.shape[0] and 0 <= cc < base.shape[1], "sprite stamp out of bounds"
            assert not painted[rr, cc], "sprite stamps must not overlap"
            painted[rr, cc] = True
            base[rr, cc] = tgt
    return base


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the separator grid, erase grey noise, then stamp and recolor the edge sprites at placeholder anchors.

    Input structure:
    - Full yellow rows and columns form a regular cell lattice with an outer separator boundary around the playable board.
    - Inside the lattice, cells are mostly black, with 2-3 colored placeholder pixels and 1-2 grey noise pixels.
    - Outside the lattice, the edge area contains one monochrome sprite per placeholder color and a vertical 2-pixel legend mapping each placeholder color to an output color.

    Rule:
    - Find the full yellow separator rows and columns, then remove them to compress the board coordinates.
    - Keep only the compressed board between the outermost separators, and clear every grey pixel to black.
    - For each placeholder color, locate the same-colored outside sprite and stamp it on the compressed board with the placeholder pixel as the sprite's top-left anchor.
    - Recolor each stamped sprite using the legend pair whose top pixel is the placeholder color and whose bottom pixel is the output color.

    Key insight:
    The output depends on four linked cues at once: separators define the compressed coordinates, placeholder colors choose the sprite shape, legends choose the final color, and grey pixels are pure distractors that must be erased before stamping.
    """
    layout = _find_layout(np.asarray(grid, dtype=int))
    legends = _parse_legends(layout["grid"], layout["outside"])
    sprites = _parse_sprites(layout["grid"], layout["outside"], set(legends))
    compressed = layout["compressed"]
    placeholder_colors = set(int(c) for c in np.unique(compressed) if int(c) in legends)
    assert placeholder_colors == set(legends), "placeholder colors must match legend colors"
    return _stamp(compressed, sprites, legends)


def _random_shape(rng: random.Random) -> tuple[tuple[int, int], ...]:
    for _ in range(100):
        area = rng.randint(3, 5)
        coords = {(rng.randrange(3), rng.randrange(3))}
        while len(coords) < area:
            frontier = []
            for r, c in coords:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < 3 and 0 <= nc < 3 and (nr, nc) not in coords:
                        frontier.append((nr, nc))
            if not frontier:
                break
            coords.add(rng.choice(frontier))
        rel = _normalize(coords)
        max_r = max(r for r, _ in rel)
        max_c = max(c for _, c in rel)
        if max_r <= 2 and max_c <= 2 and 3 <= len(rel) <= 5:
            return rel
    raise RuntimeError("failed to generate sprite shape")


def _choose_placeholder_layout(
    rng: random.Random,
    colors: list[int],
    sprites: dict[int, tuple[tuple[int, int], ...]],
    cell_rows: int,
    cell_cols: int,
    cell_h: int,
    cell_w: int,
) -> tuple[dict[int, tuple[int, int]], set[tuple[int, int]]]:
    comp_h = cell_rows * cell_h
    comp_w = cell_cols * cell_w
    cells = [(rr, cc) for rr in range(cell_rows) for cc in range(cell_cols)]
    rng.shuffle(cells)

    placements: dict[int, tuple[int, int]] = {}
    occupied: set[tuple[int, int]] = set()
    used_cells: set[tuple[int, int]] = set()

    for color in colors:
        shape = sprites[color]
        sh = 1 + max(r for r, _ in shape)
        sw = 1 + max(c for _, c in shape)
        candidates = []
        for cell_r, cell_c in cells:
            if (cell_r, cell_c) in used_cells:
                continue
            for r in range(cell_r * cell_h, (cell_r + 1) * cell_h):
                for c in range(cell_c * cell_w, (cell_c + 1) * cell_w):
                    if r + sh > comp_h or c + sw > comp_w:
                        continue
                    footprint = {(r + dr, c + dc) for dr, dc in shape}
                    if footprint & occupied:
                        continue
                    candidates.append((cell_r, cell_c, r, c, footprint))
        if not candidates:
            raise RuntimeError("failed to place non-overlapping placeholders")
        cell_r, cell_c, r, c, footprint = rng.choice(candidates)
        used_cells.add((cell_r, cell_c))
        placements[color] = (r, c)
        occupied |= footprint

    return placements, occupied


def _choose_greys(
    rng: random.Random,
    count: int,
    cell_rows: int,
    cell_cols: int,
    cell_h: int,
    cell_w: int,
    used_anchor_cells: set[tuple[int, int]],
    stamp_footprints: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    grey_positions: list[tuple[int, int]] = []
    free_cells = [(rr, cc) for rr in range(cell_rows) for cc in range(cell_cols) if (rr, cc) not in used_anchor_cells]
    rng.shuffle(free_cells)
    for cell_r, cell_c in free_cells:
        local = []
        for r in range(cell_r * cell_h, (cell_r + 1) * cell_h):
            for c in range(cell_c * cell_w, (cell_c + 1) * cell_w):
                if (r, c) not in stamp_footprints:
                    local.append((r, c))
        if local:
            grey_positions.append(rng.choice(local))
        if len(grey_positions) == count:
            break
    if len(grey_positions) != count:
        raise RuntimeError("failed to place grey distractors")
    return grey_positions


def _compressed_input(
    comp_shape: tuple[int, int],
    placements: dict[int, tuple[int, int]],
    grey_positions: list[tuple[int, int]],
) -> np.ndarray:
    board = np.zeros(comp_shape, dtype=int)
    for color, (r, c) in placements.items():
        board[r, c] = int(color)
    for r, c in grey_positions:
        board[r, c] = GREY
    return board


def _render_expected(
    comp_shape: tuple[int, int],
    placements: dict[int, tuple[int, int]],
    sprites: dict[int, tuple[tuple[int, int], ...]],
    legends: dict[int, int],
) -> np.ndarray:
    out = np.zeros(comp_shape, dtype=int)
    for src, (r, c) in placements.items():
        for dr, dc in sprites[src]:
            out[r + dr, c + dc] = legends[src]
    return out


def _lift_to_raw(
    compressed: np.ndarray,
    cell_rows: int,
    cell_cols: int,
    cell_h: int,
    cell_w: int,
    margins: tuple[int, int, int, int],
) -> tuple[np.ndarray, list[int], list[int]]:
    top, bottom, left, right = margins
    board_h = cell_rows * cell_h + (cell_rows + 1)
    board_w = cell_cols * cell_w + (cell_cols + 1)
    h = top + board_h + bottom
    w = left + board_w + right
    grid = np.zeros((h, w), dtype=int)

    sep_rows = [top + i * (cell_h + 1) for i in range(cell_rows + 1)]
    sep_cols = [left + i * (cell_w + 1) for i in range(cell_cols + 1)]
    grid[sep_rows, :] = YELLOW
    grid[:, sep_cols] = YELLOW

    keep_rows = [r for r in range(sep_rows[0] + 1, sep_rows[-1]) if r not in set(sep_rows)]
    keep_cols = [c for c in range(sep_cols[0] + 1, sep_cols[-1]) if c not in set(sep_cols)]
    grid[np.ix_(keep_rows, keep_cols)] = compressed
    return grid, sep_rows, sep_cols


def _can_place_patch(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    h, w = canvas.shape
    ph, pw = patch.shape
    if r0 < 0 or c0 < 0 or r0 + ph > h or c0 + pw > w:
        return False

    if np.any((patch != BLACK) & (canvas[r0 : r0 + ph, c0 : c0 + pw] != BLACK)):
        return False

    occupied = (canvas != BLACK) & (canvas != YELLOW)
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h, r0 + ph + 1)
    cc1 = min(w, c0 + pw + 1)
    if np.any(occupied[rr0:rr1, cc0:cc1]):
        return False
    return True


def _paste(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    ph, pw = patch.shape
    view = canvas[r0 : r0 + ph, c0 : c0 + pw]
    mask = patch != BLACK
    view[mask] = patch[mask]


def _outside_zones(shape: tuple[int, int], sep_rows: list[int], sep_cols: list[int]):
    h, w = shape
    return [
        (0, 0, sep_rows[0], w),
        (sep_rows[-1] + 1, 0, h, w),
        (sep_rows[0] + 1, 0, sep_rows[-1], sep_cols[0]),
        (sep_rows[0] + 1, sep_cols[-1] + 1, sep_rows[-1], w),
    ]


def _place_outside_items(
    rng: random.Random,
    grid: np.ndarray,
    sep_rows: list[int],
    sep_cols: list[int],
    sprites: dict[int, tuple[tuple[int, int], ...]],
    legends: dict[int, int],
) -> None:
    items = []
    for src, shape in sprites.items():
        items.append(("sprite", src, _shape_patch(shape, src)))
    for src, tgt in legends.items():
        items.append(("legend", src, np.array([[src], [tgt]], dtype=int)))
    items.sort(key=lambda item: item[2].shape[0] * item[2].shape[1], reverse=True)

    zones = _outside_zones(grid.shape, sep_rows, sep_cols)
    for kind, key, patch in items:
        candidates = []
        zone_order = list(range(len(zones)))
        rng.shuffle(zone_order)
        for zi in zone_order:
            r0, c0, r1, c1 = zones[zi]
            ph, pw = patch.shape
            for r in range(r0, r1 - ph + 1):
                for c in range(c0, c1 - pw + 1):
                    if _can_place_patch(grid, patch, r, c):
                        candidates.append((zi, r, c))
        if not candidates:
            raise RuntimeError(f"failed to place outside {kind} {key}")
        _, r, c = rng.choice(candidates)
        _paste(grid, patch, r, c)


def _validate_instance(
    inp: np.ndarray,
    expected: np.ndarray,
    active_colors: list[int],
    sprites: dict[int, tuple[tuple[int, int], ...]],
    legends: dict[int, int],
) -> bool:
    try:
        out = main(inp)
    except Exception:
        return False
    if not np.array_equal(out, expected):
        return False

    layout = _find_layout(inp)
    compressed = layout["compressed"]
    if set(int(c) for c in np.unique(compressed) if int(c) in SRC_POOL) != set(active_colors):
        return False
    if int(np.count_nonzero(compressed == GREY)) == 0:
        return False

    no_clear = compressed.copy()
    no_clear[np.isin(no_clear, active_colors)] = BLACK
    for src in active_colors:
        for r, c in np.argwhere(compressed == src):
            for dr, dc in sprites[src]:
                no_clear[int(r) + dr, int(c) + dc] = legends[src]
    if np.array_equal(no_clear, expected):
        return False

    shape_set = {sprites[src] for src in active_colors}
    if len(shape_set) != len(active_colors):
        return False
    if len(set(legends.values())) != len(active_colors):
        return False

    swapped = {src: legends[src] for src in active_colors}
    first, second = active_colors[0], active_colors[1]
    swapped[first], swapped[second] = swapped[second], swapped[first]
    try:
        if np.array_equal(_stamp(compressed, sprites, swapped), expected):
            return False
    except AssertionError:
        pass

    swapped_sprites = {src: sprites[src] for src in active_colors}
    swapped_sprites[first], swapped_sprites[second] = swapped_sprites[second], swapped_sprites[first]
    try:
        if np.array_equal(_stamp(compressed, swapped_sprites, legends), expected):
            return False
    except AssertionError:
        pass

    return True


def _generate_one(rng: random.Random) -> np.ndarray:
    cell_rows = rng.randint(3, 4)
    cell_cols = rng.randint(3, 4)
    cell_h = 3
    cell_w = 3
    k = rng.randint(2, 3)
    active_colors = rng.sample(list(SRC_POOL), k)
    target_colors = rng.sample(list(TGT_POOL), k)
    legends = {src: tgt for src, tgt in zip(active_colors, target_colors)}

    sprites: dict[int, tuple[tuple[int, int], ...]] = {}
    used_shapes: set[tuple[tuple[int, int], ...]] = set()
    for src in active_colors:
        for _ in range(100):
            shape = _random_shape(rng)
            if shape not in used_shapes:
                used_shapes.add(shape)
                sprites[src] = shape
                break
        if src not in sprites:
            raise RuntimeError("failed to find distinct sprites")

    placements, stamp_footprints = _choose_placeholder_layout(
        rng, active_colors, sprites, cell_rows, cell_cols, cell_h, cell_w
    )
    used_anchor_cells = {(r // cell_h, c // cell_w) for r, c in placements.values()}
    grey_count = rng.randint(1, 2)
    grey_positions = _choose_greys(
        rng, grey_count, cell_rows, cell_cols, cell_h, cell_w, used_anchor_cells, stamp_footprints
    )

    compressed_in = _compressed_input((cell_rows * cell_h, cell_cols * cell_w), placements, grey_positions)
    expected = _render_expected((cell_rows * cell_h, cell_cols * cell_w), placements, sprites, legends)

    margins = (
        rng.randint(4, 6),
        rng.randint(4, 6),
        rng.randint(4, 6),
        rng.randint(4, 6),
    )
    raw, sep_rows, sep_cols = _lift_to_raw(compressed_in, cell_rows, cell_cols, cell_h, cell_w, margins)
    _place_outside_items(rng, raw, sep_rows, sep_cols, sprites, legends)

    if max(raw.shape) > 30 or max(expected.shape) > 30:
        raise RuntimeError("grid exceeds ARC size limit")
    if not _validate_instance(raw, expected, active_colors, sprites, legends):
        raise RuntimeError("generated instance failed validation")
    return raw


def generate_input() -> np.ndarray:
    seed = random.randrange(1 << 30)
    rng = random.Random(seed)
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            return _generate_one(rng)
        except RuntimeError:
            continue
    raise RuntimeError("failed to generate a valid input")
