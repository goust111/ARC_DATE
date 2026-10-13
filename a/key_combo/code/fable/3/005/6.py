from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
STENCIL = 5
ANCHOR = 8
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
ALL_COLORS = tuple(range(10))


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _shape_bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_connected(cells: list[tuple[int, int]]) -> bool:
    cell_set = set(cells)
    q = deque([cells[0]])
    seen = {cells[0]}
    while q:
        r, c = q.popleft()
        for dr, dc in NEIGHBORS4:
            nxt = (r + dr, c + dc)
            if nxt in cell_set and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return len(seen) == len(cells)


def _parse(grid: np.ndarray) -> tuple[int, int, np.ndarray, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= 4 and w >= 4, "grid is too small"

    bg = _dominant_color(g)
    wall_color = int(g[h - 1, 0])
    assert wall_color != bg, "bottom edge must be a non-background wall"
    assert np.all(g[h - 1, :] == wall_color), "bottom edge must be one solid wall color"

    wall_mask = np.zeros_like(g, dtype=bool)
    wall_mask[h - 1, :] = True
    assert np.array_equal(g == wall_color, wall_mask), "wall color must appear only on the bottom edge"

    work_mask = (g != bg) & ~wall_mask
    foreground = _components(work_mask)
    stencil_components = [coords for coords in foreground if np.any(g[coords[:, 0], coords[:, 1]] == STENCIL)]
    assert len(stencil_components) == 1, "expected one grey stencil component"

    stencil_coords = stencil_components[0]
    stencil_values = g[stencil_coords[:, 0], stencil_coords[:, 1]]
    anchor_coords = stencil_coords[stencil_values != STENCIL]
    assert len(anchor_coords) == 1, "stencil must contain exactly one non-grey anchor cell"
    anchor_color = int(g[anchor_coords[0, 0], anchor_coords[0, 1]])
    assert anchor_color not in (bg, wall_color, STENCIL), "anchor must be visibly distinct"

    anchor = anchor_coords[0]
    rels = stencil_coords - anchor
    assert len(rels) >= 4, "stencil copy must contain several cells"
    assert _is_connected([tuple(map(int, rc)) for rc in rels.tolist()]), "stencil shape must be connected"

    stencil_mask = np.zeros_like(g, dtype=bool)
    stencil_mask[stencil_coords[:, 0], stencil_coords[:, 1]] = True
    seed_mask = work_mask & ~stencil_mask
    seed_components = _components(seed_mask)
    assert seed_components, "expected at least one colored seed"

    seeds: list[dict[str, object]] = []
    colors: list[int] = []
    for comp in seed_components:
        assert len(comp) == 1, "each seed must be an isolated pixel"
        r, c = map(int, comp[0])
        color = int(g[r, c])
        assert color not in (bg, wall_color, STENCIL, anchor_color), "seed color must be distinct from markers"
        seeds.append({"pos": np.array([r, c], dtype=int), "color": color})
        colors.append(color)
    assert len(colors) == len(set(colors)), "seed colors must be unique"

    for seed in seeds:
        coords = seed["pos"] + rels
        assert np.all((0 <= coords[:, 0]) & (coords[:, 0] < h)), "seed-aligned copy leaves the grid"
        assert np.all((0 <= coords[:, 1]) & (coords[:, 1] < w)), "seed-aligned copy leaves the grid"
        assert np.all(coords[:, 0] < h - 1), "seed-aligned copy overlaps the wall row"

    bottoms = [int((seed["pos"] + rels)[:, 0].max()) for seed in seeds]
    assert len(bottoms) == len(set(bottoms)), "copy vertical order must be unambiguous"
    return bg, wall_color, rels.astype(int), seeds


def _settle(
    shape: tuple[int, int],
    bg: int,
    wall_color: int,
    rels: np.ndarray,
    seeds: list[dict[str, object]],
    *,
    use_stack: bool = True,
    return_events: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[dict[str, object]]]:
    h, w = shape
    out = np.full((h, w), int(bg), dtype=int)
    out[h - 1, :] = int(wall_color)

    wall_mask = np.zeros((h, w), dtype=bool)
    wall_mask[h - 1, :] = True
    occupied = wall_mask.copy()

    copies = []
    for seed in seeds:
        coords = np.asarray(seed["pos"], dtype=int) + rels
        copies.append(
            {
                "coords": coords.astype(int),
                "color": int(seed["color"]),
                "bottom": int(coords[:, 0].max()),
                "left": int(coords[:, 1].min()),
            }
        )

    events: list[dict[str, object]] = []
    for copy in sorted(copies, key=lambda item: (-int(item["bottom"]), int(item["left"]))):
        coords = copy["coords"].copy()
        moved = 0
        blocker = "boundary"
        while True:
            nxt = coords + np.array([1, 0], dtype=int)
            in_bounds = (
                np.all(0 <= nxt[:, 0])
                and np.all(nxt[:, 0] < h)
                and np.all(0 <= nxt[:, 1])
                and np.all(nxt[:, 1] < w)
            )
            if not in_bounds:
                blocker = "boundary"
                break
            hit = occupied[nxt[:, 0], nxt[:, 1]]
            if np.any(hit):
                hit_rows = nxt[hit, 0]
                blocker = "wall" if np.any(hit_rows == h - 1) else "copy"
                break
            coords = nxt
            moved += 1

        assert not np.any(occupied[coords[:, 0], coords[:, 1]]), "settled stamps must not overlap"
        out[coords[:, 0], coords[:, 1]] = int(copy["color"])
        if use_stack:
            occupied[coords[:, 0], coords[:, 1]] = True

        events.append(
            {
                "color": int(copy["color"]),
                "moved": moved,
                "blocker": blocker,
                "final": coords.copy(),
            }
        )

    if return_events:
        return out, events
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop seed-generated stencil stamps onto the bottom wall.

    Input structure:
    - The background is empty, and the bottom edge is a complete one-cell-thick wall of one color.
    - A floating grey stencil component contains exactly one non-grey anchor pixel.
    - Every other foreground object is an isolated colored seed pixel, and seed colors are all different.

    Rule:
    - Read the full stencil mask as grey cells plus the anchor cell, using the anchor as the alignment point.
    - For each seed, create a rigid copy of that mask, recolor the whole copy with the seed color, and align the anchor cell to the seed.
    - Clear the original stencil and seeds, then drop lower starting copies first until the next downward step would hit the wall or a settled copy.

    Key insight:
    - The visible seeds are generators, not the falling objects; shape, alignment, color, and final stacking come from separate clues.
    """
    g = np.asarray(grid, dtype=int)
    bg, wall_color, rels, seeds = _parse(g)
    return _settle(g.shape, bg, wall_color, rels, seeds)


# ---------------- generator ----------------


BASE_SHAPES: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)),
    ((0, 0), (-1, 0), (0, -1), (1, -1), (1, 0), (1, 1)),
    ((0, 0), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)),
    ((0, 0), (-2, 0), (-1, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 0), (-1, 0), (-1, 1), (0, -1), (1, -1), (1, 0)),
    ((0, 0), (-1, -1), (-1, 0), (0, -1), (1, -1), (1, 0), (1, 1)),
)


def _rot90(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(c, -r) for r, c in cells]


def _random_rels() -> np.ndarray:
    for _ in range(200):
        cells = list(random.choice(BASE_SHAPES))
        for _ in range(random.randrange(4)):
            cells = _rot90(cells)
        if random.choice((False, True)):
            cells = [(r, -c) for r, c in cells]
        cells = sorted(set(cells))
        arr = np.array(cells, dtype=int)
        r0, c0, r1, c1 = _shape_bbox(arr)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if not (2 <= height <= 5 and 2 <= width <= 5):
            continue
        if r0 >= 0 and c0 >= 0:
            continue
        if r1 <= 0:
            continue
        if len(cells) < 5:
            continue
        if _is_connected(cells):
            return arr
    raise RuntimeError("failed to choose a valid stencil shape")


def _clear_with_padding(grid: np.ndarray, cells: np.ndarray, padding: int = 1) -> bool:
    h, w = grid.shape
    for r, c in cells:
        rr = int(r)
        cc = int(c)
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        for dr in range(-padding, padding + 1):
            for dc in range(-padding, padding + 1):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != BG:
                    return False
    return True


def _copy_cells(pos: tuple[int, int] | np.ndarray, rels: np.ndarray) -> np.ndarray:
    return np.asarray(pos, dtype=int) + rels


def _cells_key(cells: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in cells.tolist()}


def _initial_copies_do_not_overlap(seed_positions: list[tuple[int, int]], rels: np.ndarray) -> bool:
    occupied: set[tuple[int, int]] = set()
    for pos in seed_positions:
        cells = _cells_key(_copy_cells(pos, rels))
        if occupied & cells:
            return False
        occupied |= cells
    return True


def _single_pixel_drop(
    shape: tuple[int, int],
    bg: int,
    wall_color: int,
    seeds: list[dict[str, object]],
) -> np.ndarray:
    rels = np.array([[0, 0]], dtype=int)
    return _settle(shape, bg, wall_color, rels, seeds)


def _top_left_rels(rels: np.ndarray) -> np.ndarray:
    r0, c0, _, _ = _shape_bbox(rels)
    return rels - np.array([r0, c0], dtype=int)


def _semantic_checks(
    grid: np.ndarray,
    expected: np.ndarray,
    bg: int,
    wall_color: int,
    rels: np.ndarray,
    seeds: list[dict[str, object]],
    events: list[dict[str, object]],
    template_cells: np.ndarray,
) -> bool:
    if not np.array_equal(main(grid), expected):
        return False
    if not any(event["blocker"] == "wall" for event in events):
        return False
    if not any(event["blocker"] == "copy" for event in events):
        return False
    if sum(int(event["moved"]) > 0 for event in events) < 2:
        return False

    if np.array_equal(_single_pixel_drop(grid.shape, bg, wall_color, seeds), expected):
        return False
    if np.array_equal(_settle(grid.shape, bg, wall_color, rels, seeds, use_stack=False), expected):
        return False

    top_left = _top_left_rels(rels)
    for seed in seeds:
        cells = np.asarray(seed["pos"], dtype=int) + top_left
        if not (
            np.all((0 <= cells[:, 0]) & (cells[:, 0] < grid.shape[0] - 1))
            and np.all((0 <= cells[:, 1]) & (cells[:, 1] < grid.shape[1]))
        ):
            return False
    if np.array_equal(_settle(grid.shape, bg, wall_color, top_left, seeds), expected):
        return False

    seed_positions = [tuple(map(int, seed["pos"])) for seed in seeds]
    for r, c in seed_positions:
        if expected[r, c] != bg:
            return False
    for r, c in template_cells:
        if expected[int(r), int(c)] != bg:
            return False

    stamp_area = len(rels)
    for seed in seeds:
        if int(np.sum(expected == int(seed["color"]))) != stamp_area:
            return False
    return True


def _place_seed(
    grid: np.ndarray,
    rels: np.ndarray,
    existing_positions: list[tuple[int, int]],
    row_range: tuple[int, int],
    col_range: tuple[int, int],
    used_bottoms: set[int],
) -> tuple[int, int] | None:
    h, w = grid.shape
    rmin, rmax = row_range
    cmin, cmax = col_range
    if rmin > rmax or cmin > cmax:
        return None

    virtual_occupied: set[tuple[int, int]] = set()
    for pos in existing_positions:
        virtual_occupied |= _cells_key(_copy_cells(pos, rels))

    for _ in range(300):
        r = random.randint(rmin, rmax)
        c = random.randint(cmin, cmax)
        cells = _copy_cells((r, c), rels)
        bottom = int(cells[:, 0].max())
        if bottom in used_bottoms:
            continue
        if not (
            np.all((0 <= cells[:, 0]) & (cells[:, 0] < h - 1))
            and np.all((0 <= cells[:, 1]) & (cells[:, 1] < w))
        ):
            continue
        if virtual_occupied & _cells_key(cells):
            continue
        if not _clear_with_padding(grid, np.array([[r, c]], dtype=int), padding=1):
            continue
        return r, c
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(17, 24)
    w = random.randint(14, 26)
    rels = _random_rels()
    r0, c0, r1, c1 = _shape_bbox(rels)
    shape_height = r1 - r0 + 1

    wall_color = random.choice([2, 3, 4, 6, 7, 9])
    seed_pool = [c for c in (1, 2, 3, 4, 6, 7, 9) if c != wall_color]
    seed_count = random.randint(3, min(5, len(seed_pool)))
    seed_colors = random.sample(seed_pool, seed_count)

    grid = np.full((h, w), BG, dtype=int)
    grid[h - 1, :] = wall_color

    anchor_row_min = max(1 - r0, 1)
    anchor_row_max = min(max(anchor_row_min, h // 3), h - 5 - r1)
    anchor_col_min = max(1 - c0, 1)
    anchor_col_max = min(w - 2 - c1, w - 2)
    if anchor_row_min > anchor_row_max or anchor_col_min > anchor_col_max:
        return None

    template_anchor = None
    template_cells = None
    for _ in range(100):
        tr = random.randint(anchor_row_min, anchor_row_max)
        tc = random.randint(anchor_col_min, anchor_col_max)
        cells = _copy_cells((tr, tc), rels)
        if _clear_with_padding(grid, cells, padding=1):
            template_anchor = (tr, tc)
            template_cells = cells
            break
    if template_anchor is None or template_cells is None:
        return None

    for r, c in template_cells:
        grid[int(r), int(c)] = STENCIL
    grid[template_anchor] = ANCHOR

    row_min = max(-r0, int(template_cells[:, 0].max()) + 3)
    row_max = h - 3 - r1
    col_min = -c0
    col_max = w - 1 - c1
    if row_min > row_max or col_min > col_max:
        return None

    pair_col = random.randint(col_min, col_max)
    sep = shape_height + random.randint(1, 3)
    lower_min = row_min + sep
    lower_max = row_max - 1
    if lower_min > lower_max:
        return None

    seed_positions: list[tuple[int, int]] = []
    used_bottoms: set[int] = set()
    lower = upper = None
    for _ in range(200):
        lr = random.randint(lower_min, lower_max)
        ur = lr - sep
        lower_cells = _copy_cells((lr, pair_col), rels)
        upper_cells = _copy_cells((ur, pair_col), rels)
        if not (
            np.all((0 <= lower_cells[:, 0]) & (lower_cells[:, 0] < h - 1))
            and np.all((0 <= upper_cells[:, 0]) & (upper_cells[:, 0] < h - 1))
            and np.all((0 <= lower_cells[:, 1]) & (lower_cells[:, 1] < w))
            and np.all((0 <= upper_cells[:, 1]) & (upper_cells[:, 1] < w))
        ):
            continue
        if _cells_key(lower_cells) & _cells_key(upper_cells):
            continue
        if not _clear_with_padding(grid, np.array([[lr, pair_col], [ur, pair_col]], dtype=int), padding=1):
            continue
        lower = (lr, pair_col)
        upper = (ur, pair_col)
        break
    if lower is None or upper is None:
        return None

    seed_positions.extend([lower, upper])
    for pos in seed_positions:
        used_bottoms.add(int(_copy_cells(pos, rels)[:, 0].max()))
        grid[pos] = seed_colors[len(used_bottoms) - 1]

    for i in range(2, seed_count):
        pos = _place_seed(
            grid,
            rels,
            seed_positions,
            (row_min, row_max),
            (col_min, col_max),
            used_bottoms,
        )
        if pos is None:
            return None
        seed_positions.append(pos)
        used_bottoms.add(int(_copy_cells(pos, rels)[:, 0].max()))
        grid[pos] = seed_colors[i]

    if not _initial_copies_do_not_overlap(seed_positions, rels):
        return None

    seeds = [
        {"pos": np.array(pos, dtype=int), "color": int(color)}
        for pos, color in zip(seed_positions, seed_colors)
    ]
    expected, events = _settle(grid.shape, BG, wall_color, rels, seeds, return_events=True)

    if not _semantic_checks(grid, expected, BG, wall_color, rels, seeds, events, template_cells):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(1000):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid seed-stamp drop input")
