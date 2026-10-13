from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8)

DIRS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
CARDINALS = ((-1, 0), (0, -1), (0, 1), (1, 0))
BLOCKABLE_CARDINALS = ((0, -1), (0, 1), (1, 0))
DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))

LEGEND_H = 5
STAMP = 3
CELL_W = 4
WORKSPACE_TOP = 6
RESERVE_RADIUS = 4
GEN_RETRIES = 400


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
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
                for dr, dc in CARDINALS:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _find_templates(grid: np.ndarray) -> tuple[tuple[int, int, int, int], dict[int, tuple[tuple[int, int], ...]]]:
    grey_comps = _components4(grid == GREY)
    candidates: list[tuple[int, tuple[int, int, int, int], np.ndarray]] = []
    for comp in grey_comps:
        rows = comp[:, 0]
        cols = comp[:, 1]
        r0 = int(rows.min())
        r1 = int(rows.max())
        c0 = int(cols.min())
        c1 = int(cols.max())
        if r0 != 0 or r1 - r0 + 1 != LEGEND_H or c1 - c0 + 1 < 9:
            continue
        candidates.append((int(comp.shape[0]), (r0, r1, c0, c1), comp))
    assert len(candidates) == 1, "expected one grey legend band touching the top edge"

    _area, (r0, r1, c0, c1), _comp = candidates[0]
    sep_cols = [c for c in range(c0, c1 + 1) if np.all(grid[r0 : r1 + 1, c] == GREY)]
    assert len(sep_cols) >= 4, "legend must contain at least three stamp cells"
    assert all(sep_cols[i + 1] - sep_cols[i] == CELL_W for i in range(len(sep_cols) - 1)), "legend cells must be evenly separated"

    templates: dict[int, tuple[tuple[int, int], ...]] = {}
    for left, right in zip(sep_cols[:-1], sep_cols[1:]):
        patch = grid[r0 + 1 : r1, left + 1 : right]
        assert patch.shape == (STAMP, STAMP), "each legend cell must contain a 3x3 stamp"
        color = int(patch[1, 1])
        assert color not in (BLACK, GREY, MAROON), "stamp center must be a seed color"
        rels: list[tuple[int, int]] = []
        for rr in range(STAMP):
            for cc in range(STAMP):
                if int(patch[rr, cc]) == BLACK:
                    continue
                assert int(patch[rr, cc]) == color, "non-black stamp pixels must match the center color"
                if (rr, cc) != (1, 1):
                    rels.append((rr - 1, cc - 1))
        assert rels, "each stamp must encode at least one growth direction"
        assert color not in templates, "stamp center colors must be unique"
        templates[color] = tuple(rels)

    return (r0, r1, c0, c1), templates


def _planned_fills(
    grid: np.ndarray,
    templates: dict[int, tuple[tuple[int, int], ...]],
    legend_bottom: int,
) -> tuple[dict[tuple[int, int], int], dict[str, int]]:
    h, w = grid.shape
    fills: dict[tuple[int, int], int] = {}
    counts = {"direct": 0, "slide": 0, "blocked": 0, "ignored": 0}

    for r in range(legend_bottom + 1, h):
        for c in range(w):
            color = int(grid[r, c])
            if color not in templates:
                continue
            for dr, dc in templates[color]:
                rr = r + dr
                cc = c + dc
                if not (0 <= rr < h and 0 <= cc < w):
                    counts["ignored"] += 1
                    continue
                target = int(grid[rr, cc])
                if target == BLACK:
                    counts["direct"] += 1
                    prev = fills.get((rr, cc))
                    assert prev in (None, color), "different seeds cannot claim the same fill cell"
                    fills[(rr, cc)] = color
                    continue
                if target != MAROON:
                    counts["ignored"] += 1
                    continue

                sr = rr
                sc = cc
                while 0 <= sr < h and 0 <= sc < w and int(grid[sr, sc]) == MAROON:
                    sr += dr
                    sc += dc
                if not (0 <= sr < h and 0 <= sc < w):
                    counts["blocked"] += 1
                    continue
                if int(grid[sr, sc]) == BLACK:
                    counts["slide"] += 1
                    prev = fills.get((sr, sc))
                    assert prev in (None, color), "different seeds cannot claim the same slide landing cell"
                    fills[(sr, sc)] = color
                else:
                    counts["blocked"] += 1

    return fills, counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each colored seed according to its matching top-row stamp template, sliding through maroon walls when needed.

    Input structure:
    - The background is black.
    - A grey-framed legend band touches the top edge; it contains several 3x3 color stamps separated by grey vertical lines.
    - Each stamp's center color identifies that stamp, and the other same-color cells inside the 3x3 patch mark allowed growth directions.
    - Below the legend are single-pixel colored seeds and straight maroon wall segments on the black workspace.

    Rule:
    - Match every workspace seed to the legend stamp whose center has the same color.
    - For each non-center same-color position in that 3x3 stamp, treat its offset from the center as one growth direction from the seed.
    - If the target cell in that direction is black, fill it with the seed color.
    - If the target cell is maroon, keep moving in the same direction across the whole maroon run and fill the first black cell after the wall; if the run leaves the grid or hits a non-black blocker, do nothing for that direction.
    - Keep the legend, grey separators, walls, and original seed pixels unchanged.

    Key insight:
    - The top legend is a color-keyed lookup table for local growth shapes, and maroon walls do not stop growth outright: they convert a one-step expansion into a same-direction jump through the wall.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    (_r0, legend_bottom, _c0, _c1), templates = _find_templates(g)
    fills, _counts = _planned_fills(g, templates, legend_bottom)

    out = g.copy()
    for (r, c), color in fills.items():
        out[r, c] = color
    return out


def _template_patch(color: int, rels: tuple[tuple[int, int], ...]) -> np.ndarray:
    patch = np.zeros((STAMP, STAMP), dtype=int)
    patch[1, 1] = color
    for dr, dc in rels:
        patch[1 + dr, 1 + dc] = color
    return patch


def _make_templates(rng: random.Random, count: int) -> dict[int, tuple[tuple[int, int], ...]]:
    colors = rng.sample(SEED_COLORS, count)
    templates: dict[int, tuple[tuple[int, int], ...]] = {}
    seen: set[tuple[tuple[int, int], ...]] = set()

    while len(templates) < count:
        color = colors[len(templates)]
        allowed = list(BLOCKABLE_CARDINALS)
        base = [rng.choice(allowed)]
        remaining_cardinals = [d for d in CARDINALS if d not in base]
        extra_card = rng.sample(remaining_cardinals, rng.randint(0, min(2, len(remaining_cardinals))))
        max_diag = 4 - len(base) - len(extra_card)
        diag_count = rng.randint(0, min(2, len(DIAGONALS), max_diag))
        extra_diag = rng.sample(DIAGONALS, diag_count)
        rels = tuple(sorted(base + extra_card + extra_diag))
        if rels in seen:
            continue
        seen.add(rels)
        templates[color] = rels

    return templates


def _seed_box_ok(reserved: np.ndarray, r: int, c: int) -> bool:
    h, w = reserved.shape
    r0 = max(0, r - RESERVE_RADIUS)
    r1 = min(h - 1, r + RESERVE_RADIUS)
    c0 = max(0, c - RESERVE_RADIUS)
    c1 = min(w - 1, c + RESERVE_RADIUS)
    return not np.any(reserved[r0 : r1 + 1, c0 : c1 + 1])


def _reserve_box(reserved: np.ndarray, r: int, c: int) -> None:
    h, w = reserved.shape
    r0 = max(0, r - RESERVE_RADIUS)
    r1 = min(h - 1, r + RESERVE_RADIUS)
    c0 = max(0, c - RESERVE_RADIUS)
    c1 = min(w - 1, c + RESERVE_RADIUS)
    reserved[r0 : r1 + 1, c0 : c1 + 1] = True


def _branch_candidates(
    h: int,
    w: int,
    branch: str,
    direction: tuple[int, int],
    wall_len: int,
) -> list[tuple[int, int]]:
    dr, dc = direction
    cells: list[tuple[int, int]] = []

    if branch == "blocked":
        if direction == (0, 1):
            col_values = [w - 1 - wall_len]
            row_values = range(WORKSPACE_TOP + 2, h - 2)
        elif direction == (0, -1):
            col_values = [wall_len]
            row_values = range(WORKSPACE_TOP + 2, h - 2)
        else:
            assert direction == (1, 0)
            row_values = [h - 1 - wall_len]
            col_values = range(2, w - 2)
        for r in row_values:
            for c in col_values:
                cells.append((int(r), int(c)))
        return cells

    row_start = WORKSPACE_TOP + 2
    row_stop = h - 3
    col_start = 2
    col_stop = w - 3
    if branch == "slide":
        if dr == -1:
            row_start = max(row_start, WORKSPACE_TOP + wall_len + 1)
        if dr == 1:
            row_stop = min(row_stop, h - wall_len - 2)
        if dc == -1:
            col_start = max(col_start, wall_len + 1)
        if dc == 1:
            col_stop = min(col_stop, w - wall_len - 2)
    if row_start > row_stop or col_start > col_stop:
        return cells
    for r in range(row_start, row_stop + 1):
        for c in range(col_start, col_stop + 1):
            cells.append((r, c))
    return cells


def _candidate_ok(
    grid: np.ndarray,
    reserved: np.ndarray,
    r: int,
    c: int,
    rels: tuple[tuple[int, int], ...],
    branch: str,
    sel_dir: tuple[int, int] | None,
    wall_len: int,
) -> bool:
    h, w = grid.shape
    if grid[r, c] != BLACK or not _seed_box_ok(reserved, r, c):
        return False

    for dr, dc in rels:
        if branch in {"slide", "blocked"} and (dr, dc) == sel_dir:
            for step in range(1, wall_len + 1):
                rr = r + dr * step
                cc = c + dc * step
                if not (0 <= rr < h and 0 <= cc < w) or int(grid[rr, cc]) != BLACK:
                    return False
            rr = r + dr * (wall_len + 1)
            cc = c + dc * (wall_len + 1)
            if branch == "slide":
                if not (0 <= rr < h and 0 <= cc < w) or int(grid[rr, cc]) != BLACK:
                    return False
            else:
                if 0 <= rr < h and 0 <= cc < w:
                    return False
            continue

        rr = r + dr
        cc = c + dc
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        if int(grid[rr, cc]) != BLACK:
            return False

    return True


def _place_seed(
    grid: np.ndarray,
    reserved: np.ndarray,
    r: int,
    c: int,
    color: int,
    branch: str,
    sel_dir: tuple[int, int] | None,
    wall_len: int,
) -> None:
    grid[r, c] = color
    if branch in {"slide", "blocked"} and sel_dir is not None:
        dr, dc = sel_dir
        for step in range(1, wall_len + 1):
            grid[r + dr * step, c + dc * step] = MAROON
    _reserve_box(reserved, r, c)


def _add_decoy_walls(grid: np.ndarray, reserved: np.ndarray, rng: random.Random) -> None:
    h, w = grid.shape
    decoy_count = rng.randint(0, 2)
    for _ in range(decoy_count):
        for _attempt in range(60):
            horizontal = rng.random() < 0.5
            length = rng.randint(2, 5)
            if horizontal:
                r = rng.randint(WORKSPACE_TOP + 1, h - 2)
                c = rng.randint(1, w - length - 1)
                cells = [(r, c + k) for k in range(length)]
            else:
                r = rng.randint(WORKSPACE_TOP + 1, h - length - 1)
                c = rng.randint(1, w - 2)
                cells = [(r + k, c) for k in range(length)]
            if any(grid[rr, cc] != BLACK for rr, cc in cells):
                continue
            if any(reserved[rr, cc] for rr, cc in cells):
                continue
            for rr, cc in cells:
                grid[rr, cc] = MAROON
            break


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    num_templates = rng.randint(3, 4)
    templates = _make_templates(rng, num_templates)

    legend_left = rng.randint(0, 3)
    legend_w = 1 + CELL_W * num_templates
    min_w = max(legend_left + legend_w + 2, 16)
    w = rng.randint(min_w, 30)
    h = rng.randint(18, 30)
    if h <= WORKSPACE_TOP + 5:
        return None

    grid = np.zeros((h, w), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)
    reserved[:WORKSPACE_TOP, :] = True

    legend_right = legend_left + legend_w - 1
    grid[0, legend_left : legend_right + 1] = GREY
    grid[LEGEND_H - 1, legend_left : legend_right + 1] = GREY
    for idx in range(num_templates + 1):
        c = legend_left + idx * CELL_W
        grid[:LEGEND_H, c] = GREY
    for idx, (color, rels) in enumerate(templates.items()):
        c0 = legend_left + idx * CELL_W + 1
        grid[1:4, c0 : c0 + 3] = _template_patch(color, rels)

    colors = list(templates.keys())
    rng.shuffle(colors)
    seed_specs: list[tuple[int, str]] = []
    base_branches = ["direct", "slide", "blocked"]
    for i, color in enumerate(colors):
        seed_specs.append((color, base_branches[i % 3]))
    target_seed_count = rng.randint(num_templates + 1, min(num_templates + 3, 6))
    for i in range(len(seed_specs), target_seed_count):
        seed_specs.append((rng.choice(colors), base_branches[i % 3]))
    rng.shuffle(seed_specs)

    for color, branch in seed_specs:
        rels = templates[color]
        cardinals = [d for d in rels if d in CARDINALS]
        if branch == "blocked":
            options = [d for d in cardinals if d in BLOCKABLE_CARDINALS]
        elif branch == "slide":
            options = cardinals
        else:
            options = []
        if branch != "direct" and not options:
            return None

        sel_dir = rng.choice(options) if options else None
        wall_len = rng.randint(1, 3) if branch != "direct" else 0
        candidates = _branch_candidates(h, w, branch, sel_dir if sel_dir is not None else (0, 0), wall_len)
        rng.shuffle(candidates)

        placed = False
        for r, c in candidates:
            if not _candidate_ok(grid, reserved, r, c, rels, branch, sel_dir, wall_len):
                continue
            _place_seed(grid, reserved, r, c, color, branch, sel_dir, wall_len)
            placed = True
            break
        if not placed:
            return None

    _add_decoy_walls(grid, reserved, rng)

    try:
        (_r0, legend_bottom, _c0, _c1), parsed = _find_templates(grid)
    except AssertionError:
        return None
    if parsed != templates:
        return None

    try:
        fills, counts = _planned_fills(grid, parsed, legend_bottom)
    except AssertionError:
        return None
    if counts["direct"] == 0 or counts["slide"] == 0 or counts["blocked"] == 0:
        return None
    if len(fills) < 4:
        return None
    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid seed-stamp wall-sliding input")
