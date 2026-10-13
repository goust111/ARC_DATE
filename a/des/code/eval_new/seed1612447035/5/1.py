from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
NON_GREY_COLORS = tuple(color for color in range(1, 10) if color != GREY)

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (20, 30)
MIN_OBJECTS = 4
MAX_OBJECTS = 6
GEN_RETRIES = 400
ROOM_RETRIES = 200
PLACEMENT_RETRIES = 200


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _read_left_legend(grid: np.ndarray) -> tuple[dict[int, int], set[tuple[int, int]]]:
    h, _ = grid.shape
    mapping: dict[int, int] = {}
    legend_cells: set[tuple[int, int]] = set()
    r = 0
    while r < h:
        if int(grid[r, 0]) == BG:
            r += 1
            continue
        assert r + 1 < h and int(grid[r + 1, 0]) != BG, "legend must use 2x1 vertical blocks"
        indicator = int(grid[r, 0])
        fill = int(grid[r + 1, 0])
        assert indicator not in (BG, GREY)
        assert fill not in (BG, GREY)
        assert indicator != fill
        assert indicator not in mapping
        mapping[indicator] = fill
        legend_cells.add((r, 0))
        legend_cells.add((r + 1, 0))
        r += 2
    assert len(mapping) == 2, "expected exactly two legend entries on the left edge"
    return mapping, legend_cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill enclosed black cavities of grey rooms using a left-edge legend, but only when the cavity area is odd.

    Input structure:
    - The background is black.
    - Several closed grey shapes appear in the interior; each surrounds one black cavity.
    - The left edge contains two vertical 2x1 legend bars whose top pixel is a border-indicator color and whose bottom pixel is the paired fill color.
    - Each room border contains exactly one non-grey marker pixel whose color matches one legend indicator.

    Rule:
    - Read the left-edge legend as indicator-color -> fill-color pairs.
    - Find each black connected component that is enclosed and does not touch the outer grid boundary.
    - For an enclosed cavity, inspect its boundary: the unique non-grey boundary pixel selects the candidate fill color from the legend.
    - If the cavity area is odd, fill the whole cavity with that color; if it is even, keep the cavity black.
    - In all cases, restore the border marker pixel back to grey in the output.

    Key insight:
    - The lone colored border pixel is not decoration: it links a room to one legend entry, and a second independent rule uses the cavity's area parity to decide whether that matched fill color is actually applied.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    legend, _legend_cells = _read_left_legend(g)

    out = g.copy()
    for cavity in _components(g == BG):
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in cavity):
            continue

        boundary: set[tuple[int, int]] = set()
        for r, c in cavity:
            for nr, nc in _neighbors4(r, c, h, w):
                if int(g[nr, nc]) != BG:
                    boundary.add((nr, nc))

        marker_cells = [cell for cell in boundary if int(g[cell]) in legend]
        if len(marker_cells) != 1:
            continue

        marker = marker_cells[0]
        marker_color = int(g[marker])
        boundary_colors = {int(g[cell]) for cell in boundary}
        if not boundary_colors <= {GREY, marker_color}:
            continue

        out[marker] = GREY
        if len(cavity) % 2 == 1:
            rows, cols = zip(*cavity)
            out[np.array(rows), np.array(cols)] = int(legend[marker_color])

    return out


def _legend_rows(height: int, rng: random.Random) -> list[int]:
    gap = rng.randint(0, 2)
    total = 4 + gap
    start = rng.randint(0, height - total)
    return [start, start + 1, start + 2 + gap, start + 3 + gap]


def _choose_legend_mapping(rng: random.Random) -> tuple[list[tuple[int, int]], dict[int, int]]:
    colors = rng.sample(list(NON_GREY_COLORS), 4)
    pairs = [(int(colors[0]), int(colors[1])), (int(colors[2]), int(colors[3]))]
    rng.shuffle(pairs)
    return pairs, {indicator: fill for indicator, fill in pairs}


def _room_patch(
    rng: random.Random,
    indicator: int,
    fill: int,
    want_odd: bool,
) -> tuple[np.ndarray, np.ndarray]:
    target_parity = 1 if want_odd else 0
    for _ in range(ROOM_RETRIES):
        inner_h = rng.randint(3, 6)
        inner_w = rng.randint(3, 6)
        patch = np.full((inner_h + 2, inner_w + 2), BG, dtype=int)
        patch[0, :] = GREY
        patch[-1, :] = GREY
        patch[:, 0] = GREY
        patch[:, -1] = GREY

        family = rng.choices(["plain", "bar"], weights=[2, 3])[0]
        area = inner_h * inner_w
        if family == "bar":
            side = rng.choice(["top", "bottom", "left", "right"])
            if side in ("top", "bottom"):
                length = rng.randint(1, inner_h - 1)
                pos = rng.randrange(inner_w)
                area -= length
                if area % 2 != target_parity:
                    continue
                col = 1 + pos
                if side == "top":
                    patch[1 : 1 + length, col] = GREY
                else:
                    patch[inner_h - length + 1 : inner_h + 1, col] = GREY
            else:
                length = rng.randint(1, inner_w - 1)
                pos = rng.randrange(inner_h)
                area -= length
                if area % 2 != target_parity:
                    continue
                row = 1 + pos
                if side == "left":
                    patch[row, 1 : 1 + length] = GREY
                else:
                    patch[row, inner_w - length + 1 : inner_w + 1] = GREY
        else:
            if area % 2 != target_parity:
                continue

        cavity_mask = patch == BG
        if area < 5 or len(_components(cavity_mask)) != 1:
            continue

        marker_side = rng.choice(["top", "bottom", "left", "right"])
        if marker_side == "top":
            marker = (0, 1 + rng.randrange(inner_w))
        elif marker_side == "bottom":
            marker = (inner_h + 1, 1 + rng.randrange(inner_w))
        elif marker_side == "left":
            marker = (1 + rng.randrange(inner_h), 0)
        else:
            marker = (1 + rng.randrange(inner_h), inner_w + 1)

        patch[marker] = indicator

        solved = patch.copy()
        solved[marker] = GREY
        if want_odd:
            solved[cavity_mask] = fill
        return patch, solved
    raise RuntimeError("failed to build a valid room patch")


def _place_room(
    canvas: np.ndarray,
    solved_canvas: np.ndarray,
    blocked: np.ndarray,
    patch: np.ndarray,
    solved_patch: np.ndarray,
    rng: random.Random,
) -> bool:
    h, w = canvas.shape
    ph, pw = patch.shape
    row_min = 1
    row_max = h - ph - 1
    col_min = 4
    col_max = w - pw - 1
    if row_max < row_min or col_max < col_min:
        return False

    for _ in range(PLACEMENT_RETRIES):
        r0 = rng.randint(row_min, row_max)
        c0 = rng.randint(col_min, col_max)
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(h, r0 + ph + 1)
        cc1 = min(w, c0 + pw + 1)
        if blocked[rr0:rr1, cc0:cc1].any():
            continue

        canvas[r0 : r0 + ph, c0 : c0 + pw] = patch
        solved_canvas[r0 : r0 + ph, c0 : c0 + pw] = solved_patch
        blocked[rr0:rr1, cc0:cc1] = True
        return True

    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*GRID_H_RANGE)
    width = rng.randint(*GRID_W_RANGE)
    grid = np.full((height, width), BG, dtype=int)
    solved = grid.copy()

    legend_rows = _legend_rows(height, rng)
    legend_pairs, mapping = _choose_legend_mapping(rng)
    for pair_index, (indicator, fill) in enumerate(legend_pairs):
        top = legend_rows[2 * pair_index]
        grid[top, 0] = indicator
        grid[top + 1, 0] = fill
        solved[top, 0] = indicator
        solved[top + 1, 0] = fill

    blocked = np.zeros((height, width), dtype=bool)
    blocked[:, :3] = True

    assignments = [(0, True), (0, False), (1, True), (1, False)]
    extra = rng.randint(MIN_OBJECTS, MAX_OBJECTS) - len(assignments)
    for _ in range(extra):
        assignments.append((rng.randrange(2), bool(rng.randrange(2))))
    rng.shuffle(assignments)

    rooms = []
    for legend_index, want_odd in assignments:
        indicator, fill = legend_pairs[legend_index]
        patch, solved_patch = _room_patch(rng, indicator, fill, want_odd)
        rooms.append((patch.shape[0] * patch.shape[1], patch, solved_patch))
    rooms.sort(key=lambda item: item[0], reverse=True)

    for _, patch, solved_patch in rooms:
        if not _place_room(grid, solved, blocked, patch, solved_patch, rng):
            return None

    if not np.array_equal(main(grid), solved):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
