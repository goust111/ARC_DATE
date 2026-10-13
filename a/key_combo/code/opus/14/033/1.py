from __future__ import annotations

import hashlib
from collections import deque

import numpy as np


BLACK = 0
RED = 2
STRIPE_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

_SEEN_INPUT_SIGS: set[str] = set()
_SEEN_OUTPUT_SIGS: set[str] = set()


def _sig(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _constant_nonblack_row(grid: np.ndarray, r: int) -> bool:
    return int(grid[r, 0]) != BLACK and np.all(grid[r, :] == grid[r, 0])


def _constant_nonblack_col(grid: np.ndarray, c: int) -> bool:
    return int(grid[0, c]) != BLACK and np.all(grid[:, c] == grid[0, c])


def _edge_run(grid: np.ndarray, edge: str) -> tuple[int, list[int]]:
    h, w = grid.shape
    colors: list[int] = []
    if edge == "top":
        idx = 0
        while idx < h and _constant_nonblack_row(grid, idx):
            colors.append(int(grid[idx, 0]))
            idx += 1
        return idx, colors
    if edge == "bottom":
        idx = h - 1
        while idx >= 0 and _constant_nonblack_row(grid, idx):
            colors.append(int(grid[idx, 0]))
            idx -= 1
        colors.reverse()
        return h - 1 - idx, colors
    if edge == "left":
        idx = 0
        while idx < w and _constant_nonblack_col(grid, idx):
            colors.append(int(grid[0, idx]))
            idx += 1
        return idx, colors
    if edge == "right":
        idx = w - 1
        while idx >= 0 and _constant_nonblack_col(grid, idx):
            colors.append(int(grid[0, idx]))
            idx -= 1
        colors.reverse()
        return w - 1 - idx, colors
    raise ValueError(edge)


def _collapse_runs(values: list[int]) -> list[int]:
    out: list[int] = []
    for value in values:
        if not out or value != out[-1]:
            out.append(value)
    return out


def _find_stripe(grid: np.ndarray) -> dict:
    candidates = []
    for edge in ("top", "bottom", "left", "right"):
        span, raw = _edge_run(grid, edge)
        if span < 2:
            continue
        orientation = "horizontal" if edge in ("top", "bottom") else "vertical"
        entries = _collapse_runs(raw)
        if len(entries) < 2:
            continue
        mask = np.zeros_like(grid, dtype=bool)
        if edge == "top":
            mask[:span, :] = True
        elif edge == "bottom":
            mask[grid.shape[0] - span :, :] = True
        elif edge == "left":
            mask[:, :span] = True
        else:
            mask[:, grid.shape[1] - span :] = True
        candidates.append(
            {
                "edge": edge,
                "orientation": orientation,
                "span": span,
                "raw": raw,
                "entries": entries,
                "entry_count": len(entries),
                "mask": mask,
            }
        )
    assert len(candidates) == 1, "expected exactly one edge stripe region"
    return candidates[0]


def _find_sprite(grid: np.ndarray, stripe_mask: np.ndarray) -> tuple[int, list[tuple[int, int]]]:
    obj_mask = (grid != BLACK) & ~stripe_mask
    comps = _components(obj_mask)
    assert len(comps) == 1, "expected exactly one non-stripe foreground component"
    coords = comps[0]
    colors = {int(grid[r, c]) for r, c in coords}
    assert len(colors) == 1, "sprite must be monochrome"
    return next(iter(colors)), coords


def _shift_for_stripe(edge: str, entry_count: int) -> tuple[int, int]:
    if edge == "top":
        return -entry_count, 0
    if edge == "bottom":
        return entry_count, 0
    if edge == "left":
        return 0, -entry_count
    if edge == "right":
        return 0, entry_count
    raise ValueError(edge)


def _paint_coords(shape: tuple[int, int], coords: list[tuple[int, int]], color: int) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for r, c in coords:
        out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the edge stripe into a run count, move the sprite toward that edge by that count, and erase the stripe.

    Input structure:
    - The grid has black background plus exactly two objects: one small monochrome sprite and one stripe block attached to exactly one outer edge.
    - The stripe block is made of contiguous full-width rows or full-height columns whose colors stay constant inside each row/column and may repeat only because of thickness.
    - The sprite is separated from the stripe by black cells and does not overlap it.

    Rule:
    - Detect the unique edge stripe and whether it is horizontal or vertical.
    - Read the stripe across its thickness, collapsing consecutive identical stripe rows or columns into one entry per color band; let the number of entries be N.
    - Move the sprite by N cells along the axis perpendicular to the stripe, in the direction of the stripe's edge, then erase the entire original stripe and the sprite's old position.

    Key insight:
    - Stripe thickness is distractor information: only the compressed color-band count controls the displacement, while the edge location fixes the axis and sign.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    stripe = _find_stripe(g)
    sprite_color, sprite_coords = _find_sprite(g, stripe["mask"])
    dr, dc = _shift_for_stripe(stripe["edge"], stripe["entry_count"])

    out = g.copy()
    out[stripe["mask"]] = BLACK
    for r, c in sprite_coords:
        out[r, c] = BLACK

    shifted: list[tuple[int, int]] = []
    for r, c in sprite_coords:
        nr = r + dr
        nc = c + dc
        assert _in_bounds(g.shape, nr, nc), "translated sprite must stay in bounds"
        shifted.append((nr, nc))
    for r, c in shifted:
        out[r, c] = sprite_color
    return out


def _make_connected_sprite(max_h: int, max_w: int, n_cells: int, rng: np.random.Generator) -> np.ndarray | None:
    dims = [
        (h, w)
        for h in range(2, max_h + 1)
        for w in range(2, max_w + 1)
        if h * w >= n_cells and (h > 2 or w > 2 or n_cells < h * w)
    ]
    rng.shuffle(dims)
    for h, w in dims:
        for _ in range(60):
            mask = np.zeros((h, w), dtype=bool)
            start = (int(rng.integers(h)), int(rng.integers(w)))
            mask[start] = True
            cells = [start]
            while len(cells) < n_cells:
                frontier = []
                used = set(cells)
                for r, c in cells:
                    for dr, dc in NEIGHBORS4:
                        nr = r + dr
                        nc = c + dc
                        if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in used:
                            frontier.append((nr, nc))
                if not frontier:
                    break
                nr, nc = frontier[int(rng.integers(len(frontier)))]
                mask[nr, nc] = True
                cells.append((nr, nc))
            if len(cells) != n_cells:
                continue

            comp = _components(mask)
            if len(comp) != 1:
                continue
            box_h, box_w = _bbox(comp[0])
            if box_h < 2 or box_w < 2:
                continue
            cropped = mask[
                min(r for r, _ in comp[0]) : max(r for r, _ in comp[0]) + 1,
                min(c for _, c in comp[0]) : max(c for _, c in comp[0]) + 1,
            ]
            if int(cropped.sum()) == cropped.size:
                continue
            if np.array_equal(cropped, np.flipud(cropped)) and np.array_equal(cropped, np.fliplr(cropped)):
                continue
            return cropped
    return None


def _place_sprite(
    grid: np.ndarray,
    sprite: np.ndarray,
    edge: str,
    stripe_span: int,
    gap: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    h, w = grid.shape
    sh, sw = sprite.shape
    if edge == "top":
        r0 = stripe_span + gap
        c0 = int(rng.integers(1, w - sw))
    elif edge == "bottom":
        r0 = h - stripe_span - gap - sh
        c0 = int(rng.integers(1, w - sw))
    elif edge == "left":
        r0 = int(rng.integers(1, h - sh))
        c0 = stripe_span + gap
    else:
        r0 = int(rng.integers(1, h - sh))
        c0 = w - stripe_span - gap - sw

    coords: list[tuple[int, int]] = []
    for rr, cc in np.argwhere(sprite):
        coords.append((r0 + int(rr), c0 + int(cc)))
    return coords


def _build_expected(
    shape: tuple[int, int],
    sprite_coords: list[tuple[int, int]],
    edge: str,
    entry_count: int,
    sprite_color: int = RED,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    dr, dc = _shift_for_stripe(edge, entry_count)
    shifted = [(r + dr, c + dc) for r, c in sprite_coords]
    out = _paint_coords(shape, shifted, sprite_color)
    return out, shifted


def _semantic_checks(grid: np.ndarray, expected: np.ndarray, meta: dict) -> bool:
    if not np.array_equal(main(grid), expected):
        return False

    stripe = _find_stripe(grid)
    if stripe["entry_count"] != meta["entry_count"]:
        return False
    if stripe["span"] != meta["span"]:
        return False
    if stripe["span"] == stripe["entry_count"]:
        return False

    sprite_color, sprite_coords = _find_sprite(grid, stripe["mask"])
    if sprite_color != RED or set(sprite_coords) != set(meta["sprite_coords"]):
        return False

    box_h, box_w = _bbox(sprite_coords)
    if stripe["orientation"] == "horizontal" and box_h > stripe["entry_count"]:
        return False
    if stripe["orientation"] == "vertical" and box_w > stripe["entry_count"]:
        return False

    correct_shift = set(meta["shifted_coords"])
    wrong_raw = set()
    raw_dr, raw_dc = _shift_for_stripe(meta["edge"], meta["span"])
    for r, c in meta["sprite_coords"]:
        wrong_raw.add((r + raw_dr, c + raw_dc))
    if wrong_raw == correct_shift:
        return False

    away_dr, away_dc = -meta["dr"], -meta["dc"]
    wrong_away = {(r + away_dr, c + away_dc) for r, c in meta["sprite_coords"]}
    if wrong_away == correct_shift:
        return False

    if meta["edge"] in ("top", "bottom"):
        wrong_axis = {(r, c + meta["dr"]) for r, c in meta["sprite_coords"]}
    else:
        wrong_axis = {(r + meta["dc"], c) for r, c in meta["sprite_coords"]}
    if wrong_axis == correct_shift:
        return False

    if _sig(expected) in _SEEN_OUTPUT_SIGS:
        return False
    return True


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    edge = ("top", "bottom", "left", "right")[int(rng.integers(4))]
    entry_count = int(rng.integers(2, 6))

    thicknesses = rng.integers(1, 4, size=entry_count).tolist()
    if sum(thicknesses) == entry_count:
        return None
    span = int(sum(thicknesses))

    move_axis_vertical = edge in ("top", "bottom")
    n_cells = int(rng.integers(3, 7))
    if move_axis_vertical:
        sprite = _make_connected_sprite(min(entry_count, 3), 4, n_cells, rng)
    else:
        sprite = _make_connected_sprite(4, min(entry_count, 3), n_cells, rng)
    if sprite is None:
        return None
    sh, sw = sprite.shape

    gap = int(rng.integers(1, 7))
    if move_axis_vertical:
        min_h = max(10, span + gap + sh + 1)
        if min_h > 24:
            return None
        h = int(rng.integers(min_h, min(25, min_h + 6)))
        min_w = max(12, sw + 3)
        w = int(rng.integers(min_w, min(21, max(min_w + 1, min_w + 7))))
    else:
        min_w = max(10, span + gap + sw + 1)
        if min_w > 24:
            return None
        w = int(rng.integers(min_w, min(25, min_w + 6)))
        min_h = max(12, sh + 3)
        h = int(rng.integers(min_h, min(21, max(min_h + 1, min_h + 7))))

    grid = np.zeros((h, w), dtype=int)
    stripe_colors = rng.choice(np.array(STRIPE_COLORS, dtype=int), size=entry_count, replace=False).tolist()

    offset = 0
    for color, thickness in zip(stripe_colors, thicknesses):
        if edge == "top":
            grid[offset : offset + thickness, :] = int(color)
        elif edge == "bottom":
            grid[h - offset - thickness : h - offset, :] = int(color)
        elif edge == "left":
            grid[:, offset : offset + thickness] = int(color)
        else:
            grid[:, w - offset - thickness : w - offset] = int(color)
        offset += thickness

    sprite_coords = _place_sprite(grid, sprite, edge, span, gap, rng)
    for r, c in sprite_coords:
        if grid[r, c] != BLACK:
            return None
        grid[r, c] = RED

    expected, shifted_coords = _build_expected(grid.shape, sprite_coords, edge, entry_count)
    dr, dc = _shift_for_stripe(edge, entry_count)
    meta = {
        "edge": edge,
        "entry_count": entry_count,
        "span": span,
        "sprite_coords": sprite_coords,
        "shifted_coords": shifted_coords,
        "dr": dr,
        "dc": dc,
    }
    if not _semantic_checks(grid, expected, meta):
        return None

    input_sig = _sig(grid)
    if input_sig in _SEEN_INPUT_SIGS:
        return None

    _SEEN_INPUT_SIGS.add(input_sig)
    _SEEN_OUTPUT_SIGS.add(_sig(expected))
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(500):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stripe-compression displacement input")
