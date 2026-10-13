import random

import numpy as np


NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    winners = values[counts == counts.max()]
    assert len(winners) == 1, "expected a unique background color"
    return int(winners[0])


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)

    return comps


def _ordered_line(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    rows = {r for r, _c in coords}
    cols = {c for _r, c in coords}

    if len(rows) == 1:
        row = next(iter(rows))
        ordered = [(row, c) for c in sorted(c for _r, c in coords)]
        assert ordered[-1][1] - ordered[0][1] + 1 == len(ordered), "horizontal line must be contiguous"
        return ordered

    if len(cols) == 1:
        col = next(iter(cols))
        ordered = [(r, col) for r in sorted(r for r, _c in coords)]
        assert ordered[-1][0] - ordered[0][0] + 1 == len(ordered), "vertical line must be contiguous"
        return ordered

    raise AssertionError("each foreground object must be a straight line")


def main(grid: np.ndarray) -> np.ndarray:
    """Reverse each multicolor strip iff its two adjacent end markers have different colors.

    Input structure:
    - The background is the unique most common color.
    - Every non-background connected component is a one-cell-thick horizontal or
      vertical line made of a strip plus one single-pixel marker attached at each end.
    - The interior strip cells form an ordered color sequence; the two endpoint
      cells are control markers and are not part of the strip body.

    Rule:
    - Find every straight foreground line and read its colors from one end to the other.
    - Keep the first and last cells fixed as boundary markers.
    - If the two boundary-marker colors are the same, leave the interior strip
      sequence unchanged.
    - If the two boundary-marker colors differ, reverse only the interior strip
      sequence along the line axis.

    Key insight:
    - The endpoint pixels are functional markers rather than strip content, and
      only the equality-vs-difference relationship between those two colors
      matters; the specific marker colors themselves do not.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)

    comps = _find_components(g != bg)
    assert comps, "expected at least one strip component"

    out = g.copy()
    for comp in comps:
        line = _ordered_line(comp)
        assert len(line) >= 3, "each line needs two markers and at least one strip cell"

        values = [int(g[r, c]) for r, c in line]
        body = values[1:-1]
        if values[0] != values[-1]:
            body = body[::-1]

        for (r, c), color in zip(line[1:-1], body):
            out[r, c] = int(color)

    return out


# --------------- generator ---------------

GRID_HEIGHT_RANGE = (12, 30)
GRID_WIDTH_RANGE = (12, 30)
STRIPE_COUNT_RANGE = (3, 6)
BODY_LENGTH_RANGE = (3, 8)
EDGE_MARGIN = 1
PLACEMENT_ATTEMPTS = 120
GENERATION_ATTEMPTS = 300
PALETTE = tuple(range(10))


def _random_body(length: int, palette: list[int], require_reversal_effect: bool) -> list[int]:
    active_count = random.randint(2, min(4, len(palette)))
    active = random.sample(palette, active_count)
    for _ in range(100):
        body = [random.choice(active) for _ in range(length)]
        if len(set(body)) < 2:
            continue
        if require_reversal_effect and body == body[::-1]:
            continue
        return body
    raise RuntimeError("failed to sample a valid strip body")


def _line_cells(h: int, w: int, orientation: str, total_len: int) -> list[tuple[int, int]] | None:
    if orientation == "h":
        if w - 2 * EDGE_MARGIN < total_len or h <= 2 * EDGE_MARGIN:
            return None
        r = random.randint(EDGE_MARGIN, h - 1 - EDGE_MARGIN)
        c0 = random.randint(EDGE_MARGIN, w - EDGE_MARGIN - total_len)
        return [(r, c0 + i) for i in range(total_len)]

    if h - 2 * EDGE_MARGIN < total_len or w <= 2 * EDGE_MARGIN:
        return None
    c = random.randint(EDGE_MARGIN, w - 1 - EDGE_MARGIN)
    r0 = random.randint(EDGE_MARGIN, h - EDGE_MARGIN - total_len)
    return [(r0 + i, c) for i in range(total_len)]


def _can_place(cells: list[tuple[int, int]], occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and occupied[rr, cc]:
                    return False
    return True


def _expected_from_specs(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for spec in specs:
        body = spec["body"][::-1] if spec["flip"] else spec["body"]
        for (r, c), color in zip(spec["cells"][1:-1], body):
            out[r, c] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    stripe_count = random.randint(*STRIPE_COUNT_RANGE)

    bg = random.choice(PALETTE)
    remaining = [color for color in PALETTE if color != bg]
    marker_colors = random.sample(remaining, 2)
    body_palette = [color for color in remaining if color not in marker_colors]

    flips = [False, True] + [bool(random.getrandbits(1)) for _ in range(stripe_count - 2)]
    orientations = ["h", "v"] + [random.choice(("h", "v")) for _ in range(stripe_count - 2)]
    random.shuffle(flips)
    random.shuffle(orientations)

    specs: list[dict] = []
    occupied = np.zeros((h, w), dtype=bool)
    grid = np.full((h, w), bg, dtype=int)

    lengths = [random.randint(*BODY_LENGTH_RANGE) for _ in range(stripe_count)]
    order = sorted(range(stripe_count), key=lambda i: lengths[i], reverse=True)

    for idx in order:
        orientation = orientations[idx]
        flip = flips[idx]
        body_len = lengths[idx]
        total_len = body_len + 2
        placed = False

        axis_cap = w if orientation == "h" else h
        if total_len > axis_cap - 2 * EDGE_MARGIN:
            return None

        for _ in range(PLACEMENT_ATTEMPTS):
            cells = _line_cells(h, w, orientation, total_len)
            if cells is None or not _can_place(cells, occupied):
                continue

            body = _random_body(body_len, body_palette, require_reversal_effect=flip)
            if flip:
                left_marker, right_marker = marker_colors
                if random.random() < 0.5:
                    left_marker, right_marker = right_marker, left_marker
            else:
                left_marker = right_marker = random.choice(marker_colors)

            sequence = [left_marker, *body, right_marker]
            for (r, c), color in zip(cells, sequence):
                grid[r, c] = int(color)
                occupied[r, c] = True

            specs.append({"cells": cells, "body": body, "flip": flip})
            placed = True
            break

        if not placed:
            return None

    expected = _expected_from_specs(grid, specs)
    return grid if np.array_equal(main(grid), expected) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid strip-flip puzzle")
