from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
SIDES = ("top", "bottom", "left", "right")
GEN_ATTEMPTS = 500
PLACEMENT_ATTEMPTS = 250


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
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _frame_info(comp: list[tuple[int, int]]) -> tuple[int, int, int, int, int, int, str] | None:
    rows = [r for r, _c in comp]
    cols = [c for _r, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 < 3 or c1 - c0 < 3:
        return None

    comp_set = set(comp)
    boundary = set()
    for c in range(c0, c1 + 1):
        boundary.add((r0, c))
        boundary.add((r1, c))
    for r in range(r0 + 1, r1):
        boundary.add((r, c0))
        boundary.add((r, c1))

    if comp_set - boundary:
        return None

    missing = list(boundary - comp_set)
    if len(missing) != 1:
        return None

    gap_r, gap_c = missing[0]
    if (gap_r, gap_c) in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
        return None

    if len(comp_set) != len(boundary) - 1:
        return None

    if gap_r == r0:
        side = "top"
    elif gap_r == r1:
        side = "bottom"
    elif gap_c == c0:
        side = "left"
    else:
        side = "right"
    return r0, c0, r1, c1, gap_r, gap_c, side


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend a seed-colored ray outward through each rectangular frame gap.

    Input structure:
    - The background is black.
    - Several monochrome hollow rectangular frames appear on the grid, and each
      frame has exactly one missing border cell on one side.
    - Each frame interior contains exactly one seed pixel whose color differs from
      the frame color, and a same-colored isolated marker lies farther outside the
      frame in the gap direction.

    Rule:
    - Detect every frame by finding a single-color 4-connected component whose
      bounding-box boundary is complete except for one non-corner gap cell.
    - For each frame, locate the unique non-background interior seed pixel and use
      its color as the drawing color.
    - The gap side gives the outward direction. The ray length is the seed's
      orthogonal distance to that side of the frame.
    - Paint that many cells starting at the gap and extending outward, while
      leaving the original frame, seed, and unrelated pixels unchanged.

    Key insight:
    - The gap determines direction, while the seed encodes both the output color
      and the ray length through its distance to the gap edge.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    out = g.copy()
    for color in np.unique(g):
        color = int(color)
        if color == BG:
            continue
        for comp in _components(g == color):
            frame = _frame_info(comp)
            if frame is None:
                continue
            r0, c0, r1, c1, gap_r, gap_c, side = frame
            seed_cells = [
                (r, c)
                for r in range(r0 + 1, r1)
                for c in range(c0 + 1, c1)
                if int(g[r, c]) not in (BG, color)
            ]
            assert len(seed_cells) == 1, "each frame must contain exactly one seed"
            seed_r, seed_c = seed_cells[0]
            seed_color = int(g[seed_r, seed_c])

            if side == "top":
                length, dr, dc = seed_r - r0, -1, 0
            elif side == "bottom":
                length, dr, dc = r1 - seed_r, 1, 0
            elif side == "left":
                length, dr, dc = seed_c - c0, 0, -1
            else:
                length, dr, dc = c1 - seed_c, 0, 1

            for step in range(length):
                rr = gap_r + dr * step
                cc = gap_c + dc * step
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = seed_color
    return out


def _expanded(cells: set[tuple[int, int]], h: int, w: int, pad: int = 1) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr in range(r - pad, r + pad + 1):
            for nc in range(c - pad, c + pad + 1):
                if 0 <= nr < h and 0 <= nc < w:
                    expanded.add((nr, nc))
    return expanded


def _place_object(
    h: int,
    w: int,
    side: str,
    border_color: int,
    seed_color: int,
    occupied: set[tuple[int, int]],
) -> dict | None:
    for _ in range(PLACEMENT_ATTEMPTS):
        outer_h = random.randint(4, min(8, h - 2))
        outer_w = random.randint(4, min(8, w - 2))
        r0 = random.randint(1, h - outer_h - 1)
        c0 = random.randint(1, w - outer_w - 1)
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1

        if side in ("top", "bottom"):
            gap_c = random.randint(c0 + 1, c1 - 1)
            gap_r = r0 if side == "top" else r1
            extra_blank = random.randint(1, 3)
            outside = r0 if side == "top" else (h - 1 - r1)
            max_length = min(outer_h - 2, outside - extra_blank)
            if max_length < 1:
                continue
            length = random.randint(1, max_length)
            seed_r = r0 + length if side == "top" else r1 - length
            seed_c = random.randint(c0 + 1, c1 - 1)
            dr, dc = (-1, 0) if side == "top" else (1, 0)
        else:
            gap_r = random.randint(r0 + 1, r1 - 1)
            gap_c = c0 if side == "left" else c1
            extra_blank = random.randint(1, 3)
            outside = c0 if side == "left" else (w - 1 - c1)
            max_length = min(outer_w - 2, outside - extra_blank)
            if max_length < 1:
                continue
            length = random.randint(1, max_length)
            seed_r = random.randint(r0 + 1, r1 - 1)
            seed_c = c0 + length if side == "left" else c1 - length
            dr, dc = (0, -1) if side == "left" else (0, 1)

        marker_r = gap_r + dr * (length + extra_blank)
        marker_c = gap_c + dc * (length + extra_blank)
        if not (0 <= marker_r < h and 0 <= marker_c < w):
            continue

        boundary = set()
        for c in range(c0, c1 + 1):
            boundary.add((r0, c))
            boundary.add((r1, c))
        for r in range(r0 + 1, r1):
            boundary.add((r, c0))
            boundary.add((r, c1))
        boundary.remove((gap_r, gap_c))

        seed = (seed_r, seed_c)
        marker = (marker_r, marker_c)
        corridor = {(gap_r + dr * k, gap_c + dc * k) for k in range(length + extra_blank + 1)}
        frame_box = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
        actual = boundary | {seed, marker}
        reserved = _expanded(frame_box | corridor | {marker}, h, w, pad=1)
        if reserved & occupied:
            continue

        return {
            "bbox": (r0, c0, r1, c1),
            "gap": (gap_r, gap_c),
            "side": side,
            "seed": seed,
            "marker": marker,
            "length": length,
            "direction": (dr, dc),
            "boundary": boundary,
            "border_color": border_color,
            "seed_color": seed_color,
            "actual": actual,
            "reserved": reserved,
        }
    return None


def _render_object(inp: np.ndarray, out: np.ndarray, spec: dict) -> None:
    border_color = int(spec["border_color"])
    seed_color = int(spec["seed_color"])
    for r, c in spec["boundary"]:
        inp[r, c] = border_color
        out[r, c] = border_color

    seed_r, seed_c = spec["seed"]
    marker_r, marker_c = spec["marker"]
    inp[seed_r, seed_c] = seed_color
    inp[marker_r, marker_c] = seed_color
    out[seed_r, seed_c] = seed_color
    out[marker_r, marker_c] = seed_color

    gap_r, gap_c = spec["gap"]
    dr, dc = spec["direction"]
    for step in range(spec["length"]):
        out[gap_r + dr * step, gap_c + dc * step] = seed_color


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(16, 30)
    w = random.randint(16, 30)
    n_objects = random.randint(2, 4)

    colors = random.sample(PALETTE, 2 * n_objects)
    border_colors = colors[:n_objects]
    seed_colors = colors[n_objects:]

    sides = list(SIDES)
    random.shuffle(sides)
    sides = sides[:n_objects]

    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    occupied: set[tuple[int, int]] = set()
    specs = []

    order = list(range(n_objects))
    random.shuffle(order)
    for idx in order:
        spec = _place_object(h, w, sides[idx], border_colors[idx], seed_colors[idx], occupied)
        if spec is None:
            return None
        occupied |= spec["reserved"]
        specs.append(spec)

    for spec in specs:
        _render_object(inp, out, spec)

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
