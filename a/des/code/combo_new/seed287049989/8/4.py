import random
from collections import deque

import numpy as np


BG = 0
TEMPLATE_COLOR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 200

L_PATTERNS = {
    0: ((0, 0), (0, 1), (1, 0)),
    1: ((0, 0), (0, 1), (1, 1)),
    2: ((0, 0), (1, 0), (1, 1)),
    3: ((0, 1), (1, 0), (1, 1)),
}
PATTERN_TO_ID = {tuple(sorted(coords)): idx for idx, coords in L_PATTERNS.items()}


def neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def normalize(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = bbox(coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def l_shape_id(coords: list[tuple[int, int]]) -> int | None:
    if len(coords) != 3:
        return None
    r0, c0, r1, c1 = bbox(coords)
    if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
        return None
    return PATTERN_TO_ID.get(normalize(coords))


def coords_to_mask(shape: tuple[int, int], coords: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


def largest_non_bg_component(grid: np.ndarray, bg: int) -> dict:
    best = None
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in component_coords(grid == color):
            comp = {
                "color": color,
                "coords": coords,
                "area": len(coords),
                "bbox": bbox(coords),
            }
            if best is None or comp["area"] > best["area"]:
                best = comp
    assert best is not None, "expected at least one non-background component"
    return best


def find_notches(grid: np.ndarray, template_mask: np.ndarray, bg: int) -> list[dict]:
    r0, c0, r1, c1 = bbox(list(map(tuple, np.argwhere(template_mask))))
    notches: list[dict] = []
    used_cells: set[tuple[int, int]] = set()
    for r in range(r0, r1):
        for c in range(c0, c1):
            window = [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]
            template_cells = [(rr, cc) for rr, cc in window if template_mask[rr, cc]]
            if len(template_cells) != 1:
                continue
            notch_cells = [(rr, cc) for rr, cc in window if not template_mask[rr, cc]]
            if any(int(grid[rr, cc]) != bg for rr, cc in notch_cells):
                continue
            if any(cell in used_cells for cell in notch_cells):
                raise AssertionError("overlapping notch windows are not allowed")
            used_cells.update(notch_cells)
            orientation = l_shape_id(notch_cells)
            assert orientation is not None, "notch must be a 3-cell L shape"
            notches.append(
                {
                    "window": (r, c),
                    "cells": notch_cells,
                    "orientation": orientation,
                }
            )
    assert notches, "expected at least one template notch"
    return notches


def classify_payload_components(grid: np.ndarray, template_mask: np.ndarray, bg: int):
    payload_mask = (grid != bg) & (~template_mask)
    pieces: list[dict] = []
    markers: list[dict] = []
    for coords in component_coords(payload_mask):
        colors = {int(grid[r, c]) for r, c in coords}
        if len(coords) == 1:
            (r, c) = coords[0]
            markers.append({"coord": (r, c), "color": int(grid[r, c])})
            continue
        assert len(colors) == 1, "payload objects must be monochrome"
        orientation = l_shape_id(coords)
        assert orientation is not None, "payload components must be 3-cell L pieces"
        pieces.append(
            {
                "coords": coords,
                "color": next(iter(colors)),
                "orientation": orientation,
                "bbox": bbox(coords),
            }
        )
    return pieces, markers


def touches_notch(coord: tuple[int, int], notch_cells: list[tuple[int, int]]) -> bool:
    r, c = coord
    return any(abs(r - rr) + abs(c - cc) == 1 for rr, cc in notch_cells)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate scattered 3-cell L pieces into the grey template's L notches, then recolor each inserted piece from its local marker.

    Input structure:
    - Black is the background.
    - One large grey connected component is the template; inside its bounding box, some 2x2 edge windows contain exactly one grey cell and three black cells, forming missing L-shaped notches.
    - Away from the template are several monochrome 3-cell L pieces, one per notch, plus one isolated 1-cell color marker adjacent to each notch.

    Rule:
    - Treat each 3-black-cell 2x2 window inside the template bbox as a notch that can be filled by a rotated scattered L piece.
    - Clear every scattered piece from its original location.
    - For each notch, find the unique adjacent 1-cell marker, remove that marker, and paint the notch cells with the marker's color.
    - Keep all original grey template cells unchanged and leave the rest of the black background untouched.

    Key insight:
    The movable pieces only certify that every gap is a rotatable L-shape; the actual inserted color comes from the tiny marker beside each notch, not from the piece's own color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)

    template = largest_non_bg_component(g, bg)
    template_mask = coords_to_mask(g.shape, template["coords"])
    notches = find_notches(g, template_mask, bg)
    pieces, markers = classify_payload_components(g, template_mask, bg)

    assert len(pieces) == len(notches), "piece count must equal notch count"
    assert len(markers) == len(notches), "marker count must equal notch count"

    out = g.copy()
    for piece in pieces:
        for r, c in piece["coords"]:
            out[r, c] = bg
    for marker in markers:
        r, c = marker["coord"]
        out[r, c] = bg

    used_markers: set[int] = set()
    for notch in notches:
        matches = [i for i, marker in enumerate(markers) if touches_notch(marker["coord"], notch["cells"])]
        assert len(matches) == 1, "each notch must have exactly one adjacent marker"
        idx = matches[0]
        assert idx not in used_markers, "markers must map one-to-one to notches"
        used_markers.add(idx)
        color = int(markers[idx]["color"])
        for r, c in notch["cells"]:
            out[r, c] = color

    return out


def reserve(blocked: np.ndarray, cells: list[tuple[int, int]], radius: int = 1):
    h, w = blocked.shape
    for r, c in cells:
        for rr in range(max(0, r - radius), min(h, r + radius + 1)):
            for cc in range(max(0, c - radius), min(w, c + radius + 1)):
                blocked[rr, cc] = True


def choose_non_overlapping_starts(count: int, width: int, rng: random.Random) -> list[int] | None:
    starts = list(range(1, width - 2))
    rng.shuffle(starts)
    chosen: list[int] = []
    for start in starts:
        if all(abs(start - prev) >= 3 for prev in chosen):
            chosen.append(start)
            if len(chosen) == count:
                return chosen
    return None


def marker_candidates(
    notch_cells: list[tuple[int, int]],
    edge: str,
    grid_shape: tuple[int, int],
    template_row0: int,
    template_row1: int,
) -> list[tuple[int, int]]:
    h, w = grid_shape
    out: set[tuple[int, int]] = set()
    if edge == "top":
        for r, c in notch_cells:
            if r == template_row0 and r - 1 >= 0:
                out.add((r - 1, c))
    else:
        for r, c in notch_cells:
            if r == template_row1 and r + 1 < h:
                out.add((r + 1, c))
    return list(out)


def place_piece(
    grid: np.ndarray,
    blocked: np.ndarray,
    color: int,
    orientation: int,
    min_col: int,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    positions = [(r, c) for r in range(h - 1) for c in range(min_col, w - 1)]
    rng.shuffle(positions)
    rel = L_PATTERNS[orientation]
    for r0, c0 in positions:
        cells = [(r0 + dr, c0 + dc) for dr, dc in rel]
        if any(grid[r, c] != BG or blocked[r, c] for r, c in cells):
            continue
        for r, c in cells:
            grid[r, c] = color
        reserve(blocked, cells)
        return cells
    return None


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    notch_count = rng.randint(2, 4)
    notch_orientations = rng.sample(list(L_PATTERNS), notch_count)
    top_count = sum(orientation in (0, 1) for orientation in notch_orientations)
    bottom_count = notch_count - top_count

    width_min = 10 if max(top_count, bottom_count) <= 1 else 13
    template_h = rng.randint(8, 13)
    template_w = rng.randint(width_min, 14)

    grid_h = rng.randint(max(18, template_h + 5), 30)
    grid_w = rng.randint(max(20, template_w + 9), 30)

    max_template_col = grid_w - template_w - 6
    if max_template_col < 2:
        return None
    template_r0 = rng.randint(2, grid_h - template_h - 2)
    template_c0 = rng.randint(2, max_template_col)
    template_r1 = template_r0 + template_h - 1
    template_c1 = template_c0 + template_w - 1

    top_starts = choose_non_overlapping_starts(top_count, template_w, rng)
    bottom_starts = choose_non_overlapping_starts(bottom_count, template_w, rng)
    if top_starts is None or bottom_starts is None:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    grid[template_r0 : template_r1 + 1, template_c0 : template_c1 + 1] = TEMPLATE_COLOR

    notch_specs: list[dict] = []
    top_iter = iter(top_starts)
    bottom_iter = iter(bottom_starts)
    for orientation in notch_orientations:
        local_c0 = next(top_iter) if orientation in (0, 1) else next(bottom_iter)
        local_r0 = 0 if orientation in (0, 1) else template_h - 2
        window_r0 = template_r0 + local_r0
        window_c0 = template_c0 + local_c0
        cells = [(window_r0 + dr, window_c0 + dc) for dr, dc in L_PATTERNS[orientation]]
        for r, c in cells:
            grid[r, c] = BG
        notch_specs.append(
            {
                "cells": cells,
                "orientation": orientation,
                "edge": "top" if orientation in (0, 1) else "bottom",
            }
        )

    chosen_colors = rng.sample(PALETTE, 2 * notch_count)
    piece_colors = chosen_colors[:notch_count]
    marker_colors = chosen_colors[notch_count:]

    blocked = np.zeros_like(grid, dtype=bool)
    reserve(
        blocked,
        [(r, c) for r in range(template_r0, template_r1 + 1) for c in range(template_c0, template_c1 + 1)],
        radius=0,
    )

    for notch, marker_color in zip(notch_specs, marker_colors):
        candidates = marker_candidates(
            notch["cells"],
            notch["edge"],
            grid.shape,
            template_r0,
            template_r1,
        )
        rng.shuffle(candidates)
        placed = None
        for r, c in candidates:
            if grid[r, c] != BG or blocked[r, c]:
                continue
            grid[r, c] = marker_color
            reserve(blocked, [(r, c)])
            placed = (r, c)
            break
        if placed is None:
            return None
        notch["marker"] = placed
        notch["marker_color"] = marker_color

    min_piece_col = template_c1 + 2
    sorted_notches = sorted(notch_specs, key=lambda item: (min(r for r, _ in item["cells"]), min(c for _, c in item["cells"])))
    piece_orientations = [((notch["orientation"] - rng.randrange(4)) % 4) for notch in sorted_notches]
    rng.shuffle(piece_orientations)

    piece_cells: list[list[tuple[int, int]]] = []
    for color, orientation in zip(piece_colors, piece_orientations):
        cells = place_piece(grid, blocked, color, orientation, min_piece_col, rng)
        if cells is None:
            return None
        piece_cells.append(cells)

    if get_bg(grid) != BG:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    for cells, color in zip((notch["cells"] for notch in notch_specs), marker_colors):
        for r, c in cells:
            if int(out[r, c]) != int(color):
                return None
    for cells in piece_cells:
        for r, c in cells:
            if int(out[r, c]) != BG:
                return None
    for notch in notch_specs:
        mr, mc = notch["marker"]
        if int(out[mr, mc]) != BG:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
