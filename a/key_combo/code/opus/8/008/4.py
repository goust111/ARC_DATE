from __future__ import annotations

import random

import numpy as np


BG = 0
YELLOW = 4
GREY = 5
MARKER_COLORS = [1, 2, 3, 6, 7, 8, 9]
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _decode_arrow(coords: list[tuple[int, int]]) -> tuple[int, int]:
    if len(coords) != 6:
        raise AssertionError("each arrow must have exactly 6 grey pixels")

    rows = sorted({r for r, _ in coords})
    cols = sorted({c for _, c in coords})
    h = rows[-1] - rows[0] + 1
    w = cols[-1] - cols[0] + 1
    row_counts = {r: 0 for r in rows}
    col_counts = {c: 0 for c in cols}
    for r, c in coords:
        row_counts[r] += 1
        col_counts[c] += 1

    if (h, w) == (3, 4):
        shaft_rows = [r for r, cnt in row_counts.items() if cnt == 4]
        cross_cols = [c for c, cnt in col_counts.items() if cnt == 3]
        if len(shaft_rows) != 1 or len(cross_cols) != 1:
            raise AssertionError("invalid horizontal arrow sprite")
        shaft_mid = sum(cols) / len(cols)
        return DIRS["left"] if cross_cols[0] < shaft_mid else DIRS["right"]

    if (h, w) == (4, 3):
        shaft_cols = [c for c, cnt in col_counts.items() if cnt == 4]
        cross_rows = [r for r, cnt in row_counts.items() if cnt == 3]
        if len(shaft_cols) != 1 or len(cross_rows) != 1:
            raise AssertionError("invalid vertical arrow sprite")
        shaft_mid = sum(rows) / len(rows)
        return DIRS["up"] if cross_rows[0] < shaft_mid else DIRS["down"]

    raise AssertionError("grey component is not a valid arrow sprite")


def _find_arrows(grid: np.ndarray) -> list[dict]:
    arrows = []
    for comp in _components_of_color(grid, GREY):
        direction = _decode_arrow(comp)
        arrows.append({"coords": comp, "coord_set": set(comp), "direction": direction})
    return arrows


def _marker_positions(grid: np.ndarray) -> list[tuple[int, int, int]]:
    markers = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            value = int(grid[r, c])
            if value not in (BG, GREY):
                markers.append((r, c, value))
    return markers


def _nearest_arrow(marker: tuple[int, int, int], arrows: list[dict]) -> dict:
    r, c, _ = marker
    best_arrow = None
    best_dist = None
    tied = False
    for arrow in arrows:
        dist = min(abs(r - ar) + abs(c - ac) for ar, ac in arrow["coords"])
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_arrow = arrow
            tied = False
        elif dist == best_dist:
            tied = True
    if tied or best_arrow is None:
        raise AssertionError("each marker must have a unique nearest arrow")
    return best_arrow


def _ray_to_edge(start: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = direction
    r, c = start
    cells = []
    while 0 <= r < h and 0 <= c < w:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend parity-striped rays from marker pixels using nearby arrow sprites as direction encoders, and recolor mixed-color intersections yellow.

    Input structure:
    - The grid has a black background plus 2-4 isolated non-grey marker pixels of distinct colors.
    - Grey pixels form 2-4 separate 6-cell arrow sprites whose bbox is either 3x4 or 4x3.
    - Each marker has a unique nearest arrow under Manhattan distance to any arrow pixel.

    Rule:
    - Decode each arrow direction from the side of its perpendicular crossbar: horizontal arrows point toward the nearer shaft end, and vertical arrows do the same.
    - For each marker, choose its unique nearest arrow and cast a ray from the marker in that arrow's direction all the way to the grid edge.
    - Along a ray, even steps from the marker keep the marker color while odd steps become grey; arrow pixels themselves stay grey and are not overwritten.
    - If rays from different marker colors occupy the same non-arrow cell, recolor that cell yellow; same-color overlaps keep that ray color.

    Key insight:
    - The grey sprites are not endpoints of the trails: their crossbar offset is an implicit direction code that tells each nearby marker which axial ray to emit.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    arrows = _find_arrows(g)
    markers = _marker_positions(g)
    assert 2 <= len(arrows) <= 4, "expected 2-4 arrows"
    assert 2 <= len(markers) <= 4, "expected 2-4 markers"

    arrow_cells = set()
    for arrow in arrows:
        arrow_cells |= arrow["coord_set"]

    hits: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for marker in markers:
        r, c, color = marker
        arrow = _nearest_arrow(marker, arrows)
        path = _ray_to_edge((r, c), arrow["direction"], g.shape)
        for step, cell in enumerate(path):
            if cell in arrow_cells:
                continue
            hits.setdefault(cell, []).append((color, step))

    for (r, c), ray_hits in hits.items():
        if (r, c) in arrow_cells:
            continue
        source_colors = {color for color, _step in ray_hits}
        if len(source_colors) >= 2:
            out[r, c] = YELLOW
            continue
        color = next(iter(source_colors))
        if len(ray_hits) > 1:
            out[r, c] = color
            continue
        out[r, c] = color if ray_hits[0][1] % 2 == 0 else GREY

    return out


def _arrow_patch(direction_name: str) -> np.ndarray:
    patch = np.zeros((3, 4), dtype=int)
    if direction_name == "left":
        patch[1, :] = GREY
        patch[:, 1] = GREY
        return patch
    if direction_name == "right":
        patch[1, :] = GREY
        patch[:, 2] = GREY
        return patch
    patch = np.zeros((4, 3), dtype=int)
    if direction_name == "up":
        patch[:, 1] = GREY
        patch[1, :] = GREY
        return patch
    if direction_name == "down":
        patch[:, 1] = GREY
        patch[2, :] = GREY
        return patch
    raise ValueError("unknown direction")


def _arrow_top_left_for_marker(marker: tuple[int, int], direction_name: str) -> tuple[int, int]:
    r, c = marker
    if direction_name == "right":
        return r - 1, c - 5
    if direction_name == "left":
        return r - 1, c + 2
    if direction_name == "down":
        return r - 5, c - 1
    if direction_name == "up":
        return r + 2, c - 1
    raise ValueError("unknown direction")


def _fits_patch(top: int, left: int, patch: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    ph, pw = patch.shape
    return 0 <= top and 0 <= left and top + ph <= h and left + pw <= w


def _paste_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    ph, pw = patch.shape
    for rr in range(ph):
        for cc in range(pw):
            if int(patch[rr, cc]) != BG:
                grid[top + rr, left + cc] = int(patch[rr, cc])


def _can_place_arrow(occupied: np.ndarray, top: int, left: int, patch: np.ndarray) -> bool:
    ph, pw = patch.shape
    for rr in range(ph):
        for cc in range(pw):
            if int(patch[rr, cc]) != BG and occupied[top + rr, left + cc]:
                return False
    return True


def _cells_of_patch(top: int, left: int, patch: np.ndarray) -> set[tuple[int, int]]:
    cells = set()
    ph, pw = patch.shape
    for rr in range(ph):
        for cc in range(pw):
            if int(patch[rr, cc]) != BG:
                cells.add((top + rr, left + cc))
    return cells


def _marker_has_space(r: int, c: int, direction_name: str, h: int, w: int) -> bool:
    if direction_name == "right":
        return c >= 5 and c <= w - 3
    if direction_name == "left":
        return c >= 2 and c <= w - 6
    if direction_name == "down":
        return r >= 5 and r <= h - 3
    if direction_name == "up":
        return r >= 2 and r <= h - 6
    return False


def _distance_to_edge(r: int, c: int, direction_name: str, h: int, w: int) -> int:
    if direction_name == "right":
        return w - 1 - c
    if direction_name == "left":
        return c
    if direction_name == "down":
        return h - 1 - r
    if direction_name == "up":
        return r
    raise ValueError("unknown direction")


def _marker_too_close(marker: tuple[int, int], used_markers: list[tuple[int, int]], used_arrow_cells: set[tuple[int, int]]) -> bool:
    r, c = marker
    for mr, mc in used_markers:
        if max(abs(r - mr), abs(c - mc)) <= 1:
            return True
    for ar, ac in used_arrow_cells:
        if abs(r - ar) + abs(c - ac) <= 1:
            return True
    return False


def _choose_crossing_pair(h: int, w: int) -> list[tuple[tuple[int, int], str]]:
    horiz_dir = random.choice(["left", "right"])
    vert_dir = random.choice(["up", "down"])

    row_lo = 7 if vert_dir == "down" else 2
    row_hi = h - 4 if vert_dir == "down" else h - 8
    col_lo = 7 if horiz_dir == "right" else 2
    col_hi = w - 4 if horiz_dir == "right" else w - 8
    if row_lo > row_hi or col_lo > col_hi:
        raise ValueError("grid too small for a forced crossing pair")
    pr = random.randint(row_lo, row_hi)
    pc = random.randint(col_lo, col_hi)

    if horiz_dir == "right":
        max_gap_h = min(5, pc - 5)
        gap_h = random.randint(2, max_gap_h)
        marker_h = (pr, pc - gap_h)
    else:
        max_gap_h = min(5, w - 6 - pc)
        gap_h = random.randint(2, max_gap_h)
        marker_h = (pr, pc + gap_h)

    if vert_dir == "down":
        max_gap_v = min(5, pr - 5)
        gap_v = random.randint(2, max_gap_v)
        marker_v = (pr - gap_v, pc)
    else:
        max_gap_v = min(5, h - 6 - pr)
        gap_v = random.randint(2, max_gap_v)
        marker_v = (pr + gap_v, pc)

    return [(marker_h, horiz_dir), (marker_v, vert_dir)]


def _try_add_marker(
    h: int,
    w: int,
    used_markers: list[tuple[int, int]],
    used_arrow_cells: set[tuple[int, int]],
    pairs: list[tuple[tuple[int, int], str]],
) -> bool:
    directions = list(DIRS)
    random.shuffle(directions)
    for direction_name in directions:
        for _ in range(30):
            r = random.randint(1, h - 2)
            c = random.randint(1, w - 2)
            if not _marker_has_space(r, c, direction_name, h, w):
                continue
            if _distance_to_edge(r, c, direction_name, h, w) < 2:
                continue
            marker = (r, c)
            if _marker_too_close(marker, used_markers, used_arrow_cells):
                continue
            patch = _arrow_patch(direction_name)
            top, left = _arrow_top_left_for_marker(marker, direction_name)
            if not _fits_patch(top, left, patch, (h, w)):
                continue
            patch_cells = _cells_of_patch(top, left, patch)
            if patch_cells & used_arrow_cells:
                continue
            if any(max(abs(r - mr), abs(c - mc)) <= 1 for mr, mc in used_markers):
                continue
            if any(abs(r - ar) + abs(c - ac) <= 1 for ar, ac in patch_cells):
                continue
            pairs.append((marker, direction_name))
            used_markers.append(marker)
            used_arrow_cells |= patch_cells
            return True
    return False


def _place_unused_arrow(
    grid: np.ndarray,
    occupied: np.ndarray,
    markers: list[tuple[int, int]],
    h: int,
    w: int,
) -> bool:
    direction_name = random.choice(list(DIRS))
    patch = _arrow_patch(direction_name)
    ph, pw = patch.shape
    positions = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
    random.shuffle(positions)
    for top, left in positions:
        cells = _cells_of_patch(top, left, patch)
        if not _can_place_arrow(occupied, top, left, patch):
            continue
        if any(min(abs(mr - ar) + abs(mc - ac) for ar, ac in cells) < 5 for mr, mc in markers):
            continue
        _paste_patch(grid, patch, top, left)
        _paste_patch(occupied, patch, top, left)
        return True
    return False


def _validate_input(grid: np.ndarray) -> bool:
    try:
        arrows = _find_arrows(grid)
        markers = _marker_positions(grid)
        if not (2 <= len(arrows) <= 4 and 2 <= len(markers) <= 4):
            return False
        arrow_cells = set()
        for arrow in arrows:
            arrow_cells |= arrow["coord_set"]
        for r, c, _color in markers:
            if any(abs(r - ar) + abs(c - ac) <= 1 for ar, ac in arrow_cells):
                return False
            _nearest_arrow((r, c, 1), arrows)
        out = main(grid)
        if np.max(out.shape) > 30:
            return False
        if not np.any(out == YELLOW):
            return False
        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    for _ in range(400):
        h = random.randint(14, 20)
        w = random.randint(14, 20)
        n_markers = random.randint(2, 4)
        n_arrows = n_markers if n_markers == 4 or random.random() < 0.7 else n_markers + 1

        try:
            pairs = _choose_crossing_pair(h, w)
        except ValueError:
            continue

        used_markers = [marker for marker, _direction in pairs]
        used_arrow_cells: set[tuple[int, int]] = set()
        ok = True
        for marker, direction_name in pairs:
            patch = _arrow_patch(direction_name)
            top, left = _arrow_top_left_for_marker(marker, direction_name)
            if not _fits_patch(top, left, patch, (h, w)):
                ok = False
                break
            cells = _cells_of_patch(top, left, patch)
            if cells & used_arrow_cells:
                ok = False
                break
            if any(abs(marker[0] - ar) + abs(marker[1] - ac) <= 1 for ar, ac in cells):
                ok = False
                break
            used_arrow_cells |= cells
        if not ok:
            continue

        while len(pairs) < n_markers:
            if not _try_add_marker(h, w, used_markers, used_arrow_cells, pairs):
                ok = False
                break
        if not ok:
            continue

        colors = random.sample(MARKER_COLORS, len(pairs))
        grid = np.zeros((h, w), dtype=int)
        occupied = np.zeros((h, w), dtype=bool)

        for (marker, direction_name), color in zip(pairs, colors):
            patch = _arrow_patch(direction_name)
            top, left = _arrow_top_left_for_marker(marker, direction_name)
            if not _can_place_arrow(occupied, top, left, patch):
                ok = False
                break
            _paste_patch(grid, patch, top, left)
            _paste_patch(occupied, patch != 0, top, left)
            mr, mc = marker
            if occupied[mr, mc]:
                ok = False
                break
            grid[mr, mc] = color
            occupied[mr, mc] = True
        if not ok:
            continue

        extra_needed = n_arrows - len(pairs)
        for _extra in range(extra_needed):
            if not _place_unused_arrow(grid, occupied, [marker for marker, _d in pairs], h, w):
                ok = False
                break
        if not ok:
            continue

        if _validate_input(grid):
            return grid

    raise RuntimeError("failed to generate a valid input")
