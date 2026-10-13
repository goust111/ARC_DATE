from __future__ import annotations

from collections import deque
import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

BG_COLOR = 0
GRID_MIN = 20
GRID_MAX = 30
HOLE_COUNT_RANGE = (2, 4)
TEMPLATE_MIN_SIDE = 8
TEMPLATE_MAX_SIDE = 16
SHAPE_AREA_RANGE = (3, 6)
SHAPE_MAX_SIDE = 4
MAX_SHAPE_ATTEMPTS = 120
MAX_SET_ATTEMPTS = 240
MAX_GENERATION_ATTEMPTS = 400
PADDING = 1
QUADRANTS = ("tl", "tr", "bl", "br")


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_boundary(cells: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells)


def _cells_to_mask(cells: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(cells)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        out[r - r0, c - c0] = True
    return out


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0 = int(coords[:, 0].min())
    r1 = int(coords[:, 0].max())
    c0 = int(coords[:, 1].min())
    c1 = int(coords[:, 1].max())
    return mask[r0 : r1 + 1, c0 : c1 + 1].astype(bool)


def _mask_key(mask: np.ndarray) -> tuple[str, ...]:
    return tuple("".join("1" if bool(cell) else "0" for cell in row) for row in mask)


def _canonical_rotation_key(mask: np.ndarray) -> tuple[str, ...]:
    base = _trim_mask(mask)
    return min(_mask_key(_trim_mask(np.rot90(base, k))) for k in range(4))


def _rotation_variant_count(mask: np.ndarray) -> int:
    base = _trim_mask(mask)
    return len({_mask_key(_trim_mask(np.rot90(base, k))) for k in range(4)})


def _mask_has_hole(mask: np.ndarray) -> bool:
    bg = ~mask.astype(bool)
    h, w = bg.shape
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if bool(bg[r, c]) and not bool(seen[r, c]):
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bool(bg[r, c]) and not bool(seen[r, c]):
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and bool(bg[nr, nc]) and not bool(seen[nr, nc]):
                seen[nr, nc] = True
                q.append((nr, nc))

    return bool(np.any(bg & ~seen))


def _normalize_coord_set(cells: set[tuple[int, int]]) -> tuple[list[tuple[int, int]], int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0 = min(rows)
    c0 = min(cols)
    norm = sorted((r - r0, c - c0) for r, c in cells)
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    return norm, h, w


def _shape_from_norm(norm: list[tuple[int, int]], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        mask[r, c] = True
    return mask


def _isolated_solid_2x2_blocks(grid: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    h, w = grid.shape
    blocks: list[tuple[int, int, int]] = []
    for r0 in range(h - 1):
        for c0 in range(w - 1):
            block = grid[r0 : r0 + 2, c0 : c0 + 2]
            color = int(block[0, 0])
            if color == bg or not bool(np.all(block == color)):
                continue
            isolated = True
            for r in range(r0, r0 + 2):
                for c in range(c0, c0 + 2):
                    for dr, dc in DIR4:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and not (r0 <= nr < r0 + 2 and c0 <= nc < c0 + 2):
                            if int(grid[nr, nc]) == color:
                                isolated = False
                                break
                    if not isolated:
                        break
                if not isolated:
                    break
            if isolated:
                blocks.append((r0, c0, color))
    return blocks


def _legend_info(grid: np.ndarray, bg: int) -> tuple[dict[str, int], np.ndarray]:
    h, w = grid.shape
    blocks = _isolated_solid_2x2_blocks(grid, bg)
    assert len(blocks) >= 4, "expected four isolated 2x2 legend blocks"

    targets = {
        "tl": (0, 0),
        "tr": (0, w - 2),
        "bl": (h - 2, 0),
        "br": (h - 2, w - 2),
    }
    chosen: dict[str, int] = {}
    legend_mask = np.zeros((h, w), dtype=bool)
    used: set[int] = set()
    for quadrant, (tr, tc) in targets.items():
        candidates = []
        for idx, (r0, c0, color) in enumerate(blocks):
            if idx in used:
                continue
            dist = abs(r0 - tr) + abs(c0 - tc)
            candidates.append((dist, idx, r0, c0, color))
        assert candidates, f"missing legend block for quadrant {quadrant}"
        _, idx, r0, c0, color = min(candidates)
        used.add(idx)
        chosen[quadrant] = int(color)
        legend_mask[r0 : r0 + 2, c0 : c0 + 2] = True

    assert len(set(chosen.values())) == 4, "quadrant legend colors must be distinct"
    return chosen, legend_mask


def _quadrant_from_centroid(cells: list[tuple[int, int]], h: int, w: int) -> str:
    mean_r = sum(r for r, _ in cells) / float(len(cells))
    mean_c = sum(c for _, c in cells) / float(len(cells))
    vertical = "t" if mean_r < (h - 1) / 2.0 else "b"
    horizontal = "l" if mean_c < (w - 1) / 2.0 else "r"
    return vertical + horizontal


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the holes of a central template by matching scattered fragments and recoloring each filled hole by fragment quadrant.

    Input structure:
    - The background is the most common color.
    - A large central monochrome template contains several enclosed background holes.
    - Outside the template, several same-colored fragments each match one hole up to 90-degree rotation.
    - Four isolated 2x2 color blocks near the four grid corners encode the colors for the top-left, top-right, bottom-left, and bottom-right quadrants.

    Rule:
    - Find the corner color blocks and treat them as a quadrant-to-color legend.
    - Identify the central template, extract its enclosed holes, and match each outer fragment to exactly one hole using rotation-invariant shape comparison.
    - Determine each fragment's quadrant from the centroid of its pixels relative to the grid midlines.
    - Produce an output with the template kept in place, each hole filled by the color of the fragment's quadrant, and all legend blocks plus outer fragments removed.

    Key insight:
    - Shape matching and recoloring are independent: topology decides which hole a fragment belongs to, while the fragment's original quadrant decides the fill color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _most_common_color(g)

    quadrant_colors, legend_mask = _legend_info(g, bg)

    fg_mask = (g != bg) & ~legend_mask
    comps = _components(fg_mask)
    assert len(comps) >= 2, "expected a template plus at least one fragment"

    template_idx = max(range(len(comps)), key=lambda idx: len(comps[idx]))
    template_cells = comps[template_idx]
    fragment_cells = [comp for idx, comp in enumerate(comps) if idx != template_idx]
    assert fragment_cells, "expected outer fragments"

    template_mask = np.zeros((h, w), dtype=bool)
    for r, c in template_cells:
        template_mask[r, c] = True
    template_colors = {int(g[r, c]) for r, c in template_cells}
    assert len(template_colors) == 1, "template must be monochrome"
    template_color = template_colors.pop()

    holes: list[list[tuple[int, int]]] = []
    for comp in _components(g == bg):
        if _touches_boundary(comp, h, w):
            continue
        enclosed = True
        for r, c in comp:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not bool(g[nr, nc] == bg):
                    if not bool(template_mask[nr, nc]):
                        enclosed = False
                        break
            if not enclosed:
                break
        if enclosed:
            holes.append(comp)

    assert holes, "expected enclosed template holes"
    assert len(holes) == len(fragment_cells), "hole and fragment counts must match"

    holes_by_key: dict[tuple[str, ...], list[tuple[int, int]]] = {}
    for hole in holes:
        key = _canonical_rotation_key(_cells_to_mask(hole))
        assert key not in holes_by_key, "holes must be unique up to rotation"
        holes_by_key[key] = hole

    matched: dict[tuple[str, ...], int] = {}
    for frag in fragment_cells:
        key = _canonical_rotation_key(_cells_to_mask(frag))
        assert key in holes_by_key, "every fragment must match a hole"
        assert key not in matched, "fragment matching must be one-to-one"
        quadrant = _quadrant_from_centroid(frag, h, w)
        matched[key] = int(quadrant_colors[quadrant])

    assert len(matched) == len(holes)

    out = np.full_like(g, int(bg))
    out[template_mask] = int(template_color)
    for key, color in matched.items():
        for r, c in holes_by_key[key]:
            out[r, c] = int(color)
    return out


def _random_polyomino() -> np.ndarray | None:
    for _ in range(MAX_SHAPE_ATTEMPTS):
        area = random.randint(SHAPE_AREA_RANGE[0], SHAPE_AREA_RANGE[1])
        cells: set[tuple[int, int]] = {(0, 0)}
        while len(cells) < area:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    nbr = (r + dr, c + dc)
                    if nbr not in cells:
                        frontier.add(nbr)
            candidates = []
            for candidate in frontier:
                norm, hh, ww = _normalize_coord_set(cells | {candidate})
                if hh <= SHAPE_MAX_SIDE and ww <= SHAPE_MAX_SIDE:
                    candidates.append(candidate)
            if not candidates:
                break
            cells.add(random.choice(candidates))
        if len(cells) != area:
            continue
        norm, hh, ww = _normalize_coord_set(cells)
        if hh < 2 or ww < 2:
            continue
        mask = _shape_from_norm(norm, hh, ww)
        if hh == 2 and ww == 2 and bool(mask.all()):
            continue
        if _mask_has_hole(mask):
            continue
        if _rotation_variant_count(mask) < 2:
            continue
        return mask
    return None


def _sample_unique_shapes(count: int) -> list[np.ndarray] | None:
    for _ in range(MAX_SET_ATTEMPTS):
        shapes: list[np.ndarray] = []
        keys: set[tuple[str, ...]] = set()
        has_four_way = False
        inner_attempts = 0
        while len(shapes) < count and inner_attempts < MAX_SET_ATTEMPTS:
            inner_attempts += 1
            mask = _random_polyomino()
            if mask is None:
                continue
            key = _canonical_rotation_key(mask)
            if key in keys:
                continue
            keys.add(key)
            shapes.append(mask)
            has_four_way = has_four_way or _rotation_variant_count(mask) == 4
        if len(shapes) == count and has_four_way:
            return shapes
    return None


def _place_holes(template_h: int, template_w: int, shapes: list[np.ndarray]) -> list[tuple[int, int]] | None:
    reserved = np.zeros((template_h, template_w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(shapes)
    order = sorted(range(len(shapes)), key=lambda idx: int(shapes[idx].sum()), reverse=True)

    for idx in order:
        mask = shapes[idx]
        ph, pw = mask.shape
        candidates: list[tuple[int, int]] = []
        for r0 in range(1, template_h - ph):
            for c0 in range(1, template_w - pw):
                if bool(reserved[r0 - 1 : r0 + ph + 1, c0 - 1 : c0 + pw + 1].any()):
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        placements[idx] = (r0, c0)
        reserved[r0 - 1 : r0 + ph + 1, c0 - 1 : c0 + pw + 1] = True

    return [placement for placement in placements if placement is not None]


def _can_place(occupied: np.ndarray, mask: np.ndarray, r0: int, c0: int, padding: int) -> bool:
    h, w = occupied.shape
    ph, pw = mask.shape
    if r0 < 0 or c0 < 0 or r0 + ph > h or c0 + pw > w:
        return False
    for dr, dc in np.argwhere(mask):
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        r1 = max(0, rr - padding)
        r2 = min(h, rr + padding + 1)
        c1 = max(0, cc - padding)
        c2 = min(w, cc + padding + 1)
        if bool(occupied[r1:r2, c1:c2].any()):
            return False
    return True


def _place_piece_in_quadrant(
    occupied: np.ndarray,
    mask: np.ndarray,
    quadrant: str,
) -> tuple[int, int] | None:
    h, w = occupied.shape
    ph, pw = mask.shape
    row_mid = h // 2
    col_mid = w // 2

    if quadrant[0] == "t":
        row_range = range(0, row_mid - ph + 1)
    else:
        row_range = range(row_mid, h - ph + 1)

    if quadrant[1] == "l":
        col_range = range(0, col_mid - pw + 1)
    else:
        col_range = range(col_mid, w - pw + 1)

    candidates = [(r0, c0) for r0 in row_range for c0 in col_range if _can_place(occupied, mask, r0, c0, PADDING)]
    if not candidates:
        return None
    return random.choice(candidates)


def _render_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for dr, dc in np.argwhere(mask):
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        grid[rr, cc] = int(color)
        cells.append((rr, cc))
    return cells


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    hole_count = random.randint(HOLE_COUNT_RANGE[0], HOLE_COUNT_RANGE[1])
    colors = random.sample(list(range(1, 10)), 6)
    template_color = int(colors[0])
    piece_color = int(colors[1])
    quadrant_colors = {
        "tl": int(colors[2]),
        "tr": int(colors[3]),
        "bl": int(colors[4]),
        "br": int(colors[5]),
    }

    max_th = min(TEMPLATE_MAX_SIDE, h - 6)
    max_tw = min(TEMPLATE_MAX_SIDE, w - 6)
    min_th = max(TEMPLATE_MIN_SIDE, 2 * hole_count + 4)
    min_tw = max(TEMPLATE_MIN_SIDE, 2 * hole_count + 4)
    if max_th < min_th or max_tw < min_tw:
        return None

    template_h = random.randint(min_th, max_th)
    template_w = random.randint(min_tw, max_tw)
    template_r0 = (h - template_h) // 2
    template_c0 = (w - template_w) // 2

    shapes = _sample_unique_shapes(hole_count)
    if shapes is None:
        return None
    hole_origins = _place_holes(template_h, template_w, shapes)
    if hole_origins is None:
        return None

    grid = np.full((h, w), BG_COLOR, dtype=int)
    template_mask = np.zeros((h, w), dtype=bool)
    template_mask[template_r0 : template_r0 + template_h, template_c0 : template_c0 + template_w] = True

    hole_cells: list[list[tuple[int, int]]] = []
    for mask, (hr, hc) in zip(shapes, hole_origins):
        cells: list[tuple[int, int]] = []
        for dr, dc in np.argwhere(mask):
            rr = int(template_r0 + hr + dr)
            cc = int(template_c0 + hc + dc)
            template_mask[rr, cc] = False
            cells.append((rr, cc))
        hole_cells.append(cells)

    grid[template_mask] = template_color

    legend_positions = {
        "tl": (0, 0),
        "tr": (0, w - 2),
        "bl": (h - 2, 0),
        "br": (h - 2, w - 2),
    }
    for quadrant, (r0, c0) in legend_positions.items():
        grid[r0 : r0 + 2, c0 : c0 + 2] = quadrant_colors[quadrant]

    occupied = grid != BG_COLOR
    assigned_quadrants = random.sample(list(QUADRANTS), hole_count)

    piece_data: list[tuple[str, list[tuple[int, int]]]] = []
    order = list(range(hole_count))
    random.shuffle(order)
    for idx in order:
        rotated = _trim_mask(np.rot90(shapes[idx], random.randint(0, 3)))
        quadrant = assigned_quadrants[idx]
        position = _place_piece_in_quadrant(occupied, rotated, quadrant)
        if position is None:
            return None
        pr, pc = position
        cells = _render_mask(grid, rotated, pr, pc, piece_color)
        for r, c in cells:
            occupied[r, c] = True
        piece_data.append((quadrant, cells))

    expected = np.full((h, w), BG_COLOR, dtype=int)
    expected[template_mask] = int(template_color)
    for idx, cells in enumerate(hole_cells):
        fill = quadrant_colors[assigned_quadrants[idx]]
        for r, c in cells:
            expected[r, c] = int(fill)

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
