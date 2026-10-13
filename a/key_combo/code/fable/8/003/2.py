import random
from collections import Counter, deque

import numpy as np


BG_COLOR = 0
FRAME_COLOR = 5
MARKER_PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRY_LIMIT = 600


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_of_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
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
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def is_rectangular_frame(coords: np.ndarray) -> bool:
    if coords.shape[0] < 8:
        return False
    r0, c0, r1, c1 = bbox_from_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if coords.shape[0] != 2 * (h + w) - 4:
        return False
    cells = {tuple(map(int, xy)) for xy in coords}
    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            on_border = rr in (r0, r1) or cc in (c0, c1)
            if on_border != ((rr, cc) in cells):
                return False
    return True


def chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def ring_cells(
    shape: tuple[int, int],
    center: tuple[int, int],
    distance: int,
    *,
    bbox: tuple[int, int, int, int] | None = None,
) -> list[tuple[int, int]]:
    h, w = shape
    cr, cc = center
    r_lo = max(0, cr - distance)
    r_hi = min(h - 1, cr + distance)
    c_lo = max(0, cc - distance)
    c_hi = min(w - 1, cc + distance)
    cells = []
    for r in range(r_lo, r_hi + 1):
        for c in range(c_lo, c_hi + 1):
            if chebyshev((r, c), center) != distance:
                continue
            if bbox is not None:
                r0, c0, r1, c1 = bbox
                if not (r0 <= r <= r1 and c0 <= c <= c1):
                    continue
            cells.append((r, c))
    return cells


def parse_puzzle(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)
    colors = sorted(int(x) for x in np.unique(g) if int(x) != bg)
    assert colors, "expected non-background objects"

    parsed = None
    for frame_color in colors:
        comps = components_of_mask(g == frame_color)
        if len(comps) != 2:
            continue
        frame_candidates = [comp for comp in comps if is_rectangular_frame(comp)]
        if len(frame_candidates) != 1:
            continue
        frame_coords = frame_candidates[0]
        other = comps[0] if comps[1] is frame_coords else comps[1]
        if other.shape[0] != 1:
            continue
        r0, c0, r1, c1 = bbox_from_coords(frame_coords)
        cr, cc = map(int, other[0])
        if not (r0 < cr < r1 and c0 < cc < c1):
            continue

        markers = []
        ok = True
        for color in colors:
            if color == frame_color:
                continue
            color_comps = components_of_mask(g == color)
            if len(color_comps) != 1 or color_comps[0].shape[0] != 1:
                ok = False
                break
            mr, mc = map(int, color_comps[0][0])
            if r0 <= mr <= r1 and c0 <= mc <= c1:
                ok = False
                break
            markers.append((mr, mc, int(color)))
        if not ok or not markers:
            continue

        distances = [chebyshev((mr, mc), (cr, cc)) for mr, mc, _ in markers]
        if len(set(distances)) != len(distances):
            continue

        interior = np.zeros_like(g, dtype=bool)
        interior[r0 + 1 : r1, c0 + 1 : c1] = True
        hit_counts = [sum(1 for cell in ring_cells(g.shape, (cr, cc), d, bbox=(r0 + 1, c0 + 1, r1 - 1, c1 - 1)) if g[cell] == bg) for d in distances]
        if any(count == 0 for count in hit_counts):
            continue

        parsed = {
            "grid": g,
            "bg": int(bg),
            "frame_color": int(frame_color),
            "frame_bbox": (r0, c0, r1, c1),
            "control": (cr, cc),
            "markers": markers,
        }
        break

    assert parsed is not None, "could not identify frame, control pixel, and external singleton markers"
    return parsed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Light each interior Chebyshev ring whose radius matches an external marker's distance to the control pixel.

    Input structure:
    - The grid has a dominant background plus one one-pixel-thick rectangular frame and one singleton control pixel of the same color inside it.
    - The frame color forms exactly two 4-connected components: the border frame and the internal control pixel.
    - All other non-background colors are singleton markers placed outside the frame, and their Chebyshev distances to the control pixel are pairwise distinct.

    Rule:
    - Find the control pixel inside the rectangular frame.
    - For each external marker, compute its Chebyshev distance d to the control pixel.
    - Inside the frame interior, recolor every background cell whose Chebyshev distance to the control pixel is exactly d with that marker's color; clip the ring by the frame bounds and leave non-background interior cells unchanged.
    - Keep the frame, control pixel, and external markers exactly as they are.

    Key insight:
    - The markers do not encode rows, columns, or directions; each marker only names a radius, and that radius selects an implicit concentric square ring around the control pixel.
    """
    parsed = parse_puzzle(np.asarray(grid, dtype=int))
    g = parsed["grid"]
    out = g.copy()
    bg = parsed["bg"]
    r0, c0, r1, c1 = parsed["frame_bbox"]
    cr, cc = parsed["control"]

    for mr, mc, color in parsed["markers"]:
        distance = chebyshev((mr, mc), (cr, cc))
        for rr, cc2 in ring_cells(g.shape, (cr, cc), distance, bbox=(r0 + 1, c0 + 1, r1 - 1, c1 - 1)):
            if out[rr, cc2] == bg:
                out[rr, cc2] = int(color)
    return out


def _distance_catalog(
    shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    control: tuple[int, int],
) -> dict[int, dict]:
    h, w = shape
    r0, c0, r1, c1 = frame_bbox
    cr, cc = control
    interior_bbox = (r0 + 1, c0 + 1, r1 - 1, c1 - 1)
    catalog: dict[int, dict] = {}
    max_d = max(h, w)
    for d in range(1, max_d + 1):
        inside = ring_cells(shape, control, d, bbox=interior_bbox)
        if not inside:
            continue
        outside = []
        aligned = []
        diagonal = []
        for rr, cc2 in ring_cells(shape, control, d):
            if r0 <= rr <= r1 and c0 <= cc2 <= c1:
                continue
            outside.append((rr, cc2))
            if rr == cr or cc2 == cc:
                aligned.append((rr, cc2))
            else:
                diagonal.append((rr, cc2))
        if outside:
            catalog[d] = {
                "inside": inside,
                "outside": outside,
                "aligned": aligned,
                "diagonal": diagonal,
            }
    return catalog


def _sample_distances(
    catalog: dict[int, dict],
    control: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
) -> list[int] | None:
    valid = sorted(catalog)
    if len(valid) < 2:
        return None
    r0, c0, r1, c1 = frame_bbox
    cr, cc = control
    near_side = min(cr - (r0 + 1), (r1 - 1) - cr, cc - (c0 + 1), (c1 - 1) - cc)
    clipped = [d for d in valid if d > near_side]
    aligned = [d for d in valid if catalog[d]["aligned"]]
    diagonal = [d for d in valid if catalog[d]["diagonal"]]
    if not aligned or not diagonal or not clipped:
        return None

    upper = min(5, len(valid))
    lower = 2 if len(valid) < 4 else 3
    count = random.randint(lower, upper)
    if len(valid) >= 4:
        count = min(count, len(valid) - 1)

    chosen = {random.choice(aligned), random.choice(diagonal), random.choice(clipped)}
    while len(chosen) < count:
        chosen.add(random.choice(valid))
    chosen_list = sorted(chosen)

    if len(valid) >= 4:
        interior_gap = [d for d in valid if chosen_list[0] < d < chosen_list[-1] and d not in chosen]
        if not interior_gap:
            return None
    return chosen_list


def _pick_marker_positions(
    catalog: dict[int, dict],
    chosen: list[int],
    control: tuple[int, int],
) -> list[tuple[int, int]] | None:
    cr, cc = control
    placed: list[tuple[int, int]] = []
    used = set()
    for d in chosen:
        options = catalog[d]["outside"][:]
        random.shuffle(options)
        primary = [cell for cell in options if (cell[0] == cr or cell[1] == cc)]
        secondary = [cell for cell in options if cell not in primary]
        if d == chosen[0] and primary:
            options = primary + secondary
        elif d == chosen[1] and secondary:
            options = secondary + primary
        else:
            options = secondary + primary if random.random() < 0.6 else primary + secondary

        pick = None
        for cell in options:
            if cell in used:
                continue
            if any(chebyshev(cell, other) <= 1 for other in placed):
                continue
            pick = cell
            break
        if pick is None:
            return None
        placed.append(pick)
        used.add(pick)
    return placed


def _build_grid(
    shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    control: tuple[int, int],
    distances: list[int],
    marker_positions: list[tuple[int, int]],
) -> np.ndarray:
    h, w = shape
    r0, c0, r1, c1 = frame_bbox
    cr, cc = control
    grid = np.full((h, w), BG_COLOR, dtype=int)
    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR
    grid[cr, cc] = FRAME_COLOR
    colors = random.sample(MARKER_PALETTE, len(distances))
    for (mr, mc), color in zip(marker_positions, colors):
        grid[mr, mc] = int(color)
    return grid


def _semantic_check(grid: np.ndarray) -> bool:
    try:
        parsed = parse_puzzle(grid)
        out = main(grid)
    except AssertionError:
        return False

    g = parsed["grid"]
    bg = parsed["bg"]
    r0, c0, r1, c1 = parsed["frame_bbox"]
    cr, cc = parsed["control"]
    markers = parsed["markers"]
    if len(markers) < 2:
        return False

    aligned_count = sum(1 for mr, mc, _ in markers if mr == cr or mc == cc)
    diagonal_count = len(markers) - aligned_count
    if aligned_count == 0 or diagonal_count == 0:
        return False

    distances = [chebyshev((mr, mc), (cr, cc)) for mr, mc, _ in markers]
    if len(set(distances)) != len(distances):
        return False

    near_side = min(cr - (r0 + 1), (r1 - 1) - cr, cc - (c0 + 1), (c1 - 1) - cc)
    if not any(d > near_side for d in distances):
        return False

    valid = sorted(_distance_catalog(g.shape, parsed["frame_bbox"], parsed["control"]))
    if len(valid) >= 4:
        if not any(distances[0] < d < distances[-1] and d not in distances for d in valid):
            return False

    painted_any = False
    for mr, mc, color in markers:
        d = chebyshev((mr, mc), (cr, cc))
        ring = [(rr, cc2) for rr, cc2 in ring_cells(g.shape, (cr, cc), d, bbox=(r0 + 1, c0 + 1, r1 - 1, c1 - 1)) if g[rr, cc2] == bg]
        if not ring:
            return False
        if not all(int(out[rr, cc2]) == int(color) for rr, cc2 in ring):
            return False
        painted_any = painted_any or bool(ring)

    if not painted_any or np.array_equal(out, g):
        return False

    unchanged = g.copy()
    for mr, mc, _ in markers:
        unchanged[mr, mc] = bg
    changed = out.copy()
    for mr, mc, _ in markers:
        changed[mr, mc] = bg
    changed[r0, c0 : c1 + 1] = bg
    changed[r1, c0 : c1 + 1] = bg
    changed[r0 : r1 + 1, c0] = bg
    changed[r0 : r1 + 1, c1] = bg
    changed[cr, cc] = bg
    unchanged[r0, c0 : c1 + 1] = bg
    unchanged[r1, c0 : c1 + 1] = bg
    unchanged[r0 : r1 + 1, c0] = bg
    unchanged[r0 : r1 + 1, c1] = bg
    unchanged[cr, cc] = bg
    if np.any((changed != bg) & (unchanged != bg)):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(16, 30)
    w = random.randint(16, 30)
    frame_h = random.randint(7, min(14, h - 4))
    frame_w = random.randint(7, min(14, w - 4))
    r0 = random.randint(2, h - frame_h - 2)
    c0 = random.randint(2, w - frame_w - 2)
    r1 = r0 + frame_h - 1
    c1 = c0 + frame_w - 1

    interior_rows = list(range(r0 + 1, r1))
    interior_cols = list(range(c0 + 1, c1))
    if len(interior_rows) < 3 or len(interior_cols) < 3:
        return None

    candidate_controls = [(r, c) for r in interior_rows for c in interior_cols]
    random.shuffle(candidate_controls)
    for control in candidate_controls[:30]:
        catalog = _distance_catalog((h, w), (r0, c0, r1, c1), control)
        chosen = _sample_distances(catalog, control, (r0, c0, r1, c1))
        if chosen is None:
            continue
        positions = _pick_marker_positions(catalog, chosen, control)
        if positions is None:
            continue
        grid = _build_grid((h, w), (r0, c0, r1, c1), control, chosen, positions)
        if _semantic_check(grid):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid remote-ring-switch input")
