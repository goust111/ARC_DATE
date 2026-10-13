import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _boundary_mask(h: int, w: int, orient: str, line: int, side: str) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    if orient == "H":
        if side == "top":
            mask[0, :] = True
            mask[:line, 0] = True
            mask[:line, w - 1] = True
        else:
            mask[h - 1, :] = True
            mask[line + 1 :, 0] = True
            mask[line + 1 :, w - 1] = True
    else:
        if side == "left":
            mask[:, 0] = True
            mask[0, :line] = True
            mask[h - 1, :line] = True
        else:
            mask[:, w - 1] = True
            mask[0, line + 1 :] = True
            mask[h - 1, line + 1 :] = True
    return mask


def _count_source_candidates(grid: np.ndarray, bg: int, divider_color: int, orient: str, line: int, side: str) -> float:
    h, w = grid.shape
    if orient == "H":
        if side == "top":
            side_mask = np.zeros((h, w), dtype=bool)
            side_mask[:line, :] = True
            side_non_bg = np.sum((grid[side_mask] != bg) & (grid[side_mask] != divider_color))
            marker_mask = _boundary_mask(h, w, orient, line, side)
            marker_non_bg = np.sum((grid[marker_mask] != bg) & (grid[marker_mask] != divider_color))
            return float(side_non_bg) + 1.5 * float(marker_non_bg)
        else:
            side_mask = np.zeros((h, w), dtype=bool)
            side_mask[line + 1 :, :] = True
            side_non_bg = np.sum((grid[side_mask] != bg) & (grid[side_mask] != divider_color))
            marker_mask = _boundary_mask(h, w, orient, line, side)
            marker_non_bg = np.sum((grid[marker_mask] != bg) & (grid[marker_mask] != divider_color))
            return float(side_non_bg) + 1.5 * float(marker_non_bg)
    else:
        if side == "left":
            side_mask = np.zeros((h, w), dtype=bool)
            side_mask[:, :line] = True
            side_non_bg = np.sum((grid[side_mask] != bg) & (grid[side_mask] != divider_color))
            marker_mask = _boundary_mask(h, w, orient, line, side)
            marker_non_bg = np.sum((grid[marker_mask] != bg) & (grid[marker_mask] != divider_color))
            return float(side_non_bg) + 1.5 * float(marker_non_bg)
        else:
            side_mask = np.zeros((h, w), dtype=bool)
            side_mask[:, line + 1 :] = True
            side_non_bg = np.sum((grid[side_mask] != bg) & (grid[side_mask] != divider_color))
            marker_mask = _boundary_mask(h, w, orient, line, side)
            marker_non_bg = np.sum((grid[marker_mask] != bg) & (grid[marker_mask] != divider_color))
            return float(side_non_bg) + 1.5 * float(marker_non_bg)


def _find_divider(grid: np.ndarray):
    h, w = grid.shape
    bg = _most_common_color(grid)

    best = None
    best_score = -1.0

    # Horizontal candidates.
    for line in range(1, h - 1):
        color = int(grid[line, 0])
        if np.all(grid[line] == color) and color != bg:
            for side in ("top", "bottom"):
                score = _count_source_candidates(grid, bg, color, "H", line, side)
                if score > best_score:
                    best_score = score
                    best = ("H", line, color, side, score)

    # Vertical candidates.
    for line in range(1, w - 1):
        color = int(grid[0, line])
        if np.all(grid[:, line] == color) and color != bg:
            for side in ("left", "right"):
                score = _count_source_candidates(grid, bg, color, "V", line, side)
                if score > best_score:
                    best_score = score
                    best = ("V", line, color, side, score)

    if best is None or best_score <= 0:
        # No reliable divider found.
        return None
    _, line, color, side, _ = best
    return {
        "orient": best[0],
        "line": line,
        "divider_color": color,
        "source_side": side,
    }


def _iter_component(grid: np.ndarray, r0: int, r1: int, c0: int, c1: int, bg: int, divider_color: int):
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps = {}
    for r in range(r0, r1):
        for c in range(c0, c1):
            color = int(grid[r, c])
            if visited[r, c] or color == bg or color == divider_color:
                continue
            coords = []
            stack = [(r, c)]
            visited[r, c] = True
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                if cr > 0:
                    nr, nc = cr - 1, cc
                    if r0 <= nr < r1 and c0 <= nc < c1 and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
                if cr + 1 < h:
                    nr, nc = cr + 1, cc
                    if r0 <= nr < r1 and c0 <= nc < c1 and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
                if cc > 0:
                    nr, nc = cr, cc - 1
                    if r0 <= nr < r1 and c0 <= nc < c1 and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
                if cc + 1 < w:
                    nr, nc = cr, cc + 1
                    if r0 <= nr < r1 and c0 <= nc < c1 and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            comps.setdefault(color, []).append(coords)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """Copy paired source sprites into the opposite side by reflecting across the divider.

Input structure:
- The background is dominant.
- A single full horizontal or vertical grey divider line splits source and target regions.
- In the source region there are small color sprites (2 to 4 cells) and matching single-pixel color markers on its outer boundary.
- The target region starts as pure background.

Rule:
- Keep the original grid unchanged.
- Detect the divider and choose the source side containing the boundary markers.
- For each marker color, pair it with the source sprite of the same color.
- Infer projection direction from marker to sprite (up/down/left/right); only sprites whose inferred direction continues across the divider are cast.
- Mirror every pixel of each paired sprite across the divider line: distance to divider on the normal axis is mirrored while the divider-axis coordinate stays the same.
- Copy pixels to the target side and leave unmatched sprites unchanged.

Key insight:
- The boundary markers identify which color sprites participate and, through their position relative to the sprite, determine the shooting direction; the divider then fully defines the mirrored landing transform.
"""
    out = np.array(grid, copy=True)
    if grid.ndim != 2:
        return out

    divider_info = _find_divider(grid)
    if divider_info is None:
        return out

    orient = divider_info["orient"]
    line = divider_info["line"]
    divider_color = divider_info["divider_color"]
    source_side = divider_info["source_side"]

    h, w = grid.shape
    bg = _most_common_color(grid)

    if orient == "H":
        if source_side == "top":
            src_r0, src_r1 = 0, line
            src_c0, src_c1 = 0, w
            expect_dir = "down"
        else:
            src_r0, src_r1 = line + 1, h
            src_c0, src_c1 = 0, w
            expect_dir = "up"
    else:
        if source_side == "left":
            src_r0, src_r1 = 0, h
            src_c0, src_c1 = 0, line
            expect_dir = "right"
        else:
            src_r0, src_r1 = 0, h
            src_c0, src_c1 = line + 1, w
            expect_dir = "left"

    comps_by_color = _iter_component(grid, src_r0, src_r1, src_c0, src_c1, bg, divider_color)

    marker_mask = _boundary_mask(h, w, orient, line, source_side)
    markers = np.argwhere(marker_mask & (grid != bg) & (grid != divider_color))
    markers_by_color = {}
    for r, c in markers:
        markers_by_color.setdefault(int(grid[r, c]), []).append((int(r), int(c)))

    for color, comp_list in comps_by_color.items():
        if color not in markers_by_color or len(comp_list) != 1:
            continue
        comp = comp_list[0]
        marker = markers_by_color[color][0]

        rows = [p[0] for p in comp]
        cols = [p[1] for p in comp]
        cr = float(np.mean(rows))
        cc = float(np.mean(cols))

        dr = cr - marker[0]
        dc = cc - marker[1]

        if orient == "H":
            if abs(dr) < abs(dc):
                continue
            if (dr <= 0 and expect_dir == "down") or (dr >= 0 and expect_dir == "up"):
                continue
        else:
            if abs(dc) < abs(dr):
                continue
            if (dc <= 0 and expect_dir == "right") or (dc >= 0 and expect_dir == "left"):
                continue

        for sr, sc in comp:
            if orient == "H":
                tr, tc = 2 * line - sr, sc
            else:
                tr, tc = sr, 2 * line - sc
            if 0 <= tr < h and 0 <= tc < w:
                out[tr, tc] = color

    return out


def _random_shape(rng: np.random.Generator, size: int) -> list[tuple[int, int]]:
    while True:
        cells = [(0, 0)]
        used = {(0, 0)}
        for _ in range(1, size):
            added = False
            for _ in range(64):
                br, bc = cells[rng.integers(len(cells))]
                for dr, dc in rng.permutation([(1, 0), (-1, 0), (0, 1), (0, -1)]):
                    nr, nc = br + dr, bc + dc
                    if (nr, nc) not in used:
                        cells.append((nr, nc))
                        used.add((nr, nc))
                        added = True
                        break
                if added:
                    break
            if not added:
                break
        if len(cells) == size:
            return cells


def _place_shape(
    rng: np.random.Generator,
    grid: np.ndarray,
    shape_cells: list[tuple[int, int]],
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    color: int,
    divider_color: int,
    occupied: np.ndarray,
) -> list[tuple[int, int]]:
    h, w = grid.shape
    rows = [p[0] for p in shape_cells]
    cols = [p[1] for p in shape_cells]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    for _ in range(300):
        base_r_low = r0 - min_r
        base_r_high = r1 - max_r
        base_c_low = c0 - min_c
        base_c_high = c1 - max_c

        if base_r_low >= base_r_high or base_c_low >= base_c_high:
            return []

        base_r = int(rng.integers(base_r_low, base_r_high))
        base_c = int(rng.integers(base_c_low, base_c_high))

        placed = []
        ok = True
        for dr, dc in shape_cells:
            rr = base_r + dr
            cc = base_c + dc
            if rr < 0 or rr >= h or cc < 0 or cc >= w:
                ok = False
                break
            if occupied[rr, cc] or grid[rr, cc] == divider_color:
                ok = False
                break
            placed.append((rr, cc))

        if not ok:
            continue

        for rr, cc in placed:
            occupied[rr, cc] = True
            grid[rr, cc] = color
        return placed
    return []


def _choose_marker_position(
    rng: np.random.Generator,
    source_side: str,
    orient: str,
    shape_coords: list[tuple[int, int]],
    h: int,
    w: int,
) -> tuple[int, int]:
    if orient == "H":
        if source_side == "top":
            row = 0
            unique_rows = sorted({c for _, c in shape_coords})
            col = int(rng.choice(unique_rows))
            return row, col
        else:
            row = h - 1
            unique_rows = sorted({c for _, c in shape_coords})
            col = int(rng.choice(unique_rows))
            return row, col

    if source_side == "left":
        col = 0
        unique_rows = sorted({r for r, _ in shape_coords})
        row = int(rng.choice(unique_rows))
        return row, col

    col = w - 1
    unique_rows = sorted({r for r, _ in shape_coords})
    row = int(rng.choice(unique_rows))
    return row, col


def _source_region_bounds(h: int, w: int, orient: str, line: int, source_side: str):
    if orient == "H":
        if source_side == "top":
            return (1, line, 1, w - 1)
        return (line + 1, h - 1, 1, w - 1)

    if source_side == "left":
        return (1, h - 1, 1, line)
    return (1, h - 1, line + 1, w - 1)


def generate_input() -> np.ndarray:
    """Generate random ARC-style cases for the two-side projection task."""
    palette = [1, 2, 3, 4, 5, 6, 7, 9]
    bg = 0
    divider_color = 8

    for _ in range(300):
        rng = np.random.default_rng()

        h = int(rng.integers(14, 30))
        w = int(rng.integers(14, 30))

        orient = "H" if rng.random() < 0.5 else "V"
        if orient == "H":
            if h < 12:
                continue
            line = int(rng.integers(4, h - 4))
            source_side = "top" if rng.random() < 0.5 else "bottom"
        else:
            if w < 12:
                continue
            line = int(rng.integers(4, w - 4))
            source_side = "left" if rng.random() < 0.5 else "right"

        grid = np.full((h, w), bg, dtype=np.int64)
        if orient == "H":
            grid[line, :] = divider_color
        else:
            grid[:, line] = divider_color

        total = int(rng.integers(2, 5))
        palette_local = palette.copy()
        rng.shuffle(palette_local)
        if total > len(palette_local):
            continue
        colors = palette_local[:total]

        paired_count = total
        if total > 2 and rng.random() < 0.55:
            paired_count = total - 1

        marker_positions = set()
        occupied = np.zeros((h, w), dtype=bool)
        occupied[grid == divider_color] = True
        occupied[0, :] = True
        occupied[h - 1, :] = True
        occupied[:, 0] = True
        occupied[:, w - 1] = True

        source_r0, source_r1, source_c0, source_c1 = _source_region_bounds(h, w, orient, line, source_side)

        placed_colors = []
        for idx in range(total):
            size = int(rng.integers(2, 5))
            shape = _random_shape(rng, size)
            coords = _place_shape(
                rng, grid, shape, source_r0, source_r1, source_c0, source_c1, colors[idx], divider_color, occupied
            )
            if not coords:
                break
            placed_colors.append((colors[idx], coords))

            if idx < paired_count:
                for _ in range(16):
                    mr, mc = _choose_marker_position(rng, source_side, orient, coords, h, w)
                    if (mr, mc) in marker_positions:
                        continue
                    if grid[mr, mc] != bg:
                        continue
                    grid[mr, mc] = colors[idx]
                    marker_positions.add((mr, mc))
                    break

        # Optional unmatched extra marker branch.
        if len(placed_colors) >= 2 and rng.random() < 0.6:
            unmatched_pool = [c for c in palette_local[total:] if c not in [x[0] for x in placed_colors]]
            if unmatched_pool:
                um_color = int(unmatched_pool[0])
                if orient == "H":
                    edge_cells = [(0, x) for x in range(1, w - 1)] if source_side == "top" else [(h - 1, x) for x in range(1, w - 1)]
                else:
                    edge_cells = [(x, 0) for x in range(1, h - 1)] if source_side == "left" else [(x, w - 1) for x in range(1, h - 1)]
                edge_cells = [p for p in edge_cells if p not in marker_positions and grid[p[0], p[1]] == bg]
                if edge_cells:
                    mr, mc = edge_cells[int(rng.integers(len(edge_cells)))]
                    grid[mr, mc] = um_color
                    marker_positions.add((mr, mc))

        # Keep only examples with enough room for valid source sprites and at least one marker.
        if len(placed_colors) < total:
            continue

        detected = _find_divider(grid)
        if detected is None:
            continue

        if detected["source_side"] != source_side:
            continue

        marker_mask = _boundary_mask(h, w, orient, line, source_side)
        marker_count = int(np.sum((grid[marker_mask] != bg) & (grid[marker_mask] != divider_color)))
        if marker_count == 0:
            continue

        return grid

    # Very rare fallback using minimal valid construction.
    h, w = 14, 14
    line = 7
    orient = "H"
    source_side = "top"
    grid = np.full((h, w), bg, dtype=np.int64)
    grid[line, :] = divider_color
    grid[1, 3] = 1
    grid[2, 3] = 1
    grid[0, 3] = 1
    return grid
