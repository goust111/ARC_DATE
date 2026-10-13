from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 400
PLACEMENT_RETRIES = 250
TEMPLATE_SIZE = 3
LIBRARY_SHAPES = ((2, 2), (2, 3), (3, 2))
PATTERN_COORDS = (
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (0, 2), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (0, 1), (1, 1), (2, 1)),
    ((0, 0), (1, 0), (2, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 1), (2, 2)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
)
LIBRARY_CORNERS = ("tl", "tr", "bl", "br")


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_grid_corner(box: tuple[int, int, int, int], h: int, w: int) -> bool:
    r0, c0, r1, c1 = box
    return (
        (r0 == 0 and c0 == 0)
        or (r0 == 0 and c1 == w - 1)
        or (r1 == h - 1 and c0 == 0)
        or (r1 == h - 1 and c1 == w - 1)
    )


def _frame_color(grid: np.ndarray, bg: int) -> int:
    counts = Counter(map(int, grid.ravel()))
    for color, _count in counts.most_common():
        if int(color) != int(bg):
            return int(color)
    raise AssertionError("expected a non-background frame color")


def _find_library_bbox(grid: np.ndarray, grey: int) -> tuple[int, int, int, int]:
    h, w = grid.shape
    grey_mask = grid == int(grey)
    candidates: list[tuple[int, int, int, int, int]] = []
    for pts in _components(grey_mask):
        box = _bbox(pts)
        if not _touches_grid_corner(box, h, w):
            continue
        r0, c0, r1, c1 = box
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        full_rows = [r for r in range(sub.shape[0]) if bool(np.all(sub[r, :] == int(grey)))]
        full_cols = [c for c in range(sub.shape[1]) if bool(np.all(sub[:, c] == int(grey)))]
        if len(full_rows) < 3 or len(full_cols) < 3:
            continue
        if full_rows[0] != 0 or full_rows[-1] != sub.shape[0] - 1:
            continue
        if full_cols[0] != 0 or full_cols[-1] != sub.shape[1] - 1:
            continue
        area = int(sub.shape[0] * sub.shape[1])
        candidates.append((-area, r0, c0, r1, c1))
    assert candidates, "expected one corner template library"
    _area, r0, c0, r1, c1 = min(candidates)
    return int(r0), int(c0), int(r1), int(c1)


def _parse_library(
    grid: np.ndarray,
    library_bbox: tuple[int, int, int, int],
    bg: int,
    grey: int,
) -> tuple[dict[int, np.ndarray], dict[int, tuple[int, int]]]:
    r0, c0, r1, c1 = library_bbox
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    sep_rows = [r for r in range(sub.shape[0]) if bool(np.all(sub[r, :] == int(grey)))]
    sep_cols = [c for c in range(sub.shape[1]) if bool(np.all(sub[:, c] == int(grey)))]
    row_steps = [b - a for a, b in zip(sep_rows, sep_rows[1:])]
    col_steps = [b - a for a, b in zip(sep_cols, sep_cols[1:])]
    assert row_steps and col_steps
    assert len(set(row_steps)) == 1 and len(set(col_steps)) == 1
    cell_h = int(row_steps[0] - 1)
    cell_w = int(col_steps[0] - 1)
    assert cell_h >= 2 and cell_w >= 2

    templates: dict[int, np.ndarray] = {}
    order: dict[int, tuple[int, int]] = {}
    for ir in range(len(sep_rows) - 1):
        rr = slice(int(sep_rows[ir] + 1), int(sep_rows[ir + 1]))
        for ic in range(len(sep_cols) - 1):
            cc = slice(int(sep_cols[ic] + 1), int(sep_cols[ic + 1]))
            cell = sub[rr, cc]
            colors = [int(color) for color in np.unique(cell) if int(color) not in (int(bg), int(grey))]
            assert len(colors) == 1, "every template cell must contain exactly one key color"
            color = int(colors[0])
            templates[color] = cell == int(color)
            order[color] = (int(ir), int(ic))
    assert templates
    return templates, order


def _is_room_component(
    grey_mask: np.ndarray,
    points: list[tuple[int, int]],
) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(points)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 6 or w < 6:
        return None
    sub = grey_mask[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros_like(sub, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not bool(np.all(sub[border])):
        return None
    if bool(np.any(sub[~border])):
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _stamp(out: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    h, w = mask.shape
    block = out[int(top) : int(top + h), int(left) : int(left + w)]
    block[mask] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Choose a corner-stamp template by majority vote inside each framed region, then print that template onto the region's four corners.

    Input structure:
    - The background is the most common color.
    - One grid corner contains a template library enclosed by one frame color and split by full separator lines into equal cells.
    - Each library cell contains a monochrome template drawn in a unique key color; that color identifies the template.
    - Elsewhere, several closed rectangular regions use the same frame color as a 1-pixel border and contain scattered single-pixel votes in library key colors.

    Rule:
    - Parse the template library to recover the key-color to template mapping and the visible library order: upper rows beat lower rows, and within a row left cells beat right cells.
    - For each framed region, count interior vote colors, choose the highest count, and break ties by that library order.
    - Clear the region's vote pixels to background, then place the chosen template on the four corners of the frame: original at top-left, horizontally flipped at top-right, vertically flipped at bottom-left, and rotated 180 degrees at bottom-right.

    Key insight:
    - The colored interior pixels are votes, not paint targets; they only select a library template, and the selected shape is then copied to the border corners with four-way symmetry.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    grey = _frame_color(g, bg)
    library_bbox = _find_library_bbox(g, grey)
    templates, order = _parse_library(g, library_bbox, bg, grey)
    template_colors = np.array(sorted(templates), dtype=int)
    grey_mask = g == int(grey)

    out = g.copy()
    for pts in _components(grey_mask):
        room_bbox = _is_room_component(grey_mask, pts)
        if room_bbox is None or room_bbox == library_bbox:
            continue
        r0, c0, r1, c1 = room_bbox
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        counts = Counter(int(value) for value in inner.ravel() if int(value) in order)
        if not counts:
            continue
        best = max(counts.values())
        winner = min((color for color, count in counts.items() if int(count) == int(best)), key=lambda color: order[int(color)])

        inner_out = out[r0 + 1 : r1, c0 + 1 : c1]
        inner_out[np.isin(inner_out, template_colors)] = int(bg)

        mask = templates[int(winner)]
        h, w = mask.shape
        _stamp(out, r0, c0, mask, winner)
        _stamp(out, r0, c1 - w + 1, np.fliplr(mask), winner)
        _stamp(out, r1 - h + 1, c0, np.flipud(mask), winner)
        _stamp(out, r1 - h + 1, c1 - w + 1, np.flipud(np.fliplr(mask)), winner)

    return out


def _pattern_from_coords(coords: tuple[tuple[int, int], ...], size: int = TEMPLATE_SIZE) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    return mask


def _library_origin(corner: str, grid_h: int, grid_w: int, lib_h: int, lib_w: int) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, int(grid_w - lib_w)
    if corner == "bl":
        return int(grid_h - lib_h), 0
    if corner == "br":
        return int(grid_h - lib_h), int(grid_w - lib_w)
    raise AssertionError(f"unknown corner {corner}")


def _build_library(
    rows: int,
    cols: int,
    cell_size: int,
    bg: int,
    grey: int,
    specs: list[tuple[int, np.ndarray]],
) -> tuple[np.ndarray, dict[int, tuple[int, int]]]:
    lib_h = int(rows * cell_size + rows + 1)
    lib_w = int(cols * cell_size + cols + 1)
    library = np.full((lib_h, lib_w), int(bg), dtype=int)
    for r in range(rows + 1):
        library[int(r * (cell_size + 1)), :] = int(grey)
    for c in range(cols + 1):
        library[:, int(c * (cell_size + 1))] = int(grey)

    order: dict[int, tuple[int, int]] = {}
    for idx, (color, mask) in enumerate(specs):
        rr = idx // cols
        cc = idx % cols
        top = int(rr * (cell_size + 1) + 1)
        left = int(cc * (cell_size + 1) + 1)
        block = library[top : top + cell_size, left : left + cell_size]
        block[mask] = int(color)
        order[int(color)] = (int(rr), int(cc))
    return library, order


def _can_place(occupied: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> bool:
    h, w = occupied.shape
    r0 = max(0, int(top - pad))
    c0 = max(0, int(left - pad))
    r1 = min(h, int(top + height + pad))
    c1 = min(w, int(left + width + pad))
    return not bool(np.any(occupied[r0:r1, c0:c1]))


def _mark_occupied(occupied: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> None:
    h, w = occupied.shape
    r0 = max(0, int(top - pad))
    c0 = max(0, int(left - pad))
    r1 = min(h, int(top + height + pad))
    c1 = min(w, int(left + width + pad))
    occupied[r0:r1, c0:c1] = True


def _sorted_colors(order: dict[int, tuple[int, int]]) -> list[int]:
    return sorted(order, key=lambda color: order[int(color)])


def _vote_counts(
    order: dict[int, tuple[int, int]],
    want_tie: bool,
) -> dict[int, int]:
    ordered = _sorted_colors(order)
    by_row: dict[int, list[int]] = {}
    for color in ordered:
        row, _col = order[int(color)]
        by_row.setdefault(int(row), []).append(int(color))

    if want_tie:
        modes: list[str] = []
        if any(len(colors) >= 2 for colors in by_row.values()):
            modes.append("same_row")
        if len(by_row) >= 2:
            modes.append("different_rows")
        if len(ordered) >= 3:
            modes.append("triple")
        mode = random.choice(modes)
        top = random.randint(2, 3)

        if mode == "same_row":
            row = random.choice([r for r, colors in by_row.items() if len(colors) >= 2])
            tied = random.sample(by_row[int(row)], 2)
            counts = {int(color): int(top) for color in tied}
        elif mode == "different_rows":
            upper = random.choice(ordered[:-1])
            lower_choices = [color for color in ordered if order[int(color)][0] > order[int(upper)][0]]
            if not lower_choices:
                lower_choices = [color for color in ordered if int(color) != int(upper)]
            lower = random.choice(lower_choices)
            counts = {int(upper): int(top), int(lower): int(top)}
        else:
            tied = random.sample(ordered, 3)
            top = random.randint(1, 2)
            counts = {int(color): int(top) for color in tied}

        losers = [color for color in ordered if int(color) not in counts]
        if losers and random.random() < 0.7:
            counts[int(random.choice(losers))] = random.randint(1, max(1, top - 1))
        return counts

    winner = random.choice(ordered)
    loser_count = random.randint(1, min(2, len(ordered) - 1))
    losers = random.sample([color for color in ordered if int(color) != int(winner)], loser_count)
    win_ct = random.randint(3, 5)
    counts = {int(winner): int(win_ct)}
    remaining = 10 - win_ct
    for idx, loser in enumerate(losers):
        reserve = len(losers) - idx - 1
        max_ct = min(win_ct - 1, remaining - reserve)
        ct = random.randint(1, int(max_ct))
        counts[int(loser)] = int(ct)
        remaining -= int(ct)
    return counts


def _checkerboard_cells(inner_h: int, inner_w: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w) if (r + c) % 2 == 0]
    if len(cells) < 10:
        cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    return cells


def _draw_room(
    grid: np.ndarray,
    top: int,
    left: int,
    inner_h: int,
    inner_w: int,
    grey: int,
    vote_counts: dict[int, int],
) -> None:
    bottom = int(top + inner_h + 1)
    right = int(left + inner_w + 1)
    grid[int(top), int(left) : int(right + 1)] = int(grey)
    grid[int(bottom), int(left) : int(right + 1)] = int(grey)
    grid[int(top) : int(bottom + 1), int(left)] = int(grey)
    grid[int(top) : int(bottom + 1), int(right)] = int(grey)

    cells = _checkerboard_cells(inner_h, inner_w)
    total_votes = int(sum(vote_counts.values()))
    assert total_votes <= len(cells)
    chosen = random.sample(cells, total_votes)
    random.shuffle(chosen)
    idx = 0
    for color, count in vote_counts.items():
        for _ in range(int(count)):
            rr, cc = chosen[idx]
            grid[int(top + 1 + rr), int(left + 1 + cc)] = int(color)
            idx += 1


def _strong_frame_color(grid: np.ndarray, bg: int, grey: int) -> bool:
    counts = Counter(map(int, grid.ravel()))
    other = [count for color, count in counts.items() if int(color) not in (int(bg), int(grey))]
    return not other or int(counts[int(grey)]) > max(other)


def _generate_candidate() -> np.ndarray | None:
    cell_size = TEMPLATE_SIZE
    lib_rows, lib_cols = random.choice(LIBRARY_SHAPES)
    template_count = int(lib_rows * lib_cols)
    palette = random.sample(range(10), template_count + 2)
    bg = int(palette[0])
    grey = int(palette[1])
    template_colors = [int(color) for color in palette[2:]]

    pattern_indices = random.sample(range(len(PATTERN_COORDS)), template_count)
    masks = [_pattern_from_coords(PATTERN_COORDS[idx], cell_size) for idx in pattern_indices]
    specs = list(zip(template_colors, masks))
    library, order = _build_library(lib_rows, lib_cols, cell_size, bg, grey, specs)
    lib_h, lib_w = library.shape

    max_rooms = 4 if max(lib_h, lib_w) <= 10 else 3
    room_count = random.randint(2, max_rooms)
    min_dim = 18 + (room_count - 2) * 3
    grid_h = random.randint(max(min_dim, lib_h + 4), 30)
    grid_w = random.randint(max(min_dim, lib_w + 4), 30)
    corner = random.choice(LIBRARY_CORNERS)
    lib_top, lib_left = _library_origin(corner, grid_h, grid_w, lib_h, lib_w)

    grid = np.full((grid_h, grid_w), int(bg), dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    grid[int(lib_top) : int(lib_top + lib_h), int(lib_left) : int(lib_left + lib_w)] = library
    _mark_occupied(occupied, lib_top, lib_left, lib_h, lib_w, pad=1)

    tie_flags = [bool(random.getrandbits(1)) for _ in range(room_count)]
    if room_count >= 2:
        if not any(tie_flags):
            tie_flags[random.randrange(room_count)] = True
        if all(tie_flags):
            tie_flags[random.randrange(room_count)] = False

    for want_tie in tie_flags:
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            inner_h = random.randint(5, 8)
            inner_w = random.randint(5, 8)
            outer_h = int(inner_h + 2)
            outer_w = int(inner_w + 2)
            if outer_h >= grid_h - 1 or outer_w >= grid_w - 1:
                continue
            top = random.randint(1, grid_h - outer_h - 1)
            left = random.randint(1, grid_w - outer_w - 1)
            if not _can_place(occupied, top, left, outer_h, outer_w, pad=1):
                continue
            votes = _vote_counts(order, want_tie)
            if sum(votes.values()) > 10:
                continue
            _draw_room(grid, top, left, inner_h, inner_w, grey, votes)
            _mark_occupied(occupied, top, left, outer_h, outer_w, pad=1)
            placed = True
            break
        if not placed:
            return None

    if not _strong_frame_color(grid, bg, grey):
        return None

    out = main(grid)
    if not isinstance(out, np.ndarray) or out.shape != grid.shape:
        return None
    if not np.any(out != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        candidate = _generate_candidate()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid voting-template puzzle")
