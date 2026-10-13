from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
TEMPLATE_COLOR = 5
LEGEND_SHAPE_COLOR = 8
PALETTE = (1, 2, 3, 4, 6, 7, 9)
GEN_RETRIES = 400
PLACE_RETRIES = 300

SHAPE_LIBRARY: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 1), (0, 2), (1, 0), (1, 1), (2, 0)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
    ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (1, 1), (1, 2), (2, 2)),
    ((0, 1), (1, 0), (1, 1), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (2, 0)),
)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _bbox(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _normalize(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    r0, c0, r1, c1 = _bbox(cells)
    norm = tuple(sorted((r - r0, c - c0) for r, c in cells))
    return norm, r1 - r0 + 1, c1 - c0 + 1


def _coords_to_mask(coords: tuple[tuple[int, int], ...], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


def _scaled_coords(coords: tuple[tuple[int, int], ...], scale: int) -> tuple[tuple[tuple[int, int], ...], int, int]:
    out: list[tuple[int, int]] = []
    for r, c in coords:
        for dr in range(scale):
            for dc in range(scale):
                out.append((r * scale + dr, c * scale + dc))
    return _normalize(tuple(out))


def _matches_scaled(big_cells: list[tuple[int, int]], small_cells: list[tuple[int, int]]) -> bool:
    big, bh, bw = _normalize(big_cells)
    small, sh, sw = _normalize(small_cells)
    if bh % sh != 0 or bw % sw != 0:
        return False
    scale_h = bh // sh
    scale_w = bw // sw
    if scale_h != scale_w or scale_h < 1:
        return False
    scaled, _, _ = _scaled_coords(small, scale_h)
    return big == scaled


def _find_template(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int], list[tuple[int, int]]]:
    candidates: list[tuple[int, int, tuple[int, int, int, int], list[tuple[int, int]]]] = []
    for color in sorted(set(map(int, grid.ravel())) - {BG}):
        for comp in _components(grid == color):
            if len(comp) < 8:
                continue
            r0, c0, r1, c1 = _bbox(comp)
            if r1 - r0 < 4 or c1 - c0 < 4:
                continue
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            if not np.all(sub[0, :] == color) or not np.all(sub[-1, :] == color):
                continue
            if not np.all(sub[:, 0] == color) or not np.all(sub[:, -1] == color):
                continue
            allowed = (sub == color) | (sub == BG)
            if not np.all(allowed):
                continue
            hole_mask = sub == BG
            if not hole_mask.any():
                continue
            if np.any(hole_mask[0, :]) or np.any(hole_mask[-1, :]) or np.any(hole_mask[:, 0]) or np.any(hole_mask[:, -1]):
                continue
            hole_comps = _components(hole_mask)
            if len(hole_comps) != 1:
                continue
            hole_abs = [(r0 + r, c0 + c) for r, c in hole_comps[0]]
            bbox_area = (r1 - r0 + 1) * (c1 - c0 + 1)
            candidates.append((bbox_area, color, (r0, c0, r1, c1), hole_abs))
    assert candidates, "expected a template rectangle with one internal hole"
    candidates.sort(key=lambda x: x[0], reverse=True)
    best = candidates[0]
    if len(candidates) > 1:
        assert best[0] > candidates[1][0], "template must be unique by size"
    return int(best[1]), best[2], best[3]


def _find_legend_entries(
    grid: np.ndarray, template_bbox: tuple[int, int, int, int], template_color: int
) -> tuple[int, list[tuple[list[tuple[int, int]], int]]]:
    r0, c0, r1, c1 = template_bbox
    outside_template = np.ones(grid.shape, dtype=bool)
    outside_template[r0 : r1 + 1, c0 : c1 + 1] = False

    singletons: list[tuple[int, int, int, int]] = []
    outline_votes: list[int] = []
    for color in sorted(set(map(int, grid.ravel())) - {BG, template_color}):
        for comp in _components((grid == color) & outside_template):
            if len(comp) != 1:
                continue
            rr, cc = comp[0]
            neighbor_colors = {
                int(grid[nr, nc])
                for nr, nc in _neighbors4(rr, cc)
                if 0 <= nr < grid.shape[0]
                and 0 <= nc < grid.shape[1]
                and outside_template[nr, nc]
                and int(grid[nr, nc]) not in (BG, template_color, color)
            }
            if len(neighbor_colors) != 1:
                continue
            outline_color = next(iter(neighbor_colors))
            singletons.append((rr, cc, int(color), outline_color))
            outline_votes.append(outline_color)

    assert len(singletons) >= 3, "expected 3-5 legend markers"
    counts = Counter(outline_votes)
    outline_color, freq = counts.most_common(1)[0]
    assert freq >= 3, "legend outline color must dominate singleton adjacency"

    marker_lookup: dict[tuple[int, int], int] = {(r, c): color for r, c, color, oc in singletons if oc == outline_color}
    entries: list[tuple[list[tuple[int, int]], int]] = []
    used_markers: set[tuple[int, int]] = set()
    for comp in _components((grid == outline_color) & outside_template):
        touching = {
            (nr, nc)
            for r, c in comp
            for nr, nc in _neighbors4(r, c)
            if (nr, nc) in marker_lookup
        }
        if not touching:
            continue
        assert len(touching) == 1, "each legend shape must touch exactly one marker"
        marker_cell = next(iter(touching))
        used_markers.add(marker_cell)
        entries.append((comp, marker_lookup[marker_cell]))

    assert 3 <= len(entries) <= 5, "legend must contain 3-5 entries"
    assert len(used_markers) == len(entries), "legend markers and shapes must pair one-to-one"
    return int(outline_color), entries


def _find_piece_color(
    grid: np.ndarray,
    template_bbox: tuple[int, int, int, int],
    template_color: int,
    outline_color: int,
    marker_colors: set[int],
    hole_area: int,
) -> int:
    r0, c0, r1, c1 = template_bbox
    outside_template = np.ones(grid.shape, dtype=bool)
    outside_template[r0 : r1 + 1, c0 : c1 + 1] = False
    candidates: list[int] = []
    for color in sorted(set(map(int, grid.ravel())) - {BG, template_color, outline_color} - marker_colors):
        comps = _components((grid == color) & outside_template)
        if not comps:
            continue
        if any(len(comp) == 1 for comp in comps):
            continue
        total = sum(len(comp) for comp in comps)
        if total == hole_area:
            candidates.append(int(color))
    assert len(candidates) == 1, "piece color must be uniquely determined by area"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered monochrome fragments into the template hole, then recolor the template
    according to the matching legend silhouette.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    template_color, template_bbox, hole_cells = _find_template(g)
    outline_color, legend_entries = _find_legend_entries(g, template_bbox, template_color)
    marker_colors = {color for _, color in legend_entries}
    piece_color = _find_piece_color(
        g,
        template_bbox,
        template_color,
        outline_color,
        marker_colors,
        hole_area=len(hole_cells),
    )

    matches = [entry for entry in legend_entries if _matches_scaled(hole_cells, entry[0])]
    assert len(matches) == 1, "hole silhouette must match exactly one legend entry"
    target_color = matches[0][1]

    r0, c0, r1, c1 = template_bbox
    out = np.full_like(g, BG)
    out[r0 : r1 + 1, c0 : c1 + 1] = target_color
    for r, c in hole_cells:
        out[r, c] = piece_color
    return out


def _partition_connected(
    coords: tuple[tuple[int, int], ...], groups_n: int, rng: random.Random
) -> list[tuple[tuple[int, int], ...]] | None:
    cells = list(coords)
    seeds = rng.sample(cells, groups_n)
    groups = [{seed} for seed in seeds]
    unassigned = set(cells) - set(seeds)

    while unassigned:
        frontier: list[tuple[tuple[int, int], int]] = []
        for idx, group in enumerate(groups):
            for cell in sorted(unassigned):
                if any(abs(cell[0] - rr) + abs(cell[1] - cc) == 1 for rr, cc in group):
                    frontier.append((cell, idx))
        if not frontier:
            return None
        cell, idx = rng.choice(frontier)
        groups[idx].add(cell)
        unassigned.remove(cell)

    return [tuple(sorted(group)) for group in groups]


def _place_patch(grid: np.ndarray, patch: np.ndarray, color: int, rng: random.Random) -> bool:
    ph, pw = patch.shape
    cells = np.argwhere(patch)
    for _ in range(PLACE_RETRIES):
        r0 = rng.randint(1, grid.shape[0] - ph - 1)
        c0 = rng.randint(1, grid.shape[1] - pw - 1)
        rr0 = r0 - 1
        cc0 = c0 - 1
        rr1 = r0 + ph
        cc1 = c0 + pw
        if np.any(grid[rr0 : rr1 + 1, cc0 : cc1 + 1] != BG):
            continue
        for dr, dc in cells:
            grid[r0 + dr, c0 + dc] = color
        return True
    return False


def _make_piece_patches(
    proto: tuple[tuple[int, int], ...], scale: int, rng: random.Random
) -> list[np.ndarray] | None:
    proto_norm, _, _ = _normalize(proto)
    max_groups = min(5, len(proto_norm))
    groups_n = rng.randint(2, max_groups)
    for _ in range(80):
        groups = _partition_connected(proto_norm, groups_n, rng)
        if groups is None:
            continue
        pieces: list[np.ndarray] = []
        for group in groups:
            scaled_group, gh, gw = _scaled_coords(group, scale)
            patch = np.zeros((gh, gw), dtype=bool)
            for r, c in scaled_group:
                patch[r, c] = True
            pieces.append(patch)
        return pieces
    return None


def _right_marker_patch(proto: tuple[tuple[int, int], ...], marker_color: int) -> np.ndarray:
    coords, h, w = _normalize(proto)
    patch = np.zeros((h, w + 1), dtype=int)
    for r, c in coords:
        patch[r, c] = LEGEND_SHAPE_COLOR
    right_col = max(c for _, c in coords)
    candidate_rows = sorted(r for r, c in coords if c == right_col)
    patch[candidate_rows[len(candidate_rows) // 2], w] = marker_color
    return patch


def _bottom_marker_patch(proto: tuple[tuple[int, int], ...], marker_color: int) -> np.ndarray:
    coords, h, w = _normalize(proto)
    patch = np.zeros((h + 1, w), dtype=int)
    for r, c in coords:
        patch[r, c] = LEGEND_SHAPE_COLOR
    bottom_row = max(r for r, _ in coords)
    candidate_cols = sorted(c for r, c in coords if r == bottom_row)
    patch[h, candidate_cols[len(candidate_cols) // 2]] = marker_color
    return patch


def _draw_legend(
    grid: np.ndarray,
    entries: list[tuple[tuple[tuple[int, int], ...], int]],
    side: str,
    rng: random.Random,
) -> None:
    gap = 2
    if side == "left":
        patches = [_right_marker_patch(proto, color) for proto, color in entries]
        total_h = sum(p.shape[0] for p in patches) + gap * (len(patches) - 1)
        assert total_h <= grid.shape[0]
        top = rng.randint(0, grid.shape[0] - total_h)
        row = top
        for patch in patches:
            ph, pw = patch.shape
            grid[row : row + ph, 0:pw] = np.maximum(grid[row : row + ph, 0:pw], patch)
            row += ph + gap
        return

    patches = [_bottom_marker_patch(proto, color) for proto, color in entries]
    total_w = sum(p.shape[1] for p in patches) + gap * (len(patches) - 1)
    assert total_w <= grid.shape[1]
    left = rng.randint(0, grid.shape[1] - total_w)
    col = left
    for patch in patches:
        ph, pw = patch.shape
        grid[0:ph, col : col + pw] = np.maximum(grid[0:ph, col : col + pw], patch)
        col += pw + gap


def _build_expected(
    shape_cells: tuple[tuple[int, int], ...],
    template_bbox: tuple[int, int, int, int],
    hole_offset: tuple[int, int],
    piece_color: int,
    target_color: int,
) -> np.ndarray:
    r0, c0, r1, c1 = template_bbox
    h = r1 + 1
    w = c1 + 1
    out = np.full((h, w), BG, dtype=int)
    out[r0 : r1 + 1, c0 : c1 + 1] = target_color
    hr, hc = hole_offset
    for r, c in shape_cells:
        out[r0 + hr + r, c0 + hc + c] = piece_color
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    side = rng.choice(("left", "top"))
    n_entries = rng.randint(3, 5)
    chosen_shapes = [tuple(shape) for shape in rng.sample(SHAPE_LIBRARY, n_entries)]
    match_idx = rng.randrange(n_entries)
    match_shape = chosen_shapes[match_idx]
    scale = rng.choice((2, 3))

    colors = rng.sample(PALETTE, n_entries + 1)
    piece_color = colors[0]
    marker_colors = colors[1:]
    entries = list(zip(chosen_shapes, marker_colors))

    hole_shape, hole_h, hole_w = _scaled_coords(match_shape, scale)
    margin_top = rng.randint(2, 4)
    margin_bottom = rng.randint(2, 4)
    margin_left = rng.randint(2, 4)
    margin_right = rng.randint(2, 4)
    template_h = hole_h + margin_top + margin_bottom
    template_w = hole_w + margin_left + margin_right

    if side == "left":
        legend_patches = [_right_marker_patch(proto, col) for proto, col in entries]
        legend_w = max(p.shape[1] for p in legend_patches)
        legend_h = sum(p.shape[0] for p in legend_patches) + 2 * (len(legend_patches) - 1)
        min_h = max(20, template_h + 4, legend_h + 1)
        min_w = max(20, template_w + legend_w + 4)
    else:
        legend_patches = [_bottom_marker_patch(proto, col) for proto, col in entries]
        legend_h = max(p.shape[0] for p in legend_patches)
        legend_w = sum(p.shape[1] for p in legend_patches) + 2 * (len(legend_patches) - 1)
        min_h = max(20, template_h + legend_h + 4)
        min_w = max(20, template_w + 4, legend_w + 1)

    height = rng.randint(min_h, min_h + 6)
    width = rng.randint(min_w, min_w + 6)
    if height > 30 or width > 30:
        return None

    grid = np.full((height, width), BG, dtype=int)
    if side == "left":
        template_r0 = rng.randint(1, height - template_h - 1)
        template_c0 = rng.randint(legend_w + 2, width - template_w - 1)
    else:
        template_r0 = rng.randint(legend_h + 2, height - template_h - 1)
        template_c0 = rng.randint(1, width - template_w - 1)
    template_bbox = (
        template_r0,
        template_c0,
        template_r0 + template_h - 1,
        template_c0 + template_w - 1,
    )

    grid[template_bbox[0] : template_bbox[2] + 1, template_bbox[1] : template_bbox[3] + 1] = TEMPLATE_COLOR
    hole_offset = (margin_top, margin_left)
    for r, c in hole_shape:
        grid[template_r0 + margin_top + r, template_c0 + margin_left + c] = BG

    _draw_legend(grid, entries, side, rng)

    piece_patches = _make_piece_patches(match_shape, scale, rng)
    if piece_patches is None:
        return None
    piece_patches.sort(key=lambda p: int(np.count_nonzero(p)), reverse=True)
    for patch in piece_patches:
        if not _place_patch(grid, patch, piece_color, rng):
            return None

    if np.count_nonzero(grid == piece_color) != len(hole_shape):
        return None

    expected = np.full_like(grid, BG)
    expected[template_bbox[0] : template_bbox[2] + 1, template_bbox[1] : template_bbox[3] + 1] = marker_colors[match_idx]
    for r, c in hole_shape:
        expected[template_r0 + margin_top + r, template_c0 + margin_left + c] = piece_color

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
