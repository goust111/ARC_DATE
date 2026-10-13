from __future__ import annotations

import random

import numpy as np


NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# --------------- solver helpers ---------------


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((int(cr), int(cc)))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1])


def _find_templates(grid: np.ndarray, bg: int) -> tuple[list[dict[str, object]], np.ndarray]:
    h, w = map(int, grid.shape)
    template_pixels = np.zeros((h, w), dtype=bool)
    templates: list[dict[str, object]] = []
    for color in [int(c) for c in np.unique(grid) if int(c) != int(bg)]:
        for coords in _components(grid == int(color)):
            r0, c0, r1, c1 = _bbox(coords)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            border = np.zeros(sub.shape, dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.all(sub[border] == int(color)):
                continue
            if np.any((sub != int(color)) & (sub != int(bg))):
                continue

            comp_local = np.zeros(sub.shape, dtype=bool)
            comp_local[coords[:, 0] - r0, coords[:, 1] - c0] = True
            if not np.array_equal(comp_local, sub == int(color)):
                continue

            hole = sub == int(bg)
            if not hole[1:-1, 1:-1].any():
                continue
            if len(_components(hole)) != 1:
                continue

            hole_mask = np.zeros_like(grid, dtype=bool)
            hole_mask[r0 : r1 + 1, c0 : c1 + 1] = hole
            template_pixels[coords[:, 0], coords[:, 1]] = True
            templates.append(
                {
                    "color": int(color),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "hole_mask": hole_mask,
                    "area": int(hole.sum()),
                }
            )
    return templates, template_pixels


def _find_fragment_groups(grid: np.ndarray, bg: int, template_pixels: np.ndarray) -> tuple[list[dict[str, int]], np.ndarray]:
    fragment_mask = (grid != int(bg)) & (~template_pixels)
    groups: list[dict[str, int]] = []
    for color in [int(c) for c in np.unique(grid[fragment_mask])]:
        mask = fragment_mask & (grid == int(color))
        groups.append({"color": int(color), "area": int(mask.sum())})
    return groups, fragment_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each scattered fragment group into the template hole whose area matches it.

    Input structure:
    - The grid has one dominant background color.
    - Several large monochrome rectangles act as templates; each template encloses one connected background-colored hole.
    - Outside the templates, each fragment group consists of several disconnected pieces of one shared color, and fragment colors differ from template colors.

    Rule:
    - Detect every template and measure the area of its internal hole.
    - For each fragment color, sum the pixels across all of its scattered pieces.
    - Pair fragment groups to template holes by equal total area, erase the original fragments, and fill the matched hole with that fragment color.

    Key insight:
    The visible output depends only on area-based pairing: because every fragment group is monochrome, any valid packing fills its matched hole as one solid same-colored region.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    templates, template_pixels = _find_templates(g, bg)
    fragments, fragment_mask = _find_fragment_groups(g, bg, template_pixels)

    area_to_template = {int(t["area"]): t for t in templates}
    area_to_fragment = {int(f["area"]): f for f in fragments}
    assert len(area_to_template) == len(templates)
    assert len(area_to_fragment) == len(fragments)
    assert set(area_to_template) == set(area_to_fragment)

    out = g.copy()
    out[fragment_mask] = int(bg)
    for area, template in area_to_template.items():
        color = int(area_to_fragment[int(area)]["color"])
        hole_mask = np.asarray(template["hole_mask"], dtype=bool)
        out[hole_mask] = int(color)
    return out


# --------------- generator helpers ---------------


GENERATION_ATTEMPTS = 400
SHAPE_ATTEMPTS = 120
LAYOUT_ATTEMPTS = 120
GRID_SIDE_RANGE = (20, 30)
TEMPLATE_COUNT_OPTIONS = (2, 3)
HOLE_H_RANGE = (3, 5)
HOLE_W_RANGE = (3, 6)
HOLE_AREA_RANGE = (5, 12)
TEMPLATE_MARGIN_OPTIONS = (1, 2)
PIECE_COUNT_OPTIONS = (2, 3, 4)
PLACE_GAP = 1


def _grow_connected_mask(height: int, width: int, target_area: int) -> np.ndarray | None:
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = True
    cells = {start}
    frontier: set[tuple[int, int]] = set()
    for dr, dc in NEIGHBORS4:
        nr, nc = start[0] + dr, start[1] + dc
        if 0 <= nr < height and 0 <= nc < width:
            frontier.add((int(nr), int(nc)))
    while len(cells) < int(target_area):
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        if cell in cells:
            continue
        cells.add(cell)
        mask[cell] = True
        cr, cc = cell
        for dr, dc in NEIGHBORS4:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                frontier.add((int(nr), int(nc)))
    return mask


def _sample_hole_mask() -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        height = random.randint(*HOLE_H_RANGE)
        width = random.randint(*HOLE_W_RANGE)
        lo = max(HOLE_AREA_RANGE[0], max(height, width) + 1)
        hi = min(HOLE_AREA_RANGE[1], height * width - 1)
        if lo > hi:
            continue
        target_area = random.randint(lo, hi)
        mask = _grow_connected_mask(height, width, target_area)
        if mask is None:
            continue
        rows, cols = np.where(mask)
        spans_box = rows.min() == 0 and rows.max() == height - 1 and cols.min() == 0 and cols.max() == width - 1
        if spans_box and int(mask.sum()) < height * width and not _has_interior_void(mask):
            return mask
    return None


def _has_interior_void(mask: np.ndarray) -> bool:
    bg = ~mask
    height, width = map(int, mask.shape)
    for comp in _components(bg):
        rs = comp[:, 0]
        cs = comp[:, 1]
        touches_edge = (
            int(rs.min()) == 0
            or int(rs.max()) == height - 1
            or int(cs.min()) == 0
            or int(cs.max()) == width - 1
        )
        if not touches_edge:
            return True
    return False


def _tree_partition(mask: np.ndarray, n_parts: int) -> list[np.ndarray] | None:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    index = {coord: i for i, coord in enumerate(coords)}
    adjacency: list[list[int]] = [[] for _ in coords]
    for i, (r, c) in enumerate(coords):
        for dr, dc in NEIGHBORS4:
            nbr = (r + dr, c + dc)
            if nbr in index:
                adjacency[i].append(index[nbr])

    visited = {0}
    stack = [0]
    tree_edges: list[tuple[int, int]] = []
    while stack:
        node = stack[-1]
        choices = [nbr for nbr in adjacency[node] if nbr not in visited]
        if not choices:
            stack.pop()
            continue
        nxt = random.choice(choices)
        visited.add(nxt)
        tree_edges.append((int(node), int(nxt)))
        stack.append(int(nxt))
    if len(visited) != len(coords):
        return None

    min_piece_area = 2 if len(coords) >= 2 * n_parts else 1
    for _ in range(SHAPE_ATTEMPTS):
        cut_ids = set(random.sample(range(len(tree_edges)), n_parts - 1))
        forest: list[list[int]] = [[] for _ in coords]
        for edge_id, (u, v) in enumerate(tree_edges):
            if edge_id in cut_ids:
                continue
            forest[u].append(v)
            forest[v].append(u)

        comp_ids = [-1] * len(coords)
        parts: list[list[int]] = []
        for start in range(len(coords)):
            if comp_ids[start] != -1:
                continue
            q = [start]
            comp_ids[start] = len(parts)
            members = []
            while q:
                node = q.pop()
                members.append(int(node))
                for nbr in forest[node]:
                    if comp_ids[nbr] == -1:
                        comp_ids[nbr] = comp_ids[start]
                        q.append(int(nbr))
            parts.append(members)
        if len(parts) != n_parts or min(len(part) for part in parts) < min_piece_area:
            continue

        piece_masks: list[np.ndarray] = []
        for part in parts:
            piece = np.zeros_like(mask, dtype=bool)
            for node in part:
                r, c = coords[node]
                piece[r, c] = True
            piece_masks.append(piece)
        return piece_masks
    return None


def _mask_to_patch(mask: np.ndarray, fg: int, bg: int) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = int(fg)
    return patch


def _looks_like_template(patch: np.ndarray, fg: int, bg: int) -> bool:
    if patch.shape[0] < 3 or patch.shape[1] < 3:
        return False
    border = np.zeros(patch.shape, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.all(patch[border] == int(fg)):
        return False
    return bool(np.any(patch[1:-1, 1:-1] == int(bg)))


def _build_template_patch(hole_mask: np.ndarray, template_color: int, bg: int) -> tuple[np.ndarray, np.ndarray]:
    hole_h, hole_w = map(int, hole_mask.shape)
    top = random.choice(TEMPLATE_MARGIN_OPTIONS)
    bottom = random.choice(TEMPLATE_MARGIN_OPTIONS)
    left = random.choice(TEMPLATE_MARGIN_OPTIONS)
    right = random.choice(TEMPLATE_MARGIN_OPTIONS)
    patch = np.full((hole_h + top + bottom, hole_w + left + right), int(template_color), dtype=int)
    hole_local = np.zeros_like(patch, dtype=bool)
    hole_local[top : top + hole_h, left : left + hole_w] = hole_mask
    patch[hole_local] = int(bg)
    return patch, hole_local


def _find_slot(occupied: np.ndarray, shape: tuple[int, int], gap: int) -> tuple[int, int] | None:
    h, w = map(int, occupied.shape)
    ph, pw = map(int, shape)
    if ph > h or pw > w:
        return None
    positions = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
    random.shuffle(positions)
    for r, c in positions:
        r0 = max(0, r - gap)
        c0 = max(0, c - gap)
        r1 = min(h, r + ph + gap)
        c1 = min(w, c + pw + gap)
        if not occupied[r0:r1, c0:c1].any():
            return int(r), int(c)
    return None


def _place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    window = grid[top : top + patch.shape[0], left : left + patch.shape[1]]
    mask = patch != int(bg)
    window[mask] = patch[mask]
    grid[top : top + patch.shape[0], left : left + patch.shape[1]] = window


def _sample_template_specs(bg: int, template_colors: list[int], fragment_colors: list[int]) -> list[dict[str, object]] | None:
    specs: list[dict[str, object]] = []
    used_areas: set[int] = set()
    for template_color, fragment_color in zip(template_colors, fragment_colors, strict=True):
        built = None
        for _ in range(SHAPE_ATTEMPTS):
            hole_mask = _sample_hole_mask()
            if hole_mask is None:
                continue
            area = int(hole_mask.sum())
            if area in used_areas:
                continue

            piece_candidates = [k for k in PIECE_COUNT_OPTIONS if int(area) >= 2 * int(k)]
            n_parts = random.choice(piece_candidates or [2])
            pieces = _tree_partition(hole_mask, n_parts)
            if pieces is None:
                continue

            piece_patches = [_mask_to_patch(piece, int(fragment_color), int(bg)) for piece in pieces]
            if any(_looks_like_template(patch, int(fragment_color), int(bg)) for patch in piece_patches):
                continue

            template_patch, hole_local = _build_template_patch(hole_mask, int(template_color), int(bg))
            built = {
                "area": int(area),
                "fragment_color": int(fragment_color),
                "template_patch": template_patch,
                "hole_local": hole_local,
                "piece_patches": piece_patches,
            }
            used_areas.add(area)
            break
        if built is None:
            return None
        specs.append(built)
    return specs


def _compose_scene(specs: list[dict[str, object]], bg: int) -> tuple[np.ndarray, np.ndarray] | None:
    grid_h = random.randint(*GRID_SIDE_RANGE)
    grid_w = random.randint(*GRID_SIDE_RANGE)
    grid = np.full((grid_h, grid_w), int(bg), dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    expected = np.full((grid_h, grid_w), int(bg), dtype=int)

    template_order = sorted(specs, key=lambda spec: -int(np.asarray(spec["template_patch"]).size))
    placed_specs: list[dict[str, object]] = []

    for spec in template_order:
        template_patch = np.asarray(spec["template_patch"], dtype=int)
        slot = _find_slot(occupied, template_patch.shape, PLACE_GAP)
        if slot is None:
            return None
        top, left = slot
        _place_patch(grid, template_patch, top, left, bg)
        occupied[top : top + template_patch.shape[0], left : left + template_patch.shape[1]] = True

        hole_local = np.asarray(spec["hole_local"], dtype=bool)
        hole_global = np.zeros_like(grid, dtype=bool)
        hole_global[top : top + template_patch.shape[0], left : left + template_patch.shape[1]] = hole_local

        placed = dict(spec)
        placed["template_top"] = int(top)
        placed["template_left"] = int(left)
        placed["hole_global"] = hole_global
        placed_specs.append(placed)

        expected[top : top + template_patch.shape[0], left : left + template_patch.shape[1]] = template_patch
        expected[hole_global] = int(spec["fragment_color"])

    piece_entries: list[tuple[int, np.ndarray]] = []
    for spec in placed_specs:
        fragment_color = int(spec["fragment_color"])
        piece_patches = [np.asarray(patch, dtype=int) for patch in spec["piece_patches"]]
        random.shuffle(piece_patches)
        for patch in piece_patches:
            piece_entries.append((fragment_color, patch))
    random.shuffle(piece_entries)

    for fragment_color, piece_patch in piece_entries:
        slot = _find_slot(occupied, piece_patch.shape, PLACE_GAP)
        if slot is None:
            return None
        top, left = slot
        _place_patch(grid, piece_patch, top, left, bg)
        occupied[top : top + piece_patch.shape[0], left : left + piece_patch.shape[1]] = True
        piece_mask = piece_patch == int(fragment_color)
        expected[top : top + piece_patch.shape[0], left : left + piece_patch.shape[1]][piece_mask] = int(bg)

    return grid, expected


def _generate_attempt() -> np.ndarray | None:
    template_count = random.choice(TEMPLATE_COUNT_OPTIONS)
    colors = random.sample(range(10), 1 + 2 * template_count)
    bg = int(colors[0])
    template_colors = [int(c) for c in colors[1 : 1 + template_count]]
    fragment_colors = [int(c) for c in colors[1 + template_count : 1 + 2 * template_count]]

    specs = _sample_template_specs(bg, template_colors, fragment_colors)
    if specs is None:
        return None

    for _ in range(LAYOUT_ATTEMPTS):
        scene = _compose_scene(specs, bg)
        if scene is None:
            continue
        inp, expected = scene
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
