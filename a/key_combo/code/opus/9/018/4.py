from __future__ import annotations

from collections import Counter

import numpy as np


DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
FRAME_COLOR = 5
BORDER_COLORS = (3, 4, 7, 8)
CONTENT_COLORS = (6, 9)
TEMPLATE_COLORS = (1, 2)

# Fixed-orientation irregular polyominoes (3-5 cells). Matching preserves orientation.
SHAPES = (
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 1), (0, 2), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 1), (2, 0), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
    ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
    ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)),
)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _neighbors(r: int, c: int, connectivity: int) -> list[tuple[int, int]]:
    if connectivity == 8:
        return [
            (r - 1, c - 1), (r - 1, c), (r - 1, c + 1),
            (r, c - 1),                 (r, c + 1),
            (r + 1, c - 1), (r + 1, c), (r + 1, c + 1),
        ]
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors(cr, cc, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize_coords(coords: np.ndarray | list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    arr = np.array(coords, dtype=int)
    r0, c0, _, _ = _bbox_coords(arr)
    out = sorted((int(r - r0), int(c - c0)) for r, c in arr)
    return tuple(out)


def _flood_from_bbox_side(border_set: set[tuple[int, int]], bbox: tuple[int, int, int, int], side: str) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if side == "U":
        seeds = [(r0, c) for c in range(c0, c1 + 1) if (r0, c) not in border_set]
    elif side == "D":
        seeds = [(r1, c) for c in range(c0, c1 + 1) if (r1, c) not in border_set]
    elif side == "L":
        seeds = [(r, c0) for r in range(r0, r1 + 1) if (r, c0) not in border_set]
    else:
        seeds = [(r, c1) for r in range(r0, r1 + 1) if (r, c1) not in border_set]

    seen = set(seeds)
    stack = list(seeds)
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors(r, c, 4):
            if not (r0 <= nr <= r1 and c0 <= nc <= c1):
                continue
            if (nr, nc) in seen or (nr, nc) in border_set:
                continue
            seen.add((nr, nc))
            stack.append((nr, nc))
    return seen


def _build_canonical_cup(left_arm: int, right_arm: int, left_wall: int, right_wall: int, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    width = left_arm + right_arm + 1
    height = max(left_arm + left_wall, right_arm + right_wall) + 1
    patch = np.zeros((height, width), dtype=int)
    patch[0, left_arm] = color
    patch[np.arange(1, left_arm + 1), left_arm - np.arange(1, left_arm + 1)] = color
    patch[np.arange(1, right_arm + 1), left_arm + np.arange(1, right_arm + 1)] = color
    if left_wall:
        patch[left_arm + 1:left_arm + left_wall + 1, 0] = color
    if right_wall:
        patch[right_arm + 1:right_arm + right_wall + 1, width - 1] = color
    return patch, (0, left_arm)


def _mouth_interior_coords(patch: np.ndarray, border_color: int, mouth_side: str) -> list[tuple[int, int]]:
    h, w = patch.shape
    border = patch == border_color
    if mouth_side == "U":
        seeds = [(0, c) for c in range(w) if not border[0, c]]
    elif mouth_side == "D":
        seeds = [(h - 1, c) for c in range(w) if not border[h - 1, c]]
    elif mouth_side == "L":
        seeds = [(r, 0) for r in range(h) if not border[r, 0]]
    else:
        seeds = [(r, w - 1) for r in range(h) if not border[r, w - 1]]

    seen = set(seeds)
    stack = list(seeds)
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors(r, c, 4):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and not border[nr, nc]:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return sorted(seen)


def _rotate_cup(
    patch: np.ndarray,
    spike: tuple[int, int],
    mouth: list[tuple[int, int]],
    direction: str,
) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    h, w = patch.shape
    sr, sc = spike
    if direction == "U":
        return patch.copy(), (sr, sc), list(mouth)
    if direction == "D":
        return np.flipud(patch).copy(), (h - 1 - sr, sc), [(h - 1 - r, c) for r, c in mouth]
    if direction == "L":
        return np.rot90(patch, 1).copy(), (w - 1 - sc, sr), [(w - 1 - c, r) for r, c in mouth]
    return np.rot90(patch, -1).copy(), (sc, h - 1 - sr), [(c, h - 1 - r) for r, c in mouth]


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _translations_in_allowed(
    shape: tuple[tuple[int, int], ...],
    allowed: list[tuple[int, int]],
) -> list[list[tuple[int, int]]]:
    allowed_set = set(allowed)
    arr = np.array(allowed, dtype=int)
    r0, c0, r1, c1 = _bbox_coords(arr)
    sh, sw = _shape_dims(shape)
    placements: list[list[tuple[int, int]]] = []
    for tr in range(r0, r1 - sh + 2):
        for tc in range(c0, c1 - sw + 2):
            pts = [(tr + r, tc + c) for r, c in shape]
            if all(pt in allowed_set for pt in pts):
                placements.append(pts)
    return placements


def _extract_template(grid: np.ndarray, bg: int) -> dict:
    candidates = []
    for comp in _components_mask(grid == FRAME_COLOR, connectivity=4):
        r0, c0, r1, c1 = _bbox_coords(comp)
        frame = {(int(r), int(c)) for r, c in comp}
        expected = {(r, c1) for r in range(r0, r1 + 1)} | {(r1, c) for c in range(c0, c1 + 1)}
        if frame != expected:
            continue
        inside = []
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                v = int(grid[r, c])
                if v != bg and v != FRAME_COLOR:
                    inside.append((r, c))
        if not inside:
            continue
        inside_arr = np.array(inside, dtype=int)
        shape = _normalize_coords(inside_arr)
        candidates.append(
            {
                "bbox": (r0, c0, r1, c1),
                "shape": shape,
                "erase_cells": list(frame) + inside,
            }
        )
    assert len(candidates) == 1, "expected one grey L-frame template"
    return candidates[0]


def _extract_cups(grid: np.ndarray, bg: int) -> list[dict]:
    cups = []
    h, w = grid.shape
    for color in sorted(set(map(int, grid.flat)) - {bg, FRAME_COLOR}):
        for comp in _components_mask(grid == color, connectivity=8):
            if comp.shape[0] < 5:
                continue
            r0, c0, r1, c1 = _bbox_coords(comp)
            border = {(int(r), int(c)) for r, c in comp}
            corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
            edge_groups = (
                ("U", [p for p in border if p[0] == r0]),
                ("D", [p for p in border if p[0] == r1]),
                ("L", [p for p in border if p[1] == c0]),
                ("R", [p for p in border if p[1] == c1]),
            )
            candidates = [(d, pts[0]) for d, pts in edge_groups if len(pts) == 1 and pts[0] not in corners]
            passing = []
            for direction, spike in candidates:
                mouth = _flood_from_bbox_side(border, (r0, c0, r1, c1), OPPOSITE[direction])
                content = {}
                for r, c in mouth:
                    v = int(grid[r, c])
                    if v != bg and v != color:
                        content.setdefault(v, []).append((r, c))
                if len(content) != 1:
                    continue
                content_color, content_cells = next(iter(content.items()))
                if not content_cells:
                    continue
                local_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
                for rr, cc in content_cells:
                    local_mask[rr - r0, cc - c0] = True
                if len(_components_mask(local_mask, connectivity=4)) != 1:
                    continue
                arr = np.array(content_cells, dtype=int)
                shape = _normalize_coords(arr)
                passing.append(
                    {
                        "border_color": color,
                        "direction": direction,
                        "spike": spike,
                        "content_color": int(content_color),
                        "content_cells": content_cells,
                        "shape": shape,
                    }
                )
            if len(passing) == 1:
                cup = passing[0]
                sr, sc = cup["spike"]
                dr, dc = DIR4[cup["direction"]]
                cup["path_cells"] = [
                    (sr + dr * step, sc + dc * step)
                    for step in range(1, len(cup["content_cells"]) + 1)
                    if 0 <= sr + dr * step < h and 0 <= sc + dc * step < w
                ]
                cups.append(cup)
    assert 3 <= len(cups) <= 4, "expected 3-4 cups"
    return cups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the L-frame template shape to choose which cup contents are ejected and which are erased.

    Input structure:
    - Black background contains one grey L-frame whose right and bottom legs bound a small colored template.
    - The rest of the grid contains 3-4 hollow cup/funnel frames, each in a distinct border color, with a tail spike on one side.
    - Each cup mouth contains one small 4-connected colored content shape; some content masks match the template mask after translation normalization.

    Rule:
    - Extract the template mask inside the grey L-frame, preserving orientation and ignoring color.
    - For each cup, infer the tail direction from the unique spike edge and extract the mouth content mask.
    - If a cup content mask matches the template mask, erase that content and redraw the same number of pixels outward from the spike along the tail direction; otherwise just erase the content.
    - Finally erase the grey L-frame and its template while keeping every cup border unchanged.

    Key insight:
    - The template is only a shape-based gate: it decides which cups activate, while each activated cup uses its own tail direction and content size for the ejection.
    """
    g = np.array(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    template = _extract_template(g, bg)
    cups = _extract_cups(g, bg)

    for cup in cups:
        for r, c in cup["content_cells"]:
            out[r, c] = bg
        if cup["shape"] != template["shape"]:
            continue
        for r, c in cup["path_cells"]:
            out[r, c] = cup["content_color"]

    for r, c in template["erase_cells"]:
        out[r, c] = bg
    return out


def _make_template_patch(
    rng: np.random.Generator,
    shape: tuple[tuple[int, int], ...],
    color: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    sh, sw = _shape_dims(shape)
    height = sh + int(rng.integers(1, 3))
    width = sw + int(rng.integers(1, 3))
    patch = np.zeros((height, width), dtype=int)
    patch[:, -1] = FRAME_COLOR
    patch[-1, :] = FRAME_COLOR
    allowed = [(r, c) for r in range(height - 1) for c in range(width - 1)]
    placements = _translations_in_allowed(shape, allowed)
    pts = placements[int(rng.integers(len(placements)))]
    for r, c in pts:
        patch[r, c] = color
    return patch, pts


def _template_origins(
    rng: np.random.Generator,
    h: int,
    w: int,
    patch_h: int,
    patch_w: int,
) -> list[tuple[int, int]]:
    anchors = []
    row_hi = max(0, min(2, h - patch_h))
    col_hi = max(0, min(2, w - patch_w))
    options = {
        "TL": (range(0, row_hi + 1), range(0, col_hi + 1)),
        "TR": (range(0, row_hi + 1), range(max(0, w - patch_w - col_hi), w - patch_w + 1)),
        "BL": (range(max(0, h - patch_h - row_hi), h - patch_h + 1), range(0, col_hi + 1)),
        "BR": (
            range(max(0, h - patch_h - row_hi), h - patch_h + 1),
            range(max(0, w - patch_w - col_hi), w - patch_w + 1),
        ),
    }
    keys = list(options)
    rng.shuffle(keys)
    for key in keys:
        rows, cols = options[key]
        for r0 in rows:
            for c0 in cols:
                anchors.append((int(r0), int(c0)))
    rng.shuffle(anchors)
    return anchors


def _sample_cup_patch(
    rng: np.random.Generator,
    border_color: int,
    direction: str,
    shape: tuple[tuple[int, int], ...],
    content_color: int,
) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    for _ in range(80):
        left_arm = int(rng.integers(2, 4))
        right_arm = int(rng.integers(2, 4))
        left_wall = int(rng.integers(1, 4))
        right_wall = int(rng.integers(1, 4))
        canonical, spike = _build_canonical_cup(left_arm, right_arm, left_wall, right_wall, border_color)
        mouth = _mouth_interior_coords(canonical, border_color, "D")
        patch, spike2, mouth2 = _rotate_cup(canonical, spike, mouth, direction)
        placements = _translations_in_allowed(shape, mouth2)
        if not placements:
            continue
        pts = placements[int(rng.integers(len(placements)))]
        patch = patch.copy()
        for r, c in pts:
            patch[r, c] = content_color
        return patch, spike2, pts
    raise RuntimeError("failed to build cup patch")


def _cells_with_margin(r0: int, c0: int, h: int, w: int, grid_h: int, grid_w: int) -> list[tuple[int, int]]:
    cells = []
    for r in range(max(0, r0 - 1), min(grid_h, r0 + h + 1)):
        for c in range(max(0, c0 - 1), min(grid_w, c0 + w + 1)):
            cells.append((r, c))
    return cells


def _candidate_origins(
    direction: str,
    grid_h: int,
    grid_w: int,
    patch_h: int,
    patch_w: int,
    spike: tuple[int, int],
    length: int,
) -> list[tuple[int, int]]:
    sr, sc = spike
    origins = []
    if direction == "U":
        r_min = length + 1
        r_max = min(grid_h - patch_h - 1, max(r_min, grid_h // 3))
        c_min, c_max = 1, grid_w - patch_w - 1
        for r0 in range(r_min, r_max + 1):
            for c0 in range(c_min, c_max + 1):
                origins.append((r0, c0))
    elif direction == "D":
        r_max = grid_h - patch_h - length - 1
        r_min = max(1, r_max - max(3, grid_h // 5))
        c_min, c_max = 1, grid_w - patch_w - 1
        for r0 in range(r_min, r_max + 1):
            for c0 in range(c_min, c_max + 1):
                origins.append((r0, c0))
    elif direction == "L":
        c_min = length + 1
        c_max = min(grid_w - patch_w - 1, max(c_min, grid_w // 3))
        r_min, r_max = 1, grid_h - patch_h - 1
        for r0 in range(r_min, r_max + 1):
            for c0 in range(c_min, c_max + 1):
                origins.append((r0, c0))
    else:
        c_max = grid_w - patch_w - length - 1
        c_min = max(1, c_max - max(3, grid_w // 5))
        r_min, r_max = 1, grid_h - patch_h - 1
        for r0 in range(r_min, r_max + 1):
            for c0 in range(c_min, c_max + 1):
                origins.append((r0, c0))
    return origins


def _path_to_edge(spike_global: tuple[int, int], direction: str, grid_h: int, grid_w: int) -> list[tuple[int, int]]:
    dr, dc = DIR4[direction]
    r, c = spike_global
    out = []
    while True:
        r += dr
        c += dc
        if not (0 <= r < grid_h and 0 <= c < grid_w):
            break
        out.append((r, c))
    return out


def _paint_patch(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0:r0 + h, c0:c0 + w] = np.where(patch != 0, patch, dst[r0:r0 + h, c0:c0 + w])


def _expected_from_meta(grid: np.ndarray, template_cells: list[tuple[int, int]], cups: list[dict], template_shape: tuple[tuple[int, int], ...]) -> np.ndarray:
    bg = _get_bg(grid)
    out = grid.copy()
    for cup in cups:
        for r, c in cup["content_cells"]:
            out[r, c] = bg
        if cup["shape"] != template_shape:
            continue
        sr, sc = cup["spike_global"]
        dr, dc = DIR4[cup["direction"]]
        for step in range(1, len(cup["content_cells"]) + 1):
            rr = sr + dr * step
            cc = sc + dc * step
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                out[rr, cc] = cup["content_color"]
    for r, c in template_cells:
        out[r, c] = bg
    return out


def _semantic_ok(grid: np.ndarray, expected: np.ndarray, template_shape: tuple[tuple[int, int], ...], cups_meta: list[dict]) -> bool:
    try:
        template = _extract_template(grid, 0)
        cups = _extract_cups(grid, 0)
    except Exception:
        return False
    if template["shape"] != template_shape:
        return False
    match_count = sum(cup["shape"] == template_shape for cup in cups)
    if match_count not in (1, 2):
        return False
    if match_count >= len(cups):
        return False
    silent = [cup for cup in cups if cup["shape"] != template_shape]
    if not any(len(cup["shape"]) == len(template_shape) for cup in silent):
        return False
    active_colors = {cup["content_color"] for cup in cups if cup["shape"] == template_shape}
    silent_colors = {cup["content_color"] for cup in silent}
    if not (active_colors & silent_colors):
        return False
    if not np.array_equal(main(grid), expected):
        return False
    # The generator metadata must agree on cup count and activation count.
    if len(cups_meta) != len(cups):
        return False
    if sum(cup["shape"] == template_shape for cup in cups_meta) != match_count:
        return False
    return True


def _choose_shapes(rng: np.random.Generator, cup_count: int) -> tuple[tuple[tuple[int, int], ...], list[tuple[tuple[int, int], ...]]]:
    template_shape = SHAPES[int(rng.integers(len(SHAPES)))]
    match_count = int(rng.integers(1, min(2, cup_count - 1) + 1))
    shapes = [template_shape] * match_count
    same_area = [shape for shape in SHAPES if len(shape) == len(template_shape) and shape != template_shape]
    other = [shape for shape in SHAPES if shape != template_shape]
    if cup_count - match_count > 0:
        if same_area:
            shapes.append(same_area[int(rng.integers(len(same_area)))])
        while len(shapes) < cup_count:
            shapes.append(other[int(rng.integers(len(other)))])
    rng.shuffle(shapes)
    return template_shape, shapes


def _generate_case(rng: np.random.Generator) -> np.ndarray | None:
    grid_h = int(rng.integers(20, 29))
    grid_w = int(rng.integers(20, 29))
    cup_count = int(rng.choice(np.array([3, 4])))
    template_shape, cup_shapes = _choose_shapes(rng, cup_count)

    grid = np.zeros((grid_h, grid_w), dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)
    reserved_paths = np.zeros((grid_h, grid_w), dtype=bool)

    template_color = int(rng.choice(np.array(TEMPLATE_COLORS)))
    template_patch, template_pts = _make_template_patch(rng, template_shape, template_color)
    placed_template = False
    template_cells_global: list[tuple[int, int]] = []
    for tr0, tc0 in _template_origins(rng, grid_h, grid_w, *template_patch.shape):
        margin = _cells_with_margin(tr0, tc0, template_patch.shape[0], template_patch.shape[1], grid_h, grid_w)
        if any(blocked[r, c] for r, c in margin):
            continue
        _paint_patch(grid, template_patch, tr0, tc0)
        for r, c in margin:
            blocked[r, c] = True
        template_cells_global = [(tr0 + r, tc0 + c) for r, c in template_pts]
        template_cells_global.extend(
            [(tr0 + r, tc0 + template_patch.shape[1] - 1) for r in range(template_patch.shape[0])]
        )
        template_cells_global.extend(
            [(tr0 + template_patch.shape[0] - 1, tc0 + c) for c in range(template_patch.shape[1])]
        )
        template_cells_global = sorted(set(template_cells_global))
        placed_template = True
        break
    if not placed_template:
        return None

    directions = ["U", "D", "L", "R"]
    rng.shuffle(directions)
    directions = directions[:cup_count]
    border_colors = list(BORDER_COLORS)
    rng.shuffle(border_colors)
    border_colors = border_colors[:cup_count]

    match_indices = [i for i, shape in enumerate(cup_shapes) if shape == template_shape]
    silent_indices = [i for i, shape in enumerate(cup_shapes) if shape != template_shape]
    content_colors = [int(rng.choice(np.array(CONTENT_COLORS))) for _ in range(cup_count)]
    shared_color = int(rng.choice(np.array(CONTENT_COLORS)))
    content_colors[match_indices[0]] = shared_color
    content_colors[silent_indices[0]] = shared_color

    cups_meta = []
    for idx in range(cup_count):
        direction = directions[idx]
        shape = cup_shapes[idx]
        border_color = int(border_colors[idx])
        content_color = int(content_colors[idx])
        try:
            patch, spike_local, content_local = _sample_cup_patch(rng, border_color, direction, shape, content_color)
        except RuntimeError:
            return None

        origins = _candidate_origins(direction, grid_h, grid_w, patch.shape[0], patch.shape[1], spike_local, len(shape))
        rng.shuffle(origins)
        placed = False
        for r0, c0 in origins:
            margin = _cells_with_margin(r0, c0, patch.shape[0], patch.shape[1], grid_h, grid_w)
            if any(blocked[r, c] for r, c in margin):
                continue
            spike_global = (r0 + spike_local[0], c0 + spike_local[1])
            ray = _path_to_edge(spike_global, direction, grid_h, grid_w)
            if len(ray) < len(shape):
                continue
            if any(blocked[r, c] or reserved_paths[r, c] for r, c in ray):
                continue
            if any(grid[r, c] != 0 for r, c in ray):
                continue
            _paint_patch(grid, patch, r0, c0)
            for r, c in margin:
                blocked[r, c] = True
            for r, c in ray:
                reserved_paths[r, c] = True
            cups_meta.append(
                {
                    "direction": direction,
                    "shape": shape,
                    "content_color": content_color,
                    "content_cells": [(r0 + r, c0 + c) for r, c in content_local],
                    "spike_global": spike_global,
                }
            )
            placed = True
            break
        if not placed:
            return None

    expected = _expected_from_meta(grid, template_cells_global, cups_meta, template_shape)
    if not _semantic_ok(grid, expected, template_shape, cups_meta):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        grid = _generate_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid template-filtered cup puzzle")
