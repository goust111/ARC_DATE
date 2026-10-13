from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
_N4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bool(mask: np.ndarray) -> np.ndarray:
    box = _bbox(mask)
    if box is None:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    cropped = _crop_bool(mask.astype(bool))
    return tuple(map(int, cropped.shape)), cropped.tobytes()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = mask.astype(bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in _N4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(cells, dtype=int))
    return parts


def _is_connected(mask: np.ndarray) -> bool:
    if not bool(mask.any()):
        return False
    return len(_components(mask)) == 1


def _has_internal_void(mask: np.ndarray) -> bool:
    bg = ~mask.astype(bool)
    h, w = bg.shape
    seen = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []

    for r in range(h):
        for c in (0, w - 1):
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        for dr, dc in _N4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))

    return bool((bg & (~seen)).any())


def _is_valid_hole_shape(mask: np.ndarray) -> bool:
    return bool(mask.any()) and _is_connected(mask) and not _has_internal_void(mask)


def _mask_from_coords(shape: tuple[int, int], coords: set[tuple[int, int]]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for r, c in coords:
        out[int(r), int(c)] = True
    return out


def _most_common_color(arr: np.ndarray) -> int:
    return int(Counter(map(int, arr.ravel())).most_common(1)[0][0])


def _find_separator_row(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    candidates = []
    for r in range(h):
        color = int(grid[r, 0])
        if np.all(grid[r] == color) and int(np.count_nonzero(grid == color)) == w:
            candidates.append((r, color))
    assert len(candidates) == 1, "expected exactly one unique full-width separator row"
    return candidates[0]


def _panel_key_mask(grid: np.ndarray) -> tuple[np.ndarray, int]:
    g = np.asarray(grid, dtype=int)
    sep_r, _sep_color = _find_separator_row(g)
    panel_h = int(sep_r)
    work_r0 = 2 * panel_h + 1
    assert work_r0 < g.shape[0], "workspace must exist below the two panels"

    top = g[:panel_h, :]
    bottom = g[sep_r + 1 : sep_r + 1 + panel_h, :]
    assert top.shape == bottom.shape and top.size > 0

    panel_bg = _most_common_color(np.concatenate([top.ravel(), bottom.ravel()]))
    key_mask = (top != panel_bg) ^ (bottom != panel_bg)
    assert bool(key_mask.any()), "panel xor must produce a non-empty key"
    return key_mask, work_r0


def _enclosed_black_mask(crop: np.ndarray) -> np.ndarray:
    black = crop == BLACK
    h, w = crop.shape
    exterior = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []

    for r in range(h):
        for c in (0, w - 1):
            if black[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if black[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        for dr, dc in _N4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and black[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                stack.append((nr, nc))

    return black & (~exterior)


def _workspace_holes(workspace: np.ndarray) -> list[dict]:
    comp_mask = workspace != BLACK
    holes: list[dict] = []
    for coords in _components(comp_mask):
        colors = {int(workspace[r, c]) for r, c in coords}
        assert len(colors) == 1, "each rectangle should be monochromatic"
        rect_color = int(next(iter(colors)))
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        crop = workspace[r0 : r1 + 1, c0 : c1 + 1]
        hole_mask = _enclosed_black_mask(crop)
        assert bool(hole_mask.any()), "each workspace rectangle must contain a hole"
        holes.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "color": rect_color,
                "hole_mask": hole_mask,
                "signature": _shape_signature(hole_mask),
            }
        )
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Derive a hidden key from two aligned panels and fill the only matching workspace hole.

    Input structure:
    - The upper region contains two equal-height panels separated by one full-width separator row whose color appears nowhere else.
    - The two panels share the same background, are aligned cell-by-cell, and each uses exactly one foreground color.
    - Below the panels is a workspace containing several solid rectangles, each with one enclosed black hole.

    Rule:
    - Compare the two panels in alignment and take the cells where exactly one panel has foreground; this symmetric-difference mask is the hidden key.
    - In the workspace, find the unique hole whose shape is exactly the same as the key after bbox-normalization, allowing translation only.
    - Output only the workspace, with that one hole filled by its rectangle color and every other rectangle and hole left unchanged.

    Key insight:
    - The key never appears directly in the input; it must be reconstructed from the XOR of the two panel foreground masks and then matched by normalized shape, not by color or by rotation.
    """
    g = np.asarray(grid, dtype=int)
    key_mask, work_r0 = _panel_key_mask(g)
    key_sig = _shape_signature(key_mask)

    workspace = g[work_r0:, :].copy()
    holes = _workspace_holes(workspace)
    matches = [hole for hole in holes if hole["signature"] == key_sig]
    assert len(matches) == 1, "expected exactly one matching hole"

    match = matches[0]
    r0, c0, r1, c1 = match["bbox"]
    hole_mask = match["hole_mask"]
    patch = workspace[r0 : r1 + 1, c0 : c1 + 1]
    patch[hole_mask] = int(match["color"])
    return workspace


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _grow_connected_shape(height: int, width: int, area: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = True
    frontier = set()
    for dr, dc in _N4:
        nr, nc = start[0] + dr, start[1] + dc
        if 0 <= nr < height and 0 <= nc < width:
            frontier.add((nr, nc))
    while int(mask.sum()) < area and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        if mask[cell]:
            continue
        mask[cell] = True
        for dr, dc in _N4:
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                frontier.add((nr, nc))
    return mask


def _random_key_shape(max_h: int, max_w: int) -> np.ndarray:
    for _ in range(400):
        height = random.randint(3, max_h)
        width = random.randint(3, max_w)
        max_area = height * width - 1
        min_area = max(5, height + width - 2)
        if min_area > max_area:
            continue
        area = random.randint(min_area, max_area)
        mask = _grow_connected_shape(height, width, area)
        mask = _crop_bool(mask)
        if mask.shape[0] < 3 or mask.shape[1] < 3:
            continue
        if not _is_valid_hole_shape(mask):
            continue
        if not _touches_all_sides(mask):
            continue
        sig = _shape_signature(mask)
        rot_sig = _shape_signature(np.rot90(mask))
        flip_sig = _shape_signature(np.fliplr(mask))
        if random.random() < 0.7 and rot_sig == sig and flip_sig == sig:
            continue
        return mask
    raise RuntimeError("failed to generate a usable key shape")


def _tree_partition(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    cell_set = set(cells)
    root = random.choice(cells)
    parent: dict[tuple[int, int], tuple[int, int] | None] = {root: None}
    order: list[tuple[int, int]] = []
    stack = [root]
    while stack:
        cell = stack.pop()
        order.append(cell)
        nbrs = []
        for dr, dc in _N4:
            nxt = (cell[0] + dr, cell[1] + dc)
            if nxt in cell_set and nxt not in parent:
                nbrs.append(nxt)
        random.shuffle(nbrs)
        for nxt in nbrs:
            parent[nxt] = cell
            stack.append(nxt)

    children: dict[tuple[int, int], list[tuple[int, int]]] = {cell: [] for cell in cells}
    for cell, par in parent.items():
        if par is not None:
            children[par].append(cell)

    subtree: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for cell in reversed(order):
        nodes = {cell}
        for child in children[cell]:
            nodes |= subtree[child]
        subtree[cell] = nodes

    total = len(cells)
    preferred = [cell for cell in order[1:] if 2 <= len(subtree[cell]) <= total - 2]
    candidates = preferred or [cell for cell in order[1:] if 1 <= len(subtree[cell]) <= total - 1]
    cut = random.choice(candidates)
    part_a = subtree[cut]
    part_b = cell_set - part_a
    return _mask_from_coords(mask.shape, part_a), _mask_from_coords(mask.shape, part_b)


def _all_positions(occ: np.ndarray, patch: np.ndarray, padding: int = 0) -> list[tuple[int, int]]:
    h, w = occ.shape
    ph, pw = patch.shape
    cells = np.argwhere(patch)
    positions: list[tuple[int, int]] = []
    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            ok = True
            for rr, cc in cells:
                gr = r0 + int(rr)
                gc = c0 + int(cc)
                for dr in range(-padding, padding + 1):
                    for dc in range(-padding, padding + 1):
                        nr, nc = gr + dr, gc + dc
                        if 0 <= nr < h and 0 <= nc < w and occ[nr, nc]:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                positions.append((r0, c0))
    return positions


def _place_mask(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    ph, pw = patch.shape
    canvas[r0 : r0 + ph, c0 : c0 + pw] |= patch


def _embed_shape(mask: np.ndarray, shape: tuple[int, int], anchor: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    h, w = mask.shape
    r0, c0 = anchor
    out[r0 : r0 + h, c0 : c0 + w] = mask
    return out


def _small_overlap_shape() -> np.ndarray:
    choices = [
        np.array([[1]], dtype=bool),
        np.array([[1, 1]], dtype=bool),
        np.array([[1], [1]], dtype=bool),
        np.array([[1, 1], [1, 0]], dtype=bool),
    ]
    return random.choice(choices).copy()


def _distinct_transforms(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen = {_shape_signature(mask)}
    for cand in (
        np.rot90(mask, 1),
        np.rot90(mask, 2),
        np.rot90(mask, 3),
        np.fliplr(mask),
        np.flipud(mask),
    ):
        cropped = _crop_bool(cand)
        sig = _shape_signature(cropped)
        if sig not in seen:
            seen.add(sig)
            out.append(cropped)
    return out


def _same_area_variant(mask: np.ndarray) -> np.ndarray | None:
    base = mask.astype(bool)
    base_sig = _shape_signature(base)
    true_cells = [tuple(map(int, rc)) for rc in np.argwhere(base)]
    for _ in range(300):
        rr, cc = random.choice(true_cells)
        reduced = base.copy()
        reduced[rr, cc] = False
        if not bool(reduced.any()) or not _is_connected(reduced):
            continue
        frontier = []
        h, w = reduced.shape
        for r in range(h):
            for c in range(w):
                if reduced[r, c]:
                    continue
                for dr, dc in _N4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and reduced[nr, nc]:
                        frontier.append((r, c))
                        break
        if not frontier:
            continue
        ar, ac = random.choice(frontier)
        cand = reduced.copy()
        cand[ar, ac] = True
        cand = _crop_bool(cand)
        if _is_valid_hole_shape(cand) and _shape_signature(cand) != base_sig:
            return cand
    return None


def _random_distractor_shape(key: np.ndarray, exclude: set[tuple[tuple[int, int], bytes]]) -> np.ndarray:
    kh, kw = key.shape
    key_area = int(key.sum())
    for _ in range(400):
        height = random.randint(max(3, kh - 1), min(6, kh + 2))
        width = random.randint(max(3, kw - 1), min(7, kw + 2))
        max_area = height * width - 1
        min_area = max(4, min(max_area, key_area - 2))
        if min_area > max_area:
            continue
        if random.random() < 0.5:
            area = max(4, min(max_area, key_area + random.choice([-2, -1, 1, 2])))
        else:
            area = random.randint(min_area, max_area)
        area = max(4, min(area, max_area))
        cand = _grow_connected_shape(height, width, area)
        cand = _crop_bool(cand)
        sig = _shape_signature(cand)
        if cand.shape[0] < 2 or cand.shape[1] < 2:
            continue
        if not _is_valid_hole_shape(cand):
            continue
        if sig not in exclude:
            return cand
    raise RuntimeError("failed to generate a distractor")


def _rectangle_patch(hole_shape: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    hh, hw = hole_shape.shape
    top = random.randint(1, 2)
    bottom = random.randint(1, 2)
    left = random.randint(1, 2)
    right = random.randint(1, 2)
    rect = np.full((hh + top + bottom, hw + left + right), GREY, dtype=int)
    hole_mask = np.zeros_like(rect, dtype=bool)
    hole_mask[top : top + hh, left : left + hw] = hole_shape
    rect[hole_mask] = BLACK
    return rect, hole_mask, GREY


def _build_case() -> tuple[np.ndarray, np.ndarray]:
    width = random.randint(18, 28)
    panel_h = random.randint(5, 8)
    work_h_max = 30 - (2 * panel_h + 1)
    work_h = random.randint(max(9, work_h_max - 3), work_h_max)

    key = _random_key_shape(min(6, panel_h), min(8, width - 4))
    part_a, part_b = _tree_partition(key)
    assert bool(part_a.any()) and bool(part_b.any())

    kh, kw = key.shape
    anchor_r = random.randint(0, panel_h - kh)
    anchor_c = random.randint(1, width - kw - 1)
    key_full = _embed_shape(key, (panel_h, width), (anchor_r, anchor_c))
    a_full = _embed_shape(part_a, (panel_h, width), (anchor_r, anchor_c))
    b_full = _embed_shape(part_b, (panel_h, width), (anchor_r, anchor_c))

    overlap_patch = _small_overlap_shape()
    overlap_positions = _all_positions(key_full, overlap_patch, padding=1)
    if not overlap_positions:
        raise RuntimeError("no place for overlap patch")
    overlap_r, overlap_c = random.choice(overlap_positions)
    overlap_full = np.zeros((panel_h, width), dtype=bool)
    _place_mask(overlap_full, overlap_patch, overlap_r, overlap_c)

    top_mask = a_full | overlap_full
    bottom_mask = b_full | overlap_full

    n_rects = random.randint(3, 5)
    shapes = [key]
    near_match_added = False

    transforms = _distinct_transforms(key)
    if transforms:
        shapes.append(random.choice(transforms))
        near_match_added = True

    if not near_match_added:
        variant = _same_area_variant(key)
        if variant is not None:
            shapes.append(variant)
            near_match_added = True

    exclude = {_shape_signature(shape) for shape in shapes}
    while len(shapes) < n_rects:
        cand = _random_distractor_shape(key, exclude)
        shapes.append(cand)
        exclude.add(_shape_signature(cand))

    if not near_match_added:
        raise RuntimeError("expected at least one explicit counterexample shape")

    order = list(range(len(shapes)))
    random.shuffle(order)
    match_index = order.index(0)
    ordered_shapes = [shapes[idx] for idx in order]

    workspace = np.full((work_h, width), BLACK, dtype=int)
    occ = np.zeros((work_h, width), dtype=bool)
    hole_records: list[dict] = []

    for idx, shape in enumerate(ordered_shapes):
        rect, hole_mask, rect_color = _rectangle_patch(shape)
        footprint = np.ones(rect.shape, dtype=bool)
        positions = _all_positions(occ, footprint, padding=1)
        if not positions:
            raise RuntimeError("workspace placement failed")
        r0, c0 = random.choice(positions)
        rh, rw = rect.shape
        workspace[r0 : r0 + rh, c0 : c0 + rw] = rect
        occ[r0 : r0 + rh, c0 : c0 + rw] = True
        hole_records.append(
            {
                "shape_sig": _shape_signature(shape),
                "bbox": (r0, c0, r0 + rh - 1, c0 + rw - 1),
                "hole_mask": hole_mask,
                "color": rect_color,
                "is_match": idx == match_index,
            }
        )

    palette = [1, 2, 3, 4, 6, 7, 8, 9]
    top_color, bottom_color, sep_color = random.sample(palette, 3)

    grid = np.full((2 * panel_h + 1 + work_h, width), BLACK, dtype=int)
    grid[:panel_h, :][top_mask] = top_color
    grid[panel_h, :] = sep_color
    grid[panel_h + 1 : 2 * panel_h + 1, :][bottom_mask] = bottom_color
    grid[2 * panel_h + 1 :, :] = workspace

    expected = workspace.copy()
    true_record = next(record for record in hole_records if record["is_match"])
    r0, c0, r1, c1 = true_record["bbox"]
    expected[r0 : r1 + 1, c0 : c1 + 1][true_record["hole_mask"]] = int(true_record["color"])

    key_sig = _shape_signature(key_full)
    matched = [record for record in hole_records if record["shape_sig"] == key_sig]
    if len(matched) != 1:
        raise RuntimeError("construction lost uniqueness of the matching hole")

    if top_color == bottom_color or top_color == sep_color or bottom_color == sep_color:
        raise RuntimeError("panel and separator colors must stay distinct")
    if not _is_connected(key_full):
        raise RuntimeError("key must be connected")
    if np.array_equal(_crop_bool(top_mask), _crop_bool(key_full)) or np.array_equal(_crop_bool(bottom_mask), _crop_bool(key_full)):
        raise RuntimeError("key must not appear directly in a panel")
    if not bool((top_mask & bottom_mask).any()):
        raise RuntimeError("shared overlap must exist so xor matters")
    if not bool((top_mask & (~bottom_mask)).any()) or not bool((bottom_mask & (~top_mask)).any()):
        raise RuntimeError("both panels must contribute exclusive foreground")
    if not near_match_added:
        raise RuntimeError("counterexample hole is required")
    if _shape_signature((top_mask ^ bottom_mask)) != key_sig:
        raise RuntimeError("panel xor must reconstruct the key exactly")
    if not np.array_equal(main(grid), expected):
        raise RuntimeError("generated case does not satisfy solver semantics")
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            grid, _expected = _build_case()
            return grid
        except (RuntimeError, AssertionError):
            continue
    raise RuntimeError("failed to generate a valid input")
