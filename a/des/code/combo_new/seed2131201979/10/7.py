import random
from typing import Dict, List, Sequence, Tuple

import numpy as np


BG = 0
GREY = 5
PIECE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: List[List[Tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: List[Tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: Sequence[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    rows = [int(r) for r, _ in coords]
    cols = [int(c) for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: Sequence[Tuple[int, int]]) -> Tuple[Tuple[Tuple[int, int], ...], int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    norm = tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))
    return norm, int(r1 - r0 + 1), int(c1 - c0 + 1)


def _shape_to_mask(shape: Sequence[Tuple[int, int]]) -> np.ndarray:
    norm, h, w = _normalize(shape)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        mask[r, c] = True
    return mask


def _orientations(shape: Sequence[Tuple[int, int]]) -> List[Tuple[Tuple[int, int], ...]]:
    cur, _, _ = _normalize(shape)
    seen = set()
    out: List[Tuple[Tuple[int, int], ...]] = []
    for _ in range(4):
        if cur not in seen:
            out.append(cur)
            seen.add(cur)
        height = max(r for r, _ in cur) + 1
        cur = tuple(sorted((int(c), int(height - 1 - r)) for r, c in cur))
    return out


def _canonical_shape(shape: Sequence[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    return min(_orientations(shape))


def _extract_templates(grid: np.ndarray) -> List[Dict[str, object]]:
    templates: List[Dict[str, object]] = []
    for comp in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        assert r1 - r0 + 1 >= 3 and c1 - c0 + 1 >= 3, "template rectangle must be at least 3x3"
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(np.isin(patch, (BG, GREY))), "template bbox may contain only grey and background"
        assert np.all(patch[0, :] == GREY) and np.all(patch[-1, :] == GREY)
        assert np.all(patch[:, 0] == GREY) and np.all(patch[:, -1] == GREY)

        hole_mask = patch == BG
        assert not hole_mask[0, :].any() and not hole_mask[-1, :].any()
        assert not hole_mask[:, 0].any() and not hole_mask[:, -1].any()
        hole_comps = _components(hole_mask)
        assert len(hole_comps) == 1, "each grey template must contain exactly one enclosed hole"

        hole = hole_comps[0]
        hole_shape, _, _ = _normalize(hole)
        hr0, hc0, hr1, hc1 = _bbox(hole)
        hole_abs = tuple(sorted((r0 + r, c0 + c) for r, c in hole))
        templates.append(
            {
                "bbox": (r0, c0, r1, c1),
                "hole_shape": hole_shape,
                "hole_bbox": (r0 + hr0, c0 + hc0, r0 + hr1, c0 + hc1),
                "hole_abs": hole_abs,
            }
        )
    templates.sort(key=lambda item: item["bbox"])  # type: ignore[arg-type]
    return templates


def _extract_pieces(grid: np.ndarray) -> List[Dict[str, object]]:
    pieces: List[Dict[str, object]] = []
    for comp in _components((grid != BG) & (grid != GREY)):
        colors = {int(grid[r, c]) for r, c in comp}
        assert len(colors) == 1, "each fragment must be monochrome"
        shape, _, _ = _normalize(comp)
        orientation_map: Dict[Tuple[Tuple[int, int], ...], Tuple[Tuple[int, int], ...]] = {}
        for ori in _orientations(shape):
            orientation_map.setdefault(ori, ori)
        pieces.append(
            {
                "color": next(iter(colors)),
                "coords": tuple(sorted((int(r), int(c)) for r, c in comp)),
                "orientations": orientation_map,
            }
        )
    pieces.sort(key=lambda item: int(item["color"]))  # type: ignore[arg-type]
    return pieces


def _solve_assignment(candidates: Sequence[List[Tuple[int, Tuple[Tuple[int, int], ...]]]]) -> List[Tuple[int, Tuple[Tuple[int, int], ...]]]:
    order = sorted(range(len(candidates)), key=lambda idx: len(candidates[idx]))
    used_holes = set()
    current: List[Tuple[int, Tuple[Tuple[int, int], ...]] | None] = [None] * len(candidates)
    solutions: List[List[Tuple[int, Tuple[Tuple[int, int], ...]]]] = []

    def dfs(pos: int) -> None:
        if len(solutions) > 1:
            return
        if pos == len(order):
            solutions.append([item for item in current if item is not None])
            return

        piece_idx = order[pos]
        for hole_idx, shape in candidates[piece_idx]:
            if hole_idx in used_holes:
                continue
            used_holes.add(hole_idx)
            current[piece_idx] = (hole_idx, shape)
            dfs(pos + 1)
            current[piece_idx] = None
            used_holes.remove(hole_idx)

    dfs(0)
    assert len(solutions) == 1, "fragment-to-hole matching must be unique"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each scattered colored fragment into the unique grey template hole that matches its shape.

    Input structure:
    - Black background.
    - The left side contains several grey rectangular template objects, each solid except for one enclosed black hole.
    - The right side contains scattered monochrome colored fragments, one per template hole, and fragment shapes may be rotated relative to their matching holes.

    Rule:
    - Detect every grey rectangle and extract the exact hole shape inside it.
    - Detect every colored fragment as a separate 4-connected component and compare its shape against all hole shapes under 0, 90, 180, and 270 degree rotations.
    - For each fragment, choose the unique hole it can match, erase the fragment from its original location, and paint its color into that hole.
    - Keep the grey template material and all remaining background cells unchanged.

    Key insight:
    - A fragment does not belong to the hole with the same visible orientation; the solver has to mentally rotate each piece until one hole silhouette fits exactly.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    templates = _extract_templates(g)
    pieces = _extract_pieces(g)
    assert templates and pieces and len(templates) == len(pieces), "template count must equal piece count"

    candidate_lists: List[List[Tuple[int, Tuple[Tuple[int, int], ...]]]] = []
    for piece in pieces:
        matches: List[Tuple[int, Tuple[Tuple[int, int], ...]]] = []
        orientations = piece["orientations"]  # type: ignore[assignment]
        for hole_idx, template in enumerate(templates):
            hole_shape = template["hole_shape"]  # type: ignore[index]
            if hole_shape in orientations:
                matches.append((hole_idx, orientations[hole_shape]))  # type: ignore[index]
        assert matches, "every fragment must match at least one hole"
        candidate_lists.append(matches)

    assignment = _solve_assignment(candidate_lists)
    out = g.copy()

    for piece in pieces:
        for r, c in piece["coords"]:  # type: ignore[index]
            out[r, c] = BG

    for piece, (hole_idx, _) in zip(pieces, assignment):
        for r, c in templates[hole_idx]["hole_abs"]:  # type: ignore[index]
            out[r, c] = int(piece["color"])  # type: ignore[index]

    return out


def _random_hole_shape(rng: random.Random) -> Tuple[Tuple[int, int], ...] | None:
    for _ in range(200):
        target_area = rng.randint(3, 6)
        cells = {(0, 0)}
        while len(cells) < target_area:
            frontier = []
            for r, c in cells:
                for dr, dc in DIRS4:
                    nxt = (r + dr, c + dc)
                    if nxt not in cells:
                        frontier.append(nxt)
            cells.add(rng.choice(frontier))

        shape, h, w = _normalize(sorted(cells))
        if h < 2 or w < 2 or h > 4 or w > 4:
            continue
        if len(shape) == h * w:
            continue
        if len(_orientations(shape)) < 2:
            continue
        return shape
    return None


def _make_template_patch(
    shape: Sequence[Tuple[int, int]], rng: random.Random
) -> Tuple[np.ndarray, Tuple[Tuple[int, int], ...]]:
    mask = _shape_to_mask(shape)
    h, w = mask.shape
    top = rng.randint(1, 2)
    bottom = rng.randint(1, 2)
    left = rng.randint(1, 2)
    right = rng.randint(1, 2)
    patch = np.full((top + h + bottom, left + w + right), GREY, dtype=int)
    hole_rel: List[Tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                patch[top + r, left + c] = BG
                hole_rel.append((top + r, left + c))
    return patch, tuple(sorted(hole_rel))


def _make_piece_patch(shape: Sequence[Tuple[int, int]], color: int, rotation: int) -> np.ndarray:
    mask = _shape_to_mask(shape)
    rot_mask = np.rot90(mask, rotation)
    patch = np.full(rot_mask.shape, BG, dtype=int)
    patch[rot_mask] = color
    return patch


def _layout_block(
    patches: Sequence[np.ndarray],
    rng: random.Random,
    column_options: Sequence[int],
    gap_options: Sequence[int],
) -> Tuple[List[Tuple[int, int]], Tuple[int, int]] | None:
    count = len(patches)
    patch_shapes = [tuple(map(int, patch.shape)) for patch in patches]
    choices = list(column_options)
    rng.shuffle(choices)

    for cols in choices:
        if cols < 1:
            continue
        rows = (count + cols - 1) // cols
        order = list(range(count))
        rng.shuffle(order)
        row_heights = [0] * rows
        col_widths = [0] * cols
        slots: Dict[int, Tuple[int, int]] = {}
        for order_idx, patch_idx in enumerate(order):
            rr = order_idx // cols
            cc = order_idx % cols
            slots[patch_idx] = (rr, cc)
            ph, pw = patch_shapes[patch_idx]
            row_heights[rr] = max(row_heights[rr], ph)
            col_widths[cc] = max(col_widths[cc], pw)

        gap = rng.choice(tuple(gap_options))
        row_extra = [rng.randint(0, 1) for _ in range(rows)]
        col_extra = [rng.randint(0, 1) for _ in range(cols)]
        total_h = sum(row_heights) + sum(row_extra) + gap * max(0, rows - 1)
        total_w = sum(col_widths) + sum(col_extra) + gap * max(0, cols - 1)

        row_starts = []
        acc = 0
        for rr in range(rows):
            row_starts.append(acc)
            acc += row_heights[rr] + row_extra[rr] + gap

        col_starts = []
        acc = 0
        for cc in range(cols):
            col_starts.append(acc)
            acc += col_widths[cc] + col_extra[cc] + gap

        positions: List[Tuple[int, int]] = [(0, 0)] * count
        for patch_idx, (rr, cc) in slots.items():
            ph, pw = patch_shapes[patch_idx]
            cell_h = row_heights[rr] + row_extra[rr]
            cell_w = col_widths[cc] + col_extra[cc]
            r0 = row_starts[rr] + rng.randint(0, cell_h - ph)
            c0 = col_starts[cc] + rng.randint(0, cell_w - pw)
            positions[patch_idx] = (r0, c0)

        return positions, (total_h, total_w)

    return None


def _stamp_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    region = grid[top : top + h, left : left + w]
    mask = patch != BG
    assert not np.any(region[mask] != BG), "patch placement must not overlap existing foreground"
    region[mask] = patch[mask]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    piece_count = rng.randint(2, 4)
    colors = rng.sample(PIECE_COLORS, piece_count)

    shapes: List[Tuple[Tuple[int, int], ...]] = []
    seen_shapes = set()
    while len(shapes) < piece_count:
        shape = _random_hole_shape(rng)
        if shape is None:
            return None
        canonical = _canonical_shape(shape)
        if canonical in seen_shapes:
            continue
        shapes.append(shape)
        seen_shapes.add(canonical)

    template_patches: List[np.ndarray] = []
    piece_patches: List[np.ndarray] = []
    hole_rel_cells: List[Tuple[Tuple[int, int], ...]] = []
    rotations: List[int] = []
    for shape, color in zip(shapes, colors):
        template_patch, hole_rel = _make_template_patch(shape, rng)
        rotation = rng.randint(0, 3)
        template_patches.append(template_patch)
        piece_patches.append(_make_piece_patch(shape, color, rotation))
        hole_rel_cells.append(hole_rel)
        rotations.append(rotation)

    template_layout = _layout_block(
        template_patches,
        rng,
        column_options=(2, 1) if piece_count <= 2 else (2,),
        gap_options=(2, 3),
    )
    if template_layout is None:
        return None
    template_positions, (left_h, left_w) = template_layout

    piece_layout = _layout_block(
        piece_patches,
        rng,
        column_options=(1, 2) if piece_count >= 3 else (1, 2),
        gap_options=(2, 3),
    )
    if piece_layout is None:
        return None
    piece_positions, (right_h, right_w) = piece_layout

    sep = rng.randint(2, 4)
    outer_margin = 1
    total_h = max(left_h, right_h) + 2 * outer_margin
    total_w = left_w + sep + right_w + 2 * outer_margin
    if total_h > 30 or total_w > 30:
        outer_margin = 0
        total_h = max(left_h, right_h)
        total_w = left_w + sep + right_w
    if total_h > 30 or total_w > 30:
        return None

    grid = np.full((total_h, total_w), BG, dtype=int)
    left_top_slack = total_h - left_h - 2 * outer_margin
    right_top_slack = total_h - right_h - 2 * outer_margin
    left_top = outer_margin + (rng.randint(0, left_top_slack) if left_top_slack > 0 else 0)
    right_top = outer_margin + (rng.randint(0, right_top_slack) if right_top_slack > 0 else 0)
    left_col = outer_margin
    right_col = outer_margin + left_w + sep

    expected = grid.copy()

    for patch, (r0, c0) in zip(template_patches, template_positions):
        abs_r = left_top + r0
        abs_c = left_col + c0
        _stamp_patch(grid, patch, abs_r, abs_c)
        _stamp_patch(expected, patch, abs_r, abs_c)

    for idx, patch in enumerate(piece_patches):
        pr, pc = piece_positions[idx]
        abs_r = right_top + pr
        abs_c = right_col + pc
        _stamp_patch(grid, patch, abs_r, abs_c)

        piece_color = int(colors[idx])
        expected[abs_r : abs_r + patch.shape[0], abs_c : abs_c + patch.shape[1]][patch != BG] = BG

        tr, tc = template_positions[idx]
        for dr, dc in hole_rel_cells[idx]:
            expected[left_top + tr + dr, left_col + tc + dc] = piece_color

    if len(set(rotations)) == 1 and rotations[0] == 0:
        return None

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
