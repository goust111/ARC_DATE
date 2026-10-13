from __future__ import annotations

import random

import numpy as np


Coord = tuple[int, int]


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize(coords: np.ndarray) -> np.ndarray:
    pts = np.asarray(coords, dtype=int)
    return pts - pts.min(axis=0)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.asarray(coords, dtype=int))
    return out


def _all_orientations(coords: np.ndarray) -> list[np.ndarray]:
    pts = _normalize(coords)
    transforms = [
        lambda r, c: (r, c),
        lambda r, c: (r, -c),
        lambda r, c: (-r, c),
        lambda r, c: (-r, -c),
        lambda r, c: (c, r),
        lambda r, c: (c, -r),
        lambda r, c: (-c, r),
        lambda r, c: (-c, -r),
    ]
    out = []
    seen = set()
    for transform in transforms:
        variant = np.asarray([transform(int(r), int(c)) for r, c in pts], dtype=int)
        variant = _normalize(variant)
        key = tuple(sorted(map(tuple, variant.tolist())))
        if key not in seen:
            seen.add(key)
            out.append(variant)
    return out


def _canonical_key(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    return min(tuple(sorted(map(tuple, orient.tolist()))) for orient in _all_orientations(coords))


def _piece_patch(coords: np.ndarray, color: int, bg: int) -> np.ndarray:
    pts = _normalize(coords)
    h = int(pts[:, 0].max()) + 1
    w = int(pts[:, 1].max()) + 1
    patch = np.full((h, w), int(bg), dtype=int)
    patch[pts[:, 0], pts[:, 1]] = int(color)
    return patch


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _find_frame(grid: np.ndarray, bg: int) -> dict:
    g = np.asarray(grid, dtype=int)
    candidates = []
    for color in sorted(set(map(int, g.ravel())) - {int(bg)}):
        for comp in _components(g == color):
            r0, c0, r1, c1 = _bbox(comp)
            patch_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            rel = comp - np.array([r0, c0], dtype=int)
            patch_mask[rel[:, 0], rel[:, 1]] = True
            if not (
                np.all(patch_mask[0, :])
                and np.all(patch_mask[-1, :])
                and np.all(patch_mask[:, 0])
                and np.all(patch_mask[:, -1])
            ):
                continue
            holes = []
            for hole in _components(~patch_mask):
                if np.all(hole[:, 0] > 0) and np.all(hole[:, 0] < patch_mask.shape[0] - 1) and np.all(
                    hole[:, 1] > 0
                ) and np.all(hole[:, 1] < patch_mask.shape[1] - 1):
                    holes.append(hole)
            if len(holes) != 1:
                continue
            hole = holes[0]
            if hole.shape[0] < 6:
                continue
            candidates.append(
                {
                    "color": int(color),
                    "coords": comp,
                    "bbox": (r0, c0, r1, c1),
                    "hole_local": hole,
                    "hole_global": hole + np.array([r0, c0], dtype=int),
                }
            )
    assert candidates, "expected a hollow frame"
    candidates.sort(key=lambda item: item["coords"].shape[0], reverse=True)
    top_area = candidates[0]["coords"].shape[0]
    top = [item for item in candidates if item["coords"].shape[0] == top_area]
    assert len(top) == 1, "expected a unique largest frame"
    return top[0]


def _extract_pieces(grid: np.ndarray, frame_coords: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int).copy()
    g[frame_coords[:, 0], frame_coords[:, 1]] = int(bg)
    pieces = []
    for comp in _components(g != int(bg)):
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1, "each fragment must be monochromatic"
        color = next(iter(colors))
        r0, c0, r1, c1 = _bbox(comp)
        rel = comp - np.array([r0, c0], dtype=int)
        pieces.append(
            {
                "color": int(color),
                "coords": rel,
                "area": int(rel.shape[0]),
                "bbox": (r0, c0, r1, c1),
                "variants": _all_orientations(rel),
            }
        )
    assert len(pieces) >= 3, "expected multiple fragments outside the frame"
    return pieces


def _placements_for_piece(piece: dict, hole_set: set[Coord], patch_shape: tuple[int, int]) -> dict[Coord, list[tuple[Coord, ...]]]:
    h, w = patch_shape
    placements_by_anchor: dict[Coord, list[tuple[Coord, ...]]] = {}
    seen = set()
    for variant in piece["variants"]:
        vh = int(variant[:, 0].max()) + 1
        vw = int(variant[:, 1].max()) + 1
        for r0 in range(h - vh + 1):
            for c0 in range(w - vw + 1):
                cells = tuple(sorted((int(r + r0), int(c + c0)) for r, c in variant))
                if cells in seen:
                    continue
                if all(cell in hole_set for cell in cells):
                    seen.add(cells)
                    for cell in cells:
                        placements_by_anchor.setdefault(cell, []).append(cells)
    return placements_by_anchor


def _tile_with_leftover(pieces: list[dict], hole_set: set[Coord], patch_shape: tuple[int, int], leftover_idx: int) -> list[dict[int, tuple[Coord, ...]]]:
    active = [idx for idx in range(len(pieces)) if idx != leftover_idx]
    if sum(pieces[idx]["area"] for idx in active) != len(hole_set):
        return []

    placement_maps = {}
    for idx in active:
        placements = _placements_for_piece(pieces[idx], hole_set, patch_shape)
        if not placements:
            return []
        placement_maps[idx] = placements

    solutions: list[dict[int, tuple[Coord, ...]]] = []

    def search(uncovered: set[Coord], unused: tuple[int, ...], chosen: dict[int, tuple[Coord, ...]]) -> None:
        if len(solutions) >= 2:
            return
        if not uncovered:
            if not unused:
                solutions.append(chosen.copy())
            return
        if sum(pieces[idx]["area"] for idx in unused) != len(uncovered):
            return

        anchor = min(uncovered)
        candidates = []
        for idx in unused:
            for cells in placement_maps[idx].get(anchor, []):
                cell_set = set(cells)
                if cell_set <= uncovered:
                    candidates.append((idx, cells))
        if not candidates:
            return

        candidates.sort(key=lambda item: (len(item[1]), item[0], item[1]))
        for idx, cells in candidates:
            chosen[idx] = cells
            next_unused = tuple(j for j in unused if j != idx)
            search(uncovered - set(cells), next_unused, chosen)
            del chosen[idx]
            if len(solutions) >= 2:
                return

    search(set(hole_set), tuple(active), {})
    return solutions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the frame hole with all but one outside fragment, and erase the unique leftover piece.

    Input structure:
    - A single monochrome thick frame sits near the center, and its interior contains one irregular background hole.
    - Outside the frame are separated monochrome polyomino fragments in distinct colors.
    - Exactly one fragment is a leftover; the others can be rotated or mirrored to tile the hole exactly.

    Rule:
    - Identify the central hollow frame and the background cells of its hole.
    - Among the outside fragments, discard the unique piece whose removal allows the remaining fragments to tile the hole with no overlap or gaps.
    - Move those remaining fragments into the hole using the unique valid packing, preserve their colors, erase every outside fragment, and keep the frame unchanged.

    Key insight:
    - Area counts alone do not identify the removable piece; only a geometric exact-cover search over rotated and mirrored fragments reveals both the leftover and the final hole filling.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    frame = _find_frame(g, bg)
    pieces = _extract_pieces(g, frame["coords"], bg)

    r0, c0, r1, c1 = frame["bbox"]
    patch_shape = (r1 - r0 + 1, c1 - c0 + 1)
    hole_set = {tuple(map(int, cell)) for cell in frame["hole_local"]}

    all_solutions = []
    for leftover_idx in range(len(pieces)):
        solutions = _tile_with_leftover(pieces, hole_set, patch_shape, leftover_idx)
        for solution in solutions:
            all_solutions.append((leftover_idx, solution))
            if len(all_solutions) >= 2:
                break
        if len(all_solutions) >= 2:
            break

    assert len(all_solutions) == 1, "expected a unique leftover fragment and a unique packing"
    _leftover_idx, solution = all_solutions[0]

    out = np.full_like(g, int(bg))
    out[frame["coords"][:, 0], frame["coords"][:, 1]] = frame["color"]
    for idx, cells in solution.items():
        color = pieces[idx]["color"]
        for rr, cc in cells:
            out[r0 + rr, c0 + cc] = int(color)
    return out


_GEN_BG_OPTIONS = [0, 1, 8]
_GEN_FRAME_OPTIONS = [2, 5, 7, 9]
_GEN_SIZE_TEMPLATES = [
    (3, 3, 4),
    (4, 4, 3),
    (4, 4, 3, 3),
    (4, 4, 2, 3),
    (5, 5, 3),
    (5, 5, 4),
]
_GEN_THICKNESS = [2, 3]
_GEN_MAX_ATTEMPTS = 400


def _neighbors4(cell: Coord) -> list[Coord]:
    r, c = cell
    return [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]


def _is_connected(cells: set[Coord]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        cell = stack.pop()
        for nb in _neighbors4(cell):
            if nb in cells and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(cells)


def _random_polyomino(size: int) -> np.ndarray:
    for _ in range(120):
        cells = {(0, 0)}
        while len(cells) < size:
            frontier = []
            for cell in sorted(cells):
                for nb in _neighbors4(cell):
                    if nb not in cells:
                        frontier.append(nb)
            cells.add(random.choice(frontier))
        arr = _normalize(np.asarray(sorted(cells), dtype=int))
        if size >= 4:
            r0, c0, r1, c1 = _bbox(arr)
            if (r1 - r0 + 1) * (c1 - c0 + 1) == size and random.random() < 0.8:
                continue
        return arr
    raise RuntimeError("failed to sample a polyomino")


def _pack_shapes(shapes: list[np.ndarray]) -> list[np.ndarray] | None:
    ordered = sorted(shapes, key=lambda shape: shape.shape[0], reverse=True)
    placed = [ordered[0].copy()]
    occupied = {tuple(map(int, cell)) for cell in ordered[0]}
    for shape in ordered[1:]:
        candidates = []
        occ_rows = [r for r, _ in occupied]
        occ_cols = [c for _, c in occupied]
        for orient in _all_orientations(shape):
            oh = int(orient[:, 0].max()) + 1
            ow = int(orient[:, 1].max()) + 1
            for r0 in range(min(occ_rows) - oh - 1, max(occ_rows) + 3):
                for c0 in range(min(occ_cols) - ow - 1, max(occ_cols) + 3):
                    shifted = {(int(r + r0), int(c + c0)) for r, c in orient}
                    if shifted & occupied:
                        continue
                    if not any(nb in occupied for cell in shifted for nb in _neighbors4(cell)):
                        continue
                    union = occupied | shifted
                    union_arr = np.asarray(sorted(union), dtype=int)
                    rr0, cc0, rr1, cc1 = _bbox(union_arr)
                    if rr1 - rr0 + 1 > 8 or cc1 - cc0 + 1 > 8:
                        continue
                    candidates.append(np.asarray(sorted(shifted), dtype=int))
        if not candidates:
            return None
        choice = random.choice(candidates)
        placed.append(choice)
        occupied |= {tuple(map(int, cell)) for cell in choice}

    union = np.vstack(placed)
    offset = union.min(axis=0)
    normalized = [piece - offset for piece in placed]
    union_norm = np.vstack(normalized)
    r0, c0, r1, c1 = _bbox(union_norm)
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return None
    bbox_area = (r1 - r0 + 1) * (c1 - c0 + 1)
    if bbox_area == union_norm.shape[0]:
        return None
    if not _is_connected({tuple(map(int, cell)) for cell in union_norm}):
        return None
    return normalized


def _sample_piece_family() -> tuple[list[np.ndarray], np.ndarray] | None:
    sizes = list(random.choice(_GEN_SIZE_TEMPLATES))
    repeat_sizes = [size for size in set(sizes) if sizes.count(size) >= 2]
    if not repeat_sizes:
        return None

    shapes = []
    used_keys = set()
    for size in sizes:
        for _ in range(80):
            shape = _random_polyomino(size)
            key = _canonical_key(shape)
            if key in used_keys:
                continue
            shapes.append(shape)
            used_keys.add(key)
            break
        else:
            return None

    packed = _pack_shapes(shapes)
    if packed is None:
        return None

    target_size = random.choice(repeat_sizes)
    sibling_keys = {_canonical_key(shape) for shape in shapes if shape.shape[0] == target_size}
    leftover = None
    for _ in range(120):
        candidate = _random_polyomino(target_size)
        if _canonical_key(candidate) not in sibling_keys:
            leftover = candidate
            break
    if leftover is None:
        return None

    return packed, leftover


def _build_frame_patch(
    hole_pieces: list[np.ndarray], thickness: int, bg: int, frame_color: int
) -> tuple[np.ndarray, list[np.ndarray], tuple[int, int]]:
    union = np.vstack(hole_pieces)
    hole_h = int(union[:, 0].max()) + 1
    hole_w = int(union[:, 1].max()) + 1
    patch = np.full((hole_h + 2 * thickness, hole_w + 2 * thickness), int(frame_color), dtype=int)
    for piece in hole_pieces:
        for r, c in piece:
            patch[int(r) + thickness, int(c) + thickness] = int(bg)
    shifted = [piece + np.array([thickness, thickness], dtype=int) for piece in hole_pieces]
    return patch, shifted, (hole_h, hole_w)


def _can_place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int, padding: int = 1) -> bool:
    h, w = patch.shape
    H, W = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    fg = np.argwhere(patch != int(bg))
    for dr, dc in fg:
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        r1 = max(0, rr - padding)
        r2 = min(H, rr + padding + 1)
        c1 = max(0, cc - padding)
        c2 = min(W, cc + padding + 1)
        if np.any(grid[r1:r2, c1:c2] != int(bg)):
            return False
    return True


def _scatter_fragments(
    grid: np.ndarray, fragment_patches: list[np.ndarray], bg: int, frame_bbox: tuple[int, int, int, int]
) -> bool:
    H, W = grid.shape
    fr0, fc0, fr1, fc1 = frame_bbox
    expanded = (max(0, fr0 - 1), max(0, fc0 - 1), min(H - 1, fr1 + 1), min(W - 1, fc1 + 1))
    order = sorted(range(len(fragment_patches)), key=lambda idx: int(np.count_nonzero(fragment_patches[idx] != bg)), reverse=True)
    for idx in order:
        patch = fragment_patches[idx]
        h, w = patch.shape
        spots = []
        for r0 in range(H - h + 1):
            for c0 in range(W - w + 1):
                if not (r0 + h - 1 < expanded[0] or r0 > expanded[2] or c0 + w - 1 < expanded[1] or c0 > expanded[3]):
                    continue
                if _can_place_patch(grid, patch, r0, c0, bg, padding=1):
                    spots.append((r0, c0))
        if not spots:
            return False
        r0, c0 = random.choice(spots)
        _paste(grid, patch, r0, c0)
    return True


def _make_expected(
    grid_shape: tuple[int, int],
    frame_patch: np.ndarray,
    frame_pos: tuple[int, int],
    hole_pieces: list[np.ndarray],
    piece_colors: list[int],
    bg: int,
) -> np.ndarray:
    out = np.full(grid_shape, int(bg), dtype=int)
    fr, fc = frame_pos
    _paste(out, frame_patch, fr, fc)
    for coords, color in zip(hole_pieces, piece_colors):
        for r, c in coords:
            out[fr + int(r), fc + int(c)] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    family = _sample_piece_family()
    if family is None:
        return None
    hole_pieces, leftover = family

    thickness = random.choice(_GEN_THICKNESS)
    bg = random.choice(_GEN_BG_OPTIONS)
    frame_color_choices = [color for color in _GEN_FRAME_OPTIONS if color != bg]
    frame_color = random.choice(frame_color_choices)

    total_fragments = len(hole_pieces) + 1
    palette = [color for color in range(10) if color not in {bg, frame_color}]
    random.shuffle(palette)
    piece_colors = palette[: len(hole_pieces)]
    leftover_color = palette[len(hole_pieces)]

    frame_patch, shifted_hole_pieces, _hole_shape = _build_frame_patch(hole_pieces, thickness, bg, frame_color)
    fh, fw = frame_patch.shape

    H = min(30, max(18, fh + 8) + random.randint(0, 4))
    W = min(30, max(18, fw + 10) + random.randint(0, 4))
    if H <= fh + 2 or W <= fw + 2:
        return None

    grid = np.full((H, W), int(bg), dtype=int)
    frame_r = (H - fh) // 2 + random.randint(-1, 1)
    frame_c = (W - fw) // 2 + random.randint(-2, 2)
    frame_r = max(1, min(H - fh - 1, frame_r))
    frame_c = max(1, min(W - fw - 1, frame_c))
    _paste(grid, frame_patch, frame_r, frame_c)

    fragment_patches = []
    for coords, color in zip(hole_pieces, piece_colors):
        orient = random.choice(_all_orientations(coords))
        fragment_patches.append(_piece_patch(orient, color, bg))
    orient = random.choice(_all_orientations(leftover))
    fragment_patches.append(_piece_patch(orient, leftover_color, bg))

    frame_bbox = (frame_r, frame_c, frame_r + fh - 1, frame_c + fw - 1)
    if not _scatter_fragments(grid, fragment_patches, bg, frame_bbox):
        return None

    expected = _make_expected(grid.shape, frame_patch, (frame_r, frame_c), shifted_hole_pieces, piece_colors, bg)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if max(grid.shape) > 30 or max(actual.shape) > 30:
        return None
    if total_fragments < 4:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
