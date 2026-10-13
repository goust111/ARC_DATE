from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, components_mask, crop_bbox, find_components, get_bg


_RNG = np.random.default_rng()

_BG = Color.BLACK
_FRAME_COLOR = Color.BLUE
_SEPARATOR_COLOR = Color.GREY
_PIECE_COLORS = [Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
_GEN_ATTEMPTS = 400
_HOLE_TEMPLATES = [(7, 2, 5), (9, 3, 6)]
_DISTRACTOR_AREA = 10


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), mask.astype(np.uint8).tobytes()


def _all_transforms(mask: np.ndarray) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for flipped in (False, True):
        base = np.fliplr(mask) if flipped else mask
        for k in range(4):
            transformed = _trim_mask(np.rot90(base, k))
            key = _mask_key(transformed)
            if key not in seen:
                seen.add(key)
                masks.append(transformed)
    return masks


def _apply_symmetry(mask: np.ndarray, sym: int) -> np.ndarray:
    arr = np.fliplr(mask) if sym >= 4 else mask
    return np.rot90(arr, sym % 4)


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _is_connected_cells(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    stack = [next(iter(cells))]
    seen = set()
    while stack:
        cell = stack.pop()
        if cell in seen:
            continue
        seen.add(cell)
        for nbr in _neighbors4(cell):
            if nbr in cells and nbr not in seen:
                stack.append(nbr)
    return len(seen) == len(cells)


def _mask_from_cells(cells: set[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _grow_connected_subset(
    all_cells: set[tuple[int, int]],
    target_size: int,
    rng: np.random.Generator,
) -> set[tuple[int, int]] | None:
    cell_list = list(all_cells)
    subset = {cell_list[int(rng.integers(len(cell_list)))]}
    while len(subset) < target_size:
        frontier = []
        for cell in subset:
            for nbr in _neighbors4(cell):
                if nbr in all_cells and nbr not in subset and nbr not in frontier:
                    frontier.append(nbr)
        if not frontier:
            return None
        subset.add(frontier[int(rng.integers(len(frontier)))])
    return subset


def _random_connected_mask(rng: np.random.Generator, area: int) -> np.ndarray | None:
    for _ in range(200):
        cells = {(3, 3)}
        while len(cells) < area:
            frontier = []
            for cell in cells:
                for nbr in _neighbors4(cell):
                    r, c = nbr
                    if 1 <= r <= 5 and 1 <= c <= 5 and nbr not in cells and nbr not in frontier:
                        frontier.append(nbr)
            if not frontier:
                break
            cells.add(frontier[int(rng.integers(len(frontier)))])
        if len(cells) != area:
            continue
        mask = _trim_mask(_mask_from_cells(cells, (7, 7)))
        h, w = mask.shape
        if h < 2 or w < 2 or h > 5 or w > 5:
            continue
        if area == h * w and rng.random() < 0.75:
            continue
        return mask
    return None


def _partition_mask(
    mask: np.ndarray,
    rng: np.random.Generator,
    target_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    cells = {tuple(cell) for cell in np.argwhere(mask)}
    area = len(cells)
    for _ in range(300):
        if target_size is None:
            target = int(rng.integers(2, area - 1))
        else:
            target = int(target_size)
        if target < 2 or target > area - 2:
            continue
        subset = _grow_connected_subset(cells, target, rng)
        if subset is None:
            continue
        other = cells - subset
        if len(other) < 2 or not _is_connected_cells(other):
            continue
        return _mask_from_cells(subset, mask.shape), _mask_from_cells(other, mask.shape)
    return None


def _find_separator_col(grid: np.ndarray, bg: int) -> int:
    candidates = []
    for c in range(grid.shape[1]):
        if int(grid[0, c]) == bg:
            continue
        if not np.all(grid[:, c] == grid[0, c]):
            continue
        if c == 0 or c == grid.shape[1] - 1:
            continue
        if not np.any(grid[:, :c] != bg):
            continue
        if not np.any(grid[:, c + 1 :] != bg):
            continue
        candidates.append(c)
    assert len(candidates) == 1
    return candidates[0]


def _extract_holes(panel: np.ndarray, bg: int) -> list[dict]:
    h, w = panel.shape
    holes = []
    for coords in components_mask(panel == bg, connectivity=4):
        if (
            np.any(coords[:, 0] == 0)
            or np.any(coords[:, 0] == h - 1)
            or np.any(coords[:, 1] == 0)
            or np.any(coords[:, 1] == w - 1)
        ):
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        rel = coords - np.array([[r0, c0]])
        mask[rel[:, 0], rel[:, 1]] = True
        holes.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "mask": mask,
                "area": int(coords.shape[0]),
            }
        )
    holes.sort(key=lambda hole: (hole["bbox"][0], hole["bbox"][1]))
    return holes


def _extract_pieces(panel: np.ndarray, bg: int) -> list[dict]:
    pieces = []
    for comp in find_components(panel, bg=bg, connectivity=4, monochromatic=True):
        color = int(comp["color"])
        patch = crop_bbox(panel, comp["bbox"])
        mask = patch == color
        pieces.append({"color": color, "mask": _trim_mask(mask), "area": int(mask.sum())})
    pieces.sort(key=lambda piece: piece["color"])
    colors = [piece["color"] for piece in pieces]
    assert len(colors) == len(set(colors))
    return pieces


def _piece_placements(hole_mask: np.ndarray, piece_mask: np.ndarray) -> list[np.ndarray]:
    hole_h, hole_w = hole_mask.shape
    placed_masks: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
    for transformed in _all_transforms(piece_mask):
        ph, pw = transformed.shape
        if ph > hole_h or pw > hole_w:
            continue
        coords = np.argwhere(transformed)
        for r0 in range(hole_h - ph + 1):
            for c0 in range(hole_w - pw + 1):
                shifted = coords + np.array([[r0, c0]])
                if not np.all(hole_mask[shifted[:, 0], shifted[:, 1]]):
                    continue
                placed = np.zeros_like(hole_mask, dtype=bool)
                placed[shifted[:, 0], shifted[:, 1]] = True
                placed_masks[_mask_key(placed)] = placed
    return list(placed_masks.values())


def _hole_candidates(hole: dict, pieces: list[dict]) -> list[dict]:
    hole_mask = hole["mask"]
    hole_area = int(hole["area"])
    placements = [_piece_placements(hole_mask, piece["mask"]) for piece in pieces]
    candidates = []
    seen: set[tuple[int, int, tuple[tuple[int, int], bytes], tuple[tuple[int, int], bytes]]] = set()
    for i in range(len(pieces)):
        for j in range(i + 1, len(pieces)):
            if pieces[i]["area"] + pieces[j]["area"] != hole_area:
                continue
            placements_j = {_mask_key(mask): mask for mask in placements[j]}
            for placed_i in placements[i]:
                key_i = _mask_key(placed_i)
                remaining = hole_mask & ~placed_i
                if int(remaining.sum()) != pieces[j]["area"]:
                    continue
                key_rem = _mask_key(remaining)
                if key_rem in placements_j:
                    sig = (i, j, key_i, key_rem)
                    if sig in seen:
                        continue
                    seen.add(sig)
                    candidates.append(
                        {
                            "pieces": (i, j),
                            "placements": (placed_i, placements_j[key_rem]),
                        }
                    )
    return candidates


def _search_assignments(
    hole_candidates: list[list[dict]],
    used_pieces: set[int],
    partial: dict[int, dict],
    solutions: list[dict[int, dict]],
    limit: int = 2,
) -> None:
    if len(partial) == len(hole_candidates):
        solutions.append(dict(partial))
        return
    remaining = [idx for idx in range(len(hole_candidates)) if idx not in partial]
    best_hole = -1
    feasible: list[dict] | None = None
    for hole_idx in remaining:
        options = [
            cand
            for cand in hole_candidates[hole_idx]
            if cand["pieces"][0] not in used_pieces and cand["pieces"][1] not in used_pieces
        ]
        if not options:
            return
        if feasible is None or len(options) < len(feasible):
            best_hole = hole_idx
            feasible = options
    assert feasible is not None
    for cand in feasible:
        piece_a, piece_b = cand["pieces"]
        partial[best_hole] = cand
        used_pieces.add(piece_a)
        used_pieces.add(piece_b)
        _search_assignments(hole_candidates, used_pieces, partial, solutions, limit=limit)
        used_pieces.remove(piece_a)
        used_pieces.remove(piece_b)
        del partial[best_hole]
        if len(solutions) >= limit:
            return


def _solve_panel(left_panel: np.ndarray, right_panel: np.ndarray, bg: int) -> np.ndarray:
    holes = _extract_holes(left_panel, bg)
    pieces = _extract_pieces(right_panel, bg)
    assert holes
    assert pieces

    hole_candidates = []
    for hole in holes:
        candidates = _hole_candidates(hole, pieces)
        assert candidates
        hole_candidates.append(candidates)

    solutions: list[dict[int, dict]] = []
    _search_assignments(hole_candidates, set(), {}, solutions, limit=2)
    assert len(solutions) == 1
    assignment = solutions[0]

    out = left_panel.copy()
    for hole_idx, cand in assignment.items():
        r0, c0, _r1, _c1 = holes[hole_idx]["bbox"]
        for piece_idx, placed in zip(cand["pieces"], cand["placements"]):
            color = pieces[piece_idx]["color"]
            coords = np.argwhere(placed)
            out[r0 + coords[:, 0], c0 + coords[:, 1]] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each enclosed hole in the left panel by combining exactly two transformed pieces from the right panel.

    Input structure:
    - One solid separator column splits the grid into a left object panel and a right piece panel.
    - The left panel contains several solid objects, each enclosing one 4-connected background hole.
    - The right panel contains several single-color connected pieces with distinct colors, including distractors.

    Rule:
    - For each enclosed hole, search the right-panel pieces for the unique pair whose rotated/flipped placements cover that hole exactly with no overlap.
    - Color every hole cell by the color of the piece that lands on it.
    - Ignore unused distractor pieces and return only the completed left panel.

    Key insight:
    - No hole matches any single piece; the solution is a global two-piece assembly problem that must determine pairing, orientation, and placement simultaneously.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    sep = _find_separator_col(g, bg)
    left = g[:, :sep]
    right = g[:, sep + 1 :]
    return _solve_panel(left, right, bg)


def _build_object_patch(
    hole_mask: np.ndarray,
    piece_a_full: np.ndarray,
    piece_b_full: np.ndarray,
    color_a: int,
    color_b: int,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = hole_mask.shape
    obj = np.full((h + 2, w + 2), _FRAME_COLOR, dtype=int)
    obj_inner = obj[1 : h + 1, 1 : w + 1]
    obj_inner[hole_mask] = _BG
    filled = obj.copy()
    filled_inner = filled[1 : h + 1, 1 : w + 1]
    filled_inner[piece_a_full] = color_a
    filled_inner[piece_b_full] = color_b
    return obj, filled


def _fits_patch(canvas: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int, pad: int = 1) -> bool:
    h, w = patch.shape
    if top < 0 or left < 0 or top + h > canvas.shape[0] or left + w > canvas.shape[1]:
        return False
    occ = patch != bg
    rr, cc = np.nonzero(occ)
    for r, c in zip(rr, cc, strict=True):
        ar = top + int(r)
        ac = left + int(c)
        r0 = max(0, ar - pad)
        r1 = min(canvas.shape[0], ar + pad + 1)
        c0 = max(0, ac - pad)
        c1 = min(canvas.shape[1], ac + pad + 1)
        if np.any(canvas[r0:r1, c0:c1] != bg):
            return False
    return True


def _place_patch_random(
    canvas: np.ndarray,
    patch: np.ndarray,
    rng: np.random.Generator,
    bg: int,
    pad: int = 1,
) -> tuple[int, int] | None:
    h, w = patch.shape
    positions = [(r, c) for r in range(canvas.shape[0] - h + 1) for c in range(canvas.shape[1] - w + 1)]
    rng.shuffle(positions)
    for top, left in positions:
        if not _fits_patch(canvas, patch, top, left, bg, pad=pad):
            continue
        sl = canvas[top : top + h, left : left + w]
        sl[patch != bg] = patch[patch != bg]
        return top, left
    return None


def _make_piece_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, _BG, dtype=int)
    patch[mask] = color
    return patch


def _normalize_triplet(
    hole_mask: np.ndarray,
    piece_a_full: np.ndarray,
    piece_b_full: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.argwhere(hole_mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return (
        hole_mask[r0 : r1 + 1, c0 : c1 + 1].copy(),
        piece_a_full[r0 : r1 + 1, c0 : c1 + 1].copy(),
        piece_b_full[r0 : r1 + 1, c0 : c1 + 1].copy(),
    )


def _single_piece_can_fill(hole_mask: np.ndarray, piece_masks: list[np.ndarray]) -> bool:
    for piece_mask in piece_masks:
        if int(piece_mask.sum()) != int(hole_mask.sum()):
            continue
        if _piece_placements(hole_mask, piece_mask):
            return True
    return False


def _generate_case(rng: np.random.Generator | None = None) -> dict | None:
    rng = _RNG if rng is None else rng
    templates = list(_HOLE_TEMPLATES)
    rng.shuffle(templates)
    hole_count = len(templates)
    distractor_count = 1
    piece_colors = list(rng.permutation(_PIECE_COLORS))
    needed_piece_count = hole_count * 2 + distractor_count
    if needed_piece_count > len(piece_colors):
        return None

    left_h = int(rng.integers(13, 19))
    left_w = int(rng.integers(11, 15))
    right_w = int(rng.integers(11, 15))
    total_w = left_w + 1 + right_w
    if total_w > 30:
        return None

    left_panel = np.full((left_h, left_w), _BG, dtype=int)
    expected = left_panel.copy()
    right_panel = np.full((left_h, right_w), _BG, dtype=int)

    hole_specs = []
    piece_specs = []
    color_idx = 0

    for hole_idx, (hole_area, first_piece_area, _second_piece_area) in enumerate(templates):
        hole_mask = _random_connected_mask(rng, hole_area)
        if hole_mask is None:
            return None
        partition = _partition_mask(hole_mask, rng, target_size=first_piece_area)
        if partition is None:
            return None
        piece_a_full, piece_b_full = partition
        sym = int(rng.integers(8))
        hole_mask = _apply_symmetry(hole_mask, sym)
        piece_a_full = _apply_symmetry(piece_a_full, sym)
        piece_b_full = _apply_symmetry(piece_b_full, sym)
        hole_mask, piece_a_full, piece_b_full = _normalize_triplet(hole_mask, piece_a_full, piece_b_full)

        color_a = int(piece_colors[color_idx])
        color_b = int(piece_colors[color_idx + 1])
        color_idx += 2

        obj_patch, out_patch = _build_object_patch(hole_mask, piece_a_full, piece_b_full, color_a, color_b)
        placed = _place_patch_random(left_panel, obj_patch, rng, _BG, pad=1)
        if placed is None:
            return None
        top, left = placed
        expected[top : top + out_patch.shape[0], left : left + out_patch.shape[1]] = np.where(
            out_patch != _BG,
            out_patch,
            expected[top : top + out_patch.shape[0], left : left + out_patch.shape[1]],
        )

        piece_a_mask = _trim_mask(piece_a_full)
        piece_b_mask = _trim_mask(piece_b_full)
        hole_specs.append(
            {
                "hole_area": int(hole_mask.sum()),
                "colors": {color_a, color_b},
                "piece_masks": [piece_a_mask, piece_b_mask],
                "hole_mask": hole_mask.copy(),
                "piece_colors": (color_a, color_b),
                "index": hole_idx,
            }
        )
        piece_specs.append({"color": color_a, "mask": piece_a_mask, "kind": f"hole{hole_idx}"})
        piece_specs.append({"color": color_b, "mask": piece_b_mask, "kind": f"hole{hole_idx}"})

    while len(piece_specs) < needed_piece_count:
        distractor = _random_connected_mask(rng, _DISTRACTOR_AREA)
        if distractor is None:
            return None
        piece_specs.append(
            {
                "color": int(piece_colors[color_idx]),
                "mask": distractor,
                "kind": "distractor",
            }
        )
        color_idx += 1

    rng.shuffle(piece_specs)
    for spec in piece_specs:
        transforms = _all_transforms(spec["mask"])
        transformed = transforms[int(rng.integers(len(transforms)))].copy()
        patch = _make_piece_patch(transformed, spec["color"])
        if _place_patch_random(right_panel, patch, rng, _BG, pad=1) is None:
            return None

    inp = np.concatenate(
        [
            left_panel,
            np.full((left_h, 1), _SEPARATOR_COLOR, dtype=int),
            right_panel,
        ],
        axis=1,
    )
    return {
        "input": inp,
        "expected": expected,
        "hole_specs": hole_specs,
        "piece_specs": piece_specs,
    }


def _validate_case(case: dict) -> bool:
    inp = case["input"]
    expected = case["expected"]
    try:
        solved = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(solved, expected):
        return False

    bg = int(get_bg(inp))
    sep = _find_separator_col(inp, bg)
    left = inp[:, :sep]
    right = inp[:, sep + 1 :]
    holes = _extract_holes(left, bg)
    pieces = _extract_pieces(right, bg)
    if len(holes) != len(case["hole_specs"]):
        return False
    if any(_single_piece_can_fill(hole["mask"], [piece["mask"] for piece in pieces]) for hole in holes):
        return False

    hole_candidates = [_hole_candidates(hole, pieces) for hole in holes]
    if any(not candidates for candidates in hole_candidates):
        return False
    solutions: list[dict[int, dict]] = []
    try:
        _search_assignments(hole_candidates, set(), {}, solutions, limit=2)
    except AssertionError:
        return False
    if len(solutions) != 1:
        return False

    expected_pairs = sorted(tuple(sorted(spec["colors"])) for spec in case["hole_specs"])
    used_piece_colors = set()
    solved_pairs = []
    for hole_idx, cand in solutions[0].items():
        colors = {pieces[cand["pieces"][0]]["color"], pieces[cand["pieces"][1]]["color"]}
        used_piece_colors |= colors
        solved_pairs.append(tuple(sorted(colors)))
    if sorted(solved_pairs) != expected_pairs:
        return False
    distractor_colors = {spec["color"] for spec in case["piece_specs"] if spec["kind"] == "distractor"}
    if used_piece_colors & distractor_colors:
        return False

    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if len(np.unique(inp)) < 5:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        case = _generate_case()
        if case is not None and _validate_case(case):
            return case["input"]
    raise RuntimeError("failed to generate a valid dual-piece hole puzzle")
