from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox, crop_bbox, find_components, paste, random_free_location


BACKGROUND = Color.BLACK
ANCHOR = Color.GREY
VISIBLE_COLORS = np.array([Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON], dtype=int)
MAX_GENERATION_ATTEMPTS = 500
_SEEN_OUTPUT_SIGS: set[tuple[tuple[int, int], bytes]] = set()
_SEEN_INPUT_SIGS: set[tuple[tuple[int, int], bytes]] = set()


def _neighbors8(shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    out: list[tuple[int, int]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr = int(r + dr)
            nc = int(c + dc)
            if 0 <= nr < h and 0 <= nc < w:
                out.append((nr, nc))
    return out


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors8(mask.shape, r, c):
            if mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(coords.shape[0])


def _grid_signature(grid: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, grid.shape)), grid.tobytes()


def _relative_map(sprite: np.ndarray) -> tuple[dict[tuple[int, int], int], tuple[int, int]]:
    anchor_coords = np.argwhere(sprite == ANCHOR)
    assert anchor_coords.shape == (1, 2), "each sprite must contain exactly one gray anchor"
    ar, ac = map(int, anchor_coords[0])
    rel: dict[tuple[int, int], int] = {}
    for r, c in np.argwhere(sprite != BACKGROUND):
        rr = int(r) - ar
        cc = int(c) - ac
        rel[(rr, cc)] = int(sprite[int(r), int(c)])
    return rel, (ar, ac)


def _restore_from_relative_maps(relative_maps: list[dict[tuple[int, int], int]]) -> np.ndarray:
    votes: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    for rel in relative_maps:
        for pos, color in rel.items():
            votes[pos][int(color)] += 1

    assert votes, "expected at least one visible aligned cell"
    rows = [pos[0] for pos in votes]
    cols = [pos[1] for pos in votes]
    out = np.full((max(rows) - min(rows) + 1, max(cols) - min(cols) + 1), BACKGROUND, dtype=int)
    row_shift = -min(rows)
    col_shift = -min(cols)

    for (rr, cc), counter in votes.items():
        ordered = counter.most_common()
        assert ordered, "each voted cell must have at least one color"
        if len(ordered) > 1:
            assert ordered[0][1] > ordered[1][1], "aligned color vote must have a strict majority"
        out[rr + row_shift, cc + col_shift] = int(ordered[0][0])

    return out


def _aligned_relative_maps(grid: np.ndarray) -> list[dict[tuple[int, int], int]]:
    components = find_components(grid, bg=BACKGROUND, connectivity=8, monochromatic=False, return_format="objects")
    assert components, "expected at least one fragment"
    relative_maps: list[dict[tuple[int, int], int]] = []
    for obj in components:
        b = bbox(obj != BACKGROUND)
        assert b is not None
        sprite = crop_bbox(obj, b)
        rel, _ = _relative_map(sprite)
        relative_maps.append(rel)
    return relative_maps


def _random_connected_mask(rng: np.random.Generator) -> np.ndarray:
    for _ in range(80):
        height = int(rng.integers(4, 9))
        width = int(rng.integers(4, 9))
        max_area = min(height * width - 1, 18)
        min_area = max(9, (height * width) // 2)
        if min_area > max_area:
            continue
        target_area = int(rng.integers(min_area, max_area + 1))

        start = (int(rng.integers(height)), int(rng.integers(width)))
        cells = {start}
        frontier = {nbr for nbr in _neighbors8((height, width), *start) if nbr not in cells}
        while len(cells) < target_area and frontier:
            frontier_list = list(frontier)
            choice = frontier_list[int(rng.integers(len(frontier_list)))]
            frontier.remove(choice)
            cells.add(choice)
            for nbr in _neighbors8((height, width), *choice):
                if nbr not in cells:
                    frontier.add(nbr)

        mask = np.zeros((height, width), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        b = bbox(mask)
        assert b is not None
        mask = crop_bbox(mask.astype(int), b).astype(bool)
        if not _mask_connected(mask):
            continue
        if min(mask.shape) < 3:
            continue
        if int(mask.sum()) < 9:
            continue
        if int(mask.sum()) == mask.size:
            continue
        return mask
    raise RuntimeError("failed to generate a connected hidden mask")


def _build_truth(rng: np.random.Generator) -> np.ndarray:
    mask = _random_connected_mask(rng)
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    area = len(coords)
    anchor = coords[int(rng.integers(area))]

    palette_size = int(rng.integers(2, min(5, area)))
    palette = [int(c) for c in rng.choice(VISIBLE_COLORS, size=palette_size, replace=False)]
    truth = np.full(mask.shape, BACKGROUND, dtype=int)
    truth[anchor] = ANCHOR

    non_anchor = [coord for coord in coords if coord != anchor]
    forced = palette.copy()
    rng.shuffle(forced)
    assignments = forced + [int(rng.choice(palette)) for _ in range(len(non_anchor) - len(forced))]
    rng.shuffle(assignments)
    for (r, c), color in zip(non_anchor, assignments):
        truth[r, c] = int(color)
    return truth


def _sample_copy_mask(mask: np.ndarray, anchor: tuple[int, int], rng: np.random.Generator) -> np.ndarray | None:
    area = int(mask.sum())
    min_keep = max(4, int(np.ceil(area * 0.55)))
    target = int(rng.integers(min_keep, area))
    keep = mask.copy()
    candidates = [tuple(map(int, rc)) for rc in np.argwhere(mask) if tuple(map(int, rc)) != anchor]
    rng.shuffle(candidates)
    for r, c in candidates:
        if int(keep.sum()) <= target:
            break
        keep[r, c] = False
        if not _mask_connected(keep):
            keep[r, c] = True
    if int(keep.sum()) >= area or int(keep.sum()) < min_keep or not _mask_connected(keep):
        return None
    return keep


def _assign_unique_cells(
    candidates_per_copy: list[list[tuple[int, int]]], rng: np.random.Generator
) -> list[tuple[int, int]] | None:
    order = sorted(range(len(candidates_per_copy)), key=lambda i: len(candidates_per_copy[i]))
    assigned: list[tuple[int, int] | None] = [None] * len(candidates_per_copy)
    used: set[tuple[int, int]] = set()

    def dfs(depth: int) -> bool:
        if depth == len(order):
            return True
        copy_idx = order[depth]
        choices = candidates_per_copy[copy_idx][:]
        rng.shuffle(choices)
        for cell in choices:
            if cell in used:
                continue
            used.add(cell)
            assigned[copy_idx] = cell
            if dfs(depth + 1):
                return True
            assigned[copy_idx] = None
            used.remove(cell)
        return False

    if not dfs(0):
        return None
    return [cell for cell in assigned if cell is not None]


def _build_copy_set(
    truth: np.ndarray, rng: np.random.Generator
) -> tuple[list[np.ndarray], list[dict[tuple[int, int], int]], dict[tuple[int, int], Counter[int]]] | None:
    mask = truth != BACKGROUND
    anchor_coords = np.argwhere(truth == ANCHOR)
    assert anchor_coords.shape == (1, 2)
    anchor = tuple(map(int, anchor_coords[0]))
    truth_rel, _ = _relative_map(truth)
    truth_cells_abs = [
        tuple(map(int, rc))
        for rc in np.argwhere(truth != BACKGROUND)
        if int(truth[tuple(map(int, rc))]) != ANCHOR
    ]
    n_copies = int(rng.integers(3, 6))

    for _ in range(120):
        keep_masks: list[np.ndarray] = []
        for _copy_idx in range(n_copies):
            copy_mask = _sample_copy_mask(mask, anchor, rng)
            if copy_mask is None:
                break
            keep_masks.append(copy_mask)
        if len(keep_masks) != n_copies:
            continue

        coverage = sum(keep_mask.astype(int) for keep_mask in keep_masks)
        if np.any(coverage[mask] < 2):
            continue

        corruptible = [pos for pos in truth_cells_abs if coverage[pos[0], pos[1]] >= 3]
        if len(corruptible) < n_copies:
            continue

        candidates_per_copy = [
            [pos for pos in corruptible if keep_masks[i][pos[0], pos[1]]]
            for i in range(n_copies)
        ]
        if any(not candidates for candidates in candidates_per_copy):
            continue

        base_corruptions = _assign_unique_cells(candidates_per_copy, rng)
        if base_corruptions is None:
            continue

        copies: list[np.ndarray] = []
        relative_maps: list[dict[tuple[int, int], int]] = []
        used_cells = set(base_corruptions)
        for idx in range(n_copies):
            copy_grid = truth.copy()
            copy_grid[~keep_masks[idx]] = BACKGROUND
            corruption_cells = [base_corruptions[idx]]
            extra_candidates = [
                pos
                for pos in corruptible
                if pos not in used_cells and keep_masks[idx][pos[0], pos[1]]
            ]
            rng.shuffle(extra_candidates)
            extra_limit = int(rng.integers(0, 3))
            for pos in extra_candidates[:extra_limit]:
                if rng.random() < 0.45:
                    corruption_cells.append(pos)
                    used_cells.add(pos)

            for r, c in corruption_cells:
                true_color = int(truth[r, c])
                wrong_choices = [color for color in VISIBLE_COLORS.tolist() if color != true_color]
                copy_grid[r, c] = int(rng.choice(wrong_choices))

            copies.append(copy_grid)
            relative_maps.append(_relative_map(copy_grid)[0])

        votes: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
        for rel in relative_maps:
            for pos, color in rel.items():
                votes[pos][int(color)] += 1

        contested = 0
        valid = True
        for pos, true_color in truth_rel.items():
            if pos not in votes:
                valid = False
                break
            cell_votes = votes[pos]
            if cell_votes[int(true_color)] <= max(
                (v for color, v in cell_votes.items() if color != int(true_color)),
                default=0,
            ):
                valid = False
                break
            if len(cell_votes) > 1:
                contested += 1
        if not valid or contested < n_copies:
            continue
        return copies, relative_maps, votes

    return None


def _place_copies(copies: list[np.ndarray], rng: np.random.Generator) -> np.ndarray | None:
    sprites = []
    for copy_grid in copies:
        b = bbox(copy_grid != BACKGROUND)
        assert b is not None
        sprites.append(crop_bbox(copy_grid, b))

    sprites.sort(key=lambda arr: int(np.count_nonzero(arr != BACKGROUND)), reverse=True)
    max_h = max(int(sprite.shape[0]) for sprite in sprites)
    max_w = max(int(sprite.shape[1]) for sprite in sprites)
    total_cells = sum(int(np.count_nonzero(sprite != BACKGROUND)) for sprite in sprites)

    min_side = max(max_h + 3, max_w + 3, int(np.ceil(np.sqrt(total_cells * 4))))
    for _ in range(40):
        height = int(rng.integers(min(min_side, 30), 31))
        width = int(rng.integers(min(min_side, 30), 31))
        grid = np.full((height, width), BACKGROUND, dtype=int)
        try:
            for sprite in sprites:
                r0, c0 = random_free_location(
                    grid,
                    sprite,
                    transparent=BACKGROUND,
                    border=1,
                    padding=2,
                    padding_connectivity=8,
                )
                paste(grid, sprite, r0, c0, transparent=BACKGROUND)
            return grid
        except ValueError:
            continue
    return None


def _simple_overlay(relative_maps: list[dict[tuple[int, int], int]], *, reverse: bool) -> np.ndarray:
    ordered = list(reversed(relative_maps)) if reverse else list(relative_maps)
    positions = set()
    for rel in ordered:
        positions.update(rel)
    rows = [pos[0] for pos in positions]
    cols = [pos[1] for pos in positions]
    out = np.full((max(rows) - min(rows) + 1, max(cols) - min(cols) + 1), BACKGROUND, dtype=int)
    row_shift = -min(rows)
    col_shift = -min(cols)
    for rel in ordered:
        for (rr, cc), color in rel.items():
            out[rr + row_shift, cc + col_shift] = int(color)
    return out


def _validate_instance(grid: np.ndarray, truth: np.ndarray) -> bool:
    if grid.ndim != 2 or truth.ndim != 2:
        return False
    if max(grid.shape) > 30 or max(truth.shape) > 30:
        return False

    try:
        restored = main(grid)
    except Exception:
        return False
    if not np.array_equal(restored, truth):
        return False

    truth_rel, _ = _relative_map(truth)
    relative_maps = _aligned_relative_maps(grid)
    if len(relative_maps) < 3:
        return False

    contested = 0
    for rel in relative_maps:
        if set(rel) - set(truth_rel):
            return False
        missing = [pos for pos in truth_rel if pos not in rel]
        wrong = [pos for pos, color in rel.items() if truth_rel[pos] != color]
        if not missing or not wrong:
            return False

    votes: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    for rel in relative_maps:
        for pos, color in rel.items():
            votes[pos][int(color)] += 1
    for pos, true_color in truth_rel.items():
        counter = votes.get(pos)
        if not counter:
            return False
        if counter[int(true_color)] <= max(
            (v for color, v in counter.items() if color != int(true_color)),
            default=0,
        ):
            return False
        if len(counter) > 1:
            contested += 1
    if contested < len(relative_maps):
        return False

    if np.array_equal(_simple_overlay(relative_maps, reverse=False), truth):
        return False
    if np.array_equal(_simple_overlay(relative_maps, reverse=True), truth):
        return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore a hidden pattern by aligning gray-anchored corrupted copies and taking per-cell majorities.

    Input structure:
    - The grid is a black canvas containing several separated multicolor fragments.
    - Each fragment has exactly one gray anchor pixel and otherwise shows a partial copy of the same hidden sprite.
    - Different fragments may omit different cells and may disagree on some visible colors.

    Rule:
    - Split the input into fragments, crop each fragment, and express every non-black cell as an offset from that fragment's gray anchor.
    - Overlay all fragments by aligning those anchors, and at each aligned offset choose the color with a strict majority among the fragments that cover it.
    - Return the restored sprite cropped to the minimal bounding box of the aligned union, keeping the anchor cell gray.

    Key insight:
    - The gray pixel is a registration mark rather than ordinary content; no single fragment is trustworthy, so the answer only appears after anchor alignment plus cellwise consensus voting.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    relative_maps = _aligned_relative_maps(g)
    return _restore_from_relative_maps(relative_maps)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rng = np.random.default_rng()
        truth = _build_truth(rng)
        built = _build_copy_set(truth, rng)
        if built is None:
            continue
        copies, relative_maps, _votes = built
        grid = _place_copies(copies, rng)
        if grid is None:
            continue
        if not _validate_instance(grid, truth):
            continue

        input_sig = _grid_signature(grid)
        output_sig = _grid_signature(truth)
        if input_sig in _SEEN_INPUT_SIGS or output_sig in _SEEN_OUTPUT_SIGS:
            continue
        _SEEN_INPUT_SIGS.add(input_sig)
        _SEEN_OUTPUT_SIGS.add(output_sig)

        if np.array_equal(_simple_overlay(relative_maps, reverse=False), truth):
            continue
        return grid
    raise RuntimeError("failed to generate a valid fragmented-consensus input")
