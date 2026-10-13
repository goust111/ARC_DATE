from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
RED = 2
GREEN = 3
GREY = 5

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # N, E, S, W
MIN_TURNS = 3
MAX_TURNS = 5
MIN_SEG = 3
MAX_SEG = 7
MIN_FINAL_SEG = 4
GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                if connectivity == 4:
                    nbrs = _neighbors4(rr, cc)
                else:
                    nbrs = (
                        (rr + dr, cc + dc)
                        for dr in (-1, 0, 1)
                        for dc in (-1, 0, 1)
                        if dr != 0 or dc != 0
                    )
                for nr, nc in nbrs:
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _rot_patch_cw(patch: np.ndarray, quarter_turns: int) -> np.ndarray:
    return np.rot90(patch, k=(-quarter_turns) % 4)


def _patch_cells(patch: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in np.argwhere(patch != BG)}


def _centered_patch_cells(patch: np.ndarray, center: tuple[int, int]) -> set[tuple[int, int]]:
    cr, cc = center
    h, w = patch.shape
    assert (h, w) == (3, 3)
    out = set()
    for pr, pc in np.argwhere(patch != BG):
        out.add((int(cr + pr - 1), int(cc + pc - 1)))
    return out


def _dir_after(dir_idx: int, turn_sign: int) -> int:
    return (dir_idx + turn_sign) % 4


def _steps_to_boundary(pos: tuple[int, int], dir_idx: int, shape: tuple[int, int]) -> int:
    r, c = pos
    h, w = shape
    if dir_idx == 0:
        return r
    if dir_idx == 1:
        return w - 1 - c
    if dir_idx == 2:
        return h - 1 - r
    return c


def _segment_cells(start: tuple[int, int], dir_idx: int, length: int) -> list[tuple[int, int]]:
    dr, dc = DIRS[dir_idx]
    r, c = start
    return [(r + dr * step, c + dc * step) for step in range(1, length + 1)]


def _build_shape(coords: set[tuple[int, int]]) -> np.ndarray:
    patch = np.zeros((3, 3), dtype=int)
    for r, c in coords:
        patch[r, c] = RED
    return patch


_BASE_TEMPLATE_PATCHES = [
    _build_shape({(0, 0), (1, 0), (2, 0), (0, 1), (0, 2)}),
    _build_shape({(0, 1), (1, 1), (2, 1), (2, 0), (2, 2)}),
    _build_shape({(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)}),
]


def _all_template_variants() -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen = set()
    for patch in _BASE_TEMPLATE_PATCHES:
        for q in range(4):
            rot = _rot_patch_cw(patch, q)
            key = tuple(rot.ravel().tolist())
            if key not in seen:
                seen.add(key)
                variants.append(rot.copy())
    return variants


TEMPLATE_VARIANTS = _all_template_variants()


def _find_start_bar(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], int]:
    comps = _components_mask(grid == GREEN, connectivity=4)
    candidates = []
    for comp in comps:
        if comp.shape[0] != 2:
            continue
        pts = [tuple(map(int, rc)) for rc in comp]
        edge = [p for p in pts if p[0] in (0, grid.shape[0] - 1) or p[1] in (0, grid.shape[1] - 1)]
        if len(edge) != 1:
            continue
        edge_pt = edge[0]
        inner_pt = pts[0] if pts[1] == edge_pt else pts[1]
        dr = inner_pt[0] - edge_pt[0]
        dc = inner_pt[1] - edge_pt[1]
        if (dr, dc) not in DIRS:
            continue
        candidates.append((edge_pt, inner_pt, DIRS.index((dr, dc))))
    assert len(candidates) == 1, "expected a unique 2-pixel edge-touching start bar"
    return candidates[0]


def _find_template_patch(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    comps = _components_mask(grid == RED, connectivity=4)
    candidates = []
    for comp in comps:
        area = int(comp.shape[0])
        if not 3 <= area <= 5:
            continue
        bbox = _bbox_coords(comp)
        patch = _crop_bbox((grid == RED).astype(int) * RED, bbox)
        if patch.shape != (3, 3):
            continue
        rotations = {tuple(_rot_patch_cw(patch, q).ravel().tolist()) for q in range(4)}
        if len(rotations) != 4:
            continue
        candidates.append(comp)
    assert len(candidates) == 1, "expected a unique non-symmetric red template sprite"
    coords = candidates[0]
    bbox = _bbox_coords(coords)
    patch = _crop_bbox((grid == RED).astype(int) * RED, bbox)
    return patch, coords


def _trace_path(grid: np.ndarray):
    edge_pt, _inner_pt, dir_idx = _find_start_bar(grid)
    current = edge_pt
    visited = [edge_pt]
    turnpoints: list[tuple[int, int]] = []
    turn_signs: list[int] = []
    limit = grid.shape[0] * grid.shape[1] * 4
    steps = 0
    while True:
        steps += 1
        assert steps <= limit, "path tracing exceeded safe step bound"
        dr, dc = DIRS[dir_idx]
        nr, nc = current[0] + dr, current[1] + dc
        if not _in_bounds(grid.shape, nr, nc):
            break
        if int(grid[nr, nc]) == GREY:
            options = []
            left_idx = _dir_after(dir_idx, -1)
            right_idx = _dir_after(dir_idx, 1)
            for sign, cand_idx in [(-1, left_idx), (1, right_idx)]:
                rr, cc = current[0] + DIRS[cand_idx][0], current[1] + DIRS[cand_idx][1]
                if _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) != GREY:
                    options.append((sign, cand_idx))
            assert len(options) == 1, "each blocker must force exactly one turn"
            turn_sign, dir_idx = options[0]
            turnpoints.append(current)
            turn_signs.append(turn_sign)
            continue
        current = (nr, nc)
        visited.append(current)
    return visited, turnpoints, turn_signs


def _stamp_patch(out: np.ndarray, patch: np.ndarray, center: tuple[int, int]) -> set[tuple[int, int]]:
    r0 = center[0] - 1
    c0 = center[1] - 1
    newly = set()
    for pr, pc in np.argwhere(patch != BG):
        rr = r0 + int(pr)
        cc = c0 + int(pc)
        if out[rr, cc] == BG:
            out[rr, cc] = RED
            newly.add((rr, cc))
    return newly


def _output_stamp_cells(path: list[tuple[int, int]], turnpoints: list[tuple[int, int]], turn_signs: list[int], patch: np.ndarray):
    cumulative = 0
    all_cells = []
    path_set = set(path)
    occupied = set()
    for tp, sign in zip(turnpoints, turn_signs):
        cumulative = (cumulative + sign) % 4
        rot = _rot_patch_cw(patch, cumulative)
        cells = _centered_patch_cells(rot, tp)
        all_cells.append(cells)
        occupied |= cells
    return all_cells, occupied, path_set


def _simulate_walk(
    grid: np.ndarray,
    template_patch: np.ndarray,
    erase_coords: np.ndarray | None = None,
):
    out = np.asarray(grid, dtype=int).copy()
    if erase_coords is not None and erase_coords.size:
        out[erase_coords[:, 0], erase_coords[:, 1]] = BG

    edge_pt, _inner_pt, dir_idx = _find_start_bar(grid)
    current = edge_pt
    turnpoints: list[tuple[int, int]] = []
    turn_signs: list[int] = []
    painted_turns: list[set[tuple[int, int]]] = []
    cumulative = 0
    limit = grid.shape[0] * grid.shape[1] * 4

    for _ in range(limit):
        dr, dc = DIRS[dir_idx]
        nr, nc = current[0] + dr, current[1] + dc
        if not _in_bounds(grid.shape, nr, nc):
            return out, turnpoints, turn_signs, painted_turns
        if int(grid[nr, nc]) == GREY:
            options = []
            left_idx = _dir_after(dir_idx, -1)
            right_idx = _dir_after(dir_idx, 1)
            for sign, cand_idx in [(-1, left_idx), (1, right_idx)]:
                rr, cc = current[0] + DIRS[cand_idx][0], current[1] + DIRS[cand_idx][1]
                if _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) != GREY:
                    options.append((sign, cand_idx))
            assert len(options) == 1, "each blocker must force exactly one turn"
            sign, dir_idx = options[0]
            cumulative = (cumulative + sign) % 4
            turnpoints.append(current)
            turn_signs.append(sign)
            painted_turns.append(_stamp_patch(out, _rot_patch_cw(template_patch, cumulative), current))
            continue

        current = (nr, nc)
        if out[current] == BG:
            out[current] = GREEN

    raise AssertionError("walk simulation exceeded safe step bound")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Walk the blocker-defined corridor, then stamp rotated copies of the template at each forced turn.

    Input structure:
    - Black (0) is the background.
    - A 2-pixel green (3) start bar touches exactly one grid edge and points inward.
    - Grey (5) blocker pixels force a single one-cell-wide orthogonal walk with 3-5 right-angle turns.
    - One isolated red (2) 4-connected template sprite occupies a 3x3 corner box and has no rotational symmetry.

    Rule:
    - Start from the edge pixel of the green bar and move in the inward direction.
    - Keep moving straight through non-grey cells; whenever the next cell is black, recolor it green.
    - When the next cell in the current direction is grey, stay on the current cell, inspect the left and right neighbors, and turn toward the unique non-grey side.
    - Maintain a cumulative rotation state: right turns add +90 degrees, left turns add -90 degrees modulo 360.
    - After each forced turn, rotate the template by the cumulative angle and stamp it centered on the turn cell, recoloring only black cells to red.
    - Stop when the next step would leave the grid, and erase the original corner template back to black.

    Key insight:
    - The grey blockers define turn events, and the red stamp orientation at each event is determined by the entire traced turn history from the start bar, not by the local blocker shape alone.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    template_patch, template_coords = _find_template_patch(g)
    out, turnpoints, _turn_signs, _painted_turns = _simulate_walk(g, template_patch, erase_coords=template_coords)
    assert MIN_TURNS <= len(turnpoints) <= MAX_TURNS, "expected 3-5 forced turns"
    return out


def _search_path(
    shape: tuple[int, int],
    current: tuple[int, int],
    dir_idx: int,
    turns_left: int,
    used_path: set[tuple[int, int]],
    blockers: set[tuple[int, int]],
    turnpoints: list[tuple[int, int]],
    turn_signs: list[int],
):
    h, w = shape
    if turns_left == 0:
        final_len = _steps_to_boundary(current, dir_idx, shape)
        if final_len < MIN_FINAL_SEG:
            return None
        final_cells = _segment_cells(current, dir_idx, final_len)
        if any(cell in used_path or cell in blockers for cell in final_cells):
            return None
        return {
            "path": set(used_path) | set(final_cells),
            "blockers": set(blockers),
            "turnpoints": list(turnpoints),
            "turn_signs": list(turn_signs),
        }

    max_to_edge = _steps_to_boundary(current, dir_idx, shape)
    lengths = list(range(MIN_SEG, min(MAX_SEG, max_to_edge - 1) + 1))
    random.shuffle(lengths)
    signs = [-1, 1]
    random.shuffle(signs)
    for length in lengths:
        seg_cells = _segment_cells(current, dir_idx, length)
        if any(cell in used_path or cell in blockers for cell in seg_cells):
            continue
        tp = seg_cells[-1]
        if not (1 <= tp[0] < h - 1 and 1 <= tp[1] < w - 1):
            continue
        dr, dc = DIRS[dir_idx]
        forward_block = (tp[0] + dr, tp[1] + dc)
        if not _in_bounds(shape, *forward_block) or forward_block in used_path or forward_block in blockers:
            continue

        for sign in signs:
            new_dir = _dir_after(dir_idx, sign)
            side_open = (tp[0] + DIRS[new_dir][0], tp[1] + DIRS[new_dir][1])
            side_block_dir = _dir_after(dir_idx, -sign)
            side_block = (tp[0] + DIRS[side_block_dir][0], tp[1] + DIRS[side_block_dir][1])
            if not _in_bounds(shape, *side_open) or not _in_bounds(shape, *side_block):
                continue
            if side_open in used_path or side_open in blockers or side_block in used_path or side_block in blockers:
                continue
            if turns_left == 1:
                if _steps_to_boundary(tp, new_dir, shape) < MIN_FINAL_SEG:
                    continue
            else:
                if _steps_to_boundary(tp, new_dir, shape) < MIN_SEG + 1:
                    continue

            result = _search_path(
                shape=shape,
                current=tp,
                dir_idx=new_dir,
                turns_left=turns_left - 1,
                used_path=set(used_path) | set(seg_cells),
                blockers=set(blockers) | {forward_block, side_block},
                turnpoints=turnpoints + [tp],
                turn_signs=turn_signs + [sign],
            )
            if result is not None:
                return result
    return None


def _start_state(shape: tuple[int, int]):
    h, w = shape
    side = random.choice([0, 1, 2, 3])  # top, right, bottom, left
    if side == 0:
        edge = (0, random.randint(4, w - 5))
        inner = (1, edge[1])
        dir_idx = 2
    elif side == 1:
        edge = (random.randint(4, h - 5), w - 1)
        inner = (edge[0], w - 2)
        dir_idx = 3
    elif side == 2:
        edge = (h - 1, random.randint(4, w - 5))
        inner = (h - 2, edge[1])
        dir_idx = 0
    else:
        edge = (random.randint(4, h - 5), 0)
        inner = (edge[0], 1)
        dir_idx = 1
    return edge, inner, dir_idx


def _choose_template_and_corner(
    shape: tuple[int, int],
    edge: tuple[int, int],
    inner: tuple[int, int],
    path_cells: set[tuple[int, int]],
    blockers: set[tuple[int, int]],
    turnpoints: list[tuple[int, int]],
    turn_signs: list[int],
):
    h, w = shape
    occupied_input = set(path_cells) | set(blockers)
    skeleton = np.zeros(shape, dtype=int)
    skeleton[edge] = GREEN
    skeleton[inner] = GREEN
    for r, c in blockers:
        skeleton[r, c] = GREY

    angles = []
    cumulative = 0
    for sign in turn_signs:
        cumulative = (cumulative + sign) % 4
        angles.append(cumulative)
    if len(set(angles)) < 2:
        return None
    if not any(angle != (1 if sign == 1 else 3) for angle, sign in zip(angles, turn_signs)):
        return None

    corners = [(0, 0), (0, w - 3), (h - 3, 0), (h - 3, w - 3)]
    random.shuffle(corners)
    free_corners = []
    for r0, c0 in corners:
        box = {(r0 + rr, c0 + cc) for rr in range(3) for cc in range(3)}
        if box & occupied_input:
            continue
        free_corners.append((r0, c0))
    if not free_corners:
        return None

    for patch in random.sample(TEMPLATE_VARIANTS, len(TEMPLATE_VARIANTS)):
        out, sim_turns, sim_signs, painted_turns = _simulate_walk(skeleton, patch)
        assert sim_turns == turnpoints and sim_signs == turn_signs
        if any(len(cells) < 2 for cells in painted_turns):
            continue
        rel_patterns = []
        for tp, cells in zip(turnpoints, painted_turns):
            rel_patterns.append(frozenset((r - tp[0], c - tp[1]) for r, c in cells))
        if len(set(rel_patterns)) < 2:
            continue
        if np.count_nonzero(out == RED) < 2 * len(turnpoints):
            continue
        return patch, random.choice(free_corners)
    return None


def _manual_render(shape: tuple[int, int], edge: tuple[int, int], inner: tuple[int, int], blockers: set[tuple[int, int]], patch: np.ndarray, corner: tuple[int, int]) -> np.ndarray:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    grid[edge] = GREEN
    grid[inner] = GREEN
    for r, c in blockers:
        grid[r, c] = GREY
    r0, c0 = corner
    for pr, pc in np.argwhere(patch != BG):
        grid[r0 + int(pr), c0 + int(pc)] = RED
    return grid


def _semantic_check(inp: np.ndarray) -> bool:
    try:
        out = main(inp)
    except AssertionError:
        return False

    path, turnpoints, turn_signs = _trace_path(inp)
    if len(turnpoints) < MIN_TURNS or len(turnpoints) > MAX_TURNS:
        return False
    if set(turn_signs) != {-1, 1}:
        return False

    template_patch, template_coords = _find_template_patch(inp)
    sim_out, sim_turnpoints, sim_signs, painted_turns = _simulate_walk(inp, template_patch, erase_coords=template_coords)
    if sim_turnpoints != turnpoints or sim_signs != turn_signs:
        return False
    if not np.array_equal(out, sim_out):
        return False
    if any(len(cells) < 2 for cells in painted_turns):
        return False
    if not np.all(out[template_coords[:, 0], template_coords[:, 1]] == BG):
        return False
    rel_patterns = []
    for tp, cells in zip(turnpoints, painted_turns):
        rel_patterns.append(frozenset((r - tp[0], c - tp[1]) for r, c in cells))
    if len(set(rel_patterns)) < 2:
        return False
    if np.count_nonzero(out == RED) < 2 * len(turnpoints):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        h = random.randint(18, 24)
        w = random.randint(18, 24)
        shape = (h, w)
        turns = random.randint(MIN_TURNS, MAX_TURNS)

        edge, inner, dir_idx = _start_state(shape)
        result = _search_path(
            shape=shape,
            current=inner,
            dir_idx=dir_idx,
            turns_left=turns,
            used_path={edge, inner},
            blockers=set(),
            turnpoints=[],
            turn_signs=[],
        )
        if result is None:
            continue
        if set(result["turn_signs"]) != {-1, 1}:
            continue

        choice = _choose_template_and_corner(
            shape=shape,
            edge=edge,
            inner=inner,
            path_cells=result["path"],
            blockers=result["blockers"],
            turnpoints=result["turnpoints"],
            turn_signs=result["turn_signs"],
        )
        if choice is None:
            continue
        patch, corner = choice
        grid = _manual_render(shape, edge, inner, result["blockers"], patch, corner)
        if _semantic_check(grid):
            return grid

    raise RuntimeError("failed to generate a valid corridor-stamp-rotation input")
