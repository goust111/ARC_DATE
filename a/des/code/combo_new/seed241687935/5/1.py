from __future__ import annotations

from itertools import combinations
import random

import numpy as np


BG_COLOR = 0
MARK_COLOR = 2
FRAME_COLOR = 5
PALETTE = (1, 3, 4, 6, 7, 8, 9)
ROTATION_FROM_SIDE = {"top": 0, "right": 1, "bottom": 2, "left": 3}
MASTER_SIZE = 30
GEN_RETRIES = 500


def _rotate_cw(arr: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(arr, (-int(k)) % 4)


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _is_rotationally_distinct(patch: np.ndarray) -> bool:
    return all(not np.array_equal(patch, np.rot90(patch, k)) for k in (1, 2, 3))


def _mark_buffer(mask: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> None:
    rr0 = max(0, int(r0) - int(pad))
    cc0 = max(0, int(c0) - int(pad))
    rr1 = min(mask.shape[0], int(r0) + int(h) + int(pad))
    cc1 = min(mask.shape[1], int(c0) + int(w) + int(pad))
    mask[rr0:rr1, cc0:cc1] = True


def _random_segment(start: int, end: int, length: int, rng: random.Random) -> np.ndarray:
    seg = np.empty(length, dtype=int)
    seg[0] = int(start)
    seg[-1] = int(end)
    if length > 2:
        seg[1:-1] = np.array(rng.choices(PALETTE, k=length - 2), dtype=int)
    return seg


def _break_symmetry(patch: np.ndarray, rng: random.Random) -> np.ndarray:
    size = int(patch.shape[0])
    coords = [
        (r, c)
        for r in range(1, size - 1)
        for c in range(1, size - 1)
        if len({(r, c), (c, size - 1 - r), (size - 1 - r, size - 1 - c), (size - 1 - c, r)}) > 1
    ]
    if not coords:
        return patch
    for _ in range(16):
        if _is_rotationally_distinct(patch):
            break
        r, c = rng.choice(coords)
        current = int(patch[r, c])
        choices = [color for color in PALETTE if color != current]
        patch[r, c] = rng.choice(choices)
    return patch


def _build_patch_grid(rows: int, cols: int, tile: int, rng: random.Random) -> list[list[np.ndarray]] | None:
    vertices = np.array(rng.choices(PALETTE, k=(rows + 1) * (cols + 1)), dtype=int).reshape(rows + 1, cols + 1)
    h_edges = [[_random_segment(int(vertices[r, c]), int(vertices[r, c + 1]), tile, rng) for c in range(cols)] for r in range(rows + 1)]
    v_edges = [[_random_segment(int(vertices[r, c]), int(vertices[r + 1, c]), tile, rng) for c in range(cols + 1)] for r in range(rows)]

    patches: list[list[np.ndarray]] = []
    for r in range(rows):
        row_patches: list[np.ndarray] = []
        for c in range(cols):
            patch = np.array(rng.choices(PALETTE, k=tile * tile), dtype=int).reshape(tile, tile)
            patch[0, :] = h_edges[r][c]
            patch[-1, :] = h_edges[r + 1][c]
            patch[:, 0] = v_edges[r][c]
            patch[:, -1] = v_edges[r][c + 1]
            patch = _break_symmetry(patch, rng)
            if not _is_rotationally_distinct(patch):
                return None
            row_patches.append(patch)
        patches.append(row_patches)
    return patches


def _valid_hole_sets(rows: int, cols: int) -> dict[int, list[tuple[tuple[int, int], ...]]]:
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    valid: dict[int, list[tuple[tuple[int, int], ...]]] = {}
    for size in range(2, min(4, len(cells) - 1) + 1):
        sets_for_size = []
        for combo in combinations(cells, size):
            ok = True
            for i in range(size):
                for j in range(i + 1, size):
                    if abs(combo[i][0] - combo[j][0]) + abs(combo[i][1] - combo[j][1]) <= 1:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                sets_for_size.append(combo)
        if sets_for_size:
            valid[size] = sets_for_size
    return valid


def _draw_frame(canvas: np.ndarray, top: int, left: int, patches: list[list[np.ndarray]]) -> None:
    rows = len(patches)
    cols = len(patches[0])
    tile = int(patches[0][0].shape[0])
    frame_h = rows * tile + rows + 1
    frame_w = cols * tile + cols + 1

    for i in range(rows + 1):
        rr = top + i * (tile + 1)
        canvas[rr, left : left + frame_w] = FRAME_COLOR
    for j in range(cols + 1):
        cc = left + j * (tile + 1)
        canvas[top : top + frame_h, cc] = FRAME_COLOR

    for r in range(rows):
        for c in range(cols):
            rr = top + 1 + r * (tile + 1)
            cc = left + 1 + c * (tile + 1)
            canvas[rr : rr + tile, cc : cc + tile] = patches[r][c]


def _crop_pair(inp: np.ndarray, out: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    mask = (inp != BG_COLOR) | (out != BG_COLOR)
    points = np.argwhere(mask)
    r0, c0 = points.min(axis=0)
    r1, c1 = points.max(axis=0)
    pad_top = rng.randint(1, 2)
    pad_bottom = rng.randint(1, 2)
    pad_left = rng.randint(1, 2)
    pad_right = rng.randint(1, 2)
    rr0 = max(0, int(r0) - pad_top)
    cc0 = max(0, int(c0) - pad_left)
    rr1 = min(inp.shape[0] - 1, int(r1) + pad_bottom)
    cc1 = min(inp.shape[1] - 1, int(c1) + pad_right)
    return inp[rr0 : rr1 + 1, cc0 : cc1 + 1].copy(), out[rr0 : rr1 + 1, cc0 : cc1 + 1].copy()


def _find_frame(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[int], list[int], int]:
    mask = (grid == FRAME_COLOR) | (grid == MARK_COLOR)
    points = np.argwhere(mask)
    assert points.size > 0, "frame pixels not found"
    r0, c0 = points.min(axis=0)
    r1, c1 = points.max(axis=0)
    allowed = (FRAME_COLOR, MARK_COLOR)

    line_rows = [
        int(r)
        for r in range(int(r0), int(r1) + 1)
        if np.all(np.isin(grid[int(r), int(c0) : int(c1) + 1], allowed))
    ]
    line_cols = [
        int(c)
        for c in range(int(c0), int(c1) + 1)
        if np.all(np.isin(grid[int(r0) : int(r1) + 1, int(c)], allowed))
    ]
    assert len(line_rows) >= 3 and len(line_cols) >= 3, "grid lines not found"

    row_steps = [b - a for a, b in zip(line_rows, line_rows[1:])]
    col_steps = [b - a for a, b in zip(line_cols, line_cols[1:])]
    assert len(set(row_steps)) == 1 and len(set(col_steps)) == 1, "cell spacing must be regular"
    assert row_steps[0] == col_steps[0], "cells must be square"
    tile = row_steps[0] - 1
    assert tile >= 1, "tile size must be positive"
    return (int(r0), int(c0), int(r1), int(c1)), line_rows, line_cols, int(tile)


def _parse_puzzle(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    frame_bbox, line_rows, line_cols, tile = _find_frame(g)
    r0, c0, r1, c1 = frame_bbox
    cell_rows = len(line_rows) - 1
    cell_cols = len(line_cols) - 1

    cells: list[list[np.ndarray | None]] = [[None for _ in range(cell_cols)] for _ in range(cell_rows)]
    holes: list[dict[str, object]] = []

    for i in range(cell_rows):
        for j in range(cell_cols):
            rs0 = line_rows[i] + 1
            rs1 = line_rows[i + 1]
            cs0 = line_cols[j] + 1
            cs1 = line_cols[j + 1]
            patch = g[rs0:rs1, cs0:cs1].copy()
            if np.all(patch == BG_COLOR):
                side_hits = {"top": [], "right": [], "bottom": [], "left": []}
                for c in range(cs0, cs1):
                    if int(g[line_rows[i], c]) == MARK_COLOR:
                        side_hits["top"].append((line_rows[i], c))
                    if int(g[line_rows[i + 1], c]) == MARK_COLOR:
                        side_hits["bottom"].append((line_rows[i + 1], c))
                for r in range(rs0, rs1):
                    if int(g[r, line_cols[j]]) == MARK_COLOR:
                        side_hits["left"].append((r, line_cols[j]))
                    if int(g[r, line_cols[j + 1]]) == MARK_COLOR:
                        side_hits["right"].append((r, line_cols[j + 1]))
                active_sides = [side for side, hits in side_hits.items() if hits]
                assert len(active_sides) == 1, "each hole must have exactly one border marker"
                total_hits = sum(len(hits) for hits in side_hits.values())
                assert total_hits == 1, "each hole must have exactly one marker pixel"
                holes.append(
                    {
                        "cell": (i, j),
                        "rotation": ROTATION_FROM_SIDE[active_sides[0]],
                        "r0": rs0,
                        "c0": cs0,
                        "required": {},
                    }
                )
            else:
                cells[i][j] = patch

    for hole in holes:
        i, j = hole["cell"]
        required: dict[str, np.ndarray] = {}
        if i > 0 and cells[i - 1][j] is not None:
            required["top"] = cells[i - 1][j][-1, :].copy()
        if i + 1 < cell_rows and cells[i + 1][j] is not None:
            required["bottom"] = cells[i + 1][j][0, :].copy()
        if j > 0 and cells[i][j - 1] is not None:
            required["left"] = cells[i][j - 1][:, -1].copy()
        if j + 1 < cell_cols and cells[i][j + 1] is not None:
            required["right"] = cells[i][j + 1][:, 0].copy()
        hole["required"] = required
        assert required, "every hole must have at least one filled neighbor"

    outside_mask = g != BG_COLOR
    outside_mask[r0 : r1 + 1, c0 : c1 + 1] = False
    fragments: list[dict[str, object]] = []
    for comp in _components4(outside_mask):
        fr0, fc0, fr1, fc1 = _bbox(comp)
        patch = g[fr0 : fr1 + 1, fc0 : fc1 + 1].copy()
        fragments.append({"bbox": (fr0, fc0, fr1, fc1), "patch": patch})

    return {
        "grid": g,
        "frame_bbox": frame_bbox,
        "line_rows": line_rows,
        "line_cols": line_cols,
        "tile": tile,
        "cells": cells,
        "holes": holes,
        "fragments": fragments,
    }


def _patch_fits(required: dict[str, np.ndarray], patch: np.ndarray) -> bool:
    if "top" in required and not np.array_equal(patch[0, :], required["top"]):
        return False
    if "bottom" in required and not np.array_equal(patch[-1, :], required["bottom"]):
        return False
    if "left" in required and not np.array_equal(patch[:, 0], required["left"]):
        return False
    if "right" in required and not np.array_equal(patch[:, -1], required["right"]):
        return False
    return True


def _candidate_lists(parsed: dict[str, object]) -> list[list[tuple[int, np.ndarray]]]:
    tile = int(parsed["tile"])
    holes = parsed["holes"]
    fragments = parsed["fragments"]
    assert len(holes) == len(fragments), "number of holes and fragments must match"

    candidates: list[list[tuple[int, np.ndarray]]] = []
    for hole in holes:
        options: list[tuple[int, np.ndarray]] = []
        for frag_idx, frag in enumerate(fragments):
            patch = np.asarray(frag["patch"], dtype=int)
            if patch.shape != (tile, tile):
                continue
            rotated = _rotate_cw(patch, int(hole["rotation"]))
            if _patch_fits(hole["required"], rotated):
                options.append((frag_idx, rotated))
        candidates.append(options)
    return candidates


def _count_assignments(candidates: list[list[tuple[int, np.ndarray]]], stop_after: int = 2) -> int:
    order = sorted(range(len(candidates)), key=lambda idx: len(candidates[idx]))
    used: set[int] = set()
    count = 0

    def dfs(pos: int) -> None:
        nonlocal count
        if count >= stop_after:
            return
        if pos == len(order):
            count += 1
            return
        hole_idx = order[pos]
        for frag_idx, _patch in candidates[hole_idx]:
            if frag_idx in used:
                continue
            used.add(frag_idx)
            dfs(pos + 1)
            used.remove(frag_idx)

    dfs(0)
    return count


def _solve_assignments(candidates: list[list[tuple[int, np.ndarray]]]) -> dict[int, tuple[int, np.ndarray]]:
    order = sorted(range(len(candidates)), key=lambda idx: len(candidates[idx]))
    used: set[int] = set()
    chosen: dict[int, tuple[int, np.ndarray]] = {}

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return True
        hole_idx = order[pos]
        for frag_idx, patch in candidates[hole_idx]:
            if frag_idx in used:
                continue
            used.add(frag_idx)
            chosen[hole_idx] = (frag_idx, patch)
            if dfs(pos + 1):
                return True
            used.remove(frag_idx)
            chosen.pop(hole_idx, None)
        return False

    assert dfs(0), "valid puzzle must admit a complete assignment"
    return chosen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Insert the scattered square fragments back into the framed grid after rotating them by the red border cues.

    Input structure:
    - Black background.
    - One gray rectangular frame with 1-pixel grid lines divides its interior into equal square cells.
    - Most cells contain full multicolor square patches, while a few cells are empty black holes.
    - Every hole has exactly one red pixel on one of its four gray border segments.
    - Outside the frame sit multicolor square fragments with the same size as a cell.

    Rule:
    - For each hole, test the outside fragments against the already-filled neighboring cells: the correct fragment is the one whose shared edges match those neighbors pixel-for-pixel after placement.
    - The red border pixel gives the required clockwise rotation for that hole: top=0, right=90, bottom=180, left=270.
    - Rotate the matched fragment, place it into the hole, erase the original outside fragment, and restore the red clue pixel to gray.

    Key insight: Matching and orientation are separate clues: neighbor edge colors identify which fragment belongs in a hole, while the marker side independently tells how much to rotate it.
    """
    g = np.asarray(grid, dtype=int)
    parsed = _parse_puzzle(g)
    candidates = _candidate_lists(parsed)
    assert all(candidates), "every hole must have at least one compatible fragment"
    assignment = _solve_assignments(candidates)

    out = g.copy()
    out[out == MARK_COLOR] = FRAME_COLOR
    for frag in parsed["fragments"]:
        fr0, fc0, fr1, fc1 = frag["bbox"]
        out[int(fr0) : int(fr1) + 1, int(fc0) : int(fc1) + 1] = BG_COLOR

    for hole_idx, (_frag_idx, patch) in assignment.items():
        hole = parsed["holes"][hole_idx]
        r0 = int(hole["r0"])
        c0 = int(hole["c0"])
        out[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = patch

    return out


def _build_instance(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    rows = rng.randint(2, 4)
    cols = rng.randint(2, 4)
    tile_choices = [size for size in (4, 5) if rows * size + rows + 1 <= 20 and cols * size + cols + 1 <= 20]
    if not tile_choices:
        return None
    tile = rng.choice(tile_choices)

    patches = _build_patch_grid(rows, cols, tile, rng)
    if patches is None:
        return None

    hole_sets = _valid_hole_sets(rows, cols)
    if not hole_sets:
        return None
    hole_count = rng.choice(sorted(hole_sets))
    holes = list(rng.choice(hole_sets[hole_count]))

    frame_h = rows * tile + rows + 1
    frame_w = cols * tile + cols + 1
    if frame_h > MASTER_SIZE - 2 or frame_w > MASTER_SIZE - 2:
        return None

    full = np.full((MASTER_SIZE, MASTER_SIZE), BG_COLOR, dtype=int)
    inp = np.full((MASTER_SIZE, MASTER_SIZE), BG_COLOR, dtype=int)

    frame_top = rng.randint(1, MASTER_SIZE - frame_h - 1)
    frame_left = rng.randint(1, MASTER_SIZE - frame_w - 1)
    _draw_frame(full, frame_top, frame_left, patches)
    _draw_frame(inp, frame_top, frame_left, patches)

    fragments_to_place: list[np.ndarray] = []
    for hr, hc in holes:
        cell_top = frame_top + 1 + hr * (tile + 1)
        cell_left = frame_left + 1 + hc * (tile + 1)
        final_patch = patches[hr][hc]
        inp[cell_top : cell_top + tile, cell_left : cell_left + tile] = BG_COLOR

        rotation = rng.randrange(4)
        side = ("top", "right", "bottom", "left")[rotation]
        offset = rng.randint(1, tile - 2)
        if side == "top":
            inp[cell_top - 1, cell_left + offset] = MARK_COLOR
        elif side == "right":
            inp[cell_top + offset, cell_left + tile] = MARK_COLOR
        elif side == "bottom":
            inp[cell_top + tile, cell_left + offset] = MARK_COLOR
        else:
            inp[cell_top + offset, cell_left - 1] = MARK_COLOR

        fragments_to_place.append(np.rot90(final_patch, rotation))

    occupancy = np.zeros((MASTER_SIZE, MASTER_SIZE), dtype=bool)
    _mark_buffer(occupancy, frame_top, frame_left, frame_h, frame_w, pad=1)

    for patch in fragments_to_place:
        placed = False
        for _ in range(300):
            rr = rng.randint(1, MASTER_SIZE - tile - 1)
            cc = rng.randint(1, MASTER_SIZE - tile - 1)
            if bool(occupancy[rr : rr + tile, cc : cc + tile].any()):
                continue
            inp[rr : rr + tile, cc : cc + tile] = patch
            _mark_buffer(occupancy, rr, cc, tile, tile, pad=1)
            placed = True
            break
        if not placed:
            return None

    cropped_inp, cropped_full = _crop_pair(inp, full, rng)
    parsed = _parse_puzzle(cropped_inp)
    candidates = _candidate_lists(parsed)
    if not candidates or not all(candidates):
        return None
    if _count_assignments(candidates) != 1:
        return None
    if not np.array_equal(main(cropped_inp), cropped_full):
        return None
    return cropped_inp, cropped_full


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        built = _build_instance(rng)
        if built is None:
            continue
        inp, _out = built
        return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
