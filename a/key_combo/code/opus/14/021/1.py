from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5

DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIR8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


SPRITE_LIBRARY = [
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1], [0, 1, 0]], dtype=bool),
]


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    dirs = DIR4 if int(connectivity) == 4 else DIR8
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _draw_segment(mask: np.ndarray, p0: tuple[int, int], p1: tuple[int, int]) -> None:
    (r0, c0), (r1, c1) = p0, p1
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        for c in range(c0, c1 + step, step):
            mask[int(r0), int(c)] = True
    elif c0 == c1:
        step = 1 if r1 >= r0 else -1
        for r in range(r0, r1 + step, step):
            mask[int(r), int(c0)] = True
    else:
        raise ValueError("segments must be axis-aligned")


def _draw_polyline(mask: np.ndarray, points: list[tuple[int, int]]) -> None:
    for a, b in zip(points, points[1:]):
        _draw_segment(mask, a, b)


def _split_cells(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    grey_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == GREY)]
    grey_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == GREY)]
    assert len(grey_rows) == 3 and len(grey_cols) == 4, "expected a 2x3 grey divider grid with frame"
    row_ranges = [(grey_rows[i] + 1, grey_rows[i + 1] - 1) for i in range(2)]
    col_ranges = [(grey_cols[i] + 1, grey_cols[i + 1] - 1) for i in range(3)]
    row_sizes = {r1 - r0 + 1 for r0, r1 in row_ranges}
    col_sizes = {c1 - c0 + 1 for c0, c1 in col_ranges}
    assert len(row_sizes) == 1 and len(col_sizes) == 1 and row_sizes == col_sizes
    return row_ranges, col_ranges


def _black_region_count(cell: np.ndarray) -> int:
    return len(_components(np.asarray(cell, dtype=int) == BLACK, connectivity=4))


def _rotate_red_mask(mask: np.ndarray, turns: int) -> np.ndarray:
    turns %= 4
    if turns == 0:
        return mask.copy()
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0, r1, c1 = _bbox(mask)
    assert r0 is not None
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    pts = coords.astype(float)
    for _ in range(turns):
        dr = pts[:, 0] - center_r
        dc = pts[:, 1] - center_c
        pts = np.stack([center_r + dc, center_c - dr], axis=1)
    rounded = np.rint(pts).astype(int)
    assert np.allclose(pts, rounded)
    out = np.zeros_like(mask, dtype=bool)
    out[rounded[:, 0], rounded[:, 1]] = True
    return out


def _rotationally_distinct(mask: np.ndarray) -> bool:
    return all(not np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3))


def _wall_variant(size: int, regions: int, variant: str) -> np.ndarray:
    s = int(size)
    q1 = max(1, s // 3)
    q2 = min(s - 2, s - 1 - q1)
    mid = s // 2
    mask = np.zeros((s, s), dtype=bool)

    if variant == "1_dangle_v":
        _draw_polyline(mask, [(0, mid), (s - 3, mid)])
    elif variant == "1_hook":
        _draw_polyline(mask, [(0, q1), (mid, q1), (mid, q2)])
    elif variant == "2_split_v":
        _draw_segment(mask, (0, mid), (s - 1, mid))
    elif variant == "2_split_h":
        _draw_segment(mask, (mid, 0), (mid, s - 1))
    elif variant == "2_bent":
        _draw_polyline(mask, [(0, q1), (mid, q1), (mid, q2), (s - 1, q2)])
    elif variant == "3_tee":
        _draw_segment(mask, (0, q2), (s - 1, q2))
        _draw_segment(mask, (mid, 0), (mid, q2))
    elif variant == "3_tee_low":
        _draw_segment(mask, (q1, 0), (q1, s - 1))
        _draw_segment(mask, (q1, mid), (s - 1, mid))
    elif variant == "4_cross":
        _draw_segment(mask, (0, mid), (s - 1, mid))
        _draw_segment(mask, (mid, 0), (mid, s - 1))
    elif variant == "4_double_tee":
        _draw_segment(mask, (0, mid), (s - 1, mid))
        _draw_segment(mask, (q1, 0), (q1, mid))
        _draw_segment(mask, (q2, mid), (q2, s - 1))
    else:
        raise ValueError(f"unknown wall variant: {variant}")

    assert _black_region_count(np.where(mask, BLUE, BLACK)) == int(regions)
    return mask


WALL_VARIANTS = {
    1: ("1_dangle_v", "1_hook"),
    2: ("2_split_v", "2_split_h", "2_bent"),
    3: ("3_tee", "3_tee_low"),
    4: ("4_cross", "4_double_tee"),
}


def _random_wall_mask(size: int, regions: int, forced_variant: str | None = None) -> tuple[np.ndarray, str]:
    variant = forced_variant if forced_variant is not None else random.choice(WALL_VARIANTS[int(regions)])
    mask = _wall_variant(int(size), int(regions), variant)
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    if random.random() < 0.5:
        mask = np.flipud(mask)
    k = random.randrange(4)
    mask = np.rot90(mask, k)
    assert _black_region_count(np.where(mask, BLUE, BLACK)) == int(regions)
    return mask, variant


def _safe_sprite_positions(
    walls: np.ndarray, sprite: np.ndarray, target_regions: int
) -> list[tuple[int, int, int]]:
    s = walls.shape[0]
    n = sprite.shape[0]
    candidates: list[tuple[int, int, int]] = []
    for r0 in range(s - n + 1):
        for c0 in range(s - n + 1):
            cell = np.where(walls, BLUE, BLACK).astype(int)
            ok = True
            for rr, cc in np.argwhere(sprite):
                r = r0 + int(rr)
                c = c0 + int(cc)
                if cell[r, c] != BLACK:
                    ok = False
                    break
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < s and 0 <= nc < s and walls[nr, nc]:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            for rr, cc in np.argwhere(sprite):
                cell[r0 + int(rr), c0 + int(cc)] = RED
            if _black_region_count(cell) != int(target_regions):
                continue
            boundary_score = 0
            for rr, cc in np.argwhere(sprite):
                r = r0 + int(rr)
                c = c0 + int(cc)
                if r in (0, s - 1) or c in (0, s - 1):
                    boundary_score += 1
            candidates.append((boundary_score, r0, c0))
    candidates.sort()
    return candidates


def _build_cell(size: int, regions: int, sprite_mask: np.ndarray, forced_variant: str | None = None) -> tuple[np.ndarray, dict]:
    for _ in range(120):
        walls, variant = _random_wall_mask(size, regions, forced_variant=forced_variant)
        oriented_sprite = np.rot90(sprite_mask, random.randrange(4))
        if random.random() < 0.35:
            oriented_sprite = np.fliplr(oriented_sprite)
        candidates = _safe_sprite_positions(walls, oriented_sprite, regions)
        if not candidates:
            continue
        best_boundary = candidates[0][0]
        r0, c0 = random.choice([(r, c) for score, r, c in candidates if score == best_boundary])
        cell = np.where(walls, BLUE, BLACK).astype(int)
        for rr, cc in np.argwhere(oriented_sprite):
            cell[r0 + int(rr), c0 + int(cc)] = RED
        info = {
            "regions": int(regions),
            "variant": variant,
            "wall_components": len(_components(walls, connectivity=4)),
            "wall_area": int(walls.sum()),
        }
        return cell, info
    raise RuntimeError("failed to build a valid cell")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each red sprite by the number of enclosed black regions encoded by blue walls inside its divider cell.

    Input structure:
    - A grey framed divider grid splits the canvas into 2x3 equal cells.
    - Each cell contains blue one-pixel wall segments, black background, and one red 4-connected sprite.
    - The blue walls and the grey cell boundary partition the black background into 1 to 4 enclosed 4-connected regions.

    Rule:
    - For each cell, count the number N of 4-connected black background regions inside that cell.
    - Rotate the cell's red sprite clockwise by (N mod 4) quarter-turns around the center of its tight bounding box.
    - Keep the grey divider grid, erase all blue walls to black, and place only the rotated red sprite back into the cell.

    Key insight:
    - The blue walls are not output objects; their topology encodes a hidden integer, and that region count directly selects the sprite's rotation.
    """
    g = np.asarray(grid, dtype=int)
    row_ranges, col_ranges = _split_cells(g)
    out = np.full_like(g, BLACK)
    out[g == GREY] = GREY

    for r0, r1 in row_ranges:
        for c0, c1 in col_ranges:
            cell = g[r0 : r1 + 1, c0 : c1 + 1]
            turns = _black_region_count(cell)
            red_mask = cell == RED
            rotated = _rotate_red_mask(red_mask, turns)
            out[r0 : r1 + 1, c0 : c1 + 1][rotated] = RED

    return out


def _validate_puzzle(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    row_ranges, col_ranges = _split_cells(g)
    counts: list[int] = []
    wall_comps: list[int] = []
    wall_areas: list[int] = []
    output_signatures: list[bytes] = []

    solved = main(g)
    assert np.array_equal(main(g.copy()), solved)
    assert not np.any(solved == BLUE)

    for r0, r1 in row_ranges:
        for c0, c1 in col_ranges:
            cell = g[r0 : r1 + 1, c0 : c1 + 1]
            red_mask = cell == RED
            blue_mask = cell == BLUE
            red_comps = _components(red_mask, connectivity=4)
            assert len(red_comps) == 1
            b = _bbox(red_mask)
            assert b is not None
            br0, bc0, br1, bc1 = b
            crop = red_mask[br0 : br1 + 1, bc0 : bc1 + 1]
            assert crop.shape[0] == crop.shape[1] and crop.shape[0] in (2, 3)
            assert 3 <= int(crop.sum()) <= 6
            assert _rotationally_distinct(crop)
            for rr, cc in np.argwhere(red_mask):
                for dr, dc in DIR4:
                    nr, nc = int(rr) + dr, int(cc) + dc
                    if 0 <= nr < cell.shape[0] and 0 <= nc < cell.shape[1]:
                        assert not blue_mask[nr, nc]
            counts.append(_black_region_count(cell))
            wall_comps.append(len(_components(blue_mask, connectivity=4)))
            wall_areas.append(int(blue_mask.sum()))
            out_cell = solved[r0 : r1 + 1, c0 : c1 + 1]
            output_signatures.append((out_cell == RED).astype(np.uint8).tobytes())

    assert set(counts) == {1, 2, 3, 4}
    assert counts.count(3) >= 2
    assert len(set(output_signatures)) == 6
    assert any(
        counts[i] != counts[j] and wall_comps[i] == wall_comps[j]
        for i in range(6)
        for j in range(i + 1, 6)
    )
    assert any(
        counts[i] == counts[j] and (wall_comps[i] != wall_comps[j] or wall_areas[i] != wall_areas[j])
        for i in range(6)
        for j in range(i + 1, 6)
    )


def generate_input() -> np.ndarray:
    for sprite in SPRITE_LIBRARY:
        assert _rotationally_distinct(sprite)

    for _ in range(300):
        cell_size = random.randint(7, 8)
        grid = np.full((2 * cell_size + 3, 3 * cell_size + 4), BLACK, dtype=int)
        grey_rows = [0, cell_size + 1, 2 * cell_size + 2]
        grey_cols = [0, cell_size + 1, 2 * cell_size + 2, 3 * cell_size + 3]
        grid[grey_rows, :] = GREY
        grid[:, grey_cols] = GREY

        counts = [1, 2, 3, 3, 4, random.choice([1, 2, 4])]
        random.shuffle(counts)
        three_slots = [i for i, v in enumerate(counts) if v == 3]
        assert len(three_slots) >= 2

        forced: dict[int, str] = {
            three_slots[0]: "3_tee",
            three_slots[1]: "3_tee_low",
        }

        metas: list[dict] = []
        ok = True
        for idx, regions in enumerate(counts):
            rr = idx // 3
            cc = idx % 3
            built = None
            sprite_order = random.sample(range(len(SPRITE_LIBRARY)), len(SPRITE_LIBRARY))
            for sprite_id in sprite_order:
                sprite = SPRITE_LIBRARY[sprite_id]
                try:
                    cell, info = _build_cell(cell_size, regions, sprite, forced_variant=forced.get(idx))
                    built = (cell, info, sprite_id)
                    break
                except RuntimeError:
                    continue
            if built is None:
                ok = False
                break
            cell, info, _sprite_id = built
            metas.append(info)
            r0 = 1 + rr * (cell_size + 1)
            c0 = 1 + cc * (cell_size + 1)
            grid[r0 : r0 + cell_size, c0 : c0 + cell_size] = cell
        if not ok:
            continue

        try:
            _validate_puzzle(grid)
        except Exception:
            continue
        return grid

    raise RuntimeError("generate_input: failed after 300 attempts")
