from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
WALL = 5
PALETTE = (1, 2, 3, 4)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _component_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(cells, dtype=int))
    return components


def _longest_true_run(row_mask: np.ndarray) -> tuple[int, int, int]:
    best_len = 0
    best_start = -1
    cur_start = -1
    for idx, value in enumerate(row_mask):
        if value and cur_start < 0:
            cur_start = idx
        if not value and cur_start >= 0:
            run_len = idx - cur_start
            if run_len > best_len:
                best_len = run_len
                best_start = cur_start
            cur_start = -1
    if cur_start >= 0:
        run_len = len(row_mask) - cur_start
        if run_len > best_len:
            best_len = run_len
            best_start = cur_start
    if best_len == 0:
        return 0, -1, -1
    return best_len, best_start, best_start + best_len - 1


def _divider_row(grid: np.ndarray) -> int:
    candidates: list[tuple[int, int]] = []
    for r in range(grid.shape[0]):
        run_len, _c0, _c1 = _longest_true_run(grid[r] == WALL)
        if run_len >= 5:
            candidates.append((run_len, r))
    assert candidates, "expected a long horizontal wall run"
    best_len = max(run_len for run_len, _r in candidates)
    return min(r for run_len, r in candidates if run_len == best_len)


def _fill_chambers(grid: np.ndarray, divider_r: int) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    lower = out[divider_r + 1 :, :]
    if lower.size == 0:
        return out

    h, w = lower.shape
    seen = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if seen[r, c] or lower[r, c] == WALL:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            markers: set[int] = set()
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                color = int(lower[rr, cc])
                if color not in (BG, WALL):
                    markers.add(color)
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and lower[nr, nc] != WALL:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if markers:
                assert len(markers) == 1
                fill_color = next(iter(markers))
                for rr, cc in cells:
                    lower[rr, cc] = fill_color
    out[divider_r + 1 :, :] = lower
    return out


def _sprites_above(grid: np.ndarray, divider_r: int) -> list[dict[str, object]]:
    mask = np.zeros_like(grid, dtype=bool)
    above = grid[:divider_r, :]
    for color in PALETTE:
        mask[:divider_r, :] |= above == color

    sprites: list[dict[str, object]] = []
    for coords in _component_from_mask(mask):
        rows = coords[:, 0]
        cols = coords[:, 1]
        color = int(grid[int(coords[0, 0]), int(coords[0, 1])])
        sprites.append(
            {
                "coords": coords.copy(),
                "color": color,
                "bottom": int(rows.max()),
                "bbox": (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())),
            }
        )
    sprites.sort(key=lambda sprite: (-int(sprite["bottom"]), int(sprite["bbox"][1])))
    return sprites


def _can_enter_cell(out: np.ndarray, filled: np.ndarray, divider_r: int, color: int, r: int, c: int) -> tuple[bool, bool]:
    h, w = out.shape
    if not (0 <= r < h and 0 <= c < w):
        return False, False
    cell = int(out[r, c])
    if cell == BG:
        return True, False
    if cell == color:
        return True, False
    if cell == WALL and r == divider_r and divider_r + 1 < h and int(filled[divider_r + 1, c]) == color:
        return True, True
    return False, False


def main(grid: np.ndarray) -> np.ndarray:
    """Flood-fill the lower chambers, then drop each sprite so only matching colors may pass through the divider wall.

    Input structure:
    - Black background surrounds a lower half of gray wall-bounded rectangular chambers topped by one continuous gray divider row.
    - Each chamber contains exactly one single-pixel marker color, and the same color set appears as separate 4-connected sprites above the divider.
    - Sprites begin strictly above the divider with at least one empty-row gap, while chamber interiors are separated by gray walls.

    Rule:
    - Flood-fill every chamber interior below the divider with its marker color, treating gray walls as barriers.
    - Clear the original upper sprites, then settle them from the lowest sprite upward by rigid vertical gravity.
    - A falling sprite may move into black cells, same-color filled chamber cells, or divider-row wall cells whose chamber directly below has the same color; any gray wall, different filled color, or previously settled sprite blocks it.
    - After settling, the sprite is written at its final position, and any divider-row columns it crossed remain recolored to the sprite color.

    Key insight:
    - The chamber marker does more than paint the room: it also gates whether a sprite can punch through the divider, so flood-filling must happen before gravity is resolved.
    """
    g = np.asarray(grid, dtype=int)
    divider_r = _divider_row(g)
    filled = _fill_chambers(g, divider_r)
    sprites = _sprites_above(g, divider_r)

    out = filled.copy()
    for sprite in sprites:
        coords = np.array(sprite["coords"], dtype=int)
        color = int(sprite["color"])
        for r, c in coords:
            out[int(r), int(c)] = BG

        current = coords.copy()
        crossed_cols: set[int] = set()
        while True:
            moved = current.copy()
            moved[:, 0] += 1
            blocked = False
            step_crossed: set[int] = set()
            for rr, cc in moved:
                ok, crossed = _can_enter_cell(out, filled, divider_r, color, int(rr), int(cc))
                if not ok:
                    blocked = True
                    break
                if crossed:
                    step_crossed.add(int(cc))
            if blocked:
                break
            current = moved
            crossed_cols.update(step_crossed)

        for r, c in current:
            out[int(r), int(c)] = color
        for c in crossed_cols:
            out[divider_r, int(c)] = color

    return out


def _normalize_shape(cells: set[tuple[int, int]]) -> np.ndarray:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    normalized = sorted((r - min_r, c - min_c) for r, c in cells)
    return np.array(normalized, dtype=int)


def _random_sprite_shape(max_width: int, max_height: int) -> np.ndarray | None:
    for _ in range(80):
        width = random.randint(1, min(3, max_width))
        height = random.randint(1, min(3, max_height))
        max_area = min(6, width * height)
        if max_area < 3:
            continue
        area = random.randint(3, max_area)
        seed = (random.randrange(height), random.randrange(width))
        cells = {seed}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        if len(cells) != area:
            continue
        shape = _normalize_shape(cells)
        if len({int(c) for _r, c in shape}) > max_width:
            continue
        return shape
    return None


def _sample_layout() -> tuple[int, int, int, int, list[int], int] | None:
    n_chambers = random.choice([3, 4])
    interior_h = random.choice([2, 3])
    widths = [random.choice([3, 4]) for _ in range(n_chambers)]
    cluster_w = sum(widths) + n_chambers + 1
    max_w = min(18, cluster_w + 4)
    if cluster_w > max_w:
        return None
    width = random.randint(cluster_w, max_w)
    left = random.randint(0, width - cluster_w)
    divider_r = random.randint(5, 8)
    min_h = divider_r + interior_h + 2
    height = random.randint(min_h, min(18, min_h + 2))
    return height, width, divider_r, interior_h, widths, left


def _chamber_records(divider_r: int, interior_h: int, widths: list[int], left: int, colors: list[int]) -> list[dict[str, int]]:
    records: list[dict[str, int]] = []
    c = left + 1
    for width, color in zip(widths, colors):
        records.append(
            {
                "color": int(color),
                "c0": int(c),
                "c1": int(c + width - 1),
                "r0": int(divider_r + 1),
                "r1": int(divider_r + interior_h),
            }
        )
        c += width + 1
    return records


def _paint_chambers(grid: np.ndarray, divider_r: int, interior_h: int, records: list[dict[str, int]], left: int) -> None:
    right = records[-1]["c1"] + 1
    grid[divider_r, left : right + 1] = WALL
    grid[divider_r + interior_h + 1, left : right + 1] = WALL
    for c in [left] + [rec["c1"] + 1 for rec in records]:
        grid[divider_r : divider_r + interior_h + 2, c] = WALL
    for rec in records:
        marker_r = random.randint(rec["r0"], rec["r1"])
        marker_c = random.randint(rec["c0"], rec["c1"])
        grid[marker_r, marker_c] = rec["color"]


def _mixed_permutation(n: int) -> list[int]:
    base = list(range(n))
    for _ in range(100):
        perm = base.copy()
        random.shuffle(perm)
        if any(perm[i] == i for i in range(n)) and any(perm[i] != i for i in range(n)):
            return perm
    raise RuntimeError("failed to sample mixed permutation")


def _place_sprites(grid: np.ndarray, divider_r: int, chambers: list[dict[str, int]], perm: list[int]) -> bool:
    for src_idx, chamber in enumerate(chambers):
        target = chambers[perm[src_idx]]
        target_width = target["c1"] - target["c0"] + 1
        shape = _random_sprite_shape(target_width, min(3, divider_r - 1))
        if shape is None:
            return False

        shape_h = int(shape[:, 0].max()) + 1
        shape_w = int(shape[:, 1].max()) + 1
        max_top = divider_r - shape_h - 1
        if max_top < 0:
            return False
        top = random.randint(0, max_top)
        left = random.randint(target["c0"], target["c1"] - shape_w + 1)

        color = int(chamber["color"])
        for dr, dc in shape:
            rr = top + int(dr)
            cc = left + int(dc)
            if grid[rr, cc] != BG:
                return False
            grid[rr, cc] = color
    return True


def _attempt_generate() -> np.ndarray | None:
    layout = _sample_layout()
    if layout is None:
        return None
    height, width, divider_r, interior_h, widths, left = layout
    colors = random.sample(list(PALETTE), len(widths))
    chambers = _chamber_records(divider_r, interior_h, widths, left, colors)

    grid = np.full((height, width), BG, dtype=int)
    _paint_chambers(grid, divider_r, interior_h, chambers, left)

    perm = _mixed_permutation(len(chambers))
    if not _place_sprites(grid, divider_r, chambers, perm):
        return None

    # Keep the upper area visually clean and guarantee both gated branches appear.
    assert any(perm[i] == i for i in range(len(perm)))
    assert any(perm[i] != i for i in range(len(perm)))

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if int(np.count_nonzero(grid == BG)) <= int(grid.size // 3):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
