from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
MAX_GEN_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return np.zeros((0, 0), dtype=bool)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _is_connected(mask: np.ndarray) -> bool:
    return bool(mask.any()) and len(_components(mask)) == 1


def _count_holes(mask: np.ndarray) -> int:
    holes = 0
    h, w = mask.shape
    for comp in _components(~mask):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        holes += 1
    return holes


def _legend_info(grid: np.ndarray, bg: int) -> tuple[list[int], np.ndarray]:
    h, w = grid.shape
    candidates: list[tuple[int, list[tuple[int, int]]]] = []
    for comp in _components(grid != bg):
        rows = {r for r, _ in comp}
        if len(rows) != 1:
            continue
        row = next(iter(rows))
        if row not in (0, h - 1) or len(comp) < 2:
            continue
        cols = sorted(c for _, c in comp)
        if cols[-1] - cols[0] + 1 != len(cols):
            continue
        ordered = sorted(comp, key=lambda p: p[1])
        candidates.append((len(comp), ordered))
    assert candidates, "expected a horizontal legend on the top or bottom edge"
    candidates.sort(key=lambda item: item[0], reverse=True)
    legend_cells = candidates[0][1]
    legend_mask = np.zeros((h, w), dtype=bool)
    for r, c in legend_cells:
        legend_mask[r, c] = True
    legend_colors = [int(grid[r, c]) for r, c in legend_cells]
    return legend_colors, legend_mask


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    legend_colors, legend_mask = _legend_info(g, bg)

    out = np.full_like(g, bg)
    out[legend_mask] = g[legend_mask]

    object_mask = (g != bg) & ~legend_mask
    for comp in _components(object_mask):
        r0, c0, r1, c1 = _bbox(comp)
        sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            sub[r - r0, c - c0] = True
        hole_count = _count_holes(sub)
        if hole_count == 0:
            continue
        assert hole_count < len(legend_colors), "hole count exceeds legend length"
        assert r0 + hole_count <= g.shape[0] and c0 + hole_count <= g.shape[1], "square must fit on the grid"
        out[r0 : r0 + hole_count, c0 : c0 + hole_count] = int(legend_colors[hole_count])

    return out


def _expanded_rect(
    top: int, left: int, height: int, width: int, pad: int, grid_h: int, grid_w: int
) -> tuple[int, int, int, int]:
    return (
        max(0, top - pad),
        max(0, left - pad),
        min(grid_h - 1, top + height - 1 + pad),
        min(grid_w - 1, left + width - 1 + pad),
    )


def _rect_is_free(blocked: np.ndarray, rect: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = rect
    return not bool(blocked[r0 : r1 + 1, c0 : c1 + 1].any())


def _mark_rect(blocked: np.ndarray, rect: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = rect
    blocked[r0 : r1 + 1, c0 : c1 + 1] = True


def _add_arms(mask: np.ndarray, rng: random.Random, max_arms: int) -> np.ndarray:
    h, w = mask.shape
    canvas = np.zeros((h + 4, w + 4), dtype=bool)
    canvas[2 : 2 + h, 2 : 2 + w] = mask
    sides = ["top", "bottom", "left", "right"]
    rng.shuffle(sides)
    for side in sides[: rng.randint(0, max_arms)]:
        arm_len = rng.randint(1, 2)
        if side in ("top", "bottom"):
            arm_w = rng.randint(1, min(3, w))
            start = 2 + rng.randint(0, w - arm_w)
            if side == "top":
                canvas[2 - arm_len : 2, start : start + arm_w] = True
            else:
                canvas[2 + h : 2 + h + arm_len, start : start + arm_w] = True
        else:
            arm_h = rng.randint(1, min(3, h))
            start = 2 + rng.randint(0, h - arm_h)
            if side == "left":
                canvas[start : start + arm_h, 2 - arm_len : 2] = True
            else:
                canvas[start : start + arm_h, 2 + w : 2 + w + arm_len] = True
    return _crop_mask(canvas)


def _carve_border_notches(
    mask: np.ndarray, target_holes: int, rng: random.Random, tries: int = 4
) -> np.ndarray:
    cur = mask.copy()
    for _ in range(tries):
        h, w = cur.shape
        if h < 3 or w < 3:
            break
        trial = cur.copy()
        corner = rng.choice(["tl", "tr", "bl", "br"])
        notch_h = rng.randint(1, min(2, h - 1))
        notch_w = rng.randint(1, min(2, w - 1))
        if corner == "tl":
            trial[:notch_h, :notch_w] = False
        elif corner == "tr":
            trial[:notch_h, w - notch_w :] = False
        elif corner == "bl":
            trial[h - notch_h :, :notch_w] = False
        else:
            trial[h - notch_h :, w - notch_w :] = False
        trial = _crop_mask(trial)
        if trial.size == 0 or trial.sum() < 4:
            continue
        if _is_connected(trial) and _count_holes(trial) == target_holes:
            cur = trial
    return cur


def _build_object_mask(hole_count: int, rng: random.Random) -> np.ndarray:
    for _ in range(80):
        if hole_count == 0:
            mask = np.ones((rng.randint(3, 6), rng.randint(3, 6)), dtype=bool)
        else:
            hole_rows = rng.randint(1, hole_count)
            hole_cols = (hole_count + hole_rows - 1) // hole_rows
            pad_top = rng.randint(0, 2)
            pad_bottom = rng.randint(0, 2)
            pad_left = rng.randint(0, 2)
            pad_right = rng.randint(0, 2)
            mask = np.ones(
                (
                    2 * hole_rows + 1 + pad_top + pad_bottom,
                    2 * hole_cols + 1 + pad_left + pad_right,
                ),
                dtype=bool,
            )
            slots = [
                (pad_top + 1 + 2 * rr, pad_left + 1 + 2 * cc)
                for rr in range(hole_rows)
                for cc in range(hole_cols)
            ]
            rng.shuffle(slots)
            for r, c in slots[:hole_count]:
                mask[r, c] = False

        mask = _add_arms(mask, rng, max_arms=2)
        mask = _carve_border_notches(mask, hole_count, rng, tries=rng.randint(1, 4))
        if _is_connected(mask) and _count_holes(mask) == hole_count and mask.sum() >= 4:
            return mask
    raise RuntimeError(f"failed to build object with {hole_count} holes")


def _place_objects(
    grid: np.ndarray,
    legend_mask: np.ndarray,
    specs: list[dict[str, object]],
    legend_on_top: bool,
    rng: random.Random,
) -> dict[int, tuple[int, int]] | None:
    h, w = grid.shape
    blocked = np.zeros((h, w), dtype=bool)
    blocked[legend_mask] = True
    if legend_on_top:
        blocked[:2, :] = True
    else:
        blocked[h - 2 :, :] = True

    top_buffer = 2 if legend_on_top else 1
    bottom_buffer = 1 if legend_on_top else 2
    placements: dict[int, tuple[int, int]] = {}
    order = sorted(
        range(len(specs)),
        key=lambda i: int(specs[i]["priority"]),  # type: ignore[index]
        reverse=True,
    )

    for idx in order:
        mask = specs[idx]["mask"]  # type: ignore[index]
        holes = int(specs[idx]["holes"])  # type: ignore[index]
        mh, mw = mask.shape  # type: ignore[union-attr]
        need_h = max(mh, holes)
        need_w = max(mw, holes)
        row_min = top_buffer
        row_max = h - bottom_buffer - need_h
        col_min = 1
        col_max = w - 2 - need_w + 1
        if row_max < row_min or col_max < col_min:
            return None

        placed = False
        for _ in range(200):
            top = rng.randint(row_min, row_max)
            left = rng.randint(col_min, col_max)

            bbox_zone = _expanded_rect(top, left, mh, mw, 1, h, w)
            if not _rect_is_free(blocked, bbox_zone):
                continue

            if holes > 0:
                square_zone = _expanded_rect(top, left, holes, holes, 1, h, w)
                if not _rect_is_free(blocked, square_zone):
                    continue

            placements[idx] = (top, left)
            _mark_rect(blocked, bbox_zone)
            if holes > 0:
                _mark_rect(blocked, square_zone)
            placed = True
            break

        if not placed:
            return None

    return placements


def _expected_output(
    shape: tuple[int, int],
    legend_positions: list[tuple[int, int]],
    legend_colors: list[int],
    specs: list[dict[str, object]],
    placements: dict[int, tuple[int, int]],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for (r, c), color in zip(legend_positions, legend_colors):
        out[r, c] = int(color)
    for idx, spec in enumerate(specs):
        holes = int(spec["holes"])
        if holes == 0:
            continue
        top, left = placements[idx]
        out[top : top + holes, left : left + holes] = int(legend_colors[holes])
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    legend_len = rng.randint(4, 5)
    n_objects = rng.randint(3, 5)
    n_zero = rng.randint(1, n_objects - 1)
    hole_counts = [0] * n_zero + [rng.randint(2, legend_len - 1)]
    while len(hole_counts) < n_objects:
        hole_counts.append(rng.randint(0, legend_len - 1))
    rng.shuffle(hole_counts)

    specs: list[dict[str, object]] = []
    for holes in hole_counts:
        mask = _build_object_mask(holes, rng)
        mh, mw = mask.shape
        specs.append(
            {
                "holes": holes,
                "mask": mask,
                "priority": max(mh, mw, holes) * (mh + mw + max(1, holes)),
            }
        )

    max_need_h = max(max(spec["mask"].shape[0], int(spec["holes"])) for spec in specs)  # type: ignore[index]
    max_need_w = max(max(spec["mask"].shape[1], int(spec["holes"])) for spec in specs)  # type: ignore[index]
    grid_h = rng.randint(max(14, max_need_h + 4), 22)
    grid_w = rng.randint(max(legend_len + 6, max_need_w + 4), 24)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    legend_row = 0 if rng.random() < 0.5 else grid_h - 1
    start_col = rng.randint(1, grid_w - legend_len - 2)
    legend_colors = rng.sample(COLOR_POOL, legend_len)
    legend_positions = [(legend_row, start_col + i) for i in range(legend_len)]
    legend_mask = np.zeros((grid_h, grid_w), dtype=bool)
    for (r, c), color in zip(legend_positions, legend_colors):
        legend_mask[r, c] = True
        grid[r, c] = int(color)

    placements = _place_objects(grid, legend_mask, specs, legend_row == 0, rng)
    if placements is None:
        return None

    for idx, spec in enumerate(specs):
        top, left = placements[idx]
        mask = spec["mask"]  # type: ignore[index]
        mh, mw = mask.shape  # type: ignore[union-attr]
        for rr in range(mh):
            for cc in range(mw):
                if mask[rr, cc]:
                    grid[top + rr, left + cc] = GREY

    if len(_components((grid != BG) & ~legend_mask)) != n_objects:
        return None

    expected = _expected_output(grid.shape, legend_positions, legend_colors, specs, placements)
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GEN_ATTEMPTS} attempts")
