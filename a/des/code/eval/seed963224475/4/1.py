from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
GREY = 5
FRAME_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR_NAMES = ("up", "down", "left", "right")


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            comp: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                comp.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(points)
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return min(rs), min(cs), max(rs), max(cs)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _rotate_cw(mask: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(mask, -(turns % 4))


def _projection_origin(
    frame_bbox: tuple[int, int, int, int],
    grey_bbox: tuple[int, int, int, int],
    proj_shape: tuple[int, int],
    gap_side: str,
) -> tuple[int, int]:
    r0, c0, r1, c1 = frame_bbox
    gr0, gc0, _gr1, _gc1 = grey_bbox
    ph, pw = proj_shape
    if gap_side == "up":
        return r0 - ph, gc0
    if gap_side == "down":
        return r1 + 1, gc0
    if gap_side == "left":
        return gr0, c0 - pw
    if gap_side == "right":
        return gr0, c1 + 1
    raise ValueError(gap_side)


def _find_legend(grid: np.ndarray) -> tuple[str, list[int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    best: tuple[int, str, list[tuple[int, int]]] | None = None

    for side, row0 in (("top", 0), ("bottom", h - 2)):
        if h < 2:
            continue
        blocks: list[tuple[int, int]] = []
        c = 0
        while c <= w - 2:
            patch = g[row0 : row0 + 2, c : c + 2]
            vals = np.unique(patch)
            if len(vals) == 1:
                color = int(vals[0])
                if color not in (BG, GREY):
                    left_clear = c == 0 or np.all(g[row0 : row0 + 2, c - 1] == BG)
                    right_clear = c + 2 >= w or np.all(g[row0 : row0 + 2, c + 2] == BG)
                    isolated = True
                    for rr in range(max(0, row0 - 1), min(h, row0 + 3)):
                        for cc in range(max(0, c - 1), min(w, c + 3)):
                            if row0 <= rr < row0 + 2 and c <= cc < c + 2:
                                continue
                            if int(g[rr, cc]) == color:
                                isolated = False
                    if left_clear and right_clear and isolated:
                        blocks.append((c, color))
                        c += 2
                        continue
            c += 1
        if len(blocks) >= 2:
            cand = (len(blocks), side, blocks)
            if best is None or cand[0] > best[0]:
                best = cand

    assert best is not None, "legend must be a top/bottom sequence of 2x2 blocks"
    _, side, blocks = best
    legend_mask = np.zeros_like(g, dtype=bool)
    row0 = 0 if side == "top" else g.shape[0] - 2
    for c, _color in blocks:
        legend_mask[row0 : row0 + 2, c : c + 2] = True
    colors = [color for _c, color in sorted(blocks)]
    assert len(colors) == len(set(colors))
    return side, colors, legend_mask


def _frame_specs(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    _side, legend_colors, legend_mask = _find_legend(g)
    specs: list[dict] = []
    for order, color in enumerate(legend_colors, start=1):
        pts = np.argwhere((g == color) & (~legend_mask))
        assert len(pts) > 0, f"missing frame for color {color}"
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        r0 = int(r0)
        c0 = int(c0)
        r1 = int(r1)
        c1 = int(c1)
        assert r1 - r0 >= 3 and c1 - c0 >= 3

        gap_hits: list[tuple[str, int, int]] = []
        for c in range(c0, c1 + 1):
            if int(g[r0, c]) != color:
                gap_hits.append(("up", r0, c))
            if int(g[r1, c]) != color:
                gap_hits.append(("down", r1, c))
        for r in range(r0 + 1, r1):
            if int(g[r, c0]) != color:
                gap_hits.append(("left", r, c0))
            if int(g[r, c1]) != color:
                gap_hits.append(("right", r, c1))
        assert len(gap_hits) == 1, f"frame color {color} must have exactly one 1-pixel gap"
        gap_side, gap_r, gap_c = gap_hits[0]
        assert gap_side in DIR_NAMES
        if gap_side in ("up", "down"):
            assert c0 < gap_c < c1
        else:
            assert r0 < gap_r < r1

        perimeter_ok = True
        for c in range(c0, c1 + 1):
            if (r0, c) != (gap_r, gap_c) and int(g[r0, c]) != color:
                perimeter_ok = False
            if (r1, c) != (gap_r, gap_c) and int(g[r1, c]) != color:
                perimeter_ok = False
        for r in range(r0 + 1, r1):
            if (r, c0) != (gap_r, gap_c) and int(g[r, c0]) != color:
                perimeter_ok = False
            if (r, c1) != (gap_r, gap_c) and int(g[r, c1]) != color:
                perimeter_ok = False
        assert perimeter_ok

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        grey_pts_rel = np.argwhere(inner == GREY)
        assert len(grey_pts_rel) > 0
        grey_abs = [(r0 + 1 + int(r), c0 + 1 + int(c)) for r, c in grey_pts_rel]
        grey_bbox = _bbox(grey_abs)
        gr0, gc0, gr1, gc1 = grey_bbox
        grey_mask = g[gr0 : gr1 + 1, gc0 : gc1 + 1] == GREY
        assert len(_components(grey_mask)) == 1

        specs.append(
            {
                "color": color,
                "order": order,
                "turns": order % 4,
                "frame_bbox": (r0, c0, r1, c1),
                "gap_side": gap_side,
                "grey_bbox": grey_bbox,
                "grey_mask": grey_mask,
            }
        )
    return specs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the edge color band, reveal each grey motif inside its colored frame,
    then project a rotated copy outward through the frame's gap direction.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    specs = _frame_specs(g)

    for spec in specs:
        color = int(spec["color"])
        r0, c0, r1, c1 = spec["frame_bbox"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[interior == color] = BG

    for spec in specs:
        color = int(spec["color"])
        rotated = _rotate_cw(spec["grey_mask"], int(spec["turns"]))
        pr0, pc0 = _projection_origin(
            spec["frame_bbox"],
            spec["grey_bbox"],
            rotated.shape,
            spec["gap_side"],
        )
        rows, cols = np.where(rotated)
        for rr, cc in zip(rows.tolist(), cols.tolist(), strict=False):
            out[pr0 + rr, pc0 + cc] = color

    return out


def _expand(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for _ in range(radius):
        grown = out.copy()
        rows, cols = np.where(out)
        for r, c in zip(rows.tolist(), cols.tolist(), strict=False):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        grown[nr, nc] = True
        out = grown
    return out


def _rect_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    h, w = shape
    r0, c0, r1, c1 = bbox
    mask = np.zeros((h, w), dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int, gap_side: str, gap_pos: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    if gap_side == "up":
        grid[r0, gap_pos] = BG
    elif gap_side == "down":
        grid[r1, gap_pos] = BG
    elif gap_side == "left":
        grid[gap_pos, c0] = BG
    else:
        grid[gap_pos, c1] = BG


def _random_connected_mask(h: int, w: int, turns: int) -> np.ndarray:
    area = h * w
    min_cells = max(3, area // 2)
    max_cells = area - 1
    for _ in range(200):
        target = random.randint(min_cells, max_cells)
        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True
        frontier = set()

        def add_frontier(cell: tuple[int, int]) -> None:
            r, c = cell
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))

        add_frontier(start)
        while frontier and int(mask.sum()) < target:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            mask[cell] = True
            add_frontier(cell)

        if int(mask.sum()) != target:
            continue
        if not mask[0, :].any() or not mask[-1, :].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if turns % 4 != 0 and np.array_equal(mask, _rotate_cw(mask, turns)):
            continue
        return mask
    raise RuntimeError("failed to sample grey motif")


def _legend_cells(shape: tuple[int, int], side: str, start_col: int, colors: list[int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    grid = np.full((h, w), BG, dtype=int)
    mask = np.zeros((h, w), dtype=bool)
    row0 = 0 if side == "top" else h - 2
    c = start_col
    for color in colors:
        grid[row0 : row0 + 2, c : c + 2] = color
        mask[row0 : row0 + 2, c : c + 2] = True
        c += 3
    return grid, mask


def _sample_object_count() -> int:
    return random.choice((2, 3, 4))


def _sample_grid_shape(n_frames: int) -> tuple[int, int]:
    if n_frames == 2:
        return random.randint(18, 24), random.randint(20, 28)
    if n_frames == 3:
        return random.randint(22, 28), random.randint(24, 30)
    return random.randint(24, 30), random.randint(26, 32)


def generate_input() -> np.ndarray:
    """
    Randomly generate frames with single-pixel gaps, hidden grey motifs, and a
    top/bottom color band whose left-to-right order encodes clockwise rotations.
    """
    for _ in range(400):
        n_frames = _sample_object_count()
        h, w = _sample_grid_shape(n_frames)
        grid = np.full((h, w), BG, dtype=int)

        colors = random.sample(FRAME_COLORS, n_frames)
        legend_side = random.choice(("top", "bottom"))
        legend_width = 3 * n_frames - 1
        if w - legend_width - 2 < 1:
            continue
        legend_start = random.randint(1, w - legend_width - 1)
        legend_grid, legend_mask = _legend_cells((h, w), legend_side, legend_start, colors)
        grid[legend_mask] = legend_grid[legend_mask]
        blocked = _expand(legend_mask, 1)

        ok = True
        for order, color in enumerate(colors, start=1):
            turns = order % 4
            gap_side = random.choice(DIR_NAMES)

            placed = False
            for _try in range(250):
                motif_h = random.randint(2, 4)
                motif_w = random.randint(2, 5)
                grey_mask = _random_connected_mask(motif_h, motif_w, turns)
                motif_h, motif_w = grey_mask.shape
                rotated = _rotate_cw(grey_mask, turns)
                proj_h, proj_w = rotated.shape

                interior_h = motif_h + random.randint(1, 3)
                interior_w = motif_w + random.randint(1, 3)
                off_r = random.randint(0, interior_h - motif_h)
                off_c = random.randint(0, interior_w - motif_w)
                frame_h = interior_h + 2
                frame_w = interior_w + 2

                if frame_h >= h - 2 or frame_w >= w - 2:
                    continue

                r0 = random.randint(1, h - frame_h - 1)
                c0 = random.randint(1, w - frame_w - 1)
                r1 = r0 + frame_h - 1
                c1 = c0 + frame_w - 1
                grey_r0 = r0 + 1 + off_r
                grey_c0 = c0 + 1 + off_c
                grey_r1 = grey_r0 + motif_h - 1
                grey_c1 = grey_c0 + motif_w - 1

                proj_r0, proj_c0 = _projection_origin(
                    (r0, c0, r1, c1),
                    (grey_r0, grey_c0, grey_r1, grey_c1),
                    (proj_h, proj_w),
                    gap_side,
                )
                proj_r1 = proj_r0 + proj_h - 1
                proj_c1 = proj_c0 + proj_w - 1
                if proj_r0 < 0 or proj_c0 < 0 or proj_r1 >= h or proj_c1 >= w:
                    continue

                frame_mask = _rect_mask((h, w), (r0, c0, r1, c1))
                proj_mask = _rect_mask((h, w), (proj_r0, proj_c0, proj_r1, proj_c1))
                region = frame_mask | proj_mask
                if np.any(blocked & region):
                    continue

                free_inside: list[tuple[int, int]] = []
                for rr in range(r0 + 1, r1):
                    for cc in range(c0 + 1, c1):
                        is_grey = False
                        if grey_r0 <= rr <= grey_r1 and grey_c0 <= cc <= grey_c1:
                            is_grey = bool(grey_mask[rr - grey_r0, cc - grey_c0])
                        if not is_grey:
                            free_inside.append((rr, cc))
                if not free_inside:
                    continue

                distractor_cells = list(free_inside)
                if not distractor_cells:
                    continue
                dcount = random.randint(1, min(3, len(distractor_cells)))
                gap_pos = random.randint(c0 + 1, c1 - 1) if gap_side in ("up", "down") else random.randint(r0 + 1, r1 - 1)

                _draw_frame(grid, (r0, c0, r1, c1), color, gap_side, gap_pos)
                for rr in range(motif_h):
                    for cc in range(motif_w):
                        if grey_mask[rr, cc]:
                            grid[grey_r0 + rr, grey_c0 + cc] = GREY
                for rr, cc in random.sample(distractor_cells, dcount):
                    grid[rr, cc] = color

                blocked |= _expand(region, 1)
                placed = True
                break

            if not placed:
                ok = False
                break

        if not ok:
            continue

        try:
            specs = _frame_specs(grid)
            if len(specs) != n_frames:
                continue
            out = main(grid)
        except Exception:
            continue

        if np.array_equal(out, grid):
            continue
        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
