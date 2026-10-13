from __future__ import annotations

import random

import numpy as np


_DIRS = ("up", "down", "left", "right")
_GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("empty mask")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_mask(mask)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _looks_like_frame(mask: np.ndarray) -> bool:
    if mask.size == 0 or not mask.any():
        return False
    cropped = _crop_mask(np.asarray(mask, dtype=bool))
    h, w = map(int, cropped.shape)
    if h < 3 or w < 3:
        return False
    border = _border_mask(h, w)
    if np.any(cropped & ~border):
        return False
    missing = np.argwhere(border & ~cropped)
    if len(missing) > 1:
        return False
    if len(missing) == 1:
        mr, mc = map(int, missing[0])
        if (mr in (0, h - 1)) and (mc in (0, w - 1)):
            return False
    return int(cropped.sum()) == int(border.sum()) - int(len(missing))


def _frame_candidate(grid: np.ndarray, color: int) -> dict | None:
    coords = np.argwhere(grid == int(color))
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None

    local = np.zeros((h, w), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    border = _border_mask(h, w)
    if np.any(local & ~border):
        return None

    missing = np.argwhere(border & ~local)
    if len(missing) not in (0, 1):
        return None
    if int(local.sum()) != int(border.sum()) - int(len(missing)):
        return None
    if not local[0, :].any() or not local[-1, :].any() or not local[:, 0].any() or not local[:, -1].any():
        return None

    gap_side = None
    if len(missing) == 1:
        mr, mc = map(int, missing[0])
        if mr == 0 and 0 < mc < w - 1:
            gap_side = "up"
        elif mr == h - 1 and 0 < mc < w - 1:
            gap_side = "down"
        elif mc == 0 and 0 < mr < h - 1:
            gap_side = "left"
        elif mc == w - 1 and 0 < mr < h - 1:
            gap_side = "right"
        else:
            return None

    return {
        "color": int(color),
        "bbox": (r0, c0, r1, c1),
        "gap_side": gap_side,
        "center_key": (int(r0 + r1), int(c0 + c1)),
        "area": int(h * w),
    }


def _step_gap(inner: dict, outer: dict) -> tuple[int, int, int, int]:
    ir0, ic0, ir1, ic1 = inner["bbox"]
    or0, oc0, or1, oc1 = outer["bbox"]
    top = int(ir0 - or0 - 1)
    bottom = int(or1 - ir1 - 1)
    left = int(ic0 - oc0 - 1)
    right = int(oc1 - ic1 - 1)
    return top, bottom, left, right


def _valid_frame_step(inner: dict, outer: dict) -> bool:
    if inner["center_key"] != outer["center_key"]:
        return False
    top, bottom, left, right = _step_gap(inner, outer)
    return top >= 2 and bottom >= 2 and left >= 2 and right >= 2 and top == bottom and left == right


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    colors = [int(v) for v in np.unique(grid) if int(v) != int(bg)]
    candidates = [cand for color in colors if (cand := _frame_candidate(grid, color)) is not None]
    if not candidates:
        raise ValueError("no frame candidates")

    by_center: dict[tuple[int, int], list[dict]] = {}
    for cand in candidates:
        by_center.setdefault(cand["center_key"], []).append(cand)

    best: list[dict] = []
    for group in by_center.values():
        ordered = sorted(group, key=lambda item: (item["area"], item["bbox"]))

        def dfs(start_idx: int, chain: list[dict]) -> None:
            nonlocal best
            if len(chain) > len(best) or (len(chain) == len(best) and chain and chain[-1]["area"] > best[-1]["area"]):
                best = list(chain)
            if len(chain) >= 4:
                return
            for nxt_idx in range(start_idx + 1, len(ordered)):
                nxt = ordered[nxt_idx]
                if nxt["gap_side"] is None:
                    continue
                if _valid_frame_step(chain[-1], nxt):
                    chain.append(nxt)
                    dfs(nxt_idx, chain)
                    chain.pop()

        for idx, cand in enumerate(ordered):
            if cand["gap_side"] is not None:
                continue
            dfs(idx, [cand])

    if len(best) < 2:
        raise ValueError("failed to recover nested frame chain")
    return best


def _extract_pattern(grid: np.ndarray, bg: int, frames: list[dict]) -> tuple[tuple[int, int, int, int], np.ndarray]:
    inner = frames[0]["bbox"]
    r0, c0, r1, c1 = inner
    mask = np.zeros_like(grid, dtype=bool)
    frame_colors = {int(frame["color"]) for frame in frames}
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    inside = (interior != int(bg)) & ~np.isin(interior, list(frame_colors))
    if not inside.any():
        raise ValueError("missing inner pattern")
    mask[r0 + 1 : r1, c0 + 1 : c1] = inside
    pr0, pc0, pr1, pc1 = _bbox_from_mask(mask)
    shape = np.asarray(mask[pr0 : pr1 + 1, pc0 : pc1 + 1], dtype=bool)
    return (pr0, pc0, pr1, pc1), shape


def _project_bbox(current_bbox: tuple[int, int, int, int], frame_bbox: tuple[int, int, int, int], direction: str) -> tuple[int, int, int, int]:
    pr0, pc0, pr1, pc1 = current_bbox
    fr0, fc0, fr1, fc1 = frame_bbox
    height = int(pr1 - pr0 + 1)
    width = int(pc1 - pc0 + 1)

    if direction == "up":
        nr1 = int(fr0 - 1)
        nr0 = int(nr1 - height + 1)
        nc0 = int(pc0)
        nc1 = int(pc1)
    elif direction == "down":
        nr0 = int(fr1 + 1)
        nr1 = int(nr0 + height - 1)
        nc0 = int(pc0)
        nc1 = int(pc1)
    elif direction == "left":
        nc1 = int(fc0 - 1)
        nc0 = int(nc1 - width + 1)
        nr0 = int(pr0)
        nr1 = int(pr1)
    elif direction == "right":
        nc0 = int(fc1 + 1)
        nc1 = int(nc0 + width - 1)
        nr0 = int(pr0)
        nr1 = int(pr1)
    else:
        raise ValueError(f"bad direction: {direction}")
    return nr0, nc0, nr1, nc1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect a concentric frame chain and relay the innermost pattern outward.

    The pattern inside the innermost frame is copied ring by ring. At each step,
    the next outer frame's gap side gives the direction, while the copy drawn in
    the intervening ring uses the current frame's color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frames = _find_frames(g, bg)
    current_bbox, current_shape = _extract_pattern(g, bg, frames)

    out = g.copy()
    for idx in range(len(frames) - 1):
        current_frame = frames[idx]
        next_frame = frames[idx + 1]
        direction = next_frame["gap_side"]
        if direction is None:
            raise ValueError("outer frame is missing a gap direction")

        new_bbox = _project_bbox(current_bbox, current_frame["bbox"], direction)
        nr0, nc0, nr1, nc1 = new_bbox
        or0, oc0, or1, oc1 = next_frame["bbox"]
        if not (or0 < nr0 <= nr1 < or1 and oc0 < nc0 <= nc1 < oc1):
            raise ValueError("projection escapes its containing frame")

        out[nr0 : nr1 + 1, nc0 : nc1 + 1][current_shape] = int(current_frame["color"])
        current_bbox = new_bbox

    return out


def _random_connected_mask(rng: random.Random) -> np.ndarray:
    for _ in range(100):
        h = rng.randint(1, 5)
        w = rng.randint(1, 5)
        area_max = h * w
        area_min = max(1, (area_max + 3) // 4)
        area = rng.randint(area_min, area_max)
        start = (rng.randrange(h), rng.randrange(w))
        pts = {start}
        while len(pts) < area:
            frontier = []
            for r, c in pts:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in pts:
                        frontier.append((nr, nc))
            if not frontier:
                break
            pts.add(rng.choice(frontier))
        mask = np.zeros((h, w), dtype=bool)
        for r, c in pts:
            mask[r, c] = True
        mask = _crop_mask(mask)
        if mask.any() and not _looks_like_frame(mask):
            return mask
    raise RuntimeError("failed to sample pattern mask")


def _paint_pattern(mask: np.ndarray, colors: list[int], rng: random.Random) -> np.ndarray:
    painted = np.zeros(mask.shape, dtype=int)
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    rng.shuffle(coords)
    used_colors = [int(c) for c in colors[: min(len(colors), len(coords))]]
    for idx, color in enumerate(used_colors):
        r, c = coords[idx]
        painted[r, c] = int(color)
    for r, c in coords[len(used_colors) :]:
        painted[r, c] = int(rng.choice(colors))
    return painted


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int, bg: int, gap_side: str | None, rng: random.Random) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)

    if gap_side is None:
        return
    if gap_side == "up":
        grid[r0, rng.randint(c0 + 1, c1 - 1)] = int(bg)
    elif gap_side == "down":
        grid[r1, rng.randint(c0 + 1, c1 - 1)] = int(bg)
    elif gap_side == "left":
        grid[rng.randint(r0 + 1, r1 - 1), c0] = int(bg)
    elif gap_side == "right":
        grid[rng.randint(r0 + 1, r1 - 1), c1] = int(bg)
    else:
        raise ValueError(f"bad gap side: {gap_side}")


def _make_attempt(rng: random.Random) -> np.ndarray | None:
    bg = int(rng.randrange(10))
    n_frames = int(rng.choice((2, 3, 4)))
    pattern_mask = _random_connected_mask(rng)
    ph, pw = map(int, pattern_mask.shape)

    top_pad = rng.randint(1, 3)
    bottom_pad = rng.randint(1, 3)
    left_pad = rng.randint(1, 3)
    right_pad = rng.randint(1, 3)
    inner_h = int(ph + top_pad + bottom_pad + 2)
    inner_w = int(pw + left_pad + right_pad + 2)

    gap_dirs = [str(rng.choice(_DIRS)) for _ in range(n_frames - 1)]
    ring_thicknesses = []
    for direction in gap_dirs:
        needed = ph if direction in ("up", "down") else pw
        ring_thicknesses.append(int(rng.randint(max(2, needed), max(2, needed) + 2)))

    bboxes = [(0, 0, inner_h - 1, inner_w - 1)]
    for thickness in ring_thicknesses:
        delta = int(thickness + 1)
        r0, c0, r1, c1 = bboxes[-1]
        bboxes.append((r0 - delta, c0 - delta, r1 + delta, c1 + delta))

    outer_r0, outer_c0, outer_r1, outer_c1 = bboxes[-1]
    struct_h = int(outer_r1 - outer_r0 + 1)
    struct_w = int(outer_c1 - outer_c0 + 1)
    margin_top = rng.randint(1, 5)
    margin_bottom = rng.randint(1, 5)
    margin_left = rng.randint(1, 5)
    margin_right = rng.randint(1, 5)
    shift_r = int(margin_top - outer_r0)
    shift_c = int(margin_left - outer_c0)
    bboxes = [(r0 + shift_r, c0 + shift_c, r1 + shift_r, c1 + shift_c) for r0, c0, r1, c1 in bboxes]

    grid_h = int(struct_h + margin_top + margin_bottom)
    grid_w = int(struct_w + margin_left + margin_right)
    grid = np.full((grid_h, grid_w), bg, dtype=int)

    pool = [color for color in range(10) if color != bg]
    pattern_color_count = rng.randint(1, min(3, int(pattern_mask.sum())))
    chosen = rng.sample(pool, n_frames + pattern_color_count)
    frame_colors = [int(color) for color in chosen[:n_frames]]
    pattern_colors = [int(color) for color in chosen[n_frames:]]

    for idx, bbox in enumerate(bboxes):
        gap_side = None if idx == 0 else gap_dirs[idx - 1]
        _draw_frame(grid, bbox, frame_colors[idx], bg, gap_side, rng)

    inner_r0, inner_c0, _, _ = bboxes[0]
    pr0 = int(inner_r0 + 1 + top_pad)
    pc0 = int(inner_c0 + 1 + left_pad)
    pattern = _paint_pattern(pattern_mask, pattern_colors, rng)
    patch = grid[pr0 : pr0 + ph, pc0 : pc0 + pw]
    patch[pattern_mask] = pattern[pattern_mask]

    if _get_bg(grid) != bg:
        return None
    try:
        frames = _find_frames(grid, bg)
        if len(frames) != n_frames:
            return None
        if [int(frame["color"]) for frame in frames] != frame_colors:
            return None
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_GEN_ATTEMPTS):
        grid = _make_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame projection input")
