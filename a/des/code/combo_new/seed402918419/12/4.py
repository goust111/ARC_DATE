from __future__ import annotations

import random
import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GENERATION_ATTEMPTS = 400


def _same_color_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                }
            )
    return comps


def _is_square_frame(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    side_r = int(r1 - r0 + 1)
    side_c = int(c1 - c0 + 1)
    if side_r != side_c or side_r < 4:
        return False
    if int(comp["coords"].shape[0]) != 4 * side_r - 4:
        return False

    mask = np.zeros((side_r, side_r), dtype=bool)
    rel = comp["coords"] - np.array([[r0, c0]])
    mask[rel[:, 0], rel[:, 1]] = True
    border = np.zeros_like(mask)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.array_equal(mask, border))


def _extract_frames(grid: np.ndarray) -> list[dict]:
    frames = []
    for comp in _same_color_components(grid):
        if _is_square_frame(comp):
            frames.append(comp)
    assert frames, "no square frames detected"
    return frames


def _frame_parents(frames: list[dict]) -> list[int | None]:
    parents: list[int | None] = [None] * len(frames)
    for i, child in enumerate(frames):
        cr0, cc0, cr1, cc1 = child["bbox"]
        best = None
        for j, parent in enumerate(frames):
            if i == j:
                continue
            pr0, pc0, pr1, pc1 = parent["bbox"]
            if pr0 < cr0 and pc0 < cc0 and pr1 > cr1 and pc1 > cc1:
                side = int(pr1 - pr0 + 1)
                candidate = (side, j)
                if best is None or candidate < best:
                    best = candidate
        parents[i] = None if best is None else best[1]
    return parents


def _seed_patch(grid: np.ndarray, frame: dict) -> tuple[np.ndarray, np.ndarray]:
    r0, c0, r1, c1 = frame["bbox"]
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    points = np.argwhere(interior != BG)
    assert points.size > 0, "innermost frame has no seed"

    sr0 = int(points[:, 0].min())
    sc0 = int(points[:, 1].min())
    sr1 = int(points[:, 0].max())
    sc1 = int(points[:, 1].max())
    patch = interior[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()

    abs_points = points + np.array([[r0 + 1, c0 + 1]])
    return patch, abs_points


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each seed by the number of nested square frames around it, then redraw it outside the outer frame's top-right corner.

    Input structure:
    - The background is black.
    - The grid contains one or more non-overlapping hollow square frames with one-pixel borders; some frames are nested into chains.
    - Each nested chain ends with an innermost frame whose interior contains one small monochrome 2x2 or 3x3 asymmetric seed pattern.

    Rule:
    - For each seed, count how many square frames enclose it in its nested chain.
    - Erase the original seed, rotate its full 2x2 or 3x3 patch clockwise by K x 90 degrees, where K is that frame count.
    - Draw the rotated patch outside the outermost frame at the top-right corner, with the patch's bottom-left cell aligned to the first diagonal pixel outside that corner.

    Key insight:
    - The transformation parameter is not a color or size; it is the nesting depth of the frame chain that contains the seed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = g.copy()

    frames = _extract_frames(g)
    parents = _frame_parents(frames)
    child_count = [0] * len(frames)
    for parent in parents:
        if parent is not None:
            child_count[parent] += 1

    for leaf_idx, frame in enumerate(frames):
        if child_count[leaf_idx] != 0:
            continue

        patch, seed_points = _seed_patch(g, frame)
        out[seed_points[:, 0], seed_points[:, 1]] = BG

        depth = 1
        root_idx = leaf_idx
        while parents[root_idx] is not None:
            root_idx = parents[root_idx]
            depth += 1

        rotated = np.rot90(patch, -depth)
        rr0, rc0, rr1, rc1 = frames[root_idx]["bbox"]
        dst_r0 = int(rr0 - rotated.shape[0])
        dst_c0 = int(rc1 + 1)
        dst_r1 = int(dst_r0 + rotated.shape[0])
        dst_c1 = int(dst_c0 + rotated.shape[1])
        assert 0 <= dst_r0 and dst_r1 <= g.shape[0]
        assert 0 <= dst_c0 and dst_c1 <= g.shape[1]

        mask = rotated != BG
        target = out[dst_r0:dst_r1, dst_c0:dst_c1]
        assert not bool(np.any(mask & (target != BG))), "output seeds overlap"
        target[mask] = rotated[mask]

    return out


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _sample_seed_mask(size: int) -> np.ndarray:
    while True:
        mask = np.zeros((size, size), dtype=bool)
        for r in range(size):
            for c in range(size):
                mask[r, c] = random.random() < 0.5

        if int(mask.sum()) < 2 or int(mask.sum()) == size * size:
            continue
        if not mask[0, :].any() or not mask[-1, :].any():
            continue
        if not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if any(np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3)):
            continue
        return mask


def _sample_group() -> dict:
    depth = random.randint(1, 3)
    seed_size = random.randint(2, 3)
    seed_mask = _sample_seed_mask(seed_size)

    leaf_inner = seed_size + random.randint(0, 1)
    leaf_size = leaf_inner + 2
    seed_r = random.randint(0, leaf_inner - seed_size)
    seed_c = random.randint(0, leaf_inner - seed_size)

    boxes = [(0, 0, leaf_size)]
    cur_top = 0
    cur_left = 0
    cur_size = leaf_size
    for _ in range(depth - 1):
        margin_sum = random.randint(2, 3)
        top_gap = random.randint(1, margin_sum - 1)
        left_gap = random.randint(1, margin_sum - 1)
        parent_size = cur_size + 2 + margin_sum
        parent_top = cur_top - (1 + top_gap)
        parent_left = cur_left - (1 + left_gap)
        boxes.append((parent_top, parent_left, parent_size))
        cur_top, cur_left, cur_size = parent_top, parent_left, parent_size

    root_top, root_left, root_size = boxes[-1]
    shift_r = -root_top + seed_size
    shift_c = -root_left
    shifted_boxes = []
    for top, left, side in boxes:
        top += shift_r
        left += shift_c
        shifted_boxes.append((top, left, side))

    leaf_top, leaf_left, _leaf_side = shifted_boxes[0]
    seed_patch = np.full((seed_size, seed_size), BG, dtype=int)
    colors = random.sample(PALETTE, depth + 1)
    seed_color = colors[-1]
    seed_patch[seed_mask] = seed_color
    seed_bbox = (
        leaf_top + 1 + seed_r,
        leaf_left + 1 + seed_c,
        leaf_top + seed_r + seed_size,
        leaf_left + seed_c + seed_size,
    )

    frames = []
    for i, (top, left, side) in enumerate(shifted_boxes):
        frames.append(
            {
                "bbox": (top, left, top + side - 1, left + side - 1),
                "color": colors[i],
            }
        )

    root_bbox = frames[-1]["bbox"]
    envelope = (seed_size + root_size, root_size + seed_size)
    return {
        "depth": depth,
        "seed_size": seed_size,
        "seed_patch": seed_patch,
        "seed_bbox": seed_bbox,
        "frames": frames,
        "root_bbox": root_bbox,
        "envelope": envelope,
    }


def _rectangles_do_not_touch(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    return (
        ar + ah + 1 <= br
        or br + bh + 1 <= ar
        or ac + aw + 1 <= bc
        or bc + bw + 1 <= ac
    )


def _place_groups(groups: list[dict], h: int, w: int) -> list[tuple[int, int]] | None:
    order = sorted(range(len(groups)), key=lambda i: groups[i]["envelope"][0] * groups[i]["envelope"][1], reverse=True)
    placements: list[tuple[int, int] | None] = [None] * len(groups)
    used: list[tuple[int, int, int, int]] = []

    for idx in order:
        gh, gw = groups[idx]["envelope"]
        placed = False
        for _ in range(250):
            r = random.randint(0, h - gh)
            c = random.randint(0, w - gw)
            rect = (r, c, gh, gw)
            if all(_rectangles_do_not_touch(rect, other) for other in used):
                placements[idx] = (r, c)
                used.append(rect)
                placed = True
                break
        if not placed:
            return None

    return [(int(r), int(c)) for r, c in placements]  # type: ignore[misc]


def _render_input(groups: list[dict], placements: list[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for group, (base_r, base_c) in zip(groups, placements):
        for frame in group["frames"]:
            r0, c0, r1, c1 = frame["bbox"]
            _draw_frame(
                grid,
                (base_r + r0, base_c + c0, base_r + r1, base_c + c1),
                int(frame["color"]),
            )

        sr0, sc0, sr1, sc1 = group["seed_bbox"]
        patch = group["seed_patch"]
        target = grid[base_r + sr0 : base_r + sr1 + 1, base_c + sc0 : base_c + sc1 + 1]
        mask = patch != BG
        assert not bool(np.any(mask & (target != BG)))
        target[mask] = patch[mask]
    return grid


def _expected_output(inp: np.ndarray, groups: list[dict], placements: list[tuple[int, int]]) -> np.ndarray:
    out = inp.copy()
    for group, (base_r, base_c) in zip(groups, placements):
        sr0, sc0, sr1, sc1 = group["seed_bbox"]
        patch = group["seed_patch"]
        mask = patch != BG
        target = out[base_r + sr0 : base_r + sr1 + 1, base_c + sc0 : base_c + sc1 + 1]
        target[mask] = BG

        rot = np.rot90(patch, -int(group["depth"]))
        rr0, rc0, _rr1, rc1 = group["root_bbox"]
        dst_r0 = base_r + rr0 - rot.shape[0]
        dst_c0 = base_c + rc1 + 1
        dst = out[dst_r0 : dst_r0 + rot.shape[0], dst_c0 : dst_c0 + rot.shape[1]]
        rot_mask = rot != BG
        assert not bool(np.any(rot_mask & (dst != BG)))
        dst[rot_mask] = rot[rot_mask]
    return out


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    count = random.randint(1, 3)
    groups = [_sample_group() for _ in range(count)]

    max_h = max(group["envelope"][0] for group in groups)
    max_w = max(group["envelope"][1] for group in groups)
    total_h = sum(group["envelope"][0] for group in groups)
    total_w = sum(group["envelope"][1] for group in groups)

    min_h = max(max_h, min(30, (total_h + 1) // 2))
    min_w = max(max_w, min(30, (total_w + 1) // 2))
    if min_h > 30 or min_w > 30:
        return None

    for _ in range(80):
        h = random.randint(min_h, 30)
        w = random.randint(min_w, 30)
        placements = _place_groups(groups, h, w)
        if placements is None:
            continue
        inp = _render_input(groups, placements, (h, w))
        expected = _expected_output(inp, groups, placements)
        return inp, expected
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        case = _generate_case()
        if case is None:
            continue
        inp, expected = case
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid input")
