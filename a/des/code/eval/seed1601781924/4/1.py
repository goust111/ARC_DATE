from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}


def _find_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(int(r), int(c))])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
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
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _bbox_size(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _is_rect_ring(comp: dict) -> bool:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h, w = _bbox_size(comp["bbox"])
    if h < 3 or w < 3:
        return False
    if int(comp["area"]) != 2 * (h + w) - 4:
        return False

    mask = np.zeros((h, w), dtype=bool)
    coords = comp["coords"]
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return bool(np.array_equal(mask, expected))


def _is_solid_rect(comp: dict) -> bool:
    h, w = _bbox_size(comp["bbox"])
    if h < 1 or w < 1:
        return False
    return int(comp["area"]) == h * w


def _strictly_inside_frame(seed_bbox: tuple[int, int, int, int], frame_bbox: tuple[int, int, int, int]) -> bool:
    sr0, sc0, sr1, sc1 = map(int, seed_bbox)
    fr0, fc0, fr1, fc1 = map(int, frame_bbox)
    return (
        fr0 + 1 < sr0
        and sr1 < fr1 - 1
        and fc0 + 1 < sc0
        and sc1 < fc1 - 1
    )


def _bias_direction(frame_bbox: tuple[int, int, int, int], seed_bbox: tuple[int, int, int, int]) -> str:
    fr0, fc0, fr1, fc1 = map(int, frame_bbox)
    sr0, sc0, sr1, sc1 = map(int, seed_bbox)
    frame_cy = (fr0 + fr1) / 2.0
    frame_cx = (fc0 + fc1) / 2.0
    seed_cy = (sr0 + sr1) / 2.0
    seed_cx = (sc0 + sc1) / 2.0
    dy = seed_cy - frame_cy
    dx = seed_cx - frame_cx
    if abs(dy) > abs(dx):
        return "down" if dy > 0 else "up"
    return "right" if dx > 0 else "left"


def _external_ring_bbox(
    frame_bbox: tuple[int, int, int, int],
    ring_h: int,
    ring_w: int,
    side: str,
) -> tuple[int, int, int, int]:
    fr0, fc0, fr1, fc1 = map(int, frame_bbox)
    fh, fw = _bbox_size(frame_bbox)

    if side == "up":
        r0 = fr0 - ring_h - 1
        c0 = fc0 + (fw - ring_w) // 2
    elif side == "down":
        r0 = fr1 + 2
        c0 = fc0 + (fw - ring_w) // 2
    elif side == "left":
        r0 = fr0 + (fh - ring_h) // 2
        c0 = fc0 - ring_w - 1
    elif side == "right":
        r0 = fr0 + (fh - ring_h) // 2
        c0 = fc1 + 2
    else:
        raise ValueError(f"unknown side: {side}")

    return int(r0), int(c0), int(r0 + ring_h - 1), int(c0 + ring_w - 1)


def _draw_ring(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy a scaled fence-shaped ring outside each frame, on the side opposite the seed bias.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = map(int, g.shape)

    comps = _find_components(g, bg=BG)
    frames = [comp for comp in comps if _is_rect_ring(comp)]
    solids = [comp for comp in comps if _is_solid_rect(comp)]

    for frame in frames:
        frame_bbox = frame["bbox"]
        seed_candidates = [
            comp
            for comp in solids
            if comp is not frame and _strictly_inside_frame(comp["bbox"], frame_bbox)
        ]
        if len(seed_candidates) != 1:
            continue
        seed = seed_candidates[0]

        fh, fw = _bbox_size(frame_bbox)
        area = int(seed["area"])
        ring_h = fh // area
        ring_w = fw // area
        if ring_h < 3 or ring_w < 3:
            continue

        bias = _bias_direction(frame_bbox, seed["bbox"])
        side = OPPOSITE[bias]
        ring_bbox = _external_ring_bbox(frame_bbox, ring_h, ring_w, side)
        rr0, cc0, rr1, cc1 = ring_bbox
        if rr0 < 0 or cc0 < 0 or rr1 >= h or cc1 >= w:
            continue

        _draw_ring(out, ring_bbox, int(seed["color"]))

    return out


# --------------- generator ---------------

GEN_ATTEMPTS = 400
OBJECT_COUNT_RANGE = (1, 4)  # upper bound exclusive
FENCE_H_RANGE = (6, 14)      # upper bound exclusive
FENCE_W_RANGE = (6, 14)      # upper bound exclusive
OUTER_MARGIN_RANGE = (0, 4)  # upper bound exclusive
ROW_GAP_RANGE = (1, 5)       # upper bound exclusive
COL_GAP_RANGE = (1, 5)       # upper bound exclusive
ROW_WIDTH_MIN = 18
ROW_WIDTH_MAX = 44
SEED_BIAS_DIRS = ("up", "down", "left", "right")


def _seed_position_candidates(
    fence_h: int,
    fence_w: int,
    seed_h: int,
    seed_w: int,
    bias: str,
) -> list[tuple[int, int]]:
    frame_cy = (fence_h - 1) / 2.0
    frame_cx = (fence_w - 1) / 2.0
    candidates: list[tuple[int, int]] = []

    for sr0 in range(2, fence_h - seed_h - 1):
        for sc0 in range(2, fence_w - seed_w - 1):
            seed_cy = sr0 + (seed_h - 1) / 2.0
            seed_cx = sc0 + (seed_w - 1) / 2.0
            dy = seed_cy - frame_cy
            dx = seed_cx - frame_cx
            if bias == "up" and dy < 0 and abs(dy) > abs(dx):
                candidates.append((int(sr0), int(sc0)))
            elif bias == "down" and dy > 0 and abs(dy) > abs(dx):
                candidates.append((int(sr0), int(sc0)))
            elif bias == "left" and dx < 0 and abs(dx) > abs(dy):
                candidates.append((int(sr0), int(sc0)))
            elif bias == "right" and dx > 0 and abs(dx) > abs(dy):
                candidates.append((int(sr0), int(sc0)))
    return candidates


def _sample_object_spec(rng: np.random.Generator, bias: str, should_draw: bool) -> dict | None:
    for _ in range(300):
        fence_h = int(rng.integers(*FENCE_H_RANGE))
        fence_w = int(rng.integers(*FENCE_W_RANGE))
        safe_h = fence_h - 4
        safe_w = fence_w - 4
        if safe_h < 1 or safe_w < 1:
            continue

        max_draw_area = min(fence_h // 3, fence_w // 3)
        draw_options: list[tuple[int, int, list[tuple[int, int]]]] = []
        skip_options: list[tuple[int, int, list[tuple[int, int]]]] = []

        for seed_h in range(1, safe_h + 1):
            for seed_w in range(1, safe_w + 1):
                positions = _seed_position_candidates(fence_h, fence_w, seed_h, seed_w, bias)
                if not positions:
                    continue
                area = seed_h * seed_w
                option = (int(seed_h), int(seed_w), positions)
                if area <= max_draw_area:
                    draw_options.append(option)
                else:
                    skip_options.append(option)

        choices = draw_options if should_draw else skip_options
        if not choices:
            continue

        seed_h, seed_w, positions = choices[int(rng.integers(0, len(choices)))]
        seed_r0, seed_c0 = positions[int(rng.integers(0, len(positions)))]
        area = int(seed_h * seed_w)
        return {
            "bias": bias,
            "fence_h": int(fence_h),
            "fence_w": int(fence_w),
            "seed_h": int(seed_h),
            "seed_w": int(seed_w),
            "seed_r0": int(seed_r0),
            "seed_c0": int(seed_c0),
            "area": area,
            "ring_h": int(fence_h // area),
            "ring_w": int(fence_w // area),
            "draws": bool(should_draw),
        }
    return None


def _build_object_patch(spec: dict, fence_color: int, seed_color: int) -> tuple[np.ndarray, np.ndarray]:
    fence_h = int(spec["fence_h"])
    fence_w = int(spec["fence_w"])
    seed_h = int(spec["seed_h"])
    seed_w = int(spec["seed_w"])
    seed_r0 = int(spec["seed_r0"])
    seed_c0 = int(spec["seed_c0"])
    ring_h = int(spec["ring_h"])
    ring_w = int(spec["ring_w"])
    draws = bool(spec["draws"])
    ring_side = OPPOSITE[str(spec["bias"])]

    if draws:
        if ring_side == "up":
            patch_h, patch_w = ring_h + 1 + fence_h, fence_w
            frame_r0, frame_c0 = ring_h + 1, 0
            ring_r0, ring_c0 = 0, (fence_w - ring_w) // 2
        elif ring_side == "down":
            patch_h, patch_w = fence_h + 1 + ring_h, fence_w
            frame_r0, frame_c0 = 0, 0
            ring_r0, ring_c0 = fence_h + 1, (fence_w - ring_w) // 2
        elif ring_side == "left":
            patch_h, patch_w = fence_h, ring_w + 1 + fence_w
            frame_r0, frame_c0 = 0, ring_w + 1
            ring_r0, ring_c0 = (fence_h - ring_h) // 2, 0
        elif ring_side == "right":
            patch_h, patch_w = fence_h, fence_w + 1 + ring_w
            frame_r0, frame_c0 = 0, 0
            ring_r0, ring_c0 = (fence_h - ring_h) // 2, fence_w + 1
        else:
            raise ValueError(f"unknown ring side: {ring_side}")
    else:
        patch_h, patch_w = fence_h, fence_w
        frame_r0, frame_c0 = 0, 0
        ring_r0 = ring_c0 = 0

    inp = np.zeros((patch_h, patch_w), dtype=int)
    out = np.zeros((patch_h, patch_w), dtype=int)

    frame_bbox = (frame_r0, frame_c0, frame_r0 + fence_h - 1, frame_c0 + fence_w - 1)
    _draw_ring(inp, frame_bbox, int(fence_color))
    _draw_ring(out, frame_bbox, int(fence_color))

    inp[
        frame_r0 + seed_r0 : frame_r0 + seed_r0 + seed_h,
        frame_c0 + seed_c0 : frame_c0 + seed_c0 + seed_w,
    ] = int(seed_color)
    out[
        frame_r0 + seed_r0 : frame_r0 + seed_r0 + seed_h,
        frame_c0 + seed_c0 : frame_c0 + seed_c0 + seed_w,
    ] = int(seed_color)

    if draws:
        ring_bbox = (ring_r0, ring_c0, ring_r0 + ring_h - 1, ring_c0 + ring_w - 1)
        _draw_ring(out, ring_bbox, int(seed_color))

    return inp, out


def _pack_patches(
    rng: np.random.Generator,
    patches: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    count = len(patches)
    order = list(rng.permutation(count))
    widths = [int(patches[i][0].shape[1]) for i in range(count)]
    max_patch_w = max(widths)
    total_patch_w = sum(widths)
    row_limit_lo = max(ROW_WIDTH_MIN, max_patch_w)
    row_limit_hi = max(row_limit_lo, min(ROW_WIDTH_MAX, total_patch_w + 4 * (count - 1) + 6))
    row_limit = int(rng.integers(row_limit_lo, row_limit_hi + 1))

    top_margin = int(rng.integers(*OUTER_MARGIN_RANGE))
    left_margin = int(rng.integers(*OUTER_MARGIN_RANGE))
    right_margin = int(rng.integers(*OUTER_MARGIN_RANGE))
    bottom_margin = int(rng.integers(*OUTER_MARGIN_RANGE))

    positions: list[tuple[int, int] | None] = [None] * count
    cur_r = top_margin
    cur_c = left_margin
    row_h = 0
    used_h = 0
    used_w = 0

    for idx in order:
        patch_h, patch_w = map(int, patches[idx][0].shape)
        if cur_c > left_margin and cur_c + patch_w > row_limit:
            cur_r += row_h + int(rng.integers(*ROW_GAP_RANGE))
            cur_c = left_margin
            row_h = 0
        positions[idx] = (int(cur_r), int(cur_c))
        cur_c += patch_w + int(rng.integers(*COL_GAP_RANGE))
        row_h = max(row_h, patch_h)
        used_h = max(used_h, cur_r + patch_h)
        used_w = max(used_w, cur_c)

    final_h = used_h + bottom_margin
    final_w = max(left_margin + max_patch_w, used_w + right_margin)
    inp = np.zeros((final_h, final_w), dtype=int)
    out = np.zeros((final_h, final_w), dtype=int)

    for idx, pos in enumerate(positions):
        assert pos is not None
        r0, c0 = pos
        patch_in, patch_out = patches[idx]
        ph, pw = map(int, patch_in.shape)
        inp[r0 : r0 + ph, c0 : c0 + pw] = patch_in
        out[r0 : r0 + ph, c0 : c0 + pw] = patch_out

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n_objects = int(rng.integers(*OBJECT_COUNT_RANGE))
    directions = [SEED_BIAS_DIRS[int(i)] for i in rng.integers(0, len(SEED_BIAS_DIRS), size=n_objects)]

    if n_objects == 1:
        draw_flags = [bool(rng.integers(0, 2))]
    else:
        draw_flags = [True, False]
        for _ in range(n_objects - 2):
            draw_flags.append(bool(rng.integers(0, 2)))
        draw_flags = [draw_flags[int(i)] for i in rng.permutation(len(draw_flags))]

    colors = rng.choice(np.arange(1, 10), size=2 * n_objects, replace=False)
    fence_colors = [int(v) for v in colors[:n_objects]]
    seed_colors = [int(v) for v in colors[n_objects:]]

    patches: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_objects):
        spec = _sample_object_spec(rng, directions[i], draw_flags[i])
        if spec is None:
            return None
        patch = _build_object_patch(spec, fence_colors[i], seed_colors[i])
        patches.append(patch)

    inp, expected = _pack_patches(rng, patches)
    actual = main(inp)
    if np.array_equal(actual, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
