import itertools
import random
from collections import deque

import numpy as np


BG = 0
FENCE = 5
SIDES = ("top", "bottom", "left", "right")
PALETTE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400
MAX_GRID_DIM = 40

_RNG = random.Random()


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = np.asarray(grid == color, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and mask[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _analyze_rect_frame(
    grid: np.ndarray, color: int, coords: np.ndarray
) -> dict[str, object] | None:
    if coords.size == 0:
        return None
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return None

    pts = {(int(r), int(c)) for r, c in coords}
    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    if not corners.issubset(pts):
        return None
    for r, c in pts:
        if r not in (r0, r1) and c not in (c0, c1):
            return None

    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    if inner.size == 0 or np.any(inner != BG):
        return None

    windows: dict[str, list[int]] = {side: [] for side in SIDES}
    expected_pts: set[tuple[int, int]] = set()

    for rr, cc in corners:
        if int(grid[rr, cc]) != color:
            return None
        expected_pts.add((rr, cc))

    for t, c in enumerate(range(c0 + 1, c1)):
        val = int(grid[r0, c])
        if val == color:
            expected_pts.add((r0, c))
        elif val == BG:
            windows["top"].append(t)
        else:
            return None

    for t, c in enumerate(range(c0 + 1, c1)):
        val = int(grid[r1, c])
        if val == color:
            expected_pts.add((r1, c))
        elif val == BG:
            windows["bottom"].append(t)
        else:
            return None

    for t, r in enumerate(range(r0 + 1, r1)):
        val = int(grid[r, c0])
        if val == color:
            expected_pts.add((r, c0))
        elif val == BG:
            windows["left"].append(t)
        else:
            return None

    for t, r in enumerate(range(r0 + 1, r1)):
        val = int(grid[r, c1])
        if val == color:
            expected_pts.add((r, c1))
        elif val == BG:
            windows["right"].append(t)
        else:
            return None

    if pts != expected_pts:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "height": h,
        "width": w,
        "windows": {side: tuple(windows[side]) for side in SIDES},
    }


def _signature_from_frame(frame: dict[str, object]) -> tuple[tuple[int, ...], ...]:
    h = int(frame["height"])
    w = int(frame["width"])
    windows = frame["windows"]
    lengths = {"top": w - 2, "bottom": w - 2, "left": h - 2, "right": h - 2}
    signature = []
    for side in SIDES:
        length = lengths[side]
        bins = tuple(sorted(int(3 * t // length) for t in windows[side]))
        signature.append(bins)
    return tuple(signature)


def _detect_frames(grid: np.ndarray, color: int) -> list[dict[str, object]]:
    frames = []
    for coords in _components_of_color(grid, color):
        frame = _analyze_rect_frame(grid, color, coords)
        if frame is not None:
            frames.append(frame)
    return frames


def _fill_fence(
    out: np.ndarray, frame: dict[str, object], fill_color: int
) -> None:
    r0, c0, r1, c1 = frame["bbox"]
    windows = frame["windows"]

    vertical_lines = sorted(
        {c0 + 1 + t for side in ("top", "bottom") for t in windows[side]}
    )
    horizontal_lines = sorted(
        {r0 + 1 + t for side in ("left", "right") for t in windows[side]}
    )

    out[r0 + 1 : r1, c0 + 1 : c1] = BG

    row_cuts = [r0] + horizontal_lines + [r1]
    col_cuts = [c0] + vertical_lines + [c1]
    for band_r in range(len(row_cuts) - 1):
        rs = row_cuts[band_r] + 1
        re = row_cuts[band_r + 1]
        if rs >= re:
            continue
        for band_c in range(len(col_cuts) - 1):
            cs = col_cuts[band_c] + 1
            ce = col_cuts[band_c + 1]
            if cs >= ce:
                continue
            color = fill_color if (band_r + band_c) % 2 == 0 else BG
            out[rs:re, cs:ce] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each grey fenced rectangle to a palette entry by its window signature,
    then partition the fence interior with inward rays from the windows and
    checkerboard-fill the resulting regions starting from the upper-left region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    fences = _detect_frames(g, FENCE)
    assert fences, "expected at least one grey fence"

    palette_frames: list[dict[str, object]] = []
    for color in sorted(set(map(int, np.unique(g))) - {BG, FENCE}):
        palette_frames.extend(_detect_frames(g, color))
    assert palette_frames, "expected at least one palette frame"

    sig_to_color: dict[tuple[tuple[int, ...], ...], int] = {}
    for frame in palette_frames:
        sig = _signature_from_frame(frame)
        color = int(g[frame["bbox"][0], frame["bbox"][1]])
        assert sig not in sig_to_color, "palette signatures must be unique"
        sig_to_color[sig] = color

    out = g.copy()
    for fence in fences:
        sig = _signature_from_frame(fence)
        assert sig in sig_to_color, "every fence must match exactly one palette entry"
        _fill_fence(out, fence, sig_to_color[sig])
    return out


def _make_signature(
    top: tuple[int, ...] = (),
    bottom: tuple[int, ...] = (),
    left: tuple[int, ...] = (),
    right: tuple[int, ...] = (),
) -> tuple[tuple[int, ...], ...]:
    return (tuple(top), tuple(bottom), tuple(left), tuple(right))


def _sample_signature(
    rng: random.Random, used: set[tuple[tuple[int, ...], ...]]
) -> tuple[tuple[int, ...], ...]:
    for _ in range(200):
        mode = rng.choice(("vertical", "horizontal", "mixed"))
        sides: list[str] = []
        if mode == "vertical":
            count = rng.randint(1, 2)
            sides = rng.sample(["top", "bottom"], count)
        elif mode == "horizontal":
            count = rng.randint(1, 2)
            sides = rng.sample(["left", "right"], count)
        else:
            sides.extend(rng.sample(["top", "bottom"], rng.randint(1, 2)))
            sides.extend(rng.sample(["left", "right"], rng.randint(1, 2)))
            if len(sides) < 4 and rng.random() < 0.35:
                remaining = [side for side in SIDES if side not in sides]
                sides.append(rng.choice(remaining))

        side_bins = {side: () for side in SIDES}
        for side in sides:
            side_bins[side] = (rng.randrange(3),)
        sig = _make_signature(
            top=side_bins["top"],
            bottom=side_bins["bottom"],
            left=side_bins["left"],
            right=side_bins["right"],
        )
        if sig not in used:
            return sig
    raise RuntimeError("failed to sample a unique signature")


def _candidate_offsets(length: int, bin_idx: int, inner_only: bool) -> list[int]:
    values = [t for t in range(length) if 3 * t // length == bin_idx]
    if inner_only:
        inner = [t for t in values if 1 <= t <= length - 2]
        if inner:
            values = inner
    return values


def _choose_axis_offsets(
    length: int, side_bins: list[tuple[str, int]], rng: random.Random
) -> dict[str, int] | None:
    if not side_bins:
        return {}

    candidate_lists: list[list[int]] = []
    for _side, bin_idx in side_bins:
        candidates = _candidate_offsets(length, bin_idx, inner_only=True)
        if not candidates:
            return None
        candidate_lists.append(candidates)

    valid: list[tuple[int, ...]] = []
    for combo in itertools.product(*candidate_lists):
        unique = sorted(set(combo))
        if unique[0] < 1 or unique[-1] > length - 2:
            continue
        if any(b - a < 2 for a, b in zip(unique, unique[1:])):
            continue
        valid.append(tuple(int(x) for x in combo))

    if not valid:
        return None
    choice = rng.choice(valid)
    return {side: int(offset) for (side, _bin_idx), offset in zip(side_bins, choice)}


def _realize_fence_windows(
    sig: tuple[tuple[int, ...], ...], h: int, w: int, rng: random.Random
) -> dict[str, tuple[int, ...]] | None:
    vertical_sides = []
    horizontal_sides = []
    if sig[0]:
        vertical_sides.append(("top", int(sig[0][0])))
    if sig[1]:
        vertical_sides.append(("bottom", int(sig[1][0])))
    if sig[2]:
        horizontal_sides.append(("left", int(sig[2][0])))
    if sig[3]:
        horizontal_sides.append(("right", int(sig[3][0])))

    vertical = _choose_axis_offsets(w - 2, vertical_sides, rng)
    horizontal = _choose_axis_offsets(h - 2, horizontal_sides, rng)
    if vertical is None or horizontal is None:
        return None

    windows = {side: [] for side in SIDES}
    for side, offset in vertical.items():
        windows[side] = [offset]
    for side, offset in horizontal.items():
        windows[side] = [offset]
    return {side: tuple(windows[side]) for side in SIDES}


def _center_choice(values: list[int]) -> int:
    return int(values[len(values) // 2])


def _realize_palette_windows(
    sig: tuple[tuple[int, ...], ...], h: int, w: int
) -> dict[str, tuple[int, ...]]:
    windows = {side: [] for side in SIDES}
    lengths = {"top": w - 2, "bottom": w - 2, "left": h - 2, "right": h - 2}
    for side, bins in zip(SIDES, sig):
        length = lengths[side]
        for bin_idx in bins:
            candidates = _candidate_offsets(length, int(bin_idx), inner_only=False)
            windows[side].append(_center_choice(candidates))
    return {side: tuple(windows[side]) for side in SIDES}


def _draw_frame(
    grid: np.ndarray,
    r0: int,
    c0: int,
    h: int,
    w: int,
    color: int,
    windows: dict[str, tuple[int, ...]],
) -> None:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0 + 1 : r1, c0 + 1 : c1] = BG

    for t in windows["top"]:
        grid[r0, c0 + 1 + int(t)] = BG
    for t in windows["bottom"]:
        grid[r1, c0 + 1 + int(t)] = BG
    for t in windows["left"]:
        grid[r0 + 1 + int(t), c0] = BG
    for t in windows["right"]:
        grid[r0 + 1 + int(t), c1] = BG


def _pack_palette_top(
    sizes: list[tuple[int, int]], rng: random.Random
) -> tuple[dict[int, tuple[int, int]], int, int]:
    order = list(range(len(sizes)))
    rng.shuffle(order)
    left_margin = rng.randint(0, 2)
    right_margin = rng.randint(0, 2)
    cur_c = left_margin
    positions: dict[int, tuple[int, int]] = {}
    for i, idx in enumerate(order):
        h, w = sizes[idx]
        positions[idx] = (0, cur_c)
        cur_c += w
        if i + 1 < len(order):
            cur_c += rng.randint(1, 2)
    total_w = cur_c + right_margin
    total_h = max(h for h, _w in sizes)
    return positions, total_h, total_w


def _pack_palette_left(
    sizes: list[tuple[int, int]], rng: random.Random
) -> tuple[dict[int, tuple[int, int]], int, int]:
    order = list(range(len(sizes)))
    rng.shuffle(order)
    top_margin = rng.randint(0, 2)
    bottom_margin = rng.randint(0, 2)
    cur_r = top_margin
    positions: dict[int, tuple[int, int]] = {}
    for i, idx in enumerate(order):
        h, w = sizes[idx]
        positions[idx] = (cur_r, 0)
        cur_r += h
        if i + 1 < len(order):
            cur_r += rng.randint(1, 2)
    total_h = cur_r + bottom_margin
    total_w = max(w for _h, w in sizes)
    return positions, total_h, total_w


def _pack_fences(
    sizes: list[tuple[int, int]], rng: random.Random
) -> tuple[dict[int, tuple[int, int]], int, int]:
    n = len(sizes)
    cols = 1 if n == 1 else rng.choice((1, 2))
    order = list(range(n))
    rng.shuffle(order)
    rows = [order[i : i + cols] for i in range(0, n, cols)]

    positions: dict[int, tuple[int, int]] = {}
    cur_r = 0
    body_w = 0
    for row_idx, row in enumerate(rows):
        row_h = max(sizes[idx][0] for idx in row)
        cur_c = 0
        for i, idx in enumerate(row):
            h, w = sizes[idx]
            v_off = rng.randint(0, row_h - h)
            positions[idx] = (cur_r + v_off, cur_c)
            cur_c += w
            if i + 1 < len(row):
                cur_c += rng.randint(2, 4)
        body_w = max(body_w, cur_c)
        cur_r += row_h
        if row_idx + 1 < len(rows):
            cur_r += rng.randint(2, 4)
    return positions, cur_r, body_w


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_fences = rng.randint(1, 3)
    n_palette = n_fences + 1
    palette_colors = rng.sample(PALETTE_COLORS, n_palette)

    used_signatures: set[tuple[tuple[int, ...], ...]] = set()
    palette_signatures = []
    for _ in range(n_palette):
        sig = _sample_signature(rng, used_signatures)
        used_signatures.add(sig)
        palette_signatures.append(sig)

    distractor_idx = rng.randrange(n_palette)
    fence_signatures = [
        sig for i, sig in enumerate(palette_signatures) if i != distractor_idx
    ]
    sig_to_color = {
        sig: color for sig, color in zip(palette_signatures, palette_colors)
    }

    palette_specs = []
    for sig, color in zip(palette_signatures, palette_colors):
        h = rng.randint(5, 7)
        w = rng.randint(5, 7)
        palette_specs.append(
            {
                "signature": sig,
                "color": color,
                "size": (h, w),
                "windows": _realize_palette_windows(sig, h, w),
            }
        )

    fence_specs = []
    for sig in fence_signatures:
        h = rng.randint(8, 10)
        w = rng.randint(8, 10)
        windows = _realize_fence_windows(sig, h, w, rng)
        if windows is None:
            return None
        fence_specs.append(
            {
                "signature": sig,
                "color": sig_to_color[sig],
                "size": (h, w),
                "windows": windows,
            }
        )

    orientation = rng.choice(("top", "left"))
    palette_sizes = [spec["size"] for spec in palette_specs]
    fence_sizes = [spec["size"] for spec in fence_specs]

    palette_positions: dict[int, tuple[int, int]]
    fence_positions_rel: dict[int, tuple[int, int]]
    if orientation == "top":
        palette_positions, strip_h, strip_w = _pack_palette_top(palette_sizes, rng)
        fence_positions_rel, body_h, body_w = _pack_fences(fence_sizes, rng)
        strip_gap = rng.randint(2, 4)
        grid_w = max(strip_w, body_w) + rng.randint(0, 3)
        grid_h = strip_h + strip_gap + body_h + rng.randint(0, 3)
        if grid_h > MAX_GRID_DIM or grid_w > MAX_GRID_DIM:
            return None
        strip_dx = rng.randint(0, grid_w - strip_w)
        body_dx = rng.randint(0, grid_w - body_w)
        body_top = strip_h + strip_gap
        palette_origins = {
            idx: (r, c + strip_dx) for idx, (r, c) in palette_positions.items()
        }
        fence_origins = {
            idx: (body_top + r, body_dx + c)
            for idx, (r, c) in fence_positions_rel.items()
        }
    else:
        palette_positions, strip_h, strip_w = _pack_palette_left(palette_sizes, rng)
        fence_positions_rel, body_h, body_w = _pack_fences(fence_sizes, rng)
        strip_gap = rng.randint(2, 4)
        grid_h = max(strip_h, body_h) + rng.randint(0, 3)
        grid_w = strip_w + strip_gap + body_w + rng.randint(0, 3)
        if grid_h > MAX_GRID_DIM or grid_w > MAX_GRID_DIM:
            return None
        strip_dy = rng.randint(0, grid_h - strip_h)
        body_dy = rng.randint(0, grid_h - body_h)
        body_left = strip_w + strip_gap
        palette_origins = {
            idx: (r + strip_dy, c) for idx, (r, c) in palette_positions.items()
        }
        fence_origins = {
            idx: (body_dy + r, body_left + c)
            for idx, (r, c) in fence_positions_rel.items()
        }

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for idx, spec in enumerate(palette_specs):
        r0, c0 = palette_origins[idx]
        h, w = spec["size"]
        _draw_frame(grid, r0, c0, h, w, spec["color"], spec["windows"])
    for idx, spec in enumerate(fence_specs):
        r0, c0 = fence_origins[idx]
        h, w = spec["size"]
        _draw_frame(grid, r0, c0, h, w, FENCE, spec["windows"])

    try:
        out = main(grid)
    except AssertionError:
        return None

    if not np.any(out != grid):
        return None
    for idx, spec in enumerate(fence_specs):
        r0, c0 = fence_origins[idx]
        h, w = spec["size"]
        interior = out[r0 + 1 : r0 + h - 1, c0 + 1 : c0 + w - 1]
        if spec["color"] not in set(map(int, np.unique(interior))):
            return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
