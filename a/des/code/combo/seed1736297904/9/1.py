from __future__ import annotations

from collections import deque
import random

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
BG = 0
NON_BG_COLORS = tuple(range(1, 10))

GEN_RETRIES = 400
GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
LEGEND_COUNT_RANGE = (2, 4)
TARGET_COUNT_RANGE = (1, 3)
TARGET_OUTER_H_RANGE = (5, 10)
TARGET_OUTER_W_RANGE = (5, 10)
LEGEND_GAP = 1
PLACEMENT_RETRIES = 120


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _monochrome_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_rectangular_frames(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for comp in _monochrome_components(grid, color):
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            if height < 3 or width < 3:
                continue
            if len(comp) != 2 * (height + width) - 4:
                continue
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            mask = patch == color
            expected = np.zeros((height, width), dtype=bool)
            expected[0, :] = True
            expected[-1, :] = True
            expected[:, 0] = True
            expected[:, -1] = True
            if not np.array_equal(mask, expected):
                continue
            frames.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                }
            )
    return frames


def _touches_edge(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    h, w = shape
    return r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1


def _tile_pattern(pattern: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    ph, pw = pattern.shape
    oh, ow = out_shape
    rr = np.arange(oh)[:, None] % ph
    cc = np.arange(ow)[None, :] % pw
    return pattern[rr, cc]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each framed region by using its seed color as a key into the edge legend.

    The legend is made of small edge-touching square frames. Their border color is
    the lookup key and the inner 2x2 or 3x3 patch is the pattern to tile. Larger
    non-edge rectangular frames contain exactly one seed pixel on background; that
    seed color selects the legend pattern.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    frames = _find_rectangular_frames(g, bg)

    legends: dict[int, np.ndarray] = {}
    targets: list[tuple[tuple[int, int, int, int], int]] = []

    for frame in frames:
        color = int(frame["color"])
        r0, c0, r1, c1 = frame["bbox"]
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        ih, iw = interior.shape
        edge_frame = _touches_edge((r0, c0, r1, c1), g.shape)

        if edge_frame and ih == iw and ih in (2, 3):
            inside_colors = {int(v) for v in np.unique(interior)}
            if len(inside_colors) == 2 and bg not in inside_colors and color not in inside_colors:
                assert color not in legends, "legend keys must be unique"
                legends[color] = interior.copy()
                continue

        if edge_frame:
            continue

        seed_coords = np.argwhere(interior != bg)
        if seed_coords.shape[0] != 1:
            continue
        sr, sc = map(int, seed_coords[0])
        seed_color = int(interior[sr, sc])
        if seed_color == color:
            continue
        targets.append(((r0, c0, r1, c1), seed_color))

    assert legends, "expected at least one legend frame"
    assert targets, "expected at least one target frame"

    out = g.copy()
    for bbox, seed_color in targets:
        assert seed_color in legends, "every seed color must map to a legend"
        r0, c0, r1, c1 = bbox
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[:, :] = _tile_pattern(legends[seed_color], interior.shape)
    return out


def _make_pattern(size: int, color_a: int, color_b: int, mode: str, phase: int) -> np.ndarray:
    rr = np.arange(size)[:, None]
    cc = np.arange(size)[None, :]
    if mode == "checker":
        use_a = (rr + cc + phase) % 2 == 0
    elif mode == "hstripe":
        use_a = np.broadcast_to((rr + phase) % 2 == 0, (size, size))
    elif mode == "vstripe":
        use_a = np.broadcast_to((cc + phase) % 2 == 0, (size, size))
    else:
        raise ValueError(mode)
    pattern = np.full((size, size), color_b, dtype=int)
    pattern[use_a] = color_a
    return pattern


def _legend_patch(border_color: int, pattern: np.ndarray) -> np.ndarray:
    size = int(pattern.shape[0])
    patch = np.full((size + 2, size + 2), border_color, dtype=int)
    patch[1:-1, 1:-1] = pattern
    return patch


def _place_legends(
    grid: np.ndarray,
    rng: random.Random,
    legend_specs: list[dict[str, object]],
    side: str,
) -> int | None:
    h, w = grid.shape
    outer_sizes = [int(spec["patch"].shape[0]) for spec in legend_specs]
    max_outer = max(outer_sizes)

    if side in ("top", "bottom"):
        total = sum(int(spec["patch"].shape[1]) for spec in legend_specs) + LEGEND_GAP * (len(legend_specs) - 1)
        if total > w - 2:
            return None
        start_c = rng.randint(1, w - 1 - total)
        c = start_c
        for spec in legend_specs:
            patch = spec["patch"]
            ph, pw = patch.shape
            r0 = 0 if side == "top" else h - ph
            grid[r0 : r0 + ph, c : c + pw] = patch
            c += pw + LEGEND_GAP
    else:
        total = sum(int(spec["patch"].shape[0]) for spec in legend_specs) + LEGEND_GAP * (len(legend_specs) - 1)
        if total > h - 2:
            return None
        start_r = rng.randint(1, h - 1 - total)
        r = start_r
        for spec in legend_specs:
            patch = spec["patch"]
            ph, pw = patch.shape
            c0 = 0 if side == "left" else w - pw
            grid[r : r + ph, c0 : c0 + pw] = patch
            r += ph + LEGEND_GAP

    return max_outer + 1


def _allowed_target_area(
    grid_shape: tuple[int, int],
    side: str,
    reserved_band: int,
) -> tuple[int, int, int, int]:
    h, w = grid_shape
    r_min, c_min = 1, 1
    r_max, c_max = h - 2, w - 2
    if side == "top":
        r_min = max(r_min, reserved_band)
    elif side == "bottom":
        r_max = min(r_max, h - reserved_band - 1)
    elif side == "left":
        c_min = max(c_min, reserved_band)
    elif side == "right":
        c_max = min(c_max, w - reserved_band - 1)
    return r_min, c_min, r_max, c_max


def _can_place(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int, margin: int = 1) -> bool:
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(occupied.shape[0] - 1, r1 + margin)
    cc1 = min(occupied.shape[1] - 1, c1 + margin)
    return not bool(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _mark_occupied(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int, margin: int = 1) -> None:
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(occupied.shape[0] - 1, r1 + margin)
    cc1 = min(occupied.shape[1] - 1, c1 + margin)
    occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _draw_target(
    grid: np.ndarray,
    r0: int,
    c0: int,
    height: int,
    width: int,
    border_color: int,
    seed_color: int,
    rng: random.Random,
) -> None:
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    grid[r0 : r1 + 1, c0] = border_color
    grid[r0 : r1 + 1, c1] = border_color
    grid[r0, c0 : c1 + 1] = border_color
    grid[r1, c0 : c1 + 1] = border_color

    sr = rng.randint(r0 + 1, r1 - 1)
    sc = rng.randint(c0 + 1, c1 - 1)
    grid[sr, sc] = seed_color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_legends = rng.randint(*LEGEND_COUNT_RANGE)
    n_targets = rng.randint(*TARGET_COUNT_RANGE)
    side = rng.choice(("top", "bottom", "left", "right"))

    if n_targets > n_legends and rng.random() < 0.5:
        n_targets = n_legends

    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BG, dtype=int)

    key_colors = rng.sample(list(NON_BG_COLORS), n_legends)
    pattern_pool = [c for c in NON_BG_COLORS if c not in key_colors]
    legend_specs: list[dict[str, object]] = []
    for key_color in key_colors:
        color_a, color_b = rng.sample(pattern_pool, 2)
        size = rng.choice((2, 3))
        mode = rng.choice(("checker", "hstripe", "vstripe"))
        phase = rng.randint(0, 1)
        pattern = _make_pattern(size, color_a, color_b, mode, phase)
        legend_specs.append(
            {
                "key": key_color,
                "patch": _legend_patch(key_color, pattern),
            }
        )

    rng.shuffle(legend_specs)
    reserved_band = _place_legends(grid, rng, legend_specs, side)
    if reserved_band is None:
        return None

    legend_keys = [int(spec["key"]) for spec in legend_specs]
    if rng.random() < 0.5 and n_targets <= len(legend_keys):
        seed_colors = rng.sample(legend_keys, n_targets)
    else:
        seed_colors = [rng.choice(legend_keys) for _ in range(n_targets)]

    r_min, c_min, r_max, c_max = _allowed_target_area(grid.shape, side, reserved_band)
    if r_max - r_min + 1 < TARGET_OUTER_H_RANGE[0] or c_max - c_min + 1 < TARGET_OUTER_W_RANGE[0]:
        return None

    occupied = np.zeros_like(grid, dtype=bool)
    if side == "top":
        occupied[:reserved_band, :] = True
    elif side == "bottom":
        occupied[h - reserved_band :, :] = True
    elif side == "left":
        occupied[:, :reserved_band] = True
    else:
        occupied[:, w - reserved_band :] = True

    for seed_color in seed_colors:
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            height = rng.randint(TARGET_OUTER_H_RANGE[0], min(TARGET_OUTER_H_RANGE[1], r_max - r_min + 1))
            width = rng.randint(TARGET_OUTER_W_RANGE[0], min(TARGET_OUTER_W_RANGE[1], c_max - c_min + 1))
            max_r0 = r_max - height + 1
            max_c0 = c_max - width + 1
            if max_r0 < r_min or max_c0 < c_min:
                continue
            r0 = rng.randint(r_min, max_r0)
            c0 = rng.randint(c_min, max_c0)
            r1 = r0 + height - 1
            c1 = c0 + width - 1
            if not _can_place(occupied, r0, c0, r1, c1):
                continue
            border_choices = [c for c in NON_BG_COLORS if c != seed_color]
            border_color = rng.choice(border_choices)
            _draw_target(grid, r0, c0, height, width, border_color, seed_color, rng)
            _mark_occupied(occupied, r0, c0, r1, c1)
            placed = True
            break
        if not placed:
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is None:
            continue
        try:
            out = main(grid)
        except Exception:
            continue
        if np.array_equal(out, grid):
            continue
        return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
