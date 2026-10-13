from __future__ import annotations

import random

import numpy as np


BG = 0
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = list(range(1, 10))

GRID_H_RANGE = (17, 30)
GRID_W_RANGE = (17, 30)
FRAME_COUNT_CHOICES = [1, 2, 3, 4]
FRAME_COUNT_WEIGHTS = [2, 4, 3, 1]
INNER_SIZE_RANGE = (3, 11)
FRAME_GAP = 2
OUTER_MARGIN = 1
SEED_COUNT_RANGE = (1, 5)

GEN_RETRIES = 400
FRAME_SPEC_RETRIES = 120
FRAME_PLACE_RETRIES = 140
SEED_PLACE_RETRIES = 200


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_bbox(coords: np.ndarray) -> tuple[int, int, int, int] | None:
    if coords.shape[0] < 8:
        return None
    r0, c0, r1, c1 = _bbox(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None
    if int(coords.shape[0]) != 2 * (h + w) - 4:
        return None
    pts = {(int(r), int(c)) for r, c in coords}
    for c in range(c0, c1 + 1):
        if (r0, c) not in pts or (r1, c) not in pts:
            return None
    for r in range(r0 + 1, r1):
        if (r, c0) not in pts or (r, c1) not in pts:
            return None
    if any(r0 < r < r1 and c0 < c < c1 for r, c in pts):
        return None
    return r0, c0, r1, c1


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    frames: list[tuple[int, int, int, int, int]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BG):
        for comp in _components(grid == color):
            box = _frame_bbox(comp)
            if box is not None:
                frames.append((*box, int(color)))
    frames.sort()
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow border-colored cross rays from seed pixels inside rectangular frames.

    Each frame is a 1-pixel rectangular border on a black background. Inside a
    frame there are 1-5 seed pixels of a single seed color. For a frame with N
    seeds, each seed emits four rays of length N in the cardinal directions,
    colored with the frame color. Rays stop before they would overwrite the
    frame border. Seed pixels remain unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    frames = _find_frames(g)
    assert frames, "expected at least one rectangular frame"

    for r0, c0, r1, c1, border_color in frames:
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_mask = (interior != BG) & (interior != border_color)
        assert bool(seed_mask.any()), "each frame must contain seed pixels"
        seed_colors = np.unique(interior[seed_mask])
        assert seed_colors.size == 1, "all seeds in a frame must share one color"
        assert np.all((interior == BG) | (interior == int(seed_colors[0]))), (
            "frame interior may only contain background and one seed color"
        )

        seed_positions = np.argwhere(seed_mask)
        ray_len = int(seed_positions.shape[0])
        assert 1 <= ray_len <= 5, "seed count must be between 1 and 5"

        abs_seeds = [(int(r0 + 1 + rr), int(c0 + 1 + cc)) for rr, cc in seed_positions]
        for sr, sc in abs_seeds:
            for dr, dc in DIR4:
                for step in range(1, ray_len + 1):
                    rr = sr + dr * step
                    cc = sc + dc * step
                    if not (0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]):
                        break
                    if int(g[rr, cc]) == border_color:
                        break
                    out[rr, cc] = border_color

        for sr, sc in abs_seeds:
            out[sr, sc] = g[sr, sc]

    return out


def _max_isolated_capacity(inner_h: int, inner_w: int) -> int:
    return ((inner_h + 1) // 2) * ((inner_w + 1) // 2)


def _distance_to_inner_border(r: int, c: int, inner_h: int, inner_w: int) -> int:
    return min(r, inner_h - 1 - r, c, inner_w - 1 - c)


def _non_adjacent(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) > 1


def _pick_seed_positions(inner_h: int, inner_w: int, seed_count: int, mode: str) -> list[tuple[int, int]] | None:
    all_cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    roomy = [
        cell
        for cell in all_cells
        if _distance_to_inner_border(cell[0], cell[1], inner_h, inner_w) >= seed_count
    ]
    tight = [
        cell
        for cell in all_cells
        if _distance_to_inner_border(cell[0], cell[1], inner_h, inner_w) < seed_count
    ]

    required_groups: list[list[tuple[int, int]]] = []
    if mode == "full" and roomy:
        required_groups.append(roomy)
    elif mode == "mixed" and seed_count >= 2 and roomy and tight:
        required_groups.extend([roomy, tight])
    elif mode == "tight" and tight:
        required_groups.append(tight)
    elif roomy:
        required_groups.append(roomy)
    elif tight:
        required_groups.append(tight)

    for _ in range(SEED_PLACE_RETRIES):
        chosen: list[tuple[int, int]] = []
        ok = True
        for group in required_groups:
            options = [cell for cell in group if all(_non_adjacent(cell, other) for other in chosen)]
            if not options:
                ok = False
                break
            chosen.append(random.choice(options))
        if not ok:
            continue

        remaining = all_cells[:]
        random.shuffle(remaining)
        for cell in remaining:
            if len(chosen) >= seed_count:
                break
            if all(_non_adjacent(cell, other) for other in chosen):
                chosen.append(cell)
        if len(chosen) == seed_count:
            return chosen
    return None


def _frame_size_options(seed_count: int, mode: str) -> list[tuple[int, int]]:
    options: list[tuple[int, int]] = []
    roomy_only: list[tuple[int, int]] = []
    tight_only: list[tuple[int, int]] = []

    for inner_h in range(INNER_SIZE_RANGE[0], INNER_SIZE_RANGE[1] + 1):
        for inner_w in range(INNER_SIZE_RANGE[0], INNER_SIZE_RANGE[1] + 1):
            if _max_isolated_capacity(inner_h, inner_w) < seed_count:
                continue
            roomy = inner_h >= 2 * seed_count + 1 and inner_w >= 2 * seed_count + 1
            options.append((inner_h, inner_w))
            if roomy:
                roomy_only.append((inner_h, inner_w))
            else:
                tight_only.append((inner_h, inner_w))

    if mode == "full":
        return roomy_only or options
    if mode == "tight":
        return tight_only or options
    if mode == "mixed":
        return roomy_only or options
    return options


def _choose_frame_spec(mode: str) -> tuple[int, int, int, str] | None:
    for _ in range(FRAME_SPEC_RETRIES):
        seed_count = random.randint(*SEED_COUNT_RANGE)
        seed_mode = mode
        if mode == "mixed" and seed_count == 1:
            seed_mode = random.choice(["tight", "full"])
        size_options = _frame_size_options(seed_count, seed_mode)
        if not size_options:
            continue
        inner_h, inner_w = random.choice(size_options)
        seeds = _pick_seed_positions(inner_h, inner_w, seed_count, seed_mode)
        if seeds is None:
            continue
        return inner_h, inner_w, seed_count, seed_mode
    return None


def _place_box(
    blocked: np.ndarray,
    outer_h: int,
    outer_w: int,
) -> tuple[int, int, int, int] | None:
    h, w = blocked.shape
    if outer_h + 2 * OUTER_MARGIN > h or outer_w + 2 * OUTER_MARGIN > w:
        return None
    for _ in range(FRAME_PLACE_RETRIES):
        r0 = random.randint(OUTER_MARGIN, h - outer_h - OUTER_MARGIN)
        c0 = random.randint(OUTER_MARGIN, w - outer_w - OUTER_MARGIN)
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1
        if blocked[r0 : r1 + 1, c0 : c1 + 1].any():
            continue
        br0 = max(0, r0 - FRAME_GAP)
        bc0 = max(0, c0 - FRAME_GAP)
        br1 = min(h - 1, r1 + FRAME_GAP)
        bc1 = min(w - 1, c1 + FRAME_GAP)
        blocked[br0 : br1 + 1, bc0 : bc1 + 1] = True
        return r0, c0, r1, c1
    return None


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], border_color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = border_color
    grid[r1, c0 : c1 + 1] = border_color
    grid[r0 : r1 + 1, c0] = border_color
    grid[r0 : r1 + 1, c1] = border_color


def _render_input(
    height: int,
    width: int,
    frame_specs: list[tuple[int, int, int, str]],
) -> np.ndarray | None:
    grid = np.full((height, width), BG, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    colors = PALETTE[:]
    random.shuffle(colors)
    if len(frame_specs) * 2 > len(colors):
        return None

    boxes: list[tuple[int, int, int, int]] = []
    for inner_h, inner_w, _seed_count, _mode in frame_specs:
        box = _place_box(blocked, inner_h + 2, inner_w + 2)
        if box is None:
            return None
        boxes.append(box)

    for idx, (box, spec) in enumerate(zip(boxes, frame_specs)):
        r0, c0, r1, c1 = box
        inner_h, inner_w, seed_count, seed_mode = spec
        border_color = colors[2 * idx]
        seed_color = colors[2 * idx + 1]
        _draw_frame(grid, box, border_color)

        seeds = _pick_seed_positions(inner_h, inner_w, seed_count, seed_mode)
        if seeds is None:
            return None
        for rr, cc in seeds:
            grid[r0 + 1 + rr, c0 + 1 + cc] = seed_color

    return grid


def _validate_input(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if not np.all((0 <= grid) & (grid <= 9)):
        return False

    frames = _find_frames(grid)
    if not frames:
        return False

    frame_colors: set[int] = set()
    seed_colors: set[int] = set()
    for r0, c0, r1, c1, border_color in frames:
        if border_color in frame_colors:
            return False
        frame_colors.add(border_color)

        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size == 0:
            return False
        nz = interior[interior != BG]
        if nz.size == 0:
            return False
        colors = {int(v) for v in np.unique(nz)}
        if len(colors) != 1:
            return False
        seed_color = next(iter(colors))
        if seed_color == border_color or seed_color in seed_colors or seed_color in frame_colors:
            return False
        seed_colors.add(seed_color)

        seed_positions = [(int(r), int(c)) for r, c in np.argwhere(interior == seed_color)]
        if not (1 <= len(seed_positions) <= 5):
            return False
        for i, a in enumerate(seed_positions):
            for b in seed_positions[i + 1 :]:
                if not _non_adjacent(a, b):
                    return False

    try:
        out = main(grid)
    except Exception:
        return False
    return out.shape == grid.shape


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    frame_count = random.choices(FRAME_COUNT_CHOICES, weights=FRAME_COUNT_WEIGHTS, k=1)[0]

    scene_mode = random.choice(["tight_only", "include_full", "mixed_scene"])
    if scene_mode == "mixed_scene":
        frame_count = max(frame_count, 2)

    frame_modes: list[str] = []
    if scene_mode == "tight_only":
        frame_modes = ["tight"] * frame_count
    elif scene_mode == "include_full":
        frame_modes = ["full"] + [random.choice(["tight", "full", "mixed"]) for _ in range(frame_count - 1)]
        random.shuffle(frame_modes)
    else:
        frame_modes = ["full", "tight"]
        while len(frame_modes) < frame_count:
            frame_modes.append(random.choice(["tight", "full", "mixed"]))
        random.shuffle(frame_modes)

    frame_specs: list[tuple[int, int, int, str]] = []
    for mode in frame_modes:
        spec = _choose_frame_spec(mode)
        if spec is None:
            return None
        frame_specs.append(spec)

    grid = _render_input(height, width, frame_specs)
    if grid is None:
        return None
    if _validate_input(grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
