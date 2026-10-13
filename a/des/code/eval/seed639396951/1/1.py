import random
from collections import deque

import numpy as np


BG = 0
SEP = 5
FRAME_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIR_ORDER = ["up", "right", "down", "left"]
DIR_TO_GAP = {
    "up": (0, 1),
    "right": (1, 2),
    "down": (2, 1),
    "left": (1, 0),
}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _find_separator(grid: np.ndarray) -> int:
    h, w = grid.shape
    for c in range(1, w - 1):
        col = grid[:, c]
        if col[0] == BG:
            continue
        if np.all(col == col[0]) and np.any(grid[:, :c] != BG) and np.any(grid[:, c + 1 :] != BG):
            return c
    raise AssertionError("separator column not found")


def _read_legend(legend: np.ndarray) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for comp in _components(legend != BG):
        r0, c0, r1, c1 = _bbox(comp)
        if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
            continue
        block = legend[r0 : r1 + 1, c0 : c1 + 1]
        color = int(legend[comp[0][0], comp[0][1]])
        if len(comp) != 8 or any(int(legend[r, c]) != color for r, c in comp):
            continue
        missing = [(r, c) for r in range(3) for c in range(3) if block[r, c] == BG]
        if len(missing) != 1:
            continue
        gap = missing[0]
        if gap not in DIR_TO_GAP.values():
            continue
        direction = next(name for name, pos in DIR_TO_GAP.items() if pos == gap)
        mapping[color] = direction
    if not mapping:
        raise AssertionError("legend mapping not found")
    return mapping


def _is_frame(main_grid: np.ndarray, color: int, coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return False
    block = main_grid[r0 : r1 + 1, c0 : c1 + 1]
    if not np.all(block[0, :] == color):
        return False
    if not np.all(block[-1, :] == color):
        return False
    if not np.all(block[:, 0] == color):
        return False
    if not np.all(block[:, -1] == color):
        return False
    if np.any(block[1:-1, 1:-1] != BG):
        return False
    expected = 2 * h + 2 * w - 4
    return len(coords) == expected


def _find_frames(main_grid: np.ndarray) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(main_grid) if v not in (BG, SEP)):
        for comp in _components(main_grid == color):
            if _is_frame(main_grid, color, comp):
                frames.append({"color": color, "bbox": _bbox(comp), "coords": comp})
    if not frames:
        raise AssertionError("no frames found")
    return frames


def _pattern_component(mask: np.ndarray, seeds: list[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = mask.shape
    q = deque(seeds)
    seen = set(seeds)
    comp: list[tuple[int, int]] = []
    while q:
        r, c = q.popleft()
        comp.append((r, c))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return comp


def _extract_pattern(
    main_grid: np.ndarray, frame: dict[str, object], pattern_mask: np.ndarray
) -> tuple[str, list[tuple[int, int]], np.ndarray]:
    r0, c0, r1, c1 = frame["bbox"]  # type: ignore[index]
    h, w = main_grid.shape
    side_seeds: dict[str, list[tuple[int, int]]] = {}
    if r0 > 0:
        seeds = [(r0 - 1, c) for c in range(c0, c1 + 1) if pattern_mask[r0 - 1, c]]
        if seeds:
            side_seeds["up"] = seeds
    if c1 + 1 < w:
        seeds = [(r, c1 + 1) for r in range(r0, r1 + 1) if pattern_mask[r, c1 + 1]]
        if seeds:
            side_seeds["right"] = seeds
    if r1 + 1 < h:
        seeds = [(r1 + 1, c) for c in range(c0, c1 + 1) if pattern_mask[r1 + 1, c]]
        if seeds:
            side_seeds["down"] = seeds
    if c0 > 0:
        seeds = [(r, c0 - 1) for r in range(r0, r1 + 1) if pattern_mask[r, c0 - 1]]
        if seeds:
            side_seeds["left"] = seeds
    if len(side_seeds) != 1:
        raise AssertionError("frame must have exactly one attached side")
    side, seeds = next(iter(side_seeds.items()))
    comp = _pattern_component(pattern_mask, seeds)
    pr0, pc0, pr1, pc1 = _bbox(comp)
    patch = np.zeros((pr1 - pr0 + 1, pc1 - pc0 + 1), dtype=int)
    for r, c in comp:
        patch[r - pr0, c - pc0] = int(main_grid[r, c])
    return side, comp, patch


def _rotation_k(source_side: str, target_side: str) -> int:
    return (DIR_ORDER.index(target_side) - DIR_ORDER.index(source_side)) % 4


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    sep_col = _find_separator(g)
    legend = g[:, :sep_col]
    main_grid = g[:, sep_col + 1 :].copy()

    legend_dirs = _read_legend(legend)
    frames = _find_frames(main_grid)

    frame_mask = np.zeros_like(main_grid, dtype=bool)
    for frame in frames:
        for r, c in frame["coords"]:  # type: ignore[index]
            frame_mask[r, c] = True

    out = main_grid.copy()
    pattern_mask = (main_grid != BG) & (~frame_mask)
    used_patterns = np.zeros_like(main_grid, dtype=bool)

    for frame in frames:
        color = int(frame["color"])  # type: ignore[index]
        if color not in legend_dirs:
            raise AssertionError("frame color missing from legend")
        source_side, comp, patch = _extract_pattern(main_grid, frame, pattern_mask & (~used_patterns))
        for r, c in comp:
            used_patterns[r, c] = True
            out[r, c] = BG

        rotated = np.rot90(patch, _rotation_k(source_side, legend_dirs[color]))

        r0, c0, r1, c1 = frame["bbox"]  # type: ignore[index]
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        ph, pw = rotated.shape
        if ph > inner_h or pw > inner_w:
            raise AssertionError("rotated pattern does not fit frame interior")
        top = r0 + 1 + (inner_h - ph) // 2
        left = c0 + 1 + (inner_w - pw) // 2
        mask = rotated != BG
        out[top : top + ph, left : left + pw][mask] = rotated[mask]

    return out


def _random_pattern(size: int, palette: list[int], rng: random.Random) -> np.ndarray:
    for _ in range(200):
        num_colors = 2 if len(palette) == 2 else rng.randint(2, min(3, len(palette)))
        colors = rng.sample(palette, num_colors)
        arr = np.array([[rng.choice(colors) for _ in range(size)] for _ in range(size)], dtype=int)
        if len(set(int(v) for v in arr.flat)) < 2:
            continue
        if any(np.array_equal(arr, np.rot90(arr, k)) for k in (1, 2, 3)):
            continue
        return arr
    raise RuntimeError("failed to sample asymmetric pattern")


def _sample_dirs(n: int, rng: random.Random) -> list[str]:
    dirs = DIR_ORDER.copy()
    rng.shuffle(dirs)
    return dirs[:n]


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    grid[r0, c0 : c0 + w] = color
    grid[r0 + h - 1, c0 : c0 + w] = color
    grid[r0 : r0 + h, c0] = color
    grid[r0 : r0 + h, c0 + w - 1] = color


def _legend_block(color: int, direction: str) -> np.ndarray:
    block = np.full((3, 3), color, dtype=int)
    gr, gc = DIR_TO_GAP[direction]
    block[gr, gc] = BG
    return block


def _place_rect(
    occ: np.ndarray, h: int, w: int, rng: random.Random, pad: int = 1
) -> tuple[int, int] | None:
    H, W = occ.shape
    if h + 2 * pad > H or w + 2 * pad > W:
        return None
    for _ in range(300):
        r = rng.randint(pad, H - h - pad)
        c = rng.randint(pad, W - w - pad)
        rr0 = max(0, r - pad)
        cc0 = max(0, c - pad)
        rr1 = min(H, r + h + pad)
        cc1 = min(W, c + w + pad)
        if np.any(occ[rr0:rr1, cc0:cc1]):
            continue
        occ[rr0:rr1, cc0:cc1] = True
        return r, c
    return None


def _build_case(rng: random.Random) -> np.ndarray | None:
    num_frames = rng.randint(2, 4)
    h = rng.randint(18, 28)
    main_w = rng.randint(18, 28)

    legend_cols = 1 if num_frames <= 2 or rng.random() < 0.5 else 2
    legend_rows = (num_frames + legend_cols - 1) // legend_cols
    legend_core_h = legend_rows * 3 + (legend_rows - 1)
    legend_core_w = legend_cols * 3 + (legend_cols - 1)
    legend_w = legend_core_w + rng.randint(1, 3)
    legend_h = legend_core_h
    if legend_h > h - 2:
        return None

    total_w = legend_w + 1 + main_w
    grid = np.zeros((h, total_w), dtype=int)
    grid[:, legend_w] = SEP
    main_grid = grid[:, legend_w + 1 :]

    frame_colors = rng.sample(FRAME_COLORS, num_frames)
    pattern_palette = [c for c in FRAME_COLORS if c not in frame_colors]
    if len(pattern_palette) < 2:
        return None

    target_dirs = _sample_dirs(num_frames, rng)
    source_dirs = _sample_dirs(num_frames, rng)

    occ = np.zeros_like(main_grid, dtype=bool)
    expected = main_grid.copy()
    specs: list[tuple[int, str, str, list[tuple[int, int]], np.ndarray, tuple[int, int, int, int]]] = []

    for idx in range(num_frames):
        size = rng.choice([2, 3])
        pattern = _random_pattern(size, pattern_palette, rng)
        source_side = source_dirs[idx]
        target_side = target_dirs[idx]

        inner_h = size + 2 * rng.randint(1, 3)
        inner_w = size + 2 * rng.randint(1, 3)
        frame_h = inner_h + 2
        frame_w = inner_w + 2

        if source_side in ("up", "down"):
            obj_h, obj_w = frame_h + size, frame_w
        else:
            obj_h, obj_w = frame_h, frame_w + size

        placed = _place_rect(occ, obj_h, obj_w, rng, pad=1)
        if placed is None:
            return None
        obj_r, obj_c = placed

        if source_side == "up":
            fr0, fc0 = obj_r + size, obj_c
            pr0, pc0 = obj_r, obj_c + (frame_w - size) // 2
        elif source_side == "down":
            fr0, fc0 = obj_r, obj_c
            pr0, pc0 = obj_r + frame_h, obj_c + (frame_w - size) // 2
        elif source_side == "left":
            fr0, fc0 = obj_r, obj_c + size
            pr0, pc0 = obj_r + (frame_h - size) // 2, obj_c
        else:
            fr0, fc0 = obj_r, obj_c
            pr0, pc0 = obj_r + (frame_h - size) // 2, obj_c + frame_w

        frame_color = frame_colors[idx]
        _draw_frame(main_grid, fr0, fc0, frame_h, frame_w, frame_color)
        main_grid[pr0 : pr0 + size, pc0 : pc0 + size] = pattern

        rotated = np.rot90(pattern, _rotation_k(source_side, target_side))
        ir0, ic0 = fr0 + 1, fc0 + 1
        top = ir0 + (inner_h - size) // 2
        left = ic0 + (inner_w - size) // 2
        expected[fr0 : fr0 + frame_h, fc0 : fc0 + frame_w] = main_grid[fr0 : fr0 + frame_h, fc0 : fc0 + frame_w]
        expected[pr0 : pr0 + size, pc0 : pc0 + size] = BG
        expected[top : top + size, left : left + size] = rotated

        pattern_cells = [(r, c) for r in range(pr0, pr0 + size) for c in range(pc0, pc0 + size)]
        specs.append((frame_color, source_side, target_side, pattern_cells, pattern.copy(), (fr0, fc0, frame_h, frame_w)))

    top_offset = rng.randint(1, h - legend_h - 1)
    left_offset = rng.randint(0, legend_w - legend_core_w)
    for idx, frame_color in enumerate(frame_colors):
        rr = idx // legend_cols
        cc = idx % legend_cols
        r0 = top_offset + rr * 4
        c0 = left_offset + cc * 4
        grid[r0 : r0 + 3, c0 : c0 + 3] = _legend_block(frame_color, target_dirs[idx])

    full = grid
    solved = main(full)
    if not np.array_equal(solved, expected):
        return None
    return full


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _build_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
