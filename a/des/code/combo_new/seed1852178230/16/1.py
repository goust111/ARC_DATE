import random

import numpy as np


LEGEND_PATCH_SHAPE = (3, 6)
FRAME_COUNT_CHOICES = (3, 4)
GRID_MIN = 22
GRID_MAX = 30
INNER_SIZE_CHOICES = (2, 4, 6)
MAX_GENERATION_ATTEMPTS = 400
ALL_COLORS = tuple(range(10))

MASKS_BY_COUNT = {
    count: [
        tuple(int(bit) for bit in f"{bits:04b}")
        for bits in range(1, 16)
        if bin(bits).count("1") == count
    ]
    for count in (1, 2, 3)
}


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.asarray(pts, dtype=int))
    return parts


def _bbox(points: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(points[:, 0].min()),
        int(points[:, 1].min()),
        int(points[:, 0].max()),
        int(points[:, 1].max()),
    )


def _frame_bbox(points: np.ndarray) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(points)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    if h < 4 or w < 4:
        return None
    patch = np.zeros((h, w), dtype=bool)
    patch[points[:, 0] - r0, points[:, 1] - c0] = True
    expected = np.ones((h, w), dtype=bool)
    expected[1:-1, 1:-1] = False
    if np.array_equal(patch, expected):
        return (r0, c0, r1, c1)
    return None


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color0 in np.unique(grid):
        color = int(color0)
        if color == bg:
            continue
        for comp in _components(grid == color):
            bbox = _frame_bbox(comp)
            if bbox is not None:
                frames.append({"color": color, "bbox": bbox})
    frames.sort(key=lambda item: item["bbox"])
    return frames


def _legend_key_from_indicator(indicator: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    return tuple(int(x) for x in (indicator != bg).reshape(-1))


def _find_legend_patterns(grid: np.ndarray, bg: int) -> tuple[dict[tuple[int, int, int, int], np.ndarray], list[tuple[int, int]]]:
    h, w = grid.shape
    mapping: dict[tuple[int, int, int, int], np.ndarray] = {}
    starts: list[tuple[int, int]] = []
    for r0 in range(h - LEGEND_PATCH_SHAPE[0] + 1):
        for c0 in range(w - LEGEND_PATCH_SHAPE[1] + 1):
            patch = grid[r0 : r0 + 3, c0 : c0 + 6]
            sep = patch[:, 2]
            sep_color = int(sep[0])
            if sep_color == bg or not np.all(sep == sep_color):
                continue
            indicator = patch[:2, :2]
            mark_mask = indicator != bg
            marked = indicator[mark_mask]
            if not (1 <= int(mark_mask.sum()) <= 3):
                continue
            if marked.size == 0 or len(np.unique(marked)) != 1 or int(marked[0]) == sep_color:
                continue
            if not np.all(patch[2, :2] == bg):
                continue
            pattern = patch[:, 3:6]
            if np.any(pattern == bg) or np.any(pattern == sep_color):
                continue
            key = _legend_key_from_indicator(indicator, bg)
            if key in mapping and not np.array_equal(mapping[key], pattern):
                raise AssertionError("ambiguous legend pattern for the same quadrant key")
            if key not in mapping:
                mapping[key] = pattern.copy()
                starts.append((r0, c0))
    return mapping, starts


def _occupancy_key(mask: np.ndarray) -> tuple[int, int, int, int]:
    h, w = mask.shape
    key = [0, 0, 0, 0]
    for rr, cc in np.argwhere(mask):
        idx = (2 if rr * 2 >= h else 0) + (1 if cc * 2 >= w else 0)
        key[idx] = 1
    return tuple(key)


def _frame_key(grid: np.ndarray, bbox: tuple[int, int, int, int], bg: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    key = _occupancy_key(inner != bg)
    assert any(key), "frame interiors must contain at least one non-background pixel"
    return key


def _tile_ring(out: np.ndarray, bbox: tuple[int, int, int, int], pattern: np.ndarray) -> None:
    r0, c0, r1, c1 = bbox
    h, w = out.shape
    for r in range(max(0, r0 - 1), min(h, r1 + 2)):
        for c in range(max(0, c0 - 1), min(w, c1 + 2)):
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            out[r, c] = int(pattern[(r - r0) % 3, (c - c0) % 3])


def main(grid: np.ndarray) -> np.ndarray:
    """Match each hollow frame's interior quadrant pattern to a legend and tile that legend's 3x3 pattern around the frame.

    Input structure:
    - The background is the dominant color.
    - Several one-pixel-thick rectangular hollow frames appear in the grid; each frame contains a few colored interior pixels.
    - A legend cluster sits in one corner: every legend is a 3x6 patch with a 2x2 quadrant indicator on the left, a full separator column, and a 3x3 pattern block on the right.

    Rule:
    - For each frame, inspect which of the four interior quadrants contain at least one non-background pixel.
    - Find the legend whose 2x2 indicator marks the same occupied quadrants.
    - Tile that legend's 3x3 pattern over the one-cell-wide ring just outside the frame's bounding box, aligned to the frame bbox's top-left corner.
    - Keep the frame border and the interior pixels unchanged.

    Key insight:
    - The interior pixels encode a quadrant-occupancy key, and the corner legends translate that key into the pattern that must be repeated around the frame.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    legend_map, _ = _find_legend_patterns(g, bg)
    assert legend_map, "expected at least one legend patch"

    out = g.copy()
    for frame in _find_frames(g, bg):
        key = _frame_key(g, frame["bbox"], bg)
        assert key in legend_map, "every frame key must exist in the legend"
        _tile_ring(out, frame["bbox"], legend_map[key])
    return out


def _mark_blocked(blocked: np.ndarray, bbox: tuple[int, int, int, int], pad: int) -> None:
    h, w = blocked.shape
    r0, c0, r1, c1 = bbox
    blocked[max(0, r0 - pad) : min(h, r1 + pad + 1), max(0, c0 - pad) : min(w, c1 + pad + 1)] = True


def _pattern_from_mode(mode: str, palette: list[int], rng: random.Random) -> np.ndarray:
    pat = np.zeros((3, 3), dtype=int)
    if mode == "checker":
        for r in range(3):
            for c in range(3):
                pat[r, c] = palette[(r + c) % len(palette)]
    elif mode == "h_stripes":
        for r in range(3):
            pat[r, :] = palette[r % len(palette)]
    elif mode == "v_stripes":
        for c in range(3):
            pat[:, c] = palette[c % len(palette)]
    elif mode == "cross":
        pat.fill(palette[0])
        pat[1, :] = palette[1 % len(palette)]
        pat[:, 1] = palette[1 % len(palette)]
        if len(palette) > 2:
            pat[1, 1] = palette[2]
    elif mode == "diag":
        for r in range(3):
            for c in range(3):
                pat[r, c] = palette[(r + 2 * c) % len(palette)]
    else:
        for r in range(3):
            for c in range(3):
                pat[r, c] = rng.choice(palette)
    return pat


def _random_pattern(color_pool: list[int], rng: random.Random, seen: set[bytes]) -> np.ndarray:
    modes = ["checker", "h_stripes", "v_stripes", "cross", "diag", "random"]
    for _ in range(80):
        palette_size = 2 if len(color_pool) == 2 else rng.choice([2, 3])
        palette = rng.sample(color_pool, palette_size)
        pattern = _pattern_from_mode(rng.choice(modes), palette, rng)
        if len(np.unique(pattern)) < 2:
            continue
        sig = pattern.tobytes()
        if sig not in seen:
            seen.add(sig)
            return pattern
    raise RuntimeError("failed to sample a distinct legend pattern")


def _legend_patch(mask_key: tuple[int, int, int, int], pattern: np.ndarray, indicator_color: int, sep_color: int, bg: int) -> np.ndarray:
    patch = np.full(LEGEND_PATCH_SHAPE, bg, dtype=int)
    indicator = np.asarray(mask_key, dtype=int).reshape(2, 2)
    for r in range(2):
        for c in range(2):
            if indicator[r, c]:
                patch[r, c] = indicator_color
    patch[:, 2] = sep_color
    patch[:, 3:6] = pattern
    return patch


def _legend_group_layout(count: int, orientation: str) -> tuple[int, int, list[tuple[int, int]]]:
    if orientation == "vertical":
        offsets = [(idx * 4, 0) for idx in range(count)]
        return 4 * count - 1, 6, offsets
    offsets = [(0, idx * 7) for idx in range(count)]
    return 3, 7 * count - 1, offsets


def _corner_origin(grid_shape: tuple[int, int], block_shape: tuple[int, int], corner: str) -> tuple[int, int]:
    h, w = grid_shape
    bh, bw = block_shape
    if corner == "TL":
        return 1, 1
    if corner == "TR":
        return 1, w - bw - 1
    if corner == "BL":
        return h - bh - 1, 1
    return h - bh - 1, w - bw - 1


def _paste(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    grid[r0 : r0 + h, c0 : c0 + w] = patch


def _free_frame_positions(blocked: np.ndarray, frame_h: int, frame_w: int) -> list[tuple[int, int]]:
    h, w = blocked.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(2, h - frame_h - 1):
        for c0 in range(2, w - frame_w - 1):
            r1, c1 = r0 + frame_h - 1, c0 + frame_w - 1
            if blocked[r0 - 2 : r1 + 3, c0 - 2 : c1 + 3].any():
                continue
            positions.append((r0, c0))
    return positions


def _quadrant_cells(inner_h: int, inner_w: int, index: int) -> list[tuple[int, int]]:
    mid_r = inner_h // 2
    mid_c = inner_w // 2
    if index == 0:
        rows, cols = range(0, mid_r), range(0, mid_c)
    elif index == 1:
        rows, cols = range(0, mid_r), range(mid_c, inner_w)
    elif index == 2:
        rows, cols = range(mid_r, inner_h), range(0, mid_c)
    else:
        rows, cols = range(mid_r, inner_h), range(mid_c, inner_w)
    return [(r, c) for r in rows for c in cols]


def _draw_frame_with_contents(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    frame_color: int,
    marker_color: int,
    mask_key: tuple[int, int, int, int],
    rng: random.Random,
) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color
    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color

    inner_h, inner_w = r1 - r0 - 1, c1 - c0 - 1
    for idx, active in enumerate(mask_key):
        if not active:
            continue
        choices = _quadrant_cells(inner_h, inner_w, idx)
        count = rng.randint(1, min(3, len(choices)))
        for rr, cc in rng.sample(choices, count):
            grid[r0 + 1 + rr, c0 + 1 + cc] = marker_color


def _sample_masks(legend_count: int, rng: random.Random) -> list[tuple[int, int, int, int]]:
    masks = [
        rng.choice(MASKS_BY_COUNT[1]),
        rng.choice(MASKS_BY_COUNT[2]),
        rng.choice(MASKS_BY_COUNT[3]),
    ]
    used = set(masks)
    while len(masks) < legend_count:
        count = rng.choice([1, 2, 3])
        choices = [mask for mask in MASKS_BY_COUNT[count] if mask not in used]
        if not choices:
            choices = [mask for mask in MASKS_BY_COUNT[count] if mask not in used] or [
                mask
                for group in MASKS_BY_COUNT.values()
                for mask in group
                if mask not in used
            ]
        pick = rng.choice(choices)
        used.add(pick)
        masks.append(pick)
    rng.shuffle(masks)
    return masks


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(GRID_MIN, GRID_MAX)
    width = rng.randint(GRID_MIN, GRID_MAX)
    orientation = rng.choice(["vertical", "horizontal"])
    legend_count = rng.choice([3, 4]) if orientation == "vertical" else 3
    legend_h, legend_w, offsets = _legend_group_layout(legend_count, orientation)
    if legend_h + 2 >= height or legend_w + 2 >= width:
        return None

    bg = rng.choice(ALL_COLORS)
    sep = rng.choice([c for c in ALL_COLORS if c != bg])
    frame_count = rng.choice(FRAME_COUNT_CHOICES)

    fg_pool = [c for c in ALL_COLORS if c not in {bg, sep}]
    if len(fg_pool) < frame_count + 2:
        return None
    frame_colors = rng.sample(fg_pool, frame_count)
    accent_pool = [c for c in fg_pool if c not in frame_colors]
    if len(accent_pool) < 2:
        return None

    grid = np.full((height, width), bg, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    masks = _sample_masks(legend_count, rng)
    seen_patterns: set[bytes] = set()
    legend_specs = []
    for mask in masks:
        indicator_color = rng.choice(accent_pool)
        pattern = _random_pattern(accent_pool, rng, seen_patterns)
        legend_specs.append({"mask": mask, "patch": _legend_patch(mask, pattern, indicator_color, sep, bg), "pattern": pattern})

    corner = rng.choice(["TL", "TR", "BL", "BR"])
    group_r0, group_c0 = _corner_origin((height, width), (legend_h, legend_w), corner)
    if group_r0 < 1 or group_c0 < 1:
        return None
    for spec, (dr, dc) in zip(legend_specs, offsets):
        rr, cc = group_r0 + dr, group_c0 + dc
        _paste(grid, spec["patch"], rr, cc)
        _mark_blocked(blocked, (rr, cc, rr + 2, cc + 5), pad=1)

    by_count = {count: [spec for spec in legend_specs if sum(spec["mask"]) == count] for count in (1, 2, 3)}
    frame_specs = [rng.choice(by_count[1]), rng.choice(by_count[2]), rng.choice(by_count[3])]
    if frame_count == 4:
        frame_specs.append(rng.choice(legend_specs))
    rng.shuffle(frame_specs)

    expected = None
    placed_frames = []
    for idx in range(frame_count):
        inner_h = rng.choice(INNER_SIZE_CHOICES)
        inner_w = rng.choice(INNER_SIZE_CHOICES)
        frame_h, frame_w = inner_h + 2, inner_w + 2
        positions = _free_frame_positions(blocked, frame_h, frame_w)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        r1, c1 = r0 + frame_h - 1, c0 + frame_w - 1
        marker_color = rng.choice(accent_pool)
        spec = frame_specs[idx]
        _draw_frame_with_contents(grid, (r0, c0, r1, c1), frame_colors[idx], marker_color, spec["mask"], rng)
        _mark_blocked(blocked, (r0, c0, r1, c1), pad=2)
        placed_frames.append({"bbox": (r0, c0, r1, c1), "pattern": spec["pattern"], "mask": spec["mask"]})

    expected = grid.copy()
    for frame in placed_frames:
        _tile_ring(expected, frame["bbox"], frame["pattern"])

    legend_map, legend_starts = _find_legend_patterns(grid, bg)
    if len(legend_map) != legend_count or len(legend_starts) != legend_count:
        return None
    frames = _find_frames(grid, bg)
    if len(frames) != frame_count:
        return None
    for frame in frames:
        key = _frame_key(grid, frame["bbox"], bg)
        if key not in legend_map:
            return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
