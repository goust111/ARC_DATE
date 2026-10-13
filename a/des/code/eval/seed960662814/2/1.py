from __future__ import annotations

import random
from collections import Counter

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
ROTATION_BY_QUADRANT = {
    "TL": 0,
    "TR": -1,
    "BR": 2,
    "BL": 1,
}


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _square_ring_info(coords: np.ndarray, color: int) -> dict | None:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    size_r = int(r1 - r0 + 1)
    size_c = int(c1 - c0 + 1)
    if size_r != size_c or size_r < 3:
        return None

    local = np.zeros((size_r, size_c), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True

    ring = np.zeros_like(local)
    ring[0, :] = True
    ring[-1, :] = True
    ring[:, 0] = True
    ring[:, -1] = True
    if not np.array_equal(local, ring):
        return None

    return {
        "color": int(color),
        "bbox": (r0, c0, r1, c1),
        "size": int(size_r),
    }


def _detect_blocks(grid: np.ndarray) -> tuple[list[dict], list[dict], int]:
    bg = _get_bg(grid)
    ring_candidates = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for comp in _components_of_color(grid, color):
            info = _square_ring_info(comp, color)
            if info is not None:
                ring_candidates.append(info)

    assert ring_candidates, "expected square ring components"
    size_counts = Counter(info["size"] for info in ring_candidates)
    block_size = max(size_counts.items(), key=lambda item: (item[1], item[0]))[0]
    ring_candidates = [info for info in ring_candidates if info["size"] == block_size]

    palettes = []
    frames = []
    for info in ring_candidates:
        r0, c0, r1, c1 = info["bbox"]
        inner = grid[r0 + 1 : r1, c0 + 1 : c1].copy()
        non_bg = np.argwhere(inner != bg)
        if len(non_bg) == 1:
            sr, sc = map(int, non_bg[0])
            frames.append(
                {
                    "bbox": info["bbox"],
                    "size": info["size"],
                    "border_color": info["color"],
                    "seed_color": int(inner[sr, sc]),
                    "seed_local": (sr, sc),
                }
            )
        elif len(non_bg) > 1:
            palettes.append(
                {
                    "bbox": info["bbox"],
                    "size": info["size"],
                    "border_color": info["color"],
                    "pattern": inner,
                }
            )

    assert palettes, "expected palette blocks"
    assert frames, "expected frame blocks"
    return palettes, frames, bg


def _quadrant_from_seed(seed_local: tuple[int, int], inner_size: int) -> str:
    sr, sc = seed_local
    if inner_size % 2 == 1:
        mid = inner_size // 2
        assert sr != mid and sc != mid, "seed must be in a strict quadrant"
    top = sr < inner_size / 2
    left = sc < inner_size / 2
    if top and left:
        return "TL"
    if top and not left:
        return "TR"
    if not top and left:
        return "BL"
    return "BR"


def _rotate_pattern(pattern: np.ndarray, quadrant: str) -> np.ndarray:
    return np.rot90(pattern, ROTATION_BY_QUADRANT[quadrant])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select a palette pattern by seed color, then rotate it by the seed's quadrant and
    stamp it into each hollow frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    palettes, frames, _bg = _detect_blocks(g)
    sizes = {item["size"] for item in palettes + frames}
    assert len(sizes) == 1, "palette blocks and frames must share one common size"
    inner_size = next(iter(sizes)) - 2

    palette_map: dict[int, np.ndarray] = {}
    for item in palettes:
        color = int(item["border_color"])
        assert color not in palette_map, "palette border colors must be unique"
        palette_map[color] = item["pattern"]

    out = g.copy()
    for frame in frames:
        seed_color = int(frame["seed_color"])
        assert seed_color in palette_map, "every seed color must match a palette border color"
        quadrant = _quadrant_from_seed(frame["seed_local"], inner_size)
        rotated = _rotate_pattern(palette_map[seed_color], quadrant)
        r0, c0, r1, c1 = frame["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = rotated

    return out


# --------------- generator ---------------

GEN_RETRY_LIMIT = 300
GEN_GRID_RANGE = (24, 30)
GEN_BLOCK_SIZES = (5, 6, 7)
GEN_PALETTE_GAPS = (1, 2)
GEN_FRAME_COUNTS = (4, 5, 6)
GEN_MIN_PALETTES = 2
GEN_MAX_PALETTES = 4
GEN_TOP_MARGIN = (0, 1)
GEN_PALETTE_TO_FRAME_GAP = (2, 4)
GEN_BOX_PADDING = 1
GEN_PATTERN_RETRY_LIMIT = 200


def _canonical_rotation_key(mask: np.ndarray) -> bytes:
    return min(np.rot90(mask, k).tobytes() for k in range(4))


def _tight_crop(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _is_ring_mask(mask: np.ndarray) -> bool:
    cropped = _tight_crop(mask)
    h, w = cropped.shape
    if h != w or h < 3:
        return False
    ring = np.zeros_like(cropped)
    ring[0, :] = True
    ring[-1, :] = True
    ring[:, 0] = True
    ring[:, -1] = True
    return np.array_equal(cropped, ring)


def _random_connected_mask(size: int) -> np.ndarray | None:
    target = random.randint(3, min(size * size - 2, max(4, size * size // 2 + 1)))
    start = (random.randrange(size), random.randrange(size))
    cells = {start}
    while len(cells) < target:
        frontier = set()
        for r, c in cells:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))

    mask = np.zeros((size, size), dtype=bool)
    for r, c in cells:
        mask[r, c] = True

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) < 2 or len(cols) < 2:
        return None
    if len({np.rot90(mask, k).tobytes() for k in range(4)}) != 4:
        return None
    if _is_ring_mask(mask):
        return None
    return mask


def _make_patterns(inner_size: int, colors: list[int], bg: int) -> list[np.ndarray] | None:
    used: set[bytes] = set()
    patterns = []
    for color in colors:
        pattern = None
        for _ in range(GEN_PATTERN_RETRY_LIMIT):
            mask = _random_connected_mask(inner_size)
            if mask is None:
                continue
            key = _canonical_rotation_key(mask)
            if key in used:
                continue
            used.add(key)
            pattern = np.full((inner_size, inner_size), bg, dtype=int)
            pattern[mask] = int(color)
            break
        if pattern is None:
            return None
        patterns.append(pattern)
    return patterns


def _draw_ring(grid: np.ndarray, top: int, left: int, size: int, color: int) -> None:
    bottom = top + size - 1
    right = left + size - 1
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color


def _boxes_far_enough(box_a: tuple[int, int, int], box_b: tuple[int, int, int], pad: int) -> bool:
    ar, ac, asz = box_a
    br, bc, bsz = box_b
    return (
        ar + asz + pad <= br
        or br + bsz + pad <= ar
        or ac + asz + pad <= bc
        or bc + bsz + pad <= ac
    )


def _pick_seed_local(inner_size: int, quadrant: str) -> tuple[int, int]:
    mid = inner_size // 2
    if inner_size % 2 == 0:
        top_rows = range(0, mid)
        bottom_rows = range(mid, inner_size)
        left_cols = range(0, mid)
        right_cols = range(mid, inner_size)
    else:
        top_rows = range(0, mid)
        bottom_rows = range(mid + 1, inner_size)
        left_cols = range(0, mid)
        right_cols = range(mid + 1, inner_size)

    row_choices = top_rows if quadrant in ("TL", "TR") else bottom_rows
    col_choices = left_cols if quadrant in ("TL", "BL") else right_cols
    return int(random.choice(tuple(row_choices))), int(random.choice(tuple(col_choices)))


def _build_expected(
    grid: np.ndarray,
    frame_specs: list[dict],
    border_colors: list[int],
    patterns: list[np.ndarray],
    block_size: int,
) -> np.ndarray:
    out = grid.copy()
    for spec in frame_specs:
        top = int(spec["top"])
        left = int(spec["left"])
        quadrant = spec["quadrant"]
        pattern = patterns[int(spec["palette_index"])]
        out[top + 1 : top + block_size - 1, left + 1 : left + block_size - 1] = _rotate_pattern(pattern, quadrant)
    return out


def _generate_one() -> np.ndarray | None:
    height = random.randint(*GEN_GRID_RANGE)
    width = random.randint(*GEN_GRID_RANGE)
    block_size = random.choice(GEN_BLOCK_SIZES)

    gap = random.choice(GEN_PALETTE_GAPS)
    max_palettes = min(GEN_MAX_PALETTES, (width + gap) // (block_size + gap))
    if max_palettes < GEN_MIN_PALETTES:
        return None
    palette_count = random.randint(GEN_MIN_PALETTES, max_palettes)
    frame_count = random.choice(GEN_FRAME_COUNTS)

    palette_width = palette_count * block_size + (palette_count - 1) * gap
    palette_top = random.randint(*GEN_TOP_MARGIN)
    palette_left = random.randint(0, width - palette_width)

    frame_top_min = palette_top + block_size + random.randint(*GEN_PALETTE_TO_FRAME_GAP)
    if frame_top_min + block_size > height:
        return None

    palette_colors = random.sample(range(10), 2 + 2 * palette_count)
    bg = int(palette_colors[0])
    frame_color = int(palette_colors[1])
    border_colors = [int(v) for v in palette_colors[2 : 2 + palette_count]]
    pattern_colors = [int(v) for v in palette_colors[2 + palette_count : 2 + 2 * palette_count]]

    patterns = _make_patterns(block_size - 2, pattern_colors, bg)
    if patterns is None:
        return None

    grid = np.full((height, width), bg, dtype=int)
    reserved_boxes: list[tuple[int, int, int]] = []
    for idx in range(palette_count):
        left = palette_left + idx * (block_size + gap)
        top = palette_top
        reserved_boxes.append((top, left, block_size))
        _draw_ring(grid, top, left, block_size, border_colors[idx])
        grid[top + 1 : top + block_size - 1, left + 1 : left + block_size - 1] = patterns[idx]

    candidate_positions = [
        (r, c)
        for r in range(frame_top_min, height - block_size + 1)
        for c in range(0, width - block_size + 1)
    ]
    random.shuffle(candidate_positions)

    frame_boxes: list[tuple[int, int, int]] = []
    for top, left in candidate_positions:
        candidate = (top, left, block_size)
        if all(_boxes_far_enough(candidate, other, GEN_BOX_PADDING) for other in reserved_boxes + frame_boxes):
            frame_boxes.append(candidate)
            if len(frame_boxes) == frame_count:
                break
    if len(frame_boxes) != frame_count:
        return None

    quadrants = ["TL", "TR", "BR", "BL"]
    while len(quadrants) < frame_count:
        quadrants.append(random.choice(("TL", "TR", "BR", "BL")))
    random.shuffle(quadrants)

    palette_indices = list(range(palette_count))
    while len(palette_indices) < frame_count:
        palette_indices.append(random.randrange(palette_count))
    random.shuffle(palette_indices)

    frame_specs = []
    for (top, left, _size), quadrant, palette_index in zip(frame_boxes, quadrants, palette_indices, strict=False):
        _draw_ring(grid, top, left, block_size, frame_color)
        seed_r, seed_c = _pick_seed_local(block_size - 2, quadrant)
        seed_color = border_colors[palette_index]
        grid[top + 1 + seed_r, left + 1 + seed_c] = seed_color
        frame_specs.append(
            {
                "top": int(top),
                "left": int(left),
                "quadrant": quadrant,
                "palette_index": int(palette_index),
            }
        )

    expected = _build_expected(grid, frame_specs, border_colors, patterns, block_size)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
