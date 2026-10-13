import random
from collections import Counter

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 300
CANONICAL_SEEDS = (
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
)
TEMPLATE_MOTIFS = ("checker", "rows", "cols", "diag", "anti", "rings")


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


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
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _is_rect_frame(grid: np.ndarray, color: int, coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return False
    if coords.shape[0] != 2 * h + 2 * w - 4:
        return False
    if not np.all(grid[r0, c0 : c1 + 1] == color):
        return False
    if not np.all(grid[r1, c0 : c1 + 1] == color):
        return False
    if not np.all(grid[r0 : r1 + 1, c0] == color):
        return False
    if not np.all(grid[r0 : r1 + 1, c1] == color):
        return False
    if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == color):
        return False
    return True


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in _components(grid == color):
            if not _is_rect_frame(grid, color, coords):
                continue
            frames.append({"border": color, "bbox": _bbox(coords)})
    frames.sort(key=lambda item: item["bbox"])
    return frames


def _seed_rotation(seed_mask: np.ndarray) -> int:
    matches = []
    for base in CANONICAL_SEEDS:
        for k in range(4):
            rotated = np.rot90(base, k)
            if rotated.shape == seed_mask.shape and np.array_equal(rotated, seed_mask):
                matches.append(k)
    if not matches:
        raise AssertionError("seed shape did not match any canonical orientation")
    if len(set(matches)) != 1:
        raise AssertionError(f"ambiguous seed rotation: {matches}")
    return matches[0]


def _extract_seed(inner: np.ndarray, bg: int) -> tuple[int, int]:
    comps = _components(inner != bg)
    if len(comps) != 1:
        raise AssertionError(f"expected exactly one seed component, got {len(comps)}")
    coords = comps[0]
    colors = {int(inner[r, c]) for r, c in coords}
    if len(colors) != 1:
        raise AssertionError("seed component must be monochrome")
    color = colors.pop()
    seed_mask = _coords_to_mask(coords)
    return color, _seed_rotation(seed_mask)


def _find_templates(grid: np.ndarray, bg: int, frames: list[dict], seed_colors: set[int]) -> dict[int, np.ndarray]:
    outside = np.ones(grid.shape, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        outside[r0 : r1 + 1, c0 : c1 + 1] = False

    mapping: dict[int, np.ndarray] = {}
    for coords in _components((grid != bg) & outside):
        r0, c0, r1, c1 = _bbox(coords)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if coords.shape[0] != patch.shape[0] * patch.shape[1]:
            continue
        if np.any(patch == bg):
            continue
        marker = int(patch[0, 0])
        if marker not in seed_colors:
            continue
        mapping[marker] = patch.copy()
    if set(mapping) != set(seed_colors):
        raise AssertionError("template mapping does not cover every seed color")
    return mapping


def _tile_patch(patch: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    reps_r = (out_h + patch.shape[0] - 1) // patch.shape[0]
    reps_c = (out_w + patch.shape[1] - 1) // patch.shape[1]
    tiled = np.tile(patch, (reps_r, reps_c))
    return tiled[:out_h, :out_w]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each framed interior by color-selecting a template and rotating it by the seed's orientation.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    frames = _find_frames(g, bg)
    if not frames:
        raise AssertionError("no framed regions found")

    seed_info = []
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_color, rotation = _extract_seed(inner, bg)
        seed_info.append((frame["bbox"], seed_color, rotation))

    templates = _find_templates(g, bg, frames, {seed_color for _, seed_color, _ in seed_info})

    out = g.copy()
    for bbox, seed_color, rotation in seed_info:
        r0, c0, r1, c1 = bbox
        stamp = np.rot90(templates[seed_color], rotation)
        fill = _tile_patch(stamp, r1 - r0 - 1, c1 - c0 - 1)
        out[r0 + 1 : r1, c0 + 1 : c1] = fill
    return out


def _shape_with_rotation(rng: random.Random) -> tuple[np.ndarray, int]:
    base = rng.choice(CANONICAL_SEEDS)
    rotation = rng.randrange(4)
    return np.rot90(base, rotation), rotation


def _template_value(motif: str, r: int, c: int, size: int, colors: list[int], shift: int) -> int:
    mod = len(colors)
    if motif == "checker":
        idx = (r + c + shift) % mod
    elif motif == "rows":
        idx = (r + shift) % mod
    elif motif == "cols":
        idx = (c + shift) % mod
    elif motif == "diag":
        idx = (r - c + shift) % mod
    elif motif == "anti":
        idx = (r + c + shift) % mod
    else:
        idx = (min(r, c, size - 1 - r, size - 1 - c) + shift) % mod
    return colors[idx]


def _make_template(size: int, marker_color: int, fill_colors: list[int], rng: random.Random) -> np.ndarray:
    motif = rng.choice(TEMPLATE_MOTIFS)
    shift = rng.randrange(len(fill_colors))
    patch = np.zeros((size, size), dtype=int)
    for r in range(size):
        for c in range(size):
            patch[r, c] = _template_value(motif, r, c, size, fill_colors, shift)
    accent_cells = [(r, c) for r in range(size) for c in range(size) if (r, c) != (0, 0)]
    rng.shuffle(accent_cells)
    for r, c in accent_cells[: rng.randint(1, max(1, size))]:
        patch[r, c] = rng.choice(fill_colors)
    patch[0, 0] = marker_color
    if len(set(int(v) for v in patch.flatten() if int(v) != marker_color)) < 2:
        patch[-1, -1] = fill_colors[0]
        patch[-1, 0] = fill_colors[-1]
    return patch


def _is_free(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int, halo: int = 1) -> bool:
    h, w = occupied.shape
    rr0 = max(0, r0 - halo)
    cc0 = max(0, c0 - halo)
    rr1 = min(h - 1, r1 + halo)
    cc1 = min(w - 1, c1 + halo)
    return not bool(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _reserve(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int, halo: int = 1) -> None:
    h, w = occupied.shape
    rr0 = max(0, r0 - halo)
    cc0 = max(0, c0 - halo)
    rr1 = min(h - 1, r1 + halo)
    cc1 = min(w - 1, c1 + halo)
    occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _place_template_group(
    grid: np.ndarray,
    occupied: np.ndarray,
    blocks: list[np.ndarray],
    corner: str,
    rng: random.Random,
) -> None:
    h, w = grid.shape
    size = int(blocks[0].shape[0])
    gap = 1
    rows = min(len(blocks), rng.choice([1, 2]))
    cols = (len(blocks) + rows - 1) // rows
    group_h = rows * size + (rows - 1) * gap
    group_w = cols * size + (cols - 1) * gap
    margin = rng.randint(0, 1)

    if corner == "tl":
        gr0, gc0 = margin, margin
    elif corner == "tr":
        gr0, gc0 = margin, w - margin - group_w
    elif corner == "bl":
        gr0, gc0 = h - margin - group_h, margin
    else:
        gr0, gc0 = h - margin - group_h, w - margin - group_w

    order = blocks[:]
    rng.shuffle(order)
    for idx, block in enumerate(order):
        rr = idx // cols
        cc = idx % cols
        r0 = gr0 + rr * (size + gap)
        c0 = gc0 + cc * (size + gap)
        grid[r0 : r0 + size, c0 : c0 + size] = block

    _reserve(occupied, gr0, gc0, gr0 + group_h - 1, gc0 + group_w - 1, halo=1)


def _seed_offsets(inner_size: int, seed_size: int) -> tuple[int, int]:
    slack = inner_size - seed_size
    if slack <= 0:
        return 0, 0
    if slack >= 2:
        return 1, slack - 1
    return 0, slack


def _draw_frame(
    grid: np.ndarray,
    occupied: np.ndarray,
    border_color: int,
    seed_color: int,
    seed_mask: np.ndarray,
    inner_h: int,
    inner_w: int,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    seed_h, seed_w = seed_mask.shape

    for _ in range(400):
        r0 = rng.randint(0, h - outer_h)
        c0 = rng.randint(0, w - outer_w)
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1
        if not _is_free(occupied, r0, c0, r1, c1, halo=1):
            continue

        grid[r0, c0 : c1 + 1] = border_color
        grid[r1, c0 : c1 + 1] = border_color
        grid[r0 : r1 + 1, c0] = border_color
        grid[r0 : r1 + 1, c1] = border_color

        sr_min, sr_max = _seed_offsets(inner_h, seed_h)
        sc_min, sc_max = _seed_offsets(inner_w, seed_w)
        sr = rng.randint(sr_min, sr_max)
        sc = rng.randint(sc_min, sc_max)
        window = grid[r0 + 1 + sr : r0 + 1 + sr + seed_h, c0 + 1 + sc : c0 + 1 + sc + seed_w]
        window[seed_mask] = seed_color

        _reserve(occupied, r0, c0, r1, c1, halo=1)
        return True
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    num_frames = rng.randint(2, 4)
    template_size = rng.choice([3, 4])
    grid_h = rng.randint(24, 32)
    grid_w = rng.randint(24, 32)

    palette = list(range(10))
    bg = rng.choice(palette)
    palette.remove(bg)
    border_color = rng.choice(palette)
    palette.remove(border_color)
    seed_colors = rng.sample(palette, num_frames)
    for color in seed_colors:
        palette.remove(color)
    fill_count = 2 if rng.random() < 0.5 else 3
    fill_colors = rng.sample(palette, fill_count)

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    blocks = [_make_template(template_size, seed_color, fill_colors, rng) for seed_color in seed_colors]
    corner = rng.choice(["tl", "tr", "bl", "br"])
    _place_template_group(grid, occupied, blocks, corner, rng)

    specs = []
    for seed_color in seed_colors:
        seed_mask, _rotation = _shape_with_rotation(rng)
        seed_h, seed_w = seed_mask.shape
        inner_h = rng.randint(max(4, seed_h + 2), 9)
        inner_w = rng.randint(max(4, seed_w + 2), 10)
        specs.append(
            {
                "area": (inner_h + 2) * (inner_w + 2),
                "seed_color": seed_color,
                "seed_mask": seed_mask,
                "inner_h": inner_h,
                "inner_w": inner_w,
            }
        )

    specs.sort(key=lambda item: item["area"], reverse=True)
    for spec in specs:
        if not _draw_frame(
            grid,
            occupied,
            border_color,
            spec["seed_color"],
            spec["seed_mask"],
            spec["inner_h"],
            spec["inner_w"],
            rng,
        ):
            return None

    bg_count = int(np.count_nonzero(grid == bg))
    if bg_count <= grid.size // 2:
        return None

    frames = _find_frames(grid, bg)
    if len(frames) != num_frames:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if not np.any(out != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
