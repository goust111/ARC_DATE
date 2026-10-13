import random

import numpy as np


GREY = 5
MAX_GEN_ATTEMPTS = 400
GRID_H_MIN, GRID_H_MAX = 24, 32
GRID_W_MIN, GRID_W_MAX = 24, 32
EDGE_NAMES = ("top", "bottom", "left", "right")

SHAPE_LIBRARY = (
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
)


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def _bbox(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _tight_mask(cells):
    r0, c0, r1, c1 = _bbox(cells)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def _touches_border(cells, h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells)


def _is_rect_frame(cells) -> bool:
    mask, _ = _tight_mask(cells)
    h, w = mask.shape
    if h < 3 or w < 3:
        return False
    if len(cells) != 2 * (h + w) - 4:
        return False
    target = np.zeros_like(mask)
    target[0, :] = True
    target[-1, :] = True
    target[:, 0] = True
    target[:, -1] = True
    return bool(np.array_equal(mask, target))


def _scene_parts(grid: np.ndarray):
    h, w = grid.shape
    bg = _dominant_color(grid)
    frames = []
    templates = {}
    for value in np.unique(grid):
        color = int(value)
        if color == bg:
            continue
        for cells in _components(grid == color):
            if _is_rect_frame(cells) and not _touches_border(cells, h, w):
                frames.append((color, _bbox(cells)))
                continue
            if _touches_border(cells, h, w) and 3 <= len(cells) <= 5:
                if color in templates:
                    raise AssertionError("template colors must be unique")
                templates[color] = _tight_mask(cells)[0]
    frames.sort(key=lambda item: item[1])
    return bg, frames, templates


def _scaled_mask(mask: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    in_h, in_w = mask.shape
    assert out_h % in_h == 0 and out_w % in_w == 0
    scale_h = out_h // in_h
    scale_w = out_w // in_w
    return np.repeat(np.repeat(mask, scale_h, axis=0), scale_w, axis=1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each container's seed color to the same-colored edge template and
    scale that template to the container's inner rectangle.
    """
    g = np.asarray(grid, dtype=int)
    bg, frames, templates = _scene_parts(g)
    assert frames, "expected at least one grey frame"
    assert templates, "expected at least one edge template"

    out = g.copy()
    for frame_color, (r0, c0, r1, c1) in frames:
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_coords = np.argwhere((inner != bg) & (inner != frame_color))
        assert seed_coords.shape[0] == 1, "each frame must contain exactly one seed"
        sr, sc = map(int, seed_coords[0])
        seed_color = int(inner[sr, sc])
        assert seed_color in templates, "every seed color must have a border template"

        scaled = _scaled_mask(templates[seed_color], inner.shape[0], inner.shape[1])
        patch = np.full(inner.shape, bg, dtype=int)
        patch[scaled] = seed_color
        out[r0 + 1 : r1, c0 + 1 : c1] = patch

    return out


def _random_transform(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    out = mask.copy()
    if rng.random() < 0.5:
        out = np.fliplr(out)
    return np.rot90(out, rng.randrange(4)).copy()


def _paint_mask(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int):
    patch = grid[top : top + mask.shape[0], left : left + mask.shape[1]]
    patch[mask] = color


def _can_reserve(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int) -> bool:
    h, w = occupied.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return not bool(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _reserve(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int):
    h, w = occupied.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _place_edge_template(
    grid: np.ndarray,
    occupied: np.ndarray,
    mask: np.ndarray,
    color: int,
    edge: str,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    bh, bw = mask.shape

    if edge in ("top", "bottom"):
        min_c = 2
        max_c = w - bw - 2
        if max_c < min_c:
            return False
        candidates = list(range(min_c, max_c + 1))
        rng.shuffle(candidates)
        r0 = 0 if edge == "top" else h - bh
        for c0 in candidates:
            if not _can_reserve(occupied, r0, c0, r0 + bh - 1, c0 + bw - 1, pad=1):
                continue
            _paint_mask(grid, r0, c0, mask, color)
            _reserve(occupied, r0, c0, r0 + bh - 1, c0 + bw - 1, pad=1)
            return True
        return False

    min_r = 2
    max_r = h - bh - 2
    if max_r < min_r:
        return False
    candidates = list(range(min_r, max_r + 1))
    rng.shuffle(candidates)
    c0 = 0 if edge == "left" else w - bw
    for r0 in candidates:
        if not _can_reserve(occupied, r0, c0, r0 + bh - 1, c0 + bw - 1, pad=1):
            continue
        _paint_mask(grid, r0, c0, mask, color)
        _reserve(occupied, r0, c0, r0 + bh - 1, c0 + bw - 1, pad=1)
        return True
    return False


def _draw_frame(grid: np.ndarray, r0: int, c0: int, inner_h: int, inner_w: int, seed_color: int, rng: random.Random):
    frame_h = inner_h + 2
    frame_w = inner_w + 2
    grid[r0, c0 : c0 + frame_w] = GREY
    grid[r0 + frame_h - 1, c0 : c0 + frame_w] = GREY
    grid[r0 : r0 + frame_h, c0] = GREY
    grid[r0 : r0 + frame_h, c0 + frame_w - 1] = GREY

    sr = r0 + 1 + rng.randrange(inner_h)
    sc = c0 + 1 + rng.randrange(inner_w)
    grid[sr, sc] = seed_color


def _place_container(
    grid: np.ndarray,
    occupied: np.ndarray,
    inner_h: int,
    inner_w: int,
    seed_color: int,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    frame_h = inner_h + 2
    frame_w = inner_w + 2
    min_r = 3
    max_r = h - frame_h - 3
    min_c = 3
    max_c = w - frame_w - 3
    if max_r < min_r or max_c < min_c:
        return False

    candidates = [(r0, c0) for r0 in range(min_r, max_r + 1) for c0 in range(min_c, max_c + 1)]
    rng.shuffle(candidates)
    for r0, c0 in candidates:
        if not _can_reserve(occupied, r0, c0, r0 + frame_h - 1, c0 + frame_w - 1, pad=1):
            continue
        _draw_frame(grid, r0, c0, inner_h, inner_w, seed_color, rng)
        _reserve(occupied, r0, c0, r0 + frame_h - 1, c0 + frame_w - 1, pad=1)
        return True
    return False


def _gen_attempt(rng: random.Random):
    n_containers = rng.randint(3, 4)
    edges = list(rng.sample(EDGE_NAMES, n_containers))

    palette = [color for color in range(10) if color != GREY]
    bg = rng.choice(palette)
    palette.remove(bg)
    seed_colors = rng.sample(palette, n_containers)

    specs = []
    base_indices = rng.sample(range(len(SHAPE_LIBRARY)), n_containers)
    for color, base_index, edge in zip(seed_colors, base_indices, edges):
        mask = _random_transform(SHAPE_LIBRARY[base_index], rng)
        scale = rng.choice([2, 3])
        if n_containers == 3 and mask.size <= 6 and rng.random() < 0.35:
            scale = 4
        specs.append(
            {
                "color": color,
                "edge": edge,
                "mask": mask,
                "inner_h": mask.shape[0] * scale,
                "inner_w": mask.shape[1] * scale,
            }
        )

    total_frame_area = sum((spec["inner_h"] + 2) * (spec["inner_w"] + 2) for spec in specs)

    h = rng.randint(GRID_H_MIN, GRID_H_MAX)
    w = rng.randint(GRID_W_MIN, GRID_W_MAX)
    if total_frame_area > (h - 6) * (w - 6) * 0.55:
        return None

    grid = np.full((h, w), bg, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    rng.shuffle(specs)
    for spec in specs:
        if not _place_edge_template(grid, occupied, spec["mask"], spec["color"], spec["edge"], rng):
            return None

    specs.sort(key=lambda spec: (spec["inner_h"] + 2) * (spec["inner_w"] + 2), reverse=True)
    for spec in specs:
        if not _place_container(grid, occupied, spec["inner_h"], spec["inner_w"], spec["color"], rng):
            return None

    values, counts = np.unique(grid, return_counts=True)
    count_map = {int(value): int(count) for value, count in zip(values, counts)}
    if count_map[bg] <= max(count for color, count in count_map.items() if color != bg):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if np.count_nonzero(out != grid) < 8:
        return None
    return grid


def generate_input() -> np.ndarray:
    """
    Randomly generate framed seed containers and matching edge templates.
    """
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {MAX_GEN_ATTEMPTS} attempts")
