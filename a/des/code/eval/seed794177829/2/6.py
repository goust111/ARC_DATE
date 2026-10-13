from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np


BG_COLOR = 8      # azure
FRAME_COLOR = 5   # gray
COLOR_POOL = (1, 2, 3, 4, 6, 7, 9)
SHAPE_POOL = (
    np.array([[1, 1]], dtype=int),
    np.array([[1], [1]], dtype=int),
    np.array([[1, 1, 1]], dtype=int),
    np.array([[1], [1], [1]], dtype=int),
    np.array([[1, 1], [1, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 1, 1]], dtype=int),
    np.array([[1, 1], [1, 1], [1, 1]], dtype=int),
)
GEN_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _touches_edge(coords: list[tuple[int, int]], height: int, width: int) -> bool:
    return any(r == 0 or c == 0 or r == height - 1 or c == width - 1 for r, c in coords)


def _monochrome_components(grid: np.ndarray, *, ignore: int | None = None) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            color = int(grid[r, c])
            if ignore is not None and color == ignore:
                seen[r, c] = True
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rs = [pt[0] for pt in coords]
            cs = [pt[1] for pt in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rs), min(cs), max(rs), max(cs)),
                    "touches_edge": _touches_edge(coords, h, w),
                    "area": len(coords),
                }
            )

    return comps


def _is_rectangular_frame(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return False

    coords = set(comp["coords"])
    expected = 2 * (h + w) - 4
    if len(coords) != expected:
        return False

    for c in range(c0, c1 + 1):
        if (r0, c) not in coords or (r1, c) not in coords:
            return False
    for r in range(r0, r1 + 1):
        if (r, c0) not in coords or (r, c1) not in coords:
            return False
    return True


def _component_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in comp["coords"]:
        mask[r - r0, c - c0] = True
    return mask


def _scale_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(mask.astype(np.uint8), np.ones((factor, factor), dtype=np.uint8)).astype(bool)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each seed color to the same-colored edge template, scale the template by the square-root
    area ratio, and paint the enlarged template centered inside the framed region.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common_color(g)
    out = g.copy()

    comps = _monochrome_components(g, ignore=bg)

    templates: dict[int, dict] = {}
    frames: list[dict] = []
    for comp in comps:
        if comp["touches_edge"]:
            prev = templates.get(int(comp["color"]))
            if prev is None or int(comp["area"]) > int(prev["area"]):
                templates[int(comp["color"])] = comp
        elif _is_rectangular_frame(comp):
            frames.append(comp)

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        inner_non_bg = np.argwhere(inner != bg)
        assert inner_non_bg.shape[0] == 1, "each frame interior must contain exactly one seed"

        sr, sc = map(int, inner_non_bg[0])
        seed_color = int(inner[sr, sc])
        assert seed_color in templates, "missing edge template for seed color"

        templ_mask = _component_mask(templates[seed_color])
        templ_area = int(templ_mask.sum())
        inner_h, inner_w = inner.shape
        inner_area = int(inner_h * inner_w)

        assert templ_area > 0 and inner_area % templ_area == 0, "area ratio must be integral"
        ratio = inner_area // templ_area
        k = int(math.isqrt(ratio))
        assert k * k == ratio, "area ratio must be a perfect square"

        scaled = _scale_mask(templ_mask, k)
        sh, sw = scaled.shape
        assert sh <= inner_h and sw <= inner_w, "scaled template must fit in the region interior"

        top = r0 + 1 + (inner_h - sh) // 2
        left = c0 + 1 + (inner_w - sw) // 2
        out[r0 + 1 : r1, c0 + 1 : c1] = bg
        out[top : top + sh, left : left + sw][scaled] = seed_color

    return out


def _shape_key(mask: np.ndarray) -> bytes:
    return mask.astype(np.uint8).tobytes() + bytes(mask.shape)


def _rotated_shape(mask: np.ndarray) -> np.ndarray:
    if mask.shape[0] == mask.shape[1] or random.random() < 0.5:
        return mask.copy()
    return np.rot90(mask, 1).copy()


def _reserve(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0] - 1, r0 + h - 1 + pad)
    cc1 = min(occupied.shape[1] - 1, c0 + w - 1 + pad)
    if bool(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
        return False
    occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    return True


def _edge_positions(height: int, width: int, sh: int, sw: int) -> list[tuple[int, int]]:
    pos: set[tuple[int, int]] = set()
    for c in range(width - sw + 1):
        pos.add((0, c))
        pos.add((height - sh, c))
    for r in range(height - sh + 1):
        pos.add((r, 0))
        pos.add((r, width - sw))
    out = list(pos)
    random.shuffle(out)
    return out


def _place_templates(
    grid: np.ndarray, occupied: np.ndarray, template_masks: dict[int, np.ndarray]
) -> bool:
    h, w = grid.shape
    items = list(template_masks.items())
    random.shuffle(items)

    for color, mask in items:
        sh, sw = mask.shape
        placed = False
        for r0, c0 in _edge_positions(h, w, sh, sw):
            if not _reserve(occupied, r0, c0, sh, sw, pad=1):
                continue
            grid[r0 : r0 + sh, c0 : c0 + sw][mask] = int(color)
            placed = True
            break
        if not placed:
            return False
    return True


def _region_positions(height: int, width: int, rh: int, rw: int) -> list[tuple[int, int]]:
    out = [(r, c) for r in range(1, height - rh) for c in range(1, width - rw)]
    random.shuffle(out)
    return out


def _expected_output(grid: np.ndarray) -> np.ndarray:
    return main(grid)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(19, 28)
    w = random.randint(19, 28)
    grid = np.full((h, w), BG_COLOR, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    n_templates = random.randint(2, 4)
    colors = random.sample(COLOR_POOL, n_templates)

    chosen_shapes: list[np.ndarray] = []
    while len(chosen_shapes) < n_templates:
        chosen_shapes.append(_rotated_shape(random.choice(SHAPE_POOL)))
    if len({_shape_key(shape) for shape in chosen_shapes}) == 1:
        alt = _rotated_shape(random.choice(SHAPE_POOL))
        if _shape_key(alt) != _shape_key(chosen_shapes[0]):
            chosen_shapes[-1] = alt

    template_masks = {color: np.asarray(shape, dtype=bool) for color, shape in zip(colors, chosen_shapes)}
    if not _place_templates(grid, occupied, template_masks):
        return None

    n_regions = random.randint(n_templates, min(5, n_templates + 2))
    region_colors = list(colors)
    while len(region_colors) < n_regions:
        region_colors.append(random.choice(colors))
    random.shuffle(region_colors)

    for color in region_colors:
        templ_mask = template_masks[color]
        th, tw = templ_mask.shape
        k = random.randint(1, 4)
        inner_h = int(th * k)
        inner_w = int(tw * k)
        outer_h = inner_h + 2
        outer_w = inner_w + 2

        placed = False
        for r0, c0 in _region_positions(h, w, outer_h, outer_w):
            if not _reserve(occupied, r0, c0, outer_h, outer_w, pad=1):
                continue

            grid[r0, c0 : c0 + outer_w] = FRAME_COLOR
            grid[r0 + outer_h - 1, c0 : c0 + outer_w] = FRAME_COLOR
            grid[r0 : r0 + outer_h, c0] = FRAME_COLOR
            grid[r0 : r0 + outer_h, c0 + outer_w - 1] = FRAME_COLOR

            sr = random.randint(r0 + 1, r0 + inner_h)
            sc = random.randint(c0 + 1, c0 + inner_w)
            grid[sr, sc] = int(color)
            placed = True
            break

        if not placed:
            return None

    counts = Counter(map(int, grid.ravel()))
    if counts[BG_COLOR] <= counts[FRAME_COLOR]:
        return None

    try:
        out = _expected_output(grid)
    except Exception:
        return None

    changed = int(np.count_nonzero(out != grid))
    if changed == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
