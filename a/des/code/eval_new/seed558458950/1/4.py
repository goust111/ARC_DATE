"""Fill each enclosed region with a tiled texture selected by a seed color and anchored at the seed.
Input structure:
- Background is black (0). Grey lines draw closed boundaries that separate interior regions.
- Each enclosed region contains one non-black, non-grey seed pixel, and all other pixels in the region are background.
- Along one grid side there are several small grey framed texture samples; the interior is a repeated 2×2 or 3×3 texture.
- The top-left interior pixel of each sample is a color tag that corresponds to a region seed color.

Rule:
- Build a map from texture-tag color to its tile pattern by detecting grey rectangular frame components on the sample side.
- Find all background components that do not touch the outer grid border; each is one enclosed region to fill.
- For each such region, find the adjacent seed pixel whose color exists in the texture map.
- Tile the mapped texture into the whole region so that the seed position aligns to the tile's [0, 0] cell.
- Replace region background pixels and the seed itself with the texture value at the anchored phase.
- Keep grey walls and template frames unchanged.

Key insight:
- The seed color selects which texture to use, while the seed position fixes the tiling phase, so equal textures can appear with different offsets in different regions.
"""

from collections import deque
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np


def _neighbors(r: int, c: int, h: int, w: int) -> Iterable[Tuple[int, int]]:
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _flood_components(mask: np.ndarray) -> Iterable[Tuple[List[Tuple[int, int]], Tuple[int, int, int, int]]]:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    for sr in range(h):
        for sc in range(w):
            if not mask[sr, sc] or visited[sr, sc]:
                continue
            q: deque[Tuple[int, int]] = deque([(sr, sc)])
            visited[sr, sc] = True
            comp: List[Tuple[int, int]] = []
            min_r, max_r = sr, sr
            min_c, max_c = sc, sc
            while q:
                r, c = q.popleft()
                comp.append((r, c))
                if r < min_r:
                    min_r = r
                if r > max_r:
                    max_r = r
                if c < min_c:
                    min_c = c
                if c > max_c:
                    max_c = c
                for nr, nc in _neighbors(r, c, h, w):
                    if mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            yield comp, (min_r, max_r, min_c, max_c)


def _is_rectangle_border(
    grid: np.ndarray,
    color: int,
    min_r: int,
    max_r: int,
    min_c: int,
    max_c: int,
    comp_size: int,
) -> bool:
    h = max_r - min_r + 1
    w = max_c - min_c + 1
    if h < 3 or w < 3:
        return False
    expected = 2 * (h + w - 2)
    if comp_size != expected:
        return False
    sub = grid[min_r : max_r + 1, min_c : max_c + 1]
    if not np.all(sub[0, :] == color):
        return False
    if not np.all(sub[-1, :] == color):
        return False
    if not np.all(sub[:, 0] == color):
        return False
    if not np.all(sub[:, -1] == color):
        return False
    if h > 2 and w > 2:
        if np.any(sub[1:-1, 1:-1] == color):
            return False
    return True


def _extract_templates(grid: np.ndarray, bg: int) -> Dict[int, np.ndarray]:
    h, w = grid.shape
    frame_candidates: List[Tuple[int, int, int, int, int]] = []
    colors = np.unique(grid)

    for color in colors:
        c = int(color)
        if c == bg:
            continue
        mask = grid == c
        for comp, (min_r, max_r, min_c, max_c) in _flood_components(mask):
            if _is_rectangle_border(grid, c, min_r, max_r, min_c, max_c, len(comp)):
                frame_candidates.append((c, min_r, max_r, min_c, max_c))

    if not frame_candidates:
        return {}

    side_candidates = []
    for c, min_r, max_r, min_c, max_c in frame_candidates:
        d_top = min_r
        d_bottom = h - 1 - max_r
        d_left = min_c
        d_right = w - 1 - max_c
        if min(d_top, d_bottom, d_left, d_right) <= 1:
            side_candidates.append((c, min_r, max_r, min_c, max_c))

    use_candidates = side_candidates if side_candidates else frame_candidates
    templates: Dict[int, np.ndarray] = {}

    for c, min_r, max_r, min_c, max_c in use_candidates:
        inner = grid[min_r + 1 : max_r, min_c + 1 : max_c]
        if inner.size == 0:
            continue
        tag = int(inner[0, 0])
        if tag == bg or tag == c:
            continue
        tile = _infer_tile(inner, tag, bg)
        if tile is None:
            continue
        if tag not in templates:
            templates[tag] = tile

    return templates


def _infer_tile(inner: np.ndarray, tag_color: int, bg: int) -> Optional[np.ndarray]:
    h, w = inner.shape
    if h <= 0 or w <= 0 or tag_color == bg:
        return None

    for size in (2, 3):
        if h % size != 0 or w % size != 0:
            continue
        tile = inner[:size, :size]
        rep = np.tile(tile, (h // size, w // size))
        if np.array_equal(inner, rep):
            return tile

    if h <= 3 and w <= 3:
        return inner.copy()

    best: Optional[np.ndarray] = None
    best_err = 10.0
    for size in (2, 3):
        if h < size or w < size:
            continue
        tile = inner[:size, :size]
        rep_h = int(np.ceil(h / size))
        rep_w = int(np.ceil(w / size))
        rep = np.tile(tile, (rep_h, rep_w))[:h, :w]
        err = float(np.mean(rep != inner))
        if err < best_err:
            best_err = err
            best = tile.copy()
    return best


def _collect_enclosed_background_regions(grid: np.ndarray, bg: int) -> List[Set[Tuple[int, int]]]:
    h, w = grid.shape
    mask = grid == bg
    visited = np.zeros((h, w), dtype=bool)
    regions: List[Set[Tuple[int, int]]] = []

    for sr in range(h):
        for sc in range(w):
            if not mask[sr, sc] or visited[sr, sc]:
                continue
            q: deque[Tuple[int, int]] = deque([(sr, sc)])
            visited[sr, sc] = True
            comp: Set[Tuple[int, int]] = set()
            touches_border = False
            while q:
                r, c = q.popleft()
                comp.add((r, c))
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches_border = True
                for nr, nc in _neighbors(r, c, h, w):
                    if mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            if not touches_border:
                regions.append(comp)

    return regions


def _find_region_seed(
    grid: np.ndarray,
    region: Set[Tuple[int, int]],
    bg: int,
    valid_tags: Set[int],
) -> Optional[Tuple[int, int, int]]:
    h, w = grid.shape
    for r, c in region:
        for nr, nc in _neighbors(r, c, h, w):
            color = int(grid[nr, nc])
            if color in valid_tags and color != bg:
                return nr, nc, color
    return None


def main(grid: np.ndarray) -> np.ndarray:
    inp = np.array(grid, dtype=int)
    if inp.ndim != 2:
        raise ValueError("main expects a 2D grid")

    bg = 0
    templates = _extract_templates(inp, bg)
    if not templates:
        return inp.copy()

    out = inp.copy()
    valid_tags = set(templates.keys())

    for region in _collect_enclosed_background_regions(inp, bg):
        found = _find_region_seed(inp, region, bg, valid_tags)
        if found is None:
            continue
        sr, sc, tag = found
        tile = templates[tag]
        th, tw = tile.shape
        for r, c in region:
            out[r, c] = tile[(r - sr) % th, (c - sc) % tw]
        out[sr, sc] = tile[0, 0]

    return out


def _draw_rect_frame(grid: np.ndarray, y0: int, y1: int, x0: int, x1: int, wall: int) -> None:
    grid[y0, x0 : x1 + 1] = wall
    grid[y1, x0 : x1 + 1] = wall
    grid[y0 : y1 + 1, x0] = wall
    grid[y0 : y1 + 1, x1] = wall


def _tile_repeat(base: np.ndarray, repeats_h: int, repeats_w: int) -> np.ndarray:
    h, w = base.shape
    return np.tile(base, (int(np.ceil(repeats_h / h)), int(np.ceil(repeats_w / w))))[:repeats_h, :repeats_w]


def _pick_distinct_colors(rng: np.random.Generator, k: int) -> List[int]:
    palette = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
    k = min(k, len(palette))
    order = rng.permutation(len(palette))
    return [palette[i] for i in order[:k]]


def _is_free(occ: np.ndarray, y0: int, y1: int, x0: int, x1: int, margin: int = 1) -> bool:
    h, w = occ.shape
    ya = max(0, y0 - margin)
    yb = min(h - 1, y1 + margin)
    xa = max(0, x0 - margin)
    xb = min(w - 1, x1 + margin)
    return not np.any(occ[ya : yb + 1, xa : xb + 1])


def _randint_or_none(rng: np.random.Generator, lo: int, hi: int) -> Optional[int]:
    if hi <= lo:
        return None
    return int(rng.integers(lo, hi))


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    h = int(rng.integers(18, 30))
    w = int(rng.integers(18, 30))
    bg = 0
    wall = 7
    grid = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    side = rng.choice(np.array(["top", "right", "bottom", "left"]))
    desired_templates = int(rng.integers(2, min(5, 8)))
    colors = _pick_distinct_colors(rng, desired_templates)

    if side in ("top", "bottom"):
        x = 1
        base_y = 0 if side == "top" else h - 1
    else:
        y = 1
        base_x = 0 if side == "left" else w - 1

    used_templates = 0
    for i in range(desired_templates):
        base = int(rng.choice(np.array([2, 3])))
        repeats = int(rng.integers(1, 3))
        inner_h = base * repeats
        inner_w = base * repeats
        fh = inner_h + 2
        fw = inner_w + 2

        if side in ("top", "bottom"):
            if x + fw + 1 >= w:
                break
            y0 = base_y
            if side == "bottom":
                y0 = base_y - fh + 1
            y1 = y0 + fh - 1
            x0 = x
            x1 = x0 + fw - 1
            if y0 < 0 or y1 >= h:
                continue
            x = x1 + 2
        else:
            if y + fh + 1 >= h:
                break
            x0 = base_x
            if side == "right":
                x0 = base_x - fw + 1
            x1 = x0 + fw - 1
            y0 = y
            y1 = y0 + fh - 1
            if x0 < 0 or x1 >= w:
                continue
            y = y1 + 2

        if x0 < 0 or x0 >= w or x1 < x0 or x1 >= w or y0 < 0 or y1 >= h or y0 >= y1:
            continue

        _draw_rect_frame(grid, y0, y1, x0, x1, wall)

        # Build a repeated texture; the top-left interior pixel is the label.
        tag = colors[i]
        base_tile = rng.choice(np.array([c for c in colors if c != tag]), size=(base, base)).tolist()
        base_tile = np.array(base_tile, dtype=int)
        base_tile[0, 0] = tag
        tile = _tile_repeat(base_tile, inner_h, inner_w)
        grid[y0 + 1 : y1, x0 + 1 : x1] = tile

        occ[y0 : y1 + 1, x0 : x1 + 1] = True
        used_templates += 1

    if used_templates == 0:
        return grid

    colors = colors[:used_templates]
    num_regions = int(rng.integers(1, used_templates + 1))
    region_colors = colors[:num_regions]

    for idx, seed_color in enumerate(region_colors):
        inner_h = int(rng.integers(3, 8))
        inner_w = int(rng.integers(3, 8))
        fh = inner_h + 2
        fw = inner_w + 2

        if side == "top":
            y_lo = h // 2
            y_hi = h - fh
            x_lo = 1
            x_hi = w - fw
        elif side == "bottom":
            y_lo = 1
            y_hi = h // 2 - fh
            x_lo = 1
            x_hi = w - fw
        elif side == "left":
            y_lo = 1
            y_hi = h - fh
            x_lo = w // 2
            x_hi = w - fw
        else:
            y_lo = 1
            y_hi = h - fh
            x_lo = 1
            x_hi = w // 2 - fw

        found = False
        for _ in range(300):
            y0 = _randint_or_none(rng, y_lo, y_hi)
            x0 = _randint_or_none(rng, x_lo, x_hi)
            if y0 is None or x0 is None:
                break
            y1 = y0 + fh - 1
            x1 = x0 + fw - 1
            if y1 >= h - 1 or x1 >= w - 1:
                continue
            if not _is_free(occ, y0, y1, x0, x1, margin=1):
                continue
            _draw_rect_frame(grid, y0, y1, x0, x1, wall)
            occ[y0 : y1 + 1, x0 : x1 + 1] = True
            sy = int(rng.integers(y0 + 1, y1))
            sx = int(rng.integers(x0 + 1, x1))
            grid[sy, sx] = seed_color
            found = True
            break

        if not found:
            continue

    return grid
