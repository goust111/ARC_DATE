from __future__ import annotations

import random
from collections import deque

import numpy as np


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_solid_rectangles(grid: np.ndarray) -> list[dict]:
    rects: list[dict] = []
    for color in sorted(map(int, np.unique(grid))):
        for coords in _components(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            area = int(coords.shape[0])
            if area != height * width:
                continue
            if area < 2:
                continue
            rects.append({"color": color, "bbox": (r0, c0, r1, c1), "area": area})
    rects.sort(key=lambda item: (item["bbox"], item["color"]))
    return rects


def _infer_tile(grid: np.ndarray, occluded: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    visible = ~occluded
    best_key: tuple[int, int, int] | None = None
    best_tile: np.ndarray | None = None

    for ph in range(1, h + 1):
        for pw in range(1, w + 1):
            tile = np.full((ph, pw), -1, dtype=int)
            seen = np.zeros((ph, pw), dtype=bool)
            ok = True
            for r in range(h):
                rr = r % ph
                for c in range(w):
                    if not visible[r, c]:
                        continue
                    cc = c % pw
                    value = int(grid[r, c])
                    if seen[rr, cc]:
                        if int(tile[rr, cc]) != value:
                            ok = False
                            break
                    else:
                        tile[rr, cc] = value
                        seen[rr, cc] = True
                if not ok:
                    break
            if not ok or not bool(seen.all()):
                continue

            key = (ph * pw, ph, pw)
            if best_key is None or key < best_key:
                best_key = key
                best_tile = tile.copy()

    if best_tile is None:
        raise ValueError("could not infer a complete periodic tile from visible pixels")
    return best_tile


def _render_wallpaper(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    th, tw = tile.shape
    reps = ((h + th - 1) // th, (w + tw - 1) // tw)
    return np.tile(tile, reps)[:h, :w].copy()


def _draw_border(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a periodic wallpaper hidden by solid rectangles, then mark each
    original rectangle location with a hollow border in the rectangle's color.
    """
    g = np.asarray(grid, dtype=int)
    rects = _find_solid_rectangles(g)

    occluded = np.zeros_like(g, dtype=bool)
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        occluded[r0 : r1 + 1, c0 : c1 + 1] = True

    tile = _infer_tile(g, occluded)
    out = _render_wallpaper(tile, g.shape)
    for rect in rects:
        _draw_border(out, rect["bbox"], int(rect["color"]))
    return out


PALETTE = tuple(range(10))
GEN_RETRIES = 400
TILE_RETRIES = 120
RECT_RETRIES = 120


def _has_smaller_period(tile: np.ndarray) -> bool:
    h, w = tile.shape
    for ph in range(1, h + 1):
        for pw in range(1, w + 1):
            if ph == h and pw == w:
                continue
            ok = True
            for r in range(h):
                for c in range(w):
                    if int(tile[r, c]) != int(tile[r % ph, c % pw]):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return True
    return False


def _torus_adjacent_equal(tile: np.ndarray) -> bool:
    h, w = tile.shape
    for r in range(h):
        for c in range(w):
            if int(tile[r, c]) == int(tile[(r + 1) % h, c]):
                return True
            if int(tile[r, c]) == int(tile[r, (c + 1) % w]):
                return True
    return False


def _make_primitive_tile(height: int, width: int, colors: list[int], rng: random.Random) -> np.ndarray | None:
    tile = np.full((height, width), -1, dtype=int)
    counts = {color: 0 for color in colors}

    def backtrack(idx: int) -> bool:
        if idx == height * width:
            return all(counts[color] > 0 for color in colors) and not _has_smaller_period(tile)

        r = idx // width
        c = idx % width
        choices = colors[:]
        rng.shuffle(choices)
        for color in choices:
            if r > 0 and int(tile[r - 1, c]) == color:
                continue
            if c > 0 and int(tile[r, c - 1]) == color:
                continue
            if r == height - 1 and int(tile[0, c]) == color:
                continue
            if c == width - 1 and int(tile[r, 0]) == color:
                continue

            tile[r, c] = color
            counts[color] += 1
            remaining = height * width - idx - 1
            missing = sum(1 for value in counts.values() if value == 0)
            if missing <= remaining and backtrack(idx + 1):
                return True
            counts[color] -= 1
            tile[r, c] = -1
        return False

    for _ in range(TILE_RETRIES):
        tile.fill(-1)
        for color in colors:
            counts[color] = 0
        if backtrack(0) and not _torus_adjacent_equal(tile):
            return tile.copy()
    return None


def _visible_every_residue(visible: np.ndarray, th: int, tw: int) -> bool:
    for rr in range(th):
        for cc in range(tw):
            if not bool(visible[rr::th, cc::tw].any()):
                return False
    return True


def _place_rectangles(
    height: int,
    width: int,
    rect_colors: list[int],
    rng: random.Random,
) -> list[dict] | None:
    blocked = np.zeros((height, width), dtype=bool)
    rects: list[dict] = []
    max_rect_h = max(3, min(height - 1, height // 2 + 1))
    max_rect_w = max(3, min(width - 1, width // 2 + 1))

    for color in rect_colors:
        placed = False
        for _ in range(RECT_RETRIES):
            rect_h = rng.randint(3, max_rect_h)
            rect_w = rng.randint(3, max_rect_w)
            if rect_h > height or rect_w > width:
                continue
            r0 = rng.randint(0, height - rect_h)
            c0 = rng.randint(0, width - rect_w)
            r1 = r0 + rect_h - 1
            c1 = c0 + rect_w - 1
            pr0 = max(0, r0 - 1)
            pc0 = max(0, c0 - 1)
            pr1 = min(height - 1, r1 + 1)
            pc1 = min(width - 1, c1 + 1)
            if bool(blocked[pr0 : pr1 + 1, pc0 : pc1 + 1].any()):
                continue
            blocked[pr0 : pr1 + 1, pc0 : pc1 + 1] = True
            rects.append({"color": color, "bbox": (r0, c0, r1, c1)})
            placed = True
            break
        if not placed:
            return None
    rects.sort(key=lambda item: (item["bbox"], item["color"]))
    return rects


def _apply_filled_rectangles(base: np.ndarray, rects: list[dict]) -> np.ndarray:
    grid = base.copy()
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(rect["color"])
    return grid


def _expected_output(base: np.ndarray, rects: list[dict]) -> np.ndarray:
    out = base.copy()
    for rect in rects:
        _draw_border(out, rect["bbox"], int(rect["color"]))
    return out


def _generate_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    num_rects = rng.randint(1, 3)
    tile_h = rng.randint(2, 5)
    tile_w = rng.randint(2, 5)
    max_tile_colors = min(5, len(PALETTE) - num_rects, tile_h * tile_w)
    tile_color_count = rng.randint(3, max_tile_colors)
    all_colors = list(PALETTE)
    rng.shuffle(all_colors)
    tile_colors = all_colors[:tile_color_count]
    rect_colors = all_colors[tile_color_count : tile_color_count + num_rects]

    tile = _make_primitive_tile(tile_h, tile_w, tile_colors, rng)
    if tile is None:
        raise RuntimeError("failed to create primitive tile")

    height = rng.randint(max(12, tile_h * 3), 24)
    width = rng.randint(max(12, tile_w * 3), 24)
    wallpaper = _render_wallpaper(tile, (height, width))

    rects = _place_rectangles(height, width, rect_colors, rng)
    if rects is None:
        raise RuntimeError("failed to place rectangles")

    visible = np.ones((height, width), dtype=bool)
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        visible[r0 : r1 + 1, c0 : c1 + 1] = False
    if not _visible_every_residue(visible, tile_h, tile_w):
        raise RuntimeError("some tile residues are fully occluded")

    inp = _apply_filled_rectangles(wallpaper, rects)
    expected = _expected_output(wallpaper, rects)

    recovered_rects = _find_solid_rectangles(inp)
    recovered_signature = [(item["bbox"], item["color"]) for item in recovered_rects]
    expected_signature = [(item["bbox"], item["color"]) for item in rects]
    if recovered_signature != expected_signature:
        raise RuntimeError("rectangle detection became ambiguous")

    recovered_tile = _infer_tile(inp, ~visible)
    if recovered_tile.shape != tile.shape or not np.array_equal(recovered_tile, tile):
        raise RuntimeError("period inference does not recover the planted tile")

    if not np.array_equal(main(inp), expected):
        raise RuntimeError("main() does not match generator truth")

    return inp, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            inp, _ = _generate_case(rng)
        except Exception:
            continue
        return inp
    raise RuntimeError(f"failed to generate a valid sample after {GEN_RETRIES} attempts")
