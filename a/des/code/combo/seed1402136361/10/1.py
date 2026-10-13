import random
from collections import Counter

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _border_coords(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    frames: list[dict] = []
    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        for pts in _components(g == color):
            r0, c0, r1, c1 = _bbox(pts)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 4 or w < 4:
                continue
            if len(pts) != 2 * (h + w) - 4:
                continue
            rel = pts - np.array([[r0, c0]])
            comp_mask = np.zeros((h, w), dtype=bool)
            comp_mask[rel[:, 0], rel[:, 1]] = True
            if not np.array_equal(comp_mask, _border_coords(h, w)):
                continue
            inner = g[r0 + 1 : r1, c0 + 1 : c1]
            if not np.any(inner == bg):
                continue
            seed_locs = np.argwhere(inner != bg)
            if len(seed_locs) != 1:
                continue
            sr, sc = map(int, seed_locs[0])
            frames.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "frame_color": color,
                    "seed_color": int(inner[sr, sc]),
                }
            )
    frames.sort(key=lambda item: item["bbox"])
    return frames


def _find_tiles(grid: np.ndarray, bg: int) -> dict[int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    tiles: list[tuple[int, tuple[int, int, int, int], np.ndarray]] = []
    for pts in _components(g != bg):
        r0, c0, r1, c1 = _bbox(pts)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if not (2 <= h <= 4 and 2 <= w <= 4):
            continue
        if len(pts) != h * w:
            continue
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(patch == bg):
            continue
        border = np.concatenate((patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]))
        if len(set(map(int, border))) != 1:
            continue
        tiles.append((int(border[0]), (r0, c0, r1, c1), patch.copy()))

    by_color: dict[int, tuple[int, tuple[int, int, int, int], np.ndarray]] = {}
    for color, bbox, patch in tiles:
        area = int(patch.shape[0] * patch.shape[1])
        current = by_color.get(color)
        if current is None:
            by_color[color] = (area, bbox, patch)
            continue
        cur_area, cur_bbox, _ = current
        if area > cur_area or (area == cur_area and bbox < cur_bbox):
            by_color[color] = (area, bbox, patch)
    return {color: patch for color, (_, _, patch) in by_color.items()}


def _tile_region(tile: np.ndarray, h: int, w: int) -> np.ndarray:
    rows = np.arange(h)[:, None] % tile.shape[0]
    cols = np.arange(w)[None, :] % tile.shape[1]
    return tile[rows, cols]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each frame interior by looking up the outside tile whose border color matches the interior seed pixel.

    Frames are one-pixel-thick hollow rectangles; each contains exactly one colored seed pixel.
    Outside the frames, small filled rectangular tiles have a uniform border color. That border color is the lookup
    key: the matching tile is periodically repeated from the frame interior's top-left corner to fill the whole
    interior. The seed is overwritten, the frame stays, and the outside tiles disappear.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frames = _find_frames(g, bg)
    tiles = _find_tiles(g, bg)

    out = np.full(g.shape, bg, dtype=int)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        color = frame["frame_color"]
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color

        seed_color = frame["seed_color"]
        assert seed_color in tiles, f"missing tile for seed color {seed_color}"
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        out[r0 + 1 : r1, c0 + 1 : c1] = _tile_region(tiles[seed_color], inner_h, inner_w)
    return out


PALETTE = tuple(range(10))
GEN_ATTEMPTS = 400
GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
FRAME_COUNT_RANGE = (1, 3)
FRAME_H_RANGE = (6, 11)
FRAME_W_RANGE = (6, 12)
TILE_H_RANGE = (3, 4)
TILE_W_RANGE = (3, 4)
PADDING = 1


def _make_frame_patch(outer_h: int, outer_w: int, bg: int, frame_color: int, seed_color: int, rng: random.Random) -> np.ndarray:
    patch = np.full((outer_h, outer_w), bg, dtype=int)
    patch[0, :] = frame_color
    patch[-1, :] = frame_color
    patch[:, 0] = frame_color
    patch[:, -1] = frame_color
    sr = rng.randrange(1, outer_h - 1)
    sc = rng.randrange(1, outer_w - 1)
    patch[sr, sc] = seed_color
    return patch


def _assign_with_coverage(cells: list[tuple[int, int]], colors: list[int], rng: random.Random) -> list[int]:
    order = colors[:]
    rng.shuffle(order)
    values = order[: len(cells)]
    while len(values) < len(cells):
        values.append(rng.choice(order))
    rng.shuffle(values)
    return values


def _make_tile_patch(h: int, w: int, border_color: int, fill_colors: list[int], rng: random.Random) -> np.ndarray:
    patch = np.full((h, w), border_color, dtype=int)
    inner_cells = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    assert inner_cells
    n_fill_colors = rng.randint(1, min(len(fill_colors), len(inner_cells), 3))
    chosen = fill_colors[:]
    rng.shuffle(chosen)
    chosen = chosen[:n_fill_colors]
    vals = _assign_with_coverage(inner_cells, chosen, rng)
    for (r, c), value in zip(inner_cells, vals):
        patch[r, c] = value
    assert len(set(map(int, np.unique(patch)))) >= 2
    return patch


def _can_place(occupied: np.ndarray, patch_shape: tuple[int, int], top: int, left: int, pad: int) -> bool:
    h, w = patch_shape
    H, W = occupied.shape
    if top < 0 or left < 0 or top + h > H or left + w > W:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(H, top + h + pad)
    c1 = min(W, left + w + pad)
    return not occupied[r0:r1, c0:c1].any()


def _place_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, top: int, left: int, pad: int) -> None:
    h, w = patch.shape
    grid[top : top + h, left : left + w] = patch
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(grid.shape[0], top + h + pad)
    c1 = min(grid.shape[1], left + w + pad)
    occupied[r0:r1, c0:c1] = True


def _expected_output(shape: tuple[int, int], bg: int, frame_specs: list[dict], placements: list[dict]) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    tile_by_seed = {item["seed_color"]: item["tile"] for item in placements}
    for spec in frame_specs:
        top = spec["top"]
        left = spec["left"]
        patch = spec["patch"]
        h, w = patch.shape
        frame_color = spec["frame_color"]
        out[top, left : left + w] = frame_color
        out[top + h - 1, left : left + w] = frame_color
        out[top : top + h, left] = frame_color
        out[top : top + h, left + w - 1] = frame_color
        tile = tile_by_seed[spec["seed_color"]]
        out[top + 1 : top + h - 1, left + 1 : left + w - 1] = _tile_region(tile, h - 2, w - 2)
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    H = rng.randint(*GRID_H_RANGE)
    W = rng.randint(*GRID_W_RANGE)
    bg = rng.choice(PALETTE)
    count = rng.randint(*FRAME_COUNT_RANGE)

    usable = [c for c in PALETTE if c != bg]
    rng.shuffle(usable)
    if len(usable) < 2 * count + 1:
        return None
    seed_colors = usable[:count]
    frame_colors = usable[count : 2 * count]
    interior_palette = usable[2 * count :]
    if not interior_palette:
        return None

    frame_specs: list[dict] = []
    tile_specs: list[dict] = []
    for seed_color, frame_color in zip(seed_colors, frame_colors):
        outer_h = rng.randint(*FRAME_H_RANGE)
        outer_w = rng.randint(*FRAME_W_RANGE)
        frame_patch = _make_frame_patch(outer_h, outer_w, bg, frame_color, seed_color, rng)
        frame_specs.append(
            {
                "patch": frame_patch,
                "frame_color": frame_color,
                "seed_color": seed_color,
            }
        )

        tile_h = rng.randint(*TILE_H_RANGE)
        tile_w = rng.randint(*TILE_W_RANGE)
        tile_patch = _make_tile_patch(tile_h, tile_w, seed_color, interior_palette, rng)
        tile_specs.append({"tile": tile_patch, "seed_color": seed_color})

    grid = np.full((H, W), bg, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    objects: list[tuple[str, int, np.ndarray, dict]] = []
    for spec in frame_specs:
        patch = spec["patch"]
        objects.append(("frame", int(patch.shape[0] * patch.shape[1]), patch, spec))
    for spec in tile_specs:
        patch = spec["tile"]
        objects.append(("tile", int(patch.shape[0] * patch.shape[1]), patch, spec))
    rng.shuffle(objects)
    objects.sort(key=lambda item: -item[1])

    placed_frames: list[dict] = []
    placed_tiles: list[dict] = []
    for kind, _area, patch, spec in objects:
        h, w = patch.shape
        spots = []
        for top in range(1, H - h):
            for left in range(1, W - w):
                if _can_place(occupied, (h, w), top, left, PADDING):
                    spots.append((top, left))
        if not spots:
            return None
        top, left = rng.choice(spots)
        _place_patch(grid, occupied, patch, top, left, PADDING)
        rec = dict(spec)
        rec["top"] = top
        rec["left"] = left
        if kind == "frame":
            placed_frames.append(rec)
        else:
            placed_tiles.append(rec)

    expected = _expected_output((H, W), bg, placed_frames, placed_tiles)
    actual = main(grid)
    if not np.array_equal(expected, actual):
        return None
    if len(_find_frames(grid, bg)) != count:
        return None
    if set(_find_tiles(grid, bg)) != set(seed_colors):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
