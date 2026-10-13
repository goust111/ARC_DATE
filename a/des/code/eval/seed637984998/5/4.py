from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8], dtype=int)
MAX_RETRIES = 500
_RNG = np.random.default_rng()


DIGIT_TEMPLATES: dict[int, np.ndarray] = {
    1: np.array(
        [
            [0, 1, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    2: np.array(
        [
            [1, 1, 1],
            [0, 0, 1],
            [1, 1, 1],
            [1, 0, 0],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    4: np.array(
        [
            [1, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 0, 1],
            [0, 0, 1],
        ],
        dtype=bool,
    ),
}


BASE_PATTERNS: dict[int, list[tuple[tuple[int, int], ...]]] = {
    1: [
        ((-2, -1), (-1, -1), (0, -1), (0, 0), (0, 1), (1, 1)),
        ((-1, -2), (-1, -1), (-1, 0), (0, 0), (1, 0), (1, 1)),
        ((-2, -1), (-1, -1), (-1, 0), (0, 0), (0, 1), (1, 1), (2, 1)),
    ],
    2: [
        ((0, -2), (0, -1), (0, 0), (0, 1), (0, 2), (-1, 1), (1, -1)),
        ((-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0), (-1, -1), (1, 1)),
        ((-2, -1), (-1, -1), (0, -1), (0, 0), (0, 1), (1, 1), (2, 1)),
    ],
    4: [
        ((-2, 0), (-1, 0), (0, -2), (0, -1), (0, 0), (0, 1), (0, 2), (1, 0), (2, 0)),
        ((-2, -2), (-2, 2), (-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1), (2, -2), (2, 2)),
        ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)),
    ],
}


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _rotation_order(mask: np.ndarray) -> int:
    if np.array_equal(mask, np.rot90(mask, 1)):
        return 4
    if np.array_equal(mask, np.rot90(mask, 2)):
        return 2
    return 1


def _infer_texture(sample: np.ndarray) -> np.ndarray:
    h, w = sample.shape
    unknown = sample == MAROON
    sizes = sorted((th * tw, th, tw) for th in range(1, h + 1) for tw in range(1, w + 1))
    for _area, th, tw in sizes:
        tile = np.full((th, tw), -1, dtype=int)
        covered = np.zeros((th, tw), dtype=bool)
        ok = True
        for r in range(h):
            for c in range(w):
                if unknown[r, c]:
                    continue
                rr = r % th
                cc = c % tw
                val = int(sample[r, c])
                if tile[rr, cc] == -1:
                    tile[rr, cc] = val
                elif tile[rr, cc] != val:
                    ok = False
                    break
                covered[rr, cc] = True
            if not ok:
                break
        if ok and covered.all():
            return tile
    raise ValueError("unable to infer texture period")


def _extract_samples(grid: np.ndarray) -> dict[int, np.ndarray]:
    h, w = grid.shape
    samples: dict[int, np.ndarray] = {}
    for comp in _components(grid != BLACK):
        r0, c0, r1, c1 = _bbox(comp)
        if not (r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1):
            continue
        if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 3:
            continue
        if GREY in grid[comp[:, 0], comp[:, 1]]:
            continue
        area = int(comp.shape[0])
        if area != (r1 - r0 + 1) * (c1 - c0 + 1):
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        label_mask = sub[:5, :3] == MAROON
        label = None
        for value, template in DIGIT_TEMPLATES.items():
            if np.array_equal(label_mask, template):
                label = value
                break
        if label is None:
            continue
        samples[label] = _infer_texture(sub)
    if set(samples) != {1, 2, 4}:
        raise ValueError("expected exactly three labeled sample regions")
    return samples


def _extract_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for comp in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h != w or h < 5:
            continue
        if int(comp.shape[0]) != 2 * h + 2 * w - 4:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not np.all(sub[border] == GREY):
            continue
        if np.any(sub[~border] == GREY):
            continue
        frames.append((r0, c0, r1, c1))
    if not frames:
        raise ValueError("no valid grey square frames found")
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Choose a texture from the labeled library by the pattern's rotational symmetry order.

    Each grey square frame encloses a monochrome pattern. Determine whether the
    inner pattern has exact rotational order 1, 2, or 4, then tile the matching
    sample texture into the one-cell ring just outside that frame.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    textures = _extract_samples(g)
    for r0, c0, r1, c1 in _extract_frames(g):
        inner = g[r0 + 1 : r1, c0 + 1 : c1] != BLACK
        order = _rotation_order(inner)
        tile = textures[order]
        th, tw = tile.shape
        rr0 = r0 - 1
        cc0 = c0 - 1
        for rr in range(rr0, r1 + 2):
            for cc in range(cc0, c1 + 2):
                if r0 <= rr <= r1 and c0 <= cc <= c1:
                    continue
                out[rr, cc] = tile[(rr - rr0) % th, (cc - cc0) % tw]
    return out


def _rotate_coords(coords: tuple[tuple[int, int], ...], turns: int) -> tuple[tuple[int, int], ...]:
    turns %= 4
    out = list(coords)
    for _ in range(turns):
        out = [(-c, r) for r, c in out]
    return tuple(out)


def _coords_to_mask(coords: tuple[tuple[int, int], ...], size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mid = size // 2
    for dr, dc in coords:
        mask[mid + dr, mid + dc] = True
    return mask


def _make_texture(rng: np.random.Generator) -> np.ndarray:
    families = ["checker", "diag3", "vstripe", "hstripe", "cross", "zigzag"]
    while True:
        colors = [int(x) for x in rng.choice(PALETTE, size=3, replace=False)]
        a, b, c = colors
        family = str(rng.choice(families))
        if family == "checker":
            tile = np.array([[a, b], [b, a]], dtype=int)
        elif family == "diag3":
            tile = np.array([[a, b, c], [c, a, b], [b, c, a]], dtype=int)
        elif family == "vstripe":
            tile = np.array([[a, b, c]], dtype=int)
        elif family == "hstripe":
            tile = np.array([[a], [b], [c]], dtype=int)
        elif family == "cross":
            tile = np.array([[b, a, b], [a, a, a], [b, a, b]], dtype=int)
        else:
            tile = np.array([[a, b, c], [c, a, b]], dtype=int)
        tile = _infer_texture(tile)
        if tile.size > 1 and len(set(int(x) for x in tile.flat)) >= 2:
            return tile


def _tile_signature(tile: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return tile.shape, tuple(int(x) for x in tile.flat)


def _repeat_tile(tile: np.ndarray, h: int, w: int) -> np.ndarray:
    th, tw = tile.shape
    reps_r = (h + th - 1) // th
    reps_c = (w + tw - 1) // tw
    return np.tile(tile, (reps_r, reps_c))[:h, :w]


def _sample_rect(label: int, tile: np.ndarray, h: int, w: int) -> np.ndarray:
    rect = _repeat_tile(tile, h, w)
    rect[:5, :3][DIGIT_TEMPLATES[label]] = MAROON
    return rect


def _mark_rect(mask: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 0) -> None:
    h, w = mask.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _frame_pattern(order: int, interior_size: int, rng: np.random.Generator) -> np.ndarray:
    for _ in range(40):
        base = BASE_PATTERNS[order][int(rng.integers(0, len(BASE_PATTERNS[order])))]
        turns = 0 if order == 4 else int(rng.integers(0, 4))
        mask = _coords_to_mask(_rotate_coords(base, turns), interior_size)
        if _rotation_order(mask) == order:
            return mask
    raise RuntimeError(f"failed to build order-{order} pattern")


def _frame_specs(rng: np.random.Generator) -> list[dict[str, object]]:
    count = int(rng.integers(3, 6))
    orders = [1, 2, 4]
    while len(orders) < count:
        orders.append(int(rng.choice([1, 2, 4])))
    rng.shuffle(orders)
    specs: list[dict[str, object]] = []
    for order in orders:
        size = int(rng.choice([7, 9]))
        mask = _frame_pattern(order, size - 2, rng)
        color = int(rng.choice(PALETTE))
        specs.append({"order": order, "size": size, "mask": mask, "color": color})
    specs.sort(key=lambda d: int(d["size"]), reverse=True)
    return specs


def _place_frames(
    grid: np.ndarray,
    forbidden: np.ndarray,
    specs: list[dict[str, object]],
    sample_edge: str,
    sample_h: int,
    rng: np.random.Generator,
) -> bool:
    h, w = grid.shape
    for spec in specs:
        size = int(spec["size"])
        outer = size + 2
        if sample_edge == "top":
            min_r = sample_h + 1
            max_r = h - outer - 1
        else:
            min_r = 1
            max_r = h - sample_h - outer - 1
        min_c = 1
        max_c = w - outer - 1
        if min_r > max_r or min_c > max_c:
            return False
        placed = False
        for _ in range(250):
            or0 = int(rng.integers(min_r, max_r + 1))
            oc0 = int(rng.integers(min_c, max_c + 1))
            outer_bbox = (or0, oc0, or0 + outer - 1, oc0 + outer - 1)
            r0 = or0 + 1
            c0 = oc0 + 1
            r1 = r0 + size - 1
            c1 = c0 + size - 1
            if forbidden[outer_bbox[0] : outer_bbox[2] + 1, outer_bbox[1] : outer_bbox[3] + 1].any():
                continue
            grid[r0, c0 : c1 + 1] = GREY
            grid[r1, c0 : c1 + 1] = GREY
            grid[r0 : r1 + 1, c0] = GREY
            grid[r0 : r1 + 1, c1] = GREY
            inner = spec["mask"]
            color = int(spec["color"])
            grid[r0 + 1 : r1, c0 + 1 : c1][inner] = color
            _mark_rect(forbidden, outer_bbox, pad=1)
            placed = True
            break
        if not placed:
            return False
    return True


def _validate_grid(grid: np.ndarray) -> bool:
    try:
        samples = _extract_samples(grid)
        frames = _extract_frames(grid)
        orders = {
            _rotation_order(grid[r0 + 1 : r1, c0 + 1 : c1] != BLACK)
            for r0, c0, r1, c1 in frames
        }
        if set(samples) != {1, 2, 4} or orders != {1, 2, 4}:
            return False
        out = main(grid)
        return not np.array_equal(out, grid)
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    rng = _RNG
    sample_h = int(rng.integers(8, 11))
    widths = [int(rng.integers(8, 11)) for _ in range(3)]
    gaps = [int(rng.integers(1, 4)) for _ in range(2)]
    total_sample_w = int(sum(widths) + sum(gaps))
    specs = _frame_specs(rng)
    sample_edge = str(rng.choice(["top", "bottom"]))

    min_w = max(38, total_sample_w + 2)
    max_w = max(min_w + 1, 49)
    w = int(rng.integers(min_w, max_w))
    h = int(rng.integers(30, 41))
    grid = np.full((h, w), BLACK, dtype=int)
    forbidden = np.zeros((h, w), dtype=bool)

    seen_tiles: set[tuple[tuple[int, int], tuple[int, ...]]] = set()
    textures: dict[int, np.ndarray] = {}
    for label in (1, 2, 4):
        for _ in range(60):
            tile = _make_texture(rng)
            sig = _tile_signature(tile)
            if sig not in seen_tiles:
                seen_tiles.add(sig)
                textures[label] = tile
                break
        else:
            return None

    order = [1, 2, 4]
    rng.shuffle(order)
    start_c = int(rng.integers(0, w - total_sample_w + 1))
    row0 = 0 if sample_edge == "top" else h - sample_h
    c = start_c
    for idx, label in enumerate(order):
        rect = _sample_rect(label, textures[label], sample_h, widths[idx])
        bbox = (row0, c, row0 + sample_h - 1, c + widths[idx] - 1)
        grid[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1] = rect
        _mark_rect(forbidden, bbox, pad=1)
        c += widths[idx]
        if idx < 2:
            c += gaps[idx]

    if not _place_frames(grid, forbidden, specs, sample_edge, sample_h, rng):
        return None

    if not _validate_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-texture sample")
