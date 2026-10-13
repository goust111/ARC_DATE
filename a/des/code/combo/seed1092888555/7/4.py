import random
from typing import Iterable

import numpy as np


BG = 8
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
CORNERS = ("tl", "tr", "bl", "br")


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [p[0] for p in pts]
            cols = [p[1] for p in pts]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in pts:
                mask[rr - r0, cc - c0] = True
            objects.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "center": ((r0 + r1) / 2.0, (c0 + c1) / 2.0),
                }
            )
    return objects


def _nearest_corner(center: tuple[float, float], shape: tuple[int, int]) -> str:
    h, w = shape
    r, c = center
    corner_points = {
        "tl": (0.0, 0.0),
        "tr": (0.0, float(w - 1)),
        "bl": (float(h - 1), 0.0),
        "br": (float(h - 1), float(w - 1)),
    }
    return min(
        CORNERS,
        key=lambda name: (r - corner_points[name][0]) ** 2 + (c - corner_points[name][1]) ** 2,
    )


def _flip_mask(mask: np.ndarray, flip_r: bool, flip_c: bool) -> np.ndarray:
    out = mask
    if flip_r:
        out = out[::-1, :]
    if flip_c:
        out = out[:, ::-1]
    return out


def _orientation_score(mask: np.ndarray, corner: str) -> tuple[float, ...]:
    pts = np.argwhere(mask)
    if corner == "tl":
        primary = -pts[:, 0].mean() - pts[:, 1].mean()
    elif corner == "tr":
        primary = -pts[:, 0].mean() + pts[:, 1].mean()
    elif corner == "bl":
        primary = pts[:, 0].mean() - pts[:, 1].mean()
    else:
        primary = pts[:, 0].mean() + pts[:, 1].mean()

    h, w = mask.shape
    rr, cc = np.indices((h, w))
    if corner == "tl":
        outward_weight = (h - rr) + (w - cc)
    elif corner == "tr":
        outward_weight = (h - rr) + (cc + 1)
    elif corner == "bl":
        outward_weight = (rr + 1) + (w - cc)
    else:
        outward_weight = (rr + 1) + (cc + 1)
    weighted = float(outward_weight[mask].sum())
    return (float(primary), weighted)


def _orient_toward_corner(mask: np.ndarray, corner: str) -> np.ndarray:
    best = None
    for flip_r in (False, True):
        for flip_c in (False, True):
            candidate = _flip_mask(mask, flip_r, flip_c)
            score = _orientation_score(candidate, corner)
            if best is None or score > best[0]:
                best = (score, candidate.copy())
    assert best is not None
    return best[1]


def _place_object(out: np.ndarray, mask: np.ndarray, color: int, corner: str) -> None:
    h, w = out.shape
    oh, ow = mask.shape
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, w - ow
    elif corner == "bl":
        r0, c0 = h - oh, 0
    else:
        r0, c0 = h - oh, w - ow
    patch = out[r0 : r0 + oh, c0 : c0 + ow]
    patch[mask] = color


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(np.bincount(g.ravel()).argmax())
    out = np.full_like(g, bg)
    for obj in _components(g, bg):
        corner = _nearest_corner(obj["center"], g.shape)
        mask = _orient_toward_corner(obj["mask"], corner)
        _place_object(out, mask, int(obj["color"]), corner)
    return out


SHAPES = [
    np.array(
        [
            [1, 1, 1],
            [1, 0, 0],
            [1, 0, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1],
            [0, 0, 1],
            [0, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1, 0],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 1, 1, 1],
            [0, 0, 0, 1],
            [0, 1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [0, 1, 0, 0],
        ],
        dtype=bool,
    ),
]


def _random_shape() -> np.ndarray:
    shape = random.choice(SHAPES).copy()
    k = random.randrange(4)
    shape = np.rot90(shape, k)
    if random.random() < 0.5:
        shape = shape[::-1, :]
    if random.random() < 0.5:
        shape = shape[:, ::-1]
    return shape.copy()


def _corner_anchor_ranges(corner: str, gh: int, gw: int, oh: int, ow: int) -> tuple[range, range]:
    margin = 2
    center_band_h = gh // 2 - 1
    center_band_w = gw // 2 - 1
    if corner == "tl":
        rs = range(margin, max(margin + 1, center_band_h - oh + 1))
        cs = range(margin, max(margin + 1, center_band_w - ow + 1))
    elif corner == "tr":
        rs = range(margin, max(margin + 1, center_band_h - oh + 1))
        cs = range(max(gw // 2 + 1, margin), gw - ow - margin + 1)
    elif corner == "bl":
        rs = range(max(gh // 2 + 1, margin), gh - oh - margin + 1)
        cs = range(margin, max(margin + 1, center_band_w - ow + 1))
    else:
        rs = range(max(gh // 2 + 1, margin), gh - oh - margin + 1)
        cs = range(max(gw // 2 + 1, margin), gw - ow - margin + 1)
    return rs, cs


def _fits(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> bool:
    oh, ow = mask.shape
    area = grid[r0 : r0 + oh, c0 : c0 + ow]
    if area.shape != mask.shape:
        return False
    if np.any(area[mask] != BG):
        return False
    for rr, cc in np.argwhere(mask):
        r, c = r0 + int(rr), c0 + int(cc)
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and int(grid[nr, nc]) != BG and not (r0 <= nr < r0 + oh and c0 <= nc < c0 + ow and bool(mask[nr - r0, nc - c0])):
                return False
    return True


def _paint(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    oh, ow = mask.shape
    patch = grid[r0 : r0 + oh, c0 : c0 + ow]
    patch[mask] = color


def _output_corners_do_not_overlap(items: Iterable[tuple[str, np.ndarray]], gh: int, gw: int) -> bool:
    occ = np.zeros((gh, gw), dtype=int)
    for corner, mask in items:
        test = np.zeros((gh, gw), dtype=bool)
        _place_object(test, mask, True, corner)
        if np.any(occ & test):
            return False
        occ |= test
    return True


def _gen_attempt() -> np.ndarray | None:
    gh = random.randint(14, 20)
    gw = random.randint(14, 20)
    grid = np.full((gh, gw), BG, dtype=int)

    corner_count = random.randint(2, 4)
    chosen_corners = random.sample(list(CORNERS), corner_count)
    shapes = []
    for corner in chosen_corners:
        shape = _random_shape()
        shapes.append((corner, shape))
    if not _output_corners_do_not_overlap(shapes, gh, gw):
        return None

    colors = random.sample([1, 2, 3, 4, 5, 6, 7, 9], corner_count)
    for color, (corner, mask) in zip(colors, shapes):
        oh, ow = mask.shape
        rs, cs = _corner_anchor_ranges(corner, gh, gw, oh, ow)
        if len(rs) == 0 or len(cs) == 0:
            return None
        positions = [(r0, c0) for r0 in rs for c0 in cs]
        random.shuffle(positions)
        placed = False
        for r0, c0 in positions:
            if _fits(grid, mask, r0, c0):
                _paint(grid, mask, r0, c0, color)
                placed = True
                break
        if not placed:
            return None

    counts = [int((grid == color).sum()) for color in colors]
    if len(set(counts)) == 1 and corner_count == 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after 300 attempts")
