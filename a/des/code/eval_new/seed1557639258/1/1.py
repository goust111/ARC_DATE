from collections import deque

import numpy as np


BLACK = 0
GREY = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
_CORNERS = ("tl", "tr", "bl", "br")
_RNG = np.random.default_rng()

_GRID_SIDE_MIN = 16
_GRID_SIDE_MAX = 30
_INNER_SIDE_MIN = 4
_INNER_SIDE_MAX = 8
_MAX_ASPECT_DELTA = 3
_PLACEMENT_RETRIES = 200
_MAX_GEN_ATTEMPTS = 300


def _randint(lo: int, hi: int) -> int:
    return int(_RNG.integers(int(lo), int(hi) + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(int(r), int(c))])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _triangle_mask(height: int, width: int, corner: str) -> np.ndarray:
    rows = np.arange(int(height))[:, None]
    cols = np.arange(int(width))[None, :]
    if corner == "tl":
        return rows + cols < int(width)
    if corner == "tr":
        return cols >= rows
    if corner == "bl":
        return (int(height) - 1 - rows) + cols < int(width)
    if corner == "br":
        return cols >= (int(height) - 1 - rows)
    raise ValueError(f"unknown corner {corner}")


def _seed_corner(seed_r: int, seed_c: int, height: int, width: int) -> str:
    top = int(seed_r) * 2 < int(height)
    left = int(seed_c) * 2 < int(width)
    if top and left:
        return "tl"
    if top and not left:
        return "tr"
    if not top and left:
        return "bl"
    return "br"


def _extract_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for coords in _components(grid == GREY):
        rs = coords[:, 0]
        cs = coords[:, 1]
        r0 = int(rs.min())
        r1 = int(rs.max())
        c0 = int(cs.min())
        c1 = int(cs.max())
        if r1 - r0 < 2 or c1 - c0 < 2:
            raise AssertionError("grey component must be at least a 3x3 frame")

        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros_like(patch, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not bool(np.all(patch[border] == GREY)):
            raise AssertionError("frame border must be uniformly grey")
        if not bool(np.all(patch[~border] != GREY)):
            raise AssertionError("frame interior cannot contain grey pixels")
        if int(coords.shape[0]) != int(border.sum()):
            raise AssertionError("grey component must be exactly one rectangular border")

        inner = patch[1:-1, 1:-1]
        seeds = np.argwhere((inner != BLACK) & (inner != GREY))
        if int(seeds.shape[0]) != 1:
            raise AssertionError("each frame must contain exactly one colored seed")
        seed_r, seed_c = map(int, seeds[0])
        frames.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "inner_shape": (int(inner.shape[0]), int(inner.shape[1])),
                "seed": (int(seed_r), int(seed_c)),
                "color": int(inner[seed_r, seed_c]),
            }
        )

    if not frames:
        raise AssertionError("expected at least one grey frame")
    frames.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey rectangular container interior with a corner-anchored triangular staircase chosen by its seed quadrant.

    Input structure:
    - The background is black.
    - Several non-overlapping grey hollow rectangles appear as closed containers.
    - Each container interior has exactly one colored seed pixel and otherwise black cells.

    Rule:
    - Detect every grey rectangular frame and inspect the seed inside it.
    - Split the frame interior into top/bottom and left/right halves; the seed's half-pair identifies one of the four corners.
    - Repaint the interior with the seed color as a row-wise triangular staircase that starts from that corner, keeps the grey border unchanged, and leaves the unfilled interior cells black.

    Key insight:
    - The seed color gives the paint color, while the seed's relative quadrant encodes only the triangle's starting corner and growth direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for frame in _extract_frames(g):
        r0, c0, r1, c1 = frame["bbox"]
        inner_h, inner_w = frame["inner_shape"]
        seed_r, seed_c = frame["seed"]
        color = int(frame["color"])
        corner = _seed_corner(seed_r, seed_c, inner_h, inner_w)
        mask = _triangle_mask(inner_h, inner_w, corner)

        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        inner[:] = BLACK
        inner[mask] = color

    return out


def _quadrant_mask(height: int, width: int, corner: str) -> np.ndarray:
    rows = np.arange(int(height))[:, None]
    cols = np.arange(int(width))[None, :]
    top_half = rows < (int(height) // 2)
    bottom_half = rows >= ((int(height) + 1) // 2)
    left_half = cols < (int(width) // 2)
    right_half = cols >= ((int(width) + 1) // 2)
    if corner == "tl":
        return top_half & left_half
    if corner == "tr":
        return top_half & right_half
    if corner == "bl":
        return bottom_half & left_half
    if corner == "br":
        return bottom_half & right_half
    raise ValueError(f"unknown corner {corner}")


def _sample_inner_size() -> tuple[int, int]:
    for _ in range(40):
        height = _randint(_INNER_SIDE_MIN, _INNER_SIDE_MAX)
        width = _randint(_INNER_SIDE_MIN, _INNER_SIDE_MAX)
        if abs(int(height) - int(width)) <= _MAX_ASPECT_DELTA:
            return int(height), int(width)
    return 6, 6


def _balanced_corners(count: int) -> list[str]:
    corners: list[str] = []
    while len(corners) < int(count):
        batch = list(_CORNERS)
        _RNG.shuffle(batch)
        corners.extend(batch)
    return corners[:count]


def _gen_attempt() -> np.ndarray | None:
    height = _randint(_GRID_SIDE_MIN, _GRID_SIDE_MAX)
    width = _randint(_GRID_SIDE_MIN, _GRID_SIDE_MAX)
    area = int(height * width)
    max_count = min(5, max(1, area // 160 + 1))
    min_count = 1 if max_count == 1 else 2
    count = _randint(min_count, max_count)

    colors = list(SEED_COLORS)
    _RNG.shuffle(colors)
    corners = _balanced_corners(count)

    specs = []
    for idx in range(count):
        inner_h, inner_w = _sample_inner_size()
        specs.append(
            {
                "inner_h": int(inner_h),
                "inner_w": int(inner_w),
                "outer_h": int(inner_h + 2),
                "outer_w": int(inner_w + 2),
                "color": int(colors[idx]),
                "corner": corners[idx],
            }
        )
    specs.sort(key=lambda item: item["outer_h"] * item["outer_w"], reverse=True)

    grid = np.full((height, width), BLACK, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    for spec in specs:
        outer_h = int(spec["outer_h"])
        outer_w = int(spec["outer_w"])
        placed = False

        for _ in range(_PLACEMENT_RETRIES):
            r0 = _randint(0, height - outer_h)
            c0 = _randint(0, width - outer_w)
            r1 = int(r0 + outer_h - 1)
            c1 = int(c0 + outer_w - 1)

            br0 = max(0, int(r0 - 1))
            bc0 = max(0, int(c0 - 1))
            br1 = min(height, int(r1 + 2))
            bc1 = min(width, int(c1 + 2))
            if bool(blocked[br0:br1, bc0:bc1].any()):
                continue

            triangle = _triangle_mask(int(spec["inner_h"]), int(spec["inner_w"]), spec["corner"])
            quadrant = _quadrant_mask(int(spec["inner_h"]), int(spec["inner_w"]), spec["corner"])
            candidates = np.argwhere(triangle & quadrant)
            if int(candidates.shape[0]) == 0:
                return None
            seed_r, seed_c = map(int, candidates[int(_RNG.integers(int(candidates.shape[0])))])

            grid[r0, c0 : c1 + 1] = GREY
            grid[r1, c0 : c1 + 1] = GREY
            grid[r0 : r1 + 1, c0] = GREY
            grid[r0 : r1 + 1, c1] = GREY
            grid[r0 + 1 + seed_r, c0 + 1 + seed_c] = int(spec["color"])
            blocked[br0:br1, bc0:bc1] = True
            placed = True
            break

        if not placed:
            return None

    try:
        frames = _extract_frames(grid)
        if len(frames) != count:
            return None
        out = main(grid)
        if out.shape != grid.shape:
            return None
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
