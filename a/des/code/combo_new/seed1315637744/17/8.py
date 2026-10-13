import random

import numpy as np


BG = 0
MASTER_SIZE = 30
OPEN_TO_K = {"rd": 0, "ld": 3, "lu": 2, "ru": 1}
SLOTS = [
    (0, 13, 0, 13),
    (0, 13, 16, 29),
    (16, 29, 0, 13),
    (16, 29, 16, 29),
]


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _l_mask(size: int, opening: str) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    if opening == "rd":
        mask[0, :] = True
        mask[:, 0] = True
    elif opening == "ld":
        mask[0, :] = True
        mask[:, -1] = True
    elif opening == "lu":
        mask[-1, :] = True
        mask[:, -1] = True
    elif opening == "ru":
        mask[-1, :] = True
        mask[:, 0] = True
    else:
        raise ValueError(f"unknown opening {opening}")
    return mask


def _classify_l(mask: np.ndarray) -> str | None:
    size = int(mask.shape[0])
    if mask.shape[0] != mask.shape[1] or size not in (2, 3):
        return None
    for opening in ("rd", "ld", "lu", "ru"):
        if np.array_equal(mask, _l_mask(size, opening)):
            return opening
    return None


def _boundary_colors(square: np.ndarray) -> set[int]:
    vals = set(map(int, square[0, :]))
    vals.update(map(int, square[-1, :]))
    vals.update(map(int, square[1:-1, 0]))
    vals.update(map(int, square[1:-1, -1]))
    return vals


def _points_to(marker: dict, square: dict) -> bool:
    mr = (int(marker["r0"]) + int(marker["r1"])) / 2.0
    mc = (int(marker["c0"]) + int(marker["c1"])) / 2.0
    sr = (int(square["r0"]) + int(square["r1"])) / 2.0
    sc = (int(square["c0"]) + int(square["c1"])) / 2.0
    opening = marker["opening"]
    if opening == "rd":
        return sr > mr and sc > mc
    if opening == "ld":
        return sr > mr and sc < mc
    if opening == "lu":
        return sr < mr and sc < mc
    return sr < mr and sc > mc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each square's top-left sub-block according to its nearby L marker.

    Input structure:
    - The background is black.
    - Several large non-overlapping square regions contain multicolor pixel patterns.
    - Separate monochrome L-shaped markers sit outside the squares; each marker has equal arm length 2 or 3 and points toward one square.

    Rule:
    - Detect each filled square region and each L marker as separate foreground components.
    - Pair a marker with the unique square whose boundary contains the marker's color.
    - Let N be the marker arm length, extract the square's top-left N×N block, rotate it by the angle encoded by the marker opening, and write it back to the same top-left location.
    - Keep every marker unchanged and leave all other pixels untouched.

    Key insight:
    - The marker color identifies the target square, the marker size gives the extraction size, and the opening direction gives the rotation amount.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    comps = _find_components(g != BG)

    squares: list[dict] = []
    markers: list[dict] = []
    for coords in comps:
        rows = coords[:, 0]
        cols = coords[:, 1]
        r0 = int(rows.min())
        r1 = int(rows.max())
        c0 = int(cols.min())
        c1 = int(cols.max())
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]

        if h == w and h >= 4 and int(coords.shape[0]) == h * w and bool((patch != BG).all()):
            squares.append(
                {
                    "r0": r0,
                    "c0": c0,
                    "r1": r1,
                    "c1": c1,
                    "boundary_colors": _boundary_colors(patch),
                }
            )
            continue

        colors = {int(g[int(r), int(c)]) for r, c in coords}
        if len(colors) != 1 or h != w or h not in (2, 3) or int(coords.shape[0]) != 2 * h - 1:
            continue
        rel = np.zeros((h, w), dtype=bool)
        rel[rows - r0, cols - c0] = True
        opening = _classify_l(rel)
        if opening is None:
            continue
        markers.append(
            {
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "size": h,
                "opening": opening,
                "color": next(iter(colors)),
            }
        )

    out = g.copy()
    for marker in markers:
        candidates = [sq for sq in squares if int(marker["color"]) in sq["boundary_colors"]]
        if len(candidates) != 1:
            candidates = [sq for sq in candidates if _points_to(marker, sq)]
        assert len(candidates) == 1
        square = candidates[0]
        n = int(marker["size"])
        r0 = int(square["r0"])
        c0 = int(square["c0"])
        block = g[r0 : r0 + n, c0 : c0 + n]
        out[r0 : r0 + n, c0 : c0 + n] = np.rot90(block, OPEN_TO_K[marker["opening"]])

    return out


def _build_square(side: int, border_color: int, inner_colors: list[int], n: int, opening: str) -> np.ndarray:
    for _ in range(200):
        patch = np.full((side, side), border_color, dtype=int)
        for r in range(1, side - 1):
            for c in range(1, side - 1):
                patch[r, c] = random.choice(inner_colors)
        if side >= 6:
            stripe_col = random.choice(inner_colors)
            if random.random() < 0.5:
                patch[1:-1, random.randint(1, side - 2)] = stripe_col
            else:
                patch[random.randint(1, side - 2), 1:-1] = stripe_col

        top_left = patch[:n, :n]
        if opening != "rd" and np.array_equal(top_left, np.rot90(top_left, OPEN_TO_K[opening])):
            continue
        if len(set(map(int, patch.flatten()))) < 3:
            continue
        return patch
    raise RuntimeError("failed to build a non-degenerate square")


def _marker_top_left(square_r: int, square_c: int, side: int, gap: int, n: int, opening: str) -> tuple[int, int]:
    if opening == "rd":
        return square_r - gap - n, square_c - gap - n
    if opening == "ld":
        return square_r - gap - n, square_c + side + gap
    if opening == "lu":
        return square_r + side + gap, square_c + side + gap
    return square_r + side + gap, square_c - gap - n


def _square_ranges(slot: tuple[int, int, int, int], side: int, gap: int, n: int, opening: str) -> tuple[range, range] | None:
    r0, r1, c0, c1 = map(int, slot)
    if opening == "rd":
        rr = range(r0 + gap + n, r1 - side + 2)
        cc = range(c0 + gap + n, c1 - side + 2)
    elif opening == "ld":
        rr = range(r0 + gap + n, r1 - side + 2)
        cc = range(c0, c1 - side - gap - n + 2)
    elif opening == "lu":
        rr = range(r0, r1 - side - gap - n + 2)
        cc = range(c0, c1 - side - gap - n + 2)
    else:
        rr = range(r0, r1 - side - gap - n + 2)
        cc = range(c0 + gap + n, c1 - side + 2)
    if len(rr) == 0 or len(cc) == 0:
        return None
    return rr, cc


def _paint_marker(grid: np.ndarray, top: int, left: int, color: int, n: int, opening: str) -> None:
    mask = _l_mask(n, opening)
    grid[top : top + n, left : left + n][mask] = color


def _crop_with_margin(grid: np.ndarray) -> np.ndarray:
    coords = np.argwhere(grid != BG)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    top = random.randint(0, min(2, int(r0)))
    left = random.randint(0, min(2, int(c0)))
    bottom = random.randint(0, min(2, int(MASTER_SIZE - 1 - r1)))
    right = random.randint(0, min(2, int(MASTER_SIZE - 1 - c1)))
    return grid[int(r0) - top : int(r1) + bottom + 1, int(c0) - left : int(c1) + right + 1]


def _gen_attempt() -> np.ndarray | None:
    grid = np.zeros((MASTER_SIZE, MASTER_SIZE), dtype=int)
    n_objects = random.randint(2, 4)
    slots = random.sample(SLOTS, n_objects)
    openings = random.sample(["rd", "ld", "lu", "ru"], n_objects)
    if n_objects == 2:
        sizes = [2, 3]
    elif n_objects == 3:
        sizes = [2, 3, random.choice([2, 3])]
    else:
        sizes = [2, 2, 3, 3]
    random.shuffle(sizes)

    border_colors = random.sample(list(range(1, 10)), n_objects)
    inner_palette = [c for c in range(1, 10) if c not in border_colors]
    changed_branch = False

    for slot, opening, n, border_color in zip(slots, openings, sizes, border_colors):
        side = random.randint(max(5, n + 2), 8)
        gap = random.randint(1, 2)
        ranges = _square_ranges(slot, side, gap, n, opening)
        if ranges is None:
            return None
        row_range, col_range = ranges
        square_r = random.choice(list(row_range))
        square_c = random.choice(list(col_range))
        marker_r, marker_c = _marker_top_left(square_r, square_c, side, gap, n, opening)
        if not (0 <= marker_r and 0 <= marker_c and marker_r + n <= MASTER_SIZE and marker_c + n <= MASTER_SIZE):
            return None

        inner_colors = random.sample(inner_palette, random.randint(2, min(3, len(inner_palette))))
        square = _build_square(side, border_color, inner_colors, n, opening)
        grid[square_r : square_r + side, square_c : square_c + side] = square
        _paint_marker(grid, marker_r, marker_c, border_color, n, opening)

        if opening != "rd":
            changed_branch = True

    if not changed_branch:
        return None

    cropped = _crop_with_margin(grid)
    solved = main(cropped)
    if np.array_equal(cropped, solved):
        return None
    return cropped


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
