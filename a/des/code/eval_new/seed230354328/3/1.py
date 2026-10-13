import random

import numpy as np


BG = 0
FENCE = 5
SEED_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
ROT_K = {"up": 0, "right": 3, "down": 2, "left": 1}
PALETTE_CORNERS = ("tl", "tr", "bl", "br")
INNER_SIZES = (3, 5)
MAX_GEN_ATTEMPTS = 400
_RNG = random.Random()

PATTERN_LIBRARY = (
    np.array([[1, 1, 0], [0, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 1], [1, 0, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 0], [1, 0, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            qi = 0
            while qi < len(queue):
                cr, cc = queue[qi]
                qi += 1
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _tile_pattern(pattern: np.ndarray, shape: tuple[int, int], color: int) -> np.ndarray:
    ph, pw = pattern.shape
    oh, ow = shape
    rr = np.arange(oh)[:, None] % ph
    cc = np.arange(ow)[None, :] % pw
    mask = pattern[rr, cc]
    return np.where(mask, color, BG).astype(int)


def _extract_palette(grid: np.ndarray) -> dict[int, np.ndarray]:
    h, w = grid.shape
    candidates: list[tuple[int, int, dict[int, np.ndarray]]] = []
    for comp in _components(grid != BG):
        r0, c0, r1, c1 = _bbox(comp)
        touches_tb = r0 == 0 or r1 == h - 1
        touches_lr = c0 == 0 or c1 == w - 1
        if not (touches_tb and touches_lr):
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if patch.shape[1] < 3 or np.any(patch[:, 0] == BG):
            continue

        mapping: dict[int, np.ndarray] = {}
        row = 0
        ok = True
        while row < patch.shape[0]:
            color = int(patch[row, 0])
            end = row + 1
            while end < patch.shape[0] and int(patch[end, 0]) == color:
                end += 1
            texture = patch[row:end, 1:]
            valid = np.all((texture == BG) | (texture == color))
            if color in mapping or not valid or not np.any(texture == color):
                ok = False
                break
            mapping[color] = texture == color
            row = end

        if ok and len(mapping) >= 2:
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            candidates.append((len(mapping), area, mapping))

    assert candidates, "palette block not found"
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[-1][2]


def _find_fences(grid: np.ndarray) -> list[dict[str, int | str]]:
    fences: list[dict[str, int | str]] = []
    for comp in _components(grid == FENCE):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 4 or c1 - c0 < 4:
            continue
        coords = set(comp)
        expected = set()
        for c in range(c0, c1 + 1):
            expected.add((r0, c))
            expected.add((r1, c))
        for r in range(r0 + 1, r1):
            expected.add((r, c0))
            expected.add((r, c1))
        missing = list(expected - coords)
        extra = coords - expected
        if len(missing) != 1 or extra:
            continue

        gap_r, gap_c = missing[0]
        if gap_r == r0 and c0 < gap_c < c1:
            gap_dir = "up"
        elif gap_r == r1 and c0 < gap_c < c1:
            gap_dir = "down"
        elif gap_c == c0 and r0 < gap_r < r1:
            gap_dir = "left"
        elif gap_c == c1 and r0 < gap_r < r1:
            gap_dir = "right"
        else:
            continue

        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        if inner_h <= 0 or inner_w <= 0 or inner_h % 2 == 0 or inner_w % 2 == 0:
            continue
        center_r = r0 + 1 + inner_h // 2
        center_c = c0 + 1 + inner_w // 2
        seed_color = int(grid[center_r, center_c])
        assert seed_color not in (BG, FENCE), "fence seed must be a non-grey color"
        fences.append(
            {
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "seed_color": seed_color,
                "gap_dir": gap_dir,
            }
        )

    assert fences, "at least one fence must be present"
    fences.sort(key=lambda item: (int(item["r0"]), int(item["c0"])))
    return fences


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey fenced interior with the palette texture selected by its seed color and rotated by the fence gap direction.

    Input structure:
    - The background is black.
    - Several one-cell-thick grey rectangular fences appear in the grid; each fence has exactly one one-cell gap on its top, right, bottom, or left side.
    - Each fence contains a single colored seed pixel at the center of its black interior.
    - A palette block sits in one corner: every entry is a solid color label column next to a black-and-color texture patch in a shared default orientation.

    Rule:
    - Read the corner palette as a mapping from seed color to texture pattern.
    - For each grey fence, determine which side contains the gap.
    - Rotate the matched texture by 0 degrees for an upper gap, 90 degrees clockwise for a right gap, 180 degrees for a lower gap, and 270 degrees clockwise for a left gap, then tile it across the whole fence interior, overwriting the seed.

    Key insight:
    - The seed color chooses which texture to use, while the gap direction independently chooses how that texture must be rotated before tiling.
    """
    g = np.asarray(grid, dtype=int)
    palette = _extract_palette(g)
    out = g.copy()

    for fence in _find_fences(g):
        pattern = palette[int(fence["seed_color"])]
        rotated = np.rot90(pattern, ROT_K[str(fence["gap_dir"])])
        fill = _tile_pattern(rotated, (int(fence["inner_h"]), int(fence["inner_w"])), int(fence["seed_color"]))
        r0 = int(fence["r0"]) + 1
        c0 = int(fence["c0"]) + 1
        out[r0 : r0 + int(fence["inner_h"]), c0 : c0 + int(fence["inner_w"])] = fill

    return out


def _make_palette(entries: list[tuple[int, np.ndarray]]) -> np.ndarray:
    rows = []
    for color, pattern in entries:
        ph, pw = pattern.shape
        entry = np.zeros((ph, pw + 1), dtype=int)
        entry[:, 0] = color
        entry[:, 1:][pattern] = color
        rows.append(entry)
    return np.vstack(rows)


def _corner_position(corner: str, grid_shape: tuple[int, int], block_shape: tuple[int, int]) -> tuple[int, int]:
    gh, gw = grid_shape
    bh, bw = block_shape
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, gw - bw
    if corner == "bl":
        return gh - bh, 0
    return gh - bh, gw - bw


def _reserve(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> None:
    h, w = occupied.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h, r1 + pad + 1)
    cc1 = min(w, c1 + pad + 1)
    occupied[rr0:rr1, cc0:cc1] = True


def _fits(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> bool:
    h, w = occupied.shape
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h, r1 + pad + 1)
    cc1 = min(w, c1 + pad + 1)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _draw_fence(
    grid: np.ndarray,
    r0: int,
    c0: int,
    inner_h: int,
    inner_w: int,
    gap_dir: str,
    seed_color: int,
    gap_offset: int,
) -> dict[str, int | str]:
    r1 = r0 + inner_h + 1
    c1 = c0 + inner_w + 1
    grid[r0, c0 : c1 + 1] = FENCE
    grid[r1, c0 : c1 + 1] = FENCE
    grid[r0 : r1 + 1, c0] = FENCE
    grid[r0 : r1 + 1, c1] = FENCE

    if gap_dir == "up":
        grid[r0, c0 + gap_offset] = BG
    elif gap_dir == "down":
        grid[r1, c0 + gap_offset] = BG
    elif gap_dir == "left":
        grid[r0 + gap_offset, c0] = BG
    else:
        grid[r0 + gap_offset, c1] = BG

    center_r = r0 + 1 + inner_h // 2
    center_c = c0 + 1 + inner_w // 2
    grid[center_r, center_c] = seed_color
    return {
        "r0": r0,
        "c0": c0,
        "inner_h": inner_h,
        "inner_w": inner_w,
        "seed_color": seed_color,
        "gap_dir": gap_dir,
    }


def _expected_output(grid: np.ndarray, fences: list[dict[str, int | str]], mapping: dict[int, np.ndarray]) -> np.ndarray:
    out = grid.copy()
    for fence in fences:
        color = int(fence["seed_color"])
        rotated = np.rot90(mapping[color], ROT_K[str(fence["gap_dir"])])
        fill = _tile_pattern(rotated, (int(fence["inner_h"]), int(fence["inner_w"])), color)
        r0 = int(fence["r0"]) + 1
        c0 = int(fence["c0"]) + 1
        out[r0 : r0 + int(fence["inner_h"]), c0 : c0 + int(fence["inner_w"])] = fill
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_fences = rng.choice((2, 3, 4))
    seed_colors = rng.sample(SEED_COLORS, n_fences)
    patterns = rng.sample(PATTERN_LIBRARY, n_fences)
    color_to_pattern = {color: pattern for color, pattern in zip(seed_colors, patterns)}
    entries = list(color_to_pattern.items())
    rng.shuffle(entries)
    palette = _make_palette(entries)
    palette_corner = rng.choice(PALETTE_CORNERS)

    min_h = max(palette.shape[0] + 2, 16 + n_fences)
    min_w = max(palette.shape[1] + 8, 16 + n_fences)
    if min_h > 30 or min_w > 30:
        return None
    h = rng.randint(min_h, 30)
    w = rng.randint(min_w, 30)

    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    pr0, pc0 = _corner_position(palette_corner, grid.shape, palette.shape)
    grid[pr0 : pr0 + palette.shape[0], pc0 : pc0 + palette.shape[1]] = palette
    _reserve(occupied, pr0, pc0, pr0 + palette.shape[0] - 1, pc0 + palette.shape[1] - 1)

    directions = rng.sample(list(ROT_K.keys()), n_fences)
    fences: list[dict[str, int | str]] = []
    for color, gap_dir in zip(seed_colors, directions):
        inner_h = rng.choice(INNER_SIZES)
        inner_w = rng.choice(INNER_SIZES)
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        placed = False
        for _ in range(250):
            r0 = rng.randint(1, h - outer_h - 1)
            c0 = rng.randint(1, w - outer_w - 1)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1
            if not _fits(occupied, r0, c0, r1, c1):
                continue
            gap_offset = rng.randint(1, inner_w if gap_dir in ("up", "down") else inner_h)
            fences.append(_draw_fence(grid, r0, c0, inner_h, inner_w, gap_dir, color, gap_offset))
            _reserve(occupied, r0, c0, r1, c1)
            placed = True
            break
        if not placed:
            return None

    expected = _expected_output(grid, fences, color_to_pattern)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt(_RNG)
        if result is not None:
            return result
    raise RuntimeError(f"generate_input: failed after {MAX_GEN_ATTEMPTS} attempts")
