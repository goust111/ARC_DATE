import random

import numpy as np


BG = 0
GREY = 5
TYPE_ORDER = ("horizontal", "vertical", "rotational", "asymmetric")

LEGEND_MASKS = {
    "horizontal": np.array(
        [
            [1, 0, 0],
            [0, 1, 1],
            [1, 0, 0],
        ],
        dtype=bool,
    ),
    "vertical": np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    "rotational": np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        dtype=bool,
    ),
    "asymmetric": np.array(
        [
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
        dtype=bool,
    ),
}


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _same_color_components(grid: np.ndarray, excluded: set[int]) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, list[tuple[int, int]]]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color in excluded:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append((color, pts))
    return comps


def _find_frames(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for color, pts in _same_color_components(grid, {bg, GREY}):
        r0, c0, r1, c1 = _bbox(pts)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        border = set()
        for c in range(c0, c1 + 1):
            border.add((r0, c))
            border.add((r1, c))
        for r in range(r0 + 1, r1):
            border.add((r, c0))
            border.add((r, c1))
        if set(pts) != border:
            continue
        frames.append(
            {
                "color": int(color),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
            }
        )
    return frames


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _classify_mask(mask: np.ndarray) -> str:
    m = np.asarray(mask, dtype=bool)
    if np.array_equal(m, np.flipud(m)):
        return "horizontal"
    if np.array_equal(m, np.fliplr(m)):
        return "vertical"
    if np.array_equal(m, np.rot90(m, 2)):
        return "rotational"
    return "asymmetric"


def _legend_palette_present(grid: np.ndarray, colors: list[int]) -> bool:
    h, w = grid.shape
    target_row = np.asarray(colors, dtype=int)
    for c in range(w - 3):
        if np.array_equal(grid[h - 1, c : c + 4], target_row):
            return True
    target_col = np.asarray(colors, dtype=int)
    for r in range(h - 3):
        if np.array_equal(grid[r : r + 4, w - 1], target_col):
            return True
    return False


def _find_legend_mapping(grid: np.ndarray, bg: int, frames: list[dict[str, object]]) -> dict[str, int]:
    h, w = grid.shape
    blocked = np.zeros((h, w), dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        blocked[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = True

    best: tuple[int, dict[str, int]] | None = None
    for r0 in range(h - 11):
        for c0 in range(w - 3):
            if c0 + 3 >= w:
                continue
            colors: list[int] = []
            mapping: dict[str, int] = {}
            ok = True
            for i in range(4):
                rr = r0 + 3 * i
                if blocked[rr : rr + 3, c0 : c0 + 4].any():
                    ok = False
                    break
                sample = grid[rr : rr + 3, c0 : c0 + 3]
                if sample.shape != (3, 3):
                    ok = False
                    break
                if not np.all((sample == bg) | (sample == GREY)):
                    ok = False
                    break
                if not np.any(sample == GREY):
                    ok = False
                    break
                if int(grid[rr, c0 + 3]) != bg or int(grid[rr + 2, c0 + 3]) != bg:
                    ok = False
                    break
                color = int(grid[rr + 1, c0 + 3])
                if color in (bg, GREY):
                    ok = False
                    break
                kind = _classify_mask(sample == GREY)
                if kind in mapping:
                    ok = False
                    break
                mapping[kind] = color
                colors.append(color)
            if not ok or set(mapping) != set(TYPE_ORDER):
                continue
            if not _legend_palette_present(grid, colors):
                continue
            score = c0 + (w - (c0 + 4))
            if best is None or score > best[0]:
                best = (score, mapping)
    assert best is not None, "expected one valid legend"
    return best[1]


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each framed container according to the symmetry class of its grey pattern."""
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frames = _find_frames(g, bg)
    mapping = _find_legend_mapping(g, bg, frames)

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        inner = out[int(r0) + 1 : int(r1), int(c0) + 1 : int(c1)]
        grey_mask = inner == GREY
        if not np.any(grey_mask):
            continue
        kind = _classify_mask(_tight_mask(grey_mask))
        fill_color = mapping[kind]
        inner[inner == bg] = int(fill_color)
    return out


def _tight_shape(mask: np.ndarray) -> tuple[int, int]:
    return _tight_mask(mask).shape


def _random_horizontal(n: int, rng: random.Random) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    for r in range((n + 1) // 2):
        row = np.array([rng.random() < 0.45 for _ in range(n)], dtype=bool)
        mask[r] = row
        mask[n - 1 - r] = row
    return mask


def _random_vertical(n: int, rng: random.Random) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    for c in range((n + 1) // 2):
        col = np.array([rng.random() < 0.45 for _ in range(n)], dtype=bool)
        mask[:, c] = col
        mask[:, n - 1 - c] = col
    return mask


def _random_rotational(n: int, rng: random.Random) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    for r in range(n):
        for c in range(n):
            rr, cc = n - 1 - r, n - 1 - c
            if (r, c) > (rr, cc):
                continue
            val = rng.random() < 0.45
            mask[r, c] = val
            mask[rr, cc] = val
    return mask


def _random_asymmetric(n: int, rng: random.Random) -> np.ndarray:
    return np.array([[rng.random() < 0.45 for _ in range(n)] for _ in range(n)], dtype=bool)


def _pattern_for_kind(kind: str, n: int, rng: random.Random) -> np.ndarray:
    makers = {
        "horizontal": _random_horizontal,
        "vertical": _random_vertical,
        "rotational": _random_rotational,
        "asymmetric": _random_asymmetric,
    }
    for _ in range(300):
        mask = makers[kind](n, rng)
        count = int(mask.sum())
        if count < max(3, n - 1) or count > n * n - 2:
            continue
        if _tight_shape(mask) != (n, n):
            continue
        if _classify_mask(mask) != kind:
            continue
        return mask
    raise RuntimeError(f"failed to generate {kind} pattern")


def _draw_legend(grid: np.ndarray, colors: list[int], legend_r: int, legend_c: int) -> None:
    for i, kind in enumerate(TYPE_ORDER):
        rr = legend_r + 3 * i
        mask = LEGEND_MASKS[kind]
        grid[rr : rr + 3, legend_c : legend_c + 3][mask] = GREY
        grid[rr + 1, legend_c + 3] = int(colors[i])


def _mark_reserved(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> None:
    h, w = mask.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _make_container(border_color: int, kind: str, rng: random.Random) -> np.ndarray:
    n = rng.randint(3, 5)
    inner_h = n + rng.randint(1, 2)
    inner_w = n + rng.randint(1, 2)
    outer = np.full((inner_h + 2, inner_w + 2), BG, dtype=int)
    outer[0, :] = border_color
    outer[-1, :] = border_color
    outer[:, 0] = border_color
    outer[:, -1] = border_color

    pattern = _pattern_for_kind(kind, n, rng)
    pr = rng.randint(0, inner_h - n)
    pc = rng.randint(0, inner_w - n)
    region = outer[1 + pr : 1 + pr + n, 1 + pc : 1 + pc + n]
    region[pattern] = GREY
    return outer


def _try_place_rect(blocked: np.ndarray, shape: tuple[int, int], rng: random.Random) -> tuple[int, int] | None:
    h, w = blocked.shape
    ph, pw = shape
    rows = list(range(h - ph + 1))
    cols = list(range(w - pw + 1))
    rng.shuffle(rows)
    rng.shuffle(cols)
    for r in rows:
        for c in cols:
            if blocked[r : r + ph, c : c + pw].any():
                continue
            return r, c
    return None


def _build_grid(rng: random.Random) -> np.ndarray:
    fill_colors = rng.sample([1, 2, 3, 4, 6, 7, 8, 9], 4)
    remaining = [c for c in [1, 2, 3, 4, 6, 7, 8, 9] if c not in fill_colors]
    n_containers = rng.choice([2, 3, 4])
    border_colors = rng.sample(remaining, n_containers)
    h = rng.randint(20 + 2 * n_containers, 24 + 2 * n_containers)
    w = rng.randint(22 + 2 * n_containers, 28 + 2 * n_containers)
    grid = np.full((h, w), BG, dtype=int)

    legend_r = rng.randint(1, max(1, h - 13))
    legend_c_low = max(1, w - 8)
    legend_c_high = max(1, w - 5)
    legend_c = rng.randint(legend_c_low, legend_c_high)
    if legend_c + 3 >= w:
        legend_c = w - 4
    if legend_r + 11 >= h:
        legend_r = h - 12
    _draw_legend(grid, fill_colors, legend_r, legend_c)

    blocked = np.zeros((h, w), dtype=bool)
    _mark_reserved(blocked, legend_r, legend_c, legend_r + 11, legend_c + 3, pad=1)

    palette_side = rng.choice(["bottom", "right"])
    if palette_side == "bottom":
        palette_c = rng.randint(0, w - 4)
        grid[h - 1, palette_c : palette_c + 4] = np.array(fill_colors, dtype=int)
        _mark_reserved(blocked, h - 1, palette_c, h - 1, palette_c + 3, pad=1)
    else:
        palette_r = rng.randint(0, h - 4)
        grid[palette_r : palette_r + 4, w - 1] = np.array(fill_colors, dtype=int)
        _mark_reserved(blocked, palette_r, w - 1, palette_r + 3, w - 1, pad=1)

    kinds = rng.sample(list(TYPE_ORDER), n_containers)
    for border_color, kind in zip(border_colors, kinds, strict=False):
        patch = _make_container(border_color, kind, rng)
        loc = _try_place_rect(blocked, patch.shape, rng)
        if loc is None:
            raise RuntimeError("failed to place container")
        r, c = loc
        ph, pw = patch.shape
        grid[r : r + ph, c : c + pw] = patch
        _mark_reserved(blocked, r, c, r + ph - 1, c + pw - 1, pad=1)

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        rng = random.Random()
        try:
            grid = _build_grid(rng)
        except RuntimeError:
            continue
        out = main(grid)
        if np.array_equal(out, grid):
            continue
        return grid
    raise RuntimeError("failed to generate valid input")
