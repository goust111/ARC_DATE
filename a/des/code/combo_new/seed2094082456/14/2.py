import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _tight_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        out[r - r0, c - c0] = True
    return out


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _find_separator(grid: np.ndarray, bg: int) -> tuple[int, int]:
    candidates = [
        (r, int(grid[r, 0]))
        for r in range(grid.shape[0])
        if int(grid[r, 0]) != bg and np.all(grid[r] == grid[r, 0])
    ]
    assert len(candidates) == 1, "expected a unique full-width separator row"
    sep_row, grey = candidates[0]
    assert 0 < sep_row < grid.shape[0] - 1, "separator must split the grid into two non-empty parts"
    return sep_row, grey


def _extract_templates(top: np.ndarray, bg: int, grey: int) -> dict[int, np.ndarray]:
    templates: dict[int, np.ndarray] = {}
    for color in sorted(int(v) for v in np.unique(top) if int(v) not in {bg, grey}):
        comps = _components(top == color)
        assert len(comps) == 1, "each legend color must form exactly one template"
        mask = _tight_mask(comps[0])
        h, w = mask.shape
        assert h % 2 == 1 and w % 2 == 1, "template bbox must have odd dimensions"
        assert mask[h // 2, w // 2], "template anchor cell must be occupied"
        templates[color] = mask
    assert templates, "legend must contain at least one template"
    return templates


def _extract_seeds(bottom: np.ndarray, bg: int, grey: int, row_offset: int) -> list[tuple[int, int, int]]:
    seeds: list[tuple[int, int, int]] = []
    for r, c in np.argwhere((bottom != bg) & (bottom != grey)):
        seeds.append((int(r) + row_offset, int(c), int(bottom[r, c])))
    assert seeds, "canvas must contain at least one seed"
    return seeds


def _extract_boxes(bottom: np.ndarray, grey: int, row_offset: int) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for comp in _components(bottom == grey):
        r0, c0, r1, c1 = _bbox(comp)
        assert r1 - r0 + 1 >= 3 and c1 - c0 + 1 >= 3, "box bbox must have a non-empty interior"
        boxes.append((r0 + row_offset, c0, r1 + row_offset, c1))
    assert boxes, "canvas must contain at least one grey boundary box"
    boxes.sort()
    return boxes


def _tile_mask(mask: np.ndarray, height: int, width: int, seed_r: int, seed_c: int, top: int, left: int) -> np.ndarray:
    mh, mw = mask.shape
    cr = mh // 2
    cc = mw // 2
    rows = (np.arange(top, top + height)[:, None] - seed_r + cr) % mh
    cols = (np.arange(left, left + width)[None, :] - seed_c + cc) % mw
    return mask[rows, cols]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each seed color into a tiled fill pattern inside its matching grey box.

    Input structure:
    - The background is black, and one full grey horizontal line splits the grid into a legend area above and a canvas area below.
    - The legend area contains several separated monochrome template shapes; each template uses a distinct non-background color.
    - The canvas area contains isolated single-pixel colored seeds plus one or more grey rectangular boundary boxes, with each seed strictly inside exactly one box.

    Rule:
    - Match every seed to the legend template that has the same color.
    - Treat the matched template's tight odd-sized bbox center as the anchor, align that anchor to the seed, and repeat the template periodically in both row and column directions.
    - Replace only the interior of the seed's grey box with the resulting tiled color mask; keep the grey border and everything outside the box unchanged.

    Key insight:
    - Color selects the template, but the template is not stamped once: it becomes a wallpaper whose phase is locked to the seed and whose extent is clipped by the surrounding grey rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background_color(g)
    sep_row, grey = _find_separator(g, bg)
    templates = _extract_templates(g[:sep_row], bg, grey)
    seeds = _extract_seeds(g[sep_row + 1 :], bg, grey, sep_row + 1)
    boxes = _extract_boxes(g[sep_row + 1 :], grey, sep_row + 1)

    out = g.copy()
    used_seed_positions: set[tuple[int, int]] = set()
    for r0, c0, r1, c1 in boxes:
        matches = [(sr, sc, color) for sr, sc, color in seeds if r0 < sr < r1 and c0 < sc < c1]
        assert len(matches) == 1, "each grey box must contain exactly one seed"
        seed_r, seed_c, color = matches[0]
        assert (seed_r, seed_c) not in used_seed_positions, "a seed may only drive one box"
        assert color in templates, "every seed color must appear in the legend"
        used_seed_positions.add((seed_r, seed_c))

        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        tile = _tile_mask(templates[color], inner_h, inner_w, seed_r, seed_c, r0 + 1, c0 + 1)
        region = out[r0 + 1 : r1, c0 + 1 : c1]
        region[:] = bg
        region[tile] = color

    assert len(used_seed_positions) == len(seeds), "every seed must belong to exactly one box"
    return out


def _array_from_rows(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


def _cross(size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    center = size // 2
    mask[center, :] = True
    mask[:, center] = True
    return mask


def _diamond(size: int) -> np.ndarray:
    center = size // 2
    mask = np.zeros((size, size), dtype=bool)
    for r in range(size):
        for c in range(size):
            if abs(r - center) + abs(c - center) <= center:
                mask[r, c] = True
    return mask


def _diag_band(size: int, slope: int) -> np.ndarray:
    center = size // 2
    mask = np.zeros((size, size), dtype=bool)
    for r in range(size):
        for c in range(size):
            if abs((c - center) - slope * (r - center)) <= 1:
                mask[r, c] = True
    return mask


def _tee(orientation: str, size: int = 5) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    center = size // 2
    if orientation == "up":
        mask[0, :] = True
        mask[:, center] = True
    elif orientation == "down":
        mask[-1, :] = True
        mask[:, center] = True
    elif orientation == "left":
        mask[:, 0] = True
        mask[center, :] = True
    elif orientation == "right":
        mask[:, -1] = True
        mask[center, :] = True
    else:
        raise ValueError(orientation)
    return mask


def _halfplane(size: int, sr: int, sc: int) -> np.ndarray:
    center = size // 2
    mask = np.zeros((size, size), dtype=bool)
    for r in range(size):
        for c in range(size):
            if sr * (r - center) + sc * (c - center) >= 0:
                mask[r, c] = True
    return mask


def _build_pattern_bank() -> list[np.ndarray]:
    bank = [
        _cross(3),
        _cross(5),
        _diamond(5),
        _diag_band(5, 1),
        _diag_band(5, -1),
        _tee("up"),
        _tee("down"),
        _tee("left"),
        _tee("right"),
        _halfplane(5, 1, 1),
        _halfplane(5, 1, -1),
        _halfplane(5, -1, 1),
        _halfplane(5, -1, -1),
        _array_from_rows(["00110", "00110", "00111", "00011", "00011"]),
        _array_from_rows(["01100", "01100", "11100", "11000", "11000"]),
    ]

    unique: list[np.ndarray] = []
    signatures: set[tuple[tuple[int, int], bytes]] = set()
    for mask in bank:
        assert mask.shape[0] % 2 == 1 and mask.shape[1] % 2 == 1
        assert mask[mask.shape[0] // 2, mask.shape[1] // 2]
        assert len(_components(mask)) == 1
        sig = (mask.shape, mask.astype(np.uint8).tobytes())
        if sig not in signatures:
            signatures.add(sig)
            unique.append(mask)
    return unique


PATTERN_BANK = _build_pattern_bank()


def _rect_free(used: np.ndarray, top: int, left: int, height: int, width: int, pad: int) -> bool:
    h, w = used.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    return not np.any(used[r0:r1, c0:c1])


def _stamp_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int):
    patch = grid[top : top + mask.shape[0], left : left + mask.shape[1]]
    patch[mask] = color


def _make_box_patch(height: int, width: int, rng: random.Random) -> np.ndarray:
    patch = np.zeros((height, width), dtype=int)
    patch[0, :] = GREY
    patch[-1, :] = GREY
    patch[:, 0] = GREY
    patch[:, -1] = GREY

    if rng.random() < 0.5:
        side = rng.choice(("top", "bottom", "left", "right"))
        if side in {"top", "bottom"}:
            span = width - 2
            gap_len = rng.randint(1, min(3, span))
            start = rng.randint(1, width - gap_len - 1)
            row = 0 if side == "top" else height - 1
            patch[row, start : start + gap_len] = BG
        else:
            span = height - 2
            gap_len = rng.randint(1, min(3, span))
            start = rng.randint(1, height - gap_len - 1)
            col = 0 if side == "left" else width - 1
            patch[start : start + gap_len, col] = BG
    return patch


def _sample_box_dims(rng: random.Random, bottom_h: int, width: int, n_boxes: int) -> tuple[int, int]:
    if n_boxes >= 5:
        h_lo, h_hi = 5, min(7, bottom_h)
        w_lo, w_hi = 5, min(7, width)
    elif n_boxes == 4:
        h_lo, h_hi = 5, min(8, bottom_h)
        w_lo, w_hi = 5, min(8, width)
    elif n_boxes == 3:
        h_lo, h_hi = 5, min(8, bottom_h)
        w_lo, w_hi = 5, min(9, width)
    else:
        h_lo, h_hi = 5, min(9, bottom_h)
        w_lo, w_hi = 5, min(10, width)
    if h_hi < h_lo or w_hi < w_lo:
        raise ValueError("canvas too small for boxes")
    return rng.randint(h_lo, h_hi), rng.randint(w_lo, w_hi)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(18, 30)
    width = rng.randint(18, 30)
    top_h = rng.randint(6, min(11, height - 9))
    bottom_h = height - top_h - 1
    if bottom_h < 8:
        return None

    n_templates = rng.randint(2, 4)
    n_boxes = rng.randint(n_templates, min(5, n_templates + 2))
    colors = rng.sample(COLOR_POOL, n_templates)
    templates = [mask.copy() for mask in rng.sample(PATTERN_BANK, n_templates)]

    grid = np.full((height, width), BG, dtype=int)
    grid[top_h, :] = GREY

    top_used = np.zeros((top_h, width), dtype=bool)
    for color, mask in zip(colors, templates):
        mh, mw = mask.shape
        placed = False
        for _ in range(120):
            top_limit = top_h - mh
            if top_limit < 0 or width - mw < 0:
                return None
            r = rng.randint(0, top_limit)
            c = rng.randint(0, width - mw)
            if not _rect_free(top_used, r, c, mh, mw, pad=1):
                continue
            _stamp_mask(grid, mask, r, c, color)
            top_used[r : r + mh, c : c + mw] = True
            placed = True
            break
        if not placed:
            return None

    box_colors = colors.copy()
    while len(box_colors) < n_boxes:
        box_colors.append(rng.choice(colors))
    rng.shuffle(box_colors)

    bottom_used = np.zeros((bottom_h, width), dtype=bool)
    for color in box_colors:
        placed = False
        for _ in range(180):
            box_h, box_w = _sample_box_dims(rng, bottom_h, width, n_boxes)
            if bottom_h - box_h < 0 or width - box_w < 0:
                return None
            r = rng.randint(0, bottom_h - box_h)
            c = rng.randint(0, width - box_w)
            if not _rect_free(bottom_used, r, c, box_h, box_w, pad=1):
                continue

            patch = _make_box_patch(box_h, box_w, rng)
            abs_r = top_h + 1 + r
            box_region = grid[abs_r : abs_r + box_h, c : c + box_w]
            box_region[patch == GREY] = GREY

            seed_r = rng.randint(abs_r + 1, abs_r + box_h - 2)
            seed_c = rng.randint(c + 1, c + box_w - 2)
            grid[seed_r, seed_c] = color
            bottom_used[r : r + box_h, c : c + box_w] = True
            placed = True
            break
        if not placed:
            return None

    bg = _background_color(grid)
    if bg != BG:
        return None
    if Counter(grid.ravel())[GREY] >= Counter(grid.ravel())[BG]:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if not np.any(out != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
