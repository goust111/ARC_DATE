import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAROON = 9

SEED_COLORS = (RED, BLUE, YELLOW)
DIRECTIONS = ("up", "right", "down", "left")
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)

GEN_ATTEMPTS = 300
PLACEMENT_TRIES = 200
GRID_RANGES = {
    4: ((22, 28), (22, 30)),
    5: ((24, 31), (24, 32)),
    6: ((27, 34), (26, 34)),
}
LEGEND_LAYOUTS = (
    ((3, 11), ((0, 0), (0, 4), (0, 8))),
    ((11, 3), ((0, 0), (4, 0), (8, 0))),
    ((7, 7), ((0, 0), (0, 4), (4, 0))),
    ((7, 7), ((0, 0), (0, 4), (4, 4))),
)


def _components(mask: np.ndarray, neighbors=DIR4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
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
                for dr, dc in neighbors:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _find_legend_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for coords in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 5 or w < 5:
            continue
        if len(coords) != 2 * (h + w) - 4:
            continue
        if not np.all(grid[r0, c0 : c1 + 1] == GREY):
            continue
        if not np.all(grid[r1, c0 : c1 + 1] == GREY):
            continue
        if not np.all(grid[r0 : r1 + 1, c0] == GREY):
            continue
        if not np.all(grid[r0 : r1 + 1, c1] == GREY):
            continue
        candidates.append((h * w, (r0, c0, r1, c1)))
    assert candidates, "expected one grey legend frame"
    candidates.sort(reverse=True)
    return candidates[0][1]


def _extract_templates(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> dict[int, np.ndarray]:
    r0, c0, r1, c1 = bbox
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    ih, iw = interior.shape
    found: dict[int, np.ndarray] = {}
    used: list[tuple[int, int]] = []

    for sr in range(ih - 2):
        for sc in range(iw - 2):
            window = interior[sr : sr + 3, sc : sc + 3]
            if np.any(window == GREEN):
                continue
            vals = set(map(int, np.unique(window)))
            if not vals <= {BLACK, RED, BLUE, YELLOW}:
                continue
            colors = vals - {BLACK}
            if len(colors) != 1:
                continue
            color = colors.pop()
            if color in found:
                continue
            overlaps = any(not (sr + 2 < ur or ur + 2 < sr or sc + 2 < uc or uc + 2 < sc) for ur, uc in used)
            if overlaps:
                continue
            mask = window == color
            if not mask[2, 1]:
                continue
            found[color] = mask
            used.append((sr, sc))

    assert set(found) == set(SEED_COLORS), "legend must expose one template per seed color"
    return found


def _flowerbeds(grid: np.ndarray) -> list[tuple[int, int, str, int]]:
    beds: list[tuple[int, int, str, int]] = []
    for coords in _components(grid == MAROON):
        r0, c0, r1, c1 = _bbox(coords)
        if (r1 - r0 + 1, c1 - c0 + 1) != (5, 5):
            continue
        if len(coords) != 15:
            continue

        gap_dirs = []
        if int(grid[r0, c0 + 2]) != MAROON:
            gap_dirs.append("up")
        if int(grid[r0 + 2, c1]) != MAROON:
            gap_dirs.append("right")
        if int(grid[r1, c0 + 2]) != MAROON:
            gap_dirs.append("down")
        if int(grid[r0 + 2, c0]) != MAROON:
            gap_dirs.append("left")
        assert len(gap_dirs) == 1, "each flowerbed must have exactly one central gap"

        seed = int(grid[r0 + 2, c0 + 2])
        assert seed in SEED_COLORS, "seed color must be red, blue, or yellow"
        beds.append((r0, c0, gap_dirs[0], seed))

    assert beds, "expected at least one maroon flowerbed"
    return beds


def _rotate_template(mask: np.ndarray, direction: str) -> np.ndarray:
    if direction == "up":
        return mask
    if direction == "right":
        return np.rot90(mask, -1)
    if direction == "down":
        return np.rot90(mask, 2)
    if direction == "left":
        return np.rot90(mask, 1)
    raise AssertionError(f"unknown direction: {direction}")


def _flower_top_left(r0: int, c0: int, direction: str) -> tuple[int, int]:
    if direction == "up":
        return r0 - 3, c0 + 1
    if direction == "right":
        return r0 + 1, c0 + 5
    if direction == "down":
        return r0 + 5, c0 + 1
    if direction == "left":
        return r0 + 1, c0 - 3
    raise AssertionError(f"unknown direction: {direction}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a rotated flower template outside each flowerbed based on seed color and fence gap direction.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    legend_bbox = _find_legend_bbox(g)
    templates = _extract_templates(g, legend_bbox)

    lr0, lc0, lr1, lc1 = legend_bbox
    out[lr0 : lr1 + 1, lc0 : lc1 + 1] = GREEN

    for r0, c0, direction, seed in _flowerbeds(g):
        mask = _rotate_template(templates[seed], direction)
        fr, fc = _flower_top_left(r0, c0, direction)
        patch = out[fr : fr + 3, fc : fc + 3]
        assert patch.shape == (3, 3), "flower placement must stay in bounds"
        patch[mask] = seed

    return out


def _mask_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    seen = {tuple(map(int, pts[0]))}
    stack = [tuple(map(int, pts[0]))]
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR8:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < 3 and 0 <= nc < 3 and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _canonical_rotation(mask: np.ndarray) -> bytes:
    return min(np.rot90(mask, k).tobytes() for k in range(4))


def _build_template_candidates() -> list[np.ndarray]:
    cells = [(r, c) for r in range(3) for c in range(3) if (r, c) != (2, 1)]
    candidates: list[np.ndarray] = []
    for bits in range(1 << len(cells)):
        mask = np.zeros((3, 3), dtype=bool)
        mask[2, 1] = True
        for idx, (r, c) in enumerate(cells):
            if bits & (1 << idx):
                mask[r, c] = True
        area = int(mask.sum())
        if area < 4 or area > 6:
            continue
        if not mask[0].any():
            continue
        if not _mask_connected(mask):
            continue
        if len({_canonical_rotation(np.rot90(mask, k)) for k in range(4)}) != 1:
            continue
        if len({np.rot90(mask, k).tobytes() for k in range(4)}) < 4:
            continue
        candidates.append(mask)
    assert candidates, "template candidate pool must be non-empty"
    return candidates


TEMPLATE_CANDIDATES = _build_template_candidates()


def _sample_templates(rng: random.Random) -> dict[int, np.ndarray]:
    order = list(range(len(TEMPLATE_CANDIDATES)))
    rng.shuffle(order)
    picked: list[np.ndarray] = []
    seen: set[bytes] = set()
    for idx in order:
        mask = TEMPLATE_CANDIDATES[idx]
        sig = _canonical_rotation(mask)
        if sig in seen:
            continue
        seen.add(sig)
        picked.append(mask.copy())
        if len(picked) == 3:
            break
    assert len(picked) == 3, "need three distinct template shapes"
    rng.shuffle(picked)
    return {color: mask for color, mask in zip(SEED_COLORS, picked)}


def _balanced_choices(rng: random.Random, items: tuple[int, ...] | tuple[str, ...], count: int) -> list:
    base = list(items)
    if count <= len(base):
        picks = rng.sample(base, count)
    else:
        picks = base.copy()
        extra = count - len(base)
        full, rem = divmod(extra, len(base))
        picks.extend(base * full)
        if rem:
            picks.extend(rng.sample(base, rem))
    rng.shuffle(picks)
    return picks


def _legend_patch(rng: random.Random, templates: dict[int, np.ndarray]) -> np.ndarray:
    (ih, iw), slots = rng.choice(LEGEND_LAYOUTS)
    patch = np.full((ih + 2, iw + 2), GREEN, dtype=int)
    patch[0, :] = GREY
    patch[-1, :] = GREY
    patch[:, 0] = GREY
    patch[:, -1] = GREY

    colors = list(SEED_COLORS)
    rng.shuffle(colors)
    for color, (sr, sc) in zip(colors, slots):
        slot = np.full((3, 3), BLACK, dtype=int)
        slot[templates[color]] = color
        patch[1 + sr : 1 + sr + 3, 1 + sc : 1 + sc + 3] = slot
    return patch


def _reserve_rect(rect: tuple[int, int, int, int], margin: int = 1) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = rect
    return r0 - margin, c0 - margin, r1 + margin, c1 + margin


def _rect_free(occupied: np.ndarray, rect: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = rect
    return not occupied[r0 : r1 + 1, c0 : c1 + 1].any()


def _mark_rect(occupied: np.ndarray, rect: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = rect
    occupied[r0 : r1 + 1, c0 : c1 + 1] = True


def _random_rect_position(
    rng: random.Random,
    occupied: np.ndarray,
    shape: tuple[int, int],
) -> tuple[int, int] | None:
    h, w = occupied.shape
    ph, pw = shape
    if ph + 2 > h or pw + 2 > w:
        return None
    for _ in range(PLACEMENT_TRIES):
        r0 = rng.randint(1, h - ph - 1)
        c0 = rng.randint(1, w - pw - 1)
        rect = (r0 - 1, c0 - 1, r0 + ph, c0 + pw)
        if _rect_free(occupied, rect):
            _mark_rect(occupied, rect)
            return r0, c0
    return None


def _bed_union_bbox(r0: int, c0: int, direction: str) -> tuple[int, int, int, int]:
    if direction == "up":
        return r0 - 3, c0, r0 + 4, c0 + 4
    if direction == "right":
        return r0, c0, r0 + 4, c0 + 7
    if direction == "down":
        return r0, c0, r0 + 7, c0 + 4
    if direction == "left":
        return r0, c0 - 3, r0 + 4, c0 + 4
    raise AssertionError(f"unknown direction: {direction}")


def _random_bed_position(
    rng: random.Random,
    occupied: np.ndarray,
    direction: str,
) -> tuple[int, int] | None:
    h, w = occupied.shape
    row_range = {
        "up": (4, h - 6),
        "right": (1, h - 6),
        "down": (1, h - 9),
        "left": (1, h - 6),
    }
    col_range = {
        "up": (1, w - 6),
        "right": (1, w - 9),
        "down": (1, w - 6),
        "left": (4, w - 6),
    }
    r_lo, r_hi = row_range[direction]
    c_lo, c_hi = col_range[direction]
    if r_lo > r_hi or c_lo > c_hi:
        return None

    for _ in range(PLACEMENT_TRIES):
        r0 = rng.randint(r_lo, r_hi)
        c0 = rng.randint(c_lo, c_hi)
        reserve = _reserve_rect(_bed_union_bbox(r0, c0, direction))
        if _rect_free(occupied, reserve):
            _mark_rect(occupied, reserve)
            return r0, c0
    return None


def _draw_flowerbed(grid: np.ndarray, r0: int, c0: int, direction: str, seed: int) -> None:
    patch = np.full((5, 5), GREEN, dtype=int)
    patch[0, :] = MAROON
    patch[-1, :] = MAROON
    patch[:, 0] = MAROON
    patch[:, -1] = MAROON
    if direction == "up":
        patch[0, 2] = GREEN
    elif direction == "right":
        patch[2, -1] = GREEN
    elif direction == "down":
        patch[-1, 2] = GREEN
    elif direction == "left":
        patch[2, 0] = GREEN
    else:
        raise AssertionError(f"unknown direction: {direction}")
    patch[2, 2] = seed
    grid[r0 : r0 + 5, c0 : c0 + 5] = patch


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bed_count = rng.randint(4, 6)
    (h_min, h_max), (w_min, w_max) = GRID_RANGES[bed_count]
    h = rng.randint(h_min, h_max)
    w = rng.randint(w_min, w_max)
    grid = np.full((h, w), GREEN, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    templates = _sample_templates(rng)
    legend = _legend_patch(rng, templates)
    legend_pos = _random_rect_position(rng, occupied, legend.shape)
    if legend_pos is None:
        return None
    lr, lc = legend_pos
    grid[lr : lr + legend.shape[0], lc : lc + legend.shape[1]] = legend

    seeds = _balanced_choices(rng, SEED_COLORS, bed_count)
    directions = _balanced_choices(rng, DIRECTIONS, bed_count)
    for seed, direction in zip(seeds, directions):
        pos = _random_bed_position(rng, occupied, direction)
        if pos is None:
            return None
        r0, c0 = pos
        _draw_flowerbed(grid, r0, c0, direction, seed)

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
