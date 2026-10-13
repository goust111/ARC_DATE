import random
from collections import Counter, deque

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

GEN_RETRIES = 400
REGION_RETRIES = 200
PLACEMENT_RETRIES = 200


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
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(int(r), int(c))])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(pts)
    return comps


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    padded = np.pad(mask, 1, constant_values=False)
    out = np.zeros((h, w), dtype=bool)
    for dr in range(3):
        for dc in range(3):
            out |= padded[dr : dr + h, dc : dc + w]
    return out


def _random_transform(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    transformed = np.rot90(mask, rng.randrange(4))
    if rng.random() < 0.5:
        transformed = np.fliplr(transformed)
    if rng.random() < 0.5:
        transformed = np.flipud(transformed)
    return transformed.copy()


def _is_connected(mask: np.ndarray) -> bool:
    return len(_components(mask)) == 1


def _make_rect_shape(rng: random.Random) -> np.ndarray:
    return np.ones((rng.randint(3, 7), rng.randint(3, 7)), dtype=bool)


def _make_carved_shape(rng: random.Random) -> np.ndarray:
    h = rng.randint(4, 8)
    w = rng.randint(4, 8)
    for _ in range(80):
        mask = np.ones((h, w), dtype=bool)
        for _ in range(rng.randint(1, 3)):
            side = rng.choice(["top", "bottom", "left", "right"])
            if side in {"top", "bottom"}:
                notch_w = rng.randint(1, max(1, w - 1))
                start = rng.randint(0, w - notch_w)
                depth = rng.randint(1, max(1, h - 2))
                rows = slice(0, depth) if side == "top" else slice(h - depth, h)
                mask[rows, start : start + notch_w] = False
            else:
                notch_h = rng.randint(1, max(1, h - 1))
                start = rng.randint(0, h - notch_h)
                depth = rng.randint(1, max(1, w - 2))
                cols = slice(0, depth) if side == "left" else slice(w - depth, w)
                mask[start : start + notch_h, cols] = False
        if int(mask.sum()) >= 6 and _is_connected(mask):
            return _crop_mask(mask)
    return _make_rect_shape(rng)


def _make_poly_shape(rng: random.Random) -> np.ndarray:
    h = rng.randint(4, 8)
    w = rng.randint(4, 8)
    target = rng.randint(max(6, (h * w) // 3), max(6, h * w - 2))
    mask = np.zeros((h, w), dtype=bool)
    start = (rng.randrange(h), rng.randrange(w))
    mask[start] = True
    frontier: set[tuple[int, int]] = set()
    for nr, nc in _neighbors4(*start):
        if 0 <= nr < h and 0 <= nc < w:
            frontier.add((int(nr), int(nc)))

    while int(mask.sum()) < target and frontier:
        frontier_list = list(frontier)
        weights = []
        for r, c in frontier_list:
            adj = sum(
                1
                for nr, nc in _neighbors4(r, c)
                if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc])
            )
            edge_bonus = 1 if 0 < r < h - 1 and 0 < c < w - 1 else 0
            weights.append(adj * 3 + edge_bonus + 1)
        choice = rng.choices(frontier_list, weights=weights, k=1)[0]
        frontier.remove(choice)
        mask[choice] = True
        for nr, nc in _neighbors4(*choice):
            if 0 <= nr < h and 0 <= nc < w and not bool(mask[nr, nc]):
                frontier.add((int(nr), int(nc)))

    for _ in range(2):
        to_add = []
        for r in range(h):
            for c in range(w):
                if bool(mask[r, c]):
                    continue
                adj = sum(
                    1
                    for nr, nc in _neighbors4(r, c)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc])
                )
                if adj >= 3 or (adj == 2 and rng.random() < 0.35):
                    to_add.append((int(r), int(c)))
        for r, c in to_add:
            mask[r, c] = True

    if int(mask.sum()) < 6 or not _is_connected(mask):
        return _make_carved_shape(rng)
    return _crop_mask(mask)


def _make_shape(rng: random.Random) -> np.ndarray:
    style = rng.choice(["rect", "carved", "poly"])
    if style == "rect":
        mask = _make_rect_shape(rng)
    elif style == "carved":
        mask = _make_carved_shape(rng)
    else:
        mask = _make_poly_shape(rng)
    return _random_transform(mask, rng)


def _sample_votes(
    interior_mask: np.ndarray, rng: random.Random, used_majorities: set[int]
) -> tuple[int, dict[tuple[int, int], int]] | None:
    area = int(interior_mask.sum())
    num_colors = rng.choice([2, 3])
    if area < num_colors + 2:
        return None

    majority_choices = [color for color in PALETTE if color not in used_majorities]
    if not majority_choices:
        majority_choices = PALETTE[:]
    majority = int(rng.choice(majority_choices))
    other_colors = [color for color in PALETTE if color != majority]
    colors = [majority] + rng.sample(other_colors, num_colors - 1)

    for _ in range(60):
        majority_count = rng.randint(2, area - num_colors)
        minority_counts = [rng.randint(1, majority_count - 1) for _ in range(num_colors - 1)]
        if majority_count + sum(minority_counts) <= area - 1:
            break
    else:
        return None

    cells = [tuple(map(int, cell)) for cell in np.argwhere(interior_mask)]
    rng.shuffle(cells)
    votes: dict[tuple[int, int], int] = {}
    idx = 0
    for color, count in zip(colors, [majority_count] + minority_counts):
        for _ in range(count):
            votes[cells[idx]] = int(color)
            idx += 1

    return majority, votes


def _make_region(rng: random.Random, used_majorities: set[int]) -> dict[str, object]:
    for _ in range(REGION_RETRIES):
        interior = _make_shape(rng)
        if int(interior.sum()) < 8:
            continue

        padded_interior = np.pad(interior, 1, constant_values=False)
        boundary = _dilate8(padded_interior) & ~padded_interior
        votes_info = _sample_votes(padded_interior, rng, used_majorities)
        if votes_info is None or not bool(boundary.any()):
            continue

        majority, votes = votes_info
        patch = np.zeros(padded_interior.shape, dtype=int)
        patch[boundary] = GREY
        for (r, c), color in votes.items():
            patch[r, c] = int(color)

        return {
            "patch": patch,
            "area": int(padded_interior.sum()),
            "majority": int(majority),
        }
    raise RuntimeError("failed to generate a valid region")


def _place_regions(rng: random.Random, regions: list[dict[str, object]]) -> np.ndarray | None:
    max_h = max(int(region["patch"].shape[0]) for region in regions)
    max_w = max(int(region["patch"].shape[1]) for region in regions)
    n = len(regions)
    min_h = max(16, max_h + 4, 4 * n + 6)
    min_w = max(16, max_w + 4, 4 * n + 6)
    h = rng.randint(min_h, min(32, min_h + 8))
    w = rng.randint(min_w, min(32, min_w + 8))

    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    order = regions[:]
    rng.shuffle(order)

    for region in order:
        patch = np.asarray(region["patch"], dtype=int)
        ph, pw = patch.shape
        for _ in range(PLACEMENT_RETRIES):
            r0 = rng.randint(1, h - ph - 1)
            c0 = rng.randint(1, w - pw - 1)
            rr0 = r0 - 1
            cc0 = c0 - 1
            rr1 = r0 + ph + 1
            cc1 = c0 + pw + 1
            if bool(blocked[rr0:rr1, cc0:cc1].any()):
                continue
            grid[r0 : r0 + ph, c0 : c0 + pw][patch != 0] = patch[patch != 0]
            blocked[r0 : r0 + ph, c0 : c0 + pw] = True
            break
        else:
            return None

    return grid


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    num_regions = rng.randint(2, 5)
    regions: list[dict[str, object]] = []
    used_areas: set[int] = set()
    used_majorities: set[int] = set()

    for _ in range(num_regions):
        for _ in range(REGION_RETRIES):
            region = _make_region(rng, used_majorities)
            area = int(region["area"])
            majority = int(region["majority"])
            if area in used_areas:
                continue
            used_areas.add(area)
            used_majorities.add(majority)
            regions.append(region)
            break
        else:
            return None

    return _place_regions(rng, regions)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pick each enclosed region's majority vote color, sort regions by interior area,
    and redraw them as tightly nested concentric square frames.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    h, w = g.shape
    regions: list[tuple[int, int]] = []
    for comp in _components(g != GREY):
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp):
            continue

        comp_set = set(comp)
        if not any(
            0 <= nr < h and 0 <= nc < w and (nr, nc) not in comp_set and int(g[nr, nc]) == GREY
            for r, c in comp
            for nr, nc in _neighbors4(r, c)
        ):
            continue

        votes = [int(g[r, c]) for r, c in comp if int(g[r, c]) != BLACK]
        if not votes:
            continue
        majority = Counter(votes).most_common(1)[0][0]
        regions.append((len(comp), int(majority)))

    if not regions:
        raise ValueError("no enclosed regions found")

    areas = [area for area, _ in regions]
    if len(set(areas)) != len(areas):
        raise ValueError("area ordering is ambiguous")

    regions.sort(key=lambda item: item[0], reverse=True)
    side = 2 * len(regions) - 1
    out = np.zeros((side, side), dtype=int)
    for layer, (_, color) in enumerate(regions):
        lo = layer
        hi = side - layer
        out[lo:hi, lo] = color
        out[lo:hi, hi - 1] = color
        out[lo, lo:hi] = color
        out[hi - 1, lo:hi] = color
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
