import random
from collections import Counter

import numpy as np


ORIENTATIONS = ("ul", "ur", "ll", "lr")
COMPLEMENT = {"ul": "lr", "lr": "ul", "ur": "ll", "ll": "ur"}
PAIR_FAMILIES = (("ul", "lr"), ("ur", "ll"))
PAIR_OFFSETS = ((0, 4), (4, 0), (3, 3), (0, 5), (5, 0))


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _classify_l(comp: list[tuple[int, int]], grid: np.ndarray, color: int) -> dict | None:
    if len(comp) != 3:
        return None
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 != 1 or c1 - c0 != 1:
        return None
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    mask = patch == int(color)
    if int(mask.sum()) != 3:
        return None
    missing = np.argwhere(~mask)
    if missing.shape != (1, 2):
        return None
    mr, mc = map(int, missing[0])
    lookup = {(0, 0): "ul", (0, 1): "ur", (1, 0): "ll", (1, 1): "lr"}
    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "top_left": (int(r0), int(c0)),
        "orient": lookup[(mr, mc)],
    }


def _detect_l_sprites(grid: np.ndarray, bg: int) -> dict[int, list[dict]]:
    sprites_by_color: dict[int, list[dict]] = {}
    for color in sorted(set(map(int, np.unique(grid))) - {int(bg)}):
        sprites: list[dict] = []
        for comp in _components_of_color(grid, color):
            sprite = _classify_l(comp, grid, color)
            if sprite is not None:
                sprites.append(sprite)
        if sprites:
            sprites_by_color[int(color)] = sprites
    return sprites_by_color


def _match_pairs(group_a: list[dict], group_b: list[dict]) -> list[tuple[dict, dict]]:
    edges: list[tuple[int, tuple[int, int], tuple[int, int], int, int]] = []
    for ia, a in enumerate(group_a):
        ar, ac = a["top_left"]
        for ib, b in enumerate(group_b):
            br, bc = b["top_left"]
            dist = abs(int(ar) - int(br)) + abs(int(ac) - int(bc))
            edges.append((int(dist), (min(ar, br), min(ac, bc)), (max(ar, br), max(ac, bc)), ia, ib))
    edges.sort()

    used_a: set[int] = set()
    used_b: set[int] = set()
    pairs: list[tuple[dict, dict]] = []
    for _dist, _lo, _hi, ia, ib in edges:
        if ia in used_a or ib in used_b:
            continue
        used_a.add(int(ia))
        used_b.add(int(ib))
        pairs.append((group_a[ia], group_b[ib]))
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge complementary 2x2-minus-one L sprites into solid 2x2 squares.

    Each foreground L sprite occupies a 2x2 box with exactly one missing corner.
    Diagonally opposite missing corners are complementary: `ul` with `lr`, and
    `ur` with `ll`. Each matched complementary pair is replaced by one solid 2x2
    square at the upper-left of the two original sprite locations; the other
    sprite position is erased to background. Any leftover unpaired L stays as-is.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    out = g.copy()
    sprites_by_color = _detect_l_sprites(g, bg)

    for color, sprites in sprites_by_color.items():
        by_orient = {orient: [] for orient in ORIENTATIONS}
        for sprite in sprites:
            by_orient[sprite["orient"]].append(sprite)
        for orient in by_orient:
            by_orient[orient].sort(key=lambda item: item["top_left"])

        for a_name, b_name in PAIR_FAMILIES:
            pairs = _match_pairs(by_orient[a_name], by_orient[b_name])
            for a, b in pairs:
                if a["top_left"] <= b["top_left"]:
                    keep, move = a, b
                else:
                    keep, move = b, a
                kr0, kc0, kr1, kc1 = keep["bbox"]
                mr0, mc0, mr1, mc1 = move["bbox"]
                out[kr0 : kr1 + 1, kc0 : kc1 + 1] = int(color)
                out[mr0 : mr1 + 1, mc0 : mc1 + 1] = int(bg)

    return out
def _carve_l(grid: np.ndarray, r0: int, c0: int, orient: str, color: int, bg: int) -> None:
    grid[r0 : r0 + 2, c0 : c0 + 2] = int(color)
    if orient == "ul":
        grid[r0, c0] = int(bg)
    elif orient == "ur":
        grid[r0, c0 + 1] = int(bg)
    elif orient == "ll":
        grid[r0 + 1, c0] = int(bg)
    else:
        grid[r0 + 1, c0 + 1] = int(bg)


def _region_is_free(blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int) -> bool:
    h, w = blocked.shape
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return not bool(blocked[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _mark_region(blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int) -> None:
    h, w = blocked.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _place_pair_cluster(
    blocked: np.ndarray,
    h: int,
    w: int,
    rng: random.Random,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    offsets = list(PAIR_OFFSETS)
    rng.shuffle(offsets)
    for dr, dc in offsets:
        rmax = h - max(2, dr + 2)
        cmax = w - max(2, dc + 2)
        if rmax < 0 or cmax < 0:
            continue
        for _ in range(80):
            r0 = rng.randint(0, rmax)
            c0 = rng.randint(0, cmax)
            a = (r0, c0, r0 + 1, c0 + 1)
            b = (r0 + dr, c0 + dc, r0 + dr + 1, c0 + dc + 1)
            u0 = min(a[0], b[0])
            v0 = min(a[1], b[1])
            u1 = max(a[2], b[2])
            v1 = max(a[3], b[3])
            if not _region_is_free(blocked, u0, v0, u1, v1, pad=2):
                continue
            _mark_region(blocked, u0, v0, u1, v1, pad=2)
            return (a[0], a[1]), (b[0], b[1])
    return None


def _place_single(blocked: np.ndarray, h: int, w: int, rng: random.Random) -> tuple[int, int] | None:
    for _ in range(120):
        r0 = rng.randint(0, h - 2)
        c0 = rng.randint(0, w - 2)
        if _region_is_free(blocked, r0, c0, r0 + 1, c0 + 1, pad=2):
            _mark_region(blocked, r0, c0, r0 + 1, c0 + 1, pad=2)
            return int(r0), int(c0)
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(14, 24)
    w = rng.randint(14, 24)
    bg, color = rng.sample(list(range(10)), 2)
    grid = np.full((h, w), int(bg), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    family_pair_counts = [rng.randint(0, 3), rng.randint(0, 3)]
    if sum(family_pair_counts) == 0:
        family_pair_counts[rng.randrange(2)] = rng.randint(1, 3)

    if rng.random() < 0.4 and all(count > 0 for count in family_pair_counts):
        family_pair_counts[rng.randrange(2)] += 1

    leftovers: list[str] = []
    if rng.random() < 0.5:
        family_index = rng.randrange(2)
        family = PAIR_FAMILIES[family_index]
        leftovers.extend([family[rng.randrange(2)]] * rng.randint(1, 2))

    if rng.random() < 0.3:
        family = PAIR_FAMILIES[rng.randrange(2)]
        leftovers.append(family[rng.randrange(2)])

    placements: list[tuple[int, int, str]] = []
    for family_index, count in enumerate(family_pair_counts):
        a_orient, b_orient = PAIR_FAMILIES[family_index]
        for _ in range(count):
            pair_pos = _place_pair_cluster(blocked, h, w, rng)
            if pair_pos is None:
                return None
            (ra, ca), (rb, cb) = pair_pos
            if rng.random() < 0.5:
                placements.append((ra, ca, a_orient))
                placements.append((rb, cb, b_orient))
            else:
                placements.append((ra, ca, b_orient))
                placements.append((rb, cb, a_orient))

    for orient in leftovers:
        pos = _place_single(blocked, h, w, rng)
        if pos is None:
            return None
        r0, c0 = pos
        placements.append((r0, c0, orient))

    if len(placements) < 2:
        return None

    rng.shuffle(placements)
    for r0, c0, orient in placements:
        _carve_l(grid, r0, c0, orient, color, bg)

    sprites = _detect_l_sprites(grid, bg)
    if set(sprites.keys()) != {int(color)}:
        return None
    if len(sprites[color]) != len(placements):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if Counter(map(int, out.ravel())).most_common(1)[0][0] != int(bg):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
