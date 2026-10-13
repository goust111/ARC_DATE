import random
from typing import Iterable

import numpy as np


BLUE = 1
RED = 2
GREY = 5
ORANGE = 7

SIZE_CHOICES = (9, 11, 13, 15)
MAX_GENERATION_ATTEMPTS = 500


def _dedupe_coords(coords: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    ordered: list[tuple[int, int]] = []
    for coord in coords:
        if coord not in seen:
            seen.add(coord)
            ordered.append(coord)
    return ordered


def _mirror_refs(shape: tuple[int, int], r: int, c: int) -> list[tuple[str, tuple[int, int]]]:
    h, w = shape
    refs = [
        ("h", (r, w - 1 - c)),
        ("v", (h - 1 - r, c)),
        ("rot", (h - 1 - r, w - 1 - c)),
    ]
    out: list[tuple[str, tuple[int, int]]] = []
    seen: set[tuple[int, int]] = set()
    for name, coord in refs:
        if coord == (r, c) or coord in seen:
            continue
        seen.add(coord)
        out.append((name, coord))
    return out


def _orbit(shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]]:
    h, w = shape
    return _dedupe_coords(
        [
            (r, c),
            (r, w - 1 - c),
            (h - 1 - r, c),
            (h - 1 - r, w - 1 - c),
        ]
    )


def _reference_values(
    grid: np.ndarray,
    r: int,
    c: int,
    *,
    blocked: set[int],
) -> list[tuple[str, tuple[int, int], int]]:
    refs: list[tuple[str, tuple[int, int], int]] = []
    for name, (rr, cc) in _mirror_refs(grid.shape, r, c):
        value = int(grid[rr, cc])
        if value in blocked:
            continue
        refs.append((name, (rr, cc), value))
    return refs


def _resolve_grey(grid: np.ndarray, r: int, c: int) -> int:
    refs = _reference_values(grid, r, c, blocked={GREY, ORANGE})
    if not refs:
        return 0
    values = [value for _name, _coord, value in refs]
    if len(set(values)) == 1:
        return values[0]
    return refs[0][2]


def _resolve_orange(grid: np.ndarray, r: int, c: int) -> int:
    refs = _reference_values(grid, r, c, blocked={GREY, ORANGE, RED})
    if refs and len({value for _name, _coord, value in refs}) == 1:
        return refs[0][2]
    return RED


def main(grid: np.ndarray) -> np.ndarray:
    """Complete recoverable mirror damage in a near-bilateral pattern and expose irrecoverable defects in red.

    Input structure:
    - The grid is an odd-sized near-binary pattern organized around its central horizontal and vertical axes.
    - Ordinary content uses black `0` and blue `1`.
    - Grey `5` cells are masked pixels, and orange `7` cells are suspicious polluted pixels.

    Rule:
    - For each grey cell, inspect its horizontal, vertical, and 180-degree mirror mates.
    - If all usable mates agree, copy that value; if usable mates disagree, use the first available one in priority order: horizontal, then vertical, then 180-degree rotation.
    - After grey recovery, inspect every orange cell: if its usable mirror mates now all agree, restore that value; otherwise mark the orange cell red `2`.
    - No grey or orange remains in the output.

    Key insight:
    - Grey marks a recoverable occlusion, while orange marks a candidate defect; the two-axis mirror orbit determines whether a damaged pixel can be restored or must instead be highlighted as a true inconsistency.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    grey_positions = [tuple(map(int, coord)) for coord in np.argwhere(g == GREY)]
    for r, c in grey_positions:
        out[r, c] = _resolve_grey(out, r, c)

    post_grey = out.copy()
    orange_positions = [tuple(map(int, coord)) for coord in np.argwhere(g == ORANGE)]
    for r, c in orange_positions:
        out[r, c] = _resolve_orange(post_grey, r, c)

    return out


def _mirror_seed(seed: np.ndarray) -> np.ndarray:
    m = int(seed.shape[0])
    side = 2 * m - 1
    full = np.zeros((side, side), dtype=int)
    for r in range(side):
        rr = min(r, side - 1 - r)
        for c in range(side):
            cc = min(c, side - 1 - c)
            full[r, c] = int(seed[rr, cc])
    return full


def _seed_attempt(side: int) -> np.ndarray | None:
    m = side // 2 + 1
    seed = np.zeros((m, m), dtype=int)

    for _ in range(random.randint(4, 7)):
        kind = random.choices(("rect", "hline", "vline", "point"), weights=(5, 2, 2, 1))[0]
        if kind == "rect":
            h = random.randint(1, max(1, m - 1))
            w = random.randint(1, max(1, m - 1))
            r0 = random.randint(0, m - h)
            c0 = random.randint(0, m - w)
            seed[r0 : r0 + h, c0 : c0 + w] = BLUE
        elif kind == "hline":
            r = random.randrange(m)
            c0 = random.randrange(m)
            c1 = random.randint(c0, m - 1)
            seed[r, c0 : c1 + 1] = BLUE
        elif kind == "vline":
            c = random.randrange(m)
            r0 = random.randrange(m)
            r1 = random.randint(r0, m - 1)
            seed[r0 : r1 + 1, c] = BLUE
        else:
            seed[random.randrange(m), random.randrange(m)] = BLUE

    for _ in range(random.randint(1, 3)):
        h = random.randint(1, max(1, m // 2))
        w = random.randint(1, max(1, m // 2))
        r0 = random.randint(0, m - h)
        c0 = random.randint(0, m - w)
        seed[r0 : r0 + h, c0 : c0 + w] = 0

    toggle_mask = np.random.random((m, m)) < 0.08
    seed[toggle_mask] = 1 - seed[toggle_mask]

    full = _mirror_seed(seed)
    blue_count = int(np.sum(full == BLUE))
    if blue_count == 0 or blue_count == full.size:
        return None
    density = blue_count / float(full.size)
    if not (0.18 <= density <= 0.72):
        return None

    transitions = int(np.sum(full[:, 1:] != full[:, :-1]) + np.sum(full[1:, :] != full[:-1, :]))
    if transitions < side:
        return None
    return full


def _build_base_pattern() -> np.ndarray:
    for _ in range(200):
        side = random.choice(SIZE_CHOICES)
        base = _seed_attempt(side)
        if base is not None:
            return base
    raise RuntimeError("failed to build a non-degenerate symmetric base pattern")


def _orbit_catalog(base: np.ndarray) -> list[dict]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    out: list[dict] = []
    for r in range(base.shape[0]):
        for c in range(base.shape[1]):
            coords = tuple(sorted(_orbit(base.shape, r, c)))
            if coords in seen:
                continue
            seen.add(coords)
            out.append(
                {
                    "key": coords,
                    "coords": list(coords),
                    "size": len(coords),
                    "value": int(base[coords[0]]),
                }
            )
    return out


def _pick_orbit(candidates: list[dict], used_keys: set[tuple[tuple[int, int], ...]]) -> dict | None:
    available = [orbit for orbit in candidates if orbit["key"] not in used_keys]
    if not available:
        return None
    return random.choice(available)


def _plan_recoverable_gray(
    grid: np.ndarray,
    base: np.ndarray,
    orbit: dict,
    replacements: list[tuple[int, int, int, str]],
) -> None:
    r, c = random.choice(orbit["coords"])
    grid[r, c] = GREY
    replacements.append((r, c, int(base[r, c]), "grey_simple"))


def _plan_conflict_gray(
    grid: np.ndarray,
    base: np.ndarray,
    orbit: dict,
    replacements: list[tuple[int, int, int, str]],
) -> None:
    r, c = random.choice(orbit["coords"])
    refs = _mirror_refs(grid.shape, r, c)
    assert len(refs) == 3
    keep_h = refs[0][1]
    dirty_coords = [coord for _name, coord in refs[1:]]

    grid[r, c] = GREY
    flip_coord = random.choice(dirty_coords)
    grid[flip_coord] = 1 - int(base[flip_coord])
    if len(dirty_coords) == 2 and random.random() < 0.35:
        other = dirty_coords[0] if dirty_coords[1] == flip_coord else dirty_coords[1]
        grid[other] = 1 - int(base[other])

    assert int(grid[keep_h]) == int(base[keep_h])
    replacements.append((r, c, int(base[r, c]), "grey_conflict"))


def _plan_recoverable_orange(
    grid: np.ndarray,
    base: np.ndarray,
    orbit: dict,
    replacements: list[tuple[int, int, int, str]],
) -> None:
    r, c = random.choice(orbit["coords"])
    grid[r, c] = ORANGE
    replacements.append((r, c, int(base[r, c]), "orange_recover"))


def _plan_defect_orange(
    grid: np.ndarray,
    base: np.ndarray,
    orbit: dict,
    replacements: list[tuple[int, int, int, str]],
) -> None:
    r, c = random.choice(orbit["coords"])
    refs = _mirror_refs(grid.shape, r, c)
    assert len(refs) == 3

    grid[r, c] = ORANGE
    flip_coord = random.choice([coord for _name, coord in refs])
    grid[flip_coord] = 1 - int(base[flip_coord])
    replacements.append((r, c, RED, "orange_defect"))


def _expected_output(
    grid: np.ndarray,
    replacements: list[tuple[int, int, int, str]],
) -> tuple[np.ndarray, np.ndarray]:
    post_grey = grid.copy()
    final = grid.copy()
    for r, c, value, kind in replacements:
        if kind.startswith("grey_"):
            post_grey[r, c] = value
            final[r, c] = value
    for r, c, value, kind in replacements:
        if kind.startswith("orange_"):
            final[r, c] = value
    return post_grey, final


def _validate_generated_case(
    base: np.ndarray,
    grid: np.ndarray,
    replacements: list[tuple[int, int, int, str]],
) -> bool:
    if grid.ndim != 2 or grid.shape[0] != grid.shape[1] or grid.shape[0] % 2 != 1:
        return False
    if max(grid.shape) > 30:
        return False
    if not set(np.unique(grid)).issubset({0, BLUE, GREY, ORANGE}):
        return False

    grey_count = int(np.sum(grid == GREY))
    orange_count = int(np.sum(grid == ORANGE))
    if not (2 <= grey_count <= 4 and 1 <= orange_count <= 2):
        return False

    kinds = [kind for _r, _c, _value, kind in replacements]
    if "grey_conflict" not in kinds:
        return False
    if "grey_simple" not in kinds:
        return False
    if not any(kind.startswith("orange_") for kind in kinds):
        return False

    post_grey, expected = _expected_output(grid, replacements)
    if GREY in expected or ORANGE in expected:
        return False
    if BLUE not in expected or 0 not in expected:
        return False

    for r, c, value, kind in replacements:
        if kind.startswith("grey_"):
            refs = _reference_values(grid, r, c, blocked={GREY, ORANGE})
            if not refs:
                return False
            if kind == "grey_conflict" and len({v for _n, _coord, v in refs}) < 2:
                return False
            if kind == "grey_simple" and value not in {0, BLUE}:
                return False
        elif kind == "orange_recover":
            refs = _reference_values(post_grey, r, c, blocked={GREY, ORANGE, RED})
            if not refs or len({v for _n, _coord, v in refs}) != 1:
                return False
        elif kind == "orange_defect":
            refs = _reference_values(post_grey, r, c, blocked={GREY, ORANGE, RED})
            if refs and len({v for _n, _coord, v in refs}) == 1:
                return False

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(actual, grid):
        return False

    red_count = int(np.sum(actual == RED))
    if "orange_defect" in kinds and red_count == 0:
        return False
    if "orange_defect" not in kinds and red_count != 0:
        return False

    # The hidden binary flips make mirror recovery depend on the orbit, not just on a global recolor.
    if np.array_equal(base, actual):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    base = _build_base_pattern()
    grid = base.copy()
    orbits = _orbit_catalog(base)
    general_orbits = [orbit for orbit in orbits if orbit["size"] == 4]
    axis_orbits = [orbit for orbit in orbits if orbit["size"] == 2]
    if len(general_orbits) < 4 or len(axis_orbits) < 1:
        return None

    used_keys: set[tuple[tuple[int, int], ...]] = set()
    replacements: list[tuple[int, int, int, str]] = []

    grey_count = random.randint(2, 4)
    orange_count = random.choice((1, 2, 2))

    simple_orbit_pool = axis_orbits if random.random() < 0.55 else orbits
    orbit = _pick_orbit(simple_orbit_pool, used_keys)
    if orbit is None:
        return None
    used_keys.add(orbit["key"])
    _plan_recoverable_gray(grid, base, orbit, replacements)

    orbit = _pick_orbit(general_orbits, used_keys)
    if orbit is None:
        return None
    used_keys.add(orbit["key"])
    _plan_conflict_gray(grid, base, orbit, replacements)

    for _ in range(grey_count - 2):
        orbit = _pick_orbit(orbits, used_keys)
        if orbit is None:
            return None
        used_keys.add(orbit["key"])
        _plan_recoverable_gray(grid, base, orbit, replacements)

    orange_kinds = ["orange_defect", "orange_recover"] if orange_count == 2 else [random.choice(("orange_defect", "orange_recover"))]
    random.shuffle(orange_kinds)
    for orange_kind in orange_kinds:
        pool = general_orbits if orange_kind == "orange_defect" else (axis_orbits if random.random() < 0.35 else orbits)
        orbit = _pick_orbit(pool, used_keys)
        if orbit is None:
            return None
        used_keys.add(orbit["key"])
        if orange_kind == "orange_defect":
            _plan_defect_orange(grid, base, orbit, replacements)
        else:
            _plan_recoverable_orange(grid, base, orbit, replacements)

    if _validate_generated_case(base, grid, replacements):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-defect ARC input")
