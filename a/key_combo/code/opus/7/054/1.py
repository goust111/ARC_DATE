from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np


BG = 0
BASELINE = 5
INPUT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_ATTEMPTS = 300

# Each tuple is a bottom-packed column-height profile.
# Gravity preserves these shapes because every occupied column is already
# a contiguous suffix ending at the component's bottom row.
SHAPE_PROFILES = (
    (3,),
    (1, 2),
    (2, 1),
    (2, 2),
    (1, 3),
    (3, 1),
    (2, 3),
    (3, 2),
    (3, 3),
)

# 2-3 non-singleton families, plus an optional singleton.
GROUP_PATTERNS = (
    (2, 2, 2),
    (2, 2, 2, 1),
    (2, 2, 3),
    (2, 2, 3, 1),
    (2, 2, 4),
    (2, 3, 3),
    (2, 4),
    (2, 4, 1),
    (3, 3),
    (3, 3, 1),
    (3, 4),
    (4, 4),
)


def _profile_shape(profile: tuple[int, ...]) -> tuple[int, int]:
    return max(profile), len(profile)


def _profile_coords(profile: tuple[int, ...]) -> np.ndarray:
    h, _w = _profile_shape(profile)
    coords: list[tuple[int, int]] = []
    for c, col_height in enumerate(profile):
        for r in range(h - col_height, h):
            coords.append((r, c))
    return np.array(coords, dtype=int)


def _paint_profile(
    grid: np.ndarray,
    profile: tuple[int, ...],
    r0: int,
    c0: int,
    color: int,
) -> None:
    coords = _profile_coords(profile)
    rr = coords[:, 0] + int(r0)
    cc = coords[:, 1] + int(c0)
    grid[rr, cc] = int(color)


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _shape_signature(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    shifted = np.asarray(coords, dtype=int) - np.asarray(coords, dtype=int).min(axis=0, keepdims=True)
    return tuple(sorted((int(r), int(c)) for r, c in shifted))


def _apply_gravity(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = np.full_like(g, BG)
    out[-1, :] = BASELINE

    for c in range(w):
        column_pixels = [
            int(g[r, c])
            for r in range(h - 1)
            if int(g[r, c]) not in (BG, BASELINE)
        ]
        for offset, color in enumerate(reversed(column_pixels)):
            out[h - 1 - offset, c] = color
    return out


def _family_output_color(size: int) -> int | None:
    if size == 1:
        return None
    if size == 2:
        return 2
    if size == 3:
        return 3
    return 1


def main(grid: np.ndarray) -> np.ndarray:
    """Drop the sprites to the baseline, group the settled components by normalized shape, then recolor or erase each family by its size.

    Input structure:
    - Black (0) is the background and the bottom row is a uniform grey (5) baseline.
    - Several small 4-connected monochrome sprites are scattered in the upper half.
    - Different sprites occupy disjoint column sets, so gravity does not merge them.

    Rule:
    - Apply column-wise gravity to every non-background, non-baseline pixel; the grey baseline is passable landing space and occupied baseline cells take the falling pixel color.
    - On the settled grid, find all 4-connected non-black, non-grey components and normalize each shape by translating its bbox to the origin.
    - Recolor every family of identical normalized shapes by family size: size 2 -> red (2), size 3 -> green (3), size 4 or more -> blue (1); erase singleton families to black (0).

    Key insight:
    - The decisive signal is the frequency of each translation-normalized shape after gravity, not the original sprite colors or positions.
    """
    settled = _apply_gravity(np.asarray(grid, dtype=int))
    fg_mask = (settled != BG) & (settled != BASELINE)
    components = _find_components(fg_mask)

    families: dict[tuple[tuple[int, int], ...], list[np.ndarray]] = defaultdict(list)
    for comp in components:
        families[_shape_signature(comp)].append(comp)

    out = settled.copy()
    for family_components in families.values():
        color = _family_output_color(len(family_components))
        for comp in family_components:
            rr, cc = comp.T
            out[rr, cc] = BG if color is None else int(color)
    return out


def _build_objects(rng: np.random.Generator) -> tuple[list[dict[str, int | tuple[int, ...]]], tuple[int, ...]]:
    pattern = GROUP_PATTERNS[int(rng.integers(0, len(GROUP_PATTERNS)))]
    shape_ids = rng.permutation(len(SHAPE_PROFILES))[: len(pattern)]
    family_defs = [(SHAPE_PROFILES[int(idx)], int(size)) for idx, size in zip(shape_ids, pattern)]
    rng.shuffle(family_defs)

    colors = list(rng.permutation(INPUT_COLORS)[: sum(pattern)])
    color_idx = 0
    objects: list[dict[str, int | tuple[int, ...]]] = []
    for profile, family_size in family_defs:
        for _ in range(family_size):
            objects.append(
                {
                    "profile": profile,
                    "family_size": family_size,
                    "color": int(colors[color_idx]),
                }
            )
            color_idx += 1
    rng.shuffle(objects)
    return objects, pattern


def _assign_columns(rng: np.random.Generator, widths: list[int], total_width: int) -> list[int]:
    required = sum(widths) + len(widths) - 1
    extra = total_width - required
    slots = rng.multinomial(extra, np.full(len(widths) + 1, 1.0 / (len(widths) + 1)))
    cols: list[int] = []
    c = int(slots[0])
    for i, width in enumerate(widths):
        cols.append(c)
        c += int(width)
        if i < len(widths) - 1:
            c += 1 + int(slots[i + 1])
    return cols


def _build_expected(h: int, w: int, objects: list[dict[str, int | tuple[int, ...]]]) -> np.ndarray:
    settled = np.full((h, w), BG, dtype=int)
    settled[-1, :] = BASELINE
    for obj in objects:
        profile = obj["profile"]
        assert isinstance(profile, tuple)
        ph, _pw = _profile_shape(profile)
        _paint_profile(settled, profile, h - ph, int(obj["c0"]), int(obj["color"]))

    expected = settled.copy()
    for obj in objects:
        profile = obj["profile"]
        assert isinstance(profile, tuple)
        family_size = int(obj["family_size"])
        ph, _pw = _profile_shape(profile)
        coords = _profile_coords(profile)
        rr = coords[:, 0] + (h - ph)
        cc = coords[:, 1] + int(obj["c0"])
        color = _family_output_color(family_size)
        expected[rr, cc] = BG if color is None else int(color)
    return expected


def _validate_instance(
    grid: np.ndarray,
    expected: np.ndarray,
    objects: list[dict[str, int | tuple[int, ...]]],
    pattern: tuple[int, ...],
) -> bool:
    if grid.ndim != 2 or expected.ndim != 2 or grid.shape != expected.shape:
        return False
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    if not np.all(grid[-1, :] == BASELINE):
        return False

    input_components = _find_components((grid != BG) & (grid != BASELINE))
    if len(input_components) != len(objects):
        return False

    settled = _apply_gravity(grid)
    settled_components = _find_components((settled != BG) & (settled != BASELINE))
    if len(settled_components) != len(objects):
        return False

    family_sizes = sorted(Counter(_shape_signature(comp) for comp in settled_components).values())
    if family_sizes != sorted(pattern):
        return False

    non_singletons = sum(1 for size in family_sizes if size >= 2)
    if not (2 <= non_singletons <= 3):
        return False

    return np.array_equal(main(grid), expected)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    objects, pattern = _build_objects(rng)
    widths = [_profile_shape(obj["profile"])[1] for obj in objects]

    required_width = sum(widths) + len(widths) - 1
    min_width = max(12, required_width)
    max_width = min(24, required_width + 6, 30)
    if min_width > max_width:
        return None

    w = int(rng.integers(min_width, max_width + 1))
    h = int(rng.integers(12, 18))
    cols = _assign_columns(rng, widths, w)

    grid = np.full((h, w), BG, dtype=int)
    grid[-1, :] = BASELINE

    for obj, c0 in zip(objects, cols):
        profile = obj["profile"]
        assert isinstance(profile, tuple)
        ph, _pw = _profile_shape(profile)
        max_top = min((h // 2) - 1, h - ph - 3)
        if max_top < 0:
            return None
        r0 = int(rng.integers(0, max_top + 1))
        obj["c0"] = int(c0)
        obj["r0"] = r0
        _paint_profile(grid, profile, r0, int(c0), int(obj["color"]))

    expected = _build_expected(h, w, objects)
    if not _validate_instance(grid, expected, objects, pattern):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 1")
