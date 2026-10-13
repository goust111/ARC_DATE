from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
MARKER = 2
GREEN = 3
YELLOW = 4
WALL = 5
PINK = 6

SEED_COLORS = (BLUE, GREEN, YELLOW, PINK)
SEED_OFFSETS = ((0, 0), (0, 1), (1, 0))

_SPARSE_PATTERNS = (
    ((1, 0), (0, 0)),
    ((0, 1), (0, 0)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1)),
)
_DENSE_PATTERNS = (
    ((0, 1), (1, 1)),
    ((1, 0), (1, 1)),
    ((1, 1), (0, 1)),
    ((1, 1), (1, 0)),
)
_ADJACENT_TWO_PATTERNS = (
    ((1, 1), (0, 0)),
    ((0, 0), (1, 1)),
    ((1, 0), (1, 0)),
    ((0, 1), (0, 1)),
)
_DIAGONAL_TWO_PATTERNS = (
    ((1, 0), (0, 1)),
    ((0, 1), (1, 0)),
)
_ALL_PATTERNS = _SPARSE_PATTERNS + _DENSE_PATTERNS + _ADJACENT_TWO_PATTERNS + _DIAGONAL_TWO_PATTERNS

GEN_ATTEMPTS = 400
CHAMBER_SIZE_RANGE = (5, 7)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tile(pattern: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    ph, pw = pattern.shape
    rows = np.arange(h)[:, None] % ph
    cols = np.arange(w)[None, :] % pw
    return pattern[rows, cols]


def _expand_seed(seed_mask: np.ndarray, primary: int) -> np.ndarray:
    seed_mask = np.asarray(seed_mask, dtype=bool)
    sh, sw = seed_mask.shape
    rows = np.arange(sh * sh)[:, None]
    cols = np.arange(sw * sw)[None, :]
    inner = seed_mask[rows % sh, cols % sw]
    meta = seed_mask[(rows // sh) % sh, (cols // sw) % sw]
    out = np.full((sh * sh, sw * sw), BG, dtype=int)
    out[inner & meta] = int(primary)
    out[inner & ~meta] = MARKER
    return out


def _analyze_chamber(chamber: np.ndarray) -> dict:
    colors = sorted(int(v) for v in np.unique(chamber) if int(v) not in (BG, MARKER))
    assert len(colors) == 1, "each chamber must contain exactly one non-special seed color"
    primary = colors[0]

    fg_coords = np.argwhere(chamber == primary)
    assert 1 <= len(fg_coords) <= 3, "seed must use 1-3 foreground cells"

    marker_coords = np.argwhere(chamber == MARKER)
    assert len(marker_coords) in (0, 1), "each chamber may contain at most one marker"
    if len(marker_coords) == 1:
        mr, mc = map(int, marker_coords[0])
        assert 0 < mr < chamber.shape[0] - 1 and 0 < mc < chamber.shape[1] - 1, "marker must not touch the wall"

    candidates: list[tuple[int, int, np.ndarray]] = []
    for sr in range(min(2, chamber.shape[0] - 1)):
        for sc in range(min(2, chamber.shape[1] - 1)):
            seed = chamber[sr : sr + 2, sc : sc + 2]
            if seed.shape != (2, 2):
                continue
            if not np.all((seed == BG) | (seed == primary)):
                continue
            seed_mask = seed == primary
            if not (np.any(seed_mask) and np.any(~seed_mask)):
                continue
            if not np.all(
                (fg_coords[:, 0] >= sr)
                & (fg_coords[:, 0] <= sr + 1)
                & (fg_coords[:, 1] >= sc)
                & (fg_coords[:, 1] <= sc + 1)
            ):
                continue
            if len(marker_coords) == 1:
                mr, mc = map(int, marker_coords[0])
                if sr - 1 <= mr <= sr + 2 and sc - 1 <= mc <= sc + 2:
                    continue
            candidates.append((int(sr), int(sc), seed_mask.copy()))

    assert len(candidates) == 1, "seed window must be uniquely recoverable near the chamber corner"
    sr0, sc0, seed_mask = candidates[0]

    return {
        "primary": primary,
        "seed_mask": seed_mask,
        "seed_offset": (int(sr0), int(sc0)),
        "marked": len(marker_coords) == 1,
    }


def _chamber_components(grid: np.ndarray) -> list[dict]:
    comps = _components(grid != WALL)
    assert len(comps) == 4, "expected exactly four chambers"

    chambers: list[dict] = []
    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        local[coords[:, 0] - r0, coords[:, 1] - c0] = True
        assert np.all(local), "chambers must be rectangles"
        chambers.append({"bbox": (r0, c0, r1, c1)})
    chambers.sort(key=lambda item: item["bbox"])
    return chambers


def main(grid: np.ndarray) -> np.ndarray:
    """Apply recursive seed expansion in marked chambers and solid flood-fill in unmarked chambers.

    Input structure:
    - A black background is partitioned by gray wall lines into four rectangular chambers.
    - Each chamber contains exactly one 2x2 seed made from one foreground color plus black cells, placed near the chamber's upper-left corner.
    - Exactly two chambers also contain one interior red marker pixel outside the seed.

    Rule:
    - Treat gray as an immovable wall and split the non-wall area into four chambers.
    - In each chamber, recover the seed's binary mask from the chamber's unique non-red foreground color.
    - If the chamber has a red marker, expand the 2x2 seed one level: foreground seed cells produce primary-colored copies of the seed mask, background seed cells produce red copies, then tile/crop that 4x4 result from the chamber's upper-left corner across the chamber.
    - If the chamber has no red marker, fill the entire chamber with the seed's foreground color.

    Key insight:
    - The red marker is both the branch selector and the secondary color source, so detecting marker presence is necessary before the chamber's seed can be interpreted correctly.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for chamber in _chamber_components(g):
        r0, c0, r1, c1 = chamber["bbox"]
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        info = _analyze_chamber(patch)
        if info["marked"]:
            expanded = _expand_seed(info["seed_mask"], info["primary"])
            filled = _tile(expanded, patch.shape)
        else:
            filled = np.full(patch.shape, info["primary"], dtype=int)
        out[r0 : r1 + 1, c0 : c1 + 1] = filled

    return out


def _pattern_to_mask(pattern: tuple[tuple[int, int], tuple[int, int]]) -> np.ndarray:
    return np.array(pattern, dtype=bool)


def _marker_candidates(height: int, width: int, seed_offset: tuple[int, int]) -> list[tuple[int, int]]:
    sr, sc = seed_offset
    candidates: list[tuple[int, int]] = []
    for r in range(1, height - 1):
        for c in range(1, width - 1):
            if sr - 1 <= r <= sr + 2 and sc - 1 <= c <= sc + 2:
                continue
            candidates.append((r, c))
    return candidates


def _sample_seed_masks() -> list[np.ndarray]:
    chosen = [
        random.choice(_SPARSE_PATTERNS),
        random.choice(_DENSE_PATTERNS),
        random.choice(_ADJACENT_TWO_PATTERNS),
    ]
    remaining = [p for p in _ALL_PATTERNS if p not in chosen]
    if random.random() < 0.6:
        diagonal_remaining = [p for p in _DIAGONAL_TWO_PATTERNS if p not in chosen]
        chosen.append(random.choice(diagonal_remaining or remaining))
    else:
        chosen.append(random.choice(remaining))
    random.shuffle(chosen)
    return [_pattern_to_mask(pattern) for pattern in chosen]


def _chamber_layout(top_h: int, bottom_h: int, left_w: int, right_w: int) -> list[tuple[int, int, int, int]]:
    return [
        (1, 1, top_h, left_w),
        (1, left_w + 2, top_h, left_w + right_w + 1),
        (top_h + 2, 1, top_h + bottom_h + 1, left_w),
        (top_h + 2, left_w + 2, top_h + bottom_h + 1, left_w + right_w + 1),
    ]


def _render_input(
    top_h: int,
    bottom_h: int,
    left_w: int,
    right_w: int,
    chamber_specs: list[dict],
) -> np.ndarray:
    h = top_h + bottom_h + 3
    w = left_w + right_w + 3
    grid = np.full((h, w), WALL, dtype=int)

    for spec, (r0, c0, r1, c1) in zip(chamber_specs, _chamber_layout(top_h, bottom_h, left_w, right_w)):
        grid[r0 : r1 + 1, c0 : c1 + 1] = BG

        sr, sc = spec["seed_offset"]
        chamber = grid[r0 : r1 + 1, c0 : c1 + 1]
        chamber[sr : sr + 2, sc : sc + 2][spec["seed_mask"]] = spec["primary"]

        if spec["marked"]:
            mr, mc = spec["marker"]
            chamber[mr, mc] = MARKER

    return grid


def _validate_scene(inp: np.ndarray, out: np.ndarray) -> bool:
    if inp.ndim != 2 or out.ndim != 2 or inp.shape != out.shape:
        return False
    if np.any(inp == WALL) and not np.array_equal(inp == WALL, out == WALL):
        return False

    chambers = _chamber_components(inp)
    if len(chambers) != 4:
        return False

    marked_count = 0
    offset_count = 0
    for chamber in chambers:
        r0, c0, r1, c1 = chamber["bbox"]
        in_patch = inp[r0 : r1 + 1, c0 : c1 + 1]
        out_patch = out[r0 : r1 + 1, c0 : c1 + 1]
        info = _analyze_chamber(in_patch)
        if info["seed_offset"] != (0, 0):
            offset_count += 1

        primary = info["primary"]
        if info["marked"]:
            marked_count += 1
            expanded = _expand_seed(info["seed_mask"], primary)
            expected = _tile(expanded, in_patch.shape)
            if not np.array_equal(out_patch, expected):
                return False
            if np.array_equal(out_patch, np.full(in_patch.shape, primary, dtype=int)):
                return False
            used = set(map(int, np.unique(out_patch)))
            if MARKER not in used or primary not in used or BG not in used:
                return False
        else:
            expected = np.full(in_patch.shape, primary, dtype=int)
            if not np.array_equal(out_patch, expected):
                return False
            if len(np.unique(out_patch)) != 1:
                return False
            expanded = _tile(_expand_seed(info["seed_mask"], primary), in_patch.shape)
            if np.array_equal(out_patch, expanded):
                return False

    return marked_count == 2 and offset_count >= 1


def _gen_attempt() -> np.ndarray | None:
    top_h = random.randint(*CHAMBER_SIZE_RANGE)
    bottom_h = random.randint(*CHAMBER_SIZE_RANGE)
    left_w = random.randint(*CHAMBER_SIZE_RANGE)
    right_w = random.randint(*CHAMBER_SIZE_RANGE)

    primaries = list(SEED_COLORS)
    random.shuffle(primaries)
    seed_masks = _sample_seed_masks()
    marked_indices = set(random.sample(range(4), 2))

    chamber_specs: list[dict] = []
    shifted_used = False
    for idx in range(4):
        height = top_h if idx < 2 else bottom_h
        width = left_w if idx % 2 == 0 else right_w
        seed_offset = random.choice(SEED_OFFSETS)
        shifted_used |= seed_offset != (0, 0)
        candidates = _marker_candidates(height, width, seed_offset)
        if idx in marked_indices and not candidates:
            return None

        spec = {
            "primary": primaries[idx],
            "seed_mask": seed_masks[idx],
            "seed_offset": seed_offset,
            "marked": idx in marked_indices,
        }
        if spec["marked"]:
            spec["marker"] = random.choice(candidates)
        chamber_specs.append(spec)

    if not shifted_used:
        return None

    inp = _render_input(top_h, bottom_h, left_w, right_w, chamber_specs)
    try:
        out = main(inp)
    except AssertionError:
        return None

    if np.array_equal(inp, out):
        return None
    if not _validate_scene(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
