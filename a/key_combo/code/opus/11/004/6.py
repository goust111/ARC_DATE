from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
FRAME = 1
FG = 2
GEN_RETRIES = 400
MIN_GAP = 2


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _blue_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != FRAME:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == FRAME:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _is_frame_component(grid: np.ndarray, comp: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(comp)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 4 or width < 4:
        return False
    expected_area = 2 * height + 2 * width - 4
    if len(comp) != expected_area:
        return False
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not np.all(patch[0, :] == FRAME):
        return False
    if not np.all(patch[-1, :] == FRAME):
        return False
    if not np.all(patch[:, 0] == FRAME):
        return False
    if not np.all(patch[:, -1] == FRAME):
        return False
    if np.any(patch[1:-1, 1:-1] == FRAME):
        return False
    return True


def _extract_sprites(grid: np.ndarray) -> list[dict]:
    sprites: list[dict] = []
    for comp in _blue_components(grid):
        if not _is_frame_component(grid, comp):
            continue
        r0, c0, r1, c1 = _bbox(comp)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        sprites.append(
            {
                "bbox": (r0, c0, r1, c1),
                "sprite": patch,
                "interior": patch[1:-1, 1:-1].copy(),
            }
        )
    return sorted(sprites, key=lambda item: item["bbox"])


def _pattern_key(arr: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(x) for x in row) for row in arr.tolist())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the single non-symmetric framed sprite whose interior becomes unique only after symmetry filtering.

    Input structure:
    - The grid is a black background containing 5-7 blue one-cell-thick rectangular frames.
    - Every valid sprite is an axis-aligned complete rectangle of the same outer size.
    - Each frame encloses a red-on-black interior pattern, and different sprites are separated by background gaps.

    Rule:
    - Detect every complete blue rectangular frame and crop its interior pattern without the border.
    - Flip each interior vertically by reversing its row order; discard the sprites whose interiors stay unchanged under that flip.
    - Among the remaining non-symmetric sprites, compare interiors exactly and keep the one whose interior occurs once.
    - Output that winning sprite unchanged, including its blue border, cropped to its own bounding box.

    Key insight:
    - The answer is not the globally unique sprite: row-flip symmetry removes a decoy subset first, and only then does interior-pattern uniqueness identify the target.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sprites = _extract_sprites(g)
    assert len(sprites) >= 1, "expected at least one framed sprite"
    shapes = {item["sprite"].shape for item in sprites}
    assert len(shapes) == 1, "expected all framed sprites to share one size"

    asymmetric = [item for item in sprites if not np.array_equal(item["interior"], np.flipud(item["interior"]))]
    assert asymmetric, "expected at least one non-symmetric sprite"
    keys = [_pattern_key(item["interior"]) for item in asymmetric]
    counts = Counter(keys)
    winners = [item for item, key in zip(asymmetric, keys) if counts[key] == 1]
    assert len(winners) == 1, "expected exactly one unique non-symmetric interior"
    return winners[0]["sprite"].copy()


def _random_binary(shape: tuple[int, int]) -> np.ndarray:
    return np.random.choice([BG, FG], size=shape, p=[0.55, 0.45]).astype(int)


def _has_both_colors(arr: np.ndarray) -> bool:
    colors = set(int(x) for x in np.unique(arr))
    return colors == {BG, FG}


def _make_row_symmetric_not_col(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    half = (h + 1) // 2
    for _ in range(200):
        top = _random_binary((half, w))
        if h % 2 == 0:
            pat = np.vstack([top, np.flipud(top)])
        else:
            pat = np.vstack([top[:-1], top[-1:], np.flipud(top[:-1])])
        if _has_both_colors(pat) and np.array_equal(pat, np.flipud(pat)) and not np.array_equal(pat, np.fliplr(pat)):
            return pat
    raise RuntimeError("failed to generate row-symmetric pattern")


def _make_col_symmetric_not_row(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    half = (w + 1) // 2
    for _ in range(200):
        left = _random_binary((h, half))
        if w % 2 == 0:
            pat = np.hstack([left, np.fliplr(left)])
        else:
            pat = np.hstack([left[:, :-1], left[:, -1:], np.fliplr(left[:, :-1])])
        if _has_both_colors(pat) and np.array_equal(pat, np.fliplr(pat)) and not np.array_equal(pat, np.flipud(pat)):
            return pat
    raise RuntimeError("failed to generate col-symmetric pattern")


def _make_fully_asymmetric(shape: tuple[int, int]) -> np.ndarray:
    for _ in range(400):
        pat = _random_binary(shape)
        if _has_both_colors(pat) and not np.array_equal(pat, np.flipud(pat)) and not np.array_equal(pat, np.fliplr(pat)):
            return pat
    raise RuntimeError("failed to generate asymmetric pattern")


def _distinct_pattern(factory, shape: tuple[int, int], used: set[tuple[tuple[int, ...], ...]]) -> np.ndarray:
    for _ in range(400):
        pat = factory(shape)
        key = _pattern_key(pat)
        if key not in used:
            used.add(key)
            return pat
    raise RuntimeError("failed to generate distinct pattern")


def _frame_sprite(interior: np.ndarray) -> np.ndarray:
    h, w = interior.shape
    sprite = np.full((h + 2, w + 2), FRAME, dtype=int)
    sprite[1:-1, 1:-1] = interior
    return sprite


def _choose_counts(total: int) -> tuple[int, int]:
    if total == 5:
        return 3, 2
    if total == 6:
        return 3, 3
    return random.choice([(3, 4), (5, 2)])


def _symmetry_signature(pat: np.ndarray) -> tuple[bool, bool]:
    return np.array_equal(pat, np.flipud(pat)), np.array_equal(pat, np.fliplr(pat))


def _build_pattern_multiset(shape: tuple[int, int]) -> tuple[list[np.ndarray], np.ndarray]:
    total = random.randint(5, 7)
    asymmetric_count, symmetric_count = _choose_counts(total)
    used: set[tuple[tuple[int, ...], ...]] = set()

    repeated_a = _distinct_pattern(_make_col_symmetric_not_row, shape, used)
    unique_target = _distinct_pattern(_make_fully_asymmetric, shape, used)
    patterns: list[np.ndarray] = [repeated_a.copy(), repeated_a.copy(), unique_target.copy()]

    if asymmetric_count == 5:
        second_factory = random.choice([_make_col_symmetric_not_row, _make_fully_asymmetric])
        repeated_b = _distinct_pattern(second_factory, shape, used)
        patterns.extend([repeated_b.copy(), repeated_b.copy()])

    sym_unique = _distinct_pattern(_make_row_symmetric_not_col, shape, used)
    sym_repeated = _distinct_pattern(_make_row_symmetric_not_col, shape, used)
    symmetric_patterns: list[np.ndarray] = [sym_unique.copy()]
    symmetric_patterns.extend(sym_repeated.copy() for _ in range(symmetric_count - 1))
    patterns.extend(symmetric_patterns)
    random.shuffle(patterns)

    survivors = [pat for pat in patterns if not np.array_equal(pat, np.flipud(pat))]
    survivor_counts = Counter(_pattern_key(pat) for pat in survivors)
    assert survivor_counts[_pattern_key(unique_target)] == 1
    assert sum(count == 1 for count in survivor_counts.values()) == 1

    row_only = [pat for pat in patterns if _symmetry_signature(pat) == (True, False)]
    col_only = [pat for pat in patterns if _symmetry_signature(pat) == (False, True)]
    assert row_only, "need row-symmetric decoys"
    assert col_only, "need surviving column-symmetric confuser"
    return patterns, _frame_sprite(unique_target)


def _bboxes_far(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = MIN_GAP) -> bool:
    return a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1]


def _place_sprites(grid_shape: tuple[int, int], sprites: list[np.ndarray]) -> list[tuple[int, int]]:
    h, w = grid_shape
    positions: list[tuple[int, int]] = []
    bboxes: list[tuple[int, int, int, int]] = []
    order = list(range(len(sprites)))
    random.shuffle(order)
    placed: list[tuple[int, int] | None] = [None] * len(sprites)

    for idx in order:
        sh, sw = sprites[idx].shape
        candidates: list[tuple[int, int]] = []
        for r in range(1, h - sh):
            for c in range(1, w - sw):
                bbox = (r, c, r + sh - 1, c + sw - 1)
                if all(_bboxes_far(bbox, other) for other in bboxes):
                    candidates.append((r, c))
        if not candidates:
            raise ValueError("no placement found")
        r, c = random.choice(candidates)
        placed[idx] = (r, c)
        bboxes.append((r, c, r + sh - 1, c + sw - 1))
        positions.append((r, c))

    return [pos for pos in placed if pos is not None]


def _render(grid_shape: tuple[int, int], sprites: list[np.ndarray], positions: list[tuple[int, int]]) -> np.ndarray:
    grid = np.full(grid_shape, BG, dtype=int)
    for sprite, (r, c) in zip(sprites, positions):
        sh, sw = sprite.shape
        grid[r : r + sh, c : c + sw] = sprite
    return grid


def _validate_generated(grid: np.ndarray, expected: np.ndarray, total_count: int) -> None:
    sprites = _extract_sprites(grid)
    assert len(sprites) == total_count, "sprite detection mismatch"
    assert len({item["sprite"].shape for item in sprites}) == 1, "mixed sprite sizes"

    symmetric = [item for item in sprites if np.array_equal(item["interior"], np.flipud(item["interior"]))]
    asymmetric = [item for item in sprites if not np.array_equal(item["interior"], np.flipud(item["interior"]))]
    assert len(symmetric) >= 2, "need at least two symmetric sprites"
    assert len(asymmetric) >= 3, "need at least three asymmetric sprites"

    keys = [_pattern_key(item["interior"]) for item in asymmetric]
    counts = Counter(keys)
    assert sum(count == 1 for count in counts.values()) == 1, "need one unique asymmetric interior"

    sym_keys = Counter(_pattern_key(item["interior"]) for item in symmetric)
    assert any(count == 1 for count in sym_keys.values()), "need a symmetric unique decoy"

    row_only = [item for item in sprites if _symmetry_signature(item["interior"]) == (True, False)]
    col_only = [item for item in sprites if _symmetry_signature(item["interior"]) == (False, True)]
    assert row_only, "need row-only symmetric examples"
    assert col_only, "need column-only confusers"

    out = main(grid)
    assert np.array_equal(out, expected), "solver output mismatch"


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        inner_shape = (random.randint(2, 4), random.randint(2, 4))
        patterns, expected = _build_pattern_multiset(inner_shape)
        sprites = [_frame_sprite(pat) for pat in patterns]
        sprite_h, sprite_w = sprites[0].shape
        total = len(sprites)

        min_h = max(12, total * 2 + sprite_h + 4)
        min_w = max(12, total * 2 + sprite_w + 2)
        max_h = min(30, max(min_h, sprite_h + 18))
        max_w = min(30, max(min_w, sprite_w + 18))
        if min_h > 30 or min_w > 30:
            continue
        grid_shape = (random.randint(min_h, max_h), random.randint(min_w, max_w))

        try:
            positions = _place_sprites(grid_shape, sprites)
        except ValueError:
            continue

        grid = _render(grid_shape, sprites, positions)
        try:
            _validate_generated(grid, expected, total)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")
