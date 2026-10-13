import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
MAX_GRID_SIZE = 30
GEN_ATTEMPTS = 300


TEMPLATE_LIBRARY = [
    np.array([[2, 1, 0], [2, 2, 1], [0, 2, 1]], dtype=int),
    np.array([[2, 1, 0], [2, 2, 0], [0, 2, 1]], dtype=int),
    np.array([[1, 2, 0], [1, 2, 0], [2, 2, 2]], dtype=int),
    np.array([[2, 2, 0], [2, 2, 0], [1, 1, 2]], dtype=int),
    np.array([[2, 2, 0], [1, 2, 0], [2, 2, 1]], dtype=int),
    np.array([[2, 1, 0], [2, 2, 0], [1, 2, 2]], dtype=int),
    np.array([[0, 0, 1, 2], [2, 1, 2, 0], [2, 2, 0, 0], [0, 2, 0, 0]], dtype=int),
    np.array([[0, 0, 2, 2], [2, 1, 2, 0], [1, 2, 0, 0], [0, 2, 0, 0]], dtype=int),
]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _transform(arr: np.ndarray, rotation: int, flip_lr: bool) -> np.ndarray:
    base = np.fliplr(arr) if flip_lr else arr
    return np.rot90(base, rotation)


def _unique_rigid_variants(template: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    seen: set[bytes] = set()
    variants: list[tuple[np.ndarray, np.ndarray]] = []
    marker_mask = template == BLUE
    for flip_lr in (False, True):
        for rotation in range(4):
            transformed_marker = _transform(marker_mask, rotation, flip_lr)
            key = transformed_marker.astype(np.uint8).tobytes()
            if key in seen:
                continue
            seen.add(key)
            variants.append((transformed_marker, _transform(template, rotation, flip_lr)))
    return variants


def _stamp(grid: np.ndarray, top: tuple[int, int], sprite: np.ndarray) -> None:
    r0, c0 = top
    h, w = sprite.shape
    window = grid[r0 : r0 + h, c0 : c0 + w]
    mask = sprite != BG
    window[mask] = sprite[mask]


def _inflated_overlap(a: tuple[int, int], b: tuple[int, int], size: int, margin: int = 1) -> bool:
    ar0, ac0 = a
    br0, bc0 = b
    ar1 = ar0 + size - 1
    ac1 = ac0 + size - 1
    br1 = br0 + size - 1
    bc1 = bc0 + size - 1
    return not (
        ar1 + margin < br0
        or br1 + margin < ar0
        or ac1 + margin < bc0
        or bc1 + margin < ac0
    )


def _random_delta(size: int) -> tuple[int, int]:
    long_steps = list(range(size + 1, 7))
    short_steps = [1, 2, 3]
    mode = random.choice(
        ["horizontal", "vertical", "diag_right", "diag_left", "steep_right", "steep_left"]
    )
    if mode == "horizontal":
        return 0, random.choice(long_steps)
    if mode == "vertical":
        return random.choice(long_steps), 0
    if mode == "diag_right":
        return random.choice(short_steps), random.choice(long_steps)
    if mode == "diag_left":
        return random.choice(short_steps), -random.choice(long_steps)
    if mode == "steep_right":
        return random.choice(long_steps), random.choice([-2, -1, 1, 2])
    return random.choice(long_steps), random.choice([-2, -1, 1, 2])


def _choose_layout(size: int, full_count: int) -> tuple[int, int, list[tuple[int, int]]] | None:
    steps = full_count
    for _ in range(120):
        dr, dc = _random_delta(size)
        if dr == 0 and dc <= 0:
            continue
        min_c_shift = min(0, dc * steps)
        max_c_shift = max(0, dc * steps)
        min_h = size + dr * steps
        min_w = size + max_c_shift - min_c_shift
        if min_h > 24 or min_w > 24:
            continue

        extra_h = random.randint(1, min(6, MAX_GRID_SIZE - min_h))
        extra_w = random.randint(1, min(6, MAX_GRID_SIZE - min_w))
        height = min_h + extra_h
        width = min_w + extra_w

        r0 = random.randint(0, height - min_h)
        c0 = random.randint(-min_c_shift, width - size - max_c_shift)
        positions = [(r0 + k * dr, c0 + k * dc) for k in range(full_count + 1)]

        if any(_inflated_overlap(a, b, size) for i, a in enumerate(positions) for b in positions[i + 1 :]):
            continue
        return height, width, positions
    return None


def _render_expected(grid: np.ndarray, target_top: tuple[int, int], transformed: np.ndarray) -> np.ndarray:
    out = grid.copy()
    _stamp(out, target_top, transformed)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the next translated stamp by matching its blue marker mask to a unique rigid transform of the template.

    Input structure:
    - A black grid contains 2-3 complete red-and-blue stamp instances plus one later stamp location that only keeps the blue marker pixels.
    - The complete instances all share one square 3x3 or 4x4 bounding box and appear as separate connected components.
    - The later marker-only location has the same bounding-box size as the template and contains no red cells.

    Rule:
    - Extract the common template from the complete components and sort their top-left corners in row-major order.
    - Infer the constant translation step between consecutive complete instances, then extrapolate one more top-left corner.
    - Read the blue marker mask already present at that extrapolated location, find the unique rotation/reflection of the template whose blue mask matches it exactly, and stamp the full transformed template there.

    Key insight:
    - Position comes from sequence extrapolation, while orientation comes from the local blue marker pattern, so both clues must be solved independently and then combined.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    full_instances: list[tuple[int, int, np.ndarray]] = []
    for coords in _components(g != BG):
        colors = {int(g[r, c]) for r, c in coords}
        if RED in colors and BLUE in colors:
            r0, c0, r1, c1 = _bbox(coords)
            sprite = g[r0 : r1 + 1, c0 : c1 + 1].copy()
            full_instances.append((r0, c0, sprite))

    assert len(full_instances) >= 2, "expected at least two complete stamp instances"
    full_instances.sort(key=lambda item: (item[0], item[1]))

    template = full_instances[0][2]
    h, w = template.shape
    assert h == w and h in (3, 4), "expected a square 3x3 or 4x4 template"
    for _r0, _c0, sprite in full_instances[1:]:
        assert sprite.shape == template.shape
        assert np.array_equal(sprite, template), "complete instances must share one original template"

    deltas = [
        (full_instances[i + 1][0] - full_instances[i][0], full_instances[i + 1][1] - full_instances[i][1])
        for i in range(len(full_instances) - 1)
    ]
    assert len(set(deltas)) == 1, "complete instances must follow a constant translation"
    dr, dc = deltas[0]

    target_r0 = full_instances[-1][0] + dr
    target_c0 = full_instances[-1][1] + dc
    assert 0 <= target_r0 <= g.shape[0] - h and 0 <= target_c0 <= g.shape[1] - w

    target_crop = g[target_r0 : target_r0 + h, target_c0 : target_c0 + w]
    assert not np.any(target_crop == RED), "the extrapolated slot should only expose blue markers"
    observed_marker = target_crop == BLUE
    assert observed_marker.any(), "expected visible blue markers at the extrapolated slot"

    matches = [
        transformed
        for marker_mask, transformed in _unique_rigid_variants(template)
        if np.array_equal(marker_mask, observed_marker)
    ]
    assert len(matches) == 1, "marker mask should determine a unique rigid transform"

    out = g.copy()
    _stamp(out, (target_r0, target_c0), matches[0])
    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        template = random.choice(TEMPLATE_LIBRARY)
        size = int(template.shape[0])
        full_count = random.choice([2, 3])
        layout = _choose_layout(size, full_count)
        if layout is None:
            continue

        height, width, positions = layout
        grid = np.zeros((height, width), dtype=int)
        for top in positions[:full_count]:
            _stamp(grid, top, template)

        variants = _unique_rigid_variants(template)
        if len(variants) != 8:
            continue
        marker_mask, transformed = random.choice(variants)
        target_top = positions[-1]
        target_window = grid[target_top[0] : target_top[0] + size, target_top[1] : target_top[1] + size]
        target_window[marker_mask] = BLUE

        expected = _render_expected(grid, target_top, transformed)
        if np.array_equal(grid, expected):
            continue
        if not np.array_equal(main(grid), expected):
            continue
        return grid

    raise RuntimeError("failed to generate a valid input for puzzle 8")


for _template in TEMPLATE_LIBRARY:
    assert _template.ndim == 2
    assert _template.shape[0] == _template.shape[1]
    assert _template.shape[0] in (3, 4)
    assert 5 <= int(np.count_nonzero(_template)) <= 8
    assert len(_components(_template != BG)) == 1
    assert len(_unique_rigid_variants(_template)) == 8
