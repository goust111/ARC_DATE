from __future__ import annotations

import random
from collections import Counter

import numpy as np


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
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
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(points: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(points[:, 0].min()),
        int(points[:, 1].min()),
        int(points[:, 0].max()),
        int(points[:, 1].max()),
    )


def _touches_boundary(points: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(points[:, 0] == 0)
        or np.any(points[:, 0] == h - 1)
        or np.any(points[:, 1] == 0)
        or np.any(points[:, 1] == w - 1)
    )


def _most_common_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _find_separator(grid: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = grid.shape
    assert w % 2 == 1
    sep = w // 2
    sep_color = int(grid[0, sep])
    assert sep_color != bg
    assert bool(np.all(grid[:, sep] == sep_color))
    return sep, sep_color


def _extract_hole_mask(
    left: np.ndarray,
    bg: int,
    points: np.ndarray,
) -> tuple[int, int, int, int, np.ndarray, int]:
    r0, c0, r1, c1 = _bbox(points)
    crop = left[r0 : r1 + 1, c0 : c1 + 1]
    crop_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    crop_mask[points[:, 0] - r0, points[:, 1] - c0] = True
    hole_candidates = [
        comp
        for comp in _components(crop == bg)
        if not _touches_boundary(comp, crop.shape)
    ]
    assert len(hole_candidates) == 1
    hole = hole_candidates[0]
    hr0, hc0, hr1, hc1 = _bbox(hole)
    tight = np.zeros((hr1 - hr0 + 1, hc1 - hc0 + 1), dtype=bool)
    tight[hole[:, 0] - hr0, hole[:, 1] - hc0] = True
    color = int(crop[tuple(points[0] - np.array([r0, c0]))])
    assert color != bg
    assert bool(np.all(crop[crop_mask] == color))
    return r0, c0, r1, c1, tight, color


def _centered_start(center2: int, size: int) -> int:
    num = center2 - size + 1
    assert num % 2 == 0
    return num // 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each left-side object's hole shape to the symmetric position on the right.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    sep, sep_color = _find_separator(g, bg)
    left = g[:, :sep]

    comps = _components(left != bg)
    objects = []
    for comp in comps:
        objects.append(_extract_hole_mask(left, bg, comp))
    assert objects

    out = g.copy()
    out[:, sep + 1 :] = bg
    out[:, sep] = sep_color

    for r0, c0, r1, c1, hole_mask, color in objects:
        flipped = np.fliplr(hole_mask)
        hh, hw = map(int, flipped.shape)
        top = _centered_start(r0 + r1, hh)
        left_col = _centered_start(4 * sep - (c0 + c1), hw)
        assert 0 <= top <= g.shape[0] - hh
        assert sep + 1 <= left_col <= g.shape[1] - hw
        block = out[top : top + hh, left_col : left_col + hw]
        block[flipped] = color

    return out


# --------------- generator ---------------

BG = 0
SEPARATOR = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GRID_H_RANGE = (12, 20)
HALF_W_RANGE = (10, 16)
OBJECT_COUNT_RANGE = (1, 4)
OUTER_H_RANGE = (5, 10)
OUTER_W_RANGE = (5, 10)
MAX_SHAVED_CELLS = 5
PLACEMENT_MARGIN = 1
PLACEMENT_GAP = 1
MAX_ATTEMPTS = 400
MASK_ATTEMPTS = 120


def _interior_voids(shape_mask: np.ndarray) -> list[np.ndarray]:
    return [
        comp
        for comp in _components(~shape_mask)
        if not _touches_boundary(comp, shape_mask.shape)
    ]


def _random_connected_mask(h: int, w: int) -> np.ndarray | None:
    area_min = max(3, (h * w + 2) // 3)
    area_max = h * w - 1
    if area_min > area_max:
        return None

    for _ in range(MASK_ATTEMPTS):
        target = random.randint(area_min, area_max)
        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True

        while int(mask.sum()) < target:
            frontier: set[tuple[int, int]] = set()
            for rr, cc in np.argwhere(mask):
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            nr, nc = random.choice(tuple(frontier))
            mask[nr, nc] = True

        if int(mask.sum()) != target:
            continue
        pts = np.argwhere(mask)
        if len(pts) < 3:
            continue
        if not (
            int(pts[:, 0].min()) == 0
            and int(pts[:, 0].max()) == h - 1
            and int(pts[:, 1].min()) == 0
            and int(pts[:, 1].max()) == w - 1
        ):
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        return mask
    return None


def _hole_size_choices(outer_h: int, outer_w: int) -> list[tuple[int, int]]:
    choices = []
    for hh in range(2, outer_h - 1):
        if hh % 2 != outer_h % 2:
            continue
        for hw in range(2, outer_w - 1):
            if hw % 2 != outer_w % 2:
                continue
            if min(hh, hw) < 2 or max(hh, hw) < 3:
                continue
            choices.append((hh, hw))
    return choices


def _make_object() -> dict[str, object] | None:
    outer_h = random.randint(*OUTER_H_RANGE)
    outer_w = random.randint(*OUTER_W_RANGE)
    hole_sizes = _hole_size_choices(outer_h, outer_w)
    if not hole_sizes:
        return None
    random.shuffle(hole_sizes)

    for hole_h, hole_w in hole_sizes:
        hole_mask = _random_connected_mask(hole_h, hole_w)
        if hole_mask is None:
            continue
        hole_r = random.randint(1, outer_h - hole_h - 1)
        hole_c = random.randint(1, outer_w - hole_w - 1)

        shape_mask = np.ones((outer_h, outer_w), dtype=bool)
        hole_slice = shape_mask[hole_r : hole_r + hole_h, hole_c : hole_c + hole_w]
        hole_slice[hole_mask] = False

        shaved_goal = random.randint(0, MAX_SHAVED_CELLS)
        candidates = [
            (r, c)
            for r in range(outer_h)
            for c in range(outer_w)
            if shape_mask[r, c] and (r in (0, outer_h - 1) or c in (0, outer_w - 1))
        ]
        random.shuffle(candidates)
        shaved = 0
        for r, c in candidates:
            if shaved >= shaved_goal:
                break
            test = shape_mask.copy()
            test[r, c] = False
            if not (test[0].any() and test[-1].any() and test[:, 0].any() and test[:, -1].any()):
                continue
            if len(_components(test)) != 1:
                continue
            if len(_interior_voids(test)) != 1:
                continue
            shape_mask = test
            shaved += 1

        holes = _interior_voids(shape_mask)
        if len(holes) != 1:
            continue
        hole = holes[0]
        hr0, hc0, hr1, hc1 = _bbox(hole)
        tight = np.zeros((hr1 - hr0 + 1, hc1 - hc0 + 1), dtype=bool)
        tight[hole[:, 0] - hr0, hole[:, 1] - hc0] = True
        if np.array_equal(tight, np.fliplr(tight)):
            continue
        return {"shape_mask": shape_mask, "hole_mask": tight}

    return None


def _boxes_separate(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    gap: int,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _place_objects(
    specs: list[dict[str, object]],
    grid_h: int,
    half_w: int,
) -> bool:
    placed_boxes: list[tuple[int, int, int, int]] = []
    for spec in specs:
        shape_mask = spec["shape_mask"]
        h, w = map(int, shape_mask.shape)
        rows = list(range(PLACEMENT_MARGIN, grid_h - h - PLACEMENT_MARGIN + 1))
        cols = list(range(PLACEMENT_MARGIN, half_w - w - PLACEMENT_MARGIN + 1))
        if not rows or not cols:
            return False
        positions = [(r, c) for r in rows for c in cols]
        random.shuffle(positions)
        placed = False
        for top, left in positions:
            box = (top, left, top + h - 1, left + w - 1)
            if all(_boxes_separate(box, other, PLACEMENT_GAP) for other in placed_boxes):
                spec["top"] = top
                spec["left"] = left
                placed_boxes.append(box)
                placed = True
                break
        if not placed:
            return False
    return True


def _render_input(
    grid_h: int,
    half_w: int,
    specs: list[dict[str, object]],
) -> np.ndarray:
    grid = np.full((grid_h, 2 * half_w + 1), BG, dtype=int)
    grid[:, half_w] = SEPARATOR
    for spec in specs:
        top = int(spec["top"])
        left = int(spec["left"])
        color = int(spec["color"])
        shape_mask = np.asarray(spec["shape_mask"], dtype=bool)
        h, w = shape_mask.shape
        block = grid[top : top + h, left : left + w]
        block[shape_mask] = color
    return grid


def _expected_output(inp: np.ndarray, specs: list[dict[str, object]], half_w: int) -> np.ndarray:
    out = inp.copy()
    out[:, half_w + 1 :] = BG
    out[:, half_w] = SEPARATOR
    for spec in specs:
        top = int(spec["top"])
        left = int(spec["left"])
        shape_mask = np.asarray(spec["shape_mask"], dtype=bool)
        hole_mask = np.asarray(spec["hole_mask"], dtype=bool)
        color = int(spec["color"])
        h, w = shape_mask.shape
        flipped = np.fliplr(hole_mask)
        hh, hw = flipped.shape
        target_top = _centered_start(2 * top + h - 1, hh)
        target_left = _centered_start(4 * half_w - (2 * left + w - 1), hw)
        block = out[target_top : target_top + hh, target_left : target_left + hw]
        block[flipped] = color
    return out


def _bg_is_unique_majority(grid: np.ndarray) -> bool:
    counts = Counter(map(int, grid.ravel()))
    bg_count = counts.get(BG, 0)
    return all(bg_count > cnt for color, cnt in counts.items() if color != BG)


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    half_w = random.randint(*HALF_W_RANGE)
    n_objects = random.randint(*OBJECT_COUNT_RANGE)
    if n_objects > len(PALETTE):
        return None

    specs: list[dict[str, object]] = []
    for color in random.sample(PALETTE, n_objects):
        spec = _make_object()
        if spec is None:
            return None
        spec["color"] = color
        specs.append(spec)

    if not _place_objects(specs, grid_h, half_w):
        return None

    inp = _render_input(grid_h, half_w, specs)
    if not _bg_is_unique_majority(inp):
        return None

    expected = _expected_output(inp, specs, half_w)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
