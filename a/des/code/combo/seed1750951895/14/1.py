import random
from collections import Counter

import numpy as np


BG = 7
TEAL = 8
PALETTE = (1, 2, 3, 4, 5, 6, 9)
MAX_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _dominant_color(values: np.ndarray) -> int:
    return int(Counter(map(int, values.ravel())).most_common(1)[0][0])


def _corner_and_color(grid: np.ndarray, coords: list[tuple[int, int]], bg: int) -> tuple[tuple[int, int, int, int], tuple[int, int], int]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    colors = set(map(int, patch.ravel()))
    assert bg not in colors
    assert TEAL in colors
    assert len(colors) == 2

    teal_positions = np.argwhere(patch == TEAL)
    assert len(teal_positions) == 1
    tr, tc = map(int, teal_positions[0])
    assert (tr, tc) in {
        (0, 0),
        (0, patch.shape[1] - 1),
        (patch.shape[0] - 1, 0),
        (patch.shape[0] - 1, patch.shape[1] - 1),
    }
    assert np.all(patch != bg)

    main_colors = [color for color in colors if color != TEAL]
    assert len(main_colors) == 1
    main_color = int(main_colors[0])
    expected = np.full_like(patch, main_color)
    expected[tr, tc] = TEAL
    assert np.array_equal(patch, expected)
    return (r0, c0, r1, c1), (tr, tc), main_color


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _dominant_color(g)

    comps = _components(g != bg)
    out = np.full_like(g, bg)

    for comp in comps:
        (r0, c0, r1, c1), (tr, tc), color = _corner_and_color(g, comp, bg)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        step = min(h, w)

        if (tr, tc) == (0, 0):
            dr, dc = -step, -step
        elif (tr, tc) == (0, w - 1):
            dr, dc = -step, step
        elif (tr, tc) == (h - 1, 0):
            dr, dc = step, -step
        else:
            dr, dc = step, step

        nr0 = r0 + dr
        nc0 = c0 + dc
        nr1 = nr0 + h - 1
        nc1 = nc0 + w - 1
        assert 0 <= nr0 <= nr1 < g.shape[0]
        assert 0 <= nc0 <= nc1 < g.shape[1]
        out[nr0 : nr1 + 1, nc0 : nc1 + 1] = color

    return out


def _corner_vectors() -> list[tuple[str, tuple[int, int], callable]]:
    return [
        ("tl", (-1, -1), lambda h, w: (0, 0)),
        ("tr", (-1, 1), lambda h, w: (0, w - 1)),
        ("bl", (1, -1), lambda h, w: (h - 1, 0)),
        ("br", (1, 1), lambda h, w: (h - 1, w - 1)),
    ]


def _inflate(box: tuple[int, int, int, int], margin: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = box
    return max(0, r0 - margin), max(0, c0 - margin), min(h - 1, r1 + margin), min(w - 1, c1 + margin)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _sample_objects(rng: random.Random, h: int, w: int) -> list[dict] | None:
    count = rng.randint(2, 4)
    colors = rng.sample(PALETTE, count)
    dirs = _corner_vectors()
    if count == 4:
        chosen_dirs = dirs[:]
        rng.shuffle(chosen_dirs)
    else:
        chosen_dirs = rng.sample(dirs, count)

    placed_input: list[tuple[int, int, int, int]] = []
    placed_output: list[tuple[int, int, int, int]] = []
    objects: list[dict] = []

    for color, (_, (sr, sc), corner_fn) in zip(colors, chosen_dirs):
        placed = False
        for _ in range(80):
            rect_h = rng.randint(2, 6)
            rect_w = rng.randint(2, 6)
            step = min(rect_h, rect_w)

            out_r0_min = step if sr == 1 else 0
            out_r0_max = h - rect_h - step if sr == -1 else h - rect_h
            out_c0_min = step if sc == 1 else 0
            out_c0_max = w - rect_w - step if sc == -1 else w - rect_w
            if out_r0_min > out_r0_max or out_c0_min > out_c0_max:
                continue

            out_r0 = rng.randint(out_r0_min, out_r0_max)
            out_c0 = rng.randint(out_c0_min, out_c0_max)
            in_r0 = out_r0 - sr * step
            in_c0 = out_c0 - sc * step

            input_box = (in_r0, in_c0, in_r0 + rect_h - 1, in_c0 + rect_w - 1)
            output_box = (out_r0, out_c0, out_r0 + rect_h - 1, out_c0 + rect_w - 1)

            input_safe = _inflate(input_box, 1, (h, w))
            output_safe = _inflate(output_box, 1, (h, w))
            if any(_boxes_overlap(input_safe, prev) for prev in placed_input):
                continue
            if any(_boxes_overlap(output_safe, prev) for prev in placed_output):
                continue

            objects.append(
                {
                    "color": color,
                    "input_box": input_box,
                    "corner": corner_fn(rect_h, rect_w),
                }
            )
            placed_input.append(input_safe)
            placed_output.append(output_safe)
            placed = True
            break
        if not placed:
            return None

    return objects


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        h = rng.randint(12, 24)
        w = rng.randint(12, 24)
        objects = _sample_objects(rng, h, w)
        if objects is None:
            continue

        grid = np.full((h, w), BG, dtype=int)
        for obj in objects:
            r0, c0, r1, c1 = obj["input_box"]
            grid[r0 : r1 + 1, c0 : c1 + 1] = obj["color"]
            cr, cc = obj["corner"]
            grid[r0 + cr, c0 + cc] = TEAL

        try:
            out = main(grid)
        except AssertionError:
            continue
        if out.shape != grid.shape:
            continue
        return grid

    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
