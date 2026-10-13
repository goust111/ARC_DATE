from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG_COLOR = 0
FENCE_COLOR = 5
SEED_COLORS = tuple(color for color in range(10) if color not in (BG_COLOR, FENCE_COLOR))

GRID_SIDE_MAX = 30
MAX_GENERATION_ATTEMPTS = 300
PLACEMENT_RETRIES = 250


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask as coordinate arrays."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            comps.append(np.array(cells, dtype=int))

    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _draw_border(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _frame_bbox(grid: np.ndarray, color: int, coords: np.ndarray) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    expected_area = 2 * (h + w) - 4
    if int(coords.shape[0]) != expected_area:
        return None

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not np.all(patch[0, :] == color):
        return None
    if not np.all(patch[-1, :] == color):
        return None
    if not np.all(patch[:, 0] == color):
        return None
    if not np.all(patch[:, -1] == color):
        return None
    if np.any(patch[1:-1, 1:-1] == color):
        return None

    return r0, c0, r1, c1


def _find_frames(grid: np.ndarray, bg: int) -> list[tuple[int, int, int, int, int]]:
    """Find all monochrome hollow rectangular frame components."""
    frames: list[tuple[int, int, int, int, int]] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for coords in _components(grid == color):
            box = _frame_bbox(grid, color, coords)
            if box is not None:
                frames.append((*box, color))

    frames.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4]))
    return frames


def _unique_majority(values: np.ndarray) -> int:
    counts = Counter(map(int, values.tolist()))
    assert counts, "each frame must contain at least one seed"
    ranked = counts.most_common()
    if len(ranked) >= 2:
        assert ranked[0][1] > ranked[1][1], "seed majority must be strict"
    return int(ranked[0][0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each rectangular fence outward by its seed count and color the new rings by the majority seed color.

    Input structure:
    - bg = most common color (black).
    - One or more disconnected 1-pixel-wide hollow rectangular fence components appear on the background.
    - Each fence encloses non-bg, non-fence seed pixels, and one seed color is the unique majority inside that fence.

    Rule:
    - For each fence, count the enclosed seed pixels; that total N is the number of outward concentric border rings to add.
    - Keep the original fence and all interior seed pixels unchanged.
    - Color every added ring with the uniquely most frequent seed color inside that fence.

    Key insight:
    - The same interior seed set is used twice: total seed count controls geometric growth, while the seed-color majority controls the color of every new layer.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(Counter(map(int, g.ravel())).most_common(1)[0][0])

    frames = _find_frames(g, bg)
    assert frames, "expected at least one rectangular frame"

    out = g.copy()
    h, w = g.shape

    for r0, c0, r1, c1, frame_color in frames:
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_mask = (interior != bg) & (interior != frame_color)
        seed_values = interior[seed_mask]
        assert seed_values.size > 0, "frame must contain seeds"

        layers = int(seed_values.size)
        assert layers <= 5, "seed count exceeds puzzle bound"
        majority_color = _unique_majority(seed_values)

        for layer in range(1, layers + 1):
            rr0 = r0 - layer
            cc0 = c0 - layer
            rr1 = r1 + layer
            cc1 = c1 + layer
            assert rr0 >= 0 and cc0 >= 0 and rr1 < h and cc1 < w, "expanded frame exceeds canvas"
            _draw_border(out, rr0, cc0, rr1, cc1, majority_color)

    return out


def _sample_seed_counts(seed_total: int) -> tuple[dict[int, int], int]:
    majority_color = random.choice(SEED_COLORS)
    if seed_total == 1:
        return {majority_color: 1}, majority_color

    min_majority = seed_total // 2 + 1
    if min_majority == seed_total:
        majority_count = seed_total
    elif random.random() < 0.75:
        majority_count = random.randint(min_majority, seed_total - 1)
    else:
        majority_count = random.randint(min_majority, seed_total)

    counts = {majority_color: majority_count}
    remainder = seed_total - majority_count
    if remainder == 0:
        return counts, majority_color

    other_count = random.randint(1, min(3, remainder))
    other_colors = random.sample([c for c in SEED_COLORS if c != majority_color], other_count)

    if other_count == 1:
        parts = [remainder]
    else:
        cuts = sorted(random.sample(range(1, remainder), other_count - 1))
        points = [0] + cuts + [remainder]
        parts = [points[i + 1] - points[i] for i in range(len(points) - 1)]
        random.shuffle(parts)

    for color, amount in zip(other_colors, parts):
        counts[color] = int(amount)

    return counts, majority_color


def _make_frame_spec(frame_count: int) -> dict[str, object] | None:
    layers = random.randint(1, 5)
    inner_max = {1: 7, 2: 6, 3: 5}.get(frame_count, 5)

    for _ in range(80):
        inner_h = random.randint(2, inner_max)
        inner_w = random.randint(2, inner_max)
        if inner_h * inner_w < layers + 1:
            continue

        seed_counts, majority_color = _sample_seed_counts(layers)
        return {
            "layers": layers,
            "inner_h": inner_h,
            "inner_w": inner_w,
            "outer_h": inner_h + 2,
            "outer_w": inner_w + 2,
            "final_h": inner_h + 2 + 2 * layers,
            "final_w": inner_w + 2 + 2 * layers,
            "seed_counts": seed_counts,
            "majority_color": majority_color,
        }

    return None


def _place_specs(height: int, width: int, specs: list[dict[str, object]]) -> bool:
    occupied = np.zeros((height, width), dtype=bool)

    for spec in sorted(specs, key=lambda item: -(int(item["final_h"]) * int(item["final_w"]))):
        final_h = int(spec["final_h"])
        final_w = int(spec["final_w"])
        placed = False

        for _ in range(PLACEMENT_RETRIES):
            fr0 = random.randint(0, height - final_h)
            fc0 = random.randint(0, width - final_w)
            fr1 = fr0 + final_h - 1
            fc1 = fc0 + final_w - 1

            rr0 = max(0, fr0 - 1)
            cc0 = max(0, fc0 - 1)
            rr1 = min(height - 1, fr1 + 1)
            cc1 = min(width - 1, fc1 + 1)
            if occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                continue

            occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
            spec["final_bbox"] = (fr0, fc0, fr1, fc1)
            placed = True
            break

        if not placed:
            return False

    return True


def _build_seed_values(seed_counts: dict[int, int]) -> list[int]:
    values: list[int] = []
    for color, count in seed_counts.items():
        values.extend([int(color)] * int(count))
    random.shuffle(values)
    return values


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.choice((1, 2, 3))
    specs: list[dict[str, object]] = []
    for _ in range(frame_count):
        spec = _make_frame_spec(frame_count)
        if spec is None:
            return None
        specs.append(spec)

    buffered_area = sum((int(spec["final_h"]) + 2) * (int(spec["final_w"]) + 2) for spec in specs)
    max_final_h = max(int(spec["final_h"]) for spec in specs)
    max_final_w = max(int(spec["final_w"]) for spec in specs)
    side_hint = int(buffered_area**0.5) + frame_count

    min_h = max(max_final_h + 2, side_hint)
    min_w = max(max_final_w + 2, side_hint)
    if min_h > GRID_SIDE_MAX or min_w > GRID_SIDE_MAX:
        return None

    height = random.randint(min_h, GRID_SIDE_MAX)
    width = random.randint(min_w, GRID_SIDE_MAX)
    if not _place_specs(height, width, specs):
        return None

    inp = np.full((height, width), BG_COLOR, dtype=int)

    for spec in specs:
        layers = int(spec["layers"])
        fr0, fc0, fr1, fc1 = spec["final_bbox"]  # type: ignore[index]
        r0 = fr0 + layers
        c0 = fc0 + layers
        r1 = fr1 - layers
        c1 = fc1 - layers
        _draw_border(inp, r0, c0, r1, c1, FENCE_COLOR)

        interior_cells = [(rr, cc) for rr in range(r0 + 1, r1) for cc in range(c0 + 1, c1)]
        if len(interior_cells) < layers + 1:
            return None

        seed_positions = random.sample(interior_cells, layers)
        seed_values = _build_seed_values(spec["seed_counts"])  # type: ignore[arg-type]
        for (rr, cc), color in zip(seed_positions, seed_values):
            inp[rr, cc] = int(color)

        spec["bbox"] = (r0, c0, r1, c1)

    if Counter(map(int, inp.ravel())).most_common(1)[0][0] != BG_COLOR:
        return None

    expected = inp.copy()
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]  # type: ignore[index]
        layers = int(spec["layers"])
        majority_color = int(spec["majority_color"])
        for layer in range(1, layers + 1):
            _draw_border(expected, r0 - layer, c0 - layer, r1 + layer, c1 + layer, majority_color)

    try:
        actual = main(inp)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after repeated attempts")
