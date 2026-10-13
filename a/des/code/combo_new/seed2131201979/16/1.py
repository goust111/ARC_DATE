from __future__ import annotations

from collections import Counter
import random

import numpy as np


COLOR_COUNT = 10
GENERATION_ATTEMPTS = 300
GRID_HEIGHT_RANGE = (14, 30)
GRID_WIDTH_RANGE = (14, 30)
FRAME_MIN_OUTER = 5
FRAME_MAX_OUTER = 12
FRAME_PLACE_TRIES = 80
NOISE_PLACE_TRIES = 40
NOISE_SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(int(v) for v in grid.flat).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_by_color(grid: np.ndarray, bg: int) -> list[tuple[int, list[tuple[int, int]]]]:
    height, width = map(int, grid.shape)
    seen = np.zeros((height, width), dtype=bool)
    components: list[tuple[int, list[tuple[int, int]]]] = []

    for r in range(height):
        for c in range(width):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < height and 0 <= nc < width and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append((color, coords))

    return components


def _frame_bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None

    expected = 2 * (height + width) - 4
    if len(coords) != expected:
        return None

    coord_set = set(coords)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if on_border != ((r, c) in coord_set):
                return None
    return (r0, c0, r1, c1)


def _find_frames(grid: np.ndarray) -> tuple[int, list[dict[str, int | tuple[int, int, int, int]]]]:
    bg = _get_bg(grid)
    frames: list[dict[str, int | tuple[int, int, int, int]]] = []
    for color, coords in _components_by_color(grid, bg):
        bbox = _frame_bbox(coords)
        if bbox is None:
            continue
        frames.append({"color": int(color), "bbox": bbox})
    frames.sort(key=lambda item: (*item["bbox"], int(item["color"])))  # type: ignore[index]
    return bg, frames


def _draw_ring(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _max_layers(inner_height: int, inner_width: int) -> int:
    return (min(inner_height, inner_width) + 1) // 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each rectangular fence interior with alternating concentric rings whose count is determined by the seed pixels inside it.

    Input structure:
    - The background is the dominant color.
    - One or more non-background colors form closed one-pixel-thick rectangular fences.
    - Each fence encloses background cells plus several single-pixel seeds of another color.
    - Outside the fences there may be scattered distractor pixels sharing a fence color.

    Rule:
    - Detect every monochromatic connected component that is exactly a hollow rectangle.
    - For each fence, count the non-background seed pixels inside it and infer the seed color from those interior pixels.
    - Clear the fence interior to background, then draw that many inward-shrinking rectangular rings; the first ring uses the seed color, the next uses the fence color, and the colors alternate until the requested count is exhausted or no inner layer remains.
    - Leave the fence pixels and all pixels outside detected fences unchanged.

    Key insight:
    - The interior seed pixels are only a counter: their quantity chooses how many concentric rings to paint, while their color supplies the first alternating ring color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, frames = _find_frames(g)
    out = g.copy()

    for frame in frames:
        frame_color = int(frame["color"])
        r0, c0, r1, c1 = frame["bbox"]  # type: ignore[misc]
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0:
            continue

        seed_mask = (inner != bg) & (inner != frame_color)
        seed_count = int(seed_mask.sum())
        out[r0 + 1 : r1, c0 + 1 : c1] = bg
        if seed_count == 0:
            continue

        seed_color = int(Counter(int(v) for v in inner[seed_mask].flat).most_common(1)[0][0])
        for layer in range(seed_count):
            rr0 = r0 + 1 + layer
            cc0 = c0 + 1 + layer
            rr1 = r1 - 1 - layer
            cc1 = c1 - 1 - layer
            if rr0 > rr1 or cc0 > cc1:
                break
            ring_color = seed_color if layer % 2 == 0 else frame_color
            _draw_ring(out, rr0, cc0, rr1, cc1, ring_color)

    return out


def _sample_frame_count(rng: random.Random, area: int) -> int:
    options = [1]
    if area >= 220:
        options.extend([2, 2])
    if area >= 360:
        options.extend([3, 3])
    if area >= 520:
        options.append(4)
    return int(rng.choice(options))


def _sample_outer_size(rng: random.Random, height: int, width: int, frame_count: int) -> tuple[int, int]:
    max_h = min(FRAME_MAX_OUTER, height - 2)
    max_w = min(FRAME_MAX_OUTER, width - 2)
    if frame_count >= 4:
        max_h = min(max_h, 8)
        max_w = min(max_w, 8)
    elif frame_count == 3:
        max_h = min(max_h, 9)
        max_w = min(max_w, 9)
    elif frame_count == 2:
        max_h = min(max_h, 10)
        max_w = min(max_w, 10)
    return rng.randint(FRAME_MIN_OUTER, max_h), rng.randint(FRAME_MIN_OUTER, max_w)


def _reserve(blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(blocked.shape[0] - 1, r1 + pad)
    cc1 = min(blocked.shape[1] - 1, c1 + pad)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _seed_slots(inner_height: int, inner_width: int, parity: int) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(inner_height)
        for c in range(inner_width)
        if (r + c) % 2 == parity
    ]


def _sample_seed_count(rng: random.Random, inner_height: int, inner_width: int) -> int:
    max_layers = _max_layers(inner_height, inner_width)
    slots0 = len(_seed_slots(inner_height, inner_width, 0))
    slots1 = len(_seed_slots(inner_height, inner_width, 1))
    capacity = max(slots0, slots1)
    assert capacity >= 1

    saturating_possible = capacity > max_layers
    mode = rng.choice(("fit", "saturate")) if saturating_possible else "fit"
    if mode == "fit":
        return rng.randint(1, min(max_layers, capacity))
    return rng.randint(max_layers + 1, capacity)


def _place_frame(
    rng: random.Random,
    grid: np.ndarray,
    blocked: np.ndarray,
    frame_color: int,
    seed_color: int,
    outer_height: int,
    outer_width: int,
    seed_count: int,
) -> tuple[int, int, int, int] | None:
    height, width = map(int, grid.shape)
    if outer_height + 2 > height or outer_width + 2 > width:
        return None

    for _ in range(FRAME_PLACE_TRIES):
        r0 = rng.randint(1, height - outer_height - 1)
        c0 = rng.randint(1, width - outer_width - 1)
        r1 = r0 + outer_height - 1
        c1 = c0 + outer_width - 1
        if blocked[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2].any():
            continue

        grid[r0, c0 : c1 + 1] = frame_color
        grid[r1, c0 : c1 + 1] = frame_color
        grid[r0 : r1 + 1, c0] = frame_color
        grid[r0 : r1 + 1, c1] = frame_color

        inner_height = outer_height - 2
        inner_width = outer_width - 2
        slots_by_parity = [
            _seed_slots(inner_height, inner_width, 0),
            _seed_slots(inner_height, inner_width, 1),
        ]
        valid_parities = [idx for idx, slots in enumerate(slots_by_parity) if len(slots) >= seed_count]
        parity = int(rng.choice(valid_parities))
        for dr, dc in rng.sample(slots_by_parity[parity], seed_count):
            grid[r0 + 1 + dr, c0 + 1 + dc] = seed_color

        _reserve(blocked, r0, c0, r1, c1, pad=1)
        return (r0, c0, r1, c1)

    return None


def _place_noise(
    rng: random.Random,
    grid: np.ndarray,
    blocked: np.ndarray,
    frame_colors: list[int],
) -> None:
    if not frame_colors:
        return

    height, width = map(int, grid.shape)
    noise_count = rng.randint(0, max(2, len(frame_colors) + 1))
    for _ in range(noise_count):
        shape = rng.choice(NOISE_SHAPES)
        max_r = max(dr for dr, _ in shape)
        max_c = max(dc for _, dc in shape)
        color = int(rng.choice(frame_colors))
        for _ in range(NOISE_PLACE_TRIES):
            r0 = rng.randint(0, height - max_r - 1)
            c0 = rng.randint(0, width - max_c - 1)
            cells = [(r0 + dr, c0 + dc) for dr, dc in shape]
            if any(blocked[r, c] for r, c in cells):
                continue
            for r, c in cells:
                grid[r, c] = color
            _reserve(blocked, r0, c0, r0 + max_r, c0 + max_c, pad=1)
            break


def _valid_background(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(int(v) for v in grid.flat)
    bg_count = int(counts[bg])
    return all(color == bg or int(count) < bg_count for color, count in counts.items())


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*GRID_HEIGHT_RANGE)
    width = rng.randint(*GRID_WIDTH_RANGE)
    bg = rng.randrange(COLOR_COUNT)
    grid = np.full((height, width), bg, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    frame_count = _sample_frame_count(rng, height * width)
    palette = [color for color in range(COLOR_COUNT) if color != bg]
    placed_bboxes: list[tuple[int, int, int, int]] = []
    frame_colors: list[int] = []

    for _ in range(frame_count):
        outer_height, outer_width = _sample_outer_size(rng, height, width, frame_count)
        inner_height = outer_height - 2
        inner_width = outer_width - 2
        seed_count = _sample_seed_count(rng, inner_height, inner_width)

        frame_color = int(rng.choice(palette))
        seed_choices = [color for color in palette if color != frame_color]
        seed_color = int(rng.choice(seed_choices))
        bbox = _place_frame(
            rng,
            grid,
            blocked,
            frame_color,
            seed_color,
            outer_height,
            outer_width,
            seed_count,
        )
        if bbox is None:
            return None
        placed_bboxes.append(bbox)
        frame_colors.append(frame_color)

    _place_noise(rng, grid, blocked, frame_colors)

    if not _valid_background(grid, bg):
        return None

    detected_bg, frames = _find_frames(grid)
    detected_bboxes = sorted(tuple(frame["bbox"]) for frame in frames)
    if detected_bg != bg or detected_bboxes != sorted(placed_bboxes):
        return None

    out = main(grid)
    if out.shape != grid.shape or not np.any(out != grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
