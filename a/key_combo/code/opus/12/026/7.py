from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
ORANGE = 7

FRAME_COUNT_RANGE = (3, 5)
GRID_SIDE_RANGE = (18, 24)
INTERIOR_SIDE_RANGE = (2, 6)
SEED_SHAPES = ((2, 2), (2, 3))
GEN_RETRIES = 400


Frame = tuple[int, int, int, int]


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _component_boxes(mask: np.ndarray) -> list[Frame]:
    """Return row-major bounding boxes of 4-connected True components."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    boxes: list[Frame] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            r0 = r1 = r
            c0 = c1 = c
            while stack:
                rr, cc = stack.pop()
                r0 = min(r0, rr)
                r1 = max(r1, rr)
                c0 = min(c0, cc)
                c1 = max(c1, cc)
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            boxes.append((r0, c0, r1, c1))

    boxes.sort()
    return boxes


def _is_hollow_rect(g: np.ndarray, box: Frame) -> bool:
    r0, c0, r1, c1 = box
    if r1 - r0 < 3 or c1 - c0 < 3:
        return False
    border = np.zeros_like(g, dtype=bool)
    border[r0, c0 : c1 + 1] = True
    border[r1, c0 : c1 + 1] = True
    border[r0 : r1 + 1, c0] = True
    border[r0 : r1 + 1, c1] = True
    inside = (slice(r0 + 1, r1), slice(c0 + 1, c1))
    return bool(
        np.all(g[border] == BLUE)
        and np.all(g[inside] != BLUE)
        and np.array_equal(g[r0 : r1 + 1, c0 : c1 + 1] == BLUE, border[r0 : r1 + 1, c0 : c1 + 1])
    )


def _find_frames(g: np.ndarray) -> list[Frame]:
    frames = _component_boxes(g == BLUE)
    assert frames, "expected at least one blue frame"
    assert all(_is_hollow_rect(g, box) for box in frames), "all blue components must be hollow rectangles"
    return frames


def _extract_seed_mask(interior: np.ndarray) -> np.ndarray | None:
    orange = np.argwhere(interior == ORANGE)
    if orange.size == 0:
        return None
    seed_h = int(orange[:, 0].max()) + 1
    seed_w = int(orange[:, 1].max()) + 1
    seed = interior[:seed_h, :seed_w] == ORANGE
    return seed


def _tile_seed(seed: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    sh, sw = seed.shape
    reps = ((h + sh - 1) // sh, (w + sw - 1) // sw)
    tiled = np.tile(seed.astype(int), reps)[:h, :w]
    out = np.zeros((h, w), dtype=int)
    out[tiled.astype(bool)] = ORANGE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each blue frame by branching on whether its top-left interior contains an orange seed mask.

    Input structure:
    - A black background contains several one-cell-thick blue hollow rectangular frames.
    - Some frame interiors are fully black, while others have a small orange-and-black seed patch anchored at the
      interior's top-left corner.
    - Seeded interiors use only black and orange, and the same seed shape may appear in multiple frames with sizes
      chosen to allow exact tiling.

    Rule:
    - Detect every blue hollow rectangular frame and inspect its interior.
    - If the interior contains orange, take the rectangle from the interior's top-left cell to the furthest orange row
      and column as the seed mask, then tile that binary seed across the full interior using orange for True and black
      for False.
    - If the interior contains no orange, fill it solid red when its area is even and solid yellow when its area is odd.
      Keep all blue borders unchanged.

    Key insight:
    - The same object class splits into two separate transformations: orange presence at the interior's top-left
      identifies a seed-driven tiling branch, while completely empty interiors are solved only by area parity.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for r0, c0, r1, c1 in _find_frames(g):
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        seed = _extract_seed_mask(interior)
        if seed is None:
            fill = RED if interior.size % 2 == 0 else YELLOW
            out[r0 + 1 : r1, c0 + 1 : c1] = fill
        else:
            out[r0 + 1 : r1, c0 + 1 : c1] = _tile_seed(seed, interior.shape)

    return out


def _sample_seed_mask(shape: tuple[int, int]) -> np.ndarray:
    seed_h, seed_w = shape
    for _ in range(100):
        seed = np.zeros((seed_h, seed_w), dtype=bool)
        count = random.randint(1, seed_h * seed_w - 1)
        picks = random.sample(range(seed_h * seed_w), count)
        seed.reshape(-1)[picks] = True
        if not seed[-1].any():
            continue
        if not seed[:, -1].any():
            continue
        if seed.all() or not seed.any():
            continue
        return seed
    raise RuntimeError("failed to sample a valid seed mask")


def _choose_seeded_count(frame_count: int) -> int:
    if frame_count == 3:
        return 1
    if frame_count == 4:
        return 2
    return random.choice((2, 3))


def _seed_size_options(seed_h: int, seed_w: int) -> list[tuple[int, int]]:
    options = []
    for h in range(INTERIOR_SIDE_RANGE[0], INTERIOR_SIDE_RANGE[1] + 1):
        for w in range(INTERIOR_SIDE_RANGE[0], INTERIOR_SIDE_RANGE[1] + 1):
            if h % seed_h == 0 and w % seed_w == 0:
                options.append((h, w))
    return options


def _random_dims_with_parity(parity: int) -> tuple[int, int]:
    for _ in range(100):
        h = random.randint(*INTERIOR_SIDE_RANGE)
        w = random.randint(*INTERIOR_SIDE_RANGE)
        if (h * w) % 2 == parity:
            return h, w
    raise RuntimeError("failed to sample interior dimensions with required parity")


def _choose_frame_specs() -> tuple[np.ndarray, list[dict[str, object]]]:
    seed = _sample_seed_mask(random.choice(SEED_SHAPES))
    seed_h, seed_w = seed.shape
    seeded_options = _seed_size_options(seed_h, seed_w)
    larger_seeded = [dims for dims in seeded_options if dims != (seed_h, seed_w)]
    assert larger_seeded, "seed options must include a visibly tiled case"

    frame_count = random.randint(*FRAME_COUNT_RANGE)
    seeded_count = _choose_seeded_count(frame_count)
    empty_count = frame_count - seeded_count

    specs: list[dict[str, object]] = []
    larger_used = False
    for idx in range(seeded_count):
        choices = larger_seeded if idx == seeded_count - 1 and not larger_used else seeded_options
        inner_h, inner_w = random.choice(choices)
        larger_used = larger_used or (inner_h, inner_w) != (seed_h, seed_w)
        specs.append({"mode": "seed", "inner_h": inner_h, "inner_w": inner_w, "seed": seed.copy()})

    parity_targets: list[int] = []
    if empty_count >= 2:
        parity_targets.extend([0, 1])
    parity_targets.extend(random.randint(0, 1) for _ in range(empty_count - len(parity_targets)))
    random.shuffle(parity_targets)
    for parity in parity_targets:
        inner_h, inner_w = _random_dims_with_parity(parity)
        specs.append({"mode": "parity", "inner_h": inner_h, "inner_w": inner_w})

    random.shuffle(specs)
    return seed, specs


def _boxes_clear(new_box: Frame, boxes: list[Frame]) -> bool:
    r0, c0, r1, c1 = new_box
    for pr0, pc0, pr1, pc1 in boxes:
        if not (r1 + 1 < pr0 or pr1 + 1 < r0 or c1 + 1 < pc0 or pc1 + 1 < c0):
            return False
    return True


def _place_boxes(outer_sizes: list[tuple[int, int]], h: int, w: int) -> list[tuple[int, int]] | None:
    order = sorted(range(len(outer_sizes)), key=lambda i: outer_sizes[i][0] * outer_sizes[i][1], reverse=True)
    placed_boxes: list[Frame] = []
    placed_positions: list[tuple[int, int] | None] = [None] * len(outer_sizes)

    for idx in order:
        box_h, box_w = outer_sizes[idx]
        rows = list(range(1, h - box_h))
        cols = list(range(1, w - box_w))
        if not rows or not cols:
            return None
        candidates = [(r, c) for r in rows for c in cols]
        random.shuffle(candidates)
        for r, c in candidates:
            box = (r, c, r + box_h - 1, c + box_w - 1)
            if _boxes_clear(box, placed_boxes):
                placed_boxes.append(box)
                placed_positions[idx] = (r, c)
                break
        else:
            return None

    assert all(pos is not None for pos in placed_positions)
    return [pos for pos in placed_positions if pos is not None]


def _render_input(shape: tuple[int, int], specs: list[dict[str, object]], positions: list[tuple[int, int]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for spec, (r0, c0) in zip(specs, positions):
        inner_h = int(spec["inner_h"])
        inner_w = int(spec["inner_w"])
        r1 = r0 + inner_h + 1
        c1 = c0 + inner_w + 1
        grid[r0, c0 : c1 + 1] = BLUE
        grid[r1, c0 : c1 + 1] = BLUE
        grid[r0 : r1 + 1, c0] = BLUE
        grid[r0 : r1 + 1, c1] = BLUE
        if spec["mode"] == "seed":
            seed = np.asarray(spec["seed"], dtype=bool)
            patch = np.zeros(seed.shape, dtype=int)
            patch[seed] = ORANGE
            grid[r0 + 1 : r0 + 1 + seed.shape[0], c0 + 1 : c0 + 1 + seed.shape[1]] = patch
    return grid


def _expected_from_specs(shape: tuple[int, int], specs: list[dict[str, object]], positions: list[tuple[int, int]]) -> np.ndarray:
    out = _render_input(shape, specs, positions)
    for spec, (r0, c0) in zip(specs, positions):
        inner_h = int(spec["inner_h"])
        inner_w = int(spec["inner_w"])
        area = inner_h * inner_w
        if spec["mode"] == "seed":
            seed = np.asarray(spec["seed"], dtype=bool)
            out[r0 + 1 : r0 + 1 + inner_h, c0 + 1 : c0 + 1 + inner_w] = _tile_seed(seed, (inner_h, inner_w))
        else:
            out[r0 + 1 : r0 + 1 + inner_h, c0 + 1 : c0 + 1 + inner_w] = RED if area % 2 == 0 else YELLOW
    return out


def _validate_generation(
    inp: np.ndarray,
    seed: np.ndarray,
    specs: list[dict[str, object]],
    positions: list[tuple[int, int]],
) -> bool:
    try:
        if inp.ndim != 2 or max(inp.shape) > 30:
            return False
        if not set(np.unique(inp)).issubset({BLACK, BLUE, ORANGE}):
            return False

        if len(specs) < FRAME_COUNT_RANGE[0] or len(specs) > FRAME_COUNT_RANGE[1]:
            return False

        seed_masks = [np.asarray(spec["seed"], dtype=bool) for spec in specs if spec["mode"] == "seed"]
        if not seed_masks or len(seed_masks) == len(specs):
            return False
        if any(not np.array_equal(mask, seed) for mask in seed_masks):
            return False

        if not any(
            (int(spec["inner_h"]), int(spec["inner_w"])) != seed.shape
            for spec in specs
            if spec["mode"] == "seed"
        ):
            return False

        parity_modes = [int(spec["inner_h"]) * int(spec["inner_w"]) % 2 for spec in specs if spec["mode"] == "parity"]
        if len(parity_modes) >= 2 and set(parity_modes) != {0, 1}:
            return False

        boxes = []
        for spec, (r0, c0) in zip(specs, positions):
            box = (r0, c0, r0 + int(spec["inner_h"]) + 1, c0 + int(spec["inner_w"]) + 1)
            if not _boxes_clear(box, boxes):
                return False
            boxes.append(box)

        expected = _expected_from_specs(inp.shape, specs, positions)
        if np.array_equal(inp, expected):
            return False
        if not np.array_equal(main(inp), expected):
            return False

        frames = _find_frames(inp)
        if len(frames) != len(specs):
            return False
        if any(np.any(inp[r0 + 1 : r1, c0 + 1 : c1] == BLUE) for r0, c0, r1, c1 in frames):
            return False
        return True
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    seed, specs = _choose_frame_specs()
    outer_sizes = [(int(spec["inner_h"]) + 2, int(spec["inner_w"]) + 2) for spec in specs]

    for _ in range(60):
        shape = (
            random.randint(*GRID_SIDE_RANGE),
            random.randint(*GRID_SIDE_RANGE),
        )
        positions = _place_boxes(outer_sizes, *shape)
        if positions is None:
            continue
        inp = _render_input(shape, specs, positions)
        if _validate_generation(inp, seed, specs, positions):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
