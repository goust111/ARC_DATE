from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import py_compile
import random

import numpy as np


BG = 0
TILE_SHAPE = (3, 3)
OUTPUT_SHAPE = (9, 9)
LARGE_FRAME_SHAPE = (5, 5)
SMALL_FRAME_SHAPES = ((3, 3), (3, 4), (4, 3), (4, 4))
GRID_SIZE_RANGE = (15, 22)
FRAME_COUNT_RANGE = (2, 4)
COLOR_POOL = tuple(range(1, 10))
PLACEMENT_GAP = 1
MAX_GRID_SIZE = 30
MAX_GENERATION_ATTEMPTS = 500
SELF_TEST_TRIALS = 100


@dataclass(frozen=True)
class Frame:
    color: int
    bbox: tuple[int, int, int, int]

    @property
    def height(self) -> int:
        return self.bbox[2] - self.bbox[0] + 1

    @property
    def width(self) -> int:
        return self.bbox[3] - self.bbox[1] + 1

    @property
    def area(self) -> int:
        return self.height * self.width


@dataclass(frozen=True)
class Case:
    input_grid: np.ndarray
    output_grid: np.ndarray
    signature: tuple[object, ...]


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _connected_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _is_rectangular_frame(coords: np.ndarray) -> bool:
    if coords.size == 0:
        return False
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return False

    mask = np.zeros((height, width), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True

    expected = np.zeros_like(mask)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return bool(np.array_equal(mask, expected))


def _find_frames(grid: np.ndarray) -> list[Frame]:
    g = np.asarray(grid, dtype=int)
    frames: list[Frame] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != BG):
        for coords in _connected_components(g, color):
            if _is_rectangular_frame(coords):
                frames.append(Frame(color=color, bbox=_bbox_from_coords(coords)))
    return frames


def _tile_rare_color(tile: np.ndarray) -> int:
    flat = [int(v) for v in tile.ravel()]
    counts = Counter(flat)
    rare_count = min(counts.values())
    rare_colors = {color for color, count in counts.items() if count == rare_count}
    for value in flat:
        if value in rare_colors:
            return value
    raise AssertionError("tile must contain at least one color")


def _stamp_tile(tile: np.ndarray) -> np.ndarray:
    tile = np.asarray(tile, dtype=int)
    assert tile.shape == TILE_SHAPE
    out = np.full(OUTPUT_SHAPE, BG, dtype=int)
    rare_color = _tile_rare_color(tile)
    tile_h, tile_w = tile.shape
    for r, c in np.argwhere(tile == rare_color):
        r0 = int(r) * tile_h
        c0 = int(c) * tile_w
        out[r0 : r0 + tile_h, c0 : c0 + tile_w] = tile
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the largest rectangular frame, crop its interior tile, and stamp that tile on a 9x9 canvas at the least-frequent-color positions.

    Input structure:
    - The grid has a black background.
    - There are 2 to 4 non-overlapping monochrome rectangular frames with one-pixel-thick borders.
    - The unique largest frame encloses a 3x3 non-black multicolor tile.
    - Smaller frames may be empty or contain distractor pixels that do not matter.

    Rule:
    - Find every monochrome component that is exactly a hollow rectangle border.
    - Choose the frame with the largest bounding-box area.
    - Crop the 3x3 interior of that frame as the tile.
    - Find the least frequent tile color; if several tie, choose the one whose first occurrence is earliest in row-major order.
    - Create a 9x9 black output split into a 3x3 array of 3x3 blocks, and copy the whole tile into each block whose corresponding tile cell has that rare color.

    Key insight:
    - Frame detection decides which local patch becomes the tile, and the tile's own color statistics then decide where whole-tile stamps are placed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    frames = _find_frames(g)
    assert frames, "expected at least one rectangular frame"
    frames.sort(key=lambda frame: (frame.area, -frame.bbox[0], -frame.bbox[1]))
    largest = frames[-1]

    r0, c0, r1, c1 = largest.bbox
    tile = g[r0 + 1 : r1, c0 + 1 : c1]
    assert tile.shape == TILE_SHAPE, "largest frame interior must be 3x3"
    assert np.all(tile != BG), "tile must not contain black"

    return _stamp_tile(tile)


def _make_frame_patch(shape: tuple[int, int], color: int) -> np.ndarray:
    height, width = shape
    patch = np.full((height, width), BG, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    return patch


def _place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    grid[top : top + h, left : left + w] = patch


def _boxes_have_gap(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
    gap: int = PLACEMENT_GAP,
) -> bool:
    g = int(gap)
    return box_a[2] + g < box_b[0] or box_b[2] + g < box_a[0] or box_a[3] + g < box_b[1] or box_b[3] + g < box_a[1]


def _sample_bbox(
    rng: random.Random,
    grid_shape: tuple[int, int],
    frame_shape: tuple[int, int],
    existing: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    height, width = grid_shape
    fh, fw = frame_shape
    candidates = [
        (r, c, r + fh - 1, c + fw - 1)
        for r in range(height - fh + 1)
        for c in range(width - fw + 1)
    ]
    rng.shuffle(candidates)
    for bbox in candidates:
        if all(_boxes_have_gap(bbox, other) for other in existing):
            return bbox
    return None


def _rare_count_vectors(n_colors: int, rare_count: int) -> list[tuple[int, ...]]:
    assert n_colors in {2, 3}
    if n_colors == 2:
        other = 9 - rare_count
        return [(rare_count, other)] if other > rare_count else []

    vectors: list[tuple[int, ...]] = []
    for second in range(rare_count + 1, 9 - rare_count):
        third = 9 - rare_count - second
        if third > rare_count:
            vectors.append((rare_count, second, third))
    return vectors


def _sample_tile(rng: random.Random, colors: tuple[int, ...]) -> np.ndarray:
    n_colors = len(colors)
    rare_count = rng.choice((1, 2))
    count_vectors = _rare_count_vectors(n_colors, rare_count)
    assert count_vectors
    counts = list(rng.choice(count_vectors))
    cells: list[int] = []
    for color, count in zip(colors, counts):
        cells.extend([int(color)] * int(count))
    rng.shuffle(cells)
    tile = np.array(cells, dtype=int).reshape(TILE_SHAPE)
    assert _tile_rare_color(tile) == int(colors[0])
    return tile


def _fill_small_frame_interior(
    rng: random.Random,
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    palette: tuple[int, ...],
) -> None:
    r0, c0, r1, c1 = bbox
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if rng.random() < 0.45:
                grid[r, c] = int(rng.choice(palette))


def _signature(
    grid_shape: tuple[int, int],
    tile: np.ndarray,
    frames: list[Frame],
) -> tuple[object, ...]:
    largest = max(frames, key=lambda frame: frame.area)
    small_shapes = tuple(sorted((frame.height, frame.width) for frame in frames if frame != largest))
    rare_positions = tuple((int(r), int(c)) for r, c in np.argwhere(tile == _tile_rare_color(tile)))
    counts = tuple(sorted(Counter(int(v) for v in tile.ravel()).values()))
    return (
        int(grid_shape[0]),
        int(grid_shape[1]),
        len(frames),
        small_shapes,
        counts,
        rare_positions,
        tuple(int(v) for v in tile.ravel()),
    )


def _make_case(rng: random.Random) -> Case:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        height = rng.randint(*GRID_SIZE_RANGE)
        width = rng.randint(*GRID_SIZE_RANGE)
        frame_count = rng.randint(*FRAME_COUNT_RANGE)
        n_tile_colors = rng.choice((2, 3))

        needed_colors = frame_count + n_tile_colors
        palette = rng.sample(COLOR_POOL, needed_colors)
        frame_colors = tuple(palette[:frame_count])
        tile_colors = tuple(palette[frame_count:])

        tile = _sample_tile(rng, tile_colors)
        output = _stamp_tile(tile)

        grid = np.full((height, width), BG, dtype=int)
        bboxes: list[tuple[int, int, int, int]] = []

        large_bbox = _sample_bbox(rng, (height, width), LARGE_FRAME_SHAPE, bboxes)
        if large_bbox is None:
            continue
        bboxes.append(large_bbox)
        frames = [Frame(color=int(frame_colors[0]), bbox=large_bbox)]

        smaller_shapes = [rng.choice(SMALL_FRAME_SHAPES) for _ in range(frame_count - 1)]
        valid = True
        for idx, shape in enumerate(smaller_shapes, start=1):
            bbox = _sample_bbox(rng, (height, width), shape, bboxes)
            if bbox is None:
                valid = False
                break
            bboxes.append(bbox)
            frames.append(Frame(color=int(frame_colors[idx]), bbox=bbox))
        if not valid:
            continue

        for frame in frames:
            r0, c0, r1, c1 = frame.bbox
            _place_patch(grid, _make_frame_patch((frame.height, frame.width), frame.color), r0, c0)

        lr0, lc0, _, _ = large_bbox
        grid[lr0 + 1 : lr0 + 4, lc0 + 1 : lc0 + 4] = tile

        for frame in frames[1:]:
            _fill_small_frame_interior(rng, grid, frame.bbox, tile_colors)

        predicted = main(grid)
        if not np.array_equal(predicted, output):
            continue

        signature = _signature((height, width), tile, frames)
        return Case(input_grid=grid, output_grid=output, signature=signature)

    raise RuntimeError("failed to generate a valid case")


def generate_input() -> np.ndarray:
    seed = random.randrange(1 << 30)
    return _make_case(random.Random(seed)).input_grid


def _validate_random_cases(trials: int = SELF_TEST_TRIALS) -> dict[str, int]:
    variants: set[tuple[object, ...]] = set()
    for trial in range(trials):
        case = _make_case(random.Random(20_000 + trial))
        assert case.input_grid.shape[0] <= MAX_GRID_SIZE and case.input_grid.shape[1] <= MAX_GRID_SIZE
        assert case.output_grid.shape == OUTPUT_SHAPE
        predicted = main(case.input_grid)
        assert np.array_equal(predicted, case.output_grid), f"mismatch on trial {trial}"
        variants.add(case.signature)
    assert len(variants) >= 20, f"expected at least 20 variants, got {len(variants)}"
    return {"trials": trials, "variants": len(variants)}


def _compile_self() -> None:
    py_compile.compile(str(Path(__file__)), doraise=True)


def _cleanup_pycache() -> None:
    pycache_dir = Path(__file__).with_name("__pycache__")
    if not pycache_dir.exists():
        return
    stem = Path(__file__).stem
    for pyc_path in pycache_dir.glob(f"{stem}.*.pyc"):
        pyc_path.unlink()
    try:
        pycache_dir.rmdir()
    except OSError:
        pass


if __name__ == "__main__":
    stats = _validate_random_cases()
    _compile_self()
    _cleanup_pycache()
    print(f"validated={stats['trials']} variants={stats['variants']}")
