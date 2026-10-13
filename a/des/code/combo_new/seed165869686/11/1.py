import random
from collections import deque

import numpy as np


SHAPES = [
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
]

PATTERNS = [
    (1, 0, 1),
    (1, 1, 0, 1),
    (1, 0, 0, 1),
    (1, 0, 1, 1),
    (1, 1, 0, 0, 1),
    (1, 0, 1, 0, 1),
]

LEGEND_HEIGHT = 5
LEGEND_ACTIVE_HEIGHT = 3
LEGEND_PAIR_GAP = 2
GEN_RETRIES = 300


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each rectangular border with the legend texture selected by its interior seed shape.

    Input structure:
    - The background is the dominant color.
    - Several one-cell-thick rectangular frames of one foreground color sit away from the grid edge.
    - Each frame contains exactly one connected seed shape in another foreground color.
    - A legend band on the top or bottom contains several pairs: a seed-color sample shape and, immediately to its right, a short border-color texture strip with visible gaps.

    Rule:
    - Find the solid rectangular frames and the unique seed component inside each one.
    - Read every legend pair as a mapping from the sample shape (matched up to translation only) to the strip's one-dimensional on/off pattern.
    - For each frame, match its interior seed shape to the legend sample with the same shape.
    - Walk around that frame's border clockwise starting from the top-left corner, repeat the matched strip pattern periodically, keep border-color cells where the pattern is on, and turn border cells into background where the pattern has a gap.
    - Leave the seed shapes and the legend band unchanged.

    Key insight:
    - The seed shape is not decoration; it is a lookup key that chooses which repeating border texture should be wrapped around the enclosing rectangle.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    border_color, frames = _find_frames(g, bg)
    seed_color = _find_seed_color(g, bg, border_color, frames)
    shape_to_pattern, frame_seeds = _read_legend(g, border_color, seed_color, frames)

    out = g.copy()
    for idx, frame in enumerate(frames):
        key = _normalize_shape(frame_seeds[idx])
        pattern = shape_to_pattern[key]
        _apply_pattern(out, frame["bbox"], border_color, bg, pattern)
    return out


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _normalize_shape(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(coords)
    shifted = sorted((int(r - r0), int(c - c0)) for r, c in coords)
    return tuple(shifted)


def _shape_size(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _is_frame_component(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 4 or c1 - c0 < 4:
        return False
    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0 + 1, r1):
        expected.add((r, c0))
        expected.add((r, c1))
    actual = {tuple(map(int, cell)) for cell in coords}
    return actual == expected


def _find_frames(grid: np.ndarray, bg: int) -> tuple[int, list[dict[str, object]]]:
    candidates: list[tuple[int, list[dict[str, object]]]] = []
    for color in (int(v) for v in np.unique(grid) if int(v) != bg):
        frames = []
        for comp in _components(grid == color):
            if _is_frame_component(comp):
                frames.append({"bbox": _bbox(comp), "coords": comp})
        if frames:
            frames.sort(key=lambda item: tuple(item["bbox"]))  # deterministic
            candidates.append((color, frames))
    assert candidates, "expected at least one solid rectangular frame color"
    border_color, frames = max(candidates, key=lambda item: len(item[1]))
    return border_color, frames


def _find_seed_color(
    grid: np.ndarray, bg: int, border_color: int, frames: list[dict[str, object]]
) -> int:
    interior_mask = np.zeros_like(grid, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        interior_mask[r0 + 1 : r1, c0 + 1 : c1] = True
    candidates = [int(v) for v in np.unique(grid) if int(v) not in (bg, border_color)]
    assert candidates, "expected a separate seed color"
    seed_color = max(candidates, key=lambda color: int(np.sum((grid == color) & interior_mask)))
    assert np.any((grid == seed_color) & interior_mask), "seed color must appear inside a frame"
    return seed_color


def _frame_index_for_component(
    coords: np.ndarray, frames: list[dict[str, object]]
) -> int | None:
    owner = None
    for idx, frame in enumerate(frames):
        r0, c0, r1, c1 = frame["bbox"]
        inside = (
            np.all(coords[:, 0] > r0)
            and np.all(coords[:, 0] < r1)
            and np.all(coords[:, 1] > c0)
            and np.all(coords[:, 1] < c1)
        )
        if inside:
            assert owner is None, "seed component should belong to exactly one frame"
            owner = idx
    return owner


def _read_legend(
    grid: np.ndarray, border_color: int, seed_color: int, frames: list[dict[str, object]]
) -> tuple[dict[tuple[tuple[int, int], ...], tuple[int, ...]], dict[int, np.ndarray]]:
    seed_components = _components(grid == seed_color)
    legend_samples: list[np.ndarray] = []
    frame_seeds: dict[int, np.ndarray] = {}
    for comp in seed_components:
        owner = _frame_index_for_component(comp, frames)
        if owner is None:
            legend_samples.append(comp)
        else:
            assert owner not in frame_seeds, "each frame should contain exactly one seed component"
            frame_seeds[owner] = comp

    assert legend_samples, "expected legend seed samples outside the frames"
    assert len(frame_seeds) == len(frames), "every frame needs exactly one seed component"

    legend_border_mask = grid == border_color
    for frame in frames:
        coords = frame["coords"]
        legend_border_mask[coords[:, 0], coords[:, 1]] = False
    legend_border_coords = np.argwhere(legend_border_mask)
    assert len(legend_border_coords) > 0, "expected border-color legend strips"

    sample_rows = np.concatenate([comp[:, 0] for comp in legend_samples])
    legend_r0 = int(min(sample_rows.min(), legend_border_coords[:, 0].min()))
    legend_r1 = int(max(sample_rows.max(), legend_border_coords[:, 0].max()))

    legend_samples.sort(key=lambda comp: (_bbox(comp)[1], _bbox(comp)[0]))
    sample_bboxes = [_bbox(comp) for comp in legend_samples]

    mapping: dict[tuple[tuple[int, int], ...], tuple[int, ...]] = {}
    for idx, comp in enumerate(legend_samples):
        _r0, _c0, _r1, c1 = sample_bboxes[idx]
        next_c0 = sample_bboxes[idx + 1][1] if idx + 1 < len(sample_bboxes) else grid.shape[1]
        window = legend_border_mask[legend_r0 : legend_r1 + 1, c1 + 1 : next_c0]
        cols = np.where(np.any(window, axis=0))[0]
        assert cols.size > 0, "each legend sample must have a strip to its right"
        start = c1 + 1 + int(cols[0])
        end = c1 + 1 + int(cols[-1])
        pattern = tuple(
            1 if np.any(legend_border_mask[legend_r0 : legend_r1 + 1, col]) else 0
            for col in range(start, end + 1)
        )
        key = _normalize_shape(comp)
        assert key not in mapping, "legend sample shapes must be unique"
        mapping[key] = pattern
    return mapping, frame_seeds


def _perimeter_path(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    path = [(r0, c) for c in range(c0, c1 + 1)]
    path.extend((r, c1) for r in range(r0 + 1, r1))
    path.extend((r1, c) for c in range(c1, c0 - 1, -1))
    path.extend((r, c0) for r in range(r1 - 1, r0, -1))
    return path


def _apply_pattern(
    out: np.ndarray,
    bbox: tuple[int, int, int, int],
    border_color: int,
    bg: int,
    pattern: tuple[int, ...],
) -> None:
    path = _perimeter_path(bbox)
    period = len(pattern)
    assert period > 0, "legend strip pattern must be non-empty"
    for idx, (r, c) in enumerate(path):
        out[r, c] = border_color if pattern[idx % period] else bg


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg, border_color, seed_color = rng.sample(range(10), 3)
    legend_top = rng.random() < 0.5

    n_templates = rng.choice([2, 3])
    chosen_shapes = rng.sample(SHAPES, n_templates)
    chosen_patterns = rng.sample(PATTERNS, n_templates)
    pairs = list(zip(chosen_shapes, chosen_patterns))
    rng.shuffle(pairs)

    legend_width = _legend_width(pairs)
    n_frames = rng.randint(n_templates, 4)
    frame_template_ids = list(range(n_templates))
    while len(frame_template_ids) < n_frames:
        frame_template_ids.append(rng.randrange(n_templates))
    rng.shuffle(frame_template_ids)

    frame_sizes = []
    for shape_id in frame_template_ids:
        sh, sw = _shape_size(chosen_shapes[shape_id])
        interior_h = rng.randint(sh + 2, min(sh + 5, 8))
        interior_w = rng.randint(sw + 2, min(sw + 5, 9))
        frame_sizes.append((interior_h + 2, interior_w + 2))

    min_body_w = max(legend_width + 2, max(w for _, w in frame_sizes) + 2, 18)
    max_body_w = 29
    if min_body_w > max_body_w:
        return None
    body_w = rng.randint(min_body_w, max_body_w)

    min_body_h = max(max(h for h, _ in frame_sizes) + 2, 10)
    max_body_h = 24
    if min_body_h > max_body_h:
        return None
    body_h = rng.randint(min_body_h, max_body_h)

    total_h = body_h + 1 + LEGEND_HEIGHT
    if total_h > 30:
        return None

    placements = _place_frames(body_h, body_w, frame_sizes, rng)
    if placements is None:
        return None

    grid = np.full((total_h, body_w), bg, dtype=int)
    body_row0 = LEGEND_HEIGHT + 1 if legend_top else 0
    legend_row0 = 0 if legend_top else body_h + 1
    _draw_frames_and_seeds(
        grid,
        body_row0,
        placements,
        frame_sizes,
        frame_template_ids,
        chosen_shapes,
        border_color,
        seed_color,
        rng,
    )
    _draw_legend(grid, legend_row0, pairs, border_color, seed_color, rng)

    if not _generated_layout_is_valid(grid, bg, border_color, seed_color, n_frames, n_templates):
        return None
    return grid


def _legend_width(
    pairs: list[tuple[tuple[tuple[int, int], ...], tuple[int, ...]]]
) -> int:
    width = 0
    for idx, (shape, pattern) in enumerate(pairs):
        _sh, sw = _shape_size(shape)
        width += sw + 1 + len(pattern)
        if idx + 1 < len(pairs):
            width += LEGEND_PAIR_GAP
    return width


def _place_frames(
    body_h: int, body_w: int, frame_sizes: list[tuple[int, int]], rng: random.Random
) -> list[tuple[int, int]] | None:
    occupied = np.zeros((body_h, body_w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(frame_sizes)
    order = sorted(
        range(len(frame_sizes)),
        key=lambda idx: frame_sizes[idx][0] * frame_sizes[idx][1],
        reverse=True,
    )
    for idx in order:
        fh, fw = frame_sizes[idx]
        if body_h - fh - 1 < 1 or body_w - fw - 1 < 1:
            return None
        placed = False
        for _ in range(200):
            top = rng.randint(1, body_h - fh - 1)
            left = rng.randint(1, body_w - fw - 1)
            r0 = max(0, top - 1)
            c0 = max(0, left - 1)
            r1 = min(body_h, top + fh + 1)
            c1 = min(body_w, left + fw + 1)
            if occupied[r0:r1, c0:c1].any():
                continue
            occupied[r0:r1, c0:c1] = True
            placements[idx] = (top, left)
            placed = True
            break
        if not placed:
            return None
    return [placement for placement in placements if placement is not None]


def _draw_frames_and_seeds(
    grid: np.ndarray,
    body_row0: int,
    placements: list[tuple[int, int]],
    frame_sizes: list[tuple[int, int]],
    frame_template_ids: list[int],
    shapes: list[tuple[tuple[int, int], ...]],
    border_color: int,
    seed_color: int,
    rng: random.Random,
) -> None:
    for idx, (top, left) in enumerate(placements):
        fh, fw = frame_sizes[idx]
        top += body_row0
        bottom = top + fh - 1
        right = left + fw - 1

        grid[top, left : right + 1] = border_color
        grid[bottom, left : right + 1] = border_color
        grid[top : bottom + 1, left] = border_color
        grid[top : bottom + 1, right] = border_color

        shape = shapes[frame_template_ids[idx]]
        sh, sw = _shape_size(shape)
        seed_top = top + 1 + rng.randint(0, fh - 2 - sh)
        seed_left = left + 1 + rng.randint(0, fw - 2 - sw)
        for dr, dc in shape:
            grid[seed_top + dr, seed_left + dc] = seed_color


def _draw_legend(
    grid: np.ndarray,
    legend_row0: int,
    pairs: list[tuple[tuple[tuple[int, int], ...], tuple[int, ...]]],
    border_color: int,
    seed_color: int,
    rng: random.Random,
) -> None:
    legend_width = _legend_width(pairs)
    start_col = rng.randint(1, grid.shape[1] - legend_width - 1)
    active_top = legend_row0 + 1
    x = start_col
    for idx, (shape, pattern) in enumerate(pairs):
        sh, sw = _shape_size(shape)
        row_offset = rng.randint(0, LEGEND_ACTIVE_HEIGHT - sh)
        for dr, dc in shape:
            grid[active_top + row_offset + dr, x + dc] = seed_color
        strip_left = x + sw + 1
        for offset, bit in enumerate(pattern):
            if bit:
                grid[active_top : active_top + LEGEND_ACTIVE_HEIGHT, strip_left + offset] = border_color
        x = strip_left + len(pattern)
        if idx + 1 < len(pairs):
            x += LEGEND_PAIR_GAP


def _generated_layout_is_valid(
    grid: np.ndarray,
    bg: int,
    border_color: int,
    seed_color: int,
    n_frames: int,
    n_templates: int,
) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if len(np.unique(grid)) != 3:
        return False
    if _get_bg(grid) != bg:
        return False
    if np.count_nonzero(grid == border_color) <= np.count_nonzero(grid == seed_color):
        return False
    border, frames = _find_frames(grid, bg)
    if border != border_color or len(frames) != n_frames:
        return False
    seed = _find_seed_color(grid, bg, border_color, frames)
    if seed != seed_color:
        return False
    mapping, frame_seeds = _read_legend(grid, border_color, seed_color, frames)
    if len(mapping) != n_templates or len(frame_seeds) != n_frames:
        return False
    for key in mapping:
        if key not in {_normalize_shape(comp) for comp in frame_seeds.values()} and len(frame_seeds) < len(mapping):
            return False
    return True
