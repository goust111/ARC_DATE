from __future__ import annotations

import numpy as np


_RNG = np.random.default_rng()
_NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_BG = 0
_FRAME_COLOR = 5
_L_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
_FRAME_SIZE_CHOICES = np.array([3, 4, 5, 6, 7], dtype=int)
_MAX_RETRIES = 300


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r0 in range(h):
        for c0 in range(w):
            color = int(grid[r0, c0])
            if color == bg or seen[r0, c0]:
                continue

            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []

            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in _NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            rows = arr[:, 0]
            cols = arr[:, 1]
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (
                        int(rows.min()),
                        int(cols.min()),
                        int(rows.max()),
                        int(cols.max()),
                    ),
                    "area": int(arr.shape[0]),
                }
            )

    return components


def _component_mask(component: dict) -> np.ndarray:
    r0, c0, r1, c1 = component["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    coords = component["coords"]
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _is_hollow_square(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h != w or h < 4:
        return False
    expected = np.zeros_like(mask)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return bool(np.array_equal(mask, expected))


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[bytes, tuple[int, int]]] = set()

    for base in (mask, np.fliplr(mask)):
        for k in range(4):
            variant = np.rot90(base, k)
            key = (variant.tobytes(), tuple(map(int, variant.shape)))
            if key not in seen:
                seen.add(key)
                variants.append(variant)

    return variants


def _is_top_left_l(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 2 or w < 2:
        return False
    if not bool(mask[0, :].all()) or not bool(mask[:, 0].all()):
        return False
    if int(mask.sum()) != h + w - 1:
        return False
    return not bool(mask[1:, 1:].any())


def _normalize_l(mask: np.ndarray) -> np.ndarray | None:
    candidates: list[np.ndarray] = []
    for variant in _dihedral_variants(mask):
        if _is_top_left_l(variant):
            candidates.append(variant)

    if not candidates:
        return None

    preferred = [cand for cand in candidates if cand.shape[0] >= cand.shape[1]]
    pool = preferred if preferred else candidates
    pool.sort(key=lambda m: (-int(m.shape[0]), int(m.shape[1]), m.tobytes()))
    return pool[0]


def _paint_mask(grid: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> None:
    grid[top:top + mask.shape[0], left:left + mask.shape[1]][mask] = int(color)


def _standard_l_mask(long_len: int, short_len: int) -> np.ndarray:
    mask = np.zeros((long_len, short_len), dtype=bool)
    mask[0, :] = True
    mask[:, 0] = True
    return mask


def _apply_transform(mask: np.ndarray, code: int) -> np.ndarray:
    transformed = np.rot90(mask, int(code % 4))
    if code >= 4:
        transformed = np.fliplr(transformed)
    return transformed


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _choose_frame_sizes() -> list[int]:
    for _ in range(200):
        count = int(_RNG.integers(2, 5))
        sizes = sorted(map(int, _RNG.choice(_FRAME_SIZE_CHOICES, size=count, replace=False)))
        base_height = sum(size + 2 for size in sizes) + (count - 1)
        if base_height <= 30:
            return sizes
    raise RuntimeError("failed to choose frame sizes")


def _frame_layout(sizes: list[int]) -> tuple[int, int, list[tuple[int, int, int]]]:
    left = int(_RNG.integers(0, 3))
    base_height = sum(size + 2 for size in sizes) + (len(sizes) - 1)
    slack = int(_RNG.integers(0, 31 - base_height))
    extras = _RNG.multinomial(slack, np.ones(len(sizes) + 1) / (len(sizes) + 1))

    frames: list[tuple[int, int, int]] = []
    row = int(extras[0])
    for i, size in enumerate(sizes):
        frames.append((row, left, int(size)))
        row += size + 2
        if i < len(sizes) - 1:
            row += 1 + int(extras[i + 1])

    height = row + int(extras[-1])
    return int(height), left, frames


def _render_frames(height: int, width: int, frames: list[tuple[int, int, int]]) -> np.ndarray:
    grid = np.full((height, width), _BG, dtype=int)
    for top, left, size in frames:
        outer = size + 2
        grid[top, left:left + outer] = _FRAME_COLOR
        grid[top + outer - 1, left:left + outer] = _FRAME_COLOR
        grid[top:top + outer, left] = _FRAME_COLOR
        grid[top:top + outer, left + outer - 1] = _FRAME_COLOR
    return grid


def _build_jobs(sizes: list[int]) -> list[dict]:
    active_count = int(_RNG.integers(1, len(sizes) + 1))
    active_sizes = list(map(int, _RNG.choice(np.array(sizes, dtype=int), size=active_count, replace=False)))
    colors = list(map(int, _RNG.choice(np.array(_L_COLORS, dtype=int), size=active_count, replace=False)))
    jobs: list[dict] = []

    for size, color in zip(active_sizes, colors):
        short_len = int(_RNG.integers(2, size))
        standard = _standard_l_mask(size, short_len)
        transform_code = int(_RNG.integers(0, 8))
        jobs.append(
            {
                "size": int(size),
                "color": int(color),
                "output_mask": standard,
                "input_mask": _apply_transform(standard, transform_code),
            }
        )

    return jobs


def _place_jobs(height: int, width: int, right_start: int, jobs: list[dict]) -> list[tuple[int, int]] | None:
    occupied = np.zeros((height, width), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(jobs)
    order = sorted(
        range(len(jobs)),
        key=lambda idx: -(int(jobs[idx]["input_mask"].shape[0]) * int(jobs[idx]["input_mask"].shape[1])),
    )

    for idx in order:
        mask = jobs[idx]["input_mask"]
        h, w = map(int, mask.shape)
        if width - w < right_start:
            return None

        blocked = _dilate8(occupied)
        candidates: list[tuple[int, int]] = []
        for top in range(height - h + 1):
            for left in range(right_start, width - w + 1):
                window = blocked[top:top + h, left:left + w]
                if not bool((window & mask).any()):
                    candidates.append((int(top), int(left)))

        if not candidates:
            return None

        top, left = candidates[int(_RNG.integers(0, len(candidates)))]
        placements[idx] = (top, left)
        occupied[top:top + h, left:left + w] |= mask

    return [(int(top), int(left)) for top, left in placements if top is not None and left is not None]


def main(grid: np.ndarray) -> np.ndarray:
    """Normalize scattered L shapes and place them into same-sized template frames.

    Input structure:
    - The background is black or otherwise the dominant color.
    - Along the left side, several monochrome hollow square template frames of one gray color appear in a vertical stack, with at least one background row between neighboring frames.
    - Elsewhere, each remaining foreground component is a monochrome one-pixel-thick L shape whose corner sits at a bbox corner and whose two arms can have different lengths and arbitrary orientation.

    Rule:
    - Detect every L shape and normalize it up to rotation/flip into the standard orientation with the corner at the top-left, the shorter arm extending right, and the longer arm extending down.
    - Match a normalized L to the unique template frame whose inner side length equals that L's longer arm length.
    - Return a same-sized grid that keeps only the template frames and pastes each matched standardized L into its frame with the corner aligned to the frame interior's top-left cell; erase the original scattered L positions.

    Key insight:
    - Orientation is just a nuisance variable: once each component is canonicalized as the same top-left L, the only feature that matters for matching is the longer arm length, and placement always uses the same corner anchor.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    components = _find_components(g, bg)

    frames: dict[int, dict] = {}
    objects: dict[int, dict] = {}

    for component in components:
        mask = _component_mask(component)
        if _is_hollow_square(mask):
            interior = int(mask.shape[0] - 2)
            if interior not in frames:
                frames[interior] = component
            continue

        normalized = _normalize_l(mask)
        if normalized is None:
            continue
        long_len, short_len = map(int, normalized.shape)
        if long_len not in objects:
            objects[long_len] = {
                "color": int(component["color"]),
                "mask": normalized,
                "short_len": short_len,
            }

    out = np.full_like(g, bg)
    for component in frames.values():
        coords = component["coords"]
        out[coords[:, 0], coords[:, 1]] = int(component["color"])

    for size, frame in frames.items():
        obj = objects.get(int(size))
        if obj is None:
            continue
        r0, c0, _r1, _c1 = frame["bbox"]
        _paint_mask(out, obj["mask"], int(obj["color"]), int(r0) + 1, int(c0) + 1)

    return out


def _gen_attempt() -> np.ndarray | None:
    sizes = _choose_frame_sizes()
    height, left, frames = _frame_layout(sizes)
    jobs = _build_jobs(sizes)

    max_outer = max(size + 2 for size in sizes)
    max_input_width = max(int(job["input_mask"].shape[1]) for job in jobs)
    sep_gap = int(_RNG.integers(2, 5))
    right_start = int(left + max_outer + sep_gap)
    min_width = max(int(right_start + max_input_width), int(right_start + 3))
    if min_width > 30:
        return None

    start_width = int(_RNG.integers(min_width, 31))
    for width in range(start_width, 31):
        placements = _place_jobs(height, width, right_start, jobs)
        if placements is None:
            continue

        inp = _render_frames(height, width, frames)
        expected = _render_frames(height, width, frames)

        frame_by_size = {size: (top, left0) for top, left0, size in frames}
        for job in jobs:
            top, left0 = frame_by_size[int(job["size"])]
            _paint_mask(expected, job["output_mask"], int(job["color"]), top + 1, left0 + 1)

        for job, (top, left0) in zip(jobs, placements):
            _paint_mask(inp, job["input_mask"], int(job["color"]), int(top), int(left0))

        if not np.array_equal(main(inp), expected):
            continue
        return inp

    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
