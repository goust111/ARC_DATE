from __future__ import annotations

import random

import numpy as np


LEGEND_HEIGHT = 3
ENTRY_WIDTH = 4
ENTRY_GAP = 1
BG_COLOR = 0
FRAME_COLOR = 5
MAX_GENERATION_ATTEMPTS = 400


_BASE_PATTERNS = [
    np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
    np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 0], [1, 0, 1]], dtype=int),
]


def _all_pattern_variants() -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    for base in _BASE_PATTERNS:
        for rot in range(4):
            rotated = np.rot90(base, rot)
            for candidate in (rotated, np.fliplr(rotated)):
                key = tuple(int(x) for x in candidate.reshape(-1))
                if key not in seen:
                    seen.add(key)
                    variants.append(candidate.copy())
    return variants


PATTERN_VARIANTS = _all_pattern_variants()


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _spans(flags: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(flags):
        if not flags[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(flags) and flags[j + 1]:
            j += 1
        spans.append((i, j))
        i = j + 1
    return spans


def _parse_legend_band(band: np.ndarray, bg: int) -> dict[int, np.ndarray] | None:
    active = np.any(band != bg, axis=0)
    spans = _spans(active)
    if len(spans) < 2:
        return None

    mapping: dict[int, np.ndarray] = {}
    for c0, c1 in spans:
        if c1 - c0 + 1 != ENTRY_WIDTH:
            return None

        label_col = band[:, c0]
        non_bg_rows = np.flatnonzero(label_col != bg)
        if non_bg_rows.size != 1 or int(non_bg_rows[0]) != 1:
            return None

        label_color = int(label_col[1])
        pattern = band[:, c0 + 1 : c0 + 4].copy()
        if pattern.shape != (3, 3):
            return None
        if not np.any(pattern != bg):
            return None
        if np.any(np.all(pattern == bg, axis=0)) or np.any(np.all(pattern == bg, axis=1)):
            return None
        if len(set(int(x) for x in np.unique(pattern)) - {bg}) != 1:
            return None
        if label_color in mapping:
            return None
        mapping[label_color] = pattern

    return mapping


def _extract_legend(grid: np.ndarray, bg: int) -> tuple[dict[int, np.ndarray], slice]:
    assert grid.shape[0] >= LEGEND_HEIGHT, "grid must be tall enough to hold the legend"
    candidates: list[tuple[dict[int, np.ndarray], slice]] = []

    top = _parse_legend_band(grid[:LEGEND_HEIGHT, :], bg)
    if top is not None:
        candidates.append((top, slice(0, LEGEND_HEIGHT)))

    bottom = _parse_legend_band(grid[-LEGEND_HEIGHT:, :], bg)
    if bottom is not None:
        candidates.append((bottom, slice(grid.shape[0] - LEGEND_HEIGHT, grid.shape[0])))

    assert len(candidates) == 1, "expected exactly one valid top-or-bottom legend band"
    return candidates[0]


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
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_rectangular_frame(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return False
    if len(coords) != 2 * (height + width) - 4:
        return False

    coord_set = set(coords)
    for r in range(r0, r1 + 1):
        if (r, c0) not in coord_set or (r, c1) not in coord_set:
            return False
    for c in range(c0, c1 + 1):
        if (r0, c) not in coord_set or (r1, c) not in coord_set:
            return False
    return True


def _find_frames(grid: np.ndarray, bg: int, legend_rows: slice) -> list[tuple[int, int, int, int]]:
    work = grid.copy()
    work[legend_rows, :] = bg
    frames: list[tuple[int, int, int, int]] = []

    for comp in _components(work != bg):
        if not _is_rectangular_frame(comp):
            continue
        color = int(work[comp[0]])
        if any(int(work[r, c]) != color for r, c in comp):
            continue
        frames.append(_bbox(comp))

    frames.sort()
    return frames


def _apply_fill(
    out: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    seed_pos: tuple[int, int],
    pattern: np.ndarray,
) -> None:
    r0, c0, r1, c1 = frame_bbox
    sr, sc = seed_pos
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if r == sr and c == sc:
                continue
            out[r, c] = int(pattern[(r - sr) % 3, (c - sc) % 3])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the seed color inside each hollow frame as a key into the border legend, then tile
    the matched 3x3 pattern across the frame interior while preserving the seed pixel itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    bg = _most_common_color(g)
    legend, legend_rows = _extract_legend(g, bg)
    frames = _find_frames(g, bg, legend_rows)
    assert frames, "expected at least one hollow rectangular frame"

    out = g.copy()
    for frame_bbox in frames:
        r0, c0, r1, c1 = frame_bbox
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_coords = np.argwhere(interior != bg)
        assert seed_coords.shape[0] == 1, "each frame interior must contain exactly one seed"
        sr = int(seed_coords[0, 0]) + r0 + 1
        sc = int(seed_coords[0, 1]) + c0 + 1
        seed_color = int(g[sr, sc])
        assert seed_color in legend, "every seed color must appear in the legend"
        _apply_fill(out, frame_bbox, (sr, sc), legend[seed_color])

    return out


def _render_pattern(mask: np.ndarray, color: int) -> np.ndarray:
    return np.where(mask, color, BG_COLOR).astype(int)


def _legend_width(entry_count: int) -> int:
    return entry_count * ENTRY_WIDTH + (entry_count - 1) * ENTRY_GAP


def _sample_pattern_masks(entry_count: int) -> list[np.ndarray]:
    sampled = random.sample(PATTERN_VARIANTS, entry_count)
    return [mask.copy() for mask in sampled]


def _choose_interior_sizes(kind: str) -> tuple[int, int]:
    if kind == "multiple":
        return random.choice((3, 6)), random.choice((3, 6))
    if kind == "truncated":
        while True:
            h = random.randint(4, 8)
            w = random.randint(4, 8)
            if h % 3 != 0 or w % 3 != 0:
                return h, w
    h = random.randint(3, 8)
    w = random.randint(3, 8)
    return h, w


def _reserve_mask(height: int, width: int, legend_top: bool) -> np.ndarray:
    occupied = np.zeros((height, width), dtype=bool)
    occupied[0, :] = True
    occupied[-1, :] = True
    occupied[:, 0] = True
    occupied[:, -1] = True
    if legend_top:
        occupied[: LEGEND_HEIGHT + 1, :] = True
    else:
        occupied[height - (LEGEND_HEIGHT + 1) :, :] = True
    return occupied


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(occupied.shape[0], r0 + h + 1)
    cc1 = min(occupied.shape[1], c0 + w + 1)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _place_frames(
    grid: np.ndarray,
    legend_top: bool,
    frame_specs: list[dict[str, object]],
) -> list[dict[str, object]] | None:
    height, width = grid.shape
    occupied = _reserve_mask(height, width, legend_top)
    placed: list[dict[str, object]] = []

    specs = sorted(frame_specs, key=lambda spec: int(spec["outer_h"]) * int(spec["outer_w"]), reverse=True)
    for spec in specs:
        outer_h = int(spec["outer_h"])
        outer_w = int(spec["outer_w"])
        positions = [
            (r0, c0)
            for r0 in range(1, height - outer_h)
            for c0 in range(1, width - outer_w)
        ]
        random.shuffle(positions)

        for r0, c0 in positions:
            if not _can_place(occupied, r0, c0, outer_h, outer_w):
                continue

            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1
            grid[r0, c0 : c1 + 1] = FRAME_COLOR
            grid[r1, c0 : c1 + 1] = FRAME_COLOR
            grid[r0 : r1 + 1, c0] = FRAME_COLOR
            grid[r0 : r1 + 1, c1] = FRAME_COLOR

            interior_h = int(spec["interior_h"])
            interior_w = int(spec["interior_w"])
            sr = r0 + 1 + random.randrange(interior_h)
            sc = c0 + 1 + random.randrange(interior_w)
            grid[sr, sc] = int(spec["seed_color"])

            occupied[r0 : r1 + 1, c0 : c1 + 1] = True
            placed.append(
                {
                    **spec,
                    "bbox": (r0, c0, r1, c1),
                    "seed_pos": (sr, sc),
                }
            )
            break
        else:
            return None

    return placed


def _render_legend(
    grid: np.ndarray,
    legend_top: bool,
    entries: list[dict[str, object]],
) -> None:
    legend_w = _legend_width(len(entries))
    start_col = random.randint(1, grid.shape[1] - legend_w - 1)
    row0 = 0 if legend_top else grid.shape[0] - LEGEND_HEIGHT

    for idx, entry in enumerate(entries):
        c0 = start_col + idx * (ENTRY_WIDTH + ENTRY_GAP)
        grid[row0 + 1, c0] = int(entry["seed_color"])
        grid[row0 : row0 + 3, c0 + 1 : c0 + 4] = _render_pattern(
            np.asarray(entry["mask"], dtype=int),
            int(entry["pattern_color"]),
        )


def _expected_output(inp: np.ndarray, placed: list[dict[str, object]]) -> np.ndarray:
    out = inp.copy()
    for spec in placed:
        pattern = _render_pattern(np.asarray(spec["mask"], dtype=int), int(spec["pattern_color"]))
        _apply_fill(out, tuple(spec["bbox"]), tuple(spec["seed_pos"]), pattern)
    return out


def _gen_attempt() -> np.ndarray | None:
    entry_count = random.randint(2, 4)
    frame_count = random.randint(2, entry_count)
    legend_top = random.random() < 0.5

    legend_w = _legend_width(entry_count)
    grid_h = random.randint(18, 28)
    grid_w = random.randint(max(legend_w + 2, 18), 30)

    colors = list(range(10))
    colors.remove(BG_COLOR)
    colors.remove(FRAME_COLOR)
    random.shuffle(colors)
    seed_colors = colors[:entry_count]
    pattern_colors = colors[entry_count : 2 * entry_count]
    masks = _sample_pattern_masks(entry_count)

    entries = [
        {
            "seed_color": seed_colors[i],
            "pattern_color": pattern_colors[i],
            "mask": masks[i],
        }
        for i in range(entry_count)
    ]

    frame_entries = random.sample(entries, frame_count)
    size_kinds = ["multiple", "truncated"]
    while len(size_kinds) < frame_count:
        size_kinds.append(random.choice(("multiple", "truncated", "free")))
    random.shuffle(size_kinds)

    frame_specs: list[dict[str, object]] = []
    for entry, kind in zip(frame_entries, size_kinds, strict=False):
        interior_h, interior_w = _choose_interior_sizes(kind)
        frame_specs.append(
            {
                "seed_color": int(entry["seed_color"]),
                "pattern_color": int(entry["pattern_color"]),
                "mask": np.asarray(entry["mask"], dtype=int).copy(),
                "interior_h": interior_h,
                "interior_w": interior_w,
                "outer_h": interior_h + 2,
                "outer_w": interior_w + 2,
            }
        )

    grid = np.full((grid_h, grid_w), BG_COLOR, dtype=int)
    _render_legend(grid, legend_top, entries)
    placed = _place_frames(grid, legend_top, frame_specs)
    if placed is None:
        return None

    if _most_common_color(grid) != BG_COLOR:
        return None

    expected = _expected_output(grid, placed)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
