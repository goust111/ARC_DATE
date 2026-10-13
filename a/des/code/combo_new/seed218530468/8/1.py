from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME = 5
PALETTE_COLORS = tuple(c for c in range(10) if c not in {BG, FRAME})
GEN_RETRIES = 400
PLACEMENT_TRIES = 200


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(coords)
    return parts


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _rect_frame_bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    if len(coords) != 2 * (h + w) - 4:
        return None
    local = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        local[r - r0, c - c0] = True
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.array_equal(local, border):
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _parse_palette(row: np.ndarray, bg: int) -> dict[int, int]:
    mapping: dict[int, int] = {}
    c = 0
    while c < len(row):
        if int(row[c]) == bg:
            c += 1
            continue
        start = c
        while c < len(row) and int(row[c]) != bg:
            c += 1
        run = row[start:c]
        assert len(run) == 2, "palette row must contain isolated 1x2 pairs"
        left, right = int(run[0]), int(run[1])
        assert left != bg and right != bg
        assert left not in mapping, "seed colors must map unambiguously"
        mapping[left] = right
    assert mapping, "expected at least one palette pair"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey frame from the palette entry selected by its interior seed, then erase both palette and seed.

    Input structure:
    - The background is the dominant color.
    - The top row contains isolated 1x2 palette pairs separated by background cells.
    - Away from the palette row, several one-pixel-thick grey rectangular frames appear on the background.
    - Each frame encloses exactly one small 4-connected seed blob of a non-background, non-grey color.

    Rule:
    - Read the palette row as mappings from left color = seed color to right color = target frame color.
    - For every grey rectangular frame, inspect its interior seed color, look up the matching target color, and repaint the whole frame border to that target color.
    - Replace the seed blob with background, and erase the entire palette row to background.

    Key insight:
    - The seed inside each closed frame does not move; it only serves as a lookup key that determines the frame's final border color through the palette row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)
    palette = _parse_palette(g[0], bg)

    frame_candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in sorted(set(map(int, g.ravel())) - {bg}):
        for coords in _components(g == color):
            bbox = _rect_frame_bbox(coords)
            if bbox is not None:
                frame_candidates.append((int(color), bbox))

    assert frame_candidates, "expected at least one rectangular frame"
    frame_colors = {color for color, _bbox_value in frame_candidates}
    assert len(frame_colors) == 1, "expected one shared frame color"
    frame_color = next(iter(frame_colors))

    out = g.copy()
    out[0, :] = bg
    for _color, (r0, c0, r1, c1) in frame_candidates:
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_colors = sorted(set(map(int, inner.ravel())) - {bg, frame_color})
        assert len(seed_colors) == 1, "each frame interior must contain one seed color"
        seed_color = seed_colors[0]
        assert seed_color in palette, "seed color must appear in the palette row"
        seed_mask = inner == seed_color
        assert 1 <= int(seed_mask.sum()) <= 4, "seed size must be between 1 and 4 pixels"
        assert len(_components(seed_mask)) == 1, "seed must be 4-connected"

        target = int(palette[seed_color])
        out[r0 : r1 + 1, c0] = target
        out[r0 : r1 + 1, c1] = target
        out[r0, c0 : c1 + 1] = target
        out[r1, c0 : c1 + 1] = target
        out[r0 + 1 : r1, c0 + 1 : c1][seed_mask] = bg

    return out


def _shape_bank() -> list[np.ndarray]:
    base_shapes = [
        np.array([[1]], dtype=bool),
        np.array([[1, 1]], dtype=bool),
        np.array([[1, 1, 1]], dtype=bool),
        np.array([[1, 0], [1, 1]], dtype=bool),
        np.array([[1, 1], [1, 1]], dtype=bool),
        np.array([[1, 1, 1, 1]], dtype=bool),
        np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    ]
    seen: set[tuple[tuple[int, int], tuple[int, ...]]] = set()
    out: list[np.ndarray] = []
    for shape in base_shapes:
        for turns in range(4):
            rotated = np.rot90(shape, turns)
            for variant in (rotated, np.fliplr(rotated)):
                key = (variant.shape, tuple(int(v) for v in variant.ravel()))
                if key not in seen:
                    seen.add(key)
                    out.append(variant.copy())
    return out


SEED_SHAPES = _shape_bank()


def _candidate_seed(interior_h: int, interior_w: int) -> np.ndarray | None:
    options = [shape for shape in SEED_SHAPES if shape.shape[0] <= interior_h and shape.shape[1] <= interior_w]
    if not options:
        return None
    return random.choice(options).copy()


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _sample_frame_specs(seed_colors: list[int], n_frames: int) -> list[dict[str, object]] | None:
    specs: list[dict[str, object]] = []
    for seed_color in seed_colors:
        for _ in range(20):
            outer_h = random.randint(3, 9)
            outer_w = random.randint(3, 9)
            shape = _candidate_seed(outer_h - 2, outer_w - 2)
            if shape is not None:
                specs.append(
                    {
                        "seed_color": int(seed_color),
                        "outer_h": int(outer_h),
                        "outer_w": int(outer_w),
                        "shape": shape,
                    }
                )
                break
        else:
            return None
    specs.sort(key=lambda item: int(item["outer_h"]) * int(item["outer_w"]), reverse=True)
    assert len(specs) == n_frames
    return specs


def _place_frames(
    h: int,
    w: int,
    specs: list[dict[str, object]],
) -> list[dict[str, object]] | None:
    reserved = np.zeros((h, w), dtype=bool)
    reserved[0, :] = True
    placed: list[dict[str, object]] = []
    for spec in specs:
        outer_h = int(spec["outer_h"])
        outer_w = int(spec["outer_w"])
        shape = np.asarray(spec["shape"], dtype=bool)
        sh, sw = shape.shape
        for _ in range(PLACEMENT_TRIES):
            if h - outer_h < 2 or w - outer_w < 0:
                return None
            r0 = random.randint(2, h - outer_h)
            c0 = random.randint(0, w - outer_w)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1
            pr0 = max(0, r0 - 1)
            pc0 = max(0, c0 - 1)
            pr1 = min(h - 1, r1 + 1)
            pc1 = min(w - 1, c1 + 1)
            if reserved[pr0 : pr1 + 1, pc0 : pc1 + 1].any():
                continue

            ih = outer_h - 2
            iw = outer_w - 2
            if ih < sh or iw < sw:
                continue
            if ih - sh >= 2 and random.random() < 0.5:
                sr0 = random.randint(1, ih - sh - 1)
            else:
                sr0 = random.randint(0, ih - sh)
            if iw - sw >= 2 and random.random() < 0.5:
                sc0 = random.randint(1, iw - sw - 1)
            else:
                sc0 = random.randint(0, iw - sw)

            seed_cells = [
                (r0 + 1 + sr0 + dr, c0 + 1 + sc0 + dc)
                for dr in range(sh)
                for dc in range(sw)
                if shape[dr, dc]
            ]
            reserved[pr0 : pr1 + 1, pc0 : pc1 + 1] = True
            placed.append(
                {
                    "seed_color": int(spec["seed_color"]),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "seed_cells": seed_cells,
                }
            )
            break
        else:
            return None
    return placed


def _palette_pairs() -> tuple[list[tuple[int, int]], dict[int, int]]:
    pair_count = random.randint(2, 4)
    colors = random.sample(PALETTE_COLORS, 2 * pair_count)
    seed_colors = colors[:pair_count]
    target_colors = colors[pair_count:]
    pairs = list(zip(seed_colors, target_colors))
    return pairs, {int(seed): int(target) for seed, target in pairs}


def _frame_seed_sequence(seed_palette: list[int]) -> list[int]:
    if random.random() < 0.5:
        n_frames = random.randint(1, len(seed_palette))
        return random.sample(seed_palette, n_frames)
    n_frames = random.randint(len(seed_palette) + 1, min(5, len(seed_palette) + 2))
    return [random.choice(seed_palette) for _ in range(n_frames)]


def _build_expected(inp: np.ndarray, placed: list[dict[str, object]], palette_map: dict[int, int]) -> np.ndarray:
    out = inp.copy()
    out[0, :] = BG
    for item in placed:
        r0, c0, r1, c1 = item["bbox"]
        target = int(palette_map[int(item["seed_color"])])
        out[r0 : r1 + 1, c0] = target
        out[r0 : r1 + 1, c1] = target
        out[r0, c0 : c1 + 1] = target
        out[r1, c0 : c1 + 1] = target
        for r, c in item["seed_cells"]:
            out[int(r), int(c)] = BG
    return out


def _gen_attempt() -> np.ndarray | None:
    pairs, palette_map = _palette_pairs()
    seed_palette = [seed for seed, _target in pairs]
    frame_seed_colors = _frame_seed_sequence(seed_palette)
    frame_specs = _sample_frame_specs(frame_seed_colors, len(frame_seed_colors))
    if frame_specs is None:
        return None

    palette_width = 3 * len(pairs) - 1
    base_min = 12 if len(frame_specs) <= 3 else 18
    h = random.randint(base_min, 30)
    w = random.randint(max(base_min, palette_width + 1), 30)
    palette_start = random.randint(0, w - palette_width)

    placed = _place_frames(h, w, frame_specs)
    if placed is None:
        return None

    inp = np.full((h, w), BG, dtype=int)
    cursor = palette_start
    for index, (seed, target) in enumerate(pairs):
        inp[0, cursor] = int(seed)
        inp[0, cursor + 1] = int(target)
        cursor += 2
        if index != len(pairs) - 1:
            cursor += 1

    for item in placed:
        r0, c0, r1, c1 = item["bbox"]
        _draw_frame(inp, r0, c0, r1 - r0 + 1, c1 - c0 + 1, FRAME)
        for r, c in item["seed_cells"]:
            inp[int(r), int(c)] = int(item["seed_color"])

    if _most_common_color(inp) != BG:
        return None

    expected = _build_expected(inp, placed, palette_map)
    try:
        predicted = main(inp)
    except Exception:
        return None
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(predicted, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
