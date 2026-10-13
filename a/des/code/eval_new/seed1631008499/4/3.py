from __future__ import annotations

import numpy as np


NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
TRANSFORM_NAMES = ("identity", "flip_h", "rot90_cw", "invert")

# --------------- shared helpers ---------------


def _get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
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
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(coords)
    return parts


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    arr = np.asarray(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _component_count(mask: np.ndarray) -> int:
    return len(_components(mask))


def _apply_mask_transform(mask: np.ndarray, name: str) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if name == "identity":
        return mask.copy()
    if name == "flip_h":
        return np.fliplr(mask)
    if name == "rot90_cw":
        return np.rot90(mask, -1)
    if name == "invert":
        return ~mask
    raise ValueError(f"unknown transform: {name}")


def _apply_array_transform(arr: np.ndarray, bg: int, name: str) -> np.ndarray:
    arr = np.asarray(arr, dtype=int)
    if name == "identity":
        return arr.copy()
    if name == "flip_h":
        return np.fliplr(arr)
    if name == "rot90_cw":
        return np.rot90(arr, -1)
    if name == "invert":
        colors = [int(c) for c in np.unique(arr) if int(c) != int(bg)]
        assert len(colors) == 1, "invert expects a single foreground color"
        fg = colors[0]
        out = np.full(arr.shape, fg, dtype=int)
        out[arr != bg] = bg
        return out
    raise ValueError(f"unknown transform: {name}")


def _is_square_frame(coords: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(coords)
    side = r1 - r0 + 1
    if side < 5 or side != c1 - c0 + 1:
        return None
    rel = np.asarray(coords, dtype=int) - np.array([r0, c0], dtype=int)
    mask = np.zeros((side, side), dtype=bool)
    mask[rel[:, 0], rel[:, 1]] = True
    border = np.zeros((side, side), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if np.array_equal(mask, border):
        return int(r0), int(c0), int(r1), int(c1)
    return None


def _extract_frames(grid: np.ndarray, bg: int) -> list[dict[str, int]]:
    frames: list[dict[str, int]] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != int(bg)):
        comps = _components(grid == color)
        for coords in comps:
            frame_bbox = _is_square_frame(coords)
            if frame_bbox is None:
                continue
            r0, c0, r1, c1 = frame_bbox
            frames.append(
                {
                    "color": color,
                    "r0": r0,
                    "c0": c0,
                    "r1": r1,
                    "c1": c1,
                    "side": r1 - r0 + 1,
                }
            )
    return frames


def _find_legend_examples(grid: np.ndarray, bg: int) -> list[tuple[np.ndarray, np.ndarray]]:
    candidates: dict[tuple[int, int], list[int]] = {}
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != int(bg)):
        for coords in _components(grid == color):
            if len(coords) != 3:
                continue
            r0, c0, r1, c1 = _bbox(coords)
            if r1 - r0 + 1 != 3 or c1 - c0 + 1 != 1:
                continue
            if c0 < 3 or c0 + 3 >= grid.shape[1]:
                continue
            left = grid[r0 : r0 + 3, c0 - 3 : c0]
            right = grid[r0 : r0 + 3, c0 + 1 : c0 + 4]
            if left.shape != (3, 3) or right.shape != (3, 3):
                continue
            if not np.any(left != bg) or not np.any(right != bg):
                continue
            candidates.setdefault((color, r0), []).append(c0)

    assert candidates, "no legend separators found"
    best_key = max(candidates, key=lambda key: (key[1], len(candidates[key])))
    sep_cols = sorted(candidates[best_key])
    sep_color, r0 = best_key
    assert len(sep_cols) >= 2, "legend must contain multiple examples"

    examples: list[tuple[np.ndarray, np.ndarray]] = []
    for c0 in sep_cols:
        assert np.all(grid[r0 : r0 + 3, c0] == sep_color), "legend separator must be vertical"
        src = grid[r0 : r0 + 3, c0 - 3 : c0]
        dst = grid[r0 : r0 + 3, c0 + 1 : c0 + 4]
        examples.append((src.copy(), dst.copy()))
    return examples


def _infer_transform(src: np.ndarray, dst: np.ndarray, bg: int) -> str:
    src_mask = np.asarray(src != bg, dtype=bool)
    dst_mask = np.asarray(dst != bg, dtype=bool)
    matches = [name for name in TRANSFORM_NAMES if np.array_equal(_apply_mask_transform(src_mask, name), dst_mask)]
    assert len(matches) == 1, f"legend example is ambiguous: {matches}"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each empty colored frame with the transformed pattern from its same-colored partner.

    Input structure:
    - The background is the most common color.
    - Several same-size one-pixel-thick square frames appear in same-colored pairs; in each pair, one frame contains a single-color pattern and the other frame is empty.
    - A legend band near the bottom contains several 3x3 before/after examples, each split by a vertical separator column.

    Rule:
    - For every non-empty frame, count the number of 4-connected non-background components in its interior pattern.
    - Read the legend examples from left to right; the Nth example defines the transformation chosen by component count N.
    - Apply that transformation to the source frame's interior pattern and write the result into the empty frame with the same border color.
    - Keep the original source frames and the legend unchanged.

    Key insight:
    - The legend does not directly match colors or locations; it defines a transform lookup table indexed by the source pattern's connected-component count.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    legend_examples = _find_legend_examples(g, bg)
    transform_order = [_infer_transform(src, dst, bg) for src, dst in legend_examples]

    frames = _extract_frames(g, bg)
    by_color: dict[int, list[dict[str, int]]] = {}
    for frame in frames:
        by_color.setdefault(frame["color"], []).append(frame)

    out = g.copy()
    for color, group in by_color.items():
        if len(group) != 2:
            continue
        frame_infos = []
        for frame in group:
            r0, c0, r1, c1 = frame["r0"], frame["c0"], frame["r1"], frame["c1"]
            inner = g[r0 + 1 : r1, c0 + 1 : c1]
            frame_infos.append((frame, inner, bool(np.any(inner != bg))))

        if sum(1 for _frame, _inner, filled in frame_infos if filled) != 1:
            continue

        src_frame, src_inner = next((frame, inner) for frame, inner, filled in frame_infos if filled)
        dst_frame = next(frame for frame, inner, filled in frame_infos if not filled)

        count = _component_count(src_inner != bg)
        assert 1 <= count <= len(transform_order), f"component count {count} is outside legend range"
        transform_name = transform_order[count - 1]
        transformed = _apply_array_transform(src_inner, bg, transform_name)

        dr0, dc0, dr1, dc1 = dst_frame["r0"], dst_frame["c0"], dst_frame["r1"], dst_frame["c1"]
        out[dr0 + 1 : dr1, dc0 + 1 : dc1] = transformed

    return out


# --------------- generator (component_indexed_transform_pairs) ---------------

RNG = np.random.default_rng()
BG_COLOR = 0
SEPARATOR_COLOR = 5
LEGEND_WIDTH = 28
LEGEND_HEIGHT = 3
FRAME_SIZE_OPTIONS = (5, 6)
FRAME_PAIR_COUNT = 4
MAX_GENERATION_ATTEMPTS = 400
COLOR_POOL = tuple(c for c in range(10) if c not in {BG_COLOR, SEPARATOR_COLOR})


def _sample_seeds(size: int, count: int) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(size) for c in range(size)]
    order = RNG.permutation(len(cells))
    chosen: list[tuple[int, int]] = []
    for idx in order:
        cell = cells[int(idx)]
        if any(abs(cell[0] - r) + abs(cell[1] - c) <= 1 for r, c in chosen):
            continue
        chosen.append(cell)
        if len(chosen) == count:
            return chosen
    return None


def _random_component_mask(size: int, count: int, transform_name: str) -> np.ndarray | None:
    for _ in range(300):
        seeds = _sample_seeds(size, count)
        if seeds is None:
            continue
        labels = -np.ones((size, size), dtype=int)
        for idx, (r, c) in enumerate(seeds):
            labels[r, c] = idx

        max_fill = size * size - 1
        target_fill = int(RNG.integers(count, max_fill + 1))
        while np.count_nonzero(labels >= 0) < target_fill:
            options: list[tuple[int, int, int]] = []
            for idx in range(count):
                for r, c in np.argwhere(labels == idx):
                    for dr, dc in NEIGHBORS4:
                        nr, nc = int(r + dr), int(c + dc)
                        if not (0 <= nr < size and 0 <= nc < size):
                            continue
                        if labels[nr, nc] != -1:
                            continue
                        neighbor_ids = {
                            int(labels[rr, cc])
                            for rr, cc in ((nr - 1, nc), (nr + 1, nc), (nr, nc - 1), (nr, nc + 1))
                            if 0 <= rr < size and 0 <= cc < size and labels[rr, cc] != -1
                        }
                        if neighbor_ids - {idx}:
                            continue
                        options.append((idx, nr, nc))
            if not options:
                break
            choice = options[int(RNG.integers(len(options)))]
            labels[choice[1], choice[2]] = choice[0]

        mask = labels >= 0
        if _component_count(mask) != count:
            continue
        if mask.sum() == 0 or mask.sum() == size * size:
            continue
        if transform_name != "identity" and np.array_equal(_apply_mask_transform(mask, transform_name), mask):
            continue
        return mask
    return None


def _has_vertical_triplet_component(mask: np.ndarray) -> bool:
    for coords in _components(mask):
        if len(coords) != 3:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        if r1 - r0 + 1 == 3 and c1 - c0 + 1 == 1:
            return True
    return False


def _legend_example_mask(transform_name: str, used: set[tuple[int, ...]]) -> np.ndarray | None:
    for _ in range(400):
        density = float(RNG.uniform(0.25, 0.65))
        mask = RNG.random((3, 3)) < density
        if mask.sum() == 0 or mask.sum() == 9:
            continue
        transformed = _apply_mask_transform(mask, transform_name)
        if transform_name != "identity" and np.array_equal(mask, transformed):
            continue
        if _has_vertical_triplet_component(mask) or _has_vertical_triplet_component(transformed):
            continue
        matches = [name for name in TRANSFORM_NAMES if np.array_equal(_apply_mask_transform(mask, name), transformed)]
        if matches != [transform_name]:
            continue
        sig = tuple(mask.astype(int).ravel()) + tuple(transformed.astype(int).ravel())
        if sig in used:
            continue
        used.add(sig)
        return mask
    return None


def _paint_frame(grid: np.ndarray, r0: int, c0: int, side: int, border_color: int, inner: np.ndarray | None) -> None:
    grid[r0, c0 : c0 + side] = border_color
    grid[r0 + side - 1, c0 : c0 + side] = border_color
    grid[r0 : r0 + side, c0] = border_color
    grid[r0 : r0 + side, c0 + side - 1] = border_color
    if inner is not None:
        grid[r0 + 1 : r0 + side - 1, c0 + 1 : c0 + side - 1] = inner


def _mask_to_array(mask: np.ndarray, fg: int, bg: int = BG_COLOR) -> np.ndarray:
    arr = np.full(mask.shape, bg, dtype=int)
    arr[mask] = fg
    return arr


def _gen_attempt() -> np.ndarray | None:
    side = int(RNG.choice(FRAME_SIZE_OPTIONS))
    frame_block_h = 2 * side + 1
    frame_block_w = FRAME_PAIR_COUNT * side + (FRAME_PAIR_COUNT - 1)
    min_w = max(LEGEND_WIDTH, frame_block_w)
    width = int(RNG.integers(min_w, 31))

    gap_to_legend = int(RNG.integers(1, 3))
    bottom_margin = int(RNG.integers(0, 2))
    min_h = frame_block_h + gap_to_legend + LEGEND_HEIGHT + bottom_margin
    height = int(RNG.integers(min_h, 31))

    legend_r0 = height - LEGEND_HEIGHT - bottom_margin
    max_frame_r0 = legend_r0 - gap_to_legend - frame_block_h
    if max_frame_r0 < 0:
        return None
    frame_r0 = int(RNG.integers(0, max_frame_r0 + 1))
    frame_c0 = int(RNG.integers(0, width - frame_block_w + 1))
    legend_c0 = int(RNG.integers(0, width - LEGEND_WIDTH + 1))

    grid = np.full((height, width), BG_COLOR, dtype=int)

    colors = list(COLOR_POOL)
    RNG.shuffle(colors)
    frame_colors = colors[:FRAME_PAIR_COUNT]
    pattern_colors = colors[FRAME_PAIR_COUNT : FRAME_PAIR_COUNT * 2]
    legend_fg = int(pattern_colors[int(RNG.integers(len(pattern_colors)))])

    legend_order = list(RNG.permutation(TRANSFORM_NAMES))
    used_legend_sigs: set[tuple[int, ...]] = set()
    for idx, transform_name in enumerate(legend_order):
        src_mask = _legend_example_mask(str(transform_name), used_legend_sigs)
        if src_mask is None:
            return None
        dst_mask = _apply_mask_transform(src_mask, str(transform_name))
        start = legend_c0 + idx * 7
        grid[legend_r0 : legend_r0 + 3, start : start + 3] = _mask_to_array(src_mask, legend_fg)
        grid[legend_r0 : legend_r0 + 3, start + 3] = SEPARATOR_COLOR
        grid[legend_r0 : legend_r0 + 3, start + 4 : start + 7] = _mask_to_array(dst_mask, legend_fg)

    slots = [
        (frame_r0 + row * (side + 1), frame_c0 + col * (side + 1))
        for row in range(2)
        for col in range(FRAME_PAIR_COUNT)
    ]
    slot_order = list(RNG.permutation(len(slots)))

    inner_side = side - 2
    for count in range(1, FRAME_PAIR_COUNT + 1):
        transform_name = str(legend_order[count - 1])
        src_mask = _random_component_mask(inner_side, count, transform_name)
        if src_mask is None:
            return None
        fg = int(pattern_colors[count - 1])
        src_inner = _mask_to_array(src_mask, fg)

        slot_a = slots[slot_order[2 * (count - 1)]]
        slot_b = slots[slot_order[2 * (count - 1) + 1]]
        if RNG.random() < 0.5:
            filled_slot, empty_slot = slot_a, slot_b
        else:
            filled_slot, empty_slot = slot_b, slot_a

        _paint_frame(grid, filled_slot[0], filled_slot[1], side, int(frame_colors[count - 1]), src_inner)
        _paint_frame(grid, empty_slot[0], empty_slot[1], side, int(frame_colors[count - 1]), None)

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 3")
