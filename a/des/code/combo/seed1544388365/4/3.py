import math
import random
from collections import defaultdict

import numpy as np


BG = 8
COLOR_POOL = tuple(c for c in range(10) if c != BG)

STAMP_COUNT_RANGE = (3, 5)
STAMP_HEIGHT_RANGE = (3, 6)
STAMP_WIDTH_MAX = 7
MASK_RETRIES = 200
GENERATION_RETRIES = 400


def get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    seen = np.zeros((height, width), dtype=bool)
    comps: list[dict] = []

    for r in range(height):
        for c in range(width):
            if seen[r, c] or int(g[r, c]) == bg:
                continue

            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < height and 0 <= nc < width and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords = np.array(cells, dtype=int)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": mask,
                    "area": int(mask.sum()),
                }
            )

    return comps


def _mask_signature(mask: np.ndarray) -> tuple:
    cells = tuple(map(tuple, np.argwhere(mask).tolist()))
    return (int(mask.shape[0]), int(mask.shape[1]), cells)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find same-shape stamps, decode their stripe lengths as depth, and overprint
    them at one shared location from bottom to top.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    comps = _components(g, bg)

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for comp in comps:
        groups[_mask_signature(comp["mask"])].append(comp)

    candidates = []
    for signature, items in groups.items():
        area = int(items[0]["area"])
        if len(items) >= 2 and area > 1:
            candidates.append((area, len(items), signature))
    assert candidates, "expected repeated non-trivial stamp shape"

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    stamp_signature = candidates[0][2]
    stamps = groups[stamp_signature]

    stamp_h, stamp_w = map(int, stamps[0]["mask"].shape)
    assert len({int(stamp["color"]) for stamp in stamps}) == len(stamps), "stamp colors must be distinct"

    annotated = []
    for stamp in stamps:
        r0, c0, r1, c1 = stamp["bbox"]
        assert r1 + 1 < g.shape[0], "each stamp needs a stripe row below"
        segment = g[r1 + 1, c0 : c1 + 1]
        stripe = [int(value) for value in segment if int(value) != bg]
        assert len(stripe) <= stamp_w
        assert int(stamp["color"]) not in stripe, "stripe may not use its own stamp color"
        annotated.append({**stamp, "stripe": stripe})

    annotated.sort(key=lambda stamp: len(stamp["stripe"]))
    lengths = [len(stamp["stripe"]) for stamp in annotated]
    assert lengths == list(range(len(annotated))), "stripe lengths must rank stamps bottom-to-top"

    out = np.full((stamp_h, stamp_w), bg, dtype=int)
    below_colors: list[int] = []
    for stamp in annotated:
        assert stamp["stripe"] == below_colors, "stripe colors must list lower stamps from bottom upward"
        out[stamp["mask"]] = int(stamp["color"])
        below_colors.append(int(stamp["color"]))

    return out


def _sample_mask(height: int, width: int, stamp_count: int) -> np.ndarray | None:
    min_area = max(5, width + 1, stamp_count + 1)
    max_area = height * width - 1
    if min_area > max_area:
        return None

    for _ in range(MASK_RETRIES):
        target_area = random.randint(min_area, max_area)
        start = (random.randrange(height), random.randrange(width))
        cells = {start}

        while len(cells) < target_area:
            frontier = []
            for r, c in cells:
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        if len(cells) != target_area:
            continue

        mask = np.zeros((height, width), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        if mask.all():
            continue
        if not (mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        return mask

    return None


def _sample_layout(stamp_count: int, stamp_h: int, stamp_w: int) -> tuple[int, int, list[tuple[int, int]]] | None:
    min_cols = 1 if stamp_count <= 3 else 2
    max_cols = min(3, stamp_count)

    for _ in range(MASK_RETRIES):
        cols = random.randint(min_cols, max_cols)
        rows = int(math.ceil(stamp_count / cols))

        cell_h = stamp_h + 1 + random.randint(1, 3)
        cell_w = stamp_w + random.randint(1, 3)
        top_margin = random.randint(1, 3)
        bottom_margin = random.randint(1, 3)
        left_margin = random.randint(1, 3)
        right_margin = random.randint(1, 3)

        height = top_margin + bottom_margin + rows * cell_h
        width = left_margin + right_margin + cols * cell_w
        if not (10 <= height <= 30 and 10 <= width <= 30):
            continue

        cells = [(r, c) for r in range(rows) for c in range(cols)]
        chosen = random.sample(cells, stamp_count)
        positions = []
        for cell_r, cell_c in chosen:
            base_r = top_margin + cell_r * cell_h
            base_c = left_margin + cell_c * cell_w
            r0 = base_r + random.randint(0, cell_h - (stamp_h + 1))
            c0 = base_c + random.randint(0, cell_w - stamp_w)
            positions.append((r0, c0))
        return height, width, positions

    return None


def _render_stamp(grid: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> None:
    stamp_h, stamp_w = mask.shape
    window = grid[top : top + stamp_h, left : left + stamp_w]
    window[mask] = int(color)


def _render_stripe(grid: np.ndarray, colors: list[int], top: int, left: int, stamp_w: int) -> None:
    if not colors:
        return
    offset = random.randint(0, stamp_w - len(colors))
    for idx, color in enumerate(colors):
        grid[top, left + offset + idx] = int(color)


def _expected_output(mask: np.ndarray, stack_order: list[int], bg: int) -> np.ndarray:
    out = np.full(mask.shape, bg, dtype=int)
    for color in stack_order:
        out[mask] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    stamp_count = random.randint(*STAMP_COUNT_RANGE)

    for _ in range(MASK_RETRIES):
        stamp_h = random.randint(*STAMP_HEIGHT_RANGE)
        stamp_w = random.randint(max(3, stamp_count - 1), STAMP_WIDTH_MAX)
        mask = _sample_mask(stamp_h, stamp_w, stamp_count)
        if mask is None:
            continue

        layout = _sample_layout(stamp_count, stamp_h, stamp_w)
        if layout is None:
            continue

        grid_h, grid_w, positions = layout
        colors = random.sample(COLOR_POOL, stamp_count)
        stack_order = random.sample(colors, stamp_count)

        grid = np.full((grid_h, grid_w), BG, dtype=int)
        order_to_pos = list(zip(stack_order, positions))
        random.shuffle(order_to_pos)

        for depth, (color, (top, left)) in enumerate(order_to_pos):
            _render_stamp(grid, mask, color, top, left)
            lower_colors = stack_order[: stack_order.index(color)]
            _render_stripe(grid, lower_colors, top + stamp_h, left, stamp_w)

        expected = _expected_output(mask, stack_order, BG)
        if int(np.bincount(grid.ravel(), minlength=10).argmax()) != BG:
            continue
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
