from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
FRAME_COLORS = (5, 7, 9)
SEED_COLORS = (1, 2, 3, 4, 6)
GEN_RETRY_LIMIT = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _tile(seed: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = map(int, shape)
    sh, sw = seed.shape
    rows = np.arange(h) % sh
    cols = np.arange(w) % sw
    return seed[rows[:, None], cols[None, :]]


def _frame_candidates(grid: np.ndarray) -> list[tuple[int, tuple[int, int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    found: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in np.unique(g):
        color = int(color)
        if color == BACKGROUND:
            continue
        mask = g == color
        box = _bbox(mask)
        if box is None:
            continue
        r0, c0, r1, c1 = box
        if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
            continue
        frame = np.zeros_like(mask, dtype=bool)
        frame[r0 : r1 + 1, c0] = True
        frame[r0 : r1 + 1, c1] = True
        frame[r0, c0 : c1 + 1] = True
        frame[r1, c0 : c1 + 1] = True
        if np.array_equal(mask, frame):
            found.append((color, box))
    found.sort(key=lambda item: (item[1][0], item[1][1], item[0]))
    return found


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow rectangle's interior by cyclically tiling the marked seed pattern from its top-left corner.

    Input structure
    - The grid has a black background plus several complete single-color hollow rectangles.
    - Each rectangle color appears only on that rectangle's one-pixel border.
    - Inside a rectangle there is either no mark at all, or one compact multicolor mark near the interior top-left; the rest of the interior is black.

    Rule
    - Find every color whose pixels form exactly one complete rectangular border.
    - For each rectangle, inspect only its interior. If the interior has any non-black pixels, take the minimal bounding box of all those pixels as the seed block, keeping black cells inside that seed box.
    - Replace the whole interior with a top-left aligned 2D periodic tiling of that seed block, clipped at the interior edges. If the interior is entirely black, fill it uniformly with the rectangle's border color.

    Key insight
    - The rectangle border identifies the container, but the repeated fill is determined by the interior mark's full bounding box, not just by its colored cells; black holes inside the seed must repeat too.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    frames = _frame_candidates(g)
    assert frames, "expected at least one complete hollow rectangle"

    for frame_color, (r0, c0, r1, c1) in frames:
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        assert inner.size > 0, "rectangle must have an interior"
        nonblack = inner != BACKGROUND
        if not np.any(nonblack):
            out[r0 + 1 : r1, c0 + 1 : c1] = frame_color
            continue

        sr0, sc0, sr1, sc1 = _bbox(nonblack)
        seed = inner[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()
        out[r0 + 1 : r1, c0 + 1 : c1] = _tile(seed, inner.shape)

    return out


def _seed_masks() -> tuple[list[np.ndarray], list[np.ndarray]]:
    hole_masks: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []
    for h in range(1, 4):
        for w in range(1, 4):
            cells = h * w
            for bits in range(1 << cells):
                area = bits.bit_count()
                if area < 2 or area > min(6, cells):
                    continue
                mask = np.zeros((h, w), dtype=bool)
                for idx in range(cells):
                    if bits & (1 << idx):
                        mask[idx // w, idx % w] = True
                parts = _components(mask)
                if len(parts) != 1:
                    continue
                box = _bbox(mask)
                if box != (0, 0, h - 1, w - 1):
                    continue
                all_masks.append(mask)
                if area < cells:
                    hole_masks.append(mask)
    return hole_masks, all_masks


_HOLE_MASKS, _ALL_MASKS = _seed_masks()


def _make_seed(require_hole: bool = False) -> np.ndarray:
    mask = random.choice(_HOLE_MASKS if require_hole else _ALL_MASKS)
    h, w = mask.shape
    seed = np.zeros((h, w), dtype=int)
    cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    use_two_colors = len(cells) >= 3 and random.random() < 0.45
    palette = random.sample(SEED_COLORS, 2 if use_two_colors else 1)
    if use_two_colors:
        split = random.randint(1, len(cells) - 1)
        shuffled = cells[:]
        random.shuffle(shuffled)
        color_cells = {cell: palette[0] for cell in shuffled[:split]}
        for cell in shuffled[split:]:
            color_cells[cell] = palette[1]
    else:
        color_cells = {cell: palette[0] for cell in cells}
    for (r, c), color in color_cells.items():
        seed[r, c] = color
    return seed


def _choose_interior(seed: np.ndarray | None, must_repeat: bool, must_truncate: bool) -> tuple[int, int]:
    if seed is None:
        return random.randint(4, 8), random.randint(4, 8)

    sh, sw = seed.shape
    options: list[tuple[int, int]] = []
    for h in range(max(4, sh), 9):
        for w in range(max(4, sw), 9):
            if must_repeat and not (h > 2 * sh and w > 2 * sw):
                continue
            if must_truncate and (h % sh == 0 and w % sw == 0):
                continue
            options.append((h, w))
    assert options, "expected at least one valid interior size"
    return random.choice(options)


def _place_rectangles(
    shape: tuple[int, int], sizes: list[tuple[int, int]]
) -> list[tuple[int, int]] | None:
    h, w = shape
    order = sorted(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1], reverse=True)
    placed: list[tuple[int, int] | None] = [None] * len(sizes)
    bboxes: list[tuple[int, int, int, int]] = []
    for idx in order:
        rh, rw = sizes[idx]
        options: list[tuple[int, int]] = []
        for r0 in range(1, h - rh):
            for c0 in range(1, w - rw):
                box = (r0, c0, r0 + rh - 1, c0 + rw - 1)
                ok = True
                for pr0, pc0, pr1, pc1 in bboxes:
                    if not (box[2] + 1 < pr0 or pr1 + 1 < box[0] or box[3] + 1 < pc0 or pc1 + 1 < box[1]):
                        ok = False
                        break
                if ok:
                    options.append((r0, c0))
        if not options:
            return None
        placed[idx] = random.choice(options)
        r0, c0 = placed[idx]
        bboxes.append((r0, c0, r0 + rh - 1, c0 + rw - 1))
    return [pos for pos in placed if pos is not None]


def _render_expected(
    shape: tuple[int, int], rects: list[dict], *, tiled: bool
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for rect in rects:
        r0, c0 = rect["top_left"]
        ih, iw = rect["inner_shape"]
        frame_color = rect["frame_color"]
        r1 = r0 + ih + 1
        c1 = c0 + iw + 1
        grid[r0 : r1 + 1, c0] = frame_color
        grid[r0 : r1 + 1, c1] = frame_color
        grid[r0, c0 : c1 + 1] = frame_color
        grid[r1, c0 : c1 + 1] = frame_color
        if tiled:
            if rect["seed"] is None:
                grid[r0 + 1 : r1, c0 + 1 : c1] = frame_color
            else:
                grid[r0 + 1 : r1, c0 + 1 : c1] = _tile(rect["seed"], (ih, iw))
        else:
            seed = rect["seed"]
            if seed is None:
                continue
            sr, sc = rect["seed_offset"]
            sh, sw = seed.shape
            grid[r0 + 1 + sr : r0 + 1 + sr + sh, c0 + 1 + sc : c0 + 1 + sc + sw] = seed
    return grid


def _validate_instance(grid: np.ndarray, rects: list[dict]) -> bool:
    detected = _frame_candidates(grid)
    expected_detected = sorted(
        (
            rect["frame_color"],
            (
                rect["top_left"][0],
                rect["top_left"][1],
                rect["top_left"][0] + rect["inner_shape"][0] + 1,
                rect["top_left"][1] + rect["inner_shape"][1] + 1,
            ),
        )
        for rect in rects
    )
    if detected != expected_detected:
        return False

    expected_input = _render_expected(grid.shape, rects, tiled=False)
    if not np.array_equal(grid, expected_input):
        return False

    expected_output = _render_expected(grid.shape, rects, tiled=True)
    actual_output = main(grid)
    if not np.array_equal(actual_output, expected_output):
        return False

    if np.array_equal(actual_output, grid):
        return False

    has_empty = False
    has_hole_tiling = False
    has_truncation = False
    for rect in rects:
        r0, c0 = rect["top_left"]
        ih, iw = rect["inner_shape"]
        r1 = r0 + ih + 1
        c1 = c0 + iw + 1
        out_inner = actual_output[r0 + 1 : r1, c0 + 1 : c1]
        seed = rect["seed"]
        if seed is None:
            has_empty = True
            if not np.all(out_inner == rect["frame_color"]):
                return False
            continue
        sh, sw = seed.shape
        if ih > sh or iw > sw:
            has_truncation |= (ih % sh != 0 or iw % sw != 0)
        if np.any(seed == 0) and np.any(out_inner == 0) and (ih > sh or iw > sw):
            has_hole_tiling = True
        if not np.array_equal(out_inner, _tile(seed, (ih, iw))):
            return False

    if not has_hole_tiling or not has_truncation:
        return False
    if any(rect["seed"] is None for rect in rects) and not has_empty:
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    grid_h = random.randint(18, 26)
    grid_w = random.randint(18, 26)
    rect_count = random.randint(2, 3)
    frame_colors = random.sample(FRAME_COLORS, rect_count)

    rects: list[dict] = []
    hole_idx = random.randrange(rect_count)
    empty_idx = random.randrange(rect_count) if random.random() < 0.55 else None
    if empty_idx == hole_idx and rect_count > 1:
        empty_idx = (empty_idx + 1) % rect_count

    for idx in range(rect_count):
        if idx == empty_idx:
            seed = None
        else:
            seed = _make_seed(require_hole=(idx == hole_idx))
        inner_shape = _choose_interior(
            seed,
            must_repeat=(seed is not None and idx == hole_idx),
            must_truncate=(seed is not None and idx == hole_idx),
        )
        rects.append(
            {
                "frame_color": frame_colors[idx],
                "seed": seed,
                "inner_shape": inner_shape,
            }
        )

    total_sizes = [(ih + 2, iw + 2) for ih, iw in (rect["inner_shape"] for rect in rects)]
    positions = _place_rectangles((grid_h, grid_w), total_sizes)
    if positions is None:
        return None

    for rect, pos in zip(rects, positions):
        rect["top_left"] = pos
        seed = rect["seed"]
        if seed is None:
            rect["seed_offset"] = (0, 0)
            continue
        ih, iw = rect["inner_shape"]
        sh, sw = seed.shape
        max_r = min(1, ih - sh)
        max_c = min(1, iw - sw)
        rect["seed_offset"] = (random.randint(0, max_r), random.randint(0, max_c))

    grid = _render_expected((grid_h, grid_w), rects, tiled=False)
    return grid if _validate_instance(grid, rects) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_RETRY_LIMIT} attempts")
