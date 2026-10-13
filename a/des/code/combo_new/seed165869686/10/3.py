import random
from collections import deque

import numpy as np


BG_COLOR = 0
SIDES = ("top", "bottom", "left", "right")

# Generator configuration.
PAIR_COUNT_RANGE = (2, 4)
INTERIOR_HEIGHT_RANGE = (3, 6)
INTERIOR_WIDTH_RANGE = (3, 6)
WORK_CANVAS = 30
PLACEMENT_PADDING = 1
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            out.append(np.asarray(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _coords_to_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0, r1, c1)


def _extract_patch(grid: np.ndarray, coords: np.ndarray, bg: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = grid[coords[:, 0], coords[:, 1]]
    return patch, (r0, c0, r1, c1)


def _paste(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> None:
    h, w = patch.shape
    view = grid[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:, :] = patch
        return
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def _is_frame_mask(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 3 or w < 3:
        return False
    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.array_equal(mask, border))


def _find_frames(grid: np.ndarray, bg: int) -> tuple[dict[int, dict[str, object]], np.ndarray]:
    frames: dict[int, dict[str, object]] = {}
    frame_pixels = np.zeros_like(grid, dtype=bool)
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for comp in _components(grid == color):
            mask, bbox = _coords_to_mask(comp)
            if not _is_frame_mask(mask):
                continue
            assert color not in frames, f"expected one frame for color {color}"
            frames[color] = {
                "color": color,
                "coords": comp,
                "bbox": bbox,
                "inner_shape": (mask.shape[0] - 2, mask.shape[1] - 2),
            }
            frame_pixels[comp[:, 0], comp[:, 1]] = True
    assert frames, "expected at least one rectangular frame"
    return frames, frame_pixels


def _fragment_side_and_color(patch: np.ndarray, frame_colors: set[int], bg: int) -> tuple[str, int]:
    candidates: list[tuple[str, int]] = []
    for side, line in (
        ("top", patch[0, :]),
        ("bottom", patch[-1, :]),
        ("left", patch[:, 0]),
        ("right", patch[:, -1]),
    ):
        if not np.all(line != bg):
            continue
        colors = {int(v) for v in np.unique(line)}
        if len(colors) != 1:
            continue
        color = next(iter(colors))
        if color in frame_colors:
            candidates.append((side, color))
    assert len(candidates) == 1, f"expected one matching side, found {candidates}"
    return candidates[0]


def _placement_origin(frame_bbox: tuple[int, int, int, int], side: str) -> tuple[int, int]:
    r0, c0, _r1, _c1 = frame_bbox
    ir0, ic0 = r0 + 1, c0 + 1
    if side == "top":
        return ir0 - 1, ic0
    if side == "bottom":
        return ir0 + 1, ic0
    if side == "left":
        return ir0, ic0 - 1
    if side == "right":
        return ir0, ic0 + 1
    raise ValueError(side)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each scattered multicolor fragment into the hollow frame whose border color appears on one fragment edge.

    Input structure:
    - The background is black.
    - Several one-pixel-thick rectangular hollow frames appear in distinct border colors, with black interiors.
    - Separate 4-connected multicolor fragments sit outside the frames; in each fragment exactly one full bbox side uses one frame's border color.
    - Each fragment's bounding-box size matches the interior size of its paired frame.

    Rule:
    - Detect every rectangular frame and treat its border color as a pairing key.
    - For each outside fragment, find the unique bbox side whose pixels all share a frame color; that side determines both the target frame and which wall it must touch.
    - Clear the fragment from its original location, then translate it into the matched frame so the shared-color side tucks under the corresponding border.
    - Repaint the frame borders last, so the overlapped side row or column is hidden and the frame outline stays intact.

    Key insight:
    - The fragment does not match a frame by overall shape or dominant color; the only reliable clue is the single edge painted in a frame color, and that edge simultaneously encodes the placement direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _mode_color(g)
    frames, frame_pixels = _find_frames(g, bg)
    frame_colors = set(frames)

    fragment_mask = (g != bg) & ~frame_pixels
    fragments = _components(fragment_mask)
    assert fragments, "expected at least one fragment"

    out = g.copy()
    for coords in fragments:
        out[coords[:, 0], coords[:, 1]] = bg

    used_colors: set[int] = set()
    for coords in fragments:
        patch, _bbox_unused = _extract_patch(g, coords, bg)
        side, frame_color = _fragment_side_and_color(patch, frame_colors, bg)
        assert frame_color not in used_colors, f"duplicate fragment for frame color {frame_color}"
        used_colors.add(frame_color)

        frame = frames[frame_color]
        assert patch.shape == frame["inner_shape"], (
            f"fragment shape {patch.shape} does not match frame interior {frame['inner_shape']}"
        )
        pr0, pc0 = _placement_origin(frame["bbox"], side)
        _paste(out, patch, pr0, pc0, transparent=bg)

    for frame in frames.values():
        coords = frame["coords"]
        out[coords[:, 0], coords[:, 1]] = int(frame["color"])
    return out


def _frame_patch(inner_h: int, inner_w: int, color: int) -> np.ndarray:
    patch = np.full((inner_h + 2, inner_w + 2), BG_COLOR, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    return patch


def _frontier(cells: set[tuple[int, int]], h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                out.append((nr, nc))
    return out


def _base_fragment_cells(h: int, w: int, side: str) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    if side == "top":
        cells.update((0, c) for c in range(w))
        col = random.randrange(w)
        for r in range(1, h):
            col = max(0, min(w - 1, col + random.choice((-1, 0, 1))))
            cells.add((r, col))
        return cells
    if side == "bottom":
        cells.update((h - 1, c) for c in range(w))
        col = random.randrange(w)
        for r in range(h - 2, -1, -1):
            col = max(0, min(w - 1, col + random.choice((-1, 0, 1))))
            cells.add((r, col))
        return cells
    if side == "left":
        cells.update((r, 0) for r in range(h))
        row = random.randrange(h)
        for c in range(1, w):
            row = max(0, min(h - 1, row + random.choice((-1, 0, 1))))
            cells.add((row, c))
        return cells
    if side == "right":
        cells.update((r, w - 1) for r in range(h))
        row = random.randrange(h)
        for c in range(w - 2, -1, -1):
            row = max(0, min(h - 1, row + random.choice((-1, 0, 1))))
            cells.add((row, c))
        return cells
    raise ValueError(side)


def _fragment_mask(inner_h: int, inner_w: int, side: str) -> np.ndarray:
    cells = _base_fragment_cells(inner_h, inner_w, side)
    total = inner_h * inner_w
    max_extra = max(0, min(total - len(cells) - 1, total // 2))
    target = len(cells) + (random.randint(0, max_extra) if max_extra > 0 else 0)
    while len(cells) < target:
        frontier = _frontier(cells, inner_h, inner_w)
        if not frontier:
            break
        cells.add(random.choice(frontier))

    mask = np.zeros((inner_h, inner_w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _side_mask(shape: tuple[int, int], side: str) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    if side == "top":
        mask[0, :] = True
    elif side == "bottom":
        mask[-1, :] = True
    elif side == "left":
        mask[:, 0] = True
    elif side == "right":
        mask[:, -1] = True
    else:
        raise ValueError(side)
    return mask


def _color_fragment(mask: np.ndarray, side: str, frame_color: int, filler_pool: tuple[int, ...]) -> np.ndarray:
    patch = np.full(mask.shape, BG_COLOR, dtype=int)
    side_cells = _side_mask(mask.shape, side)
    patch[mask & side_cells] = frame_color

    others = [(int(r), int(c)) for r, c in np.argwhere(mask & ~side_cells)]
    assert others, "fragment must contain non-side cells"
    max_colors = min(3, len(filler_pool), len(others))
    n_colors = random.randint(1, max_colors)
    chosen = random.sample(list(filler_pool), n_colors)
    random.shuffle(others)
    for idx, (r, c) in enumerate(others):
        patch[r, c] = chosen[idx % n_colors]
    return patch


def _free_positions(occupied: np.ndarray, h: int, w: int, padding: int) -> list[tuple[int, int]]:
    H, W = occupied.shape
    positions: list[tuple[int, int]] = []
    for r in range(1, H - h):
        for c in range(1, W - w):
            r0 = r - padding
            c0 = c - padding
            r1 = r + h + padding
            c1 = c + w + padding
            if not occupied[r0:r1, c0:c1].any():
                positions.append((r, c))
    return positions


def _reserve(occupied: np.ndarray, r: int, c: int, h: int, w: int, padding: int) -> None:
    occupied[r - padding : r + h + padding, c - padding : c + w + padding] = True


def _crop_pair(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = (inp != BG_COLOR) | (out != BG_COLOR)
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)

    def sample_margin(space: int) -> int:
        if space <= 0:
            return 0
        lo = 1 if space >= 1 else 0
        hi = min(2, space)
        return random.randint(lo, hi)

    top = sample_margin(r0)
    left = sample_margin(c0)
    bottom = sample_margin(inp.shape[0] - 1 - r1)
    right = sample_margin(inp.shape[1] - 1 - c1)
    rr0 = r0 - top
    cc0 = c0 - left
    rr1 = r1 + bottom
    cc1 = c1 + right
    return inp[rr0 : rr1 + 1, cc0 : cc1 + 1], out[rr0 : rr1 + 1, cc0 : cc1 + 1]


def _bg_is_dominant(grid: np.ndarray) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    counts_by_color = {int(v): int(c) for v, c in zip(values, counts)}
    bg_count = counts_by_color.get(BG_COLOR, 0)
    return bg_count > max((count for color, count in counts_by_color.items() if color != BG_COLOR), default=0)


def _gen_attempt() -> np.ndarray | None:
    n_pairs = random.randint(*PAIR_COUNT_RANGE)
    frame_colors = random.sample(list(range(1, 10)), n_pairs)
    filler_pool = tuple(color for color in range(1, 10) if color not in frame_colors)
    sides = random.sample(list(SIDES), n_pairs)

    specs: list[dict[str, object]] = []
    for idx in range(n_pairs):
        inner_h = random.randint(*INTERIOR_HEIGHT_RANGE)
        inner_w = random.randint(*INTERIOR_WIDTH_RANGE)
        side = sides[idx]
        frame_color = frame_colors[idx]
        mask = _fragment_mask(inner_h, inner_w, side)
        patch = _color_fragment(mask, side, frame_color, filler_pool)
        specs.append(
            {
                "index": idx,
                "side": side,
                "frame_color": frame_color,
                "inner_shape": (inner_h, inner_w),
                "frame_patch": _frame_patch(inner_h, inner_w, frame_color),
                "fragment_patch": patch,
            }
        )

    occupied = np.zeros((WORK_CANVAS, WORK_CANVAS), dtype=bool)
    frame_positions: dict[int, tuple[int, int]] = {}
    for spec in sorted(specs, key=lambda item: int(np.prod(item["frame_patch"].shape)), reverse=True):
        patch = spec["frame_patch"]
        h, w = patch.shape
        candidates = _free_positions(occupied, h, w, PLACEMENT_PADDING)
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        frame_positions[int(spec["index"])] = (r0, c0)
        _reserve(occupied, r0, c0, h, w, PLACEMENT_PADDING)

    fragment_positions: dict[int, tuple[int, int]] = {}
    for spec in sorted(specs, key=lambda item: int(np.prod(item["fragment_patch"].shape)), reverse=True):
        patch = spec["fragment_patch"]
        h, w = patch.shape
        candidates = _free_positions(occupied, h, w, PLACEMENT_PADDING)
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        fragment_positions[int(spec["index"])] = (r0, c0)
        _reserve(occupied, r0, c0, h, w, PLACEMENT_PADDING)

    inp = np.full((WORK_CANVAS, WORK_CANVAS), BG_COLOR, dtype=int)
    out = np.full((WORK_CANVAS, WORK_CANVAS), BG_COLOR, dtype=int)

    for spec in specs:
        idx = int(spec["index"])
        fr0, fc0 = frame_positions[idx]
        frame_patch = spec["frame_patch"]
        _paste(inp, frame_patch, fr0, fc0, transparent=BG_COLOR)
        _paste(out, frame_patch, fr0, fc0, transparent=BG_COLOR)

    for spec in specs:
        idx = int(spec["index"])
        sr0, sc0 = fragment_positions[idx]
        fragment_patch = spec["fragment_patch"]
        _paste(inp, fragment_patch, sr0, sc0, transparent=BG_COLOR)

        fr0, fc0 = frame_positions[idx]
        fh, fw = spec["frame_patch"].shape
        frame_bbox = (fr0, fc0, fr0 + fh - 1, fc0 + fw - 1)
        pr0, pc0 = _placement_origin(frame_bbox, str(spec["side"]))
        _paste(out, fragment_patch, pr0, pc0, transparent=BG_COLOR)

    for spec in specs:
        idx = int(spec["index"])
        fr0, fc0 = frame_positions[idx]
        _paste(out, spec["frame_patch"], fr0, fc0, transparent=BG_COLOR)

    inp, out = _crop_pair(inp, out)
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return None
    if not _bg_is_dominant(inp):
        return None

    try:
        predicted = main(inp)
    except Exception:
        return None
    if not np.array_equal(predicted, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
