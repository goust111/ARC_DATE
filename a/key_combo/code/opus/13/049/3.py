from __future__ import annotations

import random
from collections import deque
from functools import lru_cache

import numpy as np


BG = 0
HOLE_COUNTS = (2, 3, 4)
MAX_GEN_ATTEMPTS = 500
PALETTE = tuple(range(1, 10))
MIN_FRAME_GAP = 2
MIN_MARGIN = 1
MAX_GRID_SIDE = 30


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _scale_patch(patch: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(patch, np.ones((factor, factor), dtype=int))


def _find_frames(grid: np.ndarray, bg: int) -> list[tuple[tuple[int, int, int, int], int]]:
    candidates: list[tuple[tuple[int, int, int, int], int]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in _components(grid == color):
            box = _bbox_from_coords(coords)
            r0, c0, r1, c1 = box
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 3 or w < 3 or len(coords) != 2 * h + 2 * w - 4:
                continue
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            border = np.zeros((h, w), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if np.all(patch[border] == color) and not np.any(patch[~border] == color):
                candidates.append((box, color))
    candidates.sort(key=lambda item: item[0])
    assert candidates, "expected at least one rectangular frame"
    return candidates


def _frame_infos(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    infos: list[dict] = []
    for box, frame_color in _find_frames(g, bg):
        r0, c0, r1, c1 = box
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_mask = (interior != bg) & (interior != frame_color)
        comps = _components(seed_mask)
        assert len(comps) == 1, "each frame must contain exactly one seed component"
        seed_coords = comps[0]
        seed_box = _bbox_from_coords(seed_coords)
        seed_patch = _crop(interior, seed_box).copy()
        seed_colors = {int(v) for v in np.unique(seed_patch) if int(v) != bg}
        assert len(seed_colors) == 1, "seed bbox must contain exactly one non-background color"
        holes = int(np.sum(seed_patch == bg))
        assert holes >= 1, "seed bbox must contain at least one background hole"
        infos.append(
            {
                "bbox": box,
                "frame_color": int(frame_color),
                "interior_shape": interior.shape,
                "seed_bbox": seed_box,
                "seed_patch": seed_patch,
                "holes": holes,
                "seed_color": next(iter(seed_colors)),
            }
        )
    return infos


def _render_with_factor_rule(grid: np.ndarray, factor_fn) -> np.ndarray | None:
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    out = g.copy()
    try:
        for info in _frame_infos(g):
            factor = int(factor_fn(info, bg))
            if factor <= 0:
                return None
            scaled = _scale_patch(info["seed_patch"], factor)
            if scaled.shape != info["interior_shape"]:
                return None
            r0, c0, r1, c1 = info["bbox"]
            out[r0 + 1 : r1, c0 + 1 : c1] = bg
            out[r0 + 1 : r1, c0 + 1 : c1] = scaled
    except Exception:
        return None
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count each framed seed pattern's background holes and use that count as the pixel-upscale factor that fills the frame interior.

    Input structure:
    - A black background contains several non-overlapping single-cell-thick rectangular frames of different colors.
    - Each frame encloses one small 4-connected seed pattern in a different non-background color, with the seed sitting near the frame interior's upper-left region.
    - The seed's tight bounding box contains both seed-color pixels and black background holes.

    Rule:
    - For each frame, isolate the enclosed seed component and crop its tight bounding box.
    - Count the black pixels inside that tight box; this hole count H is the frame's upscale factor.
    - Replace every pixel of the tight box, including black holes, by an HxH block of the same color.
    - Clear the frame interior to black and write the enlarged box back aligned to the frame interior's upper-left corner, preserving the frame border and everything outside the frame.

    Key insight:
    - The seed's internal black holes are not noise: they encode the enlargement factor, and the frame interior size confirms the count because the correctly scaled tight box fits it exactly.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _dominant_color(g)
    out = g.copy()

    for info in _frame_infos(g):
        factor = int(info["holes"])
        scaled = _scale_patch(info["seed_patch"], factor)
        assert scaled.shape == info["interior_shape"], "hole count must exactly match frame interior size"
        r0, c0, r1, c1 = info["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = bg
        out[r0 + 1 : r1, c0 + 1 : c1] = scaled

    return out


def _is_connected(mask: np.ndarray) -> bool:
    return len(_components(mask)) == 1


@lru_cache(maxsize=1)
def _seed_bank() -> dict[int, list[np.ndarray]]:
    bank: dict[int, list[np.ndarray]] = {}
    for h in (2, 3):
        for w in (2, 3):
            area = h * w
            for bits in range(1, 1 << area):
                fill = bits.bit_count()
                holes = area - fill
                if fill < 3 or fill > 8 or holes < 2:
                    continue
                mask = np.zeros((h, w), dtype=bool)
                for idx in range(area):
                    if bits & (1 << idx):
                        mask[idx // w, idx % w] = True
                coords = np.argwhere(mask)
                if coords.size == 0:
                    continue
                if _bbox_from_coords(coords) != (0, 0, h - 1, w - 1):
                    continue
                if not _is_connected(mask):
                    continue
                bank.setdefault(holes, []).append(mask)

    for holes, masks in bank.items():
        unique: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
        for mask in masks:
            unique[(mask.shape, mask.astype(np.uint8).tobytes())] = mask
        bank[holes] = list(unique.values())
    assert all(holes in bank for holes in HOLE_COUNTS)
    return bank


def _pack_positions(total: int, lengths: list[int], min_gap: int, margin: int) -> list[int] | None:
    count = len(lengths)
    required = sum(lengths) + min_gap * (count - 1) + 2 * margin
    if required > total:
        return None
    extras = [0] * (count + 1)
    for _ in range(total - required):
        extras[random.randrange(count + 1)] += 1
    positions: list[int] = []
    cursor = margin + extras[0]
    for i, length in enumerate(lengths):
        positions.append(cursor)
        cursor += length
        if i + 1 < count:
            cursor += min_gap + extras[i + 1]
    return positions


def _sample_seed_masks() -> list[np.ndarray]:
    bank = _seed_bank()
    masks = [random.choice(bank[holes]).copy() for holes in HOLE_COUNTS]
    if not any(mask.shape[0] != mask.shape[1] for mask in masks):
        for hole_index in random.sample(range(len(HOLE_COUNTS)), len(HOLE_COUNTS)):
            rect_bank = [m for m in bank[HOLE_COUNTS[hole_index]] if m.shape[0] != m.shape[1]]
            if rect_bank:
                masks[hole_index] = random.choice(rect_bank).copy()
                break
    return masks


def _layout_positions(
    outer_shapes: list[tuple[int, int]]
) -> tuple[tuple[int, int], dict[int, tuple[int, int, int, int]]] | None:
    indices = list(range(len(outer_shapes)))
    for _ in range(40):
        random.shuffle(indices)
        split = random.choice((1, 2))
        rows = [indices[:split], indices[split:]]
        row_heights = [max(outer_shapes[i][0] for i in row) for row in rows]
        row_widths = [sum(outer_shapes[i][1] for i in row) + MIN_FRAME_GAP * (len(row) - 1) for row in rows]
        min_h = sum(row_heights) + MIN_FRAME_GAP + 2 * MIN_MARGIN
        min_w = max(row_widths) + 2 * MIN_MARGIN
        if min_h > MAX_GRID_SIDE or min_w > MAX_GRID_SIDE:
            continue
        grid_h = random.randint(min_h, min(MAX_GRID_SIDE, min_h + 4))
        grid_w = random.randint(min_w, min(MAX_GRID_SIDE, min_w + 4))
        row_tops = _pack_positions(grid_h, row_heights, MIN_FRAME_GAP, MIN_MARGIN)
        if row_tops is None:
            continue
        placements: dict[int, tuple[int, int, int, int]] = {}
        ok = True
        for row_top, row in zip(row_tops, rows):
            widths = [outer_shapes[i][1] for i in row]
            cols = _pack_positions(grid_w, widths, MIN_FRAME_GAP, MIN_MARGIN)
            if cols is None:
                ok = False
                break
            for c0, idx in zip(cols, row):
                h, w = outer_shapes[idx]
                placements[idx] = (row_top, c0, row_top + h - 1, c0 + w - 1)
        if ok:
            return (grid_h, grid_w), placements
    return None


def _seed_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _draw_input(specs: list[dict], placements: dict[int, tuple[int, int, int, int]], shape: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for idx, spec in enumerate(specs):
        r0, c0, r1, c1 = placements[idx]
        grid[r0 : r1 + 1, c0 : c1 + 1] = BG
        frame_color = spec["frame_color"]
        grid[r0, c0 : c1 + 1] = frame_color
        grid[r1, c0 : c1 + 1] = frame_color
        grid[r0 : r1 + 1, c0] = frame_color
        grid[r0 : r1 + 1, c1] = frame_color

        inner_h, inner_w = spec["inner_shape"]
        seed = _seed_patch(spec["mask"], spec["seed_color"])
        max_seed_r = min(1, inner_h - seed.shape[0])
        max_seed_c = min(1, inner_w - seed.shape[1])
        seed_r = r0 + 1 + random.randint(0, max_seed_r)
        seed_c = c0 + 1 + random.randint(0, max_seed_c)
        seed_mask = seed != BG
        grid[seed_r : seed_r + seed.shape[0], seed_c : seed_c + seed.shape[1]][seed_mask] = seed[seed_mask]
    return grid


def _specs_from_masks(masks: list[np.ndarray]) -> list[dict]:
    colors = random.sample(PALETTE, 2 * len(masks))
    frame_colors = colors[: len(masks)]
    seed_colors = colors[len(masks) :]
    specs: list[dict] = []
    for holes, mask, frame_color, seed_color in zip(HOLE_COUNTS, masks, frame_colors, seed_colors):
        inner_shape = (mask.shape[0] * holes, mask.shape[1] * holes)
        specs.append(
            {
                "holes": holes,
                "mask": mask,
                "frame_color": frame_color,
                "seed_color": seed_color,
                "inner_shape": inner_shape,
                "outer_shape": (inner_shape[0] + 2, inner_shape[1] + 2),
            }
        )
    return specs


def _semantic_checks(grid: np.ndarray) -> bool:
    infos = _frame_infos(grid)
    if len(infos) != 3:
        return False
    holes = sorted(int(info["holes"]) for info in infos)
    if holes != list(HOLE_COUNTS):
        return False
    for info in infos:
        seed_patch = info["seed_patch"]
        if int(np.sum(seed_patch != BG)) < 3:
            return False
        if tuple(np.array(seed_patch.shape) * info["holes"]) != tuple(info["interior_shape"]):
            return False

    correct = main(grid)
    if np.array_equal(correct, grid):
        return False
    if not any(info["seed_patch"].shape[0] != info["seed_patch"].shape[1] for info in infos):
        return False

    shortcut_rules = [
        lambda info, bg: 2,
        lambda info, bg: 3,
        lambda info, bg: info["seed_patch"].shape[0],
        lambda info, bg: info["seed_patch"].shape[1],
        lambda info, bg: int(np.sum(info["seed_patch"] != bg)),
    ]
    for rule in shortcut_rules:
        alt = _render_with_factor_rule(grid, rule)
        if alt is not None and np.array_equal(alt, correct):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    masks = _sample_seed_masks()
    specs = _specs_from_masks(masks)
    layout = _layout_positions([spec["outer_shape"] for spec in specs])
    if layout is None:
        return None
    shape, placements = layout
    grid = _draw_input(specs, placements, shape)
    try:
        if _semantic_checks(grid):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    """
    Generate a valid input for the hole-count-as-upscale puzzle.

    Note:
    - The solver handles any number of frames.
    - Under the simultaneous constraints here (distinct hole counts >= 2, seed size 3-8, bbox dimensions <= 3, full output bounded by 30x30),
      four distinct frames are not packable in all cases, so the generator emits the maximal robust three-frame variant.
    """
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
