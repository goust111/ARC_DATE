from __future__ import annotations

import random
from collections import Counter

import numpy as np


_ORIENTATIONS = (
    lambda arr: np.array(arr, copy=True),
    lambda arr: np.rot90(arr, 1).copy(),
    lambda arr: np.rot90(arr, 2).copy(),
    lambda arr: np.rot90(arr, 3).copy(),
    lambda arr: np.flipud(arr).copy(),
    lambda arr: np.fliplr(arr).copy(),
    lambda arr: arr.T.copy(),
    lambda arr: np.flipud(np.fliplr(arr)).T.copy(),
)
_INVERSE_ORIENTATION = (0, 3, 2, 1, 4, 5, 6, 7)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    mask = grid == int(color)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and mask[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _line_positions(lengths: list[int]) -> list[int]:
    pos = [0]
    cur = 0
    for size in lengths:
        cur += int(size) + 1
        pos.append(cur)
    return pos


def _apply_orientation(arr: np.ndarray, orient: int) -> np.ndarray:
    return _ORIENTATIONS[int(orient)](arr)


def _parse_lattice_component(
    grid: np.ndarray, bg: int, frame_color: int, coords: np.ndarray
) -> dict | None:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    frame_mask = sub == int(frame_color)

    row_lines = [r for r in range(frame_mask.shape[0]) if bool(frame_mask[r, :].all())]
    col_lines = [c for c in range(frame_mask.shape[1]) if bool(frame_mask[:, c].all())]
    if len(row_lines) < 2 or len(col_lines) < 2:
        return None
    if row_lines[0] != 0 or row_lines[-1] != frame_mask.shape[0] - 1:
        return None
    if col_lines[0] != 0 or col_lines[-1] != frame_mask.shape[1] - 1:
        return None
    if any(a + 1 >= b for a, b in zip(row_lines, row_lines[1:])):
        return None
    if any(a + 1 >= b for a, b in zip(col_lines, col_lines[1:])):
        return None

    expected = np.zeros_like(frame_mask, dtype=bool)
    expected[row_lines, :] = True
    expected[:, col_lines] = True
    if not np.array_equal(frame_mask, expected):
        return None

    slot_rows = len(row_lines) - 1
    slot_cols = len(col_lines) - 1
    known = np.full((slot_rows, slot_cols), -1, dtype=int)
    for sr in range(slot_rows):
        for sc in range(slot_cols):
            region = sub[row_lines[sr] + 1 : row_lines[sr + 1], col_lines[sc] + 1 : col_lines[sc + 1]]
            vals = sorted(
                int(v)
                for v in np.unique(region)
                if int(v) not in (int(bg), int(frame_color))
            )
            if not vals:
                continue
            if len(vals) != 1:
                return None
            known[sr, sc] = vals[0]

    return {
        "bbox": (r0, c0, r1, c1),
        "row_lines": [r0 + r for r in row_lines],
        "col_lines": [c0 + c for c in col_lines],
        "slot_shape": (slot_rows, slot_cols),
        "known": known,
        "known_count": int(np.count_nonzero(known >= 0)),
    }


def _extract_frames(grid: np.ndarray) -> tuple[int, int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    candidates: dict[tuple[int, int, int], list[dict]] = {}
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        parsed: list[dict] = []
        for coords in _find_components_of_color(g, color):
            frame = _parse_lattice_component(g, bg, color, coords)
            if frame is not None:
                parsed.append(frame)
        if not parsed:
            continue
        key = (color, parsed[0]["slot_shape"][0], parsed[0]["slot_shape"][1])
        if all(frame["slot_shape"] == parsed[0]["slot_shape"] for frame in parsed):
            candidates[key] = parsed

    assert candidates, "expected lattice frames"
    ranked = sorted(
        candidates.items(),
        key=lambda item: (
            -len(item[1]),
            -sum((frame["bbox"][2] - frame["bbox"][0] + 1) * (frame["bbox"][3] - frame["bbox"][1] + 1) for frame in item[1]),
            item[0],
        ),
    )
    best_key, frames = ranked[0]
    assert len(frames) >= 2, "expected at least two matching lattice frames"
    frame_color = int(best_key[0])
    slot_shape = frames[0]["slot_shape"]
    assert all(frame["slot_shape"] == slot_shape for frame in frames)
    return bg, frame_color, frames


def _solve_consensus(frames: list[dict]) -> tuple[np.ndarray, list[int]]:
    order = sorted(range(len(frames)), key=lambda idx: -frames[idx]["known_count"])
    ref_idx = order[0]
    options: list[list[tuple[int, np.ndarray]]] = []
    for frame in frames:
        opts = []
        for orient in range(8):
            canonical_view = _apply_orientation(frame["known"], _INVERSE_ORIENTATION[orient])
            opts.append((orient, canonical_view))
        options.append(opts)

    solutions: list[tuple[np.ndarray, list[int]]] = []

    def search(k: int, target: np.ndarray, chosen: list[int]) -> None:
        if len(solutions) > 1:
            return
        if k == len(order):
            if np.all(target >= 0):
                solutions.append((target.copy(), chosen.copy()))
            return

        frame_idx = order[k]
        if frame_idx == ref_idx:
            search(k + 1, target, chosen)
            return
        for orient, candidate in options[frame_idx]:
            if candidate.shape != target.shape:
                continue
            merged = target.copy()
            mask = candidate >= 0
            if np.any(mask & (merged >= 0) & (merged != candidate)):
                continue
            merged[mask] = candidate[mask]
            chosen[frame_idx] = orient
            search(k + 1, merged, chosen)
            chosen[frame_idx] = -1

    chosen = [-1] * len(frames)
    chosen[ref_idx] = 0
    search(0, frames[ref_idx]["known"].copy(), chosen)
    assert len(solutions) == 1, "frame orientations must determine a unique complete consensus"
    return solutions[0]


def _paint_frame(
    out: np.ndarray,
    frame: dict,
    local_pattern: np.ndarray,
    frame_color: int,
) -> None:
    rows = frame["row_lines"]
    cols = frame["col_lines"]
    slot_rows, slot_cols = local_pattern.shape
    assert slot_rows == len(rows) - 1 and slot_cols == len(cols) - 1

    for r in rows:
        out[int(r), int(cols[0]) : int(cols[-1]) + 1] = int(frame_color)
    for c in cols:
        out[int(rows[0]) : int(rows[-1]) + 1, int(c)] = int(frame_color)

    for sr in range(slot_rows):
        for sc in range(slot_cols):
            rr0 = int(rows[sr]) + 1
            rr1 = int(rows[sr + 1])
            cc0 = int(cols[sc]) + 1
            cc1 = int(cols[sc + 1])
            out[rr0:rr1, cc0:cc1] = int(local_pattern[sr, sc])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align the partially filled lattice frames into one consensus pattern, then complete every frame in its own orientation.

    Input structure:
    - The background is dominant, and one frame color draws several one-pixel-thick rectangular lattices.
    - Every lattice is split into the same logical slot grid, though slot pixel sizes may differ from frame to frame.
    - Only some slots are filled with non-frame colors in each lattice; the rest of the slot interiors stay background.

    Rule:
    - Parse each lattice into a logical slot matrix and test all rotation/reflection orientations for that frame.
    - Reverse each candidate orientation back into one shared mother-pattern coordinate system and keep only the unique combination whose known slots do not conflict and together determine every slot color.
    - Repaint every slot of every lattice from that solved mother pattern, transformed back into the lattice's own orientation, while preserving frame lines and outside background.

    Key insight:
    - The lattices are not direct translated copies: their clues agree only after undoing each frame's rotation or mirror, so orientation solving must happen before slotwise color aggregation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _bg, frame_color, frames = _extract_frames(g)
    canonical, orientations = _solve_consensus(frames)

    out = g.copy()
    for frame, orient in zip(frames, orientations):
        local_pattern = _apply_orientation(canonical, orient)
        _paint_frame(out, frame, local_pattern, frame_color)
    return out


_GEN_BG = 0
_GEN_FRAME_COUNT_RANGE = (3, 4)
_GEN_SLOT_DIM_OPTIONS = ((3, 3), (4, 4))
_GEN_CANVAS_RANGE = (24, 30)
_GEN_PADDING = 1
_GEN_ATTEMPTS = 400
_GEN_COLORS = tuple(range(1, 10))


def _is_pattern_asymmetric(pattern: np.ndarray) -> bool:
    base = np.array(pattern, copy=True)
    for orient in range(1, 8):
        other = _apply_orientation(base, orient)
        if other.shape == base.shape and np.array_equal(base, other):
            return False
    return True


def _sample_pattern(slot_shape: tuple[int, int], palette: list[int]) -> np.ndarray:
    rows, cols = slot_shape
    total = rows * cols
    cells = [random.choice(palette) for _ in range(total)]
    inject = random.sample(range(total), min(len(palette), total))
    for idx, color in zip(inject, palette):
        cells[idx] = int(color)
    pattern = np.asarray(cells, dtype=int).reshape(rows, cols)
    return pattern


def _sample_visible_masks(slot_shape: tuple[int, int], n_frames: int) -> list[np.ndarray] | None:
    rows, cols = slot_shape
    total = rows * cols
    min_known = max(2, total // 3)
    max_known = max(min_known, total - 1)
    for _ in range(120):
        masks: list[np.ndarray] = []
        coverage = np.zeros(total, dtype=int)
        for _frame_idx in range(n_frames):
            count = random.randint(min_known, max_known)
            picks = random.sample(range(total), count)
            mask = np.zeros(total, dtype=bool)
            mask[picks] = True
            masks.append(mask.reshape(rows, cols))
            coverage[mask] += 1
        if np.any(coverage == 0):
            continue
        if int(np.count_nonzero(coverage >= 2)) < min(rows, cols):
            continue
        if len({tuple(mask.ravel().tolist()) for mask in masks}) < 2:
            continue
        return masks
    return None


def _sample_frame_lengths(slot_count: int) -> list[int]:
    return [random.randint(1, 2) for _ in range(slot_count)]


def _render_patch(
    slot_pattern: np.ndarray,
    frame_color: int,
    bg: int,
    row_sizes: list[int],
    col_sizes: list[int],
    visible_mask: np.ndarray | None,
) -> np.ndarray:
    row_lines = _line_positions(row_sizes)
    col_lines = _line_positions(col_sizes)
    patch = np.full((row_lines[-1] + 1, col_lines[-1] + 1), int(bg), dtype=int)
    patch[row_lines, :] = int(frame_color)
    patch[:, col_lines] = int(frame_color)

    for sr in range(slot_pattern.shape[0]):
        for sc in range(slot_pattern.shape[1]):
            if visible_mask is not None and not bool(visible_mask[sr, sc]):
                continue
            patch[
                row_lines[sr] + 1 : row_lines[sr + 1],
                col_lines[sc] + 1 : col_lines[sc + 1],
            ] = int(slot_pattern[sr, sc])
    return patch


def _boxes_separated(boxes: list[tuple[int, int, int, int]], candidate: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = candidate
    for er0, ec0, er1, ec1 in boxes:
        if not (r1 + _GEN_PADDING < er0 or er1 + _GEN_PADDING < r0 or c1 + _GEN_PADDING < ec0 or ec1 + _GEN_PADDING < c0):
            return False
    return True


def _place_patches(rows: int, cols: int, patches: list[np.ndarray]) -> list[tuple[int, int]] | None:
    order = sorted(range(len(patches)), key=lambda idx: patches[idx].shape[0] * patches[idx].shape[1], reverse=True)
    for _ in range(32):
        positions: list[tuple[int, int] | None] = [None] * len(patches)
        boxes: list[tuple[int, int, int, int]] = []
        ok = True
        for idx in order:
            patch = patches[idx]
            h, w = patch.shape
            if h + 2 > rows or w + 2 > cols:
                return None
            options: list[tuple[int, int]] = []
            for r0 in range(1, rows - h):
                for c0 in range(1, cols - w):
                    box = (r0, c0, r0 + h - 1, c0 + w - 1)
                    if _boxes_separated(boxes, box):
                        options.append((r0, c0))
            if not options:
                ok = False
                break
            r0, c0 = random.choice(options)
            positions[idx] = (r0, c0)
            boxes.append((r0, c0, r0 + h - 1, c0 + w - 1))
        if ok:
            return [pos for pos in positions if pos is not None]
    return None


def _attempt_generate() -> np.ndarray | None:
    n_frames = random.randint(*_GEN_FRAME_COUNT_RANGE)
    slot_shape = random.choice(_GEN_SLOT_DIM_OPTIONS)
    colors = random.sample(_GEN_COLORS, random.randint(5, 7))
    frame_color = colors[0]
    palette = colors[1:]

    pattern = _sample_pattern(slot_shape, palette)
    if not _is_pattern_asymmetric(pattern):
        return None

    masks = _sample_visible_masks(slot_shape, n_frames)
    if masks is None:
        return None

    orientations = [random.choice((1, 2, 3)), random.choice((4, 5, 6, 7))]
    while len(orientations) < n_frames:
        orientations.append(random.randrange(8))
    random.shuffle(orientations)
    if len(set(orientations)) < 2:
        return None

    input_patches: list[np.ndarray] = []
    output_patches: list[np.ndarray] = []
    for orient, mask in zip(orientations, masks):
        row_sizes = _sample_frame_lengths(slot_shape[0])
        col_sizes = _sample_frame_lengths(slot_shape[1])
        local = _apply_orientation(pattern, orient)
        input_patches.append(_render_patch(local, frame_color, _GEN_BG, row_sizes, col_sizes, mask))
        output_patches.append(_render_patch(local, frame_color, _GEN_BG, row_sizes, col_sizes, np.ones(local.shape, dtype=bool)))

    min_rows = max(patch.shape[0] for patch in input_patches) + 2
    min_cols = max(patch.shape[1] for patch in input_patches) + 2
    rows = random.randint(max(_GEN_CANVAS_RANGE[0], min_rows), _GEN_CANVAS_RANGE[1])
    cols = random.randint(max(_GEN_CANVAS_RANGE[0], min_cols), _GEN_CANVAS_RANGE[1])
    placements = _place_patches(rows, cols, input_patches)
    if placements is None:
        return None

    inp = np.full((rows, cols), _GEN_BG, dtype=int)
    out = np.full((rows, cols), _GEN_BG, dtype=int)
    for (r0, c0), in_patch, out_patch in zip(placements, input_patches, output_patches):
        h, w = in_patch.shape
        inp[r0 : r0 + h, c0 : c0 + w] = in_patch
        out[r0 : r0 + h, c0 : c0 + w] = out_patch

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rotating-lattice consensus input")
