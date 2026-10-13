import math
from collections import Counter, defaultdict

import numpy as np


BG = 0
DEFAULT_MARKER = 2
TEMPLATE_COLOR = 5
BORDER_COLORS = (1, 3, 4, 6, 7, 8, 9)
LAYOUT_OPTIONS = ((2, 2), (2, 3), (3, 3))
CELL_SPAN_OPTIONS = (1, 3)
MAX_GEN_ATTEMPTS = 300
PATTERN_ATTEMPTS = 200


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _extract_window(grid: np.ndarray, color: int, coords: np.ndarray, bg: int) -> dict | None:
    r0 = int(coords[:, 0].min())
    r1 = int(coords[:, 0].max())
    c0 = int(coords[:, 1].min())
    c1 = int(coords[:, 1].max())
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    h, w = patch.shape

    line_rows = tuple(int(r) for r in range(h) if np.all(patch[r, :] == color))
    line_cols = tuple(int(c) for c in range(w) if np.all(patch[:, c] == color))
    if len(line_rows) < 3 or len(line_cols) < 3:
        return None
    if line_rows[0] != 0 or line_rows[-1] != h - 1 or line_cols[0] != 0 or line_cols[-1] != w - 1:
        return None

    row_steps = np.diff(np.array(line_rows, dtype=int))
    col_steps = np.diff(np.array(line_cols, dtype=int))
    if len(set(map(int, row_steps))) != 1 or len(set(map(int, col_steps))) != 1:
        return None
    row_step = int(row_steps[0])
    col_step = int(col_steps[0])
    if row_step < 2 or col_step < 2 or row_step % 2 != 0 or col_step % 2 != 0:
        return None

    cells_r = len(line_rows) - 1
    cells_c = len(line_cols) - 1
    if cells_r not in (2, 3) or cells_c not in (2, 3):
        return None

    line_mask = np.zeros((h, w), dtype=bool)
    for rr in line_rows:
        line_mask[rr, :] = True
    for cc in line_cols:
        line_mask[:, cc] = True

    if np.any(patch[line_mask] != color):
        return None
    if np.any(patch[~line_mask] == color):
        return None
    if int(np.sum(patch == color)) != int(np.sum(line_mask)):
        return None

    center_rows = tuple((line_rows[i] + line_rows[i + 1]) // 2 for i in range(cells_r))
    center_cols = tuple((line_cols[j] + line_cols[j + 1]) // 2 for j in range(cells_c))
    center_mask = np.zeros((h, w), dtype=bool)
    marks = np.zeros((cells_r, cells_c), dtype=bool)
    for i, rr in enumerate(center_rows):
        for j, cc in enumerate(center_cols):
            center_mask[rr, cc] = True
            v = int(patch[rr, cc])
            marks[i, j] = v != bg and v != color

    other_mask = (~line_mask) & (~center_mask)
    if np.any(patch[other_mask] != bg):
        return None

    return {
        "color": int(color),
        "r0": r0,
        "r1": r1,
        "c0": c0,
        "c1": c1,
        "shape": (h, w),
        "line_rows": line_rows,
        "line_cols": line_cols,
        "center_rows": center_rows,
        "center_cols": center_cols,
        "marks": marks,
    }


def _render_window(color: int, marks: np.ndarray, cell_h: int, cell_w: int, marker_color: int) -> np.ndarray:
    rows, cols = map(int, marks.shape)
    h = rows * (cell_h + 1) + 1
    w = cols * (cell_w + 1) + 1
    patch = np.full((h, w), BG, dtype=int)

    line_rows = [i * (cell_h + 1) for i in range(rows + 1)]
    line_cols = [j * (cell_w + 1) for j in range(cols + 1)]
    for rr in line_rows:
        patch[rr, :] = color
    for cc in line_cols:
        patch[:, cc] = color

    for i in range(rows):
        rr = (line_rows[i] + line_rows[i + 1]) // 2
        for j in range(cols):
            if marks[i, j]:
                cc = (line_cols[j] + line_cols[j + 1]) // 2
                patch[rr, cc] = marker_color
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored windows, XOR their red markers cell-by-cell, then OR those
    XOR results into the blank template window and return only that window patch.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = BG if np.any(g == BG) else int(Counter(map(int, g.flat)).most_common(1)[0][0])

    windows: list[dict] = []
    for color in (int(c) for c in np.unique(g) if int(c) != bg):
        for coords in _components(g == color):
            window = _extract_window(g, int(color), coords, bg)
            if window is not None:
                windows.append(window)

    assert windows, "expected at least one valid window"

    marker_counts: Counter[int] = Counter()
    for win in windows:
        patch = g[win["r0"] : win["r1"] + 1, win["c0"] : win["c1"] + 1]
        for i, rr in enumerate(win["center_rows"]):
            for j, cc in enumerate(win["center_cols"]):
                if not win["marks"][i, j]:
                    continue
                v = int(patch[rr, cc])
                if v != bg and v != win["color"]:
                    marker_counts[v] += 1
    marker_color = int(marker_counts.most_common(1)[0][0]) if marker_counts else DEFAULT_MARKER

    groups: dict[int, list[dict]] = defaultdict(list)
    for win in windows:
        groups[win["color"]].append(win)

    template_candidates = [group[0] for group in groups.values() if len(group) == 1 and not group[0]["marks"].any()]
    if not template_candidates:
        template_candidates = [group[0] for group in groups.values() if len(group) == 1]
    assert len(template_candidates) == 1, "expected exactly one blank template window"

    template = template_candidates[0]
    ref_shape = template["shape"]
    ref_marks_shape = template["marks"].shape
    ref_line_rows = template["line_rows"]
    ref_line_cols = template["line_cols"]

    combined = np.zeros(ref_marks_shape, dtype=bool)
    for color, group in groups.items():
        if color == template["color"]:
            continue
        assert len(group) == 2, "every non-template color must appear exactly twice"
        a, b = group
        assert a["shape"] == ref_shape and b["shape"] == ref_shape
        assert a["marks"].shape == ref_marks_shape and b["marks"].shape == ref_marks_shape
        assert a["line_rows"] == ref_line_rows and b["line_rows"] == ref_line_rows
        assert a["line_cols"] == ref_line_cols and b["line_cols"] == ref_line_cols
        combined |= np.logical_xor(a["marks"], b["marks"])

    out = np.full(ref_shape, bg, dtype=int)
    for rr in ref_line_rows:
        out[rr, :] = template["color"]
    for cc in ref_line_cols:
        out[:, cc] = template["color"]
    for i, rr in enumerate(template["center_rows"]):
        for j, cc in enumerate(template["center_cols"]):
            if combined[i, j]:
                out[rr, cc] = marker_color
    return out


def _generate_pair_marks(
    rng: np.random.Generator, rows: int, cols: int, num_pairs: int
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    n_cells = rows * cols
    for _ in range(PATTERN_ATTEMPTS):
        common_idx = int(rng.integers(0, n_cells))
        pair_xors: list[np.ndarray] = []
        for pair_idx in range(num_pairs):
            include_common = pair_idx < 2 or bool(rng.integers(0, 2))
            count = int(rng.integers(2, n_cells))
            mask = np.zeros(n_cells, dtype=bool)
            pool = [idx for idx in range(n_cells) if idx != common_idx]
            if include_common:
                mask[common_idx] = True
                extras = rng.choice(pool, size=count - 1, replace=False)
            else:
                extras = rng.choice(pool, size=count, replace=False)
            mask[np.asarray(extras, dtype=int)] = True
            pair_xors.append(mask.reshape(rows, cols))

        final_marks = np.logical_or.reduce(pair_xors)
        final_count = int(final_marks.sum())
        if final_count <= 1 or final_count >= n_cells:
            continue

        cancel_pair = int(rng.integers(0, num_pairs))
        pair_marks: list[tuple[np.ndarray, np.ndarray]] = []
        cancel_ok = False
        for pair_idx, xor_mask in enumerate(pair_xors):
            flat = xor_mask.reshape(-1)
            xor_idxs = rng.permutation(np.flatnonzero(flat))
            split = int(rng.integers(1, len(xor_idxs)))
            a = np.zeros(n_cells, dtype=bool)
            b = np.zeros(n_cells, dtype=bool)
            a[xor_idxs[:split]] = True
            b[xor_idxs[split:]] = True

            false_idxs = np.flatnonzero(~flat)
            shared = np.zeros(n_cells, dtype=bool)
            if len(false_idxs) > 0 and (pair_idx == cancel_pair or rng.random() < 0.5):
                max_shared = min(2, len(false_idxs))
                shared_count = 1 if pair_idx == cancel_pair else int(rng.integers(0, max_shared + 1))
                if shared_count > 0:
                    chosen = rng.choice(false_idxs, size=shared_count, replace=False)
                    shared[np.asarray(chosen, dtype=int)] = True
            a[shared] = True
            b[shared] = True
            if pair_idx == cancel_pair and shared.any():
                cancel_ok = True
            pair_marks.append((a.reshape(rows, cols), b.reshape(rows, cols)))

        if cancel_ok and pair_xors[0].reshape(-1)[common_idx] and pair_xors[1].reshape(-1)[common_idx]:
            return pair_marks, final_marks

    raise RuntimeError("failed to generate non-degenerate marker patterns")


def _grid_shape_for_windows(
    patch_h: int, patch_w: int, num_windows: int, rng: np.random.Generator
) -> tuple[int, int, list[tuple[int, int]]]:
    gap_h = int(rng.integers(2, 5))
    gap_w = int(rng.integers(2, 5))
    candidates: list[tuple[int, int, int, int]] = []
    for slot_rows in range(2, min(4, num_windows) + 1):
        slot_cols = int(math.ceil(num_windows / slot_rows))
        height = slot_rows * patch_h + (slot_rows + 1) * gap_h
        width = slot_cols * patch_w + (slot_cols + 1) * gap_w
        candidates.append((max(height, width), abs(height - width), slot_rows, slot_cols))
    _, _, slot_rows, slot_cols = min(candidates)

    height = slot_rows * patch_h + (slot_rows + 1) * gap_h
    width = slot_cols * patch_w + (slot_cols + 1) * gap_w
    slots = [(sr, sc) for sr in range(slot_rows) for sc in range(slot_cols)]
    chosen = rng.permutation(len(slots))[:num_windows]
    positions: list[tuple[int, int]] = []
    for slot_idx in chosen:
        sr, sc = slots[int(slot_idx)]
        base_r = gap_h + sr * (patch_h + gap_h)
        base_c = gap_w + sc * (patch_w + gap_w)
        jitter_r = int(rng.integers(0, gap_h))
        jitter_c = int(rng.integers(0, gap_w))
        positions.append((base_r + jitter_r, base_c + jitter_c))
    return height, width, positions


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    rows, cols = LAYOUT_OPTIONS[int(rng.integers(0, len(LAYOUT_OPTIONS)))]
    cell_h = CELL_SPAN_OPTIONS[int(rng.integers(0, len(CELL_SPAN_OPTIONS)))]
    cell_w = CELL_SPAN_OPTIONS[int(rng.integers(0, len(CELL_SPAN_OPTIONS)))]

    probe = _render_window(TEMPLATE_COLOR, np.zeros((rows, cols), dtype=bool), cell_h, cell_w, DEFAULT_MARKER)
    patch_h, patch_w = map(int, probe.shape)

    if max(patch_h, patch_w) <= 9 and rng.random() < 0.5:
        num_pairs = 3
    else:
        num_pairs = 2

    pair_marks, final_marks = _generate_pair_marks(rng, rows, cols, num_pairs)
    colors = rng.choice(np.array(BORDER_COLORS, dtype=int), size=num_pairs, replace=False)

    patches: list[np.ndarray] = [_render_window(TEMPLATE_COLOR, np.zeros((rows, cols), dtype=bool), cell_h, cell_w, DEFAULT_MARKER)]
    for color, (marks_a, marks_b) in zip(colors, pair_marks):
        patches.append(_render_window(int(color), marks_a, cell_h, cell_w, DEFAULT_MARKER))
        patches.append(_render_window(int(color), marks_b, cell_h, cell_w, DEFAULT_MARKER))
    rng.shuffle(patches)

    height, width, positions = _grid_shape_for_windows(patch_h, patch_w, len(patches), rng)
    grid = np.full((height, width), BG, dtype=int)
    for patch, (r0, c0) in zip(patches, positions):
        ph, pw = patch.shape
        grid[r0 : r0 + ph, c0 : c0 + pw] = patch

    expected = _render_window(TEMPLATE_COLOR, final_marks, cell_h, cell_w, DEFAULT_MARKER)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
