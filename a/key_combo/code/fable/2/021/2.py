from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
RED = 2
BOARD_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)

BBox = tuple[int, int, int, int]

_GEN_CAVITY_CHOICES = (6, 8, 9, 10, 12)
_GEN_MAX_SKELETON_DIM = 6
_GEN_SCENARIOS = (
    {"need_unequal_scale": True, "target_exact_size": False, "extra_lure": False},
    {"need_unequal_scale": True, "target_exact_size": True, "extra_lure": False},
    {"need_unequal_scale": False, "target_exact_size": False, "extra_lure": True},
    {"need_unequal_scale": True, "target_exact_size": True, "extra_lure": True},
)
_GEN_SCENARIO_BAG: list[dict[str, bool]] = []


def _bbox_from_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox(mask: np.ndarray) -> BBox | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return _bbox_from_coords(coords)


def _crop_bbox(grid: np.ndarray, box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.asarray(cells, dtype=int))
    return components


def _find_hollow_frame(grid: np.ndarray) -> tuple[int, BBox, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    candidates: list[tuple[int, BBox, np.ndarray]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != BLACK):
        for coords in _mask_components(g == color):
            box = _bbox_from_coords(coords)
            r0, c0, r1, c1 = box
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 3 or w < 3:
                continue
            local = np.zeros((h, w), dtype=bool)
            shifted = coords - np.array([r0, c0], dtype=int)
            local[shifted[:, 0], shifted[:, 1]] = True
            frame = np.zeros((h, w), dtype=bool)
            frame[0, :] = True
            frame[-1, :] = True
            frame[:, 0] = True
            frame[:, -1] = True
            if np.array_equal(local, frame):
                full_mask = np.zeros_like(g, dtype=bool)
                full_mask[coords[:, 0], coords[:, 1]] = True
                candidates.append((color, box, full_mask))
    assert len(candidates) == 1, "expected one hollow rectangular frame"
    return candidates[0]


def _extract_partition_skeleton(board: np.ndarray) -> np.ndarray:
    g = np.asarray(board, dtype=int)
    h, w = g.shape
    assert h >= 2 and w >= 2
    assert np.all(g != BLACK), "board crop must be fully filled"

    top_lefts: list[tuple[int, int]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != BLACK):
        for coords in _mask_components(g == color):
            box = _bbox_from_coords(coords)
            r0, c0, r1, c1 = box
            block = g[r0 : r1 + 1, c0 : c1 + 1]
            assert np.all(block == color), "each partition must be a solid rectangle"
            top_lefts.append((r0, c0))

    rows = sorted({r for r, _ in top_lefts})
    cols = sorted({c for _, c in top_lefts})
    assert len(top_lefts) == len(rows) * len(cols), "partition top-lefts must form a full row/column skeleton"

    row_edges = rows[1:] + [h]
    col_edges = cols[1:] + [w]
    skeleton = np.empty((len(rows), len(cols)), dtype=int)
    for i, r0 in enumerate(rows):
        for j, c0 in enumerate(cols):
            r1 = row_edges[i]
            c1 = col_edges[j]
            color = int(g[r0, c0])
            assert np.all(g[r0:r1, c0:c1] == color), "row/column partition intervals must reconstruct the board"
            skeleton[i, j] = color
    return skeleton


def _parse_boards(grid: np.ndarray) -> tuple[int, BBox, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    frame_color, frame_box, frame_mask = _find_hollow_frame(g)

    fg_mask = g != BLACK
    fg_mask[frame_mask] = False
    boards: list[dict[str, object]] = []
    for coords in _mask_components(fg_mask):
        box = _bbox_from_coords(coords)
        crop = _crop_bbox(g, box)
        assert np.all(crop != BLACK), "each board must fill its bounding box"
        skeleton = _extract_partition_skeleton(crop)
        boards.append({"bbox": box, "crop": crop, "skeleton": skeleton})

    assert len(boards) >= 2, "expected at least two candidate boards"
    return frame_color, frame_box, boards


def _scale_skeleton(skeleton: np.ndarray, row_factor: int, col_factor: int) -> np.ndarray:
    return np.repeat(np.repeat(np.asarray(skeleton, dtype=int), row_factor, axis=0), col_factor, axis=1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the only partition skeleton whose row and column counts divide the red frame cavity, then upscale it to fill the frame crop.

    Input structure:
    - A black background contains one red hollow rectangular frame whose interior is empty.
    - Elsewhere, two or more separated multicolor boards are fully tiled by monochromatic rectangular partitions.
    - Each board's partition upper-left coordinates define a compressed row/column skeleton.

    Rule:
    - Detect each board and compress it by sampling one color at every partition upper-left coordinate.
    - Find the unique board whose skeleton row count divides the frame-interior height and whose skeleton column count divides the frame-interior width.
    - Repeat the chosen skeleton by those vertical and horizontal integer factors, fill the frame interior with it, and return only the frame crop.

    Key insight:
    - Matching happens in skeleton coordinates rather than raw pixel size, so a board whose original bbox already fits the frame can still be a distractor.
    """
    g = np.asarray(grid, dtype=int)
    frame_color, frame_box, boards = _parse_boards(g)
    r0, c0, r1, c1 = frame_box
    cavity_h = r1 - r0 - 1
    cavity_w = c1 - c0 - 1
    assert cavity_h >= 1 and cavity_w >= 1

    candidates: list[np.ndarray] = []
    for board in boards:
        skeleton = np.asarray(board["skeleton"], dtype=int)
        if cavity_h % skeleton.shape[0] == 0 and cavity_w % skeleton.shape[1] == 0:
            candidates.append(skeleton)
    assert len(candidates) == 1, "expected exactly one divisible skeleton"

    chosen = candidates[0]
    row_factor = cavity_h // chosen.shape[0]
    col_factor = cavity_w // chosen.shape[1]
    filled = _scale_skeleton(chosen, row_factor, col_factor)

    out = np.full((r1 - r0 + 1, c1 - c0 + 1), BLACK, dtype=int)
    out[0, :] = frame_color
    out[-1, :] = frame_color
    out[:, 0] = frame_color
    out[:, -1] = frame_color
    out[1:-1, 1:-1] = filled
    return out


def _next_scenario() -> dict[str, bool]:
    global _GEN_SCENARIO_BAG
    if not _GEN_SCENARIO_BAG:
        _GEN_SCENARIO_BAG = [dict(s) for s in _GEN_SCENARIOS]
        random.shuffle(_GEN_SCENARIO_BAG)
    return _GEN_SCENARIO_BAG.pop()


def _divisors(n: int) -> list[int]:
    return [d for d in range(2, min(n, _GEN_MAX_SKELETON_DIM) + 1) if n % d == 0]


def _nondivisors(n: int) -> list[int]:
    return [d for d in range(2, min(n, _GEN_MAX_SKELETON_DIM) + 1) if n % d != 0]


def _choose_frame_and_target(rng: np.random.Generator, need_unequal_scale: bool) -> tuple[int, int, int, int]:
    for _ in range(400):
        cavity_h = int(rng.choice(_GEN_CAVITY_CHOICES))
        cavity_w = int(rng.choice(_GEN_CAVITY_CHOICES))
        row_opts = [d for d in _divisors(cavity_h) if cavity_h // d <= 4]
        col_opts = [d for d in _divisors(cavity_w) if cavity_w // d <= 4]
        if not row_opts or not col_opts or not _nondivisors(cavity_h) or not _nondivisors(cavity_w):
            continue
        row_order = list(map(int, rng.permutation(row_opts)))
        col_order = list(map(int, rng.permutation(col_opts)))
        for rows in row_order:
            for cols in col_order:
                row_factor = cavity_h // rows
                col_factor = cavity_w // cols
                if row_factor == 1 and col_factor == 1:
                    continue
                if need_unequal_scale and row_factor == col_factor:
                    continue
                if not need_unequal_scale and row_factor != col_factor:
                    continue
                return cavity_h, cavity_w, rows, cols
    raise RuntimeError("failed to choose frame and target skeleton dimensions")


def _random_composition(total: int, parts: int, rng: np.random.Generator) -> list[int]:
    assert parts >= 1 and total >= parts
    remaining = total
    out: list[int] = []
    for i in range(parts - 1):
        lo = 1
        hi = remaining - (parts - i - 1)
        val = int(rng.integers(lo, hi + 1))
        out.append(val)
        remaining -= val
    out.append(int(remaining))
    rng.shuffle(out)
    return out


def _random_skeleton(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    for _ in range(200):
        palette_size = min(len(BOARD_COLORS), max(3, min(5, rows * cols)))
        palette = list(rng.choice(BOARD_COLORS, size=palette_size, replace=False))
        skel = np.zeros((rows, cols), dtype=int)
        for r in range(rows):
            for c in range(cols):
                choices = [
                    color
                    for color in palette
                    if (r == 0 or color != int(skel[r - 1, c])) and (c == 0 or color != int(skel[r, c - 1]))
                ]
                skel[r, c] = int(rng.choice(choices))
        if len(set(map(int, skel.flat))) >= min(3, rows * cols):
            return skel
    raise RuntimeError("failed to generate a non-degenerate skeleton")


def _render_board(skeleton: np.ndarray, row_sizes: list[int], col_sizes: list[int]) -> np.ndarray:
    rows, cols = skeleton.shape
    assert rows == len(row_sizes) and cols == len(col_sizes)
    board = np.zeros((sum(row_sizes), sum(col_sizes)), dtype=int)
    rr = 0
    for i, height in enumerate(row_sizes):
        cc = 0
        for j, width in enumerate(col_sizes):
            board[rr : rr + height, cc : cc + width] = int(skeleton[i, j])
            cc += width
        rr += height
    return board


def _choose_random_box_size(
    rows: int,
    cols: int,
    cavity_h: int,
    cavity_w: int,
    rng: np.random.Generator,
    *,
    exact: bool = False,
    forbid_exact: bool = False,
) -> tuple[int, int]:
    if exact:
        return cavity_h, cavity_w
    h_choices = [v for v in range(rows, min(12, rows * 4) + 1)]
    w_choices = [v for v in range(cols, min(12, cols * 4) + 1)]
    for _ in range(200):
        total_h = int(rng.choice(h_choices))
        total_w = int(rng.choice(w_choices))
        if forbid_exact and total_h == cavity_h and total_w == cavity_w:
            continue
        return total_h, total_w
    raise RuntimeError("failed to choose a board bbox size")


def _make_board(
    rows: int,
    cols: int,
    cavity_h: int,
    cavity_w: int,
    rng: np.random.Generator,
    *,
    exact_box: bool = False,
    forbid_exact_box: bool = False,
) -> dict[str, object]:
    total_h, total_w = _choose_random_box_size(
        rows,
        cols,
        cavity_h,
        cavity_w,
        rng,
        exact=exact_box,
        forbid_exact=forbid_exact_box,
    )
    skeleton = _random_skeleton(rows, cols, rng)
    row_sizes = _random_composition(total_h, rows, rng)
    col_sizes = _random_composition(total_w, cols, rng)
    board = _render_board(skeleton, row_sizes, col_sizes)
    return {
        "rows": rows,
        "cols": cols,
        "skeleton": skeleton,
        "row_sizes": row_sizes,
        "col_sizes": col_sizes,
        "board": board,
    }


def _make_frame_patch(cavity_h: int, cavity_w: int) -> np.ndarray:
    patch = np.zeros((cavity_h + 2, cavity_w + 2), dtype=int)
    patch[0, :] = RED
    patch[-1, :] = RED
    patch[:, 0] = RED
    patch[:, -1] = RED
    return patch


def _can_place(occupied: np.ndarray, shape: tuple[int, int], r0: int, c0: int, padding: int) -> bool:
    h, w = shape
    r1 = r0 + h
    c1 = c0 + w
    if r0 < 0 or c0 < 0 or r1 > occupied.shape[0] or c1 > occupied.shape[1]:
        return False
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(occupied.shape[0], r1 + padding)
    cc1 = min(occupied.shape[1], c1 + padding)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _place_rectangles(
    patches: list[np.ndarray], rng: np.random.Generator, canvas_h: int, canvas_w: int, padding: int = 1
) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    grid = np.zeros((canvas_h, canvas_w), dtype=int)
    occupied = np.zeros((canvas_h, canvas_w), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(patches)
    order = sorted(range(len(patches)), key=lambda idx: patches[idx].shape[0] * patches[idx].shape[1], reverse=True)
    for idx in order:
        patch = patches[idx]
        h, w = patch.shape
        candidates = [
            (r, c)
            for r in range(canvas_h - h + 1)
            for c in range(canvas_w - w + 1)
            if _can_place(occupied, (h, w), r, c, padding)
        ]
        if not candidates:
            return None
        r0, c0 = candidates[int(rng.integers(0, len(candidates)))]
        positions[idx] = (r0, c0)
        occupied[r0 : r0 + h, c0 : c0 + w] = True
        mask = patch != BLACK
        grid[r0 : r0 + h, c0 : c0 + w][mask] = patch[mask]
    return grid, [(int(r), int(c)) for r, c in positions if r is not None and c is not None]


def _expected_output(frame_color: int, cavity_h: int, cavity_w: int, target_skeleton: np.ndarray) -> np.ndarray:
    row_factor = cavity_h // target_skeleton.shape[0]
    col_factor = cavity_w // target_skeleton.shape[1]
    filled = _scale_skeleton(target_skeleton, row_factor, col_factor)
    out = np.full((cavity_h + 2, cavity_w + 2), BLACK, dtype=int)
    out[0, :] = frame_color
    out[-1, :] = frame_color
    out[:, 0] = frame_color
    out[:, -1] = frame_color
    out[1:-1, 1:-1] = filled
    return out


def _semantic_checks(
    grid: np.ndarray,
    frame_box: BBox,
    boards: list[dict[str, object]],
    scenario: dict[str, bool],
) -> bool:
    cavity_h = frame_box[2] - frame_box[0] - 1
    cavity_w = frame_box[3] - frame_box[1] - 1

    target = next(board for board in boards if board["role"] == "target")
    candidates = [
        board
        for board in boards
        if cavity_h % int(np.asarray(board["skeleton"]).shape[0]) == 0
        and cavity_w % int(np.asarray(board["skeleton"]).shape[1]) == 0
    ]
    if len(candidates) != 1 or candidates[0]["role"] != "target":
        return False

    has_row_only = any(
        cavity_h % int(np.asarray(board["skeleton"]).shape[0]) == 0
        and cavity_w % int(np.asarray(board["skeleton"]).shape[1]) != 0
        for board in boards
        if board["role"] != "target"
    )
    has_col_only = any(
        cavity_h % int(np.asarray(board["skeleton"]).shape[0]) != 0
        and cavity_w % int(np.asarray(board["skeleton"]).shape[1]) == 0
        for board in boards
        if board["role"] != "target"
    )
    if not has_row_only or not has_col_only:
        return False

    size_match_lures = [
        board
        for board in boards
        if board["role"] != "target" and np.asarray(board["board"]).shape == (cavity_h, cavity_w)
    ]
    if len(size_match_lures) != 1:
        return False

    raw_bbox_fit = [
        board
        for board in boards
        if cavity_h % np.asarray(board["board"]).shape[0] == 0 and cavity_w % np.asarray(board["board"]).shape[1] == 0
    ]
    if not any(board["role"] != "target" for board in raw_bbox_fit):
        return False

    target_box_matches = np.asarray(target["board"]).shape == (cavity_h, cavity_w)
    if target_box_matches != bool(scenario["target_exact_size"]):
        return False

    expected = _expected_output(RED, cavity_h, cavity_w, np.asarray(target["skeleton"], dtype=int))
    try:
        actual = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False

    parsed_color, parsed_box, parsed_boards = _parse_boards(grid)
    if parsed_color != RED or parsed_box != frame_box or len(parsed_boards) != len(boards):
        return False
    parsed_candidates = [
        np.asarray(board["skeleton"], dtype=int)
        for board in parsed_boards
        if cavity_h % np.asarray(board["skeleton"]).shape[0] == 0
        and cavity_w % np.asarray(board["skeleton"]).shape[1] == 0
    ]
    if len(parsed_candidates) != 1:
        return False
    return np.array_equal(parsed_candidates[0], np.asarray(target["skeleton"], dtype=int))


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    scenario = _next_scenario()
    cavity_h, cavity_w, target_rows, target_cols = _choose_frame_and_target(rng, scenario["need_unequal_scale"])

    row_divisors = _divisors(cavity_h)
    col_divisors = _divisors(cavity_w)
    row_nondivisors = _nondivisors(cavity_h)
    col_nondivisors = _nondivisors(cavity_w)

    boards: list[dict[str, object]] = []

    target = _make_board(
        target_rows,
        target_cols,
        cavity_h,
        cavity_w,
        rng,
        exact_box=scenario["target_exact_size"],
        forbid_exact_box=not scenario["target_exact_size"],
    )
    target["role"] = "target"
    boards.append(target)

    row_only = _make_board(
        int(rng.choice(row_divisors)),
        int(rng.choice(col_nondivisors)),
        cavity_h,
        cavity_w,
        rng,
        forbid_exact_box=True,
    )
    row_only["role"] = "row_only"
    boards.append(row_only)

    col_only = _make_board(
        int(rng.choice(row_nondivisors)),
        int(rng.choice(col_divisors)),
        cavity_h,
        cavity_w,
        rng,
        forbid_exact_box=True,
    )
    col_only["role"] = "col_only"
    boards.append(col_only)

    exact_size = _make_board(
        int(rng.choice(row_nondivisors)),
        int(rng.choice(col_nondivisors)),
        cavity_h,
        cavity_w,
        rng,
        exact_box=True,
    )
    exact_size["role"] = "exact_size"
    boards.append(exact_size)

    if scenario["extra_lure"]:
        for _ in range(100):
            rows = int(rng.choice(row_nondivisors))
            cols = int(rng.choice(col_nondivisors))
            if (rows, cols) != (int(np.asarray(exact_size["skeleton"]).shape[0]), int(np.asarray(exact_size["skeleton"]).shape[1])):
                extra = _make_board(rows, cols, cavity_h, cavity_w, rng, forbid_exact_box=True)
                extra["role"] = "both_fail"
                boards.append(extra)
                break

    frame_patch = _make_frame_patch(cavity_h, cavity_w)
    patches = [frame_patch] + [np.asarray(board["board"], dtype=int) for board in boards]
    max_h = max(patch.shape[0] for patch in patches)
    max_w = max(patch.shape[1] for patch in patches)
    for _ in range(120):
        canvas_h = int(rng.integers(max(max_h + 2, 26), 31))
        canvas_w = int(rng.integers(max(max_w + 2, 26), 31))
        placed = _place_rectangles(patches, rng, canvas_h, canvas_w, padding=1)
        if placed is None:
            continue
        grid, positions = placed
        frame_r0, frame_c0 = positions[0]
        frame_box = (frame_r0, frame_c0, frame_r0 + frame_patch.shape[0] - 1, frame_c0 + frame_patch.shape[1] - 1)
        if not _semantic_checks(grid, frame_box, boards, scenario):
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to produce a valid puzzle instance")
