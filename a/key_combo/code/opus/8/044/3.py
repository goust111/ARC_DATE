from __future__ import annotations

import random

import numpy as np


BG = 0
MARKER_COLORS = (1, 2, 3, 4)
FRAME_COLOR = 5
ORANGE = 7
MAX_OUTPUT_WIDTH = 30
MAX_ATTEMPTS = 600
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _tight_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    return len(_components_mask(mask)) == 1


def _find_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    frames = []
    for coords in _components_mask(g == FRAME_COLOR):
        r0, c0, r1, c1 = _bbox_coords(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 4 or w < 4:
            continue
        frame_mask = np.zeros((h, w), dtype=bool)
        frame_mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
        expected = np.zeros((h, w), dtype=bool)
        expected[0, :] = True
        expected[-1, :] = True
        expected[:, 0] = True
        expected[:, -1] = True
        if not np.array_equal(frame_mask, expected):
            continue
        if np.any(g[r0 + 1 : r1, c0 + 1 : c1] == FRAME_COLOR):
            continue
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "interior": g[r0 + 1 : r1, c0 + 1 : c1].copy(),
            }
        )
    frames.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return frames


def _corner_values(interior: np.ndarray) -> list[int]:
    h, w = interior.shape
    return [
        int(interior[0, 0]),
        int(interior[0, w - 1]),
        int(interior[h - 1, 0]),
        int(interior[h - 1, w - 1]),
    ]


def _scale(arr: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(arr, np.ones((factor, factor), dtype=arr.dtype))


def _erase_scaled_corners(arr: np.ndarray, factor: int) -> np.ndarray:
    out = arr.copy()
    k = int(factor)
    out[:k, :k] = BG
    out[:k, -k:] = BG
    out[-k:, :k] = BG
    out[-k:, -k:] = BG
    return out


def _result_from_interior(interior: np.ndarray) -> np.ndarray:
    k = len(set(_corner_values(interior)))
    return _erase_scaled_corners(_scale(interior, k), k)


def _pack_results(results: list[np.ndarray], max_width: int = MAX_OUTPUT_WIDTH) -> np.ndarray:
    assert results
    rows: list[list[np.ndarray]] = []
    current: list[np.ndarray] = []
    used_width = 0
    for res in results:
        w = int(res.shape[1])
        if not current:
            current = [res]
            used_width = w
            continue
        candidate = used_width + 1 + w
        if candidate <= max_width:
            current.append(res)
            used_width = candidate
        else:
            rows.append(current)
            current = [res]
            used_width = w
    if current:
        rows.append(current)

    row_heights = [max(int(item.shape[0]) for item in row) for row in rows]
    row_widths = [sum(int(item.shape[1]) for item in row) + len(row) - 1 for row in rows]
    out = np.full((sum(row_heights), max(row_widths)), BG, dtype=int)

    top = 0
    for row, row_h in zip(rows, row_heights):
        left = 0
        for item in row:
            h, w = map(int, item.shape)
            out[top : top + h, left : left + w] = item
            left += w + 1
        top += row_h
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Crop each framed interior, enlarge it by the number of distinct corner-marker colors, then erase the enlarged corner markers.

    Input structure:
    - A black background contains 2 or 3 separated one-pixel-thick grey rectangular frames.
    - Each frame encloses a small interior whose four corner cells are blue/red/green/yellow marker pixels.
    - The rest of the interior contains black plus a small orange 4-connected pattern.

    Rule:
    - Detect every grey hollow rectangle and crop only its interior area.
    - For each interior, count how many distinct colors appear in its four corner marker cells; call that count K.
    - Scale the whole interior by K in both axes with nearest-neighbor pixel replication, then erase the four KxK corner blocks back to black.
    - Order the resulting subgrids by the frames' top-to-bottom, then left-to-right positions, and pack them left-to-right with a one-column black separator; when adding the next subgrid would exceed 30 columns, wrap to the next output row.

    Key insight:
    - The corner markers are temporary encoders: their color diversity selects the scale factor, but after scaling the enlarged corner blocks must be removed, leaving only the magnified non-marker content.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames = _find_frames(g)
    assert 2 <= len(frames) <= 3

    results = []
    for frame in frames:
        interior = frame["interior"]
        assert interior.shape[0] >= 2 and interior.shape[1] >= 2
        corner_colors = _corner_values(interior)
        assert all(color in MARKER_COLORS for color in corner_colors)
        results.append(_result_from_interior(interior))
    return _pack_results(results)


def _sample_corner_assignment(k: int) -> list[int]:
    chosen = random.sample(MARKER_COLORS, k)
    colors = chosen + [random.choice(chosen) for _ in range(4 - k)]
    random.shuffle(colors)
    assert len(set(colors)) == k
    return colors


def _orange_shape_key(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], bytes]:
    coords = np.array(sorted(cells), dtype=int)
    mask = _tight_mask(coords)
    return mask.shape, mask.tobytes()


def _sample_orange_cells(h: int, w: int, used_keys: set[tuple[tuple[int, int], bytes]]) -> set[tuple[int, int]] | None:
    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    available = [(r, c) for r in range(h) for c in range(w) if (r, c) not in corners]
    if len(available) < 3:
        return None

    for _ in range(120):
        target = random.randint(3, min(6, len(available)))
        cells = {random.choice(available)}
        while len(cells) < target:
            frontier = []
            for r, c in sorted(cells):
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) in available and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != target:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if not _is_connected(mask):
            continue
        key = _orange_shape_key(cells)
        if key in used_keys:
            continue
        return cells
    return None


def _make_interior(h: int, w: int, k: int, used_shape_keys: set[tuple[tuple[int, int], bytes]]) -> tuple[np.ndarray, tuple[tuple[int, int], bytes]] | None:
    orange_cells = _sample_orange_cells(h, w, used_shape_keys)
    if orange_cells is None:
        return None

    interior = np.zeros((h, w), dtype=int)
    tl, tr, bl, br = _sample_corner_assignment(k)
    interior[0, 0] = tl
    interior[0, w - 1] = tr
    interior[h - 1, 0] = bl
    interior[h - 1, w - 1] = br
    for r, c in orange_cells:
        interior[r, c] = ORANGE
    return interior, _orange_shape_key(orange_cells)


def _planned_result_from_frame(frame: dict) -> np.ndarray:
    return _erase_scaled_corners(_scale(frame["interior"], frame["k"]), frame["k"])


def _plan_frames() -> list[dict] | None:
    frame_count = random.choice([2, 3])
    all_k_sets = {
        2: [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)],
        3: [(1, 2, 4), (1, 3, 4), (1, 2, 3), (2, 3, 4)],
    }
    ks = list(random.choice(all_k_sets[frame_count]))
    random.shuffle(ks)

    frames = []
    used_shape_keys: set[tuple[tuple[int, int], bytes]] = set()
    for k in ks:
        for _ in range(120):
            h = random.randint(3, 4)
            w = random.randint(3, 4)
            built = _make_interior(h, w, k, used_shape_keys)
            if built is None:
                continue
            interior, shape_key = built
            frame = {
                "k": k,
                "frame_h": h + 2,
                "frame_w": w + 2,
                "interior": interior,
                "shape_key": shape_key,
            }
            frames.append(frame)
            used_shape_keys.add(shape_key)
            break
        else:
            return None

    planned_results = [_planned_result_from_frame(frame) for frame in frames]
    packed = _pack_results(planned_results)
    if max(packed.shape) > 30:
        return None
    if len({frame["k"] for frame in frames}) != len(frames):
        return None
    if len({(res.shape, res.tobytes()) for res in planned_results}) != len(planned_results):
        return None
    return frames


def _place_frames(frames: list[dict]) -> np.ndarray | None:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    for frame in sorted(frames, key=lambda item: item["frame_h"] * item["frame_w"], reverse=True):
        fh = int(frame["frame_h"])
        fw = int(frame["frame_w"])
        positions = []
        for r0 in range(1, h - fh):
            for c0 in range(1, w - fw):
                r1 = r0 + fh - 1
                c1 = c0 + fw - 1
                rr0 = max(0, r0 - 1)
                cc0 = max(0, c0 - 1)
                rr1 = min(h, r1 + 2)
                cc1 = min(w, c1 + 2)
                if blocked[rr0:rr1, cc0:cc1].any():
                    continue
                positions.append((r0, c0))
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        r1 = r0 + fh - 1
        c1 = c0 + fw - 1
        grid[r0, c0 : c1 + 1] = FRAME_COLOR
        grid[r1, c0 : c1 + 1] = FRAME_COLOR
        grid[r0 : r1 + 1, c0] = FRAME_COLOR
        grid[r0 : r1 + 1, c1] = FRAME_COLOR
        grid[r0 + 1 : r1, c0 + 1 : c1] = frame["interior"]
        frame["bbox"] = (r0, c0, r1, c1)
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(h, r1 + 2)
        cc1 = min(w, c1 + 2)
        blocked[rr0:rr1, cc0:cc1] = True

    return grid


def _independent_expected(frames: list[dict]) -> np.ndarray:
    ordered = sorted(frames, key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return _pack_results([_planned_result_from_frame(frame) for frame in ordered])


def _semantic_checks(grid: np.ndarray, frames: list[dict]) -> bool:
    solver_out = main(grid)
    expected = _independent_expected(frames)
    if not np.array_equal(solver_out, expected):
        return False

    detected = _find_frames(grid)
    if len(detected) != len(frames):
        return False

    packed_without_erasure = _pack_results([_scale(frame["interior"], frame["k"]) for frame in sorted(frames, key=lambda item: (item["bbox"][0], item["bbox"][1]))])
    if np.array_equal(expected, packed_without_erasure):
        return False

    fixed_two = _pack_results(
        [_erase_scaled_corners(_scale(frame["interior"], 2), 2) for frame in sorted(frames, key=lambda item: (item["bbox"][0], item["bbox"][1]))]
    )
    if np.array_equal(expected, fixed_two):
        return False

    for frame in frames:
        interior = frame["interior"]
        orange_mask = interior == ORANGE
        if not (3 <= int(orange_mask.sum()) <= 6):
            return False
        if not _is_connected(orange_mask):
            return False
        if any(int(v) not in MARKER_COLORS for v in _corner_values(interior)):
            return False
        if len(set(_corner_values(interior))) != frame["k"]:
            return False
        result = _planned_result_from_frame(frame)
        k = frame["k"]
        if np.any(result[:k, :k] != BG):
            return False
        if np.any(result[:k, -k:] != BG):
            return False
        if np.any(result[-k:, :k] != BG):
            return False
        if np.any(result[-k:, -k:] != BG):
            return False
        sample_orange = tuple(np.argwhere(orange_mask)[0])
        rr = slice(sample_orange[0] * k, (sample_orange[0] + 1) * k)
        cc = slice(sample_orange[1] * k, (sample_orange[1] + 1) * k)
        if not np.all(result[rr, cc] == ORANGE):
            return False

    if len({frame["shape_key"] for frame in frames}) != len(frames):
        return False
    return max(solver_out.shape) <= 30


def _gen_attempt() -> np.ndarray | None:
    frames = _plan_frames()
    if frames is None:
        return None
    grid = _place_frames(frames)
    if grid is None:
        return None
    if not _semantic_checks(grid, frames):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
