from __future__ import annotations

from collections import Counter
import random

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """
    Denoise-compress a sequence of enlarged noisy panels and predict the next animation frame.

    Input structure:
    - The grid is split by same-color full-row or full-column divider lines into at least three equal panels.
    - Each panel is a scaled-up p x q image made of equal rectangular blocks; every block has one strict majority color plus a few noisy pixels.
    - After block-majority compression, all panels share the same static layout except for one moving single-color shape.

    Rule:
    - Split the input into panels using the divider lines.
    - For each panel, recover the p x q image by taking the majority color in every equal-size block.
    - Treat the compressed panels as time-ordered frames, extract the static background, infer the dynamic shape's constant translation step, and move it one more step past the last frame.

    Key insight:
    - The answer is not the cleaned last panel: denoising reveals a frame sequence, and the required output is the next unseen frame obtained by extrapolating the motion.
    """
    return _solve_grid(np.asarray(grid, dtype=int))


def _solve_grid(grid: np.ndarray) -> np.ndarray:
    panels = _split_panels(grid)
    candidates = []
    panel_h, panel_w = panels[0].shape
    for block_h in _divisors(panel_h):
        for block_w in _divisors(panel_w):
            if block_h < 2 or block_w < 2:
                continue
            coarse_h = panel_h // block_h
            coarse_w = panel_w // block_w
            if coarse_h < 2 or coarse_w < 2:
                continue
            try:
                frames = [_compress_panel(panel, block_h, block_w) for panel in panels]
                predicted = _predict_next_frame(frames)
            except AssertionError:
                continue
            candidates.append(
                {
                    "block_h": block_h,
                    "block_w": block_w,
                    "frames": frames,
                    "output": predicted,
                }
            )
    unique_outputs: list[np.ndarray] = []
    for candidate in candidates:
        out = candidate["output"]
        if not any(np.array_equal(out, seen) for seen in unique_outputs):
            unique_outputs.append(out)
    assert len(unique_outputs) == 1, "expected a unique valid coarse-grid interpretation"
    return unique_outputs[0]


def _split_panels(grid: np.ndarray) -> list[np.ndarray]:
    candidates = []
    for axis in (0, 1):
        axis_candidates = _divider_axis_candidates(grid, axis)
        for candidate in axis_candidates:
            panels = _extract_panels(grid, axis, candidate["runs"])
            if len(panels) >= 3 and all(panel.shape == panels[0].shape for panel in panels[1:]):
                candidates.append(panels)
    assert len(candidates) == 1, "expected exactly one divider axis"
    return candidates[0]


def _divider_axis_candidates(grid: np.ndarray, axis: int) -> list[dict]:
    n = grid.shape[axis]
    runs: list[tuple[int, int, int]] = []
    i = 0
    while i < n:
        line = grid[i, :] if axis == 0 else grid[:, i]
        if np.all(line == line[0]):
            color = int(line[0])
            j = i
            while j + 1 < n:
                next_line = grid[j + 1, :] if axis == 0 else grid[:, j + 1]
                if np.all(next_line == color):
                    j += 1
                else:
                    break
            runs.append((i, j, color))
            i = j + 1
        else:
            i += 1

    by_color: dict[int, list[tuple[int, int]]] = {}
    for start, end, color in runs:
        by_color.setdefault(color, []).append((start, end))

    candidates = []
    for color, color_runs in by_color.items():
        if len(color_runs) < 2:
            continue
        thicknesses = {end - start + 1 for start, end in color_runs}
        if len(thicknesses) != 1:
            continue
        panel_sizes = [color_runs[0][0]]
        for i in range(1, len(color_runs)):
            panel_sizes.append(color_runs[i][0] - color_runs[i - 1][1] - 1)
        panel_sizes.append(n - color_runs[-1][1] - 1)
        if min(panel_sizes) <= 0:
            continue
        if len(set(panel_sizes)) != 1:
            continue
        candidates.append({"axis": axis, "color": color, "runs": color_runs, "panel_size": panel_sizes[0]})
    return candidates


def _extract_panels(grid: np.ndarray, axis: int, runs: list[tuple[int, int]]) -> list[np.ndarray]:
    panels = []
    start = 0
    for run_start, run_end in runs:
        if axis == 0:
            panels.append(grid[start:run_start, :].copy())
        else:
            panels.append(grid[:, start:run_start].copy())
        start = run_end + 1
    if axis == 0:
        panels.append(grid[start:, :].copy())
    else:
        panels.append(grid[:, start:].copy())
    return panels


def _compress_panel(panel: np.ndarray, block_h: int, block_w: int) -> np.ndarray:
    H, W = panel.shape
    assert H % block_h == 0 and W % block_w == 0
    out = np.empty((H // block_h, W // block_w), dtype=int)
    for br in range(out.shape[0]):
        for bc in range(out.shape[1]):
            block = panel[br * block_h : (br + 1) * block_h, bc * block_w : (bc + 1) * block_w]
            values, counts = np.unique(block, return_counts=True)
            order = np.argsort(-counts)
            if len(order) > 1:
                assert int(counts[order[0]]) > int(counts[order[1]]), "block majority must be unique"
            out[br, bc] = int(values[order[0]])
    return out


def _predict_next_frame(frames: list[np.ndarray]) -> np.ndarray:
    assert len(frames) >= 3
    stack = np.stack(frames, axis=0)
    n_frames, H, W = stack.shape

    static = np.empty((H, W), dtype=int)
    for r in range(H):
        for c in range(W):
            values, counts = np.unique(stack[:, r, c], return_counts=True)
            order = np.argsort(-counts)
            if len(order) > 1:
                assert int(counts[order[0]]) > int(counts[order[1]]), "static color must be uniquely identifiable"
            assert int(counts[order[0]]) >= n_frames - 1, "dynamic object may cover a cell at most once"
            static[r, c] = int(values[order[0]])

    dynamic_masks = []
    dynamic_color = None
    for frame in frames:
        diff = frame != static
        assert diff.any(), "every frame must contain the moving object"
        colors = {int(x) for x in frame[diff]}
        assert len(colors) == 1, "all changed cells in a frame must belong to one dynamic color"
        color = next(iter(colors))
        if dynamic_color is None:
            dynamic_color = color
        else:
            assert color == dynamic_color, "dynamic color must stay constant"
        dynamic_masks.append(diff)
    assert dynamic_color is not None
    assert not np.any(static == dynamic_color), "dynamic color must be absent from the static layout"

    coords0 = np.argwhere(dynamic_masks[0])
    assert len(coords0) > 0
    shifts = []
    for mask in dynamic_masks:
        coords = np.argwhere(mask)
        assert len(coords) == len(coords0), "shape area must stay constant"
        delta = coords[0] - coords0[0]
        assert np.array_equal(coords, coords0 + delta), "shape must translate rigidly"
        shifts.append((int(delta[0]), int(delta[1])))

    step = (shifts[1][0] - shifts[0][0], shifts[1][1] - shifts[0][1])
    assert step != (0, 0), "motion step must be non-zero"
    for prev, curr in zip(shifts, shifts[1:]):
        assert (curr[0] - prev[0], curr[1] - prev[1]) == step, "motion step must be constant"

    next_coords = np.argwhere(dynamic_masks[-1]) + np.array(step, dtype=int)
    assert next_coords.size > 0
    assert int(next_coords[:, 0].min()) >= 0 and int(next_coords[:, 0].max()) < H
    assert int(next_coords[:, 1].min()) >= 0 and int(next_coords[:, 1].max()) < W

    out = static.copy()
    out[next_coords[:, 0], next_coords[:, 1]] = dynamic_color
    assert not any(np.array_equal(out, frame) for frame in frames), "next frame must be outside the observed sequence"
    return out


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def _scale_grid(grid: np.ndarray, block_h: int, block_w: int) -> np.ndarray:
    return np.repeat(np.repeat(grid, block_h, axis=0), block_w, axis=1)


def _random_static_grid(height: int, width: int, palette: list[int]) -> np.ndarray | None:
    for _ in range(200):
        grid = np.array([random.choice(palette) for _ in range(height * width)], dtype=int).reshape(height, width)
        if len(set(map(int, grid.flat))) < min(3, len(palette)):
            continue
        if any(len(set(map(int, grid[r, :]))) < 2 for r in range(height)):
            continue
        if any(len(set(map(int, grid[:, c]))) < 2 for c in range(width)):
            continue
        return grid
    return None


def _make_shape(max_h: int, max_w: int) -> np.ndarray | None:
    for _ in range(200):
        h = random.randint(2, max_h)
        w = random.randint(2, max_w)
        area = random.randint(2, min(5, h * w - 1))
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < area:
            r, c = random.choice(tuple(cells))
            neighbors = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
            random.shuffle(neighbors)
            for nr, nc in neighbors:
                if 0 <= nr < h and 0 <= nc < w:
                    cells.add((nr, nc))
                    break
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if mask.all():
            continue
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        mask = mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
        if mask.shape[0] == 1 and mask.shape[1] == 1:
            continue
        return mask
    return None


def _motion_vectors() -> list[tuple[int, int]]:
    vectors = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
        (-2, -1),
        (-2, 1),
        (-1, -2),
        (-1, 2),
        (1, -2),
        (1, 2),
        (2, -1),
        (2, 1),
    ]
    random.shuffle(vectors)
    return vectors


def _choose_motion_positions(shape: np.ndarray, frame_h: int, frame_w: int, n_input_frames: int) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    sh, sw = shape.shape
    shape_coords = np.argwhere(shape)
    for step_r, step_c in _motion_vectors():
        min_r = max(0, -step_r * n_input_frames)
        max_r = frame_h - sh - max(0, step_r * n_input_frames)
        min_c = max(0, -step_c * n_input_frames)
        max_c = frame_w - sw - max(0, step_c * n_input_frames)
        if min_r > max_r or min_c > max_c:
            continue
        starts = [(r, c) for r in range(min_r, max_r + 1) for c in range(min_c, max_c + 1)]
        random.shuffle(starts)
        for start_r, start_c in starts[:40]:
            positions = [(start_r + t * step_r, start_c + t * step_c) for t in range(n_input_frames + 1)]
            occupied: set[tuple[int, int]] = set()
            ok = True
            for top, left in positions:
                translated = {(int(top + r), int(left + c)) for r, c in shape_coords}
                if occupied & translated:
                    ok = False
                    break
                occupied |= translated
            if ok:
                return positions, (step_r, step_c)
    return None


def _paint_shape(frame: np.ndarray, shape: np.ndarray, top: int, left: int, color: int) -> None:
    coords = np.argwhere(shape)
    for r, c in coords:
        frame[int(top + r), int(left + c)] = int(color)


def _render_noisy_panel(frame: np.ndarray, block_h: int, block_w: int, palette: list[int]) -> np.ndarray:
    panel = _scale_grid(frame, block_h, block_w)
    area = block_h * block_w
    max_noise = max(1, (area - 1) // 2)
    for r in range(frame.shape[0]):
        for c in range(frame.shape[1]):
            color = int(frame[r, c])
            block = panel[r * block_h : (r + 1) * block_h, c * block_w : (c + 1) * block_w]
            noise_count = random.randint(1, max_noise)
            idxs = random.sample(range(area), noise_count)
            choices = [x for x in palette if x != color]
            for idx in idxs:
                rr = idx // block_w
                cc = idx % block_w
                block[rr, cc] = random.choice(choices)
    return panel


def _assemble_grid(panels: list[np.ndarray], divider_color: int, horizontal: bool) -> np.ndarray:
    if horizontal:
        H = panels[0].shape[0]
        W = sum(panel.shape[1] for panel in panels) + len(panels) - 1
        grid = np.full((H, W), divider_color, dtype=int)
        c0 = 0
        for i, panel in enumerate(panels):
            grid[:, c0 : c0 + panel.shape[1]] = panel
            c0 += panel.shape[1]
            if i != len(panels) - 1:
                c0 += 1
        return grid
    H = sum(panel.shape[0] for panel in panels) + len(panels) - 1
    W = panels[0].shape[1]
    grid = np.full((H, W), divider_color, dtype=int)
    r0 = 0
    for i, panel in enumerate(panels):
        grid[r0 : r0 + panel.shape[0], :] = panel
        r0 += panel.shape[0]
        if i != len(panels) - 1:
            r0 += 1
    return grid


def _semantic_ok(frames: list[np.ndarray], expected: np.ndarray) -> bool:
    try:
        predicted = _predict_next_frame(frames)
    except AssertionError:
        return False
    if not np.array_equal(predicted, expected):
        return False
    if any(np.array_equal(expected, frame) for frame in frames):
        return False
    return True


def _no_spurious_full_lines(grid: np.ndarray, horizontal: bool) -> bool:
    if horizontal:
        return all(len(set(map(int, grid[r, :]))) > 1 for r in range(grid.shape[0]))
    return all(len(set(map(int, grid[:, c]))) > 1 for c in range(grid.shape[1]))


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    horizontal = random.choice([False, True])
    n_input_frames = random.choice([3, 4])
    divider_color = random.choice(range(10))

    if horizontal:
        coarse_h = random.randint(4, 6)
        coarse_w = random.randint(4, min(6, (29 - (n_input_frames - 1)) // n_input_frames))
    else:
        coarse_h = random.randint(4, min(6, (29 - (n_input_frames - 1)) // n_input_frames))
        coarse_w = random.randint(4, 6)
    block_h = random.choice([2, 3])
    block_w = random.choice([2, 3])

    panel_h = coarse_h * block_h
    panel_w = coarse_w * block_w
    total_h = panel_h if horizontal else n_input_frames * panel_h + (n_input_frames - 1)
    total_w = n_input_frames * panel_w + (n_input_frames - 1) if horizontal else panel_w
    if total_h > 30 or total_w > 30:
        return None

    palette = list(range(10))
    palette.remove(divider_color)
    random.shuffle(palette)
    dynamic_color = palette[0]
    static_palette = palette[1 : random.randint(4, min(6, len(palette)))]
    static = _random_static_grid(coarse_h, coarse_w, static_palette)
    if static is None:
        return None

    shape = _make_shape(min(3, coarse_h - 1), min(3, coarse_w - 1))
    if shape is None:
        return None
    motion = _choose_motion_positions(shape, coarse_h, coarse_w, n_input_frames)
    if motion is None:
        return None
    positions, _step = motion

    frames = []
    for top, left in positions[:-1]:
        frame = static.copy()
        _paint_shape(frame, shape, top, left, dynamic_color)
        frames.append(frame)
    expected = static.copy()
    _paint_shape(expected, shape, positions[-1][0], positions[-1][1], dynamic_color)

    if not _semantic_ok(frames, expected):
        return None

    noisy_panels = [_render_noisy_panel(frame, block_h, block_w, static_palette + [dynamic_color]) for frame in frames]
    if any(not _no_spurious_full_lines(panel, horizontal) for panel in noisy_panels):
        return None
    grid = _assemble_grid(noisy_panels, divider_color, horizontal)
    if max(grid.shape) > 30:
        return None

    try:
        actual = _solve_grid(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    candidates = []
    for block_h2 in _divisors(panel_h):
        for block_w2 in _divisors(panel_w):
            if block_h2 < 2 or block_w2 < 2:
                continue
            if panel_h % block_h2 or panel_w % block_w2:
                continue
            try:
                test_frames = [_compress_panel(panel, block_h2, block_w2) for panel in noisy_panels]
                test_out = _predict_next_frame(test_frames)
            except AssertionError:
                continue
            candidates.append(test_out)
    dedup = []
    for candidate in candidates:
        if not any(np.array_equal(candidate, seen) for seen in dedup):
            dedup.append(candidate)
    if len(dedup) != 1 or not np.array_equal(dedup[0], expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(500):
        built = _build_case()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid noisy-panel sequence")
