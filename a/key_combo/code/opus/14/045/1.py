from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
TEAL = 8

COLOR_SEQ = (RED, BLUE, GREEN, YELLOW)

FRAME_MIN_OUTER = 5
FRAME_MAX_OUTER = 8
SHARED_SIZES = ((7, 7), (7, 8), (8, 7), (8, 8))
PARTITIONS = ((1, 2), (1, 3), (1, 4), (2, 3))
GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _label_mask(mask: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    cid = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            cid += 1
            q: deque[tuple[int, int]] = deque([(r, c)])
            labels[r, c] = cid
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = cid
                        q.append((nr, nc))
    return labels, cid


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY


def _ring_colors(k: int) -> list[int]:
    return [COLOR_SEQ[i % len(COLOR_SEQ)] for i in range(k)]


def _detect_frames(grid: np.ndarray) -> list[dict]:
    grey_mask = grid == GREY
    labels, n = _label_mask(grey_mask)
    frames = []
    for cid in range(1, n + 1):
        mask = labels == cid
        r0, c0, r1, c1 = _bbox_from_mask(mask)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= FRAME_MIN_OUTER and w >= FRAME_MIN_OUTER
        expected = np.zeros((h, w), dtype=bool)
        expected[0, :] = True
        expected[-1, :] = True
        expected[:, 0] = True
        expected[:, -1] = True
        assert np.array_equal(mask[r0 : r1 + 1, c0 : c1 + 1], expected)
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "sample": (r0, c0),
                "shape": (h, w),
                "interior_shape": (h - 2, w - 2),
            }
        )
    assert 3 <= len(frames) <= 5
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow grey frame with concentric color rings whose period is set by its bridge-connected frame count.

    Input structure:
    - A black background contains 3 to 5 one-cell-thick grey rectangular frames with black interiors.
    - Some pairs of frames are linked by one-cell-wide teal orthogonal bridge segments that touch frame exteriors without replacing grey border pixels.
    - Frames are separated, non-nested, and the bridge network partitions them into 4-connected non-black components of different frame counts.

    Rule:
    - Build 4-connected components over every non-black cell, treating grey frame pixels and teal bridge pixels as the same foreground.
    - Detect each complete grey rectangular frame, and for its component count how many such frames it contains; call that number K.
    - Keep every grey border pixel and teal bridge pixel unchanged, then fill that frame's interior by depth from the border: layer 0 uses red, and deeper layers cycle through the first K colors of [red, blue, green, yellow], repeating with period K.

    Key insight:
    - The teal bridges are not recolored themselves; they only matter because connectivity reveals the component's frame count K, and K alone determines the frame's concentric ring palette length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames = _detect_frames(g)
    labels, _ = _label_mask(g != BLACK)
    frame_counts: defaultdict[int, int] = defaultdict(int)
    frame_component: list[int] = []
    for frame in frames:
        comp_id = int(labels[frame["sample"]])
        assert comp_id > 0
        frame_counts[comp_id] += 1
        frame_component.append(comp_id)

    out = g.copy()
    for frame, comp_id in zip(frames, frame_component):
        k = frame_counts[comp_id]
        colors = _ring_colors(k)
        r0, c0, r1, c1 = frame["bbox"]
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                depth = min(r - r0 - 1, r1 - 1 - r, c - c0 - 1, c1 - 1 - c)
                out[r, c] = colors[depth % k]
    return out


def _random_frame_size() -> tuple[int, int]:
    return random.randint(FRAME_MIN_OUTER, FRAME_MAX_OUTER), random.randint(FRAME_MIN_OUTER, FRAME_MAX_OUTER)


def _build_component(k: int, orientation: str, shared_size: tuple[int, int], use_shared: bool) -> tuple[np.ndarray, list[dict]]:
    if k == 1:
        h, w = shared_size if use_shared else _random_frame_size()
        patch = np.zeros((h, w), dtype=int)
        _draw_frame(patch, 0, 0, h, w)
        return patch, [{"bbox": (0, 0, h - 1, w - 1), "k": k}]

    shared_index = random.randrange(k) if use_shared else None
    for _ in range(120):
        if orientation == "h":
            common_h = shared_size[0] if use_shared else random.randint(FRAME_MIN_OUTER, FRAME_MAX_OUTER)
            widths = []
            for i in range(k):
                if shared_index == i:
                    widths.append(shared_size[1])
                else:
                    widths.append(random.randint(FRAME_MIN_OUTER, FRAME_MAX_OUTER))
            bridges = [random.randint(1, 3) for _ in range(k - 1)]
            total_w = sum(widths) + sum(bridges)
            if total_w > 26:
                continue
            patch = np.zeros((common_h, total_w), dtype=int)
            gate_r = random.randint(1, common_h - 2)
            frames = []
            cur_c = 0
            prev_end = None
            for i, width in enumerate(widths):
                _draw_frame(patch, 0, cur_c, common_h, width)
                frames.append({"bbox": (0, cur_c, common_h - 1, cur_c + width - 1), "k": k})
                if prev_end is not None:
                    patch[gate_r, prev_end + 1 : cur_c] = TEAL
                prev_end = cur_c + width - 1
                if i < k - 1:
                    cur_c += width + bridges[i]
            return patch, frames

        common_w = shared_size[1] if use_shared else random.randint(FRAME_MIN_OUTER, FRAME_MAX_OUTER)
        heights = []
        for i in range(k):
            if shared_index == i:
                heights.append(shared_size[0])
            else:
                heights.append(random.randint(FRAME_MIN_OUTER, FRAME_MAX_OUTER))
        bridges = [random.randint(1, 3) for _ in range(k - 1)]
        total_h = sum(heights) + sum(bridges)
        if total_h > 26:
            continue
        patch = np.zeros((total_h, common_w), dtype=int)
        gate_c = random.randint(1, common_w - 2)
        frames = []
        cur_r = 0
        prev_end = None
        for i, height in enumerate(heights):
            _draw_frame(patch, cur_r, 0, height, common_w)
            frames.append({"bbox": (cur_r, 0, cur_r + height - 1, common_w - 1), "k": k})
            if prev_end is not None:
                patch[prev_end + 1 : cur_r, gate_c] = TEAL
            prev_end = cur_r + height - 1
            if i < k - 1:
                cur_r += height + bridges[i]
        return patch, frames
    raise RuntimeError("failed to build component")


def _feasible_layouts(shape_a: tuple[int, int], shape_b: tuple[int, int]) -> list[tuple[str, tuple[int, int]]]:
    ha, wa = shape_a
    hb, wb = shape_b
    layouts = []
    for sep in range(2, 5):
        if max(ha, hb) <= 28 and wa + wb + sep <= 28:
            layouts.append(("side", (sep, 0)))
        if max(wa, wb) <= 28 and ha + hb + sep <= 28:
            layouts.append(("stack", (sep, 0)))
    return layouts


def _place_two_patches(
    patch_a: np.ndarray,
    frames_a: list[dict],
    patch_b: np.ndarray,
    frames_b: list[dict],
) -> tuple[np.ndarray, list[dict]]:
    options = _feasible_layouts(patch_a.shape, patch_b.shape)
    if not options:
        raise RuntimeError("no feasible layout")
    mode, (sep, _dummy) = random.choice(options)

    if random.random() < 0.5:
        patch_a, patch_b = patch_b, patch_a
        frames_a, frames_b = frames_b, frames_a

    ha, wa = patch_a.shape
    hb, wb = patch_b.shape

    if mode == "side":
        extra_h = random.randint(0, 4)
        extra_w = random.randint(0, 3)
        top = random.randint(0, 2)
        left = random.randint(0, 2)
        h = max(ha, hb) + top + extra_h
        w = wa + wb + sep + left + extra_w
        assert h <= 30 and w <= 30
        grid = np.zeros((h, w), dtype=int)
        ra = random.randint(0, h - ha)
        rb = random.randint(0, h - hb)
        ca = left
        cb = left + wa + sep
    else:
        extra_h = random.randint(0, 3)
        extra_w = random.randint(0, 4)
        top = random.randint(0, 2)
        left = random.randint(0, 2)
        h = ha + hb + sep + top + extra_h
        w = max(wa, wb) + left + extra_w
        assert h <= 30 and w <= 30
        grid = np.zeros((h, w), dtype=int)
        ca = random.randint(0, w - wa)
        cb = random.randint(0, w - wb)
        ra = top
        rb = top + ha + sep

    grid[ra : ra + ha, ca : ca + wa] = patch_a
    nonzero_b = patch_b != BLACK
    target = grid[rb : rb + hb, cb : cb + wb]
    assert not np.any(nonzero_b & (target != BLACK))
    target[nonzero_b] = patch_b[nonzero_b]

    frames = []
    for frame in frames_a:
        r0, c0, r1, c1 = frame["bbox"]
        frames.append({"bbox": (r0 + ra, c0 + ca, r1 + ra, c1 + ca), "k": frame["k"]})
    for frame in frames_b:
        r0, c0, r1, c1 = frame["bbox"]
        frames.append({"bbox": (r0 + rb, c0 + cb, r1 + rb, c1 + cb), "k": frame["k"]})
    return grid, frames


def _same_size_counterexample(grid: np.ndarray) -> bool:
    frames = _detect_frames(grid)
    labels, _ = _label_mask(grid != BLACK)
    by_size: defaultdict[tuple[int, int], list[tuple[dict, int]]] = defaultdict(list)
    for frame in frames:
        comp = int(labels[frame["sample"]])
        by_size[frame["shape"]].append((frame, comp))
    out = main(grid)
    for items in by_size.values():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                frame_i, comp_i = items[i]
                frame_j, comp_j = items[j]
                if comp_i == comp_j:
                    continue
                if min(frame_i["interior_shape"]) < 5:
                    continue
                r0, c0, r1, c1 = frame_i["bbox"]
                s0, t0, s1, t1 = frame_j["bbox"]
                patch_i = out[r0 : r1 + 1, c0 : c1 + 1]
                patch_j = out[s0 : s1 + 1, t0 : t1 + 1]
                if not np.array_equal(patch_i, patch_j):
                    return True
    return False


def _semantic_check(grid: np.ndarray) -> None:
    frames = _detect_frames(grid)
    out = main(grid)
    assert out.shape == grid.shape
    assert np.any(out != grid)

    labels, comp_count = _label_mask(grid != BLACK)
    assert comp_count == 2
    frame_counts: defaultdict[int, int] = defaultdict(int)
    frame_by_comp: defaultdict[int, list[dict]] = defaultdict(list)
    for frame in frames:
        comp = int(labels[frame["sample"]])
        frame_counts[comp] += 1
        frame_by_comp[comp].append(frame)
        ih, iw = frame["interior_shape"]
        assert 3 <= ih <= 6
        assert 3 <= iw <= 6
    ks = sorted(frame_counts.values())
    assert tuple(ks) in PARTITIONS

    teal_labels, teal_n = _label_mask(grid == TEAL)
    assert teal_n == len(frames) - comp_count
    for cid in range(1, teal_n + 1):
        mask = teal_labels == cid
        coords = np.argwhere(mask)
        rows = {int(r) for r, _c in coords}
        cols = {int(c) for _r, c in coords}
        assert len(rows) == 1 or len(cols) == 1
        touched = set()
        for r, c in coords:
            teal_degree = 0
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    if grid[nr, nc] == TEAL:
                        teal_degree += 1
                    if grid[nr, nc] == GREY:
                        for idx, frame in enumerate(frames):
                            r0, c0, r1, c1 = frame["bbox"]
                            if r0 <= nr <= r1 and c0 <= nc <= c1:
                                touched.add(idx)
            assert teal_degree <= 2
        assert len(touched) == 2

    for comp, comp_frames in frame_by_comp.items():
        k = frame_counts[comp]
        colors = _ring_colors(k)
        for frame in comp_frames:
            r0, c0, r1, c1 = frame["bbox"]
            for r in range(r0 + 1, r1):
                for c in range(c0 + 1, c1):
                    depth = min(r - r0 - 1, r1 - 1 - r, c - c0 - 1, c1 - 1 - c)
                    assert out[r, c] == colors[depth % k]
    assert _same_size_counterexample(grid)


def _gen_attempt() -> np.ndarray | None:
    partition = random.choice(PARTITIONS)
    shared_size = random.choice(SHARED_SIZES)
    k1, k2 = partition

    def pick_orientation(k: int) -> str:
        if k >= 4:
            return "v"
        return random.choice(("h", "v"))

    try:
        patch1, frames1 = _build_component(k1, pick_orientation(k1), shared_size, use_shared=True)
        patch2, frames2 = _build_component(k2, pick_orientation(k2), shared_size, use_shared=True)
        grid, _ = _place_two_patches(patch1, frames1, patch2, frames2)
        _semantic_check(grid)
        return grid
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for opus/14/045/1")
