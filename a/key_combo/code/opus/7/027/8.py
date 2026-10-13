from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
FRAME = 8
SHAPE_COLORS = (1, 2, 3, 4)
COUNT_TO_COLOR = {0: 5, 1: 8}

SHAPE_TEMPLATES = {
    1: (
        np.array([(0, 0), (1, 0)], dtype=int),
        np.array([(0, 0), (1, 0), (2, 0)], dtype=int),
        np.array([(0, 0), (1, 0), (2, 0), (3, 0)], dtype=int),
    ),
    2: (
        np.array([(0, 0), (0, 1)], dtype=int),
        np.array([(0, 0), (1, 0), (1, 1)], dtype=int),
        np.array([(0, 1), (1, 0), (1, 1)], dtype=int),
        np.array([(0, 0), (0, 1), (1, 0), (1, 1)], dtype=int),
        np.array([(0, 0), (1, 0), (1, 1), (2, 1)], dtype=int),
        np.array([(0, 1), (1, 0), (1, 1), (2, 0)], dtype=int),
    ),
}


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _span_covers_frame(frame: dict, c0: int, c1: int) -> bool:
    return int(frame["c0"]) <= c0 and c1 <= int(frame["c1"])


def _span_in_gap(frame: dict, c0: int, c1: int) -> bool:
    gap = frame["gap"]
    return gap is not None and int(gap[0]) <= c0 and c1 <= int(gap[1])


def _frame_mask_from_spec(frame: dict) -> np.ndarray:
    h = int(frame["r1"] - frame["r0"] + 1)
    w = int(frame["c1"] - frame["c0"] + 1)
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    mask[-1, :] = True
    if frame["gap"] is not None:
        g0, g1 = frame["gap"]
        mask[-1, int(g0 - frame["c0"]) : int(g1 - frame["c0"]) + 1] = False
    return mask


def _parse_frames(grid: np.ndarray) -> list[dict]:
    frames = []
    for coords in _components(grid == FRAME):
        r0, c0, r1, c1 = _bbox(coords)
        if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 4:
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(patch[0, :] == FRAME):
            continue
        if not np.all(patch[:, 0] == FRAME) or not np.all(patch[:, -1] == FRAME):
            continue
        if not np.all(patch[1:-1, 1:-1] == BG):
            continue
        bottom = patch[-1, :]
        if bottom[0] != FRAME or bottom[-1] != FRAME:
            continue
        interior = bottom[1:-1]
        if not np.all(np.isin(interior, [BG, FRAME])):
            continue
        gap = None
        zeros = np.where(interior == BG)[0]
        if zeros.size:
            if int(zeros[-1] - zeros[0] + 1) != int(zeros.size):
                continue
            if zeros.size > 2:
                continue
            gap = (int(c0 + 1 + zeros[0]), int(c0 + 1 + zeros[-1]))
        expected = _frame_mask_from_spec({"r0": r0, "c0": c0, "r1": r1, "c1": c1, "gap": gap})
        if not np.array_equal(patch == FRAME, expected):
            continue
        frames.append(
            {
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "gap": gap,
                "coords": coords,
            }
        )
    frames.sort(key=lambda f: (int(f["r0"]), int(f["c0"])))
    for idx, frame in enumerate(frames):
        frame["index"] = idx
    return frames


def _parse_shapes(grid: np.ndarray) -> list[dict]:
    shapes = []
    for color in sorted(set(map(int, grid.ravel())) - {BG, FRAME}):
        for coords in _components(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            area = int(coords.shape[0])
            width = int(c1 - c0 + 1)
            if not (2 <= area <= 4 and width <= 2):
                continue
            shapes.append(
                {
                    "color": color,
                    "coords": coords,
                    "r0": r0,
                    "c0": c0,
                    "r1": r1,
                    "c1": c1,
                }
            )
    shapes.sort(key=lambda s: (int(s["c0"]), int(s["r0"])))
    return shapes


def _trace_span(frames: list[dict], c0: int, c1: int, start_bottom: int = -1) -> tuple[list[tuple[int, str]], int | None]:
    path: list[tuple[int, str]] = []
    landing = None
    for frame in frames:
        if int(frame["r0"]) <= start_bottom:
            continue
        if not _span_covers_frame(frame, c0, c1):
            continue
        action = "pass" if _span_in_gap(frame, c0, c1) else "stop"
        path.append((int(frame["index"]), action))
        if action == "stop":
            landing = int(frame["index"])
            break
    return path, landing


def _simulate(grid: np.ndarray, frames: list[dict], shapes: list[dict]) -> tuple[list[dict], list[int]]:
    h, _w = grid.shape
    results = []
    counts = [0] * len(frames)
    for shape in shapes:
        c0, c1 = int(shape["c0"]), int(shape["c1"])
        path, landing = _trace_span(frames, c0, c1, start_bottom=int(shape["r1"]))
        final_bottom = h - 1 if landing is None else int(frames[landing]["r0"]) - 1
        dr = final_bottom - int(shape["r1"])
        coords = shape["coords"].copy()
        coords[:, 0] += dr
        results.append(
            {
                "shape": shape,
                "path": path,
                "landing": landing,
                "coords": coords,
            }
        )
        if landing is not None:
            counts[landing] += 1
    return results, counts


def _render_output(grid: np.ndarray, frames: list[dict], sim: list[dict], counts: list[int]) -> np.ndarray:
    out = np.full_like(grid, BG)
    for idx, frame in enumerate(frames):
        color = COUNT_TO_COLOR.get(int(counts[idx]), 2)
        coords = frame["coords"]
        out[coords[:, 0], coords[:, 1]] = color
    for item in sim:
        coords = item["coords"]
        out[coords[:, 0], coords[:, 1]] = int(item["shape"]["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Drop each small shape through or onto cyan frames, then recolor each frame by how many shapes it catches.

    Input structure:
    - The grid has a black background, two or three cyan hollow rectangular frames, and several small 4-connected colored shapes above them.
    - A frame may have a one- or two-cell contiguous hole on its bottom edge, while its top and side edges stay intact.
    - Each small shape is monochrome, uses color 1-4, has 2-4 cells, and spans at most two columns.

    Rule:
    - Move every small shape straight downward without changing its shape or color.
    - When a shape reaches the top edge of a frame that fully covers its horizontal span, inspect that frame's bottom-edge hole over the same span: if one contiguous hole fully covers the span, the shape passes through; otherwise it stops with its bottom row one cell above the frame's top edge.
    - After all shapes settle, recolor each frame by the number of shapes resting above it: 0 -> gray (5), 1 -> cyan (8), 2 or more -> red (2).

    Key insight:
    - The hole is checked on the frame's bottom edge even though the collision happens at the top edge, so upper frames act as selective filters whose pass-through decisions change the lower frames' catch counts.
    """
    g = np.asarray(grid, dtype=int)
    frames = _parse_frames(g)
    shapes = _parse_shapes(g)
    sim, counts = _simulate(g, frames, shapes)
    return _render_output(g, frames, sim, counts)


def _make_frame(r0: int, c0: int, h: int, w: int, gap: tuple[int, int] | None) -> dict:
    return {
        "r0": int(r0),
        "c0": int(c0),
        "r1": int(r0 + h - 1),
        "c1": int(c0 + w - 1),
        "gap": None if gap is None else (int(gap[0]), int(gap[1])),
    }


def _draw_frame(grid: np.ndarray, frame: dict) -> None:
    r0, c0, r1, c1 = int(frame["r0"]), int(frame["c0"]), int(frame["r1"]), int(frame["c1"])
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME
    if frame["gap"] is not None:
        g0, g1 = frame["gap"]
        grid[r1, g0 : g1 + 1] = BG


def _layout_frames(num_frames: int) -> tuple[int, int, list[dict]]:
    overlap = random.randint(2, 3)
    w0 = random.randint(max(5, overlap + 2), 7)
    w1 = random.randint(max(6, overlap + 4), 7)
    h0 = random.randint(4, 5)
    h1 = random.randint(4, 5)
    c0 = random.randint(1, 3)
    c1 = c0 + w0 - overlap
    r0 = random.randint(6, 7)
    r1 = r0 + h0 + random.randint(2, 3)

    overlap_start = c1
    overlap_end = c0 + w0 - 2
    gap0_w = random.randint(1, min(2, overlap_end - overlap_start + 1))
    gap0_start = random.randint(overlap_start, overlap_end - gap0_w + 1)

    exclusive_start = c0 + w0
    exclusive_end = c1 + w1 - 2
    exclusive_size = exclusive_end - exclusive_start + 1
    gap1_w = 1 if exclusive_size == 2 else random.randint(1, min(2, exclusive_size - 1))
    gap1_start = random.randint(exclusive_start, exclusive_end - gap1_w + 1)

    frames = [
        _make_frame(r0, c0, h0, w0, (gap0_start, gap0_start + gap0_w - 1)),
        _make_frame(r1, c1, h1, w1, (gap1_start, gap1_start + gap1_w - 1)),
    ]

    max_right = c1 + w1
    last_bottom = r1 + h1 - 1
    if num_frames == 3:
        w2 = random.randint(4, 6)
        h2 = random.randint(4, 5)
        c2 = max_right + random.randint(2, 4)
        r2 = r1 + h1 + random.randint(2, 3)
        frames.append(_make_frame(r2, c2, h2, w2, None))
        max_right = c2 + w2
        last_bottom = r2 + h2 - 1

    min_width = max_right + 1
    max_width = min(30, max_right + 4)
    min_height = last_bottom + 3
    max_height = min(30, last_bottom + 6)
    if min_width > max_width or min_height > max_height:
        raise AssertionError("layout exceeds ARC size limit")
    width = random.randint(min_width, max_width)
    height = random.randint(min_height, max_height)
    return height, width, frames


def _enumerate_candidates(frames: list[dict], width: int) -> list[dict]:
    candidates = []
    ordered = [{**frame, "index": idx} for idx, frame in enumerate(frames)]
    for span_w in (1, 2):
        for c0 in range(0, width - span_w + 1):
            c1 = c0 + span_w - 1
            path, landing = _trace_span(ordered, c0, c1, start_bottom=-1)
            if not path:
                continue
            candidates.append(
                {
                    "c0": c0,
                    "c1": c1,
                    "width": span_w,
                    "path": path,
                    "landing": landing,
                }
            )
    return candidates


def _nonoverlap(a: dict, b: dict) -> bool:
    return int(a["c1"]) < int(b["c0"]) or int(b["c1"]) < int(a["c0"])


def _pick_spans(frames: list[dict], width: int) -> list[dict] | None:
    candidates = _enumerate_candidates(frames, width)
    role_a = [x for x in candidates if x["path"] == [(0, "pass"), (1, "stop")]]
    role_b = [x for x in candidates if x["path"] == [(0, "stop")]]
    role_c = [x for x in candidates if x["path"] == [(1, "pass")]]
    role_d = [x for x in candidates if x["path"] == [(1, "stop")]]
    if not (role_a and role_b and role_c and role_d):
        return None

    need_wide = random.random() < 0.6
    extra_target = random.randint(0, 2)
    for _ in range(800):
        chosen = []
        for pool in (
            random.sample(role_a, len(role_a)),
            random.sample(role_b, len(role_b)),
            random.sample(role_c, len(role_c)),
            random.sample(role_d, len(role_d)),
        ):
            picked = None
            for cand in pool:
                if all(_nonoverlap(cand, other) for other in chosen):
                    picked = cand
                    break
            if picked is None:
                break
            chosen.append(picked)
        if len(chosen) != 4:
            continue

        target_len = 4 + extra_target
        if len(chosen) < target_len:
            extras = random.sample(candidates, len(candidates))
            for cand in extras:
                if cand in chosen:
                    continue
                if all(_nonoverlap(cand, other) for other in chosen):
                    chosen.append(cand)
                if len(chosen) >= target_len:
                    break

        counts = [0] * len(frames)
        for cand in chosen:
            if cand["landing"] is not None:
                counts[int(cand["landing"])] += 1
        if counts[0] == 0 or counts[1] < 2:
            continue
        if len(frames) == 3 and counts[2] != 0:
            continue
        if len(set(counts)) < 2:
            continue
        if need_wide and not any(int(c["width"]) == 2 for c in chosen):
            continue
        return sorted(chosen, key=lambda x: (int(x["c0"]), int(x["c1"])))
    return None


def _place_shapes(height: int, frames: list[dict], spans: list[dict]) -> list[dict] | None:
    shapes = []
    palette = list(SHAPE_COLORS) * 2
    random.shuffle(palette)
    top_limit = int(frames[0]["r0"]) - 2
    for idx, span in enumerate(spans):
        templates = list(SHAPE_TEMPLATES[int(span["width"])])
        random.shuffle(templates)
        placed = None
        for template in templates:
            shape_h = int(template[:, 0].max()) + 1
            max_top = top_limit - shape_h
            if max_top < 0:
                continue
            top = random.randint(0, max_top)
            coords = template.copy()
            coords[:, 0] += top
            coords[:, 1] += int(span["c0"])
            placed = {
                "color": palette[idx],
                "coords": coords,
                "r0": int(coords[:, 0].min()),
                "c0": int(coords[:, 1].min()),
                "r1": int(coords[:, 0].max()),
                "c1": int(coords[:, 1].max()),
            }
            break
        if placed is None:
            return None
        shapes.append(placed)
    return shapes


def _build_grid(height: int, width: int, frames: list[dict], shapes: list[dict]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for frame in frames:
        _draw_frame(grid, frame)
    for shape in shapes:
        coords = shape["coords"]
        grid[coords[:, 0], coords[:, 1]] = int(shape["color"])
    return grid


def _semantic_checks(grid: np.ndarray) -> None:
    frames = _parse_frames(grid)
    shapes = _parse_shapes(grid)
    assert len(frames) in (2, 3)
    assert 4 <= len(shapes) <= 6
    top_frame = min(int(f["r0"]) for f in frames)
    spans = []
    for shape in shapes:
        assert int(shape["r1"]) < top_frame - 1
        spans.append((int(shape["c0"]), int(shape["c1"])))
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            assert spans[i][1] < spans[j][0] or spans[j][1] < spans[i][0]

    sim, counts = _simulate(grid, frames, shapes)
    assert any(len(item["path"]) >= 2 for item in sim)
    for idx, frame in enumerate(frames):
        if frame["gap"] is None:
            continue
        assert any(any(fi == idx and action == "pass" for fi, action in item["path"]) for item in sim)
        assert any(item["landing"] == idx for item in sim)
    assert len(set(counts)) >= 2
    if len(frames) == 3:
        assert 0 in counts

    out = main(grid)
    assert out.shape == grid.shape
    recolored = {0: 5, 1: 8}
    for idx, frame in enumerate(frames):
        expected = recolored.get(int(counts[idx]), 2)
        coords = frame["coords"]
        assert np.all(out[coords[:, 0], coords[:, 1]] == expected)
    for item in sim:
        coords = item["coords"]
        color = int(item["shape"]["color"])
        assert np.all(out[coords[:, 0], coords[:, 1]] == color)


def _gen_attempt() -> np.ndarray | None:
    num_frames = 3 if random.random() < 0.65 else 2
    try:
        height, width, frames = _layout_frames(num_frames)
        spans = _pick_spans(frames, width)
        if spans is None:
            return None
        shapes = _place_shapes(height, frames, spans)
        if shapes is None:
            return None
        grid = _build_grid(height, width, frames, shapes)
        _semantic_checks(grid)
        return grid
    except AssertionError:
        return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-8 input")
