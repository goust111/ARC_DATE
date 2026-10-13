from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BACKGROUND = 0
BLUE = 1
MARKER = 2
GREEN = 3
YELLOW = 4
FRAME = 5
ORANGE = 7
RING_CYCLE = np.array([GREEN, YELLOW, BLUE, ORANGE], dtype=int)

GRID_ROWS_RANGE = (17, 25)
GRID_COLS_RANGE = (17, 25)
FRAME_COUNT_CHOICES = (2, 3)
INNER_SIDE_RANGE = (5, 10)
PLACEMENT_GAP = 1
PLACEMENT_TRIES = 220
GEN_ATTEMPTS = 700


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _border_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _extract_frames(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for comp in _components(grid == FRAME):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= 3 and w >= 3, "frame must enclose a non-empty interior"

        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = _border_mask(h, w)
        assert len(comp) == int(border.sum()), "grey object must be a rectangular border"
        assert np.all(patch[border] == FRAME), "frame border must be solid grey"

        inner = patch[1:-1, 1:-1]
        assert inner.shape[0] >= 1 and inner.shape[1] >= 1, "frame interior must exist"
        allowed = {int(bg), MARKER}
        assert set(map(int, inner.ravel())).issubset(allowed), "frame interior may contain only background and one red marker"
        markers = np.argwhere(inner == MARKER)
        assert markers.shape == (1, 2), "each frame must contain exactly one red marker"
        mr, mc = map(int, markers[0])

        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "inner_bbox": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
                "inner_shape": (int(inner.shape[0]), int(inner.shape[1])),
                "marker_local": (mr, mc),
                "marker_global": (r0 + 1 + mr, c0 + 1 + mc),
            }
        )

    marker_count = int(np.count_nonzero(grid == MARKER))
    assert marker_count == len(frames), "all red pixels must be frame markers"
    frames.sort(key=lambda frame: tuple(int(v) for v in frame["bbox"]))
    return frames


def _distance_patch(inner_h: int, inner_w: int, marker: tuple[int, int]) -> np.ndarray:
    mr, mc = map(int, marker)
    rows, cols = np.indices((inner_h, inner_w))
    dist = np.abs(rows - mr) + np.abs(cols - mc)
    patch = np.empty((inner_h, inner_w), dtype=int)
    patch[dist == 0] = MARKER
    nonzero = dist > 0
    patch[nonzero] = RING_CYCLE[(dist[nonzero] - 1) % len(RING_CYCLE)]
    return patch


def _mirror_patch(patch: np.ndarray) -> np.ndarray:
    h, w = patch.shape
    out = patch.copy()
    if w > h:
        axis = (w - 1) // 2
        if w % 2 == 0:
            out[:, axis + 1 :] = np.fliplr(patch[:, : axis + 1])
        else:
            out[:, axis + 1 :] = np.fliplr(patch[:, :axis])
        return out

    if h > w:
        axis = (h - 1) // 2
        if h % 2 == 0:
            out[axis + 1 :, :] = np.flipud(patch[: axis + 1, :])
        else:
            out[axis + 1 :, :] = np.flipud(patch[:axis, :])
        return out

    raise AssertionError("frame interior must have a strict orientation")


def _expected_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    assert int(bg) == BACKGROUND, "expected black background"
    out = g.copy()
    for frame in _extract_frames(g, bg):
        r0, c0, r1, c1 = map(int, frame["inner_bbox"])
        inner_h, inner_w = map(int, frame["inner_shape"])
        marker = tuple(map(int, frame["marker_local"]))
        rings = _distance_patch(inner_h, inner_w, marker)
        out[r0 : r1 + 1, c0 : c1 + 1] = _mirror_patch(rings)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each rectangular frame interior with Manhattan-distance rings, then mirror those rings according to the frame's aspect ratio.

    Input structure:
    - A black background contains 2-3 disjoint one-pixel-thick grey rectangular frames.
    - Each frame encloses a black interior of at least 5x5 cells, and each interior contains exactly one red marker pixel.
    - The frame interiors are strictly wide or strictly tall, never square.

    Rule:
    - For every frame interior, compute each cell's Manhattan distance from the red marker.
    - Color distance 0 as red, then cycle distance 1->green, 2->yellow, 3->blue, 4->orange, and repeat that 4-color cycle for larger distances.
    - If the interior is wider than tall, copy the left side of that ring pattern onto the right side by left-to-right mirroring inside the interior.
    - If the interior is taller than wide, copy the top side of that ring pattern onto the bottom side by top-to-bottom mirroring inside the interior.
    - Keep the grey frame borders unchanged and leave everything outside the frames black.

    Key insight:
    - The red marker defines a truncated distance field, but the frame's own geometry decides which half of that field survives and which half is replaced by its mirror image.
    """
    return _expected_output(grid)


def _boxes_clear(candidate: tuple[int, int, int, int], placed: list[tuple[int, int, int, int]]) -> bool:
    r0, c0, r1, c1 = candidate
    for pr0, pc0, pr1, pc1 in placed:
        if not (r1 + PLACEMENT_GAP < pr0 or pr1 + PLACEMENT_GAP < r0 or c1 + PLACEMENT_GAP < pc0 or pc1 + PLACEMENT_GAP < c0):
            return False
    return True


def _sample_inner_dims(orientation: str, mirrored_even: bool) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    for inner_h in range(*INNER_SIDE_RANGE):
        for inner_w in range(*INNER_SIDE_RANGE):
            if inner_h < 5 or inner_w < 5 or inner_h == inner_w:
                continue
            if orientation == "wide" and inner_w > inner_h and (inner_w % 2 == 0) == mirrored_even:
                candidates.append((inner_h, inner_w))
            if orientation == "tall" and inner_h > inner_w and (inner_h % 2 == 0) == mirrored_even:
                candidates.append((inner_h, inner_w))
    assert candidates, f"no dimension candidates for {orientation=} {mirrored_even=}"
    return random.choice(candidates)


def _source_axis(inner_h: int, inner_w: int) -> tuple[str, int]:
    if inner_w > inner_h:
        return "col", (inner_w - 1) // 2
    return "row", (inner_h - 1) // 2


def _raw_pattern_changes_under_mirror(inner_h: int, inner_w: int, marker: tuple[int, int]) -> bool:
    raw = _distance_patch(inner_h, inner_w, marker)
    mirrored = _mirror_patch(raw)
    return not np.array_equal(raw, mirrored)


def _sample_marker(inner_h: int, inner_w: int) -> tuple[int, int] | None:
    axis_kind, axis = _source_axis(inner_h, inner_w)
    candidates: list[tuple[int, int]] = []
    for mr in range(inner_h):
        for mc in range(inner_w):
            if axis_kind == "col":
                if mc >= axis:
                    continue
            else:
                if mr >= axis:
                    continue
            if inner_h % 2 == 1 and inner_w % 2 == 1 and mr == inner_h // 2 and mc == inner_w // 2:
                continue
            if not _raw_pattern_changes_under_mirror(inner_h, inner_w, (mr, mc)):
                continue
            candidates.append((mr, mc))
    if not candidates:
        return None

    edge_biased = [
        cell
        for cell in candidates
        if min(cell[0], inner_h - 1 - cell[0], cell[1], inner_w - 1 - cell[1]) <= 1
    ]
    if edge_biased and random.random() < 0.7:
        return random.choice(edge_biased)
    return random.choice(candidates)


def _render_input(rows: int, cols: int, specs: list[dict[str, object]], boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    grid = np.full((rows, cols), BACKGROUND, dtype=int)
    for spec, (r0, c0, r1, c1) in zip(specs, boxes):
        grid[r0, c0 : c1 + 1] = FRAME
        grid[r1, c0 : c1 + 1] = FRAME
        grid[r0 : r1 + 1, c0] = FRAME
        grid[r0 : r1 + 1, c1] = FRAME
        mr, mc = map(int, spec["marker_local"])
        grid[r0 + 1 + mr, c0 + 1 + mc] = MARKER
    return grid


def _sample_specs() -> list[dict[str, object]] | None:
    frame_count = random.choice(FRAME_COUNT_CHOICES)
    orientations = ["wide", "tall"]
    if frame_count == 3:
        orientations.append(random.choice(("wide", "tall")))
    random.shuffle(orientations)

    parity_flags = [True, False]
    if frame_count == 3:
        parity_flags.append(random.choice((True, False)))
    random.shuffle(parity_flags)

    specs: list[dict[str, object]] = []
    for orientation, mirrored_even in zip(orientations, parity_flags):
        inner_h, inner_w = _sample_inner_dims(orientation, mirrored_even)
        marker = _sample_marker(inner_h, inner_w)
        if marker is None:
            return None
        specs.append(
            {
                "orientation": orientation,
                "mirrored_even": mirrored_even,
                "inner_shape": (inner_h, inner_w),
                "outer_shape": (inner_h + 2, inner_w + 2),
                "marker_local": marker,
            }
        )
    return specs


def _place_specs(rows: int, cols: int, specs: list[dict[str, object]]) -> list[tuple[int, int, int, int]] | None:
    order = sorted(
        range(len(specs)),
        key=lambda idx: int(specs[idx]["outer_shape"][0]) * int(specs[idx]["outer_shape"][1]),
        reverse=True,
    )
    placed: list[tuple[int, int, int, int] | None] = [None] * len(specs)
    used: list[tuple[int, int, int, int]] = []
    for idx in order:
        outer_h, outer_w = map(int, specs[idx]["outer_shape"])
        if rows < outer_h or cols < outer_w:
            return None
        for _ in range(PLACEMENT_TRIES):
            r0 = random.randint(0, rows - outer_h)
            c0 = random.randint(0, cols - outer_w)
            box = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            if not _boxes_clear(box, used):
                continue
            used.append(box)
            placed[idx] = box
            break
        if placed[idx] is None:
            return None
    return [box for box in placed if box is not None]


def _semantic_check(inp: np.ndarray, out: np.ndarray) -> bool:
    if inp.shape != out.shape or np.array_equal(inp, out):
        return False
    if _most_common_color(inp) != BACKGROUND:
        return False

    try:
        frames = _extract_frames(inp, BACKGROUND)
    except AssertionError:
        return False

    if len(frames) not in FRAME_COUNT_CHOICES:
        return False

    orientations = set()
    mirrored_parities = set()
    no_mirror_out = inp.copy()

    for frame in frames:
        r0, c0, r1, c1 = map(int, frame["bbox"])
        ir0, ic0, ir1, ic1 = map(int, frame["inner_bbox"])
        inner_h, inner_w = map(int, frame["inner_shape"])
        marker = tuple(map(int, frame["marker_local"]))

        if inner_h < 5 or inner_w < 5 or inner_h == inner_w:
            return False

        border = _border_mask(r1 - r0 + 1, c1 - c0 + 1)
        if not np.all(inp[r0 : r1 + 1, c0 : c1 + 1][border] == FRAME):
            return False
        if not np.all(out[r0 : r1 + 1, c0 : c1 + 1][border] == FRAME):
            return False

        rings = _distance_patch(inner_h, inner_w, marker)
        mirrored = _mirror_patch(rings)
        if not _raw_pattern_changes_under_mirror(inner_h, inner_w, marker):
            return False
        if not np.array_equal(out[ir0 : ir1 + 1, ic0 : ic1 + 1], mirrored):
            return False

        no_mirror_out[ir0 : ir1 + 1, ic0 : ic1 + 1] = rings

        if inner_w > inner_h:
            orientations.add("wide")
            mirrored_parities.add(inner_w % 2)
            if marker[1] >= (inner_w - 1) // 2:
                return False
        else:
            orientations.add("tall")
            mirrored_parities.add(inner_h % 2)
            if marker[0] >= (inner_h - 1) // 2:
                return False

    if orientations != {"wide", "tall"}:
        return False
    if mirrored_parities != {0, 1}:
        return False
    if np.array_equal(out, no_mirror_out):
        return False
    if np.any(inp == ORANGE) or np.any(inp == GREEN) or np.any(inp == YELLOW) or np.any(inp == BLUE):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    rows = random.randint(*GRID_ROWS_RANGE)
    cols = random.randint(*GRID_COLS_RANGE)
    specs = _sample_specs()
    if specs is None:
        return None
    boxes = _place_specs(rows, cols, specs)
    if boxes is None:
        return None

    inp = _render_input(rows, cols, specs, boxes)
    expected = _expected_output(inp)
    try:
        actual = main(inp)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    if not _semantic_check(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        inp = _generate_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid input for puzzle 4")
