from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

ORIENTATIONS = ("br", "bl", "tl", "tr")
SIDES = ("top", "right", "bottom", "left")

# Corner position -> CCW quarter-turns for np.rot90.
ORIENTATION_TO_ROT = {
    "br": 0,  # ┘
    "bl": 3,  # └ -> CW 90
    "tl": 2,  # ┌
    "tr": 1,  # ┐ -> CCW 90
}

ORIENTATION_TO_MARKER = {
    "br": ((0, 1), (1, 0), (1, 1)),
    "bl": ((0, 0), (1, 0), (1, 1)),
    "tl": ((0, 0), (0, 1), (1, 0)),
    "tr": ((0, 0), (0, 1), (1, 1)),
}

MISSING_TO_ORIENTATION = {
    (0, 0): "br",
    (0, 1): "bl",
    (1, 1): "tl",
    (1, 0): "tr",
}

CANVAS_SIZE = 27
SLOT_SIZE = 13
SLOT_ORIGINS = ((1, 1), (1, 14), (14, 1), (14, 14))
MAX_ATTEMPTS = 400

_GEN_CALL_COUNT = 0


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_of_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    return components_of_mask(grid == int(color))


def _frame_from_component(coords: np.ndarray) -> dict | None:
    r0, c0, r1, c1 = bbox_coords(coords)
    side_h = r1 - r0 + 1
    side_w = c1 - c0 + 1
    if side_h != side_w or side_h < 5:
        return None

    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))

    got = {tuple(map(int, xy)) for xy in coords}
    if got != expected:
        return None

    return {
        "coords": coords,
        "bbox": (r0, c0, r1, c1),
        "inner_bbox": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
    }


def _marker_from_component(coords: np.ndarray) -> dict | None:
    if len(coords) != 3:
        return None
    r0, c0, r1, c1 = bbox_coords(coords)
    if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
        return None
    local = np.zeros((2, 2), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    if local.sum() != 3:
        return None
    missing = tuple(map(int, np.argwhere(~local)[0]))
    orientation = MISSING_TO_ORIENTATION.get(missing)
    if orientation is None:
        return None
    return {"coords": coords, "bbox": (r0, c0, r1, c1), "orientation": orientation}


def _pair_distance(a: np.ndarray, b: np.ndarray) -> int:
    d = np.abs(a[:, None, :] - b[None, :, :]).sum(axis=2)
    return int(d.min())


def _detect_frames(grid: np.ndarray) -> list[dict]:
    frames = []
    for comp in components_of_color(grid, GREY):
        frame = _frame_from_component(comp)
        if frame is not None:
            frames.append(frame)
    assert frames, "expected at least one square outline frame"
    return frames


def _detect_markers(grid: np.ndarray) -> list[dict]:
    markers = []
    for comp in components_of_color(grid, GREEN):
        marker = _marker_from_component(comp)
        if marker is not None:
            markers.append(marker)
    assert markers, "expected at least one L control marker"
    return markers


def _associate_frames_and_markers(frames: list[dict], markers: list[dict]) -> list[int]:
    assert len(frames) == len(markers), "expected one marker per frame"
    chosen: list[int] = []
    for frame in frames:
        distances = [_pair_distance(frame["coords"], marker["coords"]) for marker in markers]
        best = min(distances)
        winners = [idx for idx, dist in enumerate(distances) if dist == best]
        assert len(winners) == 1, "nearest marker must be unique for each frame"
        chosen.append(winners[0])
    assert len(set(chosen)) == len(chosen), "each marker must belong to exactly one frame"
    return chosen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each framed interior pattern according to the nearby L-marker orientation, then erase the markers.

    Input structure:
    - The background is black.
    - The grid contains 2 to 4 grey one-cell-thick square outline frames.
    - Each frame encloses a small blue/red 4-connected pattern, and each frame has exactly one nearby green 3-cell L marker outside the frame.

    Rule:
    - Detect every grey square outline frame and its strict interior.
    - For each frame, find the unique green L marker with the smallest pixel-to-frame-boundary Manhattan distance.
    - Read the marker orientation by the L corner position: ┘ -> 0 degrees, └ -> CW 90, ┌ -> 180, ┐ -> CCW 90.
    - Rotate only the frame interior by that angle around the interior square center, keep the grey frame unchanged, and erase all green marker pixels to black.

    Key insight:
    - The green L does not indicate a destination location; its four geometric orientations encode which quarter-turn to apply to the pattern inside the nearest enclosing square frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = _detect_frames(g)
    markers = _detect_markers(g)
    assignment = _associate_frames_and_markers(frames, markers)

    out = g.copy()
    for marker in markers:
        out[marker["coords"][:, 0], marker["coords"][:, 1]] = BLACK

    for frame, marker_idx in zip(frames, assignment):
        r0, c0, r1, c1 = frame["inner_bbox"]
        interior = out[r0 : r1 + 1, c0 : c1 + 1].copy()
        k = ORIENTATION_TO_ROT[markers[marker_idx]["orientation"]]
        out[r0 : r1 + 1, c0 : c1 + 1] = np.rot90(interior, k=k)

    return out


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY


def _random_connected_pattern(size: int) -> np.ndarray:
    for _ in range(300):
        count = random.randint(2, min(5, size * size - 1))
        start = (random.randrange(size), random.randrange(size))
        cells = {start}
        while len(cells) < count:
            frontier = []
            for r, c in cells:
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            cells.add(random.choice(frontier))

        arr = np.zeros((size, size), dtype=int)
        use_both = count >= 3 and random.random() < 0.7
        colors = [random.choice((BLUE, RED)) for _ in range(count)]
        if use_both and len(set(colors)) == 1:
            colors[random.randrange(count)] = BLUE if colors[0] == RED else RED
        if not use_both:
            fill = random.choice((BLUE, RED))
            colors = [fill] * count

        for (r, c), color in zip(sorted(cells), colors):
            arr[r, c] = color

        rotations = [np.rot90(arr, k=k) for k in range(4)]
        if len({rot.tobytes() for rot in rotations}) == 4:
            return arr
    raise RuntimeError("failed to build a non-degenerate interior pattern")


def _marker_top_left(
    frame_bbox: tuple[int, int, int, int],
    side: str,
) -> tuple[int, int]:
    r0, c0, r1, c1 = frame_bbox
    if side == "top":
        return r0 - 3, random.randint(c0 - 1, c1)
    if side == "bottom":
        return r1 + 2, random.randint(c0 - 1, c1)
    if side == "left":
        return random.randint(r0 - 1, r1), c0 - 3
    if side == "right":
        return random.randint(r0 - 1, r1), c1 + 2
    raise ValueError("invalid side")


def _place_marker(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    side: str,
    orientation: str,
) -> np.ndarray:
    top_r, top_c = _marker_top_left(frame_bbox, side)
    coords = []
    for dr, dc in ORIENTATION_TO_MARKER[orientation]:
        rr = top_r + dr
        cc = top_c + dc
        grid[rr, cc] = GREEN
        coords.append((rr, cc))
    return np.asarray(coords, dtype=int)


def _control_plan(call_id: int, n_frames: int) -> list[tuple[str, str]]:
    if n_frames == 2:
        side = random.choice(SIDES)
        base = random.choice(ORIENTATIONS)
        other_choices = [o for o in ORIENTATIONS if o != base]
        other = random.choice(other_choices)
        pairs = [(base, side), (other, side)]
    elif n_frames == 3:
        repeated = random.choice([o for o in ORIENTATIONS if o != "br"])
        other = random.choice([o for o in ORIENTATIONS if o != repeated])
        side_a, side_b = random.sample(SIDES, 2)
        side_c = random.choice(SIDES)
        pairs = [(repeated, side_a), (repeated, side_b), (other, side_c)]
    else:
        shift = call_id % 4
        ordered = list(ORIENTATIONS[shift:] + ORIENTATIONS[:shift])
        repeated_side = random.choice(SIDES)
        other_sides = random.sample([s for s in SIDES if s != repeated_side], 2)
        pairs = [
            (ordered[0], repeated_side),
            (ordered[1], repeated_side),
            (ordered[2], other_sides[0]),
            (ordered[3], other_sides[1]),
        ]
    random.shuffle(pairs)
    if not any(orientation != "br" for orientation, _side in pairs):
        orientation, side = pairs[0]
        replacement = random.choice([o for o in ORIENTATIONS if o != orientation])
        pairs[0] = (replacement, side)
    return pairs


def _expected_from_groups(grid: np.ndarray, groups: list[dict]) -> np.ndarray:
    out = grid.copy()
    for group in groups:
        out[group["marker_coords"][:, 0], group["marker_coords"][:, 1]] = BLACK
    for group in groups:
        r0, c0, r1, c1 = group["inner_bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = np.rot90(group["pattern"], k=ORIENTATION_TO_ROT[group["orientation"]])
    return out


def _trim_canvas(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    used = (inp != BLACK) | (out != BLACK)
    coords = np.argwhere(used)
    r0, c0, r1, c1 = bbox_coords(coords)
    top = random.randint(0, min(2, r0))
    left = random.randint(0, min(2, c0))
    bottom = random.randint(0, min(2, inp.shape[0] - 1 - r1))
    right = random.randint(0, min(2, inp.shape[1] - 1 - c1))
    box = (r0 - top, c0 - left, r1 + bottom, c1 + right)
    rr0, cc0, rr1, cc1 = box
    return inp[rr0 : rr1 + 1, cc0 : cc1 + 1].copy(), out[rr0 : rr1 + 1, cc0 : cc1 + 1].copy()


def _build_attempt(call_id: int, n_frames: int) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    grid = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    groups: list[dict] = []
    slots = random.sample(list(SLOT_ORIGINS), n_frames)
    controls = _control_plan(call_id, n_frames)

    for (slot_r, slot_c), (orientation, side) in zip(slots, controls):
        outer = random.randint(5, 7)
        inner = outer - 2
        max_offset = 13 - 6 - outer
        dr = random.randint(0, max_offset)
        dc = random.randint(0, max_offset)
        r0 = slot_r + 3 + dr
        c0 = slot_c + 3 + dc
        bbox = (r0, c0, r0 + outer - 1, c0 + outer - 1)
        _draw_frame(grid, bbox)
        pattern = _random_connected_pattern(inner)
        grid[r0 + 1 : r0 + 1 + inner, c0 + 1 : c0 + 1 + inner] = pattern
        marker_coords = _place_marker(grid, bbox, side, orientation)
        groups.append(
            {
                "bbox": bbox,
                "inner_bbox": (r0 + 1, c0 + 1, r0 + inner, c0 + inner),
                "pattern": pattern,
                "orientation": orientation,
                "side": side,
                "marker_coords": marker_coords,
            }
        )

    expected = _expected_from_groups(grid, groups)
    return grid, expected, groups


def _semantic_checks(inp: np.ndarray, out: np.ndarray, groups: list[dict]) -> None:
    assert max(inp.shape) <= 30 and max(out.shape) <= 30
    assert np.array_equal(main(inp), out), "main() must match the latent construction"

    frames = _detect_frames(inp)
    markers = _detect_markers(inp)
    assert len(frames) == len(groups)
    assert len(markers) == len(groups)
    _associate_frames_and_markers(frames, markers)

    erase_only = inp.copy()
    erase_only[erase_only == GREEN] = BLACK
    assert np.array_equal(out == GREEN, np.zeros_like(out, dtype=bool))
    assert any(group["orientation"] != "br" for group in groups), "need at least one visible rotation"
    assert not np.array_equal(erase_only, out), "rotation must matter, not just marker erasure"

    side_to_orientations: dict[str, set[str]] = {}
    orientation_to_sides: dict[str, set[str]] = {}
    for group in groups:
        side_to_orientations.setdefault(group["side"], set()).add(group["orientation"])
        orientation_to_sides.setdefault(group["orientation"], set()).add(group["side"])
    assert any(len(v) >= 2 for v in side_to_orientations.values()) or any(
        len(v) >= 2 for v in orientation_to_sides.values()
    ), "sample should contain a counterexample to side-orientation shortcuts"


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    call_id = _GEN_CALL_COUNT
    _GEN_CALL_COUNT += 1

    if call_id < 3:
        n_frames = 2 + call_id
    else:
        n_frames = random.randint(2, 4)

    for _ in range(MAX_ATTEMPTS):
        inp_full, out_full, groups = _build_attempt(call_id, n_frames)
        inp, out = _trim_canvas(inp_full, out_full)
        try:
            _semantic_checks(inp, out, groups)
        except AssertionError:
            continue
        return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
