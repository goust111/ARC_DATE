from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
GREY = 5
SIDES = ("top", "bottom", "left", "right")
ACTIVE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 400


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rs = coords[:, 0]
    cs = coords[:, 1]
    return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _objects(grid: np.ndarray, bg: int) -> list[dict]:
    objs: list[dict] = []
    for color0 in np.unique(grid):
        color = int(color0)
        if color == bg:
            continue
        for coords in _components(grid == color):
            objs.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return objs


def _is_frame(grid: np.ndarray, obj: dict) -> bool:
    r0, c0, r1, c1 = obj["bbox"]
    color = int(obj["color"])
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return False
    if int(obj["area"]) != 2 * h + 2 * w - 4:
        return False
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    return np.array_equal(sub == color, border)


def _adjacent_to_frame(cell: tuple[int, int], frame_bbox: tuple[int, int, int, int]) -> bool:
    r, c = map(int, cell)
    r0, c0, r1, c1 = map(int, frame_bbox)
    return (
        (r == r0 - 1 and c0 <= c <= c1)
        or (r == r1 + 1 and c0 <= c <= c1)
        or (c == c0 - 1 and r0 <= r <= r1)
        or (c == c1 + 1 and r0 <= r <= r1)
    )


def _adjacent_marker(frame: dict, non_frames: list[dict]) -> tuple[int, tuple[int, int]]:
    r0, c0, r1, c1 = frame["bbox"]
    _ = (r0, c0, r1, c1)
    hits = []
    for obj in non_frames:
        if int(obj["area"]) != 1:
            continue
        cell = tuple(map(int, obj["coords"][0]))
        if _adjacent_to_frame(cell, frame["bbox"]):
            hits.append((int(obj["color"]), cell))
    assert len(hits) == 1, "each frame must have exactly one adjacent marker pixel"
    return hits[0]


def _fragment_side(frag_bbox: tuple[int, int, int, int], frame_bbox: tuple[int, int, int, int]) -> str:
    fr0, fc0, fr1, fc1 = map(int, frame_bbox)
    r0, c0, r1, c1 = map(int, frag_bbox)
    if r1 < fr0:
        return "top"
    if r0 > fr1:
        return "bottom"
    if c1 < fc0:
        return "left"
    if c0 > fc1:
        return "right"
    raise AssertionError("fragment must lie strictly on one side of its frame")


def _push_inside(coords: np.ndarray, side: str, frame_bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, frame_bbox)
    inner_top = int(r0 + 1)
    inner_bottom = int(r1 - 1)
    inner_left = int(c0 + 1)
    inner_right = int(c1 - 1)

    moved = coords.copy()
    if side == "top":
        moved[:, 0] += inner_top - int(coords[:, 0].min())
    elif side == "bottom":
        moved[:, 0] += inner_bottom - int(coords[:, 0].max())
    elif side == "left":
        moved[:, 1] += inner_left - int(coords[:, 1].min())
    elif side == "right":
        moved[:, 1] += inner_right - int(coords[:, 1].max())
    else:
        raise AssertionError(f"unknown side: {side}")
    return moved


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each colored fragment set to its marked grey frame and push the fragments into the frame from their side.

    Input structure:
    - The background is black.
    - One or more grey hollow rectangular frames appear on the grid, each one cell thick and black inside.
    - Every frame has exactly one adjacent single-pixel color marker on one outer side.
    - Outside the frames, same-colored 4-connected fragments are scattered around the marked frame, and each fragment lies wholly above, below, left, or right of that frame.

    Rule:
    - Detect each grey frame and read the color of its adjacent marker pixel.
    - Collect all outside fragments of that color for the frame matched by the marker.
    - Push every fragment into the frame interior from the side where it currently sits: top fragments align to the inner top wall, bottom fragments to the inner bottom wall, left fragments to the inner left wall, and right fragments to the inner right wall while preserving the orthogonal coordinate.
    - Remove the original outside fragments and all marker pixels, and keep the grey frames unchanged.

    Key insight:
    - Color establishes which fragments belong to which frame, but the fragment's spatial side relative to that frame determines the translation that places it against the correct inner wall.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(np.bincount(g.ravel(), minlength=10).argmax())
    objs = _objects(g, bg)

    frames = [obj for obj in objs if _is_frame(g, obj)]
    non_frames = [obj for obj in objs if not _is_frame(g, obj)]

    out = np.full(g.shape, bg, dtype=int)
    for frame in frames:
        for r, c in frame["coords"]:
            out[int(r), int(c)] = int(frame["color"])

    marker_cells: set[tuple[int, int]] = set()
    color_to_frame: dict[int, dict] = {}
    for frame in frames:
        color, marker_pos = _adjacent_marker(frame, non_frames)
        marker_cells.add(marker_pos)
        color_to_frame[color] = frame

    for obj in non_frames:
        if int(obj["area"]) == 1 and tuple(map(int, obj["coords"][0])) in marker_cells:
            continue
        frame = color_to_frame.get(int(obj["color"]))
        assert frame is not None, "every fragment color must match a frame marker"
        side = _fragment_side(obj["bbox"], frame["bbox"])
        moved = _push_inside(obj["coords"], side, frame["bbox"])
        for r, c in moved:
            out[int(r), int(c)] = int(obj["color"])

    return out


def _bias_line(length: int, rng: np.random.Generator) -> np.ndarray:
    vals = np.zeros(length, dtype=int)
    vals[0] = int(rng.integers(0, 3))
    for i in range(1, length):
        vals[i] = int(np.clip(vals[i - 1] + int(rng.integers(-1, 2)), 0, 3))
    return vals


def _label_partition(h: int, w: int, rng: np.random.Generator) -> tuple[np.ndarray, list[tuple[str, np.ndarray]]] | None:
    order = list(map(int, rng.permutation(4)))
    rank = {side: i for i, side in enumerate(order)}
    top_bias = _bias_line(w, rng)
    bottom_bias = _bias_line(w, rng)
    left_bias = _bias_line(h, rng)
    right_bias = _bias_line(h, rng)

    labels = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            scores = [
                int(r + top_bias[c]),
                int((h - 1 - r) + bottom_bias[c]),
                int(c + left_bias[r]),
                int((w - 1 - c) + right_bias[r]),
            ]
            best = min(scores)
            winners = [side for side, score in enumerate(scores) if score == best]
            labels[r, c] = min(winners, key=lambda side: rank[side])

    counts = np.bincount(labels.ravel(), minlength=4)
    if np.any(counts < 2):
        return None

    pieces: list[tuple[str, np.ndarray]] = []
    singletons = 0
    for side_idx, side in enumerate(SIDES):
        comps = _components(labels == side_idx)
        if not comps:
            return None
        for comp in comps:
            singletons += int(comp.shape[0] == 1)
            pieces.append((side, comp))

    if len(pieces) < 4 or len(pieces) > 10:
        return None
    if singletons > 0:
        return None
    return labels, pieces


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        assert 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]
        assert int(grid[r, c]) == BG, "objects must not overlap"
        grid[r, c] = int(color)


def _cluster_pair(
    color: int,
    rng: np.random.Generator,
    *,
    size_range: tuple[int, int],
    extra_shift: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    h = int(rng.integers(size_range[0], size_range[1] + 1))
    w = int(rng.integers(size_range[0], size_range[1] + 1))

    partition = _label_partition(h, w, rng)
    if partition is None:
        return None
    _labels, pieces = partition

    frame_cells = []
    for rr in range(h + 2):
        for cc in range(w + 2):
            if rr in (0, h + 1) or cc in (0, w + 1):
                frame_cells.append((rr, cc))

    marker_side = SIDES[int(rng.integers(0, 4))]
    if marker_side == "top":
        marker_cell = (-1, int(rng.integers(1, w + 1)))
    elif marker_side == "bottom":
        marker_cell = (h + 2, int(rng.integers(1, w + 1)))
    elif marker_side == "left":
        marker_cell = (int(rng.integers(1, h + 1)), -1)
    else:
        marker_cell = (int(rng.integers(1, h + 1)), w + 2)

    input_cells: list[tuple[int, int, int]] = [(r, c, GREY) for r, c in frame_cells]
    input_cells.append((marker_cell[0], marker_cell[1], color))

    for side, comp in pieces:
        shift = {"top": h + 2, "bottom": h + 2, "left": w + 2, "right": w + 2}[side] + int(
            rng.integers(0, extra_shift + 1)
        )
        for r0, c0 in comp:
            r = int(r0) + 1
            c = int(c0) + 1
            if side == "top":
                input_cells.append((r - shift, c, color))
            elif side == "bottom":
                input_cells.append((r + shift, c, color))
            elif side == "left":
                input_cells.append((r, c - shift, color))
            else:
                input_cells.append((r, c + shift, color))

    min_r = min(r for r, _c, _v in input_cells)
    max_r = max(r for r, _c, _v in input_cells)
    min_c = min(c for _r, c, _v in input_cells)
    max_c = max(c for _r, c, _v in input_cells)

    offset_r = 1 - min_r
    offset_c = 1 - min_c
    patch_h = int(max_r - min_r + 3)
    patch_w = int(max_c - min_c + 3)
    inp = np.zeros((patch_h, patch_w), dtype=int)
    out = np.zeros((patch_h, patch_w), dtype=int)

    for r, c, value in input_cells:
        rr = int(r + offset_r)
        cc = int(c + offset_c)
        assert int(inp[rr, cc]) == BG
        inp[rr, cc] = int(value)

    frame_row0 = int(offset_r)
    frame_col0 = int(offset_c)
    for r, c in frame_cells:
        out[int(r + offset_r), int(c + offset_c)] = GREY
    out[frame_row0 + 1 : frame_row0 + h + 1, frame_col0 + 1 : frame_col0 + w + 1] = int(color)

    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def _assemble_single(patches: list[tuple[np.ndarray, np.ndarray]], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    inp_patch, out_patch = patches[0]
    ph, pw = map(int, inp_patch.shape)
    if ph > 30 or pw > 30:
        return None
    extra_h = int(rng.integers(0, min(30 - ph, 5) + 1))
    extra_w = int(rng.integers(0, min(30 - pw, 5) + 1))
    h = ph + extra_h
    w = pw + extra_w
    inp = np.zeros((h, w), dtype=int)
    out = np.zeros((h, w), dtype=int)
    r0 = int(rng.integers(0, h - ph + 1))
    c0 = int(rng.integers(0, w - pw + 1))
    inp[r0 : r0 + ph, c0 : c0 + pw] = inp_patch
    out[r0 : r0 + ph, c0 : c0 + pw] = out_patch
    return inp, out


def _assemble_double(patches: list[tuple[np.ndarray, np.ndarray]], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    rng.shuffle(patches)
    (inp_a, out_a), (inp_b, out_b) = patches
    ha, wa = map(int, inp_a.shape)
    hb, wb = map(int, inp_b.shape)

    options: list[tuple[str, int]] = []
    for gap in (1, 2):
        if max(ha, hb) <= 30 and wa + wb + gap <= 30:
            options.append(("horizontal", gap))
        if max(wa, wb) <= 30 and ha + hb + gap <= 30:
            options.append(("vertical", gap))
    if not options:
        return None

    orient, gap = options[int(rng.integers(0, len(options)))]
    if orient == "horizontal":
        base_h = max(ha, hb)
        base_w = wa + wb + gap
        h = base_h + int(rng.integers(0, min(30 - base_h, 2) + 1))
        w = base_w + int(rng.integers(0, min(30 - base_w, 2) + 1))
        left = int(rng.integers(0, w - base_w + 1))
        ra = int(rng.integers(0, h - ha + 1))
        rb = int(rng.integers(0, h - hb + 1))
        ca = left
        cb = left + wa + gap
    else:
        base_h = ha + hb + gap
        base_w = max(wa, wb)
        h = base_h + int(rng.integers(0, min(30 - base_h, 2) + 1))
        w = base_w + int(rng.integers(0, min(30 - base_w, 2) + 1))
        top = int(rng.integers(0, h - base_h + 1))
        ca = int(rng.integers(0, w - wa + 1))
        cb = int(rng.integers(0, w - wb + 1))
        ra = top
        rb = top + ha + gap

    inp = np.zeros((h, w), dtype=int)
    out = np.zeros((h, w), dtype=int)
    inp[ra : ra + ha, ca : ca + wa] = inp_a
    inp[rb : rb + hb, cb : cb + wb] = inp_b
    out[ra : ra + ha, ca : ca + wa] = out_a
    out[rb : rb + hb, cb : cb + wb] = out_b
    return inp, out


def _generate_pair() -> tuple[np.ndarray, np.ndarray] | None:
    rng = np.random.default_rng()
    frame_count = 1
    colors = [int(c) for c in rng.choice(ACTIVE_COLORS, size=frame_count, replace=False)]

    patches: list[tuple[np.ndarray, np.ndarray]] = []
    for color in colors:
        patch = None
        for _ in range(80):
            patch = _cluster_pair(color, rng, size_range=(4, 6), extra_shift=2)
            if patch is not None:
                break
        if patch is None:
            return None
        patches.append(patch)

    pair = _assemble_single(patches, rng)
    if pair is None:
        return None

    inp, out = pair
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        pair = _generate_pair()
        if pair is not None:
            return pair[0]
    raise RuntimeError("failed to generate a valid input")
