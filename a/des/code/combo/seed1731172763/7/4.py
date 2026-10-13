from __future__ import annotations

import random

import numpy as np


BG = 0
SIDES = ("top", "bottom", "left", "right")
HIT_SIDE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}

GRID_HEIGHT_RANGE = (16, 24)
GRID_WIDTH_RANGE = (16, 24)
FRAME_COUNT_RANGE = (2, 5)
FRAME_HEIGHT_RANGE = (4, 7)
FRAME_WIDTH_RANGE = (4, 7)
MIN_RAY_STEPS = 2
FRAME_PADDING = 1
GEN_RETRY_LIMIT = 400
FRAME_PLACEMENT_TRIES = 250
BLOCKED_PAIR_TRIES = 350
BBOX_SAMPLE_TRIES = 60
BLOCKER_GAP_TRIES = 24


def _find_components(g: np.ndarray) -> list[dict]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords = np.array(cells, dtype=int)
            rows = coords[:, 0]
            cols = coords[:, 1]
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (
                        int(rows.min()),
                        int(cols.min()),
                        int(rows.max()),
                        int(cols.max()),
                    ),
                }
            )
    return components


def _frame_spec(color: int, bbox: tuple[int, int, int, int], gap_side: str, gap_lo: int, gap_hi: int) -> dict:
    return {
        "color": int(color),
        "bbox": tuple(map(int, bbox)),
        "gap_side": gap_side,
        "gap_lo": int(gap_lo),
        "gap_hi": int(gap_hi),
        "gap_width": int(gap_hi - gap_lo + 1),
    }


def _decode_frame(component: dict) -> dict:
    color = int(component["color"])
    coords = component["coords"]
    r0, c0, r1, c1 = map(int, component["bbox"])
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    assert h >= 4 and w >= 4, "frame must have room for a non-corner gap"

    pts = {(int(r), int(c)) for r, c in coords}
    for r, c in pts:
        assert r in (r0, r1) or c in (c0, c1), "frame pixels must stay on the bbox perimeter"

    side_present = {
        "top": np.array([(r0, c) in pts for c in range(c0, c1 + 1)], dtype=bool),
        "bottom": np.array([(r1, c) in pts for c in range(c0, c1 + 1)], dtype=bool),
        "left": np.array([(r, c0) in pts for r in range(r0, r1 + 1)], dtype=bool),
        "right": np.array([(r, c1) in pts for r in range(r0, r1 + 1)], dtype=bool),
    }

    missing_sides: list[tuple[str, np.ndarray]] = []
    for side, present in side_present.items():
        missing = np.flatnonzero(~present)
        if missing.size == 0:
            continue
        assert np.all(np.diff(missing) == 1), "gap must be one contiguous run"
        assert int(missing[0]) > 0 and int(missing[-1]) < present.size - 1, "gap may not remove corners"
        missing_sides.append((side, missing))

    assert len(missing_sides) == 1, "expected exactly one gapped side"
    gap_side, missing = missing_sides[0]
    gap_lo = int(missing[0])
    gap_hi = int(missing[-1])

    for side, present in side_present.items():
        if side != gap_side:
            assert bool(present.all()), "non-gapped sides must be solid"

    expected_area = 2 * (h + w) - 4 - (gap_hi - gap_lo + 1)
    assert len(pts) == expected_area, "unexpected extra or missing frame pixels"
    return _frame_spec(color, (r0, c0, r1, c1), gap_side, gap_lo, gap_hi)


def _ray_strip(frame: dict, step: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, frame["bbox"])
    gap_lo = int(frame["gap_lo"])
    gap_hi = int(frame["gap_hi"])
    side = frame["gap_side"]

    if side == "top":
        row = r0 - step
        return [(row, c0 + offset) for offset in range(gap_lo, gap_hi + 1)]
    if side == "bottom":
        row = r1 + step
        return [(row, c0 + offset) for offset in range(gap_lo, gap_hi + 1)]
    if side == "left":
        col = c0 - step
        return [(r0 + offset, col) for offset in range(gap_lo, gap_hi + 1)]
    col = c1 + step
    return [(r0 + offset, col) for offset in range(gap_lo, gap_hi + 1)]


def _in_bounds(shape: tuple[int, int], cells: list[tuple[int, int]]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _draw_beam(out: np.ndarray, obstacle_mask: np.ndarray, frame: dict) -> None:
    color = int(frame["color"])
    step = 1
    while True:
        cells = _ray_strip(frame, step)
        if not _in_bounds(out.shape, cells):
            return
        if any(bool(obstacle_mask[r, c]) for r, c in cells):
            return
        for r, c in cells:
            assert int(out[r, c]) in (BG, color), "beam overlap must be unambiguous"
            out[r, c] = color
        step += 1


def _beam_stop_reason(frame: dict, obstacle_mask: np.ndarray, shape: tuple[int, int]) -> str:
    step = 1
    while True:
        cells = _ray_strip(frame, step)
        if not _in_bounds(shape, cells):
            return "edge"
        if any(bool(obstacle_mask[r, c]) for r, c in cells):
            return "frame"
        step += 1


def _frame_patch(frame: dict) -> np.ndarray:
    r0, c0, r1, c1 = map(int, frame["bbox"])
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    color = int(frame["color"])
    patch = np.full((h, w), BG, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color

    gap_lo = int(frame["gap_lo"])
    gap_hi = int(frame["gap_hi"])
    side = frame["gap_side"]
    if side == "top":
        patch[0, gap_lo : gap_hi + 1] = BG
    elif side == "bottom":
        patch[-1, gap_lo : gap_hi + 1] = BG
    elif side == "left":
        patch[gap_lo : gap_hi + 1, 0] = BG
    else:
        patch[gap_lo : gap_hi + 1, -1] = BG
    return patch


def _render_input(shape: tuple[int, int], frames: list[dict]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for frame in frames:
        patch = _frame_patch(frame)
        r0, c0, _r1, _c1 = map(int, frame["bbox"])
        mask = patch != BG
        window = out[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
        assert not np.any(window[mask] != BG)
        window[mask] = patch[mask]
    return out


def _render_output_from_specs(inp: np.ndarray, frames: list[dict]) -> np.ndarray:
    out = inp.copy()
    obstacle_mask = inp != BG
    for frame in frames:
        _draw_beam(out, obstacle_mask, frame)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify the opening side of each colored frame and shoot a same-colored beam outward through that opening.

    The beam width equals the opening width and it keeps extending until the next strip would hit
    another frame pixel or leave the grid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = [_decode_frame(component) for component in _find_components(g)]
    out = g.copy()
    obstacle_mask = g != BG
    for frame in frames:
        _draw_beam(out, obstacle_mask, frame)
    return out


def _bbox_free(forbidden: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = map(int, bbox)
    return not bool(forbidden[r0 : r1 + 1, c0 : c1 + 1].any())


def _reserve_frame(scene: dict, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    scene["occupied"][r0 : r1 + 1, c0 : c1 + 1] = True
    pr0 = max(0, r0 - FRAME_PADDING)
    pc0 = max(0, c0 - FRAME_PADDING)
    pr1 = min(scene["occupied"].shape[0] - 1, r1 + FRAME_PADDING)
    pc1 = min(scene["occupied"].shape[1] - 1, c1 + FRAME_PADDING)
    scene["forbidden"][pr0 : pr1 + 1, pc0 : pc1 + 1] = True


def _reserve_cells(scene: dict, cells: list[tuple[int, int]]) -> None:
    if not cells:
        return
    rows = np.array([r for r, _c in cells], dtype=int)
    cols = np.array([c for _r, c in cells], dtype=int)
    scene["occupied"][rows, cols] = True
    scene["forbidden"][rows, cols] = True


def _cells_clear(occupied: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    if not cells:
        return False
    rows = np.array([r for r, _c in cells], dtype=int)
    cols = np.array([c for _r, c in cells], dtype=int)
    return not bool(occupied[rows, cols].any())


def _sample_gap(side: str, h: int, w: int) -> tuple[int, int]:
    span = w if side in ("top", "bottom") else h
    gap_width = random.randint(1, span - 2)
    gap_lo = random.randint(1, span - gap_width - 1)
    return gap_lo, gap_lo + gap_width - 1


def _sample_bbox(
    scene: dict,
    h: int,
    w: int,
    r0_lo: int,
    r0_hi: int,
    c0_lo: int,
    c0_hi: int,
) -> tuple[int, int, int, int] | None:
    if r0_lo > r0_hi or c0_lo > c0_hi:
        return None
    for _ in range(BBOX_SAMPLE_TRIES):
        r0 = random.randint(r0_lo, r0_hi)
        c0 = random.randint(c0_lo, c0_hi)
        bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
        if _bbox_free(scene["forbidden"], bbox):
            return bbox
    return None


def _edge_corridor(shape: tuple[int, int], bbox: tuple[int, int, int, int], side: str, gap_lo: int, gap_hi: int) -> list[tuple[int, int]] | None:
    h, w = shape
    r0, c0, r1, c1 = map(int, bbox)
    cells: list[tuple[int, int]] = []

    if side == "top":
        if r0 < MIN_RAY_STEPS:
            return None
        for r in range(r0 - 1, -1, -1):
            for c in range(c0 + gap_lo, c0 + gap_hi + 1):
                cells.append((r, c))
    elif side == "bottom":
        if h - 1 - r1 < MIN_RAY_STEPS:
            return None
        for r in range(r1 + 1, h):
            for c in range(c0 + gap_lo, c0 + gap_hi + 1):
                cells.append((r, c))
    elif side == "left":
        if c0 < MIN_RAY_STEPS:
            return None
        for c in range(c0 - 1, -1, -1):
            for r in range(r0 + gap_lo, r0 + gap_hi + 1):
                cells.append((r, c))
    else:
        if w - 1 - c1 < MIN_RAY_STEPS:
            return None
        for c in range(c1 + 1, w):
            for r in range(r0 + gap_lo, r0 + gap_hi + 1):
                cells.append((r, c))
    return cells


def _blocked_corridor(emitter: dict, blocker_bbox: tuple[int, int, int, int]) -> list[tuple[int, int]] | None:
    er0, ec0, er1, ec1 = map(int, emitter["bbox"])
    br0, bc0, br1, bc1 = map(int, blocker_bbox)
    gap_lo = int(emitter["gap_lo"])
    gap_hi = int(emitter["gap_hi"])
    side = emitter["gap_side"]
    cells: list[tuple[int, int]] = []

    if side == "top":
        start = br1 + 1
        stop = er0 - 1
        if stop - start + 1 < MIN_RAY_STEPS:
            return None
        for r in range(start, stop + 1):
            for c in range(ec0 + gap_lo, ec0 + gap_hi + 1):
                cells.append((r, c))
    elif side == "bottom":
        start = er1 + 1
        stop = br0 - 1
        if stop - start + 1 < MIN_RAY_STEPS:
            return None
        for r in range(start, stop + 1):
            for c in range(ec0 + gap_lo, ec0 + gap_hi + 1):
                cells.append((r, c))
    elif side == "left":
        start = bc1 + 1
        stop = ec0 - 1
        if stop - start + 1 < MIN_RAY_STEPS:
            return None
        for c in range(start, stop + 1):
            for r in range(er0 + gap_lo, er0 + gap_hi + 1):
                cells.append((r, c))
    else:
        start = ec1 + 1
        stop = bc0 - 1
        if stop - start + 1 < MIN_RAY_STEPS:
            return None
        for c in range(start, stop + 1):
            for r in range(er0 + gap_lo, er0 + gap_hi + 1):
                cells.append((r, c))
    return cells


def _gap_world_span(frame: dict) -> tuple[int, int]:
    r0, c0, _r1, _c1 = map(int, frame["bbox"])
    return (
        c0 + int(frame["gap_lo"]),
        c0 + int(frame["gap_hi"]),
    ) if frame["gap_side"] in ("top", "bottom") else (
        r0 + int(frame["gap_lo"]),
        r0 + int(frame["gap_hi"]),
    )


def _try_place_edge_frame(scene: dict, color: int, preferred_side: str | None = None) -> dict | None:
    h_lim, w_lim = scene["occupied"].shape
    side_choices = [preferred_side] if preferred_side is not None else list(SIDES)

    for _ in range(FRAME_PLACEMENT_TRIES):
        side = random.choice(side_choices)
        fh = random.randint(*FRAME_HEIGHT_RANGE)
        fw = random.randint(*FRAME_WIDTH_RANGE)
        gap_lo, gap_hi = _sample_gap(side, fh, fw)

        if side == "top":
            bbox = _sample_bbox(scene, fh, fw, MIN_RAY_STEPS, h_lim - fh, 0, w_lim - fw)
        elif side == "bottom":
            bbox = _sample_bbox(scene, fh, fw, 0, h_lim - fh - MIN_RAY_STEPS, 0, w_lim - fw)
        elif side == "left":
            bbox = _sample_bbox(scene, fh, fw, 0, h_lim - fh, MIN_RAY_STEPS, w_lim - fw)
        else:
            bbox = _sample_bbox(scene, fh, fw, 0, h_lim - fh, 0, w_lim - fw - MIN_RAY_STEPS)
        if bbox is None:
            continue

        corridor = _edge_corridor(scene["occupied"].shape, bbox, side, gap_lo, gap_hi)
        if corridor is None or not _cells_clear(scene["occupied"], corridor):
            continue

        frame = _frame_spec(color, bbox, side, gap_lo, gap_hi)
        _reserve_frame(scene, bbox)
        _reserve_cells(scene, corridor)
        scene["frames"].append(frame)
        return frame
    return None


def _random_blocker_dims(emitter_side: str, min_span: int) -> tuple[int, int] | None:
    for _ in range(40):
        bh = random.randint(*FRAME_HEIGHT_RANGE)
        bw = random.randint(*FRAME_WIDTH_RANGE)
        if emitter_side in ("top", "bottom") and bw < min_span:
            continue
        if emitter_side in ("left", "right") and bh < min_span:
            continue
        return bh, bw
    return None


def _sample_blocker_bbox(scene: dict, emitter: dict, blocker_h: int, blocker_w: int) -> tuple[int, int, int, int] | None:
    h_lim, w_lim = scene["occupied"].shape
    er0, ec0, er1, ec1 = map(int, emitter["bbox"])
    span_lo, span_hi = _gap_world_span(emitter)
    side = emitter["gap_side"]

    for _ in range(BBOX_SAMPLE_TRIES):
        if side == "top":
            c0_lo = max(0, span_hi - blocker_w + 1)
            c0_hi = min(span_lo, w_lim - blocker_w)
            r1_lo = blocker_h - 1
            r1_hi = er0 - MIN_RAY_STEPS - 1
            if c0_lo > c0_hi or r1_lo > r1_hi:
                return None
            c0 = random.randint(c0_lo, c0_hi)
            r1 = random.randint(r1_lo, r1_hi)
            bbox = (r1 - blocker_h + 1, c0, r1, c0 + blocker_w - 1)
        elif side == "bottom":
            c0_lo = max(0, span_hi - blocker_w + 1)
            c0_hi = min(span_lo, w_lim - blocker_w)
            r0_lo = er1 + MIN_RAY_STEPS + 1
            r0_hi = h_lim - blocker_h
            if c0_lo > c0_hi or r0_lo > r0_hi:
                return None
            c0 = random.randint(c0_lo, c0_hi)
            r0 = random.randint(r0_lo, r0_hi)
            bbox = (r0, c0, r0 + blocker_h - 1, c0 + blocker_w - 1)
        elif side == "left":
            r0_lo = max(0, span_hi - blocker_h + 1)
            r0_hi = min(span_lo, h_lim - blocker_h)
            c1_lo = blocker_w - 1
            c1_hi = ec0 - MIN_RAY_STEPS - 1
            if r0_lo > r0_hi or c1_lo > c1_hi:
                return None
            r0 = random.randint(r0_lo, r0_hi)
            c1 = random.randint(c1_lo, c1_hi)
            bbox = (r0, c1 - blocker_w + 1, r0 + blocker_h - 1, c1)
        else:
            r0_lo = max(0, span_hi - blocker_h + 1)
            r0_hi = min(span_lo, h_lim - blocker_h)
            c0_lo = ec1 + MIN_RAY_STEPS + 1
            c0_hi = w_lim - blocker_w
            if r0_lo > r0_hi or c0_lo > c0_hi:
                return None
            r0 = random.randint(r0_lo, r0_hi)
            c0 = random.randint(c0_lo, c0_hi)
            bbox = (r0, c0, r0 + blocker_h - 1, c0 + blocker_w - 1)

        if _bbox_free(scene["forbidden"], bbox):
            return bbox
    return None


def _reserve_frame_on_masks(occupied: np.ndarray, forbidden: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    occupied[r0 : r1 + 1, c0 : c1 + 1] = True
    pr0 = max(0, r0 - FRAME_PADDING)
    pc0 = max(0, c0 - FRAME_PADDING)
    pr1 = min(occupied.shape[0] - 1, r1 + FRAME_PADDING)
    pc1 = min(occupied.shape[1] - 1, c1 + FRAME_PADDING)
    forbidden[pr0 : pr1 + 1, pc0 : pc1 + 1] = True


def _reserve_cells_on_masks(occupied: np.ndarray, forbidden: np.ndarray, cells: list[tuple[int, int]]) -> None:
    rows = np.array([r for r, _c in cells], dtype=int)
    cols = np.array([c for _r, c in cells], dtype=int)
    occupied[rows, cols] = True
    forbidden[rows, cols] = True


def _try_place_blocked_pair(scene: dict, emitter_color: int, blocker_color: int) -> tuple[dict, dict] | None:
    h_lim, w_lim = scene["occupied"].shape
    min_frame_h = FRAME_HEIGHT_RANGE[0]
    min_frame_w = FRAME_WIDTH_RANGE[0]

    for _ in range(BLOCKED_PAIR_TRIES):
        side = random.choice(SIDES)
        eh = random.randint(*FRAME_HEIGHT_RANGE)
        ew = random.randint(*FRAME_WIDTH_RANGE)
        gap_lo, gap_hi = _sample_gap(side, eh, ew)
        gap_width = gap_hi - gap_lo + 1

        if side == "top":
            bbox = _sample_bbox(
                scene,
                eh,
                ew,
                min_frame_h + MIN_RAY_STEPS,
                h_lim - eh,
                0,
                w_lim - ew,
            )
        elif side == "bottom":
            bbox = _sample_bbox(
                scene,
                eh,
                ew,
                0,
                h_lim - eh - min_frame_h - MIN_RAY_STEPS,
                0,
                w_lim - ew,
            )
        elif side == "left":
            bbox = _sample_bbox(
                scene,
                eh,
                ew,
                0,
                h_lim - eh,
                min_frame_w + MIN_RAY_STEPS,
                w_lim - ew,
            )
        else:
            bbox = _sample_bbox(
                scene,
                eh,
                ew,
                0,
                h_lim - eh,
                0,
                w_lim - ew - min_frame_w - MIN_RAY_STEPS,
            )
        if bbox is None:
            continue

        emitter = _frame_spec(emitter_color, bbox, side, gap_lo, gap_hi)
        occupied1 = scene["occupied"].copy()
        forbidden1 = scene["forbidden"].copy()
        _reserve_frame_on_masks(occupied1, forbidden1, bbox)

        blocker_dims = _random_blocker_dims(side, gap_width)
        if blocker_dims is None:
            continue
        blocker_h, blocker_w = blocker_dims
        blocker_scene = {"occupied": occupied1, "forbidden": forbidden1}
        blocker_bbox = _sample_blocker_bbox(blocker_scene, emitter, blocker_h, blocker_w)
        if blocker_bbox is None:
            continue

        beam_cells = _blocked_corridor(emitter, blocker_bbox)
        if beam_cells is None or not _cells_clear(occupied1, beam_cells):
            continue

        occupied2 = occupied1.copy()
        forbidden2 = forbidden1.copy()
        _reserve_frame_on_masks(occupied2, forbidden2, blocker_bbox)
        _reserve_cells_on_masks(occupied2, forbidden2, beam_cells)

        blocker_gap = None
        blocker_beam = None
        hit_side = HIT_SIDE[side]
        blocker_side_choices = [candidate for candidate in SIDES if candidate != hit_side]
        random.shuffle(blocker_side_choices)
        for blocker_side in blocker_side_choices:
            for _ in range(BLOCKER_GAP_TRIES):
                bgap_lo, bgap_hi = _sample_gap(blocker_side, blocker_h, blocker_w)
                candidate_beam = _edge_corridor(
                    scene["occupied"].shape,
                    blocker_bbox,
                    blocker_side,
                    bgap_lo,
                    bgap_hi,
                )
                if candidate_beam is None or not _cells_clear(occupied2, candidate_beam):
                    continue
                blocker_gap = (blocker_side, bgap_lo, bgap_hi)
                blocker_beam = candidate_beam
                break
            if blocker_gap is not None:
                break
        if blocker_gap is None or blocker_beam is None:
            continue

        blocker = _frame_spec(blocker_color, blocker_bbox, blocker_gap[0], blocker_gap[1], blocker_gap[2])
        _reserve_frame(scene, bbox)
        _reserve_frame(scene, blocker_bbox)
        _reserve_cells(scene, beam_cells)
        _reserve_cells(scene, blocker_beam)
        scene["frames"].extend([emitter, blocker])
        return emitter, blocker
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    frame_count = random.randint(*FRAME_COUNT_RANGE)
    mode = random.choice(("edge_only", "has_blocked"))
    colors = random.sample(range(1, 10), frame_count)

    scene = {
        "occupied": np.zeros((h, w), dtype=bool),
        "forbidden": np.zeros((h, w), dtype=bool),
        "frames": [],
    }

    color_index = 0
    if mode == "has_blocked":
        placed = _try_place_blocked_pair(scene, colors[0], colors[1])
        if placed is None:
            return None
        color_index = 2

    while color_index < frame_count:
        if _try_place_edge_frame(scene, colors[color_index]) is None:
            return None
        color_index += 1

    inp = _render_input((h, w), scene["frames"])
    expected = _render_output_from_specs(inp, scene["frames"])
    obstacle_mask = inp != BG
    reasons = [_beam_stop_reason(frame, obstacle_mask, inp.shape) for frame in scene["frames"]]

    if mode == "edge_only":
        if any(reason != "edge" for reason in reasons):
            return None
    else:
        if "frame" not in reasons or "edge" not in reasons:
            return None

    try:
        decoded = [_decode_frame(component) for component in _find_components(inp)]
        if len(decoded) != frame_count:
            return None
        actual = main(inp)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid frame-beam scene")
