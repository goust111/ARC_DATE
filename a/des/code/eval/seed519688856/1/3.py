from __future__ import annotations

from collections import deque

import numpy as np


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _local_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0, r1, c1)


def _parse_frame(coords: np.ndarray) -> dict | None:
    mask, bbox = _local_mask(coords)
    h, w = mask.shape
    if h < 5 or w < 5:
        return None
    rr, cc = np.nonzero(mask)
    if np.any((rr != 0) & (rr != h - 1) & (cc != 0) & (cc != w - 1)):
        return None
    if not (mask[0, 0] and mask[0, -1] and mask[-1, 0] and mask[-1, -1]):
        return None

    sides = {
        "top": np.flatnonzero(~mask[0, 1:-1]) + 1,
        "bottom": np.flatnonzero(~mask[-1, 1:-1]) + 1,
        "left": np.flatnonzero(~mask[1:-1, 0]) + 1,
        "right": np.flatnonzero(~mask[1:-1, -1]) + 1,
    }
    gaps: list[tuple[str, int, int]] = []
    for side, miss in sides.items():
        if miss.size == 0:
            continue
        if not np.array_equal(miss, np.arange(int(miss[0]), int(miss[-1]) + 1)):
            return None
        gaps.append((side, int(miss[0]), int(miss[-1])))
    if len(gaps) != 1:
        return None

    side, gap0, gap1 = gaps[0]
    return {
        "bbox": bbox,
        "local_mask": mask,
        "side": side,
        "gap0": gap0,
        "gap1": gap1,
        "area": int(h * w),
        "shape": (int(h), int(w)),
    }


def _piece_local_offset(frame: dict, piece_mask: np.ndarray) -> tuple[int, int]:
    h, w = frame["shape"]
    ph, pw = piece_mask.shape
    gap0 = int(frame["gap0"])
    gap1 = int(frame["gap1"])
    side = frame["side"]
    gap_center = 0.5 * (gap0 + gap1)
    if side in ("top", "bottom"):
        left = int(round(gap_center - 0.5 * (pw - 1)))
        top = int(h - 1 - ph) if side == "top" else 1
    else:
        top = int(round(gap_center - 0.5 * (ph - 1)))
        left = int(w - 1 - pw) if side == "left" else 1
    return int(top), int(left)


def _frame_patch(frame: dict, bg: int, include_piece: bool) -> np.ndarray:
    h, w = frame["shape"]
    color = int(frame["color"])
    patch = np.full((h, w), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    gap0 = int(frame["gap0"])
    gap1 = int(frame["gap1"])
    side = frame["side"]
    if side == "top":
        patch[0, gap0 : gap1 + 1] = bg
    elif side == "bottom":
        patch[-1, gap0 : gap1 + 1] = bg
    elif side == "left":
        patch[gap0 : gap1 + 1, 0] = bg
    else:
        patch[gap0 : gap1 + 1, -1] = bg

    if include_piece:
        top, left = frame["piece_offset"]
        piece_mask = frame["piece_mask"]
        ph, pw = piece_mask.shape
        view = patch[top : top + ph, left : left + pw]
        view[piece_mask] = color
        patch[top : top + ph, left : left + pw] = view
    return patch


def _render_scene(frames: list[dict], shape: tuple[int, int], bg: int, include_piece: bool) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for frame in sorted(frames, key=lambda item: int(item["area"]), reverse=True):
        r0, c0, r1, c1 = frame["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = _frame_patch(frame, bg, include_piece=include_piece)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each colored fragment to the same-colored gapped frame, translate it
    inward from the gap side, and composite overlapping frames with smaller
    rectangles on top.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    colors = [int(v) for v in np.unique(g) if int(v) != bg]

    frames: list[dict] = []
    for color in colors:
        comps = _components(g == color)
        assert len(comps) == 2, "each color must have one frame and one fragment"

        parsed = [_parse_frame(comp) for comp in comps]
        frame_idx = next((i for i, info in enumerate(parsed) if info is not None), None)
        assert frame_idx is not None, "failed to identify frame component"

        frame = dict(parsed[frame_idx])
        frame["color"] = int(color)
        piece_coords = comps[1 - frame_idx]
        piece_mask, _piece_bbox = _local_mask(piece_coords)
        frame["piece_mask"] = piece_mask
        frame["piece_offset"] = _piece_local_offset(frame, piece_mask)
        frames.append(frame)

    return _render_scene(frames, g.shape, bg, include_piece=True)


# --------------- generator ---------------

BG = 8
COLOR_POOL = [1, 2, 3, 4, 6, 7, 9]
DIRECTIONS = ["top", "bottom", "left", "right"]

GRID_H_RANGE = (22, 31)
GRID_W_RANGE = (22, 31)
N_FRAMES_CHOICES = [2, 3]
OVERLAP_PROB = 0.5
FRAME_GAP = 1
PIECE_GAP = 1
MAX_GEN_TRIES = 400
POLY_TRIES = 80


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _make_piece_mask(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    area_lo = max(h + w - 1, int(np.ceil(0.45 * h * w)))
    area_hi = max(area_lo, int(np.floor(0.72 * h * w)))
    if area_hi >= h * w:
        area_hi = h * w - 1
    if area_hi < area_lo:
        area_hi = area_lo

    for _ in range(POLY_TRIES):
        target = int(rng.integers(area_lo, area_hi + 1))
        start = (int(rng.integers(0, h)), int(rng.integers(0, w)))
        cells = {start}
        while len(cells) < target:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = frontier[int(rng.integers(0, len(frontier)))]
            cells.add((nr, nc))

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if not mask.any():
            continue
        if not (
            mask[0, :].any()
            and mask[-1, :].any()
            and mask[:, 0].any()
            and mask[:, -1].any()
        ):
            continue
        return mask
    raise RuntimeError("failed to build fragment shape")


def _sample_frame_spec(
    rng: np.random.Generator,
    color: int,
    h_range: tuple[int, int],
    w_range: tuple[int, int],
) -> dict:
    while True:
        side = DIRECTIONS[int(rng.integers(0, len(DIRECTIONS)))]
        h = int(rng.integers(h_range[0], h_range[1] + 1))
        w = int(rng.integers(w_range[0], w_range[1] + 1))
        axis_len = w if side in ("top", "bottom") else h
        gap_hi = min(5, axis_len - 3)
        if gap_hi < 3:
            continue
        gap_len = int(rng.integers(3, gap_hi + 1))
        gap0 = int(rng.integers(1, axis_len - gap_len))
        gap1 = int(gap0 + gap_len - 1)
        if side in ("top", "bottom"):
            ph = int(rng.integers(2, min(gap_len, h - 3) + 1))
            pw = gap_len
        else:
            ph = gap_len
            pw = int(rng.integers(2, min(gap_len, w - 3) + 1))
        piece_mask = _make_piece_mask(rng, ph, pw)
        frame = {
            "color": int(color),
            "side": side,
            "gap0": int(gap0),
            "gap1": int(gap1),
            "shape": (int(h), int(w)),
            "area": int(h * w),
            "piece_mask": piece_mask,
        }
        frame["piece_offset"] = _piece_local_offset(frame, piece_mask)
        return frame


def _attach_bbox(frame: dict, top: int, left: int) -> dict:
    out = dict(frame)
    h, w = out["shape"]
    out["bbox"] = (int(top), int(left), int(top + h - 1), int(left + w - 1))
    return out


def _absolute_piece_mask(frame: dict, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    r0, c0, _r1, _c1 = frame["bbox"]
    pr, pc = frame["piece_offset"]
    piece = frame["piece_mask"]
    ph, pw = piece.shape
    mask[r0 + pr : r0 + pr + ph, c0 + pc : c0 + pc + pw] = piece
    return mask


def _free_positions(
    shape: tuple[int, int],
    box_shape: tuple[int, int],
    occupied: list[tuple[int, int, int, int]],
    pad: int,
) -> list[tuple[int, int]]:
    h, w = shape
    bh, bw = box_shape
    out: list[tuple[int, int]] = []
    for r in range(h - bh + 1):
        for c in range(w - bw + 1):
            cand = (r, c, r + bh - 1, c + bw - 1)
            if any(_boxes_overlap(cand, box, pad=pad) for box in occupied):
                continue
            out.append((int(r), int(c)))
    return out


def _place_nonoverlap_frames(
    rng: np.random.Generator,
    shape: tuple[int, int],
    specs: list[dict],
) -> list[dict] | None:
    placed: list[dict] = []
    occupied: list[tuple[int, int, int, int]] = []
    for spec in specs:
        positions = _free_positions(shape, spec["shape"], occupied, pad=FRAME_GAP)
        if not positions:
            return None
        top, left = positions[int(rng.integers(0, len(positions)))]
        frame = _attach_bbox(spec, top, left)
        placed.append(frame)
        occupied.append(frame["bbox"])
    return placed


def _place_overlap_frames(
    rng: np.random.Generator,
    shape: tuple[int, int],
    big_spec: dict,
    small_spec: dict,
    extras: list[dict],
) -> list[dict] | None:
    positions = _free_positions(shape, big_spec["shape"], [], pad=0)
    rng.shuffle(positions)
    for top, left in positions:
        big = _attach_bbox(big_spec, top, left)
        big_piece = _absolute_piece_mask(big, shape)
        piece_area = int(big_piece.sum())

        br0, bc0, br1, bc1 = big["bbox"]
        sh, sw = small_spec["shape"]
        small_positions: list[tuple[int, int]] = []
        for sr in range(br0 + 1, br1 - sh + 1):
            for sc in range(bc0 + 1, bc1 - sw + 1):
                small_box = (sr, sc, sr + sh - 1, sc + sw - 1)
                overlap = int(big_piece[sr : sr + sh, sc : sc + sw].sum())
                if overlap <= 0 or overlap >= piece_area:
                    continue
                small_positions.append((int(sr), int(sc)))
        if not small_positions:
            continue

        sr, sc = small_positions[int(rng.integers(0, len(small_positions)))]
        small = _attach_bbox(small_spec, sr, sc)

        placed = [big, small]
        occupied = [big["bbox"]]
        ok = True
        for spec in extras:
            extra_positions = _free_positions(shape, spec["shape"], occupied, pad=FRAME_GAP)
            if not extra_positions:
                ok = False
                break
            er, ec = extra_positions[int(rng.integers(0, len(extra_positions)))]
            frame = _attach_bbox(spec, er, ec)
            placed.append(frame)
            occupied.append(frame["bbox"])
        if ok:
            return placed
    return None


def _place_source_pieces(rng: np.random.Generator, shape: tuple[int, int], frames: list[dict]) -> bool:
    h, w = shape
    forbidden = np.zeros((h, w), dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        forbidden[max(0, r0 - PIECE_GAP) : min(h, r1 + PIECE_GAP + 1), max(0, c0 - PIECE_GAP) : min(w, c1 + PIECE_GAP + 1)] = True

    order = list(rng.permutation(len(frames)))
    for idx in order:
        piece = frames[idx]["piece_mask"]
        ph, pw = piece.shape
        positions: list[tuple[int, int]] = []
        for r in range(h - ph + 1):
            for c in range(w - pw + 1):
                if forbidden[r : r + ph, c : c + pw].any():
                    continue
                positions.append((int(r), int(c)))
        if not positions:
            return False
        r, c = positions[int(rng.integers(0, len(positions)))]
        frames[idx]["source_pos"] = (int(r), int(c))
        forbidden[max(0, r - PIECE_GAP) : min(h, r + ph + PIECE_GAP), max(0, c - PIECE_GAP) : min(w, c + pw + PIECE_GAP)] = True
    return True


def _render_input(frames: list[dict], shape: tuple[int, int]) -> np.ndarray:
    inp = _render_scene(frames, shape, BG, include_piece=False)
    for frame in frames:
        r, c = frame["source_pos"]
        piece = frame["piece_mask"]
        ph, pw = piece.shape
        view = inp[r : r + ph, c : c + pw]
        view[piece] = int(frame["color"])
        inp[r : r + ph, c : c + pw] = view
    return inp


def _sample_specs(rng: np.random.Generator, n_frames: int, with_overlap: bool) -> tuple[list[dict], bool]:
    colors = [int(v) for v in rng.choice(COLOR_POOL, size=n_frames, replace=False)]
    if with_overlap:
        big = _sample_frame_spec(rng, colors[0], (10, 13), (10, 13))
        small = _sample_frame_spec(rng, colors[1], (6, 8), (6, 8))
        if small["shape"][0] > big["shape"][0] - 2 or small["shape"][1] > big["shape"][1] - 2:
            return [], False
        specs = [big, small]
        for color in colors[2:]:
            specs.append(_sample_frame_spec(rng, color, (6, 10), (6, 10)))
    else:
        specs = [_sample_frame_spec(rng, color, (6, 10), (6, 10)) for color in colors]
    areas = [int(spec["area"]) for spec in specs]
    if len(set(areas)) != len(areas):
        return [], False
    return specs, True


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    shape = (
        int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1])),
        int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1])),
    )
    n_frames = int(rng.choice(N_FRAMES_CHOICES))
    with_overlap = bool(rng.random() < OVERLAP_PROB)
    specs, ok = _sample_specs(rng, n_frames, with_overlap)
    if not ok:
        return None

    if with_overlap:
        frames = _place_overlap_frames(rng, shape, specs[0], specs[1], specs[2:])
    else:
        frames = _place_nonoverlap_frames(rng, shape, specs)
    if frames is None:
        return None

    if not _place_source_pieces(rng, shape, frames):
        return None

    inp = _render_input(frames, shape)
    out = _render_scene(frames, shape, BG, include_piece=True)
    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GEN_TRIES} attempts")
