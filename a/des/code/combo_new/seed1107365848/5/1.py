from __future__ import annotations

from collections import deque
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _component_patch(component: dict, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = component["bbox"]
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    patch[
        component["coords"][:, 0] - r0,
        component["coords"][:, 1] - c0,
    ] = int(component["color"])
    return patch


def _side_gap(present: np.ndarray) -> tuple[int, int] | None:
    missing = np.flatnonzero(~present)
    if missing.size == 0:
        return (-1, -1)
    if missing.size < 2:
        return None
    if int(missing[-1]) - int(missing[0]) + 1 != int(missing.size):
        return None
    if int(missing[0]) == 0 or int(missing[-1]) == len(present) - 1:
        return None
    return int(missing[0]), int(missing.size)


def _detect_frame(component: dict) -> dict | None:
    r0, c0, r1, c1 = component["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return None

    pts = {tuple(map(int, rc)) for rc in component["coords"]}
    for r, c in pts:
        if r not in (r0, r1) and c not in (c0, c1):
            return None

    top = np.array([(r0, c) in pts for c in range(c0, c1 + 1)], dtype=bool)
    bottom = np.array([(r1, c) in pts for c in range(c0, c1 + 1)], dtype=bool)
    left = np.array([(r, c0) in pts for r in range(r0, r1 + 1)], dtype=bool)
    right = np.array([(r, c1) in pts for r in range(r0, r1 + 1)], dtype=bool)

    gaps: list[tuple[str, int, int]] = []
    for side, present in (
        ("top", top),
        ("bottom", bottom),
        ("left", left),
        ("right", right),
    ):
        gap = _side_gap(present)
        if gap is None:
            return None
        if gap != (-1, -1):
            gaps.append((side, gap[0], gap[1]))
        elif not bool(present.all()):
            return None

    if len(gaps) != 1:
        return None

    side, gap_start, gap_width = gaps[0]
    expected_area = 2 * (h + w) - 4 - gap_width
    if int(component["area"]) != expected_area:
        return None

    return {
        "side": side,
        "gap_start": gap_start,
        "gap_width": gap_width,
    }


def _inside_frame(frame_bbox: tuple[int, int, int, int], component: dict) -> bool:
    r0, c0, r1, c1 = frame_bbox
    pr0, pc0, pr1, pc1 = component["bbox"]
    return r0 + 1 <= pr0 <= pr1 <= r1 - 1 and c0 + 1 <= pc0 <= pc1 <= c1 - 1


def _paste_patch(out: np.ndarray, patch: np.ndarray, dest_r: int, dest_c: int, bg: int) -> None:
    h, w = patch.shape
    assert 0 <= dest_r and 0 <= dest_c
    assert dest_r + h <= out.shape[0] and dest_c + w <= out.shape[1]
    mask = patch != bg
    target = out[dest_r : dest_r + h, dest_c : dest_c + w]
    assert not bool((mask & (target != bg)).any())
    target[mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror each inner pattern across the missing side and squeeze it just outside the frame gap.

    Input structure:
    - The background is black.
    - The grid contains several monochromatic rectangular frames of thickness 1, each with one contiguous gap of width at least 2 on exactly one side.
    - Every frame encloses one smaller 4-connected monochrome pattern of a different color.

    Rule:
    - Detect the side of each frame that contains the gap.
    - Extract the enclosed pattern, flip it across the axis perpendicular to that side (top/bottom -> vertical flip, left/right -> horizontal flip).
    - Place the flipped pattern outside the frame, immediately adjacent to the missing side, keeping its offset parallel to the gap aligned with the original pattern.
    - Leave the original inner pattern and every frame unchanged.

    Key insight:
    - The gap determines both the extrusion direction and which reflection to use, so the outside copy is not a plain translation.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _find_components(g, bg)

    frames = []
    for comp in comps:
        frame_info = _detect_frame(comp)
        if frame_info is not None:
            frames.append((comp, frame_info))

    assert frames
    frames.sort(key=lambda item: (item[0]["bbox"][0], item[0]["bbox"][1]))

    out = g.copy()
    for frame_comp, frame_info in frames:
        r0, c0, r1, c1 = frame_comp["bbox"]
        inner_patterns = [
            comp
            for comp in comps
            if comp is not frame_comp and _inside_frame(frame_comp["bbox"], comp)
        ]
        assert len(inner_patterns) == 1
        pattern = inner_patterns[0]
        patch = _component_patch(pattern, bg)
        pr0, pc0, pr1, pc1 = pattern["bbox"]

        side = frame_info["side"]
        if side == "top":
            flipped = np.flipud(patch)
            dest_r = r0 - patch.shape[0]
            dest_c = pc0
        elif side == "bottom":
            flipped = np.flipud(patch)
            dest_r = r1 + 1
            dest_c = pc0
        elif side == "left":
            flipped = np.fliplr(patch)
            dest_r = pr0
            dest_c = c0 - patch.shape[1]
        else:
            assert side == "right"
            flipped = np.fliplr(patch)
            dest_r = pr0
            dest_c = c1 + 1

        _paste_patch(out, flipped, dest_r, dest_c, bg)

    return out


# --------------- generator ---------------

_GEN_BG = 0
_PALETTE = tuple(range(1, 10))
_SIDES = ("top", "bottom", "left", "right")

_GRID_MIN = 18
_GRID_MAX = 30
_FRAME_COUNT_RANGE = (2, 4)
_FRAME_SIZE_MIN = 6
_FRAME_SIZE_MAX = 10
_PATTERN_SIZE_MIN = 2
_PATTERN_SIZE_MAX = 4

_OUTER_RETRIES = 400
_ENTITY_RETRIES = 120
_SHAPE_RETRIES = 200


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _reserve_rect(mask: np.ndarray, rect: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = rect
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _candidate_mask(shape: tuple[int, int], rects: list[tuple[int, int, int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for rect in rects:
        _reserve_rect(mask, rect)
    return mask


def _build_shape(ph: int, pw: int, side: str) -> np.ndarray | None:
    for _ in range(_SHAPE_RETRIES):
        max_area = ph * pw - 1
        if max_area < 3:
            return None
        area = random.randint(max(3, max(ph, pw)), max_area)
        start = (random.randrange(ph), random.randrange(pw))
        cells = {start}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c, ph, pw):
                    if (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        if len(cells) != area:
            continue

        mask = np.zeros((ph, pw), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if not bool(mask.any(axis=0).all()) or not bool(mask.any(axis=1).all()):
            continue
        if side in ("top", "bottom"):
            if bool(np.array_equal(mask, np.flipud(mask))):
                continue
        else:
            if bool(np.array_equal(mask, np.fliplr(mask))):
                continue
        return mask
    return None


def _make_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = color
    return patch


def _draw_frame(
    grid: np.ndarray,
    r0: int,
    c0: int,
    h: int,
    w: int,
    color: int,
    side: str,
    gap_start: int,
    gap_width: int,
) -> None:
    grid[r0, c0 : c0 + w] = color
    grid[r0 + h - 1, c0 : c0 + w] = color
    grid[r0 : r0 + h, c0] = color
    grid[r0 : r0 + h, c0 + w - 1] = color

    if side == "top":
        grid[r0, c0 + gap_start : c0 + gap_start + gap_width] = _GEN_BG
    elif side == "bottom":
        grid[r0 + h - 1, c0 + gap_start : c0 + gap_start + gap_width] = _GEN_BG
    elif side == "left":
        grid[r0 + gap_start : r0 + gap_start + gap_width, c0] = _GEN_BG
    else:
        grid[r0 + gap_start : r0 + gap_start + gap_width, c0 + w - 1] = _GEN_BG


def _paste_nonzero(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    mask = patch != _GEN_BG
    target = grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
    assert not bool((mask & (target != _GEN_BG)).any())
    target[mask] = patch[mask]


def _entity_attempt(
    grid_shape: tuple[int, int],
    reserved: np.ndarray,
    side: str,
    frame_color: int,
    pattern_color: int,
) -> dict | None:
    h_total, w_total = grid_shape

    frame_h = random.randint(_FRAME_SIZE_MIN, _FRAME_SIZE_MAX)
    frame_w = random.randint(_FRAME_SIZE_MIN, _FRAME_SIZE_MAX)
    inner_h = frame_h - 2
    inner_w = frame_w - 2
    max_ph = min(_PATTERN_SIZE_MAX, inner_h)
    max_pw = min(_PATTERN_SIZE_MAX, inner_w)
    if max_ph < _PATTERN_SIZE_MIN or max_pw < _PATTERN_SIZE_MIN:
        return None

    ph = random.randint(_PATTERN_SIZE_MIN, max_ph)
    pw = random.randint(_PATTERN_SIZE_MIN, max_pw)
    shape = _build_shape(ph, pw, side)
    if shape is None:
        return None

    off_r = random.randint(0, inner_h - ph)
    off_c = random.randint(0, inner_w - pw)

    side_len = frame_w if side in ("top", "bottom") else frame_h
    span_needed = pw if side in ("top", "bottom") else ph
    gap_min = max(2, span_needed)
    gap_max = side_len - 2
    if gap_min > gap_max:
        return None
    gap_width = random.randint(gap_min, gap_max)

    if side in ("top", "bottom"):
        pat_lo = 1 + off_c
        pat_hi = pat_lo + pw - 1
        gap_start_lo = max(1, pat_hi - gap_width + 1)
        gap_start_hi = min(pat_lo, frame_w - gap_width - 1)
    else:
        pat_lo = 1 + off_r
        pat_hi = pat_lo + ph - 1
        gap_start_lo = max(1, pat_hi - gap_width + 1)
        gap_start_hi = min(pat_lo, frame_h - gap_width - 1)
    if gap_start_lo > gap_start_hi:
        return None
    gap_start = random.randint(gap_start_lo, gap_start_hi)

    if side == "top":
        r0_lo, r0_hi = ph, h_total - frame_h
        c0_lo, c0_hi = 0, w_total - frame_w
    elif side == "bottom":
        r0_lo, r0_hi = 0, h_total - frame_h - ph
        c0_lo, c0_hi = 0, w_total - frame_w
    elif side == "left":
        r0_lo, r0_hi = 0, h_total - frame_h
        c0_lo, c0_hi = pw, w_total - frame_w
    else:
        r0_lo, r0_hi = 0, h_total - frame_h
        c0_lo, c0_hi = 0, w_total - frame_w - pw
    if r0_lo > r0_hi or c0_lo > c0_hi:
        return None

    patch = _make_patch(shape, pattern_color)
    flipped = np.flipud(patch) if side in ("top", "bottom") else np.fliplr(patch)

    for _ in range(_ENTITY_RETRIES):
        r0 = random.randint(r0_lo, r0_hi)
        c0 = random.randint(c0_lo, c0_hi)
        pr0 = r0 + 1 + off_r
        pc0 = c0 + 1 + off_c

        if side == "top":
            ext_r0, ext_c0 = r0 - ph, pc0
        elif side == "bottom":
            ext_r0, ext_c0 = r0 + frame_h, pc0
        elif side == "left":
            ext_r0, ext_c0 = pr0, c0 - pw
        else:
            ext_r0, ext_c0 = pr0, c0 + frame_w

        frame_rect = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)
        ext_rect = (ext_r0, ext_c0, ext_r0 + ph - 1, ext_c0 + pw - 1)
        cand = _candidate_mask(grid_shape, [frame_rect, ext_rect])
        if bool((cand & _dilate8(reserved)).any()):
            continue

        return {
            "frame_bbox": frame_rect,
            "frame_color": frame_color,
            "frame_h": frame_h,
            "frame_w": frame_w,
            "side": side,
            "gap_start": gap_start,
            "gap_width": gap_width,
            "pattern_patch": patch,
            "pattern_r0": pr0,
            "pattern_c0": pc0,
            "ext_r0": ext_r0,
            "ext_c0": ext_c0,
            "ext_patch": flipped,
            "reserve_mask": cand,
        }
    return None


def _build_pair() -> tuple[np.ndarray, np.ndarray] | None:
    n_frames = random.randint(*_FRAME_COUNT_RANGE)
    h = random.randint(_GRID_MIN, _GRID_MAX)
    w = random.randint(_GRID_MIN, _GRID_MAX)
    if 2 * n_frames > len(_PALETTE):
        return None

    colors = random.sample(_PALETTE, 2 * n_frames)
    sides = random.sample(_SIDES, k=min(n_frames, len(_SIDES)))
    if len(sides) < n_frames:
        sides.extend(random.choices(_SIDES, k=n_frames - len(sides)))
    random.shuffle(sides)

    reserved = np.zeros((h, w), dtype=bool)
    entities = []

    for idx in range(n_frames):
        entity = None
        for _ in range(_ENTITY_RETRIES):
            entity = _entity_attempt(
                (h, w),
                reserved,
                sides[idx],
                colors[2 * idx],
                colors[2 * idx + 1],
            )
            if entity is not None:
                break
        if entity is None:
            return None
        reserved |= entity["reserve_mask"]
        entities.append(entity)

    out = np.zeros((h, w), dtype=int)
    for entity in entities:
        r0, c0, r1, c1 = entity["frame_bbox"]
        _draw_frame(
            out,
            r0,
            c0,
            entity["frame_h"],
            entity["frame_w"],
            entity["frame_color"],
            entity["side"],
            entity["gap_start"],
            entity["gap_width"],
        )
        _paste_nonzero(out, entity["pattern_patch"], entity["pattern_r0"], entity["pattern_c0"])
        _paste_nonzero(out, entity["ext_patch"], entity["ext_r0"], entity["ext_c0"])

    inp = out.copy()
    for entity in entities:
        patch = entity["ext_patch"]
        rr = entity["ext_r0"]
        cc = entity["ext_c0"]
        mask = patch != _GEN_BG
        inp[rr : rr + patch.shape[0], cc : cc + patch.shape[1]][mask] = _GEN_BG

    if np.count_nonzero(inp == _GEN_BG) <= np.count_nonzero(inp != _GEN_BG):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(_OUTER_RETRIES):
        pair = _build_pair()
        if pair is not None:
            return pair[0]
    raise RuntimeError("failed to generate a valid input")
