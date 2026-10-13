from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
RED = 2
GREY = 5
PATTERN_COLORS = (1, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
GEN_ATTEMPTS = 400
GRID_MAX = 30


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as coordinate arrays."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _frame_phase(frame: np.ndarray) -> int:
    """Infer which border parity hosts the grey/red dashes."""
    fh, fw = frame.shape
    border = _frame_border_mask(fh, fw)
    rr, cc = np.indices((fh, fw))
    valid: list[int] = []
    for phase in (0, 1):
        active = border & (((rr + cc) & 1) == phase)
        inactive = border & ~active
        if not np.all(np.isin(frame[active], (GREY, RED))):
            continue
        if not np.all(frame[inactive] == BG):
            continue
        valid.append(phase)
    assert len(valid) == 1, "expected one valid dashed-frame phase"
    return int(valid[0])


def _side_has_marker(frame: np.ndarray, phase: int, side: str) -> bool:
    fh, fw = frame.shape
    if side == "top":
        coords = [(0, c) for c in range(1, fw - 1) if ((c & 1) == phase)]
    elif side == "bottom":
        parity = (phase - (fh - 1)) & 1
        coords = [(fh - 1, c) for c in range(1, fw - 1) if ((c & 1) == parity)]
    elif side == "left":
        coords = [(r, 0) for r in range(1, fh - 1) if ((r & 1) == phase)]
    else:
        parity = (phase - (fw - 1)) & 1
        coords = [(r, fw - 1) for r in range(1, fh - 1) if ((r & 1) == parity)]
    red_count = sum(int(frame[r, c] == RED) for r, c in coords)
    assert red_count <= 1, "each frame side may contain at most one red marker"
    return red_count == 1


def _stamp(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    coords = np.argwhere(mask)
    rows = top + coords[:, 0]
    cols = left + coords[:, 1]
    grid[rows, cols] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove dashed selection frames and copy each enclosed pattern outward
    according to the red markers on the frame sides.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    out = np.full((h, w), BG, dtype=int)
    pattern_colors = [int(color) for color in np.unique(g) if int(color) not in (BG, RED, GREY)]

    for color in pattern_colors:
        for coords in _components(g == color):
            r0, c0, r1, c1 = _bbox(coords)
            ph = r1 - r0 + 1
            pw = c1 - c0 + 1
            fr0, fc0 = r0 - 2, c0 - 2
            fr1, fc1 = r1 + 2, c1 + 2
            assert 0 <= fr0 <= fr1 < h and 0 <= fc0 <= fc1 < w, "pattern frame must fit in-grid"

            frame = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
            phase = _frame_phase(frame)

            allowed = np.zeros_like(frame, dtype=bool)
            allowed[2 : 2 + ph, 2 : 2 + pw] = True
            gap_region = frame.copy()
            gap_region[allowed] = BG
            border = _frame_border_mask(*frame.shape)
            gap_region[border] = BG
            assert np.all(gap_region == BG), "frame interior outside the pattern bbox must be empty"

            local_mask = np.zeros((ph, pw), dtype=bool)
            local_mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
            _stamp(out, local_mask, r0, c0, color)

            shifts = {
                "top": (-(ph + 2), 0),
                "bottom": (ph + 2, 0),
                "left": (0, -(pw + 2)),
                "right": (0, pw + 2),
            }
            for side in SIDES:
                if not _side_has_marker(frame, phase, side):
                    continue
                dr, dc = shifts[side]
                nr0 = r0 + dr
                nc0 = c0 + dc
                assert 0 <= nr0 and nr0 + ph <= h and 0 <= nc0 and nc0 + pw <= w, "copy must fit in-grid"
                _stamp(out, local_mask, nr0, nc0, color)

    return out


def _random_mask(height: int, width: int) -> np.ndarray:
    target_lo = max(4, int(np.ceil(height * width * 0.45)))
    target_hi = max(target_lo, int(np.ceil(height * width * 0.8)))
    for _ in range(200):
        target = random.randint(target_lo, target_hi)
        seed = (random.randrange(height), random.randrange(width))
        cells = {seed}
        while len(cells) < target:
            rr, cc = random.choice(tuple(cells))
            neighbors = []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    neighbors.append((nr, nc))
            if not neighbors:
                continue
            cells.add(random.choice(neighbors))

        mask = np.zeros((height, width), dtype=bool)
        for rr, cc in cells:
            mask[rr, cc] = True
        if mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any():
            return mask
    raise RuntimeError("failed to generate a full-bbox connected pattern")


def _sample_marker_sides() -> dict[str, bool]:
    return {side: bool(random.getrandbits(1)) for side in SIDES}


def _reserve_size(mask: np.ndarray, markers: dict[str, bool]) -> tuple[int, int, int, int, int, int]:
    ph, pw = mask.shape
    top_pad = ph + 2 if markers["top"] else 2
    bottom_pad = ph + 2 if markers["bottom"] else 2
    left_pad = pw + 2 if markers["left"] else 2
    right_pad = pw + 2 if markers["right"] else 2
    return top_pad, bottom_pad, left_pad, right_pad, top_pad + ph + bottom_pad, left_pad + pw + right_pad


def _build_input(grid: np.ndarray, obj: dict) -> None:
    mask = obj["mask"]
    color = obj["color"]
    r0 = obj["pattern_r0"]
    c0 = obj["pattern_c0"]
    ph, pw = mask.shape
    _stamp(grid, mask, r0, c0, color)

    fr0 = r0 - 2
    fc0 = c0 - 2
    fh = ph + 4
    fw = pw + 4
    phase = obj["phase"]
    rr, cc = np.indices((fh, fw))
    border = _frame_border_mask(fh, fw)
    active = border & (((rr + cc) & 1) == phase)
    grid[fr0 : fr0 + fh, fc0 : fc0 + fw][active] = GREY

    markers = obj["markers"]
    side_choices = {
        "top": [(0, c) for c in range(1, fw - 1) if ((c & 1) == phase)],
        "bottom": [(fh - 1, c) for c in range(1, fw - 1) if (((fh - 1 + c) & 1) == phase)],
        "left": [(r, 0) for r in range(1, fh - 1) if ((r & 1) == phase)],
        "right": [(r, fw - 1) for r in range(1, fh - 1) if (((r + fw - 1) & 1) == phase)],
    }
    for side, enabled in markers.items():
        if enabled:
            mr, mc = random.choice(side_choices[side])
            grid[fr0 + mr, fc0 + mc] = RED


def _build_output(grid: np.ndarray, obj: dict) -> bool:
    mask = obj["mask"]
    color = obj["color"]
    r0 = obj["pattern_r0"]
    c0 = obj["pattern_c0"]
    ph, pw = mask.shape

    coords = np.argwhere(mask)
    placements = [(r0, c0)]
    if obj["markers"]["top"]:
        placements.append((r0 - (ph + 2), c0))
    if obj["markers"]["bottom"]:
        placements.append((r0 + (ph + 2), c0))
    if obj["markers"]["left"]:
        placements.append((r0, c0 - (pw + 2)))
    if obj["markers"]["right"]:
        placements.append((r0, c0 + (pw + 2)))

    for top, left in placements:
        rows = top + coords[:, 0]
        cols = left + coords[:, 1]
        if (
            rows.min() < 0
            or rows.max() >= grid.shape[0]
            or cols.min() < 0
            or cols.max() >= grid.shape[1]
        ):
            return False
        if np.any(grid[rows, cols] != BG):
            return False
        grid[rows, cols] = int(color)
    return True


def _layout_objects(objs: list[dict]) -> tuple[int, int] | None:
    margins = [random.randint(1, 3) for _ in range(4)]
    top_margin, bottom_margin, left_margin, right_margin = margins
    if len(objs) == 1:
        obj = objs[0]
        height = top_margin + obj["reserve_h"] + bottom_margin
        width = left_margin + obj["reserve_w"] + right_margin
        if height > GRID_MAX or width > GRID_MAX:
            return None
        obj["reserve_r0"] = top_margin
        obj["reserve_c0"] = left_margin
        obj["pattern_r0"] = obj["reserve_r0"] + obj["top_pad"]
        obj["pattern_c0"] = obj["reserve_c0"] + obj["left_pad"]
        return height, width

    gap = random.randint(1, 3)
    row_h = max(obj["reserve_h"] for obj in objs)
    row_w = sum(obj["reserve_w"] for obj in objs) + gap
    col_h = sum(obj["reserve_h"] for obj in objs) + gap
    col_w = max(obj["reserve_w"] for obj in objs)
    layouts: list[str] = []
    if top_margin + row_h + bottom_margin <= GRID_MAX and left_margin + row_w + right_margin <= GRID_MAX:
        layouts.append("row")
    if top_margin + col_h + bottom_margin <= GRID_MAX and left_margin + col_w + right_margin <= GRID_MAX:
        layouts.append("col")
    if not layouts:
        return None

    choice = random.choice(layouts)
    if choice == "row":
        height = top_margin + row_h + bottom_margin
        width = left_margin + row_w + right_margin
        c = left_margin
        for obj in objs:
            slack = row_h - obj["reserve_h"]
            obj["reserve_r0"] = top_margin + (random.randint(0, slack) if slack else 0)
            obj["reserve_c0"] = c
            obj["pattern_r0"] = obj["reserve_r0"] + obj["top_pad"]
            obj["pattern_c0"] = obj["reserve_c0"] + obj["left_pad"]
            c += obj["reserve_w"] + gap
        return height, width

    height = top_margin + col_h + bottom_margin
    width = left_margin + col_w + right_margin
    r = top_margin
    for obj in objs:
        slack = col_w - obj["reserve_w"]
        obj["reserve_r0"] = r
        obj["reserve_c0"] = left_margin + (random.randint(0, slack) if slack else 0)
        obj["pattern_r0"] = obj["reserve_r0"] + obj["top_pad"]
        obj["pattern_c0"] = obj["reserve_c0"] + obj["left_pad"]
        r += obj["reserve_h"] + gap
    return height, width


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.choice((1, 2, 2))
    colors = random.sample(PATTERN_COLORS, n_objects)
    objs: list[dict] = []

    for idx, color in enumerate(colors):
        size_hi = 5 if n_objects == 1 else 4
        ph = random.randint(3, size_hi)
        pw = random.randint(3, size_hi)
        mask = _random_mask(ph, pw)
        markers = _sample_marker_sides()
        objs.append(
            {
                "color": int(color),
                "mask": mask,
                "markers": markers,
                "phase": random.randint(0, 1),
            }
        )

    if not any(obj["markers"][side] for obj in objs for side in SIDES):
        obj = random.choice(objs)
        obj["markers"][random.choice(SIDES)] = True

    random.shuffle(objs)
    for obj in objs:
        pads = _reserve_size(obj["mask"], obj["markers"])
        obj["top_pad"], obj["bottom_pad"], obj["left_pad"], obj["right_pad"], obj["reserve_h"], obj["reserve_w"] = pads

    layout = _layout_objects(objs)
    if layout is None:
        return None

    height, width = layout
    inp = np.full((height, width), BG, dtype=int)
    out = np.full((height, width), BG, dtype=int)

    for obj in objs:
        _build_input(inp, obj)
    for obj in objs:
        if not _build_output(out, obj):
            return None

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
