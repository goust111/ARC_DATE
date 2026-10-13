from __future__ import annotations

import numpy as np


SCALE_COLOR = 5
BEAM_LENGTHS = (5, 7)
PAN_WIDTHS_BY_BEAM = {5: (3,), 7: (3, 5)}
PAN_HEIGHTS = (3, 4, 5)
HANGER_HEIGHTS = (1, 2)
PILLAR_HEIGHTS = (2, 3)

N_SCALES_RANGE = (2, 4)  # inclusive
GRID_BASE_MIN = 18
GRID_GROWTH_PER_EXTRA_SCALE = 3
GRID_EXTRA_RANGE = 8
PLACEMENT_PADDING = 2
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _extract_pan(mask: np.ndarray, beam_row: int, endpoint_col: int, max_row: int) -> tuple[int, int, int, int] | None:
    r = beam_row + 1
    while r <= max_row and mask[r, endpoint_col]:
        r += 1
    pan_top = r - 1
    if pan_top <= beam_row:
        return None

    left = endpoint_col
    while left - 1 >= 0 and mask[pan_top, left - 1]:
        left -= 1
    right = endpoint_col
    while right + 1 < mask.shape[1] and mask[pan_top, right + 1]:
        right += 1
    if right - left + 1 < 3:
        return None

    for bottom in range(pan_top + 2, max_row + 1):
        if not np.all(mask[bottom, left : right + 1]):
            continue
        ok = True
        for rr in range(pan_top + 1, bottom):
            if not mask[rr, left] or not mask[rr, right]:
                ok = False
                break
        if ok:
            return int(pan_top), int(left), int(bottom), int(right)
    return None


def _parse_scale(mask: np.ndarray, color: int) -> dict | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None

    beam_row = int(coords[:, 0].min())
    max_row = int(coords[:, 0].max())
    top_cols = np.where(mask[beam_row])[0]
    if top_cols.size < 5 or top_cols.size > 7:
        return None
    if int(top_cols[-1] - top_cols[0] + 1) != int(top_cols.size):
        return None

    beam_c0 = int(top_cols[0])
    beam_c1 = int(top_cols[-1])
    center_c = int(top_cols[top_cols.size // 2])

    pillar_bottom = beam_row
    while pillar_bottom + 1 < mask.shape[0] and mask[pillar_bottom + 1, center_c]:
        pillar_bottom += 1
    pillar_height = int(pillar_bottom - beam_row)
    if pillar_height not in PILLAR_HEIGHTS:
        return None

    left_pan = _extract_pan(mask, beam_row, beam_c0, max_row)
    right_pan = _extract_pan(mask, beam_row, beam_c1, max_row)
    if left_pan is None or right_pan is None:
        return None

    if (left_pan[2] - left_pan[0], left_pan[3] - left_pan[1]) != (right_pan[2] - right_pan[0], right_pan[3] - right_pan[1]):
        return None
    if left_pan[0] != right_pan[0]:
        return None
    if left_pan[2] != right_pan[2]:
        return None
    if left_pan[3] >= right_pan[1]:
        return None

    left_hanger = int(left_pan[0] - beam_row - 1)
    right_hanger = int(right_pan[0] - beam_row - 1)
    if left_hanger not in HANGER_HEIGHTS or right_hanger not in HANGER_HEIGHTS:
        return None
    if left_hanger != right_hanger:
        return None

    if not (left_pan[1] < beam_c0 < left_pan[3] and right_pan[1] < beam_c1 < right_pan[3]):
        return None

    return {
        "color": int(color),
        "beam_row": beam_row,
        "beam_c0": beam_c0,
        "beam_c1": beam_c1,
        "center_c": center_c,
        "pillar_height": pillar_height,
        "left_pan": left_pan,
        "right_pan": right_pan,
    }


def _payload_count(grid: np.ndarray, pan: tuple[int, int, int, int], bg: int, frame_color: int) -> int:
    r0, c0, r1, c1 = pan
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    return int(np.count_nonzero((interior != bg) & (interior != frame_color)))


def _core_line_rows(r0: int, c0: int, r1: int, c1: int) -> list[int]:
    if c0 == c1:
        return [int(r0)]
    rows: list[int] = []
    span = c1 - c0
    for dc in range(span + 1):
        frac = (r1 - r0) * dc / span
        rows.append(int(np.floor(r0 + frac + 0.5)))
    return rows


def _draw_tilted_beam(out: np.ndarray, left_row: int, c0: int, right_row: int, c1: int, color: int) -> list[int]:
    rows = _core_line_rows(left_row, c0, right_row, c1)
    out[rows[0], c0] = color
    prev_row = rows[0]
    for offset, row in enumerate(rows[1:], start=1):
        c = c0 + offset
        lo = min(prev_row, row)
        hi = max(prev_row, row)
        out[lo : hi + 1, c] = color
        prev_row = row
    return rows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tilt each weighing scale according to which pan contains more foreground pixels.

    Input structure:
    - Background is the dominant color.
    - Each scale is one grey connected component made from a horizontal beam,
      a central pillar, and two hanging rectangular pans of the same size.
    - Colored pixels inside the pans are the weights.

    Rule:
    - Count the non-background pixels inside the left and right pan interiors.
    - If one side is heavier, shift that pan down by one row and shift the
      other pan up by one row.
    - Tilt the beam so the heavy endpoint is one row lower and the light
      endpoint is one row higher.
    - If both sides are equal, leave the scale unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    scales: list[dict] = []
    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        for comp in _components(g == color):
            mask = np.zeros_like(g, dtype=bool)
            rr, cc = zip(*comp)
            mask[rr, cc] = True
            info = _parse_scale(mask, color)
            if info is not None:
                scales.append(info)

    scales.sort(key=lambda item: (item["beam_row"], item["beam_c0"]))

    for info in scales:
        frame_color = int(info["color"])
        left_pan = tuple(int(v) for v in info["left_pan"])
        right_pan = tuple(int(v) for v in info["right_pan"])
        left_count = _payload_count(g, left_pan, bg, frame_color)
        right_count = _payload_count(g, right_pan, bg, frame_color)
        tilt = int(np.sign(left_count - right_count))
        if tilt == 0:
            continue

        beam_row = int(info["beam_row"])
        beam_c0 = int(info["beam_c0"])
        beam_c1 = int(info["beam_c1"])
        center_c = int(info["center_c"])
        pillar_height = int(info["pillar_height"])

        clear_r0 = min(beam_row - 1, left_pan[0] - 1, right_pan[0] - 1)
        clear_r1 = max(beam_row + pillar_height, left_pan[2] + 1, right_pan[2] + 1)
        clear_c0 = left_pan[1]
        clear_c1 = right_pan[3]
        out[clear_r0 : clear_r1 + 1, clear_c0 : clear_c1 + 1] = bg

        left_patch = g[left_pan[0] : left_pan[2] + 1, left_pan[1] : left_pan[3] + 1].copy()
        right_patch = g[right_pan[0] : right_pan[2] + 1, right_pan[1] : right_pan[3] + 1].copy()

        left_new_r0 = left_pan[0] + tilt
        right_new_r0 = right_pan[0] - tilt
        out[left_new_r0 : left_new_r0 + left_patch.shape[0], left_pan[1] : left_pan[3] + 1] = left_patch
        out[right_new_r0 : right_new_r0 + right_patch.shape[0], right_pan[1] : right_pan[3] + 1] = right_patch

        beam_rows = _draw_tilted_beam(out, beam_row + tilt, beam_c0, beam_row - tilt, beam_c1, frame_color)
        center_row = int(beam_rows[center_c - beam_c0])
        out[center_row + 1 : center_row + pillar_height + 1, center_c] = frame_color

        left_hanger_bottom = left_new_r0 - 1
        if beam_row + tilt + 1 <= left_hanger_bottom:
            out[beam_row + tilt + 1 : left_hanger_bottom + 1, beam_c0] = frame_color

        right_hanger_bottom = right_new_r0 - 1
        if beam_row - tilt + 1 <= right_hanger_bottom:
            out[beam_row - tilt + 1 : right_hanger_bottom + 1, beam_c1] = frame_color

    return out


def _sample_branch_list(rng: np.random.Generator, n_scales: int) -> list[int]:
    base = [-1, 0, 1]
    if n_scales == 2:
        pairs = [(-1, 0), (0, 1), (-1, 1)]
        return [int(v) for v in pairs[int(rng.integers(0, len(pairs)))]]

    branches = list(base)
    while len(branches) < n_scales:
        branches.append(int(base[int(rng.integers(0, len(base)))]))
    rng.shuffle(branches)
    return [int(v) for v in branches]


def _sample_counts(rng: np.random.Generator, interior_area: int, branch: int) -> tuple[int, int]:
    if branch == 0:
        count = int(rng.integers(1, interior_area + 1))
        return count, count

    light = int(rng.integers(0, interior_area))
    heavy = int(rng.integers(light + 1, interior_area + 1))
    if branch > 0:
        return heavy, light
    return light, heavy


def _paint_objects(
    rng: np.random.Generator,
    patch: np.ndarray,
    pan: tuple[int, int, int, int],
    count: int,
    bg: int,
) -> None:
    if count <= 0:
        return

    r0, c0, r1, c1 = pan
    cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    choose = rng.choice(len(cells), size=count, replace=False)
    palette = np.array([c for c in range(10) if c not in (bg, SCALE_COLOR)], dtype=int)
    n_colors = int(rng.integers(1, min(4, len(palette)) + 1))
    colors = palette[rng.choice(len(palette), size=n_colors, replace=False)]
    for idx in np.atleast_1d(choose):
        rr, cc = cells[int(idx)]
        patch[rr, cc] = int(colors[int(rng.integers(0, len(colors)))])


def _make_scale_patch(rng: np.random.Generator, bg: int, branch: int) -> np.ndarray:
    beam_len = int(BEAM_LENGTHS[int(rng.integers(0, len(BEAM_LENGTHS)))])
    pan_w_choices = PAN_WIDTHS_BY_BEAM[beam_len]
    pan_w = int(pan_w_choices[int(rng.integers(0, len(pan_w_choices)))])
    pan_h = int(PAN_HEIGHTS[int(rng.integers(0, len(PAN_HEIGHTS)))])
    hanger_h = int(HANGER_HEIGHTS[int(rng.integers(0, len(HANGER_HEIGHTS)))])
    pillar_h = int(PILLAR_HEIGHTS[int(rng.integers(0, len(PILLAR_HEIGHTS)))])

    beam_row = 1
    beam_c0 = pan_w // 2
    beam_c1 = beam_c0 + beam_len - 1
    pan_top = beam_row + hanger_h + 1
    pan_bottom = pan_top + pan_h - 1
    patch_h = pan_bottom + 2
    patch_w = beam_c1 + (pan_w - 1 - pan_w // 2) + 1
    patch = np.full((patch_h, patch_w), bg, dtype=int)

    patch[beam_row, beam_c0 : beam_c1 + 1] = SCALE_COLOR
    center_c = (beam_c0 + beam_c1) // 2
    patch[beam_row + 1 : beam_row + pillar_h + 1, center_c] = SCALE_COLOR

    left_pan = (pan_top, 0, pan_bottom, pan_w - 1)
    right_pan = (pan_top, patch_w - pan_w, pan_bottom, patch_w - 1)

    patch[beam_row + 1 : pan_top, beam_c0] = SCALE_COLOR
    patch[beam_row + 1 : pan_top, beam_c1] = SCALE_COLOR

    for r0, c0, r1, c1 in (left_pan, right_pan):
        patch[r0, c0 : c1 + 1] = SCALE_COLOR
        patch[r1, c0 : c1 + 1] = SCALE_COLOR
        patch[r0 : r1 + 1, c0] = SCALE_COLOR
        patch[r0 : r1 + 1, c1] = SCALE_COLOR

    interior_area = (pan_h - 2) * (pan_w - 2)
    left_count, right_count = _sample_counts(rng, interior_area, branch)
    _paint_objects(rng, patch, left_pan, left_count, bg)
    _paint_objects(rng, patch, right_pan, right_count, bg)
    return patch


def _non_overlapping_positions(
    rng: np.random.Generator,
    rects: list[tuple[int, int]],
    grid_h: int,
    grid_w: int,
    padding: int,
) -> list[tuple[int, int]] | None:
    placed: list[tuple[int, int, int, int]] = []
    origins: list[tuple[int, int]] = []
    order = list(range(len(rects)))
    rng.shuffle(order)
    origins_by_idx: dict[int, tuple[int, int]] = {}

    for idx in order:
        ph, pw = rects[idx]
        candidates: list[tuple[int, int]] = []
        for r0 in range(grid_h - ph + 1):
            for c0 in range(grid_w - pw + 1):
                r1 = r0 + ph - 1
                c1 = c0 + pw - 1
                ok = True
                for or0, oc0, or1, oc1 in placed:
                    if not (r1 + padding < or0 or or1 + padding < r0 or c1 + padding < oc0 or oc1 + padding < c0):
                        ok = False
                        break
                if ok:
                    candidates.append((r0, c0))
        if not candidates:
            return None
        choice = candidates[int(rng.integers(0, len(candidates)))]
        r0, c0 = choice
        origins_by_idx[idx] = choice
        placed.append((r0, c0, r0 + ph - 1, c0 + pw - 1))

    for idx in range(len(rects)):
        origins.append(origins_by_idx[idx])
    return origins


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    bg_choices = np.array([c for c in range(10) if c != SCALE_COLOR], dtype=int)
    bg = int(bg_choices[int(rng.integers(0, len(bg_choices)))])

    n_scales = int(rng.integers(N_SCALES_RANGE[0], N_SCALES_RANGE[1] + 1))
    branches = _sample_branch_list(rng, n_scales)
    patches = [_make_scale_patch(rng, bg, branch) for branch in branches]

    min_side = GRID_BASE_MIN + GRID_GROWTH_PER_EXTRA_SCALE * (n_scales - N_SCALES_RANGE[0])
    grid_h = int(rng.integers(min_side, min_side + GRID_EXTRA_RANGE + 1))
    grid_w = int(rng.integers(min_side, min_side + GRID_EXTRA_RANGE + 1))
    rects = [(int(p.shape[0]), int(p.shape[1])) for p in patches]
    origins = _non_overlapping_positions(rng, rects, grid_h, grid_w, PLACEMENT_PADDING)
    if origins is None:
        return None

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    for patch, (r0, c0) in zip(patches, origins):
        ph, pw = patch.shape
        target = grid[r0 : r0 + ph, c0 : c0 + pw]
        target[patch != bg] = patch[patch != bg]

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    if _get_bg(grid) != bg:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
