from __future__ import annotations

import numpy as np


BG = 0
FRAME_COLOR = 3
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
SIDES = ("top", "bottom", "left", "right")


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(flags.tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 1))
    return runs


def _paste_clipped(out: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = out.shape
    ph, pw = patch.shape
    r0 = max(0, int(top))
    c0 = max(0, int(left))
    r1 = min(h, int(top) + ph)
    c1 = min(w, int(left) + pw)
    if r0 >= r1 or c0 >= c1:
        return

    sub = patch[r0 - int(top) : r1 - int(top), c0 - int(left) : c1 - int(left)]
    mask = sub != int(bg)
    if not np.any(mask):
        return
    target = out[r0:r1, c0:c1]
    target[mask] = sub[mask]


def _emit_strip(out: np.ndarray, patch: np.ndarray, bg: int, top: int, left: int, step_r: int, step_c: int) -> None:
    ph, pw = patch.shape
    h, w = out.shape
    r = int(top)
    c = int(left)
    while r < h and r + ph > 0 and c < w and c + pw > 0:
        _paste_clipped(out, patch, r, c, bg)
        r += int(step_r)
        c += int(step_c)


def _frame_candidates(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color == int(bg):
            continue
        for coords in _components(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            if h < 4 or w < 4:
                continue

            rs = coords[:, 0]
            cs = coords[:, 1]
            on_border = (rs == r0) | (rs == r1) | (cs == c0) | (cs == c1)
            if not np.all(on_border):
                continue
            if not (np.any(rs == r0) and np.any(rs == r1) and np.any(cs == c0) and np.any(cs == c1)):
                continue

            inner = grid[r0 + 1 : r1, c0 + 1 : c1]
            if inner.size == 0 or not np.any((inner != int(bg)) & (inner != color)):
                continue

            gap_count = 0
            gap_count += len(_runs(grid[r0, c0 + 1 : c1] == bg))
            gap_count += len(_runs(grid[r1, c0 + 1 : c1] == bg))
            gap_count += len(_runs(grid[r0 + 1 : r1, c0] == bg))
            gap_count += len(_runs(grid[r0 + 1 : r1, c1] == bg))
            if gap_count == 0:
                continue

            frames.append({"color": color, "bbox": (r0, c0, r1, c1)})
    frames.sort(key=lambda item: item["bbox"])
    return frames


def _template_patch(grid: np.ndarray, frame: dict, bg: int) -> np.ndarray:
    frame_color = int(frame["color"])
    r0, c0, r1, c1 = map(int, frame["bbox"])
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    mask = (inner != int(bg)) & (inner != frame_color)
    coords = np.argwhere(mask)
    assert coords.size > 0, "frame interior must contain a template"
    tr0, tc0, tr1, tc1 = _bbox(coords)
    patch = inner[tr0 : tr1 + 1, tc0 : tc1 + 1].copy()
    patch[patch == frame_color] = int(bg)
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the interior template outward through each frame-edge gap as a repeated strip.

    Input structure:
    - Background is black.
    - One or more green hollow rectangular frames have one or more border gaps.
    - Each frame contains a small template drawn in a non-green, non-black color.

    Rule:
    - Each gap emits copies of the full interior template outward in the gap's direction.
    - The first copy sits immediately outside the gap and later copies continue with no spacing.
    - Horizontal gaps use the template width as the strip width; vertical gaps use the template height.
    - Frames and original templates remain unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = BG if np.any(g == BG) else int(np.bincount(g.ravel()).argmax())
    out = g.copy()
    frames = _frame_candidates(g, bg)
    assert frames, "expected at least one gapped rectangular frame"

    for frame in frames:
        frame_color = int(frame["color"])
        r0, c0, r1, c1 = map(int, frame["bbox"])
        patch = _template_patch(g, frame, bg)
        ph, pw = patch.shape

        top_runs = _runs(g[r0, c0 + 1 : c1] == bg)
        bottom_runs = _runs(g[r1, c0 + 1 : c1] == bg)
        left_runs = _runs(g[r0 + 1 : r1, c0] == bg)
        right_runs = _runs(g[r0 + 1 : r1, c1] == bg)

        for start, end in top_runs:
            if end - start + 1 != pw:
                continue
            left = c0 + 1 + start
            _emit_strip(out, patch, bg, r0 - ph, left, -ph, 0)

        for start, end in bottom_runs:
            if end - start + 1 != pw:
                continue
            left = c0 + 1 + start
            _emit_strip(out, patch, bg, r1 + 1, left, ph, 0)

        for start, end in left_runs:
            if end - start + 1 != ph:
                continue
            top = r0 + 1 + start
            _emit_strip(out, patch, bg, top, c0 - pw, 0, -pw)

        for start, end in right_runs:
            if end - start + 1 != ph:
                continue
            top = r0 + 1 + start
            _emit_strip(out, patch, bg, top, c1 + 1, 0, pw)

        assert np.all(out[r0:r1 + 1, c0:c1 + 1][g[r0:r1 + 1, c0:c1 + 1] == frame_color] == frame_color)
        assert frame_color != int(bg)

    return out


def _random_connected_mask(rng: np.random.Generator, h: int, w: int, area: int) -> np.ndarray:
    cells = {(int(rng.integers(0, h)), int(rng.integers(0, w)))}
    while len(cells) < int(area):
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        pick = frontier[int(rng.integers(0, len(frontier)))]
        cells.add(pick)

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _random_template_patch(rng: np.random.Generator, color: int, bg: int) -> np.ndarray:
    for _ in range(100):
        h = int(rng.integers(2, 5))
        w = int(rng.integers(2, 5))
        lo = max(2, (h * w + 2) // 3)
        area = int(rng.integers(lo, h * w + 1))
        mask = _random_connected_mask(rng, h, w, area)
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        patch = np.full((h, w), int(bg), dtype=int)
        patch[mask] = int(color)
        return patch
    raise RuntimeError("failed to sample a valid template patch")


def _gap_start_choices(line_len: int, seg_len: int, count: int) -> list[tuple[int, ...]]:
    choices: list[tuple[int, ...]] = []

    def backtrack(next_start: int, remaining: int, prefix: list[int]) -> None:
        if remaining == 0:
            choices.append(tuple(prefix))
            return
        max_start = line_len - seg_len
        for start in range(next_start, max_start + 1):
            prefix.append(start)
            backtrack(start + seg_len + 1, remaining - 1, prefix)
            prefix.pop()

    backtrack(0, int(count), [])
    return choices


def _frame_patch(
    inner_h: int,
    inner_w: int,
    template: np.ndarray,
    template_top: int,
    template_left: int,
    side: str,
    gap_starts: tuple[int, ...],
    frame_color: int,
    bg: int,
) -> np.ndarray:
    patch = np.full((inner_h + 2, inner_w + 2), int(bg), dtype=int)
    patch[0, :] = int(frame_color)
    patch[-1, :] = int(frame_color)
    patch[:, 0] = int(frame_color)
    patch[:, -1] = int(frame_color)

    ph, pw = template.shape
    if side == "top":
        for start in gap_starts:
            patch[0, 1 + start : 1 + start + pw] = int(bg)
    elif side == "bottom":
        for start in gap_starts:
            patch[-1, 1 + start : 1 + start + pw] = int(bg)
    elif side == "left":
        for start in gap_starts:
            patch[1 + start : 1 + start + ph, 0] = int(bg)
    else:
        for start in gap_starts:
            patch[1 + start : 1 + start + ph, -1] = int(bg)

    patch[1 + template_top : 1 + template_top + ph, 1 + template_left : 1 + template_left + pw] = np.where(
        template != int(bg), template, patch[1 + template_top : 1 + template_top + ph, 1 + template_left : 1 + template_left + pw]
    )
    return patch


def _strip_rectangles(
    shape: tuple[int, int],
    frame_top: int,
    frame_left: int,
    frame_h: int,
    frame_w: int,
    template_h: int,
    template_w: int,
    side: str,
    gap_starts: tuple[int, ...],
) -> list[tuple[int, int, int, int]]:
    h, w = map(int, shape)
    rects = [(int(frame_top), int(frame_left), int(frame_top + frame_h - 1), int(frame_left + frame_w - 1))]

    if side == "top":
        for start in gap_starts:
            rr = int(frame_top - template_h)
            cc = int(frame_left + 1 + start)
            while rr < h and rr + template_h > 0:
                rects.append((max(0, rr), cc, min(h - 1, rr + template_h - 1), cc + template_w - 1))
                rr -= template_h
    elif side == "bottom":
        for start in gap_starts:
            rr = int(frame_top + frame_h)
            cc = int(frame_left + 1 + start)
            while rr < h and rr + template_h > 0:
                rects.append((rr, cc, min(h - 1, rr + template_h - 1), cc + template_w - 1))
                rr += template_h
    elif side == "left":
        for start in gap_starts:
            rr = int(frame_top + 1 + start)
            cc = int(frame_left - template_w)
            while cc < w and cc + template_w > 0:
                rects.append((rr, max(0, cc), rr + template_h - 1, min(w - 1, cc + template_w - 1)))
                cc -= template_w
    else:
        for start in gap_starts:
            rr = int(frame_top + 1 + start)
            cc = int(frame_left + frame_w)
            while cc < w and cc + template_w > 0:
                rects.append((rr, cc, rr + template_h - 1, min(w - 1, cc + template_w - 1)))
                cc += template_w

    return rects


def _mark_rectangles(mask: np.ndarray, rects: list[tuple[int, int, int, int]]) -> None:
    for r0, c0, r1, c1 in rects:
        mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = True


def _sample_frame_spec(
    rng: np.random.Generator, shape: tuple[int, int], template_color: int
) -> dict | None:
    h, w = map(int, shape)
    for _ in range(200):
        side = SIDES[int(rng.integers(0, len(SIDES)))]
        template = _random_template_patch(rng, template_color, BG)
        ph, pw = template.shape
        seg_len = pw if side in ("top", "bottom") else ph
        gap_count = int(rng.choice(np.array([1, 2, 3]), p=np.array([0.4, 0.35, 0.25])))

        if side in ("top", "bottom"):
            inner_w = int(gap_count * pw + (gap_count - 1) + rng.integers(0, 4))
            inner_h = int(ph + rng.integers(0, 4))
        else:
            inner_h = int(gap_count * ph + (gap_count - 1) + rng.integers(0, 4))
            inner_w = int(pw + rng.integers(0, 4))

        frame_h = inner_h + 2
        frame_w = inner_w + 2
        if frame_h > h or frame_w > w:
            continue
        if side in ("top", "bottom") and frame_h + ph > h:
            continue
        if side in ("left", "right") and frame_w + pw > w:
            continue

        line_len = inner_w if side in ("top", "bottom") else inner_h
        choices = _gap_start_choices(line_len, seg_len, gap_count)
        if not choices:
            continue
        gap_starts = choices[int(rng.integers(0, len(choices)))]
        template_top = int(rng.integers(0, inner_h - ph + 1))
        template_left = int(rng.integers(0, inner_w - pw + 1))
        patch = _frame_patch(
            inner_h,
            inner_w,
            template,
            template_top,
            template_left,
            side,
            gap_starts,
            FRAME_COLOR,
            BG,
        )
        return {
            "side": side,
            "template": template,
            "patch": patch,
            "gap_starts": gap_starts,
        }
    return None


def _position_choices(
    shape: tuple[int, int],
    patch: np.ndarray,
    template: np.ndarray,
    side: str,
    gap_starts: tuple[int, ...],
    reserved: np.ndarray,
    input_blocked: np.ndarray,
) -> list[tuple[int, int, list[tuple[int, int, int, int]]]]:
    h, w = map(int, shape)
    frame_h, frame_w = patch.shape
    template_h, template_w = template.shape

    if side == "top":
        row_range = range(template_h, h - frame_h + 1)
        col_range = range(0, w - frame_w + 1)
    elif side == "bottom":
        row_range = range(0, h - frame_h - template_h + 1)
        col_range = range(0, w - frame_w + 1)
    elif side == "left":
        row_range = range(0, h - frame_h + 1)
        col_range = range(template_w, w - frame_w + 1)
    else:
        row_range = range(0, h - frame_h + 1)
        col_range = range(0, w - frame_w - template_w + 1)

    positions: list[tuple[int, int, list[tuple[int, int, int, int]]]] = []
    for r0 in row_range:
        for c0 in col_range:
            pr0 = max(0, r0 - 1)
            pc0 = max(0, c0 - 1)
            pr1 = min(h, r0 + frame_h + 1)
            pc1 = min(w, c0 + frame_w + 1)
            if np.any(input_blocked[pr0:pr1, pc0:pc1]):
                continue

            rects = _strip_rectangles(shape, r0, c0, frame_h, frame_w, template_h, template_w, side, gap_starts)
            ok = True
            for rr0, cc0, rr1, cc1 in rects:
                if np.any(reserved[rr0 : rr1 + 1, cc0 : cc1 + 1]):
                    ok = False
                    break
            if ok:
                positions.append((r0, c0, rects))
    return positions


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    frame_count = int(rng.integers(1, 4))
    h = int(rng.integers(18 + 2 * (frame_count - 1), 29))
    w = int(rng.integers(18 + 2 * (frame_count - 1), 29))

    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)
    input_blocked = np.zeros((h, w), dtype=bool)
    colors = rng.choice(np.array([1, 2, 4, 5, 6, 7, 8, 9]), size=frame_count, replace=False)

    specs: list[tuple[int, dict]] = []
    for color in colors.tolist():
        spec = _sample_frame_spec(rng, grid.shape, int(color))
        if spec is None:
            return None
        specs.append((int(color), spec))

    rng.shuffle(specs)
    for color, spec in specs:
        patch = spec["patch"]
        template = spec["template"]
        side = spec["side"]
        gap_starts = spec["gap_starts"]

        positions = _position_choices(grid.shape, patch, template, side, gap_starts, reserved, input_blocked)
        if not positions:
            return None
        pick = positions[int(rng.integers(0, len(positions)))]
        r0, c0, rects = pick

        frame_view = grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
        mask = patch != BG
        frame_view[mask] = patch[mask]

        _mark_rectangles(reserved, rects)
        br0 = max(0, r0 - 1)
        bc0 = max(0, c0 - 1)
        br1 = min(h, r0 + patch.shape[0] + 1)
        bc1 = min(w, c0 + patch.shape[1] + 1)
        input_blocked[br0:br1, bc0:bc1] = True

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if not np.any((out != grid) & (grid == BG)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
