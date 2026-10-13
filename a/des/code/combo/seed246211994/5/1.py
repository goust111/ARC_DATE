from __future__ import annotations

import numpy as np


def _mode_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0]:
                continue
            color = int(g[r0, c0])
            if color == bg:
                seen[r0, c0] = True
                continue

            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            rmin = rmax = r0
            cmin = cmax = c0

            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                if r < rmin:
                    rmin = r
                if r > rmax:
                    rmax = r
                if c < cmin:
                    cmin = c
                if c > cmax:
                    cmax = c

                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc] and int(g[rr, cc]) == color:
                        seen[rr, cc] = True
                        stack.append((rr, cc))

            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": len(coords),
                    "bbox": (rmin, cmin, rmax, cmax),
                }
            )
    return comps


def _is_hollow_frame(grid: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    color = int(comp["color"])
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 4 or w < 4:
        return False

    expected_area = 2 * (h + w) - 4
    if int(comp["area"]) != expected_area:
        return False

    sub = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    if not bool(np.all(sub[border] == color)):
        return False
    if bool(np.any(sub[1:-1, 1:-1] == color)):
        return False
    return True


def _frame_payload(grid: np.ndarray, bg: int, frame: dict) -> dict[int, list[tuple[int, int]]]:
    r0, c0, r1, c1 = frame["bbox"]
    interior = np.asarray(grid[r0 + 1 : r1, c0 + 1 : c1], dtype=int)
    payload: dict[int, list[tuple[int, int]]] = {}
    for r in range(interior.shape[0]):
        for c in range(interior.shape[1]):
            color = int(interior[r, c])
            if color == bg:
                continue
            payload.setdefault(color, []).append((r, c))
    return payload


def _select_frame_group(grid: np.ndarray, bg: int) -> list[dict]:
    candidates = [comp for comp in _find_components(grid, bg) if _is_hollow_frame(grid, comp)]
    assert candidates, "expected at least one hollow rectangular frame"

    groups: dict[tuple[int, int], list[dict]] = {}
    for comp in candidates:
        r0, c0, r1, c1 = comp["bbox"]
        key = (int(r1 - r0 + 1), int(c1 - c0 + 1))
        groups.setdefault(key, []).append(comp)

    scored: list[tuple[tuple[int, int, int, int], list[dict]]] = []
    for (fh, fw), group in groups.items():
        if len(group) < 2:
            continue

        payloads = [_frame_payload(grid, bg, frame) for frame in group]
        full_palette = set()
        for payload in payloads:
            full_palette.update(payload.keys())
        if len(full_palette) < 2:
            continue

        missing_colors: list[int] = []
        ok = True
        for payload in payloads:
            missing = full_palette - set(payload)
            if len(missing) != 1:
                ok = False
                break
            missing_colors.append(int(next(iter(missing))))

        if not ok or len(set(missing_colors)) != len(missing_colors):
            continue

        for color in full_palette:
            coords = []
            for payload in payloads:
                if color not in payload:
                    continue
                if len(payload[color]) != 1:
                    ok = False
                    break
                coords.append(payload[color][0])
            if not ok or not coords or len(set(coords)) != 1:
                ok = False
                break

        if ok:
            score = (len(group), len(full_palette), fh * fw, -min(frame["bbox"][0] for frame in group))
            scored.append((score, group))

    assert scored, "expected a frame group with one distinct missing color per frame"
    best_group = max(scored, key=lambda item: item[0])[1]
    return sorted(best_group, key=lambda frame: (frame["bbox"][0], frame["bbox"][1]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare same-size hollow frames to identify which interior color is missing in
    each frame, then copy that color into the matching local coordinate.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _mode_color(g)
    frames = _select_frame_group(g, bg)
    payloads = [_frame_payload(g, bg, frame) for frame in frames]

    full_palette = set()
    for payload in payloads:
        full_palette.update(payload.keys())

    color_to_coord: dict[int, tuple[int, int]] = {}
    for color in full_palette:
        coords = []
        for payload in payloads:
            if color in payload:
                coords.append(payload[color][0])
        assert coords and len(set(coords)) == 1, "expected each color to have one shared local coordinate"
        color_to_coord[int(color)] = coords[0]

    out = g.copy()
    for frame, payload in zip(frames, payloads, strict=False):
        missing = full_palette - set(payload)
        assert len(missing) == 1, "expected exactly one missing color per frame"
        color = int(next(iter(missing)))
        rr, cc = color_to_coord[color]
        r0, c0, _r1, _c1 = frame["bbox"]
        assert int(out[r0 + 1 + rr, c0 + 1 + cc]) == bg, "missing color must land on background"
        out[r0 + 1 + rr, c0 + 1 + cc] = color

    return out


# --------------- generator ---------------

_WORK_SIZE = 30
_FRAME_COUNT_RANGE = (3, 6)  # [lo, hi)
_INTERIOR_H_RANGE = (4, 7)   # [lo, hi)
_INTERIOR_W_RANGE = (4, 7)   # [lo, hi)
_MAX_EXTRA_COLORS = 1
_PLACEMENT_BORDER = 1
_PLACEMENT_PADDING = 1
_MARGIN_CAP = 4
_MAX_ATTEMPTS = 400


def _sample_positions(rng: np.random.Generator, h: int, w: int, n: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(h) for c in range(w)]
    for min_gap in (2, 1, 0):
        for _ in range(200):
            chosen: list[tuple[int, int]] = []
            for idx in rng.permutation(len(cells)):
                cand = cells[int(idx)]
                if all(max(abs(cand[0] - r), abs(cand[1] - c)) > min_gap for r, c in chosen):
                    chosen.append(cand)
                    if len(chosen) == n:
                        return chosen
    picked = rng.choice(len(cells), size=n, replace=False)
    return [cells[int(i)] for i in np.atleast_1d(picked)]


def _make_frame_patch(
    interior_h: int,
    interior_w: int,
    bg: int,
    border_color: int,
    color_to_pos: dict[int, tuple[int, int]],
    missing_color: int | None,
) -> np.ndarray:
    patch = np.full((interior_h + 2, interior_w + 2), bg, dtype=int)
    patch[0, :] = border_color
    patch[-1, :] = border_color
    patch[:, 0] = border_color
    patch[:, -1] = border_color
    for color, (r, c) in color_to_pos.items():
        if missing_color is not None and int(color) == int(missing_color):
            continue
        patch[r + 1, c + 1] = int(color)
    return patch


def _place_patch(occ: np.ndarray, patch_shape: tuple[int, int], rng: np.random.Generator) -> tuple[int, int]:
    rows, cols = occ.shape
    ph, pw = patch_shape
    candidates = [
        (r, c)
        for r in range(_PLACEMENT_BORDER, rows - ph - _PLACEMENT_BORDER + 1)
        for c in range(_PLACEMENT_BORDER, cols - pw - _PLACEMENT_BORDER + 1)
    ]
    for idx in rng.permutation(len(candidates)):
        r0, c0 = candidates[int(idx)]
        rs = max(0, r0 - _PLACEMENT_PADDING)
        re = min(rows, r0 + ph + _PLACEMENT_PADDING)
        cs = max(0, c0 - _PLACEMENT_PADDING)
        ce = min(cols, c0 + pw + _PLACEMENT_PADDING)
        if bool(np.any(occ[rs:re, cs:ce])):
            continue
        occ[r0 : r0 + ph, c0 : c0 + pw] = True
        return int(r0), int(c0)
    raise ValueError("no free location for patch")


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    ph, pw = patch.shape
    dst[r0 : r0 + ph, c0 : c0 + pw] = patch


def _rand_margin(rng: np.random.Generator, available: int) -> int:
    limit = min(_MARGIN_CAP, int(available))
    assert limit >= 1
    return int(rng.integers(1, limit + 1))


def _generate_once(rng: np.random.Generator) -> np.ndarray | None:
    n_frames = int(rng.integers(*_FRAME_COUNT_RANGE))
    interior_h = int(rng.integers(*_INTERIOR_H_RANGE))
    interior_w = int(rng.integers(*_INTERIOR_W_RANGE))

    extra_limit = min(_MAX_EXTRA_COLORS, 8 - n_frames)
    add_extra = extra_limit > 0 and bool(rng.random() < 0.45)
    n_colors = n_frames + (1 if add_extra else 0)
    if interior_h * interior_w < n_colors:
        return None

    colors = list(map(int, rng.permutation(10)))
    bg = colors[0]
    fill_colors = colors[1 : 1 + n_colors]
    border_count = 1 + int(rng.random() < 0.5)
    border_colors = colors[1 + n_colors : 1 + n_colors + border_count]
    if len(border_colors) != border_count:
        return None

    positions = _sample_positions(rng, interior_h, interior_w, n_colors)
    color_to_pos = {int(color): pos for color, pos in zip(fill_colors, positions, strict=False)}

    missing_colors = fill_colors[:n_frames]
    rng.shuffle(missing_colors)

    frame_border_colors = [border_colors[i % border_count] for i in range(n_frames)]
    frame_border_colors = [frame_border_colors[int(i)] for i in rng.permutation(n_frames)]

    work_in = np.full((_WORK_SIZE, _WORK_SIZE), bg, dtype=int)
    work_out = np.full((_WORK_SIZE, _WORK_SIZE), bg, dtype=int)
    occ = np.zeros((_WORK_SIZE, _WORK_SIZE), dtype=bool)

    patches_in = []
    patches_out = []
    for border_color, missing_color in zip(frame_border_colors, missing_colors, strict=False):
        patches_in.append(
            _make_frame_patch(interior_h, interior_w, bg, int(border_color), color_to_pos, int(missing_color))
        )
        patches_out.append(
            _make_frame_patch(interior_h, interior_w, bg, int(border_color), color_to_pos, None)
        )

    for inp_patch, out_patch in zip(patches_in, patches_out, strict=False):
        try:
            r0, c0 = _place_patch(occ, inp_patch.shape, rng)
        except ValueError:
            return None
        _paste(work_in, inp_patch, r0, c0)
        _paste(work_out, out_patch, r0, c0)

    rows = np.where(occ.any(axis=1))[0]
    cols = np.where(occ.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return None

    top = _rand_margin(rng, int(rows[0]))
    bottom = _rand_margin(rng, int(_WORK_SIZE - 1 - rows[-1]))
    left = _rand_margin(rng, int(cols[0]))
    right = _rand_margin(rng, int(_WORK_SIZE - 1 - cols[-1]))

    r0 = int(rows[0] - top)
    r1 = int(rows[-1] + bottom)
    c0 = int(cols[0] - left)
    c1 = int(cols[-1] + right)
    inp = np.asarray(work_in[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    out = np.asarray(work_out[r0 : r1 + 1, c0 : c1 + 1], dtype=int)

    if _mode_color(inp) != bg:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_MAX_ATTEMPTS):
        grid = _generate_once(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_MAX_ATTEMPTS} attempts")
