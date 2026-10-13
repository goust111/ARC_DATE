from __future__ import annotations

import numpy as np


BG = 0
RAIL = 5
SLIDER = 9
MARKER_COLORS = np.array([1, 2, 3, 4, 6, 7, 8], dtype=int)

DIRS = ("R", "L", "D", "U")
KINDS = ("flat", "perp", "square")
MAX_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
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
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _shape_params(direction: str, kind: str, off_side: str | None) -> tuple[int, int, int]:
    if direction in ("R", "L"):
        if kind == "flat":
            return 1, 2, 0
        if kind == "perp":
            assert off_side in ("up", "down")
            return 2, 1, 1 if off_side == "up" else 0
        assert kind == "square"
        assert off_side in ("up", "down")
        return 2, 2, 1 if off_side == "up" else 0

    if kind == "flat":
        return 2, 1, 0
    if kind == "perp":
        assert off_side in ("left", "right")
        return 1, 2, 1 if off_side == "left" else 0
    assert kind == "square"
    assert off_side in ("left", "right")
    return 2, 2, 1 if off_side == "left" else 0


def _slider_patch(
    height: int,
    width: int,
    orientation: str,
    direction: str,
    rail_local: int,
    absorb: int,
    border: int | None,
) -> np.ndarray:
    patch = np.full((height, width), absorb, dtype=int)
    if border is None:
        return patch

    patch[:, :] = border

    if (height, width) == (2, 2):
        if orientation == "H":
            patch[rail_local, :] = absorb
        else:
            patch[:, rail_local] = absorb
        return patch

    if (height, width) == (1, 2):
        if orientation == "H":
            front_c = 1 if direction == "R" else 0
            patch[0, front_c] = absorb
        else:
            patch[0, rail_local] = absorb
        return patch

    if (height, width) == (2, 1):
        if orientation == "H":
            patch[rail_local, 0] = absorb
        else:
            front_r = 1 if direction == "D" else 0
            patch[front_r, 0] = absorb
        return patch

    raise AssertionError("unexpected slider shape")


def _detect_slider(
    grid: np.ndarray, coords: np.ndarray, bg: int, slider_color: int
) -> tuple[str, int, tuple[int, int, int, int]]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    assert coords.shape[0] == height * width
    assert (height, width) in {(1, 2), (2, 1), (2, 2)}

    side_hits: dict[str, list[int]] = {}
    if c1 + 1 < grid.shape[1]:
        hits = [r for r in range(r0, r1 + 1) if int(grid[r, c1 + 1]) not in (bg, slider_color)]
        if hits:
            side_hits["R"] = hits
    if c0 - 1 >= 0:
        hits = [r for r in range(r0, r1 + 1) if int(grid[r, c0 - 1]) not in (bg, slider_color)]
        if hits:
            side_hits["L"] = hits
    if r1 + 1 < grid.shape[0]:
        hits = [c for c in range(c0, c1 + 1) if int(grid[r1 + 1, c]) not in (bg, slider_color)]
        if hits:
            side_hits["D"] = hits
    if r0 - 1 >= 0:
        hits = [c for c in range(c0, c1 + 1) if int(grid[r0 - 1, c]) not in (bg, slider_color)]
        if hits:
            side_hits["U"] = hits

    assert len(side_hits) == 1
    direction, hits = next(iter(side_hits.items()))
    assert len(set(int(v) for v in hits)) == 1
    rail_abs = int(hits[0])
    return direction, rail_abs, (r0, c0, r1, c1)


def _scan_track(
    grid: np.ndarray,
    direction: str,
    rail_abs: int,
    bbox: tuple[int, int, int, int],
) -> tuple[list[tuple[int, int]], list[int]]:
    r0, c0, r1, c1 = bbox
    cells: list[tuple[int, int]] = []
    colors: list[int] = []

    if direction == "R":
        c = c1 + 1
        while c < grid.shape[1] and int(grid[rail_abs, c]) != BG and int(grid[rail_abs, c]) != SLIDER:
            cells.append((rail_abs, c))
            colors.append(int(grid[rail_abs, c]))
            c += 1
        return cells, colors

    if direction == "L":
        c = c0 - 1
        while c >= 0 and int(grid[rail_abs, c]) != BG and int(grid[rail_abs, c]) != SLIDER:
            cells.append((rail_abs, c))
            colors.append(int(grid[rail_abs, c]))
            c -= 1
        return cells, colors

    if direction == "D":
        r = r1 + 1
        while r < grid.shape[0] and int(grid[r, rail_abs]) != BG and int(grid[r, rail_abs]) != SLIDER:
            cells.append((r, rail_abs))
            colors.append(int(grid[r, rail_abs]))
            r += 1
        return cells, colors

    r = r0 - 1
    while r >= 0 and int(grid[r, rail_abs]) != BG and int(grid[r, rail_abs]) != SLIDER:
        cells.append((r, rail_abs))
        colors.append(int(grid[r, rail_abs]))
        r -= 1
    return cells, colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each maroon block along its grey rail, stop at the first marker, and absorb its color.

    With two markers, the first marker is the fill color and the second becomes the accent color.
    For 2x2 sliders, the accent occupies the strip away from the rail so the two-color state stays visible.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = _most_common_color(g)

    for comp in _components(g == SLIDER):
        direction, rail_abs, bbox = _detect_slider(g, comp, bg, SLIDER)
        r0, c0, r1, c1 = bbox
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        orientation = "H" if direction in ("R", "L") else "V"

        out[comp[:, 0], comp[:, 1]] = bg

        if orientation == "H":
            rail_local = rail_abs - r0
            out[rail_abs, c0 : c1 + 1] = RAIL
        else:
            rail_local = rail_abs - c0
            out[r0 : r1 + 1, rail_abs] = RAIL

        track_cells, track_colors = _scan_track(g, direction, rail_abs, bbox)
        assert track_cells

        markers = [
            (idx, color)
            for idx, color in enumerate(track_colors)
            if int(color) not in (bg, RAIL, SLIDER)
        ]
        assert len(markers) in (1, 2)

        for rr, cc in track_cells:
            out[rr, cc] = RAIL

        stop_idx, absorb = markers[0]
        border = markers[1][1] if len(markers) == 2 else None
        stop_cell = track_cells[int(stop_idx)]

        if direction == "R":
            dr, dc = 0, int(stop_cell[1] - c1)
        elif direction == "L":
            dr, dc = 0, int(stop_cell[1] - c0)
        elif direction == "D":
            dr, dc = int(stop_cell[0] - r1), 0
        else:
            dr, dc = int(stop_cell[0] - r0), 0

        patch = _slider_patch(height, width, orientation, direction, rail_local, int(absorb), border)
        nr0, nc0 = r0 + dr, c0 + dc
        out[nr0 : nr0 + height, nc0 : nc0 + width] = patch

    return out


def _make_spec(rng: np.random.Generator) -> dict[str, object]:
    direction = str(rng.choice(DIRS))
    kind = str(rng.choice(KINDS))
    if direction in ("R", "L"):
        off_side = None if kind == "flat" else str(rng.choice(["up", "down"]))
    else:
        off_side = None if kind == "flat" else str(rng.choice(["left", "right"]))

    height, width, rail_local = _shape_params(direction, kind, off_side)
    along = width if direction in ("R", "L") else height
    marker_count = 1 + int(rng.integers(0, 2))
    line_len = int(rng.integers(max(6, along + marker_count + 1), 14))
    marker_indices = sorted(
        int(v) for v in rng.choice(np.arange(along, line_len), size=marker_count, replace=False)
    )
    marker_colors = [int(v) for v in rng.choice(MARKER_COLORS, size=marker_count, replace=False)]
    return {
        "direction": direction,
        "kind": kind,
        "off_side": off_side,
        "height": height,
        "width": width,
        "rail_local": rail_local,
        "line_len": line_len,
        "marker_indices": marker_indices,
        "marker_colors": marker_colors,
    }


def _build_patch(spec: dict[str, object]) -> np.ndarray:
    direction = str(spec["direction"])
    height = int(spec["height"])
    width = int(spec["width"])
    rail_local = int(spec["rail_local"])
    line_len = int(spec["line_len"])
    marker_indices = [int(v) for v in spec["marker_indices"]]
    marker_colors = [int(v) for v in spec["marker_colors"]]

    if direction in ("R", "L"):
        patch = np.full((height, line_len), BG, dtype=int)
        patch[rail_local, :] = RAIL

        def axis_to_grid(idx: int) -> int:
            return idx if direction == "R" else line_len - 1 - idx

        for idx, color in zip(marker_indices, marker_colors):
            patch[rail_local, axis_to_grid(int(idx))] = int(color)

        if direction == "R":
            patch[:, :width] = SLIDER
        else:
            patch[:, line_len - width :] = SLIDER
        return patch

    patch = np.full((line_len, width), BG, dtype=int)
    patch[:, rail_local] = RAIL

    def axis_to_grid(idx: int) -> int:
        return idx if direction == "D" else line_len - 1 - idx

    for idx, color in zip(marker_indices, marker_colors):
        patch[axis_to_grid(int(idx)), rail_local] = int(color)

    if direction == "D":
        patch[:height, :] = SLIDER
    else:
        patch[line_len - height :, :] = SLIDER
    return patch


def _place_patches(rng: np.random.Generator, patches: list[np.ndarray]) -> np.ndarray | None:
    n = len(patches)
    base = 12 + 3 * n
    h = int(rng.integers(base, base + 9))
    w = int(rng.integers(base, base + 9))
    grid = np.full((h, w), BG, dtype=int)
    used = np.zeros((h, w), dtype=bool)
    padding = 2

    order = list(rng.permutation(n))
    for idx in order:
        patch = patches[int(idx)]
        ph, pw = patch.shape
        if ph > h or pw > w:
            return None
        placed = False
        for _ in range(120):
            r0 = int(rng.integers(0, h - ph + 1))
            c0 = int(rng.integers(0, w - pw + 1))
            rr0 = max(0, r0 - padding)
            rr1 = min(h, r0 + ph + padding)
            cc0 = max(0, c0 - padding)
            cc1 = min(w, c0 + pw + padding)
            if used[rr0:rr1, cc0:cc1].any():
                continue
            mask = patch != BG
            grid[r0 : r0 + ph, c0 : c0 + pw][mask] = patch[mask]
            used[r0 : r0 + ph, c0 : c0 + pw] = True
            placed = True
            break
        if not placed:
            return None

    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_objects = int(rng.integers(1, 5))
    specs = [_make_spec(rng) for _ in range(n_objects)]
    patches = [_build_patch(spec) for spec in specs]
    grid = _place_patches(rng, patches)
    if grid is None:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if not isinstance(out, np.ndarray) or out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
