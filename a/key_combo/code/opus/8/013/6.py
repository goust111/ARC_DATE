import sys
from pathlib import Path

import numpy as np

try:
    _UTIL_ROOT = Path("/home/lixinyu/snap/new")
    if str(_UTIL_ROOT) not in sys.path:
        sys.path.append(str(_UTIL_ROOT))
    from solvers.arc_utils import Color, find_components, get_bg
except Exception:  # pragma: no cover - fallback for direct loading
    class Color:
        BLACK = 0
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5
        TEAL = 8

    def get_bg(grid: np.ndarray) -> int:
        values, counts = np.unique(grid, return_counts=True)
        return int(values[np.argmax(counts)])

    def find_components(
        grid: np.ndarray,
        *,
        bg: int | None = None,
        connectivity: int = 4,
        monochromatic: bool = True,
        colors=None,
        min_area: int = 1,
        return_format: str = "dict",
    ):
        if bg is None:
            bg = get_bg(grid)
        if colors is None:
            palette = [int(c) for c in np.unique(grid) if int(c) != int(bg)]
        else:
            palette = [int(c) for c in colors if int(c) != int(bg)]
        out = []
        visited = np.zeros(grid.shape, dtype=bool)
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)] if connectivity == 4 else [
            (dr, dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr, dc) != (0, 0)
        ]
        for color in palette:
            mask = grid == color
            visited[:] = False
            for r, c in np.argwhere(mask):
                r = int(r)
                c = int(c)
                if visited[r, c]:
                    continue
                stack = [(r, c)]
                coords = []
                visited[r, c] = True
                while stack:
                    rr, cc = stack.pop()
                    coords.append((rr, cc))
                    for dr, dc in dirs:
                        nr = rr + dr
                        nc = cc + dc
                        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and not visited[nr, nc] and mask[nr, nc]:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                if len(coords) < min_area:
                    continue
                arr = np.array(coords, dtype=int)
                r0, c0 = arr.min(axis=0)
                r1, c1 = arr.max(axis=0)
                out.append(
                    {
                        "color": color,
                        "coords": arr,
                        "bbox": (int(r0), int(c0), int(r1), int(c1)),
                        "area": int(len(coords)),
                    }
                )
        return out


GREY_NOISE_FREE_COLORS = (1, 6, 7, 9)
GEN_RETRIES = 400


def _detect_rails(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    rails = []
    for comp in find_components(
        np.asarray(grid, dtype=int),
        bg=get_bg(grid),
        connectivity=4,
        monochromatic=True,
        colors=[Color.GREY],
        return_format="dict",
    ):
        r0, c0, r1, c1 = comp["bbox"]
        if c0 != c1:
            continue
        if comp["area"] != r1 - r0 + 1:
            continue
        if comp["area"] < 4:
            continue
        rails.append((int(r0), int(c0), int(r1), int(c1)))
    rails.sort(key=lambda b: (b[1], b[0]))
    assert len(rails) == 2, "expected exactly two grey vertical rails"
    return rails


def _crop_bounds_from_rails(grid: np.ndarray) -> tuple[int, int, int, int]:
    left, right = _detect_rails(grid)
    r0 = max(0, min(left[0], right[0]) - 1)
    r1 = min(grid.shape[0] - 1, max(left[2], right[2]) + 1)
    c0 = min(left[1], right[1])
    c1 = max(left[1], right[1])
    return int(r0), int(c0), int(r1), int(c1)


def _mirror_orbit(h: int, w: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    assert h % 2 == 1 and w % 2 == 1, "crop dimensions must be odd"
    cr = h // 2
    cc = w // 2
    pts = {
        (int(r), int(c)),
        (int(2 * cr - r), int(c)),
        (int(r), int(2 * cc - c)),
        (int(2 * cr - r), int(2 * cc - c)),
    }
    for rr, cc2 in pts:
        assert 0 <= rr < h and 0 <= cc2 < w
    return tuple(sorted(pts))


def _repair_crop_holes(crop: np.ndarray) -> np.ndarray:
    out = np.array(crop, dtype=int, copy=True)
    h, w = out.shape
    while np.any(out == Color.YELLOW):
        progress = False
        seen = set()
        for r, c in np.argwhere(out == Color.YELLOW):
            orbit = _mirror_orbit(h, w, int(r), int(c))
            if orbit in seen:
                continue
            seen.add(orbit)
            known = {int(out[rr, cc]) for rr, cc in orbit if int(out[rr, cc]) != Color.YELLOW}
            assert known, "every yellow orbit must have at least one determined color"
            assert known <= {Color.BLACK, Color.RED}, "yellow holes may only stand for black or red"
            if len(known) != 1:
                raise AssertionError("yellow orbit must resolve to a unique mirrored color")
            fill = known.pop()
            for rr, cc in orbit:
                if int(out[rr, cc]) == Color.YELLOW:
                    out[rr, cc] = fill
                    progress = True
        if not progress:
            raise AssertionError("failed to resolve all yellow holes")
    return out


def _fill_between_red_pairs(crop: np.ndarray) -> np.ndarray:
    out = np.array(crop, dtype=int, copy=True)
    for r in range(out.shape[0]):
        cols = np.flatnonzero(out[r] == Color.RED)
        for i in range(0, len(cols) - 1, 2):
            left = int(cols[i])
            right = int(cols[i + 1])
            if right - left <= 1:
                continue
            segment = out[r, left + 1 : right]
            out[r, left + 1 : right][segment == Color.BLACK] = Color.GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop between the grey rails, repair marker-color holes by local biaxial symmetry, then fill the closed red outlines row by row.

    Input structure:
    - The background is black, with exactly two grey vertical rails in different columns.
    - The crop rows are one cell above and below the rail endpoints, and 2-4 teal corner markers may appear on those expanded rows.
    - Inside that cropped rectangle, a red outline pattern is symmetric about the crop center, but some cells on symmetric orbits are replaced by yellow hole markers.
    - A few unrelated noise pixels may appear outside the crop.

    Rule:
    - Detect the two grey rails and crop from the left rail column to the right rail column, extending one row past the rail endpoints and clamping to the canvas.
    - Use the cropped rectangle's center row and center column as the two mirror axes.
    - For each yellow cell, inspect its full mirror orbit under those two crop-centered reflections and copy the unique resolved non-yellow color from the other orbit members.
    - After all yellow holes are repaired, scan each crop row left to right and fill black cells strictly between the 1st/2nd, 3rd/4th, ... red pixels with green.

    Key insight:
    - The symmetry axes are defined by the cropped rail-bounded rectangle rather than by the whole grid, so cropping is a required precursor to both the repair and the final fill.
    """
    g = np.asarray(grid, dtype=int)
    r0, c0, r1, c1 = _crop_bounds_from_rails(g)
    crop = g[r0 : r1 + 1, c0 : c1 + 1]
    repaired = _repair_crop_holes(crop)
    return _fill_between_red_pairs(repaired)


def _draw_rect_outline(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    grid[r0, c0 : c1 + 1] = Color.RED
    grid[r1, c0 : c1 + 1] = Color.RED
    grid[r0 : r1 + 1, c0] = Color.RED
    grid[r0 : r1 + 1, c1] = Color.RED


def _build_complete_crop(rng: np.random.Generator) -> np.ndarray:
    h = int(rng.choice([7, 9]))
    w = int(rng.choice([9, 11, 13, 15]))
    crop = np.zeros((h, w), dtype=int)
    crop[1 : h - 1, 0] = Color.GREY
    crop[1 : h - 1, w - 1] = Color.GREY

    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    rng.shuffle(corners)
    for r, c in corners[: int(rng.integers(2, 5))]:
        crop[r, c] = Color.TEAL

    style = "quad" if h >= 9 and w >= 11 and rng.random() < 0.45 else "single"
    if style == "single":
        cr = h // 2
        cc = w // 2
        half_h = int(rng.integers(2, cr))
        half_w = int(rng.integers(2, cc))
        _draw_rect_outline(crop, cr - half_h, cc - half_w, cr + half_h, cc + half_w)
    else:
        cr = h // 2
        cc = w // 2
        box_h = int(rng.integers(2, cr - 1))
        box_w = int(rng.integers(2, cc - 1))
        max_r0 = cr - box_h - 1
        max_c0 = cc - box_w - 1
        r0 = int(rng.integers(1, max_r0 + 1))
        c0 = int(rng.integers(1, max_c0 + 1))
        r1 = r0 + box_h - 1
        c1 = c0 + box_w - 1
        rects = [
            (r0, c0, r1, c1),
            (r0, w - 1 - c1, r1, w - 1 - c0),
            (h - 1 - r1, c0, h - 1 - r0, c1),
            (h - 1 - r1, w - 1 - c1, h - 1 - r0, w - 1 - c0),
        ]
        for rect in rects:
            _draw_rect_outline(crop, *rect)

    return crop


def _all_orbits_of_color(crop: np.ndarray, color: int) -> list[tuple[tuple[int, int], ...]]:
    h, w = crop.shape
    seen = set()
    out = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            orbit = _mirror_orbit(h, w, r, c)
            if orbit in seen:
                continue
            seen.add(orbit)
            if all(int(crop[rr, cc]) == int(color) for rr, cc in orbit):
                out.append(orbit)
    return out


def _make_input_crop(rng: np.random.Generator, complete_crop: np.ndarray) -> np.ndarray:
    crop = np.array(complete_crop, copy=True)
    red_orbits = _all_orbits_of_color(crop, Color.RED)
    black_orbits = _all_orbits_of_color(crop, Color.BLACK)
    rng.shuffle(red_orbits)
    rng.shuffle(black_orbits)
    assert red_orbits, "need at least one red orbit for repair holes"
    assert black_orbits, "need at least one black orbit to block the yellow->red shortcut"

    yellow_count = 0
    red_holes = 0
    used = set()

    for orbit in red_orbits:
        if yellow_count >= 2 and red_holes >= 2:
            break
        orbit_cells = [cell for cell in orbit if cell not in used]
        if len(orbit_cells) < 2:
            continue
        k = 2 if len(orbit_cells) >= 3 else 1
        if yellow_count + k > 4:
            continue
        chosen = rng.choice(len(orbit_cells), size=k, replace=False)
        for idx in np.atleast_1d(chosen):
            rr, cc = orbit_cells[int(idx)]
            crop[rr, cc] = Color.YELLOW
            used.add((rr, cc))
            yellow_count += 1
            red_holes += 1

    if yellow_count < 2 or red_holes < 2:
        raise AssertionError("failed to create enough red holes")

    for orbit in black_orbits:
        if yellow_count >= 4:
            break
        orbit_cells = [cell for cell in orbit if cell not in used]
        if len(orbit_cells) < 2:
            continue
        k = 1 if rng.random() < 0.7 else min(2, len(orbit_cells) - 1)
        if yellow_count + k > 4:
            continue
        chosen = rng.choice(len(orbit_cells), size=k, replace=False)
        for idx in np.atleast_1d(chosen):
            rr, cc = orbit_cells[int(idx)]
            crop[rr, cc] = Color.YELLOW
            used.add((rr, cc))
            yellow_count += 1
        break

    assert 2 <= yellow_count <= 4
    return crop


def _embed_crop(rng: np.random.Generator, crop: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = crop.shape
    H = int(rng.integers(max(h + 4, 14), min(25, 30) + 1))
    W = int(rng.integers(max(w + 4, 14), min(25, 30) + 1))
    grid = np.zeros((H, W), dtype=int)
    for _ in range(100):
        top = int(rng.integers(0, H - h + 1))
        left = int(rng.integers(0, W - w + 1))
        if abs((2 * top + h - 1) - (H - 1)) >= 4 or abs((2 * left + w - 1) - (W - 1)) >= 4:
            break
    else:
        raise AssertionError("failed to offset crop away from the full-grid center")

    grid[top : top + h, left : left + w] = crop

    noise_cells = [
        (r, c)
        for r in range(H)
        for c in range(W)
        if not (top <= r < top + h and left <= c < left + w)
    ]
    rng.shuffle(noise_cells)
    for r, c in noise_cells[: int(rng.integers(1, 6))]:
        grid[r, c] = int(rng.choice(GREY_NOISE_FREE_COLORS))
    return grid, (top, left)


def _wrong_global_axis_solver(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    H, W = g.shape
    gr = (H - 1) / 2.0
    gc = (W - 1) / 2.0

    while np.any(g == Color.YELLOW):
        progress = False
        for r, c in np.argwhere(g == Color.YELLOW):
            r = int(r)
            c = int(c)
            candidates = {(r, c)}
            for rr in (r, int(round(2 * gr - r))):
                for cc in (c, int(round(2 * gc - c))):
                    if 0 <= rr < H and 0 <= cc < W:
                        candidates.add((int(rr), int(cc)))
            known = {int(g[rr, cc]) for rr, cc in candidates if int(g[rr, cc]) != Color.YELLOW}
            if len(known) == 1 and known <= {Color.BLACK, Color.RED}:
                fill = known.pop()
                g[r, c] = fill
                progress = True
        if not progress:
            break

    r0, c0, r1, c1 = _crop_bounds_from_rails(grid)
    return _fill_between_red_pairs(g[r0 : r1 + 1, c0 : c1 + 1])


def _validate_instance(inp: np.ndarray, complete_crop: np.ndarray) -> bool:
    expected = _fill_between_red_pairs(complete_crop)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False
    if not np.any(actual == Color.GREEN):
        return False

    r0, c0, r1, c1 = _crop_bounds_from_rails(inp)
    crop_in = inp[r0 : r1 + 1, c0 : c1 + 1]
    if int(np.count_nonzero(crop_in == Color.YELLOW)) not in (2, 3, 4):
        return False
    if np.array_equal(_fill_between_red_pairs(crop_in), expected):
        return False

    naive = np.array(crop_in, copy=True)
    naive[naive == Color.YELLOW] = Color.RED
    if np.array_equal(_fill_between_red_pairs(naive), expected):
        return False

    if np.array_equal(_wrong_global_axis_solver(inp), expected):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = np.random.default_rng()
        try:
            complete_crop = _build_complete_crop(rng)
            input_crop = _make_input_crop(rng, complete_crop)
            inp, _offset = _embed_crop(rng, input_crop)
        except AssertionError:
            continue
        if _validate_instance(inp, complete_crop):
            return inp
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
