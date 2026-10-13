import numpy as np


BG = 0
GREY = 5
OBJECT_COLORS = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
GEN_ATTEMPTS = 400
_RNG = np.random.default_rng()
_GEN_COUNTER = 0


def _find_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
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
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                }
            )

    return components


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flatten each foreground object into a one-cell-thick density bar on its bbox's shorter edge.

    Input structure:
    - The background is Black.
    - The grid contains several 4-connected monochrome foreground objects in distinct colors.
    - Each object's tight bounding box is a non-square rectangle.

    Rule:
    - For each object, compare its bbox height and width to find the short-edge direction.
    - Project the object along the long direction: for each row or column on the short side,
      count occupied object pixels K and compare against the bbox long-side length L.
    - Draw the projection as a one-cell-thick bar on the top short edge for tall objects or the
      left short edge for wide objects: use the object's color when K == L, Grey when 0 < K < L,
      and background when K == 0.
    - Clear all original object pixels outside these projection bars.

    Key insight:
    - Each 2D object is compressed onto its shorter bbox edge, and the resulting 1D bar encodes
      whether each projected slice was fully filled or only partially occupied.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = np.full(g.shape, BG, dtype=int)
    for comp in _find_components(g):
        color = int(comp["color"])
        r0, c0, r1, c1 = comp["bbox"]
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        rows = comp["coords"][:, 0] - r0
        cols = comp["coords"][:, 1] - c0
        local[rows, cols] = True

        box_h, box_w = local.shape
        if box_h <= box_w:
            counts = local.sum(axis=1)
            for dr, count in enumerate(counts):
                k = int(count)
                if k == box_w:
                    out[r0 + dr, c0] = color
                elif k > 0:
                    out[r0 + dr, c0] = GREY
        else:
            counts = local.sum(axis=0)
            for dc, count in enumerate(counts):
                k = int(count)
                if k == box_h:
                    out[r0, c0 + dc] = color
                elif k > 0:
                    out[r0, c0 + dc] = GREY

    return out


def _wide_partial_patch(h: int, w: int) -> np.ndarray:
    for _ in range(80):
        anchor = int(_RNG.integers(0, h))
        intervals: list[tuple[int, int] | None] = [None] * h
        intervals[anchor] = (0, w - 1)

        ok = True
        for step in (-1, 1):
            prev_start, prev_end = intervals[anchor]
            r = anchor + step
            while 0 <= r < h:
                if _RNG.random() < 0.65:
                    span = int(_RNG.integers(max(1, w // 3), w))
                else:
                    span = w
                lo = max(0, prev_start - span + 1)
                hi = min(prev_end, w - span)
                if lo > hi:
                    ok = False
                    break
                start = int(_RNG.integers(lo, hi + 1))
                end = start + span - 1
                intervals[r] = (start, end)
                prev_start, prev_end = start, end
                r += step
            if not ok:
                break
        if not ok:
            continue

        assert all(interval is not None for interval in intervals)
        spans = [int(end - start + 1) for start, end in intervals]  # type: ignore[misc]
        if not any(span < w for span in spans):
            continue

        patch = np.zeros((h, w), dtype=bool)
        for r, interval in enumerate(intervals):
            start, end = interval  # type: ignore[misc]
            patch[r, start : end + 1] = True
        return patch

    raise RuntimeError("failed to build a partial wide patch")


def _make_patch(orientation: str, style: str) -> np.ndarray:
    if orientation == "wide":
        h = int(_RNG.integers(2, 7))
        w = int(_RNG.integers(max(h + 1, 4), min(12, h + 7) + 1))
        if style == "solid":
            return np.ones((h, w), dtype=bool)
        return _wide_partial_patch(h, w)

    w = int(_RNG.integers(2, 7))
    h = int(_RNG.integers(max(w + 1, 4), min(12, w + 7) + 1))
    if style == "solid":
        return np.ones((h, w), dtype=bool)
    return _wide_partial_patch(w, h).T


def _scene_specs() -> list[tuple[str, str]]:
    global _GEN_COUNTER
    idx = _GEN_COUNTER
    _GEN_COUNTER += 1

    n_objects = 2 + (idx % 3)
    if idx % 2 == 0:
        specs: list[tuple[str, str]] = [("wide", "solid"), ("tall", "partial")]
    else:
        specs = [("wide", "partial"), ("tall", "solid")]

    extra_cycle = [
        ("wide", "partial"),
        ("tall", "partial"),
        ("wide", "solid"),
        ("tall", "solid"),
    ]
    while len(specs) < n_objects:
        specs.append(extra_cycle[(idx + len(specs)) % len(extra_cycle)])

    order = _RNG.permutation(len(specs))
    return [specs[int(i)] for i in order]


def _paint_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, color: int) -> None:
    target = grid[top : top + patch.shape[0], left : left + patch.shape[1]]
    target[patch] = color


def _place_scene(patches: list[np.ndarray], colors: list[int]) -> np.ndarray | None:
    max_h = max(int(p.shape[0]) for p in patches)
    max_w = max(int(p.shape[1]) for p in patches)
    padded_area = sum((int(p.shape[0]) + 2) * (int(p.shape[1]) + 2) for p in patches)
    side_floor = max(max_h + 2, max_w + 2, int(np.ceil(np.sqrt(padded_area))) + 2, 14)

    for _ in range(120):
        h_hi = min(30, side_floor + 7)
        w_hi = min(30, side_floor + 7)
        h = int(_RNG.integers(side_floor, h_hi + 1))
        w = int(_RNG.integers(side_floor, w_hi + 1))
        grid = np.full((h, w), BG, dtype=int)
        occupied = np.zeros((h, w), dtype=bool)

        order = sorted(range(len(patches)), key=lambda i: patches[i].shape[0] * patches[i].shape[1], reverse=True)
        placements: dict[int, tuple[int, int]] = {}
        ok = True
        for idx in order:
            patch = patches[idx]
            ph, pw = map(int, patch.shape)
            valid: list[tuple[int, int]] = []
            for top in range(1, h - ph):
                for left in range(1, w - pw):
                    if occupied[top - 1 : top + ph + 1, left - 1 : left + pw + 1].any():
                        continue
                    valid.append((top, left))
            if not valid:
                ok = False
                break
            top, left = valid[int(_RNG.integers(0, len(valid)))]
            placements[idx] = (top, left)
            occupied[top - 1 : top + ph + 1, left - 1 : left + pw + 1] = True

        if not ok:
            continue

        for idx, patch in enumerate(patches):
            top, left = placements[idx]
            _paint_patch(grid, patch, top, left, colors[idx])
        return grid

    return None


def _has_required_structure(grid: np.ndarray) -> bool:
    comps = _find_components(grid)
    if len(comps) < 2:
        return False

    has_wide = False
    has_tall = False
    has_partial = False
    has_full_only = False
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        box_h = int(r1 - r0 + 1)
        box_w = int(c1 - c0 + 1)
        if box_h == box_w:
            return False
        local = np.zeros((box_h, box_w), dtype=bool)
        rows = comp["coords"][:, 0] - r0
        cols = comp["coords"][:, 1] - c0
        local[rows, cols] = True
        if box_h < box_w:
            has_wide = True
            counts = local.sum(axis=1)
            long_len = box_w
        else:
            has_tall = True
            counts = local.sum(axis=0)
            long_len = box_h
        if np.any((counts > 0) & (counts < long_len)):
            has_partial = True
        if np.all(counts == long_len):
            has_full_only = True

    return has_wide and has_tall and has_partial and has_full_only


def _gen_attempt() -> np.ndarray | None:
    specs = _scene_specs()
    patches = [_make_patch(orientation, style) for orientation, style in specs]
    colors = [int(c) for c in _RNG.choice(OBJECT_COLORS, size=len(patches), replace=False)]
    grid = _place_scene(patches, colors)
    if grid is None or not _has_required_structure(grid):
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if not np.any(out == GREY):
        return None
    if len(_find_components(grid)) != len(patches):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
