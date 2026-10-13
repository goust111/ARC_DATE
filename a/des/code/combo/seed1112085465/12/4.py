import numpy as np


BG = 8
RING = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 9)
DIRS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
SCAN_START = {
    "up": (-1, 1),
    "right": (1, 3),
    "down": (3, 1),
    "left": (1, -1),
}
MAX_ATTEMPTS = 400


def _build_specs():
    base_u = np.array(
        [
            [0, 0, 0],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=bool,
    )
    base_c = np.array(
        [
            [1, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=bool,
    )
    specs = []
    mask_to_spec = {}
    for kind, base in (("u", base_u), ("c", base_c)):
        for turns, direction in enumerate(("up", "right", "down", "left")):
            mask = np.rot90(base, -turns)
            coords = np.argwhere(mask)
            trim_top = int(coords[:, 0].min())
            trim_left = int(coords[:, 1].min())
            trim_bottom = int(coords[:, 0].max()) + 1
            trim_right = int(coords[:, 1].max()) + 1
            spec = {
                "kind": kind,
                "direction": direction,
                "mask": mask,
                "fill": [tuple(map(int, p)) for p in np.argwhere(~mask)],
                "trim_top": trim_top,
                "trim_left": trim_left,
                "trimmed_mask": mask[trim_top:trim_bottom, trim_left:trim_right],
            }
            key = tuple(mask.astype(int).ravel())
            specs.append(spec)
            mask_to_spec[key] = spec
    return specs, mask_to_spec


SPECS, MASK_TO_SPEC = _build_specs()


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            comp.append((cr, cc))
            for dr, dc in DIRS.values():
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        out.append(comp)
    return out


def _find_rings(grid: np.ndarray) -> list[tuple[int, int, dict]]:
    rings = []
    for comp in _components(grid == RING):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            mask[r - r0, c - c0] = True
        spec = None
        full_r0 = full_c0 = None
        if mask.shape == (3, 3):
            spec = MASK_TO_SPEC.get(tuple(mask.astype(int).ravel()))
            if spec is not None:
                full_r0, full_c0 = r0, c0
        if spec is None:
            for candidate in SPECS:
                if candidate["trimmed_mask"].shape != mask.shape:
                    continue
                if np.array_equal(candidate["trimmed_mask"], mask):
                    spec = candidate
                    full_r0 = r0 - candidate["trim_top"]
                    full_c0 = c0 - candidate["trim_left"]
                    break
        if spec is not None:
            rings.append((full_r0, full_c0, spec))
    rings.sort(key=lambda item: (item[0], item[1], item[2]["direction"], item[2]["kind"]))
    return rings


def _find_marker(grid: np.ndarray, r0: int, c0: int, direction: str):
    h, w = grid.shape
    dr, dc = DIRS[direction]
    sr, sc = SCAN_START[direction]
    rr, cc = r0 + sr, c0 + sc
    while 0 <= rr < h and 0 <= cc < w:
        color = int(grid[rr, cc])
        if color != BG and color != RING:
            return rr, cc, color
        rr += dr
        cc += dc
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each grey ring's opening to find a marker and fill the ring's cavity.

    Input structure:
    - Background is azure.
    - Grey 3x3 C/U-shaped rings open in one cardinal direction.
    - A colored singleton marker lies on the ring's center axis in the opening
      direction.

    Rule:
    - Detect each ring and its opening direction from the local 3x3 pattern.
    - Search from the opening outward along that direction to the first colored
      marker on the same row or column.
    - Fill every background cell inside the ring's 3x3 cavity with that marker
      color, then erase the marker to background.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for r0, c0, spec in _find_rings(g):
        marker = _find_marker(g, r0, c0, spec["direction"])
        if marker is None:
            continue
        mr, mc, color = marker
        out[mr, mc] = BG
        for rr, cc in spec["fill"]:
            out[r0 + rr, c0 + cc] = color
    return out


def _box_clear(blocked: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> bool:
    h, w = blocked.shape
    r0 = max(0, r0)
    c0 = max(0, c0)
    r1 = min(h, r1)
    c1 = min(w, c1)
    return not blocked[r0:r1, c0:c1].any()


def _mark_box(blocked: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> None:
    h, w = blocked.shape
    blocked[max(0, r0) : min(h, r1), max(0, c0) : min(w, c1)] = True


def _origin_candidates(rng: np.random.Generator, h: int, w: int, blocked: np.ndarray, direction: str):
    out = []
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            if direction == "up" and r0 < 2:
                continue
            if direction == "down" and r0 > h - 5:
                continue
            if direction == "left" and c0 < 2:
                continue
            if direction == "right" and c0 > w - 5:
                continue
            if not _box_clear(blocked, r0 - 1, r0 + 4, c0 - 1, c0 + 4):
                continue
            out.append((r0, c0))
    if out:
        order = rng.permutation(len(out))
        out = [out[int(i)] for i in order]
    return out


def _marker_candidates(
    grid: np.ndarray,
    blocked: np.ndarray,
    r0: int,
    c0: int,
    direction: str,
):
    h, w = grid.shape
    dr, dc = DIRS[direction]
    sr, sc = SCAN_START[direction]
    rr, cc = r0 + sr, c0 + sc
    step = 0
    out = []
    while 0 <= rr < h and 0 <= cc < w:
        if grid[rr, cc] != BG:
            break
        step += 1
        if step >= 2 and _box_clear(blocked, rr - 1, rr + 2, cc - 1, cc + 2):
            out.append((rr, cc))
        rr += dr
        cc += dc
    return out


def _paint_ring(grid: np.ndarray, r0: int, c0: int, spec: dict) -> None:
    patch = grid[r0 : r0 + 3, c0 : c0 + 3]
    patch[spec["mask"]] = RING


def _place_ring(
    rng: np.random.Generator,
    grid: np.ndarray,
    blocked: np.ndarray,
    spec: dict,
    color: int,
):
    h, w = grid.shape
    for r0, c0 in _origin_candidates(rng, h, w, blocked, spec["direction"]):
        markers = _marker_candidates(grid, blocked, r0, c0, spec["direction"])
        if not markers:
            continue
        mr, mc = markers[int(rng.integers(0, len(markers)))]
        _paint_ring(grid, r0, c0, spec)
        grid[mr, mc] = color
        _mark_box(blocked, r0 - 1, r0 + 4, c0 - 1, c0 + 4)
        dr, dc = DIRS[spec["direction"]]
        sr, sc = SCAN_START[spec["direction"]]
        rr, cc = r0 + sr, c0 + sc
        while (rr, cc) != (mr, mc):
            blocked[rr, cc] = True
            rr += dr
            cc += dc
        _mark_box(blocked, mr - 1, mr + 2, mc - 1, mc + 2)
        return {
            "r0": r0,
            "c0": c0,
            "spec": spec,
            "marker": (mr, mc, color),
        }
    return None


def _target_ring_count(rng: np.random.Generator, h: int, w: int) -> int:
    max_rings = 1
    if min(h, w) >= 10:
        max_rings += 1
    if h * w >= 140:
        max_rings += 1
    if h * w >= 220:
        max_rings += 1
    return int(rng.integers(1, min(4, max_rings) + 1))


def _expected_output(inp: np.ndarray, placements: list[dict]) -> np.ndarray:
    out = inp.copy()
    for info in placements:
        mr, mc, color = info["marker"]
        out[mr, mc] = BG
        for rr, cc in info["spec"]["fill"]:
            out[info["r0"] + rr, info["c0"] + cc] = color
    return out


def _build_case(rng: np.random.Generator):
    h = int(rng.integers(8, 19))
    w = int(rng.integers(8, 19))
    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    n_rings = _target_ring_count(rng, h, w)
    colors = [int(c) for c in rng.choice(MARKER_COLORS, size=n_rings, replace=False)]
    spec_ids = [int(i) for i in rng.choice(len(SPECS), size=n_rings, replace=True)]

    placements = []
    for color, spec_id in zip(colors, spec_ids):
        info = _place_ring(rng, grid, blocked, SPECS[spec_id], color)
        if info is None:
            return None
        placements.append(info)

    if len(_find_rings(grid)) != n_rings:
        return None

    expected = _expected_output(grid, placements)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _build_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
