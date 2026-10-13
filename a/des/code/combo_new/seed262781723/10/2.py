import random
from collections import deque

import numpy as np


BG = 8       # azure
SEP = 9      # maroon
PALETTE = (0, 1, 2, 3, 4, 5, 6, 7)
ORTHO_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))

GEN_RETRIES = 300
SHAPE_RETRIES = 200
PLACEMENT_RETRIES = 250


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    assert rows.size > 0
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in ORTHO_DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                nxt = (nr, nc)
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
    return len(seen) == int(mask.sum())


def _symmetry_code(mask: np.ndarray) -> str:
    horiz = np.array_equal(mask, mask[::-1, :])
    vert = np.array_equal(mask, mask[:, ::-1])
    if horiz and vert:
        return "both"
    if horiz:
        return "h"
    if vert:
        return "v"
    return "none"


def _component_records(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or g[r, c] == BG:
                continue
            color = int(g[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ORTHO_DIRS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and g[nr, nc] == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True
            components.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                }
            )
    return components


def _render_from_components(components: list[dict], width: int) -> np.ndarray:
    groups = {"h": [], "v": [], "both": []}
    ordered = sorted(components, key=lambda comp: (comp["bbox"][1], comp["bbox"][0]))
    for comp in ordered:
        code = _symmetry_code(comp["mask"])
        if code in groups:
            groups[code].append(comp)

    band_h = max((comp["mask"].shape[0] for comp in ordered if _symmetry_code(comp["mask"]) != "none"), default=1)
    out = np.full((3 * band_h + 2, width), BG, dtype=int)
    out[band_h, :] = SEP
    out[2 * band_h + 1, :] = SEP

    band_starts = {"h": 0, "v": band_h + 1, "both": 2 * band_h + 2}
    for code in ("h", "v", "both"):
        cursor = 0
        top = band_starts[code]
        for comp in groups[code]:
            mask = comp["mask"]
            color = int(comp["color"])
            hh, ww = mask.shape
            assert cursor + ww <= width, "packed output exceeds input width"
            block = out[top + band_h - hh : top + band_h, cursor : cursor + ww]
            block[mask] = color
            cursor += ww + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Repack each colored shape into the row band matching its reflection symmetry.

    Input structure:
    - The background is azure.
    - Several non-overlapping monochromatic 4-connected shapes appear at arbitrary positions.
    - Each shape has its own color, and symmetry is judged inside the shape's minimal bounding box.

    Rule:
    - Extract every foreground component and test whether its cropped mask matches its horizontal reflection, vertical reflection, or both.
    - Keep only shapes that are horizontally symmetric, vertically symmetric, or symmetric in both axes; discard non-symmetric shapes.
    - Build an azure output whose width equals the input width and whose height is split into three equal bands by two full-width maroon rows; pack horizontal-only, vertical-only, and double-symmetric shapes into the top, middle, and bottom bands, preserving left-to-right source order, inserting one background column between neighbors, and bottom-aligning each mask inside its band.

    Key insight:
    - The destination band is determined only by the visible symmetry class of the cropped shape, not by its color or absolute position.
    """
    g = np.asarray(grid, dtype=int)
    return _render_from_components(_component_records(g), g.shape[1])


def _orbit_cells(h: int, w: int, mode: str, r: int, c: int) -> tuple[tuple[int, int], ...]:
    orbit = {(r, c)}
    changed = True
    use_h = mode in {"h", "both"}
    use_v = mode in {"v", "both"}
    while changed:
        changed = False
        current = list(orbit)
        for rr, cc in current:
            candidates = {(rr, cc)}
            if use_h:
                candidates.add((h - 1 - rr, cc))
            if use_v:
                candidates.add((rr, w - 1 - cc))
            if use_h and use_v:
                candidates.add((h - 1 - rr, w - 1 - cc))
            for cell in candidates:
                if cell not in orbit:
                    orbit.add(cell)
                    changed = True
    return tuple(sorted(orbit))


def _orbits(h: int, w: int, mode: str) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[int, int]] = set()
    out = []
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            orbit = _orbit_cells(h, w, mode, r, c)
            out.append(orbit)
            seen.update(orbit)
    return out


def _orbit_graph(h: int, w: int, orbits: list[tuple[tuple[int, int], ...]]) -> list[set[int]]:
    owner: dict[tuple[int, int], int] = {}
    for idx, orbit in enumerate(orbits):
        for cell in orbit:
            owner[cell] = idx

    graph = [set() for _ in orbits]
    for (r, c), idx in owner.items():
        for dr, dc in ORTHO_DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                other = owner[(nr, nc)]
                if other != idx:
                    graph[idx].add(other)
    return graph


def _grow_connected_mask(h: int, w: int, mode: str, target_orbits: int) -> np.ndarray | None:
    orbits = _orbits(h, w, mode)
    if len(orbits) < target_orbits:
        return None
    graph = _orbit_graph(h, w, orbits)
    start_choices = [idx for idx, nbrs in enumerate(graph) if nbrs or target_orbits == 1]
    if not start_choices:
        return None
    chosen = {random.choice(start_choices)}
    frontier = set(graph[next(iter(chosen))])
    while len(chosen) < target_orbits:
        candidates = [idx for idx in frontier if idx not in chosen]
        if not candidates:
            return None
        pick = random.choice(candidates)
        chosen.add(pick)
        frontier.update(graph[pick])

    mask = np.zeros((h, w), dtype=bool)
    for idx in chosen:
        for r, c in orbits[idx]:
            mask[r, c] = True
    return mask


def _make_shape(kind: str) -> np.ndarray:
    ranges = {
        "h": ((3, 8), (3, 8)),
        "v": ((3, 8), (3, 8)),
        "both": ((3, 7), (3, 7)),
        "none": ((2, 7), (2, 7)),
    }
    orbit_mode = kind if kind in {"h", "v", "both"} else "none"
    (h0, h1), (w0, w1) = ranges[kind]

    for _ in range(SHAPE_RETRIES):
        h = random.randint(h0, h1)
        w = random.randint(w0, w1)
        orbits = _orbits(h, w, orbit_mode)
        usable = max(2, int(np.ceil(len(orbits) * 0.65)))
        target_orbits = random.randint(2, usable)
        mask = _grow_connected_mask(h, w, orbit_mode, target_orbits)
        if mask is None or mask.sum() < 3:
            continue
        mask = _trim_mask(mask)
        if min(mask.shape) < 2:
            continue
        if not _mask_connected(mask):
            continue
        if _symmetry_code(mask) != kind:
            continue
        return mask
    raise RuntimeError(f"failed to generate a {kind} shape")


def _band_width(shapes: list[dict], code: str) -> int:
    chosen = [shape for shape in shapes if shape["kind"] == code]
    if not chosen:
        return 0
    return sum(int(shape["mask"].shape[1]) for shape in chosen) + len(chosen) - 1


def _can_place(occupied: np.ndarray, top: int, left: int, h: int, w: int) -> bool:
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(occupied.shape[0], top + h + 1)
    c1 = min(occupied.shape[1], left + w + 1)
    return not bool(occupied[r0:r1, c0:c1].any())


def _place_shapes(shapes: list[dict], grid_h: int, grid_w: int) -> np.ndarray | None:
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    used_lefts: set[int] = set()

    for shape in sorted(shapes, key=lambda item: int(item["mask"].sum()), reverse=True):
        mask = shape["mask"]
        h, w = mask.shape
        candidates = []
        for _ in range(PLACEMENT_RETRIES):
            top = random.randint(0, grid_h - h)
            left = random.randint(0, grid_w - w)
            if left in used_lefts:
                continue
            if _can_place(occupied, top, left, h, w):
                candidates.append((top, left))
        if not candidates:
            return None
        top, left = random.choice(candidates)
        block = grid[top : top + h, left : left + w]
        block[mask] = int(shape["color"])
        occupied[top : top + h, left : left + w] |= mask
        shape["top"] = top
        shape["left"] = left
        used_lefts.add(left)
    return grid


def _expected_output(shapes: list[dict], width: int) -> np.ndarray:
    components = []
    for shape in shapes:
        top = int(shape["top"])
        left = int(shape["left"])
        mask = np.asarray(shape["mask"], dtype=bool)
        h, w = mask.shape
        components.append(
            {
                "color": int(shape["color"]),
                "bbox": (top, left, top + h - 1, left + w - 1),
                "mask": mask,
            }
        )
    return _render_from_components(components, width)


def _gen_attempt() -> np.ndarray | None:
    kinds = ["h", "v", "both", "none"]
    extra_count = random.randint(0, 2)
    kinds.extend(random.choice(("h", "v", "both", "none")) for _ in range(extra_count))
    random.shuffle(kinds)

    if len(kinds) > len(PALETTE):
        return None
    colors = random.sample(list(PALETTE), len(kinds))

    try:
        shapes = [{"kind": kind, "color": color, "mask": _make_shape(kind)} for kind, color in zip(kinds, colors)]
    except RuntimeError:
        return None

    band_h = max(int(shape["mask"].shape[0]) for shape in shapes if shape["kind"] != "none")
    out_h = 3 * band_h + 2
    if out_h > 30:
        return None

    required_w = max(_band_width(shapes, code) for code in ("h", "v", "both"))
    max_shape_h = max(int(shape["mask"].shape[0]) for shape in shapes)
    max_shape_w = max(int(shape["mask"].shape[1]) for shape in shapes)

    min_w = max(required_w, max_shape_w + 4, 16)
    min_h = max(out_h, max_shape_h + 4, 16)
    if min_w > 30 or min_h > 30:
        return None

    grid_w = random.randint(min_w, 30)
    grid_h = random.randint(min_h, 30)
    grid = _place_shapes(shapes, grid_h, grid_w)
    if grid is None:
        return None

    expected = _expected_output(shapes, grid_w)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid symmetry-sorting input")
