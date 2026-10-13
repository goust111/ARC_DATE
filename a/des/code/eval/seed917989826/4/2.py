from __future__ import annotations

from collections import defaultdict

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    seen = np.zeros((height, width), dtype=bool)
    components: list[dict] = []

    for r0 in range(height):
        for c0 in range(width):
            if seen[r0, c0] or int(g[r0, c0]) == int(bg):
                continue
            color = int(g[r0, c0])
            stack = [(int(r0), int(c0))]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []

            while stack:
                r, c = stack.pop()
                coords.append((int(r), int(c)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = int(r + dr)
                    nc = int(c + dc)
                    if 0 <= nr < height and 0 <= nc < width and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [r for r, _ in coords]
            cols = [c for _, c in coords]
            r1 = int(min(rows))
            c1 = int(min(cols))
            r2 = int(max(rows))
            c2 = int(max(cols))
            mask = np.zeros((r2 - r1 + 1, c2 - c1 + 1), dtype=bool)
            for r, c in coords:
                mask[int(r - r1), int(c - c1)] = True
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": len(coords),
                    "bbox": (r1, c1, r2, c2),
                    "mask": mask,
                }
            )

    return components


def _container_geometry(bbox: tuple[int, int, int, int], orientation: str) -> dict:
    r0, c0, r1, c1 = map(int, bbox)
    if orientation == "up":
        opening = [(r0, c) for c in range(c0 + 1, c1)]
        slots = [[(r, c) for c in range(c0 + 1, c1)] for r in range(r1 - 1, r0 - 1, -1)]
    elif orientation == "down":
        opening = [(r1, c) for c in range(c0 + 1, c1)]
        slots = [[(r, c) for c in range(c0 + 1, c1)] for r in range(r0 + 1, r1 + 1)]
    elif orientation == "left":
        opening = [(r, c0) for r in range(r0 + 1, r1)]
        slots = [[(r, c) for r in range(r0 + 1, r1)] for c in range(c1 - 1, c0 - 1, -1)]
    else:
        assert orientation == "right"
        opening = [(r, c1) for r in range(r0 + 1, r1)]
        slots = [[(r, c) for r in range(r0 + 1, r1)] for c in range(c0 + 1, c1 + 1)]

    return {
        "bbox": (r0, c0, r1, c1),
        "orientation": orientation,
        "depth": len(slots),
        "opening": opening,
        "slots": slots,
    }


def _detect_container(comp: dict) -> dict | None:
    mask = np.asarray(comp["mask"], dtype=bool)
    for k, orientation in ((0, "up"), (1, "right"), (2, "down"), (3, "left")):
        rotated = np.rot90(mask, k)
        h, w = map(int, rotated.shape)
        if w != 5 or not (4 <= h <= 6):
            continue
        template = np.zeros((h, 5), dtype=bool)
        template[:, 0] = True
        template[:, 4] = True
        template[-1, :] = True
        if np.array_equal(rotated, template):
            info = _container_geometry(tuple(map(int, comp["bbox"])), orientation)
            info["color"] = int(comp["color"])
            return info
    return None


def _is_bar(comp: dict) -> bool:
    if int(comp["area"]) != 3:
        return False
    h, w = map(int, comp["mask"].shape)
    return (h, w) in {(1, 3), (3, 1)} and bool(comp["mask"].all())


def _opening_distance(coords: list[tuple[int, int]], opening: list[tuple[int, int]]) -> int:
    best = 10**9
    for r, c in coords:
        for rr, cc in opening:
            best = min(best, abs(int(r) - int(rr)) + abs(int(c) - int(cc)))
    return int(best)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each monochrome 3-cell fragment into the same-colored U container.

    Fragments are assigned by color, rotated if needed to match the opening axis,
    then stacked from the closed end toward the opening in order of original
    distance from the opening (farthest first).
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    components = _find_components(g, bg)
    by_color: dict[int, list[dict]] = defaultdict(list)
    for comp in components:
        by_color[int(comp["color"])].append(comp)

    out = g.copy()

    for color, group in by_color.items():
        containers = []
        for comp in group:
            info = _detect_container(comp)
            if info is not None:
                containers.append((comp, info))

        assert len(containers) == 1, "each color must have exactly one U container"
        container_comp, container = containers[0]
        fragments = [comp for comp in group if comp is not container_comp]
        assert 1 <= len(fragments) <= 3
        assert all(_is_bar(comp) for comp in fragments)
        assert len(fragments) <= int(container["depth"])

        for frag in fragments:
            for r, c in frag["coords"]:
                out[int(r), int(c)] = int(bg)

        fragments.sort(
            key=lambda comp: (
                -_opening_distance(comp["coords"], container["opening"]),
                int(comp["bbox"][0]),
                int(comp["bbox"][1]),
            )
        )

        for frag, cells in zip(fragments, container["slots"]):
            assert len(cells) == 3
            for r, c in cells:
                out[int(r), int(c)] = int(color)

    return out


_RNG = np.random.default_rng()

_GEN_ATTEMPTS = 500
_GRID_MIN = 24
_GRID_MAX_EXCLUSIVE = 32
_BORDER = 1
_OPEN_MARGIN = 7
_FRAG_REACH = 10
_SIDE_MARGIN = 5


def _container_patch(color: int, orientation: str, depth: int) -> np.ndarray:
    base = np.zeros((depth + 1, 5), dtype=int)
    base[:, 0] = int(color)
    base[:, 4] = int(color)
    base[-1, :] = int(color)
    ccw = {"up": 0, "left": 1, "down": 2, "right": 3}[orientation]
    return np.rot90(base, ccw)


def _bar_patch(color: int, orientation: str) -> np.ndarray:
    if orientation == "h":
        return np.full((1, 3), int(color), dtype=int)
    return np.full((3, 1), int(color), dtype=int)


def _can_place(forbidden: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> bool:
    mask = patch != int(bg)
    h, w = map(int, patch.shape)
    return not bool((forbidden[r0 : r0 + h, c0 : c0 + w] & mask).any())


def _paint(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    mask = patch != int(bg)
    h, w = map(int, patch.shape)
    view = grid[r0 : r0 + h, c0 : c0 + w]
    view[mask] = patch[mask]


def _container_candidates(height: int, width: int, patch: np.ndarray, orientation: str) -> list[tuple[int, int]]:
    h, w = map(int, patch.shape)
    min_r = _BORDER
    max_r = height - h - _BORDER
    min_c = _BORDER
    max_c = width - w - _BORDER

    if orientation == "up":
        min_r = max(min_r, _OPEN_MARGIN)
    elif orientation == "down":
        max_r = min(max_r, height - h - _OPEN_MARGIN)
    elif orientation == "left":
        min_c = max(min_c, _OPEN_MARGIN)
    else:
        assert orientation == "right"
        max_c = min(max_c, width - w - _OPEN_MARGIN)

    if min_r > max_r or min_c > max_c:
        return []

    spots = [(int(r0), int(c0)) for r0 in range(min_r, max_r + 1) for c0 in range(min_c, max_c + 1)]
    _RNG.shuffle(spots)
    return spots


def _fragment_candidates(
    height: int,
    width: int,
    container: dict,
    patch: np.ndarray,
    forbidden: np.ndarray,
    bg: int,
) -> list[dict]:
    r0, c0, r1, c1 = map(int, container["bbox"])
    ph, pw = map(int, patch.shape)
    spots: list[dict] = []

    if container["orientation"] == "up":
        row_lo = max(_BORDER, r0 - _FRAG_REACH)
        row_hi = r0 - ph
        col_lo = max(_BORDER, c0 - _SIDE_MARGIN)
        col_hi = min(width - pw - _BORDER, c1 + _SIDE_MARGIN - pw + 1)
        rows = range(row_lo, row_hi + 1)
        cols = range(col_lo, col_hi + 1)
    elif container["orientation"] == "down":
        row_lo = r1 + 1
        row_hi = min(height - ph - _BORDER, r1 + _FRAG_REACH)
        col_lo = max(_BORDER, c0 - _SIDE_MARGIN)
        col_hi = min(width - pw - _BORDER, c1 + _SIDE_MARGIN - pw + 1)
        rows = range(row_lo, row_hi + 1)
        cols = range(col_lo, col_hi + 1)
    elif container["orientation"] == "left":
        row_lo = max(_BORDER, r0 - _SIDE_MARGIN)
        row_hi = min(height - ph - _BORDER, r1 + _SIDE_MARGIN - ph + 1)
        col_lo = max(_BORDER, c0 - _FRAG_REACH)
        col_hi = c0 - pw
        rows = range(row_lo, row_hi + 1)
        cols = range(col_lo, col_hi + 1)
    else:
        assert container["orientation"] == "right"
        row_lo = max(_BORDER, r0 - _SIDE_MARGIN)
        row_hi = min(height - ph - _BORDER, r1 + _SIDE_MARGIN - ph + 1)
        col_lo = c1 + 1
        col_hi = min(width - pw - _BORDER, c1 + _FRAG_REACH)
        rows = range(row_lo, row_hi + 1)
        cols = range(col_lo, col_hi + 1)

    if row_lo > row_hi or col_lo > col_hi:
        return []

    for rr in rows:
        for cc in cols:
            if not _can_place(forbidden, patch, int(rr), int(cc), bg):
                continue
            coords = [
                (int(rr + dr), int(cc + dc))
                for dr in range(ph)
                for dc in range(pw)
                if int(patch[dr, dc]) != int(bg)
            ]
            dist = _opening_distance(coords, container["opening"])
            spots.append({"origin": (int(rr), int(cc)), "distance": int(dist)})

    _RNG.shuffle(spots)
    return spots


def _place_container(
    grid: np.ndarray,
    occupied: np.ndarray,
    color: int,
    orientation: str,
    depth: int,
    bg: int,
) -> dict | None:
    patch = _container_patch(int(color), orientation, int(depth))
    forbidden = _dilate4(occupied)
    for r0, c0 in _container_candidates(grid.shape[0], grid.shape[1], patch, orientation):
        if not _can_place(forbidden, patch, int(r0), int(c0), bg):
            continue
        _paint(grid, patch, int(r0), int(c0), bg)
        occupied[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] |= patch != int(bg)
        bbox = (int(r0), int(c0), int(r0 + patch.shape[0] - 1), int(c0 + patch.shape[1] - 1))
        info = _container_geometry(bbox, orientation)
        info["color"] = int(color)
        return info
    return None


def _gen_attempt() -> np.ndarray | None:
    height = int(_RNG.integers(_GRID_MIN, _GRID_MAX_EXCLUSIVE))
    width = int(_RNG.integers(_GRID_MIN, _GRID_MAX_EXCLUSIVE))
    bg = int(_RNG.integers(0, 10))
    container_count = int(_RNG.integers(2, 4))
    palette = [color for color in range(10) if color != bg]
    _RNG.shuffle(palette)
    colors = palette[:container_count]

    grid = np.full((height, width), int(bg), dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    specs = []
    total_fragments = 0
    for color in colors:
        orientation = str(_RNG.choice(np.array(["up", "down", "left", "right"], dtype=object)))
        depth = int(_RNG.integers(3, 6))
        frag_count = int(_RNG.integers(1, 4))
        rotate_flags = [bool(_RNG.integers(0, 2)) for _ in range(frag_count)]
        total_fragments += frag_count
        specs.append(
            {
                "color": int(color),
                "orientation": orientation,
                "depth": int(depth),
                "frag_count": int(frag_count),
                "rotate_flags": rotate_flags,
            }
        )

    if total_fragments >= 2:
        all_flags = [flag for spec in specs for flag in spec["rotate_flags"]]
        if all(all_flags) or not any(all_flags):
            flip_idx = int(_RNG.integers(0, len(all_flags)))
            seen = 0
            for spec in specs:
                for i in range(len(spec["rotate_flags"])):
                    if seen == flip_idx:
                        spec["rotate_flags"][i] = not spec["rotate_flags"][i]
                    seen += 1

    containers: list[dict] = []
    for spec in sorted(specs, key=lambda item: -int(item["depth"])):
        placed = _place_container(grid, occupied, spec["color"], spec["orientation"], spec["depth"], bg)
        if placed is None:
            return None
        placed["rotate_flags"] = list(spec["rotate_flags"])
        containers.append(placed)

    for container in containers:
        need_horizontal = container["orientation"] in {"up", "down"}
        used_distances: set[int] = set()
        for rotate_flag in container["rotate_flags"]:
            frag_orientation = "v" if need_horizontal == rotate_flag else "h"
            patch = _bar_patch(container["color"], frag_orientation)
            candidates = _fragment_candidates(
                height,
                width,
                container,
                patch,
                _dilate4(occupied),
                bg,
            )
            candidates = [cand for cand in candidates if int(cand["distance"]) not in used_distances]
            if not candidates:
                return None
            pick = candidates[int(_RNG.integers(0, len(candidates)))]
            rr, cc = map(int, pick["origin"])
            _paint(grid, patch, rr, cc, bg)
            occupied[rr : rr + patch.shape[0], cc : cc + patch.shape[1]] |= patch != int(bg)
            used_distances.add(int(pick["distance"]))

    if int(_get_bg(grid)) != int(bg):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    if int(np.count_nonzero(out != bg)) != int(np.count_nonzero(grid != bg)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
