from __future__ import annotations

from dataclasses import dataclass

import numpy as np


BACKGROUND = 0
BLUE = 1
MARKER_COLORS = (2, 3, 4, 7)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


@dataclass
class ComponentInfo:
    coords: list[tuple[int, int]]
    bbox: tuple[int, int, int, int]
    mask: np.ndarray
    area: int
    marker_color: int
    marker_pos: tuple[int, int]
    thick: bool


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(coords: list[tuple[int, int]], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        out[r - r0, c - c0] = True
    return out


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _supported_blue_cells(grid: np.ndarray, coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = grid.shape
    comp_set = set(coords)
    supported: list[tuple[int, int]] = []
    for r, c in coords:
        if grid[r, c] != BLUE:
            continue
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            continue
        ok = True
        for dr, dc in DIR8:
            if (r + dr, c + dc) not in comp_set:
                ok = False
                break
        if ok:
            supported.append((r, c))
    return supported


def _analyze_components(grid: np.ndarray) -> list[ComponentInfo]:
    comps = _connected_components(grid != BACKGROUND)
    infos: list[ComponentInfo] = []
    for coords in comps:
        values = [int(grid[r, c]) for r, c in coords]
        marker_cells = [(r, c) for r, c in coords if int(grid[r, c]) != BLUE]
        marker_colors = sorted({int(grid[r, c]) for r, c in marker_cells})
        assert BLUE in values, "each component must contain blue structure"
        assert len(marker_colors) == 1, "each component must contain exactly one marker color"
        marker_color = marker_colors[0]
        assert marker_color in MARKER_COLORS, "unexpected marker color"
        assert len(marker_cells) == 1, "marker must be a singleton pixel"
        bbox = _bbox_from_coords(coords)
        infos.append(
            ComponentInfo(
                coords=coords,
                bbox=bbox,
                mask=_crop_mask(coords, bbox),
                area=len(coords),
                marker_color=marker_color,
                marker_pos=marker_cells[0],
                thick=bool(_supported_blue_cells(grid, coords)),
            )
        )
    return infos


def _pack_two_masks(mask_a: np.ndarray, mask_b: np.ndarray) -> dict[str, object]:
    coords_a = [tuple(x) for x in np.argwhere(mask_a)]
    coords_b = [tuple(x) for x in np.argwhere(mask_b)]
    set_a = set(coords_a)
    ha, wa = mask_a.shape
    hb, wb = mask_b.shape
    area_a = len(coords_a)
    area_b = len(coords_b)
    best_score: tuple[int, int, int, int, int, int] | None = None
    best_result: dict[str, object] | None = None

    for dr in range(-hb, ha + 1):
        for dc in range(-wb, wa + 1):
            overlap = False
            for r, c in coords_b:
                if (r + dr, c + dc) in set_a:
                    overlap = True
                    break
            if overlap:
                continue

            min_r = min(0, dr)
            min_c = min(0, dc)
            max_r = max(ha - 1, dr + hb - 1)
            max_c = max(wa - 1, dc + wb - 1)
            out_h = max_r - min_r + 1
            out_w = max_c - min_c + 1

            pos_a = (-min_r, -min_c)
            pos_b = (dr - min_r, dc - min_c)
            first_upper_left = 0 if pos_a[0] <= pos_b[0] and pos_a[1] <= pos_b[1] else 1
            score = (out_h * out_w, first_upper_left, pos_a[0], pos_a[1], pos_b[0], pos_b[1])
            if best_score is None or score < best_score:
                best_score = score
                best_result = {
                    "shape": (out_h, out_w),
                    "pos_a": pos_a,
                    "pos_b": pos_b,
                    "blank_count": out_h * out_w - area_a - area_b,
                }

    assert best_result is not None, "no legal non-overlapping placement found"
    return best_result


def _paint_mask(canvas: np.ndarray, mask: np.ndarray, top_left: tuple[int, int], color: int) -> None:
    r0, c0 = top_left
    rows, cols = np.where(mask)
    canvas[rows + r0, cols + c0] = int(color)


def _solve(grid: np.ndarray) -> np.ndarray:
    infos = _analyze_components(grid)
    assert len(infos) == 4, "expected exactly four components"
    thick_infos = [info for info in infos if info.thick]
    thin_infos = [info for info in infos if not info.thick]
    assert len(thick_infos) == 2 and len(thin_infos) == 2, "expected two thick and two thin components"

    thick_infos.sort(key=lambda info: (info.marker_pos[0], info.marker_pos[1]))
    packed = _pack_two_masks(thick_infos[0].mask, thick_infos[1].mask)

    out_h, out_w = packed["shape"]
    out = np.zeros((out_h, out_w), dtype=int)
    _paint_mask(out, thick_infos[0].mask, packed["pos_a"], thick_infos[0].marker_color)
    _paint_mask(out, thick_infos[1].mask, packed["pos_b"], thick_infos[1].marker_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only thick marked blue components, pack the survivors as tightly as possible, and recolor them by marker.

    Input structure:
    - A black background contains exactly four disjoint 4-connected components whose body color is blue.
    - Each component contains exactly one singleton marker pixel in a unique color from {2,3,4,7}.
    - Two components are thick and contain at least one blue pixel with a full 3x3 neighborhood inside the same component; the other two are thin width-1 style shapes.

    Rule:
    - For each component, test whether some non-marker blue pixel has all eight neighbors inside that component.
    - Erase the two thin components completely.
    - Crop the two thick components to their own bounding boxes, enumerate all non-overlapping relative translations, and keep the legal placement with minimum merged bounding-box area.
    - If several placements tie, prefer the one where the thick component whose marker is higher in the input sits more toward the upper-left.
    - Recolor every cell of each packed thick component, including its marker position, to that component's marker color, and return the cropped merged bounding box.

    Key insight:
    - The local 3x3 support test is not the final paint operation here; it is the filter that decides which components survive into the later densest packing problem.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    return _solve(g)


def _mask_supported_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    coords = [tuple(x) for x in np.argwhere(mask)]
    coord_set = set(coords)
    supported: list[tuple[int, int]] = []
    for r, c in coords:
        ok = True
        for dr, dc in DIR8:
            if (r + dr, c + dc) not in coord_set:
                ok = False
                break
        if ok:
            supported.append((r, c))
    return supported


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1].copy()


def _random_transform(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.rot90(mask, int(rng.integers(0, 4)))
    if int(rng.integers(0, 2)):
        out = np.fliplr(out)
    return out.copy()


def _frontier_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    frontier: set[tuple[int, int]] = set()
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for dr, dc in DIR4:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                frontier.add((nr, nc))
    return sorted(frontier)


def _generate_thick_mask(rng: np.random.Generator) -> np.ndarray:
    base = np.zeros((5, 5), dtype=bool)
    base[1:4, 1:4] = True
    extras = int(rng.choice(np.array([0, 1, 2, 3]), p=np.array([0.2, 0.3, 0.3, 0.2])))
    for _ in range(extras):
        frontier = _frontier_cells(base)
        if not frontier:
            break
        nr, nc = frontier[int(rng.integers(0, len(frontier)))]
        base[nr, nc] = True
    out = _trim_mask(base)
    supported = _mask_supported_cells(out)
    assert supported, "generated thick mask lost its interior"
    assert 9 <= int(out.sum()) <= 12
    return out


def _would_make_2x2(cells: set[tuple[int, int]], r: int, c: int) -> bool:
    trial = set(cells)
    trial.add((r, c))
    for r0 in (r - 1, r):
        for c0 in (c - 1, c):
            square = {
                (r0, c0),
                (r0 + 1, c0),
                (r0, c0 + 1),
                (r0 + 1, c0 + 1),
            }
            if square.issubset(trial):
                return True
    return False


def _generate_thin_mask(rng: np.random.Generator) -> np.ndarray:
    for _ in range(200):
        h = int(rng.integers(5, 8))
        w = int(rng.integers(5, 8))
        target = int(rng.integers(5, 13))
        start = (int(rng.integers(0, h)), int(rng.integers(0, w)))
        path = [start]
        cells = {start}
        while len(path) < target:
            end_index = 0 if int(rng.integers(0, 2)) == 0 else -1
            br, bc = path[end_index]
            choices = []
            for dr, dc in DIR4:
                nr, nc = br + dr, bc + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in cells or _would_make_2x2(cells, nr, nc):
                    continue
                choices.append((nr, nc))
            if not choices:
                break
            nxt = choices[int(rng.integers(0, len(choices)))]
            cells.add(nxt)
            if end_index == 0:
                path.insert(0, nxt)
            else:
                path.append(nxt)
        if len(cells) != target:
            continue
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        mask = np.zeros((max(rs) - min(rs) + 1, max(cs) - min(cs) + 1), dtype=bool)
        for r, c in cells:
            mask[r - min(rs), c - min(cs)] = True
        if _mask_supported_cells(mask):
            continue
        return mask
    raise RuntimeError("failed to generate thin mask")


def _choose_marker_for_thick(mask: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    supported = set(_mask_supported_cells(mask))
    borderish = [tuple(x) for x in np.argwhere(mask) if tuple(x) not in supported]
    assert borderish
    return borderish[int(rng.integers(0, len(borderish)))]


def _choose_marker_for_thin(mask: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    coords = [tuple(x) for x in np.argwhere(mask)]
    return coords[int(rng.integers(0, len(coords)))]


def _make_sprite(mask: np.ndarray, marker_pos: tuple[int, int], marker_color: int) -> np.ndarray:
    sprite = np.zeros(mask.shape, dtype=int)
    sprite[mask] = BLUE
    sprite[marker_pos] = int(marker_color)
    return sprite


def _find_placement(
    forbidden: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    border: int = 1,
) -> tuple[int, int] | None:
    h, w = forbidden.shape
    mh, mw = mask.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(border, h - mh - border + 1):
        for c0 in range(border, w - mw - border + 1):
            ok = True
            for r, c in np.argwhere(mask):
                if forbidden[r0 + int(r), c0 + int(c)]:
                    ok = False
                    break
            if ok:
                positions.append((r0, c0))
    if not positions:
        return None
    return positions[int(rng.integers(0, len(positions)))]


def _place_sprite(grid: np.ndarray, forbidden: np.ndarray, sprite: np.ndarray, top_left: tuple[int, int]) -> None:
    r0, c0 = top_left
    h, w = grid.shape
    for r, c in np.argwhere(sprite != BACKGROUND):
        rr = r0 + int(r)
        cc = c0 + int(c)
        grid[rr, cc] = int(sprite[int(r), int(c)])
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    forbidden[nr, nc] = True


def _input_signature(grid: np.ndarray) -> tuple:
    infos = _analyze_components(grid)
    shape_sig = []
    for info in sorted(infos, key=lambda item: (item.area, item.marker_color, item.bbox)):
        shape_sig.append((info.area, info.thick, tuple(info.mask.astype(int).ravel()), info.mask.shape))
    return tuple(shape_sig)


def _validate_generated_input(grid: np.ndarray) -> None:
    infos = _analyze_components(grid)
    assert len(infos) == 4
    assert sorted(info.marker_color for info in infos) == sorted(MARKER_COLORS)
    assert all(5 <= info.area <= 12 for info in infos)
    thick_infos = [info for info in infos if info.thick]
    thin_infos = [info for info in infos if not info.thick]
    assert len(thick_infos) == 2 and len(thin_infos) == 2
    assert all(not _mask_supported_cells(info.mask) for info in thin_infos)
    thick_infos.sort(key=lambda info: (info.marker_pos[0], info.marker_pos[1]))
    packed = _pack_two_masks(thick_infos[0].mask, thick_infos[1].mask)
    assert int(packed["blank_count"]) <= 2
    out = main(grid)
    assert out.ndim == 2 and max(out.shape) <= 30
    assert np.count_nonzero(out) == thick_infos[0].area + thick_infos[1].area


def _generate_input_once(rng: np.random.Generator) -> np.ndarray:
    thick_pair = None
    for _ in range(300):
        mask_a = _random_transform(_generate_thick_mask(rng), rng)
        mask_b = _random_transform(_generate_thick_mask(rng), rng)
        packed = _pack_two_masks(mask_a, mask_b)
        if int(packed["blank_count"]) <= 2:
            thick_pair = (mask_a, mask_b)
            break
    if thick_pair is None:
        raise RuntimeError("failed to generate complementary thick pair")

    thin_a = _random_transform(_generate_thin_mask(rng), rng)
    thin_b = _random_transform(_generate_thin_mask(rng), rng)
    marker_palette = list(rng.permutation(np.array(MARKER_COLORS, dtype=int)))

    specs = [
        ("thick", thick_pair[0], _choose_marker_for_thick(thick_pair[0], rng), marker_palette[0]),
        ("thick", thick_pair[1], _choose_marker_for_thick(thick_pair[1], rng), marker_palette[1]),
        ("thin", thin_a, _choose_marker_for_thin(thin_a, rng), marker_palette[2]),
        ("thin", thin_b, _choose_marker_for_thin(thin_b, rng), marker_palette[3]),
    ]

    rows = int(rng.integers(14, 19))
    cols = int(rng.integers(12, 18))
    grid = np.zeros((rows, cols), dtype=int)
    forbidden = np.zeros((rows, cols), dtype=bool)

    place_order = sorted(
        range(len(specs)),
        key=lambda idx: (-int(specs[idx][1].sum()), int(rng.integers(0, 1000000))),
    )
    placements: dict[int, tuple[int, int]] = {}
    for idx in place_order:
        _, mask, marker_pos, marker_color = specs[idx]
        sprite = _make_sprite(mask, marker_pos, int(marker_color))
        pos = _find_placement(forbidden, sprite != BACKGROUND, rng)
        if pos is None:
            raise RuntimeError("placement failed")
        _place_sprite(grid, forbidden, sprite, pos)
        placements[idx] = pos

    _validate_generated_input(grid)

    kinds = [spec[0] for spec in specs]
    thick_rows = [placements[i][0] + specs[i][2][0] for i, kind in enumerate(kinds) if kind == "thick"]
    thin_rows = [placements[i][0] + specs[i][2][0] for i, kind in enumerate(kinds) if kind == "thin"]
    assert thick_rows and thin_rows
    assert not np.array_equal(grid, main(grid))
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        try:
            return _generate_input_once(rng)
        except RuntimeError:
            continue
        except AssertionError:
            continue
    raise RuntimeError("failed to generate valid input")
