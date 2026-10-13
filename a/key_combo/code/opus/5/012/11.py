import random
from collections import Counter

import numpy as np


SOURCE_BODY_COLORS = (1, 2, 3)
ANCHOR_COLORS = (4, 5, 6)
BG = 0
GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 300
OVERLAP_PROBABILITY = 0.35


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(sprite, np.ones((factor, factor), dtype=sprite.dtype))


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def paste(dst: np.ndarray, patch: np.ndarray, top: int, left: int, transparent: int = BG) -> None:
    mask = patch != transparent
    h, w = patch.shape
    view = dst[top : top + h, left : left + w]
    view[mask] = patch[mask]


def _parse_components(grid: np.ndarray) -> tuple[int, list[dict], dict[int, tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    comps = components_mask(g != bg)

    sources: list[dict] = []
    markers: dict[int, tuple[int, int]] = {}

    for coords in comps:
        if coords.shape[0] == 1:
            r, c = map(int, coords[0])
            color = int(g[r, c])
            assert color in ANCHOR_COLORS, "singleton components must be anchor markers"
            assert color not in markers, "marker colors must be unique"
            markers[color] = (r, c)
            continue

        vals = [int(g[r, c]) for r, c in coords]
        counts = Counter(vals)
        assert len(counts) == 2, "each source component must have one body color and one anchor color"
        min_count = min(counts.values())
        rare_colors = [color for color, count in counts.items() if count == min_count]
        assert len(rare_colors) == 1 and min_count == 1, "anchor color must be the unique least-frequent color"
        anchor_color = int(rare_colors[0])
        assert anchor_color in ANCHOR_COLORS, "anchor color must come from the marker palette"
        body_colors = [color for color in counts if color != anchor_color]
        assert len(body_colors) == 1
        body_color = int(body_colors[0])
        anchor_coords = [tuple(map(int, rc)) for rc in coords if int(g[rc[0], rc[1]]) == anchor_color]
        assert len(anchor_coords) == 1

        box = bbox_coords(coords)
        patch = crop_bbox(g, box)
        r0, c0, _r1, _c1 = box
        anchor_r, anchor_c = anchor_coords[0]
        sources.append(
            {
                "coords": np.array(coords, dtype=int),
                "bbox": box,
                "patch": patch,
                "area": int(coords.shape[0]),
                "anchor_color": anchor_color,
                "body_color": body_color,
                "anchor_rel": (anchor_r - r0, anchor_c - c0),
            }
        )

    assert len(sources) == 3, "expected exactly three source components"
    assert len(markers) == 3, "expected exactly three target markers"
    areas = [src["area"] for src in sources]
    assert len(set(areas)) == 3, "source component areas must be distinct"
    assert set(markers) == {src["anchor_color"] for src in sources}, "markers must match source anchor colors"
    return bg, sources, markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scale each source component by its area rank and stamp it onto the marker with the same anchor color.

    Input structure:
    - The background is black and all non-background cells split into exactly three multi-pixel source components plus three singleton target markers.
    - Each source component is 4-connected, uses one body color together with one unique least-frequent anchor pixel whose color is yellow, grey, or magenta.
    - The singleton markers use those same three anchor colors and lie away from the source components.

    Rule:
    - Find the three non-background 4-connected source components and identify each component's anchor color as its unique least-frequent color.
    - Sort the source components by total area from small to large; the sorted ranks 1, 2, 3 become integer scale factors.
    - Crop each component to its bounding box, scale it by block replication with its rank, and translate it so the top-left cell of the scaled anchor block lands on the singleton marker of the same color.
    - Clear the original source area and the consumed markers, then draw the scaled components onto a blank output in descending rank order so smaller-ranked stamps overwrite larger-ranked ones on overlaps.

    Key insight:
    - The rare anchor color gives the destination match and anchor position, while the component's area rank independently controls the scale, so both ranking and color matching are required.
    """
    g = np.asarray(grid, dtype=int)
    bg, sources, markers = _parse_components(g)
    ordered = sorted(sources, key=lambda src: int(src["area"]))

    out = np.full_like(g, bg)
    placements = []
    for rank, src in enumerate(ordered, start=1):
        scaled = scale_sprite(src["patch"], rank)
        ar, ac = src["anchor_rel"]
        marker_r, marker_c = markers[src["anchor_color"]]
        top = int(marker_r - ar * rank)
        left = int(marker_c - ac * rank)
        sh, sw = scaled.shape
        assert 0 <= top and 0 <= left and top + sh <= g.shape[0] and left + sw <= g.shape[1]
        placements.append((rank, scaled, top, left))

    for _rank, scaled, top, left in sorted(placements, key=lambda item: int(item[0]), reverse=True):
        paste(out, scaled, top, left, transparent=bg)
    return out


def _normalize(coords: set[tuple[int, int]]) -> np.ndarray:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    arr = np.array(sorted((r - min_r, c - min_c) for r, c in coords), dtype=int)
    return arr


def _transform_coords(coords: np.ndarray, rotate: int, flip: bool) -> np.ndarray:
    pts = coords.copy()
    for _ in range(rotate % 4):
        pts = np.array([(c, -r) for r, c in pts], dtype=int)
    if flip:
        pts = np.array([(r, -c) for r, c in pts], dtype=int)
    return _normalize({(int(r), int(c)) for r, c in pts})


def _random_shape(area: int) -> np.ndarray:
    for _ in range(80):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = []
            for r, c in sorted(cells):
                for nr, nc in neighbors4(r, c):
                    if (nr, nc) not in cells:
                        frontier.append((nr, nc))
            cells.add(random.choice(frontier))
        coords = _normalize(cells)
        coords = _transform_coords(coords, random.randrange(4), bool(random.randrange(2)))
        if len({tuple(rc) for rc in coords}) != area:
            continue
        box = bbox_coords(coords)
        h = box[2] - box[0] + 1
        w = box[3] - box[1] + 1
        if h <= 6 and w <= 6:
            return coords
    raise RuntimeError("failed to build a connected shape")


def _build_patch(shape: np.ndarray, body_color: int, anchor_color: int) -> tuple[np.ndarray, tuple[int, int]]:
    box = bbox_coords(shape)
    h = box[2] - box[0] + 1
    w = box[3] - box[1] + 1
    patch = np.zeros((h, w), dtype=int)
    cells = [tuple(map(int, rc)) for rc in shape]
    anchor = random.choice(cells)
    for r, c in cells:
        patch[r, c] = body_color
    patch[anchor] = anchor_color
    return patch, anchor


def _coords_from_patch(patch: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    coords = set()
    for r, c in np.argwhere(patch != BG):
        coords.add((top + int(r), left + int(c)))
    return coords


def _can_place_source(
    occupied: set[tuple[int, int]],
    patch: np.ndarray,
    top: int,
    left: int,
    grid_shape: tuple[int, int],
    source_w: int,
) -> bool:
    patch_h, patch_w = patch.shape
    if top < 0 or left < 0 or top + patch_h > grid_shape[0] or left + patch_w > source_w:
        return False
    for r, c in np.argwhere(patch != BG):
        rr = top + int(r)
        cc = left + int(c)
        for nr, nc in [(rr, cc), (rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1), (rr - 1, cc - 1), (rr - 1, cc + 1), (rr + 1, cc - 1), (rr + 1, cc + 1)]:
            if (nr, nc) in occupied:
                return False
    return True


def _place_sources(specs: list[dict], grid_shape: tuple[int, int], source_w: int) -> bool:
    occupied: set[tuple[int, int]] = set()
    for spec in sorted(specs, key=lambda item: int(item["patch"].shape[0] * item["patch"].shape[1]), reverse=True):
        patch = spec["patch"]
        h, w = patch.shape
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = random.randint(1, grid_shape[0] - h - 1)
            left = random.randint(1, source_w - w - 1)
            if not _can_place_source(occupied, patch, top, left, grid_shape, source_w):
                continue
            spec["source_top_left"] = (top, left)
            spec["source_cells"] = _coords_from_patch(patch, top, left)
            occupied |= spec["source_cells"]
            placed = True
            break
        if not placed:
            return False
    return True


def _overlap_count(existing: np.ndarray, patch: np.ndarray, top: int, left: int) -> int:
    h, w = patch.shape
    mask = patch != BG
    return int(np.sum(existing[top : top + h, left : left + w] & mask))


def _pick_target_placements(specs: list[dict], grid_shape: tuple[int, int], target_left: int, allow_overlap: bool) -> bool:
    h, w = grid_shape
    occupied = np.zeros(grid_shape, dtype=bool)
    marker_positions: list[tuple[int, int]] = []
    overlap_rank = random.choice((1, 2)) if allow_overlap else None

    ordered = sorted(specs, key=lambda item: int(item["rank"]), reverse=True)
    achieved_overlap = False
    for spec in ordered:
        patch = spec["scaled_patch"]
        sh, sw = patch.shape
        ar, ac = spec["anchor_rel"]
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            if target_left > w - sw:
                return False
            top = random.randint(0, h - sh)
            left = random.randint(target_left, w - sw)
            marker = (top + ar * spec["rank"], left + ac * spec["rank"])
            if any(abs(marker[0] - mr) + abs(marker[1] - mc) <= 1 for mr, mc in marker_positions):
                continue
            overlap = _overlap_count(occupied, patch, top, left)
            if spec["rank"] == overlap_rank:
                max_overlap = max(1, spec["rank"])
                if not (1 <= overlap <= max_overlap):
                    continue
            elif overlap != 0:
                continue

            spec["stamp_top_left"] = (top, left)
            spec["marker_pos"] = marker
            marker_positions.append(marker)
            occupied[top : top + sh, left : left + sw] |= patch != BG
            achieved_overlap = achieved_overlap or overlap > 0
            placed = True
            break
        if not placed:
            return False

    if allow_overlap and not achieved_overlap:
        return False
    if not allow_overlap and achieved_overlap:
        return False
    return True


def _render_input(specs: list[dict], grid_shape: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(grid_shape, dtype=int)
    for spec in specs:
        top, left = spec["source_top_left"]
        paste(grid, spec["patch"], top, left, transparent=BG)
    for spec in specs:
        r, c = spec["marker_pos"]
        grid[r, c] = spec["anchor_color"]
    return grid


def _render_expected(specs: list[dict], grid_shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(grid_shape, dtype=int)
    for spec in sorted(specs, key=lambda item: int(item["rank"]), reverse=True):
        top, left = spec["stamp_top_left"]
        paste(out, spec["scaled_patch"], top, left, transparent=BG)
    return out


def _validate_generated_case(grid: np.ndarray, specs: list[dict], allow_overlap: bool) -> None:
    assert grid.ndim == 2 and max(grid.shape) <= 30
    bg, sources, markers = _parse_components(grid)
    assert bg == BG

    spec_by_anchor = {int(spec["anchor_color"]): spec for spec in specs}
    assert set(markers) == set(spec_by_anchor)
    parsed_areas = sorted(src["area"] for src in sources)
    expected_areas = sorted(int(spec["area"]) for spec in specs)
    assert parsed_areas == expected_areas

    for src in sources:
        spec = spec_by_anchor[int(src["anchor_color"])]
        parsed_cells = {tuple(map(int, rc)) for rc in src["coords"]}
        assert parsed_cells == spec["source_cells"]
        assert tuple(map(int, markers[src["anchor_color"]])) == tuple(spec["marker_pos"])

    expected = _render_expected(specs, grid.shape)
    actual = main(grid)
    assert np.array_equal(actual, expected)

    overlap_pixels = 0
    rank_to_mask = {}
    for spec in specs:
        mask = np.zeros(grid.shape, dtype=bool)
        top, left = spec["stamp_top_left"]
        patch = spec["scaled_patch"]
        h, w = patch.shape
        mask[top : top + h, left : left + w] = patch != BG
        rank_to_mask[int(spec["rank"])] = mask
    ranks = sorted(rank_to_mask)
    for i, ra in enumerate(ranks):
        for rb in ranks[i + 1 :]:
            overlap_pixels += int(np.sum(rank_to_mask[ra] & rank_to_mask[rb]))
    if allow_overlap:
        assert overlap_pixels > 0
    else:
        assert overlap_pixels == 0


def _gen_attempt() -> np.ndarray | None:
    areas = random.sample(range(3, 7), 3)
    body_colors = list(SOURCE_BODY_COLORS)
    anchor_colors = list(ANCHOR_COLORS)
    random.shuffle(body_colors)
    random.shuffle(anchor_colors)

    specs = []
    for area, body_color, anchor_color in zip(areas, body_colors, anchor_colors):
        shape = _random_shape(area)
        patch, anchor_rel = _build_patch(shape, body_color, anchor_color)
        specs.append(
            {
                "area": int(area),
                "body_color": int(body_color),
                "anchor_color": int(anchor_color),
                "patch": patch,
                "anchor_rel": tuple(map(int, anchor_rel)),
            }
        )

    ranked = sorted(specs, key=lambda item: int(item["area"]))
    for rank, spec in enumerate(ranked, start=1):
        spec["rank"] = int(rank)
        spec["scaled_patch"] = scale_sprite(spec["patch"], rank)

    max_unscaled_w = max(spec["patch"].shape[1] for spec in specs)
    max_scaled_w = max(spec["scaled_patch"].shape[1] for spec in specs)
    max_scaled_h = max(spec["scaled_patch"].shape[0] for spec in specs)
    source_w_min = max(8, max_unscaled_w + 3)
    source_w_max = min(10, 30 - max_scaled_w - 5)
    if source_w_min > source_w_max:
        return None
    source_w = random.randint(source_w_min, source_w_max)
    gap = random.randint(2, 4)
    target_left = source_w + gap

    h_min = max(14, max_scaled_h + 2)
    h_max = min(24, 30)
    if h_min > h_max:
        return None
    grid_h = random.randint(h_min, h_max)

    w_min = target_left + max_scaled_w + 1
    w_max = 30
    if w_min > w_max:
        return None
    w_lo = max(18, w_min)
    w_hi = min(28, w_max)
    if w_lo > w_hi:
        return None
    grid_w = random.randint(w_lo, w_hi)
    grid_shape = (grid_h, grid_w)

    if not _place_sources(specs, grid_shape, source_w):
        return None

    allow_overlap = random.random() < OVERLAP_PROBABILITY
    if not _pick_target_placements(specs, grid_shape, target_left, allow_overlap):
        return None

    grid = _render_input(specs, grid_shape)
    try:
        _validate_generated_case(grid, specs, allow_overlap)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 11")
