from __future__ import annotations

import random
from importlib.machinery import SourceFileLoader

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_key_combo_053_5",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()

Color = _ARC_UTILS.Color
crop_bbox = _ARC_UTILS.crop_bbox
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg
paste = _ARC_UTILS.paste


FRAME_COLOR = Color.TEAL
PAIR_COUNTS = (2, 3)
PAIR_COUNT_WEIGHTS = (0.35, 0.65)
BLUE_SPANS = (4, 6, 8)
GRID_SIDE_RANGE = (18, 26)
PLACEMENT_PADDING = 1
GENERATION_ATTEMPTS = 500
MIDPOINT_OFFSETS = (
    (-7, 0),
    (7, 0),
    (0, -7),
    (0, 7),
    (-6, -4),
    (-6, 4),
    (6, -4),
    (6, 4),
    (-4, -6),
    (4, -6),
    (-4, 6),
    (4, 6),
    (-8, 0),
    (8, 0),
    (0, -8),
    (0, 8),
)

GLYPH_MASKS = {
    "plus": np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    "tee": np.array(
        [
            [1, 1, 1],
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    "stair": np.array(
        [
            [1, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
        ],
        dtype=bool,
    ),
}
GLYPH_TO_COLOR = {
    "plus": Color.RED,
    "tee": Color.YELLOW,
    "stair": Color.PINK,
}

_GLYPH_SIGNATURE_TO_COLOR = {
    tuple(map(tuple, np.argwhere(mask))): int(GLYPH_TO_COLOR[name])
    for name, mask in GLYPH_MASKS.items()
}

FRAME_PATCH = np.full((5, 5), FRAME_COLOR, dtype=int)
FRAME_PATCH[1:4, 1:4] = Color.BLACK


def _sqdist(a: tuple[int, int], b: tuple[int, int]) -> int:
    dr = int(a[0] - b[0])
    dc = int(a[1] - b[1])
    return dr * dr + dc * dc


def _glyph_signature_from_patch(patch: np.ndarray) -> tuple[tuple[int, int], ...]:
    assert patch.shape == (3, 3), "glyph patch must be 3x3"
    values = set(map(int, np.unique(patch)))
    assert values <= {Color.BLACK, Color.GREY}, "glyph patch must contain only black and grey"
    coords = tuple(map(tuple, np.argwhere(patch == Color.GREY)))
    assert len(coords) == 5, "glyph patch must contain exactly five grey pixels"
    return coords


def _decode_glyph_patch(patch: np.ndarray) -> int:
    sig = _glyph_signature_from_patch(patch)
    assert sig in _GLYPH_SIGNATURE_TO_COLOR, "unknown glyph signature"
    return int(_GLYPH_SIGNATURE_TO_COLOR[sig])


def _find_regions(grid: np.ndarray) -> list[dict]:
    regions = find_components(
        grid,
        bg=Color.BLACK,
        colors=[FRAME_COLOR],
        connectivity=8,
        monochromatic=True,
        return_format="dict",
    )
    out: list[dict] = []
    for region in regions:
        r0, c0, r1, c1 = map(int, region["bbox"])
        if (r1 - r0 + 1, c1 - c0 + 1) != (5, 5):
            continue
        patch = crop_bbox(grid, region["bbox"])
        border = np.ones((5, 5), dtype=bool)
        border[1:4, 1:4] = False
        if not np.all(patch[border] == FRAME_COLOR):
            continue
        if np.any(patch[1:4, 1:4] == FRAME_COLOR):
            continue
        out.append(
            {
                "bbox": (r0, c0, r1, c1),
                "center": (r0 + 2, c0 + 2),
                "patch": patch,
            }
        )
    assert len(out) in PAIR_COUNTS, "expected two or three framed regions"
    return out


def _valid_blue_edge(
    grid: np.ndarray,
    a: tuple[int, int],
    b: tuple[int, int],
) -> tuple[int, int] | None:
    ar, ac = map(int, a)
    br, bc = map(int, b)
    same_row = ar == br
    same_col = ac == bc
    if same_row == same_col:
        return None
    span = abs(ac - bc) if same_row else abs(ar - br)
    if span not in BLUE_SPANS:
        return None
    mid_r = (ar + br) // 2
    mid_c = (ac + bc) // 2
    if (ar + br) % 2 or (ac + bc) % 2:
        return None
    if not (1 <= mid_r < grid.shape[0] - 1 and 1 <= mid_c < grid.shape[1] - 1):
        return None
    patch = grid[mid_r - 1 : mid_r + 2, mid_c - 1 : mid_c + 2]
    try:
        _decode_glyph_patch(patch)
    except AssertionError:
        return None
    return int(mid_r), int(mid_c)


def _match_blue_pairs(grid: np.ndarray) -> list[dict]:
    blues = [tuple(map(int, coord)) for coord in np.argwhere(grid == Color.BLUE)]
    assert len(blues) in (4, 6), "expected two or three locator pairs"

    edges: dict[int, list[tuple[int, tuple[int, int]]]] = {i: [] for i in range(len(blues))}
    for i in range(len(blues)):
        for j in range(i + 1, len(blues)):
            midpoint = _valid_blue_edge(grid, blues[i], blues[j])
            if midpoint is None:
                continue
            edges[i].append((j, midpoint))
            edges[j].append((i, midpoint))

    solutions: list[list[tuple[int, int, tuple[int, int]]]] = []

    def search(unused: tuple[int, ...], chosen: list[tuple[int, int, tuple[int, int]]]) -> None:
        if len(solutions) > 1:
            return
        if not unused:
            solutions.append(list(chosen))
            return
        i = unused[0]
        remaining = set(unused[1:])
        for j, midpoint in edges[i]:
            if j not in remaining:
                continue
            next_unused = tuple(v for v in unused[1:] if v != j)
            search(next_unused, chosen + [(i, j, midpoint)])

    search(tuple(range(len(blues))), [])
    assert len(solutions) == 1, "locator pixels must admit a unique pairing"

    result = []
    for i, j, midpoint in solutions[0]:
        patch = grid[midpoint[0] - 1 : midpoint[0] + 2, midpoint[1] - 1 : midpoint[1] + 2]
        result.append(
            {
                "pixels": (blues[i], blues[j]),
                "midpoint": midpoint,
                "color": _decode_glyph_patch(patch),
            }
        )
    return result


def _pair_markers_to_regions(markers: list[dict], regions: list[dict]) -> list[tuple[dict, dict]]:
    assert len(markers) == len(regions), "marker/region counts must match"
    pairs: list[tuple[dict, dict]] = []
    used_regions: set[int] = set()
    for marker in markers:
        distances = [(_sqdist(marker["midpoint"], region["center"]), idx) for idx, region in enumerate(regions)]
        distances.sort()
        assert len(distances) >= 2
        assert distances[0][0] < distances[1][0], "nearest region must be unique"
        idx = int(distances[0][1])
        assert idx not in used_regions, "nearest assignments must form a bijection"
        used_regions.add(idx)
        pairs.append((marker, regions[idx]))
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read each midpoint glyph from a blue locator pair and use its decoded color to complete the nearest framed region.

    Input structure:
    - The grid has a black background with two or three blue locator-pixel pairs.
    - Each pair is horizontal or vertical, and its midpoint centers a 3x3 grey glyph made of exactly five grey pixels.
    - Two or three 5x5 teal frames appear elsewhere; each frame surrounds a 3x3 black interior that may already contain some pixels of its target color.

    Rule:
    - Pair the blue pixels, compute each midpoint, and read the 3x3 grey glyph centered there.
    - Decode the glyph with the fixed glyph-to-color codebook shared by the task.
    - Match each glyph midpoint to its uniquely nearest frame center by Euclidean distance.
    - Inside the matched frame, recolor every still-black interior cell to the decoded color while preserving any existing clue pixels, the teal frame, and the blue/grey source objects.

    Key insight:
    - The midpoint is a read location rather than a write location: it identifies a glyph whose color meaning must be transferred to a distant frame selected by nearest-center pairing.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert int(get_bg(g)) == Color.BLACK

    regions = _find_regions(g)
    markers = _match_blue_pairs(g)
    pairs = _pair_markers_to_regions(markers, regions)

    out = g.copy()
    for marker, region in pairs:
        color = int(marker["color"])
        r0, c0, _r1, _c1 = region["bbox"]
        interior = g[r0 + 1 : r0 + 4, c0 + 1 : c0 + 4]
        seen = {int(v) for v in interior.flatten() if int(v) != Color.BLACK}
        assert seen <= {color}, "frame clue pixels must agree with the decoded glyph color"
        fill_mask = interior == Color.BLACK
        out[r0 + 1 : r0 + 4, c0 + 1 : c0 + 4][fill_mask] = color

    return out


def _expand_cells(cells: set[tuple[int, int]], shape: tuple[int, int], radius: int = PLACEMENT_PADDING) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _region_cells(top_left: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = map(int, top_left)
    return {(r0 + dr, c0 + dc) for dr in range(5) for dc in range(5)}


def _marker_cells(midpoint: tuple[int, int], orient: str, span: int) -> set[tuple[int, int]]:
    mr, mc = map(int, midpoint)
    cells = {(mr + dr, mc + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
    if orient == "h":
        cells.add((mr, mc - span // 2))
        cells.add((mr, mc + span // 2))
    else:
        cells.add((mr - span // 2, mc))
        cells.add((mr + span // 2, mc))
    return cells


def _sample_region_positions(height: int, width: int, count: int) -> list[tuple[int, int]] | None:
    candidates = [(r, c) for r in range(height - 4) for c in range(width - 4)]
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    blocked: set[tuple[int, int]] = set()
    centers: list[tuple[int, int]] = []
    for r0, c0 in candidates:
        cells = _region_cells((r0, c0))
        if cells & blocked:
            continue
        center = (r0 + 2, c0 + 2)
        if any(_sqdist(center, other) <= 80 for other in centers):
            continue
        chosen.append((r0, c0))
        centers.append(center)
        blocked |= _expand_cells(cells, (height, width))
        if len(chosen) == count:
            return chosen
    return None


def _sample_marker_spec(
    center: tuple[int, int],
    other_centers: list[tuple[int, int]],
    blocked: set[tuple[int, int]],
    shape: tuple[int, int],
) -> tuple[tuple[int, int], str, int] | None:
    height, width = shape
    offsets = list(MIDPOINT_OFFSETS)
    spans = list(BLUE_SPANS)
    orients = ["h", "v"]
    random.shuffle(offsets)
    for dr, dc in offsets:
        midpoint = (int(center[0] + dr), int(center[1] + dc))
        random.shuffle(orients)
        random.shuffle(spans)
        for orient in orients:
            for span in spans:
                cells = _marker_cells(midpoint, orient, span)
                if any(not (0 <= r < height and 0 <= c < width) for r, c in cells):
                    continue
                if _expand_cells(cells, shape) & blocked:
                    continue
                own = _sqdist(midpoint, center)
                others = [_sqdist(midpoint, other) for other in other_centers]
                if not all(own < other for other in others):
                    continue
                if others and min(others) - own < 8:
                    continue
                return midpoint, orient, span
    return None


def _choose_glyphs(count: int) -> list[str]:
    names = list(GLYPH_MASKS)
    if count == 3:
        random.shuffle(names)
        return names
    assert count == 2
    return random.sample(names, 2)


def _choose_clue_cells(count: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(3) for c in range(3)]
    return random.sample(cells, count)


def _render_input(relations: list[dict], shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, Color.BLACK, dtype=int)
    for rel in relations:
        paste(grid, FRAME_PATCH, rel["region_top_left"][0], rel["region_top_left"][1])
        color = int(GLYPH_TO_COLOR[rel["glyph"]])
        for rr, cc in rel["clues"]:
            grid[rel["region_top_left"][0] + 1 + rr, rel["region_top_left"][1] + 1 + cc] = color
        glyph_patch = np.full((3, 3), Color.BLACK, dtype=int)
        glyph_patch[GLYPH_MASKS[rel["glyph"]]] = Color.GREY
        paste(grid, glyph_patch, rel["midpoint"][0] - 1, rel["midpoint"][1] - 1, transparent=Color.BLACK)
        mr, mc = rel["midpoint"]
        half = rel["span"] // 2
        if rel["orient"] == "h":
            grid[mr, mc - half] = Color.BLUE
            grid[mr, mc + half] = Color.BLUE
        else:
            grid[mr - half, mc] = Color.BLUE
            grid[mr + half, mc] = Color.BLUE
    return grid


def _expected_output(inp: np.ndarray, relations: list[dict]) -> np.ndarray:
    out = inp.copy()
    for rel in relations:
        r0, c0 = rel["region_top_left"]
        color = int(GLYPH_TO_COLOR[rel["glyph"]])
        interior = out[r0 + 1 : r0 + 4, c0 + 1 : c0 + 4]
        interior[interior == Color.BLACK] = color
    return out


def _nearest_region_indices(relations: list[dict]) -> list[int]:
    centers = [(r0 + 2, c0 + 2) for r0, c0 in [rel["region_top_left"] for rel in relations]]
    indices = []
    for rel in relations:
        midpoint = rel["midpoint"]
        dists = sorted((_sqdist(midpoint, center), idx) for idx, center in enumerate(centers))
        if len(dists) < 2 or dists[0][0] == dists[1][0]:
            return []
        indices.append(int(dists[0][1]))
    return indices


def _semantic_instance_ok(relations: list[dict], shape: tuple[int, int]) -> bool:
    if len(relations) not in PAIR_COUNTS:
        return False
    if sum(1 for rel in relations if not rel["clues"]) != 1:
        return False
    if len({rel["glyph"] for rel in relations}) != len(relations):
        return False

    nearest = _nearest_region_indices(relations)
    if len(nearest) != len(relations) or len(set(nearest)) != len(relations):
        return False
    for idx, rel in enumerate(relations):
        if nearest[idx] != idx:
            return False

    inp = _render_input(relations, shape)
    out = _expected_output(inp, relations)
    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, out)


def _build_relations(height: int, width: int, count: int) -> list[dict] | None:
    region_positions = _sample_region_positions(height, width, count)
    if region_positions is None:
        return None

    relations = []
    blocked: set[tuple[int, int]] = set()
    centers = [(r0 + 2, c0 + 2) for r0, c0 in region_positions]
    glyphs = _choose_glyphs(count)
    clueless_index = random.randrange(count)

    for idx, top_left in enumerate(region_positions):
        blocked |= _expand_cells(_region_cells(top_left), (height, width))

    for idx, (top_left, glyph) in enumerate(zip(region_positions, glyphs)):
        center = centers[idx]
        others = [other for j, other in enumerate(centers) if j != idx]
        marker_spec = _sample_marker_spec(center, others, blocked, (height, width))
        if marker_spec is None:
            return None
        midpoint, orient, span = marker_spec
        blocked |= _expand_cells(_marker_cells(midpoint, orient, span), (height, width))
        clue_count = 0 if idx == clueless_index else random.randint(1, 4)
        relations.append(
            {
                "region_top_left": top_left,
                "glyph": glyph,
                "midpoint": midpoint,
                "orient": orient,
                "span": span,
                "clues": _choose_clue_cells(clue_count),
            }
        )

    return relations


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        height = random.randint(*GRID_SIDE_RANGE)
        width = random.randint(*GRID_SIDE_RANGE)
        count = random.choices(PAIR_COUNTS, weights=PAIR_COUNT_WEIGHTS, k=1)[0]
        relations = _build_relations(height, width, count)
        if relations is None:
            continue
        if not _semantic_instance_ok(relations, (height, width)):
            continue
        return _render_input(relations, (height, width))
    raise RuntimeError("failed to generate a valid midpoint-glyph frame-fill instance")
