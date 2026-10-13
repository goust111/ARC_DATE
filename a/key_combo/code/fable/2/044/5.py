import numpy as np


BLACK = 0
YELLOW = 4
PALETTE = tuple(range(10))
TEMPLATE_BODY_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)
CORNER_NAMES = ("tl", "tr", "bl", "br")
CORNER_OFFSETS = {
    "tl": (0, 0),
    "tr": (0, -1),
    "bl": (-1, 0),
    "br": (-1, -1),
}
RNG = np.random.default_rng()
MAX_GENERATION_ATTEMPTS = 500


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid, dtype=int)[r0 : r1 + 1, c0 : c1 + 1].copy()


def scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    k = int(factor)
    assert k >= 1
    return np.kron(np.asarray(sprite, dtype=int), np.ones((k, k), dtype=int))


def _neighbors8(shape: tuple[int, int], r: int, c: int):
    h, w = shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                yield nr, nc


def _foreground_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(g[r, c]) == bg:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            colors: set[int] = set()

            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                colors.add(int(g[rr, cc]))
                for nr, nc in _neighbors8((h, w), rr, cc):
                    if visited[nr, nc] or int(g[nr, nc]) == bg:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            components.append(
                {
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                    "colors": colors,
                }
            )

    return components


def _feature_corner_name(tile: np.ndarray, feature_color: int) -> str:
    h, w = tile.shape
    assert h == w
    matches = []
    for name, (dr, dc) in CORNER_OFFSETS.items():
        rr = 0 if dr == 0 else h - 1
        cc = 0 if dc == 0 else w - 1
        if int(tile[rr, cc]) == int(feature_color):
            matches.append(name)
    assert len(matches) == 1, "template must have exactly one feature-colored corner"
    return matches[0]


def _rotate_to_corner(tile: np.ndarray, feature_color: int, target_corner: str) -> np.ndarray:
    for k in range(4):
        rotated = np.rot90(tile, k)
        if _feature_corner_name(rotated, feature_color) == target_corner:
            return rotated
    raise AssertionError("no rotation aligns the feature corner")


def _boxes_overlap(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _extract_instance(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    components = _foreground_components(g, bg)
    assert components, "expected a template and markers"

    template_candidates = []
    markers = []
    for comp in components:
        r0, c0, r1, c1 = comp["bbox"]
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        if comp["area"] > 1 and height == width and len(comp["colors"]) >= 2:
            template_candidates.append(comp)
        else:
            markers.append(comp)

    assert len(template_candidates) == 1, "expected exactly one square multicolor template"
    assert markers, "expected at least one viewport marker group"
    assert all(comp["area"] == 1 for comp in markers), "viewport markers must be isolated pixels"

    template_comp = template_candidates[0]
    template = crop_bbox(g, template_comp["bbox"])
    template_size = int(template.shape[0])
    assert template.shape == (template_size, template_size)

    marker_map: dict[tuple[int, int], int] = {}
    for comp in markers:
        r, c = map(int, comp["coords"][0])
        marker_map[(r, c)] = int(g[r, c])

    assert len(marker_map) % 4 == 0, "markers must come in groups of four"
    marker_colors = set(marker_map.values())
    assert YELLOW in marker_colors and len(marker_colors) == 2, "markers must use yellow plus one feature color"
    feature_color = next(color for color in marker_colors if color != YELLOW)
    template_feature_corner = _feature_corner_name(template, feature_color)

    rows = sorted({r for r, _ in marker_map})
    cols = sorted({c for _, c in marker_map})
    candidates = []

    for i, r0 in enumerate(rows):
        for r1 in rows[i + 1 :]:
            side = int(r1 - r0 - 1)
            if side <= 0 or side % template_size != 0:
                continue
            for j, c0 in enumerate(cols):
                for c1 in cols[j + 1 :]:
                    if int(c1 - c0 - 1) != side:
                        continue

                    corners = {
                        "tl": (r0, c0),
                        "tr": (r0, c1),
                        "bl": (r1, c0),
                        "br": (r1, c1),
                    }
                    if not all(pos in marker_map for pos in corners.values()):
                        continue

                    colors = {name: marker_map[pos] for name, pos in corners.items()}
                    if sum(int(color == feature_color) for color in colors.values()) != 1:
                        continue
                    if sum(int(color == YELLOW) for color in colors.values()) != 3:
                        continue

                    region = g[r0 + 1 : r1, c0 + 1 : c1]
                    if region.size == 0 or np.any(region != bg):
                        continue

                    odd_corner = next(name for name, color in colors.items() if color == feature_color)
                    candidates.append(
                        {
                            "outer_bbox": (int(r0), int(c0), int(r1), int(c1)),
                            "interior_bbox": (int(r0 + 1), int(c0 + 1), int(r1 - 1), int(c1 - 1)),
                            "markers": frozenset(corners.values()),
                            "odd_corner": odd_corner,
                            "scale": int(side // template_size),
                        }
                    )

    assert candidates, "no valid viewport rectangles found"

    all_markers = frozenset(marker_map)
    by_marker = {marker: [] for marker in all_markers}
    for idx, cand in enumerate(candidates):
        for marker in cand["markers"]:
            by_marker[marker].append(idx)

    solutions: list[tuple[int, ...]] = []

    def backtrack(remaining: frozenset[tuple[int, int]], chosen: list[int]) -> None:
        if len(solutions) > 1:
            return
        if not remaining:
            solutions.append(tuple(chosen))
            return

        def viable_indices(marker: tuple[int, int]) -> list[int]:
            out = []
            for idx in by_marker[marker]:
                cand = candidates[idx]
                if not cand["markers"].issubset(remaining):
                    continue
                if any(_boxes_overlap(cand["interior_bbox"], candidates[j]["interior_bbox"]) for j in chosen):
                    continue
                out.append(idx)
            return out

        pivot = min(remaining, key=lambda marker: len(viable_indices(marker)))
        options = viable_indices(pivot)
        if not options:
            return

        for idx in options:
            chosen.append(idx)
            backtrack(remaining.difference(candidates[idx]["markers"]), chosen)
            chosen.pop()

    backtrack(all_markers, [])
    assert len(solutions) == 1, "viewport grouping must be uniquely determined"

    groups = [candidates[idx] for idx in solutions[0]]
    groups.sort(key=lambda grp: grp["outer_bbox"])

    scales = [grp["scale"] for grp in groups]
    assert len(set(scales)) == len(scales), "viewport scales must be distinct"

    return {
        "bg": bg,
        "template": template,
        "template_bbox": template_comp["bbox"],
        "template_size": template_size,
        "feature_color": feature_color,
        "template_feature_corner": template_feature_corner,
        "groups": groups,
    }


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    info = _extract_instance(g)
    out = g.copy()

    for group in info["groups"]:
        rotated = _rotate_to_corner(info["template"], info["feature_color"], group["odd_corner"])
        scaled = scale_sprite(rotated, group["scale"])
        r0, c0, r1, c1 = group["interior_bbox"]
        assert scaled.shape == (r1 - r0 + 1, c1 - c0 + 1)
        assert np.all(out[r0 : r1 + 1, c0 : c1 + 1] == info["bg"])
        out[r0 : r1 + 1, c0 : c1 + 1] = scaled

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each yellow-corner viewport with a rotated and integer-upscaled copy of the square template.

    Input structure:
    - A black background contains one multicolor square template sprite.
    - Exactly one template corner uses the same feature color that also appears on viewport markers.
    - Several hidden square viewports are marked only by four isolated corner pixels: three yellow corners and one feature-colored corner.

    Rule:
    - Recover each viewport from the two marker rows and two marker columns that form its outer rectangle.
    - The viewport interior is the square region strictly inside those four markers, and its side length is an integer multiple of the template side length.
    - Rotate the template so its feature-colored corner lands in the same corner position as the viewport's odd-colored marker, then enlarge the rotated template by that integer factor and write it into the viewport interior.
    - Leave the original template sprite and all corner markers unchanged.

    Key insight:
    - Each marker quartet encodes three independent facts at once: where the hidden viewport is, how much the template must be scaled to fit exactly, and which of the four rotations to use.
    """
    return _transform(grid)


def _rotations_are_distinct(tile: np.ndarray) -> bool:
    signatures = {tuple(np.rot90(tile, k).flatten()) for k in range(4)}
    return len(signatures) == 4


def _make_template(size: int, feature_color: int, feature_corner: str) -> np.ndarray:
    body_palette = [color for color in TEMPLATE_BODY_COLORS if color != feature_color]
    for _ in range(300):
        colors = RNG.choice(body_palette, size=2, replace=False)
        body = RNG.choice(colors, size=(size, size))
        if len(set(map(int, body.flatten()))) < 2:
            continue
        if not _rotations_are_distinct(body):
            continue

        tile = body.copy()
        rr = 0 if feature_corner[0] == "t" else size - 1
        cc = 0 if feature_corner[1] == "l" else size - 1
        tile[rr, cc] = int(feature_color)

        if len(set(map(int, tile.flatten()))) < 3:
            continue
        if _feature_corner_name(tile, feature_color) != feature_corner:
            continue
        return tile
    raise RuntimeError("failed to create a sufficiently asymmetric template")


def _expanded_overlap(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _sample_rect(
    h: int,
    w: int,
    size: int,
    occupied: list[tuple[int, int, int, int]],
    used_rows: set[int],
    used_cols: set[int],
    enforce_unique_marker_axes: bool,
) -> tuple[int, int, int, int] | None:
    for _ in range(400):
        r0 = int(RNG.integers(0, h - size + 1))
        c0 = int(RNG.integers(0, w - size + 1))
        box = (r0, c0, r0 + size - 1, c0 + size - 1)
        if any(_expanded_overlap(box, other, pad=1) for other in occupied):
            continue
        if enforce_unique_marker_axes:
            if r0 in used_rows or box[2] in used_rows or c0 in used_cols or box[3] in used_cols:
                continue
        return box
    return None


def _render_expected(
    grid: np.ndarray,
    template: np.ndarray,
    feature_color: int,
    groups: list[dict],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for group in groups:
        r0, c0, r1, c1 = group["outer_bbox"]
        rotated = _rotate_to_corner(template, feature_color, group["odd_corner"])
        scaled = scale_sprite(rotated, group["scale"])
        assert scaled.shape == (r1 - r0 - 1, c1 - c0 - 1)
        out[r0 + 1 : r1, c0 + 1 : c1] = scaled
    return out


def _render_without_rotation(
    grid: np.ndarray,
    template: np.ndarray,
    groups: list[dict],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for group in groups:
        r0, c0, r1, c1 = group["outer_bbox"]
        scaled = scale_sprite(template, group["scale"])
        out[r0 + 1 : r1, c0 + 1 : c1] = scaled
    return out


def _generate_attempt() -> np.ndarray | None:
    template_size = int(RNG.choice([3, 4]))
    max_group_count = 3
    scale_pool = [1, 2, 3] if template_size == 4 else [1, 2, 3, 4]
    group_count = 3 if (template_size == 3 and float(RNG.random()) < 0.55) else 2 + int(float(RNG.random()) < 0.35)
    group_count = min(group_count, max_group_count, len(scale_pool))

    if group_count == 3:
        usable_scales = scale_pool[:3]
    else:
        usable_scales = scale_pool
    scales = sorted(int(x) for x in RNG.choice(usable_scales, size=group_count, replace=False))

    corners = list(RNG.choice(np.array(CORNER_NAMES), size=group_count, replace=False))
    feature_corner = str(RNG.choice(np.array(CORNER_NAMES)))
    if all(corner == feature_corner for corner in corners):
        corners[0] = next(name for name in CORNER_NAMES if name != feature_corner)
    feature_color = int(RNG.choice(np.array(TEMPLATE_BODY_COLORS)))
    template = _make_template(template_size, feature_color, feature_corner)

    outer_sizes = [template_size * scale + 2 for scale in scales]
    h = int(RNG.integers(max(22, max(outer_sizes) + 4), 31))
    w = int(RNG.integers(max(22, max(outer_sizes) + 4), 31))
    grid = np.full((h, w), BLACK, dtype=int)

    occupied: list[tuple[int, int, int, int]] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()

    group_specs = [
        {"scale": scale, "odd_corner": corner, "size": template_size * scale + 2}
        for scale, corner in zip(scales, corners)
    ]
    placed_groups: list[dict] = []

    for spec in sorted(group_specs, key=lambda item: item["size"], reverse=True):
        box = _sample_rect(h, w, spec["size"], occupied, used_rows, used_cols, True)
        if box is None:
            return None
        occupied.append(box)
        used_rows.update({box[0], box[2]})
        used_cols.update({box[1], box[3]})
        placed_groups.append(
            {
                "scale": int(spec["scale"]),
                "odd_corner": str(spec["odd_corner"]),
                "outer_bbox": box,
            }
        )

    template_box = _sample_rect(h, w, template_size, occupied, used_rows, used_cols, False)
    if template_box is None:
        return None
    occupied.append(template_box)

    tr0, tc0, tr1, tc1 = template_box
    grid[tr0 : tr1 + 1, tc0 : tc1 + 1] = template

    for group in placed_groups:
        r0, c0, r1, c1 = group["outer_bbox"]
        colors = {"tl": YELLOW, "tr": YELLOW, "bl": YELLOW, "br": YELLOW}
        colors[group["odd_corner"]] = feature_color
        grid[r0, c0] = colors["tl"]
        grid[r0, c1] = colors["tr"]
        grid[r1, c0] = colors["bl"]
        grid[r1, c1] = colors["br"]

    expected = _render_expected(grid, template, feature_color, placed_groups)

    if np.array_equal(expected, grid):
        return None
    if not any(group["odd_corner"] != feature_corner for group in placed_groups):
        return None
    if np.array_equal(_render_without_rotation(grid, template, placed_groups), expected):
        return None

    try:
        parsed = _extract_instance(grid)
    except AssertionError:
        return None

    if len(parsed["groups"]) != group_count:
        return None
    if parsed["template_feature_corner"] != feature_corner:
        return None
    parsed_groups = sorted(
        [(grp["outer_bbox"], grp["scale"], grp["odd_corner"]) for grp in parsed["groups"]]
    )
    planned_groups = sorted(
        [(grp["outer_bbox"], grp["scale"], grp["odd_corner"]) for grp in placed_groups]
    )
    if parsed_groups != planned_groups:
        return None
    if not np.array_equal(_transform(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid yellow-corner viewport task")
