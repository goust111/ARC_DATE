from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
PALETTE = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400
SCENARIOS = ("count_overlap", "row_overlap", "separate")

# Token 1 is the dominant color; larger tokens are accent placeholders.
SPRITE_TEMPLATES = (
    np.array([[1, 2, 0], [1, 1, 3]], dtype=int),
    np.array([[1, 0], [2, 1], [1, 3]], dtype=int),
    np.array([[1, 1, 2], [3, 1, 0]], dtype=int),
    np.array([[1, 2], [1, 1], [3, 1]], dtype=int),
    np.array([[1, 2, 1], [0, 1, 3]], dtype=int),
)

# Only row-major-forward steps are used so the sorted markers define the direction.
STEP_OPTIONS = (
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 2),
    (1, -2),
    (1, 3),
    (1, -3),
    (2, 0),
    (2, 1),
    (2, -1),
    (2, 2),
    (2, -2),
    (3, 0),
    (3, 1),
    (3, -1),
)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _crop_patch(grid: np.ndarray, coords: list[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _dominant_color(patch: np.ndarray) -> int:
    counts = Counter(int(v) for v in patch.ravel() if int(v) != 0)
    assert counts, "sprite patch must contain non-background pixels"
    ranked = counts.most_common()
    if len(ranked) > 1:
        assert ranked[0][1] > ranked[1][1], "sprite dominant color must be unique"
    return int(ranked[0][0])


def _stamp_cells(patch: np.ndarray, origin: tuple[int, int]) -> dict[tuple[int, int], int]:
    r0, c0 = origin
    cells: dict[tuple[int, int], int] = {}
    for r, c in np.argwhere(patch != 0):
        key = (int(r0 + r), int(c0 + c))
        cells[key] = int(patch[int(r), int(c)])
    return cells


def _render_group_pixels(patch: np.ndarray, origins: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    cells: dict[tuple[int, int], int] = {}
    for origin in origins:
        part = _stamp_cells(patch, origin)
        if set(cells).intersection(part):
            raise AssertionError("same-group stamps must not overlap")
        cells.update(part)
    return cells


def _group_draw_order(groups: list[dict]) -> list[dict]:
    # Lower priority first; fewer extrapolated points means higher priority and is drawn last.
    return sorted(
        groups,
        key=lambda g: (
            -len(g["stamps"]),
            -g["markers"][0][0],
            -g["markers"][0][1],
        ),
    )


def _render_output_from_groups(groups: list[dict], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for group in _group_draw_order(groups):
        patch = group["patch"]
        for r0, c0 in group["stamps"]:
            for r, c in np.argwhere(patch != 0):
                out[int(r0 + r), int(c0 + c)] = int(patch[int(r), int(c)])
    return out


def _try_render_output_from_groups(groups: list[dict], shape: tuple[int, int]) -> np.ndarray | None:
    h, w = shape
    out = np.zeros(shape, dtype=int)
    for group in _group_draw_order(groups):
        patch = group["patch"]
        ph, pw = patch.shape
        for r0, c0 in group["stamps"]:
            if not (0 <= r0 and 0 <= c0 and r0 + ph <= h and c0 + pw <= w):
                return None
            for r, c in np.argwhere(patch != 0):
                out[int(r0 + r), int(c0 + c)] = int(patch[int(r), int(c)])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extrapolate from each three-point marker line, pick the sprite with the same dominant color, and stamp that sprite at every valid continuation point.

    Input structure:
    - A black background contains exactly two small 4-connected sprite components, each 4 to 6 pixels large, placed in separate corner regions.
    - Each sprite has one unique dominant color that appears more often inside that sprite than any other color.
    - Six extra colored marker pixels form two equally spaced collinear 3-point groups; each group's color matches the dominant color of one sprite.

    Rule:
    - Find the two sprites, crop each to its tight bounding box, and identify its dominant color.
    - For each marker color, sort its three cells in row-major order and use the repeated difference between consecutive cells as the extrapolation step.
    - Starting one step beyond the row-major last marker, keep stepping while that point can serve as the top-left corner of the matching sprite bounding box, and stamp the sprite's non-background pixels at every such anchor.
    - If the two stamped layers overlap, the group with fewer extrapolated anchors is drawn last; if those counts tie, the group whose row-major first marker is higher in the grid is drawn last.
    - Remove the original sprites and the original marker cells; return only the stamped result on the same-size black grid.

    Key insight:
    - Geometry determines where copies go, while color identity determines which sprite gets copied; both signals are needed before the final stamping step is determined.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    comps = _components(g != 0)
    sprite_comps = [comp for comp in comps if 4 <= len(comp) <= 6]
    assert len(sprite_comps) == 2, "expected exactly two sprite components"

    sprite_cells = {cell for comp in sprite_comps for cell in comp}
    marker_cells = [
        cell
        for comp in comps
        if comp not in sprite_comps
        for cell in comp
    ]
    assert len(marker_cells) == 6, "expected exactly six marker pixels"

    sprites: list[dict] = []
    by_main_color: dict[int, dict] = {}
    for comp in sprite_comps:
        patch = _crop_patch(g, comp)
        dom = _dominant_color(patch)
        sprite = {"patch": patch, "dominant_color": dom}
        sprites.append(sprite)
        assert dom not in by_main_color, "sprite dominant colors must be distinct"
        by_main_color[dom] = sprite

    groups_by_color: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)
    for r, c in marker_cells:
        groups_by_color[int(g[r, c])].append((int(r), int(c)))
    assert len(groups_by_color) == 2, "expected exactly two marker colors"

    parsed_groups: list[dict] = []
    for color, cells in groups_by_color.items():
        assert len(cells) == 3, "each marker group must contain three cells"
        assert color in by_main_color, "marker color must match a sprite dominant color"

        markers = sorted(cells)
        step = (
            int(markers[1][0] - markers[0][0]),
            int(markers[1][1] - markers[0][1]),
        )
        assert step != (0, 0), "marker points must be distinct"
        assert (
            int(markers[2][0] - markers[1][0]),
            int(markers[2][1] - markers[1][1]),
        ) == step, "marker points must be equally spaced in row-major order"

        patch = by_main_color[color]["patch"]
        ph, pw = patch.shape
        r, c = markers[-1][0] + step[0], markers[-1][1] + step[1]
        stamps: list[tuple[int, int]] = []
        while 0 <= r and 0 <= c and r + ph <= g.shape[0] and c + pw <= g.shape[1]:
            stamps.append((int(r), int(c)))
            r += step[0]
            c += step[1]
        assert stamps, "each marker group must extrapolate to at least one stamp position"

        parsed_groups.append(
            {
                "markers": markers,
                "step": step,
                "stamps": stamps,
                "patch": patch,
            }
        )

    return _render_output_from_groups(parsed_groups, g.shape)


def _trim_patch(patch: np.ndarray) -> np.ndarray:
    coords = np.argwhere(patch != 0)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return patch[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _random_transform(patch: np.ndarray) -> np.ndarray:
    out = np.rot90(patch, k=random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return _trim_patch(np.asarray(out, dtype=int))


def _instantiate_template(template: np.ndarray, main_color: int, accent_colors: list[int]) -> np.ndarray:
    patch = np.zeros(template.shape, dtype=int)
    patch[template == 1] = main_color
    for token, color in enumerate(accent_colors, start=2):
        patch[template == token] = color
    return patch


def _patch_color_count(template: np.ndarray) -> int:
    return int(max(int(v) for v in np.unique(template) if int(v) != 0))


def _corner_origins(shape: tuple[int, int], patch_a: np.ndarray, patch_b: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    h, w = shape
    corners = random.sample(("tl", "tr", "bl", "br"), 2)

    def place(tag: str, patch: np.ndarray) -> tuple[int, int]:
        ph, pw = patch.shape
        dr = random.randint(0, 2)
        dc = random.randint(0, 2)
        if tag == "tl":
            return 1 + dr, 1 + dc
        if tag == "tr":
            return 1 + dr, w - pw - 1 - dc
        if tag == "bl":
            return h - ph - 1 - dr, 1 + dc
        return h - ph - 1 - dr, w - pw - 1 - dc

    return place(corners[0], patch_a), place(corners[1], patch_b)


def _patch_occupied_cells(patch: np.ndarray, origin: tuple[int, int]) -> set[tuple[int, int]]:
    return set(_stamp_cells(patch, origin))


def _touching(cells_a: set[tuple[int, int]], cells_b: set[tuple[int, int]]) -> bool:
    for r, c in cells_a:
        if (r, c) in cells_b:
            return True
        for dr, dc in DIRS4:
            if (r + dr, c + dc) in cells_b:
                return True
    return False


def _eligible_steps(patch: np.ndarray) -> list[tuple[int, int]]:
    ph, pw = patch.shape
    steps = []
    for dr, dc in STEP_OPTIONS:
        if dr == 0 and abs(dc) >= pw:
            steps.append((dr, dc))
        elif abs(dc) == 0 and dr >= ph:
            steps.append((dr, dc))
        elif dr >= ph or abs(dc) >= pw:
            steps.append((dr, dc))
    return steps


def _stamp_positions(
    shape: tuple[int, int],
    patch_shape: tuple[int, int],
    first: tuple[int, int],
    step: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = shape
    ph, pw = patch_shape
    r, c = first
    out: list[tuple[int, int]] = []
    while 0 <= r and 0 <= c and r + ph <= h and c + pw <= w:
        out.append((int(r), int(c)))
        r += step[0]
        c += step[1]
    return out


def _candidate_groups(
    shape: tuple[int, int],
    patch: np.ndarray,
    count: int,
    step: tuple[int, int],
) -> list[dict]:
    h, w = shape
    ph, pw = patch.shape
    candidates: list[dict] = []
    for r1 in range(h - ph + 1):
        for c1 in range(w - pw + 1):
            stamps = _stamp_positions(shape, patch.shape, (r1, c1), step)
            if len(stamps) != count:
                continue
            markers = [
                (r1 - 3 * step[0], c1 - 3 * step[1]),
                (r1 - 2 * step[0], c1 - 2 * step[1]),
                (r1 - step[0], c1 - step[1]),
            ]
            if len(set(markers)) != 3:
                continue
            if any(not (0 <= r < h and 0 <= c < w) for r, c in markers):
                continue
            if sorted(markers) != markers:
                continue
            try:
                _render_group_pixels(patch, stamps)
            except AssertionError:
                continue
            candidates.append(
                {
                    "markers": [(int(r), int(c)) for r, c in markers],
                    "step": step,
                    "stamps": [(int(r), int(c)) for r, c in stamps],
                    "patch": patch,
                }
            )
    random.shuffle(candidates)
    return candidates


def _render_input(shape: tuple[int, int], sprites: list[dict], groups: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for sprite in sprites:
        origin = sprite["origin"]
        for (r, c), value in _stamp_cells(sprite["patch"], origin).items():
            grid[r, c] = value
    for group in groups:
        for r, c in group["markers"]:
            assert grid[r, c] == 0
            grid[r, c] = group["color"]
    return grid


def _swapped_groups(groups: list[dict]) -> list[dict]:
    assert len(groups) == 2
    swapped = []
    for i, group in enumerate(groups):
        other = groups[1 - i]
        swapped.append(
            {
                "markers": group["markers"],
                "step": group["step"],
                "stamps": group["stamps"],
                "patch": other["patch"],
                "color": group["color"],
            }
        )
    return swapped


def _first_only_groups(groups: list[dict]) -> list[dict]:
    trimmed = []
    for group in groups:
        trimmed.append(
            {
                "markers": group["markers"],
                "step": group["step"],
                "stamps": group["stamps"][:1],
                "patch": group["patch"],
                "color": group["color"],
            }
        )
    return trimmed


def _reversed_order_output(groups: list[dict], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for group in reversed(_group_draw_order(groups)):
        for r0, c0 in group["stamps"]:
            for r, c in np.argwhere(group["patch"] != 0):
                out[int(r0 + r), int(c0 + c)] = int(group["patch"][int(r), int(c)])
    return out


def _group_pixel_map(group: dict) -> dict[tuple[int, int], int]:
    return _render_group_pixels(group["patch"], group["stamps"])


def _build_scene(shape: tuple[int, int], scenario: str) -> tuple[np.ndarray, np.ndarray] | None:
    main_colors = random.sample(PALETTE, 2)
    template_a, template_b = random.sample(SPRITE_TEMPLATES, 2)
    template_a = _random_transform(template_a)
    template_b = _random_transform(template_b)

    accents_needed = (_patch_color_count(template_a) - 1) + (_patch_color_count(template_b) - 1)
    remaining = [c for c in PALETTE if c not in main_colors]
    accent_pool = random.sample(remaining, accents_needed)
    cut = _patch_color_count(template_a) - 1
    patch_a = _instantiate_template(template_a, main_colors[0], accent_pool[:cut])
    patch_b = _instantiate_template(template_b, main_colors[1], accent_pool[cut:])

    origin_a, origin_b = _corner_origins(shape, patch_a, patch_b)
    sprite_a = {"patch": patch_a, "origin": origin_a, "main_color": main_colors[0]}
    sprite_b = {"patch": patch_b, "origin": origin_b, "main_color": main_colors[1]}
    sprites = [sprite_a, sprite_b]
    sprite_cells = _patch_occupied_cells(patch_a, origin_a) | _patch_occupied_cells(patch_b, origin_b)
    if _touching(_patch_occupied_cells(patch_a, origin_a), _patch_occupied_cells(patch_b, origin_b)):
        return None

    if scenario == "count_overlap":
        small = random.randint(1, 3)
        large = random.randint(small + 1, 4)
        counts = [small, large]
        random.shuffle(counts)
    elif scenario == "row_overlap":
        count = random.randint(2, 4)
        counts = [count, count]
    else:
        counts = random.choice(((1, 2), (1, 3), (2, 2), (2, 3), (3, 4)))
        counts = list(counts)
        random.shuffle(counts)

    eligible_a = _eligible_steps(patch_a)
    eligible_b = _eligible_steps(patch_b)
    random.shuffle(eligible_a)
    random.shuffle(eligible_b)

    for step_a in eligible_a:
        candidates_a = _candidate_groups(shape, patch_a, counts[0], step_a)
        if not candidates_a:
            continue
        for step_b in eligible_b:
            candidates_b = _candidate_groups(shape, patch_b, counts[1], step_b)
            if not candidates_b:
                continue
            for cand_a in candidates_a[:80]:
                markers_a = set(cand_a["markers"])
                if _touching(markers_a, sprite_cells):
                    continue
                for cand_b in candidates_b[:80]:
                    markers_b = set(cand_b["markers"])
                    if _touching(markers_b, sprite_cells) or _touching(markers_a, markers_b):
                        continue
                    if counts[0] == counts[1] and cand_a["markers"][0][0] == cand_b["markers"][0][0]:
                        continue

                    groups = [
                        {
                            "markers": cand_a["markers"],
                            "step": cand_a["step"],
                            "stamps": cand_a["stamps"],
                            "patch": patch_a,
                            "color": main_colors[0],
                        },
                        {
                            "markers": cand_b["markers"],
                            "step": cand_b["step"],
                            "stamps": cand_b["stamps"],
                            "patch": patch_b,
                            "color": main_colors[1],
                        },
                    ]

                    pixels_a = _group_pixel_map(groups[0])
                    pixels_b = _group_pixel_map(groups[1])
                    overlap = set(pixels_a) & set(pixels_b)
                    expected = _render_output_from_groups(groups, shape)
                    reversed_out = _reversed_order_output(groups, shape)
                    swapped_out = _try_render_output_from_groups(_swapped_groups(groups), shape)
                    first_only_out = _render_output_from_groups(_first_only_groups(groups), shape)

                    if swapped_out is not None and np.array_equal(expected, swapped_out):
                        continue
                    if np.array_equal(expected, first_only_out):
                        continue

                    if scenario == "count_overlap":
                        if len(overlap) == 0 or len(overlap) > 4:
                            continue
                        if counts[0] == counts[1]:
                            continue
                        if np.array_equal(expected, reversed_out):
                            continue
                    elif scenario == "row_overlap":
                        if len(overlap) == 0 or len(overlap) > 4:
                            continue
                        if counts[0] != counts[1]:
                            continue
                        if np.array_equal(expected, reversed_out):
                            continue
                    else:
                        if overlap:
                            continue

                    inp = _render_input(shape, sprites, groups)
                    try:
                        actual = main(inp)
                    except AssertionError:
                        continue
                    if not np.array_equal(actual, expected):
                        continue
                    return inp, expected
    return None


def _gen_attempt() -> np.ndarray | None:
    scenario = random.choices(SCENARIOS, weights=(3, 3, 2), k=1)[0]
    h = random.randint(16, 22)
    w = random.randint(16, 22)
    scene = _build_scene((h, w), scenario)
    if scene is None:
        return None
    inp, _expected = scene
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 3")
