from __future__ import annotations

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg


def _colors(grid: np.ndarray, *, exclude: set[int]) -> set[int]:
    return set(map(int, np.unique(grid))) - set(map(int, exclude))


def _placements(source: np.ndarray, target: np.ndarray) -> tuple[set[tuple[int, int]], list[dict]]:
    source_bg = get_bg(source)
    target_bg = get_bg(target)
    target_colors = _colors(target, exclude={target_bg})
    assert target_colors, "target panel must contain marker pixels"

    target_cells = {(int(r), int(c)) for r, c in np.argwhere(target != target_bg)}
    placements: list[dict] = []
    components = find_components(source, bg=source_bg, connectivity=4, monochromatic=False)

    for component in components:
        bbox = component["bbox"]
        patch = crop_bbox(source, bbox)
        object_mask = patch != source_bg
        marker_mask = object_mask & np.isin(patch, list(target_colors))
        if not marker_mask.any() or not (object_mask & ~marker_mask).any():
            continue

        rel_markers = tuple(
            (int(r), int(c), int(patch[r, c])) for r, c in np.argwhere(marker_mask)
        )
        seen: set[tuple[int, int]] = set()
        for marker_r, marker_c, marker_color in rel_markers:
            for target_r, target_c in np.argwhere(target == marker_color):
                top_r = int(target_r) - marker_r
                top_c = int(target_c) - marker_c
                if (top_r, top_c) in seen:
                    continue
                seen.add((top_r, top_c))

                patch_h, patch_w = patch.shape
                if top_r < 0 or top_c < 0:
                    continue
                if top_r + patch_h > target.shape[0] or top_c + patch_w > target.shape[1]:
                    continue

                marker_cells = frozenset(
                    (top_r + rel_r, top_c + rel_c) for rel_r, rel_c, _ in rel_markers
                )
                if not marker_cells <= target_cells:
                    continue

                ok = True
                for r in range(patch_h):
                    for c in range(patch_w):
                        target_color = int(target[top_r + r, top_c + c])
                        if target_color == target_bg:
                            continue
                        source_color = int(patch[r, c])
                        if (r, c, source_color) not in rel_markers or target_color != source_color:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    placements.append(
                        {
                            "patch": patch,
                            "source_bg": source_bg,
                            "top": (top_r, top_c),
                            "markers": marker_cells,
                        }
                    )

    return target_cells, placements


def _solve_pair(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    target_bg = get_bg(target)
    target_cells, placements = _placements(source, target)

    by_cell: dict[tuple[int, int], list[int]] = {cell: [] for cell in target_cells}
    for index, placement in enumerate(placements):
        for cell in placement["markers"]:
            if cell in by_cell:
                by_cell[cell].append(index)
    assert all(by_cell[cell] for cell in target_cells), "every target marker must be matchable"

    solutions: list[tuple[int, ...]] = []

    def search(covered: set[tuple[int, int]], chosen: list[int]) -> None:
        if len(solutions) > 1:
            return
        if covered == target_cells:
            solutions.append(tuple(chosen))
            return

        uncovered = [cell for cell in target_cells if cell not in covered]
        cell = min(
            uncovered,
            key=lambda x: sum(not (placements[i]["markers"] & covered) for i in by_cell[x]),
        )
        for index in by_cell[cell]:
            marker_cells = placements[index]["markers"]
            if marker_cells & covered:
                continue
            search(covered | set(marker_cells), chosen + [index])

    search(set(), [])
    assert len(solutions) == 1, "expected a unique exact cover of target markers"

    written: dict[tuple[int, int], int] = {}
    for index in solutions[0]:
        placement = placements[index]
        top_r, top_c = placement["top"]
        patch = placement["patch"]
        source_bg = int(placement["source_bg"])
        for r, c in np.argwhere(patch != source_bg):
            out_cell = (top_r + int(r), top_c + int(c))
            color = int(patch[r, c])
            assert out_cell not in written or written[out_cell] == color, "sprite overlap conflict"
            written[out_cell] = color

    out = target.copy()
    for (r, c), color in written.items():
        assert int(out[r, c]) in {target_bg, color}, "sprite placement conflicts with target marker"
        out[r, c] = color
    return out


def _candidate_outputs(g: np.ndarray) -> list[np.ndarray]:
    h, w = g.shape
    candidates: list[np.ndarray] = []
    panel_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    if h % 2 == 0:
        top = g[: h // 2, :]
        bottom = g[h // 2 :, :]
        panel_pairs.extend([(top, bottom), (bottom, top)])
    if w % 2 == 0:
        left = g[:, : w // 2]
        right = g[:, w // 2 :]
        panel_pairs.extend([(left, right), (right, left)])

    for source, target in panel_pairs:
        try:
            candidate = _solve_pair(source, target)
        except AssertionError:
            continue
        if not any(np.array_equal(candidate, existing) for existing in candidates):
            candidates.append(candidate)
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy complete source-panel sprites onto the opposite marker-only panel by aligning their internal marker pixels.

    Input structure
    - The input is two equal rectangular panels joined either vertically or horizontally.
    - One panel contains complete 4-connected multicolor sprites on its own background.
    - The other panel contains only colored marker pixels on a different background.
    Rule
    - Infer the split and source-target direction for which every target marker
      pixel is covered by exactly one translated source sprite's internal marker mask.
    - Stamp each matched source sprite onto the target panel with the source
      background transparent.
    Key insight
    - The small marker pixels are not output by themselves; their relative
      pattern selects and anchors the complete sprite to copy.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    candidates = _candidate_outputs(g)
    assert len(candidates) == 1, "expected a unique valid panel split and source-target direction"
    return candidates[0]

# --------------- generator (o5x) ---------------

import itertools

from arc_utils import paste, random_free_location

# Panel geometry and placement bounds.
PANEL_MIN, PANEL_MAX_EXCLUSIVE = 8, 18
SPRITE_H_MIN, SPRITE_H_MAX = 2, 5
SPRITE_W_MIN, SPRITE_W_MAX = 2, 7
SOURCE_BORDER, TARGET_BORDER = 1, 0
PLACEMENT_PADDING, PLACEMENT_PADDING_CONNECTIVITY = 1, 4

# Marker counts, adjacency, and color rates.
MARKER_TOTAL_MIN, MARKER_TOTAL_MAX_EXCLUSIVE = 3, 8
MARKERS_PER_SPRITE_MIN, MARKERS_PER_SPRITE_MAX = 1, 4
SINGLE_COLOR_MIN_TOTAL, SINGLE_COLOR_PROB = 4, 0.45
DISTRACTOR_MARKER_COUNT, PARITY_CLASSES = 2, (0, 1)

# Palette size, optional distractor rate, and retry budget.
ARC_COLOR_COUNT, BASE_COLOR_COUNT = 10, 3
DISTRACTOR_PROB, GENERATION_RETRIES = 0.35, 300


def _gen_norm_signature(cells):
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in cells))


def _gen_marker_positions(h, w, count, forbidden=()):
    blocked = set(forbidden)
    choices = []
    for parity in PARITY_CLASSES:
        cells = [(r, c) for r in range(h) for c in range(w) if (r + c) % 2 == parity]
        if len(cells) < count:
            continue
        for combo in itertools.combinations(cells, count):
            if _gen_norm_signature(combo) not in blocked:
                choices.append(combo)
    if not choices:
        return None
    return list(choices[int(np.random.randint(len(choices)))])


def _gen_adjacent_marker_positions(h, w):
    choices = []
    for r in range(h):
        for c in range(w):
            if r + 1 < h:
                choices.append(((r, c), (r + 1, c)))
            if c + 1 < w:
                choices.append(((r, c), (r, c + 1)))
    return list(choices[int(np.random.randint(len(choices)))])


def _gen_make_sprite(h, w, body_color, marker_color, marker_count, *, adjacent=False, forbidden=()):
    patch = np.full((h, w), int(body_color), dtype=int)
    if adjacent:
        markers = _gen_adjacent_marker_positions(h, w)
    else:
        markers = _gen_marker_positions(h, w, marker_count, forbidden=forbidden)
        if markers is None:
            return None, None
    for r, c in markers:
        patch[int(r), int(c)] = int(marker_color)
    return patch, tuple((int(r), int(c)) for r, c in markers)


def _gen_dim_options(marker_count, panel_h, panel_w):
    max_h = min(SPRITE_H_MAX, panel_h - 2 * SOURCE_BORDER)
    max_w = min(SPRITE_W_MAX, panel_w - 2 * SOURCE_BORDER)
    out = []
    for h in range(SPRITE_H_MIN, max_h + 1):
        for w in range(SPRITE_W_MIN, max_w + 1):
            if (h * w + 1) // 2 >= marker_count:
                out.append((h, w))
    return out


def _gen_choose_dimensions(count1, count2, panel_h, panel_w, want_distractor):
    opts1 = _gen_dim_options(count1, panel_h, panel_w)
    opts2 = _gen_dim_options(count2, panel_h, panel_w)
    optsd = _gen_dim_options(DISTRACTOR_MARKER_COUNT, panel_h, panel_w) if want_distractor else [None]
    limit = (panel_h * panel_w - 1) // 2
    choices = []
    for d1 in opts1:
        for d2 in opts2:
            area = d1[0] * d1[1] + d2[0] * d2[1]
            for dd in optsd:
                total = area + (0 if dd is None else dd[0] * dd[1])
                if total <= limit:
                    choices.append((d1, d2, dd))
    if not choices:
        return None
    return choices[int(np.random.randint(len(choices)))]


def _gen_place_full_sprite(canvas, sprite, bg, *, border):
    r, c = random_free_location(
        canvas, sprite, transparent=int(bg), border=int(border),
        padding=PLACEMENT_PADDING, padding_connectivity=PLACEMENT_PADDING_CONNECTIVITY,
    )
    paste(canvas, sprite, r, c)
    return int(r), int(c)


def _gen_marker_plan():
    marker_total = int(np.random.randint(MARKER_TOTAL_MIN, MARKER_TOTAL_MAX_EXCLUSIVE))
    marker_color_count = 1 if marker_total >= SINGLE_COLOR_MIN_TOTAL and np.random.random() < SINGLE_COLOR_PROB else 2
    splits = []
    for count1 in range(MARKERS_PER_SPRITE_MIN, MARKERS_PER_SPRITE_MAX + 1):
        count2 = marker_total - count1
        if not (MARKERS_PER_SPRITE_MIN <= count2 <= MARKERS_PER_SPRITE_MAX):
            continue
        if marker_color_count == 1 and min(count1, count2) < DISTRACTOR_MARKER_COUNT:
            continue
        splits.append((count1, count2))
    if not splits:
        marker_color_count = 2
        splits = [(count1, marker_total - count1) for count1 in range(MARKERS_PER_SPRITE_MIN, MARKERS_PER_SPRITE_MAX + 1) if MARKERS_PER_SPRITE_MIN <= marker_total - count1 <= MARKERS_PER_SPRITE_MAX]
    count1, count2 = splits[int(np.random.randint(len(splits)))]
    return marker_color_count, count1, count2


def _gen_palette(marker_color_count):
    colors = np.random.choice(np.arange(ARC_COLOR_COUNT), size=BASE_COLOR_COUNT + marker_color_count, replace=False).astype(int).tolist()
    src_bg, tgt_bg, body_color = colors[:BASE_COLOR_COUNT]
    marker_colors = colors[BASE_COLOR_COUNT:]
    if marker_color_count == 1:
        sprite_marker_colors = [marker_colors[0], marker_colors[0]]
    else:
        if bool(np.random.randint(2)):
            sprite_marker_colors = [marker_colors[1], marker_colors[0]]
        else:
            sprite_marker_colors = [marker_colors[0], marker_colors[1]]
    return src_bg, tgt_bg, body_color, marker_colors, sprite_marker_colors


def _gen_sprite_dimensions(count1, count2, panel_h, panel_w):
    want_distractor = bool(np.random.random() < DISTRACTOR_PROB)
    dims = _gen_choose_dimensions(count1, count2, panel_h, panel_w, want_distractor)
    if dims is None and want_distractor:
        want_distractor = False
        dims = _gen_choose_dimensions(count1, count2, panel_h, panel_w, False)
    if dims is None:
        return None
    return want_distractor, dims


def _gen_required_sprites(dim1, dim2, body_color, sprite_marker_colors, count1, count2):
    sprite1, markers1 = _gen_make_sprite(dim1[0], dim1[1], body_color, sprite_marker_colors[0], count1)
    if sprite1 is None:
        return None
    forbidden = ()
    if sprite_marker_colors[0] == sprite_marker_colors[1] and count1 == count2:
        forbidden = (_gen_norm_signature(markers1),)
    sprite2, markers2 = _gen_make_sprite(dim2[0], dim2[1], body_color, sprite_marker_colors[1], count2, forbidden=forbidden)
    if sprite2 is None:
        return None
    return sprite1, markers1, sprite2, markers2


def _gen_place_target_pair(target_out, target_in, sprite1, markers1, sprite2, markers2, sprite_marker_colors):
    """Place the required sprites in output and copy only their markers to input."""
    top1 = _gen_place_full_sprite(target_out, sprite1, target_in[0, 0], border=TARGET_BORDER)
    top2 = _gen_place_full_sprite(target_out, sprite2, target_in[0, 0], border=TARGET_BORDER)
    for r, c in markers1:
        target_in[top1[0] + r, top1[1] + c] = sprite_marker_colors[0]
    for r, c in markers2:
        target_in[top2[0] + r, top2[1] + c] = sprite_marker_colors[1]


def _gen_place_source_sprites(source, sprite1, sprite2, want_distractor, dimd, body_color, marker_colors):
    """Place required source sprites and the optional invalid distractor."""
    source_bg = int(source[0, 0])
    _gen_place_full_sprite(source, sprite1, source_bg, border=SOURCE_BORDER)
    _gen_place_full_sprite(source, sprite2, source_bg, border=SOURCE_BORDER)
    if want_distractor and dimd is not None:
        distractor_color = marker_colors[int(np.random.randint(len(marker_colors)))]
        distractor, _ = _gen_make_sprite(dimd[0], dimd[1], body_color, distractor_color, DISTRACTOR_MARKER_COUNT, adjacent=True)
        _gen_place_full_sprite(source, distractor, source_bg, border=SOURCE_BORDER)


def _gen_attempt() -> np.ndarray | None:
    panel_h = int(np.random.randint(PANEL_MIN, PANEL_MAX_EXCLUSIVE))
    panel_w = int(np.random.randint(PANEL_MIN, PANEL_MAX_EXCLUSIVE))
    split_horizontal = bool(np.random.randint(2))
    marker_color_count, count1, count2 = _gen_marker_plan()
    src_bg, tgt_bg, body_color, marker_colors, sprite_marker_colors = _gen_palette(marker_color_count)

    sprite_plan = _gen_sprite_dimensions(count1, count2, panel_h, panel_w)
    if sprite_plan is None:
        return None
    want_distractor, dims = sprite_plan
    dim1, dim2, dimd = dims

    sprites = _gen_required_sprites(dim1, dim2, body_color, sprite_marker_colors, count1, count2)
    if sprites is None:
        return None
    sprite1, markers1, sprite2, markers2 = sprites

    source = np.full((panel_h, panel_w), int(src_bg), dtype=int)
    target_in = np.full((panel_h, panel_w), int(tgt_bg), dtype=int)
    target_out = np.full((panel_h, panel_w), int(tgt_bg), dtype=int)

    try:
        _gen_place_target_pair(target_out, target_in, sprite1, markers1, sprite2, markers2, sprite_marker_colors)
    except ValueError:
        return None

    try:
        _gen_place_source_sprites(source, sprite1, sprite2, want_distractor, dimd, body_color, marker_colors)
    except ValueError:
        return None

    grid = np.concatenate([source, target_in], axis=1 if split_horizontal else 0)
    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, target_out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e6721834 input")
