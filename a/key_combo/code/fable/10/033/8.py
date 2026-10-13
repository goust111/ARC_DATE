from __future__ import annotations

import random
from math import ceil

import numpy as np


BACKGROUND = 0
PALETTE = tuple(range(1, 10))

GEN_ATTEMPTS = 600
PANEL_SIZE_RANGE = (7, 10)
HOLE_SIDE_RANGE = (2, 4)
CANDIDATE_COUNT_RANGE = (3, 5)


def _get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros(mask.shape, dtype=bool)
    out: list[np.ndarray] = []
    h, w = mask.shape
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
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _square_panel_components(grid: np.ndarray, bg: int) -> list[dict]:
    panels = []
    for idx, coords in enumerate(_components(grid != bg)):
        bbox = _bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h == w and h >= 5:
            panels.append(
                {
                    "idx": idx,
                    "coords": coords,
                    "bbox": bbox,
                    "side": h,
                    "bbox_area": h * w,
                    "area": int(coords.shape[0]),
                }
            )
    panels.sort(key=lambda item: (item["bbox_area"], item["area"]), reverse=True)
    return panels


def _rectangular_hole(panel: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    holes = _components(panel == bg)
    holes = [coords for coords in holes if coords.size > 0]
    assert len(holes) == 1, "expected a single black window"
    bbox = _bbox_coords(holes[0])
    r0, c0, r1, c1 = bbox
    area = int(holes[0].shape[0])
    assert area == (r1 - r0 + 1) * (c1 - c0 + 1), "window must be rectangular"
    assert r0 > 0 and c0 > 0 and r1 < panel.shape[0] - 1 and c1 < panel.shape[1] - 1
    return bbox


def _candidate_components(
    grid: np.ndarray,
    bg: int,
    panel_ids: set[int],
    expected_shape: tuple[int, int],
) -> list[dict]:
    comps = _components(grid != bg)
    out = []
    need_h, need_w = expected_shape
    for idx, coords in enumerate(comps):
        if idx in panel_ids:
            continue
        bbox = _bbox_coords(coords)
        patch = _crop_bbox(grid, bbox)
        h, w = patch.shape
        if (h, w) != (need_h, need_w):
            continue
        if int(coords.shape[0]) != h * w:
            continue
        out.append({"coords": coords, "bbox": bbox, "patch": patch})
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the black window in one square panel with the only candidate patch that restores transpose-twin consistency.

    Input structure:
    - A black background contains two separated n x n colored square panels and several detached rectangular candidate patches.
    - One panel is complete; the other has one internal black rectangular window.
    - Outside the window, the two panels satisfy a main-diagonal transpose relation in panel-local coordinates.

    Rule:
    - Identify the two large square panels and locate the window inside the incomplete one.
    - Map the window's local rows and columns onto the complete panel by swapping row and column ranges, then transpose that corresponding rectangle back into the window shape.
    - Select the unique candidate patch equal to that recovered content, paste it into the window unchanged, and erase every detached candidate patch from the canvas.

    Key insight:
    - The counterpart of an h x w window is a w x h rectangle in the other panel; only after swapping the coordinate ranges and transposing that slice do you recover the correct h x w fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    panels = _square_panel_components(g, bg)
    assert len(panels) >= 2, "expected two large square panels"
    panel_a, panel_b = panels[:2]
    assert panel_a["side"] == panel_b["side"], "panels must have matching size"

    panel_grids = [_crop_bbox(g, panel_a["bbox"]), _crop_bbox(g, panel_b["bbox"])]
    bg_counts = [int(np.count_nonzero(panel == bg)) for panel in panel_grids]
    assert sorted(count > 0 for count in bg_counts) == [False, True], "exactly one panel must contain the window"
    hole_idx = 0 if bg_counts[0] > 0 else 1
    full_idx = 1 - hole_idx

    hole_panel = panel_grids[hole_idx]
    full_panel = panel_grids[full_idx]
    hr0, hc0, hr1, hc1 = _rectangular_hole(hole_panel, bg)
    source = full_panel[hc0 : hc1 + 1, hr0 : hr1 + 1]
    needed_patch = source.T.copy()

    candidate_comps = _candidate_components(
        g,
        bg,
        {panel_a["idx"], panel_b["idx"]},
        needed_patch.shape,
    )
    matches = [cand for cand in candidate_comps if np.array_equal(cand["patch"], needed_patch)]
    assert len(matches) == 1, "candidate patch must be unique"

    out = g.copy()
    hole_bbox = panels[hole_idx]["bbox"]
    abs_r = hole_bbox[0] + hr0
    abs_c = hole_bbox[1] + hc0
    _paste(out, needed_patch, abs_r, abs_c)

    for cand in candidate_comps:
        coords = cand["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    return out


def _random_color(exclude: set[int]) -> int:
    choices = [color for color in PALETTE if color not in exclude]
    return random.choice(choices)


def _make_structured_panel(n: int) -> np.ndarray:
    for _ in range(120):
        colors = random.sample(PALETTE, random.randint(4, 6))
        panel = np.full((n, n), colors[0], dtype=int)

        for _ in range(random.randint(2, 3)):
            h = random.randint(1, 2)
            w = random.randint(max(3, n // 2), n - 1)
            r = random.randint(0, n - h)
            c = random.randint(0, n - w)
            panel[r : r + h, c : c + w] = random.choice(colors[1:])

        for _ in range(random.randint(2, 3)):
            h = random.randint(max(3, n // 2), n - 1)
            w = random.randint(1, 2)
            r = random.randint(0, n - h)
            c = random.randint(0, n - w)
            panel[r : r + h, c : c + w] = random.choice(colors[1:])

        for _ in range(random.randint(3, 5)):
            h = random.randint(1, min(4, n - 2))
            w = random.randint(1, min(4, n - 2))
            r = random.randint(0, n - h)
            c = random.randint(0, n - w)
            panel[r : r + h, c : c + w] = random.choice(colors[1:])

        if np.unique(panel).size < 3:
            continue
        if int(np.count_nonzero(panel != panel.T)) < max(10, n + 2):
            continue
        if len({tuple(row) for row in panel}) < 4 or len({tuple(col) for col in panel.T}) < 4:
            continue
        return panel
    raise RuntimeError("failed to build a sufficiently structured panel")


def _hole_options(n: int) -> list[tuple[int, int]]:
    dims = []
    for h in range(HOLE_SIDE_RANGE[0], min(HOLE_SIDE_RANGE[1], n - 2) + 1):
        for w in range(HOLE_SIDE_RANGE[0], min(HOLE_SIDE_RANGE[1], n - 2) + 1):
            dims.append((h, w))
    nonsquare = [dim for dim in dims if dim[0] != dim[1]]
    if nonsquare and random.random() < 0.75:
        return nonsquare
    return dims


def _pick_window(panel: np.ndarray) -> tuple[int, int, np.ndarray]:
    n = panel.shape[0]
    options = _hole_options(n)
    random.shuffle(options)
    for h, w in options:
        positions = [
            (r, c)
            for r in range(1, n - h)
            for c in range(1, n - w)
        ]
        random.shuffle(positions)
        for r0, c0 in positions:
            patch = panel[r0 : r0 + h, c0 : c0 + w]
            variants = [
                np.fliplr(patch),
                np.flipud(patch),
                np.rot90(patch, 2),
            ]
            if h == w:
                variants.append(np.rot90(patch, 1))
                variants.append(np.rot90(patch, 3))
            if np.unique(patch).size < 2:
                continue
            if any(np.array_equal(patch, alt) for alt in variants):
                continue
            return r0, c0, patch.copy()
    raise RuntimeError("failed to choose a non-degenerate window")


def _mutate_patch(patch: np.ndarray, palette: tuple[int, ...]) -> np.ndarray:
    out = patch.copy()
    cells = [(r, c) for r in range(out.shape[0]) for c in range(out.shape[1])]
    changes = random.randint(1, max(1, len(cells) // 3))
    for r, c in random.sample(cells, changes):
        out[r, c] = random.choice([color for color in palette if color != int(out[r, c])])
    return out


def _candidate_patches(correct: np.ndarray) -> list[np.ndarray]:
    palette = tuple(sorted(set(PALETTE) | set(map(int, correct.ravel()))))
    candidates = [correct.copy()]
    target_count = random.randint(*CANDIDATE_COUNT_RANGE)

    mirror_variants = []
    for variant in (np.fliplr(correct), np.flipud(correct)):
        if not np.array_equal(variant, correct):
            mirror_variants.append(variant.copy())
    assert mirror_variants, "at least one mirror decoy is required"

    decoys: list[np.ndarray] = [random.choice(mirror_variants)]
    for variant in (np.rot90(correct, 2),):
        if not any(np.array_equal(variant, prev) for prev in candidates + decoys):
            decoys.append(variant.copy())
    if correct.shape[0] == correct.shape[1]:
        for k in (1, 3):
            variant = np.rot90(correct, k)
            if not any(np.array_equal(variant, prev) for prev in candidates + decoys):
                decoys.append(variant.copy())

    while len(candidates) + len(decoys) < target_count:
        variant = _mutate_patch(correct, palette)
        if not any(np.array_equal(variant, prev) for prev in candidates + decoys):
            decoys.append(variant)

    all_patches = candidates + decoys
    random.shuffle(all_patches)
    return all_patches


def _horizontal_layout(n: int, patch_h: int, patch_w: int, count: int) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    margin = random.randint(1, 2)
    panel_gap = random.randint(2, 4)
    zone_gap = random.randint(2, 4)
    cand_gap = random.randint(1, 2)
    cand_cols = random.randint(max(2, ceil(count / 2)), count)
    cand_rows = ceil(count / cand_cols)

    width = max(
        2 * margin + 2 * n + panel_gap,
        2 * margin + cand_cols * patch_w + (cand_cols - 1) * cand_gap,
    )
    height = margin + n + zone_gap + cand_rows * patch_h + (cand_rows - 1) * cand_gap + margin
    if width > 30 or height > 30:
        raise ValueError("horizontal layout too large")

    canvas = np.zeros((height, width), dtype=int)
    slack_x = width - 2 * margin - (2 * n + panel_gap)
    panel_left = margin + (random.randint(0, slack_x) if slack_x > 0 else 0)
    needed_h = n + zone_gap + cand_rows * patch_h + (cand_rows - 1) * cand_gap
    slack_y = height - 2 * margin - needed_h
    panel_top = margin + (random.randint(0, slack_y) if slack_y > 0 else 0)
    panel_positions = [(panel_top, panel_left), (panel_top, panel_left + n + panel_gap)]

    zone_top = panel_top + n + zone_gap
    zone_width = cand_cols * patch_w + (cand_cols - 1) * cand_gap
    zone_left = random.randint(margin, width - margin - zone_width)
    candidate_positions = []
    for idx in range(count):
        rr = idx // cand_cols
        cc = idx % cand_cols
        candidate_positions.append((zone_top + rr * (patch_h + cand_gap), zone_left + cc * (patch_w + cand_gap)))
    return canvas, panel_positions, candidate_positions


def _vertical_layout(n: int, patch_h: int, patch_w: int, count: int) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    margin = random.randint(1, 2)
    panel_gap = random.randint(2, 4)
    zone_gap = random.randint(2, 4)
    cand_gap = random.randint(1, 2)
    cand_rows = random.randint(max(2, ceil(count / 2)), count)
    cand_cols = ceil(count / cand_rows)

    width = margin + n + zone_gap + cand_cols * patch_w + (cand_cols - 1) * cand_gap + margin
    height = max(
        2 * margin + 2 * n + panel_gap,
        2 * margin + cand_rows * patch_h + (cand_rows - 1) * cand_gap,
    )
    if width > 30 or height > 30:
        raise ValueError("vertical layout too large")

    canvas = np.zeros((height, width), dtype=int)
    slack_y = height - 2 * margin - (2 * n + panel_gap)
    panel_top = margin + (random.randint(0, slack_y) if slack_y > 0 else 0)
    needed_w = n + zone_gap + cand_cols * patch_w + (cand_cols - 1) * cand_gap
    slack_x = width - 2 * margin - needed_w
    panel_left = margin + (random.randint(0, slack_x) if slack_x > 0 else 0)
    panel_positions = [(panel_top, panel_left), (panel_top + n + panel_gap, panel_left)]

    zone_left = panel_left + n + zone_gap
    zone_height = cand_rows * patch_h + (cand_rows - 1) * cand_gap
    zone_top = random.randint(margin, height - margin - zone_height)
    candidate_positions = []
    for idx in range(count):
        rr = idx % cand_rows
        cc = idx // cand_rows
        candidate_positions.append((zone_top + rr * (patch_h + cand_gap), zone_left + cc * (patch_w + cand_gap)))
    return canvas, panel_positions, candidate_positions


def _build_instance() -> tuple[np.ndarray, np.ndarray]:
    n = random.randint(*PANEL_SIZE_RANGE)
    left_panel = _make_structured_panel(n)
    right_panel = left_panel.T.copy()

    hole_panel_idx = random.randint(0, 1)
    full_panels = [left_panel.copy(), right_panel.copy()]
    holed_source = full_panels[hole_panel_idx]
    hole_r, hole_c, correct_patch = _pick_window(holed_source)
    patches = _candidate_patches(correct_patch)

    if random.random() < 0.5:
        canvas, panel_positions, candidate_positions = _horizontal_layout(
            n, correct_patch.shape[0], correct_patch.shape[1], len(patches)
        )
    else:
        canvas, panel_positions, candidate_positions = _vertical_layout(
            n, correct_patch.shape[0], correct_patch.shape[1], len(patches)
        )

    out = canvas.copy()
    for panel, (r0, c0) in zip(full_panels, panel_positions):
        _paste(out, panel, r0, c0)

    inp = out.copy()
    hole_panel_top, hole_panel_left = panel_positions[hole_panel_idx]
    inp[
        hole_panel_top + hole_r : hole_panel_top + hole_r + correct_patch.shape[0],
        hole_panel_left + hole_c : hole_panel_left + hole_c + correct_patch.shape[1],
    ] = BACKGROUND

    for patch, (r0, c0) in zip(patches, candidate_positions):
        _paste(inp, patch, r0, c0)

    _validate_instance(
        inp,
        out,
        full_panels,
        panel_positions,
        hole_panel_idx,
        (hole_r, hole_c),
        patches,
        candidate_positions,
    )
    return inp, out


def _validate_instance(
    inp: np.ndarray,
    out: np.ndarray,
    full_panels: list[np.ndarray],
    panel_positions: list[tuple[int, int]],
    hole_panel_idx: int,
    hole_origin: tuple[int, int],
    patches: list[np.ndarray],
    candidate_positions: list[tuple[int, int]],
) -> None:
    assert inp.shape == out.shape
    assert inp.ndim == 2 and max(inp.shape) <= 30
    assert np.array_equal(full_panels[1], full_panels[0].T)
    assert not np.array_equal(inp, out)

    hole_r, hole_c = hole_origin
    correct_patch = None
    for patch in patches:
        patch_h, patch_w = patch.shape
        candidate = full_panels[hole_panel_idx][hole_r : hole_r + patch_h, hole_c : hole_c + patch_w]
        if np.array_equal(candidate, patch):
            correct_patch = patch.copy()
            break
    assert correct_patch is not None

    expected_patch = full_panels[1 - hole_panel_idx][
        hole_c : hole_c + correct_patch.shape[1],
        hole_r : hole_r + correct_patch.shape[0],
    ].T
    assert np.array_equal(correct_patch, expected_patch)

    visible_a = full_panels[0].copy()
    visible_b = full_panels[1].copy()
    if hole_panel_idx == 0:
        visible_a[hole_r : hole_r + correct_patch.shape[0], hole_c : hole_c + correct_patch.shape[1]] = BACKGROUND
    else:
        visible_b[hole_r : hole_r + correct_patch.shape[0], hole_c : hole_c + correct_patch.shape[1]] = BACKGROUND

    for r in range(full_panels[0].shape[0]):
        for c in range(full_panels[0].shape[1]):
            if hole_panel_idx == 0 and hole_r <= r < hole_r + correct_patch.shape[0] and hole_c <= c < hole_c + correct_patch.shape[1]:
                continue
            if hole_panel_idx == 1 and hole_r <= c < hole_r + correct_patch.shape[0] and hole_c <= r < hole_c + correct_patch.shape[1]:
                continue
            assert int(visible_a[r, c]) == int(visible_b[c, r])

    patch_matches = sum(np.array_equal(patch, correct_patch) for patch in patches)
    assert patch_matches == 1
    assert any(
        np.array_equal(patch, np.fliplr(correct_patch)) or np.array_equal(patch, np.flipud(correct_patch))
        for patch in patches
        if not np.array_equal(patch, correct_patch)
    )

    comp_count_in = len(_components(inp != BACKGROUND))
    comp_count_out = len(_components(out != BACKGROUND))
    assert comp_count_in == 2 + len(patches)
    assert comp_count_out == 2

    for panel, (r0, c0) in zip(full_panels, panel_positions):
        assert np.array_equal(out[r0 : r0 + panel.shape[0], c0 : c0 + panel.shape[1]], panel)
    for patch, (r0, c0) in zip(patches, candidate_positions):
        assert np.array_equal(inp[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]], patch)

    solved = main(inp)
    assert np.array_equal(solved, out)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            inp, _ = _build_instance()
            return inp
        except (AssertionError, RuntimeError, ValueError):
            continue
    raise RuntimeError("failed to generate a valid transpose-twin window puzzle")
