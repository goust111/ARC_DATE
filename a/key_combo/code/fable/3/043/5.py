from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
WHITE = 10
SPRITE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 1200


def _bbox(points: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in points]
    cols = [c for _r, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _stripe_axes(g: np.ndarray) -> tuple[set[int], set[int]]:
    grey = g == GREY
    stripe_rows = {int(r) for r in np.flatnonzero(np.all(grey, axis=1))}
    stripe_cols = {int(c) for c in np.flatnonzero(np.all(grey, axis=0))}
    assert stripe_rows or stripe_cols, "expected at least one full grey stripe"

    for r, c in np.argwhere(grey):
        assert int(r) in stripe_rows or int(c) in stripe_cols, "grey pixels must belong to full stripes"
    return stripe_rows, stripe_cols


def _assign_visible_pixels(
    g: np.ndarray, anchors: np.ndarray, colors: list[int]
) -> tuple[dict[int, int], dict[int, set[tuple[int, int]]]]:
    color_by_anchor: dict[int, int] = {}
    rels_by_anchor: dict[int, set[tuple[int, int]]] = {}

    for color in colors:
        coords = np.argwhere(g == color)
        assert len(coords) > 0, "each copy color must have visible shape pixels"

        owners: list[int] = []
        for r, c in coords:
            distances = np.abs(anchors - np.array([int(r), int(c)])).sum(axis=1)
            best = int(np.min(distances))
            winners = np.flatnonzero(distances == best)
            assert len(winners) == 1, "visible pixels must have a unique nearest anchor"
            owners.append(int(winners[0]))

        owner_set = set(owners)
        assert len(owner_set) == 1, "one sprite color must belong to one anchored copy"
        owner = owners[0]
        assert owner not in color_by_anchor, "each anchor must have exactly one sprite color"

        rels = {
            (int(r) - int(anchors[owner, 0]), int(c) - int(anchors[owner, 1]))
            for r, c in coords
        }
        assert (0, 0) not in rels, "the anchor cell is marked white, not with the sprite color"
        color_by_anchor[owner] = int(color)
        rels_by_anchor[owner] = rels

    assert set(color_by_anchor) == set(range(len(anchors))), "every anchor must identify one colored copy"
    return color_by_anchor, rels_by_anchor


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct an occluded shared sprite by anchor-aligning copies and taking their union.

    Input structure:
    - The black canvas contains several differently colored copies of one sprite.
    - Each copy has one visible white anchor pixel at the same relative position
      inside the sprite.
    - Full-width and/or full-height grey stripes cover the canvas and hide
      different parts of the copies.

    Rule:
    - Use each white anchor as the origin for its copy, and convert every visible
      colored sprite pixel to anchor-relative coordinates.
    - Take the union of those relative coordinates to recover the complete sprite
      mask, and choose the color of the copy with the uniquely largest number of
      visible colored pixels.
    - Return the complete mask cropped to its tight bounding box, using that
      chosen color for the sprite and white for the anchor cell.

    Key insight:
    - No individual copy is complete; the hidden shape exists only after aligning
      the differently occluded copies by their anchor markers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30, "expected a 2D ARC grid"
    assert np.any(g == BLACK), "expected black background"
    _stripe_axes(g)

    anchors = np.argwhere(g == WHITE)
    assert len(anchors) >= 2, "expected multiple anchored copies"

    colors = sorted(int(c) for c in np.unique(g) if int(c) not in (BLACK, GREY, WHITE))
    assert len(colors) == len(anchors), "each copy must have one unique non-grey color"

    color_by_anchor, rels_by_anchor = _assign_visible_pixels(g, anchors.astype(int), colors)

    full_mask: set[tuple[int, int]] = {(0, 0)}
    for rels in rels_by_anchor.values():
        full_mask.update(rels)

    full_colored = full_mask - {(0, 0)}
    assert full_colored, "reconstructed sprite must contain colored cells"
    for rels in rels_by_anchor.values():
        assert rels <= full_colored
        assert len(rels) < len(full_colored), "no single copy may already be complete"

    counts = {idx: len(rels) for idx, rels in rels_by_anchor.items()}
    best_count = max(counts.values())
    winners = [idx for idx, count in counts.items() if count == best_count]
    assert len(winners) == 1, "visible-pixel maximum must be unique"
    out_color = color_by_anchor[winners[0]]

    r0, c0, r1, c1 = _bbox(full_mask)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), BLACK, dtype=int)
    for r, c in full_mask:
        out[r - r0, c - c0] = out_color
    out[-r0, -c0] = WHITE
    return out


def _add_path(cells: set[tuple[int, int]], start: tuple[int, int], end: tuple[int, int]) -> None:
    r, c = start
    er, ec = end
    cells.add((r, c))
    while (r, c) != (er, ec):
        steps: list[tuple[int, int]] = []
        if r < er:
            steps.append((r + 1, c))
        elif r > er:
            steps.append((r - 1, c))
        if c < ec:
            steps.append((r, c + 1))
        elif c > ec:
            steps.append((r, c - 1))
        r, c = random.choice(steps)
        cells.add((r, c))


def _frontier(cells: set[tuple[int, int]], h: int, w: int) -> list[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                out.add((nr, nc))
    return list(out)


def _gen_sprite() -> tuple[np.ndarray, tuple[int, int], set[tuple[int, int]]]:
    h = random.randint(5, 8)
    w = random.randint(5, 8)
    anchor = (random.randint(1, h - 2), random.randint(1, w - 2))
    cells: set[tuple[int, int]] = {anchor}

    border_targets = [
        (0, random.randrange(w)),
        (h - 1, random.randrange(w)),
        (random.randrange(h), 0),
        (random.randrange(h), w - 1),
    ]
    random.shuffle(border_targets)
    for target in border_targets:
        _add_path(cells, anchor, target)

    target_area = random.randint(max(len(cells), int(0.45 * h * w)), int(0.68 * h * w))
    while len(cells) < target_area:
        choices = _frontier(cells, h, w)
        if not choices:
            break
        cells.add(random.choice(choices))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    assert mask[0, :].any() and mask[-1, :].any()
    assert mask[:, 0].any() and mask[:, -1].any()

    rels = {(r - anchor[0], c - anchor[1]) for r, c in cells}
    return mask, anchor, rels


def _copy_bbox(anchor: tuple[int, int], rel_bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    ar, ac = anchor
    r0, c0, r1, c1 = rel_bbox
    return ar + r0, ac + c0, ar + r1, ac + c1


def _padded_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _place_anchors(
    h: int, w: int, count: int, rel_bbox: tuple[int, int, int, int]
) -> list[tuple[int, int]] | None:
    min_r, min_c, max_r, max_c = rel_bbox
    r_lo = 1 - min_r
    c_lo = 1 - min_c
    r_hi = h - 2 - max_r
    c_hi = w - 2 - max_c
    if r_lo > r_hi or c_lo > c_hi:
        return None

    anchors: list[tuple[int, int]] = []
    boxes: list[tuple[int, int, int, int]] = []
    for _ in range(count):
        placed = False
        for _try in range(400):
            anchor = (random.randint(r_lo, r_hi), random.randint(c_lo, c_hi))
            box = _copy_bbox(anchor, rel_bbox)
            if all(not _padded_overlap(box, old, pad=2) for old in boxes):
                anchors.append(anchor)
                boxes.append(box)
                placed = True
                break
        if not placed:
            return None
    return anchors


def _visible_sets(
    rels: set[tuple[int, int]],
    anchors: list[tuple[int, int]],
    stripe_rows: set[int],
    stripe_cols: set[int],
) -> list[set[tuple[int, int]]]:
    colored_rels = rels - {(0, 0)}
    visible: list[set[tuple[int, int]]] = []
    for ar, ac in anchors:
        cur: set[tuple[int, int]] = set()
        for dr, dc in colored_rels:
            if ar + dr in stripe_rows or ac + dc in stripe_cols:
                continue
            cur.add((dr, dc))
        visible.append(cur)
    return visible


def _line_candidates_by_copy(
    rels: set[tuple[int, int]], anchors: list[tuple[int, int]]
) -> tuple[list[list[int]], list[list[int]]]:
    colored_rels = rels - {(0, 0)}
    anchor_rows = {r for r, _c in anchors}
    anchor_cols = {c for _r, c in anchors}

    row_options: list[list[int]] = []
    col_options: list[list[int]] = []
    for ar, ac in anchors:
        rows = sorted({ar + dr for dr, _dc in colored_rels if ar + dr not in anchor_rows})
        cols = sorted({ac + dc for _dr, dc in colored_rels if ac + dc not in anchor_cols})
        row_options.append(rows)
        col_options.append(cols)
    return row_options, col_options


def _semantically_valid_visible_sets(
    visible: list[set[tuple[int, int]]], rels: set[tuple[int, int]]
) -> bool:
    full = rels - {(0, 0)}
    if not full:
        return False

    union: set[tuple[int, int]] = set()
    for cur in visible:
        union.update(cur)
    if union != full:
        return False

    counts = [len(cur) for cur in visible]
    if any(count == len(full) for count in counts):
        return False
    if min(counts) < max(4, int(0.32 * len(full))):
        return False
    if counts.count(max(counts)) != 1:
        return False
    if len({frozenset(cur) for cur in visible}) < min(3, len(visible)):
        return False

    winner = counts.index(max(counts))
    if not (full - visible[winner]):
        return False
    # At least one recovered cell must come from a non-winning copy, otherwise
    # "choose the biggest visible copy" would be enough.
    supplied_by_others = set().union(*(cur for i, cur in enumerate(visible) if i != winner))
    return bool((full - visible[winner]) <= supplied_by_others)


def _sample_stripes(
    rels: set[tuple[int, int]], anchors: list[tuple[int, int]]
) -> tuple[set[int], set[int], list[set[tuple[int, int]]]] | None:
    row_options, col_options = _line_candidates_by_copy(rels, anchors)
    if not all(rows or cols for rows, cols in zip(row_options, col_options)):
        return None

    for _ in range(700):
        stripe_rows: set[int] = set()
        stripe_cols: set[int] = set()
        target_total = random.randint(2, min(5, len(anchors) + 1))

        for _step in range(target_total):
            visible = _visible_sets(rels, anchors, stripe_rows, stripe_cols)
            full = rels - {(0, 0)}
            complete = [i for i, cur in enumerate(visible) if len(cur) == len(full)]
            idx = random.choice(complete) if complete else random.randrange(len(anchors))

            choices: list[tuple[str, int]] = []
            choices.extend(("r", r) for r in row_options[idx] if r not in stripe_rows)
            choices.extend(("c", c) for c in col_options[idx] if c not in stripe_cols)
            if not choices:
                continue
            orient, value = random.choice(choices)
            if orient == "r":
                stripe_rows.add(value)
            else:
                stripe_cols.add(value)

        if len(stripe_rows) + len(stripe_cols) < 2:
            continue
        visible = _visible_sets(rels, anchors, stripe_rows, stripe_cols)
        if _semantically_valid_visible_sets(visible, rels):
            return stripe_rows, stripe_cols, visible
    return None


def _expected_output(
    rels: set[tuple[int, int]], color: int
) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(rels)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), BLACK, dtype=int)
    for r, c in rels:
        out[r - r0, c - c0] = color
    out[-r0, -c0] = WHITE
    return out


def _render_input(
    shape: np.ndarray,
    anchor_in_shape: tuple[int, int],
    canvas_shape: tuple[int, int],
    anchors: list[tuple[int, int]],
    colors: list[int],
    stripe_rows: set[int],
    stripe_cols: set[int],
) -> np.ndarray:
    out = np.full(canvas_shape, BLACK, dtype=int)
    shape_coords = np.argwhere(shape)
    for (ar, ac), color in zip(anchors, colors):
        for sr, sc in shape_coords:
            dr = int(sr) - anchor_in_shape[0]
            dc = int(sc) - anchor_in_shape[1]
            gr = ar + dr
            gc = ac + dc
            if (dr, dc) == (0, 0):
                continue
            if gr in stripe_rows or gc in stripe_cols:
                continue
            out[gr, gc] = int(color)
        out[ar, ac] = WHITE

    for r in stripe_rows:
        out[int(r), :] = GREY
    for c in stripe_cols:
        out[:, int(c)] = GREY
    return out


def _nearest_anchor_is_unambiguous(
    rels: set[tuple[int, int]], anchors: list[tuple[int, int]], visible: list[set[tuple[int, int]]]
) -> bool:
    anchor_arr = np.asarray(anchors, dtype=int)
    for idx, ((ar, ac), cur) in enumerate(zip(anchors, visible)):
        for dr, dc in cur:
            p = np.array([ar + dr, ac + dc], dtype=int)
            distances = np.abs(anchor_arr - p).sum(axis=1)
            if int(np.argmin(distances)) != idx:
                return False
            if int(np.sum(distances == np.min(distances))) != 1:
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    shape, anchor_in_shape, rels = _gen_sprite()
    rel_bbox = _bbox(rels)
    copy_count = random.choice((3, 3, 4, 5))

    canvas_h = random.randint(19, 30)
    canvas_w = random.randint(19, 30)
    anchors = _place_anchors(canvas_h, canvas_w, copy_count, rel_bbox)
    if anchors is None:
        return None

    stripe_data = _sample_stripes(rels, anchors)
    if stripe_data is None:
        return None
    stripe_rows, stripe_cols, visible = stripe_data
    if not _nearest_anchor_is_unambiguous(rels, anchors, visible):
        return None

    colors = random.sample(SPRITE_COLORS, copy_count)
    counts = [len(cur) for cur in visible]
    winner = counts.index(max(counts))
    expected = _expected_output(rels, colors[winner])
    grid = _render_input(shape, anchor_in_shape, (canvas_h, canvas_w), anchors, colors, stripe_rows, stripe_cols)

    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid anchored stripe-union input")
