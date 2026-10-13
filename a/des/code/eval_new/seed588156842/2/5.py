import random

import numpy as np


BG = 0
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]

GRID_WIDTH_RANGE = (13, 29)
BODY_HEIGHT_RANGE = (8, 18)
BODY_COLOR_COUNT_RANGE = (3, 4)
SOURCE_COLOR_COUNT_RANGE = (2, 3)
MOTIF_COUNT_RANGE = (6, 12)
GENERATION_ATTEMPTS = 400


def _parse_mapping_row(row: np.ndarray) -> dict[int, int]:
    mapping: dict[int, int] = {}
    c = 0
    width = int(row.shape[0])
    while c < width:
        if int(row[c]) == BG:
            c += 1
            continue
        start = c
        while c < width and int(row[c]) != BG:
            c += 1
        run = row[start:c]
        assert len(run) == 2, "each mapping entry must be a colored pair"
        src, dst = map(int, run.tolist())
        assert src != BG and dst != BG
        assert src not in mapping, "mapping sources must be unique"
        mapping[src] = dst
    assert mapping, "mapping strip must contain at least one pair"
    return mapping


def _draw_cross(grid: np.ndarray, r: int, c: int, color: int) -> None:
    h, w = grid.shape
    for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            grid[rr, cc] = int(color)


def _find_mismatch_events(body: np.ndarray, mapping: dict[int, int]) -> list[tuple[tuple[int, int], tuple[int, int], int]]:
    h, w = body.shape
    events: list[tuple[tuple[int, int], tuple[int, int], int]] = []
    for r in range(h):
        row = body[r]
        for c in range(w // 2):
            mc = w - 1 - c
            left = int(row[c])
            right = int(row[mc])
            if left == BG or right == BG or left == right:
                continue

            left_count = int(np.count_nonzero(row == left))
            right_count = int(np.count_nonzero(row == right))
            assert left_count != right_count, "row-majority test must be unambiguous"

            if left_count < right_count:
                smudge = (r, c)
                mirror = (r, mc)
                smudge_color = left
            else:
                smudge = (r, mc)
                mirror = (r, c)
                smudge_color = right

            assert smudge_color in mapping, "every smudge color must appear in the mapping strip"
            events.append((smudge, mirror, int(mapping[smudge_color])))

    assert events, "expected at least one asymmetric colored pair"
    return events


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect mirror-breaking colored smudges and stamp mapped repair crosses at both symmetric sites.

    Input structure:
    - The background is black, and every row above the bottom strip is nearly left-right mirror symmetric.
    - Colored motifs occupy both halves of the grid; the vertical symmetry axis is the grid's middle column gap or middle column.
    - A few mirrored pixel pairs are corrupted: both sides are non-black, but the two colors differ.
    - The bottom row is a color-mapping strip made of adjacent colored pairs, with black cells separating pairs.

    Rule:
    - Parse the bottom row as source->target color mappings.
    - For every non-black mirrored mismatch above the strip, compare how many times each of the two colors appears in that row; the more frequent color is the correct one, and the less frequent color is the smudge.
    - Replace the smudge color by its mapped target and draw a 3x3 cross (center plus four orthogonal neighbors) at the smudge position and at its mirrored counterpart, clipping at the grid edges.
    - Leave the rest of the picture unchanged and turn the entire bottom mapping strip to black.

    Key insight:
    - The puzzle combines three cues: vertical reflection identifies anomaly pairs, row-level color frequency picks the wrong side, and the bottom legend converts the smudge color into the repair-cross color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2

    body = g[:-1]
    mapping = _parse_mapping_row(g[-1])
    events = _find_mismatch_events(body, mapping)

    out = g.copy()
    for smudge, mirror, color in events:
        _draw_cross(out, smudge[0], smudge[1], color)
        _draw_cross(out, mirror[0], mirror[1], color)
    out[-1, :] = BG
    return out


def _sample_mask(max_h: int, max_w: int) -> np.ndarray:
    kind = random.choices(
        ["block", "bar", "l", "t", "step"],
        weights=[4, 2, 2, 1, 1],
        k=1,
    )[0]

    if kind == "block":
        h = random.randint(1, min(4, max_h))
        w = random.randint(1, min(5, max_w))
        mask = np.ones((h, w), dtype=bool)
    elif kind == "bar":
        if random.random() < 0.5:
            h = random.randint(1, min(2, max_h))
            w = random.randint(2, min(6, max_w))
        else:
            h = random.randint(2, min(5, max_h))
            w = random.randint(1, min(2, max_w))
        mask = np.ones((h, w), dtype=bool)
    elif kind == "l":
        h = random.randint(2, min(4, max_h))
        w = random.randint(2, min(4, max_w))
        mask = np.zeros((h, w), dtype=bool)
        mask[:, 0] = True
        mask[-1, :] = True
    elif kind == "t":
        h = random.randint(2, min(4, max_h))
        w = random.randint(3, min(5, max_w))
        mask = np.zeros((h, w), dtype=bool)
        mask[0, :] = True
        mask[:, w // 2] = True
    else:
        h = random.randint(2, min(4, max_h))
        w = random.randint(3, min(5, max_w))
        mask = np.zeros((h, w), dtype=bool)
        for r in range(h):
            start = min(r, w - 1)
            mask[r, start:] = True

    if random.random() < 0.5:
        mask = np.flipud(mask)
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    return mask


def _motif_cells(mask: np.ndarray, r0: int, c0: int, width: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for dr, dc in np.argwhere(mask):
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        cells.add((rr, cc))
        cells.add((rr, width - 1 - cc))
    return cells


def _add_center_accents(body: np.ndarray, colors: list[int]) -> None:
    h, w = body.shape
    if w % 2 == 0:
        return

    center = w // 2
    target = random.randint(1, max(1, h // 4))
    placed = 0
    attempts = 0
    while placed < target and attempts < 30:
        attempts += 1
        length = random.randint(1, min(3, h))
        r0 = random.randint(0, h - length)
        if np.any(body[r0 : r0 + length, center] != BG):
            continue
        body[r0 : r0 + length, center] = random.choice(colors)
        placed += 1


def _candidate_pairs(body: np.ndarray) -> list[tuple[int, int, int]]:
    h, w = body.shape
    candidates: list[tuple[int, int, int]] = []
    for r in range(h):
        row = body[r]
        for c in range(w // 2):
            color = int(row[c])
            if color == BG or int(row[w - 1 - c]) != color:
                continue
            if int(np.count_nonzero(row == color)) < 3:
                continue
            candidates.append((r, c, color))
    return candidates


def _choose_smudges(body: np.ndarray, candidates: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    h, w = body.shape
    max_smudges = min(4, max(1, (h + 2) // 4))
    target = random.randint(1, max_smudges)

    edge_candidates = [cand for cand in candidates if cand[0] in (0, h - 1) or cand[1] == 0]
    ordered = candidates[:]
    random.shuffle(ordered)

    chosen: list[tuple[int, int, int]] = []
    if edge_candidates and random.random() < 0.5:
        first = random.choice(edge_candidates)
        chosen.append(first)

    for cand in ordered:
        if cand in chosen:
            continue
        if any(abs(cand[0] - prev[0]) <= 2 for prev in chosen):
            continue
        chosen.append(cand)
        if len(chosen) >= target:
            break

    if len(chosen) < target:
        return []

    # Favor a visible border-clipping case occasionally, but do not force it every time.
    if edge_candidates and not any(cand[0] in (0, h - 1) or cand[1] == 0 for cand in chosen) and random.random() < 0.4:
        return []

    return chosen


def _mapping_row(width: int, mapping: dict[int, int]) -> np.ndarray:
    row = np.zeros(width, dtype=int)
    items = list(mapping.items())
    random.shuffle(items)
    span = 3 * len(items) - 1
    start = random.randint(0, width - span)
    pos = start
    for idx, (src, dst) in enumerate(items):
        row[pos] = int(src)
        row[pos + 1] = int(dst)
        pos += 2
        if idx != len(items) - 1:
            pos += 1
    return row


def _gen_attempt() -> np.ndarray | None:
    width = random.randint(*GRID_WIDTH_RANGE)
    body_h = random.randint(*BODY_HEIGHT_RANGE)
    left_w = width // 2

    if left_w < 3:
        return None

    body_color_count = random.randint(*BODY_COLOR_COUNT_RANGE)
    source_count = random.randint(*SOURCE_COLOR_COUNT_RANGE)
    if body_color_count + source_count > len(COLOR_POOL):
        return None

    chosen = random.sample(COLOR_POOL, body_color_count + source_count)
    body_colors = chosen[:body_color_count]
    source_colors = chosen[body_color_count:]
    target_colors = random.sample([c for c in COLOR_POOL if c not in source_colors], source_count)
    mapping = {src: dst for src, dst in zip(source_colors, target_colors, strict=True)}

    body = np.zeros((body_h, width), dtype=int)
    motif_target = random.randint(*MOTIF_COUNT_RANGE)
    placed = 0
    attempts = 0

    while placed < motif_target and attempts < motif_target * 25:
        attempts += 1
        mask = _sample_mask(body_h, left_w)
        mh, mw = mask.shape
        r0 = random.randint(0, body_h - mh)
        c0 = random.randint(0, left_w - mw)
        cells = _motif_cells(mask, r0, c0, width)
        if any(body[r, c] != BG for r, c in cells):
            continue
        color = random.choice(body_colors)
        for r, c in cells:
            body[r, c] = color
        placed += 1

    _add_center_accents(body, body_colors)

    filled = int(np.count_nonzero(body))
    if filled < max(18, body.size // 7) or filled > (body.size * 11) // 20:
        return None
    if len({int(c) for c in np.unique(body) if int(c) != BG}) < 3:
        return None

    candidates = _candidate_pairs(body)
    if not candidates:
        return None

    chosen_smudges = _choose_smudges(body, candidates)
    if not chosen_smudges:
        return None

    mutated = body.copy()
    expected_smudges: set[tuple[int, int]] = set()
    for r, c, _color in chosen_smudges:
        smudge_color = random.choice(source_colors)
        if random.random() < 0.5:
            sc = c
        else:
            sc = width - 1 - c
        mutated[r, sc] = smudge_color
        expected_smudges.add((r, sc))

    grid = np.vstack([mutated, _mapping_row(width, mapping)])

    try:
        parsed = _parse_mapping_row(grid[-1])
        events = _find_mismatch_events(grid[:-1], parsed)
        found_smudges = {event[0] for event in events}
        if found_smudges != expected_smudges:
            return None
        out = main(grid)
        if out.shape != grid.shape or np.any(out[-1] != BG):
            return None
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-smudge puzzle")
