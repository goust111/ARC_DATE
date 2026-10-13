from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7
VOTE_COLORS = (BLUE, RED, GREEN)
GEN_ATTEMPTS = 600
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    r0 = int(r0)
    c0 = int(c0)
    h, w = map(int, patch.shape)
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.asarray(coords, dtype=int))
    return out


def _separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    assert len(rows) == 1, "expected one full-width grey separator row"
    row = int(rows[0])
    assert 0 < row < grid.shape[0] - 1
    return row


def _majority_vote_color(region: np.ndarray) -> int:
    values = region[(region != BG) & (region != GREY)]
    assert values.size > 0, "upper region must contain voting pixels"
    counts = Counter(map(int, values))
    ranked = counts.most_common()
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "vote winner must be unique"
    return int(ranked[0][0])


def _extract_candidates(lower: np.ndarray) -> list[dict]:
    candidates: list[dict] = []
    for color in VOTE_COLORS:
        for coords in _components_of_color(lower, color):
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            box_h = int(r1 - r0 + 1)
            box_w = int(c1 - c0 + 1)
            if box_h < 3 or box_w < 3:
                continue
            patch = lower[r0 : r1 + 1, c0 : c1 + 1]
            border_mask = np.zeros((box_h, box_w), dtype=bool)
            border_mask[0, :] = True
            border_mask[-1, :] = True
            border_mask[:, 0] = True
            border_mask[:, -1] = True
            if not np.all(patch[border_mask] == color):
                continue
            if np.any(patch[1:-1, 1:-1] == color):
                continue
            if int(coords.shape[0]) != int(border_mask.sum()):
                continue
            inner = patch[1:-1, 1:-1].copy()
            candidates.append(
                {
                    "color": int(color),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "inner": inner,
                }
            )
    colors = [cand["color"] for cand in candidates]
    assert len(candidates) == len(set(colors)), "candidate border colors must be unique"
    assert 2 <= len(candidates) <= 3, "expected two or three candidate sprites"
    return candidates


def _ring_output(template: np.ndarray) -> np.ndarray:
    h, w = map(int, template.shape)
    palette_size = len(np.unique(template))
    if h >= w:
        side = h + 2 * w
        center = palette_size * (h // palette_size)
        assert center > 0
        out = np.zeros((side, side), dtype=int)
        center0 = (side - center) // 2
        band0 = (side - h) // 2
        _paste(out, np.rot90(template, -1), center0 - w, band0)
        _paste(out, template, band0, center0 - w)
        _paste(out, np.fliplr(template), band0, center0 + center)
        _paste(out, np.rot90(template, 1), center0 + center, band0)
        return out

    side = w + 2 * h
    center = palette_size * (w // palette_size)
    assert center > 0
    out = np.zeros((side, side), dtype=int)
    center0 = (side - center) // 2
    band0 = (side - w) // 2
    _paste(out, template, center0 - h, band0)
    _paste(out, np.rot90(template, 1), band0, center0 - h)
    _paste(out, np.rot90(template, -1), band0, center0 + center)
    _paste(out, np.fliplr(template), center0 + center, band0)
    return out


def _canonical_template_key(template: np.ndarray) -> tuple[bytes, tuple[int, int]]:
    variants: list[tuple[bytes, tuple[int, int]]] = []
    cur = template.copy()
    for _ in range(4):
        variants.append((cur.tobytes(), tuple(map(int, cur.shape))))
        flip = np.fliplr(cur)
        variants.append((flip.tobytes(), tuple(map(int, flip.shape))))
        cur = np.rot90(cur)
    return min(variants)


def _orange_connected(template: np.ndarray) -> bool:
    orange = template == ORANGE
    coords = np.argwhere(orange)
    if coords.size == 0:
        return False
    seen = {tuple(map(int, coords[0]))}
    q = deque([tuple(map(int, coords[0]))])
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < template.shape[0] and 0 <= nc < template.shape[1] and orange[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(orange.sum())


def _not_quarter_turn_symmetric(template: np.ndarray) -> bool:
    if template.shape[0] != template.shape[1]:
        return True
    return not np.array_equal(template, np.rot90(template))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the upper-region majority color to select one bordered sprite, then build a four-sided ring from its interior.

    Input structure:
    - A full-width grey separator row splits the grid into an upper voting region and a lower sprite region.
    - The upper region is mostly black and contains scattered red/blue/green voting pixels with one unique majority color.
    - The lower region contains two or three non-overlapping rectangular candidate sprites, each defined by a one-cell-thick colored border and an orange-on-black interior pattern.

    Rule:
    - Count the non-black, non-grey colors in the upper region and find the unique majority color.
    - In the lower region, locate the candidate whose border color matches that winning color.
    - Crop away the border and keep only the interior template.
    - Let D be the number of distinct colors in that template. If the template is taller than wide, create a square canvas of side h + 2w; otherwise use w + 2h.
    - Arrange four oriented copies around a centered blank square whose side is the largest multiple of D not exceeding the template's long side: for tall templates use top=clockwise rotation, left=original, right=horizontal flip, bottom=counterclockwise rotation; for wide templates rotate that layout by ninety degrees.

    Key insight:
    - The majority vote does not write an output color directly; it acts as an index that selects which lower sprite is cropped and transformed into the final ring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    sep = _separator_row(g)
    winner = _majority_vote_color(g[:sep, :])
    candidates = _extract_candidates(g[sep + 1 :, :])
    chosen = [cand for cand in candidates if cand["color"] == winner]
    assert len(chosen) == 1, "vote winner must match exactly one candidate sprite"
    return _ring_output(chosen[0]["inner"])


def _generate_connected_pattern(h: int, w: int, count: int) -> np.ndarray | None:
    count = int(count)
    cells = {(random.randrange(h), random.randrange(w))}
    while len(cells) < count:
        frontier = []
        for r, c in cells:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))

    template = np.zeros((h, w), dtype=int)
    for r, c in cells:
        template[r, c] = ORANGE
    if len(np.unique(template)) != 2:
        return None
    if not _orange_connected(template):
        return None
    if not _not_quarter_turn_symmetric(template):
        return None
    return template


def _build_candidate_template(orientation: str | None = None, parity: str | None = None) -> np.ndarray | None:
    if orientation is None:
        orientation = random.choice(["tall", "wide"])

    if orientation == "tall":
        short_side = random.randint(3, 5)
        long_choices = [v for v in range(short_side, 8)]
    else:
        short_side = random.randint(3, 5)
        long_choices = [v for v in range(short_side, 8)]

    if parity is not None:
        long_choices = [v for v in long_choices if v % 2 == (0 if parity == "even" else 1)]
    if not long_choices:
        return None

    long_side = random.choice(long_choices)
    if orientation == "tall":
        h, w = long_side, short_side
    else:
        h, w = short_side, long_side

    max_orange = min(5, h * w - 1)
    if max_orange < 3:
        return None
    for _ in range(80):
        orange_count = random.randint(3, max_orange)
        template = _generate_connected_pattern(h, w, orange_count)
        if template is not None:
            return template
    return None


def _candidate_patch(border_color: int, template: np.ndarray) -> np.ndarray:
    h, w = template.shape
    patch = np.full((h + 2, w + 2), border_color, dtype=int)
    patch[1:-1, 1:-1] = template
    return patch


def _gap_at_least_two(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    row_gap = 0
    if ar1 < br0:
        row_gap = br0 - ar1 - 1
    elif br1 < ar0:
        row_gap = ar0 - br1 - 1
    col_gap = 0
    if ac1 < bc0:
        col_gap = bc0 - ac1 - 1
    elif bc1 < ac0:
        col_gap = ac0 - bc1 - 1
    return row_gap + col_gap >= 2


def _place_patches(lower_h: int, width: int, patches: list[np.ndarray]) -> list[tuple[int, int]] | None:
    order = list(range(len(patches)))
    random.shuffle(order)
    order.sort(key=lambda idx: patches[idx].shape[0] * patches[idx].shape[1], reverse=True)
    positions: list[tuple[int, int] | None] = [None] * len(patches)
    placed_boxes: list[tuple[int, int, int, int]] = []

    def rec(i: int) -> bool:
        if i == len(order):
            return True
        idx = order[i]
        patch = patches[idx]
        h, w = patch.shape
        candidates = []
        for r0 in range(lower_h - h + 1):
            for c0 in range(width - w + 1):
                box = (r0, c0, r0 + h - 1, c0 + w - 1)
                if all(_gap_at_least_two(box, other) for other in placed_boxes):
                    candidates.append((r0, c0, box))
        random.shuffle(candidates)
        for r0, c0, box in candidates:
            positions[idx] = (r0, c0)
            placed_boxes.append(box)
            if rec(i + 1):
                return True
            placed_boxes.pop()
            positions[idx] = None
        return False

    if not rec(0):
        return None
    return [tuple(pos) for pos in positions if pos is not None]


def _sample_vote_counts(colors: list[int], winner: int, capacity: int) -> dict[int, int] | None:
    others = [c for c in colors if c != winner]
    if len(colors) == 2:
        win_count = random.randint(6, 9)
        other_count = random.randint(3, min(7, win_count - 1))
        counts = {winner: win_count, others[0]: other_count}
    else:
        win_count = random.randint(6, 10)
        counts = {winner: win_count}
        for color in others:
            counts[color] = random.randint(3, 6)
        if any(counts[c] >= win_count for c in others):
            return None
    total = sum(counts.values())
    return counts if total <= capacity else None


def _scatter_votes(height: int, width: int, counts: dict[int, int]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    cells = [(r, c) for r in range(height) for c in range(width)]
    random.shuffle(cells)
    pos = 0
    for color, count in counts.items():
        for _ in range(count):
            r, c = cells[pos]
            grid[r, c] = int(color)
            pos += 1
    return grid


def _expected_from_meta(meta: dict) -> np.ndarray:
    return _ring_output(meta["winner_template"])


def _validate_case(grid: np.ndarray, meta: dict) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    sep = _separator_row(grid)
    if sep != meta["top_h"]:
        return False

    upper = grid[:sep, :]
    if set(np.unique(upper)) - {BG, *meta["colors"]}:
        return False
    if _majority_vote_color(upper) != meta["winner_color"]:
        return False

    lower = grid[sep + 1 :, :]
    candidates = _extract_candidates(lower)
    if sorted(c["color"] for c in candidates) != sorted(meta["colors"]):
        return False

    key_by_color = {cand["color"]: _canonical_template_key(cand["inner"]) for cand in candidates}
    for record in meta["records"]:
        if key_by_color[record["color"]] != record["key"]:
            return False

    for cand in candidates:
        template = cand["inner"]
        if len(np.unique(template)) != 2:
            return False
        orange_count = int(np.count_nonzero(template == ORANGE))
        if not (3 <= orange_count <= 5):
            return False
        if np.any((template != BG) & (template != ORANGE)):
            return False
        if not _orange_connected(template):
            return False
        if not _not_quarter_turn_symmetric(template):
            return False

    boxes = [cand["bbox"] for cand in candidates]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if not _gap_at_least_two(boxes[i], boxes[j]):
                return False

    winner_candidates = [cand for cand in candidates if cand["color"] == meta["winner_color"]]
    if len(winner_candidates) != 1:
        return False
    if not np.array_equal(winner_candidates[0]["inner"], meta["winner_template"]):
        return False

    expected = _expected_from_meta(meta)
    if max(expected.shape) > 30:
        return False
    return np.array_equal(main(grid), expected)


def _gen_attempt() -> np.ndarray | None:
    candidate_count = random.choice([2, 3])
    colors = random.sample(list(VOTE_COLORS), candidate_count)
    winner_color = random.choice(colors)
    winner_orientation = random.choice(["tall", "wide"])
    winner_parity = random.choice(["even", "odd"])
    desired_rank = random.randrange(candidate_count)

    winner_template = _build_candidate_template(winner_orientation, winner_parity)
    if winner_template is None:
        return None

    records = [
        {
            "color": winner_color,
            "template": winner_template,
            "key": _canonical_template_key(winner_template),
        }
    ]
    used_keys = {records[0]["key"]}

    for color in colors:
        if color == winner_color:
            continue
        for _ in range(120):
            template = _build_candidate_template()
            if template is None:
                continue
            key = _canonical_template_key(template)
            if key in used_keys:
                continue
            used_keys.add(key)
            records.append({"color": color, "template": template, "key": key})
            break
        else:
            return None

    random.shuffle(records)
    winner_template = next(rec["template"] for rec in records if rec["color"] == winner_color)

    areas = sorted(
        ((int(rec["template"].shape[0] * rec["template"].shape[1]), idx) for idx, rec in enumerate(records)),
        key=lambda x: x[0],
    )
    winner_idx = next(idx for idx, rec in enumerate(records) if rec["color"] == winner_color)
    winner_rank = [idx for idx, (_area, idx) in enumerate(areas) if idx == winner_idx][0]
    if winner_rank != desired_rank:
        return None

    patches = [_candidate_patch(rec["color"], rec["template"]) for rec in records]
    max_patch_h = max(patch.shape[0] for patch in patches)
    min_width = max(max(patch.shape[1] for patch in patches), 14)
    lower_h = random.randint(max(10, max_patch_h + 1), 16)
    width = random.randint(min_width, 24)
    positions = _place_patches(lower_h, width, patches)
    if positions is None:
        return None

    top_h = random.randint(4, 6)
    total_h = top_h + 1 + lower_h
    if total_h > 30:
        return None

    vote_counts = _sample_vote_counts(colors, winner_color, top_h * width)
    if vote_counts is None:
        return None
    upper = _scatter_votes(top_h, width, vote_counts)

    grid = np.zeros((total_h, width), dtype=int)
    grid[:top_h, :] = upper
    grid[top_h, :] = GREY
    lower = grid[top_h + 1 :, :]
    for patch, (r0, c0) in zip(patches, positions):
        _paste(lower, patch, r0, c0)

    meta = {
        "colors": colors,
        "winner_color": winner_color,
        "winner_template": winner_template,
        "top_h": top_h,
        "records": records,
    }
    if not _validate_case(grid, meta):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 10")
