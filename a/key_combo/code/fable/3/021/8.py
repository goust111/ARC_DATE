from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
GENERATION_ATTEMPTS = 800

_GEN_CALL_COUNT = 0


def _background(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        nbrs = ((-1, 0), (1, 0), (0, -1), (0, 1))
    else:
        nbrs = tuple(
            (dr, dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if not (dr == 0 and dc == 0)
        )

    comps: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        seen[sr, sc] = True
        cells: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            cells.append((r, c))
            for dr, dc in nbrs:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]) and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        comps.append(np.array(cells, dtype=int))
    return comps


def _border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((int(height), int(width)), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _find_template(grid: np.ndarray, bg: int) -> dict:
    candidates: list[dict] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != int(bg)):
        coords = np.argwhere(grid == int(color))
        if coords.size == 0:
            continue

        r0, c0, r1, c1 = _bbox_coords(coords)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height < 5 or width < 5:
            continue

        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = _border_mask(height, width)
        interior = ~border
        if not bool(np.all(sub[border] == int(color))):
            continue
        motif_mask = (sub == int(color)) & interior
        if not bool(motif_mask.any()):
            continue
        if not bool(np.any(sub[interior] == int(bg))):
            continue

        motif_rel = [(int(r), int(c)) for r, c in np.argwhere(motif_mask)]
        candidates.append(
            {
                "color": int(color),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "height": int(height),
                "width": int(width),
                "motif": motif_rel,
            }
        )

    assert len(candidates) == 1, "expected one complete patterned frame"
    return candidates[0]


def _corner_fragment(coords: np.ndarray) -> dict | None:
    r0, c0, r1, c1 = _bbox_coords(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 2 or width < 2:
        return None

    rel = {(int(r - r0), int(c - c0)) for r, c in coords}
    patterns = {
        "UL": ({(0, c) for c in range(width)} | {(r, 0) for r in range(height)}, (r0, c0)),
        "UR": ({(0, c) for c in range(width)} | {(r, width - 1) for r in range(height)}, (r0, c1)),
        "LL": ({(height - 1, c) for c in range(width)} | {(r, 0) for r in range(height)}, (r1, c0)),
        "LR": (
            {(height - 1, c) for c in range(width)} | {(r, width - 1) for r in range(height)},
            (r1, c1),
        ),
    }

    matches = [
        (kind, corner)
        for kind, (pattern, corner) in patterns.items()
        if rel == pattern
    ]
    if len(matches) != 1:
        return None
    kind, corner = matches[0]
    return {"kind": kind, "corner": (int(corner[0]), int(corner[1]))}


def _rect_from_opposite_corners(parts: list[dict]) -> tuple[int, int, int, int] | None:
    by_kind = {part["kind"]: part["corner"] for part in parts}
    if set(by_kind) == {"UL", "LR"}:
        top_r, left_c = by_kind["UL"]
        bottom_r, right_c = by_kind["LR"]
    elif set(by_kind) == {"UR", "LL"}:
        top_r, right_c = by_kind["UR"]
        bottom_r, left_c = by_kind["LL"]
    else:
        return None

    if top_r >= bottom_r or left_c >= right_c:
        return None
    return int(top_r), int(left_c), int(bottom_r), int(right_c)


def _draw_frame_and_motif(
    out: np.ndarray,
    rect: tuple[int, int, int, int],
    color: int,
    motif_rel: list[tuple[int, int]],
) -> None:
    r0, c0, r1, c1 = rect
    out[r0, c0 : c1 + 1] = int(color)
    out[r1, c0 : c1 + 1] = int(color)
    out[r0 : r1 + 1, c0] = int(color)
    out[r0 : r1 + 1, c1] = int(color)
    for rr, cc in motif_rel:
        out[r0 + int(rr), c0 + int(cc)] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore same-size frames from opposite corner fragments and copy the template motif.

    Input structure:
    - The background is the dominant color.
    - Exactly one foreground color forms a complete one-cell-thick rectangular frame
      with same-colored motif pixels inside it; this is the template frame.
    - Every other foreground color appears as exactly two L-shaped corner fragments,
      and the two fragments are opposite corners of one implied rectangle.
    - Some implied rectangles have the same outer height and width as the template,
      while the rest are size decoys.

    Rule:
    - For each fragment color, infer its full outer rectangle from the two opposite
      L corner pieces.
    - If that inferred rectangle has the same height and width as the template, erase
      the fragments, draw the complete one-cell border in that color, and stamp the
      template's interior motif at the same relative positions in that color.
    - If the inferred rectangle has a different size, erase both fragments to the
      background.
    - Keep the original template frame and its motif unchanged.

    Key insight:
    - The visible fragments are only residue: the template first supplies the only
      valid size and motif, then each color's opposite corners decide whether it is
      restored or erased.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = _background(g)
    template = _find_template(g, bg)
    template_size = (int(template["height"]), int(template["width"]))

    out = g.copy()
    fragment_colors = [
        int(c)
        for c in np.unique(g)
        if int(c) not in {int(bg), int(template["color"])}
    ]

    for color in fragment_colors:
        out[g == int(color)] = int(bg)

    for color in fragment_colors:
        comps = _components(g == int(color), connectivity=4)
        assert len(comps) == 2, "each non-template color must have two fragments"
        parts = [_corner_fragment(comp) for comp in comps]
        assert all(part is not None for part in parts), "fragments must be L-shaped corners"
        rect = _rect_from_opposite_corners(parts)  # type: ignore[arg-type]
        assert rect is not None, "fragments must occupy opposite rectangle corners"

        r0, c0, r1, c1 = rect
        size = (int(r1 - r0 + 1), int(c1 - c0 + 1))
        if size == template_size:
            _draw_frame_and_motif(out, rect, int(color), template["motif"])

    return out


def _paste_nonzero(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    mask = patch != BG
    view = dst[int(r0) : int(r0) + patch.shape[0], int(c0) : int(c0) + patch.shape[1]]
    view[mask] = patch[mask]


def _random_motif(height: int, width: int) -> list[tuple[int, int]]:
    pool = [(r, c) for r in range(2, height - 2) for c in range(2, width - 2)]
    if not pool:
        pool = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]

    target = random.randint(2, min(6, len(pool)))
    cells = {random.choice(pool)}
    pool_set = set(pool)
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (rr, cc) in pool_set and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        if frontier:
            cells.add(random.choice(frontier))
        else:
            cells.add(random.choice([cell for cell in pool if cell not in cells]))

    if len({r for r, _c in cells}) == 1 and height >= 7:
        r, c = next(iter(cells))
        for rr in (r - 1, r + 1):
            if (rr, c) in pool_set:
                cells.add((rr, c))
                break
    if len({c for _r, c in cells}) == 1 and width >= 7:
        r, c = next(iter(cells))
        for cc in (c - 1, c + 1):
            if (r, cc) in pool_set:
                cells.add((r, cc))
                break

    return sorted(cells)


def _frame_patch(height: int, width: int, color: int, motif: list[tuple[int, int]]) -> np.ndarray:
    patch = np.full((int(height), int(width)), BG, dtype=int)
    _draw_frame_and_motif(patch, (0, 0, int(height) - 1, int(width) - 1), int(color), motif)
    return patch


def _fragment_patch(
    height: int,
    width: int,
    color: int,
    diagonal: str,
    row_arm: int,
    col_arm: int,
) -> np.ndarray:
    patch = np.full((int(height), int(width)), BG, dtype=int)
    h = int(height)
    w = int(width)
    ra = int(row_arm)
    ca = int(col_arm)

    def draw(kind: str) -> None:
        if kind == "UL":
            patch[0:ra, 0] = int(color)
            patch[0, 0:ca] = int(color)
        elif kind == "UR":
            patch[0:ra, w - 1] = int(color)
            patch[0, w - ca : w] = int(color)
        elif kind == "LL":
            patch[h - ra : h, 0] = int(color)
            patch[h - 1, 0:ca] = int(color)
        elif kind == "LR":
            patch[h - ra : h, w - 1] = int(color)
            patch[h - 1, w - ca : w] = int(color)
        else:
            raise AssertionError("unknown fragment corner")

    if diagonal == "UL_LR":
        draw("UL")
        draw("LR")
    else:
        assert diagonal == "UR_LL"
        draw("UR")
        draw("LL")
    return patch


def _free_locations(
    occupied: np.ndarray,
    height: int,
    width: int,
    padding: int = 1,
) -> list[tuple[int, int]]:
    h, w = occupied.shape
    locs: list[tuple[int, int]] = []
    for r0 in range(padding, h - int(height) - padding + 1):
        for c0 in range(padding, w - int(width) - padding + 1):
            rr0 = max(0, r0 - padding)
            cc0 = max(0, c0 - padding)
            rr1 = min(h, r0 + int(height) + padding)
            cc1 = min(w, c0 + int(width) + padding)
            if not bool(occupied[rr0:rr1, cc0:cc1].any()):
                locs.append((int(r0), int(c0)))
    return locs


def _decoy_size(template_h: int, template_w: int, branch: int) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    for delta in (-2, -1, 1, 2):
        candidates.append((template_h, template_w + delta))
        candidates.append((template_h + delta, template_w))
        candidates.append((template_h + delta, template_w + random.choice((-1, 1))))
    if template_h != template_w:
        candidates.append((template_w, template_h))

    clean = [
        (int(h), int(w))
        for h, w in candidates
        if 5 <= int(h) <= 11 and 5 <= int(w) <= 11 and (int(h), int(w)) != (template_h, template_w)
    ]
    if branch % 4 == 0:
        focused = [(h, w) for h, w in clean if h == template_h and w != template_w]
    elif branch % 4 == 1:
        focused = [(h, w) for h, w in clean if w == template_w and h != template_h]
    elif branch % 4 == 2:
        focused = [(h, w) for h, w in clean if h == template_w and w == template_h]
    else:
        focused = [(h, w) for h, w in clean if h != template_h and w != template_w]
    return random.choice(focused or clean)


def _place_objects(
    grid_h: int,
    grid_w: int,
    objects: list[dict],
) -> tuple[np.ndarray, list[dict]] | None:
    grid = np.full((int(grid_h), int(grid_w)), BG, dtype=int)
    occupied = np.zeros((int(grid_h), int(grid_w)), dtype=bool)
    placed: list[dict] = []

    random.shuffle(objects)
    for obj in objects:
        patch = obj["patch"]
        locs = _free_locations(occupied, patch.shape[0], patch.shape[1], padding=1)
        if not locs:
            return None
        r0, c0 = random.choice(locs)
        _paste_nonzero(grid, patch, r0, c0)
        occupied[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = True
        item = dict(obj)
        item["top"] = int(r0)
        item["left"] = int(c0)
        placed.append(item)

    return grid, placed


def _expected_from_metadata(
    grid: np.ndarray,
    placed: list[dict],
    template_h: int,
    template_w: int,
    motif: list[tuple[int, int]],
    template_color: int,
) -> np.ndarray:
    out = grid.copy()
    for obj in placed:
        if obj["kind"] == "fragment":
            out[grid == int(obj["color"])] = BG

    for obj in placed:
        if obj["kind"] != "fragment":
            continue
        if (int(obj["height"]), int(obj["width"])) != (int(template_h), int(template_w)):
            continue
        r0 = int(obj["top"])
        c0 = int(obj["left"])
        rect = (r0, c0, r0 + int(template_h) - 1, c0 + int(template_w) - 1)
        _draw_frame_and_motif(out, rect, int(obj["color"]), motif)

    assert np.all(out[grid == int(template_color)] == int(template_color))
    return out


def _semantic_checks(
    grid: np.ndarray,
    expected: np.ndarray,
    placed: list[dict],
    template_h: int,
    template_w: int,
    motif: list[tuple[int, int]],
    template_color: int,
) -> bool:
    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(grid, expected):
        return False

    fragments = [obj for obj in placed if obj["kind"] == "fragment"]
    matches = [
        obj
        for obj in fragments
        if (int(obj["height"]), int(obj["width"])) == (int(template_h), int(template_w))
    ]
    decoys = [
        obj
        for obj in fragments
        if (int(obj["height"]), int(obj["width"])) != (int(template_h), int(template_w))
    ]
    if not matches or not decoys:
        return False

    if not any(
        int(obj["height"]) == int(template_h) or int(obj["width"]) == int(template_w)
        for obj in decoys
    ):
        return False

    for obj in fragments:
        comps = _components(grid == int(obj["color"]), connectivity=4)
        if len(comps) != 2:
            return False
        if any(_corner_fragment(comp) is None for comp in comps):
            return False

    border_area = 2 * int(template_h) + 2 * int(template_w) - 4
    for obj in matches:
        color = int(obj["color"])
        if not bool(np.any((actual == color) & (grid != color))):
            return False
        if int(np.sum(actual == color)) != border_area + len(motif):
            return False
        for rr, cc in motif:
            if int(actual[int(obj["top"]) + rr, int(obj["left"]) + cc]) != color:
                return False

    for obj in decoys:
        if bool(np.any(actual == int(obj["color"]))):
            return False

    if not np.array_equal(actual[grid == int(template_color)], grid[grid == int(template_color)]):
        return False
    return True


def _gen_attempt(call_index: int) -> np.ndarray | None:
    grid_h = random.randint(22, 30)
    grid_w = random.randint(22, 30)

    template_h = random.randint(6, 9)
    template_w = random.randint(6, 10)
    if template_h == template_w and random.random() < 0.75:
        template_w = random.choice([w for w in range(6, 11) if w != template_h])

    match_count = random.randint(1, 2)
    decoy_count = random.randint(1, 2)
    needed_colors = 1 + match_count + decoy_count
    chosen_colors = random.sample(list(COLORS), needed_colors)
    template_color = chosen_colors[0]
    fragment_colors = chosen_colors[1:]

    motif = _random_motif(template_h, template_w)
    objects: list[dict] = [
        {
            "kind": "template",
            "color": int(template_color),
            "patch": _frame_patch(template_h, template_w, template_color, motif),
            "height": int(template_h),
            "width": int(template_w),
        }
    ]

    diagonals = ["UL_LR", "UR_LL"]
    for i in range(match_count):
        color = fragment_colors[i]
        diagonal = diagonals[(call_index + i) % 2] if i == 0 else random.choice(diagonals)
        objects.append(
            {
                "kind": "fragment",
                "color": int(color),
                "height": int(template_h),
                "width": int(template_w),
                "diagonal": diagonal,
                "patch": _fragment_patch(
                    template_h,
                    template_w,
                    color,
                    diagonal,
                    random.randint(2, min(3, template_h - 2)),
                    random.randint(2, min(3, template_w - 2)),
                ),
            }
        )

    for j in range(decoy_count):
        color = fragment_colors[match_count + j]
        decoy_h, decoy_w = _decoy_size(template_h, template_w, call_index + j)
        diagonal = diagonals[(call_index + j + 1) % 2]
        objects.append(
            {
                "kind": "fragment",
                "color": int(color),
                "height": int(decoy_h),
                "width": int(decoy_w),
                "diagonal": diagonal,
                "patch": _fragment_patch(
                    decoy_h,
                    decoy_w,
                    color,
                    diagonal,
                    random.randint(2, min(3, decoy_h - 2)),
                    random.randint(2, min(3, decoy_w - 2)),
                ),
            }
        )

    placed_result = _place_objects(grid_h, grid_w, objects)
    if placed_result is None:
        return None
    grid, placed = placed_result
    expected = _expected_from_metadata(
        grid,
        placed,
        template_h,
        template_w,
        motif,
        template_color,
    )
    if not _semantic_checks(grid, expected, placed, template_h, template_w, motif, template_color):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    _GEN_CALL_COUNT += 1
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(_GEN_CALL_COUNT)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid patterned corner-restoration input")
