import numpy as np


BG = 0
GREY = 5
MAIN_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
ARC_MAX = 30
GEN_RETRY_LIMIT = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _crop_non_bg(arr: np.ndarray, bg: int = BG) -> np.ndarray:
    coords = np.argwhere(arr != bg)
    r0, c0, r1, c1 = _bbox(coords)
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy()


def _parse_objects(grid: np.ndarray) -> list[dict]:
    comps = _components(grid != BG)
    objects: list[dict] = []
    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        colors = set(map(int, patch[patch != BG].ravel()))
        main_colors = sorted(colors - {GREY})
        assert len(main_colors) == 1, "each fragment must have exactly one non-grey main color"
        objects.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "patch": patch,
                "main_color": int(main_colors[0]),
                "left": int(c0),
                "top": int(r0),
            }
        )
    return objects


def _blit(canvas: np.ndarray, occ: np.ndarray, patch: np.ndarray, top: int, left: int) -> bool:
    h, w = patch.shape
    mask = patch != BG
    sub_occ = occ[top : top + h, left : left + w]
    if np.any(sub_occ & mask):
        return False
    sub_canvas = canvas[top : top + h, left : left + w]
    sub_canvas[mask] = patch[mask]
    canvas[top : top + h, left : left + w] = sub_canvas
    sub_occ |= mask
    occ[top : top + h, left : left + w] = sub_occ
    return True


def _contact_count(mask_a: np.ndarray, mask_b: np.ndarray) -> int:
    return int(
        np.sum(mask_a[:, 1:] & mask_b[:, :-1])
        + np.sum(mask_a[:, :-1] & mask_b[:, 1:])
        + np.sum(mask_a[1:, :] & mask_b[:-1, :])
        + np.sum(mask_a[:-1, :] & mask_b[1:, :])
    )


def _assemble_pair(patch_a: np.ndarray, patch_b: np.ndarray) -> np.ndarray:
    ha, wa = patch_a.shape
    hb, wb = patch_b.shape
    area_a = int(np.count_nonzero(patch_a != BG))
    area_b = int(np.count_nonzero(patch_b != BG))
    total_area = area_a + area_b
    best_score: tuple[int, int, int, int] | None = None
    best_canvas: np.ndarray | None = None
    seen: set[tuple[int, int, bytes]] = set()

    for dy in range(-hb, ha + 1):
        for dx in range(-wb, wa + 1):
            top = min(0, dy)
            left = min(0, dx)
            bottom = max(ha, dy + hb)
            right = max(wa, dx + wb)
            out_h = bottom - top
            out_w = right - left
            if out_h * out_w != total_area:
                continue

            canvas = np.full((out_h, out_w), BG, dtype=int)
            occ = np.zeros((out_h, out_w), dtype=bool)
            if not _blit(canvas, occ, patch_a, -top, -left):
                continue
            if not _blit(canvas, occ, patch_b, dy - top, dx - left):
                continue

            mask_a = np.zeros((out_h, out_w), dtype=bool)
            mask_b = np.zeros((out_h, out_w), dtype=bool)
            mask_a[-top : -top + ha, -left : -left + wa] = patch_a != BG
            mask_b[dy - top : dy - top + hb, dx - left : dx - left + wb] = patch_b != BG
            contact = _contact_count(mask_a, mask_b)
            if contact <= 0:
                continue

            sig = (out_h, out_w, canvas.tobytes())
            if sig in seen:
                continue
            seen.add(sig)

            score = (
                contact,
                -(abs(dy) + abs(dx)),
                -abs(dy),
                -abs(dx),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_canvas = canvas

    assert best_canvas is not None, "same-color fragments must assemble into a rectangle"
    return best_canvas


def _stripe_is_vertical(rect: np.ndarray) -> bool:
    grey = rect == GREY
    assert np.any(grey), "assembled rectangle must contain grey stripes"
    horiz_adj = int(np.sum(grey[:, 1:] & grey[:, :-1]))
    vert_adj = int(np.sum(grey[1:, :] & grey[:-1, :]))
    if horiz_adj != vert_adj:
        return vert_adj > horiz_adj
    grey_rows = int(np.count_nonzero(grey.any(axis=1)))
    grey_cols = int(np.count_nonzero(grey.any(axis=0)))
    return grey_rows > grey_cols


def _pack_records(records: list[dict]) -> np.ndarray:
    ordered = sorted(records, key=lambda rec: (rec["area"], rec["sort_left"], rec["sort_top"]))
    out_h = max(int(rec["arr"].shape[0]) for rec in ordered)
    out_w = sum(int(rec["arr"].shape[1]) for rec in ordered) + len(ordered) - 1
    out = np.full((out_h, out_w), BG, dtype=int)
    col = 0
    for rec in ordered:
        arr = rec["arr"]
        h, w = arr.shape
        row = out_h - h
        out[row : row + h, col : col + w] = arr
        col += w + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored jagged fragments into rectangles, rotate the vertically striped ones, and repack them.

    Input structure:
    - The background is black.
    - Each foreground connected component is one fragment containing a single non-grey main color plus 1 to 3 grey stripe segments.
    - Every main color appears on exactly two fragments, and those two fragments are complementary halves of one rectangle split by a 1-pixel-deep alternating jagged edge.
    - Fragments are scattered apart and do not touch each other.

    Rule:
    - Group fragments by their non-grey main color.
    - For each same-colored pair, translate the two fragments until they fit together with no overlap and their union exactly fills a rectangle, preserving the grey stripes.
    - If the assembled rectangle has horizontal grey stripes, keep it unchanged; if the grey stripes are vertical, rotate the whole rectangle 90 degrees clockwise.
    - Sort the resulting rectangles by increasing area, breaking ties by their left-to-right position in the input, and place them left-to-right with one black separator column and aligned along the bottom edge.

    Key insight:
    - Color tells you which fragments belong together, but the actual placement is found by the complementary jagged edge: the correct translation is the one whose disjoint union becomes a perfect rectangle.
    """
    g = np.asarray(grid, dtype=int)
    objects = _parse_objects(g)
    by_color: dict[int, list[dict]] = {}
    for obj in objects:
        by_color.setdefault(obj["main_color"], []).append(obj)

    records: list[dict] = []
    for color, pair in by_color.items():
        assert len(pair) == 2, f"color {color} must have exactly two fragments"
        pair = sorted(pair, key=lambda obj: (obj["left"], obj["top"]))
        rect = _assemble_pair(pair[0]["patch"], pair[1]["patch"])
        if _stripe_is_vertical(rect):
            rect = np.rot90(rect, -1)
        area = int(rect.shape[0] * rect.shape[1])
        records.append(
            {
                "arr": rect,
                "area": area,
                "sort_left": min(pair[0]["left"], pair[1]["left"]),
                "sort_top": min(pair[0]["top"], pair[1]["top"]),
            }
        )

    return _pack_records(records)


def _sample_parallel_positions(rng: np.random.Generator, pool_a: list[int], pool_b: list[int]) -> list[int] | None:
    total = len(pool_a) + len(pool_b)
    max_count = min(3, total)
    if max_count < 2:
        return None
    count = int(rng.integers(2, max_count + 1))
    min_a = max(1, count - len(pool_b))
    max_a = min(len(pool_a), count - 1)
    if min_a > max_a:
        return None
    count_a = int(rng.integers(min_a, max_a + 1))
    count_b = count - count_a
    picks_a = list(map(int, rng.choice(pool_a, size=count_a, replace=False)))
    picks_b = list(map(int, rng.choice(pool_b, size=count_b, replace=False)))
    return sorted(picks_a + picks_b)


def _make_pair_spec(rng: np.random.Generator, color: int, max_dim: int) -> dict | None:
    split_kind = "vertical" if int(rng.integers(2)) == 0 else "horizontal"
    stripe_kind = "horizontal" if int(rng.integers(2)) == 0 else "vertical"

    if split_kind == "vertical":
        h = int(rng.integers(4, max_dim + 1))
        w = int(rng.integers(5, max_dim + 1))
        if w < 5:
            return None
        base = int(rng.integers(2, w - 2))
    else:
        h = int(rng.integers(5, max_dim + 1))
        w = int(rng.integers(4, max_dim + 1))
        if h < 5:
            return None
        base = int(rng.integers(2, h - 2))

    rect = np.full((h, w), int(color), dtype=int)

    if stripe_kind == "horizontal":
        if split_kind == "horizontal":
            top_pool = list(range(1, base))
            bottom_pool = list(range(base + 1, h - 1))
            stripe_rows = _sample_parallel_positions(rng, top_pool, bottom_pool)
            if stripe_rows is None:
                return None
        else:
            pool = list(range(1, h - 1))
            count = int(rng.integers(1, min(3, len(pool)) + 1))
            stripe_rows = sorted(map(int, rng.choice(pool, size=count, replace=False)))
        rect[np.array(stripe_rows, dtype=int), :] = GREY
    else:
        if split_kind == "vertical":
            left_pool = list(range(1, base))
            right_pool = list(range(base + 1, w - 1))
            stripe_cols = _sample_parallel_positions(rng, left_pool, right_pool)
            if stripe_cols is None:
                return None
        else:
            pool = list(range(1, w - 1))
            count = int(rng.integers(1, min(3, len(pool)) + 1))
            stripe_cols = sorted(map(int, rng.choice(pool, size=count, replace=False)))
        rect[:, np.array(stripe_cols, dtype=int)] = GREY

    if split_kind == "vertical":
        widths = base + ((np.arange(h) + int(rng.integers(2))) % 2)
        if int(rng.integers(2)) == 1:
            widths = widths[::-1]
        left_mask = np.zeros((h, w), dtype=bool)
        for r, width in enumerate(widths):
            left_mask[r, : int(width)] = True
        right_mask = ~left_mask
        piece_a = rect.copy()
        piece_b = rect.copy()
        piece_a[~left_mask] = BG
        piece_b[~right_mask] = BG
    else:
        heights = base + ((np.arange(w) + int(rng.integers(2))) % 2)
        if int(rng.integers(2)) == 1:
            heights = heights[::-1]
        top_mask = np.zeros((h, w), dtype=bool)
        for c, height in enumerate(heights):
            top_mask[: int(height), c] = True
        bottom_mask = ~top_mask
        piece_a = rect.copy()
        piece_b = rect.copy()
        piece_a[~top_mask] = BG
        piece_b[~bottom_mask] = BG

    piece_a = _crop_non_bg(piece_a)
    piece_b = _crop_non_bg(piece_b)
    if _stripe_is_vertical(rect):
        final_rect = np.rot90(rect, -1)
    else:
        final_rect = rect.copy()

    return {
        "pieces": (piece_a, piece_b),
        "rect": rect,
        "final_rect": final_rect,
        "area": int(h * w),
    }


def _can_place(blocked: np.ndarray, patch: np.ndarray, top: int, left: int) -> bool:
    h, w = patch.shape
    mask = patch != BG
    return not np.any(blocked[top : top + h, left : left + w] & mask)


def _place_patch(grid: np.ndarray, blocked: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    mask = patch != BG
    sub = grid[top : top + h, left : left + w]
    sub[mask] = patch[mask]
    grid[top : top + h, left : left + w] = sub
    coords = np.argwhere(mask)
    gh, gw = grid.shape
    for r, c in coords:
        rr = top + int(r)
        cc = left + int(c)
        blocked[max(0, rr - 1) : min(gh, rr + 2), max(0, cc - 1) : min(gw, cc + 2)] = True


def _gen_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    pair_count = int(rng.integers(2, 5))
    max_dim = 6 if pair_count == 4 else 7
    color_choices = list(map(int, rng.choice(MAIN_COLORS, size=pair_count, replace=False)))

    specs: list[dict] = []
    used_areas: set[int] = set()
    for color in color_choices:
        spec = None
        for _ in range(80):
            cand = _make_pair_spec(rng, color, max_dim)
            if cand is None or cand["area"] in used_areas:
                continue
            spec = cand
            used_areas.add(int(cand["area"]))
            break
        if spec is None:
            return None
        specs.append(spec)

    records_for_output = []
    pieces = []
    for pair_id, spec in enumerate(specs):
        for patch in spec["pieces"]:
            pieces.append({"pair_id": pair_id, "patch": patch})

    min_side = 20 + max(0, pair_count - 2)
    grid_h = int(rng.integers(min_side, ARC_MAX + 1))
    grid_w = int(rng.integers(min_side, ARC_MAX + 1))
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)

    order = list(map(int, rng.permutation(len(pieces))))
    order.sort(key=lambda idx: -int(np.count_nonzero(pieces[idx]["patch"] != BG)))
    placements: dict[int, list[tuple[int, int]]] = {pair_id: [] for pair_id in range(pair_count)}

    for idx in order:
        patch = pieces[idx]["patch"]
        h, w = patch.shape
        placed = False
        for _ in range(240):
            top = int(rng.integers(0, grid_h - h + 1))
            left = int(rng.integers(0, grid_w - w + 1))
            if not _can_place(blocked, patch, top, left):
                continue
            _place_patch(grid, blocked, patch, top, left)
            placements[pieces[idx]["pair_id"]].append((top, left))
            placed = True
            break
        if not placed:
            return None

    for pair_id, spec in enumerate(specs):
        lefts = [left for _top, left in placements[pair_id]]
        tops = [top for top, _left in placements[pair_id]]
        records_for_output.append(
            {
                "arr": spec["final_rect"],
                "area": int(spec["area"]),
                "sort_left": min(lefts),
                "sort_top": min(tops),
            }
        )

    expected = _pack_records(records_for_output)
    if max(grid.shape) > ARC_MAX or max(expected.shape) > ARC_MAX:
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRY_LIMIT):
        case = _gen_case(rng)
        if case is None:
            continue
        grid, expected = case
        if np.array_equal(main(grid), expected):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
