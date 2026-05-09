# Combo variant of 5dbc8537 (index 21)
# combo_spec: D2/0+D10/1
# Name: 不规则精灵+不规则洞区域

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

# Design analysis for the fused variant.
# a) Concrete behavior shift:
#    - Original task: a 3x4 outside sprite implied a fully-filled 3x4 hole rectangle.
#    - Fused task: a 4x4 outside sprite may be an L/T polyomino whose bbox still is 4x4,
#      but several cells inside that bbox stay background. The matching hole inside the
#      template is now the exact same irregular mask, not the whole bbox.
#    - Consequence: solving can no longer use bbox area or rectangle-cover logic. It must
#      preserve each sprite's occupancy mask and cover only those mask cells.
# b) Complexity calibration:
#    - Shapes are restricted to a visible small family (L and T polyominoes) so the rule
#      is legible from 2-4 examples.
#    - Each sample uses 5-7 pieces, with both L and T present and every piece mask unique
#      inside the sample. This prevents color-only ambiguity while keeping the puzzle richer
#      than the original rectangle-only version.
# c) Original implicit assumptions that break here:
#    - `sum(h*w) == hole_area` is invalid once sprite bboxes contain background slack;
#      we replace it with `sum(mask.sum()) == hole_area`.
#    - A candidate placement is no longer "all cells in the bbox are hole cells"; instead
#      the hole window must equal the sprite mask exactly.
#    - Occupancy during search must mark only mask cells, not the entire bbox.
# d) Placement legality:
#    - Generator keeps every hole piece and every outside sprite in-bounds.
#    - Hole pieces never overlap in occupied cells and never overlap in bbox footprint, so
#      each piece has a clean local window inside the template.
#    - Outside sprites are scattered using bbox-sized spacing, avoiding accidental touching.
# e) Visual observability / cleanliness:
#    - Every sprite has positive bbox slack (`bbox_area > mask_area`), so the irregularity
#      is visible in both the outside sprites and the inside holes.
#    - The hole union is still a single 4-connected corridor-like region, but its subparts
#      are visibly non-rectangular.
#    - Candidate uniqueness is enforced by the generator, so the examples do not rely on
#      invisible tie-breaks such as color-id ordering.

ROOT = Path(__file__).resolve()
for parent in ROOT.parents:
    solvers_dir = parent / "solvers"
    if solvers_dir.is_dir():
        sys.path.append(str(solvers_dir))
        break

from arc_utils import bbox_coords, colors_present, components_mask, crop_bbox, find_components, get_bg, neighbors4, paste, random_free_location


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble a strip template whose missing zones are irregular polyomino cavities.

    Input structure:
    - The most common color is the background, while one dominant non-background color forms a tall or wide template strip.
    - Inside that strip, the missing cells are L/T-shaped hole fragments; outside the strip, separated sprites carry the same irregular silhouettes and may include background cells inside their bboxes.

    Rule:
    - Use the largest non-background connected component color to locate the template bbox.
    - Extract every outside sprite as a colored patch plus its exact occupancy mask.
    - Find the unique hole window whose occupied cells match each sprite mask exactly, paste the sprite onto that mask, and return the filled template crop.

    Key insight:
    - Bounding-box size no longer solves the puzzle: two pieces can share similar boxes while differing by which cells inside the box are occupied, so matching must use the exact irregular mask rather than a full rectangle.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    base_candidates: list[tuple[int, int]] = []
    for col in colors_present(g, exclude={bg}):
        max_area = 0
        for coords in components_mask(g == int(col), connectivity=8):
            max_area = max(max_area, int(coords.shape[0]))
        base_candidates.append((int(max_area), int(col)))
    assert base_candidates
    base_candidates.sort(reverse=True)
    if len(base_candidates) > 1:
        assert base_candidates[0][0] > base_candidates[1][0]
    base_col = int(base_candidates[0][1])

    base_coords = np.argwhere(g == int(base_col))
    assert base_coords.size > 0
    r0, c0, r1, c1 = bbox_coords(base_coords)
    dst = crop_bbox(g, (int(r0), int(c0), int(r1), int(c1)))
    hole_mask = dst != int(base_col)
    dh, dw = map(int, dst.shape)

    masked = g.copy()
    masked[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(bg)
    sprite_comps = find_components(masked, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    sprites: list[dict] = []
    for comp in sprite_comps:
        pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (pr0, pc0, pr1, pc1))
        ph, pw = map(int, patch.shape)
        mask = np.zeros((ph, pw), dtype=bool)
        coords = comp["coords"]
        mask[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        sprites.append({"patch": patch, "mask": mask, "h": ph, "w": pw, "area": int(mask.sum())})

    assert sprites
    assert sum(int(sp["area"]) for sp in sprites) == int(hole_mask.sum())

    candidates: list[list[tuple[int, int]]] = []
    for sp in sprites:
        mask = sp["mask"]
        h, w = int(sp["h"]), int(sp["w"])
        pos: list[tuple[int, int]] = []
        if h <= dh and w <= dw:
            for rr in range(dh - h + 1):
                for cc in range(dw - w + 1):
                    if np.array_equal(hole_mask[rr : rr + h, cc : cc + w], mask):
                        pos.append((int(rr), int(cc)))
        candidates.append(pos)

    assert all(candidates)
    order = sorted(range(len(sprites)), key=lambda i: (len(candidates[i]), -int(sprites[i]["area"])))
    used = np.zeros_like(hole_mask, dtype=bool)
    placement: list[tuple[int, int] | None] = [None] * len(sprites)

    def dfs(k: int) -> bool:
        if k == len(order):
            return True
        idx = order[k]
        sp = sprites[idx]
        mask = sp["mask"]
        h, w = int(sp["h"]), int(sp["w"])
        for rr, cc in candidates[idx]:
            region = used[rr : rr + h, cc : cc + w]
            if bool((region & mask).any()):
                continue
            region[mask] = True
            placement[idx] = (int(rr), int(cc))
            if dfs(k + 1):
                return True
            placement[idx] = None
            region[mask] = False
        return False

    assert dfs(0)
    assert int(used.sum()) == int(hole_mask.sum())

    out = np.full((dh, dw), int(base_col), dtype=int)
    for sp, pos in zip(sprites, placement):
        assert pos is not None
        rr, cc = pos
        paste(out, sp["patch"], rr, cc, mask=sp["mask"], overwrite=True)
    return out


GRID_MIN, GRID_MAX = 15, 25
THICK_MIN, THICK_MAX = 8, 10
OUTER_GAP = 8
SPRITE_COUNT_MIN, SPRITE_COUNT_MAX = 4, 6
HOLE_FRAC_LO, HOLE_FRAC_HI = 0.31, 0.45
SPAN_FRAC = 0.62
MAX_RETRIES = 600


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_key(mask: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    return int(mask.shape[0]), int(mask.shape[1]), tuple(int(v) for v in mask.astype(int).ravel())


def _make_l_mask(h: int, w: int, thickness: int) -> np.ndarray | None:
    if min(h, w) <= thickness:
        return None
    mask = np.zeros((h, w), dtype=bool)
    mask[:, :thickness] = True
    mask[h - thickness :, :] = True
    if mask.all():
        return None
    return mask


def _make_t_mask(h: int, w: int, bar: int, stem: int) -> np.ndarray | None:
    if h <= bar or w <= stem:
        return None
    mask = np.zeros((h, w), dtype=bool)
    mask[:bar, :] = True
    left = (w - stem) // 2
    mask[:, left : left + stem] = True
    if mask.all():
        return None
    return mask


def _build_shape_library() -> list[dict]:
    out: list[dict] = []
    seen: set[tuple[int, int, tuple[int, ...]]] = set()
    for h in range(3, 6):
        for w in range(3, 6):
            for thickness in (1, 2):
                mask = _make_l_mask(h, w, thickness)
                if mask is None:
                    continue
                for rot in range(4):
                    mm = _trim_mask(np.rot90(mask, rot))
                    if int(mm.sum()) < 5:
                        continue
                    key = _mask_key(mm)
                    if key not in seen:
                        seen.add(key)
                        out.append({"kind": "L", "mask": mm, "area": int(mm.sum())})
            for bar in (1, 2):
                for stem in (1, 2):
                    mask = _make_t_mask(h, w, bar, stem)
                    if mask is None:
                        continue
                    for rot in range(4):
                        mm = _trim_mask(np.rot90(mask, rot))
                        if int(mm.sum()) < 5:
                            continue
                        key = _mask_key(mm)
                        if key not in seen:
                            seen.add(key)
                            out.append({"kind": "T", "mask": mm, "area": int(mm.sum())})
    return out


SHAPE_LIBRARY = _build_shape_library()


def _sample_shape_entries(long: int, thick: int) -> list[dict] | None:
    usable = [
        entry
        for entry in SHAPE_LIBRARY
        if entry["mask"].shape[0] <= long - 2 and entry["mask"].shape[1] <= thick - 2
    ]
    if len(usable) < SPRITE_COUNT_MIN:
        return None

    area_total = long * thick
    for _ in range(250):
        count = random.randint(SPRITE_COUNT_MIN, min(SPRITE_COUNT_MAX, len(usable)))
        picked = random.sample(usable, count)
        kinds = [entry["kind"] for entry in picked]
        if kinds.count("L") < 2 or kinds.count("T") < 2:
            continue
        total_h = sum(int(entry["mask"].shape[0]) for entry in picked)
        if total_h > long or total_h < int(round(SPAN_FRAC * long)):
            continue
        frac = sum(int(entry["area"]) for entry in picked) / area_total
        if HOLE_FRAC_LO <= frac <= HOLE_FRAC_HI:
            return [{"kind": entry["kind"], "mask": entry["mask"].copy(), "area": int(entry["area"])} for entry in picked]
    return None


def _touches_existing(occupied: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> bool:
    H, W = occupied.shape
    for rr, cc in np.argwhere(mask):
        r = int(r0 + rr)
        c = int(c0 + cc)
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < H and 0 <= nc < W and bool(occupied[nr, nc]):
                return True
    return False


def _place_chain(entries: list[dict], long: int, thick: int) -> list[dict] | None:
    for _ in range(24):
        order = list(entries)
        random.shuffle(order)

        total_h = sum(int(entry["mask"].shape[0]) for entry in order)
        if total_h > long:
            continue

        start = random.randint(0, long - total_h)
        current_r = start
        pieces: list[dict] = []
        prev_cols_abs: set[int] | None = None
        failed = False

        for idx, entry in enumerate(order):
            mask = entry["mask"]
            h, w = map(int, mask.shape)
            c_hi = thick - w - 1
            if c_hi < 1:
                failed = True
                break

            if idx == 0:
                choices = list(range(1, c_hi + 1))
            else:
                top_cols = [int(x) for x in np.where(mask[0])[0]]
                assert prev_cols_abs is not None
                choices = []
                for c in range(1, c_hi + 1):
                    cur_cols_abs = {int(c + x) for x in top_cols}
                    if prev_cols_abs & cur_cols_abs:
                        choices.append(int(c))
            if not choices:
                failed = True
                break

            center = thick / 2.0
            choices.sort(key=lambda c: (abs((c + w / 2.0) - center), random.random()))
            c = random.choice(choices[: min(8, len(choices))])
            pieces.append({"kind": entry["kind"], "mask": mask.copy(), "r": int(current_r), "c": int(c)})
            bottom_cols = [int(x) for x in np.where(mask[-1])[0]]
            prev_cols_abs = {int(c + x) for x in bottom_cols}
            current_r += h

        if failed:
            continue

        occupied = np.zeros((long, thick), dtype=bool)
        for piece in pieces:
            r, c = int(piece["r"]), int(piece["c"])
            h, w = map(int, piece["mask"].shape)
            if bool((occupied[r : r + h, c : c + w] & piece["mask"]).any()):
                failed = True
                break
            occupied[r : r + h, c : c + w] |= piece["mask"]
        if failed:
            continue

        comps = components_mask(occupied, connectivity=4)
        if len(comps) != 1:
            continue
        rows = np.argwhere(occupied)[:, 0]
        if int(rows.max() - rows.min() + 1) < int(round(SPAN_FRAC * long)):
            continue
        return pieces
    return None


def _make_patch(mask: np.ndarray, cols: list[int], bg: int) -> np.ndarray:
    out = np.full(mask.shape, int(bg), dtype=int)
    out[mask] = int(cols[0])
    if len(cols) == 1:
        return out

    rr, cc = np.indices(mask.shape)
    pattern = random.choice(["checker", "row-split", "col-split", "dots"])
    if pattern == "checker":
        pick = mask & (((rr + cc + random.randint(0, 1)) % 2) == 0)
    elif pattern == "row-split":
        cut = random.randint(1, max(1, mask.shape[0] - 1))
        pick = mask & (rr >= cut)
    elif pattern == "col-split":
        cut = random.randint(1, max(1, mask.shape[1] - 1))
        pick = mask & (cc >= cut)
    else:
        phase = random.randint(0, 2)
        pick = mask & (((2 * rr + cc + phase) % 3) == 0)

    count = int(mask.sum())
    picked = int(pick.sum())
    if picked == 0 or picked == count:
        coords = np.argwhere(mask)
        choose = coords[random.randrange(len(coords))]
        pick = np.zeros_like(mask, dtype=bool)
        pick[int(choose[0]), int(choose[1])] = True
    out[pick] = int(cols[1])
    return out


def _colorize_pieces(pieces: list[dict], bg: int, base: int) -> list[dict] | None:
    palette = [c for c in range(10) if c not in (bg, base)]
    if len(palette) < 4:
        return None
    active = random.sample(palette, random.randint(4, min(7, len(palette))))

    out: list[dict] = []
    for piece in pieces:
        primary = random.choice(active)
        cols = [primary]
        if random.random() < 0.4:
            others = [c for c in active if c != primary]
            if others:
                cols.append(random.choice(others))
        patch = _make_patch(piece["mask"], cols, bg)
        item = dict(piece)
        item["patch"] = patch
        item["area"] = int(piece["mask"].sum())
        out.append(item)
    return out


def _orient_pieces(pieces: list[dict], horizontal: bool) -> list[dict]:
    if not horizontal:
        return [{k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in piece.items()} for piece in pieces]

    out: list[dict] = []
    for piece in pieces:
        item = dict(piece)
        item["r"] = int(piece["c"])
        item["c"] = int(piece["r"])
        item["mask"] = piece["mask"].T.copy()
        item["patch"] = piece["patch"].T.copy()
        out.append(item)
    return out


def _render_output(dh: int, dw: int, base: int, pieces: list[dict]) -> np.ndarray:
    out = np.full((dh, dw), int(base), dtype=int)
    for piece in pieces:
        paste(out, piece["patch"], int(piece["r"]), int(piece["c"]), mask=piece["mask"], overwrite=True)
    return out


def _hole_mask(dh: int, dw: int, pieces: list[dict]) -> np.ndarray:
    mask = np.zeros((dh, dw), dtype=bool)
    for piece in pieces:
        r, c = int(piece["r"]), int(piece["c"])
        h, w = map(int, piece["mask"].shape)
        mask[r : r + h, c : c + w] |= piece["mask"]
    return mask


def _validate_piece_windows(dh: int, dw: int, pieces: list[dict]) -> bool:
    holes = _hole_mask(dh, dw, pieces)
    if len(components_mask(holes, connectivity=4)) != 1:
        return False
    seen: set[tuple[int, int, tuple[int, ...]]] = set()
    for piece in pieces:
        r, c = int(piece["r"]), int(piece["c"])
        h, w = map(int, piece["mask"].shape)
        key = _mask_key(piece["mask"])
        if key in seen:
            return False
        seen.add(key)
        if not np.array_equal(holes[r : r + h, c : c + w], piece["mask"]):
            return False
        if int(piece["mask"].size - piece["mask"].sum()) <= 0:
            return False
        candidate_count = 0
        for rr in range(dh - h + 1):
            for cc in range(dw - w + 1):
                if np.array_equal(holes[rr : rr + h, cc : cc + w], piece["mask"]):
                    candidate_count += 1
        if candidate_count != 1:
            return False
    return True


def _scatter_sprites(
    H: int,
    W: int,
    bg: int,
    base: int,
    tr0: int,
    tc0: int,
    dh: int,
    dw: int,
    pieces: list[dict],
) -> np.ndarray:
    inp = np.full((H, W), int(bg), dtype=int)
    inp[tr0 : tr0 + dh, tc0 : tc0 + dw] = int(base)
    for piece in pieces:
        r, c = int(piece["r"]), int(piece["c"])
        h, w = map(int, piece["mask"].shape)
        patch = piece["patch"]
        window = inp[tr0 + r : tr0 + r + h, tc0 + c : tc0 + c + w]
        window[piece["mask"]] = int(bg)

    bbox_canvas = np.full((H, W), int(bg), dtype=int)
    bbox_canvas[tr0 : tr0 + dh, tc0 : tc0 + dw] = int(base)

    order = list(range(len(pieces)))
    random.shuffle(order)
    order.sort(key=lambda i: pieces[i]["patch"].size, reverse=True)
    for idx in order:
        patch = pieces[idx]["patch"]
        box = np.full(patch.shape, int(base), dtype=int)
        r, c = random_free_location(bbox_canvas, box, transparent=bg, padding=1, padding_connectivity=8)
        paste(inp, patch, r, c, overwrite=True)
        paste(bbox_canvas, box, r, c, overwrite=True)
    return inp


def _fusion_stats(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    base_candidates: list[tuple[int, int]] = []
    for col in colors_present(g, exclude={bg}):
        max_area = 0
        for coords in components_mask(g == int(col), connectivity=8):
            max_area = max(max_area, int(coords.shape[0]))
        base_candidates.append((int(max_area), int(col)))
    base_candidates.sort(reverse=True)
    base = int(base_candidates[0][1])

    base_coords = np.argwhere(g == int(base))
    r0, c0, r1, c1 = bbox_coords(base_coords)
    masked = g.copy()
    masked[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    comps = find_components(masked, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    slack = 0
    irregular = 0
    for comp in comps:
        pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
        h = pr1 - pr0 + 1
        w = pc1 - pc0 + 1
        area = int(comp["coords"].shape[0])
        slack += h * w - area
        if h * w > area:
            irregular += 1

    dst = crop_bbox(g, (int(r0), int(c0), int(r1), int(c1)))
    hole_area = int((dst != int(base)).sum())
    return {"sprite_count": len(comps), "irregular_count": irregular, "bbox_slack": slack, "hole_area": hole_area}


def _gen_attempt() -> np.ndarray | None:
    vertical = random.random() < 0.5
    thick = random.randint(THICK_MIN, THICK_MAX)

    if vertical:
        H = random.randint(GRID_MIN, GRID_MAX)
        W = random.randint(max(GRID_MIN, thick + OUTER_GAP), GRID_MAX)
        long = H
        dh, dw = H, thick
        tr0, tc0 = 0, 0 if random.random() < 0.5 else W - thick
        horizontal = False
    else:
        W = random.randint(GRID_MIN, GRID_MAX)
        H = random.randint(max(GRID_MIN, thick + OUTER_GAP), GRID_MAX)
        long = W
        dh, dw = thick, W
        tr0, tc0 = 0 if random.random() < 0.5 else H - thick, 0
        horizontal = True

    entries = _sample_shape_entries(long, thick)
    if entries is None:
        return None
    placed = _place_chain(entries, long, thick)
    if placed is None:
        return None

    bg, base = random.sample(range(10), 2)
    colored = _colorize_pieces(placed, bg, base)
    if colored is None:
        return None
    pieces = _orient_pieces(colored, horizontal=horizontal)

    if not _validate_piece_windows(dh, dw, pieces):
        return None

    out = _render_output(dh, dw, base, pieces)
    try:
        inp = _scatter_sprites(H, W, bg, base, tr0, tc0, dh, dw, pieces)
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None

    stats = _fusion_stats(inp)
    if stats["sprite_count"] < SPRITE_COUNT_MIN:
        return None
    if stats["irregular_count"] != stats["sprite_count"]:
        return None
    if stats["bbox_slack"] <= 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid irregular 5dbc8537 input")
