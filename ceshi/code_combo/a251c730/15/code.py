# Combo variant of a251c730 (index 15)
# combo_spec: D2/0+D12/1
# Name: 三色精灵+最少颜色

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location

# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    - The global story stays "wallpaper + source frame + destination frame", but the source sprites are now
#      tri-color and the whole grid uses only 5-6 total colors.
#    - Old task trace:
#        two-color source sprite -> singleton anchor color -> same-color marker in destination -> stamp sprite.
#      Combo trace:
#        tri-color source sprite with counts (1,2,3/4) -> the unique 1-count color is still the anchor ->
#        destination marker matches that anchor color -> stamp the full three-color patch.
#    - Example 3x3 source type:
#          A B C
#          B B C
#          . B .
#      Here A is the singleton anchor, C is a two-pixel decoration, B is the majority body.
#      Under the combo, B may also be a wallpaper stripe color and C may also be the shared frame-border color,
#      so the solver must key off the visible frequency pattern inside each sprite rather than any global palette rule.
# b) Complexity calibration:
#    - Sprite color counts are restricted to exactly (1,2,3) or (1,2,4). This keeps the anchor visually unique
#      and the decoration visually distinct without introducing many subcases.
#    - There are still only 1-2 sprite types and 2-4 total source copies, so each type has repeated support.
#    - The low-color regime is made obvious but controlled: wallpaper uses 3 colors, both rectangles share one
#      border color, all sprite bodies share one wallpaper color, and all decorations reuse the shared border color.
#      With one sprite type this yields 5 total colors; with two sprite types the two distinct anchors force 6.
# c) Implicit assumption audit against the original `main()`:
#    - The old source-template logic effectively tolerated only two meaningful sprite roles. That is false here:
#      the stored template must preserve body + decoration + anchor.
#    - The old anchor selection used a color-id tie-break after choosing the minimum count. That is unsafe in ARC:
#      the combo requires a unique least-frequent color, not a hidden numeric ordering.
#    - The old rectangle pick grabbed the first source-like and destination-like rectangles. With only 5-6 colors
#      and shared borders, accidental candidates are more plausible, so the combo solver selects the best disjoint pair.
#    - The old code ignored color 0 in one place via `if k != 0` filtering. Under a compressed palette any ARC color,
#      including 0, may be structural, so the combo keeps all colors as ordinary visible symbols.
# d) Placement legality:
#    - Generation stays output-first. Full sprite patches are placed into a scratch destination interior; only the
#      anchor singleton survives in the input destination rectangle. This guarantees every later stamp is in bounds.
#    - Source sprites are placed with padding=1 from each other and from the frame interior, so 4-connected component
#      extraction remains stable even when anchor or decoration colors are reused elsewhere.
#    - Rectangles are placed non-overlapping on the 30x30 wallpaper exactly as in the base task.
# e) Visual observability:
#    - Humans can directly see two simultaneous changes across examples: sprites are tri-color, and the whole puzzle
#      suddenly uses only 5-6 colors with obvious cross-role reuse.
#    - The anchor remains inferable because it is the only singleton color inside each source sprite; no invisible
#      color-number ordering is needed.
# f) Visual cleanliness:
#    - The destination input still contains only 2-3 sparse singleton markers, so the target region stays readable.
#    - Small 2x3 / 3x3 sprites plus mandatory padding prevent merged source components and keep repeated evidence clear.
# g) Color inference and non-degeneracy:
#    - No hidden remapping is used: marker color equals anchor color exactly.
#    - The compressed palette is structural rather than cosmetic: every sample reuses one wallpaper color as the common
#      body color, and reuses the shared border color as the common decoration color.
#    - Quantitative combo metric after implementation:
#      every generated sample should have source-sprite count signatures in {(1,2,3), (1,2,4)}, total color count in
#      {5,6}, shared source/destination border color, and positive role reuse counts for body-in-wallpaper and decor=border.
# i) New solver assumptions that the generator must satisfy:
#    - Every source component that matters has exactly 3 colors with one unique singleton anchor.
#    - Every destination marker color is one of those anchor colors.
#    - Repeated copies of a type are identical, so any copy can act as the template for that anchor color.

# === CFG: Canvas & layout ===
GRID_SIZE = 30
WALLPAPER_COLOR_COUNT = 3
TOTAL_COLORS_TARGET = (5, 6)
GAP_RANGE = (1, 4)

# === CFG: Rectangle size bounds ===
SRC_H_RANGE = (9, 19)
SRC_W_RANGE = (8, 18)
DST_H_RANGE = (9, 19)
DST_W_RANGE = (8, 20)
DST_W_SIDE_MAX = 20

# === CFG: Sprite & placement ===
SPRITE_TYPE_CHOICES = (1, 2)
MARKER_COUNT_RANGE = (2, 3)
SOURCE_SPRITE_RANGE = (2, 4)
MARKER_PLACE_RETRIES = 25
SOURCE_SPRITE_PADDING = 1

# === CFG: Generation retries ===
MAX_GEN_ATTEMPTS = 450

SPRITE_SHAPES = [
    ("ABC", "BBC"),
    ("ABB", "CBC"),
    ("ABB", "BCC"),
    ("ABB", "BCC", "..B"),
    (".AB", "BBC", "..C"),
    ("AB.", "BBC", ".C."),
    ("A.B", "BBC", "C.."),
    ("A.B", "BBC", ".BC"),
    ("A..", "BBB", "CC."),
    ("A..", "BBB", "BCC"),
    ("..A", "BBB", ".CC"),
    ("..A", "BBB", "CCB"),
    (".AB", "BCB", "C.."),
    ("AB.", "BCB", "..C"),
    ("AB.", "BCB", ".BC"),
    ("A..", "BCB", ".BC"),
]


def _shape_info(shape_rows: tuple[str, ...]) -> tuple[tuple[int, int], Counter]:
    counts: Counter = Counter()
    anchor_rc: tuple[int, int] | None = None
    occ = np.zeros((len(shape_rows), len(shape_rows[0])), dtype=bool)
    for r, row in enumerate(shape_rows):
        for c, ch in enumerate(row):
            if ch == ".":
                continue
            occ[r, c] = True
            counts[ch] += 1
            if ch == "A":
                anchor_rc = (int(r), int(c))
    assert anchor_rc is not None
    assert counts["A"] == 1
    assert counts["C"] == 2
    assert counts["B"] in (3, 4)
    assert len(components_mask(occ, connectivity=4)) == 1
    return anchor_rc, counts


for _shape in SPRITE_SHAPES:
    _anchor_rc, _counts = _shape_info(_shape)
    assert len(_shape) in (2, 3)
    assert len(_shape[0]) == 3


def _render_sprite(shape_rows: tuple[str, ...], body: int, anchor: int, decor: int, bg: int) -> np.ndarray:
    """Render a tri-color sprite. A=anchor, B=body, C=decoration, .=bg."""
    char_map = {"A": anchor, "B": body, "C": decor}
    rows = [[char_map.get(ch, bg) for ch in row] for row in shape_rows]
    return np.array(rows, dtype=int)


def _make_wallpaper(direction: str, palette: list[int]) -> np.ndarray:
    period = len(palette)
    stripe = np.array(palette * ((GRID_SIZE + period - 1) // period), dtype=int)[:GRID_SIZE]
    if direction == "vertical":
        return np.tile(stripe, (GRID_SIZE, 1))
    return np.tile(stripe.reshape(-1, 1), (1, GRID_SIZE))


def _build_frame(h: int, w: int, border: int, bg: int) -> np.ndarray:
    rect = np.full((h, w), bg, dtype=int)
    rect[0, :] = rect[-1, :] = border
    rect[:, 0] = rect[:, -1] = border
    return rect


def _choose_counts(n_total: int, n_types: int) -> list[int]:
    counts = [1] * n_types
    for _ in range(n_total - n_types):
        counts[random.randrange(n_types)] += 1
    return counts


def _choose_dimensions(layout: str) -> tuple[int, int, int, int]:
    gap = random.randint(*GAP_RANGE)
    if layout == "side_by_side":
        src_w = random.randint(*SRC_W_RANGE)
        dst_w_max = min(DST_W_SIDE_MAX, GRID_SIZE - gap - src_w)
        if dst_w_max < DST_W_RANGE[0]:
            raise ValueError("no side-by-side width fit")
        dst_w = random.randint(DST_W_RANGE[0], dst_w_max)
        src_h = random.randint(*SRC_H_RANGE)
        dst_h = random.randint(*DST_H_RANGE)
    else:
        src_h = random.randint(*SRC_H_RANGE)
        dst_h_max = min(DST_H_RANGE[1], GRID_SIZE - gap - src_h)
        if dst_h_max < DST_H_RANGE[0]:
            raise ValueError("no stacked height fit")
        dst_h = random.randint(DST_H_RANGE[0], dst_h_max)
        src_w = random.randint(*SRC_W_RANGE)
        dst_w = random.randint(*DST_W_RANGE)
    return src_h, src_w, dst_h, dst_w


def _place_rectangles(
    layout: str, src_h: int, src_w: int, dst_h: int, dst_w: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    if layout == "side_by_side":
        gap = random.randint(1, GRID_SIZE - src_w - dst_w)
        slack = GRID_SIZE - src_w - dst_w - gap
        left_margin = random.randint(0, slack)
        src_r = random.randint(0, GRID_SIZE - src_h)
        dst_r = random.randint(0, GRID_SIZE - dst_h)
        if random.random() < 0.5:
            src_c, dst_c = left_margin, left_margin + src_w + gap
        else:
            dst_c, src_c = left_margin, left_margin + dst_w + gap
    else:
        gap = random.randint(1, GRID_SIZE - src_h - dst_h)
        slack = GRID_SIZE - src_h - dst_h - gap
        top_margin = random.randint(0, slack)
        src_c = random.randint(0, GRID_SIZE - src_w)
        dst_c = random.randint(0, GRID_SIZE - dst_w)
        if random.random() < 0.5:
            src_r, dst_r = top_margin, top_margin + src_h + gap
        else:
            dst_r, src_r = top_margin, top_margin + dst_h + gap
    return (src_r, src_c), (dst_r, dst_c)


def _classify_tricolor_sprite(colors: dict[int, int]) -> tuple[int, int, int] | None:
    if len(colors) != 3:
        return None
    ordered = sorted((int(cnt), int(col)) for col, cnt in colors.items())
    counts = [cnt for cnt, _col in ordered]
    if counts not in ([1, 2, 3], [1, 2, 4]):
        return None
    anchor_col = int(ordered[0][1])
    decor_col = int(ordered[1][1])
    body_col = int(ordered[2][1])
    return anchor_col, decor_col, body_col


def _interior_bg(g: np.ndarray, rect: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = map(int, rect)
    inner = g[r0 + 1 : r1, c0 + 1 : c1]
    return int(Counter(map(int, inner.flatten())).most_common(1)[0][0])


def _rect_components(g: np.ndarray, rect: tuple[int, int, int, int]) -> tuple[int, list[dict]]:
    r0, c0, r1, c1 = map(int, rect)
    inner = g[r0 + 1 : r1, c0 + 1 : c1]
    ibg = int(Counter(map(int, inner.flatten())).most_common(1)[0][0])
    comps: list[dict] = []

    for coords in components_mask(inner != ibg, connectivity=4):
        if coords.size == 0:
            continue
        rr0, cc0, rr1, cc1 = bbox_coords(coords)
        mask = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
        mask[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        patch = inner[rr0 : rr1 + 1, cc0 : cc1 + 1].copy()
        patch[~mask] = 0
        vals = [int(inner[int(r), int(c)]) for r, c in coords]
        colors = dict(Counter(vals))
        comps.append(
            {
                "size": int(coords.shape[0]),
                "colors": {int(k): int(v) for k, v in colors.items()},
                "bbox": (int(rr0), int(cc0), int(rr1), int(cc1)),
                "mask": mask,
                "patch": patch,
            }
        )

    comps.sort(key=lambda d: -int(d["size"]))
    return int(ibg), comps


def _find_rect_candidates(g: np.ndarray) -> list[dict]:
    H, W = map(int, g.shape)
    rects: list[dict] = []
    min_h = 6
    min_w = 6
    for r0 in range(H):
        for c0 in range(W):
            bc = int(g[r0, c0])
            for r1 in range(r0 + min_h - 1, H):
                for c1 in range(c0 + min_w - 1, W):
                    if int(g[r0, c1]) != bc or int(g[r1, c0]) != bc or int(g[r1, c1]) != bc:
                        continue
                    if not bool((g[r0, c0 : c1 + 1] == bc).all()):
                        continue
                    if not bool((g[r1, c0 : c1 + 1] == bc).all()):
                        continue
                    if not bool((g[r0 : r1 + 1, c0] == bc).all()):
                        continue
                    if not bool((g[r0 : r1 + 1, c1] == bc).all()):
                        continue
                    rect = (int(r0), int(c0), int(r1), int(c1))
                    ibg = _interior_bg(g, rect)
                    if ibg == bc:
                        continue
                    area = int((r1 - r0 + 1) * (c1 - c0 + 1))
                    bg, comps = _rect_components(g, rect)
                    if not comps:
                        continue
                    tri_like_count = sum(_classify_tricolor_sprite(comp["colors"]) is not None for comp in comps)
                    rects.append(
                        {
                            "area": area,
                            "rect": rect,
                            "bc": int(bc),
                            "bg": int(bg),
                            "comps": comps,
                            "max_sz": max(int(comp["size"]) for comp in comps),
                            "tri_like_count": int(tri_like_count),
                        }
                    )
    return rects


def _disjoint(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = map(int, a)
    br0, bc0, br1, bc1 = map(int, b)
    return ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0


def _pick_task_rectangles(g: np.ndarray) -> tuple[dict, dict]:
    candidates = _find_rect_candidates(g)
    assert candidates, "must find bordered rectangles"

    src_candidates = [
        cand
        for cand in candidates
        if int(cand["max_sz"]) > 1 and int(cand["tri_like_count"]) >= 1 and len(cand["comps"]) >= 2
    ]
    dst_candidates = [cand for cand in candidates if int(cand["max_sz"]) == 1 and len(cand["comps"]) >= 2]

    best_pair: tuple[dict, dict] | None = None
    best_score: tuple[int, int, int, int, int, int] | None = None
    for src in src_candidates:
        for dst in dst_candidates:
            if int(src["area"]) <= int(dst["area"]):
                continue
            if not _disjoint(src["rect"], dst["rect"]):
                continue
            score = (
                int(src["tri_like_count"] == len(src["comps"])),
                int(src["tri_like_count"]),
                int(src["area"]) + int(dst["area"]),
                int(src["area"]),
                len(src["comps"]),
                len(dst["comps"]),
            )
            if best_score is None or score > best_score:
                best_pair = (src, dst)
                best_score = score

    assert best_pair is not None, "must find disjoint source and destination rectangles"
    return best_pair


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the destination frame by matching low-palette tri-color source sprites to singleton markers.

    Input structure:
    - Two bordered rectangles lie on a striped wallpaper, but the whole grid uses only 5-6 colors.
    - The source rectangle contains 2-4 disconnected tri-color sprites; each sprite has one singleton anchor
      color, one two-pixel decoration color, and one majority body color.
    - The destination rectangle contains only singleton marker pixels whose colors equal source anchor colors.

    Rule:
    - Parse the source rectangle into sprite components and keep only components whose visible color counts are
      exactly (1,2,3) or (1,2,4), treating the unique singleton color as the anchor.
    - Match each destination singleton marker to the source sprite with the same anchor color.
    - Stamp the full three-color sprite into a cleared destination frame with its anchor pixel aligned to the marker.

    Key insight: Even though the palette is compressed and roles reuse colors across wallpaper, borders, and sprite
    interiors, each valid source sprite still reveals its lookup key through its own internal frequency pattern.
    """
    g = np.array(grid, dtype=int)
    src, dst = _pick_task_rectangles(g)

    dr, dc, dr1, dc1 = map(int, dst["rect"])
    dst_bc = int(dst["bc"])
    dst_bg = int(dst["bg"])

    templates: dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]] = {}
    for comp in src["comps"]:
        roles = _classify_tricolor_sprite(comp["colors"])
        if roles is None:
            continue
        anchor_col, _decor_col, _body_col = roles
        patch = comp["patch"]
        mask = comp["mask"]
        anchors = np.argwhere((patch == int(anchor_col)) & mask)
        if anchors.shape[0] != 1:
            continue
        ar, ac = map(int, anchors[0])
        if int(anchor_col) not in templates or int(mask.sum()) > int(templates[int(anchor_col)][1].sum()):
            templates[int(anchor_col)] = (patch.copy(), mask.copy(), (int(ar), int(ac)))

    assert templates, "no tri-color source templates found"

    out_h = int(dr1 - dr + 1)
    out_w = int(dc1 - dc + 1)
    out = np.full((out_h, out_w), int(dst_bg), dtype=int)
    out[0, :] = int(dst_bc)
    out[-1, :] = int(dst_bc)
    out[:, 0] = int(dst_bc)
    out[:, -1] = int(dst_bc)

    inner = g[dr + 1 : dr1, dc + 1 : dc1]
    for rr, cc in np.argwhere(inner != int(dst_bg)):
        mr = int(dr + 1 + int(rr))
        mc = int(dc + 1 + int(cc))
        mcol = int(g[mr, mc])
        if int(mcol) not in templates:
            continue
        patch, mask, (ar, ac) = templates[int(mcol)]
        top = int((mr - dr) - int(ar))
        left = int((mc - dc) - int(ac))
        paste(out, patch, top, left, mask=mask, overwrite=True, clip=True)

    return out


def _assign_palette(n_types: int) -> tuple[list[int], int, int, int, int, int, list[int], int]:
    """Allocate a 5-6 color palette with explicit cross-role reuse."""
    all_colors = list(range(10))
    wallpaper = random.sample(all_colors, WALLPAPER_COLOR_COUNT)
    src_bg, dst_bg, body = random.sample(wallpaper, WALLPAPER_COLOR_COUNT)

    remaining = [c for c in all_colors if c not in wallpaper]
    random.shuffle(remaining)

    border = int(remaining.pop())
    decor = int(border)

    if n_types == 1:
        anchors = [int(remaining.pop())]
        target_total = 5
    else:
        anchors = [int(remaining.pop()), int(remaining.pop())]
        target_total = 6

    return wallpaper, int(src_bg), int(dst_bg), int(body), int(border), int(decor), anchors, int(target_total)


def _place_markers(
    sprite_types: list[dict], marker_type_ids: list[int], dst_h: int, dst_w: int, dst_bg: int
) -> tuple[np.ndarray, list[tuple[int, int, int]]] | None:
    """Place full sprite footprints in a scratch destination interior; keep anchor coordinates."""
    inner = np.full((dst_h - 2, dst_w - 2), dst_bg, dtype=int)
    marker_points: list[tuple[int, int, int]] = []
    for type_id in marker_type_ids:
        sprite = sprite_types[type_id]
        patch = _render_sprite(sprite["shape_rows"], sprite["body"], sprite["anchor"], sprite["decor"], dst_bg)
        placed = False
        for _ in range(MARKER_PLACE_RETRIES):
            try:
                r0, c0 = random_free_location(inner, patch, transparent=dst_bg)
            except ValueError:
                return None
            mr = int(r0 + sprite["anchor_rc"][0])
            mc = int(c0 + sprite["anchor_rc"][1])
            if any(abs(mr - pr) + abs(mc - pc) <= 1 for pr, pc, _ in marker_points):
                continue
            paste(inner, patch, r0, c0, transparent=dst_bg)
            marker_points.append((mr, mc, type_id))
            placed = True
            break
        if not placed:
            return None
    return inner, marker_points


def _place_sources(
    sprite_types: list[dict], source_type_ids: list[int], src_h: int, src_w: int, src_bg: int
) -> np.ndarray | None:
    """Place source sprites with one-cell padding to keep components disconnected."""
    inner = np.full((src_h - 2, src_w - 2), src_bg, dtype=int)
    for type_id in source_type_ids:
        sprite = sprite_types[type_id]
        patch = _render_sprite(sprite["shape_rows"], sprite["body"], sprite["anchor"], sprite["decor"], src_bg)
        try:
            r0, c0 = random_free_location(
                inner,
                patch,
                transparent=src_bg,
                border=SOURCE_SPRITE_PADDING,
                padding=SOURCE_SPRITE_PADDING,
            )
        except ValueError:
            return None
        paste(inner, patch, r0, c0, transparent=src_bg)
    return inner


def _variant_checks(
    grid: np.ndarray,
    wallpaper: list[int],
    src_bg: int,
    dst_bg: int,
    border: int,
    sprite_types: list[dict],
    target_total: int,
) -> bool:
    colors = {int(x) for x in grid.flatten()}
    if len(colors) != int(target_total):
        return False
    if target_total not in TOTAL_COLORS_TARGET:
        return False
    if not all(int(sprite["body"]) in wallpaper for sprite in sprite_types):
        return False
    if len({int(sprite["body"]) for sprite in sprite_types}) != 1:
        return False
    if len({int(sprite["decor"]) for sprite in sprite_types}) != 1:
        return False
    if int(sprite_types[0]["decor"]) != int(border):
        return False
    if any(int(sprite["anchor"]) == int(sprite["body"]) for sprite in sprite_types):
        return False
    if any(int(sprite["anchor"]) == int(sprite["decor"]) for sprite in sprite_types if int(sprite["anchor"]) != int(border)):
        return False
    if len({int(sprite["anchor"]) for sprite in sprite_types}) != len(sprite_types):
        return False
    if int(src_bg) == int(dst_bg) or int(src_bg) == int(sprite_types[0]["body"]) or int(dst_bg) == int(sprite_types[0]["body"]):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_types = random.choice(list(SPRITE_TYPE_CHOICES))
    wallpaper, src_bg, dst_bg, body, border, decor, anchors, target_total = _assign_palette(n_types)

    sprite_specs = random.sample(SPRITE_SHAPES, n_types)
    sprite_types: list[dict] = []
    for shape_rows, anchor in zip(sprite_specs, anchors):
        anchor_rc, counts = _shape_info(shape_rows)
        sprite_types.append(
            {
                "shape_rows": shape_rows,
                "anchor_rc": anchor_rc,
                "body": int(body),
                "anchor": int(anchor),
                "decor": int(decor),
                "h": len(shape_rows),
                "w": len(shape_rows[0]),
                "area": int(sum(counts.values())),
            }
        )

    n_markers = random.randint(*MARKER_COUNT_RANGE)
    marker_type_ids = [i for i, ct in enumerate(_choose_counts(n_markers, n_types)) for _ in range(ct)]
    random.shuffle(marker_type_ids)

    n_source = random.randint(max(2, n_types), SOURCE_SPRITE_RANGE[1])
    source_type_ids = [i for i, ct in enumerate(_choose_counts(n_source, n_types)) for _ in range(ct)]
    random.shuffle(source_type_ids)

    layout = random.choice(["side_by_side", "stacked"])
    try:
        src_h, src_w, dst_h, dst_w = _choose_dimensions(layout)
    except ValueError:
        return None
    if src_h * src_w <= dst_h * dst_w:
        return None

    marker_result = _place_markers(sprite_types, marker_type_ids, dst_h, dst_w, dst_bg)
    if marker_result is None:
        return None
    out_inner, marker_points = marker_result

    output = _build_frame(dst_h, dst_w, border, dst_bg)
    paste(output[1:-1, 1:-1], out_inner, 0, 0)

    source_inner = _place_sources(sprite_types, source_type_ids, src_h, src_w, src_bg)
    if source_inner is None:
        return None

    source_rect = _build_frame(src_h, src_w, border, src_bg)
    paste(source_rect[1:-1, 1:-1], source_inner, 0, 0)

    dest_rect = _build_frame(dst_h, dst_w, border, dst_bg)
    for mr, mc, type_id in marker_points:
        dest_rect[1 + mr, 1 + mc] = int(sprite_types[type_id]["anchor"])

    try:
        src_pos, dst_pos = _place_rectangles(layout, src_h, src_w, dst_h, dst_w)
    except ValueError:
        return None

    grid = _make_wallpaper(random.choice(["vertical", "horizontal"]), wallpaper)
    paste(grid, source_rect, src_pos[0], src_pos[1])
    paste(grid, dest_rect, dst_pos[0], dst_pos[1])

    if not _variant_checks(grid, wallpaper, src_bg, dst_bg, border, sprite_types, target_total):
        return None

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a251c730 combo-15 input after {MAX_GEN_ATTEMPTS} attempts")


def fusion_metrics(grid: np.ndarray) -> dict[str, object]:
    """Quantify whether a sample really exhibits the combo behavior."""
    g = np.array(grid, dtype=int)
    src, dst = _pick_task_rectangles(g)

    mask = np.ones(g.shape, dtype=bool)
    for rect in (src["rect"], dst["rect"]):
        r0, c0, r1, c1 = map(int, rect)
        mask[r0 : r1 + 1, c0 : c1 + 1] = False
    wallpaper_colors = sorted({int(x) for x in g[mask]})

    tri_roles: list[tuple[int, int, int]] = []
    tri_counts: list[tuple[int, int, int]] = []
    for comp in src["comps"]:
        roles = _classify_tricolor_sprite(comp["colors"])
        if roles is None:
            continue
        tri_roles.append(roles)
        tri_counts.append(tuple(sorted(int(v) for v in comp["colors"].values())))

    body_in_wallpaper = sum(int(body in wallpaper_colors) for _anchor, _decor, body in tri_roles)
    decor_eq_border = sum(int(decor == int(src["bc"]) == int(dst["bc"])) for _anchor, decor, _body in tri_roles)
    anchor_eq_border = sum(int(anchor == int(src["bc"])) for anchor, _decor, _body in tri_roles)

    return {
        "total_colors": int(len({int(x) for x in g.flatten()})),
        "wallpaper_colors": wallpaper_colors,
        "src_border": int(src["bc"]),
        "dst_border": int(dst["bc"]),
        "shared_border": bool(int(src["bc"]) == int(dst["bc"])),
        "source_component_count": int(len(src["comps"])),
        "tricolor_component_count": int(len(tri_roles)),
        "tri_count_patterns": tri_counts,
        "body_in_wallpaper_count": int(body_in_wallpaper),
        "decor_eq_border_count": int(decor_eq_border),
        "anchor_eq_border_count": int(anchor_eq_border),
    }
