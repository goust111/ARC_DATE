# Combo variant of edb79dae (index 18)
# combo_spec: D1/3+D14/2
# Name: 连接器匹配+锚点对齐

import itertools
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# ---------------------------------------------------------------------------
# Design analysis for combo D1/3 + D14/2
# ---------------------------------------------------------------------------
#
# a) Concrete behavior and trace
# - Original task:
#   * inside the frame, each placement site was a solid NxN source-colored block
#   * outside the frame, one same-color sprite existed for each source color
#   * placement used bbox top-left alignment, then a legend recolored the result
# - This combo changes both the selection cue and the placement cue:
#   * inside each frame slot, only one source-colored target pixel remains
#   * a small connector glyph sits two rows above that slot
#   * outside each sprite template, the same connector glyph sits above the sprite
#   * each sprite body contains exactly one gray anchor pixel
# - Small trace:
#   * suppose a source color 3 target lies at offset (2,1) inside its slot
#   * above that slot is an orange T connector
#   * outside the frame, the sprite with orange T also has a gray anchor at (2,1)
#   * the solver matches orange+T to recover the sprite shape, then places it so
#     that the sprite's gray anchor lands exactly on the source-color target
#   * after stamping, the whole sprite footprint, including the former anchor
#     cell, is recolored using the legend for source color 3
# - Difference from the original:
#   * source color no longer identifies the external sprite body
#   * bbox top-left alignment is gone; the target pixel and the sprite anchor now
#     determine the translation
#
# b) Complexity calibration
# - Connector colors and connector shapes are both intentionally reused, so
#   matching by color alone or shape alone is insufficient.
# - Anchor offsets vary across sprite types whenever possible, and at least one
#   sprite uses a non-central anchor. This prevents the task from collapsing back
#   to a fixed alignment heuristic.
# - The connector vocabulary stays tiny (three glyphs), which keeps the rule
#   readable in a few ARC examples instead of introducing an uncontrolled glyph
#   space.
#
# c) Implicit-assumption audit
# - The original solver assumed every non-bg interior component was a solid
#   placeholder block. That is invalid here: interior now contains one-pixel
#   targets plus separate connector glyphs.
# - The original solver assumed sprite color matched placeholder color. That is
#   invalid here: sprite body colors are sampled from a pool disjoint from the
#   source colors.
# - The original solver aligned sprite bbox top-left to placeholder bbox top-left.
#   That is invalid here: the gray anchor offset inside the sprite determines the
#   translation.
# - A new combined assumption appears: each interior connector belongs to exactly
#   one target pixel via one sprite anchor offset. The generator guarantees this
#   by reserving one slot per placement, keeping row gaps at 2, and assigning one
#   connector plus one target to every filled slot.
#
# d) Placement legality
# - Inside the frame, body slots remain NxN with 1-column gaps and 2-row gaps.
#   The 2-row gap is what leaves room for the connector band above each slot.
# - Each external sprite patch has size (N+2) x N: a 2-row connector band on top
#   and an NxN sprite body below. The connector is attached at a fixed edge site
#   of the template, so its position relative to the target varies with the
#   anchor offset in both axes.
# - Outside patches are placed with background padding, so frame, legends,
#   connectors, and sprites stay disjoint.
# - The target pixel is placed at the anchor offset inside the slot, so when the
#   solver aligns anchor to target the entire sprite stays inside that reserved
#   slot and cannot overlap neighbors.
#
# e) Visual observability
# - The combo is visible in the raw input:
#   * inside the frame, targets are sparse one-pixel markers instead of solid
#     blocks
#   * connectors appear above those targets and above the external templates
#   * each external body contains one conspicuous gray anchor
# - Because body colors are not source colors, the old color-match story is
#   visibly wrong. Because targets are one-pixel anchors, the old bbox-placement
#   story is also visibly wrong.
#
# f) Visual cleanliness
# - Connectors occupy a dedicated two-row band above each slot instead of being
#   mixed into the body footprint.
# - Targets stay as single pixels inside otherwise empty slots, so the input does
#   not become cluttered even after adding both cues.
# - Output remains clean: only recolored sprite bodies are shown inside the frame;
#   connector cues disappear after solving.
#
# g) Color-change inferability
# - Recoloring still uses the original fixed 3x4 legend motif, so the only new
#   concepts are connector-based template selection and anchor-based translation.
# - Source colors are limited to a small set and repeat across multiple slots,
#   giving the legend enough support for humans to infer source->target mapping.
#
# h) New solver assumptions and generator guarantees
# - Solver assumption: every interior source-colored component is a single pixel.
#   Generator guarantee: only targets use source colors inside the frame.
# - Solver assumption: every interior non-source component is a small connector.
#   Generator guarantee: connector colors come from a pool disjoint from source
#   colors and each connector glyph has area 3 or 4.
# - Solver assumption: every external sprite body bbox contains exactly one gray
#   anchor and a connector band immediately above it. Generator guarantee: sprite
#   masks span every row and column after removing the anchor, so bbox recovery is
#   stable even though the anchor has a different color.


CONNECTOR_SHAPES: dict[str, tuple[tuple[int, int], ...]] = {
    "L": ((0, 0), (1, 0), (1, 1)),
    "J": ((0, 1), (1, 0), (1, 1)),
    "T": ((0, 1), (1, 0), (1, 1), (1, 2)),
}


def _normalize_coords(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _find_frame(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    """
    Find a rectangular border component:
      - all bbox-border cells are this color,
      - bbox-interior is not entirely this color.
    Return (frame_color, (r0,c0,r1,c1)).
    """
    candidates = []
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    for col in colors:
        for pts in components_mask(g == col, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            border = np.zeros(sub.shape, dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.all(sub[border] == col):
                continue
            if np.all(sub[~border] == col):
                continue
            candidates.append((col, (r0, c0, r1, c1)))
    assert len(candidates) == 1, f"expected exactly one rectangular frame component, found {len(candidates)}"
    frame_col, bbox = candidates[0]
    return int(frame_col), bbox


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Place recolored sprites into a framed region by combining connector selection with anchor alignment.

    Input structure:
    - One rectangular frame on a background, with single-pixel source-colored targets inside the frame.
    - Each target slot also has a small connector glyph in the two rows above it.
    - Outside the frame: sprite templates whose bodies contain one gray anchor pixel and whose own
      connector glyph sits directly above the body, plus the usual small two-color legends.

    Rule:
    - Find the frame, collect the interior target pixels, and read the interior connector glyphs.
    - Recover each external sprite by reading its connector key and its gray anchor offset.
    - For every target, use the matching connector-marked external template, then keep the translation
      where that template's gray anchor lands on the target pixel and its shifted edge connector lands
      exactly on the interior connector glyph.
    - Recolor the placed footprint using the legend keyed by the target's source color.

    Key insight: The connector chooses the template, but its horizontal position is tied to the
    template's anchor column rather than to the body's left edge. That means the lone target pixel is
    still necessary: only anchor alignment reveals the full placement offset that makes the connector
    coincide with the interior cue.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    frame_col, (fr0, fc0, fr1, fc1) = _find_frame(g, bg)
    crop = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    H, W = crop.shape
    assert H >= 3 and W >= 3, f"frame too small for interior: {(H, W)}"

    inner_mask = np.zeros((H, W), dtype=bool)
    inner_mask[1:-1, 1:-1] = True

    target_pts: list[tuple[int, int, int]] = []
    inner_connectors = []
    for col in [int(c) for c in np.unique(crop) if int(c) not in (bg, frame_col)]:
        for pts in components_mask((crop == col) & inner_mask, connectivity=4):
            pts_tuples = [(int(r), int(c)) for r, c in pts]
            r0, c0, r1, c1 = bbox_coords(pts)
            local = [(r - r0, c - c0) for r, c in pts_tuples]
            if len(pts_tuples) == 1:
                target_pts.append((int(col), int(pts_tuples[0][0]), int(pts_tuples[0][1])))
            else:
                inner_connectors.append(
                    {
                        "col": int(col),
                        "bbox": (r0, c0, r1, c1),
                        "pts": frozenset(pts_tuples),
                        "key": (int(col), _normalize_coords(local)),
                    }
                )

    placeholder_colors = {col for col, _, _ in target_pts}
    assert target_pts, "no target pixels found inside frame"
    assert inner_connectors, "no interior connector glyphs found"
    assert len(inner_connectors) == len(target_pts), "expected one connector per target slot"

    outside = np.ones(g.shape, dtype=bool)
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False

    comps = []
    cid = 0
    for col in [int(c) for c in np.unique(g) if int(c) not in (bg, frame_col)]:
        for pts in components_mask((g == col) & outside, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            pts_tuples = [(int(r), int(c)) for r, c in pts]
            comps.append(
                {
                    "id": cid,
                    "col": int(col),
                    "pts": pts_tuples,
                    "bbox": (r0, c0, r1, c1),
                    "area": len(pts_tuples),
                }
            )
            cid += 1

    color_map: dict[int, int] = {}
    legend_ids: set[int] = set()
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            ci = comps[i]
            cj = comps[j]
            if ci["col"] == cj["col"]:
                continue
            if not _bbox_overlap(ci["bbox"], cj["bbox"]):
                continue
            r0 = min(ci["bbox"][0], cj["bbox"][0])
            c0 = min(ci["bbox"][1], cj["bbox"][1])
            r1 = max(ci["bbox"][2], cj["bbox"][2])
            c1 = max(ci["bbox"][3], cj["bbox"][3])
            if (r1 - r0 + 1) > 8 or (c1 - c0 + 1) > 8:
                continue
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            sub_colors = set(map(int, np.unique(sub))) - {frame_col}
            if bg in sub_colors or len(sub_colors) != 2:
                continue
            if (ci["col"] in placeholder_colors) ^ (cj["col"] in placeholder_colors):
                src = ci["col"] if ci["col"] in placeholder_colors else cj["col"]
                tgt = cj["col"] if src == ci["col"] else ci["col"]
            else:
                assert ci["area"] != cj["area"], f"ambiguous equal-area legend pair: {ci['col']} and {cj['col']}"
                if ci["area"] > cj["area"]:
                    src, tgt = ci["col"], cj["col"]
                else:
                    src, tgt = cj["col"], ci["col"]
            assert src in placeholder_colors, f"legend source color {src} is not a target color"
            assert src not in color_map, f"duplicate legend mapping for target color {src}"
            color_map[src] = int(tgt)
            legend_ids.add(ci["id"])
            legend_ids.add(cj["id"])
    assert set(color_map) == placeholder_colors, (
        f"legend mappings {sorted(color_map)} do not match targets {sorted(placeholder_colors)}"
    )

    sprite_by_key: dict[tuple[int, tuple[tuple[int, int], ...]], list[dict[str, object]]] = {}
    body_sizes: set[tuple[int, int]] = set()
    for comp in comps:
        if comp["id"] in legend_ids:
            continue
        sr0, sc0, sr1, sc1 = comp["bbox"]
        bh = sr1 - sr0 + 1
        bw = sc1 - sc0 + 1
        if bh < 3 or bw < 3:
            continue
        sub = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
        sub_colors = set(map(int, np.unique(sub)))
        if not (sub_colors - {bg, frame_col} == {int(comp["col"])}):
            continue
        anchor_pts = np.argwhere(sub == frame_col)
        if anchor_pts.shape[0] != 1:
            continue
        assert sr0 >= 2, "sprite body is too close to top edge for connector band"
        band = g[sr0 - 2 : sr0, sc0 : sc1 + 1]
        band_pts = [(int(r), int(c)) for r, c in np.argwhere((band != bg) & (band != frame_col))]
        assert band_pts, f"missing connector above sprite body bbox {comp['bbox']}"
        band_colors = {int(band[r, c]) for r, c in band_pts}
        assert len(band_colors) == 1, f"expected one connector color above sprite body, found {band_colors}"
        conn_shape = _normalize_coords(band_pts)
        key = (int(next(iter(band_colors))), conn_shape)
        coords = tuple(sorted((int(r), int(c)) for r, c in np.argwhere(sub != bg)))
        assert coords, "sprite body cannot be empty"
        sprite_by_key.setdefault(key, []).append(
            {
                "coords": coords,
                "anchor": tuple(map(int, anchor_pts[0])),
                "conn_shape": conn_shape,
                "shape": (bh, bw),
            }
        )
        body_sizes.add((bh, bw))
    assert len(body_sizes) == 1, f"expected one common sprite bbox size, found {sorted(body_sizes)}"
    assert all(conn["key"] in sprite_by_key for conn in inner_connectors), "interior connector key missing from external sprites"

    out = crop.copy()
    out[1:-1, 1:-1] = bg

    used_connector_ids: set[int] = set()
    for src_col, pr, pc in target_pts:
        tgt_col = int(color_map[src_col])
        matches = []
        for idx, conn in enumerate(inner_connectors):
            if idx in used_connector_ids:
                continue
            key = conn["key"]
            if key not in sprite_by_key:
                continue
            for sprite in sprite_by_key[key]:
                ar, ac = sprite["anchor"]
                off_r = pr - ar
                off_c = pc - ac
                predicted = frozenset((off_r - 2 + dr, off_c + dc) for dr, dc in sprite["conn_shape"])
                if predicted != conn["pts"]:
                    continue
                matches.append((idx, sprite))
        assert len(matches) == 1, f"expected one connector-target match for target {(src_col, pr, pc)}, found {len(matches)}"
        idx, sprite = matches[0]
        used_connector_ids.add(idx)
        ar, ac = sprite["anchor"]
        off_r = pr - ar
        off_c = pc - ac
        for dr, dc in sprite["coords"]:
            rr = off_r + dr
            cc = off_c + dc
            assert 1 <= rr < H - 1 and 1 <= cc < W - 1, f"placed sprite for color {src_col} leaves frame interior"
            assert out[rr, cc] in (bg, tgt_col), f"sprite overlap conflict at {(rr, cc)}"
            out[rr, cc] = tgt_col
    assert len(used_connector_ids) == len(inner_connectors), "some interior connectors were unused"
    return out


# --------------- generator ---------------

FRAME_COL = 5
SPRITE_N_RANGE = (4, 6)
GRID_ROWS_RANGE = (2, 4)
GRID_COLS_RANGE = (3, 5)
TOTAL_BLOCKS_RANGE = (6, 10)
MAX_EMPTY_SLOTS = 2
NUM_COLORS_RANGE = (3, 5)
SPRITE_FILL_MIN = 0.5
SPRITE_FILL_MAX = 0.9
MIN_FRAME_H = 13
MIN_FRAME_W = 18
MAX_FRAME_DIM = 21
FRAME_MARGIN_SLACK = 4
CANVAS_H_RANGE = (22, 31)
CANVAS_W_RANGE = (23, 31)
FRAME_POS_JITTER_PROB = 0.25
PATCH_PADDING = 1
GEN_RETRIES = 500

ROW_GAP = 2
COL_GAP = 1
TOP_MARGIN_MIN = 2
BOTTOM_MARGIN_MIN = 1
SIDE_MARGIN_MIN = 1


def _legend(src: int, tgt: int) -> np.ndarray:
    return np.array(
        [[src, src, tgt, tgt], [src, src, src, tgt], [src, src, tgt, tgt]],
        dtype=int,
    )


def _is_4_connected(mask: np.ndarray) -> bool:
    if not mask.any():
        return False
    return len(components_mask(mask, connectivity=4)) == 1


def _sprite_occupancy(n: int) -> np.ndarray | None:
    lo = max((n * n + 1) // 2, int(np.ceil(SPRITE_FILL_MIN * n * n)))
    hi = min(int(np.floor(SPRITE_FILL_MAX * n * n)), n * n)
    if hi < lo:
        return None
    target = random.randint(lo, hi)

    for _ in range(80):
        mask = np.ones((n, n), dtype=bool)
        cells = [(r, c) for r in range(n) for c in range(n)]
        random.shuffle(cells)
        for r, c in cells:
            if int(mask.sum()) <= target:
                break
            trial = mask.copy()
            trial[r, c] = False
            if _is_4_connected(trial) and np.all(trial.any(axis=0)) and np.all(trial.any(axis=1)):
                mask = trial
        if int(mask.sum()) == target and _is_4_connected(mask) and np.all(mask.any(axis=0)) and np.all(mask.any(axis=1)):
            return mask
    return None


def _anchor_candidates(mask: np.ndarray) -> list[tuple[int, int]]:
    n, _ = mask.shape
    center = (n // 2, n // 2)
    preferred = []
    fallback = []
    for r, c in zip(*np.where(mask), strict=False):
        body = mask.copy()
        body[r, c] = False
        if not body.any():
            continue
        if not (_is_4_connected(body) and np.all(body.any(axis=0)) and np.all(body.any(axis=1))):
            continue
        cell = (int(r), int(c))
        if cell != center:
            preferred.append(cell)
        else:
            fallback.append(cell)
    return preferred + fallback


def _colored_output(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = int(color)
    return out


def _sprite_patch(
    mask: np.ndarray,
    body_col: int,
    anchor: tuple[int, int],
    ctrl_col: int,
    shape: tuple[tuple[int, int], ...],
    bg: int,
) -> np.ndarray:
    n = mask.shape[0]
    patch = np.full((n + 2, n), bg, dtype=int)
    for dr, dc in shape:
        patch[dr, dc] = int(ctrl_col)
    body = np.full((n, n), bg, dtype=int)
    body[mask] = int(body_col)
    ar, ac = anchor
    body[ar, ac] = FRAME_COL
    paste(patch, body, 2, 0, transparent=bg)
    return patch


def _content_extent(n: int, grid_rows: int, grid_cols: int) -> tuple[int, int]:
    return (grid_rows * n + (grid_rows - 1) * ROW_GAP, grid_cols * n + (grid_cols - 1) * COL_GAP)


def _choose_structure() -> tuple[int, int, int, int, int] | None:
    total = random.randint(*TOTAL_BLOCKS_RANGE)
    options = []
    for n in range(*SPRITE_N_RANGE):
        for rows in range(*GRID_ROWS_RANGE):
            for cols in range(*GRID_COLS_RANGE):
                ch, cw = _content_extent(n, rows, cols)
                base_fh = ch + TOP_MARGIN_MIN + BOTTOM_MARGIN_MIN + 2
                base_fw = cw + 2 * SIDE_MARGIN_MIN + 2
                slots = rows * cols
                empty = slots - total
                if base_fh <= MAX_FRAME_DIM and base_fw <= MAX_FRAME_DIM and 0 <= empty <= MAX_EMPTY_SLOTS:
                    options.append((n, rows, cols, empty, total))
    return random.choice(options) if options else None


def _choose_empty_slots(
    rows: int, cols: int, empty_count: int
) -> tuple[set[tuple[int, int]], list[tuple[int, int]]]:
    slots = [(r, c) for r in range(rows) for c in range(cols)]
    valid = []
    for combo in itertools.combinations(slots, empty_count):
        empty = set(combo)
        filled = [s for s in slots if s not in empty]
        if len({r for r, _ in filled}) == rows and len({c for _, c in filled}) == cols:
            valid.append(empty)
    empty_slots = random.choice(valid)
    filled_slots = [s for s in slots if s not in empty_slots]
    random.shuffle(filled_slots)
    return empty_slots, filled_slots


def _make_bordered_frame(h: int, w: int, bg: int) -> np.ndarray:
    frame = np.full((h, w), bg, dtype=int)
    frame[0, :] = frame[-1, :] = FRAME_COL
    frame[:, 0] = frame[:, -1] = FRAME_COL
    return frame


def _place_patch(
    grid: np.ndarray,
    reserve: np.ndarray,
    patch: np.ndarray,
    bg: int,
    *,
    transparent: bool,
) -> bool:
    blocker = np.full(patch.shape, FRAME_COL, dtype=int)
    try:
        r, c = random_free_location(reserve, blocker, transparent=bg, padding=PATCH_PADDING)
    except ValueError:
        return False
    paste(reserve, blocker, r, c)
    paste(grid, patch, r, c, transparent=bg if transparent else None)
    return True


def _sample_connector_keys(ctrl_cols: list[int], k: int) -> list[tuple[int, tuple[tuple[int, int], ...]]] | None:
    all_pairs = [(ctrl, CONNECTOR_SHAPES[name]) for ctrl in ctrl_cols for name in sorted(CONNECTOR_SHAPES)]
    for _ in range(300):
        unique_n = random.randint(max(2, k - 2), k - 1)
        chosen = random.sample(all_pairs, unique_n)
        while len(chosen) < k:
            chosen.append(random.choice(chosen))
        random.shuffle(chosen)
        cols = [ctrl for ctrl, _ in chosen]
        shapes = [shape for _, shape in chosen]
        if not any(chosen.count(key) > 1 for key in set(chosen)):
            continue
        if len(set(cols)) == k:
            continue
        if len(set(shapes)) == k:
            continue
        return chosen
    return None


def _gen_attempt() -> np.ndarray | None:
    struct = _choose_structure()
    if struct is None:
        return None
    n, rows, cols, empty_count, _total = struct

    bg = random.choice([c for c in range(10) if c != FRAME_COL])
    usable = [c for c in range(10) if c not in (bg, FRAME_COL)]

    k = random.randint(*NUM_COLORS_RANGE)
    if k > rows * cols - empty_count:
        return None
    src_cols = random.sample(usable, k)

    ctrl_pool = [c for c in usable if c not in src_cols]
    if len(ctrl_pool) < 2:
        return None
    ctrl_cols = random.sample(ctrl_pool, 2)
    conn_keys = _sample_connector_keys(ctrl_cols, k)
    if conn_keys is None:
        return None

    body_pool = [c for c in usable if c not in src_cols and c not in ctrl_cols]
    if not body_pool:
        return None
    body_cols = {src: random.choice(body_pool) for src in src_cols}

    tgt_cols = [random.choice([c for c in usable if c != src]) for src in src_cols]
    color_to_tgt = dict(zip(src_cols, tgt_cols))
    src_to_key = dict(zip(src_cols, conn_keys))

    sprite_masks: dict[int, np.ndarray] = {}
    anchor_by_src: dict[int, tuple[int, int]] = {}
    for _ in range(120):
        sprite_masks.clear()
        anchor_by_src.clear()
        used_anchors: set[tuple[int, int]] = set()
        ok = True
        for src in src_cols:
            mask = _sprite_occupancy(n)
            if mask is None:
                ok = False
                break
            candidates = _anchor_candidates(mask)
            if not candidates:
                ok = False
                break
            fresh = [a for a in candidates if a not in used_anchors]
            anchor = random.choice(fresh or candidates)
            sprite_masks[src] = mask
            anchor_by_src[src] = anchor
            used_anchors.add(anchor)
        if not ok:
            continue
        if len(set(anchor_by_src.values())) < min(2, k):
            continue
        if all(anchor == (n // 2, n // 2) for anchor in anchor_by_src.values()):
            continue
        groups: dict[tuple[int, tuple[tuple[int, int], ...]], list[int]] = {}
        for src in src_cols:
            groups.setdefault(src_to_key[src], []).append(src)
        if not any(len(group) > 1 for group in groups.values()):
            continue
        valid_groups = True
        for group in groups.values():
            if len(group) < 2:
                continue
            anchors = [anchor_by_src[src] for src in group]
            masks = [sprite_masks[src].tobytes() for src in group]
            if len(set(anchors)) != len(anchors):
                valid_groups = False
                break
            if len(set(masks)) != len(masks):
                valid_groups = False
                break
        if valid_groups:
            break
    if not sprite_masks or not anchor_by_src:
        return None
    if all(anchor[1] == 1 for anchor in anchor_by_src.values()):
        return None

    ch, cw = _content_extent(n, rows, cols)
    min_fh = max(MIN_FRAME_H, ch + TOP_MARGIN_MIN + BOTTOM_MARGIN_MIN + 2)
    min_fw = max(MIN_FRAME_W, cw + 2 * SIDE_MARGIN_MIN + 2)
    max_fh = min(MAX_FRAME_DIM, max(min_fh, ch + TOP_MARGIN_MIN + BOTTOM_MARGIN_MIN + FRAME_MARGIN_SLACK + 2))
    max_fw = min(MAX_FRAME_DIM, max(min_fw, cw + 2 * SIDE_MARGIN_MIN + FRAME_MARGIN_SLACK + 2))
    if min_fh > max_fh or min_fw > max_fw:
        return None
    frame_h = random.randint(min_fh, max_fh)
    frame_w = random.randint(min_fw, max_fw)
    y_margin = frame_h - 2 - ch
    x_margin = frame_w - 2 - cw
    if y_margin < TOP_MARGIN_MIN + BOTTOM_MARGIN_MIN or x_margin < 2 * SIDE_MARGIN_MIN:
        return None
    top_margin = random.randint(TOP_MARGIN_MIN, y_margin - BOTTOM_MARGIN_MIN)
    left_margin = random.randint(SIDE_MARGIN_MIN, x_margin - SIDE_MARGIN_MIN)

    _empty_slots, filled_slots = _choose_empty_slots(rows, cols, empty_count)
    cell_colors: dict[tuple[int, int], int] = {}
    for slot, col in zip(filled_slots[:k], src_cols, strict=False):
        cell_colors[slot] = col
    for slot in filled_slots[k:]:
        cell_colors[slot] = random.choice(src_cols)

    input_frame = _make_bordered_frame(frame_h, frame_w, bg)
    output_frame = _make_bordered_frame(frame_h, frame_w, bg)
    for (rr, cc), src in cell_colors.items():
        r0 = 1 + top_margin + rr * (n + ROW_GAP)
        c0 = 1 + left_margin + cc * (n + COL_GAP)
        ar, ac = anchor_by_src[src]
        input_frame[r0 + ar, c0 + ac] = int(src)
        ctrl_col, shape = src_to_key[src]
        for dr, dc in shape:
            input_frame[r0 - 2 + dr, c0 + dc] = int(ctrl_col)
        paste(output_frame, _colored_output(sprite_masks[src], color_to_tgt[src], bg), r0, c0, transparent=bg)

    ext_area = k * (12 + n * (n + 2) + 10)
    possible_dims = [
        (h, w)
        for h in range(*CANVAS_H_RANGE)
        for w in range(*CANVAS_W_RANGE)
        if h >= frame_h and w >= frame_w and h * w - frame_h * frame_w >= ext_area
    ]
    if not possible_dims:
        return None
    h, w = random.choice(possible_dims)
    grid = np.full((h, w), bg, dtype=int)
    reserve = np.full((h, w), bg, dtype=int)

    fr0 = random.choice([0, h - frame_h])
    fc0 = random.choice([0, w - frame_w])
    if h - frame_h > 2 and random.random() < FRAME_POS_JITTER_PROB:
        fr0 = random.randrange(0, h - frame_h + 1)
    if w - frame_w > 2 and random.random() < FRAME_POS_JITTER_PROB:
        fc0 = random.randrange(0, w - frame_w + 1)
    paste(grid, input_frame, fr0, fc0)
    paste(reserve, np.full((frame_h, frame_w), FRAME_COL, dtype=int), fr0, fc0)

    patches: list[tuple[np.ndarray, bool]] = []
    for src in src_cols:
        ctrl_col, shape = src_to_key[src]
        patches.append((_sprite_patch(sprite_masks[src], body_cols[src], anchor_by_src[src], ctrl_col, shape, bg), True))
        patches.append((_legend(src, color_to_tgt[src]), False))
    random.shuffle(patches)
    patches.sort(key=lambda item: item[0].size, reverse=True)
    for patch, transparent in patches:
        if not _place_patch(grid, reserve, patch, bg, transparent=transparent):
            return None

    try:
        if not np.array_equal(main(grid), output_frame):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        out = _gen_attempt()
        if out is not None:
            return out
    raise RuntimeError("failed to generate valid edb79dae combo input")
