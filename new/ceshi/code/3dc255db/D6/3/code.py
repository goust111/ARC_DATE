# Variant of 3dc255db
# Dimension: D6 — 变换操作 (transform)
# Change: erase — 弹射后杯口内的内容像素被擦除(变为背景色)... → 碰撞变色
#
# Design analysis
# spec:
#   variant_goal:
#     keep:
#       - cup detection logic: find each hollow cup by its border color/component
#       - tail inference: the spike is still the unique 1-pixel edge point on the bbox
#       - emission rule: the number of emitted pixels is still the number of same-colored
#         content pixels found in the cup mouth
#     change:
#       - old rule erased the source content after emission
#       - new rule keeps the source content in place and uses the emitted line as a
#         color-changing ray: if the line reaches another cup's border, that hit border
#         pixel changes to the ray color
#     why this is a real solver change:
#       - the original solver already blindly overwrote any traversed cell, so merely
#         generating collision cases would not change main(input) on the same input
#       - to avoid degeneration, the new solver must differ on the same input grid
#       - therefore the source content is no longer erased; the visible transform is the
#         added ray plus border recoloring at collisions
#   concrete_behavior:
#     - input still shows 2-3 cups and their mouth contents
#     - output keeps those same mouth contents
#     - each cup additionally emits a straight ray outward from its spike
#     - at least one ray ends exactly on another cup border, recoloring that border cell
#     - other rays may simply extend through background
#   traced_example:
#     old:
#       - cup A has 3 content cells
#       - solver removes those 3 source cells
#       - solver paints 3 cells after the spike
#       - if another cup border sits on step 3, it is overwritten incidentally
#     new:
#       - cup A still has 3 content cells in the output
#       - solver paints the same 3 outward cells
#       - if step 3 lands on cup B border, that border cell becomes A's content color
#     difference:
#       - source cells persist
#       - collision recolor is now the intended observable transform, not an accidental
#         consequence of a path-clear assumption
#   complexity_calibration:
#     - generator forces at least one collision pair per grid so the new rule is visible
#     - grids use 2 or 3 cups; 1-cup scenes would hide the cup-to-cup interaction
#     - collision rays are constrained to end on the first hit border cell, which keeps
#       the picture readable and avoids rays tunneling through another cup interior
#   hidden_assumption_audit_main:
#     - candidate spike detection still assumes a cup has at least one bbox edge with a
#       single non-corner border pixel; generator preserves this
#     - mouth detection still assumes flooding from the opposite bbox side reaches the
#       true interior/content; generator preserves this
#     - content color selection still assumes one dominant non-bg, non-border color in
#       the mouth; generator scatters exactly one content color per cup
#     - line painting must not depend on output mutations while later cups are detected;
#       main therefore first extracts all cups from the immutable input grid, then paints
#   placement_legality:
#     - cup bboxes keep a 1-cell gap
#     - clear-path cups still reserve an unobstructed path to the edge
#     - the designated collision shooter is the only cup whose active ray may touch
#       another cup, and it is forced to stop exactly on the first hit border cell
#     - no ray-path overlaps are allowed, and no later cup may be placed on an earlier
#       reserved path
#   visual_observability:
#     - every generated sample has at least one recolored border cell on output
#     - because source content remains visible, the change from input to output is easy
#       to spot even before inferring the collision condition
#     - the shooter content color is chosen different from the target border color so the
#       collision recolor is always visible

from __future__ import annotations

from collections import Counter

import numpy as np
from arc_utils import bbox_coords, colors_present, components_mask, get_bg
from scipy import ndimage


def _flood_from_bbox_side(border_set: set[tuple[int, int]], bbox: tuple[int, int, int, int], side: str) -> set[tuple[int, int]]:
    """4-conn flood within bbox, seeded from non-border cells on one bbox side."""
    r0, c0, r1, c1 = bbox
    if side == "U":
        seeds = [(r0, c) for c in range(c0, c1 + 1) if (r0, c) not in border_set]
    elif side == "D":
        seeds = [(r1, c) for c in range(c0, c1 + 1) if (r1, c) not in border_set]
    elif side == "L":
        seeds = [(r, c0) for r in range(r0, r1 + 1) if (r, c0) not in border_set]
    else:  # "R"
        seeds = [(r, c1) for r in range(r0, r1 + 1) if (r, c1) not in border_set]

    visited = set(seeds)
    stack = list(seeds)
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (r0 <= nr <= r1 and c0 <= nc <= c1):
                continue
            if (nr, nc) in visited or (nr, nc) in border_set:
                continue
            visited.add((nr, nc))
            stack.append((nr, nc))
    return visited


def _extract_cups(g: np.ndarray, bg: int) -> list[dict]:
    """Detect all actionable cups from the immutable input grid."""
    OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
    cups: list[dict] = []
    for border_col in colors_present(g, exclude={bg}):
        for border_coords in components_mask(g == int(border_col), connectivity=8):
            if border_coords.size == 0 or int(border_coords.shape[0]) < 5:
                continue

            r0, c0, r1, c1 = (int(v) for v in bbox_coords(border_coords))
            bbox = (r0, c0, r1, c1)
            border_set = {(int(r), int(c)) for r, c in border_coords}
            corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}

            edges = (
                ("U", [p for p in border_set if p[0] == r0]),
                ("D", [p for p in border_set if p[0] == r1]),
                ("L", [p for p in border_set if p[1] == c0]),
                ("R", [p for p in border_set if p[1] == c1]),
            )
            candidates = [(d, cells[0]) for d, cells in edges if len(cells) == 1 and cells[0] not in corners]
            if not candidates:
                continue

            passing = []
            for d, spike in candidates:
                mouth = _flood_from_bbox_side(border_set, bbox, OPPOSITE[d])
                cnt = Counter()
                for r, c in mouth:
                    v = int(g[r, c])
                    if v != bg and v != int(border_col):
                        cnt[v] += 1
                if not cnt:
                    continue
                content_col = int(cnt.most_common(1)[0][0])
                content_cells = [p for p in mouth if int(g[p[0], p[1]]) == content_col]
                passing.append((d, spike, content_col, content_cells))

            if len(passing) != 1:
                continue
            tail_dir, spike, content_col, content_cells = passing[0]
            cups.append(
                {
                    "tail_dir": tail_dir,
                    "spike": spike,
                    "content_col": content_col,
                    "content_count": len(content_cells),
                }
            )
    return cups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - Keep the source content inside each cup.
    - Emit a straight line from each spike using the cup's content color and count.
    - If that line reaches another cup border, the struck border pixel changes color.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
    cups = _extract_cups(g, bg)

    for cup in cups:
        dr, dc = DIR4[cup["tail_dir"]]
        sr, sc = cup["spike"]
        for t in range(1, cup["content_count"] + 1):
            rr = sr + dr * t
            cc = sc + dc * t
            if not (0 <= rr < H and 0 <= cc < W):
                break
            out[rr, cc] = cup["content_col"]

    return out


# --------------- generator ---------------

GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
NUM_CUPS_CHOICES = (2, 3)

CUP_BBOX_MAX_UD = (8, 7)
CUP_BBOX_MAX_LR = (7, 8)

ARM_LEN_RANGE = (1, 3)
WALL_EXT_RANGE = (0, 4)

CONTENT_COUNT_MAX = 7

STRAY_PROB = 0.35
STRAY_USE_BORDER_PROB = 0.5

GEN_RETRIES_PRIMARY = 400
GEN_RETRIES_FALLBACK = 400

DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
_STRUCT8 = np.ones((3, 3), dtype=bool)


def _build_canonical_cup(L: int, R: int, hL: int, hR: int, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a Y/funnel cup with tail pointing up."""
    assert L >= 1 and R >= 1 and hL >= 0 and hR >= 0 and (hL + hR) >= 1
    w = L + R + 1
    h = max(L + hL, R + hR) + 1
    g = np.zeros((h, w), dtype=int)
    g[0, L] = color
    g[np.arange(1, L + 1), L - np.arange(1, L + 1)] = color
    g[np.arange(1, R + 1), L + np.arange(1, R + 1)] = color
    if hL:
        g[L + 1 : L + hL + 1, 0] = color
    if hR:
        g[R + 1 : R + hR + 1, w - 1] = color
    return g, (0, L)


def _mouth_interior_coords(patch: np.ndarray, border_color: int, opposite_side: str) -> list[tuple[int, int]]:
    """Flood-fill the patch interior from the mouth side."""
    h, w = patch.shape
    border_mask = patch == border_color
    if opposite_side == "U":
        seeds = [(0, c) for c in range(w) if not border_mask[0, c]]
    elif opposite_side == "D":
        seeds = [(h - 1, c) for c in range(w) if not border_mask[h - 1, c]]
    elif opposite_side == "L":
        seeds = [(r, 0) for r in range(h) if not border_mask[r, 0]]
    else:  # "R"
        seeds = [(r, w - 1) for r in range(h) if not border_mask[r, w - 1]]
    visited = set(seeds)
    stack = list(seeds)
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited and not border_mask[nr, nc]:
                visited.add((nr, nc))
                stack.append((nr, nc))
    return sorted(visited)


def _rotate_to_direction(
    patch: np.ndarray, spike: tuple[int, int], mouth: list[tuple[int, int]], direction: str
) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    """Rotate a canonical (tail=U) patch to the target tail direction."""
    h, w = patch.shape
    sr, sc = spike
    if direction == "U":
        return patch.copy(), (sr, sc), list(mouth)
    if direction == "D":
        return np.flipud(patch).copy(), (h - 1 - sr, sc), [(h - 1 - r, c) for (r, c) in mouth]
    if direction == "L":
        return np.rot90(patch, k=1).copy(), (w - 1 - sc, sr), [(w - 1 - c, r) for (r, c) in mouth]
    if direction == "R":
        return np.rot90(patch, k=-1).copy(), (sc, h - 1 - sr), [(c, h - 1 - r) for (r, c) in mouth]
    raise ValueError(direction)


def _sample_cup_shape(rng: np.random.Generator, direction: str) -> tuple[int, int, int, int] | None:
    """Pick canonical cup parameters that fit after rotation."""
    if direction in ("U", "D"):
        max_h_canon, max_w_canon = CUP_BBOX_MAX_UD
    else:
        max_h_canon, max_w_canon = CUP_BBOX_MAX_LR

    lo, hi = ARM_LEN_RANGE
    L_arm = int(rng.integers(lo, hi + 1))
    R_arm = int(rng.integers(lo, hi + 1))
    while L_arm + R_arm + 1 > max_w_canon:
        if L_arm >= R_arm:
            L_arm -= 1
        else:
            R_arm -= 1

    wlo, whi = WALL_EXT_RANGE
    hL = int(rng.integers(wlo, whi + 1))
    hR = int(rng.integers(wlo, whi + 1))
    if hL + hR == 0:
        hL = 1
    while max(L_arm + hL, R_arm + hR) + 1 > max_h_canon:
        if L_arm + hL >= R_arm + hR and hL > 0:
            hL -= 1
        elif hR > 0:
            hR -= 1
        else:
            break
    if hL + hR == 0:
        return None
    return L_arm, R_arm, hL, hR


def _sample_blueprint(
    rng: np.random.Generator, border_color: int, content_color: int
) -> dict | None:
    """Sample one cup shape, direction, spike and mouth before placement."""
    direction = str(rng.choice(["U", "D", "L", "R"]))
    shape = _sample_cup_shape(rng, direction)
    if shape is None:
        return None
    L_arm, R_arm, hL, hR = shape
    patch_canon, spike_canon = _build_canonical_cup(L_arm, R_arm, hL, hR, border_color)
    mouth_canon = _mouth_interior_coords(patch_canon, border_color, opposite_side="D")
    if len(mouth_canon) < 3:
        return None
    patch, spike, mouth = _rotate_to_direction(patch_canon, spike_canon, mouth_canon, direction)
    return {
        "border_color": border_color,
        "content_color": content_color,
        "direction": direction,
        "patch": patch,
        "spike": spike,
        "mouth": mouth,
    }


def _path_to_edge(H: int, W: int, spike_abs: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    dr, dc = DIR4[direction]
    rr, cc = spike_abs[0] + dr, spike_abs[1] + dc
    path = []
    while 0 <= rr < H and 0 <= cc < W:
        path.append((rr, cc))
        rr += dr
        cc += dc
    return path


def _bbox_has_gap(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> bool:
    return b1[2] < b2[0] - 1 or b2[2] < b1[0] - 1 or b1[3] < b2[1] - 1 or b2[3] < b1[1] - 1


def _all_bboxes_have_gap(bbox: tuple[int, int, int, int], placed: list[dict]) -> bool:
    return all(_bbox_has_gap(bbox, p["bbox"]) for p in placed)


def _cell_in_bbox(cell: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = cell
    return bbox[0] <= r <= bbox[2] and bbox[1] <= c <= bbox[3]


def _occupied_bbox_cell(cell: tuple[int, int], placed: list[dict]) -> bool:
    return any(_cell_in_bbox(cell, p["bbox"]) for p in placed)


def _reserved_cells(placed: list[dict]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for p in placed:
        out.update(p["reserved_path"])
    return out


def _patch_border_cells(patch: np.ndarray, r0: int, c0: int) -> set[tuple[int, int]]:
    coords = np.argwhere(patch != 0)
    return {(r0 + int(r), c0 + int(c)) for r, c in coords}


def _commit_placed(
    blueprint: dict,
    r0: int,
    c0: int,
    path_full: list[tuple[int, int]],
    content_count: int,
    reserve_full_path: bool,
) -> dict:
    patch = blueprint["patch"]
    ph, pw = patch.shape
    border_abs = _patch_border_cells(patch, r0, c0)
    active_shot = path_full[:content_count]
    return {
        **blueprint,
        "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
        "r0": r0,
        "c0": c0,
        "border_abs": border_abs,
        "mouth_abs": [(r0 + r, c0 + c) for (r, c) in blueprint["mouth"]],
        "path_full": path_full,
        "content_count": content_count,
        "active_shot": active_shot,
        "reserved_path": set(path_full if reserve_full_path else active_shot),
    }


def _try_place_clear_cup(
    rng: np.random.Generator, H: int, W: int, blueprint: dict, placed: list[dict]
) -> dict | None:
    """Place a non-colliding cup whose full path to the edge stays unobstructed."""
    patch = blueprint["patch"]
    ph, pw = patch.shape
    reserved = _reserved_cells(placed)
    offsets = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(offsets)
    for r0, c0 in offsets:
        bbox = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        if not _all_bboxes_have_gap(bbox, placed):
            continue
        border_abs = _patch_border_cells(patch, r0, c0)
        if border_abs & reserved:
            continue
        spike_abs = (r0 + blueprint["spike"][0], c0 + blueprint["spike"][1])
        path_full = _path_to_edge(H, W, spike_abs, blueprint["direction"])
        if not path_full:
            continue
        if any(_occupied_bbox_cell(cell, placed) for cell in path_full):
            continue
        if any(cell in reserved for cell in path_full):
            continue
        max_count = min(CONTENT_COUNT_MAX, len(blueprint["mouth"]), len(path_full))
        if max_count < 1:
            continue
        content_count = int(rng.integers(1, max_count + 1))
        return _commit_placed(blueprint, r0, c0, path_full, content_count, reserve_full_path=True)
    return None


def _try_place_collision_shooter(
    rng: np.random.Generator, H: int, W: int, blueprint: dict, target: dict, placed: list[dict]
) -> dict | None:
    """Place the designated shooter so its ray ends on the first target border hit."""
    patch = blueprint["patch"]
    ph, pw = patch.shape
    reserved = _reserved_cells(placed)
    max_count = min(CONTENT_COUNT_MAX, len(blueprint["mouth"]))
    if max_count < 2:
        return None

    offsets = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(offsets)
    for r0, c0 in offsets:
        bbox = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        if not _all_bboxes_have_gap(bbox, placed):
            continue
        border_abs = _patch_border_cells(patch, r0, c0)
        if border_abs & reserved:
            continue
        spike_abs = (r0 + blueprint["spike"][0], c0 + blueprint["spike"][1])
        path_full = _path_to_edge(H, W, spike_abs, blueprint["direction"])
        if not path_full:
            continue

        hit_idx = None
        valid = True
        for idx, cell in enumerate(path_full[:max_count], start=1):
            if cell in target["border_abs"]:
                hit_idx = idx
                break
            if cell in reserved or _occupied_bbox_cell(cell, placed):
                valid = False
                break
        if not valid or hit_idx is None or hit_idx < 2:
            continue
        return _commit_placed(blueprint, r0, c0, path_full, hit_idx, reserve_full_path=False)
    return None


def _stray_candidate_mask(H: int, W: int, bbox: tuple[int, int, int, int], tail_dir: str) -> np.ndarray:
    """Cells on the mouth side of a cup bbox."""
    r0, c0, r1, c1 = bbox
    mask = np.zeros((H, W), dtype=bool)
    if tail_dir == "R":
        mask[r0 : r1 + 1, :c0] = True
    elif tail_dir == "L":
        mask[r0 : r1 + 1, c1 + 1 :] = True
    elif tail_dir == "D":
        mask[:r0, c0 : c1 + 1] = True
    else:
        mask[r1 + 1 :, c0 : c1 + 1] = True
    return mask


def _try_place_stray(
    rng: np.random.Generator, canvas_in: np.ndarray, canvas_out: np.ndarray, cups_meta: list[dict]
) -> None:
    if not cups_meta or rng.random() >= STRAY_PROB:
        return
    H, W = canvas_in.shape
    meta = cups_meta[int(rng.integers(0, len(cups_meta)))]
    stray_color = meta["border_color"] if rng.random() < STRAY_USE_BORDER_PROB else meta["content_color"]

    forbidden = ndimage.binary_dilation(canvas_in != 0, structure=_STRUCT8)
    for m in cups_meta:
        br0, bc0, br1, bc1 = m["bbox"]
        forbidden[br0 : br1 + 1, bc0 : bc1 + 1] = True
        for r, c in m["active_shot"]:
            forbidden[r, c] = True

    candidate_mask = _stray_candidate_mask(H, W, meta["bbox"], meta["direction"]) & ~forbidden
    candidates = np.argwhere(candidate_mask)
    if candidates.size == 0:
        return
    r, c = candidates[int(rng.integers(0, len(candidates)))]
    canvas_in[r, c] = stray_color
    canvas_out[r, c] = stray_color


def _scatter_content(rng: np.random.Generator, canvas_in: np.ndarray, canvas_out: np.ndarray, cups_meta: list[dict]) -> bool:
    """Scatter each cup's source content; in this variant the content remains in output."""
    for meta in cups_meta:
        mouth_abs = meta["mouth_abs"]
        count = meta["content_count"]
        if len(mouth_abs) < count:
            return False
        idxs = rng.choice(len(mouth_abs), size=count, replace=False)
        for i in idxs:
            r, c = mouth_abs[int(i)]
            canvas_in[r, c] = meta["content_color"]
            canvas_out[r, c] = meta["content_color"]
    return True


def _render_borders_and_shots(H: int, W: int, cups_meta: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    canvas_in = np.zeros((H, W), dtype=int)
    canvas_out = np.zeros((H, W), dtype=int)
    for meta in cups_meta:
        for r, c in meta["border_abs"]:
            canvas_in[r, c] = meta["border_color"]
            canvas_out[r, c] = meta["border_color"]
    for meta in cups_meta:
        for r, c in meta["active_shot"]:
            canvas_out[r, c] = meta["content_color"]
    return canvas_in, canvas_out


def _sample_colors(rng: np.random.Generator, num_cups: int) -> list[tuple[int, int]]:
    """Choose colors while keeping collision recolors visible and total colors compact."""
    colors = list(range(1, 10))
    rng.shuffle(colors)
    border_colors = colors[:num_cups]
    pool = colors[num_cups:]

    shooter_content = int(rng.choice([c for c in pool + border_colors if c not in {border_colors[0], border_colors[1]}]))
    target_content = int(rng.choice([c for c in pool + border_colors if c != border_colors[0]]))
    pairs = [(border_colors[0], target_content), (border_colors[1], shooter_content)]

    for i in range(2, num_cups):
        choices = [c for c in (pool + [target_content, shooter_content]) if c != border_colors[i]]
        pairs.append((border_colors[i], int(rng.choice(choices))))
    return pairs


def _gen_attempt(rng: np.random.Generator, num_cups: int | None = None) -> np.ndarray | None:
    """
    Build a scene with at least one real border collision.

    Cup 0 = target, cup 1 = designated collision shooter, cup 2 (optional) = clear filler.
    """
    if num_cups is None:
        num_cups = int(rng.choice(NUM_CUPS_CHOICES))
    H = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    W = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))

    color_pairs = _sample_colors(rng, num_cups)
    blueprints = []
    for border_color, content_color in color_pairs:
        bp = _sample_blueprint(rng, border_color, content_color)
        if bp is None:
            return None
        blueprints.append(bp)

    target = _try_place_clear_cup(rng, H, W, blueprints[0], placed=[])
    if target is None:
        return None

    shooter = _try_place_collision_shooter(rng, H, W, blueprints[1], target=target, placed=[target])
    if shooter is None:
        return None

    cups_meta = [target, shooter]
    for bp in blueprints[2:]:
        filler = _try_place_clear_cup(rng, H, W, bp, placed=cups_meta)
        if filler is None:
            return None
        cups_meta.append(filler)

    canvas_in, canvas_out = _render_borders_and_shots(H, W, cups_meta)
    if not _scatter_content(rng, canvas_in, canvas_out, cups_meta):
        return None
    _try_place_stray(rng, canvas_in, canvas_out, cups_meta)

    produced = main(canvas_in)
    if not np.array_equal(produced, canvas_out):
        return None

    # Ensure the new D6 behavior is actually visible: at least one target border pixel changes.
    collision_pixels = 0
    for meta in cups_meta:
        for r, c in meta["border_abs"]:
            if canvas_out[r, c] != meta["border_color"]:
                collision_pixels += 1
    if collision_pixels < 1:
        return None
    return canvas_in


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    num_cups = int(rng.choice(NUM_CUPS_CHOICES))
    for _ in range(GEN_RETRIES_PRIMARY):
        result = _gen_attempt(rng, num_cups=num_cups)
        if result is not None:
            return result
    for _ in range(GEN_RETRIES_FALLBACK):
        result = _gen_attempt(rng)
        if result is not None:
            return result
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES_PRIMARY + GEN_RETRIES_FALLBACK} attempts")
