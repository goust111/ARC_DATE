# Combo variant of 3dc255db (index 13)
# combo_spec: D4/0+D6/3
# Name: 障碍物停止+碰撞变色
#
# Design analysis
# a) Concrete behavior and trace:
#    - Keep the original cup objects: hollow Y/funnel frames with one spike on a bbox edge and
#      one scattered content color inside the mouth.
#    - New fused rule: content is still erased from the cup mouth, then emitted from the spike.
#      The ray paints background cells normally, but if the first blocker on that ray is another
#      cup's border pixel, that hit pixel itself is recolored to the ray color and the ray stops
#      there. Any remaining content pixels are discarded.
#    - Trace: suppose a right-pointing cup has 5 content pixels and its ray sees
#      [bg, bg, target-border, bg, bg]. Original task would erase the 5 source cells and paint
#      all 5 steps, overwriting the target border and continuing past it. This combo erases the
#      source cells, paints only the first 2 bg cells, recolors the 3rd hit border cell, then
#      stops. The collision pixel is both the endpoint and the recolored target.
# b) Complexity calibration:
#    - Every generated grid has 2 or 3 cups and at least one designated shooter->target pair.
#    - The shooter is forced to have content_count > collision_step, so "stop at obstacle" is a
#      real control variable instead of a decorative collision that would match the base task.
#    - The collision step varies by placement; the solver must detect geometry, not memorize a
#      fixed distance.
# c) Implicit-assumption audit of the original main():
#    - Cup detection by border-color components, bbox-edge singleton spikes, and mouth flood-fill
#      still works, so that logic is preserved.
#    - The original painting loop assumed every in-bounds tail cell was writable until the grid
#      edge. That assumption is broken: another cup border may now terminate the ray early.
#    - Output mutations can recolor target borders, so cup extraction must read only from the
#      immutable input grid `g`, not from progressively modified output.
# d) Placement legality:
#    - Cup bboxes keep a 1-cell gap.
#    - Reserved shot prefixes prevent ray/ray overlaps and prevent later cup borders from landing
#      on already active shot cells.
#    - The designated collision shooter is placed so the first occupied cell on its active prefix
#      is a target border pixel, never a target interior cell or unrelated reserved path cell.
# e) Visual observability:
#    - Each sample shows at least one ray ending exactly on another cup border.
#    - Because the source content disappears, the output difference is concentrated on a clear
#      ejection line plus one recolored border endpoint, which is legible from a few examples.
# f) Visual cleanliness:
#    - The generator omits unrelated stray noise; only cups, their mouth contents, and one forced
#      cup-to-cup collision are present.
#    - At most one forced collision is required, so the scene stays readable even with 3 cups.
# g) Color inferability:
#    - The shooter content color is chosen different from the target border color, so the collision
#      recolor is always visible as a structural endpoint of the ray.
#    - No hidden ordering by numeric color value is used; geometry alone identifies the collision.
# 9) New solver assumptions and generator guarantees:
#    - Assumption: actionable blockers are other cup border pixels on the ray. Guarantee: the
#      generator creates at least one such border hit and keeps unrelated non-bg noise off rays.
#    - Assumption: the first border hit is the stopping point. Guarantee: reserved prefixes ensure
#      no nearer blocker lies before the forced target border.
#    - Assumption: cup extraction remains unambiguous even though outputs recolor border pixels.
#      Guarantee: extraction is always performed from the untouched input, whose cup borders remain
#      single-color components.

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_coords, colors_present, components_mask, get_bg


DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}


def _flood_from_bbox_side(
    border_set: set[tuple[int, int]], bbox: tuple[int, int, int, int], side: str
) -> set[tuple[int, int]]:
    """4-connected flood within bbox from one bbox side, blocked by border cells."""
    r0, c0, r1, c1 = bbox
    if side == "U":
        seeds = [(r0, c) for c in range(c0, c1 + 1) if (r0, c) not in border_set]
    elif side == "D":
        seeds = [(r1, c) for c in range(c0, c1 + 1) if (r1, c) not in border_set]
    elif side == "L":
        seeds = [(r, c0) for r in range(r0, r1 + 1) if (r, c0) not in border_set]
    else:
        seeds = [(r, c1) for r in range(r0, r1 + 1) if (r, c1) not in border_set]

    visited = set(seeds)
    stack = list(seeds)
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4.values():
            nr, nc = r + dr, c + dc
            if not (r0 <= nr <= r1 and c0 <= nc <= c1):
                continue
            if (nr, nc) in visited or (nr, nc) in border_set:
                continue
            visited.add((nr, nc))
            stack.append((nr, nc))
    return visited


def _extract_cups(g: np.ndarray, bg: int) -> list[dict]:
    """Detect all cups from the immutable input grid."""
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
            for tail_dir, spike in candidates:
                mouth = _flood_from_bbox_side(border_set, bbox, OPPOSITE[tail_dir])
                cnt = Counter()
                for r, c in mouth:
                    v = int(g[r, c])
                    if v != bg and v != int(border_col):
                        cnt[v] += 1
                if not cnt:
                    continue
                content_col = int(cnt.most_common(1)[0][0])
                content_cells = [p for p in mouth if int(g[p[0], p[1]]) == content_col]
                passing.append((tail_dir, spike, content_col, content_cells))

            if len(passing) != 1:
                continue
            tail_dir, spike, content_col, content_cells = passing[0]
            cups.append(
                {
                    "border_col": int(border_col),
                    "border_abs": border_set,
                    "tail_dir": tail_dir,
                    "spike": spike,
                    "content_col": content_col,
                    "content_cells": content_cells,
                }
            )
    return cups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    - Erase each cup's source content from the mouth.
    - Emit a straight ray from the spike using the content color.
    - The first other-cup border hit is recolored to the ray color and terminates the ray.
    - Any leftover content pixels beyond the collision are discarded.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    cups = _extract_cups(g, bg)
    border_owner: dict[tuple[int, int], int] = {}
    for idx, cup in enumerate(cups):
        for cell in cup["border_abs"]:
            border_owner[cell] = idx

    for cup in cups:
        for r, c in cup["content_cells"]:
            out[r, c] = bg

    for idx, cup in enumerate(cups):
        dr, dc = DIR4[cup["tail_dir"]]
        sr, sc = cup["spike"]
        for t in range(1, len(cup["content_cells"]) + 1):
            rr = sr + dr * t
            cc = sc + dc * t
            if not (0 <= rr < H and 0 <= cc < W):
                break
            owner = border_owner.get((rr, cc))
            if owner is not None and owner != idx:
                out[rr, cc] = cup["content_col"]
                break
            if int(g[rr, cc]) != bg:
                break
            out[rr, cc] = cup["content_col"]

    return out


GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
NUM_CUPS_CHOICES = (2, 3)
CUP_BBOX_MAX_UD = (8, 7)
CUP_BBOX_MAX_LR = (7, 8)
ARM_LEN_RANGE = (1, 3)
WALL_EXT_RANGE = (0, 4)
CONTENT_COUNT_MAX = 7
GEN_RETRIES_PRIMARY = 1000
GEN_RETRIES_FALLBACK = 1000


def _build_canonical_cup(L: int, R: int, hL: int, hR: int, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a Y/funnel cup with tail pointing up."""
    assert L >= 1 and R >= 1 and hL >= 0 and hR >= 0 and (hL + hR) >= 1
    w = L + R + 1
    h = max(L + hL, R + hR) + 1
    patch = np.zeros((h, w), dtype=int)
    patch[0, L] = color
    patch[np.arange(1, L + 1), L - np.arange(1, L + 1)] = color
    patch[np.arange(1, R + 1), L + np.arange(1, R + 1)] = color
    if hL:
        patch[L + 1 : L + hL + 1, 0] = color
    if hR:
        patch[R + 1 : R + hR + 1, w - 1] = color
    return patch, (0, L)


def _mouth_interior_coords(patch: np.ndarray, border_color: int, opposite_side: str) -> list[tuple[int, int]]:
    """Flood-fill patch interior from the mouth side."""
    h, w = patch.shape
    border_mask = patch == border_color
    if opposite_side == "U":
        seeds = [(0, c) for c in range(w) if not border_mask[0, c]]
    elif opposite_side == "D":
        seeds = [(h - 1, c) for c in range(w) if not border_mask[h - 1, c]]
    elif opposite_side == "L":
        seeds = [(r, 0) for r in range(h) if not border_mask[r, 0]]
    else:
        seeds = [(r, w - 1) for r in range(h) if not border_mask[r, w - 1]]
    visited = set(seeds)
    stack = list(seeds)
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited and not border_mask[nr, nc]:
                visited.add((nr, nc))
                stack.append((nr, nc))
    return sorted(visited)


def _rotate_to_direction(
    patch: np.ndarray, spike: tuple[int, int], mouth: list[tuple[int, int]], direction: str
) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    """Rotate canonical tail-up patch to the target direction."""
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
    """Pick canonical cup parameters that fit the bbox constraints after rotation."""
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


def _sample_blueprint(rng: np.random.Generator, border_color: int, content_color: int) -> dict | None:
    """Sample one cup shape and orientation before placement."""
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


def _commit_clear(
    blueprint: dict, r0: int, c0: int, content_count: int, active_shot: list[tuple[int, int]]
) -> dict:
    patch = blueprint["patch"]
    ph, pw = patch.shape
    return {
        **blueprint,
        "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
        "border_abs": _patch_border_cells(patch, r0, c0),
        "mouth_abs": [(r0 + r, c0 + c) for (r, c) in blueprint["mouth"]],
        "content_count": content_count,
        "shot_cells": list(active_shot),
        "collision_cell": None,
        "collision_step": None,
        "reserved_path": set(active_shot),
    }


def _commit_collision(
    blueprint: dict,
    r0: int,
    c0: int,
    content_count: int,
    path_to_hit: list[tuple[int, int]],
    collision_cell: tuple[int, int],
    guard_cell: tuple[int, int] | None,
) -> dict:
    patch = blueprint["patch"]
    ph, pw = patch.shape
    return {
        **blueprint,
        "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
        "border_abs": _patch_border_cells(patch, r0, c0),
        "mouth_abs": [(r0 + r, c0 + c) for (r, c) in blueprint["mouth"]],
        "content_count": content_count,
        "shot_cells": list(path_to_hit),
        "collision_cell": collision_cell,
        "collision_step": len(path_to_hit) + 1,
        "reserved_path": set(path_to_hit) | {collision_cell} | ({guard_cell} if guard_cell is not None else set()),
    }


def _try_place_clear_cup(
    rng: np.random.Generator, H: int, W: int, blueprint: dict, placed: list[dict]
) -> dict | None:
    """Place a cup whose active emitted prefix remains unobstructed."""
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

        free_prefix = 0
        for cell in path_full:
            if cell in reserved or _occupied_bbox_cell(cell, placed):
                break
            free_prefix += 1
        max_count = min(CONTENT_COUNT_MAX, len(blueprint["mouth"]), free_prefix)
        if max_count < 1:
            continue
        content_count = int(rng.integers(1, max_count + 1))
        return _commit_clear(blueprint, r0, c0, content_count, path_full[:content_count])
    return None


def _try_place_collision_shooter(
    rng: np.random.Generator, H: int, W: int, blueprint: dict, target: dict, placed: list[dict]
) -> dict | None:
    """Place a shooter whose first blocker is a target border hit and whose count exceeds it."""
    patch = blueprint["patch"]
    ph, pw = patch.shape
    max_shot_len = min(CONTENT_COUNT_MAX, len(blueprint["mouth"]))
    if max_shot_len < 3:
        return None

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

        hit_idx = None
        for idx, cell in enumerate(path_full, start=1):
            if cell in reserved:
                hit_idx = None
                break
            if cell in target["border_abs"]:
                hit_idx = idx
                break
            if _occupied_bbox_cell(cell, placed):
                hit_idx = None
                break
            if idx >= max_shot_len - 1:
                break
        if hit_idx is None or hit_idx < 2 or hit_idx >= max_shot_len or hit_idx >= len(path_full):
            continue

        next_cell = path_full[hit_idx]
        if next_cell in reserved:
            continue
        if any(next_cell in p["mouth_abs"] for p in placed):
            continue
        content_count = int(rng.integers(hit_idx + 1, max_shot_len + 1))
        collision_cell = path_full[hit_idx - 1]
        return _commit_collision(
            blueprint,
            r0,
            c0,
            content_count,
            path_full[: hit_idx - 1],
            collision_cell,
            next_cell,
        )
    return None


def _render_borders_and_shots(H: int, W: int, cups_meta: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    canvas_in = np.zeros((H, W), dtype=int)
    canvas_out = np.zeros((H, W), dtype=int)
    for meta in cups_meta:
        for r, c in meta["border_abs"]:
            canvas_in[r, c] = meta["border_color"]
            canvas_out[r, c] = meta["border_color"]
    for meta in cups_meta:
        for r, c in meta["shot_cells"]:
            canvas_out[r, c] = meta["content_color"]
        if meta["collision_cell"] is not None:
            rr, cc = meta["collision_cell"]
            canvas_out[rr, cc] = meta["content_color"]
    return canvas_in, canvas_out


def _scatter_content(rng: np.random.Generator, canvas_in: np.ndarray, cups_meta: list[dict]) -> bool:
    """Scatter each cup's source content into its mouth; output keeps these cells erased."""
    for meta in cups_meta:
        mouth_abs = meta["mouth_abs"]
        count = meta["content_count"]
        if len(mouth_abs) < count:
            return False
        idxs = rng.choice(len(mouth_abs), size=count, replace=False)
        for i in idxs:
            r, c = mouth_abs[int(i)]
            canvas_in[r, c] = meta["content_color"]
    return True


def _sample_colors(rng: np.random.Generator, num_cups: int) -> list[tuple[int, int]]:
    """Choose colors while keeping the forced collision recolor visible."""
    colors = list(range(1, 10))
    rng.shuffle(colors)
    border_colors = colors[:num_cups]
    pool = colors[num_cups:]
    target_content = pool[0]
    shooter_content = pool[1]
    pairs = [(border_colors[0], target_content), (border_colors[1], shooter_content)]
    for i in range(2, num_cups):
        choices = [c for c in (pool[:3] + [target_content, shooter_content]) if c != border_colors[i]]
        pairs.append((border_colors[i], int(rng.choice(choices))))
    return pairs


def _gen_attempt(rng: np.random.Generator, num_cups: int | None = None) -> np.ndarray | None:
    """Build one scene with at least one real border-hit collision, validated by `main`."""
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
    if not _scatter_content(rng, canvas_in, cups_meta):
        return None

    try:
        produced = main(canvas_in)
    except Exception:
        return None
    if not np.array_equal(produced, canvas_out):
        return None

    collision_count = sum(1 for meta in cups_meta if meta["collision_cell"] is not None)
    if collision_count < 1:
        return None
    if shooter["collision_step"] is None or shooter["content_count"] <= shooter["collision_step"]:
        return None
    rr, cc = shooter["collision_cell"]
    if canvas_out[rr, cc] != shooter["content_color"] or canvas_in[rr, cc] != target["border_color"]:
        return None
    return canvas_in


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    num_cups = int(rng.choice(NUM_CUPS_CHOICES))
    for _ in range(GEN_RETRIES_PRIMARY):
        result = _gen_attempt(rng, num_cups=num_cups)
        if result is not None:
            return result
    fallback_rounds = max(1, GEN_RETRIES_FALLBACK // len(NUM_CUPS_CHOICES))
    for forced_num_cups in NUM_CUPS_CHOICES:
        for _ in range(fallback_rounds):
            result = _gen_attempt(rng, num_cups=int(forced_num_cups))
            if result is not None:
                return result
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES_PRIMARY + GEN_RETRIES_FALLBACK} attempts")
