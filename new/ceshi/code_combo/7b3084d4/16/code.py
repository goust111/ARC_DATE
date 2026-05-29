# Combo variant of 7b3084d4 (index 16)
# combo_spec: D1/2+D7/1
# Name: 连接器标记匹配+标记指示拼接边

import math
import random
from collections import Counter, deque

import numpy as np

from arc_utils import get_bg as _bg_color, is_contiguous, neighbors8, random_free_location


BG = 0
ANCHOR_COLOR = 7
MARKER_COLOR = 5
CONNECTOR_POOL = [6, 8, 9]
BODY_POOL = [1, 2, 3, 4]
SPECIAL_COLORS = {MARKER_COLOR, *CONNECTOR_POOL}

CANVAS_H = 20
CANVAS_W = 20
PLACEMENT_PADDING = 1
MAX_ATTEMPTS = 500
N_MIN = 4
N_MAX = 6

TL, TR, BL, BR = 0, 1, 2, 3
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}


# Design analysis
# a) Concrete behavior trace:
#    In the original task, one grey corner pixel directly chose the anchor and
#    the solver could ignore all local edge semantics; any packing that filled a
#    rectangle and moved the singleton to (0,0) was acceptable.
#    In this fusion, each piece is extracted as an 8-connected multicolor
#    component.  The assembled rectangle still comes from the original 2x2 square
#    skeleton, but the evidence layer changes:
#    - One connector color links the anchor block to both middle blocks.
#    - A second connector color links the two middle blocks to the bottom-right
#      block.
#    - Exactly one middle block also carries a grey edge pixel on the same side
#      that must face the anchor.
#    Example: if the marked middle piece is rotated so its grey pixel is on its
#    left edge, then that piece must be the top-right piece; if the grey pixel is
#    on its top edge, it must be the bottom-left piece.  Connector colors alone
#    identify the graph, but they do not distinguish those two middle roles.
# b) Complexity calibration:
#    Only two connector colors are used, sampled per instance and each repeated
#    across two interfaces.
#    This keeps the pattern visually obvious while avoiding the degenerate case
#    where four unique connector colors would make the grey marker unnecessary.
#    The output geometry stays close to the original task, but the internal
#    evidence is now mostly symbolic rather than geometric.
# c) Broken original assumptions:
#    - Color-based extraction no longer works because every piece can contain
#      body color plus connector/marker colors.
#    - The singleton grey pixel is no longer a corner anchor, so placing it at
#      output (0,0) would be wrong.
#    - Free-form rectangle packing is now underconstrained unless connector-side
#      directions are checked explicitly.
# d) Placement legality:
#    Generator pieces are still scattered with a one-cell gap on a 20x20 canvas.
#    Connector and marker pixels replace existing piece pixels, so there is no
#    overlap or out-of-bounds risk inside a piece.  Output-first generation keeps
#    the final rectangle valid by construction.
# e) Visual observability:
#    Humans can see that two connector colors recur, that the same-colored edge
#    pixels indicate docking, and that the single grey pixel sits on one of those
#    docking sides.  Across examples the grey pixel flips between a left edge and
#    a top edge, making the role-disambiguation visible.
# f) Visual cleanliness:
#    Only one connector pixel is used per interface and the grey marker is placed
#    on a different pixel along the same anchor-facing side.  That avoids stacked
#    annotations and keeps each boundary legible.
# g) New solver assumptions and generator guarantees:
#    - Exactly four 8-connected components are present.
#    - The anchor is the unique piece whose body color is 7.
#    - The bottom-right piece has two copies of one connector color and no
#      marker.
#    - The marked piece has one copy of each connector color, and the grey pixel
#      shares a side with the connector that points toward the anchor.
#    Generator construction enforces each of these conditions before oracle
#    verification with main().


def _normalize(cells: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    min_r = min(r for r, _, _ in cells)
    min_c = min(c for _, c, _ in cells)
    out = [(r - min_r, c - min_c, col) for r, c, col in cells]
    out.sort()
    return out


def _rotate_cells(cells: list[tuple[int, int, int]], k: int) -> list[tuple[int, int, int]]:
    pts = list(cells)
    for _ in range(k % 4):
        pts = [(c, -r, col) for r, c, col in pts]
    return _normalize(pts)


def _variants(cells: list[tuple[int, int, int]]) -> list[list[tuple[int, int, int]]]:
    seen: set[tuple[tuple[int, int, int], ...]] = set()
    outs: list[list[tuple[int, int, int]]] = []
    for k in range(4):
        rot = _rotate_cells(cells, k)
        key = tuple(rot)
        if key not in seen:
            seen.add(key)
            outs.append(rot)
    return outs


def _cells_to_patch(cells: list[tuple[int, int, int]]) -> np.ndarray:
    h = max(r for r, _, _ in cells) + 1
    w = max(c for _, c, _ in cells) + 1
    patch = np.zeros((h, w), dtype=int)
    for r, c, col in cells:
        patch[r, c] = col
    return patch


def _components(grid: np.ndarray) -> tuple[list[list[tuple[int, int, int]]], int]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int, int]]] = []
    for sr, sc in np.argwhere(g != bg):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        cells: list[tuple[int, int, int]] = []
        while q:
            r, c = q.popleft()
            cells.append((int(r), int(c), int(g[r, c])))
            for nr, nc in neighbors8(r, c):
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and g[nr, nc] != bg:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(_normalize(cells))
    return comps, bg


def _body_color(cells: list[tuple[int, int, int]]) -> int:
    counts = Counter(col for _, _, col in cells if col not in SPECIAL_COLORS)
    assert counts, "piece must contain at least one body-colored pixel"
    return int(counts.most_common(1)[0][0])


def _edge_dir(mask: set[tuple[int, int]], pt: tuple[int, int]) -> str | None:
    free: list[str] = []
    r, c = pt
    for name, (dr, dc) in DIRS.items():
        if (r + dr, c + dc) not in mask:
            free.append(name)
    if len(free) == 1:
        return free[0]
    return None


def _analyze_piece(cells: list[tuple[int, int, int]]) -> dict:
    mask = {(r, c) for r, c, _ in cells}
    by_color: dict[int, list[dict]] = {}
    for r, c, col in cells:
        if col in SPECIAL_COLORS:
            side = _edge_dir(mask, (r, c))
            assert side is not None, "special pixels must be on a unique non-corner edge"
            by_color.setdefault(int(col), []).append({"pos": (int(r), int(c)), "dir": side})
    return {
        "cells": cells,
        "mask": mask,
        "body_color": _body_color(cells),
        "special": by_color,
    }


def _shift_cells(cells: list[tuple[int, int, int]], dr: int, dc: int) -> list[tuple[int, int, int]]:
    return [(r + dr, c + dc, col) for r, c, col in cells]


def _render_solution(
    placements: dict[str, tuple[int, int, list[tuple[int, int, int]]]], size: int
) -> tuple[np.ndarray, np.ndarray] | None:
    board = np.full((size, size), -1, dtype=int)
    owner = np.full((size, size), -1, dtype=int)
    role_order = {"anchor": 0, "tr": 1, "bl": 2, "br": 3}
    for role, (_, _, cells) in placements.items():
        idx = role_order[role]
        for r, c, col in cells:
            if not (0 <= r < size and 0 <= c < size):
                return None
            if board[r, c] != -1:
                return None
            board[r, c] = col
            owner[r, c] = idx
    if np.any(board == -1):
        return None
    return board.astype(int), owner.astype(int)


def _count_cross_adjacencies(board: np.ndarray, owner: np.ndarray, color: int) -> int:
    h, w = board.shape
    total = 0
    for r in range(h):
        for c in range(w):
            if int(board[r, c]) != color:
                continue
            for dr, dc in ((1, 0), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and int(board[nr, nc]) == color and int(owner[nr, nc]) != int(owner[r, c]):
                    total += 1
    return total


def _expect_single_special(info: dict, color: int, wanted_dir: str) -> tuple[int, int] | None:
    pts = info["special"].get(color, [])
    hits = [tuple(p["pos"]) for p in pts if p["dir"] == wanted_dir]
    if len(hits) == 1 and len(pts) == 1:
        return hits[0]
    return None


def _expect_double_special(info: dict, color: int, wanted_dirs: set[str]) -> dict[str, tuple[int, int]] | None:
    pts = info["special"].get(color, [])
    if len(pts) != 2:
        return None
    got = {p["dir"] for p in pts}
    if got != wanted_dirs:
        return None
    return {p["dir"]: tuple(p["pos"]) for p in pts}


def _solve_from_roles(roles: dict) -> np.ndarray | None:
    anchor_conn_color = int(roles["anchor_conn_color"])
    br_conn_color = int(roles["br_conn_color"])
    pieces = {name: roles[name] for name in ("anchor", "marked", "other_mid", "br")}
    total_area = sum(len(cells) for cells in pieces.values())
    size = int(math.isqrt(total_area))
    if size * size != total_area:
        return None

    anchor_variants = [_analyze_piece(v) for v in _variants(pieces["anchor"])]
    marked_variants = [_analyze_piece(v) for v in _variants(pieces["marked"])]
    other_variants = [_analyze_piece(v) for v in _variants(pieces["other_mid"])]
    br_variants = [_analyze_piece(v) for v in _variants(pieces["br"])]

    for ainfo in anchor_variants:
        anchor_conn = _expect_double_special(ainfo, anchor_conn_color, {"right", "down"})
        if anchor_conn is None:
            continue
        anchor_cells = _shift_cells(ainfo["cells"], 0, 0)

        for minfo in marked_variants:
            marker_pts = minfo["special"].get(MARKER_COLOR, [])
            if len(marker_pts) != 1:
                continue
            marker_dir = marker_pts[0]["dir"]
            if marker_dir not in {"left", "up"}:
                continue
            marked_anchor = _expect_single_special(minfo, anchor_conn_color, marker_dir)
            if marked_anchor is None:
                continue
            marked_role = "tr" if marker_dir == "left" else "bl"
            marked_8_dir = "down" if marked_role == "tr" else "right"
            marked_br = _expect_single_special(minfo, br_conn_color, marked_8_dir)
            if marked_br is None:
                continue

            anchor_target = anchor_conn["right"] if marked_role == "tr" else anchor_conn["down"]
            mr, mc = marked_anchor
            ar, ac = anchor_target
            dr = ar - mr + (0 if marked_role == "tr" else 1)
            dc = ac - mc + (1 if marked_role == "tr" else 0)
            marked_cells = _shift_cells(minfo["cells"], dr, dc)

            other_role = "bl" if marked_role == "tr" else "tr"
            other_anchor_dir = "up" if other_role == "bl" else "left"
            other_br_dir = "right" if other_role == "bl" else "down"

            for oinfo in other_variants:
                other_anchor = _expect_single_special(oinfo, anchor_conn_color, other_anchor_dir)
                other_br = _expect_single_special(oinfo, br_conn_color, other_br_dir)
                if other_anchor is None or other_br is None:
                    continue

                anchor_target = anchor_conn["down"] if other_role == "bl" else anchor_conn["right"]
                or0, oc0 = other_anchor
                ar0, ac0 = anchor_target
                dr2 = ar0 - or0 + (1 if other_role == "bl" else 0)
                dc2 = ac0 - oc0 + (0 if other_role == "bl" else 1)
                other_cells = _shift_cells(oinfo["cells"], dr2, dc2)

                tr_info = minfo if marked_role == "tr" else oinfo
                bl_info = minfo if marked_role == "bl" else oinfo
                tr_shift = (dr, dc) if marked_role == "tr" else (dr2, dc2)
                bl_shift = (dr, dc) if marked_role == "bl" else (dr2, dc2)
                tr_br = _expect_single_special(tr_info, br_conn_color, "down")
                bl_br = _expect_single_special(bl_info, br_conn_color, "right")
                assert tr_br is not None and bl_br is not None
                tr_abs = (tr_br[0] + tr_shift[0], tr_br[1] + tr_shift[1])
                bl_abs = (bl_br[0] + bl_shift[0], bl_br[1] + bl_shift[1])

                for binfo in br_variants:
                    br_conn = _expect_double_special(binfo, br_conn_color, {"up", "left"})
                    if br_conn is None:
                        continue
                    top_pt = br_conn["up"]
                    left_pt = br_conn["left"]
                    br_shift_from_tr = (tr_abs[0] + 1 - top_pt[0], tr_abs[1] - top_pt[1])
                    br_shift_from_bl = (bl_abs[0] - left_pt[0], bl_abs[1] + 1 - left_pt[1])
                    if br_shift_from_tr != br_shift_from_bl:
                        continue
                    br_cells = _shift_cells(binfo["cells"], *br_shift_from_tr)

                    placed = {
                        "anchor": (0, 0, anchor_cells),
                        "tr": (*tr_shift, _shift_cells(tr_info["cells"], *tr_shift)),
                        "bl": (*bl_shift, _shift_cells(bl_info["cells"], *bl_shift)),
                        "br": (*br_shift_from_tr, br_cells),
                    }
                    rendered = _render_solution(placed, size)
                    if rendered is None:
                        continue
                    board, owner = rendered
                    if _count_cross_adjacencies(board, owner, anchor_conn_color) != 2:
                        continue
                    if _count_cross_adjacencies(board, owner, br_conn_color) != 2:
                        continue

                    marker_piece = placed[marked_role][2]
                    marker_variant = minfo
                    marker_pos = marker_variant["special"][MARKER_COLOR][0]["pos"]
                    marker_dir_final = marker_variant["special"][MARKER_COLOR][0]["dir"]
                    shift = tr_shift if marked_role == "tr" else bl_shift
                    abs_marker = (marker_pos[0] + shift[0], marker_pos[1] + shift[1])
                    mdr, mdc = DIRS[marker_dir_final]
                    nr, nc = abs_marker[0] + mdr, abs_marker[1] + mdc
                    if not (0 <= nr < size and 0 <= nc < size):
                        continue
                    if int(owner[nr, nc]) != 0:
                        continue

                    return board
    return None


def _connector_signature(info: dict) -> dict[int, int]:
    return {int(color): len(pts) for color, pts in info["special"].items() if color != MARKER_COLOR}


def _role_partition(comps: list[list[tuple[int, int, int]]]) -> dict[str, list[tuple[int, int, int]]] | None:
    if len(comps) != 4:
        return None

    infos = [_analyze_piece(cells) for cells in comps]
    anchor_idx = next((i for i, info in enumerate(infos) if info["body_color"] == ANCHOR_COLOR), None)
    if anchor_idx is None:
        return None

    marked_idx = next((i for i, info in enumerate(infos) if MARKER_COLOR in info["special"]), None)
    if marked_idx is None or marked_idx == anchor_idx:
        return None

    marked_info = infos[marked_idx]
    marker_pts = marked_info["special"].get(MARKER_COLOR, [])
    if len(marker_pts) != 1:
        return None
    marker_dir = marker_pts[0]["dir"]
    marked_connectors = _connector_signature(marked_info)
    if len(marked_connectors) != 2 or set(marked_connectors.values()) != {1}:
        return None
    anchor_conn_color = next(
        (color for color in marked_connectors if marked_info["special"][color][0]["dir"] == marker_dir),
        None,
    )
    if anchor_conn_color is None:
        return None
    br_conn_color = next(color for color in marked_connectors if color != anchor_conn_color)

    br_idx = None
    other_mid_idx = None
    for i, info in enumerate(infos):
        if i in {anchor_idx, marked_idx}:
            continue
        sig = _connector_signature(info)
        if sig == {br_conn_color: 2}:
            br_idx = i
        elif sig == {anchor_conn_color: 1, br_conn_color: 1}:
            other_mid_idx = i
    if br_idx is None or other_mid_idx is None:
        return None

    anchor_info = infos[anchor_idx]
    if _connector_signature(anchor_info) != {anchor_conn_color: 2} or MARKER_COLOR in anchor_info["special"]:
        return None

    if _connector_signature(marked_info) != {anchor_conn_color: 1, br_conn_color: 1}:
        return None

    return {
        "anchor": comps[anchor_idx],
        "marked": comps[marked_idx],
        "other_mid": comps[other_mid_idx],
        "br": comps[br_idx],
        "anchor_conn_color": anchor_conn_color,
        "br_conn_color": br_conn_color,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble four scattered pieces into the canonical square by matching repeated connector colors and a grey edge cue.

    Input structure:
    - Background is the dominant color.  There are exactly four separated 8-connected pieces.
    - Each piece is mostly one body color, but edge pixels may be recolored as connector marks.
    - The anchor piece keeps body color 7.  Two sampled connector colors are reused across matched interfaces.
    - Exactly one middle piece also has a single grey pixel on an edge.  That grey pixel is not a corner anchor; it marks which side of that piece must face the anchor.

    Rule:
    - Extract the four multicolor components and rotate them.
    - Use connector-side directions to assign roles: anchor in the top-left, one middle piece on the top-right, one on the bottom-left, and the remaining piece in the bottom-right.
    - The connector color that shares the grey marker's side is the anchor-facing connector; the other connector color must face the bottom-right piece.
    - Same-colored connectors must meet across piece boundaries, and the grey-marked edge must specifically touch the anchor.
    - Keep all colors unchanged in the output and return the fully assembled square.

    Key insight:
    - Connector colors recover the adjacency graph, but the two middle pieces are still symmetric until the grey edge marker says which side of one middle piece is the anchor-facing side.
    """
    comps, _bg = _components(np.asarray(grid, dtype=int))
    roles = _role_partition(comps)
    assert roles is not None, "input must contain the four-role connector/marker pattern"
    solved = _solve_from_roles(roles)
    assert solved is not None, "no assembly satisfies the connector graph and marker-directed edge"
    return solved.astype(int)


def _build_ownership(N: int, p: int, tab_specs: dict[str, tuple[int, str] | None]) -> np.ndarray | None:
    S = 2 * N
    own = np.zeros((S, S), dtype=int)
    own[:N, :N] = TL
    own[:N, N:] = TR
    own[N:, :N] = BL
    own[N:, N:] = BR

    for kind, col, up_owner, down_owner in [
        ("H_left", p, BL, TL),
        ("H_right", S - 1 - p, BR, TR),
    ]:
        spec = tab_specs.get(kind)
        if spec is None:
            continue
        depth, direction = spec
        if direction == "up":
            own[N - depth : N, col] = up_owner
        else:
            own[N : N + depth, col] = down_owner

    for kind, row, left_owner, right_owner in [
        ("V_top", p, TR, TL),
        ("V_bot", S - 1 - p, BR, BL),
    ]:
        spec = tab_specs.get(kind)
        if spec is None:
            continue
        depth, direction = spec
        if direction == "left":
            own[row, N - depth : N] = left_owner
        else:
            own[row, N : N + depth] = right_owner

    for q in (TL, TR, BL, BR):
        if not is_contiguous((own == q).astype(int), background=0, connectivity=8):
            return None
    return own


def _random_tab_specs(N: int) -> dict[str, tuple[int, str] | None]:
    return {
        "H_left": (random.randint(1, N - 1), random.choice(["up", "down"])),
        "H_right": (random.randint(1, N - 1), random.choice(["up", "down"])),
        "V_top": (random.randint(1, N - 1), random.choice(["left", "right"])),
        "V_bot": (random.randint(1, N - 1), random.choice(["left", "right"])),
    }


def _label_edge_dirs(own: np.ndarray, label: int, pt: tuple[int, int]) -> list[str]:
    dirs: list[str] = []
    r, c = pt
    h, w = own.shape
    for name, (dr, dc) in DIRS.items():
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w) or int(own[nr, nc]) != label:
            dirs.append(name)
    return dirs


def _safe_interface_pairs(own: np.ndarray, a: int, b: int, wanted_dir: str) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    h, w = own.shape
    dr, dc = DIRS[wanted_dir]
    out: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if int(own[r, c]) != a:
                continue
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if int(own[nr, nc]) != b:
                continue
            dirs_a = _label_edge_dirs(own, a, (r, c))
            dirs_b = _label_edge_dirs(own, b, (nr, nc))
            if dirs_a == [wanted_dir] and dirs_b == [OPPOSITE[wanted_dir]]:
                out.append(((r, c), (nr, nc)))
    return out


def _pick_distinct(seq: list[tuple[tuple[int, int], tuple[int, int]]], k: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    assert len(seq) >= k
    items = list(seq)
    random.shuffle(items)
    return items[:k]


def _extract_piece_cells(output: np.ndarray, own: np.ndarray, label: int) -> list[tuple[int, int, int]]:
    rows, cols = np.where(own == label)
    min_r = int(rows.min())
    min_c = int(cols.min())
    cells: list[tuple[int, int, int]] = []
    for r, c in zip(rows, cols):
        r = int(r)
        c = int(c)
        cells.append((r - min_r, c - min_c, int(output[r, c])))
    return _normalize(cells)


def _scatter_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    canvas = np.zeros((CANVAS_H, CANVAS_W), dtype=int)
    order = list(range(len(patches)))
    random.shuffle(order)
    for idx in order:
        patch = patches[idx]
        try:
            r0, c0 = random_free_location(
                canvas,
                patch,
                transparent=0,
                padding=PLACEMENT_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        region = canvas[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
        mask = patch != 0
        region[mask] = patch[mask]
    return canvas


def _build_output_and_pieces() -> tuple[np.ndarray, list[list[tuple[int, int, int]]]] | None:
    N = random.randint(N_MIN, N_MAX)
    own = np.zeros((2 * N, 2 * N), dtype=int)
    own[:N, :N] = TL
    own[:N, N:] = TR
    own[N:, :N] = BL
    own[N:, N:] = BR

    mids = BODY_POOL.copy()
    random.shuffle(mids)
    colors = {TL: ANCHOR_COLOR, TR: mids[0], BL: mids[1], BR: mids[2]}
    output = np.zeros_like(own, dtype=int)
    for label, col in colors.items():
        output[own == label] = int(col)
    anchor_conn_color, br_conn_color = random.sample(CONNECTOR_POOL, 2)

    anchor_offset = random.randint(1, N - 2)
    br_offset = random.randint(1, N - 2)
    marker_choices = [x for x in range(1, N - 1) if x != anchor_offset]
    if not marker_choices:
        return None
    marker_offset = random.choice(marker_choices)

    output[anchor_offset, N - 1] = anchor_conn_color
    output[anchor_offset, N] = anchor_conn_color
    output[N - 1, anchor_offset] = anchor_conn_color
    output[N, anchor_offset] = anchor_conn_color

    output[N - 1, N + br_offset] = br_conn_color
    output[N, N + br_offset] = br_conn_color
    output[N + br_offset, N - 1] = br_conn_color
    output[N + br_offset, N] = br_conn_color

    marked_role = random.choice([TR, BL])
    if marked_role == TR:
        output[marker_offset, N] = MARKER_COLOR
    else:
        output[N, marker_offset] = MARKER_COLOR

    pieces = [_extract_piece_cells(output, own, label) for label in (TL, TR, BL, BR)]
    return output.astype(int), pieces


def _gen_attempt() -> np.ndarray | None:
    built = _build_output_and_pieces()
    if built is None:
        return None
    expected, pieces = built

    rotated = [_rotate_cells(cells, random.randint(0, 3)) for cells in pieces]
    canvas = _scatter_patches([_cells_to_patch(cells) for cells in rotated])
    if canvas is None:
        return None

    try:
        out = main(canvas)
    except Exception:
        return None
    if not np.array_equal(out, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: exhausted attempts")
