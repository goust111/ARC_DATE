from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
PIECE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("N", "E", "S", "W")
DIRS = {
    "N": (-1, 0),
    "E": (0, 1),
    "S": (1, 0),
    "W": (0, -1),
}
OPPOSITE = {"N": "S", "E": "W", "S": "N", "W": "E"}

GEN_RETRIES = 500
PIECE_COUNT_RANGE = (3, 5)
BODY_H_RANGE = (4, 7)
BODY_W_RANGE = (4, 7)
GRID_H_RANGE = (24, 30)
GRID_W_RANGE = (24, 30)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((int(rr), int(cc)))
                for dr, dc in DIRS.values():
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(cells, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _component_info(grid: np.ndarray, coords: np.ndarray) -> dict:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()
    rel = coords - np.array([r0, c0], dtype=int)
    mask = np.zeros(patch.shape, dtype=bool)
    mask[rel[:, 0], rel[:, 1]] = True
    colors = sorted(set(map(int, patch[mask])))
    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "patch": patch,
        "mask": mask,
        "colors": colors,
    }


def _legend_order(patch: np.ndarray) -> list[int]:
    patch = np.asarray(patch, dtype=int)
    h, w = map(int, patch.shape)
    assert h == w and h % 2 == 1 and h >= 7, "legend must be an odd square"

    for i in range(h):
        if i % 2 == 0:
            assert np.all(patch[i, :] == GREY), "legend rows on the lattice must be grey"
            assert np.all(patch[:, i] == GREY), "legend columns on the lattice must be grey"

    n = (h - 1) // 2
    order: list[int] = []
    for i in range(n):
        color = int(patch[2 * i + 1, 2 * i + 1])
        assert color not in (BG, GREY), "legend diagonal cells must contain piece colors"
        order.append(color)

    assert len(order) == len(set(order)), "legend order must use each piece color once"

    for r in range(1, h, 2):
        for c in range(1, h, 2):
            cell = int(patch[r, c])
            rr = (r - 1) // 2
            cc = (c - 1) // 2
            if rr == cc:
                assert cell == order[rr], "legend diagonal must encode order"
            else:
                assert cell == BG, "off-diagonal legend cells must stay black"
    return order


def _feature_cell(body_h: int, body_w: int, side: str, kind: str, pos: int) -> tuple[int, int]:
    assert 1 <= pos
    if side == "N":
        assert pos <= body_w - 2
        return (-1, int(pos)) if kind == "protrusion" else (0, int(pos))
    if side == "S":
        assert pos <= body_w - 2
        return (int(body_h), int(pos)) if kind == "protrusion" else (int(body_h - 1), int(pos))
    if side == "W":
        assert pos <= body_h - 2
        return (int(pos), -1) if kind == "protrusion" else (int(pos), 0)
    assert side == "E"
    assert pos <= body_h - 2
    return (int(pos), int(body_w)) if kind == "protrusion" else (int(pos), int(body_w - 1))


def _materialize_piece(color: int, body_h: int, body_w: int, raw_features: list[dict]) -> dict:
    cells = {(r, c) for r in range(int(body_h)) for c in range(int(body_w))}
    features: list[dict] = []
    raw_cells: list[tuple[int, int]] = []

    for feat in raw_features:
        cell = _feature_cell(body_h, body_w, feat["side"], feat["kind"], int(feat["pos"]))
        raw_cells.append(cell)
        if feat["kind"] == "cavity":
            assert cell in cells, "cavity must remove a body cell"
            cells.remove(cell)
        else:
            cells.add(cell)

    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    norm_cells = sorted((int(r - min_r), int(c - min_c)) for r, c in cells)
    h = max(r for r, _c in norm_cells) + 1
    w = max(c for _r, c in norm_cells) + 1
    patch = np.full((h, w), BG, dtype=int)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in norm_cells:
        patch[r, c] = int(color)
        mask[r, c] = True

    for feat, cell in zip(raw_features, raw_cells):
        features.append(
            {
                "side": feat["side"],
                "kind": feat["kind"],
                "cell": (int(cell[0] - min_r), int(cell[1] - min_c)),
            }
        )

    assert len(_components(mask)) == 1, "piece must stay connected"
    return {
        "color": int(color),
        "patch": patch,
        "mask": mask,
        "occ": norm_cells,
        "features": features,
    }


def _side_from_outside(
    rr: int,
    cc: int,
    top: int,
    left: int,
    body_h: int,
    body_w: int,
    patch_h: int,
    patch_w: int,
) -> str | None:
    bottom = int(top + body_h - 1)
    right = int(left + body_w - 1)
    if top == 1 and rr == top - 1 and left + 1 <= cc <= right - 1:
        return "N"
    if bottom == patch_h - 2 and rr == bottom + 1 and left + 1 <= cc <= right - 1:
        return "S"
    if left == 1 and cc == left - 1 and top + 1 <= rr <= bottom - 1:
        return "W"
    if right == patch_w - 2 and cc == right + 1 and top + 1 <= rr <= bottom - 1:
        return "E"
    return None


def _side_from_missing(rr: int, cc: int, top: int, left: int, body_h: int, body_w: int) -> str | None:
    bottom = int(top + body_h - 1)
    right = int(left + body_w - 1)
    if rr == top and left + 1 <= cc <= right - 1:
        return "N"
    if rr == bottom and left + 1 <= cc <= right - 1:
        return "S"
    if cc == left and top + 1 <= rr <= bottom - 1:
        return "W"
    if cc == right and top + 1 <= rr <= bottom - 1:
        return "E"
    return None


def _infer_piece(color: int, mask: np.ndarray) -> dict:
    mask = np.asarray(mask, dtype=bool)
    h, w = map(int, mask.shape)
    best: list[dict] = []
    best_area = -1

    for top in (0, 1):
        for bottom_gap in (0, 1):
            for left in (0, 1):
                for right_gap in (0, 1):
                    body_h = int(h - top - bottom_gap)
                    body_w = int(w - left - right_gap)
                    if body_h < 3 or body_w < 3:
                        continue

                    body = np.zeros((h, w), dtype=bool)
                    body[top : top + body_h, left : left + body_w] = True
                    outside = np.argwhere(mask & ~body)
                    missing = np.argwhere(body & ~mask)
                    side_seen: set[str] = set()
                    feats: list[dict] = []
                    valid = True

                    for rr, cc in outside:
                        side = _side_from_outside(
                            int(rr),
                            int(cc),
                            int(top),
                            int(left),
                            int(body_h),
                            int(body_w),
                            int(h),
                            int(w),
                        )
                        if side is None or side in side_seen:
                            valid = False
                            break
                        side_seen.add(side)
                        feats.append({"side": side, "kind": "protrusion", "cell": (int(rr), int(cc))})
                    if not valid:
                        continue

                    for rr, cc in missing:
                        side = _side_from_missing(int(rr), int(cc), int(top), int(left), int(body_h), int(body_w))
                        if side is None or side in side_seen:
                            valid = False
                            break
                        side_seen.add(side)
                        feats.append({"side": side, "kind": "cavity", "cell": (int(rr), int(cc))})
                    if not valid or not (1 <= len(feats) <= 2):
                        continue

                    area = int(body_h * body_w)
                    cand = {
                        "color": int(color),
                        "mask": mask.copy(),
                        "occ": [tuple(map(int, rc)) for rc in np.argwhere(mask)],
                        "features": feats,
                    }
                    if area > best_area:
                        best_area = area
                        best = [cand]
                    elif area == best_area:
                        best.append(cand)

    assert len(best) == 1, "piece body and connectors must be uniquely recoverable"
    return best[0]


def _complementary(a: dict, b: dict) -> bool:
    return a["kind"] != b["kind"] and a["side"] == OPPOSITE[b["side"]]


def _place_piece(
    piece: dict,
    translation: tuple[int, int],
    occupied: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    tr, tc = map(int, translation)
    cells = {(int(r + tr), int(c + tc)) for r, c in piece["occ"]}
    if cells & occupied:
        return None
    return cells


def _assemble(ordered_pieces: list[dict]) -> list[dict]:
    assert ordered_pieces, "need at least one piece"
    placed = [{"piece": ordered_pieces[0], "shift": (0, 0)}]
    occupied = {(int(r), int(c)) for r, c in ordered_pieces[0]["occ"]}
    unmatched = [
        {
            "piece_index": 0,
            "feature_index": i,
            "side": feat["side"],
            "kind": feat["kind"],
            "cell": tuple(map(int, feat["cell"])),
        }
        for i, feat in enumerate(ordered_pieces[0]["features"])
    ]

    for piece_index, piece in enumerate(ordered_pieces[1:], start=1):
        candidate_shifts: set[tuple[int, int]] = set()
        for existing in unmatched:
            for feat in piece["features"]:
                if _complementary(existing, feat):
                    candidate_shifts.add(
                        (
                            int(existing["cell"][0] - feat["cell"][0]),
                            int(existing["cell"][1] - feat["cell"][1]),
                        )
                    )

        choices = []
        for shift in sorted(candidate_shifts):
            cells = _place_piece(piece, shift, occupied)
            if cells is None:
                continue

            matched_existing: set[int] = set()
            matched_new: set[int] = set()
            for ei, existing in enumerate(unmatched):
                for ni, feat in enumerate(piece["features"]):
                    world = (int(feat["cell"][0] + shift[0]), int(feat["cell"][1] + shift[1]))
                    if world == existing["cell"] and _complementary(existing, feat):
                        matched_existing.add(int(ei))
                        matched_new.add(int(ni))

            if matched_existing:
                choices.append(
                    {
                        "shift": shift,
                        "cells": cells,
                        "matched_existing": matched_existing,
                        "matched_new": matched_new,
                    }
                )

        assert len(choices) == 1, "order and lock geometry must determine one placement"
        chosen = choices[0]
        placed.append({"piece": piece, "shift": chosen["shift"]})
        occupied |= chosen["cells"]

        unmatched = [
            item
            for idx, item in enumerate(unmatched)
            if idx not in chosen["matched_existing"]
        ]
        for ni, feat in enumerate(piece["features"]):
            if ni in chosen["matched_new"]:
                continue
            unmatched.append(
                {
                    "piece_index": int(piece_index),
                    "feature_index": int(ni),
                    "side": feat["side"],
                    "kind": feat["kind"],
                    "cell": (
                        int(feat["cell"][0] + chosen["shift"][0]),
                        int(feat["cell"][1] + chosen["shift"][1]),
                    ),
                }
            )

    return placed


def _render_assembly(placed: list[dict]) -> np.ndarray:
    all_cells: list[tuple[int, int]] = []
    for item in placed:
        tr, tc = map(int, item["shift"])
        for r, c in item["piece"]["occ"]:
            all_cells.append((int(r + tr), int(c + tc)))

    rows = [r for r, _c in all_cells]
    cols = [c for _r, c in all_cells]
    min_r = min(rows)
    max_r = max(rows)
    min_c = min(cols)
    max_c = max(cols)

    out = np.full((max_r - min_r + 1, max_c - min_c + 1), BG, dtype=int)
    for item in placed:
        tr, tc = map(int, item["shift"])
        color = int(item["piece"]["color"])
        for r, c in item["piece"]["occ"]:
            out[int(r + tr - min_r), int(c + tc - min_c)] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the grey order legend, recover each piece's 1x1 protrusions/cavities,
    then assemble the pieces in that order by the unique complementary lock.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    comps = [_component_info(g, coords) for coords in _components(g != BG)]
    assert len(comps) >= 4, "need a legend and at least three pieces"

    legends = [comp for comp in comps if GREY in comp["colors"]]
    assert len(legends) == 1, "expected exactly one grey legend"
    legend = legends[0]
    order = _legend_order(legend["patch"])

    pieces_by_color: dict[int, dict] = {}
    for comp in comps:
        if comp is legend:
            continue
        colors = [color for color in comp["colors"] if color != BG]
        assert len(colors) == 1, "each piece must be monochromatic"
        color = int(colors[0])
        assert color not in pieces_by_color, "piece colors must be unique"
        pieces_by_color[color] = _infer_piece(color, comp["mask"])

    assert set(order) == set(pieces_by_color), "legend colors must match the piece set"
    ordered_pieces = [pieces_by_color[int(color)] for color in order]
    placed = _assemble(ordered_pieces)
    return _render_assembly(placed)


def _random_side_pos(side: str, body_h: int, body_w: int) -> int:
    if side in ("N", "S"):
        return random.randint(1, int(body_w - 2))
    return random.randint(1, int(body_h - 2))


def _make_legend(order: list[int]) -> np.ndarray:
    n = len(order)
    size = 2 * n + 1
    patch = np.full((size, size), BG, dtype=int)
    patch[::2, :] = GREY
    patch[:, ::2] = GREY
    for i, color in enumerate(order):
        patch[2 * i + 1, 2 * i + 1] = int(color)
    return patch


def _direction_chain(n_pieces: int) -> list[str]:
    dirs = [random.choice(SIDES)]
    while len(dirs) < n_pieces - 1:
        prev = dirs[-1]
        options = [side for side in SIDES if side != OPPOSITE[prev]]
        dirs.append(random.choice(options))
    return dirs


def _specs_from_chain(order: list[int], directions: list[str], link_modes: list[bool]) -> list[dict] | None:
    n = len(order)
    specs = [
        {
            "color": int(order[i]),
            "body_h": random.randint(*BODY_H_RANGE),
            "body_w": random.randint(*BODY_W_RANGE),
            "features": [],
        }
        for i in range(n)
    ]

    used_sides = [set() for _ in range(n)]
    for i, direction in enumerate(directions):
        prev_side = str(direction)
        next_side = str(OPPOSITE[direction])
        if prev_side in used_sides[i] or next_side in used_sides[i + 1]:
            return None
        used_sides[i].add(prev_side)
        used_sides[i + 1].add(next_side)

        prev_kind = "protrusion" if link_modes[i] else "cavity"
        next_kind = "cavity" if link_modes[i] else "protrusion"
        specs[i]["features"].append(
            {
                "side": prev_side,
                "kind": prev_kind,
                "pos": _random_side_pos(prev_side, specs[i]["body_h"], specs[i]["body_w"]),
            }
        )
        specs[i + 1]["features"].append(
            {
                "side": next_side,
                "kind": next_kind,
                "pos": _random_side_pos(next_side, specs[i + 1]["body_h"], specs[i + 1]["body_w"]),
            }
        )
    return specs


def _scatter_on_canvas(patches: list[np.ndarray], legend: np.ndarray) -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    lh, lw = map(int, legend.shape)
    corners = {
        "tl": (0, 0),
        "tr": (0, w - lw),
        "bl": (h - lh, 0),
        "br": (h - lh, w - lw),
    }
    corner = random.choice(sorted(corners))
    lr, lc = corners[corner]
    if lr < 0 or lc < 0:
        return None

    grid[lr : lr + lh, lc : lc + lw] = legend
    rr0 = max(0, lr - 1)
    rr1 = min(h, lr + lh + 1)
    cc0 = max(0, lc - 1)
    cc1 = min(w, lc + lw + 1)
    blocked[rr0:rr1, cc0:cc1] = True

    order = list(range(len(patches)))
    random.shuffle(order)
    for idx in order:
        patch = patches[idx]
        ph, pw = map(int, patch.shape)
        spots = [
            (r, c)
            for r in range(h - ph + 1)
            for c in range(w - pw + 1)
            if not bool(blocked[r : r + ph, c : c + pw].any())
            and not bool(blocked[max(0, r - 1) : min(h, r + ph + 1), max(0, c - 1) : min(w, c + pw + 1)].any())
        ]
        if not spots:
            return None
        r, c = random.choice(spots)
        mask = patch != BG
        grid[r : r + ph, c : c + pw][mask] = patch[mask]
        blocked[max(0, r - 1) : min(h, r + ph + 1), max(0, c - 1) : min(w, c + pw + 1)] = True
    return grid


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    n = random.randint(*PIECE_COUNT_RANGE)
    colors = random.sample(PIECE_COLORS, n)
    order = random.sample(colors, len(colors))
    directions = _direction_chain(n)
    link_modes = [bool(random.getrandbits(1)) for _ in range(n - 1)]
    specs = _specs_from_chain(order, directions, link_modes)
    if specs is None:
        return None

    try:
        pieces = [
            _materialize_piece(spec["color"], spec["body_h"], spec["body_w"], spec["features"])
            for spec in specs
        ]
    except AssertionError:
        return None
    try:
        assembled = _assemble(pieces)
    except AssertionError:
        return None
    expected = _render_assembly(assembled)

    legend = _make_legend(order)
    grid = _scatter_on_canvas([piece["patch"] for piece in pieces], legend)
    if grid is None:
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        case = _generate_case()
        if case is None:
            continue
        grid, expected = case
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise AssertionError("failed to generate a valid puzzle instance")
