from __future__ import annotations

import random
from collections import defaultdict
from importlib.machinery import SourceFileLoader

import numpy as np


_ARC = SourceFileLoader(
    "arc_utils_shared_for_10",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()

Color = _ARC.Color
bbox_coords = _ARC.bbox_coords
colors_present = _ARC.colors_present
find_components = _ARC.find_components


BLACK = int(Color.BLACK)
GREY = int(Color.GREY)

SHAPE_OFFSETS = {
    "solid": {(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)},
    "x": {(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)},
    "diamond": {(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)},
}
NEXT_SHAPE = {"solid": "x", "x": "diamond", "diamond": "solid"}

TARGET_LAYOUTS = {
    "tl": {"h_side": "left", "v_side": "top", "sources": ("tr", "bl")},
    "tr": {"h_side": "right", "v_side": "top", "sources": ("tl", "br")},
    "bl": {"h_side": "left", "v_side": "bottom", "sources": ("tl", "br")},
    "br": {"h_side": "right", "v_side": "bottom", "sources": ("tr", "bl")},
}

GEN_ATTEMPTS = 400
SIZE_RANGE = (5, 10)
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]


def _quadrant_specs(shape: tuple[int, int], sep_r: int, sep_c: int) -> dict[str, dict[str, tuple[int, int]]]:
    h, w = map(int, shape)
    specs = {
        "tl": {"rows": (0, sep_r - 1), "cols": (0, sep_c - 1)},
        "tr": {"rows": (0, sep_r - 1), "cols": (sep_c + 1, w - 1)},
        "bl": {"rows": (sep_r + 1, h - 1), "cols": (0, sep_c - 1)},
        "br": {"rows": (sep_r + 1, h - 1), "cols": (sep_c + 1, w - 1)},
    }
    for key, spec in specs.items():
        r0, r1 = spec["rows"]
        c0, c1 = spec["cols"]
        specs[key]["origin"] = (r0, c0)
        specs[key]["center"] = ((r0 + r1) // 2, (c0 + c1) // 2)
    return specs


def _gap_edges(sep_r: int, sep_c: int, h_gap_c: int, v_gap_r: int) -> list[tuple[str, str]]:
    edges = []
    if h_gap_c < sep_c:
        edges.append(("tl", "bl"))
    else:
        edges.append(("tr", "br"))
    if v_gap_r < sep_r:
        edges.append(("tl", "tr"))
    else:
        edges.append(("bl", "br"))
    return edges


def _normalize_offsets(coords: np.ndarray) -> set[tuple[int, int]]:
    r0, c0, _, _ = bbox_coords(coords)
    return {(int(r - r0), int(c - c0)) for r, c in coords}


def _classify_shape(coords: np.ndarray) -> str:
    normalized = _normalize_offsets(coords)
    for name, offsets in SHAPE_OFFSETS.items():
        shifted = {(r + 1, c + 1) for r, c in offsets}
        if normalized == shifted:
            return name
    raise AssertionError("unknown token shape")


def _draw_shape(grid: np.ndarray, center: tuple[int, int], color: int, shape_name: str) -> None:
    cr, cc = map(int, center)
    for dr, dc in SHAPE_OFFSETS[shape_name]:
        grid[cr + dr, cc + dc] = int(color)


def _find_separator(grid: np.ndarray) -> tuple[int, int, int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    row_counts = np.sum(g == GREY, axis=1)
    col_counts = np.sum(g == GREY, axis=0)

    sep_r = int(np.argmax(row_counts))
    sep_c = int(np.argmax(col_counts))
    assert int(row_counts[sep_r]) == w - 1, "expected one horizontal gap in the separator row"
    assert int(col_counts[sep_c]) == h - 1, "expected one vertical gap in the separator column"
    assert int(g[sep_r, sep_c]) == GREY, "separator intersection must stay grey"

    row_gaps = [c for c in range(w) if int(g[sep_r, c]) == BLACK and c != sep_c]
    col_gaps = [r for r in range(h) if int(g[r, sep_c]) == BLACK and r != sep_r]
    assert len(row_gaps) == 1 and len(col_gaps) == 1, "expected exactly one gap on each separator arm"
    return sep_r, sep_c, int(row_gaps[0]), int(col_gaps[0])


def _quadrant_of_point(r: int, c: int, sep_r: int, sep_c: int) -> str:
    if r < sep_r and c < sep_c:
        return "tl"
    if r < sep_r and c > sep_c:
        return "tr"
    if r > sep_r and c < sep_c:
        return "bl"
    if r > sep_r and c > sep_c:
        return "br"
    raise AssertionError("point lies on the separator")


def _extract_tokens(grid: np.ndarray, sep_r: int, sep_c: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    token_colors = colors_present(g, exclude={BLACK, GREY})
    assert len(token_colors) == 2, "expected exactly two token colors"

    comps = find_components(
        g,
        bg=BLACK,
        connectivity=8,
        monochromatic=True,
        colors=token_colors,
        return_format="dict",
    )
    assert len(comps) == 2, "expected exactly two token components"

    tokens = []
    for comp in comps:
        coords = comp["coords"]
        color = int(comp["color"])
        r0, c0, r1, c1 = bbox_coords(coords)
        assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3), "token bbox must be 3x3"
        shape_name = _classify_shape(coords)
        center = (r0 + 1, c0 + 1)
        quadrant = _quadrant_of_point(center[0], center[1], sep_r, sep_c)
        tokens.append(
            {
                "color": color,
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "center": center,
                "quadrant": quadrant,
                "shape": shape_name,
            }
        )
    quadrants = {token["quadrant"] for token in tokens}
    assert len(quadrants) == 2, "tokens must occupy two different quadrants"
    return tokens


def _build_attempts(tokens: list[dict], edges: list[tuple[str, str]], specs: dict[str, dict]) -> list[dict]:
    occupied = {token["quadrant"] for token in tokens}
    attempts = []
    for token in tokens:
        source = token["quadrant"]
        candidates = []
        for qa, qb in edges:
            if source not in (qa, qb):
                continue
            target = qb if source == qa else qa
            if target not in occupied:
                candidates.append(target)
        assert len(candidates) <= 1, "ambiguous gate routing"
        if not candidates:
            continue
        target = candidates[0]
        attempts.append(
            {
                "source_quadrant": source,
                "target_quadrant": target,
                "source_origin": specs[source]["origin"],
                "source_color": token["color"],
                "source_shape": token["shape"],
                "target_shape": NEXT_SHAPE[token["shape"]],
                "source_coords": token["coords"],
            }
        )
    return attempts


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    sep_r, sep_c, h_gap_c, v_gap_r = _find_separator(g)
    specs = _quadrant_specs(g.shape, sep_r, sep_c)
    edges = _gap_edges(sep_r, sep_c, h_gap_c, v_gap_r)
    tokens = _extract_tokens(g, sep_r, sep_c)
    attempts = _build_attempts(tokens, edges, specs)
    return {
        "sep_r": sep_r,
        "sep_c": sep_c,
        "h_gap_c": h_gap_c,
        "v_gap_r": v_gap_r,
        "specs": specs,
        "edges": edges,
        "tokens": tokens,
        "attempts": attempts,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route two colored 3x3 tokens through separator gaps and resolve contested arrivals by scan order.

    Input structure:
    - A black background contains a grey cross: one almost-full row and one almost-full column whose intersection stays grey.
    - The horizontal separator row has exactly one black gap and the vertical separator column has exactly one black gap.
    - Exactly two quadrants contain one monochrome 3x3 token each, using one of three shapes: solid block, X, or diamond.

    Rule:
    - Treat each gap as a gate connecting the two quadrants on opposite sides of that separator segment.
    - A token may transmit only through a gate touching its source quadrant and only if the quadrant on the other side is empty.
    - A successful transmission keeps the token color, moves the token to the target quadrant center, and advances its shape cycle: solid -> X -> diamond -> solid.
    - If two tokens target the same empty quadrant, only the source quadrant with earlier scan order (smaller quadrant top-left row, then column) transmits; the other token stays unchanged.
    - Successful source positions are cleared to black, while the grey separators and their two black gaps remain unchanged.

    Key insight:
    - The separator gaps are the routing cues: they determine which quadrant-to-quadrant transfers are legal, and only after that routing step does the shape cycle apply; scan order matters only when both legal routes converge on one target.
    """
    g = np.asarray(grid, dtype=int)
    analysis = _analyze(g)
    out = g.copy()

    attempts_by_target: dict[str, list[dict]] = defaultdict(list)
    for attempt in analysis["attempts"]:
        attempts_by_target[attempt["target_quadrant"]].append(attempt)

    winners = []
    for target, group in attempts_by_target.items():
        assert target in analysis["specs"], "unknown target quadrant"
        group.sort(key=lambda item: item["source_origin"])
        winners.append(group[0])

    for winner in winners:
        coords = winner["source_coords"]
        out[coords[:, 0], coords[:, 1]] = BLACK

    for winner in winners:
        center = analysis["specs"][winner["target_quadrant"]]["center"]
        _draw_shape(out, center, winner["source_color"], winner["target_shape"])

    return out


def _random_center(spec: dict[str, tuple[int, int]]) -> tuple[int, int]:
    r0, r1 = spec["rows"]
    c0, c1 = spec["cols"]
    return (
        random.randint(r0 + 1, r1 - 1),
        random.randint(c0 + 1, c1 - 1),
    )


def _pick_gap(side: str, start: int, end: int) -> int:
    choices = list(range(start + 1, end))
    assert choices, "quadrant side must be wide enough for a non-endpoint gap"
    if side in {"left", "top"}:
        return random.choice(choices[: max(1, len(choices) - 1)])
    return random.choice(choices[1:] if len(choices) > 1 else choices)


def _make_grid(target: str, colors: tuple[int, int], shapes: tuple[str, str]) -> np.ndarray:
    top_h = random.randint(*SIZE_RANGE)
    bottom_h = random.randint(*SIZE_RANGE)
    left_w = random.randint(*SIZE_RANGE)
    right_w = random.randint(*SIZE_RANGE)

    h = top_h + bottom_h + 1
    w = left_w + right_w + 1
    sep_r = top_h
    sep_c = left_w
    specs = _quadrant_specs((h, w), sep_r, sep_c)
    layout = TARGET_LAYOUTS[target]

    grid = np.full((h, w), BLACK, dtype=int)
    grid[sep_r, :] = GREY
    grid[:, sep_c] = GREY

    if layout["h_side"] == "left":
        h_gap_c = _pick_gap("left", 0, sep_c)
    else:
        h_gap_c = _pick_gap("right", sep_c, w - 1)
    if layout["v_side"] == "top":
        v_gap_r = _pick_gap("top", 0, sep_r)
    else:
        v_gap_r = _pick_gap("bottom", sep_r, h - 1)

    grid[sep_r, h_gap_c] = BLACK
    grid[v_gap_r, sep_c] = BLACK

    for quadrant, color, shape_name in zip(layout["sources"], colors, shapes):
        center = _random_center(specs[quadrant])
        _draw_shape(grid, center, color, shape_name)

    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    analysis = _analyze(grid)
    attempts = analysis["attempts"]
    tokens = analysis["tokens"]
    specs = analysis["specs"]

    if len(tokens) != 2 or len(attempts) != 2:
        return False
    if len({attempt["target_quadrant"] for attempt in attempts}) != 1:
        return False

    target = attempts[0]["target_quadrant"]
    occupied = {token["quadrant"] for token in tokens}
    if target in occupied:
        return False

    winner = min(attempts, key=lambda item: item["source_origin"])
    loser = max(attempts, key=lambda item: item["source_origin"])
    out = main(grid)
    out_analysis = _analyze(out)
    out_tokens = {token["color"]: token for token in out_analysis["tokens"]}

    if set(out_tokens) != {token["color"] for token in tokens}:
        return False
    if out_analysis["sep_r"] != analysis["sep_r"] or out_analysis["sep_c"] != analysis["sep_c"]:
        return False
    if out_analysis["h_gap_c"] != analysis["h_gap_c"] or out_analysis["v_gap_r"] != analysis["v_gap_r"]:
        return False

    moved = out_tokens[winner["source_color"]]
    stayed = out_tokens[loser["source_color"]]
    if moved["quadrant"] != target:
        return False
    if moved["shape"] != winner["target_shape"]:
        return False
    if moved["center"] != specs[target]["center"]:
        return False
    if stayed["quadrant"] != loser["source_quadrant"]:
        return False
    if stayed["shape"] != loser["source_shape"]:
        return False
    if np.array_equal(out, grid):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        target = random.choice(list(TARGET_LAYOUTS))
        colors = tuple(random.sample(PALETTE, 2))
        shapes = tuple(random.choice(list(SHAPE_OFFSETS)) for _ in range(2))
        grid = _make_grid(target, colors, shapes)
        if _validate_generated(grid):
            return grid
    raise RuntimeError("failed to generate a valid separator-gated token puzzle")
