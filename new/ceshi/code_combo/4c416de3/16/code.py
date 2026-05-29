# Combo variant of 4c416de3 (index 16)
# combo_spec: D1/0+D12/2
# Name: 连接线编码配对+颜色匹配

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import draw_line, get_bg


"""
Design analysis for the fused variant
-------------------------------------
- Concrete behavior:
  The original task had one global source pattern, so every target marker expanded
  into the same shape. Here each source frame contains two color-tagged source
  patterns, and the two source frames deliberately reuse one color while assigning
  that shared color different shapes. A target frame is linked to exactly one
  source frame by a colored connector line; inside that linked source frame, the
  target marker color selects which source pattern to copy.
- Complexity calibration:
  Only two source frames and two target frames are used, each source frame exposes
  exactly two colors repeated on opposite corners. This keeps the clue set small,
  while the repeated shared color across the two sources makes the connector line
  semantically necessary.
- Implicit assumptions that changed:
  The original solver assumed one canonical pattern shared by all sources and did
  not need frame-to-frame relations. The new solver cannot do that: it must first
  detect 0-frames, then recover straight midpoint-to-midpoint connectors, then
  extract per-source-frame color->pattern maps. Generator construction guarantees
  those assumptions by using full square 0-frames, one straight connector per
  target frame, and marker colors that always exist in the connected source frame.
- Placement legality:
  Frames are placed on a 2x2 lattice with size 7 and gap >= 4, so corner patterns
  with radius 2 never overlap each other or the connector corridor. Connectors use
  side midpoints, so they never touch corner neighborhoods.
- Visual observability / cleanliness:
  The shared color appears in both source frames but with different shapes, so a
  human cannot solve by color alone. The direct colored connector clearly groups a
  target frame with its source frame, and within each pair only two colors are in
  play, preventing visual clutter.
- Quantifiable anti-degeneration signal:
  Generator validation later measures how often the shared source color is reused
  across the two sources with different patterns. If that rate collapses to zero,
  the fusion would degenerate back toward the original one-pattern behavior.
"""


CORNER_KINDS = ("TL", "TR", "BL", "BR")
MARKER_CANON = (2, 2)
NEIGH_R = 2
FRAME_SIZE = 7
MAX_ATTEMPTS = 400

_CORNER_OFFSETS = {"TL": (0, 0), "TR": (0, 1), "BL": (1, 0), "BR": (1, 1)}
_MID_SIDES = {
    "N": lambda r0, c0, s: (r0, c0 + s // 2),
    "S": lambda r0, c0, s: (r0 + s - 1, c0 + s // 2),
    "W": lambda r0, c0, s: (r0 + s // 2, c0),
    "E": lambda r0, c0, s: (r0 + s // 2, c0 + s - 1),
}
_OUT_DIR = {"N": (-1, 0), "S": (1, 0), "W": (0, -1), "E": (0, 1)}
_OPPOSITE_SIDE = {"N": "S", "S": "N", "W": "E", "E": "W"}

PATTERN_BANK = (
    frozenset([(-1, 0), (0, -1), (1, 1), (2, 2)]),
    frozenset([(-1, -1), (-1, 0), (0, -1), (1, 1), (2, 2)]),
    frozenset([(-2, 0), (0, -2), (1, 1), (2, 2)]),
    frozenset([(-2, 0), (0, -2), (-1, -1), (1, 1), (2, 2)]),
    frozenset([(-1, 0), (0, -1), (0, 0), (1, 1), (2, 2)]),
    frozenset([(-2, 0), (0, -2), (0, 0), (1, 1), (2, 2)]),
)


def _rot(dr: int, dc: int, kind: str) -> tuple[int, int]:
    """Map actual offset <-> canonical TL-oriented offset; self-inverse."""
    if kind == "TL":
        return dr, dc
    if kind == "TR":
        return dr, -dc
    if kind == "BL":
        return -dr, dc
    if kind == "BR":
        return -dr, -dc
    raise ValueError(kind)


def _corner_coord(frame: dict, kind: str) -> tuple[int, int]:
    dr, dc = _CORNER_OFFSETS[kind]
    s1 = frame["size"] - 1
    return frame["r0"] + dr * s1, frame["c0"] + dc * s1


def _draw_frame(grid: np.ndarray, r0: int, c0: int, size: int) -> None:
    r1, c1 = r0 + size - 1, c0 + size - 1
    draw_line(grid, r0, c0, end_r=r0, end_c=c1, color=0)
    draw_line(grid, r0, c0, end_r=r1, end_c=c0, color=0)
    draw_line(grid, r1, c0, end_r=r1, end_c=c1, color=0)
    draw_line(grid, r0, c1, end_r=r1, end_c=c1, color=0)


def _paint_pattern(
    grid: np.ndarray,
    r: int,
    c: int,
    kind: str,
    pattern: frozenset[tuple[int, int]],
    color: int,
) -> None:
    for dr, dc in pattern:
        rr, cc = _rot(dr, dc, kind)
        grid[r + rr, c + cc] = color


def _frame_midpoint(frame: dict, side: str) -> tuple[int, int]:
    return _MID_SIDES[side](frame["r0"], frame["c0"], frame["size"])


def _colored_around(
    g: np.ndarray, r: int, c: int, kind: str, bg: int
) -> list[tuple[tuple[int, int], int]]:
    H, W = g.shape
    items = []
    for dr in range(-NEIGH_R, NEIGH_R + 1):
        for dc in range(-NEIGH_R, NEIGH_R + 1):
            rr, cc = r + dr, c + dc
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            v = int(g[rr, cc])
            if v == bg or v == 0:
                continue
            items.append((_rot(dr, dc, kind), v))
    return items


def _find_frames(g: np.ndarray) -> list[dict]:
    frames = []
    H, W = g.shape
    s = FRAME_SIZE
    bg = int(get_bg(g))
    for r0 in range(0, H - s + 1):
        for c0 in range(0, W - s + 1):
            r1, c1 = r0 + s - 1, c0 + s - 1
            if not bool((g[r0, c0 + 1 : c1] == 0).all()):
                continue
            if not bool((g[r1, c0 + 1 : c1] == 0).all()):
                continue
            if not bool((g[r0 + 1 : r1, c0] == 0).all()):
                continue
            if not bool((g[r0 + 1 : r1, c1] == 0).all()):
                continue
            corners = [int(g[r0, c0]), int(g[r0, c1]), int(g[r1, c0]), int(g[r1, c1])]
            if any(v == bg for v in corners):
                continue
            if bool((g[r0 + 1 : r1, c0 + 1 : c1] == 0).any()):
                continue
            frames.append({"r0": r0, "c0": c0, "r1": r1, "c1": c1, "size": s})
    frames.sort(key=lambda f: (f["r0"], f["c0"]))
    uniq = []
    seen = set()
    for frame in frames:
        key = (frame["r0"], frame["c0"])
        if key not in seen:
            uniq.append(frame)
            seen.add(key)
    frames = uniq
    assert len(frames) >= 2, "expected multiple frames"
    return frames


def _detect_connectors(g: np.ndarray, frames: list[dict], bg: int) -> dict[int, tuple[int, int, str]]:
    midpoint_owner = {}
    for idx, frame in enumerate(frames):
        for side in ("N", "S", "W", "E"):
            midpoint_owner[_frame_midpoint(frame, side)] = (idx, side)

    out: dict[int, tuple[int, int, str]] = {}
    H, W = g.shape
    for idx, frame in enumerate(frames):
        for side in ("N", "S", "W", "E"):
            mr, mc = _frame_midpoint(frame, side)
            dr, dc = _OUT_DIR[side]
            rr, cc = mr + dr, mc + dc
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            color = int(g[rr, cc])
            if color == bg or color == 0:
                continue
            seen = set()
            while 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == color:
                seen.add((rr, cc))
                rr += dr
                cc += dc
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            if int(g[rr, cc]) != 0:
                continue
            owner = midpoint_owner.get((rr, cc))
            if owner is None:
                continue
            jdx, other_side = owner
            if jdx == idx or other_side != _OPPOSITE_SIDE[side]:
                continue
            out[idx] = (jdx, color, side)
            break
    return out


def _frame_corner_items(g: np.ndarray, frame: dict, bg: int) -> dict[str, list[tuple[tuple[int, int], int]]]:
    items = {}
    for kind in CORNER_KINDS:
        r, c = _corner_coord(frame, kind)
        items[kind] = _colored_around(g, r, c, kind, bg)
    return items


def _is_source_corner(items: list[tuple[tuple[int, int], int]]) -> bool:
    outward = [off for off, _ in items if off[0] < 0 or off[1] < 0]
    return len(outward) >= 2


def _extract_source_map(
    frame: dict, corner_items: dict[str, list[tuple[tuple[int, int], int]]]
) -> dict[int, frozenset[tuple[int, int]]]:
    source_map: dict[int, frozenset[tuple[int, int]]] = {}
    for kind in CORNER_KINDS:
        items = corner_items[kind]
        if not _is_source_corner(items):
            continue
        color_counts = Counter(v for _, v in items)
        pat_col = int(color_counts.most_common(1)[0][0])
        canon = frozenset(off for off, col in items if int(col) == pat_col)
        assert MARKER_CANON in canon, f"source corner missing marker anchor: {frame}, {kind}"
        for dr, dc in canon:
            assert (dc, dr) in canon, f"pattern not diagonal-symmetric: {canon}"
        prev = source_map.get(pat_col)
        if prev is None:
            source_map[pat_col] = canon
        else:
            assert prev == canon, f"same color maps to different source patterns in frame: {pat_col}"
    assert len(source_map) >= 2, "source frame should expose at least two color-tagged patterns"
    return source_map


def _target_marker(frame: dict, kind: str) -> tuple[int, int]:
    r, c = _corner_coord(frame, kind)
    dr, dc = _rot(*MARKER_CANON, kind)
    return r + dr, c + dc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each target-frame corner from the color-matched source inside the connected frame.

    Input structure:
    - bg = the dominant color. Several hollow square 0-frames are present, and some frames
      are joined by straight colored connector lines that touch the midpoint of a frame side.
    - Source frames already contain full corner patterns. Within one source frame, opposite
      corners repeat one of two colors, and each color stands for one diagonally symmetric
      canonical pattern.
    - Target frames contain only one colored marker per corner at the deepest inward-diagonal
      position. Marker colors may repeat across different source frames, so color alone is not
      globally sufficient.

    Rule:
    - Detect which source frame is paired to each target frame by following the connector line.
    - Inside that paired source frame, build a mapping from color to canonical corner pattern.
    - For every target marker, copy the pattern associated with the same color in the connected
      source frame, rotate it to the target corner orientation, and paint it using the marker color.

    Key insight:
    - The connector determines the relevant source frame, while the marker color chooses one of
      the two source patterns inside that frame. The same color can exist in another source frame
      with a different shape, so both cues are necessary.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    frames = _find_frames(g)
    connectors = _detect_connectors(g, frames, bg)

    per_frame_items = [_frame_corner_items(g, frame, bg) for frame in frames]
    source_maps: dict[int, dict[int, frozenset[tuple[int, int]]]] = {}
    target_frames = []
    for idx, frame in enumerate(frames):
        items = per_frame_items[idx]
        n_source = sum(1 for kind in CORNER_KINDS if _is_source_corner(items[kind]))
        if n_source >= 2:
            source_maps[idx] = _extract_source_map(frame, items)
        else:
            target_frames.append(idx)

    assert source_maps, "no source frames found"
    assert target_frames, "no target frames found"

    out = g.copy()
    H, W = g.shape
    for idx in target_frames:
        assert idx in connectors, f"target frame {idx} has no connector"
        src_idx, _line_color, _side = connectors[idx]
        assert src_idx in source_maps, f"target frame {idx} is not connected to a source frame"
        color_to_pattern = source_maps[src_idx]
        frame = frames[idx]
        items = per_frame_items[idx]
        for kind in CORNER_KINDS:
            mr, mc = _target_marker(frame, kind)
            assert 0 <= mr < H and 0 <= mc < W
            marker = int(g[mr, mc])
            assert marker != bg and marker != 0, f"target marker missing at frame {idx} corner {kind}"
            for off, color in items[kind]:
                assert off == MARKER_CANON and int(color) == marker, (
                    f"target corner has unexpected extra paint: frame {idx}, corner {kind}, {items[kind]}"
                )
            assert marker in color_to_pattern, f"marker color {marker} absent from connected source frame {src_idx}"
            _paint_pattern(out, *_corner_coord(frame, kind), kind, color_to_pattern[marker], marker)

    return out


def _choose_source_specs(palette: list[int]) -> list[dict]:
    shared = random.choice(palette)
    rem = [c for c in palette if c != shared]
    other1, other2 = random.sample(rem, 2)
    pats = random.sample(PATTERN_BANK, 4)
    while pats[0] == pats[2]:
        pats = random.sample(PATTERN_BANK, 4)
    return [
        {"colors": (shared, other1), "patterns": {shared: pats[0], other1: pats[1]}},
        {"colors": (shared, other2), "patterns": {shared: pats[2], other2: pats[3]}},
    ]


def _assign_source_corners(frame: dict, spec: dict) -> None:
    if random.random() < 0.5:
        pair_a = ("TL", "BR")
        pair_b = ("TR", "BL")
    else:
        pair_a = ("TR", "BL")
        pair_b = ("TL", "BR")
    color_a, color_b = spec["colors"]
    frame["source_assign"] = {kind: color_a for kind in pair_a} | {kind: color_b for kind in pair_b}
    frame["source_colors"] = spec["colors"]
    frame["patterns"] = spec["patterns"]


def _assign_target_corners(frame: dict, src_spec: dict) -> None:
    color_a, color_b = src_spec["source_colors"]
    if random.random() < 0.5:
        frame["target_assign"] = {"TL": color_a, "TR": color_b, "BL": color_b, "BR": color_a}
    else:
        frame["target_assign"] = {"TL": color_b, "TR": color_a, "BL": color_a, "BR": color_b}


def _place_frames() -> tuple[np.ndarray, list[dict]]:
    top = 2
    left = 2
    bottom = 2
    right = 2
    row_gap = random.randint(4, 6)
    col_gap = random.randint(4, 6)
    H = top + FRAME_SIZE + row_gap + FRAME_SIZE + bottom
    W = left + FRAME_SIZE + col_gap + FRAME_SIZE + right
    if not (18 <= H <= 24 and 18 <= W <= 24):
        raise AssertionError("sampled grid size out of bounds")
    grid = np.empty((H, W), dtype=int)
    frames = [
        {"r0": top, "c0": left, "size": FRAME_SIZE},
        {"r0": top, "c0": left + FRAME_SIZE + col_gap, "size": FRAME_SIZE},
        {"r0": top + FRAME_SIZE + row_gap, "c0": left, "size": FRAME_SIZE},
        {"r0": top + FRAME_SIZE + row_gap, "c0": left + FRAME_SIZE + col_gap, "size": FRAME_SIZE},
    ]
    for frame in frames:
        frame["r1"] = frame["r0"] + FRAME_SIZE - 1
        frame["c1"] = frame["c0"] + FRAME_SIZE - 1
    return grid, frames


def _layout_pairing(frames: list[dict]) -> tuple[list[int], list[int], list[tuple[int, int, str, str]]]:
    mode = random.choice(("rows", "cols"))
    if mode == "rows":
        left_is_source = random.random() < 0.5
        if left_is_source:
            sources = [0, 2]
            targets = [1, 3]
            conn = [(0, 1, "E", "W"), (2, 3, "E", "W")]
        else:
            sources = [1, 3]
            targets = [0, 2]
            conn = [(1, 0, "W", "E"), (3, 2, "W", "E")]
    else:
        top_is_source = random.random() < 0.5
        if top_is_source:
            sources = [0, 1]
            targets = [2, 3]
            conn = [(0, 2, "S", "N"), (1, 3, "S", "N")]
        else:
            sources = [2, 3]
            targets = [0, 1]
            conn = [(2, 0, "N", "S"), (3, 1, "N", "S")]
    return sources, targets, conn


def _draw_connector(grid: np.ndarray, fa: dict, fb: dict, side_a: str, side_b: str, color: int) -> None:
    ar, ac = _frame_midpoint(fa, side_a)
    br, bc = _frame_midpoint(fb, side_b)
    da_r, da_c = _OUT_DIR[side_a]
    db_r, db_c = _OUT_DIR[side_b]
    start = (ar + da_r, ac + da_c)
    end = (br + db_r, bc + db_c)
    draw_line(grid, start[0], start[1], end_r=end[0], end_c=end[1], color=color)


def _render_output(
    base: np.ndarray,
    frames: list[dict],
    sources: list[int],
    targets: list[int],
    pair_map: dict[int, int],
) -> np.ndarray:
    out = base.copy()
    for idx in sources:
        frame = frames[idx]
        for kind, color in frame["source_assign"].items():
            _paint_pattern(out, *_corner_coord(frame, kind), kind, frame["patterns"][color], color)
    for idx in targets:
        frame = frames[idx]
        src_patterns = frames[pair_map[idx]]["patterns"]
        for kind, color in frame["target_assign"].items():
            _paint_pattern(out, *_corner_coord(frame, kind), kind, src_patterns[color], color)
    return out


def _derive_input(
    output: np.ndarray,
    base: np.ndarray,
    frames: list[dict],
    targets: list[int],
    pair_map: dict[int, int],
) -> np.ndarray:
    inp = output.copy()
    for idx in targets:
        frame = frames[idx]
        src_patterns = frames[pair_map[idx]]["patterns"]
        for kind, color in frame["target_assign"].items():
            marker = _target_marker(frame, kind)
            for dr, dc in src_patterns[color]:
                rr, cc = _rot(dr, dc, kind)
                rr += _corner_coord(frame, kind)[0]
                cc += _corner_coord(frame, kind)[1]
                if (rr, cc) != marker:
                    inp[rr, cc] = base[rr, cc]
    return inp


def _gen_attempt() -> np.ndarray | None:
    bg = random.randint(1, 9)
    palette = [c for c in range(1, 10) if c != bg]
    if len(palette) < 6:
        return None

    grid, frames = _place_frames()
    grid.fill(bg)
    for frame in frames:
        _draw_frame(grid, frame["r0"], frame["c0"], frame["size"])

    sources, targets, connections = _layout_pairing(frames)
    source_specs = _choose_source_specs(palette)
    for idx, spec in zip(sources, source_specs):
        _assign_source_corners(frames[idx], spec)
    pair_map = {}
    for src_idx, tgt_idx, _side_a, _side_b in connections:
        pair_map[tgt_idx] = src_idx
    for idx in targets:
        _assign_target_corners(frames[idx], frames[pair_map[idx]])

    used_source_colors = set()
    for idx in sources:
        used_source_colors |= set(frames[idx]["patterns"].keys())
    connector_palette = [c for c in palette if c not in used_source_colors]
    if len(connector_palette) < len(connections):
        return None
    connector_colors = random.sample(connector_palette, len(connections))

    for (src_idx, tgt_idx, side_a, side_b), conn_col in zip(connections, connector_colors):
        _draw_connector(grid, frames[src_idx], frames[tgt_idx], side_a, side_b, conn_col)

    base = grid.copy()
    output = _render_output(base, frames, sources, targets, pair_map)
    inp = _derive_input(output, base, frames, targets, pair_map)

    # Anti-degeneration guard: the shared color must appear in both sources but with different patterns.
    shared = set(frames[sources[0]]["patterns"]).intersection(set(frames[sources[1]]["patterns"]))
    assert len(shared) == 1
    shared_color = next(iter(shared))
    if frames[sources[0]]["patterns"][shared_color] == frames[sources[1]]["patterns"][shared_color]:
        return None

    if not np.array_equal(main(inp), output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
