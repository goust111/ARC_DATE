from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
GRID_MAX = 30
GEN_RETRIES = 300


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] != BG:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rs = [rc[0] for rc in coords]
            cs = [rc[1] for rc in coords]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(cs), max(cs)
            comps.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "patch": grid[r0 : r1 + 1, c0 : c1 + 1].copy(),
                }
            )
    return comps


def _body_color(patch: np.ndarray) -> int:
    vals = [int(v) for v in patch.reshape(-1) if int(v) != BG]
    assert vals, "piece must contain non-background pixels"
    color, _count = Counter(vals).most_common(1)[0]
    return int(color)


def _extract_horizontal_piece(patch: np.ndarray) -> dict:
    patch = np.asarray(patch, dtype=int)
    body = _body_color(patch)
    body_mask = patch == body
    h, w = patch.shape
    assert body_mask[0].any() and body_mask[-1].any(), "horizontal piece must have flat top and bottom"

    lefts: list[int] = []
    rights: list[int] = []
    for r in range(h):
        cols = np.flatnonzero(body_mask[r])
        assert len(cols) > 0, "every row must contain body pixels"
        assert cols[-1] - cols[0] + 1 == len(cols), "horizontal piece rows must stay contiguous"
        lefts.append(int(cols[0]))
        rights.append(int(cols[-1]))

    non_body = {(int(r), int(c)) for r, c in np.argwhere((patch != BG) & (patch != body))}

    def edge_info(side: str) -> tuple[int, dict | None]:
        if side == "left":
            colors = []
            line_pos = set()
            for r in range(h):
                c = lefts[r] - 1
                if c >= 0 and patch[r, c] not in (BG, body):
                    colors.append(int(patch[r, c]))
                    line_pos.add((r, c))
            if not colors:
                assert len(set(lefts)) == 1, "flat outer left edge should not vary"
                return lefts[0], None
            assert len(set(colors)) == 1, "left edge line color must be uniform"
            assert min(lefts) + 2 == max(lefts), "toothed left edge must contain both protrusions and indentations"
            base = min(lefts) + 1
            pattern = tuple(base - x for x in lefts)
            assert set(pattern) == {-1, 1}, "left tooth pattern must be pure protrusion/indentation"
            return base, {"color": colors[0], "pattern": pattern, "line_pos": line_pos}
        colors = []
        line_pos = set()
        for r in range(h):
            c = rights[r] + 1
            if c < w and patch[r, c] not in (BG, body):
                colors.append(int(patch[r, c]))
                line_pos.add((r, c))
        if not colors:
            assert len(set(rights)) == 1, "flat outer right edge should not vary"
            return rights[0], None
        assert len(set(colors)) == 1, "right edge line color must be uniform"
        assert min(rights) + 2 == max(rights), "toothed right edge must contain both protrusions and indentations"
        base = min(rights) + 1
        pattern = tuple(x - base for x in rights)
        assert set(pattern) == {-1, 1}, "right tooth pattern must be pure protrusion/indentation"
        return base, {"color": colors[0], "pattern": pattern, "line_pos": line_pos}

    core_left, left = edge_info("left")
    core_right, right = edge_info("right")
    assert core_left <= core_right, "piece core width must be positive"

    expected = set()
    if left is not None:
        expected |= left["line_pos"]
    if right is not None:
        expected |= right["line_pos"]
    assert non_body == expected, "horizontal piece may only contain side line pixels"

    return {
        "orientation": "horizontal",
        "patch": patch,
        "body_color": body,
        "body_mask": body_mask,
        "core_left": int(core_left),
        "core_right": int(core_right),
        "core_top": 0,
        "core_bottom": h - 1,
        "core_width": int(core_right - core_left + 1),
        "core_height": int(h),
        "left": None if left is None else {"color": int(left["color"]), "pattern": tuple(int(v) for v in left["pattern"])},
        "right": None if right is None else {"color": int(right["color"]), "pattern": tuple(int(v) for v in right["pattern"])},
    }


def _extract_vertical_piece(patch: np.ndarray) -> dict:
    patch = np.asarray(patch, dtype=int)
    body = _body_color(patch)
    body_mask = patch == body
    h, w = patch.shape
    assert body_mask[:, 0].any() and body_mask[:, -1].any(), "vertical piece must have flat left and right"

    tops: list[int] = []
    bottoms: list[int] = []
    for c in range(w):
        rows = np.flatnonzero(body_mask[:, c])
        assert len(rows) > 0, "every column must contain body pixels"
        assert rows[-1] - rows[0] + 1 == len(rows), "vertical piece columns must stay contiguous"
        tops.append(int(rows[0]))
        bottoms.append(int(rows[-1]))

    non_body = {(int(r), int(c)) for r, c in np.argwhere((patch != BG) & (patch != body))}

    def edge_info(side: str) -> tuple[int, dict | None]:
        if side == "top":
            colors = []
            line_pos = set()
            for c in range(w):
                r = tops[c] - 1
                if r >= 0 and patch[r, c] not in (BG, body):
                    colors.append(int(patch[r, c]))
                    line_pos.add((r, c))
            if not colors:
                assert len(set(tops)) == 1, "flat outer top edge should not vary"
                return tops[0], None
            assert len(set(colors)) == 1, "top edge line color must be uniform"
            assert min(tops) + 2 == max(tops), "toothed top edge must contain both protrusions and indentations"
            base = min(tops) + 1
            pattern = tuple(base - x for x in tops)
            assert set(pattern) == {-1, 1}, "top tooth pattern must be pure protrusion/indentation"
            return base, {"color": colors[0], "pattern": pattern, "line_pos": line_pos}
        colors = []
        line_pos = set()
        for c in range(w):
            r = bottoms[c] + 1
            if r < h and patch[r, c] not in (BG, body):
                colors.append(int(patch[r, c]))
                line_pos.add((r, c))
        if not colors:
            assert len(set(bottoms)) == 1, "flat outer bottom edge should not vary"
            return bottoms[0], None
        assert len(set(colors)) == 1, "bottom edge line color must be uniform"
        assert min(bottoms) + 2 == max(bottoms), "toothed bottom edge must contain both protrusions and indentations"
        base = min(bottoms) + 1
        pattern = tuple(x - base for x in bottoms)
        assert set(pattern) == {-1, 1}, "bottom tooth pattern must be pure protrusion/indentation"
        return base, {"color": colors[0], "pattern": pattern, "line_pos": line_pos}

    core_top, top = edge_info("top")
    core_bottom, bottom = edge_info("bottom")
    assert core_top <= core_bottom, "piece core height must be positive"

    expected = set()
    if top is not None:
        expected |= top["line_pos"]
    if bottom is not None:
        expected |= bottom["line_pos"]
    assert non_body == expected, "vertical piece may only contain side line pixels"

    return {
        "orientation": "vertical",
        "patch": patch,
        "body_color": body,
        "body_mask": body_mask,
        "core_left": 0,
        "core_right": w - 1,
        "core_top": int(core_top),
        "core_bottom": int(core_bottom),
        "core_width": int(w),
        "core_height": int(core_bottom - core_top + 1),
        "top": None if top is None else {"color": int(top["color"]), "pattern": tuple(int(v) for v in top["pattern"])},
        "bottom": None if bottom is None else {"color": int(bottom["color"]), "pattern": tuple(int(v) for v in bottom["pattern"])},
    }


def _match_chain(pieces: list[dict], first_key: str, second_key: str) -> list[int]:
    start = [i for i, piece in enumerate(pieces) if piece[first_key] is None]
    end = [i for i, piece in enumerate(pieces) if piece[second_key] is None]
    assert len(start) == 1 and len(end) == 1, "chain must have exactly one start and one end"

    next_map: dict[int, int] = {}
    prev_map: dict[int, int] = {}
    for i, piece in enumerate(pieces):
        edge = piece[second_key]
        if edge is None:
            continue
        target_pattern = tuple(-int(v) for v in edge["pattern"])
        cand = [
            j
            for j, other in enumerate(pieces)
            if other[first_key] is not None
            and int(other[first_key]["color"]) == int(edge["color"])
            and tuple(int(v) for v in other[first_key]["pattern"]) == target_pattern
        ]
        assert len(cand) == 1, "every internal seam must have exactly one matching partner"
        j = cand[0]
        assert i != j, "piece cannot match itself"
        assert i not in next_map and j not in prev_map, "chain branching is not allowed"
        next_map[i] = j
        prev_map[j] = i

    order = []
    cur = start[0]
    seen = set()
    while True:
        assert cur not in seen, "cycle detected in piece chain"
        seen.add(cur)
        order.append(cur)
        if cur not in next_map:
            break
        cur = next_map[cur]
    assert len(order) == len(pieces), "not all pieces are linked into one chain"
    assert order[-1] == end[0], "final piece does not end at the unique chain end"
    return order


def _assemble_horizontal(pieces: list[dict], order: list[int]) -> np.ndarray:
    heights = {int(pieces[i]["core_height"]) for i in order}
    assert len(heights) == 1, "horizontal chain pieces must share one height"
    out_h = next(iter(heights))
    out_w = sum(int(pieces[i]["core_width"]) for i in order)
    out = np.full((out_h, out_w), BG, dtype=int)

    offsets: dict[int, int] = {}
    x = 0
    for idx in order:
        piece = pieces[idx]
        offsets[idx] = x
        rr, cc = np.where(piece["body_mask"])
        for r, c in zip(rr, cc, strict=True):
            out_r = int(r - piece["core_top"])
            out_c = int(x + c - piece["core_left"])
            assert 0 <= out_r < out_h and 0 <= out_c < out_w
            assert out[out_r, out_c] == BG, "assembled bodies must not overlap"
            out[out_r, out_c] = int(piece["body_color"])
        x += int(piece["core_width"])

    for left_idx, right_idx in zip(order, order[1:]):
        piece = pieces[left_idx]
        edge = piece["right"]
        assert edge is not None
        boundary = offsets[left_idx] + int(piece["core_width"])
        seam_color = int(edge["color"])
        for r, sign in enumerate(edge["pattern"]):
            seam_c = boundary if int(sign) > 0 else boundary - 1
            out[r, seam_c] = seam_color

    assert np.all(out != BG), "assembled horizontal puzzle should be a full rectangle"
    return out


def _assemble_vertical(pieces: list[dict], order: list[int]) -> np.ndarray:
    widths = {int(pieces[i]["core_width"]) for i in order}
    assert len(widths) == 1, "vertical chain pieces must share one width"
    out_w = next(iter(widths))
    out_h = sum(int(pieces[i]["core_height"]) for i in order)
    out = np.full((out_h, out_w), BG, dtype=int)

    offsets: dict[int, int] = {}
    y = 0
    for idx in order:
        piece = pieces[idx]
        offsets[idx] = y
        rr, cc = np.where(piece["body_mask"])
        for r, c in zip(rr, cc, strict=True):
            out_r = int(y + r - piece["core_top"])
            out_c = int(c - piece["core_left"])
            assert 0 <= out_r < out_h and 0 <= out_c < out_w
            assert out[out_r, out_c] == BG, "assembled bodies must not overlap"
            out[out_r, out_c] = int(piece["body_color"])
        y += int(piece["core_height"])

    for top_idx, bottom_idx in zip(order, order[1:]):
        piece = pieces[top_idx]
        edge = piece["bottom"]
        assert edge is not None
        boundary = offsets[top_idx] + int(piece["core_height"])
        seam_color = int(edge["color"])
        for c, sign in enumerate(edge["pattern"]):
            seam_r = boundary if int(sign) > 0 else boundary - 1
            out[seam_r, c] = seam_color

    assert np.all(out != BG), "assembled vertical puzzle should be a full rectangle"
    return out


def _solve_horizontal(components: list[dict]) -> np.ndarray:
    pieces = [_extract_horizontal_piece(comp["patch"]) for comp in components]
    order = _match_chain(pieces, "left", "right")
    return _assemble_horizontal(pieces, order)


def _solve_vertical(components: list[dict]) -> np.ndarray:
    pieces = [_extract_vertical_piece(comp["patch"]) for comp in components]
    order = _match_chain(pieces, "top", "bottom")
    return _assemble_vertical(pieces, order)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered rectangular puzzle strips by matching complementary teeth and equal edge-line colors.

    Input structure:
    - Background is black.
    - Each piece has one solid body color plus one-pixel colored edge lines on its toothed sides.
    - Matching sides share a line color and use complementary protrusion/indentation patterns.

    Rule:
    - Identify each piece and recover the tooth pattern on each colored side.
    - Match equal-colored sides whose tooth patterns are exact complements.
    - Translate pieces so the teeth interlock into one full rectangle.
    - Remove the original edge lines and recolor the touching tooth cells as the seam color.

    Key insight:
    - The line color says which sides belong together, while the tooth complement fixes the exact ordering.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.any(g != BG), "input must contain foreground pieces"
    components = _find_components(g)
    assert len(components) >= 2, "puzzle needs at least two pieces"

    errors: list[str] = []
    for solver in (_solve_horizontal, _solve_vertical):
        try:
            out = solver(components)
        except AssertionError as exc:
            errors.append(f"{solver.__name__}: {exc}")
            continue
        return out
    raise AssertionError("unable to reconstruct puzzle; " + " | ".join(errors))


def _build_horizontal_piece(
    height: int,
    core_width: int,
    body_color: int,
    left_edge: dict | None,
    right_edge: dict | None,
) -> np.ndarray:
    left_pattern = left_edge["pattern"] if left_edge is not None else (0,) * height
    right_pattern = right_edge["pattern"] if right_edge is not None else (0,) * height

    starts = [-int(v) for v in left_pattern]
    ends = [core_width - 1 + int(v) for v in right_pattern]

    min_c = min(starts)
    max_c = max(ends)
    if left_edge is not None:
        min_c = min(min_c, min(starts) - 1)
    if right_edge is not None:
        max_c = max(max_c, max(ends) + 1)

    offset = -min_c
    patch = np.zeros((height, max_c - min_c + 1), dtype=int)
    for r in range(height):
        patch[r, starts[r] + offset : ends[r] + offset + 1] = body_color
        if left_edge is not None:
            patch[r, starts[r] + offset - 1] = int(left_edge["color"])
        if right_edge is not None:
            patch[r, ends[r] + offset + 1] = int(right_edge["color"])
    return patch


def _build_vertical_piece(
    core_height: int,
    width: int,
    body_color: int,
    top_edge: dict | None,
    bottom_edge: dict | None,
) -> np.ndarray:
    top_pattern = top_edge["pattern"] if top_edge is not None else (0,) * width
    bottom_pattern = bottom_edge["pattern"] if bottom_edge is not None else (0,) * width

    starts = [-int(v) for v in top_pattern]
    ends = [core_height - 1 + int(v) for v in bottom_pattern]

    min_r = min(starts)
    max_r = max(ends)
    if top_edge is not None:
        min_r = min(min_r, min(starts) - 1)
    if bottom_edge is not None:
        max_r = max(max_r, max(ends) + 1)

    offset = -min_r
    patch = np.zeros((max_r - min_r + 1, width), dtype=int)
    for c in range(width):
        patch[starts[c] + offset : ends[c] + offset + 1, c] = body_color
        if top_edge is not None:
            patch[starts[c] + offset - 1, c] = int(top_edge["color"])
        if bottom_edge is not None:
            patch[ends[c] + offset + 1, c] = int(bottom_edge["color"])
    return patch


def _expected_horizontal(specs: list[dict], height: int) -> np.ndarray:
    total_w = sum(int(spec["core_width"]) for spec in specs)
    out = np.full((height, total_w), BG, dtype=int)
    x = 0
    for spec in specs:
        left_pattern = spec["left_edge"]["pattern"] if spec["left_edge"] is not None else (0,) * height
        right_pattern = spec["right_edge"]["pattern"] if spec["right_edge"] is not None else (0,) * height
        for r in range(height):
            start = x - int(left_pattern[r])
            end = x + int(spec["core_width"]) - 1 + int(right_pattern[r])
            out[r, start : end + 1] = int(spec["body_color"])
        x += int(spec["core_width"])
    x = 0
    for spec in specs[:-1]:
        edge = spec["right_edge"]
        assert edge is not None
        boundary = x + int(spec["core_width"])
        for r, sign in enumerate(edge["pattern"]):
            seam_c = boundary if int(sign) > 0 else boundary - 1
            out[r, seam_c] = int(edge["color"])
        x += int(spec["core_width"])
    return out


def _expected_vertical(specs: list[dict], width: int) -> np.ndarray:
    total_h = sum(int(spec["core_height"]) for spec in specs)
    out = np.full((total_h, width), BG, dtype=int)
    y = 0
    for spec in specs:
        top_pattern = spec["top_edge"]["pattern"] if spec["top_edge"] is not None else (0,) * width
        bottom_pattern = spec["bottom_edge"]["pattern"] if spec["bottom_edge"] is not None else (0,) * width
        for c in range(width):
            start = y - int(top_pattern[c])
            end = y + int(spec["core_height"]) - 1 + int(bottom_pattern[c])
            out[start : end + 1, c] = int(spec["body_color"])
        y += int(spec["core_height"])
    y = 0
    for spec in specs[:-1]:
        edge = spec["bottom_edge"]
        assert edge is not None
        boundary = y + int(spec["core_height"])
        for c, sign in enumerate(edge["pattern"]):
            seam_r = boundary if int(sign) > 0 else boundary - 1
            out[seam_r, c] = int(edge["color"])
        y += int(spec["core_height"])
    return out


def _sample_pattern(length: int, used: set[tuple[int, ...]], rng: random.Random) -> tuple[int, ...]:
    for _ in range(200):
        pattern = tuple(rng.choice((-1, 1)) for _ in range(length))
        if 1 not in pattern or -1 not in pattern:
            continue
        if pattern in used:
            continue
        used.add(pattern)
        return pattern
    raise RuntimeError("failed to sample a unique tooth pattern")


def _reserve_ok(existing: list[tuple[int, int, int, int]], top: int, left: int, h: int, w: int) -> bool:
    r0, c0, r1, c1 = top, left, top + h - 1, left + w - 1
    for er0, ec0, er1, ec1 in existing:
        if not (r1 + 1 < er0 or er1 + 1 < r0 or c1 + 1 < ec0 or ec1 + 1 < c0):
            return False
    return True


def _scatter_patches(patches: list[np.ndarray], rng: random.Random) -> np.ndarray | None:
    max_h = max(int(p.shape[0]) for p in patches)
    max_w = max(int(p.shape[1]) for p in patches)
    min_h = max(18, max_h + 2)
    min_w = max(18, max_w + 2)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    for _ in range(120):
        canvas_h = rng.randint(min_h, GRID_MAX)
        canvas_w = rng.randint(min_w, GRID_MAX)
        placements: list[tuple[int, int, int, int]] = []
        coords: list[tuple[int, int] | None] = [None] * len(patches)
        order = list(range(len(patches)))
        rng.shuffle(order)
        ok = True
        for idx in order:
            patch = patches[idx]
            ph, pw = patch.shape
            if ph + 2 > canvas_h or pw + 2 > canvas_w:
                ok = False
                break
            placed = False
            for _ in range(250):
                top = rng.randint(1, canvas_h - ph - 1)
                left = rng.randint(1, canvas_w - pw - 1)
                if _reserve_ok(placements, top, left, ph, pw):
                    placements.append((top, left, top + ph - 1, left + pw - 1))
                    coords[idx] = (top, left)
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if not ok:
            continue
        grid = np.zeros((canvas_h, canvas_w), dtype=int)
        for idx, patch in enumerate(patches):
            top, left = coords[idx]
            ph, pw = patch.shape
            region = grid[top : top + ph, left : left + pw]
            assert np.all(region == BG)
            region[patch != BG] = patch[patch != BG]
        return grid
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    orientation = rng.choice(("horizontal", "vertical"))
    piece_count = rng.randint(2, 5)

    body_pool = list(range(1, 10))
    rng.shuffle(body_pool)
    body_colors = body_pool[:piece_count]
    seam_candidates = body_pool[piece_count:]
    seam_palette_size = rng.randint(1, min(3, piece_count - 1, len(seam_candidates)))
    seam_palette = seam_candidates[:seam_palette_size]
    if len(seam_palette) < 1:
        return None

    if orientation == "horizontal":
        height = rng.randint(5, 8)
        core_widths = [rng.randint(4, 7) for _ in range(piece_count)]
        if sum(core_widths) > 22:
            return None
        used_by_color = {color: set() for color in seam_palette}
        seam_edges = []
        for _ in range(piece_count - 1):
            seam_color = rng.choice(seam_palette)
            pattern = _sample_pattern(height, used_by_color[seam_color], rng)
            seam_edges.append({"color": seam_color, "pattern": pattern})

        specs = []
        for i in range(piece_count):
            left_edge = None if i == 0 else {"color": seam_edges[i - 1]["color"], "pattern": tuple(-v for v in seam_edges[i - 1]["pattern"])}
            right_edge = None if i == piece_count - 1 else {"color": seam_edges[i]["color"], "pattern": seam_edges[i]["pattern"]}
            specs.append(
                {
                    "body_color": body_colors[i],
                    "core_width": core_widths[i],
                    "left_edge": left_edge,
                    "right_edge": right_edge,
                }
            )
        patches = [
            _build_horizontal_piece(height, int(spec["core_width"]), int(spec["body_color"]), spec["left_edge"], spec["right_edge"])
            for spec in specs
        ]
        grid = _scatter_patches(patches, rng)
        if grid is None:
            return None
        expected = _expected_horizontal(specs, height)
    else:
        width = rng.randint(5, 8)
        core_heights = [rng.randint(4, 7) for _ in range(piece_count)]
        if sum(core_heights) > 22:
            return None
        used_by_color = {color: set() for color in seam_palette}
        seam_edges = []
        for _ in range(piece_count - 1):
            seam_color = rng.choice(seam_palette)
            pattern = _sample_pattern(width, used_by_color[seam_color], rng)
            seam_edges.append({"color": seam_color, "pattern": pattern})

        specs = []
        for i in range(piece_count):
            top_edge = None if i == 0 else {"color": seam_edges[i - 1]["color"], "pattern": tuple(-v for v in seam_edges[i - 1]["pattern"])}
            bottom_edge = None if i == piece_count - 1 else {"color": seam_edges[i]["color"], "pattern": seam_edges[i]["pattern"]}
            specs.append(
                {
                    "body_color": body_colors[i],
                    "core_height": core_heights[i],
                    "top_edge": top_edge,
                    "bottom_edge": bottom_edge,
                }
            )
        patches = [
            _build_vertical_piece(int(spec["core_height"]), width, int(spec["body_color"]), spec["top_edge"], spec["bottom_edge"])
            for spec in specs
        ]
        grid = _scatter_patches(patches, rng)
        if grid is None:
            return None
        expected = _expected_vertical(specs, width)

    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_RETRIES} attempts")
