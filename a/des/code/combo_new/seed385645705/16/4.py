import numpy as np

from collections import Counter, deque


BG = 0
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}
GEN_RETRIES = 500


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _gap_segment(side: np.ndarray) -> tuple[int, int] | None:
    idx = np.flatnonzero(side == BG)
    if idx.size not in (2, 3):
        return None
    if int(idx[-1] - idx[0] + 1) != int(idx.size):
        return None
    if int(idx[0]) == 0 or int(idx[-1]) == len(side) - 1:
        return None
    return int(idx[0]), int(idx.size)


def _gap_cells(direction: str, start: int, length: int, h: int, w: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    if direction == "up":
        for c in range(start, start + length):
            cells.add((0, c))
    elif direction == "down":
        for c in range(start, start + length):
            cells.add((h - 1, c))
    elif direction == "left":
        for r in range(start, start + length):
            cells.add((r, 0))
    else:
        for r in range(start, start + length):
            cells.add((r, w - 1))
    return cells


def _find_envelopes(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    envelopes: list[dict[str, object]] = []
    for pts in _components(g != BG):
        r0, c0, r1, c1 = _bbox(pts)
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        h, w = sub.shape
        if h < 4 or w < 4:
            continue

        inner = sub[1:-1, 1:-1]
        if inner.size == 0 or np.any(inner == BG) or not np.all(inner == inner[0, 0]):
            continue
        paper_color = int(inner[0, 0])

        fg_colors = {int(v) for v in np.unique(sub) if int(v) != BG}
        if len(fg_colors) != 3:
            continue

        side_arrays = {
            "up": sub[0, :],
            "down": sub[-1, :],
            "left": sub[:, 0],
            "right": sub[:, -1],
        }
        gap_specs = []
        for direction, side in side_arrays.items():
            seg = _gap_segment(side)
            if seg is not None:
                gap_specs.append((direction, seg[0], seg[1]))
        if len(gap_specs) != 1:
            continue
        direction, gap_start, gap_len = gap_specs[0]

        boundary_non_bg: list[int] = []
        boundary_coords: list[tuple[int, int]] = []
        for rr in range(h):
            for cc in range(w):
                if rr in (0, h - 1) or cc in (0, w - 1):
                    boundary_coords.append((rr, cc))
                    if sub[rr, cc] != BG:
                        boundary_non_bg.append(int(sub[rr, cc]))
        if not boundary_non_bg:
            continue
        border_color = Counter(boundary_non_bg).most_common(1)[0][0]

        seal_cells = [(rr, cc) for rr, cc in boundary_coords if int(sub[rr, cc]) not in (BG, border_color)]
        if len(seal_cells) != 1:
            continue
        seal_r, seal_c = seal_cells[0]
        seal_color = int(sub[seal_r, seal_c])
        if seal_color in (BG, border_color, paper_color):
            continue

        opposite = OPPOSITE[direction]
        if opposite == "up" and not (seal_r == 0 and 0 < seal_c < w - 1):
            continue
        if opposite == "down" and not (seal_r == h - 1 and 0 < seal_c < w - 1):
            continue
        if opposite == "left" and not (seal_c == 0 and 0 < seal_r < h - 1):
            continue
        if opposite == "right" and not (seal_c == w - 1 and 0 < seal_r < h - 1):
            continue

        gap = _gap_cells(direction, gap_start, gap_len, h, w)
        valid = True
        for rr, cc in boundary_coords:
            val = int(sub[rr, cc])
            if (rr, cc) in gap:
                if val != BG:
                    valid = False
                    break
            elif (rr, cc) == (seal_r, seal_c):
                if val != seal_color:
                    valid = False
                    break
            elif val != border_color:
                valid = False
                break
        if not valid:
            continue

        envelopes.append(
            {
                "bbox": (r0, c0, r1, c1),
                "paper_bbox": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
                "paper_color": paper_color,
                "seal_color": seal_color,
                "direction": direction,
            }
        )
    return envelopes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pull each letter rectangle out of its open envelope and recolor it by the opposite seal mark.

    Input structure:
    - The background is black.
    - Each foreground object is a rectangular envelope: a 1-cell border of one color around a solid interior letter of another color.
    - Exactly one border side has a 2-3 cell black gap that marks the opening.
    - The opposite border side contains one non-corner pixel of a third color, which is the seal marker.

    Rule:
    - Detect every envelope's outer rectangle, interior letter rectangle, opening side, and seal color.
    - Remove the interior letter from the envelope, turning its old location into background.
    - Translate that full letter rectangle outward along the opening direction by its own length in that direction plus 1, so it sits just outside the envelope.
    - Recolor the translated letter to the seal-marker color while keeping the border, gap, and seal pixel unchanged.

    Key insight:
    - The gap gives the movement direction, the interior size gives the travel distance, and the single opposite border marker gives the output color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    envelopes = _find_envelopes(g)

    for env in envelopes:
        pr0, pc0, pr1, pc1 = env["paper_bbox"]  # type: ignore[index]
        out[pr0 : pr1 + 1, pc0 : pc1 + 1] = BG

    for env in envelopes:
        pr0, pc0, pr1, pc1 = env["paper_bbox"]  # type: ignore[index]
        direction = env["direction"]  # type: ignore[index]
        seal_color = int(env["seal_color"])  # type: ignore[index]
        paper_h = pr1 - pr0 + 1
        paper_w = pc1 - pc0 + 1
        dy = 0
        dx = 0
        if direction == "up":
            dy = -(paper_h + 1)
        elif direction == "down":
            dy = paper_h + 1
        elif direction == "left":
            dx = -(paper_w + 1)
        else:
            dx = paper_w + 1

        for rr in range(paper_h):
            for cc in range(paper_w):
                tr = pr0 + dy + rr
                tc = pc0 + dx + cc
                if 0 <= tr < h and 0 <= tc < w:
                    out[tr, tc] = seal_color

    return out


def _randint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi + 1))


def _footprint_offsets(inner_h: int, inner_w: int, direction: str) -> tuple[int, int, int, int]:
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    if direction == "up":
        return -inner_h, 0, outer_h - 1, outer_w - 1
    if direction == "down":
        return 0, 0, outer_h - 1 + inner_h, outer_w - 1
    if direction == "left":
        return 0, -inner_w, outer_h - 1, outer_w - 1
    return 0, 0, outer_h - 1, outer_w - 1 + inner_w


def _build_patch(spec: dict[str, int]) -> np.ndarray:
    inner_h = spec["inner_h"]
    inner_w = spec["inner_w"]
    border = spec["border_color"]
    paper = spec["paper_color"]
    seal = spec["seal_color"]
    direction = spec["direction"]
    gap_len = spec["gap_len"]
    gap_start = spec["gap_start"]
    seal_pos = spec["seal_pos"]

    patch = np.full((inner_h + 2, inner_w + 2), paper, dtype=int)
    patch[0, :] = border
    patch[-1, :] = border
    patch[:, 0] = border
    patch[:, -1] = border

    if direction == "up":
        patch[0, gap_start : gap_start + gap_len] = BG
        patch[-1, seal_pos] = seal
    elif direction == "down":
        patch[-1, gap_start : gap_start + gap_len] = BG
        patch[0, seal_pos] = seal
    elif direction == "left":
        patch[gap_start : gap_start + gap_len, 0] = BG
        patch[seal_pos, -1] = seal
    else:
        patch[gap_start : gap_start + gap_len, -1] = BG
        patch[seal_pos, 0] = seal

    return patch


def _sample_spec(rng: np.random.Generator, used_borders: set[int]) -> dict[str, int]:
    direction = str(rng.choice(["up", "down", "left", "right"]))
    gap_len = int(rng.choice([2, 3]))
    if direction in ("up", "down"):
        inner_h = _randint(rng, 2, 6)
        inner_w = _randint(rng, gap_len, 6)
        gap_start = _randint(rng, 1, inner_w - gap_len + 1)
        seal_pos = _randint(rng, 1, inner_w)
    else:
        inner_h = _randint(rng, gap_len, 6)
        inner_w = _randint(rng, 2, 6)
        gap_start = _randint(rng, 1, inner_h - gap_len + 1)
        seal_pos = _randint(rng, 1, inner_h)

    border_pool = [c for c in range(1, 10) if c not in used_borders]
    border_color = int(rng.choice(border_pool))
    other_pool = [c for c in range(1, 10) if c != border_color]
    paper_color, seal_color = [int(v) for v in rng.choice(other_pool, size=2, replace=False)]
    used_borders.add(border_color)

    return {
        "direction": direction,
        "gap_len": gap_len,
        "inner_h": inner_h,
        "inner_w": inner_w,
        "gap_start": gap_start,
        "seal_pos": seal_pos,
        "border_color": border_color,
        "paper_color": paper_color,
        "seal_color": seal_color,
    }


def _place_specs(
    rng: np.random.Generator, h: int, w: int, specs: list[dict[str, int]]
) -> list[dict[str, int]] | None:
    blocked = np.zeros((h, w), dtype=bool)
    placed: list[dict[str, int]] = []
    ordered = sorted(
        specs,
        key=lambda spec: -(
            (_footprint_offsets(spec["inner_h"], spec["inner_w"], spec["direction"])[2]
             - _footprint_offsets(spec["inner_h"], spec["inner_w"], spec["direction"])[0]
             + 1)
            * (_footprint_offsets(spec["inner_h"], spec["inner_w"], spec["direction"])[3]
               - _footprint_offsets(spec["inner_h"], spec["inner_w"], spec["direction"])[1]
               + 1)
        ),
    )

    for spec in ordered:
        fr0, fc0, fr1, fc1 = _footprint_offsets(spec["inner_h"], spec["inner_w"], spec["direction"])
        min_r0 = 1 - fr0
        max_r0 = h - 2 - fr1
        min_c0 = 1 - fc0
        max_c0 = w - 2 - fc1
        if min_r0 > max_r0 or min_c0 > max_c0:
            return None

        found = None
        for _ in range(250):
            r0 = _randint(rng, min_r0, max_r0)
            c0 = _randint(rng, min_c0, max_c0)
            gr0, gc0 = r0 + fr0, c0 + fc0
            gr1, gc1 = r0 + fr1, c0 + fc1
            er0 = max(0, gr0 - 1)
            ec0 = max(0, gc0 - 1)
            er1 = min(h - 1, gr1 + 1)
            ec1 = min(w - 1, gc1 + 1)
            if blocked[er0 : er1 + 1, ec0 : ec1 + 1].any():
                continue
            found = (r0, c0, er0, ec0, er1, ec1)
            break
        if found is None:
            return None

        r0, c0, er0, ec0, er1, ec1 = found
        blocked[er0 : er1 + 1, ec0 : ec1 + 1] = True
        placed_spec = dict(spec)
        placed_spec["r0"] = r0
        placed_spec["c0"] = c0
        placed.append(placed_spec)

    return placed


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = _randint(rng, 16, 30)
    w = _randint(rng, 16, 30)
    max_count = 4 if min(h, w) >= 20 else 3
    count = _randint(rng, 1, max_count)

    used_borders: set[int] = set()
    specs = [_sample_spec(rng, used_borders) for _ in range(count)]
    placed = _place_specs(rng, h, w, specs)
    if placed is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    for spec in placed:
        patch = _build_patch(spec)
        r0 = spec["r0"]
        c0 = spec["c0"]
        ph, pw = patch.shape
        grid[r0 : r0 + ph, c0 : c0 + pw] = patch

    envelopes = _find_envelopes(grid)
    if len(envelopes) != count:
        return None
    out = main(grid)
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to produce a valid envelope layout")
