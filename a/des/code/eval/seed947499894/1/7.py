import numpy as np
from collections import deque


BG = 0
SPRITE_COLOR = 5
RED = 2
GREEN = 3
YELLOW = 4
ARROW_COLORS = (RED, GREEN, YELLOW)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = DIRS4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
RNG = np.random.default_rng()


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def connected_components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    dirs = DIRS4 if connectivity == 4 else DIRS8
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        q = deque([(r, c)])
        comp: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            comp.append((cr, cc))
            for dr, dc in dirs:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def mask_from_coords(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=np.uint8)
    for r, c in coords:
        mask[r - r0, c - c0] = 1
    return mask, (r0, c0)


def transform_mask(mask: np.ndarray, color: int, length: int) -> np.ndarray:
    if color == RED:
        return np.rot90(mask, -length)
    if color == GREEN:
        return np.fliplr(mask) if length % 2 == 1 else np.flipud(mask)
    if color == YELLOW:
        factor = length + 1
        return np.repeat(np.repeat(mask, factor, axis=0), factor, axis=1)
    raise AssertionError("unexpected arrow color")


def _parse_arrow_candidates(coords: list[tuple[int, int]]) -> list[dict]:
    pts = list(coords)
    candidates: list[dict] = []
    seen = set()

    for head in pts:
        line = [p for p in pts if p != head]
        length = len(line)
        if not (1 <= length <= 3):
            continue

        rows = sorted({r for r, _ in line})
        cols = sorted({c for _, c in line})

        if len(rows) == 1:
            row = rows[0]
            line_cols = sorted(c for _, c in line)
            if line_cols == list(range(line_cols[0], line_cols[-1] + 1)):
                cmin, cmax = line_cols[0], line_cols[-1]
                if head in ((row - 1, cmax + 1), (row + 1, cmax + 1)):
                    sig = ("right", length, row, cmax + 1, head)
                    if sig not in seen:
                        seen.add(sig)
                        candidates.append(
                            {
                                "direction": "right",
                                "length": length,
                                "front": (row, cmax + 1),
                                "head": head,
                            }
                        )
                if head in ((row - 1, cmin - 1), (row + 1, cmin - 1)):
                    sig = ("left", length, row, cmin - 1, head)
                    if sig not in seen:
                        seen.add(sig)
                        candidates.append(
                            {
                                "direction": "left",
                                "length": length,
                                "front": (row, cmin - 1),
                                "head": head,
                            }
                        )

        if len(cols) == 1:
            col = cols[0]
            line_rows = sorted(r for r, _ in line)
            if line_rows == list(range(line_rows[0], line_rows[-1] + 1)):
                rmin, rmax = line_rows[0], line_rows[-1]
                if head in ((rmin - 1, col - 1), (rmin - 1, col + 1)):
                    sig = ("up", length, rmin - 1, col, head)
                    if sig not in seen:
                        seen.add(sig)
                        candidates.append(
                            {
                                "direction": "up",
                                "length": length,
                                "front": (rmin - 1, col),
                                "head": head,
                            }
                        )
                if head in ((rmax + 1, col - 1), (rmax + 1, col + 1)):
                    sig = ("down", length, rmax + 1, col, head)
                    if sig not in seen:
                        seen.add(sig)
                        candidates.append(
                            {
                                "direction": "down",
                                "length": length,
                                "front": (rmax + 1, col),
                                "head": head,
                            }
                        )

    return candidates


def _directed_score(sprite_coords: list[tuple[int, int]], front: tuple[int, int], direction: str):
    fr, fc = front
    best = None
    for r, c in sprite_coords:
        if direction == "right":
            if c < fc:
                continue
            score = (c - fc, abs(r - fr))
        elif direction == "left":
            if c > fc:
                continue
            score = (fc - c, abs(r - fr))
        elif direction == "down":
            if r < fr:
                continue
            score = (r - fr, abs(c - fc))
        else:
            if r > fr:
                continue
            score = (fr - r, abs(c - fc))
        if best is None or score < best:
            best = score
    return best


def _extract_scene(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    sprite_mask = (g != bg) & (~np.isin(g, ARROW_COLORS))
    sprite_components = connected_components(sprite_mask, connectivity=4)
    assert sprite_components, "expected at least one sprite"

    sprite_colors = {int(g[r, c]) for comp in sprite_components for r, c in comp}
    assert len(sprite_colors) == 1, "expected one sprite color"

    sprites = []
    for comp in sprite_components:
        assert 4 <= len(comp) <= 8, "sprite size out of range"
        mask, anchor = mask_from_coords(comp)
        sprites.append(
            {
                "coords": sorted(comp),
                "mask": mask,
                "anchor": anchor,
                "bbox": bbox(comp),
            }
        )
    sprites.sort(key=lambda item: item["bbox"])

    arrow_mask = np.isin(g, ARROW_COLORS)
    arrow_components = connected_components(arrow_mask, connectivity=8)
    assert arrow_components, "expected arrows"

    arrows = []
    for comp in arrow_components:
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1, "arrow components must be monochrome"
        color = next(iter(colors))
        candidates = _parse_arrow_candidates(comp)
        assert candidates, "failed to parse arrow"
        arrows.append({"coords": sorted(comp), "color": color, "candidates": candidates})

    assert len(arrows) == len(sprites), "expected one arrow per sprite"
    return g, bg, sprites, arrows


def _best_sprite_for_candidate(candidate: dict, sprites: list[dict]):
    scored = []
    for idx, sprite in enumerate(sprites):
        primary = _directed_score(sprite["coords"], candidate["front"], candidate["direction"])
        if primary is None:
            continue
        r0, c0, _r1, _c1 = sprite["bbox"]
        scored.append((primary, r0, c0, idx))
    assert scored, "arrow does not point toward any sprite"
    scored.sort()
    if len(scored) > 1 and scored[1][0] == scored[0][0]:
        raise AssertionError("ambiguous nearest sprite")
    primary, _r0, _c0, idx = scored[0]
    return idx, primary


def _pair_arrows_to_sprites(grid: np.ndarray):
    g, bg, sprites, arrows = _extract_scene(grid)
    pairings = []
    for arrow in arrows:
        choices = []
        for candidate in arrow["candidates"]:
            try:
                sprite_idx, primary = _best_sprite_for_candidate(candidate, sprites)
            except AssertionError:
                continue
            r0, c0, _r1, _c1 = sprites[sprite_idx]["bbox"]
            choices.append((primary, r0, c0, sprite_idx, candidate))
        assert choices, "no valid arrow interpretation"
        choices.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        if len(choices) > 1 and choices[1][0] == choices[0][0]:
            raise AssertionError("ambiguous arrow direction")
        primary, _r0, _c0, sprite_idx, candidate = choices[0]
        pairings.append(
            {
                "arrow_color": arrow["color"],
                "length": candidate["length"],
                "sprite_idx": sprite_idx,
                "direction": candidate["direction"],
                "score": primary,
            }
        )

    targets = [item["sprite_idx"] for item in pairings]
    assert len(set(targets)) == len(targets), "multiple arrows chose the same sprite"
    return bg, sprites, pairings


def main(grid: np.ndarray) -> np.ndarray:
    """
    Each gray sprite is controlled by a nearby colored arrow.

    Arrow color selects the transformation:
    - red: clockwise rotation
    - green: mirror
    - yellow: scale

    Arrow length selects the parameter:
    - red 1/2/3 -> 90/180/270 degrees
    - green odd/even -> horizontal/vertical flip
    - yellow 1/2/3 -> x2/x3/x4 scale
    """
    g = np.asarray(grid, dtype=int)
    bg, sprites, pairings = _pair_arrows_to_sprites(g)
    out = np.full(g.shape, bg, dtype=int)

    for pairing in pairings:
        sprite = sprites[pairing["sprite_idx"]]
        transformed = transform_mask(sprite["mask"], pairing["arrow_color"], pairing["length"])
        r0, c0 = sprite["anchor"]
        h, w = transformed.shape
        assert r0 + h <= out.shape[0] and c0 + w <= out.shape[1], "transformed sprite out of bounds"
        for dr, dc in np.argwhere(transformed):
            rr = r0 + int(dr)
            cc = c0 + int(dc)
            assert out[rr, cc] == bg, "transformed sprites must not overlap"
            out[rr, cc] = pairing["arrow_color"]

    return out


SHAPES = [
    np.array([[1, 0], [1, 0], [1, 1]], dtype=np.uint8),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=np.uint8),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8),
    np.array([[1, 1, 1], [1, 1, 0]], dtype=np.uint8),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=np.uint8),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=np.uint8),
    np.array([[1, 1, 1], [1, 0, 1], [1, 0, 0]], dtype=np.uint8),
    np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]], dtype=np.uint8),
    np.array([[1, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.uint8),
    np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8),
]

GEN_RETRIES = 400
SPEC_RETRIES = 100
PLACEMENT_RETRIES = 400
GROUP_PAD = 2


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _random_base_mask() -> np.ndarray:
    mask = SHAPES[int(RNG.integers(len(SHAPES)))].copy()
    mask = np.rot90(mask, int(RNG.integers(4)))
    if RNG.random() < 0.5:
        mask = np.fliplr(mask)
    return _trim_mask(mask)


def _build_arrow(mask: np.ndarray, direction: str, length: int, gap: int, diag: int) -> list[tuple[int, int]]:
    coords = np.argwhere(mask > 0)
    rows = sorted({int(r) for r, _c in coords})
    cols = sorted({int(c) for _r, c in coords})

    if direction in ("left", "right"):
        row = int(RNG.choice(rows))
        if direction == "right":
            start = -(length + gap + 1)
            line = [(row, start + i) for i in range(length)]
            head = (row + diag, start + length)
        else:
            start = mask.shape[1] + gap + 1
            line = [(row, start + i) for i in range(length)]
            head = (row + diag, start - 1)
        return line + [head]

    col = int(RNG.choice(cols))
    if direction == "down":
        start = -(length + gap + 1)
        line = [(start + i, col) for i in range(length)]
        head = (start + length, col + diag)
    else:
        start = mask.shape[0] + gap + 1
        line = [(start + i, col) for i in range(length)]
        head = (start - 1, col + diag)
    return line + [head]


def _shape_is_degenerate(mask: np.ndarray, color: int, length: int) -> bool:
    transformed = transform_mask(mask, color, length)
    return transformed.shape == mask.shape and np.array_equal(transformed, mask)


def _make_spec(color: int, length: int) -> dict:
    for _ in range(SPEC_RETRIES):
        mask = _random_base_mask()
        if _shape_is_degenerate(mask, color, length):
            continue
        direction = str(RNG.choice(["up", "down", "left", "right"]))
        gap = int(RNG.integers(1, 4))
        diag = int(RNG.choice([-1, 1]))
        arrow = _build_arrow(mask, direction, length, gap, diag)
        sprite_cells = [tuple(map(int, rc)) for rc in np.argwhere(mask > 0)]
        out_mask = transform_mask(mask, color, length)

        all_input = sprite_cells + arrow
        in_r0, in_c0, in_r1, in_c1 = bbox(all_input)
        out_r1 = out_mask.shape[0] - 1
        out_c1 = out_mask.shape[1] - 1
        rel_r0 = min(in_r0, 0)
        rel_c0 = min(in_c0, 0)
        rel_r1 = max(in_r1, out_r1)
        rel_c1 = max(in_c1, out_c1)

        return {
            "color": color,
            "length": length,
            "mask": mask,
            "sprite_cells": sprite_cells,
            "arrow_cells": arrow,
            "out_mask": out_mask,
            "reserve_bbox": (rel_r0, rel_c0, rel_r1, rel_c1),
        }
    raise RuntimeError("failed to build non-degenerate sprite spec")


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _place_specs(specs: list[dict], shape: tuple[int, int]) -> list[dict] | None:
    h, w = shape
    placed: list[dict] = []
    reserved: list[tuple[int, int, int, int]] = []

    order = sorted(
        specs,
        key=lambda item: (
            -(item["reserve_bbox"][2] - item["reserve_bbox"][0] + 1)
            * (item["reserve_bbox"][3] - item["reserve_bbox"][1] + 1)
        ),
    )

    for spec in order:
        rr0, rc0, rr1, rc1 = spec["reserve_bbox"]
        row_min = -rr0
        row_max = h - 1 - rr1
        col_min = -rc0
        col_max = w - 1 - rc1
        if row_min > row_max or col_min > col_max:
            return None

        success = False
        for _ in range(PLACEMENT_RETRIES):
            ar = int(RNG.integers(row_min, row_max + 1))
            ac = int(RNG.integers(col_min, col_max + 1))
            reserve = (ar + rr0 - GROUP_PAD, ac + rc0 - GROUP_PAD, ar + rr1 + GROUP_PAD, ac + rc1 + GROUP_PAD)
            if any(_boxes_overlap(reserve, other) for other in reserved):
                continue
            placed.append({**spec, "anchor": (ar, ac)})
            reserved.append(reserve)
            success = True
            break
        if not success:
            return None

    return placed


def _render_expected(placed_specs: list[dict], shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = shape
    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)

    for spec in placed_specs:
        ar, ac = spec["anchor"]
        for dr, dc in spec["sprite_cells"]:
            rr = ar + dr
            cc = ac + dc
            if inp[rr, cc] != BG:
                return None
            inp[rr, cc] = SPRITE_COLOR

        for dr, dc in spec["arrow_cells"]:
            rr = ar + dr
            cc = ac + dc
            if not (0 <= rr < h and 0 <= cc < w):
                return None
            if inp[rr, cc] != BG:
                return None
            inp[rr, cc] = spec["color"]

        transformed = spec["out_mask"]
        for dr, dc in np.argwhere(transformed):
            rr = ar + int(dr)
            cc = ac + int(dc)
            if not (0 <= rr < h and 0 <= cc < w):
                return None
            if out[rr, cc] != BG:
                return None
            out[rr, cc] = spec["color"]

    return inp, out


def _spec_pairs(n: int) -> list[tuple[int, int]]:
    base = list(zip(RNG.permutation(ARROW_COLORS), RNG.permutation([1, 2, 3])))
    extra = [(int(RNG.choice(ARROW_COLORS)), int(RNG.integers(1, 4))) for _ in range(n - 3)]
    pairs = base + extra
    RNG.shuffle(pairs)
    return pairs


def _gen_attempt() -> np.ndarray | None:
    n_objects = int(RNG.choice([3, 4]))
    pairs = _spec_pairs(n_objects)
    specs = [_make_spec(color, length) for color, length in pairs]

    min_side = 22 if n_objects == 3 else 26
    shape = (int(RNG.integers(min_side, 31)), int(RNG.integers(min_side, 31)))
    placed = _place_specs(specs, shape)
    if placed is None:
        return None

    rendered = _render_expected(placed, shape)
    if rendered is None:
        return None
    inp, expected = rendered

    try:
        actual = main(inp)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None

    colors_in = set(map(int, np.unique(inp)))
    assert {RED, GREEN, YELLOW} <= colors_in
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
